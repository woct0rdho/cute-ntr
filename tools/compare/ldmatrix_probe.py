"""ldmatrix_probe.py - determine the effective register mapping of a
non-transposed `ldmatrix` load from an (N, K) shared tile.

The attention kernel needs the `m16n8k16` B fragment out of a K tile stored
N-major.  Rather than guessing how the four registers of an `.x4` load line up
with the 2x2 block arrangement, this probe stages a known pattern, performs the
load, and dumps the raw per-lane registers.  The host then decodes them, so the
addressing can be verified (or corrected) from evidence.

Run::

    python tools/compare/ldmatrix_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "python/cutlass_compat")]

import torch  # noqa: E402

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402
from self_cutedsl.frontend import builtins as B  # noqa: E402
from cutlass.cute._fi_copy import _gep_gmem as _gmem_ptr  # noqa: E402

N, K = 64, 64
SWIZZLE = cute.make_swizzle(3, 4, 3)


@cute.kernel
def probe_kernel(mK: cute.Tensor, mOut: cute.Tensor,
                 sK_layout: cute.Layout, copy_K: cute.TiledCopy,
                 copy_s2r_K: cute.TiledCopy, mma_QK: cute.TiledMma):
    tidx, _, _ = cute.arch.thread_idx()
    sK_mem = B.make_smem_tile("probe_k", N * K, cutlass.Float16, align=16)
    sK = sK_mem.get_tensor(sK_layout, swizzle=SWIZZLE)
    gK = cute.local_tile(mK, (N, K), (0, 0))
    k_copy = copy_K.get_slice(tidx)
    cute.copy(copy_K, k_copy.partition_S(gK),
              k_copy.partition_D(sK)[None, None, None, 0])
    cute.cp_async_commit_group()
    cute.cp_async_wait_group(0)
    cute.sync_threads()

    qk = mma_QK.get_slice(tidx)
    tK_s = qk.partition_B(sK)
    tK_r = mma_QK.make_fragment_B(tK_s[None, None, None, 0])
    k_copy_r = copy_s2r_K.get_slice(tidx)
    k_s = k_copy_r.partition_S(sK)
    k_r = k_copy_r.retile(tK_r)
    for k_step in range(K // 16):
        cute.copy(copy_s2r_K,
                  k_s[None, None, None, 0][None, None, k_step],
                  k_r[None, None, k_step])

    # dump every register of the first k-block, unpacked to two f16 values
    e = B._emitter()
    gOut = cute.local_tile(mOut, (128, 8), (0, 0))
    regs = tK_r.slots[0]
    for j in range(len(regs)):
        reg = regs[j]
        for half in range(2):
            shifted = reg
            if half:
                sh = e.ssa("i32", "arith.constant 16 : i32")
                shifted = e.ssa(
                    "i32", f"arith.shrui {reg.name}, {sh.name} : i32")
            val16 = e.ssa("i16", f"arith.trunci {shifted.name} : i32 to i16")
            val = e.ssa("f16", f"llvm.bitcast {val16.name} : i16 to f16")
            total = e.idx_binop(
                "arith.addi",
                e.idx_binop("arith.muli", tidx, e.idx_const(8)),
                e.idx_const(j * 2 + half))
            p = _gmem_ptr(e, gOut, total)
            e.raw(f"llvm.store {val.name}, {p.name} : f16, !llvm.ptr<1>")


@cute.jit
def probe(k: cute.Tensor, out: cute.Tensor, sK_layout: cute.Layout,
          copy_K: cute.TiledCopy, copy_s2r_K: cute.TiledCopy,
          mma_QK: cute.TiledMma):
    probe_kernel(k, out, sK_layout, copy_K, copy_s2r_K, mma_QK).launch(
        grid=[1, 1, 1], block=[128, 1, 1])


def main() -> int:
    native = "--x2" not in sys.argv
    torch.manual_seed(0)
    n_idx = torch.arange(N, device="cuda", dtype=torch.float32)[:, None]
    k_idx = torch.arange(K, device="cuda", dtype=torch.float32)[None, :]
    pattern = (n_idx + k_idx / 32.0).to(torch.float16).contiguous()
    out = torch.zeros(128, 8, device="cuda", dtype=torch.float16)

    k_cute = from_dlpack(pattern)
    o_cute = from_dlpack(out)

    atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(), cutlass.Float16, num_bits_per_copy=128)
    thr = cute.make_layout((128 // 8, 8), stride=(8, 1))
    val = cute.make_layout((1, 8))
    copy_K = cute.make_tiled_copy_tv(atom, thr, val)
    copy_K.operand = "B"
    copy_K.transpose_src = False
    copy_K.threads = 128

    mma_op = cute.nvgpu.warp.MmaF16BF16Op(
        cutlass.Float16, cutlass.Float32, (16, 8, 16))
    mma_QK = cute.make_tiled_mma(
        mma_op, cute.make_layout((4, 1, 1)), permutation_mnk=(16, 64, 16))
    copy_s2r_K = cute.make_tiled_copy_B(
        cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.Float16),
        mma_QK)

    sK_layout = cute.make_layout((N, K), stride=(K, 1))
    compiled = cute.compile(probe, k_cute, o_cute, sK_layout, copy_K,
                            copy_s2r_K, mma_QK, options="--gpu-arch sm86")
    compiled()
    torch.cuda.synchronize()

    got = out.cpu().float()
    print("raw out[0:4, 0:8]:")
    print(got[0:4])
    # decode: value = n + k/32
    n_dec = torch.floor(got)
    k_dec = torch.round((got - n_dec) * 32)
    print("lane reg-value-pairs (n, k) as seen by the MMA B operand:")
    for lane in range(32):
        desc = []
        for j in range(4):
            n0 = int(n_dec[lane, 2 * j].item())
            k0 = int(k_dec[lane, 2 * j].item())
            n1 = int(n_dec[lane, 2 * j + 1].item())
            k1 = int(k_dec[lane, 2 * j + 1].item())
            desc.append(f"r{j}=({n0},{k0})/({n1},{k1})")
        print(f"  lane {lane:2d}: " + "  ".join(desc))
    print()
    print("addressing under test: lanes 0-15 -> N rows 0-15 at K[0:8],")
    print("                      lanes 16-31 -> the same rows at K[8:16]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
