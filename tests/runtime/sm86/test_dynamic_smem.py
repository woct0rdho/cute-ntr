"""Dynamic shared memory above the 48 KiB static ceiling.

`sm_86` allows only 48 KiB of *static* shared memory per CTA.  A kernel that
needs more (a 64-wide K/V tile at head_dim 128, a multi-stage pipeline) must
allocate from the dynamic arena and opt in at launch.  That path is two
pieces of backend code:

  * the emitter, which declares `@__dynamic_shmem__0` as an external
    address-space-3 array and bump-allocates each tile at a byte offset -
    exactly the declaration the official CuTeDSL emits for its D128 kernel;
  * the launcher, which passes the arena size to `cuLaunchKernel` and calls
    `cuFuncSetAttribute` when it exceeds 48 KiB.

The kernel below deliberately sizes two tiles so that their union is exactly
48 KiB: the arena must span both, and clobbering offsets would corrupt the
second tile with the first.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))

import torch  # noqa: E402

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

ELEMS = 12 * 1024        # 24 KiB per tile, 48 KiB across the two


def _require_sm86():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("requires an sm86 device")


@cute.kernel
def _arena_kernel(src: cute.Tensor, dst: cute.Tensor, threads: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    a_mem = cute.make_smem_tile("t_a", ELEMS, cutlass.Float16, align=16,
                                dynamic=True)
    b_mem = cute.make_smem_tile("t_b", ELEMS, cutlass.Float16, align=16,
                                dynamic=True)
    a = a_mem.get_tensor(cute.make_layout(ELEMS))
    b = b_mem.get_tensor(cute.make_layout(ELEMS))
    for i in range(ELEMS // threads):
        a[threads * i + tidx] = src[threads * i + tidx]
    for i in range(ELEMS // threads):
        b[threads * i + tidx] = src[ELEMS + threads * i + tidx]
    cute.sync_threads()
    for i in range(ELEMS // threads):
        dst[threads * i + tidx] = a[threads * i + tidx]
    for i in range(ELEMS // threads):
        dst[ELEMS + threads * i + tidx] = b[threads * i + tidx]


@cute.jit
def _arena(src: cute.Tensor, dst: cute.Tensor, threads: cutlass.Constexpr):
    _arena_kernel(src, dst, threads).launch(grid=[1, 1, 1], block=[threads, 1, 1])


@pytest.mark.sm86
def test_dynamic_shared_memory_above_static_limit():
    _require_sm86()
    torch.manual_seed(0)
    src = torch.randn(2 * ELEMS, device="cuda", dtype=torch.float16)
    dst = torch.zeros_like(src)
    compiled = cute.compile(_arena, from_dlpack(src), from_dlpack(dst), 256,
                            options="--gpu-arch sm86")
    compiled()
    torch.cuda.synchronize()
    assert torch.equal(dst, src), "arena tiles overlapped or were not written"
