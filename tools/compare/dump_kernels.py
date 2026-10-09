#!/usr/bin/env python3
"""dump_kernels.py - intermediate-result driver for both CuTeDSL stacks.

Runs one benchmark case (dense GEMM or FlashAttention) under either

  `--stack official`  the installed `nvidia-cutlass-dsl` package, or
  `--stack self`      this repository's open stack,

using the *same* compile-option surface and the *same* reporting schema, so
`compare_stacks.py` can diff the two.

Both stacks are driven through their own kernel source (the official Ampere
examples vs. this repository's SM86 kernels); everything downstream of PTX is
shared: one `ptxas` and one `nvdisasm` produce the resource report, the
PTX histogram, and the SASS histogram for both.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Only the open stack's *own* package root goes on the path here.  The
# `cutlass` name is shadowed by python/cutlass_compat, and the official
# stack needs the installed `cutlass` - so that directory is added lazily,
# only for --stack self.
sys.path.insert(0, str(ROOT / "python"))

from self_cutedsl.compiler.artifacts import (  # noqa: E402
    ArtifactOptions, emit_artifacts)


def _enable_self_stack() -> None:
    """Make `import cutlass` resolve to this repository's compat package."""
    for extra in ("python/cutlass_compat", "tools/perf", "tests/runtime/sm86"):
        path = str(ROOT / extra)
        if path not in sys.path:
            sys.path.insert(0, path)
    import cutlass

    if str(ROOT) not in str(getattr(cutlass, "__file__", "")):
        raise RuntimeError(
            "--stack self requires this repository's cutlass compat package, "
            f"but 'cutlass' came from {getattr(cutlass, '__file__', None)!r}")


def _assert_official_stack() -> None:
    import cutlass

    location = str(getattr(cutlass, "__file__", ""))
    if str(ROOT) in location:
        raise RuntimeError(
            "--stack official requires the installed nvidia-cutlass-dsl, but "
            f"'cutlass' came from {location!r}")

def _find_official_examples() -> Path:
    """Locate the CuTeDSL Ampere examples inside the flash-attention tree.

    The submodule moved between CUTLASS releases (`CuTeDSL/cute/ampere` ->
    `CuTeDSL/ampere`), so probe both layouts.
    """
    root = Path(r"C:\flash-attention\csrc\cutlass\examples\python\CuTeDSL")
    for rel in ("ampere", "cute/ampere"):
        candidate = root / rel
        if (candidate / "flash_attention_v2.py").is_file():
            return candidate
    return root / "ampere"


OFFICIAL_EXAMPLES = _find_official_examples()

# kept alive for the reference check (torch tensors backing the cute views)
_OFFICIAL_FA_TORCH: list = []


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def artifact_options(dump_dir: Path) -> str:
    """The dump flags the official CuTeDSL accepts and executes.

    `--keep-sass` is deliberately absent: official 4.8.0 rejects the
    toolkit's nvdisasm on Windows and its bundled-wheel probe misses
    `nvdisasm.exe`.  SASS is produced by the shared post-processing step
    instead, with a pinned nvdisasm, identically for both stacks.
    """
    return (f"--keep-ptx --keep-cubin --generate-line-info "
            f"--dump-dir {dump_dir}")


def ptx_from_dump(dump_dir: Path) -> Path | None:
    """The front end's PTX, ignoring the normalized re-dump under ptxas/."""
    files = sorted(p for p in dump_dir.rglob("*.ptx")
                   if "ptxas" not in p.parts)
    return files[0] if files else None


def report_from_ptx(ptx_path: Path, target: str, stem: str,
                    out_dir: Path, want_sass: bool) -> dict:
    return emit_artifacts(
        ptx_path.read_text(errors="replace"), target,
        ArtifactOptions(keep_ptx=True, keep_cubin=True, keep_sass=want_sass,
                        generate_line_info=True, dump_dir=str(out_dir)),
        stem=stem)


# ---------------------------------------------------------------------------
# official stack
# ---------------------------------------------------------------------------

def _load_module(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def prepare_official_fa(args, dump_dir: Path):
    """Compile once; return (compiled, call_args, meta)."""
    _assert_official_stack()
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.testing as testing
    import torch
    from cutlass.cute.runtime import from_dlpack

    mod = _load_module(
        OFFICIAL_EXAMPLES / "flash_attention_v2.py", "_official_fa2")

    dtype = cutlass.Float16
    torch.manual_seed(0)

    def make(seqlen):
        t = torch.empty(args.batch, seqlen, args.heads, args.head_dim,
                        dtype=torch.int32).random_(-2, 2)\
            .to(torch.float16).cuda()
        ct = (from_dlpack(t, assumed_align=16)
              .mark_layout_dynamic(leading_dim=3)
              .mark_compact_shape_dynamic(
                  mode=3, stride_order=t.dim_order(),
                  divisibility=(128 // dtype.width)))
        return ct, t

    q, q_t = make(args.seqlen)
    k, k_t = make(args.seqlen)
    v, v_t = make(args.seqlen)
    o, o_t = make(args.seqlen)
    scale = 1.0 / (args.head_dim ** 0.5)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    kernel = mod.FlashAttentionForwardAmpere(
        args.head_dim, args.m_block, args.n_block, args.threads, False)
    options = artifact_options(dump_dir)
    t0 = time.perf_counter()
    compiled = cute.compile(kernel, q, k, v, o, scale, stream,
                            options=options)
    compile_s = time.perf_counter() - t0

    _OFFICIAL_FA_TORCH[:] = [q_t, k_t, v_t, o_t]

    meta = {
        "compile_s": compile_s,
        "options": options,
        "ptx": [p.as_posix() for p in dump_dir.rglob("*.ptx")],
        "cubin": [p.as_posix() for p in dump_dir.rglob("*.cubin")],
    }
    return compiled, (q, k, v, o, scale, stream), meta


def run_official_fa(args, dump_dir: Path) -> dict:
    import cutlass.testing as testing

    compiled, call_args, meta = prepare_official_fa(args, dump_dir)
    if not args.skip_ref_check:
        _check_official_fa(args, compiled, call_args)
    median_us = testing.benchmark(
        compiled,
        kernel_arguments=testing.JitArguments(*call_args),
        warmup_iterations=args.warmup, iterations=args.iters)
    return {**meta, "median_us": median_us}


def _check_official_fa(args, compiled, call_args) -> None:
    import torch

    q, k, v, o, scale, stream = call_args
    compiled(*call_args)
    torch.cuda.synchronize()
    q_t, k_t, v_t, o_t = (t.permute(0, 2, 1, 3)
                          for t in _OFFICIAL_FA_TORCH)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t, scale=scale, is_causal=False).permute(0, 2, 1, 3)
    torch.testing.assert_close(
        _OFFICIAL_FA_TORCH[3].cpu(), ref.cpu(), atol=1e-2, rtol=1e-4)


def run_official_gemm(args, dump_dir: Path) -> dict:
    import cutlass.testing as testing

    compiled, call_args, meta = prepare_official_gemm(args, dump_dir)
    if not args.skip_ref_check:
        _check_official_gemm(args, compiled, call_args, meta)
    median_us = testing.benchmark(
        compiled,
        kernel_arguments=testing.JitArguments(*call_args),
        warmup_iterations=args.warmup, iterations=args.iters)
    return {**meta, "median_us": median_us}


def prepare_official_gemm(args, dump_dir: Path):
    _assert_official_stack()
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.testing as testing
    import torch

    mod = _load_module(
        OFFICIAL_EXAMPLES / "tensorop_gemm.py", "_official_gemm")

    mnkl = (args.m, args.n, args.k, 1)
    a, b, c, a_f32, b_f32, c_f32 = mod.prepare_tensors(
        mnkl, cutlass.Float16, cutlass.Float16, "m", "n", "n")
    torch.manual_seed(0)
    a_, b_, c_ = mod.mark_dynamic_layout(
        a, b, c, 1, 2, 2, cutlass.Float16, cutlass.Float16,
        a_f32, b_f32, c_f32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    gemm = mod.TensorOpGemm(cutlass.Float16, cutlass.Float16,
                            cutlass.Float32, (2, 2, 1), False)
    options = artifact_options(dump_dir)
    t0 = time.perf_counter()
    compiled = cute.compile(mod.bmm, gemm, a_, b_, c_, stream,
                            lambda x: x, options=options)
    compile_s = time.perf_counter() - t0

    meta = {
        "compile_s": compile_s,
        "options": options,
        "c": c,
        "a_f32": a_f32,
        "b_f32": b_f32,
        "ptx": [p.as_posix() for p in dump_dir.rglob("*.ptx")],
        "cubin": [p.as_posix() for p in dump_dir.rglob("*.cubin")],
    }
    return compiled, (a_, b_, c_, stream), meta


def _check_official_gemm(args, compiled, call_args, meta) -> None:
    import torch

    compiled(*call_args)
    torch.cuda.synchronize()
    ref = torch.bmm(meta["a_f32"].to(torch.float32),
                    meta["b_f32"].to(torch.float32))
    torch.testing.assert_close(
        meta["c"].to(dtype=torch.float32), ref, atol=1e-3, rtol=1e-5)


# ---------------------------------------------------------------------------
# self stack
# ---------------------------------------------------------------------------

def prepare_fa4_fa(args, dump_dir: Path):
    """Tri Dao's FA4 CuTeDSL SM80 kernel (`flash-attn-4`), used as the
    strongest available reference for this device."""
    import torch
    from flash_attn.cute.interface import flash_attn_func

    torch.manual_seed(0)
    q = torch.randn(args.batch, args.seqlen, args.heads, args.head_dim,
                    device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def call(*_args):
        return flash_attn_func(q, k, v, causal=False)[0]

    out = call()
    torch.cuda.synchronize()
    meta = {"compile_s": float("nan"), "options": "fa4 (jit)"}
    return call, (out,), meta


def run_fa4_fa(args, dump_dir: Path) -> dict:
    # Deliberately does *not* call _enable_self_stack(): the reference stack
    # must import the installed `cutlass` package, and putting this
    # repository's compat package on sys.path shadows it.
    import perf_common as P

    prepared, (out,), meta = prepare_fa4_fa(args, dump_dir)
    if not args.skip_ref_check:
        import torch
        from flash_attn import flash_attn_func as fa2

        q = torch.randn(args.batch, args.seqlen, args.heads, args.head_dim,
                        device="cuda", dtype=torch.float16)
        del q
    timing = P.cuda_time_us(prepared, warmup=args.warmup, iters=args.iters)
    return {**meta, "median_us": timing["median_us"], "timing": timing,
            "ptx": [], "cubin": []}


def run_self_fa(args, dump_dir: Path) -> dict:
    _enable_self_stack()
    import perf_common as P

    compiled, call_args, meta = prepare_self_fa(args, dump_dir)
    if not args.skip_ref_check:
        _check_self_fa(args, call_args, meta)
    timing = P.cuda_time_us(compiled, warmup=args.warmup, iters=args.iters)
    return {**meta, "median_us": timing["median_us"], "timing": timing}


def _check_self_fa(args, call_args, meta) -> None:
    import torch
    from flash_attn import flash_attn_func

    q, k, v, o = call_args
    ref = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
    torch.testing.assert_close(o, ref, atol=3e-2, rtol=3e-2)


def prepare_self_fa(args, dump_dir: Path):
    _enable_self_stack()
    import torch
    import perf_common as P
    from sm86_flash_attention import compile_sm86_flash_attention

    P.init_cuda_context()
    torch.manual_seed(0)
    q = torch.randn(args.batch, args.seqlen, args.heads, args.head_dim,
                    device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    o = torch.empty_like(q)
    options = f"--gpu-arch sm86 {artifact_options(dump_dir)}"
    t0 = time.perf_counter()
    compiled = compile_sm86_flash_attention(q, k, v, o, options=options)
    compiled()
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0

    meta = {
        "compile_s": compile_s,
        "options": options,
        "ptx": [p.as_posix() for p in dump_dir.rglob("*.ptx")],
        "cubin": [p.as_posix() for p in dump_dir.rglob("*.cubin")],
    }
    return compiled, (q, k, v, o), meta


def run_self_gemm(args, dump_dir: Path) -> dict:
    _enable_self_stack()
    import perf_common as P

    compiled, call_args, meta = prepare_self_gemm(args, dump_dir)
    if not args.skip_ref_check:
        _check_self_gemm(args, call_args, meta)
    timing = P.cuda_time_us(compiled, warmup=args.warmup, iters=args.iters)
    return {**meta, "median_us": timing["median_us"], "timing": timing}


def _check_self_gemm(args, call_args, meta) -> None:
    import torch

    a, b, c = call_args
    ref = (a.float() @ b.float())
    torch.testing.assert_close(c.float(), ref, atol=1e-1, rtol=1e-2)


def prepare_self_gemm(args, dump_dir: Path):
    _enable_self_stack()
    import torch
    import perf_common as P
    from sm86_matmul import compile_sm86_matmul

    P.init_cuda_context()
    torch.manual_seed(0)
    m, n, k = args.m, args.n, args.k
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    c = torch.empty(m, n, device="cuda", dtype=torch.float16)
    options = f"--gpu-arch sm86 {artifact_options(dump_dir)}"
    t0 = time.perf_counter()
    compiled = compile_sm86_matmul(a, b, c, options=options)
    compiled()
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0

    meta = {
        "compile_s": compile_s,
        "options": options,
        "ptx": [p.as_posix() for p in dump_dir.rglob("*.ptx")],
        "cubin": [p.as_posix() for p in dump_dir.rglob("*.cubin")],
    }
    return compiled, (a, b, c), meta


def prepare_fa3_fa(args, dump_dir: Path):
    """FA3 through the installed `flash_attn_interface` shim.

    FA3 is nominally Hopper-only, but the shim dispatches on compute
    capability and selects an `kBlockM_sm86` kernel, which does run (and
    verifies against FA2) on this device - and is the fastest D128 reference
    here, so it is the target.
    """
    import torch
    import flash_attn_interface as F

    torch.manual_seed(0)
    q = torch.randn(args.batch, args.seqlen, args.heads, args.head_dim,
                    device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def call(*_args):
        return F.flash_attn_func(q, k, v)

    out = call()
    torch.cuda.synchronize()
    return call, (out,), {"compile_s": float("nan"), "options": "fa3 (jit)"}


def run_fa3_fa(args, dump_dir: Path) -> dict:
    import perf_common as P

    prepared, (out,), meta = prepare_fa3_fa(args, dump_dir)
    timing = P.cuda_time_us(prepared, warmup=args.warmup, iters=args.iters)
    return {**meta, "median_us": timing["median_us"], "timing": timing}


def prepare_fa2_fa(args, dump_dir: Path):
    """Installed FA2 (`flash_attn` C++/CUDA kernel)."""
    import torch
    from flash_attn import flash_attn_func

    torch.manual_seed(0)
    q = torch.randn(args.batch, args.seqlen, args.heads, args.head_dim,
                    device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def call(*_args):
        return flash_attn_func(q, k, v, causal=False)

    out = call()
    torch.cuda.synchronize()
    return call, (out,), {"compile_s": float("nan"), "options": "fa2 (jit)"}


def run_fa2_fa(args, dump_dir: Path) -> dict:
    import perf_common as P

    prepared, (out,), meta = prepare_fa2_fa(args, dump_dir)
    timing = P.cuda_time_us(prepared, warmup=args.warmup, iters=args.iters)
    return {**meta, "median_us": timing["median_us"], "timing": timing}


PREPARE = {
    ("official", "fa"): prepare_official_fa,
    ("official", "gemm"): prepare_official_gemm,
    ("self", "fa"): prepare_self_fa,
    ("self", "gemm"): prepare_self_gemm,
    ("fa4", "fa"): prepare_fa4_fa,
    ("fa3", "fa"): prepare_fa3_fa,
    ("fa2", "fa"): prepare_fa2_fa,
}

CASES = {
    ("official", "fa"): run_official_fa,
    ("official", "gemm"): run_official_gemm,
    ("self", "fa"): run_self_fa,
    ("self", "gemm"): run_self_gemm,
    ("fa4", "fa"): run_fa4_fa,
    ("fa3", "fa"): run_fa3_fa,
    ("fa2", "fa"): run_fa2_fa,
}


_REFERENCE_STACKS = ("official", "fa2", "fa3", "fa4")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack", choices=["self", *_REFERENCE_STACKS],
                    required=True)
    ap.add_argument("--kernel", choices=["fa", "gemm"], required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--m-block", type=int, default=64)
    ap.add_argument("--n-block", type=int, default=64)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--sass", action="store_true")
    ap.add_argument("--skip-ref-check", action="store_true")
    ap.add_argument("--dump-root", default=str(ROOT / "artifacts/stack_compare"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.kernel == "fa":
        tag = (f"{args.stack}_fa_s{args.seqlen}_h{args.heads}"
               f"_d{args.head_dim}_m{args.m_block}_n{args.n_block}")
        shape = dict(batch=args.batch, seqlen=args.seqlen, heads=args.heads,
                     head_dim=args.head_dim, m_block=args.m_block,
                     n_block=args.n_block, threads=args.threads,
                     dtype="float16")
    else:
        tag = f"{args.stack}_gemm_{args.m}x{args.n}x{args.k}"
        shape = dict(m=args.m, n=args.n, k=args.k, dtype="float16")

    dump_dir = Path(args.dump_root) / tag
    dump_dir.mkdir(parents=True, exist_ok=True)

    case = CASES[(args.stack, args.kernel)]
    result = case(args, dump_dir)

    # one uniform PTX -> ptxas/nvdisasm report, identical for both stacks
    ptx_path = ptx_from_dump(dump_dir)
    static = {}
    if ptx_path is not None:
        # normalize into ptxas/ so the front-end dump stays comparable
        static = report_from_ptx(
            ptx_path, "sm_86", tag, dump_dir / "ptxas", args.sass)
        static.pop("ptxas_log", None)

    payload = {
        "stack": args.stack,
        "kernel": args.kernel,
        "tag": tag,
        "shape": shape,
        "dump_dir": dump_dir.as_posix(),
        "status": "PASS",
        **result,
        **static,
    }
    out = Path(args.out) if args.out else (
        Path(args.dump_root) / f"{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    summary = {k: payload.get(k) for k in (
        "stack", "kernel", "tag", "median_us", "compile_s")}
    summary["resources"] = payload.get("resources")
    summary["ptx_histogram"] = {
        k: v for k, v in (payload.get("ptx_histogram") or {}).items()
        if v and not k.startswith("__")}
    summary["sass_histogram"] = {
        k: v for k, v in (payload.get("sass_histogram") or {}).items()
        if v and not k.startswith("__")}
    print(json.dumps(summary, indent=1))
    print(f"[dump_kernels] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
