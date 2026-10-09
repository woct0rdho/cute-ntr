#!/usr/bin/env python3
"""same_source.py - compile one CuTeDSL kernel with both front ends.

Everything else in `tools/compare` compares *two different kernels* (ours vs
the official example) and has to argue that they are comparable.  This tool
removes that argument: it takes the official CuTeDSL example source, compiles
it once with the installed NVIDIA front end and once with this repository's
`cutlass_compat` layer (which mirrors the public API), and reports the two
PTX/static profiles side by side.  Same source, same ptxas, same nvdisasm -
so any difference is the code generator.

Usage::

  python tools/compare/same_source.py --example gemm
  python tools/compare/same_source.py --example fa --head-dim 64 \
      --seqlen 4096 --heads 16 --m-block 64 --n-block 64
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXAMPLES = {
    "fa": ("flash_attention_v2.py", "FlashAttentionForwardAmpere"),
    "gemm": ("tensorop_gemm.py", None),
}


def _official_examples_dir() -> Path:
    root = Path(r"C:\flash-attention\csrc\cutlass\examples\python\CuTeDSL")
    for rel in ("ampere", "cute/ampere"):
        if (root / rel / "flash_attention_v2.py").is_file():
            return root / rel
    raise SystemExit("official CuTeDSL Ampere examples not found")


def run_child(stack: str, args) -> dict:
    """Compile the example in a subprocess and return the report paths."""
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--stack", stack, "--example", args.example,
           "--head-dim", str(args.head_dim), "--seqlen", str(args.seqlen),
           "--heads", str(args.heads), "--batch", str(args.batch),
           "--m-block", str(args.m_block), "--n-block", str(args.n_block),
           "--threads", str(args.threads), "--iters", str(args.iters),
           "--child"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise SystemExit(f"[same_source] {stack} child failed")
    line = [l for l in proc.stdout.splitlines() if l.startswith("REPORT ")]
    return json.loads(line[-1][len("REPORT "):])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack", choices=["official", "self"], default=None)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--example", choices=sorted(EXAMPLES), default="gemm")
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--m-block", type=int, default=64)
    ap.add_argument("--n-block", type=int, default=64)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.child:
        reports = {s: run_child(s, args) for s in ("official", "self")}
        print(report_table(reports))
        out = Path(args.out) if args.out else (
            ROOT / "artifacts/stack_compare" /
            f"same_source_{args.example}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(reports, indent=1), encoding="utf-8")
        print(f"[same_source] wrote {out}")
        return 0

    return child(args)


def report_table(reports: dict) -> str:
    lines = []
    o, s = reports.get("official", {}), reports.get("self", {})
    lines.append(f"{'metric':<28}{'official':>14}{'self':>14}{'self/off':>10}")
    lines.append("-" * 66)
    rows = [("compile_s", "compile time (s)"),
            ("median_us", "median (us)"),
            ("registers", "registers"),
            ("spill_stores", "spill stores"),
            ("spill_loads", "spill loads"),
            ("shared_bytes", "static smem (B)"),
            ("dynamic_smem_bytes", "dynamic smem (B)")]
    for key, label in rows:
        a = (o.get("resources") or {}).get(key, o.get(key))
        b = (s.get("resources") or {}).get(key, s.get(key))
        r = f"{b / a:.3f}" if isinstance(a, (int, float)) and a and \
            isinstance(b, (int, float)) else "-"
        lines.append(f"{label:<28}{fmt(a):>14}{fmt(b):>14}{r:>10}")
    for key in ("ptx_histogram", "sass_histogram"):
        ha, hb = o.get(key) or {}, s.get(key) or {}
        keys = sorted(set(ha) | set(hb), key=lambda k: -(ha.get(k, 0) + hb.get(k, 0)))
        if not keys:
            continue
        lines.append(f"-- {key} --")
        for k in keys[:16]:
            a, b = ha.get(k, 0), hb.get(k, 0)
            r = f"{b / a:.3f}" if a else "-"
            lines.append(f"  {k:<26}{a:>14}{b:>14}{r:>10}")
    return "\n".join(lines)


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def child(args) -> int:
    """Compile the selected example with one stack; print one REPORT line."""
    sys.path.insert(0, str(ROOT / "python"))
    if args.stack == "self":
        sys.path[:0] = [str(ROOT / "python/cutlass_compat")]

    import torch

    from self_cutedsl.compiler.artifacts import (
        ArtifactOptions, emit_artifacts)

    dump_dir = ROOT / "artifacts/stack_compare" / (
        f"same_source_{args.example}_{args.stack}")
    dump_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(dump_dir.rglob("*.ptx")) + list(dump_dir.rglob("*.cubin")):
        stale.unlink()

    import importlib.util

    path = _official_examples_dir() / EXAMPLES[args.example][0]
    spec = importlib.util.spec_from_file_location(f"_example_{args.example}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    import cutlass
    import cutlass.cute as cute

    if args.stack == "official":
        import cutlass.testing as testing
        opts = (f"--keep-ptx --keep-cubin --generate-line-info "
                f"--dump-dir {dump_dir}")
    else:
        opts = (f"--gpu-arch sm86 --keep-ptx --keep-cubin "
                f"--generate-line-info --dump-dir {dump_dir}")

    torch.manual_seed(0)
    import time

    if args.example == "fa":
        q = torch.randn(args.batch, args.seqlen, args.heads, args.head_dim,
                        device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        o = torch.empty_like(q)
        from cutlass.cute.runtime import from_dlpack
        fwd = mod.FlashAttentionForwardAmpere(
            args.head_dim, args.m_block, args.n_block, args.threads, False)
        t0 = time.perf_counter()
        compiled = cute.compile(fwd, from_dlpack(q), from_dlpack(k),
                                from_dlpack(v), from_dlpack(o), 1.0,
                                options=opts)
        compiled(from_dlpack(q), from_dlpack(k), from_dlpack(v),
                 from_dlpack(o), 1.0)
        torch.cuda.synchronize()
    else:
        m = n = kk = 2048
        from cutlass.cute.runtime import from_dlpack

        def mk(l, mode0, mode1, mode0_major):
            shape = (l, mode1, mode0) if mode0_major else (l, mode0, mode1)
            order = (2, 1, 0) if mode0_major else (1, 2, 0)
            t = (torch.empty(*shape, dtype=torch.int32).random_(-2, 2)
                 .to(torch.float16).permute(order).cuda())
            cute_t = from_dlpack(t, assumed_align=16)
            if hasattr(cute_t, "mark_layout_dynamic"):
                lead = 0 if mode0_major else 1
                cute_t = cute_t.mark_layout_dynamic(leading_dim=lead)
                cute_t = cute_t.mark_compact_shape_dynamic(
                    mode=lead,
                    stride_order=(2, 1, 0) if mode0_major else (2, 0, 1),
                    divisibility=8)
            return cute_t, t

        mma, a = mk(1, m, kk, True)
        mkb, b = mk(1, n, kk, True)
        mkc, c = mk(1, m, n, True)
        gemm = mod.TensorOpGemm(cutlass.Float16, cutlass.Float16,
                                cutlass.Float32, (2, 2, 1))
        t0 = time.perf_counter()
        compiled = cute.compile(gemm, mma, mkb, mkc, options=opts)
        compiled(mma, mkb, mkc)
        torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0

    times = []
    for _ in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if args.example == "fa":
            compiled(from_dlpack(q), from_dlpack(k), from_dlpack(v),
                     from_dlpack(o), 1.0)
        else:
            compiled(mma, mkb, mkc)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()

    report = {"stack": args.stack, "compile_s": compile_s,
              "median_us": times[len(times) // 2],
              "ptx_files": [p.as_posix() for p in dump_dir.rglob("*.ptx")]}
    ptx = sorted(p for p in dump_dir.rglob("*.ptx") if "ptxas" not in p.parts)
    if ptx:
        static = emit_artifacts(
            ptx[0].read_text(errors="replace"), "sm_86",
            ArtifactOptions(keep_ptx=True, keep_cubin=True, keep_sass=False,
                            generate_line_info=True,
                            dump_dir=str(dump_dir / "ptxas")),
            stem=ptx[0].stem)
        report.update({k: static.get(k) for k in
                       ("resources", "ptx_histogram", "sass_histogram")})
        report["dump_dir"] = dump_dir.as_posix()
    print("REPORT " + json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
