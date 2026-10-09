#!/usr/bin/env python3
"""fa4_dump.py - dump the FA4 (flash-attn-4) SM80 kernel's PTX for analysis.

FA4 compiles through the installed (closed) CuTeDSL front end, so this script
redirects its compile options to emit PTX, then hands the file to the same
`emit_artifacts` post-processing the open stack uses.  That makes FA4's
resource usage (registers, spills, shared memory) and instruction mix directly
comparable with `dump_kernels.py --stack self`.

Usage::

  python tools/compare/fa4_dump.py --head-dim 128 --m-block 128 --n-block 64
  python tools/compare/fa4_dump.py --head-dim 64 --m-block 64 --n-block 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from self_cutedsl.compiler.artifacts import (  # noqa: E402
    ArtifactOptions, emit_artifacts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--m-block", type=int, default=128)
    ap.add_argument("--n-block", type=int, default=64)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--dump-root",
                    default=str(ROOT / "artifacts/stack_compare"))
    args = ap.parse_args()

    import torch  # noqa: E402

    dump_dir = Path(args.dump_root) / (
        f"fa4_fa_s{args.seqlen}_h{args.heads}_d{args.head_dim}"
        f"_m{args.m_block}_n{args.n_block}")
    dump_dir.mkdir(parents=True, exist_ok=True)
    for stale in dump_dir.rglob("*.ptx"):
        stale.unlink()

    # FA4 compiles through the installed front end, which honours the
    # CUTE_DSL_KEEP / CUTE_DSL_DUMP_DIR environment variables rather than the
    # compile-option string our own harness passes.
    import os  # noqa: E402

    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(dump_dir)
    os.environ["CUTE_DSL_NO_CACHE"] = "1"

    import flash_attn.cute.interface as iface  # noqa: E402

    torch.manual_seed(0)
    q = torch.randn(args.batch, args.seqlen, args.heads, args.head_dim,
                    device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    # tile_mn is not part of the public signature; the SM80 heuristic picks
    # 128x64, which is the configuration the measurements use anyway.
    iface.flash_attn_func(q, k, v, causal=False)
    torch.cuda.synchronize()

    ptx_files = sorted(p for p in dump_dir.rglob("*.ptx") if "ptxas" not in p.parts)
    if not ptx_files:
        print(f"no PTX dumped under {dump_dir}")
        return 1
    report = emit_artifacts(
        ptx_files[0].read_text(errors="replace"), "sm_86",
        ArtifactOptions(keep_ptx=True, keep_cubin=True, keep_sass=True,
                        generate_line_info=True,
                        dump_dir=str(dump_dir / "ptxas")),
        stem=ptx_files[0].stem)
    print(json.dumps({
        "ptx": ptx_files[0].as_posix(),
        "resources": report.get("resources"),
        "ptx_histogram": {k: v for k, v in (report.get("ptx_histogram") or {}).items()
                          if v and not k.startswith("__")},
        "sass_histogram": {k: v for k, v in (report.get("sass_histogram") or {}).items()
                           if v and not k.startswith("__")},
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
