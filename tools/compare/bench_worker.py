#!/usr/bin/env python3
"""bench_worker.py - resident measurement worker for one CuTeDSL stack.

The laptop GPU throttles, so comparing two stacks by running two long blocks
back-to-back confounds the kernel difference with thermal drift.  Instead,
`paired_bench.py` starts one resident worker per stack and alternates short
measurement batches between them, so any drift is common-mode.

Protocol (line-oriented on stdin/stdout, everything else goes to stderr):

  `ready`            printed once after compile + warmup
  `run <n>`          run n iterations, print one `times` line
  `quit`             exit

Each `times` line is a JSON list of per-iteration GPU times in microseconds,
measured with CUDA events recorded around each individual launch while the
launches stay enqueued (no host sync between iterations), which keeps the
measurement granular but avoids measuring host overhead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/compare"))

import dump_kernels as DK  # noqa: E402


def _measure(launch, iterations: int) -> list[float]:
    import torch

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for i in range(iterations):
        starts[i].record()
        launch()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack", choices=["official", "self", "fa2", "fa3", "fa4"],
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
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--dump-root",
                    default=str(ROOT / "artifacts/stack_compare"))
    args = ap.parse_args()

    if args.kernel == "fa":
        tag = (f"{args.stack}_fa_s{args.seqlen}_h{args.heads}"
               f"_d{args.head_dim}_m{args.m_block}_n{args.n_block}")
    else:
        tag = f"{args.stack}_gemm_{args.m}x{args.n}x{args.k}"
    dump_dir = Path(args.dump_root) / tag
    dump_dir.mkdir(parents=True, exist_ok=True)

    prepare = DK.PREPARE[(args.stack, args.kernel)]
    compiled, call_args, meta = prepare(args, dump_dir)

    def launch():
        compiled(*call_args)

    for _ in range(args.warmup):
        launch()
    import torch

    torch.cuda.synchronize()

    info = {"tag": tag, "compile_s": meta.get("compile_s"),
            "options": meta.get("options")}
    print("ready " + json.dumps(info), flush=True)

    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "quit":
            break
        if parts[0] == "run":
            n = int(parts[1]) if len(parts) > 1 else 1
            times = _measure(launch, n)
            print("times " + json.dumps(times), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
