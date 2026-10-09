#!/usr/bin/env python3
"""paired_bench.py - throttle-robust A/B comparison of the two CuTeDSL stacks.

This laptop throttles, so a "run official for 30s, then run self for 30s"
comparison mostly measures the GPU's thermal state.  This tool instead:

  * starts one resident worker per stack (each compiled once);
  * alternates short measurement batches between the two workers, so drift is
    common-mode and largely cancels;
  * analyses timings in **log space**, because GPU timing noise is
    multiplicative: the score is the median log time, the scale is the
    robust estimate `max(stdev, 1.4826*MAD, IQR/1.349)`, and the standard
    error of the median is `1.2533 * sigma / sqrt(n)`;
  * reports a **confidence interval on the paired gap**, and only calls a
    difference real when the lower bound exceeds the indifference threshold
    (`--epsilon-pct`).  Otherwise it says the two are equivalent within
    epsilon - which is the honest answer for a throttling laptop.

Usage::

  python tools/compare/paired_bench.py --kernel fa --seqlen 4096 --heads 16 \
      --head-dim 64 --rounds 8 --batch 5
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tools/compare/bench_worker.py"
SQRT_PI_OVER_2 = 1.2533141373155001


# ---------------------------------------------------------------------------
# robust log-space statistics
# ---------------------------------------------------------------------------

def robust_stats(times_us: list[float]) -> dict:
    ys = [math.log(t) for t in times_us if t > 0]
    n = len(ys)
    if n == 0:
        return {"n": 0}
    med = statistics.median(ys)
    if n > 1:
        stdev = statistics.stdev(ys)
        mad = statistics.median([abs(y - med) for y in ys]) * 1.4826
        q = statistics.quantiles(ys, n=4, method="inclusive")
        iqr = (q[2] - q[0]) / 1.349
    else:
        stdev = mad = iqr = 0.0
    sigma = max(stdev, mad, iqr, 1e-9)
    se = SQRT_PI_OVER_2 * sigma / math.sqrt(n)
    return {
        "n": n,
        "median_us": round(math.exp(med), 3),
        "median_log": med,
        "geomean_us": round(math.exp(statistics.fmean(ys)), 3),
        "sigma_log": sigma,
        "se_log": se,
        "p10_us": round(math.exp(min(ys)), 3),
        "p90_us": round(math.exp(max(ys)), 3),
        "min_us": round(min(times_us), 3),
        "values_us": [round(t, 3) for t in times_us],
    }


def decide(a: dict, b: dict, z: float, epsilon_pct: float) -> dict:
    """Compare stack `b` against stack `a` (gap in log space)."""
    if a.get("n", 0) == 0 or b.get("n", 0) == 0:
        return {"status": "no_samples"}
    gap = b["median_log"] - a["median_log"]
    se = math.sqrt(a["se_log"] ** 2 + b["se_log"] ** 2)
    lo, hi = gap - z * se, gap + z * se
    eps = math.log(1.0 + epsilon_pct / 100.0)
    ratio = math.exp(gap)
    if lo > eps:
        status = "b_slower"
    elif hi < -eps:
        status = "b_faster"
    else:
        status = "equivalent"
    return {
        "status": status,
        "ratio": ratio,
        "ratio_pct": (ratio - 1.0) * 100.0,
        "gap_log": gap,
        "ci_low_pct": (math.exp(lo) - 1.0) * 100.0,
        "ci_high_pct": (math.exp(hi) - 1.0) * 100.0,
        "z": z,
        "epsilon_pct": epsilon_pct,
        "significant": status != "equivalent",
    }


def paired_rounds(ref_rounds: list[list[float]],
                  test_rounds: list[list[float]]) -> dict:
    """Per-round paired gap estimate.

    Thermal drift on this laptop is *common mode*: both workers slow down and
    speed up together over tens of seconds, which is exactly the timescale of
    the individual measurements.  Comparing pooled medians therefore charges
    the drift to whichever stack happened to be measured later, and with few
    rounds the worst case is a bimodal sample set (mediated by drift, not by
    the kernel).

    Pairing one test batch with the reference batch measured immediately
    before it removes the drift to first order: the gap is estimated inside
    each round and the estimator is the median of the per-round gaps.
    """
    gaps: list[float] = []
    for r, t in zip(ref_rounds, test_rounds):
        if not r or not t:
            continue
        gaps.append(statistics.median([math.log(x) for x in t])
                    - statistics.median([math.log(x) for x in r]))
    n = len(gaps)
    if n == 0:
        return {"n": 0}
    med = statistics.median(gaps)
    if n > 2:
        stdev = statistics.stdev(gaps)
        mad = statistics.median([abs(g - med) for g in gaps]) * 1.4826
        q = statistics.quantiles(gaps, n=4, method="inclusive")
        iqr = (q[2] - q[0]) / 1.349
    else:
        stdev = mad = iqr = 0.0
    sigma = max(stdev, mad, iqr, 1e-9)
    se = SQRT_PI_OVER_2 * sigma / math.sqrt(n)
    return {"n": n, "median_log": med, "sigma_log": sigma, "se_log": se,
            "round_gaps": gaps,
            "ratio": math.exp(med),
            "ci_low_log": med - 1.959963985 * se,
            "ci_high_log": med + 1.959963985 * se}


def decide_paired(paired: dict, z: float, epsilon_pct: float) -> dict:
    if paired.get("n", 0) == 0:
        return {"status": "no_samples"}
    gap = paired["median_log"]
    se = paired["se_log"]
    lo, hi = gap - z * se, gap + z * se
    eps = math.log(1.0 + epsilon_pct / 100.0)
    ratio = math.exp(gap)
    if lo > eps:
        status = "b_slower"
    elif hi < -eps:
        status = "b_faster"
    else:
        status = "equivalent"
    return {
        "status": status,
        "ratio": ratio,
        "ratio_pct": (ratio - 1.0) * 100.0,
        "gap_log": gap,
        "ci_low_pct": (math.exp(lo) - 1.0) * 100.0,
        "ci_high_pct": (math.exp(hi) - 1.0) * 100.0,
        "z": z,
        "epsilon_pct": epsilon_pct,
        "significant": status != "equivalent",
        "estimator": "paired_round",
    }


# ---------------------------------------------------------------------------
# clocks / throttle guard
# ---------------------------------------------------------------------------

def gpu_clocks() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,"
             "utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False).stdout.strip()
    except Exception:
        return {}
    if not out:
        return {}
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    keys = ("sm_clock_mhz", "mem_clock_mhz", "temp_c", "power_w", "util_pct")
    return dict(zip(keys, parts))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

class Worker:
    def __init__(self, stack: str, args) -> None:
        self.stack = stack
        cmd = [sys.executable, str(WORKER), "--stack", stack,
               "--kernel", args.kernel,
               "--batch", str(args.batch), "--seqlen", str(args.seqlen),
               "--heads", str(args.heads), "--head-dim", str(args.head_dim),
               "--m-block", str(args.m_block), "--n-block", str(args.n_block),
               "--threads", str(args.threads),
               "--m", str(args.m), "--n", str(args.n), "--k", str(args.k),
               "--warmup", str(args.warmup),
               "--dump-root", args.dump_root]
        self.stack = stack
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, bufsize=1)
        self.info = self._await("ready ")

    def _await(self, prefix: str):
        for line in self.proc.stdout:
            if line.startswith(prefix):
                return json.loads(line[len(prefix):])
            # anything else is informational noise from the front end
        raise RuntimeError(f"worker {self.stack} exited before '{prefix}'")

    def run(self, n: int) -> list[float]:
        self.proc.stdin.write(f"run {n}\n")
        self.proc.stdin.flush()
        for line in self.proc.stdout:
            if line.startswith("times "):
                return json.loads(line[len("times "):])
        raise RuntimeError(f"worker {self.stack} produced no times")

    def close(self) -> None:
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kernel", choices=["fa", "gemm"], default="fa")
    ap.add_argument("--self-stack", default="self",
                    help="stack under test (default: self)")
    ap.add_argument("--reference", default="official",
                    choices=["official", "fa2", "fa3", "fa4"],
                    help="reference stack to compare against")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--m-block", type=int, default=64)
    ap.add_argument("--n-block", type=int, default=64)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--rounds", type=int, default=6,
                    help="alternating batches per stack")
    ap.add_argument("--batch-size", type=int, default=5,
                    help="iterations per batch")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--z", type=float, default=1.96,
                    help="normal quantile for the gap CI")
    ap.add_argument("--epsilon-pct", type=float, default=3.0,
                    help="indifference zone: differences below are noise")
    ap.add_argument("--dump-root",
                    default=str(ROOT / "artifacts/stack_compare"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref, test = args.reference, args.self_stack
    clocks_before = gpu_clocks()
    workers = {ref: Worker(ref, args), test: Worker(test, args)}
    samples: dict[str, list[float]] = {ref: [], test: []}
    rounds: dict[str, list[list[float]]] = {ref: [], test: []}
    try:
        for _ in range(args.rounds):
            for stack in (ref, test):
                batch = workers[stack].run(args.batch_size)
                samples[stack].extend(batch)
                rounds[stack].append(batch)
    finally:
        for w in workers.values():
            w.close()
    clocks_after = gpu_clocks()

    stats = {k: robust_stats(v) for k, v in samples.items()}
    paired = paired_rounds(rounds[ref], rounds[test])
    verdict = decide_paired(paired, args.z, args.epsilon_pct)
    verdict_pooled = decide(stats[ref], stats[test], args.z, args.epsilon_pct)

    if args.kernel == "fa":
        tag = (f"fa_s{args.seqlen}_h{args.heads}_d{args.head_dim}"
               f"_m{args.m_block}_n{args.n_block}")
    else:
        tag = f"gemm_{args.m}x{args.n}x{args.k}"

    payload = {
        "tag": tag,
        "reference": ref,
        "stack_under_test": test,
        "protocol": {
            "rounds": args.rounds, "batch_size": args.batch_size,
            "warmup": args.warmup, "z": args.z,
            "epsilon_pct": args.epsilon_pct,
            "interleaved": True, "log_space": True,
        },
        "clocks_before": clocks_before,
        "clocks_after": clocks_after,
        "stats": stats,
        "paired": paired,
        "verdict": verdict,
        "verdict_pooled": verdict_pooled,
    }

    print(f"paired interleaved benchmark: {tag}")
    print(f"  protocol: {args.rounds} rounds x {args.batch_size} iters "
          f"(interleaved), z={args.z}, epsilon={args.epsilon_pct}%")
    for stack in (ref, test):
        s = stats[stack]
        print(f"  {stack:<9} n={s['n']:<3} median={s['median_us']:>10.1f} us  "
              f"sigma_log={s['sigma_log']:.4f}  se={s['se_log']:.4f}")
    print(f"  paired:    n={paired['n']:<3} ratio={paired['ratio']:.4f}  "
          f"sigma_log={paired['sigma_log']:.4f}  se={paired['se_log']:.4f}")
    print(f"  verdict: {verdict['status']}  ratio={verdict['ratio']:.4f} "
          f"({verdict['ratio_pct']:+.1f}%)  "
          f"CI95=[{verdict['ci_low_pct']:+.1f}%, "
          f"{verdict['ci_high_pct']:+.1f}%]  [paired-round]")
    print(f"  pooled : {verdict_pooled['status']}  "
          f"ratio={verdict_pooled['ratio']:.4f} "
          f"({verdict_pooled['ratio_pct']:+.1f}%)  "
          f"CI95=[{verdict_pooled['ci_low_pct']:+.1f}%, "
          f"{verdict_pooled['ci_high_pct']:+.1f}%]")
    if clocks_before and clocks_after:
        print(f"  clocks: before sm={clocks_before.get('sm_clock_mhz')} "
              f"temp={clocks_before.get('temp_c')}C -> "
              f"after sm={clocks_after.get('sm_clock_mhz')} "
              f"temp={clocks_after.get('temp_c')}C")

    suffix = "" if ref == "official" else f"_vs_{ref}"
    out = Path(args.out) if args.out else (
        Path(args.dump_root) / f"paired{suffix}_{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[paired_bench] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
