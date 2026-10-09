#!/usr/bin/env python3
"""sweep.py - run the paired A/B benchmark over the supported shape matrix.

Every case compares this repository's open stack against a reference
implementation on the same device, with the interleaved log-space protocol
from `paired_bench.py` (which is what makes the numbers trustworthy on a
throttling laptop).

Usage::

  python tools/compare/sweep.py                       # full FA matrix
  python tools/compare/sweep.py --reference fa4
  python tools/compare/sweep.py --cases fa_s4096_h16_d64
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAIRED = ROOT / "tools/compare/paired_bench.py"

# (tag, extra args) - the attention contract of this repository
FA_CASES = [
    ("fa_s4096_h16_d64", ["--kernel", "fa", "--seqlen", "4096",
                          "--heads", "16", "--head-dim", "64",
                          "--m-block", "64", "--n-block", "64"]),
    ("fa_s4096_h32_d64", ["--kernel", "fa", "--seqlen", "4096",
                          "--heads", "32", "--head-dim", "64",
                          "--m-block", "64", "--n-block", "64"]),
    ("fa_s8192_h16_d64", ["--kernel", "fa", "--seqlen", "8192",
                          "--heads", "16", "--head-dim", "64",
                          "--m-block", "64", "--n-block", "64"]),
    ("fa_s8192_h32_d64", ["--kernel", "fa", "--seqlen", "8192",
                          "--heads", "32", "--head-dim", "64",
                          "--m-block", "64", "--n-block", "64"]),
    ("fa_s16384_h16_d64", ["--kernel", "fa", "--seqlen", "16384",
                           "--heads", "16", "--head-dim", "64",
                           "--m-block", "64", "--n-block", "64"]),
    ("fa_s16384_h32_d64", ["--kernel", "fa", "--seqlen", "16384",
                           "--heads", "32", "--head-dim", "64",
                           "--m-block", "64", "--n-block", "64"]),
    ("fa_s4096_h16_d128", ["--kernel", "fa", "--seqlen", "4096",
                           "--heads", "16", "--head-dim", "128",
                           "--m-block", "64", "--n-block", "64"]),
    ("fa_s4096_h32_d128", ["--kernel", "fa", "--seqlen", "4096",
                           "--heads", "32", "--head-dim", "128",
                           "--m-block", "64", "--n-block", "64"]),
    ("fa_s8192_h16_d128", ["--kernel", "fa", "--seqlen", "8192",
                           "--heads", "16", "--head-dim", "128",
                           "--m-block", "64", "--n-block", "64"]),
    ("fa_s8192_h32_d128", ["--kernel", "fa", "--seqlen", "8192",
                           "--heads", "32", "--head-dim", "128",
                           "--m-block", "64", "--n-block", "64"]),
    ("fa_s16384_h16_d128", ["--kernel", "fa", "--seqlen", "16384",
                            "--heads", "16", "--head-dim", "128",
                            "--m-block", "64", "--n-block", "64"]),
    ("fa_s16384_h32_d128", ["--kernel", "fa", "--seqlen", "16384",
                            "--heads", "32", "--head-dim", "128",
                            "--m-block", "64", "--n-block", "64"]),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", default="official",
                    choices=["official", "fa2", "fa3", "fa4"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cases", default="all",
                    help="comma-separated tags, or all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    wanted = None if args.cases == "all" else set(args.cases.split(","))
    rows = []
    for tag, extra in FA_CASES:
        if wanted is not None and tag not in wanted:
            continue
        out = ROOT / f"artifacts/stack_compare/paired_{tag}.json"
        cmd = [sys.executable, str(PAIRED), *extra,
               "--reference", args.reference,
               "--rounds", str(args.rounds),
               "--batch-size", str(args.batch_size),
               "--warmup", str(args.warmup),
               "--out", str(out)]
        print(f"[sweep] {tag}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  FAILED: {proc.stderr.strip()[-400:]}")
            rows.append({"tag": tag, "status": "FAILED"})
            continue
        data = json.loads(out.read_text(encoding="utf-8"))
        ref, test = args.reference, "self"
        verdict = data["verdict"]
        rows.append({
            "tag": tag,
            "reference_us": data["stats"][ref]["median_us"],
            "self_us": data["stats"][test]["median_us"],
            "ratio": verdict["ratio"],
            "ci_low_pct": verdict["ci_low_pct"],
            "ci_high_pct": verdict["ci_high_pct"],
            "status": verdict["status"],
            "reference": args.reference,
            "n": data["stats"][test]["n"],
        })
        # keep the artifacts directory tidy: this file is the summary
        out.unlink(missing_ok=True)
        print(f"  {rows[-1]['reference_us']:.0f} -> "
              f"{rows[-1]['self_us']:.0f} us  ratio {rows[-1]['ratio']:.3f} "
              f"({rows[-1]['status']})", flush=True)

    out = Path(args.out) if args.out else (
        ROOT / f"artifacts/stack_compare/sweep_vs_{args.reference}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"reference": args.reference, "rows": rows},
                              indent=1), encoding="utf-8")

    print()
    print(f"{'case':<22} {'ref us':>9} {'self us':>9} {'ratio':>7} "
          f"{'95% CI':>18}  status")
    print("-" * 82)
    for r in rows:
        if r.get("status") == "FAILED":
            print(f"{r['tag']:<22} {'-':>9} {'-':>9} {'-':>7} {'-':>18}  FAILED")
            continue
        ci = f"[{r['ci_low_pct']:+.1f},{r['ci_high_pct']:+.1f}]%"
        print(f"{r['tag']:<22} {r['reference_us']:>9.0f} "
              f"{r['self_us']:>9.0f} {r['ratio']:>7.3f} {ci:>18}  "
              f"{r['status']}")
    print(f"\n[sweep] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
