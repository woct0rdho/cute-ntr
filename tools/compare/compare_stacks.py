#!/usr/bin/env python3
"""compare_stacks.py - diff official CuTeDSL against the open stack.

Consumes the per-case JSON reports produced by `dump_kernels.py` for the
`official` and `self` stacks and prints a side-by-side table:

  * wall time and the official/self speed ratio;
  * ptxas resource usage (registers, spills, shared memory);
  * PTX instruction mix;
  * SASS instruction mix.

The point of the static columns is *not* that the numbers should match: the
two kernels are different implementations.  They are there to localise a
gap.  Two normalizations make them comparable:

  `per-iteration`  instruction counts divided by the number of K/V tiles
                     each kernel processes (k_tiles), which removes the
                     effect of differently unrolled / multi-stage outer
                     loops;
  `ratio`          self / official, where < 1 means we emit fewer
                     instructions than the official kernel for the same
                     normalized unit of work.

Usage::

  python tools/compare/compare_stacks.py \
      --pair artifacts/stack_compare/official_fa_s4096_h16_d64_m64_n64.json \
             artifacts/stack_compare/self_fa_s4096_h16_d64_m64_n64.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PTX_KEYS = ("cp.async", "ldmatrix", "mma.sync", "bar.sync", "ld.shared",
            "st.shared", "ld.global", "st.global", "shfl.sync", "ex2.approx",
            "cvt.rn.f16.f32")
SASS_KEYS = ("LDGSTS", "LDSM", "HMMA", "LDS", "STS", "LDG", "STG", "BAR",
             "SHFL", "MUFU", "FFMA", "F2F")


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def k_tiles(case: dict) -> int:
    shape = case.get("shape", {})
    if case.get("kernel") == "fa":
        n = shape.get("n_block", 64)
        return max(1, int(shape.get("seqlen", 1)) // max(1, int(n)))
    return 1


def row(label: str, official, self, unit: str = "") -> str:
    o = "-" if official is None else f"{official}"
    s = "-" if self is None else f"{self}"
    r = ""
    if isinstance(official, (int, float)) and isinstance(self, (int, float)) \
            and official:
        r = f"{self / official:.3f}"
    return f"{label:<22} {o:>12} {s:>12} {r:>8}  {unit}"


def compare(official: dict, self_: dict) -> str:
    out: list[str] = []
    o_tiles = k_tiles(official)
    s_tiles = k_tiles(self_)
    out.append(f"kernel: {official.get('kernel')}  "
               f"tag: {official.get('tag')} vs {self_.get('tag')}")
    out.append(f"k_tiles per launch: official={o_tiles} self={s_tiles}")
    out.append("")
    out.append(row("metric", "official", "self", "self/official"))
    out.append("-" * 60)
    out.append(row("median_us", official.get("median_us"),
                   self_.get("median_us"), "us"))
    out.append(row("compile_s", round(official.get("compile_s", 0), 3),
                   round(self_.get("compile_s", 0), 3), "s"))

    o_res = official.get("resources") or {}
    s_res = self_.get("resources") or {}
    out.append("")
    out.append("-- ptxas resources --")
    for key in ("registers", "spill_stores", "spill_loads", "smem_bytes",
                "stack_bytes", "barriers"):
        out.append(row(key, o_res.get(key), s_res.get(key)))

    o_ptx = official.get("ptx_histogram") or {}
    s_ptx = self_.get("ptx_histogram") or {}
    out.append("")
    out.append(f"-- PTX instruction mix (raw / per K-tile) --")
    for key in PTX_KEYS:
        ov, sv = o_ptx.get(key), s_ptx.get(key)
        if not ov and not sv:
            continue
        o_norm = None if ov is None else round(ov / o_tiles, 2)
        s_norm = None if sv is None else round(sv / s_tiles, 2)
        out.append(f"{key:<22} {ov:>6} /{o_norm:>7} {sv:>6} /{s_norm:>7}"
                   f"   raw ratio {0 if not ov else sv / ov:.3f}")

    o_sass = official.get("sass_histogram") or {}
    s_sass = self_.get("sass_histogram") or {}
    if o_sass or s_sass:
        out.append("")
        out.append("-- SASS instruction mix (raw / per K-tile) --")
        for key in SASS_KEYS:
            ov, sv = o_sass.get(key), s_sass.get(key)
            if not ov and not sv:
                continue
            o_norm = None if ov is None else round(ov / o_tiles, 2)
            s_norm = None if sv is None else round(sv / s_tiles, 2)
            out.append(f"{key:<22} {ov:>6} /{o_norm:>7} {sv:>6} /{s_norm:>7}"
                       f"   raw ratio {0 if not ov else sv / ov:.3f}")

    out.append("")
    out.append(f"-- artifacts --")
    out.append(f"official ptx: {official.get('ptx_path')}")
    out.append(f"self     ptx: {self_.get('ptx_path')}")
    out.append(f"official sass: {official.get('sass_path')}")
    out.append(f"self     sass: {self_.get('sass_path')}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair", nargs=2, metavar=("OFFICIAL_JSON", "SELF_JSON"))
    ap.add_argument("--dir", default=str(ROOT / "artifacts/stack_compare"),
                    help="directory of <stack>_<kernel>_<tag>.json reports")
    ap.add_argument("--kernel", default=None, choices=[None, "fa", "gemm"])
    ap.add_argument("--out", default=None, help="write the table here")
    args = ap.parse_args()

    pairs: list[tuple[dict, dict]] = []
    if args.pair:
        pairs.append((load(args.pair[0]), load(args.pair[1])))
    else:
        base = Path(args.dir)
        official = {}
        self_ = {}
        for path in sorted(base.glob("*.json")):
            try:
                case = load(path)
            except json.JSONDecodeError:
                continue
            tag = case.get("tag", "")
            if args.kernel and case.get("kernel") != args.kernel:
                continue
            stem = tag.split("_", 1)[1] if "_" in tag else tag
            if case.get("stack") == "official":
                official[stem] = case
            elif case.get("stack") == "self":
                self_[stem] = case
        for stem in sorted(set(official) & set(self_)):
            pairs.append((official[stem], self_[stem]))

    if not pairs:
        print("no matching official/self report pairs found")
        return 1

    tables = [compare(o, s) for o, s in pairs]
    text = ("\n\n" + "=" * 60 + "\n\n").join(tables)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\n[compare_stacks] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
