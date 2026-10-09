"""Search the local PTX ISA text (or any doc) for a term, printing context.

Usage::

  python tools/reference/grep_doc.py maxnreg --lines 6
  python tools/reference/grep_doc.py "cp.async.wait_group" --lines 4
  python tools/reference/grep_doc.py ldmatrix --file docs/reference/ptx-isa.txt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("term")
    ap.add_argument("--file", default="docs/reference/ptx-isa.txt")
    ap.add_argument("--lines", type=int, default=5,
                    help="context lines after each hit")
    ap.add_argument("--max-hits", type=int, default=4)
    args = ap.parse_args()

    path = (ROOT / args.file) if not Path(args.file).is_absolute() else Path(args.file)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    hits = 0
    for i, line in enumerate(lines):
        if re.search(args.term, line, re.IGNORECASE):
            hits += 1
            print(f"--- line {i + 1} ---")
            for j in range(i, min(len(lines), i + 1 + args.lines)):
                row = lines[j].strip()
                if len(row) > 200:
                    row = row[:200] + " ..."
                row = row.encode("utf-8", "replace").decode("utf-8")
                print("   ", row)
            if hits >= args.max_hits:
                break
    if hits == 0:
        print(f"no match for {args.term!r} in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
