"""Count instructions inside the main loop body of a PTX kernel.

The static histogram is dominated by prologue and epilogue; what matters for a
K/V-tile loop is the body that executes `S / tile_n` times per CTA.  This
finds the backward branch that closes the innermost large loop and reports the
instruction mix of the block it covers.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

LABEL = re.compile(r"^\$L__BB\d+_(\d+):")
OPCODE = re.compile(r"(?:@!?%?\w+\s+)?([a-z][a-z0-9_.]*)")


def loop_body(path: Path) -> tuple[Counter, int]:
    lines = path.read_text(errors="replace").splitlines()
    # position of every label
    labels: dict[str, int] = {}
    for i, line in enumerate(lines):
        m = LABEL.match(line.strip())
        if m:
            labels[line.strip().rstrip(":")] = i
    best = None
    for i, line in enumerate(lines):
        s = line.strip()
        m = re.match(r"@!?%p\d+\s+bra(?:\.uni)?\s+(\$L__BB\d+_\d+);", s)
        if not m:
            m = re.match(r"bra(?:\.uni)?\s+(\$L__BB\d+_\d+);", s)
        if not m:
            continue
        tgt = m.group(1)
        if tgt not in labels:
            continue
        start = labels[tgt]
        if start < i and (best is None or i - start > best[1] - best[0]):
            best = (start, i)
    if best is None:
        return Counter(), 0
    ops = Counter()
    for line in lines[best[0]:best[1] + 1]:
        s = line.strip()
        if not s or s.startswith(("//", ".", "$", "{", "}")):
            continue
        m = OPCODE.match(s)
        if m:
            ops[m.group(1)] += 1
    return ops, best[1] - best[0]


def main() -> int:
    a, b, unroll_b = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    ha, la = loop_body(a)
    hb, lb = loop_body(b)
    print(f"A loop: {la} lines, {sum(ha.values())} instructions")
    print(f"B loop: {lb} lines, {sum(hb.values())} instructions "
          f"(divide by {unroll_b} to compare)")
    keys = sorted(set(ha) | set(hb),
                  key=lambda k: -(ha[k] + hb[k] / unroll_b))
    print(f"\n{'opcode':<44}{'A':>6}{'B/u':>8}  A/(B/u)")
    for k in keys[:40]:
        bu = hb[k] / unroll_b
        r = f"{ha[k] / bu:.2f}" if bu else "-"
        print(f"{k:<44}{ha[k]:>6}{bu:>8.1f}  {r}")
    print(f"\n{'TOTAL':<44}{sum(ha.values()):>6}"
          f"{sum(hb.values()) / unroll_b:>8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
