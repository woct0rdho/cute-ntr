"""Histogram every PTX opcode in a file, for static comparison of two kernels."""
import re
import sys
from collections import Counter
from pathlib import Path


def hist(path: Path) -> Counter:
    ops = Counter()
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("."):
            continue
        m = re.match(r"(?:@!?%?\w+\s+)?([a-z][a-z0-9_.]*)", s)
        if m:
            ops[m.group(1)] += 1
    return ops


def main() -> int:
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    ha, hb = hist(a), hist(b)
    keys = sorted(set(ha) | set(hb), key=lambda k: -(ha[k] + hb[k]))
    print(f"{'opcode':<34}{'A':>7}{'B':>7}   A/B")
    for k in keys[:45]:
        r = (f"{ha[k] / hb[k]:.2f}" if hb[k] else "-")
        print(f"{k:<34}{ha[k]:>7}{hb[k]:>7}   {r}")
    print(f"{'TOTAL':<34}{sum(ha.values()):>7}{sum(hb.values()):>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
