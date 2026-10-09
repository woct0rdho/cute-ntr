#!/usr/bin/env python3
"""A/B one source edit with the paired interleaved benchmark.

Usage::

  python tools/compare/ab_edit.py <file> <old-text> <new-text> -- <paired args>

Applies the replacement, runs `paired_bench.py`, then restores the file -
even if the benchmark fails.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    argv = sys.argv[1:]
    sep = argv.index("--")
    path = Path(argv[0])
    old, new = argv[1], argv[2]
    extra = argv[sep + 1:]
    text = path.read_text(encoding="utf-8")
    if old not in text:
        print(f"[ab_edit] pattern not present in {path}")
        return 2
    if text.count(old) != 1:
        print(f"[ab_edit] pattern appears {text.count(old)} times; ambiguous")
        return 2
    path.write_text(text.replace(old, new), encoding="utf-8")
    try:
        return subprocess.run(
            [sys.executable, str(ROOT / "tools/compare/paired_bench.py"), *extra],
            cwd=str(ROOT)).returncode
    finally:
        path.write_text(text, encoding="utf-8")
        print(f"[ab_edit] restored {path}")


if __name__ == "__main__":
    raise SystemExit(main())
