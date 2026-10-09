"""A/B one tile-policy constant pair against a reference, restoring the file.

Usage::

  python tools/compare/ab_tile.py <label> <old1>|<new1> <old2>|<new2> ... \
                                 -- <paired_bench args>
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\cute-ntr")
SRC = ROOT / "tests/runtime/sm86/sm86_flash_attention.py"


def main() -> int:
    argv = sys.argv[1:]
    sep = argv.index("--")
    label, edits, bench = argv[0], argv[1:sep], argv[sep + 1:]
    text = SRC.read_text(encoding="utf-8")
    patched = text
    for spec in edits:
        old, new = spec.split("|", 1)
        if patched.count(old) != 1:
            print(f"{label}: pattern appears {patched.count(old)} times: {old!r}")
            return 2
        patched = patched.replace(old, new)
    SRC.write_text(patched, encoding="utf-8")
    try:
        r = subprocess.run(
            [sys.executable, str(ROOT / "tools/compare/paired_bench.py"), *bench],
            cwd=str(ROOT), capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "verdict" in line or "pooled" in line or "exited before" in line:
                print(f"{label:<22} {line.strip()}")
        if r.returncode != 0:
            print(r.stdout[-900:])
            return r.returncode
    finally:
        SRC.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
