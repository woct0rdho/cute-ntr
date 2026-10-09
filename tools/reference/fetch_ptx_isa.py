#!/usr/bin/env python3
"""fetch_ptx_isa.py - cache the PTX ISA (and CUDA C) docs for local search.

The PTX ISA reference is the authority for things this project has to get
exactly right: where a performance-tuning directive may appear (11.4),
`cp.async` group semantics, `ldmatrix` fragment layouts, `.target`
attributes.  Fetching it once and grepping locally is much faster than
re-reading the web version, and `tools/reference/grep_doc.py` searches the
extracted text.

Usage::

  python tools/reference/fetch_ptx_isa.py          # both docs
  python tools/reference/fetch_ptx_isa.py --ptx    # PTX ISA only
"""
from __future__ import annotations

import argparse
import html
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/reference"

DOCS = {
    "ptx-isa": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html",
    "cuda-c": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html",
}


def fetch(name: str, url: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(url, timeout=180).read().decode(
        "utf-8", "replace")
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    path = OUT / f"{name}.txt"
    path.write_text(text, encoding="utf-8")
    print(f"{name}: {len(raw)} bytes html -> {path.stat().st_size} bytes text")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ptx", action="store_true", help="PTX ISA only")
    args = ap.parse_args()
    names = ["ptx-isa"] if args.ptx else list(DOCS)
    for name in names:
        fetch(name, DOCS[name])
    print(f"\nsearch with: python tools/reference/grep_doc.py <term> "
          f"--file docs/reference/ptx-isa.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
