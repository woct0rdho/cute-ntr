"""Render the canonical FA artifact as a markdown table."""
import json
from pathlib import Path

ROOT = Path(r"C:\cute-ntr")
d = json.loads((ROOT / "artifacts/perf/sm86_flash_attention.json")
               .read_text(encoding="utf-8"))
rows = d["cases"]
print("| batch | seqlen | heads | dim | ours (us) | flash_attn (us) | ratio | max abs err |")
print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for r in rows:
    b, s, h, dd = r["shape"]
    c, f = r["custom"], r["flash_attn"]
    print(f"| {b} | {s} | {h} | {dd} | {c['median_us']:.0f} | "
          f"{f['median_us']:.0f} | {r['custom_over_flash']:.3f} | "
          f"{r['max_abs_error']:.2e} |")
