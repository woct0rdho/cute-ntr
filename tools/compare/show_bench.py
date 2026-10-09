"""Print the medians and verdicts of a set of paired_bench JSON artifacts."""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\cute-ntr/artifacts/stack_compare")
for name in sys.argv[1:]:
    p = ROOT / f"{name}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    st = d["stats"]
    parts = [f"{k}={st[k]['median_us']:.1f}us" for k in st]
    v = d["verdict"]
    print(f"{name:<34} {' '.join(parts)}  ratio={v['ratio']:.4f} "
          f"[{v['ci_low_pct']:+.1f}%,{v['ci_high_pct']:+.1f}%] {v['status']}")
