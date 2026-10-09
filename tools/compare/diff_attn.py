"""Scratch: diff our attention kernel against flash_attn on a small case."""
import sys

sys.path[:0] = ["python", "python/cutlass_compat", "tools/perf",
                "tests/runtime/sm86"]

import torch  # noqa: E402

from sm86_flash_attention import compile_sm86_flash_attention  # noqa: E402

seqlen = int(sys.argv[1]) if len(sys.argv) > 1 else 128
head_dim = int(sys.argv[2]) if len(sys.argv) > 2 else 64
heads = 16

torch.manual_seed(0)
q = torch.randn(1, seqlen, heads, head_dim, device="cuda",
                dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
o = torch.empty_like(q)
c = compile_sm86_flash_attention(q, k, v, o)
c()
torch.cuda.synchronize()

from flash_attn import flash_attn_func  # noqa: E402

ref = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
diff = (o.float() - ref.float()).abs()
print(f"S={seqlen} D={head_dim}: max={diff.max().item():.3e} "
      f"mean={diff.mean().item():.3e} "
      f"frac>3e-2={(diff > 3e-2).float().mean().item():.4f}")
bad = (diff > 3e-2)
if bad.any():
    idx = bad.nonzero()[:8]
    for i in idx:
        b, s, h, d = (int(x) for x in i)
        print(f"  row={s} head={h} dim={d}: ours={o[b,s,h,d].item():.4f} "
              f"ref={ref[b,s,h,d].item():.4f}")
    rows = sorted({int(x[1]) for x in bad.nonzero()})
    print(f"  bad rows: {len(rows)} of {seqlen}; first={rows[:6]}")
    cols = sorted({int(x[3]) for x in bad.nonzero()})
    print(f"  bad dims: {cols[:12]}")
