"""Run one FA generation a few times so Nsight Compute can profile it."""
import sys

import torch

which = sys.argv[1] if len(sys.argv) > 1 else "fa3"
seqlen = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
heads = int(sys.argv[3]) if len(sys.argv) > 3 else 16
dim = int(sys.argv[4]) if len(sys.argv) > 4 else 128

q = torch.randn(1, seqlen, heads, dim, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

if which == "fa2":
    from flash_attn import flash_attn_func as fn
elif which == "fa3":
    from flash_attn_interface import flash_attn_func as fn
else:
    from flash_attn.cute.interface import flash_attn_func as cute_fn

    def fn(a, b, c, causal=False):
        return cute_fn(a, b, c, causal=causal)[0]

for _ in range(4):
    fn(q, k, v)
    torch.cuda.synchronize()
