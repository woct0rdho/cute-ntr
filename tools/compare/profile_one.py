"""Scratch profiling driver: run one kernel once for Nsight Compute."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

stack = sys.argv[1] if len(sys.argv) > 1 else "self"
seqlen = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
heads = int(sys.argv[3]) if len(sys.argv) > 3 else 16
head_dim = int(sys.argv[4]) if len(sys.argv) > 4 else 64

# `cutlass` must resolve to the open stack only for the self case; the
# reference stacks need the installed nvidia-cutlass-dsl.
sys.path.insert(0, str(ROOT / "python"))
if stack == "self":
    sys.path[:0] = [str(ROOT / "python/cutlass_compat"),
                    str(ROOT / "tools/perf"),
                    str(ROOT / "tests/runtime/sm86")]

import torch  # noqa: E402

torch.manual_seed(0)
q = torch.randn(1, seqlen, heads, head_dim, device="cuda",
                dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
o = torch.empty_like(q)

if stack == "self":
    from sm86_flash_attention import compile_sm86_flash_attention

    compiled = compile_sm86_flash_attention(q, k, v, o)
    compiled()
else:
    if stack == "fa2":
        from flash_attn import flash_attn_func as _fn
    elif stack == "fa3":
        from flash_attn_interface import flash_attn_func as _fn
    else:
        from flash_attn.cute.interface import flash_attn_func as _cute

        def _fn(q, k, v, causal=False):
            return _cute(q, k, v, causal=causal)[0]

    _fn(q, k, v, causal=False)

torch.cuda.synchronize()
for _ in range(2):
    if stack == "self":
        compiled()
    else:
        _fn(q, k, v, causal=False)
    torch.cuda.synchronize()
