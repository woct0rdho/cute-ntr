"""Check which FlashAttention generations run on this SM86 device and how fast.

FA3 is Hopper-specific in principle (wgmma), but the installed
`flash_attn_interface` shim dispatches by compute capability and selects an
`kBlockM_sm86` kernel, so it needs to be probed rather than assumed.  FA2 and
FA4 both ship SM80 paths.  This script runs each entry point on the same
tensors, verifies the result against FA2, and reports an interleaved time so
the comparison target is the *fastest one that actually works*.
"""
from __future__ import annotations

import torch

SHAPES = [(1, 4096, 16, 64), (1, 4096, 16, 128), (1, 8192, 16, 128),
          (1, 16384, 16, 128)]


def time_call(fn, iters=7, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def main() -> int:
    print("device:", torch.cuda.get_device_name(0),
          torch.cuda.get_device_capability())
    for b, s, h, d in SHAPES:
        torch.manual_seed(0)
        q = torch.randn(b, s, h, d, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        ref = None
        print(f"-- S={s} H={h} D={d} --")
        for name in ("fa2", "fa3", "fa4"):
            try:
                if name == "fa2":
                    from flash_attn import flash_attn_func as fn
                    call = lambda: fn(q, k, v, causal=False)  # noqa: E731
                elif name == "fa3":
                    import flash_attn_interface as F
                    call = lambda: F.flash_attn_func(q, k, v)  # noqa: E731
                else:
                    from flash_attn.cute.interface import flash_attn_func as cfa
                    call = lambda: cfa(q, k, v, causal=False)[0]  # noqa: E731
                out = call()
                torch.cuda.synchronize()
                if not torch.isfinite(out).all():
                    raise RuntimeError("non-finite output")
                note = ""
                if ref is None:
                    ref = out
                else:
                    err = float((out.float() - ref.float()).abs().max())
                    note = f"max|diff| vs FA2 = {err:.2e}"
                us = time_call(call)
                print(f"   {name}: {us:9.1f} us   {note}")
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).replace("\n", " ")[:110]
                print(f"   {name}: unavailable ({type(exc).__name__}: {msg})")
        del q, k, v, ref
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
