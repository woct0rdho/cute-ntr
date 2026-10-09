#!/usr/bin/env python3
"""Benchmark the focused SM86 FlashAttention forward path."""
import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "python", ROOT / "python/cutlass_compat", ROOT):
    sys.path.insert(0, str(path))
sys.path.insert(0, str(ROOT / "tools/perf"))

import perf_common as perf


def parse_ints(value):
    return [int(item) for item in value.split(",") if item]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=parse_ints,
                        default=[4096, 8192, 16384])
    parser.add_argument("--heads", type=parse_ints, default=[16, 32])
    parser.add_argument("--dims", type=parse_ints, default=[64, 128])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/perf/sm86_flash_attention.json")
    args = parser.parse_args()

    import torch
    from flash_attn import flash_attn_func
    from tests.runtime.sm86.sm86_flash_attention import (
        compile_sm86_flash_attention,
    )

    perf.init_cuda_context()
    capability = torch.cuda.get_device_capability()
    if capability != (8, 6):
        raise RuntimeError(f"requires SM86, got {capability}")
    payload = perf.envelope(
        "sm86_flash_attention", "self_vs_flash_attn",
        {"batch": 1, "lengths": args.lengths, "heads": args.heads,
         "head_dims": args.dims, "warmup": args.warmup,
         "iters": args.iters, "dtype": "float16", "causal": False})
    payload["flash_attn_version"] = __import__("flash_attn").__version__

    for seqlen in args.lengths:
        for heads in args.heads:
            for dim in args.dims:
                shape = (1, seqlen, heads, dim)
                case = {"shape": list(shape)}
                try:
                    torch.manual_seed(123 + seqlen + heads + dim)
                    torch.cuda.reset_peak_memory_stats()
                    q = torch.randn(shape, device="cuda", dtype=torch.float16)
                    k = torch.randn_like(q)
                    v = torch.randn_like(q)
                    out = torch.empty_like(q)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    compiled = compile_sm86_flash_attention(q, k, v, out)
                    compiled()
                    torch.cuda.synchronize()
                    case["compile_and_first_launch_s"] = round(
                        time.perf_counter() - started, 4)
                    custom = perf.cuda_time_us(
                        lambda: compiled(), args.warmup, args.iters)
                    reference = perf.cuda_time_us(
                        lambda: flash_attn_func(
                            q, k, v, dropout_p=0.0, causal=False),
                        args.warmup, args.iters)
                    expected = flash_attn_func(
                        q, k, v, dropout_p=0.0, causal=False)
                    difference = (out - expected).abs()
                    case.update({
                        "custom": custom,
                        "flash_attn": reference,
                        "custom_over_flash": round(
                            custom["median_us"] / reference["median_us"], 4),
                        "max_abs_error": float(difference.max()),
                        "mean_abs_error": float(difference.mean()),
                        "peak_allocated_mib": round(
                            torch.cuda.max_memory_allocated() / 2**20, 2),
                        "status": "PASS",
                    })
                    print(json.dumps(case), flush=True)
                    del q, k, v, out, expected, difference, compiled
                except Exception:
                    case["status"] = "FAIL"
                    case["error_tail"] = traceback.format_exc()[-1200:]
                    print(json.dumps(case), flush=True)
                payload["cases"].append(case)
                torch.cuda.empty_cache()
                gc.collect()

    perf.write_result(args.output, payload)
    return 0 if all(c["status"] == "PASS" for c in payload["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
