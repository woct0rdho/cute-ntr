#!/usr/bin/env python3
"""Benchmark the SM86 CuTe matmul against eager ``torch.mm``.

The benchmark requires the local compute-capability 8.6 device. Compilation,
first launch, allocation, and correctness checks are reported separately from
steady-state CUDA-event timings.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))
sys.path.insert(0, str(ROOT / "tools/perf"))
sys.path.insert(0, str(ROOT / "tests/runtime/sm86"))

import perf_common as P
from sm86_matmul import compile_sm86_matmul

SHAPES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (1024, 4096, 1024),
    (4096, 1024, 1024),
]


def tflops(m, n, k, us):
    return 2.0 * m * n * k / (us * 1e-6) / 1e12


def parse_shapes(value):
    if value == "required":
        return SHAPES
    result = []
    for item in value.split(","):
        dims = tuple(int(x) for x in item.split("x"))
        if len(dims) != 3:
            raise ValueError(f"shape must be MxNxK, got {item!r}")
        result.append(dims)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--shapes", default="required",
                    help="required or comma-separated MxNxK values")
    ap.add_argument("--out", default=str(ROOT / "artifacts/perf/sm86_matmul.json"))
    args = ap.parse_args()

    import torch

    P.init_cuda_context()
    capability = torch.cuda.get_device_capability()
    if capability != (8, 6):
        raise RuntimeError(f"requires sm86 device, got sm{capability[0]}{capability[1]}")
    shapes = parse_shapes(args.shapes)
    cases = []

    for M, N, K in shapes:
        if M % 64 or N % 32 or K % 16:
            raise ValueError(f"shape {(M, N, K)} is not divisible for the initial kernel")
        print(f"benchmarking M={M} N={N} K={K}", flush=True)
        torch.manual_seed(M + N + K)
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        custom_out = torch.empty(M, N, device="cuda", dtype=torch.float16)
        torch_out = torch.empty_like(custom_out)

        compile_start = time.perf_counter()
        compiled = compile_sm86_matmul(a, b, custom_out)
        compiled()
        torch.cuda.synchronize()
        compile_ms = (time.perf_counter() - compile_start) * 1000.0

        reference = torch.mm(a, b)
        torch.testing.assert_close(custom_out, reference, atol=0.25, rtol=0.05)

        custom_timing = P.cuda_time_us(
            compiled, args.warmup, args.iters)
        torch_timing = P.cuda_time_us(
            lambda: torch.mm(a, b, out=torch_out), args.warmup, args.iters)
        custom_timing["tflops"] = round(
            tflops(M, N, K, custom_timing["median_us"]), 3)
        torch_timing["tflops"] = round(
            tflops(M, N, K, torch_timing["median_us"]), 3)
        ratio = custom_timing["tflops"] / torch_timing["tflops"]
        print("  custom: %.2f us, %.3f TF/s; torch.mm: %.2f us, %.3f TF/s; ratio %.3f" % (
            custom_timing["median_us"], custom_timing["tflops"],
            torch_timing["median_us"], torch_timing["tflops"], ratio),
            flush=True)
        cases.append({
            "shape": [M, N, K],
            "compile_ms": round(compile_ms, 3),
            "custom": custom_timing,
            "torch_mm": torch_timing,
            "custom_over_torch_tflops": round(ratio, 4),
        })

    params = {
        "warmup": args.warmup,
        "iters": args.iters,
        "input_dtype": "float16",
        "output_dtype": "float16",
        "accumulation": "float32",
        "layout": "contiguous row-major",
        "target": "sm_86",
        "tile_dispatch": {
            "N <= 2048": {"tile_shape_mnk": [64, 64, 32],
                           "block_threads": 128,
                           "shared_stride_a_half": 40,
                           "shared_stride_b_half": 72},
            "N > 2048": {"tile_shape_mnk": [64, 128, 32],
                          "block_threads": 256,
                          "shared_stride_a_half": 40,
                          "shared_stride_b_half": 136},
        },
        "pipeline_stages": 2,
        "async_copy": "cp.async.ca.shared.global",
        "shared_row_padding_half": 8,
        "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "torch_allow_fp16_accumulation": getattr(
            torch.backends.cuda.matmul, "allow_fp16_accumulation", None),
        "torch_allow_fp16_reduced_precision_reduction": getattr(
            torch.backends.cuda.matmul,
            "allow_fp16_reduced_precision_reduction", None),
        "shapes": [list(x) for x in shapes],
    }
    payload = P.envelope("sm86_matmul", "self", params)
    payload["cuda_device"] = torch.cuda.get_device_name()
    payload["compute_capability"] = list(capability)
    payload["cases"] = cases
    P.write_result(args.out, payload)


if __name__ == "__main__":
    main()
