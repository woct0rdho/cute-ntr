#!/usr/bin/env python3
"""Thermal-aware repeated SM86 matmul benchmark.

Compares the CuTeDSL kernel with eager torch.mm using independent batches. The
reported comparison is calculated in log-time space following EvoTensile's
noisy-measurement guidance.
"""
import argparse
import json
import math
import statistics
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

FOCUSED_SHAPES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]
Z95 = 1.959963984540054


def parse_shapes(value):
    if value == "focused":
        return FOCUSED_SHAPES
    shapes = []
    for item in value.split(","):
        dims = tuple(int(x) for x in item.split("x"))
        if len(dims) != 3:
            raise ValueError(f"shape must be MxNxK, got {item!r}")
        shapes.append(dims)
    return shapes


def memory_estimate(M, N, K):
    # A, B, and the FP16 output. Correctness creates one temporary reference.
    matrix_bytes = (M * K + K * N + M * N) * 2
    correctness_bytes = matrix_bytes + M * N * 2
    return matrix_bytes, correctness_bytes


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def timing_stats(times):
    positive = [float(x) for x in times if x > 0 and math.isfinite(x)]
    logs = [math.log(x) for x in positive]
    center = statistics.median(logs)
    deviations = [abs(x - center) for x in logs]
    mad = statistics.median(deviations)
    q25 = percentile(logs, 0.25)
    q75 = percentile(logs, 0.75)
    iqr = q75 - q25
    stdev = statistics.stdev(logs) if len(logs) > 1 else 0.0
    sigma = max(stdev, 1.4826 * mad, iqr / 1.349)
    se = math.sqrt(math.pi / 2.0) * sigma / math.sqrt(len(logs))
    return {
        "samples": len(positive),
        "median_time_us": statistics.median(positive),
        "log_median_time_us": math.exp(center),
        "mean_log_time_us": math.exp(statistics.mean(logs)),
        "p10_us": percentile(positive, 0.10),
        "p90_us": percentile(positive, 0.90),
        "min_us": min(positive),
        "max_us": max(positive),
        "log_stdev": stdev,
        "log_mad": mad,
        "log_iqr": iqr,
        "log_sigma": sigma,
        "median_log_se": se,
        "high_side_outliers": sum(x > center + 3.0 * sigma for x in logs),
    }


def round_numbers(value):
    if isinstance(value, dict):
        return {k: round_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [round_numbers(v) for v in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def gpu_state(torch):
    free, total = torch.cuda.mem_get_info()
    return {
        "metadata": P.gpu_metadata(),
        "cuda_free_bytes": free,
        "cuda_total_bytes": total,
    }


def measure_batch(torch, custom, torch_mm, warmup, iters, custom_first):
    order = (("custom", custom), ("torch_mm", torch_mm))
    if not custom_first:
        order = tuple(reversed(order))
    for _ in range(warmup):
        for _, fn in order:
            fn()
    torch.cuda.synchronize()
    events = {"custom": [], "torch_mm": []}
    for _ in range(iters):
        for name, fn in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events[name].append((start, end))
    torch.cuda.synchronize()
    return {
        name: [start.elapsed_time(end) * 1000.0 for start, end in pairs]
        for name, pairs in events.items()
    }


def compare_stats(custom, torch_mm):
    custom_log = math.log(custom["log_median_time_us"])
    torch_log = math.log(torch_mm["log_median_time_us"])
    gap = custom_log - torch_log
    se_gap = math.sqrt(custom["median_log_se"] ** 2 +
                       torch_mm["median_log_se"] ** 2)
    low = math.exp(gap - Z95 * se_gap)
    high = math.exp(gap + Z95 * se_gap)
    max_sigma = max(custom["log_sigma"], torch_mm["log_sigma"])
    noise_pct = (math.exp(max_sigma) - 1.0) * 100.0
    return {
        "custom_over_torch_log_center": math.exp(gap),
        "custom_over_torch_95ci": [low, high],
        "comparison_status": (
            "unresolved" if low <= 1.0 <= high else
            "custom_faster" if high < 1.0 else "torch_faster"),
        "max_log_sigma": max_sigma,
        "max_multiplicative_noise_pct": noise_pct,
        "noise_status": "noisy" if noise_pct >= 5.0 else "stable",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shapes", default="focused",
                    help="focused or comma-separated MxNxK values")
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cooldown-s", type=float, default=3.0)
    ap.add_argument("--out", default=str(
        ROOT / "artifacts/perf/sm86_matmul_large_noisy.json"))
    args = ap.parse_args()
    if args.batches < 2 or args.iters < 2:
        raise ValueError("use at least two batches and two measured iterations")

    import torch

    P.init_cuda_context()
    capability = torch.cuda.get_device_capability()
    if capability != (8, 6):
        raise RuntimeError(f"requires sm86 device, got sm{capability[0]}{capability[1]}")
    cases = []
    shapes = parse_shapes(args.shapes)
    for shape_index, (M, N, K) in enumerate(shapes):
        matrix_bytes, correctness_bytes = memory_estimate(M, N, K)
        free_before, total_memory = torch.cuda.mem_get_info()
        fits = correctness_bytes < free_before
        case = {
            "shape": [M, N, K],
            "estimated_matrix_bytes": matrix_bytes,
            "estimated_peak_correctness_bytes": correctness_bytes,
            "free_bytes_before_allocation": free_before,
            "total_memory_bytes": total_memory,
            "allocation_feasible": fits,
        }
        if not fits:
            case["status"] = "skipped_memory"
            cases.append(case)
            print(f"M={M} N={N} K={K}: skipped, insufficient free memory", flush=True)
            continue

        if shape_index:
            time.sleep(args.cooldown_s)
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
        del reference
        torch.cuda.synchronize()

        batches = []
        all_times = {"custom": [], "torch_mm": []}
        for batch_index in range(args.batches):
            if batch_index or shape_index:
                time.sleep(args.cooldown_s)
            before = gpu_state(torch)
            measured = measure_batch(
                torch, compiled, lambda: torch.mm(a, b, out=torch_out),
                args.warmup, args.iters, custom_first=(batch_index % 2 == 0))
            after = gpu_state(torch)
            for name, values in measured.items():
                all_times[name].extend(values)
            batches.append({
                "batch": batch_index + 1,
                "order": "custom_then_torch" if batch_index % 2 == 0
                else "torch_then_custom",
                "custom": timing_stats(measured["custom"]),
                "torch_mm": timing_stats(measured["torch_mm"]),
                "gpu_before": before,
                "gpu_after": after,
            })
            print("  batch %d/%d: custom %.2f us, torch.mm %.2f us" % (
                batch_index + 1, args.batches,
                statistics.median(measured["custom"]),
                statistics.median(measured["torch_mm"])), flush=True)

        custom_stats = timing_stats(all_times["custom"])
        torch_stats = timing_stats(all_times["torch_mm"])
        comparison = compare_stats(custom_stats, torch_stats)
        case.update({
            "status": "ok",
            "compile_ms": compile_ms,
            "custom": custom_stats,
            "torch_mm": torch_stats,
            "comparison": comparison,
            "batches": batches,
        })
        print("  aggregate: custom %.2f us, torch.mm %.2f us, ratio %.3f [%s, %s] (%s; %s)" % (
            custom_stats["log_median_time_us"],
            torch_stats["log_median_time_us"],
            comparison["custom_over_torch_log_center"],
            comparison["custom_over_torch_95ci"][0],
            comparison["custom_over_torch_95ci"][1],
            comparison["comparison_status"], comparison["noise_status"]), flush=True)
        del a, b, custom_out, torch_out
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        cases.append(case)

    params = {
        "protocol": "independent alternating batches",
        "batches": args.batches,
        "warmup_per_batch": args.warmup,
        "iters_per_batch": args.iters,
        "cooldown_s": args.cooldown_s,
        "confidence": 0.95,
        "stats": "median log time; sigma=max(stdev,1.4826*MAD,IQR/1.349)",
        "input_dtype": "float16",
        "output_dtype": "float16",
        "accumulation": "float32 custom; torch.mm backend default",
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
        "shapes": [list(x) for x in shapes],
    }
    payload = P.envelope("sm86_matmul_large_noisy", "self", params)
    payload["cuda_device"] = torch.cuda.get_device_name()
    payload["compute_capability"] = list(capability)
    payload["cases"] = round_numbers(cases)
    P.write_result(args.out, payload)


if __name__ == "__main__":
    main()
