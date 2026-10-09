"""cutlass.testing - official CuTeDSL test/benchmark interface (compat).

The official package exposes the same measurement helpers the examples and
the autotuner use, so a benchmark driver written against the official API can
run unchanged on this stack:

  `JitArguments`      argument bundle handed to `benchmark`
  `benchmark`         warmup + per-iteration CUDA-event timing, µs
  `get_workspace_count`  cold-L2 workspace sizing
  `CuptiProfiler`     per-kernel CUPTI profiling (see note below)

`CuptiProfiler` is intentionally *not* reimplemented here: the open stack
uses Nsight Compute (`ncu`) for per-kernel counters, which is the same data
CUPTI exposes but with a stable report format.  Asking for it raises a clear
error pointing at the supported path instead of silently returning wrong
numbers.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


class JitArguments:
    """Bundle of positional/keyword arguments passed to `benchmark`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.references: list[Any] = []

    def add_to_scope(self, references: Any) -> None:
        """Keep external objects (e.g. torch tensors backing a view) alive."""
        if isinstance(references, (list, tuple)):
            self.references.extend(references)
        else:
            self.references.append(references)

    def __repr__(self) -> str:
        return (f"JitArguments(args={len(self.args)}, "
                f"kwargs={sorted(self.kwargs)})")


class CantImplementError(Exception):
    """Raised by kernels whose policy cannot serve a given problem."""


class CuptiProfiler:
    """Placeholder for the official CUPTI profiler."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "the open stack profiles kernels with Nsight Compute instead of "
            "CUPTI; use tools/compare/dump_kernels.py or "
            "'ncu --set full -k regex:<kernel>' to collect the same counters")


def get_workspace_count(one_workspace_bytes: int,
                        warmup_iterations: int,
                        iterations: int) -> int:
    """Number of workspaces needed to keep L2 cold across the run."""
    num_l2_cache_bytes = _l2_cache_size_bytes()
    num_workspaces = (num_l2_cache_bytes * 3) // max(1, one_workspace_bytes) + 1
    num_iters = warmup_iterations + iterations
    return min(num_iters, num_workspaces)


def _l2_cache_size_bytes() -> int:
    import torch

    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return int(getattr(props, "L2_cache_size", 0)) or (6 * 1024 * 1024)
    except Exception:
        return 6 * 1024 * 1024


def _cuda_events(stream=None):
    import torch

    return torch.cuda.Event(enable_timing=True)


def benchmark(
    callable: Callable,
    *,
    warmup_iterations: int = 10,
    iterations: int = 100,
    stream=None,
    kernel_arguments: Optional[JitArguments] = None,
    workspace_generator: Optional[Callable[[], JitArguments]] = None,
    workspace_count: int = 1,
    use_cuda_graphs: bool = False,
    use_cupti: bool = False,
    nvtx_tag: Optional[str] = None,
) -> float:
    """Time `callable` with CUDA events; returns the mean time in µs.

    Signature and semantics follow the official
    `cutlass.testing.benchmark`: the callable is invoked with the argument
    bundle's `args`/`kwargs` (either a fixed `kernel_arguments` or a
    fresh bundle from `workspace_generator` per workspace), each launch is
    bracketed by CUDA events, and the mean of the measured iterations is
    returned.  `use_cuda_graphs` is accepted but ignored: the open stack's
    launcher is a plain kernel launch, and capture adds nothing to the
    measurement that per-iteration events do not already isolate.
    """
    import torch

    if use_cupti:
        raise NotImplementedError(
            "use_cupti is not available on the open stack; profile with "
            "Nsight Compute instead (tools/compare/dump_kernels.py --sass, "
            "or ncu -k regex:<kernel>)")
    if workspace_count < 1:
        raise ValueError("workspace_count must be at least 1")
    if workspace_generator is None:
        if workspace_count != 1:
            raise ValueError(
                "a single workspace requires no workspace_generator")
        if kernel_arguments is None:
            raise ValueError(
                "pass kernel_arguments or a workspace_generator")
        workspace_generator = lambda: kernel_arguments  # noqa: E731

    workspaces = [workspace_generator() for _ in range(workspace_count)]
    for workspace in workspaces:
        if not isinstance(workspace, JitArguments):
            raise TypeError(
                "workspace_generator/kernel_arguments must produce "
                "JitArguments")

    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)

    def launch(ws: JitArguments) -> None:
        callable(*ws.args, **ws.kwargs)

    for ws in workspaces:
        for _ in range(max(0, warmup_iterations // workspace_count)):
            launch(ws)
    torch.cuda.synchronize(device)

    per_ws = max(1, iterations // workspace_count)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(per_ws)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(per_ws)]
    elapsed_ms: list[float] = []
    for ws in workspaces:
        for i in range(per_ws):
            starts[i].record()
            launch(ws)
            ends[i].record()
        torch.cuda.synchronize(device)
        elapsed_ms.extend(s.elapsed_time(e) for s, e in zip(starts, ends))
    if not elapsed_ms:
        return float("nan")
    return float(sum(elapsed_ms) / len(elapsed_ms) * 1000.0)


class TensorInitConfig:
    """Placeholder for the official tensor-init helper (unused here)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


__all__ = [
    "JitArguments",
    "benchmark",
    "get_workspace_count",
    "CantImplementError",
    "CuptiProfiler",
    "TensorInitConfig",
]
