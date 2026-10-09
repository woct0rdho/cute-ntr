"""cutlass.cute.testing - deprecated alias kept by the official package.

The official CuTeDSL moved these helpers to `cutlass.testing` but keeps the
`cutlass.cute.testing` spelling working (with a deprecation warning) because
the Ampere examples still import it.  Mirror that here so unmodified example
drivers run against either stack.
"""
from __future__ import annotations

from cutlass.testing import (  # noqa: F401
    CantImplementError,
    CuptiProfiler,
    JitArguments,
    TensorInitConfig,
    benchmark,
    get_workspace_count,
)

__all__ = [
    "JitArguments",
    "benchmark",
    "get_workspace_count",
    "CantImplementError",
    "CuptiProfiler",
    "TensorInitConfig",
]
