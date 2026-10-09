"""Official-interface smoke test on the SM86 kernel.

The point of this module is that the benchmark is driven through
`cutlass.testing` - the *official* interface - against this repository's
kernel, so a benchmark driver written for the official CuTeDSL runs against
either stack unchanged.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))
sys.path.insert(0, str(ROOT / "tests/runtime/sm86"))

import torch  # noqa: E402

from cutlass import testing  # noqa: E402


def _require_sm86():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("requires an sm86 device")


@pytest.mark.sm86
def test_official_benchmark_interface_drives_sm86_matmul():
    _require_sm86()
    from sm86_matmul import compile_sm86_matmul

    torch.manual_seed(0)
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    c = torch.empty(512, 512, device="cuda", dtype=torch.float16)
    compiled = compile_sm86_matmul(a, b, c)

    us = testing.benchmark(
        compiled,
        kernel_arguments=testing.JitArguments(a, b, c),
        warmup_iterations=2,
        iterations=5,
    )
    assert us > 0, "official-style benchmark returned no timing"
    torch.testing.assert_close(c.float(), (a.float() @ b.float()),
                               atol=1e-1, rtol=1e-2)


@pytest.mark.sm86
def test_workspace_count_is_sane():
    _require_sm86()
    count = testing.get_workspace_count(4 * 1024 * 1024, 3, 10)
    assert 1 <= count <= 3 + 10


@pytest.mark.sm86
def test_cupti_profiler_is_explicitly_unsupported():
    _require_sm86()
    with pytest.raises(NotImplementedError):
        testing.CuptiProfiler()
