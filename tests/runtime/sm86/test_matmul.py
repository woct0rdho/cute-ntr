"""SM86 tiled matmul correctness tests."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))

from .sm86_matmul import launch_sm86_matmul  # noqa: E402


@pytest.mark.sm86
@pytest.mark.parametrize("M,N,K", [(128, 128, 16), (128, 128, 32), (256, 128, 64)])
def test_sm86_tiled_matmul(M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 6):
        pytest.skip(f"requires sm86, got sm{capability[0]}{capability[1]}")

    torch.manual_seed(11 + M + N + K)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    launch_sm86_matmul(a, b, c)
    torch.cuda.synchronize()

    expected = torch.mm(a.float(), b.float())
    torch.testing.assert_close(c, expected, atol=0.25, rtol=0.05)


def test_sm86_tiled_matmul_fp16_output():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 6):
        pytest.skip(f"requires sm86, got sm{capability[0]}{capability[1]}")

    torch.manual_seed(22)
    a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    c = torch.zeros(128, 128, device="cuda", dtype=torch.float16)

    launch_sm86_matmul(a, b, c)
    torch.cuda.synchronize()

    expected = torch.mm(a, b)
    torch.testing.assert_close(c, expected, atol=0.25, rtol=0.05)
