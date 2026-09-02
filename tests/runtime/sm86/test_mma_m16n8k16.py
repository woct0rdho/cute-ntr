"""Ampere runtime test for the existing warp MMA fixture."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from self_cutedsl.compiler import compile_mlir_to_ptx  # noqa: E402
from self_cutedsl.runtime import DriverJit, LaunchManifest  # noqa: E402

MLIR_SRC = ROOT / "tests/ptx/mma_ldmatrix.mlir"


@pytest.mark.sm86
def test_mma_m16n8k16_on_sm86():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 6):
        pytest.skip(f"requires sm86, got sm{major}{minor}")

    ptx = compile_mlir_to_ptx(MLIR_SRC, target="sm_86")
    assert ".target sm_86" in ptx
    assert "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" in ptx
    assert "ldmatrix.sync.aligned.m8n8.x4.shared.b16" in ptx
    assert "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16" in ptx

    jit = DriverJit(ptx, target="sm_86")
    manifest = LaunchManifest(
        entry="mma16816",
        args=[{"name": "a", "type": "ptr"},
              {"name": "b", "type": "ptr"},
              {"name": "d", "type": "ptr"}],
        grid=(1, 1, 1),
        block=(32, 1, 1),
        target="sm_86",
    )

    torch.manual_seed(0)
    a = torch.randn(16, 16, device="cuda", dtype=torch.float16)
    b_t = torch.randn(8, 16, device="cuda", dtype=torch.float16)
    b = b_t.t().contiguous()
    d = torch.zeros(16, 8, device="cuda", dtype=torch.float32)

    jit.launch(manifest, a, b, d)
    jit.synchronize()

    expected = torch.matmul(a.float(), b_t.t().float())
    torch.testing.assert_close(d, expected, atol=1e-2, rtol=1e-2)
