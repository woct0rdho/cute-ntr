"""Focused SM86 FlashAttention forward tests."""
import sys
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))

from .sm86_flash_attention import (  # noqa: E402
    compile_sm86_flash_attention,
    launch_sm86_flash_attention,
)


def _require_sm86():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 6):
        pytest.skip(f"requires sm86, got sm{capability[0]}{capability[1]}")


@pytest.mark.sm86
@pytest.mark.parametrize("heads,dim", [(16, 64), (16, 128),
                                        (32, 64), (32, 128)])
def test_sm86_flash_attention_forward(heads, dim):
    _require_sm86()
    torch.manual_seed(100 + heads + dim)
    q = torch.randn(1, 128, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.empty_like(q)

    launch_sm86_flash_attention(q, k, v, out)
    torch.cuda.synchronize()
    from flash_attn import flash_attn_func

    expected = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.sm86
def test_sm86_flash_attention_requested_length():
    _require_sm86()
    torch.manual_seed(17)
    q = torch.randn(1, 4096, 16, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.empty_like(q)
    launch_sm86_flash_attention(q, k, v, out)
    torch.cuda.synchronize()
    from flash_attn import flash_attn_func

    expected = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.sm86
def test_sm86_flash_attention_ptx_contract(monkeypatch):
    _require_sm86()
    from self_cutedsl.runtime.cuda_driver import DriverJit

    dump_number = getattr(DriverJit, "_dump_n", 0) + 1
    dump_path = Path(tempfile.gettempdir()) / f"dg_mod_{dump_number}.ptx"
    monkeypatch.setenv("DG_DUMP_PTX", "1")
    q = torch.randn(1, 8192, 16, 64, device="cuda", dtype=torch.float16)
    compiled = compile_sm86_flash_attention(q, q, q, torch.empty_like(q))
    compiled()
    torch.cuda.synchronize()

    ptx = dump_path.read_text()
    assert ".target sm_86" in ptx
    assert "cp.async" in ptx
    assert "ldmatrix.sync.aligned" in ptx
    assert "mma.sync.aligned.m16n8k16" in ptx


def test_sm86_flash_attention_rejects_invalid_host_shape():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for tensor metadata")
    q = torch.randn(1, 64, 8, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="unsupported"):
        compile_sm86_flash_attention(q, q, q, torch.empty_like(q))


def test_sm86_flash_attention_rejects_non_fp16():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for tensor metadata")
    q = torch.randn(1, 64, 16, 64, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="FP16"):
        compile_sm86_flash_attention(q, q, q, torch.empty_like(q))


def test_sm86_flash_attention_rejects_mixed_dtype():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for tensor metadata")
    q = torch.randn(1, 64, 16, 64, device="cuda", dtype=torch.float16)
    k = q.float()
    with pytest.raises(ValueError, match="FP16"):
        compile_sm86_flash_attention(q, k, q, torch.empty_like(q))


def test_sm86_flash_attention_rejects_noncontiguous():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for tensor metadata")
    q = torch.randn(1, 64, 16, 64, device="cuda", dtype=torch.float16)
    k = torch.empty_strided(q.shape, (65536, 1024, 1, 16),
                            device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="contiguous"):
        compile_sm86_flash_attention(q, k, q, torch.empty_like(q))
