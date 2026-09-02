"""Compile-only target-selection and Ampere MMA regressions."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from self_cutedsl.compiler import compile_mlir_to_ptx  # noqa: E402


VECTOR = ROOT / "tests/ptx/vector_add.mlir"
MMA = ROOT / "tests/ptx/mma_ldmatrix.mlir"


def test_sm86_target_is_selectable_for_simple_kernel():
    ptx = compile_mlir_to_ptx(VECTOR, target="sm_86")
    assert ".target sm_86" in ptx
    assert ".target sm_120a" not in ptx
    assert ".entry add" in ptx


def test_sm86_mma_lowers_to_ampere_instructions():
    ptx = compile_mlir_to_ptx(MMA, target="sm_86")
    assert ".version 7.1" in ptx
    assert ".target sm_86" in ptx
    assert "ldmatrix.sync.aligned.m8n8.x4.shared.b16" in ptx
    assert "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16" in ptx
    assert "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" in ptx
    assert ".target sm_120a" not in ptx


def test_sm120_remains_the_default_target():
    ptx = compile_mlir_to_ptx(VECTOR)
    assert ".target sm_120a" in ptx
    assert ".target sm_86" not in ptx
