"""Small host-side conformance checks for the supported CuTe surface."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))

import cutlass  # noqa: E402
from cutlass import cute  # noqa: E402


def test_layout_shape_stride_and_cosize_are_preserved():
    layout = cute.make_layout((4, 8), stride=(16, 1))

    assert layout.shape == (4, 8)
    assert layout.stride == (16, 1)
    assert cute.size(layout) == 32
    assert cute.cosize(layout) == 56
    assert cute.rank(layout) == 2
    assert cute.slice_(layout, (None, 2)).stride == (16,)


def test_original_ampere_operation_construction_forms_are_accepted():
    load_mode = cute.nvgpu.LoadCacheMode.GLOBAL
    assert cute.nvgpu.cpasync.LoadCacheMode.GLOBAL.value == load_mode.value

    async_op = cute.nvgpu.cpasync.CopyG2SOp(cache_mode=load_mode)
    async_atom = cute.make_copy_atom(
        async_op, cutlass.Float16, num_bits_per_copy=128)
    assert async_atom.bits == 128
    assert async_op.cache_mode == load_mode

    universal_op = cute.nvgpu.CopyUniversalOp()
    universal_atom = cute.make_copy_atom(
        universal_op, cutlass.Float16, num_bits_per_copy=128)
    assert universal_atom.bits == 128

    mma_op = cute.nvgpu.warp.MmaF16BF16Op(
        cutlass.Float16, cutlass.Float32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(
        mma_op, cute.make_layout((2, 4, 1)))
    assert tiled_mma.shape_mnk == (16, 8, 16)
    assert tiled_mma.tile_mn == (32, 32)
    assert tiled_mma.get_slice(0).shape_mnk == (16, 8, 16)


def test_tiled_copy_tv_keeps_public_partition_relationships():
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Float16,
        num_bits_per_copy=128)
    tiled = cute.make_tiled_copy_tv(
        atom, cute.make_layout((32, 1)), cute.make_layout((1, 8)))
    thread_slice = tiled.get_slice(0)

    assert thread_slice.tidx == 0
    assert thread_slice.partition_S is not None
    assert thread_slice.partition_D is not None
    assert tiled.retile([1, 2]) == [1, 2]
