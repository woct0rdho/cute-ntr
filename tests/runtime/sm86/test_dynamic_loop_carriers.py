"""Dynamic range loop-carrier regressions for the SM86 frontend."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def list_carrier_kernel(out: cute.Tensor, n: cutlass.Int32):
    values = [cute.zero_f32(), cute.zero_f32()]
    for i in cutlass.range_dynamic(n):
        values[0:2] = [values[0] + 1.0, values[1] + 2.0]
    out[0] = values[0]
    out[1] = values[1]


@cute.jit
def launch_list_carrier(out: cute.Tensor, n: cutlass.Int32):
    list_carrier_kernel(out, n).launch(grid=(1, 1, 1), block=(1, 1, 1))


@cute.kernel
def scalar_carrier_kernel(out: cute.Tensor):
    total = 0
    for i in cutlass.range_dynamic(0, 4):
        total += i
    out[0] = total


@cute.jit
def launch_scalar_carrier(out: cute.Tensor):
    scalar_carrier_kernel(out).launch(grid=(1, 1, 1), block=(1, 1, 1))


@cute.kernel
def fragment_carrier_kernel(out: cute.Tensor, n: cutlass.Int32):
    fragment = cute.make_rmem_tensor((2,), cutlass.Float32)
    fragment.fill(0.0)
    for i in range(n):
        fragment[0] = fragment[0] + 1.0
        fragment[1] = fragment[1] + 3.0
    out[0] = fragment[0]
    out[1] = fragment[1]


@cute.jit
def launch_fragment_carrier(out: cute.Tensor, n: cutlass.Int32):
    fragment_carrier_kernel(out, n).launch(
        grid=(1, 1, 1), block=(1, 1, 1))


@cute.kernel
def conditional_list_carrier_kernel(out: cute.Tensor, flag: cutlass.Int32):
    values = [cute.zero_f32()]
    if flag > 0:
        values[0] = values[0] + 3.0
    else:
        values[0] = values[0] + 5.0
    out[0] = values[0]


@cute.jit
def launch_conditional_list_carrier(out: cute.Tensor, flag: cutlass.Int32):
    conditional_list_carrier_kernel(out, flag).launch(
        grid=(1, 1, 1), block=(1, 1, 1))


@pytest.mark.sm86
@pytest.mark.parametrize("n", [1, 5, 17])
def test_dynamic_range_preserves_list_loop_carriers(n):
    cutlass.cuda.initialize_cuda_context()
    out = torch.zeros(2, device="cuda", dtype=torch.float32)
    compiled = cute.compile(
        launch_list_carrier,
        from_dlpack(out),
        n,
        options="--gpu-arch sm86",
    )
    compiled()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out, torch.tensor([float(n), float(2 * n)], device="cuda")
    )


@pytest.mark.sm86
def test_range_dynamic_forces_a_region_for_literal_bounds():
    cutlass.cuda.initialize_cuda_context()
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    compiled = cute.compile(
        launch_scalar_carrier,
        from_dlpack(out),
        options="--gpu-arch sm86",
    )
    compiled()
    torch.cuda.synchronize()
    assert out.item() == 6


@pytest.mark.sm86
def test_dynamic_range_preserves_fragment_subscript_writes():
    cutlass.cuda.initialize_cuda_context()
    out = torch.zeros(2, device="cuda", dtype=torch.float32)
    compiled = cute.compile(
        launch_fragment_carrier,
        from_dlpack(out),
        7,
        options="--gpu-arch sm86",
    )
    compiled()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out, torch.tensor([7.0, 21.0], device="cuda")
    )


@pytest.mark.sm86
@pytest.mark.parametrize("flag, expected", [(1, 3.0), (-1, 5.0)])
def test_dynamic_if_preserves_list_loop_carriers(flag, expected):
    cutlass.cuda.initialize_cuda_context()
    out = torch.zeros(1, device="cuda", dtype=torch.float32)
    compiled = cute.compile(
        launch_conditional_list_carrier,
        from_dlpack(out),
        flag,
        options="--gpu-arch sm86",
    )
    compiled()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out, torch.tensor([expected], device="cuda")
    )
