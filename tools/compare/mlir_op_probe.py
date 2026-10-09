"""Probe which MLIR ops lower to `max.f32` and `cvt.rn.f16x2.f32`."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from self_cutedsl.compiler.ptx_pipeline import compile_mlir_to_ptx  # noqa: E402

BODY = r"""
    gpu.func @k(%a : f32, %b : f32, %p : !llvm.ptr) kernel {
      %c1 = llvm.mlir.constant(3.0 : f32) : f32
      %c2 = llvm.mlir.constant(4.0 : f32) : f32
      %m1 = arith.maximumf %a, %b : f32
      %m2 = arith.maxnumf %a, %b : f32
      %m3 = arith.maxf %a, %b : f32
      %m4 = math.maxf %a, %b : f32
      %v0 = llvm.insertelement %a, %v1 : vector<2xf32>
      %v1 = llvm.insertelement %a, %v0[1 : i32] : vector<2xf32>
      %h2 = arith.truncf %v1 : vector<2xf32> to vector<2xf16>
      %s0 = arith.truncf %a : f32 to f16
      llvm.store %m1, %p : f32, !llvm.ptr
      gpu.return
    }
"""


def build(body: str) -> str:
    return ("module attributes {gpu.container_module} {\n"
            "  gpu.module @k {\n" + body + "  }\n}\n")


def main() -> int:
    cases = {
        "broadcast+truncf": r"""
    gpu.func @k(%a : f32, %b : f32, %p : !llvm.ptr) kernel {
      %v = "llvm.mlir.undef"() : () -> vector<2xf32>
      %c0 = llvm.mlir.constant(0 : i32) : i32
      %c1 = llvm.mlir.constant(1 : i32) : i32
      %e0 = llvm.insertelement %a, %v[%c0 : i32] : vector<2xf32>
      %e1 = llvm.insertelement %b, %e0[%c1 : i32] : vector<2xf32>
      %h = arith.truncf %e1 : vector<2xf32> to vector<2xf16>
      %i = llvm.bitcast %h : vector<2xf16> to i32
      llvm.store %i, %p : i32, !llvm.ptr
      gpu.return
    }
""",
        "max-ops": r"""
    gpu.func @k(%a : f32, %b : f32, %p : !llvm.ptr) kernel {
      %m1 = arith.maximumf %a, %b : f32
      %m2 = arith.maxnumf %a, %b : f32
      %m4 = math.maxf %a, %b : f32
      llvm.store %m1, %p : f32, !llvm.ptr
      gpu.return
    }
""",
        "truncf-scalar": r"""
    gpu.func @k(%a : f32, %p : !llvm.ptr) kernel {
      %s = arith.truncf %a : f32 to f16
      llvm.store %s, %p : f16, !llvm.ptr
      gpu.return
    }
""",
    }
    for name, body in cases.items():
        print(f"== {name} ==")
        try:
            ptx = compile_mlir_to_ptx(build(body), target="sm_86")
        except Exception as exc:  # noqa: BLE001
            print("   FAILED:", str(exc)[:300].replace("\n", " | "))
            continue
        for line in ptx.splitlines():
            s = line.strip()
            if s.startswith(("cvt", "max", "selp", "setp", "mov", "shl", "prmt",
                             "and", "st.")):
                print("   ", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
