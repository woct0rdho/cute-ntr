#!/usr/bin/env python3
"""smem_dyn_probe.py - can the open MLIR -> NVPTX pipeline emit dynamic smem?

`sm_86` has a 48 KiB *static* shared-memory limit; anything larger must be
allocated from the dynamic buffer and opted in with cuFuncSetAttribute.  The
official CuTeDSL kernel already does this (its D128 PTX contains
`.extern .shared .align 1024 .b8 __dynamic_shmem__0[]`), so the backend must
support it to make a 64-wide D128 K/V tile legal.

This probe compiles three hand-written MLIR modules through the same
`compile_mlir_to_ptx` path the code generator uses and reports what shared
memory each one declares:

  `static`   an internal addrspace-3 array (what the emitter does today)
  `extern`   an external addrspace-3 array (the candidate dynamic form)
  `memref`   cute.memref.alloca in address space 3 (what the official emits)

Usage::

  python tools/compare/smem_dyn_probe.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from self_cutedsl.compiler.ptx_pipeline import compile_mlir_to_ptx  # noqa: E402

HEADER = "module attributes {gpu.container_module} {\n  gpu.module @k {{\n"
FOOTER = "    gpu.return\n    }\n  }\n}\n"


def static_module() -> str:
    return (
        "module attributes {gpu.container_module} {\n"
        '  gpu.module @k {\n'
        '    llvm.mlir.global internal @"s"() '
        '{addr_space = 3 : i32, alignment = 128 : i64} '
        ': !llvm.array<1024 x f16>\n'
        '    gpu.func @k(%a : !llvm.ptr) kernel {\n'
        '      %0 = llvm.mlir.addressof @"s" : !llvm.ptr<3>\n'
        '      %1 = llvm.mlir.constant(1 : i32) : i32\n'
        '      llvm.store %1, %0 : f16, !llvm.ptr<3>\n'
        '      gpu.return\n'
        '    }\n'
        '  }\n'
        '}\n'
    )


def extern_module(elems: int = 0) -> str:
    return (
        "module attributes {gpu.container_module} {\n"
        '  gpu.module @k {\n'
        '    llvm.mlir.global external @"__dynamic_shmem__0"() '
        f'{{addr_space = 3 : i32, alignment = 1024 : i64}} '
        f': !llvm.array<{elems} x i8>\n'
        '    gpu.func @k(%a : !llvm.ptr) kernel {\n'
        '      %0 = llvm.mlir.addressof @"__dynamic_shmem__0" '
        ': !llvm.ptr<3>\n'
        '      %1 = llvm.mlir.constant(1 : i32) : i32\n'
        '      llvm.store %1, %0 : i32, !llvm.ptr<3>\n'
        '      gpu.return\n'
        '    }\n'
        '  }\n'
        '}\n'
    )


def memref_module() -> str:
    return (
        "module attributes {gpu.container_module} {\n"
        '  gpu.module @k {\n'
        '    gpu.func @k(%a : !llvm.ptr) kernel {\n'
        '      %0 = "cute.memref.alloca"() '
        '{operandSegmentSizes = array<i32: 0, 0, 0>, '
        'alignment = 1024 : i64} '
        ': () -> memref<1024xf16, 3>\n'
        '      %c0 = arith.constant 0 : index\n'
        '      %c1 = arith.constant 1.0 : f16\n'
        '      memref.store %c1, %0[%c0] : memref<1024xf16, 3>\n'
        '      gpu.return\n'
        '    }\n'
        '  }\n'
        '}\n'
    )


def arena_module() -> str:
    """Addressof -> byte GEP -> typed GEP, the way the emitter would."""
    return (
        "module attributes {gpu.container_module} {\n"
        '  gpu.module @k {\n'
        '    llvm.mlir.global external @"__dynamic_shmem__0"() '
        '{addr_space = 3 : i32, alignment = 1024 : i64} '
        ': !llvm.array<0 x i8>\n'
        '    gpu.func @k(%a : !llvm.ptr) kernel {\n'
        '      %0 = llvm.mlir.addressof @"__dynamic_shmem__0" '
        ': !llvm.ptr<3>\n'
        '      %1 = llvm.mlir.constant(16384 : i64) : i64\n'
        '      %2 = llvm.getelementptr %0[%1] '
        ': (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8\n'
        '      %3 = llvm.mlir.constant(5 : i64) : i64\n'
        '      %4 = llvm.getelementptr %2[%3] '
        ': (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16\n'
        '      %5 = llvm.mlir.constant(1.0 : f16) : f16\n'
        '      llvm.store %5, %4 : f16, !llvm.ptr<3>\n'
        '      gpu.return\n'
        '    }\n'
        '  }\n'
        '}\n'
    )


def main() -> int:
    cases = [("static", static_module()),
             ("extern0", extern_module(0)),
             ("extern1024", extern_module(1024)),
             ("memref", memref_module()),
             ("arena", arena_module())]
    for name, mod in cases:
        print(f"== {name} ==")
        try:
            ptx = compile_mlir_to_ptx(mod, target="sm_86")
        except Exception as exc:  # noqa: BLE001
            print(f"   FAILED: {type(exc).__name__}: {str(exc)[:400]}")
            continue
        decls = [ln.strip() for ln in ptx.splitlines()
                 if ".shared" in ln and ("[" in ln or "shared" in ln)]
        for d in decls[:6]:
            print(f"   {d}")
        if not decls:
            print("   (no shared-memory declaration found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
