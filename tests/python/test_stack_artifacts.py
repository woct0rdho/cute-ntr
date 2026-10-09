"""Tests for the official-compatible compile-option surface and the shared
intermediate-result reporting used to compare against official CuTeDSL.

These do not need a GPU: they cover option parsing (including Windows paths)
and the PTX/CUBIN/SASS inspection pipeline that both stacks share.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))

from self_cutedsl.compiler.artifacts import (  # noqa: E402
    ArtifactOptions,
    find_tool,
    parse_ptxas_verbose,
    ptx_histogram,
    sass_histogram,
    split_options,
)


def test_option_split_keeps_windows_paths():
    opts = split_options(
        r"--gpu-arch sm86 --keep-ptx --dump-dir C:\cute-ntr\artifacts\x")
    assert opts[-1] == r"C:\cute-ntr\artifacts\x"


def test_option_split_handles_quoted_values():
    opts = split_options('--host-target "llvm -mtriple=aarch64" --keep-ptx')
    assert "llvm -mtriple=aarch64" in opts


def test_artifact_options_flag_and_value_forms():
    inline = ArtifactOptions.from_string(
        r"--keep-ptx --keep-cubin --generate-line-info "
        r"--ptxas-options=-v --dump-dir=C:\d")
    spaced = ArtifactOptions.from_string(
        r"--keep-ptx --keep-cubin --generate-line-info "
        r"--ptxas-options -v --dump-dir C:\d")
    for parsed in (inline, spaced):
        assert parsed.keep_ptx and parsed.keep_cubin
        assert parsed.generate_line_info
        assert parsed.ptxas_options == "-v"
        assert parsed.dump_dir == r"C:\d"
        assert parsed.any()


def test_artifact_options_relays_unknown_flags():
    parsed = ArtifactOptions.from_string("--gpu-arch sm86 --keep-sass")
    assert parsed.keep_sass
    # --gpu-arch is owned by the target parser, not the artifact parser
    assert parsed.unknown == ["--gpu-arch"]


def test_artifact_options_empty():
    assert not ArtifactOptions.from_string(None).any()
    assert not ArtifactOptions.from_string("").any()
    # a bare target selection must not trigger artifact work
    assert not ArtifactOptions.from_string("--gpu-arch sm_86").any()


def test_ptx_histogram_counts_instruction_families():
    ptx = """
.version 7.1
.target sm_86
.visible .entry k(
) {
    cp.async.ca.shared.global [%r1], [%r2], 16;
    cp.async.commit_group;
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%r5];
    ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r1, %r2}, [%r5];
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f1..}, {%r1..}, {%r2..}, {%f1..};
    bar.sync 0;
    ex2.approx.f32 %f1, %f2;
    ret;
}
"""
    hist = ptx_histogram(ptx)
    assert hist["cp.async"] == 2
    assert hist["ldmatrix"] == 2
    assert hist["ldmatrix.sync.aligned.m8n8.x4"] == 1
    assert hist["ldmatrix.sync.aligned.m8n8.x2"] == 1
    assert hist["ldmatrix.sync.aligned.m8n8.x2.trans"] == 1
    assert hist["mma.sync"] == 1
    assert hist["bar.sync"] == 1
    assert hist["ex2.approx"] == 1
    assert hist["__version__"] == "7.1"


def test_sass_histogram_counts_opcode_families():
    sass = """
/*0000*/                   HMMA.16816.F32 R4, R8, R12, R4 ;
/*0010*/                   LDSM.16.MT88.2 {R8, R9}, [R2] ;
/*0020*/                   STG.E.U16 [R2.64], R0 ;
/*0030*/                   STG.E.128 [R2.64], R4 ;
"""
    hist = sass_histogram(sass)
    assert hist["HMMA"] == 1
    assert hist["LDSM"] == 1
    assert hist["STG"] == 2


def test_parse_ptxas_verbose():
    log = ("ptxas info    : Compiling entry function 'k' for 'sm_86'\n"
           "ptxas info    : Function properties for k\n"
           "    32 bytes stack frame, 48 bytes spill stores, 28 bytes spill loads\n"
           "ptxas info    : Used 255 registers, used 1 barriers, 45056 bytes smem, "
           "388 bytes cmem[0]")
    got = parse_ptxas_verbose(log)
    assert got["registers"] == 255
    assert got["barriers"] == 1
    assert got["smem_bytes"] == 45056
    assert got["stack_bytes"] == 32
    assert got["spill_stores"] == 48
    assert got["spill_loads"] == 28


@pytest.mark.sm86
def test_inspect_ptx_produces_cubin_and_sass(tmp_path):
    """The shared pipeline assembles PTX with the pinned ptxas/nvdisasm."""
    ptx = """
.version 7.1
.target sm_86
.address_size 64
.visible .entry sc_probe(
) {
    ret;
}
"""
    from self_cutedsl.compiler.artifacts import inspect_ptx

    report = inspect_ptx(ptx, "sm_86", stem="sc_probe", keep_sass=True,
                         out_dir=tmp_path)
    if not report.get("ptxas_ok"):
        pytest.skip(f"ptxas rejected the probe: {report.get('ptxas_log')}")
    assert report["resources"].get("registers", 0) >= 1
    assert Path(report["cubin_path"]).is_file()
    assert Path(report["sass_path"]).is_file()
    assert "resources" in report


def test_tool_discovery_prefers_wheel_copy():
    ptxas = find_tool("ptxas")
    assert Path(ptxas).name.startswith("ptxas")


def test_entry_directives_land_between_entry_and_body():
    """Performance-tuning directives are per-entry and pre-body.

    PTX ISA 11.4.1/11.4.4: the `.maxnreg` / `.minnctapersm` directives
    "may be applied per-entry and must appear between an `.entry` directive
    and its body".  Placing them inside the body makes ptxas reject the module,
    which is exactly what an earlier revision did.
    """
    from self_cutedsl.compiler.ptx_pipeline import (
        _apply_entry_directives, _apply_maxnreg)

    ptx = (".version 8.7\n.target sm_86\n\n"
           ".visible .entry foo(\n\t.param .u64 a\n)\n{\n"
           "\t.reg .b32 %r<4>;\n\tret;\n}\n")
    out = _apply_entry_directives(ptx, [".maxnreg 128", ".minnctapersm 4"])
    lines = [ln.strip() for ln in out.splitlines()]
    params = lines.index(".param .u64 a")
    brace = lines.index("{")
    assert lines[brace - 1] == ".maxnreg 128, .minnctapersm 4"
    assert params < brace
    assert out.count(".maxnreg") == 1
    # a single directive through the compatibility wrapper
    assert ".maxnreg 64" in _apply_maxnreg(ptx, 64)
    # no directives means byte-identical output
    assert _apply_entry_directives(ptx, []) == ptx


def test_official_testing_interface_surface():
    """The official benchmark interface exists with matching semantics."""
    import inspect

    from cutlass import testing

    sig = inspect.signature(testing.benchmark)
    for name in ("warmup_iterations", "iterations", "stream",
                 "kernel_arguments", "workspace_generator", "workspace_count",
                 "use_cuda_graphs", "use_cupti", "nvtx_tag"):
        assert name in sig.parameters, name

    args = testing.JitArguments(1, 2, stream="s")
    assert args.args == (1, 2) and args.kwargs == {"stream": "s"}
    args.add_to_scope(["keepalive"])
    assert args.references == ["keepalive"]

    # the deprecated cute alias resolves to the same objects
    import cutlass.cute.testing as cute_testing
    assert cute_testing.JitArguments is testing.JitArguments
    assert cute_testing.benchmark is testing.benchmark

    # CUPTI is intentionally not reimplemented: it must fail loudly, not
    # silently return wrong numbers
    with pytest.raises(NotImplementedError):
        testing.CuptiProfiler()
