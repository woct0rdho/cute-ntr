"""artifacts.py - compile-time artifact dumping and inspection.

The official CuTeDSL exposes a documented compile-option surface for
intermediate results:

  `--keep-ptx` / `--dump-ptx-path` / `--keep-cubin` / `--keep-sass`
  `--ptxas-options` / `--generate-line-info` / `--dump-dir`

This module implements the same surface for the open stack.  The open stack
emits PTX directly through its in-tree LLVM NVPTX backend and normally hands
that PTX to the CUDA driver JIT.  For *inspection* we deliberately reuse the
official toolchain - `ptxas` assembles the PTX and reports its resource
usage, `nvdisasm` disassembles the result - so that both stacks produce
directly comparable numbers:

  * registers / spills / barriers / stack from `ptxas -v`;
  * a PTX instruction histogram (static code shape of the front end);
  * a SASS instruction histogram (what the assembler actually emitted);
  * the dumped PTX / CUBIN / SASS file paths.

Nothing here changes the default compile path: artifacts are only produced
when one of the options above is present.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import site
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# PTX instruction families that describe this kernel family's code shape.
PTX_PATTERNS = (
    "cp.async",
    "cp.async.commit_group",
    "cp.async.wait_group",
    "ldmatrix",
    "ldmatrix.sync.aligned.m8n8.x4",
    "ldmatrix.sync.aligned.m8n8.x2",
    "ldmatrix.sync.aligned.m8n8.x4.trans",
    "ldmatrix.sync.aligned.m8n8.x2.trans",
    "mma.sync",
    "bar.sync",
    "barrier.sync",
    "membar",
    "ld.shared",
    "st.shared",
    "ld.global",
    "st.global",
    "shfl.sync",
    "fma.rn.f32",
    "add.f32",
    "mul.f32",
    "ex2.approx",
    "rcp.approx",
    "cvt.rn.f16.f32",
)

# SASS opcode families (nvdisasm text). Same vocabulary, one level down.
SASS_PATTERNS = (
    "LDGSTS",     # cp.async
    "LDSM",       # ldmatrix
    "HMMA",       # tensor core mma
    "LDS",
    "STS",
    "LDG",
    "STG",
    "BAR",
    "SHFL",
    "MUFU",       # ex2 / rcp
    "FFMA",
    "F2F",        # f16 <-> f32 converts
)

_VERSION_RE = re.compile(r"release (\d+)\.(\d+)")

# ---------------------------------------------------------------------------
# option parsing (mirrors the official CLI flags)
# ---------------------------------------------------------------------------

_VALUE_FLAGS = {
    "--dump-ptx-path": "ptx_path",
    "--dump-dir": "dump_dir",
    "--ptxas-options": "ptxas_options",
}

_BOOL_FLAGS = {
    "--keep-ptx": "keep_ptx",
    "--keep-cubin": "keep_cubin",
    "--keep-sass": "keep_sass",
    "--generate-line-info": "generate_line_info",
    "--line-info": "generate_line_info",
}


@dataclass
class ArtifactOptions:
    """Parsed subset of the official compile-option surface."""

    keep_ptx: bool = False
    ptx_path: str | None = None
    keep_cubin: bool = False
    keep_sass: bool = False
    ptxas_options: str = ""
    generate_line_info: bool = False
    dump_dir: str | None = None
    unknown: list[str] = field(default_factory=list)

    def any(self) -> bool:
        return bool(self.keep_ptx or self.ptx_path or self.keep_cubin
                    or self.keep_sass or self.ptxas_options
                    or self.generate_line_info or self.dump_dir)

    @classmethod
    def from_string(cls, options: str | None) -> "ArtifactOptions":
        """Parse the official flags; unknown flags are ignored (the caller
        owns the rest of the option surface, e.g. `--gpu-arch`)."""
        out = cls()
        if not options:
            return out
        for token in split_options(options):
            flag, sep, inline = token.partition("=")
            if flag in _BOOL_FLAGS:
                setattr(out, _BOOL_FLAGS[flag], True)
            elif flag in _VALUE_FLAGS:
                setattr(out, _VALUE_FLAGS[flag], inline if sep else None)
                if not sep:
                    out.unknown.append(flag)   # value arrives as next token
            elif flag.startswith("-"):
                out.unknown.append(flag)
        # second pass: bind "flag value" pairs that were not written as
        # "flag=value" (the value may itself start with '-', e.g. ptxas -v)
        tokens = split_options(options)
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if "=" not in tok and tok in _VALUE_FLAGS \
                    and getattr(out, _VALUE_FLAGS[tok]) is None:
                if i + 1 < len(tokens):
                    setattr(out, _VALUE_FLAGS[tok], tokens[i + 1])
                    if tok in out.unknown:
                        out.unknown.remove(tok)
                    value = tokens[i + 1]
                    if value in out.unknown:
                        out.unknown.remove(value)
                    i += 2
                    continue
            i += 1
        return out


def split_options(options: str) -> list[str]:
    """Split a compile-option string the way the official CLI does.

    `shlex.split` in POSIX mode treats every backslash as an escape, which
    mangles Windows paths (`C:\\dir` becomes `C:dir`).  Use the
    platform's own quoting rules so both `--dump-dir C:\\dir` and
    `--host-target "llvm -mattr=+x"` parse correctly.
    """
    import shlex

    try:
        tokens = shlex.split(options, posix=(os.name != "nt"))
    except ValueError:
        tokens = options.split()
    cleaned = []
    for token in tokens:
        if len(token) > 1 and token[0] == token[-1] and token[0] in "\"'":
            token = token[1:-1]
        cleaned.append(token)
    return cleaned


# ---------------------------------------------------------------------------
# tool discovery
# ---------------------------------------------------------------------------

def _wheel_candidates(name: str) -> list[tuple[tuple[int, ...], Path]]:
    """CUDA binaries shipped by `nvidia-cuda-*` wheels.

    Wheels install into `site-packages/nvidia/<flavour>/bin` (for example
    `nvidia/cu13/bin/ptxas`).  They are preferred over a local toolkit:
    the official CuTeDSL uses them too, and a local toolkit can be *older*
    than the PTX being assembled (a 13.3 ptxas rejects `.version 9.4`).
    """
    exe = name + (".exe" if os.name == "nt" else "")
    roots = [Path(p) for p in site.getsitepackages()]
    user = site.getusersitepackages()
    if user:
        roots.append(Path(user))
    found: list[tuple[tuple[int, ...], Path]] = []
    for root in roots:
        base = root / "nvidia"
        if not base.is_dir():
            continue
        for flavour in sorted(base.iterdir()):
            if not flavour.name.startswith("cu"):
                continue
            digits = flavour.name[2:].split(".")[0]
            if not digits.isdigit():
                continue
            binary = flavour / "bin" / exe
            if binary.is_file():
                found.append(((int(digits),), binary))
    return sorted(found)


def cuda_bin_dir() -> Path | None:
    """Toolkit bin dir (`CUDA_PATH` / `CUDA_HOME` / `nvcc` on PATH)."""
    for env in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(env)
        if root and (Path(root) / "bin").is_dir():
            return Path(root) / "bin"
    which = shutil.which("ptxas")
    return Path(which).parent if which else None


def find_tool(name: str) -> str:
    candidates = _wheel_candidates(name)
    if candidates:
        return str(candidates[-1][1])
    bindir = cuda_bin_dir()
    exe = name + (".exe" if os.name == "nt" else "")
    if bindir is not None and (bindir / exe).is_file():
        return str(bindir / exe)
    which = shutil.which(name)
    if which:
        return which
    raise FileNotFoundError(
        f"{name} not found: install the CUDA toolkit or the matching "
        f"nvidia-cuda-* wheel")


def tool_version(tool: str) -> str | None:
    try:
        out = subprocess.run([tool, "--version"], capture_output=True,
                             text=True, check=False).stdout
    except OSError:
        return None
    m = _VERSION_RE.search(out or "")
    return f"{m.group(1)}.{m.group(2)}" if m else None


# ---------------------------------------------------------------------------
# histograms / resource parsing
# ---------------------------------------------------------------------------

def ptx_histogram(ptx_text: str) -> dict:
    hist = {pat: len(re.findall(re.escape(pat), ptx_text))
            for pat in PTX_PATTERNS}
    hist["__lines__"] = ptx_text.count("\n")
    hist["__version__"] = _ptx_version(ptx_text)
    hist["__regs_declared__"] = len(
        re.findall(r"^\s*\.reg\s+\.f32\s+%f<", ptx_text, re.MULTILINE))
    return hist


def _ptx_version(ptx_text: str) -> str | None:
    m = re.search(r"^\.version\s+(\S+)", ptx_text, re.MULTILINE)
    return m.group(1) if m else None


def sass_histogram(sass_text: str) -> dict:
    hist = {pat: len(re.findall(r"\b" + re.escape(pat), sass_text))
            for pat in SASS_PATTERNS}
    hist["__lines__"] = sass_text.count("\n")
    return hist


def parse_ptxas_verbose(log: str) -> dict:
    """Parse `ptxas -v` output (per-kernel resource usage)."""
    out: dict = {}
    patterns = {
        "registers": r"Used\s+(\d+)\s+registers?",
        "barriers": r"used\s+(\d+)\s+barriers?",
        "smem_bytes": r"(\d+)\s+bytes\s+smem",
        "stack_bytes": r"(\d+)\s+bytes\s+stack frame",
        "spill_stores": r"(\d+)\s+bytes spill stores",
        "spill_loads": r"(\d+)\s+bytes spill loads",
    }
    for key, pat in patterns.items():
        m = re.search(pat, log)
        if m:
            out[key] = int(m.group(1))
    return out


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------

def inspect_ptx(
    ptx_text: str,
    target: str,
    *,
    stem: str = "kernel",
    out_dir: str | Path | None = None,
    ptxas_options: str = "",
    keep_ptx: bool = True,
    keep_cubin: bool = False,
    keep_sass: bool = False,
    line_info: bool = False,
) -> dict:
    """Assemble `ptx_text` with the shared `ptxas` and report.

    `keep_sass` implies `keep_cubin`.  When `out_dir` is None the PTX
    and CUBIN go to a temporary directory that is removed before returning.
    """
    report: dict = {"target": target, "ptx_histogram": ptx_histogram(ptx_text)}
    if keep_sass:
        keep_cubin = True
    tmpdir: tempfile.TemporaryDirectory | None = None
    if out_dir is None:
        tmpdir = tempfile.TemporaryDirectory(prefix="sc_artifacts_")
        out_dir = tmpdir.name
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ptx_path = out / f"{stem}.{target}.ptx" if keep_ptx else None
    if ptx_path is not None:
        ptx_path.write_text(ptx_text, encoding="utf-8")
        report["ptx_path"] = ptx_path.as_posix()

    ptxas = find_tool("ptxas")
    report["ptxas_path"] = ptxas
    report["ptxas_version"] = tool_version(ptxas)

    work_ptx = ptx_path
    if work_ptx is None:
        fd = tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False,
                                         encoding="utf-8")
        fd.write(ptx_text)
        fd.close()
        work_ptx = Path(fd.name)

    cubin_path = out / f"{stem}.{target}.cubin" if keep_cubin else None
    cmd = [ptxas, f"-arch={target}", "-v"]
    if line_info:
        cmd.append("-lineinfo")
    if ptxas_options:
        cmd.extend(ptxas_options.split())
    cmd.append(str(work_ptx))
    if cubin_path is not None:
        cmd.extend(["-o", str(cubin_path)])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    report["ptxas_ok"] = proc.returncode == 0
    report["ptxas_log"] = (proc.stdout + proc.stderr).strip()
    report["resources"] = parse_ptxas_verbose(report["ptxas_log"])
    if ptx_path is None:
        work_ptx.unlink(missing_ok=True)
    if cubin_path is not None and cubin_path.exists():
        report["cubin_path"] = cubin_path.as_posix()

    if keep_sass and cubin_path is not None and cubin_path.exists():
        nvdisasm = find_tool("nvdisasm")
        report["nvdisasm_path"] = nvdisasm
        report["nvdisasm_version"] = tool_version(nvdisasm)
        proc = subprocess.run([nvdisasm, "-g", "-c", str(cubin_path)],
                              capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            sass_path = cubin_path.with_suffix(".sass")
            sass_path.write_text(proc.stdout, encoding="utf-8")
            report["sass_path"] = sass_path.as_posix()
            report["sass_histogram"] = sass_histogram(proc.stdout)
        else:
            report["nvdisasm_log"] = (proc.stdout + proc.stderr)[:2000]
    if tmpdir is not None:
        tmpdir.cleanup()
    return report


def emit_artifacts(ptx_text: str, target: str, options: ArtifactOptions,
                   stem: str = "kernel") -> dict:
    """Entry point used by the front end: honour the parsed options."""
    out_dir = None
    if options.dump_dir:
        out_dir = Path(options.dump_dir)
    elif options.ptx_path:
        # an explicit PTX path implies its directory is the dump directory
        out_dir = Path(options.ptx_path).parent
    report = inspect_ptx(
        ptx_text,
        target,
        stem=stem,
        out_dir=out_dir,
        ptxas_options=options.ptxas_options,
        keep_ptx=bool(options.keep_ptx or options.ptx_path or out_dir),
        keep_cubin=options.keep_cubin,
        keep_sass=options.keep_sass,
        line_info=options.generate_line_info,
    )
    if options.ptx_path:
        # an explicit path overrides the generated name
        Path(options.ptx_path).write_text(ptx_text, encoding="utf-8")
        report["ptx_path"] = Path(options.ptx_path).as_posix()
    return report


def write_report(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1), encoding="utf-8")
