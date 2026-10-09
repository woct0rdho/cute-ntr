#!/usr/bin/env python3
"""stack_report.py - stack-neutral schema for CuTeDSL intermediate results.

Both the official CuTeDSL and this repository's open stack produce the same
intermediate artifacts for a kernel:

  * textual PTX (the shared interchange format between the two front ends);
  * a CUBIN assembled from that PTX by the *same* `ptxas` binary;
  * optionally SASS disassembled by the *same* `nvdisasm` binary;
  * the `ptxas -v` resource report (registers / spills / smem);
  * a PTX instruction histogram (static code shape);
  * a CUDA-event wall time for one launch.

This module owns the file -> report half of that pipeline so neither stack's
driver has to reimplement it, and so the two reports are directly diffable.

Nothing here imports `cutlass` - it must stay usable from either stack.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# PTX instruction families worth comparing across stacks. The list is
# deliberately explicit: it is the vocabulary of this kernel family
# (Ampere cp.async / ldmatrix / mma.sync + the softmax math).
PTX_PATTERNS = (
    "cp.async",
    "cp.async.commit_group",
    "cp.async.wait_group",
    "ldmatrix",
    "ldmatrix.sync.aligned.x4",
    "ldmatrix.sync.aligned.x2",
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

# SASS opcode families (nvdisasm text). Matches the same kernel vocabulary.
SASS_PATTERNS = (
    "LDGSTS",     # cp.async on Ampere
    "LDSM",       # ldmatrix
    "HMMA",       # tensor core mma
    "LDS",        # shared load
    "STS",        # shared store
    "LDG",        # global load
    "STG",        # global store
    "BAR",
    "SHFL",
    "MUFU",       # ex2/rcp approximations
    "FFMA",
    "F2F",        # f16<->f32 converts
)

_NVCC_VERSION_RE = re.compile(r"release (\d+)\.(\d+)")


def cuda_bin_dir() -> Path | None:
    """Locate the CUDA toolkit bin dir holding ptxas/nvdisasm."""
    for env in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(env)
        if root:
            candidate = Path(root) / "bin"
            if candidate.is_dir():
                return candidate
    which = shutil.which("ptxas")
    if which:
        return Path(which).parent
    return None


def _wheel_tool(name: str) -> str | None:
    """Locate a CUDA binary shipped by an `nvidia-cuda-*` wheel.

    The wheels install into `site-packages/nvidia/<flavour>/bin` (for
    example `nvidia/cu13/bin/ptxas.exe`).  These are preferred over a
    local toolkit because the official CuTeDSL uses them, and because a
    local toolkit can be *older* than the PTX the DSL emits (a 13.3 ptxas
    rejects `.version 9.4` PTX outright).
    """
    import site

    exe = name + (".exe" if os.name == "nt" else "")
    roots = [Path(p) for p in site.getsitepackages()]
    user = site.getusersitepackages()
    if user:
        roots.append(Path(user))
    candidates: list[tuple[tuple[int, ...], Path]] = []
    for root in roots:
        base = root / "nvidia"
        if not base.is_dir():
            continue
        for flavour in base.iterdir():
            if not flavour.name.startswith("cu"):
                continue
            digits = flavour.name[2:].split(".")[0]
            if not digits.isdigit():
                continue
            binary = flavour / "bin" / exe
            if binary.is_file():
                candidates.append(((int(digits),), binary))
    if not candidates:
        return None
    candidates.sort()
    return str(candidates[-1][1])


def _tool(name: str) -> str:
    found = _wheel_tool(name)
    if found:
        return found
    exe = name + (".exe" if os.name == "nt" else "")
    bindir = cuda_bin_dir()
    if bindir is not None:
        candidate = bindir / exe
        if candidate.is_file():
            return str(candidate)
    which = shutil.which(name)
    if which:
        return which
    raise FileNotFoundError(f"{name} not found (CUDA toolkit bin dir missing?)")


def tool_version(tool: str) -> str | None:
    try:
        out = subprocess.run([tool, "--version"], capture_output=True,
                             text=True, check=False).stdout
    except OSError:
        return None
    m = _NVCC_VERSION_RE.search(out or "")
    return f"{m.group(1)}.{m.group(2)}" if m else None


def ptx_histogram(ptx_text: str) -> dict:
    hist = {pat: len(re.findall(re.escape(pat), ptx_text))
            for pat in PTX_PATTERNS}
    hist["__lines__"] = ptx_text.count("\n")
    hist["__regs_declared__"] = len(
        re.findall(r"^\s*\.reg\s+\.f32\s+%f<", ptx_text, re.MULTILINE))
    return hist


def sass_histogram(sass_text: str) -> dict:
    hist = {}
    for pat in SASS_PATTERNS:
        hist[pat] = len(re.findall(r"\b" + re.escape(pat), sass_text))
    hist["__lines__"] = sass_text.count("\n")
    return hist


def parse_ptxas_verbose(log: str) -> dict:
    """Parse `ptxas -v` output into the resource report.

    Observed forms::

        Used 225 registers, used 1 barriers, 33536 bytes smem
        32 bytes stack frame, 48 bytes spill stores, 28 bytes spill loads
        Function properties for <kernel>
    """
    out: dict = {}
    m = re.search(r"Used\s+(\d+)\s+registers?", log)
    if m:
        out["registers"] = int(m.group(1))
    m = re.search(r"used\s+(\d+)\s+barriers?", log)
    if m:
        out["barriers"] = int(m.group(1))
    m = re.search(r"(\d+)\s+bytes\s+smem", log)
    if m:
        out["smem_bytes"] = int(m.group(1))
    m = re.search(r"(\d+)\s+bytes\s+stack frame", log)
    if m:
        out["stack_bytes"] = int(m.group(1))
    m = re.search(r"(\d+)\s+bytes spill stores", log)
    if m:
        out["spill_stores"] = int(m.group(1))
    m = re.search(r"(\d+)\s+bytes spill loads", log)
    if m:
        out["spill_loads"] = int(m.group(1))
    return out


def assemble_and_inspect(
    ptx_text: str,
    target: str,
    out_dir: Path | None = None,
    stem: str = "kernel",
    ptxas_opts: str = "",
    want_sass: bool = False,
    keep_ptx: bool = True,
    keep_cubin: bool = True,
    line_info: bool = True,
) -> dict:
    """Run `ptxas` (and optionally `nvdisasm`) on PTX, like the official DSL.

    Returns the shared report fragment: file paths, ptxas verbose report,
    PTX histogram and (optionally) the SASS histogram.
    """
    report: dict = {"target": target}
    out_dir = Path(out_dir) if out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    ptx_path = None
    if out_dir is not None and keep_ptx:
        ptx_path = out_dir / f"{stem}.{target}.ptx"
        ptx_path.write_text(ptx_text, encoding="utf-8")
        report["ptx_path"] = ptx_path.as_posix()
    report["ptx_histogram"] = ptx_histogram(ptx_text)

    ptxas = _tool("ptxas")
    report["ptxas_path"] = ptxas
    report["ptxas_version"] = tool_version(ptxas)

    work_ptx = ptx_path
    tmp_ptx = None
    if work_ptx is None:
        import tempfile

        fd = tempfile.NamedTemporaryFile(
            "w", suffix=".ptx", delete=False, encoding="utf-8")
        fd.write(ptx_text)
        fd.close()
        tmp_ptx = Path(fd.name)
        work_ptx = tmp_ptx

    cubin_path = None
    if out_dir is not None and keep_cubin:
        cubin_path = out_dir / f"{stem}.{target}.cubin"

    arch = target.replace("sm_", "sm_") if target.startswith("sm_") else target
    cmd = [ptxas, f"-arch={arch}", "-v"]
    if line_info:
        cmd.append("-lineinfo")
    if ptxas_opts:
        cmd.extend(ptxas_opts.split())
    cmd.append(str(work_ptx))
    if cubin_path is not None:
        cmd.extend(["-o", str(cubin_path)])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    report["ptxas_ok"] = proc.returncode == 0
    report["ptxas_log"] = (proc.stdout + proc.stderr).strip()
    report["resources"] = parse_ptxas_verbose(report["ptxas_log"])
    if tmp_ptx is not None:
        tmp_ptx.unlink(missing_ok=True)
    if cubin_path is not None and cubin_path.exists():
        report["cubin_path"] = cubin_path.as_posix()

    if want_sass and cubin_path is not None and cubin_path.exists():
        nvdisasm = _tool("nvdisasm")
        report["nvdisasm_path"] = nvdisasm
        report["nvdisasm_version"] = tool_version(nvdisasm)
        sass_path = cubin_path.with_suffix(".sass")
        proc = subprocess.run(
            [nvdisasm, "-g", "-c", str(cubin_path)],
            capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            sass_path.write_text(proc.stdout, encoding="utf-8")
            report["sass_path"] = sass_path.as_posix()
            report["sass_histogram"] = sass_histogram(proc.stdout)
        else:
            report["nvdisasm_log"] = (proc.stdout + proc.stderr)[:2000]
    return report


def merge(entry: dict, report: dict) -> dict:
    """Merge an assembled report into a case entry (shallow, report wins)."""
    entry = dict(entry)
    entry.update(report)
    return entry


def write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def print_table(rows: list[dict], columns: list[str]) -> None:
    if not rows:
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows))
              for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("  ".join("-" * widths[c] for c in columns))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))


def main() -> int:
    """CLI: inspect an existing PTX dump with the shared pipeline."""
    import argparse

    ap = argparse.ArgumentParser(description="Inspect PTX like the official DSL")
    ap.add_argument("ptx")
    ap.add_argument("--target", default=None,
                    help="override target (default: read from the PTX)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--sass", action="store_true")
    ap.add_argument("--ptxas-options", default="")
    args = ap.parse_args()

    text = Path(args.ptx).read_text(errors="replace")
    target = args.target
    if target is None:
        m = re.search(r"^\.target\s+(\S+)", text, re.MULTILINE)
        target = m.group(1) if m else "sm_86"
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ptx).parent
    stem = Path(args.ptx).stem
    report = assemble_and_inspect(
        text, target, out_dir=out_dir, stem=stem,
        ptxas_opts=args.ptxas_options, want_sass=args.sass)
    report = {k: v for k, v in report.items() if k != "ptxas_log"}
    print(json.dumps(report, indent=1))
    return 0 if report.get("ptxas_ok") else 1


if __name__ == "__main__":
    sys.exit(main())
