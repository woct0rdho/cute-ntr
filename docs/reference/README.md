# Reference docs (not checked in)

Large upstream documentation is fetched on demand instead of being committed:

```pwsh
python tools/reference/fetch_ptx_isa.py          # PTX ISA + CUDA C guide
python tools/reference/fetch_ptx_isa.py --ptx    # PTX ISA only
python tools/reference/grep_doc.py ".maxnreg" --lines 8
```

The PTX ISA is the authority for the low-level constraints this project
depends on, and having it locally searchable is worth one command:

* 11.4 performance-tuning directives - `.maxnreg` / `.minnctapersm` "must
  appear between an `.entry` directive and its body";
* `cp.async` group semantics (`commit_group` / `wait_group` counting);
* `ldmatrix` fragment layouts for `.x4` / `.x4.trans`;
* `.target` / `.version` requirements for the driver JIT path.
