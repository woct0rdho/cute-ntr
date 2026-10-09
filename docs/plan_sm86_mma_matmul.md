# Plan: Idiomatic CuTe DSL and SM86 Matmul Refactor

Status: execution complete; final fulfillment review recorded below.

## 1. Goal and non-negotiable constraints

Refactor the compatibility CuTe DSL package and the validated SM86 FP16 GEMM
so that the kernel is written in the original CuTe programming model:

- host code builds layouts, tensors, tiled copies, and tiled MMA objects;
- the device kernel obtains CTA tiles with `cute.local_tile`;
- thread ownership comes from `get_slice(tidx)` and `partition_*`;
- global/shared and shared/register movement uses `cute.copy`;
- MMA uses partitioned fragments and `cute.gemm`;
- architecture-specific instruction emission remains behind the compatibility
  backend, not in the application kernel.

The package must be compatible with the original CuTe DSL for the supported
surface, including accepted call shapes, object relationships, layout shape /
stride behavior, slicing, and expected failure modes. Compatibility means that
a source program using the documented subset can be moved to the original
package without changing its CuTe expressions. It does not claim to implement
unrelated full CUTLASS features in this milestone.

No correctness or performance regression is acceptable. A source-model change
may be retained as an optimization only when it passes the existing numerical
suite and repeated performance measurements. Existing SM120 behavior remains
the default and must stay green.

## 2. Compatibility surface to implement and audit

Implement and test the following original-CuTe concepts:

1. `cute.make_layout`, layout shape/stride, `cute.size`, `cute.cosize`,
   `cute.rank`, `cute.slice_`, `cute.flatten`, `cute.group_modes`,
   `cute.logical_divide`, and `cute.zipped_divide`.
2. `cute.make_tensor` and tensor iterator/layout rebinding, including device
   tensor metadata and `cute.local_tile` for 2-D and 3-D views.
3. `cute.make_copy_atom`, `cute.make_tiled_copy`,
   `cute.make_tiled_copy_tv`, `get_slice`, `partition_S`, `partition_D`, and
   `cute.copy` for universal, cp.async, and SM86 ldmatrix paths.
4. `cute.make_tiled_mma`, `get_slice`, `partition_A/B/C`,
   `make_fragment_A/B/C`, fragment slicing, and `cute.gemm` for the
   `m16n8k16` FP16/BF16 atom.
5. Typed shared-memory objects with layout views and explicit alignment, while
   preserving the old low-level allocation entry points as compatibility
   aliases.
6. Explicit target capability checks for SM80+/Ampere operations. A custom
   backend implementation may exist, but it must be reached through an
   official-looking CuTe operation rather than a kernel-only raw-offset helper.

The compatibility layer will retain architecture-private implementation names
for unavoidable backend details, but those names must not be needed in the
refactored GEMM source.

## 3. Staged implementation

### Stage A - establish the source-model contract

- Add a small compatibility specification and tests for layout/tensor algebra.
- Normalize layout representation so nested and flat layouts preserve both
  shape and stride rather than silently recomputing row-major strides.
- Make `make_tensor(iterator, layout)` preserve the iterator's base, element,
  address space, and view strides.
- Make `local_tile` return a first-class tile view carrying tile geometry,
  rest coordinates, operand identity, and dynamic CTA origins.
- Preserve existing TMA tests while sharing the same view machinery.

### Stage B - official-style tiled-copy backend

- Add a generic tiled-copy descriptor with atom, thread layout, value layout,
  tile shape, and bound thread slice.
- Use partitioned source/destination views as the only application-facing
  representation of copies.
- Implement the SM86 G2S copy descriptor by lowering its partitioned views to
  the already validated aligned 16-byte `cp.async.ca.shared.global` emitter.
- Implement the SM86 S2R descriptor through the existing trait-calibrated
  `ldmatrix` backend.
- Keep `cute.copy` as the dispatch point; remove direct application calls to
  `cp_async_copy` and `ldmatrix` from the GEMM.
- Preserve the existing elementwise and TMA copy paths.

### Stage C - official-style tiled-MMA backend

- Make `MmaF16BF16Op` carry explicit operand, accumulator, and instruction
  shape metadata with validation for `(16, 8, 16)` on SM86.
- Make `TiledMma` derive warp tile shape, thread layout, and fragment shapes
  from the atom layout.
- Implement `make_fragment_A/B/C` and fragment slicing so the application only
  selects `(None, None, k)` views.
- Route `cute.gemm` through the tiled-MMA fragment operation while retaining
  the existing `mma.sync` lowering and register ordering.
- Add conformance tests for atom layout `(1,1,1)`, `(2,2,1)`, and `(2,4,1)`.

### Stage D - rewrite the SM86 kernel

- Build the CTA tile, shared layouts, copy atoms, tiled copies, and tiled MMA
  in the `@cute.jit` host function.
- Pass these objects as constexpr kernel parameters.
- In the kernel, derive rasterized CTA coordinates, call `local_tile`, bind
  thread slices, partition A/B/C, allocate fragments, and issue `cute.copy`
  and `cute.gemm` only.
- Keep the current 64x64x32 and 64x128x32 dispatch choices, two-stage pipeline,
  padding, register prefetch order, and rasterization factor initially.
- Preserve FP32 and FP16 output contracts, including the existing scalar
  epilogue if it remains faster on the measured device.
- Do not add edge masking until the layout-driven divisible path is green;
  then use identity-tensor predicates through the same copy API.

### Stage E - compatibility hardening and performance

- Compare public signatures and representative object behavior against the
  original package sources for the supported surface.
- Add negative tests for invalid ranks, incompatible atom layouts, unsupported
  architectures, and malformed copy partitions.
- Inspect generated PTX and resource usage before and after the refactor.
- Re-run the required benchmark capture on the same GPU, stream, and baseline
  settings. Tune only source-model/backend changes that preserve the CuTe
  partition contract.
- Consider optimizations enabled by the new model: hoisted CTA origins,
  layout-derived vector copies, reduced redundant arithmetic, better register
  prefetch ordering, and safe copy/MMA pipelining.

## 4. Test gates

The following gates must remain green:

- `python -m py_compile` for all changed Python files;
- `git diff --check`;
- layout/tensor/copy/MMA compatibility tests without a GPU;
- existing non-SM86 host and compile suites;
- SM86 target PTX audit (`.target sm_86`, `ldmatrix`, and `mma.sync`);
- SM86 runtime tests on compute capability 8.6;
- FP32 and FP16 output numerical comparisons;
- no SM120 regression on the existing compatible test path.

The required SM86 performance set must be measured with CUDA events after
warmup, with compile/JIT time reported separately. Keep a before/after table
for 2048, 4096, and 8192 square cases plus representative rectangular cases.
The implementation may be accepted only if the refactor does not regress the
current validated measurements; improvements should be retained and recorded.

## 5. Deliverables

- compatibility implementation and focused tests;
- idiomatic `tests/runtime/sm86/sm86_matmul.py` with no direct backend helper
  calls for G2S, S2R, or MMA;
- updated benchmark/performance artifacts when a new capture is actually run;
- this plan updated with completed stages, test commands, measurements, and
  limitations;
- a final fulfillment review as the last work step, explicitly checking each
  goal and marking it fulfilled, partially fulfilled, or blocked.

## 6. Final fulfillment review

This review was run last after cleanup, validation, PTX inspection, and fresh
benchmarks.

### Source-model and kernel review

- Fulfilled - idiomatic SM86 application source. The host JIT constructs
  layouts, tensors, copy atoms, tiled copies, and tiled MMA objects. The kernel
  uses `cute.local_tile`, `get_slice`, `partition_A/B/C`,
  `make_fragment_A/B/C`, `cute.copy`, and `cute.gemm`.
- Fulfilled - application/backend separation. The final source audit found
  no calls to `cp_async_copy`, `ldmatrix`, `mma.sync`, or manual lane-to-register
  mapping in `tests/runtime/sm86/sm86_matmul.py`. Ampere instruction lowering
  and the calibrated lane/register traits remain in the compatibility backend.
- Fulfilled - correctness gates. FP32 and FP16 output paths pass the SM86
  numerical tests, including K=16, K=32, and K=64. The K=64 staged path now
  uses the physical shared-memory stage stride, including row padding.
- Partially fulfilled - pipeline abstraction. The two-stage pipeline and
  validated `cp.async` schedule remain correct, but commit/wait operations are
  still explicit `cute.cp_async_*` calls rather than a separate structured
  pipeline object.

### Compatibility review

- Fulfilled for the supported surface - layout/tensor algebra. Shape and
  stride preservation, `cosize`, slicing, rebinding, staged views, and dynamic
  `local_tile` origins are covered by the existing algebra tests and the new
  `tests/python/test_cute_compat_surface.py` checks.
- Fulfilled for the supported surface - copy/MMA relationships. Universal
  and cp.async atom construction, cache-mode spelling, tiled-copy slices,
  `retile`, MMA tiling, fragment allocation, and C epilogue partitioning are
  represented by compatibility objects and focused tests.
- Partially fulfilled - exact original-package compatibility. This remains
  a targeted compatibility implementation, not the complete CUTLASS CuTe API.
  Unsupported or incomplete areas include broad TMA/bulk-copy variants,
  complete architecture-specific atom trait validation, edge-masked SM86
  copies, and every original failure-mode/signature combination. These are
  intentionally outside this milestone's supported surface.
- Partially fulfilled - conformance breadth. The supported examples and
  representative object relationships are covered, but exhaustive differential
  testing against every original overload was not added.

### Validation evidence

- `python -m py_compile` on all changed Python files: passed.
- `git diff --check`: passed.
- `pytest tests/python/test_sm86_target.py
  tests/python/test_cute_compat_surface.py -q`: 6 passed.
- `pytest -m "not sm120 and not sm86" -q`: 192 passed, 87 deselected.
- `pytest -m sm86 -q -s`: 11 passed, 268 deselected on the RTX 3080 Ti
  Laptop (compute capability 8.6).
- Final PTX audit over three generated SM86 modules: all contained `.target
  sm_86`, `cp.async.ca.shared.global`, `ldmatrix`, and
  `mma.sync.aligned.m16n8k16`.
- The SM120 TMA/object-model tests were not counted as regressions: on this
  SM86 machine they attempt unsupported SM120 TMA paths and fail with the
  expected `CUDA_ERROR_NOT_SUPPORTED` environment mismatch.

### Performance evidence

The fresh square capture is in
`artifacts/perf/sm86_matmul_refactored_noisy.json`; the retained comparison is
`artifacts/perf/sm86_matmul_large_noisy_v3_dispatch.json`. Values below are
median custom time in microseconds, followed by custom/torch time ratio:

| Shape | Prior noisy capture | Refactored noisy capture | Assessment |
|---|---:|---:|---|
| 2048^3 | 600.06 us / 1.0635 | 599.04 us / 1.0617 | comparable; noisy outlier |
| 4096^3 | 4609.02 us / 1.0879 | 4579.33 us / 1.0811 | comparable/slightly better |
| 8192^3 | 28155.90 us / 1.0164 | 28029.95 us / 1.0093 | comparable/slightly better |

Representative rectangular results are in
`artifacts/perf/sm86_matmul_refactored_rect_noisy.json`:
`1024x4096x1024` measured 334.85 us versus 336.90 us for torch, while
`4096x1024x1024` measured 2005.50 us versus 1898.50 us. The latter capture
ran at a sustained 210 MHz SM clock, and the former included a later
low-clock outlier; neither is a controlled before/after comparison. Therefore
performance is partially fulfilled: the like-for-like square capture shows
no measured regression, but parity for all rectangular cases is not claimed.

### Final classification

- Fulfilled: idiomatic source constructs, backend separation, SM86
  correctness, target selection, PTX instruction generation, and the tested
  supported compatibility surface.
- Partially fulfilled: complete original CuTe API coverage, structured
  pipeline abstraction, exhaustive conformance/failure-mode coverage, and
  performance parity under all rectangular/clock conditions.
- Blocked by scope/environment: exact full-package compatibility and
  SM120 device validation on the available SM86 GPU.
