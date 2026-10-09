# Plan: Focused SM86 FlashAttention Forward

Status: focused milestone complete and measured on SM86.

> Reference baseline changed. Official CuTeDSL 4.8.0 is now installed and
> runs on Windows, so the target is no longer "beat `flash_attn` by luck" but
> "match or beat the official CuTe Ampere FA2 kernel measured on this device".
> The comparison harness, the noise-robust measurement protocol, and the first
> static diff are in `docs/PLAN_official_comparison.md`. Measured status:
> D64 FA now matches the official kernel (1.02x--1.04x, statistically
> equivalent across all lengths and head counts) and matches or beats
> installed `flash_attn` (0.85x--1.02x); D128 FA is 1.09x--1.19x behind.
> The SM86 matmul is within 6 % of the official example.

## 1. Goal and scope

Implement a fixed-layout FP16 FlashAttention forward kernel using the
repository's CuTe-compatible source model and the existing SM86 instruction
lowering. The first supported contract is:

- target: NVIDIA SM86;
- inputs and output: FP16;
- accumulation and online-softmax state: FP32;
- dense, non-varlen tensors in `(batch, seqlen, heads, head_dim)` layout;
- `num_heads`: 16 or 32;
- `head_dim`: 64 or 128;
- square sequence lengths: 4096, 8192, or 16384;
- one head for Q/K/V at each logical head index (no GQA packing);
- inference/forward only, no dropout, no backward, no split-KV, no paging;
- causal and local masking are deferred until the dense unmasked path is
  correct and measured.

The implementation must preserve existing SM120 defaults and existing SM86
matmul behavior. Official CuTe-compatible names remain the application API;
backend-specific instruction emission stays in the compatibility layer.

## 2. Baseline and acceptance evidence

Correctness and CUDA-event timings are captured against installed
`flash_attn==2.9.1` for every requested batch-1 shape. The benchmark records
warmup, measured iterations, dtype, GPU state, compile/first-launch time, and
peak allocation in `artifacts/perf/sm86_flash_attention.json`.

Acceptance requires:

- source and host argument validation rejects unsupported dtype/layout/shape;
- numerical output agrees with a PyTorch FP32 reference and installed
  `flash_attn` within a documented FP16 tolerance;
- unsupported dimensions, dtypes, and sequence lengths not divisible by the
  fixed tile are rejected at the host boundary;
- generated PTX targets `.target sm_86` and contains `cp.async`, `ldmatrix`,
  and `mma.sync.aligned.m16n8k16` for the tiled path;
- focused attention tests, existing SM86 tests, and the non-SM120/non-SM86
  suite remain green;
- performance is measured end to end, excluding compile/JIT time, and no
  parity claim is made without results on the actual SM86 device.

## 3. Primitive and compatibility work

Implement only the APIs required by the fixed path, preserving original CuTe
call shapes and object relationships:

1. Normalize element-type comparisons for `Float16`/`Float32` metadata and add
   the missing finite/infinite FP32 constants used by softmax.
2. Add or complete layout operations required by the kernel: `select`,
   `domain_offset`, `recast_layout`, and the rank-preserving tensor views used
   for `(M, K)`, `(N, K)`, and `(K, N)` operands.
3. Extend tiled-copy predicates for Q/K/V boundary loads without changing the
   validated aligned SM86 divisible-copy path.
4. Generalize the existing m16n8k16 fragment backend for both QK and P@V,
   including FP32 C fragments, FP16 register score fragments, and C epilogue
   stores.
5. Add focused fragment arithmetic/reduction helpers for row-wise max, sum,
   exponentiation, scaling, and FP32-to-FP16 conversion. Use warp-level
   reductions with the existing SM86 shuffle backend where the fragment layout
   requires cross-lane state.
6. Add only the debug/inspection helpers needed for generated PTX and focused
   attention tests. The full official testing, benchmark, and autotune stacks
   remain out of scope.

Each addition gets a narrow host or runtime test before it is used by the
attention kernel. Existing backend paths must not be rewritten solely for API
completeness.

## 4. Kernel structure

Start with a persistent CTA per `(batch, head, Q tile)` and iterate over K/V
tiles. The device algorithm is:

1. Load a Q tile into shared memory using the existing SM86 `cp.async` path.
2. For each K/V tile, issue the V0 `cp.async` transfer and stage K with
   aligned `vector<8xf16>` global loads plus transposed shared stores because
   `cp.async` cannot transpose its 16-byte payload. D64 uses the Ampere
   128-bit `S<3,4,3>` shared-memory swizzle for Q, K, and V, and applies the
   same swizzle to the FP16 probability tile. The compatibility backend uses
   row/column-specialized address arithmetic for these 64-wide rows. After
   the K visibility barrier, D128 issues V1 into the Q-reclaimed shared
   storage while the CTA performs K register loads, Q@K^T, softmax,
   probability conversion, and PV0. V1 is waited on immediately before PV1.
   Retain FP32 score accumulators.
3. Apply the fixed scale and online softmax update. For each row maintain
   running maximum `m`, running denominator `l`, and output accumulator `O`.
   Rescale the previous `O` by `exp2((m_old - m_new) * scale_log2)` before
   accumulating the current `P@V` result.
4. Convert the normalized probability fragment to FP16 registers and issue
   P@V using the same MMA atom, with V presented in the transposed shared
   layout required by `ldmatrix.trans`.
5. Normalize the final O by `1/l`, convert to FP16, and store through the
   partitioned C epilogue. Optional LSE output is deferred unless needed by
   the correctness harness.

The implementation retains the validated one-stage K/V schedule with V
prefetch overlap. It does not yet have a true multi-stage K pipeline: K
transpose staging, CTA barriers, and the P shared-memory round trip remain
known bottlenecks. D64 Q/K/V/P swizzling substantially reduces measured
runtime, but does not by itself eliminate every shared-store conflict. Future
changes must preserve the long-sequence barrier requirements and be measured
against the canonical artifact.

### 4.1 Structural differences from the official Ampere kernel

Comparing the compiled SASS of this kernel against the official CuTeDSL
Ampere FA2 example on the same shape (S4096, H16, D64, 64x64 tile, 128
threads) isolates three differences; see `docs/PLAN_official_comparison.md`
for the full table and the measurement protocol.

- Epilogue store width. The MMA-to-global copy emits `STG.E.U16` (32
  scalar 2-byte stores) where the official stages `O` through shared memory
  and emits `STG.E.128` (16-byte stores). Nsight Compute independently
  estimates ~41 % of runtime lost to under-utilised global-store sectors.
- Probability round trip. `P` is written to shared memory and re-read
  with `ldmatrix` only to change fragment layout; the official reinterprets
  the score fragment in registers (`rP_mma_view`) and feeds `P@V` directly,
  removing 32 shared stores, the matching loads, and two barriers.
- K staging. This kernel keeps `sK` as `(K, N)` and transposes during
  staging (vector global load + scalar shared stores, synchronous); the
  official keeps `(N, K)` and fills it with `cp.async`, reading it with
  non-transposed `ldmatrix.x4`.

These three items replace the previous "explore direct register probability
conversion" and "improve K staging" bullets and are now specified against a
working reference implementation rather than a hypothesis.

## 5. Tile and memory choices

Compile-time dispatch for the four head configurations rather than
autotuning. The delivered dispatch uses separate fixed policies:

- D64: `tile_m=64`, `tile_n=64`, 4x1 warp grid (128 threads), 24.5 KiB of
  static shared memory, 3-4 resident CTAs per SM;
- D128: `tile_m=64`, `tile_n=64`, 4x1 warp grid (128 threads), 48 KiB of
  static shared memory, 2 resident CTAs per SM.

Both put every N column of a row slice in one warp, so the online softmax
needs no cross-warp reduction and the probability fragment feeds P@V straight
from registers.

Tile policy history. D128 originally ran `tile_m=128, tile_n=32`: at that
M extent a 64-wide K/V tile needs 64 KiB, above the 48 KiB *static* shared
limit. That tile turned out to be the wrong trade. A 128x64 tile costs 64 KiB
and therefore one resident CTA per SM, and with only four warps resident the
kernel runs at 0.10 issued warps per scheduler - the exposed K/V latency and
the softmax arithmetic are both unhidden (measured 5.8 ms at S4096 H16 D128
versus 4.7 ms for the current policy), and 128x32 measured 12 % slower than
64x64 once every other fix was in. Moving to `tile_m=64, tile_n=64` halves
the shared footprint, doubles residency to eight warps per SM, and wins
despite doubling the per-M-tile K/V read traffic; L2 throughput is only ~25 %
in this regime, so the extra traffic is free. This is also where the kernel
passed the official kernel.

For reference, the surrounding implementations choose differently on this
device and are all slower: FA2 dispatches to `kBlockM=128, kBlockN=32` for
head_dim 128 non-causal ("48 KB smem ... 2 CTAs per SM") and FA4 to
`128x64` with `Q_in_regs=False`; FA3 - the fastest of the three here - uses a
CUTLASS-style kernel at CTA tile `128x128` with 256 threads and 8 warps,
which this kernel shape could not reproduce (64x128 and 128x128 measured
19 % and 20 % slower than the delivered 64x64 policy).

K prefetch. Each iteration used to issue its K and V loads and then
immediately wait for K, exposing the full global-memory latency once per
iteration. K for tile `kt+1` is now issued in the same buffer right after the
QK gemm has consumed tile `kt` (separated by a barrier), so it overlaps the
softmax and P@V work. This is the structure FA2's C++ kernel uses -
`load_K(n_block - 1)` after the QK gemm and `load_V(n_block)` before it - and
it is what closed the D128 gap: 0.977x -> 0.931x against FA4 at S4096 H16
D128, and 1.014x -> 1.000x against the official kernel at S4096 H16 D64.

Two related capabilities are implemented and tested but not on the current
dispatch path, because no shape needs them:

- dynamic shared memory (`make_smem_tile(..., dynamic=True)`), which allocates
  from an external `__dynamic_shmem__0` arena - the same declaration the
  official CuTeDSL emits - and opts in with `cuFuncSetAttribute` at launch.
  It removes the 48 KiB ceiling for a future 128-wide D128 tile or a
  multi-stage pipeline. Covered by `tests/runtime/sm86/test_dynamic_smem.py`.
- a per-configuration V `ldmatrix` width. `x4.trans` covers two N atoms per
  instruction and wins at D128, but measures 5 % slower at D64, where the
  16-byte-per-lane request latency is not hidden by a second resident CTA.

The output/V tile remains 64 columns. D128 uses two FP32 output accumulators
that share one QK and online-softmax pass, avoiding a second score computation
for the second V/O slice.

## 5.1 Bank conflicts were the largest single gap

Every shared tile that `ldmatrix` reads must be swizzled, and the D128 path
originally swizzled only K. `ldmatrix` reads 16 bytes per lane from the *row*
the lane names, so in a plain row-major tile all eight rows of an 8x8 matrix
start at bank 0 and the load runs at 8-way bank conflict. Measured with
Nsight Compute on D128 S4096 H16:

| metric | unswizzled | swizzled | FA4 SM80 |
| --- | ---: | ---: | ---: |
| shared-load wavefronts | 219.2 M | 42.1 M | 42.1 M |
| shared-load bank conflicts | 177.1 M | 0 | 0 |
| L1/TEX throughput | 54 % | 16 % | 17 % |
| D128 time (S4096 H16) | 6.78 ms | 5.80 ms | 4.80 ms |

Swizzling sQ, sV, sV1 and sO removed every conflict and cut shared-load
wavefronts by 5.2x, to exactly FA4's count for the same work.

Instruction-level follow-ups (differential vs the FA4 SM80 kernel, same
shape and tile, D128): `fmax` as `arith.cmpf`+`arith.select` became
`arith.maxnumf` (`max.f32`, one instruction instead of two), and the
probability conversion became a single `cvt.rn.f16x2.f32`
(`arith.truncf` on a `vector<2xf32>`) emitted directly into packed half2 slots
instead of two scalar converts plus a register pack.

## 6. Tests and tooling

Add a focused SM86 test module and a temporary benchmark script. Tests cover:

- all four `(heads, head_dim)` combinations at a reduced sequence length;
- requested lengths where memory permits;
- random and deterministic inputs;
- FP32 reference attention and installed `flash_attn` comparison;
- unsupported shape/dtype/layout failures;
- PTX instruction and target checks;
- rejection of unsupported, non-tiled shapes.

The benchmark reports compile/JIT time separately from CUDA-event execution
time and compares the custom kernel with installed `flash_attn` using identical
inputs, stream, warmup, and iteration settings. It records failures caused by
device memory or unsupported launch attributes instead of treating them as
performance results.

## 7. Exit criteria and limitations

The fixed dense FP16 milestone is numerically sound, PTX-audited, tested on
SM86, and benchmarked end to end. Causal masking, variable lengths, GQA,
dropout, backward, split-KV, paging, full official utility compatibility, and
autotuning remain explicitly unsupported.

## 8. Results and fulfillment review

Validated environment:

- GPU: NVIDIA GeForce RTX 3080 Ti Laptop GPU, compute capability 8.6;
- PyTorch: `2.13.0+cu132`;
- reference: installed `flash_attn` (the FA2 C++/CUDA kernel);
- benchmark: batch 1, FP16, non-causal, 3 warmups, 10 measured iterations.

Median execution times from the final artifact are:

| S | H | D | ours (us) | flash_attn (us) | ratio | max abs error |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 16 | 64 | 2267 | 2332 | 0.972 | 1.22e-04 |
| 4096 | 16 | 128 | 4508 | 4491 | 1.004 | 1.22e-04 |
| 4096 | 32 | 64 | 3640 | 3850 | 0.946 | 1.22e-04 |
| 4096 | 32 | 128 | 7525 | 6820 | 1.103 | 1.22e-04 |
| 8192 | 16 | 64 | 7541 | 7618 | 0.990 | 6.10e-05 |
| 8192 | 16 | 128 | 14453 | 13364 | 1.081 | 6.10e-05 |
| 8192 | 32 | 64 | 14668 | 14712 | 0.997 | 1.22e-04 |
| 8192 | 32 | 128 | 28370 | 26364 | 1.076 | 6.10e-05 |
| 16384 | 16 | 64 | 28890 | 29529 | 0.978 | 6.10e-05 |
| 16384 | 16 | 128 | 56390 | 52805 | 1.068 | 6.10e-05 |
| 16384 | 32 | 64 | 57589 | 59885 | 0.962 | 6.10e-05 |
| 16384 | 32 | 128 | 113246 | 105479 | 1.074 | 6.10e-05 |

This table is the *noisy* sequential protocol and installed FA2 is not the
fastest reference; the interleaved paired numbers against the official
CuTeDSL kernel, FA4, and FA3 are in `docs/PLAN_official_comparison.md`. In
short: D64 beats FA2 and FA4 everywhere, D128 beats FA4 and is within 0.4--10 %
of FA2 (which is ~7 % faster than FA4 there), and against FA3 - the fastest
reference on this device - 10 of 12 shapes are at parity (0.985x--1.025x) and
the two largest-head-count cases are 5--6 % behind.

The largest case used approximately 896 MiB of peak allocated CUDA memory.
Timing excludes compilation; compile plus first launch ranged from about
0.13 to 0.29 seconds in this run. GPU clocks can vary on the laptop device,
so the JSON artifact includes the sampled clock, power, temperature, and
percentile data.

Exact validation commands:

```pwsh
pytest tests/runtime/sm86/test_flash_attention.py -q -s
pytest tests/runtime/sm86/test_dynamic_smem.py -q -s
pytest -m sm86 -q -s
pytest -m "not sm120 and not sm86" -q
python tools/perf/bench_sm86_flash_attention.py --warmup 3 --iters 10

# reference comparison (see docs/PLAN_official_comparison.md)
python tools/compare/fa_generations.py          # which FA runs here, how fast
python tools/compare/paired_bench.py --kernel fa --seqlen 4096 --heads 16 `
    --head-dim 64 --rounds 6 --batch-size 2
python tools/compare/sweep.py --reference fa3 --rounds 5 --batch-size 2
python tools/compare/compare_stacks.py --kernel fa
```

The PTX contract test compiles a fresh specialization with `DG_DUMP_PTX=1`
and asserts `.target sm_86`, `cp.async`, `ldmatrix.sync.aligned`, and
`mma.sync.aligned.m16n8k16` are present. A fresh PTX audit of the final
specializations reports 130 registers and 0 spills for the D64 path (24.5 KiB
shared memory, 5 barriers, 4 `STG.E.128`, 30 shared stores) and 225 registers
with 0 spills for D128; neither is register-limited (Nsight Compute reports
the D64 path at 3 resident CTAs with shared memory allowing 4, and
`.minnctapersm 4` measured neutral).

The following measured alternatives are rejected and are not part of the
final dispatch: D64 `tile_m=128, tile_n=128` (+9 %), D64 `tile_m=128`
(+3.9 %), the wide `x4.trans` V ldmatrix at D64 (+5 %), D128 `tile_m=128`
with `tile_n=32` (+12 %) or `tile_n=64` (one resident CTA per SM),
D128 `tile_m=64, tile_n=128` (+19.5 %), D128 with eight M warps (halving
per-warp MMA work while K/V replica cost doubles), D128 with a reloaded Q
operand (neutral), sequential D128 output slices, D128 with two N warps,
removing the
cross-warp softmax barriers, and a packed `vector<2xf16>` probability store
(correct but 6--10 % slower than the scalar store). Native `(N,K)` K staging
with direct `cp.async` is now accepted for both head dimensions; the
transposed `(K,N)` path remains only as the `NATIVE_K = False` fallback for
bisection.

- Fulfilled: fixed-layout FP16 forward for heads 16/32, dimensions 64/128,
  and lengths 4096/8192/16384 on SM86.
- Fulfilled: FP32 score/output accumulation and online softmax with a
  tiled, sub-quadratic-memory implementation.
- Fulfilled: host validation, focused numerical tests, PTX checks,
  requested-shape benchmarks, and compatibility regression coverage.
- Fulfilled: existing SM120 target defaults and the prior SM86 matmul path
  remain unchanged by default; attention behavior is selected explicitly.
- Partially fulfilled: performance is at parity with the official CuTeDSL
  Ampere kernel across the matrix (1.004x--1.049x, mostly ~1.01x) and at
  parity-or-faster than FA4's SM80 CuTeDSL kernel
  (`tools/compare/sweep.py --reference fa4`). Against installed FA2 the
  kernel is 0.79x--1.01x on D64 (faster on three of six shapes) and
  1.05x--1.13x on D128, where FA2 is ~7 % faster than FA4 and remains the
  strongest reference. Remaining D128 gap and the concrete unlocks are
  tracked in `docs/PLAN_official_comparison.md`.
- Out of scope: causal/local masking, varlen, GQA, paging, split-KV,
  dropout, backward, and autotuning.
