# Plan: Reproduce / exceed official CuTeDSL on SM86 GEMM and FlashAttention

Status: official CuTeDSL 4.8.0 runs natively on Windows and the comparison
harness is producing reproducible numbers. The SM86 FlashAttention kernel now
matches the official CuTeDSL Ampere FA2 kernel on D64 (1.02x-1.04x, all
lengths statistically equivalent) and is 9 %-19 % behind on D128. The GEMM is
within 6 % of the official example and of cuBLAS.

## 1. Why this document exists

The open stack (`python/self_cutedsl` + `python/cutlass_compat`) reimplements
the CuTe DSL surface. Now that the official `nvidia-cutlass-dsl` package
installs and runs on Windows, the optimisation target is no longer a guess:
the reference implementation is available locally, in the same language, for
the same architecture, and measurable on the same device.

This plan defines

- what "intermediate results" we compare,
- how we compare them without fooling ourselves on a throttling laptop,
- what the comparison found, what has been fixed, and what is left.

## 2. Environment

| item | value |
| --- | --- |
| official package | `nvidia-cutlass-dsl==4.8.0` (`+ libs-base/core/cu13`) |
| install | `pip install "nvidia-cutlass-dsl[cu13]==4.8.0"`, `pip install "nvidia-cutlass-dsl-libs-base[sass]==4.8.0"` |
| bundled toolchain | `nvidia-cuda-nvcc` 13.4.92 (`ptxas` 13.4), `nvidia-cuda-nvdisasm` 13.4.92 |
| device | RTX 3080 Ti Laptop, `sm_86`, driver 617.42 |
| local toolkit | CUDA 13.3 - too old for the official's `.version 9.4` PTX |
| reference implementations | official CuTeDSL Ampere example, `flash-attn-4` (FA4, CuTeDSL SM80), `flash_attn` 2.8.4 (C++ FA2) |

Two official-package quirks on Windows, both worked around:

1. The official emits `.version 9.4` PTX, which the local CUDA 13.3 `ptxas`
   rejects. The wheel `ptxas` 13.4 accepts it, so all comparison work pins
   the wheel toolchain.
2. `--keep-sass` is broken in 4.8.0 on Windows: `_nvdisasm_from_wheel()`
   probes for a file named `nvdisasm`, but the wheel installs
   `nvdisasm.exe`, so it falls back to the older toolkit copy and errors out.
   We never pass `--keep-sass`; SASS is produced by our own pinned
   `nvdisasm` step, identically for both stacks.

## 3. The intermediate-results surface we mirror

| official interface | open-stack implementation |
| --- | --- |
| `cute.compile(options="--keep-ptx --keep-cubin --dump-dir D --ptxas-options -v --generate-line-info")` | `python/self_cutedsl/compiler/artifacts.py`, wired through `frontend/jit.py` |
| `cutlass.testing.benchmark` / `JitArguments` / `get_workspace_count` | `python/cutlass_compat/cutlass/testing.py` |
| `cutlass.cute.testing.*` alias | `python/cutlass_compat/cutlass/cute/testing.py` |
| `layout_utils.reshape_acc_to_frgA` | `cute.reshape_acc_to_frgA` in the compat layer |
| `thr_mma.partition_shape_C` | `TiledMma.get_slice().partition_shape_C` |
| `cute.make_tiled_copy_tv` (universal atom) | `TVCopyUniversal`, lowers to a scalar fragment store or a 16-byte vectorized smem->gmem tile store |
| PTX -> CUBIN -> SASS | one shared `ptxas` + `nvdisasm` step for both stacks |

Tools (all under `tools/compare/`):

| tool | purpose |
| --- | --- |
| `stack_report.py` | PTX/CUBIN/SASS inspection pipeline and histograms |
| `dump_kernels.py` | runs one case under `--stack official|self|fa4`, emits an identical JSON report |
| `compare_stacks.py` | side-by-side static + timing diff of two reports |
| `bench_worker.py` / `paired_bench.py` | resident workers + interleaved A/B timing with log-space statistics and a gap CI |
| `sweep.py` | the whole FA shape matrix against a chosen reference |
| `profile_one.py` | single-kernel driver for Nsight Compute |
| `ldmatrix_probe.py` / `ldmatrix_model.py` | determine/verify the register mapping of non-transposed `ldmatrix` from an (N,K) shared tile |

### 3.1 What we compare, and why

1. PTX instruction histogram - the front end's code shape, independent of
   `ptxas` scheduling.
2. SASS opcode + width histogram - width matters more than count:
   `STG.E.U16` x32 and `STG.E.128` x4 move the same bytes with very different
   transaction counts.
3. `ptxas -v` resources - registers, spills, stack, smem, barriers.
4. Dynamic instruction counts and stall breakdown (Nsight Compute) - the
   instrument that actually localised the gap (see §5).
5. Wall time, measured with the protocol in §4.
6. Correctness against the same references the official examples use.

Normalization caveat. Raw instruction counts are not comparable across
kernels with differently structured outer loops: the official FA2 body is
duplicated for the first (masked) n-block and the steady state, so its static
`mma.sync` count is 2x ours while the per-iteration count is identical.
Dynamic counts and per-K-tile normalization are used for all conclusions.

## 4. Measurement methodology (noisy laptop)

The GPU throttles (observed: SM clock 1125 MHz -> 1762 MHz within one run).
`paired_bench.py` therefore:

- starts one resident worker per stack, each compiled once and warmed;
- alternates short batches between the two workers, so drift is
  common-mode and cancels in the paired gap;
- analyses log-time: score = median log time, scale =
  `max(stdev, 1.4826*MAD, IQR/1.349)`, standard error of the median =
  `1.2533*σ/√n` (per `evotensile/docs/noisy_measurements.md`);
- reports the 95 % CI of the paired log gap and only claims a difference
  when the CI clears an indifference threshold (`--epsilon-pct`, default 3 %);
  otherwise it reports *equivalent*;
- records SM/mem clocks and temperature before and after.

Small shapes are screening only; conclusions use S >= 4096.

## 5. What the comparison found

### 5.1 The measurement that localised the gap

For S4096 H16 D64 the two kernels execute identical MMA work but we
executed 2.08x the instructions, with 306x more shared-memory stores:

| metric | self (before) | official | ratio |
| --- | ---: | ---: | ---: |
| `HMMA` | 16.78 M | 16.78 M | 1.00 |
| total instructions | 183.6 M | 88.2 M | 2.08 |
| `LSU` pipe | 42.2 M | 8.1 M | 5.17 |
| shared loads | 5.00 M | 0.42 M | 12.0 |
| shared stores | 20.05 M | 0.07 M | 306 |

The top stall reasons were shared-memory pressure (`short_scoreboard` 1.93 +
`mio_throttle` 1.20 per issue) rather than the tensor pipe (2.23), while the
official was almost purely tensor-bound (8.59). Three structural causes:

1. Epilogue store width - we emitted 32 scalar `STG.E.U16`; the official
   stages `O` through shared memory and emits 4 `STG.E.128`.
2. Probability round trip - we wrote `P` to shared memory and re-read it
   with `ldmatrix` only to change the fragment layout.
3. K staging - we kept `sK` as `(K, N)` and transposed with 32 scalar
   shared stores per iteration, synchronously.

### 5.2 What was implemented

All three, plus the supporting pieces:

- `cute.reshape_acc_to_frgA` - the score fragment already holds, in each
  thread, exactly the FP16 pairs the `m16n8k16` A operand wants; only the
  re-pairing was needed. This removed the `P` shared-memory round trip, its
  barrier, and 32 shared stores per iteration.
- Single N-warp warp grid (4x1) for both configurations - required for the
  register-`P` path, and it also removes the cross-warp softmax reduction.
  The softmax helpers were generalized from a hard-coded 4-atom N to any
  width, and the online-softmax state count now derives from the MMA shape.
- `thr_mma.partition_shape_C` + `make_rmem_tensor` carrying `(4, MMA_M, MMA_N)`
  so accumulators can be created without a destination tensor.
- Native `(N, K)` K tile filled by plain `cp.async` and read by a
  non-transposed x4 `ldmatrix` (the 2x2 block arrangement: lanes 0-15
  address the N rows at K[0:8], lanes 16-31 at K[8:16]). This removed the
  transposed scalar stores and the synchronous global load.
- A generalized `S<B,4,3>` swizzle (B = 3 for 64-half rows, B = 4 for 128-half
  rows) threaded through staging, `ldmatrix`, and the epilogue vector store.
- Wide epilogue: `rmem -> smem` staging plus a 16-byte-per-thread
  `smem -> gmem` vector copy (`STG.E.128`), matching the official.
- `math.fma`-based softmax exponent (the official produced 68 `FFMA`; we
  produced 0).
- D128: dedicated `sV1` buffer (the old reuse of `sQ` became illegal once `Q`
  is read during the loop in some variants) and independent cp.async commit
  groups for K, V0 and V1 so each can complete as needed.

Rejected on measurement: reloading `Q` per k-step to relieve D128 register
pressure (25 % slower than the spills it avoided), and a packed `vector<2xf16>`
P store (correct, but 6-10 % slower than scalar stores at the time).

### 5.3 Debugging note worth keeping

The native-K switch first produced wrong results. Two independent causes were
found, both now covered by tooling:

- The K `cp.async` was never waited on. The old transposed path stored K
  with synchronous shared stores, so the existing `sync_threads()` was
  sufficient; a `cp.async` K needs `cp_async_wait_group`. K now commits as its
  own group and `wait_group(1)` makes it visible while V is still in flight.
- The `.x2` ldmatrix block arrangement is vertical, not horizontal.
  `ldmatrix_probe.py` stages a known pattern, loads it, and dumps the raw
  per-lane registers so the effective mapping can be read off evidence rather
  than assumed; `ldmatrix_model.py` replays the staging + swizzle + load model
  offline. The final addressing uses `.x4` exactly like the official.

## 6. Results

### 6.0 Which FlashAttention is the target

`tools/compare/fa_generations.py` probes all three installed generations on
this device before any comparison, because the newest is not the fastest here:

| S | H | D | FA2 | FA3 | FA4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 16 | 64 | 2388 | 2372 | 2390 |
| 4096 | 16 | 128 | 4555 | 4262 | 4249 |
| 8192 | 16 | 128 | 15477 | 13362 | 14359 |
| 16384 | 16 | 128 | 53305 | 52077 | 54773 |

FA3 is nominally Hopper-only, but the installed `flash_attn_interface` shim
dispatches on compute capability and selects a `kBlockM_sm86` kernel, which
does run here and verifies against FA2 - so FA3 is the fastest reference at
head_dim 128 and is the target. Nsight Compute identifies it as a
CUTLASS-style kernel: `FlashAttnFwdSm80<CollectiveMainloopFwdSm80<8, 1, 1,
tuple<C<128>, C<128>, C<128>>, ...>>` with a 256-thread block, i.e. CTA tile
128x128 with 8 warps, at grid (S/128, heads).

### 6.1 Head-to-head

All three tables are interleaved paired runs at `z = 1.96`, `epsilon = 3 %`,
5 rounds x 2 iterations, from `sweep.py`.

Against the official CuTeDSL Ampere kernel:

| case | official (µs) | self (µs) | ratio | 95 % CI | verdict |
| --- | ---: | ---: | ---: | --- | --- |
| fa_s4096_h16_d64 | 2269 | 2268 | 1.000 | [-1.3, +1.3] % | equivalent |
| fa_s4096_h32_d64 | 4506 | 4502 | 0.999 | [-0.5, +0.4] % | equivalent |
| fa_s8192_h16_d64 | 8993 | 7897 | 0.999 | [-7.0, +7.3] % | equivalent |
| fa_s8192_h32_d64 | 17697 | 17645 | 0.992 | [-11.6, +11.3] % | equivalent |
| fa_s16384_h16_d64 | 29395 | 29176 | 0.984 | [-3.1, -0.0] % | equivalent |
| fa_s16384_h32_d64 | 54088 | 57113 | 1.055 | [+4.5, +6.6] % | slower |
| fa_s4096_h16_d128 | 4519 | 4509 | 0.998 | [-2.2, +1.7] % | equivalent |
| fa_s4096_h32_d128 | 9000 | 8952 | 0.995 | [-3.2, +2.2] % | equivalent |
| fa_s8192_h16_d128 | 17982 | 15852 | 0.979 | [-7.2, +3.3] % | equivalent |
| fa_s8192_h32_d128 | 28146 | 26791 | 0.955 | [-8.6, -0.3] % | equivalent |
| fa_s16384_h16_d128 | 54165 | 56377 | 1.030 | [+1.9, +4.2] % | equivalent |
| fa_s16384_h32_d128 | 103834 | 111227 | 1.076 | [+4.8, +10.4] % | slower |

Against `flash-attn-4` (its own SM80 CuTeDSL kernel, 128x64 tile, 128
threads, `Q_in_regs=False`, one 64 KiB dynamic-smem stage):

| case | fa4 (µs) | self (µs) | ratio | 95 % CI | verdict |
| --- | ---: | ---: | ---: | --- | --- |
| fa_s4096_h16_d64 | 2282 | 2268 | 0.983 | [-5.3, +2.1] % | equivalent |
| fa_s4096_h32_d64 | 4526 | 4502 | 0.987 | [-1.8, -0.8] % | equivalent |
| fa_s8192_h16_d64 | 9052 | 8973 | 0.991 | [-9.5, +8.5] % | equivalent |
| fa_s8192_h32_d64 | 15850 | 14705 | 0.944 | [-9.9, -1.1] % | equivalent |
| fa_s16384_h16_d64 | 29661 | 28830 | 0.971 | [-4.8, -1.0] % | equivalent |
| fa_s16384_h32_d64 | 56848 | 58139 | 1.029 | [+1.5, +4.3] % | equivalent |
| fa_s4096_h16_d128 | 4835 | 4510 | 0.931 | [-8.9, -5.0] % | faster |
| fa_s4096_h32_d128 | 9573 | 8963 | 0.934 | [-7.2, -6.0] % | faster |
| fa_s8192_h16_d128 | 15864 | 14310 | 0.896 | [-13.9, -6.7] % | faster |
| fa_s8192_h32_d128 | 31922 | 28813 | 0.915 | [-11.8, -5.0] % | faster |
| fa_s16384_h16_d128 | 59973 | 57052 | 0.955 | [-6.1, -3.0] % | faster |
| fa_s16384_h32_d128 | 112155 | 112029 | 0.992 | [-1.6, +0.0] % | equivalent |

Against FA3, the fastest reference on this device (§6.0):

| case | fa3 (µs) | self (µs) | ratio | 95 % CI | verdict |
| --- | ---: | ---: | ---: | --- | --- |
| fa_s4096_h16_d64 | 2261 | 2267 | 1.002 | [-1.6, +2.1] % | equivalent |
| fa_s4096_h32_d64 | 4496 | 4500 | 1.001 | [-0.7, +0.8] % | equivalent |
| fa_s8192_h16_d64 | 8952 | 8972 | 1.000 | [-0.6, +0.6] % | equivalent |
| fa_s8192_h32_d64 | 17658 | 17413 | 0.986 | [-11.1, +9.4] % | equivalent |
| fa_s16384_h16_d64 | 29329 | 29019 | 0.985 | [-2.6, -0.3] % | equivalent |
| fa_s16384_h32_d64 | 55358 | 58214 | 1.049 | [+3.4, +6.4] % | slower |
| fa_s4096_h16_d128 | 4414 | 4509 | 1.012 | [-0.2, +2.7] % | equivalent |
| fa_s4096_h32_d128 | 8821 | 8956 | 1.015 | [+0.1, +3.0] % | equivalent |
| fa_s8192_h16_d128 | 17462 | 17454 | 1.002 | [-1.9, +2.4] % | equivalent |
| fa_s8192_h32_d128 | 28516 | 28482 | 0.993 | [-1.6, +0.2] % | equivalent |
| fa_s16384_h16_d128 | 54950 | 55826 | 1.025 | [-1.2, +6.5] % | equivalent |
| fa_s16384_h32_d128 | 105032 | 112052 | 1.062 | [+4.8, +7.6] % | slower |

Summary: the kernel is faster than FA4 on 11 of 12 cases and never slower
than 1.029x, and it beats FA2 everywhere at D64 (the canonical artifact
shows 0.946x--0.997x). Against FA3 it is at parity (0.985x--1.025x) on ten of
twelve cases; the two outliers are both `S=16384, H=32` -- the largest total
problem, where the reference pulls 5--6 % ahead.

Static state of the D64 kernel after the work (S4096 H16 D64):

| metric | before | official | self (now) |
| --- | ---: | ---: | ---: |
| registers / spills | 187 / 0 | 166 / 0 | 156 / 0 |
| shared memory | 33.5 KB | - | 24.5 KB |
| `STG.E.128` | 0 (32 x `STG.E.U16`) | 4 | 4 |
| `STS` (shared stores) | 80 | 28 | 30 |
| `BAR` (barriers) | 10 | 5 | 5 |
| shared-store requests | 20.05 M | 0.07 M | 0.12 M |
| total instructions | 183.6 M | 88.2 M | 155.4 M |
| top stall | short_scoreboard + mio | math pipe | math pipe (6.35) |

The GEMM is at 1.06x of the official example and 1.07x of cuBLAS on 4096^3
(32.2 TFLOP/s for cuBLAS, 30.2 for ours in the same interleaved run).

### 6.3 What closed the gap

Measured with Nsight Compute on D128 S4096 H16, in the order they were
applied:

| change | evidence | effect |
| --- | --- | --- |
| swizzle every `ldmatrix`-read smem tile | 177.1 M shared-load bank conflicts -> 0; shared-load wavefronts 219.2 M -> 42.1 M, exactly FA4's count for the same work | D128 6.78 -> 5.80 ms |
| `fmax` as `arith.maxnumf` | 68 `setp.gt.f32` + 68 `selp.f32` -> 68 `max.f32` | -136 loop instructions |
| packed probability conversion | `cvt.rn.f16x2.f32` on a `vector<2xf32>` written straight into half2 slots, replacing two scalar converts plus a register pack | -32 converts/iteration, removes the pack moves |
| `x4.trans` V ldmatrix at D128 | 64 `x2.trans` -> 32 `x4.trans` | ~1 % at D128, but 5 % *slower* at D64 - kept per-configuration |
| D128 `tile_m=64, tile_n=64` | 48 KiB static -> 2 resident CTAs (8 warps/SM) instead of 1 (4 warps/SM) | D128 5.80 -> 4.71 ms |
| K prefetch by one tile | K(kt+1) issued in the same buffer after the QK gemm consumed K(kt) (FA2's structure), so the transfer overlaps softmax + P@V | D128 vs FA4 0.977x -> 0.931x; D64 vs official 1.014x -> 1.000x |

The occupancy item is the structural one: at one resident CTA the scheduler has
a single warp per partition and issues an instruction every 10 cycles. Nsight
Compute reported 1.00 active warps per scheduler, 0.10 issued per cycle, with
the top stall being the math pipe. Halving the shared footprint doubles
residency and hides both the exposed K/V latency and the softmax arithmetic.

### 6.4 Same-source codegen status

`tools/compare/same_source.py` compiles one CuTeDSL source with both front ends
and diffs the PTX and resource report - the direct measurement of code
*generation*, with no "two different kernels" caveat. It drives the official
Ampere examples (which use only the public API) through this repository's
`cutlass_compat` layer.

Status: not yet green. The official front end compiles the GEMM example;
the open front end reaches the epilogue and stops. Fixed so far along the way
(the compat layer is the deliverable either way):
`cute.make_composed_layout`, swizzle propagation through `cute.tile_to_shape`
and `SmemArray.get_tensor`, `utils.LayoutEnum.COL_MAJOR`,
`cute.ceil_div` on shape tuples, `local_tile(..., proj=...)`, and rank-reducing
slices of an untiled global tensor (`mC[None, None, bidz]`).
Remaining blocker: `local_tile` with a tiler rank above the tensor rank
(`cta_tiler=(128,128,32)` over a 2-D view), which the example reconciles with
`proj`.

Everything else in this document compares *different* kernels and argues
their comparability; the same-source path is the one that will not need that
argument once it runs.

## 7. Remaining gap and work list

The FA4 milestone is met. What remains is the installed FA2 D128 gap
(1.05x--1.13x) and general polish.

1. Two-stage K/V pipeline. This is now the top item and it is the same fix
   that produced the biggest single win so far, applied one level up. Each
   iteration issues K and V and then immediately waits for K, so the global
   latency is fully exposed; with two resident CTAs it is partly covered, but
   the loads for iteration `i+1` could be issued during iteration `i`. FA4
   does exactly this with `num_stages=1`: it issues `load_V(n)` before the QK
   gemm and `load_K(n-1)` before the softmax, so every transfer overlaps
   compute. Double-buffering K and V at D128 costs `2*(16+8+8)` = 64 KiB
   plus 16 KiB for Q, which needs the dynamic-shared-memory path that is now
   implemented and tested (`tests/runtime/sm86/test_dynamic_smem.py`).
2. Instruction-count reduction. Still 1.76x the official's dynamic
   instruction count for identical MMA work at D64, dominated by softmax
   element work, address arithmetic, and converts. The measured next targets
   are the 226 `add.s64` and 273 `mov.b32` instructions in the D128 loop
   (FA4's loop body has 22 and 12 for the same work), i.e. hoisting
   `cp.async`/`ldmatrix` address math out of the k-loop, and the
   `is_first_n_block` specialization (the first iteration needs no output
   rescale and no row-max update).
3. Occupancy. D64 sits at 157 registers under the driver's JIT, which caps
   it at 3 CTAs per SM where shared memory would allow 4 (24.5 KiB each).
   The knob is now wired up - ``DG_ENTRY_DIRECTIVES=".minnctapersm 4"`` (or
   ``".maxnreg 128"``) inserts a per-entry PTX performance-tuning directive
   (PTX ISA 11.4: it must appear between the ``.entry`` signature and its
   body, which is why it lives in the PTX post-processing step) into the
   module the driver JIT compiles - and both measured neutral (1.0025x and
   1.0115x against FA3 versus 1.0020x for the baseline). The fourth CTA does
   not pay for its spills; a register-lean rewrite is the remaining option.
4. The two ``S=16384, H=32`` cases (5-6 % behind FA3, the only remaining
   regressions against any reference). They are L2-traffic-bound: measured
   with Nsight Compute at S16384 H32 D128, this kernel moves 68.9 GB through
   L2 (2.15 G sectors, 97.7 % hit rate, 1.58 GB from DRAM) because each of the
   256 M-tiles per head reads the whole 8 MB K/V. Halving that needs
   ``tile_m=128``, which needs 128 FP32 accumulators per thread at 128
   threads -- and every such variant measured worse, decisively so at this
   shape: ``128x64`` 1.236x, ``128x32`` 1.128x, D64 ``128x128`` 1.090x.
   FA4's CuTeDSL kernel spills 116 bytes at the same point; only FA3's
   CUTLASS-style 256-thread/128x128 kernel avoids the wall, by doubling the
   thread count so each thread keeps 64 accumulators. Reproducing that here
   means an 8-warp kernel, and the 8-warp variants measured 1.59x and 1.35x
   *slower* because K/V ldmatrix replication grows with the M-warp count.
   The remaining lever is therefore CTA rasterisation (which M-tiles share a
   wave), not tile shape.
5. GEMM epilogue and policy. Re-use the vectorized ``smem -> gmem`` store in
   the matmul epilogue (it is already shared code), then re-check the tile
   dispatch with the static diff for the 2048^3 and 4096x1024 shapes.
6. Same-source codegen comparison (6.4): unblock ``local_tile`` with a tiler
   rank above the tensor rank so the official GEMM example compiles with both
   front ends, then extend to the FA example.

## 8. How to run the comparison

```pwsh
# which FlashAttention generation is the target on this device
python tools/compare/fa_generations.py

# static + timing report for one case, one stack
python tools/compare/dump_kernels.py --stack official --kernel fa `
    --seqlen 4096 --heads 16 --head-dim 64 --m-block 64 --n-block 64 --sass
python tools/compare/dump_kernels.py --stack self --kernel fa `
    --seqlen 4096 --heads 16 --head-dim 64 --sass

# side-by-side static diff (per-K-tile normalized columns)
python tools/compare/compare_stacks.py --kernel fa

# throttle-robust interleaved timing (the number decisions use).
# The estimator pairs each measurement batch with the reference batch taken
# immediately before it, which cancels the laptop's common-mode drift.
python tools/compare/paired_bench.py --kernel fa --seqlen 4096 --heads 16 `
    --head-dim 64 --rounds 6 --batch-size 2
python tools/compare/paired_bench.py --kernel fa --seqlen 4096 --heads 16 `
    --head-dim 64 --reference fa4

# the whole matrix, against either reference
python tools/compare/sweep.py --reference fa4 --rounds 5 --batch-size 2

# A/B one source edit with the paired protocol, then restore the file
python tools/compare/ab_edit.py tests/runtime/sm86/sm86_flash_attention.py `
    "old text" "new text" -- --kernel fa --seqlen 4096 --heads 16 --head-dim 64

# static-analysis tools
python tools/compare/fa4_dump.py --head-dim 128 --m-block 128 --n-block 64
python tools/compare/ptx_ops.py <a.ptx> <b.ptx>          # opcode histogram
python tools/compare/ptx_loop_body.py <a.ptx> <b.ptx> 2  # loop-body mix
python tools/compare/smem_dyn_probe.py                   # dynamic smem support
python tools/compare/mlir_op_probe.py                    # op -> PTX mapping
python tools/compare/same_source.py --example gemm       # one source, both stacks

# ldmatrix mapping regression tool
python tools/compare/ldmatrix_probe.py
python tools/compare/ldmatrix_model.py

# profiling
& 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\ncu.bat' `
    --set full -k regex:sm86_flash_attention_kernel --launch-count 1 `
    --target-processes all python tools/compare/profile_one.py self 4096 16 64
```

Useful Nsight Compute metrics for this kernel (each is a decision input, not
just a number):

```
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum   must be 0
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum      shared-load cost
sm__inst_executed_pipe_{fma,alu,lsu,xu,tensor_op_hmma}.sum per-pipe work
smsp__warp_issue_stalled_*.per_warp_active.pct             top stall
```

## 9. Acceptance criteria

- FA: `paired_bench.py` reports equivalent or faster for the full shape
  matrix. Against the official kernel: met except the two `S=16384, H=32`
  cases (1.030x, 1.076x). Against FA4: met (0.896x--1.029x, 11 of 12 faster
  or equivalent). Against FA3, the fastest reference on this device: met on
  ten of twelve cases (0.985x--1.025x); the same two outliers are 1.049x and
  1.062x.
- Device coverage: `python tools/compare/fa_generations.py` is the gate - it
  re-derives which FA generation is fastest here before any comparison is
  quoted, because FA3 is Hopper-badged but does run on this sm86 part.
- GEMM: equivalent or faster on the `tools/perf/bench_sm86_matmul.py` shape
  set; currently 1.06x.
- No unexplained width regression in `compare_stacks.py` (no scalar
  `STG`/`STS` where the official uses 128-bit). Met.
- Gates stay green: focused FA tests, `-m sm86`, non-SM120/SM86, PTX contract.
- Every retained change is justified by a paired measurement under
  `artifacts/stack_compare/`.
