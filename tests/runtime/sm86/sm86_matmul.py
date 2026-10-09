"""Idiomatic CuTe-style SM86 FP16 Tensor Core matmul.

The application kernel deliberately uses the original CuTe ownership model:
layouts and tiled operations are built by the host JIT, CTA tiles come from
`local_tile`, and all memory/MMA ownership is obtained through `get_slice`
and `partition_*`.  The compatibility backend still lowers these objects to
the validated Ampere cp.async, ldmatrix, and mma.sync instruction sequence.
"""
import cutlass
import cutlass.cute as cute

from cutlass.cute.runtime import from_dlpack

TILE_M = 64
TILE_K = 32
PIPELINE_STAGES = 2
WARP_M = 2


def _select_tile_k(K):
    if K >= TILE_K and K % TILE_K == 0:
        return TILE_K
    if K % 32 == 0:
        return 32
    return 16


def _make_gmem_copy(atom, operand, tile_n, block_threads, tile_k):
    """Build a canonical TV tiled copy; matrix geometry is backend metadata."""
    copy_elems = 128 // 16
    if operand == "A":
        units = tile_k // copy_elems
    else:
        units = tile_n // copy_elems
    thr_layout = cute.make_layout(
        (block_threads // units, units), stride=(units, 1))
    val_layout = cute.make_layout((1, copy_elems))
    tiled = cute.make_tiled_copy_tv(atom, thr_layout, val_layout)
    # The public descriptor remains make_tiled_copy_tv; these fields only
    # select the architecture backend for the matrix-shaped specialization.
    tiled.operand = operand
    tiled.threads = block_threads
    return tiled


@cute.kernel
def sm86_matmul_kernel(
    a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
    k_tiles: cutlass.Int32, K: cutlass.Constexpr, N: cutlass.Constexpr,
    LDA: cutlass.Constexpr, LDB: cutlass.Constexpr,
    raster_factor: cutlass.Constexpr, tile_n: cutlass.Constexpr,
    tile_k: cutlass.Constexpr, warps_n: cutlass.Constexpr,
    sA_layout: cute.Layout,
    sB_layout: cute.Layout, tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy, tiled_copy_s2r_A: cute.TiledCopy,
    tiled_copy_s2r_B: cute.TiledCopy, tiled_copy_C: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx_x, bidx_y, _ = cute.arch.block_idx()
    bidx_m = bidx_x // raster_factor
    bidx_n = (bidx_x % raster_factor) + bidx_y * raster_factor

    smem_a = cute.make_smem_tile(
        "sm86_smem_a", PIPELINE_STAGES * TILE_M * (tile_k + 8),
        cutlass.Float16, align=16)
    smem_b = cute.make_smem_tile(
        "sm86_smem_b", PIPELINE_STAGES * tile_k * (tile_n + 8),
        cutlass.Float16, align=16)
    sA = smem_a.get_tensor(sA_layout)
    sB = smem_b.get_tensor(sB_layout)

    # The tiled MMA owns the per-thread accumulator and operand fragments.
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(
        cute.local_tile(c, (TILE_M, tile_n), (bidx_m, bidx_n)))
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tAsA = thr_copy_A.partition_D(sA)
    tBsB = thr_copy_B.partition_D(sB)

    thr_copy_s2r_A = tiled_copy_s2r_A.get_slice(tidx)
    thr_copy_s2r_B = tiled_copy_s2r_B.get_slice(tidx)
    tCsA_copy = thr_copy_s2r_A.partition_S(sA)
    tCsB_copy = thr_copy_s2r_B.partition_S(sB)
    tCrA_copy = thr_copy_s2r_A.retile(tCrA)
    tCrB_copy = thr_copy_s2r_B.retile(tCrB)

    # Prime stage zero using only canonical partitioned copy views.
    gA = cute.local_tile(a, (TILE_M, tile_k), (bidx_m, 0))
    gB = cute.local_tile(b, (tile_k, tile_n), (0, bidx_n))
    cute.copy(tiled_copy_A, thr_copy_A.partition_S(gA),
              tAsA[None, None, None, 0])
    cute.copy(tiled_copy_B, thr_copy_B.partition_S(gB),
              tBsB[None, None, None, 0])
    cute.cp_async_commit_group()

    for kt in range(0, k_tiles):
        cute.cp_async_wait_group(0)
        cute.sync_threads()
        next_kt = kt + 1
        if next_kt < k_tiles:
            next_stage = next_kt % PIPELINE_STAGES
            gA = cute.local_tile(a, (TILE_M, tile_k),
                                 (bidx_m, next_kt))
            gB = cute.local_tile(b, (tile_k, tile_n),
                                 (next_kt, bidx_n))
            cute.copy(tiled_copy_A, thr_copy_A.partition_S(gA),
                      tAsA[None, None, None, next_stage])
            cute.copy(tiled_copy_B, thr_copy_B.partition_S(gB),
                      tBsB[None, None, None, next_stage])
            cute.cp_async_commit_group()

        stage = kt % PIPELINE_STAGES
        # Fill every K-step's register slot before issuing MMA.  This keeps
        # the validated register prefetch order of the original kernel while
        # retaining the public partitioned-copy interface.
        for k_step in range(tile_k // 16):
            cute.copy(
                tiled_copy_s2r_A,
                tCsA_copy[None, None, None, stage][None, None, k_step],
                tCrA_copy[None, None, k_step],
            )
            cute.copy(
                tiled_copy_s2r_B,
                tCsB_copy[None, None, None, stage][None, None, k_step],
                tCrB_copy[None, None, k_step],
            )
        for k_step in range(tile_k // 16):
            # The frontend recognizes gemm's Fragment mutation as a
            # loop-carried value and emits scalar mma.sync results.
            cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_step],
                      tCrB[None, None, k_step], tCrC)

    # C ownership is the MMA C fragment layout, not a hand-written lane map.
    tiled_copy_C = tiled_copy_C.get_slice(tidx)
    tCgC = tiled_copy_C.partition_D(
        cute.local_tile(c, (TILE_M, tile_n), (bidx_m, bidx_n)))
    cute.copy(tiled_copy_C, tCrC, tCgC)


@cute.jit
def sm86_matmul(
    a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
    k_tiles: cutlass.Int32, K: cutlass.Constexpr,
    N: cutlass.Constexpr, M: cutlass.Constexpr,
    LDA: cutlass.Constexpr, LDB: cutlass.Constexpr,
    tile_n: cutlass.Constexpr, warps_n: cutlass.Constexpr,
):
    block_threads = 32 * WARP_M * warps_n
    tile_k = _select_tile_k(K)
    smem_stride_a = tile_k + 8
    smem_stride_b = tile_n + 8

    # Host-side layout algebra mirrors the original Ampere dense_gemm setup.
    mA = cute.make_tensor(a.iterator,
                          cute.make_layout((M, K), stride=(LDA, 1)))
    mB = cute.make_tensor(b.iterator,
                          cute.make_layout((K, N), stride=(LDB, 1)))
    mC = cute.make_tensor(c.iterator,
                          cute.make_layout((M, N), stride=(N, 1)))
    sA_layout = cute.make_layout(
        (TILE_M, tile_k, PIPELINE_STAGES),
        stride=(smem_stride_a, 1, TILE_M * smem_stride_a))
    sB_layout = cute.make_layout(
        (tile_k, tile_n, PIPELINE_STAGES),
        stride=(smem_stride_b, 1, tile_k * smem_stride_b))

    async_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(), cutlass.Float16,
        num_bits_per_copy=128)
    copy_A = _make_gmem_copy(async_atom, "A", tile_n, block_threads, tile_k)
    copy_B = _make_gmem_copy(async_atom, "B", tile_n, block_threads, tile_k)

    mma = cute.nvgpu.warp.MmaF16BF16Op(
        cutlass.Float16, cutlass.Float32, (16, 8, 16))
    atom_layout = cute.make_layout((WARP_M, warps_n, 1))
    tiled_mma = cute.make_tiled_mma(
        mma, atom_layout,
        # The compatibility tiled-MMA backend models the per-warp tile in
        # `tile_mn` and derives the warp grid from atom_layout.  Keep the
        # canonical CuTe permutation explicit: each warp owns 32x32 while
        # (WARP_M, warps_n) warps cover the CTA tile.
        permutation_mnk=(WARP_M * 16, 32, 16))
    s2r_A = cute.make_tiled_copy_A(
        cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
            cutlass.Float16), tiled_mma)
    s2r_B = cute.make_tiled_copy_B(
        cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 2),
            cutlass.Float16), tiled_mma)
    c_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), mC.element_type,
        num_bits_per_copy=128)
    copy_C = cute.make_tiled_copy_C_atom(c_atom, tiled_mma)

    raster_factor = min(8, N // tile_n)
    grid = [((M + TILE_M - 1) // TILE_M) * raster_factor,
            (N + tile_n - 1) // tile_n // raster_factor, 1]
    sm86_matmul_kernel(
        mA, mB, mC, k_tiles, K, N, LDA, LDB, raster_factor, tile_n,
        tile_k, warps_n, sA_layout, sB_layout, copy_A, copy_B,
        s2r_A, s2r_B,
        copy_C, tiled_mma).launch(
            grid=grid, block=[block_threads, 1, 1])


def compile_sm86_matmul(a, b, c, options="--gpu-arch sm86"):
    """Compile and return a callable using the official target option.

    `options` is the full official CuTeDSL compile-option string; artifact
    inspection callers append the official dump flags to it.
    """
    M, K = a.shape
    _, N = b.shape
    assert a.is_cuda and b.is_cuda and c.is_cuda
    tile_n = 64 if N <= 2048 else 128
    warps_n = 2 if tile_n == 64 else 4
    assert K % 16 == 0 and M % TILE_M == 0 and N % tile_n == 0
    k_tile = _select_tile_k(K)
    return cute.compile(
        sm86_matmul,
        from_dlpack(a), from_dlpack(b), from_dlpack(c),
        K // k_tile, K, N, M, a.stride(0), b.stride(0),
        tile_n, warps_n, options=options,
    )


def launch_sm86_matmul(a, b, c):
    """Compile and launch using the official CuTe compile target option."""
    compile_sm86_matmul(a, b, c)()
