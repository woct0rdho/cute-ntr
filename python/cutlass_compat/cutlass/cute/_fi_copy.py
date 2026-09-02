"""tv-layout tiled copy + SmemAllocator + arch/cpasync extensions for the
flashinfer operators (verbatim source support).

The rmsnorm/add_rmsnorm kernels build a ((T,R),(V,B)) thread-value layout
over a column-major (rows x cols) tile and copy global->shared with
cp.async 16B per thread per B-block, predicated on the k coordinate.
"""
from __future__ import annotations

import cutlass
from cutlass._bridge_helpers import _emitter, TypedScalar
from cutlass.cute import _fi_ext as _X
from cutlass.cute._fi_ext import Pointer, ReductionOp, TensorSSA  # noqa: F401


# ---------------------------------------------------------------------------
# cpasync atom
# ---------------------------------------------------------------------------
class CopyG2SOp:
    """cute.nvgpu.cpasync.CopyG2SOp — 16B cp.async global->shared."""

    def __init__(self, cache_mode=None, **kwargs):
        self.cache_mode = cache_mode
        self.num_bits_per_copy = 128
        self.kwargs = kwargs


class _CpAsyncAtom:
    def __init__(self, op, elem, num_bits_per_copy):
        self.op = op
        self.elem = elem
        self.bits = num_bits_per_copy


class _UniversalAtom:
    """Register/global universal copy using the flashinfer TV geometry."""

    def __init__(self, op, elem, num_bits_per_copy):
        self.op = op
        self.elem = elem
        self.bits = num_bits_per_copy


def make_copy_atom(op, elem=None, num_bits_per_copy=None, **kw):
    if op.__class__.__name__ == "CopyG2SOp":
        bits = num_bits_per_copy or getattr(op, "num_bits_per_copy", 128)
        return _CpAsyncAtom(op, elem, bits)
    if op.__class__.__name__ == "CopyUniversalOp":
        bits = (num_bits_per_copy or getattr(op, "num_bits_per_copy", None)
                or getattr(elem, "width", 32))
        return _UniversalAtom(op, elem, bits)
    from self_cutedsl.frontend import builtins as _b
    return _b.make_copy_atom(op, elem, **kw)


class TVCopyAsync:
    """make_tiled_copy(cpasync_atom, tv_layout, tiler_mn) — get_slice /
    partition_S / partition_D produce per-thread coordinate views."""

    def __init__(self, atom, tv_layout, tiler_mn):
        self.atom = atom
        self.tv = tv_layout            # host Layout object
        self.tiler = tiler_mn
        self.tidx = None

    def get_slice(self, tidx):
        self.tidx = tidx
        return self

    def partition_S(self, tensor):
        return TVPartView(self, tensor, "S")

    def partition_D(self, tensor):
        return TVPartView(self, tensor, "D")

    def retile(self, fragment):
        return fragment


class TVPartView:
    """Per-thread partitioned view: subscripts give per-value-unit objects
    (elem coordinate maps); drives copy()/autovec_copy/predicate_k."""

    def __init__(self, tc, tensor, role, stage=None):
        self.tc = tc
        self.tensor = tensor
        self.role = role
        self.stage = stage

    # -- geometry -------------------------------------------------------
    @property
    def _geom(self):
        tv = self.tc.tv
        try:
            (T, R), (V, B) = tv.shape[0], tv.shape[1]
        except (TypeError, ValueError):
            from self_cutedsl.frontend.cute_objects import _flatten

            flat = tuple(_flatten(tv.shape))
            T = flat[0] if flat else 32
            R, V, B = 1, 1, 1
        return int(T), int(R), int(V), int(B)

    @property
    def shape(self):
        _, _, V, B = self._geom
        return (V * B,)

    def thread_base(self):
        """(t, r) decomposition of the bound thread index."""
        T, _, _, _ = self._geom
        e = _emitter()
        tid = self.tc.tidx
        t = e.idx_binop("arith.remsi", tid, e.ssa("index", f"arith.constant {T} : index")) \
            if not isinstance(tid, int) else tid % T
        r = e.idx_binop("arith.divsi", tid, e.ssa("index", f"arith.constant {T} : index")) \
            if not isinstance(tid, int) else tid // T
        return t, r

    def coord(self, j):
        """Tile (row, col) for value-unit j of the bound thread (constexpr
        j only — the kernels iterate value units at trace time)."""
        e = _emitter()
        T, _, V, _ = self._geom
        t, r = self.thread_base()

        def _c(x):
            return x if hasattr(x, "name") else e.ssa(
                "index", f"arith.constant {int(x)} : index")

        v = j % V
        b = j // V
        if hasattr(t, "name"):
            tbase = e.idx_binop("arith.muli", t,
                                e.ssa("index", f"arith.constant {V} : index"))
            col = e.idx_binop("arith.addi", tbase, e.ssa(
                "index", f"arith.constant {v + b * V * T} : index"))
        else:
            col = int(t) * V + v + b * V * T
        return r, col

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # Canonical staged copy views use [..., stage].  Retain the
            # selected stage on the partitioned destination instead of
            # discarding it as the old elementwise-only view did.
            selected = [value for value in idx if value is not None]
            if len(idx) >= 3 and selected:
                return TVPartView(self.tc, self.tensor, self.role,
                                  stage=selected[-1])
            selector = idx[0]
            if selector is None or isinstance(selector, tuple):
                return self
            idx = selector
        return _TVUnit(self, idx)

    def store(self, view):
        """Store a register FragmentView into this shared-memory partition."""
        e = _emitter()
        arr = self.tensor.base
        ptr = arr.ptr if hasattr(arr, "ptr") else arr
        _, _, V, B = self._geom
        for j in range(V * B):
            row, col = self.coord(j)
            row = row if hasattr(row, "name") else e.ssa(
                "index", f"arith.constant {int(row)} : index")
            col = col if hasattr(col, "name") else e.ssa(
                "index", f"arith.constant {int(col)} : index")
            offset = e.idx_binop(
                "arith.addi",
                e.idx_binop(
                    "arith.muli",
                    row,
                    e.ssa(
                        "index", f"arith.constant {int(self.tc.tiler[1])} : index"),
                ),
                col,
            )
            offset64 = e.ssa(
                "i64", f"arith.index_cast {offset.name} : index to i64")
            elem_ptr = e.ssa(
                "!llvm.ptr<3>",
                f"llvm.getelementptr {ptr.name}[{offset64.name}] : "
                "(!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16",
            )
            value = view.values[j]
            e.raw(
                f"llvm.store {value.name}, {elem_ptr.name} : f16, !llvm.ptr<3>")


class _TVUnit:
    def __init__(self, view, j):
        self.view = view
        self.j = int(j)


# ---------------------------------------------------------------------------
# copy() lowering: cp.async per 16B block with predicate
# ---------------------------------------------------------------------------

def _is_matrix_view(src, dst):
    """Recognize object-model Tensor views used by dense SM86 GEMM."""
    from self_cutedsl.frontend.cute_objects import Tensor as HostTensor

    return isinstance(getattr(src, "tensor", None), HostTensor) and \
        isinstance(getattr(dst, "tensor", None), HostTensor) and \
        hasattr(src.tensor, "origin") and hasattr(dst.tensor, "base")


def _index_offset(e, terms):
    """Materialize ``sum(coord * stride)`` as an index SSA value."""
    total = e.idx_const(0)
    for coord, stride in terms:
        value = coord
        if not hasattr(value, "type"):
            value = e.idx_const(int(value))
        elif value.type != "index":
            value = e.ssa("index", f"arith.index_cast {value.name} : "
                           f"{value.type} to index")
        if int(stride) != 1:
            value = e.idx_binop("arith.muli", value, int(stride))
        total = e.idx_binop("arith.addi", total, value)
    return total


def _stage_offset(e, view, stage_stride):
    stage = getattr(view, "stage", None)
    if stage is None:
        return e.idx_const(0)
    if not hasattr(stage, "type"):
        return e.idx_const(int(stage) * int(stage_stride))
    value = stage if stage.type == "index" else e.ssa(
        "index", f"arith.index_cast {stage.name} : {stage.type} to index")
    return e.idx_binop("arith.muli", value, int(stage_stride))


def copy_matrix_cpasync(src_view, dst_view):
    """Emit the validated 16-byte SM86 matrix staging transactions.

    The layout-derived source origin and destination stage replace the former
    kernel-level ``block_m/block_n/kt`` offset expressions.  Request ownership
    is intentionally unchanged: this preserves the measured two-round
    64x64/64x128 schedule and its register/resource behavior.
    """
    from self_cutedsl.frontend.cute_objects import _flatten

    e = _emitter()
    src = src_view.tensor
    dst = dst_view.tensor
    src_shape = tuple(int(v) for v in _flatten(src.layout.shape))
    dst_shape = tuple(int(v) for v in _flatten(dst.layout.shape))
    if len(src_shape) < 2 or len(dst_shape) < 2:
        raise ValueError("matrix cp.async requires rank-2 tile views")
    operand = getattr(src_view.tc, "operand", None)
    if operand is None:
        operand = "A" if src_shape[0] >= src_shape[1] else "B"
    if operand == "A":
        tile_m, tile_k, minor = src_shape[0], src_shape[1], src_shape[1]
        smem_stride = dst_shape[1]
        stage_elems = tile_m * smem_stride
    elif operand == "B":
        tile_k, tile_n, minor = src_shape[0], src_shape[1], src_shape[1]
        smem_stride = dst_shape[1]
        stage_elems = tile_k * smem_stride
    else:
        raise ValueError(f"unknown matrix cp.async operand {operand!r}")
    if tuple(dst_shape[:2]) != tuple(src_shape[:2]):
        raise ValueError("source and destination tile shapes must match")
    src_stride = tuple(int(v) for v in _flatten(src.layout.stride))
    dst_stride = tuple(int(v) for v in _flatten(dst.layout.stride))
    src_origin = _index_offset(e, getattr(src, "origin", ()))
    stage_stride = dst_stride[2] if len(dst_stride) > 2 else stage_elems
    stage_origin = _stage_offset(e, dst_view, stage_stride)
    tile_elems = tile_m * tile_k if operand == "A" else tile_k * tile_n
    requests = tile_elems // 8
    threads = int(getattr(src_view.tc, "threads", 32 * 2 * 4))
    rounds = (requests + threads - 1) // threads
    tid = e.thread_id("x")
    for request_round in range(rounds):
        remaining = requests - request_round * threads
        active_count = min(threads, remaining)
        active = e.ssa("i1", f"arith.cmpi slt, {tid.name}, "
                        f"{e.idx_const(active_count).name} : index")
        e.open_if(active)
        elem = e.idx_binop("arith.addi", e.idx_binop(
            "arith.muli", tid, 8), request_round * threads * 8)
        row = e.idx_binop("arith.divsi", elem, minor)
        col = e.idx_binop("arith.remsi", elem, minor)
        goff = e.idx_binop(
            "arith.addi", src_origin,
            e.idx_binop("arith.addi", e.idx_binop(
                "arith.muli", row, src_stride[0]),
                e.idx_binop("arith.muli", col, src_stride[1])))
        soff = e.idx_binop(
            "arith.addi", stage_origin,
            e.idx_binop("arith.addi", e.idx_binop(
                "arith.muli", row, dst_stride[0]),
                e.idx_binop("arith.muli", col, dst_stride[1])))
        gp = e.gep(src.base, goff, "f16")
        sp = e.gep_smem(dst.base.ptr, soff)
        e.raw('nvvm.inline_ptx "cp.async.ca.shared.global [$1], [$0], 16;" '
              f'ro ({gp.name}, {sp.name} : !llvm.ptr<1>, !llvm.ptr<3>)')
        e.close_if()


def copy_cpasync(atom, tXgX, tXsX, pred):
    """Lower a canonical cp.async tiled copy.

    Matrix views produced by ``cute.local_tile`` use the SM86 matrix backend;
    the original TV path below remains the implementation for elementwise and
    predicated copies.  Both paths are selected from the partitioned view, so
    application kernels do not expose raw pointer arithmetic.
    """
    if _is_matrix_view(tXgX, tXsX):
        if pred is not None:
            raise NotImplementedError(
                "predicated SM86 matrix cp.async is not implemented")
        copy_matrix_cpasync(tXgX, tXsX)
        return
    e = _emitter()
    tc = tXgX.tc
    T, R, V, B = tXgX._geom
    gX = tXgX.tensor          # host/kernel tensor view (gmem)
    sX = tXsX.tensor          # smem tensor view
    row_stride, col_stride = _tensor_strides(gX)
    bidx = _active_bidx()
    bytes_per = atom.bits // 8
    elems = bytes_per // 2    # f16 element size assumed for G2S loads
    for b in range(B):
        row, col0 = tXgX.coord(b * V)
        # gmem address follows the source view's real strides. In particular,
        # the add-rmsnorm weight view broadcasts its row mode with stride 0.
        gr = e.idx_binop("arith.addi",
                         e.idx_binop("arith.muli", bidx,
                                     e.ssa("index", f"arith.constant {R} : index")),
                         row if not isinstance(row, int) else e.ssa("index", f"arith.constant {row} : index"))
        grow_off = e.idx_binop(
            "arith.muli",
            gr,
            e.ssa("index", f"arith.constant {row_stride} : index"),
        )
        gcol = col0 if not isinstance(col0, int) else e.ssa("index", f"arith.constant {col0} : index")
        gcol_off = e.idx_binop(
            "arith.muli",
            gcol,
            e.ssa("index", f"arith.constant {col_stride} : index"),
        ) if col_stride != 1 else gcol
        goff = e.idx_binop("arith.addi", grow_off, gcol_off)
        gp = _gep_gmem(e, gX, goff)
        # smem address: col-major tile -> row + col*R
        scol = gcol
        C_tile = _tile_cols(tXgX)
        row_s = row if not isinstance(row, int) else e.ssa("index", f"arith.constant {row} : index")
        soff = e.idx_binop(
            "arith.addi",
            e.idx_binop("arith.muli", row_s,
                        e.ssa("index", f"arith.constant {C_tile} : index")),
            scol)
        sp = _gep_smem(e, sX, soff)
        p = _pred_at(pred, b)
        e.open_if(p)
        e.raw(f'nvvm.inline_ptx "cp.async.cg.shared.global [$1], [$0], {bytes_per};" '
              f'ro ({gp.name}, {sp.name} : !llvm.ptr<1>, !llvm.ptr<3>)')
        e.close_if()


def copy_universal(atom, src, dst, pred):
    """Fragment -> global TV partition used for residual in-place stores."""
    from self_cutedsl.frontend.kernel_objects import Fragment

    if not isinstance(src, Fragment) or not isinstance(dst, TVPartView):
        raise TypeError(
            f"universal copy needs Fragment -> TVPartView, got "
            f"{type(src)} -> {type(dst)}"
        )
    e = _emitter()
    _, R, V, B = dst._geom
    row_stride, col_stride = _tensor_strides(dst.tensor)
    bidx = _active_bidx()
    for j in range(V * B):
        row, col = dst.coord(j)
        row = row if hasattr(row, "name") else e.ssa(
            "index", f"arith.constant {int(row)} : index")
        col = col if hasattr(col, "name") else e.ssa(
            "index", f"arith.constant {int(col)} : index")
        global_row = e.idx_binop(
            "arith.addi",
            e.idx_binop(
                "arith.muli",
                bidx,
                e.ssa("index", f"arith.constant {R} : index"),
            ),
            row,
        )
        offset = e.idx_binop(
            "arith.addi",
            e.idx_binop(
                "arith.muli",
                global_row,
                e.ssa("index", f"arith.constant {row_stride} : index"),
            ),
            e.idx_binop(
                "arith.muli",
                col,
                e.ssa("index", f"arith.constant {col_stride} : index"),
            ) if col_stride != 1 else col,
        )
        ptr = _gep_gmem(e, dst.tensor, offset)
        guard = _pred_at(pred, j // V)
        e.open_if(guard)
        value = src.slots[j]
        e.raw(f"llvm.store {value.name}, {ptr.name} : f16, !llvm.ptr<1>")
        e.close_if()


def _pred_at(pred, b):
    e = _emitter()
    if pred is None:
        return e.ssa("i1", "arith.constant 1 : i1")
    if hasattr(pred, "slots"):  # Fragment of i1 predicates
        return pred.slots[b]
    if isinstance(pred, (list, tuple)):
        return pred[b]
    if hasattr(pred, "name"):
        return pred
    return e.ssa("i1", "arith.constant 1 : i1")


def _tile_cols(view):
    tiler = view.tc.tiler
    return int(tiler[1])


def _tensor_strides(tensor):
    from self_cutedsl.frontend.cute_objects import _flatten

    strides = tuple(int(v) for v in _flatten(tensor.layout.stride))
    if not strides:
        return 1, 1
    if len(strides) == 1:
        return strides[0], 1
    return strides[0], strides[1]


def _active_bidx():
    from self_cutedsl.frontend import builtins as _b
    e = _emitter()
    return e.block_id("x")


def _gep_gmem(e, tensor, off):
    ptr = tensor.base if not hasattr(tensor.base, "ptr") else tensor.base.ptr
    if hasattr(off, "type"):
        if off.type == "index":
            off = e.ssa("i64", f"arith.index_cast {off.name} : index to i64")
        elif off.type == "i32":
            off = e.ssa("i64", f"arith.extsi {off.name} : i32 to i64")
    else:
        off = e.ssa("i64", f"arith.constant {int(off)} : i64")
    return e.ssa("!llvm.ptr<1>",
                 f"llvm.getelementptr {ptr.name}[{off.name}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16")


def _gep_smem(e, tensor, off):
    arr = tensor.base
    if hasattr(arr, "ptr"):
        p = arr.ptr
    else:
        p = arr
    off32 = e.ssa("i64", f"arith.index_cast {off.name} : index to i64")
    return e.ssa("!llvm.ptr<3>",
                 f"llvm.getelementptr {p.name}[{off32.name}] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16")


def copy_linear_cpasync(src, dst, operand, kt, tile_m, tile_n, tile_k,
                        block_m, block_n, stride_a, stride_b, stage,
                        threads=512, smem_stride_a=None, smem_stride_b=None,
                        swizzle_bits=0):
    """Stage one SM86 tile with aligned 16-byte cp.async requests.

    Requests are distributed in rounds when a tile needs more than one
    transaction per thread. This supports both the 512-thread 128x128
    profile and smaller CTA profiles without changing the source layout.
    """
    e = _emitter()
    tid = e.thread_id("x")
    tile_elems = int(tile_m) * int(tile_k) if operand == "A" else int(tile_k) * int(tile_n)
    smem_stride = (int(smem_stride_a) if operand == "A" and smem_stride_a is not None
                   else int(smem_stride_b) if operand == "B" and smem_stride_b is not None
                   else int(tile_k) if operand == "A" else int(tile_n))
    smem_tile_elems = (int(tile_m) * smem_stride if operand == "A"
                       else int(tile_k) * smem_stride)
    requests = tile_elems // 8
    rounds = (requests + int(threads) - 1) // int(threads)
    kt = kt if hasattr(kt, "name") else e.idx_const(int(kt))
    stage = stage if hasattr(stage, "name") else e.idx_const(int(stage))
    block_m = block_m if hasattr(block_m, "name") else e.idx_const(int(block_m))
    block_n = block_n if hasattr(block_n, "name") else e.idx_const(int(block_n))
    for request_round in range(rounds):
        remaining = requests - request_round * int(threads)
        active_count = min(int(threads), remaining)
        active = e.ssa("i1", f"arith.cmpi slt, {tid.name}, "
                       f"{e.idx_const(active_count).name} : index")
        e.open_if(active)
        elem = e.idx_binop("arith.addi", e.idx_binop("arith.muli", tid, 8),
                            request_round * int(threads) * 8)
        if operand == "A":
            local_row = e.idx_binop("arith.divsi", elem, int(tile_k))
            row = e.idx_binop("arith.addi",
                              e.idx_binop("arith.muli", block_m, int(tile_m)),
                              local_row)
            col = e.idx_binop("arith.remsi", elem, int(tile_k))
            goff = e.idx_binop(
                "arith.addi",
                e.idx_binop("arith.muli", row, int(stride_a)),
                e.idx_binop("arith.addi",
                            e.idx_binop("arith.muli", kt, int(tile_k)), col))
            soff = e.idx_binop(
                "arith.addi", e.idx_binop("arith.muli", stage, smem_tile_elems),
                e.idx_binop("arith.addi",
                            e.idx_binop("arith.muli", local_row, smem_stride), col))
        elif operand == "B":
            kk = e.idx_binop("arith.divsi", elem, int(tile_n))
            col = e.idx_binop("arith.remsi", elem, int(tile_n))
            krow = e.idx_binop("arith.addi", e.idx_binop("arith.muli", kt, int(tile_k)), kk)
            goff = e.idx_binop(
                "arith.addi",
                e.idx_binop("arith.muli", krow, int(stride_b)),
                e.idx_binop("arith.addi",
                            e.idx_binop("arith.muli", block_n, int(tile_n)), col))
            soff = e.idx_binop(
                "arith.addi", e.idx_binop("arith.muli", stage, smem_tile_elems),
                e.idx_binop("arith.addi",
                            e.idx_binop("arith.muli", kk, smem_stride), col))
        else:
            raise ValueError(f"unknown cp.async operand {operand!r}")
        gp = e.gep(src.ptr, goff, "f16")
        if swizzle_bits:
            soff64 = e.ssa(
                "i64", f"arith.index_cast {soff.name} : index to i64")
            shift6 = e.ssa("i64", "arith.constant 6 : i64")
            row_bits = e.ssa(
                "i64", f"arith.shrui {soff64.name}, {shift6.name} : i64")
            mask = e.ssa(
                "i64", f"arith.constant {(1 << int(swizzle_bits)) - 1} : i64")
            row_bits = e.ssa(
                "i64", f"arith.andi {row_bits.name}, {mask.name} : i64")
            shift3 = e.ssa("i64", "arith.constant 3 : i64")
            xor_bits = e.ssa(
                "i64", f"arith.shli {row_bits.name}, {shift3.name} : i64")
            physical = e.ssa(
                "i64", f"arith.xori {soff64.name}, {xor_bits.name} : i64")
            soff = e.ssa(
                "index", f"arith.index_cast {physical.name} : i64 to index")
        sp = e.gep_smem(dst.ptr, soff)
        e.raw('nvvm.inline_ptx "cp.async.ca.shared.global [$1], [$0], 16;" '
              f'ro ({gp.name}, {sp.name} : !llvm.ptr<1>, !llvm.ptr<3>)')
        e.close_if()


def cp_async_commit_group():
    _emitter().raw('nvvm.inline_ptx "cp.async.commit_group;"')


def cp_async_wait_group(count=0):
    _emitter().raw(f'nvvm.inline_ptx "cp.async.wait_group {int(count)};"')


class CoordTile:
    """local_tile(identity_tensor, tiler, coord): coordinate tile marker."""

    def __init__(self, meta, tiler, coord):
        self.meta = meta
        self.tiler = tiler
        self.coord = coord


class _CoordUnit:
    """tXcX[(0, j), 0, k] -> indexable [0]=row SSA, [1]=col SSA."""

    def __init__(self, view, j):
        from cutlass._bridge_helpers import _emitter
        e = _emitter()
        self.view = view
        T, R, V, B = view._geom
        t, r = view.thread_base()

        def _idx(x):
            return x if hasattr(x, "name") else e.ssa(
                "index", f"arith.constant {int(x)} : index")

        v = j % V
        b = j // V
        # global row for bounds predicates (tile row + block offset)
        gr = r
        if hasattr(r, "name"):
            bid = e.block_id("x")
            gr = e.idx_binop(
                "arith.addi",
                e.idx_binop("arith.muli", bid,
                            e.ssa("index", f"arith.constant {R} : index")),
                r)
        else:
            from cutlass._bridge_helpers import _emitter as _E
            gr = int(_E().block_id("x").name and 0 or 0) + 0  # constexpr fallback below
        self.row = _idx(gr) if hasattr(gr, "name") else _idx(r)
        tbase = e.idx_binop("arith.muli", _idx(t),
                            e.ssa("index", f"arith.constant {V} : index")) \
            if hasattr(t, "name") else _idx(int(t) * V)
        c = e.idx_binop("arith.addi", tbase, _idx(v))
        if b:
            c = e.idx_binop("arith.addi", c,
                            e.ssa("index", f"arith.constant {b * V * T} : index"))
        self.col = c

    def __getitem__(self, which):
        return self.row if int(which) == 0 else self.col


# TVPartView: hierarchical predicate-style indexing over coord partitions
_old_get = TVPartView.__getitem__


def _tvpart_getitem(self, idx):
    if isinstance(idx, tuple) and len(idx) == 3 \
            and isinstance(idx[0], tuple):
        (a, j) = idx[0]
        if int(a) == 0 and isinstance(self.tensor, CoordTile):
            return _CoordUnit(self, j)
    return _old_get(self, idx)


TVPartView.__getitem__ = _tvpart_getitem

_old_shape = TVPartView.shape


@property
def _tvpart_shape(self):
    if isinstance(self.tensor, CoordTile):
        n = _old_shape.fget(self)[0]
        return (n, 1, 1)
    return _old_shape.fget(self)


TVPartView.shape = _tvpart_shape
