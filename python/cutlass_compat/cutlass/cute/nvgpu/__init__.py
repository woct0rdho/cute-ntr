"""cutlass.cute.nvgpu - device op markers (compat)."""
from enum import Enum


class LoadCacheMode(str, Enum):
    """Cache policy names accepted by the original cp.async atom API."""

    ALWAYS = "always"
    GLOBAL = "global"
    STREAMING = "streaming"
    LAST_USE = "last_use"
    NONE = "none"


class CopyUniversalOp:
    """Plain universal load/store (ld.global/st.global)."""

    def __init__(self, num_bits_per_copy=None, **kwargs):
        self.num_bits_per_copy = num_bits_per_copy
        self.kwargs = kwargs


class LdMatrixOp:
    pass


class StMatrixOp:
    pass


def __getattr__(name):
    from importlib import import_module

    if name in ("warp", "cpasync", "warpgroup"):
        return import_module(f"cutlass.cute.nvgpu.{name}")
    raise AttributeError(name)
