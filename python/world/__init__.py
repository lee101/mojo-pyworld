"""ctypes bindings for the mojo-pyworld kernels.

The shared library is built by `pixi run build` into build/libworld.so.
Buffers cross the ABI as Int addresses (MOJO_NOTES.md section 2), so every
entry point takes `ndpointer` rather than a `POINTER` type.
"""

from __future__ import annotations

import ctypes
import os
from typing import Any

import numpy as np

__all__ = [
    "KERNELS",
    "build_path",
    "zero",
    "scale",
    "axpy",
    "dot",
    "heat_step",
    "simd_width",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
_DEFAULT_LIB = os.path.join(_PROJECT_ROOT, "build", "libworld.so")

_F64P = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")


class KernelError(RuntimeError):
    """Raised when the compiled kernels cannot be located."""


def build_path() -> str:
    """Return the path the compiled library is expected at."""
    return os.environ.get("WORLD_LIB", _DEFAULT_LIB)


_LIB: ctypes.CDLL | None = None
KERNELS: tuple[str, ...] = ()


def _load(path: str | None = None) -> ctypes.CDLL:
    """Load (once) and return the kernel library."""
    global _LIB, KERNELS
    if _LIB is not None:
        return _LIB

    lib_path = path or build_path()
    if not os.path.exists(lib_path):
        raise KernelError(
            f"{lib_path} not found; run `pixi run build` first "
            "(override with the WORLD_LIB environment variable)"
        )

    lib = ctypes.CDLL(lib_path)

    lib.world_zero.argtypes = [_F64P, ctypes.c_int]
    lib.world_zero.restype = None

    lib.world_scale.argtypes = [_F64P, ctypes.c_int, ctypes.c_double]
    lib.world_scale.restype = None

    lib.world_axpy.argtypes = [
        ctypes.c_double,
        _F64P,
        ctypes.c_int,
        _F64P,
        ctypes.c_int,
    ]
    lib.world_axpy.restype = None

    lib.world_dot.argtypes = [_F64P, ctypes.c_int, _F64P, ctypes.c_int]
    lib.world_dot.restype = ctypes.c_double

    lib.world_heat_step.argtypes = [_F64P, _F64P, ctypes.c_int, ctypes.c_double]
    lib.world_heat_step.restype = None

    lib.world_simd_width.argtypes = []
    lib.world_simd_width.restype = ctypes.c_int

    _LIB = lib
    KERNELS = (
        "world_zero",
        "world_scale",
        "world_axpy",
        "world_dot",
        "world_heat_step",
        "world_simd_width",
    )
    return lib


def _arr(a: Any, name: str) -> np.ndarray:
    if not isinstance(a, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(a).__name__}")
    return a


def zero(a: np.ndarray) -> np.ndarray:
    """Set every element of `a` to 0.0, in place."""
    _load().world_zero(a, a.size)
    return a


def scale(a: np.ndarray, alpha: float) -> np.ndarray:
    """Scale `a` by `alpha`, in place."""
    _load().world_scale(a, a.size, float(alpha))
    return a


def axpy(alpha: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y += alpha * x, in place. Operates over min(len(x), len(y)) elements."""
    _load().world_axpy(
        float(alpha),
        _arr(x, "x"),
        x.size,
        _arr(y, "y"),
        y.size,
    )
    return y


def dot(x: np.ndarray, y: np.ndarray) -> float:
    """Dot product over min(len(x), len(y)) elements."""
    return float(
        _load().world_dot(
            _arr(x, "x"),
            x.size,
            _arr(y, "y"),
            y.size,
        )
    )


def heat_step(a: np.ndarray, alpha: float) -> np.ndarray:
    """One explicit-Euler diffusion step into a fresh array.

    Boundaries are copied through unchanged, so the result is defined for
    n >= 1. `a` is not modified.
    """
    a = _arr(a, "a")
    if a.size < 1:
        raise ValueError("heat_step needs at least one element")
    out = np.empty_like(a)
    _load().world_heat_step(out, a, a.size, float(alpha))
    return out


def simd_width() -> int:
    """SIMD lane count the kernels were compiled for (float64)."""
    return int(_load().world_simd_width())