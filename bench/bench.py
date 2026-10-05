"""Micro-benchmark for the mojo-pyworld kernels against NumPy.

Run with `pixi run bench`. Reports GB/s for the bandwidth-bound kernels and
GFLOP/s for the dot product.
"""

from __future__ import annotations

import time

import numpy as np

import world

SIZES = (1 << 10, 1 << 14, 1 << 18, 1 << 22)
ALPHA = 1.0001
REPEATS = 50


def _best(fn, *args) -> float:
    """Return the fastest of REPEATS timings, in seconds."""
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def _report(label: str, n: int, secs: float, unit: str, unit_label: str) -> None:
    rate = (n * unit) / secs / 1e9
    print(f"{label:<10} n={n:<10} {secs * 1e6:10.2f} us   {rate:8.2f} {unit_label}")


def main() -> None:
    world._load()
    print(f"simd width (float64): {world.simd_width()}\n")

    print("world_scale   in-place, 8 B read + 8 B write per element")
    for n in SIZES:
        a = np.random.default_rng(0).standard_normal(n)
        secs = _best(world.scale, a, ALPHA)
        _report("scale", n, secs, 16.0, "GB/s")

    print("\nworld_zero    in-place, 8 B write per element")
    for n in SIZES:
        a = np.zeros(n, dtype=np.float64)
        secs = _best(world.zero, a)
        _report("zero", n, secs, 8.0, "GB/s")

    print("\nworld_dot     2x8 B read per element, 2 FLOP")
    for n in SIZES:
        rng = np.random.default_rng(1)
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        secs = _best(world.dot, x, y)
        _report("dot", n, secs, 16.0, "GB/s")

    print("\nworld_heat_step   2 buffers, ~5 FLOP")
    for n in SIZES:
        u = np.random.default_rng(2).standard_normal(n)
        secs = _best(world.heat_step, u, 0.1)
        _report("heat", n, secs, 16.0, "GB/s")


if __name__ == "__main__":
    main()