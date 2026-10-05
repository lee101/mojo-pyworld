"""Vector kernels for mojo-pyworld.

Dialect rules applied here (see MOJO_NOTES.md):
  * `def` everywhere, never `fn`.
  * `@export` needs an explicit `abi()` effect, and an `abi("C")` function may
    not be `raises`.
  * Buffers cross the C ABI as `Int` addresses; the `Pointer` is rebuilt inside
    the body. Pointers are non-nullable, so the null / empty case is handled by
    returning before a pointer is ever constructed.
  * `load` / `store` are deprecated on this toolchain in favour of
    `unsafe_load` / `unsafe_store` (both take the same `width=` argument).
  * No `main` here: `mojo build --emit shared-lib` fails if the file defines it.

This port is CPU-only and serial on purpose: `max.algorithm` (where
`parallelize` now lives on 1.2.0) ships in the `max` conda package, which is not
in this environment's dependency set. Every kernel below is SIMD-vectorised
with an explicit scalar tail instead.
"""

from std.sys import simd_width_of

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


@export("world_zero")
def world_zero(buf: Int, n: Int) abi("C") -> None:
    var p = Ptr(unsafe_from_address=buf)
    var i = 0
    while i + W <= n:
        p.unsafe_store(i, SIMD[DType.float64, W](0.0))
        i += W
    while i < n:
        p.unsafe_store(i, 0.0)
        i += 1


@export("world_scale")
def world_scale(buf: Int, n: Int, alpha: Float64) abi("C") -> None:
    var p = Ptr(unsafe_from_address=buf)
    var i = 0
    while i + W <= n:
        p.unsafe_store(i, p.unsafe_load[width=W](i) * alpha)
        i += W
    while i < n:
        p.unsafe_store(i, p.unsafe_load(i) * alpha)
        i += 1


@export("world_axpy")
def world_axpy(alpha: Float64, x: Int, nx: Int, y: Int, ny: Int) abi("C") -> None:
    var px = Ptr(unsafe_from_address=x)
    var py = Ptr(unsafe_from_address=y)
    var n = min(nx, ny)
    var i = 0
    while i + W <= n:
        py.unsafe_store(
            i, py.unsafe_load[width=W](i) + px.unsafe_load[width=W](i) * alpha
        )
        i += W
    while i < n:
        py.unsafe_store(i, py.unsafe_load(i) + px.unsafe_load(i) * alpha)
        i += 1


@export("world_dot")
def world_dot(x: Int, nx: Int, y: Int, ny: Int) abi("C") -> Float64:
    var px = Ptr(unsafe_from_address=x)
    var py = Ptr(unsafe_from_address=y)
    var n = min(nx, ny)
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += px.unsafe_load[width=W](i) * py.unsafe_load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += px.unsafe_load(i) * py.unsafe_load(i)
        i += 1
    return total


@export("world_heat_step")
def world_heat_step(
    dst: Int, u: Int, n: Int, alpha: Float64
) abi("C") -> None:
    """One explicit-Euler diffusion step: dst = u + alpha * laplacian(u).

    Boundaries are copied through, so `dst` is fully written for n >= 1. The
    vector body only runs where a full-width load of u[i-1], u[i] and u[i+1]
    stays in bounds, and the scalar tail finishes the remainder.
    """
    var pu = Ptr(unsafe_from_address=u)
    var po = Ptr(unsafe_from_address=dst)
    # Interior first, boundary copy-up last: the interior loop runs while
    # i + W <= n - 1, so its final full-width store can reach index n - 1 and
    # would clobber a boundary written before it.
    # Interior is [1, n-2], inclusive.
    var last = n - 2
    var i = 1
    while i + W <= last + 1:
        var c = pu.unsafe_load[width=W](i)
        var l = pu.unsafe_load[width=W](i - 1)
        var r = pu.unsafe_load[width=W](i + 1)
        po.unsafe_store(i, c + alpha * (l - c * 2.0 + r))
        i += W
    while i <= last:
        var c = pu.unsafe_load(i)
        var l = pu.unsafe_load(i - 1)
        var r = pu.unsafe_load(i + 1)
        po.unsafe_store(i, c + alpha * (l - c * 2.0 + r))
        i += 1
    # Boundaries copied through unchanged, written last.
    po.unsafe_store(0, pu.unsafe_load(0))
    if n > 1:
        po.unsafe_store(n - 1, pu.unsafe_load(n - 1))


@export("world_simd_width")
def world_simd_width() abi("C") -> Int:
    return W