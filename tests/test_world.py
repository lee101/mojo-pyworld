"""Tests for the mojo-pyworld kernels.

Every kernel is checked against a NumPy reference, including lengths that are
not a multiple of the SIMD width so the scalar tails are exercised, and short
inputs where the vector body never runs at all.
"""

import numpy as np
import pytest

import world


@pytest.fixture(scope="session", autouse=True)
def _built():
    """Fail fast with a clear message if the kernels were not built."""
    try:
        world._load()
    except world.KernelError as exc:  # pragma: no cover - build is expected
        pytest.fail(str(exc))


def _lengths() -> list[int]:
    """Lengths covering no-vector, partial-tail and exact-multiple cases."""
    w = world.simd_width()
    return [1, 2, 3, w - 1, w, w + 1, 2 * w, 2 * w + 3, 1000]


@pytest.mark.parametrize("n", _lengths())
def test_zero(n):
    a = np.ones(n, dtype=np.float64)
    world.zero(a)
    np.testing.assert_array_equal(a, np.zeros(n))


@pytest.mark.parametrize("n", _lengths())
def test_scale(n):
    a = np.arange(n, dtype=np.float64) - 3.0
    expect = a * 2.5
    world.scale(a, 2.5)
    np.testing.assert_allclose(a, expect)


@pytest.mark.parametrize("n", _lengths())
def test_axpy(n):
    x = np.arange(n, dtype=np.float64)
    y = np.ones(n, dtype=np.float64)
    expect = y + 3.0 * x
    world.axpy(3.0, x, y)
    np.testing.assert_allclose(y, expect)


@pytest.mark.parametrize("n", _lengths())
def test_dot(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    assert world.dot(x, y) == pytest.approx(float(np.dot(x, y)))


@pytest.mark.parametrize("n", [n for n in _lengths() if n >= 2])
def test_heat_step(n):
    rng = np.random.default_rng(n)
    u = rng.standard_normal(n)
    alpha = 0.1
    before = u.copy()

    # Zero-flux (copy-through) boundaries: laplacian is 0 at index 0 and n-1.
    lap = np.zeros(n, dtype=np.float64)
    lap[1:-1] = u[:-2] - 2.0 * u[1:-1] + u[2:]
    expect = u + alpha * lap

    got = world.heat_step(u, alpha)
    np.testing.assert_allclose(got, expect)
    # input must not be aliased or clobbered
    np.testing.assert_array_equal(u, before)


def test_heat_step_boundaries_are_copied_through():
    rng = np.random.default_rng(7)
    u = rng.standard_normal(100)
    out = world.heat_step(u, 0.1)
    assert out[0] == u[0]
    assert out[-1] == u[-1]


def test_heat_step_does_not_alias_input():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    before = a.copy()
    out = world.heat_step(a, 0.25)
    np.testing.assert_array_equal(a, before)
    assert out is not a


def test_heat_step_single_element():
    a = np.array([5.0])
    np.testing.assert_array_equal(world.heat_step(a, 0.5), a)


def test_heat_step_rejects_empty():
    with pytest.raises(ValueError):
        world.heat_step(np.zeros(0), 0.5)


def test_ragged_lengths_truncate_to_shortest():
    x = np.ones(5, dtype=np.float64)
    y = np.zeros(3, dtype=np.float64)
    world.axpy(2.0, x, y)
    np.testing.assert_array_equal(y, np.full(3, 2.0))


def test_simd_width_is_power_of_two():
    w = world.simd_width()
    assert w > 1
    assert w & (w - 1) == 0


def test_exports_present():
    assert set(world.KERNELS) == {
        "world_zero",
        "world_scale",
        "world_axpy",
        "world_dot",
        "world_heat_step",
        "world_simd_width",
    }