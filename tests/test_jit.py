"""jit correctness tests: a jit-compiled solve must match the eager result."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

import pardiso_mkl_jax as pardiso


def test_jit_matches_eager(nonsymmetric_system):
    indptr, indices, values, dense, right_hand_side = nonsymmetric_system

    @functools.partial(jax.jit, static_argnames=("matrix_type",))
    def solve(indptr, indices, values, right_hand_side, matrix_type):
        return pardiso.solve(indptr, indices, values, right_hand_side, matrix_type=matrix_type)

    jit_solution = solve(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        jnp.asarray(values),
        jnp.asarray(right_hand_side),
        pardiso.MatrixType.REAL_NONSYMMETRIC,
    )
    eager_solution = pardiso.solve(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        jnp.asarray(values),
        jnp.asarray(right_hand_side),
        matrix_type=pardiso.MatrixType.REAL_NONSYMMETRIC,
    )
    np.testing.assert_allclose(np.asarray(jit_solution), np.asarray(eager_solution), rtol=1e-10)
    expected = np.linalg.solve(dense, right_hand_side)
    np.testing.assert_allclose(np.asarray(jit_solution), expected, rtol=1e-8, atol=1e-10)


def test_jit_reuses_compiled_executable_across_calls(nonsymmetric_system):
    """A jitted solve gives correct results across repeated calls with different right-hand sides.

    Uses different right-hand sides so it also confirms the compiled
    executable is reused rather than retracing on every call.
    """
    indptr, indices, values, dense, right_hand_side = nonsymmetric_system

    solve = jax.jit(
        lambda right_hand_side: pardiso.solve(
            jnp.asarray(indptr),
            jnp.asarray(indices),
            jnp.asarray(values),
            right_hand_side,
            matrix_type=pardiso.MatrixType.REAL_NONSYMMETRIC,
        )
    )

    first = solve(jnp.asarray(right_hand_side))
    second = solve(jnp.asarray(right_hand_side * 2.0))
    np.testing.assert_allclose(
        np.asarray(first), np.linalg.solve(dense, right_hand_side), atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(second), np.linalg.solve(dense, right_hand_side * 2.0), atol=1e-10
    )
