"""Tests for the token a solve returns and the version stamp that guards it.

A stateful solve reads the factorization the handle holds, and a later factor
overwrites it. Nothing sequences the two unless the solve hands back a value the
later call can wait on. These tests cover that returned token and the version
check that catches a token left over from before a later write.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pardiso_mkl_jax import primitive
from pardiso_mkl_jax.matrix import MatrixType


def test_solve_stateful_default_arity_is_unchanged(any_system):
    """Without return_token the solve still returns just (solution, final_iparm)."""
    indptr, indices, values, dense, right_hand_side = any_system
    matrix_type = MatrixType.REAL_NONSYMMETRIC
    indptr, indices, values = jnp.asarray(indptr), jnp.asarray(indices), jnp.asarray(values)
    right_hand_side = jnp.asarray(right_hand_side)

    token, _ = primitive.analyze(indptr, indices, values, matrix_type=matrix_type)
    token, _ = primitive.factor(token, indptr, indices, values, matrix_type=matrix_type)
    result = primitive.solve_stateful(
        token, indptr, indices, values, right_hand_side[None, :], matrix_type=matrix_type
    )
    assert len(result) == 2
    solution, _final_iparm = result
    np.testing.assert_allclose(
        np.asarray(solution[0]), np.linalg.solve(dense, right_hand_side), rtol=1e-8, atol=1e-10
    )
    primitive.release(token)


def test_solve_stateful_returns_a_threadable_token(system):
    """return_token gives back a token whose id and version come from the solve.

    Feeding that token into a later factor gives the factor a data dependency
    on the solve, so the solve is read before the factor overwrites it.
    """
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    indptr, indices, values = jnp.asarray(indptr), jnp.asarray(indices), jnp.asarray(values)
    right_hand_side = jnp.asarray(right_hand_side)
    other_values = values * 2.0

    token, _ = primitive.analyze(indptr, indices, values, matrix_type=matrix_type)
    token, _ = primitive.factor(token, indptr, indices, values, matrix_type=matrix_type)
    first, token, _ = primitive.solve_stateful(
        token,
        indptr,
        indices,
        values,
        right_hand_side[None, :],
        matrix_type=matrix_type,
        return_token=True,
    )
    token, _ = primitive.factor(token, indptr, indices, other_values, matrix_type=matrix_type)
    second, token, _ = primitive.solve_stateful(
        token,
        indptr,
        indices,
        other_values,
        right_hand_side[None, :],
        matrix_type=matrix_type,
        return_token=True,
    )
    np.testing.assert_allclose(
        np.asarray(first[0]), np.linalg.solve(dense, right_hand_side), rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(second[0]),
        np.linalg.solve(dense * 2.0, right_hand_side),
        rtol=1e-8,
        atol=1e-10,
    )
    primitive.release(token)


def test_reused_handle_stays_correct_under_jit(system):
    """A factor, solve, factor, solve chain threaded through the solve tokens.

    Every access to the handle sits on one data-dependency chain, so the second
    factor cannot run before the first solve reads what it needs. Both solves
    come back matching their own matrix.
    """
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    indptr, indices = jnp.asarray(indptr), jnp.asarray(indices)

    def run(values, other_values, right_hand_side):
        token, _ = primitive.analyze(indptr, indices, values, matrix_type=matrix_type)
        token, _ = primitive.factor(token, indptr, indices, values, matrix_type=matrix_type)
        first, token, _ = primitive.solve_stateful(
            token,
            indptr,
            indices,
            values,
            right_hand_side[None, :],
            matrix_type=matrix_type,
            return_token=True,
        )
        token, _ = primitive.factor(token, indptr, indices, other_values, matrix_type=matrix_type)
        second, token, _ = primitive.solve_stateful(
            token,
            indptr,
            indices,
            other_values,
            right_hand_side[None, :],
            matrix_type=matrix_type,
            return_token=True,
        )
        primitive.release(token.track(first, second))
        return first[0], second[0]

    first, second = jax.jit(run)(
        jnp.asarray(values), jnp.asarray(values * 2.0), jnp.asarray(right_hand_side)
    )
    np.testing.assert_allclose(
        np.asarray(first), np.linalg.solve(dense, right_hand_side), rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(second), np.linalg.solve(dense * 2.0, right_hand_side), rtol=1e-8, atol=1e-10
    )


def test_solve_rejects_a_stale_token_after_a_later_factor(any_system):
    """A solve on a token whose factorization was already replaced is refused.

    The second factor bumps the handle's version. Solving with the token from
    before that factor no longer matches the version the handle holds, so the
    native call reports a mismatch rather than returning the wrong matrix's
    answer.
    """
    indptr, indices, values, _dense, right_hand_side = any_system
    matrix_type = MatrixType.REAL_NONSYMMETRIC
    indptr, indices, values = jnp.asarray(indptr), jnp.asarray(indices), jnp.asarray(values)
    right_hand_side = jnp.asarray(right_hand_side)

    token, _ = primitive.analyze(indptr, indices, values, matrix_type=matrix_type)
    stale, _ = primitive.factor(token, indptr, indices, values, matrix_type=matrix_type)
    # A second factor moves the handle on to a newer version.
    current, _ = primitive.factor(stale, indptr, indices, values * 2.0, matrix_type=matrix_type)
    with pytest.raises(Exception, match="version"):
        primitive.solve_stateful(
            stale, indptr, indices, values, right_hand_side[None, :], matrix_type=matrix_type
        )
    primitive.release(current)


def test_factor_and_solve_returns_a_token_with_the_version_passed_through(system):
    """factor_and_solve hands back a token, and its version is the one it took in.

    The combined phase writes and reads the factorization in one call, so the
    token that named the handle still names the result and does not need a new
    version.
    """
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    indptr, indices, values = jnp.asarray(indptr), jnp.asarray(indices), jnp.asarray(values)
    right_hand_side = jnp.asarray(right_hand_side)

    token, _ = primitive.analyze(indptr, indices, values, matrix_type=matrix_type)
    solution, threaded, _ = primitive.factor_and_solve_stateful(
        token,
        indptr,
        indices,
        values,
        right_hand_side[None, :],
        matrix_type=matrix_type,
        return_token=True,
    )
    np.testing.assert_allclose(
        np.asarray(solution[0]), np.linalg.solve(dense, right_hand_side), rtol=1e-8, atol=1e-10
    )
    assert int(threaded.version) == int(token.version)
    # The threaded token still solves against the same handle.
    again, _ = primitive.solve_stateful(
        threaded, indptr, indices, values, right_hand_side[None, :], matrix_type=matrix_type
    )
    np.testing.assert_allclose(
        np.asarray(again[0]), np.linalg.solve(dense, right_hand_side), rtol=1e-8, atol=1e-10
    )
    primitive.release(threaded)


def test_token_is_a_pytree_with_id_version_and_counter(any_system):
    """The token flattens to three leaves, so jit and vmap can carry it."""
    indptr, indices, values, _dense, _right_hand_side = any_system
    matrix_type = MatrixType.REAL_NONSYMMETRIC
    indptr, indices, values = jnp.asarray(indptr), jnp.asarray(indices), jnp.asarray(values)

    token, _ = primitive.analyze(indptr, indices, values, matrix_type=matrix_type)
    leaves = jax.tree_util.tree_leaves(token)
    assert len(leaves) == 3
    primitive.release(token)
