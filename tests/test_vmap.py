"""vmap correctness tests: batched right-hand sides, batched values, and both."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pardiso_mkl_jax as pmj


def _stacked_single_solves(indptr, indices, values_batch, right_hand_side_batch, matrix_type):
    """Reference implementation: call solve() once per batch element, no vmap."""
    return jnp.stack(
        [
            pmj.solve(
                jnp.asarray(indptr),
                jnp.asarray(indices),
                values,
                right_hand_side,
                matrix_type=matrix_type,
            )
            for values, right_hand_side in zip(values_batch, right_hand_side_batch, strict=True)
        ]
    )


def test_vmap_over_right_hand_side_matches_stacked_single_solves(system):
    matrix_type, indptr, indices, values, _dense, _right_hand_side = system
    dimension = indptr.shape[0] - 1
    batch_size = 4
    random_state = np.random.default_rng(7)
    right_hand_side_batch = random_state.uniform(-1.0, 1.0, size=(batch_size, dimension))
    values_batch = jnp.tile(jnp.asarray(values), (batch_size, 1))

    def solve_one(right_hand_side):
        return pmj.solve(
            jnp.asarray(indptr),
            jnp.asarray(indices),
            jnp.asarray(values),
            right_hand_side,
            matrix_type=matrix_type,
        )

    batched = jax.vmap(solve_one)(jnp.asarray(right_hand_side_batch))
    expected = _stacked_single_solves(
        indptr, indices, values_batch, jnp.asarray(right_hand_side_batch), matrix_type
    )
    np.testing.assert_allclose(np.asarray(batched), np.asarray(expected), rtol=1e-8, atol=1e-10)


def test_vmap_over_values_matches_stacked_single_solves(system):
    matrix_type, indptr, indices, values, _dense, right_hand_side = system
    batch_size = 4
    random_state = np.random.default_rng(11)
    # A positive scale factor preserves every supported matrix type's
    # defining property (definiteness, invertibility, the sign pattern of
    # an indefinite matrix's eigenvalues), so scaling is a safe way to
    # produce a batch of distinct but still-valid values for any type.
    scales = random_state.uniform(0.5, 2.0, size=batch_size)
    values_batch = jnp.asarray(values)[None, :] * jnp.asarray(scales)[:, None]
    right_hand_side_batch = jnp.tile(jnp.asarray(right_hand_side), (batch_size, 1))

    def solve_one(values_row):
        return pmj.solve(
            jnp.asarray(indptr),
            jnp.asarray(indices),
            values_row,
            jnp.asarray(right_hand_side),
            matrix_type=matrix_type,
        )

    batched = jax.vmap(solve_one)(values_batch)
    expected = _stacked_single_solves(
        indptr, indices, values_batch, right_hand_side_batch, matrix_type
    )
    np.testing.assert_allclose(np.asarray(batched), np.asarray(expected), rtol=1e-8, atol=1e-10)


def test_vmap_over_both_values_and_right_hand_side(system):
    matrix_type, indptr, indices, values, _dense, _right_hand_side = system
    dimension = indptr.shape[0] - 1
    batch_size = 3
    random_state = np.random.default_rng(13)
    scales = random_state.uniform(0.5, 2.0, size=batch_size)
    values_batch = jnp.asarray(values)[None, :] * jnp.asarray(scales)[:, None]
    right_hand_side_batch = jnp.asarray(
        random_state.uniform(-1.0, 1.0, size=(batch_size, dimension))
    )

    def solve_one(values_row, right_hand_side_row):
        return pmj.solve(
            jnp.asarray(indptr),
            jnp.asarray(indices),
            values_row,
            right_hand_side_row,
            matrix_type=matrix_type,
        )

    batched = jax.vmap(solve_one)(values_batch, right_hand_side_batch)
    expected = _stacked_single_solves(
        indptr, indices, values_batch, right_hand_side_batch, matrix_type
    )
    np.testing.assert_allclose(np.asarray(batched), np.asarray(expected), rtol=1e-8, atol=1e-10)


def test_vmap_over_pattern_arrays_is_rejected(system):
    matrix_type, indptr, indices, values, _dense, right_hand_side = system
    indices_batch = jnp.tile(jnp.asarray(indices), (2, 1))
    values_batch = jnp.tile(jnp.asarray(values), (2, 1))

    def solve_one(indices_row, values_row):
        return pmj.solve(
            jnp.asarray(indptr),
            indices_row,
            values_row,
            jnp.asarray(right_hand_side),
            matrix_type=matrix_type,
        )

    with pytest.raises(NotImplementedError, match="sparsity pattern"):
        jax.vmap(solve_one)(indices_batch, values_batch)
