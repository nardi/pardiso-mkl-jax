"""vmap correctness tests: batched right-hand sides, batched values, and both."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pardiso_mkl_jax as pmj
from pardiso_mkl_jax import primitive

MATRIX_TYPE = pmj.MatrixType.REAL_NONSYMMETRIC


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


# --- stateful vmap rules ---


def _analyze_factor(indptr, indices, values):
    """Analyze then factor, returning a ready-to-solve token."""
    token, _ = primitive.analyze(indptr, indices, values, matrix_type=MATRIX_TYPE)
    token, _, _ = primitive.factor(token, indptr, indices, values, matrix_type=MATRIX_TYPE)
    return token


def test_vmap_solve_stateful_over_right_hand_side(any_system):
    """Batching RHS through solve_stateful fuses into one native call."""
    indptr, indices, values, dense, _rhs = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    batch_size = 4
    rhs_batch = jnp.asarray(np.random.default_rng(20).uniform(-1, 1, (batch_size, dense.shape[0])))

    token = _analyze_factor(indptr, indices, values)

    def solve_one(b):
        sol, _, _ = primitive.solve_stateful(
            token, indptr, indices, values, b[None, :], matrix_type=MATRIX_TYPE
        )
        return sol[0]

    batched = jax.vmap(solve_one)(rhs_batch)
    expected = np.linalg.solve(dense, np.asarray(rhs_batch).T).T
    np.testing.assert_allclose(np.asarray(batched), expected, rtol=1e-8, atol=1e-10)


def test_vmap_factor_and_solve_stateful_over_right_hand_side(any_system):
    """Batching RHS through factor_and_solve_stateful fuses into one call."""
    indptr, indices, values, dense, _rhs = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    batch_size = 4
    rhs_batch = jnp.asarray(np.random.default_rng(21).uniform(-1, 1, (batch_size, dense.shape[0])))

    token, _ = primitive.analyze(indptr, indices, values, matrix_type=MATRIX_TYPE)

    def solve_one(b):
        sol, _, _ = primitive.factor_and_solve_stateful(
            token, indptr, indices, values, b[None, :], matrix_type=MATRIX_TYPE
        )
        return sol[0]

    batched = jax.vmap(solve_one)(rhs_batch)
    expected = np.linalg.solve(dense, np.asarray(rhs_batch).T).T
    np.testing.assert_allclose(np.asarray(batched), expected, rtol=1e-8, atol=1e-10)


def test_vmap_factor_and_solve_stateful_over_values(any_system):
    """Batching values through factor_and_solve_stateful refactors per element."""
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    right_hand_side = jnp.asarray(right_hand_side)
    batch_size = 3
    scales = np.random.default_rng(22).uniform(0.5, 2.0, size=batch_size)
    values_batch = values[None, :] * jnp.asarray(scales)[:, None]

    token, _ = primitive.analyze(indptr, indices, values, matrix_type=MATRIX_TYPE)

    def solve_one(v):
        sol, _, _ = primitive.factor_and_solve_stateful(
            token, indptr, indices, v, right_hand_side[None, :], matrix_type=MATRIX_TYPE
        )
        return sol[0]

    batched = jax.vmap(solve_one)(values_batch)
    for i in range(batch_size):
        scaled_dense = dense * scales[i]
        expected = np.linalg.solve(scaled_dense, np.asarray(right_hand_side))
        np.testing.assert_allclose(np.asarray(batched[i]), expected, rtol=1e-8, atol=1e-10)


def test_vmap_solve_stateful_rejects_batched_values(any_system):
    """solve_stateful rejects vmapping over values, pointing to factor_and_solve."""
    indptr, indices, values, _dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    values_batch = jnp.stack([values, values * 2.0])

    token = _analyze_factor(indptr, indices, values)
    stacked_rhs = jnp.asarray(right_hand_side)[None, :]

    def solve_one(v):
        sol, _, _ = primitive.solve_stateful(
            token, indptr, indices, v, stacked_rhs, matrix_type=MATRIX_TYPE
        )
        return sol[0]

    with pytest.raises(NotImplementedError, match="factor_and_solve_stateful"):
        jax.vmap(solve_one)(values_batch)


def test_vmap_stateful_rejects_batched_pattern(any_system):
    """Both stateful calls reject vmapping over sparsity pattern arrays."""
    indptr, indices, values, _dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    indices_batch = jnp.stack([indices, indices])
    values_batch = jnp.stack([values, values])

    token = _analyze_factor(indptr, indices, values)

    def solve_one(idx, v):
        sol, _, _ = primitive.solve_stateful(
            token, indptr, idx, v, jnp.asarray(right_hand_side)[None, :], matrix_type=MATRIX_TYPE
        )
        return sol[0]

    with pytest.raises(NotImplementedError, match="sparsity pattern"):
        jax.vmap(solve_one)(indices_batch, values_batch)


# --- factor vmap rule ---


def _reasons(reasons):
    return [primitive.RebuildReason(int(reason)) for reason in np.asarray(reasons)]


def test_vmap_factor_over_values(any_system):
    """Batching values through factor gives one token per element, all solvable."""
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    scales = jnp.asarray([21.0, 22.0, 23.0])
    values_batch = values[None, :] * scales[:, None]

    token, _ = primitive.analyze(indptr, indices, values * 21.0, matrix_type=MATRIX_TYPE)

    def factor_one(batch_values):
        return primitive.factor(token, indptr, indices, batch_values, matrix_type=MATRIX_TYPE)

    batched_token, final_iparm, reason = jax.vmap(factor_one)(values_batch)
    assert batched_token.id.shape == (3,)
    assert final_iparm.shape == (3, 64)
    assert _reasons(reason) == [primitive.RebuildReason.NONE] * 3

    # Every element token solves correctly. Tokens of earlier elements were
    # re-keyed by the chain, so they rebuild and report SUPERSEDED.
    for index, scale in enumerate(np.asarray(scales)):
        element_token = primitive.FactorizationToken(
            batched_token.id[index], batched_token.n_dependent_solutions
        )
        solution, _, _ = primitive.solve_stateful(
            element_token,
            indptr,
            indices,
            values_batch[index],
            jnp.asarray(right_hand_side)[None, :],
            matrix_type=MATRIX_TYPE,
        )
        np.testing.assert_allclose(
            np.asarray(solution[0]),
            np.linalg.solve(dense * scale, right_hand_side),
            rtol=1e-8,
            atol=1e-10,
        )


def test_vmap_factor_rebuilds_missing_analysis_once(any_system):
    """A chain rebuilds a missing analysis for the first element and not for the others."""
    indptr, indices, values, _dense, _right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    token, _ = primitive.analyze(indptr, indices, values * 24.0, matrix_type=MATRIX_TYPE)
    primitive.release(token)
    values_batch = values[None, :] * jnp.asarray([25.0, 26.0, 27.0])[:, None]

    primitive.reset_rebuild_count()
    _, _, reason = jax.vmap(
        lambda batch_values: primitive.factor(
            token, indptr, indices, batch_values, matrix_type=MATRIX_TYPE
        )
    )(values_batch)
    assert _reasons(reason) == [
        primitive.RebuildReason.FREED,
        primitive.RebuildReason.NONE,
        primitive.RebuildReason.NONE,
    ]
    assert primitive.rebuild_count() == 1


def test_vmap_factor_with_batched_token(any_system):
    """A batched token gives each element its own input, so there is no chain."""
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    scales = [28.0, 29.0]
    tokens = [
        primitive.analyze(indptr, indices, values * scale, matrix_type=MATRIX_TYPE)[0]
        for scale in scales
    ]
    batched_token = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *tokens)
    values_batch = values[None, :] * jnp.asarray(scales)[:, None]

    new_token, _, reason = jax.vmap(
        lambda token, batch_values: primitive.factor(
            token, indptr, indices, batch_values, matrix_type=MATRIX_TYPE
        )
    )(batched_token, values_batch)
    assert _reasons(reason) == [primitive.RebuildReason.NONE] * 2
    for index, scale in enumerate(scales):
        element_token = primitive.FactorizationToken(
            new_token.id[index], new_token.n_dependent_solutions
        )
        solution, _, solve_reason = primitive.solve_stateful(
            element_token,
            indptr,
            indices,
            values_batch[index],
            jnp.asarray(right_hand_side)[None, :],
            matrix_type=MATRIX_TYPE,
        )
        # Both elements own a separate analysis, so both factorizations are resident.
        assert primitive.RebuildReason(int(solve_reason)) == primitive.RebuildReason.NONE
        np.testing.assert_allclose(
            np.asarray(solution[0]),
            np.linalg.solve(dense * scale, right_hand_side),
            rtol=1e-8,
            atol=1e-10,
        )


def test_vmap_factor_with_only_right_hand_side_batched(any_system):
    """A batched right-hand side that flows through a refactoring loop needs no batched factor."""
    indptr, indices, values, dense, _right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    batch_size = 3
    rhs_batch = jnp.asarray(np.random.default_rng(30).uniform(-1, 1, (batch_size, dense.shape[0])))
    token, _ = primitive.analyze(indptr, indices, values, matrix_type=MATRIX_TYPE)

    def refactor_loop(right_hand_side):
        def body(carry, _):
            loop_values = values * 1.5
            new_token, _, _ = primitive.factor(
                token, indptr, indices, loop_values, matrix_type=MATRIX_TYPE
            )
            solution, _, _ = primitive.solve_stateful(
                new_token,
                indptr,
                indices,
                loop_values,
                (carry)[None, :],
                matrix_type=MATRIX_TYPE,
            )
            return solution[0], None

        result, _ = jax.lax.scan(body, right_hand_side, None, length=2)
        return result

    batched = jax.vmap(refactor_loop)(rhs_batch)
    expected = np.linalg.solve(dense * 1.5, np.linalg.solve(dense * 1.5, np.asarray(rhs_batch).T))
    np.testing.assert_allclose(np.asarray(batched), expected.T, rtol=1e-8, atol=1e-10)


def test_vmap_factor_rejects_batched_pattern(any_system):
    """Batching indptr or indices through factor raises the shared pattern error."""
    indptr, indices, values, _dense, _right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    token, _ = primitive.analyze(indptr, indices, values, matrix_type=MATRIX_TYPE)
    indices_batch = jnp.stack([indices, indices])

    with pytest.raises(NotImplementedError, match="sparsity pattern"):
        jax.vmap(
            lambda batch_indices: primitive.factor(
                token, indptr, batch_indices, values, matrix_type=MATRIX_TYPE
            )
        )(indices_batch)
