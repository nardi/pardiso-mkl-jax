"""Tests for the iparm overlay (PardisoOption, options=) and diagnostics readback."""

from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pardiso_mkl_jax as pmj
from pardiso_mkl_jax.iparm import PardisoOption, canonicalize_overlay
from pardiso_mkl_jax.primitive import _make_solve_core


def test_overlay_applies_and_reflects_in_diagnostics(system):
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    solution, diagnostics = pmj.solve(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        jnp.asarray(values),
        jnp.asarray(right_hand_side),
        matrix_type=matrix_type,
        options={PardisoOption.FILL_IN_REDUCING_ORDERING: 0},
        return_diagnostics=True,
    )
    expected = np.linalg.solve(dense, right_hand_side)
    np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)
    assert int(diagnostics.raw[PardisoOption.FILL_IN_REDUCING_ORDERING]) == 0


def test_enum_key_and_raw_int_key_are_equivalent(any_system):
    assert canonicalize_overlay({PardisoOption.SCALING: 0}) == canonicalize_overlay({10: 0})

    indptr, indices, values, _dense, right_hand_side = any_system
    _make_solve_core.cache_clear()
    pmj.solve(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        jnp.asarray(values),
        jnp.asarray(right_hand_side),
        matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
        options={PardisoOption.SCALING: 0},
    )
    pmj.solve(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        jnp.asarray(values),
        jnp.asarray(right_hand_side),
        matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
        options={10: 0},
    )
    cache_info = _make_solve_core.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 1


def test_reserved_and_output_only_indices_are_rejected():
    with pytest.raises(ValueError, match="reserved"):
        canonicalize_overlay({2: 0})
    with pytest.raises(ValueError, match="output-only"):
        canonicalize_overlay({6: 0})


@pytest.mark.parametrize(
    "index",
    [
        PardisoOption.USER_PERMUTATION,
        PardisoOption.PARTIAL_SOLVE_CONTROL,
        PardisoOption.SCHUR_COMPLEMENT_CONTROL,
    ],
    ids=lambda option: option.name,
)
def test_guarded_indices_reject_nonzero_but_allow_zero(index):
    with pytest.raises(ValueError, match="cannot be enabled"):
        canonicalize_overlay({index: 1})
    # The default value is a harmless no-op overlay entry.
    assert canonicalize_overlay({index: 0}) == ((int(index), 0),)


def test_transpose_solve_option_is_always_rejected():
    with pytest.raises(ValueError, match="transpose"):
        canonicalize_overlay({PardisoOption.TRANSPOSE_SOLVE: 0})
    with pytest.raises(ValueError, match="transpose"):
        canonicalize_overlay({PardisoOption.TRANSPOSE_SOLVE: 2})


def test_indexing_style_warns_but_does_not_raise():
    with pytest.warns(UserWarning, match="INDEXING_STYLE"):
        result = canonicalize_overlay({PardisoOption.INDEXING_STYLE: 1})
    assert result == ((PardisoOption.INDEXING_STYLE, 1),)


def test_use_default_values_and_weighted_matching_are_silent():
    # Neither raises nor warns: both are deliberately left to the caller's
    # judgment, per the "Solver settings" docs. Checked only at the
    # canonicalize_overlay level, never through an actual native call: a
    # real solve with USE_DEFAULT_VALUES=0 is exactly the segfault this
    # package otherwise works around.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        canonicalize_overlay({PardisoOption.USE_DEFAULT_VALUES: 0})
        canonicalize_overlay({PardisoOption.WEIGHTED_MATCHING: 1})
    assert caught == []


def test_overlay_changes_analyze_behavior(system):
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=matrix_type
    ) as solver:
        # Minimum degree instead of the default METIS nested dissection: a
        # genuinely different Pardiso code path, still expected to solve
        # correctly.
        solver.analyze(jnp.asarray(values), options={PardisoOption.FILL_IN_REDUCING_ORDERING: 0})
        assert solver.last_diagnostics is not None
        assert int(solver.last_diagnostics.raw[PardisoOption.FILL_IN_REDUCING_ORDERING]) == 0
        solver.factorize(jnp.asarray(values))
        solution = solver.solve(jnp.asarray(right_hand_side))
    expected = np.linalg.solve(dense, right_hand_side)
    np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)


def test_diagnostics_eigenvalue_counts_sum_to_dimension():
    from conftest import build_system

    matrix_type = pmj.MatrixType.REAL_SYMMETRIC_INDEFINITE
    indptr, indices, values, dense, _right_hand_side = build_system(matrix_type)
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=matrix_type
    ) as solver:
        solver.analyze(jnp.asarray(values))
        solver.factorize(jnp.asarray(values))
        diagnostics = solver.last_diagnostics
    assert diagnostics is not None
    positive = int(diagnostics.positive_eigenvalues)
    negative = int(diagnostics.negative_eigenvalues)
    assert positive > 0
    assert negative > 0
    assert positive + negative == dense.shape[0]


def test_return_diagnostics_false_matches_default_return_type(system):
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    plain = pmj.solve(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        jnp.asarray(values),
        jnp.asarray(right_hand_side),
        matrix_type=matrix_type,
    )
    explicit_false = pmj.solve(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        jnp.asarray(values),
        jnp.asarray(right_hand_side),
        matrix_type=matrix_type,
        return_diagnostics=False,
    )
    assert not isinstance(plain, tuple)
    assert not isinstance(explicit_false, tuple)
    np.testing.assert_allclose(np.asarray(plain), np.asarray(explicit_false))


def test_diagnostics_under_jit(system):
    matrix_type, indptr, indices, values, dense, right_hand_side = system

    jit_solve = functools.partial(
        jax.jit(pmj.solve, static_argnames=("matrix_type", "return_diagnostics")),
        matrix_type=matrix_type,
        return_diagnostics=True,
    )
    jit_solution, jit_diagnostics = jit_solve(
        jnp.asarray(indptr), jnp.asarray(indices), jnp.asarray(values), jnp.asarray(right_hand_side)
    )
    eager_solution, eager_diagnostics = pmj.solve(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        jnp.asarray(values),
        jnp.asarray(right_hand_side),
        matrix_type=matrix_type,
        return_diagnostics=True,
    )
    np.testing.assert_allclose(np.asarray(jit_solution), np.asarray(eager_solution), rtol=1e-10)
    np.testing.assert_array_equal(
        np.asarray(jit_diagnostics.raw), np.asarray(eager_diagnostics.raw)
    )


def test_diagnostics_under_vmap_right_hand_side_batched(system):
    matrix_type, indptr, indices, values, dense, _right_hand_side = system
    random_state = np.random.default_rng(23)
    right_hand_side_batch = random_state.uniform(-1.0, 1.0, size=(3, dense.shape[0]))

    def solve_one(right_hand_side):
        return pmj.solve(
            jnp.asarray(indptr),
            jnp.asarray(indices),
            jnp.asarray(values),
            right_hand_side,
            matrix_type=matrix_type,
            return_diagnostics=True,
        )

    solutions, diagnostics = jax.vmap(solve_one)(jnp.asarray(right_hand_side_batch))
    assert diagnostics.raw.shape == (3, 64)
    # One native call handled the whole batch, so every row is the same
    # broadcast value, not independently computed.
    for row in range(1, 3):
        np.testing.assert_array_equal(
            np.asarray(diagnostics.raw[0]), np.asarray(diagnostics.raw[row])
        )
    for index in range(3):
        expected = np.linalg.solve(dense, right_hand_side_batch[index])
        np.testing.assert_allclose(np.asarray(solutions[index]), expected, rtol=1e-8, atol=1e-10)

    # The compact, un-broadcast form is available through jax.vmap's own
    # out_axes, for callers who want it.
    _solutions, compact_diagnostics = jax.vmap(solve_one, out_axes=(0, None))(
        jnp.asarray(right_hand_side_batch)
    )
    assert compact_diagnostics.raw.shape == (64,)


def test_diagnostics_under_vmap_values_batched(system):
    matrix_type, indptr, indices, values, _dense, right_hand_side = system
    random_state = np.random.default_rng(29)
    scales = random_state.uniform(0.5, 2.0, size=3)
    values_batch = jnp.asarray(values)[None, :] * jnp.asarray(scales)[:, None]

    def solve_one(values_row):
        return pmj.solve(
            jnp.asarray(indptr),
            jnp.asarray(indices),
            values_row,
            jnp.asarray(right_hand_side),
            matrix_type=matrix_type,
            return_diagnostics=True,
        )

    _solutions, diagnostics = jax.vmap(solve_one)(values_batch)
    assert diagnostics.raw.shape == (3, 64)
    # Each batch element was factorized separately, so its diagnostics can
    # genuinely differ (memory figures in particular need not match exactly
    # across factorizations even on the same pattern).
    assert diagnostics.raw.shape[0] == 3


def test_last_diagnostics_is_none_after_a_jitted_call():
    """A traced call leaves last_diagnostics unset, and return_diagnostics still works.

    Under jit the diagnostics only exist as tracers. Storing one would leak it
    out of its trace, so reading it back later would raise instead of
    returning anything usable.
    """
    from conftest import build_system

    matrix_type = pmj.MatrixType.REAL_NONSYMMETRIC
    indptr, indices, values, dense, right_hand_side = build_system(matrix_type)
    expected = np.linalg.solve(dense, right_hand_side)

    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=matrix_type
    ) as solver:
        solver.analyze(jnp.asarray(values))
        eager_diagnostics = solver.factorize(jnp.asarray(values), return_diagnostics=True)
        assert solver.last_diagnostics is not None

        jit_solve = jax.jit(solver.solve)
        solution = jit_solve(jnp.asarray(right_hand_side))
        np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)
        assert solver.last_diagnostics is None

        # The returned form is the one that survives tracing, and it agrees
        # with what the same call reports eagerly.
        jit_solve_with_diagnostics = jax.jit(lambda rhs: solver.solve(rhs, return_diagnostics=True))
        solution, diagnostics = jit_solve_with_diagnostics(jnp.asarray(right_hand_side))
        np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)
        assert int(diagnostics.perturbed_pivot_count) == int(
            eager_diagnostics.perturbed_pivot_count
        )
        assert solver.last_diagnostics is None
