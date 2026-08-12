"""Tests for the iparm overlay (PardisoOption, options=) and diagnostics readback."""

from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pardiso_mkl_jax as pmj
from pardiso_mkl_jax.iparm import PardisoOption, canonicalize_overlay, merge_overlays
from pardiso_mkl_jax.primitive import _make_solve_core, default_iparm


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


@pytest.mark.parametrize("batch_values", [False, True], ids=["rhs_batched", "values_batched"])
def test_overlay_reaches_both_vmap_branches(any_system, batch_values):
    """An overlay survives batching down either vmap path.

    The two branches take different routes into Pardiso: batched right-hand
    sides go through one multi-rhs call, batched values through an analyze
    plus a factor/solve per element. Both have to carry the overlay.
    """
    indptr, indices, values, _dense, right_hand_side = any_system
    indptr = jnp.asarray(indptr)
    indices = jnp.asarray(indices)
    options: dict[int, int] = {PardisoOption.FILL_IN_REDUCING_ORDERING: 0}

    run = functools.partial(
        pmj.solve,
        matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
        options=options,
        return_diagnostics=True,
    )
    if batch_values:
        batched_values = jnp.stack([jnp.asarray(values) * scale for scale in (1.0, 2.0, 3.0)])
        _solution, diagnostics = jax.vmap(
            lambda v: run(indptr, indices, v, jnp.asarray(right_hand_side))
        )(batched_values)
    else:
        batched_right_hand_side = jnp.stack(
            [jnp.asarray(right_hand_side) * scale for scale in (1.0, 2.0, 3.0)]
        )
        _solution, diagnostics = jax.vmap(lambda b: run(indptr, indices, jnp.asarray(values), b))(
            batched_right_hand_side
        )

    ordering = np.asarray(diagnostics.raw)[..., PardisoOption.FILL_IN_REDUCING_ORDERING]
    assert (ordering == 0).all()


def test_default_iparm_matches_the_documented_defaults():
    """The defaults read out of the native layer are the ones the docs describe."""
    nonsymmetric = default_iparm(pmj.MatrixType.REAL_NONSYMMETRIC)
    assert int(nonsymmetric[PardisoOption.USE_DEFAULT_VALUES]) == 1
    assert int(nonsymmetric[PardisoOption.FILL_IN_REDUCING_ORDERING]) == 2
    assert int(nonsymmetric[PardisoOption.PIVOTING_PERTURBATION]) == 13
    assert int(nonsymmetric[PardisoOption.SCALING]) == 1
    assert int(nonsymmetric[PardisoOption.WEIGHTED_MATCHING]) == 1
    assert int(nonsymmetric[PardisoOption.INDEXING_STYLE]) == 1

    # Scaling and matching are non-symmetric only, and Bunch-Kaufman pivoting
    # is the symmetric indefinite counterpart.
    symmetric_indefinite = default_iparm(pmj.MatrixType.REAL_SYMMETRIC_INDEFINITE)
    assert int(symmetric_indefinite[PardisoOption.SCALING]) == 0
    assert int(symmetric_indefinite[PardisoOption.WEIGHTED_MATCHING]) == 0
    assert int(symmetric_indefinite[PardisoOption.PIVOTING_STRATEGY]) == 1


def test_default_iparm_is_read_only():
    """The cached array is shared, so a caller must not be able to mutate it."""
    defaults = default_iparm(pmj.MatrixType.REAL_NONSYMMETRIC)
    with pytest.raises(ValueError):
        defaults[0] = 99


def test_merge_overlays_lets_the_override_win():
    base: dict[int, int] = {PardisoOption.SCALING: 0, PardisoOption.WEIGHTED_MATCHING: 0}
    override: dict[int, int] = {PardisoOption.SCALING: 1, PardisoOption.MATRIX_CHECKER: 1}
    assert merge_overlays(base, override) == (
        (PardisoOption.SCALING, 1),
        (PardisoOption.WEIGHTED_MATCHING, 0),
        (PardisoOption.MATRIX_CHECKER, 1),
    )
    # Either side may be empty, and both sides are still validated.
    assert merge_overlays(None, base) == canonicalize_overlay(base)
    assert merge_overlays(base, None) == canonicalize_overlay(base)
    with pytest.raises(ValueError, match="reserved"):
        merge_overlays({2: 1}, None)


def test_solver_wide_options_apply_to_every_phase(any_system):
    """The constructor overlay reaches analyze, factorize, and solve alike.

    This is what a per-call overlay cannot do: each native call rebuilds iparm
    from the package defaults, so an entry passed only to analyze() is gone by
    the time factorize() runs.
    """
    indptr, indices, values, _dense, right_hand_side = any_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
        options={PardisoOption.WEIGHTED_MATCHING: 0, PardisoOption.SCALING: 0},
    ) as solver:
        analyze_diagnostics = solver.analyze(jnp.asarray(values), return_diagnostics=True)
        factorize_diagnostics = solver.factorize(jnp.asarray(values), return_diagnostics=True)
        _solution, solve_diagnostics = solver.solve(
            jnp.asarray(right_hand_side), return_diagnostics=True
        )

    for diagnostics in (analyze_diagnostics, factorize_diagnostics, solve_diagnostics):
        assert int(diagnostics.raw[PardisoOption.WEIGHTED_MATCHING]) == 0
        assert int(diagnostics.raw[PardisoOption.SCALING]) == 0


def test_per_call_options_override_the_solver_wide_ones(any_system):
    indptr, indices, values, _dense, _right_hand_side = any_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
        options={PardisoOption.MAX_ITERATIVE_REFINEMENT_STEPS: 2},
    ) as solver:
        solver.analyze(jnp.asarray(values))
        overridden = solver.factorize(
            jnp.asarray(values),
            options={PardisoOption.MAX_ITERATIVE_REFINEMENT_STEPS: 5},
            return_diagnostics=True,
        )
        assert int(overridden.raw[PardisoOption.MAX_ITERATIVE_REFINEMENT_STEPS]) == 5

        # The override applied to that call only; the solver-wide value is back.
        restored = solver.refactorize(jnp.asarray(values), return_diagnostics=True)
        assert int(restored.raw[PardisoOption.MAX_ITERATIVE_REFINEMENT_STEPS]) == 2


@pytest.mark.parametrize("option", [PardisoOption.SCALING, PardisoOption.WEIGHTED_MATCHING])
def test_scaling_or_matching_disagreeing_with_the_analysis_is_rejected(any_system, option):
    """Pardiso expects these unchanged after analysis, and nothing native enforces it."""
    indptr, indices, values, _dense, right_hand_side = any_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    ) as solver:
        solver.analyze(jnp.asarray(values), options={option: 0})

        # The default for this matrix type is 1, so a plain call disagrees.
        with pytest.raises(ValueError, match=option.name):
            solver.factorize(jnp.asarray(values))

        # Passing the same value again agrees, and unblocks the later phases.
        solver.factorize(jnp.asarray(values), options={option: 0})
        with pytest.raises(ValueError, match=option.name):
            solver.solve(jnp.asarray(right_hand_side))
        solver.solve(jnp.asarray(right_hand_side), options={option: 0})
        with pytest.raises(ValueError, match=option.name):
            solver.refactorize(jnp.asarray(values))
        with pytest.raises(ValueError, match=option.name):
            solver.refactor_and_solve(jnp.asarray(values), jnp.asarray(right_hand_side))


def test_the_mismatch_guard_raises_at_trace_time(any_system):
    """The guard compares static values, so it fires while tracing rather than at runtime."""
    indptr, indices, values, _dense, right_hand_side = any_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    ) as solver:
        solver.analyze(jnp.asarray(values), options={PardisoOption.SCALING: 0})
        with pytest.raises(ValueError, match="SCALING"):
            jax.jit(solver.refactor_and_solve)(jnp.asarray(values), jnp.asarray(right_hand_side))


def test_re_analysis_adopts_the_new_scaling_and_matching(any_system):
    """Re-analyzing with a new overlay moves the baseline the guard checks against."""
    indptr, indices, values, _dense, _right_hand_side = any_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    ) as solver:
        solver.analyze(jnp.asarray(values))
        solver.factorize(jnp.asarray(values))

        solver.analyze(jnp.asarray(values), options={PardisoOption.WEIGHTED_MATCHING: 0})
        with pytest.raises(ValueError, match="WEIGHTED_MATCHING"):
            solver.factorize(jnp.asarray(values))
        solver.factorize(jnp.asarray(values), options={PardisoOption.WEIGHTED_MATCHING: 0})


def test_factorize_reports_perturbed_pivots(zero_diagonal_system):
    """perturbed_pivot_count is Pardiso's own report that it could not pivot cleanly.

    The zero-diagonal saddle-point matrix needs weighted matching to factor
    stably. With matching off, Pardiso perturbs the tiny pivots it finds
    instead and reports how many, which is the signal a caller uses to fall
    back rather than trusting the solution.
    """
    indptr, indices, values, _dense, _right_hand_side = zero_diagonal_system

    with pmj.PardisoSolver(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
        options={PardisoOption.WEIGHTED_MATCHING: 0},
    ) as solver:
        solver.analyze(jnp.asarray(values))
        diagnostics = solver.factorize(jnp.asarray(values), return_diagnostics=True)
    assert int(diagnostics.perturbed_pivot_count) > 0

    # With the package default (matching on) the same matrix factors cleanly,
    # so the count is reporting something real rather than always tripping.
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    ) as solver:
        solver.analyze(jnp.asarray(values))
        diagnostics = solver.factorize(jnp.asarray(values), return_diagnostics=True)
    assert int(diagnostics.perturbed_pivot_count) == 0


def test_return_diagnostics_agrees_with_last_diagnostics(any_system):
    indptr, indices, values, _dense, right_hand_side = any_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    ) as solver:

        def check(returned):
            """Every method's returned diagnostics match the ones it recorded."""
            recorded = solver.last_diagnostics
            assert recorded is not None
            np.testing.assert_array_equal(returned.raw, recorded.raw)
            assert int(returned.perturbed_pivot_count) == int(recorded.perturbed_pivot_count)

        check(solver.analyze(jnp.asarray(values), return_diagnostics=True))
        check(solver.factorize(jnp.asarray(values), return_diagnostics=True))
        check(solver.refactorize(jnp.asarray(values), return_diagnostics=True))
        _solution, returned = solver.solve(jnp.asarray(right_hand_side), return_diagnostics=True)
        check(returned)


def test_return_diagnostics_defaults_to_the_previous_return_types(any_system):
    """Leaving return_diagnostics off keeps every method's original return value."""
    indptr, indices, values, _dense, right_hand_side = any_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    ) as solver:
        assert solver.analyze(jnp.asarray(values)) is None
        assert solver.factorize(jnp.asarray(values)) is None
        assert solver.refactorize(jnp.asarray(values)) is None
        solution = solver.solve(jnp.asarray(right_hand_side))
        assert solution.shape == right_hand_side.shape


def test_solver_wide_options_survive_a_jitted_solve(any_system):
    indptr, indices, values, _dense, right_hand_side = any_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
        options={PardisoOption.WEIGHTED_MATCHING: 0, PardisoOption.SCALING: 0},
    ) as solver:
        solver.analyze(jnp.asarray(values))
        solver.factorize(jnp.asarray(values))
        _solution, eager = solver.solve(jnp.asarray(right_hand_side), return_diagnostics=True)
        _solution, traced = jax.jit(lambda rhs: solver.solve(rhs, return_diagnostics=True))(
            jnp.asarray(right_hand_side)
        )
    np.testing.assert_array_equal(np.asarray(eager.raw), np.asarray(traced.raw))
    assert int(traced.raw[PardisoOption.WEIGHTED_MATCHING]) == 0
    assert int(traced.raw[PardisoOption.SCALING]) == 0
