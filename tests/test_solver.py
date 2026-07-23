"""Tests for PardisoSolver: factorization reuse, context manager enforcement,
and the separation between analyze, factorize, refactorize, and solve.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import pardiso_mkl_jax as pmj
from pardiso_mkl_jax import _ffi


def test_solve_matches_dense_reference(system):
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=matrix_type
    ) as solver:
        solver.analyze(jnp.asarray(values))
        solver.factorize(jnp.asarray(values))
        solution = solver.solve(jnp.asarray(right_hand_side))
    expected = np.linalg.solve(dense, right_hand_side)
    np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)


def test_solve_reused_across_many_right_hand_sides(system):
    matrix_type, indptr, indices, values, dense, _right_hand_side = system
    random_state = np.random.default_rng(42)
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=matrix_type
    ) as solver:
        solver.analyze(jnp.asarray(values))
        solver.factorize(jnp.asarray(values))
        for _ in range(5):
            right_hand_side = random_state.uniform(-1.0, 1.0, size=dense.shape[0])
            solution = solver.solve(jnp.asarray(right_hand_side))
            expected = np.linalg.solve(dense, right_hand_side)
            np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)


def test_solve_transpose_reuses_factorization(system):
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=matrix_type
    ) as solver:
        solver.analyze(jnp.asarray(values))
        solver.factorize(jnp.asarray(values))
        assert _ffi.analysis_count(solver._solver_id) == 1

        # Alternating transpose and non-transpose solves on the same
        # factorization must each give the right answer, with no
        # re-analysis and no state left over from the previous call.
        non_transpose = solver.solve(jnp.asarray(right_hand_side))
        transpose = solver.solve(jnp.asarray(right_hand_side), transpose=True)
        non_transpose_again = solver.solve(jnp.asarray(right_hand_side))
        assert _ffi.analysis_count(solver._solver_id) == 1

    np.testing.assert_allclose(
        np.asarray(non_transpose), np.linalg.solve(dense, right_hand_side), rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(transpose), np.linalg.solve(dense.T, right_hand_side), rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(non_transpose_again),
        np.linalg.solve(dense, right_hand_side),
        rtol=1e-8,
        atol=1e-10,
    )


def test_refactorize_updates_values_without_reanalyzing(system):
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=matrix_type
    ) as solver:
        solver.analyze(jnp.asarray(values))
        assert _ffi.analysis_count(solver._solver_id) == 1

        solver.factorize(jnp.asarray(values))
        assert _ffi.analysis_count(solver._solver_id) == 1

        new_values = values * 2.0
        solver.refactorize(jnp.asarray(new_values))
        assert _ffi.analysis_count(solver._solver_id) == 1

        solution = solver.solve(jnp.asarray(right_hand_side))
        expected = np.linalg.solve(dense * 2.0, right_hand_side)
        np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)


def test_many_create_close_cycles_do_not_leak(system):
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    expected = np.linalg.solve(dense, right_hand_side)
    for _ in range(20):
        with pmj.PardisoSolver(
            jnp.asarray(indptr), jnp.asarray(indices), matrix_type=matrix_type
        ) as solver:
            solver.analyze(jnp.asarray(values))
            solver.factorize(jnp.asarray(values))
            solution = solver.solve(jnp.asarray(right_hand_side))
        np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)


# The lifecycle and precondition checks below (context manager enforcement,
# method ordering, idempotent close) do not depend on the matrix type: they
# are raised before Pardiso ever sees the values, so they run once against
# a single representative system rather than once per matrix type.


def test_methods_require_context_manager(any_system):
    indptr, indices, values, _dense, _right_hand_side = any_system
    solver = pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    )
    with pytest.raises(RuntimeError, match="context manager"):
        solver.analyze(jnp.asarray(values))


def test_methods_reject_use_after_close(any_system):
    indptr, indices, values, _dense, _right_hand_side = any_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    ) as solver:
        solver.analyze(jnp.asarray(values))
    with pytest.raises(RuntimeError, match="closed"):
        solver.factorize(jnp.asarray(values))


def test_factorize_requires_prior_analyze(any_system):
    indptr, indices, values, _dense, _right_hand_side = any_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    ) as solver:
        with pytest.raises(RuntimeError, match="analyze"):
            solver.factorize(jnp.asarray(values))


def test_refactorize_requires_prior_factorize(any_system):
    indptr, indices, values, _dense, _right_hand_side = any_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    ) as solver:
        solver.analyze(jnp.asarray(values))
        with pytest.raises(RuntimeError, match="factorize"):
            solver.refactorize(jnp.asarray(values))


def test_solve_requires_prior_factorize(any_system):
    indptr, indices, values, _dense, right_hand_side = any_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    ) as solver:
        solver.analyze(jnp.asarray(values))
        with pytest.raises(RuntimeError, match="factorize"):
            solver.solve(jnp.asarray(right_hand_side))


def test_close_is_idempotent(any_system):
    indptr, indices, values, _dense, _right_hand_side = any_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    ) as solver:
        solver.analyze(jnp.asarray(values))
    # __exit__ already closed it; closing again must not raise.
    solver.close()
    solver.close()
