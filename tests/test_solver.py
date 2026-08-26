"""Tests for PardisoSolver: factorization reuse, context manager enforcement,
and the separation between analyze, factorize, refactorize, and solve.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pardiso_mkl_jax as pmj
from pardiso_mkl_jax import _ffi, primitive


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


def test_solve_matches_dense_reference_with_zeros_on_the_diagonal(zero_diagonal_system):
    # The stateful counterpart to the same-named regression test in
    # test_solve.py: analyze, factorize, and solve are separate native calls
    # here, each re-initializing iparm, so this also pins that the matching
    # setting survives being applied three times rather than once.
    indptr, indices, values, dense, right_hand_side = zero_diagonal_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
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
        assert _ffi.analysis_count(handle_value(solver)) == 1

        # Alternating transpose and non-transpose solves on the same
        # factorization must each give the right answer, with no
        # re-analysis and no state left over from the previous call.
        non_transpose = solver.solve(jnp.asarray(right_hand_side))
        transpose = solver.solve(jnp.asarray(right_hand_side), transpose=True)
        non_transpose_again = solver.solve(jnp.asarray(right_hand_side))
        assert _ffi.analysis_count(handle_value(solver)) == 1

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
        assert _ffi.analysis_count(handle_value(solver)) == 1

        solver.factorize(jnp.asarray(values))
        assert _ffi.analysis_count(handle_value(solver)) == 1

        new_values = values * 2.0
        solver.refactorize(jnp.asarray(new_values))
        assert _ffi.analysis_count(handle_value(solver)) == 1

        solution = solver.solve(jnp.asarray(right_hand_side))
        expected = np.linalg.solve(dense * 2.0, right_hand_side)
        np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)


def test_refactor_and_solve_reuses_analysis_under_jit(system):
    """refactor_and_solve runs inside jit and reuses the analysis across value changes.

    This is the path that factorize() plus solve() cannot take: the latter stores the
    values on the solver, which leaks a tracer out of the jit. refactor_and_solve passes
    them explicitly, so a jitted call with fresh values reuses the analysis and stays
    correct without re-analyzing.
    """
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=matrix_type
    ) as solver:
        solver.analyze(jnp.asarray(values))
        assert _ffi.analysis_count(handle_value(solver)) == 1

        run = jax.jit(lambda v, b: solver.refactor_and_solve(v, b))
        first = run(jnp.asarray(values), jnp.asarray(right_hand_side))
        second = run(jnp.asarray(values * 2.0), jnp.asarray(right_hand_side))
        assert _ffi.analysis_count(handle_value(solver)) == 1

    np.testing.assert_allclose(
        np.asarray(first), np.linalg.solve(dense, right_hand_side), rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(second),
        np.linalg.solve(dense * 2.0, right_hand_side),
        rtol=1e-8,
        atol=1e-10,
    )


def test_refactor_and_solve_transpose_matches_dense(system):
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=matrix_type
    ) as solver:
        solver.analyze(jnp.asarray(values))
        solution = solver.refactor_and_solve(
            jnp.asarray(values), jnp.asarray(right_hand_side), transpose=True
        )
    expected = np.linalg.solve(dense.T, right_hand_side)
    np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)


def test_refactor_and_solve_requires_prior_analyze(any_system):
    indptr, indices, values, _dense, right_hand_side = any_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    ) as solver:
        with pytest.raises(RuntimeError, match="analyze"):
            solver.refactor_and_solve(jnp.asarray(values), jnp.asarray(right_hand_side))


def test_whole_lifecycle_inside_jit_reuses_analysis(system):
    """analyze, factor_and_solve (twice), and release all run inside one jit trace.

    This is the scenario the handle redesign exists for: with the factorization
    identified by a JAX value rather than a Python-side id, XLA can order the
    entire lifecycle by data dependency, so it no longer needs to start outside
    the trace the way PardisoSolver.analyze() does.
    """
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    indptr = jnp.asarray(indptr)
    indices = jnp.asarray(indices)

    def run(values, other_values, right_hand_side):
        handle, _iparm = primitive.analyze(indptr, indices, values, matrix_type=matrix_type)
        first, _iparm = primitive.factor_and_solve_stateful(
            handle, indptr, indices, values, right_hand_side[None, :], matrix_type=matrix_type
        )
        second, _iparm = primitive.factor_and_solve_stateful(
            handle,
            indptr,
            indices,
            other_values,
            right_hand_side[None, :],
            matrix_type=matrix_type,
        )
        # release() and the two solves above all consume the token's id
        # directly, so nothing otherwise orders release after them. Tracking
        # both solutions ties the release to them so it runs last.
        status = primitive.release(handle.track(first, second))
        return first[0], second[0], status

    first, second, _status = jax.jit(run)(
        jnp.asarray(values), jnp.asarray(values * 2.0), jnp.asarray(right_hand_side)
    )
    np.testing.assert_allclose(
        np.asarray(first), np.linalg.solve(dense, right_hand_side), rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(second), np.linalg.solve(dense * 2.0, right_hand_side), rtol=1e-8, atol=1e-10
    )


def test_whole_lifecycle_inside_jit_does_not_leak_handles(system):
    """Each traced analyze/release pair frees its native state, even run many times."""
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    indptr = jnp.asarray(indptr)
    indices = jnp.asarray(indices)
    values = jnp.asarray(values)
    right_hand_side = jnp.asarray(right_hand_side)

    def run(values, right_hand_side):
        handle, _iparm = primitive.analyze(indptr, indices, values, matrix_type=matrix_type)
        solution, _iparm = primitive.factor_and_solve_stateful(
            handle, indptr, indices, values, right_hand_side[None, :], matrix_type=matrix_type
        )
        # Tracking the solution forces release() to run after the solve above,
        # for the same reason as in test_whole_lifecycle_inside_jit_reuses_analysis.
        primitive.release(handle.track(solution))
        return solution[0], handle

    run_jit = jax.jit(run)
    handles_seen = set()
    for _ in range(10):
        solution, handle = run_jit(values, right_hand_side)
        handles_seen.add(int(handle.id))
        np.testing.assert_allclose(
            np.asarray(solution), np.linalg.solve(dense, right_hand_side), rtol=1e-8, atol=1e-10
        )
        # release() already ran inside the trace, so the registry entry for
        # this handle must be gone: analysis_count falls back to 0 for a
        # handle it does not recognize.
        assert _ffi.analysis_count(handle.id) == 0

    # A fresh handle is allocated on every call, never reused while a prior
    # one might still be referenced.
    assert len(handles_seen) == 10


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


def handle_value(solver) -> int:
    """The solver's native handle as a plain int, for comparing across calls."""
    assert solver._handle is not None
    return int(solver._handle.id)


def test_analyze_again_reanalyzes_in_place(system):
    """A second analyze() reuses the handle instead of allocating another one.

    The point of re-analysis: a caller recovering from a bad factorization can
    redo the symbolic phase without ending up holding two handles and having
    to free the old one conditionally.
    """
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=matrix_type
    ) as solver:
        solver.analyze(jnp.asarray(values))
        first_handle = handle_value(solver)
        assert _ffi.analysis_count(first_handle) == 1

        new_values = values * 2.0
        solver.analyze(jnp.asarray(new_values))
        assert handle_value(solver) == first_handle
        assert _ffi.analysis_count(first_handle) == 2

        solver.factorize(jnp.asarray(new_values))
        solution = solver.solve(jnp.asarray(right_hand_side))

    expected = np.linalg.solve(dense * 2.0, right_hand_side)
    np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)


def test_reanalysis_discards_the_factorization(any_system):
    """Re-analysis frees the factors, so solve() is back to needing a factorize()."""
    indptr, indices, values, _dense, right_hand_side = any_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    ) as solver:
        solver.analyze(jnp.asarray(values))
        solver.factorize(jnp.asarray(values))
        solver.solve(jnp.asarray(right_hand_side))

        solver.analyze(jnp.asarray(values))
        with pytest.raises(RuntimeError, match="factorize"):
            solver.solve(jnp.asarray(right_hand_side))
        # refactorize() has the same precondition, and re-analysis clears it.
        with pytest.raises(RuntimeError, match="factorize"):
            solver.refactorize(jnp.asarray(values))

        solver.factorize(jnp.asarray(values))
        solver.solve(jnp.asarray(right_hand_side))


def test_repeated_reanalysis_does_not_leak(any_system):
    """Twenty re-analyses on one solver stay on one handle and keep solving correctly."""
    indptr, indices, values, dense, right_hand_side = any_system
    expected = np.linalg.solve(dense, right_hand_side)
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    ) as solver:
        solver.analyze(jnp.asarray(values))
        handle = handle_value(solver)
        for _ in range(20):
            solver.analyze(jnp.asarray(values))
            solver.factorize(jnp.asarray(values))
            solution = solver.solve(jnp.asarray(right_hand_side))
            np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)
        assert handle_value(solver) == handle
        assert _ffi.analysis_count(handle) == 21


def test_reanalyze_primitive_inside_and_outside_jit(system):
    """The re-analysis is ordered correctly inside a trace and agrees with eager.

    The handle goes into reanalyze and comes back out, which is what gives XLA
    a data dependency to order it against the factor and solve around it.
    """
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    indptr = jnp.asarray(indptr)
    indices = jnp.asarray(indices)

    def run(first_values, second_values, right_hand_side):
        handle, _iparm = primitive.analyze(indptr, indices, first_values, matrix_type=matrix_type)
        handle, _iparm = primitive.factor(
            handle, indptr, indices, first_values, matrix_type=matrix_type
        )
        handle, _iparm = primitive.reanalyze(
            handle, indptr, indices, second_values, matrix_type=matrix_type
        )
        handle, _iparm = primitive.factor(
            handle, indptr, indices, second_values, matrix_type=matrix_type
        )
        solution, _iparm = primitive.solve_stateful(
            handle,
            indptr,
            indices,
            second_values,
            right_hand_side[None, :],
            matrix_type=matrix_type,
        )
        return solution[0], handle

    arguments = (jnp.asarray(values), jnp.asarray(values * 2.0), jnp.asarray(right_hand_side))
    traced, traced_handle = jax.jit(run)(*arguments)
    eager, eager_handle = run(*arguments)

    # Two analyses ran on each handle, so the re-analysis was neither dropped
    # nor reordered ahead of the analyze that created the entry.
    assert _ffi.analysis_count(traced_handle.id) == 2
    assert _ffi.analysis_count(eager_handle.id) == 2
    primitive.release(traced_handle)
    primitive.release(eager_handle)

    expected = np.linalg.solve(dense * 2.0, right_hand_side)
    np.testing.assert_allclose(np.asarray(traced), expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.asarray(eager), expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.asarray(traced), np.asarray(eager), rtol=1e-12, atol=1e-14)


def test_reanalyze_rebuilds_an_evicted_handle(any_system):
    """Re-analyzing a released handle rebuilds it rather than failing.

    A handle whose factorization was released (or evicted from the bounded
    cache) is not dead. reanalyze runs a fresh analysis under the same handle
    from the matrix it is handed, so this checks the rebuild is counted and
    leaves a handle that a later factor and solve still use correctly.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    indptr = jnp.asarray(indptr)
    indices = jnp.asarray(indices)
    values = jnp.asarray(values)
    matrix_type = pmj.MatrixType.REAL_NONSYMMETRIC

    handle, _iparm = primitive.analyze(indptr, indices, values, matrix_type=matrix_type)
    primitive.release(handle)

    primitive.reset_rebuild_count()
    handle, _iparm = primitive.reanalyze(handle, indptr, indices, values, matrix_type=matrix_type)
    assert primitive.rebuild_count() == 1

    handle, _iparm = primitive.factor(handle, indptr, indices, values, matrix_type=matrix_type)
    stacked_rhs = jnp.asarray(right_hand_side)[None, :]
    solution, _iparm = primitive.solve_stateful(
        handle, indptr, indices, values, stacked_rhs, matrix_type=matrix_type
    )
    primitive.release(handle)
    expected = np.linalg.solve(dense, right_hand_side)
    np.testing.assert_allclose(np.asarray(solution[0]), expected, rtol=1e-8, atol=1e-10)


def test_solver_self_heals_after_a_backdoor_release(any_system):
    """A solve still works after the handle is released behind the solver's back.

    Releasing through the primitive frees the native factorization the solver
    believes it still holds. The next solve lands on a missing handle and
    rebuilds from the matrix it carries, so the answer stays correct and the
    rebuild counter records the heal. This is the core use-after-free-is-safe
    guarantee, exercised from the solver.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    with pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    ) as solver:
        solver.analyze(jnp.asarray(values))
        solver.factorize(jnp.asarray(values))
        primitive.release(solver._handle)

        primitive.reset_rebuild_count()
        solution = solver.solve(jnp.asarray(right_hand_side))
        assert primitive.rebuild_count() >= 1
    expected = np.linalg.solve(dense, right_hand_side)
    np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)


# The lifecycle and precondition checks below (method ordering, idempotent
# close, use without a with-block) do not depend on the matrix type, so they
# run once against a single representative system rather than once per type.


def test_methods_work_without_context_manager(any_system):
    """The solver no longer needs a with-block to be used.

    Self-healing removed the leak that once made the context manager mandatory,
    so a plain instance analyzes, factorizes, and solves correctly on its own.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    solver = pmj.PardisoSolver(
        jnp.asarray(indptr), jnp.asarray(indices), matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
    )
    solver.analyze(jnp.asarray(values))
    solver.factorize(jnp.asarray(values))
    solution = solver.solve(jnp.asarray(right_hand_side))
    solver.close()
    expected = np.linalg.solve(dense, right_hand_side)
    np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)


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
