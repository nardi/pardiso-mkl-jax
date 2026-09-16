"""Tests for the bounded handle cache: eviction, rebuild-on-miss, strict mode.

These pin the memory-safety contract. A handle is a key into a bounded cache,
never a raw pointer, so forgetting to release leaks at most the cache, and a
handle whose factorization was evicted or freed rebuilds itself from the matrix
the call carries rather than reading freed memory. Strict mode turns that
rebuild into an error so a lost factorization can be found while debugging.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pardiso_mkl_jax as pmj
from pardiso_mkl_jax import primitive

MATRIX_TYPE = pmj.MatrixType.REAL_NONSYMMETRIC


def _dense_from_csr(indptr, indices, values):
    """Build a dense matrix from zero-based CSR arrays, for a reference solve.

    The fixtures hand back a dense reference for their own values. A test that
    solves with different values on the same pattern needs the dense form of
    those values, which this reconstructs.
    """
    indptr = np.asarray(indptr)
    indices = np.asarray(indices)
    values = np.asarray(values)
    dimension = indptr.shape[0] - 1
    dense = np.zeros((dimension, dimension), dtype=np.float64)
    for row in range(dimension):
        for entry in range(indptr[row], indptr[row + 1]):
            dense[row, indices[entry]] = values[entry]
    return dense


def _analyze_factor(indptr, indices, values):
    """Analyze then factor a system, returning a ready-to-solve token."""
    token, _ = primitive.analyze(indptr, indices, values, matrix_type=MATRIX_TYPE)
    token, _ = primitive.factor(token, indptr, indices, values, matrix_type=MATRIX_TYPE)
    return token


def test_forgotten_handles_are_bounded_and_rebuild(any_system, monkeypatch):
    """Never releasing handles stays bounded, and evicted ones still solve.

    With the cache capped at two, factoring many systems without releasing any
    cannot grow the registry without end, because older handles are evicted.
    Solving through the very first handle then rebuilds its factorization. This
    checks both that the rebuild stays correct and that the rebuild counter
    records it, covering the bounded-leaks and use-after-eviction guarantees
    together.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    monkeypatch.setenv("PARDISO_MKL_JAX_FACTOR_CACHE", "2")

    # The first handle is the one we come back to; the loop evicts it.
    first_handle = _analyze_factor(indptr, indices, values)
    for _ in range(5):
        _analyze_factor(indptr, indices, values)

    primitive.reset_rebuild_count()
    solution, _, _ = primitive.solve_stateful(
        first_handle,
        indptr,
        indices,
        values,
        jnp.asarray(right_hand_side)[None, :],
        matrix_type=MATRIX_TYPE,
    )
    assert primitive.rebuild_count() >= 1
    expected = np.linalg.solve(dense, right_hand_side)
    np.testing.assert_allclose(np.asarray(solution[0]), expected, rtol=1e-8, atol=1e-10)


def test_solve_with_different_values_rebuilds_rather_than_using_stale_factors(
    any_system, monkeypatch
):
    """A solve whose values differ from the stored factorization rebuilds.

    The handle names a cache slot, not the matrix in it, and the slot's numeric
    factors are what a solve reuses. If a solve is handed values for a different
    matrix on the same pattern than the one factored into the slot, solving
    against the stored factors would silently answer for the wrong matrix. The
    solve fingerprints its matrix and, on a mismatch, rebuilds from the values
    it was given, so the answer is always for that matrix. This pins that: it
    factors one matrix, then solves the same handle with a second matrix's
    values and checks the answer matches the second matrix and a rebuild ran.
    """
    indptr, indices, values, _dense, right_hand_side = any_system
    indptr, indices = jnp.asarray(indptr), jnp.asarray(indices)
    values = jnp.asarray(values)
    other_values = values * 2.0 + 1.0
    other_dense = _dense_from_csr(indptr, indices, other_values)

    handle = _analyze_factor(indptr, indices, values)

    primitive.reset_rebuild_count()
    solution, _, _ = primitive.solve_stateful(
        handle,
        indptr,
        indices,
        other_values,
        jnp.asarray(right_hand_side)[None, :],
        matrix_type=MATRIX_TYPE,
    )
    assert primitive.rebuild_count() >= 1
    expected = np.linalg.solve(other_dense, right_hand_side)
    np.testing.assert_allclose(np.asarray(solution[0]), expected, rtol=1e-8, atol=1e-10)


def test_aliased_refactor_does_not_corrupt_a_solve_on_the_original_matrix(any_system):
    """Refactoring a handle for a new matrix cannot silently change another solve.

    This is the aliasing hazard directly: one factorization, then a second
    factor() on the same handle for a different matrix (as an aliased or stale
    token would do), replacing the factors in place. A later solve that still
    means the original matrix passes the original values, so its fingerprint no
    longer matches the slot, and it rebuilds rather than solving against the
    matrix that overwrote it. Checks the original solve still gets the original
    answer.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices = jnp.asarray(indptr), jnp.asarray(indices)
    values = jnp.asarray(values)
    other_values = values * 3.0 - 0.5

    handle = _analyze_factor(indptr, indices, values)
    # Second factorization on the same handle, for a different matrix. This is
    # what an aliased handle or stale token does: the slot now holds these
    # factors, not the original ones.
    handle, _ = primitive.factor(handle, indptr, indices, other_values, matrix_type=MATRIX_TYPE)

    solution, _, _ = primitive.solve_stateful(
        handle,
        indptr,
        indices,
        values,  # still the original matrix
        jnp.asarray(right_hand_side)[None, :],
        matrix_type=MATRIX_TYPE,
    )
    expected = np.linalg.solve(dense, right_hand_side)
    np.testing.assert_allclose(np.asarray(solution[0]), expected, rtol=1e-8, atol=1e-10)


def test_strict_mode_errors_when_the_slot_holds_a_different_matrix(any_system, monkeypatch):
    """Strict mode raises on a content mismatch, not only on an evicted handle.

    A mismatch is a caller bug (an aliased handle or a stale token), distinct
    from an eviction, so strict mode surfaces it loudly rather than rebuilding.
    This factors one matrix, then solves the same handle with a second matrix's
    values under strict mode and checks it raises and says the slot holds a
    different matrix.
    """
    indptr, indices, values, _dense, right_hand_side = any_system
    indptr, indices = jnp.asarray(indptr), jnp.asarray(indices)
    values = jnp.asarray(values)

    handle = _analyze_factor(indptr, indices, values)
    monkeypatch.setenv("PARDISO_MKL_JAX_STRICT_CACHE", "1")

    with pytest.raises(Exception, match="different matrix"):
        primitive.solve_stateful(
            handle,
            indptr,
            indices,
            values * 2.0,
            jnp.asarray(right_hand_side)[None, :],
            matrix_type=MATRIX_TYPE,
        )


def test_rebuild_reason_reports_none_on_a_plain_reuse(any_system):
    """A solve that reuses the cached factorization reports RebuildReason.NONE.

    This is the common path: analyze, factor, then solve the same matrix. The
    reason surfaces both as the primitive's third return and, decoded, on
    PardisoDiagnostics.rebuild_reason.
    """
    indptr, indices, values, _dense, right_hand_side = any_system
    indptr, indices = jnp.asarray(indptr), jnp.asarray(indices)
    values = jnp.asarray(values)

    handle = _analyze_factor(indptr, indices, values)
    _solution, final_iparm, reason = primitive.solve_stateful(
        handle,
        indptr,
        indices,
        values,
        jnp.asarray(right_hand_side)[None, :],
        matrix_type=MATRIX_TYPE,
    )
    assert int(reason) == pmj.RebuildReason.NONE
    diagnostics = pmj.PardisoDiagnostics.from_iparm(final_iparm, reason)
    assert int(diagnostics.rebuild_reason) == pmj.RebuildReason.NONE


def test_rebuild_reason_reports_matrix_mismatch(any_system):
    """A solve handed a different matrix reports RebuildReason.MATRIX_MISMATCH."""
    indptr, indices, values, _dense, right_hand_side = any_system
    indptr, indices = jnp.asarray(indptr), jnp.asarray(indices)
    values = jnp.asarray(values)

    handle = _analyze_factor(indptr, indices, values)
    _solution, _final_iparm, reason = primitive.solve_stateful(
        handle,
        indptr,
        indices,
        values * 2.0,
        jnp.asarray(right_hand_side)[None, :],
        matrix_type=MATRIX_TYPE,
    )
    assert int(reason) == pmj.RebuildReason.MATRIX_MISMATCH


def test_rebuild_reason_reports_eviction(any_system, monkeypatch):
    """A solve through an evicted handle reports RebuildReason.EVICTED_OR_RELEASED."""
    indptr, indices, values, _dense, right_hand_side = any_system
    indptr, indices = jnp.asarray(indptr), jnp.asarray(indices)
    values = jnp.asarray(values)

    monkeypatch.setenv("PARDISO_MKL_JAX_FACTOR_CACHE", "1")
    handle = _analyze_factor(indptr, indices, values)
    _analyze_factor(indptr, indices, values)  # evicts handle (capacity 1)

    _solution, _final_iparm, reason = primitive.solve_stateful(
        handle,
        indptr,
        indices,
        values,
        jnp.asarray(right_hand_side)[None, :],
        matrix_type=MATRIX_TYPE,
    )
    assert int(reason) == pmj.RebuildReason.EVICTED_OR_RELEASED


def test_pardiso_solver_reports_rebuild_reason_through_diagnostics(any_system, monkeypatch):
    """PardisoSolver.solve surfaces the rebuild reason on its diagnostics.

    A plain reuse reports NONE. After the factorization is evicted, the next
    solve rebuilds and reports EVICTED_OR_RELEASED. PardisoSolver always solves
    with the values it factored, so a matrix mismatch cannot arise here. That
    case is covered at the primitive level above.
    """
    indptr, indices, values, _dense, right_hand_side = any_system
    indptr, indices = jnp.asarray(indptr), jnp.asarray(indices)
    values = jnp.asarray(values)
    right_hand_side = jnp.asarray(right_hand_side)

    monkeypatch.setenv("PARDISO_MKL_JAX_FACTOR_CACHE", "1")
    solver = pmj.PardisoSolver(indptr, indices, matrix_type=MATRIX_TYPE)
    solver.analyze(values)
    solver.factorize(values)

    _first, diagnostics = solver.solve(right_hand_side, return_diagnostics=True)
    assert int(diagnostics.rebuild_reason) == pmj.RebuildReason.NONE

    # Evict this solver's factorization by overrunning the single cache slot.
    other = _analyze_factor(indptr, indices, values)
    del other

    _second, diagnostics = solver.solve(right_hand_side, return_diagnostics=True)
    assert int(diagnostics.rebuild_reason) == pmj.RebuildReason.EVICTED_OR_RELEASED


def test_transpose_alternation_reuses_the_factorization_without_rebuilding(any_system):
    """Alternating transpose on one handle reuses the factors, no rebuild.

    The transpose mode is not part of the matrix fingerprint, so switching it
    between solves on the same handle must not be mistaken for a different
    matrix. This solves once each way and checks neither caused a rebuild.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices = jnp.asarray(indptr), jnp.asarray(indices)
    values = jnp.asarray(values)
    stacked = jnp.asarray(right_hand_side)[None, :]

    handle = _analyze_factor(indptr, indices, values)

    primitive.reset_rebuild_count()
    forward, _, _ = primitive.solve_stateful(
        handle, indptr, indices, values, stacked, matrix_type=MATRIX_TYPE
    )
    transposed, _, _ = primitive.solve_stateful(
        handle, indptr, indices, values, stacked, matrix_type=MATRIX_TYPE, transpose=True
    )
    assert primitive.rebuild_count() == 0
    np.testing.assert_allclose(
        np.asarray(forward[0]), np.linalg.solve(dense, right_hand_side), rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(transposed[0]), np.linalg.solve(dense.T, right_hand_side), rtol=1e-8, atol=1e-10
    )


def test_strict_mode_turns_a_rebuild_into_an_error(any_system, monkeypatch):
    """Strict mode raises instead of silently rebuilding an evicted handle.

    A silent rebuild is correct but slow, so strict mode is the switch that
    makes a lost factorization loud while debugging performance. This forces an
    eviction with a tiny cache, then checks the next solve raises and names the
    handle rather than quietly redoing the work.
    """
    indptr, indices, values, _dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))

    monkeypatch.setenv("PARDISO_MKL_JAX_FACTOR_CACHE", "1")
    monkeypatch.setenv("PARDISO_MKL_JAX_STRICT_CACHE", "1")

    first_handle = _analyze_factor(indptr, indices, values)
    _analyze_factor(indptr, indices, values)  # evicts first_handle (capacity 1)

    with pytest.raises(Exception, match="strict cache mode"):
        primitive.solve_stateful(
            first_handle,
            indptr,
            indices,
            values,
            jnp.asarray(right_hand_side)[None, :],
            matrix_type=MATRIX_TYPE,
        )


def test_released_handle_rebuilds_on_next_solve(any_system):
    """An explicitly released handle self-heals when solved through again.

    Release frees the cache slot but does not invalidate the handle, so reusing
    it is allowed and rebuilds from the matrix the solve carries. This is the
    "freeing is optional, never fatal" half of the contract.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))

    handle = _analyze_factor(indptr, indices, values)
    primitive.release(handle)

    primitive.reset_rebuild_count()
    solution, _, _ = primitive.solve_stateful(
        handle,
        indptr,
        indices,
        values,
        jnp.asarray(right_hand_side)[None, :],
        matrix_type=MATRIX_TYPE,
    )
    assert primitive.rebuild_count() == 1
    expected = np.linalg.solve(dense, right_hand_side)
    np.testing.assert_allclose(np.asarray(solution[0]), expected, rtol=1e-8, atol=1e-10)


def test_full_lifecycle_inside_jit(any_system):
    """analyze, factor, solve, and release run correctly inside one jit.

    The handle is threaded as data and every stage is side-effecting, so XLA
    keeps the lifecycle ordered even though it is opaque to it. This checks the
    jitted path gives the same answer as a dense solve, which is what lets the
    whole cache scheme be used from inside compiled code.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    right_hand_side = jnp.asarray(right_hand_side)

    @jax.jit
    def run(rhs):
        handle, _ = primitive.analyze(indptr, indices, values, matrix_type=MATRIX_TYPE)
        handle, _ = primitive.factor(handle, indptr, indices, values, matrix_type=MATRIX_TYPE)
        solution, _, _ = primitive.solve_stateful(
            handle, indptr, indices, values, rhs[None, :], matrix_type=MATRIX_TYPE
        )
        primitive.release(handle)
        return solution[0]

    solution = run(right_hand_side)
    expected = np.linalg.solve(dense, np.asarray(right_hand_side))
    np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)


def test_eager_analyze_then_jitted_solve_self_heals(any_system, monkeypatch):
    """A handle made eagerly, then evicted, still solves inside a later jit.

    The factorization is built eagerly, the cache is then overrun so the handle
    is evicted, and a jitted solve reaches the missing handle and rebuilds.
    Checks the answer is still correct, so an
    eager factorization stays usable across a jit boundary even under eviction.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    right_hand_side = jnp.asarray(right_hand_side)

    monkeypatch.setenv("PARDISO_MKL_JAX_FACTOR_CACHE", "1")
    handle = _analyze_factor(indptr, indices, values)
    _analyze_factor(indptr, indices, values)  # evicts handle (capacity 1)

    @jax.jit
    def solve(rhs):
        solution, _, _ = primitive.solve_stateful(
            handle, indptr, indices, values, rhs[None, :], matrix_type=MATRIX_TYPE
        )
        return solution[0]

    primitive.reset_rebuild_count()
    solution = solve(right_hand_side)
    assert primitive.rebuild_count() >= 1
    expected = np.linalg.solve(dense, np.asarray(right_hand_side))
    np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)


def test_track_orders_a_release_after_the_solve(any_system):
    """token.track makes an in-trace release wait for the solve, so no rebuild.

    A release inside a jit trace is otherwise unordered against a solve on the
    same token and may run first, forcing a rebuild. Tracking the solution
    before releasing ties the release to it, so this checks the release costs no
    rebuild and the answer is still correct.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    right_hand_side = jnp.asarray(right_hand_side)

    token = _analyze_factor(indptr, indices, values)

    @jax.jit
    def solve_and_release(token, rhs):
        solution, _, _ = primitive.solve_stateful(
            token, indptr, indices, values, rhs[None, :], matrix_type=MATRIX_TYPE
        )
        primitive.release(token.track(solution))
        return solution

    primitive.reset_rebuild_count()
    solution = solve_and_release(token, right_hand_side)
    assert primitive.rebuild_count() == 0
    expected = np.linalg.solve(dense, np.asarray(right_hand_side))
    np.testing.assert_allclose(np.asarray(solution[0]), expected, rtol=1e-8, atol=1e-10)


def test_release_dependency_orders_after_the_solve(any_system):
    """release(token, dependency=solution) orders the release after that solve.

    Same guarantee as track, but passing the solution explicitly rather than
    threading it through the token. Checks it costs no rebuild.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    right_hand_side = jnp.asarray(right_hand_side)

    token = _analyze_factor(indptr, indices, values)

    @jax.jit
    def solve_and_release(token, rhs):
        solution, _, _ = primitive.solve_stateful(
            token, indptr, indices, values, rhs[None, :], matrix_type=MATRIX_TYPE
        )
        primitive.release(token, dependency=solution)
        return solution

    primitive.reset_rebuild_count()
    solution = solve_and_release(token, right_hand_side)
    assert primitive.rebuild_count() == 0
    expected = np.linalg.solve(dense, np.asarray(right_hand_side))
    np.testing.assert_allclose(np.asarray(solution[0]), expected, rtol=1e-8, atol=1e-10)


def test_track_counts_solutions(any_system):
    """n_dependent_solutions counts the solutions tracked into a token.

    The counter is what release depends on, and it is observable, so this
    checks it equals the number of track calls for finite solves.
    """
    indptr, indices, values, _dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    right_hand_side = jnp.asarray(right_hand_side)

    token = _analyze_factor(indptr, indices, values)
    assert int(token.n_dependent_solutions) == 0
    token = token.track(right_hand_side).track(right_hand_side).track(right_hand_side)
    assert int(token.n_dependent_solutions) == 3


def test_token_pytree_roundtrip(any_system):
    """A FactorizationToken flattens and rebuilds with its id and counter intact.

    This is what lets it cross jit boundaries and ride in a scan carry, so it
    must survive a flatten and unflatten unchanged.
    """
    indptr, indices, values, _dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))

    token = _analyze_factor(indptr, indices, values).track(jnp.asarray(right_hand_side))
    leaves, treedef = jax.tree_util.tree_flatten(token)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert int(rebuilt.id) == int(token.id)
    assert int(rebuilt.n_dependent_solutions) == int(token.n_dependent_solutions)


def test_many_solves_in_a_scan_then_release(any_system):
    """A token threads through lax.scan, and a release after it costs no rebuild.

    Each step solves and tracks its solution, so after the loop the release is
    ordered behind every solve. Checks no rebuild, the counter equals the step
    count, and each solution matches a dense solve. This also confirms the token
    is a valid scan carry, since its shape stays fixed across steps.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    rhs_sequence = jnp.stack([jnp.asarray(right_hand_side) * scale for scale in (1.0, 2.0, 3.0)])

    token = _analyze_factor(indptr, indices, values)

    @jax.jit
    def run(token, rhs_sequence):
        def step(token, rhs):
            solution, _, _ = primitive.solve_stateful(
                token, indptr, indices, values, rhs[None, :], matrix_type=MATRIX_TYPE
            )
            return token.track(solution[0]), solution[0]

        token, solutions = jax.lax.scan(step, token, rhs_sequence)
        primitive.release(token)
        return token.n_dependent_solutions, solutions

    primitive.reset_rebuild_count()
    count, solutions = run(token, rhs_sequence)
    assert primitive.rebuild_count() == 0
    assert int(count) == rhs_sequence.shape[0]
    expected = np.linalg.solve(dense, np.asarray(rhs_sequence).T).T
    np.testing.assert_allclose(np.asarray(solutions), expected, rtol=1e-8, atol=1e-10)


def test_many_solves_batched_then_release(any_system):
    """A batched right-hand-side solve tracked and released costs no rebuild.

    Pardiso solves many right-hand sides in one native call, which is how the
    stateful path does many solves at once. Tracking the batched solution orders
    the release after it, so this checks no rebuild and a correct batched answer.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    right_hand_sides = jnp.stack(
        [jnp.asarray(right_hand_side) * scale for scale in (1.0, 2.0, 3.0)]
    )

    token = _analyze_factor(indptr, indices, values)

    @jax.jit
    def solve_and_release(token, right_hand_sides):
        solutions, _, _ = primitive.solve_stateful(
            token, indptr, indices, values, right_hand_sides, matrix_type=MATRIX_TYPE
        )
        primitive.release(token.track(solutions))
        return solutions

    primitive.reset_rebuild_count()
    solutions = solve_and_release(token, right_hand_sides)
    assert primitive.rebuild_count() == 0
    expected = np.linalg.solve(dense, np.asarray(right_hand_sides).T).T
    np.testing.assert_allclose(np.asarray(solutions), expected, rtol=1e-8, atol=1e-10)
