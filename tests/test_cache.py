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


def _analyze_factor(indptr, indices, values):
    """Analyze then factor a system, returning a ready-to-solve token."""
    token, _ = primitive.analyze(indptr, indices, values, matrix_type=MATRIX_TYPE)
    token, _ = primitive.factor(token, indptr, indices, values, matrix_type=MATRIX_TYPE)
    return token


def test_forgotten_handles_are_bounded_and_rebuild(any_system, monkeypatch):
    """Never releasing handles stays bounded, and evicted ones still solve.

    With the cache capped at two, factoring many distinct systems without
    releasing any cannot grow the registry without end, because older handles
    are evicted. Solving through the very first handle then rebuilds its
    factorization. This checks both that the rebuild stays correct and that the
    rebuild counter records it, covering the bounded-leaks and use-after-eviction
    guarantees together. The systems must differ, since identical matrices hash
    to one content-addressed entry and would not evict each other.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    monkeypatch.setenv("PARDISO_MKL_JAX_FACTOR_CACHE", "2")

    # The first handle is the one we come back to. The loop factors distinct
    # matrices, which evict it.
    first_handle = _analyze_factor(indptr, indices, values)
    for scale in (2.0, 3.0, 4.0, 5.0, 6.0):
        _analyze_factor(indptr, indices, values * scale)

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
    # A distinct matrix evicts first_handle at capacity 1.
    _analyze_factor(indptr, indices, values * 2.0)

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

    The handle is threaded as data, so the data dependency keeps the lifecycle
    ordered even though the calls are opaque to XLA. This checks the jitted path
    gives the same answer as a dense solve, which is what lets the whole cache
    scheme be used from inside compiled code.
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
    # A distinct matrix evicts handle at capacity 1.
    _analyze_factor(indptr, indices, values * 2.0)

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


def test_identical_factor_deduplicates(any_system):
    """Factoring the same matrix twice gives one handle; a different one differs.

    A handle is a content hash of the matrix, so identical inputs hash equal and
    share a cache entry, and a changed matrix takes a new handle.
    """
    indptr, indices, values, _dense, _right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))

    first = _analyze_factor(indptr, indices, values)
    second = _analyze_factor(indptr, indices, values)
    assert int(first.id) == int(second.id)

    other = _analyze_factor(indptr, indices, values * 2.0)
    assert int(other.id) != int(first.id)


def test_cse_merged_factor_stays_correct(any_system):
    """Two identical factor calls may merge to one, and the answer stays correct.

    factor carries no side effect, so XLA is free to fold two identical factor
    calls into a single custom call. Solving through both handles must still
    match a dense solve. The compiled program is checked to hold one factor
    custom call, so the merge is really happening.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    rhs = jnp.asarray(right_hand_side)

    def run(values, rhs):
        token_a, _ = primitive.analyze(indptr, indices, values, matrix_type=MATRIX_TYPE)
        token_a, _ = primitive.factor(token_a, indptr, indices, values, matrix_type=MATRIX_TYPE)
        token_b, _ = primitive.analyze(indptr, indices, values, matrix_type=MATRIX_TYPE)
        token_b, _ = primitive.factor(token_b, indptr, indices, values, matrix_type=MATRIX_TYPE)
        first, _, _ = primitive.solve_stateful(
            token_a, indptr, indices, values, rhs[None, :], matrix_type=MATRIX_TYPE
        )
        second, _, _ = primitive.solve_stateful(
            token_b, indptr, indices, values, rhs[None, :], matrix_type=MATRIX_TYPE
        )
        return first[0], second[0]

    compiled_text = jax.jit(run).lower(values, rhs).compile().as_text() or ""
    assert compiled_text.count('custom_call_target="pardiso_mkl_jax_factor"') == 1

    first, second = jax.jit(run)(values, rhs)
    expected = np.linalg.solve(dense, right_hand_side)
    np.testing.assert_allclose(np.asarray(first), expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.asarray(second), expected, rtol=1e-8, atol=1e-10)


def test_alias_refactor_supersedes(any_system):
    """A refactor re-keys, so a stale alias of the old handle rebuilds the original.

    Factoring gives a handle for the first values. Refactoring to new values
    returns a new handle and retires the old one as superseded. A solve through
    the stale old handle rebuilds and returns the original matrix's solution,
    reporting SUPERSEDED, while the new handle returns the new matrix's solution.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    rhs = jnp.asarray(right_hand_side)[None, :]
    # A scale no other test uses, so the refactored key is genuinely fresh and
    # the factor call re-keys rather than deduplicating onto a cached entry.
    scale = 3.5
    new_values = values * scale

    old = _analyze_factor(indptr, indices, values)
    new, _ = primitive.factor(old, indptr, indices, new_values, matrix_type=MATRIX_TYPE)

    stale, _, stale_reason = primitive.solve_stateful(
        old, indptr, indices, values, rhs, matrix_type=MATRIX_TYPE
    )
    fresh, _, fresh_reason = primitive.solve_stateful(
        new, indptr, indices, new_values, rhs, matrix_type=MATRIX_TYPE
    )

    assert primitive.RebuildReason(int(stale_reason)) == primitive.RebuildReason.SUPERSEDED
    assert primitive.RebuildReason(int(fresh_reason)) == primitive.RebuildReason.NONE
    np.testing.assert_allclose(
        np.asarray(stale[0]), np.linalg.solve(dense, right_hand_side), rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(fresh[0]),
        np.linalg.solve(dense * scale, right_hand_side),
        rtol=1e-8,
        atol=1e-10,
    )


def test_residency_does_not_change_the_answer(any_system, monkeypatch):
    """A solve gives the same answer resident or rebuilt, and reports which.

    A resident factorization solves and reports NONE. After the cache evicts it,
    the same solve rebuilds from the matrix it carries and returns an identical
    answer, now reporting a rebuild rather than a hit.
    """
    indptr, indices, values, dense, right_hand_side = any_system
    indptr, indices, values = map(jnp.asarray, (indptr, indices, values))
    rhs = jnp.asarray(right_hand_side)[None, :]
    expected = np.linalg.solve(dense, right_hand_side)

    monkeypatch.setenv("PARDISO_MKL_JAX_FACTOR_CACHE", "1")
    token = _analyze_factor(indptr, indices, values)
    resident, _, resident_reason = primitive.solve_stateful(
        token, indptr, indices, values, rhs, matrix_type=MATRIX_TYPE
    )
    assert primitive.RebuildReason(int(resident_reason)) == primitive.RebuildReason.NONE

    # A distinct factorization evicts the token at capacity 1.
    _analyze_factor(indptr, indices, values * 2.0)
    rebuilt, _, rebuilt_reason = primitive.solve_stateful(
        token, indptr, indices, values, rhs, matrix_type=MATRIX_TYPE
    )
    assert primitive.RebuildReason(int(rebuilt_reason)) != primitive.RebuildReason.NONE

    np.testing.assert_allclose(np.asarray(resident[0]), expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.asarray(rebuilt[0]), expected, rtol=1e-8, atol=1e-10)
