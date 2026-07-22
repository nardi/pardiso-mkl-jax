"""Low-level JAX bindings for the compiled Pardiso FFI targets.

Wraps each XLA custom call target registered by _ffi.pyx as a plain JAX
function. `factor`, `solve_stateful`, and `release` operate on a persistent
native factorization identified by an integer solver_id, and back the
PardisoSolver class in solver.py. `solve` is the stateless, functional
one-shot entry point, and carries a custom vmap rule so that batching over
right-hand sides, matrix values, or both stays close to what native Pardiso
calls can do, instead of falling back to a naive per-example Python loop.

Right-hand-side and solution buffers are shaped (num_right_hand_sides, n)
throughout this module, not (n, num_right_hand_sides). Pardiso itself stores
these arrays column-major as (n, num_right_hand_sides), and a row-major array
shaped (num_right_hand_sides, n) has exactly the same byte layout, so this
choice avoids a transpose on every call. See the layout comment in
_pardiso_ffi.cc for the full explanation.
"""

from __future__ import annotations

import functools
import itertools
import threading

import jax
import jax.numpy as jnp
import numpy as np

from pardiso_mkl_jax import _ffi  # noqa: F401  (import registers the FFI targets)
from pardiso_mkl_jax.matrix import (
    MatrixType,
    check_csr_arrays,
    check_matrix_type_supported,
    check_upper_triangular,
)

# Pardiso phase codes used from Python. 13 (combined analyze, factor, solve)
# lives next to its handler in _pardiso_ffi.cc instead, since the stateless
# one-shot path never varies its phase.
PHASE_ANALYZE = 11
PHASE_FACTORIZE = 22

_solver_id_lock = threading.Lock()
_solver_id_counter = itertools.count(1)


def allocate_solver_id() -> int:
    """Return a fresh integer id, unique for the process, for the native state registry.

    A monotonic counter is enough: ids are never reused, so a live solver can
    never collide with one that has since been released.
    """
    with _solver_id_lock:
        return next(_solver_id_counter)


def factor(indptr, indices, values, *, solver_id: int, phase: int, matrix_type: MatrixType):
    """Run the analyze (phase 11) or numeric factorization (phase 22) step.

    Mutates the native state kept for solver_id and returns a status code (0
    means success; a non-zero Pardiso error code also raises before this
    returns, so callers mainly use the status for logging).
    """
    dimension = indptr.shape[0] - 1
    return jax.ffi.ffi_call(
        "pardiso_mkl_jax_factor",
        jax.ShapeDtypeStruct((), jnp.int32),
        has_side_effect=True,
    )(
        indptr,
        indices,
        values,
        solver_id=np.int64(solver_id),
        phase=np.int64(phase),
        matrix_type=np.int64(matrix_type),
        dimension=np.int64(dimension),
    )


def solve_stateful(
    indptr, indices, values, right_hand_side, *, solver_id: int, matrix_type: MatrixType
):
    """Solve (phase 33) against a factorization already produced for solver_id."""
    dimension = indptr.shape[0] - 1
    number_of_right_hand_sides = right_hand_side.shape[0]
    return jax.ffi.ffi_call(
        "pardiso_mkl_jax_solve",
        jax.ShapeDtypeStruct(right_hand_side.shape, jnp.float64),
        has_side_effect=True,
    )(
        indptr,
        indices,
        values,
        right_hand_side,
        solver_id=np.int64(solver_id),
        matrix_type=np.int64(matrix_type),
        dimension=np.int64(dimension),
        number_of_right_hand_sides=np.int64(number_of_right_hand_sides),
    )


def release(*, solver_id: int):
    """Free the native factorization state for solver_id."""
    return jax.ffi.ffi_call(
        "pardiso_mkl_jax_release",
        jax.ShapeDtypeStruct((), jnp.int32),
        has_side_effect=True,
    )(solver_id=np.int64(solver_id))


def _solve_once(indptr, indices, values, right_hand_side, *, matrix_type: MatrixType):
    """Stateless combined analyze, factor, and solve (phase 13). Never reuses state."""
    dimension = indptr.shape[0] - 1
    number_of_right_hand_sides = right_hand_side.shape[0]
    return jax.ffi.ffi_call(
        "pardiso_mkl_jax_solve_once",
        jax.ShapeDtypeStruct(right_hand_side.shape, jnp.float64),
    )(
        indptr,
        indices,
        values,
        right_hand_side,
        matrix_type=np.int64(matrix_type),
        dimension=np.int64(dimension),
        number_of_right_hand_sides=np.int64(number_of_right_hand_sides),
    )


def _factor_shared_pattern(indptr, indices, values, *, solver_id: int, matrix_type: MatrixType):
    """Analyze once, then factor with the given values. Used to start a batch loop."""
    factor(
        indptr, indices, values, solver_id=solver_id, phase=PHASE_ANALYZE, matrix_type=matrix_type
    )
    factor(
        indptr, indices, values, solver_id=solver_id, phase=PHASE_FACTORIZE, matrix_type=matrix_type
    )


@functools.cache
def _make_solve_core(matrix_type: MatrixType):
    """Build a custom_vmap-decorated solve function specialized to one matrix type.

    matrix_type is bound by closure here rather than passed as an ordinary
    argument, because custom_vmap traces every argument it is given as an
    abstract value and has no mechanism for a static, non-array argument
    (unlike jax.jit's static_argnums). Caching by matrix_type means each of
    the handful of matrix types only builds and registers its closure once.
    """

    @jax.custom_batching.custom_vmap
    def solve_core(indptr, indices, values, right_hand_side):
        """Solve A x = right_hand_side for a single matrix and a single right-hand side.

        This is the plain, non-batched case: right_hand_side has shape (n,),
        and the result has shape (n,). Batching this with jax.vmap is handled
        by the vmap rule below, which reuses Pardiso's own
        multiple-right-hand-side solve and its analysis-reuse mechanism, so
        vmap stays efficient rather than looping.
        """
        stacked_right_hand_side = right_hand_side[None, :]
        solution = _solve_once(
            indptr, indices, values, stacked_right_hand_side, matrix_type=matrix_type
        )
        return solution[0]

    @solve_core.def_vmap
    def vmap_rule(axis_size, in_batched, indptr, indices, values, right_hand_side):
        indptr_batched, indices_batched, values_batched, right_hand_side_batched = in_batched
        if indptr_batched or indices_batched:
            raise NotImplementedError(
                "vmap over indptr or indices is not supported: every matrix in a batch must "
                "share the same sparsity pattern. Batch over values instead."
            )

        if not values_batched and right_hand_side_batched:
            # Only the right-hand sides vary. Pardiso can solve all of them
            # against one factorization in a single call: the vmap batch axis
            # becomes the num_right_hand_sides axis directly, with no
            # transpose (see the module docstring on the
            # (num_right_hand_sides, n) layout).
            solution = _solve_once(
                indptr, indices, values, right_hand_side, matrix_type=matrix_type
            )
            return solution, True

        if values_batched:
            # The matrices vary, so each needs its own numeric factorization,
            # but they share one sparsity pattern and so share one symbolic
            # analysis. The analysis is run once, using the first batch
            # element's values (analysis for non-symmetric matrices can use
            # numeric values for scaling and matching, so the choice of
            # representative values can affect pivoting quality, though not
            # correctness), then each matrix is factored and solved in turn.
            solver_id = allocate_solver_id()
            try:
                _factor_shared_pattern(
                    indptr, indices, values[0], solver_id=solver_id, matrix_type=matrix_type
                )
                solutions = []
                for index in range(axis_size):
                    if index > 0:
                        factor(
                            indptr,
                            indices,
                            values[index],
                            solver_id=solver_id,
                            phase=PHASE_FACTORIZE,
                            matrix_type=matrix_type,
                        )
                    current_right_hand_side = (
                        right_hand_side[index][None, :]
                        if right_hand_side_batched
                        else right_hand_side[None, :]
                    )
                    solution = solve_stateful(
                        indptr,
                        indices,
                        values[index],
                        current_right_hand_side,
                        solver_id=solver_id,
                        matrix_type=matrix_type,
                    )
                    solutions.append(solution[0])
            finally:
                release(solver_id=solver_id)
            return jnp.stack(solutions), True

        # Neither values nor right_hand_side is batched. custom_vmap can
        # still reach this rule if unrelated arguments elsewhere in a larger
        # vmapped computation were batched, in which case this call is
        # unaffected.
        solution = solve_core(indptr, indices, values, right_hand_side)
        return solution, False

    return solve_core


def solve(indptr, indices, values, right_hand_side, *, matrix_type: MatrixType):
    """Solve A x = right_hand_side for a sparse matrix A given in CSR format.

    Runs analysis, factorization, and solve in a single call, and does not
    keep the factorization around afterward: use PardisoSolver instead if the
    same pattern will be solved again. Works under jit and vmap, batching
    over values, right_hand_side, or both.
    """
    check_matrix_type_supported(matrix_type)
    check_csr_arrays(indptr, indices, values)
    # check_upper_triangular returns indices threaded through a runtime
    # check (see its docstring): the returned value, not the original
    # indices, must be what actually reaches the solve below, or the check
    # is dead-code-eliminated whenever indptr/indices are traced.
    indices = check_upper_triangular(indptr, indices, matrix_type)
    solve_core = _make_solve_core(MatrixType(matrix_type))
    return solve_core(indptr, indices, values, right_hand_side)
