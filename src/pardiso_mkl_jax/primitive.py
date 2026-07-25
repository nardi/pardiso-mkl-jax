"""Low-level JAX bindings for the compiled Pardiso FFI targets.

Wraps each XLA custom call target registered by _ffi.pyx as a plain JAX
function. `analyze`, `factor`, `solve_stateful`, and `release` operate on a
persistent native factorization identified by a handle, an ordinary int64
JAX array value returned by `analyze` and threaded through every later call,
and back the PardisoSolver class in solver.py. Because the handle is a JAX
value rather than a Python-side id baked in at trace time, XLA orders
analyze, factor, solve, and release by the same data dependencies it uses
for any other computation, so the whole lifecycle can run inside a jitted
function. `solve` is the stateless, functional one-shot entry point, and
carries a custom vmap rule so that batching over right-hand sides, matrix
values, or both stays close to what native Pardiso calls can do, instead of
falling back to a naive per-example Python loop.

Right-hand-side and solution buffers are shaped (num_right_hand_sides, n)
throughout this module, not (n, num_right_hand_sides). Pardiso itself stores
these arrays column-major as (n, num_right_hand_sides), and a row-major array
shaped (num_right_hand_sides, n) has exactly the same byte layout, so this
choice avoids a transpose on every call. See the layout comment in
_pardiso_ffi.cc for the full explanation.
"""

from __future__ import annotations

import functools

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

# iparm[11] values controlling which system a solve step solves. TRANSPOSE
# is Pardiso's "transposed" mode (as opposed to "conjugate transposed",
# value 1), which coincides with it anyway for the real-valued matrices
# this package supports.
TRANSPOSE_NONE = 0
TRANSPOSE_TRANSPOSE = 2


def _transpose_mode(transpose: bool) -> np.int64:
    return np.int64(TRANSPOSE_TRANSPOSE if transpose else TRANSPOSE_NONE)


def analyze(indptr, indices, values, *, matrix_type: MatrixType):
    """Run the analyze (phase 11) step and allocate a fresh native factorization.

    Returns the handle for the new factorization, an int64 array value that
    every later call (factor, solve_stateful, factor_and_solve_stateful,
    release) takes as an input. Threading the handle as data, rather than
    addressing the native state by a Python-side id, is what lets XLA order
    the whole analyze-factor-solve-release lifecycle and lets it run inside a
    jitted function.
    """
    dimension = indptr.shape[0] - 1
    handle, _status = jax.ffi.ffi_call(
        "pardiso_mkl_jax_analyze",
        (
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int32),
        ),
        has_side_effect=True,
    )(
        indptr,
        indices,
        values,
        matrix_type=np.int64(matrix_type),
        dimension=np.int64(dimension),
    )
    return handle


def factor(handle, indptr, indices, values, *, matrix_type: MatrixType):
    """Run the numeric factorization (phase 22) step against handle.

    Returns handle unchanged, so a later call that consumes this function's
    return value is ordered after the factorization it performed.
    """
    dimension = indptr.shape[0] - 1
    handle_out, _status = jax.ffi.ffi_call(
        "pardiso_mkl_jax_factor",
        (
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int32),
        ),
        has_side_effect=True,
    )(
        handle,
        indptr,
        indices,
        values,
        matrix_type=np.int64(matrix_type),
        dimension=np.int64(dimension),
    )
    return handle_out


def solve_stateful(
    handle,
    indptr,
    indices,
    values,
    right_hand_side,
    *,
    matrix_type: MatrixType,
    transpose: bool = False,
):
    """Solve (phase 33) against the factorization already produced for handle.

    transpose solves A^T x = right_hand_side instead of A x = right_hand_side,
    reusing the same factorization: no call to factor() is needed to switch
    between the two for a given handle.
    """
    dimension = indptr.shape[0] - 1
    number_of_right_hand_sides = right_hand_side.shape[0]
    return jax.ffi.ffi_call(
        "pardiso_mkl_jax_solve",
        jax.ShapeDtypeStruct(right_hand_side.shape, jnp.float64),
        has_side_effect=True,
    )(
        handle,
        indptr,
        indices,
        values,
        right_hand_side,
        matrix_type=np.int64(matrix_type),
        dimension=np.int64(dimension),
        number_of_right_hand_sides=np.int64(number_of_right_hand_sides),
        transpose_mode=_transpose_mode(transpose),
    )


def factor_and_solve_stateful(
    handle,
    indptr,
    indices,
    values,
    right_hand_side,
    *,
    matrix_type: MatrixType,
    transpose: bool = False,
):
    """Refactor and solve in one call, reusing the analysis produced for handle.

    Runs Pardiso's combined phase 23 (numeric factorization then solve) for the
    given values against the stored analysis. This is a single FFI call, so the
    factorization and the solve stay ordered under jit, unlike a factor()
    followed by a separate solve_stateful(): those share no data dependency XLA
    must honor, so the solve could otherwise run before the factor.
    """
    dimension = indptr.shape[0] - 1
    number_of_right_hand_sides = right_hand_side.shape[0]
    return jax.ffi.ffi_call(
        "pardiso_mkl_jax_factor_solve",
        jax.ShapeDtypeStruct(right_hand_side.shape, jnp.float64),
        has_side_effect=True,
    )(
        handle,
        indptr,
        indices,
        values,
        right_hand_side,
        matrix_type=np.int64(matrix_type),
        dimension=np.int64(dimension),
        number_of_right_hand_sides=np.int64(number_of_right_hand_sides),
        transpose_mode=_transpose_mode(transpose),
    )


def release(handle):
    """Free the native factorization state for handle."""
    return jax.ffi.ffi_call(
        "pardiso_mkl_jax_release",
        jax.ShapeDtypeStruct((), jnp.int32),
        has_side_effect=True,
    )(handle)


def _solve_once(
    indptr, indices, values, right_hand_side, *, matrix_type: MatrixType, transpose: bool = False
):
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
        transpose_mode=_transpose_mode(transpose),
    )


@functools.cache
def _make_solve_core(matrix_type: MatrixType, transpose: bool):
    """Build a custom_vmap-decorated solve function specialized to one matrix type.

    matrix_type and transpose are bound by closure here rather than passed
    as ordinary arguments, because custom_vmap traces every argument it is
    given as an abstract value and has no mechanism for a static, non-array
    argument (unlike jax.jit's static_argnums). Caching by (matrix_type,
    transpose) means each combination only builds and registers its closure
    once.
    """

    @jax.custom_batching.custom_vmap
    def solve_core(indptr, indices, values, right_hand_side):
        """Solve A x = right_hand_side for a single matrix and a single right-hand side.

        Solves A^T x = right_hand_side instead when transpose is set. This is
        the plain, non-batched case: right_hand_side has shape (n,),
        and the result has shape (n,). Batching this with jax.vmap is handled
        by the vmap rule below, which reuses Pardiso's own
        multiple-right-hand-side solve and its analysis-reuse mechanism, so
        vmap stays efficient rather than looping.
        """
        stacked_right_hand_side = right_hand_side[None, :]
        solution = _solve_once(
            indptr,
            indices,
            values,
            stacked_right_hand_side,
            matrix_type=matrix_type,
            transpose=transpose,
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
            # becomes the num_right_hand_sides axis directly, with no array
            # transpose needed (see the module docstring on the
            # (num_right_hand_sides, n) layout).
            solution = _solve_once(
                indptr,
                indices,
                values,
                right_hand_side,
                matrix_type=matrix_type,
                transpose=transpose,
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
            handle = analyze(indptr, indices, values[0], matrix_type=matrix_type)
            try:
                handle = factor(handle, indptr, indices, values[0], matrix_type=matrix_type)
                solutions = []
                for index in range(axis_size):
                    if index > 0:
                        handle = factor(
                            handle, indptr, indices, values[index], matrix_type=matrix_type
                        )
                    current_right_hand_side = (
                        right_hand_side[index][None, :]
                        if right_hand_side_batched
                        else right_hand_side[None, :]
                    )
                    solution = solve_stateful(
                        handle,
                        indptr,
                        indices,
                        values[index],
                        current_right_hand_side,
                        matrix_type=matrix_type,
                        transpose=transpose,
                    )
                    solutions.append(solution[0])
            finally:
                release(handle)
            return jnp.stack(solutions), True

        # Neither values nor right_hand_side is batched. custom_vmap can
        # still reach this rule if unrelated arguments elsewhere in a larger
        # vmapped computation were batched, in which case this call is
        # unaffected.
        solution = solve_core(indptr, indices, values, right_hand_side)
        return solution, False

    return solve_core


def solve(
    indptr, indices, values, right_hand_side, *, matrix_type: MatrixType, transpose: bool = False
):
    """Solve A x = right_hand_side for a sparse matrix A given in CSR format.

    Solves A^T x = right_hand_side instead when transpose is set, using the
    same factorization Pardiso would use for A: no separate factorization of
    A^T is needed. Runs analysis, factorization, and solve in a single call,
    and does not keep the factorization around afterward: use PardisoSolver
    instead if the same pattern will be solved again. Works under jit and
    vmap, batching over values, right_hand_side, or both.
    """
    check_matrix_type_supported(matrix_type)
    check_csr_arrays(indptr, indices, values)
    # check_upper_triangular returns indices threaded through a runtime
    # check (see its docstring): the returned value, not the original
    # indices, must be what actually reaches the solve below, or the check
    # is dead-code-eliminated whenever indptr/indices are traced.
    indices = check_upper_triangular(indptr, indices, matrix_type)
    solve_core = _make_solve_core(MatrixType(matrix_type), transpose)
    return solve_core(indptr, indices, values, right_hand_side)
