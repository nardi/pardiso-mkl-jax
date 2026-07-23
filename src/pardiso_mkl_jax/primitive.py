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

Every call takes an iparm overlay (see pardiso_mkl_jax.iparm) applied on top
of the package defaults, and returns the final iparm array alongside its
usual result, for decoding into a PardisoDiagnostics.

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
from pardiso_mkl_jax.iparm import (
    OptionsLike,
    PardisoDiagnostics,
    canonicalize_overlay,
    overlay_to_arrays,
)
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


def _overlay_buffers(options: OptionsLike) -> tuple[jax.Array, jax.Array]:
    """Validate and expand an iparm overlay into the (mask, values) buffers the FFI call needs.

    Runs canonicalize_overlay on every call, relying on it being cheap and
    idempotent (see its docstring): call sites in this module pass an
    already-canonical tuple down from a caller that validated it once, and
    validating it again here is what makes that safe to do without a
    separate internal-only calling convention.
    """
    mask, values = overlay_to_arrays(canonicalize_overlay(options))
    return jnp.asarray(mask), jnp.asarray(values)


def analyze(indptr, indices, values, *, matrix_type: MatrixType, options: OptionsLike = None):
    """Run the analyze (phase 11) step and allocate a fresh native factorization.

    Returns (handle, final_iparm). The handle identifies the new
    factorization, an int64 array value that every later call (factor,
    solve_stateful, factor_and_solve_stateful, release) takes as an input.
    Threading the handle as data, rather than addressing the native state by
    a Python-side id, is what lets XLA order the whole
    analyze-factor-solve-release lifecycle and lets it run inside a jitted
    function. final_iparm is the complete iparm array as Pardiso left it, for
    decoding into a PardisoDiagnostics.
    """
    dimension = indptr.shape[0] - 1
    overlay_mask, overlay_values = _overlay_buffers(options)
    handle, _status, final_iparm = jax.ffi.ffi_call(
        "pardiso_mkl_jax_analyze",
        (
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((64,), jnp.int32),
        ),
        has_side_effect=True,
    )(
        indptr,
        indices,
        values,
        overlay_mask,
        overlay_values,
        matrix_type=np.int64(matrix_type),
        dimension=np.int64(dimension),
    )
    return handle, final_iparm


def factor(
    handle, indptr, indices, values, *, matrix_type: MatrixType, options: OptionsLike = None
):
    """Run the numeric factorization (phase 22) step against handle.

    Returns (handle, final_iparm). The handle comes back unchanged, so a
    later call that consumes this function's return value is ordered after
    the factorization it performed. final_iparm is the complete iparm array
    as Pardiso left it, for decoding into a PardisoDiagnostics.
    """
    dimension = indptr.shape[0] - 1
    overlay_mask, overlay_values = _overlay_buffers(options)
    handle_out, _status, final_iparm = jax.ffi.ffi_call(
        "pardiso_mkl_jax_factor",
        (
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((64,), jnp.int32),
        ),
        has_side_effect=True,
    )(
        handle,
        indptr,
        indices,
        values,
        overlay_mask,
        overlay_values,
        matrix_type=np.int64(matrix_type),
        dimension=np.int64(dimension),
    )
    return handle_out, final_iparm


def solve_stateful(
    handle,
    indptr,
    indices,
    values,
    right_hand_side,
    *,
    matrix_type: MatrixType,
    transpose: bool = False,
    options: OptionsLike = None,
):
    """Solve (phase 33) against the factorization already produced for handle.

    transpose solves A^T x = right_hand_side instead of A x = right_hand_side,
    reusing the same factorization: no call to factor() is needed to switch
    between the two for a given handle. Returns (solution, final_iparm), the
    latter for decoding into a PardisoDiagnostics.
    """
    dimension = indptr.shape[0] - 1
    number_of_right_hand_sides = right_hand_side.shape[0]
    overlay_mask, overlay_values = _overlay_buffers(options)
    return jax.ffi.ffi_call(
        "pardiso_mkl_jax_solve",
        (
            jax.ShapeDtypeStruct(right_hand_side.shape, jnp.float64),
            jax.ShapeDtypeStruct((64,), jnp.int32),
        ),
        has_side_effect=True,
    )(
        handle,
        indptr,
        indices,
        values,
        right_hand_side,
        overlay_mask,
        overlay_values,
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
    options: OptionsLike = None,
):
    """Refactor and solve in one call, reusing the analysis produced for handle.

    Runs Pardiso's combined phase 23 (numeric factorization then solve) for the
    given values against the stored analysis. This is a single FFI call, so the
    factorization and the solve stay ordered under jit, unlike a factor()
    followed by a separate solve_stateful(): those share no data dependency XLA
    must honor, so the solve could otherwise run before the factor. Returns
    (solution, final_iparm), the latter for decoding into a PardisoDiagnostics.
    """
    dimension = indptr.shape[0] - 1
    number_of_right_hand_sides = right_hand_side.shape[0]
    overlay_mask, overlay_values = _overlay_buffers(options)
    return jax.ffi.ffi_call(
        "pardiso_mkl_jax_factor_solve",
        (
            jax.ShapeDtypeStruct(right_hand_side.shape, jnp.float64),
            jax.ShapeDtypeStruct((64,), jnp.int32),
        ),
        has_side_effect=True,
    )(
        handle,
        indptr,
        indices,
        values,
        right_hand_side,
        overlay_mask,
        overlay_values,
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
    indptr,
    indices,
    values,
    right_hand_side,
    *,
    matrix_type: MatrixType,
    transpose: bool = False,
    options: OptionsLike = None,
):
    """Stateless combined analyze, factor, and solve (phase 13). Never reuses state.

    Returns (solution, final_iparm), the latter for decoding into a
    PardisoDiagnostics.
    """
    dimension = indptr.shape[0] - 1
    number_of_right_hand_sides = right_hand_side.shape[0]
    overlay_mask, overlay_values = _overlay_buffers(options)
    return jax.ffi.ffi_call(
        "pardiso_mkl_jax_solve_once",
        (
            jax.ShapeDtypeStruct(right_hand_side.shape, jnp.float64),
            jax.ShapeDtypeStruct((64,), jnp.int32),
        ),
    )(
        indptr,
        indices,
        values,
        right_hand_side,
        overlay_mask,
        overlay_values,
        matrix_type=np.int64(matrix_type),
        dimension=np.int64(dimension),
        number_of_right_hand_sides=np.int64(number_of_right_hand_sides),
        transpose_mode=_transpose_mode(transpose),
    )


@functools.cache
def _make_solve_core(
    matrix_type: MatrixType, transpose: bool, overlay_key: tuple[tuple[int, int], ...]
):
    """Build a custom_vmap-decorated solve function specialized to one (matrix_type,
    transpose, overlay) combination.

    All three are bound by closure here rather than passed as ordinary
    arguments, because custom_vmap traces every argument it is given as an
    abstract value and has no mechanism for a static, non-array argument
    (unlike jax.jit's static_argnums). Caching means each distinct
    combination only builds and registers its closure once. overlay_key
    joining this cache key means a caller who varies the overlay per call in
    a hot loop grows this cache by one compiled closure per distinct
    overlay, unboundedly: keep the overlay stable across calls in
    performance-sensitive code, the same way matrix_type and transpose
    already need to be.
    """

    @jax.custom_batching.custom_vmap
    def solve_core(indptr, indices, values, right_hand_side):
        """Solve A x = right_hand_side for a single matrix and a single right-hand side.

        Solves A^T x = right_hand_side instead when transpose is set. This is
        the plain, non-batched case: right_hand_side has shape (n,), and the
        result has shape (n,). Returns (solution, PardisoDiagnostics).
        Batching this with jax.vmap is handled by the vmap rule below, which
        reuses Pardiso's own multiple-right-hand-side solve and its
        analysis-reuse mechanism, so vmap stays efficient rather than
        looping.
        """
        stacked_right_hand_side = right_hand_side[None, :]
        solution, final_iparm = _solve_once(
            indptr,
            indices,
            values,
            stacked_right_hand_side,
            matrix_type=matrix_type,
            transpose=transpose,
            options=overlay_key,
        )
        return solution[0], PardisoDiagnostics.from_iparm(final_iparm)

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
            # (num_right_hand_sides, n) layout). Pardiso reports diagnostics
            # once per native call regardless, so batched=False here tells
            # jax.vmap not to recompute them per element: it still broadcasts
            # this single value across the output's batch dimension by
            # default (jax.vmap's out_axes=0 default applies to every output,
            # batched or not), so every entry ends up identical rather than
            # the array coming out unbatched. A caller who wants the compact
            # unbatched form can request it with jax.vmap's own out_axes.
            solution, final_iparm = _solve_once(
                indptr,
                indices,
                values,
                right_hand_side,
                matrix_type=matrix_type,
                transpose=transpose,
                options=overlay_key,
            )
            diagnostics = PardisoDiagnostics.from_iparm(final_iparm)
            result = solution, diagnostics
            batched = (True, jax.tree_util.tree_map(lambda _: False, diagnostics))
            return result, batched

        if values_batched:
            # The matrices vary, so each needs its own numeric factorization,
            # but they share one sparsity pattern and so share one symbolic
            # analysis. The analysis is run once, using the first batch
            # element's values (analysis for non-symmetric matrices can use
            # numeric values for scaling and matching, so the choice of
            # representative values can affect pivoting quality, though not
            # correctness), then each matrix is factored and solved in turn.
            # This is also the one branch where diagnostics genuinely differ
            # per batch element (eigenvalue counts, perturbed pivots, and so
            # on can differ per matrix), so they are collected and stacked
            # alongside the solutions.
            handle, _final_iparm = analyze(
                indptr, indices, values[0], matrix_type=matrix_type, options=overlay_key
            )
            try:
                handle, _final_iparm = factor(
                    handle, indptr, indices, values[0], matrix_type=matrix_type, options=overlay_key
                )
                solutions = []
                diagnostics_per_element = []
                for index in range(axis_size):
                    if index > 0:
                        handle, _final_iparm = factor(
                            handle,
                            indptr,
                            indices,
                            values[index],
                            matrix_type=matrix_type,
                            options=overlay_key,
                        )
                    current_right_hand_side = (
                        right_hand_side[index][None, :]
                        if right_hand_side_batched
                        else right_hand_side[None, :]
                    )
                    solution, final_iparm = solve_stateful(
                        handle,
                        indptr,
                        indices,
                        values[index],
                        current_right_hand_side,
                        matrix_type=matrix_type,
                        transpose=transpose,
                        options=overlay_key,
                    )
                    solutions.append(solution[0])
                    diagnostics_per_element.append(PardisoDiagnostics.from_iparm(final_iparm))
            finally:
                release(handle)
            stacked_diagnostics = jax.tree_util.tree_map(
                lambda *per_element: jnp.stack(per_element), *diagnostics_per_element
            )
            result = jnp.stack(solutions), stacked_diagnostics
            batched = (True, jax.tree_util.tree_map(lambda _: True, stacked_diagnostics))
            return result, batched

        # Neither values nor right_hand_side is batched. custom_vmap can
        # still reach this rule if unrelated arguments elsewhere in a larger
        # vmapped computation were batched, in which case this call is
        # unaffected.
        result = solve_core(indptr, indices, values, right_hand_side)
        batched = jax.tree_util.tree_map(lambda _: False, result)
        return result, batched

    return solve_core


def solve(
    indptr,
    indices,
    values,
    right_hand_side,
    *,
    matrix_type: MatrixType,
    transpose: bool = False,
    options: OptionsLike = None,
    return_diagnostics: bool = False,
):
    """Solve A x = right_hand_side for a sparse matrix A given in CSR format.

    Solves A^T x = right_hand_side instead when transpose is set, using the
    same factorization Pardiso would use for A: no separate factorization of
    A^T is needed. options overrides Pardiso's iparm defaults for this call;
    see pardiso_mkl_jax.iparm.PardisoOption. Runs analysis, factorization,
    and solve in a single call, and does not keep the factorization around
    afterward: use PardisoSolver instead if the same pattern will be solved
    again. Works under jit and vmap, batching over values, right_hand_side,
    or both.

    Returns just the solution by default. If return_diagnostics is set,
    returns (solution, PardisoDiagnostics) instead, valid and correctly
    shaped whether this call runs eagerly or is itself wrapped in jax.jit,
    and only ever available on success: a failed Pardiso call raises instead
    of returning anything, diagnostics included.
    """
    check_matrix_type_supported(matrix_type)
    check_csr_arrays(indptr, indices, values)
    # check_upper_triangular returns indices threaded through a runtime
    # check (see its docstring): the returned value, not the original
    # indices, must be what actually reaches the solve below, or the check
    # is dead-code-eliminated whenever indptr/indices are traced.
    indices = check_upper_triangular(indptr, indices, matrix_type)
    overlay_key = canonicalize_overlay(options)
    solve_core = _make_solve_core(MatrixType(matrix_type), transpose, overlay_key)
    solution, diagnostics = solve_core(indptr, indices, values, right_hand_side)
    if return_diagnostics:
        return solution, diagnostics
    return solution
