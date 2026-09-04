"""Low-level JAX bindings for the compiled Pardiso FFI targets.

Wraps each XLA custom call target registered by _ffi.pyx as a plain JAX
function. `analyze`, `reanalyze`, `factor`, `solve_stateful`, and `release`
operate on a persistent native factorization identified by a FactorizationToken
returned by `analyze` and threaded through every later call, and back the
PardisoSolver class in solver.py. Because the token carries its cache id as a
JAX value rather than a Python-side id baked in at trace time, XLA orders
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

from pardiso_mkl_jax import _ffi  # importing also registers the FFI targets
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


@functools.cache
def default_iparm(matrix_type: MatrixType) -> np.ndarray:
    """This package's iparm defaults for matrix_type, before any overlay is applied.

    Read out of InitializeIparm in _pardiso_ffi.cc rather than restated here,
    so there is only ever one copy of those defaults. Callers need this to
    work out the value an entry will actually take for a call, which is the
    overlay entry if the overlay has one and this default otherwise.

    The returned array is cached and shared, so it is made read-only to stop a
    caller mutating every later reader's copy.
    """
    defaults = _ffi.default_iparm(int(matrix_type))
    defaults.flags.writeable = False
    return defaults


def rebuild_count() -> int:
    """Number of factorization rebuilds since load or the last reset.

    A factorization is rebuilt whenever a call reaches a handle that was
    evicted from the bounded cache or released, using the matrix the call
    already carries. Rebuilds keep results correct but cost the redone work, so
    a steadily rising count means the cache (PARDISO_MKL_JAX_FACTOR_CACHE) is
    too small for how many factorizations are kept live at once.
    """
    return int(_ffi.rebuild_count())


def reset_rebuild_count() -> None:
    """Reset the rebuild counter to zero."""
    _ffi.reset_rebuild_count()


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


def _ordering_witness(solution):
    """A zero-valued int32 that XLA cannot fold away, so it forces ordering.

    Multiplying by 0.0 keeps the value zero but stays live, since 0.0 times a
    NaN or infinity is NaN under IEEE 754. Reading one element makes the result
    depend on the whole solve. See release.
    """
    return (0.0 * jnp.real(jnp.ravel(solution)[0])).astype(jnp.int32)


@jax.tree_util.register_pytree_node_class
class FactorizationToken:
    """Handle to a native factorization: a cache id, a version, and a solve counter.

    id is the key into the native cache. version is a generation stamp the
    native layer sets on every factor or reanalyze and passes through unchanged
    on a solve, so a solve can tell whether the handle still holds the
    factorization the token names, and rejects the call otherwise.
    n_dependent_solutions counts the solutions passed to track. release consumes
    it, so a release is ordered after those solves even inside a jit trace.
    """

    def __init__(self, id, version, n_dependent_solutions):
        self.id = id
        self.version = version
        self.n_dependent_solutions = n_dependent_solutions

    def track(self, *solutions):
        """Return a token whose release is ordered after these solutions."""
        count = self.n_dependent_solutions
        for solution in solutions:
            count = count + jnp.int32(1) + _ordering_witness(solution)
        return FactorizationToken(self.id, self.version, count)

    def tree_flatten(self):
        return (self.id, self.version, self.n_dependent_solutions), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def _ordering_operand(token, dependency):
    """The int32 release consumes to order itself after the wanted solves."""
    if dependency is None:
        return token.n_dependent_solutions
    witness = jnp.zeros((), jnp.int32)
    for leaf in jax.tree_util.tree_leaves(dependency):
        leaf = jnp.asarray(leaf)
        # Only float or complex leaves can carry an edge XLA will not fold.
        if jnp.issubdtype(leaf.dtype, jnp.inexact):
            witness = witness + _ordering_witness(leaf)
    return witness


def analyze(indptr, indices, values, *, matrix_type: MatrixType, options: OptionsLike = None):
    """Run the analyze (phase 11) step and allocate a fresh native factorization.

    Returns (token, final_iparm). The token is a FactorizationToken carrying
    the native factorization's cache id and its version stamp, which every
    later call (factor, solve_stateful, factor_and_solve_stateful, release)
    takes as an input.
    Threading the id as data, rather than addressing the native state by
    a Python-side id, is what lets XLA order the whole
    analyze-factor-solve-release lifecycle and lets it run inside a jitted
    function. final_iparm is the complete iparm array as Pardiso left it, for
    decoding into a PardisoDiagnostics.

    Every call allocates a new factorization. To redo the analysis for a
    token that already has one, use reanalyze instead, which reuses the
    id rather than leaving the old one for the caller to release.
    """
    dimension = indptr.shape[0] - 1
    overlay_mask, overlay_values = _overlay_buffers(options)
    handle, version, _status, final_iparm = jax.ffi.ffi_call(
        "pardiso_mkl_jax_analyze",
        (
            jax.ShapeDtypeStruct((), jnp.int64),
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
    return FactorizationToken(handle, version, jnp.zeros((), jnp.int32)), final_iparm


def reanalyze(
    token, indptr, indices, values, *, matrix_type: MatrixType, options: OptionsLike = None
):
    """Re-run the analyze (phase 11) step in place on an existing token.

    Frees the factorization currently held for the token and runs a fresh
    symbolic analysis into the same native state, so this is how a caller
    redoes the analysis (for a new sparsity-compatible pattern, different
    values, or a different overlay) without ending up holding two ids.
    Returns (token, final_iparm) with the id unchanged, which keeps later
    calls ordered against it by data dependency exactly as factor does. The
    returned token carries a fresh version stamp and its solve counter is reset
    to zero.

    The numeric factorization is gone afterwards, so factor must run again
    before any solve. If the id was evicted or released, this rebuilds it
    from scratch instead of raising, the same self-healing behavior every
    other stateful call has (see rebuild_count). Set
    PARDISO_MKL_JAX_STRICT_CACHE to turn that rebuild into an error instead.
    """
    dimension = indptr.shape[0] - 1
    overlay_mask, overlay_values = _overlay_buffers(options)
    handle_out, version_out, _status, final_iparm = jax.ffi.ffi_call(
        "pardiso_mkl_jax_reanalyze",
        (
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((64,), jnp.int32),
        ),
        has_side_effect=True,
    )(
        token.id,
        indptr,
        indices,
        values,
        overlay_mask,
        overlay_values,
        matrix_type=np.int64(matrix_type),
        dimension=np.int64(dimension),
    )
    return FactorizationToken(handle_out, version_out, jnp.zeros((), jnp.int32)), final_iparm


def factor(token, indptr, indices, values, *, matrix_type: MatrixType, options: OptionsLike = None):
    """Run the numeric factorization (phase 22) step against token.

    Returns (token, final_iparm). The id comes back unchanged, so a
    later call that consumes this function's returned token is ordered after
    the factorization it performed. The returned token carries a fresh version
    stamp, since this call replaced the factorization, and its solve counter is
    reset to zero. final_iparm is the complete iparm array as Pardiso left it,
    for decoding into a PardisoDiagnostics.
    """
    dimension = indptr.shape[0] - 1
    overlay_mask, overlay_values = _overlay_buffers(options)
    handle_out, version_out, _status, final_iparm = jax.ffi.ffi_call(
        "pardiso_mkl_jax_factor",
        (
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((64,), jnp.int32),
        ),
        has_side_effect=True,
    )(
        token.id,
        indptr,
        indices,
        values,
        overlay_mask,
        overlay_values,
        matrix_type=np.int64(matrix_type),
        dimension=np.int64(dimension),
    )
    return FactorizationToken(handle_out, version_out, jnp.zeros((), jnp.int32)), final_iparm


@functools.cache
def _make_solve_stateful_core(
    matrix_type: MatrixType, transpose: bool, overlay_key: tuple[tuple[int, int], ...]
):
    """Build a custom_vmap-decorated solve-stateful function for one
    (matrix_type, transpose, overlay) combination.

    Same closure-capture pattern as _make_solve_core: custom_vmap traces
    every argument as an abstract value, so static args must be bound by
    closure rather than passed in.
    """

    @jax.custom_batching.custom_vmap
    def solve_stateful_core(token_id, version, indptr, indices, values, right_hand_side):
        dimension = indptr.shape[0] - 1
        number_of_right_hand_sides = right_hand_side.shape[0]
        overlay_mask, overlay_values = _overlay_buffers(overlay_key)
        return jax.ffi.ffi_call(
            "pardiso_mkl_jax_solve",
            (
                jax.ShapeDtypeStruct(right_hand_side.shape, jnp.float64),
                jax.ShapeDtypeStruct((), jnp.int64),
                jax.ShapeDtypeStruct((), jnp.int64),
                jax.ShapeDtypeStruct((64,), jnp.int32),
            ),
            has_side_effect=True,
        )(
            token_id,
            version,
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

    @solve_stateful_core.def_vmap
    def vmap_rule(
        _axis_size, in_batched, token_id, version, indptr, indices, values, right_hand_side
    ):
        (
            token_id_batched,
            version_batched,
            indptr_batched,
            indices_batched,
            values_batched,
            rhs_batched,
        ) = in_batched
        if indptr_batched or indices_batched:
            raise NotImplementedError(
                "vmap over indptr or indices is not supported: every matrix in a batch must "
                "share the same sparsity pattern. Batch over values instead."
            )
        if token_id_batched or version_batched:
            raise NotImplementedError("vmap over the token is not supported.")
        if values_batched:
            raise NotImplementedError(
                "vmap over values is not supported for solve_stateful (phase 33 only). "
                "Use factor_and_solve_stateful to refactor per batch element."
            )

        # The handle and version outputs are scalars the native call echoes, so
        # they never come back batched.
        if rhs_batched:
            # Fuse the batch dim into Pardiso's multi-RHS support.
            # rhs shape is (batch, num_rhs, n), reshape to (batch*num_rhs, n)
            # so Pardiso solves all of them in one native call.
            original_shape = right_hand_side.shape
            fused = right_hand_side.reshape(-1, original_shape[-1])
            solution, handle_out, version_out, final_iparm = solve_stateful_core(
                token_id, version, indptr, indices, values, fused
            )
            solution = solution.reshape(original_shape)
            return (solution, handle_out, version_out, final_iparm), (True, False, False, False)

        # Nothing batched. custom_vmap can still reach here if unrelated
        # arguments elsewhere in a larger vmapped computation were batched.
        result = solve_stateful_core(token_id, version, indptr, indices, values, right_hand_side)
        return result, (False, False, False, False)

    return solve_stateful_core


def solve_stateful(
    token,
    indptr,
    indices,
    values,
    right_hand_side,
    *,
    matrix_type: MatrixType,
    transpose: bool = False,
    options: OptionsLike = None,
    return_token: bool = False,
):
    """Solve (phase 33) against the factorization already produced for token.

    transpose solves A^T x = right_hand_side instead of A x = right_hand_side,
    reusing the same factorization. No call to factor() is needed to switch
    between the two for a given token.

    Returns (solution, final_iparm) by default, the second for decoding into a
    PardisoDiagnostics. With return_token set it returns
    (solution, token, final_iparm) instead, where the token is a distinct value
    the native solve produced. Threading that token into a later call gives
    that call a data dependency on this solve, which is what stops a later
    factor from overwriting the factorization before this solve reads it. See
    the advanced-usage guide on reusing one handle across ordered solves.

    To order a later release after this solve, thread the returned token, or
    pass the solution to token.track (see release).
    """
    overlay_key = canonicalize_overlay(options)
    core = _make_solve_stateful_core(MatrixType(matrix_type), transpose, overlay_key)
    solution, handle_out, version_out, final_iparm = core(
        token.id, token.version, indptr, indices, values, right_hand_side
    )
    if return_token:
        threaded = FactorizationToken(handle_out, version_out, token.n_dependent_solutions)
        return solution, threaded, final_iparm
    return solution, final_iparm


@functools.cache
def _make_factor_and_solve_stateful_core(
    matrix_type: MatrixType, transpose: bool, overlay_key: tuple[tuple[int, int], ...]
):
    """Build a custom_vmap-decorated factor-and-solve function for one
    (matrix_type, transpose, overlay) combination.

    Same closure-capture pattern as _make_solve_core.
    """

    @jax.custom_batching.custom_vmap
    def factor_and_solve_core(token_id, version, indptr, indices, values, right_hand_side):
        dimension = indptr.shape[0] - 1
        number_of_right_hand_sides = right_hand_side.shape[0]
        overlay_mask, overlay_values = _overlay_buffers(overlay_key)
        return jax.ffi.ffi_call(
            "pardiso_mkl_jax_factor_solve",
            (
                jax.ShapeDtypeStruct(right_hand_side.shape, jnp.float64),
                jax.ShapeDtypeStruct((), jnp.int64),
                jax.ShapeDtypeStruct((), jnp.int64),
                jax.ShapeDtypeStruct((64,), jnp.int32),
            ),
            has_side_effect=True,
        )(
            token_id,
            version,
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

    @factor_and_solve_core.def_vmap
    def vmap_rule(
        axis_size, in_batched, token_id, version, indptr, indices, values, right_hand_side
    ):
        (
            token_id_batched,
            version_batched,
            indptr_batched,
            indices_batched,
            values_batched,
            rhs_batched,
        ) = in_batched
        if indptr_batched or indices_batched:
            raise NotImplementedError(
                "vmap over indptr or indices is not supported: every matrix in a batch must "
                "share the same sparsity pattern. Batch over values instead."
            )
        if token_id_batched or version_batched:
            raise NotImplementedError("vmap over the token is not supported.")

        if not values_batched and rhs_batched:
            # Fuse the batch dim into Pardiso's multi-RHS support.
            # Phase 23 factors once and solves all RHS in one native call.
            original_shape = right_hand_side.shape
            fused = right_hand_side.reshape(-1, original_shape[-1])
            solution, handle_out, version_out, final_iparm = factor_and_solve_core(
                token_id, version, indptr, indices, values, fused
            )
            solution = solution.reshape(original_shape)
            return (solution, handle_out, version_out, final_iparm), (True, False, False, False)

        if values_batched:
            # Each batch element needs its own numeric factorization, but
            # they share the analysis behind token_id. Loop and call phase
            # 23 per element, then stack. The handle and version are echoed
            # unchanged, so they come back the same for every element and stay
            # unbatched.
            solutions = []
            iparms = []
            handle_out = version_out = None
            for i in range(axis_size):
                current_rhs = right_hand_side[i] if rhs_batched else right_hand_side
                sol, handle_out, version_out, iparm = factor_and_solve_core(
                    token_id, version, indptr, indices, values[i], current_rhs
                )
                solutions.append(sol)
                iparms.append(iparm)
            stacked = (jnp.stack(solutions), handle_out, version_out, jnp.stack(iparms))
            return stacked, (True, False, False, True)

        result = factor_and_solve_core(token_id, version, indptr, indices, values, right_hand_side)
        return result, (False, False, False, False)

    return factor_and_solve_core


def factor_and_solve_stateful(
    token,
    indptr,
    indices,
    values,
    right_hand_side,
    *,
    matrix_type: MatrixType,
    transpose: bool = False,
    options: OptionsLike = None,
    return_token: bool = False,
):
    """Refactor and solve in one call, reusing the analysis produced for token.

    Runs Pardiso's combined phase 23 (numeric factorization then solve) for the
    given values against the stored analysis. This is a single FFI call, so the
    factorization and the solve stay ordered under jit, unlike a factor()
    followed by a separate solve_stateful(). Those share no data dependency XLA
    must honor, so the solve could otherwise run before the factor.

    Returns (solution, final_iparm) by default. With return_token set it returns
    (solution, token, final_iparm), where the token carries the new version this
    call's factorization was stamped with. Because this call both writes and
    reads the factorization, threading that token into later calls is what keeps
    them ordered against it, the same as with solve_stateful.
    """
    overlay_key = canonicalize_overlay(options)
    core = _make_factor_and_solve_stateful_core(MatrixType(matrix_type), transpose, overlay_key)
    solution, handle_out, version_out, final_iparm = core(
        token.id, token.version, indptr, indices, values, right_hand_side
    )
    if return_token:
        threaded = FactorizationToken(handle_out, version_out, token.n_dependent_solutions)
        return solution, threaded, final_iparm
    return solution, final_iparm


def release(token, dependency=None):
    """Free the native factorization state for token.

    Always safe for correctness. A later call on the token rebuilds what was
    released. Whether it actually frees depends on ordering. Inside a jit trace
    a release is only ordered after a solve if it consumes something the solve
    produced. Pass that solution as dependency, or call token.track(solution)
    before releasing, so the release waits for it. With neither, the release is
    unordered and may run first and be undone by the solve's rebuild.
    """
    ordering = _ordering_operand(token, dependency)
    return jax.ffi.ffi_call(
        "pardiso_mkl_jax_release",
        jax.ShapeDtypeStruct((), jnp.int32),
        has_side_effect=True,
    )(token.id, ordering)


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
