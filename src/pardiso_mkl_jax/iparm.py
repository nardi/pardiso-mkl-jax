"""Pardiso iparm option codes, override handling, and diagnostics readback.

Pardiso is controlled through a 64-entry integer array (`iparm`, "integer
parameter array") that both takes configuration input and receives some
outputs back after a call. `_pardiso_ffi.cc` fills in a fixed set of safe
defaults (see the "Solver settings" section of the user guide for why each
one is what it is), and this module is what lets callers override any of
them, plus decode the entries Pardiso writes back.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import warnings
from collections.abc import Iterable, Mapping
from typing import cast

import jax
import numpy as np


class PardisoOption(enum.IntEnum):
    """Settable Pardiso iparm entries, one member per documented input option.

    Values are the 0-based iparm index. Reserved entries (which must stay 0)
    and pure-output entries (which Pardiso only ever writes, never reads) are
    intentionally not members here: reserved entries have no reason to be
    touched, and outputs are read back through PardisoDiagnostics instead.

    A few members need special handling in canonicalize_overlay beyond what
    their docstring alone conveys:

    - USE_DEFAULT_VALUES and WEIGHTED_MATCHING can be overridden freely, with
      no runtime warning: setting USE_DEFAULT_VALUES away from 1 is what a
      confirmed MKL segfault in weighted matching was worked around by
      forcing to 1 in the first place (see the "Solver settings" docs), and
      WEIGHTED_MATCHING is the actual crash-triggering setting that
      workaround exists to keep disabled, so a caller touching either one is
      knowingly reaching for the same danger.
    - INDEXING_STYLE can be overridden, but raises a runtime warning: every
      CSR array this package builds is zero-based, so anything else risks
      Pardiso silently misreading indptr and indices.
    - USER_PERMUTATION and PARTIAL_SOLVE_CONTROL cannot be enabled (nonzero
      values are rejected): both read or write through Pardiso's `perm`
      argument, which every handler in _pardiso_ffi.cc passes as null, so
      enabling either dereferences a null pointer.
    - SCHUR_COMPLEMENT_CONTROL cannot be enabled: it needs output buffers
      this package's FFI signatures do not have.
    - TRANSPOSE_SOLVE cannot be set through an overlay at all: use the
      existing `transpose` argument on `solve`/`PardisoSolver.solve` instead,
      which already covers every value this index can meaningfully take for
      the real-valued matrices this package supports.
    """

    USE_DEFAULT_VALUES = 0
    """Whether Pardiso fills in its own defaults (0) or every entry is used as given (nonzero)."""

    FILL_IN_REDUCING_ORDERING = 1
    """Reordering algorithm: 0 minimum degree, 2 METIS nested dissection, 3 parallel nested
    dissection. Default 2."""

    PRECONDITIONED_CGS = 3
    """Krylov subspace iteration and stopping criteria, form 10*L+K with K in {0,1,2}, L>=0.
    Default 0 (off)."""

    USER_PERMUTATION = 4
    """Use of a user-supplied fill-reducing permutation. Default 0. Cannot be enabled here: see
    the class docstring."""

    WRITE_SOLUTION_LOCATION = 5
    """Where the solution is written: 0 to the solution buffer, 1 overwrites the right-hand
    side buffer in place. Default 0."""

    MAX_ITERATIVE_REFINEMENT_STEPS = 7
    """Max iterative refinement iterations for solve: 0 automatic (2 steps), positive an
    explicit cap, negative a cap with extended precision. Default 0."""

    REFINEMENT_TOLERANCE = 8
    """Relative residual tolerance for stopping iterative refinement. Default 0 (Pardiso's own
    default checks apply)."""

    PIVOTING_PERTURBATION = 9
    """Small/zero pivots are perturbed by eps = 10^-iparm[9]. Default 13 for nonsymmetric and
    structurally symmetric matrices, 8 for symmetric ones."""

    SCALING = 10
    """Matrix scaling for diagonal dominance. Default 1 (on) for REAL_NONSYMMETRIC, 0 (off)
    otherwise."""

    TRANSPOSE_SOLVE = 11
    """Which system a solve step solves: 0 Ax=b, 1 conjugate transpose, 2 transpose. Cannot be
    set here: use the `transpose` argument instead, see the class docstring."""

    WEIGHTED_MATCHING = 12
    """Maximum weighted matching for diagonal elements. Default 1 (on) for REAL_NONSYMMETRIC, 0
    (off) otherwise, but this package always defaults it to 0: see the class docstring."""

    REPORT_NONZEROS_IN_FACTORS = 17
    """Report the count of non-zeros in the L and U factors as a diagnostic. Negative enables
    reporting, non-negative disables it. Default -1 (enabled)."""

    REPORT_FACTORIZATION_MFLOPS = 18
    """Report factorization cost in units of 10^6 floating point operations. Negative enables
    reporting, non-negative disables it. Default 0 (disabled)."""

    PIVOTING_STRATEGY = 20
    """Pivoting strategy for symmetric indefinite matrices: 0 pure 1x1, 1 1x1 plus 2x2
    Bunch-Kaufman, 2 and 3 the same without auto-refinement. Default 1."""

    PARALLEL_FACTORIZATION_CONTROL = 23
    """Factorization algorithm variant: 0 classic, 1 two-level, 10 improved two-level
    (nonsymmetric only). Default 0."""

    PARALLEL_SOLVE_CONTROL = 24
    """Parallelization strategy for the solve step: 0 automatic, 1 sequential, 2 matrix
    partitioning. Default 0."""

    MATRIX_CHECKER = 26
    """Validate the ia/ja arrays and column ordering before use. Default 0 (no check)."""

    PRECISION = 27
    """Single (1) or double (0) precision computation. Default 0."""

    PARTIAL_SOLVE_CONTROL = 30
    """Sparse right-hand-side and selective solution output. Default 0. Cannot be enabled here:
    see the class docstring."""

    CNR_THREAD_COUNT = 33
    """OpenMP thread count for conditional numerical reproducibility: 0 auto-determine, positive
    an explicit count. Default 0."""

    INDEXING_STYLE = 34
    """One-based (0) or zero-based (1) array indexing. This package always defaults it to 1.
    Overriding it raises a runtime warning: see the class docstring."""

    SCHUR_COMPLEMENT_CONTROL = 35
    """Compute a Schur complement and in what format. Default 0 (none). Cannot be enabled here:
    see the class docstring."""

    MATRIX_STORAGE_FORMAT = 36
    """Input matrix storage format: 0 CSR, positive a BSR block size, negative a VBSR
    conversion threshold. Default 0."""

    LOW_RANK_UPDATE = 38
    """Enable a low-rank update for a factorization similar to a previous one. Requires
    PARALLEL_FACTORIZATION_CONTROL = 10. Default 0 (off)."""

    INVERSE_DIAGONAL_COMPUTATION = 42
    """Compute the diagonal of the matrix inverse during factorization. Requires
    PARALLEL_FACTORIZATION_CONTROL = 1 and a symmetric matrix type. Default 0 (off). This
    package has no retrieval path for the result, so enabling it has no observable effect here."""

    DIAGONAL_PIVOTING_CALLBACK_CONTROL = 55
    """Enable pivot control and diagonal extraction callbacks (in-core mode only). Default 0
    (off). This package never registers a callback, so enabling it has no observable effect
    here."""

    IN_CORE_MODE = 59
    """In-core (0), automatic (1), or forced out-of-core (2) execution mode. Default 0."""


# Reserved entries, which Pardiso requires to stay 0 and this package never
# needs to touch.
_RESERVED_INDICES = frozenset(
    {2, 25, 28, 31, 32, 37, 39, 40, 41, *range(43, 55), 56, 57, 58, 60, 61, 63}
)

# Entries Pardiso only ever writes, never reads. Read back through
# PardisoDiagnostics instead of set through an overlay.
_OUTPUT_ONLY_INDICES = frozenset({6, 13, 14, 15, 16, 19, 21, 22, 29, 62})

# Entries that cannot be set through an overlay at all, regardless of value:
# TRANSPOSE_SOLVE is fully covered by the existing transpose argument.
_REJECT_ALWAYS_INDICES = frozenset({PardisoOption.TRANSPOSE_SOLVE})

# Entries that can only be left at their default (0): enabling any of these
# reads or writes through a pointer this package always passes as null.
_REJECT_NONZERO_INDICES = frozenset(
    {
        PardisoOption.USER_PERMUTATION,
        PardisoOption.PARTIAL_SOLVE_CONTROL,
        PardisoOption.SCHUR_COMPLEMENT_CONTROL,
    }
)

_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1

# The type callers pass as an iparm overlay: a mapping keyed by PardisoOption
# or a raw int index, canonicalize_overlay's own canonical tuple output (see
# its docstring on idempotency), or None for no overrides. Shared by every
# function in this package that accepts an overlay.
OptionsLike = Mapping[PardisoOption | int, int] | tuple[tuple[int, int], ...] | None


def canonicalize_overlay(options: OptionsLike) -> tuple[tuple[int, int], ...]:
    """Validate an iparm overlay and reduce it to a sorted, hashable (index, value) tuple.

    Accepts either a mapping (the public, user-facing form) or this
    function's own tuple output (idempotent, so internal call sites can pass
    an already-canonicalized overlay straight through without re-deriving it
    from a dict). None is treated as an empty overlay.

    Runs on every call site that accepts an overlay, not just once: this is
    what makes idempotency load-bearing rather than a nicety, since a single
    user-facing call fans out into several internal calls (analyze, factor,
    solve) that each independently canonicalize the same overlay. It also
    means INDEXING_STYLE's runtime warning can fire more than once for a
    single user-facing call; Python's default warning filter deduplicates
    identical warnings from the same source line, so this does not spam.
    """
    if options is None:
        return ()
    items: Iterable[tuple[PardisoOption | int, int]]
    if isinstance(options, Mapping):
        # isinstance narrowing against the bare (unparametrized) Mapping
        # loses the key/value types options was originally declared with,
        # widening them to object: the cast restores what we already know
        # from that declaration.
        items = cast("Iterable[tuple[PardisoOption | int, int]]", options.items())
    else:
        items = options
    canonical: dict[int, int] = {}
    for key, value in items:
        index = int(key)
        if not (0 <= index < 64):
            raise ValueError(f"iparm index {index} is out of range: must satisfy 0 <= index < 64.")
        if index in _RESERVED_INDICES:
            raise ValueError(f"iparm[{index}] is reserved by Pardiso and must not be set.")
        if index in _OUTPUT_ONLY_INDICES:
            raise ValueError(
                f"iparm[{index}] is an output-only entry Pardiso writes after a call, not "
                "something callers can set. Read it back through PardisoDiagnostics instead."
            )
        try:
            name = PardisoOption(index).name
        except ValueError:
            name = index
        if index in _REJECT_ALWAYS_INDICES:
            raise ValueError(
                f"iparm[{index}] ({name}) cannot be set through options: use the transpose "
                "argument instead."
            )
        value = int(value)
        if index in _REJECT_NONZERO_INDICES and value != 0:
            raise ValueError(
                f"iparm[{index}] ({name}) cannot be enabled through options: pardiso_mkl_jax "
                "always passes the corresponding native argument as null, so enabling this "
                "would read or write through a null pointer."
            )
        if index == PardisoOption.INDEXING_STYLE:
            warnings.warn(
                "Overriding INDEXING_STYLE (iparm[34]) changes whether Pardiso expects "
                "one-based or zero-based indices. Every CSR array pardiso_mkl_jax builds is "
                "zero-based, so anything but the default risks Pardiso silently misreading "
                "indptr and indices.",
                stacklevel=2,
            )
        if not (_INT32_MIN <= value <= _INT32_MAX):
            raise ValueError(f"iparm[{index}] value {value} does not fit in a 32-bit integer.")
        canonical[index] = value
    return tuple(sorted(canonical.items()))


def merge_overlays(base: OptionsLike, override: OptionsLike) -> tuple[tuple[int, int], ...]:
    """Combine two iparm overlays, with override winning on any index both set.

    Used for PardisoSolver's solver-wide options, which every call layers its
    own per-call options on top of. Both sides go through
    canonicalize_overlay, so an invalid entry is rejected the same way here as
    anywhere else and the result is canonical in its own right.
    """
    merged = dict(canonicalize_overlay(base))
    merged.update(canonicalize_overlay(override))
    return tuple(sorted(merged.items()))


@functools.cache
def overlay_to_arrays(canonical: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, np.ndarray]:
    """Expand a canonical overlay into the (mask, values) int32[64] pair the FFI call expects.

    Cached on the canonical tuple: call sites that share the same overlay
    across many calls (a values-batched vmap loop, repeated PardisoSolver
    calls) reuse the same pair of arrays instead of rebuilding them each
    time. The returned arrays are marked read-only, since a cached array
    must never be mutated in place by a caller.
    """
    mask = np.zeros(64, dtype=np.int32)
    values = np.zeros(64, dtype=np.int32)
    for index, value in canonical:
        mask[index] = 1
        values[index] = value
    mask.setflags(write=False)
    values.setflags(write=False)
    return mask, values


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PardisoDiagnostics:
    """Pardiso's write-back outputs, read from the final iparm array after a call.

    Built by from_iparm, which is plain array indexing with no concretizing
    operation anywhere in it, so it is trace-safe: it produces a correct
    result whether iparm is a concrete array (the ordinary eager case) or
    still traced (when the call producing it is itself wrapped in jax.jit).
    This is why every field here is a JAX scalar array rather than a Python
    int, and why the class is registered as a JAX pytree: a plain dataclass
    of Python ints could not flow through a jitted function as an output.

    Under vmap, every field's array still gains a batch dimension by default
    (jax.vmap's default out_axes=0 broadcasts any output, batched or not),
    but the values differ by case: when a solve genuinely runs once per
    batch element (batching over matrix values), each entry is that
    element's own diagnostics; when one native call already covers the
    whole batch (batching only over right-hand sides), Pardiso reports
    diagnostics once, and every entry along the batch dimension is that same
    value, broadcast rather than recomputed. A caller who wants the compact,
    un-broadcast form in the second case can request it explicitly with
    jax.vmap's own out_axes argument. See the "Batching with vmap" and
    "Diagnostics" sections of the user guide for worked examples.

    Only ever available on a successful call: a Pardiso error raises a
    Python exception instead of returning a value, and a function that
    raises cannot also return diagnostics, on success or partial failure.
    """

    refinement_steps_performed: jax.Array
    """Iterative refinement steps actually run (iparm[6])."""

    perturbed_pivot_count: jax.Array
    """Number of pivots perturbed during factorization (iparm[13])."""

    peak_memory_symbolic_kb: jax.Array
    """Peak memory used during the symbolic (analysis) phase, in KB (iparm[14])."""

    permanent_memory_symbolic_kb: jax.Array
    """Permanent memory retained from the symbolic phase, in KB (iparm[15])."""

    peak_memory_numerical_kb: jax.Array
    """Peak memory used during numerical factorization, in KB (iparm[16])."""

    nonzeros_in_factors: jax.Array
    """Non-zero count in the L and U factors, if REPORT_NONZEROS_IN_FACTORS was set (iparm[17])."""

    factorization_mflops: jax.Array
    """Factorization cost in units of 10^6 FLOPs, if REPORT_FACTORIZATION_MFLOPS was set
    (iparm[18])."""

    cgs_diagnostics: jax.Array
    """Krylov (CG/CGS) iteration diagnostics, if PRECONDITIONED_CGS was set (iparm[19])."""

    positive_eigenvalues: jax.Array
    """Positive eigenvalue count, for symmetric indefinite matrices (iparm[21])."""

    negative_eigenvalues: jax.Array
    """Negative eigenvalue count, for symmetric indefinite matrices (iparm[22])."""

    zero_or_negative_pivot_position: jax.Array
    """Position of the first zero or negative pivot encountered (iparm[29])."""

    min_out_of_core_memory_kb: jax.Array
    """Minimum memory required for out-of-core factorization, in KB (iparm[62])."""

    raw: jax.Array
    """All 64 iparm entries, for anything not decoded into a named field above."""

    @staticmethod
    def from_iparm(iparm: jax.Array) -> PardisoDiagnostics:
        """Decode a final iparm array (shape (..., 64)) into a PardisoDiagnostics."""
        return PardisoDiagnostics(
            refinement_steps_performed=iparm[..., 6],
            perturbed_pivot_count=iparm[..., 13],
            peak_memory_symbolic_kb=iparm[..., 14],
            permanent_memory_symbolic_kb=iparm[..., 15],
            peak_memory_numerical_kb=iparm[..., 16],
            nonzeros_in_factors=iparm[..., 17],
            factorization_mflops=iparm[..., 18],
            cgs_diagnostics=iparm[..., 19],
            positive_eigenvalues=iparm[..., 21],
            negative_eigenvalues=iparm[..., 22],
            zero_or_negative_pivot_position=iparm[..., 29],
            min_out_of_core_memory_kb=iparm[..., 62],
            raw=iparm,
        )
