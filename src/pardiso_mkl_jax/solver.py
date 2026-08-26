"""PardisoSolver: keeps a Pardiso factorization alive so it can be reused."""

from __future__ import annotations

from typing import Literal, overload

import jax
import jax.core

from pardiso_mkl_jax import primitive
from pardiso_mkl_jax.iparm import (
    OptionsLike,
    PardisoDiagnostics,
    PardisoOption,
    canonicalize_overlay,
    merge_overlays,
)
from pardiso_mkl_jax.matrix import (
    MatrixType,
    check_csr_arrays,
    check_matrix_type_supported,
    check_upper_triangular,
    matrix_dimension,
)


class PardisoSolver:
    """Reuses a single Pardiso factorization across many solves.

    The sparsity pattern (indptr and indices) is fixed for the solver's
    lifetime. The three Pardiso stages are kept as separate calls so callers
    control exactly what work happens on each one:

    - analyze() runs the symbolic phase for the pattern. Calling it again on
      the same solver re-analyzes in place, freeing the numeric factorization
      and reusing the same native handle rather than allocating a second one.
    - factorize() runs the first numeric factorization, and requires a prior
      analyze().
    - refactorize() updates the numeric factorization for new values on the
      same pattern, and requires a prior factorize(). It runs the same
      Pardiso phase as factorize(); the separate name and precondition make
      the reuse explicit at the call site.
    - solve() solves against whatever factorization is currently stored, and
      requires a prior factorize().
    - refactor_and_solve() factorizes for new values and solves in one call,
      reusing the analysis and requiring only a prior analyze(). It keeps no
      reference to the values, so unlike factorize() plus solve() it is safe
      to call from inside a jitted function where the values and right-hand
      side are tracers.

    Pardiso's parameters are recomputed fresh on every native call rather
    than persisted, so an options overlay (see
    pardiso_mkl_jax.iparm.PardisoOption) passed to a single method applies
    only to that call. The constructor's own options argument is the way to
    set an overlay for the solver's whole lifetime: it is applied to every
    call, and a per-call options argument layers on top of it, winning on any
    entry both set.

    Each call also records its diagnostics (see PardisoDiagnostics),
    readable afterward from last_diagnostics. Every method also takes
    return_diagnostics, which hands them back directly instead. That is the
    form to use under jit, where last_diagnostics is unavailable; see its
    docstring.

    PardisoSolver may be used as a context manager, which releases its native
    memory on exit, but this is optional. The cache behind every handle is
    bounded (set by PARDISO_MKL_JAX_FACTOR_CACHE), and any factorization that is
    evicted or released is rebuilt on next use from the matrix the call carries.
    So a solver that is never closed leaks at most one cache slot rather than
    unbounded memory, and reusing it after close() still works, it just rebuilds
    once. Close it, or use the with-block, to free that slot promptly.

        with PardisoSolver(indptr, indices, matrix_type=MatrixType.REAL_NONSYMMETRIC) as solver:
            solver.analyze(values)
            solver.factorize(values)
            x = solver.solve(b)

    The same calls work without the with-block, and close() stays available to
    release early.
    """

    def __init__(self, indptr, indices, *, matrix_type: MatrixType, options: OptionsLike = None):
        check_matrix_type_supported(matrix_type)
        if indptr.dtype.name != "int32" or indices.dtype.name != "int32":
            raise TypeError("indptr and indices must have dtype int32.")
        self._indptr = indptr
        self._indices = check_upper_triangular(indptr, indices, matrix_type)
        self._matrix_type = matrix_type
        self._dimension = matrix_dimension(indptr)
        # Validated once here rather than on every call that merges it in.
        self._options = canonicalize_overlay(options)
        # The token is only obtained from analyze(), which allocates the native
        # factorization it names, so there is nothing to hold until then.
        self._handle: primitive.FactorizationToken | None = None
        self._values = None
        self._closed = False
        self._analyzed = False
        self._factorized = False
        self._last_diagnostics: PardisoDiagnostics | None = None
        # Effective SCALING and WEIGHTED_MATCHING at the last analyze, which
        # every later phase is checked against. See _check_pivot_settings.
        self._analysis_pivot_settings: dict[int, int] = {}

    def __enter__(self) -> PardisoSolver:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Release the native factorization. Called automatically by __exit__.

        Safe to call at any time. A later use of the solver rebuilds what was
        released. It only reclaims memory without a wasted rebuild when it runs
        after the last solve, which is the case here since close runs eagerly in
        Python. See the advanced-usage guide on releasing explicitly.
        """
        if not self._closed:
            if self._handle is not None:
                primitive.release(self._handle)
            self._closed = True

    @property
    def last_diagnostics(self) -> PardisoDiagnostics | None:
        """Diagnostics from the most recent analyze, factorize, refactorize, or solve call.

        None before any call has been made. Only ever reflects a successful
        call: a Pardiso error raises instead of returning, so a failed call
        leaves this at whatever it was before that call.

        Also None after a call that ran under jit, since the diagnostics only
        exist as tracers there and storing one would leak it out of its trace.
        Pass return_diagnostics to read diagnostics from a traced call, which
        is what refactor_and_solve() does for every call, traced or not: it
        never updates this at all.
        """
        return self._last_diagnostics

    def _merge(self, options: OptionsLike) -> tuple[tuple[int, int], ...]:
        """Layer a per-call overlay on top of the solver-wide one, per-call winning."""
        return merge_overlays(self._options, options)

    def _pivot_settings(self, overlay: tuple[tuple[int, int], ...]) -> dict[int, int]:
        """The values SCALING and WEIGHTED_MATCHING will actually take for a call.

        An entry the overlay sets takes that value; anything else falls back
        to this package's default for the matrix type.
        """
        defaults = primitive.default_iparm(self._matrix_type)
        entries = dict(overlay)
        return {
            index: int(entries.get(index, defaults[index]))
            for index in (PardisoOption.SCALING, PardisoOption.WEIGHTED_MATCHING)
        }

    def _check_pivot_settings(self, overlay: tuple[tuple[int, int], ...], stage: str) -> None:
        """Reject a call whose scaling or matching disagrees with the analysis.

        Pardiso computes its scaling and matching during analysis and expects
        the same settings at every later phase. Nothing in the native layer
        enforces that: each handler rebuilds iparm from scratch, so a
        disagreement is silently accepted and produces a wrong answer rather
        than an error. Catching it here is the only place it shows up.
        """
        for index, value in self._pivot_settings(overlay).items():
            analysis_value = self._analysis_pivot_settings[index]
            if value != analysis_value:
                name = PardisoOption(index).name
                raise ValueError(
                    f"{stage} would run with {name} (iparm[{index}]) = {value}, but the "
                    f"analysis for this solver ran with {analysis_value}. Pardiso computes "
                    "scaling and matching during analysis and expects them unchanged "
                    "afterwards. Set this option on the PardisoSolver constructor so it "
                    "applies to every call, or re-run analyze() with the new value."
                )

    def _record_diagnostics(self, final_iparm) -> PardisoDiagnostics:
        """Decode diagnostics, store them on the solver, and return them.

        Stores None instead when final_iparm is a tracer, which it is whenever
        the call is running under jit. Keeping the decoded tracer would leak
        it out of its trace, so reading last_diagnostics afterwards would
        raise rather than return anything useful. The returned value is the
        real one either way, so return_diagnostics still works under jit.
        """
        diagnostics = PardisoDiagnostics.from_iparm(final_iparm)
        self._last_diagnostics = None if isinstance(final_iparm, jax.core.Tracer) else diagnostics
        return diagnostics

    def _check_usable(self) -> None:
        # The context manager is optional now that a released factorization
        # rebuilds itself, so only a genuine close() blocks further use.
        if self._closed:
            raise RuntimeError("PardisoSolver is closed and can no longer be used.")

    # The overloads on analyze, factorize, and refactorize are here so that
    # `diagnostics = solver.factorize(values, return_diagnostics=True)` types
    # as a plain PardisoDiagnostics for callers, rather than something
    # optional they have to narrow before reading a field off it.
    @overload
    def analyze(
        self,
        values,
        *,
        options: OptionsLike = None,
        return_diagnostics: Literal[False] = False,
    ) -> None: ...

    @overload
    def analyze(
        self, values, *, options: OptionsLike = None, return_diagnostics: Literal[True]
    ) -> PardisoDiagnostics: ...

    def analyze(
        self, values, *, options: OptionsLike = None, return_diagnostics: bool = False
    ) -> PardisoDiagnostics | None:
        """Run the symbolic analysis (fill-reducing ordering) for the stored pattern.

        Takes a representative values array because Pardiso's default
        heuristics for non-symmetric matrices, scaling and matching, look at
        the numeric values during analysis. The permutation and scaling this
        produces stay valid for a later factorize() call with different
        values on the same pattern, so this only needs to run once.

        Calling it again on the same solver re-analyzes in place: the existing
        numeric factorization is freed and the same native handle is reused,
        so no second handle is allocated and nothing extra needs releasing.
        factorize() must run again afterwards before any solve(), since the
        factorization the re-analysis discarded is the one solve() would have
        used.

        Must be called outside jit. It stores the native handle on the solver,
        and under jit that handle is a tracer, which would escape its trace.
        Callers who need the whole lifecycle inside a jitted function use the
        pardiso_mkl_jax.primitive functions and thread the handle themselves.

        Returns the call's PardisoDiagnostics if return_diagnostics is set,
        and None otherwise.
        """
        self._check_usable()
        check_csr_arrays(self._indptr, self._indices, values)
        overlay = self._merge(options)
        if self._handle is None:
            self._handle, final_iparm = primitive.analyze(
                self._indptr,
                self._indices,
                values,
                matrix_type=self._matrix_type,
                options=overlay,
            )
        else:
            # Cleared before the call, not after. Re-analysis frees the
            # existing factorization first thing, so if it then fails there is
            # no analysis and no factors left to use, and the solver has to
            # say so rather than report the state it had going in.
            self._analyzed = False
            self._factorized = False
            self._values = None
            self._handle, final_iparm = primitive.reanalyze(
                self._handle,
                self._indptr,
                self._indices,
                values,
                matrix_type=self._matrix_type,
                options=overlay,
            )
        self._analysis_pivot_settings = self._pivot_settings(overlay)
        self._analyzed = True
        diagnostics = self._record_diagnostics(final_iparm)
        return diagnostics if return_diagnostics else None

    @overload
    def factorize(
        self,
        values,
        *,
        options: OptionsLike = None,
        return_diagnostics: Literal[False] = False,
    ) -> None: ...

    @overload
    def factorize(
        self, values, *, options: OptionsLike = None, return_diagnostics: Literal[True]
    ) -> PardisoDiagnostics: ...

    def factorize(
        self, values, *, options: OptionsLike = None, return_diagnostics: bool = False
    ) -> PardisoDiagnostics | None:
        """Run the first numeric factorization for values. Requires a prior analyze().

        Returns the call's PardisoDiagnostics if return_diagnostics is set,
        and None otherwise. That is where perturbed_pivot_count lives, which
        is Pardiso's own report that it could not pivot cleanly and the
        factorization may be unusable.
        """
        self._check_usable()
        if not self._analyzed:
            raise RuntimeError("factorize() requires analyze() to have been called first.")
        diagnostics = self._run_numeric_factorization(values, options=options, stage="factorize()")
        self._factorized = True
        return diagnostics if return_diagnostics else None

    @overload
    def refactorize(
        self,
        values,
        *,
        options: OptionsLike = None,
        return_diagnostics: Literal[False] = False,
    ) -> None: ...

    @overload
    def refactorize(
        self, values, *, options: OptionsLike = None, return_diagnostics: Literal[True]
    ) -> PardisoDiagnostics: ...

    def refactorize(
        self, values, *, options: OptionsLike = None, return_diagnostics: bool = False
    ) -> PardisoDiagnostics | None:
        """Update the numeric factorization with new values on the same pattern.

        Requires a prior factorize(). Skips the analysis phase, which is the
        cheap path when only the matrix values change between solves.

        Returns the call's PardisoDiagnostics if return_diagnostics is set,
        and None otherwise.
        """
        self._check_usable()
        if not self._factorized:
            raise RuntimeError("refactorize() requires factorize() to have been called first.")
        diagnostics = self._run_numeric_factorization(
            values, options=options, stage="refactorize()"
        )
        return diagnostics if return_diagnostics else None

    def _run_numeric_factorization(
        self, values, *, options: OptionsLike, stage: str
    ) -> PardisoDiagnostics:
        check_csr_arrays(self._indptr, self._indices, values)
        overlay = self._merge(options)
        self._check_pivot_settings(overlay, stage)
        self._handle, final_iparm = primitive.factor(
            self._handle,
            self._indptr,
            self._indices,
            values,
            matrix_type=self._matrix_type,
            options=overlay,
        )
        self._values = values
        return self._record_diagnostics(final_iparm)

    def solve(
        self,
        right_hand_side,
        *,
        transpose: bool = False,
        options: OptionsLike = None,
        return_diagnostics: bool = False,
    ):
        """Solve against the current factorization. Requires a prior factorize().

        Solves against A^T instead of A when transpose is set, reusing the
        same factorization: no extra factorize() call is needed to switch
        between the two, and consecutive calls with different transpose
        values are safe.

        Returns just the solution by default, or (solution, PardisoDiagnostics)
        if return_diagnostics is set. The latter is the form to use under jit,
        where last_diagnostics stays None.
        """
        self._check_usable()
        if not self._factorized:
            raise RuntimeError("solve() requires factorize() to have been called first.")
        overlay = self._merge(options)
        self._check_pivot_settings(overlay, "solve()")
        stacked_right_hand_side = right_hand_side[None, :]
        solution, final_iparm = primitive.solve_stateful(
            self._handle,
            self._indptr,
            self._indices,
            self._values,
            stacked_right_hand_side,
            matrix_type=self._matrix_type,
            transpose=transpose,
            options=overlay,
        )
        diagnostics = self._record_diagnostics(final_iparm)
        if return_diagnostics:
            return solution[0], diagnostics
        return solution[0]

    def refactor_and_solve(
        self,
        values,
        right_hand_side,
        *,
        transpose: bool = False,
        options: OptionsLike = None,
        return_diagnostics: bool = False,
    ):
        """Factorize for values and solve in one call, reusing the analysis.

        Requires a prior analyze(). Runs the numeric factorization and the
        solve as one combined Pardiso step (phase 23), reusing the symbolic
        analysis rather than re-running it. Because it is a single FFI call it
        stays correct inside a jitted function, where a separate factorize()
        and solve() would not be ordered, and it keeps no reference to values
        on the solver, so values and right_hand_side may be tracers.

        Solves against A^T instead of A when transpose is set.

        This is the one method that does not record its diagnostics on
        last_diagnostics even when it runs eagerly, since keeping nothing at
        all on the solver is the whole point of it. Pass return_diagnostics to
        get them back as a second return value instead.
        """
        self._check_usable()
        if not self._analyzed:
            raise RuntimeError("refactor_and_solve() requires analyze() to have been called first.")
        check_csr_arrays(self._indptr, self._indices, values)
        overlay = self._merge(options)
        self._check_pivot_settings(overlay, "refactor_and_solve()")
        stacked_right_hand_side = right_hand_side[None, :]
        solution, final_iparm = primitive.factor_and_solve_stateful(
            self._handle,
            self._indptr,
            self._indices,
            values,
            stacked_right_hand_side,
            matrix_type=self._matrix_type,
            transpose=transpose,
            options=overlay,
        )
        if return_diagnostics:
            return solution[0], PardisoDiagnostics.from_iparm(final_iparm)
        return solution[0]
