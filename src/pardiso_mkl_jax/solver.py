"""PardisoSolver: keeps a Pardiso factorization alive so it can be reused."""

from __future__ import annotations

from pardiso_mkl_jax import primitive
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

    - analyze() runs the symbolic phase once for the pattern.
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

    PardisoSolver must be used as a context manager. Its native memory is
    released in __exit__, not in a destructor: Python does not guarantee when
    or whether __del__ runs, so relying on it could leave the native
    factorization alive far longer than intended, or leak it entirely if the
    interpreter is shutting down.

        with PardisoSolver(indptr, indices, matrix_type=MatrixType.REAL_NONSYMMETRIC) as solver:
            solver.analyze(values)
            solver.factorize(values)
            x = solver.solve(b)
    """

    def __init__(self, indptr, indices, *, matrix_type: MatrixType):
        check_matrix_type_supported(matrix_type)
        if indptr.dtype.name != "int32" or indices.dtype.name != "int32":
            raise TypeError("indptr and indices must have dtype int32.")
        self._indptr = indptr
        self._indices = check_upper_triangular(indptr, indices, matrix_type)
        self._matrix_type = matrix_type
        self._dimension = matrix_dimension(indptr)
        self._solver_id = primitive.allocate_solver_id()
        self._values = None
        self._entered = False
        self._closed = False
        self._analyzed = False
        self._factorized = False

    def __enter__(self) -> PardisoSolver:
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Release the native factorization. Called automatically by __exit__."""
        if not self._closed:
            primitive.release(solver_id=self._solver_id)
            self._closed = True

    def _check_usable(self) -> None:
        if not self._entered:
            raise RuntimeError(
                "PardisoSolver must be used as a context manager: "
                "'with PardisoSolver(...) as solver: ...'."
            )
        if self._closed:
            raise RuntimeError("PardisoSolver is closed and can no longer be used.")

    def analyze(self, values) -> None:
        """Run the symbolic analysis (fill-reducing ordering) for the stored pattern.

        Takes a representative values array because Pardiso's default
        heuristics for non-symmetric matrices, scaling and matching, look at
        the numeric values during analysis. The permutation and scaling this
        produces stay valid for a later factorize() call with different
        values on the same pattern, so this only needs to run once.
        """
        self._check_usable()
        check_csr_arrays(self._indptr, self._indices, values)
        primitive.factor(
            self._indptr,
            self._indices,
            values,
            solver_id=self._solver_id,
            phase=primitive.PHASE_ANALYZE,
            matrix_type=self._matrix_type,
        )
        self._analyzed = True

    def factorize(self, values) -> None:
        """Run the first numeric factorization for values. Requires a prior analyze()."""
        self._check_usable()
        if not self._analyzed:
            raise RuntimeError("factorize() requires analyze() to have been called first.")
        self._run_numeric_factorization(values)
        self._factorized = True

    def refactorize(self, values) -> None:
        """Update the numeric factorization with new values on the same pattern.

        Requires a prior factorize(). Skips the analysis phase, which is the
        cheap path when only the matrix values change between solves.
        """
        self._check_usable()
        if not self._factorized:
            raise RuntimeError("refactorize() requires factorize() to have been called first.")
        self._run_numeric_factorization(values)

    def _run_numeric_factorization(self, values) -> None:
        check_csr_arrays(self._indptr, self._indices, values)
        primitive.factor(
            self._indptr,
            self._indices,
            values,
            solver_id=self._solver_id,
            phase=primitive.PHASE_FACTORIZE,
            matrix_type=self._matrix_type,
        )
        self._values = values

    def solve(self, right_hand_side, *, transpose: bool = False):
        """Solve against the current factorization. Requires a prior factorize().

        Solves against A^T instead of A when transpose is set, reusing the
        same factorization: no extra factorize() call is needed to switch
        between the two, and consecutive calls with different transpose
        values are safe.
        """
        self._check_usable()
        if not self._factorized:
            raise RuntimeError("solve() requires factorize() to have been called first.")
        stacked_right_hand_side = right_hand_side[None, :]
        solution = primitive.solve_stateful(
            self._indptr,
            self._indices,
            self._values,
            stacked_right_hand_side,
            solver_id=self._solver_id,
            matrix_type=self._matrix_type,
            transpose=transpose,
        )
        return solution[0]

    def refactor_and_solve(self, values, right_hand_side, *, transpose: bool = False):
        """Factorize for values and solve in one call, reusing the analysis.

        Requires a prior analyze(). Runs the numeric factorization and the
        solve as one combined Pardiso step (phase 23), reusing the symbolic
        analysis rather than re-running it. Because it is a single FFI call it
        stays correct inside a jitted function, where a separate factorize()
        and solve() would not be ordered, and it keeps no reference to values
        on the solver, so values and right_hand_side may be tracers.

        Solves against A^T instead of A when transpose is set.
        """
        self._check_usable()
        if not self._analyzed:
            raise RuntimeError("refactor_and_solve() requires analyze() to have been called first.")
        check_csr_arrays(self._indptr, self._indices, values)
        stacked_right_hand_side = right_hand_side[None, :]
        solution = primitive.factor_and_solve_stateful(
            self._indptr,
            self._indices,
            values,
            stacked_right_hand_side,
            solver_id=self._solver_id,
            matrix_type=self._matrix_type,
            transpose=transpose,
        )
        return solution[0]
