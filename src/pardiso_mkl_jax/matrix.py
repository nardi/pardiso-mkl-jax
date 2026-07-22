"""Pardiso matrix type codes and CSR input validation."""

from __future__ import annotations

import enum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class MatrixType(enum.IntEnum):
    """Pardiso matrix type codes, matching the `mtype` parameter.

    Only the real-valued members can be used in this version of the package,
    since it works with float64 values throughout. The complex members are
    included so the full set of matrix types Pardiso itself supports is
    documented here, and raise NotImplementedError if selected.

    The members whose values are mathematically symmetric or Hermitian
    (REAL_SYMMETRIC_POSITIVE_DEFINITE, REAL_SYMMETRIC_INDEFINITE,
    COMPLEX_HERMITIAN_POSITIVE_DEFINITE, COMPLEX_HERMITIAN_INDEFINITE, and
    COMPLEX_SYMMETRIC) require the CSR arrays to hold only the upper
    triangle, including the diagonal, not the full matrix: passing the full
    matrix corrupts Pardiso's factorization instead of raising a clear
    error. check_upper_triangular enforces this.

    The structurally symmetric members (REAL_STRUCTURALLY_SYMMETRIC and
    COMPLEX_STRUCTURALLY_SYMMETRIC) only assume a symmetric sparsity
    pattern, not symmetric values, so they need the full matrix like the
    nonsymmetric members do.
    """

    REAL_STRUCTURALLY_SYMMETRIC = 1
    """Real values, symmetric sparsity pattern, no assumption on the values themselves."""

    REAL_SYMMETRIC_POSITIVE_DEFINITE = 2
    """Real, symmetric, and positive definite. The cheapest and most stable case to factor."""

    REAL_SYMMETRIC_INDEFINITE = -2
    """Real and symmetric, but not guaranteed positive definite."""

    COMPLEX_STRUCTURALLY_SYMMETRIC = 3
    """Complex values, symmetric sparsity pattern. Not yet supported: requires complex values."""

    COMPLEX_HERMITIAN_POSITIVE_DEFINITE = 4
    """Complex and Hermitian positive definite. Not yet supported: requires complex values."""

    COMPLEX_HERMITIAN_INDEFINITE = -4
    """Complex and Hermitian, not guaranteed positive definite. Not yet supported."""

    COMPLEX_SYMMETRIC = 6
    """Complex and symmetric. Not yet supported: requires complex values."""

    REAL_NONSYMMETRIC = 11
    """Real, with no assumed symmetry. The general-purpose default matrix type."""

    COMPLEX_NONSYMMETRIC = 13
    """Complex, with no assumed symmetry. Not yet supported: requires complex values."""


# The matrix types this version can actually solve. All others require complex
# values, which the extension does not accept yet.
_SUPPORTED_MATRIX_TYPES = frozenset(
    (
        MatrixType.REAL_STRUCTURALLY_SYMMETRIC,
        MatrixType.REAL_SYMMETRIC_POSITIVE_DEFINITE,
        MatrixType.REAL_SYMMETRIC_INDEFINITE,
        MatrixType.REAL_NONSYMMETRIC,
    )
)

# Matrix types for which Pardiso expects only the upper triangle (including
# the diagonal) to be stored, not the full matrix. This is exactly the
# matrix types whose values are mathematically symmetric or Hermitian, not
# the structurally symmetric ones: those only assume a symmetric sparsity
# pattern, and Pardiso still needs both triangles of values for them.
_SYMMETRIC_MATRIX_TYPES = frozenset(
    (
        MatrixType.REAL_SYMMETRIC_POSITIVE_DEFINITE,
        MatrixType.REAL_SYMMETRIC_INDEFINITE,
        MatrixType.COMPLEX_HERMITIAN_POSITIVE_DEFINITE,
        MatrixType.COMPLEX_HERMITIAN_INDEFINITE,
        MatrixType.COMPLEX_SYMMETRIC,
    )
)


def check_matrix_type_supported(matrix_type: MatrixType) -> None:
    """Raise NotImplementedError for a matrix type this version cannot solve."""
    if matrix_type not in _SUPPORTED_MATRIX_TYPES:
        raise NotImplementedError(
            f"{matrix_type.name} requires complex values, which pardiso_mkl_jax does not "
            "support yet. Use one of the real matrix types instead."
        )


def check_csr_arrays(indptr, indices, values) -> None:
    """Validate that indptr and indices are int32 and values is float64.

    These dtypes are required, not just preferred: they are exactly what the
    compiled extension expects. Silently casting a mismatched array would
    make a copy and break the zero-copy interface this package promises, so a
    mismatch is treated as a caller error instead. Convert explicitly before
    calling in if your arrays are in another dtype.
    """
    for name, array, expected_dtype_name in (
        ("indptr", indptr, "int32"),
        ("indices", indices, "int32"),
        ("values", values, "float64"),
    ):
        if array.dtype.name != expected_dtype_name:
            raise TypeError(
                f"{name} must have dtype {expected_dtype_name}, got {array.dtype}. "
                "pardiso_mkl_jax never converts array dtypes implicitly, since that would "
                "silently copy data and break the zero-copy interface."
            )
    if indptr.ndim != 1 or indices.ndim != 1 or values.ndim != 1:
        raise ValueError("indptr, indices, and values must all be one-dimensional.")
    if indices.shape != values.shape:
        raise ValueError(
            f"indices and values must have the same length, got {indices.shape[0]} "
            f"and {values.shape[0]}."
        )


def _upper_triangular_message(matrix_type: MatrixType) -> str:
    return (
        f"{matrix_type.name} requires the CSR arrays to store only the upper triangle "
        "(column index >= row index for every entry), including the diagonal. Drop the "
        "lower-triangle entries before calling in."
    )


def check_upper_triangular(indptr, indices, matrix_type: MatrixType):
    """For symmetric matrix types, validate that only the upper triangle is stored.

    Pardiso's symmetric matrix types expect the CSR arrays to hold only the
    upper triangle, including the diagonal, not the full matrix. This is not
    just wasted memory if violated: passing the full matrix corrupts
    Pardiso's internal factorization instead of raising a clear error, which
    was confirmed directly against the compiled library rather than inferred
    from documentation. Non-symmetric matrix types are unaffected: the full
    matrix is exactly what they expect, so this check is skipped for them.

    Returns indices unchanged. When indptr and indices are concrete (the
    common eager case), a violation raises ValueError immediately. When
    they are traced instead (under jit or vmap), the same check is built
    into the traced program with equinox.error_if, so it still fires at
    runtime rather than being silently skipped, though it then reports a
    RuntimeError: JAX has no mechanism to raise a plain Python exception
    from inside a traced program. Callers must use the returned value, not
    the original indices, or the traced check is optimized away as dead
    code.
    """
    if matrix_type not in _SYMMETRIC_MATRIX_TYPES:
        return indices
    try:
        indptr_values = np.asarray(indptr)
        indices_values = np.asarray(indices)
    except jax.errors.TracerArrayConversionError:
        row_of_entry = jnp.repeat(
            jnp.arange(indptr.shape[0] - 1),
            jnp.diff(indptr),
            total_repeat_length=indices.shape[0],
        )
        violation = jnp.any(indices < row_of_entry)
        return eqx.error_if(indices, violation, _upper_triangular_message(matrix_type))
    row_of_entry = np.repeat(np.arange(indptr_values.shape[0] - 1), np.diff(indptr_values))
    if np.any(indices_values < row_of_entry):
        raise ValueError(_upper_triangular_message(matrix_type))
    return indices


def matrix_dimension(indptr) -> int:
    """The order n of a square CSR matrix, derived from an indptr of length n + 1."""
    if indptr.shape[0] < 1:
        raise ValueError("indptr must have at least one entry.")
    return indptr.shape[0] - 1
