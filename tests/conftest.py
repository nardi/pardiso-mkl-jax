"""Shared pytest fixtures and helpers for the pardiso_mkl_jax test suite."""

from __future__ import annotations

import jax
import numpy as np
import pytest
import scipy.sparse

import pardiso_mkl_jax as pmj

jax.config.update("jax_enable_x64", True)

# Every matrix type this version of the package can actually solve. Tests
# that perform a solve are parametrized over this tuple through the
# `system` fixture below, so each one runs once per matrix type instead of
# only against a single default.
SUPPORTED_MATRIX_TYPES = (
    pmj.MatrixType.REAL_STRUCTURALLY_SYMMETRIC,
    pmj.MatrixType.REAL_SYMMETRIC_POSITIVE_DEFINITE,
    pmj.MatrixType.REAL_SYMMETRIC_INDEFINITE,
    pmj.MatrixType.REAL_NONSYMMETRIC,
)

# The matrix types this version cannot solve, since they require complex
# values. Used to check that each one is rejected the same way.
UNSUPPORTED_MATRIX_TYPES = (
    pmj.MatrixType.COMPLEX_STRUCTURALLY_SYMMETRIC,
    pmj.MatrixType.COMPLEX_HERMITIAN_POSITIVE_DEFINITE,
    pmj.MatrixType.COMPLEX_HERMITIAN_INDEFINITE,
    pmj.MatrixType.COMPLEX_SYMMETRIC,
    pmj.MatrixType.COMPLEX_NONSYMMETRIC,
)

# The supported matrix types whose values are mathematically symmetric, and
# so require upper-triangular CSR storage. Used to check that passing the
# full matrix is rejected for each of them.
UPPER_TRIANGULAR_MATRIX_TYPES = (
    pmj.MatrixType.REAL_SYMMETRIC_POSITIVE_DEFINITE,
    pmj.MatrixType.REAL_SYMMETRIC_INDEFINITE,
)


def random_nonsymmetric_csr(dimension: int, density: float, seed: int):
    """A random, diagonally dominant, invertible CSR matrix with no symmetry assumed.

    Diagonal dominance keeps the matrix comfortably invertible for random
    tests, without needing to reject singular draws.
    """
    random_state = np.random.default_rng(seed)
    dense = scipy.sparse.random(
        dimension, dimension, density=density, random_state=random_state, format="csr"
    ).toarray()
    np.fill_diagonal(dense, np.abs(dense).sum(axis=1) + 1.0)
    sparse = scipy.sparse.csr_matrix(dense)
    return (
        sparse.indptr.astype(np.int32),
        sparse.indices.astype(np.int32),
        sparse.data.astype(np.float64),
        dense,
    )


def random_structurally_symmetric_csr(dimension: int, density: float, seed: int):
    """A random, diagonally dominant, invertible CSR matrix with a symmetric sparsity pattern.

    The nonzero positions are mirrored across the diagonal, but each
    position gets an independent random value, so the values themselves are
    not symmetric. Pardiso's structurally symmetric matrix types assume
    only the former, and need the full matrix, not just the upper
    triangle, since the values are not assumed symmetric.
    """
    random_state = np.random.default_rng(seed)
    pattern = scipy.sparse.random(
        dimension, dimension, density=density, random_state=random_state, format="csr"
    ).toarray()
    pattern = (pattern != 0.0) | (pattern != 0.0).T
    dense = random_state.uniform(-1.0, 1.0, size=(dimension, dimension)) * pattern
    np.fill_diagonal(dense, np.abs(dense).sum(axis=1) + 1.0)
    sparse = scipy.sparse.csr_matrix(dense)
    return (
        sparse.indptr.astype(np.int32),
        sparse.indices.astype(np.int32),
        sparse.data.astype(np.float64),
        dense,
    )


def random_spd_csr(dimension: int, density: float, seed: int):
    """A random symmetric positive definite matrix, stored as upper-triangular CSR.

    Pardiso's symmetric matrix types expect only the upper triangle, so the
    dense reference matrix (used to check results) is the full symmetric
    matrix, but the CSR arrays returned only cover its upper triangle.
    """
    random_state = np.random.default_rng(seed)
    factor = scipy.sparse.random(
        dimension, dimension, density=density, random_state=random_state, format="csr"
    ).toarray()
    dense = factor @ factor.T
    np.fill_diagonal(dense, np.abs(dense).sum(axis=1) + 1.0)
    upper = scipy.sparse.triu(dense, format="csr")
    return (
        upper.indptr.astype(np.int32),
        upper.indices.astype(np.int32),
        upper.data.astype(np.float64),
        dense,
    )


def random_symmetric_indefinite_csr(dimension: int, density: float, seed: int):
    """A random symmetric, invertible, but not positive definite matrix, upper-triangular CSR.

    The diagonal alternates sign and dominates its row, which by
    Gershgorin's theorem keeps the matrix invertible while guaranteeing
    both positive and negative eigenvalues, so the result genuinely
    exercises the indefinite code path instead of happening to be positive
    definite.
    """
    random_state = np.random.default_rng(seed)
    factor = scipy.sparse.random(
        dimension, dimension, density=density, random_state=random_state, format="csr"
    ).toarray()
    dense = factor + factor.T
    row_sum = np.abs(dense).sum(axis=1)
    alternating_sign = np.where(np.arange(dimension) % 2 == 0, 1.0, -1.0)
    np.fill_diagonal(dense, alternating_sign * (row_sum + 1.0))
    eigenvalues = np.linalg.eigvalsh(dense)
    assert eigenvalues.min() < 0.0 < eigenvalues.max(), "constructed matrix is not indefinite"
    upper = scipy.sparse.triu(dense, format="csr")
    return (
        upper.indptr.astype(np.int32),
        upper.indices.astype(np.int32),
        upper.data.astype(np.float64),
        dense,
    )


# Builder and distinct (pattern_seed, right_hand_side_seed) pair per
# supported matrix type, so every type gets its own reproducible random
# system instead of all types sharing one draw.
_SYSTEM_BUILDERS = {
    pmj.MatrixType.REAL_STRUCTURALLY_SYMMETRIC: (random_structurally_symmetric_csr, (0, 1)),
    pmj.MatrixType.REAL_SYMMETRIC_POSITIVE_DEFINITE: (random_spd_csr, (2, 3)),
    pmj.MatrixType.REAL_SYMMETRIC_INDEFINITE: (random_symmetric_indefinite_csr, (4, 5)),
    pmj.MatrixType.REAL_NONSYMMETRIC: (random_nonsymmetric_csr, (6, 7)),
}


def build_system(matrix_type):
    """Build (indptr, indices, values, dense, right_hand_side) for a given matrix type.

    A plain function rather than a fixture, so tests that only need one
    specific matrix type (not every supported one) can call it directly
    instead of filtering down the parametrized `system` fixture below.
    """
    build, (pattern_seed, right_hand_side_seed) = _SYSTEM_BUILDERS[matrix_type]
    indptr, indices, values, dense = build(dimension=8, density=0.4, seed=pattern_seed)
    right_hand_side = np.random.default_rng(right_hand_side_seed).uniform(-1.0, 1.0, size=8)
    return indptr, indices, values, dense, right_hand_side


@pytest.fixture(params=SUPPORTED_MATRIX_TYPES, ids=lambda matrix_type: matrix_type.name)
def system(request):
    """A (matrix_type, indptr, indices, values, dense, right_hand_side) system.

    Parametrized over every matrix type pardiso_mkl_jax currently supports,
    so a test written against this fixture runs once per matrix type.
    """
    matrix_type = request.param
    indptr, indices, values, dense, right_hand_side = build_system(matrix_type)
    return matrix_type, indptr, indices, values, dense, right_hand_side


@pytest.fixture
def any_system():
    """A single representative system, for tests where the matrix type itself is not under test."""
    indptr, indices, values, dense = random_nonsymmetric_csr(dimension=8, density=0.4, seed=100)
    right_hand_side = np.random.default_rng(101).uniform(-1.0, 1.0, size=8)
    return indptr, indices, values, dense, right_hand_side
