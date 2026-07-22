"""Shared pytest fixtures and helpers for the pardiso_mkl_jax test suite."""

from __future__ import annotations

import jax
import numpy as np
import pytest
import scipy.sparse

jax.config.update("jax_enable_x64", True)


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


@pytest.fixture
def nonsymmetric_system():
    """A small nonsymmetric CSR system: (indptr, indices, values, dense, right_hand_side)."""
    indptr, indices, values, dense = random_nonsymmetric_csr(dimension=8, density=0.4, seed=0)
    right_hand_side = np.random.default_rng(1).uniform(-1.0, 1.0, size=8)
    return indptr, indices, values, dense, right_hand_side


@pytest.fixture
def spd_system():
    """A small SPD upper-triangular system: (indptr, indices, values, dense, right_hand_side)."""
    indptr, indices, values, dense = random_spd_csr(dimension=8, density=0.4, seed=2)
    right_hand_side = np.random.default_rng(3).uniform(-1.0, 1.0, size=8)
    return indptr, indices, values, dense, right_hand_side
