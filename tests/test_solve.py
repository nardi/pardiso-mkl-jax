"""Correctness tests for the functional pardiso_mkl_jax.solve entry point."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import pardiso_mkl_jax as pardiso


def test_solve_matches_dense_reference_nonsymmetric(nonsymmetric_system):
    indptr, indices, values, dense, right_hand_side = nonsymmetric_system
    solution = pardiso.solve(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        jnp.asarray(values),
        jnp.asarray(right_hand_side),
        matrix_type=pardiso.MatrixType.REAL_NONSYMMETRIC,
    )
    expected = np.linalg.solve(dense, right_hand_side)
    np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)


def test_solve_matches_dense_reference_spd(spd_system):
    indptr, indices, values, dense, right_hand_side = spd_system
    solution = pardiso.solve(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        jnp.asarray(values),
        jnp.asarray(right_hand_side),
        matrix_type=pardiso.MatrixType.REAL_SYMMETRIC_POSITIVE_DEFINITE,
    )
    expected = np.linalg.solve(dense, right_hand_side)
    np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)


def test_solve_rejects_wrong_index_dtype(nonsymmetric_system):
    indptr, indices, values, _dense, right_hand_side = nonsymmetric_system
    with pytest.raises(TypeError, match="int32"):
        pardiso.solve(
            jnp.asarray(indptr.astype(np.int64)),
            jnp.asarray(indices),
            jnp.asarray(values),
            jnp.asarray(right_hand_side),
            matrix_type=pardiso.MatrixType.REAL_NONSYMMETRIC,
        )


def test_solve_rejects_wrong_values_dtype(nonsymmetric_system):
    indptr, indices, values, _dense, right_hand_side = nonsymmetric_system
    with pytest.raises(TypeError, match="float64"):
        pardiso.solve(
            jnp.asarray(indptr),
            jnp.asarray(indices),
            jnp.asarray(values.astype(np.float32)),
            jnp.asarray(right_hand_side),
            matrix_type=pardiso.MatrixType.REAL_NONSYMMETRIC,
        )


def test_solve_rejects_unsupported_complex_matrix_type(nonsymmetric_system):
    indptr, indices, values, _dense, right_hand_side = nonsymmetric_system
    with pytest.raises(NotImplementedError, match="complex"):
        pardiso.solve(
            jnp.asarray(indptr),
            jnp.asarray(indices),
            jnp.asarray(values),
            jnp.asarray(right_hand_side),
            matrix_type=pardiso.MatrixType.COMPLEX_NONSYMMETRIC,
        )


def test_solve_rejects_full_matrix_for_symmetric_type(spd_system):
    # Rebuild the full (not upper-triangular) matrix for the same SPD system.
    _indptr, _indices, _values, dense, right_hand_side = spd_system
    import scipy.sparse

    full = scipy.sparse.csr_matrix(dense)
    with pytest.raises(ValueError, match="upper triangle"):
        pardiso.solve(
            jnp.asarray(full.indptr.astype(np.int32)),
            jnp.asarray(full.indices.astype(np.int32)),
            jnp.asarray(full.data.astype(np.float64)),
            jnp.asarray(right_hand_side),
            matrix_type=pardiso.MatrixType.REAL_SYMMETRIC_POSITIVE_DEFINITE,
        )
