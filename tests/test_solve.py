"""Correctness tests for the functional pardiso_mkl_jax.solve entry point."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse
from conftest import UNSUPPORTED_MATRIX_TYPES, UPPER_TRIANGULAR_MATRIX_TYPES, build_system

import pardiso_mkl_jax as pmj


def test_solve_matches_dense_reference(system):
    matrix_type, indptr, indices, values, dense, right_hand_side = system
    solution = pmj.solve(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        jnp.asarray(values),
        jnp.asarray(right_hand_side),
        matrix_type=matrix_type,
    )
    expected = np.linalg.solve(dense, right_hand_side)
    np.testing.assert_allclose(np.asarray(solution), expected, rtol=1e-8, atol=1e-10)


def test_solve_rejects_wrong_index_dtype(system):
    matrix_type, indptr, indices, values, _dense, right_hand_side = system
    with pytest.raises(TypeError, match="int32"):
        pmj.solve(
            jnp.asarray(indptr.astype(np.int64)),
            jnp.asarray(indices),
            jnp.asarray(values),
            jnp.asarray(right_hand_side),
            matrix_type=matrix_type,
        )


def test_solve_rejects_wrong_values_dtype(system):
    matrix_type, indptr, indices, values, _dense, right_hand_side = system
    with pytest.raises(TypeError, match="float64"):
        pmj.solve(
            jnp.asarray(indptr),
            jnp.asarray(indices),
            jnp.asarray(values.astype(np.float32)),
            jnp.asarray(right_hand_side),
            matrix_type=matrix_type,
        )


@pytest.mark.parametrize("matrix_type", UNSUPPORTED_MATRIX_TYPES, ids=lambda mt: mt.name)
def test_solve_rejects_unsupported_complex_matrix_types(any_system, matrix_type):
    indptr, indices, values, _dense, right_hand_side = any_system
    with pytest.raises(NotImplementedError, match="complex"):
        pmj.solve(
            jnp.asarray(indptr),
            jnp.asarray(indices),
            jnp.asarray(values),
            jnp.asarray(right_hand_side),
            matrix_type=matrix_type,
        )


@pytest.mark.parametrize("matrix_type", UPPER_TRIANGULAR_MATRIX_TYPES, ids=lambda mt: mt.name)
def test_solve_rejects_full_matrix_for_upper_triangular_types(matrix_type):
    _indptr, _indices, _values, dense, right_hand_side = build_system(matrix_type)
    full = scipy.sparse.csr_matrix(dense)
    with pytest.raises(ValueError, match="upper triangle"):
        pmj.solve(
            jnp.asarray(full.indptr.astype(np.int32)),
            jnp.asarray(full.indices.astype(np.int32)),
            jnp.asarray(full.data.astype(np.float64)),
            jnp.asarray(right_hand_side),
            matrix_type=matrix_type,
        )


@pytest.mark.parametrize("matrix_type", UPPER_TRIANGULAR_MATRIX_TYPES, ids=lambda mt: mt.name)
def test_solve_rejects_full_matrix_for_upper_triangular_types_under_jit(matrix_type):
    # indptr and indices are traced under jit, so this exercises the
    # equinox.error_if runtime check in check_upper_triangular rather than
    # the plain, concrete-array ValueError path above.
    _indptr, _indices, _values, dense, right_hand_side = build_system(matrix_type)
    full = scipy.sparse.csr_matrix(dense)

    jit_solve = jax.jit(pmj.solve, static_argnames=("matrix_type",))
    with pytest.raises(RuntimeError, match="upper triangle"):
        jit_solve(
            jnp.asarray(full.indptr.astype(np.int32)),
            jnp.asarray(full.indices.astype(np.int32)),
            jnp.asarray(full.data.astype(np.float64)),
            jnp.asarray(right_hand_side),
            matrix_type=matrix_type,
        )
