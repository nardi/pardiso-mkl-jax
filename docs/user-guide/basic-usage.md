# Basic usage

[`solve`][pardiso_mkl_jax.solve] is the one-shot entry point: given a sparse
matrix in CSR format and a right-hand side, it runs analysis, factorization,
and solve in a single call, and does not keep the factorization around
afterward. It works under `jit` and `vmap` like any other JAX function. See
[Advanced usage](advanced-usage.md) if you need to reuse a factorization
across many solves.

## A first solve

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pardiso_mkl_jax as pardiso

# A = [[4, 1, 0],
#      [0, 3, 0],
#      [0, 0, 2]]
indptr = jnp.array([0, 2, 3, 4], dtype=jnp.int32)
indices = jnp.array([0, 1, 1, 2], dtype=jnp.int32)
values = jnp.array([4.0, 1.0, 3.0, 2.0], dtype=jnp.float64)
right_hand_side = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

solution = pardiso.solve(
    indptr, indices, values, right_hand_side, matrix_type=pardiso.MatrixType.REAL_NONSYMMETRIC
)
assert jnp.allclose(solution, jnp.array([1 / 12, 2 / 3, 3 / 2]))
```

The CSR arrays have fixed dtype requirements: `indptr` and `indices` must be
`int32`, and `values` must be `float64`. These are checked, not silently
converted: passing an array in another dtype raises a `TypeError` rather than
copying it, since a silent copy would break the zero-copy contract this
package is built around.

## Matrix types

[`MatrixType`][pardiso_mkl_jax.MatrixType] tells Pardiso what structure to
assume in the matrix, which determines both the factorization algorithm used
and the CSR layout expected:

- `REAL_NONSYMMETRIC` accepts the full matrix, as above, and assumes no
  structure. It is the safe default.
- `REAL_SYMMETRIC_POSITIVE_DEFINITE` and `REAL_SYMMETRIC_INDEFINITE` are
  cheaper for matrices that are actually symmetric, but expect the CSR
  arrays to hold only the upper triangle, including the diagonal. Passing
  the full matrix for a symmetric type raises a `ValueError`.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pardiso_mkl_jax as pardiso

# A = [[4, 1, 0],
#      [1, 3, 0],
#      [0, 0, 2]], symmetric positive definite.
# Only the upper triangle (including the diagonal) is stored.
indptr = jnp.array([0, 2, 3, 4], dtype=jnp.int32)
indices = jnp.array([0, 1, 1, 2], dtype=jnp.int32)
values = jnp.array([4.0, 1.0, 3.0, 2.0], dtype=jnp.float64)
right_hand_side = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

solution = pardiso.solve(
    indptr,
    indices,
    values,
    right_hand_side,
    matrix_type=pardiso.MatrixType.REAL_SYMMETRIC_POSITIVE_DEFINITE,
)
```

Complex matrix types are listed on [`MatrixType`][pardiso_mkl_jax.MatrixType]
for completeness, but are not usable yet: this version of the package works
with float64 values only.
