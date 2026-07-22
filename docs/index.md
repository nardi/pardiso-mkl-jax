---
icon: lucide/rocket
---

# pardiso-mkl-jax

`pardiso-mkl-jax` is a JAX-compatible interface to the oneMKL Pardiso direct
sparse solver. It exposes the solver as ordinary JAX code: it works under
`jit`, batches efficiently under `vmap`, and passes your matrix and vector
arrays through to the native solver without copying them.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pardiso_mkl_jax as pardiso

indptr = jnp.array([0, 2, 3, 4], dtype=jnp.int32)
indices = jnp.array([0, 1, 1, 2], dtype=jnp.int32)
values = jnp.array([4.0, 1.0, 3.0, 2.0], dtype=jnp.float64)
right_hand_side = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

solution = pardiso.solve(
    indptr, indices, values, right_hand_side, matrix_type=pardiso.MatrixType.REAL_NONSYMMETRIC
)
```

## Where to go next

- [Installation](user-guide/installation.md) covers getting the package and
  its oneMKL dependency installed.
- [Basic usage](user-guide/basic-usage.md) walks through the one-shot
  `solve` function shown above.
- [Advanced usage](user-guide/advanced-usage.md) covers reusing a
  factorization across many solves with `PardisoSolver`, and batching with
  `vmap`.
- The [API reference](reference.md) documents every public function, class,
  and matrix type.

## Why

Direct sparse solvers like Pardiso factor a matrix once and reuse that
factorization for every subsequent solve, which is far cheaper than
re-factoring from scratch. `pardiso-mkl-jax` keeps that model available from
JAX: `PardisoSolver` separates the symbolic analysis, numeric factorization,
and solve steps into distinct calls, so a program that solves the same
sparsity pattern many times, for example across timesteps or optimization
iterations, only pays the analysis and factorization cost once.
