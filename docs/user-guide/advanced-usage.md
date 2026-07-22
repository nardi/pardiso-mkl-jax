# Advanced usage

## Reusing a factorization

[`PardisoSolver`][pardiso_mkl_jax.PardisoSolver] keeps a factorization alive
so it can be reused across many solves. It splits Pardiso's three stages
into separate calls, so you control exactly what work happens on each one:

- `analyze(values)` runs the symbolic analysis (fill-reducing ordering) for
  the sparsity pattern. This is the expensive step you want to avoid
  repeating, and needs to run only once per pattern.
- `factorize(values)` runs the first numeric factorization, and requires a
  prior `analyze()`.
- `refactorize(values)` updates the numeric factorization for new values on
  the same pattern, skipping analysis. This is the cheap path when the
  matrix values change but its sparsity pattern does not.
- `solve(right_hand_side)` solves against whatever factorization is
  currently stored, and can be called many times.

`PardisoSolver` must be used as a context manager: its native memory is
released in `__exit__`, not in a destructor, since Python does not guarantee
when or whether `__del__` runs.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pardiso_mkl_jax as pardiso

indptr = jnp.array([0, 2, 3, 4], dtype=jnp.int32)
indices = jnp.array([0, 1, 1, 2], dtype=jnp.int32)
values = jnp.array([4.0, 1.0, 3.0, 2.0], dtype=jnp.float64)

with pardiso.PardisoSolver(
    indptr, indices, matrix_type=pardiso.MatrixType.REAL_NONSYMMETRIC
) as solver:
    solver.analyze(values)
    solver.factorize(values)

    first = solver.solve(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64))
    second = solver.solve(jnp.array([4.0, 5.0, 6.0], dtype=jnp.float64))

    # The matrix values changed, but the sparsity pattern did not, so this
    # skips the analysis step that factorize() needed the first time.
    solver.refactorize(values * 2.0)
    third = solver.solve(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64))
```

## Batching with vmap

`solve` carries a custom batching rule, so `jax.vmap` reuses what Pardiso
can do natively instead of looping in Python. Three cases are handled:

- **Batched right-hand sides, one matrix.** Pardiso solves multiple
  right-hand sides against one factorization in a single native call, so
  vmapping over the right-hand side fuses into that call directly.
- **Batched matrices, one right-hand side (or matching batch of them).**
  Each matrix needs its own numeric factorization, but if they share a
  sparsity pattern, analysis runs once and is reused, and only the numeric
  factorization is repeated per matrix.
- **Both batched.** The same analysis reuse applies, with each matrix's
  solve also batched over its right-hand side.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pardiso_mkl_jax as pardiso

indptr = jnp.array([0, 2, 3, 4], dtype=jnp.int32)
indices = jnp.array([0, 1, 1, 2], dtype=jnp.int32)
values = jnp.array([4.0, 1.0, 3.0, 2.0], dtype=jnp.float64)


def solve_with_fixed_matrix(right_hand_side):
    return pardiso.solve(
        indptr, indices, values, right_hand_side, matrix_type=pardiso.MatrixType.REAL_NONSYMMETRIC
    )


# One matrix, three right-hand sides: fused into a single native call.
right_hand_sides = jnp.array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=jnp.float64
)
solutions = jax.vmap(solve_with_fixed_matrix)(right_hand_sides)
```

Batching over `indptr` or `indices` is not supported: every matrix in a
batch must share the same sparsity pattern. Batch over `values` instead,
which is exactly the case Pardiso's own analysis-reuse mechanism is built
for.
