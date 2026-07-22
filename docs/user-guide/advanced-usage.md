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
import pardiso_mkl_jax as pmj

indptr = jnp.array([0, 2, 3, 4], dtype=jnp.int32)
indices = jnp.array([0, 1, 1, 2], dtype=jnp.int32)
values = jnp.array([4.0, 1.0, 3.0, 2.0], dtype=jnp.float64)

with pmj.PardisoSolver(
    indptr, indices, matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
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
import pardiso_mkl_jax as pmj

indptr = jnp.array([0, 2, 3, 4], dtype=jnp.int32)
indices = jnp.array([0, 1, 1, 2], dtype=jnp.int32)
values = jnp.array([4.0, 1.0, 3.0, 2.0], dtype=jnp.float64)


def solve_with_fixed_matrix(right_hand_side):
    return pmj.solve(
        indptr, indices, values, right_hand_side, matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
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

## Solver settings

Pardiso is controlled through a 64-entry `iparm` array. pardiso_mkl_jax does
not expose this array to callers in this version: it fills it with a fixed
set of defaults, chosen for correctness and predictable performance across
the matrix types this package supports. This section documents and
motivates each of them.

`iparm[0]` is set to 1, meaning every other entry below is used exactly as
given rather than left for Pardiso to fill in on its own. This is forced
rather than optional, for a reason unrelated to the other settings: the MKL
build this package links against has a bug where its own default for
nonsymmetric matrices enables weighted matching (`iparm[12] = 1`), and that
heuristic segfaults inside `mkl_pds_lp64_kuhn_munkres`, reproduced with a
minimal standalone Pardiso call outside this package entirely, on ordinary,
well-conditioned matrices, not just degenerate ones. Overriding `iparm[12]`
back to 0 alone does not avoid the crash, since MKL resets it internally
before the matching step runs whenever `iparm[0]` is left at 0. Taking over
`iparm[0] = 1` is the only way to keep weighted matching disabled, which is
why every other entry has to be specified explicitly too, even the ones
that just restate what Pardiso's own default would have been.

- **`iparm[1] = 2`**, nested dissection (METIS-based) fill-reducing
  ordering. This matches Pardiso's own default: it is restated here only
  because `iparm[0] = 1` requires every entry to be given explicitly.
- **`iparm[9]`**, the pivot perturbation exponent, is 13 for the
  nonsymmetric and structurally symmetric matrix types and 8 for the
  symmetric and Hermitian ones. This also matches Pardiso's own default per
  matrix type, restated for the same reason as `iparm[1]`.
- **`iparm[10]`**, maximum weighted matching's companion scaling step, is
  enabled (1) only for `REAL_NONSYMMETRIC`, and disabled (0) otherwise,
  again matching Pardiso's own default. Scaling itself is not affected by
  the weighted matching bug described above, so there was no reason to
  change it.
- **`iparm[12] = 0`**, weighted matching, is disabled for every matrix
  type. This is the one setting that is not just Pardiso's own default
  restated: it is a deliberate departure, forced by the crash explained
  above. The practical effect is a fixed row/column permutation and
  scaling strategy for nonsymmetric matrices, rather than one adapted to
  the specific values each time, which can mean somewhat more conservative
  pivoting on badly scaled matrices than a working weighted matching would
  give.
- **`iparm[34] = 1`**, zero-based indexing. Pardiso defaults to one-based
  (Fortran-style) indexing. pardiso_mkl_jax works with zero-based CSR
  arrays throughout, matching NumPy, SciPy, and JAX, so leaving Pardiso's
  default here would mean re-indexing every array before every call,
  breaking the zero-copy interface this package promises.
