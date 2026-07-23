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
- `refactor_and_solve(values, right_hand_side)` factorizes for new values and
  solves in one call, reusing the analysis, and requires only a prior
  `analyze()`. Unlike `refactorize()` followed by `solve()`, it keeps no
  reference to `values` on the solver, so both `values` and `right_hand_side`
  may be tracers. This is what makes `PardisoSolver` usable from inside a
  jitted function, see [Composing inside jax.jit](#composing-inside-jaxjit)
  below.

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

## Composing inside jax.jit

A `PardisoSolver`'s factorization is identified by a handle, an ordinary
`int64` JAX array value threaded through `analyze`, `factor`, `solve`, and
`release` under the hood. Once a solver has
been analyzed, `refactor_and_solve` and `solve` can be called any number of
times entirely inside a jitted function, with the analysis reused across
calls:

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

    @jax.jit
    def solve_two(values, other_values, right_hand_side):
        first = solver.refactor_and_solve(values, right_hand_side)
        second = solver.refactor_and_solve(other_values, right_hand_side)
        return first, second

    right_hand_side = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    first, second = solve_two(values, values * 2.0, right_hand_side)
```

To use `analyze` itself inside JIT, you need to use the lower-level API (see
next section). This is necessary because the memory associated with the handle
must be freed, and XLA can arbitrarily reorder operations if there is no
dependency between them. Because the handle is data, XLA orders the whole
lifecycle by data dependency, the same way it orders any other computation,
which requires manual handle management and creating an explicit dependency on
the results, for example with `jax.lax.optimization_barrier`..

## Building on the low-level primitives

`PardisoSolver` itself is built on the functions in
[`pardiso_mkl_jax.primitive`][pardiso_mkl_jax.primitive]: `analyze`,
`factor`, `solve_stateful`, `factor_and_solve_stateful`, and `release`. Each
one threads a handle value in and, except for `solve_stateful`, back out
again. Library authors who want to manage a factorization's lifetime
explicitly, rather than through `PardisoSolver`'s own context manager, for
example tying it to a scope object or to another jit-traced dependency, can
call these functions directly:

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pardiso_mkl_jax as pmj
from pardiso_mkl_jax import primitive

indptr = jnp.array([0, 2, 3, 4], dtype=jnp.int32)
indices = jnp.array([0, 1, 1, 2], dtype=jnp.int32)
values = jnp.array([4.0, 1.0, 3.0, 2.0], dtype=jnp.float64)
right_hand_side = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
matrix_type = pmj.MatrixType.REAL_NONSYMMETRIC

handle, _final_iparm = primitive.analyze(indptr, indices, values, matrix_type=matrix_type)
solution, final_iparm = primitive.factor_and_solve_stateful(
    handle, indptr, indices, values, right_hand_side[None, :], matrix_type=matrix_type
)
primitive.release(handle)
```

Every primitive returns the raw `iparm` array Pardiso left behind alongside
its usual result, and takes an `options` overlay; see
[Overriding solver settings](#overriding-solver-settings) below. Decode the
raw array with
[`PardisoDiagnostics.from_iparm`][pardiso_mkl_jax.PardisoDiagnostics].

As with `PardisoSolver`, releasing a handle while something else might still
use it is a bug: since `release` and any other consumer of `handle` share no
ordering beyond their common input, a caller that wants a release ordered
after a particular use should force that dependency explicitly, for example
with `jax.lax.optimization_barrier`.

## Solving the transpose

Both [`solve`][pardiso_mkl_jax.solve] and
[`PardisoSolver.solve`][pardiso_mkl_jax.PardisoSolver.solve] accept a
`transpose` argument. Setting it solves `A^T x = right_hand_side` instead of
`A x = right_hand_side`, reusing exactly the same factorization: an LU (or
LDL^T) factorization of `A` supports triangular solves in either direction,
so no separate factorization of `A^T` is needed, and no explicit transpose of
the CSR arrays either.

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

    right_hand_side = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    x = solver.solve(right_hand_side)
    x_transpose = solver.solve(right_hand_side, transpose=True)
```

Alternating `transpose` between calls on the same `PardisoSolver` is safe:
each `solve()` call sets it fresh, so it never leaks into a later call that
does not ask for it. For matrix types whose values are mathematically
symmetric or Hermitian (`REAL_SYMMETRIC_POSITIVE_DEFINITE`,
`REAL_SYMMETRIC_INDEFINITE`), `A^T` equals `A`, so `transpose=True` gives the
same result as `transpose=False`.

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

Pardiso is controlled through a 64-entry `iparm` array. pardiso_mkl_jax fills
it with a fixed set of defaults, chosen for correctness and predictable
performance across the matrix types this package supports. This section
documents and motivates each of them. Every one of them can be overridden,
see "Overriding solver settings" below.

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

## Overriding solver settings

[`solve`][pardiso_mkl_jax.solve], and every [`PardisoSolver`][pardiso_mkl_jax.PardisoSolver]
method (`analyze`, `factorize`, `refactorize`, `solve`,
`refactor_and_solve`), accept an `options` argument: a `{option: value}` overlay applied on top of the defaults
documented above. Keys are members of
[`PardisoOption`][pardiso_mkl_jax.PardisoOption], or the raw `iparm` index as
a plain int, for anything not given a named member.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pardiso_mkl_jax as pmj
from pardiso_mkl_jax import PardisoOption

indptr = jnp.array([0, 2, 3, 4], dtype=jnp.int32)
indices = jnp.array([0, 1, 1, 2], dtype=jnp.int32)
values = jnp.array([4.0, 1.0, 3.0, 2.0], dtype=jnp.float64)
right_hand_side = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

# Minimum degree ordering instead of the default METIS nested dissection.
x = pmj.solve(
    indptr,
    indices,
    values,
    right_hand_side,
    matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
    options={PardisoOption.FILL_IN_REDUCING_ORDERING: 0},
)

with pmj.PardisoSolver(
    indptr, indices, matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
) as solver:
    solver.analyze(values, options={PardisoOption.FILL_IN_REDUCING_ORDERING: 0})
    solver.factorize(values)
    x = solver.solve(right_hand_side)
```

An overlay entry does not persist beyond the call it was passed to.
`PardisoSolver`'s defaults are recomputed fresh on every `analyze`,
`factorize`, or `refactorize` call, so an overlay entry passed to `analyze`
does not automatically apply to a later `factorize` or `refactorize` on the
same solver: pass it again wherever it needs to apply.

Most `iparm` entries can be overridden freely. A few are handled specially:

- `USE_DEFAULT_VALUES` (`iparm[0]`) and `WEIGHTED_MATCHING` (`iparm[12]`) can
  be set to anything, with no runtime warning. Setting `USE_DEFAULT_VALUES`
  away from 1 is exactly what the crash workaround above depends on staying
  at 1, and `WEIGHTED_MATCHING` is the specific setting that workaround keeps
  disabled, so touching either one is knowingly reaching for that risk.
- `INDEXING_STYLE` (`iparm[34]`) can be set, but raises a `UserWarning`:
  every CSR array this package builds is zero-based, so anything else risks
  Pardiso silently misreading `indptr` and `indices`.
- `USER_PERMUTATION`, `PARTIAL_SOLVE_CONTROL`, and `SCHUR_COMPLEMENT_CONTROL`
  cannot be enabled (a `ValueError` rejects any non-zero value): each needs a
  native argument this package always passes as null, so enabling one would
  read or write through a null pointer.
- `TRANSPOSE_SOLVE` cannot be set through `options` at all: use the
  `transpose` argument instead (see "Solving the transpose" above), which
  already covers everything this index can express for the real-valued
  matrices this package supports.

Keep the overlay stable across many calls in performance-sensitive code.
Internally, `solve` builds one compiled batching rule per distinct
`(matrix_type, transpose, options)` combination and keeps it cached for the
life of the process, the same way it already does for `matrix_type` and
`transpose` alone. A fixed overlay is free after the first call; an overlay
that changes on every call (for example, sweeping a setting in a parameter
search) grows that cache by one compiled rule per distinct value, with no
upper bound.

## Diagnostics

Pardiso writes several outputs back into `iparm` after a call: memory used,
iterative refinement steps actually run, perturbed pivot count, eigenvalue
counts for symmetric indefinite matrices, and more. `solve` and every
`PardisoSolver` method surface these through
[`PardisoDiagnostics`][pardiso_mkl_jax.PardisoDiagnostics].

Pass `return_diagnostics=True` to `solve` to get `(solution, diagnostics)`
back instead of just the solution. `PardisoSolver` records diagnostics from
`analyze`, `factorize`, `refactorize`, and `solve` automatically, readable
from `last_diagnostics`:

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pardiso_mkl_jax as pmj

# An indefinite matrix, so it has both positive and negative eigenvalues.
indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
indices = jnp.array([0, 1, 1], dtype=jnp.int32)
values = jnp.array([1.0, 1.0, -1.0], dtype=jnp.float64)

with pmj.PardisoSolver(
    indptr, indices, matrix_type=pmj.MatrixType.REAL_SYMMETRIC_INDEFINITE
) as solver:
    solver.analyze(values)
    solver.factorize(values)
    diagnostics = solver.last_diagnostics

assert diagnostics is not None
positive_eigenvalues = diagnostics.positive_eigenvalues
negative_eigenvalues = diagnostics.negative_eigenvalues
```

Fields not given a name are still reachable through `diagnostics.raw`, the
full 64-entry `iparm` array Pardiso left behind.

`refactor_and_solve` is the one method that does not update
`last_diagnostics`. It is meant to be callable from inside a jitted function,
where storing anything derived from its results would leave a tracer on the
solver, so it takes the same `return_diagnostics=True` flag as `solve` and
returns `(solution, diagnostics)` instead.

Diagnostics work under `jit`, including when `solve` itself is wrapped
directly, and under `vmap`. Under the default `vmap` batching (batching over
`values`, or over both `values` and the right-hand side), each batch element
gets its own diagnostics, since each one was genuinely factorized separately.
Batching only over the right-hand side reuses a single native call for the
whole batch, so Pardiso only reports diagnostics once: the output array still
gains a batch dimension, matching `jax.vmap`'s usual behavior, but every
entry along it is that same value, broadcast rather than recomputed. Pass
`out_axes` on your own `jax.vmap` call if the compact, un-broadcast form is
what you want instead:

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
        indptr,
        indices,
        values,
        right_hand_side,
        matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
        return_diagnostics=True,
    )


right_hand_sides = jnp.array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=jnp.float64
)
solutions, diagnostics = jax.vmap(solve_with_fixed_matrix, out_axes=(0, None))(
    right_hand_sides
)
```

Diagnostics are only ever available on success. A failed Pardiso call raises
a Python exception instead of returning anything, diagnostics included: a
function that raises cannot also return a value.
