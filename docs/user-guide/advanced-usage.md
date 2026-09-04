# Advanced usage

## Reusing a factorization

[`PardisoSolver`][pardiso_mkl_jax.PardisoSolver] keeps a factorization alive
so it can be reused across many solves. It splits Pardiso's three stages
into separate calls, so you control exactly what work happens on each one:

- `analyze(values)` runs the symbolic analysis (fill-reducing ordering) for
  the sparsity pattern. This is the expensive step you want to avoid
  repeating, and needs to run only once per pattern. Calling it again
  re-analyzes in place, see [Re-analyzing in place](#re-analyzing-in-place)
  below.
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

`PardisoSolver` can be used as a context manager, which releases its native
memory on exit, but this is optional. Every handle is a key into a bounded
cache (see [Memory and the handle cache](#memory-and-the-handle-cache) below),
so a solver you never close leaks at most one cache slot, not unbounded memory,
and reusing it after it was released still works. Use the `with` block, or call
`close()`, to free that slot promptly.

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

## Re-analyzing in place

Calling `analyze()` a second time on the same solver redoes the symbolic
phase on the handle the solver already holds. The existing factorization is
freed first, and no second handle is allocated, so there is nothing extra to
release. This is the recovery path for a factorization that came back
unusable: re-analyze with different settings and try again, without having to
build a second solver and free the first one conditionally.

Because the re-analysis discards the factorization, `factorize()` has to run
again before the next `solve()`. Calling `solve()` or `refactorize()` in
between raises, rather than solving against memory that is no longer there.

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

with pmj.PardisoSolver(
    indptr, indices, matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
) as solver:
    solver.analyze(values)
    diagnostics = solver.factorize(values, return_diagnostics=True)

    if diagnostics.perturbed_pivot_count > 0:
        # Pardiso could not pivot cleanly. Re-analyze on the same handle with
        # matching disabled, then factorize again.
        solver.analyze(values, options={PardisoOption.WEIGHTED_MATCHING: 0})
        solver.factorize(values, options={PardisoOption.WEIGHTED_MATCHING: 0})

    solution = solver.solve(right_hand_side)
```

`analyze()` must be called outside `jax.jit`, both the first time and for a
re-analysis: it stores the native handle on the solver, and under jit that
handle is a tracer, which would escape its trace. To re-analyze inside a
traced function, use [`primitive.reanalyze`][pardiso_mkl_jax.primitive] and
thread the handle yourself, as in
[Building on the low-level primitives](#building-on-the-low-level-primitives)
below. That works under jit: the handle goes in and comes back out, so XLA
orders the re-analysis against the calls around it by data dependency.

## Composing inside jax.jit

A `PardisoSolver`'s factorization is identified by a token, a small bundle
carrying an `int64` cache id and a version stamp, threaded through `analyze`,
`factor`, `solve`, and `release` under the hood. Once a solver has
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

To use `analyze` itself inside JIT, use the lower-level API (see next section)
and thread the token through `analyze`, `factor`, `solve`, and `release`
yourself. Because the token's id is data, XLA orders analyze, factor, and solve
by data dependency, the same way it orders any other computation. Ordering a
`release` after the solves that used it takes one extra step, covered in
[When is it safe to release explicitly?](#when-is-it-safe-to-release-explicitly)
below.

## Building on the low-level primitives

`PardisoSolver` itself is built on the functions in
[`pardiso_mkl_jax.primitive`][pardiso_mkl_jax.primitive]: `analyze`,
`reanalyze`, `factor`, `solve_stateful`, `factor_and_solve_stateful`, and
`release`. Each one takes a
[`FactorizationToken`][pardiso_mkl_jax.FactorizationToken]. `analyze`,
`reanalyze`, and `factor` return one. `solve_stateful` and
`factor_and_solve_stateful` return one too when passed `return_token=True`,
which matters for ordering reused solves and is covered in
[Reusing one handle across ordered solves](#reusing-one-handle-across-ordered-solves)
below. Library authors
who want to manage a factorization's lifetime
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

token, _final_iparm = primitive.analyze(indptr, indices, values, matrix_type=matrix_type)
solution, final_iparm = primitive.factor_and_solve_stateful(
    token, indptr, indices, values, right_hand_side[None, :], matrix_type=matrix_type
)
primitive.release(token)
```

Every primitive returns the raw `iparm` array Pardiso left behind alongside
its usual result, and takes an `options` overlay; see
[Overriding solver settings](#overriding-solver-settings) below. Decode the
raw array with
[`PardisoDiagnostics.from_iparm`][pardiso_mkl_jax.PardisoDiagnostics].

Releasing a token that something else still uses is safe for correctness.
`release` frees the cache slot, but the token stays valid. The next time it is used the factorization is rebuilt from the matrix it carries. Whether an
early release actually frees anything is a separate question, covered in
[When is it safe to release explicitly?](#when-is-it-safe-to-release-explicitly)
below.

## Reusing one handle across ordered solves

A handle holds one factorization at a time. `solve_stateful` reads it and
`factor` replaces it, and by default a solve returns only its solution, so a
later `factor` shares no value with the solve before it. Under `jit`, XLA is
free to run that `factor` before the solve that still needs the earlier
factorization, and the solve then returns the wrong answer.

Passing `return_token=True` to `solve_stateful` (and to
`factor_and_solve_stateful`) fixes this. The solve then returns a token whose id
and version come out of the native call, so threading that token into the next
`factor` gives the `factor` a data dependency on the solve. Every access to the
handle sits on one chain, and the compiler cannot reorder them:

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pardiso_mkl_jax as pmj
from pardiso_mkl_jax import primitive

indptr = jnp.array([0, 2, 3, 4], dtype=jnp.int32)
indices = jnp.array([0, 1, 1, 2], dtype=jnp.int32)
values = jnp.array([4.0, 1.0, 3.0, 2.0], dtype=jnp.float64)
other_values = values * 2.0
right_hand_side = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
matrix_type = pmj.MatrixType.REAL_NONSYMMETRIC


@jax.jit
def solve_both(values, other_values, right_hand_side):
    token, _ = primitive.analyze(indptr, indices, values, matrix_type=matrix_type)
    token, _ = primitive.factor(token, indptr, indices, values, matrix_type=matrix_type)
    first, token, _ = primitive.solve_stateful(
        token, indptr, indices, values, right_hand_side[None, :],
        matrix_type=matrix_type, return_token=True,
    )
    # This factor consumes the token the solve returned, so it waits for the
    # solve rather than overwriting the factorization first.
    token, _ = primitive.factor(token, indptr, indices, other_values, matrix_type=matrix_type)
    second, token, _ = primitive.solve_stateful(
        token, indptr, indices, other_values, right_hand_side[None, :],
        matrix_type=matrix_type, return_token=True,
    )
    primitive.release(token.track(first, second))
    return first[0], second[0]


first, second = solve_both(values, other_values, right_hand_side)
```

The token's version stamp is a second, independent guard. Every `factor` and
`reanalyze` stamps a fresh version on the handle, and a solve carries the
version it expects. If a solve reaches a handle whose version has already moved
past the token, the factorization it named was replaced, so the native call
reports a mismatch rather than solving the wrong matrix. This catches a token
threaded into two separate writes, which `jit` cannot rule out on its own.
`factor_and_solve_stateful` passes the version through unchanged, since it both
writes and reads the factorization in one call, so the token stays valid for a
later solve.

## Memory and the handle cache

A token's cache id is not a raw pointer. It is a key into a process-wide cache of
factorizations, and this is what makes the token API memory-safe:

- **Forgetting to release leaks a bounded amount.** The cache holds a fixed
  number of factorizations, eight by default. Once it is full the
  least-recently-used one is evicted, so tokens you never release cost at most
  a full cache rather than growing without end. Set the size with the
  `PARDISO_MKL_JAX_FACTOR_CACHE` environment variable.
- **Using an evicted or released token is safe.** Every stateful call carries
  the matrix it needs, so a call that lands on a token no longer in the cache
  rebuilds its factorization on the spot and continues. The answer is the same,
  it just costs the rebuild.

Rebuilds are correct but not free, so a program that keeps more factorizations
live than the cache holds pays to rebuild them over and over. Two tools help
find that:

- [`pardiso_mkl_jax.rebuild_count`][pardiso_mkl_jax.rebuild_count] returns how
  many rebuilds have happened. A count that climbs during steady-state solving
  means the cache is too small for the working set. Reset it with
  `reset_rebuild_count`.
- Setting `PARDISO_MKL_JAX_STRICT_CACHE` turns any rebuild into an error that
  names the token, so a lost factorization fails loudly instead of quietly
  slowing things down. Leave it off in production and switch it on while
  debugging performance.

### When is it safe to release explicitly?

There are two aspects to distunguish here:

**Correctness: always safe.** A released token rebuilds itself on next use, so
`release` (or `PardisoSolver.close`) can be called at any point without risk of a
crash or a wrong answer.

**Effectiveness: only when the release runs after the token's last use.** A
release reclaims memory without forcing a wasted rebuild only if nothing uses the
token afterward. Releasing eagerly in Python, after the step that used the
token has run, gives you that ordering for free:

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

token, _ = primitive.analyze(indptr, indices, values, matrix_type=matrix_type)
token, _ = primitive.factor(token, indptr, indices, values, matrix_type=matrix_type)

solution, _ = primitive.solve_stateful(
    token, indptr, indices, values, right_hand_side[None, :], matrix_type=matrix_type
)
# Runs after the solve above, so it actually frees rather than costing a rebuild.
primitive.release(token)
```

Inside a single `jax.jit` trace there is no ordering between a release and a
solve that both take the same token, because a bare `release` does not consume
the solve's output. XLA may run the release first, in which case the solve
rebuilds the factorization and the release reclaims nothing. To order the
release after a solve, give it that solve's solution to depend on, either with
`release(token, dependency=solution)` or by calling `token.track(solution)`
first. Both make the release wait for the solve, so it actually frees:

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

token, _ = primitive.analyze(indptr, indices, values, matrix_type=matrix_type)
token, _ = primitive.factor(token, indptr, indices, values, matrix_type=matrix_type)


@jax.jit
def solve_and_release(token, values, right_hand_side):
    solution, _ = primitive.solve_stateful(
        token, indptr, indices, values, right_hand_side[None, :], matrix_type=matrix_type
    )
    primitive.release(token, dependency=solution)
    return solution


solution = solve_and_release(token, values, right_hand_side)
```

`token.track(solution)` does the same by folding the solve into the token, which
is handy in a loop: track each step's solution, then release the token once after
the loop to free it after every solve.

Explicit releases are rarely worth it. Letting the cache evict costs the same in
the worst case, with none of the bookkeeping.

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
given rather than left for Pardiso to fill in on its own. That is what makes
`iparm[34]` (zero-based indexing) and `iparm[11]` (transpose solves) reliable
settings rather than ones Pardiso might overwrite. It also comes with a sharp
edge worth stating plainly, because it caused a real bug: with `iparm[0] = 1`,
every entry that is *not* assigned stays at 0, and 0 is not the same as "the
default Pardiso would have picked". Any entry whose default is non-zero has to
be restated explicitly, per matrix type, or it is silently switched off. The
list below therefore reproduces `pardisoinit`'s defaults for every supported
matrix type, with the two exceptions called out as such.

- **`iparm[1] = 2`**, serial nested dissection (METIS-based) fill-reducing
  ordering. This is the one entry chosen for its own sake rather than
  copied from Pardiso, whose default is parallel nested dissection (3).
  Serial ordering makes the factorization reproducible run to run
  regardless of thread count.
- **`iparm[7] = 2`**, maximum steps of iterative refinement, matching
  Pardiso's own default for every matrix type. Refinement is the backstop
  for a factorization that had to perturb a pivot: without it, a perturbed
  solve returns whatever the perturbed factors give and never corrects it.
- **`iparm[9]`**, the pivot perturbation exponent, is 13 for the
  nonsymmetric and structurally symmetric matrix types and 8 for the
  symmetric and Hermitian ones, matching Pardiso's own default per matrix
  type.
- **`iparm[10]`**, maximum weighted matching's companion scaling step, is
  enabled (1) for the nonsymmetric matrix types and disabled (0)
  otherwise, again matching Pardiso's own default.
- **`iparm[12]`**, weighted matching, is enabled (1) for the nonsymmetric
  matrix types and disabled (0) otherwise, matching Pardiso's own default.
  Matching permutes large entries onto the diagonal before factoring, and
  it is not optional in practice for any matrix with zeros on its diagonal.
  Saddle-point systems, KKT systems, and constraint blocks all qualify.
  Without it, Pardiso finds tiny pivots there, perturbs them, and returns a
  solution with a large residual while reporting success, so nothing but
  the numbers themselves reveals the problem.
- **`iparm[20]`**, Bunch-Kaufman 1x1 and 2x2 pivoting, is enabled (1) for
  the symmetric indefinite matrix types and disabled (0) otherwise,
  matching Pardiso's own default. Symmetric indefinite matrices need it for
  the same reason nonsymmetric ones need matching: a zero diagonal entry
  with no 2x2 pivot to fall back on gets perturbed instead.
- **`iparm[17]` and `iparm[18]`** are left at 0 rather than Pardiso's
  default of -1. This is the second deliberate departure: both only request
  statistics (the number of non-zeros in the factors, and an MFLOP count)
  that this package does not surface, and computing them is not free.
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

An overlay passed to a single method does not persist beyond that call.
Pardiso's parameters are recomputed fresh on every native call, so an entry
passed to `analyze` does not carry over to a later `factorize` or
`refactorize` on the same solver.

To set an overlay for a solver's whole lifetime, pass it to the constructor
instead. It is applied to every call, and a per-call `options` argument
layers on top of it, winning on any entry both set:

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

# Matching and scaling both look at the numeric values during analysis.
# Turning them off keeps the analysis independent of the values, so one
# handle stays valid across matrices that are scaled differently.
with pmj.PardisoSolver(
    indptr,
    indices,
    matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
    options={PardisoOption.WEIGHTED_MATCHING: 0, PardisoOption.SCALING: 0},
) as solver:
    solver.analyze(values)
    solver.factorize(values * 1e6)
    solution = solver.solve(right_hand_side)
```

`SCALING` and `WEIGHTED_MATCHING` are the two entries where getting this
wrong is a correctness problem rather than an inconvenience. Pardiso computes
both during analysis and expects them unchanged at every later phase, and
nothing in the native layer enforces that: a disagreement is silently
accepted and gives a wrong answer instead of an error. `PardisoSolver` checks
for it and raises a `ValueError` naming the option, so setting either of
these on `analyze` alone is rejected rather than quietly mishandled. Set them
on the constructor so they apply everywhere, or pass the same value to every
call.

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

Every method takes a `return_diagnostics=True` flag. `solve` and
`refactor_and_solve` return `(solution, diagnostics)` with it set, and
`analyze`, `factorize`, and `refactorize` return the diagnostics directly
where they otherwise return nothing. `PardisoSolver` also records diagnostics
from `analyze`, `factorize`, `refactorize`, and `solve` automatically,
readable from `last_diagnostics`:

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

### Perturbed pivots

`perturbed_pivot_count` is Pardiso's own report that it could not pivot
cleanly during a numeric factorization: it replaced pivots that were too
small with a perturbation and carried on. A non-zero count means the
factorization may not be usable, and Pardiso signals this here rather than
through an error code, so a caller that does not check it gets a solution
with a large residual and no indication anything went wrong.

The count is only meaningful after a call that actually factorized:
`factorize`, `refactorize`, `refactor_and_solve`, or the functional `solve`.
After `analyze` or a `PardisoSolver.solve` it reports whatever the last
factorization left behind. See
[Re-analyzing in place](#re-analyzing-in-place) for the recovery path.

### Diagnostics under jit and vmap

`last_diagnostics` is `None` after a call that ran under `jax.jit`. The
diagnostics only exist as tracers there, and storing one on the solver would
leak it out of its trace, so reading it back afterwards would raise rather
than return anything usable. Use `return_diagnostics=True` for traced calls,
which returns the diagnostics as ordinary jit outputs.

`refactor_and_solve` never updates `last_diagnostics`, traced or not: keeping
nothing at all on the solver is the point of that method. It takes the same
`return_diagnostics=True` flag.

Diagnostics work under `vmap` as well. Under the default `vmap` batching (batching over
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
