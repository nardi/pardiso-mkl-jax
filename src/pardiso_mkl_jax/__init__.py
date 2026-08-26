"""JAX-compatible interface to the oneMKL Pardiso direct sparse solver.

The two entry points most users need are `solve`, a functional one-shot
solve that works under jit and vmap, and `PardisoSolver`, a context manager
for reusing a factorization across many solves. See the user guide for both.
"""

# Must run before anything below imports the compiled _ffi extension: it
# preloads libmkl_rt so the extension's own link against it resolves. This
# has to stay the first import in this file specifically, since Python always
# runs a package's __init__.py before any of its submodules, which is the one
# ordering guarantee tools like ruff's import sorter cannot rearrange, unlike
# a plain sequence of imports inside a single module.
from pardiso_mkl_jax import _mkl_loader  # noqa: F401, I001

from pardiso_mkl_jax.iparm import PardisoDiagnostics, PardisoOption
from pardiso_mkl_jax.matrix import MatrixType
from pardiso_mkl_jax.primitive import (
    FactorizationToken,
    rebuild_count,
    reset_rebuild_count,
    solve,
)
from pardiso_mkl_jax.solver import PardisoSolver

__all__ = [
    "FactorizationToken",
    "MatrixType",
    "PardisoDiagnostics",
    "PardisoOption",
    "PardisoSolver",
    "rebuild_count",
    "reset_rebuild_count",
    "solve",
]
