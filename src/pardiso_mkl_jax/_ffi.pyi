# Type stub for the compiled _ffi extension (built from _ffi.pyx).
#
# Importing this module registers the pardiso_mkl_jax_* XLA FFI targets as a
# side effect; the only names meant for use from Python are the analysis-count
# test hooks below.

from typing import Any

# handle accepts a plain int or a JAX/NumPy scalar array, as returned by
# analyze(), so it is typed loosely here rather than as plain int.
def analysis_count(handle: int | Any) -> int: ...
def reset_analysis_count(handle: int | Any) -> None: ...
