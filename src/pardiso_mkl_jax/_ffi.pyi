# Type stub for the compiled _ffi extension (built from _ffi.pyx).
#
# Importing this module registers the pardiso_mkl_jax_* XLA FFI targets as a
# side effect; the only names meant for use from Python are the analysis-count
# test hooks below.

def analysis_count(solver_id: int) -> int: ...
def reset_analysis_count(solver_id: int) -> None: ...
