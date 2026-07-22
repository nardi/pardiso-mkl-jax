# distutils: language = c++
# cython: language_level=3
"""Registers the compiled Pardiso FFI handlers as JAX custom call targets.

The handler addresses come from _pardiso_ffi.cc, which is compiled directly
into this extension. Each address is wrapped in a PyCapsule, the format JAX's
FFI registration expects, and handed to jax.ffi.register_ffi_target. Nothing
else in this module is meant for public use, except the analysis-count hooks
used by the test suite.
"""

from cpython.pycapsule cimport PyCapsule_New


cdef extern from "_pardiso_ffi.h":
    void* pardiso_factor_handler_address()
    void* pardiso_solve_handler_address()
    void* pardiso_release_handler_address()
    void* pardiso_solve_once_handler_address()
    long pardiso_analysis_count(long solver_id)
    void pardiso_reset_analysis_count(long solver_id)


cdef object _capsule(void* address):
    # No name and no destructor, matching the convention jax.ffi.pycapsule
    # uses for handlers loaded through ctypes.
    return PyCapsule_New(address, NULL, NULL)


def _register_targets():
    import jax

    jax.ffi.register_ffi_target(
        "pardiso_mkl_jax_factor", _capsule(pardiso_factor_handler_address()), platform="cpu"
    )
    jax.ffi.register_ffi_target(
        "pardiso_mkl_jax_solve", _capsule(pardiso_solve_handler_address()), platform="cpu"
    )
    jax.ffi.register_ffi_target(
        "pardiso_mkl_jax_release", _capsule(pardiso_release_handler_address()), platform="cpu"
    )
    jax.ffi.register_ffi_target(
        "pardiso_mkl_jax_solve_once",
        _capsule(pardiso_solve_once_handler_address()),
        platform="cpu",
    )


_register_targets()


def analysis_count(solver_id):
    """Number of analysis (phase 11) calls run for solver_id. Test hook."""
    return pardiso_analysis_count(solver_id)


def reset_analysis_count(solver_id):
    """Reset the analysis call counter for solver_id. Test hook."""
    pardiso_reset_analysis_count(solver_id)
