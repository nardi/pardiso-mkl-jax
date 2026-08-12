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
from libc.stdint cimport int32_t

import numpy as np


cdef extern from "_pardiso_ffi.h":
    void* pardiso_analyze_handler_address()
    void* pardiso_reanalyze_handler_address()
    void* pardiso_factor_handler_address()
    void* pardiso_solve_handler_address()
    void* pardiso_factor_solve_handler_address()
    void* pardiso_release_handler_address()
    void* pardiso_solve_once_handler_address()
    void pardiso_default_iparm(long matrix_type, int32_t* out)
    long pardiso_analysis_count(long handle)
    void pardiso_reset_analysis_count(long handle)


cdef object _capsule(void* address):
    # No name and no destructor, matching the convention jax.ffi.pycapsule
    # uses for handlers loaded through ctypes.
    return PyCapsule_New(address, NULL, NULL)


def _register_targets():
    import jax

    jax.ffi.register_ffi_target(
        "pardiso_mkl_jax_analyze", _capsule(pardiso_analyze_handler_address()), platform="cpu"
    )
    jax.ffi.register_ffi_target(
        "pardiso_mkl_jax_reanalyze", _capsule(pardiso_reanalyze_handler_address()), platform="cpu"
    )
    jax.ffi.register_ffi_target(
        "pardiso_mkl_jax_factor", _capsule(pardiso_factor_handler_address()), platform="cpu"
    )
    jax.ffi.register_ffi_target(
        "pardiso_mkl_jax_solve", _capsule(pardiso_solve_handler_address()), platform="cpu"
    )
    jax.ffi.register_ffi_target(
        "pardiso_mkl_jax_factor_solve",
        _capsule(pardiso_factor_solve_handler_address()),
        platform="cpu",
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


def default_iparm(matrix_type):
    """This package's 64 iparm defaults for matrix_type, as an int32 numpy array.

    Read straight out of InitializeIparm in _pardiso_ffi.cc, so the Python
    side never has to restate those defaults. See primitive.default_iparm for
    the cached wrapper callers should use.
    """
    cdef int32_t[::1] out = np.zeros(64, dtype=np.int32)
    pardiso_default_iparm(int(matrix_type), &out[0])
    return np.asarray(out)


def analysis_count(handle):
    """Number of analysis (phase 11) calls run for handle. Test hook.

    handle may be a plain int or a JAX/NumPy scalar array, as returned by
    analyze(), so it is coerced through int() before reaching the C function.
    """
    return pardiso_analysis_count(int(handle))


def reset_analysis_count(handle):
    """Reset the analysis call counter for handle. Test hook."""
    pardiso_reset_analysis_count(int(handle))
