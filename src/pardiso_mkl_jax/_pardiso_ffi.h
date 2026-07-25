// Accessors that expose the compiled XLA FFI handler symbols to the Cython
// layer. Each function returns the address of a handler compiled in
// _pardiso_ffi.cc, which the Python side wraps in a PyCapsule and registers
// as an XLA custom call target via jax.ffi.register_ffi_target.

#ifndef PARDISO_MKL_JAX_FFI_H_
#define PARDISO_MKL_JAX_FFI_H_

extern "C" {

// Stateful handler for the analyze step (phase 11). Allocates a fresh
// registry entry and returns its key as an int64 handle value, which every
// later stage threads through as ordinary data.
void* pardiso_analyze_handler_address();

// Stateful handler for the numeric factorization step (phase 22). Takes the
// handle returned by analyze and passes it through unchanged, so a
// downstream solve that consumes this handler's output is ordered after it.
void* pardiso_factor_handler_address();

// Stateful handler for the solve step (phase 33), run against a
// factorization already produced by the factor (or analyze) handler for the
// same handle.
void* pardiso_solve_handler_address();

// Stateful handler for the fused numeric factorization and solve (phase 23),
// reusing the analysis already produced for the same handle.
void* pardiso_factor_solve_handler_address();

// Releases the native memory associated with a handle and removes it from
// the registry.
void* pardiso_release_handler_address();

// Stateless handler for the one-shot functional solve. Runs analyze, factor,
// and solve (combined phase 13) with a local handle that is released again
// before the call returns, so it never touches the registry.
void* pardiso_solve_once_handler_address();

// Read and reset the analysis call counter for a handle. Used by tests to
// assert that a reused factorization does not re-run the symbolic phase.
long pardiso_analysis_count(long handle);
void pardiso_reset_analysis_count(long handle);

}  // extern "C"

#endif  // PARDISO_MKL_JAX_FFI_H_
