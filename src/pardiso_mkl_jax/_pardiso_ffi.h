// Accessors that expose the compiled XLA FFI handler symbols to the Cython
// layer. Each function returns the address of a handler compiled in
// _pardiso_ffi.cc, which the Python side wraps in a PyCapsule and registers
// as an XLA custom call target via jax.ffi.register_ffi_target.

#ifndef PARDISO_MKL_JAX_FFI_H_
#define PARDISO_MKL_JAX_FFI_H_

#include <cstdint>

extern "C" {

// Handler for the analyze step (phase 11). Returns a content-hash handle for
// the matrix, which every later stage threads through as ordinary data. Pure:
// two identical analyze calls dedup to one cache entry.
void* pardiso_analyze_handler_address();

// Handler that re-runs the analyze step (phase 11) for a possibly new matrix
// or options, retiring the old handle and returning the content-hash handle of
// the new analysis. The numeric factorization is gone afterwards.
void* pardiso_reanalyze_handler_address();

// Handler for the numeric factorization step (phase 22). Takes an analyze
// handle and returns a distinct factor-domain handle naming the factorization
// of these values, so a downstream solve that consumes it is ordered after it.
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

// Writes this package's 64 iparm defaults for a matrix type into out. Lets
// the Python side work out what value an entry will take for a call without
// keeping its own copy of those defaults.
void pardiso_default_iparm(long matrix_type, int32_t* out);

// Read and reset the analysis call counter for a handle. Used by tests to
// assert that a reused factorization does not re-run the symbolic phase.
long pardiso_analysis_count(unsigned long long handle);
void pardiso_reset_analysis_count(unsigned long long handle);

// Read and reset the process-wide rebuild counter. It rises whenever a call
// lands on an evicted or freed handle and rebuilds the factorization, so a
// rising count is the signal that the cache is too small. Resetting also
// clears the per-reason totals.
long pardiso_rebuild_count();
void pardiso_reset_rebuild_count();

// Per-reason rebuild total, indexed by RebuildReason, for rebuild_stats().
long pardiso_rebuild_reason_count(int reason);

}  // extern "C"

#endif  // PARDISO_MKL_JAX_FFI_H_
