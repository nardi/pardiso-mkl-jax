// XLA FFI handlers wrapping the oneMKL Pardiso direct sparse solver.
//
// Pardiso keeps its factorization in an opaque native handle ("pt") that
// must persist across calls to be reused. XLA FFI calls are stateless from
// JAX's point of view, so we keep a process-global registry mapping an
// integer key to the native state. That key is itself threaded through JAX
// as an ordinary int64 array value ("the handle"): analyze allocates a fresh
// key and returns it, factor and solve take it as an input, and release
// consumes it. Because every stage takes the previous stage's handle as
// data, XLA orders the whole analyze -> factor -> solve -> release lifecycle
// by data dependency, the same way it orders any other computation, so the
// lifecycle can be expressed entirely inside a jit trace and each runtime
// invocation of a compiled function gets its own registry entry.
//
// All buffers are read directly from the pointers XLA hands us. There is no
// copying: the CSR arrays and right-hand sides passed in from Python flow
// straight into Pardiso, and Pardiso writes its solution straight into the
// output buffer XLA allocated.
//
// One layout detail matters here. Pardiso stores its b and x arrays for
// multiple right-hand sides in column-major order as (n, num_right_hand_sides):
// element (row, column) lives at offset row + column * n. XLA buffers are
// row-major. The Python side accounts for this by shaping right_hand_side and
// the solution as (num_right_hand_sides, n) rather than (n,
// num_right_hand_sides): a row-major array of that shape has exactly the same
// byte layout as the column-major array Pardiso expects, so no transpose is
// needed on either side of the call.

#include "_pardiso_ffi.h"

#include <mkl.h>
#include <mkl_pardiso.h>
#include <mkl_service.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace pardiso_mkl_jax {
namespace {

// Native state for one handle: the opaque Pardiso handle, its parameter
// array, and a little bookkeeping used by tests to check that analysis is
// only run when expected.
struct PardisoState {
  void* handle[64] = {};
  MKL_INT iparm[64] = {};
  MKL_INT matrix_type = 0;
  MKL_INT dimension = 0;
  long analysis_count = 0;
};

// Forces the LP64 interface layer, matching the int32 CSR indices this
// package uses throughout. Without this, MKL_INTERFACE_LAYER in the
// environment could silently switch MKL to ILP64, which would misinterpret
// our buffers. Must run before any other MKL call, which a namespace-scope
// static initializer guarantees.
const bool kInterfaceLayerInitialized = [] {
  mkl_set_interface_layer(MKL_INTERFACE_LP64);
  return true;
}();

std::mutex& RegistryMutex() {
  static std::mutex mutex;
  return mutex;
}

// Must only be accessed while holding RegistryMutex.
std::unordered_map<int64_t, PardisoState>& Registry() {
  static std::unordered_map<int64_t, PardisoState> registry;
  return registry;
}

// Monotonic source of fresh registry keys, allocated at runtime inside the
// analyze handler rather than baked in at Python trace time. Never reused,
// so two concurrent or repeated invocations of a compiled function each get
// their own registry entry instead of colliding on a trace-time id.
std::atomic<int64_t>& HandleCounter() {
  static std::atomic<int64_t> counter{1};
  return counter;
}

// Reinterprets a buffer of our zero-copy int32 CSR arrays as MKL_INT, the
// integer type Pardiso's C API expects under the LP64 interface layer we
// select at module load. On every platform we support, MKL_INT is a plain
// 32-bit int here, the same width and representation as int32_t.
MKL_INT* AsMklInt(const int32_t* data) {
  return const_cast<MKL_INT*>(reinterpret_cast<const MKL_INT*>(data));
}

// Fills iparm with this package's safe defaults. iparm[0] is set to 1 (every
// entry supplied explicitly) rather than left at 0 (let MKL fill in its own
// defaults), because MKL's own default for non-symmetric matrices enables
// weighted matching (iparm[12] = 1), and that heuristic segfaults inside
// mkl_pds_lp64_kuhn_munkres in this MKL build, reproduced with a minimal
// standalone Pardiso call outside this package entirely, on ordinary,
// well-conditioned matrices, not just degenerate ones. Individually
// overriding iparm[12] back to 0 while leaving iparm[0] at 0 does not avoid
// the crash: MKL resets it internally before the matching step runs. Taking
// over iparm[0] = 1 is the only way to keep it disabled. Scaling (iparm[10])
// is unaffected by this bug and left enabled for non-symmetric matrices.
void InitializeIparm(MKL_INT* iparm, MKL_INT matrix_type) {
  iparm[0] = 1;  // every entry below is used as given, MKL fills nothing in
  iparm[1] = 2;  // nested dissection (METIS-based) fill-reducing ordering
  iparm[9] = (matrix_type == 11 || matrix_type == 1) ? 13 : 8;  // pivot perturbation exponent
  iparm[10] = (matrix_type == 11) ? 1 : 0;  // scaling, non-symmetric matrices only
  iparm[12] = 0;  // weighted matching disabled: see the crash explained above
  iparm[34] = 1;  // zero-based indexing, so our CSR arrays need no reindexing
}

std::string PardisoErrorMessage(const char* stage, MKL_INT error) {
  return std::string("pardiso ") + stage + " failed with error code " + std::to_string(error);
}

// Applies a caller-supplied override on top of the defaults InitializeIparm
// already set: overlay_mask[i] != 0 means overlay_values[i] replaces
// iparm[i]. All validation and any user-facing warnings happen entirely on
// the Python side (canonicalize_overlay in iparm.py) before this ever runs,
// so this is purely mechanical.
void ApplyOverlay(MKL_INT* iparm, const int32_t* overlay_mask, const int32_t* overlay_values) {
  for (int i = 0; i < 64; ++i) {
    if (overlay_mask[i] != 0) {
      iparm[i] = static_cast<MKL_INT>(overlay_values[i]);
    }
  }
}

}  // namespace

extern "C" long pardiso_analysis_count(long handle) {
  std::lock_guard<std::mutex> lock(RegistryMutex());
  auto iterator = Registry().find(handle);
  return iterator == Registry().end() ? 0 : iterator->second.analysis_count;
}

extern "C" void pardiso_reset_analysis_count(long handle) {
  std::lock_guard<std::mutex> lock(RegistryMutex());
  auto iterator = Registry().find(handle);
  if (iterator != Registry().end()) {
    iterator->second.analysis_count = 0;
  }
}

namespace {

// Analyze (phase 11). Allocates a fresh registry key, runs the symbolic
// factorization into a new PardisoState, and returns the key as an int64
// handle value. Every later stage (factor, solve, release) takes this
// handle as an ordinary input, which is what lets XLA order the lifecycle by
// data dependency instead of by a static, trace-time-baked id.
ffi::Error PardisoAnalyzeImpl(int64_t matrix_type, int64_t dimension,
                               ffi::Buffer<ffi::S32> indptr, ffi::Buffer<ffi::S32> indices,
                               ffi::Buffer<ffi::F64> values, ffi::Buffer<ffi::S32> options_mask,
                               ffi::Buffer<ffi::S32> options_values,
                               ffi::ResultBuffer<ffi::S64> handle_out,
                               ffi::ResultBuffer<ffi::S32> status,
                               ffi::ResultBuffer<ffi::S32> final_iparm) {
  int64_t handle = HandleCounter().fetch_add(1);

  std::lock_guard<std::mutex> lock(RegistryMutex());
  PardisoState& state = Registry()[handle];
  state.matrix_type = static_cast<MKL_INT>(matrix_type);
  state.dimension = static_cast<MKL_INT>(dimension);
  InitializeIparm(state.iparm, state.matrix_type);
  ApplyOverlay(state.iparm, options_mask.typed_data(), options_values.typed_data());

  MKL_INT maxfct = 1;
  MKL_INT mnum = 1;
  MKL_INT phase_value = 11;
  MKL_INT number_of_right_hand_sides = 0;
  MKL_INT message_level = 0;
  MKL_INT error = 0;

  pardiso(state.handle, &maxfct, &mnum, &state.matrix_type, &phase_value, &state.dimension,
          const_cast<double*>(values.typed_data()), AsMklInt(indptr.typed_data()),
          AsMklInt(indices.typed_data()), /*perm=*/nullptr, &number_of_right_hand_sides,
          state.iparm, &message_level, /*b=*/nullptr, /*x=*/nullptr, &error);

  state.analysis_count += 1;
  handle_out->typed_data()[0] = handle;
  std::memcpy(final_iparm->typed_data(), state.iparm, sizeof(MKL_INT) * 64);
  status->typed_data()[0] = static_cast<int32_t>(error);
  if (error != 0) {
    return ffi::Error::Internal(PardisoErrorMessage("analyze", error));
  }
  return ffi::Error::Success();
}

// Numeric factorization (phase 22) against the state already allocated by
// analyze for this handle. Returns the same handle unchanged, so a later
// solve that takes this handler's output as input is ordered after the
// factorization.
ffi::Error PardisoFactorImpl(int64_t matrix_type, int64_t dimension,
                              ffi::Buffer<ffi::S64> handle_in, ffi::Buffer<ffi::S32> indptr,
                              ffi::Buffer<ffi::S32> indices, ffi::Buffer<ffi::F64> values,
                              ffi::Buffer<ffi::S32> options_mask,
                              ffi::Buffer<ffi::S32> options_values,
                              ffi::ResultBuffer<ffi::S64> handle_out,
                              ffi::ResultBuffer<ffi::S32> status,
                              ffi::ResultBuffer<ffi::S32> final_iparm) {
  int64_t handle = handle_in.typed_data()[0];

  std::lock_guard<std::mutex> lock(RegistryMutex());
  PardisoState& state = Registry()[handle];
  state.matrix_type = static_cast<MKL_INT>(matrix_type);
  state.dimension = static_cast<MKL_INT>(dimension);
  InitializeIparm(state.iparm, state.matrix_type);
  ApplyOverlay(state.iparm, options_mask.typed_data(), options_values.typed_data());

  MKL_INT maxfct = 1;
  MKL_INT mnum = 1;
  MKL_INT phase_value = 22;
  MKL_INT number_of_right_hand_sides = 0;
  MKL_INT message_level = 0;
  MKL_INT error = 0;

  pardiso(state.handle, &maxfct, &mnum, &state.matrix_type, &phase_value, &state.dimension,
          const_cast<double*>(values.typed_data()), AsMklInt(indptr.typed_data()),
          AsMklInt(indices.typed_data()), /*perm=*/nullptr, &number_of_right_hand_sides,
          state.iparm, &message_level, /*b=*/nullptr, /*x=*/nullptr, &error);

  handle_out->typed_data()[0] = handle;
  std::memcpy(final_iparm->typed_data(), state.iparm, sizeof(MKL_INT) * 64);
  status->typed_data()[0] = static_cast<int32_t>(error);
  if (error != 0) {
    return ffi::Error::Internal(PardisoErrorMessage("factor", error));
  }
  return ffi::Error::Success();
}

// Solve (phase 33) against a factorization already produced for handle.
// transpose_mode is iparm[11] directly: 0 solves Ax = b, 2 solves A^T x = b
// (conjugate transpose, value 1, coincides with plain transpose for the
// real-valued matrices this package supports). Reuses the same
// factorization either way: an LU (or LDL^T) factorization of A supports
// solving with A^T through forward/back substitution in the opposite
// order, with no need to refactorize.
ffi::Error PardisoSolveImpl(int64_t matrix_type, int64_t dimension,
                             int64_t number_of_right_hand_sides, int64_t transpose_mode,
                             ffi::Buffer<ffi::S64> handle_in, ffi::Buffer<ffi::S32> indptr,
                             ffi::Buffer<ffi::S32> indices, ffi::Buffer<ffi::F64> values,
                             ffi::Buffer<ffi::F64> right_hand_side,
                             ffi::Buffer<ffi::S32> options_mask,
                             ffi::Buffer<ffi::S32> options_values,
                             ffi::ResultBuffer<ffi::F64> solution,
                             ffi::ResultBuffer<ffi::S32> final_iparm) {
  int64_t handle = handle_in.typed_data()[0];

  std::lock_guard<std::mutex> lock(RegistryMutex());
  PardisoState& state = Registry()[handle];
  state.matrix_type = static_cast<MKL_INT>(matrix_type);
  state.dimension = static_cast<MKL_INT>(dimension);
  InitializeIparm(state.iparm, state.matrix_type);
  ApplyOverlay(state.iparm, options_mask.typed_data(), options_values.typed_data());
  // Set unconditionally (not only when transposed) so a later solve on the
  // same handle without transpose is not left with a stale value from an
  // earlier call. Applied after ApplyOverlay: canonicalize_overlay in
  // iparm.py guarantees a caller-supplied overlay never touches index 11
  // (transpose_mode is the sole owner of it), so there is no real conflict
  // to resolve here.
  state.iparm[11] = static_cast<MKL_INT>(transpose_mode);

  MKL_INT maxfct = 1;
  MKL_INT mnum = 1;
  MKL_INT phase_value = 33;
  MKL_INT nrhs = static_cast<MKL_INT>(number_of_right_hand_sides);
  MKL_INT message_level = 0;
  MKL_INT error = 0;

  pardiso(state.handle, &maxfct, &mnum, &state.matrix_type, &phase_value, &state.dimension,
          const_cast<double*>(values.typed_data()), AsMklInt(indptr.typed_data()),
          AsMklInt(indices.typed_data()), /*perm=*/nullptr, &nrhs, state.iparm, &message_level,
          const_cast<double*>(right_hand_side.typed_data()), solution->typed_data(), &error);

  std::memcpy(final_iparm->typed_data(), state.iparm, sizeof(MKL_INT) * 64);
  if (error != 0) {
    return ffi::Error::Internal(PardisoErrorMessage("solve", error));
  }
  return ffi::Error::Success();
}

// Numeric factorization and solve in a single call (combined phase 23),
// reusing the symbolic analysis already produced for handle. Doing both in
// one FFI call keeps stateful reuse safe even when the handle otherwise
// carries the ordering, since it also collapses two native calls that touch
// the same registry entry into one. The analysis (phase 11) is not
// repeated, so analysis_count is left untouched.
ffi::Error PardisoFactorSolveImpl(int64_t matrix_type, int64_t dimension,
                                   int64_t number_of_right_hand_sides, int64_t transpose_mode,
                                   ffi::Buffer<ffi::S64> handle_in, ffi::Buffer<ffi::S32> indptr,
                                   ffi::Buffer<ffi::S32> indices, ffi::Buffer<ffi::F64> values,
                                   ffi::Buffer<ffi::F64> right_hand_side,
                                   ffi::Buffer<ffi::S32> options_mask,
                                   ffi::Buffer<ffi::S32> options_values,
                                   ffi::ResultBuffer<ffi::F64> solution,
                                   ffi::ResultBuffer<ffi::S32> final_iparm) {
  int64_t handle = handle_in.typed_data()[0];

  std::lock_guard<std::mutex> lock(RegistryMutex());
  PardisoState& state = Registry()[handle];
  state.matrix_type = static_cast<MKL_INT>(matrix_type);
  state.dimension = static_cast<MKL_INT>(dimension);
  InitializeIparm(state.iparm, state.matrix_type);
  ApplyOverlay(state.iparm, options_mask.typed_data(), options_values.typed_data());
  // Set unconditionally, matching PardisoSolveImpl, so a later call without
  // transpose is not left with a stale value from an earlier one. Applied
  // after ApplyOverlay for the same reason given there.
  state.iparm[11] = static_cast<MKL_INT>(transpose_mode);

  MKL_INT maxfct = 1;
  MKL_INT mnum = 1;
  MKL_INT phase_value = 23;
  MKL_INT nrhs = static_cast<MKL_INT>(number_of_right_hand_sides);
  MKL_INT message_level = 0;
  MKL_INT error = 0;

  pardiso(state.handle, &maxfct, &mnum, &state.matrix_type, &phase_value, &state.dimension,
          const_cast<double*>(values.typed_data()), AsMklInt(indptr.typed_data()),
          AsMklInt(indices.typed_data()), /*perm=*/nullptr, &nrhs, state.iparm, &message_level,
          const_cast<double*>(right_hand_side.typed_data()), solution->typed_data(), &error);

  std::memcpy(final_iparm->typed_data(), state.iparm, sizeof(MKL_INT) * 64);
  if (error != 0) {
    return ffi::Error::Internal(PardisoErrorMessage("factor_and_solve", error));
  }
  return ffi::Error::Success();
}

// Frees the native memory for handle (phase -1) and drops it from the
// registry. A handle that is not present is treated as already released.
ffi::Error PardisoReleaseImpl(ffi::Buffer<ffi::S64> handle_in, ffi::ResultBuffer<ffi::S32> status) {
  int64_t handle = handle_in.typed_data()[0];

  std::lock_guard<std::mutex> lock(RegistryMutex());
  auto iterator = Registry().find(handle);
  if (iterator == Registry().end()) {
    status->typed_data()[0] = 0;
    return ffi::Error::Success();
  }
  PardisoState& state = iterator->second;

  MKL_INT maxfct = 1;
  MKL_INT mnum = 1;
  MKL_INT phase_value = -1;
  MKL_INT nrhs = 0;
  MKL_INT message_level = 0;
  MKL_INT error = 0;

  pardiso(state.handle, &maxfct, &mnum, &state.matrix_type, &phase_value, &state.dimension,
          /*a=*/nullptr, /*ia=*/nullptr, /*ja=*/nullptr, /*perm=*/nullptr, &nrhs, state.iparm,
          &message_level, /*b=*/nullptr, /*x=*/nullptr, &error);

  Registry().erase(iterator);
  status->typed_data()[0] = static_cast<int32_t>(error);
  if (error != 0) {
    return ffi::Error::Internal(PardisoErrorMessage("release", error));
  }
  return ffi::Error::Success();
}

// Stateless one-shot solve: analyze, factor, and solve in a single call
// (combined phase 13) against a local handle, released again before
// returning. Used by the functional solve() entry point, which never reuses
// a factorization and so never needs a registry entry.
ffi::Error PardisoSolveOnceImpl(int64_t matrix_type, int64_t dimension,
                                 int64_t number_of_right_hand_sides, int64_t transpose_mode,
                                 ffi::Buffer<ffi::S32> indptr, ffi::Buffer<ffi::S32> indices,
                                 ffi::Buffer<ffi::F64> values,
                                 ffi::Buffer<ffi::F64> right_hand_side,
                                 ffi::Buffer<ffi::S32> options_mask,
                                 ffi::Buffer<ffi::S32> options_values,
                                 ffi::ResultBuffer<ffi::F64> solution,
                                 ffi::ResultBuffer<ffi::S32> final_iparm) {
  void* handle[64] = {};
  MKL_INT iparm[64] = {};
  MKL_INT mtype_value = static_cast<MKL_INT>(matrix_type);
  InitializeIparm(iparm, mtype_value);
  ApplyOverlay(iparm, options_mask.typed_data(), options_values.typed_data());
  iparm[11] = static_cast<MKL_INT>(transpose_mode);

  MKL_INT maxfct = 1;
  MKL_INT mnum = 1;
  MKL_INT n = static_cast<MKL_INT>(dimension);
  MKL_INT nrhs = static_cast<MKL_INT>(number_of_right_hand_sides);
  MKL_INT message_level = 0;
  MKL_INT solve_phase = 13;
  MKL_INT error = 0;

  pardiso(handle, &maxfct, &mnum, &mtype_value, &solve_phase, &n,
          const_cast<double*>(values.typed_data()), AsMklInt(indptr.typed_data()),
          AsMklInt(indices.typed_data()), /*perm=*/nullptr, &nrhs, iparm, &message_level,
          const_cast<double*>(right_hand_side.typed_data()), solution->typed_data(), &error);

  // Captured right after the solving call and before the release call
  // below, which reuses the same iparm array and could otherwise overwrite
  // these diagnostics with whatever the release phase leaves behind.
  std::memcpy(final_iparm->typed_data(), iparm, sizeof(MKL_INT) * 64);

  // Always release the local handle, even on failure, so a failed solve
  // never leaks native memory.
  MKL_INT release_phase = -1;
  MKL_INT release_error = 0;
  pardiso(handle, &maxfct, &mnum, &mtype_value, &release_phase, &n, nullptr, nullptr, nullptr,
          nullptr, &nrhs, iparm, &message_level, nullptr, nullptr, &release_error);

  if (error != 0) {
    return ffi::Error::Internal(PardisoErrorMessage("solve", error));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kPardisoAnalyzeHandler, PardisoAnalyzeImpl,
                               ffi::Ffi::Bind()
                                   .Attr<int64_t>("matrix_type")
                                   .Attr<int64_t>("dimension")
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indptr
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indices
                                   .Arg<ffi::Buffer<ffi::F64>>()  // values
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_mask
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_values
                                   .Ret<ffi::Buffer<ffi::S64>>()  // handle
                                   .Ret<ffi::Buffer<ffi::S32>>()  // status
                                   .Ret<ffi::Buffer<ffi::S32>>()  // final_iparm
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kPardisoFactorHandler, PardisoFactorImpl,
                               ffi::Ffi::Bind()
                                   .Attr<int64_t>("matrix_type")
                                   .Attr<int64_t>("dimension")
                                   .Arg<ffi::Buffer<ffi::S64>>()  // handle
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indptr
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indices
                                   .Arg<ffi::Buffer<ffi::F64>>()  // values
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_mask
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_values
                                   .Ret<ffi::Buffer<ffi::S64>>()  // handle
                                   .Ret<ffi::Buffer<ffi::S32>>()  // status
                                   .Ret<ffi::Buffer<ffi::S32>>()  // final_iparm
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kPardisoSolveHandler, PardisoSolveImpl,
                               ffi::Ffi::Bind()
                                   .Attr<int64_t>("matrix_type")
                                   .Attr<int64_t>("dimension")
                                   .Attr<int64_t>("number_of_right_hand_sides")
                                   .Attr<int64_t>("transpose_mode")
                                   .Arg<ffi::Buffer<ffi::S64>>()  // handle
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indptr
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indices
                                   .Arg<ffi::Buffer<ffi::F64>>()  // values
                                   .Arg<ffi::Buffer<ffi::F64>>()  // right_hand_side
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_mask
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_values
                                   .Ret<ffi::Buffer<ffi::F64>>()  // solution
                                   .Ret<ffi::Buffer<ffi::S32>>()  // final_iparm
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kPardisoFactorSolveHandler, PardisoFactorSolveImpl,
                               ffi::Ffi::Bind()
                                   .Attr<int64_t>("matrix_type")
                                   .Attr<int64_t>("dimension")
                                   .Attr<int64_t>("number_of_right_hand_sides")
                                   .Attr<int64_t>("transpose_mode")
                                   .Arg<ffi::Buffer<ffi::S64>>()  // handle
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indptr
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indices
                                   .Arg<ffi::Buffer<ffi::F64>>()  // values
                                   .Arg<ffi::Buffer<ffi::F64>>()  // right_hand_side
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_mask
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_values
                                   .Ret<ffi::Buffer<ffi::F64>>()  // solution
                                   .Ret<ffi::Buffer<ffi::S32>>()  // final_iparm
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kPardisoReleaseHandler, PardisoReleaseImpl,
                               ffi::Ffi::Bind()
                                   .Arg<ffi::Buffer<ffi::S64>>()  // handle
                                   .Ret<ffi::Buffer<ffi::S32>>()  // status
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kPardisoSolveOnceHandler, PardisoSolveOnceImpl,
                               ffi::Ffi::Bind()
                                   .Attr<int64_t>("matrix_type")
                                   .Attr<int64_t>("dimension")
                                   .Attr<int64_t>("number_of_right_hand_sides")
                                   .Attr<int64_t>("transpose_mode")
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indptr
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indices
                                   .Arg<ffi::Buffer<ffi::F64>>()  // values
                                   .Arg<ffi::Buffer<ffi::F64>>()  // right_hand_side
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_mask
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_values
                                   .Ret<ffi::Buffer<ffi::F64>>()  // solution
                                   .Ret<ffi::Buffer<ffi::S32>>()  // final_iparm
);

}  // namespace

}  // namespace pardiso_mkl_jax

extern "C" void* pardiso_analyze_handler_address() {
  return reinterpret_cast<void*>(pardiso_mkl_jax::kPardisoAnalyzeHandler);
}

extern "C" void* pardiso_factor_handler_address() {
  return reinterpret_cast<void*>(pardiso_mkl_jax::kPardisoFactorHandler);
}

extern "C" void* pardiso_solve_handler_address() {
  return reinterpret_cast<void*>(pardiso_mkl_jax::kPardisoSolveHandler);
}

extern "C" void* pardiso_factor_solve_handler_address() {
  return reinterpret_cast<void*>(pardiso_mkl_jax::kPardisoFactorSolveHandler);
}

extern "C" void* pardiso_release_handler_address() {
  return reinterpret_cast<void*>(pardiso_mkl_jax::kPardisoReleaseHandler);
}

extern "C" void* pardiso_solve_once_handler_address() {
  return reinterpret_cast<void*>(pardiso_mkl_jax::kPardisoSolveOnceHandler);
}
