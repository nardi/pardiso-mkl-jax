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
#include <cstdlib>
#include <cstring>
#include <list>
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
  // Generation stamp for the factorization this state currently holds. Every
  // write (analyze, reanalyze, factor, factor_and_solve) sets a fresh value
  // from VersionCounter, and a solve carries the version it expects so the
  // solve handler can reject a token left over from before a later write.
  int64_t version = 0;
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

// Monotonic source of factorization version stamps, never reused. Every write
// takes a fresh value and stores it on its state, and a solve checks the
// version it was given against the state's current one. Because the counter
// only ever goes up, a token from before a later write always compares
// unequal, which is how a stale token is caught. See PardisoState::version.
std::atomic<int64_t>& VersionCounter() {
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

// Fills iparm with this package's defaults. iparm[0] is set to 1, meaning
// every entry below is used exactly as given and MKL fills nothing in
// itself. That takeover is what lets us set iparm[34] (zero-based indexing)
// and iparm[11] (transpose solves) and rely on them surviving, but it comes
// with a sharp edge: every entry we do not assign stays at the 0 this array
// was zero-initialized to, which is *not* the same as the default MKL would
// have chosen. So each entry whose MKL default is non-zero has to be
// restated here explicitly, per matrix type, or it is silently turned off.
//
// The values below reproduce pardisoinit's defaults for every matrix type
// this package supports, with two deliberate exceptions, both noted inline:
// iparm[1] and the reporting-only entries iparm[17] / iparm[18].
void InitializeIparm(MKL_INT* iparm, MKL_INT matrix_type) {
  const bool nonsymmetric = matrix_type == 11 || matrix_type == 13;
  const bool symmetric_indefinite =
      matrix_type == -2 || matrix_type == -4 || matrix_type == 6;

  iparm[0] = 1;  // every entry below is used as given, MKL fills nothing in
  // Serial nested dissection, rather than pardisoinit's parallel nested
  // dissection (3). The only entry here chosen for its own sake: it makes
  // the fill-reducing ordering, and so the whole factorization, reproducible
  // run to run regardless of thread count.
  iparm[1] = 2;
  iparm[7] = 2;  // iterative refinement steps, the backstop for perturbed pivots
  iparm[9] = (matrix_type == 11 || matrix_type == 1) ? 13 : 8;  // pivot perturbation exponent
  iparm[10] = nonsymmetric ? 1 : 0;  // scaling, non-symmetric matrices only
  // Weighted matching: permutes large entries onto the diagonal before
  // factoring. Enabled for non-symmetric matrices, as pardisoinit does.
  // Without it, a matrix with zeros on its diagonal (common for the
  // saddle-point and constraint blocks this solver is used on) drives
  // Pardiso into pivot perturbation, and it then happily returns a solution
  // with a large residual and no error code at all.
  iparm[12] = nonsymmetric ? 1 : 0;
  // Bunch-Kaufman pivoting for symmetric indefinite matrices, which need it
  // for the same reason: without it, a zero diagonal entry has no 2x2 pivot
  // to fall back on and gets perturbed instead.
  iparm[20] = symmetric_indefinite ? 1 : 0;
  iparm[34] = 1;  // zero-based indexing, so our CSR arrays need no reindexing
  // iparm[17] and iparm[18] are left at 0 rather than pardisoinit's -1: both
  // only request statistics (non-zeros in the factors, MFLOP count) that
  // this package does not surface, and computing them is not free.
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

// Cache bookkeeping ==================================================================
//
// The registry is bounded so that forgetting to release a handle leaks a
// limited amount of memory rather than growing without end. Handles are kept
// in an LRU list, and the least recently used one is evicted once the map
// grows past the cache size. An evicted or released handle is not gone for
// good: every stateful handler carries the matrix it needs, so a call landing
// on a missing handle rebuilds its factorization from that matrix (see the
// rebuild helpers below). This is what makes use of a freed handle safe.

// Ordering of handles, most recently used at the front. Guarded by
// RegistryMutex, like Registry itself.
std::list<int64_t>& LruList() {
  static std::list<int64_t> lru;
  return lru;
}

// Counts how often a missing handle had to rebuild its factorization. A rising
// count means the cache is too small for the working set, so this is what the
// rebuild_count() test/diagnostic hook reports.
std::atomic<long>& RebuildCounter() {
  static std::atomic<long> counter{0};
  return counter;
}

// Cache size, from PARDISO_MKL_JAX_FACTOR_CACHE, defaulting to 8 live handles.
// Read on every access rather than cached so a test can set it per case.
size_t CacheCapacity() {
  const char* env = std::getenv("PARDISO_MKL_JAX_FACTOR_CACHE");
  if (env != nullptr) {
    char* end = nullptr;
    long value = std::strtol(env, &end, 10);
    if (end != env && value > 0) {
      return static_cast<size_t>(value);
    }
  }
  return 8;
}

// Whether PARDISO_MKL_JAX_STRICT_CACHE is set. In strict mode a rebuild is
// turned into an error instead of happening silently, so a lost factorization
// (the performance bug this guards against) surfaces loudly.
bool StrictCache() {
  const char* env = std::getenv("PARDISO_MKL_JAX_STRICT_CACHE");
  return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

// Move a handle to the front of the LRU list. remove() is O(n) but n is the
// cache size, a handful of entries, so this stays cheap.
void TouchLru(int64_t handle) {
  LruList().remove(handle);
  LruList().push_front(handle);
}

// Free a state's native factorization (phase -1). Errors are ignored: this
// runs during eviction, where there is no caller to report to and the memory
// is being dropped regardless.
void FreeState(PardisoState& state) {
  MKL_INT maxfct = 1, mnum = 1, phase = -1, nrhs = 0, message_level = 0, error = 0;
  pardiso(state.handle, &maxfct, &mnum, &state.matrix_type, &phase, &state.dimension,
          /*a=*/nullptr, /*ia=*/nullptr, /*ja=*/nullptr, /*perm=*/nullptr, &nrhs, state.iparm,
          &message_level, /*b=*/nullptr, /*x=*/nullptr, &error);
}

// Drop least-recently-used handles until the map is back within the cache
// size. The handle in use by the current call is at the front, so it is never
// the victim as long as the capacity is at least one.
void EvictIfNeeded() {
  const size_t capacity = CacheCapacity();
  while (Registry().size() > capacity && !LruList().empty()) {
    int64_t victim = LruList().back();
    LruList().pop_back();
    auto iterator = Registry().find(victim);
    if (iterator == Registry().end()) {
      continue;
    }
    FreeState(iterator->second);
    Registry().erase(iterator);
  }
}

// Run the symbolic analysis (phase 11) into state, from the given matrix. The
// caller has already set state's matrix type, dimension, and iparm. pt is
// zeroed first because Pardiso expects a clean handle for a fresh analysis.
MKL_INT RunAnalysis(PardisoState& state, const int32_t* indptr, const int32_t* indices,
                    const double* values) {
  std::memset(state.handle, 0, sizeof(state.handle));
  MKL_INT maxfct = 1, mnum = 1, phase = 11, nrhs = 0, message_level = 0, error = 0;
  pardiso(state.handle, &maxfct, &mnum, &state.matrix_type, &phase, &state.dimension,
          const_cast<double*>(values), AsMklInt(indptr), AsMklInt(indices), /*perm=*/nullptr,
          &nrhs, state.iparm, &message_level, /*b=*/nullptr, /*x=*/nullptr, &error);
  state.analysis_count += 1;
  return error;
}

// Run the numeric factorization (phase 22) into state, from the given matrix.
// Used only to rebuild a missing factorization before a solve.
MKL_INT RunNumeric(PardisoState& state, const int32_t* indptr, const int32_t* indices,
                   const double* values) {
  MKL_INT maxfct = 1, mnum = 1, phase = 22, nrhs = 0, message_level = 0, error = 0;
  pardiso(state.handle, &maxfct, &mnum, &state.matrix_type, &phase, &state.dimension,
          const_cast<double*>(values), AsMklInt(indptr), AsMklInt(indices), /*perm=*/nullptr,
          &nrhs, state.iparm, &message_level, /*b=*/nullptr, /*x=*/nullptr, &error);
  return error;
}

}  // namespace

// Hands this package's iparm defaults for a matrix type to the Python side,
// so nothing there has to keep a second copy of InitializeIparm in sync.
// solver.py needs them to work out the value an entry will actually take for
// a call, which is the overlay entry if there is one and this default
// otherwise.
extern "C" void pardiso_default_iparm(long matrix_type, int32_t* out) {
  MKL_INT iparm[64] = {};
  InitializeIparm(iparm, static_cast<MKL_INT>(matrix_type));
  for (int i = 0; i < 64; ++i) {
    out[i] = static_cast<int32_t>(iparm[i]);
  }
}

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

// Total number of rebuilds since load (or since the last reset). Process-wide,
// not per-handle, since a rebuild happens exactly because the handle is gone.
extern "C" long pardiso_rebuild_count() {
  return RebuildCounter().load();
}

extern "C" void pardiso_reset_rebuild_count() {
  RebuildCounter().store(0);
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
                               ffi::ResultBuffer<ffi::S64> version_out,
                               ffi::ResultBuffer<ffi::S32> status,
                               ffi::ResultBuffer<ffi::S32> final_iparm) {
  int64_t handle = HandleCounter().fetch_add(1);

  std::lock_guard<std::mutex> lock(RegistryMutex());
  PardisoState& state = Registry()[handle];
  state.matrix_type = static_cast<MKL_INT>(matrix_type);
  state.dimension = static_cast<MKL_INT>(dimension);
  state.version = VersionCounter().fetch_add(1);
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
  // Newest handle goes to the front, then evict so the cache stays bounded.
  TouchLru(handle);
  EvictIfNeeded();
  handle_out->typed_data()[0] = handle;
  version_out->typed_data()[0] = state.version;
  std::memcpy(final_iparm->typed_data(), state.iparm, sizeof(MKL_INT) * 64);
  status->typed_data()[0] = static_cast<int32_t>(error);
  if (error != 0) {
    return ffi::Error::Internal(PardisoErrorMessage("analyze", error));
  }
  return ffi::Error::Success();
}

// Re-analyze (phase 11) in place, against the state an earlier analyze
// already allocated for this handle. Frees the existing factorization first,
// then runs a fresh symbolic phase into the same registry entry, so the
// handle value never changes and later calls stay ordered against it by data
// dependency. That is the point of this handler: re-analyzing through the
// plain analyze handler would mint a second handle the caller then has to
// free separately.
//
// The free and the re-analysis happen under a single lock hold, so the entry
// is never observable in the half-freed state between them.
ffi::Error PardisoReanalyzeImpl(int64_t matrix_type, int64_t dimension,
                                 ffi::Buffer<ffi::S64> handle_in, ffi::Buffer<ffi::S32> indptr,
                                 ffi::Buffer<ffi::S32> indices, ffi::Buffer<ffi::F64> values,
                                 ffi::Buffer<ffi::S32> options_mask,
                                 ffi::Buffer<ffi::S32> options_values,
                                 ffi::ResultBuffer<ffi::S64> handle_out,
                                 ffi::ResultBuffer<ffi::S64> version_out,
                                 ffi::ResultBuffer<ffi::S32> status,
                                 ffi::ResultBuffer<ffi::S32> final_iparm) {
  int64_t handle = handle_in.typed_data()[0];

  // Every early return below still fills the result buffers. Returning an
  // error makes JAX raise rather than read them, but XLA allocated them
  // uninitialized and leaving them that way is a trap for anyone who later
  // makes a failure path non-fatal.
  handle_out->typed_data()[0] = handle;
  version_out->typed_data()[0] = 0;
  std::memset(final_iparm->typed_data(), 0, sizeof(int32_t) * 64);

  std::lock_guard<std::mutex> lock(RegistryMutex());
  auto iterator = Registry().find(handle);
  bool missing = iterator == Registry().end();
  // A missing handle is no longer an error: it was evicted or released, so we
  // just analyze fresh in place under the same handle. Strict mode is the
  // exception, turning that rebuild into a loud error for debugging.
  if (missing && StrictCache()) {
    status->typed_data()[0] = -1;
    return ffi::Error::Internal("pardiso reanalyze: handle " + std::to_string(handle) +
                                " was evicted or freed and strict cache mode is on");
  }
  if (missing) {
    RebuildCounter().fetch_add(1);
  }
  PardisoState& state = Registry()[handle];

  MKL_INT maxfct = 1;
  MKL_INT mnum = 1;
  MKL_INT number_of_right_hand_sides = 0;
  MKL_INT message_level = 0;

  if (!missing) {
    // Release the existing factorization first, against the matrix type and
    // dimension it was allocated for. The call attributes describe the *new*
    // analysis and only take effect below.
    MKL_INT release_phase = -1;
    MKL_INT release_error = 0;
    pardiso(state.handle, &maxfct, &mnum, &state.matrix_type, &release_phase, &state.dimension,
            /*a=*/nullptr, /*ia=*/nullptr, /*ja=*/nullptr, /*perm=*/nullptr,
            &number_of_right_hand_sides, state.iparm, &message_level, /*b=*/nullptr,
            /*x=*/nullptr, &release_error);
    if (release_error != 0) {
      status->typed_data()[0] = static_cast<int32_t>(release_error);
      return ffi::Error::Internal(PardisoErrorMessage("reanalyze release", release_error));
    }
  }

  // Pardiso expects a zeroed pt going into a fresh phase 11. The release above
  // (when there was one) frees what pt pointed at but does not clear the array.
  std::memset(state.handle, 0, sizeof(state.handle));
  state.matrix_type = static_cast<MKL_INT>(matrix_type);
  state.dimension = static_cast<MKL_INT>(dimension);
  state.version = VersionCounter().fetch_add(1);
  InitializeIparm(state.iparm, state.matrix_type);
  ApplyOverlay(state.iparm, options_mask.typed_data(), options_values.typed_data());

  MKL_INT phase_value = 11;
  MKL_INT error = 0;

  pardiso(state.handle, &maxfct, &mnum, &state.matrix_type, &phase_value, &state.dimension,
          const_cast<double*>(values.typed_data()), AsMklInt(indptr.typed_data()),
          AsMklInt(indices.typed_data()), /*perm=*/nullptr, &number_of_right_hand_sides,
          state.iparm, &message_level, /*b=*/nullptr, /*x=*/nullptr, &error);

  state.analysis_count += 1;
  TouchLru(handle);
  EvictIfNeeded();
  handle_out->typed_data()[0] = handle;
  version_out->typed_data()[0] = state.version;
  std::memcpy(final_iparm->typed_data(), state.iparm, sizeof(MKL_INT) * 64);
  status->typed_data()[0] = static_cast<int32_t>(error);
  if (error != 0) {
    return ffi::Error::Internal(PardisoErrorMessage("reanalyze", error));
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
                              ffi::ResultBuffer<ffi::S64> version_out,
                              ffi::ResultBuffer<ffi::S32> status,
                              ffi::ResultBuffer<ffi::S32> final_iparm) {
  int64_t handle = handle_in.typed_data()[0];

  std::lock_guard<std::mutex> lock(RegistryMutex());
  bool missing = Registry().find(handle) == Registry().end();
  handle_out->typed_data()[0] = handle;
  version_out->typed_data()[0] = 0;
  // A missing handle lost its analysis to eviction or release. We rebuild it
  // below from the matrix this call carries. Strict mode reports it instead.
  if (missing && StrictCache()) {
    std::memset(final_iparm->typed_data(), 0, sizeof(int32_t) * 64);
    status->typed_data()[0] = -1;
    return ffi::Error::Internal("pardiso factor: handle " + std::to_string(handle) +
                                " was evicted or freed and strict cache mode is on");
  }

  PardisoState& state = Registry()[handle];
  state.matrix_type = static_cast<MKL_INT>(matrix_type);
  state.dimension = static_cast<MKL_INT>(dimension);
  // A numeric factorization always replaces whatever the state held, so it
  // takes a fresh version stamp regardless of the token that came in.
  state.version = VersionCounter().fetch_add(1);
  InitializeIparm(state.iparm, state.matrix_type);
  ApplyOverlay(state.iparm, options_mask.typed_data(), options_values.typed_data());

  if (missing) {
    RebuildCounter().fetch_add(1);
    MKL_INT analyze_error =
        RunAnalysis(state, indptr.typed_data(), indices.typed_data(), values.typed_data());
    if (analyze_error != 0) {
      std::memcpy(final_iparm->typed_data(), state.iparm, sizeof(MKL_INT) * 64);
      status->typed_data()[0] = static_cast<int32_t>(analyze_error);
      return ffi::Error::Internal(PardisoErrorMessage("factor rebuild analyze", analyze_error));
    }
  }

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

  TouchLru(handle);
  EvictIfNeeded();
  version_out->typed_data()[0] = state.version;
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
                             ffi::Buffer<ffi::S64> handle_in, ffi::Buffer<ffi::S64> version_in,
                             ffi::Buffer<ffi::S32> indptr,
                             ffi::Buffer<ffi::S32> indices, ffi::Buffer<ffi::F64> values,
                             ffi::Buffer<ffi::F64> right_hand_side,
                             ffi::Buffer<ffi::S32> options_mask,
                             ffi::Buffer<ffi::S32> options_values,
                             ffi::ResultBuffer<ffi::F64> solution,
                             ffi::ResultBuffer<ffi::S64> handle_out,
                             ffi::ResultBuffer<ffi::S64> version_out,
                             ffi::ResultBuffer<ffi::S32> final_iparm) {
  int64_t handle = handle_in.typed_data()[0];
  int64_t version = version_in.typed_data()[0];

  // The solve echoes the handle and version so a later call that consumes
  // this token is ordered after the solve by data dependency, which is what
  // keeps a reused factorization from being overwritten before this solve
  // reads it. See primitive.solve_stateful.
  handle_out->typed_data()[0] = handle;
  version_out->typed_data()[0] = version;

  std::lock_guard<std::mutex> lock(RegistryMutex());
  bool missing = Registry().find(handle) == Registry().end();
  // A missing handle lost both its analysis and its factorization. A solve
  // needs both, so we rebuild them from this call's matrix before solving.
  // Strict mode reports the miss instead of quietly redoing the work.
  if (missing && StrictCache()) {
    std::memset(solution->typed_data(), 0, solution->element_count() * sizeof(double));
    std::memset(final_iparm->typed_data(), 0, sizeof(int32_t) * 64);
    return ffi::Error::Internal("pardiso solve: handle " + std::to_string(handle) +
                                " was evicted or freed and strict cache mode is on");
  }

  PardisoState& state = Registry()[handle];
  // A present handle whose version has moved past the one this token expects
  // means a later write already replaced the factorization, so solving now
  // would return the wrong matrix's answer. Reject it rather than solve. A
  // missing handle is the self-healing rebuild path below, which adopts the
  // token's version, so this check only fires on a live state.
  if (!missing && state.version != version) {
    std::memset(solution->typed_data(), 0, solution->element_count() * sizeof(double));
    std::memset(final_iparm->typed_data(), 0, sizeof(int32_t) * 64);
    return ffi::Error::Internal("pardiso solve: token version " + std::to_string(version) +
                                " does not match the factorization now held for handle " +
                                std::to_string(handle) + " (version " +
                                std::to_string(state.version) +
                                "); it was replaced by a later factor or reanalyze");
  }
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

  if (missing) {
    RebuildCounter().fetch_add(1);
    // The rebuilt factorization stands in for the one the token named, so it
    // takes the token's version rather than a fresh stamp.
    state.version = version;
    MKL_INT rebuild_error =
        RunAnalysis(state, indptr.typed_data(), indices.typed_data(), values.typed_data());
    if (rebuild_error == 0) {
      rebuild_error =
          RunNumeric(state, indptr.typed_data(), indices.typed_data(), values.typed_data());
    }
    if (rebuild_error != 0) {
      std::memcpy(final_iparm->typed_data(), state.iparm, sizeof(MKL_INT) * 64);
      return ffi::Error::Internal(PardisoErrorMessage("solve rebuild", rebuild_error));
    }
  }

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

  TouchLru(handle);
  EvictIfNeeded();
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
//
// The version is passed through unchanged, not bumped the way a plain factor
// bumps it. This call replaces the numeric factorization and reads it back in
// the same step, so the token that named it stays the one that names the
// result, which is what lets PardisoSolver keep using its stored token after
// a refactor_and_solve without having to restamp it under jit.
ffi::Error PardisoFactorSolveImpl(int64_t matrix_type, int64_t dimension,
                                   int64_t number_of_right_hand_sides, int64_t transpose_mode,
                                   ffi::Buffer<ffi::S64> handle_in, ffi::Buffer<ffi::S64> version_in,
                                   ffi::Buffer<ffi::S32> indptr,
                                   ffi::Buffer<ffi::S32> indices, ffi::Buffer<ffi::F64> values,
                                   ffi::Buffer<ffi::F64> right_hand_side,
                                   ffi::Buffer<ffi::S32> options_mask,
                                   ffi::Buffer<ffi::S32> options_values,
                                   ffi::ResultBuffer<ffi::F64> solution,
                                   ffi::ResultBuffer<ffi::S64> handle_out,
                                   ffi::ResultBuffer<ffi::S64> version_out,
                                   ffi::ResultBuffer<ffi::S32> final_iparm) {
  int64_t handle = handle_in.typed_data()[0];
  int64_t version = version_in.typed_data()[0];

  // Echoed so a later call that consumes this token is ordered after the
  // combined factor-and-solve, the same threading the plain solve does.
  handle_out->typed_data()[0] = handle;
  version_out->typed_data()[0] = version;

  std::lock_guard<std::mutex> lock(RegistryMutex());
  bool missing = Registry().find(handle) == Registry().end();
  // A missing handle lost its analysis. Phase 23 refactors and solves but
  // still needs an analysis to reuse, so we rebuild that from this call's
  // matrix. Strict mode reports the miss instead.
  if (missing && StrictCache()) {
    std::memset(solution->typed_data(), 0, solution->element_count() * sizeof(double));
    std::memset(final_iparm->typed_data(), 0, sizeof(int32_t) * 64);
    return ffi::Error::Internal("pardiso factor_and_solve: handle " + std::to_string(handle) +
                                " was evicted or freed and strict cache mode is on");
  }

  PardisoState& state = Registry()[handle];
  state.matrix_type = static_cast<MKL_INT>(matrix_type);
  state.dimension = static_cast<MKL_INT>(dimension);
  // Keep the state's version equal to the token's, so a later solve on this
  // same token still matches.
  state.version = version;
  InitializeIparm(state.iparm, state.matrix_type);
  ApplyOverlay(state.iparm, options_mask.typed_data(), options_values.typed_data());
  // Set unconditionally, matching PardisoSolveImpl, so a later call without
  // transpose is not left with a stale value from an earlier one. Applied
  // after ApplyOverlay for the same reason given there.
  state.iparm[11] = static_cast<MKL_INT>(transpose_mode);

  if (missing) {
    RebuildCounter().fetch_add(1);
    MKL_INT analyze_error =
        RunAnalysis(state, indptr.typed_data(), indices.typed_data(), values.typed_data());
    if (analyze_error != 0) {
      std::memcpy(final_iparm->typed_data(), state.iparm, sizeof(MKL_INT) * 64);
      return ffi::Error::Internal(
          PardisoErrorMessage("factor_and_solve rebuild analyze", analyze_error));
    }
  }

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

  TouchLru(handle);
  EvictIfNeeded();
  std::memcpy(final_iparm->typed_data(), state.iparm, sizeof(MKL_INT) * 64);
  if (error != 0) {
    return ffi::Error::Internal(PardisoErrorMessage("factor_and_solve", error));
  }
  return ffi::Error::Success();
}

// Frees the native memory for handle (phase -1) and drops it from the
// registry. A handle that is not present is treated as already released.
// ordering is an unused operand whose only job is to give XLA a data
// dependency, so a release inside a jit trace runs after the solves it must
// follow. See primitive.release.
ffi::Error PardisoReleaseImpl(ffi::Buffer<ffi::S64> handle_in, ffi::Buffer<ffi::S32> ordering,
                              ffi::ResultBuffer<ffi::S32> status) {
  (void)ordering;
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
  LruList().remove(handle);
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
                                   .Ret<ffi::Buffer<ffi::S64>>()  // version
                                   .Ret<ffi::Buffer<ffi::S32>>()  // status
                                   .Ret<ffi::Buffer<ffi::S32>>()  // final_iparm
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kPardisoReanalyzeHandler, PardisoReanalyzeImpl,
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
                                   .Ret<ffi::Buffer<ffi::S64>>()  // version
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
                                   .Ret<ffi::Buffer<ffi::S64>>()  // version
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
                                   .Arg<ffi::Buffer<ffi::S64>>()  // version
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indptr
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indices
                                   .Arg<ffi::Buffer<ffi::F64>>()  // values
                                   .Arg<ffi::Buffer<ffi::F64>>()  // right_hand_side
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_mask
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_values
                                   .Ret<ffi::Buffer<ffi::F64>>()  // solution
                                   .Ret<ffi::Buffer<ffi::S64>>()  // handle
                                   .Ret<ffi::Buffer<ffi::S64>>()  // version
                                   .Ret<ffi::Buffer<ffi::S32>>()  // final_iparm
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kPardisoFactorSolveHandler, PardisoFactorSolveImpl,
                               ffi::Ffi::Bind()
                                   .Attr<int64_t>("matrix_type")
                                   .Attr<int64_t>("dimension")
                                   .Attr<int64_t>("number_of_right_hand_sides")
                                   .Attr<int64_t>("transpose_mode")
                                   .Arg<ffi::Buffer<ffi::S64>>()  // handle
                                   .Arg<ffi::Buffer<ffi::S64>>()  // version
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indptr
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indices
                                   .Arg<ffi::Buffer<ffi::F64>>()  // values
                                   .Arg<ffi::Buffer<ffi::F64>>()  // right_hand_side
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_mask
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_values
                                   .Ret<ffi::Buffer<ffi::F64>>()  // solution
                                   .Ret<ffi::Buffer<ffi::S64>>()  // handle
                                   .Ret<ffi::Buffer<ffi::S64>>()  // version
                                   .Ret<ffi::Buffer<ffi::S32>>()  // final_iparm
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kPardisoReleaseHandler, PardisoReleaseImpl,
                               ffi::Ffi::Bind()
                                   .Arg<ffi::Buffer<ffi::S64>>()  // handle
                                   .Arg<ffi::Buffer<ffi::S32>>()  // ordering (unused)
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

extern "C" void* pardiso_reanalyze_handler_address() {
  return reinterpret_cast<void*>(pardiso_mkl_jax::kPardisoReanalyzeHandler);
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
