// XLA FFI handlers wrapping the oneMKL Pardiso direct sparse solver.
//
// Pardiso keeps its factorization in an opaque native handle ("pt") that
// must persist across calls to be reused. XLA cannot hold that object in a
// buffer, so we keep a process-global cache of them. A handle is a content
// hash of the matrix it names (matrix type, dimension, sparsity pattern,
// values, options), never a raw pointer, so whatever the cache returns for a
// key was built from content whose hash is that key. A handle names a matrix,
// not a mutable slot.
//
// This makes the cache a memoization layer, not part of correctness. A call
// that arrives with a key no longer resident rebuilds the factorization from
// the arrays the call carries, so a stale handle is always safe. A refactor
// re-keys to the new values instead of overwriting the old key, so a stale
// alias of the old handle still names the original matrix and rebuilds it. At
// worst a call redoes the whole analyze then factor chain. Because a handle is
// a pure function of the inputs, analyze and factor carry no side effect, and
// XLA may eliminate, reorder, or merge them.
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
#include <map>
#include <mutex>
#include <string>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace pardiso_mkl_jax {
namespace {

// Native state for one cache entry: the opaque Pardiso handle, its parameter
// array, the matrix type and dimension it was built for, and a count of the
// analysis (phase 11) runs the test suite checks.
struct PardisoState {
  void* handle[64] = {};
  MKL_INT iparm[64] = {};
  MKL_INT matrix_type = 0;
  MKL_INT dimension = 0;
  long analysis_count = 0;
};

// Why a call rebuilt a factorization instead of using a resident one. Mirrored
// by RebuildReason in primitive.py, and by the same enum in splineax-klujax.
// NONE is a cache hit. EVICTED, FREED, and SUPERSEDED name how the key was
// retired. UNKNOWN means the key was never resident or its tombstone aged out.
// DTYPE and STALE are reserved so the numbering matches the other library.
enum class RebuildReason : int32_t {
  NONE = 0,
  EVICTED = 1,
  FREED = 2,
  SUPERSEDED = 3,
  DTYPE = 4,
  UNKNOWN = 5,
  STALE = 6,
};
constexpr int kNumRebuildReasons = 7;

// A 128-bit content hash split into a primary key and a second check lane. The
// primary key flows through JAX as the handle. The check lane is compared on
// every lookup so a primary-key collision never hands back the wrong matrix.
struct ContentKey {
  uint64_t key = 0;
  uint64_t check = 0;
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

// Two-lane FNV-1a-style mixing. Not cryptographic. It only needs to spread
// content across 128 bits well enough that distinct matrices collide with
// negligible probability.
inline void HashFold(uint64_t& a, uint64_t& b, const void* data, size_t bytes) {
  const unsigned char* p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < bytes; ++i) {
    a = (a ^ p[i]) * 0x100000001b3ULL;
    b = (b + p[i]) * 0x9E3779B97F4A7C15ULL;
    b ^= b >> 29;
  }
}

// Content key of a CSR matrix and the options it will be built with. domain
// separates the analyze keyspace ('a') from the factor keyspace ('f'), so the
// same matrix gets a different key at each stage. The values and the options
// overlay are both folded in: Pardiso's analysis reads the values when
// scaling or weighted matching is on, and different options give a different
// factorization.
ContentKey HashCsr(char domain, MKL_INT matrix_type, MKL_INT dimension, const int32_t* indptr,
                   const int32_t* indices, const double* values, const int32_t* overlay_mask,
                   const int32_t* overlay_values) {
  int n = static_cast<int>(dimension);
  int nnz = indptr[n];
  uint64_t a = 0xcbf29ce484222325ULL;
  uint64_t b = 0x27d4eb2f165667c5ULL;
  unsigned char tag = static_cast<unsigned char>(domain);
  HashFold(a, b, &tag, sizeof(tag));
  HashFold(a, b, &matrix_type, sizeof(matrix_type));
  HashFold(a, b, &dimension, sizeof(dimension));
  HashFold(a, b, &nnz, sizeof(nnz));
  HashFold(a, b, indptr, sizeof(int32_t) * static_cast<size_t>(n + 1));
  HashFold(a, b, indices, sizeof(int32_t) * static_cast<size_t>(nnz));
  HashFold(a, b, values, sizeof(double) * static_cast<size_t>(nnz));
  HashFold(a, b, overlay_mask, sizeof(int32_t) * 64);
  HashFold(a, b, overlay_values, sizeof(int32_t) * 64);
  ContentKey ck;
  ck.key = a ^ (b << 1);
  ck.check = b ^ (a >> 1);
  // Zero is the null-handle sentinel, so a real key is never zero.
  if (ck.key == 0) {
    ck.key = 1;
  }
  return ck;
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

// One cache entry: a native Pardiso state and how far it has been taken. An
// analyzed entry has run phase 11. A factored entry has also run phase 22, so
// its numeric factorization is valid for a solve. The content key and check
// lane it is filed under travel with it across a re-key.
struct CacheEntry {
  PardisoState state;
  bool factored = false;
  uint64_t key = 0;
  uint64_t check = 0;
};

// Cache bookkeeping ==================================================================
//
// The cache is bounded so that forgetting to release a handle leaks a limited
// amount of memory rather than growing without end. Entries are kept in an LRU
// list, and the least recently used one is evicted once the map grows past the
// cache size. A retired key leaves a bounded tombstone recording why it went
// away, so a later rebuild on it can report EVICTED, FREED, or SUPERSEDED.

std::mutex& RegistryMutex() {
  static std::mutex mutex;
  return mutex;
}

// Must only be accessed while holding RegistryMutex. Keyed by content hash.
std::map<uint64_t, CacheEntry>& Registry() {
  static std::map<uint64_t, CacheEntry> registry;
  return registry;
}

// Order of live keys, most recently used at the front. Guarded by
// RegistryMutex, like Registry itself.
std::list<uint64_t>& LruList() {
  static std::list<uint64_t> lru;
  return lru;
}

// Why each retired key went away, bounded like the live cache. Guarded by
// RegistryMutex.
std::map<uint64_t, RebuildReason>& Tombstones() {
  static std::map<uint64_t, RebuildReason> tombstones;
  return tombstones;
}

std::list<uint64_t>& TombstoneLru() {
  static std::list<uint64_t> lru;
  return lru;
}

// Counts how often a call had to rebuild a missing factorization. A rising
// count means the cache is too small for the working set, so this is what the
// rebuild_count() diagnostic reports.
std::atomic<long>& RebuildCounter() {
  static std::atomic<long> counter{0};
  return counter;
}

// Per-reason rebuild totals, indexed by RebuildReason. Surfaced by
// rebuild_stats() so a rising SUPERSEDED points at stale-alias use and a
// rising EVICTED at cache pressure.
std::atomic<long>* ReasonCounts() {
  static std::atomic<long> counts[kNumRebuildReasons];
  return counts;
}

// Cache size, from PARDISO_MKL_JAX_FACTOR_CACHE, defaulting to 8 live entries.
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

void TouchLru(uint64_t key) {
  LruList().remove(key);
  LruList().push_front(key);
}

// Record why a retired key went away, keeping the tombstone map bounded to the
// cache capacity. Caller holds RegistryMutex.
void RecordTombstone(uint64_t key, RebuildReason why) {
  if (key == 0) {
    return;
  }
  if (Tombstones().find(key) == Tombstones().end()) {
    TombstoneLru().push_front(key);
  } else {
    TombstoneLru().remove(key);
    TombstoneLru().push_front(key);
  }
  Tombstones()[key] = why;
  size_t capacity = CacheCapacity();
  while (Tombstones().size() > capacity && !TombstoneLru().empty()) {
    uint64_t victim = TombstoneLru().back();
    TombstoneLru().pop_back();
    Tombstones().erase(victim);
  }
}

// Why a key is no longer resident. UNKNOWN when there is no tombstone, which
// means it was never resident or its tombstone itself aged out. Caller holds
// RegistryMutex.
RebuildReason TombstoneReason(uint64_t key) {
  auto iterator = Tombstones().find(key);
  return iterator == Tombstones().end() ? RebuildReason::UNKNOWN : iterator->second;
}

// A live key is no longer a tombstone. Caller holds RegistryMutex.
void ClearTombstone(uint64_t key) {
  if (Tombstones().erase(key) != 0) {
    TombstoneLru().remove(key);
  }
}

// Free a state's native factorization (phase -1). Errors are ignored: this
// runs during eviction or a re-key, where there is no caller to report to and
// the memory is being dropped regardless.
void FreeState(PardisoState& state) {
  MKL_INT maxfct = 1, mnum = 1, phase = -1, nrhs = 0, message_level = 0, error = 0;
  pardiso(state.handle, &maxfct, &mnum, &state.matrix_type, &phase, &state.dimension,
          /*a=*/nullptr, /*ia=*/nullptr, /*ja=*/nullptr, /*perm=*/nullptr, &nrhs, state.iparm,
          &message_level, /*b=*/nullptr, /*x=*/nullptr, &error);
}

// Drop least-recently-used entries until the map is back within the cache
// size, freeing each one and leaving an EVICTED tombstone. The entry in use by
// the current call is at the front, so it is never the victim as long as the
// capacity is at least one. Caller holds RegistryMutex.
void EvictIfNeeded() {
  const size_t capacity = CacheCapacity();
  while (Registry().size() > capacity && !LruList().empty()) {
    uint64_t victim = LruList().back();
    LruList().pop_back();
    auto iterator = Registry().find(victim);
    if (iterator == Registry().end()) {
      continue;
    }
    FreeState(iterator->second.state);
    Registry().erase(iterator);
    RecordTombstone(victim, RebuildReason::EVICTED);
  }
}

void NoteRebuild(RebuildReason why) {
  RebuildCounter().fetch_add(1);
  int index = static_cast<int>(why);
  if (index >= 0 && index < kNumRebuildReasons) {
    ReasonCounts()[index].fetch_add(1);
  }
}

// Run the symbolic analysis (phase 11) into state, from the given matrix. The
// caller has already set state's matrix type, dimension, and iparm. Any pt the
// state still holds is freed first, then pt is zeroed, which Pardiso expects
// for a fresh analysis.
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

// Run the numeric factorization (phase 22) into state, reusing the analysis
// already in its pt. The caller guarantees the pattern matches that analysis.
MKL_INT RunNumeric(PardisoState& state, const int32_t* indptr, const int32_t* indices,
                   const double* values) {
  MKL_INT maxfct = 1, mnum = 1, phase = 22, nrhs = 0, message_level = 0, error = 0;
  pardiso(state.handle, &maxfct, &mnum, &state.matrix_type, &phase, &state.dimension,
          const_cast<double*>(values), AsMklInt(indptr), AsMklInt(indices), /*perm=*/nullptr,
          &nrhs, state.iparm, &message_level, /*b=*/nullptr, /*x=*/nullptr, &error);
  return error;
}

// Set an entry's matrix type, dimension, and iparm for a call. The transpose
// entry (iparm[11]) is left at its default here and set separately by solve.
void PrepareIparm(PardisoState& state, MKL_INT matrix_type, MKL_INT dimension,
                  const int32_t* overlay_mask, const int32_t* overlay_values) {
  state.matrix_type = matrix_type;
  state.dimension = dimension;
  InitializeIparm(state.iparm, state.matrix_type);
  ApplyOverlay(state.iparm, overlay_mask, overlay_values);
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

extern "C" long pardiso_analysis_count(unsigned long long handle) {
  std::lock_guard<std::mutex> lock(RegistryMutex());
  auto iterator = Registry().find(static_cast<uint64_t>(handle));
  return iterator == Registry().end() ? 0 : iterator->second.state.analysis_count;
}

extern "C" void pardiso_reset_analysis_count(unsigned long long handle) {
  std::lock_guard<std::mutex> lock(RegistryMutex());
  auto iterator = Registry().find(static_cast<uint64_t>(handle));
  if (iterator != Registry().end()) {
    iterator->second.state.analysis_count = 0;
  }
}

// Total number of rebuilds since load (or since the last reset). Process-wide,
// not per-handle, since a rebuild happens exactly because the handle is gone.
extern "C" long pardiso_rebuild_count() {
  return RebuildCounter().load();
}

extern "C" void pardiso_reset_rebuild_count() {
  RebuildCounter().store(0);
  for (int i = 0; i < kNumRebuildReasons; ++i) {
    ReasonCounts()[i].store(0);
  }
}

// Per-reason rebuild total, for rebuild_stats(). An out-of-range reason
// returns 0.
extern "C" long pardiso_rebuild_reason_count(int reason) {
  if (reason < 0 || reason >= kNumRebuildReasons) {
    return 0;
  }
  return ReasonCounts()[reason].load();
}

namespace {

// Analyze (phase 11). Content-addressed and pure: the returned handle is the
// hash of the matrix and options, so two identical analyze calls dedup to one
// entry. On a hit the resident analysis is reused. On a miss a fresh entry is
// built and filed under its key. Every later stage takes this handle as data,
// which is what orders the analyze -> factor -> solve chain and lets it run
// inside a jit trace.
ffi::Error PardisoAnalyzeImpl(int64_t matrix_type, int64_t dimension,
                               ffi::Buffer<ffi::S32> indptr, ffi::Buffer<ffi::S32> indices,
                               ffi::Buffer<ffi::F64> values, ffi::Buffer<ffi::S32> options_mask,
                               ffi::Buffer<ffi::S32> options_values,
                               ffi::ResultBuffer<ffi::U64> handle_out,
                               ffi::ResultBuffer<ffi::S32> status,
                               ffi::ResultBuffer<ffi::S32> final_iparm) {
  MKL_INT mtype = static_cast<MKL_INT>(matrix_type);
  MKL_INT dim = static_cast<MKL_INT>(dimension);
  ContentKey ck = HashCsr('a', mtype, dim, indptr.typed_data(), indices.typed_data(),
                          values.typed_data(), options_mask.typed_data(),
                          options_values.typed_data());

  std::lock_guard<std::mutex> lock(RegistryMutex());
  handle_out->typed_data()[0] = ck.key;

  auto iterator = Registry().find(ck.key);
  if (iterator != Registry().end() && iterator->second.check == ck.check) {
    // Dedup: an analysis of this exact matrix and options is already resident.
    TouchLru(ck.key);
    std::memcpy(final_iparm->typed_data(), iterator->second.state.iparm, sizeof(MKL_INT) * 64);
    status->typed_data()[0] = 0;
    return ffi::Error::Success();
  }

  CacheEntry entry;
  entry.key = ck.key;
  entry.check = ck.check;
  PrepareIparm(entry.state, mtype, dim, options_mask.typed_data(), options_values.typed_data());
  MKL_INT error = RunAnalysis(entry.state, indptr.typed_data(), indices.typed_data(),
                              values.typed_data());
  std::memcpy(final_iparm->typed_data(), entry.state.iparm, sizeof(MKL_INT) * 64);
  status->typed_data()[0] = static_cast<int32_t>(error);
  if (error != 0) {
    return ffi::Error::Internal(PardisoErrorMessage("analyze", error));
  }
  Registry()[ck.key] = std::move(entry);
  ClearTombstone(ck.key);
  TouchLru(ck.key);
  EvictIfNeeded();
  return ffi::Error::Success();
}

// Re-analyze (phase 11) for a possibly new matrix or options, retiring the old
// handle. Content-addressed like analyze, so re-analyzing the same inputs is a
// dedup, and the numeric factorization is gone afterwards, so factor must run
// again before any solve. A missing input handle is a rebuild, reported as an
// error under strict mode.
ffi::Error PardisoReanalyzeImpl(int64_t matrix_type, int64_t dimension,
                                 ffi::Buffer<ffi::U64> handle_in, ffi::Buffer<ffi::S32> indptr,
                                 ffi::Buffer<ffi::S32> indices, ffi::Buffer<ffi::F64> values,
                                 ffi::Buffer<ffi::S32> options_mask,
                                 ffi::Buffer<ffi::S32> options_values,
                                 ffi::ResultBuffer<ffi::U64> handle_out,
                                 ffi::ResultBuffer<ffi::S32> status,
                                 ffi::ResultBuffer<ffi::S32> final_iparm) {
  uint64_t old_key = handle_in.typed_data()[0];
  MKL_INT mtype = static_cast<MKL_INT>(matrix_type);
  MKL_INT dim = static_cast<MKL_INT>(dimension);
  ContentKey ck = HashCsr('a', mtype, dim, indptr.typed_data(), indices.typed_data(),
                          values.typed_data(), options_mask.typed_data(),
                          options_values.typed_data());

  std::lock_guard<std::mutex> lock(RegistryMutex());
  handle_out->typed_data()[0] = ck.key;
  std::memset(final_iparm->typed_data(), 0, sizeof(int32_t) * 64);

  auto old_iterator = Registry().find(old_key);
  bool old_missing = old_iterator == Registry().end();
  if (old_missing && StrictCache()) {
    status->typed_data()[0] = -1;
    return ffi::Error::Internal("pardiso reanalyze: handle was evicted or freed and strict "
                                "cache mode is on");
  }

  // The old handle is retired either way. Free its state and leave a
  // SUPERSEDED tombstone, unless the new key is the same matrix (a dedup).
  if (!old_missing && old_key != ck.key) {
    FreeState(old_iterator->second.state);
    Registry().erase(old_iterator);
    LruList().remove(old_key);
    RecordTombstone(old_key, RebuildReason::SUPERSEDED);
  }
  if (old_missing) {
    NoteRebuild(TombstoneReason(old_key));
  }

  auto existing = Registry().find(ck.key);
  if (existing != Registry().end() && existing->second.check == ck.check) {
    TouchLru(ck.key);
    std::memcpy(final_iparm->typed_data(), existing->second.state.iparm, sizeof(MKL_INT) * 64);
    status->typed_data()[0] = 0;
    return ffi::Error::Success();
  }

  CacheEntry entry;
  entry.key = ck.key;
  entry.check = ck.check;
  PrepareIparm(entry.state, mtype, dim, options_mask.typed_data(), options_values.typed_data());
  MKL_INT error = RunAnalysis(entry.state, indptr.typed_data(), indices.typed_data(),
                              values.typed_data());
  std::memcpy(final_iparm->typed_data(), entry.state.iparm, sizeof(MKL_INT) * 64);
  status->typed_data()[0] = static_cast<int32_t>(error);
  if (error != 0) {
    return ffi::Error::Internal(PardisoErrorMessage("reanalyze", error));
  }
  Registry()[ck.key] = std::move(entry);
  ClearTombstone(ck.key);
  TouchLru(ck.key);
  EvictIfNeeded();
  return ffi::Error::Success();
}

// Numeric factorization (phase 22). Returns a factor-domain handle naming the
// factorization of these values, distinct from the analyze handle it takes.
// When the input analysis is resident, its pt is reused (the analysis is not
// redone) and re-keyed to the factor key, so a stale alias of the analysis
// handle rebuilds. When it is gone, the whole analyze then factor chain is
// rebuilt from the carried matrix.
ffi::Error PardisoFactorImpl(int64_t matrix_type, int64_t dimension,
                              ffi::Buffer<ffi::U64> handle_in, ffi::Buffer<ffi::S32> indptr,
                              ffi::Buffer<ffi::S32> indices, ffi::Buffer<ffi::F64> values,
                              ffi::Buffer<ffi::S32> options_mask,
                              ffi::Buffer<ffi::S32> options_values,
                              ffi::ResultBuffer<ffi::U64> handle_out,
                              ffi::ResultBuffer<ffi::S32> status,
                              ffi::ResultBuffer<ffi::S32> final_iparm) {
  uint64_t in_key = handle_in.typed_data()[0];
  MKL_INT mtype = static_cast<MKL_INT>(matrix_type);
  MKL_INT dim = static_cast<MKL_INT>(dimension);
  ContentKey ck = HashCsr('f', mtype, dim, indptr.typed_data(), indices.typed_data(),
                          values.typed_data(), options_mask.typed_data(),
                          options_values.typed_data());

  std::lock_guard<std::mutex> lock(RegistryMutex());
  handle_out->typed_data()[0] = ck.key;

  auto factored = Registry().find(ck.key);
  if (factored != Registry().end() && factored->second.check == ck.check &&
      factored->second.factored) {
    // Dedup: this exact factorization is already resident.
    TouchLru(ck.key);
    std::memcpy(final_iparm->typed_data(), factored->second.state.iparm, sizeof(MKL_INT) * 64);
    status->typed_data()[0] = 0;
    return ffi::Error::Success();
  }

  CacheEntry entry;
  auto source = Registry().find(in_key);
  bool reuse_analysis = source != Registry().end();
  if (reuse_analysis) {
    // Take the analyzed pt and re-key it. Its options may differ from this
    // call's, so reset iparm before the numeric phase.
    entry = std::move(source->second);
    Registry().erase(source);
    if (in_key != ck.key) {
      LruList().remove(in_key);
      RecordTombstone(in_key, RebuildReason::SUPERSEDED);
    }
    PrepareIparm(entry.state, mtype, dim, options_mask.typed_data(), options_values.typed_data());
  } else {
    if (StrictCache()) {
      std::memset(final_iparm->typed_data(), 0, sizeof(int32_t) * 64);
      status->typed_data()[0] = -1;
      return ffi::Error::Internal("pardiso factor: analysis handle was evicted or freed and "
                                  "strict cache mode is on");
    }
    NoteRebuild(TombstoneReason(in_key));
    PrepareIparm(entry.state, mtype, dim, options_mask.typed_data(), options_values.typed_data());
    MKL_INT analyze_error = RunAnalysis(entry.state, indptr.typed_data(), indices.typed_data(),
                                        values.typed_data());
    if (analyze_error != 0) {
      std::memcpy(final_iparm->typed_data(), entry.state.iparm, sizeof(MKL_INT) * 64);
      status->typed_data()[0] = static_cast<int32_t>(analyze_error);
      return ffi::Error::Internal(PardisoErrorMessage("factor rebuild analyze", analyze_error));
    }
  }

  MKL_INT error = RunNumeric(entry.state, indptr.typed_data(), indices.typed_data(),
                             values.typed_data());
  entry.factored = true;
  entry.key = ck.key;
  entry.check = ck.check;
  std::memcpy(final_iparm->typed_data(), entry.state.iparm, sizeof(MKL_INT) * 64);
  status->typed_data()[0] = static_cast<int32_t>(error);
  if (error != 0) {
    return ffi::Error::Internal(PardisoErrorMessage("factor", error));
  }
  Registry()[ck.key] = std::move(entry);
  ClearTombstone(ck.key);
  TouchLru(ck.key);
  EvictIfNeeded();
  return ffi::Error::Success();
}

// Solve (phase 33) against the factorization the handle names. A hit reuses
// the resident numeric factorization, which is safe because the factor key
// encodes the exact values it was built from. A miss rebuilds the analysis and
// numeric factorization from the carried matrix, files them under the handle,
// and reports why through rebuild_reason. transpose_mode is iparm[11]: 0 solves
// Ax = b, 2 solves A^T x = b, reusing the same factorization either way.
ffi::Error PardisoSolveImpl(int64_t matrix_type, int64_t dimension,
                             int64_t number_of_right_hand_sides, int64_t transpose_mode,
                             ffi::Buffer<ffi::U64> handle_in, ffi::Buffer<ffi::S32> indptr,
                             ffi::Buffer<ffi::S32> indices, ffi::Buffer<ffi::F64> values,
                             ffi::Buffer<ffi::F64> right_hand_side,
                             ffi::Buffer<ffi::S32> options_mask,
                             ffi::Buffer<ffi::S32> options_values,
                             ffi::ResultBuffer<ffi::F64> solution,
                             ffi::ResultBuffer<ffi::S32> final_iparm,
                             ffi::ResultBuffer<ffi::S32> rebuild_reason) {
  uint64_t key = handle_in.typed_data()[0];
  MKL_INT mtype = static_cast<MKL_INT>(matrix_type);
  MKL_INT dim = static_cast<MKL_INT>(dimension);

  std::lock_guard<std::mutex> lock(RegistryMutex());
  rebuild_reason->typed_data()[0] = static_cast<int32_t>(RebuildReason::NONE);

  auto iterator = Registry().find(key);
  bool hit = iterator != Registry().end() && iterator->second.factored;
  if (!hit && StrictCache()) {
    std::memset(solution->typed_data(), 0, solution->element_count() * sizeof(double));
    std::memset(final_iparm->typed_data(), 0, sizeof(int32_t) * 64);
    return ffi::Error::Internal("pardiso solve: handle was evicted or freed and strict cache "
                                "mode is on");
  }

  CacheEntry* entry = nullptr;
  if (hit) {
    entry = &iterator->second;
  } else {
    // Rebuild the analysis and numeric factorization from the carried matrix,
    // then file the entry under the handle the caller used so a later solve
    // with the same handle hits.
    RebuildReason reason = TombstoneReason(key);
    rebuild_reason->typed_data()[0] = static_cast<int32_t>(reason);
    NoteRebuild(reason);
    if (iterator != Registry().end()) {
      FreeState(iterator->second.state);
      Registry().erase(iterator);
      LruList().remove(key);
    }
    CacheEntry rebuilt;
    rebuilt.key = key;
    PrepareIparm(rebuilt.state, mtype, dim, options_mask.typed_data(),
                 options_values.typed_data());
    MKL_INT rebuild_error = RunAnalysis(rebuilt.state, indptr.typed_data(),
                                        indices.typed_data(), values.typed_data());
    if (rebuild_error == 0) {
      rebuild_error = RunNumeric(rebuilt.state, indptr.typed_data(), indices.typed_data(),
                                 values.typed_data());
    }
    if (rebuild_error != 0) {
      std::memcpy(final_iparm->typed_data(), rebuilt.state.iparm, sizeof(MKL_INT) * 64);
      return ffi::Error::Internal(PardisoErrorMessage("solve rebuild", rebuild_error));
    }
    rebuilt.factored = true;
    Registry()[key] = std::move(rebuilt);
    ClearTombstone(key);
    entry = &Registry()[key];
  }

  // Set the transpose entry unconditionally so a later solve on the same entry
  // without transpose is not left with a stale value. canonicalize_overlay in
  // iparm.py keeps a caller-supplied overlay off index 11, so nothing conflicts.
  entry->state.iparm[11] = static_cast<MKL_INT>(transpose_mode);

  MKL_INT maxfct = 1, mnum = 1, phase_value = 33;
  MKL_INT nrhs = static_cast<MKL_INT>(number_of_right_hand_sides);
  MKL_INT message_level = 0, error = 0;
  pardiso(entry->state.handle, &maxfct, &mnum, &entry->state.matrix_type, &phase_value,
          &entry->state.dimension, const_cast<double*>(values.typed_data()),
          AsMklInt(indptr.typed_data()), AsMklInt(indices.typed_data()), /*perm=*/nullptr, &nrhs,
          entry->state.iparm, &message_level,
          const_cast<double*>(right_hand_side.typed_data()), solution->typed_data(), &error);

  TouchLru(key);
  EvictIfNeeded();
  std::memcpy(final_iparm->typed_data(), entry->state.iparm, sizeof(MKL_INT) * 64);
  if (error != 0) {
    return ffi::Error::Internal(PardisoErrorMessage("solve", error));
  }
  return ffi::Error::Success();
}

// Numeric factorization and solve in one call (combined phase 23), reusing the
// analysis the handle names. Unlike factor, this does not re-key: it recomputes
// the numeric factorization in place and never returns a token, so the analysis
// handle stays usable across many calls (the fast path for a jitted loop). A
// miss rebuilds the analysis from the carried matrix and reports why.
ffi::Error PardisoFactorSolveImpl(int64_t matrix_type, int64_t dimension,
                                   int64_t number_of_right_hand_sides, int64_t transpose_mode,
                                   ffi::Buffer<ffi::U64> handle_in, ffi::Buffer<ffi::S32> indptr,
                                   ffi::Buffer<ffi::S32> indices, ffi::Buffer<ffi::F64> values,
                                   ffi::Buffer<ffi::F64> right_hand_side,
                                   ffi::Buffer<ffi::S32> options_mask,
                                   ffi::Buffer<ffi::S32> options_values,
                                   ffi::ResultBuffer<ffi::F64> solution,
                                   ffi::ResultBuffer<ffi::S32> final_iparm,
                                   ffi::ResultBuffer<ffi::S32> rebuild_reason) {
  uint64_t key = handle_in.typed_data()[0];
  MKL_INT mtype = static_cast<MKL_INT>(matrix_type);
  MKL_INT dim = static_cast<MKL_INT>(dimension);

  std::lock_guard<std::mutex> lock(RegistryMutex());
  rebuild_reason->typed_data()[0] = static_cast<int32_t>(RebuildReason::NONE);

  auto iterator = Registry().find(key);
  bool has_analysis = iterator != Registry().end();
  if (!has_analysis && StrictCache()) {
    std::memset(solution->typed_data(), 0, solution->element_count() * sizeof(double));
    std::memset(final_iparm->typed_data(), 0, sizeof(int32_t) * 64);
    return ffi::Error::Internal("pardiso factor_and_solve: handle was evicted or freed and "
                                "strict cache mode is on");
  }

  CacheEntry* entry = nullptr;
  if (has_analysis) {
    entry = &iterator->second;
    PrepareIparm(entry->state, mtype, dim, options_mask.typed_data(),
                 options_values.typed_data());
  } else {
    RebuildReason reason = TombstoneReason(key);
    rebuild_reason->typed_data()[0] = static_cast<int32_t>(reason);
    NoteRebuild(reason);
    CacheEntry rebuilt;
    rebuilt.key = key;
    PrepareIparm(rebuilt.state, mtype, dim, options_mask.typed_data(),
                 options_values.typed_data());
    MKL_INT analyze_error = RunAnalysis(rebuilt.state, indptr.typed_data(),
                                        indices.typed_data(), values.typed_data());
    if (analyze_error != 0) {
      std::memcpy(final_iparm->typed_data(), rebuilt.state.iparm, sizeof(MKL_INT) * 64);
      return ffi::Error::Internal(
          PardisoErrorMessage("factor_and_solve rebuild analyze", analyze_error));
    }
    Registry()[key] = std::move(rebuilt);
    ClearTombstone(key);
    entry = &Registry()[key];
  }

  entry->state.iparm[11] = static_cast<MKL_INT>(transpose_mode);

  MKL_INT maxfct = 1, mnum = 1, phase_value = 23;
  MKL_INT nrhs = static_cast<MKL_INT>(number_of_right_hand_sides);
  MKL_INT message_level = 0, error = 0;
  pardiso(entry->state.handle, &maxfct, &mnum, &entry->state.matrix_type, &phase_value,
          &entry->state.dimension, const_cast<double*>(values.typed_data()),
          AsMklInt(indptr.typed_data()), AsMklInt(indices.typed_data()), /*perm=*/nullptr, &nrhs,
          entry->state.iparm, &message_level,
          const_cast<double*>(right_hand_side.typed_data()), solution->typed_data(), &error);
  entry->factored = true;

  TouchLru(key);
  EvictIfNeeded();
  std::memcpy(final_iparm->typed_data(), entry->state.iparm, sizeof(MKL_INT) * 64);
  if (error != 0) {
    return ffi::Error::Internal(PardisoErrorMessage("factor_and_solve", error));
  }
  return ffi::Error::Success();
}

// Frees the native memory for a handle (phase -1) and drops it from the cache,
// leaving a FREED tombstone so a later use reports why it rebuilt. A handle
// that is not present is treated as already released. ordering is an unused
// operand whose only job is to give XLA a data dependency, so a release inside
// a jit trace runs after the solves it must follow. See primitive.release.
ffi::Error PardisoReleaseImpl(ffi::Buffer<ffi::U64> handle_in, ffi::Buffer<ffi::S32> ordering,
                              ffi::ResultBuffer<ffi::S32> status) {
  (void)ordering;
  uint64_t key = handle_in.typed_data()[0];

  std::lock_guard<std::mutex> lock(RegistryMutex());
  auto iterator = Registry().find(key);
  if (iterator == Registry().end()) {
    status->typed_data()[0] = 0;
    return ffi::Error::Success();
  }

  MKL_INT maxfct = 1, mnum = 1, phase_value = -1, nrhs = 0, message_level = 0, error = 0;
  PardisoState& state = iterator->second.state;
  pardiso(state.handle, &maxfct, &mnum, &state.matrix_type, &phase_value, &state.dimension,
          /*a=*/nullptr, /*ia=*/nullptr, /*ja=*/nullptr, /*perm=*/nullptr, &nrhs, state.iparm,
          &message_level, /*b=*/nullptr, /*x=*/nullptr, &error);

  Registry().erase(iterator);
  LruList().remove(key);
  RecordTombstone(key, RebuildReason::FREED);
  status->typed_data()[0] = static_cast<int32_t>(error);
  if (error != 0) {
    return ffi::Error::Internal(PardisoErrorMessage("release", error));
  }
  return ffi::Error::Success();
}

// Stateless one-shot solve: analyze, factor, and solve in a single call
// (combined phase 13) against a local handle, released again before
// returning. Used by the functional solve() entry point, which never reuses
// a factorization and so never touches the cache.
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

  MKL_INT maxfct = 1, mnum = 1;
  MKL_INT n = static_cast<MKL_INT>(dimension);
  MKL_INT nrhs = static_cast<MKL_INT>(number_of_right_hand_sides);
  MKL_INT message_level = 0, solve_phase = 13, error = 0;

  pardiso(handle, &maxfct, &mnum, &mtype_value, &solve_phase, &n,
          const_cast<double*>(values.typed_data()), AsMklInt(indptr.typed_data()),
          AsMklInt(indices.typed_data()), /*perm=*/nullptr, &nrhs, iparm, &message_level,
          const_cast<double*>(right_hand_side.typed_data()), solution->typed_data(), &error);

  // Captured right after the solving call and before the release call below,
  // which reuses the same iparm array and could otherwise overwrite these
  // diagnostics with whatever the release phase leaves behind.
  std::memcpy(final_iparm->typed_data(), iparm, sizeof(MKL_INT) * 64);

  // Always release the local handle, even on failure, so a failed solve never
  // leaks native memory.
  MKL_INT release_phase = -1, release_error = 0;
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
                                   .Ret<ffi::Buffer<ffi::U64>>()  // handle
                                   .Ret<ffi::Buffer<ffi::S32>>()  // status
                                   .Ret<ffi::Buffer<ffi::S32>>()  // final_iparm
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kPardisoReanalyzeHandler, PardisoReanalyzeImpl,
                               ffi::Ffi::Bind()
                                   .Attr<int64_t>("matrix_type")
                                   .Attr<int64_t>("dimension")
                                   .Arg<ffi::Buffer<ffi::U64>>()  // handle
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indptr
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indices
                                   .Arg<ffi::Buffer<ffi::F64>>()  // values
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_mask
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_values
                                   .Ret<ffi::Buffer<ffi::U64>>()  // handle
                                   .Ret<ffi::Buffer<ffi::S32>>()  // status
                                   .Ret<ffi::Buffer<ffi::S32>>()  // final_iparm
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kPardisoFactorHandler, PardisoFactorImpl,
                               ffi::Ffi::Bind()
                                   .Attr<int64_t>("matrix_type")
                                   .Attr<int64_t>("dimension")
                                   .Arg<ffi::Buffer<ffi::U64>>()  // handle
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indptr
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indices
                                   .Arg<ffi::Buffer<ffi::F64>>()  // values
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_mask
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_values
                                   .Ret<ffi::Buffer<ffi::U64>>()  // handle
                                   .Ret<ffi::Buffer<ffi::S32>>()  // status
                                   .Ret<ffi::Buffer<ffi::S32>>()  // final_iparm
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kPardisoSolveHandler, PardisoSolveImpl,
                               ffi::Ffi::Bind()
                                   .Attr<int64_t>("matrix_type")
                                   .Attr<int64_t>("dimension")
                                   .Attr<int64_t>("number_of_right_hand_sides")
                                   .Attr<int64_t>("transpose_mode")
                                   .Arg<ffi::Buffer<ffi::U64>>()  // handle
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indptr
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indices
                                   .Arg<ffi::Buffer<ffi::F64>>()  // values
                                   .Arg<ffi::Buffer<ffi::F64>>()  // right_hand_side
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_mask
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_values
                                   .Ret<ffi::Buffer<ffi::F64>>()  // solution
                                   .Ret<ffi::Buffer<ffi::S32>>()  // final_iparm
                                   .Ret<ffi::Buffer<ffi::S32>>()  // rebuild_reason
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kPardisoFactorSolveHandler, PardisoFactorSolveImpl,
                               ffi::Ffi::Bind()
                                   .Attr<int64_t>("matrix_type")
                                   .Attr<int64_t>("dimension")
                                   .Attr<int64_t>("number_of_right_hand_sides")
                                   .Attr<int64_t>("transpose_mode")
                                   .Arg<ffi::Buffer<ffi::U64>>()  // handle
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indptr
                                   .Arg<ffi::Buffer<ffi::S32>>()  // indices
                                   .Arg<ffi::Buffer<ffi::F64>>()  // values
                                   .Arg<ffi::Buffer<ffi::F64>>()  // right_hand_side
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_mask
                                   .Arg<ffi::Buffer<ffi::S32>>()  // options_values
                                   .Ret<ffi::Buffer<ffi::F64>>()  // solution
                                   .Ret<ffi::Buffer<ffi::S32>>()  // final_iparm
                                   .Ret<ffi::Buffer<ffi::S32>>()  // rebuild_reason
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kPardisoReleaseHandler, PardisoReleaseImpl,
                               ffi::Ffi::Bind()
                                   .Arg<ffi::Buffer<ffi::U64>>()  // handle
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
