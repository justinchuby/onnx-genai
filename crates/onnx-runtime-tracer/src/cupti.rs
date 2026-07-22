//! CUPTI GPU kernel collector — Activity API path (§48.8.3 / §49), behind the
//! `cupti` cargo feature.
//!
//! CUPTI (the CUDA Profiling Tools Interface) is **dlopen'd at runtime**, never
//! linked. That is the whole point of §48.8.10's "CUPTI as dlopen" decision: a
//! wheel built with this feature still loads and runs on a machine that has no
//! NVIDIA driver, no `libcupti.so`, or an AMD GPU — the collector simply reports
//! [`available`](CuptiProfiler::available) `== false` and produces an empty
//! record set, never a panic or a link error (§48.8.10 "Provider unavailable →
//! graceful skip").
//!
//! ## What this module implements (Phase 1)
//!
//! * A **dlopen shim** ([`CuptiApi`]) over the CUPTI Activity API using the
//!   [`libloading`] crate — the only symbols we resolve are the handful listed
//!   in [`CuptiApi::load`]. Any missing symbol degrades the whole profiler to
//!   `available == false`.
//! * [`CuptiProfiler`] — enables `CUPTI_ACTIVITY_KIND_KERNEL` (+ concurrent
//!   kernel, memcpy, memset), registers activity buffer callbacks, and drains
//!   completed activity records into [`GpuKernelRecord`]s on flush.
//! * [`CuptiCollector`] — a [`TraceCollector`](crate::TraceCollector) that
//!   registers op→kernel correlations on compute *Begin* events and merges the
//!   drained GPU kernel records on flush.
//! * [`CuptiFactory`] — a graceful factory whose
//!   [`try_create`](CuptiFactory::try_create) returns `Ok(None)` when CUPTI is
//!   unavailable (so a caller that requested `"cupti"` on a non-NVIDIA box is
//!   silently skipped rather than failed), matching the §48.8.4 factory shape.
//!
//! ## What is deferred to Phase 2 (documented stubs)
//!
//! * [`GpuKernelMetrics`] / [`CuptiMetric`] — the CUPTI Profiling API (PM
//!   counters: occupancy, roofline, warp-stall reasons) needs kernel *replay*
//!   and device props. The types are declared per §49.3 but their collection is
//!   a stub ([`CuptiProfiler::collect_metrics`] returns a clear "not yet"
//!   error).
//! * Executor correlation wiring — [`CuptiProfiler::correlate`] is the public
//!   entry point the CUDA EP will call with a real CUDA correlation id
//!   (§49.4). Until that lands, kernel records are captured with
//!   `node_id == None` unless an event already carries a `correlation_id`.
//!
//! ## Safety
//!
//! This is the **only** module in the crate permitted to use `unsafe` — the
//! crate keeps `#![forbid(unsafe_code)]` for every build that does not enable
//! `cupti` (see `lib.rs`). All FFI, dlopen, and raw activity-record parsing is
//! confined here.

use std::alloc::{Layout, alloc, dealloc};
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};

use crate::collector::TraceCollector;
use crate::error::{Result, TracerError};
use crate::event::{TraceEvent, TracePhase};

/// Node identifier used to correlate a GPU kernel back to the runtime op that
/// dispatched it.
///
/// The tracer crate is deliberately decoupled from `onnx-runtime-ir`, so we do
/// **not** import its `NodeId`. This mirrors that type's inner representation
/// (`NodeId(pub u32)`): a plain `u32` that the runtime carries in a
/// [`TraceEvent`]'s `args` under the `"node_id"` key.
pub type NodeId = u32;

// --- CUPTI ABI constants (from cupti_activity.h / cupti_result.h) ------------

/// `CUPTI_ACTIVITY_KIND_MEMCPY`.
const CUPTI_ACTIVITY_KIND_MEMCPY: u32 = 1;
/// `CUPTI_ACTIVITY_KIND_MEMSET`.
const CUPTI_ACTIVITY_KIND_MEMSET: u32 = 2;
/// `CUPTI_ACTIVITY_KIND_KERNEL` (serialized kernels).
const CUPTI_ACTIVITY_KIND_KERNEL: u32 = 3;
/// `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` (concurrent kernels — same record
/// layout as `KERNEL`).
const CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: u32 = 10;

/// `CUPTI_SUCCESS`.
const CUPTI_SUCCESS: u32 = 0;
/// `CUPTI_ERROR_MAX_LIMIT_REACHED` — returned by `cuptiActivityGetNextRecord`
/// once the buffer has been fully drained (the normal loop terminator).
const CUPTI_ERROR_MAX_LIMIT_REACHED: u32 = 12;

/// Required alignment of an activity buffer (`ACTIVITY_RECORD_ALIGNMENT`).
const ACTIVITY_RECORD_ALIGNMENT: usize = 8;
/// Size of each activity buffer handed to CUPTI (§49.3 used 8 × 1 MiB; a single
/// larger buffer is simpler and fine for Phase 1).
const ACTIVITY_BUFFER_SIZE: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TargetOs {
    Linux,
    Macos,
    Windows,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TargetArch {
    Aarch64,
    Other,
}

fn target_os() -> TargetOs {
    if cfg!(target_os = "linux") {
        TargetOs::Linux
    } else if cfg!(target_os = "macos") {
        TargetOs::Macos
    } else if cfg!(target_os = "windows") {
        TargetOs::Windows
    } else {
        TargetOs::Other
    }
}

fn target_arch() -> TargetArch {
    if cfg!(target_arch = "aarch64") {
        TargetArch::Aarch64
    } else {
        TargetArch::Other
    }
}

/// CUPTI library names tried through the platform loader before package paths.
fn libcupti_names_for(os: TargetOs) -> &'static [&'static str] {
    match os {
        TargetOs::Linux => &["libcupti.so.13", "libcupti.so.12", "libcupti.so"],
        TargetOs::Macos => &["libcupti.dylib"],
        TargetOs::Windows => &["cupti64_13.dll", "cupti64_12.dll", "cupti.dll"],
        TargetOs::Other => &[],
    }
}

fn cupti_supported(os: TargetOs, arch: TargetArch) -> bool {
    !(os == TargetOs::Windows && arch == TargetArch::Aarch64)
}

// --- FFI signatures ----------------------------------------------------------

/// Opaque CUPTI activity record header. Its first field is a `CUpti_ActivityKind`
/// (a `u32`) that discriminates the concrete record type.
#[repr(C)]
struct CuptiActivity {
    _opaque: [u8; 0],
}

type BufferRequestedFn =
    unsafe extern "C" fn(buffer: *mut *mut u8, size: *mut usize, max_num_records: *mut usize);
type BufferCompletedFn = unsafe extern "C" fn(
    context: *mut c_void,
    stream_id: u32,
    buffer: *mut u8,
    size: usize,
    valid_size: usize,
);

type FnActivityEnable = unsafe extern "C" fn(kind: u32) -> u32;
type FnActivityDisable = unsafe extern "C" fn(kind: u32) -> u32;
type FnActivityRegisterCallbacks =
    unsafe extern "C" fn(requested: BufferRequestedFn, completed: BufferCompletedFn) -> u32;
type FnActivityFlushAll = unsafe extern "C" fn(flag: u32) -> u32;
type FnActivityGetNextRecord = unsafe extern "C" fn(
    buffer: *mut u8,
    valid_size: usize,
    record: *mut *mut CuptiActivity,
) -> u32;
type FnGetTimestamp = unsafe extern "C" fn(timestamp: *mut u64) -> u32;

/// The resolved CUPTI entry points.
///
/// The owning [`libloading::Library`] is deliberately **not** stored here — it
/// is loaded once into a process-lifetime global ([`cupti_library`]) and never
/// unloaded. Unloading libcupti (`dlclose`) while its asynchronous activity
/// worker thread is alive unmaps code the driver is still executing and
/// segfaults the process at exit, so the library must outlive every profiler.
/// The function pointers below therefore stay valid for the whole process.
struct CuptiApi {
    activity_enable: FnActivityEnable,
    activity_disable: FnActivityDisable,
    activity_register_callbacks: FnActivityRegisterCallbacks,
    activity_flush_all: FnActivityFlushAll,
    activity_get_next_record: FnActivityGetNextRecord,
    get_timestamp: FnGetTimestamp,
}

// SAFETY: every field is a bare `extern "C"` function pointer (trivially
// `Send + Sync`, no interior state), referencing code inside the never-unloaded
// libcupti (see `cupti_library`).
unsafe impl Send for CuptiApi {}
unsafe impl Sync for CuptiApi {}

/// The dlopen'd libcupti, loaded at most once and kept mapped for the entire
/// process. See [`CuptiApi`] for why it must never be unloaded.
///
/// The `Err` arm retains the searched paths and the underlying loader error so
/// the explicit-request path can produce an actionable [`TracerError`]
/// (RULES.md #1) instead of a silent `None`.
static CUPTI_LIBRARY: OnceLock<std::result::Result<LoadedCuptiLibrary, CuptiLoadError>> =
    OnceLock::new();

/// Extra library search roots injected by an embedding runtime (e.g. the PyO3
/// Python binding feeding real `sys.path` / the extension's site-packages) so
/// pip-installed CUPTI is found even when `VIRTUAL_ENV`/`PYTHONPATH` are unset.
///
/// Populate this via [`set_search_paths`] **before** any tracing is initialized,
/// because [`CUPTI_LIBRARY`] caches the discovery result on first use.
static INJECTED_SEARCH_PATHS: OnceLock<Vec<PathBuf>> = OnceLock::new();

/// Why libcupti could not be loaded, retained for an actionable diagnostic.
struct CuptiLoadError {
    /// Every path we attempted to `dlopen`, in order.
    attempted: Vec<PathBuf>,
    /// The last underlying loader error (or a note if there were no candidates).
    cause: String,
}

/// A loaded libcupti together with the paths tried through the successful
/// candidate, retained for missing-symbol diagnostics.
struct LoadedCuptiLibrary {
    library: libloading::Library,
    attempted: Vec<PathBuf>,
}

/// Inject extra library search roots (typically Python `sys.path` entries and
/// the loaded extension module's directory) that the CUPTI loader should probe
/// for the pip layout `<root>/nvidia/cuda_cupti/{lib,bin}/<platform name>`.
///
/// This is the runtime-agnostic seam the Python binding uses: the tracer never
/// depends on PyO3, it just accepts a plain `Vec<PathBuf>`. Call this **once, at
/// import/startup time, before any tracing begins** — the loader caches its
/// discovery on first use, so paths injected afterwards are ignored. Repeated
/// calls after the first are no-ops.
pub fn set_search_paths(paths: Vec<PathBuf>) {
    let _ = INJECTED_SEARCH_PATHS.set(paths);
}

fn cupti_library() -> std::result::Result<&'static LoadedCuptiLibrary, &'static CuptiLoadError> {
    CUPTI_LIBRARY
        .get_or_init(|| {
            let os = target_os();
            let arch = target_arch();
            if !cupti_supported(os, arch) {
                return Err(CuptiLoadError {
                    attempted: Vec::new(),
                    cause: "CUPTI is unavailable on Windows ARM64 because NVIDIA ships \
                            x64-only CUPTI libraries; GPU profiling is disabled"
                        .to_string(),
                });
            }
            let candidates = libcupti_candidates();
            let mut last_error = None;
            let mut attempted = Vec::new();
            for path in &candidates {
                attempted.push(path.clone());
                // SAFETY: loading a shared library runs its initializers;
                // libcupti is a well-behaved NVIDIA library.
                match unsafe { libloading::Library::new(path) } {
                    Ok(library) => return Ok(LoadedCuptiLibrary { library, attempted }),
                    Err(error) => last_error = Some(error.to_string()),
                }
            }
            Err(CuptiLoadError {
                cause: last_error.unwrap_or_else(|| {
                    "no libcupti candidate paths were available to try".to_string()
                }),
                attempted: candidates,
            })
        })
        .as_ref()
}

fn required_symbol<T: Copy>(loaded: &LoadedCuptiLibrary, name: &'static [u8]) -> Result<T> {
    // SAFETY: callers provide the exact C signature declared by CUPTI. The
    // copied function pointer remains valid because the library is never
    // unloaded.
    unsafe { loaded.library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| {
            let symbol = String::from_utf8_lossy(name.strip_suffix(&[0]).unwrap_or(name));
            TracerError::CuptiUnavailable {
                attempted: loaded.attempted.clone(),
                cause: format!(
                    "libcupti was found and loaded, but the required CUPTI Activity API \
                     symbol `{symbol}` is missing or unusable ({error}) — the installed \
                     libcupti is likely too old for CUDA 13",
                ),
            }
        })
}

/// Build the runtime search list for CUDA 13 CUPTI.
///
/// In addition to the normal loader path, NVIDIA's pip package installs CUPTI
/// below `site-packages/nvidia/cuda_cupti/lib` on Unix and `bin` on Windows.
/// Python locations are derived at runtime from injected `sys.path` roots (see
/// [`set_search_paths`]), explicit environment hints, `PYTHONPATH`, and likely
/// interpreter prefixes; no build-machine path is embedded in the binary.
fn libcupti_candidates() -> Vec<PathBuf> {
    let injected = INJECTED_SEARCH_PATHS
        .get()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    collect_libcupti_candidates(injected)
}

/// The pure, testable core of [`libcupti_candidates`]: given explicitly injected
/// site-packages roots, plus the process environment, produce the ordered list
/// of `libcupti` paths to try.
fn collect_libcupti_candidates(injected: &[PathBuf]) -> Vec<PathBuf> {
    collect_libcupti_candidates_for(injected, target_os(), target_arch())
}

fn collect_libcupti_candidates_for(
    injected: &[PathBuf],
    os: TargetOs,
    arch: TargetArch,
) -> Vec<PathBuf> {
    if !cupti_supported(os, arch) {
        return Vec::new();
    }

    let mut candidates = libcupti_names_for(os)
        .iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let mut site_packages = Vec::new();

    // Injected roots (e.g. the Python interpreter's real `sys.path` and the
    // loaded extension's own directory) come first. They are the reliable
    // mechanism for an unactivated venv — where `VIRTUAL_ENV` is unset and
    // `/proc/self/exe` resolves to the base interpreter — and for user-site
    // (`~/.local/.../site-packages`) installs.
    site_packages.extend(injected.iter().cloned());

    if let Some(paths) = std::env::var_os("NXRT_PYTHON_SITE_PACKAGES") {
        site_packages.extend(std::env::split_paths(&paths));
    }
    if let Some(paths) = std::env::var_os("PYTHONPATH") {
        site_packages.extend(std::env::split_paths(&paths));
    }

    let mut prefixes = ["VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONHOME"]
        .into_iter()
        .filter_map(std::env::var_os)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if let Some(argv0) = std::env::args_os().next().map(PathBuf::from)
        && let Some(prefix) = argv0.parent().and_then(Path::parent)
    {
        prefixes.push(prefix.to_path_buf());
    }
    if let Ok(executable) = std::env::current_exe()
        && let Some(prefix) = executable.parent().and_then(Path::parent)
    {
        prefixes.push(prefix.to_path_buf());
    }

    for prefix in prefixes {
        discover_site_packages(&prefix, &mut site_packages);
    }
    for root in site_packages {
        push_pip_cupti_candidates(&root, os, &mut candidates);
    }
    candidates
}

fn discover_site_packages(prefix: &Path, roots: &mut Vec<PathBuf>) {
    for lib_dir in [
        prefix.join("lib"),
        prefix.join("lib64"),
        prefix.join("local/lib"),
    ] {
        let Ok(entries) = std::fs::read_dir(lib_dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if entry.file_name().to_string_lossy().starts_with("python") {
                roots.push(entry.path().join("site-packages"));
            }
        }
    }
}

fn push_pip_cupti_candidates(site_packages: &Path, os: TargetOs, candidates: &mut Vec<PathBuf>) {
    let library_subdir = if os == TargetOs::Windows {
        "bin"
    } else {
        "lib"
    };
    let library_dir = site_packages.join("nvidia/cuda_cupti").join(library_subdir);
    for soname in libcupti_names_for(os) {
        let candidate = library_dir.join(soname);
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }
}

impl CuptiApi {
    /// Resolve every CUPTI symbol we need from the process-lifetime library.
    /// Returns `None` (never an error) if the library or **any** required symbol
    /// is absent, so an **availability probe** can degrade gracefully.
    ///
    /// The explicit-request path uses [`require`](CuptiApi::require) instead,
    /// which preserves the actionable diagnostic.
    fn load() -> Option<Arc<CuptiApi>> {
        Self::require().ok()
    }

    /// Resolve every CUPTI symbol, returning an actionable [`TracerError`]
    /// (RULES.md #1) when the library cannot be loaded or a required symbol is
    /// missing. Use this on the explicit "start GPU tracing" path so the user
    /// gets a what/why/how-to-fix message instead of a silent no-op.
    fn require() -> Result<Arc<CuptiApi>> {
        let lib = cupti_library().map_err(|error| TracerError::CuptiUnavailable {
            attempted: error.attempted.clone(),
            cause: error.cause.clone(),
        })?;

        // Resolve a single symbol, mapping a missing/unusable entry point into
        // an actionable error that names it (usually a libcupti too old for the
        // requested CUPTI Activity API).
        macro_rules! symbol {
            ($ty:ty, $name:literal) => {{ required_symbol::<$ty>(lib, $name)? }};
        }

        let activity_enable = symbol!(FnActivityEnable, b"cuptiActivityEnable\0");
        let activity_disable = symbol!(FnActivityDisable, b"cuptiActivityDisable\0");
        let activity_register_callbacks = symbol!(
            FnActivityRegisterCallbacks,
            b"cuptiActivityRegisterCallbacks\0"
        );
        let activity_flush_all = symbol!(FnActivityFlushAll, b"cuptiActivityFlushAll\0");
        let activity_get_next_record =
            symbol!(FnActivityGetNextRecord, b"cuptiActivityGetNextRecord\0");
        let get_timestamp = symbol!(FnGetTimestamp, b"cuptiGetTimestamp\0");

        Ok(Arc::new(CuptiApi {
            activity_enable,
            activity_disable,
            activity_register_callbacks,
            activity_flush_all,
            activity_get_next_record,
            get_timestamp,
        }))
    }
}

/// Build the actionable "CUPTI was explicitly requested but is unavailable"
/// error (RULES.md #1). Reuses [`CuptiApi::require`]'s captured detail (attempted
/// paths + loader/symbol cause) so the message is debuggable.
fn cupti_unavailable_error() -> TracerError {
    match CuptiApi::require() {
        // A profiler with `available == false` implies loading failed, so this
        // arm is only reached on a genuine race where CUPTI just became usable.
        Ok(_) => TracerError::CuptiUnavailable {
            attempted: Vec::new(),
            cause: "CUPTI became available after the profiler reported it absent".to_string(),
        },
        Err(error) => error,
    }
}

/// The leading, ABI-stable prefix of `CUpti_ActivityKernel{4..10}`.
///
/// The header marks the record `__attribute__((packed))` with 8-byte overall
/// alignment, so this is `#[repr(C, packed)]` and fields are read with
/// [`ptr::read_unaligned`]. Only the prefix up to and including `name` is
/// modelled — CUPTI guarantees a record buffer is at least as large as its
/// record, so reading this prefix out of any kernel record version is sound.
/// This prefix has been layout-stable across every recent CUPTI major version.
#[repr(C, packed)]
struct CuptiActivityKernelPrefix {
    kind: u32,
    cache_config: u8,
    shared_memory_config: u8,
    registers_per_thread: u16,
    partitioned_global_cache_requested: u32,
    partitioned_global_cache_executed: u32,
    start: u64,
    end: u64,
    completed: u64,
    device_id: u32,
    context_id: u32,
    stream_id: u32,
    grid_x: i32,
    grid_y: i32,
    grid_z: i32,
    block_x: i32,
    block_y: i32,
    block_z: i32,
    static_shared_memory: i32,
    dynamic_shared_memory: i32,
    local_memory_per_thread: u32,
    local_memory_total: u32,
    correlation_id: u32,
    grid_id: i64,
    name: *const c_char,
}

// --- Public data model (§49.3) -----------------------------------------------

/// Correlation from a CUDA correlation id back to the runtime op that launched
/// the kernel.
#[derive(Debug, Clone)]
pub struct KernelCorrelation {
    /// The runtime node that dispatched the kernel.
    pub node_id: NodeId,
    /// The ONNX op type of that node (e.g. `"MatMul"`).
    pub op_type: String,
}

/// One GPU kernel execution record drained from the CUPTI Activity API (§49.3).
#[derive(Debug, Clone)]
pub struct GpuKernelRecord {
    /// Correlation back to the runtime op dispatch, if the launching op
    /// registered one via [`CuptiProfiler::correlate`]. `None` when the kernel
    /// could not be tied to a runtime node (e.g. driver-internal kernels, or
    /// before executor correlation wiring lands — see the module docs).
    pub node_id: Option<NodeId>,
    /// The ONNX op type of the correlated node, or `""` if uncorrelated.
    pub op_type: String,
    /// The GPU kernel name (mangled device symbol, e.g. `volta_sgemm_128x64`).
    pub kernel_name: String,
    /// Kernel start timestamp (GPU clock, nanoseconds).
    pub start_ns: u64,
    /// Kernel end timestamp (GPU clock, nanoseconds).
    pub end_ns: u64,
    /// Kernel duration in nanoseconds (`end_ns - start_ns`).
    pub duration_ns: u64,
    /// Grid dimensions `(x, y, z)`.
    pub grid: (u32, u32, u32),
    /// Block dimensions `(x, y, z)`.
    pub block: (u32, u32, u32),
    /// Total shared memory (static + dynamic) reserved for the kernel, in bytes.
    pub shared_memory_bytes: u32,
    /// Registers used per thread.
    pub registers_per_thread: u32,
    /// The CUDA stream the kernel ran on.
    pub stream_id: u32,
    /// Theoretical occupancy (0.0–1.0). **Phase 2**: computing this needs device
    /// props + the CUDA occupancy calculator, so it is `0.0` for now.
    pub theoretical_occupancy: f32,
    /// Achieved occupancy (0.0–1.0). Requires PM-counter metric collection
    /// ([`GpuKernelMetrics`]); `None` in the Activity-API path.
    pub achieved_occupancy: Option<f32>,
}

/// Hardware performance metrics from the CUPTI Profiling API / PM sampling
/// (§49.3). **Phase 2 stub**: the type is declared per the design, but its
/// collection ([`CuptiProfiler::collect_metrics`]) is not implemented — PM
/// counters require kernel replay and device configuration.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GpuKernelMetrics {
    /// The kernel these metrics describe.
    pub kernel_name: String,
    /// Correlated runtime node, if any.
    pub node_id: Option<NodeId>,
    /// Percentage of SMs active.
    pub sm_efficiency: f32,
    /// Achieved occupancy (percentage of max warps).
    pub achieved_occupancy: f32,
    /// Percentage of Tensor Core active cycles.
    pub tensor_core_utilization: f32,
    /// DRAM bytes read.
    pub dram_read_bytes: u64,
    /// DRAM bytes written.
    pub dram_write_bytes: u64,
    /// DRAM throughput as a percentage of theoretical peak.
    pub dram_throughput_pct: f32,
    /// L2 cache hit rate.
    pub l2_hit_rate: f32,
}

/// A hardware metric requestable from the CUPTI Profiling API (§49.3).
/// **Phase 2**: paired with [`GpuKernelMetrics`]; collection is not yet wired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CuptiMetric {
    /// Percentage of SMs active.
    SmEfficiency,
    /// Percentage of max warps resident.
    AchievedOccupancy,
    /// Tensor Core active-cycle percentage.
    TensorCoreUtilization,
    /// DRAM throughput vs. theoretical peak.
    DramThroughput,
    /// L2 cache hit rate.
    L2HitRate,
    /// Warp stall-reason breakdown.
    WarpStallReasons,
    /// FLOP counts (single- and half-precision).
    FlopCount,
}

// --- Shared state reachable from the C activity callbacks --------------------

/// State the C activity-buffer callbacks need. The `complete` callback is a bare
/// `extern "C"` function pointer with no user-data argument, so the drained
/// records and the CUPTI entry points must be reachable through a global.
struct SharedState {
    /// Correlation map: CUDA correlation id → launching node/op (§49.3).
    correlations: Mutex<HashMap<u32, KernelCorrelation>>,
    /// Kernel records drained from completed activity buffers.
    records: Mutex<Vec<GpuKernelRecord>>,
}

impl SharedState {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            correlations: Mutex::new(HashMap::new()),
            records: Mutex::new(Vec::new()),
        })
    }
}

/// The currently-active tracing session, published for the C callbacks.
///
/// CUPTI's buffer callbacks are process-global (one registration at a time), so
/// a single global session is the correct model. It is set by
/// [`CuptiProfiler::start_activity_tracing`] and cleared by
/// [`CuptiProfiler::stop_and_flush`].
struct ActiveSession {
    api: Arc<CuptiApi>,
    shared: Arc<SharedState>,
}

static ACTIVE: OnceLock<Mutex<Option<ActiveSession>>> = OnceLock::new();

fn active_slot() -> &'static Mutex<Option<ActiveSession>> {
    ACTIVE.get_or_init(|| Mutex::new(None))
}

/// CUPTI callback: hand CUPTI an aligned buffer to fill with activity records.
unsafe extern "C" fn buffer_requested(
    buffer: *mut *mut u8,
    size: *mut usize,
    max_num_records: *mut usize,
) {
    // SAFETY: CUPTI passes valid out-pointers per the Activity API contract.
    unsafe {
        let layout = Layout::from_size_align(ACTIVITY_BUFFER_SIZE, ACTIVITY_RECORD_ALIGNMENT)
            .expect("activity buffer layout is valid");
        let ptr = alloc(layout);
        *buffer = ptr;
        *size = if ptr.is_null() {
            0
        } else {
            ACTIVITY_BUFFER_SIZE
        };
        *max_num_records = 0; // 0 = fill with as many records as fit.
    }
}

/// CUPTI callback: drain a completed activity buffer, then free it.
unsafe extern "C" fn buffer_completed(
    _context: *mut c_void,
    _stream_id: u32,
    buffer: *mut u8,
    size: usize,
    valid_size: usize,
) {
    if !buffer.is_null() && valid_size > 0 {
        // Copy the Arcs out under the lock so we don't hold it while parsing.
        let session = {
            let guard = active_slot()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard
                .as_ref()
                .map(|s| (Arc::clone(&s.api), Arc::clone(&s.shared)))
        };
        if let Some((api, shared)) = session {
            // SAFETY: `buffer`/`valid_size` are the CUPTI-owned buffer we are
            // draining; parsing only reads the ABI prefix of each record.
            unsafe { drain_buffer(&api, &shared, buffer, valid_size) };
        }
    }

    if !buffer.is_null() {
        // SAFETY: `buffer` was allocated in `buffer_requested` with this exact
        // size/alignment; CUPTI returns it here for us to free.
        unsafe {
            let layout = Layout::from_size_align(size, ACTIVITY_RECORD_ALIGNMENT)
                .expect("activity buffer layout is valid");
            dealloc(buffer, layout);
        }
    }
}

/// Iterate every record in a completed buffer and push kernel records.
unsafe fn drain_buffer(api: &CuptiApi, shared: &SharedState, buffer: *mut u8, valid_size: usize) {
    let mut record: *mut CuptiActivity = ptr::null_mut();
    loop {
        // SAFETY: standard CUPTI drain loop; `get_next_record` advances `record`
        // and returns MAX_LIMIT_REACHED once the buffer is exhausted.
        let status = unsafe { (api.activity_get_next_record)(buffer, valid_size, &mut record) };
        if status != CUPTI_SUCCESS {
            // MAX_LIMIT_REACHED is the normal terminator; any other status also
            // ends the loop (degrade gracefully rather than spin).
            let _ = CUPTI_ERROR_MAX_LIMIT_REACHED;
            break;
        }
        if record.is_null() {
            break;
        }
        // SAFETY: the first field of every activity record is its kind (u32).
        let kind = unsafe { ptr::read_unaligned(record.cast::<u32>()) };
        if kind == CUPTI_ACTIVITY_KIND_KERNEL || kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL {
            // SAFETY: a kernel record begins with the modelled prefix.
            if let Some(rec) = unsafe { parse_kernel_record(shared, record) } {
                shared
                    .records
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(rec);
            }
        }
        // MEMCPY / MEMSET records are enabled (for completeness / lower dropped
        // counts) but not modelled as kernel records in Phase 1.
    }
}

/// Parse one kernel activity record into a [`GpuKernelRecord`].
unsafe fn parse_kernel_record(
    shared: &SharedState,
    record: *mut CuptiActivity,
) -> Option<GpuKernelRecord> {
    let prefix = record.cast::<CuptiActivityKernelPrefix>();
    // SAFETY: `record` points at a kernel record whose prefix matches the
    // modelled layout; every field is read unaligned (the record is packed).
    unsafe {
        let start = ptr::read_unaligned(ptr::addr_of!((*prefix).start));
        let end = ptr::read_unaligned(ptr::addr_of!((*prefix).end));
        let registers_per_thread =
            ptr::read_unaligned(ptr::addr_of!((*prefix).registers_per_thread));
        let stream_id = ptr::read_unaligned(ptr::addr_of!((*prefix).stream_id));
        let grid_x = ptr::read_unaligned(ptr::addr_of!((*prefix).grid_x));
        let grid_y = ptr::read_unaligned(ptr::addr_of!((*prefix).grid_y));
        let grid_z = ptr::read_unaligned(ptr::addr_of!((*prefix).grid_z));
        let block_x = ptr::read_unaligned(ptr::addr_of!((*prefix).block_x));
        let block_y = ptr::read_unaligned(ptr::addr_of!((*prefix).block_y));
        let block_z = ptr::read_unaligned(ptr::addr_of!((*prefix).block_z));
        let static_shared = ptr::read_unaligned(ptr::addr_of!((*prefix).static_shared_memory));
        let dynamic_shared = ptr::read_unaligned(ptr::addr_of!((*prefix).dynamic_shared_memory));
        let correlation_id = ptr::read_unaligned(ptr::addr_of!((*prefix).correlation_id));
        let name_ptr = ptr::read_unaligned(ptr::addr_of!((*prefix).name));

        let kernel_name = cstr_to_string(name_ptr);
        let (node_id, op_type) = shared
            .correlations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&correlation_id)
            .map(|c| (Some(c.node_id), c.op_type.clone()))
            .unwrap_or((None, String::new()));

        Some(GpuKernelRecord {
            node_id,
            op_type,
            kernel_name,
            start_ns: start,
            end_ns: end,
            duration_ns: end.saturating_sub(start),
            grid: (
                grid_x.max(0) as u32,
                grid_y.max(0) as u32,
                grid_z.max(0) as u32,
            ),
            block: (
                block_x.max(0) as u32,
                block_y.max(0) as u32,
                block_z.max(0) as u32,
            ),
            shared_memory_bytes: (static_shared.max(0) as u32)
                .saturating_add(dynamic_shared.max(0) as u32),
            registers_per_thread: u32::from(registers_per_thread),
            stream_id,
            theoretical_occupancy: 0.0,
            achieved_occupancy: None,
        })
    }
}

/// Copy a borrowed C string into an owned `String` (lossy for non-UTF-8).
unsafe fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: CUPTI kernel-name pointers are NUL-terminated and stay valid for
    // the lifetime of the record; we only borrow to copy the bytes out.
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

// --- CuptiProfiler -----------------------------------------------------------

/// A CUPTI profiling session wrapping the Activity API (§49.3).
///
/// Construct with [`CuptiProfiler::new`], which dlopen's `libcupti` and reports
/// [`available`](CuptiProfiler::available). All methods are safe no-ops (or
/// return empty results) when CUPTI is unavailable, so callers never need to
/// branch on platform.
pub struct CuptiProfiler {
    available: bool,
    api: Option<Arc<CuptiApi>>,
    shared: Arc<SharedState>,
    tracing: Mutex<bool>,
}

impl CuptiProfiler {
    /// Initialize a profiler, attempting to dlopen `libcupti` at runtime.
    ///
    /// Never fails on a machine without CUPTI: it returns a profiler with
    /// [`available`](CuptiProfiler::available) `== false` (graceful
    /// degradation, §48.8.10).
    ///
    /// # Errors
    ///
    /// Currently infallible; returns [`Result`] to match §49.3 and to leave
    /// room for future initialization that can fail.
    pub fn new() -> Result<Self> {
        let api = CuptiApi::load();
        Ok(Self {
            available: api.is_some(),
            api,
            shared: SharedState::new(),
            tracing: Mutex::new(false),
        })
    }

    /// Initialize a profiler for an **explicit** GPU-tracing request, returning
    /// an actionable [`TracerError::CuptiUnavailable`] (RULES.md #1) when CUPTI
    /// cannot be loaded.
    ///
    /// Use this — not [`new`](CuptiProfiler::new) — when the user has explicitly
    /// asked for CUPTI/GPU tracing and a silent "unavailable" would be wrong.
    /// [`new`](CuptiProfiler::new) remains the graceful availability probe for
    /// capability detection and factory auto-selection.
    ///
    /// # Errors
    ///
    /// [`TracerError::CuptiUnavailable`] if `libcupti` is missing, unusable, or
    /// too old — the message names the attempted paths, the underlying cause,
    /// and the `pip install nvidia-cuda-cupti-cu13` fix.
    pub fn require() -> Result<Self> {
        let api = CuptiApi::require()?;
        Ok(Self {
            available: true,
            api: Some(api),
            shared: SharedState::new(),
            tracing: Mutex::new(false),
        })
    }

    /// Whether CUPTI was successfully dlopen'd and all required symbols
    /// resolved.
    #[must_use]
    pub fn available(&self) -> bool {
        self.available
    }

    /// Start Activity-API tracing (the low-overhead mode, §49.6).
    ///
    /// Enables the kernel/memcpy/memset activity kinds and registers the buffer
    /// callbacks. This is an **explicit tracing request**: if CUPTI is
    /// unavailable it returns an actionable [`TracerError::CuptiUnavailable`]
    /// (RULES.md #1) rather than silently succeeding, so the caller learns what
    /// to install. Availability probing that must degrade quietly should check
    /// [`available`](CuptiProfiler::available) first.
    ///
    /// # Errors
    ///
    /// [`TracerError::CuptiUnavailable`] if CUPTI could not be loaded, or
    /// [`TracerError::Cupti`] if a CUPTI call reports a non-success status.
    pub fn start_activity_tracing(&self) -> Result<()> {
        let Some(api) = self.api.as_ref() else {
            return Err(cupti_unavailable_error());
        };

        {
            let mut slot = active_slot()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *slot = Some(ActiveSession {
                api: Arc::clone(api),
                shared: Arc::clone(&self.shared),
            });
        }

        // SAFETY: FFI into resolved CUPTI symbols with valid arguments.
        unsafe {
            check(
                "cuptiActivityRegisterCallbacks",
                (api.activity_register_callbacks)(buffer_requested, buffer_completed),
            )?;
            check(
                "cuptiActivityEnable(KERNEL)",
                (api.activity_enable)(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
            )
            .or_else(|_| {
                check(
                    "cuptiActivityEnable(KERNEL)",
                    (api.activity_enable)(CUPTI_ACTIVITY_KIND_KERNEL),
                )
            })?;
            // MEMCPY/MEMSET are best-effort: a failure to enable them must not
            // fail the whole session (kernels are the primary signal).
            let _ = (api.activity_enable)(CUPTI_ACTIVITY_KIND_MEMCPY);
            let _ = (api.activity_enable)(CUPTI_ACTIVITY_KIND_MEMSET);
        }

        *self
            .tracing
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
        Ok(())
    }

    /// Stop tracing, flush all pending activity buffers, and return the drained
    /// kernel records. A no-op returning an empty `Vec` when CUPTI is
    /// unavailable.
    ///
    /// # Errors
    ///
    /// [`TracerError::Cupti`] if the flush reports a non-success status.
    pub fn stop_and_flush(&self) -> Result<Vec<GpuKernelRecord>> {
        let Some(api) = self.api.as_ref() else {
            return Ok(Vec::new());
        };

        let was_tracing = {
            let mut t = self
                .tracing
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            std::mem::replace(&mut *t, false)
        };

        if was_tracing {
            // SAFETY: FFI into resolved CUPTI symbols; `0` flushes everything.
            unsafe {
                check("cuptiActivityFlushAll", (api.activity_flush_all)(0))?;
                let _ = (api.activity_disable)(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
                let _ = (api.activity_disable)(CUPTI_ACTIVITY_KIND_KERNEL);
                let _ = (api.activity_disable)(CUPTI_ACTIVITY_KIND_MEMCPY);
                let _ = (api.activity_disable)(CUPTI_ACTIVITY_KIND_MEMSET);
            }
        }

        {
            let mut slot = active_slot()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *slot = None;
        }

        let records = std::mem::take(
            &mut *self
                .shared
                .records
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        Ok(records)
    }

    /// Register a kernel-launch correlation (called from EP dispatch, §49.4).
    ///
    /// Maps a CUDA `correlation_id` to the launching runtime node so drained
    /// kernel records can be tied back to their op.
    pub fn correlate(&self, correlation_id: u32, node_id: NodeId, op_type: &str) {
        self.shared
            .correlations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                correlation_id,
                KernelCorrelation {
                    node_id,
                    op_type: op_type.to_string(),
                },
            );
    }

    /// Read the current CUPTI timestamp (GPU clock, nanoseconds), if available.
    #[must_use]
    pub fn timestamp_ns(&self) -> Option<u64> {
        let api = self.api.as_ref()?;
        let mut ts: u64 = 0;
        // SAFETY: FFI into a resolved CUPTI symbol with a valid out-pointer.
        let status = unsafe { (api.get_timestamp)(&mut ts) };
        (status == CUPTI_SUCCESS).then_some(ts)
    }

    /// Collect detailed PM-counter metrics for specific kernels. **Phase 2**:
    /// the CUPTI Profiling API requires kernel replay and is not yet wired.
    ///
    /// # Errors
    ///
    /// Always returns [`TracerError::Cupti`] describing that metric collection
    /// is a Phase-2 capability and pointing at Activity-mode as the available
    /// alternative.
    pub fn collect_metrics(
        &self,
        _kernel_names: &[&str],
        _metrics: &[CuptiMetric],
        _num_runs: usize,
    ) -> Result<Vec<GpuKernelMetrics>> {
        Err(TracerError::Cupti {
            op: "collect_metrics",
            message: "CUPTI PM-counter metrics (occupancy, roofline, warp-stall \
                      reasons) require the CUPTI Profiling API with kernel replay, \
                      which is a Phase-2 capability and is not yet implemented. Use \
                      Activity mode (start_activity_tracing / stop_and_flush) for \
                      kernel timing and launch-config records in the meantime."
                .to_string(),
        })
    }
}

/// Map a raw CUPTI status into a [`Result`], with an actionable message
/// (RULES.md #1).
fn check(op: &'static str, status: u32) -> Result<()> {
    if status == CUPTI_SUCCESS {
        Ok(())
    } else {
        Err(TracerError::Cupti {
            op,
            message: format!(
                "CUPTI call `{op}` failed with status {status}. This usually means \
                 another profiler holds CUPTI (only one CUPTI client per process — \
                 e.g. Nsight Systems/Compute or another nxrt profiler), the CUDA \
                 driver is too old for the loaded libcupti, or no CUDA context \
                 exists yet. Ensure no other CUDA profiler is attached and that a \
                 CUDA context is initialized, then retry; GPU tracing is optional, \
                 so the run can also proceed without it."
            ),
        })
    }
}

// --- CuptiCollector ----------------------------------------------------------

/// A [`TraceCollector`] that captures GPU kernel activity via CUPTI (§48.8.3).
///
/// On each compute *Begin* event it registers an op→kernel correlation (when the
/// event carries the needed ids); on [`flush`](TraceCollector::flush) it stops
/// and flushes CUPTI and retains the drained [`GpuKernelRecord`]s, readable via
/// [`gpu_records`](CuptiCollector::gpu_records).
///
/// Merging those records into the Perfetto trace as `gpu.stream*` tracks is
/// **Phase 2** (executor wiring, §49.4) — Phase 1 captures and exposes them.
pub struct CuptiCollector {
    profiler: CuptiProfiler,
    drained: Mutex<Vec<GpuKernelRecord>>,
}

impl CuptiCollector {
    /// Create a collector and start Activity-API tracing.
    ///
    /// This is an **explicit** GPU-tracing request: when CUPTI is unavailable it
    /// returns an actionable [`TracerError::CuptiUnavailable`] (RULES.md #1)
    /// naming the `pip install nvidia-cuda-cupti-cu13` fix. Callers that want the
    /// graceful "skip if absent" behavior (auto-selection) should go through
    /// [`CuptiFactory::try_create`], which returns `Ok(None)` when CUPTI is
    /// missing.
    ///
    /// # Errors
    ///
    /// [`TracerError::CuptiUnavailable`] if CUPTI is not present/usable, or
    /// [`TracerError::Cupti`] if CUPTI is present but tracing could not be
    /// started.
    pub fn new() -> Result<Self> {
        let profiler = CuptiProfiler::new()?;
        profiler.start_activity_tracing()?;
        Ok(Self {
            profiler,
            drained: Mutex::new(Vec::new()),
        })
    }

    /// Whether the underlying [`CuptiProfiler`] found CUPTI.
    #[must_use]
    pub fn available(&self) -> bool {
        self.profiler.available()
    }

    /// The GPU kernel records drained by the most recent
    /// [`flush`](TraceCollector::flush).
    #[must_use]
    pub fn gpu_records(&self) -> Vec<GpuKernelRecord> {
        self.drained
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// The underlying profiler, for direct correlation registration (§49.4).
    #[must_use]
    pub fn profiler(&self) -> &CuptiProfiler {
        &self.profiler
    }
}

/// Extract a `u32` field from an event's `args` object, if present.
fn arg_u32(event: &TraceEvent, key: &str) -> Option<u32> {
    event
        .args
        .as_ref()?
        .get(key)?
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
}

impl TraceCollector for CuptiCollector {
    fn emit(&self, event: &TraceEvent) {
        // Register a correlation on kernel dispatch (compute Begin). We can only
        // do so if the event already carries both the runtime node id and the
        // CUDA correlation id; supplying the latter is the Phase-2 executor
        // wiring point (§49.4). Absent it, the kernel is still recorded, just
        // without an op correlation.
        if event.cat != "compute" || event.ph != TracePhase::Begin {
            return;
        }
        if let (Some(node_id), Some(corr_id)) =
            (arg_u32(event, "node_id"), arg_u32(event, "correlation_id"))
        {
            self.profiler.correlate(corr_id, node_id, &event.name);
        }
    }

    fn flush(&self) -> Result<()> {
        let records = self.profiler.stop_and_flush()?;
        let mut drained = self
            .drained
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        drained.extend(records);
        Ok(())
    }
}

// --- CuptiFactory ------------------------------------------------------------

/// A graceful factory for [`CuptiCollector`] (§48.8.4).
///
/// The crate has no collector *registry* yet (that is a later phase), so this is
/// a standalone factory: [`try_create`](CuptiFactory::try_create) returns
/// `Ok(None)` when CUPTI is unavailable — the "provider unavailable → graceful
/// skip" contract from §48.8.10 — and `Ok(Some(collector))` otherwise, ready to
/// add to a [`CompositeCollector`](crate::CompositeCollector).
#[derive(Debug, Default, Clone, Copy)]
pub struct CuptiFactory;

impl CuptiFactory {
    /// Try to build a CUPTI collector. Returns `Ok(None)` if CUPTI is not
    /// present on this system (so a caller that requested `"cupti"` on an AMD /
    /// CPU-only box is skipped, not failed).
    ///
    /// # Errors
    ///
    /// [`TracerError::Cupti`] if CUPTI is present but tracing could not be
    /// started.
    pub fn try_create(&self) -> Result<Option<Box<dyn TraceCollector>>> {
        let profiler = CuptiProfiler::new()?;
        if !profiler.available() {
            return Ok(None);
        }
        let collector = CuptiCollector::new()?;
        Ok(Some(Box::new(collector)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn profiler_new_never_panics_and_is_consistent() {
        // Whether or not libcupti is present, construction must succeed and
        // `available` must be a coherent flag (true only if the api loaded).
        let profiler = CuptiProfiler::new().expect("CuptiProfiler::new is infallible");
        assert_eq!(profiler.available(), profiler.api.is_some());
    }

    #[test]
    fn generates_per_os_cupti_names() {
        assert_eq!(
            libcupti_names_for(TargetOs::Linux),
            ["libcupti.so.13", "libcupti.so.12", "libcupti.so"]
        );
        assert_eq!(libcupti_names_for(TargetOs::Macos), ["libcupti.dylib"]);
        assert_eq!(
            libcupti_names_for(TargetOs::Windows),
            ["cupti64_13.dll", "cupti64_12.dll", "cupti.dll"]
        );
        assert!(libcupti_names_for(TargetOs::Other).is_empty());
    }

    #[test]
    fn pip_wheel_candidates_use_per_os_layout() {
        let site_packages = Path::new("/venv/lib/python3.12/site-packages");
        let mut linux = Vec::new();
        push_pip_cupti_candidates(site_packages, TargetOs::Linux, &mut linux);
        assert_eq!(
            linux[0],
            PathBuf::from(
                "/venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib/libcupti.so.13"
            )
        );

        let mut windows = Vec::new();
        push_pip_cupti_candidates(site_packages, TargetOs::Windows, &mut windows);
        assert_eq!(
            windows[0],
            PathBuf::from(
                "/venv/lib/python3.12/site-packages/nvidia/cuda_cupti/bin/cupti64_13.dll"
            )
        );
    }

    #[test]
    fn windows_arm64_degrades_to_unavailable() {
        assert!(
            collect_libcupti_candidates_for(&[], TargetOs::Windows, TargetArch::Aarch64).is_empty()
        );
        assert!(cupti_supported(TargetOs::Windows, TargetArch::Other));
    }

    #[test]
    fn stop_and_flush_is_empty_without_tracing() {
        let profiler = CuptiProfiler::new().unwrap();
        // Never started → no records, regardless of CUPTI availability.
        let records = profiler.stop_and_flush().unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn correlate_records_mapping() {
        let profiler = CuptiProfiler::new().unwrap();
        profiler.correlate(42, 7, "MatMul");
        let map = profiler.shared.correlations.lock().unwrap();
        let corr = map.get(&42).expect("correlation stored");
        assert_eq!(corr.node_id, 7);
        assert_eq!(corr.op_type, "MatMul");
    }

    #[test]
    fn collect_metrics_is_phase2_stub() {
        let profiler = CuptiProfiler::new().unwrap();
        let err = profiler
            .collect_metrics(&["k"], &[CuptiMetric::AchievedOccupancy], 1)
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Phase-2"),
            "message should name the phase: {msg}"
        );
        assert!(
            msg.contains("Activity mode"),
            "message should suggest an alternative: {msg}"
        );
    }

    #[test]
    fn collector_construction_is_graceful_or_actionable() {
        // The factory is the graceful auto-selection path: absent CUPTI must be
        // Ok(None), present CUPTI Ok(Some).
        let factory = CuptiFactory
            .try_create()
            .expect("factory never errors on absence");
        let profiler = CuptiProfiler::new().unwrap();
        if profiler.available() {
            assert!(
                factory.is_some(),
                "collector expected when CUPTI is present"
            );
            // Explicit construction succeeds and emit/flush do not panic.
            let collector = CuptiCollector::new().expect("collector construction with CUPTI");
            let event = TraceEvent {
                name: "MatMul_0".to_string(),
                cat: "compute".to_string(),
                ph: TracePhase::Begin,
                ts: 0,
                dur: None,
                pid: 1,
                tid: 1,
                scope: None,
                args: Some(json!({ "node_id": 7, "correlation_id": 42 })),
            };
            collector.emit(&event);
            collector.flush().expect("flush is graceful");
        } else {
            assert!(
                factory.is_none(),
                "graceful skip expected when CUPTI is absent"
            );
            // But the explicit collector path surfaces the actionable error.
            let err = CuptiCollector::new()
                .err()
                .expect("explicit request must error when absent");
            assert!(
                err.to_string().contains("nvidia-cuda-cupti-cu13"),
                "actionable: {err}"
            );
        }
    }

    #[test]
    fn explicit_request_errors_actionably_when_unavailable() {
        // On a host without a usable CUPTI, the explicit "start GPU trace" paths
        // must return a what/why/how-to-fix diagnostic (RULES.md #1), never a
        // silent success or a bare `None`.
        let profiler = CuptiProfiler::new().unwrap();
        if profiler.available() {
            // A real CUPTI is present (GPU host); nothing to assert here.
            return;
        }

        for err in [
            profiler
                .start_activity_tracing()
                .err()
                .expect("start must error when absent"),
            CuptiProfiler::require()
                .err()
                .expect("require must error when absent"),
            cupti_unavailable_error(),
        ] {
            let msg = err.to_string();
            // WHAT: names CUPTI / GPU tracing as the thing that failed.
            assert!(
                msg.contains("CUPTI") && msg.to_lowercase().contains("gpu"),
                "WHAT missing: {msg}"
            );
            // WHY: GPU tracing needs the CUDA-13 CUPTI runtime.
            assert!(
                msg.contains("CUDA-13") || msg.contains("CUDA 13"),
                "WHY missing: {msg}"
            );
            // HOW: the concrete pip fix, version-matched.
            assert!(
                msg.contains("pip install nvidia-cuda-cupti-cu13"),
                "HOW missing: {msg}"
            );
            // The variant carries structured, debuggable context.
            assert!(matches!(err, TracerError::CuptiUnavailable { .. }));
        }
    }

    #[test]
    fn missing_symbol_error_retains_loaded_path_and_underlying_error() {
        let loaded_path = PathBuf::from("libc.so.6");
        // SAFETY: libc is loaded only to exercise a real failed dlsym.
        let library =
            unsafe { libloading::Library::new(&loaded_path) }.expect("load libc for symbol test");
        let loaded = LoadedCuptiLibrary {
            library,
            attempted: vec![loaded_path.clone()],
        };
        let missing_symbol = b"nxrt_cupti_symbol_that_does_not_exist\0";

        let err = match required_symbol::<unsafe extern "C" fn()>(&loaded, missing_symbol) {
            Ok(_) => panic!("test symbol unexpectedly resolved"),
            Err(err) => err,
        };
        let msg = err.to_string();

        assert!(
            msg.contains("CUPTI") && msg.to_lowercase().contains("gpu"),
            "WHAT missing: {msg}"
        );
        assert!(
            msg.contains("CUDA-13") || msg.contains("CUDA 13"),
            "WHY missing: {msg}"
        );
        assert!(
            msg.contains("pip install nvidia-cuda-cupti-cu13"),
            "HOW missing: {msg}"
        );
        assert!(msg.contains("libc.so.6"), "loaded path missing: {msg}");
        assert!(
            msg.contains("nxrt_cupti_symbol_that_does_not_exist"),
            "symbol name missing: {msg}"
        );
        assert!(
            msg.contains("undefined symbol"),
            "underlying symbol error missing: {msg}"
        );
        assert!(matches!(
            err,
            TracerError::CuptiUnavailable { attempted, .. }
                if attempted == vec![loaded_path]
        ));
    }

    #[test]
    fn availability_probe_degrades_quietly() {
        // The probe path (new / available / factory) must NEVER error, so a
        // CPU/normal run that only asks "is CUPTI here?" is unaffected.
        let profiler = CuptiProfiler::new().expect("new is a graceful probe");
        assert_eq!(profiler.available(), profiler.api.is_some());
        let _ = CuptiFactory
            .try_create()
            .expect("factory auto-selection never errors");
    }

    #[test]
    fn injected_site_packages_are_probed_with_pip_layout() {
        // Fix #2: a pip-style env where VIRTUAL_ENV/PYTHONPATH are irrelevant —
        // discovery must still find CUPTI under an injected site-packages root
        // using the platform's standard `nvidia/cuda_cupti/{lib,bin}` layout.
        let base = std::env::temp_dir().join(format!(
            "nxrt-cupti-discovery-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let os = target_os();
        let Some(library_name) = libcupti_names_for(os).first() else {
            return;
        };
        let library_subdir = if os == TargetOs::Windows {
            "bin"
        } else {
            "lib"
        };
        let lib_dir = base.join("nvidia/cuda_cupti").join(library_subdir);
        std::fs::create_dir_all(&lib_dir).expect("create dummy site-packages layout");
        let dummy = lib_dir.join(library_name);
        std::fs::write(&dummy, b"not a real library").expect("write dummy libcupti");

        let candidates = collect_libcupti_candidates(std::slice::from_ref(&base));

        assert!(
            candidates.contains(&dummy),
            "injected site-packages libcupti not discovered: {candidates:?}"
        );
        assert!(
            dummy.exists(),
            "dummy libcupti should exist on disk for the probe"
        );

        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn set_search_paths_injects_into_discovery() {
        // The runtime-agnostic seam the Python binding uses: injected paths feed
        // the OnceLock-backed discovery. Only one test sets the global.
        let base = std::env::temp_dir().join(format!("nxrt-cupti-inject-{}", std::process::id()));
        set_search_paths(vec![base.clone()]);
        let expected = base
            .join("nvidia/cuda_cupti")
            .join(if cfg!(target_os = "windows") {
                "bin/cupti64_13.dll"
            } else if cfg!(target_os = "macos") {
                "lib/libcupti.dylib"
            } else {
                "lib/libcupti.so.13"
            });
        assert!(
            libcupti_candidates().contains(&expected),
            "set_search_paths did not reach the discovery list",
        );
    }

    #[test]
    fn emit_registers_correlation_when_ids_present() {
        // Requires a live CUPTI (explicit construction now errors otherwise).
        if !CuptiProfiler::new().unwrap().available() {
            return;
        }
        let collector = CuptiCollector::new().unwrap();
        let event = TraceEvent {
            name: "Gemm_3".to_string(),
            cat: "compute".to_string(),
            ph: TracePhase::Begin,
            ts: 0,
            dur: None,
            pid: 1,
            tid: 1,
            scope: None,
            args: Some(json!({ "node_id": 3, "correlation_id": 99 })),
        };
        collector.emit(&event);
        let map = collector.profiler().shared.correlations.lock().unwrap();
        assert_eq!(map.get(&99).map(|c| c.node_id), Some(3));
    }
}
