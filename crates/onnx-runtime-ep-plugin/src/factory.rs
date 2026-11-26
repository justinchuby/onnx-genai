//! `ExportedFactory` — the heap object behind an opaque `OrtEpFactory*`.
//!
//! Owns an EP constructor and fills the `OrtEpFactory` vtable so ORT can
//! discover devices, create EPs, and release them.

use std::collections::HashMap;
use std::ffi::{CString, c_char};
use std::panic::AssertUnwindSafe;
use std::ptr;
use std::sync::{Arc, Mutex};

use onnx_genai_ort_sys as ort;
use onnx_runtime_ep_api::provider::ExecutionProvider;

use crate::device::{DeviceAllocator, DeviceSupport, DeviceSyncStream};
use crate::ep::ExportedEp;
use crate::status::{fail_status, host_api, ok_status, set_host_api};

/// A heap-allocated factory whose raw pointer is returned as `OrtEpFactory*`.
///
/// The first field is the `OrtEpFactory` vtable struct so the pointer can be
/// cast directly to `*mut OrtEpFactory` (ORT dereferences the vtable fields
/// at fixed offsets from the factory pointer).
#[repr(C)]
pub struct ExportedFactory {
    /// The vtable ORT reads through the `OrtEpFactory*` pointer.
    pub vtable: ort::OrtEpFactory,
    /// The EP name as a C string, kept alive for `GetName`.
    pub name_cstr: CString,
    /// Vendor name as a C string, kept alive for `GetVendor`.
    pub vendor_cstr: CString,
    /// Version string as a C string, kept alive for `GetVersion`.
    pub version_cstr: CString,
    /// Constructor that produces a fresh EP instance.
    pub constructor: Box<dyn Fn() -> Box<dyn ExecutionProvider> + Send + Sync>,
    /// Optional kernel registry entries for type-constraint advertisement.
    /// When non-empty, `create_ep` builds an ORT kernel registry so that ORT
    /// routes f16/bf16 (and other typed) nodes to this EP.
    pub kernel_registry_entries: Vec<crate::ep::KernelRegistryEntry>,
    /// Device support configuration for generalized enumeration.
    pub device_support: DeviceSupport,
    /// Device allocators remain factory-owned after ORT releases them. The
    /// release callback closes their allocation tables, while retaining the
    /// adapter allocation makes duplicate release and late defensive Free
    /// calls harmless until the factory itself is released.
    pub device_allocators: Mutex<HashMap<usize, Box<DeviceAllocator>>>,
    /// Optional shared EP instance for device EPs that require a single
    /// runtime/context shared across allocator, stream, and data transfer.
    /// When set, factory callbacks use this instead of calling `constructor`
    /// for each component. (Fixes defect #1: separate CUDA runtime/context.)
    ///
    /// The EP is wrapped in `Arc<Mutex<..>>` so multiple ORT callbacks can
    /// borrow it safely. The `Mutex` is only held briefly during each callback.
    pub shared_ep: Option<Arc<Mutex<Box<dyn ExecutionProvider + Send>>>>,
    /// Optional native stream handle for device EPs. Returned from
    /// `DeviceSyncStream::GetHandle`. For CUDA, this would be `cudaStream_t`.
    pub stream_handle: *mut std::os::raw::c_void,
    workspace_execution: Option<crate::workspace::OrderedWorkspace>,
    /// Custom operator domains contributed by this factory. ORT borrows these
    /// pointers until the factory is released.
    pub custom_op_domains: Vec<*mut ort::OrtCustomOpDomain>,
}

// SAFETY: The raw stream_handle pointer is only accessed from ORT callbacks
// which are single-threaded per factory. The Arc<Mutex<..>> for shared_ep is
// inherently Send+Sync. All other fields are Send+Sync by construction.
unsafe impl Send for ExportedFactory {}
unsafe impl Sync for ExportedFactory {}

/// Initialize the ORT host API from an `OrtApiBase` pointer.
///
/// Returns `(api_ptr, error_status)`. On success, `error_status` is null and
/// `api_ptr` is valid. On failure, `api_ptr` is null and `error_status` should
/// be returned to ORT (after writing `*out_num = 0`).
///
/// # Safety
///
/// `api_base` must be valid or null.
unsafe fn init_host_api(
    api_base: *const ort::OrtApiBase,
    out_num: *mut usize,
) -> Result<*const ort::OrtApi, *mut ort::OrtStatus> {
    if api_base.is_null() {
        unsafe {
            if !out_num.is_null() {
                *out_num = 0;
            }
        }
        return Err(ptr::null_mut());
    }

    let get_api = unsafe { (*api_base).GetApi };
    let api = match get_api {
        Some(get_api) => unsafe { get_api(ort::ORT_API_VERSION) },
        None => {
            unsafe {
                if !out_num.is_null() {
                    *out_num = 0;
                }
            }
            return Err(ptr::null_mut());
        }
    };

    if api.is_null() {
        let fallback_api = unsafe { (get_api.unwrap())(1) };
        if let Some(create_status) = (!fallback_api.is_null())
            .then(|| unsafe { (*fallback_api).CreateStatus })
            .flatten()
        {
            let msg = c"EP plugin requires ORT API version 28 but host does not support it. \
                       Plugin will not load (fail-closed).";
            unsafe {
                if !out_num.is_null() {
                    *out_num = 0;
                }
            }
            return Err(unsafe { create_status(ort::ORT_FAIL, msg.as_ptr()) });
        }
        unsafe {
            if !out_num.is_null() {
                *out_num = 0;
            }
        }
        return Err(ptr::null_mut());
    }

    // Ordered deliberately: the pin must be in place *before* this library
    // publishes a pointer that outlives ORT's reference to it. `set_host_api`
    // caches the host `OrtApi` in a process-global that lives in this library's
    // data segment, and every entry point reaches this line before building a
    // factory, so this is the one place that covers `export_ep_factories!`, the
    // CPU plugin's hand-written copy, and the shared-EP path alike. See
    // [`crate::pin`] for the hazard.
    //
    // The early returns above deliberately do *not* pin. On those paths the
    // library published nothing ORT can reach, and an instrumented build's
    // profile writer runs at unload rather than being left dangling, so there
    // is nothing to protect -- while pinning would strand the mapping of a
    // plugin that failed to load at all.
    crate::pin::pin_plugin_library();
    unsafe { set_host_api(api) };
    Ok(api)
}

/// Build an `ExportedFactory` with all vtable callbacks wired.
fn build_factory(
    name_cstr: CString,
    constructor: Box<dyn Fn() -> Box<dyn ExecutionProvider> + Send + Sync>,
) -> Box<ExportedFactory> {
    let vendor_cstr = CString::new("nxrt").unwrap();
    let version_cstr = CString::new("0.1.0").unwrap();

    Box::new(ExportedFactory {
        vtable: ort::OrtEpFactory {
            ort_version_supported: ort::ORT_API_VERSION,
            GetName: Some(factory_get_name),
            GetVendor: Some(factory_get_vendor),
            GetSupportedDevices: Some(factory_get_supported_devices),
            CreateEp: Some(factory_create_ep),
            ReleaseEp: Some(factory_release_ep),
            GetVendorId: Some(factory_get_vendor_id),
            GetVersion: Some(factory_get_version),
            ValidateCompiledModelCompatibilityInfo: Some(factory_validate_compiled_model),
            CreateAllocator: Some(factory_create_allocator),
            ReleaseAllocator: Some(factory_release_allocator),
            CreateDataTransfer: Some(factory_create_data_transfer),
            IsStreamAware: Some(factory_is_stream_aware),
            CreateSyncStreamForDevice: Some(factory_create_sync_stream),
            GetHardwareDeviceIncompatibilityDetails: Some(factory_get_hw_incompatibility),
            CreateExternalResourceImporterForDevice: Some(factory_create_resource_importer),
            GetNumCustomOpDomains: Some(factory_get_num_custom_op_domains),
            GetCustomOpDomains: Some(factory_get_custom_op_domains),
            InitGraphicsInterop: Some(factory_init_graphics_interop),
            DeinitGraphicsInterop: Some(factory_deinit_graphics_interop),
            // Optional, new in ORT 1.28. It only matters for EPs that publish
            // several compiled variants of one model (e.g. speed- vs
            // memory-optimised) and need to rank them. We publish a single
            // variant, so leaving this null makes ORT fall back to
            // `ValidateCompiledModelCompatibilityInfo`, which we do implement
            // and which is the correct answer for one candidate.
            SelectBestModelCandidate: None,
            // Optional in ORT 1.32. Capacity is not exposed by the generic EP
            // contract; leave this unsupported rather than report a process
            // budget as device-global capacity.
            GetDeviceMemoryCapacity: None,
        },
        name_cstr,
        vendor_cstr,
        version_cstr,
        constructor,
        kernel_registry_entries: Vec::new(),
        device_support: DeviceSupport::cpu_only(),
        device_allocators: Mutex::new(HashMap::new()),
        shared_ep: None,
        stream_handle: ptr::null_mut(),
        workspace_execution: None,
        custom_op_domains: Vec::new(),
    })
}

impl Drop for ExportedFactory {
    fn drop(&mut self) {
        let api = host_api();
        if api.is_null() {
            return;
        }
        let Some(release) = (unsafe { (*api).ReleaseCustomOpDomain }) else {
            return;
        };
        for domain in self.custom_op_domains.drain(..).rev() {
            if !domain.is_null() {
                unsafe { release(domain) };
            }
        }
    }
}

/// Attach an ORT custom-op domain to a factory returned by this crate.
///
/// # Safety
/// `factory` must be a live pointer returned by one of this module's factory
/// constructors, and `domain` must be a live ORT custom-op domain whose custom
/// op descriptors outlive the factory.
pub unsafe fn attach_custom_op_domain(
    factory: *mut ort::OrtEpFactory,
    domain: *mut ort::OrtCustomOpDomain,
) {
    if factory.is_null() || domain.is_null() {
        return;
    }
    let exported = unsafe { &mut *factory.cast::<ExportedFactory>() };
    exported.custom_op_domains.push(domain);
}

/// Write a single factory into ORT's output array.
///
/// # Safety
///
/// `out_factories` and `out_num` must be valid, `max_factories >= 1`.
unsafe fn emit_factory(
    factory: Box<ExportedFactory>,
    out_factories: *mut *mut ort::OrtEpFactory,
    max_factories: usize,
    out_num: *mut usize,
) {
    let factory_ptr = Box::into_raw(factory);
    unsafe {
        for i in 0..max_factories {
            *out_factories.add(i) = ptr::null_mut();
        }
        *out_factories = factory_ptr.cast::<ort::OrtEpFactory>();
        *out_num = 1;
    }
}

/// Implementation of `CreateEpFactories` — called by the macro-generated export.
///
/// # Safety
///
/// All pointer arguments must be valid per the ORT plugin-EP C ABI.
pub unsafe fn create_ep_factories<F>(
    api_base: *const ort::OrtApiBase,
    out_factories: *mut *mut ort::OrtEpFactory,
    max_factories: usize,
    out_num: *mut usize,
    constructor: F,
) -> *mut ort::OrtStatus
where
    F: Fn() -> Box<dyn ExecutionProvider> + Send + Sync + 'static,
{
    if unsafe { init_host_api(api_base, out_num) }.is_err() {
        return ptr::null_mut();
    }

    if max_factories == 0 || out_factories.is_null() || out_num.is_null() {
        return fail_status("CreateEpFactories: out_factories is null or max_factories is 0");
    }

    // Create a temporary EP to get the name.
    let ep = constructor();
    let name = ep.name().to_string();
    drop(ep);

    let name_cstr =
        CString::new(name.as_str()).unwrap_or_else(|_| CString::new("nxrt_ep").unwrap());

    let factory = build_factory(name_cstr, Box::new(move || constructor()));
    unsafe { emit_factory(factory, out_factories, max_factories, out_num) };
    ok_status()
}

/// Like [`create_ep_factories`] but also registers kernel-registry entries for
/// type-constraint metadata. This enables ORT to route typed nodes (e.g.
/// f16/bf16) to the EP.
///
/// # Safety
///
/// All pointer arguments must be valid per the ORT plugin-EP C ABI.
pub unsafe fn create_ep_factories_with_registry<F>(
    api_base: *const ort::OrtApiBase,
    out_factories: *mut *mut ort::OrtEpFactory,
    max_factories: usize,
    out_num: *mut usize,
    constructor: F,
    entries: Vec<crate::ep::KernelRegistryEntry>,
) -> *mut ort::OrtStatus
where
    F: Fn() -> Box<dyn ExecutionProvider> + Send + Sync + 'static,
{
    // Delegate to the standard path — it builds the factory.
    let status = unsafe {
        create_ep_factories(api_base, out_factories, max_factories, out_num, constructor)
    };
    if !status.is_null() {
        return status;
    }
    // Patch the factory's kernel_registry_entries field.
    if !out_factories.is_null() {
        let factory_ptr = unsafe { *out_factories };
        if !factory_ptr.is_null() {
            let exported = unsafe { &mut *(factory_ptr.cast::<ExportedFactory>()) };
            exported.kernel_registry_entries = entries;
        }
    }
    ok_status()
}

/// Like [`create_ep_factories_with_registry`] but also sets the device support
/// configuration for generalized device enumeration.
///
/// # Safety
///
/// All pointer arguments must be valid per the ORT plugin-EP C ABI.
pub unsafe fn create_ep_factories_with_device_support<F>(
    api_base: *const ort::OrtApiBase,
    out_factories: *mut *mut ort::OrtEpFactory,
    max_factories: usize,
    out_num: *mut usize,
    constructor: F,
    entries: Vec<crate::ep::KernelRegistryEntry>,
    support: DeviceSupport,
) -> *mut ort::OrtStatus
where
    F: Fn() -> Box<dyn ExecutionProvider> + Send + Sync + 'static,
{
    let status = unsafe {
        create_ep_factories_with_registry(
            api_base,
            out_factories,
            max_factories,
            out_num,
            constructor,
            entries,
        )
    };
    if !status.is_null() {
        return status;
    }
    if !out_factories.is_null() {
        let factory_ptr = unsafe { *out_factories };
        if !factory_ptr.is_null() {
            let exported = unsafe { &mut *(factory_ptr.cast::<ExportedFactory>()) };
            exported.device_support = support;
        }
    }
    ok_status()
}

/// Create a factory for a pre-constructed shared EP.
///
/// Unlike [`create_ep_factories_with_device_support`], this variant does **not**
/// call the constructor to read the EP name — the name is passed explicitly.
/// This avoids the S4 problem where the constructor is a panic bomb.
///
/// The shared EP is set directly on the factory; callbacks that need the EP
/// clone the `Arc` rather than extracting a raw pointer from a `MutexGuard`.
///
/// # Safety
///
/// All pointer arguments must be valid per the ORT plugin-EP C ABI.
#[allow(clippy::too_many_arguments)]
pub unsafe fn create_ep_factories_for_shared_ep(
    api_base: *const ort::OrtApiBase,
    out_factories: *mut *mut ort::OrtEpFactory,
    max_factories: usize,
    out_num: *mut usize,
    ep_name: &str,
    shared_ep: Arc<Mutex<Box<dyn ExecutionProvider + Send>>>,
    entries: Vec<crate::ep::KernelRegistryEntry>,
    support: DeviceSupport,
    stream_handle: *mut std::os::raw::c_void,
) -> *mut ort::OrtStatus {
    let api = match unsafe { init_host_api(api_base, out_num) } {
        Ok(api) => api,
        Err(status) => return status,
    };
    let _ = api;

    if max_factories == 0 || out_factories.is_null() || out_num.is_null() {
        return fail_status("CreateEpFactories: out_factories is null or max_factories is 0");
    }

    let name_cstr = CString::new(ep_name).unwrap_or_else(|_| CString::new("nxrt_ep").unwrap());

    // The constructor closure is only used by `factory_create_ep` when
    // `shared_ep` is `None`. Since we set `shared_ep`, this should never be
    // called — but fail closed with an actionable status if it is.
    let ep_name_owned = ep_name.to_string();
    let constructor: Box<dyn Fn() -> Box<dyn ExecutionProvider> + Send + Sync> =
        Box::new(move || {
            panic!(
                "EP constructor called but shared_ep should be used instead (EP: {ep_name_owned})",
            );
        });

    let mut factory = build_factory(name_cstr, constructor);
    factory.kernel_registry_entries = entries;
    factory.device_support = support;
    factory.shared_ep = Some(shared_ep);
    factory.stream_handle = stream_handle;

    unsafe { emit_factory(factory, out_factories, max_factories, out_num) };
    ok_status()
}

/// A non-synchronizing execution-domain check run once per Compute, before any
/// upload or kernel. It must not reenter Compute or allocation callbacks. Device
/// integrations use this to reject external capture without caller-owned pins.
pub type WorkspaceExecutionGuard = Arc<dyn Fn() -> Result<(), String> + Send + Sync>;

/// Register ordered use of persistent workspace for this factory's providers.
/// No workspace is allocated here. Unregistered factories still support ORT
/// step scratch, but reject nonzero persistent requests.
///
/// # Safety
/// `factory` must be a live factory returned by this module, before creating EPs.
/// For every compiled kernel, all workspace accesses (including helper streams
/// and error paths) must complete synchronously or be ordered in one execution
/// domain before subsequent uses on that compute state. Provider replacement and
/// release must settle all outstanding uses before recycling backing. If a native
/// stream is advertised, that domain must match it. Host locks, host-accessible
/// memory and nonnull stream pointers alone do not establish these guarantees.
/// If external graph capture is possible, `preflight` must reject it before any
/// submission until a real graph-owned workspace pin contract is implemented.
/// A rejected preflight retires the compute state; recreate it after correcting
/// the execution mode. Synchronous providers may omit the callback.
pub unsafe fn register_ordered_workspace_execution(
    factory: *mut ort::OrtEpFactory,
    preflight: Option<WorkspaceExecutionGuard>,
) -> *mut ort::OrtStatus {
    if factory.is_null() {
        return fail_status(
            "Register workspace execution: factory is null; register after successful factory creation and before CreateEp",
        );
    }
    let exported = unsafe { &mut *factory.cast::<ExportedFactory>() };
    exported.workspace_execution =
        Some(unsafe { crate::workspace::OrderedWorkspace::new(preflight) });
    ok_status()
}

/// Outcome of the explicit shared-EP teardown attempted by
/// [`release_ep_factory_with_teardown`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedEpTeardown {
    /// There was no shared EP (the owned/CPU path).
    NotShared,
    /// The factory held the last reference and `shutdown()` was called here.
    ShutdownCalled,
    /// `shutdown()` failed; the EP is still dropped.
    ShutdownFailed,
    /// Other surfaces (allocators, sync streams, data transfer, `OrtEp`s) were
    /// still alive, or a `Weak` handle to the shared EP was outstanding, so
    /// `shutdown()` could not run. The EP's `Drop` releases its resources once
    /// the last strong reference goes away.
    ///
    /// Both counts are reported because they are separately sufficient to block
    /// teardown: `Arc::get_mut` refuses when `strong_count > 1` **or** when any
    /// `Weak` exists, and reporting only `strong_count` produced the nonsense
    /// diagnostic "0 other reference(s) are still alive" in the weak-only case.
    ///
    /// `strong_count == 1 && weak_count == 0` is reachable only when references
    /// are being manipulated concurrently with `ReleaseEpFactory`: exclusive
    /// access is retried once before this is reported, and the diagnostic then
    /// names the race rather than a blocker that the counts do not support.
    StillReferenced {
        strong_count: usize,
        weak_count: usize,
    },
}

/// Explicitly tear down a factory's shared EP, then drop the factory.
///
/// # Shutdown semantics for shared EPs
///
/// A shared EP is reachable from four kinds of ORT-owned surface: the
/// `OrtAllocator`, the `OrtSyncStreamImpl`, the `OrtDataTransferImpl`, and one
/// `OrtEp` per session — the last through [`crate::ep::EpHandle::Shared`]. Each
/// holds an `Arc` clone, so **no single `Release*` callback may call
/// `shutdown()`**: doing so would tear down the CUDA runtime/context another
/// live session still needs. That is exactly why
/// a shared [`crate::ep::EpHandle`] never shuts down on facade release.
///
/// `ReleaseEpFactory` is the one point in the ORT lifecycle that happens after
/// every other surface has been released, so it is where explicit shutdown
/// belongs. When the factory holds the last reference we call `shutdown()`
/// here — the explicit, documented cleanup path for normal teardown. Because
/// `EpHandle::Shared` never shuts down, this is also the only place a shared
/// EP's `shutdown()` can run at all, so it cannot double-shut-down.
///
/// # Inverted teardown
///
/// When surfaces are somehow still alive (an ORT contract violation, or an
/// embedder that leaked a handle) we do **not** shut down out from under them.
/// The invariant then in force is **Drop-only**: `Arc` drops the EP when the
/// last surface releases it, and the EP's own `Drop` frees its device
/// resources. `Arc::get_mut` is also refused while any `Weak` handle exists
/// even if this factory holds the only strong reference, so the diagnostic
/// reports `weak_count` alongside `strong_count` and names whichever of the two
/// actually blocked exclusive access.
///
/// A poisoned EP mutex is recovered rather than propagated, matching
/// [`crate::ep::EpHandle::with`] and `EpRef::with_ep`: refusing to shut down
/// because some unrelated callback panicked would leak the device context.
///
/// # Audit: what else can keep EP-derived state alive
///
/// Compiled info and compute states retain their original provider. A late
/// shared compute/state owner therefore participates in the same Arc checks as
/// allocators, streams and transfers: the factory cannot shut it down early.
/// If those owners outlive the factory, the established Drop-only rule applies.
/// Owned providers instead shut down exactly once when their final facade,
/// compute-info or state keepalive is released.
///
/// # Safety
///
/// `factory` must be a pointer returned by `create_ep_factories*`.
pub unsafe fn release_ep_factory_with_teardown(
    factory: *mut ort::OrtEpFactory,
) -> (SharedEpTeardown, *mut ort::OrtStatus) {
    if factory.is_null() {
        return (SharedEpTeardown::NotShared, ok_status());
    }
    // SAFETY: The pointer was created by Box::into_raw in create_ep_factories.
    let mut exported = unsafe { Box::from_raw(factory.cast::<ExportedFactory>()) };

    // Allocators are the authority for every DeviceBuffer ORT obtained through
    // them. Close and drop them before taking the shared EP, so each remaining
    // owner is returned through `ExecutionProvider::deallocate` while its
    // context and deferred-release queue are still alive.
    let allocators = {
        let mut registered = exported
            .device_allocators
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::mem::take(&mut *registered)
    };
    drop(allocators);

    let outcome = match exported.shared_ep.take() {
        None => SharedEpTeardown::NotShared,
        Some(mut shared) => {
            // `Arc::get_mut` and the count reads below are three separate
            // atomic operations, so a concurrent owner can release *between*
            // them. Retry once when the counts read back as exclusive rather
            // than printing "1 strong, 0 weak blocked exclusive access", which
            // is self-contradictory, and would also give up a shutdown() that
            // is now legal.
            let mut retried = false;
            loop {
                if let Some(mutex) = Arc::get_mut(&mut shared) {
                    let ep = mutex
                        .get_mut()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    break match ep.shutdown() {
                        Ok(()) => SharedEpTeardown::ShutdownCalled,
                        Err(e) => {
                            eprintln!(
                                "nxrt ep plugin: shared EP shutdown() failed during \
                                 ReleaseEpFactory: {e}"
                            );
                            SharedEpTeardown::ShutdownFailed
                        }
                    };
                }
                let strong_count = Arc::strong_count(&shared);
                let weak_count = Arc::weak_count(&shared);
                if strong_count == 1 && weak_count == 0 && !retried {
                    retried = true;
                    continue;
                }
                // `Arc::get_mut` returns `None` for either reason, and the two
                // call for different follow-up, so name the one that applies.
                let blocker = if strong_count > 1 {
                    format!(
                        "{} other strong reference(s) to the shared EP are still alive \
                         (allocator / sync stream / data transfer / OrtEp)",
                        strong_count - 1
                    )
                } else if weak_count > 0 {
                    format!(
                        "the factory holds the only strong reference, but {weak_count} Weak \
                         handle(s) to the shared EP are outstanding, which is also enough to \
                         block exclusive access"
                    )
                } else {
                    // Exclusive by the counts, yet still not exclusive when
                    // asked. Another thread is manipulating references
                    // concurrently; say that, and do not invent a blocker.
                    "exclusive access was refused twice even though the counts now read \
                     strong=1 weak=0, so references are being created or released \
                     concurrently with ReleaseEpFactory"
                        .to_string()
                };
                eprintln!(
                    "nxrt ep plugin: ReleaseEpFactory called while {blocker}. ORT should release \
                     those before releasing the factory. Skipping explicit shutdown() and falling \
                     back to the Drop-only invariant: the EP is released when the last strong \
                     reference goes away. (strong_count={strong_count}, weak_count={weak_count})"
                );
                break SharedEpTeardown::StillReferenced {
                    strong_count,
                    weak_count,
                };
            }
        }
    };

    drop(exported);
    (outcome, ok_status())
}

/// Implementation of `ReleaseEpFactory`.
///
/// Thin wrapper over [`release_ep_factory_with_teardown`] that discards the
/// teardown outcome (which exists so tests can assert on it).
///
/// # Safety
///
/// `factory` must be a pointer returned by `create_ep_factories`.
pub unsafe fn release_ep_factory(factory: *mut ort::OrtEpFactory) -> *mut ort::OrtStatus {
    unsafe { release_ep_factory_with_teardown(factory) }.1
}

// ─── OrtEpFactory callbacks ─────────────────────────────────────────────────

unsafe extern "C" fn factory_get_name(factory: *const ort::OrtEpFactory) -> *const c_char {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if factory.is_null() {
            return c"unknown".as_ptr();
        }
        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };
        exported.name_cstr.as_ptr()
    }));
    result.unwrap_or(c"unknown".as_ptr())
}

unsafe extern "C" fn factory_get_vendor(factory: *const ort::OrtEpFactory) -> *const c_char {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if factory.is_null() {
            return c"unknown".as_ptr();
        }
        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };
        exported.vendor_cstr.as_ptr()
    }));
    result.unwrap_or(c"unknown".as_ptr())
}

unsafe extern "C" fn factory_get_vendor_id(factory: *const ort::OrtEpFactory) -> u32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if factory.is_null() {
            return 0;
        }
        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };
        exported.device_support.vendor_id
    }));
    result.unwrap_or(0)
}

unsafe extern "C" fn factory_get_version(factory: *const ort::OrtEpFactory) -> *const c_char {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if factory.is_null() {
            return c"0.0.0".as_ptr();
        }
        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };
        exported.version_cstr.as_ptr()
    }));
    result.unwrap_or(c"0.0.0".as_ptr())
}

unsafe extern "C" fn factory_get_supported_devices(
    factory: *mut ort::OrtEpFactory,
    in_devices: *const *const ort::OrtHardwareDevice,
    num_in: usize,
    out_devices: *mut *mut ort::OrtEpDevice,
    max_out: usize,
    out_num: *mut usize,
) -> *mut ort::OrtStatus {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        // Null/bounds checks.
        if out_num.is_null() {
            return fail_status("GetSupportedDevices: out_num is null");
        }
        // Default to zero so early returns are safe.
        unsafe { *out_num = 0 };

        if factory.is_null() || out_devices.is_null() || max_out == 0 {
            return fail_status("GetSupportedDevices: invalid arguments");
        }
        if in_devices.is_null() || num_in == 0 {
            // No hardware devices from ORT — nothing to match. Clean return.
            return ok_status();
        }

        // Get the ORT API and EP API.
        let api = host_api();
        if api.is_null() {
            return fail_status("GetSupportedDevices: host ORT API not initialized");
        }
        let hw_type_fn = unsafe { (*api).HardwareDevice_Type };
        let hw_type_fn = match hw_type_fn {
            Some(f) => f,
            None => return fail_status("GetSupportedDevices: HardwareDevice_Type not available"),
        };
        // Vendor is load-bearing, not cosmetic: hardware type alone cannot tell a
        // discrete NVIDIA GPU from the integrated Intel/AMD GPU that sits beside
        // it on most laptops. Claiming the wrong one hands a CUDA EP a device
        // that has no CUDA context, and the failure surfaces far away as a
        // synchronous memcpy that never returns (#982).
        let hw_vendor_fn = unsafe { (*api).HardwareDevice_VendorId };
        let get_ep_api = match unsafe { (*api).GetEpApi } {
            Some(f) => f,
            None => return fail_status("GetSupportedDevices: GetEpApi not available"),
        };
        let ep_api = unsafe { get_ep_api() };
        if ep_api.is_null() {
            return fail_status("GetSupportedDevices: GetEpApi returned null");
        }
        let create_ep_device = match unsafe { (*ep_api).CreateEpDevice } {
            Some(f) => f,
            None => return fail_status("GetSupportedDevices: CreateEpDevice not available"),
        };

        // Iterate input hardware devices, filter using DeviceSupport config,
        // create an OrtEpDevice for each matching one (up to max_out).
        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };
        let support = &exported.device_support;
        let mut count: usize = 0;
        for i in 0..num_in {
            if count >= max_out {
                break;
            }
            let hw_device = unsafe { *in_devices.add(i) };
            if hw_device.is_null() {
                continue;
            }
            let dev_type = unsafe { hw_type_fn(hw_device) };
            if !support.serves(dev_type) {
                continue;
            }
            // A zero `vendor_id` means "any vendor" (the CPU EP). When an EP
            // names a vendor, honour it: a CUDA EP must not claim an Intel iGPU
            // just because both report as GPU.
            if support.vendor_id != 0
                && let Some(vendor_fn) = hw_vendor_fn
                && unsafe { vendor_fn(hw_device) } != support.vendor_id
            {
                continue;
            }

            // Create an OrtEpDevice. ORT takes ownership of the returned pointer.
            // Create empty metadata and options KVPs — ORT may dereference these internally.
            let create_kvp = unsafe { (*api).CreateKeyValuePairs };
            let release_kvp = unsafe { (*api).ReleaseKeyValuePairs };
            let mut ep_metadata: *mut ort::OrtKeyValuePairs = ptr::null_mut();
            let mut ep_options: *mut ort::OrtKeyValuePairs = ptr::null_mut();
            if let Some(create_kvp) = create_kvp {
                unsafe { create_kvp(&mut ep_metadata) };
                unsafe { create_kvp(&mut ep_options) };
            }

            let mut ep_device: *mut ort::OrtEpDevice = ptr::null_mut();
            let status = unsafe {
                create_ep_device(factory, hw_device, ep_metadata, ep_options, &mut ep_device)
            };

            // Release KVPs (CreateEpDevice copies them per the doc).
            if let Some(release) = release_kvp {
                if !ep_metadata.is_null() {
                    unsafe { release(ep_metadata) };
                }
                if !ep_options.is_null() {
                    unsafe { release(ep_options) };
                }
            }

            if !status.is_null() {
                return status;
            }
            if ep_device.is_null() {
                return fail_status("GetSupportedDevices: CreateEpDevice returned null device");
            }

            // Register memory info so ORT knows how to allocate for this device.
            // Required: ORT internally accesses the device's allocator info.
            let add_alloc_info = match unsafe { (*ep_api).EpDevice_AddAllocatorInfo } {
                Some(f) => f,
                None => {
                    return fail_status(
                        "GetSupportedDevices: EpDevice_AddAllocatorInfo not available",
                    );
                }
            };

            // Use CreateMemoryInfo_V2 to produce a properly-typed OrtMemoryInfo
            // with OrtMemoryInfoDeviceType and OrtDeviceMemoryType fields that the
            // EP device system requires. The legacy CreateCpuMemoryInfo creates
            // old-format memory info whose device-type/memory-type fields are
            // uninitialized in the new EP ABI, producing garbage values.
            let create_mem_info_v2 = match unsafe { (*api).CreateMemoryInfo_V2 } {
                Some(f) => f,
                None => {
                    return fail_status("GetSupportedDevices: CreateMemoryInfo_V2 not available");
                }
            };

            // Create allocator name as a C string from the support config.
            let alloc_name = std::ffi::CString::new(support.allocator_name)
                .unwrap_or_else(|_| std::ffi::CString::new("Cpu").unwrap());
            let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
            let status = unsafe {
                create_mem_info_v2(
                    alloc_name.as_ptr(),
                    support.memory_device_type(), // device_type from support
                    support.vendor_id,            // vendor_id from support
                    0,                            // device_id
                    ort::OrtDeviceMemoryType_DEFAULT, // mem_type
                    0,                            // alignment (default)
                    ort::OrtDeviceAllocator,      // allocator_type
                    &mut mem_info,
                )
            };
            if !status.is_null() {
                return status;
            }
            if mem_info.is_null() {
                return fail_status("GetSupportedDevices: CreateMemoryInfo_V2 returned null");
            }

            // EpDevice_AddAllocatorInfo stores the OrtMemoryInfo pointer inside
            // the OrtEpDevice. ORT accesses it later (e.g. CreateAllocator,
            // EpDevice_MemoryInfo). We must NOT release the OrtMemoryInfo here —
            // ORT will release it when the OrtEpDevice is released via
            // ReleaseEpDevice. Releasing here causes a use-after-free that
            // manifests as garbage DeviceType/MemoryType after repeated
            // register/unregister cycles.
            let status = unsafe { add_alloc_info(ep_device, mem_info) };
            if !status.is_null() {
                // Release mem_info only on failure since it was not consumed.
                if let Some(release) = unsafe { (*api).ReleaseMemoryInfo } {
                    unsafe { release(mem_info) };
                }
                return status;
            }

            unsafe { *out_devices.add(count) = ep_device };
            count += 1;
        }

        unsafe { *out_num = count };
        ok_status()
    }));
    result.unwrap_or_else(|_| fail_status("GetSupportedDevices: internal panic"))
}

unsafe extern "C" fn factory_create_ep(
    factory: *mut ort::OrtEpFactory,
    _hardware: *const *const ort::OrtHardwareDevice,
    _metadata: *const *const ort::OrtKeyValuePairs,
    _num_devices: usize,
    _session_options: *const ort::OrtSessionOptions,
    _logger: *const ort::OrtLogger,
    out_ep: *mut *mut ort::OrtEp,
) -> *mut ort::OrtStatus {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if factory.is_null() || out_ep.is_null() {
            return fail_status("CreateEp: invalid arguments");
        }

        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };

        let registry_outcome = crate::ep::build_ort_kernel_registry(
            &exported.kernel_registry_entries,
            exported.name_cstr.to_str().unwrap_or("nxrt_ep"),
        );
        if !registry_outcome.failures.is_empty() {
            return fail_status(&format!(
                "CreateEp: kernel registry build had {} failure(s): {}",
                registry_outcome.failures.len(),
                registry_outcome.failures.join("; ")
            ));
        }

        // Device EPs (CUDA) advertise a shared EP: the same instance already
        // backs the factory's allocator, stream, and data transfer. `CreateEp`
        // must reuse it so the compiled graph runs on the exact CUDA context
        // ORT uses for host<->device transfers. CPU EPs construct a fresh,
        // owned instance.
        let exported_ep = if let Some(ref shared) = exported.shared_ep {
            Box::new(ExportedEp::new_shared(
                std::sync::Arc::clone(shared),
                exported.name_cstr.to_str().unwrap_or("nxrt_ep"),
                registry_outcome.registry,
                exported.kernel_registry_entries.clone(),
            ))
        } else {
            let mut ep = (exported.constructor)();
            let config = onnx_runtime_ep_api::provider::EpConfig::default();
            if let Err(e) = ep.initialize(&config) {
                return fail_status(&format!("CreateEp: EP initialization failed: {e}"));
            }
            Box::new(ExportedEp::new_with_registry_and_entries(
                ep,
                registry_outcome.registry,
                exported.kernel_registry_entries.clone(),
            ))
        };
        // Carry the factory's own placement decision into the EP: this is the
        // flag `CreateAllocator` above branches on, so a subgraph compiled by
        // this EP can trust it to say whether ORT's tensors are host-resident.
        let mut exported_ep = exported_ep;
        exported_ep.host_accessible = exported.device_support.host_accessible;
        exported_ep.workspace_execution = exported.workspace_execution.clone();

        let ep_ptr = Box::into_raw(exported_ep);
        unsafe { *out_ep = ep_ptr.cast::<ort::OrtEp>() };
        ok_status()
    }));
    result.unwrap_or_else(|_| fail_status("CreateEp: internal panic"))
}

unsafe extern "C" fn factory_release_ep(_factory: *mut ort::OrtEpFactory, ep: *mut ort::OrtEp) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ep.is_null() {
            return;
        }
        // SAFETY: ep was created by factory_create_ep via Box::into_raw.
        unsafe {
            // Release only this facade. Compiled info/state may still require
            // the original provider to retire their resources.
            drop(Box::from_raw(ep.cast::<ExportedEp>()));
        }
    }));
}

/// Creates an allocator. For CPU EPs, uses ORT's default CPU allocator.
/// For device EPs (GPU/NPU), creates a DeviceAllocator backed by the EP.
///
/// **B1 fix:** When `shared_ep` is set, the `Arc` is cloned and stored in the
/// `DeviceAllocator`. The allocator locks the `Arc<Mutex<..>>` on each use,
/// so the EP pointer is always valid while being dereferenced.
unsafe extern "C" fn factory_create_allocator(
    factory: *mut ort::OrtEpFactory,
    memory_info: *const ort::OrtMemoryInfo,
    _allocator_options: *const ort::OrtKeyValuePairs,
    allocator: *mut *mut ort::OrtAllocator,
) -> *mut ort::OrtStatus {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if allocator.is_null() {
            return fail_status("CreateAllocator: allocator output pointer is null");
        }
        // Every failure leaves a deterministic null output. This also makes a
        // constructor panic unable to expose stale caller memory as an allocator.
        unsafe { *allocator = ptr::null_mut() };
        if factory.is_null() {
            return fail_status("CreateAllocator: factory is null");
        }

        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };
        let support = &exported.device_support;

        if support.host_accessible {
            // CPU path: use ORT's built-in default allocator.
            let api = host_api();
            if api.is_null() {
                return fail_status("CreateAllocator: host API not available");
            }
            let get_default = unsafe { (*api).GetAllocatorWithDefaultOptions };
            match get_default {
                Some(f) => return unsafe { f(allocator) },
                None => {
                    return fail_status(
                        "CreateAllocator: GetAllocatorWithDefaultOptions unavailable",
                    );
                }
            }
        }

        // Device path: create a DeviceAllocator.
        if let Some(ref shared) = exported.shared_ep {
            // Clone the Arc — the allocator holds a strong reference.
            let mut dev_alloc =
                unsafe { DeviceAllocator::new_shared(Arc::clone(shared), memory_info) };
            let alloc_ptr = (&mut *dev_alloc as *mut DeviceAllocator).cast::<ort::OrtAllocator>();
            exported
                .device_allocators
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(alloc_ptr as usize, dev_alloc);
            unsafe { *allocator = alloc_ptr };
            ok_status()
        } else {
            // No shared EP and not host-accessible — construct a fresh EP.
            let mut ep = (exported.constructor)();
            let config = onnx_runtime_ep_api::provider::EpConfig::default();
            if let Err(e) = ep.initialize(&config) {
                return fail_status(&format!("CreateAllocator: EP init failed: {e}"));
            }
            let ep_ptr = Box::into_raw(ep) as *const dyn ExecutionProvider;
            let mut dev_alloc = unsafe { DeviceAllocator::new_owned(ep_ptr, memory_info) };
            let alloc_ptr = (&mut *dev_alloc as *mut DeviceAllocator).cast::<ort::OrtAllocator>();
            exported
                .device_allocators
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(alloc_ptr as usize, dev_alloc);
            unsafe { *allocator = alloc_ptr };
            ok_status()
        }
    }));
    result.unwrap_or_else(|_| fail_status("CreateAllocator: internal panic"))
}

/// Release an allocator. For default ORT allocators (CPU path), this is a no-op.
/// For device allocators, closes and drains the allocation table. The factory
/// retains the closed adapter until factory teardown for duplicate-call safety.
unsafe extern "C" fn factory_release_allocator(
    factory: *mut ort::OrtEpFactory,
    allocator: *mut ort::OrtAllocator,
) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if allocator.is_null() || factory.is_null() {
            return;
        }
        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };
        if exported.device_support.host_accessible {
            return; // ORT's default CPU allocator; the plugin does not own it.
        }

        let registered = exported
            .device_allocators
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(device_allocator) = registered.get(&(allocator as usize)) else {
            eprintln!(
                "onnx-runtime-ep-plugin: ignoring ReleaseAllocator for an unknown device \
                 allocator {allocator:p}; it was not created by this factory"
            );
            return;
        };
        if let Err(error) = device_allocator.release_tracked() {
            eprintln!("onnx-runtime-ep-plugin: ERROR: ReleaseAllocator failed: {error}");
        }
    }));
}

/// Stream-awareness derived from the factory's DeviceSupport config.
unsafe extern "C" fn factory_is_stream_aware(factory: *const ort::OrtEpFactory) -> bool {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if factory.is_null() {
            return false;
        }
        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };
        exported.device_support.stream_aware
    }));
    result.unwrap_or(false)
}

/// Creates data transfer for device EPs. CPU EPs don't need data transfer.
///
/// **B1 fix:** When `shared_ep` is set, the `Arc` is cloned and stored in the
/// `DeviceDataTransferFull`. The transfer locks the `Arc` on each use.
unsafe extern "C" fn factory_create_data_transfer(
    factory: *mut ort::OrtEpFactory,
    data_transfer: *mut *mut ort::OrtDataTransferImpl,
) -> *mut ort::OrtStatus {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if data_transfer.is_null() {
            return ok_status();
        }
        if factory.is_null() {
            unsafe { *data_transfer = ptr::null_mut() };
            return fail_status("CreateDataTransfer: factory is null");
        }

        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };
        let support = &exported.device_support;

        if support.host_accessible {
            // CPU path: no data transfer needed.
            unsafe { *data_transfer = ptr::null_mut() };
            return ok_status();
        }

        // Device path: create a DeviceDataTransferFull with ORT API access.
        let api = host_api();
        if api.is_null() {
            unsafe { *data_transfer = ptr::null_mut() };
            return fail_status(
                "CreateDataTransfer: host ORT API not available (cannot extract tensor data)",
            );
        }

        // Resolve OrtEpApi for MemoryDevice_GetDeviceType (used by CanCopy).
        let ep_api: *const ort::OrtEpApi = unsafe {
            match (*api).GetEpApi {
                Some(get_ep_api) => get_ep_api(),
                None => ptr::null(),
            }
        };

        if let Some(ref shared) = exported.shared_ep {
            // Clone the Arc — the transfer holds a strong reference.
            let transfer = unsafe {
                crate::transfer::DeviceDataTransferFull::new_shared(
                    Arc::clone(shared),
                    support.clone(),
                    api,
                    ep_api,
                )
            };
            let raw = Box::into_raw(transfer);
            unsafe { *data_transfer = raw.cast::<ort::OrtDataTransferImpl>() };
            ok_status()
        } else {
            // No shared EP — construct a fresh one.
            let mut ep = (exported.constructor)();
            let config = onnx_runtime_ep_api::provider::EpConfig::default();
            if let Err(e) = ep.initialize(&config) {
                unsafe { *data_transfer = ptr::null_mut() };
                return fail_status(&format!("CreateDataTransfer: EP init failed: {e}"));
            }
            let ep_ptr = Box::into_raw(ep) as *const dyn ExecutionProvider;
            let transfer = unsafe {
                crate::transfer::DeviceDataTransferFull::new_owned(
                    ep_ptr,
                    support.clone(),
                    api,
                    ep_api,
                )
            };
            let raw = Box::into_raw(transfer);
            unsafe { *data_transfer = raw.cast::<ort::OrtDataTransferImpl>() };
            ok_status()
        }
    }));
    result.unwrap_or_else(|_| fail_status("CreateDataTransfer: internal panic"))
}

/// Creates a sync stream. For non-stream-aware EPs (CPU), returns null (no-op).
/// For stream-aware EPs (GPU/NPU), creates a DeviceSyncStream.
///
/// **B1 fix:** When `shared_ep` is set, the `Arc` is cloned and stored in the
/// `DeviceSyncStream`. The stream locks the `Arc` on each use.
unsafe extern "C" fn factory_create_sync_stream(
    factory: *mut ort::OrtEpFactory,
    _memory_device: *const ort::OrtMemoryDevice,
    _stream_options: *const ort::OrtKeyValuePairs,
    stream: *mut *mut ort::OrtSyncStreamImpl,
) -> *mut ort::OrtStatus {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if stream.is_null() {
            return fail_status("CreateSyncStream: stream output pointer is null");
        }
        if factory.is_null() {
            unsafe { *stream = ptr::null_mut() };
            return fail_status("CreateSyncStream: factory is null");
        }

        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };
        let support = &exported.device_support;

        if !support.stream_aware {
            // Non-stream-aware EP: fail closed.
            unsafe { *stream = ptr::null_mut() };
            return fail_status(
                "CreateSyncStream: EP is not stream-aware; cannot create sync stream",
            );
        }

        let stream_handle = exported.stream_handle;

        if let Some(ref shared) = exported.shared_ep {
            // Clone the Arc — the stream holds a strong reference.
            let sync_stream = DeviceSyncStream::new_shared(Arc::clone(shared), stream_handle);
            let stream_ptr = Box::into_raw(sync_stream);
            unsafe { *stream = stream_ptr.cast::<ort::OrtSyncStreamImpl>() };
            ok_status()
        } else {
            // No shared EP — construct a fresh one.
            let mut ep = (exported.constructor)();
            let config = onnx_runtime_ep_api::provider::EpConfig::default();
            if let Err(e) = ep.initialize(&config) {
                unsafe { *stream = ptr::null_mut() };
                return fail_status(&format!("CreateSyncStream: EP init failed: {e}"));
            }
            let ep_ptr = Box::into_raw(ep) as *const dyn ExecutionProvider;
            let sync_stream = unsafe { DeviceSyncStream::new_owned(ep_ptr, stream_handle) };
            let stream_ptr = Box::into_raw(sync_stream);
            unsafe { *stream = stream_ptr.cast::<ort::OrtSyncStreamImpl>() };
            ok_status()
        }
    }));
    result.unwrap_or_else(|_| fail_status("CreateSyncStream: internal panic"))
}

/// Always compatible — no compiled model validation.
unsafe extern "C" fn factory_validate_compiled_model(
    _factory: *mut ort::OrtEpFactory,
    _devices: *const *const ort::OrtHardwareDevice,
    _num_devices: usize,
    _compatibility_info: *const c_char,
    model_compatibility: *mut ort::OrtCompiledModelCompatibility,
) -> *mut ort::OrtStatus {
    if !model_compatibility.is_null() {
        unsafe { *model_compatibility = 0 }; // Compatible
    }
    ok_status()
}

/// No hardware incompatibility details to report.
unsafe extern "C" fn factory_get_hw_incompatibility(
    _factory: *mut ort::OrtEpFactory,
    _hw: *const ort::OrtHardwareDevice,
    _details: *mut ort::OrtDeviceEpIncompatibilityDetails,
) -> *mut ort::OrtStatus {
    ok_status()
}

/// No external resource importer for CPU EP.
unsafe extern "C" fn factory_create_resource_importer(
    _factory: *mut ort::OrtEpFactory,
    _ep_device: *const ort::OrtEpDevice,
    out_importer: *mut *mut ort::OrtExternalResourceImporterImpl,
) -> *mut ort::OrtStatus {
    if !out_importer.is_null() {
        unsafe { *out_importer = ptr::null_mut() };
    }
    ok_status()
}

unsafe extern "C" fn factory_get_num_custom_op_domains(
    factory: *mut ort::OrtEpFactory,
    num_domains: *mut usize,
) -> *mut ort::OrtStatus {
    if factory.is_null() || num_domains.is_null() {
        return fail_status("GetNumCustomOpDomains: null argument");
    }
    let exported = unsafe { &*factory.cast::<ExportedFactory>() };
    unsafe { *num_domains = exported.custom_op_domains.len() };
    ok_status()
}

unsafe extern "C" fn factory_get_custom_op_domains(
    factory: *mut ort::OrtEpFactory,
    domains: *mut *mut ort::OrtCustomOpDomain,
    num_domains: usize,
) -> *mut ort::OrtStatus {
    if factory.is_null() {
        return fail_status("GetCustomOpDomains: factory is null");
    }
    let exported = unsafe { &*factory.cast::<ExportedFactory>() };
    if num_domains != exported.custom_op_domains.len() {
        return fail_status("GetCustomOpDomains: domain count mismatch");
    }
    if num_domains != 0 && domains.is_null() {
        return fail_status("GetCustomOpDomains: output array is null");
    }
    for (index, &domain) in exported.custom_op_domains.iter().enumerate() {
        unsafe { *domains.add(index) = domain };
    }
    ok_status()
}

/// Graphics interop not supported for CPU EP.
unsafe extern "C" fn factory_init_graphics_interop(
    _factory: *mut ort::OrtEpFactory,
    _ep_device: *const ort::OrtEpDevice,
    _config: *const ort::OrtGraphicsInteropConfig,
) -> *mut ort::OrtStatus {
    ok_status()
}

/// Graphics interop not supported for CPU EP.
unsafe extern "C" fn factory_deinit_graphics_interop(
    _factory: *mut ort::OrtEpFactory,
    _ep_device: *const ort::OrtEpDevice,
) -> *mut ort::OrtStatus {
    ok_status()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use onnx_runtime_ep_api::provider::{DeviceBuffer, EpConfig, Fence};
    use onnx_runtime_ep_api::{EpError, Kernel, KernelMatch, Result as EpResult};
    use onnx_runtime_ir::{DataType, DeviceId, DeviceType, Node, Shape, TensorLayout};

    use super::*;

    #[test]
    fn linear_attention_factory_excludes_schema_29_before_ort_partitioning() {
        use onnx_runtime_ep_api::abi::OrtGraphView;
        use onnx_runtime_ep_cpu::CpuExecutionProvider;
        use onnx_runtime_ir::{Attribute, Graph, GraphView, GraphViewCache, NodeId, static_shape};

        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 29);
        let x = graph.create_named_value("x", DataType::Float32, static_shape([1, 2, 4]));
        graph.add_input(x);
        let mut q = x;
        let mut nodes = Vec::new();
        for revision in [27, 29, 28] {
            let y = graph.create_named_value(
                &format!("y{revision}"),
                DataType::Float32,
                static_shape([1, 2, 4]),
            );
            let state = graph.create_named_value(
                &format!("state{revision}"),
                DataType::Float32,
                static_shape([1, 1, 4, 4]),
            );
            let mut node = onnx_runtime_ir::Node::new(
                NodeId(0),
                "LinearAttention",
                vec![Some(q), Some(x), Some(x)],
                vec![y, state],
            );
            node.version = Some(revision);
            node.attributes
                .insert("q_num_heads".into(), Attribute::Int(1));
            node.attributes
                .insert("kv_num_heads".into(), Attribute::Int(1));
            node.attributes
                .insert("update_rule".into(), Attribute::String(b"linear".to_vec()));
            nodes.push(graph.insert_node(node));
            q = y;
        }
        graph.add_output(q);
        let cache = GraphViewCache::build(&graph).unwrap();
        let view = GraphView::new(&graph, &cache);
        let ort_view = OrtGraphView::new(&view);

        let mut native = CpuExecutionProvider::new();
        native.initialize(&EpConfig::default()).unwrap();
        let native_claims = ort_view.query_capabilities(&native).unwrap();
        assert_eq!(native_claims.len(), 1);
        assert_eq!(native_claims[0].node_ids, nodes);

        let mut factory = build_factory(
            CString::new("linear_attention_scope_test").unwrap(),
            Box::new(|| Box::new(CpuExecutionProvider::for_ort_plugin())),
        );
        let factory_ptr = std::ptr::from_mut(factory.as_mut()).cast::<ort::OrtEpFactory>();
        let mut raw_ep = std::ptr::null_mut();
        // SAFETY: this test uses a live factory with no host registry to build.
        let status = unsafe {
            ((*factory_ptr).CreateEp.unwrap())(
                factory_ptr,
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                &mut raw_ep,
            )
        };
        assert!(status.is_null() && !raw_ep.is_null());
        // SAFETY: CreateEp returns a boxed ExportedEp owned by this test.
        let exported = unsafe { Box::from_raw(raw_ep.cast::<ExportedEp>()) };
        let claims = exported
            .ep
            .with(|ep| ort_view.query_capabilities(ep))
            .unwrap();
        assert_eq!(claims.len(), 2);
        assert_eq!(claims[0].node_ids, [nodes[0]]);
        assert_eq!(claims[1].node_ids, [nodes[2]]);
    }

    /// Counts `shutdown()` and `Drop` so a test can tell the explicit teardown
    /// path apart from the Drop-only fallback, and can catch a double shutdown.
    struct CountingEp {
        shutdowns: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
        shutdown_result: Result<(), &'static str>,
        initialize_result: Result<(), &'static str>,
        panic_on_shutdown: bool,
        allocations: Arc<AtomicUsize>,
        reject_allocations: Arc<std::sync::atomic::AtomicBool>,
        deallocations: Arc<AtomicUsize>,
        drains: Arc<AtomicUsize>,
    }

    impl Drop for CountingEp {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl ExecutionProvider for CountingEp {
        fn name(&self) -> &str {
            "counting_ep"
        }
        fn device_type(&self) -> DeviceType {
            DeviceType::Cuda
        }
        fn device_id(&self) -> DeviceId {
            DeviceId::cuda(0)
        }
        fn initialize(&mut self, _config: &EpConfig) -> EpResult<()> {
            self.initialize_result
                .map_err(|message| EpError::KernelFailed(message.into()))
        }
        fn shutdown(&mut self) -> EpResult<()> {
            self.shutdowns.fetch_add(1, Ordering::SeqCst);
            assert!(!self.panic_on_shutdown, "planned mock shutdown panic");
            match self.shutdown_result {
                Ok(()) => Ok(()),
                Err(msg) => Err(EpError::KernelFailed(msg.into())),
            }
        }
        fn supports_op(
            &self,
            _op: &Node,
            _opset: u64,
            _shapes: &[Shape],
            _input_dtypes: &[DataType],
            _layouts: &[TensorLayout],
        ) -> KernelMatch {
            KernelMatch::unsupported("mock")
        }
        fn get_kernel(
            &self,
            _op: &Node,
            _shapes: &[Vec<usize>],
            _opset: u64,
        ) -> EpResult<Box<dyn Kernel>> {
            Err(EpError::KernelFailed("mock".into()))
        }
        fn allocate(&self, size: usize, alignment: usize) -> EpResult<DeviceBuffer> {
            self.allocations.fetch_add(1, Ordering::SeqCst);
            if self.reject_allocations.load(Ordering::SeqCst) {
                return Err(EpError::KernelFailed(
                    "planned workspace allocation failure".into(),
                ));
            }
            let layout = std::alloc::Layout::from_size_align(size.max(1), alignment)
                .map_err(|error| EpError::KernelFailed(error.to_string()))?;
            let pointer = unsafe { std::alloc::alloc(layout) };
            if pointer.is_null() {
                return Err(EpError::OutOfMemory {
                    requested: size,
                    available: 0,
                });
            }
            Ok(unsafe {
                DeviceBuffer::from_raw_parts(pointer.cast(), DeviceId::cuda(0), size, alignment)
            })
        }
        fn deallocate(&self, buffer: DeviceBuffer) -> EpResult<()> {
            self.deallocations.fetch_add(1, Ordering::SeqCst);
            let size = buffer.len();
            let alignment = buffer.alignment();
            let pointer = buffer.into_raw();
            let layout = std::alloc::Layout::from_size_align(size.max(1), alignment)
                .map_err(|error| EpError::KernelFailed(error.to_string()))?;
            unsafe { std::alloc::dealloc(pointer.cast(), layout) };
            Ok(())
        }
        fn wait_for_deferred_releases(&self) -> EpResult<()> {
            self.drains.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn copy(&self, _s: &DeviceBuffer, _d: &mut DeviceBuffer, _n: usize) -> EpResult<()> {
            Ok(())
        }
        fn copy_async(
            &self,
            _s: &DeviceBuffer,
            _d: &mut DeviceBuffer,
            _n: usize,
        ) -> EpResult<Fence> {
            Ok(Fence::signalled())
        }
        fn sync(&self) -> EpResult<()> {
            Ok(())
        }
    }

    type SharedEp = Arc<Mutex<Box<dyn ExecutionProvider + Send>>>;

    fn counting_ep(
        shutdowns: &Arc<AtomicUsize>,
        drops: &Arc<AtomicUsize>,
        shutdown_result: Result<(), &'static str>,
    ) -> SharedEp {
        let ep: Box<dyn ExecutionProvider + Send> = Box::new(CountingEp {
            shutdowns: Arc::clone(shutdowns),
            drops: Arc::clone(drops),
            shutdown_result,
            initialize_result: Ok(()),
            panic_on_shutdown: false,
            allocations: Arc::new(AtomicUsize::new(0)),
            reject_allocations: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            deallocations: Arc::new(AtomicUsize::new(0)),
            drains: Arc::new(AtomicUsize::new(0)),
        });
        Arc::new(Mutex::new(ep))
    }

    /// Build a bare factory carrying `shared_ep`, bypassing `init_host_api`
    /// (which needs a live ORT). Returns the raw pointer
    /// `release_ep_factory_with_teardown` expects.
    fn raw_shared_factory(shared: Option<SharedEp>) -> *mut ort::OrtEpFactory {
        let mut factory = build_factory(
            CString::new("test_factory").unwrap(),
            Box::new(|| unreachable!("shared EP factories never call the constructor")),
        );
        factory.shared_ep = shared;
        Box::into_raw(factory).cast::<ort::OrtEpFactory>()
    }

    fn counting_allocator_ep(
        deallocations: &Arc<AtomicUsize>,
        drains: &Arc<AtomicUsize>,
        drops: &Arc<AtomicUsize>,
        initialize_result: Result<(), &'static str>,
    ) -> Box<dyn ExecutionProvider + Send> {
        Box::new(CountingEp {
            shutdowns: Arc::new(AtomicUsize::new(0)),
            drops: Arc::clone(drops),
            shutdown_result: Ok(()),
            initialize_result,
            panic_on_shutdown: false,
            allocations: Arc::new(AtomicUsize::new(0)),
            reject_allocations: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            deallocations: Arc::clone(deallocations),
            drains: Arc::clone(drains),
        })
    }

    #[test]
    fn release_allocator_callback_is_idempotent_and_retains_closed_adapter() {
        let deallocations = Arc::new(AtomicUsize::new(0));
        let drains = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let shared = Arc::new(Mutex::new(counting_allocator_ep(
            &deallocations,
            &drains,
            &drops,
            Ok(()),
        )));
        let raw = raw_shared_factory(Some(shared));
        unsafe { &mut *raw.cast::<ExportedFactory>() }.device_support =
            DeviceSupport::gpu("Test", 0);

        let mut allocator = ptr::null_mut();
        let status =
            unsafe { factory_create_allocator(raw, ptr::null(), ptr::null(), &mut allocator) };
        assert!(status.is_null());
        assert!(!allocator.is_null());
        let allocation = unsafe { ((*allocator).Alloc.unwrap())(allocator, 128) };
        assert!(!allocation.is_null());

        unsafe { factory_release_allocator(raw, allocator) };
        unsafe { factory_release_allocator(raw, allocator) };
        assert_eq!(deallocations.load(Ordering::SeqCst), 1);
        assert_eq!(drains.load(Ordering::SeqCst), 1);

        // The factory retains the closed adapter until its own teardown, so a
        // late defensive Free cannot use freed callback state or double-free.
        unsafe { ((*allocator).Free.unwrap())(allocator, allocation) };
        assert_eq!(deallocations.load(Ordering::SeqCst), 1);
        unsafe { release_ep_factory_with_teardown(raw) };
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failed_allocator_initialization_drops_ep_and_nulls_output() {
        let deallocations = Arc::new(AtomicUsize::new(0));
        let drains = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let constructor_deallocations = Arc::clone(&deallocations);
        let constructor_drains = Arc::clone(&drains);
        let constructor_drops = Arc::clone(&drops);
        let mut factory = build_factory(
            CString::new("failing_allocator").unwrap(),
            Box::new(move || {
                counting_allocator_ep(
                    &constructor_deallocations,
                    &constructor_drains,
                    &constructor_drops,
                    Err("injected allocator initialization failure"),
                )
            }),
        );
        factory.device_support = DeviceSupport::gpu("Test", 0);
        let raw = Box::into_raw(factory).cast::<ort::OrtEpFactory>();
        let mut output = std::ptr::dangling_mut::<ort::OrtAllocator>();

        let _status =
            unsafe { factory_create_allocator(raw, ptr::null(), ptr::null(), &mut output) };
        assert!(output.is_null(), "failure must always null the output");
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(deallocations.load(Ordering::SeqCst), 0);
        assert_eq!(drains.load(Ordering::SeqCst), 0);
        unsafe { release_ep_factory_with_teardown(raw) };
    }

    fn owned_counting_facade(
        shutdowns: &Arc<AtomicUsize>,
        drops: &Arc<AtomicUsize>,
        result: Result<(), &'static str>,
        panic_on_shutdown: bool,
    ) -> ExportedEp {
        ExportedEp::new(Box::new(CountingEp {
            shutdowns: shutdowns.clone(),
            drops: drops.clone(),
            shutdown_result: result,
            initialize_result: Ok(()),
            panic_on_shutdown,
            allocations: Arc::new(AtomicUsize::new(0)),
            reject_allocations: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            deallocations: Arc::new(AtomicUsize::new(0)),
            drains: Arc::new(AtomicUsize::new(0)),
        }))
    }

    #[test]
    fn owned_provider_keepalive_survives_facade_info_and_state_release_orders() {
        for info_first in [false, true] {
            let shutdowns = Arc::new(AtomicUsize::new(0));
            let drops = Arc::new(AtomicUsize::new(0));
            let facade = owned_counting_facade(&shutdowns, &drops, Ok(()), false);
            let provider = facade.ep.clone();
            assert_eq!(provider.name(), "counting_ep");
            let raw = Box::into_raw(Box::new(facade)).cast::<ort::OrtEp>();
            let mut info = Box::new(crate::compute::ExportedComputeInfo::new(
                Vec::new(),
                Some(provider),
            ));
            let create = info.vtable.CreateState.unwrap();
            let release = info.vtable.ReleaseState.unwrap();
            let mut states = [ptr::null_mut(); 2];
            for state in &mut states {
                assert!(unsafe { create(&mut info.vtable, ptr::null_mut(), state) }.is_null());
                assert!(!state.is_null());
            }
            if info_first {
                drop(info);
                unsafe {
                    factory_release_ep(ptr::null_mut(), raw);
                }
            } else {
                unsafe {
                    factory_release_ep(ptr::null_mut(), raw);
                }
                drop(info);
            }
            assert_eq!(shutdowns.load(Ordering::SeqCst), 0);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            unsafe {
                release(ptr::null_mut(), states[0]);
            }
            assert_eq!(
                shutdowns.load(Ordering::SeqCst),
                0,
                "second state independently retains its provider"
            );
            unsafe {
                release(ptr::null_mut(), states[1]);
            }
            assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
        }
    }

    struct DropOrderKernel {
        shutdowns: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
        saw_shutdown: Arc<AtomicUsize>,
    }
    impl Kernel for DropOrderKernel {
        fn execute(
            &self,
            _inputs: &[onnx_runtime_ep_api::TensorView],
            _outputs: &mut [onnx_runtime_ep_api::TensorMut],
        ) -> EpResult<()> {
            Ok(())
        }
    }
    impl Drop for DropOrderKernel {
        fn drop(&mut self) {
            self.saw_shutdown
                .store(self.shutdowns.load(Ordering::SeqCst), Ordering::SeqCst);
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn owned_provider_keepalive_outlives_compiled_kernel_resources() {
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let kernel_drops = Arc::new(AtomicUsize::new(0));
        let saw_shutdown = Arc::new(AtomicUsize::new(0));
        let facade = owned_counting_facade(&shutdowns, &drops, Ok(()), false);
        let provider = facade.ep.clone();
        let entry = crate::compute::CompiledKernelEntry {
            kernel: Box::new(DropOrderKernel {
                shutdowns: shutdowns.clone(),
                drops: kernel_drops.clone(),
                saw_shutdown: saw_shutdown.clone(),
            }),
            num_inputs: 0,
            num_outputs: 0,
            output_dtypes: Vec::new(),
            absent_output_slots: Default::default(),
            shape_inference: crate::compute::ShapeInference::ElementwiseBroadcast,
            input_slots: Vec::new(),
        };
        let info = crate::compute::ExportedComputeInfo::new(vec![entry], Some(provider));
        let raw = Box::into_raw(Box::new(facade)).cast::<ort::OrtEp>();
        unsafe {
            factory_release_ep(ptr::null_mut(), raw);
        }
        assert_eq!(shutdowns.load(Ordering::SeqCst), 0);
        drop(info);
        assert_eq!(kernel_drops.load(Ordering::SeqCst), 1);
        assert_eq!(saw_shutdown.load(Ordering::SeqCst), 0);
        assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn shared_provider_keepalive_retains_existing_late_factory_drop_policy() {
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let shared = counting_ep(&shutdowns, &drops, Ok(()));
        let factory = raw_shared_factory(Some(shared.clone()));
        let facade = ExportedEp::new_shared(shared, "counting_ep", None, Vec::new());
        let mut info = Box::new(crate::compute::ExportedComputeInfo::new(
            Vec::new(),
            Some(facade.ep.clone()),
        ));
        let raw = Box::into_raw(Box::new(facade)).cast::<ort::OrtEp>();
        let mut state = ptr::null_mut();
        let release = info.vtable.ReleaseState.unwrap();
        assert!(
            unsafe {
                info.vtable.CreateState.unwrap()(&mut info.vtable, ptr::null_mut(), &mut state)
            }
            .is_null()
        );
        unsafe {
            factory_release_ep(ptr::null_mut(), raw);
        }
        drop(info);
        let (outcome, status) = unsafe { release_ep_factory_with_teardown(factory) };
        assert!(matches!(outcome, SharedEpTeardown::StillReferenced { .. }));
        assert!(status.is_null());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(shutdowns.load(Ordering::SeqCst), 0);
        unsafe {
            release(ptr::null_mut(), state);
        }
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(
            shutdowns.load(Ordering::SeqCst),
            0,
            "late Shared owners retain the established Drop-only policy"
        );
    }

    #[test]
    fn owned_provider_keepalive_shutdown_failure_still_drops_once() {
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        drop(owned_counting_facade(
            &shutdowns,
            &drops,
            Err("planned final shutdown failure"),
            false,
        ));
        assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn owned_provider_keepalive_shutdown_panic_is_contained_during_unwind() {
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let facade = owned_counting_facade(&shutdowns, &drops, Ok(()), true);
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _facade = facade;
            panic!("request unwind");
        }));
        assert!(result.is_err());
        assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    struct LaneFixture {
        allocations: Arc<AtomicUsize>,
        deallocations: Arc<AtomicUsize>,
        reject: Arc<std::sync::atomic::AtomicBool>,
        provider: crate::ep::EpHandle,
    }
    impl LaneFixture {
        fn new() -> Self {
            let allocations = Arc::new(AtomicUsize::new(0));
            let deallocations = Arc::new(AtomicUsize::new(0));
            let reject = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let facade = ExportedEp::new(Box::new(CountingEp {
                shutdowns: Arc::new(AtomicUsize::new(0)),
                drops: Arc::new(AtomicUsize::new(0)),
                shutdown_result: Ok(()),
                initialize_result: Ok(()),
                panic_on_shutdown: false,
                allocations: allocations.clone(),
                deallocations: deallocations.clone(),
                reject_allocations: reject.clone(),
                drains: Arc::new(AtomicUsize::new(0)),
            }));
            Self {
                allocations,
                deallocations,
                reject,
                provider: facade.ep.clone(),
            }
        }
        fn lane(&self) -> crate::workspace::WorkspaceLane {
            // SAFETY: this fixture's CPU allocations are used synchronously.
            crate::workspace::WorkspaceLane::new(
                Some(self.provider.clone()),
                Some(unsafe { crate::workspace::OrderedWorkspace::new(None) }),
            )
        }
    }

    #[test]
    fn persistent_workspace_lane_warm_fit_and_peak_preserve_capacity_and_alignment() {
        let fixture = LaneFixture::new();
        let mut lane = fixture.lane();
        let first = lane.borrow(128, 16).unwrap();
        for _ in 0..32 {
            assert_eq!(lane.borrow(32, 8).unwrap().ptr().0, first.ptr().0);
        }
        assert_eq!(fixture.allocations.load(Ordering::SeqCst), 1);
        assert_eq!(fixture.deallocations.load(Ordering::SeqCst), 0);
        lane.borrow(16, 64).unwrap();
        for _ in 0..32 {
            lane.borrow(128, 16).unwrap();
        }
        assert_eq!(
            fixture.allocations.load(Ordering::SeqCst),
            2,
            "alignment growth must preserve peak capacity"
        );
        assert_eq!(fixture.deallocations.load(Ordering::SeqCst), 1);
        drop(lane);
        assert_eq!(fixture.deallocations.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn persistent_workspace_lane_growth_failure_consumes_old_once_and_refuses_retry() {
        let fixture = LaneFixture::new();
        let mut lane = fixture.lane();
        lane.borrow(32, 16).unwrap();
        fixture.reject.store(true, Ordering::SeqCst);
        assert!(
            lane.borrow(64, 16)
                .unwrap_err()
                .contains("planned workspace allocation failure")
        );
        assert_eq!(fixture.deallocations.load(Ordering::SeqCst), 1);
        fixture.reject.store(false, Ordering::SeqCst);
        assert!(
            lane.borrow(16, 16)
                .unwrap_err()
                .contains("recreate the session")
        );
        assert_eq!(
            fixture.allocations.load(Ordering::SeqCst),
            2,
            "failed old owner cannot be revived"
        );
        drop(lane);
        assert_eq!(
            fixture.deallocations.load(Ordering::SeqCst),
            1,
            "no duplicate release after consuming replacement"
        );
    }

    #[test]
    fn persistent_workspace_lane_requires_registration_and_real_provider() {
        let fixture = LaneFixture::new();
        let mut unknown =
            crate::workspace::WorkspaceLane::new(Some(fixture.provider.clone()), None);
        assert!(
            unknown
                .borrow(16, 16)
                .unwrap_err()
                .contains("registered execution ordering")
        );
        let mut missing = crate::workspace::WorkspaceLane::new(
            None,
            Some(unsafe { crate::workspace::OrderedWorkspace::new(None) }),
        );
        assert!(
            missing
                .borrow(16, 16)
                .unwrap_err()
                .contains("originating provider")
        );
        assert_eq!(fixture.allocations.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn persistent_workspace_lane_retired_execution_keeps_owner_until_release() {
        let fixture = LaneFixture::new();
        let mut lane = fixture.lane();
        lane.borrow(64, 16).unwrap();
        lane.retire("planned upload failure");
        assert!(
            lane.borrow(8, 8)
                .unwrap_err()
                .contains("planned upload failure")
        );
        assert_eq!(fixture.deallocations.load(Ordering::SeqCst), 0);
        drop(lane);
        assert_eq!(fixture.deallocations.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn persistent_workspace_lane_concurrent_states_never_alias() {
        let fixture = LaneFixture::new();
        let mut first = fixture.lane();
        let mut second = fixture.lane();
        let a = first.borrow(64, 16).unwrap();
        let b = second.borrow(64, 16).unwrap();
        assert_ne!(a.ptr().0, b.ptr().0);
        let lane = Arc::new(Mutex::new(first));
        let active = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let lane = lane.clone();
                let active = active.clone();
                scope.spawn(move || {
                    for _ in 0..16 {
                        let mut guard = lane.lock().unwrap();
                        let view = guard.borrow(64, 16).unwrap();
                        assert_eq!(active.fetch_add(1, Ordering::SeqCst), 0);
                        unsafe {
                            std::ptr::write_bytes(view.ptr().0.cast::<u8>(), 7, view.bytes());
                        }
                        assert_eq!(active.fetch_sub(1, Ordering::SeqCst), 1);
                    }
                });
            }
        });
        assert_eq!(fixture.allocations.load(Ordering::SeqCst), 2);
        drop(lane);
        drop(second);
        assert_eq!(fixture.deallocations.load(Ordering::SeqCst), 2);
    }

    /// Normal teardown: ORT releases every other surface first, so the factory
    /// is the last owner and must run the explicit `shutdown()` exactly once.
    #[test]
    fn releasing_the_factory_shuts_down_a_solely_owned_shared_ep() {
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let raw = raw_shared_factory(Some(counting_ep(&shutdowns, &drops, Ok(()))));

        // SAFETY: `raw` came from `raw_shared_factory`.
        let (outcome, status) = unsafe { release_ep_factory_with_teardown(raw) };
        assert_eq!(
            outcome,
            SharedEpTeardown::ShutdownCalled,
            "normal teardown must take the explicit shutdown path, not Drop-only"
        );
        assert!(status.is_null(), "release must report success");
        assert_eq!(
            shutdowns.load(Ordering::SeqCst),
            1,
            "shutdown() must run exactly once — not zero (leak) and not twice"
        );
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "the EP must also be dropped"
        );
    }

    /// Inverted teardown: a surface that outlives the factory (ORT contract
    /// violation or a leaked handle) must NOT be shut down out from under; the
    /// documented fallback is the Drop-only invariant.
    #[test]
    fn releasing_the_factory_defers_to_drop_when_surfaces_are_still_alive() {
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let shared = counting_ep(&shutdowns, &drops, Ok(()));
        // Stand-in for a still-live allocator / sync stream / OrtEp.
        let surface = Arc::clone(&shared);
        let raw = raw_shared_factory(Some(shared));

        // SAFETY: `raw` came from `raw_shared_factory`.
        let (outcome, status) = unsafe { release_ep_factory_with_teardown(raw) };
        assert_eq!(
            outcome,
            SharedEpTeardown::StillReferenced {
                strong_count: 2,
                weak_count: 0
            },
            "a shared EP with a live surface must not be shut down"
        );
        assert!(status.is_null());
        assert_eq!(
            shutdowns.load(Ordering::SeqCst),
            0,
            "shutting down here would tear down a runtime the surviving surface still uses"
        );
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "the EP must still be alive"
        );

        drop(surface);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "Drop-only invariant: releasing the last surface must release the EP"
        );
        assert_eq!(
            shutdowns.load(Ordering::SeqCst),
            0,
            "the Drop-only path must not retroactively call shutdown()"
        );
    }

    /// A `Weak` handle alone blocks `Arc::get_mut`, so teardown must report
    /// `weak_count` and must not describe the situation as "0 other
    /// reference(s) are still alive".
    ///
    /// Falsifier for the previous diagnostic: with only `strong_count`
    /// reported, this case yielded `StillReferenced { strong_count: 1 }` and
    /// printed `strong_count - 1 == 0` — a message that names no blocker at
    /// all. Deleting `weak_count` from the outcome, or computing it as
    /// `strong_count - 1`, turns this test red.
    #[test]
    fn releasing_the_factory_reports_weak_handles_as_the_blocker() {
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let shared = counting_ep(&shutdowns, &drops, Ok(()));
        // An embedder that kept a Weak (no strong reference at all).
        let weak = Arc::downgrade(&shared);
        let raw = raw_shared_factory(Some(shared));

        // SAFETY: `raw` came from `raw_shared_factory`.
        let (outcome, status) = unsafe { release_ep_factory_with_teardown(raw) };
        assert!(status.is_null());
        assert_eq!(
            outcome,
            SharedEpTeardown::StillReferenced {
                strong_count: 1,
                weak_count: 1
            },
            "a live Weak handle blocks Arc::get_mut even though the factory holds the only \
             strong reference; the outcome must say so instead of reporting one strong owner \
             and no blocker"
        );
        assert_eq!(
            shutdowns.load(Ordering::SeqCst),
            0,
            "shutdown() must not run while exclusive access is refused"
        );
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "the last strong reference went away with the factory, so Drop still ran"
        );
        assert!(
            weak.upgrade().is_none(),
            "the Weak must not be upgradable after the EP was dropped"
        );
    }

    /// `ReleaseEp` for a shared `OrtEp` must not shut the EP down; only
    /// `ReleaseEpFactory` may. Together with the test above this proves no
    /// double shutdown across the two release callbacks.
    #[test]
    fn releasing_a_shared_ort_ep_then_the_factory_shuts_down_exactly_once() {
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let shared = counting_ep(&shutdowns, &drops, Ok(()));

        // What `factory_create_ep` builds for the shared path.
        let exported_ep = Box::new(crate::ep::ExportedEp::new_shared(
            Arc::clone(&shared),
            "counting_ep",
            None,
            Vec::new(),
        ));
        let ep_ptr = Box::into_raw(exported_ep).cast::<ort::OrtEp>();
        let raw = raw_shared_factory(Some(shared));

        // ORT's release ordering: every `OrtEp` first, then the factory.
        // SAFETY: `ep_ptr` came from `Box::into_raw` above.
        unsafe { factory_release_ep(raw, ep_ptr) };
        assert_eq!(
            shutdowns.load(Ordering::SeqCst),
            0,
            "releasing one session's OrtEp must never shut down the shared EP"
        );
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "the EP must still be alive"
        );

        // SAFETY: `raw` came from `raw_shared_factory`.
        let (outcome, _status) = unsafe { release_ep_factory_with_teardown(raw) };
        assert_eq!(outcome, SharedEpTeardown::ShutdownCalled);
        assert_eq!(
            shutdowns.load(Ordering::SeqCst),
            1,
            "exactly one shutdown() across both release callbacks"
        );
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    /// A failing `shutdown()` must be reported, not swallowed — and must not
    /// prevent the EP from being dropped.
    #[test]
    fn failing_shutdown_is_reported_and_the_ep_is_still_dropped() {
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let raw = raw_shared_factory(Some(counting_ep(
            &shutdowns,
            &drops,
            Err("shutdown refused"),
        )));

        // SAFETY: `raw` came from `raw_shared_factory`.
        let (outcome, status) = unsafe { release_ep_factory_with_teardown(raw) };
        assert_eq!(outcome, SharedEpTeardown::ShutdownFailed);
        assert!(status.is_null());
        assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    /// A poisoned EP mutex must not block teardown — that would leak the
    /// device context.
    #[test]
    fn poisoned_ep_mutex_still_shuts_down() {
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let shared = counting_ep(&shutdowns, &drops, Ok(()));
        let poisoner = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _guard = shared.lock().unwrap();
            panic!("poison the shared EP mutex");
        }));
        assert!(poisoner.is_err(), "the poisoning panic must be observed");
        assert!(shared.is_poisoned(), "mutex must now be poisoned");

        let raw = raw_shared_factory(Some(shared));
        // SAFETY: `raw` came from `raw_shared_factory`.
        let (outcome, _status) = unsafe { release_ep_factory_with_teardown(raw) };
        assert_eq!(outcome, SharedEpTeardown::ShutdownCalled);
        assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    /// The owned (non-shared, e.g. CPU) path is unchanged.
    #[test]
    fn releasing_a_non_shared_factory_reports_not_shared() {
        let raw = raw_shared_factory(None);
        // SAFETY: `raw` came from `raw_shared_factory`.
        let (outcome, status) = unsafe { release_ep_factory_with_teardown(raw) };
        assert_eq!(outcome, SharedEpTeardown::NotShared);
        assert!(status.is_null());
    }

    /// A null factory pointer must be tolerated (ORT may call release twice on
    /// a failed registration).
    #[test]
    fn releasing_a_null_factory_is_a_no_op() {
        // SAFETY: a null factory is explicitly allowed.
        let (outcome, status) = unsafe { release_ep_factory_with_teardown(ptr::null_mut()) };
        assert_eq!(outcome, SharedEpTeardown::NotShared);
        assert!(status.is_null());
    }
}
