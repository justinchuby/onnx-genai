//! Factory-level conformance tests for the shared-EP surfaces (allocator,
//! sync stream, data transfer, and `CreateEp`) that the `CreateEp`
//! ownership-unification fix makes usable end-to-end.
//!
//! # Scope: CPU-host mock conformance — **NOT** GPU hardware validation
//!
//! No NVIDIA GPU is available in this environment (`nvidia-smi` is absent).
//! These tests dlopen a real upstream ONNX Runtime and drive the actual
//! `extern "C"` vtable function pointers our factory installs
//! (`CreateAllocator`, `CreateSyncStreamForDevice`, `CreateDataTransfer`,
//! `CreateEp`, and their `Release*` counterparts) — but the
//! `ExecutionProvider` behind them (`MockCudaLikeEp`) is backed by ordinary
//! host heap memory and merely *tags* itself as `DeviceType::Cuda` to
//! exercise the device-classification code paths that a real CUDA plugin
//! would hit.
//!
//! **This proves ABI wiring, single-shared-instance ownership, and
//! fail-closed classification logic. It proves nothing about real CUDA
//! correctness, performance, or device-memory behavior.** Real-GPU
//! validation remains a separate, hardware-gated task — see
//! `docs/CUDA_EP_STATUS.md` and issue #768.
//!
//! # What this proves
//!
//! 1. **`CreateEp` no longer fails for a shared EP.** This is the core
//!    regression fixed by this milestone: `factory_create_ep` previously
//!    returned an unconditional fail status whenever `shared_ep` was set,
//!    meaning no ORT session could ever execute a compiled subgraph on a
//!    shared-EP factory (e.g. CUDA) — even with real hardware present.
//! 2. **Single shared runtime/context instance.** The
//!    `Arc<Mutex<Box<dyn ExecutionProvider + Send>>>` handed to `CreateEp`
//!    (as `EpHandle::Shared`) is *the exact same allocation* used to construct
//!    the allocator, sync stream, and data transfer (verified via
//!    `Arc::ptr_eq`), and calls dispatched through each of those surfaces land
//!    in one shared call log tagged with one `instance_id`.
//! 3. **Non-null, correctly-owned stream handles**: the opaque handle set at
//!    factory-construction time round-trips unchanged through
//!    `CreateSyncStreamForDevice` → `GetHandle`, and is never freed by the
//!    adapter (ownership stays with the EP/runtime, matching a real
//!    `cudaStream_t`).
//! 4. **Correct allocation/deallocation, including size-zero.** `Alloc(0)`
//!    returns a valid non-null pointer and `Free` on it does not panic or
//!    double-free.
//! 5. **`CanCopy`/`CopyTensors` classification** using real `OrtMemoryInfo`/
//!    `OrtValue` objects obtained from genuine ORT API calls (not fabricated
//!    structs), for host→device and device→host directions.
//! 6. **`shutdown()` is correctly skipped** while the factory (and other
//!    surfaces) still hold `Arc` clones — releasing one `OrtEp` must never
//!    tear down a runtime other surfaces depend on.
//!
//! # Relationship to #832
//!
//! PR #832 fixed `CreateEp` for shared EPs and validated it on a physical H200.
//! These tests are the CPU-runnable falsifier for that fix: they fail on any
//! host if the shared-instance invariant or the release ordering regresses,
//! without needing a GPU.

/// Canonical ORT discovery lives in the `onnx-runtime-ort-testkit` crate —
/// aliased here so existing `ort_discovery::` call sites keep working.
use onnx_runtime_ort_testkit as ort_discovery;

use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use onnx_genai_ort_sys as ort;
use onnx_runtime_ep_api::{
    DeviceBuffer, EpConfig, EpError, ExecutionProvider, Fence, Kernel, KernelMatch,
    Result as EpResult,
};
use onnx_runtime_ep_plugin::device::DeviceSupport;
use onnx_runtime_ep_plugin::ep::{EpHandle, ExportedEp};
use onnx_runtime_ep_plugin::factory::{create_ep_factories_for_shared_ep, release_ep_factory};
use onnx_runtime_ir::{DataType, DeviceId, DeviceType, Node, Shape, TensorLayout};

/// Skip a test loudly when a required resource (real ORT) is missing.
/// When `NXRT_REQUIRE_ORT_TESTS=1`, panics instead of skipping.
macro_rules! skip_if_missing {
    ($resource:expr, $msg:literal) => {
        match $resource {
            Some(v) => v,
            None => {
                if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
                    panic!(
                        "NXRT_REQUIRE_ORT_TESTS=1 but required resource unavailable — {} cannot run",
                        $msg
                    );
                }
                eprintln!("\n*** SKIPPED: {} ***\n", $msg);
                return;
            }
        }
    };
}

/// Obtain the raw `OrtApiBase*` from a loaded libonnxruntime.
///
/// # Safety
/// `lib` must be a valid loaded libonnxruntime handle.
unsafe fn get_api_base(lib: &libloading::Library) -> *const ort::OrtApiBase {
    type GetApiBaseFn = unsafe extern "C" fn() -> *const ort::OrtApiBase;
    let get_api_base: libloading::Symbol<'_, GetApiBaseFn> =
        unsafe { lib.get(b"OrtGetApiBase") }.expect("OrtGetApiBase not found in libonnxruntime");
    unsafe { get_api_base() }
}

/// Obtain the `OrtApi` vtable from a loaded libonnxruntime.
///
/// # Safety
/// `lib` must be a valid loaded libonnxruntime handle.
unsafe fn get_ort_api(lib: &libloading::Library) -> *const ort::OrtApi {
    let api_base = unsafe { get_api_base(lib) };
    assert!(!api_base.is_null(), "OrtGetApiBase returned null");
    let get_api = unsafe { (*api_base).GetApi }.expect("OrtApiBase::GetApi is null");
    let api = unsafe { get_api(ort::ORT_API_VERSION) };
    assert!(
        !api.is_null(),
        "GetApi(ORT_API_VERSION={}) returned null — ORT version mismatch?",
        ort::ORT_API_VERSION
    );
    api
}

/// Assert an `OrtStatus` is null (success); panic with the error message otherwise.
///
/// # Safety
/// `api` and `status` must be valid (or null for status).
unsafe fn check_status(api: *const ort::OrtApi, status: *mut ort::OrtStatus, stage: &str) {
    if !status.is_null() {
        let get_msg = unsafe { (*api).GetErrorMessage }.expect("GetErrorMessage not in OrtApi");
        let msg_ptr = unsafe { get_msg(status) };
        let msg = if msg_ptr.is_null() {
            "(no message)".to_owned()
        } else {
            unsafe { std::ffi::CStr::from_ptr(msg_ptr) }
                .to_string_lossy()
                .into_owned()
        };
        if let Some(release) = unsafe { (*api).ReleaseStatus } {
            unsafe { release(status) };
        }
        panic!("STAGE [{stage}] FAILED: {msg}");
    }
}

// ─── Mock CUDA-*tagged* EP ───────────────────────────────────────────────────

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

/// Which method on the shared EP a given call-log entry originated from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CallKind {
    Allocate,
    Deallocate,
    Sync,
    CopyFromHost,
    CopyToHost,
}

/// A `DeviceType::Cuda`-*tagged* execution provider backed entirely by
/// ordinary host heap memory.
///
/// This is **never** valid for real CUDA usage — it exists only to drive the
/// factory's device-classification code paths (which key off
/// `DeviceSupport`/`DeviceType`, not off real hardware) without requiring a
/// GPU. Every call is appended to a shared `log` tagged with this instance's
/// `instance_id`, so tests can prove that the allocator, sync stream, data
/// transfer, and `CreateEp`-produced `OrtEp` all dispatch to the *same*
/// concrete instance.
/// Shared call log: each entry records which method was called and which
/// `MockCudaLikeEp::instance_id` it was called on.
type CallLog = Arc<Mutex<Vec<(CallKind, u64)>>>;

struct MockCudaLikeEp {
    instance_id: u64,
    log: CallLog,
    /// Set by `Drop`, so the test can confirm the mock is torn down exactly
    /// once when the last `Arc` reference is released (mirrors the real
    /// CUDA EP, whose actual resource teardown happens via `Drop` impls, not
    /// via an explicit `shutdown()` call — see `CudaExecutionProvider`).
    dropped: Arc<AtomicBool>,
}

impl Drop for MockCudaLikeEp {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

impl MockCudaLikeEp {
    fn new() -> (Self, CallLog, Arc<AtomicBool>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let dropped = Arc::new(AtomicBool::new(false));
        let instance_id = NEXT_INSTANCE_ID.fetch_add(1, Ordering::SeqCst);
        (
            Self {
                instance_id,
                log: Arc::clone(&log),
                dropped: Arc::clone(&dropped),
            },
            log,
            dropped,
        )
    }

    fn log_call(&self, kind: CallKind) {
        self.log.lock().unwrap().push((kind, self.instance_id));
    }
}

impl ExecutionProvider for MockCudaLikeEp {
    fn name(&self) -> &str {
        "mock_cuda_like_ep"
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cuda
    }

    fn device_id(&self) -> DeviceId {
        DeviceId::cuda(0)
    }

    fn initialize(&mut self, _config: &EpConfig) -> EpResult<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> EpResult<()> {
        Ok(())
    }

    fn supports_op(
        &self,
        _op: &Node,
        _opset: u64,
        _shapes: &[Shape],
        _input_dtypes: &[DataType],
        _layouts: &[TensorLayout],
    ) -> KernelMatch {
        KernelMatch::Unsupported {
            reason: "mock: no ops supported".into(),
        }
    }

    fn get_kernel(
        &self,
        _op: &Node,
        _shapes: &[Vec<usize>],
        _opset: u64,
    ) -> EpResult<Box<dyn Kernel>> {
        Err(EpError::KernelFailed("mock: no kernel".into()))
    }

    fn allocate(&self, size: usize, _alignment: usize) -> EpResult<DeviceBuffer> {
        self.log_call(CallKind::Allocate);
        // Size-zero handling: Rust's global allocator forbids a zero-size
        // `Layout`, so — like the real device allocator adapter — round up to
        // 1 byte. `deallocate` below uses the identical `size.max(1)` layout,
        // so the alloc/dealloc layouts always match exactly. The returned
        // pointer must still be non-null for size 0.
        let layout = std::alloc::Layout::from_size_align(size.max(1), 16).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(EpError::OutOfMemory {
                requested: size,
                available: 0,
            });
        }
        Ok(unsafe { DeviceBuffer::from_raw_parts(ptr.cast(), DeviceId::cuda(0), size, 16) })
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> EpResult<()> {
        self.log_call(CallKind::Deallocate);
        let ptr = buffer.as_ptr();
        let size = buffer.len();
        let _ = buffer; // DeviceBuffer has no Drop; discard the handle metadata.
        if !ptr.is_null() {
            let layout = std::alloc::Layout::from_size_align(size.max(1), 16).unwrap();
            unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
        }
        Ok(())
    }

    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> EpResult<()> {
        if size > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().cast::<u8>(),
                    dst.as_mut_ptr().cast::<u8>(),
                    size,
                );
            }
        }
        Ok(())
    }

    fn copy_async(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> EpResult<Fence> {
        self.copy(src, dst, size)?;
        Ok(Fence::signalled())
    }

    fn sync(&self) -> EpResult<()> {
        self.log_call(CallKind::Sync);
        Ok(())
    }

    fn copy_from_host(&self, src: &[u8], dst: &mut DeviceBuffer) -> EpResult<()> {
        self.log_call(CallKind::CopyFromHost);
        if !src.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    dst.as_mut_ptr().cast::<u8>(),
                    src.len(),
                );
            }
        }
        Ok(())
    }

    fn copy_to_host(&self, src: &DeviceBuffer, dst: &mut [u8]) -> EpResult<()> {
        self.log_call(CallKind::CopyToHost);
        if !dst.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().cast::<u8>(),
                    dst.as_mut_ptr(),
                    dst.len(),
                );
            }
        }
        Ok(())
    }
}

/// Build a real `OrtMemoryInfo*` tagged as CPU memory via `CreateCpuMemoryInfo`.
///
/// # Safety
/// `api` must be a valid `OrtApi*`.
unsafe fn make_cpu_memory_info(api: *const ort::OrtApi) -> *mut ort::OrtMemoryInfo {
    let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
    let status = unsafe {
        ((*api).CreateCpuMemoryInfo.unwrap())(
            ort::OrtDeviceAllocator,
            ort::OrtMemTypeDefault,
            &mut mem_info,
        )
    };
    unsafe { check_status(api, status, "CreateCpuMemoryInfo") };
    assert!(!mem_info.is_null());
    mem_info
}

/// Build a real `OrtMemoryInfo*` tagged as GPU (vendor 0x10de / NVIDIA, device
/// 0) memory via `CreateMemoryInfo_V2` — mirrors exactly what
/// `factory_get_supported_devices` does for a real device EP.
///
/// # Safety
/// `api` must be a valid `OrtApi*`.
unsafe fn make_gpu_memory_info(api: *const ort::OrtApi) -> *mut ort::OrtMemoryInfo {
    let name = CString::new("MockCuda").unwrap();
    let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
    let status = unsafe {
        ((*api).CreateMemoryInfo_V2.unwrap())(
            name.as_ptr(),
            ort::OrtMemoryInfoDeviceType_GPU,
            0x10de, // NVIDIA PCI vendor id
            0,      // device_id
            ort::OrtDeviceMemoryType_DEFAULT,
            0,
            ort::OrtDeviceAllocator,
            &mut mem_info,
        )
    };
    unsafe { check_status(api, status, "CreateMemoryInfo_V2") };
    assert!(!mem_info.is_null());
    mem_info
}

/// # Safety
/// `ep_api` must be a valid `OrtEpApi*`, `mem_info` a valid `OrtMemoryInfo*`.
unsafe fn memory_device_of(
    ep_api: *const ort::OrtEpApi,
    mem_info: *const ort::OrtMemoryInfo,
) -> *const ort::OrtMemoryDevice {
    let f = unsafe { (*ep_api).MemoryInfo_GetMemoryDevice }
        .expect("MemoryInfo_GetMemoryDevice missing");
    unsafe { f(mem_info) }
}

// ─── Test 1: minimal regression test for the CreateEp fix ───────────────────

/// The narrowest possible reproduction of the defect this milestone fixes:
/// before the fix, `factory_create_ep` unconditionally returned a fail
/// status whenever `ExportedFactory::shared_ep` was set — meaning **no**
/// ORT session could ever run inference through a shared-EP factory (e.g.
/// CUDA), even with real hardware present. This test proves `CreateEp` now
/// succeeds and hands back the exact same runtime instance used elsewhere.
#[test]
fn create_ep_succeeds_for_shared_ep_and_shares_the_runtime_instance() {
    let ort_lib_dir = skip_if_missing!(
        ort_discovery::find_ort_lib_dir(),
        "create_ep_succeeds_for_shared_ep: ORT not found; run `cargo build -p onnx-genai-ort-sys` first"
    );
    let ort_lib_path = ort_lib_dir.join(ort_discovery::ort_lib_name());
    let lib = unsafe { libloading::Library::new(&ort_lib_path) }
        .unwrap_or_else(|e| panic!("Failed to dlopen {}: {e}", ort_lib_path.display()));
    let api_base = unsafe { get_api_base(&lib) };
    assert!(!api_base.is_null());

    let (mock, _log, dropped) = MockCudaLikeEp::new();
    let mut ep_dyn: Box<dyn ExecutionProvider + Send> = Box::new(mock);
    ep_dyn.initialize(&EpConfig::default()).expect("initialize");
    let shared = Arc::new(Mutex::new(ep_dyn));
    // Keep a clone to compare identity against what `CreateEp` produces, and
    // to prove `Arc::ptr_eq` — not name matching — is the identity channel.
    let shared_for_check = Arc::clone(&shared);

    let name = "mock_cuda_like_ep";
    let mut out_factories: [*mut ort::OrtEpFactory; 1] = [ptr::null_mut()];
    let mut out_num: usize = 0;
    let status = unsafe {
        create_ep_factories_for_shared_ep(
            api_base,
            out_factories.as_mut_ptr(),
            1,
            &mut out_num,
            name,
            shared,
            Vec::new(),
            DeviceSupport::gpu("MockCuda", 0x10de),
            ptr::null_mut(), // no stream handle needed for this minimal test
        )
    };
    assert!(status.is_null(), "create_ep_factories_for_shared_ep failed");
    assert_eq!(out_num, 1);
    let factory_ptr = out_factories[0];
    assert!(!factory_ptr.is_null());

    // The regression: this call used to unconditionally fail for any shared EP.
    let mut out_ep: *mut ort::OrtEp = ptr::null_mut();
    let status = unsafe {
        ((*factory_ptr).CreateEp.unwrap())(
            factory_ptr,
            ptr::null(),
            ptr::null(),
            0,
            ptr::null(),
            ptr::null(),
            &mut out_ep,
        )
    };
    assert!(
        status.is_null(),
        "CreateEp failed for a shared EP — this is the exact regression this milestone fixes"
    );
    assert!(
        !out_ep.is_null(),
        "CreateEp returned null OrtEp* despite ok status"
    );
    eprintln!("✓ CreateEp succeeded for a shared EP (previously always failed)");

    // GetName must reflect the shared EP's name.
    let get_name = unsafe { (*out_ep).GetName }.expect("OrtEp::GetName missing");
    let name_ptr = unsafe { get_name(out_ep) };
    let got_name = unsafe { std::ffi::CStr::from_ptr(name_ptr) }.to_string_lossy();
    assert_eq!(got_name, "mock_cuda_like_ep");

    // Identity proof: the `Arc` inside the returned `ExportedEp`'s
    // `EpHandle::Shared` is the *exact same allocation* as `shared_for_check` —
    // not merely an EP with a matching name. This is possible because `ep.rs`
    // publicly exposes `ExportedEp` and its `ep` field precisely so
    // white-box tests like this one can verify the ownership invariant.
    let exported_ep = unsafe { &*(out_ep.cast::<ExportedEp>()) };
    match &exported_ep.ep {
        EpHandle::Shared(arc) => assert!(
            Arc::ptr_eq(arc, &shared_for_check),
            "CreateEp's shared EP is a DIFFERENT instance than the one shared with the \
             allocator/stream/transfer surfaces — single-runtime invariant violated"
        ),
        EpHandle::Owned(_) => panic!(
            "CreateEp built an Owned EP for a shared factory — it would run on a different \
             runtime/context than the memory ORT allocated through the factory"
        ),
    }
    eprintln!("✓ CreateEp shares the exact same shared-EP instance (EpHandle::Shared)");

    // ReleaseEp must not call shutdown() while the factory (and our local
    // `shared_for_check` clone) still hold other strong references — dropping
    // one session's OrtEp must never tear down a runtime other surfaces
    // still depend on.
    let release_ep = unsafe { (*factory_ptr).ReleaseEp }.expect("ReleaseEp missing");
    unsafe { release_ep(factory_ptr, out_ep) };
    assert!(
        !dropped.load(Ordering::SeqCst),
        "shutdown/drop must NOT happen while other Arc clones (factory, shared_for_check) are alive"
    );
    eprintln!(
        "✓ ReleaseEp correctly skipped shutdown while other surfaces still hold the shared EP"
    );

    drop(shared_for_check);
    let status = unsafe { release_ep_factory(factory_ptr) };
    assert!(status.is_null());
    assert!(
        dropped.load(Ordering::SeqCst),
        "MockCudaLikeEp must be dropped exactly once all Arc clones (factory included) are gone"
    );
    eprintln!("✓ Releasing the factory drops the last Arc reference and tears down the mock EP");
}

// ─── Test 2: allocator + stream + transfer + CreateEp, full walkthrough ─────

/// End-to-end conformance across every shared-EP surface: allocator
/// (including size-zero alloc/free), sync stream (non-null, correctly-owned
/// handle), data transfer (`CanCopy`/`CopyTensors` via real `OrtValue`s), and
/// `CreateEp`. Cross-checks the shared call log to prove every surface
/// dispatches to the one shared runtime instance.
#[test]
fn shared_ep_surfaces_all_dispatch_to_one_runtime_instance() {
    let ort_lib_dir = skip_if_missing!(
        ort_discovery::find_ort_lib_dir(),
        "shared_ep_surfaces_all_dispatch_to_one_runtime_instance: ORT not found"
    );
    let ort_lib_path = ort_lib_dir.join(ort_discovery::ort_lib_name());
    let lib = unsafe { libloading::Library::new(&ort_lib_path) }
        .unwrap_or_else(|e| panic!("Failed to dlopen {}: {e}", ort_lib_path.display()));
    let api_base = unsafe { get_api_base(&lib) };
    let api = unsafe { get_ort_api(&lib) };
    let ep_api: *const ort::OrtEpApi =
        unsafe { (*api).GetEpApi.expect("OrtApi::GetEpApi missing")() };
    assert!(!ep_api.is_null());

    let (mock, log, dropped) = MockCudaLikeEp::new();
    let instance_id = mock.instance_id;
    let mut ep_dyn: Box<dyn ExecutionProvider + Send> = Box::new(mock);
    ep_dyn.initialize(&EpConfig::default()).expect("initialize");
    let shared = Arc::new(Mutex::new(ep_dyn));
    let shared_for_check = Arc::clone(&shared);

    // A stream handle we own for the test's duration (stands in for a real
    // `cudaStream_t`). The adapter must pass it through unmodified and never
    // free it — real ownership stays with the EP/runtime.
    let stream_token: Box<u64> = Box::new(0xDEAD_BEEF_u64);
    let stream_handle_value = Box::into_raw(stream_token) as *mut c_void;

    let mut out_factories: [*mut ort::OrtEpFactory; 1] = [ptr::null_mut()];
    let mut out_num: usize = 0;
    let status = unsafe {
        create_ep_factories_for_shared_ep(
            api_base,
            out_factories.as_mut_ptr(),
            1,
            &mut out_num,
            "mock_cuda_like_ep",
            shared,
            Vec::new(),
            DeviceSupport::gpu("MockCuda", 0x10de),
            stream_handle_value,
        )
    };
    assert!(status.is_null());
    assert_eq!(out_num, 1);
    let factory_ptr = out_factories[0];
    assert!(!factory_ptr.is_null());

    // ── Allocator: alloc/free round trip, including size-zero ──────────────
    let gpu_mem_info = unsafe { make_gpu_memory_info(api) };
    let mut allocator: *mut ort::OrtAllocator = ptr::null_mut();
    let status = unsafe {
        ((*factory_ptr).CreateAllocator.unwrap())(
            factory_ptr,
            gpu_mem_info,
            ptr::null(),
            &mut allocator,
        )
    };
    assert!(status.is_null(), "CreateAllocator failed");
    assert!(!allocator.is_null());

    let alloc_fn = unsafe { (*allocator).Alloc }.expect("Alloc missing");
    let free_fn = unsafe { (*allocator).Free }.expect("Free missing");

    let p1 = unsafe { alloc_fn(allocator, 1024) };
    assert!(!p1.is_null(), "Alloc(1024) returned null");
    unsafe { free_fn(allocator, p1) };

    // Size-zero: must return a valid, non-null, uniquely-owned pointer, and
    // Free on it must not panic or double-free.
    let p0 = unsafe { alloc_fn(allocator, 0) };
    assert!(
        !p0.is_null(),
        "Alloc(0) returned null — size-zero must still yield a valid pointer"
    );
    unsafe { free_fn(allocator, p0) };
    eprintln!("✓ Allocator: Alloc/Free round-trip correct, including size-zero");

    {
        let log = log.lock().unwrap();
        let alloc_count = log
            .iter()
            .filter(|(k, id)| *k == CallKind::Allocate && *id == instance_id)
            .count();
        let dealloc_count = log
            .iter()
            .filter(|(k, id)| *k == CallKind::Deallocate && *id == instance_id)
            .count();
        assert_eq!(
            alloc_count, 2,
            "expected 2 allocate calls logged from the shared instance"
        );
        assert_eq!(
            dealloc_count, 2,
            "expected 2 deallocate calls logged from the shared instance"
        );
    }

    let release_allocator =
        unsafe { (*factory_ptr).ReleaseAllocator }.expect("ReleaseAllocator missing");
    unsafe { release_allocator(factory_ptr, allocator) };

    // ── Sync stream: non-null, correctly-owned handle + Flush ──────────────
    let mut stream_impl: *mut ort::OrtSyncStreamImpl = ptr::null_mut();
    let status = unsafe {
        ((*factory_ptr).CreateSyncStreamForDevice.unwrap())(
            factory_ptr,
            ptr::null(),
            ptr::null(),
            &mut stream_impl,
        )
    };
    assert!(status.is_null(), "CreateSyncStreamForDevice failed");
    assert!(!stream_impl.is_null());

    let get_handle = unsafe { (*stream_impl).GetHandle }.expect("GetHandle missing");
    let handle = unsafe { get_handle(stream_impl) };
    assert!(!handle.is_null(), "stream handle must be non-null");
    assert_eq!(
        handle, stream_handle_value,
        "GetHandle must return exactly the handle set at factory construction"
    );

    let flush = unsafe { (*stream_impl).Flush }.expect("Flush missing");
    let status = unsafe { flush(stream_impl) };
    assert!(status.is_null(), "Flush failed");
    eprintln!("✓ Sync stream: non-null, correctly-owned handle round-trips through GetHandle");

    let stream_release = unsafe { (*stream_impl).Release }.expect("stream Release missing");
    unsafe { stream_release(stream_impl) };

    // The stream handle itself must NOT have been freed by Release — it is
    // owned by the EP/runtime, not by the adapter. Reclaim it ourselves now.
    {
        let log = log.lock().unwrap();
        let sync_count = log
            .iter()
            .filter(|(k, id)| *k == CallKind::Sync && *id == instance_id)
            .count();
        assert_eq!(
            sync_count, 1,
            "expected exactly 1 sync() call logged from Flush"
        );
    }

    // ── Data transfer: CanCopy classification + CopyTensors via real OrtValue ──
    let mut data_transfer: *mut ort::OrtDataTransferImpl = ptr::null_mut();
    let status =
        unsafe { ((*factory_ptr).CreateDataTransfer.unwrap())(factory_ptr, &mut data_transfer) };
    assert!(status.is_null(), "CreateDataTransfer failed");
    assert!(!data_transfer.is_null());

    let cpu_mem_info = unsafe { make_cpu_memory_info(api) };
    let cpu_mem_device = unsafe { memory_device_of(ep_api, cpu_mem_info) };
    let gpu_mem_device = unsafe { memory_device_of(ep_api, gpu_mem_info) };

    let can_copy = unsafe { (*data_transfer).CanCopy }.expect("CanCopy missing");
    assert!(
        unsafe { can_copy(data_transfer, cpu_mem_device, gpu_mem_device) },
        "CanCopy(CPU -> GPU) must be true"
    );
    assert!(
        unsafe { can_copy(data_transfer, gpu_mem_device, cpu_mem_device) },
        "CanCopy(GPU -> CPU) must be true"
    );
    assert!(
        !unsafe { can_copy(data_transfer, cpu_mem_device, cpu_mem_device) },
        "CanCopy(CPU -> CPU) must be false — not this EP's responsibility"
    );
    assert!(
        unsafe { can_copy(data_transfer, gpu_mem_device, gpu_mem_device) },
        "CanCopy(GPU -> GPU, same device) must be true"
    );
    eprintln!("✓ CanCopy correctly classifies H2D/D2H/H2H/same-device directions");

    // CopyTensors: Host -> "Device" (real OrtValue objects; the "device"
    // buffer is ordinary host memory tagged with GPU memory info).
    let shape: [i64; 1] = [4];
    let mut host_src: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let mut host_src_val: *mut ort::OrtValue = ptr::null_mut();
    let status = unsafe {
        ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            cpu_mem_info,
            host_src.as_mut_ptr().cast(),
            4 * std::mem::size_of::<f32>(),
            shape.as_ptr(),
            1,
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut host_src_val,
        )
    };
    unsafe { check_status(api, status, "CreateTensorWithDataAsOrtValue(host_src)") };

    let mut device_dst_backing: [f32; 4] = [0.0; 4];
    let mut device_dst_val: *mut ort::OrtValue = ptr::null_mut();
    let status = unsafe {
        ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            gpu_mem_info,
            device_dst_backing.as_mut_ptr().cast(),
            4 * std::mem::size_of::<f32>(),
            shape.as_ptr(),
            1,
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut device_dst_val,
        )
    };
    unsafe { check_status(api, status, "CreateTensorWithDataAsOrtValue(device_dst)") };

    let copy_tensors = unsafe { (*data_transfer).CopyTensors }.expect("CopyTensors missing");
    let mut src_tensors: [*const ort::OrtValue; 1] = [host_src_val];
    let mut dst_tensors: [*mut ort::OrtValue; 1] = [device_dst_val];
    let status = unsafe {
        copy_tensors(
            data_transfer,
            src_tensors.as_mut_ptr(),
            dst_tensors.as_mut_ptr(),
            ptr::null_mut(),
            1,
        )
    };
    unsafe { check_status(api, status, "CopyTensors(H2D)") };
    assert_eq!(
        device_dst_backing, host_src,
        "H2D CopyTensors must copy the exact bytes"
    );
    eprintln!("✓ CopyTensors(H2D) copies real tensor data via copy_from_host");

    // CopyTensors: "Device" -> Host.
    let mut device_src_backing: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
    let mut device_src_val: *mut ort::OrtValue = ptr::null_mut();
    let status = unsafe {
        ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            gpu_mem_info,
            device_src_backing.as_mut_ptr().cast(),
            4 * std::mem::size_of::<f32>(),
            shape.as_ptr(),
            1,
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut device_src_val,
        )
    };
    unsafe { check_status(api, status, "CreateTensorWithDataAsOrtValue(device_src)") };

    let mut host_dst_backing: [f32; 4] = [0.0; 4];
    let mut host_dst_val: *mut ort::OrtValue = ptr::null_mut();
    let status = unsafe {
        ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            cpu_mem_info,
            host_dst_backing.as_mut_ptr().cast(),
            4 * std::mem::size_of::<f32>(),
            shape.as_ptr(),
            1,
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut host_dst_val,
        )
    };
    unsafe { check_status(api, status, "CreateTensorWithDataAsOrtValue(host_dst)") };

    let mut src_tensors: [*const ort::OrtValue; 1] = [device_src_val];
    let mut dst_tensors: [*mut ort::OrtValue; 1] = [host_dst_val];
    let status = unsafe {
        copy_tensors(
            data_transfer,
            src_tensors.as_mut_ptr(),
            dst_tensors.as_mut_ptr(),
            ptr::null_mut(),
            1,
        )
    };
    unsafe { check_status(api, status, "CopyTensors(D2H)") };
    assert_eq!(
        host_dst_backing, device_src_backing,
        "D2H CopyTensors must copy the exact bytes"
    );
    eprintln!("✓ CopyTensors(D2H) copies real tensor data via copy_to_host");

    {
        let log = log.lock().unwrap();
        let cfh = log
            .iter()
            .filter(|(k, id)| *k == CallKind::CopyFromHost && *id == instance_id)
            .count();
        let cth = log
            .iter()
            .filter(|(k, id)| *k == CallKind::CopyToHost && *id == instance_id)
            .count();
        assert_eq!(
            cfh, 1,
            "expected exactly 1 copy_from_host call from the shared instance"
        );
        assert_eq!(
            cth, 1,
            "expected exactly 1 copy_to_host call from the shared instance"
        );
    }

    // Release OrtValues + memory infos.
    let release_value = unsafe { (*api).ReleaseValue }.expect("ReleaseValue missing");
    for v in [host_src_val, device_dst_val, device_src_val, host_dst_val] {
        unsafe { release_value(v) };
    }
    let transfer_release = unsafe { (*data_transfer).Release }.expect("transfer Release missing");
    unsafe { transfer_release(data_transfer) };

    // ── CreateEp: the regression fix, verified alongside the other surfaces ──
    let mut out_ep: *mut ort::OrtEp = ptr::null_mut();
    let status = unsafe {
        ((*factory_ptr).CreateEp.unwrap())(
            factory_ptr,
            ptr::null(),
            ptr::null(),
            0,
            ptr::null(),
            ptr::null(),
            &mut out_ep,
        )
    };
    assert!(status.is_null(), "CreateEp failed for a shared EP");
    assert!(!out_ep.is_null());

    let exported_ep = unsafe { &*(out_ep.cast::<ExportedEp>()) };
    match &exported_ep.ep {
        EpHandle::Shared(arc) => assert!(
            Arc::ptr_eq(arc, &shared_for_check),
            "CreateEp's EP is not the same instance shared with allocator/stream/transfer"
        ),
        EpHandle::Owned(_) => {
            panic!("CreateEp built an Owned EP for a shared factory")
        }
    }
    eprintln!("✓ CreateEp shares the same runtime instance as allocator/stream/transfer");

    let release_ep = unsafe { (*factory_ptr).ReleaseEp }.expect("ReleaseEp missing");
    unsafe { release_ep(factory_ptr, out_ep) };
    assert!(
        !dropped.load(Ordering::SeqCst),
        "shutdown must not run while the factory + shared_for_check clones are alive"
    );

    // ── Teardown ─────────────────────────────────────────────────────────
    drop(shared_for_check);
    let status = unsafe { release_ep_factory(factory_ptr) };
    assert!(status.is_null());
    assert!(
        dropped.load(Ordering::SeqCst),
        "mock EP must be dropped once the factory releases its Arc"
    );

    // Reclaim the stream token we leaked for the adapter to hold — the
    // adapter itself never frees it (ownership stays with the EP/runtime).
    unsafe { drop(Box::from_raw(stream_handle_value as *mut u64)) };

    let release_mem_info = unsafe { (*api).ReleaseMemoryInfo }.expect("ReleaseMemoryInfo missing");
    unsafe {
        release_mem_info(cpu_mem_info);
        release_mem_info(gpu_mem_info);
    }

    eprintln!("✓ shared_ep_surfaces_all_dispatch_to_one_runtime_instance: all surfaces verified");
}
