//! Real-ORT conformance for the **shared-EP** path.
//!
//! Everything here drives genuine upstream ONNX Runtime:
//! `RegisterExecutionProviderLibrary` → `GetEpDevices` →
//! `SessionOptionsAppendExecutionProvider_V2` → `CreateSession` → `Run` →
//! `ReleaseSession` → `UnregisterExecutionProviderLibrary`. Calling our own
//! `extern "C"` vtable entries directly (as
//! `onnx-runtime-ep-plugin/tests/shared_gpu_conformance.rs` does) proves the
//! function pointers are wired; it does **not** prove ORT can build a session
//! on a shared-EP factory, partition nodes onto it, execute them correctly, or
//! tear the shared EP down exactly once. These tests do.
//!
//! # What is (and is not) proven
//!
//! The EP under test ([`onnx_runtime_ep_shared_mock_plugin::SharedMockEp`]) is
//! CPU-typed and host-memory backed, because `factory_get_supported_devices`
//! only matches hardware ORT actually enumerates — a GPU-typed mock is never
//! selected on a GPU-less host, so no session could be created at all. These
//! tests therefore prove **shared-EP ownership, workspace plumbing, routed
//! subgraph intermediates and teardown ordering**. They prove nothing about
//! CUDA correctness or device memory; see `docs/CUDA_EP_STATUS.md` and #768.
//!
//! # Counter observability
//!
//! The plugin exports `nxrt_mock_shared_ep_*` C symbols. The test `dlopen`s the
//! same cdylib path ORT loads; on every supported platform that returns the
//! already-mapped image, so both sides observe one set of statics. Holding the
//! handle for the whole test also keeps the mapping (and therefore the
//! counters) alive across `UnregisterExecutionProviderLibrary`.
//!
//! # Environment
//!
//! No env vars required. Skips loudly when ORT or the cdylib is missing;
//! `NXRT_REQUIRE_ORT_TESTS=1` turns those skips into failures.

use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{Mutex, MutexGuard};

use onnx_genai_ort_sys as ort;
use onnx_runtime_ort_testkit as testkit;

/// Factory name declared by the mock plugin (`OrtEpFactory::GetName`), which is
/// what `EpDevice_EpName` reports — *not* the registration key.
const EP_FACTORY_NAME: &str = "nxrt_shared_mock_ep";

const PLUGIN_PACKAGE: &str = "onnx-runtime-ep-shared-mock-plugin";

/// Serialises tests that register the plugin with ORT: ORT keeps per-process EP
/// device state, and concurrent register/unregister races corrupt it.
static ORT_EP_LOCK: Mutex<()> = Mutex::new(());

fn lock_ort_ep() -> MutexGuard<'static, ()> {
    ORT_EP_LOCK.lock().unwrap_or_else(|poisoned| {
        eprintln!(
            "WARNING: ORT_EP_LOCK was poisoned by a prior test panic — recovering. \
             Investigate the original failure above."
        );
        poisoned.into_inner()
    })
}

// ─── ORT plumbing ────────────────────────────────────────────────────────────

/// # Safety
/// `lib` must be a loaded libonnxruntime.
unsafe fn get_ort_api(lib: &libloading::Library) -> *const ort::OrtApi {
    type GetApiBaseFn = unsafe extern "C" fn() -> *const ort::OrtApiBase;
    let get_api_base: libloading::Symbol<'_, GetApiBaseFn> =
        unsafe { lib.get(b"OrtGetApiBase") }.expect("OrtGetApiBase not found in libonnxruntime");
    let api_base = unsafe { get_api_base() };
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

/// # Safety
/// `api` must be a valid `OrtApi`; `status` may be null.
unsafe fn check_status(api: *const ort::OrtApi, status: *mut ort::OrtStatus, stage: &str) {
    if !status.is_null() {
        let get_msg = unsafe { (*api).GetErrorMessage }.expect("GetErrorMessage not in OrtApi");
        let msg_ptr = unsafe { get_msg(status) };
        let msg = if msg_ptr.is_null() {
            "(no message)".to_owned()
        } else {
            unsafe { CStr::from_ptr(msg_ptr) }
                .to_string_lossy()
                .into_owned()
        };
        if let Some(release) = unsafe { (*api).ReleaseStatus } {
            unsafe { release(status) };
        }
        panic!("STAGE [{stage}] FAILED: {msg}");
    }
}

/// Handles required by every test; `None` means "prerequisite missing".
struct Fixture {
    _ort_lib: libloading::Library,
    plugin_lib: libloading::Library,
    api: *const ort::OrtApi,
    plugin_path: PathBuf,
}

impl Fixture {
    fn acquire(what: &str) -> Option<Self> {
        let ort_lib_path = testkit::require_or_skip(
            testkit::find_ort_lib(),
            &format!("{what}: real ORT not found (build onnx-genai-ort-sys first)"),
        )?;
        let plugin_path = testkit::require_or_skip(
            testkit::find_plugin_cdylib(PLUGIN_PACKAGE),
            &format!("{what}: {PLUGIN_PACKAGE} cdylib not found"),
        )?;

        // SAFETY: both paths point at real shared libraries produced by this
        // workspace's build.
        let ort_lib = unsafe { libloading::Library::new(&ort_lib_path) }
            .unwrap_or_else(|e| panic!("dlopen {}: {e}", ort_lib_path.display()));
        let plugin_lib = unsafe { libloading::Library::new(&plugin_path) }
            .unwrap_or_else(|e| panic!("dlopen {}: {e}", plugin_path.display()));
        let api = unsafe { get_ort_api(&ort_lib) };
        Some(Self {
            _ort_lib: ort_lib,
            plugin_lib,
            api,
            plugin_path,
        })
    }

    /// Read one of the plugin's exported `usize` counters.
    fn counter(&self, name: &str) -> usize {
        type CounterFn = unsafe extern "C" fn() -> usize;
        let mut sym = name.as_bytes().to_vec();
        sym.push(0);
        // SAFETY: the plugin exports these as `extern "C" fn() -> usize`.
        let f: libloading::Symbol<'_, CounterFn> = unsafe { self.plugin_lib.get(&sym) }
            .unwrap_or_else(|e| panic!("counter symbol {name} missing from plugin: {e}"));
        unsafe { f() }
    }

    /// Flip the mock `Add` kernel's declared workspace lifetime.
    fn set_persistent_workspace(&self, on: bool) {
        type SetFn = unsafe extern "C" fn(usize);
        // SAFETY: exported by the plugin as `extern "C" fn(usize)`.
        let f: libloading::Symbol<'_, SetFn> = unsafe {
            self.plugin_lib
                .get(b"nxrt_mock_shared_ep_set_persistent_workspace\0")
        }
        .expect("nxrt_mock_shared_ep_set_persistent_workspace missing from plugin");
        unsafe { f(usize::from(on)) }
    }

    fn reset_counters(&self) {
        type ResetFn = unsafe extern "C" fn();
        // SAFETY: exported by the plugin as `extern "C" fn()`.
        let f: libloading::Symbol<'_, ResetFn> =
            unsafe { self.plugin_lib.get(b"nxrt_mock_shared_ep_reset_counters\0") }
                .expect("nxrt_mock_shared_ep_reset_counters missing from plugin");
        unsafe { f() }
    }
}

/// Resolve a shared model fixture.
///
/// The fixtures live in `onnx-runtime-ep-cpu-plugin/tests/fixtures/` and are
/// committed as git-friendly ONNX protobuf TextFormat (`model.onnx.textproto`);
/// no binary `model.onnx` is committed. They are referenced in place rather than
/// copied, and loaded through [`create_session_on_our_ep`], which parses the
/// textproto to binary in-memory (`onnx_std::textproto::to_binary`) exactly like
/// `onnx-runtime-ep-cpu-plugin`'s `tests/common/ort_session.rs`.
fn fixture_model(name: &str) -> PathBuf {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir must have a parent")
        .join("onnx-runtime-ep-cpu-plugin/tests/fixtures")
        .join(name)
        .join("model.onnx.textproto");
    assert!(
        p.exists(),
        "missing model fixture: {} — regenerate with \
         `python3 crates/onnx-runtime-ep-cpu-plugin/tests/fixtures/generate_fixtures.py`",
        p.display()
    );
    p
}

/// Create an env and register the plugin under `reg_name`.
///
/// # Safety
/// `api` must be a valid `OrtApi`.
unsafe fn create_env_with_plugin(
    api: *const ort::OrtApi,
    log_id: &str,
    reg_name: &CStr,
    plugin_path: &Path,
) -> *mut ort::OrtEnv {
    let mut env: *mut ort::OrtEnv = ptr::null_mut();
    let logid = CString::new(log_id).unwrap();
    let status = unsafe {
        ((*api).CreateEnv.unwrap())(ort::ORT_LOGGING_LEVEL_WARNING, logid.as_ptr(), &mut env)
    };
    unsafe { check_status(api, status, "CreateEnv") };

    let plugin_c = testkit::OrtPathBuf::new(plugin_path);
    let status = unsafe {
        ((*api).RegisterExecutionProviderLibrary.unwrap())(
            env,
            reg_name.as_ptr(),
            plugin_c.as_ptr(),
        )
    };
    unsafe { check_status(api, status, "RegisterExecutionProviderLibrary") };
    env
}

/// Locate our EP among `GetEpDevices`.
///
/// # Safety
/// `api`/`env` must be valid.
unsafe fn find_our_ep_device(
    api: *const ort::OrtApi,
    env: *mut ort::OrtEnv,
) -> *const ort::OrtEpDevice {
    let mut ep_devices: *const *const ort::OrtEpDevice = ptr::null();
    let mut num_devices: usize = 0;
    let status = unsafe { ((*api).GetEpDevices.unwrap())(env, &mut ep_devices, &mut num_devices) };
    unsafe { check_status(api, status, "GetEpDevices") };

    let ep_name_fn = unsafe { (*api).EpDevice_EpName }.expect("EpDevice_EpName");
    let mut found: *const ort::OrtEpDevice = ptr::null();
    for i in 0..num_devices {
        let dev = unsafe { *ep_devices.add(i) };
        let name_ptr = unsafe { ep_name_fn(dev) };
        if name_ptr.is_null() {
            continue;
        }
        let name = unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy();
        if name == EP_FACTORY_NAME {
            found = dev;
        }
    }
    assert!(
        !found.is_null(),
        "shared-EP factory {EP_FACTORY_NAME:?} did not appear in GetEpDevices \
         ({num_devices} device(s) enumerated)"
    );
    found
}

/// Build a session bound to our EP device.
///
/// # Safety
/// All pointers must be valid; the returned session and options must be released.
unsafe fn create_session_on_our_ep(
    api: *const ort::OrtApi,
    env: *mut ort::OrtEnv,
    device: *const ort::OrtEpDevice,
    model: &Path,
) -> (*mut ort::OrtSession, *mut ort::OrtSessionOptions) {
    let mut options: *mut ort::OrtSessionOptions = ptr::null_mut();
    let status = unsafe { ((*api).CreateSessionOptions.unwrap())(&mut options) };
    unsafe { check_status(api, status, "CreateSessionOptions") };

    let devices: [*const ort::OrtEpDevice; 1] = [device];
    let status = unsafe {
        ((*api).SessionOptionsAppendExecutionProvider_V2.unwrap())(
            options,
            env,
            devices.as_ptr(),
            1,
            ptr::null(),
            ptr::null(),
            0,
        )
    };
    unsafe { check_status(api, status, "SessionOptionsAppendExecutionProvider_V2") };

    let mut session: *mut ort::OrtSession = ptr::null_mut();
    let status = if is_textproto_path(model) {
        // Fixtures are committed as ONNX TextFormat; ORT's on-disk parser only
        // understands binary protobuf, so parse to binary in-memory and load
        // through `CreateSessionFromArray` (mirrors the cpu-plugin harness'
        // `tests/common/ort_session.rs`).
        let text = std::fs::read_to_string(model)
            .unwrap_or_else(|e| panic!("read textproto fixture {model:?}: {e}"));
        let bytes = onnx_std::textproto::to_binary(&text).unwrap_or_else(|e| {
            panic!("convert textproto fixture {model:?} to binary protobuf: {e}")
        });
        unsafe {
            ((*api).CreateSessionFromArray.unwrap())(
                env,
                bytes.as_ptr() as *const std::ffi::c_void,
                bytes.len(),
                options,
                &mut session,
            )
        }
    } else {
        let model_c = testkit::OrtPathBuf::new(model);
        unsafe { ((*api).CreateSession.unwrap())(env, model_c.as_ptr(), options, &mut session) }
    };
    unsafe { check_status(api, status, "CreateSession") };
    (session, options)
}

/// Returns true if `path` is an ONNX TextFormat fixture (`*.textproto`).
fn is_textproto_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("textproto"))
        .unwrap_or(false)
}

/// Run a model whose inputs and single output are all `[1,4]` f32.
///
/// # Safety
/// `api`/`session` must be valid.
unsafe fn run_1x4(
    api: *const ort::OrtApi,
    session: *mut ort::OrtSession,
    input_names: &[&CStr],
    input_data: &[[f32; 4]],
    output_name: &CStr,
) -> [f32; 4] {
    assert_eq!(input_names.len(), input_data.len());

    let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
    let status = unsafe {
        ((*api).CreateCpuMemoryInfo.unwrap())(
            ort::OrtDeviceAllocator,
            ort::OrtMemTypeDefault,
            &mut mem_info,
        )
    };
    unsafe { check_status(api, status, "CreateCpuMemoryInfo") };

    let shape: [i64; 2] = [1, 4];
    let mut owned: Vec<[f32; 4]> = input_data.to_vec();
    let mut values: Vec<*mut ort::OrtValue> = Vec::with_capacity(owned.len());
    for buf in owned.iter_mut() {
        let mut v: *mut ort::OrtValue = ptr::null_mut();
        let status = unsafe {
            ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
                mem_info,
                buf.as_mut_ptr().cast(),
                4 * std::mem::size_of::<f32>(),
                shape.as_ptr(),
                2,
                ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &mut v,
            )
        };
        unsafe { check_status(api, status, "CreateTensorWithDataAsOrtValue") };
        values.push(v);
    }

    let name_ptrs: Vec<*const std::os::raw::c_char> =
        input_names.iter().map(|n| n.as_ptr()).collect();
    let const_values: Vec<*const ort::OrtValue> = values.iter().map(|v| *v as *const _).collect();
    let out_names = [output_name.as_ptr()];
    let mut output: *mut ort::OrtValue = ptr::null_mut();

    let status = unsafe {
        ((*api).Run.unwrap())(
            session,
            ptr::null(),
            name_ptrs.as_ptr(),
            const_values.as_ptr(),
            const_values.len(),
            out_names.as_ptr(),
            1,
            &mut output,
        )
    };
    unsafe { check_status(api, status, "Run") };
    assert!(!output.is_null(), "Run produced a null output value");

    let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
    let status = unsafe { ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr) };
    unsafe { check_status(api, status, "GetTensorMutableData") };
    let slice = unsafe { std::slice::from_raw_parts(data_ptr as *const f32, 4) };
    let result = [slice[0], slice[1], slice[2], slice[3]];

    unsafe { ((*api).ReleaseValue.unwrap())(output) };
    for v in values {
        unsafe { ((*api).ReleaseValue.unwrap())(v) };
    }
    unsafe { ((*api).ReleaseMemoryInfo.unwrap())(mem_info) };
    result
}

/// Run a `[1,4]` f32 model and return the failure message instead of panicking.
///
/// Used by the fail-closed falsifiers, where a *successful* `Run()` is the bug.
///
/// # Safety
/// `api`/`session` must be valid.
unsafe fn try_run_1x4(
    api: *const ort::OrtApi,
    session: *mut ort::OrtSession,
    input_names: &[&CStr],
    input_data: &[[f32; 4]],
    output_name: &CStr,
) -> Result<[f32; 4], String> {
    let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
    let status = unsafe {
        ((*api).CreateCpuMemoryInfo.unwrap())(
            ort::OrtDeviceAllocator,
            ort::OrtMemTypeDefault,
            &mut mem_info,
        )
    };
    unsafe { check_status(api, status, "CreateCpuMemoryInfo") };

    let shape: [i64; 2] = [1, 4];
    let mut owned: Vec<[f32; 4]> = input_data.to_vec();
    let mut values: Vec<*mut ort::OrtValue> = Vec::with_capacity(owned.len());
    for buf in owned.iter_mut() {
        let mut v: *mut ort::OrtValue = ptr::null_mut();
        let status = unsafe {
            ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
                mem_info,
                buf.as_mut_ptr().cast(),
                4 * std::mem::size_of::<f32>(),
                shape.as_ptr(),
                2,
                ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &mut v,
            )
        };
        unsafe { check_status(api, status, "CreateTensorWithDataAsOrtValue") };
        values.push(v);
    }

    let name_ptrs: Vec<*const std::os::raw::c_char> =
        input_names.iter().map(|n| n.as_ptr()).collect();
    let const_values: Vec<*const ort::OrtValue> = values.iter().map(|v| *v as *const _).collect();
    let out_names = [output_name.as_ptr()];
    let mut output: *mut ort::OrtValue = ptr::null_mut();

    let status = unsafe {
        ((*api).Run.unwrap())(
            session,
            ptr::null(),
            name_ptrs.as_ptr(),
            const_values.as_ptr(),
            const_values.len(),
            out_names.as_ptr(),
            1,
            &mut output,
        )
    };

    let result = if status.is_null() {
        assert!(!output.is_null(), "Run produced a null output value");
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let s2 = unsafe { ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr) };
        unsafe { check_status(api, s2, "GetTensorMutableData") };
        let slice = unsafe { std::slice::from_raw_parts(data_ptr as *const f32, 4) };
        Ok([slice[0], slice[1], slice[2], slice[3]])
    } else {
        let get_msg = unsafe { (*api).GetErrorMessage }.expect("GetErrorMessage");
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
        Err(msg)
    };

    if !output.is_null() {
        unsafe { ((*api).ReleaseValue.unwrap())(output) };
    }
    for v in values {
        unsafe { ((*api).ReleaseValue.unwrap())(v) };
    }
    unsafe { ((*api).ReleaseMemoryInfo.unwrap())(mem_info) };
    result
}

fn assert_close(got: [f32; 4], want: [f32; 4], label: &str) {
    for i in 0..4 {
        assert!(
            (got[i] - want[i]).abs() < 1e-6,
            "{label}: output[{i}] = {}, want {} (full: {got:?} vs {want:?})",
            got[i],
            want[i]
        );
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// End-to-end proof that a shared-EP factory yields a usable ORT session, and
/// that the executor honours the workspace contract.
///
/// The mock `Add` kernel's `execute()` always fails, so a correct result is
/// only obtainable via `workspace_requirement` + `execute_with_workspace`.
#[test]
fn shared_ep_session_runs_and_workspace_is_plumbed() {
    let _lock = lock_ort_ep();
    let Some(fx) = Fixture::acquire("shared_ep_session_runs_and_workspace_is_plumbed") else {
        return;
    };
    fx.reset_counters();
    let api = fx.api;
    let model = fixture_model("add_1x4");
    let reg_name = CString::new("shared_mock_ws").unwrap();

    unsafe {
        let env = create_env_with_plugin(api, "nxrt_shared_ws", &reg_name, &fx.plugin_path);
        let device = find_our_ep_device(api, env);
        let (session, options) = create_session_on_our_ep(api, env, device, &model);

        let got = run_1x4(
            api,
            session,
            &[c"X", c"Y"],
            &[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            c"Z",
        );
        assert_close(got, [6.0, 8.0, 10.0, 12.0], "add_1x4");

        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_execute_without_workspace"),
            0,
            "executor called Kernel::execute() instead of execute_with_workspace() — \
             workspace plumbing regressed"
        );
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_workspace_missing"),
            0,
            "executor passed None despite a non-zero workspace_requirement"
        );
        assert!(
            fx.counter("nxrt_mock_shared_ep_workspace_ok") >= 1,
            "no dispatch received a valid workspace"
        );
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_alloc_calls"),
            0,
            "workspace must come from ORT scratch (KernelContext_GetScratchBuffer), not from a \
             per-dispatch ExecutionProvider::allocate/deallocate pair — that pair synchronises \
             the device and is illegal during CUDA-graph capture"
        );
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_alloc_live"),
            0,
            "EP allocations outlived the Run() that made them — per-call leak"
        );

        ((*api).ReleaseSession.unwrap())(session);
        ((*api).ReleaseSessionOptions.unwrap())(options);
        let status = ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_name.as_ptr());
        check_status(api, status, "UnregisterExecutionProviderLibrary");
        ((*api).ReleaseEnv.unwrap())(env);
    }
}

/// A fused, routed multi-node subgraph must produce correct values, which
/// requires correctly allocated intermediates threaded between nodes.
///
/// `chain_add_mul` is `T = (A + B) * C + D`; both `Add` nodes go through the
/// workspace-requiring kernel and the `Mul` node through the zero-workspace
/// one, so a single Run covers both halves of the contract plus two
/// intermediates.
#[test]
fn shared_ep_routed_subgraph_intermediates_come_from_ort_scratch() {
    let _lock = lock_ort_ep();
    let Some(fx) =
        Fixture::acquire("shared_ep_routed_subgraph_intermediates_come_from_ort_scratch")
    else {
        return;
    };
    fx.reset_counters();
    let api = fx.api;
    let model = fixture_model("chain_add_mul");
    let reg_name = CString::new("shared_mock_chain").unwrap();

    unsafe {
        let env = create_env_with_plugin(api, "nxrt_shared_chain", &reg_name, &fx.plugin_path);
        let device = find_our_ep_device(api, env);
        let (session, options) = create_session_on_our_ep(api, env, device, &model);

        let got = run_1x4(
            api,
            session,
            &[c"A", c"B", c"C", c"D"],
            &[
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            c"T",
        );
        assert_close(got, [4.0, 6.0, 8.0, 10.0], "chain_add_mul");

        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_execute_without_workspace"),
            0,
            "routed path bypassed execute_with_workspace()"
        );
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_workspace_missing"),
            0,
            "routed path passed None despite a non-zero workspace_requirement"
        );
        assert!(
            fx.counter("nxrt_mock_shared_ep_workspace_ok") >= 2,
            "expected both Add nodes to receive a workspace, saw {}",
            fx.counter("nxrt_mock_shared_ep_workspace_ok")
        );
        assert!(
            fx.counter("nxrt_mock_shared_ep_mul_executed") >= 1,
            "the zero-workspace Mul kernel never ran — the subgraph was not fused onto our EP"
        );
        // Both workspaces and both routed intermediates come from ORT scratch.
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_alloc_calls"),
            0,
            "routed workspaces/intermediates must come from ORT scratch, not from per-dispatch \
             ExecutionProvider::allocate/deallocate"
        );
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_alloc_calls"),
            fx.counter("nxrt_mock_shared_ep_dealloc_calls"),
            "every EP allocation made during Run() must be freed"
        );
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_alloc_live"),
            0,
            "routed intermediates leaked"
        );

        ((*api).ReleaseSession.unwrap())(session);
        ((*api).ReleaseSessionOptions.unwrap())(options);
        let status = ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_name.as_ptr());
        check_status(api, status, "UnregisterExecutionProviderLibrary");
        ((*api).ReleaseEnv.unwrap())(env);
    }
}

/// Two independent ORT sessions built on one registered shared-EP factory must
/// share a single `ExecutionProvider` instance and both execute correctly.
///
/// `CreateEpFactories` constructs exactly one `SharedMockEp` per registration,
/// and `create_ep_factories_for_shared_ep` installs a constructor closure that
/// *panics* if ORT ever asks for a fresh EP — so an instance delta of 1 across
/// two sessions is direct evidence the `Arc` was shared, not re-created.
#[test]
fn shared_ep_two_sessions_share_one_instance() {
    let _lock = lock_ort_ep();
    let Some(fx) = Fixture::acquire("shared_ep_two_sessions_share_one_instance") else {
        return;
    };
    fx.reset_counters();
    let api = fx.api;
    let model = fixture_model("add_1x4");
    let reg_name = CString::new("shared_mock_two").unwrap();

    let created_before = fx.counter("nxrt_mock_shared_ep_instances_created");
    let live_before = fx.counter("nxrt_mock_shared_ep_instances_live");

    unsafe {
        let env = create_env_with_plugin(api, "nxrt_shared_two", &reg_name, &fx.plugin_path);
        let device = find_our_ep_device(api, env);

        let (session_a, options_a) = create_session_on_our_ep(api, env, device, &model);
        let (session_b, options_b) = create_session_on_our_ep(api, env, device, &model);

        let got_a = run_1x4(
            api,
            session_a,
            &[c"X", c"Y"],
            &[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            c"Z",
        );
        let got_b = run_1x4(
            api,
            session_b,
            &[c"X", c"Y"],
            &[[10.0, 20.0, 30.0, 40.0], [1.0, 1.0, 1.0, 1.0]],
            c"Z",
        );
        assert_close(got_a, [6.0, 8.0, 10.0, 12.0], "session A");
        assert_close(got_b, [11.0, 21.0, 31.0, 41.0], "session B");

        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_instances_created") - created_before,
            1,
            "two sessions on one registered shared-EP factory must not construct \
             more than one ExecutionProvider"
        );
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_alloc_live"),
            0,
            "allocations leaked across two sessions"
        );

        ((*api).ReleaseSession.unwrap())(session_a);
        ((*api).ReleaseSession.unwrap())(session_b);
        ((*api).ReleaseSessionOptions.unwrap())(options_a);
        ((*api).ReleaseSessionOptions.unwrap())(options_b);
        let status = ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_name.as_ptr());
        check_status(api, status, "UnregisterExecutionProviderLibrary");
        ((*api).ReleaseEnv.unwrap())(env);
    }

    assert_eq!(
        fx.counter("nxrt_mock_shared_ep_instances_live"),
        live_before,
        "the shared EP was not dropped after unregistering the library"
    );
}

/// Real ORT release ordering must drive the shared EP's explicit teardown
/// exactly once, and only after every dependent surface is gone.
///
/// This is the regression guard for the shared-EP shutdown semantics: because
/// `factory_release_ep` can never be the last owner of a *shared* EP, the
/// explicit `shutdown()` has to happen in `ReleaseEpFactory` — which ORT calls
/// from `UnregisterExecutionProviderLibrary`, after sessions, allocators,
/// streams and `OrtEp`s are released. Releasing a session must **not** shut the
/// runtime down; unregistering must.
#[test]
fn shared_ep_shutdown_runs_once_at_library_unregister() {
    let _lock = lock_ort_ep();
    let Some(fx) = Fixture::acquire("shared_ep_shutdown_runs_once_at_library_unregister") else {
        return;
    };
    fx.reset_counters();
    let api = fx.api;
    let model = fixture_model("add_1x4");
    let reg_name = CString::new("shared_mock_shutdown").unwrap();

    let shutdowns_before = fx.counter("nxrt_mock_shared_ep_shutdown_calls");
    let live_before = fx.counter("nxrt_mock_shared_ep_instances_live");

    unsafe {
        let env = create_env_with_plugin(api, "nxrt_shared_shutdown", &reg_name, &fx.plugin_path);
        let device = find_our_ep_device(api, env);
        let (session, options) = create_session_on_our_ep(api, env, device, &model);

        let got = run_1x4(
            api,
            session,
            &[c"X", c"Y"],
            &[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            c"Z",
        );
        assert_close(got, [6.0, 8.0, 10.0, 12.0], "add_1x4");

        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_shutdown_calls"),
            shutdowns_before,
            "shutdown() must not run while a session is alive"
        );

        ((*api).ReleaseSession.unwrap())(session);
        ((*api).ReleaseSessionOptions.unwrap())(options);

        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_shutdown_calls"),
            shutdowns_before,
            "releasing a session must not tear down a shared EP other surfaces may still use"
        );

        let status = ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_name.as_ptr());
        check_status(api, status, "UnregisterExecutionProviderLibrary");

        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_shutdown_calls"),
            shutdowns_before + 1,
            "UnregisterExecutionProviderLibrary must drive exactly one explicit \
             shared-EP shutdown() via ReleaseEpFactory"
        );
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_instances_live"),
            live_before,
            "the shared EP must be dropped once the factory is released"
        );

        ((*api).ReleaseEnv.unwrap())(env);

        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_shutdown_calls"),
            shutdowns_before + 1,
            "shutdown() ran more than once"
        );
    }
}

/// `WorkspaceLifetime::SessionPersistent` must be **declined** (`None`), never
/// served from the step-scoped ORT scratch this executor can actually provide.
///
/// ORT reclaims `KernelContext_GetScratchBuffer` memory when `Compute` returns,
/// so handing that block to a kernel that asked for session-persistent scratch
/// gives it memory recycled behind its back on the next `Run` — a silent
/// correctness bug on the second decode step, not a loud one.
///
/// Declining is also the behaviour `main` (#832, H200-validated) has today: the
/// executor called bare `Kernel::execute`, which for a persistent declarer such
/// as `GroupQueryAttention` is exactly `execute_with_workspace(.., None)` and
/// routes it to its own pooled scratch. Hard-failing instead would turn every
/// GQA-bearing model into a plugin-path error on hardware that runs it today,
/// so the contract is: decline, never downgrade, and let a kernel that truly
/// cannot cope fail closed itself.
///
/// Falsifier: `persistent_downgraded` must be `0` (nobody handed over recycled
/// memory) while `persistent_declined` must be `> 0` (the path was actually
/// exercised, so the test cannot pass vacuously).
#[test]
fn session_persistent_workspace_is_declined_not_downgraded() {
    let _lock = lock_ort_ep();
    let Some(fx) = Fixture::acquire("session_persistent_workspace_is_declined") else {
        return;
    };
    fx.reset_counters();
    fx.set_persistent_workspace(true);
    let api = fx.api;
    let model = fixture_model("add_1x4");
    let reg_name = CString::new("shared_mock_persistent").unwrap();

    unsafe {
        let env = create_env_with_plugin(api, "nxrt_shared_persistent", &reg_name, &fx.plugin_path);
        let device = find_our_ep_device(api, env);
        let (session, options) = create_session_on_our_ep(api, env, device, &model);

        let outcome = try_run_1x4(
            api,
            session,
            &[c"X", c"Y"],
            &[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            c"Z",
        );

        match outcome {
            Ok(values) => assert_close(values, [6.0, 8.0, 10.0, 12.0], "add_1x4 (persistent)"),
            Err(msg) => panic!(
                "Run() failed for a SessionPersistent request ({msg}). Declining must route the \
                 kernel to its own self-owned scratch, not break a model that runs on main"
            ),
        }

        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_persistent_downgraded"),
            0,
            "the executor served a SessionPersistent request from step-scoped ORT scratch — that \
             block is recycled when Compute returns"
        );
        assert!(
            fx.counter("nxrt_mock_shared_ep_persistent_declined") >= 1,
            "the persistent path was never exercised, so this test proves nothing"
        );
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_execute_without_workspace"),
            0,
            "declining a workspace must still go through execute_with_workspace()"
        );
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_workspace_ok"),
            0,
            "no workspace may have been served while the persistent request was in force"
        );
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_alloc_calls"),
            0,
            "declining must not fall back to per-dispatch EP allocate/free"
        );

        ((*api).ReleaseSession.unwrap())(session);
        ((*api).ReleaseSessionOptions.unwrap())(options);
        let status = ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_name.as_ptr());
        check_status(api, status, "UnregisterExecutionProviderLibrary");
        ((*api).ReleaseEnv.unwrap())(env);
    }
    fx.set_persistent_workspace(false);
}

/// Run a `[rows,4]` f32 `Add` model on a dynamic-batch fixture and return the
/// flattened output.
///
/// # Safety
/// `api`/`session` must be valid.
unsafe fn run_rows_x4(
    api: *const ort::OrtApi,
    session: *mut ort::OrtSession,
    input_names: &[&CStr],
    input_data: &[Vec<f32>],
    rows: usize,
    output_name: &CStr,
) -> Vec<f32> {
    assert_eq!(input_names.len(), input_data.len());
    let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
    let status = unsafe {
        ((*api).CreateCpuMemoryInfo.unwrap())(
            ort::OrtDeviceAllocator,
            ort::OrtMemTypeDefault,
            &mut mem_info,
        )
    };
    unsafe { check_status(api, status, "CreateCpuMemoryInfo") };

    let shape: [i64; 2] = [rows as i64, 4];
    let mut owned: Vec<Vec<f32>> = input_data.to_vec();
    let mut values: Vec<*mut ort::OrtValue> = Vec::with_capacity(owned.len());
    for buf in owned.iter_mut() {
        assert_eq!(buf.len(), rows * 4, "input buffer must match [rows,4]");
        let mut v: *mut ort::OrtValue = ptr::null_mut();
        let status = unsafe {
            ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
                mem_info,
                buf.as_mut_ptr().cast(),
                buf.len() * std::mem::size_of::<f32>(),
                shape.as_ptr(),
                2,
                ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &mut v,
            )
        };
        unsafe { check_status(api, status, "CreateTensorWithDataAsOrtValue") };
        values.push(v);
    }

    let name_ptrs: Vec<*const std::os::raw::c_char> =
        input_names.iter().map(|n| n.as_ptr()).collect();
    let const_values: Vec<*const ort::OrtValue> = values.iter().map(|v| *v as *const _).collect();
    let out_names = [output_name.as_ptr()];
    let mut output: *mut ort::OrtValue = ptr::null_mut();
    let status = unsafe {
        ((*api).Run.unwrap())(
            session,
            ptr::null(),
            name_ptrs.as_ptr(),
            const_values.as_ptr(),
            const_values.len(),
            out_names.as_ptr(),
            1,
            &mut output,
        )
    };
    unsafe { check_status(api, status, "Run") };
    assert!(!output.is_null(), "Run produced a null output value");

    let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
    let status = unsafe { ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr) };
    unsafe { check_status(api, status, "GetTensorMutableData") };
    let result = unsafe { std::slice::from_raw_parts(data_ptr as *const f32, rows * 4) }.to_vec();

    unsafe { ((*api).ReleaseValue.unwrap())(output) };
    for v in values {
        unsafe { ((*api).ReleaseValue.unwrap())(v) };
    }
    unsafe { ((*api).ReleaseMemoryInfo.unwrap())(mem_info) };
    result
}

/// Repeated `Run`s of an unchanged shape must plan the workspace once.
///
/// `Kernel::workspace_requirement` is where the CUDA GEMM kernels run a
/// `cublasLtMatmulAlgoGetHeuristic` search. Before the executor memoized it,
/// every dispatch ran that search, had the result declined as
/// `SessionPersistent`, and then the kernel ran it a *second* time inside
/// `execute` — twice per node per decode step, for one usable plan.
///
/// Falsifier: delete `WorkspacePlanCache::lookup` (or key it on anything the
/// kernel cannot see) and this counter grows by one per `Run`, so `plans_total`
/// becomes `plans_after_first * RUNS` and the assertion fails with the exact
/// numbers.
#[test]
fn workspace_plans_do_not_repeat_for_an_unchanged_shape() {
    let _lock = lock_ort_ep();
    let Some(fx) = Fixture::acquire("workspace_plans_do_not_repeat") else {
        return;
    };
    fx.reset_counters();
    let api = fx.api;
    let model = fixture_model("add_1x4");
    let reg_name = CString::new("shared_mock_plan_cache").unwrap();
    const RUNS: usize = 12;

    unsafe {
        let env = create_env_with_plugin(api, "nxrt_shared_plan_cache", &reg_name, &fx.plugin_path);
        let device = find_our_ep_device(api, env);
        let (session, options) = create_session_on_our_ep(api, env, device, &model);

        let first = run_1x4(
            api,
            session,
            &[c"X", c"Y"],
            &[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            c"Z",
        );
        assert_close(first, [6.0, 8.0, 10.0, 12.0], "add_1x4 run 1");
        let plans_after_first = fx.counter("nxrt_mock_shared_ep_workspace_plans");
        assert!(
            plans_after_first >= 1,
            "the kernel was never asked for a workspace requirement, so this test proves nothing"
        );

        for run in 2..=RUNS {
            let got = run_1x4(
                api,
                session,
                &[c"X", c"Y"],
                &[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                c"Z",
            );
            assert_close(got, [6.0, 8.0, 10.0, 12.0], &format!("add_1x4 run {run}"));
        }

        let plans_total = fx.counter("nxrt_mock_shared_ep_workspace_plans");
        assert_eq!(
            plans_total, plans_after_first,
            "{RUNS} Runs of one unchanged shape re-planned the workspace: {plans_total} plans \
             where the first Run already needed {plans_after_first}. Each of those is a cuBLASLt \
             heuristic search on the CUDA kernels."
        );
        assert!(
            fx.counter("nxrt_mock_shared_ep_workspace_ok") >= RUNS,
            "every Run must still have been served a real workspace: {} served for {RUNS} runs",
            fx.counter("nxrt_mock_shared_ep_workspace_ok")
        );
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_workspace_missing"),
            0,
            "caching the plan must never turn into serving no workspace"
        );
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_alloc_calls"),
            0,
            "workspaces must still come from ORT scratch, not per-dispatch EP allocate/free"
        );

        ((*api).ReleaseSession.unwrap())(session);
        ((*api).ReleaseSessionOptions.unwrap())(options);
        let status = ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_name.as_ptr());
        check_status(api, status, "UnregisterExecutionProviderLibrary");
        ((*api).ReleaseEnv.unwrap())(env);
    }
}

/// A cached plan must never be served to a different shape.
///
/// Counter-test to the one above: on a dynamic-batch model, alternating between
/// two row counts must re-plan on every *change* of shape (so the workspace is
/// sized for the geometry that asked) while still avoiding a re-plan for a
/// repeat of a shape already seen. The kernel writes `numel` f32 sums through
/// the workspace and rejects one that is too small, so a stale 1-row plan
/// served to a 3-row dispatch is a `Run` failure, not a silent pass.
///
/// Falsifier: drop `shape` from the cache key and the 3-row Run fails with
/// *"workspace too small — need 48 bytes, got 16"*.
#[test]
fn a_changed_shape_gets_its_own_workspace_plan() {
    let _lock = lock_ort_ep();
    let Some(fx) = Fixture::acquire("changed_shape_gets_its_own_plan") else {
        return;
    };
    fx.reset_counters();
    let api = fx.api;
    let model = fixture_model("add_dynamic_dim");
    let reg_name = CString::new("shared_mock_dynamic_plan").unwrap();

    unsafe {
        let env = create_env_with_plugin(api, "nxrt_shared_dyn_plan", &reg_name, &fx.plugin_path);
        let device = find_our_ep_device(api, env);
        let (session, options) = create_session_on_our_ep(api, env, device, &model);

        let mut expected_plans = 0usize;
        // rows → (x, y) with a distinct value per element so a wrong-length
        // workspace cannot accidentally produce the right answer.
        for (round, rows) in [1usize, 3, 1, 3, 3, 1].into_iter().enumerate() {
            let x: Vec<f32> = (0..rows * 4).map(|i| i as f32).collect();
            let y: Vec<f32> = (0..rows * 4).map(|i| (i * 10) as f32).collect();
            let want: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
            let got = run_rows_x4(api, session, &[c"X", c"Y"], &[x, y], rows, c"Z");
            assert_eq!(
                got, want,
                "round {round} (rows={rows}) produced the wrong sums, which is what a stale \
                 workspace plan looks like when the buffer happens to be big enough"
            );
            // Rounds 0 and 1 are the first sighting of rows=1 and rows=3.
            if round < 2 {
                expected_plans += 1;
            }
            assert_eq!(
                fx.counter("nxrt_mock_shared_ep_workspace_plans"),
                expected_plans,
                "round {round} (rows={rows}): expected {expected_plans} cumulative plans — one \
                 per distinct shape, none for a repeat"
            );
        }

        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_workspace_missing"),
            0,
            "every dispatch must have been served a workspace"
        );
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_alloc_calls"),
            0,
            "dynamic shapes must not reintroduce per-dispatch EP allocate/free"
        );

        ((*api).ReleaseSession.unwrap())(session);
        ((*api).ReleaseSessionOptions.unwrap())(options);
        let status = ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_name.as_ptr());
        check_status(api, status, "UnregisterExecutionProviderLibrary");
        ((*api).ReleaseEnv.unwrap())(env);
    }
}

/// A declined `SessionPersistent` request must not resolve operand placement.
///
/// Resolving where a node's operands live costs up to `2n` ORT FFI calls plus
/// `n-1` `CompareMemoryInfo` calls, and the answer is only ever used to place a
/// workspace. If the executor derives it before deciding whether it will serve
/// one, every declining node on every dispatch pays for an answer that is
/// thrown away — which is what revision 3 did.
///
/// Falsifier: move the `operand_mem_info` call in `prepare_workspace` back
/// above the `lifetime != StepScoped` gate and this test goes red.
#[test]
fn a_declined_workspace_never_asks_ort_where_the_operands_live() {
    let _lock = lock_ort_ep();
    let Some(fx) = Fixture::acquire("a_declined_workspace_never_asks_ort") else {
        return;
    };
    fx.reset_counters();
    fx.set_persistent_workspace(true);
    let api = fx.api;
    let model = fixture_model("add_1x4");
    let reg_name = CString::new("shared_mock_lazy_declined").unwrap();
    const RUNS: usize = 4;

    unsafe {
        let env =
            create_env_with_plugin(api, "nxrt_shared_lazy_declined", &reg_name, &fx.plugin_path);
        let device = find_our_ep_device(api, env);
        let (session, options) = create_session_on_our_ep(api, env, device, &model);

        for run in 1..=RUNS {
            let got = run_1x4(
                api,
                session,
                &[c"X", c"Y"],
                &[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                c"Z",
            );
            assert_close(
                got,
                [6.0, 8.0, 10.0, 12.0],
                &format!("add_1x4 declined run {run}"),
            );
        }

        // Guard against a vacuous pass: the declining path must actually have
        // been exercised.
        assert!(
            fx.counter("nxrt_mock_shared_ep_persistent_declined") >= RUNS,
            "the SessionPersistent path never ran ({} declines for {RUNS} runs), so this test \
             would pass even if placement were still derived eagerly",
            fx.counter("nxrt_mock_shared_ep_persistent_declined")
        );
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_placement_queries"),
            0,
            "the executor resolved where the operands live for a node whose workspace request \
             it then declined. That is {} wasted placement resolutions per dispatch, on every \
             node of every decode step.",
            fx.counter("nxrt_mock_shared_ep_placement_queries")
        );

        ((*api).ReleaseSession.unwrap())(session);
        ((*api).ReleaseSessionOptions.unwrap())(options);
        let status = ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_name.as_ptr());
        check_status(api, status, "UnregisterExecutionProviderLibrary");
        ((*api).ReleaseEnv.unwrap())(env);
    }
}

/// The other half of the same contract: a node that *is* served must query.
///
/// `chain_add_mul` mixes both kinds of node in one fused subgraph — two `Add`
/// nodes that receive a `StepScoped` workspace and one `Mul` node that declares
/// [`WorkspaceRequirement::NONE`]. Exactly the served dispatches may resolve
/// placement, so the query count must equal the served-workspace count: any
/// higher and the zero-byte `Mul` is paying too, any lower and a served
/// workspace was placed without checking where its kernel runs.
#[test]
fn only_the_nodes_that_receive_a_workspace_query_placement() {
    let _lock = lock_ort_ep();
    let Some(fx) = Fixture::acquire("only_served_nodes_query_placement") else {
        return;
    };
    fx.reset_counters();
    let api = fx.api;
    let model = fixture_model("chain_add_mul");
    let reg_name = CString::new("shared_mock_lazy_served").unwrap();

    unsafe {
        let env =
            create_env_with_plugin(api, "nxrt_shared_lazy_served", &reg_name, &fx.plugin_path);
        let device = find_our_ep_device(api, env);
        let (session, options) = create_session_on_our_ep(api, env, device, &model);

        let got = run_1x4(
            api,
            session,
            &[c"A", c"B", c"C", c"D"],
            &[
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            c"T",
        );
        assert_close(got, [4.0, 6.0, 8.0, 10.0], "chain_add_mul (lazy placement)");

        let served = fx.counter("nxrt_mock_shared_ep_workspace_ok");
        let queries = fx.counter("nxrt_mock_shared_ep_placement_queries");
        let mul_ran = fx.counter("nxrt_mock_shared_ep_mul_executed");
        assert!(
            served >= 2 && mul_ran >= 1,
            "the fixture did not exercise both kinds of node (served={served}, mul={mul_ran}), \
             so the equality below would prove nothing"
        );
        assert_eq!(
            queries, served,
            "placement was resolved {queries} times for {served} served workspaces. Equality is \
             the contract: the zero-workspace Mul node must not query, and every served node \
             must."
        );
        assert_eq!(
            fx.counter("nxrt_mock_shared_ep_workspace_missing"),
            0,
            "deferring the placement resolution must never turn into serving no workspace"
        );

        ((*api).ReleaseSession.unwrap())(session);
        ((*api).ReleaseSessionOptions.unwrap())(options);
        let status = ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_name.as_ptr());
        check_status(api, status, "UnregisterExecutionProviderLibrary");
        ((*api).ReleaseEnv.unwrap())(env);
    }
}
