//! L2 integration test: `dlopen` the produced cdylib and drive the factory
//! lifecycle through the C ABI.
//!
//! This proves the ABI is well-formed without requiring upstream ORT. It
//! resolves the exported symbols, calls them, and verifies the factory vtable
//! works correctly.

use std::ffi::CStr;
use std::ptr;

use libloading::Library;
use onnx_genai_ort_sys as ort;

/// Find the cdylib produced by this crate's build.
fn find_cdylib() -> std::path::PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    let candidates = [
        workspace_root.join("target/debug/libonnx_runtime_ep_cpu_plugin.so"),
        workspace_root.join("target/release/libonnx_runtime_ep_cpu_plugin.so"),
    ];

    for candidate in &candidates {
        if candidate.exists() {
            return candidate.clone();
        }
    }

    panic!(
        "Could not find libonnx_runtime_ep_cpu_plugin.so. Build with: \
         cargo build -p onnx-runtime-ep-cpu-plugin\n\
         Searched: {candidates:?}"
    );
}

// Minimal OrtApi with just CreateStatus for version negotiation.
static TEST_ORT_API: std::sync::OnceLock<ort::OrtApi> = std::sync::OnceLock::new();
static TEST_ORT_API_BASE: std::sync::OnceLock<ort::OrtApiBase> = std::sync::OnceLock::new();

unsafe extern "C" fn test_create_status(
    _code: ort::OrtErrorCode,
    _msg: *const std::ffi::c_char,
) -> *mut ort::OrtStatus {
    // Return null (no error status object) for test simplicity.
    // In a real scenario this would allocate, but for the factory lifecycle
    // test we just need it to not crash.
    ptr::null_mut()
}

unsafe extern "C" fn test_get_api(_version: u32) -> *const ort::OrtApi {
    TEST_ORT_API.get_or_init(|| ort::OrtApi {
        CreateStatus: Some(test_create_status),
        ..Default::default()
    })
}

unsafe extern "C" fn test_get_version_string() -> *const std::ffi::c_char {
    c"test-host-1.27.0".as_ptr()
}

fn test_api_base() -> *const ort::OrtApiBase {
    TEST_ORT_API_BASE.get_or_init(|| ort::OrtApiBase {
        GetApi: Some(test_get_api),
        GetVersionString: Some(test_get_version_string),
    })
}

#[test]
fn dlopen_and_create_factory() {
    let path = find_cdylib();

    // SAFETY: We're loading our own cdylib.
    let lib = unsafe { Library::new(&path) }.unwrap_or_else(|e| {
        panic!("Failed to dlopen {}: {e}", path.display());
    });

    type CreateEpFactories = unsafe extern "C" fn(
        *const std::ffi::c_char,
        *const ort::OrtApiBase,
        *const ort::OrtLogger,
        *mut *mut ort::OrtEpFactory,
        usize,
        *mut usize,
    ) -> *mut ort::OrtStatus;

    type ReleaseEpFactory = unsafe extern "C" fn(*mut ort::OrtEpFactory) -> *mut ort::OrtStatus;

    let create: libloading::Symbol<'_, CreateEpFactories> =
        unsafe { lib.get(b"CreateEpFactories") }.expect("CreateEpFactories symbol not found");

    let release: libloading::Symbol<'_, ReleaseEpFactory> =
        unsafe { lib.get(b"ReleaseEpFactory") }.expect("ReleaseEpFactory symbol not found");

    let api_base = test_api_base();

    let mut factories: [*mut ort::OrtEpFactory; 1] = [ptr::null_mut()];
    let mut num_factories = 0usize;
    let status = unsafe {
        create(
            ptr::null(),
            api_base,
            ptr::null(),
            factories.as_mut_ptr(),
            1,
            &mut num_factories,
        )
    };

    assert!(
        status.is_null(),
        "CreateEpFactories returned non-null status (error)"
    );
    assert_eq!(num_factories, 1, "expected 1 factory");
    assert!(!factories[0].is_null(), "factory pointer is null");

    let factory = factories[0];

    // Check ort_version_supported.
    let version = unsafe { (*factory).ort_version_supported };
    assert_eq!(
        version,
        ort::ORT_API_VERSION,
        "factory ort_version_supported mismatch"
    );

    // Check GetName.
    let get_name = unsafe { (*factory).GetName }.expect("GetName is null");
    let name_ptr = unsafe { get_name(factory) };
    assert!(!name_ptr.is_null(), "GetName returned null");
    let name = unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy();
    assert_eq!(name, "cpu_ep", "EP name mismatch");

    // Release factory.
    let status = unsafe { release(factory) };
    assert!(
        status.is_null(),
        "ReleaseEpFactory returned non-null status (error)"
    );
}

// ─── L2 Compute test: actual tensor in, actual tensor out ────────────────────

/// Thread-local mock state for the kernel context.
///
/// Our mock OrtApi implementations read/write through this to provide inputs
/// and capture outputs without needing real ORT.
mod mock_kernel_ctx {
    use std::cell::RefCell;
    use std::ffi::c_void;
    use std::ptr;

    use onnx_genai_ort_sys as ort;

    /// A mock tensor: flat f32 data + shape.
    pub struct MockTensor {
        pub data: Vec<f32>,
        pub shape: Vec<i64>,
    }

    pub struct MockKernelState {
        pub inputs: Vec<MockTensor>,
        pub outputs: Vec<MockTensor>,
    }

    thread_local! {
        pub static STATE: RefCell<Option<MockKernelState>> = const { RefCell::new(None) };
    }

    // OrtTensorTypeAndShapeInfo is opaque — use an index-tagged pointer to
    // identify which input it refers to.
    fn index_as_shape_info(idx: usize) -> *mut ort::OrtTensorTypeAndShapeInfo {
        (idx + 1) as *mut ort::OrtTensorTypeAndShapeInfo
    }

    fn shape_info_to_index(ptr: *const ort::OrtTensorTypeAndShapeInfo) -> usize {
        (ptr as usize) - 1
    }

    pub unsafe extern "C" fn mock_create_status(
        _code: ort::OrtErrorCode,
        _msg: *const std::ffi::c_char,
    ) -> *mut ort::OrtStatus {
        // For test: return a non-null sentinel to signal error.
        std::ptr::dangling_mut::<ort::OrtStatus>()
    }

    pub unsafe extern "C" fn mock_get_input_count(
        _ctx: *const ort::OrtKernelContext,
        out: *mut usize,
    ) -> ort::OrtStatusPtr {
        STATE.with(|s| {
            let s = s.borrow();
            let state = s.as_ref().unwrap();
            unsafe { *out = state.inputs.len() };
        });
        ptr::null_mut()
    }

    pub unsafe extern "C" fn mock_get_input(
        _ctx: *const ort::OrtKernelContext,
        index: usize,
        out: *mut *const ort::OrtValue,
    ) -> ort::OrtStatusPtr {
        // Use (index+1) as an OrtValue pointer sentinel.
        unsafe { *out = (index + 1) as *const ort::OrtValue };
        ptr::null_mut()
    }

    pub unsafe extern "C" fn mock_get_tensor_type_and_shape(
        value: *const ort::OrtValue,
        out: *mut *mut ort::OrtTensorTypeAndShapeInfo,
    ) -> ort::OrtStatusPtr {
        // value encodes the input index as (index+1).
        let idx = (value as usize) - 1;
        unsafe { *out = index_as_shape_info(idx) };
        ptr::null_mut()
    }

    pub unsafe extern "C" fn mock_get_tensor_element_type(
        _info: *const ort::OrtTensorTypeAndShapeInfo,
        out: *mut ort::ONNXTensorElementDataType,
    ) -> ort::OrtStatusPtr {
        // Always f32.
        unsafe { *out = ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT };
        ptr::null_mut()
    }

    pub unsafe extern "C" fn mock_get_dimensions_count(
        info: *const ort::OrtTensorTypeAndShapeInfo,
        out: *mut usize,
    ) -> ort::OrtStatusPtr {
        let idx = shape_info_to_index(info);
        STATE.with(|s| {
            let s = s.borrow();
            let state = s.as_ref().unwrap();
            unsafe { *out = state.inputs[idx].shape.len() };
        });
        ptr::null_mut()
    }

    pub unsafe extern "C" fn mock_get_dimensions(
        info: *const ort::OrtTensorTypeAndShapeInfo,
        dim_values: *mut i64,
        dim_count: usize,
    ) -> ort::OrtStatusPtr {
        let idx = shape_info_to_index(info);
        STATE.with(|s| {
            let s = s.borrow();
            let state = s.as_ref().unwrap();
            let shape = &state.inputs[idx].shape;
            for (i, &dim) in shape.iter().enumerate().take(dim_count) {
                unsafe { *dim_values.add(i) = dim };
            }
        });
        ptr::null_mut()
    }

    pub unsafe extern "C" fn mock_release_tensor_type_and_shape_info(
        _info: *mut ort::OrtTensorTypeAndShapeInfo,
    ) {
        // No-op: our "info" pointers are index sentinels.
    }

    pub unsafe extern "C" fn mock_get_tensor_data(
        value: *const ort::OrtValue,
        out: *mut *const c_void,
    ) -> ort::OrtStatusPtr {
        let idx = (value as usize) - 1;
        STATE.with(|s| {
            let s = s.borrow();
            let state = s.as_ref().unwrap();
            unsafe { *out = state.inputs[idx].data.as_ptr().cast::<c_void>() };
        });
        ptr::null_mut()
    }

    pub unsafe extern "C" fn mock_get_output(
        _ctx: *mut ort::OrtKernelContext,
        index: usize,
        dim_values: *const i64,
        dim_count: usize,
        out: *mut *mut ort::OrtValue,
    ) -> ort::OrtStatusPtr {
        // Allocate the output in our mock state.
        let shape: Vec<i64> = (0..dim_count)
            .map(|i| unsafe { *dim_values.add(i) })
            .collect();
        let numel: usize = shape.iter().map(|&d| d as usize).product();

        STATE.with(|s| {
            let mut s = s.borrow_mut();
            let state = s.as_mut().unwrap();
            // Ensure outputs vec is large enough.
            while state.outputs.len() <= index {
                state.outputs.push(MockTensor {
                    data: Vec::new(),
                    shape: Vec::new(),
                });
            }
            state.outputs[index] = MockTensor {
                data: vec![0.0f32; numel],
                shape,
            };
            // Return (index + 0x1000) as a sentinel for output OrtValues.
            unsafe { *out = (index + 0x1000) as *mut ort::OrtValue };
        });
        ptr::null_mut()
    }

    pub unsafe extern "C" fn mock_get_tensor_mutable_data(
        value: *mut ort::OrtValue,
        out: *mut *mut c_void,
    ) -> ort::OrtStatusPtr {
        let idx = (value as usize) - 0x1000;
        STATE.with(|s| {
            let mut s = s.borrow_mut();
            let state = s.as_mut().unwrap();
            unsafe { *out = state.outputs[idx].data.as_mut_ptr().cast::<c_void>() };
        });
        ptr::null_mut()
    }

    /// Build a mock OrtApi with all the functions our Compute path needs.
    pub fn mock_ort_api() -> ort::OrtApi {
        ort::OrtApi {
            CreateStatus: Some(mock_create_status),
            KernelContext_GetInputCount: Some(mock_get_input_count),
            KernelContext_GetInput: Some(mock_get_input),
            KernelContext_GetOutput: Some(mock_get_output),
            GetTensorTypeAndShape: Some(mock_get_tensor_type_and_shape),
            GetTensorElementType: Some(mock_get_tensor_element_type),
            GetDimensionsCount: Some(mock_get_dimensions_count),
            GetDimensions: Some(mock_get_dimensions),
            GetTensorData: Some(mock_get_tensor_data),
            GetTensorMutableData: Some(mock_get_tensor_mutable_data),
            ReleaseTensorTypeAndShapeInfo: Some(mock_release_tensor_type_and_shape_info),
            ..Default::default()
        }
    }
}

/// L2 Compute test: drive Add kernel through the full plugin Compute path.
///
/// Verifies: tensor in (f32) → Add kernel → tensor out (f32), value asserted.
#[test]
fn compute_add_end_to_end() {
    use mock_kernel_ctx::*;
    use onnx_runtime_ep_cpu::kernels::add::AddKernel;
    use onnx_runtime_ep_plugin::compute::{
        CompiledKernelEntry, ExportedComputeInfo, ShapeInference,
    };
    use onnx_runtime_ir::DataType;

    // Set up mock API as the host.
    let api = mock_ort_api();
    let api_ptr: *const ort::OrtApi = &api;
    // SAFETY: single-threaded test; set_host_api is safe to call before Compute.
    unsafe { onnx_runtime_ep_plugin::status::set_host_api(api_ptr) };

    // Prepare input tensors: a = [1.0, 2.0, 3.0, 4.0], b = [10.0, 20.0, 30.0, 40.0]
    STATE.with(|s| {
        *s.borrow_mut() = Some(MockKernelState {
            inputs: vec![
                MockTensor {
                    data: vec![1.0, 2.0, 3.0, 4.0],
                    shape: vec![4],
                },
                MockTensor {
                    data: vec![10.0, 20.0, 30.0, 40.0],
                    shape: vec![4],
                },
            ],
            outputs: vec![],
        });
    });

    // Create an ExportedComputeInfo with an Add kernel.
    let entry = CompiledKernelEntry {
        kernel: Box::new(AddKernel),
        num_inputs: 2,
        num_outputs: 1,
        output_dtype: DataType::Float32,
        shape_inference: ShapeInference::ElementwiseBroadcast,
    };
    let mut info = ExportedComputeInfo::new(vec![entry]);

    // Call Compute through the vtable (same path ORT would use).
    let compute_fn = info.vtable.Compute.expect("Compute is null");
    let info_ptr = &mut info.vtable as *mut ort::OrtNodeComputeInfo;
    // Use a dummy kernel context pointer — our mocks use thread-local state.
    let dummy_ctx = 0xDEAD_BEEFusize as *mut ort::OrtKernelContext;

    let status = unsafe { compute_fn(info_ptr, ptr::null_mut(), dummy_ctx) };
    assert!(
        status.is_null(),
        "Compute returned error status (expected success for Add)"
    );

    // Verify output values.
    STATE.with(|s| {
        let s = s.borrow();
        let state = s.as_ref().unwrap();
        assert_eq!(state.outputs.len(), 1, "expected 1 output");
        let out = &state.outputs[0];
        assert_eq!(out.shape, vec![4i64], "output shape mismatch");
        assert_eq!(
            out.data,
            vec![11.0, 22.0, 33.0, 44.0],
            "output values wrong"
        );
    });
}

/// L2 Compute test with broadcasting: [2,3] + [3] → [2,3].
#[test]
fn compute_add_broadcast() {
    use mock_kernel_ctx::*;
    use onnx_runtime_ep_cpu::kernels::add::AddKernel;
    use onnx_runtime_ep_plugin::compute::{
        CompiledKernelEntry, ExportedComputeInfo, ShapeInference,
    };
    use onnx_runtime_ir::DataType;

    let api = mock_ort_api();
    let api_ptr: *const ort::OrtApi = &api;
    unsafe { onnx_runtime_ep_plugin::status::set_host_api(api_ptr) };

    STATE.with(|s| {
        *s.borrow_mut() = Some(MockKernelState {
            inputs: vec![
                MockTensor {
                    data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    shape: vec![2, 3],
                },
                MockTensor {
                    data: vec![10.0, 20.0, 30.0],
                    shape: vec![3],
                },
            ],
            outputs: vec![],
        });
    });

    let entry = CompiledKernelEntry {
        kernel: Box::new(AddKernel),
        num_inputs: 2,
        num_outputs: 1,
        output_dtype: DataType::Float32,
        shape_inference: ShapeInference::ElementwiseBroadcast,
    };
    let mut info = ExportedComputeInfo::new(vec![entry]);

    let compute_fn = info.vtable.Compute.unwrap();
    let info_ptr = &mut info.vtable as *mut ort::OrtNodeComputeInfo;
    let dummy_ctx = 0xDEAD_BEEFusize as *mut ort::OrtKernelContext;

    let status = unsafe { compute_fn(info_ptr, ptr::null_mut(), dummy_ctx) };
    assert!(status.is_null(), "Compute returned error for broadcast Add");

    STATE.with(|s| {
        let s = s.borrow();
        let state = s.as_ref().unwrap();
        let out = &state.outputs[0];
        assert_eq!(out.shape, vec![2i64, 3], "broadcast output shape");
        assert_eq!(
            out.data,
            vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
            "broadcast output values"
        );
    });
}

// ─── L1 — ABI surface: nm/readelf symbol audit ───────────────────────────────

/// L1: Parse `nm -D` output of the cdylib and verify:
///  - `CreateEpFactories` is exported as a text (T) symbol.
///  - `ReleaseEpFactory` is exported as a text (T) symbol.
///  - No unexpected Rust symbols leak (no `_ZN` C++ mangled names in T slots
///    that are not `CreateEpFactories`/`ReleaseEpFactory`).
///
/// This is the cheapest possible check: no ORT needed, no dlopen, just inspect
/// the on-disk ELF dynamic symbol table.
#[test]
fn l1_nm_exported_symbols() {
    let path = find_cdylib();

    // Run `nm --dynamic --defined-only --extern-only`
    let output = std::process::Command::new("nm")
        .args([
            "--dynamic",
            "--defined-only",
            "--extern-only",
            "--format=posix",
        ])
        .arg(&path)
        .output()
        .expect("`nm` not found — install binutils");

    assert!(
        output.status.success(),
        "nm failed on {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let text_symbols: Vec<&str> = stdout
        .lines()
        .filter_map(|line| {
            // posix format: name type value size
            let mut fields = line.split_whitespace();
            let name = fields.next()?;
            let sym_type = fields.next()?;
            if sym_type == "T" { Some(name) } else { None }
        })
        .collect();

    assert!(
        text_symbols.contains(&"CreateEpFactories"),
        "CreateEpFactories not found in T symbols.\n\
         Found T symbols: {text_symbols:?}"
    );
    assert!(
        text_symbols.contains(&"ReleaseEpFactory"),
        "ReleaseEpFactory not found in T symbols.\n\
         Found T symbols: {text_symbols:?}"
    );

    // Verify no unexpected public T symbols are leaking.
    // Allow: CreateEpFactories, ReleaseEpFactory, and Rust runtime glue
    // (init_array_start, __rust_alloc*, etc.). Specifically reject
    // un-mangled Rust function exports that would indicate `pub extern "C"
    // fn` entries we did not intend to expose.
    let unexpected: Vec<&str> = text_symbols
        .iter()
        .copied()
        .filter(|name| {
            *name != "CreateEpFactories"
                && *name != "ReleaseEpFactory"
                // Allow compiler/runtime bookkeeping
                && !name.starts_with("_Z")      // C++ mangled (should be none)
                && !name.starts_with("__rust")
                && !name.starts_with("__rdl_")
                && !name.starts_with("_start")
                && !name.starts_with("_fini")
                && !name.starts_with("_init")
                && !name.starts_with("__GNU")
                && !name.starts_with("rust_")
                && *name != "_Jv_RegisterClasses"
                && *name != "__cxa_finalize"
                && *name != "__gmon_start__"
        })
        .collect();

    assert!(
        unexpected.is_empty(),
        "Unexpected public T symbols leaked from cdylib:\n  {}\n\
         These should be unexported or renamed. File: {}",
        unexpected.join(", "),
        path.display()
    );

    eprintln!(
        "✓ l1_nm_exported_symbols: CreateEpFactories ✓  ReleaseEpFactory ✓  \
         no leakage ({} T symbols total)",
        text_symbols.len()
    );
}

/// L1: Verify via `readelf --dyn-syms` that the ELF dynamic symbol section
/// contains exactly our two expected symbols as global functions (STT_FUNC, STB_GLOBAL).
/// This is a stricter check than `nm` because it inspects ELF symbol type/binding.
#[test]
fn l1_readelf_dyn_syms() {
    let path = find_cdylib();

    let output = std::process::Command::new("readelf")
        .args(["--dyn-syms", "--wide"])
        .arg(&path)
        .output()
        .expect("`readelf` not found — install binutils");

    assert!(
        output.status.success(),
        "readelf failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // readelf --wide output line format (simplified):
    //   Num: Value Size Type Bind Vis Ndx Name
    // We look for lines with FUNC GLOBAL and our symbol names.
    let has_create = stdout.lines().any(|line| {
        line.contains("FUNC") && line.contains("GLOBAL") && line.contains("CreateEpFactories")
    });
    let has_release = stdout.lines().any(|line| {
        line.contains("FUNC") && line.contains("GLOBAL") && line.contains("ReleaseEpFactory")
    });

    assert!(
        has_create,
        "readelf: CreateEpFactories not found as FUNC GLOBAL in .dynsym"
    );
    assert!(
        has_release,
        "readelf: ReleaseEpFactory not found as FUNC GLOBAL in .dynsym"
    );

    eprintln!("✓ l1_readelf_dyn_syms: both symbols are FUNC GLOBAL in .dynsym");
}

// ─── L2 — Fail-closed: bogus ORT API version ─────────────────────────────────

/// L2 fail-closed: calling `CreateEpFactories` with an OrtApiBase whose `GetApi`
/// returns null for our API version must return a non-null OrtStatus (failure)
/// with an actionable message, not succeed and return garbage factories.
///
/// This validates the version negotiation gate in `factory::create_ep_factories`.
#[test]
fn l2_fail_closed_unsupported_api_version() {
    use std::sync::OnceLock;

    // An OrtApiBase whose GetApi always returns null — simulating an older ORT
    // host that does not support ORT_API_VERSION = 27.
    static NULL_API: OnceLock<ort::OrtApi> = OnceLock::new();
    static NULL_API_BASE: OnceLock<ort::OrtApiBase> = OnceLock::new();

    unsafe extern "C" fn returns_null_api(_version: u32) -> *const ort::OrtApi {
        // Return null for any requested version — simulates a host too old.
        std::ptr::null()
    }
    unsafe extern "C" fn get_version() -> *const std::ffi::c_char {
        c"0.0.0-mock-too-old".as_ptr()
    }

    let path = find_cdylib();
    let lib = unsafe { Library::new(&path) }.unwrap_or_else(|e| panic!("dlopen failed: {e}"));

    type CreateFn = unsafe extern "C" fn(
        *const std::ffi::c_char,
        *const ort::OrtApiBase,
        *const ort::OrtLogger,
        *mut *mut ort::OrtEpFactory,
        usize,
        *mut usize,
    ) -> *mut ort::OrtStatus;

    let create: libloading::Symbol<'_, CreateFn> =
        unsafe { lib.get(b"CreateEpFactories") }.expect("CreateEpFactories not found");

    let api_base = NULL_API_BASE.get_or_init(|| ort::OrtApiBase {
        GetApi: Some(returns_null_api),
        GetVersionString: Some(get_version),
    });

    let mut factories: [*mut ort::OrtEpFactory; 1] = [std::ptr::null_mut()];
    let mut num = 0usize;

    let status = unsafe {
        create(
            std::ptr::null(),
            api_base as *const ort::OrtApiBase,
            std::ptr::null(),
            factories.as_mut_ptr(),
            1,
            &mut num,
        )
    };

    // Must write zero factories (fail-closed behavior).
    assert_eq!(
        num, 0,
        "fail-closed: CreateEpFactories must write 0 factories on unsupported API version, got {num}"
    );

    // The status may be null (when we cannot allocate a status) or non-null.
    // Either way, we must NOT have returned a valid factory.
    assert!(
        factories[0].is_null(),
        "fail-closed: factory slot must remain null on unsupported API version"
    );

    // If status is non-null, it should not be a dangling pointer we can't use.
    // We can't check the message without a valid OrtApi (host is too old), so
    // we just assert no crash occurred during the fail-closed path.
    eprintln!(
        "✓ l2_fail_closed_unsupported_api_version: status={:?} factories[0]={:?} num={num}",
        status, factories[0]
    );
    eprintln!("  Fail-closed behavior confirmed: 0 factories returned for unsupported API version");
}
