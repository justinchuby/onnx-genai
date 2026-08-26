//! L2 integration test: `dlopen` the produced cdylib and drive the factory
//! lifecycle through the C ABI.
//!
//! This proves the ABI is well-formed without requiring upstream ORT. It
//! resolves the exported symbols, calls them, and verifies the factory vtable
//! works correctly.

mod cdylib_resolve;

use std::ffi::CStr;
use std::ptr;

use libloading::Library;
use onnx_genai_ort_sys as ort;

/// Find the cdylib produced by this crate's build.
fn find_cdylib() -> std::path::PathBuf {
    cdylib_resolve::find_cpu_plugin_cdylib()
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

    // ReleaseEpFactory returns OrtStatus* per onnxruntime_ep_c_api.h:2669,
    // NOT void. We previously broke this assertion ourselves on arm64/macOS
    // (changed the test to match a wrong implementation). This restores the
    // correct ABI signature. Do NOT "fix" it back to void — the *implementation*
    // was wrong, not this test.
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

    // Release factory — ReleaseEpFactory returns OrtStatus* (null on success).
    let release_status = unsafe { release(factory) };
    assert!(
        release_status.is_null(),
        "ReleaseEpFactory returned non-null status (error) on success"
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

    /// Stand in for the host's `CreateStatus`, and **say what went wrong**.
    ///
    /// The sentinel alone tells a failing assertion only that `Compute`
    /// returned an error, never which of its many fail-closed branches
    /// produced it. When #2200 made residency mandatory and this table did not
    /// yet supply `GetTensorMemoryInfo`, all seven `Compute` tests reported
    /// `Compute failed` and nothing else — while the plugin was saying
    /// `Compute: OrtApi.GetTensorMemoryInfo is null` the whole time. The cause
    /// had to be recovered by reading the diff instead of the failure.
    /// libtest captures this stream per test and prints it only for tests that
    /// fail, so the message costs nothing on a green run.
    pub unsafe extern "C" fn mock_create_status(
        _code: ort::OrtErrorCode,
        msg: *const std::ffi::c_char,
    ) -> *mut ort::OrtStatus {
        if !msg.is_null() {
            // SAFETY: the plugin passes a live `CString` for the duration of
            // this call, so the pointer is a valid NUL-terminated string here.
            let text = unsafe { std::ffi::CStr::from_ptr(msg) }.to_string_lossy();
            eprintln!("mock host CreateStatus: {text}");
        }
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

    pub unsafe extern "C" fn mock_get_tensor_memory_info(
        _value: *const ort::OrtValue,
        out: *mut *const ort::OrtMemoryInfo,
    ) -> ort::OrtStatusPtr {
        // The production adapter now requires the same residency hooks real ORT
        // API 27 provides. This mock models a CPU allocator; omitting the hooks
        // is not an older supported plugin ABI, it is an incomplete OrtApi
        // fixture that correctly makes Compute fail closed.
        //
        // What this does *not* cover: reporting a CUDA allocator here instead
        // (device type GPU, name `Cuda`) leaves all twelve tests green, so
        // nothing in this file asserts that residency reaches the kernel.
        // `kernel_ctx.rs`'s own tests do assert it; the exported-ABI layer does
        // not. Measured, not assumed. #2224.
        unsafe { *out = ptr::dangling::<ort::OrtMemoryInfo>() };
        ptr::null_mut()
    }

    pub unsafe extern "C" fn mock_memory_info_get_device_type(
        _memory_info: *const ort::OrtMemoryInfo,
        out: *mut ort::OrtMemoryInfoDeviceType,
    ) {
        unsafe { *out = ort::OrtMemoryInfoDeviceType_CPU };
    }

    pub unsafe extern "C" fn mock_memory_info_get_name(
        _memory_info: *const ort::OrtMemoryInfo,
        out: *mut *const std::ffi::c_char,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = c"Cpu".as_ptr() };
        ptr::null_mut()
    }

    pub unsafe extern "C" fn mock_memory_info_get_id(
        _memory_info: *const ort::OrtMemoryInfo,
        out: *mut i32,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = 0 };
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
    /// Publish the mock table through the plugin's host-API global.
    ///
    /// That global is process-wide, so whatever it points at has to outlive
    /// every test rather than just the one that published it. Passing a stack
    /// local left it dangling the moment that test returned, and libtest runs
    /// these in parallel: a test still inside `Compute` would read a frame
    /// another test had already popped, and see an operand count of zero from
    /// reused stack bytes. Leaking one table sidesteps the lifetime entirely,
    /// and since every caller built an identical one there is nothing left for
    /// the ordering to vary.
    pub fn install_host_api() {
        static API: std::sync::OnceLock<&'static ort::OrtApi> = std::sync::OnceLock::new();
        let api: &'static ort::OrtApi = API.get_or_init(|| Box::leak(Box::new(mock_ort_api())));
        // SAFETY: `api` is `'static`, and the plugin only reads through it.
        unsafe { onnx_runtime_ep_plugin::status::set_host_api(api as *const ort::OrtApi) };
    }

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
            GetTensorMemoryInfo: Some(mock_get_tensor_memory_info),
            MemoryInfoGetDeviceType: Some(mock_memory_info_get_device_type),
            MemoryInfoGetName: Some(mock_memory_info_get_name),
            MemoryInfoGetId: Some(mock_memory_info_get_id),
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
    install_host_api();

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
        output_dtypes: vec![DataType::Float32],
        absent_output_slots: std::collections::HashSet::new(),
        shape_inference: ShapeInference::ElementwiseBroadcast,
        input_slots: vec![Some(0), Some(1)],
    };
    let mut info = ExportedComputeInfo::new(vec![entry]);
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

    install_host_api();

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
        output_dtypes: vec![DataType::Float32],
        absent_output_slots: std::collections::HashSet::new(),
        shape_inference: ShapeInference::ElementwiseBroadcast,
        input_slots: vec![Some(0), Some(1)],
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

/// Records the operand arity and absent-ness the plugin actually handed a
/// kernel, so a test can assert on what the binding produced rather than only
/// on the numbers that came out the other end.
#[derive(Default)]
struct SeenOperands {
    /// One entry per operand: `None` when the slot arrived absent, otherwise
    /// the operand's first element, which identifies *which* ORT input landed
    /// there. Recording identity and not merely absent-ness is what makes a
    /// mis-paired binding visible: a pattern of absent slots can be symmetric,
    /// but the values are not.
    inputs: Vec<Option<f32>>,
    outputs: usize,
}

thread_local! {
    static SEEN: std::cell::RefCell<SeenOperands> =
        std::cell::RefCell::new(SeenOperands::default());
}

/// A kernel that records what it was given and writes its first present input
/// through to output 0.
#[derive(Debug)]
struct RecordingKernel;

impl onnx_runtime_ep_api::kernel::Kernel for RecordingKernel {
    fn execute(
        &self,
        inputs: &[onnx_runtime_ep_api::tensor::TensorView],
        outputs: &mut [onnx_runtime_ep_api::tensor::TensorMut],
    ) -> onnx_runtime_ep_api::Result<()> {
        SEEN.with(|c| {
            let mut c = c.borrow_mut();
            c.inputs = inputs
                .iter()
                .map(|v| {
                    if v.is_absent() {
                        None
                    } else {
                        // SAFETY: present operands in these tests are f32 with
                        // at least one element.
                        Some(unsafe { *v.data_ptr::<f32>() })
                    }
                })
                .collect();
            c.outputs = outputs.len();
        });
        let src = inputs.iter().find(|v| !v.is_absent()).ok_or_else(|| {
            onnx_runtime_ep_api::EpError::KernelFailed("RecordingKernel: no present input".into())
        })?;
        let n = src.shape.iter().product::<usize>();
        let src_ptr = src.data_ptr::<f32>();
        // SAFETY: both operands are f32, host-resident and `n` elements long;
        // the plugin sized output 0 from this kernel's shape inference.
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, outputs[0].data_ptr_mut::<f32>(), n);
        }
        Ok(())
    }
}

/// Drive one single-node `Compute` over `slots`, with `present` ORT inputs
/// bound and one output, and return what the kernel saw.
fn run_slots(slots: Vec<Option<usize>>, present: usize) -> (bool, Vec<Option<f32>>, usize) {
    run_node(slots, present, 1)
}

/// As [`run_slots`], but with a chosen output arity so the output storage can
/// be pushed past its inline width independently of the operand storage.
fn run_node(
    slots: Vec<Option<usize>>,
    present: usize,
    num_outputs: usize,
) -> (bool, Vec<Option<f32>>, usize) {
    use mock_kernel_ctx::*;
    use onnx_runtime_ep_plugin::compute::{
        CompiledKernelEntry, ExportedComputeInfo, ShapeInference,
    };
    use onnx_runtime_ir::DataType;

    install_host_api();

    STATE.with(|s| {
        *s.borrow_mut() = Some(MockKernelState {
            inputs: (0..present)
                .map(|k| MockTensor {
                    data: vec![1.0 + k as f32, 2.0, 3.0, 4.0],
                    shape: vec![4],
                })
                .collect(),
            outputs: vec![],
        });
    });
    SEEN.with(|c| *c.borrow_mut() = SeenOperands::default());

    let entry = CompiledKernelEntry {
        kernel: Box::new(RecordingKernel),
        num_inputs: slots.len(),
        num_outputs,
        output_dtypes: vec![DataType::Float32; num_outputs],
        absent_output_slots: std::collections::HashSet::new(),
        shape_inference: if num_outputs == 1 {
            ShapeInference::ElementwiseBroadcast
        } else {
            ShapeInference::SameAsInputMultiOutput {
                idx: 0,
                count: num_outputs,
            }
        },
        input_slots: slots,
    };
    let mut info = ExportedComputeInfo::new(vec![entry]);
    let compute_fn = info.vtable.Compute.unwrap();
    let info_ptr = &mut info.vtable as *mut ort::OrtNodeComputeInfo;
    let dummy_ctx = 0xDEAD_BEEFusize as *mut ort::OrtKernelContext;
    let status = unsafe { compute_fn(info_ptr, ptr::null_mut(), dummy_ctx) };
    let ok = status.is_null();
    let (seen_in, seen_out) = SEEN.with(|c| {
        let c = c.borrow();
        (c.inputs.clone(), c.outputs)
    });
    (ok, seen_in, seen_out)
}

/// A node with more outputs than the inline array holds spills them to the
/// heap, and still gets exactly its own output arity.
///
/// The input and output storages have separate widths decided by separate
/// lengths, so the operand spill test says nothing about this one. Without a
/// case here the output heap arm inherits its confidence from the input arm by
/// analogy, which is not evidence.
#[test]
fn a_node_wider_than_the_inline_array_spills_its_outputs_to_the_heap() {
    for outs in [INLINE_OPERAND_WIDTH, INLINE_OPERAND_WIDTH + 1, 6] {
        let (ok, seen, seen_outs) = run_node(vec![Some(0)], 1, outs);
        assert!(ok, "{outs} outputs: Compute failed");
        assert_eq!(seen_outs, outs, "{outs} outputs: kernel saw {seen_outs}");
        assert_eq!(seen, vec![Some(1.0)], "{outs} outputs: operands disturbed");
    }
}

/// Mirror of `compute::INLINE_OPERANDS`, which is private. A drift between the
/// two only weakens the boundary cases below into ordinary ones, and the
/// mutation harness raises the real constant to catch that.
const INLINE_OPERAND_WIDTH: usize = 4;

/// The kernel sees exactly as many operands as the node has slots.
///
/// The operand views are built in a fixed-width stack array, so the arity the
/// kernel observes is a property of the slicing, not of the storage. A binding
/// that handed over the whole array would show up here as extra absent
/// operands.
#[test]
fn a_node_sees_exactly_its_own_operand_count() {
    for arity in 1..=3usize {
        let (ok, seen, _) = run_slots((0..arity).map(Some).collect(), arity);
        assert!(ok, "arity {arity}: Compute failed");
        assert_eq!(
            seen.len(),
            arity,
            "arity {arity}: kernel saw {} operands, not {arity}",
            seen.len()
        );
        assert!(
            seen.iter().all(|v| v.is_some()),
            "arity {arity}: a bound operand arrived absent: {seen:?}"
        );
    }
}

/// An unbound optional slot arrives at the kernel marked absent.
///
/// The stack array is seeded with the absent sentinel and the fill loop skips
/// unbound slots, so this is what proves the seed is the right value and is
/// actually reached — not merely that nothing crashed.
#[test]
fn an_unbound_slot_arrives_absent_and_in_position() {
    // Asymmetric on purpose. `[present, absent, present]` reads as a fine
    // test and is not one: it is a palindrome, so a binding that paired
    // operands with slots in reverse order produces exactly the same absent
    // pattern and the test passes. The values are what break the symmetry.
    let (ok, seen, _) = run_slots(vec![Some(0), None, Some(1)], 2);
    assert!(ok, "Compute failed with an unbound middle slot");
    assert_eq!(
        seen,
        vec![Some(1.0), None, Some(2.0)],
        "operands did not land in their own slots"
    );

    let (ok, seen, _) = run_slots(vec![None, Some(0), Some(1)], 2);
    assert!(ok, "Compute failed with an unbound leading slot");
    assert_eq!(
        seen,
        vec![None, Some(1.0), Some(2.0)],
        "a leading unbound slot shifted the operands"
    );
}

/// A node wider than the inline operand array still works, through the heap.
///
/// `INLINE_OPERANDS` is 4, so 6 forces the spill path. Without this the whole
/// fallback branch would be untested and a node with many inputs would be the
/// first thing to discover it.
#[test]
fn a_node_wider_than_the_inline_array_spills_to_the_heap() {
    let arity = 6usize;
    let (ok, seen, _) = run_slots((0..arity).map(Some).collect(), arity);
    assert!(ok, "Compute failed for a {arity}-operand node");
    assert_eq!(seen.len(), arity, "wide node lost or gained operands");
    assert_eq!(
        seen,
        (0..arity).map(|k| Some(1.0 + k as f32)).collect::<Vec<_>>(),
        "wide node: operands arrived absent or out of order"
    );
}

/// The boundary itself: exactly `INLINE_OPERANDS` operands stays inline and
/// one more spills, and both give the kernel the same arity.
#[test]
fn the_inline_operand_boundary_is_off_by_none() {
    for arity in [4usize, 5] {
        let (ok, seen, outs) = run_slots((0..arity).map(Some).collect(), arity);
        assert!(ok, "arity {arity}: Compute failed at the inline boundary");
        assert_eq!(
            seen,
            (0..arity).map(|k| Some(1.0 + k as f32)).collect::<Vec<_>>(),
            "arity {arity}: operands arrived absent or out of order"
        );
        assert_eq!(outs, 1, "arity {arity}: wrong output count");
    }
}

// ─── L1 — ABI surface: exported symbol audit ─────────────────────────────────

/// L1 (portable): Verify the two required symbols resolve via `dlsym`/`LoadLibrary`.
///
/// This is the strongest portable assertion: if `dlsym` finds them, they are
/// The cdylib reports the optional features it was built with.
///
/// A packaged cdylib is opaque, and the difference between an MLAS build and a
/// pure-Rust one is an order of magnitude on the quantized matmul paths. The
/// wheel's smoke test reads this symbol to prove what it shipped, so the
/// symbol must exist, must be a valid C string, and must agree with the
/// feature set this test binary was compiled with -- a stale string that
/// always says "mlas" would make that proof worthless.
#[test]
fn l1_build_features_match_the_compiled_feature_set() {
    let path = find_cdylib();
    let lib = unsafe { Library::new(&path) }
        .unwrap_or_else(|e| panic!("dlopen failed for {}: {e}", path.display()));
    let features: libloading::Symbol<'_, unsafe extern "C" fn() -> *const std::os::raw::c_char> =
        unsafe { lib.get(b"nxrt_ep_build_features") }
            .expect("nxrt_ep_build_features not exported from cdylib");
    let ptr = unsafe { features() };
    assert!(!ptr.is_null(), "nxrt_ep_build_features returned null");
    let reported = unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_str()
        .expect("build features are ASCII");
    let expected = if cfg!(feature = "mlas") { "mlas" } else { "" };
    assert_eq!(
        reported,
        expected,
        "the cdylib at {} reports features {reported:?} but this test binary was \
         built with {expected:?}",
        path.display()
    );
}

/// genuinely exported and callable on this platform. Works on Linux, macOS, and
/// Windows without requiring `nm`, `readelf`, or `dumpbin`.
#[test]
fn l1_required_symbols_resolve() {
    let path = find_cdylib();
    let lib = unsafe { Library::new(&path) }
        .unwrap_or_else(|e| panic!("dlopen failed for {}: {e}", path.display()));

    // Both symbols must resolve.
    let _create: libloading::Symbol<'_, unsafe extern "C" fn()> =
        unsafe { lib.get(b"CreateEpFactories") }
            .expect("CreateEpFactories not exported from cdylib");
    let _release: libloading::Symbol<'_, unsafe extern "C" fn()> =
        unsafe { lib.get(b"ReleaseEpFactory") }.expect("ReleaseEpFactory not exported from cdylib");

    // The hardware-validation harness (`scripts/validate_ep_workspace_h200.py`)
    // reads these through `dlopen` on the very library ORT loaded. If they stop
    // being exported the harness silently loses its proof that a workspace was
    // served, so their absence has to fail here instead.
    for symbol in [
        &b"nxrt_ep_compiled_node_count"[..],
        &b"nxrt_ep_reset_compiled_node_count"[..],
        &b"nxrt_ep_workspace_placement_queries"[..],
        &b"nxrt_ep_reset_workspace_placement_queries"[..],
        &b"nxrt_ep_constant_weight_inputs"[..],
        &b"nxrt_ep_reset_constant_weight_inputs"[..],
        &b"nxrt_ep_executed_node_count"[..],
        &b"nxrt_ep_reset_executed_node_count"[..],
    ] {
        let _counter: libloading::Symbol<'_, unsafe extern "C" fn() -> usize> =
            unsafe { lib.get(symbol) }.unwrap_or_else(|e| {
                panic!(
                    "{} not exported from cdylib: {e}",
                    String::from_utf8_lossy(symbol)
                )
            });
    }

    // `nxrt_ep_persistent_decode_pool_built` returns `i32`, not `usize`, so it
    // cannot ride the loop above. It is the only proof that `CreateEpFactories`
    // opted this process out of the persistent SPMD decode pool, and the test
    // that reads it is ORT-gated; requiring the export here keeps a check that
    // runs even when ONNX Runtime is unavailable.
    let _pool_probe: libloading::Symbol<'_, unsafe extern "C" fn() -> i32> =
        unsafe { lib.get(&b"nxrt_ep_persistent_decode_pool_built"[..]) }.unwrap_or_else(|e| {
            panic!("nxrt_ep_persistent_decode_pool_built not exported from cdylib: {e}")
        });

    eprintln!("✓ l1_required_symbols_resolve: CreateEpFactories ✓  ReleaseEpFactory ✓");
}

/// L1 (Linux-only): Verify no unexpected symbols leak from the cdylib.
///
/// Uses `nm --dynamic` on ELF targets to inspect the dynamic symbol table.
/// On non-ELF platforms (macOS, Windows) this test is skipped with a clear
/// message — the `l1_required_symbols_resolve` test still runs everywhere.
#[test]
fn l1_no_symbol_leakage() {
    if !cfg!(target_os = "linux") {
        eprintln!(
            "⏭ l1_no_symbol_leakage: skipped (ELF-only check, this is {})",
            std::env::consts::OS
        );
        return;
    }

    let path = find_cdylib();

    let output = std::process::Command::new("nm")
        .args([
            "--dynamic",
            "--defined-only",
            "--extern-only",
            "--format=posix",
        ])
        .arg(&path)
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            eprintln!(
                "⏭ l1_no_symbol_leakage: skipped — nm failed: {}",
                String::from_utf8_lossy(&o.stderr)
            );
            return;
        }
        Err(e) => {
            eprintln!("⏭ l1_no_symbol_leakage: skipped — nm not found: {e}");
            return;
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let text_symbols: Vec<&str> = stdout
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let name = fields.next()?;
            let sym_type = fields.next()?;
            if sym_type == "T" { Some(name) } else { None }
        })
        .collect();

    assert!(
        text_symbols.contains(&"CreateEpFactories"),
        "CreateEpFactories not found in T symbols.\nFound: {text_symbols:?}"
    );
    assert!(
        text_symbols.contains(&"ReleaseEpFactory"),
        "ReleaseEpFactory not found in T symbols.\nFound: {text_symbols:?}"
    );

    let unexpected: Vec<&str> = text_symbols
        .iter()
        .copied()
        .filter(|name| {
            *name != "CreateEpFactories"
                && *name != "ReleaseEpFactory"
                && *name != "nxrt_ep_compiled_node_count"
                && *name != "nxrt_ep_reset_compiled_node_count"
                && *name != "nxrt_ep_workspace_placement_queries"
                && *name != "nxrt_ep_reset_workspace_placement_queries"
                && *name != "nxrt_ep_constant_weight_inputs"
                && *name != "nxrt_ep_reset_constant_weight_inputs"
                && *name != "nxrt_ep_executed_node_count"
                && *name != "nxrt_ep_reset_executed_node_count"
                && *name != "nxrt_ep_build_features"
                && *name != "nxrt_ep_persistent_decode_pool_built"
                // The dispatch probe is a research build. Its four exports
                // exist only under the `dispatch_probe` feature, which no
                // shipped build sets, so in a production cdylib these are
                // absent and this arm never fires. Gated on the same `cfg` as
                // the exports so that a future ungating fails this test rather
                // than silently widening the shipped ABI.
                && !(cfg!(feature = "dispatch_probe")
                    && matches!(
                        *name,
                        "nxrt_dispatch_probe_snapshot"
                            | "nxrt_dispatch_probe_reset"
                            | "nxrt_dispatch_probe_available"
                            | "nxrt_dispatch_probe_phase_name"
                    ))
                && !name.starts_with("_Z")
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
        "Unexpected public T symbols leaked from cdylib:\n  {}\nFile: {}",
        unexpected.join(", "),
        path.display()
    );

    eprintln!(
        "✓ l1_no_symbol_leakage: no unexpected symbols ({} T symbols total)",
        text_symbols.len()
    );
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
