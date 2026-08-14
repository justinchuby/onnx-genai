//! An ORT custom op that reduces a tensor's last axis to the index of its
//! maximum, on the CUDA execution provider.
//!
//! # Why this exists
//!
//! ORT's own CUDA `ArgMax` gives each reduced *row* one lane, so a reduction
//! over a single very wide row runs serially. For a decode step that is exactly
//! the shape that matters: `[1, vocab]` with `vocab` in the hundreds of
//! thousands. Measured on an H200 with `vocab = 202048`, ORT's kernel takes
//! 4.44 ms. Splitting the row into a `rows x tile` grid (see
//! [`crate`]-adjacent `arg_reduce` in `onnx-genai-engine`) recovers most of
//! that using only standard ONNX ops and stays portable, but it is still bound
//! by that kernel's per-row serial scan: two stages of ~450 elements measure
//! 35.4 us of kernel time on the same machine.
//!
//! This op replaces both with a parallel two-stage reduction (3.6 us for the
//! same shape). It is a *supplement*, not a replacement: the graph-only tiling
//! remains the portable fallback and is what runs anywhere this op is not
//! registered (non-CUDA providers, ORT builds without custom op support).
//!
//! # Contract
//!
//! - Domain [`DOMAIN`], op name [`OP_NAME`], provider `CUDAExecutionProvider`.
//! - Input 0: a tensor of `float`, `float16` or `bfloat16` with rank >= 1. All
//!   axes but the last are batch axes and are reduced independently, so the op
//!   is batch-aware and handles heterogeneous batches without rebinding.
//! - Output 0: `int64` with the input's shape minus its last axis, matching
//!   ONNX `ArgMax(axis=-1, keepdims=0, select_last_index=0)`.
//! - Ties resolve to the lowest index. `NaN` is never selected; a row that is
//!   entirely `NaN` yields index 0. This matches the device sampler's argmax
//!   and ORT's CUDA `ArgMax` for every input except rows whose leading elements
//!   are `NaN`, where ORT's sequential kernel is order-dependent and this op is
//!   not. ONNX leaves `NaN` undefined for `ArgMax`.
//!
//! # Capture safety
//!
//! Compilation, module loading and scratch allocation all happen when the
//! kernel is created (session initialisation), never in `Compute`. `Compute`
//! issues exactly two `cuLaunchKernel` calls on the stream ORT hands it and
//! performs no allocation, no synchronisation and no host transfer, so it is
//! safe to record into an ORT-managed CUDA graph. The only case that would
//! allocate is a batch larger than any seen before; that is rejected during
//! capture rather than silently perturbing the graph.

use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::sync::{Mutex, OnceLock};

use cudarc::driver::CudaContext;
use cudarc::driver::sys::{CUdeviceptr, CUfunction, CUmodule, CUstream};
use onnx_genai_ort_sys as sys;

use crate::device_sampler::{ARGMAX_SRC, BLOCK, argmax_parts, compile_image_for_device};
use crate::error::{OrtError, Result};

/// Custom op domain. Namespaced to this project so it cannot collide with
/// `com.microsoft` or `ai.onnx.contrib` ops a model might also use.
pub const DOMAIN: &str = "com.github.onnx_genai";

/// Custom op name.
pub const OP_NAME: &str = "ArgMaxLastAxis";

/// Environment switch. Set to `0` to keep every session on the portable
/// graph-only tiling even where the op could be registered, which is how the
/// two paths are compared.
const ENABLE_VAR: &str = "ONNX_GENAI_FUSED_ARGMAX";

/// Whether the fused op should be offered to CUDA sessions.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !matches!(std::env::var(ENABLE_VAR).as_deref(), Ok("0")))
}

// ---------------------------------------------------------------------------
// Device runtime
// ---------------------------------------------------------------------------

/// The compiled kernels for one device, plus one partial-result buffer per
/// stream that has used them.
struct Runtime {
    /// Kept alive so the primary context outlives the module.
    _ctx: std::sync::Arc<CudaContext>,
    _module: CUmodule,
    part_f32: CUfunction,
    part_f16: CUfunction,
    part_bf16: CUfunction,
    join: CUfunction,
    /// Keyed by the ORT stream the kernels are launched on. The part kernel
    /// writes this buffer and the join kernel reads it, so two executions may
    /// only share one buffer if something orders them — and within a CUDA
    /// stream, that ordering is guaranteed. ONNX Runtime documents concurrent
    /// `Run` calls on one session as safe and gives each concurrent run its own
    /// stream, so keying on the stream is what keeps one run's partials from
    /// being overwritten by another's before the join reads them. Executions
    /// that do share a stream (several sampler nodes in a graph, or repeated
    /// steps of a decode loop) are serialised by that stream and can safely
    /// reuse the buffer.
    scratch: Mutex<HashMap<usize, Scratch>>,
}

/// Growable device buffer holding `cap` partial `f32` values followed by `cap`
/// partial `i32` indices.
///
/// Growing keeps the superseded buffers alive until the process drops the
/// runtime. A CUDA graph records the *addresses* its kernels were captured
/// with, so freeing a buffer a captured graph still replays against would be a
/// use-after-free on every later replay of that shape's graph. Retiring instead
/// of freeing costs a few kilobytes per distinct batch size and keeps every
/// captured graph valid.
struct Scratch {
    ptr: CUdeviceptr,
    cap: usize,
    retired: Vec<CUdeviceptr>,
}

// SAFETY: the module and functions are immutable after construction and belong
// to the device's primary context, which every thread binds before use; the
// growable scratch is behind its own `Mutex`.
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl Runtime {
    fn new(device: usize) -> Result<Self> {
        let ctx = CudaContext::new(device).map_err(|e| {
            OrtError::Cuda(format!("fused argmax: CudaContext::new({device}): {e:?}"))
        })?;
        let capability = crate::device_sampler::compute_capability(&ctx)?;
        let image = compile_image_for_device(ARGMAX_SRC, "fused argmax", capability)?;
        // SAFETY: `image` is a cubin or NUL-terminated PTX produced above, and
        // the primary context is current after `CudaContext::new`.
        let module = unsafe { cudarc::driver::result::module::load_data(image.as_ptr().cast()) }
            .map_err(|e| OrtError::Cuda(format!("fused argmax: load module: {e:?}")))?;
        let get = |name: &str| -> Result<CUfunction> {
            let c = CString::new(name).expect("kernel name has no interior NUL");
            // SAFETY: `module` was just loaded and exports this entry point.
            unsafe { cudarc::driver::result::module::get_function(module, c) }
                .map_err(|e| OrtError::Cuda(format!("fused argmax: load {name}: {e:?}")))
        };
        Ok(Self {
            part_f32: get("argmax_part_f32")?,
            part_f16: get("argmax_part_f16")?,
            part_bf16: get("argmax_part_bf16")?,
            join: get("argmax_join_i64")?,
            _module: module,
            _ctx: ctx,
            scratch: Mutex::new(HashMap::new()),
        })
    }

    /// Ensure `stream`'s buffer has room for `pairs` partial results, returning
    /// its value and index pointers. Growth allocates, so callers must reserve
    /// every shape they will capture before capturing it.
    fn reserve(&self, stream: CUstream, pairs: usize) -> Result<(CUdeviceptr, CUdeviceptr)> {
        let mut scratch = self.scratch.lock().expect("fused argmax scratch poisoned");
        let entry = scratch.entry(stream as usize).or_insert(Scratch {
            ptr: 0,
            cap: 0,
            retired: Vec::new(),
        });
        if pairs > entry.cap {
            let bytes = pairs
                .checked_mul(8)
                .ok_or_else(|| OrtError::Cuda("fused argmax scratch overflow".into()))?;
            // SAFETY: primary context is current; the old buffer is retired
            // only after the new one exists, so a failure leaves it usable.
            let new_ptr = unsafe { cudarc::driver::result::malloc_sync(bytes) }
                .map_err(|e| OrtError::Cuda(format!("fused argmax: grow scratch: {e:?}")))?;
            if entry.cap > 0 {
                // Retired rather than freed: an already-captured graph replays
                // against the old address.
                entry.retired.push(entry.ptr);
            }
            entry.ptr = new_ptr;
            entry.cap = pairs;
        }
        Ok((entry.ptr, entry.ptr + (entry.cap * 4) as CUdeviceptr))
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        if let Ok(scratch) = self.scratch.lock() {
            let live = scratch
                .values()
                .filter(|entry| entry.cap > 0)
                .flat_map(|entry| std::iter::once(entry.ptr).chain(entry.retired.iter().copied()));
            for ptr in live {
                // SAFETY: every pointer came from `malloc_sync` and each is
                // freed exactly once, here, after the last graph that could
                // replay against it is gone.
                let _ = unsafe { cudarc::driver::result::free_sync(ptr) };
            }
        }
    }
}

/// Largest device ordinal this op serves. A module is loaded per device because
/// each device has its own primary context, and a kernel can only be launched on
/// a stream belonging to the context its module was loaded into.
const MAX_DEVICES: usize = 16;

/// Per-device runtime, built on first use. The error is cached too: a machine
/// that cannot compile these kernels will not retry once per session.
fn runtime_for(device: usize) -> std::result::Result<&'static Runtime, &'static str> {
    static RUNTIMES: [OnceLock<std::result::Result<Runtime, String>>; MAX_DEVICES] =
        [const { OnceLock::new() }; MAX_DEVICES];
    let slot = RUNTIMES
        .get(device)
        .ok_or("device ordinal is beyond the supported range")?;
    match slot.get_or_init(|| Runtime::new(device).map_err(|e| e.to_string())) {
        Ok(runtime) => Ok(runtime),
        Err(message) => Err(message.as_str()),
    }
}

/// The device a device pointer's memory belongs to.
///
/// The kernel must run where the data is, and a session's device is not visible
/// through `OrtKernelInfo`, so this asks the driver about the tensor ORT just
/// handed over.
fn device_of(ptr: CUdeviceptr) -> std::result::Result<usize, String> {
    use cudarc::driver::sys::{CUpointer_attribute, cuPointerGetAttribute};
    let mut ordinal: i32 = -1;
    // SAFETY: `ptr` is a live device pointer from ORT and `ordinal` is a valid
    // out-parameter for this attribute's `int` result.
    let status = unsafe {
        cuPointerGetAttribute(
            std::ptr::addr_of_mut!(ordinal).cast(),
            CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
            ptr,
        )
    };
    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS || ordinal < 0 {
        return Err(format!(
            "could not resolve the input tensor device: {status:?}"
        ));
    }
    Ok(ordinal as usize)
}

// ---------------------------------------------------------------------------
// Custom op
// ---------------------------------------------------------------------------

/// Per-node kernel state. The compiled kernels are process-wide; a node only
/// remembers the largest batch it has reserved scratch for.
struct Kernel {
    /// Device this node's tensors live on, resolved from the first input.
    device: OnceLock<usize>,
}

const ELEM_FLOAT: sys::ONNXTensorElementDataType = sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
const ELEM_FLOAT16: sys::ONNXTensorElementDataType = sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
const ELEM_BFLOAT16: sys::ONNXTensorElementDataType = sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
const ELEM_INT64: sys::ONNXTensorElementDataType = sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
const ELEM_UNDEFINED: sys::ONNXTensorElementDataType = sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

/// Build an `OrtStatus` carrying `message` as a FAIL.
fn fail(api: &sys::OrtApi, message: &str) -> sys::OrtStatusPtr {
    let create = api.CreateStatus.expect("ORT always provides CreateStatus");
    let text = CString::new(message).unwrap_or_else(|_| c"fused argmax failed".to_owned());
    // SAFETY: `text` outlives the call; ORT copies the message.
    unsafe { create(sys::ORT_FAIL, text.as_ptr()) }
}

unsafe extern "C" fn create_kernel(
    _op: *const sys::OrtCustomOp,
    api: *const sys::OrtApi,
    _info: *const sys::OrtKernelInfo,
    out: *mut *mut c_void,
) -> sys::OrtStatusPtr {
    // SAFETY: ORT passes a valid API table for the kernel's lifetime.
    let api = unsafe { &*api };
    // Surface a compile/driver problem at session creation, where it names the
    // op, rather than at the first token. The node's real device is not known
    // until it runs, so this compiles for device 0; a multi-device process
    // compiles again, once, for each other device it touches.
    if let Err(message) = runtime_for(0) {
        return fail(api, &format!("{OP_NAME}: {message}"));
    }
    let kernel = Box::new(Kernel {
        device: OnceLock::new(),
    });
    // SAFETY: `out` is ORT's out-parameter; ownership passes to ORT, which
    // returns it to `destroy_kernel`.
    unsafe { *out = Box::into_raw(kernel).cast() };
    std::ptr::null_mut()
}

unsafe extern "C" fn destroy_kernel(op_kernel: *mut c_void) {
    if !op_kernel.is_null() {
        // SAFETY: `op_kernel` came from `Box::into_raw` in `create_kernel`.
        drop(unsafe { Box::from_raw(op_kernel.cast::<Kernel>()) });
    }
}

/// Read a tensor's element type and shape.
unsafe fn tensor_shape(
    api: &sys::OrtApi,
    value: *const sys::OrtValue,
) -> std::result::Result<(sys::ONNXTensorElementDataType, Vec<i64>), String> {
    let get_info = api.GetTensorTypeAndShape.ok_or("GetTensorTypeAndShape")?;
    let mut info = std::ptr::null_mut();
    // SAFETY: `value` is a live tensor supplied by ORT.
    let status = unsafe { get_info(value, &mut info) };
    if !status.is_null() {
        return Err("GetTensorTypeAndShape failed".into());
    }
    let release = api.ReleaseTensorTypeAndShapeInfo;
    let mut elem = ELEM_UNDEFINED;
    let mut rank = 0usize;
    // SAFETY: `info` is live until released below.
    unsafe {
        if let Some(f) = api.GetTensorElementType {
            f(info, &mut elem);
        }
        if let Some(f) = api.GetDimensionsCount {
            f(info, &mut rank);
        }
    }
    let mut dims = vec![0i64; rank];
    // SAFETY: `dims` has exactly `rank` slots.
    unsafe {
        if let Some(f) = api.GetDimensions {
            f(info, dims.as_mut_ptr(), rank);
        }
        if let Some(f) = release {
            f(info);
        }
    }
    Ok((elem, dims))
}

unsafe extern "C" fn compute(
    op_kernel: *mut c_void,
    context: *mut sys::OrtKernelContext,
) -> sys::OrtStatusPtr {
    let api = match crate::error::api() {
        Ok(api) => api,
        Err(_) => return std::ptr::null_mut(),
    };
    // SAFETY: `op_kernel` is the `Kernel` created by `create_kernel`.
    let kernel = unsafe { &*op_kernel.cast::<Kernel>() };
    match unsafe { compute_inner(api, kernel, context) } {
        Ok(()) => std::ptr::null_mut(),
        Err(message) => fail(api, &format!("{OP_NAME}: {message}")),
    }
}

unsafe fn compute_inner(
    api: &sys::OrtApi,
    kernel: &Kernel,
    context: *mut sys::OrtKernelContext,
) -> std::result::Result<(), String> {
    let get_input = api.KernelContext_GetInput.ok_or("KernelContext_GetInput")?;
    let mut input = std::ptr::null();
    // SAFETY: `context` is ORT's live kernel context and index 0 exists.
    let status = unsafe { get_input(context, 0, &mut input) };
    if !status.is_null() || input.is_null() {
        return Err("input 0 is missing".into());
    }
    // SAFETY: `input` is a live tensor.
    let (elem, dims) = unsafe { tensor_shape(api, input) }?;
    if dims.is_empty() {
        return Err("input must have rank >= 1".into());
    }
    let vocab = dims[dims.len() - 1];
    if vocab <= 0 {
        return Err(format!("last axis must be positive, got {vocab}"));
    }
    let out_dims = &dims[..dims.len() - 1];
    let rows: i64 = out_dims.iter().product();
    if rows <= 0 {
        return Err(format!("batch axes must be positive, got {out_dims:?}"));
    }

    let get_output = api
        .KernelContext_GetOutput
        .ok_or("KernelContext_GetOutput")?;
    let mut output = std::ptr::null_mut();
    // SAFETY: `out_dims` is a valid slice of `out_dims.len()` dimensions.
    let status = unsafe { get_output(context, 0, out_dims.as_ptr(), out_dims.len(), &mut output) };
    if !status.is_null() || output.is_null() {
        return Err("output 0 could not be allocated".into());
    }

    let get_data = api.GetTensorData.ok_or("GetTensorData")?;
    let get_mut = api.GetTensorMutableData.ok_or("GetTensorMutableData")?;
    let mut src = std::ptr::null();
    let mut dst = std::ptr::null_mut();
    // SAFETY: both values are live tensors owned by ORT for this call.
    unsafe {
        get_data(input, &mut src);
        get_mut(output, &mut dst);
    }
    if src.is_null() || dst.is_null() {
        return Err("tensor data pointer was null".into());
    }

    let device = match kernel.device.get() {
        Some(device) => *device,
        None => *kernel
            .device
            .get_or_init(|| device_of(src as CUdeviceptr).unwrap_or(usize::MAX)),
    };
    if device == usize::MAX {
        return Err("the input tensor is not device memory".into());
    }
    let runtime = runtime_for(device)?;
    let part = match elem {
        ELEM_FLOAT => runtime.part_f32,
        ELEM_FLOAT16 => runtime.part_f16,
        ELEM_BFLOAT16 => runtime.part_bf16,
        other => return Err(format!("unsupported input element type {other}")),
    };

    let get_stream = api
        .KernelContext_GetGPUComputeStream
        .ok_or("KernelContext_GetGPUComputeStream")?;
    let mut stream: *mut c_void = std::ptr::null_mut();
    // SAFETY: `context` is live; a null stream means ORT's default stream.
    unsafe { get_stream(context, &mut stream) };
    let stream = stream as CUstream;

    let rows = rows as usize;
    let vocab_usize = usize::try_from(vocab).map_err(|_| "last axis exceeds usize")?;
    let parts = argmax_parts(vocab_usize).max(1);
    let pairs = rows
        .checked_mul(parts)
        .ok_or("partial result count overflow")?;
    // Only a batch larger than any seen before allocates; every other step just
    // reads the current pointers. Replays of a captured graph never reach this
    // code at all, because ORT does not call `Compute` for a replayed node.
    let (pval, pidx) = runtime.reserve(stream, pairs).map_err(|e| e.to_string())?;

    let rows_i = i32::try_from(rows).map_err(|_| "batch exceeds i32")?;
    let vocab_i = i32::try_from(vocab_usize).map_err(|_| "last axis exceeds i32")?;
    let parts_i = parts as i32;
    let src_ptr = src as CUdeviceptr;
    let dst_ptr = dst as CUdeviceptr;

    let mut part_args: [*mut c_void; 6] = [
        std::ptr::addr_of!(src_ptr) as *mut c_void,
        std::ptr::addr_of!(rows_i) as *mut c_void,
        std::ptr::addr_of!(vocab_i) as *mut c_void,
        std::ptr::addr_of!(parts_i) as *mut c_void,
        std::ptr::addr_of!(pval) as *mut c_void,
        std::ptr::addr_of!(pidx) as *mut c_void,
    ];
    // SAFETY: the argument list matches the kernel's
    // (const T*, int, int, int, float*, int*) signature; `src_ptr` covers
    // `rows * vocab` elements and the scratch holds `rows * parts` pairs.
    unsafe {
        cudarc::driver::result::launch_kernel(
            part,
            (parts_i as u32, rows_i as u32, 1),
            (BLOCK, 1, 1),
            0,
            stream,
            &mut part_args,
        )
    }
    .map_err(|e| format!("launch part kernel: {e:?}"))?;

    let mut join_args: [*mut c_void; 5] = [
        std::ptr::addr_of!(pval) as *mut c_void,
        std::ptr::addr_of!(pidx) as *mut c_void,
        std::ptr::addr_of!(rows_i) as *mut c_void,
        std::ptr::addr_of!(parts_i) as *mut c_void,
        std::ptr::addr_of!(dst_ptr) as *mut c_void,
    ];
    // SAFETY: the argument list matches the join kernel's
    // (const float*, const int*, int, int, long long*) signature; the output
    // holds `rows` int64 slots.
    unsafe {
        cudarc::driver::result::launch_kernel(
            runtime.join,
            (rows_i as u32, 1, 1),
            (BLOCK, 1, 1),
            0,
            stream,
            &mut join_args,
        )
    }
    .map_err(|e| format!("launch join kernel: {e:?}"))?;
    Ok(())
}

unsafe extern "C" fn op_name(_op: *const sys::OrtCustomOp) -> *const c_char {
    c"ArgMaxLastAxis".as_ptr()
}

unsafe extern "C" fn op_provider(_op: *const sys::OrtCustomOp) -> *const c_char {
    c"CUDAExecutionProvider".as_ptr()
}

unsafe extern "C" fn input_count(_op: *const sys::OrtCustomOp) -> usize {
    1
}

unsafe extern "C" fn output_count(_op: *const sys::OrtCustomOp) -> usize {
    1
}

/// Undefined means "any tensor element type": the kernel dispatches on the
/// actual type at compute time, so one registration covers f32/f16/bf16.
unsafe extern "C" fn input_type(
    _op: *const sys::OrtCustomOp,
    _index: usize,
) -> sys::ONNXTensorElementDataType {
    ELEM_UNDEFINED
}

unsafe extern "C" fn output_type(
    _op: *const sys::OrtCustomOp,
    _index: usize,
) -> sys::ONNXTensorElementDataType {
    ELEM_INT64
}

/// Both the input and the output are plain required tensors. ORT calls these
/// while building the op's schema, without checking whether they are set, so
/// they must exist even though they only ever answer "required".
unsafe extern "C" fn input_characteristic(
    _op: *const sys::OrtCustomOp,
    _index: usize,
) -> sys::OrtCustomOpInputOutputCharacteristic {
    sys::INPUT_OUTPUT_REQUIRED
}

unsafe extern "C" fn output_characteristic(
    _op: *const sys::OrtCustomOp,
    _index: usize,
) -> sys::OrtCustomOpInputOutputCharacteristic {
    sys::INPUT_OUTPUT_REQUIRED
}

/// The kernel reads device memory: the logits stay where the decoder wrote them.
unsafe extern "C" fn input_memory_type(
    _op: *const sys::OrtCustomOp,
    _index: usize,
) -> sys::OrtMemType {
    sys::OrtMemTypeDefault
}

/// Arity answers for the variadic case, which this op does not use. They exist
/// because ORT reads them while building the schema.
unsafe extern "C" fn variadic_min_arity(_op: *const sys::OrtCustomOp) -> ::std::os::raw::c_int {
    1
}

unsafe extern "C" fn variadic_homogeneity(_op: *const sys::OrtCustomOp) -> ::std::os::raw::c_int {
    0
}

/// The lowest `OrtCustomOp::version` at which ORT dispatches to
/// `CreateKernelV2`/`KernelComputeV2` (`min_ort_version_with_compute_v2` in
/// ORT's `custom_ops.cc`). Declaring less makes ORT call the V1 entry points.
const MIN_VERSION_FOR_COMPUTE_V2: u32 = 16;

/// Wrapper making the op descriptor shareable. The descriptor is immutable
/// after construction and contains only function pointers.
struct OpDescriptor(sys::OrtCustomOp);

// SAFETY: `OrtCustomOp` is a table of `extern "C"` function pointers that ORT
// only reads.
unsafe impl Sync for OpDescriptor {}
unsafe impl Send for OpDescriptor {}

fn descriptor() -> &'static OpDescriptor {
    static OP: OnceLock<OpDescriptor> = OnceLock::new();
    OP.get_or_init(|| {
        // SAFETY: `OrtCustomOp` is a plain C struct of scalars and nullable
        // function pointers, so an all-zero value is a valid "nothing
        // provided" descriptor that the fields below then fill in.
        let mut op: sys::OrtCustomOp = unsafe { std::mem::zeroed() };
        // ORT only takes the status-returning `CreateKernelV2` /
        // `KernelComputeV2` path when the descriptor declares at least
        // [`MIN_VERSION_FOR_COMPUTE_V2`]; below that it calls the V1 entry
        // points, which are null here. Declaring exactly that version keeps the
        // descriptor to fields every supported ORT reads, so nothing newer
        // (shape inference, custom opset ranges, in-place hints) is consulted.
        op.version = MIN_VERSION_FOR_COMPUTE_V2;
        op.CreateKernelV2 = Some(create_kernel);
        op.KernelComputeV2 = Some(compute);
        op.KernelDestroy = Some(destroy_kernel);
        op.GetName = Some(op_name);
        op.GetExecutionProviderType = Some(op_provider);
        op.GetInputTypeCount = Some(input_count);
        op.GetInputType = Some(input_type);
        op.GetOutputTypeCount = Some(output_count);
        op.GetOutputType = Some(output_type);
        op.GetInputCharacteristic = Some(input_characteristic);
        op.GetOutputCharacteristic = Some(output_characteristic);
        op.GetInputMemoryType = Some(input_memory_type);
        op.GetVariadicInputMinArity = Some(variadic_min_arity);
        op.GetVariadicInputHomogeneity = Some(variadic_homogeneity);
        op.GetVariadicOutputMinArity = Some(variadic_min_arity);
        op.GetVariadicOutputHomogeneity = Some(variadic_homogeneity);
        OpDescriptor(op)
    })
}

/// A registered custom op domain. ORT requires the domain to outlive every
/// session created with it, so this is created once and never released.
struct Domain(*mut sys::OrtCustomOpDomain);

// SAFETY: the handle is only read after construction and is never released.
unsafe impl Sync for Domain {}
unsafe impl Send for Domain {}

fn domain() -> std::result::Result<*mut sys::OrtCustomOpDomain, &'static str> {
    static REGISTERED: OnceLock<std::result::Result<Domain, String>> = OnceLock::new();
    let built = REGISTERED.get_or_init(|| {
        let api = crate::error::api().map_err(|e| e.to_string())?;
        let create = api
            .CreateCustomOpDomain
            .ok_or_else(|| "this ORT build has no CreateCustomOpDomain".to_string())?;
        let add = api
            .CustomOpDomain_Add
            .ok_or_else(|| "this ORT build has no CustomOpDomain_Add".to_string())?;
        let name = CString::new(DOMAIN).expect("domain name has no interior NUL");
        let mut handle = std::ptr::null_mut();
        // SAFETY: `name` outlives the call; ORT copies the domain name.
        crate::error::check_status(unsafe { create(name.as_ptr(), &mut handle) })
            .map_err(|e| e.to_string())?;
        // SAFETY: the descriptor is `'static`, satisfying ORT's requirement
        // that it outlive the domain.
        crate::error::check_status(unsafe { add(handle, &descriptor().0) })
            .map_err(|e| e.to_string())?;
        Ok(Domain(handle))
    });
    match built {
        Ok(domain) => Ok(domain.0),
        Err(message) => Err(message.as_str()),
    }
}

/// Register the fused op on `options`, if it is enabled and this machine can
/// build the kernels.
///
/// Registration failure is not fatal: the graph-only tiling still produces the
/// same answer, so the session is created either way and the reason is logged
/// once. Returns whether the op was registered.
pub(crate) fn register(options: *mut sys::OrtSessionOptions) -> bool {
    if !enabled() {
        return false;
    }
    let Ok(api) = crate::error::api() else {
        return false;
    };
    let Some(add) = api.AddCustomOpDomain else {
        return false;
    };
    match domain() {
        Ok(handle) => {
            // SAFETY: `handle` is a live domain that is never released, and
            // `options` is a live session options handle.
            match crate::error::check_status(unsafe { add(options, handle) }) {
                Ok(()) => true,
                Err(error) => {
                    tracing::debug!(%error, "fused argmax op could not be added to the session");
                    false
                }
            }
        }
        Err(message) => {
            tracing::debug!(reason = message, "fused argmax op is unavailable");
            false
        }
    }
}

/// Whether a CUDA session would get the fused op, used by the graph planner to
/// decide between emitting it and emitting the portable tiling.
pub fn available() -> bool {
    enabled() && domain().is_ok() && runtime_for(0).is_ok()
}

/// The op's qualified name, for planners emitting the node.
pub fn qualified_name() -> String {
    format!("{DOMAIN}::{OP_NAME}")
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::*;

    #[test]
    fn descriptor_is_complete() {
        let op = &descriptor().0;
        assert_eq!(op.version, MIN_VERSION_FOR_COMPUTE_V2);
        assert!(op.CreateKernelV2.is_some());
        assert!(op.KernelComputeV2.is_some());
        assert!(op.KernelDestroy.is_some());
        // V1 entry points must stay unset so ORT takes the status-returning
        // V2 path rather than the aborting V1 one.
        assert!(op.CreateKernel.is_none());
        assert!(op.KernelCompute.is_none());
        // ORT reads these while building the schema without checking whether
        // they are set, so a missing one is a null call, not a default.
        assert!(op.GetInputCharacteristic.is_some());
        assert!(op.GetOutputCharacteristic.is_some());
        assert!(op.GetInputMemoryType.is_some());
        assert!(op.GetVariadicInputMinArity.is_some());
        assert!(op.GetVariadicInputHomogeneity.is_some());
        assert!(op.GetVariadicOutputMinArity.is_some());
        assert!(op.GetVariadicOutputHomogeneity.is_some());
    }

    #[test]
    fn descriptor_declares_one_int64_output() {
        let op = &descriptor().0;
        let name = unsafe { CStr::from_ptr(op.GetName.unwrap()(&raw const *op)) };
        assert_eq!(name.to_str().unwrap(), OP_NAME);
        let provider =
            unsafe { CStr::from_ptr(op.GetExecutionProviderType.unwrap()(&raw const *op)) };
        assert_eq!(provider.to_str().unwrap(), "CUDAExecutionProvider");
        assert_eq!(unsafe { op.GetInputTypeCount.unwrap()(&raw const *op) }, 1);
        assert_eq!(unsafe { op.GetOutputTypeCount.unwrap()(&raw const *op) }, 1);
        assert_eq!(
            unsafe { op.GetOutputType.unwrap()(&raw const *op, 0) },
            ELEM_INT64
        );
        // Any input element type: the kernel dispatches at compute time.
        assert_eq!(
            unsafe { op.GetInputType.unwrap()(&raw const *op, 0) },
            ELEM_UNDEFINED
        );
    }

    #[test]
    fn qualified_name_is_namespaced() {
        assert_eq!(qualified_name(), "com.github.onnx_genai::ArgMaxLastAxis");
    }
}
