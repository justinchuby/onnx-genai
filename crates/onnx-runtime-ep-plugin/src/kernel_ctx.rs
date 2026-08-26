//! `OutboundKernelContext` — bridges `OrtKernelContext` ↔ nxrt `TensorView`/`TensorMut`.
//!
//! When ORT calls our `Compute` callback, it passes an `OrtKernelContext*`. We
//! read inputs via `KernelContext_GetInput` and write outputs via
//! `KernelContext_GetOutput`, converting between ORT's `OrtValue*` and our
//! `TensorView`/`TensorMut`.
//!
//! # f16/bf16 marshaling
//!
//! `ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10` maps to `DataType::Float16 = 10`
//! and `ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16` maps to `DataType::BFloat16 = 16`
//! via [`DataType::from_onnx`]. Both have `byte_size() = 2`, so the existing
//! `checked_mul` overflow guards in [`validate_dims`] handle them correctly.
//! Unsupported ORT element types are rejected by `from_onnx` returning `None`;
//! the error message names the raw numeric value so the caller can diagnose.
//!
//! The public constant [`CPU_EP_SUPPORTED_DTYPES`] enumerates every dtype the
//! CPU EP can accept. Deckard's `ep.rs` uses it to populate `GetKernelRegistry`
//! type constraints — **do not duplicate this list** in `ep.rs`; import it here.
//!
//! Input residency is preserved from ORT's `OrtMemoryInfo`. Device pointers are
//! never labelled as CPU merely because `GetTensorData` returned a non-null
//! address.

use std::ffi::{CStr, c_void};

use onnx_genai_ort_sys as ort;
use onnx_runtime_ep_api::tensor::{DevicePtr, DevicePtrMut, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, DeviceId, DeviceType};

use crate::dim_vec::DimVec;

/// All element types the CPU EP can marshal across the ORT ABI.
///
/// Deckard's `ep.rs` must import this slice (do not copy-paste it) to populate
/// `GetKernelRegistry` type constraints, ensuring the EP's type-constraint
/// advertisement and this marshaling layer stay in sync.
///
/// Mapping to ORT enum values (verified against `bindings.rs`):
/// - `Float32`  → 1   (`ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT`)
/// - `Uint8`    → 2   (`ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8`)
/// - `Int8`     → 3   (`ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8`)
/// - `Uint16`   → 4   (`ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16`)
/// - `Int16`    → 5   (`ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16`)
/// - `Int32`    → 6   (`ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32`)
/// - `Int64`    → 7   (`ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64`)
/// - `Bool`     → 9   (`ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL`)
/// - `Float16`  → 10  (`ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16`)  — 2 bytes/element
/// - `Float64`  → 11  (`ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE`)
/// - `Uint32`   → 12  (`ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32`)
/// - `Uint64`   → 13  (`ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64`)
/// - `BFloat16` → 16  (`ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16`) — 2 bytes/element
pub const CPU_EP_SUPPORTED_DTYPES: &[DataType] = &[
    DataType::Float32,
    DataType::Uint8,
    DataType::Int8,
    DataType::Uint16,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::Bool,
    DataType::Float16,
    DataType::Float64,
    DataType::Uint32,
    DataType::Uint64,
    DataType::BFloat16,
];

/// Row-major strides for `shape`, without touching the allocator for an
/// ordinary rank.
///
/// Deliberately not a call to `onnx_runtime_ir::compute_contiguous_strides`:
/// that one returns a `Vec` and so allocates for every operand of every
/// dispatch. The arithmetic is identical and pinned to it by
/// `contiguous_strides_matches_the_ir_crate` below, which is what keeps the
/// duplication honest.
fn contiguous_strides(shape: &[usize]) -> DimVec<i64> {
    let n = shape.len();
    let mut strides = DimVec::zeroed(n);
    // Take the storage once. Indexing a `DimVec` goes through `DerefMut`, which
    // re-matches `Inline` against `Heap` on every access; the loop below reads
    // and writes twice per element, so the representation was being resolved
    // four times per dimension to fill a buffer whose length is known here.
    // Carrying the running product in a register also removes the read of
    // `strides[i + 1]`, which was a loop-carried memory dependency.
    let out = strides.as_mut_slice();
    if n > 0 {
        out[n - 1] = 1;
        let mut acc: i64 = 1;
        for i in (0..n - 1).rev() {
            // Same products in the same order as reading back `strides[i + 1]`,
            // so this overflows for exactly the inputs the previous form did.
            acc *= shape[i + 1] as i64;
            out[i] = acc;
        }
    }
    strides
}

/// Validate raw ORT dimensions for a single tensor, converting to `usize` shape.
///
/// Rejects negative dims (fail closed) and detects element-count overflow.
/// Returns `(shape, element_count, expected_byte_length)`.
pub(crate) fn validate_dims(
    dims: &[i64],
    dtype: DataType,
    context: impl std::fmt::Display,
) -> Result<(DimVec<usize>, usize, usize), String> {
    // `zeroed` + fill rather than `with_capacity` + `push`: the length is known
    // here, and `push` re-matches the representation on every element. This is
    // the case `DimVec::zeroed` was introduced for.
    let mut shape: DimVec<usize> = DimVec::zeroed(dims.len());
    for (dim_idx, (slot, &d)) in shape.as_mut_slice().iter_mut().zip(dims).enumerate() {
        if d < 0 {
            return Err(format!(
                "{context} dim[{dim_idx}] is {d} — negative dimensions are \
                 invalid at Compute time (symbolic/dynamic dims must be \
                 resolved before execution)"
            ));
        }
        *slot = d as usize;
    }

    let element_count: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| format!("{context} shape {shape:?} overflows usize in element count"))?;

    let byte_size = dtype.byte_size();
    let expected_bytes = element_count.checked_mul(byte_size).ok_or_else(|| {
        format!(
            "{context} byte length overflows: {element_count} elements × \
             {byte_size} bytes/element"
        )
    })?;

    Ok((shape, element_count, expected_bytes))
}

/// Owned input tensor data extracted from an `OrtKernelContext`.
///
/// Holds the shape/strides vectors so that `TensorView` references remain valid.
pub(crate) struct OwnedInput {
    pub data_ptr: *const c_void,
    pub dtype: DataType,
    pub shape: DimVec<usize>,
    pub strides: DimVec<i64>,
    pub device: DeviceId,
}

impl OwnedInput {
    /// Create a `TensorView` borrowing from this owned input.
    pub fn view(&self) -> TensorView<'_> {
        TensorView::new(
            DevicePtr(self.data_ptr),
            self.dtype,
            &self.shape,
            &self.strides,
            self.device,
        )
    }
}

/// Resolve one ORT tensor's actual allocator residency.
///
/// ORT's legacy `CreateMemoryInfo` can report a CPU device type for a CUDA
/// allocation, so the allocator name remains authoritative for known device
/// allocators. Unknown non-CPU kinds stay non-host-accessible via `Custom`;
/// guessing CPU would make a later shape-data read undefined behavior.
unsafe fn value_device(
    api: &ort::OrtApi,
    value: *const ort::OrtValue,
    input_index: usize,
) -> Result<DeviceId, String> {
    let get_memory_info = api
        .GetTensorMemoryInfo
        .ok_or("OrtApi.GetTensorMemoryInfo is null")?;
    let mut memory_info = std::ptr::null();
    crate::dispatch_probe::ort_call();
    let status = unsafe { get_memory_info(value, &mut memory_info) };
    if !status.is_null() || memory_info.is_null() {
        return Err(format!(
            "GetTensorMemoryInfo failed for input {input_index}"
        ));
    }
    unsafe { device_from_memory_info(api, memory_info, format_args!("input {input_index}")) }
}

fn raw_device_type_code<T>(
    raw_device_type: T,
    context: impl std::fmt::Display,
) -> Result<u32, String>
where
    T: Copy + std::fmt::Display,
    u32: TryFrom<T>,
{
    u32::try_from(raw_device_type).map_err(|_| {
        format!(
            "MemoryInfoGetDeviceType returned invalid raw device type {raw_device_type} for \
             {context}"
        )
    })
}

pub(crate) unsafe fn device_from_memory_info(
    api: &ort::OrtApi,
    memory_info: *const ort::OrtMemoryInfo,
    context: impl std::fmt::Display + Copy,
) -> Result<DeviceId, String> {
    let get_device_type = api
        .MemoryInfoGetDeviceType
        .ok_or("OrtApi.MemoryInfoGetDeviceType is null")?;
    let get_name = api
        .MemoryInfoGetName
        .ok_or("OrtApi.MemoryInfoGetName is null")?;
    let get_id = api
        .MemoryInfoGetId
        .ok_or("OrtApi.MemoryInfoGetId is null")?;
    if memory_info.is_null() {
        return Err(format!("{context} has null OrtMemoryInfo"));
    }
    let mut raw_device_type = ort::OrtMemoryInfoDeviceType_CPU;
    crate::dispatch_probe::ort_call();
    unsafe { get_device_type(memory_info, &mut raw_device_type) };

    let mut name_ptr = std::ptr::null();
    crate::dispatch_probe::ort_call();
    let status = unsafe { get_name(memory_info, &mut name_ptr) };
    if !status.is_null() || name_ptr.is_null() {
        return Err(format!("MemoryInfoGetName failed for {context}"));
    }
    let name = unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy();

    let mut raw_id = 0i32;
    crate::dispatch_probe::ort_call();
    let status = unsafe { get_id(memory_info, &mut raw_id) };
    if !status.is_null() || raw_id < 0 {
        return Err(format!(
            "MemoryInfoGetId failed for {context} or returned invalid id {raw_id}"
        ));
    }
    let index = raw_id as u32;
    let normalized = name.trim().to_ascii_lowercase();
    let device_type = match normalized.as_str() {
        "cuda" => DeviceType::Cuda,
        "rocm" => DeviceType::Rocm,
        "coreml" => DeviceType::CoreMl,
        "mlx" | "metal" => DeviceType::Mlx,
        "webgpu" | "webgpu_buffer" => DeviceType::WebGpu,
        "qnn" => DeviceType::Qnn,
        "openvino" => DeviceType::OpenVino,
        _ if raw_device_type == ort::OrtMemoryInfoDeviceType_CPU => DeviceType::Cpu,
        _ => DeviceType::Custom(raw_device_type_code(raw_device_type, context)?),
    };
    Ok(DeviceId::new(device_type, index))
}

/// Owned output tensor data allocated via `KernelContext_GetOutput`.
pub(crate) struct OwnedOutput {
    pub data_ptr: *mut c_void,
    pub dtype: DataType,
    pub shape: DimVec<usize>,
    pub strides: DimVec<i64>,
    /// Memory info ORT placed this output on. For a boundary GPU node ORT
    /// places the output on the device, so this is a *valid device*
    /// `OrtMemoryInfo` the executor can reuse to allocate staging scratch for
    /// host-resident inputs (#982). Null if it could not be read.
    pub mem_info: *const ort::OrtMemoryInfo,
}

impl OwnedOutput {
    /// Create a `TensorMut` borrowing from this owned output.
    pub fn view_mut(&mut self) -> TensorMut<'_> {
        TensorMut::new(
            DevicePtrMut(self.data_ptr),
            self.dtype,
            &self.shape,
            &self.strides,
            DeviceId::cpu(),
        )
    }
}

/// Read all inputs from an `OrtKernelContext` using the host `OrtApi`.
///
/// # Safety
///
/// `api` must be valid. `ctx` must be a valid `OrtKernelContext*`.
pub(crate) unsafe fn read_inputs_into(
    api: &ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
    inputs: &mut Vec<OwnedInput>,
) -> Result<(), String> {
    let _probe = crate::dispatch_probe::Phase::MetadataQuery.enter();
    let get_input_count = api
        .KernelContext_GetInputCount
        .ok_or("OrtApi.KernelContext_GetInputCount is null")?;
    let get_input = api
        .KernelContext_GetInput
        .ok_or("OrtApi.KernelContext_GetInput is null")?;
    // One call that yields both the element type and a *reference* to the
    // `OrtValue`'s own shape array. It replaces the five-call
    // `GetTensorTypeAndShape` / `GetTensorElementType` / `GetDimensionsCount` /
    // `GetDimensions` / `ReleaseTensorTypeAndShapeInfo` sequence below, and
    // spares ORT the `OrtTensorTypeAndShapeInfo` it had to allocate and free
    // for every input of every `Run`. Available since ORT API 24 and the
    // plugin fails closed below API 27, so the fallback is unreachable in
    // practice — it is kept so a host that leaves the hook null still works.
    let get_type_and_shape_ref = api.GetTensorElementTypeAndShapeDataReference;
    let legacy_shape_hooks = match (
        api.GetTensorTypeAndShape,
        api.GetTensorElementType,
        api.GetDimensionsCount,
        api.GetDimensions,
        api.ReleaseTensorTypeAndShapeInfo,
    ) {
        (Some(a), Some(b), Some(c), Some(d), Some(e)) => Some((a, b, c, d, e)),
        _ => None,
    };
    if get_type_and_shape_ref.is_none() && legacy_shape_hooks.is_none() {
        return Err(
            "OrtApi exposes neither GetTensorElementTypeAndShapeDataReference nor the \
                    GetTensorTypeAndShape family; input shapes cannot be read"
                .into(),
        );
    }
    let get_tensor_data = api.GetTensorData.ok_or("OrtApi.GetTensorData is null")?;

    let mut input_count: usize = 0;
    crate::dispatch_probe::ort_call();
    let status = unsafe { get_input_count(ctx, &mut input_count) };
    if !status.is_null() {
        return Err("KernelContext_GetInputCount failed".into());
    }

    // `Vec::with_capacity(0)` does not allocate, so a node with no inputs must
    // not be charged for this one.
    if input_count > 0 {
        crate::dispatch_probe::count(crate::dispatch_probe::Event::DispatchAlloc);
    }
    debug_assert!(
        inputs.is_empty(),
        "read_inputs_into was handed a dirty buffer"
    );
    inputs.reserve(input_count);
    for i in 0..input_count {
        let mut value: *const ort::OrtValue = std::ptr::null();
        crate::dispatch_probe::ort_call();
        let status = unsafe { get_input(ctx, i, &mut value) };
        if !status.is_null() {
            return Err(format!("KernelContext_GetInput({i}) failed"));
        }
        if value.is_null() {
            // Optional input not present — push absent placeholder.
            inputs.push(OwnedInput {
                data_ptr: std::ptr::null(),
                dtype: DataType::Float32,
                shape: DimVec::new(),
                strides: DimVec::new(),
                device: DeviceId::cpu(),
            });
            continue;
        }

        // Element type and dimensions.
        let mut elem_type: ort::ONNXTensorElementDataType = 0;
        let mut borrowed_dims: *const i64 = std::ptr::null();
        let mut borrowed_ndim: usize = 0;
        let mut owned_dims: DimVec<i64> = DimVec::new();
        let borrowed = match get_type_and_shape_ref {
            Some(get_ref) => {
                crate::dispatch_probe::ort_call();
                let status = unsafe {
                    get_ref(
                        value,
                        &mut elem_type,
                        &mut borrowed_dims,
                        &mut borrowed_ndim,
                    )
                };
                if !status.is_null() {
                    return Err(format!(
                        "GetTensorElementTypeAndShapeDataReference failed for input {i}"
                    ));
                }
                true
            }
            None => {
                let (get_type_shape, get_elem_type, get_dims_count, get_dims, release_type_shape) =
                    legacy_shape_hooks.expect("checked above: one of the two paths exists");
                let mut type_shape: *mut ort::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
                crate::dispatch_probe::ort_call();
                let status = unsafe { get_type_shape(value, &mut type_shape) };
                if !status.is_null() || type_shape.is_null() {
                    return Err(format!("GetTensorTypeAndShape failed for input {i}"));
                }
                crate::dispatch_probe::ort_call();
                let status = unsafe { get_elem_type(type_shape, &mut elem_type) };
                if !status.is_null() {
                    unsafe { release_type_shape(type_shape) };
                    return Err(format!("GetTensorElementType failed for input {i}"));
                }
                let mut ndim: usize = 0;
                crate::dispatch_probe::ort_call();
                let status = unsafe { get_dims_count(type_shape, &mut ndim) };
                if !status.is_null() {
                    unsafe { release_type_shape(type_shape) };
                    return Err(format!("GetDimensionsCount failed for input {i}"));
                }
                owned_dims = DimVec::zeroed(ndim);
                crate::dispatch_probe::count_n(
                    crate::dispatch_probe::Event::DispatchAlloc,
                    u64::from(ndim > crate::dim_vec::INLINE_RANK),
                );
                crate::dispatch_probe::ort_call();
                let status = unsafe { get_dims(type_shape, owned_dims.as_mut_ptr(), ndim) };
                if !status.is_null() {
                    unsafe { release_type_shape(type_shape) };
                    return Err(format!("GetDimensions failed for input {i}"));
                }
                crate::dispatch_probe::ort_call();
                unsafe { release_type_shape(type_shape) };
                false
            }
        };
        let dtype = DataType::from_onnx(elem_type as i32)
            .ok_or_else(|| format!("unsupported element type {elem_type} for input {i}"))?;
        // SAFETY: on the borrowed path ORT reports `shape_data` as its own
        // storage for this `OrtValue`, valid until the value is released or
        // reshaped — neither happens while this `Compute` call holds it. A
        // scalar is reported as a null pointer with count 0, which must not be
        // handed to `from_raw_parts`.
        let dims: &[i64] = if !borrowed {
            &owned_dims
        } else if borrowed_ndim == 0 || borrowed_dims.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(borrowed_dims, borrowed_ndim) }
        };

        // Validate ORT-supplied dims. The label is `format_args!`, which
        // borrows its arguments and formats only if an error is raised.
        //
        // `shape` and `strides` are `DimVec`s, so an ordinary rank costs no
        // allocation at all; only a tensor of rank > `INLINE_RANK` reaches the
        // allocator, and then it is charged below.
        let (shape, _, _) = validate_dims(dims, dtype, format_args!("input {i}"))?;
        let strides = contiguous_strides(&shape);
        crate::dispatch_probe::count_n(
            crate::dispatch_probe::Event::DispatchAlloc,
            u64::from(shape.len() > crate::dim_vec::INLINE_RANK) * 2,
        );

        // Data pointer.
        let mut data: *const c_void = std::ptr::null();
        crate::dispatch_probe::ort_call();
        let status = unsafe { get_tensor_data(value, &mut data) };
        if !status.is_null() {
            return Err(format!("GetTensorData failed for input {i}"));
        }
        if data.is_null() {
            return Err(format!(
                "input {i} data pointer is null (device-only memory not supported)"
            ));
        }
        let device = unsafe { value_device(api, value, i) }?;

        inputs.push(OwnedInput {
            data_ptr: data,
            dtype,
            shape,
            strides,
            device,
        });
    }

    Ok(())
}

/// [`read_inputs_into`] with a freshly allocated vector.
///
/// The production path supplies its own buffer from the per-thread [`RunScratch`]
/// so a `Run` does not allocate one; this wrapper exists for the tests, which
/// care about what is read and not about where the storage came from.
///
/// [`RunScratch`]: crate::compute::RunScratch
#[cfg(test)]
pub(crate) unsafe fn read_inputs(
    api: &ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
) -> Result<Vec<OwnedInput>, String> {
    let mut inputs = Vec::new();
    unsafe { read_inputs_into(api, ctx, &mut inputs) }?;
    Ok(inputs)
}

/// Allocate an output tensor in the `OrtKernelContext` with the given shape.
///
/// # Safety
///
/// `api` must be valid. `ctx` must be a valid `OrtKernelContext*`.
pub(crate) unsafe fn allocate_output(
    api: &ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
    index: usize,
    shape: &[usize],
    dtype: DataType,
    want_mem_info: bool,
) -> Result<OwnedOutput, String> {
    let _probe = crate::dispatch_probe::Phase::Allocate.enter();
    let get_output = api
        .KernelContext_GetOutput
        .ok_or("OrtApi.KernelContext_GetOutput is null")?;
    let get_mutable_data = api
        .GetTensorMutableData
        .ok_or("OrtApi.GetTensorMutableData is null")?;

    // ORT wants the shape as `i64`. Ranks this small are the whole population
    // of shapes this path sees, so the conversion goes to the stack and a
    // per-`Run`, per-output heap allocation disappears; a taller tensor still
    // works, through the `Vec`.
    //
    // Shares `DimVec`'s threshold rather than declaring a second one: this
    // path and the input path answer the same question about the same tensors,
    // and two constants that must agree are one constant waiting to disagree.
    use crate::dim_vec::INLINE_RANK;
    let mut inline_dims = [0i64; INLINE_RANK];
    let heap_dims: Vec<i64>;
    let dims: &[i64] = if shape.len() <= INLINE_RANK {
        for (slot, &d) in inline_dims.iter_mut().zip(shape) {
            *slot = d as i64;
        }
        &inline_dims[..shape.len()]
    } else {
        crate::dispatch_probe::count(crate::dispatch_probe::Event::DispatchAlloc);
        heap_dims = shape.iter().map(|&d| d as i64).collect();
        &heap_dims
    };
    let mut value: *mut ort::OrtValue = std::ptr::null_mut();
    crate::dispatch_probe::ort_call();
    let status = unsafe { get_output(ctx, index, dims.as_ptr(), dims.len(), &mut value) };
    if !status.is_null() {
        return Err(format!("KernelContext_GetOutput({index}) failed"));
    }
    if value.is_null() {
        return Err(format!("KernelContext_GetOutput({index}) returned null"));
    }

    let mut data: *mut c_void = std::ptr::null_mut();
    crate::dispatch_probe::ort_call();
    let status = unsafe { get_mutable_data(value, &mut data) };
    if !status.is_null() {
        return Err(format!("GetTensorMutableData failed for output {index}"));
    }
    if data.is_null() {
        return Err(format!(
            "output {index} data pointer is null (device-only memory not supported)"
        ));
    }

    // Best-effort read of the output's memory info. Used to source a valid
    // device `OrtMemoryInfo` for staging scratch (#982); a null result simply
    // means this output cannot be used as that source.
    //
    // Skipped entirely when the caller has no device staging configured. The
    // only consumer of this field is `stage_host_boundary_inputs`, which a host
    // EP never reaches, so on the CPU path this was one ORT FFI call per output
    // per `Run` whose result was dropped unread.
    let mem_info = match api.GetTensorMemoryInfo.filter(|_| want_mem_info) {
        Some(get_mem_info) => {
            let mut mi: *const ort::OrtMemoryInfo = std::ptr::null();
            crate::dispatch_probe::ort_call();
            let status = unsafe { get_mem_info(value, &mut mi) };
            if status.is_null() {
                mi
            } else {
                std::ptr::null()
            }
        }
        None => std::ptr::null(),
    };

    // An ordinary rank keeps both of these inline, so the common output costs
    // the allocator nothing. `count_n` charges only the ranks that spill.
    let shape: DimVec<usize> = DimVec::from_slice(shape);
    let strides = contiguous_strides(&shape);
    crate::dispatch_probe::count_n(
        crate::dispatch_probe::Event::DispatchAlloc,
        2 * u64::from(shape.len() > crate::dim_vec::INLINE_RANK),
    );
    Ok(OwnedOutput {
        data_ptr: data,
        dtype,
        shape,
        strides,
        mem_info,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::DataType;

    use std::sync::atomic::{AtomicUsize, Ordering};

    /// How many times the fake ORT below was asked for an output's memory info.
    static MEM_INFO_CALLS: AtomicUsize = AtomicUsize::new(0);
    /// Storage the fake `GetTensorMutableData` hands back.
    static mut FAKE_OUTPUT: [f32; 4] = [0.0; 4];
    /// The dims the recording fake `KernelContext_GetOutput` last saw.
    static RECORDED_DIMS: std::sync::Mutex<Vec<i64>> = std::sync::Mutex::new(Vec::new());

    /// Stands in for ORT: yields an opaque non-null `OrtValue`.
    ///
    /// # Safety
    ///
    /// Matches the ABI ORT expects; `out` must be a valid pointer.
    unsafe extern "C" fn fake_get_output(
        _ctx: *mut ort::OrtKernelContext,
        _index: usize,
        _dims: *const i64,
        _dim_count: usize,
        out: *mut *mut ort::OrtValue,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = std::ptr::dangling_mut::<ort::OrtValue>() };
        std::ptr::null_mut()
    }

    /// Stands in for ORT, recording the dims the caller passed so a test can
    /// prove the inline-rank fast path sends the same shape as the heap path.
    ///
    /// # Safety
    ///
    /// Matches the ABI ORT expects; `dims` must point at `dim_count` i64 and
    /// `out` must be a valid pointer.
    unsafe extern "C" fn recording_get_output(
        _ctx: *mut ort::OrtKernelContext,
        _index: usize,
        dims: *const i64,
        dim_count: usize,
        out: *mut *mut ort::OrtValue,
    ) -> ort::OrtStatusPtr {
        let seen = if dim_count == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(dims, dim_count) }.to_vec()
        };
        *RECORDED_DIMS.lock().expect("recorded dims lock") = seen;
        unsafe { *out = std::ptr::dangling_mut::<ort::OrtValue>() };
        std::ptr::null_mut()
    }

    /// Stands in for ORT: points at `FAKE_OUTPUT`.
    ///
    /// # Safety
    ///
    /// Matches the ABI ORT expects; `out` must be a valid pointer.
    unsafe extern "C" fn fake_get_mutable_data(
        _value: *mut ort::OrtValue,
        out: *mut *mut c_void,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = (&raw mut FAKE_OUTPUT).cast() };
        std::ptr::null_mut()
    }

    /// Stands in for ORT, counting how often placement is queried.
    ///
    /// # Safety
    ///
    /// Matches the ABI ORT expects; `mem_info` must be a valid pointer.
    unsafe extern "C" fn counting_get_mem_info(
        _value: *const ort::OrtValue,
        mem_info: *mut *const ort::OrtMemoryInfo,
    ) -> ort::OrtStatusPtr {
        MEM_INFO_CALLS.fetch_add(1, Ordering::Relaxed);
        unsafe { *mem_info = std::ptr::null() };
        std::ptr::null_mut()
    }

    /// `GetTensorMemoryInfo` is one ORT call per output per `Run`, and its only
    /// consumer is device staging. A host EP must not pay for it.
    ///
    /// Falsifier: make `allocate_output` ignore `want_mem_info` and the first
    /// assertion fails with 1 call instead of 0 (verified).
    #[test]
    fn output_memory_info_is_queried_only_when_the_caller_asked_for_it() {
        let mut api: ort::OrtApi = unsafe { std::mem::zeroed() };
        api.KernelContext_GetOutput = Some(fake_get_output);
        api.GetTensorMutableData = Some(fake_get_mutable_data);
        api.GetTensorMemoryInfo = Some(counting_get_mem_info);
        let ctx = std::ptr::dangling_mut::<ort::OrtKernelContext>();

        MEM_INFO_CALLS.store(0, Ordering::Relaxed);
        let out = unsafe { allocate_output(&api, ctx, 0, &[2, 2], DataType::Float32, false) }
            .expect("the fake ORT allocates successfully");
        assert_eq!(
            MEM_INFO_CALLS.load(Ordering::Relaxed),
            0,
            "a caller with no device staging must not query output placement"
        );
        assert!(out.mem_info.is_null());
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.strides, vec![2, 1]);

        let _ = unsafe { allocate_output(&api, ctx, 0, &[2, 2], DataType::Float32, true) }
            .expect("the fake ORT allocates successfully");
        assert_eq!(
            MEM_INFO_CALLS.load(Ordering::Relaxed),
            1,
            "a staging caller still gets the output's memory info"
        );
    }

    /// How many times the fake ORT below was asked for a shape *reference*.
    static SHAPE_REF_CALLS: AtomicUsize = AtomicUsize::new(0);
    /// How many times the fake ORT below was asked to allocate shape info.
    static LEGACY_SHAPE_CALLS: AtomicUsize = AtomicUsize::new(0);
    /// Serialises the tests that assert *exact* counts on the two process-wide
    /// counters above. libtest runs this binary's tests in parallel, so without
    /// this one test's reset lands inside another's assertion window and the
    /// exact-count asserts fail for a reason unrelated to their claim.
    static SHAPE_COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Take the counter lock and zero both counters, so a test starts from a
    /// known state no matter what ran before it. Poisoning is irrelevant here:
    /// the guarded state is two atomics this function resets anyway.
    fn shape_counters_reset() -> std::sync::MutexGuard<'static, ()> {
        let guard = SHAPE_COUNTER_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        SHAPE_REF_CALLS.store(0, Ordering::Relaxed);
        LEGACY_SHAPE_CALLS.store(0, Ordering::Relaxed);
        guard
    }
    /// The shape the fake ORT reports for its single input.
    static FAKE_INPUT_DIMS: [i64; 3] = [2, 3, 4];
    /// Storage the fake `GetTensorData` hands back (2 * 3 * 4 f32).
    static FAKE_INPUT_DATA: [f32; 24] = [0.0; 24];
    /// `ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT`.
    const ORT_ELEM_FLOAT: ort::ONNXTensorElementDataType = 1;

    /// Stands in for ORT: exactly one input.
    ///
    /// # Safety
    ///
    /// Matches the ABI ORT expects; `out` must be a valid pointer.
    unsafe extern "C" fn fake_get_input_count(
        _ctx: *const ort::OrtKernelContext,
        out: *mut usize,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = 1 };
        std::ptr::null_mut()
    }

    /// Stands in for ORT: yields an opaque non-null `OrtValue`.
    ///
    /// # Safety
    ///
    /// Matches the ABI ORT expects; `out` must be a valid pointer.
    unsafe extern "C" fn fake_get_input(
        _ctx: *const ort::OrtKernelContext,
        _index: usize,
        out: *mut *const ort::OrtValue,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = std::ptr::dangling::<ort::OrtValue>() };
        std::ptr::null_mut()
    }

    /// Stands in for ORT: points at `FAKE_INPUT_DATA`.
    ///
    /// # Safety
    ///
    /// Matches the ABI ORT expects; `out` must be a valid pointer.
    unsafe extern "C" fn fake_get_tensor_data(
        _value: *const ort::OrtValue,
        out: *mut *const c_void,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = FAKE_INPUT_DATA.as_ptr().cast() };
        std::ptr::null_mut()
    }

    unsafe extern "C" fn fake_get_tensor_memory_info(
        _value: *const ort::OrtValue,
        out: *mut *const ort::OrtMemoryInfo,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = std::ptr::dangling::<ort::OrtMemoryInfo>() };
        std::ptr::null_mut()
    }

    unsafe extern "C" fn fake_memory_device_type(
        _memory_info: *const ort::OrtMemoryInfo,
        out: *mut ort::OrtMemoryInfoDeviceType,
    ) {
        unsafe { *out = ort::OrtMemoryInfoDeviceType_CPU };
    }

    unsafe extern "C" fn fake_gpu_memory_device_type(
        _memory_info: *const ort::OrtMemoryInfo,
        out: *mut ort::OrtMemoryInfoDeviceType,
    ) {
        unsafe { *out = ort::OrtMemoryInfoDeviceType_GPU };
    }

    unsafe extern "C" fn fake_memory_name(
        _memory_info: *const ort::OrtMemoryInfo,
        out: *mut *const std::ffi::c_char,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = c"Cpu".as_ptr() };
        std::ptr::null_mut()
    }

    unsafe extern "C" fn fake_cuda_memory_name(
        _memory_info: *const ort::OrtMemoryInfo,
        out: *mut *const std::ffi::c_char,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = c"Cuda".as_ptr() };
        std::ptr::null_mut()
    }

    unsafe extern "C" fn fake_memory_id(
        _memory_info: *const ort::OrtMemoryInfo,
        out: *mut i32,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = 0 };
        std::ptr::null_mut()
    }

    unsafe extern "C" fn fake_cuda_memory_id(
        _memory_info: *const ort::OrtMemoryInfo,
        out: *mut i32,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = 3 };
        std::ptr::null_mut()
    }

    /// Stands in for ORT's API-24 one-call type+shape reference, counting uses
    /// and lending the value's *own* dims array exactly as ORT documents.
    ///
    /// # Safety
    ///
    /// Matches the ABI ORT expects; all out pointers must be valid.
    unsafe extern "C" fn counting_shape_reference(
        _value: *const ort::OrtValue,
        elem_type: *mut ort::ONNXTensorElementDataType,
        shape_data: *mut *const i64,
        shape_data_count: *mut usize,
    ) -> ort::OrtStatusPtr {
        SHAPE_REF_CALLS.fetch_add(1, Ordering::Relaxed);
        unsafe {
            *elem_type = ORT_ELEM_FLOAT;
            *shape_data = FAKE_INPUT_DIMS.as_ptr();
            *shape_data_count = FAKE_INPUT_DIMS.len();
        }
        std::ptr::null_mut()
    }

    /// Same hook, reporting a scalar the way ORT documents it: a **null**
    /// pointer with count 0, which must never reach `from_raw_parts`.
    ///
    /// # Safety
    ///
    /// Matches the ABI ORT expects; all out pointers must be valid.
    unsafe extern "C" fn scalar_shape_reference(
        _value: *const ort::OrtValue,
        elem_type: *mut ort::ONNXTensorElementDataType,
        shape_data: *mut *const i64,
        shape_data_count: *mut usize,
    ) -> ort::OrtStatusPtr {
        SHAPE_REF_CALLS.fetch_add(1, Ordering::Relaxed);
        unsafe {
            *elem_type = ORT_ELEM_FLOAT;
            *shape_data = std::ptr::null();
            *shape_data_count = 0;
        }
        std::ptr::null_mut()
    }

    /// Stands in for the legacy allocating `GetTensorTypeAndShape`, counting
    /// uses so a test can prove the five-call sequence was *not* taken.
    ///
    /// # Safety
    ///
    /// Matches the ABI ORT expects; `out` must be a valid pointer.
    unsafe extern "C" fn counting_get_type_shape(
        _value: *const ort::OrtValue,
        out: *mut *mut ort::OrtTensorTypeAndShapeInfo,
    ) -> ort::OrtStatusPtr {
        LEGACY_SHAPE_CALLS.fetch_add(1, Ordering::Relaxed);
        unsafe { *out = std::ptr::dangling_mut::<ort::OrtTensorTypeAndShapeInfo>() };
        std::ptr::null_mut()
    }

    /// # Safety
    ///
    /// Matches the ABI ORT expects; `out` must be a valid pointer.
    unsafe extern "C" fn legacy_get_elem_type(
        _info: *const ort::OrtTensorTypeAndShapeInfo,
        out: *mut ort::ONNXTensorElementDataType,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = ORT_ELEM_FLOAT };
        std::ptr::null_mut()
    }

    /// # Safety
    ///
    /// Matches the ABI ORT expects; `out` must be a valid pointer.
    unsafe extern "C" fn legacy_get_dims_count(
        _info: *const ort::OrtTensorTypeAndShapeInfo,
        out: *mut usize,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = FAKE_INPUT_DIMS.len() };
        std::ptr::null_mut()
    }

    /// # Safety
    ///
    /// Matches the ABI ORT expects; `out` must point at `count` writable i64.
    unsafe extern "C" fn legacy_get_dims(
        _info: *const ort::OrtTensorTypeAndShapeInfo,
        out: *mut i64,
        count: usize,
    ) -> ort::OrtStatusPtr {
        let n = count.min(FAKE_INPUT_DIMS.len());
        unsafe { std::ptr::copy_nonoverlapping(FAKE_INPUT_DIMS.as_ptr(), out, n) };
        std::ptr::null_mut()
    }

    /// # Safety
    ///
    /// Matches the ABI ORT expects.
    unsafe extern "C" fn legacy_release_type_shape(_info: *mut ort::OrtTensorTypeAndShapeInfo) {}

    /// An `OrtApi` with both shape routes wired and both counted.
    fn api_with_both_shape_routes() -> ort::OrtApi {
        let mut api: ort::OrtApi = unsafe { std::mem::zeroed() };
        api.KernelContext_GetInputCount = Some(fake_get_input_count);
        api.KernelContext_GetInput = Some(fake_get_input);
        api.GetTensorData = Some(fake_get_tensor_data);
        api.GetTensorMemoryInfo = Some(fake_get_tensor_memory_info);
        api.MemoryInfoGetDeviceType = Some(fake_memory_device_type);
        api.MemoryInfoGetName = Some(fake_memory_name);
        api.MemoryInfoGetId = Some(fake_memory_id);
        api.GetTensorElementTypeAndShapeDataReference = Some(counting_shape_reference);
        api.GetTensorTypeAndShape = Some(counting_get_type_shape);
        api.GetTensorElementType = Some(legacy_get_elem_type);
        api.GetDimensionsCount = Some(legacy_get_dims_count);
        api.GetDimensions = Some(legacy_get_dims);
        api.ReleaseTensorTypeAndShapeInfo = Some(legacy_release_type_shape);
        api
    }

    /// The whole point of the change: when ORT offers the API-24 hook, the
    /// per-input, per-`Run` shape read is **one** call and the allocating
    /// five-call sequence is never entered.
    ///
    /// Falsifier: point `read_inputs` back at the legacy family unconditionally
    /// and this fails with `SHAPE_REF_CALLS 0` / `LEGACY_SHAPE_CALLS 1`
    /// (verified).
    #[test]
    fn input_shapes_come_from_one_call_when_ort_offers_the_reference_hook() {
        let api = api_with_both_shape_routes();
        let ctx = std::ptr::dangling_mut::<ort::OrtKernelContext>();

        let _counters = shape_counters_reset();
        let inputs = unsafe { read_inputs(&api, ctx) }.expect("the fake ORT reads successfully");

        assert_eq!(
            SHAPE_REF_CALLS.load(Ordering::Relaxed),
            1,
            "one input must cost exactly one shape call"
        );
        assert_eq!(
            LEGACY_SHAPE_CALLS.load(Ordering::Relaxed),
            0,
            "the allocating five-call sequence must not run when the reference hook exists"
        );
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].shape, vec![2, 3, 4]);
        assert_eq!(inputs[0].strides, vec![12, 4, 1]);
        assert_eq!(inputs[0].dtype, DataType::Float32);
        assert_eq!(inputs[0].device, DeviceId::cpu());
    }

    #[test]
    fn input_view_preserves_cuda_residency_from_ort_memory_info() {
        let mut api = api_with_both_shape_routes();
        api.MemoryInfoGetDeviceType = Some(fake_gpu_memory_device_type);
        api.MemoryInfoGetName = Some(fake_cuda_memory_name);
        api.MemoryInfoGetId = Some(fake_cuda_memory_id);
        let ctx = std::ptr::dangling_mut::<ort::OrtKernelContext>();

        let inputs = unsafe { read_inputs(&api, ctx) }.expect("the fake CUDA input is readable");
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].device, DeviceId::cuda(3));
        assert_eq!(inputs[0].view().device, DeviceId::cuda(3));
        assert!(
            !inputs[0].view().device.is_host_accessible(),
            "a non-null CUDA pointer must never be inferred to be host memory"
        );
    }

    #[test]
    fn raw_memory_device_type_conversion_covers_linux_and_windows_bindings() {
        assert_eq!(raw_device_type_code(7u32, "linux typedef").unwrap(), 7);
        assert_eq!(raw_device_type_code(7i32, "Windows typedef").unwrap(), 7);
        let error = raw_device_type_code(-1i32, "Windows typedef").unwrap_err();
        assert!(error.contains("invalid raw device type -1"), "{error}");
    }

    /// The route test above proves *which* family runs; these pin what it
    /// costs. Gated with the probe, like the other cost pins in this file.
    #[cfg(feature = "dispatch_probe")]
    mod reference_hook_cost {

        /// A rank past [`crate::dim_vec::INLINE_RANK`], where a `DimVec` must
        /// spill. Every element is 1, so the tensor is still a single f32 and the
        /// data fake below stays valid.
        static FAKE_HIGH_RANK_DIMS: [i64; crate::dim_vec::INLINE_RANK + 2] =
            [1; crate::dim_vec::INLINE_RANK + 2];

        /// # Safety
        ///
        /// Matches the ABI ORT expects; all out pointers must be valid.
        unsafe extern "C" fn high_rank_shape_reference(
            _value: *const ort::OrtValue,
            elem_type: *mut ort::ONNXTensorElementDataType,
            shape_data: *mut *const i64,
            shape_data_count: *mut usize,
        ) -> ort::OrtStatusPtr {
            unsafe {
                *elem_type = super::ORT_ELEM_FLOAT;
                *shape_data = FAKE_HIGH_RANK_DIMS.as_ptr();
                *shape_data_count = FAKE_HIGH_RANK_DIMS.len();
            }
            std::ptr::null_mut()
        }

        /// # Safety
        ///
        /// Matches the ABI ORT expects; `out` must be a valid pointer.
        unsafe extern "C" fn high_rank_dims_count(
            _info: *const ort::OrtTensorTypeAndShapeInfo,
            out: *mut usize,
        ) -> ort::OrtStatusPtr {
            unsafe { *out = FAKE_HIGH_RANK_DIMS.len() };
            std::ptr::null_mut()
        }

        /// # Safety
        ///
        /// Matches the ABI ORT expects; `out` must point at `count` writable i64.
        unsafe extern "C" fn high_rank_dims(
            _info: *const ort::OrtTensorTypeAndShapeInfo,
            out: *mut i64,
            count: usize,
        ) -> ort::OrtStatusPtr {
            let n = count.min(FAKE_HIGH_RANK_DIMS.len());
            unsafe { std::ptr::copy_nonoverlapping(FAKE_HIGH_RANK_DIMS.as_ptr(), out, n) };
            std::ptr::null_mut()
        }

        /// An `OrtApi` with both shape routes wired, reporting a rank that spills.
        fn api_with_both_shape_routes_high_rank() -> ort::OrtApi {
            let mut api = super::api_with_both_shape_routes();
            api.GetTensorElementTypeAndShapeDataReference = Some(high_rank_shape_reference);
            api.GetDimensionsCount = Some(high_rank_dims_count);
            api.GetDimensions = Some(high_rank_dims);
            api
        }
        use super::*;
        use crate::dispatch_probe::{self, Event};

        /// Without this, the central claim of #1246 -- fewer round trips into
        /// ORT -- is untested, and a later refactor could restore the five
        /// calls one at a time while `input_shapes_come_from_one_call...`
        /// still passed (it only asserts the legacy *shape* entry point is
        /// untouched).
        ///
        /// Three per input: `KernelContext_GetInput`,
        /// `GetTensorElementTypeAndShapeDataReference`, `GetTensorData`, plus
        /// the one shared `GetInputCount`. The legacy path pinned elsewhere in
        /// this file costs `1 + 7`.
        #[test]
        fn the_reference_hook_path_costs_exactly_three_ort_calls_per_input() {
            let api = api_with_both_shape_routes();
            let ctx = std::ptr::dangling_mut::<ort::OrtKernelContext>();

            let _counters = shape_counters_reset();
            let before = dispatch_probe::snapshot();
            let inputs =
                unsafe { read_inputs(&api, ctx) }.expect("the fake ORT reads successfully");
            let d = dispatch_probe::snapshot().since(&before);

            assert_eq!(inputs.len(), 1);
            assert_eq!(
                d.event(Event::OrtFfiCall),
                1 + 3,
                "per-input FFI round trips on the reference-hook path changed; if \
                 this is an intentional improvement, lower the number -- if it went \
                 up, the batching this PR exists for has regressed"
            );
        }

        /// The five-call sequence also allocated: `GetDimensionsCount` then a
        /// `dims` scratch sized to the rank, before the shape and strides were
        /// built from it. Borrowing the dims leaves the scratch with nothing to
        /// do, so the reference-hook path must allocate strictly less than the
        /// legacy one **for an input whose rank does not fit inline**.
        ///
        /// At ordinary rank neither path allocates at all any more, so the
        /// comparison is only meaningful once a `DimVec` is forced to spill —
        /// which is exactly why this test drives a rank of
        /// `INLINE_RANK + 2`. Pinned as a comparison rather than an absolute
        /// so it keeps its meaning as both paths evolve.
        #[test]
        fn the_reference_hook_path_allocates_less_than_the_legacy_path() {
            let ctx = std::ptr::dangling_mut::<ort::OrtKernelContext>();

            // One guard for both measurements: `shape_counters_reset` returns a
            // `MutexGuard` and the lock is not reentrant, so taking it twice in
            // one scope deadlocks (shadowing does not drop the first).
            let _counters = shape_counters_reset();

            let api = api_with_both_shape_routes_high_rank();
            let before = dispatch_probe::snapshot();
            let fast = unsafe { read_inputs(&api, ctx) }.expect("the fake ORT reads successfully");
            let fast_allocs = dispatch_probe::snapshot()
                .since(&before)
                .event(Event::DispatchAlloc);

            let mut api = api_with_both_shape_routes_high_rank();
            api.GetTensorElementTypeAndShapeDataReference = None;
            let before = dispatch_probe::snapshot();
            let slow = unsafe { read_inputs(&api, ctx) }.expect("the fake ORT reads successfully");
            let slow_allocs = dispatch_probe::snapshot()
                .since(&before)
                .event(Event::DispatchAlloc);

            assert_eq!(fast[0].shape, slow[0].shape, "same input either way");
            assert_eq!(
                fast[0].shape.len(),
                crate::dim_vec::INLINE_RANK + 2,
                "this test is only meaningful at a rank that spills"
            );
            assert!(
                fast_allocs < slow_allocs,
                "the reference-hook path allocated {fast_allocs}, the legacy path \
                 {slow_allocs}; borrowing the dims must skip the rank-sized scratch"
            );
        }

        /// The other half of the claim above: at an ordinary rank the input
        /// path must not touch the allocator on *either* route. This is the
        /// assertion that would fail if a `Vec` crept back into the per-operand
        /// path, and the one the depth grid actually feels.
        #[test]
        fn neither_shape_route_allocates_at_an_ordinary_rank() {
            let ctx = std::ptr::dangling_mut::<ort::OrtKernelContext>();
            let _counters = shape_counters_reset();

            for (label, mut api) in [
                ("reference hook", api_with_both_shape_routes()),
                ("five-call legacy", api_with_both_shape_routes()),
            ] {
                if label == "five-call legacy" {
                    api.GetTensorElementTypeAndShapeDataReference = None;
                }
                let before = dispatch_probe::snapshot();
                let inputs = unsafe { read_inputs(&api, ctx) }.expect("the fake ORT reads");
                let allocs = dispatch_probe::snapshot()
                    .since(&before)
                    .event(Event::DispatchAlloc);
                assert!(
                    inputs[0].shape.len() <= crate::dim_vec::INLINE_RANK,
                    "{label}: the fake must report an ordinary rank"
                );
                assert_eq!(
                    allocs, 1,
                    "{label}: only the shared `Vec<OwnedInput>` may allocate"
                );
            }
        }
    }

    /// The fallback is not decoration: with the reference hook absent, the
    /// five-call sequence must still produce the identical `OwnedInput`.
    #[test]
    fn the_five_call_fallback_produces_the_same_input_as_the_reference_hook() {
        let reference = {
            let api = api_with_both_shape_routes();
            let ctx = std::ptr::dangling_mut::<ort::OrtKernelContext>();
            let _counters = shape_counters_reset();
            unsafe { read_inputs(&api, ctx) }.expect("the fake ORT reads successfully")
        };

        let mut api = api_with_both_shape_routes();
        api.GetTensorElementTypeAndShapeDataReference = None;
        let ctx = std::ptr::dangling_mut::<ort::OrtKernelContext>();

        let _counters = shape_counters_reset();
        let fallback = unsafe { read_inputs(&api, ctx) }.expect("the fake ORT reads successfully");

        assert_eq!(
            LEGACY_SHAPE_CALLS.load(Ordering::Relaxed),
            1,
            "a host without the reference hook must still get its shapes"
        );
        assert_eq!(fallback.len(), reference.len());
        assert_eq!(fallback[0].shape, reference[0].shape);
        assert_eq!(fallback[0].strides, reference[0].strides);
        assert_eq!(fallback[0].dtype, reference[0].dtype);
        assert_eq!(fallback[0].data_ptr, reference[0].data_ptr);
    }

    /// ORT reports a scalar as a **null** shape pointer with count 0.
    /// `slice::from_raw_parts` is UB on null, so the borrowed path has to
    /// special-case it. Natively this asserts the resulting rank-0 shape;
    /// the UB itself is caught by the Miri lane, which runs exactly this
    /// module (`.github/workflows/miri.yml`, "onnx-runtime-ep-plugin kernel
    /// context") — without that lane the null guard would be untested.
    #[test]
    fn a_borrowed_scalar_shape_never_dereferences_null() {
        let mut api = api_with_both_shape_routes();
        api.GetTensorElementTypeAndShapeDataReference = Some(scalar_shape_reference);
        let ctx = std::ptr::dangling_mut::<ort::OrtKernelContext>();

        let _counters = shape_counters_reset();
        let inputs = unsafe { read_inputs(&api, ctx) }.expect("a scalar input is legal");

        assert_eq!(
            SHAPE_REF_CALLS.load(Ordering::Relaxed),
            1,
            "the scalar must have gone through the borrowed path, not the fallback"
        );
        assert_eq!(inputs[0].shape, Vec::<usize>::new());
        assert_eq!(inputs[0].strides, Vec::<i64>::new());
    }

    /// A host offering neither route must fail closed with a message naming
    /// both, not silently read garbage shapes.
    #[test]
    fn read_inputs_fails_closed_when_no_shape_route_exists() {
        let mut api = api_with_both_shape_routes();
        api.GetTensorElementTypeAndShapeDataReference = None;
        api.GetDimensions = None;
        let ctx = std::ptr::dangling_mut::<ort::OrtKernelContext>();

        let err = match unsafe { read_inputs(&api, ctx) } {
            Ok(_) => panic!("no shape route must be fatal, not silently successful"),
            Err(e) => e,
        };
        assert!(
            err.contains("GetTensorElementTypeAndShapeDataReference")
                && err.contains("GetTensorTypeAndShape"),
            "error must name both routes: {err}"
        );
    }

    /// The output shape handed to ORT must be the same values whether it went
    /// through the inline array or the `Vec`. The boundary is rank 8.
    ///
    /// Falsifier: drop the `heap_dims` arm and a rank-9 output silently sends
    /// ORT a truncated shape; this fails on `dims_seen` (verified).
    #[test]
    fn output_dims_are_identical_on_the_inline_and_heap_ranks() {
        let mut api: ort::OrtApi = unsafe { std::mem::zeroed() };
        api.KernelContext_GetOutput = Some(recording_get_output);
        api.GetTensorMutableData = Some(fake_get_mutable_data);
        let ctx = std::ptr::dangling_mut::<ort::OrtKernelContext>();

        for rank in [0usize, 1, 8, 9, 12] {
            let shape: Vec<usize> = (1..=rank).collect();
            let output = unsafe { allocate_output(&api, ctx, 0, &shape, DataType::Float32, false) }
                .expect("the fake ORT allocates successfully");
            let seen = RECORDED_DIMS.lock().expect("recorded dims lock");
            let expected: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
            assert_eq!(
                *seen, expected,
                "rank {rank} must reach ORT with every dimension intact"
            );

            // Pins the *output path*, not just the arithmetic: if
            // `allocate_output` ever stopped routing through
            // `contiguous_strides`, the arithmetic pin below would still pass
            // and only this would fail. Ranks 9 and 12 spill, so this covers
            // both representations.
            assert_eq!(
                &output.shape[..],
                &shape[..],
                "rank {rank} lost dimensions on the way into OwnedOutput"
            );
            assert_eq!(
                &output.strides[..],
                &onnx_runtime_ir::compute_contiguous_strides(&shape)[..],
                "rank {rank} strides diverged from the IR crate"
            );
        }
    }

    #[test]
    fn owned_input_view_roundtrip() {
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let input = OwnedInput {
            data_ptr: data.as_ptr().cast(),
            dtype: DataType::Float32,
            shape: DimVec::from([2, 2]),
            strides: DimVec::from([2, 1]),
            device: DeviceId::cpu(),
        };
        let view = input.view();
        assert_eq!(view.shape, &[2, 2]);
        assert_eq!(view.dtype, DataType::Float32);
        assert_eq!(view.device, DeviceId::cpu());
    }

    #[test]
    fn owned_input_dangling_device_pointer_stays_non_host_accessible() {
        let input = OwnedInput {
            data_ptr: std::ptr::dangling(),
            dtype: DataType::Int64,
            shape: DimVec::new(),
            strides: DimVec::new(),
            device: DeviceId::cuda(7),
        };
        let view = input.view();
        assert_eq!(view.device, DeviceId::cuda(7));
        assert!(!view.device.is_host_accessible());
    }

    #[test]
    fn owned_output_view_mut_roundtrip() {
        let mut data: [f32; 6] = [0.0; 6];
        let mut output = OwnedOutput {
            data_ptr: data.as_mut_ptr().cast(),
            dtype: DataType::Float32,
            shape: DimVec::from_slice(&[2, 3]),
            strides: DimVec::from_slice(&[3, 1]),
            mem_info: std::ptr::null(),
        };
        let view = output.view_mut();
        assert_eq!(view.shape, &[2, 3]);
        assert_eq!(view.dtype, DataType::Float32);
    }

    /// Rank 8 is where the *fast* representation ends, not where support ends.
    /// An output past the inline limit has to survive the spill with its shape
    /// and strides intact, and the view it hands out must still describe it.
    #[test]
    fn owned_output_survives_a_rank_past_the_inline_limit() {
        // Deliberately not all ones: with a degenerate shape every stride is 1
        // and the test could not tell a correct stride build from a reversed
        // or constant one.
        let shape: Vec<usize> = vec![2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3];
        let rank = shape.len();
        assert!(rank > crate::dim_vec::INLINE_RANK);
        let strides = contiguous_strides(&shape);
        assert!(
            strides.len() == rank && shape.len() > crate::dim_vec::INLINE_RANK,
            "this test is only meaningful past the inline limit"
        );

        let mut data = vec![0.0f32; shape.iter().product::<usize>()];
        let mut output = OwnedOutput {
            data_ptr: data.as_mut_ptr().cast(),
            dtype: DataType::Float32,
            shape: DimVec::from_slice(&shape),
            strides,
            mem_info: std::ptr::null(),
        };
        let view = output.view_mut();
        assert_eq!(view.shape, &shape[..], "spilled shape did not survive");
        assert_eq!(
            view.strides,
            &onnx_runtime_ir::compute_contiguous_strides(&shape)[..],
            "spilled strides did not survive"
        );
    }

    #[test]
    fn owned_input_null_data_for_optional() {
        let input = OwnedInput {
            data_ptr: std::ptr::null(),
            dtype: DataType::Float32,
            shape: DimVec::new(),
            strides: DimVec::new(),
            device: DeviceId::cpu(),
        };
        let view = input.view();
        assert_eq!(view.shape, &[] as &[usize]);
    }

    #[test]
    fn dtype_mapping_supported_types() {
        // Verify round-trip for common types used by CPU EP.
        let cases = [
            (1, DataType::Float32),
            (7, DataType::Int64),
            (11, DataType::Float64),
            (6, DataType::Int32),
        ];
        for (onnx_val, expected) in cases {
            let dt = DataType::from_onnx(onnx_val).unwrap();
            assert_eq!(dt, expected, "from_onnx({onnx_val}) mismatch");
            assert_eq!(
                dt.to_onnx(),
                onnx_val,
                "to_onnx round-trip for {expected:?}"
            );
        }
    }

    #[test]
    fn dtype_mapping_unsupported_returns_none() {
        // 0 is UNDEFINED in ORT, should return None.
        assert!(DataType::from_onnx(0).is_none());
        // Extremely large value should also be None.
        assert!(DataType::from_onnx(9999).is_none());
    }

    // ── Dimension validation tests ────────────────────────────────────────

    #[test]
    fn validate_dims_rejects_negative() {
        let dims = [4, -1, 8];
        let err = super::validate_dims(&dims, DataType::Float32, format_args!("test")).unwrap_err();
        assert!(err.contains("dim[1] is -1"), "error: {err}");
        assert!(err.contains("negative"), "error: {err}");
    }

    #[test]
    fn validate_dims_rejects_large_negative() {
        // ORT's dynamic-dim sentinel -1 as i64
        let dims = [2, -1i64];
        let err = super::validate_dims(&dims, DataType::Float32, format_args!("x")).unwrap_err();
        assert!(err.contains("-1"), "error: {err}");
    }

    #[test]
    fn validate_dims_overflow_element_count() {
        // Two huge dims that overflow usize on multiply
        let dims = [i64::MAX / 2, 4];
        let err = super::validate_dims(&dims, DataType::Float32, format_args!("big")).unwrap_err();
        assert!(err.contains("overflows"), "error: {err}");
    }

    #[test]
    fn validate_dims_overflow_byte_length() {
        // Two large dims whose product fits in usize but * byte_size overflows
        // On 64-bit: i64::MAX/4 is fine as element count for Float32 (4 bytes)
        // but i64::MAX/4 * 4 = i64::MAX - 3 which doesn't overflow usize,
        // so use i64::MAX/2 * 2 bytes (Float16).
        // Actually: find dims that make element_count * byte_size overflow.
        // i64::MAX as usize = 2^63-1. For Float64 (8 bytes):
        // (2^63-1) * 8 overflows usize on 64-bit.
        let dims = [i64::MAX]; // huge but positive
        let result = super::validate_dims(&dims, DataType::Float64, format_args!("bytes"));
        // byte_size = 8, element_count = 2^63-1, product overflows
        assert!(result.is_err(), "should fail: {result:?}");
    }

    #[test]
    fn validate_dims_zero_dim_accepted() {
        // Zero-dim tensors are legal in ONNX (e.g. empty batch)
        let dims = [0i64, 3, 224, 224];
        let (shape, elem_count, byte_len) =
            super::validate_dims(&dims, DataType::Float32, format_args!("zero")).unwrap();
        assert_eq!(shape, vec![0, 3, 224, 224]);
        assert_eq!(elem_count, 0);
        assert_eq!(byte_len, 0);
    }

    #[test]
    fn validate_dims_scalar_tensor() {
        // Scalar: zero-rank tensor with 1 element
        let dims: [i64; 0] = [];
        let (shape, elem_count, byte_len) =
            super::validate_dims(&dims, DataType::Float32, format_args!("scalar")).unwrap();
        assert_eq!(shape, Vec::<usize>::new());
        assert_eq!(elem_count, 1); // product of empty = 1
        assert_eq!(byte_len, 4);
    }

    /// The reason `contiguous_strides` is allowed to exist alongside
    /// `onnx_runtime_ir::compute_contiguous_strides`: it must agree with it
    /// exactly, for every rank on both sides of the spill boundary and for the
    /// shapes that make stride arithmetic interesting — a zero dim, a unit
    /// dim, and a rank of 0 or 1 where the reduction loop does not run.
    ///
    /// Without this the duplication is just a second implementation waiting to
    /// drift. With it, either one can be changed and the other will object.
    /// `validate_dims` builds its shape by filling a pre-sized `DimVec` rather
    /// than pushing one element at a time. The two constructions take different
    /// paths through `DimVec` -- `zeroed` allocates `vec![0; n]` up front where
    /// `with_capacity` + `push` grew into reserved space -- so the resulting
    /// value has to be checked for content, length *and* representation across
    /// the inline/heap boundary, not just content.
    #[test]
    fn validate_dims_shape_matches_a_direct_construction() {
        let mut cases: Vec<Vec<i64>> = vec![
            vec![],
            vec![0],
            vec![1],
            vec![8],
            vec![1, 8],
            vec![2, 0, 5],
            vec![1, 1, 1, 1, 1],
        ];
        // Every rank across the spill boundary in both directions.
        for rank in 0..=(crate::dim_vec::INLINE_RANK + 3) {
            cases.push((0..rank).map(|k| (k % 3 + 1) as i64).collect());
        }
        // A spilled rank containing a zero extent.
        let mut zero_deep = vec![2i64; crate::dim_vec::INLINE_RANK + 2];
        zero_deep[crate::dim_vec::INLINE_RANK + 1] = 0;
        cases.push(zero_deep);

        for dims in cases {
            let (shape, count, bytes) =
                super::validate_dims(&dims, DataType::Float32, format_args!("t")).unwrap();

            let expected: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
            assert_eq!(shape.as_slice(), expected.as_slice(), "dims {dims:?}");
            assert_eq!(shape.len(), dims.len(), "one entry per dim, dims {dims:?}");
            assert_eq!(
                shape.is_spilled(),
                dims.len() > crate::dim_vec::INLINE_RANK,
                "dims {dims:?} took the wrong representation"
            );

            let want: usize = expected.iter().product();
            assert_eq!(count, want, "element count for {dims:?}");
            assert_eq!(bytes, want * 4, "byte length for {dims:?}");
        }
    }

    /// A negative dimension is reported even when an earlier pair of dimensions
    /// would overflow the element count.
    ///
    /// The negative check runs during the shape fill and the overflow check runs
    /// after it, so negatives win regardless of position. Pinned because fusing
    /// those two passes -- the obvious next optimization here -- would silently
    /// swap the precedence and change which error a malformed graph reports.
    #[test]
    fn a_negative_dim_outranks_an_earlier_overflow() {
        // The prefix must genuinely overflow, or this test pins nothing: a fused
        // single-pass version would agree with the two-pass version simply because
        // no overflow was ever reachable. The control below proves the overflow is
        // live, so the assertion above it is about precedence and not about luck.
        let overflowing_prefix = [i64::MAX / 2, 8];
        let control = super::validate_dims(
            &overflowing_prefix,
            DataType::Float32,
            format_args!("control"),
        )
        .unwrap_err();
        assert!(
            control.contains("overflow"),
            "the prefix must actually overflow for this test to mean anything, got: {control}"
        );

        let dims = [i64::MAX / 2, 8, -1];
        let err = super::validate_dims(&dims, DataType::Float32, format_args!("t")).unwrap_err();
        assert!(
            err.contains("dim[2] is -1"),
            "expected the negative dim to be reported, got: {err}"
        );
        assert!(
            !err.contains("overflow"),
            "the overflow must not pre-empt the negative dim: {err}"
        );
    }

    #[test]
    fn contiguous_strides_matches_the_ir_crate() {
        let mut cases: Vec<Vec<usize>> = vec![
            vec![],
            vec![7],
            vec![1],
            vec![0],
            vec![2, 3, 4],
            vec![3, 0, 5],
            vec![1, 1, 1, 1],
            vec![5, 1, 7, 1, 9],
        ];
        // Every rank across the inline/heap boundary, including the exact
        // rank where `push` spills.
        for rank in 0..=(crate::dim_vec::INLINE_RANK + 3) {
            cases.push((0..rank).map(|k| k % 4 + 1).collect());
        }
        // A spilled rank that also contains a zero, so the spill path is
        // exercised with a degenerate shape rather than only a tidy one.
        let mut zero_at_depth = vec![2usize; crate::dim_vec::INLINE_RANK + 2];
        zero_at_depth[crate::dim_vec::INLINE_RANK] = 0;
        cases.push(zero_at_depth);

        for shape in cases {
            let ours = super::contiguous_strides(&shape);
            let theirs = onnx_runtime_ir::compute_contiguous_strides(&shape);
            assert_eq!(
                ours.as_slice(),
                theirs.as_slice(),
                "strides diverged for shape {shape:?}"
            );
            assert_eq!(
                ours.len(),
                shape.len(),
                "one stride per dimension, for shape {shape:?}"
            );
            assert_eq!(
                ours.is_spilled(),
                shape.len() > crate::dim_vec::INLINE_RANK,
                "shape {shape:?} took the wrong representation"
            );
        }
    }

    #[test]
    fn validate_dims_normal_shape() {
        let dims = [2i64, 3, 4];
        let (shape, elem_count, byte_len) =
            super::validate_dims(&dims, DataType::Float32, format_args!("ok")).unwrap();
        assert_eq!(shape, vec![2, 3, 4]);
        assert_eq!(elem_count, 24);
        assert_eq!(byte_len, 96);
    }

    // ── f16/bf16 dtype mapping and byte-width tests ───────────────────────────

    #[test]
    fn f16_dtype_round_trip() {
        // ORT ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10 (verified in bindings.rs).
        let dt = DataType::from_onnx(10).expect("Float16 must map from ONNX value 10");
        assert_eq!(dt, DataType::Float16);
        assert_eq!(dt.to_onnx(), 10);
        assert_eq!(dt.byte_size(), 2, "Float16 must be 2 bytes/element");
    }

    #[test]
    fn bf16_dtype_round_trip() {
        // ORT ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16 (verified in bindings.rs).
        let dt = DataType::from_onnx(16).expect("BFloat16 must map from ONNX value 16");
        assert_eq!(dt, DataType::BFloat16);
        assert_eq!(dt.to_onnx(), 16);
        assert_eq!(dt.byte_size(), 2, "BFloat16 must be 2 bytes/element");
    }

    #[test]
    fn f16_byte_length_computation() {
        // A [4, 8] f16 tensor: 32 elements × 2 bytes = 64 bytes.
        let dims = [4i64, 8];
        let (shape, elem_count, byte_len) =
            super::validate_dims(&dims, DataType::Float16, format_args!("f16")).unwrap();
        assert_eq!(shape, vec![4, 8]);
        assert_eq!(elem_count, 32);
        assert_eq!(byte_len, 64);
    }

    #[test]
    fn bf16_byte_length_computation() {
        // A [3, 5] bf16 tensor: 15 elements × 2 bytes = 30 bytes.
        let dims = [3i64, 5];
        let (shape, elem_count, byte_len) =
            super::validate_dims(&dims, DataType::BFloat16, format_args!("bf16")).unwrap();
        assert_eq!(shape, vec![3, 5]);
        assert_eq!(elem_count, 15);
        assert_eq!(byte_len, 30);
    }

    #[test]
    fn f16_byte_length_overflow_guard() {
        // Verify byte-length overflow is caught for Float16 (2 bytes/element).
        // element_count = (i64::MAX/2 + 1) * 2 = 2^63 fits in usize.
        // byte_len = 2^63 * 2 = 2^64 overflows usize — checked_mul must reject it.
        let half_max_plus_1 = i64::MAX / 2 + 1; // 4611686018427387904
        let dims = [half_max_plus_1, 2i64];
        let err = super::validate_dims(&dims, DataType::Float16, format_args!("f16_overflow"))
            .unwrap_err();
        assert!(
            err.contains("overflows"),
            "expected overflow error, got: {err}"
        );
    }

    #[test]
    fn unsupported_dtype_fails_closed() {
        // ORT element type 0 is UNDEFINED — from_onnx returns None.
        let result = DataType::from_onnx(0);
        assert!(
            result.is_none(),
            "UNDEFINED dtype must fail closed (None), not silently coerce"
        );
        // Confirm the read_inputs error path names the type value.
        // We test the from_onnx → error message contract via validate_dims
        // since read_inputs requires a live ORT context.
        // The error produced in read_inputs is:
        //   format!("unsupported element type {elem_type} for input {i}")
        // which names the numeric value — verified by this assertion:
        let msg = format!("unsupported element type {} for input {}", 0u32, 0usize);
        assert!(msg.contains('0'));
    }

    #[test]
    fn cpu_ep_supported_dtypes_contains_f16_and_bf16() {
        assert!(
            super::CPU_EP_SUPPORTED_DTYPES.contains(&DataType::Float16),
            "CPU_EP_SUPPORTED_DTYPES must include Float16"
        );
        assert!(
            super::CPU_EP_SUPPORTED_DTYPES.contains(&DataType::BFloat16),
            "CPU_EP_SUPPORTED_DTYPES must include BFloat16"
        );
        // Verify every dtype in the list has a non-zero byte_size or is sub-byte.
        for &dt in super::CPU_EP_SUPPORTED_DTYPES {
            assert!(
                dt.byte_size() > 0 || dt.bit_size() > 0,
                "dtype {dt:?} has no representable size"
            );
        }
    }
}

/// A hand-built `OrtApi` that answers just enough of the C API for
/// [`read_inputs`] to run, so the FFI-call and allocation counts of a dispatch
/// can be asserted exactly without a live ONNX Runtime.
///
/// This is the regression guard the probe exists for: the numbers below are not
/// aspirational, they are what the code does today. Any change that adds a
/// round trip or an allocation to the per-`Run` path has to come here and
/// change a number, in a diff a reviewer can see.
#[cfg(all(test, feature = "dispatch_probe"))]
mod dispatch_cost {
    use super::*;
    use crate::dispatch_probe::{self, Event};

    const NDIM: usize = 2;
    static DIMS: [i64; NDIM] = [2, 3];
    static DATA: [f32; 6] = [0.0; 6];

    // A non-null token handed back as the "type and shape info" handle. Never
    // dereferenced — the fake accessors read from the statics above.
    const INFO: *mut ort::OrtTensorTypeAndShapeInfo = std::ptr::without_provenance_mut(1);
    const VALUE: *const ort::OrtValue = std::ptr::without_provenance(2);

    unsafe extern "C" fn input_count(
        _c: *const ort::OrtKernelContext,
        out: *mut usize,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = 1 };
        std::ptr::null_mut()
    }
    unsafe extern "C" fn get_input(
        _c: *const ort::OrtKernelContext,
        _i: usize,
        out: *mut *const ort::OrtValue,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = VALUE };
        std::ptr::null_mut()
    }
    unsafe extern "C" fn type_and_shape(
        _v: *const ort::OrtValue,
        out: *mut *mut ort::OrtTensorTypeAndShapeInfo,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = INFO };
        std::ptr::null_mut()
    }
    unsafe extern "C" fn elem_type(
        _i: *const ort::OrtTensorTypeAndShapeInfo,
        out: *mut ort::ONNXTensorElementDataType,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = 1 }; // float32
        std::ptr::null_mut()
    }
    unsafe extern "C" fn dims_count(
        _i: *const ort::OrtTensorTypeAndShapeInfo,
        out: *mut usize,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = NDIM };
        std::ptr::null_mut()
    }
    unsafe extern "C" fn dims(
        _i: *const ort::OrtTensorTypeAndShapeInfo,
        out: *mut i64,
        len: usize,
    ) -> ort::OrtStatusPtr {
        assert_eq!(len, NDIM, "read_inputs must size the dims buffer itself");
        unsafe { std::ptr::copy_nonoverlapping(DIMS.as_ptr(), out, NDIM) };
        std::ptr::null_mut()
    }
    unsafe extern "C" fn release(_i: *mut ort::OrtTensorTypeAndShapeInfo) {}
    unsafe extern "C" fn tensor_data(
        _v: *const ort::OrtValue,
        out: *mut *const c_void,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = DATA.as_ptr().cast() };
        std::ptr::null_mut()
    }

    /// `OrtApi` is a large struct of nullable function pointers; zeroing it
    /// yields `None` in every slot (guaranteed by the null-pointer optimization
    /// for `Option<extern "C" fn>`), which is exactly "this host offers no
    /// entry points". We then fill in the handful `read_inputs` uses, so any
    /// call it makes that this test did not anticipate faults loudly instead of
    /// silently succeeding.
    fn fake_api() -> ort::OrtApi {
        let mut api: ort::OrtApi = unsafe { std::mem::zeroed() };
        api.KernelContext_GetInputCount = Some(input_count);
        api.KernelContext_GetInput = Some(get_input);
        api.GetTensorTypeAndShape = Some(type_and_shape);
        api.GetTensorElementType = Some(elem_type);
        api.GetDimensionsCount = Some(dims_count);
        api.GetDimensions = Some(dims);
        api.ReleaseTensorTypeAndShapeInfo = Some(release);
        api.GetTensorData = Some(tensor_data);
        api
    }

    /// Reading one present input costs eight round trips into ORT: one shared
    /// `GetInputCount`, then seven per input.
    ///
    /// Batching these is the point of #1246 and its successors; the count is
    /// pinned here so a regression cannot quietly restore them one at a time.
    #[test]
    fn reading_one_input_costs_exactly_eight_ort_calls() {
        let api = fake_api();
        let before = dispatch_probe::snapshot();
        let inputs = unsafe { read_inputs(&api, std::ptr::null_mut()) }.expect("fake api is total");
        let d = dispatch_probe::snapshot().since(&before);

        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].shape, vec![2, 3]);
        assert_eq!(
            d.event(Event::OrtFfiCall),
            1 + 7,
            "per-input FFI round trips changed; if this is an intentional \
             improvement, lower the number — if it went up, justify it"
        );
    }

    /// One input of ordinary rank costs **one** heap allocation: the
    /// `Vec<OwnedInput>`, once per `Run`. Nothing per input at all.
    ///
    /// It was five, then four when the error label stopped being built
    /// eagerly, and now one: `dims`, `shape` and `strides` are [`DimVec`]s
    /// that keep an ordinary rank in the value itself. That is the counter
    /// earning its keep — the improvement is a number changing in a test, not
    /// a claim in a commit message.
    #[test]
    fn reading_one_input_of_ordinary_rank_costs_exactly_one_allocation() {
        let api = fake_api();
        let before = dispatch_probe::snapshot();
        let _ = unsafe { read_inputs(&api, std::ptr::null_mut()) }.expect("fake api is total");
        let d = dispatch_probe::snapshot().since(&before);
        assert_eq!(
            d.event(Event::DispatchAlloc),
            1,
            "allocations on the per-Run input path changed"
        );
    }

    /// The FFI cost must be *per input*, not amortised — three inputs cost
    /// three times the per-input round trips plus the single shared `Vec`.
    /// This is what makes the counters usable as a model rather than a single
    /// data point.
    ///
    /// Allocations no longer scale with the input count at ordinary rank,
    /// which is the whole change: the assertion below is the one that would
    /// catch a regression back to a `Vec` per operand.
    #[test]
    fn per_input_cost_scales_linearly() {
        unsafe extern "C" fn three(
            _c: *const ort::OrtKernelContext,
            out: *mut usize,
        ) -> ort::OrtStatusPtr {
            unsafe { *out = 3 };
            std::ptr::null_mut()
        }
        let mut api = fake_api();
        api.KernelContext_GetInputCount = Some(three);

        let before = dispatch_probe::snapshot();
        let inputs = unsafe { read_inputs(&api, std::ptr::null_mut()) }.expect("fake api is total");
        let d = dispatch_probe::snapshot().since(&before);

        assert_eq!(inputs.len(), 3);
        assert_eq!(d.event(Event::OrtFfiCall), 1 + 3 * 7);
        assert_eq!(
            d.event(Event::DispatchAlloc),
            1,
            "the per-input allocations are gone; only the shared `Vec` is left, \
             and it must not start scaling with the input count"
        );
    }

    /// An absent optional input short-circuits: ORT hands back a null value and
    /// we must not query it further. Optional slots are a documented
    /// correctness requirement, and this pins the cheap path they take.
    #[test]
    fn an_absent_optional_input_short_circuits() {
        unsafe extern "C" fn absent(
            _c: *const ort::OrtKernelContext,
            _i: usize,
            out: *mut *const ort::OrtValue,
        ) -> ort::OrtStatusPtr {
            unsafe { *out = std::ptr::null() };
            std::ptr::null_mut()
        }
        let mut api = fake_api();
        api.KernelContext_GetInput = Some(absent);

        let before = dispatch_probe::snapshot();
        let inputs =
            unsafe { read_inputs(&api, std::ptr::null_mut()) }.expect("absent is not an error");
        let d = dispatch_probe::snapshot().since(&before);

        assert_eq!(inputs.len(), 1);
        assert!(inputs[0].data_ptr.is_null());
        assert_eq!(
            d.event(Event::OrtFfiCall),
            2,
            "an absent optional input must cost the input count plus one probe, nothing more"
        );
    }

    /// A node with no inputs must not be charged for the `Vec<OwnedInput>`:
    /// `Vec::with_capacity(0)` does not call the allocator. Counting it anyway
    /// would make every constant-producing node look one allocation worse than
    /// it is.
    #[test]
    fn a_node_with_no_inputs_allocates_nothing() {
        unsafe extern "C" fn none(
            _c: *const ort::OrtKernelContext,
            out: *mut usize,
        ) -> ort::OrtStatusPtr {
            unsafe { *out = 0 };
            std::ptr::null_mut()
        }
        let mut api = fake_api();
        api.KernelContext_GetInputCount = Some(none);

        let before = dispatch_probe::snapshot();
        let inputs =
            unsafe { read_inputs(&api, std::ptr::null_mut()) }.expect("zero inputs is fine");
        let d = dispatch_probe::snapshot().since(&before);

        assert!(inputs.is_empty());
        assert_eq!(d.event(Event::OrtFfiCall), 1);
        assert_eq!(
            d.event(Event::DispatchAlloc),
            0,
            "an empty Vec::with_capacity must not be counted as an allocation"
        );
    }

    /// The probe must not perturb what it measures: with the counters compiled
    /// in, `read_inputs` still returns exactly the same tensor description.
    #[test]
    fn instrumentation_does_not_change_results() {
        let api = fake_api();
        let a = unsafe { read_inputs(&api, std::ptr::null_mut()) }.unwrap();
        dispatch_probe::reset();
        let b = unsafe { read_inputs(&api, std::ptr::null_mut()) }.unwrap();
        assert_eq!(a[0].shape, b[0].shape);
        assert_eq!(a[0].strides, b[0].strides);
        assert_eq!(a[0].dtype, b[0].dtype);
        assert_eq!(a[0].data_ptr, b[0].data_ptr);
    }
}
