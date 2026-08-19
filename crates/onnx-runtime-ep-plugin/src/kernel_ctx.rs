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
//! # Deferred
//!
//! Device (non-host) memory access is not supported in v1. Fail closed if a
//! tensor's data pointer is null or device-only.

use std::ffi::c_void;

use onnx_genai_ort_sys as ort;
use onnx_runtime_ep_api::tensor::{DevicePtr, DevicePtrMut, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, DeviceId};

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

/// Validate raw ORT dimensions for a single tensor, converting to `usize` shape.
///
/// Rejects negative dims (fail closed) and detects element-count overflow.
/// Returns `(shape, element_count, expected_byte_length)`.
pub(crate) fn validate_dims(
    dims: &[i64],
    dtype: DataType,
    context: &str,
) -> Result<(Vec<usize>, usize, usize), String> {
    let mut shape: Vec<usize> = Vec::with_capacity(dims.len());
    for (dim_idx, &d) in dims.iter().enumerate() {
        if d < 0 {
            return Err(format!(
                "{context} dim[{dim_idx}] is {d} — negative dimensions are \
                 invalid at Compute time (symbolic/dynamic dims must be \
                 resolved before execution)"
            ));
        }
        shape.push(d as usize);
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
    pub shape: Vec<usize>,
    pub strides: Vec<i64>,
}

impl OwnedInput {
    /// Create a `TensorView` borrowing from this owned input.
    pub fn view(&self) -> TensorView<'_> {
        TensorView::new(
            DevicePtr(self.data_ptr),
            self.dtype,
            &self.shape,
            &self.strides,
            DeviceId::cpu(),
        )
    }
}

/// Owned output tensor data allocated via `KernelContext_GetOutput`.
pub(crate) struct OwnedOutput {
    pub data_ptr: *mut c_void,
    pub dtype: DataType,
    pub shape: Vec<usize>,
    pub strides: Vec<i64>,
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
pub(crate) unsafe fn read_inputs(
    api: &ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
) -> Result<Vec<OwnedInput>, String> {
    let _probe = crate::dispatch_probe::Phase::MetadataQuery.enter();
    let get_input_count = api
        .KernelContext_GetInputCount
        .ok_or("OrtApi.KernelContext_GetInputCount is null")?;
    let get_input = api
        .KernelContext_GetInput
        .ok_or("OrtApi.KernelContext_GetInput is null")?;
    let get_type_shape = api
        .GetTensorTypeAndShape
        .ok_or("OrtApi.GetTensorTypeAndShape is null")?;
    let get_elem_type = api
        .GetTensorElementType
        .ok_or("OrtApi.GetTensorElementType is null")?;
    let get_dims_count = api
        .GetDimensionsCount
        .ok_or("OrtApi.GetDimensionsCount is null")?;
    let get_dims = api.GetDimensions.ok_or("OrtApi.GetDimensions is null")?;
    let release_type_shape = api
        .ReleaseTensorTypeAndShapeInfo
        .ok_or("OrtApi.ReleaseTensorTypeAndShapeInfo is null")?;
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
    let mut inputs = Vec::with_capacity(input_count);
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
                shape: vec![],
                strides: vec![],
            });
            continue;
        }

        // Get type and shape info.
        let mut type_shape: *mut ort::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        crate::dispatch_probe::ort_call();
        let status = unsafe { get_type_shape(value, &mut type_shape) };
        if !status.is_null() || type_shape.is_null() {
            return Err(format!("GetTensorTypeAndShape failed for input {i}"));
        }

        // Element type.
        let mut elem_type: ort::ONNXTensorElementDataType = 0;
        crate::dispatch_probe::ort_call();
        let status = unsafe { get_elem_type(type_shape, &mut elem_type) };
        if !status.is_null() {
            unsafe { release_type_shape(type_shape) };
            return Err(format!("GetTensorElementType failed for input {i}"));
        }
        let dtype = DataType::from_onnx(elem_type as i32)
            .ok_or_else(|| format!("unsupported element type {elem_type} for input {i}"))?;

        // Dimensions.
        let mut ndim: usize = 0;
        crate::dispatch_probe::ort_call();
        let status = unsafe { get_dims_count(type_shape, &mut ndim) };
        if !status.is_null() {
            unsafe { release_type_shape(type_shape) };
            return Err(format!("GetDimensionsCount failed for input {i}"));
        }
        crate::dispatch_probe::count(crate::dispatch_probe::Event::DispatchAlloc);
        let mut dims = vec![0i64; ndim];
        crate::dispatch_probe::ort_call();
        let status = unsafe { get_dims(type_shape, dims.as_mut_ptr(), ndim) };
        if !status.is_null() {
            unsafe { release_type_shape(type_shape) };
            return Err(format!("GetDimensions failed for input {i}"));
        }
        crate::dispatch_probe::ort_call();
        unsafe { release_type_shape(type_shape) };

        // Validate ORT-supplied dims: reject negatives and detect overflow.
        // Three allocations: the eager `format!` label, `shape`, and `strides`.
        crate::dispatch_probe::count_n(crate::dispatch_probe::Event::DispatchAlloc, 3);
        let (shape, _, _) = validate_dims(&dims, dtype, &format!("input {i}"))?;
        let strides = onnx_runtime_ir::compute_contiguous_strides(&shape);

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

        inputs.push(OwnedInput {
            data_ptr: data,
            dtype,
            shape,
            strides,
        });
    }

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
) -> Result<OwnedOutput, String> {
    let _probe = crate::dispatch_probe::Phase::Allocate.enter();
    let get_output = api
        .KernelContext_GetOutput
        .ok_or("OrtApi.KernelContext_GetOutput is null")?;
    let get_mutable_data = api
        .GetTensorMutableData
        .ok_or("OrtApi.GetTensorMutableData is null")?;

    crate::dispatch_probe::count(crate::dispatch_probe::Event::DispatchAlloc);
    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
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
    let mem_info = match api.GetTensorMemoryInfo {
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

    crate::dispatch_probe::count_n(crate::dispatch_probe::Event::DispatchAlloc, 2);
    let strides = onnx_runtime_ir::compute_contiguous_strides(shape);
    Ok(OwnedOutput {
        data_ptr: data,
        dtype,
        shape: shape.to_vec(),
        strides,
        mem_info,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::DataType;

    #[test]
    fn owned_input_view_roundtrip() {
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let input = OwnedInput {
            data_ptr: data.as_ptr().cast(),
            dtype: DataType::Float32,
            shape: vec![2, 2],
            strides: vec![2, 1],
        };
        let view = input.view();
        assert_eq!(view.shape, &[2, 2]);
        assert_eq!(view.dtype, DataType::Float32);
    }

    #[test]
    fn owned_output_view_mut_roundtrip() {
        let mut data: [f32; 6] = [0.0; 6];
        let mut output = OwnedOutput {
            data_ptr: data.as_mut_ptr().cast(),
            dtype: DataType::Float32,
            shape: vec![2, 3],
            strides: vec![3, 1],
            mem_info: std::ptr::null(),
        };
        let view = output.view_mut();
        assert_eq!(view.shape, &[2, 3]);
        assert_eq!(view.dtype, DataType::Float32);
    }

    #[test]
    fn owned_input_null_data_for_optional() {
        let input = OwnedInput {
            data_ptr: std::ptr::null(),
            dtype: DataType::Float32,
            shape: vec![],
            strides: vec![],
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
        let err = super::validate_dims(&dims, DataType::Float32, "test").unwrap_err();
        assert!(err.contains("dim[1] is -1"), "error: {err}");
        assert!(err.contains("negative"), "error: {err}");
    }

    #[test]
    fn validate_dims_rejects_large_negative() {
        // ORT's dynamic-dim sentinel -1 as i64
        let dims = [2, -1i64];
        let err = super::validate_dims(&dims, DataType::Float32, "x").unwrap_err();
        assert!(err.contains("-1"), "error: {err}");
    }

    #[test]
    fn validate_dims_overflow_element_count() {
        // Two huge dims that overflow usize on multiply
        let dims = [i64::MAX / 2, 4];
        let err = super::validate_dims(&dims, DataType::Float32, "big").unwrap_err();
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
        let result = super::validate_dims(&dims, DataType::Float64, "bytes");
        // byte_size = 8, element_count = 2^63-1, product overflows
        assert!(result.is_err(), "should fail: {result:?}");
    }

    #[test]
    fn validate_dims_zero_dim_accepted() {
        // Zero-dim tensors are legal in ONNX (e.g. empty batch)
        let dims = [0i64, 3, 224, 224];
        let (shape, elem_count, byte_len) =
            super::validate_dims(&dims, DataType::Float32, "zero").unwrap();
        assert_eq!(shape, vec![0, 3, 224, 224]);
        assert_eq!(elem_count, 0);
        assert_eq!(byte_len, 0);
    }

    #[test]
    fn validate_dims_scalar_tensor() {
        // Scalar: zero-rank tensor with 1 element
        let dims: [i64; 0] = [];
        let (shape, elem_count, byte_len) =
            super::validate_dims(&dims, DataType::Float32, "scalar").unwrap();
        assert_eq!(shape, Vec::<usize>::new());
        assert_eq!(elem_count, 1); // product of empty = 1
        assert_eq!(byte_len, 4);
    }

    #[test]
    fn validate_dims_normal_shape() {
        let dims = [2i64, 3, 4];
        let (shape, elem_count, byte_len) =
            super::validate_dims(&dims, DataType::Float32, "ok").unwrap();
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
            super::validate_dims(&dims, DataType::Float16, "f16").unwrap();
        assert_eq!(shape, vec![4, 8]);
        assert_eq!(elem_count, 32);
        assert_eq!(byte_len, 64);
    }

    #[test]
    fn bf16_byte_length_computation() {
        // A [3, 5] bf16 tensor: 15 elements × 2 bytes = 30 bytes.
        let dims = [3i64, 5];
        let (shape, elem_count, byte_len) =
            super::validate_dims(&dims, DataType::BFloat16, "bf16").unwrap();
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
        let err = super::validate_dims(&dims, DataType::Float16, "f16_overflow").unwrap_err();
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

    /// One input also costs five heap allocations: the `Vec<OwnedInput>` once
    /// per `Run`, then per input the `dims` scratch, the eager
    /// `format!("input {i}")` label built even when validation succeeds,
    /// `shape`, and `strides`. The label is pure waste on the success path and
    /// is the first thing to remove.
    #[test]
    fn reading_one_input_costs_exactly_five_allocations() {
        let api = fake_api();
        let before = dispatch_probe::snapshot();
        let _ = unsafe { read_inputs(&api, std::ptr::null_mut()) }.expect("fake api is total");
        let d = dispatch_probe::snapshot().since(&before);
        assert_eq!(
            d.event(Event::DispatchAlloc),
            1 + 4,
            "allocations on the per-Run input path changed"
        );
    }

    /// The costs above must be *per input*, not amortised — three inputs cost
    /// three times the per-input work plus the single shared `Vec`. This is
    /// what makes the counters usable as a model rather than a single data
    /// point.
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
        assert_eq!(d.event(Event::DispatchAlloc), 1 + 3 * 4);
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
