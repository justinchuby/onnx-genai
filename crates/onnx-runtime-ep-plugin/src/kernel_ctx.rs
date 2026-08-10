//! `OutboundKernelContext` — bridges `OrtKernelContext` ↔ nxrt `TensorView`/`TensorMut`.
//!
//! When ORT calls our `Compute` callback, it passes an `OrtKernelContext*`. We
//! read inputs via `KernelContext_GetInput` and write outputs via
//! `KernelContext_GetOutput`, converting between ORT's `OrtValue*` and our
//! `TensorView`/`TensorMut`.
//!
//! # Deferred
//!
//! Device (non-host) memory access is not supported in v1. Fail closed if a
//! tensor's data pointer is null or device-only.

use std::ffi::c_void;

use onnx_genai_ort_sys as ort;
use onnx_runtime_ep_api::tensor::{DevicePtr, DevicePtrMut, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, DeviceId};

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
    let get_dims = api
        .GetDimensions
        .ok_or("OrtApi.GetDimensions is null")?;
    let release_type_shape = api
        .ReleaseTensorTypeAndShapeInfo
        .ok_or("OrtApi.ReleaseTensorTypeAndShapeInfo is null")?;
    let get_tensor_data = api
        .GetTensorData
        .ok_or("OrtApi.GetTensorData is null")?;

    let mut input_count: usize = 0;
    let status = unsafe { get_input_count(ctx, &mut input_count) };
    if !status.is_null() {
        return Err("KernelContext_GetInputCount failed".into());
    }

    let mut inputs = Vec::with_capacity(input_count);
    for i in 0..input_count {
        let mut value: *const ort::OrtValue = std::ptr::null();
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
        let status = unsafe { get_type_shape(value, &mut type_shape) };
        if !status.is_null() || type_shape.is_null() {
            return Err(format!("GetTensorTypeAndShape failed for input {i}"));
        }

        // Element type.
        let mut elem_type: ort::ONNXTensorElementDataType = 0;
        let status = unsafe { get_elem_type(type_shape, &mut elem_type) };
        if !status.is_null() {
            unsafe { release_type_shape(type_shape) };
            return Err(format!("GetTensorElementType failed for input {i}"));
        }
        let dtype = DataType::from_onnx(elem_type as i32).ok_or_else(|| {
            format!("unsupported element type {elem_type} for input {i}")
        })?;

        // Dimensions.
        let mut ndim: usize = 0;
        let status = unsafe { get_dims_count(type_shape, &mut ndim) };
        if !status.is_null() {
            unsafe { release_type_shape(type_shape) };
            return Err(format!("GetDimensionsCount failed for input {i}"));
        }
        let mut dims = vec![0i64; ndim];
        let status = unsafe { get_dims(type_shape, dims.as_mut_ptr(), ndim) };
        if !status.is_null() {
            unsafe { release_type_shape(type_shape) };
            return Err(format!("GetDimensions failed for input {i}"));
        }
        unsafe { release_type_shape(type_shape) };

        let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        let strides = onnx_runtime_ir::compute_contiguous_strides(&shape);

        // Data pointer.
        let mut data: *const c_void = std::ptr::null();
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
    let get_output = api
        .KernelContext_GetOutput
        .ok_or("OrtApi.KernelContext_GetOutput is null")?;
    let get_mutable_data = api
        .GetTensorMutableData
        .ok_or("OrtApi.GetTensorMutableData is null")?;

    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let mut value: *mut ort::OrtValue = std::ptr::null_mut();
    let status =
        unsafe { get_output(ctx, index, dims.as_ptr(), dims.len(), &mut value) };
    if !status.is_null() {
        return Err(format!("KernelContext_GetOutput({index}) failed"));
    }
    if value.is_null() {
        return Err(format!("KernelContext_GetOutput({index}) returned null"));
    }

    let mut data: *mut c_void = std::ptr::null_mut();
    let status = unsafe { get_mutable_data(value, &mut data) };
    if !status.is_null() {
        return Err(format!("GetTensorMutableData failed for output {index}"));
    }
    if data.is_null() {
        return Err(format!(
            "output {index} data pointer is null (device-only memory not supported)"
        ));
    }

    let strides = onnx_runtime_ir::compute_contiguous_strides(shape);
    Ok(OwnedOutput {
        data_ptr: data,
        dtype,
        shape: shape.to_vec(),
        strides,
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
            assert_eq!(dt.to_onnx(), onnx_val, "to_onnx round-trip for {expected:?}");
        }
    }

    #[test]
    fn dtype_mapping_unsupported_returns_none() {
        // 0 is UNDEFINED in ORT, should return None.
        assert!(DataType::from_onnx(0).is_none());
        // Extremely large value should also be None.
        assert!(DataType::from_onnx(9999).is_none());
    }
}
