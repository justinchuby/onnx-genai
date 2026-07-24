#![allow(dead_code)]

use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, DeviceId, Node, NodeId, compute_contiguous_strides};

#[derive(Clone, Copy, Debug)]
pub enum FloatDType {
    F32,
    F16,
    Bf16,
}

impl FloatDType {
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    fn data_type(self) -> DataType {
        match self {
            Self::F32 => DataType::Float32,
            Self::F16 => DataType::Float16,
            Self::Bf16 => DataType::BFloat16,
        }
    }
}

enum Storage {
    F32(Vec<f32>),
    U16(Vec<u16>),
    I64(Vec<i64>),
    I32(Vec<i32>),
}

pub struct Tensor {
    storage: Storage,
    shape: Vec<usize>,
    strides: Vec<i64>,
    dtype: DataType,
}

impl Tensor {
    pub fn floats(dtype: FloatDType, shape: &[usize], values: &[f32]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        let storage = match dtype {
            FloatDType::F32 => Storage::F32(values.to_vec()),
            FloatDType::F16 => {
                Storage::U16(values.iter().map(|&v| f16::from_f32(v).to_bits()).collect())
            }
            FloatDType::Bf16 => Storage::U16(
                values
                    .iter()
                    .map(|&v| bf16::from_f32(v).to_bits())
                    .collect(),
            ),
        };
        Self::new(storage, dtype.data_type(), shape)
    }

    pub fn zeros(dtype: FloatDType, shape: &[usize]) -> Self {
        let len = shape.iter().product();
        let storage = match dtype {
            FloatDType::F32 => Storage::F32(vec![0.0; len]),
            FloatDType::F16 | FloatDType::Bf16 => Storage::U16(vec![0; len]),
        };
        Self::new(storage, dtype.data_type(), shape)
    }

    pub fn i64(shape: &[usize], values: &[i64]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        Self::new(Storage::I64(values.to_vec()), DataType::Int64, shape)
    }

    pub fn i32(shape: &[usize], values: &[i32]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        Self::new(Storage::I32(values.to_vec()), DataType::Int32, shape)
    }

    fn new(storage: Storage, dtype: DataType, shape: &[usize]) -> Self {
        Self {
            storage,
            shape: shape.to_vec(),
            strides: compute_contiguous_strides(shape),
            dtype,
        }
    }

    fn const_ptr(&self) -> *const std::ffi::c_void {
        match &self.storage {
            Storage::F32(values) => values.as_ptr().cast(),
            Storage::U16(values) => values.as_ptr().cast(),
            Storage::I64(values) => values.as_ptr().cast(),
            Storage::I32(values) => values.as_ptr().cast(),
        }
    }

    fn mut_ptr(&mut self) -> *mut std::ffi::c_void {
        match &mut self.storage {
            Storage::F32(values) => values.as_mut_ptr().cast(),
            Storage::U16(values) => values.as_mut_ptr().cast(),
            Storage::I64(values) => values.as_mut_ptr().cast(),
            Storage::I32(values) => values.as_mut_ptr().cast(),
        }
    }

    pub fn view(&self) -> TensorView<'_> {
        TensorView::new(
            DevicePtr(self.const_ptr()),
            self.dtype,
            &self.shape,
            &self.strides,
            DeviceId::cpu(),
        )
    }

    pub fn view_mut(&mut self) -> TensorMut<'_> {
        let data = DevicePtrMut(self.mut_ptr());
        TensorMut::new(
            data,
            self.dtype,
            &self.shape,
            &self.strides,
            DeviceId::cpu(),
        )
    }

    pub fn to_f32(&self) -> Vec<f32> {
        match (&self.storage, self.dtype) {
            (Storage::F32(values), DataType::Float32) => values.clone(),
            (Storage::U16(values), DataType::Float16) => values
                .iter()
                .map(|&bits| f16::from_bits(bits).to_f32())
                .collect(),
            (Storage::U16(values), DataType::BFloat16) => values
                .iter()
                .map(|&bits| bf16::from_bits(bits).to_f32())
                .collect(),
            _ => panic!("tensor is not a supported floating-point tensor"),
        }
    }
}

pub fn float_values(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % 251) as i32 - 125) as f32 / 64.0)
        .collect()
}

pub fn make_kernel(
    op_type: &str,
    attributes: impl IntoIterator<Item = (&'static str, Attribute)>,
    input_shapes: &[Vec<usize>],
    opset: u64,
) -> Box<dyn Kernel> {
    let mut node = Node::new(NodeId(0), op_type, vec![], vec![]);
    for (name, value) in attributes {
        node.attributes.insert(name.into(), value);
    }
    CpuExecutionProvider::new()
        .get_kernel(&node, input_shapes, opset)
        .unwrap()
}

pub fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: got {actual}, expected {expected} (tolerance {tolerance})"
        );
    }
}
