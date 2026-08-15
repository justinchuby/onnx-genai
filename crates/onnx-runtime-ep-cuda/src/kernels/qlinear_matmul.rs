//! CUDA `QLinearMatMul` with integer accumulation and ONNX requantization.

use std::ffi::c_void;
use std::mem::size_of;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg, sys::CUdeviceptr};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node, broadcast_shapes, compute_contiguous_strides};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
__device__ __forceinline__ int load_quantized(
    const void* values, int dtype, unsigned long long index) {
  return dtype == 0 ? (int)((const signed char*)values)[index]
                    : (int)((const unsigned char*)values)[index];
}

__device__ __forceinline__ void store_quantized(
    void* values, int dtype, unsigned long long index, long long value) {
  if (dtype == 0) {
    value = value < -128 ? -128 : (value > 127 ? 127 : value);
    ((signed char*)values)[index] = (signed char)value;
  } else {
    value = value < 0 ? 0 : (value > 255 ? 255 : value);
    ((unsigned char*)values)[index] = (unsigned char)value;
  }
}

extern "C" __global__ void qlinear_matmul(
    const void* a, const float* a_scale, const void* a_zero_point,
    const void* b, const float* b_scale, const void* b_zero_point,
    const float* y_scale, const void* y_zero_point, void* y,
    const unsigned long long* batch_offsets, unsigned long long elements,
    unsigned long long m, unsigned long long k, unsigned long long n,
    unsigned long long a_axis_length, unsigned long long b_axis_length,
    int a_dtype, int b_dtype, int y_dtype, int a_per_axis, int b_per_axis) {
  const int output_zero_point = load_quantized(y_zero_point, y_dtype, 0);
  const float output_scale = *y_scale;
  for (unsigned long long output_index =
           blockIdx.x * blockDim.x + threadIdx.x;
       output_index < elements;
       output_index += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long matrix_index = output_index / (m * n);
    const unsigned long long within_matrix = output_index % (m * n);
    const unsigned long long row = within_matrix / n;
    const unsigned long long column = within_matrix % n;
    const unsigned long long a_batch = batch_offsets[matrix_index * 2];
    const unsigned long long b_batch = batch_offsets[matrix_index * 2 + 1];
    const unsigned long long a_base = a_batch * m * k;
    const unsigned long long b_base = b_batch * k * n;
    const unsigned long long a_parameter =
        a_per_axis ? a_batch * a_axis_length + row : 0;
    const unsigned long long b_parameter =
        b_per_axis ? b_batch * b_axis_length + column : 0;
    const int a_zero = load_quantized(a_zero_point, a_dtype, a_parameter);
    const int b_zero = load_quantized(b_zero_point, b_dtype, b_parameter);
    unsigned int accumulated_bits = 0;
    for (unsigned long long inner = 0; inner < k; ++inner) {
      const int av = load_quantized(a, a_dtype, a_base + row * k + inner) - a_zero;
      const int bv = load_quantized(b, b_dtype, b_base + inner * n + column) - b_zero;
      accumulated_bits += (unsigned int)(av * bv);
    }
    const int accumulated = (int)accumulated_bits;
    const float scale = a_scale[a_parameter] * b_scale[b_parameter] / output_scale;
    const float rounded = nearbyintf((float)accumulated * scale);
    const long long rounded_integer =
        isnan(rounded) ? 0 : (rounded > 512.0f ? 512 : (rounded < -512.0f ? -512 : (long long)rounded));
    const long long requantized = rounded_integer + (long long)output_zero_point;
    store_quantized(y, y_dtype, output_index, requantized);
  }
}
"#;

pub struct QLinearMatMulFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for QLinearMatMulFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(QLinearMatMulKernel {
            runtime: self.runtime.clone(),
        }))
    }
}

struct QLinearMatMulKernel {
    runtime: Arc<CudaRuntime>,
}

#[derive(Clone, Copy)]
enum QuantAxis {
    Row,
    Column,
}

struct QuantParameters {
    axis_length: usize,
    per_axis: bool,
}

impl QuantParameters {
    fn validate(
        name: &str,
        scale: &TensorView,
        zero_point: &TensorView,
        operand: &[usize],
        axis: QuantAxis,
        quantized_dtype: DataType,
    ) -> Result<Self> {
        if scale.dtype != DataType::Float32 || zero_point.dtype != quantized_dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep QLinearMatMul: {name}_scale must be Float32 and {name}_zero_point must match the operand dtype"
            )));
        }
        if scale.shape != zero_point.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep QLinearMatMul: {name}_scale and {name}_zero_point shapes must match"
            )));
        }
        let axis_length = match axis {
            QuantAxis::Row => {
                if operand.len() == 1 {
                    1
                } else {
                    operand[operand.len() - 2]
                }
            }
            QuantAxis::Column => *operand.last().unwrap_or(&1),
        };
        let scalar = scale.shape.is_empty() || scale.shape == [1];
        let per_axis = if scalar {
            false
        } else if quant_shape_matches(scale.shape, operand, axis) {
            true
        } else {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep QLinearMatMul: invalid {name} scale/zero-point shape {:?} for operand {operand:?}",
                scale.shape
            )));
        };
        Ok(Self {
            axis_length,
            per_axis,
        })
    }
}

fn quant_shape_matches(shape: &[usize], operand: &[usize], axis: QuantAxis) -> bool {
    match operand.len() {
        0 | 1 => false,
        2 => {
            shape
                == [operand[match axis {
                    QuantAxis::Row => 0,
                    QuantAxis::Column => 1,
                }]]
        }
        rank => {
            if shape.len() != rank || shape[..rank - 2] != operand[..rank - 2] {
                return false;
            }
            match axis {
                QuantAxis::Row => shape[rank - 2] == operand[rank - 2] && shape[rank - 1] == 1,
                QuantAxis::Column => shape[rank - 2] == 1 && shape[rank - 1] == operand[rank - 1],
            }
        }
    }
}

fn quant_dtype(dtype: DataType, name: &str) -> Result<i32> {
    match dtype {
        DataType::Int8 => Ok(0),
        DataType::Uint8 => Ok(1),
        other => Err(not_implemented(format!(
            "QLinearMatMul {name} dtype {other:?} (supported: Int8, Uint8)"
        ))),
    }
}

struct Geometry {
    m: usize,
    k: usize,
    n: usize,
    batch_shape: Vec<usize>,
    a_batch_shape: Vec<usize>,
    b_batch_shape: Vec<usize>,
    a_batch_strides: Vec<i64>,
    b_batch_strides: Vec<i64>,
    output_shape: Vec<usize>,
}

impl Geometry {
    fn new(a: &[usize], b: &[usize]) -> Result<Self> {
        if a.is_empty() || b.is_empty() {
            return Err(EpError::KernelFailed(
                "cuda_ep QLinearMatMul: operands must be at least 1-D".into(),
            ));
        }
        let a_was_vector = a.len() == 1;
        let b_was_vector = b.len() == 1;
        let promoted_a = if a_was_vector {
            vec![1, a[0]]
        } else {
            a.to_vec()
        };
        let promoted_b = if b_was_vector {
            vec![b[0], 1]
        } else {
            b.to_vec()
        };
        let m = promoted_a[promoted_a.len() - 2];
        let k = promoted_a[promoted_a.len() - 1];
        let b_k = promoted_b[promoted_b.len() - 2];
        let n = promoted_b[promoted_b.len() - 1];
        if k != b_k {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep QLinearMatMul: inner dimensions disagree ({k} vs {b_k})"
            )));
        }
        let a_batch_shape = promoted_a[..promoted_a.len() - 2].to_vec();
        let b_batch_shape = promoted_b[..promoted_b.len() - 2].to_vec();
        let batch_shape = broadcast_shapes(&a_batch_shape, &b_batch_shape).map_err(|error| {
            EpError::KernelFailed(format!(
                "cuda_ep QLinearMatMul: batch dimensions do not broadcast: {error}"
            ))
        })?;
        let mut output_shape = batch_shape.clone();
        if !a_was_vector {
            output_shape.push(m);
        }
        if !b_was_vector {
            output_shape.push(n);
        }
        Ok(Self {
            m,
            k,
            n,
            a_batch_strides: compute_contiguous_strides(&a_batch_shape),
            b_batch_strides: compute_contiguous_strides(&b_batch_shape),
            a_batch_shape,
            b_batch_shape,
            batch_shape,
            output_shape,
        })
    }

    fn batch_offsets(&self) -> Vec<u64> {
        let count = self.batch_shape.iter().product::<usize>();
        let mut offsets = Vec::with_capacity(count * 2);
        let mut index = vec![0; self.batch_shape.len()];
        for batch in 0..count {
            offsets
                .push(broadcast_offset(&index, &self.a_batch_shape, &self.a_batch_strides) as u64);
            offsets
                .push(broadcast_offset(&index, &self.b_batch_shape, &self.b_batch_strides) as u64);
            if batch + 1 < count {
                next_index(&self.batch_shape, &mut index);
            }
        }
        offsets
    }
}

fn broadcast_offset(index: &[usize], shape: &[usize], strides: &[i64]) -> usize {
    let leading = index.len() - shape.len();
    shape
        .iter()
        .zip(strides)
        .enumerate()
        .map(|(axis, (&dimension, &stride))| {
            if dimension == 1 {
                0
            } else {
                index[leading + axis] * stride as usize
            }
        })
        .sum()
}

fn next_index(shape: &[usize], index: &mut [usize]) {
    for (&dimension, coordinate) in shape.iter().zip(index).rev() {
        *coordinate += 1;
        if *coordinate < dimension {
            return;
        }
        *coordinate = 0;
    }
}

fn upload_offsets(runtime: &CudaRuntime, values: &[u64]) -> Result<CUdeviceptr> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let pointer = runtime.alloc_raw(bytes.len())?;
    if let Err(error) = unsafe { runtime.htod(&bytes, pointer) } {
        let _ = unsafe { runtime.free_raw(pointer) };
        return Err(error);
    }
    Ok(pointer)
}

fn validate_scale_values(runtime: &CudaRuntime, scale: &TensorView) -> Result<()> {
    let mut bytes = vec![0_u8; scale.numel() * size_of::<f32>()];
    if !bytes.is_empty() {
        // SAFETY: the validated scale tensor is contiguous Float32 storage and `bytes`
        // has exactly the corresponding allocation length.
        unsafe { runtime.dtoh(&mut bytes, cuptr(scale.data_ptr::<f32>() as *const c_void))? };
    }
    if bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .any(|value| value <= 0.0 || !value.is_finite())
    {
        return Err(EpError::KernelFailed(
            "cuda_ep QLinearMatMul: scales must be finite and positive".into(),
        ));
    }
    Ok(())
}

impl Kernel for QLinearMatMulKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 8 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep QLinearMatMul: expected 8 inputs and 1 output".into(),
            ));
        }
        if inputs.iter().any(|input| !input.is_contiguous()) || !outputs[0].is_contiguous() {
            return Err(not_implemented("QLinearMatMul with non-contiguous tensors"));
        }
        let a_dtype = quant_dtype(inputs[0].dtype, "A")?;
        let b_dtype = quant_dtype(inputs[3].dtype, "B")?;
        let y_dtype = quant_dtype(outputs[0].dtype, "Y")?;
        if inputs[7].dtype != outputs[0].dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep QLinearMatMul: y_zero_point dtype must match Y".into(),
            ));
        }
        let geometry = Geometry::new(inputs[0].shape, inputs[3].shape)?;
        if outputs[0].shape != geometry.output_shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep QLinearMatMul: output shape {:?}, expected {:?}",
                outputs[0].shape, geometry.output_shape
            )));
        }
        let a_parameters = QuantParameters::validate(
            "a",
            &inputs[1],
            &inputs[2],
            inputs[0].shape,
            QuantAxis::Row,
            inputs[0].dtype,
        )?;
        let b_parameters = QuantParameters::validate(
            "b",
            &inputs[4],
            &inputs[5],
            inputs[3].shape,
            QuantAxis::Column,
            inputs[3].dtype,
        )?;
        if inputs[6].dtype != DataType::Float32
            || inputs[6].shape != inputs[7].shape
            || !(inputs[6].shape.is_empty() || inputs[6].shape == [1])
        {
            return Err(EpError::KernelFailed(
                "cuda_ep QLinearMatMul: output scale and zero point must be matching scalars"
                    .into(),
            ));
        }
        if self.runtime.is_capturing()? {
            return Err(not_implemented(
                "QLinearMatMul during CUDA graph capture because scale values require validation",
            ));
        }
        for scale in [&inputs[1], &inputs[4], &inputs[6]] {
            validate_scale_values(&self.runtime, scale)?;
        }
        let elements = outputs[0].numel() as u64;
        if elements == 0 {
            return Ok(());
        }
        let offsets = geometry.batch_offsets();
        let offsets_pointer = upload_offsets(&self.runtime, &offsets)?;
        let function =
            self.runtime
                .nvrtc_function("qlinear_matmul_v1", SOURCE, "qlinear_matmul")?;
        let pointers = inputs
            .iter()
            .map(|input| cuptr(input.data_ptr::<u8>() as *const c_void))
            .collect::<Vec<_>>();
        let output_pointer = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let m = geometry.m as u64;
        let k = geometry.k as u64;
        let n = geometry.n as u64;
        let a_axis_length = a_parameters.axis_length as u64;
        let b_axis_length = b_parameters.axis_length as u64;
        let a_per_axis = i32::from(a_parameters.per_axis);
        let b_per_axis = i32::from(b_parameters.per_axis);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&pointers[0])
            .arg(&pointers[1])
            .arg(&pointers[2])
            .arg(&pointers[3])
            .arg(&pointers[4])
            .arg(&pointers[5])
            .arg(&pointers[6])
            .arg(&pointers[7])
            .arg(&output_pointer)
            .arg(&offsets_pointer)
            .arg(&elements)
            .arg(&m)
            .arg(&k)
            .arg(&n)
            .arg(&a_axis_length)
            .arg(&b_axis_length)
            .arg(&a_dtype)
            .arg(&b_dtype)
            .arg(&y_dtype)
            .arg(&a_per_axis)
            .arg(&b_per_axis);
        let launch = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    elements.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32,
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch QLinearMatMul", error));
        let sync = launch.and_then(|_| self.runtime.synchronize());
        let free = unsafe { self.runtime.free_raw(offsets_pointer) };
        sync.and(free)
    }
}
