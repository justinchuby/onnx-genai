//! Per-tensor ONNX linear quantization kernels.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node, Shape};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
extern "C" __global__ void quantize_u8(
    const float* x, const float* scale, const unsigned char* zero_point,
    unsigned char* y, unsigned long long n) {
  const float s = *scale;
  const int zp = zero_point ? (int)*zero_point : 0;
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
    int q = __float2int_rn(x[i] / s) + zp;
    y[i] = (unsigned char)(q < 0 ? 0 : (q > 255 ? 255 : q));
  }
}
extern "C" __global__ void quantize_i8(
    const float* x, const float* scale, const signed char* zero_point,
    signed char* y, unsigned long long n) {
  const float s = *scale;
  const int zp = zero_point ? (int)*zero_point : 0;
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
    int q = __float2int_rn(x[i] / s) + zp;
    y[i] = (signed char)(q < -128 ? -128 : (q > 127 ? 127 : q));
  }
}
extern "C" __global__ void dequantize_u8(
    const unsigned char* x, const float* scale, const unsigned char* zero_point,
    float* y, unsigned long long n) {
  const float s = *scale;
  const int zp = zero_point ? (int)*zero_point : 0;
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (unsigned long long)gridDim.x * blockDim.x)
    y[i] = ((int)x[i] - zp) * s;
}
extern "C" __global__ void dequantize_i8(
    const signed char* x, const float* scale, const signed char* zero_point,
    float* y, unsigned long long n) {
  const float s = *scale;
  const int zp = zero_point ? (int)*zero_point : 0;
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (unsigned long long)gridDim.x * blockDim.x)
    y[i] = ((int)x[i] - zp) * s;
}
"#;

pub fn unsupported_reason(op: &Node, shapes: &[Shape]) -> Option<String> {
    if !(2..=3).contains(&op.inputs.len()) || shapes.len() != op.inputs.len() {
        return Some(format!(
            "{} requires 2 or 3 present inputs with shape metadata",
            op.op_type
        ));
    }
    if !shapes[1].is_empty() {
        return Some(format!(
            "{} CUDA coverage requires a scalar scale",
            op.op_type
        ));
    }
    if shapes.get(2).is_some_and(|shape| !shape.is_empty()) {
        return Some(format!(
            "{} CUDA coverage requires a scalar zero_point",
            op.op_type
        ));
    }
    None
}

#[derive(Clone, Copy, Debug)]
pub enum LinearQuantOp {
    Quantize,
    Dequantize,
}

pub struct LinearQuantFactory {
    pub op: LinearQuantOp,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for LinearQuantFactory {
    fn create(&self, _: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(LinearQuantKernel {
            op: self.op,
            runtime: self.runtime.clone(),
        }))
    }
}

struct LinearQuantKernel {
    op: LinearQuantOp,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for LinearQuantKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let name = match self.op {
            LinearQuantOp::Quantize => "QuantizeLinear",
            LinearQuantOp::Dequantize => "DequantizeLinear",
        };
        if !(2..=3).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {name}: expected 2..=3 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        if inputs.iter().any(|v| !v.is_contiguous()) || !outputs[0].is_contiguous() {
            return Err(not_implemented(format!("{name} with strided tensors")));
        }
        let x = &inputs[0];
        let scale = &inputs[1];
        if scale.dtype != DataType::Float32 || !scale.shape.is_empty() {
            return Err(not_implemented(format!(
                "{name} requires a scalar Float32 scale"
            )));
        }
        let quant_dtype = match self.op {
            LinearQuantOp::Quantize => outputs[0].dtype,
            LinearQuantOp::Dequantize => x.dtype,
        };
        if !matches!(quant_dtype, DataType::Int8 | DataType::Uint8) {
            return Err(not_implemented(format!(
                "{name} quantized dtype {quant_dtype:?} (supported: Int8, Uint8)"
            )));
        }
        if let Some(zero_point) = inputs.get(2)
            && (zero_point.dtype != quant_dtype || !zero_point.shape.is_empty())
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {name}: zero_point must be a scalar {quant_dtype:?}"
            )));
        }
        match self.op {
            LinearQuantOp::Quantize
                if x.dtype != DataType::Float32 || outputs[0].shape != x.shape =>
            {
                return Err(EpError::KernelFailed(
                    "cuda_ep QuantizeLinear: Float32 input and same-shape output required".into(),
                ));
            }
            LinearQuantOp::Dequantize
                if outputs[0].dtype != DataType::Float32 || outputs[0].shape != x.shape =>
            {
                return Err(EpError::KernelFailed(
                    "cuda_ep DequantizeLinear: Float32 same-shape output required".into(),
                ));
            }
            _ => {}
        }
        let n = x.numel() as u64;
        if n == 0 {
            return Ok(());
        }
        let stem = match (self.op, quant_dtype) {
            (LinearQuantOp::Quantize, DataType::Uint8) => "quantize_u8",
            (LinearQuantOp::Quantize, DataType::Int8) => "quantize_i8",
            (LinearQuantOp::Dequantize, DataType::Uint8) => "dequantize_u8",
            (LinearQuantOp::Dequantize, DataType::Int8) => "dequantize_i8",
            _ => unreachable!(),
        };
        let function = self
            .runtime
            .nvrtc_function("linear_quant_per_tensor_v1", SOURCE, stem)?;
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let scale_ptr = cuptr(scale.data_ptr::<u8>() as *const c_void);
        let zero_point_ptr = inputs
            .get(2)
            .map_or(0, |v| cuptr(v.data_ptr::<u8>() as *const c_void));
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&x_ptr)
            .arg(&scale_ptr)
            .arg(&zero_point_ptr)
            .arg(&y_ptr)
            .arg(&n);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (n.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err(&format!("launch {name}"), error))?;
        Ok(())
    }
}
