//! Global pooling and axis-wise `LpNormalization`.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float load_float_value(
    const void* values, int dtype, unsigned long long index) {
  if (dtype == 0) return ((const float*)values)[index];
  if (dtype == 1) return __half2float(((const __half*)values)[index]);
  return __bfloat162float(((const __nv_bfloat16*)values)[index]);
}

__device__ __forceinline__ void store_float_value(
    void* values, int dtype, unsigned long long index, float value) {
  if (dtype == 0) ((float*)values)[index] = value;
  else if (dtype == 1) ((__half*)values)[index] = __float2half_rn(value);
  else ((__nv_bfloat16*)values)[index] = __float2bfloat16_rn(value);
}

extern "C" __global__ void global_pool(
    const void* x, void* y, unsigned long long groups,
    unsigned long long spatial, int dtype, int kind, int p) {
  const unsigned long long group = blockIdx.x;
  if (group >= groups) return;
  extern __shared__ float reduction[];
  const float negative_infinity = -__int_as_float(0x7f800000);
  float value = kind == 1 ? negative_infinity : 0.0f;
  for (unsigned long long i = threadIdx.x; i < spatial; i += blockDim.x) {
    const float input = load_float_value(x, dtype, group * spatial + i);
    if (kind == 0) value += input;
    else if (kind == 1) value = fmaxf(value, input);
    else value += powf(fabsf(input), (float)p);
  }
  reduction[threadIdx.x] = value;
  __syncthreads();
  for (unsigned int offset = blockDim.x >> 1; offset; offset >>= 1) {
    if (threadIdx.x < offset) {
      if (kind == 1)
        reduction[threadIdx.x] =
            fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + offset]);
      else
        reduction[threadIdx.x] += reduction[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float output = reduction[0];
    if (spatial == 0) output = kind == 1 ? negative_infinity : 0.0f;
    else if (kind == 0) output /= (float)spatial;
    else if (kind == 2) output = powf(output, 1.0f / (float)p);
    store_float_value(y, dtype, group, output);
  }
}

extern "C" __global__ void lp_normalization(
    const void* x, void* y, unsigned long long groups,
    unsigned long long axis_length, unsigned long long inner,
    int dtype, int p) {
  const unsigned long long group = blockIdx.x;
  if (group >= groups) return;
  const unsigned long long outer_index = group / inner;
  const unsigned long long inner_index = group % inner;
  const unsigned long long base = outer_index * axis_length * inner + inner_index;
  extern __shared__ float reduction[];
  float norm = 0.0f;
  for (unsigned long long axis_index = threadIdx.x;
       axis_index < axis_length; axis_index += blockDim.x) {
    const float value =
        fabsf(load_float_value(x, dtype, base + axis_index * inner));
    norm += p == 1 ? value : value * value;
  }
  reduction[threadIdx.x] = norm;
  __syncthreads();
  for (unsigned int offset = blockDim.x >> 1; offset; offset >>= 1) {
    if (threadIdx.x < offset)
      reduction[threadIdx.x] += reduction[threadIdx.x + offset];
    __syncthreads();
  }
  norm = p == 1 ? reduction[0] : sqrtf(reduction[0]);
  norm = fmaxf(norm, 1.1754943508222875e-38f);
  for (unsigned long long axis_index = threadIdx.x;
       axis_index < axis_length; axis_index += blockDim.x) {
    const unsigned long long index = base + axis_index * inner;
    store_float_value(y, dtype, index, load_float_value(x, dtype, index) / norm);
  }
}
"#;

fn dtype_code(dtype: DataType, op: &str) -> Result<i32> {
    match dtype {
        DataType::Float32 => Ok(0),
        DataType::Float16 => Ok(1),
        DataType::BFloat16 => Ok(2),
        other => Err(not_implemented(format!(
            "{op} dtype {other:?} (supported: Float32, Float16, BFloat16)"
        ))),
    }
}

#[derive(Clone, Copy)]
pub enum GlobalPoolKind {
    Average,
    Max,
    Lp(i32),
}

pub struct GlobalPoolFactory {
    pub kind: GlobalPoolKind,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for GlobalPoolFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let kind = match self.kind {
            GlobalPoolKind::Lp(_) => {
                let p = node
                    .attr("p")
                    .and_then(|attribute| attribute.as_int())
                    .unwrap_or(2);
                let p = i32::try_from(p).ok().filter(|p| *p > 0).ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GlobalLpPool: p must be a positive 32-bit integer".into(),
                    )
                })?;
                GlobalPoolKind::Lp(p)
            }
            other => other,
        };
        Ok(Box::new(GlobalPoolKernel {
            kind,
            runtime: self.runtime.clone(),
        }))
    }
}

struct GlobalPoolKernel {
    kind: GlobalPoolKind,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for GlobalPoolKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep GlobalPool: expected 1 input and 1 output".into(),
            ));
        }
        let input = &inputs[0];
        let output = &mut outputs[0];
        if input.shape.len() < 3 {
            return Err(EpError::KernelFailed(
                "cuda_ep GlobalPool: input must have rank at least 3".into(),
            ));
        }
        if input.dtype != output.dtype || !input.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented(
                "GlobalPool requires contiguous input/output with matching dtypes",
            ));
        }
        let dtype = dtype_code(input.dtype, "GlobalPool")?;
        if dtype != 0 {
            self.runtime.require_nvrtc_half_headers("GlobalPool")?;
        }
        let expected = [input.shape[0], input.shape[1]]
            .into_iter()
            .chain(std::iter::repeat_n(1, input.shape.len() - 2))
            .collect::<Vec<_>>();
        if output.shape != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GlobalPool: output shape {:?}, expected {expected:?}",
                output.shape
            )));
        }
        let groups = input.shape[0].saturating_mul(input.shape[1]) as u64;
        if groups == 0 {
            return Ok(());
        }
        let spatial = input.shape[2..].iter().product::<usize>() as u64;
        let (kind, p) = match self.kind {
            GlobalPoolKind::Average => (0i32, 1i32),
            GlobalPoolKind::Max => (1, 1),
            GlobalPoolKind::Lp(p) => (2, p),
        };
        let function = self
            .runtime
            .nvrtc_function("global_reduction_v1", SOURCE, "global_pool")?;
        let x = cuptr(input.data_ptr::<u8>() as *const c_void);
        let y = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&x)
            .arg(&y)
            .arg(&groups)
            .arg(&spatial)
            .arg(&dtype)
            .arg(&kind)
            .arg(&p);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    u32::try_from(groups).map_err(|_| {
                        EpError::KernelFailed("cuda_ep GlobalPool: group count exceeds u32".into())
                    })?,
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * 4,
            })
        }
        .map(|_| ())
        .map_err(|error| driver_err("launch GlobalPool", error))
    }
}

pub struct LpNormalizationFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for LpNormalizationFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let p = node
            .attr("p")
            .and_then(|attribute| attribute.as_int())
            .unwrap_or(2);
        if !matches!(p, 1 | 2) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LpNormalization: p must be 1 or 2, got {p}"
            )));
        }
        Ok(Box::new(LpNormalizationKernel {
            axis: node
                .attr("axis")
                .and_then(|attribute| attribute.as_int())
                .unwrap_or(-1),
            p: p as i32,
            runtime: self.runtime.clone(),
        }))
    }
}

struct LpNormalizationKernel {
    axis: i64,
    p: i32,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for LpNormalizationKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep LpNormalization: expected 1 input and 1 output".into(),
            ));
        }
        let input = &inputs[0];
        let output = &mut outputs[0];
        if input.shape != output.shape
            || input.dtype != output.dtype
            || !input.is_contiguous()
            || !output.is_contiguous()
        {
            return Err(not_implemented(
                "LpNormalization requires contiguous same-shape, same-dtype tensors",
            ));
        }
        let rank = input.shape.len();
        let axis = if self.axis < 0 {
            self.axis + rank as i64
        } else {
            self.axis
        };
        if axis < 0 || axis as usize >= rank {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LpNormalization: axis {} out of range for rank {rank}",
                self.axis
            )));
        }
        let dtype = dtype_code(input.dtype, "LpNormalization")?;
        if dtype != 0 {
            self.runtime.require_nvrtc_half_headers("LpNormalization")?;
        }
        let axis = axis as usize;
        let outer = input.shape[..axis].iter().product::<usize>();
        let axis_length = input.shape[axis];
        let inner = input.shape[axis + 1..].iter().product::<usize>();
        let groups = outer.saturating_mul(inner) as u64;
        if groups == 0 || axis_length == 0 {
            return Ok(());
        }
        let axis_length = axis_length as u64;
        let inner = inner as u64;
        let function =
            self.runtime
                .nvrtc_function("global_reduction_v1", SOURCE, "lp_normalization")?;
        let x = cuptr(input.data_ptr::<u8>() as *const c_void);
        let y = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&x)
            .arg(&y)
            .arg(&groups)
            .arg(&axis_length)
            .arg(&inner)
            .arg(&dtype)
            .arg(&self.p);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    u32::try_from(groups).map_err(|_| {
                        EpError::KernelFailed(
                            "cuda_ep LpNormalization: group count exceeds u32".into(),
                        )
                    })?,
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * 4,
            })
        }
        .map(|_| ())
        .map_err(|error| driver_err("launch LpNormalization", error))
    }
}
