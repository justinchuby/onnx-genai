//! Deterministic f32 `TopK`, optimized for small router K values.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
__device__ bool before(float a, float b, long long ia, long long ib, int largest) {
  int ka = __float_as_int(a);
  int kb = __float_as_int(b);
  ka ^= (int)(((unsigned int)(ka >> 31)) >> 1);
  kb ^= (int)(((unsigned int)(kb >> 31)) >> 1);
  if (ka == kb) return ia < ib;
  return largest ? ka > kb : ka < kb;
}
extern "C" __global__ void topk_f32(
    const float* input, float* values, long long* indices,
    unsigned long long slices, unsigned long long width,
    unsigned long long inner, unsigned long long k, int largest) {
  for (unsigned long long slice = blockIdx.x * blockDim.x + threadIdx.x; slice < slices;
       slice += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long outer = slice / inner, i = slice % inner;
    for (unsigned long long out = 0; out < k; ++out) {
      long long best_index = -1;
      float best = 0.0f;
      for (unsigned long long candidate = 0; candidate < width; ++candidate) {
        bool used = false;
        for (unsigned long long prior = 0; prior < out; ++prior)
          if (indices[(outer * k + prior) * inner + i] == (long long)candidate) used = true;
        if (used) continue;
        float value = input[(outer * width + candidate) * inner + i];
        if (best_index < 0 || before(value, best, (long long)candidate, best_index, largest)) {
          best = value;
          best_index = (long long)candidate;
        }
      }
      unsigned long long offset = (outer * k + out) * inner + i;
      values[offset] = best;
      indices[offset] = best_index;
    }
  }
}
"#;

pub struct TopKFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for TopKFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(TopKKernel {
            runtime: self.runtime.clone(),
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1),
            largest: node.attr("largest").and_then(|a| a.as_int()).unwrap_or(1) != 0,
            _sorted: node.attr("sorted").and_then(|a| a.as_int()).unwrap_or(1) != 0,
        }))
    }
}

struct TopKKernel {
    runtime: Arc<CudaRuntime>,
    axis: i64,
    largest: bool,
    _sorted: bool,
}

impl Kernel for TopKKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 2 {
            return Err(EpError::KernelFailed(
                "cuda_ep TopK: expected 2 inputs and 2 outputs".into(),
            ));
        }
        let input = &inputs[0];
        let k_input = &inputs[1];
        if !input.is_contiguous()
            || !k_input.is_contiguous()
            || outputs.iter().any(|v| !v.is_contiguous())
        {
            return Err(not_implemented("TopK with non-contiguous tensors"));
        }
        if input.dtype != DataType::Float32 || outputs[0].dtype != DataType::Float32 {
            return Err(not_implemented("TopK currently supports Float32 values"));
        }
        if outputs[1].dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(
                "cuda_ep TopK: indices output must be Int64".into(),
            ));
        }
        if k_input.dtype != DataType::Int64 || k_input.numel() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep TopK: K must be an Int64 scalar".into(),
            ));
        }
        let rank = input.shape.len();
        let normalized = if self.axis < 0 {
            self.axis + rank as i64
        } else {
            self.axis
        };
        if normalized < 0 || normalized as usize >= rank {
            return Err(EpError::KernelFailed(
                "cuda_ep TopK: axis out of range".into(),
            ));
        }
        let mut bytes = [0_u8; 8];
        unsafe {
            self.runtime.dtoh(
                &mut bytes,
                cuptr(k_input.data_ptr::<i64>() as *const c_void),
            )?
        };
        let raw_k = i64::from_ne_bytes(bytes);
        if raw_k < 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep TopK: K must be non-negative".into(),
            ));
        }
        let axis = normalized as usize;
        let width = input.shape[axis];
        let k = raw_k as usize;
        if k > width {
            return Err(EpError::KernelFailed(
                "cuda_ep TopK: K exceeds selected axis".into(),
            ));
        }
        let mut expected = input.shape.to_vec();
        expected[axis] = k;
        if outputs[0].shape != expected || outputs[1].shape != expected {
            return Err(EpError::KernelFailed(
                "cuda_ep TopK: output shapes are invalid".into(),
            ));
        }
        if k == 0 {
            return Ok(());
        }
        let inner = input.shape[axis + 1..].iter().product::<usize>();
        let outer = input.shape[..axis].iter().product::<usize>();
        let slices = outer * inner;
        let func = self.runtime.nvrtc_function("topk", SOURCE, "topk_f32")?;
        let input_ptr = cuptr(input.data_ptr::<f32>() as *const c_void);
        let values_ptr = cuptr(outputs[0].data_ptr_mut::<f32>() as *const c_void);
        let indices_ptr = cuptr(outputs[1].data_ptr_mut::<i64>() as *const c_void);
        let slices = slices as u64;
        let width = width as u64;
        let inner = inner as u64;
        let k = k as u64;
        let largest = i32::from(self.largest);
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&input_ptr)
            .arg(&values_ptr)
            .arg(&indices_ptr)
            .arg(&slices)
            .arg(&width)
            .arg(&inner)
            .arg(&k)
            .arg(&largest);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    (slices.div_ceil(BLOCK as u64).min(65_535).max(1) as u32),
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch TopK", e))?;
        self.runtime.synchronize()
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn cuda_graph_compatible(&self) -> bool {
        false
    }
}
