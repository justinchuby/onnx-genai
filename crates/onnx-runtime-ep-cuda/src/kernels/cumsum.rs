//! Deterministic per-lane CUDA implementation of ONNX `CumSum`.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
extern "C" __global__ void cumsum_f32(
    const float* input, float* output, unsigned long long lanes,
    unsigned long long width, unsigned long long inner, int exclusive, int reverse) {
  for (unsigned long long lane = blockIdx.x * blockDim.x + threadIdx.x; lane < lanes;
       lane += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long outer = lane / inner, i = lane % inner;
    float total = 0.0f;
    for (unsigned long long n = 0; n < width; ++n) {
      unsigned long long d = reverse ? width - 1 - n : n;
      unsigned long long offset = (outer * width + d) * inner + i;
      float value = input[offset];
      if (exclusive) { output[offset] = total; total += value; }
      else { total += value; output[offset] = total; }
    }
  }
}
extern "C" __global__ void cumsum_i64(
    const long long* input, long long* output, unsigned long long lanes,
    unsigned long long width, unsigned long long inner, int exclusive, int reverse) {
  for (unsigned long long lane = blockIdx.x * blockDim.x + threadIdx.x; lane < lanes;
       lane += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long outer = lane / inner, i = lane % inner;
    unsigned long long total = 0;
    for (unsigned long long n = 0; n < width; ++n) {
      unsigned long long d = reverse ? width - 1 - n : n;
      unsigned long long offset = (outer * width + d) * inner + i;
      unsigned long long value = (unsigned long long)input[offset];
      if (exclusive) { output[offset] = (long long)total; total += value; }
      else { total += value; output[offset] = (long long)total; }
    }
  }
}
"#;

pub struct CumSumFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for CumSumFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(CumSumKernel {
            runtime: self.runtime.clone(),
            exclusive: node.attr("exclusive").and_then(|a| a.as_int()).unwrap_or(0) != 0,
            reverse: node.attr("reverse").and_then(|a| a.as_int()).unwrap_or(0) != 0,
        }))
    }
}

struct CumSumKernel {
    runtime: Arc<CudaRuntime>,
    exclusive: bool,
    reverse: bool,
}

impl Kernel for CumSumKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep CumSum: expected 2 inputs and 1 output".into(),
            ));
        }
        let input = &inputs[0];
        let axis_input = &inputs[1];
        let output = &mut outputs[0];
        if !input.is_contiguous() || !axis_input.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented("CumSum with non-contiguous tensors"));
        }
        if output.dtype != input.dtype || output.shape != input.shape {
            return Err(EpError::KernelFailed(
                "cuda_ep CumSum: output must match input shape and dtype".into(),
            ));
        }
        if !matches!(input.dtype, DataType::Float32 | DataType::Int64) {
            return Err(not_implemented(format!(
                "CumSum supports Float32 and Int64, got {:?}",
                input.dtype
            )));
        }
        if axis_input.dtype != DataType::Int64 || axis_input.numel() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep CumSum: axis must be an Int64 scalar".into(),
            ));
        }
        let mut bytes = [0_u8; 8];
        unsafe {
            self.runtime.dtoh(
                &mut bytes,
                cuptr(axis_input.data_ptr::<i64>() as *const c_void),
            )?
        };
        let raw = i64::from_ne_bytes(bytes);
        let rank = input.shape.len();
        let normalized = if raw < 0 { raw + rank as i64 } else { raw };
        if normalized < 0 || normalized as usize >= rank {
            return Err(EpError::KernelFailed(
                "cuda_ep CumSum: axis out of range".into(),
            ));
        }
        if output.numel() == 0 {
            return Ok(());
        }
        let axis = normalized as usize;
        let inner = input.shape[axis + 1..].iter().product::<usize>();
        let width = input.shape[axis];
        let outer = input.shape[..axis].iter().product::<usize>();
        let lanes = outer * inner;
        let entry = if input.dtype == DataType::Float32 {
            "cumsum_f32"
        } else {
            "cumsum_i64"
        };
        let func = self.runtime.nvrtc_function("cumsum", SOURCE, entry)?;
        let input_ptr = cuptr(input.data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let lanes = lanes as u64;
        let width = width as u64;
        let inner = inner as u64;
        let exclusive = i32::from(self.exclusive);
        let reverse = i32::from(self.reverse);
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&input_ptr)
            .arg(&output_ptr)
            .arg(&lanes)
            .arg(&width)
            .arg(&inner)
            .arg(&exclusive)
            .arg(&reverse);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    (lanes.div_ceil(BLOCK as u64).min(65_535).max(1) as u32),
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch CumSum", e))?;
        self.runtime.synchronize()
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn cuda_graph_compatible(&self) -> bool {
        false
    }
}
