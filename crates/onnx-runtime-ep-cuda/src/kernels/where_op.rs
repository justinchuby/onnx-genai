//! Dtype-agnostic, three-way broadcasting CUDA implementation of ONNX `Where`.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::elementwise::{broadcast_strides, u64_bytes};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const WHERE_SOURCE: &str = r#"
extern "C" __global__ void where_bytes(
    const unsigned char* condition, const unsigned char* x, const unsigned char* y,
    unsigned char* output, const unsigned long long* metadata, int rank,
    int elem_bytes, unsigned long long elements) {
  const unsigned long long* dims = metadata;
  const unsigned long long* c_strides = metadata + rank;
  const unsigned long long* x_strides = metadata + 2 * rank;
  const unsigned long long* y_strides = metadata + 3 * rank;
  for (unsigned long long out = blockIdx.x * blockDim.x + threadIdx.x; out < elements;
       out += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long rem = out, ci = 0, xi = 0, yi = 0;
    for (int axis = rank - 1; axis >= 0; --axis) {
      unsigned long long coord = rem % dims[axis];
      rem /= dims[axis];
      ci += coord * c_strides[axis];
      xi += coord * x_strides[axis];
      yi += coord * y_strides[axis];
    }
    const unsigned char* source = condition[ci] ? x + xi * elem_bytes : y + yi * elem_bytes;
    for (int byte = 0; byte < elem_bytes; ++byte)
      output[out * elem_bytes + byte] = source[byte];
  }
}
"#;

pub struct WhereFactory {
    pub runtime: Arc<CudaRuntime>,
}
impl KernelFactory for WhereFactory {
    fn create(&self, _: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(WhereKernel {
            runtime: self.runtime.clone(),
        }))
    }
}

#[derive(Debug)]
struct WhereKernel {
    runtime: Arc<CudaRuntime>,
}
impl Kernel for WhereKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 3 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Where: expected 3 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let (condition, x, y) = (&inputs[0], &inputs[1], &inputs[2]);
        let output = &mut outputs[0];
        if condition.dtype != DataType::Bool {
            return Err(EpError::KernelFailed(
                "cuda_ep Where: condition must be Bool".into(),
            ));
        }
        if x.dtype != y.dtype || x.dtype != output.dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep Where: branch/output dtypes must match".into(),
            ));
        }
        if inputs.iter().any(|v| !v.is_contiguous()) || !output.is_contiguous() {
            return Err(not_implemented("Where with non-contiguous input/output"));
        }
        let elem_bytes = x.dtype.byte_size();
        if elem_bytes == 0 {
            return Err(not_implemented("Where for packed or variable-width dtype"));
        }
        let xy = onnx_runtime_ir::broadcast_shapes(x.shape, y.shape).map_err(EpError::Ir)?;
        let expected =
            onnx_runtime_ir::broadcast_shapes(condition.shape, &xy).map_err(EpError::Ir)?;
        if output.shape != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Where: output shape {:?}, expected {expected:?}",
                output.shape
            )));
        }
        let elements = output.numel();
        if elements == 0 {
            return Ok(());
        }
        let mut metadata = expected.iter().map(|&v| v as u64).collect::<Vec<_>>();
        metadata.extend(broadcast_strides(condition.shape, &expected));
        metadata.extend(broadcast_strides(x.shape, &expected));
        metadata.extend(broadcast_strides(y.shape, &expected));
        if metadata.is_empty() {
            metadata.push(0);
        }
        let bytes = u64_bytes(&metadata);
        let func = self
            .runtime
            .nvrtc_function("where_op", WHERE_SOURCE, "where_bytes")?;
        let metadata_ptr = self.runtime.alloc_raw(bytes.len())?;
        if let Err(error) = unsafe { self.runtime.htod(bytes, metadata_ptr) } {
            let _ = unsafe { self.runtime.free_raw(metadata_ptr) };
            return Err(error);
        }
        let condition_ptr = cuptr(condition.data_ptr::<u8>() as *const c_void);
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(y.data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let rank = expected.len() as i32;
        let elem_bytes = elem_bytes as i32;
        let elements_u64 = elements as u64;
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&condition_ptr)
            .arg(&x_ptr)
            .arg(&y_ptr)
            .arg(&output_ptr)
            .arg(&metadata_ptr)
            .arg(&rank)
            .arg(&elem_bytes)
            .arg(&elements_u64);
        let launch = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    (elements as u64).div_ceil(BLOCK as u64).min(65_535).max(1) as u32,
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch where_bytes", e));
        let sync = launch.and_then(|_| self.runtime.synchronize());
        let free = unsafe { self.runtime.free_raw(metadata_ptr) };
        sync.and(free)
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn cuda_graph_compatible(&self) -> bool {
        false
    }
}
