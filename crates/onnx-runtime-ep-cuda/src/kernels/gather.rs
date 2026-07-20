//! `Gather`: axis-parametric indexed copies via an NVRTC kernel.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
pub const GATHER_CAPTURE_ERROR_INDEX: u32 = 64;
const GATHER_SOURCE: &str = r#"
extern "C" __global__ void gather_bytes(
    const unsigned char* data, const void* indices, unsigned char* output,
    int output_bytes, int elem_bytes, int axis_dim, int num_indices, int inner,
    int index_is_i64, unsigned int* capture_error) {
  for (int byte = blockIdx.x * blockDim.x + threadIdx.x; byte < output_bytes;
       byte += gridDim.x * blockDim.x) {
    int element = byte / elem_bytes;
    int within = byte - element * elem_bytes;
    int inner_index = element % inner;
    int selected = (element / inner) % num_indices;
    int outer = element / (inner * num_indices);
    long long index = index_is_i64
        ? ((const long long*)indices)[selected]
        : (long long)((const int*)indices)[selected];
    if (index < 0) index += axis_dim;
    if (index < 0 || index >= axis_dim) {
      if (capture_error) atomicOr(capture_error, 64u);
      continue;
    }
    output[byte] = data[((outer * axis_dim + index) * inner + inner_index) * elem_bytes + within];
  }
}
"#;

pub struct GatherFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for GatherFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(GatherKernel {
            runtime: self.runtime.clone(),
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(0),
            last_call_capture_safe: AtomicBool::new(false),
        }))
    }
}

#[derive(Debug)]
pub struct GatherKernel {
    runtime: Arc<CudaRuntime>,
    axis: i64,
    last_call_capture_safe: AtomicBool,
}

fn checked_product(dims: &[usize], what: &str) -> Result<usize> {
    dims.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .ok_or_else(|| EpError::KernelFailed(format!("cuda_ep Gather: {what} overflows usize")))
    })
}

impl Kernel for GatherKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Gather: expected 2 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let data = &inputs[0];
        let indices = &inputs[1];
        let output = &mut outputs[0];
        if !data.is_contiguous() || !indices.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented("Gather with non-contiguous input/output"));
        }
        if output.dtype != data.dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep Gather: output dtype must match data dtype".into(),
            ));
        }
        let rank = data.shape.len();
        if rank == 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep Gather: data must have rank >= 1".into(),
            ));
        }
        let axis = if self.axis < 0 {
            self.axis + rank as i64
        } else {
            self.axis
        };
        if !(0..rank as i64).contains(&axis) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Gather: axis {} out of range for rank {rank}",
                self.axis
            )));
        }
        let axis = axis as usize;
        if !matches!(indices.dtype, DataType::Int32 | DataType::Int64) {
            return Err(not_implemented("Gather indices must be Int32 or Int64"));
        }
        let elem_bytes = data.dtype.byte_size();
        if elem_bytes == 0 {
            return Err(not_implemented(
                "Gather does not support packed or variable-width data dtype",
            ));
        }
        let outer = checked_product(&data.shape[..axis], "outer size")?;
        let inner = checked_product(&data.shape[axis + 1..], "inner size")?;
        let axis_dim = data.shape[axis];
        let num_indices = indices.numel();
        let expected_shape: Vec<_> = data.shape[..axis]
            .iter()
            .chain(indices.shape)
            .chain(&data.shape[axis + 1..])
            .copied()
            .collect();
        if output.shape != expected_shape.as_slice() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Gather: output shape {:?}, expected {:?}",
                output.shape, expected_shape
            )));
        }
        let expected_elements = outer
            .checked_mul(num_indices)
            .and_then(|n| n.checked_mul(inner))
            .ok_or_else(|| {
                EpError::KernelFailed("cuda_ep Gather: output size overflows usize".into())
            })?;
        if output.numel() != expected_elements {
            return Err(EpError::KernelFailed(
                "cuda_ep Gather: output element count does not match input geometry".into(),
            ));
        }
        if axis_dim == 0 && num_indices != 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep Gather: indices cannot select from an empty axis".into(),
            ));
        }
        let capturing = self.runtime.is_capturing()?;
        if !capturing {
            // A device-side out-of-range index would be an out-of-bounds load.
            // Eager execution reports it synchronously; captured execution uses
            // the shared device error latch checked before token consumption.
            let index_bytes = indices.dtype.storage_bytes(num_indices);
            let mut host_indices = vec![0u8; index_bytes];
            if !host_indices.is_empty() {
                // SAFETY: `indices` is a live contiguous device tensor and the
                // host buffer is exactly its fixed-width storage size.
                unsafe {
                    self.runtime.dtoh(
                        &mut host_indices,
                        cuptr(indices.data_ptr::<u8>() as *const c_void),
                    )?
                };
            }
            for raw in host_indices.chunks_exact(indices.dtype.byte_size()) {
                let raw = match indices.dtype {
                    DataType::Int32 => i32::from_ne_bytes(raw.try_into().unwrap()) as i64,
                    DataType::Int64 => i64::from_ne_bytes(raw.try_into().unwrap()),
                    _ => unreachable!("validated above"),
                };
                let normalized = if raw < 0 { raw + axis_dim as i64 } else { raw };
                if normalized < 0 || normalized >= axis_dim as i64 {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep Gather: index {raw} out of range for axis dimension {axis_dim}"
                    )));
                }
            }
        }
        let output_bytes = output.dtype.storage_bytes(output.numel());
        if output_bytes == 0 {
            return Ok(());
        }
        for (value, name) in [
            (output_bytes, "output byte count"),
            (elem_bytes, "element byte count"),
            (axis_dim, "axis dimension"),
            (num_indices, "index count"),
            (inner, "inner size"),
        ] {
            i32::try_from(value).map_err(|_| {
                EpError::KernelFailed(format!("cuda_ep Gather: {name} exceeds i32"))
            })?;
        }
        let func = self
            .runtime
            .nvrtc_function("gather_bytes", GATHER_SOURCE, "gather_bytes")?;
        let output_bytes = output_bytes as i32;
        let elem_bytes = elem_bytes as i32;
        let axis_dim = axis_dim as i32;
        let num_indices = num_indices as i32;
        let inner = inner as i32;
        let index_is_i64 = i32::from(indices.dtype == DataType::Int64);
        let capture_error = if capturing {
            self.runtime.capture_error_ptr()
        } else {
            0
        };
        let data_ptr = cuptr(data.data_ptr::<u8>() as *const c_void);
        let indices_ptr = cuptr(indices.data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let grid = (output_bytes as u32).div_ceil(BLOCK).clamp(1, 65_535);
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&data_ptr)
            .arg(&indices_ptr)
            .arg(&output_ptr)
            .arg(&output_bytes)
            .arg(&elem_bytes)
            .arg(&axis_dim)
            .arg(&num_indices)
            .arg(&inner)
            .arg(&index_is_i64)
            .arg(&capture_error);
        // SAFETY: argument types and order match `gather_bytes`; all pointers refer
        // to live contiguous device allocations validated above.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch gather_bytes", e))?;
        self.last_call_capture_safe.store(true, Ordering::Relaxed);
        if capturing {
            Ok(())
        } else {
            self.runtime.synchronize()
        }
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        if self.last_call_capture_safe.load(Ordering::Relaxed) {
            onnx_runtime_ep_api::CaptureSupport::Supported
        } else {
            onnx_runtime_ep_api::CaptureSupport::unsupported(
                "requires a warmed fixed-shape int64-index Gather path with device-side bounds validation",
            )
        }
    }
}
