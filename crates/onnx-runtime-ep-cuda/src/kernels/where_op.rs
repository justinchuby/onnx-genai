//! Dtype-agnostic, three-way broadcasting CUDA implementation of ONNX `Where`.
//!
//! The kernel persists its right-aligned broadcast stride/shape metadata in a
//! shared [`DeviceMetadataCache`] so a repeated call with unchanged operand shapes
//! reuses one immutable payload without device allocation, metadata upload,
//! retirement or stream synchronization. Capture-signature handling still
//! constructs Host shape vectors. Payload reuse lets `Where` fold into a captured
//! CUDA graph instead of forcing a seam.
//!
//! Capture is advertised after an exact dtype/shape signature has warmed. The
//! condition remains device-resident and may vary between replays; only the
//! broadcast geometry must remain fixed.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    DeviceGraphResource, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{DataType, Node};

use super::elementwise::{broadcast_strides, require_matching_capture_signature, u64_bytes};
use crate::device_metadata::{DeviceMetadataCache, WarmMetadata, publish_metadata_candidate};
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
            warm: Mutex::new(WarmMetadata::new(DeviceMetadataCache::new(
                self.runtime.clone(),
            ))),
        }))
    }
}

/// The stable identity of a `Where` launch: its three input shapes and the
/// broadcast output shape. Unchanged across steps ⇒ the cached device metadata
/// is reused and no host round-trip happens.
#[derive(Clone, Debug, PartialEq, Eq)]
struct WhereMetadataKey {
    condition_shape: Vec<usize>,
    x_shape: Vec<usize>,
    y_shape: Vec<usize>,
    out_shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WhereCaptureSignature {
    dtype: DataType,
    shapes: WhereMetadataKey,
}

#[derive(Debug)]
struct WhereKernel {
    runtime: Arc<CudaRuntime>,
    warm: Mutex<WarmMetadata<DeviceMetadataCache<WhereMetadataKey>, WhereCaptureSignature>>,
}

impl Kernel for WhereKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let mut candidate = self.warm.lock().map_err(|_| EpError::KernelFailed(
            "cuda_ep Where metadata/signature state is poisoned; rebuild the provider before warming again".into(),
        ))?.clone();

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

        let current_signature = Some(WhereCaptureSignature {
            dtype: x.dtype,
            shapes: WhereMetadataKey {
                condition_shape: condition.shape.to_vec(),
                x_shape: x.shape.to_vec(),
                y_shape: y.shape.to_vec(),
                out_shape: expected.clone(),
            },
        });
        require_matching_capture_signature(
            &self.runtime,
            "Where",
            candidate.signature.as_ref(),
            current_signature.as_ref(),
        )?;

        let elements = output.numel();
        if elements == 0 {
            candidate.signature = current_signature;
            return publish_metadata_candidate(&self.warm, candidate, "Where");
        }

        let func = self
            .runtime
            .nvrtc_function("where_op", WHERE_SOURCE, "where_bytes")?;
        let metadata_ptr = candidate.metadata.get_or_prepare(
            "Where broadcast metadata",
            |key| {
                key.condition_shape == condition.shape
                    && key.x_shape == x.shape
                    && key.y_shape == y.shape
                    && key.out_shape == output.shape
            },
            || {
                let mut values = expected.iter().map(|&d| d as u64).collect::<Vec<_>>();
                values.extend(broadcast_strides(condition.shape, &expected));
                values.extend(broadcast_strides(x.shape, &expected));
                values.extend(broadcast_strides(y.shape, &expected));
                if values.is_empty() {
                    values.push(0);
                }
                Ok((
                    WhereMetadataKey {
                        condition_shape: condition.shape.to_vec(),
                        x_shape: x.shape.to_vec(),
                        y_shape: y.shape.to_vec(),
                        out_shape: expected.clone(),
                    },
                    u64_bytes(&values).to_vec(),
                ))
            },
        )?;
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
        // SAFETY: pointer types match the byte-wise kernel; metadata holds four
        // rank-length u64 arrays and the broadcast strides keep every read in
        // bounds.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    (elements as u64).div_ceil(BLOCK as u64).clamp(1, 65_535) as u32,
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch where_bytes", e))?;
        candidate.signature = current_signature;
        publish_metadata_candidate(&self.warm, candidate, "Where")
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn device_graph_resources(&self) -> Vec<DeviceGraphResource> {
        self.warm
            .lock()
            .ok()
            .and_then(|state| state.metadata.device_graph_resource())
            .into_iter()
            .collect()
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        match self.warm.lock() {
            Ok(state) if state.signature.is_some() => {
                onnx_runtime_ep_api::CaptureSupport::Supported
            }
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "Where must warm its exact broadcast shape/dtype signature before capture",
            ),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "Where capture signature is unavailable because its state lock was poisoned",
            ),
        }
    }
}
