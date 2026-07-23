//! Dtype-agnostic, three-way broadcasting CUDA implementation of ONNX `Where`.
//!
//! The kernel persists its right-aligned broadcast stride/shape metadata in a
//! [`WhereMetadataCache`] (mirroring the elementwise binary kernel's
//! `BroadcastMetadataCache`) so a repeated call with unchanged operand shapes
//! performs **no** per-step host allocation, upload, free, or stream
//! synchronize. That is the prerequisite for a `Where` to fold into a captured
//! CUDA graph instead of forcing a seam.
//!
//! Capture is only advertised for the **loop-invariant scalar-predicate select**
//! shape — a single-element condition selecting between two equal-shaped
//! operands (`x.shape == y.shape == output.shape`). That is exactly the shape
//! produced when a capture-unsafe `If` whose branches are pure constant
//! selections is lowered to an on-device `Where` (see
//! `CudaOnDeviceConstantSelect` in `crate::optimizer`): the two branch caches
//! stay resident as constants and the scalar predicate is evaluated on-device
//! every step, so the whole decode collapses into one captured graph with no
//! per-step host cond readback. A general broadcasting / data-dependent `Where`
//! keeps the conservative `CaptureSupport::Unsupported` disposition and still
//! runs correctly as an eager seam node.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg, sys::CUdeviceptr};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::elementwise::{broadcast_strides, require_matching_capture_signature, u64_bytes};
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
            metadata: Mutex::new(WhereMetadataCache::new(self.runtime.clone())),
            last_capture_safe_signature: Mutex::new(None),
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

/// A persistent device buffer holding the right-aligned broadcast stride/shape
/// metadata for a three-input `Where`. Reused across decode steps whenever the
/// operand shapes are unchanged, so a captured launch performs **no** per-step
/// host allocation, upload, free, or synchronize.
#[derive(Debug)]
struct WhereMetadataCache {
    runtime: Arc<CudaRuntime>,
    key: Option<WhereMetadataKey>,
    ptr: CUdeviceptr,
}

impl WhereMetadataCache {
    fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            runtime,
            key: None,
            ptr: 0,
        }
    }

    fn prepare(&mut self, key: &WhereMetadataKey) -> Result<CUdeviceptr> {
        if self.key.as_ref() == Some(key) {
            return Ok(self.ptr);
        }
        if self.runtime.is_capturing()? {
            return Err(EpError::KernelFailed(
                "cuda_ep Where: broadcast metadata shape changed during CUDA graph capture; warm the fixed decode shape before capture".into(),
            ));
        }
        if self.ptr != 0 {
            self.runtime.synchronize()?;
        }

        let mut metadata = key.out_shape.iter().map(|&d| d as u64).collect::<Vec<_>>();
        metadata.extend(broadcast_strides(&key.condition_shape, &key.out_shape));
        metadata.extend(broadcast_strides(&key.x_shape, &key.out_shape));
        metadata.extend(broadcast_strides(&key.y_shape, &key.out_shape));
        if metadata.is_empty() {
            metadata.push(0);
        }
        let metadata_bytes = u64_bytes(&metadata);
        let ptr = self.runtime.alloc_raw(metadata_bytes.len())?;
        // SAFETY: allocation exactly covers the metadata byte slice.
        if let Err(error) = unsafe { self.runtime.htod(metadata_bytes, ptr) } {
            // SAFETY: `ptr` is still owned by this cache and no launch used it.
            let _ = unsafe { self.runtime.free_raw(ptr) };
            return Err(error);
        }
        if self.ptr != 0 {
            // SAFETY: synchronization completed all prior launches using the old
            // pointer, which remains exclusively owned by this cache.
            if let Err(error) = unsafe { self.runtime.free_raw(self.ptr) } {
                // SAFETY: the replacement has not escaped or been launched.
                let _ = unsafe { self.runtime.free_raw(ptr) };
                return Err(error);
            }
        }
        self.key = Some(key.clone());
        self.ptr = ptr;
        Ok(ptr)
    }
}

impl Drop for WhereMetadataCache {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: the live pointer was allocated by this runtime and remains
            // exclusively owned by this cache.
            let _ = unsafe { self.runtime.free_raw(self.ptr) };
            self.ptr = 0;
        }
    }
}

#[derive(Debug)]
struct WhereKernel {
    runtime: Arc<CudaRuntime>,
    metadata: Mutex<WhereMetadataCache>,
    last_capture_safe_signature: Mutex<Option<WhereCaptureSignature>>,
}

/// Whether a `Where` launch is the capture-safe loop-invariant scalar-predicate
/// select: a single-element condition choosing between two operands whose shapes
/// already equal the output (a pure select, no data-operand broadcast). Its
/// metadata is invariant across decode steps, so it can fold into a captured
/// graph. Any other `Where` (data-dependent condition, broadcasting operands)
/// stays an eager seam.
fn is_invariant_scalar_select(
    condition: &TensorView,
    x: &TensorView,
    y: &TensorView,
    out_shape: &[usize],
) -> bool {
    condition.numel() == 1 && x.shape == out_shape && y.shape == out_shape
}

impl Kernel for WhereKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let mut last_signature = self.last_capture_safe_signature.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep Where capture signature lock was poisoned".into())
        })?;
        let warmed_signature = last_signature.take();

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

        // Only the invariant scalar-select shape may enter capture. Recording a
        // signature for it lets `capture_support` advertise `Supported`; the
        // signature guard below rejects any drift between the warmed shape and
        // the shape seen during capture, exactly as the binary kernel does.
        let current_signature =
            is_invariant_scalar_select(condition, x, y, &expected).then(|| WhereCaptureSignature {
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
            warmed_signature.as_ref(),
            current_signature.as_ref(),
        )?;

        let elements = output.numel();
        if elements == 0 {
            *last_signature = current_signature;
            return Ok(());
        }

        let key = WhereMetadataKey {
            condition_shape: condition.shape.to_vec(),
            x_shape: x.shape.to_vec(),
            y_shape: y.shape.to_vec(),
            out_shape: expected.clone(),
        };
        let func = self
            .runtime
            .nvrtc_function("where_op", WHERE_SOURCE, "where_bytes")?;
        let metadata_ptr = {
            let mut metadata = self.metadata.lock().map_err(|_| {
                EpError::KernelFailed("cuda_ep Where metadata lock was poisoned".into())
            })?;
            metadata.prepare(&key)?
        };
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
        *last_signature = current_signature;
        Ok(())
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        // Only the invariant scalar-select signature recorded by the most recent
        // successful call may enter capture; every other `Where` stays an eager
        // seam (the general broadcasting path allocates per-call metadata).
        match self.last_capture_safe_signature.lock() {
            Ok(signature) if signature.is_some() => onnx_runtime_ep_api::CaptureSupport::Supported,
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "Where is capture-safe only as an invariant scalar-predicate select over \
                 equal-shaped operands; this launch broadcasts or has a non-scalar condition",
            ),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "Where capture signature is unavailable because its state lock was poisoned",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::is_invariant_scalar_select;
    use onnx_runtime_ep_api::{DevicePtr, DeviceId, TensorView};
    use onnx_runtime_ir::DataType;

    fn view<'a>(shape: &'a [usize], strides: &'a [i64], dtype: DataType) -> TensorView<'a> {
        TensorView::new(
            DevicePtr(std::ptr::null()),
            dtype,
            shape,
            strides,
            DeviceId::cpu(),
        )
    }

    // Capture is safe only when the predicate is a single scalar and both branch
    // operands already match the output shape (no broadcast), mirroring the
    // loop-invariant on-device LongRoPE cos/sin select the optimizer lowers to.
    #[test]
    fn scalar_predicate_equal_shape_select_is_capture_safe() {
        let cond = view(&[1], &[1], DataType::Bool);
        let x = view(&[131072, 48], &[48, 1], DataType::Float16);
        let y = view(&[131072, 48], &[48, 1], DataType::Float16);
        assert!(is_invariant_scalar_select(&cond, &x, &y, &[131072, 48]));
    }

    #[test]
    fn non_scalar_condition_is_not_capture_safe() {
        let cond = view(&[131072, 48], &[48, 1], DataType::Bool);
        let x = view(&[131072, 48], &[48, 1], DataType::Float16);
        let y = view(&[131072, 48], &[48, 1], DataType::Float16);
        assert!(!is_invariant_scalar_select(&cond, &x, &y, &[131072, 48]));
    }

    #[test]
    fn broadcasting_branch_is_not_capture_safe() {
        let cond = view(&[1], &[1], DataType::Bool);
        // y broadcasts along axis 0 -> shape differs from output, not capture-safe.
        let x = view(&[131072, 48], &[48, 1], DataType::Float16);
        let y = view(&[1, 48], &[48, 1], DataType::Float16);
        assert!(!is_invariant_scalar_select(&cond, &x, &y, &[131072, 48]));
    }
}
