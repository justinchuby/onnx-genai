//! CUDA `TensorScatter` (ai.onnx opset 24): the standardized KV-cache update.
//!
//! Formulated as a single pass over the **output** rather than "copy the cache,
//! then scatter the update into it". For each present-cache element we invert
//! the write mapping to ask which update row, if any, lands there:
//!
//! ```text
//! linear:    s = seq - write_indices[batch]
//! circular:  s = (seq - write_indices[batch]) mod max_sequence_length
//! written    <=> 0 <= s < sequence_length
//! ```
//!
//! `sequence_length <= max_sequence_length` makes that inverse unique, so no
//! two update rows contend for one slot. One kernel, no separate device copy,
//! no write ordering hazard, and capture-safe. It is also dtype-agnostic — the
//! op moves bytes rather than computing — so a single byte-wise entry serves
//! every element type instead of a per-dtype macro expansion.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node, compute_contiguous_strides};

use super::movement::PersistentMetadata;
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
/// Bit raised on the shared capture-error word when a `write_indices` entry is
/// negative. The slot then keeps its past-cache contents rather than writing
/// somewhere arbitrary.
pub const TENSOR_SCATTER_CAPTURE_ERROR_INDEX: u32 = 16_384;

const SOURCE: &str = r#"
extern "C" __global__ void tensor_scatter(
    const unsigned char* past_cache, const unsigned char* update,
    const long long* write_indices, unsigned char* present_cache,
    const unsigned long long* meta, int rank, int axis, int elem_bytes,
    unsigned long long elements, unsigned long long max_sequence_length,
    unsigned long long sequence_length, int circular,
    unsigned int* capture_error) {
  const unsigned long long* cache_strides = meta;
  const unsigned long long* update_strides = meta + rank;
  for (unsigned long long linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < elements; linear += (unsigned long long)gridDim.x * blockDim.x) {
    // The cache is dense and the output shares its layout, so `linear` is
    // already the past-cache offset; only the update offset needs building.
    unsigned long long rem = linear;
    unsigned long long batch = 0, sequence = 0, update_offset = 0;
    for (int d = 0; d < rank; ++d) {
      unsigned long long coordinate = rem / cache_strides[d];
      rem %= cache_strides[d];
      if (d == 0) batch = coordinate;
      if (d == axis) sequence = coordinate;
      else update_offset += coordinate * update_strides[d];
    }

    long long write_index = write_indices ? write_indices[batch] : 0;
    bool written = false;
    unsigned long long source_row = 0;
    if (write_index < 0) {
      if (capture_error) atomicOr(capture_error, 16384u);
    } else {
      unsigned long long start = (unsigned long long)write_index;
      if (circular) {
        // Work modulo the capacity so the subtraction cannot underflow.
        unsigned long long shifted = sequence + max_sequence_length
                                   - (start % max_sequence_length);
        source_row = shifted % max_sequence_length;
        written = source_row < sequence_length;
      } else if (sequence >= start) {
        source_row = sequence - start;
        written = source_row < sequence_length;
      }
    }

    if (written) {
      const unsigned long long source =
          update_offset + source_row * update_strides[axis];
      for (int byte = 0; byte < elem_bytes; ++byte)
        present_cache[linear * elem_bytes + byte] =
            update[source * elem_bytes + byte];
    } else {
      for (int byte = 0; byte < elem_bytes; ++byte)
        present_cache[linear * elem_bytes + byte] =
            past_cache[linear * elem_bytes + byte];
    }
  }
}
"#;

#[derive(Clone, PartialEq, Eq)]
struct TensorScatterCaptureSignature {
    dtype: DataType,
    cache_shape: Vec<usize>,
    update_shape: Vec<usize>,
    has_write_indices: bool,
}

pub struct TensorScatterFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for TensorScatterFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let circular = match node.attr("mode").and_then(Attribute::as_str) {
            None | Some("linear") => false,
            Some("circular") => true,
            Some(value) => {
                return Err(not_implemented(format!(
                    "cuda_ep TensorScatter: mode {value:?} (expected \"linear\" or \"circular\")"
                )));
            }
        };
        Ok(Box::new(TensorScatterKernel {
            runtime: self.runtime.clone(),
            raw_axis: node.attr("axis").and_then(Attribute::as_int).unwrap_or(-2),
            circular,
            metadata: Mutex::new(PersistentMetadata::new(self.runtime.clone())),
            warmed_signature: Mutex::new(None),
        }))
    }
}

struct TensorScatterKernel {
    runtime: Arc<CudaRuntime>,
    raw_axis: i64,
    circular: bool,
    metadata: Mutex<PersistentMetadata>,
    warmed_signature: Mutex<Option<TensorScatterCaptureSignature>>,
}

impl Kernel for TensorScatterKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !(2..=3).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep TensorScatter: expected 2 or 3 inputs and 1 output".into(),
            ));
        }
        if inputs.iter().any(|view| !view.is_contiguous())
            || outputs.iter().any(|view| !view.is_contiguous())
        {
            return Err(not_implemented(
                "TensorScatter with non-contiguous tensors".to_string(),
            ));
        }
        let past = &inputs[0];
        let update = &inputs[1];
        let write_indices = inputs.get(2);
        let output = &mut outputs[0];

        if update.dtype != past.dtype || output.dtype != past.dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep TensorScatter: past_cache, update, and present_cache must share a dtype"
                    .into(),
            ));
        }
        if output.shape != past.shape {
            return Err(EpError::KernelFailed(
                "cuda_ep TensorScatter: present_cache shape must match past_cache".into(),
            ));
        }
        let rank = past.shape.len();
        if update.shape.len() != rank {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep TensorScatter: update rank {} must match past_cache rank {rank}",
                update.shape.len()
            )));
        }
        if rank < 2 {
            return Err(EpError::KernelFailed(
                "cuda_ep TensorScatter: past_cache must have rank at least 2".into(),
            ));
        }
        let axis = {
            let normalized = if self.raw_axis < 0 {
                self.raw_axis + rank as i64
            } else {
                self.raw_axis
            };
            if normalized <= 0 || normalized >= rank as i64 {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep TensorScatter: axis {} must select a dimension after the batch \
                     dimension for rank {rank}",
                    self.raw_axis
                )));
            }
            normalized as usize
        };
        for dimension in 0..rank {
            if dimension != axis && update.shape[dimension] != past.shape[dimension] {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep TensorScatter: update dimension {} at index {dimension} must match \
                     past_cache dimension {}",
                    update.shape[dimension], past.shape[dimension]
                )));
            }
        }
        let max_sequence_length = past.shape[axis];
        let sequence_length = update.shape[axis];
        if sequence_length > max_sequence_length {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep TensorScatter: update sequence length {sequence_length} exceeds cache \
                 capacity {max_sequence_length}"
            )));
        }
        if let Some(indices) = write_indices {
            if indices.dtype != DataType::Int64 {
                return Err(not_implemented(format!(
                    "TensorScatter supports Int64 write_indices, got {:?}",
                    indices.dtype
                )));
            }
            if indices.shape != [past.shape[0]] {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep TensorScatter: write_indices shape {:?} must be [{}]",
                    indices.shape, past.shape[0]
                )));
            }
        }

        let capturing = self.runtime.is_capturing()?;
        let signature = TensorScatterCaptureSignature {
            dtype: past.dtype,
            cache_shape: past.shape.to_vec(),
            update_shape: update.shape.to_vec(),
            has_write_indices: write_indices.is_some(),
        };
        let mut warmed_signature = self.warmed_signature.lock().map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep TensorScatter: capture signature lock was poisoned".into(),
            )
        })?;
        if capturing && warmed_signature.as_ref() != Some(&signature) {
            return Err(EpError::KernelFailed(
                "cuda_ep TensorScatter: shape or dtype changed during CUDA graph capture; warm \
                 the exact signature first"
                    .into(),
            ));
        }

        let elements = past.numel();
        if elements == 0 || sequence_length == 0 {
            // Nothing to write; the present cache is the past cache verbatim.
            if elements != 0 {
                unsafe {
                    self.runtime.dtod_async(
                        cuptr(past.data_ptr::<u8>() as *const c_void),
                        cuptr(output.data_ptr_mut::<u8>() as *const c_void),
                        past.dtype.storage_bytes(elements),
                    )?
                };
            }
            if !capturing {
                *warmed_signature = Some(signature);
            }
            return Ok(());
        }

        let mut metadata_values = compute_contiguous_strides(past.shape)
            .into_iter()
            .map(|value| value as u64)
            .collect::<Vec<_>>();
        metadata_values.extend(
            compute_contiguous_strides(update.shape)
                .into_iter()
                .map(|value| value as u64),
        );
        let metadata_ptr = self
            .metadata
            .lock()
            .map_err(|_| {
                EpError::KernelFailed("cuda_ep TensorScatter: metadata lock was poisoned".into())
            })?
            .prepare(&metadata_values, "TensorScatter")?;

        let function =
            self.runtime
                .nvrtc_function("tensor_scatter_v1", SOURCE, "tensor_scatter")?;
        let past_ptr = cuptr(past.data_ptr::<u8>() as *const c_void);
        let update_ptr = cuptr(update.data_ptr::<u8>() as *const c_void);
        let indices_ptr = write_indices
            .map(|view| cuptr(view.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let rank_i32 = i32::try_from(rank)
            .map_err(|_| EpError::KernelFailed("cuda_ep TensorScatter: rank exceeds i32".into()))?;
        let axis_i32 = i32::try_from(axis)
            .map_err(|_| EpError::KernelFailed("cuda_ep TensorScatter: axis exceeds i32".into()))?;
        let elem_bytes = i32::try_from(past.dtype.storage_bytes(1)).map_err(|_| {
            EpError::KernelFailed("cuda_ep TensorScatter: element size exceeds i32".into())
        })?;
        let elements_u64 = elements as u64;
        let max_sequence_length_u64 = max_sequence_length as u64;
        let sequence_length_u64 = sequence_length as u64;
        let circular_i32 = i32::from(self.circular);
        let capture_error = if capturing || self.runtime.eager_sync_deferred() {
            self.runtime.capture_error_ptr()
        } else {
            0
        };

        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&past_ptr)
            .arg(&update_ptr)
            .arg(&indices_ptr)
            .arg(&output_ptr)
            .arg(&metadata_ptr)
            .arg(&rank_i32)
            .arg(&axis_i32)
            .arg(&elem_bytes)
            .arg(&elements_u64)
            .arg(&max_sequence_length_u64)
            .arg(&sequence_length_u64)
            .arg(&circular_i32)
            .arg(&capture_error);
        let blocks = elements.div_ceil(BLOCK as usize).max(1);
        let blocks = u32::try_from(blocks.min(65_535)).unwrap_or(65_535);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch tensor_scatter", error))?;
        if !capturing {
            *warmed_signature = Some(signature);
        }
        Ok(())
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        match self.warmed_signature.lock() {
            Ok(signature) if signature.is_some() => onnx_runtime_ep_api::CaptureSupport::Supported,
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "TensorScatter must warm its exact shape/dtype signature before capture",
            ),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "TensorScatter capture signature lock was poisoned",
            ),
        }
    }
}
