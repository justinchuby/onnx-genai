//! Standard `ai.onnx::RotaryEmbedding` (opset 23): rotary position embedding
//! (RoPE) applied to query/key token embeddings.
//!
//! Faithful port of the ONNX reference
//! (`onnx/reference/ops/op_rotary_embedding.py`). The op rotates pairs of
//! channels in the last (head) dimension by a position-dependent angle supplied
//! precomputed as `cos_cache` / `sin_cache`:
//!
//! ```text
//! real = cos·x1 - sin·x2
//! imag = sin·x1 + cos·x2
//! ```
//!
//! where `(x1, x2)` are either the two halves of the rotary sub-vector
//! (`interleaved=0`, the GPT-NeoX / rotate-half convention) or adjacent
//! even/odd channels (`interleaved=1`, the GPT-J convention).
//!
//! ## Inputs / attributes (per the spec)
//!
//! * `X` — 4D `(batch, num_heads, seq, head_size)` or 3D
//!   `(batch, seq, hidden)`. For the 3D form `num_heads` (attribute) must be
//!   set and `hidden = num_heads·head_size`.
//! * `cos_cache`, `sin_cache` — when `position_ids` is provided: 2D
//!   `(max_pos+1, rotary_dim/2)`, gathered by position. When `position_ids` is
//!   absent: 3D `(batch, seq, rotary_dim/2)`, indexed directly.
//! * `position_ids` (optional) — 2D `(batch, seq)` integer indices.
//! * `interleaved` (default 0), `num_heads` (default 0), `rotary_embedding_dim`
//!   (default 0 → full rotation over `head_size`).
//!
//! The same `cos`/`sin` row applies to every head at a given `(batch, seq)`.
//! Channels at or beyond `rotary_embedding_dim` pass through unrotated.

use std::borrow::Cow;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
pub const ROTARY_EMBEDDING_CAPTURE_ERROR_POSITION: u32 = 256;
const ROTARY_EMBEDDING_MODULE: &str = "rotary_embedding_bf16_v2";
const ROTARY_EMBEDDING_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void rotary_embedding_f32(
    const float* x, const float* cos_cache, const float* sin_cache,
    const long long* position_ids, float* y,
    unsigned long long batch, unsigned long long seq,
    unsigned long long heads, unsigned long long head_size,
    unsigned long long rotary_dim, unsigned long long cache_rows,
    int is_4d, int interleaved, int has_position_ids,
    unsigned long long elements, unsigned int* capture_error) {
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
       i += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long b, h, s, d;
    if (is_4d) {
      d = i % head_size;
      unsigned long long rem = i / head_size;
      s = rem % seq;
      rem /= seq;
      h = rem % heads;
      b = rem / heads;
    } else {
      d = i % head_size;
      unsigned long long rem = i / head_size;
      h = rem % heads;
      rem /= heads;
      s = rem % seq;
      b = rem / seq;
    }

    if (d >= rotary_dim) {
      y[i] = x[i];
      continue;
    }

    const unsigned long long half = rotary_dim / 2;
    const long long cache_row = has_position_ids
        ? position_ids[b * seq + s]
        : (long long)(b * seq + s);
    // Eager position_ids are host-validated. Captured execution uses this guard
    // and the runtime's sticky error latch instead of synchronizing with the host.
    if (cache_row < 0 || (unsigned long long)cache_row >= cache_rows) {
      if (capture_error) atomicOr(capture_error, 256u);
      y[i] = x[i];
      continue;
    }

    unsigned long long k, partner;
    if (interleaved) {
      k = d / 2;
      partner = d ^ 1ULL;
    } else if (d < half) {
      k = d;
      partner = d + half;
    } else {
      k = d - half;
      partner = d - half;
    }
    const float cos = cos_cache[(unsigned long long)cache_row * half + k];
    const float sin = sin_cache[(unsigned long long)cache_row * half + k];
    if (interleaved) {
      if ((d & 1ULL) == 0) {
        y[i] = __fsub_rn(__fmul_rn(cos, x[i]), __fmul_rn(sin, x[partner + i - d]));
      } else {
        y[i] = __fadd_rn(__fmul_rn(sin, x[partner + i - d]), __fmul_rn(cos, x[i]));
      }
    } else if (d < half) {
      y[i] = __fsub_rn(__fmul_rn(cos, x[i]), __fmul_rn(sin, x[partner + i - d]));
    } else {
      y[i] = __fadd_rn(__fmul_rn(sin, x[partner + i - d]), __fmul_rn(cos, x[i]));
    }
  }
}

extern "C" __global__ void rotary_embedding_f16(
    const __half* x, const __half* cos_cache, const __half* sin_cache,
    const long long* position_ids, __half* y,
    unsigned long long batch, unsigned long long seq,
    unsigned long long heads, unsigned long long head_size,
    unsigned long long rotary_dim, unsigned long long cache_rows,
    int is_4d, int interleaved, int has_position_ids,
    unsigned long long elements, unsigned int* capture_error) {
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
       i += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long b, h, s, d;
    if (is_4d) {
      d = i % head_size;
      unsigned long long rem = i / head_size;
      s = rem % seq;
      rem /= seq;
      h = rem % heads;
      b = rem / heads;
    } else {
      d = i % head_size;
      unsigned long long rem = i / head_size;
      h = rem % heads;
      rem /= heads;
      s = rem % seq;
      b = rem / seq;
    }

    if (d >= rotary_dim) {
      y[i] = x[i];
      continue;
    }

    const unsigned long long half = rotary_dim / 2;
    const long long cache_row = has_position_ids
        ? position_ids[b * seq + s]
        : (long long)(b * seq + s);
    if (cache_row < 0 || (unsigned long long)cache_row >= cache_rows) {
      if (capture_error) atomicOr(capture_error, 256u);
      y[i] = x[i];
      continue;
    }

    unsigned long long k, partner;
    if (interleaved) {
      k = d / 2;
      partner = d ^ 1ULL;
    } else if (d < half) {
      k = d;
      partner = d + half;
    } else {
      k = d - half;
      partner = d - half;
    }
    const float cos =
        __half2float(cos_cache[(unsigned long long)cache_row * half + k]);
    const float sin =
        __half2float(sin_cache[(unsigned long long)cache_row * half + k]);
    const float current = __half2float(x[i]);
    const float paired = __half2float(x[partner + i - d]);
    float output;
    if (interleaved) {
      output = (d & 1ULL) == 0
          ? __fsub_rn(__fmul_rn(cos, current), __fmul_rn(sin, paired))
          : __fadd_rn(__fmul_rn(sin, paired), __fmul_rn(cos, current));
    } else if (d < half) {
      output = __fsub_rn(__fmul_rn(cos, current), __fmul_rn(sin, paired));
    } else {
      output = __fadd_rn(__fmul_rn(sin, paired), __fmul_rn(cos, current));
    }
    y[i] = __float2half_rn(output);
  }
}

extern "C" __global__ void rotary_embedding_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache, const long long* position_ids,
    __nv_bfloat16* y, unsigned long long batch, unsigned long long seq,
    unsigned long long heads, unsigned long long head_size,
    unsigned long long rotary_dim, unsigned long long cache_rows,
    int is_4d, int interleaved, int has_position_ids,
    unsigned long long elements, unsigned int* capture_error) {
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
       i += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long b, h, s, d;
    if (is_4d) {
      d = i % head_size;
      unsigned long long rem = i / head_size;
      s = rem % seq;
      rem /= seq;
      h = rem % heads;
      b = rem / heads;
    } else {
      d = i % head_size;
      unsigned long long rem = i / head_size;
      h = rem % heads;
      rem /= heads;
      s = rem % seq;
      b = rem / seq;
    }

    if (d >= rotary_dim) {
      y[i] = x[i];
      continue;
    }

    const unsigned long long half = rotary_dim / 2;
    const long long cache_row = has_position_ids
        ? position_ids[b * seq + s]
        : (long long)(b * seq + s);
    if (cache_row < 0 || (unsigned long long)cache_row >= cache_rows) {
      if (capture_error) atomicOr(capture_error, 256u);
      y[i] = x[i];
      continue;
    }

    unsigned long long k, partner;
    if (interleaved) {
      k = d / 2;
      partner = d ^ 1ULL;
    } else if (d < half) {
      k = d;
      partner = d + half;
    } else {
      k = d - half;
      partner = d - half;
    }
    const float cos =
        __bfloat162float(cos_cache[(unsigned long long)cache_row * half + k]);
    const float sin =
        __bfloat162float(sin_cache[(unsigned long long)cache_row * half + k]);
    const float current = __bfloat162float(x[i]);
    const float paired = __bfloat162float(x[partner + i - d]);
    float output;
    if (interleaved) {
      output = (d & 1ULL) == 0
          ? __fsub_rn(__fmul_rn(cos, current), __fmul_rn(sin, paired))
          : __fadd_rn(__fmul_rn(sin, paired), __fmul_rn(cos, current));
    } else if (d < half) {
      output = __fsub_rn(__fmul_rn(cos, current), __fmul_rn(sin, paired));
    } else {
      output = __fadd_rn(__fmul_rn(sin, paired), __fmul_rn(cos, current));
    }
    y[i] = __float2bfloat16_rn(output);
  }
}
"#;

/// Return the claim-time dtype denial for RoPE's floating-point inputs.
pub(crate) fn unsupported_reason(input_dtypes: &[DataType]) -> Option<Cow<'static, str>> {
    let &dtype = input_dtypes.first()?;
    if !matches!(
        dtype,
        DataType::Float16 | DataType::Float32 | DataType::BFloat16
    ) {
        let dtype = match dtype {
            DataType::BFloat16 => "bf16".into(),
            other => format!("{other:?}"),
        };
        return Some(Cow::Owned(format!(
            "RotaryEmbedding: dtype {dtype} not supported on CUDA (expected f16, bf16, or f32)"
        )));
    }
    if input_dtypes
        .iter()
        .take(3)
        .any(|&input_dtype| input_dtype != dtype)
    {
        return Some(Cow::Borrowed(
            "RotaryEmbedding: X, cos_cache, and sin_cache must have the same f16/bf16/f32 dtype",
        ));
    }
    None
}

fn check_arity(
    name: &str,
    inputs: &[TensorView],
    outputs: &[TensorMut],
    min: usize,
    max: usize,
    expected_outputs: usize,
) -> Result<()> {
    if !(min..=max).contains(&inputs.len()) || outputs.len() != expected_outputs {
        return Err(EpError::KernelFailed(format!(
            "{name}: invalid input/output arity"
        )));
    }
    Ok(())
}

fn rotary_entry(dtype: DataType) -> &'static str {
    match dtype {
        DataType::Float16 => "rotary_embedding_f16",
        DataType::BFloat16 => "rotary_embedding_bf16",
        DataType::Float32 => "rotary_embedding_f32",
        _ => unreachable!("RotaryEmbedding dtype must be validated before dispatch"),
    }
}

/// f32/f16/bf16 RotaryEmbedding kernel carrying the resolved attributes.
pub struct RotaryEmbeddingKernel {
    runtime: Arc<CudaRuntime>,
    interleaved: bool,
    num_heads: usize,
    rotary_embedding_dim: usize,
    warmed_signature: Mutex<Option<RotaryCaptureSignature>>,
    last_call_capture_safe: AtomicBool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RotaryCaptureSignature {
    dtype: DataType,
    x_shape: Vec<usize>,
    cos_shape: Vec<usize>,
    sin_shape: Vec<usize>,
    position_shape: Option<Vec<usize>>,
    output_shape: Vec<usize>,
}

/// Factory reading `interleaved` (0), `num_heads` (0), `rotary_embedding_dim` (0).
pub struct RotaryEmbeddingFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for RotaryEmbeddingFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let interleaved = match node.attr("interleaved") {
            None | Some(onnx_runtime_ir::Attribute::Int(0)) => false,
            Some(onnx_runtime_ir::Attribute::Int(1)) => true,
            Some(_) => {
                return Err(EpError::KernelFailed(
                    "RotaryEmbedding: interleaved must be 0 or 1".into(),
                ));
            }
        };
        let non_negative = |name: &str| -> Result<usize> {
            match node.attr(name) {
                None => Ok(0),
                Some(attribute) => attribute
                    .as_int()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| {
                        EpError::KernelFailed(format!(
                            "RotaryEmbedding: {name} must be a non-negative integer"
                        ))
                    }),
            }
        };
        let num_heads = non_negative("num_heads")?;
        let rotary_embedding_dim = non_negative("rotary_embedding_dim")?;
        Ok(Box::new(RotaryEmbeddingKernel {
            runtime: self.runtime.clone(),
            interleaved,
            num_heads,
            rotary_embedding_dim,
            warmed_signature: Mutex::new(None),
            last_call_capture_safe: AtomicBool::new(false),
        }))
    }
}

impl Kernel for RotaryEmbeddingKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.last_call_capture_safe.store(false, Ordering::Relaxed);
        check_arity("RotaryEmbedding", inputs, outputs, 3, 4, 1)?;
        let dtype = inputs[0].dtype;
        if !matches!(
            dtype,
            DataType::Float16 | DataType::Float32 | DataType::BFloat16
        ) || inputs[..3].iter().any(|input| input.dtype != dtype)
        {
            return Err(EpError::KernelFailed(
                "RotaryEmbedding: X/cos_cache/sin_cache must have the same f16, bf16, or f32 dtype"
                    .into(),
            ));
        }
        if inputs.iter().any(|input| !input.is_contiguous()) || !outputs[0].is_contiguous() {
            return Err(EpError::KernelFailed(
                "RotaryEmbedding: non-contiguous input/output".into(),
            ));
        }
        if outputs[0].dtype != dtype || outputs[0].shape != inputs[0].shape {
            return Err(EpError::KernelFailed(
                "RotaryEmbedding: output dtype/shape must match X".into(),
            ));
        }
        let has_position_ids = inputs.len() == 4;
        if has_position_ids && inputs[3].dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(
                "RotaryEmbedding: position_ids must be int64".into(),
            ));
        }

        let x_shape = inputs[0].shape;
        // Resolve batch/heads/seq/head_size in the canonical [B, S, H, D] view.
        let (batch, seq, heads, head_size, is_4d) = match x_shape.len() {
            4 => {
                // [batch, num_heads, seq, head_size]
                (x_shape[0], x_shape[2], x_shape[1], x_shape[3], true)
            }
            3 => {
                if self.num_heads == 0 {
                    return Err(EpError::KernelFailed(
                        "RotaryEmbedding: num_heads must be set for a 3D input".into(),
                    ));
                }
                let hidden = x_shape[2];
                if !hidden.is_multiple_of(self.num_heads) {
                    return Err(EpError::KernelFailed(format!(
                        "RotaryEmbedding: hidden {hidden} not divisible by num_heads {}",
                        self.num_heads
                    )));
                }
                (
                    x_shape[0],
                    x_shape[1],
                    self.num_heads,
                    hidden / self.num_heads,
                    false,
                )
            }
            r => {
                return Err(EpError::KernelFailed(format!(
                    "RotaryEmbedding: X must be rank 3 or 4, got rank {r}"
                )));
            }
        };

        let rotary_dim = if self.rotary_embedding_dim == 0 {
            head_size
        } else {
            self.rotary_embedding_dim
        };
        if rotary_dim > head_size || !rotary_dim.is_multiple_of(2) {
            return Err(EpError::KernelFailed(format!(
                "RotaryEmbedding: rotary_embedding_dim {rotary_dim} invalid for head_size {head_size}"
            )));
        }
        let half = rotary_dim / 2;

        // Zero-sized input: nothing to rotate. Emit an empty output rather than
        // underflowing on the `batch-1`/`seq-1` bounds computation below.
        if inputs[0].numel() == 0 {
            return Ok(());
        }

        // With `position_ids` present, validate its shape matches [batch, seq].
        if has_position_ids {
            let pos_shape = inputs[3].shape;
            let expected = batch * seq;
            if inputs[3].numel() != expected {
                return Err(EpError::KernelFailed(format!(
                    "RotaryEmbedding: position_ids has {} elements, expected {expected} ([batch={batch}, seq={seq}]); shape {pos_shape:?}",
                    inputs[3].numel()
                )));
            }
        }

        let cache_rows = inputs[1].numel() / half;
        if inputs[2].numel() / half != cache_rows
            || (!has_position_ids && cache_rows < batch.saturating_mul(seq))
        {
            return Err(EpError::KernelFailed(format!(
                "RotaryEmbedding: cos/sin cache extent invalid for row width {half}"
            )));
        }

        let position_ids_ptr = inputs
            .get(3)
            .map(|input| cuptr(input.data_ptr::<i64>().cast()))
            .unwrap_or(0);
        let capturing = self.runtime.is_capturing()?;
        if has_position_ids && !capturing {
            let mut host_positions = vec![0u8; batch * seq * std::mem::size_of::<i64>()];
            // SAFETY: position_ids is contiguous and the host buffer has its exact byte size.
            unsafe {
                self.runtime.dtoh(
                    &mut host_positions,
                    cuptr(inputs[3].data_ptr::<u8>() as *const c_void),
                )?
            };
            if host_positions.chunks_exact(8).any(|bytes| {
                let position = i64::from_ne_bytes(bytes.try_into().unwrap());
                position < 0 || position as usize >= cache_rows
            }) {
                return Err(EpError::KernelFailed(
                    "RotaryEmbedding: position_ids contain a value outside the cos/sin cache range"
                        .into(),
                ));
            }
        }

        let signature = RotaryCaptureSignature {
            dtype,
            x_shape: inputs[0].shape.to_vec(),
            cos_shape: inputs[1].shape.to_vec(),
            sin_shape: inputs[2].shape.to_vec(),
            position_shape: inputs.get(3).map(|input| input.shape.to_vec()),
            output_shape: outputs[0].shape.to_vec(),
        };
        let mut warmed_signature = self
            .warmed_signature
            .lock()
            .expect("cuda_ep RotaryEmbedding capture signature poisoned");
        if capturing && warmed_signature.as_ref() != Some(&signature) {
            return Err(EpError::KernelFailed(
                "RotaryEmbedding: dtype or shape changed during CUDA graph capture; warm the exact decode signature before capture"
                    .into(),
            ));
        }

        let entry = rotary_entry(dtype);
        let func =
            self.runtime
                .nvrtc_function(ROTARY_EMBEDDING_MODULE, ROTARY_EMBEDDING_SOURCE, entry)?;
        let x_ptr = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let cos_ptr = cuptr(inputs[1].data_ptr::<u8>() as *const c_void);
        let sin_ptr = cuptr(inputs[2].data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let batch = batch as u64;
        let seq = seq as u64;
        let heads = heads as u64;
        let head_size = head_size as u64;
        let rotary_dim = rotary_dim as u64;
        let cache_rows = cache_rows as u64;
        let is_4d = i32::from(is_4d);
        let interleaved = i32::from(self.interleaved);
        let has_position_ids = i32::from(has_position_ids);
        let elements = inputs[0].numel() as u64;
        let capture_error = if capturing {
            self.runtime.capture_error_ptr()
        } else {
            0
        };
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&x_ptr)
            .arg(&cos_ptr)
            .arg(&sin_ptr)
            .arg(&position_ids_ptr)
            .arg(&output_ptr)
            .arg(&batch)
            .arg(&seq)
            .arg(&heads)
            .arg(&head_size)
            .arg(&rotary_dim)
            .arg(&cache_rows)
            .arg(&is_4d)
            .arg(&interleaved)
            .arg(&has_position_ids)
            .arg(&elements)
            .arg(&capture_error);
        unsafe {
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
        .map_err(|error| driver_err(&format!("launch {entry}"), error))?;
        if !capturing {
            *warmed_signature = Some(signature);
        }
        self.last_call_capture_safe.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        if self.last_call_capture_safe.load(Ordering::Relaxed) {
            onnx_runtime_ep_api::CaptureSupport::Supported
        } else {
            onnx_runtime_ep_api::CaptureSupport::unsupported(
                "requires a warmed exact f16/bf16/f32 shape signature",
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider};
    use onnx_runtime_ir::compute_contiguous_strides;

    use super::*;
    use crate::CudaExecutionProvider;

    fn bf16_bytes(values: &[bf16]) -> &[u8] {
        // SAFETY: bf16 is plain two-byte data and the byte slice retains the input lifetime.
        unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }

    fn run_bf16_rope(ep: &CudaExecutionProvider, interleaved: bool) -> (Vec<bf16>, Vec<bf16>) {
        let shape = [2, 2, 3, 7];
        let strides = compute_contiguous_strides(&shape);
        let cache_shape = [shape[0] * shape[2], 3];
        let cache_strides = compute_contiguous_strides(&cache_shape);
        let input = (0..shape.iter().product())
            .map(|index| bf16::from_f32(((index * 17 % 97) as f32 - 48.0) / 23.0))
            .collect::<Vec<_>>();
        let cos = (0..cache_shape.iter().product())
            .map(|index| bf16::from_f32((index as f32 * 0.071).cos()))
            .collect::<Vec<_>>();
        let sin = (0..cache_shape.iter().product())
            .map(|index| bf16::from_f32((index as f32 * 0.071).sin()))
            .collect::<Vec<_>>();
        let bytes = std::mem::size_of_val(input.as_slice());
        let cache_bytes = std::mem::size_of_val(cos.as_slice());
        let input_buffer = ep.allocate(bytes, 256).unwrap();
        let cos_buffer = ep.allocate(cache_bytes, 256).unwrap();
        let sin_buffer = ep.allocate(cache_bytes, 256).unwrap();
        let mut output_buffer = ep.allocate(bytes, 256).unwrap();
        let runtime = ep.runtime();
        unsafe {
            runtime
                .htod(bf16_bytes(&input), cuptr(input_buffer.as_ptr()))
                .unwrap();
            runtime
                .htod(bf16_bytes(&cos), cuptr(cos_buffer.as_ptr()))
                .unwrap();
            runtime
                .htod(bf16_bytes(&sin), cuptr(sin_buffer.as_ptr()))
                .unwrap();
        }
        let inputs = [
            TensorView::new(
                DevicePtr(input_buffer.as_ptr()),
                DataType::BFloat16,
                &shape,
                &strides,
                ep.device_id(),
            ),
            TensorView::new(
                DevicePtr(cos_buffer.as_ptr()),
                DataType::BFloat16,
                &cache_shape,
                &cache_strides,
                ep.device_id(),
            ),
            TensorView::new(
                DevicePtr(sin_buffer.as_ptr()),
                DataType::BFloat16,
                &cache_shape,
                &cache_strides,
                ep.device_id(),
            ),
        ];
        let output = TensorMut::new(
            DevicePtrMut(output_buffer.as_mut_ptr()),
            DataType::BFloat16,
            &shape,
            &strides,
            ep.device_id(),
        );
        RotaryEmbeddingKernel {
            runtime: runtime.clone(),
            interleaved,
            num_heads: 0,
            rotary_embedding_dim: 6,
            warmed_signature: Mutex::new(None),
            last_call_capture_safe: AtomicBool::new(false),
        }
        .execute(&inputs, &mut [output])
        .unwrap();

        let mut output_bytes = vec![0u8; bytes];
        unsafe {
            runtime
                .dtoh(&mut output_bytes, cuptr(output_buffer.as_ptr()))
                .unwrap();
        }
        let output = output_bytes
            .chunks_exact(2)
            .map(|raw| bf16::from_bits(u16::from_ne_bytes(raw.try_into().unwrap())))
            .collect();
        ep.deallocate(input_buffer).unwrap();
        ep.deallocate(cos_buffer).unwrap();
        ep.deallocate(sin_buffer).unwrap();
        ep.deallocate(output_buffer).unwrap();

        let mut reference = input.clone();
        let head_size = shape[3];
        let rotary_dim = 6;
        let half = rotary_dim / 2;
        for b in 0..shape[0] {
            for h in 0..shape[1] {
                for s in 0..shape[2] {
                    let cache_row = b * shape[2] + s;
                    for d in 0..rotary_dim {
                        let index = ((b * shape[1] + h) * shape[2] + s) * head_size + d;
                        let (k, partner) = if interleaved {
                            (d / 2, d ^ 1)
                        } else if d < half {
                            (d, d + half)
                        } else {
                            (d - half, d - half)
                        };
                        let paired_index = index + partner - d;
                        let cos_value = cos[cache_row * half + k].to_f32();
                        let sin_value = sin[cache_row * half + k].to_f32();
                        let current = input[index].to_f32();
                        let paired = input[paired_index].to_f32();
                        let value = if (interleaved && d % 2 == 0) || (!interleaved && d < half) {
                            cos_value * current - sin_value * paired
                        } else {
                            sin_value * paired + cos_value * current
                        };
                        reference[index] = bf16::from_f32(value);
                    }
                }
            }
        }
        (output, reference)
    }

    #[test]
    fn rotary_dispatch_preserves_existing_entries_and_adds_bf16() {
        assert_eq!(rotary_entry(DataType::Float16), "rotary_embedding_f16");
        assert_eq!(rotary_entry(DataType::Float32), "rotary_embedding_f32");
        assert_eq!(rotary_entry(DataType::BFloat16), "rotary_embedding_bf16");
    }

    #[test]
    fn bf16_rope_matches_fp32_reference_for_both_layouts_and_odd_head_size() {
        let ep = match CudaExecutionProvider::new_default() {
            Ok(ep) => ep,
            Err(error) => {
                eprintln!("skip: no CUDA GPU/runtime available ({error})");
                return;
            }
        };
        for interleaved in [false, true] {
            let (output, reference) = run_bf16_rope(&ep, interleaved);
            let max_error = output
                .iter()
                .zip(&reference)
                .map(|(actual, expected)| (actual.to_f32() - expected.to_f32()).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= 0.015625,
                "interleaved={interleaved} max bf16 error {max_error}"
            );
            for index in (6..output.len()).step_by(7) {
                assert_eq!(
                    output[index].to_bits(),
                    reference[index].to_bits(),
                    "unrotated odd head tail changed at element {index}"
                );
            }
        }
    }
}
