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
const ROTARY_EMBEDDING_MODULE: &str = "rotary_embedding_f16_v1";
const ROTARY_EMBEDDING_SOURCE: &str = r#"
#include <cuda_fp16.h>

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
"#;

/// Return the claim-time dtype denial for RoPE's floating-point inputs.
pub(crate) fn unsupported_reason(input_dtypes: &[DataType]) -> Option<Cow<'static, str>> {
    let Some(&dtype) = input_dtypes.first() else {
        return None;
    };
    if !matches!(dtype, DataType::Float16 | DataType::Float32) {
        let dtype = match dtype {
            DataType::BFloat16 => "bf16".into(),
            other => format!("{other:?}"),
        };
        return Some(Cow::Owned(format!(
            "RotaryEmbedding: dtype {dtype} not supported on CUDA (expected f16 or f32)"
        )));
    }
    if input_dtypes
        .iter()
        .take(3)
        .any(|&input_dtype| input_dtype != dtype)
    {
        return Some(Cow::Borrowed(
            "RotaryEmbedding: X, cos_cache, and sin_cache must have the same f16/f32 dtype",
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
/// f32/f16 RotaryEmbedding kernel carrying the resolved attributes.
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
        if !matches!(dtype, DataType::Float16 | DataType::Float32)
            || inputs[..3].iter().any(|input| input.dtype != dtype)
        {
            return Err(EpError::KernelFailed(
                "RotaryEmbedding: X/cos_cache/sin_cache must have the same f16 or f32 dtype".into(),
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

        let entry = if dtype == DataType::Float16 {
            "rotary_embedding_f16"
        } else {
            "rotary_embedding_f32"
        };
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
                    elements.div_ceil(BLOCK as u64).min(65_535).max(1) as u32,
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
                "requires a warmed exact f16/f32 shape signature",
            )
        }
    }
}
