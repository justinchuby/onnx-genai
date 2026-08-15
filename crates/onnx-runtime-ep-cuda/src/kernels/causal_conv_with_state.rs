//! `com.microsoft::CausalConvWithState` — causal depthwise 1-D convolution with
//! a rolling state cache (Mamba / linear-attention "short conv"), the CUDA EP
//! port of the CPU kernel in `onnx-runtime-ep-cpu` (which is itself a faithful
//! port of ONNX Runtime's `causal_conv_with_state.cc`).
//!
//! ## Contract (`ndim = 1`, channels-first)
//!
//! * `x` — `[B, C, L]` activations.
//! * `weight` — `[C, 1, K]` depthwise filter (`group == C`).
//! * `bias` — optional `[C]`.
//! * `past_state` — optional `[B, C, K-1]`; treated as zeros when absent.
//!
//! Outputs:
//!
//! * `y` — `[B, C, L]`.
//! * `present_state` — optional `[B, C, K-1]`.
//!
//! Per `(b, c)` let `seq = concat(past_state[b, c, :], x[b, c, :])`. Then
//! `y[b, c, t] = activation(bias[c] + Σ_{k} weight[c, 0, k] · seq[t + k])` and
//! `present_state[b, c, :] = seq[-(K-1):]`. `activation` is `none` or
//! `silu`/`swish` (both `x·sigmoid(x)`; the op has no learnable β).
//!
//! Each device thread owns one `(b, c)` row and accumulates in `f32` in the same
//! `k = 0..K` order as the CPU kernel, so the pre-activation values are
//! bit-comparable; `f16`/`bf16` are widened to `f32` for the arithmetic and
//! narrowed (round-to-nearest) on store, matching the CPU EP's dtype policy.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;

/// One row `(b, c)` per thread. Templated per storage dtype via three named
/// entry points that only differ in their load/store helpers (accumulation is
/// always `f32`). `bias`/`state`/`present` pointers may be null (`has_*` flags).
const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float ccws_silu(float v) {
  return v / (1.0f + expf(-v));
}

#define CCWS_KERNEL(NAME, T, LOAD, STORE)                                       \
extern "C" __global__ void NAME(                                               \
    const T* __restrict__ x, const T* __restrict__ weight,                     \
    const T* __restrict__ bias, const T* __restrict__ state,                   \
    T* __restrict__ y, T* __restrict__ present,                                \
    long long batch, long long channels, long long length,                     \
    long long kernel_size, long long pad,                                       \
    int has_bias, int has_state, int has_present, int activation) {            \
  const long long rows = batch * channels;                                     \
  for (long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;        \
       row < rows; row += (long long)gridDim.x * blockDim.x) {                  \
    const long long c = row % channels;                                        \
    const long long x_base = row * length;                                     \
    const long long s_base = row * pad;                                        \
    const long long w_base = c * kernel_size;                                  \
    const float bias_c = has_bias ? LOAD(bias[c]) : 0.0f;                       \
    for (long long t = 0; t < length; ++t) {                                    \
      float acc = bias_c;                                                       \
      for (long long k = 0; k < kernel_size; ++k) {                             \
        const long long pos = t + k;                                           \
        float val;                                                              \
        if (pos < pad) {                                                        \
          val = has_state ? LOAD(state[s_base + pos]) : 0.0f;                    \
        } else {                                                                \
          val = LOAD(x[x_base + (pos - pad)]);                                  \
        }                                                                       \
        acc += LOAD(weight[w_base + k]) * val;                                  \
      }                                                                         \
      if (activation == 1) acc = ccws_silu(acc);                                \
      y[x_base + t] = STORE(acc);                                               \
    }                                                                           \
    if (has_present) {                                                          \
      for (long long p = 0; p < pad; ++p) {                                     \
        const long long pos = length + p;                                      \
        float val;                                                              \
        if (pos < pad) {                                                        \
          val = has_state ? LOAD(state[s_base + pos]) : 0.0f;                    \
        } else {                                                                \
          val = LOAD(x[x_base + (pos - pad)]);                                  \
        }                                                                       \
        present[s_base + p] = STORE(val);                                       \
      }                                                                         \
    }                                                                           \
  }                                                                             \
}

#define CCWS_ID(v) (v)
CCWS_KERNEL(causal_conv_with_state_f32, float, CCWS_ID, CCWS_ID)
CCWS_KERNEL(causal_conv_with_state_f16, __half, __half2float, __float2half_rn)
CCWS_KERNEL(causal_conv_with_state_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16_rn)
"#;

/// Post-convolution activation. `Swish` with the op's implicit `β = 1` equals
/// `Silu`, so both map to the same pass.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ConvActivation {
    None,
    Silu,
}

/// Claim-time gate: keep placement in sync with the runtime limits so a node is
/// never claimed only to fail while constructing or executing the kernel.
pub(crate) fn unsupported_reason(node: &Node, input_dtypes: &[DataType]) -> Option<String> {
    let ndim = node.attr("ndim").and_then(|a| a.as_int()).unwrap_or(1);
    if ndim != 1 {
        return Some(format!(
            "CausalConvWithState CUDA supports only ndim=1 (channels-first [B, C, L]), got ndim={ndim}"
        ));
    }
    match node.attr("activation").and_then(|a| a.as_str()) {
        None | Some("none") | Some("silu") | Some("swish") => {}
        Some(other) => {
            return Some(format!(
                "CausalConvWithState activation must be one of none, silu, swish; got {other:?}"
            ));
        }
    }
    if !(2..=4).contains(&input_dtypes.len()) {
        return Some(format!(
            "CausalConvWithState expects 2..=4 inputs (x, weight, [bias], [past_state]), got {}",
            input_dtypes.len()
        ));
    }
    for (index, name) in [(0, "x"), (1, "weight")] {
        if !matches!(
            input_dtypes[index],
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Some(format!(
                "CausalConvWithState input {index} ('{name}') dtype {:?} is unsupported (expected Float32, Float16, or BFloat16)",
                input_dtypes[index]
            ));
        }
    }
    for index in 2..input_dtypes.len() {
        if input_dtypes[index] != input_dtypes[0] {
            return Some(format!(
                "CausalConvWithState input {index} dtype {:?} must match x dtype {:?}",
                input_dtypes[index], input_dtypes[0]
            ));
        }
    }
    None
}

pub struct CausalConvWithStateFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for CausalConvWithStateFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let ndim = node.attr("ndim").and_then(|a| a.as_int()).unwrap_or(1);
        if ndim != 1 {
            return Err(not_implemented(format!(
                "CausalConvWithState CUDA supports only ndim=1, got ndim={ndim}"
            )));
        }
        let activation = match node.attr("activation").and_then(|a| a.as_str()) {
            None | Some("none") => ConvActivation::None,
            Some("silu") | Some("swish") => ConvActivation::Silu,
            Some(other) => {
                return Err(not_implemented(format!(
                    "CausalConvWithState activation must be one of none, silu, swish; got {other:?}"
                )));
            }
        };
        Ok(Box::new(CausalConvWithStateKernel {
            runtime: self.runtime.clone(),
            activation,
        }))
    }
}

#[derive(Debug)]
pub struct CausalConvWithStateKernel {
    runtime: Arc<CudaRuntime>,
    activation: ConvActivation,
}

impl std::fmt::Debug for ConvActivation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ConvActivation::None => "none",
            ConvActivation::Silu => "silu",
        })
    }
}

impl Kernel for CausalConvWithStateKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !(2..=4).contains(&inputs.len()) || !(1..=2).contains(&outputs.len()) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep CausalConvWithState: expected 2..=4 inputs and 1..=2 outputs, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        let weight = &inputs[1];
        let bias = inputs.get(2).filter(|t| !t.is_absent());
        let state = inputs.get(3).filter(|t| !t.is_absent());

        if x.shape.len() != 3 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep CausalConvWithState: x must be rank 3 [B, C, L], got {:?}",
                x.shape
            )));
        }
        let (batch, channels, length) = (x.shape[0], x.shape[1], x.shape[2]);
        if weight.shape.len() != 3 || weight.shape[0] != channels || weight.shape[1] != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep CausalConvWithState: weight must be depthwise [C, 1, K] with C={channels}, got {:?}",
                weight.shape
            )));
        }
        let kernel_size = weight.shape[2];
        if kernel_size == 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep CausalConvWithState: kernel size K must be >= 1".into(),
            ));
        }
        let pad = kernel_size - 1;

        let dtype = x.dtype;
        if !matches!(
            dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Err(not_implemented(format!(
                "CausalConvWithState dtype {dtype:?} (supported: Float32, Float16, BFloat16)"
            )));
        }
        for (name, tensor) in [
            ("weight", Some(weight)),
            ("bias", bias),
            ("past_state", state),
        ] {
            if let Some(tensor) = tensor
                && tensor.dtype != dtype
            {
                return Err(not_implemented(format!(
                    "CausalConvWithState requires {name} dtype {:?} to match x dtype {dtype:?}",
                    tensor.dtype
                )));
            }
        }
        if let Some(bias) = bias
            && bias.shape != [channels]
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep CausalConvWithState: bias must be [C={channels}], got {:?}",
                bias.shape
            )));
        }
        if let Some(state) = state
            && state.shape != [batch, channels, pad]
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep CausalConvWithState: past_state must be [B={batch}, C={channels}, K-1={pad}], got {:?}",
                state.shape
            )));
        }

        let y = &outputs[0];
        if y.dtype != dtype || y.shape != [batch, channels, length] {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep CausalConvWithState: y must be {dtype:?} [B, C, L]={:?}, got {:?} {:?}",
                [batch, channels, length],
                y.dtype,
                y.shape
            )));
        }
        let has_present = outputs.len() == 2;
        if has_present {
            let present = &outputs[1];
            if present.dtype != dtype || present.shape != [batch, channels, pad] {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep CausalConvWithState: present_state must be {dtype:?} [B, C, K-1]={:?}, got {:?} {:?}",
                    [batch, channels, pad],
                    present.dtype,
                    present.shape
                )));
            }
        }

        for (name, contiguous) in [
            ("x", x.is_contiguous()),
            ("weight", weight.is_contiguous()),
            ("bias", bias.is_none_or(TensorView::is_contiguous)),
            ("past_state", state.is_none_or(TensorView::is_contiguous)),
            ("y", outputs[0].is_contiguous()),
            (
                "present_state",
                outputs.get(1).is_none_or(|p| p.is_contiguous()),
            ),
        ] {
            if !contiguous {
                return Err(not_implemented(format!(
                    "CausalConvWithState requires contiguous {name}"
                )));
            }
        }

        let rows = batch * channels;
        if rows == 0 || length == 0 {
            return Ok(());
        }

        let stem = match dtype {
            DataType::Float32 => "causal_conv_with_state_f32",
            DataType::Float16 => "causal_conv_with_state_f16",
            DataType::BFloat16 => "causal_conv_with_state_bf16",
            _ => unreachable!(),
        };
        if dtype != DataType::Float32 {
            self.runtime
                .require_nvrtc_half_headers("CausalConvWithState")?;
        }
        let func = self
            .runtime
            .nvrtc_function("causal_conv_with_state_v1", SOURCE, stem)?;

        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let weight_ptr = cuptr(weight.data_ptr::<u8>() as *const c_void);
        let bias_ptr = bias
            .map(|t| cuptr(t.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let state_ptr = state
            .map(|t| cuptr(t.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let present_ptr = if has_present {
            cuptr(outputs[1].data_ptr_mut::<u8>() as *const c_void)
        } else {
            0
        };

        let batch_i64 = batch as i64;
        let channels_i64 = channels as i64;
        let length_i64 = length as i64;
        let kernel_size_i64 = kernel_size as i64;
        let pad_i64 = pad as i64;
        let has_bias = i32::from(bias.is_some());
        let has_state = i32::from(state.is_some());
        let has_present_flag = i32::from(has_present);
        let activation = i32::from(self.activation == ConvActivation::Silu);

        let grid = u32::try_from(rows.div_ceil(BLOCK as usize))
            .unwrap_or(u32::MAX)
            .clamp(1, 65_535);
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&x_ptr)
            .arg(&weight_ptr)
            .arg(&bias_ptr)
            .arg(&state_ptr)
            .arg(&y_ptr)
            .arg(&present_ptr)
            .arg(&batch_i64)
            .arg(&channels_i64)
            .arg(&length_i64)
            .arg(&kernel_size_i64)
            .arg(&pad_i64)
            .arg(&has_bias)
            .arg(&has_state)
            .arg(&has_present_flag)
            .arg(&activation);
        // SAFETY: argument types/order match the `causal_conv_with_state_*`
        // signature; every non-null pointer refers to a live contiguous device
        // allocation validated above. The kernel is register-only (no shared
        // memory, no per-call allocation, no host sync), so it is capture-safe.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| driver_err("launch CausalConvWithState", e))
    }
}
