//! Device-resident greedy token-feedback writer for the native CUDA decode
//! chained-replay loop (the "device token loop").
//!
//! After the device argmax has selected the next token on-device
//! ([`super::device_argmax`]), this single-thread kernel stitches that token
//! straight into the persistent decode bindings — device-to-device, no host
//! sync — so the next captured graph replay can run back-to-back with the
//! previous one and the host leaves the per-step critical path:
//!
//! * write the selected token id (`result[0]`) as an `i64` into the persistent
//!   `input_ids` binding slot,
//! * write the next absolute position into the persistent `position_ids`
//!   binding slot,
//! * set the `1` in the persistent attention-mask binding at the next position
//!   (guarded so a chain that reaches physical capacity never writes out of
//!   bounds — that final slot is never consumed and the next chain re-primes on
//!   the host anyway),
//! * append the selected token to the host-drainable token-log at slot `step`,
//! * OR the shared capture-error word (`result[1]`) into the error accumulator
//!   so a captured-replay validation violation is still rejected before the
//!   token is consumed, checked once when the host drains the chain.
//!
//! Every write targets the *same* persistent bindings the captured graph
//! already reads, so the graph shape is unchanged and capture stays valid.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{DeviceBuffer, EpError, Result};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const SOURCE: &str = r#"
extern "C" __global__ void device_token_writer(
    const unsigned int* result,
    long long* input_ids,
    long long* position_ids,
    long long* attention_mask,
    unsigned int* token_log,
    unsigned int* error_accum,
    long long next_position,
    unsigned long long mask_len,
    unsigned int write_position,
    unsigned int step) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned int token = result[0];
    input_ids[0] = (long long)token;
    if (write_position) {
      position_ids[0] = next_position;
    }
    if (next_position >= 0 &&
        (unsigned long long)next_position < mask_len) {
      attention_mask[next_position] = 1;
    }
    token_log[step] = token;
    error_accum[0] |= result[1];
  }
}
"#;

/// Launch the single-thread device token-writer that folds the just-selected
/// greedy token into the persistent decode bindings for the next replay.
///
/// `result` is the device-argmax result buffer (`result[0]` = token id,
/// `result[1]` = latched capture-error word). `input_ids`, `position_ids` and
/// `attention_mask` are the persistent decode bindings the captured graph
/// reads. `scratch` holds `capacity` u32 token-log slots followed by one u32
/// capture-error accumulator word at index `capacity`; `step` (`< capacity`)
/// selects the token-log slot. `next_position` is the absolute position for the
/// next step and `mask_len` the physical mask width used to guard the mask
/// write at the capacity boundary.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    runtime: &CudaRuntime,
    result: &DeviceBuffer,
    input_ids: &DeviceBuffer,
    position_ids: &DeviceBuffer,
    attention_mask: &DeviceBuffer,
    scratch: &DeviceBuffer,
    capacity: usize,
    next_position: i64,
    mask_len: usize,
    write_position: bool,
    step: u32,
) -> Result<()> {
    const U32: usize = std::mem::size_of::<u32>();
    const I64: usize = std::mem::size_of::<i64>();
    if (step as usize) >= capacity {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep device token writer: step {step} out of range for token-log capacity {capacity}"
        )));
    }
    if result.len() < 2 * U32 {
        return Err(EpError::KernelFailed(
            "cuda_ep device token writer: result buffer smaller than the token/capture-error pair"
                .into(),
        ));
    }
    if input_ids.len() < I64 || position_ids.len() < I64 {
        return Err(EpError::KernelFailed(
            "cuda_ep device token writer: input_ids/position_ids binding smaller than one i64"
                .into(),
        ));
    }
    if scratch.len() < (capacity + 1) * U32 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep device token writer: scratch buffer has {} bytes, need {}",
            scratch.len(),
            (capacity + 1) * U32
        )));
    }
    let mask_bytes = mask_len.checked_mul(I64).ok_or_else(|| {
        EpError::KernelFailed("cuda_ep device token writer: mask byte size overflows".into())
    })?;
    if attention_mask.len() < mask_bytes {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep device token writer: attention-mask binding has {} bytes, need {mask_bytes}",
            attention_mask.len()
        )));
    }
    let device = result.device();
    if input_ids.device() != device
        || position_ids.device() != device
        || attention_mask.device() != device
        || scratch.device() != device
    {
        return Err(EpError::KernelFailed(
            "cuda_ep device token writer: bindings are on different devices".into(),
        ));
    }

    let function =
        runtime.nvrtc_function("native_device_token_writer", SOURCE, "device_token_writer")?;
    let result_ptr = cuptr(result.as_ptr());
    let input_ids_ptr = cuptr(input_ids.as_ptr());
    let position_ids_ptr = cuptr(position_ids.as_ptr());
    let mask_ptr = cuptr(attention_mask.as_ptr());
    let token_log_ptr = cuptr(scratch.as_ptr());
    // The capture-error accumulator is the u32 word just past the `capacity`
    // token-log slots.
    let error_accum_ptr = cuptr(
        unsafe { (scratch.as_ptr() as *const u8).add(capacity * U32) } as *const std::ffi::c_void,
    );
    let mask_len = mask_len as u64;
    let write_position_flag: u32 = if write_position { 1 } else { 0 };
    let mut builder = runtime.stream().launch_builder(&function);
    builder
        .arg(&result_ptr)
        .arg(&input_ids_ptr)
        .arg(&position_ids_ptr)
        .arg(&mask_ptr)
        .arg(&token_log_ptr)
        .arg(&error_accum_ptr)
        .arg(&next_position)
        .arg(&mask_len)
        .arg(&write_position_flag)
        .arg(&step);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| driver_err("launch native device token writer", error))
}
