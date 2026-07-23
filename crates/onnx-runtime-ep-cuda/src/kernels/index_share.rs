//! `pkg.nxrt::IndexShare` v1: device-resident selected-token attention for the
//! GLM-5.2 IndexShare / DeepSeek DSA building block.
//!
//! The frozen CPU reference in
//! `crates/onnx-runtime-ep-cpu/src/kernels/index_share.rs` is the authoritative
//! numerical oracle. This kernel reproduces its math **on the device**: the
//! `past ⧺ current` KV cache concatenation, the per-row selected-token gather,
//! the scaled dot-product scores, a numerically-stable fp32 softmax over the
//! selected keys, and the probability·value reduction all run in NVRTC kernels.
//! Query, key, value, bias, the present KV cache, and the output tensor stay
//! resident on the device; the only host round-trip is the small
//! `selected_indices` tensor, copied D2H so the ONNX-required deterministic
//! index validation (strictly-increasing order, trailing `-1` padding, range)
//! produces the same errors as the CPU oracle. This mirrors the
//! [`super::sparse_kv_gather`] precedent, where the candidate index table is the
//! only tensor validated host-side.
//!
//! ## Determinism / bit-parity
//!
//! Every reduction sums in the same fixed ascending order as the CPU reference:
//! each score's dot product accumulates over `head_size` in one thread, the
//! softmax max/exp/sum runs sequentially in the block's lead thread, and the
//! `probs·V` accumulation sums over the selected keys in ascending order in one
//! thread per output channel. `sqrt(scale)` is folded into each Q and K operand
//! (matching the reference's `(Q·√scale)·(K·√scale)`), so results are
//! byte-identical to the CPU oracle.
//!
//! ## Capture support
//!
//! Like [`super::sparse_kv_gather`], the host-side index validation copies the
//! `selected_indices` tensor D2H and synchronizes the stream, which is not legal
//! during CUDA-graph capture. Capture is therefore gated off with an explicit
//! reason; the bulk attention math is fully device-resident.
//!
//! ## Claim-time gating
//!
//! [`unsupported_reason`] (in the CUDA provider) delegates to the CPU oracle's
//! own `unsupported_reason`, so the two backends reject exactly the same
//! dtype/layout/arity/shape combinations at claim time rather than claiming a
//! node and falling back inside the kernel.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const OP: &str = "IndexShare";
const INPUT_NAMES: [&str; 7] = [
    "query",
    "key",
    "value",
    "past_key",
    "past_value",
    "selected_indices",
    "attention_bias",
];

/// Threads per block for the bulk-copy present-cache builder.
const BLOCK: u32 = 256;
/// Threads per block for `index_share_row` (one block services one output row).
const ROW_THREADS: u32 = 128;
const MODULE: &str = "index_share_f32_v1";
const SOURCE: &str = r#"
#define NEG_INF __int_as_float(0xff800000)

// Gather a K/V input plus an optional past cache into a contiguous
// [batch, kv_heads, total_seq, head_size] present buffer (past ++ current along
// the sequence axis). This is a pure copy, so the present outputs are
// bit-identical to the CPU reference's concatenation.
extern "C" __global__ void build_present(
    const float* past, const float* cur, float* out, int has_past,
    unsigned long long batch, unsigned long long heads,
    unsigned long long past_seq, unsigned long long cur_seq,
    unsigned long long total_seq, unsigned long long dim,
    unsigned long long elements) {
  for (unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
       idx < elements; idx += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long d = idx % dim;
    unsigned long long rem = idx / dim;
    unsigned long long t = rem % total_seq;
    rem /= total_seq;
    unsigned long long h = rem % heads;
    unsigned long long b = rem / heads;
    float val;
    if (has_past && t < past_seq) {
      val = past[((b * heads + h) * past_seq + t) * dim + d];
    } else {
      unsigned long long c = has_past ? (t - past_seq) : t;
      val = cur[((b * heads + h) * cur_seq + c) * dim + d];
    }
    out[idx] = val;
  }
}

// Additive attention bias for logical index (b, h, q, k), broadcasting a
// rank<=4 bias right-aligned against [b, h, q, k]. Mirrors the CPU reference's
// Bias::at exactly (size-1 axes broadcast; no -inf padding for a short last
// dim, which the claim gate already forbids).
__device__ __forceinline__ float bias_at(
    const float* bias, int rank,
    unsigned long long bd0, unsigned long long bd1,
    unsigned long long bd2, unsigned long long bd3,
    unsigned long long b, unsigned long long h,
    unsigned long long q, unsigned long long k) {
  unsigned long long logical[4] = {b, h, q, k};
  unsigned long long dims[4] = {bd0, bd1, bd2, bd3};
  unsigned long long off = 0;
  for (int axis = 4 - rank; axis < 4; ++axis) {
    unsigned long long dim = dims[axis];
    unsigned long long index = (dim == 1ULL) ? 0ULL : logical[axis];
    off = off * dim + index;
  }
  return bias[off];
}

__device__ __forceinline__ long long load_index(
    const void* indices, unsigned long long offset, int index_is_i64) {
  return index_is_i64
      ? ((const long long*)indices)[offset]
      : (long long)((const int*)indices)[offset];
}

// One block per (batch, q_head, query) output row. Gathers the selected keys,
// computes scaled QK scores (+ optional bias), a numerically-stable softmax
// over the valid selections, and the probability-weighted value sum.
extern "C" __global__ void index_share_row(
    const float* q, const float* present_k, const float* present_v,
    const void* indices, const float* bias, float* scores, float* y,
    unsigned long long batch, unsigned long long q_heads, unsigned long long kv_heads,
    unsigned long long q_seq, unsigned long long total_seq, unsigned long long head_size,
    unsigned long long index_heads, unsigned long long selected_width,
    unsigned long long group, float sqrt_scale,
    int index_is_i64, int has_bias, int bias_rank,
    unsigned long long bd0, unsigned long long bd1,
    unsigned long long bd2, unsigned long long bd3) {
  const unsigned long long row = blockIdx.x;
  const unsigned long long total_rows = batch * q_heads * q_seq;
  if (row >= total_rows) {
    return;
  }
  const unsigned long long qi = row % q_seq;
  unsigned long long rem = row / q_seq;
  const unsigned long long qh = rem % q_heads;
  const unsigned long long b = rem / q_heads;
  const unsigned long long kvh = qh / group;
  const unsigned long long ih = (index_heads == 1ULL) ? 0ULL : qh;
  const unsigned long long index_row =
      ((b * index_heads + ih) * q_seq + qi) * selected_width;
  const unsigned long long score_row = row * selected_width;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  // Valid count: entries before the trailing -1 padding. The host validation
  // guarantees strictly-increasing indices with only trailing -1 padding, so
  // counting non-(-1) entries reproduces the CPU take_while.
  __shared__ unsigned long long valid_sh;
  if (tid == 0) {
    unsigned long long valid = 0;
    for (unsigned long long s = 0; s < selected_width; ++s) {
      if (load_index(indices, index_row + s, index_is_i64) == -1) {
        break;
      }
      valid += 1;
    }
    valid_sh = valid;
  }
  __syncthreads();
  const unsigned long long valid = valid_sh;

  const unsigned long long qoff = ((b * q_heads + qh) * q_seq + qi) * head_size;

  // Stage 1: scaled QK score per selected key (sqrt(scale) folded into each
  // operand), plus optional additive bias.
  for (unsigned long long s = tid; s < valid; s += nthreads) {
    const unsigned long long key = (unsigned long long)load_index(indices, index_row + s, index_is_i64);
    const unsigned long long koff = ((b * kv_heads + kvh) * total_seq + key) * head_size;
    float acc = 0.0f;
    for (unsigned long long d = 0; d < head_size; ++d) {
      acc += (q[qoff + d] * sqrt_scale) * (present_k[koff + d] * sqrt_scale);
    }
    if (has_bias) {
      acc += bias_at(bias, bias_rank, bd0, bd1, bd2, bd3, b, qh, qi, key);
    }
    scores[score_row + s] = acc;
  }
  __syncthreads();

  // Stage 2: numerically-stable softmax over the valid scores. The lead thread
  // reduces in ascending order to match the CPU reference bit-for-bit.
  __shared__ float inv_sum_sh;
  __shared__ int all_masked_sh;
  if (tid == 0) {
    float m = NEG_INF;
    for (unsigned long long s = 0; s < valid; ++s) {
      m = fmaxf(m, scores[score_row + s]);
    }
    if (m == NEG_INF) {
      all_masked_sh = 1;
      inv_sum_sh = 0.0f;
    } else {
      all_masked_sh = 0;
      float sum = 0.0f;
      for (unsigned long long s = 0; s < valid; ++s) {
        const float e = expf(scores[score_row + s] - m);
        scores[score_row + s] = e;
        sum += e;
      }
      inv_sum_sh = 1.0f / sum;
    }
  }
  __syncthreads();

  const unsigned long long ybase = ((b * q_heads + qh) * q_seq + qi) * head_size;
  if (all_masked_sh) {
    for (unsigned long long d = tid; d < head_size; d += nthreads) {
      y[ybase + d] = 0.0f;
    }
    return;
  }

  // Normalize probabilities in place (prob = exp * inv_sum), matching the CPU
  // reference which stores the normalized weights before the value reduction.
  const float inv = inv_sum_sh;
  for (unsigned long long s = tid; s < valid; s += nthreads) {
    scores[score_row + s] *= inv;
  }
  __syncthreads();

  // Stage 3: Y = sum_s prob[s] * V[key_s]. Each thread owns whole output
  // channels and sums over selected keys in ascending order.
  for (unsigned long long d = tid; d < head_size; d += nthreads) {
    float acc = 0.0f;
    for (unsigned long long s = 0; s < valid; ++s) {
      const unsigned long long key = (unsigned long long)load_index(indices, index_row + s, index_is_i64);
      const unsigned long long voff = ((b * kv_heads + kvh) * total_seq + key) * head_size + d;
      acc += scores[score_row + s] * present_v[voff];
    }
    y[ybase + d] = acc;
  }
}
"#;

/// Resolved geometry shared between the CPU reference and this kernel.
#[derive(Clone, Copy)]
struct Dims {
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    q_seq: usize,
    current_seq: usize,
    past_seq: usize,
    total_seq: usize,
    head_size: usize,
    index_heads: usize,
    selected_width: usize,
}

/// Right-aligned broadcast metadata for the optional additive bias.
struct BiasMeta {
    ptr: CUdeviceptr,
    present: bool,
    rank: i32,
    dims: [u64; 4],
}

pub struct IndexShareFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for IndexShareFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let num_heads = required_positive_int(node, "num_heads")?;
        let kv_num_heads = optional_positive_int(node, "kv_num_heads")?.unwrap_or(num_heads);
        if num_heads % kv_num_heads != 0 {
            return Err(error(format!(
                "num_heads {num_heads} must be a multiple of kv_num_heads {kv_num_heads}"
            )));
        }
        let scale = node
            .attr("scale")
            .map(|attribute| {
                attribute
                    .as_float()
                    .ok_or_else(|| error("attribute 'scale' must be a float"))
            })
            .transpose()?;
        if scale.is_some_and(|scale| !scale.is_finite() || scale <= 0.0) {
            return Err(error("attribute 'scale' must be finite and > 0"));
        }
        Ok(Box::new(IndexShareKernel {
            runtime: self.runtime.clone(),
            num_heads,
            kv_num_heads,
            scale,
        }))
    }
}

#[derive(Debug)]
pub struct IndexShareKernel {
    runtime: Arc<CudaRuntime>,
    num_heads: usize,
    kv_num_heads: usize,
    scale: Option<f32>,
}

impl Kernel for IndexShareKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !(6..=7).contains(&inputs.len()) {
            return Err(error(format!(
                "expected 6 or 7 inputs, got {}",
                inputs.len()
            )));
        }
        if !matches!(outputs.len(), 1 | 3) {
            return Err(error(format!(
                "expected 1 output or 3 outputs (paired present K/V), got {}",
                outputs.len()
            )));
        }
        // Inputs may have been uploaded asynchronously on the EP stream.
        self.runtime.synchronize()?;

        for &index in &[0, 1, 2, 5] {
            if inputs[index].is_absent() {
                return Err(error(format!(
                    "required input {index} ('{}') is absent",
                    INPUT_NAMES[index]
                )));
            }
        }
        let has_past_key = optional_input(inputs, 3).is_some();
        let has_past_value = optional_input(inputs, 4).is_some();
        if has_past_key != has_past_value {
            return Err(error("past_key and past_value must be provided together"));
        }
        for &index in &[0, 1, 2] {
            require_dtype(index, &inputs[index], DataType::Float32)?;
        }
        for index in [3, 4, 6] {
            if let Some(input) = optional_input(inputs, index) {
                require_dtype(index, input, DataType::Float32)?;
            }
        }
        if !matches!(inputs[5].dtype, DataType::Int32 | DataType::Int64) {
            return Err(error(format!(
                "input 5 ('selected_indices') dtype {:?} unsupported; expected Int32 or Int64",
                inputs[5].dtype
            )));
        }
        for (index, output) in outputs.iter().enumerate() {
            if output.dtype != DataType::Float32 {
                return Err(error(format!(
                    "output {index} dtype {:?} unsupported; expected Float32",
                    output.dtype
                )));
            }
        }
        for &index in &[0, 1, 2, 5] {
            if !inputs[index].is_contiguous() {
                return Err(error(format!(
                    "input {index} ('{}') must be contiguous",
                    INPUT_NAMES[index]
                )));
            }
        }
        for index in [3, 4, 6] {
            if let Some(input) = optional_input(inputs, index)
                && !input.is_contiguous()
            {
                return Err(error(format!(
                    "input {index} ('{}') must be contiguous",
                    INPUT_NAMES[index]
                )));
            }
        }
        for output in outputs.iter() {
            if !output.is_contiguous() {
                return Err(error("outputs must be contiguous"));
            }
        }

        let dims = self.validate_shapes(inputs, outputs)?;

        // Copy the small index tensor to the host for the deterministic ONNX
        // range/order validation (bulk Q/K/V/bias stay on device).
        let indices = self.read_indices(&inputs[5], dims)?;
        validate_indices(&indices, dims)?;

        let bias = self.bias_meta(inputs, dims)?;

        let q_ptr = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let key_ptr = cuptr(inputs[1].data_ptr::<u8>() as *const c_void);
        let value_ptr = cuptr(inputs[2].data_ptr::<u8>() as *const c_void);
        let past_key_ptr = optional_input(inputs, 3)
            .map(|view| cuptr(view.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let past_value_ptr = optional_input(inputs, 4)
            .map(|view| cuptr(view.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let indices_ptr = cuptr(inputs[5].data_ptr::<u8>() as *const c_void);
        let index_is_i64 = i32::from(inputs[5].dtype == DataType::Int64);

        // Present K/V element counts and the output element count.
        let present_elements = dims.batch * dims.kv_heads * dims.total_seq * dims.head_size;
        let output_elements = dims.batch * dims.q_heads * dims.q_seq * dims.head_size;
        let scores_elements = dims.batch * dims.q_heads * dims.q_seq * dims.selected_width;

        // Present outputs are written directly into the caller's output slots
        // when requested (outputs 1 and 2); otherwise present K/V land in scratch
        // that only feeds the attention kernel.
        let want_present = outputs.len() == 3;
        let (output_head, output_tail) = outputs.split_at_mut(1);
        let y_ptr = cuptr(output_head[0].data_ptr_mut::<u8>() as *const c_void);
        let (present_key_out, present_value_out) = if want_present {
            (
                cuptr(output_tail[0].data_ptr_mut::<u8>() as *const c_void),
                cuptr(output_tail[1].data_ptr_mut::<u8>() as *const c_void),
            )
        } else {
            (0, 0)
        };

        let mut owned: Vec<CUdeviceptr> = Vec::new();
        let result = (|| -> Result<()> {
            let alloc = |owned: &mut Vec<CUdeviceptr>, bytes: usize| -> Result<CUdeviceptr> {
                let ptr = self.runtime.alloc_raw(bytes.max(1))?;
                owned.push(ptr);
                Ok(ptr)
            };

            let present_key_ptr = if want_present {
                present_key_out
            } else {
                alloc(&mut owned, present_elements * 4)?
            };
            let present_value_ptr = if want_present {
                present_value_out
            } else {
                alloc(&mut owned, present_elements * 4)?
            };
            let scores_ptr = alloc(&mut owned, scores_elements * 4)?;

            self.build_present(past_key_ptr, key_ptr, present_key_ptr, has_past_key, dims)?;
            self.build_present(
                past_value_ptr,
                value_ptr,
                present_value_ptr,
                has_past_value,
                dims,
            )?;

            self.launch_rows(
                q_ptr,
                present_key_ptr,
                present_value_ptr,
                indices_ptr,
                &bias,
                scores_ptr,
                y_ptr,
                dims,
                index_is_i64,
                output_elements,
            )?;
            self.runtime.synchronize()
        })();

        for ptr in owned {
            // SAFETY: every pointer came from this runtime's `alloc_raw`.
            unsafe {
                let _ = self.runtime.free_raw(ptr);
            }
        }
        result
    }

    fn supports_strided_input(&self, _index: usize) -> bool {
        false
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "host-side selected_indices validation copies the index tensor D2H and synchronizes the stream",
        )
    }
}

impl IndexShareKernel {
    fn validate_shapes(&self, inputs: &[TensorView], outputs: &[TensorMut]) -> Result<Dims> {
        for &index in &[0, 1, 2, 5] {
            require_rank(index, inputs[index].shape)?;
        }
        for index in [3, 4] {
            if let Some(input) = optional_input(inputs, index) {
                require_rank(index, input.shape)?;
            }
        }
        let q = inputs[0].shape;
        let key = inputs[1].shape;
        let value = inputs[2].shape;
        let (batch, q_heads, q_seq, head_size) = (q[0], q[1], q[2], q[3]);
        if q_heads != self.num_heads {
            return Err(error(format!(
                "query head dimension {q_heads} must equal num_heads {}",
                self.num_heads
            )));
        }
        if key[0] != batch || value[0] != batch {
            return Err(error("query, key, and value batch dimensions must match"));
        }
        if key[1] != self.kv_num_heads || value[1] != self.kv_num_heads {
            return Err(error(format!(
                "key/value head dimensions must equal kv_num_heads {}",
                self.kv_num_heads
            )));
        }
        if key[2] != value[2] || key[3] != head_size || value[3] != head_size {
            return Err(error(
                "key/value sequence and head dimensions must match query/schema",
            ));
        }
        let current_seq = key[2];
        let mut past_seq = 0;
        if let (Some(past_key), Some(past_value)) =
            (optional_input(inputs, 3), optional_input(inputs, 4))
        {
            if past_key.shape != past_value.shape {
                return Err(error("past_key and past_value shapes must match"));
            }
            if past_key.shape[0] != batch
                || past_key.shape[1] != self.kv_num_heads
                || past_key.shape[3] != head_size
            {
                return Err(error(
                    "past key/value must have shape [B, kv_num_heads, S_past, H]",
                ));
            }
            past_seq = past_key.shape[2];
        }
        let total_seq = past_seq
            .checked_add(current_seq)
            .ok_or_else(|| error("total cache sequence length overflow"))?;
        let selected = inputs[5].shape;
        let index_heads = selected[1];
        if selected[0] != batch
            || (index_heads != 1 && index_heads != q_heads)
            || selected[2] != q_seq
        {
            return Err(error(format!(
                "selected_indices must have shape [B, 1|N, S_q, K], got {selected:?}"
            )));
        }
        if selected[3] == 0 {
            return Err(error("selected_indices K dimension must be nonzero"));
        }
        if let Some(bias) = optional_input(inputs, 6) {
            validate_bias_shape(bias.shape, [batch, q_heads, q_seq, total_seq])?;
        }
        if outputs[0].shape != q {
            return Err(error(format!(
                "output shape {:?} must equal query shape {q:?}",
                outputs[0].shape
            )));
        }
        if outputs.len() == 3 {
            let expected = [batch, self.kv_num_heads, total_seq, head_size];
            if outputs[1].shape != expected || outputs[2].shape != expected {
                return Err(error(format!(
                    "present_key and present_value shapes must be {expected:?}"
                )));
            }
        }
        Ok(Dims {
            batch,
            q_heads,
            kv_heads: self.kv_num_heads,
            q_seq,
            current_seq,
            past_seq,
            total_seq,
            head_size,
            index_heads,
            selected_width: selected[3],
        })
    }

    fn read_indices(&self, view: &TensorView, dims: Dims) -> Result<Vec<i64>> {
        let count = dims.batch * dims.index_heads * dims.q_seq * dims.selected_width;
        let byte_len = view.dtype.storage_bytes(count);
        let mut host = vec![0u8; byte_len];
        if !host.is_empty() {
            // SAFETY: `view` is a live contiguous device tensor and the host
            // buffer is exactly its fixed-width storage size.
            unsafe {
                self.runtime
                    .dtoh(&mut host, cuptr(view.data_ptr::<u8>() as *const c_void))?;
            }
        }
        Ok(host
            .chunks_exact(view.dtype.byte_size())
            .map(|raw| match view.dtype {
                DataType::Int32 => i32::from_ne_bytes(raw.try_into().unwrap()) as i64,
                DataType::Int64 => i64::from_ne_bytes(raw.try_into().unwrap()),
                _ => unreachable!("index dtype was validated"),
            })
            .collect())
    }

    fn bias_meta(&self, inputs: &[TensorView], dims: Dims) -> Result<BiasMeta> {
        match optional_input(inputs, 6) {
            Some(view) => {
                let rank = view.shape.len();
                if rank > 4 {
                    return Err(error(format!(
                        "attention_bias rank {rank} exceeds 4"
                    )));
                }
                let expected = view.numel();
                let actual = view.shape.iter().product::<usize>();
                if expected != actual {
                    return Err(error("attention_bias element count mismatch"));
                }
                let _ = dims;
                let mut broadcast = [1u64; 4];
                for (axis, &dim) in view.shape.iter().enumerate() {
                    broadcast[4 - rank + axis] = dim as u64;
                }
                Ok(BiasMeta {
                    ptr: cuptr(view.data_ptr::<u8>() as *const c_void),
                    present: true,
                    rank: rank as i32,
                    dims: broadcast,
                })
            }
            None => Ok(BiasMeta {
                ptr: 0,
                present: false,
                rank: 0,
                dims: [1u64; 4],
            }),
        }
    }

    fn build_present(
        &self,
        past_ptr: CUdeviceptr,
        current_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        has_past: bool,
        dims: Dims,
    ) -> Result<()> {
        let elements = (dims.batch * dims.kv_heads * dims.total_seq * dims.head_size) as u64;
        if elements == 0 {
            return Ok(());
        }
        let func = self
            .runtime
            .nvrtc_function(MODULE, SOURCE, "build_present")?;
        let has_past_i = i32::from(has_past);
        let batch = dims.batch as u64;
        let heads = dims.kv_heads as u64;
        let past_seq = dims.past_seq as u64;
        let cur_seq = dims.current_seq as u64;
        let total_seq = dims.total_seq as u64;
        let head_size = dims.head_size as u64;
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&past_ptr)
            .arg(&current_ptr)
            .arg(&out_ptr)
            .arg(&has_past_i)
            .arg(&batch)
            .arg(&heads)
            .arg(&past_seq)
            .arg(&cur_seq)
            .arg(&total_seq)
            .arg(&head_size)
            .arg(&elements);
        // SAFETY: argument types/order match `build_present`; all pointers refer
        // to live contiguous device allocations validated above.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (elements.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch build_present", e))
        .map(|_| ())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_rows(
        &self,
        q_ptr: CUdeviceptr,
        present_key_ptr: CUdeviceptr,
        present_value_ptr: CUdeviceptr,
        indices_ptr: CUdeviceptr,
        bias: &BiasMeta,
        scores_ptr: CUdeviceptr,
        y_ptr: CUdeviceptr,
        dims: Dims,
        index_is_i64: i32,
        output_elements: usize,
    ) -> Result<()> {
        let total_rows = (dims.batch * dims.q_heads * dims.q_seq) as u64;
        if total_rows == 0 || output_elements == 0 {
            return Ok(());
        }
        let func = self
            .runtime
            .nvrtc_function(MODULE, SOURCE, "index_share_row")?;
        let scale = self
            .scale
            .unwrap_or_else(|| 1.0 / (dims.head_size as f32).sqrt());
        let sqrt_scale = scale.sqrt();
        let group = (dims.q_heads / dims.kv_heads) as u64;
        let batch = dims.batch as u64;
        let q_heads = dims.q_heads as u64;
        let kv_heads = dims.kv_heads as u64;
        let q_seq = dims.q_seq as u64;
        let total_seq = dims.total_seq as u64;
        let head_size = dims.head_size as u64;
        let index_heads = dims.index_heads as u64;
        let selected_width = dims.selected_width as u64;
        let has_bias = i32::from(bias.present);
        let bias_rank = bias.rank;
        let (bd0, bd1, bd2, bd3) = (bias.dims[0], bias.dims[1], bias.dims[2], bias.dims[3]);
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&q_ptr)
            .arg(&present_key_ptr)
            .arg(&present_value_ptr)
            .arg(&indices_ptr)
            .arg(&bias.ptr)
            .arg(&scores_ptr)
            .arg(&y_ptr)
            .arg(&batch)
            .arg(&q_heads)
            .arg(&kv_heads)
            .arg(&q_seq)
            .arg(&total_seq)
            .arg(&head_size)
            .arg(&index_heads)
            .arg(&selected_width)
            .arg(&group)
            .arg(&sqrt_scale)
            .arg(&index_is_i64)
            .arg(&has_bias)
            .arg(&bias_rank)
            .arg(&bd0)
            .arg(&bd1)
            .arg(&bd2)
            .arg(&bd3);
        // SAFETY: argument types/order match `index_share_row`; all pointers
        // refer to live contiguous device allocations, the scores scratch is
        // sized for `batch*q_heads*q_seq*selected_width` f32, and every index was
        // range-checked on the host.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (total_rows.min(u32::MAX as u64).max(1) as u32, 1, 1),
                block_dim: (ROW_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch index_share_row", e))
        .map(|_| ())
    }
}

fn validate_indices(indices: &[i64], dims: Dims) -> Result<()> {
    for b in 0..dims.batch {
        for h in 0..dims.index_heads {
            for q in 0..dims.q_seq {
                let row = ((b * dims.index_heads + h) * dims.q_seq + q) * dims.selected_width;
                let mut previous = None;
                let mut padding = false;
                let mut count = 0;
                for (column, &index) in indices[row..row + dims.selected_width].iter().enumerate() {
                    if index == -1 {
                        padding = true;
                        continue;
                    }
                    if index < -1 {
                        return Err(index_error(b, h, q, column, format!("invalid sentinel {index}")));
                    }
                    if padding {
                        return Err(index_error(
                            b,
                            h,
                            q,
                            column,
                            format!("index {index} follows trailing -1 padding"),
                        ));
                    }
                    if index as usize >= dims.total_seq {
                        return Err(index_error(
                            b,
                            h,
                            q,
                            column,
                            format!(
                                "index {index} is out of range for cache length {}",
                                dims.total_seq
                            ),
                        ));
                    }
                    if let Some(previous) = previous
                        && index <= previous
                    {
                        let reason = if index == previous {
                            format!("duplicate index {index}")
                        } else {
                            format!("indices are not strictly increasing: {previous} then {index}")
                        };
                        return Err(index_error(b, h, q, column, reason));
                    }
                    previous = Some(index);
                    count += 1;
                }
                if count == 0 {
                    return Err(error(format!(
                        "selected_indices row [batch={b}, head={h}, query={q}] is all -1"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn validate_bias_shape(shape: &[usize], target: [usize; 4]) -> Result<()> {
    if shape.len() > 4 {
        return Err(error(format!(
            "attention_bias rank {} exceeds 4",
            shape.len()
        )));
    }
    for (axis, &dimension) in shape.iter().enumerate() {
        let expected = target[4 - shape.len() + axis];
        if dimension != 1 && dimension != expected {
            return Err(error(format!(
                "attention_bias dimension {dimension} is not broadcastable to {target:?}"
            )));
        }
    }
    Ok(())
}

fn index_error(batch: usize, head: usize, query: usize, column: usize, reason: String) -> EpError {
    error(format!(
        "selected_indices [batch={batch}, head={head}, query={query}, column={column}]: {reason}"
    ))
}

fn require_rank(index: usize, shape: &[usize]) -> Result<()> {
    if shape.len() != 4 {
        return Err(error(format!(
            "input {index} ('{}') rank {} unsupported; expected 4",
            INPUT_NAMES[index],
            shape.len()
        )));
    }
    Ok(())
}

fn require_dtype(index: usize, input: &TensorView, expected: DataType) -> Result<()> {
    if input.dtype != expected {
        return Err(error(format!(
            "input {index} ('{}') dtype {:?} unsupported; expected {expected:?}",
            INPUT_NAMES[index], input.dtype
        )));
    }
    Ok(())
}

fn required_positive_int(node: &Node, name: &str) -> Result<usize> {
    let value = node
        .attr(name)
        .ok_or_else(|| error(format!("missing required integer attribute '{name}'")))?
        .as_int()
        .ok_or_else(|| error(format!("attribute '{name}' must be an integer")))?;
    usize::try_from(value)
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| error(format!("attribute '{name}' must be > 0")))
}

fn optional_positive_int(node: &Node, name: &str) -> Result<Option<usize>> {
    node.attr(name)
        .map(|attribute| {
            let value = attribute
                .as_int()
                .ok_or_else(|| error(format!("attribute '{name}' must be an integer")))?;
            usize::try_from(value)
                .ok()
                .filter(|&value| value > 0)
                .ok_or_else(|| error(format!("attribute '{name}' must be > 0")))
        })
        .transpose()
}

fn optional_input<'a>(inputs: &'a [TensorView<'a>], index: usize) -> Option<&'a TensorView<'a>> {
    inputs.get(index).filter(|input| !input.is_absent())
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep {OP}: {}", message.into()))
}
