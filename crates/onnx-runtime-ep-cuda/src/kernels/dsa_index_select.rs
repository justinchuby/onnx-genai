//! `pkg.nxrt::DsaIndexSelect` v1: device-resident query-dependent sparse index
//! selection for the GLM-5.2 DSA / IndexShare indexer.
//!
//! The frozen CPU reference in
//! `crates/onnx-runtime-ep-cpu/src/kernels/dsa_index_select.rs` is the
//! authoritative numerical oracle. This kernel reproduces its math **on the
//! device**: for every `(batch, query)` row it scores every key position with
//! the head-weighted, ReLU'd, scaled dot product, adds the additive f32 causal
//! bias, applies the `bias > -1e30` allow mask, selects the top
//! `min(#allowed, top_k)` positions by `(score desc, index asc)`, and emits them
//! sorted ascending by position, right-padded with the `-1` sentinel to the
//! fixed width `top_k`. Query, key, weights, bias, and the output all stay
//! resident on the device.
//!
//! ## Determinism / bit-parity
//!
//! Every reduction sums in the same fixed ascending order as the CPU reference:
//! each key's dot product accumulates over `head_dim` in one thread, the
//! head-weighted sum accumulates over `heads` ascending in that same thread, and
//! the final `weighted + bias` add matches the reference exactly. The top-k tie
//! break reproduces the CPU oracle's `f32::total_cmp` order bit-for-bit via the
//! `total_order_key` integer transform (`(score desc, index asc)`; lower index
//! wins on an exact tie). `scale`, `weights_scale`, and the ReLU clamp are
//! applied in the reference's operand order, so scores are byte-identical to the
//! CPU oracle across f16/bf16/f32 storage.
//!
//! ## Capture support
//!
//! The op *produces* the `selected_indices` tensor — it consumes no index input
//! — so there is no host D2H validation to skip and no capture-error latch: the
//! kernel never synchronizes the stream or copies to the host on any path. After
//! a warmed eager execution has sized the executor-owned persistent workspace
//! (the per-row score and selection-state scratch keep stable device addresses
//! across warmup → capture → replay) and compiled the NVRTC kernel, the launch
//! path is legal to record into a CUDA graph and replay with only device-buffer
//! contents changing:
//!
//!   * No `stream.synchronize()` on the capturing path.
//!   * No per-call `cudaMalloc`/`cudaFree`: scratch is supplied through the
//!     session workspace contract, so captured replay keeps fixed addresses.
//!   * No host round-trip of any kind.
//!
//! Capture stays gated off until such a warmup has run (mirroring
//! [`super::index_share`]); until then [`capture_support`] reports the missing
//! precondition.
//!
//! ## Claim-time gating
//!
//! [`unsupported_reason`] (called by the CUDA provider) delegates to the CPU
//! oracle's own `unsupported_reason`, so the two backends reject exactly the same
//! attr/dtype/rank/shape combinations at claim time rather than claiming a node
//! and falling back inside the kernel. Query/key/weights are projected to f32 for
//! the structural CPU check while the strict f32-only `attention_bias` contract
//! is preserved (its dtype is not projected).

use std::borrow::Cow;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut, TensorView,
    WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node};
use onnx_runtime_memory_governor::MemoryRole;

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const OP: &str = "DsaIndexSelect";

const INPUT_NAMES: [&str; 4] = ["query", "key", "weights", "attention_bias"];

/// Threads per block; one block services one `(batch, query)` output row.
const ROW_THREADS: u32 = 128;
const WORKSPACE_ALIGNMENT: usize = 256;

const MODULE: &str = "dsa_index_select_f32_f16_bf16_v1";
const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define NEG_INF __int_as_float(0xff800000)
// Bias at or below this magnitude is the -inf / finfo.min causal fill.
#define MASK_THRESHOLD (-1e30f)

// dtype is 0 for f32, 1 for f16, and 2 for bf16. Scores/reductions stay fp32;
// only the externally visible query/key/weights tensors use this storage type.
__device__ __forceinline__ float load_float(
    const void* data, unsigned long long index, int dtype) {
  if (dtype == 0) {
    return ((const float*)data)[index];
  }
  if (dtype == 1) {
    return __half2float(((const __half*)data)[index]);
  }
  return __bfloat162float(((const __nv_bfloat16*)data)[index]);
}

// Integer key that orders floats identically to Rust's `f32::total_cmp`
// (used by the CPU oracle's `score_order`): the sign bit is flipped for
// positives and every bit for negatives, so a plain signed-integer compare of
// the keys reproduces total order (including -0 < +0 and NaN placement).
__device__ __forceinline__ int total_order_key(float x) {
  int bits = __float_as_int(x);
  int sign = bits >> 31;                          // 0 or -1 (arithmetic shift)
  int flip = (int)(((unsigned int)sign) >> 1);    // 0 or 0x7fffffff
  return bits ^ flip;
}

// One block per (batch, query) row. Threads cooperatively score every key
// position (Phase 1); the lead thread then runs the deterministic top-k
// selection and ascending emit (Phase 2). `state` marks each position as
// 0 = allowed/unselected, 1 = masked, 2 = selected.
extern "C" __global__ void dsa_index_select_row(
    const void* query, const void* key, const void* weights, const float* bias,
    float* scores, unsigned char* state, long long* out,
    unsigned long long batch, unsigned long long q_seq, unsigned long long heads,
    unsigned long long head_dim, unsigned long long key_seq,
    unsigned long long top_k, float scale, float weights_scale, int dtype) {
  const unsigned long long row = blockIdx.x;
  const unsigned long long total_rows = batch * q_seq;
  if (row >= total_rows) {
    return;
  }
  const unsigned long long b = row / q_seq;
  const unsigned long long s = row - b * q_seq;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const unsigned long long weights_base = (b * q_seq + s) * heads;
  const unsigned long long bias_base = (b * q_seq + s) * key_seq;  // bias[b,0,s,.]
  const unsigned long long score_base = row * key_seq;
  const unsigned long long out_base = row * top_k;

  // Initialise the output row to the -1 padding sentinel.
  for (unsigned long long i = tid; i < top_k; i += nthreads) {
    out[out_base + i] = -1;
  }

  // Phase 1: head-weighted, ReLU'd, scaled score for every allowed key.
  for (unsigned long long t = tid; t < key_seq; t += nthreads) {
    const float bias_bt = bias[bias_base + t];
    // NaN and the -inf / finfo.min causal fill are both "not allowed"; binding
    // the comparison keeps the negation on a bool (partial-order safe).
    const bool allowed = bias_bt > MASK_THRESHOLD;
    if (!allowed) {
      state[score_base + t] = 1;         // masked / -inf causal fill
      scores[score_base + t] = NEG_INF;
      continue;
    }
    float weighted = 0.0f;
    for (unsigned long long h = 0; h < heads; ++h) {
      const unsigned long long q_base = ((b * q_seq + s) * heads + h) * head_dim;
      const unsigned long long k_base = (b * key_seq + t) * head_dim;
      float dot = 0.0f;
      for (unsigned long long d = 0; d < head_dim; ++d) {
        dot += load_float(query, q_base + d, dtype) *
               load_float(key, k_base + d, dtype);
      }
      const float scored = fmaxf(scale * dot, 0.0f);  // Relu(scale * dot)
      const float wprod = load_float(weights, weights_base + h, dtype) * weights_scale;
      weighted += scored * wprod;
    }
    scores[score_base + t] = weighted + bias_bt;
    state[score_base + t] = 0;            // allowed, unselected
  }
  __syncthreads();

  // Phase 2: deterministic top-k on the lead thread. `keep` rounds of argmax by
  // (total_cmp desc, index asc), then emit the winners ascending by position.
  if (tid == 0) {
    unsigned long long allowed_count = 0;
    for (unsigned long long t = 0; t < key_seq; ++t) {
      if (state[score_base + t] == 0) {
        allowed_count += 1;
      }
    }
    const unsigned long long keep = allowed_count < top_k ? allowed_count : top_k;
    for (unsigned long long r = 0; r < keep; ++r) {
      long long best_t = -1;
      int best_key = 0;
      for (unsigned long long t = 0; t < key_seq; ++t) {
        if (state[score_base + t] != 0) {
          continue;                       // masked or already selected
        }
        const int candidate = total_order_key(scores[score_base + t]);
        // Strict `>` keeps the lower index on an exact tie (ascending scan).
        if (best_t < 0 || candidate > best_key) {
          best_key = candidate;
          best_t = (long long)t;
        }
      }
      if (best_t < 0) {
        break;                            // defensive; `keep` rules this out
      }
      state[score_base + (unsigned long long)best_t] = 2;   // selected
    }
    unsigned long long slot = 0;
    for (unsigned long long t = 0; t < key_seq; ++t) {
      if (state[score_base + t] == 2) {
        out[out_base + slot] = (long long)t;
        slot += 1;
      }
    }
  }
}
"#;

/// Claim-time validation preserving the CPU oracle's structural ABI checks while
/// extending its f32-only execution oracle to CUDA's f16/bf16 storage variants.
/// Query/key/weights are projected to f32 for the CPU structural check after this
/// method enforces CUDA's homogeneous floating dtype contract; the strict
/// `attention_bias` f32-only contract is preserved (dtype 3 is not projected).
pub(crate) fn unsupported_reason(
    node: &Node,
    shapes: &[onnx_runtime_ir::Shape],
    input_dtypes: &[DataType],
) -> Option<Cow<'static, str>> {
    let dtype_at = |index: usize| {
        input_dtypes
            .get(index)
            .copied()
            .unwrap_or(DataType::Undefined)
    };
    let dtype = dtype_at(0);
    if !matches!(
        dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        return Some(Cow::Owned(format!(
            "DsaIndexSelect: query dtype {dtype:?} unsupported on CUDA (expected f32, f16, or bf16)"
        )));
    }
    for index in [1, 2] {
        let candidate = dtype_at(index);
        if candidate != DataType::Undefined && candidate != dtype {
            return Some(Cow::Borrowed(
                "DsaIndexSelect: query, key, and weights must use the same floating dtype on CUDA",
            ));
        }
    }
    // Project only the shared query/key/weights float trio to f32 for the CPU
    // structural check. `attention_bias` (index 3) is left untouched so the CPU
    // validator still enforces its strict f32-only contract.
    let projected: Vec<_> = input_dtypes
        .iter()
        .enumerate()
        .map(|(index, &candidate)| {
            if index != 3
                && matches!(
                    candidate,
                    DataType::Float32 | DataType::Float16 | DataType::BFloat16
                )
            {
                DataType::Float32
            } else {
                candidate
            }
        })
        .collect();
    onnx_runtime_ep_cpu::kernels::dsa_index_select::unsupported_reason(node, shapes, &projected)
}

/// Resolved geometry shared between the CPU reference and this kernel.
#[derive(Clone, Copy)]
struct Dims {
    batch: usize,
    q_seq: usize,
    heads: usize,
    head_dim: usize,
    key_seq: usize,
}

#[derive(Clone, Copy, Debug)]
struct WorkspaceLayout {
    bytes: usize,
    scores_offset: usize,
    state_offset: usize,
}

impl WorkspaceLayout {
    fn ptr_at(self, workspace: WorkspaceView, offset: usize) -> CUdeviceptr {
        workspace.ptr().0 as CUdeviceptr + offset as CUdeviceptr
    }

    fn scores(self, workspace: WorkspaceView) -> CUdeviceptr {
        self.ptr_at(workspace, self.scores_offset)
    }

    fn state(self, workspace: WorkspaceView) -> CUdeviceptr {
        self.ptr_at(workspace, self.state_offset)
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    let mask = alignment
        .checked_sub(1)
        .ok_or_else(|| error("workspace alignment underflow"))?;
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or_else(|| error("DsaIndexSelect workspace size overflow"))
}

fn append_workspace_segment(offset: &mut usize, bytes: usize) -> Result<usize> {
    let start = align_up(*offset, WORKSPACE_ALIGNMENT)?;
    *offset = start
        .checked_add(bytes.max(1))
        .ok_or_else(|| error("DsaIndexSelect workspace size overflow"))?;
    Ok(start)
}

fn dsa_index_select_workspace_layout(dims: Dims) -> Result<WorkspaceLayout> {
    let rows = dims
        .batch
        .checked_mul(dims.q_seq)
        .ok_or_else(|| error("DsaIndexSelect workspace row count overflow"))?;
    let cells = rows
        .checked_mul(dims.key_seq)
        .ok_or_else(|| error("DsaIndexSelect workspace cell count overflow"))?;
    let scores_bytes = cells
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| error("DsaIndexSelect score workspace byte count overflow"))?;
    let state_bytes = cells; // one u8 per cell

    let mut offset = 0usize;
    let scores_offset = append_workspace_segment(&mut offset, scores_bytes)?;
    let state_offset = append_workspace_segment(&mut offset, state_bytes)?;
    let bytes = align_up(offset, WORKSPACE_ALIGNMENT)?;
    Ok(WorkspaceLayout {
        bytes,
        scores_offset,
        state_offset,
    })
}

pub struct DsaIndexSelectFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for DsaIndexSelectFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let top_k = required_positive_int(node, "top_k")?;
        let scale = required_finite_positive_float(node, "scale")?;
        let weights_scale = match node.attr("weights_scale") {
            Some(attribute) => {
                let value = attribute
                    .as_float()
                    .ok_or_else(|| error("attribute 'weights_scale' must be a float"))?;
                if !value.is_finite() || value <= 0.0 {
                    return Err(error("attribute 'weights_scale' must be finite and > 0"));
                }
                value
            }
            None => 1.0,
        };
        // Reject any attribute outside the frozen v1 ABI (mirrors the CPU oracle).
        for name in node.attributes.keys() {
            if !matches!(name.as_str(), "top_k" | "scale" | "weights_scale") {
                return Err(error(format!(
                    "attribute '{name}' is not part of the frozen v1 ABI"
                )));
            }
        }
        Ok(Box::new(DsaIndexSelectKernel {
            runtime: self.runtime.clone(),
            top_k,
            scale,
            weights_scale,
            warmed: AtomicBool::new(false),
        }))
    }
}

#[derive(Debug)]
pub struct DsaIndexSelectKernel {
    runtime: Arc<CudaRuntime>,
    top_k: usize,
    scale: f32,
    weights_scale: f32,
    /// Set after a successful eager execution has compiled the NVRTC kernel and
    /// sized the persistent workspace, the precondition for CUDA-graph capture.
    warmed: AtomicBool,
}

impl Kernel for DsaIndexSelectKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let _ = (inputs, outputs);
        Err(error(
            "DsaIndexSelect requires executor-provided persistent workspace",
        ))
    }

    fn workspace_requirement(&self, inputs: &[TensorMetadata<'_>]) -> Result<WorkspaceRequirement> {
        let dims = self.validate_metadata(inputs)?;
        let layout = dsa_index_select_workspace_layout(dims)?;
        Ok(WorkspaceRequirement {
            bytes: u64::try_from(layout.bytes)
                .map_err(|_| error("DsaIndexSelect workspace does not fit u64"))?,
            alignment: WORKSPACE_ALIGNMENT,
            lifetime: WorkspaceLifetime::SessionPersistent,
            role: MemoryRole::Workspace { step_scoped: false },
        })
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let workspace = workspace.ok_or_else(|| {
            error("DsaIndexSelect requires executor-provided persistent workspace")
        })?;
        if inputs.len() != 4 {
            return Err(error(format!("expected 4 inputs, got {}", inputs.len())));
        }
        if outputs.len() != 1 {
            return Err(error(format!("expected 1 output, got {}", outputs.len())));
        }
        for (index, name) in INPUT_NAMES.iter().enumerate() {
            if inputs[index].is_absent() {
                return Err(error(format!(
                    "required input {index} ('{name}') is absent"
                )));
            }
        }
        let dtype = require_floating_dtype(&inputs[0], 0)?;
        for index in [1, 2] {
            if inputs[index].dtype != dtype {
                return Err(error(format!(
                    "input {index} ('{}') dtype {:?} must match query dtype {dtype:?}",
                    INPUT_NAMES[index], inputs[index].dtype
                )));
            }
        }
        // The additive bias is an f32 mask whose -inf / finfo.min fill magnitude
        // is only meaningful in f32.
        if inputs[3].dtype != DataType::Float32 {
            return Err(error(format!(
                "input 3 ('attention_bias') dtype {:?} unsupported; expected Float32",
                inputs[3].dtype
            )));
        }
        if outputs[0].dtype != DataType::Int64 {
            return Err(error(format!(
                "output 0 ('selected_indices') dtype {:?} unsupported; expected Int64",
                outputs[0].dtype
            )));
        }
        for index in 0..4 {
            if !inputs[index].is_contiguous() {
                return Err(error(format!(
                    "input {index} ('{}') must be contiguous",
                    INPUT_NAMES[index]
                )));
            }
        }
        if !outputs[0].is_contiguous() {
            return Err(error("output 0 ('selected_indices') must be contiguous"));
        }

        let dims = self.derive_dims(inputs)?;
        let expected_out = [dims.batch, 1, dims.q_seq, self.top_k];
        if outputs[0].shape != expected_out {
            return Err(error(format!(
                "output 0 ('selected_indices') shape {:?} unsupported; expected {expected_out:?} (B, 1, S, top_k)",
                outputs[0].shape
            )));
        }

        let capturing = self.runtime.is_capturing()?;

        let layout = dsa_index_select_workspace_layout(dims)?;
        if workspace.bytes() < layout.bytes {
            return Err(error(format!(
                "DsaIndexSelect workspace invariant mismatch: execute requires {} bytes, prepared {} bytes",
                layout.bytes,
                workspace.bytes()
            )));
        }

        let query_ptr = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let key_ptr = cuptr(inputs[1].data_ptr::<u8>() as *const c_void);
        let weights_ptr = cuptr(inputs[2].data_ptr::<u8>() as *const c_void);
        let bias_ptr = cuptr(inputs[3].data_ptr::<u8>() as *const c_void);
        let out_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let scores_ptr = layout.scores(workspace);
        let state_ptr = layout.state(workspace);

        self.launch_row(
            query_ptr,
            key_ptr,
            weights_ptr,
            bias_ptr,
            scores_ptr,
            state_ptr,
            out_ptr,
            dims,
            dtype_code(dtype)?,
        )?;

        if capturing {
            Ok(())
        } else {
            // Eager completion barrier; also the point at which the NVRTC kernel
            // is guaranteed compiled and the workspace sized, so capture may now
            // record this kernel with stable device addresses.
            self.runtime.synchronize()?;
            self.warmed.store(true, Ordering::Relaxed);
            Ok(())
        }
    }

    fn supports_strided_input(&self, _index: usize) -> bool {
        false
    }

    fn capture_support(&self) -> CaptureSupport {
        if self.warmed.load(Ordering::Relaxed) {
            CaptureSupport::Supported
        } else {
            CaptureSupport::unsupported(
                "requires a warmed fixed-shape eager DsaIndexSelect pass to compile the NVRTC \
                 kernel and size the prepared workspace",
            )
        }
    }
}

impl DsaIndexSelectKernel {
    fn validate_metadata(&self, inputs: &[TensorMetadata<'_>]) -> Result<Dims> {
        if inputs.len() != 4 {
            return Err(error(format!(
                "expected 4 input metadata entries, got {}",
                inputs.len()
            )));
        }
        let query = inputs[0].shape;
        let key = inputs[1].shape;
        let weights = inputs[2].shape;
        let bias = inputs[3].shape;
        self.resolve_dims(query, key, weights, bias)
    }

    fn derive_dims(&self, inputs: &[TensorView]) -> Result<Dims> {
        self.resolve_dims(
            inputs[0].shape,
            inputs[1].shape,
            inputs[2].shape,
            inputs[3].shape,
        )
    }

    fn resolve_dims(
        &self,
        query: &[usize],
        key: &[usize],
        weights: &[usize],
        bias: &[usize],
    ) -> Result<Dims> {
        if query.len() != 4 {
            return Err(error(format!(
                "input 0 ('query') rank {} unsupported; expected 4 (B, S, H, D)",
                query.len()
            )));
        }
        if key.len() != 3 {
            return Err(error(format!(
                "input 1 ('key') rank {} unsupported; expected 3 (B, T, D)",
                key.len()
            )));
        }
        if weights.len() != 3 {
            return Err(error(format!(
                "input 2 ('weights') rank {} unsupported; expected 3 (B, S, H)",
                weights.len()
            )));
        }
        if bias.len() != 4 {
            return Err(error(format!(
                "input 3 ('attention_bias') rank {} unsupported; expected 4 (B, 1, S, T)",
                bias.len()
            )));
        }
        let dims = Dims {
            batch: query[0],
            q_seq: query[1],
            heads: query[2],
            head_dim: query[3],
            key_seq: key[1],
        };
        require_eq("query/key batch", query[0], key[0])?;
        require_eq("query/weights batch", query[0], weights[0])?;
        require_eq("query/bias batch", query[0], bias[0])?;
        require_eq("query/weights seq", dims.q_seq, weights[1])?;
        require_eq("query/weights heads", dims.heads, weights[2])?;
        require_eq("query/key head_dim", dims.head_dim, key[2])?;
        if bias[1] != 1 {
            return Err(error(format!(
                "input 3 ('attention_bias') dim 1 must be 1 (head-broadcast), got {}",
                bias[1]
            )));
        }
        require_eq("bias/query seq", dims.q_seq, bias[2])?;
        require_eq("bias/key seq", dims.key_seq, bias[3])?;
        Ok(dims)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_row(
        &self,
        query_ptr: CUdeviceptr,
        key_ptr: CUdeviceptr,
        weights_ptr: CUdeviceptr,
        bias_ptr: CUdeviceptr,
        scores_ptr: CUdeviceptr,
        state_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        dims: Dims,
        dtype: i32,
    ) -> Result<()> {
        let total_rows = (dims.batch * dims.q_seq) as u64;
        if total_rows == 0 {
            return Ok(());
        }
        let func = self
            .runtime
            .nvrtc_function(MODULE, SOURCE, "dsa_index_select_row")?;
        let batch = dims.batch as u64;
        let q_seq = dims.q_seq as u64;
        let heads = dims.heads as u64;
        let head_dim = dims.head_dim as u64;
        let key_seq = dims.key_seq as u64;
        let top_k = self.top_k as u64;
        let scale = self.scale;
        let weights_scale = self.weights_scale;
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&query_ptr)
            .arg(&key_ptr)
            .arg(&weights_ptr)
            .arg(&bias_ptr)
            .arg(&scores_ptr)
            .arg(&state_ptr)
            .arg(&out_ptr)
            .arg(&batch)
            .arg(&q_seq)
            .arg(&heads)
            .arg(&head_dim)
            .arg(&key_seq)
            .arg(&top_k)
            .arg(&scale)
            .arg(&weights_scale)
            .arg(&dtype);
        // SAFETY: argument types/order match `dsa_index_select_row`; all pointers
        // refer to live contiguous device allocations, and the scores/state
        // scratch is sized for `batch*q_seq*key_seq` f32/u8 by
        // `dsa_index_select_workspace_layout`.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (total_rows.min(u32::MAX as u64).max(1) as u32, 1, 1),
                block_dim: (ROW_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch dsa_index_select_row", e))
        .map(|_| ())
    }
}

fn require_eq(what: &str, left: usize, right: usize) -> Result<()> {
    if left != right {
        return Err(error(format!("{what} mismatch: {left} != {right}")));
    }
    Ok(())
}

fn require_floating_dtype(input: &TensorView, index: usize) -> Result<DataType> {
    if !matches!(
        input.dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        return Err(error(format!(
            "input {index} ('{}') dtype {:?} unsupported; expected Float32, Float16, or BFloat16",
            INPUT_NAMES[index], input.dtype
        )));
    }
    Ok(input.dtype)
}

fn dtype_code(dtype: DataType) -> Result<i32> {
    match dtype {
        DataType::Float32 => Ok(0),
        DataType::Float16 => Ok(1),
        DataType::BFloat16 => Ok(2),
        _ => Err(error(format!("unsupported floating dtype {dtype:?}"))),
    }
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

fn required_finite_positive_float(node: &Node, name: &str) -> Result<f32> {
    let value = node
        .attr(name)
        .ok_or_else(|| error(format!("missing required float attribute '{name}'")))?
        .as_float()
        .ok_or_else(|| error(format!("attribute '{name}' must be a float")))?;
    if !value.is_finite() || value <= 0.0 {
        return Err(error(format!("attribute '{name}' must be finite and > 0")));
    }
    Ok(value)
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep {OP}: {}", message.into()))
}
