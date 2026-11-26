//! `pkg.nxrt._private::DsaIndexSelect` v1: device-resident query-dependent sparse index
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
//! Scoring is flattened across `(row, key-position)`, so a decode row uses many
//! CTAs while each key retains the CPU reference's fixed ascending `head_dim`
//! then `heads` reduction order. Selection uses 32 parallel radix-count passes
//! to find the exact kth `f32::total_cmp` key, followed by stable block-scan
//! compaction; no thread performs `top_k` full scans. Final NaNs are
//! canonicalized to positive quiet NaN (`0x7fc00000`) before ordering on both
//! CPU and CUDA, removing backend-specific sign/payload differences after
//! finite overflow. Ties still prefer the lower index, and emitted indices are
//! strictly ascending with `-1` right padding.
//!
//! ## Capture support
//!
//! Scores and allowed-mask storage borrow one caller-funded persistent lane.
//! Metadata preparation reports the checked geometry peak; no kernel-owned
//! arena, allocation counter or resource pin is maintained here. Recording
//! requires a real caller owner pin covering the complete used span, and a
//! successful eager warmup compiles the pipeline before capture. Every score
//! and allow bit read by selection is rewritten on each invocation. The eager
//! completion boundary is preserved; recording never adds a host round trip.
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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut, TensorView,
    WorkspaceLayout, WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node};

const OP: &str = "DsaIndexSelect";

const INPUT_NAMES: [&str; 4] = ["query", "key", "weights", "attention_bias"];

const SCORE_THREADS: u32 = 256;
const SELECT_THREADS: u32 = 256;
const WORKSPACE_ALIGNMENT: usize = 256;

const MODULE: &str = "dsa_index_select_parallel_radix_v2";
const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define NEG_INF __int_as_float(0xff800000)
#define CANONICAL_NAN __int_as_float(0x7fc00000)
#define SELECT_THREADS 256
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

// Canonicalize at the same logical point as the CPU oracle: after the final
// additive score and before total ordering. Finite overflow may otherwise leave
// backend-specific NaN signs/payloads (notably infinity * 0).
__device__ __forceinline__ float canonicalize_score(float score) {
  return isnan(score) ? CANONICAL_NAN : score;
}

// Unsigned key whose ordinary order is Rust `f32::total_cmp` order. The signed
// total-order transform is biased by 0x80000000 so radix selection can compare
// all 32 bits with unsigned arithmetic.
__device__ __forceinline__ unsigned int ordered_total_key(float x) {
  int bits = __float_as_int(x);
  const int sign = bits >> 31;                       // 0 or -1
  const int flip = (int)(((unsigned int)sign) >> 1); // 0 or 0x7fffffff
  return ((unsigned int)(bits ^ flip)) ^ 0x80000000u;
}

// Phase 1 is flattened over every (row, key-position) cell. A realistic decode
// row therefore occupies many CTAs (T / blockDim rather than one CTA), while
// each cell retains the CPU oracle's exact ascending D then H reduction order.
extern "C" __global__ void dsa_index_select_score(
    const void* query, const void* key, const void* weights, const float* bias,
    float* scores, unsigned char* allowed,
    unsigned long long batch, unsigned long long q_seq, unsigned long long heads,
    unsigned long long head_dim, unsigned long long key_seq,
    float scale, float weights_scale, int dtype) {
  const unsigned long long total_rows = batch * q_seq;
  const unsigned long long total_cells = total_rows * key_seq;
  const unsigned long long stride =
      (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;
  for (unsigned long long cell =
           (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
       cell < total_cells;) {
    const unsigned long long row = cell / key_seq;
    const unsigned long long t = cell - row * key_seq;
    const unsigned long long b = row / q_seq;
    const unsigned long long weights_base = row * heads;
    const float bias_bt = bias[cell];
    const bool is_allowed = bias_bt > MASK_THRESHOLD;
    if (!is_allowed) {
      allowed[cell] = 0;
      scores[cell] = NEG_INF;
    } else {
      float weighted = 0.0f;
      for (unsigned long long h = 0; h < heads; ++h) {
        const unsigned long long q_base = (row * heads + h) * head_dim;
        const unsigned long long k_base = (b * key_seq + t) * head_dim;
        float dot = 0.0f;
        for (unsigned long long d = 0; d < head_dim; ++d) {
          // Keep multiply/add un-fused so every cell matches the CPU oracle's
          // fixed reduction and rounding order bit-for-bit.
          dot = __fadd_rn(dot, __fmul_rn(load_float(query, q_base + d, dtype),
                                         load_float(key, k_base + d, dtype)));
        }
        const float scored = fmaxf(scale * dot, 0.0f);
        const float wprod =
            load_float(weights, weights_base + h, dtype) * weights_scale;
        weighted = __fadd_rn(weighted, __fmul_rn(scored, wprod));
      }
      scores[cell] = canonicalize_score(__fadd_rn(weighted, bias_bt));
      allowed[cell] = 1;
    }
    if (stride >= total_cells - cell) {
      break;
    }
    cell += stride;
  }
}

__device__ __forceinline__ unsigned long long block_sum(
    unsigned long long value, unsigned long long* scratch) {
  const unsigned int tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const unsigned long long total = scratch[0];
  __syncthreads();
  return total;
}

__device__ __forceinline__ unsigned long long block_inclusive_scan(
    unsigned long long value, unsigned long long* first,
    unsigned long long* second) {
  const unsigned int tid = threadIdx.x;
  first[tid] = value;
  __syncthreads();
  unsigned long long* src = first;
  unsigned long long* dst = second;
  for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1) {
    unsigned long long sum = src[tid];
    if (tid >= offset) {
      sum += src[tid - offset];
    }
    dst[tid] = sum;
    __syncthreads();
    unsigned long long* swap = src;
    src = dst;
    dst = swap;
  }
  return src[tid];
}

// Phase 2 uses one block per row, but no thread performs top_k full scans.
// Thirty-two parallel radix-count passes find the exact kth total-order key in
// O(32*T/blockDim), then a block scan selects the lowest indices at the
// threshold and compacts every winner in ascending index order.
extern "C" __global__ void dsa_index_select_select(
    const float* scores, const unsigned char* allowed, long long* out,
    unsigned long long total_rows, unsigned long long key_seq,
    unsigned long long top_k) {
  __shared__ unsigned long long scan_a[SELECT_THREADS];
  __shared__ unsigned long long scan_b[SELECT_THREADS];
  __shared__ unsigned long long shared_total;

  const unsigned int tid = threadIdx.x;
  const unsigned long long threads = blockDim.x;
  for (unsigned long long row = blockIdx.x; row < total_rows;) {
    const unsigned long long score_base = row * key_seq;
    const unsigned long long out_base = row * top_k;
    for (unsigned long long slot = tid; slot < top_k; slot += threads) {
      out[out_base + slot] = -1;
    }

    // Contiguous per-thread chunks preserve index order during final compaction
    // and avoid overflow in key_seq * tid.
    const unsigned long long base = key_seq / threads;
    const unsigned long long extra = key_seq % threads;
    const unsigned long long tid64 = tid;
    const unsigned long long begin =
        tid64 * base + (tid64 < extra ? tid64 : extra);
    const unsigned long long end = begin + base + (tid64 < extra ? 1 : 0);

    unsigned long long local_allowed = 0;
    for (unsigned long long t = begin; t < end; ++t) {
      local_allowed += allowed[score_base + t] != 0;
    }
    const unsigned long long allowed_count = block_sum(local_allowed, scan_a);
    const unsigned long long keep =
        allowed_count < top_k ? allowed_count : top_k;
    if (keep != 0) {
      unsigned int prefix = 0;
      unsigned int prefix_mask = 0;
      unsigned long long rank = keep; // one-based rank among descending keys
      for (unsigned int bit = 0x80000000u; bit != 0; bit >>= 1) {
        unsigned long long local_ones = 0;
        for (unsigned long long t = begin; t < end; ++t) {
          if (!allowed[score_base + t]) {
            continue;
          }
          const unsigned int key = ordered_total_key(scores[score_base + t]);
          local_ones +=
              ((key & prefix_mask) == prefix) && ((key & bit) != 0);
        }
        const unsigned long long ones = block_sum(local_ones, scan_a);
        if (ones >= rank) {
          prefix |= bit;
        } else {
          rank -= ones;
        }
        prefix_mask |= bit;
      }
      const unsigned int threshold = prefix;

      unsigned long long local_greater = 0;
      unsigned long long local_equal = 0;
      for (unsigned long long t = begin; t < end; ++t) {
        if (!allowed[score_base + t]) {
          continue;
        }
        const unsigned int key = ordered_total_key(scores[score_base + t]);
        local_greater += key > threshold;
        local_equal += key == threshold;
      }
      const unsigned long long greater_prefix =
          block_inclusive_scan(local_greater, scan_a, scan_b);
      if (tid == blockDim.x - 1) {
        shared_total = greater_prefix;
      }
      __syncthreads();
      const unsigned long long total_greater = shared_total;

      const unsigned long long equal_prefix =
          block_inclusive_scan(local_equal, scan_a, scan_b);
      const unsigned long long equal_before = equal_prefix - local_equal;
      const unsigned long long ties_needed = keep - total_greater;
      const unsigned long long local_ties =
          equal_before >= ties_needed
              ? 0
              : (local_equal < ties_needed - equal_before
                     ? local_equal
                     : ties_needed - equal_before);
      const unsigned long long local_selected = local_greater + local_ties;
      const unsigned long long selected_prefix =
          block_inclusive_scan(local_selected, scan_a, scan_b);
      unsigned long long out_slot = selected_prefix - local_selected;
      unsigned long long equal_seen = 0;
      for (unsigned long long t = begin; t < end; ++t) {
        if (!allowed[score_base + t]) {
          continue;
        }
        const unsigned int key = ordered_total_key(scores[score_base + t]);
        const bool take_greater = key > threshold;
        const bool take_equal = key == threshold && equal_seen < local_ties;
        equal_seen += key == threshold;
        if (take_greater || take_equal) {
          out[out_base + out_slot] = (long long)t;
          out_slot += 1;
        }
      }
    }
    __syncthreads();
    if ((unsigned long long)gridDim.x >= total_rows - row) {
      break;
    }
    row += gridDim.x;
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

fn dsa_index_select_workspace_layout(dims: Dims) -> Result<WorkspaceLayout<2>> {
    let rows = dims
        .batch
        .checked_mul(dims.q_seq)
        .ok_or_else(|| error("workspace row count overflow; reduce batch or query sequence"))?;
    let cells = rows
        .checked_mul(dims.key_seq)
        .ok_or_else(|| error("workspace cell count overflow; reduce sequence dimensions"))?;
    let scores = std::alloc::Layout::array::<f32>(cells).and_then(|layout| layout.align_to(WORKSPACE_ALIGNMENT)).map_err(|_| error("score workspace byte count exceeds addressable capacity; reduce sequence dimensions"))?;
    let allowed =
        std::alloc::Layout::from_size_align(cells, WORKSPACE_ALIGNMENT).map_err(|_| {
            error("allowed-mask workspace exceeds addressable capacity; reduce sequence dimensions")
        })?;
    WorkspaceLayout::new([scores, allowed])
}

fn dsa_workspace_requirement(
    top_k: usize,
    inputs: &[TensorMetadata<'_>],
) -> Result<WorkspaceRequirement> {
    let dims = DsaIndexSelectKernel::validate_metadata(top_k, inputs)?;
    let layout = dsa_index_select_workspace_layout(dims)?;
    Ok(if layout.bytes() == 0 {
        WorkspaceRequirement::NONE
    } else {
        WorkspaceRequirement {
            bytes: layout.bytes() as u64,
            alignment: layout.alignment(),
            lifetime: WorkspaceLifetime::SessionPersistent,
        }
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DsaLaunchStats {
    pub executions: u64,
    pub score_launches: u64,
    pub selection_launches: u64,
    pub last_score_grid_x: u64,
    pub last_selection_grid_x: u64,
}

static DSA_EXECUTIONS: AtomicU64 = AtomicU64::new(0);
static DSA_SCORE_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static DSA_SELECTION_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static DSA_LAST_SCORE_GRID_X: AtomicU64 = AtomicU64::new(0);
static DSA_LAST_SELECTION_GRID_X: AtomicU64 = AtomicU64::new(0);

pub fn dsa_launch_stats() -> DsaLaunchStats {
    DsaLaunchStats {
        executions: DSA_EXECUTIONS.load(Ordering::Relaxed),
        score_launches: DSA_SCORE_LAUNCHES.load(Ordering::Relaxed),
        selection_launches: DSA_SELECTION_LAUNCHES.load(Ordering::Relaxed),
        last_score_grid_x: DSA_LAST_SCORE_GRID_X.load(Ordering::Relaxed),
        last_selection_grid_x: DSA_LAST_SELECTION_GRID_X.load(Ordering::Relaxed),
    }
}

pub fn reset_dsa_launch_stats() {
    DSA_EXECUTIONS.store(0, Ordering::Relaxed);
    DSA_SCORE_LAUNCHES.store(0, Ordering::Relaxed);
    DSA_SELECTION_LAUNCHES.store(0, Ordering::Relaxed);
    DSA_LAST_SCORE_GRID_X.store(0, Ordering::Relaxed);
    DSA_LAST_SELECTION_GRID_X.store(0, Ordering::Relaxed);
}

pub struct DsaIndexSelectFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for DsaIndexSelectFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(self.create_kernel(node)?))
    }
}

impl DsaIndexSelectFactory {
    fn create_kernel(&self, node: &Node) -> Result<DsaIndexSelectKernel> {
        self.create_kernel_with_max_grid_x(node, None)
    }

    #[doc(hidden)]
    pub fn create_kernel_with_grid_limit(
        &self,
        node: &Node,
        max_grid_x: u32,
    ) -> Result<DsaIndexSelectKernel> {
        if max_grid_x == 0 {
            return Err(error("test grid limit must be greater than zero"));
        }
        self.create_kernel_with_max_grid_x(node, Some(max_grid_x))
    }

    fn create_kernel_with_max_grid_x(
        &self,
        node: &Node,
        max_grid_x: Option<u32>,
    ) -> Result<DsaIndexSelectKernel> {
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
        Ok(DsaIndexSelectKernel {
            runtime: self.runtime.clone(),
            top_k,
            scale,
            weights_scale,
            max_grid_x,
            warmed: AtomicBool::new(false),
            executions: AtomicU64::new(0),
            score_launches: AtomicU64::new(0),
            selection_launches: AtomicU64::new(0),
            last_score_grid_x: AtomicU64::new(0),
            last_selection_grid_x: AtomicU64::new(0),
        })
    }
}

#[derive(Debug)]
pub struct DsaIndexSelectKernel {
    runtime: Arc<CudaRuntime>,
    top_k: usize,
    scale: f32,
    weights_scale: f32,
    max_grid_x: Option<u32>,
    /// Set after a successful eager execution has compiled the NVRTC kernel and
    /// used the prepared caller workspace, the precondition for CUDA-graph capture.
    warmed: AtomicBool,
    executions: AtomicU64,
    score_launches: AtomicU64,
    selection_launches: AtomicU64,
    last_score_grid_x: AtomicU64,
    last_selection_grid_x: AtomicU64,
}

impl Kernel for DsaIndexSelectKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.execute_impl(inputs, outputs, None)
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.execute_impl(inputs, outputs, workspace)
    }

    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        _outputs: &[onnx_runtime_ep_api::OutputMetadata<'_>],
    ) -> Result<onnx_runtime_ep_api::WorkspacePreparation> {
        Ok(onnx_runtime_ep_api::WorkspacePreparation::Ready(
            dsa_workspace_requirement(self.top_k, inputs)?,
        ))
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
                 kernel with prepared caller workspace",
            )
        }
    }
}

impl DsaIndexSelectKernel {
    #[doc(hidden)]
    pub fn launch_snapshot(&self) -> DsaLaunchStats {
        DsaLaunchStats {
            executions: self.executions.load(Ordering::Relaxed),
            score_launches: self.score_launches.load(Ordering::Relaxed),
            selection_launches: self.selection_launches.load(Ordering::Relaxed),
            last_score_grid_x: self.last_score_grid_x.load(Ordering::Relaxed),
            last_selection_grid_x: self.last_selection_grid_x.load(Ordering::Relaxed),
        }
    }

    fn execute_impl(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.warmed.store(false, Ordering::Relaxed);
        if inputs.len() != 4 {
            return Err(error(format!("expected 4 inputs, got {}", inputs.len())));
        }
        if outputs.len() != 1 {
            return Err(error(format!("expected 1 output, got {}", outputs.len())));
        }
        let metadata: [_; 4] = std::array::from_fn(|index| {
            TensorMetadata::new(
                inputs[index].dtype,
                inputs[index].shape,
                !inputs[index].is_absent(),
            )
        });
        let dims = Self::validate_metadata(self.top_k, &metadata)?;
        let dtype = inputs[0].dtype;
        if outputs[0].is_absent() {
            return Err(error("required output 0 ('selected_indices') is absent"));
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

        let expected_out = [dims.batch, 1, dims.q_seq, self.top_k];
        if outputs[0].shape != expected_out {
            return Err(error(format!(
                "output 0 ('selected_indices') shape {:?} unsupported; expected {expected_out:?} (B, 1, S, top_k)",
                outputs[0].shape
            )));
        }

        let capturing = self.runtime.is_capturing()?;

        let layout = dsa_index_select_workspace_layout(dims)?;
        let [scores, allowed] = layout.bind(workspace).map_err(|source| EpError::Backend {
            operation: "bind caller workspace for DsaIndexSelect scores and allowed mask",
            source: Box::new(source),
        })?;
        if capturing && layout.bytes() != 0 {
            let used = WorkspaceView::new(
                workspace
                    .expect("nonempty layout binding required backing")
                    .ptr(),
                layout.bytes(),
            );
            self.runtime.require_registered_workspace_capture(
                used,
                "DsaIndexSelect scores and allowed mask",
            )?;
        }

        let query_ptr = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let key_ptr = cuptr(inputs[1].data_ptr::<u8>() as *const c_void);
        let weights_ptr = cuptr(inputs[2].data_ptr::<u8>() as *const c_void);
        let bias_ptr = cuptr(inputs[3].data_ptr::<u8>() as *const c_void);
        let out_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let scores_ptr = cuptr(scores.ptr().0);
        let allowed_ptr = cuptr(allowed.ptr().0);

        self.launch_pipeline(
            query_ptr,
            key_ptr,
            weights_ptr,
            bias_ptr,
            scores_ptr,
            allowed_ptr,
            out_ptr,
            dims,
            dtype_code(dtype)?,
        )?;

        if !capturing {
            // Preserve the existing eager completion boundary. Shared capacity
            // ownership/admission belongs to the caller, not this kernel.
            self.runtime.synchronize()?;
        }
        self.warmed.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn validate_metadata(top_k: usize, inputs: &[TensorMetadata<'_>]) -> Result<Dims> {
        if inputs.len() != 4 {
            return Err(error(format!(
                "expected 4 input metadata entries, got {}",
                inputs.len()
            )));
        }
        for (index, input) in inputs.iter().enumerate() {
            if !input.present {
                return Err(error(format!(
                    "required input {index} ('{}') is absent",
                    INPUT_NAMES[index]
                )));
            }
            if (index < 3
                && !matches!(
                    input.dtype,
                    DataType::Float32 | DataType::Float16 | DataType::BFloat16
                ))
                || (index == 3 && input.dtype != DataType::Float32)
            {
                return Err(error(format!(
                    "input {index} ('{}') has unsupported dtype {:?}; query/key/weights require floating storage and attention_bias requires Float32",
                    INPUT_NAMES[index], input.dtype
                )));
            }
            if index < 3 && input.dtype != inputs[0].dtype {
                return Err(error(format!(
                    "input {index} ('{}') dtype {:?} must match query {:?}",
                    INPUT_NAMES[index], input.dtype, inputs[0].dtype
                )));
            }
        }
        let query = inputs[0].shape;
        let key = inputs[1].shape;
        let weights = inputs[2].shape;
        let bias = inputs[3].shape;
        Self::resolve_dims(top_k, query, key, weights, bias)
    }

    fn resolve_dims(
        top_k: usize,
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
        checked_elements(
            "query",
            &[dims.batch, dims.q_seq, dims.heads, dims.head_dim],
        )?;
        checked_elements("key", &[dims.batch, dims.key_seq, dims.head_dim])?;
        checked_elements("weights", &[dims.batch, dims.q_seq, dims.heads])?;
        checked_elements("attention_bias", &[dims.batch, dims.q_seq, dims.key_seq])?;
        checked_elements("selected_indices", &[dims.batch, dims.q_seq, top_k])?;
        Ok(dims)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_pipeline(
        &self,
        query_ptr: CUdeviceptr,
        key_ptr: CUdeviceptr,
        weights_ptr: CUdeviceptr,
        bias_ptr: CUdeviceptr,
        scores_ptr: CUdeviceptr,
        allowed_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        dims: Dims,
        dtype: i32,
    ) -> Result<()> {
        let total_rows = dims
            .batch
            .checked_mul(dims.q_seq)
            .and_then(|rows| u64::try_from(rows).ok())
            .ok_or_else(|| error("DsaIndexSelect row count overflow"))?;
        if total_rows == 0 {
            return Ok(());
        }
        let score_func = self
            .runtime
            .nvrtc_function(MODULE, SOURCE, "dsa_index_select_score")?;
        let select_func = self
            .runtime
            .nvrtc_function(MODULE, SOURCE, "dsa_index_select_select")?;
        let batch = u64::try_from(dims.batch)
            .map_err(|_| error("DsaIndexSelect batch does not fit u64"))?;
        let q_seq = u64::try_from(dims.q_seq)
            .map_err(|_| error("DsaIndexSelect query sequence does not fit u64"))?;
        let heads = u64::try_from(dims.heads)
            .map_err(|_| error("DsaIndexSelect head count does not fit u64"))?;
        let head_dim = u64::try_from(dims.head_dim)
            .map_err(|_| error("DsaIndexSelect head dimension does not fit u64"))?;
        let key_seq = u64::try_from(dims.key_seq)
            .map_err(|_| error("DsaIndexSelect key sequence does not fit u64"))?;
        let top_k = u64::try_from(self.top_k)
            .map_err(|_| error("DsaIndexSelect top_k does not fit u64"))?;
        let total_cells = total_rows
            .checked_mul(key_seq)
            .ok_or_else(|| error("DsaIndexSelect score cell count does not fit u64"))?;
        let scale = self.scale;
        let weights_scale = self.weights_scale;
        let device_max_grid_x = self.runtime.capabilities().max_grid_dim_x();
        let max_grid_x = self
            .max_grid_x
            .unwrap_or(device_max_grid_x)
            .min(device_max_grid_x);

        if total_cells != 0 {
            let score_threads = u64::from(SCORE_THREADS);
            let score_blocks = (total_cells / score_threads
                + u64::from(total_cells % score_threads != 0))
            .min(u64::from(max_grid_x))
            .max(1) as u32;
            let mut score = self.runtime.stream().launch_builder(&score_func);
            score
                .arg(&query_ptr)
                .arg(&key_ptr)
                .arg(&weights_ptr)
                .arg(&bias_ptr)
                .arg(&scores_ptr)
                .arg(&allowed_ptr)
                .arg(&batch)
                .arg(&q_seq)
                .arg(&heads)
                .arg(&head_dim)
                .arg(&key_seq)
                .arg(&scale)
                .arg(&weights_scale)
                .arg(&dtype);
            // SAFETY: argument order matches `dsa_index_select_score`; score and
            // allowed scratch span `batch*q_seq*key_seq` cells.
            unsafe {
                score.launch(LaunchConfig {
                    grid_dim: (score_blocks, 1, 1),
                    block_dim: (SCORE_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| driver_err("launch dsa_index_select_score", e))?;
            self.score_launches.fetch_add(1, Ordering::Relaxed);
            self.last_score_grid_x
                .store(u64::from(score_blocks), Ordering::Relaxed);
            DSA_SCORE_LAUNCHES.fetch_add(1, Ordering::Relaxed);
            DSA_LAST_SCORE_GRID_X.store(u64::from(score_blocks), Ordering::Relaxed);
        }

        let selection_blocks = total_rows.min(u64::from(max_grid_x)).max(1) as u32;
        let mut select = self.runtime.stream().launch_builder(&select_func);
        select
            .arg(&scores_ptr)
            .arg(&allowed_ptr)
            .arg(&out_ptr)
            .arg(&total_rows)
            .arg(&key_seq)
            .arg(&top_k);
        // SAFETY: argument order matches `dsa_index_select_select`; output spans
        // `total_rows*top_k`, and score/allowed scratch spans all score cells.
        unsafe {
            select.launch(LaunchConfig {
                grid_dim: (selection_blocks, 1, 1),
                block_dim: (SELECT_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch dsa_index_select_select", e))?;
        self.selection_launches.fetch_add(1, Ordering::Relaxed);
        self.last_selection_grid_x
            .store(u64::from(selection_blocks), Ordering::Relaxed);
        self.executions.fetch_add(1, Ordering::Relaxed);
        DSA_SELECTION_LAUNCHES.fetch_add(1, Ordering::Relaxed);
        DSA_LAST_SELECTION_GRID_X.store(u64::from(selection_blocks), Ordering::Relaxed);
        DSA_EXECUTIONS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

fn checked_elements(name: &str, dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |elements, &dim| {
        elements
            .checked_mul(dim)
            .ok_or_else(|| error(format!("{name} element count overflow")))
    })
}

fn require_eq(what: &str, left: usize, right: usize) -> Result<()> {
    if left != right {
        return Err(error(format!("{what} mismatch: {left} != {right}")));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_layout_rejects_row_and_cell_overflow() {
        let row_overflow = dsa_index_select_workspace_layout(Dims {
            batch: usize::MAX,
            q_seq: 2,
            heads: 1,
            head_dim: 1,
            key_seq: 1,
        })
        .expect_err("batch*q_seq overflow must be typed-rejected");
        assert!(format!("{row_overflow}").contains("row count overflow"));

        let cell_overflow = dsa_index_select_workspace_layout(Dims {
            batch: 1,
            q_seq: usize::MAX,
            heads: 1,
            head_dim: 1,
            key_seq: 2,
        })
        .expect_err("rows*key_seq overflow must be typed-rejected");
        assert!(format!("{cell_overflow}").contains("cell count overflow"));
    }

    fn metadata(dtype: DataType) -> [TensorMetadata<'static>; 4] {
        [
            TensorMetadata::new(dtype, &[1, 4, 2, 8], true),
            TensorMetadata::new(dtype, &[1, 12, 8], true),
            TensorMetadata::new(dtype, &[1, 4, 2], true),
            TensorMetadata::new(DataType::Float32, &[1, 1, 4, 12], true),
        ]
    }

    #[test]
    fn workspace_requirement_reports_the_real_shared_peak_for_every_storage_dtype() {
        for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
            let requirement = dsa_workspace_requirement(4, &metadata(dtype)).unwrap();
            assert_eq!((requirement.bytes, requirement.alignment), (304, 256));
            assert_eq!(requirement.lifetime, WorkspaceLifetime::SessionPersistent);
        }
    }

    #[test]
    fn workspace_partitions_scores_and_allowed_without_overlapping_or_owned_storage() {
        let dims =
            DsaIndexSelectKernel::validate_metadata(4, &metadata(DataType::Float32)).unwrap();
        let layout = dsa_index_select_workspace_layout(dims).unwrap();
        let base = onnx_runtime_ep_api::DevicePtrMut(0x1000usize as *mut _);
        let [scores, allowed] = layout.bind(Some(WorkspaceView::new(base, 4096))).unwrap();
        assert_eq!((scores.bytes(), allowed.bytes()), (192, 48));
        assert_eq!(scores.ptr().0 as usize, 0x1000);
        assert_eq!(allowed.ptr().0 as usize, 0x1100);
        for view in [
            None,
            Some(WorkspaceView::new(base, 303)),
            Some(WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(0x1001usize as *mut _),
                304,
            )),
        ] {
            let error = layout.bind(view).unwrap_err().to_string();
            assert!(error.contains("304 bytes aligned to 256"), "{error}");
        }
    }

    #[test]
    fn zero_cells_need_no_scratch_even_when_selection_must_emit_sentinels() {
        for dims in [
            Dims {
                batch: 0,
                q_seq: 4,
                heads: 2,
                head_dim: 8,
                key_seq: 12,
            },
            Dims {
                batch: 1,
                q_seq: 4,
                heads: 2,
                head_dim: 8,
                key_seq: 0,
            },
        ] {
            let layout = dsa_index_select_workspace_layout(dims).unwrap();
            assert_eq!(layout.bytes(), 0);
            layout.bind(None).unwrap();
        }
    }

    #[test]
    fn metadata_requirement_rejects_missing_mixed_and_inconsistent_facts() {
        let mut inputs = metadata(DataType::Float32);
        inputs[0].present = false;
        assert!(
            dsa_workspace_requirement(4, &inputs)
                .unwrap_err()
                .to_string()
                .contains("required input 0")
        );
        inputs = metadata(DataType::BFloat16);
        inputs[3].dtype = DataType::Float16;
        assert!(
            dsa_workspace_requirement(4, &inputs)
                .unwrap_err()
                .to_string()
                .contains("attention_bias")
        );
        inputs = metadata(DataType::BFloat16);
        inputs[1].dtype = DataType::Float32;
        assert!(
            dsa_workspace_requirement(4, &inputs)
                .unwrap_err()
                .to_string()
                .contains("must match query")
        );
        inputs = metadata(DataType::Float32);
        inputs[3].shape = &[1, 1, 4, 13];
        assert!(
            dsa_workspace_requirement(4, &inputs)
                .unwrap_err()
                .to_string()
                .contains("bias/key seq")
        );
    }

    #[test]
    fn tensor_index_spaces_reject_overflow() {
        let error = checked_elements("query", &[usize::MAX, 2])
            .expect_err("tensor element overflow must be typed-rejected");
        assert!(format!("{error}").contains("query element count overflow"));
    }
}
