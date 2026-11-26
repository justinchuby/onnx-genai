//! CUDA implementation of `com.microsoft::MultiHeadAttention` (opset 1).
//!
//! This is a **thin adapter over the shared SDPA core**
//! ([`super::attention::run_attention_blas_softmax`]), exactly mirroring the CPU EP's
//! `multi_head_attention.rs` (the conformance oracle): it parses/validates the
//! separate-Q/K/V MHA inputs, normalizes the Whisper-style layouts, projects the
//! optional `bias`, concatenates any past-KV cache into dense `BNSH` device
//! buffers, folds `attention_bias` and `key_padding_mask` into a single additive
//! mask, then hands the `QKᵀ → scale → +bias → +mask → softmax → ·V` math to the
//! cuBLASLt → softmax → cuBLASLt engine. Every layout transform runs as
//! an NVRTC kernel; only the (small, integer) `key_padding_mask` round-trips
//! through the host, matching ORT's `PrepareMask`.
//!
//! ## Semantics (byte-parity target: ORT `ComputeAttentionProbs`)
//!
//! ```text
//! scores = scale · (Q · Kᵀ)              # scale defaults to 1/sqrt(qk_head_size)
//! scores = scores + attention_bias       # optional additive float bias
//! scores = scores + mask                 # key_padding_mask → mask_filter_value,
//!                                        # causal (unidirectional & S>1) → excluded
//! probs  = softmax(scores, axis=-1)
//! out    = probs · V
//! ```
//!
//! ## Supported layouts
//!
//! * `Q_K_V_BSNH`: query `(B, S, D)`, key `(B, L, D)`, value `(B, L, D_v)`.
//! * `Q_K_V_BSNH_BNSH_BNSH` (cross-attention): query `(B, S, D)`, key/value
//!   already `(B, num_heads, L, H)`. Key/value bias is assumed zero (ORT).
//! * Optional in-op KV cache: `past_key`/`past_value` `(B, num_heads, P, H)` →
//!   `present_key`/`present_value` `(B, num_heads, P+L, H)`.
//!
//! ## Optional inputs (ORT slot order, tested with [`TensorView::is_absent`])
//!
//! `query(0)`, `key(1)`, `value(2)`, `bias(3)`, `key_padding_mask(4)`,
//! `attention_bias(5)`, `past_key(6)`, `past_value(7)`. Slots 8/9
//! (`past_sequence_length`, `cache_indirection`, `DecoderMaskedMHA`) are
//! rejected.
//!
//! ## Declined (clean claim-time / execute-time errors, never a miscompute)
//!
//! * Packed-QKV (rank-5 query) / packed-KV, and DecoderMaskedMHA extras.
//! * `v_head_size != qk_head_size`: the shared cuBLASLt/softmax core assumes a single
//!   head dimension for Q/K/V/O (measured from `run_attention_blas_softmax`, which
//!   sizes the P·V GEMM and the output stride with the Q/K head dim `d`).
//! * `bias` / `attention_bias` / `past_*` whose dtype differs from the query
//!   dtype (the device transforms read them at the query element width).
//!
//! ## Precision note (measured)
//!
//! A *fully* key-padded row (`key_padding_mask` masking every key) adds the
//! `mask_filter_value` (`-10000`) constant to every logit; under softmax the
//! constant cancels, so ORT/the CPU oracle resolve it to `softmax(raw scores)`.
//! The shared cuBLASLt/softmax softmax writes each masked logit (`raw − 10000`) back to
//! the score buffer between its two passes; at f16/bf16 the mantissa step near
//! 10000 is 8, so every value rounds to exactly `-10000` and the row collapses
//! to a uniform distribution. This matches the CPU oracle at f32 but not at
//! reduced precision — an inherent property of the shared core (which GQA and
//! `Attention` also use), not a defect of this adapter.

use std::alloc::Layout;
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, OutputMetadata, Result, TensorMetadata,
    TensorMut, TensorView, WorkspaceLayout, WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node, Shape};

use super::attention::{
    AttentionDtype, BLAS_SOFTMAX_SCRATCH_ALIGN, BlasSoftmaxScratch, blas_softmax_scratch_layout,
    run_attention_blas_softmax,
};
use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

/// Threads per block for the memory-mover NVRTC kernels.
const BLOCK: u32 = 256;

const MODULE: &str = "multi_head_attention_v1";

fn mha_error(detail: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep MultiHeadAttention: {}", detail.into()))
}

fn checked_add(left: usize, right: usize, name: impl std::fmt::Display) -> Result<usize> {
    left.checked_add(right).ok_or_else(|| {
        mha_error(format!(
            "{name} addition {left} + {right} exceeds usize limits"
        ))
    })
}

fn checked_product(factors: &[usize], name: impl std::fmt::Display) -> Result<usize> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or_else(|| mha_error(format!("{name} product {factors:?} exceeds usize limits")))
    })
}

fn checked_bytes(elements: usize, element_size: usize, name: &str) -> Result<usize> {
    let bytes = elements.checked_mul(element_size).ok_or_else(|| {
        mha_error(format!(
            "{name} byte size {elements} * {element_size} exceeds usize limits"
        ))
    })?;
    if bytes > isize::MAX as usize {
        return Err(mha_error(format!(
            "{name} byte size {bytes} exceeds isize::MAX"
        )));
    }
    Ok(bytes)
}

fn checked_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| mha_error(format!("{name}={value} exceeds CUDA i32 limits")))
}

fn checked_grid(count: usize, name: &str) -> Result<u32> {
    let blocks = count.div_ceil(BLOCK as usize);
    u32::try_from(blocks).map_err(|_| {
        mha_error(format!(
            "{name} grid blocks={blocks} for element count {count} exceeds CUDA u32 limits"
        ))
    })
}

fn require_positive(value: usize, name: &str) -> Result<()> {
    if value == 0 {
        return Err(mha_error(format!("{name} must be > 0, got 0")));
    }
    Ok(())
}

fn checked_pointer_offset(base: CUdeviceptr, offset: usize, name: &str) -> Result<CUdeviceptr> {
    let offset = u64::try_from(offset).map_err(|_| {
        mha_error(format!(
            "{name} byte offset {offset} exceeds CUDA u64 limits"
        ))
    })?;
    base.checked_add(offset)
        .ok_or_else(|| mha_error(format!("{name} device pointer offset overflows u64")))
}

fn checked_index2(
    row: usize,
    column: usize,
    width: usize,
    name: impl std::fmt::Display,
) -> Result<usize> {
    checked_add(
        checked_product(&[row, width], format_args!("{name} row offset"))?,
        column,
        name,
    )
}

fn checked_index3(
    plane: usize,
    row: usize,
    column: usize,
    rows: usize,
    columns: usize,
    name: impl std::fmt::Display,
) -> Result<usize> {
    let plane_row = checked_add(
        checked_product(&[plane, rows], format_args!("{name} plane offset"))?,
        row,
        format_args!("{name} row"),
    )?;
    checked_index2(plane_row, column, columns, name)
}

fn validate_static_shape(name: &str, shape: &Shape, dtype: DataType) -> Result<()> {
    if shape.is_empty() {
        return Ok(());
    }
    let mut known_product = 1usize;
    let mut fully_static = true;
    for (axis, dim) in shape.iter().enumerate() {
        match dim.as_static() {
            Some(value) => {
                require_positive(value, &format!("{name} dim {axis}"))?;
                checked_i32(value, &format!("{name} dim {axis}"))?;
                known_product = checked_product(
                    &[known_product, value],
                    format_args!("{name} known-dimension element count"),
                )?;
            }
            None => fully_static = false,
        }
    }
    if fully_static {
        if dtype != DataType::Undefined {
            checked_bytes(known_product, dtype.byte_size(), name)?;
        }
        checked_grid(known_product, &format!("{name} element traversal"))?;
    }
    Ok(())
}

fn synchronize_runtime(runtime: &CudaRuntime) -> Result<()> {
    runtime.synchronize()
}

/// Shape-derived geometry shared by preparation and execution. No tensor data
/// or runtime/device query participates in workspace admission.
#[derive(Clone, Copy, Debug)]
struct MhaGeometry {
    dtype: AttentionDtype,
    batch: usize,
    heads: usize,
    q_seq: usize,
    q_hidden: usize,
    head_size: usize,
    k_is_bnsh: bool,
    v_is_bnsh: bool,
    cur_seq: usize,
    past_seq: usize,
    total_seq: usize,
    q_elements: usize,
    kv_elements: usize,
    mask_elements: usize,
    pad_elements: usize,
    mask_planes: usize,
}

impl MhaGeometry {
    fn from_inputs(heads: usize, inputs: &[TensorMetadata<'_>]) -> Result<Self> {
        require_positive(heads, "num_heads")?;
        let required = |index: usize, name: &str| {
            inputs
                .get(index)
                .filter(|input| input.present)
                .ok_or_else(|| {
                    mha_error(format!(
                        "separate {name} input {index} is required; packed QKV/KV is unsupported"
                    ))
                })
        };
        let q = required(0, "query")?;
        let k = required(1, "key")?;
        let v = required(2, "value")?;
        let dtype = AttentionDtype::from_onnx(q.dtype).map_err(|error| {
            mha_error(format!("query dtype {:?} is unsupported: {error}", q.dtype))
        })?;
        if q.shape.len() != 3 {
            return Err(mha_error(format!(
                "query must be rank 3 (B, S, hidden); packed-QKV layouts (rank {}) are unsupported",
                q.shape.len()
            )));
        }
        if k.shape.len() != v.shape.len() || !matches!(k.shape.len(), 3 | 4) {
            return Err(mha_error(format!(
                "key and value must both be rank 3 (B, L, hidden) or rank 4 (B, N, L, head_size), got key rank {}, value rank {}",
                k.shape.len(),
                v.shape.len()
            )));
        }
        for (name, input) in [("query", q), ("key", k), ("value", v)] {
            if input.dtype != q.dtype {
                return Err(mha_error(format!(
                    "{name} dtype {:?} must match query dtype {:?}",
                    input.dtype, q.dtype
                )));
            }
            for (axis, &dim) in input.shape.iter().enumerate() {
                require_positive(dim, &format!("{name} dim {axis}"))?;
                checked_i32(dim, &format!("{name} dim {axis}"))?;
            }
            checked_bytes(
                checked_product(input.shape, format_args!("{name} element count"))?,
                dtype.element_size() as usize,
                name,
            )?;
        }
        checked_i32(heads, "num_heads")?;
        let (batch, q_seq, q_hidden) = (q.shape[0], q.shape[1], q.shape[2]);
        if !q_hidden.is_multiple_of(heads) {
            return Err(mha_error(format!(
                "query hidden {q_hidden} not divisible by num_heads {heads}"
            )));
        }
        let head_size = q_hidden / heads;
        let kv = |input: &TensorMetadata<'_>, name: &str| -> Result<(bool, usize)> {
            if input.shape[0] != batch {
                return Err(mha_error(format!(
                    "{name} batch {} must equal query batch {batch}",
                    input.shape[0]
                )));
            }
            let (bnsh, seq, dim) = if input.shape.len() == 3 {
                if !input.shape[2].is_multiple_of(heads) {
                    return Err(mha_error(format!(
                        "{name} hidden {} not divisible by num_heads {heads}",
                        input.shape[2]
                    )));
                }
                (false, input.shape[1], input.shape[2] / heads)
            } else {
                if input.shape[1] != heads {
                    return Err(mha_error(format!(
                        "rank-4 {name} dim 1 ({}) must equal num_heads {heads}",
                        input.shape[1]
                    )));
                }
                (true, input.shape[2], input.shape[3])
            };
            if dim != head_size {
                return Err(mha_error(format!(
                    "{name} head_size {dim} must equal query head_size {head_size} (v_head_size must equal qk_head_size for the shared attention core)"
                )));
            }
            Ok((bnsh, seq))
        };
        let (k_is_bnsh, cur_seq) = kv(k, "key")?;
        let (v_is_bnsh, v_seq) = kv(v, "value")?;
        if cur_seq != v_seq {
            return Err(mha_error(format!("key seq {cur_seq} != value seq {v_seq}")));
        }
        let optional = |index: usize| inputs.get(index).filter(|input| input.present);
        if optional(8).is_some() || optional(9).is_some() {
            return Err(mha_error(
                "DecoderMaskedMultiHeadAttention inputs (past_sequence_length / cache_indirection) are not supported",
            ));
        }
        let past_seq = match (optional(6), optional(7)) {
            (None, None) => 0,
            (Some(pk), Some(pv)) => {
                for (name, input) in [("past_key", pk), ("past_value", pv)] {
                    if input.dtype != q.dtype
                        || input.shape.len() != 4
                        || input.shape[0] != batch
                        || input.shape[1] != heads
                        || input.shape[3] != head_size
                    {
                        return Err(mha_error(format!(
                            "{name} must match query dtype {:?} and rank-4 shape [B={batch}, N={heads}, P, H={head_size}], got {:?} {:?}",
                            q.dtype, input.dtype, input.shape
                        )));
                    }
                    for (axis, &dim) in input.shape.iter().enumerate() {
                        require_positive(dim, &format!("{name} dim {axis}"))?;
                        checked_i32(dim, &format!("{name} dim {axis}"))?;
                    }
                    checked_bytes(
                        checked_product(input.shape, format_args!("{name} element count"))?,
                        dtype.element_size() as usize,
                        name,
                    )?;
                }
                if pk.shape[2] != pv.shape[2] {
                    return Err(mha_error("past_key and past_value seq lengths differ"));
                }
                pk.shape[2]
            }
            _ => {
                return Err(mha_error(
                    "past_key and past_value must be provided together",
                ));
            }
        };
        let total_seq = checked_add(past_seq, cur_seq, "past_seq + current key sequence")?;
        checked_i32(total_seq, "total sequence length")?;
        let q_elements = checked_product(&[batch, heads, q_seq, head_size], "query BNSH elements")?;
        let kv_elements = checked_product(
            &[batch, heads, total_seq, head_size],
            "present KV BNSH elements",
        )?;
        let mask_elements =
            checked_product(&[batch, heads, q_seq, total_seq], "attention mask elements")?;
        let pad_elements = checked_product(&[batch, q_seq, total_seq], "padding mask elements")?;
        let mask_planes = checked_product(&[batch, heads], "attention mask planes")?;
        checked_i32(mask_planes, "attention mask planes")?;
        checked_i32(
            checked_product(&[mask_planes, q_seq], "attention rows")?,
            "attention rows",
        )?;
        for (elements, name) in [
            (q_elements, "query BNSH scratch"),
            (kv_elements, "present KV storage"),
            (mask_elements, "attention mask scratch"),
        ] {
            checked_bytes(elements, dtype.element_size() as usize, name)?;
            checked_grid(elements, &format!("{name} launch"))?;
        }
        Ok(Self {
            dtype,
            batch,
            heads,
            q_seq,
            q_hidden,
            head_size,
            k_is_bnsh,
            v_is_bnsh,
            cur_seq,
            past_seq,
            total_seq,
            q_elements,
            kv_elements,
            mask_elements,
            pad_elements,
            mask_planes,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct MhaWorkspace {
    regions: WorkspaceLayout<7>,
    blas_softmax: WorkspaceLayout<2>,
}

impl MhaWorkspace {
    fn new(g: MhaGeometry, present: [bool; 2], pad: bool, bias: bool) -> Result<Self> {
        let elem = g.dtype.element_size() as usize;
        let region = |elements, width, name| {
            let bytes = checked_bytes(elements, width, name)?;
            Layout::from_size_align(bytes, BLAS_SOFTMAX_SCRATCH_ALIGN).map_err(|_| {
                mha_error(format!(
                    "{name} workspace layout exceeds addressable capacity; reduce the input shape"
                ))
            })
        };
        let blas_softmax =
            blas_softmax_scratch_layout(g.dtype, g.batch, g.heads, g.q_seq, g.total_seq).map_err(
                |error| {
                    mha_error(format!(
                        "cuBLASLt/softmax workspace planning failed: {error}"
                    ))
                },
            )?;
        let regions = WorkspaceLayout::new([
            region(g.q_elements, elem, "query BNSH")?,
            region(
                if present[0] { 0 } else { g.kv_elements },
                elem,
                "temporary key",
            )?,
            region(
                if present[1] { 0 } else { g.kv_elements },
                elem,
                "temporary value",
            )?,
            region(g.q_elements, elem, "output BNSH")?,
            region(
                if pad || bias { g.mask_elements } else { 0 },
                elem,
                "additive mask",
            )?,
            region(
                if pad { g.pad_elements } else { 0 },
                std::mem::size_of::<f32>(),
                "padding upload",
            )?,
            Layout::from_size_align(blas_softmax.bytes(), blas_softmax.alignment()).map_err(
                |_| mha_error("cuBLASLt/softmax workspace layout exceeds addressable capacity"),
            )?,
        ])
        .map_err(|error| mha_error(format!("composite workspace planning failed: {error}")))?;
        Ok(Self {
            regions,
            blas_softmax,
        })
    }

    fn requirement(&self) -> Result<WorkspaceRequirement> {
        Ok(WorkspaceRequirement {
            bytes: u64::try_from(self.regions.bytes())
                .map_err(|_| mha_error("workspace bytes exceed u64; reduce the input shape"))?,
            alignment: self.regions.alignment(),
            // Deferred device-binding runs retain the funded peak between steps.
            lifetime: WorkspaceLifetime::SessionPersistent,
        })
    }

    fn bind(&self, workspace: Option<WorkspaceView>) -> Result<[WorkspaceView; 7]> {
        self.regions.bind(workspace).map_err(|error| mha_error(format!(
            "cannot bind prepared workspace: {error}; query workspace_requirement and pass executor-owned backing to execute_with_workspace"
        )))
    }
}

/// Output presence is a compile-time fact; never omit scratch based on an
/// output buffer that a different runtime invocation might fail to supply.
fn present_output_slots(node: &Node) -> Result<[bool; 2]> {
    if !(1..=3).contains(&node.outputs.len()) || node.outputs[0].is_none() {
        return Err(mha_error(
            "declare the required output and at most two optional present KV output slots",
        ));
    }
    Ok(std::array::from_fn(|index| {
        node.outputs.get(index + 1).is_some_and(Option::is_some)
    }))
}

fn check_present_slots(compiled: [bool; 2], actual: [bool; 2]) -> Result<()> {
    if actual != compiled {
        return Err(mha_error(format!(
            "runtime present KV output slots {actual:?} differ from compiled slots {compiled:?}; execute the kernel with its declared positional outputs"
        )));
    }
    Ok(())
}

/// Output facts do not size MHA's exact-shape caches, but they must agree with
/// the declared slots before those buffers can replace temporary KV regions.
fn validate_output_metadata(
    g: MhaGeometry,
    compiled: [bool; 2],
    outputs: &[OutputMetadata<'_>],
) -> Result<()> {
    if outputs.is_empty() {
        return Ok(());
    }
    if outputs.len() > 3 || matches!(outputs[0], OutputMetadata::Absent) {
        return Err(mha_error(
            "output metadata must include the required output and at most two positional present KV slots",
        ));
    }
    let actual = std::array::from_fn(|index| {
        outputs
            .get(index + 1)
            .is_some_and(|fact| !matches!(fact, OutputMetadata::Absent))
    });
    check_present_slots(compiled, actual)?;
    for (index, output) in outputs.iter().enumerate() {
        let (dtype, shape) = match output {
            OutputMetadata::Absent => continue,
            OutputMetadata::Unresolved { dtype } => (*dtype, None),
            OutputMetadata::Tensor { dtype, shape } => (*dtype, Some(shape.as_ref())),
        };
        let expected_dtype = match g.dtype {
            AttentionDtype::F32 => DataType::Float32,
            AttentionDtype::F16 => DataType::Float16,
            AttentionDtype::Bf16 => DataType::BFloat16,
        };
        if dtype != expected_dtype {
            return Err(mha_error(format!(
                "output slot {index} dtype {dtype:?} must match query dtype {expected_dtype:?}"
            )));
        }
        if let Some(shape) = shape {
            let cache = [g.batch, g.heads, g.total_seq, g.head_size];
            let output = [g.batch, g.q_seq, g.q_hidden];
            let expected: &[usize] = if index == 0 { &output } else { &cache };
            if shape != expected {
                return Err(mha_error(format!(
                    "output slot {index} shape {shape:?} must be {expected:?}; provide the exact logical MHA output geometry"
                )));
            }
        }
    }
    Ok(())
}

trait MhaStreamRuntime {
    fn apply_sync_policy(&self) -> Result<()>;
}
impl MhaStreamRuntime for CudaRuntime {
    fn apply_sync_policy(&self) -> Result<()> {
        synchronize_runtime(self)
    }
}

/// Stream-use guard only: runtime policy may defer its sync. An Ok result is
/// not a completion witness. Physical retirement stays with provider fences.
struct MhaStreamUse<'a, R: MhaStreamRuntime + ?Sized> {
    runtime: &'a R,
    active: bool,
}
impl<'a, R: MhaStreamRuntime + ?Sized> MhaStreamUse<'a, R> {
    fn new(runtime: &'a R) -> Self {
        Self {
            runtime,
            active: false,
        }
    }
    fn mark_stream_use(&mut self) {
        self.active = true;
    }
    fn finish(mut self, result: Result<()>) -> Result<()> {
        let completion = if self.active {
            self.active = false;
            self.runtime.apply_sync_policy()
        } else {
            Ok(())
        };
        match (result, completion) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) => Err(error),
            (Ok(()), Err(error)) => Err(mha_error(format!(
                "workspace stream sync policy failed: {error}; settle the failed execution context before retrying"
            ))),
            (Err(error), Err(completion)) => Err(mha_error(format!(
                "execution failed: {error}; workspace stream sync policy also failed: {completion}; settle the failed execution context before retrying"
            ))),
        }
    }
}
impl<R: MhaStreamRuntime + ?Sized> Drop for MhaStreamUse<'_, R> {
    fn drop(&mut self) {
        if self.active
            && let Err(error) = self.runtime.apply_sync_policy()
        {
            eprintln!(
                "cuda_ep MultiHeadAttention: applying stream sync policy during unwinding failed: {error}; backing remains caller/provider owned and needs execution-context settlement"
            );
        }
    }
}

/// NVRTC source: three memory-mover families, templated per storage dtype.
///
/// * `mha_build_bnsh_*` builds a dense `[B, N, past+cur, dim]` buffer from a BSH
///   or BNSH current tensor, an optional past cache prepended along the sequence
///   axis, and an optional per-`(head,dim)` bias — the Q transpose, the K/V
///   transpose, and the KV-cache concat in one pass. Accumulation is `f32`.
/// * `mha_transpose_out_*` writes the `[B, N, S, dim]` attention context back to
///   the operator's `[B, S, N·dim]` BSH output layout.
/// * `mha_build_mask_*` folds the optional `attention_bias` (broadcast over
///   `(B|1, N|1, S, T)`) and the host-resolved additive `key_padding_mask`
///   (`[B, S, T]` f32) into the `[B, N, S, T]` additive mask the softmax reads.
const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define MHA_BUILD_BNSH(NAME, T, LOAD, STORE)                                    \
extern "C" __global__ void NAME(                                               \
    const T* __restrict__ cur, int cur_is_bnsh,                                \
    const T* __restrict__ past, int has_past,                                  \
    const T* __restrict__ bias, int has_bias,                                  \
    T* __restrict__ dst,                                                       \
    int batch, int heads, int cur_seq, int past_seq, int dim) {                \
  const long long total = (long long)past_seq + cur_seq;                       \
  const long long count = (long long)batch * heads * total * dim;              \
  for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;        \
       idx < count; idx += (long long)gridDim.x * blockDim.x) {                 \
    long long x = idx;                                                         \
    const int d = (int)(x % dim); x /= dim;                                     \
    const long long s = x % total; x /= total;                                 \
    const int h = (int)(x % heads); const int b = (int)(x / heads);            \
    float v;                                                                    \
    if (s < past_seq) {                                                        \
      v = LOAD(past[(((long long)b * heads + h) * past_seq + s) * dim + d]);    \
    } else {                                                                    \
      const long long sc = s - past_seq;                                       \
      if (cur_is_bnsh) {                                                       \
        v = LOAD(cur[(((long long)b * heads + h) * cur_seq + sc) * dim + d]);   \
      } else {                                                                  \
        v = LOAD(cur[(((long long)b * cur_seq + sc) * heads + h) * dim + d]);   \
      }                                                                         \
      if (has_bias) v += LOAD(bias[(long long)h * dim + d]);                    \
    }                                                                           \
    dst[idx] = STORE(v);                                                        \
  }                                                                             \
}

#define MHA_TRANSPOSE_OUT(NAME, T)                                              \
extern "C" __global__ void NAME(                                               \
    const T* __restrict__ src, T* __restrict__ dst,                            \
    int batch, int heads, int seq, int dim) {                                  \
  const long long count = (long long)batch * heads * seq * dim;                \
  for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;        \
       idx < count; idx += (long long)gridDim.x * blockDim.x) {                 \
    long long x = idx;                                                         \
    const int d = (int)(x % dim); x /= dim;                                     \
    const int s = (int)(x % seq); x /= seq;                                     \
    const int h = (int)(x % heads); const int b = (int)(x / heads);            \
    dst[(((long long)b * seq + s) * heads + h) * dim + d] = src[idx];           \
  }                                                                             \
}

#define MHA_BUILD_MASK(NAME, T, LOAD, STORE)                                    \
extern "C" __global__ void NAME(                                               \
    T* __restrict__ out,                                                       \
    const T* __restrict__ abias, int has_abias, int abias_d0, int abias_d1,    \
    const float* __restrict__ pad, int has_pad,                                \
    int batch, int heads, int sq, int total) {                                 \
  const long long count = (long long)batch * heads * sq * total;               \
  for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;        \
       idx < count; idx += (long long)gridDim.x * blockDim.x) {                 \
    long long x = idx;                                                         \
    const int j = (int)(x % total); x /= total;                                \
    const int i = (int)(x % sq); x /= sq;                                       \
    const int h = (int)(x % heads); const int b = (int)(x / heads);            \
    float v = 0.0f;                                                            \
    if (has_abias) {                                                           \
      const int b0 = (abias_d0 == 1) ? 0 : b;                                  \
      const int b1 = (abias_d1 == 1) ? 0 : h;                                  \
      v += LOAD(abias[(((long long)b0 * abias_d1 + b1) * sq + i) * total + j]); \
    }                                                                           \
    if (has_pad) v += pad[((long long)b * sq + i) * total + j];                 \
    out[idx] = STORE(v);                                                        \
  }                                                                             \
}

#define MHA_ID(v) (v)
MHA_BUILD_BNSH(mha_build_bnsh_f32, float, MHA_ID, MHA_ID)
MHA_BUILD_BNSH(mha_build_bnsh_f16, __half, __half2float, __float2half_rn)
MHA_BUILD_BNSH(mha_build_bnsh_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16_rn)
MHA_TRANSPOSE_OUT(mha_transpose_out_f32, float)
MHA_TRANSPOSE_OUT(mha_transpose_out_f16, __half)
MHA_TRANSPOSE_OUT(mha_transpose_out_bf16, __nv_bfloat16)
MHA_BUILD_MASK(mha_build_mask_f32, float, MHA_ID, MHA_ID)
MHA_BUILD_MASK(mha_build_mask_f16, __half, __half2float, __float2half_rn)
MHA_BUILD_MASK(mha_build_mask_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16_rn)
"#;

/// Per-dtype NVRTC entry-point stems.
fn stems(dtype: DataType) -> (&'static str, &'static str, &'static str) {
    match dtype {
        DataType::Float32 => (
            "mha_build_bnsh_f32",
            "mha_transpose_out_f32",
            "mha_build_mask_f32",
        ),
        DataType::Float16 => (
            "mha_build_bnsh_f16",
            "mha_transpose_out_f16",
            "mha_build_mask_f16",
        ),
        DataType::BFloat16 => (
            "mha_build_bnsh_bf16",
            "mha_transpose_out_bf16",
            "mha_build_mask_bf16",
        ),
        _ => ("", "", ""),
    }
}

/// `com.microsoft::MultiHeadAttention` kernel carrying the resolved attributes.
#[derive(Debug)]
pub struct MultiHeadAttentionKernel {
    runtime: Arc<CudaRuntime>,
    num_heads: usize,
    /// Explicit score scale; `None` → default `1/sqrt(qk_head_size)`.
    scale: Option<f32>,
    /// Additive fill for padding-masked positions (ORT default `-10000`).
    mask_filter_value: f32,
    /// Apply a causal (lower-triangular) mask when the query length is `> 1`.
    unidirectional: bool,
    present_outputs: [bool; 2],
}

/// Factory for [`MultiHeadAttentionKernel`], reading the contrib-op attributes.
pub struct MultiHeadAttentionFactory {
    pub runtime: Arc<CudaRuntime>,
}

/// Resolved attributes shared by the factory and the claim-time gate.
struct MhaAttributes {
    num_heads: usize,
    scale: Option<f32>,
    mask_filter_value: f32,
    unidirectional: bool,
}

fn attributes_from_node(node: &Node) -> Result<MhaAttributes> {
    let num_heads = node
        .attr("num_heads")
        .and_then(|a| a.as_int())
        .ok_or_else(|| {
            EpError::KernelFailed(
                "cuda_ep MultiHeadAttention: missing required `num_heads` attribute".into(),
            )
        })?;
    if num_heads <= 0 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep MultiHeadAttention: num_heads must be > 0, got {num_heads}"
        )));
    }
    // `> 0` is deliberately broader than ORT's `== 0` sentinel: a zero or
    // negative scale is meaningless and both kernels read it as "use the
    // 1/sqrt(head_size) default" rather than literally zeroing every score.
    let scale = node
        .attr("scale")
        .and_then(|a| a.as_float())
        .filter(|s| *s > 0.0);
    let mask_filter_value = node
        .attr("mask_filter_value")
        .and_then(|a| a.as_float())
        .unwrap_or(-10000.0);
    let unidirectional = node
        .attr("unidirectional")
        .and_then(|a| a.as_int())
        .unwrap_or(0)
        == 1;
    let num_heads = usize::try_from(num_heads).map_err(|_| {
        mha_error(format!(
            "num_heads={num_heads} cannot be represented as usize"
        ))
    })?;
    checked_i32(num_heads, "num_heads")?;
    Ok(MhaAttributes {
        num_heads,
        scale,
        mask_filter_value,
        unidirectional,
    })
}

/// Claim-time capability gate. Declines nodes this EP cannot execute *before*
/// ORT commits them to a fused partition, so a rejection is never raised from
/// `execute` (a hard session failure with no fallback). Only conditions
/// derivable from attributes and the (possibly partial) claim shapes/dtypes are
/// checked; symbolic dims are conservatively accepted and re-validated in
/// `execute`.
pub(crate) fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<String> {
    let attrs = match attributes_from_node(node) {
        Ok(attrs) => attrs,
        Err(e) => return Some(e.to_string()),
    };
    let num_heads = attrs.num_heads;

    let float_ok = |dt: DataType| {
        matches!(
            dt,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        )
    };
    let dtype_at = |i: usize| input_dtypes.get(i).copied().unwrap_or(DataType::Undefined);
    let present = |i: usize| {
        // A supplied optional input has a known (non-Undefined) dtype; ORT hands
        // omitted trailing slots as absent placeholders we never see here.
        dtype_at(i) != DataType::Undefined
    };
    let shape_at = |i: usize| shapes.get(i).map(Vec::as_slice).unwrap_or(&[][..]);

    for (slot, name) in [
        (0usize, "query"),
        (1, "key"),
        (2, "value"),
        (3, "bias"),
        (4, "key_padding_mask"),
        (5, "attention_bias"),
        (6, "past_key"),
        (7, "past_value"),
    ] {
        if let Some(shape) = shapes.get(slot)
            && let Err(error) = validate_static_shape(name, shape, dtype_at(slot))
        {
            return Some(error.to_string());
        }
    }

    let q_dtype = dtype_at(0);
    if q_dtype != DataType::Undefined && !float_ok(q_dtype) {
        return Some(format!(
            "cuda_ep MultiHeadAttention: query dtype {q_dtype:?} not supported (expected f32/f16/bf16)"
        ));
    }
    // Q/K/V and every float side input share the query element width.
    for (slot, name) in [
        (1usize, "key"),
        (2, "value"),
        (3, "bias"),
        (5, "attention_bias"),
        (6, "past_key"),
        (7, "past_value"),
    ] {
        if present(slot) && q_dtype != DataType::Undefined && dtype_at(slot) != q_dtype {
            return Some(format!(
                "cuda_ep MultiHeadAttention: {name} dtype {:?} must match query dtype {q_dtype:?}",
                dtype_at(slot)
            ));
        }
    }
    if present(4) && !matches!(dtype_at(4), DataType::Int32 | DataType::Int64) {
        return Some(format!(
            "cuda_ep MultiHeadAttention: key_padding_mask dtype {:?} must be int32 or int64",
            dtype_at(4)
        ));
    }
    // DecoderMaskedMHA extras (slots 8/9) are out of scope.
    if present(8) || present(9) {
        return Some(
            "cuda_ep MultiHeadAttention: DecoderMaskedMultiHeadAttention inputs (past_sequence_length / cache_indirection) are not supported".into(),
        );
    }

    let q = shape_at(0);
    if !q.is_empty() && q.len() != 3 {
        return Some(format!(
            "cuda_ep MultiHeadAttention: query must be rank 3 (B, S, hidden); packed-QKV layouts (rank {}) are unsupported",
            q.len()
        ));
    }
    // Packed-KV supplies a rank-5 `key` and omits `value` entirely. Both halves
    // of that shape must be declined *here*: a rejection raised from `execute`
    // arrives after ORT has already compiled the node onto this EP, which is a
    // hard session failure that no fallback recovers from. The two checks below
    // mirror the execute-time guards so the pair cannot disagree.
    if present(1) && !present(2) {
        return Some(
            "cuda_ep MultiHeadAttention: value is required when key is present; packed-KV layouts are unsupported".into(),
        );
    }
    for (slot, name) in [(1usize, "key"), (2, "value")] {
        let rank = shape_at(slot).len();
        // Rank zero is the IR's current representation for "no static shape
        // recorded"; every non-zero unsupported rank is concrete and declined.
        if rank != 0 && !matches!(rank, 3 | 4) {
            return Some(format!(
                "cuda_ep MultiHeadAttention: {name} must be rank 3 (B, T, hidden) or rank 4 (B, N, T, H); rank {rank} is unsupported"
            ));
        }
    }
    // Static extent of `shape[axis]`, or `None` when symbolic / out of range.
    let dim =
        |shape: &[onnx_runtime_ir::Dim], axis: usize| shape.get(axis).and_then(|d| d.as_static());
    if let Some(q_hidden) = dim(q, 2)
        && !q_hidden.is_multiple_of(num_heads)
    {
        return Some(format!(
            "cuda_ep MultiHeadAttention: query hidden {q_hidden} not divisible by num_heads {num_heads}"
        ));
    }
    for (slot, name) in [(1usize, "key"), (2, "value")] {
        let shape = shape_at(slot);
        if shape.len() == 3
            && let Some(hidden) = dim(shape, 2)
            && !hidden.is_multiple_of(num_heads)
        {
            return Some(format!(
                "cuda_ep MultiHeadAttention: {name} hidden {hidden} not divisible by num_heads {num_heads}"
            ));
        }
        if shape.len() == 4
            && let Some(heads) = dim(shape, 1)
            && heads != num_heads
        {
            return Some(format!(
                "cuda_ep MultiHeadAttention: rank-4 {name} dim 1 ({heads}) must equal num_heads {num_heads}"
            ));
        }
    }
    // Prove the equal-head-size requirement only when every relevant dim is
    // concretely known; a symbolic dim is conservatively accepted here and
    // re-validated in `execute`.
    if let Some(q_hidden) = dim(q, 2)
        && num_heads != 0
        && q_hidden % num_heads == 0
    {
        let head_size = q_hidden / num_heads;
        let key = shape_at(1);
        let value = shape_at(2);
        let v_head_size = match value.len() {
            3 => dim(value, 2)
                .filter(|h| h % num_heads == 0)
                .map(|h| h / num_heads),
            4 => dim(value, 3),
            _ => None,
        };
        if let Some(v_head_size) = v_head_size
            && v_head_size != head_size
        {
            return Some(format!(
                "cuda_ep MultiHeadAttention: v_head_size {v_head_size} must equal qk_head_size {head_size} (the shared attention core assumes one head dimension)"
            ));
        }
        if key.len() == 4
            && let Some(k_head) = dim(key, 3)
            && k_head != head_size
        {
            return Some(format!(
                "cuda_ep MultiHeadAttention: key head_size {k_head} must equal query head_size {head_size}"
            ));
        }
        for (shape, name) in [(shape_at(6), "past_key"), (shape_at(7), "past_value")] {
            if shape.len() == 4
                && let Some(past_head) = dim(shape, 3)
                && past_head != head_size
            {
                return Some(format!(
                    "cuda_ep MultiHeadAttention: {name} head_size {past_head} must equal query head_size {head_size}"
                ));
            }
        }
    }
    let key = shape_at(1);
    let value = shape_at(2);
    let key_seq = match key.len() {
        3 => dim(key, 1),
        4 => dim(key, 2),
        _ => None,
    };
    let value_seq = match value.len() {
        3 => dim(value, 1),
        4 => dim(value, 2),
        _ => None,
    };
    if let (Some(key_seq), Some(value_seq)) = (key_seq, value_seq)
        && key_seq != value_seq
    {
        return Some(format!(
            "cuda_ep MultiHeadAttention: key seq {key_seq} != value seq {value_seq}"
        ));
    }
    if let Some(q_batch) = dim(q, 0) {
        for (slot, name) in [
            (1usize, "key"),
            (2, "value"),
            (6, "past_key"),
            (7, "past_value"),
        ] {
            let shape = shape_at(slot);
            if !shape.is_empty()
                && let Some(input_batch) = dim(shape, 0)
                && input_batch != q_batch
            {
                return Some(format!(
                    "cuda_ep MultiHeadAttention: {name} batch {input_batch} must equal query batch {q_batch}"
                ));
            }
        }
    }
    let past_key = shape_at(6);
    let past_value = shape_at(7);
    if !past_key.is_empty() != !past_value.is_empty() {
        return Some(
            "cuda_ep MultiHeadAttention: past_key and past_value must be provided together".into(),
        );
    }
    for (shape, name) in [(past_key, "past_key"), (past_value, "past_value")] {
        if !shape.is_empty() && shape.len() != 4 {
            return Some(format!(
                "cuda_ep MultiHeadAttention: {name} must be rank 4 (B, N, P, H), got rank {}",
                shape.len()
            ));
        }
        if shape.len() == 4
            && let Some(heads) = dim(shape, 1)
            && heads != num_heads
        {
            return Some(format!(
                "cuda_ep MultiHeadAttention: {name} dim 1 ({heads}) must equal num_heads {num_heads}"
            ));
        }
    }
    if let (Some(pk_seq), Some(pv_seq)) = (dim(past_key, 2), dim(past_value, 2))
        && pk_seq != pv_seq
    {
        return Some(format!(
            "cuda_ep MultiHeadAttention: past_key seq {pk_seq} != past_value seq {pv_seq}"
        ));
    }
    let past_seq = dim(past_key, 2).unwrap_or(0);
    let total_seq = match key_seq {
        Some(cur_seq) => match checked_add(past_seq, cur_seq, "past_seq + current key seq") {
            Ok(total) => {
                if let Err(error) = checked_i32(total, "total sequence length") {
                    return Some(error.to_string());
                }
                Some(total)
            }
            Err(error) => return Some(error.to_string()),
        },
        None => None,
    };
    let bias = shape_at(3);
    if !bias.is_empty() && bias.len() != 1 {
        return Some(format!(
            "cuda_ep MultiHeadAttention: bias must be rank 1, got rank {}",
            bias.len()
        ));
    }
    if let (Some(actual), Some(q_hidden)) = (dim(bias, 0), dim(q, 2)) {
        let expected = match checked_add(
            match checked_product(&[2, q_hidden], "bias Q/K element count") {
                Ok(value) => value,
                Err(error) => return Some(error.to_string()),
            },
            q_hidden,
            "bias Q/K/V element count",
        ) {
            Ok(value) => value,
            Err(error) => return Some(error.to_string()),
        };
        if actual != expected {
            return Some(format!(
                "cuda_ep MultiHeadAttention: bias length {actual} must equal 3*query hidden = {expected}"
            ));
        }
    }
    let attention_bias = shape_at(5);
    if !attention_bias.is_empty() && attention_bias.len() != 4 {
        return Some(format!(
            "cuda_ep MultiHeadAttention: attention_bias must be rank 4, got rank {}",
            attention_bias.len()
        ));
    }
    if attention_bias.len() == 4
        && let (Some(q_batch), Some(q_seq), Some(total_seq)) = (dim(q, 0), dim(q, 1), total_seq)
    {
        let d0 = dim(attention_bias, 0);
        let d1 = dim(attention_bias, 1);
        let d2 = dim(attention_bias, 2);
        let d3 = dim(attention_bias, 3);
        if d0.is_some_and(|value| value != 1 && value != q_batch)
            || d1.is_some_and(|value| value != 1 && value != num_heads)
            || d2.is_some_and(|value| value != q_seq)
            || d3.is_some_and(|value| value != total_seq)
        {
            return Some(format!(
                "cuda_ep MultiHeadAttention: attention_bias static shape {attention_bias:?} is incompatible with B={q_batch}, N={num_heads}, S={q_seq}, T={total_seq}"
            ));
        }
    }
    let padding_mask = shape_at(4);
    if !padding_mask.is_empty() && !matches!(padding_mask.len(), 1..=3) {
        return Some(format!(
            "cuda_ep MultiHeadAttention: key_padding_mask must be rank 1, 2, or 3, got rank {}",
            padding_mask.len()
        ));
    }
    if !padding_mask.is_empty()
        && let (Some(q_batch), Some(q_seq), Some(total_seq)) = (dim(q, 0), dim(q, 1), total_seq)
    {
        let matches = match padding_mask.len() {
            1 => dim(padding_mask, 0).is_none_or(|length| {
                let packed = checked_product(&[3, q_batch], "key_padding_mask 3B")
                    .and_then(|value| checked_add(value, 2, "key_padding_mask 3B+2"));
                length == q_batch || packed.is_ok_and(|packed| length == packed)
            }),
            2 => {
                dim(padding_mask, 0).is_none_or(|value| value == q_batch)
                    && dim(padding_mask, 1).is_none_or(|value| value == total_seq)
            }
            3 => {
                dim(padding_mask, 0).is_none_or(|value| value == q_batch)
                    && dim(padding_mask, 1).is_none_or(|value| value == q_seq)
                    && dim(padding_mask, 2).is_none_or(|value| value == total_seq)
            }
            _ => false,
        };
        if !matches {
            return Some(format!(
                "cuda_ep MultiHeadAttention: key_padding_mask static shape {padding_mask:?} is incompatible with B={q_batch}, S={q_seq}, T={total_seq}"
            ));
        }
    }
    None
}

impl KernelFactory for MultiHeadAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let attrs = attributes_from_node(node)?;
        Ok(Box::new(MultiHeadAttentionKernel {
            runtime: self.runtime.clone(),
            num_heads: attrs.num_heads,
            scale: attrs.scale,
            mask_filter_value: attrs.mask_filter_value,
            unidirectional: attrs.unidirectional,
            present_outputs: present_output_slots(node)?,
        }))
    }
}

/// Launch one memory-mover NVRTC entry (`MODULE`/`SOURCE`) over `count` linear
/// elements. A macro rather than a method so the caller supplies the argument
/// ABI inline without naming cudarc's launch-builder type.
macro_rules! mha_launch_1d {
    ($self:expr, $entry:expr, $count:expr, $builder:ident, $args:block) => {{
        let count: usize = $count;
        if count != 0 {
            let function = $self.runtime.nvrtc_function(MODULE, SOURCE, $entry)?;
            let grid = checked_grid(count, $entry)?.max(1);
            let mut $builder = $self.runtime.stream().launch_builder(&function);
            $args
            // SAFETY: each caller supplies the exact argument ABI for `$entry`;
            // every buffer is a live contiguous device allocation sized for
            // `count`, and the kernels use no shared memory.
            unsafe {
                $builder.launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| driver_err(&format!("launch {}", $entry), e))?;
        }
    }};
}

impl MultiHeadAttentionKernel {
    /// Build a dense `[B, N, past+cur, dim]` device buffer (`build_bnsh` entry).
    #[allow(clippy::too_many_arguments)]
    fn build_bnsh(
        &self,
        entry: &str,
        cur_ptr: CUdeviceptr,
        cur_is_bnsh: bool,
        past_ptr: CUdeviceptr,
        has_past: bool,
        bias_ptr: CUdeviceptr,
        has_bias: bool,
        dst: CUdeviceptr,
        batch: usize,
        heads: usize,
        cur_seq: usize,
        past_seq: usize,
        dim: usize,
    ) -> Result<()> {
        let (batch_i, heads_i, cur_i, past_i, dim_i) = (
            checked_i32(batch, "batch")?,
            checked_i32(heads, "num_heads")?,
            checked_i32(cur_seq, "current sequence length")?,
            checked_i32(past_seq, "past sequence length")?,
            checked_i32(dim, "head_size")?,
        );
        let cur_is_bnsh_i = i32::from(cur_is_bnsh);
        let has_past_i = i32::from(has_past);
        let has_bias_i = i32::from(has_bias);
        let total_seq = checked_add(past_seq, cur_seq, "past_seq + current sequence length")?;
        let count = checked_product(&[batch, heads, total_seq, dim], "BNSH build element count")?;
        mha_launch_1d!(self, entry, count, builder, {
            builder
                .arg(&cur_ptr)
                .arg(&cur_is_bnsh_i)
                .arg(&past_ptr)
                .arg(&has_past_i)
                .arg(&bias_ptr)
                .arg(&has_bias_i)
                .arg(&dst)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&cur_i)
                .arg(&past_i)
                .arg(&dim_i);
        });
        Ok(())
    }
}

impl MultiHeadAttentionKernel {
    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        if inputs.len() < 3 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MultiHeadAttention: expected at least query/key/value, got {} inputs",
                inputs.len()
            )));
        }
        if outputs.is_empty() || outputs[0].is_absent() {
            return Err(EpError::KernelFailed(
                "cuda_ep MultiHeadAttention: missing output".into(),
            ));
        }
        let optional = |i: usize| inputs.get(i).filter(|t| !t.is_absent());
        if optional(8).is_some() || optional(9).is_some() {
            return Err(EpError::KernelFailed(
                "cuda_ep MultiHeadAttention: DecoderMaskedMultiHeadAttention inputs (past_sequence_length / cache_indirection) are not supported".into(),
            ));
        }

        let query = &inputs[0];
        let key = &inputs[1];
        let value = &inputs[2];
        let metadata: [_; 8] = std::array::from_fn(|index| match inputs.get(index) {
            Some(input) => TensorMetadata::new(input.dtype, input.shape, !input.is_absent()),
            None => TensorMetadata::new(DataType::Undefined, &[], false),
        });
        let geometry = MhaGeometry::from_inputs(self.num_heads, &metadata)?;
        let MhaGeometry {
            batch,
            heads: num_heads,
            q_seq,
            q_hidden,
            head_size,
            k_is_bnsh,
            v_is_bnsh,
            cur_seq,
            past_seq,
            total_seq,
            q_elements,
            kv_elements: _,
            mask_elements,
            pad_elements: _,
            mask_planes,
            dtype: attn_dtype,
        } = geometry;
        let dtype = query.dtype;
        let elem = attn_dtype.element_size() as usize;
        let v_head_size = head_size;
        let v_hidden = q_hidden;
        let is_cross_bnsh = k_is_bnsh;
        if dtype != DataType::Float32 {
            self.runtime
                .require_nvrtc_half_headers("MultiHeadAttention")?;
        }
        for (name, input) in [("query", query), ("key", key), ("value", value)] {
            if !input.is_contiguous() {
                return Err(mha_error(format!("{name} must be contiguous")));
            }
        }

        // Optional bias `(D + D + D_v)` split into per-projection slices.
        let bias_in = optional(3);
        if let Some(bias) = bias_in {
            if bias.dtype != dtype {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: bias dtype {:?} must match query dtype {dtype:?}",
                    bias.dtype
                )));
            }
            if bias.shape.len() != 1 {
                return Err(mha_error(format!(
                    "bias must be rank 1, got rank {}",
                    bias.shape.len()
                )));
            }
            let expected = checked_add(
                checked_product(&[2, q_hidden], "bias Q/K element count")?,
                v_hidden,
                "bias Q/K/V element count",
            )?;
            let bias_elements = checked_product(bias.shape, "bias element count")?;
            checked_bytes(bias_elements, elem, "bias")?;
            if bias_elements != expected {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: bias length {bias_elements} must equal 2*hidden + v_hidden = {expected}"
                )));
            }
            if !bias.is_contiguous() {
                return Err(EpError::KernelFailed(
                    "cuda_ep MultiHeadAttention: bias must be contiguous".into(),
                ));
            }
        }
        let bias_base = bias_in.map(|b| cuptr(b.data_ptr::<u8>() as *const c_void));
        // In the rank-4 (BNSH) key/value layout ORT applies only the query bias.
        let (q_bias, k_bias, v_bias) = match bias_base {
            None => (0, 0, 0),
            Some(base) => {
                let q_b = base;
                if is_cross_bnsh {
                    (q_b, 0, 0)
                } else {
                    let k_offset = checked_bytes(q_hidden, elem, "key bias offset")?;
                    let v_offset = checked_bytes(
                        checked_product(&[2, q_hidden], "value bias offset elements")?,
                        elem,
                        "value bias offset",
                    )?;
                    let k_b = checked_pointer_offset(base, k_offset, "key bias")?;
                    let v_b = checked_pointer_offset(base, v_offset, "value bias")?;
                    (q_b, k_b, v_b)
                }
            }
        };

        // Optional in-op KV cache (inputs 6 and 7), rank-4 (B, N, P, H).
        let past_key = optional(6);
        let past_value = optional(7);
        for (name, past) in [("past_key", past_key), ("past_value", past_value)] {
            if let Some(past) = past
                && !past.is_contiguous()
            {
                return Err(mha_error(format!("{name} must be contiguous")));
            }
        }

        // present_key / present_value outputs share the built cache buffers, so
        // build them directly into the output allocations when requested.
        let want_present_k = outputs.len() >= 2 && !outputs[1].is_absent();
        let want_present_v = outputs.len() >= 3 && !outputs[2].is_absent();
        if outputs.len() > 3 {
            return Err(mha_error("expected at most three positional outputs"));
        }
        let output_facts: [_; 3] = std::array::from_fn(|index| {
            outputs
                .get(index)
                .map_or(OutputMetadata::Absent, OutputMetadata::from_view)
        });
        validate_output_metadata(
            geometry,
            self.present_outputs,
            &output_facts[..outputs.len()],
        )?;
        for (index, name) in [(0usize, "output"), (1, "present_key"), (2, "present_value")] {
            if let Some(output) = outputs.get(index)
                && !output.is_absent()
                && !output.is_contiguous()
            {
                return Err(mha_error(format!("{name} must be contiguous")));
            }
        }
        let output_elements = q_elements;
        checked_bytes(
            checked_product(outputs[0].shape, "output element count")?,
            elem,
            "output",
        )?;
        if want_present_k {
            checked_bytes(
                checked_product(outputs[1].shape, "present_key output element count")?,
                elem,
                "present_key output",
            )?;
        }
        if want_present_v {
            checked_bytes(
                checked_product(outputs[2].shape, "present_value output element count")?,
                elem,
                "present_value output",
            )?;
        }
        checked_grid(mask_elements, "attention mask launch")?;

        let layout = MhaWorkspace::new(
            geometry,
            self.present_outputs,
            optional(4).is_some(),
            optional(5).is_some(),
        )?;
        let [
            q_region,
            k_region,
            v_region,
            o_region,
            mask_region,
            pad_region,
            blas_softmax_region,
        ] = layout.bind(workspace)?;
        let blas_softmax_scratch =
            BlasSoftmaxScratch::bind(&layout.blas_softmax, Some(blas_softmax_region))?;

        // Resolve key_padding_mask on the host (small, integer) exactly like
        // ORT's PrepareMask, into an additive [B, S, total] f32 buffer.
        let pad_additive = match optional(4) {
            Some(m) => Some(self.resolve_pad_mask(m, batch, q_seq, total_seq)?),
            None => None,
        };

        // Optional attention_bias (input 5): additive `(B|1, N|1, S, T)`.
        let attn_bias = optional(5);
        let (abias_d0, abias_d1) = if let Some(m) = attn_bias {
            if m.dtype != dtype {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: attention_bias dtype {:?} must match query dtype {dtype:?}",
                    m.dtype
                )));
            }
            if m.shape.len() != 4 {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: attention_bias must be rank 4 (B|1, N|1, S, T), got rank {}",
                    m.shape.len()
                )));
            }
            let (d0, d1, d2, d3) = (m.shape[0], m.shape[1], m.shape[2], m.shape[3]);
            if !(d0 == batch || d0 == 1)
                || !(d1 == num_heads || d1 == 1)
                || d2 != q_seq
                || d3 != total_seq
            {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: attention_bias shape {:?} incompatible with (B|1={batch}, N|1={num_heads}, S={q_seq}, T={total_seq})",
                    m.shape
                )));
            }
            if !m.is_contiguous() {
                return Err(EpError::KernelFailed(
                    "cuda_ep MultiHeadAttention: attention_bias must be contiguous".into(),
                ));
            }
            checked_bytes(
                checked_product(m.shape, "attention_bias element count")?,
                elem,
                "attention_bias",
            )?;
            checked_i32(d0, "attention_bias dim 0")?;
            checked_i32(d1, "attention_bias dim 1")?;
            (d0, d1)
        } else {
            (0, 0)
        };

        let scale = self.scale.unwrap_or(1.0 / (head_size as f32).sqrt());
        // ORT: causal only when unidirectional AND the query spans > 1 token (an
        // incremental decode step attends the whole cache).
        let causal = self.unidirectional && q_seq > 1;

        let (build_entry, transpose_entry, mask_entry) = stems(dtype);
        let want_mask = pad_additive.is_some() || attn_bias.is_some();

        let mut submission = MhaStreamUse::new(self.runtime.as_ref());
        let q_bnsh = cuptr(q_region.ptr().0.cast_const());
        let present_k = if want_present_k {
            cuptr(outputs[1].data_ptr_mut::<u8>() as *const c_void)
        } else {
            cuptr(k_region.ptr().0.cast_const())
        };
        let present_v = if want_present_v {
            cuptr(outputs[2].data_ptr_mut::<u8>() as *const c_void)
        } else {
            cuptr(v_region.ptr().0.cast_const())
        };
        let o_bnsh = cuptr(o_region.ptr().0.cast_const());
        let mask_ptr = if want_mask {
            cuptr(mask_region.ptr().0.cast_const())
        } else {
            0
        };
        let pad_ptr = if pad_additive.is_some() {
            cuptr(pad_region.ptr().0.cast_const())
        } else {
            0
        };
        let abias_ptr = attn_bias
            .map(|m| cuptr(m.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);

        let q_base = cuptr(query.data_ptr::<u8>() as *const c_void);
        let k_base = cuptr(key.data_ptr::<u8>() as *const c_void);
        let v_base = cuptr(value.data_ptr::<u8>() as *const c_void);
        let past_k_base = past_key
            .map(|p| cuptr(p.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let past_v_base = past_value
            .map(|p| cuptr(p.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let o_out = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        submission.mark_stream_use();
        let result = (|| -> Result<()> {
            if let Some(padding) = &pad_additive {
                let bytes = bytemuck_f32_bytes(padding);
                // SAFETY: the checked region covers all uploaded bytes. htod
                // stages the source synchronously; unsettled destination DMA
                // is latched for default-stream fencing by provider retirement.
                unsafe { self.runtime.htod(&bytes, pad_ptr) }?;
            }
            self.build_bnsh(
                build_entry,
                q_base,
                false,
                0,
                false,
                q_bias,
                q_bias != 0,
                q_bnsh,
                batch,
                num_heads,
                q_seq,
                0,
                head_size,
            )?;
            self.build_bnsh(
                build_entry,
                k_base,
                k_is_bnsh,
                past_k_base,
                past_key.is_some(),
                k_bias,
                k_bias != 0,
                present_k,
                batch,
                num_heads,
                cur_seq,
                past_seq,
                head_size,
            )?;
            self.build_bnsh(
                build_entry,
                v_base,
                v_is_bnsh,
                past_v_base,
                past_value.is_some(),
                v_bias,
                v_bias != 0,
                present_v,
                batch,
                num_heads,
                cur_seq,
                past_seq,
                v_head_size,
            )?;

            let mask_planes_i = if want_mask {
                let (batch_i, heads_i, sq_i, total_i) = (
                    checked_i32(batch, "batch")?,
                    checked_i32(num_heads, "num_heads")?,
                    checked_i32(q_seq, "query sequence length")?,
                    checked_i32(total_seq, "total sequence length")?,
                );
                let has_abias_i = i32::from(attn_bias.is_some());
                let abias_d0_i = checked_i32(abias_d0, "attention_bias dim 0")?;
                let abias_d1_i = checked_i32(abias_d1, "attention_bias dim 1")?;
                let has_pad_i = i32::from(pad_additive.is_some());
                mha_launch_1d!(self, mask_entry, mask_elements, builder, {
                    builder
                        .arg(&mask_ptr)
                        .arg(&abias_ptr)
                        .arg(&has_abias_i)
                        .arg(&abias_d0_i)
                        .arg(&abias_d1_i)
                        .arg(&pad_ptr)
                        .arg(&has_pad_i)
                        .arg(&batch_i)
                        .arg(&heads_i)
                        .arg(&sq_i)
                        .arg(&total_i);
                });
                checked_i32(mask_planes, "attention mask planes")?
            } else {
                0
            };

            // The shared core can fail after submitting an earlier phase, so arm
            // stream-ordered cleanup before entering it.
            run_attention_blas_softmax(
                &self.runtime,
                attn_dtype,
                num_heads,
                num_heads,
                causal,
                batch,
                q_seq,
                total_seq,
                head_size,
                total_seq,
                1,
                scale,
                q_bnsh,
                present_k,
                present_v,
                o_bnsh,
                mask_ptr,
                mask_planes_i,
                0,
                0,
                0,
                0.0,
                blas_softmax_scratch,
            )?;

            let (batch_i, heads_i, sq_i, dim_i) = (
                checked_i32(batch, "batch")?,
                checked_i32(num_heads, "num_heads")?,
                checked_i32(q_seq, "query sequence length")?,
                checked_i32(head_size, "head_size")?,
            );
            mha_launch_1d!(self, transpose_entry, output_elements, builder, {
                builder
                    .arg(&o_bnsh)
                    .arg(&o_out)
                    .arg(&batch_i)
                    .arg(&heads_i)
                    .arg(&sq_i)
                    .arg(&dim_i);
            });

            Ok(())
        })();
        submission.finish(result)
    }
}

impl Kernel for MultiHeadAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<onnx_runtime_ep_api::WorkspacePreparation> {
        let geometry = MhaGeometry::from_inputs(self.num_heads, inputs)?;
        validate_output_metadata(geometry, self.present_outputs, outputs)?;
        let present = |index: usize| inputs.get(index).is_some_and(|input| input.present);
        MhaWorkspace::new(geometry, self.present_outputs, present(4), present(5))?
            .requirement()
            .map(onnx_runtime_ep_api::WorkspacePreparation::Ready)
    }
    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.run(inputs, outputs, workspace)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        // Contiguity is validated per input; a strided view returns an actionable
        // error rather than being silently mis-read.
        true
    }

    fn capture_support(&self) -> CaptureSupport {
        // The cuBLASLt/softmax core and final output transpose require stream completion.
        // Prepared backing does not make these synchronization points capturable.
        CaptureSupport::unsupported(
            "cuda_ep MultiHeadAttention retains non-capturable cuBLASLt/softmax, padding and output-transpose synchronization paths",
        )
    }
}

impl MultiHeadAttentionKernel {
    /// Resolve `key_padding_mask` into an additive `[B, S, total]` f32 buffer,
    /// matching ORT's `PrepareMask` (`0` keeps a key, `mask_filter_value` masks
    /// it). The mask input is small and integer, so it round-trips through the
    /// host exactly as the CPU oracle reads it.
    fn resolve_pad_mask(
        &self,
        view: &TensorView,
        batch: usize,
        q_seq: usize,
        total_seq: usize,
    ) -> Result<Vec<f32>> {
        checked_i32(batch, "padding mask batch")?;
        checked_i32(q_seq, "padding mask query sequence")?;
        checked_i32(total_seq, "padding mask total sequence")?;
        let raw = self.read_i64(view)?;
        let filter = self.mask_filter_value;
        let keep_or = |keep: bool| if keep { 0.0f32 } else { filter };
        let output_elements =
            checked_product(&[batch, q_seq, total_seq], "resolved padding mask elements")?;
        checked_bytes(
            output_elements,
            std::mem::size_of::<f32>(),
            "resolved padding mask",
        )?;
        let mut out = vec![0.0f32; output_elements];
        let dims = view.shape;
        let total_i64 = i64::from(checked_i32(total_seq, "padding mask total sequence")?);
        let index = |b: usize, i: usize, j: usize| {
            checked_index3(b, i, j, q_seq, total_seq, "resolved padding mask index")
        };
        match *dims {
            [b] if b == batch => {
                for b in 0..batch {
                    let end = raw[b].clamp(0, total_i64);
                    for i in 0..q_seq {
                        for j in 0..total_seq {
                            let key = i64::from(checked_i32(j, "padding mask key index")?);
                            out[index(b, i, j)?] = keep_or(key < end);
                        }
                    }
                }
            }
            [b] if b
                == checked_add(
                    checked_product(&[3, batch], "key_padding_mask 3B")?,
                    2,
                    "key_padding_mask 3B+2",
                )? =>
            {
                for b in 0..batch {
                    let end = raw[b].clamp(0, total_i64);
                    let start = raw[checked_add(batch, b, "key_padding_mask start index")?]
                        .clamp(0, total_i64);
                    for i in 0..q_seq {
                        for j in 0..total_seq {
                            let key = i64::from(checked_i32(j, "padding mask key index")?);
                            out[index(b, i, j)?] = keep_or(key < end && key >= start);
                        }
                    }
                }
            }
            [b, t] if b == batch && t == total_seq => {
                for b in 0..batch {
                    for i in 0..q_seq {
                        for j in 0..total_seq {
                            let raw_index =
                                checked_index2(b, j, total_seq, "2-D key_padding_mask index")?;
                            out[index(b, i, j)?] = keep_or(raw[raw_index] > 0);
                        }
                    }
                }
            }
            [b, s, t] if b == batch && s == q_seq && t == total_seq => {
                for b in 0..batch {
                    for i in 0..q_seq {
                        for j in 0..total_seq {
                            let raw_index = checked_index3(
                                b,
                                i,
                                j,
                                q_seq,
                                total_seq,
                                "3-D key_padding_mask index",
                            )?;
                            out[index(b, i, j)?] = keep_or(raw[raw_index] > 0);
                        }
                    }
                }
            }
            _ => {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: unsupported key_padding_mask shape {dims:?} for batch={batch}, q_seq={q_seq}, total_seq={total_seq} (expected (B,), (3B+2), (B, T) or (B, S, T))"
                )));
            }
        }
        Ok(out)
    }

    /// Copy an int32/int64 device tensor to the host as `i64`.
    fn read_i64(&self, view: &TensorView) -> Result<Vec<i64>> {
        if !view.is_contiguous() {
            return Err(EpError::KernelFailed(
                "cuda_ep MultiHeadAttention: key_padding_mask must be contiguous".into(),
            ));
        }
        let n = checked_product(view.shape, "key_padding_mask element count")?;
        let src = cuptr(view.data_ptr::<u8>() as *const c_void);
        match view.dtype {
            DataType::Int64 => {
                let byte_count = checked_bytes(n, 8, "key_padding_mask int64")?;
                let mut bytes = vec![0u8; byte_count];
                // SAFETY: `src` is a live device tensor of `n` int64 elements.
                unsafe { self.runtime.dtoh(&mut bytes, src)? };
                Ok(bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_ne_bytes(c.try_into().unwrap()))
                    .collect())
            }
            DataType::Int32 => {
                let byte_count = checked_bytes(n, 4, "key_padding_mask int32")?;
                let mut bytes = vec![0u8; byte_count];
                // SAFETY: `src` is a live device tensor of `n` int32 elements.
                unsafe { self.runtime.dtoh(&mut bytes, src)? };
                Ok(bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_ne_bytes(c.try_into().unwrap()) as i64)
                    .collect())
            }
            other => Err(EpError::KernelFailed(format!(
                "cuda_ep MultiHeadAttention: key_padding_mask dtype {other:?} must be int32 or int64"
            ))),
        }
    }
}

/// Reinterpret an `f32` slice as its little-endian bytes for an H2D upload.
fn bytemuck_f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, NodeId, static_shape};
    use std::cell::Cell;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    fn node(attrs: &[(&str, Attribute)]) -> Node {
        let mut node = Node::new(NodeId(0), "MultiHeadAttention", vec![], vec![]);
        node.domain = "com.microsoft".to_string();
        for (name, value) in attrs {
            node.attributes.insert((*name).to_string(), value.clone());
        }
        node
    }

    fn shapes(dims: &[&[usize]]) -> Vec<Shape> {
        dims.iter()
            .map(|d| static_shape(d.iter().copied()))
            .collect()
    }

    #[test]
    fn claim_requires_positive_num_heads() {
        assert!(unsupported_reason(&node(&[]), &[], &[]).is_some());
        assert!(unsupported_reason(&node(&[("num_heads", Attribute::Int(0))]), &[], &[]).is_some());
        assert!(unsupported_reason(&node(&[("num_heads", Attribute::Int(2))]), &[], &[]).is_none());
    }

    #[test]
    fn claim_declines_packed_qkv_rank5_query() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let s = shapes(&[&[1, 2, 2, 3, 4], &[1, 2, 8], &[1, 2, 8]]);
        assert!(unsupported_reason(&n, &s, &[]).is_some());
    }

    // Packed-KV is a rank-5 `key` with `value` omitted. Both halves of that
    // shape must be refused at *claim* time: this EP has no packed-KV path, and
    // a rejection from `execute` lands after ORT has compiled the node here,
    // which fails the session outright instead of falling back. These two cases
    // were previously rejected only at execute time.
    #[test]
    fn claim_declines_packed_kv_rank5_key() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let s = shapes(&[&[1, 2, 8], &[1, 2, 2, 2, 4], &[]]);
        // Query f32, key f32, value absent — the packed-KV input signature.
        let dtypes = [DataType::Float32, DataType::Float32];
        let reason = unsupported_reason(&n, &s, &dtypes).expect("packed KV declined at claim time");
        assert!(
            reason.contains("packed-KV"),
            "expected a packed-KV decline, got: {reason}"
        );
    }

    #[test]
    fn claim_declines_key_of_unsupported_rank() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let s = shapes(&[&[1, 2, 8], &[1, 2, 2, 2, 4], &[1, 2, 8]]);
        let dtypes = [DataType::Float32, DataType::Float32, DataType::Float32];
        let reason = unsupported_reason(&n, &s, &dtypes).expect("rank-5 key declined");
        assert!(
            reason.contains("key must be rank"),
            "expected a key-rank decline, got: {reason}"
        );
    }

    // The guard must not over-decline: rank 3 and rank 4 key/value are both
    // supported layouts, and a symbolic shape (recorded as empty here) is
    // accepted at claim time and re-validated in `execute`.
    #[test]
    fn claim_accepts_supported_key_value_ranks() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let dtypes = [DataType::Float32, DataType::Float32, DataType::Float32];
        for s in [
            shapes(&[&[1, 2, 8], &[1, 2, 8], &[1, 2, 8]]),
            shapes(&[&[1, 2, 8], &[1, 2, 2, 4], &[1, 2, 2, 4]]),
            shapes(&[&[1, 2, 8], &[], &[]]),
        ] {
            assert!(
                unsupported_reason(&n, &s, &dtypes).is_none(),
                "supported layout wrongly declined: {s:?}"
            );
        }
    }

    #[test]
    fn claim_declines_mismatched_v_head_size() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        // q hidden 8 → head_size 4; value hidden 12 → v_head_size 6 ≠ 4.
        let s = shapes(&[&[1, 2, 8], &[1, 2, 8], &[1, 2, 12]]);
        let reason = unsupported_reason(&n, &s, &[]).expect("v_head_size mismatch declined");
        assert!(reason.contains("v_head_size"));
    }

    #[test]
    fn claim_declines_non_float_query() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let dtypes = [DataType::Int32, DataType::Int32, DataType::Int32];
        assert!(unsupported_reason(&n, &[], &dtypes).is_some());
    }

    #[test]
    fn claim_declines_dtype_mismatch_between_qkv() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let dtypes = [DataType::Float16, DataType::Float32, DataType::Float16];
        assert!(unsupported_reason(&n, &[], &dtypes).is_some());
    }

    #[test]
    fn claim_accepts_plain_self_attention() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let s = shapes(&[&[1, 4, 8], &[1, 4, 8], &[1, 4, 8]]);
        let dtypes = [DataType::Float32, DataType::Float32, DataType::Float32];
        assert!(unsupported_reason(&n, &s, &dtypes).is_none());
    }

    #[derive(Default)]
    struct SyncPolicyRuntime {
        calls: Cell<usize>,
        fail: Cell<bool>,
    }
    impl MhaStreamRuntime for SyncPolicyRuntime {
        fn apply_sync_policy(&self) -> Result<()> {
            self.calls.set(self.calls.get() + 1);
            if self.fail.get() {
                Err(mha_error("injected sync policy failure"))
            } else {
                Ok(())
            }
        }
    }
    fn planned_geometry(dtype: DataType, cross: bool, past: bool) -> MhaGeometry {
        let q = [2, 5, 12];
        let kv = if cross {
            vec![2, 3, 7, 4]
        } else {
            vec![2, 7, 12]
        };
        let cache = [2, 3, 2, 4];
        let absent = TensorMetadata::new(DataType::Undefined, &[], false);
        MhaGeometry::from_inputs(
            3,
            &[
                TensorMetadata::new(dtype, &q, true),
                TensorMetadata::new(dtype, &kv, true),
                TensorMetadata::new(dtype, &kv, true),
                absent,
                absent,
                absent,
                TensorMetadata::new(dtype, &cache, past),
                TensorMetadata::new(dtype, &cache, past),
            ],
        )
        .unwrap()
    }

    #[test]
    fn prepared_composite_layout_covers_all_live_regions_and_output_modes() {
        for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
            for cross in [false, true] {
                for past in [false, true] {
                    let geometry = planned_geometry(dtype, cross, past);
                    assert_eq!(geometry.total_seq, if past { 9 } else { 7 });
                    for present in [[false, false], [true, false], [false, true], [true, true]] {
                        for (pad, bias) in
                            [(false, false), (true, false), (false, true), (true, true)]
                        {
                            let layout = MhaWorkspace::new(geometry, present, pad, bias).unwrap();
                            // Opaque device backing: layout binding must not read it.
                            let base = 0x10000usize;
                            let backing = WorkspaceView::new(
                                onnx_runtime_ep_api::DevicePtrMut(base as *mut _),
                                layout.regions.bytes(),
                            );
                            let views = layout.bind(Some(backing)).unwrap();
                            let kv_bytes = 2 * 3 * geometry.total_seq * 4 * dtype.byte_size();
                            let mask_bytes = 2 * 3 * 5 * geometry.total_seq * dtype.byte_size();
                            assert_eq!(
                                views.map(|view| view.bytes()),
                                [
                                    2 * 3 * 5 * 4 * dtype.byte_size(),
                                    if present[0] { 0 } else { kv_bytes },
                                    if present[1] { 0 } else { kv_bytes },
                                    2 * 3 * 5 * 4 * dtype.byte_size(),
                                    if pad || bias { mask_bytes } else { 0 },
                                    if pad {
                                        2 * 5 * geometry.total_seq * 4
                                    } else {
                                        0
                                    },
                                    mask_bytes.next_multiple_of(BLAS_SOFTMAX_SCRATCH_ALIGN)
                                        + crate::blas::WORKSPACE_BYTES,
                                ]
                            );
                            let mut end = base;
                            for region in views.into_iter().filter(|view| view.bytes() != 0) {
                                let start = region.ptr().0 as usize;
                                assert!(
                                    start >= end
                                        && start.is_multiple_of(BLAS_SOFTMAX_SCRATCH_ALIGN)
                                );
                                end = start + region.bytes();
                                assert!(end <= base + layout.regions.bytes());
                            }
                            assert!(
                                BlasSoftmaxScratch::bind(&layout.blas_softmax, Some(views[6]))
                                    .is_ok()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mask_indices_format_diagnostics_only_after_overflow() {
        struct Context<'a>(&'a Cell<usize>);
        impl std::fmt::Display for Context<'_> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.0.set(self.0.get() + 1);
                formatter.write_str("padding mask index")
            }
        }
        let formats = Cell::new(0);
        let context = Context(&formats);
        for plane in 0..3 {
            for row in 0..5 {
                for column in 0..7 {
                    assert_eq!(
                        checked_index3(plane, row, column, 5, 7, &context).unwrap(),
                        (plane * 5 + row) * 7 + column
                    );
                }
            }
        }
        assert_eq!(
            formats.get(),
            0,
            "successful per-cell index checks must not construct diagnostic strings"
        );
        let error = checked_index3(usize::MAX, 0, 0, 2, 7, &context)
            .unwrap_err()
            .to_string();
        assert_eq!(formats.get(), 1, "overflow must retain its lazy context");
        assert!(
            error.contains("padding mask index plane offset") && error.contains("exceeds usize"),
            "{error}"
        );
        let index = checked_index2(0, 1, 7, &context).unwrap();
        assert_eq!(index, 1);
        assert_eq!(
            formats.get(),
            1,
            "valid addition must not format its context either"
        );
    }

    #[test]
    fn prepared_composite_retains_one_funded_peak_for_deferred_runs() {
        let geometry = planned_geometry(DataType::Float32, false, true);
        let layout = MhaWorkspace::new(geometry, [false, true], true, true).unwrap();
        let requirement = layout.requirement().unwrap();
        assert_eq!(requirement.bytes, layout.regions.bytes() as u64);
        assert_eq!(requirement.alignment, layout.regions.alignment());
        assert_eq!(requirement.lifetime, WorkspaceLifetime::SessionPersistent);
        assert_eq!(
            requirement.role(),
            onnx_runtime_memory_api::MemoryRole::Workspace { step_scoped: false }
        );
    }

    #[test]
    fn prepared_output_metadata_checks_declared_slots_dtype_and_exact_geometry() {
        for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
            let g = planned_geometry(dtype, true, true);
            let y = [g.batch, g.q_seq, g.q_hidden];
            let cache = [g.batch, g.heads, g.total_seq, g.head_size];
            for present in [[false, false], [true, false], [false, true], [true, true]] {
                let outputs = [
                    OutputMetadata::tensor(dtype, &y),
                    if present[0] {
                        OutputMetadata::tensor(dtype, &cache)
                    } else {
                        OutputMetadata::Absent
                    },
                    if present[1] {
                        OutputMetadata::tensor(dtype, &cache)
                    } else {
                        OutputMetadata::Absent
                    },
                ];
                validate_output_metadata(g, present, &outputs).unwrap();
                let unresolved = [
                    OutputMetadata::Unresolved { dtype },
                    if present[0] {
                        OutputMetadata::Unresolved { dtype }
                    } else {
                        OutputMetadata::Absent
                    },
                    if present[1] {
                        OutputMetadata::Unresolved { dtype }
                    } else {
                        OutputMetadata::Absent
                    },
                ];
                validate_output_metadata(g, present, &unresolved).unwrap();
            }
            let wrong_cache = [g.batch, g.heads, g.total_seq + 1, g.head_size];
            let error = validate_output_metadata(
                g,
                [false, true],
                &[
                    OutputMetadata::tensor(dtype, &y),
                    OutputMetadata::Absent,
                    OutputMetadata::tensor(dtype, &wrong_cache),
                ],
            )
            .unwrap_err()
            .to_string();
            assert!(
                error.contains("slot 2 shape") && error.contains("exact logical MHA"),
                "{error}"
            );
            let error =
                validate_output_metadata(g, [true, false], &[OutputMetadata::tensor(dtype, &y)])
                    .unwrap_err()
                    .to_string();
            assert!(error.contains("compiled slots"), "{error}");
        }
        let g = planned_geometry(DataType::Float32, false, false);
        let y = [g.batch, g.q_seq, g.q_hidden];
        let error = validate_output_metadata(
            g,
            [false, false],
            &[OutputMetadata::tensor(DataType::Float16, &y)],
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("slot 0 dtype") && error.contains("Float32"),
            "{error}"
        );
        assert!(
            validate_output_metadata(g, [false, false], &[OutputMetadata::Absent])
                .unwrap_err()
                .to_string()
                .contains("required output")
        );
    }

    #[test]
    fn prepared_composite_rejects_missing_short_misaligned_and_wrapping_backing() {
        let layout = MhaWorkspace::new(
            planned_geometry(DataType::Float32, false, true),
            [false; 2],
            true,
            true,
        )
        .unwrap();
        for backing in [
            None,
            Some(WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(std::ptr::null_mut()),
                layout.regions.bytes(),
            )),
            Some(WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(256usize as *mut _),
                layout.regions.bytes() - 1,
            )),
            Some(WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(257usize as *mut _),
                layout.regions.bytes(),
            )),
            Some(WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut((usize::MAX - 255) as *mut _),
                layout.regions.bytes(),
            )),
        ] {
            let message = layout.bind(backing).unwrap_err().to_string();
            assert!(
                message.contains("MultiHeadAttention")
                    && message.contains("execute_with_workspace"),
                "{message}"
            );
        }
    }

    #[test]
    fn geometry_rejects_invalid_input_without_a_runtime_or_allocation() {
        let q_shape = [2, 5, 12];
        let kv_shape = [2, 7, 12];
        let q = TensorMetadata::new(DataType::Float32, &q_shape, true);
        let kv = TensorMetadata::new(DataType::Float32, &kv_shape, true);
        for (heads, inputs, expected) in [
            (0, vec![q, kv, kv], "num_heads"),
            (3, vec![q], "key"),
            (5, vec![q, kv, kv], "not divisible"),
            (
                3,
                vec![
                    q,
                    kv,
                    TensorMetadata::new(DataType::Float16, &kv_shape, true),
                ],
                "dtype",
            ),
            (
                3,
                vec![
                    q,
                    kv,
                    TensorMetadata::new(DataType::Float32, &[2, 7, 15], true),
                ],
                "head_size",
            ),
        ] {
            let message = MhaGeometry::from_inputs(heads, &inputs)
                .unwrap_err()
                .to_string();
            assert!(message.contains(expected), "{message}");
        }
        let huge = [1, i32::MAX as usize, 4];
        let inputs = [TensorMetadata::new(DataType::Float32, &huge, true); 3];
        let message = MhaGeometry::from_inputs(4, &inputs)
            .unwrap_err()
            .to_string();
        assert!(message.contains("attention rows"), "{message}");
    }

    #[test]
    fn declared_present_output_slots_preserve_omitted_positions() {
        use onnx_runtime_ir::ValueId;
        let node =
            |outputs| Node::new_with_output_slots(NodeId(0), "MultiHeadAttention", vec![], outputs);
        assert_eq!(
            present_output_slots(&node(vec![Some(ValueId(0)), None, Some(ValueId(2))])).unwrap(),
            [false, true]
        );
        assert_eq!(
            present_output_slots(&node(vec![Some(ValueId(0))])).unwrap(),
            [false; 2]
        );
        assert!(present_output_slots(&node(vec![None])).is_err());
        assert!(present_output_slots(&node(vec![])).is_err());
        for compiled in [[false; 2], [true, false], [false, true], [true; 2]] {
            assert!(check_present_slots(compiled, compiled).is_ok());
            let mismatched = [!compiled[0], compiled[1]];
            let message = check_present_slots(compiled, mismatched)
                .unwrap_err()
                .to_string();
            assert!(
                message.contains("compiled slots")
                    && message.contains("declared positional outputs"),
                "{message}"
            );
        }
    }

    #[test]
    fn submission_sync_policy_preserves_both_errors_and_is_not_retried() {
        let runtime = SyncPolicyRuntime::default();
        runtime.fail.set(true);
        let mut submission = MhaStreamUse::new(&runtime);
        submission.mark_stream_use();
        let message = submission
            .finish(Err(mha_error("injected launch failure")))
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("launch failure") && message.contains("sync policy failure"),
            "{message}"
        );
        assert_eq!(runtime.calls.get(), 1);
    }

    #[test]
    fn submission_guard_applies_runtime_policy_on_success_error_and_panic() {
        let runtime = SyncPolicyRuntime::default();
        MhaStreamUse::new(&runtime)
            .finish(Err(mha_error("before submission")))
            .unwrap_err();
        assert_eq!(runtime.calls.get(), 0);
        let mut submission = MhaStreamUse::new(&runtime);
        submission.mark_stream_use();
        submission.finish(Ok(())).unwrap();
        assert_eq!(runtime.calls.get(), 1);
        let mut submission = MhaStreamUse::new(&runtime);
        submission.mark_stream_use();
        submission
            .finish(Err(mha_error("after submission")))
            .unwrap_err();
        assert_eq!(runtime.calls.get(), 2);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let mut submission = MhaStreamUse::new(&runtime);
                submission.mark_stream_use();
                panic!("injected panic after launch");
            }))
            .is_err()
        );
        assert_eq!(runtime.calls.get(), 3);
    }

    #[test]
    fn checked_geometry_rejects_product_and_byte_overflow() {
        let product = checked_product(&[usize::MAX, 2], "query elements").unwrap_err();
        assert!(product.to_string().contains("query elements product"));
        let bytes = checked_bytes(usize::MAX, 2, "mask").unwrap_err();
        assert!(bytes.to_string().contains("mask byte size"));
    }

    #[test]
    fn nvrtc_linear_indices_promote_before_every_int_product() {
        // Every pointer/linear-index product rooted in CUDA `int` geometry must
        // promote its first operand, not cast the already-overflowed product.
        // This list is the complete index audit of SOURCE (counts/loop strides
        // already start with long long and are deliberately not repeated).
        for required in [
            "((long long)b * heads + h) * past_seq",
            "((long long)b * heads + h) * cur_seq",
            "((long long)b * cur_seq + sc) * heads",
            "bias[(long long)h * dim + d]",
            "((long long)b * seq + s) * heads",
            "((long long)b0 * abias_d1 + b1) * sq",
            "((long long)b * sq + i) * total",
        ] {
            assert!(
                SOURCE.contains(required),
                "MHA NVRTC source must promote before multiplication: {required}"
            );
        }
        for forbidden in [
            "(long long)(b * heads + h)",
            "(long long)(b * cur_seq + sc)",
            "(long long)(b * seq + s)",
            "(long long)(b0 * abias_d1 + b1)",
            "(long long)(b * sq + i)",
            "bias[h * dim + d]",
        ] {
            assert!(
                !SOURCE.contains(forbidden),
                "MHA NVRTC source casts after an overflowing int expression: {forbidden}"
            );
        }
    }

    #[test]
    fn host_geometry_allows_products_above_i32_when_offsets_fit_usize() {
        let batch = 65_536usize;
        let sequence = 65_536usize;
        assert!(checked_i32(batch, "batch").is_ok());
        assert!(checked_i32(sequence, "sequence").is_ok());
        let elements = checked_product(&[batch, sequence, 2, 4], "large MHA elements").unwrap();
        assert_eq!(elements, 34_359_738_368);
        assert!(elements > i32::MAX as usize);
        assert_eq!(
            checked_bytes(elements, std::mem::size_of::<f32>(), "large MHA").unwrap(),
            137_438_953_472
        );
    }

    #[test]
    fn checked_cuda_conversions_reject_i32_and_u32_overflow() {
        let i32_error = checked_i32(i32::MAX as usize + 1, "past sequence length").unwrap_err();
        assert!(
            i32_error
                .to_string()
                .contains("past sequence length=2147483648")
        );
        let too_many_blocks = (u32::MAX as usize + 1) * BLOCK as usize;
        let grid_error = checked_grid(too_many_blocks, "mask launch").unwrap_err();
        assert!(grid_error.to_string().contains("mask launch grid blocks"));
    }

    #[test]
    fn claim_rejects_statically_known_zero_and_cuda_i32_geometry() {
        let n = node(&[("num_heads", Attribute::Int(1))]);
        let zero = shapes(&[&[1, 0, 4], &[1, 1, 4], &[1, 1, 4]]);
        let zero_reason = unsupported_reason(&n, &zero, &[]).unwrap();
        assert!(zero_reason.contains("query dim 1 must be > 0"));

        let oversized = shapes(&[
            &[1, 1, i32::MAX as usize + 1],
            &[1, 1, i32::MAX as usize + 1],
            &[1, 1, i32::MAX as usize + 1],
        ]);
        let oversized_reason = unsupported_reason(&n, &oversized, &[]).unwrap();
        assert!(oversized_reason.contains("query dim 2=2147483648"));
    }

    #[test]
    fn claim_rejects_rank_one_qkv() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let q_reason = unsupported_reason(&n, &shapes(&[&[8]]), &[]).unwrap();
        assert!(q_reason.contains("query must be rank 3"));

        let kv = shapes(&[&[1, 2, 8], &[8], &[1, 2, 8]]);
        let kv_reason = unsupported_reason(&n, &kv, &[]).unwrap();
        assert!(kv_reason.contains("key must be rank 3"));
    }

    #[test]
    fn claim_rejects_statically_known_batch_past_and_mask_mismatches() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let batch_mismatch = shapes(&[&[2, 1, 8], &[1, 1, 8], &[2, 1, 8]]);
        let reason = unsupported_reason(&n, &batch_mismatch, &[]).unwrap();
        assert!(reason.contains("key batch 1 must equal query batch 2"));

        let mask_mismatch = shapes(&[&[2, 1, 8], &[2, 1, 8], &[2, 1, 8], &[], &[2, 9]]);
        let reason = unsupported_reason(&n, &mask_mismatch, &[]).unwrap();
        assert!(reason.contains("key_padding_mask static shape"));

        let past_mismatch = shapes(&[
            &[2, 1, 8],
            &[2, 1, 8],
            &[2, 1, 8],
            &[],
            &[],
            &[],
            &[2, 2, 4, 4],
            &[2, 2, 5, 4],
        ]);
        let reason = unsupported_reason(&n, &past_mismatch, &[]).unwrap();
        assert!(reason.contains("past_key seq 4 != past_value seq 5"));
    }
}
