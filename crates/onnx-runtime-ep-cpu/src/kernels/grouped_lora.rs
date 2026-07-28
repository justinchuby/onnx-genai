//! `GroupedLoraDelta` — the Phase-2 batched-LoRA custom op (design
//! `docs/NATIVE_LORA_DESIGN.md` §J.1), CPU **group-by-adapter dense** path.
//!
//! One op instance per target projection, placed exactly where Phase-1 places
//! its delta branch: it reads the base projection activation `x[tokens, K]` and a
//! per-row routing descriptor `segments[tokens]`, and produces a delta
//! `[tokens, N]` that the reused Phase-1 `Add` folds onto the base `MatMulNBits`
//! output. The base int4 weights are never touched.
//!
//! ## Kernel scope (this task)
//!
//! The kernel implemented here is the **group-by-adapter dense path** only
//! (design §J.3 "single-adapter fallback" generalized to a per-adapter loop):
//! partition rows by adapter, run a plain dense `X_g @ A_t` then `@ B_t` per
//! group with **fp32 accumulators**, scatter the scaled result back. The fused
//! BGMV/SGMV grouped kernels (design §J.3) are the *next* task, gated on the P2d
//! measurements — they are deliberately **not** built here.
//!
//! ## Numerics — fp32 accumulators (mandatory, design §J.3)
//!
//! `x` and the pool's `A_t`/`B_t` may be fp16; both matmuls accumulate in fp32
//! (the shared [`gemm`] is fp32) and the per-module `scale` is applied in fp32.
//! The delta is narrowed to the branch dtype only at the final store. fp16
//! accumulation flips greedy-argmax ties at realistic activation scale (the
//! flash-attention lesson, §I item 5), so it is never used here.
//!
//! ## Pool delivery (CPU)
//!
//! The paged adapter pool ([`LoraWeightPool`]) is resolved by a `pool_id`
//! attribute through the process [`LoraPoolRegistry`]. See that type's docs for
//! why the CPU host pool binds by id rather than through the lazy-weight seam
//! (`LazyWeightBoundary::GroupedLora`), which is the *paging*-EP device binding.

use std::cell::RefCell;
use std::sync::Arc;

use onnx_runtime_ep_api::{
    AdapterId, EpError, Kernel, KernelFactory, LoraFactorView, LoraModuleId, LoraPagePair,
    LoraPoolId, LoraPoolRegistry, LoraWeightPool, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{DataType, Node};

use super::matmul::gemm;
use super::{check_arity, to_dense_i64};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::numel;

const OP: &str = "GroupedLoraDelta";

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{OP}: {}", message.into()))
}

/// A `GroupedLoraDelta` kernel bound to one target projection and one adapter
/// pool.
pub struct GroupedLoraDeltaKernel {
    /// Inner dimension `K` (base projection input features).
    k: usize,
    /// This op's output width `N` (the slice width for a fused-QKV target).
    n: usize,
    /// Which target module within each adapter's pages this op reads.
    module_id: LoraModuleId,
    /// Column budget for the intermediate `[tokens, r]` (validation only here).
    max_rank: usize,
    /// The shared paged adapter pool.
    pool: Arc<LoraWeightPool>,
}

pub struct GroupedLoraDeltaFactory;

impl KernelFactory for GroupedLoraDeltaFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let k = required_positive_attr(node, "K")?;
        let n = required_positive_attr(node, "N")?;
        let module_id = required_nonneg_attr(node, "module_id")?;
        let max_rank = required_positive_attr(node, "max_rank")?;
        let pool_id = required_nonneg_attr(node, "pool_id")? as u64;
        let pool = LoraPoolRegistry::global()
            .get(LoraPoolId(pool_id))
            .ok_or_else(|| {
                error(format!(
                    "no adapter pool registered under pool_id {pool_id}; register the pool with \
                     LoraPoolRegistry before building the session"
                ))
            })?;
        Ok(Box::new(GroupedLoraDeltaKernel {
            k,
            n,
            module_id: LoraModuleId(module_id as u32),
            max_rank,
            pool,
        }))
    }
}

impl Kernel for GroupedLoraDeltaKernel {
    fn set_constant_inputs(&mut self, _constant_inputs: &[bool]) {
        // Neither `x` nor `segments` is ever a graph constant: `segments` changes
        // every run (per-batch routing) and must never be prepacked. Nothing to
        // memoize.
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(OP, inputs, outputs, 2, 2, 1)?;
        let x = &inputs[0];
        let segments = &inputs[1];
        let out = &mut outputs[0];

        require_float(x.dtype, "x")?;
        if out.dtype != x.dtype {
            return Err(error(format!(
                "delta dtype {:?} must match x dtype {:?}",
                out.dtype, x.dtype
            )));
        }

        let x_shape = x.shape;
        if x_shape.is_empty() || x_shape[x_shape.len() - 1] != self.k {
            return Err(error(format!(
                "x must have rank >= 1 with last dimension K={}, got shape {:?}",
                self.k, x_shape
            )));
        }
        let tokens = numel(&x_shape[..x_shape.len() - 1]);

        let expected_out = [&x_shape[..x_shape.len() - 1], &[self.n]].concat();
        if out.shape != expected_out.as_slice() {
            return Err(error(format!(
                "delta shape {:?} must be {:?}",
                out.shape, expected_out
            )));
        }

        if numel(segments.shape) != tokens {
            return Err(error(format!(
                "segments must have {tokens} elements (one per token), got shape {:?}",
                segments.shape
            )));
        }

        SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            self.compute(x, segments, tokens, &mut scratch, out)
        })
    }
}

/// Reusable per-thread scratch so the decode hot path performs no fresh heap
/// allocation once buffers reach steady-state size (buffers only ever grow).
#[derive(Default)]
struct Scratch {
    /// Widened `x` `[tokens, K]` (only owned when `x` is not already dense f32).
    delta: Vec<f32>,
    /// Gathered group activations `[m, K]`.
    xg: Vec<f32>,
    /// Intermediate `[m, r]`.
    t: Vec<f32>,
    /// Group delta `[m, N]`.
    d: Vec<f32>,
    /// Widened `A_t` / `B_t`.
    a_f32: Vec<f32>,
    b_f32: Vec<f32>,
    /// Row indices per adapter group.
    rows: Vec<usize>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

impl GroupedLoraDeltaKernel {
    fn compute(
        &self,
        x: &TensorView,
        segments: &TensorView,
        tokens: usize,
        scratch: &mut Scratch,
        out: &mut TensorMut,
    ) -> Result<()> {
        let x_f32 = to_dense_f32_widen(OP, x)?;
        let segment_ids = to_dense_i64(segments)?;

        // delta accumulator, zero-filled: null-adapter rows stay zero.
        scratch.delta.clear();
        scratch.delta.resize(tokens * self.n, 0.0);

        // Group rows by adapter id, preserving first-seen order. Rows routed to a
        // negative id (or the reserved null adapter) are base-only (zero delta).
        // Fast path: a batch that resolves to one adapter (the common
        // single-tenant / decode case) takes the dense branch with no gather.
        let uniform = segment_ids.first().copied();
        let all_uniform = uniform.is_some_and(|first| segment_ids.iter().all(|&s| s == first));

        if all_uniform {
            let id = uniform.expect("non-empty tokens have a first segment");
            if let Some(adapter) = adapter_of(id) {
                // Whole batch, contiguous rows in order: feed x directly.
                self.apply_group(adapter, &x_f32, tokens, None, scratch)?;
            }
            return self.store(scratch, out);
        }

        // Multi-adapter: partition rows by adapter id (group-by-adapter). This
        // grouping allocates a per-call row-index vector; the allocation-free
        // fused BGMV/SGMV path is the next task (design §J.3).
        let mut distinct: Vec<i64> = Vec::new();
        for &id in &segment_ids {
            if adapter_of(id).is_some() && !distinct.contains(&id) {
                distinct.push(id);
            }
        }
        for id in distinct {
            let adapter = adapter_of(id).expect("filtered to real adapters");
            scratch.rows.clear();
            for (row, &s) in segment_ids.iter().enumerate() {
                if s == id {
                    scratch.rows.push(row);
                }
            }
            let group_rows = std::mem::take(&mut scratch.rows);
            let m = group_rows.len();
            scratch.xg.clear();
            scratch.xg.resize(m * self.k, 0.0);
            for (local, &row) in group_rows.iter().enumerate() {
                let src = &x_f32[row * self.k..(row + 1) * self.k];
                scratch.xg[local * self.k..(local + 1) * self.k].copy_from_slice(src);
            }
            // Borrow-safe: move xg out, run, then apply_group reads xg via scratch.
            self.apply_group_gathered(adapter, m, &group_rows, scratch)?;
            scratch.rows = group_rows;
        }
        self.store(scratch, out)
    }

    /// Dense delta for a group whose activations are the whole contiguous `x`
    /// (single-adapter fast path). Writes scaled result into `scratch.delta` for
    /// all `tokens` rows.
    fn apply_group(
        &self,
        adapter: AdapterId,
        x_f32: &[f32],
        m: usize,
        _rows: Option<&[usize]>,
        scratch: &mut Scratch,
    ) -> Result<()> {
        let pair = self.pair_for(adapter)?;
        let (rank, n, scale) = self.validate_pair(adapter, &pair)?;

        // Take the reusable factor/intermediate buffers out of `scratch` so the
        // factor slices (which may borrow `scratch`) do not collide with the
        // `&mut` intermediates. For f32 pool factors the factor slice is a
        // zero-copy reinterpret of the aligned pool bytes and the buffers stay
        // empty — the decode hot path performs no per-call widening copy.
        let mut a_buf = std::mem::take(&mut scratch.a_f32);
        let mut b_buf = std::mem::take(&mut scratch.b_f32);
        let mut t = std::mem::take(&mut scratch.t);
        let mut d = std::mem::take(&mut scratch.d);
        {
            let a_slice = factor_slice(&pair.a, self.k * rank, &mut a_buf)?;
            let b_slice = factor_slice(&pair.b, rank * n, &mut b_buf)?;
            // t = X @ A_t -> [m, rank]   (fp32 accumulators)
            t.clear();
            t.resize(m * rank, 0.0);
            gemm(x_f32, a_slice, &mut t, m, self.k, rank)?;
            // d = t @ B_t -> [m, N]      (fp32 accumulators)
            d.clear();
            d.resize(m * n, 0.0);
            gemm(&t, b_slice, &mut d, m, rank, n)?;
        }
        for (dst, value) in scratch.delta.iter_mut().zip(&d) {
            *dst = value * scale;
        }
        scratch.a_f32 = a_buf;
        scratch.b_f32 = b_buf;
        scratch.t = t;
        scratch.d = d;
        Ok(())
    }

    /// Dense delta for a gathered group `scratch.xg = [m, K]`, scattering the
    /// scaled result back to the original rows in `scratch.delta`.
    fn apply_group_gathered(
        &self,
        adapter: AdapterId,
        m: usize,
        rows: &[usize],
        scratch: &mut Scratch,
    ) -> Result<()> {
        let pair = self.pair_for(adapter)?;
        let (rank, n, scale) = self.validate_pair(adapter, &pair)?;

        let mut a_buf = std::mem::take(&mut scratch.a_f32);
        let mut b_buf = std::mem::take(&mut scratch.b_f32);
        let xg = std::mem::take(&mut scratch.xg);
        let mut t = std::mem::take(&mut scratch.t);
        let mut d = std::mem::take(&mut scratch.d);
        {
            let a_slice = factor_slice(&pair.a, self.k * rank, &mut a_buf)?;
            let b_slice = factor_slice(&pair.b, rank * n, &mut b_buf)?;
            t.clear();
            t.resize(m * rank, 0.0);
            gemm(&xg, a_slice, &mut t, m, self.k, rank)?;
            d.clear();
            d.resize(m * n, 0.0);
            gemm(&t, b_slice, &mut d, m, rank, n)?;
        }
        for (local, &row) in rows.iter().enumerate() {
            let src = &d[local * n..(local + 1) * n];
            let dst = &mut scratch.delta[row * self.n..(row + 1) * self.n];
            for (delta, value) in dst.iter_mut().zip(src) {
                *delta = value * scale;
            }
        }
        scratch.a_f32 = a_buf;
        scratch.b_f32 = b_buf;
        scratch.xg = xg;
        scratch.t = t;
        scratch.d = d;
        Ok(())
    }

    /// Look up this adapter/module's resident factor pair, failing loud if the
    /// page is missing.
    fn pair_for(&self, adapter: AdapterId) -> Result<LoraPagePair<'_>> {
        self.pool.pair(adapter, self.module_id).ok_or_else(|| {
            error(format!(
                "adapter {} module {} has no resident page in the pool",
                adapter.0, self.module_id.0
            ))
        })
    }

    /// Validate a resident pair's geometry against this op's declared dims and
    /// return `(rank, n, scale)`. Fails loud on any disagreement.
    fn validate_pair(
        &self,
        adapter: AdapterId,
        pair: &LoraPagePair<'_>,
    ) -> Result<(usize, usize, f32)> {
        let rank = pair.a.cols;
        if pair.b.rows != rank {
            return Err(error(format!(
                "adapter {} module {}: A_t rank {rank} != B_t rank {}",
                adapter.0, self.module_id.0, pair.b.rows
            )));
        }
        if pair.b.cols != self.n {
            return Err(error(format!(
                "adapter {} module {}: B_t width {} != op width N={}",
                adapter.0, self.module_id.0, pair.b.cols, self.n
            )));
        }
        if pair.a.rows != self.k {
            return Err(error(format!(
                "adapter {} module {}: A_t K {} != op K={}",
                adapter.0, self.module_id.0, pair.a.rows, self.k
            )));
        }
        if rank > self.max_rank {
            return Err(error(format!(
                "adapter {} module {}: rank {rank} exceeds max_rank {}",
                adapter.0, self.module_id.0, self.max_rank
            )));
        }
        Ok((rank, self.n, pair.scale))
    }

    fn store(&self, scratch: &Scratch, out: &mut TensorMut) -> Result<()> {
        write_dense_f32_narrow(OP, out, &scratch.delta)
    }
}

/// Reinterpret 64-byte-aligned f32 factor bytes as `&[f32]` with no copy — the
/// zero-copy decode hot path for f32 pool factors.
fn f32_bytes_as_slice(bytes: &[u8]) -> &[f32] {
    // SAFETY: pool factor pages are 64-byte aligned (LORA_PAGE_ALIGNMENT), so the
    // start is f32-aligned, and an f32 factor stores exactly 4 bytes per element,
    // so `bytes.len()` is a multiple of 4. The returned slice borrows the same
    // immutable bytes for the same lifetime; no mutable alias exists (the pool
    // hands out `&self` views only).
    debug_assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<f32>(), 0);
    debug_assert_eq!(bytes.len() % 4, 0);
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4) }
}

/// Produce a contiguous `&[f32]` view of a factor: zero-copy for an f32 pool
/// page, or widened once into the reusable `buf` for an f16/bf16 page (both
/// matmuls then accumulate in fp32 regardless — design §J.3).
fn factor_slice<'a>(
    view: &LoraFactorView<'a>,
    count: usize,
    buf: &'a mut Vec<f32>,
) -> Result<&'a [f32]> {
    if view.dtype == DataType::Float32 {
        let slice = f32_bytes_as_slice(view.bytes);
        if slice.len() != count {
            return Err(error(format!(
                "adapter factor has {} elements, expected {count}",
                slice.len()
            )));
        }
        Ok(slice)
    } else {
        decode_f32(view.bytes, view.dtype, count, buf)?;
        Ok(&*buf)
    }
}

/// Map a raw segment id to an adapter, or `None` for the base-only row (a
/// negative id or the reserved null adapter).
fn adapter_of(id: i64) -> Option<AdapterId> {
    if id < 0 {
        return None;
    }
    let adapter = AdapterId(id as u64);
    if adapter.is_null() {
        None
    } else {
        Some(adapter)
    }
}

fn decode_f32(bytes: &[u8], dtype: DataType, count: usize, out: &mut Vec<f32>) -> Result<()> {
    out.clear();
    out.reserve(count);
    match dtype {
        DataType::Float32 => {
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        DataType::Float16 => {
            for chunk in bytes.chunks_exact(2) {
                out.push(half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
            }
        }
        DataType::BFloat16 => {
            for chunk in bytes.chunks_exact(2) {
                out.push(half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
            }
        }
        other => {
            return Err(error(format!(
                "adapter factor dtype {other:?} is not a supported float (f32/f16/bf16)"
            )));
        }
    }
    if out.len() != count {
        return Err(error(format!(
            "adapter factor has {} elements, expected {count}",
            out.len()
        )));
    }
    Ok(())
}

fn require_float(dtype: DataType, name: &str) -> Result<()> {
    match dtype {
        DataType::Float32 | DataType::Float16 | DataType::BFloat16 => Ok(()),
        other => Err(error(format!(
            "{name} must be a float tensor (f32/f16/bf16), got {other:?}"
        ))),
    }
}

fn required_positive_attr(node: &Node, name: &str) -> Result<usize> {
    let value = required_int_attr(node, name)?;
    if value <= 0 {
        return Err(error(format!("attribute '{name}' must be positive, got {value}")));
    }
    Ok(value as usize)
}

fn required_nonneg_attr(node: &Node, name: &str) -> Result<i64> {
    let value = required_int_attr(node, name)?;
    if value < 0 {
        return Err(error(format!(
            "attribute '{name}' must be non-negative, got {value}"
        )));
    }
    Ok(value)
}

fn required_int_attr(node: &Node, name: &str) -> Result<i64> {
    node.attr(name)
        .and_then(|attribute| attribute.as_int())
        .ok_or_else(|| error(format!("missing required integer attribute '{name}'")))
}

#[cfg(test)]
mod tests;
