//! `com.microsoft::GroupQueryAttention` — optimized CPU GQA kernel.
//!
//! Implements unpacked Q/K/V and packed QKV inputs, BNSH KV caches, causal and
//! local-window masking, rotary embedding, and score softcap. Packed KV,
//! quantized caches, attention bias, smooth softmax/head sink, and QK capture
//! are rejected.
//!
//! ## Performance design (M=1 decode, long context)
//!
//! The decode hot path is a GEMV over the KV cache, executed per
//! `(batch, query_head, query_seq)` row.  Three targeted optimizations reduce
//! GQA latency at long context relative to the scalar reference:
//!
//! 1. **Attended-window scoring only**: scores are computed and stored only for
//!    the `[local_start, causal_limit]` range; unattended positions are never
//!    written to a full-length scratch buffer.
//! 2. **SIMD dot-product** ([`dot_f32`] / [`dot_avx2_fma`]): the Q·K dot
//!    product uses AVX2+FMA (two accumulators to hide latency, scalar tail) on
//!    x86-64 hosts where `is_x86_feature_detected!("avx2") && "fma"` holds;
//!    falls back to a scalar sum on other targets.
//! 3. **Cache-friendly P·V accumulation** ([`axpy_f32`] / [`axpy_avx2_fma`]):
//!    the weighted-sum loop is reordered to ks-outer, d-inner so that the V row
//!    (`head_dim` contiguous f32s) is accessed sequentially per key, matching
//!    cache-line width and enabling AVX2 FMADD.
//!
//! ### Precision contract (RULES.md §4 / cross-EP parity)
//! Softmax uses the **exact** `(score - max) as f64).exp() as f32` path, unchanged
//! from the original.  The dot-product and AXPY SIMD paths may reorder f32
//! additions (parallel accumulator reduction).  Under the standard
//! floating-point model, a length-`n` dot product has forward error proportional
//! to `γ_n × Σ|a_i b_i|`, where `γ_n = n u / (1 - n u)` and the unit roundoff
//! for round-to-nearest f32 is `u = 0.5 × f32::EPSILON`.  This is a numerical
//! parity contract, not a universal greedy-token identity guarantee; model-level
//! greedy parity is established empirically by profiling.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;
use super::{check_arity, to_dense_i64};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

// Below this many row × key × head-dimension elements, Rayon synchronization
// costs more than the attention work on the decode pool.
const MIN_PARALLEL_ATTENTION_WORK: usize = 160 * 1024;

pub struct GroupQueryAttentionKernel {
    num_heads: usize,
    kv_num_heads: usize,
    scale: Option<f32>,
    do_rotary: bool,
    rotary_interleaved: bool,
    local_window_size: i64,
    softcap: f32,
}

pub struct GroupQueryAttentionFactory;

impl KernelFactory for GroupQueryAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let required_heads = |name: &str| -> Result<usize> {
            let value = node.attr(name).and_then(|a| a.as_int()).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "GroupQueryAttention: missing required `{name}` attribute"
                ))
            })?;
            usize::try_from(value)
                .ok()
                .filter(|&v| v > 0)
                .ok_or_else(|| {
                    EpError::KernelFailed(format!("GroupQueryAttention: `{name}` must be > 0"))
                })
        };
        let num_heads = required_heads("num_heads")?;
        let kv_num_heads = required_heads("kv_num_heads")?;
        if !num_heads.is_multiple_of(kv_num_heads) {
            return Err(EpError::KernelFailed(format!(
                "GroupQueryAttention: num_heads {num_heads} must be a multiple of kv_num_heads {kv_num_heads}"
            )));
        }

        for name in ["k_quant_type", "v_quant_type"] {
            if let Some(value) = node.attr(name)
                && value.as_str() != Some("NONE")
            {
                return Err(EpError::KernelFailed(format!(
                    "GroupQueryAttention: `{name}` other than NONE is not yet supported by the f32 CPU kernel"
                )));
            }
        }
        if node
            .attr("kv_cache_bit_width")
            .and_then(|a| a.as_int())
            .unwrap_or(0)
            != 0
        {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: quantized KV cache is not yet supported".into(),
            ));
        }
        if node.attr("qk_output").and_then(|a| a.as_int()).unwrap_or(0) != 0 {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: qk_output is not yet supported".into(),
            ));
        }
        if node
            .attr("smooth_softmax")
            .and_then(|a| a.as_int())
            .unwrap_or(0)
            != 0
        {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: smooth_softmax is not yet supported".into(),
            ));
        }

        let softcap = node
            .attr("softcap")
            .and_then(|a| a.as_float())
            .unwrap_or(0.0);
        if softcap < 0.0 {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: softcap must be non-negative".into(),
            ));
        }

        Ok(Box::new(GroupQueryAttentionKernel {
            num_heads,
            kv_num_heads,
            scale: node.attr("scale").and_then(|a| a.as_float()),
            do_rotary: node.attr("do_rotary").and_then(|a| a.as_int()).unwrap_or(0) != 0,
            rotary_interleaved: node
                .attr("rotary_interleaved")
                .and_then(|a| a.as_int())
                .unwrap_or(0)
                != 0,
            local_window_size: node
                .attr("local_window_size")
                .and_then(|a| a.as_int())
                .unwrap_or(-1),
            softcap,
        }))
    }
}

struct Bhsd {
    data: Vec<f32>,
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
}

impl Bhsd {
    fn from_bsh(view: &TensorView, heads: usize, name: &str) -> Result<Self> {
        if view.shape.len() != 3 {
            return Err(EpError::KernelFailed(format!(
                "GroupQueryAttention: unpacked {name} must be rank 3 [B,S,H*D], got {:?}",
                view.shape
            )));
        }
        let (batch, seq, hidden) = (view.shape[0], view.shape[1], view.shape[2]);
        if !hidden.is_multiple_of(heads) {
            return Err(EpError::KernelFailed(format!(
                "GroupQueryAttention: {name} hidden size {hidden} is not divisible by {heads} heads"
            )));
        }
        let dim = hidden / heads;
        let src = to_dense_f32_widen("GroupQueryAttention", view)?;
        let mut data = vec![0.0; src.len()];
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..heads {
                    for d in 0..dim {
                        data[((b * heads + h) * seq + s) * dim + d] =
                            src[((b * seq + s) * heads + h) * dim + d];
                    }
                }
            }
        }
        Ok(Self {
            data,
            batch,
            heads,
            seq,
            dim,
        })
    }

    fn from_packed_qkv(
        view: &TensorView,
        num_heads: usize,
        kv_num_heads: usize,
    ) -> Result<(Self, Self, Self)> {
        if view.shape.len() != 3 {
            return Err(EpError::KernelFailed(format!(
                "GroupQueryAttention: packed query must be rank 3 [B,S,(N+2*Nk)*D], got {:?}",
                view.shape
            )));
        }
        let (batch, seq, hidden) = (view.shape[0], view.shape[1], view.shape[2]);
        let packed_heads = num_heads + 2 * kv_num_heads;
        if !hidden.is_multiple_of(packed_heads) {
            return Err(EpError::KernelFailed(format!(
                "GroupQueryAttention: packed QKV hidden size {hidden} is not divisible by num_heads + 2*kv_num_heads ({packed_heads})"
            )));
        }
        let dim = hidden / packed_heads;
        if dim == 0 {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: packed QKV head size must be positive".into(),
            ));
        }

        let src = to_dense_f32_widen("GroupQueryAttention", view)?;
        let q_hidden = num_heads * dim;
        let kv_hidden = kv_num_heads * dim;
        let mut q = vec![0.0; batch * num_heads * seq * dim];
        let mut k = vec![0.0; batch * kv_num_heads * seq * dim];
        let mut v = vec![0.0; k.len()];
        for b in 0..batch {
            for s in 0..seq {
                let src_base = (b * seq + s) * hidden;
                for h in 0..num_heads {
                    for d in 0..dim {
                        q[((b * num_heads + h) * seq + s) * dim + d] = src[src_base + h * dim + d];
                    }
                }
                for h in 0..kv_num_heads {
                    for d in 0..dim {
                        let dst = ((b * kv_num_heads + h) * seq + s) * dim + d;
                        k[dst] = src[src_base + q_hidden + h * dim + d];
                        v[dst] = src[src_base + q_hidden + kv_hidden + h * dim + d];
                    }
                }
            }
        }

        Ok((
            Self {
                data: q,
                batch,
                heads: num_heads,
                seq,
                dim,
            },
            Self {
                data: k,
                batch,
                heads: kv_num_heads,
                seq,
                dim,
            },
            Self {
                data: v,
                batch,
                heads: kv_num_heads,
                seq,
                dim,
            },
        ))
    }
}

/// Borrowed reference to a BNSH KV cache input that widens **incrementally**
/// into the caller's `present` buffer.
///
/// The decode hot path used to widen the entire growing past cache (`f16`→`f32`)
/// into an owned buffer and then copy it again into `present_k`/`present_v` every
/// token — an `O(sequence_length)` widen plus an `O(sequence_length)` copy per
/// step. Profiling attributed ~40% of GroupQueryAttention to that pair. Instead,
/// this keeps only the raw view (for the common contiguous `f16`/`f32` cache) and
/// widens each per-head run *directly into* the destination `present` slice via
/// [`widen_run`](PastCache::widen_run), eliminating the intermediate materialize
/// and the copy. Exotic layouts (strided, `bf16`, `f64`) fall back to a one-time
/// dense widen, so generality is preserved.
struct PastCache<'a> {
    src: PastSrc<'a>,
    seq: usize,
    dim: usize,
    batch: usize,
}

/// Backing storage strategy for a [`PastCache`] head-run widen.
enum PastSrc<'a> {
    /// Contiguous `f32` cache: the run is copied verbatim.
    F32(&'a [f32]),
    /// Contiguous `f16` cache (raw `u16` bits): the run is F16C/scalar widened.
    F16(&'a [u16]),
    /// Non-contiguous or non-`f16`/`f32` cache widened once up front.
    Dense(Vec<f32>),
}

impl<'a> PastCache<'a> {
    fn from_cache(view: &'a TensorView<'a>, heads: usize, name: &str) -> Result<Self> {
        if view.shape.len() != 4 || view.shape[1] != heads {
            return Err(EpError::KernelFailed(format!(
                "GroupQueryAttention: {name} must use BNSH layout [B,{heads},S,D], got {:?}",
                view.shape
            )));
        }
        view.validate()?;
        let len = view.numel();
        let src = if len == 0 {
            PastSrc::Dense(Vec::new())
        } else if view.dtype == onnx_runtime_ir::DataType::Float32 && view.is_contiguous() {
            // SAFETY: a validated contiguous Float32 view addresses exactly `len`
            // initialized f32 elements from `data_ptr`, kept alive for `'a`.
            PastSrc::F32(unsafe { std::slice::from_raw_parts(view.data_ptr::<f32>(), len) })
        } else if view.dtype == onnx_runtime_ir::DataType::Float16 && view.is_contiguous() {
            // SAFETY: a validated contiguous Float16 view addresses exactly `len`
            // 2-byte elements; `half::f16` is `repr(transparent)` over `u16`.
            PastSrc::F16(unsafe { std::slice::from_raw_parts(view.data_ptr::<u16>(), len) })
        } else {
            PastSrc::Dense(to_dense_f32_widen("GroupQueryAttention", view)?.into_owned())
        };
        Ok(Self {
            src,
            seq: view.shape[2],
            dim: view.shape[3],
            batch: view.shape[0],
        })
    }

    /// Widen the contiguous `[start, start + dst.len())` element run of this
    /// cache (row-major BNSH element offsets) into `dst`.
    #[inline]
    fn widen_run(&self, start: usize, dst: &mut [f32]) {
        let len = dst.len();
        match &self.src {
            PastSrc::F32(s) => dst.copy_from_slice(&s[start..start + len]),
            PastSrc::F16(s) => crate::dtype::widen_f16_slice_into(&s[start..start + len], dst),
            PastSrc::Dense(s) => dst.copy_from_slice(&s[start..start + len]),
        }
    }
}

fn scalar_i64(view: &TensorView, name: &str) -> Result<usize> {
    let values = to_dense_i64(view)?;
    if values.len() != 1 || values[0] < 0 {
        return Err(EpError::KernelFailed(format!(
            "GroupQueryAttention: {name} must be one non-negative int32 scalar"
        )));
    }
    Ok(values[0] as usize)
}

fn rotate(
    tensor: &mut Bhsd,
    cos: &[f32],
    sin: &[f32],
    cache_rows: usize,
    positions: &[usize],
    interleaved: bool,
) -> Result<()> {
    if !tensor.dim.is_multiple_of(2) {
        return Err(EpError::KernelFailed(
            "GroupQueryAttention: do_rotary requires an even head_size".into(),
        ));
    }
    let half = tensor.dim / 2;
    if cos.len() != cache_rows * half || sin.len() != cache_rows * half {
        return Err(EpError::KernelFailed(format!(
            "GroupQueryAttention: cos_cache/sin_cache must have shape [max_sequence_length,{half}]"
        )));
    }
    for b in 0..tensor.batch {
        for s in 0..tensor.seq {
            let pos = positions[b * tensor.seq + s];
            if pos >= cache_rows {
                return Err(EpError::KernelFailed(format!(
                    "GroupQueryAttention: rotary position {pos} exceeds cache rows {cache_rows}"
                )));
            }
            for h in 0..tensor.heads {
                for k in 0..half {
                    let (d0, d1) = if interleaved {
                        (2 * k, 2 * k + 1)
                    } else {
                        (k, k + half)
                    };
                    let i0 = ((b * tensor.heads + h) * tensor.seq + s) * tensor.dim + d0;
                    let i1 = ((b * tensor.heads + h) * tensor.seq + s) * tensor.dim + d1;
                    let (x0, x1) = (tensor.data[i0], tensor.data[i1]);
                    let (c, sn) = (cos[pos * half + k], sin[pos * half + k]);
                    tensor.data[i0] = c * x0 - sn * x1;
                    tensor.data[i1] = sn * x0 + c * x1;
                }
            }
        }
    }
    Ok(())
}

fn write_decode_output(out: &mut TensorMut, data: &[f32]) -> Result<()> {
    if out.dtype != onnx_runtime_ir::DataType::Float32 || !out.is_contiguous() {
        return write_dense_f32_narrow("GroupQueryAttention", out, data);
    }
    out.validate()?;
    if out.numel() != data.len() {
        return Err(EpError::KernelFailed(format!(
            "GroupQueryAttention: output element count {} does not match produced {}",
            out.numel(),
            data.len()
        )));
    }
    if data.is_empty() {
        return Ok(());
    }
    // SAFETY: validation plus the contiguous Float32 layout prove the output
    // spans exactly data.len() writable f32 elements.
    let dst = unsafe { std::slice::from_raw_parts_mut(out.data_ptr_mut::<f32>(), data.len()) };
    dst.copy_from_slice(data);
    Ok(())
}

// ── SIMD helpers ─────────────────────────────────────────────────────────────

/// Dot product `sum(a[i] * b[i])` using AVX2+FMA when available, scalar
/// otherwise.  Two AVX2 accumulators hide FMA latency; a scalar tail handles
/// lengths that are not a multiple of 16.
///
/// The AVX2 path reorders f32 additions across the two accumulators relative to
/// a purely sequential scalar sum.  Its standard forward-error scale is
/// `γ_n × Σ|a_i b_i|`, where `γ_n = n u / (1 - n u)` and
/// `u = 0.5 × f32::EPSILON`; cancellation can therefore make a relative-error
/// bound inappropriate.
#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if crate::backend::has_simd_x86() {
        // SAFETY: `has_simd_x86()` confirms AVX2 + FMA at runtime.
        return unsafe { dot_avx2_fma(a, b) };
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// AXPY `dst[d] += scalar * src[d]` for all d, using AVX2+FMA when available.
///
/// Used for the probability-weighted V accumulation (P·V step).  The inner
/// loop is over `head_dim` contiguous f32s, which maps directly to 256-bit
/// vector FMADD instructions.
#[inline(always)]
fn axpy_f32(dst: &mut [f32], scalar: f32, src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if crate::backend::has_simd_x86() {
        // SAFETY: `has_simd_x86()` confirms AVX2 + FMA at runtime.
        unsafe { axpy_avx2_fma(dst, scalar, src) };
        return;
    }
    for (d, s) in dst.iter_mut().zip(src) {
        *d += scalar * s;
    }
}

/// AVX2+FMA dot product.  Two accumulators hide the 5-cycle FMA latency on
/// Sapphire Rapids; the 8-lane reduction is a standard 4→2→1 horizontal add.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        // Process 16 elements per iteration (two 8-wide AVX2 loads + FMAs).
        let chunks16 = n / 16;
        for i in 0..chunks16 {
            let av0 = _mm256_loadu_ps(a_ptr.add(i * 16));
            let bv0 = _mm256_loadu_ps(b_ptr.add(i * 16));
            acc0 = _mm256_fmadd_ps(av0, bv0, acc0);
            let av1 = _mm256_loadu_ps(a_ptr.add(i * 16 + 8));
            let bv1 = _mm256_loadu_ps(b_ptr.add(i * 16 + 8));
            acc1 = _mm256_fmadd_ps(av1, bv1, acc1);
        }

        // Remaining 8-element chunk (if any).
        let mut tail = chunks16 * 16;
        if tail + 8 <= n {
            let av = _mm256_loadu_ps(a_ptr.add(tail));
            let bv = _mm256_loadu_ps(b_ptr.add(tail));
            acc0 = _mm256_fmadd_ps(av, bv, acc0);
            tail += 8;
        }

        // Merge the two accumulators.
        let acc = _mm256_add_ps(acc0, acc1);

        // Horizontal reduce: 8 → 4 → 2 → 1 lane.
        let lo = _mm256_extractf128_ps(acc, 0);
        let hi = _mm256_extractf128_ps(acc, 1);
        let v4 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(v4);
        let v2 = _mm_add_ps(v4, shuf);
        let shuf2 = _mm_movehl_ps(shuf, v2);
        let v1 = _mm_add_ss(v2, shuf2);
        let mut result = _mm_cvtss_f32(v1);

        // Scalar tail for lengths not a multiple of 8.
        for i in tail..n {
            result += *a_ptr.add(i) * *b_ptr.add(i);
        }
        result
    }
}

/// AVX2+FMA AXPY: `dst[d] += scalar * src[d]` for all d.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2_fma(dst: &mut [f32], scalar: f32, src: &[f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = dst.len();
    let s = _mm256_set1_ps(scalar);
    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();

    unsafe {
        let mut i = 0;
        while i + 8 <= n {
            let d = _mm256_loadu_ps(dst_ptr.add(i));
            let x = _mm256_loadu_ps(src_ptr.add(i));
            _mm256_storeu_ps(dst_ptr.add(i), _mm256_fmadd_ps(s, x, d));
            i += 8;
        }
        // Scalar tail.
        while i < n {
            *dst_ptr.add(i) += scalar * *src_ptr.add(i);
            i += 1;
        }
    }
}

// ── temporary within-GQA phase profiling (gated by ONNX_GENAI_PROFILE_GQA) ────
#[cfg(feature = "gqa_phase_profile")]
mod phase_prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    pub static WIDEN_NS: AtomicU64 = AtomicU64::new(0);
    pub static PRESENT_NS: AtomicU64 = AtomicU64::new(0);
    pub static ATTN_NS: AtomicU64 = AtomicU64::new(0);
    pub static OUT_NS: AtomicU64 = AtomicU64::new(0);
    pub static CALLS: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *E.get_or_init(|| std::env::var("ONNX_GENAI_PROFILE_GQA").is_ok_and(|v| v == "1"))
    }

    pub struct Phase(Option<(Instant, &'static AtomicU64)>);
    impl Phase {
        pub fn start(acc: &'static AtomicU64) -> Self {
            if enabled() {
                Phase(Some((Instant::now(), acc)))
            } else {
                Phase(None)
            }
        }
    }
    impl Drop for Phase {
        fn drop(&mut self) {
            if let Some((t, acc)) = self.0 {
                acc.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
        }
    }

    pub fn tick() {
        if !enabled() {
            return;
        }
        let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        if calls.is_multiple_of(240) {
            let w = WIDEN_NS.load(Ordering::Relaxed) as f64 / 1e6;
            let p = PRESENT_NS.load(Ordering::Relaxed) as f64 / 1e6;
            let a = ATTN_NS.load(Ordering::Relaxed) as f64 / 1e6;
            let o = OUT_NS.load(Ordering::Relaxed) as f64 / 1e6;
            let tot = w + p + a + o;
            eprintln!(
                "[gqa-phase] calls={calls} total={tot:.1}ms widen={w:.1}ms({wp:.1}%) present={p:.1}ms({pp:.1}%) attn={a:.1}ms({ap:.1}%) out={o:.1}ms({op:.1}%)",
                wp = 100.0 * w / tot,
                pp = 100.0 * p / tot,
                ap = 100.0 * a / tot,
                op = 100.0 * o / tot,
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

impl Kernel for GroupQueryAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("GroupQueryAttention", inputs, outputs, 7, 14, 1)?;
        if outputs.len() > 3 {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: output_qk is not yet supported".into(),
            ));
        }
        let packed_qkv = inputs[1].is_absent() && inputs[2].is_absent();
        if inputs[1].is_absent() != inputs[2].is_absent() {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: key and value must both be present for unpacked Q/K/V or both absent for packed QKV".into(),
            ));
        }
        if inputs.get(10).is_some_and(|v| !v.is_absent()) {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: attention_bias is not yet supported".into(),
            ));
        }
        if inputs.get(11).is_some_and(|v| !v.is_absent()) {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: head_sink is not yet supported".into(),
            ));
        }
        if inputs.get(12).is_some_and(|v| !v.is_absent())
            || inputs.get(13).is_some_and(|v| !v.is_absent())
        {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: quantized-cache k_scale/v_scale inputs are not yet supported"
                    .into(),
            ));
        }
        if self.local_window_size == 0 || self.local_window_size < -1 {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: local_window_size must be -1 or a positive integer".into(),
            ));
        }

        let (mut q, mut k, v) = if packed_qkv {
            Bhsd::from_packed_qkv(&inputs[0], self.num_heads, self.kv_num_heads)?
        } else {
            (
                Bhsd::from_bsh(&inputs[0], self.num_heads, "query")?,
                Bhsd::from_bsh(&inputs[1], self.kv_num_heads, "key")?,
                Bhsd::from_bsh(&inputs[2], self.kv_num_heads, "value")?,
            )
        };
        if q.batch != k.batch
            || q.batch != v.batch
            || k.seq != v.seq
            || k.dim != q.dim
            || v.dim != q.dim
        {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: incompatible query/key/value batch, sequence, or head dimensions".into(),
            ));
        }

        let has_past_key = !inputs[3].is_absent();
        let has_past_value = !inputs[4].is_absent();
        if has_past_key != has_past_value {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: past_key and past_value must be provided together".into(),
            ));
        }
        #[cfg(feature = "gqa_phase_profile")]
        let _widen_phase = phase_prof::Phase::start(&phase_prof::WIDEN_NS);
        let past_key = has_past_key
            .then(|| PastCache::from_cache(&inputs[3], self.kv_num_heads, "past_key"))
            .transpose()?;
        let past_value = has_past_value
            .then(|| PastCache::from_cache(&inputs[4], self.kv_num_heads, "past_value"))
            .transpose()?;
        #[cfg(feature = "gqa_phase_profile")]
        drop(_widen_phase);
        if let (Some(pk), Some(pv)) = (&past_key, &past_value)
            && (pk.batch != q.batch
                || pv.batch != q.batch
                || pk.seq != pv.seq
                || pk.dim != q.dim
                || pv.dim != q.dim)
        {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: past_key/past_value dimensions are incompatible with Q/K/V"
                    .into(),
            ));
        }

        let seqlens = to_dense_i64(&inputs[5])?;
        if seqlens.len() != q.batch || seqlens.iter().any(|&x| x < 0) {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: seqlens_k must be non-negative int32 [batch_size]".into(),
            ));
        }
        let total_sequence_length = scalar_i64(&inputs[6], "total_sequence_length")?;
        let totals: Vec<usize> = seqlens.iter().map(|&x| x as usize + 1).collect();
        if totals.iter().copied().max().unwrap_or(0) != total_sequence_length {
            return Err(EpError::KernelFailed(format!(
                "GroupQueryAttention: total_sequence_length {total_sequence_length} must equal max(seqlens_k + 1)"
            )));
        }
        let mut past_lengths = Vec::with_capacity(q.batch);
        let mut query_starts = Vec::with_capacity(q.batch);
        for &total in &totals {
            let past = total.checked_sub(k.seq).ok_or_else(|| {
                EpError::KernelFailed(
                    "GroupQueryAttention: seqlens_k + 1 is shorter than current key sequence"
                        .into(),
                )
            })?;
            if past > past_key.as_ref().map_or(0, |cache| cache.seq) {
                return Err(EpError::KernelFailed(
                    "GroupQueryAttention: effective past length exceeds past cache extent".into(),
                ));
            }
            past_lengths.push(past);
            query_starts.push(total.checked_sub(q.seq).ok_or_else(|| {
                EpError::KernelFailed(
                    "GroupQueryAttention: total sequence is shorter than query sequence".into(),
                )
            })?);
        }

        if self.do_rotary {
            let cos_view = inputs.get(7).filter(|v| !v.is_absent()).ok_or_else(|| {
                EpError::KernelFailed("GroupQueryAttention: do_rotary=1 requires cos_cache".into())
            })?;
            let sin_view = inputs.get(8).filter(|v| !v.is_absent()).ok_or_else(|| {
                EpError::KernelFailed("GroupQueryAttention: do_rotary=1 requires sin_cache".into())
            })?;
            if cos_view.shape.len() != 2 || sin_view.shape != cos_view.shape {
                return Err(EpError::KernelFailed(
                    "GroupQueryAttention: cos_cache and sin_cache must have equal rank-2 shapes"
                        .into(),
                ));
            }
            if cos_view.shape[1] != q.dim / 2 {
                return Err(EpError::KernelFailed(format!(
                    "GroupQueryAttention: cos_cache/sin_cache second dimension must be head_size/2 ({})",
                    q.dim / 2
                )));
            }
            let explicit_position_ids = inputs.get(9).filter(|v| !v.is_absent());
            let query_positions = if let Some(position_ids) = explicit_position_ids {
                let ids = to_dense_i64(position_ids)?;
                if position_ids.shape != [q.batch, q.seq] || ids.iter().any(|&x| x < 0) {
                    return Err(EpError::KernelFailed(
                        "GroupQueryAttention: position_ids must be non-negative int64 [batch_size, sequence_length]".into(),
                    ));
                }
                ids.into_iter().map(|x| x as usize).collect()
            } else {
                let mut ids = Vec::with_capacity(q.batch * q.seq);
                for &total in &totals {
                    let start = total.checked_sub(q.seq).ok_or_else(|| {
                        EpError::KernelFailed(
                            "GroupQueryAttention: total sequence is shorter than query sequence"
                                .into(),
                        )
                    })?;
                    ids.extend((0..q.seq).map(|s| start + s));
                }
                ids
            };
            let key_positions = if explicit_position_ids.is_some() && k.seq == q.seq {
                query_positions.clone()
            } else {
                let mut ids = Vec::with_capacity(k.batch * k.seq);
                for &total in &totals {
                    let start = total.checked_sub(k.seq).ok_or_else(|| {
                        EpError::KernelFailed(
                            "GroupQueryAttention: total sequence is shorter than key sequence"
                                .into(),
                        )
                    })?;
                    ids.extend((0..k.seq).map(|s| start + s));
                }
                ids
            };
            let cos = to_dense_f32_widen("GroupQueryAttention", cos_view)?;
            let sin = to_dense_f32_widen("GroupQueryAttention", sin_view)?;
            rotate(
                &mut q,
                &cos,
                &sin,
                cos_view.shape[0],
                &query_positions,
                self.rotary_interleaved,
            )?;
            rotate(
                &mut k,
                &cos,
                &sin,
                cos_view.shape[0],
                &key_positions,
                self.rotary_interleaved,
            )?;
        }

        let cache_dim = q.dim;
        #[cfg(feature = "gqa_phase_profile")]
        let _present_phase = phase_prof::Phase::start(&phase_prof::PRESENT_NS);
        let present_sequence_length = past_key.as_ref().map_or(total_sequence_length, |cache| {
            cache.seq.max(total_sequence_length)
        });
        let present_len = q.batch * self.kv_num_heads * present_sequence_length * cache_dim;
        // A "tail" is any padding row beyond a batch's logical `total` that is
        // emitted into the present output but never attended; those rows must be
        // zero. In steady decode every batch's `total` exactly fills
        // `present_sequence_length`, so the per-(b,h) loop below overwrites every
        // element and pre-zeroing is pure waste — skip it in that case.
        let has_tail = totals.iter().any(|&t| t < present_sequence_length);
        let (mut present_k, mut present_v) = if has_tail {
            (vec![0.0f32; present_len], vec![0.0f32; present_len])
        } else {
            let mut present_k = Vec::<f32>::with_capacity(present_len);
            let mut present_v = Vec::<f32>::with_capacity(present_len);
            // SAFETY: `!has_tail` ⇒ every batch's `total == present_sequence_length`,
            // so for each `(b, h)` the loop below writes the past prefix
            // `[0, past_len)` and the current span `[past_len, total)` =
            // `[0, present_sequence_length)` rows, i.e. every element of both
            // buffers, before any read (attention and the output narrow both run
            // strictly after this loop). No uninitialized element is observed.
            unsafe {
                present_k.set_len(present_len);
                present_v.set_len(present_len);
            }
            (present_k, present_v)
        };
        for (b, &past_len) in past_lengths.iter().enumerate() {
            for h in 0..self.kv_num_heads {
                let head = b * self.kv_num_heads + h;
                let dst_base = head * present_sequence_length * cache_dim;
                // `present_k`/`present_v` and the past caches are both
                // BNSH-contiguous, so for a fixed (b, h) the `[s, d]` block is a
                // single contiguous run in each: widen the whole past prefix
                // directly into `present` (F16C for f16), skipping the separate
                // owned widen + f32 copy the decode path used to pay every token.
                if past_len > 0 {
                    let copy = past_len * cache_dim;
                    let pk = past_key.as_ref().unwrap();
                    let pv = past_value.as_ref().unwrap();
                    let src = head * pk.seq * cache_dim;
                    pk.widen_run(src, &mut present_k[dst_base..dst_base + copy]);
                    pv.widen_run(src, &mut present_v[dst_base..dst_base + copy]);
                }
                // Append the current token(s) directly after the past prefix;
                // the current K/V blocks are contiguous in `[s, d]` as well.
                let cur = k.seq * cache_dim;
                let dst_cur = dst_base + past_len * cache_dim;
                let src_cur = head * k.seq * cache_dim;
                present_k[dst_cur..dst_cur + cur].copy_from_slice(&k.data[src_cur..src_cur + cur]);
                present_v[dst_cur..dst_cur + cur].copy_from_slice(&v.data[src_cur..src_cur + cur]);
            }
        }

        let scale = self
            .scale
            .filter(|&scale| scale != 0.0)
            .unwrap_or_else(|| 1.0 / (cache_dim as f32).sqrt());
        #[cfg(feature = "gqa_phase_profile")]
        {
            drop(_present_phase);
        }
        #[cfg(feature = "gqa_phase_profile")]
        let _attn_phase = phase_prof::Phase::start(&phase_prof::ATTN_NS);
        let group = self.num_heads / self.kv_num_heads;
        let attention_rows = q.batch * q.seq * self.num_heads;
        let mut y_bhsd = vec![0.0; attention_rows * v.dim];
        let compute_row = |b: usize, qh: usize, qs: usize, output_row: &mut [f32]| {
            let kvh = qh / group;
            let causal_limit = query_starts[b] + qs;
            let local_start = if self.local_window_size > 0 {
                (causal_limit + 1).saturating_sub(self.local_window_size as usize)
            } else {
                0
            };
            // Number of keys in the attended causal window [local_start, causal_limit].
            let attended = causal_limit + 1 - local_start;

            // Extract the query row slice once to avoid per-element index arithmetic
            // inside the scoring loop.
            let q_base = ((b * self.num_heads + qh) * q.seq + qs) * cache_dim;
            let q_row = &q.data[q_base..q_base + cache_dim];

            // Base sequence index for this (batch, kv_head) in present_k / present_v.
            let kv_head_stride = (b * self.kv_num_heads + kvh) * present_sequence_length;

            // ── QK scores: dot(q_row, k_row) for each key in the attended window ──
            // Allocate only `attended` elements rather than `total_sequence_length`
            // so unattended positions are never touched.
            let mut scores = vec![0.0f32; attended];
            for (i, ks) in (local_start..=causal_limit).enumerate() {
                let k_base = (kv_head_stride + ks) * cache_dim;
                let k_row = &present_k[k_base..k_base + cache_dim];
                let mut score = dot_f32(q_row, k_row);
                score *= scale;
                if self.softcap != 0.0 {
                    score = self.softcap * (score / self.softcap).tanh();
                }
                scores[i] = score;
            }

            // ── Softmax over the attended window ──
            // PRECISION CONTRACT (RULES.md §4): the f64 exp + single f32 rounding
            // path matches CUDA's device-side computation and is kept unchanged.
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for score in &mut scores {
                *score = ((*score - max) as f64).exp() as f32;
                sum += *score;
            }
            // Normalize once so P·V can multiply without per-element division.
            if sum > 0.0 {
                for score in &mut scores {
                    *score /= sum;
                }
            }

            // ── P·V accumulation: cache-friendly AXPY (ks-outer, d-inner) ──
            // Loop order: ks outer (sequential through probability weights),
            // d inner (contiguous in both the V row and output_row) →
            // sequential cache access + AVX2 FMADD via axpy_f32.
            output_row.fill(0.0);
            for (i, ks) in (local_start..=causal_limit).enumerate() {
                let prob = scores[i];
                if prob == 0.0 {
                    continue;
                }
                let v_base = (kv_head_stride + ks) * v.dim;
                let v_row = &present_v[v_base..v_base + v.dim];
                axpy_f32(output_row, prob, v_row);
            }
        };
        let attention_work = attention_rows
            .saturating_mul(total_sequence_length)
            .saturating_mul(cache_dim);
        if attention_rows > 1 && attention_work >= MIN_PARALLEL_ATTENTION_WORK {
            // Route through the active decode pool (the same resident workers the
            // MatMulNBits projections use). Under the persistent SPMD scope this
            // avoids falling to the global Rayon pool, which would contend with
            // the SPMD pool's pinned, spinning workers; under numa-split/flat
            // scopes it runs on their bounded pool exactly as before.
            crate::kernels::matmul_nbits::decode_parallel_output_row_blocks(
                &mut y_bhsd,
                v.dim,
                attention_rows,
                |row_index, output_row| {
                    let b = row_index / (self.num_heads * q.seq);
                    let row_in_batch = row_index % (self.num_heads * q.seq);
                    let qh = row_in_batch / q.seq;
                    let qs = row_in_batch % q.seq;
                    compute_row(b, qh, qs, output_row);
                },
            );
        } else {
            for b in 0..q.batch {
                for qh in 0..self.num_heads {
                    for qs in 0..q.seq {
                        let row_index = (b * self.num_heads + qh) * q.seq + qs;
                        compute_row(
                            b,
                            qh,
                            qs,
                            &mut y_bhsd[row_index * v.dim..(row_index + 1) * v.dim],
                        );
                    }
                }
            }
        }
        let mut output = vec![0.0; y_bhsd.len()];
        #[cfg(feature = "gqa_phase_profile")]
        {
            drop(_attn_phase);
        }
        #[cfg(feature = "gqa_phase_profile")]
        let _out_phase = phase_prof::Phase::start(&phase_prof::OUT_NS);
        for b in 0..q.batch {
            for s in 0..q.seq {
                for h in 0..self.num_heads {
                    for d in 0..v.dim {
                        output[((b * q.seq + s) * self.num_heads + h) * v.dim + d] =
                            y_bhsd[((b * self.num_heads + h) * q.seq + s) * v.dim + d];
                    }
                }
            }
        }
        let decode_fast_write = q.seq == 1 && k.seq == 1;
        if decode_fast_write {
            write_decode_output(&mut outputs[0], &output)?;
        } else {
            write_dense_f32_narrow("GroupQueryAttention", &mut outputs[0], &output)?;
        }
        if outputs.len() >= 2 {
            if decode_fast_write {
                write_decode_output(&mut outputs[1], &present_k)?;
            } else {
                write_dense_f32_narrow("GroupQueryAttention", &mut outputs[1], &present_k)?;
            }
        }
        if outputs.len() >= 3 {
            if decode_fast_write {
                write_decode_output(&mut outputs[2], &present_v)?;
            } else {
                write_dense_f32_narrow("GroupQueryAttention", &mut outputs[2], &present_v)?;
            }
        }
        #[cfg(feature = "gqa_phase_profile")]
        {
            drop(_out_phase);
            phase_prof::tick();
        }
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_simd_x86_for_test() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            crate::backend::has_simd_x86()
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::{ExecutionProvider, TensorView};
    use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, static_shape};
    use onnx_runtime_loader::Model;

    fn absent() -> TensorView<'static> {
        TensorView::absent(DataType::Float32)
    }

    fn kernel(attrs: &[(&str, Attribute)]) -> Box<dyn Kernel> {
        let mut graph = Graph::new();
        graph.opset_imports.insert("com.microsoft".into(), 1);
        let inputs = [
            ("query", DataType::Float32, vec![1, 1, 8]),
            ("key", DataType::Float32, vec![1, 1, 4]),
            ("value", DataType::Float32, vec![1, 1, 4]),
            ("past_key", DataType::Float32, vec![1, 2, 0, 2]),
            ("past_value", DataType::Float32, vec![1, 2, 0, 2]),
            ("seqlens_k", DataType::Int32, vec![1]),
            ("total_sequence_length", DataType::Int32, vec![]),
        ]
        .into_iter()
        .map(|(name, dtype, shape)| {
            let value = graph.create_named_value(name, dtype, static_shape(shape));
            graph.add_input(value);
            Some(value)
        })
        .collect();
        let output = graph.create_named_value("output", DataType::Float32, static_shape([1, 1, 8]));
        let mut node = Node::new(NodeId(0), "GroupQueryAttention", inputs, vec![output]);
        node.domain = "com.microsoft".into();
        for (name, value) in attrs {
            node.attributes.insert((*name).into(), value.clone());
        }
        let id = graph.insert_node(node);
        graph.add_output(output);
        let model = Model::new(&graph);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(id), &[], 1)
            .unwrap()
    }

    fn gqa_kernel(extra: &[(&str, Attribute)]) -> Box<dyn Kernel> {
        let mut attrs = vec![
            ("num_heads", Attribute::Int(4)),
            ("kv_num_heads", Attribute::Int(2)),
        ];
        attrs.extend_from_slice(extra);
        kernel(&attrs)
    }

    fn reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_seq: usize,
        total: usize,
        past: usize,
    ) -> Vec<f32> {
        let (qh, kvh, d) = (4, 2, 2);
        let mut out = vec![0.0; q_seq * qh * d];
        for s in 0..q_seq {
            for h in 0..qh {
                let kh = h / (qh / kvh);
                let mut scores = vec![0.0; past + s + 1];
                for j in 0..scores.len() {
                    scores[j] = (0..d)
                        .map(|x| q[(s * qh + h) * d + x] * k[(kh * total + j) * d + x])
                        .sum::<f32>()
                        / (d as f32).sqrt();
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores
                    .iter_mut()
                    .map(|x| {
                        *x = ((*x - max) as f64).exp() as f32;
                        *x
                    })
                    .sum();
                for x in &mut scores {
                    *x /= sum;
                }
                for x in 0..d {
                    out[(s * qh + h) * d + x] = scores
                        .iter()
                        .enumerate()
                        .map(|(j, p)| p * v[(kh * total + j) * d + x])
                        .sum();
                }
            }
        }
        out
    }

    fn reference_with_geometry(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        query_sequence_length: usize,
        total_sequence_length: usize,
        past_sequence_length: usize,
        query_head_count: usize,
        key_value_head_count: usize,
        head_width: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; query_sequence_length * query_head_count * head_width];
        for sequence_index in 0..query_sequence_length {
            for query_head_index in 0..query_head_count {
                let key_value_head_index =
                    query_head_index / (query_head_count / key_value_head_count);
                let attended_key_count = past_sequence_length + sequence_index + 1;
                let mut scores = vec![0.0; attended_key_count];
                for (key_index, score) in scores.iter_mut().enumerate() {
                    let query_base =
                        (sequence_index * query_head_count + query_head_index) * head_width;
                    let key_base =
                        (key_value_head_index * total_sequence_length + key_index) * head_width;
                    *score = query[query_base..query_base + head_width]
                        .iter()
                        .zip(&key[key_base..key_base + head_width])
                        .map(|(query_element, key_element)| query_element * key_element)
                        .sum::<f32>()
                        / (head_width as f32).sqrt();
                }
                let maximum_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let probability_sum: f32 = scores
                    .iter_mut()
                    .map(|score| {
                        *score = ((*score - maximum_score) as f64).exp() as f32;
                        *score
                    })
                    .sum();
                for score in &mut scores {
                    *score /= probability_sum;
                }
                let output_base =
                    (sequence_index * query_head_count + query_head_index) * head_width;
                for dimension_index in 0..head_width {
                    output[output_base + dimension_index] = scores
                        .iter()
                        .enumerate()
                        .map(|(key_index, probability)| {
                            probability
                                * value[(key_value_head_index * total_sequence_length + key_index)
                                    * head_width
                                    + dimension_index]
                        })
                        .sum();
                }
            }
        }
        output
    }

    fn mixed_scale_value(index: usize, seed: u64) -> f32 {
        let mut state = (index as u64)
            .wrapping_add(seed)
            .wrapping_add(0x9e37_79b9_7f4a_7c15);
        state ^= state >> 30;
        state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        state ^= state >> 27;
        state = state.wrapping_mul(0x94d0_49bb_1331_11eb);
        state ^= state >> 31;
        let signed_unit = (((state >> 40) as u32) as f32 / ((1_u32 << 24) as f32)) * 2.0 - 1.0;
        let scale = [0.03125_f32, 0.125, 0.5, 2.0][((state >> 8) & 3) as usize];
        signed_unit * scale
    }

    fn dot_comparison_tolerance(left: &[f32], right: &[f32]) -> f32 {
        let unit_roundoff = 0.5 * f32::EPSILON;
        let operation_count = left.len() as f32;
        let gamma = operation_count * unit_roundoff / (1.0 - operation_count * unit_roundoff);
        let absolute_product_sum: f32 = left
            .iter()
            .zip(right)
            .map(|(left_element, right_element)| (left_element * right_element).abs())
            .sum();
        2.0 * gamma * absolute_product_sum + 2.0 * f32::MIN_POSITIVE
    }

    /// Attention reference that mirrors the production kernel's *exact*
    /// arithmetic — score = `dot(q,k) * (1/sqrt(head_width))` (the reciprocal
    /// `scale` the kernel multiplies by, not a `/sqrt` divide), the `f64`-exp
    /// softmax, and ascending-key accumulation. Because it differs from the
    /// kernel only in *how the KV cache is widened* (this takes an already-dense
    /// f32 cache; the kernel widens f16 incrementally), it isolates the widen
    /// path: any output difference is a widen bug, not a softmax-ordering
    /// artifact. `key`/`value` are BNSH `[kv_heads, total, head_width]`.
    fn kernel_exact_reference(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        total_sequence_length: usize,
        past_sequence_length: usize,
        query_head_count: usize,
        key_value_head_count: usize,
        head_width: usize,
    ) -> Vec<f32> {
        let scale = 1.0f32 / (head_width as f32).sqrt();
        let group = query_head_count / key_value_head_count;
        let mut output = vec![0.0f32; query_head_count * head_width];
        for query_head in 0..query_head_count {
            let kv_head = query_head / group;
            let attended = past_sequence_length + 1;
            let query_base = query_head * head_width;
            let mut scores = vec![0.0f32; attended];
            for (key_index, score) in scores.iter_mut().enumerate() {
                let key_base = (kv_head * total_sequence_length + key_index) * head_width;
                *score = query[query_base..query_base + head_width]
                    .iter()
                    .zip(&key[key_base..key_base + head_width])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for score in &mut scores {
                *score = ((*score - max) as f64).exp() as f32;
                sum += *score;
            }
            if sum > 0.0 {
                for score in &mut scores {
                    *score /= sum;
                }
            }
            let output_base = query_head * head_width;
            for (key_index, &prob) in scores.iter().enumerate() {
                let value_base = (kv_head * total_sequence_length + key_index) * head_width;
                for dimension in 0..head_width {
                    output[output_base + dimension] += prob * value[value_base + dimension];
                }
            }
        }
        output
    }

    fn close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            assert!((a - b).abs() < 1e-5, "{i}: {a} != {b}");
        }
    }

    /// Relative-tolerance comparison for kernel-vs-scalar-reference attention
    /// output. The scalar reference divides the score by `sqrt(head_width)`
    /// while the kernel multiplies by the reciprocal `scale`; those differ in
    /// the last bit and softmax amplifies the difference, so an absolute 1e-5
    /// bound is too tight (see also `gqa_decode_long_context_matches_reference`).
    fn close_relative(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            let tolerance = 2.0e-5 + 2.0e-5 * b.abs();
            assert!(
                (a - b).abs() <= tolerance,
                "{i}: {a} != {b} (difference {}, tolerance {tolerance})",
                (a - b).abs()
            );
        }
    }

    fn reference_rope_bsh(
        input: &[f32],
        seq: usize,
        heads: usize,
        positions: &[usize],
        cos: &[f32],
        sin: &[f32],
    ) -> Vec<f32> {
        let mut output = input.to_vec();
        for s in 0..seq {
            for h in 0..heads {
                let base = (s * heads + h) * 2;
                let (x0, x1) = (input[base], input[base + 1]);
                output[base] = cos[positions[s]] * x0 - sin[positions[s]] * x1;
                output[base + 1] = sin[positions[s]] * x0 + cos[positions[s]] * x1;
            }
        }
        output
    }

    fn bsh_to_bnsh(input: &[f32], seq: usize, heads: usize) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        for s in 0..seq {
            for h in 0..heads {
                output[(h * seq + s) * 2..(h * seq + s + 1) * 2]
                    .copy_from_slice(&input[(s * heads + h) * 2..(s * heads + h + 1) * 2]);
            }
        }
        output
    }

    #[test]
    fn prefill_gqa_grouping_and_causal_match_reference() {
        let q = vec![
            1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1., -1.,
            1., -1., 1.,
        ];
        let k_bsh = vec![1., 0., 0., 1., 0., 1., 1., 0., 1., 1., -1., 1.];
        let v_bsh = vec![1., 2., 10., 20., 3., 4., 30., 40., 5., 6., 50., 60.];
        let mut k_bnsh = vec![0.0; 12];
        let mut v_bnsh = vec![0.0; 12];
        for h in 0..2 {
            for s in 0..3 {
                for d in 0..2 {
                    k_bnsh[(h * 3 + s) * 2 + d] = k_bsh[(s * 2 + h) * 2 + d];
                    v_bnsh[(h * 3 + s) * 2 + d] = v_bsh[(s * 2 + h) * 2 + d];
                }
            }
        }
        let mut out = Owned::zeros_f32(&[1, 3, 8]);
        let mut pk = Owned::zeros_f32(&[1, 2, 3, 2]);
        let mut pv = Owned::zeros_f32(&[1, 2, 3, 2]);
        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[1, 3, 8], &q).view(),
                    Owned::f32(&[1, 3, 4], &k_bsh).view(),
                    Owned::f32(&[1, 3, 4], &v_bsh).view(),
                    absent(),
                    absent(),
                    Owned::i32(&[1], &[2]).view(),
                    Owned::i32(&[], &[3]).view(),
                ],
                &mut [out.view_mut(), pk.view_mut(), pv.view_mut()],
            )
            .unwrap();
        close(&out.to_f32(), &reference(&q, &k_bnsh, &v_bnsh, 3, 3, 0));
        assert_eq!(pk.shape, vec![1, 2, 3, 2]);
        close(&pk.to_f32(), &k_bnsh);
        close(&pv.to_f32(), &v_bnsh);
    }

    #[test]
    fn unit_batch_scalar_seqlens_matches_canonical_vector() {
        let q = [1., 0., 1., 0., 0., 1., 0., 1.];
        let k = [1., 0., 0., 1.];
        let v = [1., 2., 10., 20.];
        let run = |seqlens_shape: &[usize]| {
            let mut out = Owned::zeros_f32(&[1, 1, 8]);
            let mut present_k = Owned::zeros_f32(&[1, 2, 1, 2]);
            let mut present_v = Owned::zeros_f32(&[1, 2, 1, 2]);
            gqa_kernel(&[])
                .execute(
                    &[
                        Owned::f32(&[1, 1, 8], &q).view(),
                        Owned::f32(&[1, 1, 4], &k).view(),
                        Owned::f32(&[1, 1, 4], &v).view(),
                        absent(),
                        absent(),
                        Owned::i32(seqlens_shape, &[0]).view(),
                        Owned::i32(&[], &[1]).view(),
                    ],
                    &mut [out.view_mut(), present_k.view_mut(), present_v.view_mut()],
                )
                .unwrap();
            (out.to_f32(), present_k.to_f32(), present_v.to_f32())
        };

        assert_eq!(run(&[]), run(&[1]));
    }

    #[test]
    fn large_prefill_parallel_path_matches_reference() {
        let seq = 160;
        let q = (0..seq * 8)
            .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
            .collect::<Vec<_>>();
        let k_bsh = (0..seq * 4)
            .map(|i| ((i % 13) as f32 - 6.0) / 7.0)
            .collect::<Vec<_>>();
        let v_bsh = (0..seq * 4)
            .map(|i| ((i % 19) as f32 - 9.0) / 9.0)
            .collect::<Vec<_>>();
        let k_bnsh = bsh_to_bnsh(&k_bsh, seq, 2);
        let v_bnsh = bsh_to_bnsh(&v_bsh, seq, 2);
        let mut out = Owned::zeros_f32(&[1, seq, 8]);

        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[1, seq, 8], &q).view(),
                    Owned::f32(&[1, seq, 4], &k_bsh).view(),
                    Owned::f32(&[1, seq, 4], &v_bsh).view(),
                    absent(),
                    absent(),
                    Owned::i32(&[1], &[(seq - 1) as i32]).view(),
                    Owned::i32(&[], &[seq as i32]).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();

        close(&out.to_f32(), &reference(&q, &k_bnsh, &v_bnsh, seq, seq, 0));
    }

    #[test]
    fn packed_qkv_matches_unpacked_and_independent_reference() {
        let q = vec![
            1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 1., 1., 0., 1., 0.,
        ];
        let k_bsh = vec![1., 0., 0., 1., 0., 1., 1., 0.];
        let v_bsh = vec![1., 2., 10., 20., 3., 4., 30., 40.];
        let mut packed = Vec::with_capacity(q.len() + k_bsh.len() + v_bsh.len());
        for s in 0..2 {
            packed.extend_from_slice(&q[s * 8..(s + 1) * 8]);
            packed.extend_from_slice(&k_bsh[s * 4..(s + 1) * 4]);
            packed.extend_from_slice(&v_bsh[s * 4..(s + 1) * 4]);
        }
        let k_bnsh = bsh_to_bnsh(&k_bsh, 2, 2);
        let v_bnsh = bsh_to_bnsh(&v_bsh, 2, 2);
        let want = reference(&q, &k_bnsh, &v_bnsh, 2, 2, 0);

        let mut unpacked_out = Owned::zeros_f32(&[1, 2, 8]);
        let mut packed_out = Owned::zeros_f32(&[1, 2, 8]);
        let mut unpacked_k = Owned::zeros_f32(&[1, 2, 2, 2]);
        let mut unpacked_v = Owned::zeros_f32(&[1, 2, 2, 2]);
        let mut packed_k = Owned::zeros_f32(&[1, 2, 2, 2]);
        let mut packed_v = Owned::zeros_f32(&[1, 2, 2, 2]);
        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[1, 2, 8], &q).view(),
                    Owned::f32(&[1, 2, 4], &k_bsh).view(),
                    Owned::f32(&[1, 2, 4], &v_bsh).view(),
                    absent(),
                    absent(),
                    Owned::i32(&[1], &[1]).view(),
                    Owned::i32(&[], &[2]).view(),
                ],
                &mut [
                    unpacked_out.view_mut(),
                    unpacked_k.view_mut(),
                    unpacked_v.view_mut(),
                ],
            )
            .unwrap();
        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[1, 2, 16], &packed).view(),
                    absent(),
                    absent(),
                    absent(),
                    absent(),
                    Owned::i32(&[1], &[1]).view(),
                    Owned::i32(&[], &[2]).view(),
                ],
                &mut [
                    packed_out.view_mut(),
                    packed_k.view_mut(),
                    packed_v.view_mut(),
                ],
            )
            .unwrap();

        close(&unpacked_out.to_f32(), &want);
        close(&packed_out.to_f32(), &want);
        close(&packed_out.to_f32(), &unpacked_out.to_f32());
        close(&packed_k.to_f32(), &unpacked_k.to_f32());
        close(&packed_v.to_f32(), &unpacked_v.to_f32());
    }

    #[test]
    fn decode_appends_past_and_matches_reference() {
        let q = vec![1., 0., 1., 0., 0., 1., 0., 1.];
        let past_k = vec![1., 0., 0., 1., 10., 0., 0., 10.];
        let past_v = vec![1., 2., 3., 4., 10., 20., 30., 40.];
        let cur_k = vec![1., 1., 10., 10.];
        let cur_v = vec![5., 6., 50., 60.];
        let mut all_k = vec![0.0; 12];
        let mut all_v = vec![0.0; 12];
        for h in 0..2 {
            all_k[h * 6..h * 6 + 4].copy_from_slice(&past_k[h * 4..h * 4 + 4]);
            all_v[h * 6..h * 6 + 4].copy_from_slice(&past_v[h * 4..h * 4 + 4]);
            all_k[h * 6 + 4..h * 6 + 6].copy_from_slice(&cur_k[h * 2..h * 2 + 2]);
            all_v[h * 6 + 4..h * 6 + 6].copy_from_slice(&cur_v[h * 2..h * 2 + 2]);
        }
        let mut out = Owned::zeros_f32(&[1, 1, 8]);
        let mut pk = Owned::zeros_f32(&[1, 2, 3, 2]);
        let mut pv = Owned::zeros_f32(&[1, 2, 3, 2]);
        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[1, 1, 8], &q).view(),
                    Owned::f32(&[1, 1, 4], &cur_k).view(),
                    Owned::f32(&[1, 1, 4], &cur_v).view(),
                    Owned::f32(&[1, 2, 2, 2], &past_k).view(),
                    Owned::f32(&[1, 2, 2, 2], &past_v).view(),
                    Owned::i32(&[1], &[2]).view(),
                    Owned::i32(&[], &[3]).view(),
                ],
                &mut [out.view_mut(), pk.view_mut(), pv.view_mut()],
            )
            .unwrap();
        close(&pk.to_f32(), &all_k);
        close(&pv.to_f32(), &all_v);
        close(&out.to_f32(), &reference(&q, &all_k, &all_v, 1, 3, 2));
    }

    #[test]
    fn decode_widens_f16_past_cache_before_materializing_present_cache() {
        let q = vec![1., 0., 1., 0., 0., 1., 0., 1.];
        let past_k = vec![1., 0., 0., 1., 10., 0., 0., 10.];
        let past_v = vec![1., 2., 3., 4., 10., 20., 30., 40.];
        let cur_k = vec![1., 1., 10., 10.];
        let cur_v = vec![5., 6., 50., 60.];
        let expected_k = vec![1., 0., 0., 1., 1., 1., 10., 0., 0., 10., 10., 10.];
        let expected_v = vec![1., 2., 3., 4., 5., 6., 10., 20., 30., 40., 50., 60.];
        let mut out = Owned::zeros_f32(&[1, 1, 8]);
        let mut pk = Owned::zeros_f32(&[1, 2, 3, 2]);
        let mut pv = Owned::zeros_f32(&[1, 2, 3, 2]);
        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[1, 1, 8], &q).view(),
                    Owned::f32(&[1, 1, 4], &cur_k).view(),
                    Owned::f32(&[1, 1, 4], &cur_v).view(),
                    Owned::f16(&[1, 2, 2, 2], &past_k).view(),
                    Owned::f16(&[1, 2, 2, 2], &past_v).view(),
                    Owned::i32(&[1], &[2]).view(),
                    Owned::i32(&[], &[3]).view(),
                ],
                &mut [out.view_mut(), pk.view_mut(), pv.view_mut()],
            )
            .unwrap();
        close(&pk.to_f32(), &expected_k);
        close(&pv.to_f32(), &expected_v);
        close(
            &out.to_f32(),
            &reference(&q, &expected_k, &expected_v, 1, 3, 2),
        );
    }

    /// Locks the incremental `PastCache::widen_run` path (which widens each
    /// past-cache head-run directly into `present`) against a full-widen
    /// reference across a *multi-step* decode: every step chains the previous
    /// step's f16 `present` KV back in as `past`, exactly as the engine's
    /// owned-KV loop does. The kernel's f32 attention output at each step must
    /// equal `reference_with_geometry` computed over the fully accumulated
    /// (f16-rounded past + raw-f32 current) cache. This guards the optimization
    /// against silent drift and proves the resident/incremental widen equals
    /// the original whole-cache widen for a real decode sequence.
    #[test]
    fn decode_multistep_incremental_widen_matches_full_widen_reference() {
        const QUERY_HEAD_COUNT: usize = 4;
        const KEY_VALUE_HEAD_COUNT: usize = 2;
        const HEAD_WIDTH: usize = 2;
        const STEPS: usize = 12;

        // Accumulated f16-rounded past rows per kv-head, grown one row per step
        // by carrying forward the kernel's own f16 `present` (as the engine does).
        let mut past_key_rows: Vec<Vec<f32>> = vec![Vec::new(); KEY_VALUE_HEAD_COUNT];
        let mut past_value_rows: Vec<Vec<f32>> = vec![Vec::new(); KEY_VALUE_HEAD_COUNT];

        for step in 0..STEPS {
            let past_sequence_length = step;
            let total_sequence_length = past_sequence_length + 1;

            let query: Vec<f32> = (0..QUERY_HEAD_COUNT * HEAD_WIDTH)
                .map(|index| mixed_scale_value(index + step * 97, 0x1111))
                .collect();
            let current_key: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * HEAD_WIDTH)
                .map(|index| mixed_scale_value(index + step * 31, 0x2222))
                .collect();
            let current_value: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * HEAD_WIDTH)
                .map(|index| mixed_scale_value(index + step * 57, 0x3333))
                .collect();

            // BNSH f16 past tensors from the accumulated rows.
            let mut past_key_flat = vec![0.0f32; KEY_VALUE_HEAD_COUNT * past_sequence_length * HEAD_WIDTH];
            let mut past_value_flat =
                vec![0.0f32; KEY_VALUE_HEAD_COUNT * past_sequence_length * HEAD_WIDTH];
            for head in 0..KEY_VALUE_HEAD_COUNT {
                let run = past_sequence_length * HEAD_WIDTH;
                past_key_flat[head * run..(head + 1) * run].copy_from_slice(&past_key_rows[head]);
                past_value_flat[head * run..(head + 1) * run]
                    .copy_from_slice(&past_value_rows[head]);
            }

            // Full-widen reference cache: f16-rounded past rows ++ raw-f32 current.
            let mut full_key = vec![0.0f32; KEY_VALUE_HEAD_COUNT * total_sequence_length * HEAD_WIDTH];
            let mut full_value =
                vec![0.0f32; KEY_VALUE_HEAD_COUNT * total_sequence_length * HEAD_WIDTH];
            for head in 0..KEY_VALUE_HEAD_COUNT {
                let full_base = head * total_sequence_length * HEAD_WIDTH;
                let past_run = past_sequence_length * HEAD_WIDTH;
                full_key[full_base..full_base + past_run].copy_from_slice(&past_key_rows[head]);
                full_value[full_base..full_base + past_run].copy_from_slice(&past_value_rows[head]);
                for dimension in 0..HEAD_WIDTH {
                    full_key[full_base + past_run + dimension] = current_key[head * HEAD_WIDTH + dimension];
                    full_value[full_base + past_run + dimension] =
                        current_value[head * HEAD_WIDTH + dimension];
                }
            }
            let expected = kernel_exact_reference(
                &query,
                &full_key,
                &full_value,
                total_sequence_length,
                past_sequence_length,
                QUERY_HEAD_COUNT,
                KEY_VALUE_HEAD_COUNT,
                HEAD_WIDTH,
            );

            let mut output = Owned::zeros_f32(&[1, 1, QUERY_HEAD_COUNT * HEAD_WIDTH]);
            let mut present_key =
                Owned::zeros_f32(&[1, KEY_VALUE_HEAD_COUNT, total_sequence_length, HEAD_WIDTH]);
            let mut present_value =
                Owned::zeros_f32(&[1, KEY_VALUE_HEAD_COUNT, total_sequence_length, HEAD_WIDTH]);
            gqa_kernel(&[])
                .execute(
                    &[
                        Owned::f32(&[1, 1, QUERY_HEAD_COUNT * HEAD_WIDTH], &query).view(),
                        Owned::f32(&[1, 1, KEY_VALUE_HEAD_COUNT * HEAD_WIDTH], &current_key).view(),
                        Owned::f32(&[1, 1, KEY_VALUE_HEAD_COUNT * HEAD_WIDTH], &current_value).view(),
                        Owned::f16(
                            &[1, KEY_VALUE_HEAD_COUNT, past_sequence_length, HEAD_WIDTH],
                            &past_key_flat,
                        )
                        .view(),
                        Owned::f16(
                            &[1, KEY_VALUE_HEAD_COUNT, past_sequence_length, HEAD_WIDTH],
                            &past_value_flat,
                        )
                        .view(),
                        Owned::i32(&[1], &[past_sequence_length as i32]).view(),
                        Owned::i32(&[], &[total_sequence_length as i32]).view(),
                    ],
                    &mut [
                        output.view_mut(),
                        present_key.view_mut(),
                        present_value.view_mut(),
                    ],
                )
                .unwrap();

            close_relative(&output.to_f32(), &expected);

            // Chain: the f16 `present` becomes the next step's `past`. The kernel
            // wrote `present` as f32 here (test convenience), but the engine's KV
            // cache is f16, so round each carried-forward value to f16 exactly as
            // the next-step widen will re-read it.
            let present_key_f32 = present_key.to_f32();
            let present_value_f32 = present_value.to_f32();
            let to_f16 = |x: f32| half::f16::from_f32(x).to_f32();
            for head in 0..KEY_VALUE_HEAD_COUNT {
                let run = total_sequence_length * HEAD_WIDTH;
                past_key_rows[head] = present_key_f32[head * run..(head + 1) * run]
                    .iter()
                    .map(|&x| to_f16(x))
                    .collect();
                past_value_rows[head] = present_value_f32[head * run..(head + 1) * run]
                    .iter()
                    .map(|&x| to_f16(x))
                    .collect();
            }
        }
    }

    #[test]
    fn decode_batch_ragged_past_lengths_materialize_independently() {
        let q = vec![
            1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 1., -1., -1., 1., -1., -1.,
        ];
        let past_k = vec![
            1., 0., 91., 92., 93., 94., 0., 1., 95., 96., 97., 98., 2., 0., 3., 0., 4., 0., 5., 0.,
            6., 0., 7., 0.,
        ];
        let past_v = vec![
            1., 2., 71., 72., 73., 74., 3., 4., 75., 76., 77., 78., 10., 20., 30., 40., 50., 60.,
            70., 80., 90., 100., 110., 120.,
        ];
        let cur_k = vec![1., 1., 10., 10., 8., 0., 9., 0.];
        let cur_v = vec![5., 6., 50., 60., 130., 140., 150., 160.];
        let expected_k = vec![
            1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 10., 10., 0., 0., 0., 0., 2., 0., 3., 0., 4.,
            0., 8., 0., 5., 0., 6., 0., 7., 0., 9., 0.,
        ];
        let expected_v = vec![
            1., 2., 5., 6., 0., 0., 0., 0., 3., 4., 50., 60., 0., 0., 0., 0., 10., 20., 30., 40.,
            50., 60., 130., 140., 70., 80., 90., 100., 110., 120., 150., 160.,
        ];
        let mut out = Owned::zeros_f32(&[2, 1, 8]);
        let mut pk = Owned::zeros_f32(&[2, 2, 4, 2]);
        let mut pv = Owned::zeros_f32(&[2, 2, 4, 2]);
        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[2, 1, 8], &q).view(),
                    Owned::f32(&[2, 1, 4], &cur_k).view(),
                    Owned::f32(&[2, 1, 4], &cur_v).view(),
                    Owned::f32(&[2, 2, 3, 2], &past_k).view(),
                    Owned::f32(&[2, 2, 3, 2], &past_v).view(),
                    Owned::i32(&[2], &[1, 3]).view(),
                    Owned::i32(&[], &[4]).view(),
                ],
                &mut [out.view_mut(), pk.view_mut(), pv.view_mut()],
            )
            .unwrap();
        close(&pk.to_f32(), &expected_k);
        close(&pv.to_f32(), &expected_v);
        let mut want = reference(
            &q[..8],
            &[1., 0., 1., 1., 0., 1., 10., 10.],
            &[1., 2., 5., 6., 3., 4., 50., 60.],
            1,
            2,
            1,
        );
        want.extend(reference(
            &q[8..],
            &expected_k[16..],
            &expected_v[16..],
            1,
            4,
            3,
        ));
        close(&out.to_f32(), &want);
    }

    #[test]
    fn decode_preserves_fixed_cache_capacity_and_appends_at_logical_length() {
        let q = vec![1., 0., 1., 0., 0., 1., 0., 1.];
        let past_k = vec![
            1., 0., 0., 1., 91., 92., 93., 94., 95., 96., 10., 0., 0., 10., 81., 82., 83., 84.,
            85., 86.,
        ];
        let past_v = vec![
            1., 2., 3., 4., 71., 72., 73., 74., 75., 76., 10., 20., 30., 40., 61., 62., 63., 64.,
            65., 66.,
        ];
        let cur_k = vec![1., 1., 10., 10.];
        let cur_v = vec![5., 6., 50., 60.];
        let expected_k = vec![
            1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 10., 0., 0., 10., 10., 10., 0., 0., 0., 0.,
        ];
        let expected_v = vec![
            1., 2., 3., 4., 5., 6., 0., 0., 0., 0., 10., 20., 30., 40., 50., 60., 0., 0., 0., 0.,
        ];
        let mut out = Owned::zeros_f32(&[1, 1, 8]);
        let mut pk = Owned::zeros_f32(&[1, 2, 5, 2]);
        let mut pv = Owned::zeros_f32(&[1, 2, 5, 2]);
        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[1, 1, 8], &q).view(),
                    Owned::f32(&[1, 1, 4], &cur_k).view(),
                    Owned::f32(&[1, 1, 4], &cur_v).view(),
                    Owned::f32(&[1, 2, 5, 2], &past_k).view(),
                    Owned::f32(&[1, 2, 5, 2], &past_v).view(),
                    Owned::i32(&[1], &[2]).view(),
                    Owned::i32(&[], &[3]).view(),
                ],
                &mut [out.view_mut(), pk.view_mut(), pv.view_mut()],
            )
            .unwrap();
        assert_eq!(pk.shape, vec![1, 2, 5, 2]);
        assert_eq!(pv.shape, vec![1, 2, 5, 2]);
        close(&pk.to_f32(), &expected_k);
        close(&pv.to_f32(), &expected_v);
        close(
            &out.to_f32(),
            &reference(&q, &expected_k, &expected_v, 1, 5, 2),
        );
    }

    #[test]
    fn rotary_path_matches_rotated_reference() {
        let q = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let k = vec![1., 2., 3., 4.];
        let v = vec![1., 2., 3., 4.];
        let cos = vec![0.0];
        let sin = vec![1.0];
        let q_rot = vec![-2., 1., -4., 3., -6., 5., -8., 7.];
        let k_rot_bsh = vec![-2., 1., -4., 3.];
        let k_rot_bnsh = vec![-2., 1., -4., 3.];
        let mut out = Owned::zeros_f32(&[1, 1, 8]);
        gqa_kernel(&[("do_rotary", Attribute::Int(1))])
            .execute(
                &[
                    Owned::f32(&[1, 1, 8], &q).view(),
                    Owned::f32(&[1, 1, 4], &k).view(),
                    Owned::f32(&[1, 1, 4], &v).view(),
                    absent(),
                    absent(),
                    Owned::i32(&[1], &[0]).view(),
                    Owned::i32(&[], &[1]).view(),
                    Owned::f32(&[1, 1], &cos).view(),
                    Owned::f32(&[1, 1], &sin).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        let _ = k_rot_bsh;
        close(&out.to_f32(), &reference(&q_rot, &k_rot_bnsh, &v, 1, 1, 0));
    }

    #[test]
    fn rotary_explicit_position_ids_apply_to_query_and_key() {
        let q = vec![
            1., 2., 2., -1., -1., 3., 4., 2., 3., -2., 1., 4., -3., 1., 2., 5.,
        ];
        let k = vec![2., 1., -1., 3., 4., -2., 2., 5.];
        let v = vec![1., 2., 10., 20., 3., 4., 30., 40.];
        let angles = [0.0_f32, 0.2, 0.7, 1.1, 1.6];
        let cos: Vec<f32> = angles.iter().map(|angle| angle.cos()).collect();
        let sin: Vec<f32> = angles.iter().map(|angle| angle.sin()).collect();
        let positions = [2_usize, 4];
        let q_rot = reference_rope_bsh(&q, 2, 4, &positions, &cos, &sin);
        let k_rot_bsh = reference_rope_bsh(&k, 2, 2, &positions, &cos, &sin);
        let k_rot_bnsh = bsh_to_bnsh(&k_rot_bsh, 2, 2);
        let v_bnsh = bsh_to_bnsh(&v, 2, 2);
        let mut out = Owned::zeros_f32(&[1, 2, 8]);
        let mut present_k = Owned::zeros_f32(&[1, 2, 2, 2]);
        gqa_kernel(&[("do_rotary", Attribute::Int(1))])
            .execute(
                &[
                    Owned::f32(&[1, 2, 8], &q).view(),
                    Owned::f32(&[1, 2, 4], &k).view(),
                    Owned::f32(&[1, 2, 4], &v).view(),
                    absent(),
                    absent(),
                    Owned::i32(&[1], &[1]).view(),
                    Owned::i32(&[], &[2]).view(),
                    Owned::f32(&[5, 1], &cos).view(),
                    Owned::f32(&[5, 1], &sin).view(),
                    Owned::i64(&[1, 2], &[2, 4]).view(),
                ],
                &mut [out.view_mut(), present_k.view_mut()],
            )
            .unwrap();
        close(&present_k.to_f32(), &k_rot_bnsh);
        close(
            &out.to_f32(),
            &reference(&q_rot, &k_rot_bnsh, &v_bnsh, 2, 2, 0),
        );
    }

    #[test]
    fn local_window_masks_older_cache_tokens() {
        let q = [0.0; 8];
        let past_k = [0.0; 8];
        let past_v = [1., 1., 2., 2., 10., 10., 20., 20.];
        let cur_k = [0.0; 4];
        let cur_v = [9., 9., 90., 90.];
        let mut out = Owned::zeros_f32(&[1, 1, 8]);
        gqa_kernel(&[("local_window_size", Attribute::Int(1))])
            .execute(
                &[
                    Owned::f32(&[1, 1, 8], &q).view(),
                    Owned::f32(&[1, 1, 4], &cur_k).view(),
                    Owned::f32(&[1, 1, 4], &cur_v).view(),
                    Owned::f32(&[1, 2, 2, 2], &past_k).view(),
                    Owned::f32(&[1, 2, 2, 2], &past_v).view(),
                    Owned::i32(&[1], &[2]).view(),
                    Owned::i32(&[], &[3]).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        close(&out.to_f32(), &[9., 9., 9., 9., 90., 90., 90., 90.]);
    }

    #[test]
    fn softcap_matches_independent_score_transform() {
        let q = [
            2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0.,
        ];
        let k = [1., 0., 1., 0., 4., 0., 4., 0.];
        let v = [1., 0., 10., 0., 3., 0., 30., 0.];
        let mut out = Owned::zeros_f32(&[1, 2, 8]);
        gqa_kernel(&[("softcap", Attribute::Float(1.5))])
            .execute(
                &[
                    Owned::f32(&[1, 2, 8], &q).view(),
                    Owned::f32(&[1, 2, 4], &k).view(),
                    Owned::f32(&[1, 2, 4], &v).view(),
                    absent(),
                    absent(),
                    Owned::i32(&[1], &[1]).view(),
                    Owned::i32(&[], &[2]).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        let s0 = 1.5 * ((2.0 / 2.0_f32.sqrt()) / 1.5_f32).tanh();
        let s1 = 1.5 * ((8.0 / 2.0_f32.sqrt()) / 1.5_f32).tanh();
        let p1 = (s1 - s0).exp() / (1.0 + (s1 - s0).exp());
        let expected_second = 1.0 * (1.0 - p1) + 3.0 * p1;
        let expected = [
            1.,
            0.,
            1.,
            0.,
            10.,
            0.,
            10.,
            0.,
            expected_second,
            0.,
            expected_second,
            0.,
            expected_second * 10.0,
            0.,
            expected_second * 10.0,
            0.,
        ];
        close(&out.to_f32(), &expected);
    }

    #[test]
    fn explicit_zero_scale_matches_default_scale() {
        let q = [
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
        ];
        let k = [0., 0., 0., 0., 4., 0., 4., 0.];
        let v = [1., 0., 1., 0., 9., 0., 9., 0.];
        let run = |attrs: &[(&str, Attribute)]| {
            let mut out = Owned::zeros_f32(&[1, 2, 8]);
            gqa_kernel(attrs)
                .execute(
                    &[
                        Owned::f32(&[1, 2, 8], &q).view(),
                        Owned::f32(&[1, 2, 4], &k).view(),
                        Owned::f32(&[1, 2, 4], &v).view(),
                        absent(),
                        absent(),
                        Owned::i32(&[1], &[1]).view(),
                        Owned::i32(&[], &[2]).view(),
                    ],
                    &mut [out.view_mut()],
                )
                .unwrap();
            out.to_f32()
        };
        let default = run(&[]);
        let zero = run(&[("scale", Attribute::Float(0.0))]);
        close(&zero, &default);
        assert!(zero[8] > 8.0, "zero scale produced uniform attention");
    }

    // ── New tests covering the vectorized decode hot path ──────────────────

    /// Verifies realistic-width M=1 decode against a scalar full-attention
    /// implementation. The pseudo-random mixed-scale inputs are non-periodic
    /// over the fixture and produce cancellation in the 128-element dot products.
    ///
    /// The tolerance covers both dot-product reordering and hundreds of fused
    /// probability-weighted value accumulations. It validates the runtime
    /// dispatch path, including the scalar fallback on hosts without AVX2+FMA.
    #[test]
    fn gqa_decode_long_context_matches_reference() {
        const PAST_SEQUENCE_LENGTH: usize = 255;
        const TOTAL_SEQUENCE_LENGTH: usize = PAST_SEQUENCE_LENGTH + 1;
        const QUERY_HEAD_COUNT: usize = 4;
        const KEY_VALUE_HEAD_COUNT: usize = 2;
        const HEAD_WIDTH: usize = 128;

        let query: Vec<f32> = (0..QUERY_HEAD_COUNT * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0x1234))
            .collect();
        let current_key: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0x5678))
            .collect();
        let current_value: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0x9abc))
            .collect();
        let past_key: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * PAST_SEQUENCE_LENGTH * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0xdef0))
            .collect();
        let past_value: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * PAST_SEQUENCE_LENGTH * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0x2468))
            .collect();

        let mut full_key = vec![0.0f32; KEY_VALUE_HEAD_COUNT * TOTAL_SEQUENCE_LENGTH * HEAD_WIDTH];
        let mut full_value =
            vec![0.0f32; KEY_VALUE_HEAD_COUNT * TOTAL_SEQUENCE_LENGTH * HEAD_WIDTH];
        for head_index in 0..KEY_VALUE_HEAD_COUNT {
            let past_base = head_index * PAST_SEQUENCE_LENGTH * HEAD_WIDTH;
            let full_base = head_index * TOTAL_SEQUENCE_LENGTH * HEAD_WIDTH;
            full_key[full_base..full_base + PAST_SEQUENCE_LENGTH * HEAD_WIDTH].copy_from_slice(
                &past_key[past_base..past_base + PAST_SEQUENCE_LENGTH * HEAD_WIDTH],
            );
            full_value[full_base..full_base + PAST_SEQUENCE_LENGTH * HEAD_WIDTH].copy_from_slice(
                &past_value[past_base..past_base + PAST_SEQUENCE_LENGTH * HEAD_WIDTH],
            );
            for dimension_index in 0..HEAD_WIDTH {
                full_key[full_base + PAST_SEQUENCE_LENGTH * HEAD_WIDTH + dimension_index] =
                    current_key[head_index * HEAD_WIDTH + dimension_index];
                full_value[full_base + PAST_SEQUENCE_LENGTH * HEAD_WIDTH + dimension_index] =
                    current_value[head_index * HEAD_WIDTH + dimension_index];
            }
        }

        let expected = reference_with_geometry(
            &query,
            &full_key,
            &full_value,
            1,
            TOTAL_SEQUENCE_LENGTH,
            PAST_SEQUENCE_LENGTH,
            QUERY_HEAD_COUNT,
            KEY_VALUE_HEAD_COUNT,
            HEAD_WIDTH,
        );

        let mut output = Owned::zeros_f32(&[1, 1, QUERY_HEAD_COUNT * HEAD_WIDTH]);
        let mut present_key =
            Owned::zeros_f32(&[1, KEY_VALUE_HEAD_COUNT, TOTAL_SEQUENCE_LENGTH, HEAD_WIDTH]);
        let mut present_value =
            Owned::zeros_f32(&[1, KEY_VALUE_HEAD_COUNT, TOTAL_SEQUENCE_LENGTH, HEAD_WIDTH]);
        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[1, 1, QUERY_HEAD_COUNT * HEAD_WIDTH], &query).view(),
                    Owned::f32(&[1, 1, KEY_VALUE_HEAD_COUNT * HEAD_WIDTH], &current_key).view(),
                    Owned::f32(&[1, 1, KEY_VALUE_HEAD_COUNT * HEAD_WIDTH], &current_value).view(),
                    Owned::f32(
                        &[1, KEY_VALUE_HEAD_COUNT, PAST_SEQUENCE_LENGTH, HEAD_WIDTH],
                        &past_key,
                    )
                    .view(),
                    Owned::f32(
                        &[1, KEY_VALUE_HEAD_COUNT, PAST_SEQUENCE_LENGTH, HEAD_WIDTH],
                        &past_value,
                    )
                    .view(),
                    Owned::i32(&[1], &[PAST_SEQUENCE_LENGTH as i32]).view(),
                    Owned::i32(&[], &[TOTAL_SEQUENCE_LENGTH as i32]).view(),
                ],
                &mut [
                    output.view_mut(),
                    present_key.view_mut(),
                    present_value.view_mut(),
                ],
            )
            .unwrap();

        for (index, (actual, expected)) in output.to_f32().iter().zip(&expected).enumerate() {
            let tolerance = 2.0e-5 + 2.0e-5 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "attention output {index}: actual {actual}, expected {expected}, difference {}, tolerance {tolerance}",
                (actual - expected).abs()
            );
        }
    }

    /// Verifies SIMD dot products against a scalar sequential sum using the
    /// standard `2 γ_n Σ|a_i b_i|` comparison tolerance. The factor two accounts
    /// for comparing two rounded evaluation orders rather than either one to the
    /// exact real-number dot product.
    #[test]
    fn dot_f32_matches_scalar_reference_for_various_lengths() {
        let lengths = [1, 7, 8, 9, 15, 16, 17, 32, 64, 128, 133];
        if !has_simd_x86_for_test() {
            eprintln!("skipping AVX2+FMA dot-product regression: SIMD is unavailable");
            return;
        }
        for length in lengths {
            let left: Vec<f32> = (0..length)
                .map(|index| mixed_scale_value(index, 0x1357))
                .collect();
            let right: Vec<f32> = (0..length)
                .map(|index| mixed_scale_value(index, 0x9753))
                .collect();
            let scalar: f32 = left
                .iter()
                .zip(&right)
                .map(|(left_element, right_element)| left_element * right_element)
                .sum();
            let actual = dot_f32(&left, &right);
            let tolerance = dot_comparison_tolerance(&left, &right);
            assert!(
                (actual - scalar).abs() <= tolerance,
                "dot_f32 length={length}: actual {actual}, scalar {scalar}, difference {}, tolerance {}",
                (actual - scalar).abs(),
                tolerance
            );
        }
    }

    /// Verifies that `axpy_f32` accumulates `dst[d] += scalar * src[d]`
    /// correctly for various lengths relative to the scalar path.
    #[test]
    fn axpy_f32_matches_scalar_reference_for_various_lengths() {
        let lengths = [1, 7, 8, 9, 15, 16, 17, 32, 64, 128, 133];
        for n in lengths {
            let src: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) / 13.0).collect();
            let init: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) / 7.0).collect();
            let scalar_val = 0.37_f32;

            // Scalar reference.
            let mut want = init.clone();
            for (d, s) in want.iter_mut().zip(&src) {
                *d += scalar_val * s;
            }

            // axpy_f32 path.
            let mut got = init.clone();
            axpy_f32(&mut got, scalar_val, &src);

            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() < 1e-6,
                    "axpy_f32 n={n} i={i}: got {g}, want {w}"
                );
            }
        }
    }

    /// Mirrors the P·V decode loop by applying 257 probability-weighted value
    /// rows. This covers repeated AVX2 FMADD accumulation rather than only one
    /// helper call.
    #[test]
    fn axpy_f32_many_weighted_rows_matches_scalar_reference() {
        const KEY_COUNT: usize = 257;
        const HEAD_WIDTH: usize = 128;

        if !has_simd_x86_for_test() {
            eprintln!("skipping AVX2+FMA AXPY regression: SIMD is unavailable");
            return;
        }

        let unnormalized_probabilities: Vec<f32> = (0..KEY_COUNT)
            .map(|key_index| (mixed_scale_value(key_index, 0xabcd).abs() + 0.01).exp())
            .collect();
        let probability_sum: f32 = unnormalized_probabilities.iter().sum();
        let probabilities: Vec<f32> = unnormalized_probabilities
            .iter()
            .map(|probability| probability / probability_sum)
            .collect();
        let values: Vec<f32> = (0..KEY_COUNT * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0xcafe))
            .collect();

        let mut expected = vec![0.0_f32; HEAD_WIDTH];
        let mut actual = vec![0.0_f32; HEAD_WIDTH];
        for (key_index, probability) in probabilities.iter().copied().enumerate() {
            let value_row = &values[key_index * HEAD_WIDTH..(key_index + 1) * HEAD_WIDTH];
            for (destination, source) in expected.iter_mut().zip(value_row) {
                *destination += probability * source;
            }
            axpy_f32(&mut actual, probability, value_row);
        }

        for dimension_index in 0..HEAD_WIDTH {
            let absolute_term_sum: f32 = (0..KEY_COUNT)
                .map(|key_index| {
                    (probabilities[key_index] * values[key_index * HEAD_WIDTH + dimension_index])
                        .abs()
                })
                .sum();
            let unit_roundoff = 0.5 * f32::EPSILON;
            let gamma = KEY_COUNT as f32 * unit_roundoff / (1.0 - KEY_COUNT as f32 * unit_roundoff);
            let tolerance = 2.0 * gamma * absolute_term_sum + 2.0 * f32::MIN_POSITIVE;
            assert!(
                (actual[dimension_index] - expected[dimension_index]).abs() <= tolerance,
                "P·V dimension {dimension_index}: actual {}, expected {}, difference {}, tolerance {tolerance}",
                actual[dimension_index],
                expected[dimension_index],
                (actual[dimension_index] - expected[dimension_index]).abs()
            );
        }
    }
}
