//! Shared **scaled-dot-product-attention (SDPA) core** — the one place the
//! attention math lives, so the many attention ops in this crate
//! (`com.microsoft::MultiHeadAttention`, `ai.onnx::Attention`,
//! `GroupQueryAttention`, `com.microsoft::FusedAttention`, …) stop
//! copy-pasting the `QKᵀ → scale → [softcap] → +bias → +mask → softmax → ·V`
//! sequence and instead adapt onto this primitive.
//!
//! ## What lives here vs. in the adapter
//!
//! This core is deliberately **pure f32 math over dense `BNSH` buffers**. It
//! knows nothing about tensor layouts, packed QKV, bias projection, or KV
//! caches — those are *adapter* responsibilities, because they differ per op
//! and are cheap reshapes/concats. The adapter's job is to normalize its
//! op-specific inputs into the [`SdpaTensors`] contract (query
//! `[B, Nq, Sq, Dh]`, key `[B, Nkv, Tk, Dh]`, value `[B, Nkv, Tk, Dv]`, all
//! contiguous f32), then call [`sdpa_f32`]. This keeps the numerics in exactly
//! one place while letting each op keep its own I/O quirks.
//!
//! The pluggable variation the core itself expresses:
//!
//! * **GQA / MQA head sharing** — `num_kv_heads ≤ num_heads`; query head `n`
//!   reads kv head `n / (num_heads / num_kv_heads)`. `num_kv_heads == num_heads`
//!   is plain MHA.
//! * **Differing V head size** — `v_head_size` (`Dv`) is independent of the
//!   Q/K `head_size` (`Dh`).
//! * **Scale placement** — [`ScaleMode::PostDot`] multiplies the raw dot by
//!   `scale` (ORT's MHA/fused path, folded into the GEMM `alpha`);
//!   [`ScaleMode::SplitSqrt`] pre-scales each operand by `√scale` (ORT's
//!   `ai.onnx::Attention` overflow-safe path).
//! * **Softcap** — optional `softcap · tanh(score / softcap)` logit clamp
//!   (`ai.onnx::Attention`), applied right after the scale as ORT does.
//! * **Additive attention bias** — a per-`(b, head, i, j)` float addend
//!   ([`AttnBias`]); [`BroadcastBias`] covers the `(B|1, N|1, S, T)` broadcast
//!   the contrib ops use.
//! * **Additive key mask** — a per-`(b, i, j)` float addend ([`KeyMask`]),
//!   covering key-padding masks; it is head-independent, matching ORT.
//! * **Causal masking with a past-KV offset** — key `j` is masked for query `i`
//!   when `j > past_seq + i`, using a caller-chosen fill (`f32::MIN` for MHA).
//! * **Optional QK score capture** — the logits or probabilities
//!   (`[B, Nq, Sq, Tk]`) at a caller-chosen pipeline stage ([`QkCaptureStage`])
//!   for ops that emit `qk_matmul_output`.
//!
//! ## Numerical contract (why this is a *drop-in* factoring)
//!
//! The per-`(b, head, i)` inner sequence is byte-for-byte the loop the
//! standalone MHA kernel used to run:
//!
//! ```text
//! score = dot(Q_i, K_j)                 # plain f32 fma-free accumulation
//! score = scale · score                 # PostDot   (or operands pre-scaled)
//! score = softcap·tanh(score/softcap)   # only when softcap set
//! score += attn_bias(b, n, i, j)        # 0.0 when absent (identity add)
//! score += key_mask(b, i, j)            # 0.0 when absent (identity add)
//! score  = causal_fill  if j > past+i   # override, matching ORT's merged mask
//! probs  = softmax(score)               # subtract row max, then normalize
//! out_i += probs_j · V_j                # plain f32 accumulation
//! ```
//!
//! The addends are applied in this exact order (never pre-summed) so that a
//! migrated op reproduces its reference goldens *bit-for-bit*, not merely
//! within tolerance. `f16`/`bf16` widen at the adapter boundary (Q/K/V are
//! already f32 here).
//!
//! ## Scalar reference vs. MLAS-GEMM fast path
//!
//! [`sdpa_f32_scalar`] is the byte-exact reference above: a scalar triple loop
//! whose numerics the parity goldens pin. It is retained unchanged as the
//! oracle the tolerance tests cross-check against.
//!
//! [`sdpa_f32`] is the adapter-facing entry point. When the crate is built
//! `--features mlas` and no [`QkCapture`] is requested, it runs a **fast path**
//! that (a) computes `QKᵀ` and `P·V` as real MLAS SGEMMs (batched over
//! `batch·head`, GQA/MQA kv heads gathered by group), (b) applies
//! `scale → softcap → bias → mask → causal` per **row** on plain slices (same
//! order as the scalar loop), and (c) rayon-parallelizes across the
//! `(batch, head)` tiles on the crate's shared pool (no oversubscription — MLAS
//! itself tiles onto that same pool). GEMM reorders float accumulation, so the
//! fast path is **not** bit-identical to the scalar loop; it is gated by
//! tolerance against both the scalar reference and live ORT 1.26 (which also
//! uses MLAS, so the fast path often matches ORT *more* closely than the scalar
//! path). Any shape the fast path cannot serve — or a [`QkCapture`] request, or
//! a non-`mlas` build — transparently falls back to [`sdpa_f32_scalar`], so the
//! output is always correct.

/// Query/key/value operands for one SDPA call, as dense contiguous f32 buffers
/// in `BNSH` (`[batch, heads, seq, dim]`) order.
///
/// * `q`  — `[batch, num_heads, q_seq, head_size]`
/// * `k`  — `[batch, num_kv_heads, kv_seq, head_size]`
/// * `v`  — `[batch, num_kv_heads, kv_seq, v_head_size]`
pub struct SdpaTensors<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub batch: usize,
    /// Number of query heads (`Nq`).
    pub num_heads: usize,
    /// Number of key/value heads (`Nkv ≤ Nq`); `Nq` for plain MHA.
    pub num_kv_heads: usize,
    /// Query sequence length (`Sq`).
    pub q_seq: usize,
    /// Total key/value sequence length after any cache concat (`Tk`).
    pub kv_seq: usize,
    /// Q/K head dimension (`Dh`).
    pub head_size: usize,
    /// V head dimension (`Dv`); may differ from `head_size`.
    pub v_head_size: usize,
}

/// How the score `scale` is applied to the raw `Q·Kᵀ` dot product.
#[derive(Clone, Copy, Debug)]
pub enum ScaleMode {
    /// Multiply the completed dot product by `scale` (ORT folds this into the
    /// GEMM `alpha`; used by MHA and `FusedAttention`).
    PostDot(f32),
    /// Pre-scale each Q and K element by `√scale` before the dot, so extreme
    /// magnitudes can't overflow the accumulation (ORT's `ai.onnx::Attention`).
    SplitSqrt(f32),
}

/// Precision used to evaluate the exponential in the softmax epilogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftmaxExp {
    /// Evaluate `exp(score - max)` in f32 (the existing SDPA behavior).
    F32,
    /// Evaluate in f64 and round once to f32 (the GQA decode contract).
    F64Intermediate,
}

impl SoftmaxExp {
    #[inline]
    fn exp(self, value: f32) -> f32 {
        match self {
            Self::F32 => value.exp(),
            Self::F64Intermediate => (value as f64).exp() as f32,
        }
    }
}

/// Fixed SDPA parameters (everything that isn't the Q/K/V data or the
/// bias/mask hooks).
pub struct SdpaConfig {
    /// Score scaling strategy.
    pub scale: ScaleMode,
    /// Optional `softcap · tanh(score / softcap)` logit clamp; `None` disables.
    pub softcap: Option<f32>,
    /// Apply lower-triangular causal masking (with the `past_seq` offset).
    pub causal: bool,
    /// Length of any KV already in the cache, shifting the causal frontier:
    /// key `j` is visible to query `i` iff `j <= past_seq + i`.
    pub past_seq: usize,
    /// Additive fill written into causally-masked positions (`f32::MIN` in ORT).
    pub causal_fill: f32,
}

/// Per-`(batch, head, query, key)` additive attention bias.
///
/// Called once per score; return `0.0` to contribute nothing. Kept as a trait
/// (rather than an `Option<&[f32]>`) so ops with exotic bias broadcasts plug in
/// without the core knowing their layout.
///
/// The `Sync` bound lets the [`sdpa_f32`] fast path share a single `&dyn
/// AttnBias` across the rayon workers that own disjoint `(batch, head)` tiles;
/// every adapter hook here holds only shared `&[f32]`/scalars, so it is `Sync`.
pub trait AttnBias: Sync {
    fn at(&self, b: usize, head: usize, i: usize, j: usize) -> f32;

    /// `true` when [`AttnBias::at`] returns `0.0` for every index, letting the
    /// fast paths drop the per-element virtual call entirely.
    fn is_identity(&self) -> bool {
        false
    }
}

/// Per-`(batch, query, key)` additive key mask (head-independent, as in ORT's
/// key-padding masks). Return `0.0` to keep a key, a large negative fill to
/// mask it.
pub trait KeyMask: Sync {
    fn at(&self, b: usize, i: usize, j: usize) -> f32;

    /// `true` when [`KeyMask::at`] returns `0.0` for every index, letting the
    /// fast paths drop the per-element virtual call entirely.
    fn is_identity(&self) -> bool {
        false
    }
}

/// No-op attention bias (contributes `0.0` everywhere).
pub struct NoBias;
impl AttnBias for NoBias {
    #[inline]
    fn at(&self, _b: usize, _head: usize, _i: usize, _j: usize) -> f32 {
        0.0
    }

    #[inline]
    fn is_identity(&self) -> bool {
        true
    }
}

/// No-op key mask (keeps every key).
pub struct NoMask;
impl KeyMask for NoMask {
    #[inline]
    fn at(&self, _b: usize, _i: usize, _j: usize) -> f32 {
        0.0
    }

    #[inline]
    fn is_identity(&self) -> bool {
        true
    }
}

/// Additive attention bias with the contrib-op `(B|1, N|1, S, T)` broadcast:
/// leading batch and head dims may each be `1` (broadcast) or full.
pub struct BroadcastBias<'a> {
    data: &'a [f32],
    dims: [usize; 4],
}

impl<'a> BroadcastBias<'a> {
    /// `dims` is the bias tensor's `[B|1, N|1, S, T]` shape; `data` its
    /// row-major contents.
    pub fn new(data: &'a [f32], dims: [usize; 4]) -> Self {
        Self { data, dims }
    }
}

impl AttnBias for BroadcastBias<'_> {
    #[inline]
    fn at(&self, b: usize, head: usize, i: usize, j: usize) -> f32 {
        let b0 = if self.dims[0] == 1 { 0 } else { b };
        let n0 = if self.dims[1] == 1 { 0 } else { head };
        let off = (((b0 * self.dims[1] + n0) * self.dims[2] + i) * self.dims[3]) + j;
        self.data[off]
    }
}

/// Which point in the per-score pipeline a [`QkCapture`] records.
///
/// `ai.onnx::Attention`'s `qk_matmul_output_mode` selects one of these; MHA and
/// `FusedAttention` capture at [`PreSoftmax`](QkCaptureStage::PreSoftmax).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QkCaptureStage {
    /// Right after the score scale, before softcap (Attention mode `0`).
    PostScale,
    /// After softcap, before bias/mask (Attention mode `1`; identical to
    /// [`PostScale`](QkCaptureStage::PostScale) when softcap is disabled).
    PostSoftcap,
    /// After bias/mask/causal, before softmax (default; MHA/Fused
    /// `qk_matmul_output`, Attention mode `2`).
    PreSoftmax,
    /// After the softmax normalization — i.e. the probabilities (Attention
    /// mode `3`).
    PostSoftmax,
}

/// Optional QK score capture target for ops that emit `qk_matmul_output`.
///
/// Holds the logits (or, for [`QkCaptureStage::PostSoftmax`], the
/// probabilities) in `[batch, num_heads, q_seq, kv_seq]` order, recorded at the
/// pipeline point named by `stage`.
pub struct QkCapture<'a> {
    pub scores: &'a mut [f32],
    pub stage: QkCaptureStage,
}

#[cfg(all(
    test,
    any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")
))]
static SDPA_SIMD_TEST_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Test counter: incremented when the Accelerate (cblas_sgemm/AMX) SDPA fast
/// path fires on macOS/iOS.
#[cfg(all(
    test,
    any(target_os = "macos", target_os = "ios"),
    not(feature = "mlas")
))]
static SDPA_ACCELERATE_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Test counter: incremented when the inline NEON decode SDPA path fires
/// (q_seq=1, small per-head work, bypassing Accelerate cblas_sgemm overhead).
#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "ios"),
    not(feature = "mlas")
))]
static SDPA_NEON_DECODE_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Run scaled-dot-product attention over `t`, writing the context into `y`
/// (`[batch, num_heads, q_seq, v_head_size]`, `BNSH`).
///
/// This is the **adapter-facing entry point**. It dispatches to the
/// MLAS-GEMM + rayon fast path ([`sdpa_f32_fast`]) when the crate is built
/// `--features mlas`, no [`QkCapture`] is requested, and the shape is
/// non-empty; otherwise it runs the scalar reference ([`sdpa_f32_scalar`]).
/// Both honour the exact `scale → softcap → bias → mask → causal → softmax`
/// sequence documented at the module level; the fast path only reorders the two
/// matmul accumulations (via GEMM), so it agrees with the scalar path to tight
/// tolerance rather than bit-for-bit.
///
/// `bias` and `mask` are applied additively in that order (pass [`NoBias`] /
/// [`NoMask`] to skip). When `qk` is `Some`, the requested pipeline stage is
/// copied out; that path is always served by the scalar reference so the
/// captured logits stay bit-identical.
pub fn sdpa_f32(
    t: &SdpaTensors,
    cfg: &SdpaConfig,
    bias: &dyn AttnBias,
    mask: &dyn KeyMask,
    y: &mut [f32],
    qk: Option<QkCapture>,
) {
    #[cfg(feature = "mlas")]
    {
        // The fast path handles every masking/scale mode, but it does not emit
        // a QkCapture (that stays on the scalar reference so the captured
        // logits are bit-identical) and needs a non-empty problem.
        let non_empty = t.batch > 0
            && t.num_heads > 0
            && t.q_seq > 0
            && t.kv_seq > 0
            && t.head_size > 0
            && t.v_head_size > 0;
        if qk.is_none() && non_empty {
            sdpa_f32_fast(t, cfg, bias, mask, y);
            return;
        }
    }
    // Accelerate (cblas_sgemm) fast path for macOS/iOS: replaces the NEON
    // scalar dot/axpy loops with AMX-backed GEMMs for QK^T and probs·V,
    // parallelized across (batch, head) tiles via Rayon.
    //
    // For decode (q_seq=1) with small per-head work, bypass Accelerate and use
    // inline NEON instead. The cblas_sgemm framework call + AMX dispatch setup
    // costs ~2-3µs per invocation, and the Accelerate path makes 2 cblas calls
    // per head tile. When head_size × kv_seq is small, this fixed overhead
    // dominates the actual arithmetic. The threshold below (total per-head
    // element count ≤ 8192) is derived from the crossover point measured across
    // head_size={4,64,128} and kv_seq={32..256}: it depends only on the ratio
    // of cblas call overhead to NEON throughput, both of which scale with the
    // hardware's SIMD width and memory subsystem — properties that are constant
    // across Apple Silicon generations (all share 128-bit NEON, same cache line
    // size, and comparable per-core issue width). The element count threshold
    // is independent of core count, frequency, or AMX generation.
    #[cfg(all(any(target_os = "macos", target_os = "ios"), not(feature = "mlas")))]
    {
        let non_empty = t.batch > 0
            && t.num_heads > 0
            && t.q_seq > 0
            && t.kv_seq > 0
            && t.head_size > 0
            && t.v_head_size > 0;
        if qk.is_none() && non_empty {
            // Decode with small per-head work: inline NEON beats Accelerate.
            // Threshold: kv_seq × max(head_size, v_head_size) ≤ 8192.
            // At head_size=64, this covers kv_seq ≤ 128; at head_size=128,
            // kv_seq ≤ 64. Beyond this, Accelerate's AMX throughput wins.
            let max_dim = t.head_size.max(t.v_head_size);
            let per_head_elements = t.kv_seq.saturating_mul(max_dim);
            if t.q_seq == 1 && per_head_elements <= 8192 {
                #[cfg(all(test, target_arch = "aarch64"))]
                SDPA_NEON_DECODE_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                sdpa_f32_simd(t, cfg, bias, mask, y);
                return;
            }
            sdpa_f32_accelerate(t, cfg, bias, mask, y);
            return;
        }
    }
    // SIMD path: same semantics as the scalar reference, but `dot_f32` and
    // `axpy_f32` use 4×-unrolled NEON on aarch64 and AVX2+FMA on x86. A
    // `QkCapture` still goes to the scalar reference so captured logits stay
    // bit-identical to the oracle the goldens pin.
    #[cfg(target_arch = "aarch64")]
    if qk.is_none() {
        sdpa_f32_simd(t, cfg, bias, mask, y);
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if qk.is_none() && crate::backend::has_simd_x86() {
        sdpa_f32_simd(t, cfg, bias, mask, y);
        return;
    }
    sdpa_f32_scalar(t, cfg, bias, mask, y, qk);
}

/// The route a **production** (default, MLAS-free) build takes for this shape,
/// selected explicitly rather than through [`sdpa_f32`]'s `cfg` ladder.
///
/// [`sdpa_f32`] short-circuits to [`sdpa_f32_fast`] whenever the crate is built
/// `--features mlas`, so under that feature it cannot be used as the native
/// half of an A/B: both halves would be MLAS. This entry point names the native
/// route directly, so [`crate::backend_ab`] can hold the shipped route and the
/// reference route side by side in one binary.
pub fn sdpa_f32_native(
    t: &SdpaTensors,
    cfg: &SdpaConfig,
    bias: &dyn AttnBias,
    mask: &dyn KeyMask,
    y: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    sdpa_f32_simd(t, cfg, bias, mask, y);

    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::backend::has_simd_x86() {
            sdpa_f32_simd(t, cfg, bias, mask, y);
            return;
        }
        sdpa_f32_scalar(t, cfg, bias, mask, y, None);
    }
}

/// The vendored-MLAS reference route for this shape, selected explicitly.
///
/// Only exists `--features mlas`; see [`crate::backend_ab`] for why both halves
/// have to be reachable from one binary.
#[cfg(feature = "mlas")]
pub fn sdpa_f32_mlas(
    t: &SdpaTensors,
    cfg: &SdpaConfig,
    bias: &dyn AttnBias,
    mask: &dyn KeyMask,
    y: &mut [f32],
) {
    sdpa_f32_fast(t, cfg, bias, mask, y);
}

/// Run one decode query row against the caller-selected KV window `[lo, hi)`.
///
/// The caller retains ownership of GQA-specific causal/sliding-window policy
/// and passes the resulting bounds here. `k` and `v` contain one full KV head
/// with `kv_seq` rows; `q` and `output` are one query/output row.
pub fn sdpa_decode_row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv_seq: usize,
    lo: usize,
    hi: usize,
    scale: f32,
    softcap: Option<f32>,
    exp: SoftmaxExp,
    output: &mut [f32],
) {
    debug_assert!(lo <= hi && hi <= kv_seq);
    debug_assert_eq!(k.len(), kv_seq * q.len());
    debug_assert_eq!(v.len(), kv_seq * output.len());

    let head_size = q.len();
    let v_head_size = output.len();
    sdpa_decode_row_accessor(
        q,
        |ks| {
            let base = ks * head_size;
            &k[base..base + head_size]
        },
        |ks| {
            let base = ks * v_head_size;
            &v[base..base + v_head_size]
        },
        lo,
        hi,
        scale,
        softcap,
        exp,
        output,
    );
}

/// [`sdpa_decode_row`] over a KV window `[lo, hi)` where each key/value row is
/// supplied by an accessor closure instead of one contiguous per-head buffer.
///
/// This is the reuse point for **runtime-managed (paged) KV** attention: a paged
/// store can attend directly over its pages by returning each token's K/V row in
/// place, with no per-step concat into a fresh `present` buffer and no output
/// round-trip. `k_row(ks)` must return this KV head's key row for token `ks`
/// (length `q.len()`) and `v_row(ks)` its value row (length `output.len()`).
///
/// The scoring (`dot_f32`, `scale`, optional `softcap`), the f64-intermediate
/// softmax, and the `axpy_f32` value reduction are evaluated in exactly the same
/// operations and order as [`sdpa_decode_row`], so for identical row *values*
/// the output is **bit-for-bit identical** to the contiguous fresh-present path.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode_row_accessor<'a>(
    q: &[f32],
    k_row: impl Fn(usize) -> &'a [f32],
    v_row: impl Fn(usize) -> &'a [f32],
    lo: usize,
    hi: usize,
    scale: f32,
    softcap: Option<f32>,
    exp: SoftmaxExp,
    output: &mut [f32],
) {
    debug_assert!(lo <= hi);

    let mut scores = vec![0.0f32; hi - lo];
    for (i, ks) in (lo..hi).enumerate() {
        let mut score = dot_f32(q, k_row(ks));
        score *= scale;
        if let Some(softcap) = softcap {
            score = softcap * (score / softcap).tanh();
        }
        scores[i] = score;
    }

    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for score in &mut scores {
        *score = exp.exp(*score - max);
        sum += *score;
    }
    if sum > 0.0 {
        for score in &mut scores {
            *score /= sum;
        }
    }

    output.fill(0.0);
    for (i, ks) in (lo..hi).enumerate() {
        let probability = scores[i];
        if probability == 0.0 {
            continue;
        }
        axpy_f32(output, probability, v_row(ks));
    }
}

/// Run **every query head that shares one KV head** against that head's KV
/// window `[lo, hi)` in a single streaming pass — the GQA/MQA decode analogue of
/// [`sdpa_decode_row`].
///
/// # Why this exists
///
/// `sdpa_decode_row` scores one query row against the whole KV window. Calling
/// it once per query head means the *same* K and V bytes are streamed
/// `group = num_heads / kv_num_heads` times per decode step. At M=1 decode the
/// window is a GEMV — arithmetic intensity is ~2 flops per loaded float — so
/// that repeat traffic is the cost: a 7-way group (Qwen2.5-0.5B: 14 query heads
/// over 2 KV heads) reads the KV cache seven times per layer per token.
///
/// This routine keeps `k_row(ks)` / `v_row(ks)` in registers/L1 for the whole
/// group, so the KV cache is streamed **once** per group instead of once per
/// query head. Arithmetic is unchanged; only the loop nesting is inverted.
///
/// # Bit-identity contract
///
/// For query head `g` the operations are, in order, exactly those
/// [`sdpa_decode_row`] performs for that head:
///
/// * the score for key `ks` is `dot_f32(q_g, k_row(ks)) * scale` with the same
///   optional softcap, evaluated with the same [`dot_f32`] accumulation order;
/// * the max / `exp` / sum / normalize sequence runs over that head's own score
///   slice, in ascending `ks` order;
/// * the value reduction accumulates `axpy_f32(out_g, p, v_row(ks))` in
///   ascending `ks` order, skipping `p == 0.0` exactly as the row path does.
///
/// Nothing is summed across heads, so inverting the loop nesting cannot change
/// any rounding. The output is **bit-for-bit identical** to `group` successive
/// [`sdpa_decode_row`] calls; `group_decode_matches_row_decode_bitwise` pins it.
///
/// # Layout
///
/// `q` holds the group's query rows back to back (`group * head_size` floats,
/// query head `g` at `g * head_size`) and `out` the group's output rows
/// (`group * v_head_size` floats). That is exactly the `BHSD` layout of a
/// `q_seq == 1` decode step, so the caller passes sub-slices with no copy.
///
/// `scores` is caller-owned scratch, resized to `group * (hi - lo)`; passing the
/// same buffer across calls keeps the per-call allocation out of the hot loop.
///
/// # Blocking
///
/// Both passes walk the KV window in tiles of [`group_kv_tile`] keys. Within a
/// tile the group loop is *outside* the key loop, so each query head writes (and
/// later reads) a contiguous run of scores rather than striding by `window`, and
/// the tile's K/V rows stay L1-resident across the whole group. Tiling only
/// reorders work between independent query heads, so the bit-identity contract
/// above is unaffected.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode_group(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv_seq: usize,
    group: usize,
    head_size: usize,
    v_head_size: usize,
    lo: usize,
    hi: usize,
    scale: f32,
    softcap: Option<f32>,
    exp: SoftmaxExp,
    out: &mut [f32],
    scores: &mut Vec<f32>,
) {
    debug_assert!(group >= 1);
    debug_assert!(lo <= hi && hi <= kv_seq);
    debug_assert_eq!(q.len(), group * head_size);
    debug_assert_eq!(out.len(), group * v_head_size);
    debug_assert_eq!(k.len(), kv_seq * head_size);
    debug_assert_eq!(v.len(), kv_seq * v_head_size);

    let window = hi - lo;
    scores.clear();
    scores.resize(group * window, 0.0);

    // QK^T: stream each key tile once, score it against all `group` query rows
    // while the tile is still in L1.
    let k_tile = group_kv_tile(head_size);
    for tile in (0..window).step_by(k_tile) {
        let tile_end = (tile + k_tile).min(window);
        for g in 0..group {
            let q_row = &q[g * head_size..(g + 1) * head_size];
            let dst = &mut scores[g * window + tile..g * window + tile_end];
            for (slot, ks) in (lo + tile..lo + tile_end).enumerate() {
                let k_base = ks * head_size;
                let mut score = dot_f32(q_row, &k[k_base..k_base + head_size]);
                score *= scale;
                if let Some(softcap) = softcap {
                    score = softcap * (score / softcap).tanh();
                }
                dst[slot] = score;
            }
        }
    }

    // Softmax per query head, over that head's own contiguous score slice.
    for g in 0..group {
        let row = &mut scores[g * window..(g + 1) * window];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for score in row.iter_mut() {
            *score = exp.exp(*score - max);
            sum += *score;
        }
        if sum > 0.0 {
            for score in row.iter_mut() {
                *score /= sum;
            }
        }
    }

    // P·V: stream each value tile once, accumulating into all `group` outputs
    // while the tile is still in L1.
    out.fill(0.0);
    let v_tile = group_kv_tile(v_head_size);
    for tile in (0..window).step_by(v_tile) {
        let tile_end = (tile + v_tile).min(window);
        for g in 0..group {
            let probabilities = &scores[g * window + tile..g * window + tile_end];
            let output = &mut out[g * v_head_size..(g + 1) * v_head_size];
            for (slot, ks) in (lo + tile..lo + tile_end).enumerate() {
                let probability = probabilities[slot];
                if probability == 0.0 {
                    continue;
                }
                let v_base = ks * v_head_size;
                axpy_f32(output, probability, &v[v_base..v_base + v_head_size]);
            }
        }
    }
}

/// Keys per KV tile in [`sdpa_decode_group`], sized so one tile of K (or V) rows
/// is about 16 KiB and therefore stays in L1D while the whole query group reads
/// it. Head sizes are small and few in practice (64/80/96/128/256), so this is
/// a cheap division rather than a tuned table.
#[inline]
fn group_kv_tile(head_size: usize) -> usize {
    const TILE_BYTES: usize = 16 * 1024;
    (TILE_BYTES / (head_size.max(1) * size_of::<f32>())).max(1)
}

/// One chunk's contribution to a flash-decoding (split-KV) softmax reduction.
///
/// `max` is the running maximum score over the chunk's KV sub-window and `sum`
/// is the unnormalized softmax denominator `Σ exp(score - max)` for that chunk;
/// both are accumulated in f64. An empty or fully-masked chunk reports
/// `max = f64::NEG_INFINITY` and `sum = 0.0`. The matching unnormalized
/// weighted-value accumulator is written out-of-band by [`sdpa_decode_partial`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodePartial {
    /// Running maximum score over the chunk (`f64::NEG_INFINITY` when empty).
    pub max: f64,
    /// Chunk-local softmax denominator `Σ exp(score - max)` in f64.
    pub sum: f64,
}

/// Partial flash-decoding reduction over a single KV sub-window `[lo, hi)`.
///
/// Mirrors [`sdpa_decode_row`]'s scoring exactly (same [`dot_f32`], `scale`, and
/// optional `softcap`) but stops **before** the final softmax normalization: it
/// returns this chunk's [`DecodePartial`] (running max and denominator) and
/// writes the unnormalized weighted-value accumulator
/// `o = Σ exp(score - max) · v` into `partial_output` (length = `v` head size).
///
/// The exponential, the denominator, and the value accumulator are all evaluated
/// in f64 so the two-level [`combine_decode_partials`] reduction stays as close
/// to the sequential [`SoftmaxExp::F64Intermediate`] reference as the split
/// reordering allows. Splitting reorders the additions and introduces the online
/// rescale, so the combined result is *not* bit-identical to [`sdpa_decode_row`]
/// — it is held to a tight max-abs-error bar instead (see the kernel tests).
///
/// An empty or fully-masked window (`hi <= lo`) yields
/// `DecodePartial { max: f64::NEG_INFINITY, sum: 0.0 }` and a zeroed
/// `partial_output`; [`combine_decode_partials`] skips such chunks.
pub fn sdpa_decode_partial(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv_seq: usize,
    lo: usize,
    hi: usize,
    scale: f32,
    softcap: Option<f32>,
    partial_output: &mut [f64],
) -> DecodePartial {
    debug_assert!(lo <= hi && hi <= kv_seq);
    debug_assert_eq!(k.len(), kv_seq * q.len());
    debug_assert_eq!(v.len(), kv_seq * partial_output.len());

    partial_output.fill(0.0);
    if hi <= lo {
        return DecodePartial {
            max: f64::NEG_INFINITY,
            sum: 0.0,
        };
    }

    let mut scores = vec![0.0f32; hi - lo];
    for (i, ks) in (lo..hi).enumerate() {
        let k_base = ks * q.len();
        let mut score = dot_f32(q, &k[k_base..k_base + q.len()]);
        score *= scale;
        if let Some(softcap) = softcap {
            score = softcap * (score / softcap).tanh();
        }
        scores[i] = score;
    }

    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let max_f64 = max as f64;
    let mut sum = 0.0f64;
    for (i, ks) in (lo..hi).enumerate() {
        // Match the reference's f64-intermediate exponential, but keep the
        // weight, denominator, and value accumulation in f64 so the per-chunk
        // partials carry full precision into the online-rescale combine.
        let weight = ((scores[i] as f64) - max_f64).exp();
        sum += weight;
        let v_base = ks * partial_output.len();
        let v_row = &v[v_base..v_base + partial_output.len()];
        for (o, &value) in partial_output.iter_mut().zip(v_row) {
            *o += weight * value as f64;
        }
    }
    DecodePartial { max: max_f64, sum }
}

/// Combine per-chunk [`sdpa_decode_partial`] results into one normalized decode
/// output row using the flash-decoding online-rescale reduction.
///
/// Given chunks with local max `m_j`, denominator `l_j`, and unnormalized value
/// accumulator `o_j`, the global softmax is recovered (in exact arithmetic) as
///
/// ```text
/// M = max_j m_j
/// L = Σ_j exp(m_j - M) · l_j
/// O = Σ_j exp(m_j - M) · o_j
/// output = O / L
/// ```
///
/// The rescale factor `exp(m_j - M) ∈ (0, 1]` re-bases every chunk onto the
/// global maximum before summing — that invariant is what lets the KV windows be
/// reduced independently. All arithmetic is f64; the result rounds to f32 once at
/// the end. `partial_outputs` is chunk-major: chunk `j`'s accumulator occupies
/// `[j * v_head_size, (j + 1) * v_head_size)`.
pub fn combine_decode_partials(
    partials: &[DecodePartial],
    partial_outputs: &[f64],
    v_head_size: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(output.len(), v_head_size);
    debug_assert_eq!(partial_outputs.len(), partials.len() * v_head_size);

    let global_max = partials
        .iter()
        .map(|partial| partial.max)
        .fold(f64::NEG_INFINITY, f64::max);
    if global_max == f64::NEG_INFINITY {
        output.fill(0.0);
        return;
    }

    let mut denominator = 0.0f64;
    let mut accumulator = vec![0.0f64; v_head_size];
    for (chunk, partial) in partials.iter().enumerate() {
        if partial.max == f64::NEG_INFINITY {
            continue;
        }
        let rescale = (partial.max - global_max).exp();
        denominator += rescale * partial.sum;
        let base = chunk * v_head_size;
        let chunk_output = &partial_outputs[base..base + v_head_size];
        for (acc, &value) in accumulator.iter_mut().zip(chunk_output) {
            *acc += rescale * value;
        }
    }

    if denominator > 0.0 {
        let inverse = 1.0 / denominator;
        for (out, &acc) in output.iter_mut().zip(&accumulator) {
            *out = (acc * inverse) as f32;
        }
    } else {
        output.fill(0.0);
    }
}

#[inline]
fn softmax_in_place(scores: &mut [f32], exp: SoftmaxExp) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        scores.fill(0.0);
        return;
    }
    let mut sum = 0.0f32;
    for score in scores.iter_mut() {
        let e = exp.exp(*score - max);
        *score = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for score in scores.iter_mut() {
        *score *= inv;
    }
}

/// Dot product using the decode path's AVX2+FMA accumulation order when
/// available, NEON on aarch64, with a scalar fallback on other targets.
#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if crate::backend::has_simd_x86() {
        // SAFETY: `has_simd_x86()` confirms AVX2 + FMA at runtime.
        return unsafe { dot_avx2_fma(a, b) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        return dot_neon(a, b);
    }
    #[allow(unreachable_code)]
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// AXPY using the decode path's AVX2+FMA accumulation order when available,
/// NEON on aarch64, with a scalar fallback on other targets.
#[inline(always)]
fn axpy_f32(dst: &mut [f32], scalar: f32, src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if crate::backend::has_simd_x86() {
        // SAFETY: `has_simd_x86()` confirms AVX2 + FMA at runtime.
        unsafe { axpy_avx2_fma(dst, scalar, src) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        axpy_neon(dst, scalar, src);
        return;
    }
    #[allow(unreachable_code)]
    for (d, s) in dst.iter_mut().zip(src) {
        *d += scalar * s;
    }
}

/// NEON 4×-unrolled dot product for aarch64 (ARMv8 baseline).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let mut acc0 = unsafe { vdupq_n_f32(0.0) };
    let mut acc1 = unsafe { vdupq_n_f32(0.0) };
    let mut acc2 = unsafe { vdupq_n_f32(0.0) };
    let mut acc3 = unsafe { vdupq_n_f32(0.0) };
    let mut j = 0;
    while j + 16 <= n {
        unsafe {
            acc0 = vfmaq_f32(
                acc0,
                vld1q_f32(a.as_ptr().add(j)),
                vld1q_f32(b.as_ptr().add(j)),
            );
            acc1 = vfmaq_f32(
                acc1,
                vld1q_f32(a.as_ptr().add(j + 4)),
                vld1q_f32(b.as_ptr().add(j + 4)),
            );
            acc2 = vfmaq_f32(
                acc2,
                vld1q_f32(a.as_ptr().add(j + 8)),
                vld1q_f32(b.as_ptr().add(j + 8)),
            );
            acc3 = vfmaq_f32(
                acc3,
                vld1q_f32(a.as_ptr().add(j + 12)),
                vld1q_f32(b.as_ptr().add(j + 12)),
            );
        }
        j += 16;
    }
    while j + 4 <= n {
        unsafe {
            acc0 = vfmaq_f32(
                acc0,
                vld1q_f32(a.as_ptr().add(j)),
                vld1q_f32(b.as_ptr().add(j)),
            );
        }
        j += 4;
    }
    let mut sum = unsafe { vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3))) };
    while j < n {
        sum += a[j] * b[j];
        j += 1;
    }
    sum
}

/// NEON 4×-unrolled AXPY for aarch64: dst[i] += scalar * src[i].
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn axpy_neon(dst: &mut [f32], scalar: f32, src: &[f32]) {
    use std::arch::aarch64::*;
    let n = dst.len();
    let vs = unsafe { vdupq_n_f32(scalar) };
    let mut j = 0;
    while j + 16 <= n {
        unsafe {
            let d0 = vld1q_f32(dst.as_ptr().add(j));
            let d1 = vld1q_f32(dst.as_ptr().add(j + 4));
            let d2 = vld1q_f32(dst.as_ptr().add(j + 8));
            let d3 = vld1q_f32(dst.as_ptr().add(j + 12));
            vst1q_f32(
                dst.as_mut_ptr().add(j),
                vfmaq_f32(d0, vs, vld1q_f32(src.as_ptr().add(j))),
            );
            vst1q_f32(
                dst.as_mut_ptr().add(j + 4),
                vfmaq_f32(d1, vs, vld1q_f32(src.as_ptr().add(j + 4))),
            );
            vst1q_f32(
                dst.as_mut_ptr().add(j + 8),
                vfmaq_f32(d2, vs, vld1q_f32(src.as_ptr().add(j + 8))),
            );
            vst1q_f32(
                dst.as_mut_ptr().add(j + 12),
                vfmaq_f32(d3, vs, vld1q_f32(src.as_ptr().add(j + 12))),
            );
        }
        j += 16;
    }
    while j + 4 <= n {
        unsafe {
            let d = vld1q_f32(dst.as_ptr().add(j));
            vst1q_f32(
                dst.as_mut_ptr().add(j),
                vfmaq_f32(d, vs, vld1q_f32(src.as_ptr().add(j))),
            );
        }
        j += 4;
    }
    while j < n {
        dst[j] += scalar * src[j];
        j += 1;
    }
}

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
        let chunks16 = n / 16;
        for i in 0..chunks16 {
            let av0 = _mm256_loadu_ps(a_ptr.add(i * 16));
            let bv0 = _mm256_loadu_ps(b_ptr.add(i * 16));
            acc0 = _mm256_fmadd_ps(av0, bv0, acc0);
            let av1 = _mm256_loadu_ps(a_ptr.add(i * 16 + 8));
            let bv1 = _mm256_loadu_ps(b_ptr.add(i * 16 + 8));
            acc1 = _mm256_fmadd_ps(av1, bv1, acc1);
        }
        let mut tail = chunks16 * 16;
        if tail + 8 <= n {
            let av = _mm256_loadu_ps(a_ptr.add(tail));
            let bv = _mm256_loadu_ps(b_ptr.add(tail));
            acc0 = _mm256_fmadd_ps(av, bv, acc0);
            tail += 8;
        }
        let acc = _mm256_add_ps(acc0, acc1);
        let lo = _mm256_extractf128_ps(acc, 0);
        let hi = _mm256_extractf128_ps(acc, 1);
        let v4 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(v4);
        let v2 = _mm_add_ps(v4, shuf);
        let shuf2 = _mm_movehl_ps(shuf, v2);
        let v1 = _mm_add_ss(v2, shuf2);
        let mut result = _mm_cvtss_f32(v1);
        for i in tail..n {
            result += *a_ptr.add(i) * *b_ptr.add(i);
        }
        result
    }
}

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
        while i < n {
            *dst_ptr.add(i) += scalar * *src_ptr.add(i);
            i += 1;
        }
    }
}

/// SIMD-vectorized SDPA — same semantics as the scalar reference, but the
/// inner loops go through [`dot_f32`] and [`axpy_f32`], which dispatch to
/// unrolled NEON on aarch64 and to AVX2+FMA on x86.
///
/// Note this is not merely the scalar reference with wider lanes: the P·V
/// accumulation here walks V row-major and touches each row once, whereas the
/// scalar oracle walks it column-major and re-reads V once per output column.
/// That locality difference is a large part of why this is faster, alongside
/// the vectorization itself.
///
/// The body is arch-neutral: it was written for aarch64 but contains no
/// intrinsics, so the only thing that ever kept it off x86 was its `cfg`.
///
/// Handles all features (GQA, causal, softcap, bias, mask).
#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
fn sdpa_f32_simd(
    t: &SdpaTensors,
    cfg: &SdpaConfig,
    bias: &dyn AttnBias,
    mask: &dyn KeyMask,
    y: &mut [f32],
) {
    #[cfg(test)]
    SDPA_SIMD_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let SdpaTensors {
        q,
        k,
        v,
        batch,
        num_heads,
        num_kv_heads,
        q_seq,
        kv_seq,
        head_size,
        v_head_size,
    } = *t;

    debug_assert_eq!(q.len(), batch * num_heads * q_seq * head_size);
    debug_assert_eq!(k.len(), batch * num_kv_heads * kv_seq * head_size);
    debug_assert_eq!(v.len(), batch * num_kv_heads * kv_seq * v_head_size);
    debug_assert_eq!(y.len(), batch * num_heads * q_seq * v_head_size);
    debug_assert!(num_kv_heads > 0 && num_heads.is_multiple_of(num_kv_heads));

    let heads_per_kv = num_heads / num_kv_heads;

    let (post_scale, operand_scale) = match cfg.scale {
        ScaleMode::PostDot(s) => (s, 1.0f32),
        ScaleMode::SplitSqrt(s) => (1.0f32, s.sqrt()),
    };
    let combined_scale = post_scale * operand_scale * operand_scale;

    let mut scores = vec![0.0f32; kv_seq];
    for b in 0..batch {
        for n in 0..num_heads {
            let kv_n = n / heads_per_kv;
            for i in 0..q_seq {
                let q_base = ((b * num_heads + n) * q_seq + i) * head_size;
                let q_slice = &q[q_base..q_base + head_size];

                // scores[j] = combined_scale * dot(Q, K_j) [+softcap] + bias + mask
                for (j, sc) in scores.iter_mut().enumerate() {
                    let k_base = ((b * num_kv_heads + kv_n) * kv_seq + j) * head_size;
                    let mut s = dot_f32(q_slice, &k[k_base..k_base + head_size]) * combined_scale;
                    if let Some(softcap) = cfg.softcap {
                        s = softcap * (s / softcap).tanh();
                    }
                    s += bias.at(b, n, i, j);
                    s += mask.at(b, i, j);
                    if cfg.causal && (j as i64) > cfg.past_seq as i64 + i as i64 {
                        s = cfg.causal_fill;
                    }
                    *sc = s;
                }

                softmax_in_place(&mut scores, SoftmaxExp::F32);

                // context: Y[i] = sum_j(probs[j] * V[j, :]) via NEON AXPY
                let y_base = ((b * num_heads + n) * q_seq + i) * v_head_size;
                let y_slice = &mut y[y_base..y_base + v_head_size];
                y_slice.fill(0.0);
                for (j, &probability) in scores.iter().enumerate() {
                    if probability == 0.0 {
                        continue;
                    }
                    let v_base = ((b * num_kv_heads + kv_n) * kv_seq + j) * v_head_size;
                    axpy_f32(y_slice, probability, &v[v_base..v_base + v_head_size]);
                }
            }
        }
    }
}

/// Byte-exact scalar SDPA reference — the oracle the parity goldens pin.
///
/// See the module docs for the exact numerical sequence; it is a bit-for-bit
/// factoring of the standalone MHA loop and is retained unchanged so the
/// tolerance tests (and the fast path) have a fixed reference to check against.
pub fn sdpa_f32_scalar(
    t: &SdpaTensors,
    cfg: &SdpaConfig,
    bias: &dyn AttnBias,
    mask: &dyn KeyMask,
    y: &mut [f32],
    mut qk: Option<QkCapture>,
) {
    let SdpaTensors {
        q,
        k,
        v,
        batch,
        num_heads,
        num_kv_heads,
        q_seq,
        kv_seq,
        head_size,
        v_head_size,
    } = *t;

    debug_assert_eq!(q.len(), batch * num_heads * q_seq * head_size);
    debug_assert_eq!(k.len(), batch * num_kv_heads * kv_seq * head_size);
    debug_assert_eq!(v.len(), batch * num_kv_heads * kv_seq * v_head_size);
    debug_assert_eq!(y.len(), batch * num_heads * q_seq * v_head_size);
    debug_assert!(num_kv_heads > 0 && num_heads.is_multiple_of(num_kv_heads));

    // Query heads per kv head (GQA/MQA sharing factor; 1 for plain MHA).
    let heads_per_kv = num_heads / num_kv_heads;

    // Score-scale placement.
    let (post_scale, operand_scale) = match cfg.scale {
        ScaleMode::PostDot(s) => (s, 1.0f32),
        ScaleMode::SplitSqrt(s) => (1.0f32, s.sqrt()),
    };

    let mut scores = vec![0.0f32; kv_seq];
    for b in 0..batch {
        for n in 0..num_heads {
            let kv_n = n / heads_per_kv;
            for i in 0..q_seq {
                let q_base = ((b * num_heads + n) * q_seq + i) * head_size;
                let cap_base = ((b * num_heads + n) * q_seq + i) * kv_seq;
                // scores[j] = scale·(Q·Kᵀ) [+softcap] + bias + mask [→ causal].
                for (j, sc) in scores.iter_mut().enumerate() {
                    let k_base = ((b * num_kv_heads + kv_n) * kv_seq + j) * head_size;
                    let mut acc = 0.0f32;
                    for p in 0..head_size {
                        acc += (q[q_base + p] * operand_scale) * (k[k_base + p] * operand_scale);
                    }
                    let mut s = acc * post_scale;
                    if let Some(cap) = qk.as_mut()
                        && cap.stage == QkCaptureStage::PostScale
                    {
                        cap.scores[cap_base + j] = s;
                    }
                    if let Some(softcap) = cfg.softcap {
                        s = softcap * (s / softcap).tanh();
                    }
                    if let Some(cap) = qk.as_mut()
                        && cap.stage == QkCaptureStage::PostSoftcap
                    {
                        cap.scores[cap_base + j] = s;
                    }
                    s += bias.at(b, n, i, j);
                    s += mask.at(b, i, j);
                    if cfg.causal && (j as i64) > cfg.past_seq as i64 + i as i64 {
                        s = cfg.causal_fill;
                    }
                    *sc = s;
                }

                if let Some(cap) = qk.as_mut()
                    && cap.stage == QkCaptureStage::PreSoftmax
                {
                    cap.scores[cap_base..cap_base + kv_seq].copy_from_slice(&scores);
                }

                // Numerically-stable softmax (subtract row max, matching ORT's
                // MlasComputeSoftmax and this crate's softmax kernel). A fully
                // masked row (every score `-inf`) yields a zero row rather than
                // NaN — matching ORT's guarded softmax. Fills that stay finite
                // (e.g. MHA's `f32::MIN`) never trigger this branch, so MHA's
                // numerics are unchanged.
                softmax_in_place(&mut scores, SoftmaxExp::F32);

                if let Some(cap) = qk.as_mut()
                    && cap.stage == QkCaptureStage::PostSoftmax
                {
                    cap.scores[cap_base..cap_base + kv_seq].copy_from_slice(&scores);
                }

                // context = probs · V.
                let y_base = ((b * num_heads + n) * q_seq + i) * v_head_size;
                for c in 0..v_head_size {
                    let mut acc = 0.0f32;
                    for (j, &p) in scores.iter().enumerate() {
                        let v_idx = ((b * num_kv_heads + kv_n) * kv_seq + j) * v_head_size + c;
                        acc += p * v[v_idx];
                    }
                    y[y_base + c] = acc;
                }
            }
        }
    }
}

/// MLAS-GEMM + rayon fast path behind [`sdpa_f32`].
///
/// Per `(batch, head)` tile it runs two SGEMMs — `logits = scale · Q·Kᵀ` and
/// `context = probs · V` — with the `softcap → bias → mask → causal → softmax`
/// epilogue applied per row on plain slices, in the exact order the scalar
/// reference uses. GQA/MQA share kv heads by group (`kv = head / (Nq/Nkv)`).
/// Tiles are fanned across the crate's shared rayon pool via
/// `par_chunks_mut`; MLAS tiles its own GEMM work onto that same pool, so there
/// is no oversubscription.
#[cfg(feature = "mlas")]
fn sdpa_f32_fast(
    t: &SdpaTensors,
    cfg: &SdpaConfig,
    bias: &dyn AttnBias,
    mask: &dyn KeyMask,
    y: &mut [f32],
) {
    use rayon::prelude::*;

    let SdpaTensors {
        q,
        k,
        v,
        batch,
        num_heads,
        num_kv_heads,
        q_seq,
        kv_seq,
        head_size,
        v_head_size,
    } = *t;

    debug_assert_eq!(q.len(), batch * num_heads * q_seq * head_size);
    debug_assert_eq!(k.len(), batch * num_kv_heads * kv_seq * head_size);
    debug_assert_eq!(v.len(), batch * num_kv_heads * kv_seq * v_head_size);
    debug_assert_eq!(y.len(), batch * num_heads * q_seq * v_head_size);
    debug_assert!(num_kv_heads > 0 && num_heads.is_multiple_of(num_kv_heads));

    let heads_per_kv = num_heads / num_kv_heads;

    // Both scale modes reduce to `alpha · (Q·K)` under a GEMM: `PostDot(s)`
    // multiplies the dot by `s`, and `SplitSqrt(s)` pre-scales each operand by
    // `√s` so the product carries `s`. Folding `s` into the GEMM `alpha` matches
    // ORT's own MLAS path (`alpha = scale`) and stays within tolerance of the
    // scalar loop's per-operand scaling.
    let alpha = match cfg.scale {
        ScaleMode::PostDot(s) => s,
        ScaleMode::SplitSqrt(s) => s,
    };

    // With no softcap and provably-identity hooks the per-element epilogue is
    // pure overhead: two dynamic dispatches and a branch per logit, which at
    // encoder shapes costs more than either GEMM.
    let plain_epilogue = cfg.softcap.is_none() && bias.is_identity() && mask.is_identity();

    // One tile per `(b, head)`, contiguous in `y` as `[b, head, q_seq, Dv]`.
    let tile_v = q_seq * v_head_size;
    let tile_logits = q_seq * kv_seq;
    y.par_chunks_mut(tile_v)
        .enumerate()
        // The logits scratch is reused across every tile a worker handles: at
        // BERT-base that is a 64 KiB allocation per `(batch, head)` pair, and
        // glibc mmaps (and so re-faults) buffers of that size.
        .for_each_init(Vec::<f32>::new, |logits, (bh, y_tile)| {
            let b = bh / num_heads;
            let n = bh % num_heads;
            let kv_n = n / heads_per_kv;

            let q_off = ((b * num_heads + n) * q_seq) * head_size;
            let k_off = ((b * num_kv_heads + kv_n) * kv_seq) * head_size;
            let v_off = ((b * num_kv_heads + kv_n) * kv_seq) * v_head_size;
            let q_tile = &q[q_off..q_off + q_seq * head_size];
            let k_tile = &k[k_off..k_off + kv_seq * head_size];
            let v_tile = &v[v_off..v_off + kv_seq * v_head_size];

            // logits[q_seq, kv_seq] = alpha · Q · Kᵀ.
            logits.clear();
            logits.resize(tile_logits, 0.0);
            let logits: &mut [f32] = logits.as_mut_slice();
            mlas_sys::sgemm(
                false, true, q_seq, kv_seq, head_size, alpha, q_tile, head_size, k_tile, head_size,
                0.0, logits, kv_seq,
            );

            // Per-row epilogue: softcap → bias → mask → causal → softmax, on
            // plain slices, in the scalar reference's exact add order.
            //
            // When there is no softcap and both hooks are provably identities
            // the whole element loop collapses to an optional causal tail fill,
            // which matters: at BERT-base encoder shapes the two `&dyn` calls
            // per logit cost more than either GEMM.
            if plain_epilogue {
                if cfg.causal {
                    for i in 0..q_seq {
                        let keep = (cfg.past_seq + i + 1).min(kv_seq);
                        logits[i * kv_seq + keep..(i + 1) * kv_seq].fill(cfg.causal_fill);
                    }
                }
            } else {
                for i in 0..q_seq {
                    let row = &mut logits[i * kv_seq..i * kv_seq + kv_seq];
                    for (j, s) in row.iter_mut().enumerate() {
                        let mut val = *s;
                        if let Some(softcap) = cfg.softcap {
                            val = softcap * (val / softcap).tanh();
                        }
                        val += bias.at(b, n, i, j);
                        val += mask.at(b, i, j);
                        if cfg.causal && (j as i64) > cfg.past_seq as i64 + i as i64 {
                            val = cfg.causal_fill;
                        }
                        *s = val;
                    }
                }
            }

            // Numerically-stable softmax with the fully-masked-row → zero guard
            // (matching the scalar reference and ORT). MLAS normalizes all
            // `q_seq` rows with a vectorized exp — the same primitive ORT's own
            // attention kernels use — but it has no `-inf` row convention, so a
            // block that contains any `-inf` falls back to the scalar loop.
            if logits.contains(&f32::NEG_INFINITY) {
                for i in 0..q_seq {
                    softmax_in_place(&mut logits[i * kv_seq..(i + 1) * kv_seq], SoftmaxExp::F32);
                }
            } else {
                mlas_sys::compute_softmax_in_place(logits, q_seq, kv_seq);
            }

            // context[q_seq, Dv] = probs · V.
            mlas_sys::sgemm(
                false,
                false,
                q_seq,
                v_head_size,
                kv_seq,
                1.0,
                logits,
                kv_seq,
                v_tile,
                v_head_size,
                0.0,
                y_tile,
                v_head_size,
            );
        });
}

/// Accelerate (cblas_sgemm) fast path for SDPA on macOS/iOS.
///
/// Mirrors [`sdpa_f32_fast`] (MLAS path) but uses Apple's Accelerate framework
/// which reaches the AMX coprocessor. Per `(batch, head)` tile it runs two
/// SGEMMs — `logits = alpha · Q · Kᵀ` and `context = probs · V` — with the
/// `softcap → bias → mask → causal → softmax` epilogue applied per row.
/// Tiles are parallelized across the crate's shared Rayon pool.
#[cfg(all(any(target_os = "macos", target_os = "ios"), not(feature = "mlas")))]
fn sdpa_f32_accelerate(
    t: &SdpaTensors,
    cfg: &SdpaConfig,
    bias: &dyn AttnBias,
    mask: &dyn KeyMask,
    y: &mut [f32],
) {
    use rayon::prelude::*;

    #[cfg(test)]
    SDPA_ACCELERATE_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
    const CBLAS_ROW_MAJOR: i32 = 101;
    const CBLAS_NO_TRANS: i32 = 111;
    const CBLAS_TRANS: i32 = 112;

    let SdpaTensors {
        q,
        k,
        v,
        batch,
        num_heads,
        num_kv_heads,
        q_seq,
        kv_seq,
        head_size,
        v_head_size,
    } = *t;

    debug_assert_eq!(q.len(), batch * num_heads * q_seq * head_size);
    debug_assert_eq!(k.len(), batch * num_kv_heads * kv_seq * head_size);
    debug_assert_eq!(v.len(), batch * num_kv_heads * kv_seq * v_head_size);
    debug_assert_eq!(y.len(), batch * num_heads * q_seq * v_head_size);
    debug_assert!(num_kv_heads > 0 && num_heads.is_multiple_of(num_kv_heads));

    let heads_per_kv = num_heads / num_kv_heads;

    let alpha = match cfg.scale {
        ScaleMode::PostDot(s) => s,
        ScaleMode::SplitSqrt(s) => s,
    };

    let tile_v = q_seq * v_head_size;
    y.par_chunks_mut(tile_v)
        .enumerate()
        .for_each(|(bh, y_tile)| {
            let b = bh / num_heads;
            let n = bh % num_heads;
            let kv_n = n / heads_per_kv;

            let q_off = ((b * num_heads + n) * q_seq) * head_size;
            let k_off = ((b * num_kv_heads + kv_n) * kv_seq) * head_size;
            let v_off = ((b * num_kv_heads + kv_n) * kv_seq) * v_head_size;
            let q_tile = &q[q_off..q_off + q_seq * head_size];
            let k_tile = &k[k_off..k_off + kv_seq * head_size];
            let v_tile = &v[v_off..v_off + kv_seq * v_head_size];

            // logits[q_seq, kv_seq] = alpha · Q[q_seq, head_size] · K[kv_seq, head_size]ᵀ
            let mut logits = vec![0.0f32; q_seq * kv_seq];
            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_TRANS,
                    q_seq as i32,
                    kv_seq as i32,
                    head_size as i32,
                    alpha,
                    q_tile.as_ptr(),
                    head_size as i32,
                    k_tile.as_ptr(),
                    head_size as i32,
                    0.0,
                    logits.as_mut_ptr(),
                    kv_seq as i32,
                );
            }

            // Per-row epilogue: softcap → bias → mask → causal → softmax.
            for i in 0..q_seq {
                let row = &mut logits[i * kv_seq..i * kv_seq + kv_seq];
                for (j, s) in row.iter_mut().enumerate() {
                    let mut val = *s;
                    if let Some(softcap) = cfg.softcap {
                        val = softcap * (val / softcap).tanh();
                    }
                    val += bias.at(b, n, i, j);
                    val += mask.at(b, i, j);
                    if cfg.causal && (j as i64) > cfg.past_seq as i64 + i as i64 {
                        val = cfg.causal_fill;
                    }
                    *s = val;
                }

                softmax_in_place(row, SoftmaxExp::F32);
            }

            // context[q_seq, v_head_size] = probs[q_seq, kv_seq] · V[kv_seq, v_head_size]
            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
                    q_seq as i32,
                    v_head_size as i32,
                    kv_seq as i32,
                    1.0,
                    logits.as_ptr(),
                    kv_seq as i32,
                    v_tile.as_ptr(),
                    v_head_size as i32,
                    0.0,
                    y_tile.as_mut_ptr(),
                    v_head_size as i32,
                );
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `sdpa_decode_group` must be a pure loop-nesting inversion of
    /// `sdpa_decode_row`: same operations, same order, per query head. Anything
    /// less would silently change decode logits, so this asserts on raw bits
    /// (not a tolerance) across a group of 7 heads with softcap, a non-trivial
    /// window, and `head_size != v_head_size`.
    #[test]
    fn group_decode_matches_row_decode_bitwise() {
        for &(group, kv_seq, dh, dv, lo, hi) in &[
            (7usize, 37usize, 64usize, 64usize, 0usize, 30usize),
            (2, 23, 133, 17, 5, 21),
            (4, 64, 128, 128, 0, 64),
            (8, 9, 48, 48, 3, 9),
            // Degenerate: empty window and a single-element window.
            (3, 11, 32, 32, 4, 4),
            (3, 11, 32, 32, 4, 5),
            // Windows longer than one KV tile (`group_kv_tile`), including a
            // ragged final tile and asymmetric K/V head sizes (so the QK and
            // P·V passes tile at different widths).
            (5, 600, 64, 64, 0, 600),
            (2, 700, 32, 128, 13, 691),
        ] {
            let scale = 1.0 / (dh as f32).sqrt();
            for softcap in [None, Some(7.5f32)] {
                let q: Vec<f32> = (0..group * dh)
                    .map(|i| ((i * 17 % 101) as f32 - 50.0) / 37.0)
                    .collect();
                let k: Vec<f32> = (0..kv_seq * dh)
                    .map(|i| ((i * 29 % 211) as f32 - 105.0) / 61.0)
                    .collect();
                let v: Vec<f32> = (0..kv_seq * dv)
                    .map(|i| ((i * 43 % 157) as f32 - 78.0) / 53.0)
                    .collect();

                let mut expected = vec![f32::NAN; group * dv];
                for g in 0..group {
                    sdpa_decode_row(
                        &q[g * dh..(g + 1) * dh],
                        &k,
                        &v,
                        kv_seq,
                        lo,
                        hi,
                        scale,
                        softcap,
                        SoftmaxExp::F64Intermediate,
                        &mut expected[g * dv..(g + 1) * dv],
                    );
                }

                let mut actual = vec![f32::NAN; group * dv];
                let mut scores = Vec::new();
                sdpa_decode_group(
                    &q,
                    &k,
                    &v,
                    kv_seq,
                    group,
                    dh,
                    dv,
                    lo,
                    hi,
                    scale,
                    softcap,
                    SoftmaxExp::F64Intermediate,
                    &mut actual,
                    &mut scores,
                );

                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "group={group} kv_seq={kv_seq} dh={dh} dv={dv} window={lo}..{hi} softcap={softcap:?}"
                );
            }
        }
    }

    /// The caller-owned `scores` scratch is reused across calls with different
    /// window lengths and group widths; a stale tail must never leak into a
    /// later result.
    #[test]
    fn group_decode_reuses_scratch_without_leaking_state() {
        let (kv_seq, dh, dv) = (48usize, 32usize, 32usize);
        let scale = 1.0 / (dh as f32).sqrt();
        let k: Vec<f32> = (0..kv_seq * dh)
            .map(|i| ((i * 29 % 211) as f32 - 105.0) / 61.0)
            .collect();
        let v: Vec<f32> = (0..kv_seq * dv)
            .map(|i| ((i * 43 % 157) as f32 - 78.0) / 53.0)
            .collect();
        let mut scores = Vec::new();
        // A long, wide call first so the scratch is oversized for the next one.
        for &(group, lo, hi) in &[(8usize, 0usize, 48usize), (2, 10, 20), (4, 0, 7)] {
            let q: Vec<f32> = (0..group * dh)
                .map(|i| ((i * 17 % 101) as f32 - 50.0) / 37.0)
                .collect();
            let mut fresh_scratch = Vec::new();
            let mut reused = vec![f32::NAN; group * dv];
            let mut fresh = vec![f32::NAN; group * dv];
            sdpa_decode_group(
                &q,
                &k,
                &v,
                kv_seq,
                group,
                dh,
                dv,
                lo,
                hi,
                scale,
                None,
                SoftmaxExp::F64Intermediate,
                &mut reused,
                &mut scores,
            );
            sdpa_decode_group(
                &q,
                &k,
                &v,
                kv_seq,
                group,
                dh,
                dv,
                lo,
                hi,
                scale,
                None,
                SoftmaxExp::F64Intermediate,
                &mut fresh,
                &mut fresh_scratch,
            );
            assert_eq!(
                reused.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                fresh.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "group={group} window={lo}..{hi}"
            );
        }
    }

    #[test]
    fn decode_row_f64_intermediate_is_bit_exact_with_gqa_reference() {
        let (kv_seq, dh, dv) = (23usize, 133usize, 17usize);
        let (lo, hi) = (5usize, 21usize);
        let scale = 1.0 / (dh as f32).sqrt();
        let softcap = 7.5f32;
        let q: Vec<f32> = (0..dh)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) / 37.0)
            .collect();
        let k: Vec<f32> = (0..kv_seq * dh)
            .map(|i| ((i * 29 % 211) as f32 - 105.0) / 61.0)
            .collect();
        let v: Vec<f32> = (0..kv_seq * dv)
            .map(|i| ((i * 43 % 157) as f32 - 78.0) / 53.0)
            .collect();

        // The pre-consolidation GQA decode loop, retained here as the bit oracle.
        let mut scores = vec![0.0f32; hi - lo];
        for (i, ks) in (lo..hi).enumerate() {
            let k_base = ks * dh;
            let mut score = dot_f32(&q, &k[k_base..k_base + dh]);
            score *= scale;
            score = softcap * (score / softcap).tanh();
            scores[i] = score;
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
        let mut expected = vec![0.0f32; dv];
        for (i, ks) in (lo..hi).enumerate() {
            let probability = scores[i];
            if probability == 0.0 {
                continue;
            }
            axpy_f32(&mut expected, probability, &v[ks * dv..(ks + 1) * dv]);
        }

        let mut actual = vec![f32::NAN; dv];
        sdpa_decode_row(
            &q,
            &k,
            &v,
            kv_seq,
            lo,
            hi,
            scale,
            Some(softcap),
            SoftmaxExp::F64Intermediate,
            &mut actual,
        );
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }

    /// Split-KV (flash-decoding) parity: [`sdpa_decode_partial`] +
    /// [`combine_decode_partials`] over many `(kv_len, split_count, head_size)`
    /// combinations must reproduce the sequential [`sdpa_decode_row`] reference to
    /// a tight max-abs-error bar. It is deliberately *not* bit-exact: the split
    /// reorders the float additions and adds the online rescale multiplies, so a
    /// small drift is expected. The bound is set to `1e-6`; the f64 intermediates
    /// keep every observed case far under it. Edge cases covered: `P = 1`,
    /// `P > kv_len` (empty chunks), sliding window `lo > 0`, `kv_len` not
    /// divisible by `P`, and a fully-masked/empty window.
    #[test]
    fn split_decode_matches_sequential_reference_within_tolerance() {
        const TOLERANCE: f32 = 1e-6;

        fn run_case(kv_seq: usize, dh: usize, dv: usize, lo: usize, hi: usize, split_count: usize) {
            let scale = 1.0 / (dh.max(1) as f32).sqrt();
            let softcap = Some(6.25f32);
            let q: Vec<f32> = (0..dh)
                .map(|i| ((i * 13 % 97) as f32 - 48.0) / 29.0)
                .collect();
            let k: Vec<f32> = (0..kv_seq * dh)
                .map(|i| ((i * 31 % 199) as f32 - 99.0) / 57.0)
                .collect();
            let v: Vec<f32> = (0..kv_seq * dv)
                .map(|i| ((i * 37 % 173) as f32 - 86.0) / 47.0)
                .collect();

            let mut reference = vec![f32::NAN; dv];
            sdpa_decode_row(
                &q,
                &k,
                &v,
                kv_seq,
                lo,
                hi,
                scale,
                softcap,
                SoftmaxExp::F64Intermediate,
                &mut reference,
            );

            // Split `[lo, hi)` into `split_count` contiguous chunks the same way
            // the GQA scheduler does, compute each chunk's partial, then combine.
            let length = hi - lo;
            let base = length / split_count;
            let remainder = length % split_count;
            let mut partials = Vec::with_capacity(split_count);
            let mut partial_outputs = vec![0.0f64; split_count * dv];
            for chunk in 0..split_count {
                let chunk_lo = lo + chunk * base + chunk.min(remainder);
                let chunk_hi = chunk_lo + base + usize::from(chunk < remainder);
                let slot = &mut partial_outputs[chunk * dv..(chunk + 1) * dv];
                partials.push(sdpa_decode_partial(
                    &q, &k, &v, kv_seq, chunk_lo, chunk_hi, scale, softcap, slot,
                ));
            }
            let mut combined = vec![f32::NAN; dv];
            combine_decode_partials(&partials, &partial_outputs, dv, &mut combined);

            let mut max_abs_error = 0.0f32;
            for (&reference_value, &combined_value) in reference.iter().zip(&combined) {
                max_abs_error = max_abs_error.max((reference_value - combined_value).abs());
            }
            assert!(
                max_abs_error <= TOLERANCE,
                "kv_seq={kv_seq} dh={dh} dv={dv} lo={lo} hi={hi} split_count={split_count}: \
                 max abs error {max_abs_error} exceeds {TOLERANCE}"
            );
        }

        // Full window, exact and non-divisible splits, several head sizes.
        for &(dh, dv) in &[(64usize, 64usize), (128, 128), (96, 40), (133, 17)] {
            for &kv_seq in &[1usize, 2, 7, 64, 200, 1024] {
                for &split_count in &[1usize, 2, 3, 4, 8, 16] {
                    run_case(kv_seq, dh, dv, 0, kv_seq, split_count);
                }
            }
        }
        // Sliding window (lo > 0), including kv_len not divisible by P.
        run_case(200, 128, 128, 37, 200, 4);
        run_case(200, 128, 128, 37, 200, 7);
        run_case(1024, 96, 40, 511, 1024, 5);
        // P greater than the window length -> trailing chunks are empty.
        run_case(5, 64, 64, 0, 5, 8);
        run_case(5, 64, 64, 2, 5, 16);
        // Fully-masked / empty window -> all chunks empty, output must be zero.
        run_case(16, 64, 64, 8, 8, 4);
    }

    /// Straightforward f32 SDPA reference for cross-checking the core on small
    /// shapes (single head, no bias/mask, PostDot scale).
    fn reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        s: usize,
        dh: usize,
        dv: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; s * dv];
        for i in 0..s {
            let mut scores = vec![0.0f32; s];
            for (j, sc) in scores.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for p in 0..dh {
                    acc += q[i * dh + p] * k[j * dh + p];
                }
                *sc = acc * scale;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|x| (x - m).exp()).sum();
            for c in 0..dv {
                let mut acc = 0.0f32;
                for (j, sc) in scores.iter().enumerate() {
                    acc += ((sc - m).exp() / sum) * v[j * dv + c];
                }
                out[i * dv + c] = acc;
            }
        }
        out
    }

    #[test]
    fn postdot_matches_reference() {
        let (s, dh, dv) = (3usize, 4usize, 2usize);
        let q: Vec<f32> = (0..s * dh).map(|x| (x as f32) * 0.1 - 0.5).collect();
        let k: Vec<f32> = (0..s * dh).map(|x| (x as f32) * 0.05).collect();
        let v: Vec<f32> = (0..s * dv).map(|x| (x as f32) * 0.2).collect();
        let scale = 1.0 / (dh as f32).sqrt();
        let t = SdpaTensors {
            q: &q,
            k: &k,
            v: &v,
            batch: 1,
            num_heads: 1,
            num_kv_heads: 1,
            q_seq: s,
            kv_seq: s,
            head_size: dh,
            v_head_size: dv,
        };
        let cfg = SdpaConfig {
            scale: ScaleMode::PostDot(scale),
            softcap: None,
            causal: false,
            past_seq: 0,
            causal_fill: f32::MIN,
        };
        let mut y = vec![0.0f32; s * dv];
        sdpa_f32_scalar(&t, &cfg, &NoBias, &NoMask, &mut y, None);
        let want = reference(&q, &k, &v, s, dh, dv, scale);
        for (a, b) in y.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-6, "got {y:?} want {want:?}");
        }
    }

    #[test]
    fn causal_masks_future_keys() {
        // With causal masking and past_seq=0, query 0 must attend only key 0.
        let (s, dh, dv) = (2usize, 2usize, 2usize);
        let q = vec![1.0f32, 0.0, 0.0, 1.0];
        let k = vec![1.0f32, 0.0, 0.0, 1.0];
        let v = vec![10.0f32, 20.0, 30.0, 40.0];
        let t = SdpaTensors {
            q: &q,
            k: &k,
            v: &v,
            batch: 1,
            num_heads: 1,
            num_kv_heads: 1,
            q_seq: s,
            kv_seq: s,
            head_size: dh,
            v_head_size: dv,
        };
        let cfg = SdpaConfig {
            scale: ScaleMode::PostDot(1.0),
            softcap: None,
            causal: true,
            past_seq: 0,
            causal_fill: f32::MIN,
        };
        let mut y = vec![0.0f32; s * dv];
        sdpa_f32_scalar(&t, &cfg, &NoBias, &NoMask, &mut y, None);
        // Query 0 attends only key 0 → exactly V row 0.
        assert!((y[0] - 10.0).abs() < 1e-6 && (y[1] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn gqa_head_sharing_reads_grouped_kv() {
        // 2 query heads, 1 kv head: both query heads must read the same kv head.
        let (s, dh, dv) = (1usize, 2usize, 2usize);
        let q = vec![1.0f32, 0.0, /*h1*/ 0.0, 1.0];
        let k = vec![1.0f32, 1.0]; // single kv head, single key
        let v = vec![5.0f32, 7.0];
        let t = SdpaTensors {
            q: &q,
            k: &k,
            v: &v,
            batch: 1,
            num_heads: 2,
            num_kv_heads: 1,
            q_seq: s,
            kv_seq: s,
            head_size: dh,
            v_head_size: dv,
        };
        let cfg = SdpaConfig {
            scale: ScaleMode::PostDot(1.0),
            softcap: None,
            causal: false,
            past_seq: 0,
            causal_fill: f32::MIN,
        };
        let mut y = vec![0.0f32; 2 * s * dv];
        sdpa_f32_scalar(&t, &cfg, &NoBias, &NoMask, &mut y, None);
        // Single key → softmax is 1.0 → both heads output V row 0.
        for h in 0..2 {
            assert!((y[h * dv] - 5.0).abs() < 1e-6 && (y[h * dv + 1] - 7.0).abs() < 1e-6);
        }
    }

    #[test]
    fn splitsqrt_scale_equivalent_to_postdot_for_moderate_values() {
        // √scale-on-operands and scale-on-dot agree closely for moderate mags.
        let (s, dh, dv) = (2usize, 3usize, 2usize);
        let q: Vec<f32> = (0..s * dh).map(|x| (x as f32) * 0.3).collect();
        let k: Vec<f32> = (0..s * dh).map(|x| (x as f32) * 0.2 - 0.1).collect();
        let v: Vec<f32> = (0..s * dv).map(|x| (x as f32) * 0.5).collect();
        let scale = 1.0 / (dh as f32).sqrt();
        let base = SdpaTensors {
            q: &q,
            k: &k,
            v: &v,
            batch: 1,
            num_heads: 1,
            num_kv_heads: 1,
            q_seq: s,
            kv_seq: s,
            head_size: dh,
            v_head_size: dv,
        };
        let mut y_post = vec![0.0f32; s * dv];
        sdpa_f32_scalar(
            &base,
            &SdpaConfig {
                scale: ScaleMode::PostDot(scale),
                softcap: None,
                causal: false,
                past_seq: 0,
                causal_fill: f32::MIN,
            },
            &NoBias,
            &NoMask,
            &mut y_post,
            None,
        );
        let mut y_split = vec![0.0f32; s * dv];
        sdpa_f32_scalar(
            &base,
            &SdpaConfig {
                scale: ScaleMode::SplitSqrt(scale),
                softcap: None,
                causal: false,
                past_seq: 0,
                causal_fill: f32::MIN,
            },
            &NoBias,
            &NoMask,
            &mut y_split,
            None,
        );
        for (a, b) in y_post.iter().zip(y_split.iter()) {
            assert!((a - b).abs() < 1e-5, "post {y_post:?} split {y_split:?}");
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
    fn deterministic_values(n: usize, seed: u64, magnitude: f32) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                s ^= s >> 30;
                s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                s ^= s >> 27;
                let unit = ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
                unit * magnitude
            })
            .collect()
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
    struct PatternBias {
        q_seq: usize,
        kv_seq: usize,
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
    impl AttnBias for PatternBias {
        fn at(&self, b: usize, head: usize, i: usize, j: usize) -> f32 {
            let idx = (((b * 17 + head * 13 + i) * self.kv_seq + j) % 19) as f32;
            debug_assert!(i < self.q_seq);
            (idx - 9.0) * 0.03125
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
    struct PatternMask {
        q_seq: usize,
        kv_seq: usize,
        fully_masked_query: Option<usize>,
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
    impl KeyMask for PatternMask {
        fn at(&self, b: usize, i: usize, j: usize) -> f32 {
            debug_assert!(i < self.q_seq && j < self.kv_seq);
            if self
                .fully_masked_query
                .is_some_and(|query| query == b * self.q_seq + i)
            {
                return f32::NEG_INFINITY;
            }
            if (b * 31 + i * 7 + j * 3).is_multiple_of(11) {
                f32::NEG_INFINITY
            } else if (i + j).is_multiple_of(13) {
                -37.0
            } else {
                0.0
            }
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
    fn sdpa_f64_reference(
        t: &SdpaTensors,
        cfg: &SdpaConfig,
        bias: &dyn AttnBias,
        mask: &dyn KeyMask,
    ) -> Vec<f32> {
        let SdpaTensors {
            q,
            k,
            v,
            batch,
            num_heads,
            num_kv_heads,
            q_seq,
            kv_seq,
            head_size,
            v_head_size,
        } = *t;
        let heads_per_kv = num_heads / num_kv_heads;
        let (post_scale, operand_scale) = match cfg.scale {
            ScaleMode::PostDot(s) => (s as f64, 1.0f64),
            ScaleMode::SplitSqrt(s) => (1.0f64, (s as f64).sqrt()),
        };
        let mut y = vec![0.0f32; batch * num_heads * q_seq * v_head_size];
        let mut scores = vec![0.0f64; kv_seq];
        for b in 0..batch {
            for n in 0..num_heads {
                let kv_n = n / heads_per_kv;
                for i in 0..q_seq {
                    let q_base = ((b * num_heads + n) * q_seq + i) * head_size;
                    for (j, score) in scores.iter_mut().enumerate() {
                        let k_base = ((b * num_kv_heads + kv_n) * kv_seq + j) * head_size;
                        let mut acc = 0.0f64;
                        for p in 0..head_size {
                            acc += (q[q_base + p] as f64 * operand_scale)
                                * (k[k_base + p] as f64 * operand_scale);
                        }
                        let mut s = acc * post_scale;
                        if let Some(softcap) = cfg.softcap {
                            let softcap = softcap as f64;
                            s = softcap * (s / softcap).tanh();
                        }
                        s += bias.at(b, n, i, j) as f64;
                        s += mask.at(b, i, j) as f64;
                        if cfg.causal && (j as i64) > cfg.past_seq as i64 + i as i64 {
                            s = cfg.causal_fill as f64;
                        }
                        *score = s;
                    }
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let y_base = ((b * num_heads + n) * q_seq + i) * v_head_size;
                    if max == f64::NEG_INFINITY {
                        y[y_base..y_base + v_head_size].fill(0.0);
                        continue;
                    }
                    let mut denominator = 0.0f64;
                    for score in &mut scores {
                        *score = (*score - max).exp();
                        denominator += *score;
                    }
                    let mut acc = vec![0.0f64; v_head_size];
                    if denominator > 0.0 {
                        for (j, &weight) in scores.iter().enumerate() {
                            let probability = weight / denominator;
                            if probability == 0.0 {
                                continue;
                            }
                            let v_base = ((b * num_kv_heads + kv_n) * kv_seq + j) * v_head_size;
                            for c in 0..v_head_size {
                                acc[c] += probability * v[v_base + c] as f64;
                            }
                        }
                    }
                    for c in 0..v_head_size {
                        y[y_base + c] = acc[c] as f32;
                    }
                }
            }
        }
        y
    }

    /// The SIMD SDPA path must agree with the scalar oracle (and with an f64
    /// reference) across GQA / causal / softcap / bias / mask / fully-masked
    /// cases. Runs on aarch64 *and* x86, since both dispatch to `sdpa_f32_simd`.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn sdpa_simd_matches_scalar_and_f64_reference_on_decode_shapes() {
        struct Case {
            name: &'static str,
            batch: usize,
            num_heads: usize,
            num_kv_heads: usize,
            q_seq: usize,
            kv_seq: usize,
            head_size: usize,
            v_head_size: usize,
            magnitude: f32,
            cfg: SdpaConfig,
            use_bias: bool,
            use_mask: bool,
            fully_masked_query: Option<usize>,
        }

        let cases = [
            Case {
                name: "qwen-style-decode-gqa",
                batch: 1,
                num_heads: 14,
                num_kv_heads: 2,
                q_seq: 1,
                kv_seq: 257,
                head_size: 64,
                v_head_size: 64,
                magnitude: 0.75,
                cfg: SdpaConfig {
                    scale: ScaleMode::PostDot(1.0 / 8.0),
                    softcap: None,
                    causal: false,
                    past_seq: 256,
                    causal_fill: f32::MIN,
                },
                use_bias: false,
                use_mask: false,
                fully_masked_query: None,
            },
            Case {
                name: "odd-dh-dv-tail-masked",
                batch: 1,
                num_heads: 8,
                num_kv_heads: 2,
                q_seq: 3,
                kv_seq: 129,
                head_size: 133,
                v_head_size: 65,
                magnitude: 0.5,
                cfg: SdpaConfig {
                    scale: ScaleMode::SplitSqrt(1.0 / 133.0_f32.sqrt()),
                    softcap: Some(7.5),
                    causal: true,
                    past_seq: 126,
                    causal_fill: f32::NEG_INFINITY,
                },
                use_bias: true,
                use_mask: true,
                fully_masked_query: Some(2),
            },
            Case {
                name: "large-score-softmax-stability",
                batch: 1,
                num_heads: 4,
                num_kv_heads: 1,
                q_seq: 2,
                kv_seq: 33,
                head_size: 65,
                v_head_size: 17,
                magnitude: 48.0,
                cfg: SdpaConfig {
                    scale: ScaleMode::PostDot(1.0),
                    softcap: None,
                    causal: false,
                    past_seq: 0,
                    causal_fill: f32::NEG_INFINITY,
                },
                use_bias: true,
                use_mask: true,
                fully_masked_query: None,
            },
        ];

        for case in cases {
            let q_len = case.batch * case.num_heads * case.q_seq * case.head_size;
            let k_len = case.batch * case.num_kv_heads * case.kv_seq * case.head_size;
            let v_len = case.batch * case.num_kv_heads * case.kv_seq * case.v_head_size;
            let y_len = case.batch * case.num_heads * case.q_seq * case.v_head_size;
            let q = deterministic_values(q_len, 0x1000 + q_len as u64, case.magnitude);
            let k = deterministic_values(k_len, 0x2000 + k_len as u64, case.magnitude);
            let v = deterministic_values(v_len, 0x3000 + v_len as u64, 0.75);
            let tensors = SdpaTensors {
                q: &q,
                k: &k,
                v: &v,
                batch: case.batch,
                num_heads: case.num_heads,
                num_kv_heads: case.num_kv_heads,
                q_seq: case.q_seq,
                kv_seq: case.kv_seq,
                head_size: case.head_size,
                v_head_size: case.v_head_size,
            };
            let bias = PatternBias {
                q_seq: case.q_seq,
                kv_seq: case.kv_seq,
            };
            let mask = PatternMask {
                q_seq: case.q_seq,
                kv_seq: case.kv_seq,
                fully_masked_query: case.fully_masked_query,
            };
            let bias_ref: &dyn AttnBias = if case.use_bias { &bias } else { &NoBias };
            let mask_ref: &dyn KeyMask = if case.use_mask { &mask } else { &NoMask };
            let mut scalar = vec![f32::NAN; y_len];
            let mut simd = vec![f32::NAN; y_len];
            sdpa_f32_scalar(&tensors, &case.cfg, bias_ref, mask_ref, &mut scalar, None);
            sdpa_f32_simd(&tensors, &case.cfg, bias_ref, mask_ref, &mut simd);
            let f64_ref = sdpa_f64_reference(&tensors, &case.cfg, bias_ref, mask_ref);

            let mut max_scalar_abs = 0.0f32;
            let mut max_f64_abs = 0.0f32;
            let mut max_scalar_rel = 0.0f32;
            for ((&got, &scalar), &f64_value) in simd.iter().zip(&scalar).zip(&f64_ref) {
                assert!(got.is_finite(), "{} produced non-finite output", case.name);
                let scalar_abs = (got - scalar).abs();
                let f64_abs = (got - f64_value).abs();
                max_scalar_abs = max_scalar_abs.max(scalar_abs);
                max_f64_abs = max_f64_abs.max(f64_abs);
                max_scalar_rel = max_scalar_rel.max(scalar_abs / scalar.abs().max(1e-4));
            }
            // The SIMD path reduces through independent accumulators while the
            // scalar reference is a single sequential chain, so exact parity is
            // not expected. The
            // bound is still tight enough to catch dropped tail lanes, missing
            // max subtraction, and accumulator corruption; guard-break probes
            // for those fail.
            assert!(
                max_scalar_abs <= 5e-4 && max_scalar_rel <= 2e-3,
                "{}: simd vs scalar max_abs={max_scalar_abs:e} max_rel={max_scalar_rel:e}",
                case.name
            );
            assert!(
                max_f64_abs <= 1e-3,
                "{}: simd vs f64 max_abs={max_f64_abs:e}",
                case.name
            );
        }
    }

    /// The *dispatcher* (not the vectorised body called directly) must stay
    /// within tolerance of the scalar oracle for batched, grouped, causal and
    /// softcapped shapes. This is the entry point `Attention`,
    /// `MultiHeadAttention`, `com.microsoft.Attention` and `VarLenAttention`
    /// actually call, so it is what production numerics depend on.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn sdpa_dispatch_matches_scalar_oracle_across_shapes() {
        // (batch, heads, kv_heads, q_seq, kv_seq, head_size, v_head_size)
        let shapes = [
            (2usize, 8usize, 8usize, 1usize, 1usize, 64usize, 64usize),
            (2, 12, 4, 1, 513, 64, 64),
            (1, 32, 8, 1, 2049, 128, 128),
            (3, 6, 3, 7, 71, 33, 17),
            (1, 4, 1, 129, 129, 80, 80),
        ];
        for (bi, &(batch, heads, kv_heads, q_seq, kv_seq, dh, dv)) in shapes.iter().enumerate() {
            let q = deterministic_values(batch * heads * q_seq * dh, 0x0005_EED0 + bi as u64, 0.9);
            let k =
                deterministic_values(batch * kv_heads * kv_seq * dh, 0x0005_EED1 + bi as u64, 0.9);
            let v =
                deterministic_values(batch * kv_heads * kv_seq * dv, 0x0005_EED2 + bi as u64, 0.9);
            let tensors = SdpaTensors {
                q: &q,
                k: &k,
                v: &v,
                batch,
                num_heads: heads,
                num_kv_heads: kv_heads,
                q_seq,
                kv_seq,
                head_size: dh,
                v_head_size: dv,
            };
            for (label, cfg) in [
                (
                    "plain",
                    SdpaConfig {
                        scale: ScaleMode::PostDot(1.0 / (dh as f32).sqrt()),
                        softcap: None,
                        causal: false,
                        past_seq: 0,
                        causal_fill: f32::NEG_INFINITY,
                    },
                ),
                (
                    "causal-softcap",
                    SdpaConfig {
                        scale: ScaleMode::SplitSqrt(1.0 / (dh as f32).sqrt().sqrt()),
                        softcap: Some(30.0),
                        causal: true,
                        past_seq: kv_seq.saturating_sub(q_seq),
                        causal_fill: f32::MIN,
                    },
                ),
            ] {
                let y_len = batch * heads * q_seq * dv;
                let mut dispatched = vec![f32::NAN; y_len];
                let mut oracle = vec![f32::NAN; y_len];
                sdpa_f32(&tensors, &cfg, &NoBias, &NoMask, &mut dispatched, None);
                sdpa_f32_scalar(&tensors, &cfg, &NoBias, &NoMask, &mut oracle, None);
                for (&got, &want) in dispatched.iter().zip(&oracle) {
                    assert!(
                        got.is_finite(),
                        "shape {bi} {label}: dispatcher produced non-finite output"
                    );
                    let abs = (got - want).abs();
                    // Observed worst case on AVX2 is abs=2.7e-7 / rel=1.1e-3
                    // (the relative figure is dominated by the 1e-4 floor on
                    // near-zero outputs), so these bounds leave ~75x abs
                    // headroom while still catching a single dropped KV lane,
                    // which perturbs an output by ~1e-3.
                    assert!(
                        abs <= 2e-5 && abs / want.abs().max(1e-4) <= 2e-3,
                        "shape {bi} {label}: dispatcher vs scalar oracle abs={abs:e}"
                    );
                }
            }
        }
    }

    /// A `QkCapture` request must keep taking the scalar reference on every
    /// architecture: the captured logits are the oracle the parity goldens pin,
    /// so they have to stay bit-identical to `sdpa_f32_scalar`.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn qk_capture_stays_bit_identical_to_the_scalar_oracle() {
        let (batch, heads, kv_heads, q_seq, kv_seq, dh) =
            (2usize, 8usize, 2usize, 3usize, 67usize, 64usize);
        let q = deterministic_values(batch * heads * q_seq * dh, 0x00C0_FFE1, 0.8);
        let k = deterministic_values(batch * kv_heads * kv_seq * dh, 0x00C0_FFE2, 0.8);
        let v = deterministic_values(batch * kv_heads * kv_seq * dh, 0x00C0_FFE3, 0.8);
        let tensors = SdpaTensors {
            q: &q,
            k: &k,
            v: &v,
            batch,
            num_heads: heads,
            num_kv_heads: kv_heads,
            q_seq,
            kv_seq,
            head_size: dh,
            v_head_size: dh,
        };
        let cfg = SdpaConfig {
            scale: ScaleMode::PostDot(1.0 / (dh as f32).sqrt()),
            softcap: None,
            causal: true,
            past_seq: kv_seq - q_seq,
            causal_fill: f32::NEG_INFINITY,
        };
        let y_len = batch * heads * q_seq * dh;
        let score_len = batch * heads * q_seq * kv_seq;
        for stage in [QkCaptureStage::PreSoftmax, QkCaptureStage::PostSoftmax] {
            let mut y_dispatch = vec![f32::NAN; y_len];
            let mut y_oracle = vec![f32::NAN; y_len];
            let mut scores_dispatch = vec![f32::NAN; score_len];
            let mut scores_oracle = vec![f32::NAN; score_len];
            sdpa_f32(
                &tensors,
                &cfg,
                &NoBias,
                &NoMask,
                &mut y_dispatch,
                Some(QkCapture {
                    scores: &mut scores_dispatch,
                    stage,
                }),
            );
            sdpa_f32_scalar(
                &tensors,
                &cfg,
                &NoBias,
                &NoMask,
                &mut y_oracle,
                Some(QkCapture {
                    scores: &mut scores_oracle,
                    stage,
                }),
            );
            assert_eq!(
                scores_dispatch.to_bits_vec(),
                scores_oracle.to_bits_vec(),
                "captured scores diverged from the scalar oracle"
            );
            assert_eq!(
                y_dispatch.to_bits_vec(),
                y_oracle.to_bits_vec(),
                "capture-path output diverged from the scalar oracle"
            );
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
    trait ToBitsVec {
        fn to_bits_vec(&self) -> Vec<u32>;
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
    impl ToBitsVec for [f32] {
        fn to_bits_vec(&self) -> Vec<u32> {
            self.iter().map(|value| value.to_bits()).collect()
        }
    }

    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"),
        not(feature = "mlas")
    ))]
    #[test]
    fn sdpa_dispatcher_reaches_simd_path() {
        use std::sync::atomic::Ordering;

        // On x86 the SIMD path is runtime-detected; a pre-AVX2 host legitimately
        // stays on the scalar reference.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if !crate::backend::has_simd_x86() {
            return;
        }

        // Use q_seq > 1 so the Accelerate path fires on macOS (the NEON decode
        // bypass only applies when q_seq == 1 with small per-head work).
        let (batch, num_heads, num_kv_heads, q_seq, kv_seq, dh, dv) =
            (1usize, 4usize, 2usize, 2usize, 11usize, 17usize, 9usize);
        let q = deterministic_values(batch * num_heads * q_seq * dh, 0xA11CE, 0.5);
        let k = deterministic_values(batch * num_kv_heads * kv_seq * dh, 0xB0B, 0.5);
        let v = deterministic_values(batch * num_kv_heads * kv_seq * dv, 0xCAFE, 0.5);
        let tensors = SdpaTensors {
            q: &q,
            k: &k,
            v: &v,
            batch,
            num_heads,
            num_kv_heads,
            q_seq,
            kv_seq,
            head_size: dh,
            v_head_size: dv,
        };
        let cfg = SdpaConfig {
            scale: ScaleMode::PostDot(1.0 / (dh as f32).sqrt()),
            softcap: None,
            causal: false,
            past_seq: 0,
            causal_fill: f32::NEG_INFINITY,
        };
        // On macOS/iOS without MLAS, the Accelerate path fires instead of NEON.
        // On other aarch64 (Linux), the NEON path fires.
        #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
        let before = SDPA_ACCELERATE_TEST_HITS.load(Ordering::Relaxed);
        #[cfg(not(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios"))))]
        let before = SDPA_SIMD_TEST_HITS.load(Ordering::Relaxed);

        let mut y = vec![f32::NAN; batch * num_heads * q_seq * dv];
        sdpa_f32(&tensors, &cfg, &NoBias, &NoMask, &mut y, None);

        #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
        let after = SDPA_ACCELERATE_TEST_HITS.load(Ordering::Relaxed);
        #[cfg(not(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios"))))]
        let after = SDPA_SIMD_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            after > before,
            "sdpa_f32 dispatcher did not execute the accelerated path"
        );
        assert!(y.iter().all(|value| value.is_finite()));
    }

    /// Verify the inline NEON decode branch fires for small per-head work
    /// (q_seq=1, kv_seq×head_size ≤ 8192) on macOS, bypassing Accelerate.
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "ios"),
        not(feature = "mlas")
    ))]
    #[test]
    fn sdpa_neon_decode_small_dispatch_fires() {
        use std::sync::atomic::Ordering;

        // Decode shape: q_seq=1, head_size=64, kv_seq=64 → 4096 ≤ 8192 → NEON
        let (batch, num_heads, num_kv_heads, q_seq, kv_seq, dh, dv) =
            (1usize, 14usize, 2usize, 1usize, 64usize, 64usize, 64usize);
        let q = deterministic_values(batch * num_heads * q_seq * dh, 0xDEC_0DEA, 0.5);
        let k = deterministic_values(batch * num_kv_heads * kv_seq * dh, 0xDEC_0DEB, 0.5);
        let v = deterministic_values(batch * num_kv_heads * kv_seq * dv, 0xDEC_0DEC, 0.5);
        let tensors = SdpaTensors {
            q: &q,
            k: &k,
            v: &v,
            batch,
            num_heads,
            num_kv_heads,
            q_seq,
            kv_seq,
            head_size: dh,
            v_head_size: dv,
        };
        let cfg = SdpaConfig {
            scale: ScaleMode::PostDot(1.0 / (dh as f32).sqrt()),
            softcap: None,
            causal: true,
            past_seq: kv_seq - 1,
            causal_fill: f32::MIN,
        };
        let before = SDPA_NEON_DECODE_TEST_HITS.load(Ordering::Relaxed);
        let mut y = vec![f32::NAN; batch * num_heads * q_seq * dv];
        sdpa_f32(&tensors, &cfg, &NoBias, &NoMask, &mut y, None);
        let after = SDPA_NEON_DECODE_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            after > before,
            "sdpa_f32 did not take the inline NEON decode path for small per-head work"
        );
        assert!(y.iter().all(|v| v.is_finite()));
    }

    /// Numerics parity: inline NEON decode path vs Accelerate and scalar
    /// reference on a Qwen-0.5B-like decode shape (14 heads, 2 KV heads, Dh=64).
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "ios"),
        not(feature = "mlas")
    ))]
    #[test]
    fn sdpa_neon_decode_small_vs_accelerate_and_scalar_parity() {
        // Shape that hits the NEON decode path: kv_seq×head_size=64×64=4096 ≤ 8192
        let (batch, num_heads, num_kv_heads, q_seq, kv_seq, dh, dv) =
            (1usize, 14usize, 2usize, 1usize, 64usize, 64usize, 64usize);
        let q = deterministic_values(batch * num_heads * q_seq * dh, 0xA1_CE0A, 0.5);
        let k = deterministic_values(batch * num_kv_heads * kv_seq * dh, 0xA1_CE0B, 0.5);
        let v = deterministic_values(batch * num_kv_heads * kv_seq * dv, 0xA1_CE0C, 0.5);
        let tensors = SdpaTensors {
            q: &q,
            k: &k,
            v: &v,
            batch,
            num_heads,
            num_kv_heads,
            q_seq,
            kv_seq,
            head_size: dh,
            v_head_size: dv,
        };
        let cfg = SdpaConfig {
            scale: ScaleMode::PostDot(1.0 / (dh as f32).sqrt()),
            softcap: None,
            causal: true,
            past_seq: kv_seq - 1,
            causal_fill: f32::MIN,
        };
        let out_len = batch * num_heads * q_seq * dv;

        // Get NEON decode path result (via sdpa_f32 dispatch)
        let mut neon_out = vec![f32::NAN; out_len];
        sdpa_f32(&tensors, &cfg, &NoBias, &NoMask, &mut neon_out, None);

        // Get Accelerate path result (force it by calling directly)
        let mut accel_out = vec![f32::NAN; out_len];
        sdpa_f32_accelerate(&tensors, &cfg, &NoBias, &NoMask, &mut accel_out);

        // Get scalar reference
        let mut scalar_out = vec![f32::NAN; out_len];
        sdpa_f32_scalar(&tensors, &cfg, &NoBias, &NoMask, &mut scalar_out, None);

        // Get f64 reference
        let f64_ref = sdpa_f64_reference(&tensors, &cfg, &NoBias, &NoMask);

        // NEON path uses the same accumulation order as the scalar reference
        // (no GEMM reorder), so it should match scalar bit-for-bit or very
        // closely. Against f64, allow the same tolerance as the NEON parity
        // tests.
        let mut max_neon_scalar = 0.0f32;
        let mut max_neon_f64 = 0.0f32;
        let mut max_neon_accel = 0.0f32;
        for i in 0..out_len {
            max_neon_scalar = max_neon_scalar.max((neon_out[i] - scalar_out[i]).abs());
            max_neon_f64 = max_neon_f64.max((neon_out[i] - f64_ref[i]).abs());
            max_neon_accel = max_neon_accel.max((neon_out[i] - accel_out[i]).abs());
        }

        assert!(
            max_neon_scalar <= 1e-5,
            "NEON decode vs scalar max_abs={max_neon_scalar:e} (expected ≤ 1e-5)"
        );
        assert!(
            max_neon_f64 <= 1e-3,
            "NEON decode vs f64 max_abs={max_neon_f64:e} (expected ≤ 1e-3)"
        );
        // NEON vs Accelerate: both are float32, different accumulation order
        assert!(
            max_neon_accel <= 1e-4,
            "NEON decode vs Accelerate max_abs={max_neon_accel:e} (expected ≤ 1e-4)"
        );
    }

    /// Accelerate vs scalar parity at Whisper's encoder shape: 6 heads attend
    /// over 1500 frames with head_size=64. This exercises the threaded GEMM
    /// path at a scale far beyond LLM decode (M=1) or prefill (M~40).
    /// The Accelerate path parallelizes across (batch, head) tiles via Rayon,
    /// which alters accumulation order. We verify determinism across runs and
    /// check tolerance against the f64 reference.
    #[cfg(all(
        any(target_os = "macos", target_os = "ios"),
        not(feature = "mlas"),
        target_arch = "aarch64"
    ))]
    #[test]
    fn sdpa_accelerate_vs_scalar_parity_matrix() {
        use std::sync::atomic::Ordering;

        let (batch, num_heads, num_kv_heads, q_seq, kv_seq, dh, dv) = (
            1usize, 6usize, 6usize, 1500usize, 1500usize, 64usize, 64usize,
        );
        let q = deterministic_values(batch * num_heads * q_seq * dh, 0x1500_A1CE, 0.5);
        let k = deterministic_values(batch * num_kv_heads * kv_seq * dh, 0x1500_B0B0, 0.5);
        let v = deterministic_values(batch * num_kv_heads * kv_seq * dv, 0x1500_CAFE, 0.75);
        let tensors = SdpaTensors {
            q: &q,
            k: &k,
            v: &v,
            batch,
            num_heads,
            num_kv_heads,
            q_seq,
            kv_seq,
            head_size: dh,
            v_head_size: dv,
        };
        let cfg = SdpaConfig {
            scale: ScaleMode::PostDot(1.0 / (dh as f32).sqrt()),
            softcap: None,
            causal: false,
            past_seq: 0,
            causal_fill: f32::NEG_INFINITY,
        };

        // Verify the Accelerate path is reached.
        let before = SDPA_ACCELERATE_TEST_HITS.load(Ordering::Relaxed);
        let mut accelerate_out = vec![f32::NAN; batch * num_heads * q_seq * dv];
        sdpa_f32(&tensors, &cfg, &NoBias, &NoMask, &mut accelerate_out, None);
        let after = SDPA_ACCELERATE_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            after > before,
            "sdpa_f32 did not reach the Accelerate path for Whisper shape"
        );

        // Determinism: run again and confirm bit-exact (no thread-ordering drift).
        let mut accelerate_out2 = vec![f32::NAN; batch * num_heads * q_seq * dv];
        sdpa_f32(&tensors, &cfg, &NoBias, &NoMask, &mut accelerate_out2, None);
        assert_eq!(
            accelerate_out
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            accelerate_out2
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            "Accelerate SDPA is non-deterministic across runs at Whisper shape"
        );

        // Tolerance vs f64 reference — the Accelerate path uses GEMM (different
        // accumulation order from sequential scalar), so we compare against the
        // f64 oracle rather than insisting on scalar agreement.
        let f64_ref = sdpa_f64_reference(&tensors, &cfg, &NoBias, &NoMask);
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (&got, &expected) in accelerate_out.iter().zip(f64_ref.iter()) {
            assert!(got.is_finite(), "Accelerate produced non-finite output");
            let abs_err = (got - expected).abs();
            max_abs = max_abs.max(abs_err);
            max_rel = max_rel.max(abs_err / expected.abs().max(1e-6));
        }
        // At 1500x1500 with 64-dim heads, accumulation differences are larger
        // than decode shapes. The bounds catch real defects (dropped tiles,
        // transposed operands, wrong softmax normalization) while accommodating
        // GEMM vs sequential accumulation order differences. The relative bound
        // is loose because near-zero output values (where softmax concentrates
        // probability elsewhere) inflate the metric.
        assert!(
            max_abs <= 2e-3,
            "Accelerate vs f64 max_abs={max_abs:e} (Whisper 1500-frame shape)"
        );
        assert!(
            max_rel <= 5e-2,
            "Accelerate vs f64 max_rel={max_rel:e} (Whisper 1500-frame shape)"
        );

        // Also verify against scalar reference with a looser tolerance (both are
        // valid float orderings, but neither is authoritative over the other).
        let mut scalar_out = vec![f32::NAN; batch * num_heads * q_seq * dv];
        sdpa_f32_scalar(&tensors, &cfg, &NoBias, &NoMask, &mut scalar_out, None);
        let mut max_scalar_abs = 0.0f32;
        for (&accel, &scalar) in accelerate_out.iter().zip(scalar_out.iter()) {
            max_scalar_abs = max_scalar_abs.max((accel - scalar).abs());
        }
        assert!(
            max_scalar_abs <= 5e-3,
            "Accelerate vs scalar max_abs={max_scalar_abs:e} (Whisper 1500-frame shape)"
        );
    }

    /// Deterministic pseudo-random f32 fill in `[-1, 1)` for parity fixtures.
    #[cfg(feature = "mlas")]
    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                s ^= s >> 30;
                s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                s ^= s >> 27;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// A dense additive key mask driven from a `[batch, q, kv]` buffer, used to
    /// exercise the fast path's per-row mask application.
    #[cfg(feature = "mlas")]
    struct DenseKeyMask<'a> {
        data: &'a [f32],
        q_seq: usize,
        kv_seq: usize,
    }
    #[cfg(feature = "mlas")]
    impl KeyMask for DenseKeyMask<'_> {
        fn at(&self, b: usize, i: usize, j: usize) -> f32 {
            self.data[(b * self.q_seq + i) * self.kv_seq + j]
        }
    }

    /// The MLAS-GEMM fast path must agree with the scalar reference to tight
    /// tolerance across the full mode matrix (GQA, scale placement, softcap,
    /// bias, mask, causal, decode `Sq=1` and prefill shapes). GEMM reorders the
    /// accumulation, so this is a tolerance — not byte — check.
    #[cfg(feature = "mlas")]
    #[test]
    fn fast_path_matches_scalar_reference() {
        struct Shape {
            name: &'static str,
            batch: usize,
            nq: usize,
            nkv: usize,
            sq: usize,
            tk: usize,
            dh: usize,
            dv: usize,
            causal: bool,
            past: usize,
            softcap: Option<f32>,
            split_sqrt: bool,
            with_bias: bool,
            with_mask: bool,
        }
        let shapes = [
            Shape {
                name: "mha-prefill",
                batch: 2,
                nq: 4,
                nkv: 4,
                sq: 7,
                tk: 7,
                dh: 8,
                dv: 8,
                causal: false,
                past: 0,
                softcap: None,
                split_sqrt: false,
                with_bias: false,
                with_mask: false,
            },
            Shape {
                name: "mha-causal",
                batch: 1,
                nq: 3,
                nkv: 3,
                sq: 6,
                tk: 6,
                dh: 5,
                dv: 5,
                causal: true,
                past: 0,
                softcap: None,
                split_sqrt: false,
                with_bias: false,
                with_mask: false,
            },
            Shape {
                name: "gqa",
                batch: 2,
                nq: 8,
                nkv: 2,
                sq: 5,
                tk: 5,
                dh: 4,
                dv: 4,
                causal: false,
                past: 0,
                softcap: None,
                split_sqrt: false,
                with_bias: false,
                with_mask: false,
            },
            Shape {
                name: "mqa-decode",
                batch: 2,
                nq: 6,
                nkv: 1,
                sq: 1,
                tk: 9,
                dh: 8,
                dv: 8,
                causal: false,
                past: 8,
                softcap: None,
                split_sqrt: false,
                with_bias: false,
                with_mask: false,
            },
            Shape {
                name: "cross-diff-dv",
                batch: 1,
                nq: 2,
                nkv: 2,
                sq: 4,
                tk: 6,
                dh: 5,
                dv: 3,
                causal: false,
                past: 0,
                softcap: None,
                split_sqrt: false,
                with_bias: true,
                with_mask: false,
            },
            Shape {
                name: "softcap",
                batch: 1,
                nq: 2,
                nkv: 2,
                sq: 5,
                tk: 5,
                dh: 6,
                dv: 6,
                causal: false,
                past: 0,
                softcap: Some(30.0),
                split_sqrt: false,
                with_bias: false,
                with_mask: false,
            },
            Shape {
                name: "split-sqrt-mask",
                batch: 2,
                nq: 3,
                nkv: 3,
                sq: 4,
                tk: 5,
                dh: 7,
                dv: 7,
                causal: false,
                past: 0,
                softcap: None,
                split_sqrt: true,
                with_bias: false,
                with_mask: true,
            },
            Shape {
                name: "causal-past-decode",
                batch: 1,
                nq: 4,
                nkv: 4,
                sq: 1,
                tk: 12,
                dh: 8,
                dv: 8,
                causal: true,
                past: 11,
                softcap: None,
                split_sqrt: false,
                with_bias: false,
                with_mask: false,
            },
        ];

        for sh in &shapes {
            let q = fill(sh.batch * sh.nq * sh.sq * sh.dh, 1 + sh.sq as u64);
            let k = fill(sh.batch * sh.nkv * sh.tk * sh.dh, 2 + sh.tk as u64);
            let v = fill(sh.batch * sh.nkv * sh.tk * sh.dv, 3 + sh.dv as u64);
            let scale = 1.0 / (sh.dh as f32).sqrt();
            let t = SdpaTensors {
                q: &q,
                k: &k,
                v: &v,
                batch: sh.batch,
                num_heads: sh.nq,
                num_kv_heads: sh.nkv,
                q_seq: sh.sq,
                kv_seq: sh.tk,
                head_size: sh.dh,
                v_head_size: sh.dv,
            };
            let cfg = SdpaConfig {
                scale: if sh.split_sqrt {
                    ScaleMode::SplitSqrt(scale)
                } else {
                    ScaleMode::PostDot(scale)
                },
                softcap: sh.softcap,
                causal: sh.causal,
                past_seq: sh.past,
                causal_fill: f32::MIN,
            };
            let bias_data = fill(sh.batch * sh.nq * sh.sq * sh.tk, 7);
            let mask_data: Vec<f32> = fill(sh.batch * sh.sq * sh.tk, 9)
                .into_iter()
                .map(|x| if x < -0.5 { -1.0e9 } else { 0.0 })
                .collect();
            let no_bias = NoBias;
            let bc_bias = BroadcastBias::new(&bias_data, [sh.batch, sh.nq, sh.sq, sh.tk]);
            let bias: &dyn AttnBias = if sh.with_bias { &bc_bias } else { &no_bias };
            let no_mask = NoMask;
            let dm = DenseKeyMask {
                data: &mask_data,
                q_seq: sh.sq,
                kv_seq: sh.tk,
            };
            let mask: &dyn KeyMask = if sh.with_mask { &dm } else { &no_mask };

            let out_len = sh.batch * sh.nq * sh.sq * sh.dv;
            let mut y_scalar = vec![0.0f32; out_len];
            sdpa_f32_scalar(&t, &cfg, bias, mask, &mut y_scalar, None);
            let mut y_fast = vec![0.0f32; out_len];
            sdpa_f32_fast(&t, &cfg, bias, mask, &mut y_fast);

            let mut max_abs = 0.0f32;
            let mut worst = 0.0f32;
            for (a, b) in y_fast.iter().zip(y_scalar.iter()) {
                let abs = (a - b).abs();
                max_abs = max_abs.max(abs);
                // Combined tolerance `atol + rtol·|ref|` (numpy allclose style),
                // so a near-zero reference doesn't inflate a pure relative ratio.
                worst = worst.max(abs - (1e-5 + 1e-4 * b.abs()));
            }
            // GEMM reassociation over these small K (≤8) reduces the f32 dot to
            // a few ULP; softmax + P·V keep it bounded. atol 1e-5 / rtol 1e-4
            // matches the crate's ORT-parity tolerances with margin.
            assert!(
                worst <= 0.0,
                "shape {}: fast vs scalar exceeds atol+rtol (max_abs={max_abs:e})",
                sh.name
            );
        }
    }

    /// Provisional fast-vs-scalar throughput probe (run with
    /// `cargo test -p onnx-runtime-ep-cpu --features mlas -- --ignored --nocapture
    /// sdpa_fast_provisional_bench`). Numbers are PROVISIONAL — the CI host is
    /// shared, so treat the printed speedups as indicative, not authoritative.
    #[cfg(feature = "mlas")]
    #[test]
    #[ignore = "provisional microbench; shared host — run manually with --nocapture"]
    fn sdpa_fast_provisional_bench() {
        use std::time::Instant;

        fn run(name: &str, batch: usize, nq: usize, nkv: usize, sq: usize, tk: usize, dh: usize) {
            let q = fill(batch * nq * sq * dh, 11);
            let k = fill(batch * nkv * tk * dh, 22);
            let v = fill(batch * nkv * tk * dh, 33);
            let t = SdpaTensors {
                q: &q,
                k: &k,
                v: &v,
                batch,
                num_heads: nq,
                num_kv_heads: nkv,
                q_seq: sq,
                kv_seq: tk,
                head_size: dh,
                v_head_size: dh,
            };
            let cfg = SdpaConfig {
                scale: ScaleMode::PostDot(1.0 / (dh as f32).sqrt()),
                softcap: None,
                causal: sq > 1,
                past_seq: tk - sq,
                causal_fill: f32::MIN,
            };
            let out_len = batch * nq * sq * dh;
            let mut y = vec![0.0f32; out_len];

            let iters = 20;
            // Warm up + time scalar.
            sdpa_f32_scalar(&t, &cfg, &NoBias, &NoMask, &mut y, None);
            let t0 = Instant::now();
            for _ in 0..iters {
                sdpa_f32_scalar(&t, &cfg, &NoBias, &NoMask, &mut y, None);
            }
            let scalar = t0.elapsed().as_secs_f64() / iters as f64;
            // Warm up + time fast.
            sdpa_f32_fast(&t, &cfg, &NoBias, &NoMask, &mut y);
            let t1 = Instant::now();
            for _ in 0..iters {
                sdpa_f32_fast(&t, &cfg, &NoBias, &NoMask, &mut y);
            }
            let fast = t1.elapsed().as_secs_f64() / iters as f64;
            println!(
                "[sdpa-bench PROVISIONAL] {name:>16}: scalar {:>9.3} ms  fast {:>9.3} ms  speedup {:>5.2}x",
                scalar * 1e3,
                fast * 1e3,
                scalar / fast
            );
        }

        println!("[sdpa-bench] PROVISIONAL numbers — shared host, treat as indicative only");
        run("prefill", 1, 32, 32, 512, 512, 128);
        run("decode", 1, 32, 32, 1, 513, 128);
        run("gqa-prefill", 1, 32, 8, 512, 512, 128);
    }

    /// A hook that contributes nothing but refuses to advertise itself as an
    /// identity, forcing the fast path's general per-element epilogue. The
    /// specialized branch must agree with it bit for bit.
    struct OpaqueZeroBias;
    impl AttnBias for OpaqueZeroBias {
        fn at(&self, _b: usize, _head: usize, _i: usize, _j: usize) -> f32 {
            0.0
        }
    }
    struct OpaqueZeroMask;
    impl KeyMask for OpaqueZeroMask {
        fn at(&self, _b: usize, _i: usize, _j: usize) -> f32 {
            0.0
        }
    }

    #[test]
    fn identity_hook_specialization_matches_the_general_epilogue() {
        // (batch, heads, kv_heads, q_seq, kv_seq, head_size, v_head_size)
        let shapes = [
            (2usize, 8usize, 8usize, 5usize, 37usize, 64usize, 64usize),
            (1, 12, 4, 128, 128, 64, 64),
            (3, 6, 3, 7, 71, 33, 17),
            (1, 4, 1, 1, 129, 80, 80),
        ];
        for (si, &(batch, heads, kv_heads, q_seq, kv_seq, dh, dv)) in shapes.iter().enumerate() {
            let q = deterministic_values(batch * heads * q_seq * dh, 0x00A1_0000 + si as u64, 0.8);
            let k =
                deterministic_values(batch * kv_heads * kv_seq * dh, 0x00A1_1000 + si as u64, 0.8);
            let v =
                deterministic_values(batch * kv_heads * kv_seq * dv, 0x00A1_2000 + si as u64, 0.8);
            let tensors = SdpaTensors {
                q: &q,
                k: &k,
                v: &v,
                batch,
                num_heads: heads,
                num_kv_heads: kv_heads,
                q_seq,
                kv_seq,
                head_size: dh,
                v_head_size: dv,
            };
            for causal in [false, true] {
                let cfg = SdpaConfig {
                    scale: ScaleMode::PostDot(1.0 / (dh as f32).sqrt()),
                    softcap: None,
                    causal,
                    past_seq: kv_seq.saturating_sub(q_seq),
                    causal_fill: f32::MIN,
                };
                // Prefilled with NaN rather than zero. A dropped write is then
                // unambiguous: it survives as a NaN, and NaN != NaN makes the
                // comparison below fail loudly at the offending element. With
                // zero prefill an unwritten element is indistinguishable from a
                // legitimate `0.0`, and a hole that lands in the same place in
                // both runs cancels out and passes silently — which is how
                // #1685 stayed invisible here while it was live.
                let mut fast = vec![f32::NAN; batch * heads * q_seq * dv];
                let mut general = vec![f32::NAN; batch * heads * q_seq * dv];
                sdpa_f32(&tensors, &cfg, &NoBias, &NoMask, &mut fast, None);
                sdpa_f32(
                    &tensors,
                    &cfg,
                    &OpaqueZeroBias,
                    &OpaqueZeroMask,
                    &mut general,
                    None,
                );
                for (route, out) in [("identity-specialized", &fast), ("general", &general)] {
                    if let Some(idx) = out.iter().position(|x| x.is_nan()) {
                        let unwritten = out.iter().filter(|x| x.is_nan()).count();
                        let tile = idx / (q_seq * dv);
                        let row = (idx % (q_seq * dv)) / dv;
                        panic!(
                            "shape {si} causal={causal}: the {route} route left {unwritten} of \
                             {} output elements unwritten; first at index {idx} = \
                             (tile {tile}, row {row}, column {}). Contiguous runs of \
                             `v_head_size` starting on a row boundary mean a dropped GEMM row \
                             partition (see #1685).",
                            out.len(),
                            idx % dv,
                        );
                    }
                }
                assert_eq!(
                    fast, general,
                    "shape {si} causal={causal}: identity specialization diverged"
                );
            }
        }
    }

    #[test]
    fn fully_masked_rows_stay_zero_when_the_fill_is_negative_infinity() {
        // A row that is entirely `-inf` has no defined softmax; the kernel's
        // convention (shared with the scalar oracle and ORT) is to emit zeros.
        // MLAS has no such convention, so this exercises the guarded fallback.
        struct MaskEverySecondRow;
        impl KeyMask for MaskEverySecondRow {
            fn at(&self, _b: usize, i: usize, _j: usize) -> f32 {
                if i % 2 == 1 { f32::NEG_INFINITY } else { 0.0 }
            }
        }

        let (batch, heads, q_seq, kv_seq, dh) = (2usize, 4usize, 6usize, 13usize, 32usize);
        let q = deterministic_values(batch * heads * q_seq * dh, 0x00B2_0000, 0.7);
        let k = deterministic_values(batch * heads * kv_seq * dh, 0x00B2_1000, 0.7);
        let v = deterministic_values(batch * heads * kv_seq * dh, 0x00B2_2000, 0.7);
        let tensors = SdpaTensors {
            q: &q,
            k: &k,
            v: &v,
            batch,
            num_heads: heads,
            num_kv_heads: heads,
            q_seq,
            kv_seq,
            head_size: dh,
            v_head_size: dh,
        };
        let cfg = SdpaConfig {
            scale: ScaleMode::PostDot(1.0 / (dh as f32).sqrt()),
            softcap: None,
            causal: false,
            past_seq: 0,
            causal_fill: f32::NEG_INFINITY,
        };
        let mut y = vec![f32::NAN; batch * heads * q_seq * dh];
        sdpa_f32(&tensors, &cfg, &NoBias, &MaskEverySecondRow, &mut y, None);
        for b in 0..batch {
            for h in 0..heads {
                for i in 0..q_seq {
                    let row = &y[(((b * heads + h) * q_seq) + i) * dh..][..dh];
                    assert!(
                        row.iter().all(|x| x.is_finite()),
                        "b{b} h{h} row {i} not finite"
                    );
                    if i % 2 == 1 {
                        assert!(
                            row.iter().all(|x| *x == 0.0),
                            "b{b} h{h} row {i} should be zeroed, got {row:?}"
                        );
                    }
                }
            }
        }
    }
}
