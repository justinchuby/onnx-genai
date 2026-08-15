//! Two things the #759 dummy-page design turns on that are *not* CUDA calls and
//! so are proven here on the CPU, deterministically: which fill value is
//! correctness-safe, and where the fixed-stride+dummy scheme's committed bytes
//! cross bucket growth's.
//!
//! ## Fill choice — masking decides it, and it forbids NaN
//!
//! An earlier draft of this probe filled the dummy tail with a NaN "sentinel"
//! so a stray read would be loud. That is **wrong in the case that matters**.
//! #721 stage 4 recorded that the decode kernel reads the *full padded shape*
//! and relies on **additive masking** (score forced to `-inf` for positions
//! `>= live_len`) for correctness. NaN defeats additive masking:
//! `q · NaN = NaN`, `NaN + (-inf) = NaN`, `exp(NaN) = NaN` — so a NaN dummy
//! poisons the softmax output that masking would otherwise have made correct,
//! and it poisons the *live* positions too, not just the masked ones. A NaN
//! fill would turn a working masked kernel into a broken one.
//!
//! [`masking_determines_the_safe_dummy_fill`] measures the rule:
//! * masking effective -> a finite fill (zeros) is annihilated; **zeros are the
//!   right choice**, NaN is forbidden;
//! * masking not effective -> even zeros are wrong (a zero key scores `0`, which
//!   softmax weights `exp(0) = 1` — a full-strength contribution), so *no* fill
//!   is correctness-safe and the dummy page is a fault-avoidance device only,
//!   requiring the kernel to be bounded.
//!
//! Whether *our production decode kernel* actually masks those tail positions is
//! an execution-provider fact this memory-crate probe cannot exercise (there is
//! no attention kernel here). It must be verified in the EP before zeros are
//! trusted for correctness; until then the dummy page is crash-safe only.
//!
//! ## Crossover — when does fixed-stride+dummy beat bucket growth?
//!
//! A shared dummy tail changes the KV requirement from "must be really
//! committed" to "must be addressable", so a permanently-stable full-context
//! stride (which never re-captures a graph on growth) no longer has to honestly
//! commit the whole padded shape. Its committed cost becomes the
//! `objects x granule` floor plus live content. That floor is a **constant**
//! the dummy page does *not* remove (those granules hold real, distinct live
//! data), so it amortizes: below a crossover length fixed-stride+dummy commits
//! more than bucket growth, above it the same or less — and never re-captures.
//!
//! [`fixed_stride_plus_dummy_crossover_vs_bucket_growth`] computes the crossover
//! from the real KV binding geometry of qwen2.5-0.5b and qwen14b (read from
//! their `genai_config.json`), with the closed form
//! `crossover ~= objects x granule / kv_bytes_per_token = granule / (head_dim x elem)`.
//!
//! ## The crossover is model-size independent, and the granule is its lever
//!
//! Because `objects = layers x 2 x kv_heads` cancels the identical factor in
//! `kv_bytes_per_token = objects x head_dim x elem`, the crossover reduces to
//! `granule / (head_dim x elem)` — it depends only on the granule, the head
//! dimension and the KV dtype, *never on model size*.
//! [`crossover_is_model_size_independent`] verifies the cancellation against a
//! toy 480x size range and the two real models, and
//! [`granularity_is_the_crossover_lever`] publishes the crossover as a table
//! over {MINIMUM, RECOMMENDED} granule x head_dim x KV dtype. The consequence:
//! at the 2 MiB RECOMMENDED granule the crossover is 8K-16K tokens (a high bar),
//! and a 64 KiB minimum granule *would* collapse it ~32x into the low hundreds —
//! **but `vmm_granularity_gpu` measured this device's MINIMUM granule at 2 MiB,
//! equal to RECOMMENDED, so that lever is unavailable here.** The 64 KiB row is
//! therefore a counterfactual: it shows the win a finer-granule device would
//! get, not one this box can realize.

/// The VMM allocation granule measured on this hardware by the sibling GPU
/// tests (`vmm_graph_remap_gpu`, `vmm_kv_contiguous_tail_gpu`): 2 MiB. This is
/// CUDA's `CU_MEM_ALLOC_GRANULARITY_RECOMMENDED` **and, as `vmm_granularity_gpu`
/// measured on this device, its `MINIMUM` too** — the two are equal at 2 MiB
/// here, so the granule cannot be reduced on this box. The crossover scales
/// linearly with the granule actually mapped at, so
/// [`crossover_is_model_size_independent`] and
/// [`granularity_is_the_crossover_lever`] also evaluate a hypothetical finer
/// 64 KiB granule ([`MIN_GRANULE`]) to show what a device that *offered* it
/// would gain.
const GRANULE: usize = 2 * 1024 * 1024;

/// A hypothetical finer VMM granule. **Measured reality on this device:
/// `CU_MEM_ALLOC_GRANULARITY_MINIMUM` == `RECOMMENDED` == 2 MiB** (printed by
/// `vmm_granularity_gpu::report_both_allocation_granularities`), so the finer
/// granule the owner hoped would pull the crossover down ~32x is **not available
/// here**. 64 KiB is kept only to publish what the crossover *would* be on a
/// device whose minimum granule is 64 KiB — a counterfactual, not this box.
const MIN_GRANULE: usize = 64 * 1024;

/// KV cache geometry taken verbatim from a model's `genai_config.json`.
struct KvGeometry {
    name: &'static str,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    elem_bytes: usize,
    context_length: usize,
}

impl KvGeometry {
    /// Granule-floor units under a head-major fixed full-context stride: each
    /// `(layer, key|value, kv_head)` head-stripe lands in its own granule, so
    /// the near-empty commit floor is one granule per unit.
    fn objects(&self) -> usize {
        self.num_hidden_layers * 2 * self.num_key_value_heads
    }

    /// Bytes of KV a single token adds across every binding and head.
    fn kv_bytes_per_token(&self) -> usize {
        self.objects() * self.head_dim * self.elem_bytes
    }

    /// One head's full-context stripe. When this exceeds a granule the head has
    /// an uncommitted tail the dummy page can back; when it equals a granule
    /// there is no per-head tail at all and the dummy page saves no memory.
    fn per_head_stride(&self) -> usize {
        self.context_length * self.head_dim * self.elem_bytes
    }

    /// `objects x granule` — committed floor of the fixed-stride+dummy scheme.
    fn floor_bytes(&self) -> usize {
        self.objects() * GRANULE
    }

    /// Honest full-context commit (no dummy): every head commits its whole
    /// stripe, rounded up to granules.
    fn honest_full_bytes(&self) -> usize {
        self.objects() * self.per_head_stride().div_ceil(GRANULE) * GRANULE
    }

    /// Bucket growth's committed bytes at full context ~= live content.
    fn bucket_full_bytes(&self) -> usize {
        self.kv_bytes_per_token() * self.context_length
    }

    /// Tokens at which fixed-stride+dummy's constant floor is matched by bucket
    /// growth's live content, at the 2 MiB RECOMMENDED granule.
    fn crossover_tokens(&self) -> usize {
        self.floor_bytes() / self.kv_bytes_per_token()
    }

    /// The same crossover at an arbitrary granule. The closed form
    /// `objects x granule / kv_bytes_per_token` reduces to
    /// `granule / (head_dim x elem_bytes)` because `objects` cancels the
    /// `objects` inside `kv_bytes_per_token` — so it depends only on the granule,
    /// the head dimension and the KV element size, never on `layers` or
    /// `kv_heads`, i.e. never on model size.
    fn crossover_tokens_at(&self, granule: usize) -> usize {
        (self.objects() * granule) / self.kv_bytes_per_token()
    }
}

fn qwen2_5_0_5b() -> KvGeometry {
    // models/qwen05b-fresh/genai_config.json
    KvGeometry {
        name: "qwen2.5-0.5b",
        num_hidden_layers: 24,
        num_key_value_heads: 2,
        head_dim: 64,
        elem_bytes: 2, // f16
        context_length: 32768,
    }
}

fn qwen14b() -> KvGeometry {
    // models/qwen14b-zp/genai_config.json (the #721 stage-4 1.5 GB model)
    KvGeometry {
        name: "qwen14b",
        num_hidden_layers: 48,
        num_key_value_heads: 8,
        head_dim: 128,
        elem_bytes: 2, // f16; kv_bytes_per_token = 196608
        context_length: 8192,
    }
}

/// Softmax over `scores + mask`, NaN-honest (mirrors a real attention kernel's
/// additive mask then softmax). Returns per-position weights.
fn softmax_with_additive_mask(scores: &[f32], mask: &[f32]) -> Vec<f32> {
    let z: Vec<f32> = scores.iter().zip(mask).map(|(s, m)| s + m).collect();
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = z.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Weighted sum of scalar "value" entries by softmax weights — the attention
/// output in miniature.
fn attention_output(weights: &[f32], values: &[f32]) -> f32 {
    weights.iter().zip(values).map(|(w, v)| w * v).sum()
}

/// The rule that picks the dummy fill: masking decides, and it forbids NaN.
/// One live position (value 1.0) and one dummy tail position; we vary the
/// tail's fill and whether the tail is masked, and measure the attention
/// output.
#[test]
fn masking_determines_the_safe_dummy_fill() {
    let live_score = 2.0_f32;
    // A read of a masked position still produces some score; what matters is the
    // mask, so the tail's own score is a stand-in for `q · k_tail`.
    let finite_tail_score = 0.0_f32; // a zero key scores q·0 = 0
    let nan_tail_score = f32::NAN; // a NaN key scores q·NaN = NaN

    let masked = [0.0_f32, f32::NEG_INFINITY]; // live kept, tail masked
    let unmasked = [0.0_f32, 0.0_f32]; // both kept (masking not effective)

    // 1. Masking effective + finite (zero) tail: the tail is annihilated and the
    //    output is exactly the live value. Zeros are safe under masking.
    let w = softmax_with_additive_mask(&[live_score, finite_tail_score], &masked);
    let out_zero_masked = attention_output(&w, &[1.0, 0.0]);
    assert!(
        w[1] == 0.0 && (out_zero_masked - 1.0).abs() < 1e-6,
        "with masking, a zero-filled dummy tail contributes nothing (weight {}, output {})",
        w[1],
        out_zero_masked
    );

    // 2. Masking effective + NaN tail: NaN defeats the mask and poisons the whole
    //    output, including the live position. NaN is forbidden.
    let w_nan = softmax_with_additive_mask(&[live_score, nan_tail_score], &masked);
    let out_nan_masked = attention_output(&w_nan, &[1.0, f32::NAN]);
    assert!(
        out_nan_masked.is_nan(),
        "a NaN-filled dummy tail defeats additive masking and poisons the output ({out_nan_masked}); \
         NaN must never be the fill"
    );

    // 3. Masking NOT effective + zero tail: the zero key is weighted exp(0)=1, a
    //    full-strength contribution that dilutes the live output. Without masking
    //    even zeros are wrong, so no fill is correctness-safe.
    let w_unmasked = softmax_with_additive_mask(&[live_score, finite_tail_score], &unmasked);
    let out_zero_unmasked = attention_output(&w_unmasked, &[1.0, 0.0]);
    assert!(
        w_unmasked[1] > 0.0 && out_zero_unmasked < 1.0 - 1e-6,
        "without masking, a zero dummy key still gets softmax weight exp(0)=1 and dilutes the \
         output (weight {}, output {})",
        w_unmasked[1],
        out_zero_unmasked
    );

    eprintln!(
        "Fill rule (measured): masking effective -> zeros safe (output {out_zero_masked:.4}), \
         NaN forbidden (output {out_nan_masked}); masking not effective -> zeros wrong \
         (output {out_zero_unmasked:.4}). Choose ZERO when the decode kernel's tail masking is \
         verified in the EP; never NaN; if masking is unverified the dummy page is crash-safe only."
    );
}

/// The crossover between fixed-stride+dummy and bucket growth, computed from the
/// real KV geometry of both models. Prints the numbers and asserts the closed
/// forms so the recommendation rests on derived, not guessed, figures.
#[test]
fn fixed_stride_plus_dummy_crossover_vs_bucket_growth() {
    for geo in [qwen2_5_0_5b(), qwen14b()] {
        let objects = geo.objects();
        let kv_per_token = geo.kv_bytes_per_token();
        let floor = geo.floor_bytes();
        let honest_full = geo.honest_full_bytes();
        let bucket_full = geo.bucket_full_bytes();
        let crossover = geo.crossover_tokens();
        let granules_per_head = geo.per_head_stride() / GRANULE;

        eprintln!(
            "\n{}: objects = {objects} (layers {} x 2 x kv_heads {}); kv_bytes/token = {kv_per_token}; \
             per-head stride = {} MiB ({granules_per_head} granule(s)); \
             floor = {} MiB; honest full commit = {} MiB; bucket growth @ {} tok = {} MiB.\n  \
             crossover = objects x granule / kv_bytes_per_token = {crossover} tokens \
             (= granule / (head_dim x elem) = {} / {}). Context length = {}. \
             Below {crossover} tok fixed-stride+dummy commits MORE than bucket growth; \
             at/above it, the same-or-less AND never re-captures.",
            geo.name,
            geo.num_hidden_layers,
            geo.num_key_value_heads,
            geo.per_head_stride() / (1024 * 1024),
            floor / (1024 * 1024),
            honest_full / (1024 * 1024),
            geo.context_length,
            bucket_full / (1024 * 1024),
            GRANULE,
            geo.head_dim * geo.elem_bytes,
            geo.context_length,
        );

        // Closed form: crossover is independent of context length.
        assert_eq!(crossover, GRANULE / (geo.head_dim * geo.elem_bytes));
    }

    // qwen2.5-0.5b: per-head stride is 2 granules, so the dummy page leaves a
    // real per-head tail; crossover lands at half the 32768 context.
    let small = qwen2_5_0_5b();
    assert_eq!(small.objects(), 96);
    assert_eq!(small.floor_bytes(), 192 * 1024 * 1024);
    assert_eq!(small.crossover_tokens(), 16384);
    assert_eq!(small.crossover_tokens() * 2, small.context_length);

    // qwen14b: per-head stride is EXACTLY one granule, so there is no per-head
    // uncommitted tail -- the dummy page saves no memory, the floor already
    // equals the honest full commit, and the crossover is the full context. Its
    // only benefit here is stride stability (no re-capture on growth).
    let big = qwen14b();
    assert_eq!(big.objects(), 768);
    assert_eq!(big.floor_bytes(), 1536 * 1024 * 1024);
    assert_eq!(big.kv_bytes_per_token(), 196608);
    assert_eq!(
        big.per_head_stride(),
        GRANULE,
        "14b head-stripe is exactly one granule"
    );
    assert_eq!(
        big.floor_bytes(),
        big.honest_full_bytes(),
        "14b: dummy tail saves no memory"
    );
    assert_eq!(big.crossover_tokens(), 8192);
    assert_eq!(big.crossover_tokens(), big.context_length);
}

/// The owner's key derivation, verified: the crossover is **model-size
/// independent**. `objects = layers x 2 x kv_heads` cancels the identical factor
/// inside `kv_bytes_per_token = objects x head_dim x elem`, leaving
/// `crossover = granule / (head_dim x elem)`. So two models with wildly
/// different layer and head counts but the same head_dim and KV dtype share a
/// crossover — it is a property of the *layout unit*, not the model.
#[test]
fn crossover_is_model_size_independent() {
    // A deliberately tiny model and a huge one, same head_dim (64) and fp16.
    let tiny = KvGeometry {
        name: "tiny",
        num_hidden_layers: 4,
        num_key_value_heads: 1,
        head_dim: 64,
        elem_bytes: 2,
        context_length: 4096,
    };
    let huge = KvGeometry {
        name: "huge",
        num_hidden_layers: 120,
        num_key_value_heads: 16,
        head_dim: 64,
        elem_bytes: 2,
        context_length: 131072,
    };

    // Despite 480x the objects, identical crossover at any granule.
    assert_eq!(tiny.objects() * 480, huge.objects());
    for granule in [MIN_GRANULE, GRANULE] {
        let expected = granule / (64 * 2);
        assert_eq!(tiny.crossover_tokens_at(granule), expected);
        assert_eq!(huge.crossover_tokens_at(granule), expected);
        assert_eq!(
            tiny.crossover_tokens_at(granule),
            huge.crossover_tokens_at(granule),
            "layers and kv_heads cancel; crossover depends only on granule/(head_dim x elem)"
        );
    }

    // The cancellation is exact for the two real models too: qwen2.5-0.5b
    // (head_dim 64) and qwen14b (head_dim 128) differ in crossover ONLY because
    // their head_dim differs, not because one is 28x the other in size.
    assert_eq!(
        qwen2_5_0_5b().crossover_tokens_at(GRANULE),
        GRANULE / (64 * 2)
    );
    assert_eq!(qwen14b().crossover_tokens_at(GRANULE), GRANULE / (128 * 2));

    eprintln!(
        "Crossover is model-size independent: crossover = granule / (head_dim x elem). \
         layers and kv_heads cancel (tiny objects {} vs huge objects {} -> same crossover).",
        tiny.objects(),
        huge.objects()
    );
}

/// The granule is the crossover lever. Publish the crossover as a table over
/// {MINIMUM, RECOMMENDED} granule x {head_dim 64, 128} x {fp16, fp8/int8}, so it
/// can be applied to any model without re-measuring. Quantized KV halves bytes
/// per token, which *doubles* the crossover in tokens.
#[test]
fn granularity_is_the_crossover_lever() {
    let crossover = |granule: usize, head_dim: usize, elem: usize| granule / (head_dim * elem);

    eprintln!(
        "\nCrossover table (tokens) = granule / (head_dim x elem_bytes), model-size independent:\n  \
         granule    | hd64 fp16 | hd64 fp8 | hd128 fp16 | hd128 fp8\n  \
         MIN  {:>4} KiB | {:>9} | {:>8} | {:>10} | {:>9}\n  \
         REC  {:>4} KiB | {:>9} | {:>8} | {:>10} | {:>9}\n  \
         (fp8/int8 halves bytes/token, so it doubles the crossover in tokens.)",
        MIN_GRANULE / 1024,
        crossover(MIN_GRANULE, 64, 2),
        crossover(MIN_GRANULE, 64, 1),
        crossover(MIN_GRANULE, 128, 2),
        crossover(MIN_GRANULE, 128, 1),
        GRANULE / 1024,
        crossover(GRANULE, 64, 2),
        crossover(GRANULE, 64, 1),
        crossover(GRANULE, 128, 2),
        crossover(GRANULE, 128, 1),
    );

    // At the RECOMMENDED 2 MiB granule the crossover is 8K-16K tokens: a high
    // bar that loses in most serving windows. (On THIS device the MINIMUM
    // granule is also 2 MiB -- see vmm_granularity_gpu -- so the coarse row is
    // the only one realizable here; the fine row below is a counterfactual.)
    assert_eq!(crossover(GRANULE, 64, 2), 16384);
    assert_eq!(crossover(GRANULE, 128, 2), 8192);

    // At a 64 KiB MINIMUM granule the crossover WOULD collapse ~32x, into the
    // low hundreds -- fixed-stride+dummy would then win in essentially every
    // realistic context. This device does not expose a granule that fine, so
    // this is what a finer-granule device would gain, not this box.
    assert_eq!(crossover(MIN_GRANULE, 64, 2), 512);
    assert_eq!(crossover(MIN_GRANULE, 128, 2), 256);
    assert_eq!(GRANULE / MIN_GRANULE, 32);
    assert_eq!(
        crossover(GRANULE, 128, 2) / crossover(MIN_GRANULE, 128, 2),
        32,
        "the crossover scales linearly with the granule"
    );

    // Quantized KV doubles the crossover (halved bytes/token), tempering the
    // win at the fine granule but not reversing it.
    assert_eq!(crossover(MIN_GRANULE, 128, 1), 512);
    assert_eq!(
        crossover(MIN_GRANULE, 128, 1),
        2 * crossover(MIN_GRANULE, 128, 2)
    );
}
