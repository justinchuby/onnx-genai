//! Equivalence and rejection tests for the `PagedAttention` validator, backend
//! gate, and CPU oracle.
//!
//! Coverage (task slice-2 test matrix):
//! * schema accept/reject for `block_size`, LATENT constraints, the scale trap,
//!   invalid block tables / sequence-length tensors;
//! * backend gate typed `NOT_IMPLEMENTED` for every unsupported optional mode
//!   (both the WebGPU-SEPARATE and GLM-dense-MLA-LATENT subsets);
//! * paged == contiguous: SEPARATE GQA oracle vs a naive dense reference, with
//!   partial RoPE;
//! * paging placement-invariance across a permuted block table (multi-block);
//! * `slot_mapping = -1` suppresses the cache write;
//! * absorbed LATENT MLA == decomposed multi-head latent attention (the core
//!   applicability claim), at both tiny and DeepSeek-V3 576/512/64 dims;
//! * head_sink smooth-softmax and q/k RMSNorm paths.

#![allow(clippy::too_many_arguments)]

use onnx_genai_paged_attention::backend::{NativeSubset, check_backend_support};
use onnx_genai_paged_attention::oracle::{PagedAttentionData, paged_attention_reference};
use onnx_genai_paged_attention::params::{PagedAttentionAttributes, PagedAttentionInputs, Shape};
use onnx_genai_paged_attention::types::{KvCacheDtype, KvCacheLayout, KvQuantType};
use onnx_genai_paged_attention::validate::check_inputs;

// ----------------------------------------------------------------------------
// tiny deterministic helpers (no external rand dependency)
// ----------------------------------------------------------------------------

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    /// Uniform in [-1, 1).
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 40) as u32; // 24 high-ish bits
        (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * scale).collect()
    }
}

fn build_cos_sin(max_pos: usize, rotary_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = rotary_dim / 2;
    let mut cos = vec![0.0f32; max_pos * half];
    let mut sin = vec![0.0f32; max_pos * half];
    for p in 0..max_pos {
        for i in 0..half {
            let inv_freq = base.powf(-2.0 * i as f32 / rotary_dim as f32);
            let angle = p as f32 * inv_freq;
            cos[p * half + i] = angle.cos();
            sin[p * half + i] = angle.sin();
        }
    }
    (cos, sin)
}

/// Non-interleaved RoPE over `vec[offset..offset+rotary_dim)` at `pos`, matching
/// the oracle's `apply_rope`.
fn rope_inplace(
    vec: &mut [f32],
    offset: usize,
    rotary_dim: usize,
    pos: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = rotary_dim / 2;
    let base = pos * half;
    for i in 0..half {
        let c = cos[base + i];
        let s = sin[base + i];
        let i1 = offset + i;
        let i2 = offset + half + i;
        let x1 = vec[i1];
        let x2 = vec[i2];
        vec[i1] = x1 * c - x2 * s;
        vec[i2] = x2 * c + x1 * s;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn softmax(scores: &mut [f32]) {
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denom = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        denom += *s;
    }
    for s in scores.iter_mut() {
        *s /= denom;
    }
}

fn max_abs_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0.0f32;
    let mut den = 1e-6f32;
    for (x, y) in a.iter().zip(b) {
        num = num.max((x - y).abs());
        den = den.max(x.abs()).max(y.abs());
    }
    num / den
}

// ----------------------------------------------------------------------------
// input builders
// ----------------------------------------------------------------------------

fn base_attrs(num_heads: i64, kv_num_heads: i64) -> PagedAttentionAttributes {
    PagedAttentionAttributes {
        num_heads,
        kv_num_heads,
        ..Default::default()
    }
}

/// Build a minimal, schema-valid SEPARATE input-shape set.
fn separate_inputs(
    token_count: i64,
    num_heads: i64,
    kv_num_heads: i64,
    head_size: i64,
    batch: i64,
    num_blocks: i64,
    block_size: i64,
    max_blocks_per_seq: i64,
) -> PagedAttentionInputs {
    let q_hidden = num_heads * head_size;
    let kv_hidden = kv_num_heads * head_size;
    PagedAttentionInputs {
        query: Shape::new(vec![token_count, q_hidden]),
        key: Some(Shape::new(vec![token_count, kv_hidden])),
        value: Some(Shape::new(vec![token_count, kv_hidden])),
        key_cache: Shape::new(vec![num_blocks, block_size, kv_num_heads, head_size]),
        key_cache_storage_dtype: KvCacheDtype::Float16,
        value_cache: Some(Shape::new(vec![
            num_blocks,
            block_size,
            kv_num_heads,
            head_size,
        ])),
        value_cache_storage_dtype: KvCacheDtype::Float16,
        cumulative_sequence_length: Shape::new(vec![batch + 1]),
        past_seqlens: Shape::new(vec![batch]),
        block_table: Shape::new(vec![batch, max_blocks_per_seq]),
        cos_cache: None,
        sin_cache: None,
        slot_mapping: None,
        head_sink: None,
        q_norm_weight: None,
        k_norm_weight: None,
        k_scale: None,
        v_scale: None,
        attention_metadata: None,
        rotary_dim: 0,
    }
}

/// Build a minimal, schema-valid LATENT input-shape set.
fn latent_inputs(
    token_count: i64,
    num_heads: i64,
    head_size: i64,
    batch: i64,
    num_blocks: i64,
    block_size: i64,
    max_blocks_per_seq: i64,
    rotary_dim: i64,
) -> PagedAttentionInputs {
    let q_hidden = num_heads * head_size;
    PagedAttentionInputs {
        query: Shape::new(vec![token_count, q_hidden]),
        key: Some(Shape::new(vec![token_count, head_size])),
        value: None,
        key_cache: Shape::new(vec![num_blocks, block_size, 1, head_size]),
        key_cache_storage_dtype: KvCacheDtype::Float16,
        value_cache: None,
        value_cache_storage_dtype: KvCacheDtype::Default,
        cumulative_sequence_length: Shape::new(vec![batch + 1]),
        past_seqlens: Shape::new(vec![batch]),
        block_table: Shape::new(vec![batch, max_blocks_per_seq]),
        cos_cache: if rotary_dim > 0 {
            Some(Shape::new(vec![4096, rotary_dim / 2]))
        } else {
            None
        },
        sin_cache: if rotary_dim > 0 {
            Some(Shape::new(vec![4096, rotary_dim / 2]))
        } else {
            None
        },
        slot_mapping: None,
        head_sink: None,
        q_norm_weight: None,
        k_norm_weight: None,
        k_scale: None,
        v_scale: None,
        attention_metadata: None,
        rotary_dim,
    }
}

// ============================================================================
// schema validation
// ============================================================================

#[test]
fn block_size_must_be_power_of_two_at_least_16() {
    let attrs = base_attrs(2, 2);
    for bs in [8, 24, 48, 100] {
        let inputs = separate_inputs(4, 2, 2, 16, 1, 2, bs, 1);
        let err = check_inputs(&attrs, &inputs).unwrap_err();
        assert!(
            err.is_invalid_argument(),
            "block_size={bs} must be rejected"
        );
        assert!(err.to_string().contains("block_size"), "msg: {err}");
    }
    for bs in [16, 32, 64, 256] {
        let inputs = separate_inputs(4, 2, 2, 16, 1, 2, bs, 1);
        assert!(
            check_inputs(&attrs, &inputs).is_ok(),
            "block_size={bs} must be accepted"
        );
    }
}

#[test]
fn v_head_size_narrower_requires_latent() {
    // v_head_size < head_size under SEPARATE is a schema error.
    let mut attrs = base_attrs(2, 2);
    attrs.v_head_size = 8;
    attrs.scale = Some(0.1);
    let inputs = separate_inputs(4, 2, 2, 16, 1, 2, 16, 1);
    let err = check_inputs(&attrs, &inputs).unwrap_err();
    assert!(err.is_invalid_argument());
    assert!(err.to_string().contains("LATENT"), "msg: {err}");
}

#[test]
fn narrower_v_head_size_requires_explicit_scale() {
    // LATENT with v_head_size != head_size but no explicit scale → the trap.
    let mut attrs = base_attrs(2, 1);
    attrs.kv_cache_layout = KvCacheLayout::Latent;
    attrs.v_head_size = 16;
    attrs.rotary_offset = 16;
    attrs.do_rotary = true;
    let inputs = latent_inputs(4, 2, 32, 1, 2, 16, 1, 16);
    let err = check_inputs(&attrs, &inputs).unwrap_err();
    assert!(err.is_invalid_argument());
    assert!(err.to_string().contains("scale"), "msg: {err}");
}

#[test]
fn latent_accepts_deepseek_v3_shape() {
    // v_head_size=512, head_size=576, rope suffix=64 — the schema's own example.
    let mut attrs = base_attrs(8, 1);
    attrs.kv_cache_layout = KvCacheLayout::Latent;
    attrs.v_head_size = 512;
    attrs.rotary_offset = 512;
    attrs.do_rotary = true;
    attrs.scale = Some(1.0 / (192.0f32).sqrt());
    let inputs = latent_inputs(4, 8, 576, 1, 2, 16, 1, 64);
    let params = check_inputs(&attrs, &inputs).expect("DSV3 latent shape valid");
    assert!(params.is_latent_kv);
    assert_eq!(params.head_size, 576);
    assert_eq!(params.v_head_size, 512);
    assert_eq!(params.rotary_offset, 512);
    assert_eq!(params.rotary_dim, 64);
}

#[test]
fn invalid_block_table_and_seqlens_rejected() {
    let attrs = base_attrs(2, 2);
    // block_table dim0 (=3) must equal batch_size (=1).
    let mut inputs = separate_inputs(4, 2, 2, 16, 1, 2, 16, 1);
    inputs.block_table = Shape::new(vec![3, 1]);
    let err = check_inputs(&attrs, &inputs).unwrap_err();
    assert!(
        err.is_invalid_argument() && err.to_string().contains("block_table"),
        "msg: {err}"
    );

    // cumulative_sequence_length must be rank-1 with >= 2 elements.
    let mut inputs = separate_inputs(4, 2, 2, 16, 1, 2, 16, 1);
    inputs.cumulative_sequence_length = Shape::new(vec![1]);
    let err = check_inputs(&attrs, &inputs).unwrap_err();
    assert!(
        err.is_invalid_argument() && err.to_string().contains("cumulative"),
        "msg: {err}"
    );
}

#[test]
fn num_heads_must_be_multiple_of_kv_num_heads() {
    // helper.h:425-428 — invalid GQA ratio (4 % 3 != 0) is a schema error.
    let attrs = base_attrs(4, 3);
    let inputs = separate_inputs(6, 4, 3, 16, 1, 1, 16, 1);
    let err = check_inputs(&attrs, &inputs).unwrap_err();
    assert!(err.is_invalid_argument(), "msg: {err}");
    assert!(
        err.to_string().contains("multiple of kv_num_heads"),
        "msg: {err}"
    );
    // valid ratios still pass
    assert!(
        check_inputs(
            &base_attrs(4, 2),
            &separate_inputs(6, 4, 2, 16, 1, 1, 16, 1)
        )
        .is_ok()
    );
    assert!(
        check_inputs(
            &base_attrs(4, 4),
            &separate_inputs(6, 4, 4, 16, 1, 1, 16, 1)
        )
        .is_ok()
    );
}

#[test]
fn invalid_rotary_caches_rejected() {
    let mut attrs = base_attrs(2, 2);
    attrs.do_rotary = true;

    // cos_cache dim 1 must be a multiple of 8 (rotary_dim a multiple of 16).
    // head_size=32, cos dim1=4 → rotary_dim=8 → rejected.
    let mut inputs = separate_inputs(4, 2, 2, 32, 1, 1, 16, 1);
    inputs.cos_cache = Some(Shape::new(vec![64, 4]));
    inputs.sin_cache = Some(Shape::new(vec![64, 4]));
    inputs.rotary_dim = 8;
    let err = check_inputs(&attrs, &inputs).unwrap_err();
    assert!(
        err.is_invalid_argument() && err.to_string().contains("multiple of 8"),
        "msg: {err}"
    );

    // head_size must be a multiple of 16 when rotary caches are present.
    let mut inputs = separate_inputs(4, 2, 2, 24, 1, 1, 16, 1);
    inputs.cos_cache = Some(Shape::new(vec![64, 8]));
    inputs.sin_cache = Some(Shape::new(vec![64, 8]));
    inputs.rotary_dim = 16;
    let err = check_inputs(&attrs, &inputs).unwrap_err();
    assert!(
        err.is_invalid_argument() && err.to_string().contains("multiple of 16"),
        "msg: {err}"
    );

    // cos and sin dim 1 must match.
    let mut inputs = separate_inputs(4, 2, 2, 32, 1, 1, 16, 1);
    inputs.cos_cache = Some(Shape::new(vec![64, 8]));
    inputs.sin_cache = Some(Shape::new(vec![64, 16]));
    inputs.rotary_dim = 16;
    let err = check_inputs(&attrs, &inputs).unwrap_err();
    assert!(
        err.is_invalid_argument() && err.to_string().contains("must be the same"),
        "msg: {err}"
    );
}

// ============================================================================
// backend capability gate (typed NOT_IMPLEMENTED)
// ============================================================================

#[test]
fn webgpu_subset_rejects_every_unsupported_mode() {
    let sub = NativeSubset::webgpu_separate();

    // plain SEPARATE is accepted
    let attrs = base_attrs(4, 2);
    let inputs = separate_inputs(4, 4, 2, 16, 1, 2, 16, 1);
    let params = check_inputs(&attrs, &inputs).unwrap();
    assert!(check_backend_support(&params, &inputs, &sub).is_ok());

    // LATENT rejected
    let mut la = base_attrs(2, 1);
    la.kv_cache_layout = KvCacheLayout::Latent;
    la.v_head_size = 16;
    la.rotary_offset = 16;
    la.do_rotary = true;
    la.scale = Some(0.2);
    let li = latent_inputs(4, 2, 32, 1, 2, 16, 1, 16);
    let lp = check_inputs(&la, &li).unwrap();
    let err = check_backend_support(&lp, &li, &sub).unwrap_err();
    assert!(
        err.is_not_implemented() && err.to_string().contains("LATENT"),
        "msg: {err}"
    );

    // softcap rejected
    let mut sa = base_attrs(4, 2);
    sa.softcap = 30.0;
    let si = separate_inputs(4, 4, 2, 16, 1, 2, 16, 1);
    let sp = check_inputs(&sa, &si).unwrap();
    let err = check_backend_support(&sp, &si, &sub).unwrap_err();
    assert!(
        err.is_not_implemented() && err.to_string().contains("softcap"),
        "msg: {err}"
    );

    // head_sink rejected
    let ha = base_attrs(4, 2);
    let mut hi = separate_inputs(4, 4, 2, 16, 1, 2, 16, 1);
    hi.head_sink = Some(Shape::new(vec![4]));
    let hp = check_inputs(&ha, &hi).unwrap();
    let err = check_backend_support(&hp, &hi, &sub).unwrap_err();
    assert!(
        err.is_not_implemented() && err.to_string().contains("head_sink"),
        "msg: {err}"
    );

    // slot_mapping rejected
    let mut mi = separate_inputs(4, 4, 2, 16, 1, 2, 16, 1);
    mi.slot_mapping = Some(Shape::new(vec![4]));
    let mp = check_inputs(&base_attrs(4, 2), &mi).unwrap();
    let err = check_backend_support(&mp, &mi, &sub).unwrap_err();
    assert!(
        err.is_not_implemented() && err.to_string().contains("slot_mapping"),
        "msg: {err}"
    );

    // quantized cache rejected
    let mut qa = base_attrs(4, 2);
    qa.k_quant_type = KvQuantType::PerTensor;
    qa.v_quant_type = KvQuantType::PerTensor;
    let mut qi = separate_inputs(4, 4, 2, 16, 1, 2, 16, 1);
    qi.key_cache_storage_dtype = KvCacheDtype::Int8;
    qi.value_cache_storage_dtype = KvCacheDtype::Int8;
    qi.k_scale = Some(Shape::new(vec![1]));
    qi.v_scale = Some(Shape::new(vec![1]));
    let qp = check_inputs(&qa, &qi).unwrap();
    let err = check_backend_support(&qp, &qi, &sub).unwrap_err();
    assert!(
        err.is_not_implemented() && err.to_string().contains("quantized"),
        "msg: {err}"
    );

    // local_window_size rejected
    let mut wa = base_attrs(4, 2);
    wa.local_window_size = 64;
    let wi = separate_inputs(4, 4, 2, 16, 1, 2, 16, 1);
    let wp = check_inputs(&wa, &wi).unwrap();
    let err = check_backend_support(&wp, &wi, &sub).unwrap_err();
    assert!(
        err.is_not_implemented() && err.to_string().contains("local_window_size"),
        "msg: {err}"
    );

    // rotary_offset rejected
    let mut ra = base_attrs(4, 2);
    ra.do_rotary = true;
    ra.rotary_offset = 8;
    let mut ri = separate_inputs(4, 4, 2, 32, 1, 2, 16, 1);
    ri.cos_cache = Some(Shape::new(vec![64, 8]));
    ri.sin_cache = Some(Shape::new(vec![64, 8]));
    ri.rotary_dim = 16;
    let rp = check_inputs(&ra, &ri).unwrap();
    let err = check_backend_support(&rp, &ri, &sub).unwrap_err();
    assert!(
        err.is_not_implemented() && err.to_string().contains("rotary_offset"),
        "msg: {err}"
    );

    // q/k norm rejected
    let na = base_attrs(4, 2);
    let mut ni = separate_inputs(4, 4, 2, 16, 1, 2, 16, 1);
    ni.q_norm_weight = Some(Shape::new(vec![16]));
    ni.k_norm_weight = Some(Shape::new(vec![16]));
    let np = check_inputs(&na, &ni).unwrap();
    let err = check_backend_support(&np, &ni, &sub).unwrap_err();
    assert!(
        err.is_not_implemented() && err.to_string().contains("norm_weight"),
        "msg: {err}"
    );

    // attention_metadata rejected
    let aa = base_attrs(4, 2);
    let mut ai = separate_inputs(4, 4, 2, 16, 1, 2, 16, 1);
    ai.attention_metadata = Some(Shape::new(vec![2]));
    let ap = check_inputs(&aa, &ai).unwrap();
    let err = check_backend_support(&ap, &ai, &sub).unwrap_err();
    assert!(
        err.is_not_implemented() && err.to_string().contains("attention_metadata"),
        "msg: {err}"
    );
}

#[test]
fn glm_dense_mla_subset_accepts_latent_rejects_rest() {
    let sub = NativeSubset::glm_dense_mla_latent();

    // LATENT with narrower v + rope suffix is accepted
    let mut la = base_attrs(2, 1);
    la.kv_cache_layout = KvCacheLayout::Latent;
    la.v_head_size = 16;
    la.rotary_offset = 16;
    la.do_rotary = true;
    la.scale = Some(0.2);
    let li = latent_inputs(4, 2, 32, 1, 2, 16, 1, 16);
    let lp = check_inputs(&la, &li).unwrap();
    assert!(check_backend_support(&lp, &li, &sub).is_ok());

    // plain SEPARATE is NOT in this subset
    let attrs = base_attrs(4, 2);
    let inputs = separate_inputs(4, 4, 2, 16, 1, 2, 16, 1);
    let params = check_inputs(&attrs, &inputs).unwrap();
    let err = check_backend_support(&params, &inputs, &sub).unwrap_err();
    assert!(
        err.is_not_implemented() && err.to_string().contains("SEPARATE"),
        "msg: {err}"
    );

    // LATENT + softcap is still rejected (softcap not implemented)
    let mut sa = la.clone();
    sa.softcap = 20.0;
    let sp = check_inputs(&sa, &li).unwrap();
    let err = check_backend_support(&sp, &li, &sub).unwrap_err();
    assert!(
        err.is_not_implemented() && err.to_string().contains("softcap"),
        "msg: {err}"
    );
}

// ============================================================================
// SEPARATE GQA: oracle (paged) == naive dense (contiguous)
// ============================================================================

#[test]
fn separate_gqa_matches_naive_dense_with_partial_rope() {
    let (nh, kvnh, hs) = (4usize, 2usize, 32usize);
    let t = 6usize;
    let group = nh / kvnh;
    let scale = 1.0 / (hs as f32).sqrt();
    let rotary_dim = 16usize; // partial RoPE over the first 16 of 32 channels
    let (cos, sin) = build_cos_sin(64, rotary_dim, 10000.0);

    let mut rng = Lcg::new(42);
    let query = rng.vec(t * nh * hs, 0.5);
    let key = rng.vec(t * kvnh * hs, 0.5);
    let value = rng.vec(t * kvnh * hs, 0.5);

    // ---- naive contiguous reference ----
    let mut naive = vec![0.0f32; t * nh * hs];
    // pre-rope K per token/head (V is not roped)
    let mut k_roped = key.clone();
    for tok in 0..t {
        for kh in 0..kvnh {
            let off = tok * kvnh * hs + kh * hs;
            rope_inplace(&mut k_roped[off..off + hs], 0, rotary_dim, tok, &cos, &sin);
        }
    }
    for tok in 0..t {
        for h in 0..nh {
            let kh = h / group;
            let mut q = query[tok * nh * hs + h * hs..tok * nh * hs + h * hs + hs].to_vec();
            rope_inplace(&mut q, 0, rotary_dim, tok, &cos, &sin);
            let mut scores = Vec::new();
            for j in 0..=tok {
                let ko = j * kvnh * hs + kh * hs;
                scores.push(dot(&q, &k_roped[ko..ko + hs]) * scale);
            }
            softmax(&mut scores);
            let ob = tok * nh * hs + h * hs;
            for (j, p) in scores.iter().enumerate() {
                let vo = j * kvnh * hs + kh * hs;
                for d in 0..hs {
                    naive[ob + d] += p * value[vo + d];
                }
            }
        }
    }

    // ---- oracle (paged) ----
    let mut attrs = base_attrs(nh as i64, kvnh as i64);
    attrs.do_rotary = true;
    let inputs = {
        let mut i = separate_inputs(t as i64, nh as i64, kvnh as i64, hs as i64, 1, 1, 16, 1);
        i.cos_cache = Some(Shape::new(vec![64, (rotary_dim / 2) as i64]));
        i.sin_cache = Some(Shape::new(vec![64, (rotary_dim / 2) as i64]));
        i.rotary_dim = rotary_dim as i64;
        i
    };
    let params = check_inputs(&attrs, &inputs).unwrap();
    let mut key_cache = vec![0.0f32; 16 * kvnh * hs];
    let mut value_cache = vec![0.0f32; 16 * kvnh * hs];
    let cumulative = vec![0i32, t as i32];
    let past = vec![0i32];
    let block_table = vec![0i32];
    let data = PagedAttentionData {
        params: &params,
        query: &query,
        key: &key,
        value: Some(&value),
        cumulative_sequence_length: &cumulative,
        past_seqlens: &past,
        block_table: &block_table,
        slot_mapping: None,
        cos_cache: Some(&cos),
        sin_cache: Some(&sin),
        head_sink: None,
        q_norm_weight: None,
        k_norm_weight: None,
    };
    let out = paged_attention_reference(&data, &mut key_cache, Some(&mut value_cache));

    assert_eq!(out.len(), naive.len());
    let rel = max_abs_rel_diff(&out, &naive);
    assert!(rel < 1e-4, "SEPARATE oracle vs naive rel diff {rel}");
}

// ============================================================================
// paging placement-invariance (multi-block, permuted physical blocks)
// ============================================================================

#[test]
fn paging_is_placement_invariant_across_permuted_block_table() {
    let (nh, kvnh, hs) = (2usize, 1usize, 16usize);
    let t = 20usize; // spans 2 blocks of size 16
    let bs = 16usize;
    let mut rng = Lcg::new(7);
    let query = rng.vec(t * nh * hs, 0.5);
    let key = rng.vec(t * kvnh * hs, 0.5);
    let value = rng.vec(t * kvnh * hs, 0.5);

    let attrs = base_attrs(nh as i64, kvnh as i64);
    let run = |num_blocks: i64, block_table: Vec<i32>| -> Vec<f32> {
        let inputs = separate_inputs(
            t as i64,
            nh as i64,
            kvnh as i64,
            hs as i64,
            1,
            num_blocks,
            bs as i64,
            2,
        );
        let params = check_inputs(&attrs, &inputs).unwrap();
        let mut kc = vec![0.0f32; num_blocks as usize * bs * kvnh * hs];
        let mut vc = vec![0.0f32; num_blocks as usize * bs * kvnh * hs];
        let cumulative = vec![0i32, t as i32];
        let past = vec![0i32];
        let data = PagedAttentionData {
            params: &params,
            query: &query,
            key: &key,
            value: Some(&value),
            cumulative_sequence_length: &cumulative,
            past_seqlens: &past,
            block_table: &block_table,
            slot_mapping: None,
            cos_cache: None,
            sin_cache: None,
            head_sink: None,
            q_norm_weight: None,
            k_norm_weight: None,
        };
        paged_attention_reference(&data, &mut kc, Some(&mut vc))
    };

    let identity = run(2, vec![0, 1]);
    let permuted = run(4, vec![3, 1]);
    assert_eq!(identity.len(), permuted.len());
    for (a, b) in identity.iter().zip(&permuted) {
        assert!((a - b).abs() < 1e-6, "placement changed result: {a} vs {b}");
    }
}

// ============================================================================
// slot_mapping = -1 suppresses the write
// ============================================================================

#[test]
fn slot_mapping_negative_one_skips_write() {
    let (nh, kvnh, hs) = (1usize, 1usize, 16usize);
    let t = 3usize;
    let bs = 16usize;
    let num_blocks = 4i64;
    let mut rng = Lcg::new(9);
    let query = rng.vec(t * nh * hs, 0.5);
    let key = rng.vec(t * kvnh * hs, 0.5);
    let value = rng.vec(t * kvnh * hs, 0.5);

    let mut inputs = separate_inputs(
        t as i64,
        nh as i64,
        kvnh as i64,
        hs as i64,
        1,
        num_blocks,
        bs as i64,
        1,
    );
    inputs.slot_mapping = Some(Shape::new(vec![t as i64]));
    let attrs = base_attrs(nh as i64, kvnh as i64);
    let params = check_inputs(&attrs, &inputs).unwrap();

    const SENTINEL: f32 = 12345.0;
    let mut kc = vec![SENTINEL; num_blocks as usize * bs * kvnh * hs];
    let mut vc = vec![SENTINEL; num_blocks as usize * bs * kvnh * hs];
    let cumulative = vec![0i32, t as i32];
    let past = vec![0i32];
    let block_table = vec![0i32]; // unused because slot_mapping is provided
    let slot_mapping = vec![0i32, -1i32, 32i32]; // token 1 suppressed

    let data = PagedAttentionData {
        params: &params,
        query: &query,
        key: &key,
        value: Some(&value),
        cumulative_sequence_length: &cumulative,
        past_seqlens: &past,
        block_table: &block_table,
        slot_mapping: Some(&slot_mapping),
        cos_cache: None,
        sin_cache: None,
        head_sink: None,
        q_norm_weight: None,
        k_norm_weight: None,
    };
    let _ = paged_attention_reference(&data, &mut kc, Some(&mut vc));

    // Exactly the slots 0 and 32 must be written; every other row stays SENTINEL.
    let row = kvnh * hs;
    let written: Vec<usize> = (0..num_blocks as usize * bs)
        .filter(|&slot| {
            kc[slot * row..slot * row + row]
                .iter()
                .any(|&x| x != SENTINEL)
        })
        .collect();
    assert_eq!(written, vec![0, 32], "only non-(-1) slots may be written");
}

// ============================================================================
// absorbed LATENT MLA == decomposed multi-head latent attention
// ============================================================================

/// Runs the absorbed-vs-decomposed knockout for a given MLA geometry.
///
/// * `l` = kv_lora_rank = v_head_size (absorbed content / value width)
/// * `r` = decoupled RoPE dim (head_size = l + r)
/// * `d` = qk_nope_head_dim (W_UK middle dim)
/// * `dv` = value head dim (W_UV output width)
fn assert_absorbed_equals_decomposed(
    nh: usize,
    l: usize,
    r: usize,
    d: usize,
    dv: usize,
    t: usize,
    out_dim: usize,
    seed: u64,
) {
    let hs = l + r; // absorbed head_size
    let scale = 1.0 / ((d + r) as f32).sqrt();
    let (cos, sin) = build_cos_sin(64, r, 10000.0);
    let mut rng = Lcg::new(seed);

    // per-token latent content c[j] (dim l) and decoupled rope key krope[j] (dim r)
    let c: Vec<Vec<f32>> = (0..t)
        .map(|_| rng.vec(l, 1.0 / (l as f32).sqrt()))
        .collect();
    let krope: Vec<Vec<f32>> = (0..t).map(|_| rng.vec(r, 1.0)).collect();
    // per-token/head query nope (dim d) and rope (dim r)
    let qnope: Vec<Vec<Vec<f32>>> = (0..t)
        .map(|_| {
            (0..nh)
                .map(|_| rng.vec(d, 1.0 / (d as f32).sqrt()))
                .collect()
        })
        .collect();
    let qrope: Vec<Vec<Vec<f32>>> = (0..t)
        .map(|_| (0..nh).map(|_| rng.vec(r, 1.0)).collect())
        .collect();
    // per-head projections
    let w_uk: Vec<Vec<f32>> = (0..nh)
        .map(|_| rng.vec(d * l, 1.0 / (l as f32).sqrt()))
        .collect(); // [d][l]
    let w_uv: Vec<Vec<f32>> = (0..nh)
        .map(|_| rng.vec(dv * l, 1.0 / (l as f32).sqrt()))
        .collect(); // [dv][l]
    let w_o = rng.vec(out_dim * nh * dv, 1.0 / ((nh * dv) as f32).sqrt()); // [out][nh*dv]

    let matvec = |m: &[f32], rows: usize, cols: usize, x: &[f32]| -> Vec<f32> {
        (0..rows)
            .map(|i| dot(&m[i * cols..i * cols + cols], x))
            .collect()
    };
    let matvec_t = |m: &[f32], rows: usize, cols: usize, x: &[f32]| -> Vec<f32> {
        // returns m^T x, dim = cols
        let mut out = vec![0.0f32; cols];
        for i in 0..rows {
            for jc in 0..cols {
                out[jc] += m[i * cols + jc] * x[i];
            }
        }
        out
    };

    // ---- decomposed reference ----
    let mut decomposed = vec![0.0f32; t * out_dim];
    for tok in 0..t {
        let mut concat = vec![0.0f32; nh * dv];
        for h in 0..nh {
            let mut qr = qrope[tok][h].clone();
            rope_inplace(&mut qr, 0, r, tok, &cos, &sin);
            let mut scores = Vec::with_capacity(tok + 1);
            for j in 0..=tok {
                let kn = matvec(&w_uk[h], d, l, &c[j]); // W_UK c_j  (dim d)
                let mut kr = krope[j].clone();
                rope_inplace(&mut kr, 0, r, j, &cos, &sin);
                let s = (dot(&qnope[tok][h], &kn) + dot(&qr, &kr)) * scale;
                scores.push(s);
            }
            softmax(&mut scores);
            // ctx = sum p * (W_UV c_j)  (dim dv)
            let mut ctx = vec![0.0f32; dv];
            for (j, &p) in scores.iter().enumerate() {
                let v = matvec(&w_uv[h], dv, l, &c[j]);
                for k in 0..dv {
                    ctx[k] += p * v[k];
                }
            }
            concat[h * dv..h * dv + dv].copy_from_slice(&ctx);
        }
        let fin = matvec(&w_o, out_dim, nh * dv, &concat);
        decomposed[tok * out_dim..tok * out_dim + out_dim].copy_from_slice(&fin);
    }

    // ---- absorbed via the LATENT oracle ----
    // query buffer: per token/head [ a_h = W_UK^T qnope (dim l) ; qrope (dim r) ]
    let mut query = vec![0.0f32; t * nh * hs];
    for tok in 0..t {
        for h in 0..nh {
            let a = matvec_t(&w_uk[h], d, l, &qnope[tok][h]); // dim l
            let base = (tok * nh + h) * hs;
            query[base..base + l].copy_from_slice(&a);
            query[base + l..base + hs].copy_from_slice(&qrope[tok][h]);
        }
    }
    // key buffer: per token [ c_j (dim l) ; krope_j (dim r) ]  (pre-rope)
    let mut key = vec![0.0f32; t * hs];
    for j in 0..t {
        key[j * hs..j * hs + l].copy_from_slice(&c[j]);
        key[j * hs + l..j * hs + hs].copy_from_slice(&krope[j]);
    }

    let mut attrs = base_attrs(nh as i64, 1);
    attrs.kv_cache_layout = KvCacheLayout::Latent;
    attrs.v_head_size = l as i64;
    attrs.rotary_offset = l as i64;
    attrs.do_rotary = true;
    attrs.scale = Some(scale);
    let num_blocks = t.div_ceil(16).max(1) as i64;
    let inputs = latent_inputs(
        t as i64, nh as i64, hs as i64, 1, num_blocks, 16, 1, r as i64,
    );
    let params = check_inputs(&attrs, &inputs).unwrap();

    let mut key_cache = vec![0.0f32; num_blocks as usize * 16 * hs];
    let cumulative = vec![0i32, t as i32];
    let past = vec![0i32];
    let block_table: Vec<i32> = (0..num_blocks as i32).collect();
    let data = PagedAttentionData {
        params: &params,
        query: &query,
        key: &key,
        value: None,
        cumulative_sequence_length: &cumulative,
        past_seqlens: &past,
        block_table: &block_table,
        slot_mapping: None,
        cos_cache: Some(&cos),
        sin_cache: Some(&sin),
        head_sink: None,
        q_norm_weight: None,
        k_norm_weight: None,
    };
    let absorbed_ctx = paged_attention_reference(&data, &mut key_cache, None); // [t, nh*l]

    // fold W_UV then W_O back in (deferred to after the op)
    let mut absorbed = vec![0.0f32; t * out_dim];
    for tok in 0..t {
        let mut concat = vec![0.0f32; nh * dv];
        for h in 0..nh {
            let ac = &absorbed_ctx[tok * nh * l + h * l..tok * nh * l + h * l + l];
            let real = matvec(&w_uv[h], dv, l, ac);
            concat[h * dv..h * dv + dv].copy_from_slice(&real);
        }
        let fin = matvec(&w_o, out_dim, nh * dv, &concat);
        absorbed[tok * out_dim..tok * out_dim + out_dim].copy_from_slice(&fin);
    }

    let rel = max_abs_rel_diff(&absorbed, &decomposed);
    assert!(
        rel < 1e-3,
        "absorbed vs decomposed rel diff {rel} (l={l},r={r},d={d})"
    );
}

#[test]
fn absorbed_latent_equals_decomposed_mla_tiny() {
    assert_absorbed_equals_decomposed(2, 16, 16, 10, 6, 5, 8, 1234);
}

#[test]
fn absorbed_latent_equals_decomposed_mla_deepseek_v3_dims() {
    // kv_lora_rank=512, rope=64 → head_size=576; qk_nope=128, v head dim=128.
    assert_absorbed_equals_decomposed(2, 512, 64, 128, 128, 4, 16, 99);
}

// ============================================================================
// head_sink smooth-softmax and q/k RMSNorm
// ============================================================================

#[test]
fn head_sink_shrinks_attention_mass() {
    let (nh, kvnh, hs) = (2usize, 2usize, 16usize);
    let t = 4usize;
    let mut rng = Lcg::new(5);
    let query = rng.vec(t * nh * hs, 0.5);
    let key = rng.vec(t * kvnh * hs, 0.5);
    let value = rng.vec(t * kvnh * hs, 0.5);

    let run = |sink: Option<&[f32]>, head_sink_shape: bool| -> Vec<f32> {
        let mut inputs = separate_inputs(t as i64, nh as i64, kvnh as i64, hs as i64, 1, 1, 16, 1);
        if head_sink_shape {
            inputs.head_sink = Some(Shape::new(vec![nh as i64]));
        }
        let params = check_inputs(&base_attrs(nh as i64, kvnh as i64), &inputs).unwrap();
        let mut kc = vec![0.0f32; 16 * kvnh * hs];
        let mut vc = vec![0.0f32; 16 * kvnh * hs];
        let cumulative = vec![0i32, t as i32];
        let past = vec![0i32];
        let block_table = vec![0i32];
        let data = PagedAttentionData {
            params: &params,
            query: &query,
            key: &key,
            value: Some(&value),
            cumulative_sequence_length: &cumulative,
            past_seqlens: &past,
            block_table: &block_table,
            slot_mapping: None,
            cos_cache: None,
            sin_cache: None,
            head_sink: sink,
            q_norm_weight: None,
            k_norm_weight: None,
        };
        paged_attention_reference(&data, &mut kc, Some(&mut vc))
    };

    let no_sink = run(None, false);
    // A very large sink logit dominates the denominator → context → 0.
    let big = vec![50.0f32; nh];
    let with_sink = run(Some(&big), true);
    let diff = max_abs_rel_diff(&no_sink, &with_sink);
    assert!(
        diff > 0.5,
        "head_sink must change the output, rel diff {diff}"
    );
    let mag: f32 = with_sink.iter().map(|x| x.abs()).sum();
    assert!(
        mag < 1e-2,
        "huge sink logit should drive context to ~0, got mag {mag}"
    );
}

#[test]
fn qk_norm_path_runs_and_changes_output() {
    let (nh, kvnh, hs) = (2usize, 2usize, 16usize);
    let t = 4usize;
    let mut rng = Lcg::new(11);
    let query = rng.vec(t * nh * hs, 2.0);
    let key = rng.vec(t * kvnh * hs, 2.0);
    let value = rng.vec(t * kvnh * hs, 0.5);
    let qn = vec![1.0f32; hs];
    let kn = vec![1.0f32; hs];

    let run = |use_norm: bool| -> Vec<f32> {
        let mut inputs = separate_inputs(t as i64, nh as i64, kvnh as i64, hs as i64, 1, 1, 16, 1);
        if use_norm {
            inputs.q_norm_weight = Some(Shape::new(vec![hs as i64]));
            inputs.k_norm_weight = Some(Shape::new(vec![hs as i64]));
        }
        let params = check_inputs(&base_attrs(nh as i64, kvnh as i64), &inputs).unwrap();
        let mut kc = vec![0.0f32; 16 * kvnh * hs];
        let mut vc = vec![0.0f32; 16 * kvnh * hs];
        let cumulative = vec![0i32, t as i32];
        let past = vec![0i32];
        let block_table = vec![0i32];
        let data = PagedAttentionData {
            params: &params,
            query: &query,
            key: &key,
            value: Some(&value),
            cumulative_sequence_length: &cumulative,
            past_seqlens: &past,
            block_table: &block_table,
            slot_mapping: None,
            cos_cache: None,
            sin_cache: None,
            head_sink: None,
            q_norm_weight: if use_norm { Some(&qn) } else { None },
            k_norm_weight: if use_norm { Some(&kn) } else { None },
        };
        paged_attention_reference(&data, &mut kc, Some(&mut vc))
    };

    let plain = run(false);
    let normed = run(true);
    let diff = max_abs_rel_diff(&plain, &normed);
    assert!(
        diff > 1e-3,
        "RMSNorm must change the scores, rel diff {diff}"
    );
}
