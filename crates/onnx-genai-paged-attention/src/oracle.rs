//! CPU reference oracle for dense paged attention (`SEPARATE` and `LATENT`).
//!
//! This mirrors what the ORT CUDA kernel computes, in `f32`, over a block cache
//! addressed by `block_table` / `slot_mapping`. It is the correctness gate for a
//! future CUDA `LATENT` kernel and the tool that proves — numerically — that
//! absorbed MLA is expressible as `PagedAttention` with
//! `kv_cache_layout="LATENT"` (see `tests/equivalence.rs`).
//!
//! Upstream ORT ships no CPU kernel; this is test/reference code only.

use crate::params::PagedAttentionParameters;

/// Immutable inputs for one `paged_attention_reference` call. All buffers are
/// row-major `f32`; the KV caches are passed separately (they are mutated).
///
/// * `query` — `[token_count, num_heads * head_size]`.
/// * `key` — new-token keys: `[token_count, kv_num_heads * head_size]` for
///   `SEPARATE`, or the latent rows `[token_count, head_size]` (kv_num_heads=1)
///   for `LATENT`.
/// * `value` — new-token values `[token_count, kv_num_heads * head_size]`;
///   `None` for `LATENT`.
/// * `cumulative_sequence_length` — `[batch+1]`; `past_seqlens` — `[batch]`.
/// * `block_table` — `[batch * max_num_blocks_per_seq]`.
/// * `slot_mapping` — `[token_count]`, `-1` skips the write; when `None`, slots
///   are derived from `past_seqlens` + `block_table`.
/// * `cos_cache` / `sin_cache` — `[max_pos * (rotary_dim/2)]`.
pub struct PagedAttentionData<'a> {
    pub params: &'a PagedAttentionParameters,
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: Option<&'a [f32]>,
    pub cumulative_sequence_length: &'a [i32],
    pub past_seqlens: &'a [i32],
    pub block_table: &'a [i32],
    pub slot_mapping: Option<&'a [i32]>,
    pub cos_cache: Option<&'a [f32]>,
    pub sin_cache: Option<&'a [f32]>,
    pub head_sink: Option<&'a [f32]>,
    pub q_norm_weight: Option<&'a [f32]>,
    pub k_norm_weight: Option<&'a [f32]>,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn rms_norm_inplace(vec: &mut [f32], gain: &[f32], epsilon: f32) {
    let n = vec.len() as f32;
    let mean_sq = vec.iter().map(|x| x * x).sum::<f32>() / n;
    let inv = 1.0 / (mean_sq + epsilon).sqrt();
    for (x, g) in vec.iter_mut().zip(gain) {
        *x = *x * inv * g;
    }
}

/// Apply RoPE to `vec[offset .. offset+rotary_dim)` at absolute `pos`.
fn apply_rope(
    vec: &mut [f32],
    offset: usize,
    rotary_dim: usize,
    pos: usize,
    cos_cache: &[f32],
    sin_cache: &[f32],
    interleaved: bool,
) {
    if rotary_dim == 0 {
        return;
    }
    let half = rotary_dim / 2;
    let base = pos * half;
    for i in 0..half {
        let c = cos_cache[base + i];
        let s = sin_cache[base + i];
        let (i1, i2) = if interleaved {
            (offset + 2 * i, offset + 2 * i + 1)
        } else {
            (offset + i, offset + half + i)
        };
        let x1 = vec[i1];
        let x2 = vec[i2];
        vec[i1] = x1 * c - x2 * s;
        vec[i2] = x2 * c + x1 * s;
    }
}

/// Map (batch `b`, in-sequence position `pos`) to a flat cache slot via the
/// block table, or read the scheduler-provided `slot_mapping` value.
fn write_slot(data: &PagedAttentionData, token: usize, b: usize, pos: usize) -> i64 {
    if let Some(sm) = data.slot_mapping {
        return i64::from(sm[token]);
    }
    let p = data.params;
    let block_in_seq = pos / p.block_size as usize;
    let phys_block = data.block_table[b * p.max_num_blocks_per_seq as usize + block_in_seq] as i64;
    phys_block * p.block_size + (pos % p.block_size as usize) as i64
}

fn read_slot(data: &PagedAttentionData, b: usize, j: usize) -> usize {
    let p = data.params;
    let block_in_seq = j / p.block_size as usize;
    let phys_block =
        data.block_table[b * p.max_num_blocks_per_seq as usize + block_in_seq] as usize;
    phys_block * p.block_size as usize + (j % p.block_size as usize)
}

/// Compute dense paged attention, mutating the KV cache(s) in place, and return
/// the output `[token_count, num_heads * v_head_size]`.
///
/// # Panics
/// Panics if a buffer is too small for the geometry in `data.params` — this is a
/// test oracle, so a malformed scenario is a test bug, not a runtime condition.
#[allow(clippy::too_many_lines)]
pub fn paged_attention_reference(
    data: &PagedAttentionData,
    key_cache: &mut [f32],
    mut value_cache: Option<&mut [f32]>,
) -> Vec<f32> {
    let p = data.params;
    let nh = p.num_heads as usize;
    let kvnh = p.kv_num_heads as usize;
    let hs = p.head_size as usize;
    let vhs = p.v_head_size as usize;
    let batch = p.batch_size as usize;
    let token_count = p.token_count as usize;
    let group = nh / kvnh; // GQA fan-out (1 for LATENT and MHA)
    let is_latent = p.is_latent_kv;
    let rotary_offset = p.rotary_offset as usize;
    let rotary_dim = p.rotary_dim as usize;
    let q_hidden = nh * hs;
    let kv_hidden = kvnh * hs;

    // Per global token: its batch and absolute in-sequence position.
    let mut token_batch = vec![0usize; token_count];
    let mut token_pos = vec![0usize; token_count];
    for b in 0..batch {
        let start = data.cumulative_sequence_length[b] as usize;
        let end = data.cumulative_sequence_length[b + 1] as usize;
        for (local, j) in (start..end).enumerate() {
            token_batch[j] = b;
            token_pos[j] = data.past_seqlens[b] as usize + local;
        }
    }

    let cos = data.cos_cache;
    let sin = data.sin_cache;

    // ---- Write phase: store post-norm, post-RoPE K (and raw V) into the cache.
    for t in 0..token_count {
        let b = token_batch[t];
        let pos = token_pos[t];
        let slot = write_slot(data, t, b, pos);
        if slot < 0 {
            continue; // -1 → suppressed write (prefix hit / rejected token)
        }
        let slot = slot as usize;
        if is_latent {
            let mut row = data.key[t * hs..t * hs + hs].to_vec();
            if p.do_rotary {
                apply_rope(
                    &mut row,
                    rotary_offset,
                    rotary_dim,
                    pos,
                    cos.expect("cos"),
                    sin.expect("sin"),
                    p.rotary_interleaved,
                );
            }
            let dst = slot * kvnh * hs; // kv head 0
            key_cache[dst..dst + hs].copy_from_slice(&row);
        } else {
            for kh in 0..kvnh {
                let mut k =
                    data.key[t * kv_hidden + kh * hs..t * kv_hidden + kh * hs + hs].to_vec();
                if let Some(w) = data.k_norm_weight {
                    rms_norm_inplace(&mut k, w, p.qk_norm_epsilon);
                }
                if p.do_rotary {
                    apply_rope(
                        &mut k,
                        rotary_offset,
                        rotary_dim,
                        pos,
                        cos.expect("cos"),
                        sin.expect("sin"),
                        p.rotary_interleaved,
                    );
                }
                let dst = (slot * kvnh + kh) * hs;
                key_cache[dst..dst + hs].copy_from_slice(&k);

                let value = data.value.expect("value present in SEPARATE");
                let v = &value[t * kv_hidden + kh * hs..t * kv_hidden + kh * hs + hs];
                let vc = value_cache.as_deref_mut().expect("value_cache in SEPARATE");
                vc[dst..dst + hs].copy_from_slice(v);
            }
        }
    }

    // ---- Read phase: dense causal attention over the cached prefix.
    let mut output = vec![0.0f32; token_count * nh * vhs];
    for t in 0..token_count {
        let b = token_batch[t];
        let pos = token_pos[t];
        for h in 0..nh {
            let kh = if is_latent { 0 } else { h / group };
            let mut q = data.query[t * q_hidden + h * hs..t * q_hidden + h * hs + hs].to_vec();
            if let Some(w) = data.q_norm_weight {
                rms_norm_inplace(&mut q, w, p.qk_norm_epsilon);
            }
            if p.do_rotary {
                apply_rope(
                    &mut q,
                    rotary_offset,
                    rotary_dim,
                    pos,
                    cos.expect("cos"),
                    sin.expect("sin"),
                    p.rotary_interleaved,
                );
            }

            // Scores over cached positions 0..=pos (causal), honoring the window.
            let mut scores = Vec::with_capacity(pos + 1);
            for j in 0..=pos {
                if p.local_window_size >= 0 && (pos - j) as i64 > p.local_window_size {
                    continue;
                }
                let slot = read_slot(data, b, j);
                let k = &key_cache[(slot * kvnh + kh) * hs..(slot * kvnh + kh) * hs + hs];
                let mut s = dot(&q, k) * p.scale;
                if p.softcap > 0.0 {
                    s = p.softcap * (s / p.softcap).tanh();
                }
                scores.push((j, s));
            }

            // Softmax with an optional per-head sink logit in the denominator.
            let mut m = f32::NEG_INFINITY;
            for &(_, s) in &scores {
                m = m.max(s);
            }
            let sink = data.head_sink.map(|hsk| hsk[h]);
            if let Some(sv) = sink {
                m = m.max(sv);
            }
            let mut denom = 0.0f32;
            for &(_, s) in &scores {
                denom += (s - m).exp();
            }
            if let Some(sv) = sink {
                denom += (sv - m).exp(); // contributes to denom, not to value
            }

            let out_base = t * nh * vhs + h * vhs;
            for &(j, s) in &scores {
                let prob = (s - m).exp() / denom;
                let slot = read_slot(data, b, j);
                if is_latent {
                    // V = leading v_head_size channels of the latent row.
                    let vbase = slot * kvnh * hs;
                    for d in 0..vhs {
                        output[out_base + d] += prob * key_cache[vbase + d];
                    }
                } else {
                    let vc = value_cache.as_deref().expect("value_cache in SEPARATE");
                    let vbase = (slot * kvnh + kh) * hs;
                    for d in 0..vhs {
                        output[out_base + d] += prob * vc[vbase + d];
                    }
                }
            }
        }
    }

    output
}
