#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::uninlined_format_args,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
//! On-GPU parity test for the native `com.microsoft::PagedAttention` v1 LATENT
//! (absorbed MLA) kernel used by GLM-5.2 dense MLA.
//!
//! The native CUDA kernel (`kernels::paged_attention`) is checked, byte-in /
//! numbers-out, against the shared CPU oracle
//! (`onnx_genai_paged_attention::oracle::paged_attention_reference`) — the same
//! reference that gates the typed validator. Both sides consume the *identical*
//! dtype-rounded inputs and identical initial cache, so the only permitted
//! divergence is the f16/bf16 rounding of the post-RoPE K the kernel writes to
//! the cache; tolerances are sized for exactly that.
//!
//! Gated on a real device: without the `gpu-tests` feature these are ignored.
//! Run on an idle GPU with, e.g.:
//!   CUDA_HOME=/path/to/cuda CUDA_VISIBLE_DEVICES=1 \
//!   LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH \
//!   cargo test -p onnx-runtime-ep-cuda --features gpu-tests --test paged_attention_latent_gpu

use half::{bf16, f16};
use onnx_genai_paged_attention::oracle::{PagedAttentionData, paged_attention_reference};
use onnx_genai_paged_attention::params::PagedAttentionParameters;
use onnx_genai_paged_attention::types::KvQuantType;
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, KernelFactory as _, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::kernels::paged_attention::PagedAttentionFactory;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{Attribute, DataType, DeviceId, Node, NodeId, compute_contiguous_strides};
use std::sync::Arc;

// ---------------------------------------------------------------- byte helpers

fn encode(values: &[f32], dtype: DataType) -> Vec<u8> {
    match dtype {
        DataType::Float32 => values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        DataType::Float16 => values
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_bits().to_ne_bytes())
            .collect(),
        DataType::BFloat16 => values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_ne_bytes())
            .collect(),
        _ => unreachable!("float dtype only"),
    }
}

fn decode(bytes: &[u8], dtype: DataType) -> Vec<f32> {
    match dtype {
        DataType::Float32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DataType::Float16 => bytes
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_ne_bytes([c[0], c[1]])).to_f32())
            .collect(),
        DataType::BFloat16 => bytes
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_ne_bytes([c[0], c[1]])).to_f32())
            .collect(),
        _ => unreachable!("float dtype only"),
    }
}

/// Round f32 values through the storage dtype so the oracle and the kernel see
/// the exact same inputs.
fn round_dtype(values: &[f32], dtype: DataType) -> Vec<f32> {
    decode(&encode(values, dtype), dtype)
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in comparison");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ------------------------------------------------------------------------- RNG

struct Lcg(u64);
impl Lcg {
    fn f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
}

// ----------------------------------------------------------------- scenario

/// A LATENT paged-attention case. Sequences may carry `past` (already-cached)
/// positions plus `new` (this-call) tokens, exercising prefill and decode.
struct Scenario {
    dtype: DataType,
    num_heads: usize,
    head_size: usize,
    v_head_size: usize,
    block_size: usize,
    rotary_dim: usize,
    rotary_offset: usize,
    interleaved: bool,
    scale: f32,
    /// (past_len, new_len) per batch sequence.
    seqs: Vec<(usize, usize)>,
    /// Optionally force some new-token slots to -1 (skip write). Indexed by
    /// global new-token index.
    skip_tokens: Vec<usize>,
}

struct BuiltScenario {
    dtype: DataType,
    params: PagedAttentionParameters,
    query: Vec<f32>,
    key: Vec<f32>,
    cache_init: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    cumseq: Vec<i32>,
    past: Vec<i32>,
    block_table: Vec<i32>,
    slot_mapping: Vec<i32>,
    token_count: usize,
    num_rows: usize,
    max_pos: usize,
    max_blocks_per_seq: usize,
}

fn build(s: &Scenario) -> BuiltScenario {
    let batch = s.seqs.len();
    let hs = s.head_size;
    let mut rng = Lcg(0x1234_5678_9abc_def0 ^ (hs as u64) ^ ((s.num_heads as u64) << 20));

    let totals: Vec<usize> = s.seqs.iter().map(|(p, n)| p + n).collect();
    let max_pos = *totals.iter().max().unwrap();
    let max_blocks_per_seq = max_pos.div_ceil(s.block_size);

    // Assign a unique, contiguous run of physical blocks to each sequence.
    let mut block_table = vec![0i32; batch * max_blocks_per_seq];
    let mut next_block = 0usize;
    for b in 0..batch {
        let need = totals[b].div_ceil(s.block_size).max(1);
        for k in 0..max_blocks_per_seq {
            block_table[b * max_blocks_per_seq + k] = if k < need {
                let phys = next_block + k;
                phys as i32
            } else {
                0
            };
        }
        next_block += need;
    }
    let num_blocks = next_block.max(1);
    let num_rows = num_blocks * s.block_size;

    let read_slot = |b: usize, pos: usize| -> usize {
        let blk = pos / s.block_size;
        let phys = block_table[b * max_blocks_per_seq + blk] as usize;
        phys * s.block_size + pos % s.block_size
    };

    // Initial cache: fill the already-cached (past) rows with rounded randoms.
    let mut cache_init = vec![0f32; num_rows * hs];
    for b in 0..batch {
        let (past_len, _) = s.seqs[b];
        for p in 0..past_len {
            let slot = read_slot(b, p);
            let row = rng.fill(hs);
            let row = round_dtype(&row, s.dtype);
            cache_init[slot * hs..slot * hs + hs].copy_from_slice(&row);
        }
    }

    // New tokens.
    let token_count: usize = s.seqs.iter().map(|(_, n)| n).sum();
    let mut cumseq = vec![0i32; batch + 1];
    for b in 0..batch {
        cumseq[b + 1] = cumseq[b] + s.seqs[b].1 as i32;
    }
    let past: Vec<i32> = s.seqs.iter().map(|(p, _)| *p as i32).collect();

    let mut query = round_dtype(&rng.fill(token_count * s.num_heads * hs), s.dtype);
    let mut key = round_dtype(&rng.fill(token_count * hs), s.dtype);
    // Guard against accidental zero-length.
    if query.is_empty() {
        query = vec![0.0];
    }
    if key.is_empty() {
        key = vec![0.0];
    }

    let mut slot_mapping = vec![0i32; token_count];
    for b in 0..batch {
        let (past_len, new_len) = s.seqs[b];
        let start = cumseq[b] as usize;
        for l in 0..new_len {
            let pos = past_len + l;
            slot_mapping[start + l] = read_slot(b, pos) as i32;
        }
    }
    for &t in &s.skip_tokens {
        slot_mapping[t] = -1;
    }

    // Rotary caches sized [max_pos, rotary_dim/2].
    let half = s.rotary_dim / 2;
    let cos = round_dtype(&rng.fill(max_pos.max(1) * half.max(1)), s.dtype);
    let sin = round_dtype(&rng.fill(max_pos.max(1) * half.max(1)), s.dtype);

    let params = PagedAttentionParameters {
        batch_size: batch as i64,
        token_count: token_count as i64,
        num_heads: s.num_heads as i64,
        kv_num_heads: 1,
        head_size: hs as i64,
        v_head_size: s.v_head_size as i64,
        hidden_size: (s.num_heads * hs) as i64,
        v_hidden_size: (s.num_heads * s.v_head_size) as i64,
        kv_hidden_size: hs as i64,
        is_latent_kv: true,
        is_packed_qkv: false,
        rotary_offset: s.rotary_offset as i64,
        rotary_dim: s.rotary_dim as i64,
        block_size: s.block_size as i64,
        num_blocks: num_blocks as i64,
        max_num_blocks_per_seq: max_blocks_per_seq as i64,
        scale: s.scale,
        softcap: 0.0,
        local_window_size: -1,
        do_rotary: s.rotary_dim > 0,
        rotary_interleaved: s.interleaved,
        use_head_sink: false,
        use_qk_norm: false,
        qk_norm_epsilon: 0.0,
        k_quant_type: KvQuantType::None,
        v_quant_type: KvQuantType::None,
    };

    BuiltScenario {
        dtype: s.dtype,
        params,
        query,
        key,
        cache_init,
        cos,
        sin,
        cumseq,
        past,
        block_table,
        slot_mapping,
        token_count,
        num_rows,
        max_pos,
        max_blocks_per_seq,
    }
}

// --------------------------------------------------------------- oracle side

fn run_oracle(b: &BuiltScenario) -> (Vec<f32>, Vec<f32>) {
    let mut cache = b.cache_init.clone();
    let data = PagedAttentionData {
        params: &b.params,
        query: &b.query,
        key: &b.key,
        value: None,
        cumulative_sequence_length: &b.cumseq,
        past_seqlens: &b.past,
        block_table: &b.block_table,
        slot_mapping: Some(&b.slot_mapping),
        cos_cache: if b.params.do_rotary {
            Some(&b.cos)
        } else {
            None
        },
        sin_cache: if b.params.do_rotary {
            Some(&b.sin)
        } else {
            None
        },
        head_sink: None,
        q_norm_weight: None,
        k_norm_weight: None,
    };
    let out = paged_attention_reference(&data, &mut cache, None);
    (out, cache)
}

// ---------------------------------------------------------------- kernel side

struct Ctx {
    ep: CudaExecutionProvider,
    runtime: Arc<onnx_runtime_ep_cuda::CudaRuntime>,
    dev: DeviceId,
}

fn upload(ctx: &Ctx, bytes: &[u8]) -> onnx_runtime_ep_api::DeviceBuffer {
    let buf = ctx.ep.allocate(bytes.len().max(1), 256).unwrap();
    unsafe {
        ctx.runtime.htod(bytes, cuptr(buf.as_ptr())).unwrap();
    }
    buf
}

fn build_node(b: &BuiltScenario) -> Node {
    let p = &b.params;
    let mut node = Node::new(NodeId(0), "PagedAttention", vec![None; 11], vec![]);
    node.domain = "com.microsoft".to_string();
    let mut set = |k: &str, v: Attribute| {
        node.attributes.insert(k.to_string(), v);
    };
    set("kv_cache_layout", Attribute::String(b"LATENT".to_vec()));
    set("num_heads", Attribute::Int(p.num_heads));
    set("kv_num_heads", Attribute::Int(1));
    set("v_head_size", Attribute::Int(p.v_head_size));
    set("rotary_offset", Attribute::Int(p.rotary_offset));
    set("do_rotary", Attribute::Int(i64::from(p.do_rotary)));
    set(
        "rotary_interleaved",
        Attribute::Int(i64::from(p.rotary_interleaved)),
    );
    set("scale", Attribute::Float(p.scale));
    set("softcap", Attribute::Float(p.softcap));
    set("local_window_size", Attribute::Int(p.local_window_size));
    node
}

/// Returns `(output, cache_after)` from the native kernel.
fn run_native(ctx: &Ctx, b: &BuiltScenario) -> (Vec<f32>, Vec<f32>) {
    let dtype = b.dtype;
    let hs = b.params.head_size as usize;
    let vhs = b.params.v_head_size as usize;
    let nh = b.params.num_heads as usize;
    let batch = b.params.batch_size as usize;
    let elem = 2usize; // f16/bf16

    // Upload float inputs (rounded already).
    let q_buf = upload(ctx, &encode(&b.query, dtype));
    let k_buf = upload(ctx, &encode(&b.key, dtype));
    let mut cache_buf = upload(ctx, &encode(&b.cache_init, dtype));
    let cos_buf = upload(ctx, &encode(&b.cos, dtype));
    let sin_buf = upload(ctx, &encode(&b.sin, dtype));

    // Integer index inputs.
    let cumseq_buf = upload(ctx, &i32_bytes(&b.cumseq));
    let past_buf = upload(ctx, &i32_bytes(&b.past));
    let block_buf = upload(ctx, &i32_bytes(&b.block_table));
    let slot_buf = upload(ctx, &i32_bytes(&b.slot_mapping));

    let out_len = b.token_count * nh * vhs;
    let mut out_buf = ctx.ep.allocate((out_len * elem).max(1), 256).unwrap();

    // Shapes.
    let q_shape = vec![b.token_count, nh * hs];
    let k_shape = vec![b.token_count, hs];
    let cache_shape = vec![
        b.num_rows / b.params.block_size as usize,
        b.params.block_size as usize,
        1usize,
        hs,
    ];
    let half = b.params.rotary_dim as usize / 2;
    let cos_shape = vec![b.max_pos.max(1), half.max(1)];
    let cumseq_shape = vec![batch + 1];
    let past_shape = vec![batch];
    let block_shape = vec![batch, b.max_blocks_per_seq];
    let slot_shape = vec![b.token_count];
    let out_shape = vec![b.token_count, nh * vhs];

    let q_str = compute_contiguous_strides(&q_shape);
    let k_str = compute_contiguous_strides(&k_shape);
    let cache_str = compute_contiguous_strides(&cache_shape);
    let cos_str = compute_contiguous_strides(&cos_shape);
    let cumseq_str = compute_contiguous_strides(&cumseq_shape);
    let past_str = compute_contiguous_strides(&past_shape);
    let block_str = compute_contiguous_strides(&block_shape);
    let slot_str = compute_contiguous_strides(&slot_shape);
    let out_str = compute_contiguous_strides(&out_shape);

    let cache_ptr = cache_buf.as_ptr();
    let cache_mut_ptr = cache_buf.as_mut_ptr();

    let inputs = vec![
        TensorView::new(DevicePtr(q_buf.as_ptr()), dtype, &q_shape, &q_str, ctx.dev),
        TensorView::new(DevicePtr(k_buf.as_ptr()), dtype, &k_shape, &k_str, ctx.dev),
        TensorView::absent(dtype),
        TensorView::new(
            DevicePtr(cache_ptr),
            dtype,
            &cache_shape,
            &cache_str,
            ctx.dev,
        ),
        TensorView::absent(dtype),
        TensorView::new(
            DevicePtr(cumseq_buf.as_ptr()),
            DataType::Int32,
            &cumseq_shape,
            &cumseq_str,
            ctx.dev,
        ),
        TensorView::new(
            DevicePtr(past_buf.as_ptr()),
            DataType::Int32,
            &past_shape,
            &past_str,
            ctx.dev,
        ),
        TensorView::new(
            DevicePtr(block_buf.as_ptr()),
            DataType::Int32,
            &block_shape,
            &block_str,
            ctx.dev,
        ),
        TensorView::new(
            DevicePtr(cos_buf.as_ptr()),
            dtype,
            &cos_shape,
            &cos_str,
            ctx.dev,
        ),
        TensorView::new(
            DevicePtr(sin_buf.as_ptr()),
            dtype,
            &cos_shape,
            &cos_str,
            ctx.dev,
        ),
        TensorView::new(
            DevicePtr(slot_buf.as_ptr()),
            DataType::Int32,
            &slot_shape,
            &slot_str,
            ctx.dev,
        ),
    ];

    let out_view = TensorMut::new(
        DevicePtrMut(out_buf.as_mut_ptr()),
        dtype,
        &out_shape,
        &out_str,
        ctx.dev,
    );
    let cache_out_view = TensorMut::new(
        DevicePtrMut(cache_mut_ptr),
        dtype,
        &cache_shape,
        &cache_str,
        ctx.dev,
    );
    let mut outputs = vec![out_view, cache_out_view];

    let node = build_node(b);
    let kernel = PagedAttentionFactory {
        runtime: ctx.runtime.clone(),
    }
    .create(&node, &[])
    .expect("kernel factory create");
    kernel
        .execute(&inputs, &mut outputs)
        .expect("kernel execute");

    // Read back.
    let mut out_bytes = vec![0u8; out_len * elem];
    unsafe {
        ctx.runtime
            .dtoh(&mut out_bytes, cuptr(out_buf.as_ptr()))
            .unwrap();
    }
    let mut cache_bytes = vec![0u8; b.cache_init.len() * elem];
    unsafe {
        ctx.runtime
            .dtoh(&mut cache_bytes, cuptr(cache_ptr))
            .unwrap();
    }

    ctx.ep.deallocate(q_buf).unwrap();
    ctx.ep.deallocate(k_buf).unwrap();
    ctx.ep.deallocate(cache_buf).unwrap();
    ctx.ep.deallocate(cos_buf).unwrap();
    ctx.ep.deallocate(sin_buf).unwrap();
    ctx.ep.deallocate(cumseq_buf).unwrap();
    ctx.ep.deallocate(past_buf).unwrap();
    ctx.ep.deallocate(block_buf).unwrap();
    ctx.ep.deallocate(slot_buf).unwrap();
    ctx.ep.deallocate(out_buf).unwrap();

    (decode(&out_bytes, dtype), decode(&cache_bytes, dtype))
}

fn tol(dtype: DataType) -> f32 {
    match dtype {
        DataType::Float16 => 3.0e-2,
        DataType::BFloat16 => 1.2e-1,
        _ => 1.0e-4,
    }
}

fn make_ctx() -> Option<Ctx> {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            return None;
        }
    };
    let runtime = ep.runtime().clone();
    let dev = ep.device_id();
    Some(Ctx { ep, runtime, dev })
}

fn check(ctx: &Ctx, s: &Scenario, label: &str) {
    let built = build(s);
    let (oref, cref) = run_oracle(&built);
    let (onat, cnat) = run_native(ctx, &built);
    let t = tol(s.dtype);
    let oerr = max_abs_err(&oref, &onat);
    let cerr = max_abs_err(&cref, &cnat);
    assert!(
        oerr <= t,
        "{label} [{:?}]: output max abs err {oerr} exceeds tol {t}",
        s.dtype
    );
    assert!(
        cerr <= t,
        "{label} [{:?}]: cache max abs err {cerr} exceeds tol {t}",
        s.dtype
    );
    println!(
        "{label} [{:?}]: output err {oerr:.2e}, cache err {cerr:.2e} (tol {t:.0e})",
        s.dtype
    );
}

fn glm_scale(head_size: usize) -> f32 {
    1.0f32 / (head_size as f32).sqrt()
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn latent_tiny_prefill_matches_oracle() {
    let Some(ctx) = make_ctx() else {
        panic!("CUDA test path did not run; report as failed GPU test, not a pass");
    };
    for dtype in [DataType::Float16, DataType::BFloat16] {
        let s = Scenario {
            dtype,
            num_heads: 2,
            head_size: 32,
            v_head_size: 16,
            block_size: 16,
            rotary_dim: 16,
            rotary_offset: 16,
            interleaved: false,
            scale: glm_scale(32),
            seqs: vec![(0, 3)],
            skip_tokens: vec![],
        };
        check(&ctx, &s, "tiny_prefill");
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn latent_tiny_decode_matches_oracle() {
    let Some(ctx) = make_ctx() else {
        panic!("CUDA test path did not run; report as failed GPU test, not a pass");
    };
    for dtype in [DataType::Float16, DataType::BFloat16] {
        let s = Scenario {
            dtype,
            num_heads: 2,
            head_size: 32,
            v_head_size: 16,
            block_size: 16,
            rotary_dim: 16,
            rotary_offset: 16,
            interleaved: false,
            scale: glm_scale(32),
            seqs: vec![(5, 1)],
            skip_tokens: vec![],
        };
        check(&ctx, &s, "tiny_decode");
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn latent_glm_dims_prefill_and_decode() {
    let Some(ctx) = make_ctx() else {
        panic!("CUDA test path did not run; report as failed GPU test, not a pass");
    };
    // GLM-5.2 dense MLA geometry: qk=192, v=128, partial RoPE (rotary_dim=64,
    // rotary_offset=128), explicit scale.
    for dtype in [DataType::Float16, DataType::BFloat16] {
        let base = |seqs: Vec<(usize, usize)>| Scenario {
            dtype,
            num_heads: 2,
            head_size: 192,
            v_head_size: 128,
            block_size: 16,
            rotary_dim: 64,
            rotary_offset: 128,
            interleaved: false,
            scale: glm_scale(192),
            seqs,
            skip_tokens: vec![],
        };
        // First token.
        check(&ctx, &base(vec![(0, 1)]), "glm_first_token");
        // Prefill several tokens.
        check(&ctx, &base(vec![(0, 5)]), "glm_prefill");
        // Decode with a populated cache.
        check(&ctx, &base(vec![(7, 1)]), "glm_decode");
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn latent_block_boundary_and_multi_request() {
    let Some(ctx) = make_ctx() else {
        panic!("CUDA test path did not run; report as failed GPU test, not a pass");
    };
    let dtype = DataType::Float16;
    // Sequence crossing a 16-token block boundary (prefill 18 tokens), plus a
    // second, shorter request in the same batch — exercises device isolation of
    // per-request block tables and context lengths.
    let s = Scenario {
        dtype,
        num_heads: 2,
        head_size: 64,
        v_head_size: 48,
        block_size: 16,
        rotary_dim: 32,
        rotary_offset: 32,
        interleaved: false,
        scale: glm_scale(64),
        seqs: vec![(0, 18), (3, 2)],
        skip_tokens: vec![],
    };
    check(&ctx, &s, "block_boundary_multi_request");
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn latent_slot_minus_one_skips_write() {
    let Some(ctx) = make_ctx() else {
        panic!("CUDA test path did not run; report as failed GPU test, not a pass");
    };
    let dtype = DataType::Float16;
    // Token index 1 (global) is marked slot -1: its latent K must NOT be written
    // to the cache, matching the oracle. The oracle honours slot_mapping == -1
    // the same way, so cache parity proves the skip.
    let s = Scenario {
        dtype,
        num_heads: 2,
        head_size: 32,
        v_head_size: 16,
        block_size: 16,
        rotary_dim: 16,
        rotary_offset: 16,
        interleaved: false,
        scale: glm_scale(32),
        seqs: vec![(0, 4)],
        skip_tokens: vec![1],
    };
    check(&ctx, &s, "slot_minus_one_skip");
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn latent_no_rotary_matches_oracle() {
    let Some(ctx) = make_ctx() else {
        panic!("CUDA test path did not run; report as failed GPU test, not a pass");
    };
    let dtype = DataType::Float16;
    // do_rotary = 0 path (rotary_dim = 0): pure absorbed MLA without RoPE.
    let s = Scenario {
        dtype,
        num_heads: 3,
        head_size: 40,
        v_head_size: 40,
        block_size: 16,
        rotary_dim: 0,
        rotary_offset: 0,
        interleaved: false,
        scale: glm_scale(40),
        seqs: vec![(2, 3)],
        skip_tokens: vec![],
    };
    check(&ctx, &s, "no_rotary");
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn latent_eager_equivalence_repeated_calls() {
    let Some(ctx) = make_ctx() else {
        panic!("CUDA test path did not run; report as failed GPU test, not a pass");
    };
    let dtype = DataType::Float16;
    let s = Scenario {
        dtype,
        num_heads: 2,
        head_size: 192,
        v_head_size: 128,
        block_size: 16,
        rotary_dim: 64,
        rotary_offset: 128,
        interleaved: false,
        scale: glm_scale(192),
        seqs: vec![(4, 2)],
        skip_tokens: vec![],
    };
    let built = build(&s);
    let (o0, c0) = run_native(&ctx, &built);
    // Repeated identical eager launches are bit-stable (deterministic kernel,
    // fresh buffers each call).
    for _ in 0..3 {
        let (o1, c1) = run_native(&ctx, &built);
        assert_eq!(o0, o1, "eager output not reproducible across calls");
        assert_eq!(c0, c1, "eager cache not reproducible across calls");
    }
    let (oref, cref) = run_oracle(&built);
    assert!(max_abs_err(&oref, &o0) <= tol(dtype));
    assert!(max_abs_err(&cref, &c0) <= tol(dtype));
    println!("eager equivalence OK on {:?}", ctx.dev);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn latent_capture_replay_matches_eager() {
    let Some(ctx) = make_ctx() else {
        panic!("CUDA test path did not run; report as failed GPU test, not a pass");
    };
    let dtype = DataType::Float16;
    // GLM-dim decode: the CUDA-graph-capturable shape for a single decode step.
    let s = Scenario {
        dtype,
        num_heads: 2,
        head_size: 192,
        v_head_size: 128,
        block_size: 16,
        rotary_dim: 64,
        rotary_offset: 128,
        interleaved: false,
        scale: glm_scale(192),
        seqs: vec![(6, 1)],
        skip_tokens: vec![],
    };
    let b = build(&s);
    let (oref, _cref) = run_oracle(&b);

    let hs = b.params.head_size as usize;
    let vhs = b.params.v_head_size as usize;
    let nh = b.params.num_heads as usize;
    let batch = b.params.batch_size as usize;
    let elem = 2usize;

    // Persistent device buffers (allocated once, before capture — capture must
    // not allocate or sync on the host).
    let q_buf = upload(&ctx, &encode(&b.query, dtype));
    let k_buf = upload(&ctx, &encode(&b.key, dtype));
    let mut cache_buf = upload(&ctx, &encode(&b.cache_init, dtype));
    let cos_buf = upload(&ctx, &encode(&b.cos, dtype));
    let sin_buf = upload(&ctx, &encode(&b.sin, dtype));
    let cumseq_buf = upload(&ctx, &i32_bytes(&b.cumseq));
    let past_buf = upload(&ctx, &i32_bytes(&b.past));
    let block_buf = upload(&ctx, &i32_bytes(&b.block_table));
    let slot_buf = upload(&ctx, &i32_bytes(&b.slot_mapping));
    let out_len = b.token_count * nh * vhs;
    let mut out_buf = ctx.ep.allocate((out_len * elem).max(1), 256).unwrap();

    let q_shape = vec![b.token_count, nh * hs];
    let k_shape = vec![b.token_count, hs];
    let cache_shape = vec![
        b.num_rows / b.params.block_size as usize,
        b.params.block_size as usize,
        1usize,
        hs,
    ];
    let half = b.params.rotary_dim as usize / 2;
    let cos_shape = vec![b.max_pos.max(1), half.max(1)];
    let cumseq_shape = vec![batch + 1];
    let past_shape = vec![batch];
    let block_shape = vec![batch, b.max_blocks_per_seq];
    let slot_shape = vec![b.token_count];
    let out_shape = vec![b.token_count, nh * vhs];

    let q_str = compute_contiguous_strides(&q_shape);
    let k_str = compute_contiguous_strides(&k_shape);
    let cache_str = compute_contiguous_strides(&cache_shape);
    let cos_str = compute_contiguous_strides(&cos_shape);
    let cumseq_str = compute_contiguous_strides(&cumseq_shape);
    let past_str = compute_contiguous_strides(&past_shape);
    let block_str = compute_contiguous_strides(&block_shape);
    let slot_str = compute_contiguous_strides(&slot_shape);
    let out_str = compute_contiguous_strides(&out_shape);

    let cache_ptr = cache_buf.as_ptr();
    let cache_mut_ptr = cache_buf.as_mut_ptr();
    let out_ptr = out_buf.as_ptr();
    let out_mut_ptr = out_buf.as_mut_ptr();

    let node = build_node(&b);
    let kernel = PagedAttentionFactory {
        runtime: ctx.runtime.clone(),
    }
    .create(&node, &[])
    .expect("kernel factory create");

    let make_inputs = || {
        vec![
            TensorView::new(DevicePtr(q_buf.as_ptr()), dtype, &q_shape, &q_str, ctx.dev),
            TensorView::new(DevicePtr(k_buf.as_ptr()), dtype, &k_shape, &k_str, ctx.dev),
            TensorView::absent(dtype),
            TensorView::new(
                DevicePtr(cache_ptr),
                dtype,
                &cache_shape,
                &cache_str,
                ctx.dev,
            ),
            TensorView::absent(dtype),
            TensorView::new(
                DevicePtr(cumseq_buf.as_ptr()),
                DataType::Int32,
                &cumseq_shape,
                &cumseq_str,
                ctx.dev,
            ),
            TensorView::new(
                DevicePtr(past_buf.as_ptr()),
                DataType::Int32,
                &past_shape,
                &past_str,
                ctx.dev,
            ),
            TensorView::new(
                DevicePtr(block_buf.as_ptr()),
                DataType::Int32,
                &block_shape,
                &block_str,
                ctx.dev,
            ),
            TensorView::new(
                DevicePtr(cos_buf.as_ptr()),
                dtype,
                &cos_shape,
                &cos_str,
                ctx.dev,
            ),
            TensorView::new(
                DevicePtr(sin_buf.as_ptr()),
                dtype,
                &cos_shape,
                &cos_str,
                ctx.dev,
            ),
            TensorView::new(
                DevicePtr(slot_buf.as_ptr()),
                DataType::Int32,
                &slot_shape,
                &slot_str,
                ctx.dev,
            ),
        ]
    };
    let read_out = || -> Vec<f32> {
        let mut bytes = vec![0u8; out_len * elem];
        unsafe {
            ctx.runtime.dtoh(&mut bytes, cuptr(out_ptr)).unwrap();
        }
        decode(&bytes, dtype)
    };

    // 1) Warm the exact signature eagerly.
    {
        let inputs = make_inputs();
        let mut outputs = vec![
            TensorMut::new(
                DevicePtrMut(out_mut_ptr),
                dtype,
                &out_shape,
                &out_str,
                ctx.dev,
            ),
            TensorMut::new(
                DevicePtrMut(cache_mut_ptr),
                dtype,
                &cache_shape,
                &cache_str,
                ctx.dev,
            ),
        ];
        kernel.execute(&inputs, &mut outputs).expect("warm execute");
    }
    assert!(
        kernel.cuda_graph_compatible(),
        "warmed LATENT PagedAttention must be capture-supported"
    );
    let eager = read_out();
    assert!(
        max_abs_err(&oref, &eager) <= tol(dtype),
        "warm eager output diverged from oracle"
    );

    // 2) Capture into a CUDA graph (no host alloc/sync inside execute).
    let kernels: [&dyn Kernel; 1] = [kernel.as_ref()];
    ctx.runtime
        .begin_graph_capture(&kernels)
        .expect("begin graph capture");
    {
        let inputs = make_inputs();
        let mut outputs = vec![
            TensorMut::new(
                DevicePtrMut(out_mut_ptr),
                dtype,
                &out_shape,
                &out_str,
                ctx.dev,
            ),
            TensorMut::new(
                DevicePtrMut(cache_mut_ptr),
                dtype,
                &cache_shape,
                &cache_str,
                ctx.dev,
            ),
        ];
        kernel
            .execute(&inputs, &mut outputs)
            .expect("capture execute");
    }
    ctx.runtime
        .end_graph_capture()
        .expect("LATENT PagedAttention must record without host fallback");
    assert!(
        ctx.runtime.has_graph_executable().unwrap(),
        "no CUDA graph installed"
    );

    // 3) Replay >= 3 times; every replay must match eager and latch no error.
    for r in 0..3 {
        ctx.runtime.replay_graph().expect("replay graph");
        let got = read_out();
        assert_eq!(got, eager, "captured replay {r} diverged from eager");
        assert_eq!(
            ctx.runtime.check_capture_error().unwrap(),
            0,
            "capture error latched on replay {r}"
        );
    }
    assert!(
        ctx.runtime.reset_graph().unwrap(),
        "graph was not installed"
    );
    println!("capture + 3 replays matched eager on {:?}", ctx.dev);

    for buf in [
        q_buf, k_buf, cache_buf, cos_buf, sin_buf, cumseq_buf, past_buf, block_buf, slot_buf,
        out_buf,
    ] {
        ctx.ep.deallocate(buf).unwrap();
    }
}

/// Shared device-side setup for the alias/measurement tests: owns every buffer
/// and shape/stride vector so views can be rebuilt cheaply per launch.
struct NativeRun {
    dtype: DataType,
    q_buf: onnx_runtime_ep_api::DeviceBuffer,
    k_buf: onnx_runtime_ep_api::DeviceBuffer,
    cache_buf: onnx_runtime_ep_api::DeviceBuffer,
    cos_buf: onnx_runtime_ep_api::DeviceBuffer,
    sin_buf: onnx_runtime_ep_api::DeviceBuffer,
    cumseq_buf: onnx_runtime_ep_api::DeviceBuffer,
    past_buf: onnx_runtime_ep_api::DeviceBuffer,
    block_buf: onnx_runtime_ep_api::DeviceBuffer,
    slot_buf: onnx_runtime_ep_api::DeviceBuffer,
    out_buf: onnx_runtime_ep_api::DeviceBuffer,
    q_shape: Vec<usize>,
    k_shape: Vec<usize>,
    cache_shape: Vec<usize>,
    cos_shape: Vec<usize>,
    cumseq_shape: Vec<usize>,
    past_shape: Vec<usize>,
    block_shape: Vec<usize>,
    slot_shape: Vec<usize>,
    out_shape: Vec<usize>,
}

impl NativeRun {
    fn new(ctx: &Ctx, b: &BuiltScenario) -> Self {
        let dtype = b.dtype;
        let hs = b.params.head_size as usize;
        let vhs = b.params.v_head_size as usize;
        let nh = b.params.num_heads as usize;
        let batch = b.params.batch_size as usize;
        let elem = 2usize;
        let half = b.params.rotary_dim as usize / 2;
        NativeRun {
            dtype,
            q_buf: upload(ctx, &encode(&b.query, dtype)),
            k_buf: upload(ctx, &encode(&b.key, dtype)),
            cache_buf: upload(ctx, &encode(&b.cache_init, dtype)),
            cos_buf: upload(ctx, &encode(&b.cos, dtype)),
            sin_buf: upload(ctx, &encode(&b.sin, dtype)),
            cumseq_buf: upload(ctx, &i32_bytes(&b.cumseq)),
            past_buf: upload(ctx, &i32_bytes(&b.past)),
            block_buf: upload(ctx, &i32_bytes(&b.block_table)),
            slot_buf: upload(ctx, &i32_bytes(&b.slot_mapping)),
            out_buf: ctx
                .ep
                .allocate((b.token_count * nh * vhs * elem).max(1), 256)
                .unwrap(),
            q_shape: vec![b.token_count, nh * hs],
            k_shape: vec![b.token_count, hs],
            cache_shape: vec![
                b.num_rows / b.params.block_size as usize,
                b.params.block_size as usize,
                1,
                hs,
            ],
            cos_shape: vec![b.max_pos.max(1), half.max(1)],
            cumseq_shape: vec![batch + 1],
            past_shape: vec![batch],
            block_shape: vec![batch, b.max_blocks_per_seq],
            slot_shape: vec![b.token_count],
            out_shape: vec![b.token_count, nh * vhs],
        }
    }

    fn inputs<'a>(&'a self, ctx: &Ctx, strides: &'a Strides) -> Vec<TensorView<'a>> {
        let d = self.dtype;
        vec![
            TensorView::new(
                DevicePtr(self.q_buf.as_ptr()),
                d,
                &self.q_shape,
                &strides.q,
                ctx.dev,
            ),
            TensorView::new(
                DevicePtr(self.k_buf.as_ptr()),
                d,
                &self.k_shape,
                &strides.k,
                ctx.dev,
            ),
            TensorView::absent(d),
            TensorView::new(
                DevicePtr(self.cache_buf.as_ptr()),
                d,
                &self.cache_shape,
                &strides.cache,
                ctx.dev,
            ),
            TensorView::absent(d),
            TensorView::new(
                DevicePtr(self.cumseq_buf.as_ptr()),
                DataType::Int32,
                &self.cumseq_shape,
                &strides.cumseq,
                ctx.dev,
            ),
            TensorView::new(
                DevicePtr(self.past_buf.as_ptr()),
                DataType::Int32,
                &self.past_shape,
                &strides.past,
                ctx.dev,
            ),
            TensorView::new(
                DevicePtr(self.block_buf.as_ptr()),
                DataType::Int32,
                &self.block_shape,
                &strides.block,
                ctx.dev,
            ),
            TensorView::new(
                DevicePtr(self.cos_buf.as_ptr()),
                d,
                &self.cos_shape,
                &strides.cos,
                ctx.dev,
            ),
            TensorView::new(
                DevicePtr(self.sin_buf.as_ptr()),
                d,
                &self.cos_shape,
                &strides.cos,
                ctx.dev,
            ),
            TensorView::new(
                DevicePtr(self.slot_buf.as_ptr()),
                DataType::Int32,
                &self.slot_shape,
                &strides.slot,
                ctx.dev,
            ),
        ]
    }

    fn free(self, ctx: &Ctx) {
        for buf in [
            self.q_buf,
            self.k_buf,
            self.cache_buf,
            self.cos_buf,
            self.sin_buf,
            self.cumseq_buf,
            self.past_buf,
            self.block_buf,
            self.slot_buf,
            self.out_buf,
        ] {
            ctx.ep.deallocate(buf).unwrap();
        }
    }
}

struct Strides {
    q: Vec<i64>,
    k: Vec<i64>,
    cache: Vec<i64>,
    cos: Vec<i64>,
    cumseq: Vec<i64>,
    past: Vec<i64>,
    block: Vec<i64>,
    slot: Vec<i64>,
    out: Vec<i64>,
}

impl Strides {
    fn new(r: &NativeRun) -> Self {
        Strides {
            q: compute_contiguous_strides(&r.q_shape),
            k: compute_contiguous_strides(&r.k_shape),
            cache: compute_contiguous_strides(&r.cache_shape),
            cos: compute_contiguous_strides(&r.cos_shape),
            cumseq: compute_contiguous_strides(&r.cumseq_shape),
            past: compute_contiguous_strides(&r.past_shape),
            block: compute_contiguous_strides(&r.block_shape),
            slot: compute_contiguous_strides(&r.slot_shape),
            out: compute_contiguous_strides(&r.out_shape),
        }
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn latent_rejects_non_aliased_cache_out_and_missing_inputs() {
    let Some(ctx) = make_ctx() else {
        panic!("CUDA test path did not run; report as failed GPU test, not a pass");
    };
    let dtype = DataType::Float16;
    let s = Scenario {
        dtype,
        num_heads: 2,
        head_size: 32,
        v_head_size: 16,
        block_size: 16,
        rotary_dim: 16,
        rotary_offset: 16,
        interleaved: false,
        scale: glm_scale(32),
        seqs: vec![(0, 3)],
        skip_tokens: vec![],
    };
    let b = build(&s);
    let mut run = NativeRun::new(&ctx, &b);
    let strides = Strides::new(&run);
    let node = build_node(&b);
    let kernel = PagedAttentionFactory {
        runtime: ctx.runtime.clone(),
    }
    .create(&node, &[])
    .unwrap();

    let out_mut_ptr = run.out_buf.as_mut_ptr();
    // A separate buffer for key_cache_out that does NOT alias key_cache input.
    let mut bogus = ctx
        .ep
        .allocate(run.cache_shape.iter().product::<usize>() * 2, 256)
        .unwrap();
    let bogus_ptr = bogus.as_mut_ptr();

    // 1) Non-aliased key_cache_out must be rejected (in-place contract).
    {
        let inputs = run.inputs(&ctx, &strides);
        let mut outputs = vec![
            TensorMut::new(
                DevicePtrMut(out_mut_ptr),
                dtype,
                &run.out_shape,
                &strides.out,
                ctx.dev,
            ),
            TensorMut::new(
                DevicePtrMut(bogus_ptr),
                dtype,
                &run.cache_shape,
                &strides.cache,
                ctx.dev,
            ),
        ];
        let err = kernel.execute(&inputs, &mut outputs).unwrap_err();
        assert!(
            format!("{err}").contains("alias"),
            "expected alias error, got: {err}"
        );
    }

    // 2) Missing required input (block_table absent) must be rejected.
    {
        let mut inputs = run.inputs(&ctx, &strides);
        inputs[7] = TensorView::absent(DataType::Int32);
        let mut outputs = vec![TensorMut::new(
            DevicePtrMut(out_mut_ptr),
            dtype,
            &run.out_shape,
            &strides.out,
            ctx.dev,
        )];
        let err = kernel.execute(&inputs, &mut outputs).unwrap_err();
        assert!(
            format!("{err}").contains("block_table"),
            "expected block_table error, got: {err}"
        );
    }

    // 3) Geometry mismatch: a num_heads attribute that disagrees with the query
    //    hidden dim must be a typed error, not an out-of-bounds device read.
    {
        let mut bad_node = build_node(&b);
        bad_node.attributes.insert(
            "num_heads".to_string(),
            Attribute::Int(s.num_heads as i64 + 1),
        );
        let bad_kernel = PagedAttentionFactory {
            runtime: ctx.runtime.clone(),
        }
        .create(&bad_node, &[])
        .unwrap();
        let inputs = run.inputs(&ctx, &strides);
        let mut outputs = vec![TensorMut::new(
            DevicePtrMut(out_mut_ptr),
            dtype,
            &run.out_shape,
            &strides.out,
            ctx.dev,
        )];
        let err = bad_kernel.execute(&inputs, &mut outputs).unwrap_err();
        assert!(
            format!("{err}").contains("num_heads"),
            "expected num_heads geometry error, got: {err}"
        );
    }

    ctx.ep.deallocate(bogus).unwrap();
    run.free(&ctx);
    println!("alias + missing-input + geometry rejections OK");
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn latent_measure_prefill_decode_cuda_events() {
    use cudarc::driver::sys::CUevent_flags;

    let Some(ctx) = make_ctx() else {
        panic!("CUDA test path did not run; report as failed GPU test, not a pass");
    };
    let dtype = DataType::Float16;
    let iters = 5usize; // n >= 3

    let time_case = |label: &str, seqs: Vec<(usize, usize)>| {
        let s = Scenario {
            dtype,
            num_heads: 2,
            head_size: 192,
            v_head_size: 128,
            block_size: 16,
            rotary_dim: 64,
            rotary_offset: 128,
            interleaved: false,
            scale: glm_scale(192),
            seqs,
            skip_tokens: vec![],
        };
        let b = build(&s);
        let mut run = NativeRun::new(&ctx, &b);
        let strides = Strides::new(&run);
        let out_mut_ptr = run.out_buf.as_mut_ptr();
        let cache_mut_ptr = run.cache_buf.as_mut_ptr();
        let node = build_node(&b);
        let kernel = PagedAttentionFactory {
            runtime: ctx.runtime.clone(),
        }
        .create(&node, &[])
        .unwrap();

        let launch = |kernel: &dyn Kernel| {
            let inputs = run.inputs(&ctx, &strides);
            let mut outputs = vec![
                TensorMut::new(
                    DevicePtrMut(out_mut_ptr),
                    dtype,
                    &run.out_shape,
                    &strides.out,
                    ctx.dev,
                ),
                TensorMut::new(
                    DevicePtrMut(cache_mut_ptr),
                    dtype,
                    &run.cache_shape,
                    &strides.cache,
                    ctx.dev,
                ),
            ];
            kernel.execute(&inputs, &mut outputs).unwrap();
        };

        // Warm (NVRTC compile + allocator/library state) before timing.
        launch(kernel.as_ref());
        ctx.runtime.synchronize().unwrap();

        let ctxt = ctx.runtime.cuda_context();
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = ctxt
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap();
            let end = ctxt
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap();
            start.record(ctx.runtime.stream()).unwrap();
            launch(kernel.as_ref());
            end.record(ctx.runtime.stream()).unwrap();
            let ms = start.elapsed_ms(&end).unwrap();
            samples.push(ms);
        }
        samples.sort_by(f32::total_cmp);
        let cache_bytes = b.num_rows * b.params.head_size as usize * 2;
        println!(
            "{label} [f16 GLM qk=192 v=128]: kernel event ms n={iters} min={:.4} med={:.4} max={:.4}; \
             cache={} rows ({:.2} MiB, op-side alloc=0 — aliases caller buffers)",
            samples[0],
            samples[iters / 2],
            samples[iters - 1],
            b.num_rows,
            cache_bytes as f64 / (1024.0 * 1024.0),
        );
        run.free(&ctx);
    };

    // Prefill (16 new tokens) and decode (1 new token over a populated cache).
    time_case("prefill", vec![(0, 16)]);
    time_case("decode", vec![(31, 1)]);
    println!(
        "measurement is a tiny-shape correctness gate on {:?}; NO full-size performance claim.",
        ctx.dev
    );
}
