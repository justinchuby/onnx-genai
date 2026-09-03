//! GPU parity oracle for the seq-major (BSNH) KV cache capability.
//!
//! The native backend may store its KV cache seq-major (`[batch, capacity,
//! kv_heads, head_dim]`) instead of the head-major BNSH (`[batch, kv_heads,
//! capacity, head_dim]`) layout that ORT requires. Only the fused fp16
//! single-token decode pair is converted in this increment: the fused
//! decode-prep append write and the split-K fp16 decode read. This test drives
//! that exact pair through the real CUDA kernels for one decode step under both
//! layouts with identical logical content and asserts the attention output is
//! **bit-identical**, and that a head-major run matches an independent CPU
//! reference so the two GPU runs cannot be symmetrically wrong.
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::uninlined_format_args,
    clippy::manual_is_multiple_of
)]
use half::f16;
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::{
    CudaExecutionProvider, GroupQueryAttentionBackend, GroupQueryAttentionKernel,
};
use onnx_runtime_ir::{DataType, compute_contiguous_strides};

const BATCH: usize = 1;
const QUERY_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const CACHE_CAPACITY: usize = 64;
const PAST_LEN: usize = 10;
const GROUP: usize = QUERY_HEADS / KV_HEADS;

fn typed_bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: test data contains plain-old-data values with no padding.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn require_cuda() -> CudaExecutionProvider {
    match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        Ok(Err(error)) => panic!(
            "CUDA test requires CUDA device/runtime; CPU-only runs must leave this test ignored: {error}"
        ),
        Err(_) => panic!(
            "CUDA test requires CUDA runtime libraries; CPU-only runs must leave this test ignored"
        ),
    }
}

fn upload(ep: &CudaExecutionProvider, bytes: &[u8]) -> onnx_runtime_ep_api::Result<DeviceBuffer> {
    let buffer = ep.allocate(bytes.len().max(1), 256)?;
    if !bytes.is_empty() {
        // SAFETY: the allocation is at least `bytes.len()` bytes.
        unsafe {
            ep.runtime().htod(bytes, cuptr(buffer.as_ptr()))?;
        }
    }
    Ok(buffer)
}

fn read(
    ep: &CudaExecutionProvider,
    buffer: &DeviceBuffer,
    bytes: usize,
) -> onnx_runtime_ep_api::Result<Vec<u8>> {
    let mut host = vec![0_u8; bytes];
    // SAFETY: callers request exactly the initialized tensor extent.
    unsafe {
        ep.runtime().dtoh(&mut host, cuptr(buffer.as_ptr()))?;
    }
    Ok(host)
}

/// Logical past/appended K or V, indexed `[kv_head][position][dim]`, laid out
/// into a flat cache buffer. `seq_major == false` gives BNSH
/// `((h*capacity)+t)*head_dim+d`; `true` gives BSNH `((t*kv_heads)+h)*head_dim+d`.
fn seed_cache(logical: &[Vec<Vec<f16>>], seq_major: bool) -> Vec<f16> {
    let mut buffer = vec![f16::ZERO; KV_HEADS * CACHE_CAPACITY * HEAD_DIM];
    for h in 0..KV_HEADS {
        for t in 0..logical[h].len() {
            for d in 0..HEAD_DIM {
                let index = if seq_major {
                    (t * KV_HEADS + h) * HEAD_DIM + d
                } else {
                    (h * CACHE_CAPACITY + t) * HEAD_DIM + d
                };
                buffer[index] = logical[h][t][d];
            }
        }
    }
    buffer
}

fn cpu_reference(
    query: &[f16],
    key: &[Vec<Vec<f16>>],
    value: &[Vec<Vec<f16>>],
    scale: f32,
) -> Vec<f32> {
    // Softmax attention over positions 0..=PAST_LEN (past plus the appended
    // token) per query head; GQA maps query head `qh` to kv head `qh / GROUP`.
    let mut out = vec![0.0_f32; QUERY_HEADS * HEAD_DIM];
    let valid = PAST_LEN + 1;
    for qh in 0..QUERY_HEADS {
        let kvh = qh / GROUP;
        let mut scores = vec![0.0_f32; valid];
        let mut max_score = f32::NEG_INFINITY;
        for p in 0..valid {
            let mut dot = 0.0_f32;
            for d in 0..HEAD_DIM {
                dot += query[qh * HEAD_DIM + d].to_f32() * key[kvh][p][d].to_f32();
            }
            let score = dot * scale;
            scores[p] = score;
            max_score = max_score.max(score);
        }
        let mut denom = 0.0_f32;
        for p in 0..valid {
            scores[p] = (scores[p] - max_score).exp();
            denom += scores[p];
        }
        for d in 0..HEAD_DIM {
            let mut acc = 0.0_f32;
            for p in 0..valid {
                acc += scores[p] / denom * value[kvh][p][d].to_f32();
            }
            out[qh * HEAD_DIM + d] = acc;
        }
    }
    out
}

fn cpu_prefill_reference(
    query: &[f16],
    key: &[Vec<Vec<f16>>],
    value: &[Vec<Vec<f16>>],
    sequence_length: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; sequence_length * QUERY_HEADS * HEAD_DIM];
    for query_pos in 0..sequence_length {
        for qh in 0..QUERY_HEADS {
            let kvh = qh / GROUP;
            let mut scores = vec![0.0_f32; query_pos + 1];
            let mut max_score = f32::NEG_INFINITY;
            for key_pos in 0..=query_pos {
                let mut dot = 0.0_f32;
                for d in 0..HEAD_DIM {
                    let q_index = (query_pos * QUERY_HEADS + qh) * HEAD_DIM + d;
                    dot += query[q_index].to_f32() * key[kvh][key_pos][d].to_f32();
                }
                scores[key_pos] = dot * scale;
                max_score = max_score.max(scores[key_pos]);
            }
            let mut denominator = 0.0_f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                denominator += *score;
            }
            for d in 0..HEAD_DIM {
                let mut acc = 0.0_f32;
                for key_pos in 0..=query_pos {
                    acc += scores[key_pos] / denominator * value[kvh][key_pos][d].to_f32();
                }
                out[(query_pos * QUERY_HEADS + qh) * HEAD_DIM + d] = acc;
            }
        }
    }
    out
}

fn fp16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn seq_major_decode_is_bit_identical_to_head_major() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let scale = 1.0_f32;

    // Deterministic query and logical past/new K/V. Positions 0..PAST_LEN are
    // the shared past; index PAST_LEN is the token this step appends.
    let query: Vec<f16> = (0..QUERY_HEADS * HEAD_DIM)
        .map(|i| f16::from_f32(((i * 17 % 97) as f32 - 48.0) / 256.0))
        .collect();
    let mut key: Vec<Vec<Vec<f16>>> = vec![vec![vec![f16::ZERO; HEAD_DIM]; PAST_LEN + 1]; KV_HEADS];
    let mut value: Vec<Vec<Vec<f16>>> =
        vec![vec![vec![f16::ZERO; HEAD_DIM]; PAST_LEN + 1]; KV_HEADS];
    for h in 0..KV_HEADS {
        for t in 0..=PAST_LEN {
            for d in 0..HEAD_DIM {
                let base = ((h * 131 + t * 13 + d * 7) % 101) as f32;
                key[h][t][d] = f16::from_f32((base - 50.0) / 256.0);
                let vbase = ((h * 29 + t * 19 + d * 3) % 113) as f32;
                value[h][t][d] = f16::from_f32((vbase - 56.0) / 128.0);
            }
        }
    }

    // The new token (position PAST_LEN) is fed as the current K/V input; the
    // past-only view seeds the cache (positions 0..PAST_LEN-1).
    let mut current_key: Vec<f16> = vec![f16::ZERO; KV_HEADS * HEAD_DIM];
    let mut current_value: Vec<f16> = vec![f16::ZERO; KV_HEADS * HEAD_DIM];
    for h in 0..KV_HEADS {
        for d in 0..HEAD_DIM {
            current_key[h * HEAD_DIM + d] = key[h][PAST_LEN][d];
            current_value[h * HEAD_DIM + d] = value[h][PAST_LEN][d];
        }
    }
    let past_key: Vec<Vec<Vec<f16>>> = key.iter().map(|h| h[..PAST_LEN].to_vec()).collect();
    let past_value: Vec<Vec<Vec<f16>>> = value.iter().map(|h| h[..PAST_LEN].to_vec()).collect();

    // seqlens_k[0] = past length; the kernel derives total = past + 1 and writes
    // the appended token at index `past`.
    let seqlens = [PAST_LEN as i32];
    let total = [CACHE_CAPACITY as i32];

    let query_shape = [BATCH, 1, QUERY_HEADS * HEAD_DIM];
    let current_shape = [BATCH, 1, KV_HEADS * HEAD_DIM];
    let cache_shape = [BATCH, KV_HEADS, CACHE_CAPACITY, HEAD_DIM];
    let seqlens_shape = [BATCH];
    let scalar_shape: [usize; 0] = [];
    let query_strides = compute_contiguous_strides(&query_shape);
    let current_strides = compute_contiguous_strides(&current_shape);
    let cache_strides = compute_contiguous_strides(&cache_shape);
    let seqlens_strides = compute_contiguous_strides(&seqlens_shape);
    let scalar_strides = compute_contiguous_strides(&scalar_shape);
    let output_shape = query_shape;
    let output_strides = compute_contiguous_strides(&output_shape);
    let device = ep.device_id();

    let query_buffer = upload(&ep, typed_bytes(&query)).unwrap();
    let current_key_buffer = upload(&ep, typed_bytes(&current_key)).unwrap();
    let current_value_buffer = upload(&ep, typed_bytes(&current_value)).unwrap();
    let seqlens_buffer = upload(&ep, typed_bytes(&seqlens)).unwrap();
    let total_buffer = upload(&ep, typed_bytes(&total)).unwrap();

    // Runs one fused fp16 decode step (append + read) for the given layout,
    // returning the fp16 output bytes and the resulting cache bytes.
    let run_layout = |seq_major: bool, capture: bool| -> (Vec<u8>, Vec<u8>) {
        let kernel = GroupQueryAttentionKernel::new(
            runtime.clone(),
            QUERY_HEADS,
            KV_HEADS,
            Some(scale),
            false, // do_rotary
            false,
            -1, // local_window disabled
            0.0,
        )
        .unwrap()
        .with_kv_layout(if seq_major { 1 } else { 0 });

        let seeded_key = seed_cache(&past_key, seq_major);
        let seeded_value = seed_cache(&past_value, seq_major);
        let mut key_buffer = upload(&ep, typed_bytes(&seeded_key)).unwrap();
        let mut value_buffer = upload(&ep, typed_bytes(&seeded_value)).unwrap();
        let mut output_buffer = ep
            .allocate(QUERY_HEADS * HEAD_DIM * std::mem::size_of::<f16>(), 256)
            .unwrap();

        {
            let inputs = [
                TensorView::new(
                    DevicePtr(query_buffer.as_ptr()),
                    DataType::Float16,
                    &query_shape,
                    &query_strides,
                    device,
                ),
                TensorView::new(
                    DevicePtr(current_key_buffer.as_ptr()),
                    DataType::Float16,
                    &current_shape,
                    &current_strides,
                    device,
                ),
                TensorView::new(
                    DevicePtr(current_value_buffer.as_ptr()),
                    DataType::Float16,
                    &current_shape,
                    &current_strides,
                    device,
                ),
                TensorView::new(
                    DevicePtr(key_buffer.as_ptr()),
                    DataType::Float16,
                    &cache_shape,
                    &cache_strides,
                    device,
                ),
                TensorView::new(
                    DevicePtr(value_buffer.as_ptr()),
                    DataType::Float16,
                    &cache_shape,
                    &cache_strides,
                    device,
                ),
                TensorView::new(
                    DevicePtr(seqlens_buffer.as_ptr()),
                    DataType::Int32,
                    &seqlens_shape,
                    &seqlens_strides,
                    device,
                ),
                TensorView::new(
                    DevicePtr(total_buffer.as_ptr()),
                    DataType::Int32,
                    &scalar_shape,
                    &scalar_strides,
                    device,
                ),
            ];
            let mut outputs = [
                TensorMut::new(
                    DevicePtrMut(output_buffer.as_mut_ptr()),
                    DataType::Float16,
                    &output_shape,
                    &output_strides,
                    device,
                ),
                TensorMut::new(
                    DevicePtrMut(key_buffer.as_mut_ptr()),
                    DataType::Float16,
                    &cache_shape,
                    &cache_strides,
                    device,
                ),
                TensorMut::new(
                    DevicePtrMut(value_buffer.as_mut_ptr()),
                    DataType::Float16,
                    &cache_shape,
                    &cache_strides,
                    device,
                ),
            ];
            kernel.execute(&inputs, &mut outputs).unwrap();
            if capture {
                // Prove the fixed-stride seq-major decode records and replays
                // inside a CUDA graph: warm up above, capture one step, replay.
                runtime
                    .begin_graph_capture(&[&kernel as &dyn onnx_runtime_ep_api::Kernel])
                    .unwrap();
                kernel.execute(&inputs, &mut outputs).unwrap();
                runtime.end_graph_capture().unwrap();
                runtime.replay_graph().unwrap();
                runtime.replay_graph().unwrap();
                runtime.reset_graph().unwrap();
            }
        }

        let out_bytes = read(
            &ep,
            &output_buffer,
            QUERY_HEADS * HEAD_DIM * std::mem::size_of::<f16>(),
        )
        .unwrap();
        let cache_bytes = read(&ep, &key_buffer, typed_bytes(&seeded_key).len()).unwrap();
        ep.deallocate(output_buffer).unwrap();
        ep.deallocate(value_buffer).unwrap();
        ep.deallocate(key_buffer).unwrap();
        (out_bytes, cache_bytes)
    };

    let (bnsh_out, bnsh_cache) = run_layout(false, false);
    let (bsnh_out, bsnh_cache) = run_layout(true, false);
    let (bsnh_captured_out, _) = run_layout(true, true);

    assert_eq!(
        bnsh_out, bsnh_out,
        "seq-major (BSNH) decode output must be byte-identical to head-major (BNSH)"
    );
    assert_eq!(
        bsnh_out, bsnh_captured_out,
        "seq-major decode under CUDA-graph capture must match the eager (capture-off) output"
    );

    // The appended token must land at the layout-correct slot in each cache.
    for h in 0..KV_HEADS {
        for d in 0..HEAD_DIM {
            let bnsh_index = (h * CACHE_CAPACITY + PAST_LEN) * HEAD_DIM + d;
            let bsnh_index = (PAST_LEN * KV_HEADS + h) * HEAD_DIM + d;
            let bnsh_val = f16::from_bits(u16::from_ne_bytes([
                bnsh_cache[bnsh_index * 2],
                bnsh_cache[bnsh_index * 2 + 1],
            ]));
            let bsnh_val = f16::from_bits(u16::from_ne_bytes([
                bsnh_cache[bsnh_index * 2],
                bsnh_cache[bsnh_index * 2 + 1],
            ]));
            assert_eq!(
                bnsh_val,
                current_key[h * HEAD_DIM + d],
                "head-major append wrote wrong slot at h={h} d={d}"
            );
            assert_eq!(
                bsnh_val,
                current_key[h * HEAD_DIM + d],
                "seq-major append wrote wrong slot at h={h} d={d}"
            );
        }
    }

    // Independent CPU oracle so the two GPU layouts cannot be symmetrically wrong.
    let got = fp16_values(&bnsh_out);
    let expected = cpu_reference(&query, &key, &value, scale);
    let max_err = got
        .iter()
        .zip(&expected)
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 3e-3,
        "head-major GPU decode diverged from CPU oracle: max_abs={max_err:e}"
    );

    for buffer in [
        total_buffer,
        seqlens_buffer,
        current_value_buffer,
        current_key_buffer,
        query_buffer,
    ] {
        ep.deallocate(buffer).unwrap();
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn seq_major_full_generation_prefill_and_decode_is_bit_identical() {
    const PREFILL_LEN: usize = 10;

    let ep = require_cuda();
    let runtime = ep.runtime();
    let scale = 1.0_f32;
    let device = ep.device_id();

    let prefill_query: Vec<f16> = (0..PREFILL_LEN * QUERY_HEADS * HEAD_DIM)
        .map(|i| f16::from_f32(((i * 17 % 97) as f32 - 48.0) / 256.0))
        .collect();
    let decode_query: Vec<f16> = (0..QUERY_HEADS * HEAD_DIM)
        .map(|i| f16::from_f32(((i * 23 % 89) as f32 - 44.0) / 256.0))
        .collect();
    let mut key = vec![vec![vec![f16::ZERO; HEAD_DIM]; PREFILL_LEN + 1]; KV_HEADS];
    let mut value = vec![vec![vec![f16::ZERO; HEAD_DIM]; PREFILL_LEN + 1]; KV_HEADS];
    for h in 0..KV_HEADS {
        for t in 0..=PREFILL_LEN {
            for d in 0..HEAD_DIM {
                key[h][t][d] =
                    f16::from_f32((((h * 131 + t * 13 + d * 7) % 101) as f32 - 50.0) / 256.0);
                value[h][t][d] =
                    f16::from_f32((((h * 29 + t * 19 + d * 3) % 113) as f32 - 56.0) / 128.0);
            }
        }
    }
    let flatten_current = |position: usize| {
        let mut values = vec![f16::ZERO; KV_HEADS * HEAD_DIM];
        for h in 0..KV_HEADS {
            for d in 0..HEAD_DIM {
                values[h * HEAD_DIM + d] = key[h][position][d];
            }
        }
        values
    };
    let flatten_prefill = |logical: &[Vec<Vec<f16>>]| {
        let mut values = vec![f16::ZERO; PREFILL_LEN * KV_HEADS * HEAD_DIM];
        for t in 0..PREFILL_LEN {
            for h in 0..KV_HEADS {
                for d in 0..HEAD_DIM {
                    values[(t * KV_HEADS + h) * HEAD_DIM + d] = logical[h][t][d];
                }
            }
        }
        values
    };
    let prefill_key = flatten_prefill(&key);
    let prefill_value = flatten_prefill(&value);
    let decode_key = flatten_current(PREFILL_LEN);
    let decode_value = {
        let mut values = vec![f16::ZERO; KV_HEADS * HEAD_DIM];
        for h in 0..KV_HEADS {
            for d in 0..HEAD_DIM {
                values[h * HEAD_DIM + d] = value[h][PREFILL_LEN][d];
            }
        }
        values
    };

    let prefill_query_shape = [BATCH, PREFILL_LEN, QUERY_HEADS * HEAD_DIM];
    let prefill_current_shape = [BATCH, PREFILL_LEN, KV_HEADS * HEAD_DIM];
    let decode_query_shape = [BATCH, 1, QUERY_HEADS * HEAD_DIM];
    let decode_current_shape = [BATCH, 1, KV_HEADS * HEAD_DIM];
    let cache_shape = [BATCH, KV_HEADS, CACHE_CAPACITY, HEAD_DIM];
    let seqlens_shape = [BATCH];
    let scalar_shape: [usize; 0] = [];
    let prefill_query_strides = compute_contiguous_strides(&prefill_query_shape);
    let prefill_current_strides = compute_contiguous_strides(&prefill_current_shape);
    let cache_strides = compute_contiguous_strides(&cache_shape);
    let seqlens_strides = compute_contiguous_strides(&seqlens_shape);
    let scalar_strides = compute_contiguous_strides(&scalar_shape);

    let prefill_query_buffer = upload(&ep, typed_bytes(&prefill_query)).unwrap();
    let prefill_key_buffer = upload(&ep, typed_bytes(&prefill_key)).unwrap();
    let prefill_value_buffer = upload(&ep, typed_bytes(&prefill_value)).unwrap();
    let decode_query_buffer = upload(&ep, typed_bytes(&decode_query)).unwrap();
    let decode_key_buffer = upload(&ep, typed_bytes(&decode_key)).unwrap();
    let decode_value_buffer = upload(&ep, typed_bytes(&decode_value)).unwrap();
    let prefill_seqlens = [(PREFILL_LEN - 1) as i32];
    let decode_seqlens = [PREFILL_LEN as i32];
    let capacity = [CACHE_CAPACITY as i32];
    let prefill_seqlens_buffer = upload(&ep, typed_bytes(&prefill_seqlens)).unwrap();
    let decode_seqlens_buffer = upload(&ep, typed_bytes(&decode_seqlens)).unwrap();
    let capacity_buffer = upload(&ep, typed_bytes(&capacity)).unwrap();

    let run_layout = |seq_major: bool, capture: bool| -> (Vec<u8>, Vec<u8>) {
        let kernel = GroupQueryAttentionKernel::new(
            runtime.clone(),
            QUERY_HEADS,
            KV_HEADS,
            Some(scale),
            false,
            false,
            -1,
            0.0,
        )
        .unwrap()
        .with_backend(GroupQueryAttentionBackend::Fused)
        .with_kv_layout(if seq_major { 1 } else { 0 });
        let mut cache_key = ep
            .allocate(
                BATCH * KV_HEADS * CACHE_CAPACITY * HEAD_DIM * std::mem::size_of::<f16>(),
                256,
            )
            .unwrap();
        let mut cache_value = ep
            .allocate(
                BATCH * KV_HEADS * CACHE_CAPACITY * HEAD_DIM * std::mem::size_of::<f16>(),
                256,
            )
            .unwrap();
        let mut prefill_output = ep
            .allocate(
                PREFILL_LEN * QUERY_HEADS * HEAD_DIM * std::mem::size_of::<f16>(),
                256,
            )
            .unwrap();
        let mut decode_output = ep
            .allocate(QUERY_HEADS * HEAD_DIM * std::mem::size_of::<f16>(), 256)
            .unwrap();

        {
            let prefill_inputs = [
                TensorView::new(
                    DevicePtr(prefill_query_buffer.as_ptr()),
                    DataType::Float16,
                    &prefill_query_shape,
                    &prefill_query_strides,
                    device,
                ),
                TensorView::new(
                    DevicePtr(prefill_key_buffer.as_ptr()),
                    DataType::Float16,
                    &prefill_current_shape,
                    &prefill_current_strides,
                    device,
                ),
                TensorView::new(
                    DevicePtr(prefill_value_buffer.as_ptr()),
                    DataType::Float16,
                    &prefill_current_shape,
                    &prefill_current_strides,
                    device,
                ),
                TensorView::absent(DataType::Float16),
                TensorView::absent(DataType::Float16),
                TensorView::new(
                    DevicePtr(prefill_seqlens_buffer.as_ptr()),
                    DataType::Int32,
                    &seqlens_shape,
                    &seqlens_strides,
                    device,
                ),
                TensorView::new(
                    DevicePtr(capacity_buffer.as_ptr()),
                    DataType::Int32,
                    &scalar_shape,
                    &scalar_strides,
                    device,
                ),
            ];
            let mut prefill_outputs = [
                TensorMut::new(
                    DevicePtrMut(prefill_output.as_mut_ptr()),
                    DataType::Float16,
                    &prefill_query_shape,
                    &prefill_query_strides,
                    device,
                ),
                TensorMut::new(
                    DevicePtrMut(cache_key.as_mut_ptr()),
                    DataType::Float16,
                    &cache_shape,
                    &cache_strides,
                    device,
                ),
                TensorMut::new(
                    DevicePtrMut(cache_value.as_mut_ptr()),
                    DataType::Float16,
                    &cache_shape,
                    &cache_strides,
                    device,
                ),
            ];
            kernel
                .execute(&prefill_inputs, &mut prefill_outputs)
                .unwrap();
        }

        {
            let decode_query_strides = compute_contiguous_strides(&decode_query_shape);
            let decode_current_strides = compute_contiguous_strides(&decode_current_shape);
            let decode_inputs = [
                TensorView::new(
                    DevicePtr(decode_query_buffer.as_ptr()),
                    DataType::Float16,
                    &decode_query_shape,
                    &decode_query_strides,
                    device,
                ),
                TensorView::new(
                    DevicePtr(decode_key_buffer.as_ptr()),
                    DataType::Float16,
                    &decode_current_shape,
                    &decode_current_strides,
                    device,
                ),
                TensorView::new(
                    DevicePtr(decode_value_buffer.as_ptr()),
                    DataType::Float16,
                    &decode_current_shape,
                    &decode_current_strides,
                    device,
                ),
                TensorView::new(
                    DevicePtr(cache_key.as_ptr()),
                    DataType::Float16,
                    &cache_shape,
                    &cache_strides,
                    device,
                ),
                TensorView::new(
                    DevicePtr(cache_value.as_ptr()),
                    DataType::Float16,
                    &cache_shape,
                    &cache_strides,
                    device,
                ),
                TensorView::new(
                    DevicePtr(decode_seqlens_buffer.as_ptr()),
                    DataType::Int32,
                    &seqlens_shape,
                    &seqlens_strides,
                    device,
                ),
                TensorView::new(
                    DevicePtr(capacity_buffer.as_ptr()),
                    DataType::Int32,
                    &scalar_shape,
                    &scalar_strides,
                    device,
                ),
            ];
            let mut decode_outputs = [
                TensorMut::new(
                    DevicePtrMut(decode_output.as_mut_ptr()),
                    DataType::Float16,
                    &decode_query_shape,
                    &decode_query_strides,
                    device,
                ),
                TensorMut::new(
                    DevicePtrMut(cache_key.as_mut_ptr()),
                    DataType::Float16,
                    &cache_shape,
                    &cache_strides,
                    device,
                ),
                TensorMut::new(
                    DevicePtrMut(cache_value.as_mut_ptr()),
                    DataType::Float16,
                    &cache_shape,
                    &cache_strides,
                    device,
                ),
            ];
            kernel.execute(&decode_inputs, &mut decode_outputs).unwrap();
            if capture {
                runtime
                    .begin_graph_capture(&[&kernel as &dyn onnx_runtime_ep_api::Kernel])
                    .unwrap();
                kernel.execute(&decode_inputs, &mut decode_outputs).unwrap();
                runtime.end_graph_capture().unwrap();
                assert_eq!(
                    runtime.graph_segment_count().unwrap(),
                    1,
                    "full-generation decode must install one captured segment"
                );
                runtime.replay_graph().unwrap();
                runtime.reset_graph().unwrap();
            }
        }

        let prefill_bytes = read(
            &ep,
            &prefill_output,
            PREFILL_LEN * QUERY_HEADS * HEAD_DIM * std::mem::size_of::<f16>(),
        )
        .unwrap();
        let decode_bytes = read(
            &ep,
            &decode_output,
            QUERY_HEADS * HEAD_DIM * std::mem::size_of::<f16>(),
        )
        .unwrap();
        for buffer in [decode_output, prefill_output, cache_value, cache_key] {
            ep.deallocate(buffer).unwrap();
        }
        (prefill_bytes, decode_bytes)
    };

    let (bnsh_prefill, bnsh_decode) = run_layout(false, false);
    let (bsnh_prefill, bsnh_decode) = run_layout(true, false);
    let (bnsh_captured_prefill, bnsh_captured_decode) = run_layout(false, true);
    let (bsnh_captured_prefill, bsnh_captured_decode) = run_layout(true, true);
    assert_eq!(bnsh_prefill, bsnh_prefill);
    assert_eq!(bnsh_decode, bsnh_decode);
    assert_eq!(bnsh_prefill, bnsh_captured_prefill);
    assert_eq!(bnsh_decode, bnsh_captured_decode);
    assert_eq!(bsnh_prefill, bsnh_captured_prefill);
    assert_eq!(bsnh_decode, bsnh_captured_decode);

    let prefill_expected = cpu_prefill_reference(&prefill_query, &key, &value, PREFILL_LEN, scale);
    let prefill_got = fp16_values(&bnsh_prefill);
    let prefill_max_err = prefill_got
        .iter()
        .zip(&prefill_expected)
        .map(|(got, expected)| (got - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        prefill_max_err < 3e-3,
        "head-major GPU prefill diverged from CPU oracle: max_abs={prefill_max_err:e}"
    );
    let decode_expected = cpu_reference(&decode_query, &key, &value, scale);
    let decode_got = fp16_values(&bnsh_decode);
    let decode_max_err = decode_got
        .iter()
        .zip(&decode_expected)
        .map(|(got, expected)| (got - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        decode_max_err < 3e-3,
        "head-major GPU full-generation decode diverged from CPU oracle: \
         max_abs={decode_max_err:e}"
    );

    for buffer in [
        capacity_buffer,
        decode_seqlens_buffer,
        prefill_seqlens_buffer,
        decode_value_buffer,
        decode_key_buffer,
        decode_query_buffer,
        prefill_value_buffer,
        prefill_key_buffer,
        prefill_query_buffer,
    ] {
        ep.deallocate(buffer).unwrap();
    }
}
