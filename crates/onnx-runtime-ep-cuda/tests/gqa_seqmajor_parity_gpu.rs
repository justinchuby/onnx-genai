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
use onnx_runtime_ep_cuda::{CudaExecutionProvider, GroupQueryAttentionKernel};
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
                runtime.begin_graph_capture(&[]).unwrap();
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
