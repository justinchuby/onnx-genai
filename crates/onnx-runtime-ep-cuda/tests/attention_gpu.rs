//! On-GPU integration test for the Phase-2a SDPA/GQA attention kernel
//! (`docs/ORT2.md` §13 + §15.5).
//!
//! Gated on a real device: if no CUDA GPU is present (or the driver / cuBLASLt /
//! NVRTC can't be loaded), the test prints `skip` and returns, so the crate
//! still tests cleanly on non-GPU machines. On a GPU it runs several attention
//! shapes — non-causal MHA, causal MHA, and grouped-query (GQA) with a causal
//! mask, plus an additive-mask case — and checks the numerics against an
//! independent CPU reference implementing naive softmax attention.
//!
//! Run with the CUDA runtime libs on the loader path, e.g.:
//!   PATH=/conda/env/bin:$PATH LD_LIBRARY_PATH=/conda/env/lib \
//!     cargo test -p onnx-runtime-ep-cuda --test attention_gpu

use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, Kernel as _, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::{AttentionKernel, CudaExecutionProvider};
use onnx_runtime_ir::{compute_contiguous_strides, DataType, DeviceId};
use std::sync::Arc;

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: `f32` is `Copy` with no padding; same lifetime, 4x length.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Independent CPU reference: `O = softmax(scale·Q·Kᵀ [+causal] [+mask])·V`.
///
/// Q `[B,Hq,Sq,D]`, K/V `[B,Hkv,Sk,D]`, O `[B,Hq,Sq,D]`, row-major. GQA maps
/// query head `h` to KV head `h / (Hq/Hkv)`. Causal masks keys
/// `j > Sk - Sq + i`. Optional additive mask is `[Sq,Sk]` (shared over B,H).
#[allow(clippy::too_many_arguments)]
fn cpu_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    hq: usize,
    hkv: usize,
    sq: usize,
    sk: usize,
    d: usize,
    scale: f32,
    causal: bool,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    let group = hq / hkv;
    let mut out = vec![0.0f32; batch * hq * sq * d];
    for b in 0..batch {
        for h in 0..hq {
            let kv = h / group;
            let q_off = (b * hq + h) * sq * d;
            let k_off = (b * hkv + kv) * sk * d;
            let v_off = (b * hkv + kv) * sk * d;
            let o_off = (b * hq + h) * sq * d;
            for i in 0..sq {
                // scores row
                let mut row = vec![f32::NEG_INFINITY; sk];
                let causal_max = sk as isize - sq as isize + i as isize;
                for j in 0..sk {
                    if causal && (j as isize) > causal_max {
                        continue;
                    }
                    let mut acc = 0.0f32;
                    for p in 0..d {
                        acc += q[q_off + i * d + p] * k[k_off + j * d + p];
                    }
                    acc *= scale;
                    if let Some(m) = mask {
                        acc += m[i * sk + j];
                    }
                    row[j] = acc;
                }
                // stable softmax
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for x in row.iter_mut() {
                    *x = if x.is_finite() { (*x - max).exp() } else { 0.0 };
                    sum += *x;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                // weighted sum of V
                for c in 0..d {
                    let mut acc = 0.0f32;
                    for j in 0..sk {
                        acc += row[j] * inv * v[v_off + j * d + c];
                    }
                    out[o_off + i * d + c] = acc;
                }
            }
        }
    }
    out
}

/// Deterministic pseudo-random f32 fill in [-1, 1).
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

struct Buf(onnx_runtime_ep_api::DeviceBuffer);

#[allow(clippy::too_many_arguments)]
fn run_gpu_attention(
    ep: &CudaExecutionProvider,
    runtime: &Arc<onnx_runtime_ep_cuda::CudaRuntime>,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    q_shape: &[usize],
    k_shape: &[usize],
    v_shape: &[usize],
    mask_shape: Option<&[usize]>,
    out_shape: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    causal: bool,
    scale: Option<f32>,
) -> Vec<f32> {
    let dev: DeviceId = ep.device_id();

    let up = |data: &[f32]| -> Buf {
        let buf = ep.allocate(std::mem::size_of_val(data), 256).unwrap();
        // SAFETY: buffer sized for the byte slice we copy in.
        unsafe {
            runtime
                .htod(f32_bytes(data), cuptr(buf.as_ptr()))
                .unwrap();
        }
        Buf(buf)
    };

    let q_buf = up(q);
    let k_buf = up(k);
    let v_buf = up(v);
    let mask_buf = mask.map(up);

    let out_len: usize = out_shape.iter().product();
    let mut out_buf = ep.allocate(out_len * 4, 256).unwrap();

    let q_str = compute_contiguous_strides(q_shape);
    let k_str = compute_contiguous_strides(k_shape);
    let v_str = compute_contiguous_strides(v_shape);
    let o_str = compute_contiguous_strides(out_shape);

    let qv = TensorView::new(DevicePtr(q_buf.0.as_ptr()), DataType::Float32, q_shape, &q_str, dev);
    let kv = TensorView::new(DevicePtr(k_buf.0.as_ptr()), DataType::Float32, k_shape, &k_str, dev);
    let vv = TensorView::new(DevicePtr(v_buf.0.as_ptr()), DataType::Float32, v_shape, &v_str, dev);

    let mask_str = mask_shape.map(compute_contiguous_strides);
    let mut inputs = vec![qv, kv, vv];
    if let (Some(mb), Some(ms), Some(mstr)) = (&mask_buf, mask_shape, &mask_str) {
        inputs.push(TensorView::new(
            DevicePtr(mb.0.as_ptr()),
            DataType::Float32,
            ms,
            mstr,
            dev,
        ));
    }

    let ov = TensorMut::new(
        DevicePtrMut(out_buf.as_mut_ptr()),
        DataType::Float32,
        out_shape,
        &o_str,
        dev,
    );

    let kernel =
        AttentionKernel::new(runtime.clone(), causal, num_heads, num_kv_heads, scale).unwrap();
    kernel.execute(&inputs, &mut [ov]).unwrap();

    let mut out_bytes = vec![0u8; out_len * 4];
    // SAFETY: out_buf holds out_len f32.
    unsafe {
        runtime
            .dtoh(&mut out_bytes, cuptr(out_buf.as_ptr()))
            .unwrap();
    }

    ep.deallocate(q_buf.0).unwrap();
    ep.deallocate(k_buf.0).unwrap();
    ep.deallocate(v_buf.0).unwrap();
    if let Some(mb) = mask_buf {
        ep.deallocate(mb.0).unwrap();
    }
    ep.deallocate(out_buf).unwrap();

    bytes_to_f32(&out_bytes)
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn attention_f32_on_gpu_matches_cpu_reference() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            return;
        }
    };
    let runtime = ep.runtime().clone();
    println!("CUDA EP up on {:?}", ep.device_id());

    let tol = 2e-4f32;

    // Case 1: non-causal MHA. B=2, H=4, Sq=Sk=16, D=64.
    {
        let (b, h, s, d) = (2usize, 4usize, 16usize, 64usize);
        let scale = 1.0 / (d as f32).sqrt();
        let q = fill(b * h * s * d, 1);
        let k = fill(b * h * s * d, 2);
        let v = fill(b * h * s * d, 3);
        let got = run_gpu_attention(
            &ep, &runtime, &q, &k, &v, None,
            &[b, h, s, d], &[b, h, s, d], &[b, h, s, d], None, &[b, h, s, d],
            h, h, false, Some(scale),
        );
        let want = cpu_reference(&q, &k, &v, b, h, h, s, s, d, scale, false, None);
        let err = max_abs_err(&got, &want);
        println!("case1 non-causal MHA  B={b} H={h} S={s} D={d}  max_abs_err={err:e}");
        assert!(err < tol, "non-causal MHA err {err:e} >= {tol:e}");
    }

    // Case 2: causal MHA. B=1, H=8, Sq=Sk=128, D=64.
    {
        let (b, h, s, d) = (1usize, 8usize, 128usize, 64usize);
        let scale = 1.0 / (d as f32).sqrt();
        let q = fill(b * h * s * d, 10);
        let k = fill(b * h * s * d, 20);
        let v = fill(b * h * s * d, 30);
        let got = run_gpu_attention(
            &ep, &runtime, &q, &k, &v, None,
            &[b, h, s, d], &[b, h, s, d], &[b, h, s, d], None, &[b, h, s, d],
            h, h, true, Some(scale),
        );
        let want = cpu_reference(&q, &k, &v, b, h, h, s, s, d, scale, true, None);
        let err = max_abs_err(&got, &want);
        println!("case2 causal MHA      B={b} H={h} S={s} D={d}  max_abs_err={err:e}");
        assert!(err < tol, "causal MHA err {err:e} >= {tol:e}");
    }

    // Case 3: GQA, causal. Hq=8, Hkv=2 (group=4). B=2, Sq=16, Sk=16, D=64.
    {
        let (b, hq, hkv, s, d) = (2usize, 8usize, 2usize, 16usize, 64usize);
        let scale = 0.125f32;
        let q = fill(b * hq * s * d, 4);
        let k = fill(b * hkv * s * d, 5);
        let v = fill(b * hkv * s * d, 6);
        let got = run_gpu_attention(
            &ep, &runtime, &q, &k, &v, None,
            &[b, hq, s, d], &[b, hkv, s, d], &[b, hkv, s, d], None, &[b, hq, s, d],
            hq, hkv, true, Some(scale),
        );
        let want = cpu_reference(&q, &k, &v, b, hq, hkv, s, s, d, scale, true, None);
        let err = max_abs_err(&got, &want);
        println!("case3 GQA causal      B={b} Hq={hq} Hkv={hkv} S={s} D={d}  max_abs_err={err:e}");
        assert!(err < tol, "GQA causal err {err:e} >= {tol:e}");
    }

    // Case 4: GQA with cross-attention seq mismatch (Sq=8, Sk=128) + additive
    // mask shared over [Sq,Sk], non-causal. Hq=8, Hkv=2.
    {
        let (b, hq, hkv, sq, sk, d) = (1usize, 8usize, 2usize, 8usize, 128usize, 64usize);
        let scale = 1.0 / (d as f32).sqrt();
        let q = fill(b * hq * sq * d, 7);
        let k = fill(b * hkv * sk * d, 8);
        let v = fill(b * hkv * sk * d, 9);
        // additive mask [Sq, Sk] with a band of strong suppression.
        let mut mask = vec![0.0f32; sq * sk];
        for i in 0..sq {
            for j in 0..sk {
                if (j + i) % 3 == 0 {
                    mask[i * sk + j] = -1e4;
                }
            }
        }
        let got = run_gpu_attention(
            &ep, &runtime, &q, &k, &v, Some(&mask),
            &[b, hq, sq, d], &[b, hkv, sk, d], &[b, hkv, sk, d], Some(&[sq, sk]), &[b, hq, sq, d],
            hq, hkv, false, Some(scale),
        );
        let want = cpu_reference(&q, &k, &v, b, hq, hkv, sq, sk, d, scale, false, Some(&mask));
        let err = max_abs_err(&got, &want);
        println!("case4 GQA+mask xattn  Sq={sq} Sk={sk} Hq={hq} Hkv={hkv} D={d}  max_abs_err={err:e}");
        assert!(err < tol, "GQA+mask err {err:e} >= {tol:e}");
    }

    println!("all attention cases passed on {:?}", ep.device_id());
}

#[test]
fn attention_rejects_unsupported_dtype_and_rank() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            return;
        }
    };
    let runtime = ep.runtime().clone();
    let dev = ep.device_id();

    // A rank-3 Q is out of Phase-2a scope: expect an actionable error, not a panic.
    let data = [1.0f32; 64];
    let buf = ep.allocate(256, 256).unwrap();
    // SAFETY: 64 f32 = 256 bytes.
    unsafe {
        runtime.htod(f32_bytes(&data), cuptr(buf.as_ptr())).unwrap();
    }
    let mut out = ep.allocate(256, 256).unwrap();
    let shape3 = [1usize, 4, 16];
    let s3 = compute_contiguous_strides(&shape3);
    let qv = TensorView::new(DevicePtr(buf.as_ptr()), DataType::Float32, &shape3, &s3, dev);
    let kv = TensorView::new(DevicePtr(buf.as_ptr()), DataType::Float32, &shape3, &s3, dev);
    let vv = TensorView::new(DevicePtr(buf.as_ptr()), DataType::Float32, &shape3, &s3, dev);
    let ov = TensorMut::new(
        DevicePtrMut(out.as_mut_ptr()),
        DataType::Float32,
        &shape3,
        &s3,
        dev,
    );
    let kernel = AttentionKernel::new(runtime.clone(), false, 4, 4, None).unwrap();
    let err = kernel
        .execute(&[qv, kv, vv], &mut [ov])
        .expect_err("rank-3 attention must be rejected in Phase 2a");
    let msg = format!("{err}");
    println!("rank error: {msg}");
    assert!(msg.contains("Phase 2a") || msg.contains("Phase-2a"), "{msg}");

    ep.deallocate(buf).unwrap();
    ep.deallocate(out).unwrap();
}
