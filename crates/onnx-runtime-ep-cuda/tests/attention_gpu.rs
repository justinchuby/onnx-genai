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

use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, Kernel as _, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::{AttentionKernel, CudaExecutionProvider};
use onnx_runtime_ir::{DataType, DeviceId, compute_contiguous_strides};
use std::sync::Arc;
use std::time::Instant;

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: `f32` is `Copy` with no padding; same lifetime, 4x length.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn encode(values: &[f32], dtype: DataType) -> Vec<u8> {
    match dtype {
        DataType::Float32 => f32_bytes(values).to_vec(),
        DataType::Float16 => values
            .iter()
            .flat_map(|&value| f16::from_f32(value).to_bits().to_ne_bytes())
            .collect(),
        DataType::BFloat16 => values
            .iter()
            .flat_map(|&value| bf16::from_f32(value).to_bits().to_ne_bytes())
            .collect(),
        _ => unreachable!("test helper only supports floating attention dtypes"),
    }
}

fn decode(bytes: &[u8], dtype: DataType) -> Vec<f32> {
    match dtype {
        DataType::Float32 => bytes_to_f32(bytes),
        DataType::Float16 | DataType::BFloat16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_ne_bytes([chunk[0], chunk[1]]);
                match dtype {
                    DataType::Float16 => f16::from_bits(bits).to_f32(),
                    DataType::BFloat16 => bf16::from_bits(bits).to_f32(),
                    _ => unreachable!(),
                }
            })
            .collect(),
        _ => unreachable!("test helper only supports floating attention dtypes"),
    }
}

fn quantize(values: &[f32], dtype: DataType) -> Vec<f32> {
    decode(&encode(values, dtype), dtype)
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

#[derive(Clone, Copy)]
enum AttentionTestBackend {
    Auto,
    Fused,
    Phase2a,
}

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
    dtype: DataType,
) -> Vec<f32> {
    run_gpu_attention_with_backend(
        ep,
        runtime,
        q,
        k,
        v,
        mask,
        q_shape,
        k_shape,
        v_shape,
        mask_shape,
        out_shape,
        num_heads,
        num_kv_heads,
        causal,
        scale,
        dtype,
        AttentionTestBackend::Auto,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_gpu_attention_with_backend(
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
    dtype: DataType,
    backend: AttentionTestBackend,
) -> Vec<f32> {
    let dev: DeviceId = ep.device_id();
    let element_size = match dtype {
        DataType::Float32 => 4,
        DataType::Float16 | DataType::BFloat16 => 2,
        _ => unreachable!("test helper only supports floating attention dtypes"),
    };

    let up = |data: &[f32]| -> Buf {
        let bytes = encode(data, dtype);
        let buf = ep.allocate(bytes.len(), 256).unwrap();
        // SAFETY: buffer sized for the byte slice we copy in.
        unsafe {
            runtime.htod(&bytes, cuptr(buf.as_ptr())).unwrap();
        }
        Buf(buf)
    };

    let q_buf = up(q);
    let k_buf = up(k);
    let v_buf = up(v);
    let mask_buf = mask.map(up);

    let out_len: usize = out_shape.iter().product();
    let mut out_buf = ep.allocate(out_len * element_size, 256).unwrap();

    let q_str = compute_contiguous_strides(q_shape);
    let k_str = compute_contiguous_strides(k_shape);
    let v_str = compute_contiguous_strides(v_shape);
    let o_str = compute_contiguous_strides(out_shape);

    let qv = TensorView::new(DevicePtr(q_buf.0.as_ptr()), dtype, q_shape, &q_str, dev);
    let kv = TensorView::new(DevicePtr(k_buf.0.as_ptr()), dtype, k_shape, &k_str, dev);
    let vv = TensorView::new(DevicePtr(v_buf.0.as_ptr()), dtype, v_shape, &v_str, dev);

    let mask_str = mask_shape.map(compute_contiguous_strides);
    let mut inputs = vec![qv, kv, vv];
    if let (Some(mb), Some(ms), Some(mstr)) = (&mask_buf, mask_shape, &mask_str) {
        inputs.push(TensorView::new(
            DevicePtr(mb.0.as_ptr()),
            dtype,
            ms,
            mstr,
            dev,
        ));
    }

    let ov = TensorMut::new(
        DevicePtrMut(out_buf.as_mut_ptr()),
        dtype,
        out_shape,
        &o_str,
        dev,
    );

    let kernel = match backend {
        AttentionTestBackend::Auto => {
            AttentionKernel::new(runtime.clone(), causal, num_heads, num_kv_heads, scale)
        }
        AttentionTestBackend::Fused => {
            AttentionKernel::new_fused(runtime.clone(), causal, num_heads, num_kv_heads, scale)
        }
        AttentionTestBackend::Phase2a => {
            AttentionKernel::new_phase2a(runtime.clone(), causal, num_heads, num_kv_heads, scale)
        }
    }
    .unwrap();
    if let Err(error) = kernel.execute(&inputs, &mut [ov]) {
        let message = format!("{error}");
        ep.deallocate(q_buf.0).unwrap();
        ep.deallocate(k_buf.0).unwrap();
        ep.deallocate(v_buf.0).unwrap();
        if let Some(mb) = mask_buf {
            ep.deallocate(mb.0).unwrap();
        }
        ep.deallocate(out_buf).unwrap();
        if message.contains("CUDA_ERROR_UNSUPPORTED_PTX_VERSION") {
            eprintln!("skip: NVRTC PTX is newer than the installed CUDA driver ({message})");
            return Vec::new();
        }
        panic!("attention GPU execution failed: {message}");
    }

    let mut out_bytes = vec![0u8; out_len * element_size];
    // SAFETY: out_buf holds out_len elements of the selected dtype.
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

    decode(&out_bytes, dtype)
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
            &ep,
            &runtime,
            &q,
            &k,
            &v,
            None,
            &[b, h, s, d],
            &[b, h, s, d],
            &[b, h, s, d],
            None,
            &[b, h, s, d],
            h,
            h,
            false,
            Some(scale),
            DataType::Float32,
        );
        if got.is_empty() {
            return;
        }
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
            &ep,
            &runtime,
            &q,
            &k,
            &v,
            None,
            &[b, h, s, d],
            &[b, h, s, d],
            &[b, h, s, d],
            None,
            &[b, h, s, d],
            h,
            h,
            true,
            Some(scale),
            DataType::Float32,
        );
        if got.is_empty() {
            return;
        }
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
            &ep,
            &runtime,
            &q,
            &k,
            &v,
            None,
            &[b, hq, s, d],
            &[b, hkv, s, d],
            &[b, hkv, s, d],
            None,
            &[b, hq, s, d],
            hq,
            hkv,
            true,
            Some(scale),
            DataType::Float32,
        );
        if got.is_empty() {
            return;
        }
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
            &ep,
            &runtime,
            &q,
            &k,
            &v,
            Some(&mask),
            &[b, hq, sq, d],
            &[b, hkv, sk, d],
            &[b, hkv, sk, d],
            Some(&[sq, sk]),
            &[b, hq, sq, d],
            hq,
            hkv,
            false,
            Some(scale),
            DataType::Float32,
        );
        if got.is_empty() {
            return;
        }
        let want = cpu_reference(&q, &k, &v, b, hq, hkv, sq, sk, d, scale, false, Some(&mask));
        let err = max_abs_err(&got, &want);
        println!(
            "case4 GQA+mask xattn  Sq={sq} Sk={sk} Hq={hq} Hkv={hkv} D={d}  max_abs_err={err:e}"
        );
        assert!(err < tol, "GQA+mask err {err:e} >= {tol:e}");
    }

    println!("all attention cases passed on {:?}", ep.device_id());
}

fn assert_close(got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len());
    for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
        let tolerance = atol + rtol * want.abs();
        assert!(
            (got - want).abs() <= tolerance,
            "index {index}: got {got}, want {want}, abs_err={}, tolerance={tolerance}",
            (got - want).abs()
        );
    }
}

fn attention_half_matches_f32_reference(dtype: DataType, atol: f32, rtol: f32) {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            return;
        }
    };
    let runtime = ep.runtime().clone();
    let (b, h, s, d) = (1usize, 2usize, 16usize, 32usize);
    let scale = 1.0 / (d as f32).sqrt();
    let q = quantize(&fill(b * h * s * d, 101), dtype);
    let k = quantize(&fill(b * h * s * d, 102), dtype);
    let v = quantize(&fill(b * h * s * d, 103), dtype);
    let got = run_gpu_attention(
        &ep,
        &runtime,
        &q,
        &k,
        &v,
        None,
        &[b, h, s, d],
        &[b, h, s, d],
        &[b, h, s, d],
        None,
        &[b, h, s, d],
        h,
        h,
        true,
        Some(scale),
        dtype,
    );
    if got.is_empty() {
        return;
    }
    let want = cpu_reference(&q, &k, &v, b, h, h, s, s, d, scale, true, None);
    assert_close(&got, &want, atol, rtol);
}

#[test]
fn attention_f16_on_gpu_matches_f32_reference() {
    // Two native-f16 GEMM stores plus the probability store can each contribute
    // roughly one fp16 ulp, so allow a small multiple of fp16 epsilon.
    attention_half_matches_f32_reference(DataType::Float16, 3e-3, 3e-3);
}

#[test]
fn attention_bf16_on_gpu_matches_f32_reference() {
    // bf16 has a 7-bit mantissa (8x fp16 epsilon), so its bound is correspondingly
    // wider while still catching incorrect accumulation or dtype dispatch.
    attention_half_matches_f32_reference(DataType::BFloat16, 2e-2, 2e-2);
}

#[test]
fn fused_attention_matches_phase2a_baseline() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            return;
        }
    };
    let runtime = ep.runtime().clone();

    for (dtype, atol, rtol) in [
        (DataType::Float32, 1e-4, 1e-5),
        (DataType::Float16, 5e-3, 5e-3),
        (DataType::BFloat16, 3e-2, 3e-2),
    ] {
        // Small non-causal MHA without a mask.
        let (b, h, s, d) = (1usize, 4usize, 17usize, 64usize);
        let scale = 1.0 / (d as f32).sqrt();
        let q = quantize(&fill(b * h * s * d, 501), dtype);
        let k = quantize(&fill(b * h * s * d, 502), dtype);
        let v = quantize(&fill(b * h * s * d, 503), dtype);
        let fused = run_gpu_attention_with_backend(
            &ep,
            &runtime,
            &q,
            &k,
            &v,
            None,
            &[b, h, s, d],
            &[b, h, s, d],
            &[b, h, s, d],
            None,
            &[b, h, s, d],
            h,
            h,
            false,
            Some(scale),
            dtype,
            AttentionTestBackend::Fused,
        );
        let baseline = run_gpu_attention_with_backend(
            &ep,
            &runtime,
            &q,
            &k,
            &v,
            None,
            &[b, h, s, d],
            &[b, h, s, d],
            &[b, h, s, d],
            None,
            &[b, h, s, d],
            h,
            h,
            false,
            Some(scale),
            dtype,
            AttentionTestBackend::Phase2a,
        );
        assert_close(&fused, &baseline, atol, rtol);

        // Non-trivial causal GQA with a shared additive mask.
        let (b, hq, hkv, s, d) = (1usize, 8usize, 2usize, 129usize, 128usize);
        let scale = 1.0 / (d as f32).sqrt();
        let q = quantize(&fill(b * hq * s * d, 511), dtype);
        let k = quantize(&fill(b * hkv * s * d, 512), dtype);
        let v = quantize(&fill(b * hkv * s * d, 513), dtype);
        let mask = quantize(
            &(0..s * s)
                .map(|index| if index % 11 == 0 { -8.0 } else { 0.0 })
                .collect::<Vec<_>>(),
            dtype,
        );
        let fused = run_gpu_attention_with_backend(
            &ep,
            &runtime,
            &q,
            &k,
            &v,
            Some(&mask),
            &[b, hq, s, d],
            &[b, hkv, s, d],
            &[b, hkv, s, d],
            Some(&[s, s]),
            &[b, hq, s, d],
            hq,
            hkv,
            true,
            Some(scale),
            dtype,
            AttentionTestBackend::Fused,
        );
        let baseline = run_gpu_attention_with_backend(
            &ep,
            &runtime,
            &q,
            &k,
            &v,
            Some(&mask),
            &[b, hq, s, d],
            &[b, hkv, s, d],
            &[b, hkv, s, d],
            Some(&[s, s]),
            &[b, hq, s, d],
            hq,
            hkv,
            true,
            Some(scale),
            dtype,
            AttentionTestBackend::Phase2a,
        );
        println!(
            "fused parity dtype={dtype:?} GQA causal+mask S={s} D={d}: max_abs_err={:e}",
            max_abs_err(&fused, &baseline)
        );
        assert_close(&fused, &baseline, atol, rtol);
    }

    // Large-magnitude f32 scores exercise running-max rescaling. A naive
    // online exp(score) implementation overflows here.
    let (b, h, s, d) = (1usize, 2usize, 65usize, 64usize);
    let q = fill(b * h * s * d, 521)
        .into_iter()
        .map(|value| value * 40.0)
        .collect::<Vec<_>>();
    let k = fill(b * h * s * d, 522)
        .into_iter()
        .map(|value| value * 40.0)
        .collect::<Vec<_>>();
    let v = fill(b * h * s * d, 523);
    let fused = run_gpu_attention_with_backend(
        &ep,
        &runtime,
        &q,
        &k,
        &v,
        None,
        &[b, h, s, d],
        &[b, h, s, d],
        &[b, h, s, d],
        None,
        &[b, h, s, d],
        h,
        h,
        false,
        Some(1.0),
        DataType::Float32,
        AttentionTestBackend::Fused,
    );
    let baseline = run_gpu_attention_with_backend(
        &ep,
        &runtime,
        &q,
        &k,
        &v,
        None,
        &[b, h, s, d],
        &[b, h, s, d],
        &[b, h, s, d],
        None,
        &[b, h, s, d],
        h,
        h,
        false,
        Some(1.0),
        DataType::Float32,
        AttentionTestBackend::Phase2a,
    );
    assert!(fused.iter().all(|value| value.is_finite()));
    assert_close(&fused, &baseline, 2e-4, 1e-5);
}

#[test]
#[ignore = "H200 performance benchmark; run explicitly with --ignored --nocapture"]
fn attention_prefill_h200_benchmark() {
    let ep = CudaExecutionProvider::new_default().expect("benchmark requires a CUDA GPU");
    let runtime = ep.runtime().clone();
    let dev = ep.device_id();

    for (s, iterations) in [(512usize, 10usize), (2048usize, 3usize)] {
        let (b, h, d) = (1usize, 32usize, 128usize);
        let shape = [b, h, s, d];
        let strides = compute_contiguous_strides(&shape);
        let scale = 1.0 / (d as f32).sqrt();
        let upload = |values: Vec<f32>| {
            let bytes = encode(&values, DataType::Float16);
            let buffer = ep.allocate(bytes.len(), 256).unwrap();
            unsafe {
                runtime.htod(&bytes, cuptr(buffer.as_ptr())).unwrap();
            }
            buffer
        };
        let q = upload(fill(b * h * s * d, 601 + s as u64));
        let k = upload(fill(b * h * s * d, 602 + s as u64));
        let v = upload(fill(b * h * s * d, 603 + s as u64));
        let mut fused_out = ep.allocate(b * h * s * d * 2, 256).unwrap();
        let mut baseline_out = ep.allocate(b * h * s * d * 2, 256).unwrap();
        let fused = AttentionKernel::new_fused(runtime.clone(), true, h, h, Some(scale)).unwrap();
        let baseline =
            AttentionKernel::new_phase2a(runtime.clone(), true, h, h, Some(scale)).unwrap();

        let mut run = |kernel: &AttentionKernel, output: &mut onnx_runtime_ep_api::DeviceBuffer| {
            let inputs = [
                TensorView::new(
                    DevicePtr(q.as_ptr()),
                    DataType::Float16,
                    &shape,
                    &strides,
                    dev,
                ),
                TensorView::new(
                    DevicePtr(k.as_ptr()),
                    DataType::Float16,
                    &shape,
                    &strides,
                    dev,
                ),
                TensorView::new(
                    DevicePtr(v.as_ptr()),
                    DataType::Float16,
                    &shape,
                    &strides,
                    dev,
                ),
            ];
            let output = TensorMut::new(
                DevicePtrMut(output.as_mut_ptr()),
                DataType::Float16,
                &shape,
                &strides,
                dev,
            );
            kernel.execute(&inputs, &mut [output]).unwrap();
        };

        // Compile NVRTC and warm allocator/library state before timing.
        run(&fused, &mut fused_out);
        run(&baseline, &mut baseline_out);

        let measure = |kernel: &AttentionKernel,
                       output: &mut onnx_runtime_ep_api::DeviceBuffer,
                       run: &mut dyn FnMut(
            &AttentionKernel,
            &mut onnx_runtime_ep_api::DeviceBuffer,
        )| {
            let mut samples = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let start = Instant::now();
                run(kernel, output);
                samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        let fused_ms = measure(&fused, &mut fused_out, &mut run);
        let baseline_ms = measure(&baseline, &mut baseline_out, &mut run);
        let score_bytes = b * h * s * s * 2;
        let baseline_scratch = score_bytes + 32 * 1024 * 1024;
        println!(
            "H200 f16 causal prefill B={b} H={h} S={s} D={d}: \
             fused={fused_ms:.3} ms baseline={baseline_ms:.3} ms speedup={:.2}x; \
             global scratch fused=0 MiB baseline={:.1} MiB (score={:.1} MiB + cuBLASLt=32 MiB)",
            baseline_ms / fused_ms,
            baseline_scratch as f64 / (1024.0 * 1024.0),
            score_bytes as f64 / (1024.0 * 1024.0),
        );

        ep.deallocate(q).unwrap();
        ep.deallocate(k).unwrap();
        ep.deallocate(v).unwrap();
        ep.deallocate(fused_out).unwrap();
        ep.deallocate(baseline_out).unwrap();
    }
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
    let qv = TensorView::new(
        DevicePtr(buf.as_ptr()),
        DataType::Float32,
        &shape3,
        &s3,
        dev,
    );
    let kv = TensorView::new(
        DevicePtr(buf.as_ptr()),
        DataType::Float32,
        &shape3,
        &s3,
        dev,
    );
    let vv = TensorView::new(
        DevicePtr(buf.as_ptr()),
        DataType::Float32,
        &shape3,
        &s3,
        dev,
    );
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
    assert!(
        msg.contains("Phase 2a") || msg.contains("Phase-2a"),
        "{msg}"
    );

    ep.deallocate(buf).unwrap();
    ep.deallocate(out).unwrap();
}
