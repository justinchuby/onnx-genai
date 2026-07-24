//! On-GPU integration test for the cuBLASLt MatMul kernel.
//!
//! Gated on a real device: if no CUDA GPU is present (or the driver / cuBLASLt
//! can't be loaded), the test prints `skip` and returns, so the crate still
//! tests cleanly on non-GPU machines. On a GPU it runs f32 (integer, fractional,
//! and batched) MatMuls and checks the numerics against an independently
//! computed CPU reference.
//!
//! Run with the CUDA runtime libs on the loader path, e.g.:
//!   LD_LIBRARY_PATH=/path/to/cuda/lib cargo test -p onnx-runtime-ep-cuda

use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{DataType, DeviceId, Node, NodeId, compute_contiguous_strides};

/// Reinterpret an `&[f32]` as its little-endian bytes (host side).
fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: `f32` is `Copy` with no padding; the byte view has the same
    // lifetime and 4x the length.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Independent row-major reference GEMM: `C[b,M,N] = A[b,M,K] · B[b,K,N]`.
fn cpu_reference(a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[bi * m * k + i * k + p] * b[bi * k * n + p * n + j];
                }
                c[bi * m * n + i * n + j] = acc;
            }
        }
    }
    c
}

fn cpu_reference_nd(a: &[f32], b: &[f32], a_shape: &[usize], b_shape: &[usize]) -> Vec<f32> {
    let (m, k, n) = (
        a_shape[a_shape.len() - 2],
        a_shape[a_shape.len() - 1],
        b_shape[b_shape.len() - 1],
    );
    let batch_rank = (a_shape.len() - 2).max(b_shape.len() - 2);
    let mut ad = vec![1; batch_rank];
    let mut bd = vec![1; batch_rank];
    ad[batch_rank - (a_shape.len() - 2)..].copy_from_slice(&a_shape[..a_shape.len() - 2]);
    bd[batch_rank - (b_shape.len() - 2)..].copy_from_slice(&b_shape[..b_shape.len() - 2]);
    let out_batch: Vec<usize> = ad.iter().zip(&bd).map(|(&x, &y)| x.max(y)).collect();
    let batches: usize = out_batch.iter().product();
    let mut out = vec![0.0; batches * m * n];

    for batch in 0..batches {
        let mut rem = batch;
        let mut a_batch = 0;
        let mut b_batch = 0;
        let mut a_stride = 1;
        let mut b_stride = 1;
        for axis in (0..batch_rank).rev() {
            let coord = rem % out_batch[axis];
            rem /= out_batch[axis];
            if ad[axis] != 1 {
                a_batch += coord * a_stride;
            }
            if bd[axis] != 1 {
                b_batch += coord * b_stride;
            }
            a_stride *= ad[axis];
            b_stride *= bd[axis];
        }
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    out[batch * m * n + i * n + j] +=
                        a[a_batch * m * k + i * k + p] * b[b_batch * k * n + p * n + j];
                }
            }
        }
    }
    out
}

/// Run one f32 MatMul on the GPU and return the host result.
fn run_gpu_matmul_f32(
    ep: &CudaExecutionProvider,
    a: &[f32],
    b: &[f32],
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Vec<f32> {
    let dev: DeviceId = ep.device_id();
    let rt = ep.runtime();

    let a_buf = ep.allocate(std::mem::size_of_val(a), 256).unwrap();
    let b_buf = ep.allocate(std::mem::size_of_val(b), 256).unwrap();
    let out_len: usize = out_shape.iter().product();
    let mut c_buf = ep.allocate(out_len * 4, 256).unwrap();

    // SAFETY: device buffers are sized for the byte slices we copy in.
    unsafe {
        rt.htod(f32_bytes(a), cuptr(a_buf.as_ptr())).unwrap();
        rt.htod(f32_bytes(b), cuptr(b_buf.as_ptr())).unwrap();
    }

    let a_strides = compute_contiguous_strides(a_shape);
    let b_strides = compute_contiguous_strides(b_shape);
    let out_strides = compute_contiguous_strides(out_shape);

    let a_view = TensorView::new(
        DevicePtr(a_buf.as_ptr()),
        DataType::Float32,
        a_shape,
        &a_strides,
        dev,
    );
    let b_view = TensorView::new(
        DevicePtr(b_buf.as_ptr()),
        DataType::Float32,
        b_shape,
        &b_strides,
        dev,
    );
    let out_view = TensorMut::new(
        DevicePtrMut(c_buf.as_mut_ptr()),
        DataType::Float32,
        out_shape,
        &out_strides,
        dev,
    );

    let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
    let kernel = ep
        .get_kernel(&node, &[a_shape.to_vec(), b_shape.to_vec()], 17)
        .unwrap();
    kernel.execute(&[a_view, b_view], &mut [out_view]).unwrap();

    let mut out_bytes = vec![0u8; out_len * 4];
    // SAFETY: c_buf holds `out_len` f32 = out_len*4 bytes.
    unsafe {
        rt.dtoh(&mut out_bytes, cuptr(c_buf.as_ptr())).unwrap();
    }

    ep.deallocate(a_buf).unwrap();
    ep.deallocate(b_buf).unwrap();
    ep.deallocate(c_buf).unwrap();

    bytes_to_f32(&out_bytes)
}

fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn matmul_f32_on_gpu_matches_cpu_reference() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            return;
        }
    };
    println!("CUDA EP up on {:?}", ep.device_id());

    // Case 1: the canonical [2,3]x[3,2] integer case → known result.
    let a = [1., 2., 3., 4., 5., 6.];
    let b = [7., 8., 9., 10., 11., 12.];
    let got = run_gpu_matmul_f32(&ep, &a, &b, &[2, 3], &[3, 2], &[2, 2]);
    let expected = cpu_reference(&a, &b, 1, 2, 3, 2);
    println!("case1 gpu={got:?} expected={expected:?}");
    assert_eq!(expected, vec![58., 64., 139., 154.], "reference sanity");
    assert!(
        approx_eq(&got, &expected, 1e-4),
        "gpu {got:?} vs {expected:?}"
    );

    // Case 2: fractional f32 values — this WOULD fail under a TF32 compute type
    // (~1e-3 error), so passing at 1e-4 proves we requested true fp32.
    let a2 = [
        0.6279, -0.5686, 0.2195, 1.1292, -0.1345, 0.9021, 0.4894, -0.3443, -1.6810, 0.3060,
        -0.9714, -0.3611,
    ];
    let b2 = [
        0.5220, -0.0040, -0.3835, -0.3801, 0.4199, -0.2248, -1.6661, -0.4569, -0.9043, 0.3913,
        1.3413, 3.4747,
    ];
    let got2 = run_gpu_matmul_f32(&ep, &a2, &b2, &[3, 4], &[4, 3], &[3, 3]);
    let expected2 = cpu_reference(&a2, &b2, 1, 3, 4, 3);
    println!("case2 gpu={got2:?} expected={expected2:?}");
    assert!(
        approx_eq(&got2, &expected2, 1e-4),
        "fractional fp32 mismatch (TF32 leak?) gpu {got2:?} vs {expected2:?}"
    );

    // Case 3: batched 3-D — two independent [2,2] matmuls.
    let a3 = [1., 2., 3., 4., 5., 6., 7., 8.];
    let b3 = [1., 0., 0., 1., 2., 0., 0., 2.];
    let got3 = run_gpu_matmul_f32(&ep, &a3, &b3, &[2, 2, 2], &[2, 2, 2], &[2, 2, 2]);
    let expected3 = cpu_reference(&a3, &b3, 2, 2, 2, 2);
    println!("case3 gpu={got3:?} expected={expected3:?}");
    assert!(
        approx_eq(&got3, &expected3, 1e-4),
        "batched mismatch gpu {got3:?} vs {expected3:?}"
    );

    // Case 4: equal-batch 4-D.
    let a4: Vec<f32> = (0..48).map(|i| (i as f32 - 17.0) / 11.0).collect();
    let b4: Vec<f32> = (0..80).map(|i| (23.0 - i as f32) / 13.0).collect();
    let got4 = run_gpu_matmul_f32(&ep, &a4, &b4, &[2, 2, 3, 4], &[2, 2, 4, 5], &[2, 2, 3, 5]);
    let expected4 = cpu_reference_nd(&a4, &b4, &[2, 2, 3, 4], &[2, 2, 4, 5]);
    assert!(
        approx_eq(&got4, &expected4, 2e-4),
        "4-D batched mismatch gpu {got4:?} vs {expected4:?}"
    );

    // Case 5: both operands broadcast a different leading batch dimension.
    let a5: Vec<f32> = (0..24).map(|i| (i as f32 + 1.0) / 9.0).collect();
    let b5: Vec<f32> = (0..60).map(|i| (i as f32 - 7.0) / 10.0).collect();
    let got5 = run_gpu_matmul_f32(&ep, &a5, &b5, &[2, 1, 3, 4], &[1, 3, 4, 5], &[2, 3, 3, 5]);
    let expected5 = cpu_reference_nd(&a5, &b5, &[2, 1, 3, 4], &[1, 3, 4, 5]);
    assert!(
        approx_eq(&got5, &expected5, 2e-4),
        "broadcast batched mismatch gpu {got5:?} vs {expected5:?}"
    );

    println!("all cuBLASLt MatMul cases passed on {:?}", ep.device_id());
}

/// Run one dense fp16 MatMul on the GPU and return the host result as f32.
/// Exercises the M==1 decode GEMV fast path when `m == 1`.
fn run_gpu_matmul_f16(
    ep: &CudaExecutionProvider,
    a: &[f32],
    b: &[f32],
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Vec<f32> {
    let dev: DeviceId = ep.device_id();
    let rt = ep.runtime();

    let a_h: Vec<half::f16> = a.iter().map(|&x| half::f16::from_f32(x)).collect();
    let b_h: Vec<half::f16> = b.iter().map(|&x| half::f16::from_f32(x)).collect();
    let a_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(a_h.as_ptr() as *const u8, std::mem::size_of_val(&a_h[..]))
    };
    let b_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(b_h.as_ptr() as *const u8, std::mem::size_of_val(&b_h[..]))
    };

    let a_buf = ep.allocate(a_bytes.len(), 256).unwrap();
    let b_buf = ep.allocate(b_bytes.len(), 256).unwrap();
    let out_len: usize = out_shape.iter().product();
    let mut c_buf = ep.allocate(out_len * 2, 256).unwrap();

    // SAFETY: device buffers are sized for the byte slices we copy in.
    unsafe {
        rt.htod(a_bytes, cuptr(a_buf.as_ptr())).unwrap();
        rt.htod(b_bytes, cuptr(b_buf.as_ptr())).unwrap();
    }

    let a_strides = compute_contiguous_strides(a_shape);
    let b_strides = compute_contiguous_strides(b_shape);
    let out_strides = compute_contiguous_strides(out_shape);

    let a_view = TensorView::new(
        DevicePtr(a_buf.as_ptr()),
        DataType::Float16,
        a_shape,
        &a_strides,
        dev,
    );
    let b_view = TensorView::new(
        DevicePtr(b_buf.as_ptr()),
        DataType::Float16,
        b_shape,
        &b_strides,
        dev,
    );
    let out_view = TensorMut::new(
        DevicePtrMut(c_buf.as_mut_ptr()),
        DataType::Float16,
        out_shape,
        &out_strides,
        dev,
    );

    let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
    let kernel = ep
        .get_kernel(&node, &[a_shape.to_vec(), b_shape.to_vec()], 17)
        .unwrap();
    kernel.execute(&[a_view, b_view], &mut [out_view]).unwrap();
    // The M==1 fp16 path takes the capturable GEMV; assert the kernel advertises it.
    if a_shape[a_shape.len() - 2] == 1 {
        assert!(
            kernel.capture_support().is_supported(),
            "dense fp16 M==1 GEMV must advertise capture support"
        );
    }

    let mut out_bytes = vec![0u8; out_len * 2];
    // SAFETY: c_buf holds `out_len` f16 = out_len*2 bytes.
    unsafe {
        rt.dtoh(&mut out_bytes, cuptr(c_buf.as_ptr())).unwrap();
    }

    ep.deallocate(a_buf).unwrap();
    ep.deallocate(b_buf).unwrap();
    ep.deallocate(c_buf).unwrap();

    out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// The dense fp16 M==1 GEMV fast path (used by an fp16 language-model head)
/// matches an independent CPU reference and is capture-eligible. Generic over
/// `K`/`N`: nothing here is tied to a model dimension.
#[test]
fn matmul_f16_gemv_on_gpu_matches_cpu_reference() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            return;
        }
    };

    // A non-square GEMV that crosses several thread blocks (N=300 > 256) and a
    // K that is not a multiple of the block width, exercising the tail path.
    let (k, n) = (259usize, 300usize);
    let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013 - 1.7).sin()).collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| (((i * 37) % 101) as f32 / 101.0) - 0.5)
        .collect();

    let got = run_gpu_matmul_f16(&ep, &a, &b, &[1, k], &[k, n], &[1, n]);
    let expected = cpu_reference(&a, &b, 1, 1, k, n);

    // fp16 storage of A/B/Y plus fp32 accumulation: a relative tolerance scaled
    // by K is appropriate (each output sums K fp16 products).
    assert_eq!(got.len(), n);
    for (col, (g, e)) in got.iter().zip(&expected).enumerate() {
        let tol = 1e-2 + 1e-2 * e.abs();
        assert!(
            (g - e).abs() <= tol,
            "col {col}: gpu {g} vs cpu {e} (tol {tol})"
        );
    }
    println!(
        "dense fp16 GEMV matches CPU reference on {:?}",
        ep.device_id()
    );
}

#[test]
fn matmul_rejects_unsupported_rank_and_dtype() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            return;
        }
    };
    let dev = ep.device_id();

    // The formerly rejected 4-D case now executes and is checked against CPU.
    let a = [1.0f32; 16];
    let b = [1.0f32; 16];
    let a_shape = [2usize, 2, 2, 2];
    let b_shape = [2usize, 2, 2, 2];
    let out_shape = [2usize, 2, 2, 2];
    let got = run_gpu_matmul_f32(&ep, &a, &b, &a_shape, &b_shape, &out_shape);
    let expected = cpu_reference_nd(&a, &b, &a_shape, &b_shape);
    assert!(approx_eq(&got, &expected, 1e-4), "{got:?} vs {expected:?}");

    // Int64 remains unsupported, with the current actionable dtype wording.
    let a_buf = ep.allocate(8, 256).unwrap();
    let b_buf = ep.allocate(8, 256).unwrap();
    let mut c_buf = ep.allocate(8, 256).unwrap();
    let shape = [1usize, 1];
    let strides = compute_contiguous_strides(&shape);
    let av = TensorView::new(
        DevicePtr(a_buf.as_ptr()),
        DataType::Int64,
        &shape,
        &strides,
        dev,
    );
    let bv = TensorView::new(
        DevicePtr(b_buf.as_ptr()),
        DataType::Int64,
        &shape,
        &strides,
        dev,
    );
    let ov = TensorMut::new(
        DevicePtrMut(c_buf.as_mut_ptr()),
        DataType::Int64,
        &shape,
        &strides,
        dev,
    );
    let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
    let kernel = ep.get_kernel(&node, &[], 17).unwrap();
    let err = kernel
        .execute(&[av, bv], &mut [ov])
        .expect_err("Int64 MatMul must be rejected");
    let msg = format!("{err}");
    println!("dtype error: {msg}");
    assert!(
        msg.contains("MatMul with dtype Int64 is not yet implemented on the CUDA EP"),
        "{msg}"
    );

    ep.deallocate(a_buf).unwrap();
    ep.deallocate(b_buf).unwrap();
    ep.deallocate(c_buf).unwrap();
}
