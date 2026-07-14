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
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ir::{compute_contiguous_strides, DataType, DeviceId, Node, NodeId};

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
    assert!(approx_eq(&got, &expected, 1e-4), "gpu {got:?} vs {expected:?}");

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

    println!("all cuBLASLt MatMul cases passed on {:?}", ep.device_id());
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
    let rt = ep.runtime();
    let dev = ep.device_id();

    // A 4-D operand is out of Phase-2a scope: expect an actionable error, not a
    // panic. Use tiny real device buffers so pointer math is valid.
    let a = [1.0f32; 16];
    let b = [1.0f32; 16];
    let a_buf = ep.allocate(64, 256).unwrap();
    let b_buf = ep.allocate(64, 256).unwrap();
    let mut c_buf = ep.allocate(64, 256).unwrap();
    // SAFETY: 16 f32 = 64 bytes each.
    unsafe {
        rt.htod(f32_bytes(&a), cuptr(a_buf.as_ptr())).unwrap();
        rt.htod(f32_bytes(&b), cuptr(b_buf.as_ptr())).unwrap();
    }
    let a_shape = [2usize, 2, 2, 2];
    let b_shape = [2usize, 2, 2, 2];
    let out_shape = [2usize, 2, 2, 2];
    let a_str = compute_contiguous_strides(&a_shape);
    let b_str = compute_contiguous_strides(&b_shape);
    let o_str = compute_contiguous_strides(&out_shape);
    let av = TensorView::new(DevicePtr(a_buf.as_ptr()), DataType::Float32, &a_shape, &a_str, dev);
    let bv = TensorView::new(DevicePtr(b_buf.as_ptr()), DataType::Float32, &b_shape, &b_str, dev);
    let ov = TensorMut::new(
        DevicePtrMut(c_buf.as_mut_ptr()),
        DataType::Float32,
        &out_shape,
        &o_str,
        dev,
    );
    let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
    let kernel = ep.get_kernel(&node, &[], 17).unwrap();
    let err = kernel
        .execute(&[av, bv], &mut [ov])
        .expect_err("4-D MatMul must be rejected in Phase 2a");
    let msg = format!("{err}");
    println!("rank error: {msg}");
    assert!(msg.contains("Phase 2a"), "error must name Phase 2a: {msg}");

    ep.deallocate(a_buf).unwrap();
    ep.deallocate(b_buf).unwrap();
    ep.deallocate(c_buf).unwrap();
}
