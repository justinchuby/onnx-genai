#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::unusual_byte_groupings,
    clippy::doc_lazy_continuation,
    clippy::uninlined_format_args,
    clippy::cloned_ref_to_slice_refs,
    clippy::type_complexity,
    clippy::drop_non_drop,
    clippy::manual_repeat_n,
    clippy::manual_is_multiple_of,
    clippy::err_expect,
    clippy::clone_on_copy
)]
//! On-GPU integration test for the cuBLASLt MatMul kernel.
//!
//! Gated on a real device: CPU-only CI reports these as ignored unless
//! `gpu-tests` is enabled. Feature-enabled runs fail loudly if CUDA cannot run.
//! On a GPU it runs f32 (integer, fractional,
//! and batched) MatMuls and checks the numerics against an independently
//! computed CPU reference.
//!
//! Run with the CUDA runtime libs on the loader path, e.g.:
//!   LD_LIBRARY_PATH=/path/to/cuda/lib cargo test -p onnx-runtime-ep-cuda

mod common;

#[cfg(feature = "gpu-tests")]
use std::sync::Arc;

use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::blas::{self, GemmDtype, GemmParams, WORKSPACE_BYTES};
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{DataType, DeviceId, Node, NodeId, compute_contiguous_strides};
#[cfg(feature = "gpu-tests")]
use onnx_runtime_memory_api::DeviceAllocator;

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
    let mut workspace = None;
    common::execute_kernel(
        ep,
        kernel.as_ref(),
        &[a_view, b_view],
        &mut [out_view],
        &mut workspace,
    )
    .unwrap();

    let mut out_bytes = vec![0u8; out_len * 4];
    // SAFETY: c_buf holds `out_len` f32 = out_len*4 bytes.
    unsafe {
        rt.dtoh(&mut out_bytes, cuptr(c_buf.as_ptr())).unwrap();
    }

    if let Some(workspace) = workspace {
        ep.deallocate_workspace(workspace).unwrap();
    }
    ep.deallocate(a_buf).unwrap();
    ep.deallocate(b_buf).unwrap();
    ep.deallocate(c_buf).unwrap();

    bytes_to_f32(&out_bytes)
}

/// Run the pre-fast-path cuBLASLt implementation directly as a numeric oracle.
fn run_cublaslt_gemv_f32(
    ep: &CudaExecutionProvider,
    a: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
) -> Vec<f32> {
    let rt = ep.runtime();
    let a_buf = ep.allocate(std::mem::size_of_val(a), 256).unwrap();
    let b_buf = ep.allocate(std::mem::size_of_val(b), 256).unwrap();
    let c_buf = ep.allocate(n * std::mem::size_of::<f32>(), 256).unwrap();
    let workspace = rt.alloc_raw(WORKSPACE_BYTES).unwrap();
    unsafe {
        rt.htod(f32_bytes(a), cuptr(a_buf.as_ptr())).unwrap();
        rt.htod(f32_bytes(b), cuptr(b_buf.as_ptr())).unwrap();
        blas::gemm(
            rt.blas(),
            rt.stream_ptr(),
            &GemmParams {
                dtype: GemmDtype::F32,
                a: cuptr(a_buf.as_ptr()),
                b: cuptr(b_buf.as_ptr()),
                c: cuptr(c_buf.as_ptr()),
                m: 1,
                k,
                n,
                batch: 1,
                a_batch_stride: 0,
                b_batch_stride: 0,
                epilogue: None,
            },
            workspace,
            WORKSPACE_BYTES,
        )
        .unwrap();
    }
    rt.synchronize().unwrap();
    let mut bytes = vec![0u8; n * std::mem::size_of::<f32>()];
    unsafe {
        rt.dtoh(&mut bytes, cuptr(c_buf.as_ptr())).unwrap();
        rt.free_raw(workspace).unwrap();
    }
    ep.deallocate(a_buf).unwrap();
    ep.deallocate(b_buf).unwrap();
    ep.deallocate(c_buf).unwrap();
    bytes_to_f32(&bytes)
}

fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn matmul_f32_on_gpu_matches_cpu_reference() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
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

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn matmul_f32_gemv_matches_cublaslt_bitwise() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    for &(k, n) in &[(37usize, 17usize), (259, 300), (2048, 64)] {
        let a: Vec<f32> = (0..k)
            .map(|i| (((i * 29 + 7) % 251) as f32 - 125.0) / 97.0)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| (((i * 37 + 11) % 509) as f32 - 254.0) / 193.0)
            .collect();
        let gemv = run_gpu_matmul_f32(&ep, &a, &b, &[1, k], &[k, n], &[1, n]);
        let cublaslt = run_cublaslt_gemv_f32(&ep, &a, &b, k, n);
        assert_eq!(
            gemv.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            cublaslt
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "f32 GEMV must be bit-identical to cuBLASLt for K={k}, N={n}"
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn matmul_f32_gemv_is_capture_safe_after_warmup() {
    #[cfg(feature = "gpu-tests")]
    matmul_f32_gemv_is_capture_safe_after_warmup_gpu();
}

#[cfg(feature = "gpu-tests")]
fn matmul_f32_gemv_is_capture_safe_after_warmup_gpu() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };
    let rt = ep.runtime();
    let dev = ep.device_id();
    let (k, n) = (259usize, 300usize);
    let a: Vec<f32> = (0..k).map(|i| i as f32 / 257.0 - 0.5).collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| (i.wrapping_mul(17) % 127) as f32 / 127.0 - 0.5)
        .collect();
    let a_buf = ep.allocate(std::mem::size_of_val(&a[..]), 256).unwrap();
    let b_buf = ep.allocate(std::mem::size_of_val(&b[..]), 256).unwrap();
    let mut c_buf = ep.allocate(n * std::mem::size_of::<f32>(), 256).unwrap();
    unsafe {
        rt.htod(f32_bytes(&a), cuptr(a_buf.as_ptr())).unwrap();
        rt.htod(f32_bytes(&b), cuptr(b_buf.as_ptr())).unwrap();
    }
    let a_shape = [1, k];
    let b_shape = [k, n];
    let c_shape = [1, n];
    let a_strides = compute_contiguous_strides(&a_shape);
    let b_strides = compute_contiguous_strides(&b_shape);
    let c_strides = compute_contiguous_strides(&c_shape);
    let a_view = TensorView::new(
        DevicePtr(a_buf.as_ptr()),
        DataType::Float32,
        &a_shape,
        &a_strides,
        dev,
    );
    let b_view = TensorView::new(
        DevicePtr(b_buf.as_ptr()),
        DataType::Float32,
        &b_shape,
        &b_strides,
        dev,
    );
    let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
    let kernel = ep
        .get_kernel(&node, &[a_shape.to_vec(), b_shape.to_vec()], 17)
        .unwrap();
    let workspace = std::cell::RefCell::new(None);
    let execute = |c_buf: &mut onnx_runtime_ep_api::DeviceBuffer| {
        common::execute_kernel(
            &ep,
            kernel.as_ref(),
            &[a_view.clone(), b_view.clone()],
            &mut [TensorMut::new(
                DevicePtrMut(c_buf.as_mut_ptr()),
                DataType::Float32,
                &c_shape,
                &c_strides,
                dev,
            )],
            &mut workspace.borrow_mut(),
        )
        .unwrap();
    };

    execute(&mut c_buf);
    assert!(
        kernel.capture_support().is_supported(),
        "warmed f32 M==1 signature must advertise capture support"
    );
    assert!(kernel.device_graph_resources().is_empty());
    let b_m = 2usize;
    let b_a: Vec<f32> = (0..b_m * k)
        .map(|i| (i.wrapping_mul(29) % 211) as f32 / 211.0 - 0.5)
        .collect();
    let b_a_buf = ep.allocate(std::mem::size_of_val(&b_a[..]), 256).unwrap();
    let mut b_c_buf = ep
        .allocate(b_m * n * std::mem::size_of::<f32>(), 256)
        .unwrap();
    unsafe {
        rt.htod(f32_bytes(&b_a), cuptr(b_a_buf.as_ptr())).unwrap();
    }
    let b_a_shape = [b_m, k];
    let b_c_shape = [b_m, n];
    let b_a_strides = compute_contiguous_strides(&b_a_shape);
    let b_c_strides = compute_contiguous_strides(&b_c_shape);
    let b_a_view = TensorView::new(
        DevicePtr(b_a_buf.as_ptr()),
        DataType::Float32,
        &b_a_shape,
        &b_a_strides,
        dev,
    );
    let execute_b = |b_c_buf: &mut onnx_runtime_ep_api::DeviceBuffer| {
        common::execute_kernel(
            &ep,
            kernel.as_ref(),
            &[b_a_view.clone(), b_view.clone()],
            &mut [TensorMut::new(
                DevicePtrMut(b_c_buf.as_mut_ptr()),
                DataType::Float32,
                &b_c_shape,
                &b_c_strides,
                dev,
            )],
            &mut workspace.borrow_mut(),
        )
    };

    let counts_before_failure = rt.allocation_counts();
    rt.fail_warm_transaction_at_for_test(1);
    let failure = execute_b(&mut b_c_buf).unwrap_err().to_string();
    assert!(
        failure.contains("injected staged warm-cache failure after MatMul dense host-plan"),
        "{failure}"
    );
    assert!(
        !kernel.capture_support().is_supported(),
        "a failed refinement must revoke capture readiness before admission"
    );
    assert!(kernel.device_graph_resources().is_empty());
    assert_eq!(
        rt.allocation_counts(),
        counts_before_failure,
        "host planning must not allocate private device scratch"
    );
    // Prepare both host plans and the shared physical high-water mark before
    // installing any graph; neither route may grow that owner afterwards.
    execute_b(&mut b_c_buf).unwrap();
    execute(&mut c_buf);
    let allocations = rt.allocation_counts();
    let pins = workspace
        .borrow()
        .as_ref()
        .map(|owner| {
            owner
                .device_graph_resource()
                .unwrap()
                .expect("real workspace owner")
        })
        .into_iter()
        .collect();
    rt.begin_graph_capture_with_resources(&[kernel.as_ref()], pins)
        .unwrap();
    execute(&mut c_buf);
    assert!(
        rt.is_capturing().unwrap(),
        "execute must not synchronize and invalidate stream capture"
    );
    assert_eq!(
        rt.allocation_counts(),
        allocations,
        "captured f32 GEMV must not allocate or free"
    );
    rt.end_graph_capture().unwrap();
    assert_eq!(rt.allocation_counts(), allocations);
    execute_b(&mut b_c_buf).unwrap();
    rt.replay_graph().unwrap();
    rt.synchronize().unwrap();
    let mut replay_bytes = vec![0u8; n * std::mem::size_of::<f32>()];
    unsafe {
        rt.dtoh(&mut replay_bytes, cuptr(c_buf.as_ptr())).unwrap();
    }
    let replay = replay_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    let expected = cpu_reference_nd(&a, &b, &a_shape, &b_shape);
    assert!(approx_eq(&replay, &expected, 1e-4));
    assert!(rt.reset_graph().unwrap());

    if let Some(workspace) = workspace.into_inner() {
        ep.deallocate_workspace(workspace).unwrap();
    }
    ep.deallocate(a_buf).unwrap();
    ep.deallocate(b_buf).unwrap();
    ep.deallocate(c_buf).unwrap();
    ep.deallocate(b_a_buf).unwrap();
    ep.deallocate(b_c_buf).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn matmul_dense_plans_borrow_one_caller_workspace() {
    #[cfg(feature = "gpu-tests")]
    matmul_dense_plans_borrow_one_caller_workspace_gpu();
}

#[cfg(feature = "gpu-tests")]
fn matmul_dense_plans_borrow_one_caller_workspace_gpu() {
    let injected = Arc::new(common::ExternalEagerAllocator::new(
        common::require_context("MatMul workspace-authority test"),
    ));
    let ep = CudaExecutionProvider::from_config(
        0,
        onnx_runtime_ep_cuda::CudaProviderConfig {
            memory: Some(Arc::clone(&injected) as Arc<dyn DeviceAllocator>),
            ..Default::default()
        },
    )
    .expect("install constrained test allocator");
    let runtime = ep.runtime();
    let device = ep.device_id();

    let (m, k, n) = (64usize, 512usize, 384usize);
    let a_shape = [m, k];
    let b_shape = [k, n];
    let output_shape = [m, n];
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let a_buffer = ep.allocate(a.len() * 4, 256).unwrap();
    let b_buffer = ep.allocate(b.len() * 4, 256).unwrap();
    let mut output_buffer = ep.allocate(m * n * 4 + 4, 256).unwrap();
    unsafe {
        runtime
            .htod(f32_bytes(&a), cuptr(a_buffer.as_ptr()))
            .unwrap();
        runtime
            .htod(f32_bytes(&b), cuptr(b_buffer.as_ptr()))
            .unwrap();
    }
    let a_strides = compute_contiguous_strides(&a_shape);
    let b_strides = compute_contiguous_strides(&b_shape);
    let output_strides = compute_contiguous_strides(&output_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(a_buffer.as_ptr()),
            DataType::Float32,
            &a_shape,
            &a_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(b_buffer.as_ptr()),
            DataType::Float32,
            &b_shape,
            &b_strides,
            device,
        ),
    ];
    let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
    let kernel = ep
        .get_kernel(&node, &[a_shape.to_vec(), b_shape.to_vec()], 17)
        .unwrap();

    macro_rules! output {
        () => {
            TensorMut::new(
                DevicePtrMut(output_buffer.as_mut_ptr()),
                DataType::Float32,
                &output_shape,
                &output_strides,
                device,
            )
        };
    }
    macro_rules! refine {
        () => {
            common::runtime_workspace_requirement(
                kernel.as_ref(),
                &inputs,
                &[onnx_runtime_ep_api::OutputMetadata::tensor(
                    DataType::Float32,
                    &output_shape,
                )],
                &[output!()],
            )
            .unwrap()
        };
    }
    macro_rules! execute {
        ($view:expr) => {
            kernel.execute_with_workspace(&inputs, &mut [output!()], $view)
        };
    }
    let external_after_tensors = injected.cumemalloc_calls();
    let runtime_before_planning = runtime.allocation_counts();
    let requirement = refine!();
    assert!(
        requirement.bytes > 0,
        "this scratch regression needs a non-zero cuBLASLt requirement"
    );
    assert_eq!(
        requirement.lifetime,
        onnx_runtime_ep_api::WorkspaceLifetime::SessionPersistent
    );
    assert_eq!(
        runtime.allocation_counts(),
        runtime_before_planning,
        "the candidate contains host descriptors only"
    );
    assert_eq!(injected.cumemalloc_calls(), external_after_tensors);
    assert!(!kernel.capture_support().is_supported());
    let mut workspace = None;
    injected.fail_next_allocation();
    let error = common::prepare_workspace(&ep, requirement, &mut workspace)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("injected workspace allocation failure"),
        "{error}"
    );
    assert!(
        !kernel.capture_support().is_supported(),
        "quota failure cannot leave capture proof or trigger fallback"
    );
    let requirement = refine!();
    let mut view = common::prepare_workspace(&ep, requirement, &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(injected.cumemalloc_calls(), external_after_tensors + 1);
    let error = execute!(None).unwrap_err().to_string();
    assert!(error.contains("workspace needs"), "{error}");
    assert!(!kernel.capture_support().is_supported());
    refine!();
    let short = onnx_runtime_ep_api::WorkspaceView::new(view.ptr(), requirement.bytes as usize - 1);
    assert!(
        execute!(Some(short))
            .unwrap_err()
            .to_string()
            .contains("workspace needs")
    );
    refine!();
    execute!(Some(view)).unwrap();
    assert!(kernel.capture_support().is_supported());
    assert!(
        kernel.device_graph_resources().is_empty(),
        "all physical ownership belongs to the caller lane"
    );
    let counts = runtime.allocation_counts();
    let external = injected.cumemalloc_calls();
    for _ in 0..8 {
        let repeated = refine!();
        assert_eq!(repeated, requirement);
        common::prepare_workspace(&ep, repeated, &mut workspace).unwrap();
        execute!(Some(view)).unwrap();
    }
    assert_eq!(runtime.allocation_counts(), counts);
    assert_eq!(
        injected.cumemalloc_calls(),
        external,
        "warm fit must reuse the caller owner"
    );
    // Establish both nodes' peak before installing a graph; the owner is
    // frozen throughout replay, even if two heuristic choices differ in size.
    let other = ep
        .get_kernel(&node, &[a_shape.to_vec(), b_shape.to_vec()], 17)
        .unwrap();
    runtime.synchronize().unwrap();
    common::execute_kernel(
        &ep,
        other.as_ref(),
        &inputs,
        &mut [output!()],
        &mut workspace,
    )
    .unwrap();
    let owner = workspace.as_mut().unwrap();
    view = onnx_runtime_ep_api::WorkspaceView::new(DevicePtrMut(owner.as_mut_ptr()), owner.len());
    refine!();
    execute!(Some(view)).unwrap();
    let counts = runtime.allocation_counts();
    let external = injected.cumemalloc_calls();
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    refine!();
    let missing_pin = execute!(Some(view)).unwrap_err().to_string();
    runtime.abort_graph_capture().unwrap();
    assert!(
        missing_pin.contains("real workspace owner"),
        "{missing_pin}"
    );
    assert!(!kernel.capture_support().is_supported());
    refine!();
    execute!(Some(view)).unwrap();
    let pin = workspace
        .as_ref()
        .unwrap()
        .device_graph_resource()
        .unwrap()
        .expect("real caller workspace pin");
    runtime
        .begin_graph_capture_with_resources(&[kernel.as_ref()], vec![pin])
        .unwrap();
    refine!();
    execute!(Some(view)).unwrap();
    runtime.end_graph_capture().unwrap();
    assert_eq!(runtime.allocation_counts(), counts);
    // Another node borrows the same physical owner between replays. Scratch is
    // disposable, so neither cached plan may rely on its previous contents.
    for _ in 0..3 {
        unsafe {
            runtime
                .htod(&vec![0xff; view.bytes()], cuptr(view.ptr().0.cast_const()))
                .unwrap();
        }
        common::execute_kernel(
            &ep,
            other.as_ref(),
            &inputs,
            &mut [output!()],
            &mut workspace,
        )
        .unwrap();
        runtime.replay_graph().unwrap();
        let mut output = vec![0u8; m * n * 4];
        unsafe {
            runtime
                .dtoh(&mut output, cuptr(output_buffer.as_ptr()))
                .unwrap();
        }
        assert!(bytes_to_f32(&output).iter().all(|&value| value == k as f32));
    }
    assert_eq!(runtime.allocation_counts(), counts);
    assert_eq!(injected.cumemalloc_calls(), external);
    drop(other);
    drop(kernel);
    runtime.replay_graph().unwrap();
    runtime.synchronize().unwrap();
    assert_eq!(
        runtime.allocation_counts(),
        counts,
        "dropping kernels owns no device scratch"
    );
    assert!(runtime.reset_graph().unwrap());
    // A weaker-aligned output subview must be planned from its element origin,
    // including byte_offset, rather than the aligned backing allocation.
    let offset_kernel = ep
        .get_kernel(&node, &[a_shape.to_vec(), b_shape.to_vec()], 17)
        .unwrap();
    let mut shifted = output!();
    shifted.byte_offset = 4;
    common::execute_kernel(
        &ep,
        offset_kernel.as_ref(),
        &inputs,
        &mut [shifted],
        &mut workspace,
    )
    .unwrap();
    let mut shifted_output = vec![0u8; m * n * 4];
    unsafe {
        runtime
            .dtoh(
                &mut shifted_output,
                cuptr(output_buffer.as_ptr().cast::<u8>().wrapping_add(4).cast()),
            )
            .unwrap();
    }
    assert!(
        bytes_to_f32(&shifted_output)
            .iter()
            .all(|&value| value == k as f32)
    );
    ep.deallocate_workspace(workspace.take().unwrap()).unwrap();
    for buffer in [a_buffer, b_buffer, output_buffer] {
        ep.deallocate(buffer).unwrap();
    }
    common::drain_releases(&ep, "MatMul caller-workspace teardown");
    assert_eq!(injected.frees(), injected.cumemalloc_calls());
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
    let mut workspace = None;
    common::execute_kernel(
        ep,
        kernel.as_ref(),
        &[a_view, b_view],
        &mut [out_view],
        &mut workspace,
    )
    .unwrap();
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

    if let Some(workspace) = workspace {
        ep.deallocate_workspace(workspace).unwrap();
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn matmul_f16_gemv_on_gpu_matches_cpu_reference() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
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

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn matmul_rejects_unsupported_rank_and_dtype() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
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
