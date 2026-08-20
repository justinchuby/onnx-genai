//! End-to-end GPU checks for the CubeCL WebGPU backend.
//!
//! These exercise the pieces that unit tests cannot reach: the synthetic
//! address table against a real allocator, host staging in both directions, and
//! each kernel actually running on a device. They are the only place the
//! numeric results are proven, so a change to the memory model or a kernel is
//! not verified until this file passes.
//!
//! # Skipping
//!
//! No GPU means no test rather than a failure, because a developer machine or a
//! CI container may legitimately have no adapter. Set `NXRT_REQUIRE_GPU_TESTS=1`
//! to turn "no adapter" into a hard failure, which is what CI on GPU runners
//! should do so a silently-skipping suite cannot masquerade as a passing one.

#![cfg(feature = "webgpu")]

use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, EpConfig, ExecutionProvider, TensorMut, TensorView,
};
use onnx_runtime_ep_cubecl::backend::CubeclBackend;
use onnx_runtime_ep_cubecl::provider::CubeclExecutionProvider;
use onnx_runtime_ep_cubecl::runtime::WebGpuRuntime;
use onnx_runtime_ir::{DataType, Node, NodeId};

type Provider = CubeclExecutionProvider<WebGpuRuntime>;

/// Open and initialise the WebGPU provider, or `None` when this host has no
/// usable adapter.
fn provider() -> Option<Provider> {
    match Provider::new(CubeclBackend::WebGpu, 0) {
        Ok(mut provider) => {
            provider
                .initialize(&EpConfig::default())
                .expect("initialize must succeed");
            Some(provider)
        }
        Err(error) => {
            if std::env::var("NXRT_REQUIRE_GPU_TESTS").is_ok_and(|value| value == "1") {
                panic!("NXRT_REQUIRE_GPU_TESTS=1 but no cubecl-webgpu device: {error}");
            }
            eprintln!("skipping: no cubecl-webgpu device on this host ({error})");
            None
        }
    }
}

/// Upload `values` to a fresh device allocation.
fn upload(provider: &Provider, values: &[f32]) -> onnx_runtime_ep_api::DeviceBuffer {
    let bytes = std::mem::size_of_val(values);
    let mut buffer = provider
        .allocate(bytes, 256)
        .expect("allocate must succeed");
    let host: &[u8] = bytemuck_cast(values);
    provider
        .copy_from_host(host, &mut buffer)
        .expect("host upload must succeed");
    buffer
}

fn download(
    provider: &Provider,
    buffer: &onnx_runtime_ep_api::DeviceBuffer,
    len: usize,
) -> Vec<f32> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    provider.sync().expect("sync must succeed");
    provider
        .copy_to_host(buffer, &mut bytes)
        .expect("host download must succeed");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Reinterpret f32s as bytes without pulling in a dependency for four lines.
fn bytemuck_cast(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding or invalid bit patterns, and the resulting
    // slice covers exactly the same bytes with a smaller alignment requirement.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn node(op_type: &str) -> Node {
    let mut node = Node::new(NodeId(0), op_type, Vec::new(), Vec::new());
    node.name = format!("test_{op_type}");
    node
}

fn contiguous_strides(shape: &[usize]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1] as i64;
    }
    strides
}

/// Run a single node with the given inputs and read back the output.
fn run(
    provider: &Provider,
    op_type: &str,
    opset: u64,
    inputs: &[(&[usize], &[f32])],
    out_shape: &[usize],
) -> Vec<f32> {
    let device = provider.device_id();
    let shapes: Vec<Vec<usize>> = inputs.iter().map(|(shape, _)| shape.to_vec()).collect();
    let kernel = provider
        .get_kernel(&node(op_type), &shapes, opset)
        .expect("kernel must be built");

    let buffers: Vec<_> = inputs
        .iter()
        .map(|(_, values)| upload(provider, values))
        .collect();
    let strides: Vec<Vec<i64>> = shapes
        .iter()
        .map(|shape| contiguous_strides(shape))
        .collect();
    let views: Vec<TensorView<'_>> = buffers
        .iter()
        .enumerate()
        .map(|(index, buffer)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                DataType::Float32,
                &shapes[index],
                &strides[index],
                device,
            )
        })
        .collect();

    let out_len: usize = out_shape.iter().product();
    let mut out_buffer = provider
        .allocate(out_len * 4, 256)
        .expect("output allocation must succeed");
    let out_strides = contiguous_strides(out_shape);
    let mut outputs = vec![TensorMut::new(
        DevicePtrMut(out_buffer.as_mut_ptr()),
        DataType::Float32,
        out_shape,
        &out_strides,
        device,
    )];

    kernel
        .execute(&views, &mut outputs)
        .expect("kernel execution must succeed");
    let result = download(provider, &out_buffer, out_len);

    provider
        .deallocate(out_buffer)
        .expect("output deallocation must succeed");
    for buffer in buffers {
        provider
            .deallocate(buffer)
            .expect("input deallocation must succeed");
    }
    result
}

#[test]
fn host_round_trip_preserves_bytes() {
    let Some(provider) = provider() else { return };
    let values: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
    let buffer = upload(&provider, &values);
    assert_eq!(download(&provider, &buffer, values.len()), values);
    provider.deallocate(buffer).unwrap();
}

#[test]
fn allocations_get_distinct_non_overlapping_addresses() {
    let Some(provider) = provider() else { return };
    let a = provider.allocate(64, 256).unwrap();
    let b = provider.allocate(64, 256).unwrap();
    assert_ne!(
        a.as_ptr(),
        b.as_ptr(),
        "two live allocations must not share an address"
    );
    // Writing one must not disturb the other, which is what the guard gap and
    // the per-allocation handle mapping exist to guarantee.
    let mut a = a;
    let mut b = b;
    provider
        .copy_from_host(bytemuck_cast(&[1.0f32; 16]), &mut a)
        .unwrap();
    provider
        .copy_from_host(bytemuck_cast(&[2.0f32; 16]), &mut b)
        .unwrap();
    assert_eq!(download(&provider, &a, 16), vec![1.0f32; 16]);
    assert_eq!(download(&provider, &b, 16), vec![2.0f32; 16]);
    provider.deallocate(a).unwrap();
    provider.deallocate(b).unwrap();
}

#[test]
fn double_free_is_reported_not_ignored() {
    let Some(provider) = provider() else { return };
    let buffer = provider.allocate(64, 256).unwrap();
    let ptr = buffer.as_ptr();
    let device = provider.device_id();
    provider.deallocate(buffer).unwrap();
    // SAFETY: the address is stale by construction; that is exactly what is
    // under test, and the provider never dereferences it on the host.
    let stale = unsafe {
        onnx_runtime_ep_api::DeviceBuffer::from_raw_parts(ptr.cast_mut(), device, 64, 256)
    };
    let error = provider
        .deallocate(stale)
        .expect_err("a double free must be rejected");
    assert!(error.to_string().contains("double free"), "{error}");
}

#[test]
fn add_matches_host_arithmetic() {
    let Some(provider) = provider() else { return };
    let lhs: Vec<f32> = (0..300).map(|i| i as f32).collect();
    let rhs: Vec<f32> = (0..300).map(|i| (i * 2) as f32).collect();
    let expected: Vec<f32> = lhs.iter().zip(&rhs).map(|(a, b)| a + b).collect();
    let result = run(
        &provider,
        "Add",
        14,
        &[(&[300], &lhs), (&[300], &rhs)],
        &[300],
    );
    assert_eq!(result, expected);
}

#[test]
fn mul_broadcasts_a_scalar_operand() {
    let Some(provider) = provider() else { return };
    let lhs: Vec<f32> = (0..300).map(|i| i as f32).collect();
    let expected: Vec<f32> = lhs.iter().map(|a| a * 3.0).collect();
    let result = run(
        &provider,
        "Mul",
        14,
        &[(&[300], &lhs), (&[1], &[3.0f32])],
        &[300],
    );
    assert_eq!(result, expected);
}

#[test]
fn relu_clamps_negatives() {
    let Some(provider) = provider() else { return };
    let input: Vec<f32> = (0..300).map(|i| i as f32 - 150.0).collect();
    let expected: Vec<f32> = input.iter().map(|v| v.max(0.0)).collect();
    let result = run(&provider, "Relu", 14, &[(&[300], &input)], &[300]);
    assert_eq!(result, expected);
}

#[test]
fn matmul_matches_a_host_reference() {
    let Some(provider) = provider() else { return };
    // Dimensions deliberately straddle the 16-wide tile so the kernel's
    // out-of-range staging and its final k-tile are both exercised.
    let (m, k, n) = (37usize, 23usize, 19usize);
    let lhs: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) - 3.0).collect();
    let rhs: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) - 2.0).collect();
    let mut expected = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += lhs[row * k + i] * rhs[i * n + col];
            }
            expected[row * n + col] = acc;
        }
    }
    let result = run(
        &provider,
        "MatMul",
        14,
        &[(&[m, k], &lhs), (&[k, n], &rhs)],
        &[m, n],
    );
    for (index, (actual, want)) in result.iter().zip(&expected).enumerate() {
        assert!(
            (actual - want).abs() < 1e-3,
            "element {index}: {actual} != {want}"
        );
    }
}

#[test]
fn batched_matmul_shares_the_right_hand_side() {
    let Some(provider) = provider() else { return };
    let (batch, m, k, n) = (3usize, 5usize, 4usize, 6usize);
    let lhs: Vec<f32> = (0..batch * m * k).map(|i| (i % 9) as f32).collect();
    let rhs: Vec<f32> = (0..k * n).map(|i| (i % 4) as f32).collect();
    let mut expected = vec![0.0f32; batch * m * n];
    for b in 0..batch {
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += lhs[b * m * k + row * k + i] * rhs[i * n + col];
                }
                expected[b * m * n + row * n + col] = acc;
            }
        }
    }
    let result = run(
        &provider,
        "MatMul",
        14,
        &[(&[batch, m, k], &lhs), (&[k, n], &rhs)],
        &[batch, m, n],
    );
    for (index, (actual, want)) in result.iter().zip(&expected).enumerate() {
        assert!(
            (actual - want).abs() < 1e-3,
            "element {index}: {actual} != {want}"
        );
    }
}

#[test]
fn unsupported_dtype_is_refused_with_a_named_reason() {
    let Some(provider) = provider() else { return };
    let match_result = provider.supports_op(
        &node("Add"),
        14,
        &[],
        &[DataType::Float16, DataType::Float16],
        &[],
    );
    let reason = match match_result {
        onnx_runtime_ep_api::KernelMatch::Unsupported { reason } => reason,
        other => panic!("f16 must be refused, got {other:?}"),
    };
    assert!(reason.contains("Float16"), "{reason}");
    assert!(reason.contains("f32"), "{reason}");
}
