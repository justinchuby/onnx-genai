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

/// Open the WebGPU provider, or `None` when this host has no usable adapter.
///
/// Deliberately does **not** call `initialize`. The ORT plugin EP ABI has no
/// initialize hook and dispatches straight into `get_kernel`, so a harness that
/// initialises first exercises a path real ORT never takes. It did, and it hid
/// a gate that made every node fail under ORT while these tests were green.
fn provider() -> Option<Provider> {
    match Provider::new(CubeclBackend::WebGpu, 0) {
        Ok(provider) => Some(provider),
        Err(error) => {
            if std::env::var("NXRT_REQUIRE_GPU_TESTS").is_ok_and(|value| value == "1") {
                panic!("NXRT_REQUIRE_GPU_TESTS=1 but no cubecl-webgpu device: {error}");
            }
            eprintln!("skipping: no cubecl-webgpu device on this host ({error})");
            None
        }
    }
}

/// The element types the kernels are generic over, as the tests need them.
///
/// Both dtypes go through exactly the same upload/run/download path so an f16
/// regression cannot hide behind a separate, weaker harness.
trait Elem: Copy + std::fmt::Debug {
    const DTYPE: DataType;
    fn to_bytes(values: &[Self]) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Vec<Self>;
}

impl Elem for f32 {
    const DTYPE: DataType = DataType::Float32;

    fn to_bytes(values: &[Self]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }
}

impl Elem for half::f16 {
    const DTYPE: DataType = DataType::Float16;

    fn to_bytes(values: &[Self]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        bytes
            .chunks_exact(2)
            .map(|chunk| half::f16::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }
}

/// Upload `values` to a fresh device allocation.
fn upload<E: Elem>(provider: &Provider, values: &[E]) -> onnx_runtime_ep_api::DeviceBuffer {
    let host = E::to_bytes(values);
    let mut buffer = provider
        .allocate(host.len().max(1), 256)
        .expect("allocate must succeed");
    provider
        .copy_from_host(&host, &mut buffer)
        .expect("host upload must succeed");
    buffer
}

fn download<E: Elem>(
    provider: &Provider,
    buffer: &onnx_runtime_ep_api::DeviceBuffer,
    len: usize,
) -> Vec<E> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<E>()];
    provider.sync().expect("sync must succeed");
    provider
        .copy_to_host(buffer, &mut bytes)
        .expect("host download must succeed");
    E::from_bytes(&bytes)
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
fn run<E: Elem>(
    provider: &Provider,
    op_type: &str,
    opset: u64,
    inputs: &[(&[usize], &[E])],
    out_shape: &[usize],
) -> Vec<E> {
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
                E::DTYPE,
                &shapes[index],
                &strides[index],
                device,
            )
        })
        .collect();

    let out_len: usize = out_shape.iter().product();
    let mut out_buffer = provider
        .allocate(out_len * std::mem::size_of::<E>(), 256)
        .expect("output allocation must succeed");
    let out_strides = contiguous_strides(out_shape);
    let mut outputs = vec![TensorMut::new(
        DevicePtrMut(out_buffer.as_mut_ptr()),
        E::DTYPE,
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
    assert_eq!(download::<f32>(&provider, &buffer, values.len()), values);
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
        .copy_from_host(&f32::to_bytes(&[1.0f32; 16]), &mut a)
        .unwrap();
    provider
        .copy_from_host(&f32::to_bytes(&[2.0f32; 16]), &mut b)
        .unwrap();
    assert_eq!(download::<f32>(&provider, &a, 16), vec![1.0f32; 16]);
    assert_eq!(download::<f32>(&provider, &b, 16), vec![2.0f32; 16]);
    provider.deallocate(a).unwrap();
    provider.deallocate(b).unwrap();
}

/// A provider must be usable without `initialize`, because ORT never calls it.
///
/// This is the exact shape of a bug that shipped: `get_kernel` was gated on an
/// explicit `initialize`, every test here called it, and under real ORT 1.28
/// every node failed with "get_kernel was called before initialize()". The
/// assertion is on `allocate` and `get_kernel` specifically -- the two entry
/// points ORT reaches first.
#[test]
fn a_freshly_constructed_provider_dispatches_without_initialize() {
    let Some(provider) = provider() else {
        return;
    };
    // Nothing between construction and use: no initialize, no config.
    let buffer = provider
        .allocate(64, 256)
        .expect("allocate must work on a freshly constructed provider");
    provider
        .deallocate(buffer)
        .expect("deallocate must succeed");

    let node = node("Relu");
    provider
        .get_kernel(&node, &[vec![4]], 13)
        .expect("get_kernel must work on a freshly constructed provider");
}

/// `initialize` stays accepted and stays a no-op, so a host that does call it
/// (the nxrt path does) is neither required to nor punished for it.
#[test]
fn initialize_remains_accepted_and_idempotent() {
    let Some(mut provider) = provider() else {
        return;
    };
    provider
        .initialize(&EpConfig::default())
        .expect("first initialize");
    provider
        .initialize(&EpConfig::default())
        .expect("second initialize must not fail");
    let out = run::<f32>(&provider, "Relu", 13, &[(&[2], &[-1.0, 2.0])], &[2]);
    assert_eq!(out, vec![0.0, 2.0]);
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
fn an_unimplemented_dtype_is_refused_with_a_named_reason() {
    let Some(provider) = provider() else { return };
    let match_result = provider.supports_op(
        &node("Add"),
        14,
        &[],
        &[DataType::Int32, DataType::Int32],
        &[],
    );
    let reason = match match_result {
        onnx_runtime_ep_api::KernelMatch::Unsupported { reason } => reason,
        other => panic!("i32 must be refused, got {other:?}"),
    };
    assert!(reason.contains("Int32"), "{reason}");
    assert!(reason.contains("f32 and f16"), "{reason}");
}

/// f16 must track the device probe in both directions.
///
/// An adapter without `shader-f16` has to be refused with a reason that names
/// the device feature, and an adapter with it has to actually accept the node.
/// Asserting only the branch this machine happens to take would let the other
/// one rot, so both are pinned here against the same probe the provider used.
#[test]
fn f16_support_follows_the_device_probe() {
    let Some(provider) = provider() else { return };
    let available = provider.supports_f16();
    let match_result = provider.supports_op(
        &node("Add"),
        14,
        &[],
        &[DataType::Float16, DataType::Float16],
        &[],
    );
    match (available, match_result) {
        (true, onnx_runtime_ep_api::KernelMatch::Supported { .. }) => {}
        (false, onnx_runtime_ep_api::KernelMatch::Unsupported { reason }) => {
            assert!(reason.contains("Float16"), "{reason}");
            assert!(reason.contains("shader-f16"), "{reason}");
        }
        (available, other) => panic!("probe said f16={available} but supports_op said {other:?}"),
    }
}

/// Skip an f16 test when this adapter cannot do f16, honouring the same
/// require-GPU escape hatch as the no-adapter case.
fn f16_or_skip(provider: &Provider) -> bool {
    if provider.supports_f16() {
        return true;
    }
    if std::env::var("NXRT_REQUIRE_GPU_TESTS").is_ok_and(|value| value == "1") {
        panic!("NXRT_REQUIRE_GPU_TESTS=1 but this adapter reports no f16 support");
    }
    eprintln!("skipping: this adapter reports no f16 support");
    false
}

fn f16s(values: &[f32]) -> Vec<half::f16> {
    values.iter().copied().map(half::f16::from_f32).collect()
}

#[test]
fn f16_add_matches_host_arithmetic() {
    let Some(provider) = provider() else { return };
    if !f16_or_skip(&provider) {
        return;
    }
    let lhs = f16s(&(0..300).map(|i| i as f32 * 0.25).collect::<Vec<_>>());
    let rhs = f16s(&(0..300).map(|i| i as f32 * -0.5).collect::<Vec<_>>());
    let result = run(
        &provider,
        "Add",
        7,
        &[(&[300], &lhs), (&[300], &rhs)],
        &[300],
    );
    for (index, value) in result.iter().enumerate() {
        let expected = lhs[index].to_f32() + rhs[index].to_f32();
        assert!(
            (value.to_f32() - expected).abs() <= 0.01,
            "index {index}: got {value:?}, expected {expected}"
        );
    }
}

#[test]
fn f16_relu_clamps_negatives() {
    let Some(provider) = provider() else { return };
    if !f16_or_skip(&provider) {
        return;
    }
    let input = f16s(&[-3.0, -0.5, 0.0, 0.5, 7.0, -1e-3]);
    let result = run(&provider, "Relu", 6, &[(&[6], &input)], &[6]);
    let expected = [0.0f32, 0.0, 0.0, 0.5, 7.0, 0.0];
    for (index, value) in result.iter().enumerate() {
        assert_eq!(value.to_f32(), expected[index], "index {index}");
    }
}

#[test]
fn f16_mul_broadcasts_a_scalar_operand() {
    let Some(provider) = provider() else { return };
    if !f16_or_skip(&provider) {
        return;
    }
    let lhs = f16s(&(0..64).map(|i| i as f32).collect::<Vec<_>>());
    let rhs = f16s(&[0.5]);
    let result = run(&provider, "Mul", 7, &[(&[64], &lhs), (&[1], &rhs)], &[64]);
    for (index, value) in result.iter().enumerate() {
        assert_eq!(value.to_f32(), index as f32 * 0.5, "index {index}");
    }
}

/// f16 `MatMul` over general values, against a reference over the same rounded
/// inputs so any difference is the kernel's arithmetic, not input quantisation.
#[test]
fn f16_matmul_matches_a_host_reference() {
    let Some(provider) = provider() else { return };
    if !f16_or_skip(&provider) {
        return;
    }
    let (m, k, n) = (8usize, 256usize, 8usize);
    let lhs = f16s(
        &(0..m * k)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.25)
            .collect::<Vec<_>>(),
    );
    let rhs = f16s(
        &(0..k * n)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.5)
            .collect::<Vec<_>>(),
    );

    let result = run(
        &provider,
        "MatMul",
        9,
        &[(&[m, k], &lhs), (&[k, n], &rhs)],
        &[m, n],
    );

    for row in 0..m {
        for col in 0..n {
            let mut expected = 0.0f32;
            for inner in 0..k {
                expected += lhs[row * k + inner].to_f32() * rhs[inner * n + col].to_f32();
            }
            let got = result[row * n + col].to_f32();
            // The output is stored as f16, so the tolerance is that type's
            // resolution at this magnitude, not an f32 epsilon.
            let tolerance = (expected.abs() * 1e-3).max(0.05);
            assert!(
                (got - expected).abs() <= tolerance,
                "[{row}][{col}]: got {got}, expected {expected}"
            );
        }
    }
}

/// Pin the f32 accumulator in `matmul_tiled` with a case that f16 accumulation
/// cannot pass.
///
/// This is a negative control, not a general correctness check. Summing 4096
/// ones is the classic large-plus-small cancellation: once an f16 running total
/// passes 2048 its ulp is 2, so adding 1.0 rounds straight back down and the
/// sum sticks near 2048 forever. An f32 accumulator reaches 4096 exactly. The
/// earlier, more natural-looking test with K=256 was verified *not* to
/// distinguish the two, so without this case the accumulator choice would be
/// an unverified comment.
#[test]
fn f16_matmul_accumulates_in_f32() {
    let Some(provider) = provider() else { return };
    if !f16_or_skip(&provider) {
        return;
    }
    let (m, k, n) = (8usize, 4096usize, 8usize);
    let ones = vec![half::f16::from_f32(1.0); m * k];
    let rhs = vec![half::f16::from_f32(1.0); k * n];

    let result = run(
        &provider,
        "MatMul",
        9,
        &[(&[m, k], &ones), (&[k, n], &rhs)],
        &[m, n],
    );

    // 4096 is exactly representable in f16, so a correct kernel is exact here
    // and there is no tolerance to argue about.
    for (index, value) in result.iter().enumerate() {
        assert_eq!(
            value.to_f32(),
            4096.0,
            "element {index}: an f16 accumulator stalls near 2048 here"
        );
    }
}

/// Covers `matmul_regtiled`, which nothing else in this file reaches: every
/// other MatMul test is below `REGTILE_MIN_M` and so exercises `matmul_tiled`.
///
/// The dimensions deliberately do not divide the block sizes — `M = 200`
/// against a 128-row block, `N = 150` against a 128-column block, `K = 43`
/// against an 8-deep block — so the partial blocks on all three axes are
/// exercised. A shape that divided evenly would pass even if every bounds
/// check were wrong.
#[test]
fn regtiled_matmul_matches_a_host_reference() {
    let Some(provider) = provider() else { return };
    let (m, k, n) = (200usize, 43usize, 150usize);
    let lhs: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32) - 5.0).collect();
    let rhs: Vec<f32> = (0..k * n).map(|i| ((i % 9) as f32) - 4.0).collect();
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
        70,
        &[(&[m, k], &lhs), (&[k, n], &rhs)],
        &[m, n],
    );
    for (index, (actual, want)) in result.iter().zip(&expected).enumerate() {
        assert!(
            (actual - want).abs() < 1e-3,
            "element {index} (row {}, col {}): {actual} != {want}",
            index / n,
            index % n
        );
    }
}

/// Covers the vec4 register-tiled kernel with partial blocks on every axis.
/// `K` and `N` satisfy the vec4 alignment precondition but deliberately do not
/// divide `BK`/`BN`, so vectorized loads still hit the zero-padding edges.
#[test]
fn vec4_regtiled_matmul_matches_a_host_reference() {
    let Some(provider) = provider() else { return };
    let (m, k, n) = (200usize, 44usize, 148usize);
    let lhs: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) - 6.0).collect();
    let rhs: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) - 3.0).collect();
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
        72,
        &[(&[m, k], &lhs), (&[k, n], &rhs)],
        &[m, n],
    );
    for (index, (actual, want)) in result.iter().zip(&expected).enumerate() {
        assert!(
            (actual - want).abs() < 1e-3,
            "element {index} (row {}, col {}): {actual} != {want}",
            index / n,
            index % n
        );
    }
}

/// The register-tiled path must also hold the f32-accumulation guarantee, and
/// batching must still address the right slice of a batched operand.
#[test]
fn regtiled_matmul_handles_batches() {
    let Some(provider) = provider() else { return };
    let (batch, m, k, n) = (3usize, 128usize, 32usize, 128usize);
    let lhs: Vec<f32> = (0..batch * m * k).map(|i| ((i % 7) as f32) - 3.0).collect();
    let rhs: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) - 2.0).collect();
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
        71,
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
