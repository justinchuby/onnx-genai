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
//! CUDA-graph capture coverage for the float (`f32`/`f16`) reduce paths.
//!
//! Before Lever A, every `f32`/`f16` `ReduceSum`/`ReduceMean` took the cuDNN
//! branch, which allocated a fresh device workspace (`cuMemAlloc`) per call and
//! ended with an unconditional stream `synchronize()`. Both are illegal inside a
//! CUDA graph capture, so the segmenter shredded the graph at every float
//! reduce — the dense-fallback MoE per-expert mask reduce fires this thousands
//! of times per decode step (256 experts × 40 layers on Qwen3.6-35B-A3B), which
//! is 98%+ of the eager seams that made native decode host/sync-bound.
//!
//! Lever A moved the live cuDNN workspace into the executor-owned persistent
//! workspace path, cached only the host descriptors + queried byte size across
//! calls with a stable signature, and gated the trailing sync on `!capturing`.
//! `ReduceKernel` now also routes **well-parallelised** f32 sum/mean (enough
//! outputs to fill the SMs, or a small per-output group — the common decode
//! shape) to the same capture-safe NVRTC block reduction f16/bf16 already use,
//! and keeps cuDNN only for the low-parallelism "few outputs, huge group"
//! regime. Both paths record cleanly into a captured segment. These tests prove:
//!   * a warmed float reduce reports capture-supported and its captured replay
//!     is **byte-identical** to the eager result (both f16 and f32, and both the
//!     NVRTC and the retained-cuDNN f32 regimes);
//!   * the descriptor cache plus prepared persistent workspace repopulate
//!     correctly when the input shape changes across eager calls (no stale
//!     workspace → no wrong bytes);
//!   * a signature change *during* capture is rejected (no silent stale reuse).
//!
//! The suite skips cleanly when no CUDA runtime is present.

mod common;

use std::sync::Arc;

use half::f16;
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, TensorMut, TensorView,
    WorkspaceAllocation,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

fn bytes<T: Copy>(values: &[T]) -> Vec<u8> {
    // SAFETY: test inputs are fixed-width plain data.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)).to_vec()
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

/// Encode f32 logical values into a dtype's storage bytes (f32 or f16).
fn encode(values: &[f32], dtype: DataType) -> Vec<u8> {
    match dtype {
        DataType::Float32 => bytes(values),
        DataType::Float16 => bytes(&values.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()),
        other => panic!("unsupported dtype {other:?}"),
    }
}

/// Decode a dtype's storage bytes back to f32 logical values.
fn decode(raw: &[u8], dtype: DataType) -> Vec<f32> {
    match dtype {
        DataType::Float32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect(),
        DataType::Float16 => raw
            .chunks_exact(2)
            .map(|c| f16::from_ne_bytes(c.try_into().unwrap()).to_f32())
            .collect(),
        other => panic!("unsupported dtype {other:?}"),
    }
}

/// Build a persistent standalone reduce kernel (`ReduceSum`/`ReduceMean`) with
/// the axes supplied as an Int64 **input** (opset 18), exercising the warmed
/// axes capture path used by real decode graphs.
fn reduce_kernel(
    ep: &CudaExecutionProvider,
    op: &str,
    dtype: DataType,
    in_shape: &[usize],
    n_axes: usize,
    keepdims: bool,
    out_shape: &[usize],
) -> Box<dyn Kernel> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 18);
    let data = graph.create_named_value("data", dtype, static_shape(in_shape.to_vec()));
    let axes = graph.create_named_value("axes", DataType::Int64, static_shape([n_axes]));
    graph.add_input(data);
    graph.add_input(axes);
    let y = graph.create_named_value("reduced", dtype, static_shape(out_shape.to_vec()));
    graph.add_output(y);
    let mut node = Node::new(NodeId(0), op, vec![Some(data), Some(axes)], vec![y]);
    node.attributes
        .insert("keepdims".into(), Attribute::Int(keepdims as i64));
    let node_id = graph.insert_node(node);
    let model = Model::new(&graph);
    ep.get_kernel(model.graph.node(node_id), &[], 18)
        .expect("standalone reduce kernel must be supported")
}

#[allow(clippy::too_many_arguments)]
fn execute_reduce(
    ep: &CudaExecutionProvider,
    kernel: &dyn Kernel,
    dtype: DataType,
    data: &DeviceBuffer,
    data_shape: &[usize],
    axes: &DeviceBuffer,
    n_axes: usize,
    output: &mut DeviceBuffer,
    out_shape: &[usize],
    workspace: &mut Option<WorkspaceAllocation>,
) {
    let data_strides = compute_contiguous_strides(data_shape);
    let axes_shape = [n_axes];
    let axes_strides = compute_contiguous_strides(&axes_shape);
    let out_strides = compute_contiguous_strides(out_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(data.as_ptr()),
            dtype,
            data_shape,
            &data_strides,
            data.device(),
        ),
        TensorView::new(
            DevicePtr(axes.as_ptr()),
            DataType::Int64,
            &axes_shape,
            &axes_strides,
            axes.device(),
        ),
    ];
    let mut outputs = [TensorMut::new(
        DevicePtrMut(output.as_mut_ptr()),
        dtype,
        out_shape,
        &out_strides,
        output.device(),
    )];
    common::execute_kernel(ep, kernel, &inputs, &mut outputs, workspace)
        .expect("standalone reduce execute");
}

#[allow(clippy::too_many_arguments)]
fn try_execute_reduce(
    ep: &CudaExecutionProvider,
    kernel: &dyn Kernel,
    dtype: DataType,
    data: &DeviceBuffer,
    data_shape: &[usize],
    axes: &DeviceBuffer,
    n_axes: usize,
    output: &mut DeviceBuffer,
    out_shape: &[usize],
    workspace: &mut Option<WorkspaceAllocation>,
) -> onnx_runtime_ep_api::Result<()> {
    let data_strides = compute_contiguous_strides(data_shape);
    let axes_shape = [n_axes];
    let axes_strides = compute_contiguous_strides(&axes_shape);
    let out_strides = compute_contiguous_strides(out_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(data.as_ptr()),
            dtype,
            data_shape,
            &data_strides,
            data.device(),
        ),
        TensorView::new(
            DevicePtr(axes.as_ptr()),
            DataType::Int64,
            &axes_shape,
            &axes_strides,
            axes.device(),
        ),
    ];
    let mut outputs = [TensorMut::new(
        DevicePtrMut(output.as_mut_ptr()),
        dtype,
        out_shape,
        &out_strides,
        output.device(),
    )];
    common::execute_kernel(ep, kernel, &inputs, &mut outputs, workspace)
}

fn read(ep: &CudaExecutionProvider, buffer: &DeviceBuffer, len: usize) -> Vec<u8> {
    let mut host = vec![0; len];
    // SAFETY: `buffer` owns at least `len` bytes in every caller.
    unsafe {
        ep.runtime()
            .dtoh(&mut host, cuptr(buffer.as_ptr()))
            .expect("copy CUDA output to host");
    }
    host
}

/// f32 CPU oracle: reduce the trailing axis of a `[rows, cols]` view.
fn oracle_reduce_last_axis(values: &[f32], rows: usize, cols: usize, is_mean: bool) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let sum: f32 = (0..cols).map(|c| values[r * cols + c]).sum();
            if is_mean { sum / cols as f32 } else { sum }
        })
        .collect()
}

/// A deterministic, non-trivial signed distribution so the reduction actually
/// accumulates (not an all-equal degenerate case).
fn varied(count: usize, seed: f32) -> Vec<f32> {
    (0..count)
        .map(|i| (i as f32 * 0.7 + seed).sin() * 2.5 + ((i % 5) as f32 - 2.0) * 0.6)
        .collect()
}

/// Warm the exact float reduce signature, capture it into a CUDA graph, and
/// assert the replayed output is byte-identical to the eager output and matches
/// the f32 oracle. Runs several decode steps with fresh inputs each step.
fn capture_matches_eager_and_oracle(op: &str, dtype: DataType, is_mean: bool) {
    // Router-style shape: [1,5,8] reduce trailing axis -> [1,5,1], keepdims.
    // 5 outputs / 8-element groups is the well-parallelised regime, so f32
    // takes the NVRTC block reduction (the f16/bf16 default); f16 always does.
    capture_matches_eager_and_oracle_shape(op, dtype, is_mean, 5, 8);
}

/// Same capture/oracle contract for the **low-parallelism** f32 regime
/// (`out_count` below the SM count *and* a large per-output group), where
/// `ReduceKernel` keeps the cuDNN `cudnnReduceTensor` path. Guards that the
/// retained cuDNN reduce still records cleanly into a single captured segment
/// and stays byte-identical to eager — coverage that the well-parallelised
/// shapes above now route to NVRTC instead.
fn capture_matches_eager_and_oracle_shape(
    op: &str,
    dtype: DataType,
    is_mean: bool,
    rows: usize,
    cols: usize,
) {
    let ep = require_cuda();
    let runtime = ep.runtime();

    let in_shape = [1usize, rows, cols];
    let out_shape = [1usize, rows, 1];
    let n = rows * cols;
    let axes_values = [2i64];

    let kernel = reduce_kernel(&ep, op, dtype, &in_shape, 1, true, &out_shape);

    let elem = if dtype == DataType::Float16 { 2 } else { 4 };
    let data = ep.allocate(n * elem, 256).expect("allocate data");
    let axes = ep
        .allocate(std::mem::size_of::<i64>(), 256)
        .expect("allocate axes");
    let mut eager = ep.allocate(rows * elem, 256).expect("allocate eager out");
    let mut captured = ep
        .allocate(rows * elem, 256)
        .expect("allocate captured out");
    let mut workspace = None;

    // SAFETY: the axes allocation exactly covers the single-axis tensor.
    unsafe {
        runtime
            .htod(&bytes(&axes_values), cuptr(axes.as_ptr()))
            .unwrap();
    }

    for step in 0..3 {
        let logical = varied(n, step as f32 * 0.31);
        // SAFETY: `data` exactly covers `n` elements of `dtype`.
        unsafe {
            runtime
                .htod(&encode(&logical, dtype), cuptr(data.as_ptr()))
                .unwrap();
        }

        // Warm the exact signature (eager). This populates the cuDNN descriptor
        // + workspace-byte cache, allocates the executor-owned persistent
        // workspace, warms the axes copy, and marks the reduce
        // capture-supported.
        execute_reduce(
            &ep,
            kernel.as_ref(),
            dtype,
            &data,
            &in_shape,
            &axes,
            1,
            &mut eager,
            &out_shape,
            &mut workspace,
        );
        assert!(
            kernel.cuda_graph_compatible(),
            "warmed float reduce must be capture-supported ({op} {dtype:?})"
        );

        let kernels: [&dyn Kernel; 1] = [kernel.as_ref()];
        runtime
            .begin_graph_capture(&kernels)
            .expect("begin float reduce CUDA graph capture");
        execute_reduce(
            &ep,
            kernel.as_ref(),
            dtype,
            &data,
            &in_shape,
            &axes,
            1,
            &mut captured,
            &out_shape,
            &mut workspace,
        );
        runtime
            .end_graph_capture()
            .expect("float reduce must record without host fallback");
        assert!(
            runtime.has_graph_executable().unwrap(),
            "float reduce did not install a CUDA graph"
        );
        assert_eq!(
            runtime.graph_segment_count().unwrap(),
            1,
            "warmed float reduce must fold into a single captured segment (no eager seam)"
        );
        runtime
            .replay_graph()
            .expect("replay captured float reduce");

        // Byte-identical: capture must not perturb numerics.
        assert_eq!(
            read(&ep, &captured, rows * elem),
            read(&ep, &eager, rows * elem),
            "captured float reduce diverged from eager at step {step} ({op} {dtype:?})"
        );
        assert_eq!(
            runtime.check_capture_error().unwrap(),
            0,
            "float reduce capture error latched at step {step}"
        );

        // And both match the f32 oracle over the same (dtype-rounded) inputs.
        let seen = decode(&encode(&logical, dtype), dtype);
        let expected = oracle_reduce_last_axis(&seen, rows, cols, is_mean);
        let got = decode(&read(&ep, &captured, rows * elem), dtype);
        let tol = if dtype == DataType::Float16 {
            1e-2
        } else {
            1e-4
        };
        for (g, e) in got.iter().zip(&expected) {
            assert!(
                (g - e).abs() <= tol + tol * e.abs(),
                "float reduce mismatch vs oracle at step {step}: got {g}, expected {e} ({op} {dtype:?})"
            );
        }

        assert!(
            runtime.reset_graph().unwrap(),
            "captured float reduce graph was not installed"
        );
    }

    if let Some(workspace) = workspace {
        ep.deallocate_workspace(workspace)
            .expect("free prepared reduce workspace");
    }
    for buffer in [captured, eager, axes, data] {
        ep.deallocate(buffer).expect("free CUDA test buffer");
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn fp16_reduce_sum_captures_and_matches_eager() {
    capture_matches_eager_and_oracle("ReduceSum", DataType::Float16, false);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn fp16_reduce_mean_captures_and_matches_eager() {
    capture_matches_eager_and_oracle("ReduceMean", DataType::Float16, true);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn f32_reduce_sum_captures_and_matches_eager() {
    capture_matches_eager_and_oracle("ReduceSum", DataType::Float32, false);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn f32_reduce_mean_captures_and_matches_eager() {
    capture_matches_eager_and_oracle("ReduceMean", DataType::Float32, true);
}

/// Low-parallelism f32 regime (1 output, 4096-element group) — below the SM
/// count with a large group, so `ReduceKernel` keeps the cuDNN reduce. Proves
/// the retained cuDNN f32 capture path still folds into one segment and matches
/// eager/oracle after the well-parallelised shapes were rerouted to NVRTC.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn f32_low_parallelism_reduce_sum_uses_cudnn_and_captures() {
    capture_matches_eager_and_oracle_shape("ReduceSum", DataType::Float32, false, 1, 4096);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn f32_low_parallelism_reduce_mean_uses_cudnn_and_captures() {
    capture_matches_eager_and_oracle_shape("ReduceMean", DataType::Float32, true, 1, 4096);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn f32_low_parallelism_reduce_prepared_workspace_uses_the_injected_allocator() {
    let injected = Arc::new(common::ExternalEagerAllocator::new(
        common::require_context("cuDNN Reduce prepared-workspace allocator substitution"),
    ));
    let ep =
        require_cuda()
            .with_memory(
                Arc::clone(&injected) as Arc<dyn onnx_runtime_memory_governor::DeviceAllocator>
            )
            .expect("the injected allocator must be accepted for the visible CUDA device");
    let runtime = ep.runtime();
    let dtype = DataType::Float32;
    let in_shape = [1usize, 1, 4096];
    let out_shape = [1usize, 1, 1];
    let kernel = reduce_kernel(&ep, "ReduceSum", dtype, &in_shape, 1, true, &out_shape);
    let n = in_shape.iter().product::<usize>();
    let data = ep.allocate(n * 4, 256).expect("allocate data");
    let axes = ep.allocate(8, 256).expect("allocate axes");
    let mut eager = ep.allocate(4, 256).expect("allocate eager out");
    let mut captured = ep.allocate(4, 256).expect("allocate captured out");
    unsafe {
        runtime
            .htod(&encode(&varied(n, 0.0), dtype), cuptr(data.as_ptr()))
            .unwrap();
        runtime.htod(&bytes(&[2i64]), cuptr(axes.as_ptr())).unwrap();
    }

    let calls_after_buffers = injected.cumemalloc_calls();
    let mut workspace = None;
    execute_reduce(
        &ep,
        kernel.as_ref(),
        dtype,
        &data,
        &in_shape,
        &axes,
        1,
        &mut eager,
        &out_shape,
        &mut workspace,
    );
    assert!(
        workspace.is_some(),
        "prepared reduce workspace must be allocated"
    );
    let calls_after_first = injected.cumemalloc_calls();
    assert_eq!(
        calls_after_first,
        calls_after_buffers + 1,
        "the first cuDNN reduce execute must allocate exactly one prepared workspace"
    );
    execute_reduce(
        &ep,
        kernel.as_ref(),
        dtype,
        &data,
        &in_shape,
        &axes,
        1,
        &mut captured,
        &out_shape,
        &mut workspace,
    );
    let calls_after_second = injected.cumemalloc_calls();
    assert_eq!(
        calls_after_second, calls_after_first,
        "repeated cuDNN reduce executes must reuse the prepared workspace"
    );
    let eager_bytes = read(&ep, &eager, 4);
    let repeat_bytes = read(&ep, &captured, 4);
    assert_eq!(
        repeat_bytes, eager_bytes,
        "reusing the prepared reduce workspace changed the output"
    );
    assert!(kernel.cuda_graph_compatible());

    let kernels: [&dyn Kernel; 1] = [kernel.as_ref()];
    runtime
        .begin_graph_capture(&kernels)
        .expect("begin reduce capture");
    execute_reduce(
        &ep,
        kernel.as_ref(),
        dtype,
        &data,
        &in_shape,
        &axes,
        1,
        &mut captured,
        &out_shape,
        &mut workspace,
    );
    runtime.end_graph_capture().expect("end reduce capture");
    runtime.replay_graph().expect("replay captured reduce");
    assert_eq!(
        injected.cumemalloc_calls(),
        calls_after_second,
        "recording or replaying the reduce graph must not allocate a new workspace"
    );
    assert_eq!(
        read(&ep, &captured, 4),
        eager_bytes,
        "captured reduce output diverged from eager output"
    );
    assert!(
        runtime.reset_graph().unwrap(),
        "reset captured reduce graph"
    );

    if let Some(workspace) = workspace.take() {
        ep.deallocate_workspace(workspace)
            .expect("free prepared reduce workspace");
    }
    for buffer in [captured, eager, axes, data] {
        ep.deallocate(buffer).expect("free CUDA test buffer");
    }
    common::drain_releases(&ep, "Reduce prepared workspace teardown");
    assert_eq!(
        injected.frees(),
        injected.cumemalloc_calls(),
        "the injected allocator must observe every reduce buffer/workspace free"
    );
}

/// A single kernel driven across two shapes to prove the cache key includes the
/// shape: warm shape A, then run shape B, then shape A again — each must be
/// numerically correct (a shape-blind cache would reuse A's workspace/descriptor
/// for B and corrupt the result).
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn float_reduce_same_kernel_alternating_shapes_stays_correct() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let dtype = DataType::Float16;
    let axes_values = [1i64];
    let axes = ep
        .allocate(std::mem::size_of::<i64>(), 256)
        .expect("allocate axes");
    // SAFETY: axes allocation covers the single-axis tensor.
    unsafe {
        runtime
            .htod(&bytes(&axes_values), cuptr(axes.as_ptr()))
            .unwrap();
    }

    // One shape-generic kernel: attribute-free, axes via input, so the same
    // kernel legitimately handles multiple runtime shapes.
    let kernel = reduce_kernel(&ep, "ReduceSum", dtype, &[4, 8], 1, true, &[4, 1]);
    let mut workspace = None;

    for &(rows, cols) in &[(4usize, 8usize), (2usize, 32usize), (4usize, 8usize)] {
        let in_shape = [rows, cols];
        let out_shape = [rows, 1usize];
        let n = rows * cols;
        let data = ep.allocate(n * 2, 256).expect("allocate data");
        let mut out = ep.allocate(rows * 2, 256).expect("allocate out");
        let logical = varied(n, 1.0 + cols as f32 * 0.02);
        // SAFETY: `data` covers `n` f16 elements.
        unsafe {
            runtime
                .htod(&encode(&logical, dtype), cuptr(data.as_ptr()))
                .unwrap();
        }
        execute_reduce(
            &ep,
            kernel.as_ref(),
            dtype,
            &data,
            &in_shape,
            &axes,
            1,
            &mut out,
            &out_shape,
            &mut workspace,
        );
        let seen = decode(&encode(&logical, dtype), dtype);
        let expected = oracle_reduce_last_axis(&seen, rows, cols, false);
        let got = decode(&read(&ep, &out, rows * 2), dtype);
        for (g, e) in got.iter().zip(&expected) {
            assert!(
                (g - e).abs() <= 1e-2 + 1e-2 * e.abs(),
                "alternating shape [{rows},{cols}] mismatch: got {g}, expected {e}"
            );
        }
        ep.deallocate(data).unwrap();
        ep.deallocate(out).unwrap();
    }
    if let Some(workspace) = workspace {
        ep.deallocate_workspace(workspace)
            .expect("free prepared reduce workspace");
    }
    ep.deallocate(axes).unwrap();
}

/// Shape-change *during* capture must be rejected rather than silently reusing a
/// stale workspace/descriptor. Warm shape A, begin capture, then execute a
/// different shape B: the cache miss during capture must error. The capture is
/// then aborted cleanly.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn float_reduce_shape_change_under_capture_is_rejected() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let dtype = DataType::Float16;
    let axes_values = [1i64];
    let axes = ep
        .allocate(std::mem::size_of::<i64>(), 256)
        .expect("allocate axes");
    // SAFETY: axes allocation covers the single-axis tensor.
    unsafe {
        runtime
            .htod(&bytes(&axes_values), cuptr(axes.as_ptr()))
            .unwrap();
    }

    let kernel = reduce_kernel(&ep, "ReduceSum", dtype, &[4, 8], 1, true, &[4, 1]);
    let mut workspace = None;

    // Shape A: warm it.
    let a_in = [4usize, 8];
    let a_out = [4usize, 1];
    let data_a = ep.allocate(4 * 8 * 2, 256).expect("allocate A data");
    let mut out_a = ep.allocate(4 * 2, 256).expect("allocate A out");
    // SAFETY: covers 32 f16 elements.
    unsafe {
        runtime
            .htod(&encode(&varied(32, 0.0), dtype), cuptr(data_a.as_ptr()))
            .unwrap();
    }
    execute_reduce(
        &ep,
        kernel.as_ref(),
        dtype,
        &data_a,
        &a_in,
        &axes,
        1,
        &mut out_a,
        &a_out,
        &mut workspace,
    );
    assert!(kernel.cuda_graph_compatible());

    // Shape B: a different, larger input the warmed cache does not cover.
    let b_in = [4usize, 16];
    let b_out = [4usize, 1];
    let data_b = ep.allocate(4 * 16 * 2, 256).expect("allocate B data");
    let mut out_b = ep.allocate(4 * 2, 256).expect("allocate B out");
    // SAFETY: covers 64 f16 elements.
    unsafe {
        runtime
            .htod(&encode(&varied(64, 1.0), dtype), cuptr(data_b.as_ptr()))
            .unwrap();
    }

    let kernels: [&dyn Kernel; 1] = [kernel.as_ref()];
    runtime
        .begin_graph_capture(&kernels)
        .expect("begin capture");
    let result = try_execute_reduce(
        &ep,
        kernel.as_ref(),
        dtype,
        &data_b,
        &b_in,
        &axes,
        1,
        &mut out_b,
        &b_out,
        &mut workspace,
    );
    assert!(
        result.is_err(),
        "a reduce signature change during capture must be rejected, not silently reuse the stale workspace"
    );
    runtime
        .abort_graph_capture()
        .expect("abort the half-recorded capture");
    let _ = runtime.reset_graph();

    if let Some(workspace) = workspace {
        ep.deallocate_workspace(workspace)
            .expect("free prepared reduce workspace");
    }
    for buffer in [out_b, data_b, out_a, data_a, axes] {
        ep.deallocate(buffer).expect("free CUDA test buffer");
    }
}
