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
//! CUDA-graph capture coverage for the **NVRTC block-reduction** reduce path
//! (`ReduceSumSquare` and the other `ext_tags` ops, plus bf16, which cuDNN
//! cannot reduce). This is the Lever-B counterpart to the cuDNN-path coverage in
//! `reduce_capture_gpu.rs`.
//!
//! Before Lever B, only the Int64 DATA reduce used the cached i64 base/delta
//! offset tables. Every float/bf16 reduce that fell to the NVRTC kernel
//! (`ReduceSumSquare`/L1/L2/Prod/LogSum…, and all bf16 reduces) instead
//! allocated fresh `base`/`delta` device buffers (`cuMemAlloc`), uploaded them,
//! launched with an unconditional trailing `synchronize()`, and freed them —
//! all illegal inside a CUDA graph capture, and it never marked the call
//! capture-safe. So each such reduce was an eager seam; on Qwen3.6-35B-A3B the
//! RMSNorm-chain `ReduceSumSquare` in the 30 linear-attn hybrid layers were the
//! top remaining seam op after Lever A (60 per decode pass).
//!
//! The fix routes the NVRTC path through the same shape-keyed metadata cache the
//! Int64 path already used and gates the trailing sync on `!capturing`, so a
//! warmed fixed-shape NVRTC reduce records cleanly into a captured segment.
//! These tests prove:
//!   * a warmed `ReduceSumSquare` (f16 and f32) reports capture-supported and its
//!     captured replay is **byte-identical** to the eager result and matches the
//!     f32 oracle — single-axis/keepdims and multi-axis/reduce-all;
//!   * bf16 `ReduceSumSquare` (NVRTC-only — cuDNN cannot reduce bf16) also folds
//!     into a captured segment and replays byte-identically;
//!   * the metadata cache repopulates on an input-shape change across eager calls
//!     (no stale offset tables → no wrong bytes);
//!   * a signature change *during* capture is rejected (no silent stale reuse).
//!
//! The suite skips cleanly when no CUDA runtime is present.

use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, TensorMut, TensorView,
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

fn elem_size(dtype: DataType) -> usize {
    match dtype {
        DataType::Float32 => 4,
        DataType::Float16 | DataType::BFloat16 => 2,
        other => panic!("unsupported dtype {other:?}"),
    }
}

/// Encode f32 logical values into a dtype's storage bytes (f32/f16/bf16).
fn encode(values: &[f32], dtype: DataType) -> Vec<u8> {
    match dtype {
        DataType::Float32 => bytes(values),
        DataType::Float16 => bytes(&values.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()),
        DataType::BFloat16 => bytes(
            &values
                .iter()
                .map(|&v| bf16::from_f32(v))
                .collect::<Vec<_>>(),
        ),
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
        DataType::BFloat16 => raw
            .chunks_exact(2)
            .map(|c| bf16::from_ne_bytes(c.try_into().unwrap()).to_f32())
            .collect(),
        other => panic!("unsupported dtype {other:?}"),
    }
}

/// Build a persistent standalone `ReduceSumSquare` kernel with the axes supplied
/// as an Int64 **input** (opset 18), exercising the warmed-axes capture path
/// used by real decode graphs.
fn sumsquare_kernel(
    ep: &CudaExecutionProvider,
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
    let mut node = Node::new(
        NodeId(0),
        "ReduceSumSquare",
        vec![Some(data), Some(axes)],
        vec![y],
    );
    node.attributes
        .insert("keepdims".into(), Attribute::Int(keepdims as i64));
    let node_id = graph.insert_node(node);
    let model = Model::new(&graph);
    ep.get_kernel(model.graph.node(node_id), &[], 18)
        .expect("standalone ReduceSumSquare kernel must be supported")
}

#[allow(clippy::too_many_arguments)]
fn execute_reduce(
    kernel: &dyn Kernel,
    dtype: DataType,
    data: &DeviceBuffer,
    data_shape: &[usize],
    axes: &DeviceBuffer,
    n_axes: usize,
    output: &mut DeviceBuffer,
    out_shape: &[usize],
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
    kernel.execute(&inputs, &mut outputs)
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

/// f32 sum-of-squares oracle reducing the trailing `cols` of each of `rows`.
fn oracle_sumsquare_last_axis(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| (0..cols).map(|c| values[r * cols + c].powi(2)).sum())
        .collect()
}

/// f32 sum-of-squares oracle reducing every element (reduce-all).
fn oracle_sumsquare_all(values: &[f32]) -> f32 {
    values.iter().map(|v| v.powi(2)).sum()
}

/// A deterministic, non-trivial signed distribution so the reduction actually
/// accumulates (not an all-equal degenerate case).
fn varied(count: usize, seed: f32) -> Vec<f32> {
    (0..count)
        .map(|i| (i as f32 * 0.7 + seed).sin() * 2.5 + ((i % 5) as f32 - 2.0) * 0.6)
        .collect()
}

fn tol_for(dtype: DataType) -> f32 {
    match dtype {
        DataType::Float32 => 1e-4,
        DataType::Float16 => 1e-2,
        // bf16 carries only ~8 bits of mantissa; sum-of-squares magnifies the
        // rounding, so allow a looser relative band.
        DataType::BFloat16 => 6e-2,
        other => panic!("unsupported dtype {other:?}"),
    }
}

/// Warm the exact `ReduceSumSquare` signature, capture it into a CUDA graph, and
/// assert the replayed output is byte-identical to the eager output, folds into
/// a single captured segment (no eager seam), and matches the f32 oracle. Runs
/// several decode steps with fresh inputs each step.
fn sumsquare_capture_matches_eager_and_oracle(
    dtype: DataType,
    in_shape: &[usize],
    axes_values: &[i64],
    out_shape: &[usize],
    rows: usize,
    cols: usize,
) {
    let ep = require_cuda();
    let runtime = ep.runtime();

    let n = rows * cols;
    let out_elems: usize = out_shape.iter().product();
    let n_axes = axes_values.len();
    let elem = elem_size(dtype);

    let kernel = sumsquare_kernel(&ep, dtype, in_shape, n_axes, true, out_shape);

    let data = ep.allocate(n * elem, 256).expect("allocate data");
    let axes = ep
        .allocate(n_axes.max(1) * std::mem::size_of::<i64>(), 256)
        .expect("allocate axes");
    let mut eager = ep
        .allocate(out_elems * elem, 256)
        .expect("allocate eager out");
    let mut captured = ep
        .allocate(out_elems * elem, 256)
        .expect("allocate captured out");

    // SAFETY: the axes allocation exactly covers the axes tensor.
    unsafe {
        runtime
            .htod(&bytes(axes_values), cuptr(axes.as_ptr()))
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

        // Warm the exact signature (eager): populates the cached base/delta
        // offset tables and warmed axes, and marks the reduce capture-supported.
        execute_reduce(
            kernel.as_ref(),
            dtype,
            &data,
            in_shape,
            &axes,
            n_axes,
            &mut eager,
            out_shape,
        )
        .expect("eager ReduceSumSquare");
        assert!(
            kernel.cuda_graph_compatible(),
            "warmed ReduceSumSquare must be capture-supported ({dtype:?})"
        );

        let kernels: [&dyn Kernel; 1] = [kernel.as_ref()];
        runtime
            .begin_graph_capture(&kernels)
            .expect("begin ReduceSumSquare CUDA graph capture");
        execute_reduce(
            kernel.as_ref(),
            dtype,
            &data,
            in_shape,
            &axes,
            n_axes,
            &mut captured,
            out_shape,
        )
        .expect("captured ReduceSumSquare must record without host fallback");
        runtime
            .end_graph_capture()
            .expect("ReduceSumSquare must record without host fallback");
        assert!(
            runtime.has_graph_executable().unwrap(),
            "ReduceSumSquare did not install a CUDA graph"
        );
        assert_eq!(
            runtime.graph_segment_count().unwrap(),
            1,
            "warmed ReduceSumSquare must fold into a single captured segment (no eager seam)"
        );
        runtime
            .replay_graph()
            .expect("replay captured ReduceSumSquare");

        // Byte-identical: capture must not perturb numerics.
        assert_eq!(
            read(&ep, &captured, out_elems * elem),
            read(&ep, &eager, out_elems * elem),
            "captured ReduceSumSquare diverged from eager at step {step} ({dtype:?})"
        );
        assert_eq!(
            runtime.check_capture_error().unwrap(),
            0,
            "ReduceSumSquare capture error latched at step {step}"
        );

        // And both match the f32 oracle over the same (dtype-rounded) inputs.
        let seen = decode(&encode(&logical, dtype), dtype);
        let expected: Vec<f32> = if out_elems == 1 {
            vec![oracle_sumsquare_all(&seen)]
        } else {
            oracle_sumsquare_last_axis(&seen, rows, cols)
        };
        let got = decode(&read(&ep, &captured, out_elems * elem), dtype);
        let tol = tol_for(dtype);
        for (g, e) in got.iter().zip(&expected) {
            assert!(
                (g - e).abs() <= tol + tol * e.abs(),
                "ReduceSumSquare mismatch vs oracle at step {step}: got {g}, expected {e} ({dtype:?})"
            );
        }

        assert!(
            runtime.reset_graph().unwrap(),
            "captured ReduceSumSquare graph was not installed"
        );
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
fn fp16_reduce_sumsquare_captures_and_matches_eager() {
    // Router-style shape: [1,5,8] reduce trailing axis -> [1,5,1], keepdims.
    sumsquare_capture_matches_eager_and_oracle(
        DataType::Float16,
        &[1, 5, 8],
        &[2],
        &[1, 5, 1],
        5,
        8,
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn f32_reduce_sumsquare_captures_and_matches_eager() {
    sumsquare_capture_matches_eager_and_oracle(
        DataType::Float32,
        &[1, 5, 8],
        &[2],
        &[1, 5, 1],
        5,
        8,
    );
}

/// bf16 has no cuDNN reduce, so `ReduceSumSquare` bf16 exercises the NVRTC path
/// exclusively. Proves it, too, is now capture-eligible (was always an eager
/// seam before Lever B).
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn bf16_reduce_sumsquare_captures_and_matches_eager() {
    sumsquare_capture_matches_eager_and_oracle(
        DataType::BFloat16,
        &[1, 5, 8],
        &[2],
        &[1, 5, 1],
        5,
        8,
    );
}

/// Multi-axis reduce-all (`axes = [1,2]`, keepdims) folds into a captured
/// segment and matches the f32 oracle — proving the cached offset tables cover
/// multi-axis reductions, not just the trailing axis.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn fp16_reduce_sumsquare_multi_axis_captures_and_matches_eager() {
    sumsquare_capture_matches_eager_and_oracle(
        DataType::Float16,
        &[1, 5, 8],
        &[1, 2],
        &[1, 1, 1],
        5,
        8,
    );
}

/// A single kernel driven across two shapes proves the metadata cache key
/// includes the shape: warm shape A, run shape B, then A again — each must be
/// numerically correct (a shape-blind cache would reuse A's base/delta offset
/// tables for B and corrupt the result).
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn nvrtc_reduce_sumsquare_alternating_shapes_stays_correct() {
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

    let kernel = sumsquare_kernel(&ep, dtype, &[4, 8], 1, true, &[4, 1]);

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
            kernel.as_ref(),
            dtype,
            &data,
            &in_shape,
            &axes,
            1,
            &mut out,
            &out_shape,
        )
        .expect("eager ReduceSumSquare");
        let seen = decode(&encode(&logical, dtype), dtype);
        let expected = oracle_sumsquare_last_axis(&seen, rows, cols);
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
    ep.deallocate(axes).unwrap();
}

/// Shape-change *during* capture must be rejected rather than silently reusing
/// stale offset tables. Warm shape A, begin capture, then execute a different
/// shape B: the metadata cache miss during capture must error. The capture is
/// then aborted cleanly.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn nvrtc_reduce_sumsquare_shape_change_under_capture_is_rejected() {
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

    let kernel = sumsquare_kernel(&ep, dtype, &[4, 8], 1, true, &[4, 1]);

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
        kernel.as_ref(),
        dtype,
        &data_a,
        &a_in,
        &axes,
        1,
        &mut out_a,
        &a_out,
    )
    .expect("eager warm ReduceSumSquare");
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
    let result = execute_reduce(
        kernel.as_ref(),
        dtype,
        &data_b,
        &b_in,
        &axes,
        1,
        &mut out_b,
        &b_out,
    );
    assert!(
        result.is_err(),
        "a ReduceSumSquare signature change during capture must be rejected, not silently reuse stale offset tables"
    );
    runtime
        .abort_graph_capture()
        .expect("abort the half-recorded capture");
    let _ = runtime.reset_graph();

    for buffer in [out_b, data_b, out_a, data_a, axes] {
        ep.deallocate(buffer).expect("free CUDA test buffer");
    }
}
