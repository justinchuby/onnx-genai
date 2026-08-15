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
//! GPU parity tests for the half/bf16 `ReduceSum`/`ReduceMean` compute-type fix.
//!
//! `cudnnReduceTensor` rejects a half/bf16 `reduceTensorCompType`
//! (`CUDNN_STATUS_NOT_SUPPORTED`, reproduced on cuDNN 9.10 and 9.20); it requires
//! an `CUDNN_DATA_FLOAT` compute type for half I/O, and it cannot reduce bf16 at
//! all (rejected even with an f32 compute type). Before the fix, the CUDA EP's
//! `reduce_t` derived the compute type from the I/O dtype (via cudarc's
//! `create_reduction_no_indices::<T>`), so any half/bf16 `ReduceSum`/`ReduceMean`
//! placed on CUDA failed at execution — the exact blocker hit by the Qwen3.6
//! 35B-A3B MoE-router weight-normalization node
//! `decoder/model/layers.0/mlp/ReduceSum_node_121`
//! (`Float16 [1,5,8] -> Float16 [1,5,1]`).
//!
//! The fix reduces f16 through cuDNN with an f32 compute type and routes bf16 to
//! the typed NVRTC block reduction (which also accumulates in f32). Each case
//! runs the real CUDA kernel in half/bf16 and compares against an f32 CPU oracle
//! fed the *same* half-rounded inputs the GPU saw, so the surviving difference is
//! only the final cast of the (f32-accumulated) result back to half — proving the
//! reduction accumulates in f32, per ONNX semantics. The reduction failing on
//! CUDA would surface here as a panic (the kernel is unwrapped), making this a
//! regression guard for `CUDNN_STATUS_NOT_SUPPORTED`.
//!
//! The suite skips cleanly when no CUDA runtime is present.

mod common;

use common::{assert_close, decode_floats, float_input, input, require_cuda, run_cpu, run_cuda};
use onnx_runtime_ir::{Attribute, DataType};

/// Output shape for a keepdims-honouring reduction over `axes` (negative axes
/// allowed) of an input with shape `in_shape`.
fn reduce_out_shape(in_shape: &[usize], axes: &[i64], keepdims: bool) -> Vec<usize> {
    let rank = in_shape.len() as i64;
    let reduced: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (a + rank) as usize
            } else {
                a as usize
            }
        })
        .collect();
    let mut out = Vec::new();
    for (axis, &dim) in in_shape.iter().enumerate() {
        if reduced.contains(&axis) {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(dim);
        }
    }
    out
}

/// A deterministic, non-trivial value distribution: distinct signed magnitudes
/// so the reduction actually accumulates (not an all-equal degenerate case).
fn varied_values(count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (i as f32 * 0.7).sin() * 2.5 + ((i % 5) as f32 - 2.0) * 0.6)
        .collect()
}

/// Tolerance for comparing a half/bf16 CUDA reduction against an f32 oracle.
///
/// Both paths accumulate in f32; the residual is the single cast of the result
/// back to half, i.e. ~0.5 ulp of the output's magnitude. The relative terms
/// (f16 ≈ 2^-10, bf16 ≈ 2^-8) are set a few multiples above that bound.
fn half_reduce_tolerance(dtype: DataType, expected: &[f32]) -> f32 {
    let max_abs = expected.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    let (rel, abs) = match dtype {
        DataType::Float16 => (2e-3, 1e-2),
        DataType::BFloat16 => (1e-2, 5e-2),
        other => panic!("unsupported half dtype {other:?}"),
    };
    abs + rel * max_abs
}

/// Run `op` (`ReduceSum`/`ReduceMean`) in `dtype` on CUDA and assert it matches
/// an f32 CPU oracle fed the same half-rounded inputs.
fn assert_half_reduce_matches_f32_oracle(
    ep: &onnx_runtime_ep_cuda::CudaExecutionProvider,
    op: &str,
    dtype: DataType,
    in_shape: &[usize],
    axes: &[i64],
    keepdims: bool,
) {
    let count: usize = in_shape.iter().product();
    let raw = varied_values(count);
    let half_input = float_input(dtype, in_shape, &raw);
    // The exact values the GPU reduces, after rounding into the half dtype.
    let seen = decode_floats(&half_input.bytes, dtype);

    let out_shape = reduce_out_shape(in_shape, axes, keepdims);
    let axes_input = input(DataType::Int64, &[axes.len()], axes);
    let keepdims_attr = [("keepdims", Attribute::Int(keepdims as i64))];

    // CUDA in half: exercises the f16 f32-compute-type cuDNN path or, for bf16,
    // the typed NVRTC block reduction (both accumulate in f32).
    let cuda_inputs = [half_input, axes_input.clone()];
    let cuda_outputs = [(dtype, out_shape.clone())];
    let cuda = run_cuda(ep, op, "", 18, &cuda_inputs, &cuda_outputs, &keepdims_attr);
    let got = decode_floats(&cuda[0], dtype);

    // f32 CPU oracle over the identical (half-rounded) inputs.
    let oracle_inputs = [input(DataType::Float32, in_shape, &seen), axes_input];
    let oracle_outputs = [(DataType::Float32, out_shape)];
    let cpu = run_cpu(op, "", 18, &oracle_inputs, &oracle_outputs, &keepdims_attr);
    let expected = decode_floats(&cpu[0], DataType::Float32);

    let tolerance = half_reduce_tolerance(dtype, &expected);
    assert_close(
        &format!("{op} {dtype:?} axes={axes:?} keepdims={keepdims}"),
        dtype,
        &got,
        &expected,
        tolerance,
    );
    eprintln!("{op} {dtype:?} axes={axes:?} keepdims={keepdims}: CUDA matches f32 oracle");
}

/// The exact blocker node's shape: `[1,5,8] -> [1,5,1]`, reduce last axis,
/// keepdims, for both `ReduceSum` and `ReduceMean` in fp16 and bf16.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn half_reduce_router_shape_matches_f32_oracle() {
    let ep = require_cuda();
    for dtype in [DataType::Float16, DataType::BFloat16] {
        for op in ["ReduceSum", "ReduceMean"] {
            assert_half_reduce_matches_f32_oracle(&ep, op, dtype, &[1, 5, 8], &[2], true);
        }
    }
}

/// A non-trivial multi-axis, non-keepdims reduction (`[2,3,4]` over axes `[0,2]`
/// → `[3]`) in fp16 and bf16, to cover rank/stride handling beyond the router
/// node's trailing-axis case.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn half_reduce_multi_axis_no_keepdims_matches_f32_oracle() {
    let ep = require_cuda();
    for dtype in [DataType::Float16, DataType::BFloat16] {
        for op in ["ReduceSum", "ReduceMean"] {
            assert_half_reduce_matches_f32_oracle(&ep, op, dtype, &[2, 3, 4], &[0, 2], false);
        }
    }
}

/// Guard the fix's numeric intent directly: an fp16 `ReduceSum` whose true sum
/// (5.0) is representable, but whose partial sums overflow fp16's max (65504) —
/// `[60000, 60000, -60000, -60000, 5]`. An fp16-compute-type accumulator would
/// saturate to `+inf`/`-inf` mid-reduction and lose the result; an f32
/// accumulator (the fix) returns 5.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn fp16_reduce_sum_accumulates_in_f32_without_overflow() {
    let ep = require_cuda();
    let in_shape = [1usize, 5];
    let values = [60000.0_f32, 60000.0, -60000.0, -60000.0, 5.0];
    let half_input = float_input(DataType::Float16, &in_shape, &values);
    let axes = [1_i64];
    let inputs = [half_input, input(DataType::Int64, &[1], &axes)];
    let outputs = [(DataType::Float16, vec![1usize, 1])];
    let attrs = [("keepdims", Attribute::Int(1))];
    let cuda = run_cuda(&ep, "ReduceSum", "", 18, &inputs, &outputs, &attrs);
    let got = decode_floats(&cuda[0], DataType::Float16);
    assert_eq!(got.len(), 1);
    assert!(
        got[0].is_finite(),
        "fp16 ReduceSum overflowed (half-compute-type accumulator?): {got:?}"
    );
    // The result (5.0) is exact in fp16; only f32 accumulation preserves it.
    assert!(
        (got[0] - 5.0).abs() <= 0.5,
        "fp16 ReduceSum should accumulate in f32 to ~5.0, got {}",
        got[0]
    );
    eprintln!(
        "fp16 ReduceSum accumulates in f32: partial-sum overflow avoided, got {}",
        got[0]
    );
}
