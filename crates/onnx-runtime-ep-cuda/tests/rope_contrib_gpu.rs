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
//! GPU parity for `com.microsoft::RotaryEmbedding` (issue #67): the CUDA EP
//! contrib RoPE (input order `X, position_ids, cos_cache, sin_cache`) checked
//! tol-exact against the CPU EP oracle across fp32/fp16/bf16 on a non-trivial
//! 4D decode shape. The rotation math is identical to the standard `ai.onnx`
//! op; only the input ordering differs, so this guards the contrib wiring.

mod common;

use common::{assert_close, decode_floats, float_input, input, require_cuda, run_cpu, run_cuda};
use onnx_runtime_ir::{Attribute, DataType};

const OP: &str = "RotaryEmbedding";
const DOMAIN: &str = "com.microsoft";
const OPSET: u64 = 1;

fn tolerance(dtype: DataType) -> f32 {
    match dtype {
        DataType::Float32 => 1e-4,
        DataType::Float16 => 3e-3,
        DataType::BFloat16 => 3e-2,
        _ => 0.0,
    }
}

fn check(ep: &onnx_runtime_ep_cuda::CudaExecutionProvider, dtype: DataType, interleaved: i64) {
    // X: [batch=1, heads=2, seq=2, head_size=8]; full rotation (rotary_dim=8).
    let batch = 1usize;
    let heads = 2usize;
    let seq = 2usize;
    let head_size = 8usize;
    let half = head_size / 2;
    let cache_rows = 8usize;

    let x_values: Vec<f32> = (0..batch * heads * seq * head_size)
        .map(|i| (i as f32) * 0.05 - 1.3)
        .collect();
    let cos_values: Vec<f32> = (0..cache_rows * half)
        .map(|i| ((i as f32) * 0.11).cos())
        .collect();
    let sin_values: Vec<f32> = (0..cache_rows * half)
        .map(|i| ((i as f32) * 0.11).sin())
        .collect();
    let positions: Vec<i64> = vec![3, 5];

    let inputs = vec![
        float_input(dtype, &[batch, heads, seq, head_size], &x_values),
        input::<i64>(DataType::Int64, &[batch, seq], &positions),
        float_input(dtype, &[cache_rows, half], &cos_values),
        float_input(dtype, &[cache_rows, half], &sin_values),
    ];
    let outputs = vec![(dtype, vec![batch, heads, seq, head_size])];
    let attrs = vec![("interleaved", Attribute::Int(interleaved))];

    let cuda = run_cuda(ep, OP, DOMAIN, OPSET, &inputs, &outputs, &attrs);
    let cpu = run_cpu(OP, DOMAIN, OPSET, &inputs, &outputs, &attrs);
    assert_eq!(cuda.len(), cpu.len(), "output count mismatch");
    let got = decode_floats(&cuda[0], dtype);
    let want = decode_floats(&cpu[0], dtype);
    assert_close(
        &format!("RotaryEmbedding[contrib,{dtype:?},interleaved={interleaved}]"),
        dtype,
        &got,
        &want,
        tolerance(dtype),
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn contrib_rotary_matches_cpu() {
    let ep = require_cuda();
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        for interleaved in [0, 1] {
            check(&ep, dtype, interleaved);
        }
    }
}
