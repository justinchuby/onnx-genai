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
//! GPU parity for `com.microsoft::GatherBlockQuantized` (issue #67): the CUDA EP
//! blockwise-quantized embedding lookup (already registered; this suite plus the
//! `CUDA_COVERED_OPS` entry close the coverage-of-coverage gap). Checked against
//! the CPU EP oracle for `bits ∈ {8, 4}`, fp32/fp16 scales+output, and
//! int32/int64 indices. `zero_points` is always supplied so both EPs take the
//! identical dequant branch (the layout the real `embedding.onnx` exports use).

mod common;

use common::{assert_close, decode_floats, float_input, input, require_cuda, run_cpu, run_cuda};
use onnx_runtime_ir::{Attribute, DataType};

const OP: &str = "GatherBlockQuantized";
const DOMAIN: &str = "com.microsoft";
const OPSET: u64 = 1;

#[allow(clippy::too_many_arguments)]
fn check(
    ep: &onnx_runtime_ep_cuda::CudaExecutionProvider,
    scales_dtype: DataType,
    index_dtype: DataType,
    bits: i64,
    data: &[u8],
    data_shape: &[usize],
    scales: &[f32],
    zero_points: &[u8],
    indices_i64: &[i64],
    out_shape: &[usize],
) {
    let scale_shape = [scales.len()];
    let zp_shape = [zero_points.len()];
    let idx_shape = [indices_i64.len()];
    let indices = match index_dtype {
        DataType::Int32 => input(
            DataType::Int32,
            &idx_shape,
            &indices_i64.iter().map(|&v| v as i32).collect::<Vec<_>>(),
        ),
        DataType::Int64 => input(DataType::Int64, &idx_shape, indices_i64),
        _ => unreachable!(),
    };
    let inputs = vec![
        input(DataType::Uint8, data_shape, data),
        indices,
        float_input(scales_dtype, &scale_shape, scales),
        input(DataType::Uint8, &zp_shape, zero_points),
    ];
    let outputs = vec![(scales_dtype, out_shape.to_vec())];
    let attrs = vec![
        ("gather_axis", Attribute::Int(0)),
        ("quantize_axis", Attribute::Int(1)),
        ("block_size", Attribute::Int(16)),
        ("bits", Attribute::Int(bits)),
    ];
    let cuda = run_cuda(ep, OP, DOMAIN, OPSET, &inputs, &outputs, &attrs);
    let cpu = run_cpu(OP, DOMAIN, OPSET, &inputs, &outputs, &attrs);
    let tol = if scales_dtype == DataType::Float16 {
        4e-3
    } else {
        2e-5
    };
    assert_close(
        OP,
        scales_dtype,
        &decode_floats(&cuda[0], scales_dtype),
        &decode_floats(&cpu[0], scales_dtype),
        tol,
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn gather_block_quantized_bits8_matches_cpu() {
    let ep = require_cuda();
    // 4 vocab rows × 16 hidden; block_size 16 → one block per row (4 blocks).
    let data: Vec<u8> = (0..64u16).map(|v| (v * 3 % 251) as u8).collect();
    let scales = [0.05f32, -0.1, 0.2, 0.03];
    let zero_points = [128u8, 100, 130, 8];
    let indices = [2i64, 0, 3];
    for scales_dtype in [DataType::Float32, DataType::Float16] {
        for index_dtype in [DataType::Int32, DataType::Int64] {
            check(
                &ep,
                scales_dtype,
                index_dtype,
                8,
                &data,
                &[4, 16],
                &scales,
                &zero_points,
                &indices,
                &[3, 16],
            );
        }
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn gather_block_quantized_bits4_matches_cpu() {
    let ep = require_cuda();
    // 4 rows × 32 logical (packed → 16 uint8 bytes per row), block_size 16 →
    // 2 blocks per row (even, so CPU per-row and CUDA global nibble-packing of
    // zero_points coincide — the layout real int4 embeddings export).
    let data: Vec<u8> = (0..64u16).map(|v| (v * 7 % 253) as u8).collect();
    // block_count = 4 rows × 2 = 8 scales.
    let scales = [0.05f32, -0.1, 0.2, 0.03, 0.08, -0.04, 0.12, -0.15];
    // 8 nibble zero-points packed 2/byte → 4 bytes.
    let zero_points = [
        0x8u8 | (0x7 << 4),
        0x2 | (0x9 << 4),
        0x5 | (0x1 << 4),
        0xA | (0x3 << 4),
    ];
    let indices = [1i64, 3, 0];
    for scales_dtype in [DataType::Float32, DataType::Float16] {
        for index_dtype in [DataType::Int32, DataType::Int64] {
            check(
                &ep,
                scales_dtype,
                index_dtype,
                4,
                &data,
                &[4, 16],
                &scales,
                &zero_points,
                &indices,
                &[3, 32],
            );
        }
    }
}

/// Signed native `Int4` blockwise gather (gpt-oss `model.embed_tokens.weight_Q4`
/// class): data is a real ONNX `Int4` tensor (`components == 1`, sign-extended
/// nibble) with NO `zero_points`, so the CUDA kernel must take the symmetric
/// `offset == 0` branch. The CPU EP is uint8-packed-only here and is NOT a valid
/// oracle for signed int4, so this checks against a hand-computed reference:
/// `out = signed_nibble * scale[row, col / block_size]`. Locks the signed-unpack
/// + symmetric-default path that enables gpt-oss's quantized embedding.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn gather_block_quantized_signed_int4_no_zero_points_matches_reference() {
    let ep = require_cuda();
    const ROWS: usize = 4;
    const COLS: usize = 32;
    const BLOCK: usize = 16;
    const BLOCKS_PER_ROW: usize = COLS / BLOCK; // 2

    // Deterministic signed nibbles in [-8, 7].
    let nib = |r: usize, c: usize| -> i32 { (((r * 7 + c * 3) % 16) as i32) - 8 };

    // Pack 2 nibbles/byte, low nibble first (matches the kernel's idx%2 order).
    let mut data = vec![0u8; ROWS * COLS / 2];
    for r in 0..ROWS {
        for c in 0..COLS {
            let idx = r * COLS + c;
            let code = (nib(r, c) & 0xF) as u8;
            if idx % 2 == 0 {
                data[idx / 2] |= code;
            } else {
                data[idx / 2] |= code << 4;
            }
        }
    }

    let scales: Vec<f32> = (0..ROWS * BLOCKS_PER_ROW)
        .map(|i| 0.05 + 0.01 * i as f32)
        .collect();
    let indices = [1i64, 3, 0];
    let out_shape = [indices.len(), COLS];

    let inputs = vec![
        input(DataType::Int4, &[ROWS, COLS], &data),
        input(DataType::Int64, &[indices.len()], &indices),
        float_input(DataType::Float32, &[scales.len()], &scales),
        // No zero_points => symmetric offset 0.
    ];
    let outputs = vec![(DataType::Float32, out_shape.to_vec())];
    let attrs = vec![
        ("gather_axis", Attribute::Int(0)),
        ("quantize_axis", Attribute::Int(1)),
        ("block_size", Attribute::Int(BLOCK as i64)),
        // No `bits` attribute => defaults to 4 (the gpt-oss embedding layout).
    ];

    let cuda = run_cuda(&ep, OP, DOMAIN, OPSET, &inputs, &outputs, &attrs);
    let got = decode_floats(&cuda[0], DataType::Float32);

    let mut expected = vec![0f32; indices.len() * COLS];
    for (o, &row_i64) in indices.iter().enumerate() {
        let row = row_i64 as usize;
        for c in 0..COLS {
            let scale = scales[row * BLOCKS_PER_ROW + c / BLOCK];
            expected[o * COLS + c] = nib(row, c) as f32 * scale;
        }
    }
    assert_close(OP, DataType::Float32, &got, &expected, 2e-5);
}
