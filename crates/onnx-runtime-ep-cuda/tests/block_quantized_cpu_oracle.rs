//! CPU-only regressions for the prepared oracle used by block-format GPU suites.
mod common;

use onnx_runtime_ir::{Attribute, DataType};

fn ones(blocks: usize) -> Vec<u8> {
    (0..blocks)
        .flat_map(|_| std::iter::once(127).chain([0x22; 16]))
        .collect()
}

#[test]
fn cpu_block_matmul_oracle_refines_before_launch_without_cuda() {
    for rows in [1, 3] {
        let output = common::run_cpu(
            "BlockQuantizedMatMul",
            "pkg.nxrt",
            1,
            &[
                common::input(DataType::Float32, &[rows, 32], &vec![1f32; rows * 32]),
                common::input(DataType::Uint8, &[64, 1, 17], &ones(64)),
            ],
            &[(DataType::Float32, vec![rows, 64])],
            &[
                ("K", Attribute::Int(32)),
                ("N", Attribute::Int(64)),
                ("format", Attribute::String(b"mxfp4".to_vec())),
                ("block_layout_version", Attribute::Int(1)),
            ],
        );
        assert_eq!(output[0].len(), rows * 64 * 4);
        for value in output[0].chunks_exact(4) {
            assert_eq!(f32::from_ne_bytes(value.try_into().unwrap()), 32.);
        }
    }
}

#[test]
fn cpu_block_moe_oracle_refines_before_launch_without_cuda() {
    for rows in [1, 3] {
        let output = common::run_cpu(
            "BlockQuantizedMoE",
            "pkg.nxrt",
            1,
            &[
                common::input(DataType::Float32, &[rows, 32], &vec![1f32; rows * 32]),
                common::input(
                    DataType::Float32,
                    &[rows, 4],
                    &[4f32, 3., -4., -5.].repeat(rows),
                ),
                common::input(DataType::Uint8, &[4, 64, 1, 17], &ones(4 * 64)),
                common::absent_input(DataType::Float32),
                common::input(DataType::Uint8, &[4, 32, 2, 17], &ones(4 * 64)),
            ],
            &[(DataType::Float32, vec![rows, 32])],
            &[
                ("fc1_format", Attribute::String(b"mxfp4".to_vec())),
                ("fc2_format", Attribute::String(b"mxfp4".to_vec())),
                ("block_layout_version", Attribute::Int(1)),
                ("k", Attribute::Int(2)),
                ("normalize_routing_weights", Attribute::Int(1)),
                ("activation_type", Attribute::String(b"identity".to_vec())),
            ],
        );
        assert_eq!(output[0].len(), rows * 32 * 4);
        for value in output[0].chunks_exact(4) {
            assert_eq!(f32::from_ne_bytes(value.try_into().unwrap()), 2048.);
        }
    }
}
