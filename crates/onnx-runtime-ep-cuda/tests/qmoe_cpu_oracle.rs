//! Host-only regression for the CPU oracle shared by CUDA parity suites.
mod common;

use onnx_runtime_ir::{Attribute, DataType};

#[test]
fn cpu_qmoe_oracle_prepares_declared_workspace_without_cuda() {
    const H: usize = 16;
    const E: usize = 4;
    for rows in [1, 6] {
        let packed = common::input(
            DataType::Uint8,
            &[E, H, H / 2],
            &vec![0x99u8; E * H * H / 2],
        );
        let scales = common::input(DataType::Float32, &[E, H, 1], &[1f32; E * H]);
        let inputs = [
            common::input(DataType::Float32, &[rows, H], &vec![1f32; rows * H]),
            common::input(
                DataType::Float32,
                &[rows, E],
                &[4f32, 3., 2., 1.].repeat(rows),
            ),
            packed.clone(),
            scales.clone(),
            common::absent_input(DataType::Float32),
            packed,
            scales,
        ];
        let output = common::run_cpu(
            "QMoE",
            "com.microsoft",
            1,
            &inputs,
            &[(DataType::Float32, vec![rows, H])],
            &[
                ("expert_weight_bits", Attribute::Int(4)),
                ("block_size", Attribute::Int(16)),
                ("k", Attribute::Int(2)),
                ("activation_type", Attribute::String(b"identity".to_vec())),
                ("normalize_routing_weights", Attribute::Int(1)),
            ],
        );
        assert_eq!(output[0].len(), rows * H * 4);
        // Every expert has unit weights: FC1=H, FC2=H*H. Normalized
        // routing combines identical experts, so the f64 oracle is exact.
        let expected = (H * H) as f64;
        for bytes in output[0].chunks_exact(4) {
            let actual = f32::from_ne_bytes(bytes.try_into().unwrap());
            assert!(
                actual.is_finite()
                    && (actual as f64 - expected).abs() <= 1e-4 + 1e-5 * expected.abs()
            );
        }
    }
}
