//! Observe the two removed payload allocations at the actual EP launch seam.
//! Size-specific checks do not assert that unrelated kernel metadata is allocation-free.
#[path = "../benches/common/mod.rs"]
mod common;

use common::{FloatDType, PreparedKernel, Tensor};
use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{Attribute, Node, NodeId};
#[path = "support/allocation_observer.rs"]
mod allocation_observer;
use allocation_observer::watched;
use std::hint::black_box;

const M: usize = 3;
const K: usize = 35;
const N: usize = 67;
const OUTPUT_BYTES: usize = M * N * 4;
const BIAS_BYTES: usize = N * 4;

#[test]
fn packed_matmul_reuses_output_staging_and_borrows_contiguous_f32_bias() {
    // A positive control must detect the exact payload sizes removed below.
    let (_, control) = watched([OUTPUT_BYTES, BIAS_BYTES], || {
        black_box(vec![0.0f32; black_box(M * N)]);
        black_box(vec![0.0f32; black_box(N)]);
    });
    assert_eq!(control, [1, 1]);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let cpu = CpuExecutionProvider::new();
        let mut packed = vec![1u8; N * 2 * 34];
        for block in packed.chunks_exact_mut(34) {
            block[..2].copy_from_slice(&half::f16::ONE.to_le_bytes());
        }
        let w = Tensor::u8(&[N, 2, 34], &packed);
        for dtype in [FloatDType::F32, FloatDType::F16, FloatDType::Bf16] {
            let x = Tensor::floats(dtype, &[M, K], &[127.; M * K]);
            let bias = Tensor::floats(dtype, &[N], &[1.; N]);
            for (arithmetic, streamed) in [("float32", true), ("float32", false), ("q8_0", true)] {
                let mut node = Node::new(NodeId(0), "BlockQuantizedMatMul", vec![], vec![]);
                node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
                for (name, value) in [
                    ("K", Attribute::Int(K as i64)),
                    ("N", Attribute::Int(N as i64)),
                    ("format", Attribute::String(b"q8_0".to_vec())),
                    ("activation_arithmetic", Attribute::String(arithmetic.as_bytes().to_vec())),
                    ("block_layout_version", Attribute::Int(if arithmetic == "q8_0" { 2 } else { 1 })),
                    (onnx_runtime_optimizer::quantized_matmul::STREAM_DECODED_ROWS, Attribute::Int(i64::from(streamed))),
                ] {
                    node.attributes.insert(name.into(), value);
                }
                let mut kernel = cpu.get_kernel(&node, &[], 1).unwrap();
                kernel.set_constant_inputs(&[false, true, false]);
                let inputs = [x.view(), w.view(), bias.view()];
                let mut y = Tensor::zeros(dtype, &[M, N]);
                let mut prepared = PreparedKernel::new(&cpu, kernel, &inputs, &[y.view_mut()]);
                // Warm up weight admission, code dispatch and the selected pool.
                prepared.execute(&inputs, &mut [y.view_mut()]).unwrap();
                let expected = match dtype {
                    FloatDType::F32 => 4446.,
                    FloatDType::F16 => half::f16::from_f32(4446.).to_f32(),
                    FloatDType::Bf16 => half::bf16::from_f32(4446.).to_f32(),
                };
                assert!(y.to_f32().iter().all(|value| *value == expected));
                for _ in 0..4 {
                    let (result, counts) = watched([OUTPUT_BYTES, BIAS_BYTES], || prepared.execute(&inputs, &mut [y.view_mut()]));
                    result.unwrap();
                    assert_eq!(counts[0], 0, "{dtype:?}/{arithmetic}/stream={streamed}: output payload allocated");
                    assert_eq!(counts[1], usize::from(!matches!(dtype, FloatDType::F32)),
                        "{dtype:?}/{arithmetic}/stream={streamed}: only half bias widening should allocate");
                    assert!(y.to_f32().iter().all(|value| *value == expected));
                }
            }
        }
    });
}
