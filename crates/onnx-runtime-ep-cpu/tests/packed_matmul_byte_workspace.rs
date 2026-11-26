//! Packed payload allocation witness for cold and warm standalone projections.
#[path = "support/allocation_observer.rs"]
mod allocation_observer;
#[path = "../benches/common/mod.rs"]
mod common;
use allocation_observer::watched;
use common::{FloatDType, PreparedKernel, Tensor};
use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ep_cpu::{CpuExecutionProvider, CpuMemoryResources, CpuWeightCacheLimits};
use onnx_runtime_ir::{Attribute, Node, NodeId, ValueId};
use std::hint::black_box;

#[test]
fn strided_packed_weights_use_one_admitted_lane_in_every_projection_path() {
    const K: usize = 96;
    const N: usize = 19;
    const PACKED: usize = N * (K / 32) * 34;
    let (_, control) = watched([PACKED, PACKED * 2], || {
        black_box(vec![0u8; black_box(PACKED)]);
        black_box(vec![0u8; black_box(PACKED * 2)]);
    });
    assert_eq!(control, [1, 1]);
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            let x = Tensor::floats(FloatDType::F32, &[3, K], &[127.; 3 * K]);
            let mut logical = vec![1; PACKED];
            for block in logical.chunks_exact_mut(34) {
                block[..2].copy_from_slice(&half::f16::ONE.to_le_bytes());
            }
            let physical = logical.iter().flat_map(|&v| [v, 0xa5]).collect::<Vec<_>>();
            for strided in [false, true] {
                let weights = Tensor::u8(
                    &[if strided { PACKED * 2 } else { PACKED }],
                    if strided { &physical } else { &logical },
                );
                let mut w = weights.view();
                w.shape = &[N, K / 32, 34];
                w.strides = if strided {
                    &[204, 68, 2]
                } else {
                    &[102, 34, 1]
                };
                let inputs = [x.view(), w];
                for (arithmetic, dense) in [("float32", false), ("float32", true), ("q8_0", false)]
                {
                    let mut node = Node::new(
                        NodeId(0),
                        "BlockQuantizedMatMul",
                        vec![Some(ValueId(0)), Some(ValueId(1))],
                        vec![ValueId(2)],
                    );
                    node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
                    for (name, value) in [
                        ("K", Attribute::Int(K as i64)),
                        ("N", Attribute::Int(N as i64)),
                        ("format", Attribute::String(b"q8_0".to_vec())),
                        (
                            "activation_arithmetic",
                            Attribute::String(arithmetic.as_bytes().to_vec()),
                        ),
                        (
                            "block_layout_version",
                            Attribute::Int(if arithmetic == "q8_0" { 2 } else { 1 }),
                        ),
                    ] {
                        node.attributes.insert(name.into(), value);
                    }
                    let cpu = CpuExecutionProvider::with_resources(
                        CpuMemoryResources::new(
                            Default::default(),
                            CpuWeightCacheLimits {
                                dense_bytes: if dense { 1 << 20 } else { 0 },
                                expert_bytes: 0,
                            },
                        )
                        .unwrap(),
                    );
                    let mut kernel = cpu.get_kernel(&node, &[], 1).unwrap();
                    kernel.set_constant_inputs(&[false, true]);
                    let mut y = Tensor::zeros(FloatDType::F32, &[3, N]);
                    let mut prepared = PreparedKernel::new(&cpu, kernel, &inputs, &[y.view_mut()]);
                    // Includes cold hashing/building, a resident hit, and repeated streaming/Q8.
                    for _ in 0..3 {
                        let (result, counts) = watched([PACKED, PACKED * 2], || {
                            prepared.execute(&inputs, &mut [y.view_mut()])
                        });
                        result.unwrap();
                        assert_eq!(
                            counts,
                            [0, 0],
                            "{arithmetic} dense={dense} strided={strided}"
                        );
                        assert_eq!(y.f32s(), &[127. * K as f32; 3 * N]);
                    }
                }
            }
        });
}
