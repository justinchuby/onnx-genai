//! Size-specific routing allocation witness. Input/weight copies and grant
//! metadata are outside this claim; they are not a whole-kernel malloc count.
#[path = "support/allocation_observer.rs"]
mod allocation_observer;
#[path = "../benches/common/mod.rs"]
mod common;
use allocation_observer::watched;
use common::{FloatDType, PreparedKernel, Tensor};
use onnx_runtime_ep_api::{ExecutionProvider, TensorView};
use onnx_runtime_ep_cpu::{CpuExecutionProvider, CpuMemoryResources, CpuWeightCacheLimits};
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, ValueId};
use std::hint::black_box;
#[test]
fn qmoe_topk_and_softmax_storage_reuse_the_admitted_arena() {
    assert!(
        std::env::var_os("ONNX_GENAI_QMOE_ROUTE_DUMP").is_none(),
        "routing allocation witness requires normal execution; unset ONNX_GENAI_QMOE_ROUTE_DUMP to exclude explicit diagnostic Vecs"
    );
    const R: usize = 7;
    const H: usize = 16;
    const E: usize = 257;
    let sizes = [
        [E * size_of::<usize>(), E * 4],
        [E * size_of::<(usize, f32)>(), E * size_of::<usize>()],
    ];
    for pair in sizes {
        let (_, control) = watched(pair, || {
            black_box(vec![0u8; black_box(pair[0])]);
            black_box(vec![0u8; black_box(pair[1])]);
        });
        assert_eq!(control, [1, 1]);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            let x = Tensor::floats(FloatDType::F32, &[R, H], &[1.; R * H]);
            let router = Tensor::floats(FloatDType::F32, &[R, E], &[0.; R * E]);
            let mut bytes = vec![128u8; E * H * H];
            for expert in 0..E {
                for feature in 0..H {
                    bytes[(expert * H + feature) * H + feature] = 129;
                }
            }
            let w1 = Tensor::u8(&[E, H, H], &bytes);
            let w2 = Tensor::u8(&[E, H, H], &bytes);
            let s1 = Tensor::floats(FloatDType::F32, &[E, H, 1], &[1.; E * H]);
            let s2 = Tensor::floats(FloatDType::F32, &[E, H, 1], &[1.; E * H]);
            let inputs = [
                x.view(),
                router.view(),
                w1.view(),
                s1.view(),
                TensorView::absent(DataType::Float32),
                w2.view(),
                s2.view(),
            ];
            for k in [3, E] {
                let mut node = Node::new(
                    NodeId(0),
                    "QMoE",
                    vec![
                        Some(ValueId(0)),
                        Some(ValueId(1)),
                        Some(ValueId(2)),
                        Some(ValueId(3)),
                        None,
                        Some(ValueId(5)),
                        Some(ValueId(6)),
                    ],
                    vec![ValueId(7)],
                );
                node.domain = "com.microsoft".into();
                for (name, value) in [
                    ("expert_weight_bits", Attribute::Int(8)),
                    ("block_size", Attribute::Int(16)),
                    ("k", Attribute::Int(k as i64)),
                    ("normalize_routing_weights", Attribute::Int(1)),
                    ("activation_type", Attribute::String(b"identity".to_vec())),
                ] {
                    node.attributes.insert(name.into(), value);
                }
                let cpu = CpuExecutionProvider::with_resources(
                    CpuMemoryResources::new(
                        Default::default(),
                        CpuWeightCacheLimits {
                            dense_bytes: 0,
                            expert_bytes: 0,
                        },
                    )
                    .unwrap(),
                );
                let kernel = cpu.get_kernel(&node, &[], 1).unwrap();
                let mut y = Tensor::zeros(FloatDType::F32, &[R, H]);
                let mut prepared = PreparedKernel::new(&cpu, kernel, &inputs, &[y.view_mut()]);
                prepared.execute(&inputs, &mut [y.view_mut()]).unwrap();
                let mut expected = 0f32;
                for _ in 0..k {
                    expected += 1f32 / k as f32;
                }
                for pair in sizes {
                    for _ in 0..3 {
                        let (result, counts) =
                            watched(pair, || prepared.execute(&inputs, &mut [y.view_mut()]));
                        result.unwrap();
                        assert_eq!(counts, [0, 0], "k={k} sizes={pair:?}");
                        assert_eq!(y.f32s(), &[expected; R * H]);
                    }
                }
            }
        });
}
