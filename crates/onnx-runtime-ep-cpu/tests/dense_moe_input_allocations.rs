//! Witness only floating source payload copies, not all MoE allocations.
#[path = "support/allocation_observer.rs"]
mod allocation_observer;
#[path = "../benches/common/mod.rs"]
mod common;
use allocation_observer::watched;
use common::{FloatDType, PreparedKernel, Tensor};
use onnx_runtime_ep_api::{DevicePtr, ExecutionProvider};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, ValueId, compute_contiguous_strides};
use std::hint::black_box;

#[test]
fn dense_moe_floating_payloads_use_the_admitted_snapshots() {
    const R: usize = 3;
    const H: usize = 13;
    const I: usize = 17;
    const E: usize = 7;
    let pairs = [
        [R * H * 4, R * E * 4],
        [E * I * H * 4, E * I * 4],
        [E * H * 4, R * H * 4],
    ];
    for pair in pairs {
        let (_, counts) = watched(pair, || {
            black_box(vec![0u8; black_box(pair[0])]);
            black_box(vec![0u8; black_box(pair[1])]);
        });
        assert_eq!(counts, [1, 1]);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            let shapes = [
                vec![R, H],
                vec![R, E],
                vec![E, I, H],
                vec![E, I],
                vec![E, H, I],
                vec![E, H],
                vec![E, I, H],
                vec![E, I],
            ];
            let strides = shapes
                .iter()
                .map(|s| compute_contiguous_strides(s).unwrap())
                .collect::<Vec<_>>();
            for (dtype, ir_dtype) in [
                (FloatDType::F32, DataType::Float32),
                (FloatDType::F16, DataType::Float16),
                (FloatDType::Bf16, DataType::BFloat16),
            ] {
                for layout in ["contiguous", "strided", "unaligned"] {
                    let owners = shapes
                        .iter()
                        .enumerate()
                        .map(|(slot, shape)| {
                            let len = shape.iter().product::<usize>();
                            let value: f32 = if slot == 5 { 2. } else { 0. };
                            if layout == "unaligned" {
                                let mut bytes = vec![0x7f];
                                for _ in 0..len {
                                    match ir_dtype {
                                        DataType::Float32 => {
                                            bytes.extend_from_slice(&value.to_le_bytes())
                                        }
                                        DataType::Float16 => bytes.extend_from_slice(
                                            &half::f16::from_f32(value).to_le_bytes(),
                                        ),
                                        _ => bytes.extend_from_slice(
                                            &half::bf16::from_f32(value).to_le_bytes(),
                                        ),
                                    }
                                }
                                Tensor::u8(&[bytes.len()], &bytes)
                            } else if layout == "strided" {
                                let values =
                                    (0..len).flat_map(|_| [value, f32::NAN]).collect::<Vec<_>>();
                                Tensor::floats(dtype, &[len * 2], &values)
                            } else {
                                Tensor::floats(dtype, shape, &vec![value; len])
                            }
                        })
                        .collect::<Vec<_>>();
                    let physical_strides = strides
                        .iter()
                        .map(|s| {
                            s.iter()
                                .map(|v| v * if layout == "strided" { 2 } else { 1 })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let inputs = owners
                        .iter()
                        .enumerate()
                        .map(|(slot, owner)| {
                            let mut view = owner.view();
                            view.shape = &shapes[slot];
                            view.strides = &physical_strides[slot];
                            view.dtype = ir_dtype;
                            if layout == "unaligned" {
                                view.data = DevicePtr(view.data.0.wrapping_add(1));
                            }
                            view
                        })
                        .collect::<Vec<_>>();
                    let mut node = Node::new(
                        NodeId(0),
                        "MoE",
                        (0..8).map(|i| Some(ValueId(i))).collect(),
                        vec![ValueId(8)],
                    );
                    node.domain = "com.microsoft".into();
                    node.attributes.insert("k".into(), Attribute::Int(2));
                    node.attributes.insert(
                        "activation_type".into(),
                        Attribute::String(b"silu".to_vec()),
                    );
                    node.attributes
                        .insert("normalize_routing_weights".into(), Attribute::Int(1));
                    let cpu = CpuExecutionProvider::new();
                    let kernel = cpu.get_kernel(&node, &[], 1).unwrap();
                    let mut y = Tensor::zeros(dtype, &[R, H]);
                    let mut prepared = PreparedKernel::new(&cpu, kernel, &inputs, &[y.view_mut()]);
                    prepared.execute(&inputs, &mut [y.view_mut()]).unwrap();
                    for pair in pairs {
                        for _ in 0..3 {
                            let (result, counts) =
                                watched(pair, || prepared.execute(&inputs, &mut [y.view_mut()]));
                            result.unwrap();
                            assert_eq!(counts, [0, 0], "{ir_dtype:?}/{layout}/{pair:?}");
                        }
                    }
                    // Read the result through dtype-independent conversion outside observation.
                    let values = match ir_dtype {
                        DataType::Float32 => y.f32s().to_vec(),
                        _ => y.to_f32(),
                    };
                    assert!(values.iter().all(|v| *v == 2.));
                }
            }
        });
}
