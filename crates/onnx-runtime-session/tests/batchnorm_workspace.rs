use onnx_runtime_ir::{Attribute, DataType, Dim, Graph, Node, NodeId};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use onnx_runtime_session::{InferenceSession, Tensor};
use std::sync::Arc;

fn model(opset: u64, dtype: DataType) -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let shape = (0..3)
        .map(|i| Dim::Symbolic(graph.create_symbol(Some(format!("d{i}")))))
        .collect::<Vec<_>>();
    let x = graph.create_named_value("X", dtype, shape.clone());
    graph.add_input(x);
    let parameters = ["scale", "B", "mean", "variance"].map(|name| {
        let v = graph.create_named_value(name, dtype, vec![shape[1]]);
        graph.add_input(v);
        v
    });
    let first = graph.create_named_value("first", dtype, shape.clone());
    let y = graph.create_named_value("Y", dtype, shape);
    graph.add_output(y);
    for (id, x, y) in [(0, x, first), (1, first, y)] {
        let mut inputs = vec![Some(x)];
        inputs.extend(parameters.map(Some));
        let mut node = Node::new(NodeId(id), "BatchNormalization", inputs, vec![y]);
        node.attributes
            .insert("epsilon".into(), Attribute::Float(0.125));
        graph.insert_node(node);
    }
    onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&graph)).unwrap()
}
fn encode(dtype: DataType, values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| match dtype {
            DataType::Float32 => (v as f32).to_ne_bytes().to_vec(),
            DataType::Float64 => v.to_ne_bytes().to_vec(),
            DataType::Float16 => half::f16::from_f64(v).to_bits().to_ne_bytes().to_vec(),
            DataType::BFloat16 => half::bf16::from_f64(v).to_bits().to_ne_bytes().to_vec(),
            _ => unreachable!(),
        })
        .collect()
}
fn decode(dtype: DataType, bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(dtype.byte_size())
        .map(|b| match dtype {
            DataType::Float32 => f64::from(f32::from_ne_bytes(b.try_into().unwrap())),
            DataType::Float64 => f64::from_ne_bytes(b.try_into().unwrap()),
            DataType::Float16 => {
                half::f16::from_bits(u16::from_ne_bytes(b.try_into().unwrap())).to_f64()
            }
            DataType::BFloat16 => {
                half::bf16::from_bits(u16::from_ne_bytes(b.try_into().unwrap())).to_f64()
            }
            _ => unreachable!(),
        })
        .collect()
}
fn reference(
    dtype: DataType,
    x: &[f64],
    parameters: &[Vec<f64>; 4],
    shape: [usize; 3],
) -> Vec<f64> {
    let x = decode(dtype, &encode(dtype, x));
    let p = parameters
        .each_ref()
        .map(|p| decode(dtype, &encode(dtype, p)));
    let result = x
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let c = (i / shape[2]) % shape[1];
            if dtype == DataType::Float64 {
                ((v - p[2][c]) / (p[3][c] + 0.125).sqrt()) * p[0][c] + p[1][c]
            } else {
                f64::from(
                    ((v as f32 - p[2][c] as f32) / (p[3][c] as f32 + 0.125).sqrt())
                        * p[0][c] as f32
                        + p[1][c] as f32,
                )
            }
        })
        .collect::<Vec<_>>();
    decode(dtype, &encode(dtype, &result))
}

#[test]
fn two_nodes_share_one_storage_peak_across_dynamic_channels_empty_batch_and_trim() {
    for opset in [7, 15] {
        for dtype in [
            DataType::Float16,
            DataType::BFloat16,
            DataType::Float32,
            DataType::Float64,
        ] {
            // Legacy schema excludes bf16; keep encoded fixtures schema-valid.
            // https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
            if opset < 14 && dtype == DataType::BFloat16 {
                continue;
            }
            let peak = (24 * dtype.byte_size()) as u64;
            for budget in [peak - 1, peak] {
                let books = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
                    DeviceKey::HOST,
                    0,
                    budget,
                    0,
                )));
                let manager = ProcessMemoryManager::new().unwrap();
                let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
                let resources = onnx_runtime_ep_cpu::CpuMemoryResources::new(
                    onnx_runtime_ep_cpu::DerivedWeightMemory::governed(
                        books.as_ref(),
                        HolderId::new(13),
                    )
                    .unwrap(),
                    Default::default(),
                )
                .unwrap()
                .with_workspace_pool(pool.clone())
                .unwrap();
                let cpu = onnx_runtime_ep_cpu::CpuExecutionProvider::initialized_with_resources(
                    resources,
                )
                .unwrap();
                let mut session = InferenceSession::builder()
                    .model_bytes(&model(opset, dtype))
                    .execution_provider(Arc::new(cpu))
                    .build()
                    .unwrap();
                let mut identity = None;
                for (iteration, shape) in [[2, 3, 4], [1, 2, 5], [0, 3, 4], [2, 3, 4]]
                    .into_iter()
                    .enumerate()
                {
                    let values = (0..shape.iter().product())
                        .map(|i| {
                            if dtype == DataType::Float64 {
                                16_777_217. + i as f64 * 0.25
                            } else {
                                (i as f64 - 9.) * 0.25
                            }
                        })
                        .collect::<Vec<_>>();
                    let parameters: [Vec<f64>; 4] = std::array::from_fn(|role| {
                        (0..shape[1])
                            .map(|c| match role {
                                0 => 1. + c as f64 * 0.25,
                                1 => -0.25 * c as f64,
                                2 => 0.5 * c as f64,
                                _ => 1. + 0.25 * c as f64,
                            })
                            .collect()
                    });
                    let x =
                        Tensor::from_raw(dtype, shape.to_vec(), &encode(dtype, &values)).unwrap();
                    let p = parameters.each_ref().map(|v| {
                        Tensor::from_raw(dtype, vec![shape[1]], &encode(dtype, v)).unwrap()
                    });
                    let inputs = [
                        ("X", &x),
                        ("scale", &p[0]),
                        ("B", &p[1]),
                        ("mean", &p[2]),
                        ("variance", &p[3]),
                    ];
                    if budget < peak {
                        assert!(
                            session
                                .prepare_with_device_bindings(&inputs, &mut [])
                                .is_err()
                        );
                        assert!(session.run(&inputs).is_err());
                        assert!(manager.snapshot().unwrap().allocations.is_empty());
                        assert_eq!(books.used(Tier::Host), 0);
                        break;
                    }
                    let report = session
                        .prepare_with_device_bindings(&inputs, &mut [])
                        .unwrap();
                    assert_eq!(
                        report.require_complete().unwrap().step().bytes,
                        (values.len() * dtype.byte_size()) as u64
                    );
                    let first = reference(dtype, &values, &parameters, shape);
                    let expected = encode(dtype, &reference(dtype, &first, &parameters, shape));
                    for _ in 0..2 {
                        let outputs = session.run(&inputs).unwrap();
                        assert_eq!(
                            outputs[0].as_bytes(),
                            expected,
                            "opset={opset} dtype={dtype:?} shape={shape:?}"
                        );
                        let snapshot = manager.snapshot().unwrap();
                        if values.is_empty() {
                            assert!(snapshot.allocations.is_empty());
                            assert_eq!(books.used(Tier::Host), 0);
                        } else {
                            assert_eq!(snapshot.allocations.len(), 1);
                            assert_eq!(
                                snapshot.allocations[0].identity,
                                *identity.get_or_insert(snapshot.allocations[0].identity)
                            );
                            assert_eq!(books.used(Tier::Host), peak);
                        }
                    }
                    if iteration == 1 {
                        pool.trim_idle(u64::MAX).unwrap();
                        assert_eq!(books.used(Tier::Host), 0);
                        identity = None;
                    }
                }
                drop(session);
                pool.trim_idle(u64::MAX).unwrap();
                assert_eq!(books.used(Tier::Host), 0);
                assert_eq!(manager.process_used(Tier::Host), 0);
            }
        }
    }
}
