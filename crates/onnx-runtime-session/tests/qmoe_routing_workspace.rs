//! QMoE routing shares the exact operation workspace; weight grants stay local
//! here so this test isolates workspace admission rather than a process RSS cap.
use onnx_runtime_ir::{Attribute, DataType, Dim, Graph, Node, NodeId, static_shape};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use onnx_runtime_session::{InferenceSession, Tensor};
use std::sync::Arc;
const H: usize = 16;
const E: usize = 3;
const K: usize = 2;
fn model(aggregation: bool) -> Vec<u8> {
    let mut g = Graph::new();
    g.opset_imports.insert("com.microsoft".into(), 1);
    let rows = Dim::Symbolic(g.create_symbol(Some("rows".into())));
    let x = g.create_named_value("x", DataType::Float32, vec![rows, Dim::Static(H)]);
    let router = g.create_named_value("router", DataType::Float32, vec![rows, Dim::Static(E)]);
    let w1 = g.create_named_value("w1", DataType::Uint8, static_shape([E, H, H]));
    let s1 = g.create_named_value("s1", DataType::Float32, static_shape([E, H, 1]));
    let w2 = g.create_named_value("w2", DataType::Uint8, static_shape([E, H, H]));
    let s2 = g.create_named_value("s2", DataType::Float32, static_shape([E, H, 1]));
    for v in [x, router, w1, s1, w2, s2] {
        g.add_input(v);
    }
    let mut inputs = vec![
        Some(x),
        Some(router),
        Some(w1),
        Some(s1),
        None,
        Some(w2),
        Some(s2),
    ];
    if aggregation {
        let a = g.create_named_value("aggregation", DataType::Float32, vec![rows, Dim::Static(E)]);
        g.add_input(a);
        inputs.resize(14, None);
        inputs.push(Some(a));
    }
    let y = g.create_named_value("y", DataType::Float32, vec![rows, Dim::Static(H)]);
    g.add_output(y);
    let mut node = Node::new(NodeId(0), "QMoE", inputs, vec![y]);
    node.domain = "com.microsoft".into();
    for (name, value) in [
        ("expert_weight_bits", Attribute::Int(8)),
        ("block_size", Attribute::Int(16)),
        ("k", Attribute::Int(K as i64)),
        ("normalize_routing_weights", Attribute::Int(1)),
        ("activation_type", Attribute::String(b"identity".to_vec())),
    ] {
        node.attributes.insert(name.into(), value);
    }
    g.insert_node(node);
    onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&g)).unwrap()
}
fn bytes(rows: usize, aggregation: bool) -> u64 {
    if rows == 0 {
        return 0;
    }
    let bank = 2 * rows * H * 4;
    let contribution = rows * K * H * 4;
    let decoder = (if rows > 1 { rows * H } else { 0 } + H) * 4;
    let index = size_of::<usize>();
    let selected = size_of::<(usize, f32)>();
    let route =
        (rows * K * (selected + index) + (3 * E + 1) * index + if aggregation { 0 } else { E * 4 })
            .next_multiple_of(align_of::<(usize, f32)>())
            + K * selected;
    ((bank + contribution + decoder).next_multiple_of(align_of::<usize>()) + route) as u64
}
#[test]
fn exact_routing_composite_refuses_short_budget_and_reuses_one_owner() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            for aggregation in [false, true] {
                let exact = bytes(3, aggregation);
                for budget in [exact - 1, exact] {
                    let books = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
                        DeviceKey::HOST,
                        0,
                        budget,
                        0,
                    )));
                    let manager = ProcessMemoryManager::new().unwrap();
                    let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
                    let resources = onnx_runtime_ep_cpu::CpuMemoryResources::default()
                        .with_workspace_pool(pool.clone())
                        .unwrap();
                    let cpu =
                        onnx_runtime_ep_cpu::CpuExecutionProvider::initialized_with_resources(
                            resources,
                        )
                        .unwrap();
                    let mut session = InferenceSession::builder()
                        .model_bytes(&model(aggregation))
                        .execution_provider(Arc::new(cpu))
                        .build()
                        .unwrap();
                    let mut codes = vec![128u8; E * H * H];
                    for expert in 0..E {
                        for feature in 0..H {
                            codes[(expert * H + feature) * H + feature] = 129;
                        }
                    }
                    let w1 = Tensor::from_raw(DataType::Uint8, vec![E, H, H], &codes).unwrap();
                    let w2 = Tensor::from_raw(DataType::Uint8, vec![E, H, H], &codes).unwrap();
                    let s1 = Tensor::from_f32(&[E, H, 1], &[1.; E * H]).unwrap();
                    let s2 = Tensor::from_f32(&[E, H, 1], &[1.; E * H]).unwrap();
                    let mut identity = None;
                    for rows in [3, 1, 0, 3] {
                        let x = Tensor::from_f32(&[rows, H], &vec![1.; rows * H]).unwrap();
                        let router = Tensor::from_f32(&[rows, E], &vec![0.; rows * E]).unwrap();
                        let a = Tensor::from_f32(&[rows, E], &vec![1.; rows * E]).unwrap();
                        let mut inputs = vec![
                            ("x", &x),
                            ("router", &router),
                            ("w1", &w1),
                            ("s1", &s1),
                            ("w2", &w2),
                            ("s2", &s2),
                        ];
                        if aggregation {
                            inputs.push(("aggregation", &a));
                        }
                        if budget < exact {
                            assert!(
                                session
                                    .prepare_with_device_bindings(&inputs, &mut [])
                                    .is_err()
                            );
                            assert!(session.run(&inputs).is_err());
                            assert_eq!(books.used(Tier::Host), 0);
                            assert!(manager.snapshot().unwrap().allocations.is_empty());
                            break;
                        }
                        let report = session
                            .prepare_with_device_bindings(&inputs, &mut [])
                            .unwrap();
                        assert_eq!(
                            report.require_complete().unwrap().step().bytes,
                            bytes(rows, aggregation)
                        );
                        for _ in 0..3 {
                            let y = session.run(&inputs).unwrap();
                            assert_eq!(
                                y[0].as_bytes(),
                                vec![1f32.to_ne_bytes(); rows * H].concat()
                            );
                            let allocations = manager.snapshot().unwrap().allocations;
                            assert_eq!(allocations.len(), 1);
                            assert_eq!(
                                allocations[0].identity,
                                *identity.get_or_insert(allocations[0].identity)
                            );
                            assert_eq!(books.used(Tier::Host), exact);
                        }
                        if rows == 0 {
                            pool.trim_idle(u64::MAX).unwrap();
                            identity = None;
                            assert_eq!(books.used(Tier::Host), 0);
                        }
                    }
                    drop(session);
                    pool.trim_idle(u64::MAX).unwrap();
                    assert_eq!(manager.process_used(Tier::Host), 0);
                }
            }
        });
}
