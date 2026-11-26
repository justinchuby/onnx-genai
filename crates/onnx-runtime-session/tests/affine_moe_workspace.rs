use onnx_runtime_ir::{
    Attribute, DataType, Dim, Graph, Node, NodeId, RUNTIME_DOMAIN, static_shape,
};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use onnx_runtime_session::{InferenceSession, Tensor};
use std::sync::Arc;

fn model() -> Vec<u8> {
    let mut g = Graph::new();
    g.opset_imports.insert(RUNTIME_DOMAIN.into(), 1);
    let t = Dim::Symbolic(g.create_symbol(Some("tokens".into())));
    let shapes = [
        vec![t, Dim::Static(4)],
        vec![t, Dim::Static(3)],
        static_shape([3, 8, 2]),
        static_shape([3, 8, 2]),
        static_shape([3, 8, 1]),
        static_shape([3, 4, 2]),
        static_shape([3, 4, 2]),
        static_shape([3, 4, 1]),
    ];
    let names = ["x", "router", "w1", "s1", "z1", "w2", "s2", "z2"];
    let inputs = shapes
        .into_iter()
        .zip(names)
        .enumerate()
        .map(|(i, (shape, name))| {
            let v = g.create_named_value(
                name,
                if [2, 4, 5, 7].contains(&i) {
                    DataType::Uint8
                } else {
                    DataType::Float32
                },
                shape,
            );
            g.add_input(v);
            Some(v)
        })
        .collect();
    let y = g.create_named_value("y", DataType::Float32, vec![t, Dim::Static(4)]);
    g.add_output(y);
    let mut n = Node::new(
        NodeId(0),
        onnx_runtime_optimizer::affine_moe::OP,
        inputs,
        vec![y],
    );
    n.domain = RUNTIME_DOMAIN.into();
    for (key, value) in [
        ("k", 2),
        ("first_block", 2),
        ("first_zero", 7),
        ("down_block", 2),
        ("down_zero", 7),
    ] {
        n.attributes.insert(key.into(), Attribute::Int(value));
    }
    for (key, value) in [("alpha", 1.3), ("beta", 0.2), ("limit", 7.)] {
        n.attributes.insert(key.into(), Attribute::Float(value));
    }
    g.insert_node(n);
    onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&g)).unwrap()
}
#[test]
fn governed_moe_uses_one_exact_execution_lane_and_reuses_dynamic_peak() {
    let peak = ((20 * size_of::<f32>()).next_multiple_of(align_of::<usize>())
        + 3 * size_of::<usize>()) as u64;
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
            onnx_runtime_ep_cpu::DerivedWeightMemory::governed(books.as_ref(), HolderId::new(13))
                .unwrap(),
            Default::default(),
        )
        .unwrap()
        .with_workspace_pool(pool.clone())
        .unwrap();
        let cpu = onnx_runtime_ep_cpu::CpuExecutionProvider::initialized_with_resources(resources)
            .unwrap();
        let mut session = InferenceSession::builder()
            .model_bytes(&model())
            .execution_provider(Arc::new(cpu))
            .build()
            .unwrap();
        let mut identity = None;
        for (iteration, t) in [3, 1, 2, 3].into_iter().enumerate() {
            let x = Tensor::from_f32(&[t, 4], &vec![0.5; t * 4]).unwrap();
            let router = Tensor::from_f32(&[t, 3], &vec![1.; t * 3]).unwrap();
            let w1 = Tensor::from_raw(DataType::Uint8, vec![3, 8, 2], &[0x88; 48]).unwrap();
            let s1 = Tensor::from_f32(&[3, 8, 2], &[0.125; 48]).unwrap();
            let z1 = Tensor::from_raw(DataType::Uint8, vec![3, 8, 1], &[0x77; 24]).unwrap();
            let w2 = Tensor::from_raw(DataType::Uint8, vec![3, 4, 2], &[0x88; 24]).unwrap();
            let s2 = Tensor::from_f32(&[3, 4, 2], &[0.25; 24]).unwrap();
            let z2 = Tensor::from_raw(DataType::Uint8, vec![3, 4, 1], &[0x77; 12]).unwrap();
            let inputs = [
                ("x", &x),
                ("router", &router),
                ("w1", &w1),
                ("s1", &s1),
                ("z1", &z1),
                ("w2", &w2),
                ("s2", &s2),
                ("z2", &z2),
            ];
            if budget < peak {
                assert!(session.run(&inputs).is_err());
                assert_eq!(books.used(Tier::Host), 0);
                assert!(manager.snapshot().unwrap().allocations.is_empty());
                break;
            }
            for _ in 0..2 {
                let output = session.run(&inputs).unwrap();
                assert_eq!(output[0].shape, [t, 4]);
                assert!(
                    output[0]
                        .as_bytes()
                        .chunks_exact(4)
                        .all(|b| f32::from_ne_bytes(b.try_into().unwrap()).is_finite())
                );
                let snapshot = manager.snapshot().unwrap();
                assert_eq!(snapshot.allocations.len(), 1);
                assert_eq!(
                    snapshot.allocations[0].identity,
                    *identity.get_or_insert(snapshot.allocations[0].identity)
                );
                assert_eq!(books.used(Tier::Host), peak);
            }
            if iteration == 2 {
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
