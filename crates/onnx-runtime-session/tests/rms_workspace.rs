use std::sync::Arc;

use onnx_runtime_ir::{DataType, Graph, Node, NodeId, static_shape};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use onnx_runtime_session::{InferenceSession, Tensor};

fn graph() -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 23);
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let x = graph.create_named_value("X", DataType::Float32, static_shape([2, 4]));
    let scale = graph.create_named_value("scale", DataType::Float32, static_shape([4]));
    let first = graph.create_named_value("first", DataType::Float32, static_shape([2, 4]));
    let y = graph.create_named_value("Y", DataType::Float32, static_shape([2, 4]));
    let inv = graph.create_named_value("inv", DataType::Float32, static_shape([2, 1]));
    graph.add_input(x);
    graph.add_input(scale);
    graph.insert_node(Node::new(
        NodeId(0),
        "RMSNormalization",
        vec![Some(x), Some(scale)],
        vec![first],
    ));
    let mut second = Node::new(
        NodeId(1),
        "SimplifiedLayerNormalization",
        vec![Some(first), Some(scale)],
        vec![y, inv],
    );
    second.domain = "com.microsoft".into();
    graph.insert_node(second);
    graph.add_output(y);
    graph.add_output(inv);
    graph
}

#[test]
fn rms_graph_uses_one_peak_refuses_unfunded_staging_and_readmits_after_trim() {
    let bytes =
        onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&graph())).unwrap();
    let values = [1., 2., 3., 4., -2., -3., 4., 5.];
    let scales = [1., 2., 0.5, -1.];
    let x = Tensor::from_f32(&[2, 4], &values).unwrap();
    let scale = Tensor::from_f32(&[4], &scales).unwrap();
    let inputs = [("X", &x), ("scale", &scale)];
    let mut expected = values.map(f64::from);
    let mut expected_inv = [0.; 2];
    for _ in 0..2 {
        for (g, row) in expected.chunks_exact_mut(4).enumerate() {
            let inv = 1. / (row.iter().map(|x| x * x).sum::<f64>() / 4. + 1e-5).sqrt();
            expected_inv[g] = inv;
            for (value, scale) in row.iter_mut().zip(scales) {
                *value = *value * inv * f64::from(scale);
            }
        }
    }
    for budget in [39u64, 40] {
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
            .model_bytes(&bytes)
            .execution_provider(Arc::new(cpu))
            .build()
            .unwrap();
        if budget < 40 {
            assert!(
                session
                    .prepare_with_device_bindings(&inputs, &mut [])
                    .is_err()
            );
            assert!(
                session.run(&inputs).is_err(),
                "no unfunded staging fallback"
            );
            assert_eq!(books.used(Tier::Host), 0);
            assert_eq!(manager.process_used(Tier::Host), 0);
            continue;
        }
        let mut identity = None;
        for _ in 0..8 {
            let peak = session
                .prepare_with_device_bindings(&inputs, &mut [])
                .unwrap()
                .require_complete()
                .unwrap();
            assert_eq!(
                peak.step().bytes,
                40,
                "max(32-byte RMS, 40-byte Simplified), never the sum"
            );
            let outputs = session.run(&inputs).unwrap();
            assert_eq!(outputs.len(), 2);
            for (&actual, expected) in outputs[0].try_as_slice_f32().unwrap().iter().zip(expected) {
                assert!((f64::from(actual) - expected).abs() < 1e-5);
            }
            for (&actual, expected) in outputs[1]
                .try_as_slice_f32()
                .unwrap()
                .iter()
                .zip(expected_inv)
            {
                assert!((f64::from(actual) - expected).abs() < 1e-5);
            }
            let snapshot = manager.snapshot().unwrap();
            assert_eq!(snapshot.allocations.len(), 1);
            let expected = identity.get_or_insert(snapshot.allocations[0].identity);
            assert_eq!(snapshot.allocations[0].identity, *expected);
            assert_eq!(books.used(Tier::Host), 40);
        }
        pool.trim_idle(40).unwrap();
        assert_eq!(books.used(Tier::Host), 0);
        session.run(&inputs).unwrap();
        assert_eq!(
            books.used(Tier::Host),
            40,
            "cached metadata re-admits reclaimed storage"
        );
        drop(session);
        pool.trim_idle(u64::MAX).unwrap();
        assert_eq!(books.used(Tier::Host), 0);
        assert_eq!(manager.process_used(Tier::Host), 0);
    }
}
