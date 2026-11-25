use std::sync::Arc;

use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, static_shape};
use onnx_runtime_session::{InferenceSession, Tensor};

fn roundtrip_graph(version: u64) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), version);
    let input = graph.create_named_value("signal", DataType::Float32, static_shape([2, 8, 1]));
    let spectrum = graph.create_named_value("spectrum", DataType::Float32, static_shape([2, 5, 2]));
    let output = graph.create_named_value("recovered", DataType::Float32, static_shape([2, 8, 1]));
    graph.add_input(input);
    graph.add_output(output);
    for (id, x, y, inverse) in [(0, input, spectrum, 0), (1, spectrum, output, 1)] {
        let mut node = Node::new(NodeId(id), "DFT", vec![Some(x)], vec![y]);
        node.attributes.insert("onesided".into(), Attribute::Int(1));
        node.attributes
            .insert("inverse".into(), Attribute::Int(inverse));
        if version < 20 {
            node.attributes.insert("axis".into(), Attribute::Int(-2));
        }
        graph.insert_node(node);
    }
    graph
}

#[test]
fn real_fft_roundtrip_loads_and_executes_with_real_inverse_output() {
    for version in [17, 20] {
        let graph = roundtrip_graph(version);
        let bytes =
            onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&graph)).unwrap();
        let cpu = onnx_runtime_ep_cpu::CpuExecutionProvider::initialized_with_resources(
            Default::default(),
        )
        .unwrap();
        let mut session = InferenceSession::builder()
            .model_bytes(&bytes)
            .execution_provider(Arc::new(cpu))
            .build()
            .unwrap();
        let values = (0..16)
            .map(|i| (i as f32 * 0.375).sin() + i as f32 * 0.125)
            .collect::<Vec<_>>();
        let input = Tensor::from_f32(&[2, 8, 1], &values).unwrap();
        for _ in 0..3 {
            let outputs = session.run(&[("signal", &input)]).unwrap();
            assert_eq!(outputs[0].shape, [2, 8, 1]);
            for (&actual, &expected) in outputs[0].try_as_slice_f32().unwrap().iter().zip(&values) {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "opset {version}: {actual} versus {expected}"
                );
            }
        }
    }
}

#[test]
fn governed_fourier_graph_reuses_one_peak_and_refuses_unfunded_scratch() {
    use onnx_runtime_memory_api::{DeviceKey, Tier};
    use onnx_runtime_memory_governor::{
        HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor,
        ProcessMemoryManager,
    };
    for budget in [127u64, 128] {
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
        let bytes = onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(
            &roundtrip_graph(20),
        ))
        .unwrap();
        let mut session = InferenceSession::builder()
            .model_bytes(&bytes)
            .execution_provider(Arc::new(cpu))
            .build()
            .unwrap();
        let values = (0..16).map(|i| i as f32 * 0.125 - 0.75).collect::<Vec<_>>();
        let input = Tensor::from_f32(&[2, 8, 1], &values).unwrap();
        let inputs = [("signal", &input)];
        if budget < 128 {
            assert!(
                session
                    .prepare_with_device_bindings(&inputs, &mut [])
                    .is_err()
            );
            assert!(
                session.run(&inputs).is_err(),
                "no uncharged frame-buffer fallback"
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
                128,
                "one 8-point transform, never the sum of nodes or batches"
            );
            let outputs = session.run(&inputs).unwrap();
            for (&actual, &expected) in outputs[0].try_as_slice_f32().unwrap().iter().zip(&values) {
                assert!((actual - expected).abs() < 1e-5);
            }
            let snapshot = manager.snapshot().unwrap();
            assert_eq!(snapshot.allocations.len(), 1);
            let expected = identity.get_or_insert(snapshot.allocations[0].identity);
            assert_eq!(snapshot.allocations[0].identity, *expected);
            assert_eq!(books.used(Tier::Host), 128);
            assert_eq!(manager.process_used(Tier::Host), 128);
        }
        pool.trim_idle(128).unwrap();
        assert_eq!(books.used(Tier::Host), 0);
        session.run(&inputs).unwrap();
        assert_eq!(
            books.used(Tier::Host),
            128,
            "a cached metadata plan must re-admit reclaimed backing"
        );
        drop(session);
        pool.trim_idle(u64::MAX).unwrap();
        assert_eq!(books.used(Tier::Host), 0);
        assert_eq!(manager.process_used(Tier::Host), 0);
    }
}
