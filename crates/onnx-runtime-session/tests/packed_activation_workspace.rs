use onnx_runtime_ir::{
    Attribute, DataType, Dim, Graph, Node, NodeId, RUNTIME_DOMAIN, static_shape,
};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use onnx_runtime_session::{InferenceSession, Tensor};
use std::sync::Arc;

fn bytes(rows: usize) -> u64 {
    let columns = if rows > 1 { rows * 2 * 4 } else { 0 };
    (columns + (rows * 34).next_multiple_of(4) + rows * 2 * 4) as u64
}

fn model() -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(RUNTIME_DOMAIN.into(), 1);
    let rows = Dim::Symbolic(graph.create_symbol(Some("rows".into())));
    let x = graph.create_named_value("x", DataType::Float32, vec![rows, Dim::Static(32)]);
    let w = graph.create_named_value("w", DataType::Uint8, static_shape([2, 1, 34]));
    let y = graph.create_named_value("y", DataType::Float32, vec![rows, Dim::Static(2)]);
    graph.add_input(x);
    graph.add_input(w);
    graph.add_output(y);
    let mut node = Node::new(
        NodeId(0),
        "BlockQuantizedMatMul",
        vec![Some(x), Some(w)],
        vec![y],
    );
    node.domain = RUNTIME_DOMAIN.into();
    for (name, value) in [
        ("K", Attribute::Int(32)),
        ("N", Attribute::Int(2)),
        ("format", Attribute::String(b"q8_0".to_vec())),
        ("activation_arithmetic", Attribute::String(b"q8_0".to_vec())),
        ("block_layout_version", Attribute::Int(2)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    graph.insert_node(node);
    onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&graph)).unwrap()
}

#[test]
fn q8_session_admits_exact_whole_operation_reuses_and_readmits_after_trim() {
    for budget in [bytes(3) - 1, bytes(3)] {
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
        let mut packed = vec![0u8; 68];
        for feature in 0..2 {
            packed[feature * 34..feature * 34 + 2].copy_from_slice(&half::f16::ONE.to_le_bytes());
            packed[feature * 34 + 2 + feature] = 1;
        }
        let w = Tensor::from_raw(DataType::Uint8, vec![2, 1, 34], &packed).unwrap();
        let mut identity = None;
        for rows in [3, 1, 3] {
            let x = Tensor::from_f32(&[rows, 32], &vec![127.; rows * 32]).unwrap();
            let inputs = [("x", &x), ("w", &w)];
            if budget < bytes(3) {
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
            assert_eq!(report.require_complete().unwrap().step().bytes, bytes(rows));
            for _ in 0..3 {
                let output = session.run(&inputs).unwrap();
                assert_eq!(output[0].shape, [rows, 2]);
                assert_eq!(
                    output[0].as_bytes(),
                    vec![127f32.to_ne_bytes(); rows * 2].concat()
                );
                let allocations = manager.snapshot().unwrap().allocations;
                assert_eq!(allocations.len(), 1);
                assert_eq!(
                    allocations[0].identity,
                    *identity.get_or_insert(allocations[0].identity)
                );
                assert_eq!(books.used(Tier::Host), bytes(3));
            }
            if rows == 1 {
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
