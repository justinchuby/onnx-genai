//! Native executor admission uses the actual packed-bank layout, once per op.
use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, RUNTIME_DOMAIN, static_shape};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use onnx_runtime_session::{InferenceSession, Tensor};
use std::sync::Arc;

fn model() -> Vec<u8> {
    let mut g = Graph::new();
    g.opset_imports.insert(RUNTIME_DOMAIN.into(), 1);
    let x = g.create_named_value("x", DataType::Float32, static_shape([3, 32]));
    let router = g.create_named_value("router", DataType::Float32, static_shape([3, 2]));
    let w1 = g.create_named_value("w1", DataType::Uint8, static_shape([2, 32, 1, 34]));
    let w2 = g.create_named_value("w2", DataType::Uint8, static_shape([2, 32, 1, 34]));
    for v in [x, router, w1, w2] {
        g.add_input(v);
    }
    let y = g.create_named_value("y", DataType::Float32, static_shape([3, 32]));
    g.add_output(y);
    // Fixed-stride device bindings pair an input with a declared output.
    g.add_output(w1);
    let mut node = Node::new(
        NodeId(0),
        "BlockQuantizedMoE",
        vec![Some(x), Some(router), Some(w1), None, Some(w2)],
        vec![y],
    );
    node.domain = RUNTIME_DOMAIN.into();
    for (name, value) in [
        ("fc1_format", Attribute::String(b"q8_0".to_vec())),
        ("fc2_format", Attribute::String(b"q8_0".to_vec())),
        ("activation_type", Attribute::String(b"identity".to_vec())),
        ("k", Attribute::Int(1)),
        ("normalize_routing_weights", Attribute::Int(1)),
        ("activation_arithmetic", Attribute::String(b"q8_0".to_vec())),
        ("block_layout_version", Attribute::Int(2)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    g.insert_node(node);
    onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&g)).unwrap()
}

#[test]
fn native_packed_moe_defers_metadata_and_admits_exact_layout_then_reuses_one_owner() {
    // Three Q8 activation rows + result columns; two FC banks + accumulator.
    let base = (3 * 34usize).next_multiple_of(4) + 3 * 32 * 4 + 3 * 32 * 4 * 3;
    let packed = 32 * 34;
    let codes = (0..64)
        .flat_map(|_| {
            let mut block = [0u8; 34];
            block[..2].copy_from_slice(&half::f16::ONE.to_le_bytes());
            block[2] = 1;
            block
        })
        .collect::<Vec<_>>();
    for strided in [false, true] {
        let exact = (base + if strided { packed } else { 0 }) as u64;
        for budget in [exact - 1, exact] {
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
            let cpu =
                onnx_runtime_ep_cpu::CpuExecutionProvider::initialized_with_resources(resources)
                    .unwrap();
            let mut session = InferenceSession::builder()
                .model_bytes(&model())
                .execution_provider(Arc::new(cpu))
                .build()
                .unwrap();
            let x = Tensor::from_f32(&[3, 32], &[127.; 96]).unwrap();
            let router = Tensor::from_f32(&[3, 2], &[1., 0., 1., 0., 1., 0.]).unwrap();
            let w1 = Tensor::from_raw(DataType::Uint8, vec![2, 32, 1, 34], &codes).unwrap();
            let w2 = Tensor::from_raw(DataType::Uint8, vec![2, 32, 1, 34], &codes).unwrap();
            let mut bindings = Vec::new();
            if strided {
                let mut binding = session
                    .allocate_device_binding_fixed_strides(
                        "w1",
                        "w1",
                        DataType::Uint8,
                        vec![2, 32, 1, 68],
                        vec![2, 32, 1, 34],
                    )
                    .unwrap();
                let physical = codes
                    .chunks_exact(34)
                    .flat_map(|b| b.iter().copied().chain([0xa5; 34]))
                    .collect::<Vec<_>>();
                binding.write_bytes(0, &physical).unwrap();
                bindings.push(binding);
            }
            let mut inputs = vec![("x", &x), ("router", &router), ("w2", &w2)];
            if !strided {
                inputs.push(("w1", &w1));
            }
            let report = session
                .prepare_with_device_bindings(&inputs, &mut bindings)
                .unwrap();
            assert!(
                report.require_complete().is_err(),
                "packed strides need live refinement"
            );
            assert_eq!(
                books.used(Tier::Host),
                0,
                "metadata must not retain a conservative packed lane"
            );
            assert!(manager.snapshot().unwrap().allocations.is_empty());
            if budget < exact {
                assert!(
                    session
                        .run_with_device_bindings(&inputs, &mut bindings)
                        .is_err()
                );
                assert_eq!(books.used(Tier::Host), 0);
                assert!(manager.snapshot().unwrap().allocations.is_empty());
                continue;
            }
            let mut identity = None;
            for iteration in 0..4 {
                let outputs = session
                    .run_with_device_bindings(&inputs, &mut bindings)
                    .unwrap();
                assert_eq!(
                    outputs[0].as_ref().unwrap().as_bytes(),
                    vec![127f32.to_ne_bytes(); 96].concat()
                );
                assert_eq!(books.used(Tier::Host), exact);
                let allocations = manager.snapshot().unwrap().allocations;
                assert_eq!(allocations.len(), 1);
                assert_eq!(
                    allocations[0].identity,
                    *identity.get_or_insert(allocations[0].identity)
                );
                if iteration == 1 {
                    pool.trim_idle(u64::MAX).unwrap();
                    identity = None;
                    assert_eq!(books.used(Tier::Host), 0);
                }
            }
            drop(session);
            pool.trim_idle(u64::MAX).unwrap();
            assert_eq!(books.used(Tier::Host), 0);
            assert_eq!(manager.process_used(Tier::Host), 0);
        }
    }
}
