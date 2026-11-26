use onnx_runtime_ir::{
    Attribute, DataType, Dim, Graph, Node, NodeId, RUNTIME_DOMAIN, static_shape,
};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use onnx_runtime_session::{InferenceSession, Tensor};
use std::sync::Arc;
fn bytes(t: usize) -> u64 {
    if t == 0 {
        0
    } else {
        (((6 * t + 4) * 4).next_multiple_of(align_of::<usize>()) + 2 * t * size_of::<usize>())
            as u64
    }
}
fn model(int32: bool, echo: bool) -> Vec<u8> {
    let mut g = Graph::new();
    g.opset_imports.insert(RUNTIME_DOMAIN.into(), 1);
    g.opset_imports.insert(String::new(), 29);
    let t = Dim::Symbolic(g.create_symbol(Some("tokens".into())));
    let shapes = [
        vec![t, Dim::Static(4)],
        static_shape([3, 3, 2]),
        vec![t, Dim::Static(2)],
        static_shape([3, 3, 2]),
        static_shape([3, 3, 1]),
    ];
    let mut x = None;
    let inputs = shapes
        .into_iter()
        .zip(["x", "weights", "routes", "scales", "zeros"])
        .enumerate()
        .map(|(i, (shape, name))| {
            let dtype = match i {
                1 | 4 => DataType::Uint8,
                2 => {
                    if int32 {
                        DataType::Int32
                    } else {
                        DataType::Int64
                    }
                }
                _ => DataType::Float32,
            };
            let v = g.create_named_value(name, dtype, shape);
            g.add_input(v);
            if i == 0 {
                x = Some(v);
            }
            Some(v)
        })
        .collect();
    let y = g.create_named_value(
        "y",
        DataType::Float32,
        vec![t, Dim::Static(2), Dim::Static(3)],
    );
    g.add_output(y);
    let mut n = Node::new(
        NodeId(0),
        onnx_runtime_ir::quantized_grouped::OP,
        inputs,
        vec![y],
    );
    n.domain = RUNTIME_DOMAIN.into();
    for (key, value) in [("depth", 4), ("block_size", 2), ("zero", 7)] {
        n.attributes.insert(key.into(), Attribute::Int(value));
    }
    g.insert_node(n);
    if echo {
        g.add_output(x.unwrap());
    }
    onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&g)).unwrap()
}
fn setup(
    budget: u64,
    int32: bool,
    echo: bool,
) -> (
    InferenceSession,
    Arc<LedgerGovernor>,
    HostWorkspacePool,
    ProcessMemoryManager,
) {
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
    let cpu =
        onnx_runtime_ep_cpu::CpuExecutionProvider::initialized_with_resources(resources).unwrap();
    let session = InferenceSession::builder()
        .model_bytes(&model(int32, echo))
        .execution_provider(Arc::new(cpu))
        .build()
        .unwrap();
    (session, books, pool, manager)
}
fn banks(t: usize, int32: bool) -> [Tensor; 4] {
    let routes = (0..2 * t).map(|i| (i % 3) as i64).collect::<Vec<_>>();
    let routes = if int32 {
        Tensor::from_raw(
            DataType::Int32,
            vec![t, 2],
            &routes
                .iter()
                .flat_map(|&i| (i as i32).to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    } else {
        Tensor::from_i64(&[t, 2], &routes).unwrap()
    };
    [
        Tensor::from_raw(DataType::Uint8, vec![3, 3, 2], &[0x88; 18]).unwrap(),
        routes,
        Tensor::from_f32(&[3, 3, 2], &[1.; 18]).unwrap(),
        Tensor::from_raw(DataType::Uint8, vec![3, 3, 1], &[0x77; 9]).unwrap(),
    ]
}
fn check(output: &Tensor, t: usize) {
    assert_eq!(output.shape, [t, 2, 3]);
    assert_eq!(
        output.as_bytes(),
        vec![2f32.to_ne_bytes(); 6 * t]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
    );
}
#[test]
fn metadata_ready_budget_is_exact_and_dynamic_requests_reuse_one_retained_peak() {
    for int32 in [false, true] {
        for budget in [bytes(3) - 1, bytes(3)] {
            let (mut session, books, pool, manager) = setup(budget, int32, false);
            let mut identity = None;
            for (iteration, t) in [3, 1, 0, 3].into_iter().enumerate() {
                let x = Tensor::from_f32(&[t, 4], &vec![0.5; t * 4]).unwrap();
                let b = banks(t, int32);
                let inputs = [
                    ("x", &x),
                    ("weights", &b[0]),
                    ("routes", &b[1]),
                    ("scales", &b[2]),
                    ("zeros", &b[3]),
                ];
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
                assert_eq!(report.require_complete().unwrap().step().bytes, bytes(t));
                for _ in 0..2 {
                    let outputs = session.run(&inputs).unwrap();
                    check(&outputs[0], t);
                    let snapshot = manager.snapshot().unwrap();
                    if t == 0 {
                        assert!(snapshot.allocations.is_empty());
                        assert_eq!(books.used(Tier::Host), 0);
                    } else {
                        assert_eq!(snapshot.allocations.len(), 1);
                        assert_eq!(
                            snapshot.allocations[0].identity,
                            *identity.get_or_insert(snapshot.allocations[0].identity)
                        );
                        assert_eq!(books.used(Tier::Host), bytes(3));
                    }
                }
                if iteration == 1 {
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
}
#[test]
fn fixed_outer_strides_reach_affine_without_full_activation_snapshot() {
    let t = 3;
    let (mut session, books, pool, manager) = setup(bytes(t), false, true);
    let b = banks(t, false);
    let mut binding = session
        .allocate_device_binding_fixed_strides("x", "x", DataType::Float32, vec![t, 8], vec![t, 4])
        .unwrap();
    let mut physical = vec![123f32; t * 8];
    for row in 0..t {
        physical[row * 8..row * 8 + 4].fill(0.5);
    }
    binding
        .write_bytes(
            0,
            &physical
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let inputs = [
        ("weights", &b[0]),
        ("routes", &b[1]),
        ("scales", &b[2]),
        ("zeros", &b[3]),
    ];
    let report = session
        .prepare_with_device_bindings(&inputs, std::slice::from_mut(&mut binding))
        .unwrap();
    assert_eq!(report.require_complete().unwrap().step().bytes, bytes(t));
    let identity = manager.snapshot().unwrap().allocations[0].identity;
    for tokens in [3, 1, 3] {
        binding.set_logical_shape(vec![tokens, 4]).unwrap();
        // Reserve against physical symbolic capacity even at a smaller logical
        // prefix; the ordinary route input supplies the matching capacity fact.
        let report = session
            .prepare_with_device_bindings(&inputs, std::slice::from_mut(&mut binding))
            .unwrap();
        assert_eq!(report.require_complete().unwrap().step().bytes, bytes(t));
        let current = banks(tokens, false);
        let current = [
            ("weights", &current[0]),
            ("routes", &current[1]),
            ("scales", &current[2]),
            ("zeros", &current[3]),
        ];
        let output = session
            .run_with_device_bindings(&current, std::slice::from_mut(&mut binding))
            .unwrap();
        check(output[0].as_ref().unwrap(), tokens);
        let snapshot = manager.snapshot().unwrap();
        assert_eq!(snapshot.allocations.len(), 1);
        assert_eq!(snapshot.allocations[0].identity, identity);
        assert_eq!(books.used(Tier::Host), bytes(t));
    }
    drop(binding);
    drop(session);
    pool.trim_idle(u64::MAX).unwrap();
    assert_eq!(books.used(Tier::Host), 0);
}

#[test]
fn binding_capacity_validation_rejects_wrong_static_prefix_and_shared_symbol_conflicts() {
    let (mut session, books, pool, manager) = setup(bytes(3), false, true);
    let b = banks(3, false);
    let inputs = [
        ("weights", &b[0]),
        ("routes", &b[1]),
        ("scales", &b[2]),
        ("zeros", &b[3]),
    ];
    let mut bad = session
        .allocate_device_binding_fixed_strides("x", "x", DataType::Float32, vec![3, 8], vec![3, 3])
        .unwrap();
    let text = session
        .prepare_with_device_bindings(&inputs, std::slice::from_mut(&mut bad))
        .unwrap_err()
        .to_string();
    assert!(
        text.contains("axis 1") && text.contains("logical extent 4"),
        "{text}"
    );
    assert!(
        session
            .allocate_device_binding_fixed_strides(
                "x",
                "x",
                DataType::Float32,
                vec![3, 3],
                vec![3, 4]
            )
            .is_err()
    );
    let mut binding = session
        .allocate_device_binding_fixed_strides("x", "x", DataType::Float32, vec![3, 8], vec![3, 4])
        .unwrap();
    let current = banks(2, false);
    let current = [
        ("weights", &current[0]),
        ("routes", &current[1]),
        ("scales", &current[2]),
        ("zeros", &current[3]),
    ];
    assert!(matches!(
        session.prepare_with_device_bindings(&current, std::slice::from_mut(&mut binding)),
        Err(onnx_runtime_session::SessionError::SymbolConflict { .. })
    ));
    assert_eq!(books.used(Tier::Host), 0);
    assert!(manager.snapshot().unwrap().allocations.is_empty());
    drop(session);
    pool.trim_idle(u64::MAX).unwrap();
}
#[test]
fn ordinary_logical_prefix_binding_uses_the_same_static_capacity_rule() {
    let (mut session, books, pool, manager) = setup(bytes(3), false, true);
    let b = banks(3, false);
    let inputs = [
        ("weights", &b[0]),
        ("routes", &b[1]),
        ("scales", &b[2]),
        ("zeros", &b[3]),
    ];
    let mut binding = session
        .allocate_device_binding("x", Some("x"), DataType::Float32, vec![3, 8], vec![3, 4])
        .unwrap();
    assert!(binding.exposes_logical_input_shape());
    binding
        .write_bytes(
            0,
            &vec![0.5f32.to_ne_bytes(); 12]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let report = session
        .prepare_with_device_bindings(&inputs, std::slice::from_mut(&mut binding))
        .unwrap();
    assert_eq!(report.require_complete().unwrap().step().bytes, bytes(3));
    let output = session
        .run_with_device_bindings(&inputs, std::slice::from_mut(&mut binding))
        .unwrap();
    check(output[0].as_ref().unwrap(), 3);
    assert_eq!(manager.snapshot().unwrap().allocations.len(), 1);
    drop(binding);
    drop(session);
    pool.trim_idle(u64::MAX).unwrap();
    assert_eq!(books.used(Tier::Host), 0);
}
