use onnx_runtime_ir::{DataType, Dim, Graph, Node, NodeId, static_shape};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use onnx_runtime_session::{InferenceSession, Tensor, WorkspacePreparationReport};
use std::sync::Arc;
fn bytes(t: usize, dt: DataType, strided: bool) -> u64 {
    let values = if t == 0 {
        0
    } else {
        4 * t
            + if dt != DataType::Float32 {
                64 * t + 384
            } else if strided {
                64 * t
            } else {
                0
            }
    };
    let width = if dt == DataType::Float64 { 8 } else { 4 };
    ((values * width).next_multiple_of(align_of::<usize>()) + 2 * t * size_of::<usize>()) as u64
}
fn model(int32: bool, echo: bool, dt: DataType) -> Vec<u8> {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 29);
    let t = Dim::Symbolic(g.create_symbol(Some("tokens".into())));
    let x = g.create_named_value("x", dt, vec![t, Dim::Static(64)]);
    let w = g.create_named_value("weights", dt, static_shape([3, 64, 2]));
    let r = g.create_named_value(
        "routes",
        if int32 {
            DataType::Int32
        } else {
            DataType::Int64
        },
        vec![t, Dim::Static(2)],
    );
    for v in [x, w, r] {
        g.add_input(v);
    }
    let y = g.create_named_value("y", dt, vec![t, Dim::Static(2), Dim::Static(2)]);
    g.add_output(y);
    g.insert_node(Node::new(
        NodeId(0),
        "GroupedMatMul",
        vec![Some(x), Some(w), Some(r)],
        vec![y],
    ));
    if echo {
        g.add_output(x);
    }
    onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&g)).unwrap()
}
fn floats(dt: DataType, shape: &[usize], value: f32) -> Tensor {
    let one = match dt {
        DataType::Float64 => f64::from(value).to_ne_bytes().to_vec(),
        DataType::Float16 => half::f16::from_f32(value).to_bits().to_ne_bytes().to_vec(),
        DataType::BFloat16 => half::bf16::from_f32(value).to_bits().to_ne_bytes().to_vec(),
        _ => value.to_ne_bytes().to_vec(),
    };
    Tensor::from_raw(dt, shape.to_vec(), &one.repeat(shape.iter().product())).unwrap()
}
fn banks(t: usize, int32: bool, dt: DataType) -> [Tensor; 2] {
    let routes = (0..2 * t).map(|i| (i % 3) as i64).collect::<Vec<_>>();
    let r = if int32 {
        Tensor::from_raw(
            DataType::Int32,
            vec![t, 2],
            &routes
                .iter()
                .flat_map(|&v| (v as i32).to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    } else {
        Tensor::from_i64(&[t, 2], &routes).unwrap()
    };
    [floats(dt, &[3, 64, 2], 1.), r]
}
fn check(y: &Tensor, t: usize, dt: DataType) {
    assert_eq!(y.shape, [t, 2, 2]);
    assert_eq!(y.as_bytes(), floats(dt, &[t, 2, 2], 16.).as_bytes());
}
fn setup(
    budget: u64,
    int32: bool,
    echo: bool,
    dtype: DataType,
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
        .model_bytes(&model(int32, echo, dtype))
        .execution_provider(Arc::new(cpu))
        .build()
        .unwrap();
    (session, books, pool, manager)
}

#[test]
fn exact_budgets_dynamic_shapes_and_trim_cover_deferred_f32_and_ready_other_floats() {
    for dt in [
        DataType::Float32,
        DataType::Float64,
        DataType::Float16,
        DataType::BFloat16,
    ] {
        for int32 in [false, true] {
            for short in [false, true] {
                let budget = bytes(3, dt, false) - u64::from(short);
                let (mut session, books, pool, manager) = setup(budget, int32, false, dt);
                let mut identity = None;
                for (iteration, t) in [3, 1, 0, 3].into_iter().enumerate() {
                    let x = floats(dt, &[t, 64], 0.25);
                    let b = banks(t, int32, dt);
                    let inputs = [("x", &x), ("weights", &b[0]), ("routes", &b[1])];
                    if short {
                        if dt == DataType::Float32 {
                            assert!(matches!(
                                session
                                    .prepare_with_device_bindings(&inputs, &mut [])
                                    .unwrap(),
                                WorkspacePreparationReport::Deferred { .. }
                            ));
                        } else {
                            assert!(
                                session
                                    .prepare_with_device_bindings(&inputs, &mut [])
                                    .is_err()
                            );
                        }
                        assert!(session.run(&inputs).is_err());
                        assert_eq!(books.used(Tier::Host), 0);
                        assert!(manager.snapshot().unwrap().allocations.is_empty());
                        break;
                    }
                    let report = session
                        .prepare_with_device_bindings(&inputs, &mut [])
                        .unwrap();
                    if dt == DataType::Float32 && t != 0 {
                        assert!(matches!(
                            report,
                            WorkspacePreparationReport::Deferred { .. }
                        ));
                    } else {
                        assert_eq!(
                            report.require_complete().unwrap().step().bytes,
                            bytes(t, dt, false)
                        );
                    }
                    for _ in 0..2 {
                        let output = session.run(&inputs).unwrap();
                        check(&output[0], t, dt);
                        let s = manager.snapshot().unwrap();
                        if t == 0 {
                            assert!(s.allocations.is_empty());
                            assert_eq!(books.used(Tier::Host), 0);
                        } else {
                            assert_eq!(s.allocations.len(), 1);
                            assert_eq!(
                                s.allocations[0].identity,
                                *identity.get_or_insert(s.allocations[0].identity)
                            );
                            assert_eq!(books.used(Tier::Host), bytes(3, dt, false));
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
}
#[test]
fn fixed_outer_strides_are_charged_exactly_and_never_gathered_upstream() {
    let dt = DataType::Float32;
    for short in [false, true] {
        let (mut session, books, pool, manager) =
            setup(bytes(3, dt, true) - u64::from(short), false, true, dt);
        let mut binding = session
            .allocate_device_binding_fixed_strides("x", "x", dt, vec![3, 128], vec![3, 64])
            .unwrap();
        let mut physical = vec![123f32; 3 * 128];
        for row in 0..3 {
            physical[row * 128..row * 128 + 64].fill(0.25);
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
        let mut identity = None;
        let capacity = banks(3, false, dt);
        let capacity_inputs = [("weights", &capacity[0]), ("routes", &capacity[1])];
        for t in [3, 1, 3] {
            binding.set_logical_shape(vec![t, 64]).unwrap();
            let b = banks(t, false, dt);
            let inputs = [("weights", &b[0]), ("routes", &b[1])];
            let report = session
                .prepare_with_device_bindings(&capacity_inputs, std::slice::from_mut(&mut binding))
                .unwrap();
            assert!(matches!(
                report,
                WorkspacePreparationReport::Deferred { .. }
            ));
            let result =
                session.run_with_device_bindings(&inputs, std::slice::from_mut(&mut binding));
            if short {
                assert!(result.is_err());
                assert_eq!(books.used(Tier::Host), 0);
                assert!(manager.snapshot().unwrap().allocations.is_empty());
                break;
            }
            let output = result.unwrap();
            check(output[0].as_ref().unwrap(), t, dt);
            let snapshot = manager.snapshot().unwrap();
            assert_eq!(snapshot.allocations.len(), 1);
            assert_eq!(
                snapshot.allocations[0].identity,
                *identity.get_or_insert(snapshot.allocations[0].identity)
            );
            assert_eq!(books.used(Tier::Host), bytes(3, dt, true));
        }
        drop(binding);
        drop(session);
        pool.trim_idle(u64::MAX).unwrap();
        assert_eq!(books.used(Tier::Host), 0);
    }
}
