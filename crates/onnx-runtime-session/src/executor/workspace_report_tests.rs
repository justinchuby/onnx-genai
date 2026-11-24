use super::*;
use onnx_runtime_memory_api::DeviceKey;
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, ProcessMemoryManager,
};

fn mixed_report_fixture() -> (
    Executor,
    Arc<OutputWorkspaceProbeEp>,
    Arc<WorkspaceCountingAuthority>,
    HostWorkspacePool,
    ProcessMemoryManager,
) {
    let books = Arc::new(WorkspaceCountingAuthority(Arc::new(WorkspaceAccounting {
        inner: LedgerGovernor::new(LeaseLedger::new_for_device(DeviceKey::HOST, 0, 256, 0)),
        claims: AtomicUsize::new(0),
        returns: AtomicUsize::new(0),
    })));
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
    let resources = onnx_runtime_ep_cpu::CpuMemoryResources::new(
        onnx_runtime_ep_cpu::DerivedWeightMemory::governed(books.as_ref(), HolderId::new(31))
            .unwrap(),
        Default::default(),
    )
    .unwrap()
    .with_workspace_pool(pool.clone())
    .unwrap();
    let ep = Arc::new(OutputWorkspaceProbeEp {
        deferred: true,
        failure: Default::default(),
        cpu: CpuExecutionProvider::initialized_with_resources(resources).unwrap(),
        plans: Default::default(),
    });
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    for (id, count) in [(0, 3), (1, 8)] {
        let output = graph.create_named_value(
            format!("output_{id}"),
            DataType::Float32,
            static_shape([count]),
        );
        let mut node = Node::new(NodeId(id), "OutputWorkspaceProbe", vec![], vec![output]);
        if id == 0 {
            node.attributes
                .insert("workspace_known".into(), Attribute::Int(1));
        }
        graph.insert_node(node);
        graph.add_output(output);
    }
    let executor = Executor::build(graph, Arc::new(WeightStore::new()), ep.clone(), None).unwrap();
    (executor, ep, books, pool, manager)
}

#[test]
fn deferred_report_reserves_known_peak_without_publishing_complete_memo() {
    let (mut executor, _ep, books, pool, manager) = mixed_report_fixture();
    let report = executor.prepare_with_device_bindings(&[], &mut []).unwrap();
    assert_eq!(report.known().step().bytes, 24);
    assert_eq!(report.known().persistent().bytes, 0);
    assert!(
        matches!(&report, WorkspacePreparationReport::Deferred { pending, .. } if pending.len() == 1 && pending[0].contains("node #1"))
    );
    assert!(
        report
            .require_complete()
            .unwrap_err()
            .to_string()
            .contains("live tensor bindings")
    );
    assert!(executor.workspace_preparation_memo.is_none());
    assert_eq!(books.used(Tier::Host), 24);
    assert_eq!(manager.snapshot().unwrap().allocations.len(), 1);
    let callbacks = books.callbacks();
    executor.slot_capture[1].device_graph_token = Some(DeviceGraphToken::new(
        executor.graph_owner,
        DeviceGraphSlot::Verify,
        1,
    ));
    let repeated = executor.prepare_with_device_bindings(&[], &mut []).unwrap();
    assert!(matches!(
        repeated,
        WorkspacePreparationReport::Deferred { .. }
    ));
    assert_eq!(
        books.callbacks(),
        callbacks,
        "fitting known bounds must not negotiate quota"
    );
    assert!(
        executor.slot_capture[1].device_graph_token.is_some(),
        "a metadata deferral must not destroy an existing capture"
    );
    executor.slot_capture[1].device_graph_token = None;
    let outputs = executor.run(&[]).unwrap();
    assert_eq!(outputs[0].as_bytes(), &[0x5a; 12]);
    assert_eq!(outputs[1].as_bytes(), &[0x5a; 32]);
    assert!(
        executor.workspace_preparation_memo.is_none(),
        "one eager pass does not prove every deferred specialization complete"
    );
    drop(outputs);
    drop(executor);
    pool.trim_idle(256).unwrap();
    assert_eq!(books.used(Tier::Host), 0);
}

#[test]
fn recording_never_grows_an_unprepared_workspace_but_accepts_a_fitting_owner() {
    let (mut executor, ep, books, _manager) = completed_workspace_fixture(true);
    let input = completed_workspace_input(2);
    let external = ExternalBindings::default();
    let symbols = executor
        .bind_symbols(&[("input", &input)], &external)
        .unwrap();
    let mut resolved = executor.resolve_all(&symbols).unwrap();
    executor.ensure_provider_artifacts_ready(&resolved).unwrap();
    executor
        .prepare_run_storage(&external, &resolved, None)
        .unwrap();
    let value = executor.input_index["input"];
    ep.cpu
        .copy_from_host(input.as_bytes(), executor.buffers.get_mut(&value).unwrap())
        .unwrap();
    let callbacks = books.callbacks();
    let syncs = ep.sync_calls.load(Ordering::Relaxed);
    let error = executor
        .exec_plan_node(
            0,
            &mut resolved,
            &HashMap::new(),
            &external,
            OpCaptureTrace::Captured,
        )
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("while recording") && error.contains("warm"),
        "{error}"
    );
    assert_eq!(books.callbacks(), callbacks);
    assert_eq!(ep.workspace_allocations.load(Ordering::Relaxed), 0);
    assert_eq!(ep.sync_calls.load(Ordering::Relaxed), syncs);
    executor
        .prepare_with_device_bindings(&[("input", &input)], &mut [])
        .unwrap()
        .require_complete()
        .unwrap();
    let callbacks = books.callbacks();
    executor
        .exec_plan_node(
            0,
            &mut resolved,
            &HashMap::new(),
            &external,
            OpCaptureTrace::Captured,
        )
        .unwrap();
    assert_eq!(books.callbacks(), callbacks);
    assert_eq!(ep.workspace_allocations.load(Ordering::Relaxed), 1);
    drop(executor);
    assert_eq!(books.used(Tier::Host), 0);
}

#[test]
fn nested_unknown_rank_defers_instead_of_becoming_a_scalar_specialization() {
    for unknown_rank in [true, false] {
        let ep = Arc::new(OutputWorkspaceProbeEp {
            deferred: false,
            failure: Default::default(),
            cpu: CpuExecutionProvider::initialized_with_resources(Default::default()).unwrap(),
            plans: Default::default(),
        });
        let mut branch = Graph::new();
        let data = branch.create_named_value("data", DataType::Float32, static_shape([]));
        if unknown_rank {
            branch.mark_value_shape_unknown(data);
        }
        let output = branch.create_named_value("result", DataType::Float32, static_shape([3]));
        branch.insert_node(Node::new(
            NodeId(0),
            "OutputWorkspaceProbe",
            vec![Some(data)],
            vec![output],
        ));
        branch.add_output(output);
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let condition = graph.create_named_value("condition", DataType::Bool, static_shape([]));
        graph.add_input(condition);
        let data = graph.create_named_value("data", DataType::Float32, static_shape([]));
        if unknown_rank {
            graph.mark_value_shape_unknown(data);
        }
        graph.add_input(data);
        let output = graph.create_named_value("output", DataType::Float32, static_shape([3]));
        graph.insert_node(Node::new(
            NodeId(0),
            "If",
            vec![Some(condition)],
            vec![output],
        ));
        graph.add_output(output);
        graph
            .subgraphs
            .insert((NodeId(0), "then_branch".into()), branch.clone());
        graph
            .subgraphs
            .insert((NodeId(0), "else_branch".into()), branch);
        let mut executor =
            Executor::build(graph, Arc::new(WeightStore::new()), ep.clone(), None).unwrap();
        let condition = Tensor::from_raw(DataType::Bool, vec![], &[1]).unwrap();
        let data = Tensor::from_f32(&[], &[1.]).unwrap();
        let report = executor
            .prepare_with_device_bindings(&[("condition", &condition), ("data", &data)], &mut [])
            .unwrap();
        if unknown_rank {
            assert!(
                matches!(&report, WorkspacePreparationReport::Deferred { pending, .. } if pending.iter().any(|node| node.contains("then_branch")) && pending.iter().any(|node| node.contains("else_branch")))
            );
            assert!(report.require_complete().is_err());
            assert!(ep.plans.lock().unwrap().is_empty());
            assert!(executor.workspace_preparation_memo.is_none());
        } else {
            assert_eq!(report.require_complete().unwrap().step().bytes, 24);
            assert!(!ep.plans.lock().unwrap().is_empty());
            assert!(executor.workspace_preparation_memo.is_some());
        }
    }
}
