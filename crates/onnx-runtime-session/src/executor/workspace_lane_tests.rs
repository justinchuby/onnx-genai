use super::*;

fn mixed_graph() -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("input", DataType::Float32, static_shape([2]));
    let middle = graph.create_named_value("middle", DataType::Float32, static_shape([2]));
    let output = graph.create_named_value("output", DataType::Float32, static_shape([2]));
    graph.add_input(input);
    graph.add_output(output);
    let mut persistent = Node::new(
        NodeId(0),
        "DeferredValidation",
        vec![Some(input)],
        vec![middle],
    );
    persistent
        .attributes
        .insert("persistent_workspace".into(), Attribute::Int(1));
    persistent
        .attributes
        .insert("workspace_scale".into(), Attribute::Int(2));
    graph.insert_node(persistent);
    graph.insert_node(Node::new(
        NodeId(1),
        "DeferredValidation",
        vec![Some(middle)],
        vec![output],
    ));
    graph
}

fn mixed_fixture(
    expose_registry: bool,
) -> (
    Executor,
    Arc<DeferredValidationEp>,
    Arc<WorkspaceCountingAuthority>,
    onnx_runtime_memory_governor::ProcessMemoryManager,
) {
    let (base, ep, books, manager) = completed_workspace_fixture(expose_registry);
    let executor =
        Executor::build(mixed_graph(), base.weights.clone(), base.ep.clone(), None).unwrap();
    drop(base);
    (executor, ep, books, manager)
}

#[test]
fn completed_lifetimes_share_one_idle_cell_and_pressure_keeps_the_unneeded_owner() {
    let (mut executor, ep, books, manager) = mixed_fixture(true);
    let input = completed_workspace_input(2);
    executor.run(&[("input", &input)]).unwrap();
    assert!(executor.workspaces.active().is_none());
    assert_eq!(books.used(Tier::Host), 192);
    let persistent = manager
        .snapshot()
        .unwrap()
        .allocations
        .into_iter()
        .find(|a| matches!(a.role, MemoryRole::Workspace { step_scoped: false }))
        .unwrap()
        .identity;
    let callbacks = books.callbacks();
    let locks = executor.workspaces.reuse.lock_count();
    for iteration in 1..=8 {
        executor.run(&[("input", &input)]).unwrap();
        assert!(executor.workspaces.active().is_none());
        assert_eq!(books.callbacks(), callbacks);
        assert_eq!(
            executor.workspaces.reuse.lock_count(),
            locks + 2 * iteration,
            "one atomic pair restore and one publish, never a per-node idle lookup"
        );
    }
    assert_eq!(ep.workspace_registry_calls.load(Ordering::Relaxed), 1);
    let registry = ep.workspace_registry.as_ref().unwrap();
    assert_eq!(
        registry.reclaim(1).unwrap(),
        64,
        "reclaim the smaller adequate whole physical owner"
    );
    assert_eq!(books.used(Tier::Host), 128);
    executor.run(&[("input", &input)]).unwrap();
    assert_eq!(
        ep.workspace_allocations.load(Ordering::Relaxed),
        3,
        "only the reclaimed Step owner needs admission again"
    );
    assert!(
        manager
            .snapshot()
            .unwrap()
            .allocations
            .iter()
            .any(|a| a.identity == persistent)
    );
    assert_eq!(
        registry.reclaim(128).unwrap(),
        128,
        "one adequate Persistent owner leaves Step warm"
    );
    assert_eq!(books.used(Tier::Host), 64);
    assert_eq!(registry.reclaim(1).unwrap(), 64);
    assert_eq!(books.used(Tier::Host), 0);
    assert_eq!(registry.reclaim(1).unwrap(), 0);
}

#[test]
fn active_deferred_pair_adds_no_idle_lock_issuer_consume_or_sync() {
    let (mut executor, ep, books, _) = mixed_fixture(true);
    ep.completion_observation_enabled
        .store(true, Ordering::Relaxed);
    executor
        .run(&[("input", &completed_workspace_input(2))])
        .unwrap();
    let mut bindings = vec![
        executor
            .allocate_device_binding("input".into(), None, DataType::Float32, vec![2], vec![2])
            .unwrap(),
        executor
            .allocate_device_output_binding("output".into(), DataType::Float32, vec![2], vec![2])
            .unwrap(),
    ];
    bindings[0]
        .write_bytes(0, completed_workspace_input(2).as_bytes())
        .unwrap();
    executor
        .run_with_device_bindings(&[], &mut bindings)
        .unwrap();
    let callbacks = books.callbacks();
    let locks = executor.workspaces.reuse.lock_count();
    assert!(
        locks > 0,
        "positive control: prior completed publication/restoration used the cell"
    );
    for _ in 0..8 {
        bindings[1].read_bytes().unwrap();
        let syncs = ep.sync_calls.load(Ordering::Relaxed);
        let consumes = ep.validation_consume_attempts.load(Ordering::Relaxed);
        executor
            .run_with_device_bindings(&[], &mut bindings)
            .unwrap();
        assert_eq!(executor.workspaces.reuse.lock_count(), locks);
        assert_eq!(books.callbacks(), callbacks);
        assert_eq!(ep.sync_calls.load(Ordering::Relaxed), syncs);
        assert_eq!(
            ep.validation_consume_attempts.load(Ordering::Relaxed),
            consumes
        );
    }
    bindings[1].read_bytes().unwrap();
    executor.finish_device_validation_boundary().unwrap();
    assert!(executor.workspaces.active().is_none());
    assert_eq!(executor.workspaces.reuse.lock_count(), locks + 1);
    assert_eq!(
        ep.workspace_registry
            .as_ref()
            .unwrap()
            .reclaim(192)
            .unwrap(),
        192
    );
    assert_eq!(books.used(Tier::Host), 0);
}

#[test]
fn any_lane_pin_keeps_both_completed_lifetimes_out_of_pressure() {
    let (mut executor, ep, books, _) = mixed_fixture(true);
    executor
        .run(&[("input", &completed_workspace_input(2))])
        .unwrap();
    executor.restore_completed_workspaces();
    let registry = ep.workspace_registry.as_ref().unwrap();
    executor.set_pin_workspace_lane(true);
    executor.release_workspaces().unwrap();
    assert_eq!(registry.reclaim(192).unwrap(), 0);
    assert_eq!(books.used(Tier::Host), 192);
    executor.set_pin_workspace_lane(false);
    executor.slot_capture[1].device_graph_token = Some(DeviceGraphToken::new(
        executor.graph_owner,
        DeviceGraphSlot::Verify,
        1,
    ));
    executor.release_workspaces().unwrap();
    assert_eq!(registry.reclaim(192).unwrap(), 0);
    executor.slot_capture[1].device_graph_token = None;
    executor.release_workspaces().unwrap();
    assert_eq!(registry.reclaim(192).unwrap(), 192);
    assert_eq!(books.used(Tier::Host), 0);
}

#[test]
fn missing_pressure_capability_preserves_persistent_and_old_step_retirement() {
    let (mut executor, ep, books, _) = mixed_fixture(false);
    for _ in 0..4 {
        executor
            .run(&[("input", &completed_workspace_input(2))])
            .unwrap();
    }
    let active = executor.workspaces.active().unwrap();
    assert!(active.persistent.is_some());
    assert!(active.step.is_none());
    assert!(matches!(
        executor.workspaces.reuse,
        WorkspaceReuse::Disabled
    ));
    assert_eq!(ep.workspace_registry_calls.load(Ordering::Relaxed), 1);
    assert_eq!(ep.workspace_allocations.load(Ordering::Relaxed), 5);
    assert_eq!(ep.workspace_releases.load(Ordering::Relaxed), 4);
    assert_eq!(
        books.used(Tier::Host),
        192,
        "existing provider pool retains Step charge separately from the active Persistent owner"
    );
}

#[test]
fn consuming_release_failure_never_refunds_or_reinserts_the_failed_owner() {
    let (mut executor, ep, books, _) = mixed_fixture(true);
    executor
        .run(&[("input", &completed_workspace_input(2))])
        .unwrap();
    let registry = ep.workspace_registry.as_ref().unwrap();
    ep.fail_workspace_release.store(true, Ordering::Relaxed);
    let error = registry.reclaim(1).unwrap_err().to_string();
    assert!(error.contains("workspace release failure"), "{error}");
    assert_eq!(books.used(Tier::Host), 192);
    ep.fail_workspace_release.store(false, Ordering::Relaxed);
    assert_eq!(
        registry.reclaim(1).unwrap(),
        128,
        "only the untouched Persistent owner remains idle"
    );
    assert_eq!(
        books.used(Tier::Host),
        64,
        "failed consuming Step release stays charged"
    );
    drop(executor);
    assert_eq!(ep.workspace_releases.load(Ordering::Relaxed), 2);
    assert_eq!(books.used(Tier::Host), 64);
}

#[test]
fn pair_restore_and_partial_pressure_preserve_exactly_one_owner_per_lifetime() {
    let (mut executor, ep, books, _) = mixed_fixture(true);
    for _ in 0..24 {
        executor
            .run(&[("input", &completed_workspace_input(2))])
            .unwrap();
        let reuse = &executor.workspaces.reuse;
        let registry = ep.workspace_registry.as_ref().unwrap();
        let (pair, reclaimed) = std::thread::scope(|scope| {
            let restore = scope.spawn(|| reuse.take());
            let pressure = scope.spawn(|| registry.reclaim(192).unwrap());
            (restore.join().unwrap(), pressure.join().unwrap())
        });
        let pair = pair.unwrap_or_default();
        let restored = pair
            .persistent
            .as_ref()
            .map_or(0, |owner| owner.bytes as u64)
            + pair.step.as_ref().map_or(0, |owner| owner.bytes as u64);
        assert_eq!(
            restored + reclaimed,
            192,
            "Host physical charges are exact in this fixture"
        );
        for owner in [pair.persistent, pair.step].into_iter().flatten() {
            ep.deallocate_workspace(owner.buffer).unwrap();
        }
        assert_eq!(books.used(Tier::Host), 0);
    }
}

#[test]
fn nested_control_flow_borrows_both_lifetimes_without_child_idle_registration() {
    let (base, ep, books, manager) = mixed_fixture(true);
    let mut body = mixed_graph();
    body.inputs.clear();
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("input", DataType::Float32, static_shape([2]));
    let cond = graph.create_named_value("cond", DataType::Bool, static_shape([]));
    let output = graph.create_named_value("output", DataType::Float32, static_shape([2]));
    graph.add_input(input);
    graph.add_input(cond);
    graph.add_output(output);
    graph.insert_node(Node::new(NodeId(0), "If", vec![Some(cond)], vec![output]));
    graph
        .subgraphs
        .insert((NodeId(0), "then_branch".into()), body.clone());
    graph
        .subgraphs
        .insert((NodeId(0), "else_branch".into()), body);
    let mut executor = Executor::build(graph, base.weights.clone(), base.ep.clone(), None).unwrap();
    drop(base);
    let input = completed_workspace_input(2);
    let cond = Tensor::from_raw(DataType::Bool, vec![], &[1]).unwrap();
    executor.run(&[("input", &input), ("cond", &cond)]).unwrap();
    let callbacks = books.callbacks();
    for _ in 0..6 {
        executor.run(&[("input", &input), ("cond", &cond)]).unwrap();
        assert_eq!(books.callbacks(), callbacks);
        assert_eq!(manager.snapshot().unwrap().allocations.len(), 2);
        for child in executor
            .subgraph_execs
            .values()
            .flat_map(|child| &child.compiled)
        {
            assert!(child.exec.workspaces.active().unwrap().is_empty());
            assert!(matches!(
                child.exec.workspaces.reuse,
                WorkspaceReuse::Uninitialized
            ));
        }
    }
    assert_eq!(ep.workspace_registry_calls.load(Ordering::Relaxed), 1);
    drop(executor);
    assert_eq!(books.used(Tier::Host), 0);
}

#[test]
fn pressure_never_detaches_a_third_owner_after_concurrent_republication() {
    let (mut executor, ep, books, _) = mixed_fixture(true);
    let input = completed_workspace_input(2);
    executor.run(&[("input", &input)]).unwrap();
    let entered = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    *ep.workspace_release_hook.lock().unwrap() = Some({
        let entered = entered.clone();
        let resume = resume.clone();
        Arc::new(move |release| {
            if release == 2 {
                entered.wait();
                resume.wait();
            }
        })
    });
    let registry = ep.workspace_registry.as_ref().unwrap();
    std::thread::scope(|scope| {
        let pressure = scope.spawn(|| registry.reclaim(193).unwrap());
        entered.wait();
        // Both old owners are detached, while the second release is outside
        // the cell lock. Restore and publish a newly funded complete pair.
        executor.run(&[("input", &input)]).unwrap();
        resume.wait();
        assert_eq!(pressure.join().unwrap(), 192);
    });
    *ep.workspace_release_hook.lock().unwrap() = None;
    assert_eq!(books.used(Tier::Host), 192);
    assert_eq!(registry.reclaim(192).unwrap(), 192);
    assert_eq!(books.used(Tier::Host), 0);
}

#[test]
fn failed_execution_never_publishes_either_lifetime_even_with_clean_device_flags() {
    for host_failure in [false, true] {
        let (mut executor, ep, books, _) = mixed_fixture(true);
        ep.host_failure_next.store(host_failure, Ordering::Relaxed);
        ep.fail_next.store(!host_failure, Ordering::Relaxed);
        let error = executor
            .run(&[("input", &completed_workspace_input(2))])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(if host_failure {
                "forced host kernel failure"
            } else {
                "0x40"
            }),
            "{error}"
        );
        if host_failure {
            assert_eq!(ep.validation_latch.load(Ordering::Relaxed), 0);
        }
        assert!(executor.workspace_completion == WorkspaceCompletion::Pending);
        assert!(executor.workspaces.active().unwrap().persistent.is_some());
        assert!(executor.workspaces.active().unwrap().step.is_none());
        assert_eq!(ep.workspace_registry_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            ep.workspace_registry
                .as_ref()
                .unwrap()
                .reclaim(192)
                .unwrap(),
            0
        );
        assert_eq!(books.used(Tier::Host), 128);
        drop(executor);
        assert_eq!(books.used(Tier::Host), 0);
    }
}

#[test]
fn failed_enrollment_retires_step_but_preserves_persistent_for_retry() {
    let (mut executor, ep, books, _) = mixed_fixture(true);
    ep.fail_workspace_registry.store(true, Ordering::Relaxed);
    let error = executor
        .run(&[("input", &completed_workspace_input(2))])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("forced pressure registry failure"),
        "{error}"
    );
    let pair = executor.workspaces.active().unwrap();
    assert!(pair.persistent.is_some());
    assert!(pair.step.is_none());
    assert_eq!(books.used(Tier::Host), 128);
    assert_eq!(ep.workspace_releases.load(Ordering::Relaxed), 1);
    ep.fail_workspace_registry.store(false, Ordering::Relaxed);
    executor
        .run(&[("input", &completed_workspace_input(2))])
        .unwrap();
    assert!(executor.workspaces.active().is_none());
    assert_eq!(
        ep.workspace_registry
            .as_ref()
            .unwrap()
            .reclaim(192)
            .unwrap(),
        192
    );
    assert_eq!(books.used(Tier::Host), 0);
}
