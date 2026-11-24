use super::*;

fn nested_workspace_graph() -> Graph {
    let mut graph = Graph::new();
    let input = graph.create_named_value("input", DataType::Float32, static_shape([2]));
    let output = graph.create_named_value("result", DataType::Float32, static_shape([2]));
    graph.insert_node(Node::new(
        NodeId(0),
        "DeferredValidation",
        vec![Some(input)],
        vec![output],
    ));
    graph.add_output(output);
    for _ in 0..2 {
        let mut enclosing = Graph::new();
        let condition = enclosing.create_named_value("condition", DataType::Bool, static_shape([]));
        let output = enclosing.create_named_value("result", DataType::Float32, static_shape([2]));
        enclosing.insert_node(Node::new(
            NodeId(0),
            "If",
            vec![Some(condition)],
            vec![output],
        ));
        enclosing.add_output(output);
        enclosing
            .subgraphs
            .insert((NodeId(0), "then_branch".into()), graph.clone());
        enclosing
            .subgraphs
            .insert((NodeId(0), "else_branch".into()), graph);
        graph = enclosing;
    }
    graph.opset_imports.insert(String::new(), 17);
    let condition = graph
        .values
        .iter()
        .find(|(_, value)| value.name.as_deref() == Some("condition"))
        .unwrap()
        .0;
    graph.add_input(condition);
    let input = graph.create_named_value("input", DataType::Float32, static_shape([2]));
    graph.add_input(input);
    graph
}

fn assert_children_returned_owners(executor: &Executor) -> usize {
    let mut count = 0;
    for child in executor
        .subgraph_execs
        .values()
        .flat_map(|child| &child.compiled)
    {
        count += 1;
        assert!(child.exec.persistent_workspace.is_none());
        assert!(child.exec.step_workspace.is_none());
        assert!(child.exec.parent_workspace_policy.is_none());
        assert!(matches!(
            child.exec.workspace_reuse,
            WorkspaceReuse::Uninitialized
        ));
        count += assert_children_returned_owners(&child.exec);
    }
    count
}

#[test]
fn nested_workspace_loans_grow_one_root_owner_and_warm_calls_skip_the_issuer() {
    let (base, ep, books, manager) = completed_workspace_fixture(true);
    let mut executor = Executor::build(
        nested_workspace_graph(),
        base.weights.clone(),
        base.ep.clone(),
        None,
    )
    .unwrap();
    drop(base);
    let condition = Tensor::from_raw(DataType::Bool, vec![], &[1]).unwrap();
    let input = completed_workspace_input(2);
    let inputs = [("condition", &condition), ("input", &input)];
    for multiplier in [32usize, 64] {
        ep.workspace_bytes.store(multiplier, Ordering::Relaxed);
        let output = executor.run(&inputs).unwrap();
        assert_eq!(output[0].as_bytes(), input.as_bytes());
        let callbacks = books.callbacks();
        let allocations = ep.workspace_allocations.load(Ordering::Relaxed);
        let identity = manager.snapshot().unwrap().allocations[0].identity;
        for _ in 0..4 {
            assert_eq!(
                executor.run(&inputs).unwrap()[0].as_bytes(),
                input.as_bytes()
            );
            assert_eq!(
                books.callbacks(),
                callbacks,
                "warm child loans must not negotiate quota"
            );
            assert_eq!(
                ep.workspace_allocations.load(Ordering::Relaxed),
                allocations
            );
            let owners = manager.snapshot().unwrap().allocations;
            assert_eq!(owners.len(), 1);
            assert_eq!(owners[0].identity, identity);
            assert_eq!(books.used(Tier::Host), (2 * multiplier) as u64);
            assert!(assert_children_returned_owners(&executor) >= 2);
        }
    }
    drop(executor);
    assert_eq!(books.used(Tier::Host), 0);
    assert_eq!(manager.process_used(Tier::Host), 0);
}

#[test]
fn nested_workspace_growth_obeys_the_root_secondary_capture_slot() {
    let (base, ep, books, _) = completed_workspace_fixture(true);
    let mut executor = Executor::build(
        nested_workspace_graph(),
        base.weights.clone(),
        base.ep.clone(),
        None,
    )
    .unwrap();
    drop(base);
    let condition = Tensor::from_raw(DataType::Bool, vec![], &[1]).unwrap();
    let input = completed_workspace_input(2);
    let inputs = [("condition", &condition), ("input", &input)];
    executor.run(&inputs).unwrap();
    executor.restore_completed_step_workspace();
    let origin = executor.step_workspace.as_ref().unwrap().buffer.as_ptr();
    let callbacks = books.callbacks();
    executor.slot_capture[1].device_graph_token = Some(DeviceGraphToken::new(
        executor.graph_owner,
        DeviceGraphSlot::Verify,
        1,
    ));
    ep.workspace_bytes.store(64, Ordering::Relaxed);
    let error = executor.run(&inputs).unwrap_err().to_string();
    assert!(error.contains("reset_device_graph"), "{error}");
    assert_eq!(
        executor.step_workspace.as_ref().unwrap().buffer.as_ptr(),
        origin
    );
    assert_eq!(books.callbacks(), callbacks);
    assert_children_returned_owners(&executor);
    executor.slot_capture[1].device_graph_token = None;
    executor.run(&inputs).unwrap();
    assert_eq!(books.used(Tier::Host), 128);
    drop(executor);
    assert_eq!(books.used(Tier::Host), 0);
}

#[test]
fn nested_workspace_replace_refusal_returns_final_slots_without_a_stale_address() {
    let (base, ep, books, _) = completed_workspace_fixture(true);
    let mut executor = Executor::build(
        nested_workspace_graph(),
        base.weights.clone(),
        base.ep.clone(),
        None,
    )
    .unwrap();
    drop(base);
    let condition = Tensor::from_raw(DataType::Bool, vec![], &[1]).unwrap();
    let input = completed_workspace_input(2);
    let inputs = [("condition", &condition), ("input", &input)];
    executor.run(&inputs).unwrap();
    let executions = ep.executions.load(Ordering::Relaxed);
    ep.workspace_bytes.store(4096, Ordering::Relaxed); // Request exceeds the 4096-byte authority.
    assert!(executor.run(&inputs).is_err());
    assert_eq!(ep.executions.load(Ordering::Relaxed), executions);
    assert!(executor.step_workspace.is_none());
    assert_children_returned_owners(&executor);
    ep.workspace_bytes.store(32, Ordering::Relaxed);
    assert_eq!(
        executor.run(&inputs).unwrap()[0].as_bytes(),
        input.as_bytes()
    );
    drop(executor);
    assert_eq!(books.used(Tier::Host), 0);
}

#[test]
fn workspace_peak_keeps_bytes_when_alignment_grows_in_prepare_and_dispatch() {
    for prepare_first in [true, false] {
        let (mut executor, ep, books, _) = completed_workspace_fixture(true);
        let role = MemoryRole::Workspace { step_scoped: true };
        let owner = ep.allocate_workspace(256, 4, role).unwrap();
        executor.step_workspace = Some(PreparedWorkspace {
            bytes: owner.len(),
            alignment: owner.alignment(),
            buffer: owner,
        });
        executor.pin_step_workspace = true;
        let input = completed_workspace_input(1);
        if prepare_first {
            executor
                .prepare_with_device_bindings(&[("input", &input)], &mut [])
                .unwrap();
        } else {
            executor.run(&[("input", &input)]).unwrap();
        }
        let owner = executor.step_workspace.as_ref().unwrap();
        assert!(owner.bytes >= 256);
        assert!(owner.alignment >= 64);
        let origin = owner.buffer.as_ptr();
        let callbacks = books.callbacks();
        executor
            .run(&[("input", &completed_workspace_input(8))])
            .unwrap();
        assert_eq!(
            executor.step_workspace.as_ref().unwrap().buffer.as_ptr(),
            origin
        );
        assert_eq!(books.callbacks(), callbacks);
        drop(executor);
        assert_eq!(books.used(Tier::Host), 0);
    }
}

#[test]
fn child_workspace_scope_returns_owners_on_planner_kernel_error_and_unwind() {
    for stage in [1, 2, 3, 4] {
        let (mut child, ep, _, _) = run_storage_fixture(stage);
        let mut parent = Executor::build(
            child.graph.clone(),
            child.weights.clone(),
            child.ep.clone(),
            None,
        )
        .unwrap();
        parent.workspace_completion = WorkspaceCompletion::Completed;
        let first = Tensor::from_f32(&[64], &[1.; 64]).unwrap();
        let second = Tensor::from_f32(&[64], &[2.; 64]).unwrap();
        let inputs = [("first", &first), ("second", &second)];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let workspace = ParentWorkspaces {
                persistent: &mut parent.persistent_workspace,
                step: &mut parent.step_workspace,
                completion: &mut parent.workspace_completion,
                replacement: WorkspaceReplacement::Allowed,
                provider: &parent.ep,
            };
            let mut loan = ChildWorkspaceLoan::new(&mut child, workspace).unwrap();
            let capture = loan
                .executor
                .workspace_capture_resources()
                .err()
                .unwrap()
                .to_string();
            assert!(capture.contains("parent executor"), "{capture}");
            loan.run(&inputs, &HashMap::new())
        }));
        if stage == 3 {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_err());
        }
        assert!(child.step_workspace.is_none());
        assert!(child.persistent_workspace.is_none());
        assert!(child.parent_workspace_policy.is_none());
        assert!(parent.workspace_completion == WorkspaceCompletion::Pending);
        assert_eq!(parent.step_workspace.is_some(), matches!(stage, 2 | 3));
        assert_eq!(ep.failure.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn child_workspace_scope_refuses_a_private_idle_owner_and_a_different_provider() {
    let (mut child, ep, books, _) = completed_workspace_fixture(true);
    child
        .run(&[("input", &completed_workspace_input(2))])
        .unwrap();
    assert!(child.step_workspace.is_none());
    let mut parent = Executor::build(
        child.graph.clone(),
        child.weights.clone(),
        child.ep.clone(),
        None,
    )
    .unwrap();
    let callbacks = books.callbacks();
    let workspace = ParentWorkspaces {
        persistent: &mut parent.persistent_workspace,
        step: &mut parent.step_workspace,
        completion: &mut parent.workspace_completion,
        replacement: WorkspaceReplacement::Allowed,
        provider: &parent.ep,
    };
    let error = ChildWorkspaceLoan::new(&mut child, workspace)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("independent lane"), "{error}");
    assert!(child.step_workspace.is_some());
    assert!(parent.step_workspace.is_none());
    assert_eq!(books.callbacks(), callbacks);
    let mut other = Executor::build(
        child.graph.clone(),
        child.weights.clone(),
        auto_detect_cpu_ep().unwrap(),
        None,
    )
    .unwrap();
    let workspace = ParentWorkspaces {
        persistent: &mut other.persistent_workspace,
        step: &mut other.step_workspace,
        completion: &mut other.workspace_completion,
        replacement: WorkspaceReplacement::Allowed,
        provider: &other.ep,
    };
    let error = ChildWorkspaceLoan::new(&mut child, workspace)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("identical provider instance"), "{error}");
    assert_eq!(ep.workspace_allocations.load(Ordering::Relaxed), 1);
    drop(parent);
    drop(child);
    assert_eq!(books.used(Tier::Host), 0);
}
