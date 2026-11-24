use super::*;

fn bindings(executor: &mut Executor) -> Vec<DeviceIoBinding> {
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
    bindings
}

#[test]
fn exact_completed_binding_reads_reuse_one_pending_lane_without_new_issuer_or_sync() {
    let (mut executor, ep, books, manager) = completed_workspace_fixture(true);
    ep.completion_observation_enabled
        .store(true, Ordering::Relaxed);
    let mut bindings = bindings(&mut executor);
    executor
        .prepare_with_device_bindings(&[], &mut bindings)
        .unwrap()
        .require_complete()
        .unwrap();
    let callbacks = books.callbacks();
    assert!(
        callbacks.0 > 0,
        "positive control: the original physical owner was charged"
    );
    let identity = manager.snapshot().unwrap().allocations[0].identity;
    for iteration in 0..9 {
        let syncs = ep.sync_calls.load(Ordering::Relaxed);
        let consumes = ep.validation_consume_attempts.load(Ordering::Relaxed);
        executor
            .run_with_device_bindings(&[], &mut bindings)
            .unwrap();
        assert_eq!(
            ep.sync_calls.load(Ordering::Relaxed),
            syncs,
            "reuse adds no synchronization"
        );
        assert_eq!(
            ep.validation_consume_attempts.load(Ordering::Relaxed),
            consumes,
            "poll never consumes or performs device validation"
        );
        assert!(executor.workspace_completion == WorkspaceCompletion::AwaitingValidation);
        assert!(executor.workspaces.active_mut().step.is_some());
        assert!(executor.workspaces.reuse.take().is_none());
        assert_eq!(
            ep.workspace_registry.as_ref().unwrap().reclaim(1).unwrap(),
            0,
            "pending ownership is not idle pressure capacity"
        );
        assert_eq!(books.callbacks(), callbacks);
        assert_eq!(
            manager.snapshot().unwrap().allocations[0].identity,
            identity
        );
        match iteration % 3 {
            0 => assert_eq!(
                bindings[1].read_bytes().unwrap(),
                completed_workspace_input(2).as_bytes()
            ),
            1 => assert_eq!(
                bindings[1].read_bytes_range(4, 4).unwrap(),
                3f32.to_le_bytes()
            ),
            _ => assert!(bindings[1].read_bytes_range(0, 0).unwrap().is_empty()),
        }
    }
    assert_eq!(ep.workspace_allocations.load(Ordering::Relaxed), 1);
    executor.release_workspaces().unwrap();
    assert!(executor.workspace_completion == WorkspaceCompletion::Completed);
    assert!(executor.workspaces.active().is_none());
    assert_eq!(
        ep.workspace_registry.as_ref().unwrap().reclaim(1).unwrap(),
        64
    );
    assert_eq!(books.used(Tier::Host), 0);
}

#[test]
fn pending_receipt_refuses_prepare_and_next_generation_without_consuming_or_overwriting() {
    let (mut executor, ep, books, _) = completed_workspace_fixture(true);
    ep.completion_observation_enabled
        .store(true, Ordering::Relaxed);
    let mut bindings = bindings(&mut executor);
    executor
        .run_with_device_bindings(&[], &mut bindings)
        .unwrap();
    let token = executor.pending_device_validation;
    let callbacks = books.callbacks();
    let resets = ep.resets.load(Ordering::Relaxed);
    let syncs = ep.sync_calls.load(Ordering::Relaxed);
    let consumes = ep.validation_consume_attempts.load(Ordering::Relaxed);
    let error = executor
        .prepare_with_device_bindings(&[], &mut bindings)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("still pending") && error.contains("read a bound output"),
        "{error}"
    );
    let error = executor
        .run_with_device_bindings(&[], &mut bindings)
        .unwrap_err()
        .to_string();
    assert!(error.contains("still pending"), "{error}");
    assert_eq!(executor.pending_device_validation, token);
    assert_eq!(ep.resets.load(Ordering::Relaxed), resets);
    assert_eq!(ep.sync_calls.load(Ordering::Relaxed), syncs);
    assert_eq!(
        ep.validation_consume_attempts.load(Ordering::Relaxed),
        consumes
    );
    assert_eq!(books.callbacks(), callbacks);
    bindings[1].read_bytes_range(0, 0).unwrap();
    executor
        .prepare_with_device_bindings(&[], &mut bindings)
        .unwrap()
        .require_complete()
        .unwrap();
    executor
        .run_with_device_bindings(&[], &mut bindings)
        .unwrap();
    assert_ne!(executor.pending_device_validation, token);
    assert_eq!(books.callbacks(), callbacks);
    bindings[1].read_bytes().unwrap();
}

#[test]
fn completed_failed_receipt_stays_failed_before_a_new_generation() {
    let (mut executor, ep, books, _) = completed_workspace_fixture(true);
    ep.completion_observation_enabled
        .store(true, Ordering::Relaxed);
    ep.fail_next.store(true, Ordering::Relaxed);
    let mut bindings = bindings(&mut executor);
    executor
        .run_with_device_bindings(&[], &mut bindings)
        .unwrap();
    assert!(
        bindings[1]
            .read_bytes()
            .unwrap_err()
            .to_string()
            .contains("0x40")
    );
    let token = executor.pending_device_validation;
    let resets = ep.resets.load(Ordering::Relaxed);
    for _ in 0..2 {
        let error = executor
            .run_with_device_bindings(&[], &mut bindings)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("0x40") && error.contains("failed request boundary"),
            "{error}"
        );
        assert_eq!(executor.pending_device_validation, token);
    }
    assert_eq!(ep.resets.load(Ordering::Relaxed), resets);
    assert_eq!(books.used(Tier::Host), 64);
    assert_eq!(
        ep.workspace_registry.as_ref().unwrap().reclaim(1).unwrap(),
        0
    );
    assert!(
        executor
            .finish_device_validation_boundary()
            .unwrap_err()
            .to_string()
            .contains("0x40")
    );
    assert!(executor.workspace_completion == WorkspaceCompletion::Pending);
    executor.release_workspaces().unwrap();
    assert_eq!(books.used(Tier::Host), 0);
    assert!(executor.workspaces.reuse.take().is_none());
}

#[test]
fn host_failure_with_zero_device_flags_cannot_publish_reusable_workspace() {
    let (mut executor, ep, books, _) = completed_workspace_fixture(true);
    ep.completion_observation_enabled
        .store(true, Ordering::Relaxed);
    ep.host_failure_next.store(true, Ordering::Relaxed);
    let mut bindings = bindings(&mut executor);
    let error = executor
        .run_with_device_bindings(&[], &mut bindings)
        .unwrap_err()
        .to_string();
    assert!(error.contains("forced host kernel failure"), "{error}");
    assert_eq!(ep.validation_latch.load(Ordering::Relaxed), 0);
    assert_eq!(
        ep.completion_observations.load(Ordering::Relaxed),
        0,
        "unknown/error work must not be upgraded by a clean receipt"
    );
    assert!(executor.workspace_completion == WorkspaceCompletion::Pending);
    assert!(executor.workspaces.reuse.take().is_none());
    assert_eq!(ep.workspace_releases.load(Ordering::Relaxed), 1);
    assert_eq!(books.used(Tier::Host), 0);
}

#[test]
fn capture_pin_and_parent_loan_do_not_enroll_pending_capacity_as_idle() {
    let (mut executor, ep, books, _) = completed_workspace_fixture(true);
    ep.completion_observation_enabled
        .store(true, Ordering::Relaxed);
    executor.set_pin_workspace_lane(true);
    let mut bindings = bindings(&mut executor);
    executor
        .run_with_device_bindings(&[], &mut bindings)
        .unwrap();
    assert_eq!(
        ep.completion_observations.load(Ordering::Relaxed),
        0,
        "pinned release retains active ownership without polling"
    );
    executor.set_pin_workspace_lane(false);
    executor.parent_workspace_policy = Some(WorkspaceReplacement::Allowed);
    executor.release_workspaces().unwrap();
    assert_eq!(
        ep.completion_observations.load(Ordering::Relaxed),
        0,
        "a loaned child cannot observe/idle the root's receipt"
    );
    assert_eq!(
        ep.workspace_registry.as_ref().unwrap().reclaim(1).unwrap(),
        0
    );
    executor.parent_workspace_policy = None;
    bindings[1].read_bytes().unwrap();
    executor.release_workspaces().unwrap();
    assert_eq!(
        ep.workspace_registry.as_ref().unwrap().reclaim(1).unwrap(),
        64
    );
    assert_eq!(books.used(Tier::Host), 0);
}

#[test]
fn completion_from_another_executor_cannot_idle_or_reuse_the_pending_owner() {
    let (mut first, ep, books, _) = completed_workspace_fixture(true);
    ep.completion_observation_enabled
        .store(true, Ordering::Relaxed);
    let mut second = Executor::build(
        first.graph.clone(),
        Arc::new(WeightStore::new()),
        ep.clone(),
        None,
    )
    .unwrap();
    let mut first_bindings = bindings(&mut first);
    let mut second_bindings = bindings(&mut second);
    second
        .run_with_device_bindings(&[], &mut second_bindings)
        .unwrap();
    second_bindings[1].read_bytes().unwrap();
    first
        .run_with_device_bindings(&[], &mut first_bindings)
        .unwrap();
    let first_token = first.pending_device_validation;
    // A stale completed output read may synchronize, but carries no receipt for
    // the currently pending first executor's validation generation.
    second_bindings[1].read_bytes().unwrap();
    second.release_workspaces().unwrap();
    assert_eq!(
        ep.workspace_registry.as_ref().unwrap().reclaim(1).unwrap(),
        64
    );
    assert_eq!(books.used(Tier::Host), 64);
    let error = first
        .run_with_device_bindings(&[], &mut first_bindings)
        .unwrap_err()
        .to_string();
    assert!(error.contains("still pending"), "{error}");
    assert_eq!(first.pending_device_validation, first_token);
    assert_eq!(
        ep.workspace_registry.as_ref().unwrap().reclaim(1).unwrap(),
        0
    );
    first_bindings[1].read_bytes().unwrap();
    first.release_workspaces().unwrap();
    assert_eq!(
        ep.workspace_registry.as_ref().unwrap().reclaim(1).unwrap(),
        64
    );
    assert_eq!(books.used(Tier::Host), 0);
}

#[test]
fn no_step_lane_still_preserves_pending_and_failed_generation_receipts() {
    for fail in [false, true] {
        let (mut executor, ep, books, _) = completed_workspace_fixture(true);
        ep.workspace_bytes.store(0, Ordering::Relaxed);
        ep.completion_observation_enabled
            .store(true, Ordering::Relaxed);
        ep.fail_next.store(fail, Ordering::Relaxed);
        let mut bindings = bindings(&mut executor);
        executor
            .run_with_device_bindings(&[], &mut bindings)
            .unwrap();
        assert!(executor.workspaces.active_mut().step.is_none());
        let token = executor.pending_device_validation;
        let resets = ep.resets.load(Ordering::Relaxed);
        let error = executor
            .run_with_device_bindings(&[], &mut bindings)
            .unwrap_err()
            .to_string();
        assert!(error.contains("still pending"), "{error}");
        assert_eq!(ep.resets.load(Ordering::Relaxed), resets);
        let read = bindings[1].read_bytes();
        let completed_resets = ep.resets.load(Ordering::Relaxed);
        if fail {
            assert!(read.unwrap_err().to_string().contains("0x40"));
            let error = executor
                .prepare_with_device_bindings(&[], &mut bindings)
                .unwrap_err()
                .to_string();
            assert!(error.contains("0x40"), "{error}");
            let error = executor
                .run_with_device_bindings(&[], &mut bindings)
                .unwrap_err()
                .to_string();
            assert!(error.contains("0x40"), "{error}");
            assert_eq!(ep.resets.load(Ordering::Relaxed), completed_resets);
            assert_eq!(executor.pending_device_validation, token);
        } else {
            read.unwrap();
            executor
                .run_with_device_bindings(&[], &mut bindings)
                .unwrap();
            assert_ne!(executor.pending_device_validation, token);
            bindings[1].read_bytes().unwrap();
        }
        assert_eq!(books.used(Tier::Host), 0);
        assert_eq!(ep.workspace_allocations.load(Ordering::Relaxed), 0);
        assert_eq!(ep.workspace_registry_calls.load(Ordering::Relaxed), 0);
    }
}
