use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use onnx_runtime_ep_api::{
    ExecutionProvider, ExecutorArtifactFinalization, ExecutorArtifactPending,
    ExecutorArtifactReadinessEpoch, ExecutorInstanceId, ExternalMmapRegion, FinalizedExpertBank,
    FinalizedExpertWeight, LazyWeight, LazyWeightBoundary, ResidentWeight, expert_weight_groups,
};
use onnx_runtime_ep_cuda::coarse_residency::COARSE_RESIDENCY_ENABLE_ENV;
use onnx_runtime_ep_cuda::route_residency::{
    RouteResidencyBindingReject, RouteResidencyInstallOutcome,
};
use onnx_runtime_ep_cuda::weight_paging::{DeviceOffloadPolicy, RouteBankReservationReject};
use onnx_runtime_ep_cuda::{CudaExecutionProvider, RouteFinalizationCommitInterlock};
use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, ValueId, WeightRef, static_shape};
use onnx_runtime_loader::{ExpertStorageOrder, ExpertTensorLayout, WeightRegionCatalog};
use onnx_runtime_memory_governor::{DeviceKey, LeaseLedger, LedgerGovernor, MemoryGovernor};

const EXPERTS: usize = 4;
const EXPERT_BYTES: usize = 2 << 20;
const TENSOR_BYTES: usize = EXPERTS * EXPERT_BYTES;
static SERIAL: Mutex<()> = Mutex::new(());

struct GateGuard(Option<String>);

impl GateGuard {
    fn set(enabled: bool) -> Self {
        let previous = std::env::var(COARSE_RESIDENCY_ENABLE_ENV).ok();
        if enabled {
            unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };
        } else {
            unsafe { std::env::remove_var(COARSE_RESIDENCY_ENABLE_ENV) };
        }
        Self(previous)
    }
}

impl Drop for GateGuard {
    fn drop(&mut self) {
        match self.0.take() {
            Some(value) => unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, value) },
            None => unsafe { std::env::remove_var(COARSE_RESIDENCY_ENABLE_ENV) },
        }
    }
}

fn provider_or_skip(label: &str) -> Option<CudaExecutionProvider> {
    let ledger = LeaseLedger::new_for_device(DeviceKey::device(0), 1 << 30, 1 << 30, 0);
    let governor: Arc<dyn MemoryGovernor + Send + Sync> = Arc::new(LedgerGovernor::new(ledger));
    let policy = DeviceOffloadPolicy {
        enabled: true,
        device_budget_bytes: Some(64 << 20),
        ..DeviceOffloadPolicy::default()
    };
    match CudaExecutionProvider::initialized_with_offload_policy_and_governor(0, policy, governor) {
        Ok(provider) => Some(provider),
        Err(error) => {
            eprintln!("SKIP {label}: {error}");
            None
        }
    }
}

fn external_initializer(graph: &mut Graph, name: &str, dtype: DataType, offset: usize) -> ValueId {
    let storage_elements = match dtype {
        DataType::Uint8 => EXPERT_BYTES,
        DataType::Float32 => EXPERT_BYTES / 4,
        _ => unreachable!(),
    };
    let value = graph.create_named_value(name, dtype, static_shape([EXPERTS, 1, storage_elements]));
    graph.set_initializer(
        value,
        WeightRef::External {
            path: PathBuf::from("route-bank.bin"),
            offset,
            length: TENSOR_BYTES,
            dtype,
            dims: vec![EXPERTS, 1, storage_elements],
        },
    );
    value
}

fn qmoe_graph_and_bank(mapping_id: usize) -> (Graph, NodeId, FinalizedExpertBank) {
    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let input = graph.create_named_value("hidden", DataType::Float32, static_shape([1, 4]));
    let router = graph.create_named_value(
        "router_probs",
        DataType::Float32,
        static_shape([1, EXPERTS]),
    );
    let fc1_w = external_initializer(&mut graph, "fc1_w", DataType::Uint8, 0);
    let fc1_s = external_initializer(&mut graph, "fc1_s", DataType::Float32, TENSOR_BYTES);
    let fc2_w = external_initializer(&mut graph, "fc2_w", DataType::Uint8, 2 * TENSOR_BYTES);
    let fc2_s = external_initializer(&mut graph, "fc2_s", DataType::Float32, 3 * TENSOR_BYTES);
    let output = graph.create_named_value("output", DataType::Float32, static_shape([1, 4]));
    let mut node = Node::new(
        NodeId(0),
        "QMoE",
        vec![
            Some(input),
            Some(router),
            Some(fc1_w),
            Some(fc1_s),
            None,
            Some(fc2_w),
            Some(fc2_s),
            None,
            None,
            None,
        ],
        vec![output],
    );
    node.domain = "com.microsoft".into();
    for (name, value) in [
        ("expert_weight_bits", Attribute::Int(4)),
        ("block_size", Attribute::Int(16)),
        ("k", Attribute::Int(1)),
        ("activation_type", Attribute::String(b"silu".to_vec())),
        ("normalize_routing_weights", Attribute::Int(0)),
        ("swiglu_fusion", Attribute::Int(0)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    let node_id = graph.insert_node(node);
    let group = expert_weight_groups(&graph)
        .into_iter()
        .next()
        .expect("QMoE group");
    let members = group
        .members
        .iter()
        .map(|&value| {
            let weight_ref = graph.initializers[&value].clone();
            let (path, offset, dtype, shape) = match &weight_ref {
                WeightRef::External {
                    path,
                    offset,
                    dtype,
                    dims,
                    ..
                } => (path.clone(), *offset, *dtype, dims.clone()),
                WeightRef::Inline(_) => unreachable!(),
            };
            let layout = ExpertTensorLayout {
                version: 1,
                experts: EXPERTS,
                rows_per_expert: 1,
                storage_elements_per_row: shape[2],
                order: ExpertStorageOrder::ExpertMajor,
                quantization: None,
            };
            let catalog = WeightRegionCatalog::classify(&weight_ref, layout);
            assert!(catalog.is_pageable());
            let bytes: Arc<[u8]> = vec![value.0 as u8 + 1; TENSOR_BYTES].into();
            let resident_shape = shape.clone();
            let lazy = LazyWeight::new(
                LazyWeightBoundary::QMoe,
                dtype,
                shape,
                vec![ExternalMmapRegion {
                    mapping_id,
                    offset,
                    len: TENSOR_BYTES,
                }],
                move || ResidentWeight::new(dtype, resident_shape.clone(), Arc::clone(&bytes)),
            )
            .expect("lazy weight");
            FinalizedExpertWeight {
                value,
                external_path: path,
                weight: lazy,
                catalog,
            }
        })
        .collect();
    (graph, node_id, FinalizedExpertBank { group, members })
}

fn compile_real_qmoe(
    provider: &CudaExecutionProvider,
    executor: ExecutorInstanceId,
    graph: &Graph,
    node: NodeId,
) -> Box<dyn onnx_runtime_ep_api::Kernel> {
    provider
        .get_kernel_for_executor(executor, graph.node(node), &[], 1)
        .expect("compile real QMoE producer")
}

#[test]
#[ignore = "requires idle CUDA device with HOST_NUMA VMM support"]
fn real_producer_installs_executor_scoped_banks_once() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let _gate = GateGuard::set(true);
    let Some(provider) = provider_or_skip("install") else {
        return;
    };
    let provider = Arc::new(provider);
    let (graph, node, bank) = qmoe_graph_and_bank(11);
    let first = ExecutorInstanceId::fresh();
    let second = ExecutorInstanceId::fresh();
    let _first_kernel = compile_real_qmoe(&provider, first, &graph, node);
    let _second_kernel = compile_real_qmoe(&provider, second, &graph, node);

    assert_eq!(
        provider
            .finalize_executor_artifacts(
                first,
                &graph,
                ExecutorArtifactReadinessEpoch::new(1),
                std::slice::from_ref(&bank),
            )
            .expect("finalize first executor"),
        ExecutorArtifactFinalization::Complete
    );
    assert_eq!(
        provider
            .finalize_executor_artifacts(
                second,
                &graph,
                ExecutorArtifactReadinessEpoch::new(1),
                std::slice::from_ref(&bank),
            )
            .expect("finalize second executor"),
        ExecutorArtifactFinalization::Complete
    );
    for executor in [first, second] {
        let status = provider.route_residency_executor_status(executor);
        assert_eq!(status.finalization_attempts, 1);
        assert_eq!(
            status.outcome,
            Some(RouteResidencyInstallOutcome::Installed { banks: 4 })
        );
    }
    assert_eq!(
        provider
            .residency()
            .expect("residency")
            .route_reservation_count(),
        2
    );
    let first_ranges: Vec<_> = bank
        .group
        .members
        .iter()
        .map(|&value| {
            provider
                .residency()
                .unwrap()
                .coarse_route_bank_reservation(first, value)
                .unwrap()
                .with_reservation_mut(|reservation, _| reservation.base_ptr())
        })
        .collect();
    let second_ranges: Vec<_> = bank
        .group
        .members
        .iter()
        .map(|&value| {
            provider
                .residency()
                .unwrap()
                .coarse_route_bank_reservation(second, value)
                .unwrap()
                .with_reservation_mut(|reservation, _| reservation.base_ptr())
        })
        .collect();
    assert!(
        first_ranges
            .iter()
            .all(|first| !second_ranges.contains(first)),
        "sibling executors own distinct stable addresses"
    );
    let first_requirement = provider
        .executor_artifact_requirement(first)
        .expect("query first executor requirement")
        .expect("installed reservations publish an exact requirement");

    let _specialization = provider
        .get_kernel_for_executor(first, graph.node(node), &[vec![2, 4]], 1)
        .expect("dynamic specialization");
    assert_eq!(
        provider
            .finalize_executor_artifacts(
                first,
                &graph,
                ExecutorArtifactReadinessEpoch::new(2),
                std::slice::from_ref(&bank),
            )
            .expect("finalize later specialization"),
        ExecutorArtifactFinalization::Complete
    );
    assert_eq!(
        provider
            .route_residency_executor_status(first)
            .finalization_attempts,
        1
    );

    let holder = first_requirement
        .acquire_use()
        .expect("acquire pre-retirement use lease");
    let releases_before = provider.deferred_release_stats();
    let drain_provider = Arc::clone(&provider);
    let (returned_tx, returned_rx) = std::sync::mpsc::channel();
    let drain = std::thread::spawn(move || {
        drain_provider.drain_executor_artifacts(first);
        returned_tx.send(()).unwrap();
    });
    returned_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("public drain must return without waiting for the held requirement guard");
    drain.join().unwrap();
    let replay_rejection = first_requirement
        .acquire_use()
        .err()
        .expect("retiring requirement rejects replay");
    assert!(replay_rejection.to_string().contains("retiring"));
    assert_eq!(
        provider.residency().unwrap().route_reservation_count(),
        2,
        "the deferred authority must keep the first executor's mappings owned while held"
    );
    assert_eq!(
        provider
            .route_residency_executor_status(first)
            .reservation_removals,
        0
    );
    let deferred = provider.route_residency_retirement_census();
    assert_eq!(deferred.active_registry_entries, 1);
    assert_eq!(deferred.retirement_registry_entries, 1);
    assert_eq!(deferred.live_retirement_records, 1);
    assert_eq!(deferred.reservation_registry_entries, 2);
    assert_eq!(deferred.retirements_started, 1);
    assert_eq!(deferred.deferred_cleanups, 1);
    assert_eq!(deferred.cleanups_scheduled, 0);
    assert_eq!(deferred.cleanups_executed, 0);
    drop(holder);
    assert!(
        provider
            .release_queue()
            .wait_until_idle(std::time::Duration::from_secs(30)),
        "last guard release must enqueue and complete deferred cleanup: {:?}",
        provider.deferred_release_stats()
    );
    provider.drain_executor_artifacts(first);
    assert_eq!(provider.residency().unwrap().route_reservation_count(), 1);
    assert!(provider.route_residency_scopes().contains(&second));
    assert_eq!(
        provider.route_residency_executor_status(first).drain_calls,
        1
    );
    assert_eq!(
        provider
            .route_residency_executor_status(first)
            .reservation_removals,
        1
    );
    let releases_after = provider.deferred_release_stats();
    assert!(
        releases_after.completed >= releases_before.completed + 5,
        "one cleanup action plus four reservation unmaps must complete: before={releases_before:?} \
         after={releases_after:?}"
    );
    assert!(
        releases_after.mapped_refunded_bytes
            >= releases_before.mapped_refunded_bytes + 4 * TENSOR_BYTES as u64,
        "all four exact bank reservations must report their unmapped bytes"
    );
    let completed = provider.route_residency_retirement_census();
    assert_eq!(completed.cleanups_scheduled, 1);
    assert_eq!(completed.cleanups_executed, 1);
    let retired = first_requirement
        .acquire_use()
        .err()
        .expect("requirement survives registry removal as retired");
    assert!(retired.to_string().contains("retired"));
    drop(first_requirement);
    assert_eq!(
        provider
            .route_residency_retirement_census()
            .retirement_registry_entries,
        0,
        "dead generation tombstones are pruned once no graph/requirement/lease can reference them"
    );
    provider.drain_executor_artifacts(second);
}

#[test]
#[ignore = "requires idle CUDA device with HOST_NUMA VMM support"]
fn repeated_retirement_invalidates_admitted_finalizer_and_rolls_back_once_per_epoch() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let _gate = GateGuard::set(true);
    let Some(provider) = provider_or_skip("retire before finalization commit") else {
        return;
    };
    let provider = Arc::new(provider);
    let (graph, node, bank) = qmoe_graph_and_bank(13);
    let executor = ExecutorInstanceId::fresh();
    let rollbacks_before = provider
        .route_residency_retirement_census()
        .prepared_rollbacks;

    for cycle in 0..4 {
        let kernel = compile_real_qmoe(&provider, executor, &graph, node);
        let interlock = Arc::new(RouteFinalizationCommitInterlock::new());
        std::thread::scope(|scope| {
            let finalize_provider = Arc::clone(&provider);
            let finalize_interlock = Arc::clone(&interlock);
            let finalize_graph = &graph;
            let finalize_bank = &bank;
            let finalize = scope.spawn(move || {
                finalize_provider.finalize_route_residency_for_executor_with_commit_interlock(
                    executor,
                    finalize_graph,
                    ExecutorArtifactReadinessEpoch::new(cycle + 1),
                    std::slice::from_ref(finalize_bank),
                    &finalize_interlock,
                )
            });
            interlock.wait_until_admitted();
            assert_eq!(
                provider.residency().unwrap().route_reservation_count(),
                1,
                "cycle {cycle} must pause after a real reservation was prepared"
            );

            provider.drain_executor_artifacts(executor);
            interlock.resume_commit();
            let error = finalize
                .join()
                .expect("finalizer thread")
                .expect_err("retirement must invalidate the admitted commit");
            assert!(error.to_string().contains("invalidated before commit"));
        });
        drop(kernel);

        assert_eq!(
            provider.residency().unwrap().route_reservation_count(),
            0,
            "cycle {cycle} must remove the rejected preparation exactly once"
        );
        let census = provider.route_residency_retirement_census();
        assert_eq!(census.active_registry_entries, 0, "cycle {cycle}");
        assert_eq!(census.retirement_registry_entries, 0, "cycle {cycle}");
        assert_eq!(census.reservation_registry_entries, 0, "cycle {cycle}");
        assert_eq!(
            census.prepared_rollbacks,
            rollbacks_before + cycle + 1,
            "cycle {cycle} must transfer rollback ownership exactly once"
        );
    }

    let replacement_kernel = compile_real_qmoe(&provider, executor, &graph, node);
    assert_eq!(
        provider
            .finalize_executor_artifacts(
                executor,
                &graph,
                ExecutorArtifactReadinessEpoch::new(5),
                std::slice::from_ref(&bank),
            )
            .expect("replacement after stale rollback"),
        ExecutorArtifactFinalization::Complete
    );
    assert_eq!(provider.residency().unwrap().route_reservation_count(), 1);
    provider.drain_executor_artifacts(executor);
    drop(replacement_kernel);
}

#[test]
#[ignore = "requires idle CUDA device with HOST_NUMA VMM support"]
fn commit_authority_admits_exactly_one_concurrent_finalizer() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let _gate = GateGuard::set(true);
    let Some(provider) = provider_or_skip("commit before retirement") else {
        return;
    };
    let provider = Arc::new(provider);
    let (graph, node, bank) = qmoe_graph_and_bank(15);
    let executor = ExecutorInstanceId::fresh();
    let _kernel = compile_real_qmoe(&provider, executor, &graph, node);
    let interlock = Arc::new(RouteFinalizationCommitInterlock::new());

    std::thread::scope(|scope| {
        let finalize_provider = Arc::clone(&provider);
        let finalize_interlock = Arc::clone(&interlock);
        let finalize_graph = &graph;
        let finalize_bank = &bank;
        let finalize = scope.spawn(move || {
            finalize_provider.finalize_route_residency_for_executor_with_commit_interlock(
                executor,
                finalize_graph,
                ExecutorArtifactReadinessEpoch::new(1),
                std::slice::from_ref(finalize_bank),
                &finalize_interlock,
            )
        });
        interlock.wait_until_admitted();
        let sibling = provider
            .finalize_executor_artifacts(
                executor,
                &graph,
                ExecutorArtifactReadinessEpoch::new(1),
                std::slice::from_ref(&bank),
            )
            .expect("concurrent finalizer observes admission");
        assert!(matches!(
            sibling,
            ExecutorArtifactFinalization::Pending(
                ExecutorArtifactPending::ProviderReadiness { .. }
            )
        ));
        assert_eq!(provider.residency().unwrap().route_reservation_count(), 1);

        interlock.resume_commit();
        assert_eq!(
            finalize
                .join()
                .expect("finalizer thread")
                .expect("admitted finalizer commits"),
            ExecutorArtifactFinalization::Complete
        );
    });

    let status = provider.route_residency_executor_status(executor);
    assert_eq!(status.finalization_attempts, 1);
    assert!(matches!(
        status.outcome,
        Some(RouteResidencyInstallOutcome::Installed { banks: 4 })
    ));
    assert_eq!(provider.residency().unwrap().route_reservation_count(), 1);
    provider.drain_executor_artifacts(executor);
}

#[test]
#[ignore = "requires idle CUDA device with HOST_NUMA VMM support"]
fn retirement_registry_reclaims_churn_and_blocks_live_generation_aba() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let _gate = GateGuard::set(true);
    let Some(provider) = provider_or_skip("retirement churn") else {
        return;
    };
    let provider = Arc::new(provider);
    let (graph, node, bank) = qmoe_graph_and_bank(17);
    let reused = ExecutorInstanceId::fresh();
    let mut first_generation = None;

    for cycle in 0..8 {
        let executor = if cycle == 0 {
            reused
        } else {
            ExecutorInstanceId::fresh()
        };
        let kernel = compile_real_qmoe(&provider, executor, &graph, node);
        assert_eq!(
            provider
                .finalize_executor_artifacts(
                    executor,
                    &graph,
                    ExecutorArtifactReadinessEpoch::new(1),
                    std::slice::from_ref(&bank),
                )
                .expect("finalize churn executor"),
            ExecutorArtifactFinalization::Complete
        );
        let requirement = provider
            .executor_artifact_requirement(executor)
            .expect("query churn requirement")
            .expect("installed generation");
        let generation = provider
            .route_residency_executor_status(executor)
            .reservation_generation
            .expect("generation");
        first_generation.get_or_insert(generation);
        provider.drain_executor_artifacts(executor);
        assert!(
            provider
                .release_queue()
                .wait_until_idle(std::time::Duration::from_secs(30)),
            "cycle {cycle} cleanup must drain: {:?}",
            provider.deferred_release_stats()
        );
        if cycle == 0 {
            let error = provider
                .finalize_executor_artifacts(
                    executor,
                    &graph,
                    ExecutorArtifactReadinessEpoch::new(2),
                    std::slice::from_ref(&bank),
                )
                .expect_err("a live baked requirement must block executor-id reuse");
            assert!(error.to_string().contains("identities cannot be reused"));
            assert_eq!(
                provider
                    .route_residency_retirement_census()
                    .retirement_registry_entries,
                1
            );
        }
        drop(requirement);
        drop(kernel);
        let census = provider.route_residency_retirement_census();
        assert_eq!(census.active_registry_entries, 0, "cycle {cycle}");
        assert_eq!(census.retirement_registry_entries, 0, "cycle {cycle}");
        assert_eq!(census.live_retirement_records, 0, "cycle {cycle}");
        assert_eq!(census.reservation_registry_entries, 0, "cycle {cycle}");
    }

    let replacement_kernel = compile_real_qmoe(&provider, reused, &graph, node);
    assert_eq!(
        provider
            .finalize_executor_artifacts(
                reused,
                &graph,
                ExecutorArtifactReadinessEpoch::new(3),
                std::slice::from_ref(&bank),
            )
            .expect("reuse is safe only after the old generation has no references"),
        ExecutorArtifactFinalization::Complete
    );
    let replacement_generation = provider
        .route_residency_executor_status(reused)
        .reservation_generation
        .expect("replacement generation");
    assert_ne!(replacement_generation, first_generation.unwrap());
    let replacement_requirement = provider
        .executor_artifact_requirement(reused)
        .expect("replacement requirement")
        .expect("replacement installed");
    provider.drain_executor_artifacts(reused);
    drop(replacement_requirement);
    drop(replacement_kernel);
    assert!(
        provider
            .release_queue()
            .wait_until_idle(std::time::Duration::from_secs(30))
    );
    let census = provider.route_residency_retirement_census();
    assert_eq!(census.retirement_registry_entries, 0);
    assert_eq!(census.reservation_registry_entries, 0);
    assert_eq!(census.retirements_started, 9);
    assert_eq!(census.cleanups_scheduled, 9);
    assert_eq!(census.cleanups_executed, 9);
}

#[test]
#[ignore = "requires idle CUDA device with HOST_NUMA VMM support"]
fn readiness_absence_is_pending_and_concurrent_finalize_is_idempotent() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let _gate = GateGuard::set(true);
    let Some(provider) = provider_or_skip("readiness") else {
        return;
    };
    let provider = Arc::new(provider);
    let (graph, node, bank) = qmoe_graph_and_bank(21);
    let graph = Arc::new(graph);
    let bank = Arc::new(bank);
    let executor = ExecutorInstanceId::fresh();
    let declines = provider.route_residency_diagnostics().declines();
    assert_eq!(
        provider
            .finalize_executor_artifacts(
                executor,
                &graph,
                ExecutorArtifactReadinessEpoch::new(1),
                std::slice::from_ref(bank.as_ref()),
            )
            .expect("missing producer is pending"),
        ExecutorArtifactFinalization::Pending(ExecutorArtifactPending::ProducerUnavailable {
            node
        })
    );
    assert_eq!(provider.route_residency_diagnostics().declines(), declines);
    assert_eq!(provider.residency().unwrap().route_reservation_count(), 0);
    let _kernel = compile_real_qmoe(&provider, executor, &graph, node);

    let results = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..2 {
            let provider = Arc::clone(&provider);
            let graph = Arc::clone(&graph);
            let bank = Arc::clone(&bank);
            handles.push(scope.spawn(move || {
                provider
                    .finalize_executor_artifacts(
                        executor,
                        &graph,
                        ExecutorArtifactReadinessEpoch::new(2),
                        std::slice::from_ref(bank.as_ref()),
                    )
                    .expect("concurrent finalization")
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("finalizer thread"))
            .collect::<Vec<_>>()
    });
    assert!(results.contains(&ExecutorArtifactFinalization::Complete));
    assert!(results.iter().all(|result| matches!(
        result,
        ExecutorArtifactFinalization::Complete
            | ExecutorArtifactFinalization::Pending(
                ExecutorArtifactPending::ProviderReadiness { .. }
            )
    )));
    assert_eq!(
        provider
            .finalize_executor_artifacts(
                executor,
                &graph,
                ExecutorArtifactReadinessEpoch::new(2),
                std::slice::from_ref(bank.as_ref()),
            )
            .expect("idempotent retry after concurrent admission"),
        ExecutorArtifactFinalization::Complete
    );
    assert_eq!(
        provider
            .route_residency_executor_status(executor)
            .finalization_attempts,
        2,
        "one pending attempt plus one serialized install"
    );
    assert_eq!(provider.residency().unwrap().route_reservation_count(), 1);
    provider.drain_executor_artifacts(executor);
}

#[test]
#[ignore = "requires idle CUDA device"]
fn default_off_retains_allocates_and_registers_nothing() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let _gate = GateGuard::set(false);
    let Some(provider) = provider_or_skip("default-off") else {
        return;
    };
    let (graph, node, bank) = qmoe_graph_and_bank(31);
    let executor = ExecutorInstanceId::fresh();
    let _kernel = compile_real_qmoe(&provider, executor, &graph, node);
    assert!(!provider.wants_finalized_route_residency_banks());
    assert_eq!(
        provider
            .finalize_executor_artifacts(
                executor,
                &graph,
                ExecutorArtifactReadinessEpoch::new(1),
                &[bank],
            )
            .expect("default-off finalization"),
        ExecutorArtifactFinalization::Complete
    );
    assert_eq!(provider.route_residency_diagnostics().installs(), 0);
    assert_eq!(provider.route_residency_diagnostics().declines(), 0);
    assert_eq!(provider.residency().unwrap().route_reservation_count(), 0);
    assert!(provider.route_residency_scopes().is_empty());
    assert!(
        provider
            .executor_artifact_requirement(executor)
            .expect("query default-off requirement")
            .is_none(),
        "NeverInstalled is the only state represented by no requirement"
    );
    provider.drain_executor_artifacts(executor);
}

#[test]
#[ignore = "requires idle CUDA device with HOST_NUMA VMM support"]
fn bqmoe_without_real_telemetry_and_catalog_contract_typed_declines() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let _gate = GateGuard::set(true);
    let Some(provider) = provider_or_skip("BQMoE decline") else {
        return;
    };
    let (mut graph, node, bank) = qmoe_graph_and_bank(41);
    let bqmoe = graph.nodes.get_mut(node).expect("QMoE node");
    bqmoe.domain = "pkg.nxrt".into();
    bqmoe.op_type = "BlockQuantizedMoE".into();
    let executor = ExecutorInstanceId::fresh();

    assert_eq!(
        provider
            .finalize_executor_artifacts(
                executor,
                &graph,
                ExecutorArtifactReadinessEpoch::new(1),
                &[bank],
            )
            .expect("BQMoE finalization"),
        ExecutorArtifactFinalization::Complete
    );
    let outcome = provider.route_residency_executor_status(executor).outcome;
    assert!(
        matches!(
            outcome,
            Some(RouteResidencyInstallOutcome::Rejected(
                RouteResidencyBindingReject::UnsupportedBoundary {
                    boundary: LazyWeightBoundary::BlockQuantizedMoe,
                    ..
                }
            ))
        ),
        "unexpected BQMoE outcome: {outcome:?}"
    );
    assert_eq!(provider.residency().unwrap().route_reservation_count(), 0);
    provider.drain_executor_artifacts(executor);
}

#[test]
#[ignore = "requires idle CUDA device with HOST_NUMA VMM support"]
fn overlapping_external_bank_properties_decline_before_reservation() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let _gate = GateGuard::set(true);
    let Some(provider) = provider_or_skip("overlap decline") else {
        return;
    };
    let (graph, node, mut bank) = qmoe_graph_and_bank(51);
    bank.members[1].weight.regions[0].offset = bank.members[0].weight.regions[0].offset;
    let executor = ExecutorInstanceId::fresh();
    let _kernel = compile_real_qmoe(&provider, executor, &graph, node);

    assert_eq!(
        provider
            .finalize_executor_artifacts(
                executor,
                &graph,
                ExecutorArtifactReadinessEpoch::new(1),
                &[bank],
            )
            .expect("overlap finalization"),
        ExecutorArtifactFinalization::Complete
    );
    assert!(matches!(
        provider.route_residency_executor_status(executor).outcome,
        Some(RouteResidencyInstallOutcome::Rejected(
            RouteResidencyBindingReject::Reservation(
                RouteBankReservationReject::OverlappingExternalRange { .. }
            )
        ))
    ));
    assert_eq!(provider.residency().unwrap().route_reservation_count(), 0);
    provider.drain_executor_artifacts(executor);
}
