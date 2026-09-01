use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use onnx_runtime_ep_api::{
    ExecutionProvider, ExecutorArtifactFinalization, ExecutorArtifactPending,
    ExecutorArtifactReadinessEpoch, ExecutorInstanceId, ExternalMmapRegion, FinalizedExpertBank,
    FinalizedExpertWeight, LazyWeight, LazyWeightBoundary, ResidentWeight, expert_weight_groups,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::coarse_residency::COARSE_RESIDENCY_ENABLE_ENV;
use onnx_runtime_ep_cuda::route_residency::{
    RouteResidencyBindingReject, RouteResidencyInstallOutcome,
};
use onnx_runtime_ep_cuda::weight_paging::{DeviceOffloadPolicy, RouteBankReservationReject};
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

fn finalize(
    provider: &CudaExecutionProvider,
    executor: ExecutorInstanceId,
    graph: &Graph,
    readiness_epoch: u64,
    banks: &[FinalizedExpertBank],
) -> ExecutorArtifactFinalization {
    provider
        .finalize_executor_artifacts(
            executor,
            graph,
            banks,
            ExecutorArtifactReadinessEpoch::new(readiness_epoch),
        )
        .expect("executor artifact finalization")
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
    let (graph, node, bank) = qmoe_graph_and_bank(11);
    let first = ExecutorInstanceId::fresh();
    let second = ExecutorInstanceId::fresh();
    let _first_kernel = compile_real_qmoe(&provider, first, &graph, node);
    let _second_kernel = compile_real_qmoe(&provider, second, &graph, node);

    assert_eq!(
        finalize(&provider, first, &graph, 1, std::slice::from_ref(&bank)),
        ExecutorArtifactFinalization::Complete
    );
    assert_eq!(
        finalize(&provider, second, &graph, 1, std::slice::from_ref(&bank)),
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

    let _specialization = provider
        .get_kernel_for_executor(first, graph.node(node), &[vec![2, 4]], 1)
        .expect("dynamic specialization");
    assert_eq!(
        finalize(&provider, first, &graph, 2, std::slice::from_ref(&bank)),
        ExecutorArtifactFinalization::Complete
    );
    assert_eq!(
        provider
            .route_residency_executor_status(first)
            .finalization_attempts,
        1
    );

    provider
        .drain_executor_artifacts(first)
        .expect("first executor drain");
    let stale_finalization = provider
        .finalize_executor_artifacts(
            first,
            &graph,
            std::slice::from_ref(&bank),
            ExecutorArtifactReadinessEpoch::new(3),
        )
        .expect_err("a drained executor must reject late finalization");
    assert!(stale_finalization.to_string().contains("already drained"));
    let stale_kernel =
        match provider.get_kernel_for_executor(first, graph.node(node), &[vec![3, 4]], 1) {
            Ok(_) => panic!("a drained executor must reject late kernel publication"),
            Err(error) => error,
        };
    assert!(stale_kernel.to_string().contains("already drained"));
    provider
        .drain_executor_artifacts(first)
        .expect("idempotent first executor drain");
    assert_eq!(provider.residency().unwrap().route_reservation_count(), 1);
    assert!(provider.route_residency_scopes().contains(&second));
    assert_eq!(
        provider.route_residency_executor_status(first).drain_calls,
        1
    );
    provider
        .drain_executor_artifacts(second)
        .expect("second executor drain");
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
    let declines = provider.route_residency_diagnostics(executor).declines();
    assert!(matches!(
        finalize(
            &provider,
            executor,
            &graph,
            0,
            std::slice::from_ref(bank.as_ref())
        ),
        ExecutorArtifactFinalization::Pending(ExecutorArtifactPending::ProducerUnavailable {
            node: pending_node
        }) if pending_node == node
    ));
    assert_eq!(
        provider.route_residency_diagnostics(executor).declines(),
        declines
    );
    assert_eq!(provider.residency().unwrap().route_reservation_count(), 0);
    let _kernel = compile_real_qmoe(&provider, executor, &graph, node);

    std::thread::scope(|scope| {
        for _ in 0..2 {
            let provider = Arc::clone(&provider);
            let graph = Arc::clone(&graph);
            let bank = Arc::clone(&bank);
            scope.spawn(move || {
                assert_eq!(
                    finalize(
                        &provider,
                        executor,
                        &graph,
                        1,
                        std::slice::from_ref(bank.as_ref()),
                    ),
                    ExecutorArtifactFinalization::Complete
                );
            });
        }
    });
    assert_eq!(
        provider
            .route_residency_executor_status(executor)
            .finalization_attempts,
        2,
        "one pending attempt plus one serialized install"
    );
    assert_eq!(provider.residency().unwrap().route_reservation_count(), 1);
    provider
        .drain_executor_artifacts(executor)
        .expect("executor drain");
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
        finalize(&provider, executor, &graph, 1, &[bank]),
        ExecutorArtifactFinalization::Complete
    );
    assert_eq!(provider.route_residency_diagnostics(executor).installs(), 0);
    assert_eq!(provider.route_residency_diagnostics(executor).declines(), 0);
    assert_eq!(provider.residency().unwrap().route_reservation_count(), 0);
    assert!(provider.route_residency_scopes().is_empty());
    provider
        .drain_executor_artifacts(executor)
        .expect("executor drain");
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
        finalize(&provider, executor, &graph, 0, &[bank]),
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
    provider
        .drain_executor_artifacts(executor)
        .expect("executor drain");
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
        finalize(&provider, executor, &graph, 1, &[bank]),
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
    provider
        .drain_executor_artifacts(executor)
        .expect("executor drain");
}
