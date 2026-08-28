//! #1810 Slice 7E — GPU tests for the *real-producer* route-residency install
//! seam on the CUDA EP.
//!
//! Slice 7D (`route_residency_boundary_gpu.rs`) drove the boundary *consumer*
//! through the production caller, but its producer window came from a
//! controllable `RouteTelemetrySource` double, its per-bank artifacts were not
//! retained by the EP, and no session/model-build call constructed the binding.
//! Slice 7E removes those three limitations. These tests therefore use **no**
//! telemetry double: they register the **actual** executing
//! [`QMoEKernel`](onnx_runtime_ep_cuda) producer through the EP's own kernel
//! factory (exactly as the session's `compile_all` pre-warm does), then invoke
//! the **same trait method the session executor calls after resolved compile**
//! (`ExecutionProvider::finalize_executor_artifacts`) and
//! assert the whole real seam:
//!
//! * the EP owns the real producer source for the compiled `QMoE` node,
//! * the enabled build path discovers the bank, retains its per-bank artifacts,
//!   and reaches the shipped honest typed decline
//!   (`RouteResidencyBindingReject::NoPerBankReservation`) — the residual is the
//!   per-bank dedicated VMM reservation, disclosed in the decision record — with
//!   **no** silent whole-bank / default-success and **no** boundary installed,
//! * the default-off path allocates/registers no producer, installs nothing,
//!   retains nothing, and touches no route diagnostics at all (a pure inert
//!   early return).
//!
//! No transition is fabricated: on the shipped shared-reservation residency the
//! honest outcome is a typed decline, and these tests assert exactly that.
//!
//! Requires an idle CUDA device. Run solo, after verifying the target GPU is
//! idle with `nvidia-smi`:
//! ```text
//! CUDA_VISIBLE_DEVICES=<idle> cargo test -p onnx-runtime-ep-cuda \
//!   --features cuda,cuda-13000,gpu-tests --release \
//!   --test route_residency_install_gpu \
//!   -- --ignored --nocapture --test-threads=1
//! ```

#![allow(clippy::uninlined_format_args)]

use std::sync::{Arc, Mutex};

use onnx_runtime_ep_api::{
    ExecutionProvider, ExecutorArtifactFinalization, ExecutorArtifactPending,
    ExecutorArtifactReadinessEpoch, ExecutorInstanceId,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::coarse_residency::COARSE_RESIDENCY_ENABLE_ENV;
use onnx_runtime_ep_cuda::weight_paging::DeviceOffloadPolicy;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef, static_shape,
};
use onnx_runtime_memory_governor::{LeaseLedger, LedgerGovernor};

// Serialize GPU test bodies in this binary: the coarse gate is a process-global
// env var, so two tests toggling it concurrently would race.
static GPU_SERIAL: Mutex<()> = Mutex::new(());

fn gate_on() {
    unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };
}

fn gate_off() {
    unsafe { std::env::remove_var(COARSE_RESIDENCY_ENABLE_ENV) };
}

fn ambient_gate_is_on() -> bool {
    matches!(
        std::env::var(COARSE_RESIDENCY_ENABLE_ENV)
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

/// A CUDA EP with weight offload/coarse residency *available* (so `self.residency`
/// is `Some` and the install seam can get past `OffloadDisabled`), or `None` when
/// no CUDA device is present (the test then skips, staying green on CPU-only CI).
fn offload_provider_or_skip(label: &str) -> Option<CudaExecutionProvider> {
    let governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync> =
        Arc::new(LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0)));
    let policy = DeviceOffloadPolicy {
        enabled: true,
        device_budget_bytes: Some((2usize << 20) as u64),
        ..DeviceOffloadPolicy::default()
    };
    match CudaExecutionProvider::initialized_with_offload_policy_and_governor(0, policy, governor) {
        Ok(p) => Some(p),
        Err(e) => {
            println!("SKIP [{label}]: no CUDA device / offload EP unavailable: {e}");
            None
        }
    }
}

fn inline_u8_initializer(graph: &mut Graph, name: &str) -> ValueId {
    let value = graph.create_named_value(name, DataType::Uint8, static_shape([4]));
    graph.set_initializer(
        value,
        WeightRef::Inline(TensorData::from_raw(DataType::Uint8, vec![4], vec![0u8; 4])),
    );
    value
}

/// A shape-faithful single-layer `QMoE` graph: the routed multi-tensor expert
/// group (fc1/fc2/fc3 weights+scales+bias, `com.microsoft` layout) that
/// `expert_weight_groups` discovers, carrying the attributes a real
/// `QMoEKernel` needs to be constructed (`create_kernel` reads only attributes,
/// no weight *data* is required to build the executing kernel instance). Returns
/// the inserted node id and its ordered bank member `ValueId`s.
fn qmoe_graph() -> (Graph, NodeId, Vec<ValueId>) {
    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let input = graph.create_named_value("hidden", DataType::Float32, static_shape([4]));
    let router = graph.create_named_value("router_probs", DataType::Float32, static_shape([4]));
    let fc1_w = inline_u8_initializer(&mut graph, "fc1_experts_weights");
    let fc1_s = inline_u8_initializer(&mut graph, "fc1_scales");
    let fc1_b = inline_u8_initializer(&mut graph, "fc1_experts_bias");
    let fc2_w = inline_u8_initializer(&mut graph, "fc2_experts_weights");
    let fc2_s = inline_u8_initializer(&mut graph, "fc2_scales");
    let fc3_w = inline_u8_initializer(&mut graph, "fc3_experts_weights");
    let fc3_s = inline_u8_initializer(&mut graph, "fc3_scales");
    let output = graph.create_named_value("output", DataType::Float32, static_shape([4]));
    let mut node = Node::new(
        NodeId(0),
        "QMoE",
        vec![
            Some(input),
            Some(router),
            Some(fc1_w),
            Some(fc1_s),
            Some(fc1_b),
            Some(fc2_w),
            Some(fc2_s),
            None,
            Some(fc3_w),
            Some(fc3_s),
        ],
        vec![output],
    );
    node.domain = "com.microsoft".to_string();
    for (name, value) in [
        ("expert_weight_bits", Attribute::Int(4)),
        ("block_size", Attribute::Int(16)),
        ("k", Attribute::Int(2)),
        ("activation_type", Attribute::String(b"silu".to_vec())),
        ("normalize_routing_weights", Attribute::Int(0)),
        ("swiglu_fusion", Attribute::Int(0)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    let node_id = graph.insert_node(node);
    graph.add_output(output);
    (
        graph,
        node_id,
        vec![fc1_w, fc1_s, fc1_b, fc2_w, fc2_s, fc3_w, fc3_s],
    )
}

fn block_quantized_moe_graph() -> (Graph, NodeId) {
    let mut graph = Graph::new();
    graph.opset_imports.insert("pkg.nxrt".into(), 1);
    let input = graph.create_named_value("hidden", DataType::Float32, static_shape([4]));
    let router = graph.create_named_value("router_probs", DataType::Float32, static_shape([4]));
    let weight = inline_u8_initializer(&mut graph, "experts");
    let output = graph.create_named_value("output", DataType::Float32, static_shape([4]));
    let mut node = Node::new(
        NodeId(0),
        "BlockQuantizedMoE",
        vec![Some(input), Some(router), Some(weight)],
        vec![output],
    );
    node.domain = "pkg.nxrt".to_string();
    for (name, value) in [
        ("k", Attribute::Int(2)),
        ("activation_type", Attribute::String(b"silu".to_vec())),
        ("normalize_routing_weights", Attribute::Int(0)),
        ("swiglu_fusion", Attribute::Int(0)),
        ("fc1_format", Attribute::String(b"mxfp4".to_vec())),
        ("fc2_format", Attribute::String(b"mxfp4".to_vec())),
    ] {
        node.attributes.insert(name.into(), value);
    }
    let node_id = graph.insert_node(node);
    graph.add_output(output);
    (graph, node_id)
}

/// Compile the graph's `QMoE` node through the EP's own factory so the actual
/// executing `QMoEKernel` registers as this EP's route-telemetry producer —
/// exactly the path the session's `compile_all` pre-warm takes. Returns the
/// boxed kernel so it (and thus the shared `Arc`) stays alive for the assertion.
fn compile_qmoe_through_ep(
    provider: &CudaExecutionProvider,
    executor: ExecutorInstanceId,
    graph: &Graph,
    node_id: NodeId,
) -> Box<dyn onnx_runtime_ep_api::Kernel> {
    provider
        .get_kernel_for_executor(executor, graph.node(node_id), &[], 1)
        .expect("EP constructs the real QMoE kernel from its attributes")
}

// ---------------------------------------------------------------------------
// Test 1: the enabled build path binds this EP's *real* producer and reaches the
// shipped honest typed decline over real, retained artifacts — no double, no
// silent success, no boundary installed.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn enabled_build_binds_real_producer_and_declines_without_per_bank_reservation() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!(
        "\n=== enabled_build_binds_real_producer_and_declines_without_per_bank_reservation ==="
    );
    let provider = match offload_provider_or_skip("enabled") {
        Some(p) => p,
        None => return,
    };

    let (graph, node_id, members) = qmoe_graph();
    let executor = ExecutorInstanceId::fresh();

    // Before compile the EP owns no producer source and has retained nothing.
    assert!(
        provider.route_telemetry_sources(executor).is_empty(),
        "no producer registered before the QMoE node is compiled"
    );
    assert!(
        provider
            .retained_route_residency_artifacts(executor)
            .is_none(),
        "nothing retained before an enabled build"
    );

    gate_on();

    // Compile the QMoE node through the EP's factory: the executing kernel
    // registers itself as the EP-owned producer (goal 2, no test double).
    let _kernel = compile_qmoe_through_ep(&provider, executor, &graph, node_id);
    let sources = provider.route_telemetry_sources(executor);
    assert!(
        sources.contains_key(&node_id),
        "the actual executing QMoE kernel is the EP-owned producer for its node"
    );
    assert!(
        provider
            .route_telemetry_producer(executor, node_id)
            .is_some(),
        "the concrete QMoE producer is reachable by node id"
    );

    // Invoke the exact transition the session executor calls after resolved
    // compilation.
    assert_eq!(
        provider
            .finalize_executor_artifacts(executor, &graph, ExecutorArtifactReadinessEpoch::new(1),)
            .expect("finalize compiled executor"),
        ExecutorArtifactFinalization::Complete
    );
    let status = provider.route_residency_executor_status(executor);
    assert_eq!(status.finalization_attempts, 1);
    let stable_source = provider
        .route_telemetry_producer(executor, node_id)
        .expect("compiled QMoE has a stable producer");
    let _specialization = provider
        .get_kernel_for_executor(executor, graph.node(node_id), &[vec![2]], 1)
        .expect("dynamic QMoE specialization compiles");
    assert!(Arc::ptr_eq(
        &stable_source,
        &provider
            .route_telemetry_producer(executor, node_id)
            .expect("specialization retains the source")
    ));
    assert_eq!(
        provider
            .finalize_executor_artifacts(executor, &graph, ExecutorArtifactReadinessEpoch::new(2),)
            .expect("finalize specialized executor"),
        ExecutorArtifactFinalization::Complete
    );
    assert_eq!(
        provider
            .route_residency_executor_status(executor)
            .finalization_attempts,
        1,
        "terminal structural outcome must latch exactly once"
    );
    gate_off();

    let diag = provider.route_residency_diagnostics();
    // Honest typed decline: real discovery + retention succeeded, but the shipped
    // shared-reservation residency has no per-bank VMM reservation for the coarse
    // plan to address, so the seam fail-closes with the precise reason.
    assert_eq!(
        diag.installs(),
        0,
        "no boundary is installed on the shipped shared-reservation residency"
    );
    assert_eq!(
        diag.boundaries(),
        0,
        "no consumer boundary runs from a declined install"
    );
    assert!(diag.declines() >= 1, "the enabled build recorded a decline");
    let reason = diag
        .last_install_reason()
        .expect("a decline reason is surfaced to diagnostics");
    assert!(
        reason.contains("per-bank") && reason.contains("reservation"),
        "the decline discloses the per-bank-reservation residual: {reason}"
    );

    // Goal 1: the EP retained the real property-discovered per-bank artifacts.
    let retained = provider
        .retained_route_residency_artifacts(executor)
        .expect("enabled discovery retained the bank artifacts");
    assert_eq!(retained.len(), 1, "exactly one discovered expert bank");
    assert_eq!(retained[0].node, node_id, "retained the real bank node id");
    assert_eq!(
        retained[0].members, members,
        "retained the exact fc1/fc2/fc3 member ranges"
    );
}

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn readiness_absence_does_not_latch_and_concurrent_finalize_is_idempotent() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let provider = match offload_provider_or_skip("readiness") {
        Some(provider) => Arc::new(provider),
        None => return,
    };
    let (graph, node_id, _) = qmoe_graph();
    let graph = Arc::new(graph);
    let executor = ExecutorInstanceId::fresh();
    gate_on();

    let declines_before = provider.route_residency_diagnostics().declines();
    assert_eq!(
        provider
            .finalize_executor_artifacts(executor, &graph, ExecutorArtifactReadinessEpoch::new(1),)
            .expect("pending finalization is not an EP error"),
        ExecutorArtifactFinalization::Pending(ExecutorArtifactPending::ProducerUnavailable {
            node: node_id
        })
    );
    let pending = provider.route_residency_executor_status(executor);
    assert_eq!(pending.finalization_attempts, 1);
    assert_eq!(
        pending.readiness_epoch,
        Some(ExecutorArtifactReadinessEpoch::new(1))
    );
    assert_eq!(
        pending.pending,
        Some(ExecutorArtifactPending::ProducerUnavailable { node: node_id })
    );
    assert_eq!(
        provider
            .finalize_executor_artifacts(executor, &graph, ExecutorArtifactReadinessEpoch::new(1),)
            .expect("same pending epoch is cached"),
        ExecutorArtifactFinalization::Pending(ExecutorArtifactPending::ProducerUnavailable {
            node: node_id
        })
    );
    assert_eq!(
        provider
            .route_residency_executor_status(executor)
            .finalization_attempts,
        1,
        "same readiness epoch must not busy-retry provider finalization"
    );
    assert!(pending.outcome.is_none());
    assert_eq!(
        provider.route_residency_diagnostics().declines(),
        declines_before,
        "readiness absence is not a structural decline"
    );

    let _kernel = compile_qmoe_through_ep(&provider, executor, &graph, node_id);
    std::thread::scope(|scope| {
        for _ in 0..2 {
            let provider = Arc::clone(&provider);
            let graph = Arc::clone(&graph);
            scope.spawn(move || {
                assert_eq!(
                    provider
                        .finalize_executor_artifacts(
                            executor,
                            &graph,
                            ExecutorArtifactReadinessEpoch::new(2),
                        )
                        .expect("concurrent finalization"),
                    ExecutorArtifactFinalization::Complete
                );
            });
        }
    });
    let finalized = provider.route_residency_executor_status(executor);
    assert_eq!(
        finalized.finalization_attempts, 2,
        "one pending epoch plus one terminal install attempt; concurrent duplicate is idempotent"
    );
    assert!(matches!(
        finalized.outcome,
        Some(onnx_runtime_ep_cuda::route_residency::RouteResidencyInstallOutcome::Rejected(
            onnx_runtime_ep_cuda::route_residency::RouteResidencyBindingReject::NoPerBankReservation {
                ..
            }
        ))
    ));

    provider.drain_executor_artifacts(executor);
    provider.drain_executor_artifacts(executor);
    let drained = provider.route_residency_executor_status(executor);
    assert_eq!(drained.drain_calls, 1);
    assert!(drained.drained);
    gate_off();
}

// ---------------------------------------------------------------------------
// Test 2: BQMoE has no executor-scoped producer publication path, so source
// absence is a terminal typed decline rather than permanent Pending.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn block_quantized_moe_without_producer_is_terminal_not_pending() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let provider = match offload_provider_or_skip("bqmoe-terminal") {
        Some(provider) => provider,
        None => return,
    };
    let (graph, node_id) = block_quantized_moe_graph();
    let executor = ExecutorInstanceId::fresh();
    gate_on();

    let _kernel = provider
        .get_kernel_for_executor(executor, graph.node(node_id), &[], 1)
        .expect("compile the real BlockQuantizedMoE kernel");
    assert!(
        provider.route_telemetry_sources(executor).is_empty(),
        "BlockQuantizedMoE compilation has no executor-scoped producer publication path"
    );

    assert_eq!(
        provider
            .finalize_executor_artifacts(executor, &graph, ExecutorArtifactReadinessEpoch::new(1),)
            .expect("unsupported producer capability is a typed decline"),
        ExecutorArtifactFinalization::Complete
    );
    let status = provider.route_residency_executor_status(executor);
    assert_eq!(status.finalization_attempts, 1);
    assert!(status.pending.is_none(), "BQMoE must not remain pending");
    assert!(matches!(
        status.outcome,
        Some(onnx_runtime_ep_cuda::route_residency::RouteResidencyInstallOutcome::Rejected(
            onnx_runtime_ep_cuda::route_residency::RouteResidencyBindingReject::TelemetryProducerUnsupported {
                node,
                boundary: onnx_runtime_ep_api::LazyWeightBoundary::BlockQuantizedMoe,
            }
        )) if node == node_id
    ));

    assert_eq!(
        provider
            .finalize_executor_artifacts(executor, &graph, ExecutorArtifactReadinessEpoch::new(2),)
            .expect("terminal decline remains idempotent"),
        ExecutorArtifactFinalization::Complete
    );
    assert_eq!(
        provider
            .route_residency_executor_status(executor)
            .finalization_attempts,
        1,
        "unsupported producer capability cannot busy-retry"
    );
    gate_off();
}

// ---------------------------------------------------------------------------
// Test 3: the default-off build path is inert — no producer allocation or
// registration, no install/retention, and no route diagnostics.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn disabled_build_installs_and_retains_nothing() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== disabled_build_installs_and_retains_nothing ===");
    if ambient_gate_is_on() {
        println!("SKIP: {COARSE_RESIDENCY_ENABLE_ENV} is truthy in the ambient env");
        return;
    }
    let provider = match offload_provider_or_skip("disabled") {
        Some(p) => p,
        None => return,
    };

    let (graph, node_id, _members) = qmoe_graph();
    let executor = ExecutorInstanceId::fresh();
    let _kernel = compile_qmoe_through_ep(&provider, executor, &graph, node_id);
    assert!(
        provider.route_telemetry_sources(executor).is_empty(),
        "default-off compilation must not register a producer"
    );
    assert!(
        provider
            .route_telemetry_producer(executor, node_id)
            .is_none(),
        "default-off compilation must not allocate/retain a telemetry Arc"
    );

    gate_off();
    assert_eq!(
        provider
            .finalize_executor_artifacts(executor, &graph, ExecutorArtifactReadinessEpoch::new(1),)
            .expect("default-off finalization"),
        ExecutorArtifactFinalization::Complete
    );

    let diag = provider.route_residency_diagnostics();
    assert_eq!(diag.installs(), 0, "default-off installs nothing");
    assert_eq!(diag.boundaries(), 0, "default-off runs no consumer");
    assert_eq!(
        diag.declines(),
        0,
        "default-off is a pure inert early return: it records no decline"
    );
    assert!(
        diag.last_install_reason().is_none(),
        "default-off touches no install diagnostics"
    );
    assert!(
        provider
            .retained_route_residency_artifacts(executor)
            .is_none(),
        "default-off retains no bank artifacts"
    );
    let status = provider.route_residency_executor_status(executor);
    assert_eq!(status.producer_nodes, 0);
    assert_eq!(status.finalization_attempts, 0);
    assert!(status.pending.is_none());
    assert!(status.outcome.is_none());

    // Draining the never-installed boundary is a safe no-op.
    provider.drain_executor_artifacts(executor);
    assert!(
        provider.route_telemetry_sources(executor).is_empty(),
        "teardown drains the EP-owned producer registry"
    );
    assert!(
        provider
            .retained_route_residency_artifacts(executor)
            .is_none(),
        "teardown leaves nothing retained"
    );
}
