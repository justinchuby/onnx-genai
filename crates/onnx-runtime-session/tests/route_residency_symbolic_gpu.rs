#![cfg(all(feature = "cuda", feature = "gpu-tests"))]

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use onnx_runtime_ep_api::{ExecutionProvider, ExecutorKernelScope, ExecutorRouteResidencyConfig};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::coarse_residency::COARSE_RESIDENCY_ENABLE_ENV;
use onnx_runtime_ep_cuda::route_residency::{
    RouteResidencyBindingReject, RouteResidencyInstallOutcome,
};
use onnx_runtime_ep_cuda::weight_paging::DeviceOffloadPolicy;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, TensorData, WeightRef, static_shape,
};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_memory_governor::{LeaseLedger, LedgerGovernor};
use onnx_runtime_session::{InferenceSession, Tensor};

static GPU_SERIAL: Mutex<()> = Mutex::new(());

struct GateGuard(Option<OsString>);

impl GateGuard {
    fn enable() -> Self {
        let previous = std::env::var_os(COARSE_RESIDENCY_ENABLE_ENV);
        unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };
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

fn provider_or_skip() -> Option<Arc<CudaExecutionProvider>> {
    let governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync> =
        Arc::new(LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0)));
    let policy = DeviceOffloadPolicy {
        enabled: true,
        device_budget_bytes: Some(8 << 30),
        ..DeviceOffloadPolicy::default()
    };
    match CudaExecutionProvider::initialized_with_offload_policy_and_governor(0, policy, governor) {
        Ok(provider) => Some(Arc::new(provider)),
        Err(error) => {
            println!("SKIP: CUDA offload provider unavailable: {error}");
            None
        }
    }
}

fn provider_with_route_config_or_skip(
    route_config: ExecutorRouteResidencyConfig,
) -> Option<Arc<CudaExecutionProvider>> {
    let governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync> =
        Arc::new(LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0)));
    let policy = DeviceOffloadPolicy {
        enabled: true,
        device_budget_bytes: Some(8 << 30),
        ..DeviceOffloadPolicy::default()
    };
    match CudaExecutionProvider::initialized_with_offload_policy_governor_and_route_config(
        0,
        policy,
        governor,
        route_config,
    ) {
        Ok(provider) => Some(Arc::new(provider)),
        Err(error) => {
            println!("SKIP: CUDA offload provider unavailable: {error}");
            None
        }
    }
}

fn unary_chain_model() -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("input", DataType::Float32, static_shape([4]));
    graph.add_input(input);
    let relu = graph.create_named_value("relu", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(input)], vec![relu]));
    let output = graph.create_named_value("output", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(1), "Abs", vec![Some(relu)], vec![output]));
    graph.add_output(output);
    encode_model(&Model::new(&graph)).expect("encode unary CUDA chain")
}

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-deepseek-v2-qmoe-attention/model.onnx.textproto")
}

fn static_qmoe_model() -> Vec<u8> {
    fn initializer(graph: &mut Graph, name: &str) -> onnx_runtime_ir::ValueId {
        let value = graph.create_named_value(name, DataType::Uint8, static_shape([4]));
        graph.set_initializer(
            value,
            WeightRef::Inline(TensorData::from_raw(DataType::Uint8, vec![4], vec![0; 4])),
        );
        value
    }

    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let hidden = graph.create_named_value("hidden", DataType::Float32, static_shape([4]));
    let router = graph.create_named_value("router", DataType::Float32, static_shape([4]));
    graph.add_input(hidden);
    graph.add_input(router);
    let fc1_w = initializer(&mut graph, "fc1_experts_weights");
    let fc1_s = initializer(&mut graph, "fc1_scales");
    let fc1_b = initializer(&mut graph, "fc1_experts_bias");
    let fc2_w = initializer(&mut graph, "fc2_experts_weights");
    let fc2_s = initializer(&mut graph, "fc2_scales");
    let fc3_w = initializer(&mut graph, "fc3_experts_weights");
    let fc3_s = initializer(&mut graph, "fc3_scales");
    let output = graph.create_named_value("output", DataType::Float32, static_shape([4]));
    let mut node = Node::new(
        NodeId(0),
        "QMoE",
        vec![
            Some(hidden),
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
    node.domain = "com.microsoft".into();
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
    graph.insert_node(node);
    graph.add_output(output);
    encode_model(&Model::new(&graph)).expect("encode static QMoE model")
}

fn symbolic_fixture_qmoe_model() -> Vec<u8> {
    let source = onnx_runtime_loader::load_model(fixture_model()).expect("load QMoE fixture");
    let source_node = source
        .nodes
        .values()
        .find(|node| node.domain == "com.microsoft" && node.op_type == "QMoE")
        .expect("fixture contains QMoE");

    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let batch = graph.intern_symbol("batch");
    let sequence = graph.intern_symbol("sequence");
    let hidden = graph.create_named_value(
        "hidden",
        DataType::Float32,
        vec![
            onnx_runtime_ir::Dim::Symbolic(batch),
            onnx_runtime_ir::Dim::Symbolic(sequence),
            onnx_runtime_ir::Dim::Static(16),
        ],
    );
    let router = graph.create_named_value(
        "router_probs",
        DataType::Float32,
        vec![
            onnx_runtime_ir::Dim::Symbolic(batch),
            onnx_runtime_ir::Dim::Symbolic(sequence),
            onnx_runtime_ir::Dim::Static(4),
        ],
    );
    graph.add_input(hidden);
    graph.add_input(router);

    let mut inputs = vec![Some(hidden), Some(router)];
    for source_input in source_node.inputs.iter().skip(2) {
        let Some(source_value) = source_input else {
            inputs.push(None);
            continue;
        };
        let value = source.value(*source_value);
        let copied = graph.create_named_value(
            value.name.clone().expect("fixture initializer is named"),
            value.dtype,
            value.shape.clone(),
        );
        graph.set_initializer(
            copied,
            source
                .initializers
                .get(source_value)
                .expect("QMoE weight is an initializer")
                .clone(),
        );
        inputs.push(Some(copied));
    }
    let output = graph.create_named_value(
        "output",
        DataType::Float32,
        vec![
            onnx_runtime_ir::Dim::Symbolic(batch),
            onnx_runtime_ir::Dim::Symbolic(sequence),
            onnx_runtime_ir::Dim::Static(16),
        ],
    );
    let mut node = Node::new(NodeId(0), "QMoE", inputs, vec![output]);
    node.domain = source_node.domain.clone();
    node.name = source_node.name.clone();
    node.attributes = source_node.attributes.clone();
    graph.insert_node(node);
    graph.add_output(output);
    encode_model(&Model::new(&graph)).expect("encode symbolic fixture-derived QMoE")
}

fn run_symbolic_qmoe(
    session: &mut InferenceSession,
    batch: usize,
    sequence: usize,
    tokens: &[i64],
) {
    assert_eq!(tokens.len(), batch * sequence);
    let hidden_values = tokens
        .iter()
        .flat_map(|token| (0..16).map(move |index| (*token as f32 + index as f32) * 0.01))
        .collect::<Vec<_>>();
    let router_values = tokens
        .iter()
        .flat_map(|token| {
            let offset = (*token as f32) * 0.001;
            [0.7 + offset, 0.2, 0.1, 0.0]
        })
        .collect::<Vec<_>>();
    let hidden = Tensor::from_f32(&[batch, sequence, 16], &hidden_values).unwrap();
    let router = Tensor::from_f32(&[batch, sequence, 4], &router_values).unwrap();
    session
        .run(&[("hidden", &hidden), ("router_probs", &router)])
        .expect("run real symbolic QMoE producer");
}

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn enabled_multinode_sessions_share_one_generation_each_without_cpu_fallback() {
    let _serial = GPU_SERIAL.lock().unwrap();
    let Some(ep) = provider_with_route_config_or_skip(ExecutorRouteResidencyConfig::Enabled) else {
        return;
    };
    let model = unary_chain_model();
    let input = Tensor::from_f32(&[4], &[-1.0, 2.0, -3.0, 4.0]).unwrap();

    let mut first = InferenceSession::builder()
        .model_bytes(&model)
        .execution_provider(ep.clone())
        .build()
        .expect("build first enabled multi-node CUDA session");
    assert!(
        first.execution_provider_fallback_report().is_none(),
        "two scoped lookups in one executor generation must remain CUDA-claimed"
    );
    first
        .run(&[("input", &input)])
        .expect("execute first CUDA session");

    let mut foreign = InferenceSession::builder()
        .model_bytes(&model)
        .execution_provider(ep.clone())
        .build()
        .expect("build foreign executor on the shared CUDA provider");
    assert!(
        foreign.execution_provider_fallback_report().is_none(),
        "a foreign executor must receive its own generation without invalidating the first"
    );
    foreign
        .run(&[("input", &input)])
        .expect("execute foreign CUDA session");

    let claims = ep.executor_artifact_generation_claims();
    assert_eq!(
        claims.len(),
        2,
        "each executor must claim exactly one generation despite two preflight and compile lookups"
    );
    assert_ne!(claims[0].0, claims[1].0, "foreign executors must differ");
    assert_ne!(
        claims[0].1, claims[1].1,
        "foreign executors must not share a generation"
    );
}

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn enabled_required_kernel_rejects_missing_scope_while_disabled_is_inert() {
    let _serial = GPU_SERIAL.lock().unwrap();
    let Some(enabled) = provider_with_route_config_or_skip(ExecutorRouteResidencyConfig::Enabled)
    else {
        return;
    };
    let mut qmoe = Node::new(NodeId(0), "QMoE", Vec::new(), Vec::new());
    qmoe.domain = "com.microsoft".into();
    assert_eq!(
        enabled.executor_kernel_scope(&qmoe),
        ExecutorKernelScope::Required
    );
    let error = match enabled.get_kernel(&qmoe, &[], 1) {
        Ok(_) => panic!("enabled QMoE must reject an unscoped kernel lookup"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("requires a session-owned executor lifecycle")
            && error.to_string().contains("onnx-runtime-session"),
        "unexpected missing-scope diagnostic: {error}"
    );
    assert!(
        enabled.executor_artifact_generation_claims().is_empty(),
        "a rejected unscoped lookup must not claim or publish a generation"
    );

    let Some(disabled) = provider_with_route_config_or_skip(ExecutorRouteResidencyConfig::Disabled)
    else {
        return;
    };
    assert_eq!(
        disabled.executor_kernel_scope(&qmoe),
        ExecutorKernelScope::Unscoped
    );
    assert!(
        disabled.executor_artifact_generation_claims().is_empty(),
        "disabled construction must remain zero-work"
    );
}

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn symbolic_qmoe_finalizes_after_real_compile_and_shared_ep_state_is_isolated() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let _gate = GateGuard::enable();
    let Some(provider) = provider_or_skip() else {
        return;
    };

    let ep: Arc<dyn ExecutionProvider> = provider.clone();
    let model = symbolic_fixture_qmoe_model();
    let mut first = InferenceSession::builder()
        .model_bytes(&model)
        .execution_provider(ep.clone())
        .build()
        .expect("build first symbolic QMoE session");
    let first_id = first.executor_instance_id();
    let qmoe_node = first
        .graph()
        .nodes
        .values()
        .find(|node| node.domain == "com.microsoft" && node.op_type == "QMoE")
        .expect("fixture contains real QMoE")
        .id;

    let built = provider.route_residency_executor_status(first_id);
    assert_eq!(built.finalization_attempts, 0);
    assert!(built.outcome.is_none());
    assert_eq!(built.producer_nodes, 0);
    assert!(
        provider
            .retained_route_residency_artifacts(first_id)
            .is_none()
    );
    run_symbolic_qmoe(&mut first, 1, 2, &[3, 4]);
    let first_run = provider.route_residency_executor_status(first_id);
    assert_eq!(
        first_run.finalization_attempts, 1,
        "real QMoEFactory compilation advances readiness before execution"
    );
    assert!(first_run.pending.is_none());
    assert_eq!(first_run.producer_nodes, 1);
    assert_eq!(first_run.retained_banks, 1);
    assert!(matches!(
        first_run.outcome,
        Some(RouteResidencyInstallOutcome::Rejected(
            RouteResidencyBindingReject::NoPerBankReservation { .. }
        ))
    ));
    let stable_source = provider
        .route_telemetry_producer(first_id, qmoe_node)
        .expect("resolved compilation registered the real QMoE producer");

    run_symbolic_qmoe(&mut first, 2, 2, &[3, 4, 3, 4]);
    let specialized = provider.route_residency_executor_status(first_id);
    assert_eq!(
        specialized.finalization_attempts, 1,
        "dynamic specialization must not reinstall"
    );
    assert!(Arc::ptr_eq(
        &stable_source,
        &provider
            .route_telemetry_producer(first_id, qmoe_node)
            .expect("specialization keeps the stable producer source")
    ));

    let mut second = InferenceSession::builder()
        .model_bytes(&model)
        .execution_provider(ep)
        .build()
        .expect("build second symbolic QMoE session");
    let second_id = second.executor_instance_id();
    assert_ne!(first_id, second_id);
    run_symbolic_qmoe(&mut second, 1, 2, &[3, 4]);
    assert_eq!(
        provider
            .route_residency_executor_status(second_id)
            .finalization_attempts,
        1
    );

    drop(first);
    let drained = provider.route_residency_executor_status(first_id);
    assert_eq!(drained.drain_calls, 1);
    assert!(drained.drained);
    assert_eq!(drained.producer_nodes, 0);
    assert_eq!(drained.retained_banks, 0);

    let sibling = provider.route_residency_executor_status(second_id);
    assert_eq!(sibling.drain_calls, 0);
    assert!(!sibling.drained);
    assert_eq!(sibling.producer_nodes, 1);
    assert_eq!(sibling.retained_banks, 1);
    run_symbolic_qmoe(&mut second, 2, 2, &[3, 4, 3, 4]);
    assert_eq!(
        provider
            .route_residency_executor_status(second_id)
            .finalization_attempts,
        1
    );

    drop(second);
    let second_drained = provider.route_residency_executor_status(second_id);
    assert_eq!(second_drained.drain_calls, 1);
    assert!(second_drained.drained);
    assert_eq!(second_drained.producer_nodes, 0);
}

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn static_qmoe_build_uses_same_finalization_transition() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let _gate = GateGuard::enable();
    let Some(provider) = provider_or_skip() else {
        return;
    };
    let ep: Arc<dyn ExecutionProvider> = provider.clone();
    let session = InferenceSession::builder()
        .model_bytes(&static_qmoe_model())
        .execution_provider(ep)
        .build()
        .expect("build static QMoE session");
    let executor = session.executor_instance_id();
    let status = provider.route_residency_executor_status(executor);
    assert_eq!(status.finalization_attempts, 1);
    assert_eq!(status.producer_nodes, 1);
    assert_eq!(status.retained_banks, 1);
    assert!(matches!(
        status.outcome,
        Some(RouteResidencyInstallOutcome::Rejected(
            RouteResidencyBindingReject::NoPerBankReservation { .. }
        ))
    ));
    drop(session);
    let drained = provider.route_residency_executor_status(executor);
    assert_eq!(drained.drain_calls, 1);
    assert_eq!(drained.producer_nodes, 0);
}
