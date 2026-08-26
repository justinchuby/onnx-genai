use std::collections::HashSet;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use onnx_runtime_ep_api::{
    CaptureRegionShapeStatus, DeviceBuffer, EpConfig, EpError, ExecutionProvider, Fence,
    KernelMatch, PluginCompiledKernel, PluginExecutionPlan, StructuralCaptureDecline,
};
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, DeviceType, Graph, Node, NodeId, Shape, TensorLayout,
};
use onnx_runtime_optimizer::{OptimizationPass, OptimizerError, PassContext};
use onnx_runtime_tracer::{Args, annotate_current_span_with};

/// Synthetic op type used after a plugin EP compiles and replaces a claimed subgraph.
pub(crate) const PLUGIN_FUSED_OP: &str = "NativePluginFused";

/// The device a plugin runs on when its configured name is not one we know.
///
/// One owner for the fallback, so the node's placement and the provider's
/// reported `device_type` cannot drift apart and describe the same work as
/// running in two places.
const UNKNOWN_PLUGIN_DEVICE: DeviceType = DeviceType::Custom(0);
/// Our own domain, not `com.microsoft`.
///
/// This operator is invented here -- it stands for "a subgraph some plugin
/// claimed" and has no meaning outside this runtime. Putting it in Microsoft's
/// namespace would assert a specification that does not exist, and would
/// collide the moment they define something by this name.
pub(crate) const PLUGIN_FUSED_DOMAIN: &str = onnx_runtime_ir::RUNTIME_DOMAIN;
const FUSION_ID_ATTR: &str = "native_plugin_fusion_id";
pub(crate) const FUSION_SHAPE_GRAPH_ATTR: &str = "native_plugin_shape_graph";

/// A native execution provider that dispatches plugin-claimed fused subgraphs
/// through ORT's plugin-EP ABI and delegates all unclaimed nodes to the CPU EP.
pub struct PluginExecutionProvider {
    library: PathBuf,
    registration_name: Option<String>,
    provider_name: String,
    device_label: String,
    cpu: onnx_runtime_ep_cpu::CpuExecutionProvider,
    compiled: Arc<Mutex<Option<PluginExecutionPlan>>>,
}

impl PluginExecutionProvider {
    /// Create a plugin bridge. Names come from `ep_compat` metadata; the bridge
    /// itself does not special-case vendor/provider spellings.
    pub fn new(
        library: PathBuf,
        registration_name: Option<String>,
        provider_name: impl Into<String>,
        device_label: impl Into<String>,
    ) -> onnx_runtime_ep_api::Result<Self> {
        let mut cpu = onnx_runtime_ep_cpu::CpuExecutionProvider::new();
        cpu.initialize(&EpConfig::default()).map_err(|err| {
            EpError::KernelFailed(format!(
                "initialize CPU fallback for plugin EP failed: {err}; fix by enabling the native CPU EP"
            ))
        })?;
        Ok(Self {
            library,
            registration_name,
            provider_name: provider_name.into(),
            device_label: device_label.into(),
            cpu,
            compiled: Arc::new(Mutex::new(None)),
        })
    }
}

impl ExecutionProvider for PluginExecutionProvider {
    fn consume_route_residency_at_boundary(&self) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.consume_route_residency_at_boundary()
    }

    fn name(&self) -> &str {
        &self.provider_name
    }

    fn device_type(&self) -> DeviceType {
        // Derived from the device name the provider was configured with, not
        // hardcoded: reporting `Cpu` (or an opaque `Custom`) would put every
        // span for work the plugin ran on an accelerator onto the host's
        // device in the trace. Unrecognised names stay `Custom`, which is
        // honest about not knowing rather than wrong about knowing.
        DeviceType::from_trace_name(&self.device_label).unwrap_or(UNKNOWN_PLUGIN_DEVICE)
    }

    fn device_id(&self) -> DeviceId {
        DeviceId::cpu()
    }

    fn initialize(&mut self, _config: &EpConfig) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        input_dtypes: &[DataType],
        layouts: &[TensorLayout],
    ) -> KernelMatch {
        if is_plugin_fused(op) {
            return KernelMatch::Supported {
                cost: onnx_runtime_ep_api::Cost::ZERO,
                required_input_layouts: None,
                output_layouts: vec![TensorLayout::contiguous(); op.outputs.len()],
            };
        }
        self.cpu
            .supports_op(op, opset, shapes, input_dtypes, layouts)
    }

    fn get_kernel(
        &self,
        op: &Node,
        shapes: &[Vec<usize>],
        opset: u64,
    ) -> onnx_runtime_ep_api::Result<Box<dyn onnx_runtime_ep_api::Kernel>> {
        if is_plugin_fused(op) {
            let id = fusion_id(op)?;
            let guard = self.compiled.lock().map_err(|_| {
                EpError::KernelFailed(
                    "plugin compiled-plan lock was poisoned; recreate the session".into(),
                )
            })?;
            let plan = guard.as_ref().ok_or_else(|| {
                EpError::KernelFailed(
                    "plugin fused node has no compiled OrtNodeComputeInfo; fix by running the plugin fusion pass before execution".into(),
                )
            })?;
            let kernel: PluginCompiledKernel = plan.kernel(id).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "plugin fused node references missing fusion id {id}; fix the plugin fusion pass"
                ))
            })?;
            return Ok(Box::new(kernel));
        }
        self.cpu.get_kernel(op, shapes, opset)
    }

    fn plan_capture_region(
        &self,
        node: &Node,
        shape_status: CaptureRegionShapeStatus,
    ) -> Option<StructuralCaptureDecline> {
        self.cpu.plan_capture_region(node, shape_status)
    }

    fn allocate(&self, size: usize, alignment: usize) -> onnx_runtime_ep_api::Result<DeviceBuffer> {
        self.cpu.allocate(size, alignment)
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.deallocate(buffer)
    }

    fn copy(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.copy(src, dst, size)
    }

    fn copy_async(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> onnx_runtime_ep_api::Result<Fence> {
        self.cpu.copy_async(src, dst, size)
    }

    fn sync(&self) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.sync()
    }

    fn custom_passes(&self) -> Vec<Box<dyn OptimizationPass>> {
        vec![Box::new(PluginFusionPass {
            library: self.library.clone(),
            registration_name: self.registration_name.clone(),
            device_label: self.device_label.clone(),
            compiled: Arc::clone(&self.compiled),
        })]
    }
}

struct PluginFusionPass {
    library: PathBuf,
    registration_name: Option<String>,
    device_label: String,
    compiled: Arc<Mutex<Option<PluginExecutionPlan>>>,
}

#[derive(Clone, Debug)]
struct ClaimCostEstimate {
    node_count: usize,
    op_arity: usize,
    runtime_input_count: usize,
    boundary_bytes: Option<u64>,
    constant_foldable: bool,
    all_boundary_inputs_on_plugin: bool,
    fixed_overhead_score: i64,
    transfer_penalty_score: i64,
    benefit_score: i64,
    net_score: i64,
}

impl ClaimCostEstimate {
    fn empty() -> Self {
        Self {
            node_count: 0,
            op_arity: 0,
            runtime_input_count: 0,
            boundary_bytes: Some(0),
            constant_foldable: true,
            all_boundary_inputs_on_plugin: true,
            fixed_overhead_score: 0,
            transfer_penalty_score: 0,
            benefit_score: 0,
            net_score: i64::MIN,
        }
    }
}

#[derive(Clone, Debug)]
/// One claim's cost decision, for the trace.
///
/// Whether it was accepted and its position are not recorded: the two lists
/// are kept separate, and the audit groups claims by the shape of the decision
/// rather than reporting them positionally.
struct ClaimDecisionAudit {
    estimate: ClaimCostEstimate,
    reason: String,
}

struct PluginFusionCostModel {
    plugin_device: DeviceType,
}

impl PluginFusionCostModel {
    const NODE_SCORE: i64 = 1_000;
    const ARITY_SCORE: i64 = 125;
    const RUNTIME_INPUT_SCORE: i64 = 500;
    const UNKNOWN_BOUNDARY_SCORE: i64 = 1_500;
    const BOUNDARY_KIB_SCORE: i64 = 1;
    const FIXED_ABI_OVERHEAD_SCORE: i64 = 1_500;
    const NON_HOST_TRANSFER_PENALTY_PER_KIB: i64 = 4;

    fn for_device_label(device_label: &str) -> Self {
        Self {
            plugin_device: DeviceType::from_trace_name(device_label)
                .unwrap_or(DeviceType::Custom(0)),
        }
    }

    fn estimate(
        &self,
        graph: &Graph,
        claim: &onnx_runtime_ep_api::SubgraphClaim,
    ) -> ClaimCostEstimate {
        let node_count = claim.node_ids.len();
        let mut op_arity = 0usize;
        for &node_id in &claim.node_ids {
            let node = graph.node(node_id);
            op_arity += node.input_values().count();
            op_arity += node.outputs.len();
        }
        let runtime_input_count = claim
            .input_values
            .iter()
            .filter(|&&value_id| is_runtime_input(graph, value_id))
            .count();
        let constant_foldable = runtime_input_count == 0;
        let all_boundary_inputs_on_plugin = claim.input_values.iter().all(|&value_id| {
            graph
                .value(value_id)
                .device
                .is_some_and(|device| device.device_type == self.plugin_device)
        });
        let boundary_bytes = boundary_bytes(graph, claim);
        let boundary_kib = boundary_bytes.map(|bytes| bytes.div_ceil(1024) as i64);
        let byte_score = boundary_kib
            .map(|kib| kib * Self::BOUNDARY_KIB_SCORE)
            .unwrap_or(Self::UNKNOWN_BOUNDARY_SCORE);
        let benefit_score = node_count as i64 * Self::NODE_SCORE
            + op_arity as i64 * Self::ARITY_SCORE
            + runtime_input_count as i64 * Self::RUNTIME_INPUT_SCORE
            + byte_score;
        let transfer_penalty_score =
            if self.plugin_device.is_host_accessible() || all_boundary_inputs_on_plugin {
                0
            } else {
                boundary_kib
                    .map(|kib| kib * Self::NON_HOST_TRANSFER_PENALTY_PER_KIB)
                    .unwrap_or(Self::UNKNOWN_BOUNDARY_SCORE)
            };
        let fixed_overhead_score = Self::FIXED_ABI_OVERHEAD_SCORE;
        let net_score = benefit_score - fixed_overhead_score - transfer_penalty_score;
        ClaimCostEstimate {
            node_count,
            op_arity,
            runtime_input_count,
            boundary_bytes,
            constant_foldable,
            all_boundary_inputs_on_plugin,
            fixed_overhead_score,
            transfer_penalty_score,
            benefit_score,
            net_score,
        }
    }
}

impl OptimizationPass for PluginFusionPass {
    fn name(&self) -> &str {
        "plugin_ep_fusion"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> onnx_runtime_optimizer::Result<()> {
        let registration = self
            .registration_name
            .as_ref()
            .map(|name| {
                CString::new(name.as_str()).map_err(|_| {
                    OptimizerError::Fusion(
                        "plugin registration name contains an interior NUL byte".into(),
                    )
                })
            })
            .transpose()?;
        let (claims, plan) = PluginExecutionPlan::compile_with_device_label(
            graph,
            &self.library,
            registration.as_deref(),
            self.device_label.clone(),
        )
        .map_err(|err| OptimizerError::Fusion(format!("{err}")))?;

        let cost_model = PluginFusionCostModel::for_device_label(&self.device_label);
        let min_net_score = onnx_genai_runtime_config::runtime_config().plugin_fusion_min_net_score;
        let mut groups = Vec::new();
        let mut accepted = Vec::new();
        let mut declined = Vec::new();
        for (index, claim) in claims.iter().enumerate() {
            if claim.node_ids.is_empty() {
                declined.push(ClaimDecisionAudit {
                    estimate: ClaimCostEstimate::empty(),
                    reason: "empty plugin claim".to_string(),
                });
                continue;
            }
            let estimate = cost_model.estimate(graph, claim);
            if estimate.net_score < min_net_score {
                declined.push(ClaimDecisionAudit {
                    estimate: estimate.clone(),
                    reason: format!(
                        "estimated net score {} is below ONNX_GENAI_PLUGIN_FUSION_MIN_NET_SCORE={}",
                        estimate.net_score, min_net_score
                    ),
                });
                continue;
            }
            let shape_graph = shape_graph_for_claim(graph, claim)?;
            let mut node = Node::new(
                NodeId(0),
                PLUGIN_FUSED_OP,
                claim.input_values.iter().copied().map(Some).collect(),
                claim.output_values.clone(),
            );
            node.domain = PLUGIN_FUSED_DOMAIN.to_string();
            node.name = format!("native_plugin_fused_{index}");
            node.attributes
                .insert(FUSION_ID_ATTR.to_string(), Attribute::Int(index as i64));
            node.attributes.insert(
                FUSION_SHAPE_GRAPH_ATTR.to_string(),
                Attribute::Graph(Box::new(shape_graph)),
            );
            // Placed on the plugin's device, so a trace attributes this work to
            // the accelerator that did it. Left unset, the executor records no
            // device at all and the whole fused subgraph -- which is most of
            // the model -- shows up unattributed.
            //
            // Always set, including for a device label we do not recognise:
            // that resolves to `Custom(0)`, which says "some device we cannot
            // name" and is strictly more informative than silence. Resolved
            // the same way the provider resolves its own `device_type`, so the
            // node and the provider cannot disagree about where work ran.
            node.device = Some(DeviceId::new(
                DeviceType::from_trace_name(&self.device_label).unwrap_or(UNKNOWN_PLUGIN_DEVICE),
                0,
            ));
            groups.push((claim.node_ids.clone(), node));
            accepted.push(ClaimDecisionAudit {
                estimate,
                reason: "estimated net benefit meets threshold".to_string(),
            });
        }
        annotate_plugin_fusion_decisions(&accepted, &declined, min_net_score);
        if groups.is_empty() {
            let mut guard = self.compiled.lock().map_err(|_| {
                OptimizerError::Fusion(
                    "plugin compiled-plan lock was poisoned while recording no accepted fused groups"
                        .into(),
                )
            })?;
            *guard = None;
            return Ok(());
        }
        let graph_outputs = graph.outputs.iter().copied().collect::<HashSet<_>>();
        // Declare the domain these fused nodes live in. While they were
        // (wrongly) in `com.microsoft` this was covered by the import the
        // contrib fusions already record; our own domain has no such
        // coincidence to rely on, and a graph carrying operators from a domain
        // it never imported is not self-consistent.
        graph
            .opset_imports
            .entry(PLUGIN_FUSED_DOMAIN.to_string())
            .or_insert(1);
        graph.replace_node_groups(groups, &graph_outputs);
        let mut guard = self.compiled.lock().map_err(|_| {
            OptimizerError::Fusion(
                "plugin compiled-plan lock was poisoned while installing fused groups".into(),
            )
        })?;
        *guard = Some(plan);
        Ok(())
    }
}

fn is_runtime_input(graph: &Graph, value_id: onnx_runtime_ir::ValueId) -> bool {
    !graph.initializers.contains_key(&value_id)
}

fn boundary_bytes(graph: &Graph, claim: &onnx_runtime_ep_api::SubgraphClaim) -> Option<u64> {
    claim
        .input_values
        .iter()
        .chain(claim.output_values.iter())
        .try_fold(0u64, |total, &value_id| {
            let value = graph.value(value_id);
            if !graph.value_shape_is_known(value_id) || !graph.value_type_is_known(value_id) {
                return None;
            }
            let dims = onnx_runtime_ir::as_static_shape(&value.shape)?;
            let elements = dims
                .iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))?;
            let bytes = if matches!(value.dtype, DataType::String | DataType::Undefined) {
                return None;
            } else {
                value.dtype.checked_storage_bytes(elements)?
            };
            total.checked_add(bytes as u64)
        })
}

fn annotate_plugin_fusion_decisions(
    accepted: &[ClaimDecisionAudit],
    declined: &[ClaimDecisionAudit],
    min_net_score: i64,
) {
    annotate_current_span_with(|| {
        let accepted_nodes: usize = accepted
            .iter()
            .map(|decision| decision.estimate.node_count)
            .sum();
        let declined_nodes: usize = declined
            .iter()
            .map(|decision| decision.estimate.node_count)
            .sum();
        Args::new()
            .with(
                "plugin_claims_total",
                (accepted.len() + declined.len()) as u64,
            )
            .with("plugin_claims_accepted", accepted.len() as u64)
            .with("plugin_claims_declined", declined.len() as u64)
            .with("plugin_claim_nodes_accepted", accepted_nodes as u64)
            .with("plugin_claim_nodes_declined", declined_nodes as u64)
            .with("plugin_fusion_min_net_score", min_net_score)
            .with("plugin_claims_declined_detail", decision_details(declined))
            .with("plugin_claims_accepted_detail", decision_details(accepted))
    });
}

/// Summarise declined claims for the trace.
///
/// Grouped rather than listed. A plugin that claims every constant in a model
/// declines them all for the same reason, and writing one line each made this
/// field 24 KB -- 11% of a whole trace -- to say one thing 96 times. The
/// grouping keeps the counts exact and shows one worked example per reason, so
/// the arithmetic is still checkable without the repetition.
fn decision_details(decisions: &[ClaimDecisionAudit]) -> String {
    if decisions.is_empty() {
        return String::new();
    }
    let mut groups: Vec<(String, usize, String)> = Vec::new();
    for decision in decisions {
        let estimate = &decision.estimate;
        // The shape of the decision, not its exact numbers: claims that differ
        // only in which constant they hold are one group.
        let key = format!(
            "nodes={},constant_foldable={},inputs_on_plugin={},net={}",
            estimate.node_count,
            estimate.constant_foldable,
            estimate.all_boundary_inputs_on_plugin,
            estimate.net_score,
        );
        if let Some(group) = groups.iter_mut().find(|(k, _, _)| *k == key) {
            group.1 += 1;
        } else {
            let example = format!(
                "arity={},runtime_inputs={},boundary_bytes={},benefit={},fixed_overhead={},transfer_penalty={},reason={}",
                estimate.op_arity,
                estimate.runtime_input_count,
                estimate
                    .boundary_bytes
                    .map(|bytes| bytes.to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                estimate.benefit_score,
                estimate.fixed_overhead_score,
                estimate.transfer_penalty_score,
                decision.reason,
            );
            groups.push((key, 1, example));
        }
    }
    groups
        .into_iter()
        .map(|(key, count, example)| format!("{count}x[{key},{example}]"))
        .collect::<Vec<_>>()
        .join("; ")
}

fn shape_graph_for_claim(
    graph: &Graph,
    claim: &onnx_runtime_ep_api::SubgraphClaim,
) -> onnx_runtime_optimizer::Result<Graph> {
    let selected: HashSet<_> = claim.node_ids.iter().copied().collect();
    let mut shape_graph = graph.clone();
    shape_graph.set_inputs(claim.input_values.clone());
    shape_graph.set_outputs(claim.output_values.clone());
    let remove = shape_graph
        .nodes
        .keys()
        .filter(|node| !selected.contains(node))
        .collect::<Vec<_>>();
    shape_graph.remove_nodes(&remove);
    let live_values = shape_graph.values.keys().collect::<HashSet<_>>();
    shape_graph
        .initializers
        .retain(|value, _| live_values.contains(value));
    shape_graph.topological_order().map_err(|err| {
        OptimizerError::Fusion(format!(
            "cannot build shape replay graph for plugin claim because it is not a DAG: {err}"
        ))
    })?;
    Ok(shape_graph)
}

/// Identify the native runtime's synthetic plugin-fused node.
///
/// Callers outside this module use this to avoid treating the shape-replay
/// subgraph metadata as executable host work.
pub fn is_plugin_fused_node(node: &Node) -> bool {
    node.domain == PLUGIN_FUSED_DOMAIN && node.op_type == PLUGIN_FUSED_OP
}

fn is_plugin_fused(node: &Node) -> bool {
    is_plugin_fused_node(node)
}

fn fusion_id(node: &Node) -> onnx_runtime_ep_api::Result<usize> {
    node.attr(FUSION_ID_ATTR)
        .and_then(Attribute::as_int)
        .and_then(|id| usize::try_from(id).ok())
        .ok_or_else(|| {
            EpError::KernelFailed(
                "plugin fused node is missing a valid native_plugin_fusion_id attribute; fix the plugin fusion pass".into(),
            )
        })
}

#[cfg(test)]
mod domain_tests {
    use super::*;
    use onnx_runtime_ep_api::{EpId, SubgraphClaim};
    use onnx_runtime_ir::{DataType, ValueId, static_shape};

    /// The fused operator must live in our domain, not someone else's.
    ///
    /// It is invented here and means "a subgraph some plugin claimed"; it has
    /// no meaning outside this runtime and no specification anyone else wrote.
    /// Putting it in `com.microsoft` — which it was, briefly — asserts a
    /// provenance that does not exist and collides the moment they define an
    /// operator by this name. A reader uses the domain to decide whose
    /// definition applies, so this is a factual claim about the graph.
    #[test]
    fn the_fused_operator_is_in_this_runtimes_own_domain() {
        assert_eq!(PLUGIN_FUSED_DOMAIN, onnx_runtime_ir::RUNTIME_DOMAIN);
        assert_ne!(PLUGIN_FUSED_DOMAIN, "com.microsoft");
        assert_ne!(
            PLUGIN_FUSED_DOMAIN, "",
            "the default ONNX domain belongs to the spec, not to us"
        );
        assert_ne!(PLUGIN_FUSED_DOMAIN, "ai.onnx");
    }

    #[test]
    fn cost_model_declines_tiny_runtime_invariant_claims_by_default() {
        let mut graph = Graph::new();
        let output = graph.create_value(DataType::Float32, static_shape([1]));
        let node = graph.insert_node(Node::new(
            NodeId(0),
            "AnyRuntimeInvariantOp",
            vec![],
            vec![output],
        ));
        let claim = claim(vec![node], vec![], vec![output]);

        let estimate = PluginFusionCostModel::for_device_label("mlx").estimate(&graph, &claim);

        assert!(estimate.constant_foldable);
        assert!(
            estimate.net_score < 0,
            "a one-node runtime-invariant claim should not beat the fixed plugin ABI overhead"
        );
    }

    #[test]
    fn cost_model_accepts_runtime_dependent_single_node_claims_by_default() {
        let mut graph = Graph::new();
        let input = graph.create_value(DataType::Float32, static_shape([1]));
        graph.add_input(input);
        let output = graph.create_value(DataType::Float32, static_shape([1]));
        let node = graph.insert_node(Node::new(
            NodeId(0),
            "AnyRuntimeDependentOp",
            vec![Some(input)],
            vec![output],
        ));
        let claim = claim(vec![node], vec![input], vec![output]);

        let estimate = PluginFusionCostModel::for_device_label("mlx").estimate(&graph, &claim);

        assert!(!estimate.constant_foldable);
        assert!(
            estimate.net_score >= 0,
            "runtime-dependent work carries enough metadata-estimated benefit to fuse by default"
        );
    }

    fn claim(
        node_ids: Vec<NodeId>,
        input_values: Vec<ValueId>,
        output_values: Vec<ValueId>,
    ) -> SubgraphClaim {
        SubgraphClaim {
            ep_id: EpId(0),
            node_ids,
            input_values,
            output_values,
            meta_def: None,
        }
    }
}
