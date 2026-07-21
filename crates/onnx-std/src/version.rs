//! Opset version conversion with composable per-operator adapters (ONNX_RS §10).

use std::collections::HashMap;

use onnx_runtime_ir::{Attribute, DataType, Dim, Graph, Node, NodeId};

use crate::{Model, SchemaRegistry};

type AdapterKey = (String, String, u32);

/// Converts one operator from a source schema version to a target version.
pub trait OpAdapter: Send + Sync {
    /// Source `(domain, op_type, version)`.
    fn source(&self) -> (&str, &str, u32);

    /// Target schema version.
    fn target_version(&self) -> u32;

    /// Adapt `node`, optionally mutating it through `graph`.
    fn adapt(&self, node: &Node, graph: &mut Graph) -> Result<AdaptResult, ConvertError>;
}

/// Result of applying one [`OpAdapter`].
#[derive(Clone, Debug)]
pub enum AdaptResult {
    /// The node is compatible as-is.
    Compatible,
    /// The adapter rewrote the node in place.
    Rewritten,
    /// The node must be replaced by the supplied nodes.
    Decomposed { replacement_nodes: Vec<Node> },
    /// The source node cannot be represented at the target version.
    Incompatible { reason: String },
}

/// One node that prevented a complete conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IncompatibleOp {
    pub node_name: String,
    pub domain: String,
    pub op_type: String,
    pub source_version: u32,
    pub target_version: u32,
    pub reason: String,
}

/// Summary of a conversion attempt.
///
/// Conversion is transactional: when `ops_rejected` is non-zero, the model is
/// left unchanged and the rejected nodes explain which adapters are missing.
#[derive(Clone, Debug, Default)]
pub struct ConvertReport {
    /// Nodes rewritten in place.
    pub ops_converted: usize,
    /// Nodes proven compatible without a rewrite.
    pub ops_unchanged: usize,
    /// Nodes replaced by adapter-provided subgraphs.
    pub ops_decomposed: usize,
    /// Nodes that could not be converted.
    pub ops_rejected: usize,
    /// Details for rejected nodes.
    pub ops_incompatible: Vec<IncompatibleOp>,
    /// Human-readable conversion diagnostics.
    pub messages: Vec<String>,
    pub source_opset: u32,
    pub target_opset: u32,
}

/// Failure to execute a version conversion.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("model has no default-domain opset import")]
    MissingDefaultOpset,
    #[error("opset downgrade from {source_version} to {target} is not supported")]
    DowngradeUnsupported { source_version: u32, target: u32 },
    #[error(
        "invalid adapter for {domain}::{op_type}: target version {target} must exceed source version {source_version}"
    )]
    InvalidAdapter {
        domain: String,
        op_type: String,
        source_version: u32,
        target: u32,
    },
    #[error("adapter for node '{node}' failed: {message}")]
    Adapter { node: String, message: String },
    #[error("adapter for node '{node}' returned an empty decomposition")]
    EmptyDecomposition { node: String },
}

/// Registry and executor for opset conversion adapters.
pub struct VersionConverter {
    adapters: HashMap<AdapterKey, Box<dyn OpAdapter>>,
    schemas: SchemaRegistry,
}

impl VersionConverter {
    /// Create a converter with the built-in adapters.
    pub fn new() -> Self {
        let mut converter = Self::empty();
        converter.register(ReshapeAllowZeroAdapter::new(5));
        converter.register(ReshapeAllowZeroAdapter::new(13));
        converter.register(Softmax12To13Adapter::new("Softmax"));
        converter.register(Softmax12To13Adapter::new("LogSoftmax"));
        converter
    }

    /// Create a converter without operator-specific adapters.
    ///
    /// Schema-proven compatible bumps remain available.
    pub fn empty() -> Self {
        Self {
            adapters: HashMap::new(),
            schemas: SchemaRegistry::builtins(),
        }
    }

    /// Register or replace an adapter for its source tuple.
    pub fn register<A: OpAdapter + 'static>(&mut self, adapter: A) {
        let (domain, op_type, source) = adapter.source();
        self.adapters.insert(
            (
                normalize_domain(domain).to_string(),
                op_type.to_string(),
                source,
            ),
            Box::new(adapter),
        );
    }

    /// List registered `(source, target)` conversions for an operator.
    pub fn available_conversions(&self, domain: &str, op_type: &str) -> Vec<(u32, u32)> {
        let domain = normalize_domain(domain);
        let mut conversions = self
            .adapters
            .iter()
            .filter_map(|((candidate_domain, candidate_op, source), adapter)| {
                (candidate_domain == domain && candidate_op == op_type)
                    .then_some((*source, adapter.target_version()))
            })
            .collect::<Vec<_>>();
        conversions.sort_unstable();
        conversions
    }

    /// Convert the default ONNX domain to `target_opset`.
    ///
    /// A node without an adapter is accepted only when the schema registry has
    /// positive evidence that one schema covers the full conversion interval.
    /// Otherwise it is reported as incompatible and no changes are committed
    /// to `model`.
    pub fn convert(
        &self,
        model: &mut Model,
        target_opset: u32,
    ) -> Result<ConvertReport, ConvertError> {
        let source_opset = default_opset(&model.graph)?;
        if target_opset < source_opset {
            return Err(ConvertError::DowngradeUnsupported {
                source_version: source_opset,
                target: target_opset,
            });
        }

        let mut report = ConvertReport {
            source_opset,
            target_opset,
            ..ConvertReport::default()
        };
        if source_opset == target_opset {
            return Ok(report);
        }

        let mut graph = model.graph.clone();
        self.convert_graph(&mut graph, source_opset, target_opset, &mut report)?;

        if report.ops_rejected == 0 {
            set_default_opset(&mut graph, target_opset);
            model.graph = graph;
        } else {
            report.messages.push(format!(
                "conversion was not applied because {} node(s) were rejected",
                report.ops_rejected
            ));
        }
        Ok(report)
    }

    fn convert_graph(
        &self,
        graph: &mut Graph,
        source_opset: u32,
        target_opset: u32,
        report: &mut ConvertReport,
    ) -> Result<(), ConvertError> {
        let node_ids = graph.nodes.keys().collect::<Vec<_>>();
        for node_id in node_ids {
            self.convert_subgraphs(graph, node_id, source_opset, target_opset, report)?;

            let node = graph.node(node_id).clone();
            if is_default_domain(&node.domain) {
                self.convert_node(graph, node_id, source_opset, target_opset, report)?;
            }
        }
        Ok(())
    }

    fn convert_subgraphs(
        &self,
        graph: &mut Graph,
        node_id: NodeId,
        source_opset: u32,
        target_opset: u32,
        report: &mut ConvertReport,
    ) -> Result<(), ConvertError> {
        let attribute_names = graph
            .node(node_id)
            .attributes
            .keys()
            .cloned()
            .collect::<Vec<_>>();

        for attribute_name in attribute_names {
            let converted = {
                let Some(attribute) = graph.node_mut(node_id).attributes.get_mut(&attribute_name)
                else {
                    continue;
                };
                match attribute {
                    Attribute::Graph(subgraph) => {
                        self.convert_graph(subgraph, source_opset, target_opset, report)?;
                        vec![(attribute_name.clone(), (**subgraph).clone())]
                    }
                    Attribute::Graphs(subgraphs) => {
                        let mut converted = Vec::with_capacity(subgraphs.len());
                        for (index, subgraph) in subgraphs.iter_mut().enumerate() {
                            self.convert_graph(subgraph, source_opset, target_opset, report)?;
                            converted
                                .push((format!("{attribute_name}[{index}]"), subgraph.clone()));
                        }
                        converted
                    }
                    _ => continue,
                }
            };

            for (key, subgraph) in converted {
                graph.subgraphs.insert((node_id, key), subgraph);
            }
        }
        Ok(())
    }

    fn convert_node(
        &self,
        graph: &mut Graph,
        node_id: NodeId,
        source_opset: u32,
        target_opset: u32,
        report: &mut ConvertReport,
    ) -> Result<(), ConvertError> {
        let original = graph.node(node_id).clone();
        let domain = normalize_domain(&original.domain);
        let mut current = source_opset;
        let mut rewritten = false;
        let mut decomposed = false;

        while current < target_opset {
            let key = (domain.to_string(), original.op_type.clone(), current);
            let Some(adapter) = self.adapters.get(&key) else {
                if self.schema_compatible(&original.op_type, domain, current, target_opset) {
                    break;
                }
                reject(
                    report,
                    &original,
                    source_opset,
                    target_opset,
                    format!(
                        "no adapter from opset {current} to {target_opset}, and schemas do not prove compatibility"
                    ),
                );
                return Ok(());
            };

            let next = adapter.target_version();
            if next <= current || next > target_opset {
                return Err(ConvertError::InvalidAdapter {
                    domain: domain.to_string(),
                    op_type: original.op_type.clone(),
                    source_version: current,
                    target: next,
                });
            }

            let node = graph.node(node_id).clone();
            match adapter
                .adapt(&node, graph)
                .map_err(|error| ConvertError::Adapter {
                    node: display_node(&original),
                    message: error.to_string(),
                })? {
                AdaptResult::Compatible => {}
                AdaptResult::Rewritten => rewritten = true,
                AdaptResult::Decomposed { replacement_nodes } => {
                    if replacement_nodes.is_empty() {
                        return Err(ConvertError::EmptyDecomposition {
                            node: display_node(&original),
                        });
                    }
                    let mut replacements = replacement_nodes.into_iter();
                    graph.replace_node(node_id, replacements.next().expect("checked non-empty"));
                    for replacement in replacements {
                        graph.insert_node(replacement);
                    }
                    decomposed = true;
                }
                AdaptResult::Incompatible { reason } => {
                    reject(report, &original, source_opset, target_opset, reason);
                    return Ok(());
                }
            }
            current = next;

            if decomposed && current < target_opset {
                reject(
                    report,
                    &original,
                    source_opset,
                    target_opset,
                    "adapter decomposed the node before the final target opset".to_string(),
                );
                return Ok(());
            }
        }

        if decomposed {
            report.ops_decomposed += 1;
        } else if rewritten {
            report.ops_converted += 1;
        } else {
            report.ops_unchanged += 1;
        }
        Ok(())
    }

    fn schema_compatible(&self, op_type: &str, domain: &str, source: u32, target: u32) -> bool {
        let Some(source_schema) = self.schemas.lookup(op_type, domain, u64::from(source)) else {
            return false;
        };
        let Some(target_schema) = self.schemas.lookup(op_type, domain, u64::from(target)) else {
            return false;
        };
        if source_schema.since_version != target_schema.since_version {
            return false;
        }

        let target = u64::from(target);
        source_schema
            .until_version
            .is_some_and(|until| target <= until)
            || self
                .schemas
                .iter()
                .filter(|schema| {
                    schema.name == op_type
                        && normalize_domain(&schema.domain) == normalize_domain(domain)
                        && schema.since_version > source_schema.since_version
                })
                .map(|schema| schema.since_version)
                .min()
                .is_some_and(|next_version| target < next_version)
    }
}

impl Default for VersionConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Reshape opset 14 introduced `allowzero` with a default of zero. Materializing
/// that default makes the old zero-copy semantics explicit.
struct ReshapeAllowZeroAdapter {
    source: u32,
}

impl ReshapeAllowZeroAdapter {
    const fn new(source: u32) -> Self {
        Self { source }
    }
}

impl OpAdapter for ReshapeAllowZeroAdapter {
    fn source(&self) -> (&str, &str, u32) {
        ("", "Reshape", self.source)
    }

    fn target_version(&self) -> u32 {
        14
    }

    fn adapt(&self, node: &Node, graph: &mut Graph) -> Result<AdaptResult, ConvertError> {
        if node.attributes.contains_key("allowzero") {
            return Ok(AdaptResult::Compatible);
        }
        graph
            .node_mut(node.id)
            .attributes
            .insert("allowzero".to_string(), Attribute::Int(0));
        Ok(AdaptResult::Rewritten)
    }
}

/// Opset 13 changed Softmax and LogSoftmax from flattening at `axis` to
/// operating on exactly one axis. This is the same Shape/Flatten/op/Reshape
/// rewrite used by the official ONNX 12→13 adapter.
struct Softmax12To13Adapter {
    op_type: &'static str,
}

impl Softmax12To13Adapter {
    const fn new(op_type: &'static str) -> Self {
        Self { op_type }
    }
}

impl OpAdapter for Softmax12To13Adapter {
    fn source(&self) -> (&str, &str, u32) {
        ("", self.op_type, 12)
    }

    fn target_version(&self) -> u32 {
        13
    }

    fn adapt(&self, node: &Node, graph: &mut Graph) -> Result<AdaptResult, ConvertError> {
        if node.inputs.len() != 1 || node.outputs.len() != 1 {
            return Ok(AdaptResult::Incompatible {
                reason: format!("{} requires exactly one input and one output", self.op_type),
            });
        }
        let Some(input) = node.inputs.first().copied().flatten() else {
            return Ok(AdaptResult::Incompatible {
                reason: format!("{} requires one present input", self.op_type),
            });
        };
        let Some(&output) = node.outputs.first() else {
            return Ok(AdaptResult::Incompatible {
                reason: format!("{} requires one output", self.op_type),
            });
        };
        if !graph.value_shape_is_known(input) {
            return Ok(AdaptResult::Incompatible {
                reason: format!(
                    "{} 12→13 conversion requires a known input rank",
                    self.op_type
                ),
            });
        }
        let rank = graph.value(input).rank();
        let Ok(rank_i64) = i64::try_from(rank) else {
            return Ok(AdaptResult::Incompatible {
                reason: format!(
                    "{} input rank exceeds the supported integer range",
                    self.op_type
                ),
            });
        };
        if rank == 0 {
            return Ok(AdaptResult::Incompatible {
                reason: format!("{} input must have rank at least 1", self.op_type),
            });
        }
        let old_axis = node
            .attributes
            .get("axis")
            .and_then(Attribute::as_int)
            .unwrap_or(1);
        let normalized_axis = if old_axis < 0 {
            old_axis.checked_add(rank_i64)
        } else {
            Some(old_axis)
        };
        let Some(normalized_axis) = normalized_axis.filter(|axis| (0..rank_i64).contains(axis))
        else {
            return Ok(AdaptResult::Incompatible {
                reason: format!(
                    "{} axis {old_axis} is outside [-{rank}, {rank})",
                    self.op_type
                ),
            });
        };

        if normalized_axis == rank_i64 - 1 {
            graph
                .node_mut(node.id)
                .attributes
                .insert("axis".into(), Attribute::Int(-1));
            return Ok(AdaptResult::Rewritten);
        }

        let input_value = graph.value(input);
        let dtype = input_value.dtype;
        let shape_value = graph.create_value(DataType::Int64, vec![Dim::Static(rank)]);
        let flattened = graph.create_value(dtype, Vec::new());
        let intermediate = graph.create_value(dtype, Vec::new());
        graph.mark_value_shape_unknown(flattened);
        graph.mark_value_shape_unknown(intermediate);

        let helper_name =
            |suffix: &str| (!node.name.is_empty()).then(|| format!("{}_{}", node.name, suffix));

        let mut shape = Node::new(NodeId(0), "Shape", vec![Some(input)], vec![shape_value]);
        shape.name = helper_name("shape").unwrap_or_default();
        shape.device = node.device;

        let mut flatten = Node::new(NodeId(0), "Flatten", vec![Some(input)], vec![flattened]);
        flatten.name = helper_name("flatten").unwrap_or_default();
        flatten
            .attributes
            .insert("axis".into(), Attribute::Int(normalized_axis));
        flatten.device = node.device;

        let mut softmax = node.clone();
        softmax.id = NodeId(0);
        softmax.inputs = vec![Some(flattened)];
        softmax.outputs = vec![intermediate];
        softmax.attributes.insert("axis".into(), Attribute::Int(-1));

        let mut reshape = Node::new(
            NodeId(0),
            "Reshape",
            vec![Some(intermediate), Some(shape_value)],
            vec![output],
        );
        reshape.name = helper_name("reshape").unwrap_or_default();
        reshape.device = node.device;

        Ok(AdaptResult::Decomposed {
            replacement_nodes: vec![shape, flatten, softmax, reshape],
        })
    }
}

fn default_opset(graph: &Graph) -> Result<u32, ConvertError> {
    graph
        .opset_imports
        .get("")
        .or_else(|| graph.opset_imports.get("ai.onnx"))
        .copied()
        .ok_or(ConvertError::MissingDefaultOpset)?
        .try_into()
        .map_err(|_| ConvertError::MissingDefaultOpset)
}

fn set_default_opset(graph: &mut Graph, target: u32) {
    graph.opset_imports.remove("ai.onnx");
    graph.opset_imports.insert(String::new(), u64::from(target));
}

fn is_default_domain(domain: &str) -> bool {
    domain.is_empty() || domain == "ai.onnx"
}

fn normalize_domain(domain: &str) -> &str {
    if is_default_domain(domain) {
        "ai.onnx"
    } else {
        domain
    }
}

fn display_node(node: &Node) -> String {
    if node.name.is_empty() {
        node.op_type.clone()
    } else {
        node.name.clone()
    }
}

fn reject(
    report: &mut ConvertReport,
    node: &Node,
    source_version: u32,
    target_version: u32,
    reason: String,
) {
    let detail = IncompatibleOp {
        node_name: node.name.clone(),
        domain: node.domain.clone(),
        op_type: node.op_type.clone(),
        source_version,
        target_version,
        reason,
    };
    report
        .messages
        .push(format!("{}: {}", display_node(node), detail.reason));
    report.ops_rejected += 1;
    report.ops_incompatible.push(detail);
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{DataType, Node, static_shape};

    struct If13To14Adapter;

    impl OpAdapter for If13To14Adapter {
        fn source(&self) -> (&str, &str, u32) {
            ("", "If", 13)
        }

        fn target_version(&self) -> u32 {
            14
        }

        fn adapt(&self, _node: &Node, _graph: &mut Graph) -> Result<AdaptResult, ConvertError> {
            Ok(AdaptResult::Compatible)
        }
    }

    fn unary_model(op_type: &str, opset: u32) -> Model {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), u64::from(opset));
        let input = graph.create_named_value("X", DataType::Float32, static_shape([2, 3]));
        let output = graph.create_named_value("Y", DataType::Float32, static_shape([2, 3]));
        graph.add_input(input);
        graph.add_output(output);
        let mut node = Node::new(NodeId(0), op_type, vec![Some(input)], vec![output]);
        node.name = format!("{op_type}_0");
        graph.insert_node(node);
        Model::new(graph)
    }

    fn unary_graph(op_type: &str) -> Graph {
        let mut graph = Graph::new();
        let input = graph.create_named_value("branch_X", DataType::Float32, static_shape([2, 3]));
        let output = graph.create_named_value("branch_Y", DataType::Float32, static_shape([2, 3]));
        graph.add_input(input);
        graph.add_output(output);
        let mut node = Node::new(NodeId(0), op_type, vec![Some(input)], vec![output]);
        node.name = format!("{op_type}_branch");
        graph.insert_node(node);
        graph
    }

    fn if_model(branch_op: &str, include_root_reshape: bool) -> Model {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 13);

        if include_root_reshape {
            let input = graph.create_named_value("root_X", DataType::Float32, static_shape([2, 3]));
            let output =
                graph.create_named_value("root_Y", DataType::Float32, static_shape([2, 3]));
            graph.add_input(input);
            let mut reshape = Node::new(NodeId(0), "Reshape", vec![Some(input)], vec![output]);
            reshape.name = "Reshape_root".to_string();
            graph.insert_node(reshape);
        }

        let condition = graph.create_named_value("condition", DataType::Bool, static_shape([]));
        let output = graph.create_named_value("if_Y", DataType::Float32, static_shape([2, 3]));
        graph.add_input(condition);
        graph.add_output(output);

        let branch = unary_graph(branch_op);
        let mut if_node = Node::new(NodeId(0), "If", vec![Some(condition)], vec![output]);
        if_node.name = "If_0".to_string();
        if_node.attributes.insert(
            "then_branch".to_string(),
            Attribute::Graph(Box::new(branch.clone())),
        );
        let if_node_id = graph.insert_node(if_node);
        graph
            .subgraphs
            .insert((if_node_id, "then_branch".to_string()), branch);

        Model::new(graph)
    }

    #[test]
    fn schema_compatible_bump_requires_a_known_later_boundary() {
        let mut model = unary_model("StableOp", 1);
        let before = model.graph.nodes.values().next().unwrap().clone();
        let mut converter = VersionConverter::empty();
        converter
            .schemas
            .load_yaml(
                r#"
domain: ""
name: StableOp
since_version: 1
"#,
            )
            .unwrap();
        converter
            .schemas
            .load_yaml(
                r#"
domain: ""
name: StableOp
since_version: 5
"#,
            )
            .unwrap();

        let report = converter.convert(&mut model, 4).unwrap();

        let after = model.graph.nodes.values().next().unwrap();
        assert_eq!(model.graph.opset_imports[""], 4);
        assert_eq!(after.op_type, before.op_type);
        assert_eq!(after.inputs, before.inputs);
        assert_eq!(after.outputs, before.outputs);
        assert_eq!(report.ops_unchanged, 1);
        assert_eq!(report.ops_converted, 0);
        assert_eq!(report.ops_rejected, 0);
    }

    #[test]
    fn nested_subgraph_nodes_are_converted_and_indexes_stay_in_sync() {
        let mut model = if_model("Reshape", false);
        let mut converter = VersionConverter::new();
        converter.register(If13To14Adapter);

        let report = converter.convert(&mut model, 14).unwrap();

        assert_eq!(model.graph.opset_imports[""], 14);
        assert_eq!(report.ops_converted, 1);
        assert_eq!(report.ops_unchanged, 1);
        assert_eq!(report.ops_rejected, 0);

        let if_node = model
            .graph
            .nodes
            .values()
            .find(|node| node.op_type == "If")
            .unwrap();
        let Attribute::Graph(branch) = &if_node.attributes["then_branch"] else {
            panic!("then_branch must be a graph");
        };
        assert_eq!(
            branch.nodes.values().next().unwrap().attributes["allowzero"].as_int(),
            Some(0)
        );
        let indexed = &model.graph.subgraphs[&(if_node.id, "then_branch".to_string())];
        assert_eq!(
            indexed.nodes.values().next().unwrap().attributes["allowzero"].as_int(),
            Some(0)
        );
    }

    #[test]
    fn nested_rejection_rolls_back_root_rewrites_and_opset_bump() {
        let mut model = if_model("UnknownChangedOp", true);
        let mut converter = VersionConverter::new();
        converter.register(If13To14Adapter);

        let report = converter.convert(&mut model, 14).unwrap();

        assert_eq!(model.graph.opset_imports[""], 13);
        assert_eq!(report.ops_rejected, 1);
        assert_eq!(
            report.ops_incompatible[0].node_name,
            "UnknownChangedOp_branch"
        );
        let root_reshape = model
            .graph
            .nodes
            .values()
            .find(|node| node.name == "Reshape_root")
            .unwrap();
        assert!(!root_reshape.attributes.contains_key("allowzero"));
    }

    #[test]
    fn target_beyond_known_schema_history_is_rejected() {
        let mut model = unary_model("Conv", 11);

        let report = VersionConverter::new().convert(&mut model, 22).unwrap();

        assert_eq!(model.graph.opset_imports[""], 11);
        assert_eq!(report.ops_rejected, 1);
        assert_eq!(report.ops_incompatible[0].op_type, "Conv");
        assert!(
            report.ops_incompatible[0]
                .reason
                .contains("schemas do not prove compatibility")
        );
    }

    #[test]
    fn reshape_adapter_materializes_allowzero_default() {
        let mut model = unary_model("Reshape", 13);

        let report = VersionConverter::new().convert(&mut model, 14).unwrap();

        let node = model.graph.nodes.values().next().unwrap();
        assert_eq!(
            node.attributes.get("allowzero").and_then(Attribute::as_int),
            Some(0)
        );
        assert_eq!(model.graph.opset_imports[""], 14);
        assert_eq!(report.ops_converted, 1);
        assert_eq!(report.ops_unchanged, 0);
        assert_eq!(report.ops_rejected, 0);
    }

    #[test]
    fn softmax_and_log_softmax_12_to_13_adapters_are_registered() {
        let converter = VersionConverter::new();
        assert_eq!(converter.available_conversions("", "Softmax"), [(12, 13)]);
        assert_eq!(
            converter.available_conversions("ai.onnx", "LogSoftmax"),
            [(12, 13)]
        );
    }

    #[test]
    fn softmax_last_axis_is_rewritten_without_decomposition() {
        let mut model = unary_model("Softmax", 12);

        let report = VersionConverter::new().convert(&mut model, 13).unwrap();

        let node = model.graph.nodes.values().next().unwrap();
        assert_eq!(node.attributes["axis"].as_int(), Some(-1));
        assert_eq!(model.graph.opset_imports[""], 13);
        assert_eq!(report.ops_converted, 1);
        assert_eq!(report.ops_decomposed, 0);
        assert_eq!(report.ops_rejected, 0);
    }

    #[test]
    fn softmax_non_last_axis_uses_official_flatten_reshape_rewrite() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 12);
        let input = graph.create_named_value("X", DataType::Float32, static_shape([2, 3, 4]));
        let output = graph.create_named_value("Y", DataType::Float32, static_shape([2, 3, 4]));
        graph.add_input(input);
        graph.add_output(output);
        let mut softmax = Node::new(NodeId(0), "Softmax", vec![Some(input)], vec![output]);
        softmax.name = "softmax".into();
        softmax.attributes.insert("axis".into(), Attribute::Int(1));
        graph.insert_node(softmax);
        let mut model = Model::new(graph);

        let report = VersionConverter::new().convert(&mut model, 13).unwrap();

        assert_eq!(model.graph.opset_imports[""], 13);
        assert_eq!(report.ops_decomposed, 1);
        assert_eq!(report.ops_converted, 0);
        assert_eq!(report.ops_rejected, 0);
        assert_eq!(model.graph.num_nodes(), 4);
        model.graph.validate().unwrap();

        let shape = model
            .graph
            .nodes
            .values()
            .find(|node| node.op_type == "Shape")
            .unwrap();
        let flatten = model
            .graph
            .nodes
            .values()
            .find(|node| node.op_type == "Flatten")
            .unwrap();
        let converted = model
            .graph
            .nodes
            .values()
            .find(|node| node.op_type == "Softmax")
            .unwrap();
        let reshape = model
            .graph
            .nodes
            .values()
            .find(|node| node.op_type == "Reshape")
            .unwrap();

        assert_eq!(flatten.attributes["axis"].as_int(), Some(1));
        assert_eq!(converted.attributes["axis"].as_int(), Some(-1));
        assert_eq!(
            model.graph.value(flatten.inputs[0].unwrap()).id,
            shape.inputs[0].unwrap()
        );
        assert_eq!(
            model.graph.value(converted.inputs[0].unwrap()).producer,
            Some(flatten.id)
        );
        assert_eq!(
            model.graph.value(reshape.inputs[0].unwrap()).producer,
            Some(converted.id)
        );
        assert_eq!(
            model.graph.value(reshape.inputs[1].unwrap()).producer,
            Some(shape.id)
        );
        assert_eq!(model.graph.value(output).producer, Some(reshape.id));
        let flattened_output = flatten.outputs[0];
        let converted_output = converted.outputs[0];

        crate::infer_shapes(&mut model).unwrap();
        assert_eq!(
            model.graph.value(flattened_output).shape,
            static_shape([2, 12])
        );
        assert_eq!(
            model.graph.value(converted_output).shape,
            static_shape([2, 12])
        );
        assert_eq!(model.graph.value(output).shape, static_shape([2, 3, 4]));
    }

    #[test]
    fn missing_adapter_is_reported_and_model_is_unchanged() {
        let mut model = unary_model("UnknownChangedOp", 10);

        let report = VersionConverter::new().convert(&mut model, 11).unwrap();

        assert_eq!(model.graph.opset_imports[""], 10);
        assert_eq!(report.ops_rejected, 1);
        assert_eq!(report.ops_incompatible.len(), 1);
        assert!(report.ops_incompatible[0].reason.contains("no adapter"));
        assert!(
            report
                .messages
                .iter()
                .any(|message| message.contains("not applied"))
        );
    }

    #[test]
    fn downgrade_returns_clear_error() {
        let mut model = unary_model("Relu", 14);

        let error = VersionConverter::new().convert(&mut model, 13).unwrap_err();

        assert!(matches!(
            error,
            ConvertError::DowngradeUnsupported {
                source_version: 14,
                target: 13
            }
        ));
        assert_eq!(model.graph.opset_imports[""], 14);
    }
}
