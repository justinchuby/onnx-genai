//! Built-in validation rules (ONNX_RS §8.2).
//!
//! * [`MissingOpsetImportRule`] — every operator domain used by a node must have
//!   a matching `opset_import` (§8.2 "IR rules").
//! * [`DuplicateValueNameRule`] — no two values may share a name (§8.2
//!   "structural rules": unique value names / single producer).
//! * [`GraphAcyclicRule`] — the dataflow graph must be acyclic (§8.2).
//! * [`SchemaNodeConformsRule`] — nodes match their resolved operator schema.
//! * [`InputOutputDeclaredRule`] — graph inputs and outputs are named, live values.
//! * [`NoUnconnectedNodesRule`] — node inputs resolve to graph sources or producers.
//! * [`TypeConstraintSatisfiedRule`] — node value types satisfy schema constraints.
//! * [`InitializerTypeMatchesDeclaredRule`] — initializer and value dtypes agree.
//! * [`IrVersionSupportedRule`] — the model declares a valid ONNX IR version.
//! * [`MultiDeviceConfigurationRule`] — IR v11+ distributed annotations are
//!   internally consistent.

mod graph_topology;
mod ir_version_functions;
mod multi_device;
mod schema_types;
mod tensor_sparse_payloads;

pub use graph_topology::{
    DuplicateValueNameRule, GraphAcyclicRule, InputOutputDeclaredRule, MissingOpsetImportRule,
    NoUnconnectedNodesRule,
};
pub use ir_version_functions::{
    FunctionProtoValidityRule, IrVersionFeatureRule, IrVersionSupportedRule,
};
pub use multi_device::MultiDeviceConfigurationRule;
pub use schema_types::{
    AttributeProtoValidityRule, InitializerTypeMatchesDeclaredRule, MetadataKeysUniqueRule,
    ProtoTypeValidityRule, SchemaNodeConformsRule, TypeConstraintSatisfiedRule,
};
pub use tensor_sparse_payloads::{SparseTensorValidityRule, TensorPayloadValidityRule};

use super::{Severity, Violation, ViolationLocation};
use onnx_runtime_ir::ValueId;
use onnx_runtime_loader::proto::onnx::NodeProto;

/// Normalise a node/opset domain: the empty string and `"ai.onnx"` both denote
/// the default ONNX domain.
fn normalize_domain(domain: &str) -> &str {
    if domain.is_empty() { "ai.onnx" } else { domain }
}

fn model_violation(rule_id: &str, message: impl Into<String>) -> Violation {
    Violation {
        rule_id: rule_id.to_string(),
        severity: Severity::Error,
        message: message.into(),
        location: ViolationLocation::Model,
    }
}

fn proto_node_violation(
    rule_id: &str,
    graph_name: &str,
    node: &NodeProto,
    message: impl Into<String>,
) -> Violation {
    Violation {
        rule_id: rule_id.to_string(),
        severity: Severity::Error,
        message: message.into(),
        location: ViolationLocation::Node {
            graph_name: graph_name.to_string(),
            node_name: if node.name.is_empty() {
                format!("<{}>", node.op_type)
            } else {
                node.name.clone()
            },
        },
    }
}

fn node_violation(
    rule_id: &str,
    graph_name: &str,
    node: &onnx_runtime_ir::Node,
    message: String,
) -> Violation {
    Violation {
        rule_id: rule_id.to_string(),
        severity: Severity::Error,
        message,
        location: ViolationLocation::Node {
            graph_name: graph_name.to_string(),
            node_name: node_label(node),
        },
    }
}

/// A stable display label for a node: its name, or `<op_type#id>` if anonymous.
fn node_label(node: &onnx_runtime_ir::Node) -> String {
    if node.name.is_empty() {
        format!("<{}#{}>", node.op_type, node.id.0)
    } else {
        node.name.clone()
    }
}

fn value_label(value_id: ValueId, name: Option<&str>) -> String {
    match name {
        Some(name) if !name.is_empty() => name.to_string(),
        _ => format!("<value#{}>", value_id.0),
    }
}

fn arity_message(kind: &str, actual: usize, min: usize, max: Option<usize>) -> String {
    match max {
        Some(max) if min == max => {
            format!("has {actual} {kind}s but schema requires exactly {min}")
        }
        Some(max) => format!("has {actual} {kind}s but schema permits {min}..={max}"),
        None => format!("has {actual} {kind}s but schema requires at least {min}"),
    }
}

#[cfg(test)]
mod tests {
    use super::schema_types::{check_attribute_proto, check_type_proto, check_value_type};
    use super::tensor_sparse_payloads::{check_sparse_tensor, check_tensor_payload};
    use super::*;
    use crate::check::{ValidationContext, ValidationRule};
    use crate::model::Model;
    use crate::schema::SchemaRegistry;
    use onnx_runtime_ir::{
        Attribute, DataType, Dim, Graph, Node, NodeId, TensorData, WeightRef, static_shape,
    };
    use onnx_runtime_loader::proto::onnx::{
        AttributeProto, FunctionProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
        SparseTensorProto, StringStringEntryProto, TensorProto, TypeProto, ValueInfoProto,
        attribute_proto, tensor_proto, type_proto,
    };

    fn if_model(subgraph: Graph, output_count: usize, include_else_branch: bool) -> Model {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let cond = graph.create_named_value("cond", DataType::Bool, static_shape([]));
        let outputs = (0..output_count)
            .map(|index| {
                graph.create_named_value(
                    format!("out{index}"),
                    DataType::Float32,
                    static_shape([1]),
                )
            })
            .collect::<Vec<_>>();
        graph.add_input(cond);
        let mut node = Node::new(NodeId(0), "If", vec![Some(cond)], outputs.clone());
        node.attributes.insert(
            "then_branch".into(),
            Attribute::Graph(Box::new(subgraph.clone())),
        );
        if include_else_branch {
            node.attributes.insert(
                "else_branch".into(),
                Attribute::Graph(Box::new(subgraph.clone())),
            );
        }
        let node_id = graph.insert_node(node);
        graph
            .subgraphs
            .insert((node_id, "then_branch".into()), subgraph.clone());
        if include_else_branch {
            graph
                .subgraphs
                .insert((node_id, "else_branch".into()), subgraph);
        }
        for output in outputs {
            graph.add_output(output);
        }
        Model::new(graph)
    }

    fn nested_model(subgraph: Graph) -> Model {
        if_model(subgraph, 1, true)
    }

    fn one_node_model(op_type: &str, inputs: usize, outputs: usize) -> Model {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let input_ids = (0..inputs)
            .map(|index| {
                graph.create_named_value(
                    format!("input{index}"),
                    DataType::Float32,
                    static_shape([1]),
                )
            })
            .collect::<Vec<_>>();
        let output_ids = (0..outputs)
            .map(|index| {
                graph.create_named_value(
                    format!("output{index}"),
                    DataType::Float32,
                    static_shape([1]),
                )
            })
            .collect::<Vec<_>>();
        graph.insert_node(Node::new(
            NodeId(0),
            op_type,
            input_ids.iter().copied().map(Some).collect(),
            output_ids.clone(),
        ));
        for input in input_ids {
            graph.add_input(input);
        }
        for output in output_ids {
            graph.add_output(output);
        }
        Model::new(graph)
    }

    fn assert_error(rule_id: &str, violations: &[Violation], location: ViolationLocation) {
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert_eq!(violations[0].rule_id, rule_id);
        assert_eq!(violations[0].severity, Severity::Error);
        assert_eq!(violations[0].location, location);
    }

    fn retained_model(proto: ModelProto) -> Model {
        retained_model_at(proto, 13)
    }

    fn retained_model_at(mut proto: ModelProto, ir_version: i64) -> Model {
        proto.ir_version = ir_version;
        if ir_version >= 3 && proto.opset_import.is_empty() {
            proto.opset_import.push(OperatorSetIdProto {
                domain: String::new(),
                version: 24,
            });
        }
        if proto.graph.is_none() {
            proto.graph = Some(GraphProto {
                name: "graph".into(),
                ..Default::default()
            });
        }
        Model::from_proto(proto).unwrap()
    }

    fn function(name: &str, nodes: Vec<NodeProto>) -> FunctionProto {
        FunctionProto {
            name: name.into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            node: nodes,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 24,
            }],
            domain: "local.test".into(),
            ..Default::default()
        }
    }

    fn local_function_call_model(function: FunctionProto, call: NodeProto) -> Model {
        let graph_inputs = call
            .input
            .iter()
            .filter(|name| !name.is_empty())
            .map(|name| ValueInfoProto {
                name: name.clone(),
                ..Default::default()
            })
            .collect();
        let graph_outputs = call
            .output
            .iter()
            .filter(|name| !name.is_empty())
            .map(|name| ValueInfoProto {
                name: name.clone(),
                ..Default::default()
            })
            .collect();
        retained_model(ModelProto {
            opset_import: vec![
                OperatorSetIdProto {
                    domain: String::new(),
                    version: 24,
                },
                OperatorSetIdProto {
                    domain: "local.test".into(),
                    version: 1,
                },
            ],
            graph: Some(GraphProto {
                name: "graph".into(),
                input: graph_inputs,
                output: graph_outputs,
                node: vec![call],
                ..Default::default()
            }),
            functions: vec![function],
            ..Default::default()
        })
    }

    #[test]
    fn metadata_rule_rejects_duplicate_keys() {
        let model = retained_model(ModelProto {
            metadata_props: vec![
                StringStringEntryProto {
                    key: "owner".into(),
                    value: "one".into(),
                },
                StringStringEntryProto {
                    key: "owner".into(),
                    value: "two".into(),
                },
            ],
            ..Default::default()
        });
        let violations = MetadataKeysUniqueRule.check(&model, &ValidationContext::default());
        assert_eq!(violations.len(), 1);
        assert!(violations[0].message.contains("duplicate key 'owner'"));
    }

    #[test]
    fn attribute_rule_checks_discriminator_union_and_references() {
        let location = ViolationLocation::Model;
        let mut violations = Vec::new();
        check_attribute_proto(
            &AttributeProto {
                name: "weight".into(),
                r#type: attribute_proto::AttributeType::Tensor as i32,
                i: 7,
                ..Default::default()
            },
            location.clone(),
            13,
            "test",
            &mut violations,
        );
        assert_eq!(violations.len(), 2, "{violations:?}");

        violations.clear();
        check_attribute_proto(
            &AttributeProto {
                name: "alpha".into(),
                ref_attr_name: "alpha".into(),
                r#type: attribute_proto::AttributeType::Float as i32,
                ..Default::default()
            },
            location,
            13,
            "test",
            &mut violations,
        );
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn attribute_type_is_required_starting_with_ir_v2() {
        let attribute = AttributeProto {
            name: "axis".into(),
            i: 1,
            ..Default::default()
        };
        let mut violations = Vec::new();
        check_attribute_proto(
            &attribute,
            ViolationLocation::Model,
            1,
            "test",
            &mut violations,
        );
        assert!(violations.is_empty(), "{violations:?}");

        check_attribute_proto(
            &attribute,
            ViolationLocation::Model,
            2,
            "test",
            &mut violations,
        );
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(violations[0].message.contains("undefined discriminator"));
    }

    #[test]
    fn type_rule_accepts_opaque_and_rejects_invalid_containers() {
        let mut violations = Vec::new();
        check_type_proto(
            &TypeProto {
                value: Some(type_proto::Value::OpaqueType(type_proto::Opaque {
                    domain: "example".into(),
                    name: "State".into(),
                })),
                ..Default::default()
            },
            ViolationLocation::Model,
            false,
            "test",
            &mut violations,
        );
        assert!(violations.is_empty());

        check_type_proto(
            &TypeProto {
                value: Some(type_proto::Value::MapType(Box::new(type_proto::Map {
                    key_type: tensor_proto::DataType::Float as i32,
                    value_type: None,
                }))),
                ..Default::default()
            },
            ViolationLocation::Model,
            false,
            "test",
            &mut violations,
        );
        assert_eq!(violations.len(), 2, "{violations:?}");
    }

    #[test]
    fn top_level_tensor_requires_shape_but_nested_tensor_does_not() {
        let tensor_without_shape = TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: tensor_proto::DataType::Float as i32,
                shape: None,
            })),
            ..Default::default()
        };
        let mut violations = Vec::new();
        check_value_type(
            &ValueInfoProto {
                name: "input".into(),
                r#type: Some(tensor_without_shape.clone()),
                ..Default::default()
            },
            true,
            "test",
            &mut violations,
        );
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(violations[0].message.contains("must declare a shape"));

        violations.clear();
        check_value_type(
            &ValueInfoProto {
                name: "sequence".into(),
                r#type: Some(TypeProto {
                    value: Some(type_proto::Value::SequenceType(Box::new(
                        type_proto::Sequence {
                            elem_type: Some(Box::new(tensor_without_shape)),
                        },
                    ))),
                    ..Default::default()
                }),
                ..Default::default()
            },
            true,
            "test",
            &mut violations,
        );
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn tensor_payload_rule_checks_sizes_and_external_offsets() {
        let mut violations = Vec::new();
        check_tensor_payload(
            &TensorProto {
                dims: vec![2],
                data_type: tensor_proto::DataType::Float as i32,
                raw_data: vec![0; 4],
                name: "short".into(),
                ..Default::default()
            },
            "test",
            &mut violations,
        );
        assert_eq!(violations.len(), 1);
        assert!(violations[0].message.contains("8 are required"));

        violations.clear();
        check_tensor_payload(
            &TensorProto {
                dims: vec![1],
                data_type: tensor_proto::DataType::Float as i32,
                data_location: tensor_proto::DataLocation::External as i32,
                external_data: vec![
                    StringStringEntryProto {
                        key: "location".into(),
                        value: "weights.bin".into(),
                    },
                    StringStringEntryProto {
                        key: "offset".into(),
                        value: usize::MAX.to_string(),
                    },
                    StringStringEntryProto {
                        key: "length".into(),
                        value: "1".into(),
                    },
                ],
                ..Default::default()
            },
            "test",
            &mut violations,
        );
        assert!(
            violations
                .iter()
                .any(|violation| violation.message.contains("offset + length overflows"))
        );
    }

    #[test]
    fn tensor_payload_rule_accepts_unused_packed_bits_but_checks_length() {
        let mut tensor = TensorProto {
            dims: vec![3],
            data_type: tensor_proto::DataType::Uint4 as i32,
            raw_data: vec![0x21, 0xf3],
            ..Default::default()
        };
        let mut violations = Vec::new();
        check_tensor_payload(&tensor, "test", &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        tensor.raw_data.pop();
        check_tensor_payload(&tensor, "test", &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.message.contains("but 2 are required")),
            "{violations:?}"
        );
    }

    #[test]
    fn sparse_tensor_rule_checks_order_and_bounds() {
        let sparse = SparseTensorProto {
            values: Some(TensorProto {
                dims: vec![2],
                data_type: tensor_proto::DataType::Float as i32,
                float_data: vec![1.0, 2.0],
                name: "sparse".into(),
                ..Default::default()
            }),
            indices: Some(TensorProto {
                dims: vec![2],
                data_type: tensor_proto::DataType::Int64 as i32,
                int64_data: vec![1, 1],
                ..Default::default()
            }),
            dims: vec![2, 2],
        };
        let mut violations = Vec::new();
        check_sparse_tensor(&sparse, true, "test", &mut violations);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(violations[0].message.contains("strictly increasing"));

        let mut valid = sparse;
        valid.indices.as_mut().unwrap().int64_data = vec![0, 3];
        violations.clear();
        check_sparse_tensor(&valid, true, "test", &mut violations);
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn sparse_tensor_indices_are_optional_only_for_zero_nnz() {
        let mut sparse = SparseTensorProto {
            values: Some(TensorProto {
                dims: vec![0],
                data_type: tensor_proto::DataType::Float as i32,
                name: "empty_sparse".into(),
                ..Default::default()
            }),
            indices: None,
            dims: vec![2, 3],
        };
        let mut violations = Vec::new();
        check_sparse_tensor(&sparse, true, "test", &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        sparse.values.as_mut().unwrap().dims = vec![1];
        check_sparse_tensor(&sparse, true, "test", &mut violations);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(violations[0].message.contains("when NNZ is nonzero"));
    }

    #[test]
    fn sparse_tensor_requires_positive_rank_and_dimensions() {
        let mut sparse = SparseTensorProto {
            values: Some(TensorProto {
                dims: vec![0],
                data_type: tensor_proto::DataType::Float as i32,
                name: "empty_sparse".into(),
                ..Default::default()
            }),
            indices: None,
            dims: Vec::new(),
        };
        let mut violations = Vec::new();
        check_sparse_tensor(&sparse, true, "test", &mut violations);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(
            violations[0]
                .message
                .contains("positive rank and dimensions")
        );

        sparse.dims = vec![2, 0];
        violations.clear();
        check_sparse_tensor(&sparse, true, "test", &mut violations);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(
            violations[0]
                .message
                .contains("positive rank and dimensions")
        );
    }

    #[test]
    fn declared_graph_io_passes() {
        let rule = InputOutputDeclaredRule;
        let mut graph = Graph::new();
        let input = graph.create_named_value("X", DataType::Float32, static_shape([1]));
        let output = graph.create_named_value("Y", DataType::Float32, static_shape([1]));
        graph.add_input(input);
        graph.add_output(output);

        assert!(
            rule.check(&Model::new(graph), &ValidationContext::default())
                .is_empty()
        );
    }

    #[test]
    fn unnamed_graph_io_is_flagged() {
        let rule = InputOutputDeclaredRule;
        let mut graph = Graph::new();
        let input = graph.create_value(DataType::Float32, static_shape([1]));
        graph.add_input(input);

        let violations = rule.check(&Model::new(graph), &ValidationContext::default());
        assert_error(
            rule.id(),
            &violations,
            ViolationLocation::Value {
                value_name: format!("<value#{}>", input.0),
            },
        );
    }

    #[test]
    fn connected_node_inputs_pass() {
        let rule = NoUnconnectedNodesRule;
        let mut graph = Graph::new();
        let input = graph.create_named_value("X", DataType::Float32, static_shape([1]));
        let output = graph.create_named_value("Y", DataType::Float32, static_shape([1]));
        let final_output = graph.create_named_value("Z", DataType::Float32, static_shape([1]));
        graph.add_input(input);
        let node_id = graph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(input)],
            vec![output],
        ));
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(output)],
            vec![final_output],
        ));
        let mut subgraph = Graph::new();
        let capture = subgraph.create_named_value("X", DataType::Float32, static_shape([1]));
        let subgraph_output =
            subgraph.create_named_value("subgraph_output", DataType::Float32, static_shape([1]));
        subgraph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(capture)],
            vec![subgraph_output],
        ));
        subgraph.add_output(subgraph_output);
        graph.subgraphs.insert((node_id, "body".into()), subgraph);
        graph.add_output(final_output);

        assert!(
            rule.check(&Model::new(graph), &ValidationContext::default())
                .is_empty()
        );
    }

    #[test]
    fn omitted_optional_node_output_passes() {
        let rule = NoUnconnectedNodesRule;
        let mut graph = Graph::new();
        let input = graph.create_named_value("X", DataType::Float32, static_shape([1]));
        let omitted = graph.create_value(DataType::Float32, Vec::new());
        graph.add_input(input);
        graph.insert_node(Node::new(
            NodeId(0),
            "OptionalOutput",
            vec![Some(input)],
            vec![omitted],
        ));

        assert!(
            rule.check(&Model::new(graph), &ValidationContext::default())
                .is_empty()
        );
    }

    #[test]
    fn captured_node_output_passes() {
        let rule = NoUnconnectedNodesRule;
        let mut graph = Graph::new();
        let input = graph.create_named_value("X", DataType::Float32, static_shape([1]));
        let captured = graph.create_named_value("captured", DataType::Float32, static_shape([1]));
        graph.add_input(input);
        let node_id = graph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(input)],
            vec![captured],
        ));

        let mut subgraph = Graph::new();
        let capture = subgraph.create_named_value("captured", DataType::Float32, static_shape([1]));
        let output = subgraph.create_named_value("Y", DataType::Float32, static_shape([1]));
        subgraph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(capture)],
            vec![output],
        ));
        subgraph.add_output(output);
        graph.subgraphs.insert((node_id, "body".into()), subgraph);

        assert!(
            rule.check(&Model::new(graph), &ValidationContext::default())
                .is_empty()
        );
    }

    #[test]
    fn dangling_node_output_is_flagged() {
        let rule = NoUnconnectedNodesRule;
        let mut graph = Graph::new();
        let input = graph.create_named_value("X", DataType::Float32, static_shape([1]));
        let dangling = graph.create_named_value("dangling", DataType::Float32, static_shape([1]));
        graph.add_input(input);
        let mut node = Node::new(NodeId(0), "Relu", vec![Some(input)], vec![dangling]);
        node.name = "relu".into();
        graph.insert_node(node);

        let violations = rule.check(&Model::new(graph), &ValidationContext::default());
        assert_error(
            rule.id(),
            &violations,
            ViolationLocation::Node {
                graph_name: String::new(),
                node_name: "relu".into(),
            },
        );
    }

    #[test]
    fn undefined_node_input_is_flagged() {
        let rule = NoUnconnectedNodesRule;
        let mut graph = Graph::new();
        let dangling = graph.create_named_value("dangling", DataType::Float32, static_shape([1]));
        let output = graph.create_named_value("Y", DataType::Float32, static_shape([1]));
        let mut node = Node::new(NodeId(0), "Relu", vec![Some(dangling)], vec![output]);
        node.name = "relu".into();
        graph.insert_node(node);
        graph.add_output(output);

        let violations = rule.check(&Model::new(graph), &ValidationContext::default());
        assert_error(
            rule.id(),
            &violations,
            ViolationLocation::Node {
                graph_name: String::new(),
                node_name: "relu".into(),
            },
        );
    }

    #[test]
    fn schema_type_constraints_accept_consistent_allowed_types() {
        let rule = TypeConstraintSatisfiedRule;
        let model = one_node_model("Add", 2, 1);

        assert!(rule.check(&model, &ValidationContext::default()).is_empty());
    }

    #[test]
    fn schema_type_constraints_skip_unknown_placeholder_types() {
        let rule = TypeConstraintSatisfiedRule;
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let input = graph.create_named_value("X", DataType::Int64, static_shape([1]));
        let output = graph.create_named_value("Y", DataType::Float32, Vec::new());
        graph.mark_value_type_unknown(output);
        graph.mark_value_shape_unknown(output);
        graph.add_input(input);
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(input)],
            vec![output],
        ));
        graph.add_output(output);

        assert!(
            rule.check(&Model::new(graph), &ValidationContext::default())
                .is_empty()
        );
    }

    #[test]
    fn schema_type_constraint_violation_is_flagged() {
        let rule = TypeConstraintSatisfiedRule;
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let input = graph.create_named_value("X", DataType::Int64, static_shape([1]));
        let output = graph.create_named_value("Y", DataType::Int64, static_shape([1]));
        graph.add_input(input);
        let mut node = Node::new(NodeId(0), "Relu", vec![Some(input)], vec![output]);
        node.name = "relu".into();
        graph.insert_node(node);
        graph.add_output(output);

        let violations = rule.check(&Model::new(graph), &ValidationContext::default());
        assert!(
            violations.iter().all(|violation| {
                violation.rule_id == rule.id()
                    && violation.severity == Severity::Error
                    && violation.location
                        == ViolationLocation::Node {
                            graph_name: String::new(),
                            node_name: "relu".into(),
                        }
            }),
            "{violations:?}"
        );
        assert_eq!(violations.len(), 2);
    }

    #[test]
    fn concrete_schema_input_type_is_enforced() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 24);
        let data = graph.create_named_value("data", DataType::Float32, static_shape([2]));
        let shape = graph.create_named_value("shape", DataType::Float32, static_shape([1]));
        let output = graph.create_named_value("output", DataType::Float32, static_shape([2]));
        graph.add_input(data);
        graph.add_input(shape);
        graph.insert_node(Node::new(
            NodeId(0),
            "Reshape",
            vec![Some(data), Some(shape)],
            vec![output],
        ));
        graph.add_output(output);
        let mut model = Model::new(graph);
        let violations = TypeConstraintSatisfiedRule.check(&model, &ValidationContext::default());
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(violations[0].message.contains("tensor(int64)"));

        model.graph.value_mut(shape).dtype = DataType::Int64;
        assert!(
            TypeConstraintSatisfiedRule
                .check(&model, &ValidationContext::default())
                .is_empty()
        );
    }

    #[test]
    fn matching_initializer_type_passes() {
        let rule = InitializerTypeMatchesDeclaredRule;
        let mut graph = Graph::new();
        let value = graph.create_named_value("W", DataType::Float32, static_shape([1]));
        graph.set_initializer(
            value,
            WeightRef::Inline(TensorData::from_raw(DataType::Float32, vec![1], vec![0; 4])),
        );

        assert!(
            rule.check(&Model::new(graph), &ValidationContext::default())
                .is_empty()
        );
    }

    #[test]
    fn initializer_symbolic_and_unknown_declared_shapes_pass() {
        let rule = InitializerTypeMatchesDeclaredRule;
        let mut graph = Graph::new();
        let batch = graph.create_symbol(Some("batch".into()));
        let symbolic = graph.create_named_value(
            "symbolic",
            DataType::Float32,
            vec![Dim::Symbolic(batch), Dim::Static(3)],
        );
        graph.set_initializer(
            symbolic,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![2, 3],
                vec![0; 24],
            )),
        );
        let unknown = graph.create_named_value("unknown", DataType::Float32, Vec::new());
        graph.mark_value_shape_unknown(unknown);
        graph.set_initializer(
            unknown,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![4],
                vec![0; 16],
            )),
        );

        assert!(
            rule.check(&Model::new(graph), &ValidationContext::default())
                .is_empty()
        );
    }

    #[test]
    fn mismatched_initializer_shape_is_flagged() {
        let rule = InitializerTypeMatchesDeclaredRule;
        for dims in [vec![2, 4], vec![2, 3, 1]] {
            let mut graph = Graph::new();
            let value = graph.create_named_value("W", DataType::Float32, static_shape([2, 3]));
            graph.set_initializer(
                value,
                WeightRef::Inline(TensorData::from_raw(
                    DataType::Float32,
                    dims.clone(),
                    Vec::new(),
                )),
            );

            let violations = rule.check(&Model::new(graph), &ValidationContext::default());
            assert_error(
                rule.id(),
                &violations,
                ViolationLocation::Value {
                    value_name: "W".into(),
                },
            );
        }
    }

    #[test]
    fn mismatched_initializer_type_is_flagged() {
        let rule = InitializerTypeMatchesDeclaredRule;
        let mut graph = Graph::new();
        let value = graph.create_named_value("W", DataType::Float32, static_shape([1]));
        graph.set_initializer(
            value,
            WeightRef::Inline(TensorData::from_raw(DataType::Int64, vec![1], vec![0; 8])),
        );

        let violations = rule.check(&Model::new(graph), &ValidationContext::default());
        assert_error(
            rule.id(),
            &violations,
            ViolationLocation::Value {
                value_name: "W".into(),
            },
        );
    }

    #[test]
    fn supported_ir_versions_pass_and_future_version_fails() {
        let rule = IrVersionSupportedRule;
        for ir_version in [1, 10, 13] {
            let mut model = Model::new(Graph::new());
            model.metadata.ir_version = ir_version;
            assert!(
                rule.check(&model, &ValidationContext::default()).is_empty(),
                "ir_version {ir_version}"
            );
        }
        let mut model = Model::new(Graph::new());
        model.metadata.ir_version = 14;
        let violations = rule.check(&model, &ValidationContext::default());
        assert_eq!(violations.len(), 1);
        assert!(violations[0].message.contains("newer than"));
    }

    #[test]
    fn absent_ir_version_is_flagged() {
        let rule = IrVersionSupportedRule;
        let mut model = Model::new(Graph::new());
        model.metadata.ir_version = 0;

        let violations = rule.check(&model, &ValidationContext::default());
        assert_error(rule.id(), &violations, ViolationLocation::Model);
    }

    #[test]
    fn ir_feature_rule_checks_opset_initializer_and_dtype_gates() {
        let rule = IrVersionFeatureRule;
        let model = retained_model_at(
            ModelProto {
                opset_import: vec![OperatorSetIdProto {
                    domain: String::new(),
                    version: 1,
                }],
                ..Default::default()
            },
            2,
        );
        assert!(
            rule.check(&model, &ValidationContext::default())
                .iter()
                .any(|violation| violation.message.contains("must not specify opset_import"))
        );

        let initializer = TensorProto {
            dims: vec![1],
            data_type: tensor_proto::DataType::Float as i32,
            float_data: vec![1.0],
            name: "W".into(),
            ..Default::default()
        };
        let model = retained_model_at(
            ModelProto {
                graph: Some(GraphProto {
                    name: "graph".into(),
                    initializer: vec![initializer],
                    ..Default::default()
                }),
                ..Default::default()
            },
            3,
        );
        assert!(
            rule.check(&model, &ValidationContext::default())
                .iter()
                .any(|violation| violation.message.contains("must also be a graph input"))
        );

        let model = retained_model_at(
            ModelProto {
                graph: Some(GraphProto {
                    name: "graph".into(),
                    initializer: vec![TensorProto {
                        dims: vec![1],
                        data_type: tensor_proto::DataType::Uint2 as i32,
                        raw_data: vec![0],
                        name: "packed".into(),
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                ..Default::default()
            },
            12,
        );
        assert!(
            rule.check(&model, &ValidationContext::default())
                .iter()
                .any(|violation| violation.message.contains("requires IR version 13"))
        );
    }

    #[test]
    fn ir_feature_rule_accepts_model_metadata_before_ir10() {
        let model = retained_model_at(
            ModelProto {
                metadata_props: vec![StringStringEntryProto {
                    key: "owner".into(),
                    value: "onnx".into(),
                }],
                ..Default::default()
            },
            7,
        );
        let violations = IrVersionFeatureRule.check(&model, &ValidationContext::default());
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn function_rule_accepts_valid_signature_and_body() {
        let model = retained_model(ModelProto {
            functions: vec![function(
                "Pass",
                vec![NodeProto {
                    input: vec!["X".into()],
                    output: vec!["Y".into()],
                    op_type: "Identity".into(),
                    ..Default::default()
                }],
            )],
            ..Default::default()
        });
        let violations = FunctionProtoValidityRule.check(&model, &ValidationContext::default());
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn function_rule_accepts_empty_default_domain() {
        let mut default_domain = function(
            "Pass",
            vec![NodeProto {
                input: vec!["X".into()],
                output: vec!["Y".into()],
                op_type: "Identity".into(),
                ..Default::default()
            }],
        );
        default_domain.domain.clear();
        let model = retained_model(ModelProto {
            functions: vec![default_domain],
            ..Default::default()
        });
        let violations = FunctionProtoValidityRule.check(&model, &ValidationContext::default());
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn function_rule_checks_defaults_topology_imports_and_unique_ids() {
        let mut invalid = function(
            "Broken",
            vec![NodeProto {
                input: vec!["missing".into()],
                output: vec!["Y".into()],
                op_type: "Identity".into(),
                ..Default::default()
            }],
        );
        invalid.attribute = vec!["alpha".into()];
        invalid.attribute_proto = vec![AttributeProto {
            name: "alpha".into(),
            r#type: attribute_proto::AttributeType::Float as i32,
            ..Default::default()
        }];
        invalid.opset_import.clear();
        let model = retained_model(ModelProto {
            functions: vec![invalid.clone(), invalid],
            ..Default::default()
        });
        let violations = FunctionProtoValidityRule.check(&model, &ValidationContext::default());
        for expected in [
            "is not unique",
            "both attribute and attribute_proto",
            "neither a function/graph input",
            "no opset import",
        ] {
            assert!(
                violations
                    .iter()
                    .any(|violation| violation.message.contains(expected)),
                "{expected}: {violations:?}"
            );
        }
    }

    #[test]
    fn function_rule_rejects_undeclared_attribute_refs_and_recursion() {
        let make_call = |callee: &str| NodeProto {
            input: vec!["X".into()],
            output: vec!["Y".into()],
            op_type: callee.into(),
            domain: "local.test".into(),
            attribute: vec![AttributeProto {
                name: "alpha".into(),
                ref_attr_name: "undeclared".into(),
                r#type: attribute_proto::AttributeType::Float as i32,
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut first = function("First", vec![make_call("Second")]);
        let mut second = function("Second", vec![make_call("First")]);
        for function in [&mut first, &mut second] {
            function.opset_import.push(OperatorSetIdProto {
                domain: "local.test".into(),
                version: 1,
            });
        }
        let model = retained_model(ModelProto {
            functions: vec![first, second],
            ..Default::default()
        });
        let violations = FunctionProtoValidityRule.check(&model, &ValidationContext::default());
        assert!(
            violations
                .iter()
                .any(|violation| violation.message.contains("undeclared function attribute"))
        );
        assert!(
            violations
                .iter()
                .any(|violation| violation.message.contains("are recursive"))
        );
    }

    #[test]
    fn function_rule_accepts_consistent_call_with_omitted_default_attribute() {
        let mut callee = function(
            "Affine",
            vec![NodeProto {
                input: vec!["X".into()],
                output: vec!["Y".into()],
                op_type: "Identity".into(),
                ..Default::default()
            }],
        );
        callee.attribute = vec!["alpha".into()];
        callee.attribute_proto = vec![AttributeProto {
            name: "beta".into(),
            r#type: attribute_proto::AttributeType::Float as i32,
            f: 1.0,
            ..Default::default()
        }];
        let model = local_function_call_model(
            callee,
            NodeProto {
                input: vec!["X".into()],
                output: vec!["Y".into()],
                op_type: "Affine".into(),
                domain: "local.test".into(),
                attribute: vec![AttributeProto {
                    name: "alpha".into(),
                    r#type: attribute_proto::AttributeType::Float as i32,
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        let violations = FunctionProtoValidityRule.check(&model, &ValidationContext::default());
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn function_rule_accepts_call_site_arity_and_attribute_mismatches() {
        let callee = function(
            "Affine",
            vec![NodeProto {
                input: vec!["X".into()],
                output: vec!["Y".into()],
                op_type: "Identity".into(),
                ..Default::default()
            }],
        );
        let mut caller = function(
            "Caller",
            vec![NodeProto {
                input: vec!["X".into(), "X".into()],
                output: vec!["Y".into(), "unused".into()],
                op_type: "Affine".into(),
                domain: "local.test".into(),
                attribute: vec![AttributeProto {
                    name: "gamma".into(),
                    r#type: attribute_proto::AttributeType::Float as i32,
                    ..Default::default()
                }],
                ..Default::default()
            }],
        );
        caller.opset_import.push(OperatorSetIdProto {
            domain: "local.test".into(),
            version: 1,
        });
        let mut callee = callee;
        callee.attribute = vec!["alpha".into()];
        callee.attribute_proto = vec![AttributeProto {
            name: "beta".into(),
            r#type: attribute_proto::AttributeType::Float as i32,
            f: 1.0,
            ..Default::default()
        }];
        let model = retained_model(ModelProto {
            functions: vec![callee, caller],
            ..Default::default()
        });
        let violations = FunctionProtoValidityRule.check(&model, &ValidationContext::default());
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn missing_opset_import_is_flagged() {
        let mut g = Graph::new();
        // Deliberately declare no opset import.
        let x = g.create_named_value("X", DataType::Float32, static_shape([2]));
        let y = g.create_named_value("Y", DataType::Float32, static_shape([2]));
        g.add_input(x);
        let mut node = Node::new(NodeId(0), "Relu", vec![Some(x)], vec![y]);
        node.name = "r".to_string();
        g.insert_node(node);
        g.add_output(y);

        let result = Model::new(g).validate();
        assert!(!result.is_valid());
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.rule_id == "ir.opset_import_present")
        );
    }

    #[test]
    fn present_opset_import_passes() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 21);
        let x = g.create_named_value("X", DataType::Float32, static_shape([2]));
        let y = g.create_named_value("Y", DataType::Float32, static_shape([2]));
        g.add_input(x);
        let node = Node::new(NodeId(0), "Relu", vec![Some(x)], vec![y]);
        g.insert_node(node);
        g.add_output(y);

        let result = Model::new(g).validate();
        assert!(result.is_valid(), "{:?}", result.violations);
    }

    #[test]
    fn duplicate_value_name_is_flagged() {
        let rule = DuplicateValueNameRule;
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 21);
        // Two distinct values that share the name "dup".
        g.create_named_value("dup", DataType::Float32, static_shape([1]));
        g.create_named_value("dup", DataType::Float32, static_shape([1]));
        let model = Model::new(g);
        let violations = rule.check(&model, &ValidationContext::default());
        assert_eq!(violations.len(), 1);
        assert_eq!(
            violations[0].location,
            ViolationLocation::Value {
                value_name: "dup".to_string()
            }
        );
    }

    #[test]
    fn duplicate_value_name_inside_subgraph_is_flagged() {
        let mut subgraph = Graph::new();
        subgraph.create_named_value("nested_dup", DataType::Float32, static_shape([1]));
        subgraph.create_named_value("nested_dup", DataType::Float32, static_shape([1]));

        let result = nested_model(subgraph).validate();
        assert!(result.violations.iter().any(|violation| {
            violation.rule_id == "structure.duplicate_value_name"
                && violation.location
                    == ViolationLocation::Value {
                        value_name: "nested_dup".into(),
                    }
        }));
    }

    #[test]
    fn missing_opset_import_inside_subgraph_is_flagged() {
        let mut subgraph = Graph::new();
        let x = subgraph.create_named_value("x", DataType::Float32, static_shape([1]));
        let y = subgraph.create_named_value("y", DataType::Float32, static_shape([1]));
        subgraph.add_input(x);
        let mut node = Node::new(NodeId(0), "Custom", vec![Some(x)], vec![y]);
        node.domain = "example.custom".into();
        subgraph.insert_node(node);
        subgraph.add_output(y);

        let result = nested_model(subgraph).validate();
        assert!(
            result
                .violations
                .iter()
                .any(|violation| violation.rule_id == "ir.opset_import_present")
        );
    }

    #[test]
    fn valid_nested_model_passes() {
        let mut subgraph = Graph::new();
        let x = subgraph.create_named_value("x", DataType::Float32, static_shape([1]));
        let y = subgraph.create_named_value("y", DataType::Float32, static_shape([1]));
        subgraph.add_input(x);
        subgraph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(x)], vec![y]));
        subgraph.add_output(y);

        let result = nested_model(subgraph).validate();
        assert!(result.is_valid(), "{:?}", result.violations);
    }

    #[test]
    fn if_requires_else_branch() {
        let result = if_model(Graph::new(), 1, false).validate();
        assert!(result.violations.iter().any(|violation| {
            violation.rule_id == "schema.node_conforms"
                && violation
                    .message
                    .contains("required attribute 'else_branch'")
        }));
    }

    #[test]
    fn if_variadic_outputs_require_at_least_one_value() {
        let result = if_model(Graph::new(), 0, true).validate();
        assert!(result.violations.iter().any(|violation| {
            violation.rule_id == "schema.node_conforms"
                && violation
                    .message
                    .contains("has 0 outputs but schema requires at least 1")
        }));

        let result = if_model(Graph::new(), 1, true).validate();
        assert!(result.is_valid(), "{:?}", result.violations);
    }

    #[test]
    fn schema_rule_accepts_conforming_node() {
        let result = one_node_model("Add", 2, 1).validate();
        assert!(result.is_valid(), "{:?}", result.violations);
    }

    #[test]
    fn round_four_optional_inputs_and_attributes_may_be_omitted() {
        for (op_type, inputs) in [
            ("ReduceMax", 1),
            ("ReduceMin", 1),
            ("ReduceProd", 1),
            ("ReduceL1", 1),
            ("ReduceL2", 1),
            ("ReduceLogSum", 1),
            ("ReduceLogSumExp", 1),
            ("ReduceSumSquare", 1),
            ("ArgMax", 1),
            ("ArgMin", 1),
            ("LogSoftmax", 1),
            ("RMSNormalization", 2),
        ] {
            let mut model = one_node_model(op_type, inputs, 1);
            model.graph.opset_imports.insert(String::new(), 24);
            if matches!(op_type, "ArgMax" | "ArgMin") {
                let output = model.graph.outputs[0];
                model.graph.value_mut(output).dtype = DataType::Int64;
            }
            let result = model.validate();
            assert!(
                result.is_valid(),
                "{op_type}: optional fields should be omittable: {:?}",
                result.violations
            );
        }
    }

    #[test]
    fn round_five_schemas_accept_official_minimal_forms() {
        for (op_type, inputs, outputs, int64_inputs, bool_inputs, int64_outputs, bool_outputs) in [
            ("GatherElements", 2, 1, vec![1], vec![], vec![], vec![]),
            ("GatherND", 2, 1, vec![1], vec![], vec![], vec![]),
            ("Equal", 2, 1, vec![], vec![], vec![], vec![0]),
            ("Greater", 2, 1, vec![], vec![], vec![], vec![0]),
            ("Less", 2, 1, vec![], vec![], vec![], vec![0]),
            ("And", 2, 1, vec![], vec![0, 1], vec![], vec![0]),
            ("Or", 2, 1, vec![], vec![0, 1], vec![], vec![0]),
            ("Not", 1, 1, vec![], vec![0], vec![], vec![0]),
            ("Shape", 1, 1, vec![], vec![], vec![0], vec![]),
            ("Size", 1, 1, vec![], vec![], vec![0], vec![]),
            ("NonZero", 1, 1, vec![], vec![], vec![0], vec![]),
            ("Range", 3, 1, vec![], vec![], vec![], vec![]),
            ("Split", 1, 2, vec![], vec![], vec![], vec![]),
        ] {
            let mut model = one_node_model(op_type, inputs, outputs);
            model.graph.opset_imports.insert(String::new(), 25);
            for index in int64_inputs {
                let value = model.graph.inputs[index];
                model.graph.value_mut(value).dtype = DataType::Int64;
            }

            for index in bool_inputs {
                let value = model.graph.inputs[index];
                model.graph.value_mut(value).dtype = DataType::Bool;
            }
            for index in int64_outputs {
                let value = model.graph.outputs[index];
                model.graph.value_mut(value).dtype = DataType::Int64;
            }
            for index in bool_outputs {
                let value = model.graph.outputs[index];
                model.graph.value_mut(value).dtype = DataType::Bool;
            }

            let result = model.validate();
            assert!(
                result.is_valid(),
                "{op_type}: optional fields should be omittable: {:?}",
                result.violations
            );
        }

        let mut cast = one_node_model("Cast", 1, 1);
        cast.graph.opset_imports.insert(String::new(), 25);
        let output = cast.graph.outputs[0];
        cast.graph.value_mut(output).dtype = DataType::Int64;
        let node = cast.graph.nodes.keys().next().unwrap();
        cast.graph.node_mut(node).attributes.insert(
            "to".into(),
            Attribute::Int(i64::from(DataType::Int64.to_onnx())),
        );
        assert!(cast.validate().is_valid());

        cast.graph.node_mut(node).attributes.remove("to");
        assert!(cast.validate().violations.iter().any(|violation| {
            violation.rule_id == "schema.node_conforms"
                && violation.message.contains("required attribute 'to'")
        }));
    }

    #[test]
    fn round_six_schemas_accept_official_minimal_forms() {
        for (op_type, inputs, int64_inputs) in [
            ("Tile", 2, vec![1]),
            ("Pad", 2, vec![1]),
            ("ScatterND", 3, vec![1]),
            ("ScatterElements", 3, vec![1]),
            ("ConstantOfShape", 1, vec![0]),
        ] {
            let mut model = one_node_model(op_type, inputs, 1);
            model.graph.opset_imports.insert(String::new(), 25);
            for index in int64_inputs {
                let value = model.graph.inputs[index];
                model.graph.value_mut(value).dtype = DataType::Int64;
            }
            let result = model.validate();
            assert!(
                result.is_valid(),
                "{op_type}: official minimal form should pass: {:?}",
                result.violations
            );
        }

        let mut pad_with_axes = one_node_model("Pad", 4, 1);
        pad_with_axes.graph.opset_imports.insert(String::new(), 25);
        let pads = pad_with_axes.graph.inputs[1];
        let axes = pad_with_axes.graph.inputs[3];
        pad_with_axes.graph.value_mut(pads).dtype = DataType::Int64;
        pad_with_axes.graph.value_mut(axes).dtype = DataType::Int32;
        let node = pad_with_axes.graph.nodes.keys().next().unwrap();
        pad_with_axes.graph.replace_input(node, 2, None);
        assert!(
            pad_with_axes.validate().is_valid(),
            "optional constant_value may be omitted while axes is present"
        );
    }

    #[test]
    fn round_seven_schemas_accept_official_minimal_forms() {
        for op_type in ["MaxPool", "AveragePool"] {
            let mut model = one_node_model(op_type, 1, 1);
            model.graph.opset_imports.insert(String::new(), 25);
            let node = model.graph.nodes.keys().next().unwrap();
            model
                .graph
                .node_mut(node)
                .attributes
                .insert("kernel_shape".into(), Attribute::Ints(vec![1]));
            assert!(
                model.validate().is_valid(),
                "{op_type}: only kernel_shape is required"
            );
        }

        for op_type in ["GlobalAveragePool", "GlobalMaxPool"] {
            let mut model = one_node_model(op_type, 1, 1);
            model.graph.opset_imports.insert(String::new(), 25);
            assert!(model.validate().is_valid(), "{op_type}");
        }

        let mut max_pool_indices = one_node_model("MaxPool", 1, 2);
        max_pool_indices
            .graph
            .opset_imports
            .insert(String::new(), 25);
        let node = max_pool_indices.graph.nodes.keys().next().unwrap();
        max_pool_indices
            .graph
            .node_mut(node)
            .attributes
            .insert("kernel_shape".into(), Attribute::Ints(vec![1]));
        let indices = max_pool_indices.graph.outputs[1];
        max_pool_indices.graph.value_mut(indices).dtype = DataType::Int64;
        assert!(max_pool_indices.validate().is_valid());

        let mut resize = one_node_model("Resize", 1, 1);
        resize.graph.opset_imports.insert(String::new(), 25);
        assert!(
            resize.validate().is_valid(),
            "the official checker schema does not require either optional input"
        );

        let mut quantize = one_node_model("QuantizeLinear", 2, 1);
        quantize.graph.opset_imports.insert(String::new(), 25);
        let output = quantize.graph.outputs[0];
        quantize.graph.value_mut(output).dtype = DataType::Uint8;
        assert!(
            quantize.validate().is_valid(),
            "zero_point must remain optional"
        );

        let mut dequantize = one_node_model("DequantizeLinear", 2, 1);
        dequantize.graph.opset_imports.insert(String::new(), 25);
        let input = dequantize.graph.inputs[0];
        dequantize.graph.value_mut(input).dtype = DataType::Uint8;
        assert!(
            dequantize.validate().is_valid(),
            "zero_point must remain optional"
        );

        let mut dynamic = one_node_model("DynamicQuantizeLinear", 1, 3);
        dynamic.graph.opset_imports.insert(String::new(), 25);
        for output in [dynamic.graph.outputs[0], dynamic.graph.outputs[2]] {
            dynamic.graph.value_mut(output).dtype = DataType::Uint8;
        }
        assert!(dynamic.validate().is_valid());
    }

    #[test]
    fn common_builtin_arity_boundaries_pass() {
        for (op_type, inputs, outputs) in [
            ("MatMul", 2, 1),
            ("Gemm", 2, 1),
            ("Gemm", 3, 1),
            ("Add", 2, 1),
            ("Relu", 1, 1),
            ("Conv", 2, 1),
            ("Conv", 3, 1),
            ("Mul", 2, 1),
            ("Identity", 1, 1),
        ] {
            let result = one_node_model(op_type, inputs, outputs).validate();
            assert!(
                result.is_valid(),
                "{op_type}({inputs}, {outputs}): {:?}",
                result.violations
            );
        }
    }

    #[test]
    fn schema_rule_checks_arity_required_and_typed_attributes() {
        let mut model = one_node_model("Gemm", 1, 2);
        let node_id = model.graph.nodes.iter().next().unwrap().0;
        let node = model.graph.node_mut(node_id);
        node.attributes.insert("alpha".into(), Attribute::Int(1));
        node.attributes.insert("unknown".into(), Attribute::Int(1));
        let result = model.validate();
        let messages = result
            .violations
            .iter()
            .filter(|violation| violation.rule_id == "schema.node_conforms")
            .map(|violation| violation.message.as_str())
            .collect::<Vec<_>>();
        assert!(messages.iter().any(|message| message.contains("1 inputs")));
        assert!(messages.iter().any(|message| message.contains("2 outputs")));
        assert!(
            messages
                .iter()
                .any(|message| message.contains("requires Float"))
        );
        assert!(
            messages
                .iter()
                .any(|message| message.contains("not declared"))
        );
    }

    #[test]
    fn schema_rule_checks_required_attributes_and_opset_range() {
        let yaml = r#"
domain: ""
name: NeedsAxis
since_version: 13
until_version: 20
attributes:
  - { name: axis, type: int, required: true }
inputs: [{ name: X, type_str: T }]
outputs: [{ name: Y, type_str: T }]
"#;
        let mut schemas = SchemaRegistry::new();
        schemas.load_yaml(yaml).unwrap();
        let mut model = one_node_model("NeedsAxis", 1, 1);
        model.graph.opset_imports.insert(String::new(), 13);
        let checker = super::super::OnnxChecker::with_schema_registry(schemas.clone());
        let result = checker.check(&model);
        assert!(result.violations.iter().any(|violation| {
            violation.rule_id == "schema.node_conforms"
                && violation.message.contains("required attribute 'axis'")
        }));

        model.graph.opset_imports.insert(String::new(), 21);
        let result = super::super::OnnxChecker::with_schema_registry(schemas).check(&model);
        assert!(result.violations.iter().any(|violation| {
            violation.rule_id == "schema.node_conforms"
                && violation.message.contains("no schema valid at opset 21")
        }));
    }

    #[test]
    fn schema_rule_reports_unregistered_operator() {
        let result = one_node_model("NotARealOp", 1, 1).validate();
        assert!(result.violations.iter().any(|violation| {
            violation.rule_id == "schema.node_conforms"
                && violation
                    .message
                    .contains("not present in the schema registry")
        }));
    }
}
