use prost::Message;

use onnx_runtime_ir::ModelFunctionKey;
use onnx_runtime_loader::proto::onnx::{
    AttributeProto, FunctionProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
    ValueInfoProto, attribute_proto::AttributeType,
};
use onnx_runtime_loader::{Model, encode_model_proto, load_model_ir_bytes};

fn value(name: &str) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        ..Default::default()
    }
}

fn node(op_type: &str, domain: &str, overload: &str, input: &str, output: &str) -> NodeProto {
    NodeProto {
        op_type: op_type.to_string(),
        domain: domain.to_string(),
        overload: overload.to_string(),
        input: vec![input.to_string()],
        output: vec![output.to_string()],
        ..Default::default()
    }
}

fn model(functions: Vec<FunctionProto>, call_overload: &str) -> ModelProto {
    ModelProto {
        ir_version: 11,
        opset_import: vec![
            OperatorSetIdProto {
                domain: String::new(),
                version: 24,
            },
            OperatorSetIdProto {
                domain: "local".to_string(),
                version: 1,
            },
        ],
        graph: Some(GraphProto {
            input: vec![value("x")],
            output: vec![value("y")],
            node: vec![node("F", "local", call_overload, "x", "y")],
            ..Default::default()
        }),
        functions,
        ..Default::default()
    }
}

fn identity_function(overload: &str) -> FunctionProto {
    FunctionProto {
        name: "F".to_string(),
        domain: "local".to_string(),
        overload: overload.to_string(),
        input: vec!["X".to_string()],
        output: vec!["Y".to_string()],
        node: vec![node("Identity", "", "", "X", "Y")],
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 24,
        }],
        ..Default::default()
    }
}

#[test]
fn overload_identity_and_calls_round_trip() {
    let mut fast = identity_function("fast");
    fast.opset_import[0].version = 17;
    let source = model(vec![fast, identity_function("precise")], "precise");
    let graph = load_model_ir_bytes(&source.encode_to_vec()).unwrap();

    assert!(
        graph
            .model_functions
            .contains_key(&ModelFunctionKey::new("local", "F", "fast"))
    );
    assert!(
        graph
            .model_functions
            .contains_key(&ModelFunctionKey::new("local", "F", "precise"))
    );
    assert_eq!(graph.nodes.values().next().unwrap().overload, "precise");
    assert_eq!(
        graph.model_functions[&ModelFunctionKey::new("local", "F", "fast")]
            .body
            .opset_imports[""],
        17
    );

    let encoded = encode_model_proto(&Model::new(&graph)).unwrap();
    assert_eq!(encoded.graph.unwrap().node[0].overload, "precise");
    let mut overloads: Vec<_> = encoded
        .functions
        .iter()
        .map(|function| function.overload.as_str())
        .collect();
    overloads.sort_unstable();
    assert_eq!(overloads, ["fast", "precise"]);
    assert_eq!(
        encoded
            .functions
            .iter()
            .find(|f| f.overload == "fast")
            .unwrap()
            .opset_import[0]
            .version,
        17
    );
}

#[test]
fn malformed_function_and_model_imports_are_not_silently_discarded() {
    for in_function in [false, true] {
        for duplicate in [false, true] {
            let mut source = model(vec![identity_function("")], "");
            let imports = if in_function {
                &mut source.functions[0].opset_import
            } else {
                &mut source.opset_import
            };
            if duplicate {
                imports.push(OperatorSetIdProto {
                    domain: "ai.onnx".into(),
                    version: 17,
                });
            } else {
                imports[0].version = 0;
            }
            let error = load_model_ir_bytes(&source.encode_to_vec())
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(if duplicate {
                    "duplicate opset"
                } else {
                    "positive version"
                }),
                "{error}"
            );
        }
    }
}

#[test]
fn formal_defaults_and_nested_refs_round_trip() {
    let nested = GraphProto {
        input: vec![value("X")],
        output: vec![value("Z")],
        node: vec![NodeProto {
            op_type: "LeakyRelu".to_string(),
            input: vec!["X".to_string()],
            output: vec!["Z".to_string()],
            attribute: vec![AttributeProto {
                name: "alpha".to_string(),
                ref_attr_name: "slope".to_string(),
                r#type: AttributeType::Float as i32,
                ..Default::default()
            }],
            ..Default::default()
        }],
        ..Default::default()
    };
    let function = FunctionProto {
        name: "F".to_string(),
        domain: "local".to_string(),
        input: vec!["X".to_string()],
        output: vec!["Y".to_string()],
        attribute: vec!["slope".to_string()],
        attribute_proto: vec![AttributeProto {
            name: "axis".to_string(),
            i: -1,
            r#type: AttributeType::Int as i32,
            ..Default::default()
        }],
        node: vec![NodeProto {
            op_type: "If".to_string(),
            input: vec!["X".to_string()],
            output: vec!["Y".to_string()],
            attribute: vec![AttributeProto {
                name: "then_branch".to_string(),
                g: Some(nested),
                r#type: AttributeType::Graph as i32,
                ..Default::default()
            }],
            ..Default::default()
        }],
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 24,
        }],
        ..Default::default()
    };
    let graph = load_model_ir_bytes(&model(vec![function], "").encode_to_vec()).unwrap();
    let encoded = encode_model_proto(&Model::new(&graph)).unwrap();
    let function = &encoded.functions[0];

    assert_eq!(function.attribute, ["slope"]);
    assert_eq!(function.attribute_proto[0].name, "axis");
    let nested_ref = &function.node[0].attribute[0].g.as_ref().unwrap().node[0].attribute[0];
    assert_eq!(nested_ref.ref_attr_name, "slope");
    assert_eq!(nested_ref.r#type, AttributeType::Float as i32);
}

#[test]
fn duplicate_function_identity_is_rejected() {
    let bytes = model(
        vec![identity_function("same"), identity_function("same")],
        "same",
    )
    .encode_to_vec();
    let error = load_model_ir_bytes(&bytes).unwrap_err().to_string();
    assert!(error.contains("declared more than once"), "{error}");
    assert!(error.contains("same"), "{error}");
}

#[test]
fn unresolved_nested_formal_ref_is_rejected() {
    let mut function = identity_function("");
    function.node[0].attribute.push(AttributeProto {
        name: "alpha".to_string(),
        ref_attr_name: "missing".to_string(),
        r#type: AttributeType::Float as i32,
        ..Default::default()
    });
    let error = load_model_ir_bytes(&model(vec![function], "").encode_to_vec())
        .unwrap_err()
        .to_string();
    assert!(error.contains("undeclared formal attribute"), "{error}");
    assert!(error.contains("missing"), "{error}");
}

#[test]
fn ir_function_binding_uses_declared_default() {
    let mut function = identity_function("");
    function.attribute_proto = vec![AttributeProto {
        name: "slope".to_string(),
        f: 0.125,
        r#type: AttributeType::Float as i32,
        ..Default::default()
    }];
    function.node[0].attribute = vec![AttributeProto {
        name: "alpha".to_string(),
        ref_attr_name: "slope".to_string(),
        r#type: AttributeType::Float as i32,
        ..Default::default()
    }];
    let graph = load_model_ir_bytes(&model(vec![function], "").encode_to_vec()).unwrap();
    let call = graph.nodes.values().next().unwrap().clone();
    let function = graph
        .model_functions
        .get(&ModelFunctionKey::new("local", "F", ""))
        .unwrap();
    let mut body = function.body.clone();

    onnx_runtime_loader::function_inline::bind_ir_function_attributes(&call, function, &mut body)
        .unwrap();

    let body_node = body.nodes.values().next().unwrap();
    assert_eq!(
        body_node
            .attr("alpha")
            .and_then(onnx_runtime_ir::Attribute::as_float),
        Some(0.125)
    );
}

#[test]
fn ir_function_binding_rejects_missing_required_value() {
    let mut function = identity_function("");
    function.attribute = vec!["slope".to_string()];
    function.node[0].attribute = vec![AttributeProto {
        name: "alpha".to_string(),
        ref_attr_name: "slope".to_string(),
        r#type: AttributeType::Float as i32,
        ..Default::default()
    }];
    let graph = load_model_ir_bytes(&model(vec![function], "").encode_to_vec()).unwrap();
    let call = graph.nodes.values().next().unwrap().clone();
    let function = graph
        .model_functions
        .get(&ModelFunctionKey::new("local", "F", ""))
        .unwrap();
    let mut body = function.body.clone();

    let error = onnx_runtime_loader::function_inline::bind_ir_function_attributes(
        &call, function, &mut body,
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("missing required attribute 'slope'"),
        "{error}"
    );
}

#[test]
fn function_attribute_binding_rejects_call_site_and_default_kind_mismatches() {
    for supplied in [false, true] {
        let mut function = identity_function("");
        function.node[0].attribute = vec![AttributeProto {
            name: "alpha".into(),
            ref_attr_name: "slope".into(),
            r#type: AttributeType::Float as i32,
            ..Default::default()
        }];
        let wrong = AttributeProto {
            name: "slope".into(),
            i: 7,
            r#type: AttributeType::Int as i32,
            ..Default::default()
        };
        if supplied {
            function.attribute = vec!["slope".into()];
        } else {
            function.attribute_proto = vec![wrong.clone()];
        }
        let mut source = model(vec![function], "");
        if supplied {
            source.graph.as_mut().unwrap().node[0].attribute.push(wrong);
        }
        let proto_error = onnx_runtime_loader::function_inline::inline_functions(&source)
            .unwrap_err()
            .to_string();
        let graph = load_model_ir_bytes(&source.encode_to_vec()).unwrap();
        let call = graph.nodes.values().next().unwrap();
        let function = &graph.model_functions[&ModelFunctionKey::new("local", "F", "")];
        let mut body = function.body.clone();
        let ir_error = onnx_runtime_loader::function_inline::bind_ir_function_attributes(
            call, function, &mut body,
        )
        .unwrap_err()
        .to_string();
        for message in [proto_error, ir_error] {
            for context in ["local::F", "slope", "Float", "Int"] {
                assert!(message.contains(context), "missing {context:?}: {message}");
            }
        }
    }
}
