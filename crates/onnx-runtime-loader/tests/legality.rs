//! Focused protobuf-level model legality tests.

use prost::Message;

use onnx_runtime_loader::{LoaderError, proto::onnx};

fn value_info(name: &str) -> onnx::ValueInfoProto {
    onnx::ValueInfoProto {
        name: name.to_string(),
        ..Default::default()
    }
}

fn node(op_type: &str, inputs: &[&str], outputs: &[&str]) -> onnx::NodeProto {
    onnx::NodeProto {
        op_type: op_type.to_string(),
        input: inputs.iter().map(|name| (*name).to_string()).collect(),
        output: outputs.iter().map(|name| (*name).to_string()).collect(),
        ..Default::default()
    }
}

fn graph_attr(name: &str, graph: onnx::GraphProto) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Graph as i32,
        g: Some(graph),
        ..Default::default()
    }
}

fn node_with_attrs(op_type: &str, attributes: Vec<onnx::AttributeProto>) -> onnx::NodeProto {
    onnx::NodeProto {
        op_type: op_type.to_string(),
        attribute: attributes,
        ..Default::default()
    }
}

fn model(graph: onnx::GraphProto) -> onnx::ModelProto {
    onnx::ModelProto {
        ir_version: 8,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        graph: Some(graph),
        ..Default::default()
    }
}

fn f32_initializer(name: &str) -> onnx::TensorProto {
    onnx::TensorProto {
        name: name.to_string(),
        data_type: 1,
        dims: vec![1],
        raw_data: vec![0; 4],
        ..Default::default()
    }
}

#[test]
fn duplicate_value_producer_is_rejected_and_unique_values_load() {
    let duplicate = onnx::GraphProto {
        input: vec![value_info("X")],
        output: vec![value_info("Y")],
        node: vec![
            node("Identity", &["X"], &["X"]),
            node("Identity", &["X"], &["Y"]),
        ],
        ..Default::default()
    };
    assert!(matches!(
        onnx_runtime_loader::load_model_bytes(&model(duplicate).encode_to_vec()),
        Err(LoaderError::DuplicateValueProducer { tensor, .. }) if tensor == "X"
    ));

    let legal = onnx::GraphProto {
        input: vec![value_info("X")],
        output: vec![value_info("Y")],
        node: vec![
            node("Identity", &["X"], &["mid"]),
            node("Identity", &["mid"], &["Y"]),
        ],
        ..Default::default()
    };
    onnx_runtime_loader::load_model_bytes(&model(legal).encode_to_vec()).expect("unique values");

    let nested_duplicate = model(onnx::GraphProto {
        node: vec![node_with_attrs(
            "If",
            vec![graph_attr(
                "then_branch",
                onnx::GraphProto {
                    input: vec![value_info("captured")],
                    node: vec![node("Identity", &["captured"], &["captured"])],
                    ..Default::default()
                },
            )],
        )],
        ..Default::default()
    });
    assert!(matches!(
        onnx_runtime_loader::validate_model_proto(&nested_duplicate),
        Err(LoaderError::DuplicateValueProducer { tensor, .. }) if tensor == "captured"
    ));
}

#[test]
fn ref_attribute_outside_function_is_rejected_but_function_references_are_allowed() {
    let ref_attr = onnx::AttributeProto {
        name: "axis".to_string(),
        ref_attr_name: "axis_parameter".to_string(),
        r#type: onnx::attribute_proto::AttributeType::Int as i32,
        ..Default::default()
    };
    let invalid = model(onnx::GraphProto {
        node: vec![node_with_attrs("Identity", vec![ref_attr.clone()])],
        ..Default::default()
    });
    assert!(matches!(
        onnx_runtime_loader::load_model_bytes(&invalid.encode_to_vec()),
        Err(LoaderError::RefAttributeOutsideFunction { ref_attr_name, .. })
            if ref_attr_name == "axis_parameter"
    ));

    let mut legal = model(onnx::GraphProto::default());
    legal.functions.push(onnx::FunctionProto {
        name: "AllowedReference".to_string(),
        node: vec![node_with_attrs("Identity", vec![ref_attr.clone()])],
        ..Default::default()
    });
    onnx_runtime_loader::validate_model_proto(&legal).expect("function reference is legal");

    let nested_ref = model(onnx::GraphProto {
        node: vec![node_with_attrs(
            "If",
            vec![graph_attr(
                "then_branch",
                onnx::GraphProto {
                    node: vec![node_with_attrs("Identity", vec![ref_attr])],
                    ..Default::default()
                },
            )],
        )],
        ..Default::default()
    });
    assert!(matches!(
        onnx_runtime_loader::validate_model_proto(&nested_ref),
        Err(LoaderError::RefAttributeOutsideFunction { .. })
    ));
}

#[test]
fn invalid_and_future_ir_versions_are_rejected_and_supported_version_loads() {
    for (ir_version, expected_invalid) in [
        (0, true),
        (onnx_runtime_loader::MAX_SUPPORTED_IR_VERSION + 1, false),
    ] {
        let mut invalid = model(onnx::GraphProto::default());
        invalid.ir_version = ir_version;
        let result = onnx_runtime_loader::load_model_bytes(&invalid.encode_to_vec());
        if expected_invalid {
            assert!(matches!(result, Err(LoaderError::InvalidIrVersion { .. })));
        } else {
            assert!(matches!(
                result,
                Err(LoaderError::UnsupportedIrVersion { .. })
            ));
        }
    }
    onnx_runtime_loader::load_model_bytes(&model(onnx::GraphProto::default()).encode_to_vec())
        .expect("supported IR version");
}

#[test]
fn missing_default_opset_for_ir3_is_rejected_and_default_domain_alias_is_allowed() {
    let mut invalid = model(onnx::GraphProto::default());
    invalid.opset_import = vec![onnx::OperatorSetIdProto {
        domain: "com.example".to_string(),
        version: 1,
    }];
    assert!(matches!(
        onnx_runtime_loader::load_model_bytes(&invalid.encode_to_vec()),
        Err(LoaderError::MissingDefaultOpsetImport { ir_version: 8 })
    ));

    invalid.opset_import.push(onnx::OperatorSetIdProto {
        domain: "ai.onnx".to_string(),
        version: 17,
    });
    onnx_runtime_loader::load_model_bytes(&invalid.encode_to_vec())
        .expect("ai.onnx alias is a default import");
}

#[test]
fn subgraph_input_cannot_shadow_outer_initializer() {
    let invalid = model(onnx::GraphProto {
        initializer: vec![f32_initializer("W")],
        node: vec![node_with_attrs(
            "If",
            vec![graph_attr(
                "then_branch",
                onnx::GraphProto {
                    input: vec![value_info("W")],
                    output: vec![value_info("W")],
                    ..Default::default()
                },
            )],
        )],
        ..Default::default()
    });
    assert!(matches!(
        onnx_runtime_loader::load_model_bytes(&invalid.encode_to_vec()),
        Err(LoaderError::SubgraphInputShadowsInitializer { tensor }) if tensor == "W"
    ));

    let legal = model(onnx::GraphProto {
        initializer: vec![f32_initializer("W")],
        node: vec![node_with_attrs(
            "If",
            vec![graph_attr(
                "then_branch",
                onnx::GraphProto {
                    input: vec![value_info("inner")],
                    output: vec![value_info("inner")],
                    ..Default::default()
                },
            )],
        )],
        ..Default::default()
    });
    onnx_runtime_loader::validate_model_proto(&legal).expect("distinct subgraph input");
}

#[test]
fn graph_output_must_be_locally_sourced_and_input_passthrough_is_allowed() {
    let invalid = model(onnx::GraphProto {
        output: vec![value_info("missing")],
        ..Default::default()
    });
    assert!(matches!(
        onnx_runtime_loader::load_model_bytes(&invalid.encode_to_vec()),
        Err(LoaderError::GraphOutputMissingProducer { tensor }) if tensor == "missing"
    ));

    let legal = onnx::GraphProto {
        input: vec![value_info("X")],
        output: vec![value_info("X")],
        ..Default::default()
    };
    onnx_runtime_loader::load_model_bytes(&model(legal).encode_to_vec())
        .expect("input passthrough");

    let nested_missing = model(onnx::GraphProto {
        node: vec![node_with_attrs(
            "If",
            vec![graph_attr(
                "then_branch",
                onnx::GraphProto {
                    output: vec![value_info("missing_nested")],
                    ..Default::default()
                },
            )],
        )],
        ..Default::default()
    });
    assert!(matches!(
        onnx_runtime_loader::validate_model_proto(&nested_missing),
        Err(LoaderError::GraphOutputMissingProducer { tensor }) if tensor == "missing_nested"
    ));
}
