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
fn constant_fp8_tensor_requires_its_schema_opset() {
    use onnx_runtime_ir::DataType;
    use onnx_runtime_loader::load_model_bytes;

    for (dtype, minimum) in [
        (DataType::Float8E4M3FN, 19),
        (DataType::Float8E4M3FNUZ, 19),
        (DataType::Float8E5M2, 19),
        (DataType::Float8E5M2FNUZ, 19),
        (DataType::Float8E8M0, 24),
    ] {
        let payload = vec![0x01, 0x7f, 0xff];
        let mut constant = node("Constant", &[], &["Y"]);
        constant.name = "fp8_value".into();
        constant.attribute = vec![onnx::AttributeProto {
            name: "value".into(),
            r#type: onnx::attribute_proto::AttributeType::Tensor as i32,
            t: Some(onnx::TensorProto {
                data_type: dtype.to_onnx(),
                dims: vec![3],
                raw_data: payload.clone(),
                ..Default::default()
            }),
            ..Default::default()
        }];
        let mut model = model(onnx::GraphProto {
            output: vec![value_info("Y")],
            node: vec![constant],
            ..Default::default()
        });
        model.ir_version = 12;
        model.opset_import[0].version = minimum - 1;
        let err = load_model_bytes(&model.encode_to_vec()).unwrap_err();
        assert!(
            matches!(&err, LoaderError::ConstantValueOpset {
                node, dtype: rejected, opset, minimum: required
            } if node.contains("fp8_value")
                && *rejected == dtype
                && *opset == (minimum - 1) as u64
                && *required == minimum as u64),
            "{dtype:?}: {err}"
        );

        model.opset_import[0].version = minimum;
        let graph = load_model_bytes(&model.encode_to_vec()).unwrap();
        let (_, node) = graph.nodes.iter().next().unwrap();
        let Some(onnx_runtime_ir::Attribute::Tensor(value)) = node.attr("value") else {
            panic!("loader must preserve the Constant node's tensor-valued attribute");
        };
        assert_eq!(value.dtype, dtype);
        assert_eq!(value.data, payload);
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
fn invalid_ir_version_is_rejected_and_new_ir_versions_load_without_a_ceiling() {
    // Lower sanity bound: absent/zero (or negative) ir_version is rejected.
    for bad in [0, -1] {
        let mut invalid = model(onnx::GraphProto::default());
        invalid.ir_version = bad;
        assert!(matches!(
            onnx_runtime_loader::load_model_bytes(&invalid.encode_to_vec()),
            Err(LoaderError::InvalidIrVersion { .. })
        ));
    }

    // No upper bound: a far-future IR version loads (new IR versions are
    // backward-compatible; we never gate on the version number).
    let mut future = model(onnx::GraphProto::default());
    future.ir_version = 999;
    onnx_runtime_loader::load_model_bytes(&future.encode_to_vec())
        .expect("future IR versions are not rejected by a ceiling");

    onnx_runtime_loader::load_model_bytes(&model(onnx::GraphProto::default()).encode_to_vec())
        .expect("baseline IR version");
}

#[test]
fn ir13_with_default_opset_loads() {
    let mut legal = model(onnx::GraphProto::default());
    legal.ir_version = 13;
    onnx_runtime_loader::load_model_bytes(&legal.encode_to_vec())
        .expect("IR 13 model with default opset");
}

#[test]
fn ir3_requires_a_nonempty_opset_import_but_not_a_default_domain_import() {
    let mut invalid = model(onnx::GraphProto::default());
    invalid.opset_import.clear();
    assert!(matches!(
        onnx_runtime_loader::load_model_bytes(&invalid.encode_to_vec()),
        Err(LoaderError::MissingModelOpsetImport { ir_version: 8 })
    ));

    invalid.opset_import.push(onnx::OperatorSetIdProto {
        domain: "com.example".to_string(),
        version: 1,
    });
    onnx_runtime_loader::load_model_bytes(&invalid.encode_to_vec())
        .expect("custom-domain-only opset import is legal");
}

#[test]
fn repeated_empty_optional_outputs_are_allowed() {
    let legal = onnx::GraphProto {
        input: vec![value_info("X")],
        output: vec![value_info("Y")],
        node: vec![
            node("Identity", &["X"], &["mid", ""]),
            node("Identity", &["mid"], &["Y", ""]),
        ],
        ..Default::default()
    };
    onnx_runtime_loader::load_model_bytes(&model(legal).encode_to_vec())
        .expect("empty optional outputs are not SSA values");
}

#[test]
fn graph_input_that_is_an_initializer_loads_for_pre_ir4_model() {
    let mut legal = model(onnx::GraphProto {
        input: vec![value_info("W")],
        initializer: vec![f32_initializer("W")],
        output: vec![value_info("W")],
        ..Default::default()
    });
    legal.ir_version = 3;
    onnx_runtime_loader::load_model_bytes(&legal.encode_to_vec())
        .expect("graph input and initializer are both legal before IR 4");
}

#[test]
fn identical_value_names_in_separate_subgraph_scopes_load() {
    let branch = || onnx::GraphProto {
        input: vec![value_info("shared")],
        output: vec![value_info("shared")],
        ..Default::default()
    };
    let legal = model(onnx::GraphProto {
        node: vec![node_with_attrs(
            "If",
            vec![
                graph_attr("then_branch", branch()),
                graph_attr("else_branch", branch()),
            ],
        )],
        ..Default::default()
    });
    onnx_runtime_loader::load_model_bytes(&legal.encode_to_vec())
        .expect("subgraph scopes may shadow one another");
}

#[test]
fn ref_attribute_in_nested_function_subgraph_loads() {
    let ref_attr = onnx::AttributeProto {
        name: "axis".to_string(),
        ref_attr_name: "axis_parameter".to_string(),
        r#type: onnx::attribute_proto::AttributeType::Int as i32,
        ..Default::default()
    };
    let mut legal = model(onnx::GraphProto::default());
    legal.functions.push(onnx::FunctionProto {
        name: "NestedReference".to_string(),
        attribute: vec!["axis_parameter".to_string()],
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
    onnx_runtime_loader::load_model_bytes(&legal.encode_to_vec())
        .expect("function subgraphs may reference function attributes");
}

#[test]
fn initializer_that_is_a_graph_output_loads() {
    let legal = onnx::GraphProto {
        initializer: vec![f32_initializer("W")],
        output: vec![value_info("W")],
        ..Default::default()
    };
    onnx_runtime_loader::load_model_bytes(&model(legal).encode_to_vec())
        .expect("initializer graph output is a valid passthrough");
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

/// `build_graph` is an internal, pre-weight-attachment step. The retained
/// public loading API must still reject an output that has no source.
#[test]
fn public_load_model_bytes_rejects_producerless_graph_output() {
    let malformed = model(onnx::GraphProto {
        output: vec![value_info("missing")],
        ..Default::default()
    });

    assert!(matches!(
        onnx_runtime_loader::load_model_bytes(&malformed.encode_to_vec()),
        Err(LoaderError::GraphOutputMissingProducer { tensor }) if tensor == "missing"
    ));
}

#[test]
fn public_load_model_bytes_rejects_dependency_cycle() {
    let cyclic = model(onnx::GraphProto {
        output: vec![value_info("a")],
        node: vec![
            node("Identity", &["b"], &["a"]),
            node("Identity", &["a"], &["b"]),
        ],
        ..Default::default()
    });

    assert!(matches!(
        onnx_runtime_loader::load_model_bytes(&cyclic.encode_to_vec()),
        Err(LoaderError::GraphBuild(error)) if error.contains("CycleDetected")
    ));
}
