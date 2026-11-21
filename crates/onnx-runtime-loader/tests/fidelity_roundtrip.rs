//! Source-preserving IR must retain distinctions that execution lowering may
//! intentionally reject or refine.

use onnx_runtime_ir::pattern::{MatchOptions, MatchSession, Pattern, PatternLibrary};
use onnx_runtime_ir::{DataType, onnx};
use onnx_runtime_loader::{Model, encode_model_proto, load_model_ir_bytes};
use prost::Message;

fn tensor(shape: Option<onnx::TensorShapeProto>) -> onnx::TypeProto {
    onnx::TypeProto {
        value: Some(onnx::type_proto::Value::TensorType(
            onnx::type_proto::Tensor {
                elem_type: DataType::Float32.to_onnx(),
                shape,
            },
        )),
        ..Default::default()
    }
}

fn info(name: &str, ty: Option<onnx::TypeProto>) -> onnx::ValueInfoProto {
    onnx::ValueInfoProto {
        name: name.into(),
        r#type: ty,
        ..Default::default()
    }
}

fn model(graph: onnx::GraphProto) -> onnx::ModelProto {
    onnx::ModelProto {
        ir_version: 11,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: 24,
        }],
        graph: Some(graph),
        ..Default::default()
    }
}

fn identity(ty: Option<onnx::TypeProto>) -> onnx::ModelProto {
    model(onnx::GraphProto {
        input: vec![info("x", ty.clone())],
        output: vec![info("y", ty)],
        node: vec![onnx::NodeProto {
            op_type: "Identity".into(),
            input: vec!["x".into()],
            output: vec!["y".into()],
            ..Default::default()
        }],
        ..Default::default()
    })
}

#[test]
fn recursive_declared_types_unknown_rank_and_dimensions_roundtrip() {
    use onnx::tensor_shape_proto::{Dimension, dimension::Value};
    use onnx::type_proto;
    let dimensional = tensor(Some(onnx::TensorShapeProto {
        dim: vec![
            Dimension {
                value: Some(Value::DimValue(0)),
                denotation: "DATA_BATCH".into(),
            },
            Dimension {
                value: Some(Value::DimParam("N".into())),
                ..Default::default()
            },
            Dimension::default(),
            Dimension {
                value: Some(Value::DimParam(String::new())),
                ..Default::default()
            },
        ],
    }));
    let mut annotated = dimensional.clone();
    annotated.denotation = "IMAGE".into();
    let recursive = onnx::TypeProto {
        value: Some(type_proto::Value::OptionalType(Box::new(
            type_proto::Optional {
                elem_type: Some(Box::new(onnx::TypeProto {
                    value: Some(type_proto::Value::SequenceType(Box::new(
                        type_proto::Sequence {
                            elem_type: Some(Box::new(onnx::TypeProto {
                                value: Some(type_proto::Value::MapType(Box::new(
                                    type_proto::Map {
                                        key_type: DataType::String.to_onnx(),
                                        value_type: Some(Box::new(annotated)),
                                    },
                                ))),
                                ..Default::default()
                            })),
                        },
                    ))),
                    ..Default::default()
                })),
            },
        ))),
        ..Default::default()
    };
    for ty in [
        None,
        Some(onnx::TypeProto::default()),
        Some(tensor(None)),
        Some(tensor(Some(onnx::TensorShapeProto::default()))),
        Some(dimensional),
        Some(recursive),
        Some(onnx::TypeProto {
            value: Some(type_proto::Value::SparseTensorType(
                type_proto::SparseTensor {
                    elem_type: DataType::Int64.to_onnx(),
                    shape: None,
                },
            )),
            ..Default::default()
        }),
        Some(onnx::TypeProto {
            value: Some(type_proto::Value::OpaqueType(type_proto::Opaque {
                domain: "vendor".into(),
                name: "resource".into(),
            })),
            ..Default::default()
        }),
    ] {
        let source = identity(ty.clone());
        let graph = load_model_ir_bytes(&source.encode_to_vec()).unwrap();
        assert_eq!(graph.value(graph.inputs[0]).declared_type, ty);
        let expected_rank = match ty.as_ref().and_then(|tp| tp.value.as_ref()) {
            Some(type_proto::Value::TensorType(tensor)) => {
                tensor.shape.as_ref().map(|shape| shape.dim.len())
            }
            _ => None,
        };
        assert_eq!(graph.value(graph.inputs[0]).rank(), expected_rank);
        assert_eq!(graph.clone().value(graph.inputs[0]).rank(), expected_rank);
        let encoded = encode_model_proto(&Model::new(&graph))
            .unwrap()
            .graph
            .unwrap();
        assert_eq!(encoded.input[0].r#type, ty);
        assert_eq!(encoded.output[0].r#type, ty);
        let root = graph.nodes.keys().next().unwrap();
        let session = MatchSession::new(&graph).unwrap();
        assert!(
            session
                .match_at(
                    &Pattern::parse_ort("(x)=>return Identity(x)").unwrap(),
                    root,
                    MatchOptions::default()
                )
                .unwrap()
                .is_some()
        );
        let is_tensor = matches!(
            ty.as_ref().and_then(|tp| tp.value.as_ref()),
            Some(type_proto::Value::TensorType(_))
        );
        assert_eq!(
            session
                .match_at(
                    &Pattern::parse_ort("(x:tensor[...])=>return Identity(x)").unwrap(),
                    root,
                    MatchOptions::default()
                )
                .unwrap()
                .is_some(),
            is_tensor
        );
        if !is_tensor {
            assert_eq!(graph.value(graph.inputs[0]).dtype, DataType::Undefined);
            assert!(!graph.value_type_is_known(graph.inputs[0]));
        }
    }
}

#[test]
fn malformed_recursive_types_do_not_turn_into_float_scalars() {
    for (attribute, expected) in [
        (
            onnx::AttributeProto {
                name: "missing_type".into(),
                r#type: onnx::attribute_proto::AttributeType::TypeProto as i32,
                ..Default::default()
            },
            "missing tp",
        ),
        (
            onnx::AttributeProto {
                name: "unknown_kind".into(),
                r#type: 99,
                ..Default::default()
            },
            "unknown kind",
        ),
        (
            onnx::AttributeProto {
                name: "untyped_ref".into(),
                ref_attr_name: "formal".into(),
                ..Default::default()
            },
            "outside a FunctionProto",
        ),
    ] {
        let mut source = identity(None);
        source.graph.as_mut().unwrap().node[0]
            .attribute
            .push(attribute);
        let error = load_model_ir_bytes(&source.encode_to_vec())
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "{error}");
    }
    let undefined_initializer = model(onnx::GraphProto {
        initializer: vec![onnx::TensorProto {
            name: "untyped_payload".into(),
            raw_data: vec![1],
            ..Default::default()
        }],
        ..Default::default()
    });
    assert!(
        load_model_ir_bytes(&undefined_initializer.encode_to_vec()).is_err(),
        "an absent value type is unknown, but an initializer payload must have a concrete dtype"
    );
    let missing = onnx::TypeProto {
        value: Some(onnx::type_proto::Value::SequenceType(Box::default())),
        ..Default::default()
    };
    let error = load_model_ir_bytes(&identity(Some(missing)).encode_to_vec()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("missing its element/value descriptor"),
        "{error}"
    );
    let negative = tensor(Some(onnx::TensorShapeProto {
        dim: vec![onnx::tensor_shape_proto::Dimension {
            value: Some(onnx::tensor_shape_proto::dimension::Value::DimValue(-1)),
            ..Default::default()
        }],
    }));
    assert!(
        load_model_ir_bytes(&identity(Some(negative)).encode_to_vec())
            .unwrap_err()
            .to_string()
            .contains("negative dimension")
    );
}

#[test]
fn arbitrary_string_tensor_elements_survive_initializers_and_attributes() {
    let payloads = vec![vec![0xff, 0x00, 0xfe], Vec::new(), b"a\0b".to_vec()];
    let strings = onnx::TensorProto {
        name: "bytes".into(),
        data_type: DataType::String.to_onnx(),
        dims: vec![3],
        string_data: payloads.clone(),
        ..Default::default()
    };
    let source = model(onnx::GraphProto {
        initializer: vec![strings.clone()],
        output: vec![info("out", None)],
        node: vec![onnx::NodeProto {
            op_type: "Identity".into(),
            input: vec!["bytes".into()],
            output: vec!["out".into()],
            attribute: vec![onnx::AttributeProto {
                name: "opaque_tensor".into(),
                r#type: onnx::attribute_proto::AttributeType::Tensor as i32,
                t: Some(strings),
                ..Default::default()
            }],
            ..Default::default()
        }],
        ..Default::default()
    });
    let graph = load_model_ir_bytes(&source.encode_to_vec()).unwrap();
    let encoded = encode_model_proto(&Model::new(&graph))
        .unwrap()
        .graph
        .unwrap();
    assert_eq!(encoded.initializer[0].string_data, payloads);
    assert_eq!(
        encoded.node[0].attribute[0].t.as_ref().unwrap().string_data,
        payloads
    );
}

fn sparse() -> onnx::SparseTensorProto {
    onnx::SparseTensorProto {
        values: Some(onnx::TensorProto {
            name: "weight".into(),
            data_type: DataType::Float32.to_onnx(),
            dims: vec![1],
            raw_data: 3.0f32.to_le_bytes().to_vec(),
            ..Default::default()
        }),
        indices: Some(onnx::TensorProto {
            data_type: DataType::Int64.to_onnx(),
            dims: vec![1, 2],
            int64_data: vec![0, 1],
            ..Default::default()
        }),
        dims: vec![2, 2],
    }
}

#[test]
fn sparse_initializer_payload_identity_and_input_overlap_roundtrip() {
    for overridable in [false, true] {
        let sparse = sparse();
        let source = model(onnx::GraphProto {
            input: if overridable {
                vec![info("weight", Some(tensor(None)))]
            } else {
                vec![]
            },
            output: vec![info("out", Some(tensor(None)))],
            sparse_initializer: vec![sparse.clone()],
            node: vec![onnx::NodeProto {
                op_type: "Identity".into(),
                input: vec!["weight".into()],
                output: vec!["out".into()],
                ..Default::default()
            }],
            ..Default::default()
        });
        let graph = load_model_ir_bytes(&source.encode_to_vec()).unwrap();
        let (&id, preserved) = graph.sparse_initializers.iter().next().unwrap();
        assert_eq!(preserved, &sparse);
        assert_eq!(graph.inputs.contains(&id), overridable);
        assert_eq!(graph.is_constant_initializer(id), !overridable);
        let encoded = encode_model_proto(&Model::new(&graph))
            .unwrap()
            .graph
            .unwrap();
        assert_eq!(encoded.sparse_initializer, vec![sparse]);
        assert_eq!(encoded.input.len(), usize::from(overridable));
        let pattern = Pattern::parse_ort("(x:const tensor[...])=>return Identity(x)").unwrap();
        assert_eq!(
            MatchSession::new(&graph)
                .unwrap()
                .match_at(
                    &pattern,
                    graph.nodes.keys().next().unwrap(),
                    Default::default()
                )
                .unwrap()
                .is_some(),
            !overridable
        );
    }
}

#[test]
fn pre_ir4_initializer_inputs_are_constants_without_losing_source_presence() {
    let mut source = model(onnx::GraphProto {
        input: vec![info("weight", None)],
        output: vec![info("out", Some(tensor(None)))],
        initializer: vec![onnx::TensorProto {
            name: "weight".into(),
            data_type: DataType::Float32.to_onnx(),
            dims: vec![2],
            raw_data: [3.0f32, 4.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
            ..Default::default()
        }],
        node: vec![onnx::NodeProto {
            op_type: "Identity".into(),
            input: vec!["weight".into()],
            output: vec!["out".into()],
            ..Default::default()
        }],
        ..Default::default()
    });
    for version in [3, 4] {
        source.ir_version = version;
        let graph = load_model_ir_bytes(&source.encode_to_vec()).unwrap();
        let value = graph.inputs[0];
        assert_eq!(graph.ir_version, Some(version));
        assert_eq!(graph.is_constant_initializer(value), version < 4);
        assert_eq!(
            graph.value(value).tensor_dtype(),
            (version < 4).then_some(DataType::Float32)
        );
        assert_eq!(graph.value(value).tensor_shape().is_some(), version < 4);
        let encoded = encode_model_proto(&Model::new(&graph)).unwrap();
        assert_eq!(encoded.ir_version, version);
        assert_eq!(encoded.graph.unwrap().input[0].name, "weight");
        assert!(graph.value(value).declared_type.is_none());
        let mut execution = graph.clone();
        onnx_runtime_loader::prepare_graph_for_execution(&mut execution).unwrap();
        assert_eq!(execution.inputs.len(), usize::from(version >= 4));
        assert_eq!(execution.ir_version, Some(4));
        let lowered = encode_model_proto(&Model::new(&execution)).unwrap();
        assert_eq!(lowered.ir_version, 4);
        assert_eq!(
            lowered.graph.unwrap().input.len(),
            usize::from(version >= 4)
        );
        assert_eq!(
            graph.inputs,
            vec![value],
            "execution lowering must not mutate the source snapshot"
        );
        if version < 4 {
            let runtime = onnx_runtime_loader::load_model_bytes(&source.encode_to_vec()).unwrap();
            let encoded = encode_model_proto(&Model::new(&runtime)).unwrap();
            assert_eq!(encoded.ir_version, 4);
            assert!(encoded.graph.unwrap().input.is_empty());
        }
    }
}

#[test]
fn dense_initializer_remains_an_overridable_graph_input_not_a_literal() {
    let source = model(onnx::GraphProto {
        input: vec![info("weight", Some(tensor(None)))],
        output: vec![info("out", Some(tensor(None)))],
        initializer: vec![onnx::TensorProto {
            name: "weight".into(),
            data_type: DataType::Float32.to_onnx(),
            raw_data: 3.0f32.to_le_bytes().to_vec(),
            ..Default::default()
        }],
        node: vec![onnx::NodeProto {
            op_type: "Identity".into(),
            input: vec!["weight".into()],
            output: vec!["out".into()],
            ..Default::default()
        }],
        ..Default::default()
    });
    let graph = load_model_ir_bytes(&source.encode_to_vec()).unwrap();
    let id = graph.inputs[0];
    assert!(graph.initializers.contains_key(&id));
    assert!(!graph.is_constant_initializer(id));
    assert_eq!(graph.value(id).tensor_dtype(), Some(DataType::Float32));
    assert_eq!(
        graph.value(id).tensor_shape(),
        None,
        "the default's scalar shape must not specialize an unknown-rank overridable input"
    );
    let encoded = encode_model_proto(&Model::new(&graph))
        .unwrap()
        .graph
        .unwrap();
    assert_eq!(encoded.input[0].name, encoded.initializer[0].name);
    let pattern = Pattern::parse_ort("()=>return Identity(3.0)").unwrap();
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(
                &pattern,
                graph.nodes.keys().next().unwrap(),
                Default::default()
            )
            .unwrap()
            .is_none()
    );
}

#[test]
fn malformed_sparse_initializers_fail_instead_of_disappearing() {
    let mut missing_payload = sparse();
    missing_payload.values.as_mut().unwrap().raw_data.clear();
    let mut out_of_range = sparse();
    out_of_range.indices.as_mut().unwrap().int64_data = vec![0, 5];
    let mut invalid_external = sparse();
    let external = invalid_external.values.as_mut().unwrap();
    external.raw_data.clear();
    external.data_location = onnx::tensor_proto::DataLocation::External as i32;
    external.external_data = vec![
        onnx::StringStringEntryProto {
            key: "location".into(),
            value: "values.bin".into(),
        },
        onnx::StringStringEntryProto {
            key: "offset".into(),
            value: "not-an-offset".into(),
        },
    ];
    for sparse in [
        onnx::SparseTensorProto {
            values: None,
            ..sparse()
        },
        onnx::SparseTensorProto {
            indices: None,
            ..sparse()
        },
        onnx::SparseTensorProto {
            dims: vec![-1],
            ..sparse()
        },
        missing_payload,
        out_of_range,
        invalid_external,
    ] {
        let source = model(onnx::GraphProto {
            sparse_initializer: vec![sparse],
            ..Default::default()
        });
        assert!(load_model_ir_bytes(&source.encode_to_vec()).is_err());
    }
}

#[test]
fn mismatched_initializer_input_type_is_an_error_not_fake_evidence() {
    let mut source = model(onnx::GraphProto {
        input: vec![info("weight", Some(tensor(None)))],
        output: vec![info("weight", Some(tensor(None)))],
        initializer: vec![onnx::TensorProto {
            name: "weight".into(),
            data_type: DataType::Int64.to_onnx(),
            int64_data: vec![1],
            ..Default::default()
        }],
        ..Default::default()
    });
    assert!(
        load_model_ir_bytes(&source.encode_to_vec())
            .unwrap_err()
            .to_string()
            .contains("dtype")
    );
    source.graph.as_mut().unwrap().initializer[0].data_type = DataType::Float32.to_onnx();
    source.graph.as_mut().unwrap().initializer[0]
        .int64_data
        .clear();
    source.graph.as_mut().unwrap().initializer[0].float_data = vec![1.0];
    assert!(load_model_ir_bytes(&source.encode_to_vec()).is_ok());
}

#[test]
fn recursive_type_attributes_preserve_unknown_opaque_and_denotations() {
    let ty = onnx::TypeProto {
        value: Some(onnx::type_proto::Value::SequenceType(Box::new(
            onnx::type_proto::Sequence {
                elem_type: Some(Box::new(onnx::TypeProto {
                    value: Some(onnx::type_proto::Value::OpaqueType(
                        onnx::type_proto::Opaque {
                            domain: "vendor".into(),
                            name: "handle".into(),
                        },
                    )),
                    denotation: "resource".into(),
                })),
            },
        ))),
        denotation: "collection".into(),
    };
    let attribute = onnx::AttributeProto {
        name: "types".into(),
        r#type: onnx::attribute_proto::AttributeType::TypeProtos as i32,
        type_protos: vec![ty.clone(), tensor(None), onnx::TypeProto::default()],
        ..Default::default()
    };
    let mut source = identity(Some(tensor(None)));
    source.graph.as_mut().unwrap().node[0]
        .attribute
        .push(attribute.clone());
    let graph = load_model_ir_bytes(&source.encode_to_vec()).unwrap();
    let encoded = encode_model_proto(&Model::new(&graph))
        .unwrap()
        .graph
        .unwrap();
    assert_eq!(encoded.node[0].attribute, vec![attribute]);
}

#[test]
fn passive_value_info_is_preserved_but_does_not_define_a_lexical_source() {
    let mut source = identity(None);
    let metadata = info(
        "ghost",
        Some(onnx::TypeProto {
            value: Some(onnx::type_proto::Value::OpaqueType(
                onnx::type_proto::Opaque {
                    domain: "local".into(),
                    name: "Metadata".into(),
                },
            )),
            ..Default::default()
        }),
    );
    source
        .graph
        .as_mut()
        .unwrap()
        .value_info
        .push(metadata.clone());
    let graph = load_model_ir_bytes(&source.encode_to_vec()).unwrap();
    let encoded = encode_model_proto(&Model::new(&graph))
        .unwrap()
        .graph
        .unwrap();
    assert!(encoded.value_info.iter().any(|info| info == &metadata));
    source.graph.as_mut().unwrap().node[0]
        .attribute
        .push(onnx::AttributeProto {
            name: "body".into(),
            r#type: onnx::attribute_proto::AttributeType::Graph as i32,
            g: Some(onnx::GraphProto {
                output: vec![info("z", None)],
                node: vec![onnx::NodeProto {
                    op_type: "Identity".into(),
                    input: vec!["ghost".into()],
                    output: vec!["z".into()],
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        });
    let error = load_model_ir_bytes(&source.encode_to_vec())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("ghost") && error.contains("no visible ancestor"),
        "{error}"
    );
}

#[test]
fn loaded_graph_lists_match_each_authoritative_body_once() {
    let body = identity(Some(tensor(None))).graph.unwrap();
    let mut source = identity(Some(tensor(None)));
    source.opset_import.push(onnx::OperatorSetIdProto {
        domain: "custom".into(),
        version: 1,
    });
    let owner = &mut source.graph.as_mut().unwrap().node[0];
    owner.op_type = "Region".into();
    owner.domain = "custom".into();
    owner.attribute.push(onnx::AttributeProto {
        name: "bodies".into(),
        r#type: onnx::attribute_proto::AttributeType::Graphs as i32,
        graphs: vec![body.clone(), body],
        ..Default::default()
    });
    let mut graph = load_model_ir_bytes(&source.encode_to_vec()).unwrap();
    let root = graph.nodes.keys().next().unwrap();
    let pattern = PatternLibrary::parse_native(
        "pattern Body(x){return Identity(x)} pattern Root(x){return custom.Region(x,bodies=[Body,Body])}",
        &|domain, op| domain.is_empty() && op == "Identity",
    ).unwrap().pattern("Root").unwrap();
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .is_some()
    );
    graph
        .subgraphs
        .get_mut(&(root, onnx_runtime_ir::GraphAttribute::element("bodies", 1)))
        .unwrap()
        .nodes
        .values_mut()
        .next()
        .unwrap()
        .op_type = "Neg".into();
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .is_none()
    );
    let encoded = encode_model_proto(&Model::new(&graph)).unwrap();
    assert_eq!(
        encoded.graph.unwrap().node[0].attribute[0].graphs[1].node[0].op_type,
        "Neg"
    );
}

#[test]
fn nested_external_initializer_shadows_ancestor_and_retains_live_bytes() {
    let directory = std::env::temp_dir().join(format!(
        "nxrt-ir-fidelity-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&directory).unwrap();
    let payload = 7.0f32.to_le_bytes().to_vec();
    std::fs::write(directory.join("nested.bin"), &payload).unwrap();
    let mut source = identity(Some(tensor(None)));
    source.opset_import.push(onnx::OperatorSetIdProto {
        domain: "custom".into(),
        version: 1,
    });
    let mut body = identity(Some(tensor(None))).graph.unwrap();
    body.input.clear();
    body.initializer.push(onnx::TensorProto {
        name: "x".into(),
        data_type: DataType::Float32.to_onnx(),
        data_location: onnx::tensor_proto::DataLocation::External as i32,
        external_data: vec![onnx::StringStringEntryProto {
            key: "location".into(),
            value: "nested.bin".into(),
        }],
        ..Default::default()
    });
    let owner = &mut source.graph.as_mut().unwrap().node[0];
    owner.op_type = "Region".into();
    owner.domain = "custom".into();
    owner.attribute.push(onnx::AttributeProto {
        name: "body".into(),
        r#type: onnx::attribute_proto::AttributeType::Graph as i32,
        g: Some(body),
        ..Default::default()
    });
    let (graph, store) =
        onnx_runtime_loader::load_model_ir_bytes_with_weights(&source.encode_to_vec(), &directory)
            .unwrap();
    let root = graph.nodes.keys().next().unwrap();
    let child = &graph.subgraphs[&(root, "body".into())];
    let (&id, weight) = child.initializers.iter().next().unwrap();
    assert_eq!(store.bytes(weight), Some(payload.as_slice()));
    assert!(child.is_constant_initializer(id));
    let pattern = PatternLibrary::parse_ort(
        "pattern Body(x){return Identity(x)} pattern Root(x){owner=custom.Region(x,body=Body);return owner}",
        &|domain, op| domain.is_empty() && op == "Identity",
    )
    .unwrap()
    .pattern("Root")
    .unwrap();
    let session = MatchSession::new(&graph).unwrap();
    let matched = session
        .match_at(&pattern, root, Default::default())
        .unwrap()
        .unwrap();
    assert!(
        matched
            .instance("owner.body")
            .unwrap()
            .resolve_input("x", &graph)
            .is_err()
    );
    let encoded = encode_model_proto(&Model::new(&graph).with_weights(&store)).unwrap();
    assert_eq!(
        encoded.graph.unwrap().node[0].attribute[0]
            .g
            .as_ref()
            .unwrap()
            .initializer[0]
            .raw_data,
        payload
    );
    drop(session);
    drop(graph);
    drop(store);
    std::fs::remove_file(directory.join("nested.bin")).unwrap();
    std::fs::remove_dir(directory).unwrap();
}

#[test]
fn literal_graph_name_cannot_alias_a_graph_list_element() {
    let body = identity(None).graph.unwrap();
    let mut single = body.clone();
    single.node[0].op_type = "Neg".into();
    let mut source = identity(None);
    source.opset_import.push(onnx::OperatorSetIdProto {
        domain: "custom".into(),
        version: 1,
    });
    let owner = &mut source.graph.as_mut().unwrap().node[0];
    owner.domain = "custom".into();
    owner.op_type = "Region".into();
    owner.attribute = vec![
        onnx::AttributeProto {
            name: "bodies[0]".into(),
            r#type: onnx::attribute_proto::AttributeType::Graph as i32,
            g: Some(single),
            ..Default::default()
        },
        onnx::AttributeProto {
            name: "bodies".into(),
            r#type: onnx::attribute_proto::AttributeType::Graphs as i32,
            graphs: vec![body],
            ..Default::default()
        },
    ];
    let graph = load_model_ir_bytes(&source.encode_to_vec()).unwrap();
    let root = graph.nodes.keys().next().unwrap();
    assert_eq!(graph.subgraphs.len(), 2);
    assert_eq!(
        graph.subgraphs[&(root, "bodies[0]".into())]
            .nodes
            .values()
            .next()
            .unwrap()
            .op_type,
        "Neg"
    );
    let list = onnx_runtime_ir::GraphAttribute::element("bodies", 0);
    assert_eq!(
        graph.subgraphs[&(root, list)]
            .nodes
            .values()
            .next()
            .unwrap()
            .op_type,
        "Identity"
    );
    MatchSession::new(&graph).unwrap();
    let encoded = encode_model_proto(&Model::new(&graph))
        .unwrap()
        .graph
        .unwrap();
    let attrs = &encoded.node[0].attribute;
    assert_eq!(
        attrs
            .iter()
            .find(|a| a.name == "bodies[0]")
            .unwrap()
            .g
            .as_ref()
            .unwrap()
            .node[0]
            .op_type,
        "Neg"
    );
    assert_eq!(
        attrs.iter().find(|a| a.name == "bodies").unwrap().graphs[0].node[0].op_type,
        "Identity"
    );
}

#[test]
fn unbound_function_attribute_constraints_are_not_ordinary_nonmatches() {
    let mut graph = onnx_runtime_ir::Graph::new();
    let x = graph.create_named_value("x", DataType::Float32, vec![]);
    let y = graph.create_named_value("y", DataType::Float32, vec![]);
    graph.add_input(x);
    let mut node = onnx_runtime_ir::Node::new(
        onnx_runtime_ir::NodeId(0),
        "Identity",
        vec![Some(x)],
        vec![y],
    );
    node.attributes.insert(
        "axis".into(),
        onnx_runtime_ir::Attribute::Ref {
            name: "formal_axis".into(),
            kind: onnx::attribute_proto::AttributeType::Int,
        },
    );
    let root = graph.insert_node(node);
    let session = MatchSession::new(&graph).unwrap();
    assert!(
        session
            .match_at(
                &Pattern::parse_ort("(x)=>return Identity(x)").unwrap(),
                root,
                Default::default(),
            )
            .unwrap()
            .is_some()
    );
    let constrained = Pattern::parse_ort("(x)=>return Identity(x,axis=0)").unwrap();
    let error = session
        .match_at(&constrained, root, Default::default())
        .unwrap_err();
    assert_eq!(
        error.kind,
        onnx_runtime_ir::pattern::PatternErrorKind::MissingMetadata
    );
    assert!(error.message.contains("formal_axis"));
    assert!(error.message.contains("bind"));
}
