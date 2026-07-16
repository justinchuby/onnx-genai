use std::collections::BTreeSet;

use onnx_rs::{Json, Model, Text, TextCodec, TextProto, load_model, save_model};
use onnx_runtime_loader::proto::{
    FILE_DESCRIPTOR_SET,
    onnx::{
        AttributeProto, DeviceConfigurationProto, FunctionProto, GraphProto, IntIntListEntryProto,
        ModelProto, NodeDeviceConfigurationProto, NodeProto, OperatorSetIdProto, ShardedDimProto,
        ShardingSpecProto, SimpleShardedDimProto, SparseTensorProto, StringStringEntryProto,
        TensorAnnotation, TensorProto, TensorShapeProto, TrainingInfoProto, TypeProto,
        ValueInfoProto, attribute_proto, simple_sharded_dim_proto, tensor_proto,
        tensor_shape_proto, type_proto,
    },
};
use prost::Message;
use prost_reflect::DescriptorPool;

const ALL_DTYPES: &[tensor_proto::DataType] = &[
    tensor_proto::DataType::Undefined,
    tensor_proto::DataType::Float,
    tensor_proto::DataType::Uint8,
    tensor_proto::DataType::Int8,
    tensor_proto::DataType::Uint16,
    tensor_proto::DataType::Int16,
    tensor_proto::DataType::Int32,
    tensor_proto::DataType::Int64,
    tensor_proto::DataType::String,
    tensor_proto::DataType::Bool,
    tensor_proto::DataType::Float16,
    tensor_proto::DataType::Double,
    tensor_proto::DataType::Uint32,
    tensor_proto::DataType::Uint64,
    tensor_proto::DataType::Complex64,
    tensor_proto::DataType::Complex128,
    tensor_proto::DataType::Bfloat16,
    tensor_proto::DataType::Float8e4m3fn,
    tensor_proto::DataType::Float8e4m3fnuz,
    tensor_proto::DataType::Float8e5m2,
    tensor_proto::DataType::Float8e5m2fnuz,
    tensor_proto::DataType::Uint4,
    tensor_proto::DataType::Int4,
    tensor_proto::DataType::Float4e2m1,
    tensor_proto::DataType::Float8e8m0,
    tensor_proto::DataType::Uint2,
    tensor_proto::DataType::Int2,
];

fn entry(key: &str, value: &str) -> StringStringEntryProto {
    StringStringEntryProto {
        key: key.into(),
        value: value.into(),
    }
}

fn shape(dims: &[i64]) -> TensorShapeProto {
    TensorShapeProto {
        dim: dims
            .iter()
            .map(|&value| tensor_shape_proto::Dimension {
                value: Some(tensor_shape_proto::dimension::Value::DimValue(value)),
                denotation: "DATA_BATCH".into(),
            })
            .collect(),
    }
}

fn tensor_type(dtype: tensor_proto::DataType, dims: &[i64]) -> TypeProto {
    TypeProto {
        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
            elem_type: dtype as i32,
            shape: Some(shape(dims)),
        })),
        denotation: "TENSOR".into(),
    }
}

fn sparse_tensor() -> SparseTensorProto {
    SparseTensorProto {
        values: Some(TensorProto {
            dims: vec![2],
            data_type: tensor_proto::DataType::Float as i32,
            float_data: vec![1.0, -2.0],
            name: "sparse_weight".into(),
            ..Default::default()
        }),
        indices: Some(TensorProto {
            dims: vec![2],
            data_type: tensor_proto::DataType::Int64 as i32,
            int64_data: vec![0, 3],
            ..Default::default()
        }),
        dims: vec![2, 2],
    }
}

fn typed_tensor(dtype: tensor_proto::DataType, name: String) -> TensorProto {
    let mut tensor = TensorProto {
        dims: vec![1],
        data_type: dtype as i32,
        name,
        doc_string: "typed payload".into(),
        metadata_props: vec![entry("encoding", "typed")],
        ..Default::default()
    };
    match dtype {
        tensor_proto::DataType::Undefined => {}
        tensor_proto::DataType::Float => tensor.float_data = vec![1.25],
        tensor_proto::DataType::Double => tensor.double_data = vec![2.5],
        tensor_proto::DataType::Complex64 => tensor.float_data = vec![1.0, -2.0],
        tensor_proto::DataType::Complex128 => tensor.double_data = vec![1.0, -2.0],
        tensor_proto::DataType::Int64 => tensor.int64_data = vec![-7],
        tensor_proto::DataType::String => tensor.string_data = vec![b"onnx".to_vec()],
        tensor_proto::DataType::Uint32 | tensor_proto::DataType::Uint64 => {
            tensor.uint64_data = vec![7]
        }
        tensor_proto::DataType::Uint4
        | tensor_proto::DataType::Int4
        | tensor_proto::DataType::Float4e2m1 => {
            tensor.dims = vec![2];
            tensor.int32_data = vec![0x21];
        }
        tensor_proto::DataType::Uint2 | tensor_proto::DataType::Int2 => {
            tensor.dims = vec![4];
            tensor.int32_data = vec![0b11_10_01_00];
        }
        _ => tensor.int32_data = vec![1],
    }
    tensor
}

fn raw_tensor(dtype: tensor_proto::DataType, name: String) -> TensorProto {
    let element_bytes = match dtype {
        tensor_proto::DataType::Double
        | tensor_proto::DataType::Int64
        | tensor_proto::DataType::Uint64
        | tensor_proto::DataType::Complex64 => 8,
        tensor_proto::DataType::Complex128 => 16,
        tensor_proto::DataType::Float
        | tensor_proto::DataType::Int32
        | tensor_proto::DataType::Uint32 => 4,
        tensor_proto::DataType::Float16
        | tensor_proto::DataType::Bfloat16
        | tensor_proto::DataType::Int16
        | tensor_proto::DataType::Uint16 => 2,
        _ => 1,
    };
    TensorProto {
        dims: vec![1],
        data_type: dtype as i32,
        name,
        raw_data: vec![0x5a; element_bytes],
        metadata_props: vec![entry("encoding", "raw")],
        ..Default::default()
    }
}

fn all_payload_tensors() -> Vec<TensorProto> {
    ALL_DTYPES
        .iter()
        .copied()
        .filter(|dtype| *dtype != tensor_proto::DataType::Undefined)
        .flat_map(|dtype| {
            let stem = dtype.as_str_name().to_ascii_lowercase();
            let mut tensors = vec![typed_tensor(dtype, format!("{stem}_typed"))];
            if dtype != tensor_proto::DataType::String {
                tensors.push(raw_tensor(dtype, format!("{stem}_raw")));
            }
            tensors
        })
        .collect()
}

fn full_spec_proto() -> ModelProto {
    let nested_types = vec![
        tensor_type(tensor_proto::DataType::Float, &[2, 3]),
        TypeProto {
            value: Some(type_proto::Value::SparseTensorType(
                type_proto::SparseTensor {
                    elem_type: tensor_proto::DataType::Int4 as i32,
                    shape: Some(shape(&[5])),
                },
            )),
            denotation: "SPARSE".into(),
        },
        TypeProto {
            value: Some(type_proto::Value::SequenceType(Box::new(
                type_proto::Sequence {
                    elem_type: Some(Box::new(TypeProto {
                        value: Some(type_proto::Value::OptionalType(Box::new(
                            type_proto::Optional {
                                elem_type: Some(Box::new(tensor_type(
                                    tensor_proto::DataType::Float8e5m2fnuz,
                                    &[-1],
                                ))),
                            },
                        ))),
                        denotation: "OPTIONAL".into(),
                    })),
                },
            ))),
            denotation: "SEQUENCE".into(),
        },
        TypeProto {
            value: Some(type_proto::Value::MapType(Box::new(type_proto::Map {
                key_type: tensor_proto::DataType::String as i32,
                value_type: Some(Box::new(tensor_type(
                    tensor_proto::DataType::Complex128,
                    &[],
                ))),
            }))),
            denotation: "MAP".into(),
        },
    ];

    let sparse = sparse_tensor();
    let nested_graph = GraphProto {
        name: "nested".into(),
        doc_string: "subgraph documentation".into(),
        metadata_props: vec![entry("scope", "nested")],
        ..Default::default()
    };
    let attributes = vec![
        AttributeProto {
            name: "sparse".into(),
            r#type: attribute_proto::AttributeType::SparseTensor as i32,
            sparse_tensor: Some(sparse.clone()),
            doc_string: "sparse attribute".into(),
            ..Default::default()
        },
        AttributeProto {
            name: "tensors".into(),
            r#type: attribute_proto::AttributeType::Tensors as i32,
            tensors: vec![typed_tensor(
                tensor_proto::DataType::Float4e2m1,
                "tensor_attr".into(),
            )],
            ..Default::default()
        },
        AttributeProto {
            name: "external_tensor".into(),
            r#type: attribute_proto::AttributeType::Tensor as i32,
            t: Some(TensorProto {
                dims: vec![1],
                data_type: tensor_proto::DataType::Float as i32,
                name: "external_attr".into(),
                external_data: vec![entry("location", "weights.bin")],
                data_location: tensor_proto::DataLocation::External as i32,
                ..Default::default()
            }),
            ..Default::default()
        },
        AttributeProto {
            name: "graphs".into(),
            r#type: attribute_proto::AttributeType::Graphs as i32,
            graphs: vec![nested_graph.clone()],
            ..Default::default()
        },
        AttributeProto {
            name: "sparse_tensors".into(),
            r#type: attribute_proto::AttributeType::SparseTensors as i32,
            sparse_tensors: vec![sparse.clone()],
            ..Default::default()
        },
        AttributeProto {
            name: "type_protos".into(),
            r#type: attribute_proto::AttributeType::TypeProtos as i32,
            type_protos: nested_types.clone(),
            ..Default::default()
        },
        AttributeProto {
            name: "type_proto".into(),
            r#type: attribute_proto::AttributeType::TypeProto as i32,
            tp: Some(nested_types[0].clone()),
            ..Default::default()
        },
        AttributeProto {
            name: "graph".into(),
            r#type: attribute_proto::AttributeType::Graph as i32,
            g: Some(nested_graph),
            ..Default::default()
        },
        AttributeProto {
            name: "opaque_string".into(),
            r#type: attribute_proto::AttributeType::String as i32,
            s: vec![0xff, 0xfe],
            ..Default::default()
        },
        AttributeProto {
            name: "opaque_strings".into(),
            r#type: attribute_proto::AttributeType::Strings as i32,
            strings: vec![b"readable".to_vec(), vec![0xff]],
            ..Default::default()
        },
    ];

    ModelProto {
        ir_version: 13,
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
        producer_name: "onnx-rs-full-spec".into(),
        producer_version: "1".into(),
        domain: "test.model".into(),
        model_version: 42,
        doc_string: "full bound ONNX schema fixture".into(),
        graph: Some(GraphProto {
            node: vec![NodeProto {
                input: vec!["cond".into()],
                output: vec!["Y".into()],
                name: "full_node".into(),
                op_type: "If".into(),
                domain: String::new(),
                overload: "overload-id".into(),
                attribute: attributes,
                doc_string: "node documentation".into(),
                metadata_props: vec![entry("node-key", "node-value")],
                device_configurations: vec![NodeDeviceConfigurationProto {
                    configuration_id: "mesh".into(),
                    sharding_spec: vec![ShardingSpecProto {
                        tensor_name: "Y".into(),
                        device: vec![0, 1, 2, 3],
                        index_to_device_group_map: vec![IntIntListEntryProto {
                            key: 4,
                            value: vec![0, 2],
                        }],
                        sharded_dim: vec![
                            ShardedDimProto {
                                axis: 0,
                                simple_sharding: vec![SimpleShardedDimProto {
                                    dim: Some(simple_sharded_dim_proto::Dim::DimValue(4)),
                                    num_shards: 4,
                                }],
                            },
                            ShardedDimProto {
                                axis: 1,
                                simple_sharding: vec![SimpleShardedDimProto {
                                    dim: Some(simple_sharded_dim_proto::Dim::DimParam("N".into())),
                                    num_shards: 2,
                                }],
                            },
                        ],
                    }],
                    pipeline_stage: 2,
                }],
            }],
            name: "full_graph".into(),
            initializer: {
                let mut tensors = all_payload_tensors();
                tensors.push(TensorProto {
                    dims: vec![4],
                    data_type: tensor_proto::DataType::Float as i32,
                    segment: Some(tensor_proto::Segment { begin: 1, end: 3 }),
                    float_data: vec![1.0, 2.0],
                    name: "segmented".into(),
                    doc_string: "segment coverage".into(),
                    external_data: vec![entry("checksum", "abc")],
                    data_location: tensor_proto::DataLocation::Default as i32,
                    metadata_props: vec![entry("tensor-key", "tensor-value")],
                    ..Default::default()
                });
                tensors
            },
            sparse_initializer: vec![sparse],
            doc_string: "graph documentation".into(),
            input: vec![
                ValueInfoProto {
                    name: "cond".into(),
                    r#type: Some(tensor_type(tensor_proto::DataType::Bool, &[])),
                    doc_string: "condition".into(),
                    metadata_props: vec![entry("value-key", "value-value")],
                },
                ValueInfoProto {
                    name: "sequence_input".into(),
                    r#type: Some(nested_types[2].clone()),
                    doc_string: "container input".into(),
                    metadata_props: vec![entry("kind", "sequence")],
                },
            ],
            output: vec![ValueInfoProto {
                name: "Y".into(),
                r#type: Some(tensor_type(tensor_proto::DataType::Float, &[1])),
                ..Default::default()
            }],
            value_info: nested_types
                .iter()
                .enumerate()
                .map(|(index, r#type)| ValueInfoProto {
                    name: format!("nested_type_{index}"),
                    r#type: Some(r#type.clone()),
                    doc_string: "nested type".into(),
                    metadata_props: vec![entry("kind", "container")],
                })
                .collect(),
            quantization_annotation: vec![TensorAnnotation {
                tensor_name: "Y".into(),
                quant_parameter_tensor_names: vec![
                    entry("SCALE_TENSOR", "scale"),
                    entry("ZERO_POINT_TENSOR", "zero"),
                ],
            }],
            metadata_props: vec![entry("graph-key", "graph-value")],
        }),
        metadata_props: vec![entry("model-key", "model-value")],
        training_info: vec![TrainingInfoProto {
            initialization: Some(GraphProto {
                name: "training_init".into(),
                ..Default::default()
            }),
            algorithm: Some(GraphProto {
                name: "training_algorithm".into(),
                ..Default::default()
            }),
            initialization_binding: vec![entry("weight", "initialized_weight")],
            update_binding: vec![entry("weight", "updated_weight")],
        }],
        functions: vec![FunctionProto {
            name: "LocalFunction".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            attribute: vec!["alpha".into()],
            attribute_proto: vec![AttributeProto {
                name: "beta".into(),
                r#type: attribute_proto::AttributeType::Float as i32,
                f: 0.5,
                ..Default::default()
            }],
            node: vec![NodeProto {
                input: vec!["X".into()],
                output: vec!["Y".into()],
                op_type: "LeakyRelu".into(),
                attribute: vec![AttributeProto {
                    name: "alpha".into(),
                    ref_attr_name: "alpha".into(),
                    r#type: attribute_proto::AttributeType::Float as i32,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            doc_string: "local function".into(),
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 24,
            }],
            domain: "local.test".into(),
            overload: "v1".into(),
            value_info: vec![ValueInfoProto {
                name: "X".into(),
                r#type: Some(tensor_type(tensor_proto::DataType::Bfloat16, &[1])),
                ..Default::default()
            }],
            metadata_props: vec![entry("function-key", "function-value")],
        }],
        configuration: vec![DeviceConfigurationProto {
            name: "mesh".into(),
            num_devices: 4,
            device: vec![
                "cuda:0".into(),
                "cuda:1".into(),
                "cuda:2".into(),
                "cuda:3".into(),
            ],
        }],
    }
}

fn assert_proto_equal(expected: &ModelProto, actual: &Model) {
    assert_eq!(
        actual.to_proto().unwrap().encode_to_vec(),
        expected.encode_to_vec()
    );
}

#[test]
fn descriptor_inventory_is_the_complete_bound_spec() {
    let pool = DescriptorPool::decode(FILE_DESCRIPTOR_SET).unwrap();
    let messages: BTreeSet<_> = pool
        .all_messages()
        .map(|message| message.full_name().to_string())
        .collect();
    let expected_messages: BTreeSet<_> = [
        "onnx.AttributeProto",
        "onnx.DeviceConfigurationProto",
        "onnx.FunctionProto",
        "onnx.GraphProto",
        "onnx.IntIntListEntryProto",
        "onnx.ModelProto",
        "onnx.NodeDeviceConfigurationProto",
        "onnx.NodeProto",
        "onnx.OperatorSetIdProto",
        "onnx.ShardedDimProto",
        "onnx.ShardingSpecProto",
        "onnx.SimpleShardedDimProto",
        "onnx.SparseTensorProto",
        "onnx.StringStringEntryProto",
        "onnx.TensorAnnotation",
        "onnx.TensorProto",
        "onnx.TensorProto.Segment",
        "onnx.TensorShapeProto",
        "onnx.TensorShapeProto.Dimension",
        "onnx.TrainingInfoProto",
        "onnx.TypeProto",
        "onnx.TypeProto.Map",
        "onnx.TypeProto.Optional",
        "onnx.TypeProto.Sequence",
        "onnx.TypeProto.SparseTensor",
        "onnx.TypeProto.Tensor",
        "onnx.ValueInfoProto",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    assert_eq!(messages, expected_messages);

    let enums: BTreeSet<_> = pool
        .all_enums()
        .map(|value| value.full_name().to_string())
        .collect();
    let expected_enums: BTreeSet<_> = [
        "onnx.AttributeProto.AttributeType",
        "onnx.OperatorStatus",
        "onnx.TensorProto.DataLocation",
        "onnx.TensorProto.DataType",
        "onnx.Version",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    assert_eq!(enums, expected_enums);

    assert!(
        pool.get_message_by_name("onnx.DeviceConfigurationProto")
            .is_some()
    );
    assert!(
        pool.get_message_by_name("onnx.NodeDeviceConfigurationProto")
            .is_some()
    );
    assert!(pool.get_message_by_name("onnx.ShardingSpecProto").is_some());
    let dtype = pool.get_enum_by_name("onnx.TensorProto.DataType").unwrap();
    assert!(dtype.get_value_by_name("FLOAT8E8M0").is_some());
    assert!(dtype.get_value_by_name("INT2").is_some());
    assert!(dtype.get_value_by_name("UINT2").is_some());
}

#[test]
fn every_bound_dtype_round_trips_typed_and_raw_payloads() {
    let proto = full_spec_proto();
    let graph = proto.graph.as_ref().unwrap();
    for dtype in ALL_DTYPES.iter().copied().filter(|value| {
        *value != tensor_proto::DataType::Undefined && *value != tensor_proto::DataType::String
    }) {
        let stem = dtype.as_str_name().to_ascii_lowercase();
        assert!(
            graph
                .initializer
                .iter()
                .any(|tensor| tensor.name == format!("{stem}_typed"))
        );
        assert!(
            graph
                .initializer
                .iter()
                .any(|tensor| tensor.name == format!("{stem}_raw"))
        );
    }
    assert!(
        graph
            .initializer
            .iter()
            .any(|tensor| tensor.name == "string_typed")
    );

    let model = Model::from_proto(proto.clone()).unwrap();
    assert_proto_equal(&proto, &model);
    let node = model.graph.nodes.values().next().unwrap();
    assert!(matches!(
        node.attr("tensors"),
        Some(onnx_rs::ir::Attribute::Tensors(values)) if values.len() == 1
    ));
    assert!(matches!(
        node.attr("sparse_tensors"),
        Some(onnx_rs::ir::Attribute::SparseTensors(values)) if values.len() == 1
    ));
    assert!(matches!(
        node.attr("type_protos"),
        Some(onnx_rs::ir::Attribute::TypeProtos(values)) if values.len() == 4
    ));
    assert_eq!(onnx_rs::ir::DataType::Complex64.to_onnx(), 14);
    assert_eq!(onnx_rs::ir::DataType::Complex128.to_onnx(), 15);
    assert_eq!(onnx_rs::ir::DataType::Undefined.to_onnx(), 0);
    assert_eq!(onnx_rs::ir::DataType::Float8E8M0.to_onnx(), 24);
    assert_eq!(onnx_rs::ir::DataType::Uint2.storage_bytes(5), 2);
    assert_eq!(onnx_rs::ir::DataType::Int2.storage_bytes(5), 2);
}

#[test]
fn full_spec_is_lossless_across_all_three_textual_codecs() {
    let proto = full_spec_proto();
    let model = Model::from_proto(proto.clone()).unwrap();

    let json = Json::serialize(&model, &()).unwrap();
    assert!(json.contains("\"trainingInfo\""));
    assert!(json.contains("\"functions\""));
    assert!(json.contains("\"sparseInitializer\""));
    assert!(json.contains("\"quantizationAnnotation\""));
    assert!(json.contains("\"configuration\""));
    assert!(json.contains("\"deviceConfigurations\""));
    let from_json = Json::deserialize(&json).unwrap();
    assert_proto_equal(&proto, &from_json);

    let textproto = TextProto::serialize(&from_json, &()).unwrap();
    assert!(textproto.contains("training_info"));
    assert!(textproto.contains("functions"));
    assert!(textproto.contains("sparse_initializer"));
    assert!(textproto.contains("type_protos"));
    assert!(textproto.contains("device_configurations"));
    assert!(textproto.contains("sharding_spec"));
    let from_textproto = TextProto::deserialize(&textproto).unwrap();
    assert_proto_equal(&proto, &from_textproto);

    let text = Text::serialize(&from_textproto, &Default::default()).unwrap();
    assert!(
        !text
            .lines()
            .any(|line| line.trim_start().starts_with("proto:"))
    );
    assert!(text.contains("__onnx_extensions_begin__"));
    assert!(text.contains("device_configurations"));
    let from_text = Text::deserialize(&text).unwrap();
    assert_proto_equal(&proto, &from_text);

    let json_again = Json::serialize(&from_text, &()).unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&json_again).unwrap(),
        serde_json::from_str::<serde_json::Value>(&json).unwrap()
    );
}

#[test]
fn readable_body_edits_are_authoritative_over_extensions() {
    let proto = full_spec_proto();
    let model = Model::from_proto(proto.clone()).unwrap();
    let text = Text::serialize(&model, &Default::default()).unwrap();
    assert!(text.contains("type_protos = <4 types>"), "{text}");
    assert!(
        text.contains("sparse_tensors = <1 sparse tensors>"),
        "{text}"
    );
    assert!(text.contains("tensors = <1 tensors>"), "{text}");
    assert!(text.contains("opaque_string = <2 bytes>"), "{text}");
    assert!(text.contains("opaque_strings = <2 strings>"), "{text}");

    let edited = text
        .replacen("full_graph (", "edited_graph (", 1)
        .replacen("bool[] cond", "int64[2] condition", 1)
        .replacen("float[] sequence_input", "uint8[7] sequence_input", 1)
        .replacen("(cond)", "(condition)", 1)
        .replacen(
            "external_tensor = <tensor float[[1]]>",
            "external_tensor = <tensor int64[[2]]>",
            1,
        )
        .replacen("sparse = <sparse tensor>", "sparse = <type>", 1)
        .replacen("type_proto = <type>", "type_proto = <sparse tensor>", 1)
        .replacen("<1 sparse tensors>", "<2 sparse tensors>", 1)
        .replacen("<4 types>", "<2 types>", 1)
        .replacen("<1 tensors>", "<2 tensors>", 1)
        .replacen("opaque_string = <2 bytes>", "opaque_string = \"edited\"", 1)
        .replacen(
            "opaque_strings = <2 strings>",
            "opaque_strings = [\"left\", \"right\"]",
            1,
        )
        .replacen(
            " () => () {",
            " (int32[3] nested_value) => (int32[3] nested_value) {",
            1,
        );
    assert_ne!(edited, text);

    let parsed = Text::deserialize(&edited).unwrap();
    let parsed_proto = parsed.to_proto().unwrap();
    let graph = parsed_proto.graph.as_ref().unwrap();
    assert_eq!(graph.name, "edited_graph");
    let node = &graph.node[0];
    assert_eq!(graph.input[0].name, "condition");
    let condition_type = graph.input[0].r#type.as_ref().unwrap();
    let Some(type_proto::Value::TensorType(condition_type)) = &condition_type.value else {
        panic!("edited graph input must be a tensor");
    };
    assert_eq!(
        condition_type.elem_type,
        tensor_proto::DataType::Int64 as i32
    );
    assert_eq!(
        condition_type.shape.as_ref().unwrap().dim[0].value,
        Some(tensor_shape_proto::dimension::Value::DimValue(2))
    );
    let sequence_type = graph.input[1].r#type.as_ref().unwrap();
    let Some(type_proto::Value::TensorType(sequence_type)) = &sequence_type.value else {
        panic!("edited container projection must become the readable tensor type");
    };
    assert_eq!(
        sequence_type.elem_type,
        tensor_proto::DataType::Uint8 as i32
    );
    assert_eq!(
        sequence_type.shape.as_ref().unwrap().dim[0].value,
        Some(tensor_shape_proto::dimension::Value::DimValue(7))
    );
    assert_eq!(node.input, ["condition"]);
    assert_eq!(node_attribute(node, "type_protos").type_protos.len(), 2);
    assert_eq!(
        node_attribute(node, "type_protos").type_protos,
        proto.graph.as_ref().unwrap().node[0]
            .attribute
            .iter()
            .find(|attribute| attribute.name == "type_protos")
            .unwrap()
            .type_protos[..2]
    );
    assert_eq!(
        node_attribute(node, "sparse_tensors").sparse_tensors.len(),
        2
    );
    assert_eq!(
        node_attribute(node, "sparse_tensors").sparse_tensors[0],
        proto.graph.as_ref().unwrap().node[0]
            .attribute
            .iter()
            .find(|attribute| attribute.name == "sparse_tensors")
            .unwrap()
            .sparse_tensors[0]
    );
    assert_eq!(node_attribute(node, "tensors").tensors.len(), 2);
    assert_eq!(
        node_attribute(node, "tensors").tensors[0],
        proto.graph.as_ref().unwrap().node[0]
            .attribute
            .iter()
            .find(|attribute| attribute.name == "tensors")
            .unwrap()
            .tensors[0]
    );
    assert_eq!(node_attribute(node, "opaque_string").s, b"edited".to_vec());
    assert_eq!(
        node_attribute(node, "opaque_strings").strings,
        vec![b"left".to_vec(), b"right".to_vec()]
    );
    assert_eq!(
        node_attribute(node, "sparse").r#type,
        attribute_proto::AttributeType::TypeProto as i32
    );
    assert!(node_attribute(node, "sparse").tp.is_some());
    assert!(node_attribute(node, "sparse").sparse_tensor.is_none());
    assert_eq!(
        node_attribute(node, "type_proto").r#type,
        attribute_proto::AttributeType::SparseTensor as i32
    );
    assert!(node_attribute(node, "type_proto").sparse_tensor.is_some());
    assert!(node_attribute(node, "type_proto").tp.is_none());
    let tensor = node_attribute(node, "external_tensor").t.as_ref().unwrap();
    assert_eq!(tensor.data_type, tensor_proto::DataType::Int64 as i32);
    assert_eq!(tensor.dims, [2]);
    let nested = node_attribute(node, "graph").g.as_ref().unwrap();
    assert_eq!(nested.input[0].name, "nested_value");
    assert_eq!(nested.output[0].name, "nested_value");
    assert_eq!(node.device_configurations.len(), 1);
    assert_eq!(parsed_proto.configuration[0].name, "mesh");

    let unedited = Text::deserialize(&text).unwrap();
    assert_proto_equal(&proto, &unedited);
}

fn node_attribute<'a>(node: &'a NodeProto, name: &str) -> &'a AttributeProto {
    node.attribute
        .iter()
        .find(|attribute| attribute.name == name)
        .unwrap_or_else(|| panic!("missing attribute {name}"))
}

#[test]
fn full_spec_binary_model_io_is_lossless() {
    let proto = full_spec_proto();
    let model = Model::from_proto(proto.clone()).unwrap();
    let path = std::env::current_dir()
        .unwrap()
        .join("target")
        .join("onnx_rs_full_spec_roundtrip.onnx");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    save_model(&model, &path).unwrap();
    let loaded = load_model(&path).unwrap();
    assert_proto_equal(&proto, &loaded);
    std::fs::remove_file(path).unwrap();
}
