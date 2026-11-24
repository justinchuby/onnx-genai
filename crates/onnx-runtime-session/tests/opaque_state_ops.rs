use prost::Message;

use onnx_runtime_loader::proto::onnx;
use onnx_runtime_session::{
    AttentionCacheMode, CausalConvStateMode, InferenceSession, LinearAttentionStateMode,
    SessionInput, SessionOutput, StateRuntimeConfig, Tensor,
};

const DOMAIN: &str = "org.onnx.experimental.state";

fn tensor_type(dims: &[Option<i64>], dtype: i32) -> onnx::TypeProto {
    use onnx::tensor_shape_proto::{Dimension, dimension::Value};
    onnx::TypeProto {
        value: Some(onnx::type_proto::Value::TensorType(
            onnx::type_proto::Tensor {
                elem_type: dtype,
                shape: Some(onnx::TensorShapeProto {
                    dim: dims
                        .iter()
                        .enumerate()
                        .map(|(axis, dim)| Dimension {
                            value: Some(match dim {
                                Some(value) => Value::DimValue(*value),
                                None => Value::DimParam(format!("d{axis}")),
                            }),
                            ..Default::default()
                        })
                        .collect(),
                }),
            },
        )),
        ..Default::default()
    }
}

fn unbound_tensor_type(dims: &[Option<i64>], dtype: i32) -> onnx::TypeProto {
    use onnx::tensor_shape_proto::{Dimension, dimension::Value};
    onnx::TypeProto {
        value: Some(onnx::type_proto::Value::TensorType(
            onnx::type_proto::Tensor {
                elem_type: dtype,
                shape: Some(onnx::TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|dim| Dimension {
                            value: dim.map(Value::DimValue),
                            ..Default::default()
                        })
                        .collect(),
                }),
            },
        )),
        ..Default::default()
    }
}

fn opaque_type(name: &str) -> onnx::TypeProto {
    onnx::TypeProto {
        value: Some(onnx::type_proto::Value::OpaqueType(
            onnx::type_proto::Opaque {
                domain: DOMAIN.to_string(),
                name: name.to_string(),
            },
        )),
        ..Default::default()
    }
}

fn info(name: &str, ty: onnx::TypeProto) -> onnx::ValueInfoProto {
    onnx::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(ty),
        ..Default::default()
    }
}

fn int_attr(name: &str, value: i64) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Int as i32,
        i: value,
        ..Default::default()
    }
}

fn string_attr(name: &str, value: &str) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::String as i32,
        s: value.as_bytes().to_vec(),
        ..Default::default()
    }
}

fn model(graph: onnx::GraphProto) -> Vec<u8> {
    onnx::ModelProto {
        ir_version: 10,
        opset_import: vec![
            onnx::OperatorSetIdProto {
                domain: String::new(),
                version: 18,
            },
            onnx::OperatorSetIdProto {
                domain: DOMAIN.to_string(),
                version: 1,
            },
        ],
        graph: Some(graph),
        ..Default::default()
    }
    .encode_to_vec()
}

fn attention_model() -> Vec<u8> {
    let qkv = tensor_type(&[Some(1), None, Some(1), Some(2)], 1);
    let positions = tensor_type(&[Some(1), None], 7);
    model(onnx::GraphProto {
        input: vec![
            info("Q", qkv.clone()),
            info("K", qkv.clone()),
            info("V", qkv.clone()),
            info("cache", opaque_type("KVCache")),
            info("q_positions", positions.clone()),
            info("kv_positions", positions),
        ],
        output: vec![
            info("Y", qkv),
            info("present_cache", opaque_type("KVCache")),
        ],
        node: vec![onnx::NodeProto {
            name: "attention".to_string(),
            op_type: "AttentionWithCache".to_string(),
            domain: DOMAIN.to_string(),
            input: vec![
                "Q".into(),
                "K".into(),
                "V".into(),
                String::new(),
                "cache".into(),
                "q_positions".into(),
                "kv_positions".into(),
            ],
            output: vec!["Y".into(), "present_cache".into()],
            ..Default::default()
        }],
        ..Default::default()
    })
}

fn ragged_attention_model() -> Vec<u8> {
    let qkv = tensor_type(&[Some(2), Some(3), Some(1), Some(1)], 1);
    let positions = tensor_type(&[Some(2), Some(3)], 7);
    model(onnx::GraphProto {
        input: vec![
            info("Q", qkv.clone()),
            info("K", qkv.clone()),
            info("V", qkv.clone()),
            info("q_positions", positions.clone()),
            info("kv_positions", positions),
            info("nonpad", tensor_type(&[Some(2)], 7)),
        ],
        output: vec![
            info("Y", qkv),
            info("present_cache", opaque_type("KVCache")),
        ],
        node: vec![onnx::NodeProto {
            name: "attention".to_string(),
            op_type: "AttentionWithCache".to_string(),
            domain: DOMAIN.to_string(),
            input: vec![
                "Q".into(),
                "K".into(),
                "V".into(),
                String::new(),
                String::new(),
                "q_positions".into(),
                "kv_positions".into(),
                "nonpad".into(),
            ],
            output: vec!["Y".into(), "present_cache".into()],
            ..Default::default()
        }],
        ..Default::default()
    })
}

fn packed_attention_model(compact_visibility: bool) -> Vec<u8> {
    let packed = unbound_tensor_type(&[None, Some(1)], 1);
    let offsets = unbound_tensor_type(&[None], 7);
    let positions = unbound_tensor_type(&[None], 7);
    let mut inputs = vec![
        info("Q", packed.clone()),
        info("K", packed.clone()),
        info("V", packed.clone()),
        info("cache", opaque_type("KVCache")),
        info("q_positions", positions.clone()),
        info("kv_positions", positions),
        info("q_offsets", offsets.clone()),
        info("kv_offsets", offsets.clone()),
    ];
    let mut node_inputs = vec![
        "Q".into(),
        "K".into(),
        "V".into(),
        String::new(),
        "cache".into(),
        "q_positions".into(),
        "kv_positions".into(),
        String::new(),
        "q_offsets".into(),
        "kv_offsets".into(),
    ];
    let mut attributes = vec![int_attr("q_num_heads", 1), int_attr("kv_num_heads", 1)];
    if compact_visibility {
        inputs.extend([
            info("visibility_rows", offsets.clone()),
            info("visibility_blocks", offsets.clone()),
            info("partial_indices", offsets),
            info(
                "partial_mask",
                unbound_tensor_type(&[None, Some(1), Some(1)], 13),
            ),
            info(
                "attn_bias",
                unbound_tensor_type(&[None, Some(1), Some(2)], 1),
            ),
        ]);
        node_inputs.extend([
            "visibility_rows".into(),
            "visibility_blocks".into(),
            "partial_indices".into(),
            "partial_mask".into(),
            "attn_bias".into(),
        ]);
        attributes.extend([
            int_attr("visibility_q_block_size", 1),
            int_attr("visibility_k_block_size", 2),
        ]);
    }
    model(onnx::GraphProto {
        input: inputs,
        output: vec![
            info("Y", packed),
            info("present_cache", opaque_type("KVCache")),
        ],
        node: vec![onnx::NodeProto {
            name: "attention".to_string(),
            op_type: "AttentionWithCache".to_string(),
            domain: DOMAIN.to_string(),
            input: node_inputs,
            output: vec!["Y".into(), "present_cache".into()],
            attribute: attributes,
            ..Default::default()
        }],
        ..Default::default()
    })
}

fn packed_per_head_visibility_model() -> Vec<u8> {
    let query = unbound_tensor_type(&[None, Some(2)], 1);
    let kv = unbound_tensor_type(&[None, Some(1)], 1);
    let offsets = unbound_tensor_type(&[None], 7);
    let positions = unbound_tensor_type(&[None], 7);
    model(onnx::GraphProto {
        input: vec![
            info("Q", query.clone()),
            info("K", kv.clone()),
            info("V", kv),
            info("cache", opaque_type("KVCache")),
            info("q_positions", positions.clone()),
            info("kv_positions", positions),
            info("q_offsets", offsets.clone()),
            info("kv_offsets", offsets.clone()),
            info("visibility_rows", offsets.clone()),
            info("visibility_blocks", offsets),
        ],
        output: vec![
            info("Y", query),
            info("present_cache", opaque_type("KVCache")),
        ],
        node: vec![onnx::NodeProto {
            name: "attention".to_string(),
            op_type: "AttentionWithCache".to_string(),
            domain: DOMAIN.to_string(),
            input: vec![
                "Q".into(),
                "K".into(),
                "V".into(),
                String::new(),
                "cache".into(),
                "q_positions".into(),
                "kv_positions".into(),
                String::new(),
                "q_offsets".into(),
                "kv_offsets".into(),
                "visibility_rows".into(),
                "visibility_blocks".into(),
            ],
            output: vec!["Y".into(), "present_cache".into()],
            attribute: vec![
                int_attr("q_num_heads", 2),
                int_attr("kv_num_heads", 1),
                int_attr("visibility_num_heads", 2),
                int_attr("visibility_q_block_size", 1),
                int_attr("visibility_k_block_size", 2),
            ],
            ..Default::default()
        }],
        ..Default::default()
    })
}

fn u64_tensor(shape: &[usize], values: &[u64]) -> Tensor {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_raw(onnx_runtime_ir::DataType::Uint64, shape.to_vec(), &bytes).unwrap()
}

fn two_attention_slots_model() -> Vec<u8> {
    let qkv = tensor_type(&[Some(1), Some(1), Some(1), Some(1)], 1);
    let positions = tensor_type(&[Some(1), Some(1)], 7);
    let make_node = |name: &str, cache: &str, y: &str, present: &str| onnx::NodeProto {
        name: name.to_string(),
        op_type: "AttentionWithCache".to_string(),
        domain: DOMAIN.to_string(),
        input: vec![
            "Q".into(),
            "K".into(),
            "V".into(),
            String::new(),
            cache.into(),
            "positions".into(),
            "positions".into(),
        ],
        output: vec![y.into(), present.into()],
        ..Default::default()
    };
    model(onnx::GraphProto {
        input: vec![
            info("Q", qkv.clone()),
            info("K", qkv.clone()),
            info("V", qkv.clone()),
            info("cache0", opaque_type("KVCache")),
            info("cache1", opaque_type("KVCache")),
            info("positions", positions),
        ],
        output: vec![
            info("Y0", qkv.clone()),
            info("present0", opaque_type("KVCache")),
            info("Y1", qkv),
            info("present1", opaque_type("KVCache")),
        ],
        node: vec![
            make_node("attention0", "cache0", "Y0", "present0"),
            make_node("attention1", "cache1", "Y1", "present1"),
        ],
        ..Default::default()
    })
}

fn linear_model() -> Vec<u8> {
    let qkv = tensor_type(&[Some(1), None, Some(2)], 1);
    model(onnx::GraphProto {
        input: vec![
            info("query", qkv.clone()),
            info("key", qkv.clone()),
            info("value", qkv.clone()),
            info("state", opaque_type("LinearAttentionState")),
        ],
        output: vec![
            info("output", qkv),
            info("present_state", opaque_type("LinearAttentionState")),
        ],
        node: vec![onnx::NodeProto {
            name: "linear".to_string(),
            op_type: "LinearAttentionWithState".to_string(),
            domain: DOMAIN.to_string(),
            input: vec!["query".into(), "key".into(), "value".into(), "state".into()],
            output: vec!["output".into(), "present_state".into()],
            attribute: vec![
                int_attr("q_num_heads", 1),
                int_attr("kv_num_heads", 1),
                string_attr("update_rule", "linear"),
            ],
            ..Default::default()
        }],
        ..Default::default()
    })
}

fn conv_model() -> Vec<u8> {
    let x = tensor_type(&[Some(1), Some(1), None], 1);
    model(onnx::GraphProto {
        input: vec![
            info("input", x.clone()),
            info("weight", tensor_type(&[Some(1), Some(1), Some(3)], 1)),
            info("state", opaque_type("CausalConvState")),
        ],
        output: vec![
            info("output", x),
            info("present_state", opaque_type("CausalConvState")),
        ],
        node: vec![onnx::NodeProto {
            name: "conv".to_string(),
            op_type: "CausalConvWithState".to_string(),
            domain: DOMAIN.to_string(),
            input: vec![
                "input".into(),
                "weight".into(),
                String::new(),
                "state".into(),
            ],
            output: vec!["output".into(), "present_state".into()],
            ..Default::default()
        }],
        ..Default::default()
    })
}

fn tensor_output(outputs: &[SessionOutput]) -> Vec<f32> {
    outputs[0].as_tensor().unwrap().to_vec_f32()
}

#[test]
fn attention_dense_and_paged_match_chunked_and_preserve_snapshots() {
    let bytes = attention_model();
    let mut reference = InferenceSession::load_bytes(&bytes).unwrap();
    let q = Tensor::from_f32(&[1, 3, 1, 2], &[1., 0., 0., 1., 1., 1.]).unwrap();
    let k = Tensor::from_f32(&[1, 3, 1, 2], &[1., 0., 0., 1., 1., 1.]).unwrap();
    let v = Tensor::from_f32(&[1, 3, 1, 2], &[2., 0., 0., 4., 6., 8.]).unwrap();
    let pos = Tensor::from_i64(&[1, 3], &[0, 1, 2]).unwrap();
    let full = reference
        .run_values(&[
            ("Q", SessionInput::Tensor(&q)),
            ("K", SessionInput::Tensor(&k)),
            ("V", SessionInput::Tensor(&v)),
            ("q_positions", SessionInput::Tensor(&pos)),
            ("kv_positions", SessionInput::Tensor(&pos)),
        ])
        .unwrap();
    let expected = tensor_output(&full);

    for attention in [
        AttentionCacheMode::Dense,
        AttentionCacheMode::Paged { page_size: 2 },
    ] {
        let mut session = InferenceSession::load_bytes(&bytes).unwrap();
        session
            .set_state_runtime_config(StateRuntimeConfig {
                attention,
                ..Default::default()
            })
            .unwrap();
        let q0 = Tensor::from_f32(&[1, 2, 1, 2], &[1., 0., 0., 1.]).unwrap();
        let k0 = Tensor::from_f32(&[1, 2, 1, 2], &[1., 0., 0., 1.]).unwrap();
        let v0 = Tensor::from_f32(&[1, 2, 1, 2], &[2., 0., 0., 4.]).unwrap();
        let p0 = Tensor::from_i64(&[1, 2], &[0, 1]).unwrap();
        let first = session
            .run_values(&[
                ("Q", SessionInput::Tensor(&q0)),
                ("K", SessionInput::Tensor(&k0)),
                ("V", SessionInput::Tensor(&v0)),
                ("q_positions", SessionInput::Tensor(&p0)),
                ("kv_positions", SessionInput::Tensor(&p0)),
            ])
            .unwrap();
        let snapshot = first[1].as_opaque().unwrap().clone();
        let q1 = Tensor::from_f32(&[1, 1, 1, 2], &[1., 1.]).unwrap();
        let k1 = Tensor::from_f32(&[1, 1, 1, 2], &[1., 1.]).unwrap();
        let v1 = Tensor::from_f32(&[1, 1, 1, 2], &[6., 8.]).unwrap();
        let p1 = Tensor::from_i64(&[1, 1], &[2]).unwrap();
        let second = session
            .run_values(&[
                ("Q", SessionInput::Tensor(&q1)),
                ("K", SessionInput::Tensor(&k1)),
                ("V", SessionInput::Tensor(&v1)),
                ("cache", SessionInput::Opaque(&snapshot)),
                ("q_positions", SessionInput::Tensor(&p1)),
                ("kv_positions", SessionInput::Tensor(&p1)),
            ])
            .unwrap();
        let mut chunked = tensor_output(&first);
        chunked.extend(tensor_output(&second));
        assert_eq!(chunked, expected);

        let alternate_v = Tensor::from_f32(&[1, 1, 1, 2], &[60., 80.]).unwrap();
        let alternate = session
            .run_values(&[
                ("Q", SessionInput::Tensor(&q1)),
                ("K", SessionInput::Tensor(&k1)),
                ("V", SessionInput::Tensor(&alternate_v)),
                ("cache", SessionInput::Opaque(&snapshot)),
                ("q_positions", SessionInput::Tensor(&p1)),
                ("kv_positions", SessionInput::Tensor(&p1)),
            ])
            .unwrap();
        assert_ne!(tensor_output(&alternate), tensor_output(&second));
    }
}

#[test]
fn packed_attention_matches_padded_oracle_across_modes_and_zero_length_rows() {
    let padded_bytes = ragged_attention_model();
    let mut padded = InferenceSession::load_bytes(&padded_bytes).unwrap();
    let padded_q = Tensor::from_f32(&[2, 3, 1, 1], &[1.; 6]).unwrap();
    let padded_k = Tensor::from_f32(&[2, 3, 1, 1], &[1.; 6]).unwrap();
    let padded_v = Tensor::from_f32(&[2, 3, 1, 1], &[1., 2., 3., 10., 0., 0.]).unwrap();
    let padded_positions = Tensor::from_i64(&[2, 3], &[0, 1, 2, 0, 1, 2]).unwrap();
    let nonpad = Tensor::from_i64(&[2], &[3, 1]).unwrap();
    let oracle = padded
        .run_values(&[
            ("Q", SessionInput::Tensor(&padded_q)),
            ("K", SessionInput::Tensor(&padded_k)),
            ("V", SessionInput::Tensor(&padded_v)),
            ("q_positions", SessionInput::Tensor(&padded_positions)),
            ("kv_positions", SessionInput::Tensor(&padded_positions)),
            ("nonpad", SessionInput::Tensor(&nonpad)),
        ])
        .unwrap();
    let oracle = tensor_output(&oracle);

    let bytes = packed_attention_model(false);
    for attention in [
        AttentionCacheMode::Dense,
        AttentionCacheMode::Paged { page_size: 2 },
    ] {
        let mut session = InferenceSession::load_bytes(&bytes).unwrap();
        session
            .set_state_runtime_config(StateRuntimeConfig {
                attention,
                ..Default::default()
            })
            .unwrap();
        let q0 = Tensor::from_f32(&[3, 1], &[1.; 3]).unwrap();
        let k0 = Tensor::from_f32(&[3, 1], &[1.; 3]).unwrap();
        let v0 = Tensor::from_f32(&[3, 1], &[1., 2., 10.]).unwrap();
        let qpos0 = Tensor::from_i64(&[3], &[0, 1, 0]).unwrap();
        let kvpos0 = Tensor::from_i64(&[3], &[0, 1, 0]).unwrap();
        let offsets0 = Tensor::from_i64(&[3], &[0, 2, 3]).unwrap();
        let first = session
            .run_values(&[
                ("Q", SessionInput::Tensor(&q0)),
                ("K", SessionInput::Tensor(&k0)),
                ("V", SessionInput::Tensor(&v0)),
                ("q_positions", SessionInput::Tensor(&qpos0)),
                ("kv_positions", SessionInput::Tensor(&kvpos0)),
                ("q_offsets", SessionInput::Tensor(&offsets0)),
                ("kv_offsets", SessionInput::Tensor(&offsets0)),
            ])
            .unwrap();
        assert_eq!(tensor_output(&first), vec![oracle[0], oracle[1], oracle[3]]);
        let snapshot = first[1].as_opaque().unwrap().clone();

        let q1 = Tensor::from_f32(&[1, 1], &[1.]).unwrap();
        let k1 = Tensor::from_f32(&[1, 1], &[1.]).unwrap();
        let v1 = Tensor::from_f32(&[1, 1], &[3.]).unwrap();
        let pos1 = Tensor::from_i64(&[1], &[2]).unwrap();
        let q_offsets1 = Tensor::from_i64(&[3], &[0, 1, 1]).unwrap();
        let kv_offsets1 = Tensor::from_i64(&[3], &[0, 1, 1]).unwrap();
        let second = session
            .run_values(&[
                ("Q", SessionInput::Tensor(&q1)),
                ("K", SessionInput::Tensor(&k1)),
                ("V", SessionInput::Tensor(&v1)),
                ("cache", SessionInput::Opaque(&snapshot)),
                ("q_positions", SessionInput::Tensor(&pos1)),
                ("kv_positions", SessionInput::Tensor(&pos1)),
                ("q_offsets", SessionInput::Tensor(&q_offsets1)),
                ("kv_offsets", SessionInput::Tensor(&kv_offsets1)),
            ])
            .unwrap();
        assert_eq!(tensor_output(&second), vec![oracle[2]]);
    }
}

#[test]
fn compact_visibility_skips_blocks_supports_partial_bias_and_zero_rows() {
    let bytes = packed_attention_model(true);
    for attention in [
        AttentionCacheMode::Dense,
        AttentionCacheMode::Paged { page_size: 2 },
    ] {
        let mut session = InferenceSession::load_bytes(&bytes).unwrap();
        session
            .set_state_runtime_config(StateRuntimeConfig {
                attention,
                ..Default::default()
            })
            .unwrap();
        let empty = Tensor::from_f32(&[0, 1], &[]).unwrap();
        let keys = Tensor::from_f32(&[5, 1], &[0.; 5]).unwrap();
        let values = Tensor::from_f32(&[5, 1], &[1., 2., 3., 4., 5.]).unwrap();
        let empty_positions = Tensor::from_i64(&[0], &[]).unwrap();
        let kv_positions = Tensor::from_i64(&[5], &[0, 1, 2, 3, 4]).unwrap();
        let q_offsets = Tensor::from_i64(&[2], &[0, 0]).unwrap();
        let kv_offsets = Tensor::from_i64(&[2], &[0, 5]).unwrap();
        let visibility_rows = Tensor::from_i64(&[1], &[0]).unwrap();
        let no_blocks = Tensor::from_i64(&[0], &[]).unwrap();
        let no_mask = u64_tensor(&[0, 1, 1], &[]);
        let no_bias = Tensor::from_f32(&[0, 1, 2], &[]).unwrap();
        let seeded = session
            .run_values(&[
                ("Q", SessionInput::Tensor(&empty)),
                ("K", SessionInput::Tensor(&keys)),
                ("V", SessionInput::Tensor(&values)),
                ("q_positions", SessionInput::Tensor(&empty_positions)),
                ("kv_positions", SessionInput::Tensor(&kv_positions)),
                ("q_offsets", SessionInput::Tensor(&q_offsets)),
                ("kv_offsets", SessionInput::Tensor(&kv_offsets)),
                ("visibility_rows", SessionInput::Tensor(&visibility_rows)),
                ("visibility_blocks", SessionInput::Tensor(&no_blocks)),
                ("partial_indices", SessionInput::Tensor(&no_blocks)),
                ("partial_mask", SessionInput::Tensor(&no_mask)),
                ("attn_bias", SessionInput::Tensor(&no_bias)),
            ])
            .unwrap();
        assert!(tensor_output(&seeded).is_empty());
        let cache = seeded[1].as_opaque().unwrap().clone();

        let queries = Tensor::from_f32(&[2, 1], &[0., 0.]).unwrap();
        let query_positions = Tensor::from_i64(&[2], &[4, 4]).unwrap();
        let q_offsets = Tensor::from_i64(&[2], &[0, 2]).unwrap();
        let kv_offsets = Tensor::from_i64(&[2], &[0, 0]).unwrap();
        let rows = Tensor::from_i64(&[3], &[0, 2, 3]).unwrap();
        let blocks = Tensor::from_i64(&[3], &[0, 2, 0]).unwrap();
        let partial_indices = Tensor::from_i64(&[2], &[0, 2]).unwrap();
        let partial_mask = u64_tensor(&[2, 1, 1], &[0b10, 0]);
        let bias = Tensor::from_f32(&[3, 1, 2], &[0., 0., 3.0f32.ln(), 0., 0., 0.]).unwrap();
        let output = session
            .run_values(&[
                ("Q", SessionInput::Tensor(&queries)),
                ("K", SessionInput::Tensor(&empty)),
                ("V", SessionInput::Tensor(&empty)),
                ("cache", SessionInput::Opaque(&cache)),
                ("q_positions", SessionInput::Tensor(&query_positions)),
                ("kv_positions", SessionInput::Tensor(&empty_positions)),
                ("q_offsets", SessionInput::Tensor(&q_offsets)),
                ("kv_offsets", SessionInput::Tensor(&kv_offsets)),
                ("visibility_rows", SessionInput::Tensor(&rows)),
                ("visibility_blocks", SessionInput::Tensor(&blocks)),
                ("partial_indices", SessionInput::Tensor(&partial_indices)),
                ("partial_mask", SessionInput::Tensor(&partial_mask)),
                ("attn_bias", SessionInput::Tensor(&bias)),
            ])
            .unwrap();
        let actual = tensor_output(&output);
        assert!((actual[0] - 4.25).abs() < 1e-6, "{actual:?}");
        assert_eq!(actual[1], 0.0);
        let stats = session.attention_traversal_stats();
        assert_eq!(stats.logical_key_candidates, 10);
        assert_eq!(stats.evaluated_key_candidates, 2);
        assert_eq!(stats.full_block_visits, 1);
        assert_eq!(stats.partial_block_visits, 2);
    }
}

#[test]
fn compact_visibility_supports_exact_per_query_head_rows() {
    let bytes = packed_per_head_visibility_model();
    let mut session = InferenceSession::load_bytes(&bytes).unwrap();
    let empty_query = Tensor::from_f32(&[0, 2], &[]).unwrap();
    let empty_kv = Tensor::from_f32(&[0, 1], &[]).unwrap();
    let keys = Tensor::from_f32(&[3, 1], &[0.; 3]).unwrap();
    let values = Tensor::from_f32(&[3, 1], &[1., 2., 3.]).unwrap();
    let no_positions = Tensor::from_i64(&[0], &[]).unwrap();
    let key_positions = Tensor::from_i64(&[3], &[0, 1, 2]).unwrap();
    let q_offsets = Tensor::from_i64(&[2], &[0, 0]).unwrap();
    let kv_offsets = Tensor::from_i64(&[2], &[0, 3]).unwrap();
    let empty_rows = Tensor::from_i64(&[1], &[0]).unwrap();
    let no_blocks = Tensor::from_i64(&[0], &[]).unwrap();
    let seeded = session
        .run_values(&[
            ("Q", SessionInput::Tensor(&empty_query)),
            ("K", SessionInput::Tensor(&keys)),
            ("V", SessionInput::Tensor(&values)),
            ("q_positions", SessionInput::Tensor(&no_positions)),
            ("kv_positions", SessionInput::Tensor(&key_positions)),
            ("q_offsets", SessionInput::Tensor(&q_offsets)),
            ("kv_offsets", SessionInput::Tensor(&kv_offsets)),
            ("visibility_rows", SessionInput::Tensor(&empty_rows)),
            ("visibility_blocks", SessionInput::Tensor(&no_blocks)),
        ])
        .unwrap();
    let cache = seeded[1].as_opaque().unwrap().clone();

    let query = Tensor::from_f32(&[1, 2], &[0., 0.]).unwrap();
    let query_position = Tensor::from_i64(&[1], &[2]).unwrap();
    let q_offsets = Tensor::from_i64(&[2], &[0, 1]).unwrap();
    let kv_offsets = Tensor::from_i64(&[2], &[0, 0]).unwrap();
    let rows = Tensor::from_i64(&[3], &[0, 1, 2]).unwrap();
    let blocks = Tensor::from_i64(&[2], &[0, 1]).unwrap();
    let output = session
        .run_values(&[
            ("Q", SessionInput::Tensor(&query)),
            ("K", SessionInput::Tensor(&empty_kv)),
            ("V", SessionInput::Tensor(&empty_kv)),
            ("cache", SessionInput::Opaque(&cache)),
            ("q_positions", SessionInput::Tensor(&query_position)),
            ("kv_positions", SessionInput::Tensor(&no_positions)),
            ("q_offsets", SessionInput::Tensor(&q_offsets)),
            ("kv_offsets", SessionInput::Tensor(&kv_offsets)),
            ("visibility_rows", SessionInput::Tensor(&rows)),
            ("visibility_blocks", SessionInput::Tensor(&blocks)),
        ])
        .unwrap();
    assert_eq!(tensor_output(&output), vec![1.5, 3.0]);
}

#[test]
fn packed_attention_rejects_malformed_offsets_and_visibility() {
    let bytes = packed_attention_model(false);
    let mut session = InferenceSession::load_bytes(&bytes).unwrap();
    let one = Tensor::from_f32(&[1, 1], &[1.]).unwrap();
    let position = Tensor::from_i64(&[1], &[0]).unwrap();
    let bad_offsets = Tensor::from_i64(&[2], &[0, 2]).unwrap();
    let error = session
        .run_values(&[
            ("Q", SessionInput::Tensor(&one)),
            ("K", SessionInput::Tensor(&one)),
            ("V", SessionInput::Tensor(&one)),
            ("q_positions", SessionInput::Tensor(&position)),
            ("kv_positions", SessionInput::Tensor(&position)),
            ("q_offsets", SessionInput::Tensor(&bad_offsets)),
            ("kv_offsets", SessionInput::Tensor(&bad_offsets)),
        ])
        .unwrap_err()
        .to_string();
    assert!(error.contains("ends at 2"), "{error}");
    assert!(error.contains("has 1 tokens"), "{error}");

    let bytes = packed_attention_model(true);
    let mut session = InferenceSession::load_bytes(&bytes).unwrap();
    let four = Tensor::from_f32(&[4, 1], &[1.; 4]).unwrap();
    let offsets = Tensor::from_i64(&[2], &[0, 4]).unwrap();
    let positions = Tensor::from_i64(&[4], &[0, 1, 2, 3]).unwrap();
    let rows = Tensor::from_i64(&[5], &[0, 2, 2, 2, 2]).unwrap();
    let unsorted = Tensor::from_i64(&[2], &[1, 0]).unwrap();
    let no_partial = Tensor::from_i64(&[0], &[]).unwrap();
    let no_mask = u64_tensor(&[0, 1, 1], &[]);
    let bias = Tensor::from_f32(&[2, 1, 2], &[0.; 4]).unwrap();
    let error = session
        .run_values(&[
            ("Q", SessionInput::Tensor(&four)),
            ("K", SessionInput::Tensor(&four)),
            ("V", SessionInput::Tensor(&four)),
            ("q_positions", SessionInput::Tensor(&positions)),
            ("kv_positions", SessionInput::Tensor(&positions)),
            ("q_offsets", SessionInput::Tensor(&offsets)),
            ("kv_offsets", SessionInput::Tensor(&offsets)),
            ("visibility_rows", SessionInput::Tensor(&rows)),
            ("visibility_blocks", SessionInput::Tensor(&unsorted)),
            ("partial_indices", SessionInput::Tensor(&no_partial)),
            ("partial_mask", SessionInput::Tensor(&no_mask)),
            ("attn_bias", SessionInput::Tensor(&bias)),
        ])
        .unwrap_err()
        .to_string();
    assert!(error.contains("sorted and unique"), "{error}");
}

#[test]
fn opaque_state_rejects_foreign_session_owner() {
    let bytes = attention_model();
    let mut first_session = InferenceSession::load_bytes(&bytes).unwrap();
    let mut second_session = InferenceSession::load_bytes(&bytes).unwrap();
    let q = Tensor::from_f32(&[1, 1, 1, 2], &[1., 0.]).unwrap();
    let v = Tensor::from_f32(&[1, 1, 1, 2], &[2., 3.]).unwrap();
    let pos = Tensor::from_i64(&[1, 1], &[0]).unwrap();
    let first = first_session
        .run_values(&[
            ("Q", SessionInput::Tensor(&q)),
            ("K", SessionInput::Tensor(&q)),
            ("V", SessionInput::Tensor(&v)),
            ("q_positions", SessionInput::Tensor(&pos)),
            ("kv_positions", SessionInput::Tensor(&pos)),
        ])
        .unwrap();
    let state = first[1].as_opaque().unwrap();
    let error = second_session
        .run_values(&[
            ("Q", SessionInput::Tensor(&q)),
            ("K", SessionInput::Tensor(&q)),
            ("V", SessionInput::Tensor(&v)),
            ("cache", SessionInput::Opaque(state)),
            ("q_positions", SessionInput::Tensor(&pos)),
            ("kv_positions", SessionInput::Tensor(&pos)),
        ])
        .unwrap_err()
        .to_string();
    assert!(error.contains("belongs to executor"), "{error}");
    assert!(error.contains("exact session"), "{error}");
}

#[test]
fn paged_attention_honors_ragged_kv_lengths_across_page_boundary() {
    let bytes = ragged_attention_model();
    let mut session = InferenceSession::load_bytes(&bytes).unwrap();
    session
        .set_state_runtime_config(StateRuntimeConfig {
            attention: AttentionCacheMode::Paged { page_size: 2 },
            ..Default::default()
        })
        .unwrap();
    let q = Tensor::from_f32(&[2, 3, 1, 1], &[1.; 6]).unwrap();
    let k = Tensor::from_f32(&[2, 3, 1, 1], &[1.; 6]).unwrap();
    let v = Tensor::from_f32(&[2, 3, 1, 1], &[1., 2., 3., 10., 20., 30.]).unwrap();
    let positions = Tensor::from_i64(&[2, 3], &[0, 1, 2, 0, 1, 2]).unwrap();
    let nonpad = Tensor::from_i64(&[2], &[3, 1]).unwrap();
    let output = session
        .run_values(&[
            ("Q", SessionInput::Tensor(&q)),
            ("K", SessionInput::Tensor(&k)),
            ("V", SessionInput::Tensor(&v)),
            ("q_positions", SessionInput::Tensor(&positions)),
            ("kv_positions", SessionInput::Tensor(&positions)),
            ("nonpad", SessionInput::Tensor(&nonpad)),
        ])
        .unwrap();
    assert_eq!(tensor_output(&output), vec![1., 1.5, 2., 10., 10., 10.]);
}

#[test]
fn opaque_state_rejects_wrong_operator_slot() {
    let bytes = two_attention_slots_model();
    let mut session = InferenceSession::load_bytes(&bytes).unwrap();
    let q = Tensor::from_f32(&[1, 1, 1, 1], &[1.]).unwrap();
    let v = Tensor::from_f32(&[1, 1, 1, 1], &[2.]).unwrap();
    let positions = Tensor::from_i64(&[1, 1], &[0]).unwrap();
    let first = session
        .run_values(&[
            ("Q", SessionInput::Tensor(&q)),
            ("K", SessionInput::Tensor(&q)),
            ("V", SessionInput::Tensor(&v)),
            ("positions", SessionInput::Tensor(&positions)),
        ])
        .unwrap();
    let state0 = first[1].as_opaque().unwrap();
    let error = session
        .run_values(&[
            ("Q", SessionInput::Tensor(&q)),
            ("K", SessionInput::Tensor(&q)),
            ("V", SessionInput::Tensor(&v)),
            ("cache1", SessionInput::Opaque(state0)),
            ("positions", SessionInput::Tensor(&positions)),
        ])
        .unwrap_err()
        .to_string();
    assert!(error.contains("state slot node#"), "{error}");
    assert!(error.contains("same operator"), "{error}");
}

#[test]
fn opaque_state_rejects_wrong_nominal_kind_before_execution() {
    let linear_bytes = linear_model();
    let mut linear = InferenceSession::load_bytes(&linear_bytes).unwrap();
    let x = Tensor::from_f32(&[1, 1, 2], &[1., 0.]).unwrap();
    let linear_outputs = linear
        .run_values(&[
            ("query", SessionInput::Tensor(&x)),
            ("key", SessionInput::Tensor(&x)),
            ("value", SessionInput::Tensor(&x)),
        ])
        .unwrap();
    let linear_state = linear_outputs[1].as_opaque().unwrap();

    let attention_bytes = attention_model();
    let mut attention = InferenceSession::load_bytes(&attention_bytes).unwrap();
    let q = Tensor::from_f32(&[1, 1, 1, 2], &[1., 0.]).unwrap();
    let pos = Tensor::from_i64(&[1, 1], &[0]).unwrap();
    let error = attention
        .run_values(&[
            ("Q", SessionInput::Tensor(&q)),
            ("K", SessionInput::Tensor(&q)),
            ("V", SessionInput::Tensor(&q)),
            ("cache", SessionInput::Opaque(linear_state)),
            ("q_positions", SessionInput::Tensor(&pos)),
            ("kv_positions", SessionInput::Tensor(&pos)),
        ])
        .unwrap_err()
        .to_string();
    assert!(error.contains("expects opaque KVCache"), "{error}");
    assert!(error.contains("LinearAttentionState"), "{error}");
    let recovered = attention
        .run_values(&[
            ("Q", SessionInput::Tensor(&q)),
            ("K", SessionInput::Tensor(&q)),
            ("V", SessionInput::Tensor(&q)),
            ("q_positions", SessionInput::Tensor(&pos)),
            ("kv_positions", SessionInput::Tensor(&pos)),
        ])
        .unwrap();
    assert_eq!(tensor_output(&recovered), vec![1., 0.]);
}

#[test]
fn linear_attention_modes_match_independent_recurrence() {
    let bytes = linear_model();
    let query = [1., 0., 0., 1., 1., 1.];
    let key = [1., 0., 0., 1., 1., 1.];
    let value = [2., 3., 4., 5., 6., 7.];
    let mut state = [0.0f32; 4];
    let mut expected = Vec::new();
    for t in 0..3 {
        for i in 0..2 {
            for j in 0..2 {
                state[i * 2 + j] += key[t * 2 + i] * value[t * 2 + j];
            }
        }
        for j in 0..2 {
            expected
                .push((query[t * 2] * state[j] + query[t * 2 + 1] * state[2 + j]) / 2.0f32.sqrt());
        }
    }
    for mode in [
        LinearAttentionStateMode::Contiguous,
        LinearAttentionStateMode::Blocked,
    ] {
        let mut session = InferenceSession::load_bytes(&bytes).unwrap();
        session
            .set_state_runtime_config(StateRuntimeConfig {
                linear_attention: mode,
                ..Default::default()
            })
            .unwrap();
        let q0 = Tensor::from_f32(&[1, 2, 2], &query[..4]).unwrap();
        let k0 = Tensor::from_f32(&[1, 2, 2], &key[..4]).unwrap();
        let v0 = Tensor::from_f32(&[1, 2, 2], &value[..4]).unwrap();
        let first = session
            .run_values(&[
                ("query", SessionInput::Tensor(&q0)),
                ("key", SessionInput::Tensor(&k0)),
                ("value", SessionInput::Tensor(&v0)),
            ])
            .unwrap();
        let q1 = Tensor::from_f32(&[1, 1, 2], &query[4..]).unwrap();
        let k1 = Tensor::from_f32(&[1, 1, 2], &key[4..]).unwrap();
        let v1 = Tensor::from_f32(&[1, 1, 2], &value[4..]).unwrap();
        let state = first[1].as_opaque().unwrap();
        let second = session
            .run_values(&[
                ("query", SessionInput::Tensor(&q1)),
                ("key", SessionInput::Tensor(&k1)),
                ("value", SessionInput::Tensor(&v1)),
                ("state", SessionInput::Opaque(state)),
            ])
            .unwrap();
        let mut actual = tensor_output(&first);
        actual.extend(tensor_output(&second));
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }
}

#[test]
fn causal_conv_modes_match_direct_convolution_across_chunks() {
    let bytes = conv_model();
    let input = [1., 2., 3., 4., 5.];
    let weight = [0.5, 1.0, 2.0];
    let mut padded = vec![0.0, 0.0];
    padded.extend(input);
    let expected: Vec<f32> = (0..input.len())
        .map(|t| (0..3).map(|k| weight[k] * padded[t + k]).sum())
        .collect();
    for mode in [CausalConvStateMode::Contiguous, CausalConvStateMode::Ring] {
        let mut session = InferenceSession::load_bytes(&bytes).unwrap();
        session
            .set_state_runtime_config(StateRuntimeConfig {
                causal_conv: mode,
                ..Default::default()
            })
            .unwrap();
        let w = Tensor::from_f32(&[1, 1, 3], &weight).unwrap();
        let x0 = Tensor::from_f32(&[1, 1, 2], &input[..2]).unwrap();
        let first = session
            .run_values(&[
                ("input", SessionInput::Tensor(&x0)),
                ("weight", SessionInput::Tensor(&w)),
            ])
            .unwrap();
        let x1 = Tensor::from_f32(&[1, 1, 3], &input[2..]).unwrap();
        let state = first[1].as_opaque().unwrap();
        let second = session
            .run_values(&[
                ("input", SessionInput::Tensor(&x1)),
                ("weight", SessionInput::Tensor(&w)),
                ("state", SessionInput::Opaque(state)),
            ])
            .unwrap();
        let mut actual = tensor_output(&first);
        actual.extend(tensor_output(&second));
        assert_eq!(actual, expected);
    }
}
