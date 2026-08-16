use super::*;
use serde_json::{Value, json};

fn processor_config(smart_resize: Value, temporal_patch_size: Value) -> ProcessorConfig {
    serde_json::from_value(json!({
        "processor": {
            "transforms": [
                { "operation": { "type": "DecodeImage" } },
                {
                    "operation": {
                        "type": "Resize",
                        "attrs": {
                            "width": 32,
                            "height": 32,
                            "smart_resize": smart_resize
                        }
                    }
                },
                {
                    "operation": {
                        "type": "Rescale",
                        "attrs": { "rescale_factor": 0.00392156862745098_f64 }
                    }
                },
                {
                    "operation": {
                        "type": "Normalize",
                        "attrs": {
                            "mean": [0.5, 0.5, 0.5],
                            "std": [0.5, 0.5, 0.5]
                        }
                    }
                },
                {
                    "operation": {
                        "type": "PatchImage",
                        "attrs": {
                            "patch_size": 16,
                            "temporal_patch_size": temporal_patch_size,
                            "merge_size": 2
                        }
                    }
                }
            ]
        }
    }))
    .expect("processor fixture")
}

fn processor_vision() -> GenAiVision {
    serde_json::from_value(json!({
        "patch_size": 16,
        "spatial_merge_size": 2,
        "inputs": {
            "pixel_values": "pixel_values",
            "image_grid_thw": "image_grid_thw"
        }
    }))
    .expect("vision fixture")
}

fn processor_tensor(name: &str, dtype: &str) -> GraphTensorInfo {
    GraphTensorInfo {
        name: name.to_owned(),
        dtype: dtype.to_owned(),
        dimensions: vec![None, None],
    }
}

#[test]
fn processor_requires_numeric_smart_resize_flag() {
    let mut missing = processor_config(json!(0), json!(1));
    missing
        .processor
        .transforms
        .iter_mut()
        .find(|transform| transform.operation.operation_type == "Resize")
        .expect("resize transform")
        .operation
        .attrs
        .remove("smart_resize");
    let missing_error = processor_program_json(
        &missing,
        &processor_vision(),
        &processor_tensor("pixel_values", "float32"),
        &processor_tensor("image_grid_thw", "int64"),
    )
    .expect_err("missing smart_resize must fail")
    .to_string();
    assert!(missing_error.contains("smart_resize"));
    assert!(missing_error.contains("numeric flag 0 or 1"));

    for value in [Value::Null, json!("false"), json!(2)] {
        let error = processor_program_json(
            &processor_config(value, json!(1)),
            &processor_vision(),
            &processor_tensor("pixel_values", "float32"),
            &processor_tensor("image_grid_thw", "int64"),
        )
        .expect_err("invalid smart_resize must fail")
        .to_string();
        assert!(error.contains("smart_resize"));
        assert!(error.contains("numeric flag 0 or 1"));
    }
}

#[test]
fn processor_signals_unrepresentable_smart_resize_for_text_only_fallback() {
    let error = processor_program_json(
        &processor_config(json!(1), json!(1)),
        &processor_vision(),
        &processor_tensor("pixel_values", "float32"),
        &processor_tensor("image_grid_thw", "int64"),
    )
    .expect_err("smart resize has no lossless runtime encoding")
    .to_string();
    // Signalled as the distinct unrepresentable-preprocessing decline so the
    // loader can fall back to a text-only decode pipeline.
    assert!(error.contains("not representable by the runtime"));
    assert!(error.contains("stretch/crop/pad"));
    assert!(error.contains("text-only decode"));
}

#[test]
fn processor_emits_executable_temporal_patch_size() {
    let program = processor_program_json(
        &processor_config(json!(0), json!(2)),
        &processor_vision(),
        &processor_tensor("pixel_values", "float32"),
        &processor_tensor("image_grid_thw", "int64"),
    )
    .expect("temporal patching is executable");
    let patchify = program["image"]["transforms"]
        .as_array()
        .expect("transforms")
        .iter()
        .find(|transform| transform["op"] == "patchify")
        .expect("patchify");
    assert_eq!(patchify["temporal_patch_size"], 2);
    assert_eq!(patchify["merge_size"], 2);
    assert_eq!(patchify["channel_order"], "channels_first");
}

fn hybrid_graph_tensor(name: &str, dtype: &str, dims: &[Option<usize>]) -> GraphTensorInfo {
    GraphTensorInfo {
        name: name.to_string(),
        dtype: dtype.to_string(),
        dimensions: dims.to_vec(),
    }
}

/// A hybrid SSM/attention decoder (qwen3.5-shaped): four layers where the
/// odd layers are dense full-attention (`key`/`value`) and the even layers
/// are linear-attention recurrent (`conv_state`/`recurrent_state`). The
/// genai_config only carries the uniform `%d` KV pattern and a layer count,
/// so deriving metadata from the graph is the only way to avoid declaring the
/// six non-existent dense-KV ports for the recurrent layers.
fn hybrid_config() -> GenAiConfig {
    serde_json::from_str(
        r#"{
            "model": {
                "type": "qwen3_5_text",
                "context_length": 4096,
                "decoder": {
                    "head_size": 256,
                    "hidden_size": 2048,
                    "num_attention_heads": 8,
                    "num_hidden_layers": 4,
                    "num_key_value_heads": 2,
                    "inputs": {
                        "input_ids": "input_ids",
                        "attention_mask": "attention_mask",
                        "position_ids": "position_ids",
                        "past_key_names": "past_key_values.%d.key",
                        "past_value_names": "past_key_values.%d.value"
                    },
                    "outputs": {
                        "logits": "logits",
                        "present_key_names": "present.%d.key",
                        "present_value_names": "present.%d.value"
                    }
                }
            },
            "search": { "past_present_share_buffer": true, "max_length": 4096 }
        }"#,
    )
    .expect("valid hybrid genai_config")
}

fn hybrid_decoder_graph() -> ModelGraphInfo {
    let sym = |_n: &str| None;
    let dense = [Some(1), Some(2), sym("seq"), Some(256)];
    let conv = [Some(1), Some(6144), Some(3)];
    let recur = [Some(1), Some(16), Some(128), Some(128)];
    let mut inputs = vec![
        hybrid_graph_tensor("input_ids", "int64", &[Some(1), sym("seq")]),
        hybrid_graph_tensor("attention_mask", "int64", &[Some(1), sym("seq")]),
        hybrid_graph_tensor("position_ids", "int64", &[Some(1), sym("seq")]),
    ];
    let mut outputs = vec![hybrid_graph_tensor(
        "logits",
        "float32",
        &[Some(1), sym("seq"), Some(248320)],
    )];
    for layer in 0..4 {
        if layer % 2 == 1 {
            inputs.push(hybrid_graph_tensor(
                &format!("past_key_values.{layer}.key"),
                "float32",
                &dense,
            ));
            inputs.push(hybrid_graph_tensor(
                &format!("past_key_values.{layer}.value"),
                "float32",
                &dense,
            ));
            outputs.push(hybrid_graph_tensor(
                &format!("present.{layer}.key"),
                "float32",
                &dense,
            ));
            outputs.push(hybrid_graph_tensor(
                &format!("present.{layer}.value"),
                "float32",
                &dense,
            ));
        } else {
            inputs.push(hybrid_graph_tensor(
                &format!("past_key_values.{layer}.conv_state"),
                "float32",
                &conv,
            ));
            inputs.push(hybrid_graph_tensor(
                &format!("past_key_values.{layer}.recurrent_state"),
                "float32",
                &recur,
            ));
            outputs.push(hybrid_graph_tensor(
                &format!("present.{layer}.conv_state"),
                "float32",
                &conv,
            ));
            outputs.push(hybrid_graph_tensor(
                &format!("present.{layer}.recurrent_state"),
                "float32",
                &recur,
            ));
        }
    }
    ModelGraphInfo { inputs, outputs }
}

#[test]
fn hybrid_decoder_derives_sparse_kv_and_state_pairs() {
    let cfg = hybrid_config();
    let graph = hybrid_decoder_graph();
    let md = cfg
        .to_inference_metadata_with_graph(Some("float32"), &graph)
        .expect("hybrid metadata");
    let io = md
        .model
        .as_ref()
        .and_then(|m| m.io.as_ref())
        .expect("decoder io");

    // Only the two dense full-attention layers (1, 3) expose key/value; the
    // recurrent layers must NOT appear in the KV lists.
    assert_eq!(
        io.kv_inputs.as_deref(),
        Some(
            [
                "past_key_values.1.key",
                "past_key_values.1.value",
                "past_key_values.3.key",
                "past_key_values.3.value",
            ]
            .map(String::from)
            .as_slice()
        )
    );
    assert_eq!(
        io.kv_outputs.as_deref(),
        Some(
            [
                "present.1.key",
                "present.1.value",
                "present.3.key",
                "present.3.value",
            ]
            .map(String::from)
            .as_slice()
        )
    );

    // The four recurrent ports (conv_state + recurrent_state for layers 0, 2)
    // are declared as fixed loop-carried state pairs, replaced each step.
    let pairs = io.state_pairs.as_ref().expect("state pairs");
    let mut got: Vec<(String, String)> = pairs
        .iter()
        .map(|pair| (pair.input.clone(), pair.output.clone()))
        .collect();
    got.sort();
    let mut want = vec![
        (
            "past_key_values.0.conv_state".to_string(),
            "present.0.conv_state".to_string(),
        ),
        (
            "past_key_values.0.recurrent_state".to_string(),
            "present.0.recurrent_state".to_string(),
        ),
        (
            "past_key_values.2.conv_state".to_string(),
            "present.2.conv_state".to_string(),
        ),
        (
            "past_key_values.2.recurrent_state".to_string(),
            "present.2.recurrent_state".to_string(),
        ),
    ];
    want.sort();
    assert_eq!(got, want);
    for pair in pairs {
        assert_eq!(pair.init.as_deref(), Some("zeros"));
        assert_eq!(pair.update.as_deref(), Some("replace"));
    }
}

/// A qwen3.6-27b-shaped hybrid graph: 64 decoder layers where every 4th layer
/// (indices 3, 7, ... 63) is dense GQA full-attention (`key`/`value`) and the
/// remaining 48 layers are linear-attention recurrent
/// (`conv_state`/`recurrent_state`), using the conventional onnxruntime-genai
/// `past_key_values.%d.*` / `present.%d.*` port names the config-free
/// derivation keys off.
fn qwen27b_hybrid_graph() -> ModelGraphInfo {
    let dense = [Some(1), Some(8), None, Some(128)];
    let conv = [Some(1), Some(10240), Some(3)];
    let recur = [Some(1), Some(16), Some(128), Some(128)];
    let mut inputs = vec![
        hybrid_graph_tensor("input_ids", "int64", &[Some(1), None]),
        hybrid_graph_tensor("attention_mask", "int64", &[Some(1), None]),
        hybrid_graph_tensor("position_ids", "int64", &[Some(1), None]),
    ];
    let mut outputs = vec![hybrid_graph_tensor(
        "logits",
        "float32",
        &[Some(1), None, Some(151936)],
    )];
    for layer in 0..64 {
        if layer % 4 == 3 {
            inputs.push(hybrid_graph_tensor(
                &format!("past_key_values.{layer}.key"),
                "float16",
                &dense,
            ));
            inputs.push(hybrid_graph_tensor(
                &format!("past_key_values.{layer}.value"),
                "float16",
                &dense,
            ));
            outputs.push(hybrid_graph_tensor(
                &format!("present.{layer}.key"),
                "float16",
                &dense,
            ));
            outputs.push(hybrid_graph_tensor(
                &format!("present.{layer}.value"),
                "float16",
                &dense,
            ));
        } else {
            inputs.push(hybrid_graph_tensor(
                &format!("past_key_values.{layer}.conv_state"),
                "float16",
                &conv,
            ));
            inputs.push(hybrid_graph_tensor(
                &format!("past_key_values.{layer}.recurrent_state"),
                "float16",
                &recur,
            ));
            // Stock exports leave the present recurrent-state shapes fully
            // symbolic even though the paired past input carries the concrete
            // running-state extent; the config-free fallback must accept this.
            outputs.push(hybrid_graph_tensor(
                &format!("present.{layer}.conv_state"),
                "float16",
                &[None, None, None],
            ));
            outputs.push(hybrid_graph_tensor(
                &format!("present.{layer}.recurrent_state"),
                "float16",
                &[None, None, None, None],
            ));
        }
    }
    ModelGraphInfo { inputs, outputs }
}

/// A dense qwen3-0.6b-shaped graph: 28 decoder layers, all dense GQA
/// (`key`/`value`), no recurrent state ports.
fn qwen06b_dense_graph() -> ModelGraphInfo {
    let dense = [Some(1), Some(8), None, Some(128)];
    let mut inputs = vec![
        hybrid_graph_tensor("input_ids", "int64", &[Some(1), None]),
        hybrid_graph_tensor("attention_mask", "int64", &[Some(1), None]),
        hybrid_graph_tensor("position_ids", "int64", &[Some(1), None]),
    ];
    let mut outputs = vec![hybrid_graph_tensor(
        "logits",
        "float32",
        &[Some(1), None, Some(151936)],
    )];
    for layer in 0..28 {
        inputs.push(hybrid_graph_tensor(
            &format!("past_key_values.{layer}.key"),
            "float16",
            &dense,
        ));
        inputs.push(hybrid_graph_tensor(
            &format!("past_key_values.{layer}.value"),
            "float16",
            &dense,
        ));
        outputs.push(hybrid_graph_tensor(
            &format!("present.{layer}.key"),
            "float16",
            &dense,
        ));
        outputs.push(hybrid_graph_tensor(
            &format!("present.{layer}.value"),
            "float16",
            &dense,
        ));
    }
    ModelGraphInfo { inputs, outputs }
}

#[test]
fn derive_decoder_io_from_graph_splits_hybrid_kv_and_state() {
    // The 27b hybrid layout: 16 dense GQA layers (2 ports each = 32 kv entries)
    // and 48 recurrent layers (conv_state + recurrent_state = 96 state pairs).
    let derived = GenAiConfig::derive_decoder_io_from_graph(&qwen27b_hybrid_graph())
        .expect("hybrid graph derives decoder io");
    assert_eq!(derived.kv_inputs.len(), 32);
    assert_eq!(derived.kv_outputs.len(), 32);
    assert_eq!(derived.state_pairs.len(), 96);
    assert_eq!(derived.kv_dtype, "float16");

    // KV lists must contain ONLY the dense full-attention ports.
    assert!(
        derived
            .kv_inputs
            .contains(&"past_key_values.3.key".to_string())
    );
    assert!(
        derived
            .kv_inputs
            .contains(&"past_key_values.63.value".to_string())
    );
    assert!(
        !derived
            .kv_inputs
            .iter()
            .any(|name| name.contains("conv_state") || name.contains("recurrent_state"))
    );

    // Every recurrent port is a loop-carried state pair, past→present paired.
    assert!(derived.state_pairs.iter().any(|pair| {
        pair.input == "past_key_values.0.conv_state" && pair.output == "present.0.conv_state"
    }));
    assert!(derived.state_pairs.iter().any(|pair| {
        pair.input == "past_key_values.62.recurrent_state"
            && pair.output == "present.62.recurrent_state"
    }));
    assert!(
        derived
            .state_pairs
            .iter()
            .all(|pair| pair.input.contains("_state") && pair.output.contains("_state"))
    );
}

#[test]
fn derive_decoder_io_from_graph_dense_has_no_state_pairs() {
    // A pure-dense GQA decoder must derive KV ports but NEVER over-derive state
    // pairs — the loader's safety gate leaves such models on the existing
    // shape-inference path unchanged.
    let derived = GenAiConfig::derive_decoder_io_from_graph(&qwen06b_dense_graph())
        .expect("dense graph derives decoder io");
    assert_eq!(derived.kv_inputs.len(), 56);
    assert_eq!(derived.kv_outputs.len(), 56);
    assert!(
        derived.state_pairs.is_empty(),
        "dense decoder must not gain recurrent state pairs, got {:?}",
        derived.state_pairs
    );
}

#[test]
fn derive_model_io_spec_from_graph_dense_binds_kv_ports() {
    // A pure-dense decoder now DOES auto-derive, because the only caller runs
    // after a declared or pattern-expanded `io` block failed to materialise —
    // so returning None there does not preserve a working path, it leaves the
    // model with no KV geometry and fails the load (#1012, DeepSeek-V2 MLA).
    // The gate moved from "has recurrent state pairs" to "yielded KV ports".
    let io = GenAiConfig::derive_model_io_spec_from_graph(&qwen06b_dense_graph())
        .expect("a dense decoder with KV ports must auto-derive an io spec");
    assert!(
        io.kv_inputs.as_ref().is_some_and(|v| !v.is_empty()),
        "dense derivation must bind KV inputs"
    );
    assert!(
        io.kv_outputs.as_ref().is_some_and(|v| !v.is_empty()),
        "dense derivation must bind KV outputs"
    );
    assert!(
        io.state_pairs.is_none(),
        "a dense decoder has no recurrent state; that must be None, not an empty list"
    );
}

#[test]
fn derive_model_io_spec_from_graph_hybrid_binds_ports() {
    // The recurrent-hybrid case: the helper reuses the guarded classifier,
    // passes the non-empty state_pairs safety gate, binds the conventional
    // non-KV ports by name-presence, and assembles the ModelIoSpec.
    let io = GenAiConfig::derive_model_io_spec_from_graph(&qwen27b_hybrid_graph())
        .expect("hybrid graph auto-derives an io spec");
    assert_eq!(io.token_input.as_deref(), Some("input_ids"));
    assert_eq!(io.attention_mask_input.as_deref(), Some("attention_mask"));
    assert_eq!(io.position_ids_input.as_deref(), Some("position_ids"));
    assert_eq!(io.logits_output.as_deref(), Some("logits"));
    assert_eq!(io.kv_inputs.as_ref().map(Vec::len), Some(32));
    assert_eq!(io.kv_outputs.as_ref().map(Vec::len), Some(32));
    let state_pairs = io
        .state_pairs
        .expect("hybrid derives recurrent state pairs");
    assert_eq!(state_pairs.len(), 96);
    assert!(
        state_pairs
            .iter()
            .all(|pair| pair.init.as_deref() == Some("zeros")
                && pair.update.as_deref() == Some("replace"))
    );
    // Fields the auto-derive path never populates stay unset.
    assert!(io.kv_layout.is_none());
    assert!(io.sequence_source.is_none());
    assert!(io.kv_update.is_none());
    assert!(io.static_cache.is_none());
}

#[test]
fn uniform_decoder_graph_matches_pattern_expansion() {
    // A dense-KV model must produce the SAME kv_inputs whether or not the
    // graph is supplied, and must never gain state pairs.
    let cfg = qwen_config();
    let without_graph = cfg.to_inference_metadata(Some("float16")).unwrap();
    let expected_kv = without_graph
        .model
        .as_ref()
        .and_then(|m| m.io.as_ref())
        .and_then(|io| io.kv_inputs.clone())
        .expect("pattern-expanded kv inputs");

    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for layer in 0..24 {
        inputs.push(hybrid_graph_tensor(
            &format!("past_key_values.{layer}.key"),
            "float16",
            &[Some(1), Some(2), None, Some(64)],
        ));
        inputs.push(hybrid_graph_tensor(
            &format!("past_key_values.{layer}.value"),
            "float16",
            &[Some(1), Some(2), None, Some(64)],
        ));
        outputs.push(hybrid_graph_tensor(
            &format!("present.{layer}.key"),
            "float16",
            &[Some(1), Some(2), None, Some(64)],
        ));
        outputs.push(hybrid_graph_tensor(
            &format!("present.{layer}.value"),
            "float16",
            &[Some(1), Some(2), None, Some(64)],
        ));
    }
    let graph = ModelGraphInfo { inputs, outputs };
    let with_graph = cfg
        .to_inference_metadata_with_graph(Some("float16"), &graph)
        .unwrap();
    let io = with_graph
        .model
        .as_ref()
        .and_then(|m| m.io.as_ref())
        .expect("io");
    assert_eq!(io.kv_inputs.as_ref(), Some(&expected_kv));
    assert!(io.state_pairs.is_none());
}

fn qwen_config() -> GenAiConfig {
    serde_json::from_str(
        r#"{
            "model": {
                "type": "qwen2",
                "context_length": 32768,
                "decoder": {
                    "head_size": 64,
                    "hidden_size": 896,
                    "num_attention_heads": 14,
                    "num_hidden_layers": 24,
                    "num_key_value_heads": 2
                }
            },
            "search": { "past_present_share_buffer": true, "max_length": 32768 }
        }"#,
    )
    .expect("valid genai_config")
}

#[test]
fn detects_gqa_and_capacity() {
    let cfg = qwen_config();
    assert!(cfg.is_group_query_attention());
    assert_eq!(cfg.max_sequence_length(), Some(32768));
    assert!(cfg.shared_kv_buffer_supported());
}

#[test]
fn converts_and_enables_share_buffer_with_fp16() {
    let cfg = qwen_config();
    let md = cfg.to_inference_metadata(Some("float16")).unwrap();
    assert_eq!(md.schema_version.as_deref(), Some("v1"));
    let attention = md
        .model
        .as_ref()
        .and_then(|m| m.attention.as_ref())
        .expect("attention");
    assert_eq!(attention.attention_type, "group_query_attention");
    assert_eq!(attention.num_kv_heads, Some(2));
    assert_eq!(attention.num_attention_heads, Some(14));
    assert_eq!(attention.head_dim, Some(64));
    assert_eq!(
        attention
            .key_sequence_lengths
            .as_ref()
            .and_then(|spec| spec.scalar_broadcast),
        Some(onnx_genai_metadata::SequenceLengthScalarBroadcast::UnitBatch)
    );
    assert_eq!(
        md.model.as_ref().and_then(|m| m.max_sequence_length),
        Some(32768)
    );
    assert_eq!(
        md.kv_cache
            .as_ref()
            .and_then(|kv| kv.native_dtype.as_deref()),
        Some("float16")
    );
}

#[test]
fn converts_and_enables_share_buffer_with_bf16() {
    let cfg = qwen_config();
    let md = cfg.to_inference_metadata(Some("bfloat16")).unwrap();
    assert_eq!(
        md.kv_cache
            .as_ref()
            .and_then(|kv| kv.native_dtype.as_deref()),
        Some("bfloat16")
    );
}

#[test]
fn omits_kv_cache_when_share_buffer_disabled() {
    let mut cfg = qwen_config();
    cfg.search.past_present_share_buffer = Some(false);
    let md = cfg.to_inference_metadata(Some("float16")).unwrap();
    assert!(md.kv_cache.is_none());
}

#[test]
fn omits_kv_cache_for_unsupported_dtype() {
    let cfg = qwen_config();
    let md = cfg.to_inference_metadata(Some("int8")).unwrap();
    assert!(md.kv_cache.is_none());
}

#[test]
fn omits_kv_cache_when_dtype_unknown() {
    let cfg = qwen_config();
    let md = cfg.to_inference_metadata(None).unwrap();
    assert!(md.kv_cache.is_none());
    assert!(md.model.and_then(|m| m.attention).is_some());
}

#[test]
fn full_mha_via_gqa_op_is_share_buffer_eligible() {
    let mut cfg = qwen_config();
    cfg.model.decoder.num_attention_heads = Some(14);
    cfg.model.decoder.num_key_value_heads = Some(14);
    let md = cfg.to_inference_metadata(Some("float16")).unwrap();
    assert!(!cfg.is_group_query_attention());
    assert!(cfg.uses_group_query_attention_op());
    assert!(cfg.shared_kv_buffer_supported());
    assert!(md.kv_cache.is_some());
    assert_eq!(
        md.model
            .and_then(|m| m.attention)
            .map(|a| (a.attention_type, a.key_sequence_lengths)),
        Some((
            "group_query_attention".to_string(),
            Some(onnx_genai_metadata::KeySequenceLengthsSpec {
                scalar_broadcast: Some(
                    onnx_genai_metadata::SequenceLengthScalarBroadcast::UnitBatch
                ),
            })
        ))
    );
}

#[test]
fn non_gqa_op_omits_scalar_key_sequence_lengths_permission() {
    let mut cfg = qwen_config();
    cfg.model.decoder.num_key_value_heads = None;
    let md = cfg.to_inference_metadata(Some("float16")).unwrap();
    assert_eq!(
        md.model
            .and_then(|m| m.attention)
            .map(|a| (a.attention_type, a.key_sequence_lengths)),
        Some(("multi_head_attention".to_string(), None))
    );
}

#[test]
fn model_without_kv_heads_is_multi_head_and_not_share_buffer() {
    let mut cfg = qwen_config();
    cfg.model.decoder.num_key_value_heads = None;
    let md = cfg.to_inference_metadata(Some("float16")).unwrap();
    assert!(!cfg.uses_group_query_attention_op());
    assert!(!cfg.shared_kv_buffer_supported());
    assert!(md.kv_cache.is_none());
    assert_eq!(
        md.model.and_then(|m| m.attention).map(|a| a.attention_type),
        Some("multi_head_attention".to_string())
    );
}

// ---- Complete-coverage conversion tests -----------------------------

/// gpt2: combined `past_%d` / `present_%d` KV patterns, scalar token ids,
/// no `search` block.
fn gpt2_config() -> GenAiConfig {
    serde_json::from_str(
        r#"{
            "model": {
                "type": "gpt2",
                "pad_token_id": 98,
                "bos_token_id": 98,
                "eos_token_id": 98,
                "vocab_size": 1000,
                "context_length": 512,
                "decoder": {
                    "num_key_value_heads": 4,
                    "head_size": 8,
                    "num_hidden_layers": 5,
                    "inputs": { "past_names": "past_%d" },
                    "outputs": { "present_names": "present_%d" }
                }
            }
        }"#,
    )
    .expect("valid gpt2 genai_config")
}

#[test]
fn gpt2_expands_combined_kv_and_tokens() {
    let md = gpt2_config().to_inference_metadata(None).unwrap();

    let io = md
        .model
        .as_ref()
        .and_then(|m| m.io.as_ref())
        .expect("decoder io");
    // Combined pattern -> one entry per layer, in order.
    assert_eq!(
        io.kv_inputs.as_deref(),
        Some(&["past_0", "past_1", "past_2", "past_3", "past_4"].map(String::from)[..])
    );
    assert_eq!(
        io.kv_outputs.as_deref(),
        Some(
            &[
                "present_0",
                "present_1",
                "present_2",
                "present_3",
                "present_4"
            ]
            .map(String::from)[..]
        )
    );
    // No inputs_embeds -> token-driven with the conventional default name.
    assert_eq!(io.token_input.as_deref(), Some("input_ids"));
    assert_eq!(io.logits_output.as_deref(), Some("logits"));

    let tokens = md.tokens.as_ref().expect("tokens");
    assert_eq!(tokens.pad_token_id, Some(98));
    assert_eq!(tokens.bos_token_id, Some(98));
    assert_eq!(tokens.eos_token_id.as_deref(), Some(&[98i64][..]));

    // No `search` block -> no generation defaults.
    assert!(md.generation.is_none());
    assert_eq!(md.model.and_then(|m| m.vocab_size), Some(1000));
}

/// Loads the real onnxruntime-genai fixtures from disk and asserts every
/// one converts without error. Gated on `ORT_GENAI_TEST_MODELS` pointing at
/// `onnxruntime-genai/test/test_models` so it stays hermetic by default.
#[test]
fn real_fixtures_convert_without_error() {
    let Ok(root) = std::env::var("ORT_GENAI_TEST_MODELS") else {
        return;
    };
    let root = std::path::Path::new(&root);
    let fixtures = [
        "hf-internal-testing/tiny-random-gpt2-fp32",
        "audio-preprocessing",
        "vision-preprocessing",
        "qwen-vision-preprocessing",
        "pipeline-model",
    ];
    for fixture in fixtures {
        let dir = root.join(fixture);
        if !dir.join(GENAI_CONFIG_FILE).is_file() {
            continue;
        }
        let md = inference_metadata_from_dir(&dir, Some("float16"))
            .unwrap_or_else(|e| panic!("{fixture}: {e}"))
            .unwrap_or_else(|| panic!("{fixture}: no genai_config.json"));
        assert_eq!(md.schema_version.as_deref(), Some("v1"), "{fixture}");
    }
}

#[test]
fn whisper_encoder_decoder_pipeline_with_cross_kv() {
    let cfg: GenAiConfig = serde_json::from_str(WHISPER_JSON).unwrap();
    let md = cfg.to_inference_metadata(None).unwrap();

    let pipeline = md.pipeline.as_ref().expect("asr pipeline");
    assert!(pipeline.models.contains_key("encoder"));
    assert!(pipeline.models.contains_key("decoder"));
    assert!(matches!(
        pipeline.strategy.kind,
        onnx_genai_metadata::PipelineStrategyKind::Composite
    ));
    // encoder -> decoder cross-attention hidden-states dataflow.
    assert_eq!(pipeline.dataflow.len(), 1);
    assert_eq!(pipeline.dataflow[0].from, "encoder.encoder_hidden_states");
    assert_eq!(pipeline.dataflow[0].to, "decoder.encoder_hidden_states");

    let io = pipeline.models["decoder"].io.as_ref().expect("decoder io");
    assert_eq!(io.token_input.as_deref(), Some("input_ids"));
    assert_eq!(
        io.kv_inputs.as_deref(),
        Some(&["past_key_self_0", "past_value_self_0"].map(String::from)[..])
    );
    assert_eq!(
        io.kv_outputs.as_deref(),
        Some(&["present_key_self_0", "present_value_self_0"].map(String::from)[..])
    );
    assert_eq!(
        io.cross_kv_inputs.as_deref(),
        Some(&["past_key_cross_0", "past_value_cross_0"].map(String::from)[..])
    );
    assert_eq!(
        io.cross_kv_outputs.as_deref(),
        Some(&["present_key_cross_0", "present_value_cross_0"].map(String::from)[..])
    );
    assert_eq!(
        io.encoder_hidden_states_input.as_deref(),
        Some("encoder_hidden_states")
    );

    // Generation defaults come from `search`.
    let generation = md.generation.as_ref().expect("generation");
    assert_eq!(generation.max_length, Some(448));
    assert_eq!(generation.do_sample, Some(false));
    assert_eq!(generation.num_beams, Some(1));
}

#[test]
fn whisper_strict_encoder_decoder_synth_routes_cross_kv() {
    // Strict, graph-verified encoder-decoder synth (the path the ORT compat
    // loader uses). Unlike the pattern-expanded `to_inference_metadata`, the
    // cross-attention KV is wired as explicit encoder->decoder dataflow edges
    // (static, computed once by the encoder), and the audio prompt input is
    // surfaced on the encoder component.
    let cfg: GenAiConfig = serde_json::from_str(WHISPER_JSON).unwrap();
    let graphs = EncoderDecoderGraphInfo {
        encoder: ModelGraphInfo {
            inputs: vec![hybrid_graph_tensor(
                "audio_features",
                "float32",
                &[Some(1), Some(80), Some(3000)],
            )],
            outputs: vec![
                hybrid_graph_tensor(
                    "encoder_hidden_states",
                    "float32",
                    &[Some(1), Some(1500), Some(384)],
                ),
                hybrid_graph_tensor(
                    "present_key_cross_0",
                    "float32",
                    &[Some(1), Some(6), Some(1500), Some(64)],
                ),
                hybrid_graph_tensor(
                    "present_value_cross_0",
                    "float32",
                    &[Some(1), Some(6), Some(1500), Some(64)],
                ),
            ],
        },
        decoder: ModelGraphInfo {
            inputs: vec![
                hybrid_graph_tensor("input_ids", "int64", &[Some(1), None]),
                hybrid_graph_tensor(
                    "past_key_self_0",
                    "float32",
                    &[Some(1), Some(6), None, Some(64)],
                ),
                hybrid_graph_tensor(
                    "past_value_self_0",
                    "float32",
                    &[Some(1), Some(6), None, Some(64)],
                ),
                hybrid_graph_tensor(
                    "past_key_cross_0",
                    "float32",
                    &[Some(1), Some(6), Some(1500), Some(64)],
                ),
                hybrid_graph_tensor(
                    "past_value_cross_0",
                    "float32",
                    &[Some(1), Some(6), Some(1500), Some(64)],
                ),
            ],
            outputs: vec![
                hybrid_graph_tensor("logits", "float32", &[Some(1), None, Some(51865)]),
                hybrid_graph_tensor(
                    "present_key_self_0",
                    "float32",
                    &[Some(1), Some(6), None, Some(64)],
                ),
                hybrid_graph_tensor(
                    "present_value_self_0",
                    "float32",
                    &[Some(1), Some(6), None, Some(64)],
                ),
            ],
        },
    };

    let metadata = cfg
        .to_strict_encoder_decoder_pipeline_metadata(&graphs)
        .expect("strict encoder-decoder synth");
    let pipeline = metadata.pipeline.as_ref().expect("pipeline");
    onnx_genai_metadata::validate_pipeline_spec(pipeline).expect("valid pipeline spec");

    // Encoder + decoder components.
    assert_eq!(pipeline.models["encoder"].role, "encoder");
    assert_eq!(pipeline.models["decoder"].role, "decoder");

    // Audio prompt input surfaced on the encoder.
    let encoder_io = pipeline.models["encoder"].io.as_ref().expect("encoder io");
    assert_eq!(
        encoder_io.audio_features_input.as_deref(),
        Some("audio_features")
    );

    // Decoder self-KV grows; cross-KV is present as static routing.
    let decoder_io = pipeline.models["decoder"].io.as_ref().expect("decoder io");
    assert_eq!(decoder_io.logits_output.as_deref(), Some("logits"));
    assert_eq!(decoder_io.kv_update.as_deref(), Some("append"));
    assert_eq!(
        decoder_io.kv_inputs.as_deref(),
        Some(&["past_key_self_0", "past_value_self_0"].map(String::from)[..])
    );
    assert_eq!(
        decoder_io.kv_outputs.as_deref(),
        Some(&["present_key_self_0", "present_value_self_0"].map(String::from)[..])
    );
    assert_eq!(
        decoder_io.cross_kv_inputs.as_deref(),
        Some(&["past_key_cross_0", "past_value_cross_0"].map(String::from)[..])
    );
    assert_eq!(
        decoder_io.cross_kv_outputs.as_deref(),
        Some(&["present_key_cross_0", "present_value_cross_0"].map(String::from)[..])
    );

    // Cross-attention KV static routing is declared by the positional pairing
    // of the decoder's cross_kv_inputs (past_*_cross) with cross_kv_outputs
    // (the encoder-produced present_*_cross), computed ONCE by the encoder —
    // NOT recomputed each step and NOT a per-step dataflow edge. This decoder
    // has no encoder_hidden_states input, so no dataflow edge is synthesized.
    assert!(
        pipeline.dataflow.is_empty(),
        "cross-KV is stateful routing, not per-step edges: {:?}",
        pipeline.dataflow
    );

    assert!(matches!(
        pipeline.strategy.kind,
        onnx_genai_metadata::PipelineStrategyKind::Composite
    ));
}

// A faithful, trimmed synthetic derived from the real Microsoft
// `nemotron_speech` genai_config.json (Conformer-Transducer / RNN-T):
// a streaming Conformer encoder with cache state, an LSTM prediction
// network (`targets` + `lstm_hidden_state`/`lstm_cell_state`, no attention
// KV), a joint (joiner) network, and a Silero VAD. The multi-GB .onnx
// weights are not needed — recognition is driven from the JSON alone.
const NEMOTRON_TRANSDUCER_JSON: &str = r#"{
    "model": {
        "type": "nemotron_speech",
        "vocab_size": 13088,
        "subsampling_factor": 8,
        "blank_id": 13087,
        "max_symbols_per_step": 10,
        "encoder": {
            "filename": "encoder.onnx",
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "inputs": {
                "audio_features": "audio_signal",
                "cache_last_channel": "cache_last_channel",
                "cache_last_time": "cache_last_time",
                "cache_last_channel_len": "cache_last_channel_len",
                "lang_id": "lang_id"
            },
            "outputs": {
                "encoder_outputs": "outputs",
                "output_lengths": "encoded_lengths",
                "cache_last_channel_next": "cache_last_channel_next",
                "cache_last_time_next": "cache_last_time_next",
                "cache_last_channel_len_next": "cache_last_channel_len_next"
            }
        },
        "decoder": {
            "filename": "decoder.onnx",
            "hidden_size": 640,
            "num_hidden_layers": 2,
            "inputs": {
                "targets": "targets",
                "lstm_hidden_state": "h_in",
                "lstm_cell_state": "c_in"
            },
            "outputs": {
                "outputs": "decoder_output",
                "lstm_hidden_state": "h_out",
                "lstm_cell_state": "c_out"
            }
        },
        "joiner": {
            "filename": "joint.onnx",
            "inputs": {
                "encoder_outputs": "encoder_output",
                "decoder_outputs": "decoder_output"
            },
            "outputs": { "logits": "joint_output" }
        },
        "vad": {
            "filename": "silero_vad.onnx",
            "threshold": 0.3
        }
    }
}"#;

#[test]
fn nemotron_transducer_is_not_encoder_decoder() {
    let cfg: GenAiConfig = serde_json::from_str(NEMOTRON_TRANSDUCER_JSON).unwrap();
    // Detected structurally as a transducer even though it declares
    // `model.encoder` (which alone would look like an encoder-decoder).
    assert!(cfg.is_transducer());
    assert_eq!(cfg.shape(), ModelShape::Transducer);
    assert_ne!(cfg.shape(), ModelShape::EncoderDecoder);
}

#[test]
fn nemotron_transducer_declines_instead_of_fabricating_cross_kv() {
    let cfg: GenAiConfig = serde_json::from_str(NEMOTRON_TRANSDUCER_JSON).unwrap();
    // The non-strict synthesis path (the auto-detection fallback) must NOT
    // fabricate a Whisper-style encoder-decoder spec (with default
    // `input_ids`/`logits` ports and non-existent `past_key_values.*` /
    // `present.*` cross/self KV). It declines with the honest family error.
    let error = cfg
        .to_inference_metadata(None)
        .expect_err("transducer must not synthesize a pipeline");
    match error {
        GenAiConfigError::UnsupportedPipelineFamily { family, .. } => {
            assert_eq!(family, "RNN-T transducer");
        }
        other => panic!("expected UnsupportedPipelineFamily, got {other:?}"),
    }
}

#[test]
fn nemotron_transducer_strict_from_dir_declines() {
    // The strict encoder-decoder loader entry point declines a transducer
    // directory explicitly rather than returning Ok(None) or fabricating.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("test-tmp")
        .join(format!(
            "nemotron_transducer_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(GENAI_CONFIG_FILE), NEMOTRON_TRANSDUCER_JSON).unwrap();
    let graphs = EncoderDecoderGraphInfo::default();
    let result = encoder_decoder_pipeline_inference_metadata_from_dir(&dir, &graphs);
    std::fs::remove_dir_all(&dir).ok();
    match result {
        Err(GenAiConfigError::UnsupportedPipelineFamily { family, .. }) => {
            assert_eq!(family, "RNN-T transducer");
        }
        other => panic!("expected UnsupportedPipelineFamily, got {other:?}"),
    }
}

#[test]
fn transducer_detected_from_lstm_decoder_without_joiner() {
    // Even without a `joiner` section, an LSTM prediction network (targets +
    // LSTM hidden/cell state, no attention KV) is a transducer signal.
    let json = r#"{
        "model": {
            "type": "some_transducer",
            "encoder": {
                "filename": "encoder.onnx",
                "inputs": { "audio_features": "audio_signal" },
                "outputs": { "encoder_outputs": "outputs" }
            },
            "decoder": {
                "filename": "decoder.onnx",
                "num_hidden_layers": 2,
                "inputs": {
                    "targets": "targets",
                    "lstm_hidden_state": "h_in",
                    "lstm_cell_state": "c_in"
                },
                "outputs": { "outputs": "decoder_output" }
            }
        }
    }"#;
    let cfg: GenAiConfig = serde_json::from_str(json).unwrap();
    assert!(cfg.is_transducer());
    assert_eq!(cfg.shape(), ModelShape::Transducer);
}

#[test]
fn whisper_still_classifies_as_encoder_decoder_not_transducer() {
    // No regression: a real cross-attention encoder-decoder (Whisper) is
    // still EncoderDecoder and is never mistaken for a transducer.
    let cfg: GenAiConfig = serde_json::from_str(WHISPER_JSON).unwrap();
    assert!(!cfg.is_transducer());
    assert_eq!(cfg.shape(), ModelShape::EncoderDecoder);
}

#[test]
fn phi3v_and_decoder_pipeline_are_not_transducers() {
    // No regression for the other shapes.
    let vlm: GenAiConfig = serde_json::from_str(PHI3V_JSON).unwrap();
    assert!(!vlm.is_transducer());
    assert_eq!(vlm.shape(), ModelShape::Multimodal);
    let pipe: GenAiConfig = serde_json::from_str(DECODER_PIPELINE_JSON).unwrap();
    assert!(!pipe.is_transducer());
    assert_eq!(pipe.shape(), ModelShape::DecoderPipeline);
}

#[test]
fn phi3v_multimodal_pipeline_with_image_token() {
    let cfg: GenAiConfig = serde_json::from_str(PHI3V_JSON).unwrap();
    let md = cfg.to_inference_metadata(None).unwrap();

    let pipeline = md.pipeline.as_ref().expect("multimodal pipeline");
    assert!(pipeline.models.contains_key("vision_encoder"));
    assert!(pipeline.models.contains_key("embedding"));
    assert!(pipeline.models.contains_key("decoder"));

    // vision -> embedding -> decoder dataflow.
    let edges: Vec<(&str, &str)> = pipeline
        .dataflow
        .iter()
        .map(|e| (e.from.as_str(), e.to.as_str()))
        .collect();
    assert!(edges.contains(&("vision_encoder.image_features", "embedding.image_features")));
    assert!(edges.contains(&("embedding.inputs_embeds", "decoder.inputs_embeds")));

    // Embeds-driven decoder io.
    let io = pipeline.models["decoder"].io.as_ref().expect("decoder io");
    assert_eq!(io.inputs_embeds_input.as_deref(), Some("inputs_embeds"));
    assert!(io.token_input.is_none());

    // phi3v declares no image_token_id, so no vision expansion contract.
    assert!(pipeline.vision.is_none());
}

#[test]
fn qwen_vlm_image_token_id_is_propagated() {
    let cfg: GenAiConfig = serde_json::from_str(QWEN_VLM_JSON).unwrap();
    let md = cfg.to_inference_metadata(None).unwrap();
    let pipeline = md.pipeline.as_ref().expect("multimodal pipeline");
    assert_eq!(
        pipeline
            .vision
            .as_ref()
            .and_then(|v| v.image_placeholder_token_id),
        Some(151_655)
    );
    let tokens = md.tokens.as_ref().expect("tokens");
    assert_eq!(tokens.image_token_id, Some(151_655));
    assert_eq!(tokens.video_token_id, Some(151_656));
    assert_eq!(tokens.vision_start_token_id, Some(151_652));
    // eos as array normalizes to a vec.
    assert_eq!(
        tokens.eos_token_id.as_deref(),
        Some(&[151_645, 151_643][..])
    );
}

#[test]
fn decoder_pipeline_emits_split_models() {
    let cfg: GenAiConfig = serde_json::from_str(DECODER_PIPELINE_JSON).unwrap();
    let md = cfg.to_inference_metadata(None).unwrap();
    let pipeline = md.pipeline.as_ref().expect("decoder pipeline");
    assert!(pipeline.models.contains_key("embeddings"));
    assert!(pipeline.models.contains_key("transformer"));
    assert!(pipeline.models.contains_key("language_model_head"));
    assert_eq!(pipeline.models["embeddings"].role, "embedding");
    assert_eq!(pipeline.models["language_model_head"].role, "lm_head");
    assert_eq!(pipeline.models["transformer"].role, "decoder");
}

const WHISPER_JSON: &str = r#"{
    "model": {
        "type": "whisper",
        "bos_token_id": 50257,
        "eos_token_id": 50257,
        "pad_token_id": 50257,
        "context_length": 448,
        "vocab_size": 51865,
        "decoder": {
            "filename": "dummy_decoder.onnx",
            "head_size": 64,
            "num_attention_heads": 6,
            "num_hidden_layers": 1,
            "num_key_value_heads": 6,
            "inputs": {
                "input_ids": "input_ids",
                "past_key_names": "past_key_self_%d",
                "past_value_names": "past_value_self_%d",
                "cross_past_key_names": "past_key_cross_%d",
                "cross_past_value_names": "past_value_cross_%d"
            },
            "outputs": {
                "logits": "logits",
                "present_key_names": "present_key_self_%d",
                "present_value_names": "present_value_self_%d",
                "output_cross_qk_names": "output_cross_qk_%d"
            }
        },
        "encoder": {
            "filename": "dummy_encoder.onnx",
            "num_attention_heads": 6,
            "num_hidden_layers": 1,
            "inputs": { "audio_features": "audio_features" },
            "outputs": {
                "encoder_hidden_states": "encoder_hidden_states",
                "cross_present_key_names": "present_key_cross_%d",
                "cross_present_value_names": "present_value_cross_%d"
            }
        }
    },
    "search": {
        "do_sample": false,
        "early_stopping": true,
        "length_penalty": 1.0,
        "max_length": 448,
        "min_length": 0,
        "num_beams": 1,
        "num_return_sequences": 1,
        "past_present_share_buffer": false,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0
    }
}"#;

const PHI3V_JSON: &str = r#"{
    "model": {
        "type": "phi3v",
        "bos_token_id": 1,
        "eos_token_id": 32007,
        "pad_token_id": 32000,
        "context_length": 131072,
        "vocab_size": 32064,
        "decoder": {
            "filename": "dummy_text.onnx",
            "head_size": 96,
            "num_attention_heads": 32,
            "num_hidden_layers": 1,
            "num_key_value_heads": 32,
            "inputs": {
                "inputs_embeds": "inputs_embeds",
                "attention_mask": "attention_mask",
                "past_key_names": "past_key_values.%d.key",
                "past_value_names": "past_key_values.%d.value"
            },
            "outputs": {
                "logits": "logits",
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value"
            }
        },
        "embedding": {
            "filename": "dummy_embedding.onnx",
            "inputs": { "input_ids": "input_ids", "image_features": "image_features" },
            "outputs": { "inputs_embeds": "inputs_embeds" }
        },
        "vision": {
            "filename": "dummy_vision.onnx",
            "inputs": { "pixel_values": "pixel_values", "image_sizes": "image_sizes" },
            "outputs": { "image_features": "image_features" }
        }
    },
    "search": { "past_present_share_buffer": true, "max_length": 131072 }
}"#;

const QWEN_VLM_JSON: &str = r#"{
    "model": {
        "type": "qwen2_5_vl",
        "bos_token_id": 151643,
        "eos_token_id": [151645, 151643],
        "pad_token_id": 151643,
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "context_length": 128000,
        "vocab_size": 152064,
        "decoder": {
            "filename": "dummy_text.onnx",
            "head_size": 128,
            "num_attention_heads": 28,
            "num_hidden_layers": 1,
            "num_key_value_heads": 4,
            "inputs": {
                "inputs_embeds": "inputs_embeds",
                "attention_mask": "attention_mask",
                "position_ids": "position_ids",
                "past_key_names": "past_key_values.%d.key",
                "past_value_names": "past_key_values.%d.value"
            },
            "outputs": {
                "logits": "logits",
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value"
            }
        },
        "embedding": {
            "filename": "dummy_embedding.onnx",
            "inputs": { "input_ids": "input_ids", "image_features": "image_features" },
            "outputs": { "inputs_embeds": "inputs_embeds" }
        },
        "vision": {
            "filename": "dummy_vision.onnx",
            "inputs": { "pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw" },
            "outputs": { "image_features": "image_features" }
        }
    },
    "search": { "past_present_share_buffer": true, "max_length": 128000 }
}"#;

const DECODER_PIPELINE_JSON: &str = r#"{
    "model": {
        "type": "decoder-pipeline",
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "pad_token_id": 50256,
        "context_length": 2048,
        "vocab_size": 51200,
        "decoder": {
            "head_size": 80,
            "num_attention_heads": 32,
            "num_hidden_layers": 1,
            "num_key_value_heads": 32,
            "inputs": {
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
                "past_key_names": "past_key_values.%d.key",
                "past_value_names": "past_key_values.%d.value"
            },
            "outputs": {
                "logits": "logits",
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value"
            },
            "pipeline": [
                {
                    "embeddings": {
                        "filename": "embeds.onnx",
                        "inputs": ["input_ids"],
                        "outputs": ["/model/embed_tokens/Gather/output_0"]
                    },
                    "transformer": {
                        "filename": "transformer.onnx",
                        "inputs": ["/model/embed_tokens/Gather/output_0", "attention_mask", "past_key_values.0.key", "past_key_values.0.value"],
                        "outputs": ["hidden_states", "present.0.key", "present.0.value"]
                    },
                    "language_model_head": {
                        "filename": "lm_head.onnx",
                        "inputs": ["hidden_states"],
                        "outputs": ["logits"]
                    }
                }
            ]
        }
    },
    "search": { "past_present_share_buffer": true, "max_length": 2048 }
}"#;
