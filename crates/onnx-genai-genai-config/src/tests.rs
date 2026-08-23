use super::*;

#[test]
fn composite_compatibility_synthesis_is_rejected() {
    let config: GenAiConfig = serde_json::from_value(serde_json::json!({
        "model": {
            "type": "generic",
            "decoder": { "filename": "decoder.onnx" },
            "vision": { "filename": "vision.onnx" },
            "embedding": { "filename": "embedding.onnx" }
        }
    }))
    .expect("compatibility config parses");
    let error = config
        .to_inference_metadata(None)
        .expect_err("composite fallback must be rejected");
    assert!(error.to_string().contains("pipeline.workflow"));
}

fn graph_tensor(name: &str, dtype: &str, dims: &[Option<usize>]) -> GraphTensorInfo {
    GraphTensorInfo {
        name: name.to_string(),
        dtype: dtype.to_string(),
        dimensions: dims.to_vec(),
    }
}

/// A pure-dense GQA decoder (qwen3-0.6b-shaped): 28 layers of
/// `past_key_values.<layer>.key`/`.value` inputs and matching `present.<layer>`
/// outputs, with no recurrent state ports at all.
fn qwen06b_dense_graph() -> ModelGraphInfo {
    let dense = [Some(1), Some(8), None, Some(128)];
    let mut inputs = vec![
        graph_tensor("input_ids", "int64", &[Some(1), None]),
        graph_tensor("attention_mask", "int64", &[Some(1), None]),
        graph_tensor("position_ids", "int64", &[Some(1), None]),
    ];
    let mut outputs = vec![graph_tensor(
        "logits",
        "float32",
        &[Some(1), None, Some(151936)],
    )];
    for layer in 0..28 {
        inputs.push(graph_tensor(
            &format!("past_key_values.{layer}.key"),
            "float16",
            &dense,
        ));
        inputs.push(graph_tensor(
            &format!("past_key_values.{layer}.value"),
            "float16",
            &dense,
        ));
        outputs.push(graph_tensor(
            &format!("present.{layer}.key"),
            "float16",
            &dense,
        ));
        outputs.push(graph_tensor(
            &format!("present.{layer}.value"),
            "float16",
            &dense,
        ));
    }
    ModelGraphInfo { inputs, outputs }
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
fn derive_decoder_abi_from_graph_dense_binds_kv_ports() {
    // A pure-dense decoder now DOES auto-derive, because the only caller runs
    // after a declared or pattern-expanded `io` block failed to materialise —
    // so returning None there does not preserve a working path, it leaves the
    // model with no KV geometry and fails the load (#1012, DeepSeek-V2 MLA).
    // The gate moved from "has recurrent state pairs" to "yielded KV ports".
    let io = GenAiConfig::derive_decoder_abi_from_graph(&qwen06b_dense_graph())
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
