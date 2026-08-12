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
