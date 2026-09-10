use onnx_runtime_shape_inference::{InferenceRegistry, MergePolicy};

/// End-to-end: load the committed `bert_toy` model and assert that
/// `infer_graph` resolves every value in the graph.
#[test]
fn bert_toy_fully_resolves() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../onnx-runtime-session/tests/fixtures/bert_toy/model.onnx.textproto"
    );
    let mut graph = onnx_runtime_loader::load_model(path).expect("load bert_toy");

    let total = graph.num_values();
    assert!(total > 0, "model has values");

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let report = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer bert_toy");

    assert_eq!(
        report.num_unresolved(),
        0,
        "these values did not resolve: {:?}",
        report.unresolved
    );
    assert!(report.fully_resolved());
    assert_eq!(report.num_resolved(), total);
    assert!(*opsets.get("").unwrap_or(&0) >= 1);
}
