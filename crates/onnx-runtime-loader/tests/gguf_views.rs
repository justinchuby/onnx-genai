use onnx_runtime_loader::{GgufMetadataValue, gguf_views};
mod wire {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/build_gguf_lora.rs"
    ));
}
fn facts() -> Vec<wire::Fact> {
    vec![
        ("general.type".into(), 8, wire::string("adapter")),
        ("adapter.type".into(), 8, wire::string("lora")),
        ("adapter.lora.alpha".into(), 6, 4f32.to_le_bytes().to_vec()),
        ("general.architecture".into(), 8, wire::string("llama")),
    ]
}
fn tensor() -> wire::Tensor {
    (
        "explicit.lora_a".into(),
        0,
        vec![2, 3],
        [1f32, 2., 3., 4., 5., 6.]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
    )
}
#[test]
fn gguf_views_preserve_typed_facts_canonical_rectangular_shape_and_borrowed_payload() {
    let mut facts = facts();
    facts.push((
        "general.author".into(),
        8,
        wire::string("harmless provenance"),
    ));
    let bytes = wire::container(&facts, &[tensor()]);
    let views = gguf_views(&bytes).unwrap();
    assert!(
        matches!(views.adapter_metadata["adapter.lora.alpha"],GgufMetadataValue::Float32(value) if value==4.0)
    );
    assert!(matches!(
        views.adapter_metadata["general.type"],
        GgufMetadataValue::String("adapter")
    ));
    assert!(!views.adapter_metadata.contains_key("general.author"));
    let view = &views.tensors["explicit.lora_a"];
    assert_eq!(view.shape, [2, 3]);
    assert!(view.packed_format.is_none());
    assert_eq!(view.data.len(), 24);
    assert_eq!(view.data.as_ptr(), bytes[bytes.len() - 24..].as_ptr());
}
#[test]
fn gguf_views_distinguish_missing_and_wrong_type_and_reject_duplicates_and_truncation() {
    let mut facts = facts();
    facts.remove(2);
    let bytes = wire::container(&facts, &[tensor()]);
    assert!(
        !gguf_views(&bytes)
            .unwrap()
            .adapter_metadata
            .contains_key("adapter.lora.alpha")
    );
    facts.push(("adapter.lora.alpha".into(), 4, 4u32.to_le_bytes().to_vec()));
    let bytes = wire::container(&facts, &[tensor()]);
    assert!(matches!(
        gguf_views(&bytes).unwrap().adapter_metadata["adapter.lora.alpha"],
        GgufMetadataValue::Other { kind: 4 }
    ));
    let mut truncated = bytes;
    truncated.pop();
    assert!(gguf_views(&truncated).is_err());
    facts.push(("adapter.lora.alpha".into(), 6, 4f32.to_le_bytes().to_vec()));
    let error = gguf_views(&wire::container(&facts, &[tensor()]))
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("repeated GGUF metadata key") && error.contains("adapter.lora.alpha"),
        "{error}"
    );
}
