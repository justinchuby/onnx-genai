//! The borrowed inspection API must retain the canonical keyed-loader validation.
use onnx_runtime_loader::safetensors_views;
fn container(header: &str, payload: &[u8]) -> Vec<u8> {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    bytes
}
#[test]
fn validated_views_borrow_the_original_payload_and_preserve_shape_dtype() {
    let header = r#"{"factor":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]}}"#;
    let values = [1_f32.to_le_bytes(), 2_f32.to_le_bytes()].concat();
    let bytes = container(header, &values);
    let views = safetensors_views(&bytes).unwrap();
    let view = &views["factor"];
    assert_eq!(view.dtype, onnx_runtime_ir::DataType::Float32);
    assert_eq!(view.shape, [1, 2]);
    assert_eq!(view.data, values);
    assert_eq!(view.data.as_ptr(), bytes[8 + header.len()..].as_ptr());
}
#[test]
fn inspection_rejects_duplicate_names_ranges_and_unindexed_payload() {
    for (header, payload) in [
        (
            r#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#,
            &[][..],
        ),
        (
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}"#,
            &[0_u8; 5][..],
        ),
        (
            r#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#,
            &[0_u8; 4][..],
        ),
    ] {
        assert!(
            safetensors_views(&container(header, payload)).is_err(),
            "{header}"
        );
    }
}
