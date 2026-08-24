//! Batching facts an encoder declares, and the contradictions they must reject.
//!
//! An encoder batches on its own terms: a request may carry zero, one, or many
//! images, so rows either pad up to a common extent or pack together with an
//! offsets/owner pair that maps items back to rows. Both readings are declared,
//! never guessed, and both are only useful if the values they name exist and are
//! shaped the way the mapping requires — which is what these tests pin.

use onnx_genai_metadata::{
    BatchLayout, ComponentBatchCapacity, ImageOutputBinding, InferenceMetadata, TensorContract,
    inference_metadata_schema_json, validate_metadata,
};

fn parse(document: &str) -> InferenceMetadata {
    serde_yaml::from_str(document).expect("metadata parses")
}

fn errors(document: &str) -> Vec<String> {
    validate_metadata(&parse(document)).expect_err("metadata must be rejected")
}

fn assert_reports(document: &str, expected: &str) {
    let errors = errors(document);
    assert!(
        errors.iter().any(|error| error.contains(expected)),
        "expected an error containing {expected:?}, got {errors:#?}"
    );
}

/// A vision encoder that pads: every request contributes one row of at most
/// `max_tiles` tiles, and `pixel_mask` says which of those tiles are real. The
/// encoder declares that eight such rows may share one invocation, provided the
/// rows already agree on the two spatial axes its artifact was built for.
const PADDED_VISION_ENCODER: &str = r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, linear_effects, typed_emit]
    inputs:
      pixel_values:
        contract:
          dtype: float32
          rank: 4
          shape: [batch, max_tiles, height, width]
          batch_layout: { kind: request_aligned, axis: 0 }
          pad_mask: pixel_mask
        role: { kind: opaque }
        source: { kind: application, name: pixel_values }
      pixel_mask:
        contract: { dtype: bool, rank: 2, shape: [batch, max_tiles], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: pixel_mask }
      prompt:
        contract: { dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: prompt }
    outputs:
      tokens:
        contract: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }
        role: tokens
        stage: pre_adapter
    components:
      vision:
        implementation: { kind: onnx, artifact: vision.onnx }
        batch_capacity: { axis: 0, max_rows: 8, uniform_axes: [2, 3] }
        ports:
          inputs:
            pixels:
              dtype: float32
              rank: 4
              shape: [batch, max_tiles, height, width]
              batch_layout: { kind: request_aligned, axis: 0 }
              pad_mask: mask
            mask: { dtype: bool, rank: 2, shape: [batch, max_tiles], batch_layout: { kind: request_aligned, axis: 0 } }
          outputs:
            embeddings: { dtype: float32, rank: 3, shape: [batch, max_tiles, hidden], batch_layout: { kind: request_aligned, axis: 0 } }
      decoder:
        implementation: { kind: onnx, artifact: decoder.onnx }
        ports:
          inputs:
            prompt: { dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: { kind: request_aligned, axis: 0 } }
            embeddings: { dtype: float32, rank: 3, shape: [batch, max_tiles, hidden], batch_layout: { kind: request_aligned, axis: 0 } }
          outputs:
            token: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }
    steps:
      - kind: invoke
        component: vision
        inputs: { pixels: pixel_values, mask: pixel_mask }
        outputs: { embeddings: vision.embeddings }
      - kind: invoke
        component: decoder
        inputs: { prompt: prompt, embeddings: vision.embeddings }
        outputs: { token: raw }
      - kind: emit
        value: raw
        output: tokens
        mode: replace
"#;

/// The same encoder, packing instead of padding: the images of every request in
/// flight are concatenated on one axis, `image_offsets` says how many items each
/// row contributed, and `image_owner` names the owning row of each item.
const PACKED_VISION_ENCODER: &str = r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, linear_effects, typed_emit]
    inputs:
      image_pixels:
        contract:
          dtype: float32
          rank: 4
          shape: [items, channels, height, width]
          batch_layout: { kind: token_packed, offsets: image_offsets, owner: image_owner, axis: 0 }
        role: { kind: opaque }
        source: { kind: application, name: image_pixels }
      image_offsets:
        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: image_offsets }
      image_owner:
        contract:
          dtype: int64
          rank: 1
          shape: [items]
          batch_layout: { kind: token_packed, offsets: image_offsets, owner: image_owner, axis: 0 }
        role: { kind: opaque }
        source: { kind: application, name: image_owner }
      prompt:
        contract: { dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: prompt }
    outputs:
      tokens:
        contract: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }
        role: tokens
        stage: pre_adapter
    components:
      vision:
        implementation: { kind: onnx, artifact: vision.onnx }
        batch_capacity: { axis: 0, uniform_axes: [1] }
        ports:
          inputs:
            pixels:
              dtype: float32
              rank: 4
              shape: [items, channels, height, width]
              batch_layout: { kind: token_packed, offsets: image_offsets, owner: image_owner, axis: 0 }
          outputs:
            features:
              dtype: float32
              rank: 2
              shape: [items, hidden]
              batch_layout: { kind: token_packed, offsets: image_offsets, owner: image_owner, axis: 0 }
      splice:
        implementation: { kind: onnx, artifact: splice.onnx }
        ports:
          inputs:
            prompt: { dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: { kind: request_aligned, axis: 0 } }
            features:
              dtype: float32
              rank: 2
              shape: [items, hidden]
              batch_layout: { kind: token_packed, offsets: image_offsets, owner: image_owner, axis: 0 }
          outputs:
            token: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }
    steps:
      - kind: invoke
        component: vision
        inputs: { pixels: image_pixels }
        outputs: { features: image_features }
      - kind: invoke
        component: splice
        inputs: { prompt: prompt, features: image_features }
        outputs: { token: raw }
      - kind: emit
        value: raw
        output: tokens
        mode: replace
"#;

#[test]
fn a_component_without_a_declared_capacity_carries_one_request_row() {
    let metadata = parse(PADDED_VISION_ENCODER);
    let decoder = metadata
        .pipeline
        .as_ref()
        .expect("fixture has a pipeline")
        .workflow
        .components
        .get("decoder")
        .expect("fixture declares a decoder");

    // Absence is the fact, not a missing fact: a component that says nothing
    // about batching carries exactly one request row per invocation.
    assert_eq!(decoder.batch_capacity, None);
    let round_trip = serde_yaml::to_string(decoder).expect("component serializes");
    assert!(
        !round_trip.contains("batch_capacity"),
        "an absent capacity must not be spelled out: {round_trip}"
    );
    let restored: onnx_genai_metadata::WorkflowComponent =
        serde_yaml::from_str(&round_trip).expect("component round-trips");
    assert_eq!(&restored, decoder);
}

#[test]
fn a_padded_encoder_declares_its_capacity_and_its_mask() {
    let metadata = parse(PADDED_VISION_ENCODER);
    validate_metadata(&metadata).expect("padded vision encoder is valid");

    let workflow = &metadata
        .pipeline
        .as_ref()
        .expect("fixture has a pipeline")
        .workflow;
    assert_eq!(
        workflow.components["vision"].batch_capacity,
        Some(ComponentBatchCapacity {
            axis: 0,
            max_rows: Some(8),
            uniform_axes: vec![2, 3],
        })
    );
    assert_eq!(
        workflow.inputs["pixel_values"].contract.pad_mask.as_deref(),
        Some("pixel_mask")
    );

    let round_trip =
        serde_yaml::to_string(&workflow.components["vision"]).expect("component serializes");
    let restored: onnx_genai_metadata::WorkflowComponent =
        serde_yaml::from_str(&round_trip).expect("component round-trips");
    assert_eq!(&restored, &workflow.components["vision"]);
}

#[test]
fn a_packed_encoder_result_maps_items_back_to_rows() {
    let metadata = parse(PACKED_VISION_ENCODER);
    validate_metadata(&metadata).expect("packed vision encoder is valid");

    let layout = &metadata
        .pipeline
        .as_ref()
        .expect("fixture has a pipeline")
        .workflow
        .inputs["image_pixels"]
        .contract
        .batch_layout;
    assert_eq!(layout.packed_axis(), Some(0));
    assert_eq!(layout.batch_axis(), Some(0));
    assert_eq!(layout.request_axis(), None);
    assert_eq!(layout.packing(), Some(("image_offsets", "image_owner")));
    assert_eq!(layout.kind_name(), "token_packed");
}

#[test]
fn packed_offsets_must_name_a_declared_value() {
    let document = PACKED_VISION_ENCODER.replace("offsets: image_offsets", "offsets: item_starts");
    assert_reports(
        &document,
        "token_packed offsets reference 'item_starts', which this workflow does not declare",
    );
}

#[test]
fn a_packed_owner_map_must_name_a_declared_value() {
    let document = PACKED_VISION_ENCODER.replace(
        "batch_layout: { kind: token_packed, offsets: image_offsets, owner: image_owner, axis: 0 }\n        role: { kind: opaque }\n        source: { kind: application, name: image_pixels }",
        "batch_layout: { kind: token_packed, offsets: image_offsets, owner: image_rows, axis: 0 }\n        role: { kind: opaque }\n        source: { kind: application, name: image_pixels }",
    );
    assert_reports(
        &document,
        "token_packed owner map references 'image_rows', which this workflow does not declare",
    );
}

#[test]
fn packed_offsets_must_be_request_aligned_integers() {
    let float_offsets = PACKED_VISION_ENCODER.replace(
        "contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }",
        "contract: { dtype: float32, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }",
    );
    assert_reports(
        &float_offsets,
        "token_packed offsets 'image_offsets' must have an integer dtype, not 'float32'",
    );

    let shared_offsets = PACKED_VISION_ENCODER.replace(
        "contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }",
        "contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: shared } }",
    );
    assert_reports(
        &shared_offsets,
        "token_packed offsets 'image_offsets' must declare a request_aligned batch_layout on axis 0",
    );

    let matrix_offsets = PACKED_VISION_ENCODER.replace(
        "contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }",
        "contract: { dtype: int64, rank: 2, shape: [batch, 2], batch_layout: { kind: request_aligned, axis: 0 } }",
    );
    assert_reports(
        &matrix_offsets,
        "token_packed offsets 'image_offsets' must be rank one with one prefix offset per request \
         row, not rank 2",
    );
}

#[test]
fn a_packed_owner_map_is_never_one_row_per_request() {
    let document = PACKED_VISION_ENCODER.replace(
        "          batch_layout: { kind: token_packed, offsets: image_offsets, owner: image_owner, axis: 0 }\n        role: { kind: opaque }\n        source: { kind: application, name: image_owner }",
        "          batch_layout: { kind: request_aligned, axis: 0 }\n        role: { kind: opaque }\n        source: { kind: application, name: image_owner }",
    );
    assert_reports(
        &document,
        "token_packed owner map 'image_owner' declares a request_aligned batch_layout",
    );
}

#[test]
fn a_packed_owner_map_describes_its_own_items() {
    let document = PACKED_VISION_ENCODER.replace(
        "          batch_layout: { kind: token_packed, offsets: image_offsets, owner: image_owner, axis: 0 }\n        role: { kind: opaque }\n        source: { kind: application, name: image_owner }",
        "          batch_layout: { kind: token_packed, offsets: image_offsets, owner: image_pixels, axis: 0 }\n        role: { kind: opaque }\n        source: { kind: application, name: image_owner }",
    );
    assert_reports(
        &document,
        "owner map 'image_owner' names 'image_pixels' as its own owner map",
    );
}

#[test]
fn packed_offsets_and_owner_cannot_be_one_value() {
    let document = PACKED_VISION_ENCODER.replace(
        "owner: image_owner, axis: 0",
        "owner: image_offsets, axis: 0",
    );
    assert_reports(
        &document,
        "names 'image_offsets' as both its packed offsets and its owner map",
    );
}

#[test]
fn a_packed_value_cannot_pack_on_an_axis_it_does_not_have() {
    let document = PACKED_VISION_ENCODER.replace(
        "batch_layout: { kind: token_packed, offsets: image_offsets, owner: image_owner, axis: 0 }\n        role: { kind: opaque }\n        source: { kind: application, name: image_pixels }",
        "batch_layout: { kind: token_packed, offsets: image_offsets, owner: image_owner, axis: 7 }\n        role: { kind: opaque }\n        source: { kind: application, name: image_pixels }",
    );
    assert_reports(
        &document,
        "workflow input 'image_pixels' packs items on axis 7, outside rank 4",
    );
}

#[test]
fn a_pad_mask_must_name_a_declared_value() {
    let document = PADDED_VISION_ENCODER.replace("pad_mask: pixel_mask", "pad_mask: tile_mask");
    assert_reports(
        &document,
        "pad_mask references 'tile_mask', which this workflow does not declare",
    );
}

#[test]
fn a_pad_mask_must_be_boolean_or_integer() {
    let document = PADDED_VISION_ENCODER.replace(
        "        contract: { dtype: bool, rank: 2, shape: [batch, max_tiles], batch_layout: { kind: request_aligned, axis: 0 } }\n        role: { kind: opaque }\n        source: { kind: application, name: pixel_mask }",
        "        contract: { dtype: float32, rank: 2, shape: [batch, max_tiles], batch_layout: { kind: request_aligned, axis: 0 } }\n        role: { kind: opaque }\n        source: { kind: application, name: pixel_mask }",
    );
    assert_reports(
        &document,
        "pad_mask 'pixel_mask' must have a bool or integer dtype marking valid entries, not \
         'float32'",
    );
}

#[test]
fn a_pad_mask_moves_with_the_rows_it_describes() {
    let document = PADDED_VISION_ENCODER.replace(
        "        contract: { dtype: bool, rank: 2, shape: [batch, max_tiles], batch_layout: { kind: request_aligned, axis: 0 } }\n        role: { kind: opaque }\n        source: { kind: application, name: pixel_mask }",
        "        contract: { dtype: bool, rank: 2, shape: [batch, max_tiles], batch_layout: { kind: shared } }\n        role: { kind: opaque }\n        source: { kind: application, name: pixel_mask }",
    );
    assert_reports(
        &document,
        "pad_mask 'pixel_mask' is shared with no request axis but the value it masks is \
         request-aligned on axis 0",
    );
}

#[test]
fn a_packed_value_has_no_padding_to_mask() {
    let document = PACKED_VISION_ENCODER.replace(
        "          batch_layout: { kind: token_packed, offsets: image_offsets, owner: image_owner, axis: 0 }\n        role: { kind: opaque }\n        source: { kind: application, name: image_pixels }",
        "          batch_layout: { kind: token_packed, offsets: image_offsets, owner: image_owner, axis: 0 }\n          pad_mask: image_owner\n        role: { kind: opaque }\n        source: { kind: application, name: image_pixels }",
    );
    assert_reports(
        &document,
        "workflow input 'image_pixels' is token_packed and declares pad_mask 'image_owner'",
    );
}

#[test]
fn a_capacity_batches_the_axis_its_ports_batch() {
    let document = PADDED_VISION_ENCODER.replace(
        "batch_capacity: { axis: 0, max_rows: 8, uniform_axes: [2, 3] }",
        "batch_capacity: { axis: 1, max_rows: 8 }",
    );
    assert_reports(
        &document,
        "workflow component 'vision' input port 'mask' batches on axis 0 but the component \
         declares batch_capacity axis 1",
    );
}

#[test]
fn a_capacity_axis_must_exist_on_the_ports_it_batches() {
    let document = PADDED_VISION_ENCODER.replace(
        "batch_capacity: { axis: 0, max_rows: 8, uniform_axes: [2, 3] }",
        "batch_capacity: { axis: 4, max_rows: 8 }",
    );
    assert_reports(
        &document,
        "workflow component 'vision' declares batch_capacity axis 4 but input port 'mask' has \
         rank 2",
    );
}

#[test]
fn a_capacity_carries_at_least_one_row() {
    let document = PADDED_VISION_ENCODER.replace(
        "batch_capacity: { axis: 0, max_rows: 8, uniform_axes: [2, 3] }",
        "batch_capacity: { axis: 0, max_rows: 0 }",
    );
    assert_reports(
        &document,
        "workflow component 'vision' declares batch_capacity max_rows 0",
    );
}

#[test]
fn uniform_axes_are_distinct_in_range_and_never_the_batch_axis() {
    let repeated = PADDED_VISION_ENCODER.replace(
        "batch_capacity: { axis: 0, max_rows: 8, uniform_axes: [2, 3] }",
        "batch_capacity: { axis: 0, max_rows: 8, uniform_axes: [2, 2] }",
    );
    assert_reports(
        &repeated,
        "workflow component 'vision' repeats axis 2 in batch_capacity.uniform_axes",
    );

    let out_of_range = PADDED_VISION_ENCODER.replace(
        "batch_capacity: { axis: 0, max_rows: 8, uniform_axes: [2, 3] }",
        "batch_capacity: { axis: 0, max_rows: 8, uniform_axes: [9] }",
    );
    assert_reports(
        &out_of_range,
        "workflow component 'vision' declares batch_capacity uniform axis 9 but its widest \
         request-scoped port has rank 4",
    );

    let batch_axis = PADDED_VISION_ENCODER.replace(
        "batch_capacity: { axis: 0, max_rows: 8, uniform_axes: [2, 3] }",
        "batch_capacity: { axis: 0, max_rows: 8, uniform_axes: [0] }",
    );
    assert_reports(
        &batch_axis,
        "workflow component 'vision' requires uniformity on batch_capacity axis 0",
    );
}

#[test]
fn a_capacity_cannot_outgrow_a_fixed_row_dimension() {
    let document = PADDED_VISION_ENCODER
        .replace(
            "            mask: { dtype: bool, rank: 2, shape: [batch, max_tiles], batch_layout: { kind: request_aligned, axis: 0 } }",
            "            mask: { dtype: bool, rank: 2, shape: [1, max_tiles], batch_layout: { kind: request_aligned, axis: 0 } }",
        );
    assert_reports(
        &document,
        "workflow component 'vision' declares batch_capacity max_rows 8 but input port 'mask' pins \
         axis 0 to a fixed dimension of 1",
    );
}

#[test]
fn a_capacity_needs_ports_that_carry_rows() {
    let document = PADDED_VISION_ENCODER.replace(
        "      decoder:\n        implementation: { kind: onnx, artifact: decoder.onnx }",
        "      decoder:\n        implementation: { kind: onnx, artifact: decoder.onnx }\n        batch_capacity: { axis: 0, max_rows: 4 }\n",
    );
    let document = document.replace(
        "            prompt: { dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: { kind: request_aligned, axis: 0 } }\n            embeddings:",
        "            prompt: { dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: { kind: shared } }\n            embeddings:",
    );
    let document = document.replace(
        "            embeddings: { dtype: float32, rank: 3, shape: [batch, max_tiles, hidden], batch_layout: { kind: request_aligned, axis: 0 } }\n          outputs:\n            token:",
        "            embeddings: { dtype: float32, rank: 3, shape: [batch, max_tiles, hidden], batch_layout: { kind: shared } }\n          outputs:\n            token:",
    );
    let document = document.replace(
        "            token: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }\n    steps:",
        "            token: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: shared } }\n    steps:",
    );
    assert_reports(
        &document,
        "workflow component 'decoder' declares batch_capacity but no port declares a \
         request-scoped or token_packed batch_layout",
    );
}

#[test]
fn a_capacity_stacks_rows_on_the_axis_the_runtime_compacts() {
    let document = PADDED_VISION_ENCODER.replace(
        "        batch_capacity: { axis: 0, max_rows: 8, uniform_axes: [2, 3] }",
        "        batch_capacity: { axis: 0, max_rows: 8, uniform_axes: [2, 3] }\n        row_scope: { axis: 1, stateful: true }",
    );
    assert_reports(
        &document,
        "workflow component 'vision' declares batch_capacity axis 0 but row_scope axis 1",
    );
}

#[test]
fn a_capacity_denies_unknown_fields_and_defaults_the_optional_ones() {
    let capacity: ComponentBatchCapacity =
        serde_yaml::from_str("axis: 0\n").expect("a bare capacity parses");
    assert_eq!(capacity.max_rows, None);
    assert!(capacity.uniform_axes.is_empty());

    let round_trip = serde_yaml::to_string(&capacity).expect("capacity serializes");
    assert_eq!(round_trip.trim(), "axis: 0");
    assert_eq!(
        serde_yaml::from_str::<ComponentBatchCapacity>(&round_trip).expect("capacity round-trips"),
        capacity
    );

    let unknown = serde_yaml::from_str::<ComponentBatchCapacity>("axis: 0\nmax_items: 4\n")
        .expect_err("an unknown capacity field must be rejected");
    assert!(
        unknown.to_string().contains("max_items"),
        "the error must name the rejected field: {unknown}"
    );

    let zero = serde_yaml::from_str::<ComponentBatchCapacity>("axis: 0\nmax_rows: 0\n")
        .expect("zero parses so the validator can explain it");
    assert_eq!(zero.max_rows, Some(0));
}

#[test]
fn a_contract_without_a_pad_mask_stays_free_of_one() {
    let contract: TensorContract =
        serde_yaml::from_str("dtype: float32\nrank: 2\n").expect("contract parses");
    assert_eq!(contract.pad_mask, None);
    assert_eq!(contract.batch_layout, BatchLayout::Shared);

    let round_trip = serde_yaml::to_string(&contract).expect("contract serializes");
    assert!(
        !round_trip.contains("pad_mask"),
        "an absent pad_mask must not be spelled out: {round_trip}"
    );
    assert_eq!(
        serde_yaml::from_str::<TensorContract>(&round_trip).expect("contract round-trips"),
        contract
    );

    let masked: TensorContract =
        serde_yaml::from_str("dtype: float32\nrank: 2\npad_mask: valid_entries\n")
            .expect("masked contract parses");
    assert_eq!(masked.pad_mask.as_deref(), Some("valid_entries"));
}

#[test]
fn image_programs_can_emit_the_offsets_and_owner_of_a_packed_batch() {
    for content in ["item_offsets", "item_owner"] {
        let binding: ImageOutputBinding = serde_yaml::from_str(&format!(
            "source: packing\nname: image.{content}\ncontent: {content}\ndtype: int64\n"
        ))
        .expect("image output binding parses");
        assert_eq!(binding.content, content);
    }

    // The vocabulary is what a runtime dispatches on, so the published schema
    // has to offer the same two roles the parser accepts.
    let schema = inference_metadata_schema_json().expect("schema serializes");
    let schema: serde_json::Value = serde_json::from_str(&schema).expect("schema is JSON");
    let known = schema["$defs"]["ImageOutputContent"]["oneOf"][0]["enum"]
        .as_array()
        .expect("image output content vocabulary");
    for content in ["item_offsets", "item_owner"] {
        assert!(
            known.iter().any(|value| value == content),
            "the published vocabulary must offer {content}: {known:#?}"
        );
    }
}
