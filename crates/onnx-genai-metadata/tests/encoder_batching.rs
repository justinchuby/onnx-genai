//! Batching facts an encoder declares, and the contradictions they must reject.
//!
//! An encoder batches on its own terms: a request may carry zero, one, or many
//! images or clips, so rows either pad up to a common extent or pack together
//! with an offsets/owner pair that maps items back to rows. Both readings are
//! declared, never guessed, and both are only useful if the values they name
//! exist and are shaped the way the mapping requires — which is what these tests
//! pin. Nothing here is image-specific: a clip pads on a temporal axis and packs
//! at frame or clip granularity through the same contracts.

use onnx_genai_metadata::{
    BatchLayout, ComponentBatchCapacity, InferenceMetadata, TensorContract, VisionOutputBinding,
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

/// Every packing role a pixel program can emit, at each granularity a packed
/// batch has: the items a vision encoder consumes, the frames a clip
/// contributes, and the clips a request carries.
const PACKING_CONTENT_ROLES: [&str; 6] = [
    "item_offsets",
    "item_owner",
    "frame_offsets",
    "frame_owner",
    "clip_offsets",
    "clip_owner",
];

#[test]
fn pixel_programs_can_emit_the_offsets_and_owner_of_a_packed_batch() {
    for content in PACKING_CONTENT_ROLES {
        let binding: VisionOutputBinding = serde_yaml::from_str(&format!(
            "source: packing\nname: media.{content}\ncontent: {content}\ndtype: int64\n"
        ))
        .expect("pixel output binding parses");
        assert_eq!(binding.content, content);
    }

    // The vocabulary is what a runtime dispatches on, so the published schema
    // has to offer the same roles the parser accepts.
    let schema = inference_metadata_schema_json().expect("schema serializes");
    let schema: serde_json::Value = serde_json::from_str(&schema).expect("schema is JSON");
    let known = schema["$defs"]["VisionOutputContent"]["oneOf"][0]["enum"]
        .as_array()
        .expect("pixel output content vocabulary");
    for content in PACKING_CONTENT_ROLES {
        assert!(
            known.iter().any(|value| value == content),
            "the published vocabulary must offer {content}: {known:#?}"
        );
    }

    let ops = schema["$defs"]["VisionTransformOp"]["oneOf"][0]["enum"]
        .as_array()
        .expect("pixel transform operation vocabulary");
    for op in ["sample_frames", "pad_frames"] {
        assert!(
            ops.iter().any(|value| value == op),
            "the published vocabulary must offer {op}: {ops:#?}"
        );
    }
}

/// A clip encoder that pads in time: every request contributes one row of at
/// most `frames` frames, and `video.frame_mask` says which of those frames are
/// real rather than sampled-out or padded. The declaration is the same one a
/// still-image encoder makes — a temporal axis is an axis, and the mask that
/// describes it is an ordinary pad mask.
const PADDED_VIDEO_ENCODER: &str = r#"
schema_version: v1
preprocessing:
  video:
    transforms:
      - op: decode
        outputs: [video.decoded]
      - op: sample_frames
        fps: 2.0
        num_frames: 16
        frame_stride: 2
        inputs: [video.decoded]
        outputs: [video.sampled]
      - op: resize
        size: 224
        inputs: [video.sampled]
        outputs: [video.resized]
      - op: normalize
        mean: [0.5, 0.5, 0.5]
        std: [0.5, 0.5, 0.5]
        inputs: [video.resized]
        outputs: [video.normalized]
      - op: pad_frames
        target_length: 16
        pad_value: 0.0
        inputs: [video.normalized]
        outputs: [video.padded]
      - op: emit_validity_mask
        inputs: [video.padded]
        outputs: [video.frame_validity]
    outputs:
      - name: video.pixel_values
        source: video.padded
        content: pixels
        dtype: float32
        contract:
          dtype: float32
          rank: 5
          shape: [batch, frames, channels, height, width]
          batch_layout: { kind: request_aligned, axis: 0 }
          pad_mask: video.frame_mask
      - name: video.frame_mask
        source: video.frame_validity
        content: validity_mask
        dtype: bool
        contract: { dtype: bool, rank: 2, shape: [batch, frames], batch_layout: { kind: request_aligned, axis: 0 } }
pipeline:
  workflow:
    manifest:
      adapter_abis: { onnx-genai.video-preprocess: "1" }
      capabilities: [workflow_ssa, typed_emit]
    inputs:
      request.video:
        contract: { dtype: uint8, rank: 1, shape: [encoded_bytes] }
        role: { kind: runtime, version: "1.0", role: media }
        source: { kind: request }
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
      video_preprocess:
        implementation: { kind: adapter, abi: onnx-genai.video-preprocess, version: "1" }
        ports:
          inputs:
            encoded: { dtype: uint8, rank: 1, shape: [encoded_bytes] }
          outputs:
            pixel_values:
              dtype: float32
              rank: 5
              shape: [batch, frames, channels, height, width]
              batch_layout: { kind: request_aligned, axis: 0 }
              pad_mask: video.frame_mask
            frame_mask: { dtype: bool, rank: 2, shape: [batch, frames], batch_layout: { kind: request_aligned, axis: 0 } }
      video_encoder:
        implementation: { kind: onnx, artifact: video_encoder.onnx }
        batch_capacity: { axis: 0, max_rows: 4, uniform_axes: [3, 4] }
        ports:
          inputs:
            frames:
              dtype: float32
              rank: 5
              shape: [batch, frames, channels, height, width]
              batch_layout: { kind: request_aligned, axis: 0 }
              pad_mask: mask
            mask: { dtype: bool, rank: 2, shape: [batch, frames], batch_layout: { kind: request_aligned, axis: 0 } }
          outputs:
            embeddings: { dtype: float32, rank: 3, shape: [batch, video_tokens, hidden], batch_layout: { kind: request_aligned, axis: 0 } }
      decoder:
        implementation: { kind: onnx, artifact: decoder.onnx }
        ports:
          inputs:
            prompt: { dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: { kind: request_aligned, axis: 0 } }
            embeddings: { dtype: float32, rank: 3, shape: [batch, video_tokens, hidden], batch_layout: { kind: request_aligned, axis: 0 } }
          outputs:
            token: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }
    steps:
      - kind: invoke
        component: video_preprocess
        inputs: { encoded: request.video }
        outputs: { pixel_values: video.pixel_values, frame_mask: video.frame_mask }
      - kind: invoke
        component: video_encoder
        inputs: { frames: video.pixel_values, mask: video.frame_mask }
        outputs: { embeddings: video.embeddings }
      - kind: invoke
        component: decoder
        inputs: { prompt: prompt, embeddings: video.embeddings }
        outputs: { token: raw }
      - kind: emit
        value: raw
        output: tokens
        mode: replace
"#;

/// The same clips, packed instead of padded, and packed at two granularities at
/// once: the patches every clip contributes are concatenated on one item axis,
/// and the frames themselves are concatenated on another. Each packing names its
/// own offsets and owner map, so neither granularity is inferred from the other.
const PACKED_VIDEO_ENCODER: &str = r#"
schema_version: v1
preprocessing:
  video:
    transforms:
      - op: decode
        outputs: [video.decoded]
      - op: sample_frames
        fps: 1.0
        inputs: [video.decoded]
        outputs: [video.sampled]
      - op: patchify
        patch_size: 14
        temporal_patch_size: 2
        inputs: [video.sampled]
        outputs: [video.patches]
    outputs:
      - name: video.patches
        source: video.patches
        content: pixels
        dtype: float32
        contract:
          dtype: float32
          rank: 2
          shape: [items, patch_features]
          batch_layout: { kind: token_packed, offsets: video.item_offsets, owner: video.item_owner, axis: 0 }
      - name: video.item_offsets
        source: video.item_offsets
        content: item_offsets
        dtype: int64
        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
      - name: video.item_owner
        source: video.item_owner
        content: item_owner
        dtype: int64
        contract:
          dtype: int64
          rank: 1
          shape: [items]
          batch_layout: { kind: token_packed, offsets: video.item_offsets, owner: video.item_owner, axis: 0 }
      - name: video.frame_offsets
        source: video.frame_offsets
        content: frame_offsets
        dtype: int64
        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
      - name: video.frame_owner
        source: video.frame_owner
        content: frame_owner
        dtype: int64
        contract:
          dtype: int64
          rank: 1
          shape: [frames]
          batch_layout: { kind: token_packed, offsets: video.frame_offsets, owner: video.frame_owner, axis: 0 }
pipeline:
  workflow:
    manifest:
      adapter_abis: { onnx-genai.video-preprocess: "1" }
      capabilities: [workflow_ssa, typed_emit]
    inputs:
      request.video:
        contract: { dtype: uint8, rank: 1, shape: [encoded_bytes] }
        role: { kind: runtime, version: "1.0", role: media }
        source: { kind: request }
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
      video_preprocess:
        implementation: { kind: adapter, abi: onnx-genai.video-preprocess, version: "1" }
        ports:
          inputs:
            encoded: { dtype: uint8, rank: 1, shape: [encoded_bytes] }
          outputs:
            patches:
              dtype: float32
              rank: 2
              shape: [items, patch_features]
              batch_layout: { kind: token_packed, offsets: video.item_offsets, owner: video.item_owner, axis: 0 }
            item_offsets: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
            item_owner:
              dtype: int64
              rank: 1
              shape: [items]
              batch_layout: { kind: token_packed, offsets: video.item_offsets, owner: video.item_owner, axis: 0 }
            frame_offsets: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
            frame_owner:
              dtype: int64
              rank: 1
              shape: [frames]
              batch_layout: { kind: token_packed, offsets: video.frame_offsets, owner: video.frame_owner, axis: 0 }
      video_encoder:
        implementation: { kind: onnx, artifact: video_encoder.onnx }
        batch_capacity: { axis: 0, uniform_axes: [1] }
        ports:
          inputs:
            patches:
              dtype: float32
              rank: 2
              shape: [items, patch_features]
              batch_layout: { kind: token_packed, offsets: video.item_offsets, owner: video.item_owner, axis: 0 }
          outputs:
            features:
              dtype: float32
              rank: 2
              shape: [items, hidden]
              batch_layout: { kind: token_packed, offsets: video.item_offsets, owner: video.item_owner, axis: 0 }
      splice:
        implementation: { kind: onnx, artifact: splice.onnx }
        ports:
          inputs:
            prompt: { dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: { kind: request_aligned, axis: 0 } }
            features:
              dtype: float32
              rank: 2
              shape: [items, hidden]
              batch_layout: { kind: token_packed, offsets: video.item_offsets, owner: video.item_owner, axis: 0 }
          outputs:
            token: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }
    steps:
      - kind: invoke
        component: video_preprocess
        inputs: { encoded: request.video }
        outputs:
          patches: video.patches
          item_offsets: video.item_offsets
          item_owner: video.item_owner
          frame_offsets: video.frame_offsets
          frame_owner: video.frame_owner
      - kind: invoke
        component: video_encoder
        inputs: { patches: video.patches }
        outputs: { features: video.features }
      - kind: invoke
        component: splice
        inputs: { prompt: prompt, features: video.features }
        outputs: { token: raw }
      - kind: emit
        value: raw
        output: tokens
        mode: replace
"#;

#[test]
fn a_clip_pads_in_time_and_names_the_mask_that_says_which_frames_are_real() {
    let metadata = parse(PADDED_VIDEO_ENCODER);
    validate_metadata(&metadata).expect("temporally padded video encoder is valid");

    let program = metadata
        .preprocessing
        .as_ref()
        .and_then(|preprocessing| preprocessing.video.as_ref())
        .expect("fixture declares a video preprocessing program");
    let sample = &program.transforms[1];
    assert_eq!(sample.op, "sample_frames");
    assert_eq!(sample.fps, Some(2.0));
    assert_eq!(sample.num_frames, Some(16));
    assert_eq!(sample.frame_stride, Some(2));
    let pad = &program.transforms[4];
    assert_eq!(pad.op, "pad_frames");
    assert_eq!(pad.target_length, Some(16));

    // The temporal padding is only meaningful because the mask that describes
    // it is named by the padded value's own contract.
    let pixels = &program.outputs[0];
    assert_eq!(pixels.content, "pixels");
    assert_eq!(
        pixels
            .contract
            .as_ref()
            .and_then(|contract| contract.pad_mask.as_deref()),
        Some("video.frame_mask")
    );
    assert_eq!(
        metadata
            .pipeline
            .as_ref()
            .expect("fixture has a pipeline")
            .workflow
            .components["video_encoder"]
            .batch_capacity,
        Some(ComponentBatchCapacity {
            axis: 0,
            max_rows: Some(4),
            uniform_axes: vec![3, 4],
        })
    );
}

#[test]
fn a_video_program_is_a_pixel_program_with_a_temporal_axis() {
    let metadata = parse(PADDED_VIDEO_ENCODER);
    let preprocessing = metadata
        .preprocessing
        .as_ref()
        .expect("fixture declares preprocessing");

    // Stills and clips are declared by one program type; a package that
    // preprocesses only clips says nothing about images.
    assert!(preprocessing.image.is_none());
    let video = preprocessing
        .video
        .as_ref()
        .expect("fixture declares a video program");
    assert_eq!(video.outputs.len(), 2);

    // The spatial parameters a clip never sets stay absent rather than
    // acquiring a default a runtime would then have to second-guess.
    let sample = &video.transforms[1];
    assert_eq!(sample.tile_size, None);
    assert_eq!(sample.patch_size, None);
    // ...and the temporal parameters a still image never sets are absent there.
    let resize = &video.transforms[2];
    assert_eq!(resize.op, "resize");
    assert_eq!(resize.fps, None);
    assert_eq!(resize.num_frames, None);
    assert_eq!(resize.frame_stride, None);
}

#[test]
fn a_frame_mask_moves_with_the_rows_it_describes() {
    // A temporal mask that is shared cannot say which frames of *which row* are
    // real, so it cannot be permuted with those rows when the batch compacts.
    let document = PADDED_VIDEO_ENCODER.replace(
        "contract: { dtype: bool, rank: 2, shape: [batch, frames], batch_layout: { kind: request_aligned, axis: 0 } }\npipeline:",
        "contract: { dtype: bool, rank: 2, shape: [batch, frames], batch_layout: { kind: shared } }\npipeline:",
    );
    let document = document.replace(
        "            frame_mask: { dtype: bool, rank: 2, shape: [batch, frames], batch_layout: { kind: request_aligned, axis: 0 } }",
        "            frame_mask: { dtype: bool, rank: 2, shape: [batch, frames], batch_layout: { kind: shared } }",
    );
    assert_reports(
        &document,
        "pad_mask 'video.frame_mask' is shared with no request axis but the value it masks is \
         request-aligned on axis 0",
    );
}

#[test]
fn a_frame_mask_cannot_out_rank_the_clip_it_masks() {
    let document = PADDED_VIDEO_ENCODER.replace(
        "        contract: { dtype: bool, rank: 2, shape: [batch, frames], batch_layout: { kind: request_aligned, axis: 0 } }\npipeline:",
        "        contract: { dtype: bool, rank: 6, shape: [batch, frames, channels, height, width, extra], batch_layout: { kind: request_aligned, axis: 0 } }\npipeline:",
    );
    let document = document.replace(
        "            frame_mask: { dtype: bool, rank: 2, shape: [batch, frames], batch_layout: { kind: request_aligned, axis: 0 } }",
        "            frame_mask: { dtype: bool, rank: 6, shape: [batch, frames, channels, height, width, extra], batch_layout: { kind: request_aligned, axis: 0 } }",
    );
    assert_reports(
        &document,
        "pad_mask 'video.frame_mask' has rank 6, which cannot mark the valid entries of a rank-5 \
         value",
    );
}

#[test]
fn a_video_program_needs_its_adapter() {
    let document = PADDED_VIDEO_ENCODER.replace(
        "        implementation: { kind: adapter, abi: onnx-genai.video-preprocess, version: \"1\" }",
        "        implementation: { kind: adapter, abi: onnx-genai.image-preprocess, version: \"1\" }",
    );
    assert_reports(
        &document,
        "preprocessing.video requires exactly one workflow adapter component using \
         onnx-genai.video-preprocess@1, found 0",
    );
}

#[test]
fn a_packed_clip_maps_items_and_frames_back_to_rows_independently() {
    let metadata = parse(PACKED_VIDEO_ENCODER);
    validate_metadata(&metadata).expect("packed video encoder is valid");

    let workflow = &metadata
        .pipeline
        .as_ref()
        .expect("fixture has a pipeline")
        .workflow;
    let ports = &workflow.components["video_preprocess"].ports.outputs;
    assert_eq!(
        ports["patches"].batch_layout.packing(),
        Some(("video.item_offsets", "video.item_owner"))
    );
    assert_eq!(
        ports["frame_owner"].batch_layout.packing(),
        Some(("video.frame_offsets", "video.frame_owner"))
    );
    // Each granularity is its own packing: the frame map is not derived from the
    // item map, and the validator holds each to the same rule.
    assert_ne!(
        ports["patches"].batch_layout.packing(),
        ports["frame_owner"].batch_layout.packing()
    );
}

#[test]
fn a_frame_owner_map_belongs_to_the_frame_packing_it_names() {
    // Borrowing the item-level owner map for the frame packing confuses two
    // granularities: a map addressed by patch offsets cannot own frames.
    let document = PACKED_VIDEO_ENCODER.replace(
        "owner: video.frame_owner, axis: 0 }",
        "owner: video.item_owner, axis: 0 }",
    );
    assert_reports(
        &document,
        "token_packed owner map 'video.item_owner' is packed against offsets \
         'video.item_offsets', not the 'video.frame_offsets' this value is packed against",
    );
}
