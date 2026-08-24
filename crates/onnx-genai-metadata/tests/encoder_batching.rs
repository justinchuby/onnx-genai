//! Batching facts an encoder declares, and the contradictions they must reject.
//!
//! Grouping faces three independent kinds of raggedness, and each is declared on
//! its own dimension of one contract. How many items a request owns is an
//! ownership level. How far two items differ along a dimension is padding, whose
//! truth is one length per enclosing position. An item that is itself a group is
//! a second ownership level over the same physical packed axis. Nothing here is
//! image-specific: a clip pads on a temporal dimension and packs at frame or
//! clip granularity through the same contracts a still image uses.

use onnx_genai_metadata::{
    BatchLayout, ComponentBatchCapacity, InferenceMetadata, PackedExtent, TensorContract,
    VisionOutputBinding, inference_metadata_schema_json, validate_metadata,
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
/// `max_tiles` tiles, and `tile_lengths` says how many of those tiles are real.
/// The encoder groups such rows only when they already agree on the two spatial
/// extents its artifact was built for, and bounds what one call materializes on
/// the dimension the rows differ in.
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
          padding: [{ dimension: max_tiles, valid_lengths: tile_lengths }]
        role: { kind: opaque }
        source: { kind: application, name: pixel_values }
      tile_lengths:
        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: tile_lengths }
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
        batch_capacity:
          uniform_dimensions: [height, width]
          budgets:
            - { dimensions: [max_tiles], max_total: 64 }
        ports:
          inputs:
            pixels:
              dtype: float32
              rank: 4
              shape: [batch, max_tiles, height, width]
              batch_layout: { kind: request_aligned, axis: 0 }
              padding: [{ dimension: max_tiles, valid_lengths: lengths }]
            lengths: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: shared } }
          outputs:
            features: { dtype: float32, rank: 3, shape: [batch, tiles, hidden], batch_layout: { kind: request_aligned, axis: 0 } }
      splice:
        implementation: { kind: onnx, artifact: splice.onnx }
        ports:
          inputs:
            prompt: { dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: { kind: request_aligned, axis: 0 } }
            features: { dtype: float32, rank: 3, shape: [batch, tiles, hidden], batch_layout: { kind: request_aligned, axis: 0 } }
          outputs:
            token: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }
    steps:
      - kind: invoke
        component: vision
        inputs: { pixels: pixel_values, lengths: tile_lengths }
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

/// The same encoder, packing instead of padding: the images of every request are
/// concatenated onto one item axis and one ownership level maps each item back
/// to the row that asked for it.
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
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: image_offsets, owner: image_owner }
        role: { kind: opaque }
        source: { kind: application, name: image_pixels }
      image_offsets:
        contract: { dtype: int64, rank: 1, shape: [rows_plus_one], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: image_offsets }
      image_owner:
        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }
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
        batch_capacity:
          uniform_dimensions: [channels, height, width]
          budgets:
            - { dimensions: [items], max_total: 32 }
        ports:
          inputs:
            pixels:
              dtype: float32
              rank: 4
              shape: [items, channels, height, width]
              batch_layout:
                kind: token_packed
                axis: 0
                levels:
                  - { offsets: offsets, owner: owner }
            offsets: { dtype: int64, rank: 1, shape: [rows_plus_one], batch_layout: { kind: shared } }
            owner: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }
          outputs:
            features:
              dtype: float32
              rank: 2
              shape: [items, hidden]
              batch_layout:
                kind: token_packed
                axis: 0
                levels:
                  - { offsets: offsets, owner: owner }
                packed_extent: preserved
      splice:
        implementation: { kind: onnx, artifact: splice.onnx }
        ports:
          inputs:
            prompt: { dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: { kind: request_aligned, axis: 0 } }
            features:
              dtype: float32
              rank: 2
              shape: [items, hidden]
              batch_layout:
                kind: token_packed
                axis: 0
                levels:
                  - { offsets: image_offsets, owner: image_owner }
          outputs:
            token: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }
    steps:
      - kind: invoke
        component: vision
        inputs: { pixels: image_pixels, offsets: image_offsets, owner: image_owner }
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
    // Absence is the statement, so nothing has to be written to keep every
    // package that predates this surface valid.
    let document = PADDED_VISION_ENCODER
        .replace(
            "        batch_capacity:\n          uniform_dimensions: [height, width]\n          budgets:\n            - { dimensions: [max_tiles], max_total: 64 }\n",
            "",
        );
    let metadata = parse(&document);
    validate_metadata(&metadata).expect("a component with no capacity is valid");
    let workflow = &metadata.pipeline.expect("pipeline").workflow;
    assert!(
        workflow.components["vision"].batch_capacity.is_none(),
        "an undeclared capacity stays undeclared"
    );

    // And it does not round-trip into existence.
    let capacity: Option<ComponentBatchCapacity> =
        serde_yaml::from_str("null").expect("an absent capacity parses");
    assert!(capacity.is_none());
}

#[test]
fn a_padded_encoder_declares_its_capacity_and_its_validity() {
    let metadata = parse(PADDED_VISION_ENCODER);
    validate_metadata(&metadata).expect("padded vision encoder is valid");
    let workflow = &metadata.pipeline.expect("pipeline").workflow;
    let capacity = workflow.components["vision"]
        .batch_capacity
        .as_ref()
        .expect("fixture declares a capacity");
    assert_eq!(capacity.uniform_dimensions, ["height", "width"]);
    assert_eq!(capacity.budgets.len(), 1);
    assert_eq!(capacity.budgets[0].dimensions, ["max_tiles"]);
    assert_eq!(capacity.budgets[0].max_total, 64);

    let padding = &workflow.inputs["pixel_values"].contract.padding;
    assert_eq!(padding.len(), 1);
    assert_eq!(padding[0].dimension, "max_tiles");
    assert_eq!(padding[0].valid_lengths, "tile_lengths");
}

#[test]
fn a_packed_encoder_result_maps_items_back_to_rows() {
    let metadata = parse(PACKED_VISION_ENCODER);
    validate_metadata(&metadata).expect("packed vision encoder is valid");
    let workflow = &metadata.pipeline.expect("pipeline").workflow;
    let layout = &workflow.inputs["image_pixels"].contract.batch_layout;
    match layout {
        BatchLayout::TokenPacked {
            axis,
            levels,
            packed_extent,
        } => {
            assert_eq!(*axis, 0);
            assert_eq!(levels.len(), 1, "one level owns items straight into rows");
            assert_eq!(levels[0].offsets, "image_offsets");
            assert_eq!(levels[0].owner, "image_owner");
            assert_eq!(
                *packed_extent, None,
                "a value a component consumes carries the extent its caller assembled"
            );
        }
        other => panic!("fixture declares a packed layout: {other:?}"),
    }
    assert_eq!(
        workflow.components["vision"].ports.outputs["features"]
            .batch_layout
            .packed_extent(),
        Some(PackedExtent::Preserved)
    );
}

#[test]
fn packed_offsets_must_name_a_declared_value() {
    let document = PACKED_VISION_ENCODER.replace("offsets: image_offsets", "offsets: nowhere");
    assert_reports(
        &document,
        "names 'nowhere' as its level 0 offsets, which is not a declared value or port in that \
         scope",
    );
}

#[test]
fn a_packed_owner_map_must_name_a_declared_value() {
    let document = PACKED_VISION_ENCODER.replace("owner: image_owner", "owner: nowhere");
    assert_reports(
        &document,
        "names 'nowhere' as its level 0 owner map, which is not a declared value or port in that \
         scope",
    );
}

#[test]
fn packed_companions_are_shared_rank_one_integers() {
    let document = PACKED_VISION_ENCODER.replace(
        "      image_offsets:\n        contract: { dtype: int64, rank: 1, shape: [rows_plus_one], batch_layout: { kind: shared } }",
        "      image_offsets:\n        contract: { dtype: float32, rank: 2, shape: [rows_plus_one, pad], batch_layout: { kind: shared } }",
    );
    let reported = errors(&document);
    assert!(
        reported
            .iter()
            .any(|error| error
                .contains("level 0 offsets 'image_offsets' is float32 but must be int64")),
        "an index vector must be able to hold an index: {reported:#?}"
    );
    assert!(
        reported
            .iter()
            .any(|error| error
                .contains("level 0 offsets 'image_offsets' has rank 2 but must be rank 1")),
        "a companion carries one entry per unit and nothing else: {reported:#?}"
    );
}

#[test]
fn an_ownership_companion_is_rebuilt_rather_than_permuted() {
    // A prefix sum is not permutation-followable: permuting rows does not
    // permute it, it invalidates it. Calling it request-aligned would invite a
    // gather that silently produces nonsense.
    let document = PACKED_VISION_ENCODER.replace(
        "        contract: { dtype: int64, rank: 1, shape: [rows_plus_one], batch_layout: { kind: shared } }",
        "        contract: { dtype: int64, rank: 1, shape: [rows_plus_one], batch_layout: { kind: request_aligned, axis: 0 } }",
    );
    assert_reports(
        &document,
        "level 0 offsets 'image_offsets' declares request_aligned but must declare shared; an \
         exclusive prefix sum is not permutation-followable",
    );
}

#[test]
fn packed_offsets_and_owner_cannot_be_one_value() {
    let document = PACKED_VISION_ENCODER.replace("owner: image_owner", "owner: image_offsets");
    assert_reports(
        &document,
        "names 'image_offsets' as both its level 0 offsets and its level 0 owner; the two are \
         different vectors of different lengths, so one value cannot be both",
    );
}

#[test]
fn a_packed_value_cannot_pack_on_an_axis_it_does_not_have() {
    let document = PACKED_VISION_ENCODER.replace(
        "            kind: token_packed\n            axis: 0\n",
        "            kind: token_packed\n            axis: 7\n",
    );
    assert_reports(&document, "packs items along axis 7, outside its rank 4");
}

#[test]
fn packed_items_sit_on_the_outermost_axis() {
    // Only an outermost packed axis makes each request's span a contiguous
    // range; an inner one turns every split into a device-side gather.
    let document = PACKED_VISION_ENCODER.replace(
        "            kind: token_packed\n            axis: 0\n",
        "            kind: token_packed\n            axis: 1\n",
    );
    assert_reports(
        &document,
        "a packed axis must be axis 0, because only then is each request's span a contiguous \
         range that can be aliased rather than gathered",
    );
}

#[test]
fn an_owner_map_is_as_long_as_the_packing_it_describes() {
    let document = PACKED_VISION_ENCODER.replace(
        "      image_owner:\n        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }",
        "      image_owner:\n        contract: { dtype: int64, rank: 1, shape: [other_items], batch_layout: { kind: shared } }",
    );
    assert_reports(
        &document,
        "packs 'items' items on axis 0 but its level 0 owner map 'image_owner' is 'other_items' \
         long; the owner map has exactly one entry per packed item",
    );
}

#[test]
fn offsets_are_one_longer_than_the_map_they_index() {
    let document = PACKED_VISION_ENCODER.replace(
        "      image_offsets:\n        contract: { dtype: int64, rank: 1, shape: [rows_plus_one], batch_layout: { kind: shared } }",
        "      image_offsets:\n        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }",
    );
    assert_reports(
        &document,
        "declares level 0 offsets 'image_offsets' and owner map 'image_owner' with the same \
         extent 'items'; offsets carries one entry per parent plus a final total",
    );
}

#[test]
fn a_packed_value_declares_at_least_one_ownership_level() {
    let document = PACKED_VISION_ENCODER.replace(
        "            levels:\n              - { offsets: image_offsets, owner: image_owner }\n",
        "            levels: []\n",
    );
    assert_reports(
        &document,
        "packs items but declares no ownership levels; a packed run is only attributable to \
         requests through at least one offsets/owner pair",
    );
}

#[test]
fn an_ownership_companion_carries_no_padding_of_its_own() {
    let document = PACKED_VISION_ENCODER.replace(
        "      image_owner:\n        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }",
        "      image_owner:\n        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared }, padding: [{ dimension: items, valid_lengths: image_offsets }] }",
    );
    assert_reports(
        &document,
        "level 0 owner map 'image_owner' declares padding of its own; a companion has exactly one \
         entry per unit, so there is nothing in it to pad",
    );
}

#[test]
fn a_packed_axis_carries_no_padding() {
    // Packed items are contiguous by construction. Padding the same dimension
    // would leave two contradictory accounts of where a unit's entries end.
    let document = PACKED_VISION_ENCODER.replace(
        "              - { offsets: image_offsets, owner: image_owner }\n        role: { kind: opaque }\n        source: { kind: application, name: image_pixels }",
        "              - { offsets: image_offsets, owner: image_owner }\n          padding: [{ dimension: items, valid_lengths: image_owner }]\n        role: { kind: opaque }\n        source: { kind: application, name: image_pixels }",
    );
    assert_reports(
        &document,
        "declares padding on dimension 'items', which is the axis it packs items along; packed \
         items are contiguous and carry no padding",
    );
}

#[test]
fn two_values_that_share_offsets_share_the_grouping() {
    // One offsets vector is a complete account of how many units each parent
    // owns, so two values naming it at the same level are claiming one grouping.
    let document = PACKED_VISION_ENCODER.replace(
        "      tokens:\n        contract: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }\n        role: tokens",
        "      tokens:\n        contract: { dtype: int64, rank: 2, shape: [items, generated], batch_layout: { kind: token_packed, axis: 0, levels: [{ offsets: image_offsets, owner: prompt }] } }\n        role: tokens",
    );
    assert_reports(
        &document,
        "pairs level 0 offsets 'image_offsets' with owner map 'prompt', but workflow input \
         'image_pixels' pairs the same offsets with 'image_owner'",
    );
}

#[test]
fn a_valid_lengths_companion_must_name_a_declared_value() {
    let document = PADDED_VISION_ENCODER.replace(
        "valid_lengths: tile_lengths }]",
        "valid_lengths: nowhere }]",
    );
    assert_reports(
        &document,
        "names 'nowhere' as the valid_lengths of dimension 'max_tiles', which is not a declared \
         value or port in that scope",
    );
}

#[test]
fn a_valid_lengths_companion_counts_with_integers() {
    let document = PADDED_VISION_ENCODER.replace(
        "      tile_lengths:\n        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: shared } }",
        "      tile_lengths:\n        contract: { dtype: bool, rank: 1, shape: [batch], batch_layout: { kind: shared } }",
    );
    assert_reports(
        &document,
        "valid_lengths 'tile_lengths' is bool but must be int64; it counts real entries of \
         dimension 'max_tiles'",
    );
}

#[test]
fn a_valid_lengths_companion_is_read_on_the_host_not_gathered() {
    let document = PADDED_VISION_ENCODER.replace(
        "      tile_lengths:\n        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: shared } }",
        "      tile_lengths:\n        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }",
    );
    assert_reports(
        &document,
        "valid_lengths 'tile_lengths' declares request_aligned but must declare shared; it has \
         one entry per position of the axes outer to 'max_tiles'",
    );
}

#[test]
fn padding_names_a_dimension_the_value_declares() {
    let document = PADDED_VISION_ENCODER.replace(
        "padding: [{ dimension: max_tiles,",
        "padding: [{ dimension: depth,",
    );
    assert_reports(
        &document,
        "declares padding on dimension 'depth', which is not a shape symbol of the value it pads",
    );
}

#[test]
fn a_valid_lengths_companion_stops_where_the_dimension_it_bounds_begins() {
    // A length applies to a whole slice, so the axes inner to the padded one are
    // not indexed and the companion's rank is the padded axis itself.
    let document = PADDED_VISION_ENCODER.replace(
        "      tile_lengths:\n        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: shared } }",
        "      tile_lengths:\n        contract: { dtype: int64, rank: 2, shape: [batch, max_tiles], batch_layout: { kind: shared } }",
    );
    assert_reports(
        &document,
        "valid_lengths 'tile_lengths' has rank 2 but dimension 'max_tiles' is axis 1, so it must \
         have rank 1: one entry per position of the axes outer to 'max_tiles'",
    );
}

#[test]
fn a_valid_lengths_companion_matches_the_axes_outer_to_the_padded_one() {
    let document = PADDED_VISION_ENCODER.replace(
        "      tile_lengths:\n        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: shared } }",
        "      tile_lengths:\n        contract: { dtype: int64, rank: 1, shape: [rows], batch_layout: { kind: shared } }",
    );
    assert_reports(
        &document,
        "valid_lengths 'tile_lengths' declares 'rows' on axis 0 but the value it bounds declares \
         'batch' there",
    );
}

#[test]
fn one_dimension_is_padded_once() {
    let document = PADDED_VISION_ENCODER.replace(
        "          padding: [{ dimension: max_tiles, valid_lengths: tile_lengths }]",
        "          padding:\n            - { dimension: max_tiles, valid_lengths: tile_lengths }\n            - { dimension: max_tiles, valid_lengths: prompt }",
    );
    assert_reports(
        &document,
        "declares padding on dimension 'max_tiles' more than once; one dimension has one padded \
         extent",
    );
}

#[test]
fn padding_never_covers_the_axis_rows_stack_on() {
    // A compacted batch has no padding rows: the runtime drops a finished row
    // rather than blanking it.
    let document = PADDED_VISION_ENCODER.replace(
        "padding: [{ dimension: max_tiles, valid_lengths: tile_lengths }]",
        "padding: [{ dimension: batch, valid_lengths: tile_lengths }]",
    );
    assert_reports(
        &document,
        "declares padding on dimension 'batch', which is the axis its request rows stack along",
    );
}

#[test]
fn a_padded_value_is_not_its_own_length_vector() {
    let document = PADDED_VISION_ENCODER.replace(
        "              padding: [{ dimension: max_tiles, valid_lengths: lengths }]",
        "              padding: [{ dimension: max_tiles, valid_lengths: pixels }]",
    );
    assert_reports(
        &document,
        "names itself as the valid_lengths of its own dimension 'max_tiles'",
    );
}

#[test]
fn a_packed_output_says_where_its_extent_came_from() {
    // An output of the same rank and symbols as its input may be a per-item
    // transform or a token merger, and the two split at different boundaries.
    let document = PACKED_VISION_ENCODER.replace("                packed_extent: preserved\n", "");
    assert_reports(
        &document,
        "packs items but declares no packed_extent; an output either preserves an input's units \
         one for one or produces its own",
    );
}

#[test]
fn a_produced_extent_is_described_by_the_components_own_outputs() {
    // Reusing an input's offsets for an extent the graph decided would describe
    // a length the output does not have, and the split would land between items.
    let document = NESTED_VIDEO_ENCODER.replace(
        "                  - { offsets: clip_offsets, owner: clip_owner }\n                packed_extent: preserved",
        "                  - { offsets: clip_offsets, owner: clip_owner }\n                packed_extent: produced",
    );
    assert_reports(
        &document,
        "declares packed_extent produced but its level 0 offsets 'clip_offsets' is not an output \
         port of the same component; an extent the graph decides is described by companions the \
         graph emits",
    );
}

#[test]
fn a_preserved_extent_reuses_the_companions_that_already_described_it() {
    let document = NESTED_VIDEO_ENCODER.replace(
        "                  - { offsets: media_token_offsets, owner: media_token_owner }\n                  - { offsets: clip_offsets, owner: clip_owner }\n                packed_extent: produced",
        "                  - { offsets: media_token_offsets, owner: media_token_owner }\n                  - { offsets: clip_offsets, owner: clip_owner }\n                packed_extent: preserved",
    );
    assert_reports(
        &document,
        "declares packed_extent preserved but its level 0 offsets 'media_token_offsets' is an \
         output port of the same component; preserving an extent means reusing the companions \
         that already described it",
    );
}

#[test]
fn a_consumed_value_does_not_state_where_its_extent_came_from() {
    let document = PACKED_VISION_ENCODER.replace(
        "                levels:\n                  - { offsets: offsets, owner: owner }\n            offsets:",
        "                levels:\n                  - { offsets: offsets, owner: owner }\n                packed_extent: preserved\n            offsets:",
    );
    assert_reports(
        &document,
        "declares packed_extent preserved; the extent of a value a component consumes is the one \
         its caller assembled",
    );
}

#[test]
fn uniform_dimensions_are_distinct_declared_symbols() {
    let duplicated = PACKED_VISION_ENCODER.replace(
        "uniform_dimensions: [channels, height, width]",
        "uniform_dimensions: [channels, channels, height, width]",
    );
    assert_reports(
        &duplicated,
        "lists uniform dimension 'channels' twice; the list states which extents must agree",
    );

    let unknown = PACKED_VISION_ENCODER.replace(
        "uniform_dimensions: [channels, height, width]",
        "uniform_dimensions: [channels, height, width, depth]",
    );
    assert_reports(
        &unknown,
        "requires uniform dimension 'depth', which no port of the component declares; declared \
         symbols are",
    );
}

#[test]
fn a_symbol_is_pinned_or_budgeted_but_never_both() {
    let document = PACKED_VISION_ENCODER.replace(
        "            - { dimensions: [items], max_total: 32 }",
        "            - { dimensions: [items], max_total: 32 }\n            - { dimensions: [channels], max_total: 3 }",
    );
    assert_reports(
        &document,
        "both pins 'channels' across the group and budgets it; a pinned dimension has one extent \
         for every item",
    );
}

#[test]
fn a_budget_names_symbols_the_component_declares() {
    let document = PACKED_VISION_ENCODER.replace(
        "            - { dimensions: [items], max_total: 32 }",
        "            - { dimensions: [items], max_total: 32 }\n            - { dimensions: [voxels], max_total: 8 }",
    );
    assert_reports(
        &document,
        "budgets 'voxels', which no port of the component declares; declared symbols are",
    );
}

#[test]
fn a_footprint_is_bounded_once() {
    let document = PACKED_VISION_ENCODER.replace(
        "            - { dimensions: [items], max_total: 32 }",
        "            - { dimensions: [items], max_total: 32 }\n            - { dimensions: [items], max_total: 16 }",
    );
    assert_reports(
        &document,
        "budgets ['items'] more than once; two bounds on one footprint is two numbers for one \
         fact",
    );
}

#[test]
fn a_composed_budget_multiplies_distinct_extents() {
    let document = PACKED_VISION_ENCODER.replace(
        "            - { dimensions: [items], max_total: 32 }",
        "            - { dimensions: [items, items], max_total: 32 }",
    );
    assert_reports(
        &document,
        "budgets ['items', 'items'], which names 'items' twice; a composed budget multiplies \
         distinct extents",
    );
}

#[test]
fn a_budget_leaves_room_for_one_item() {
    let document = PACKED_VISION_ENCODER.replace(
        "            - { dimensions: [items], max_total: 32 }",
        "            - { dimensions: [items], max_total: 0 }",
    );
    assert_reports(
        &document,
        "budgets ['items'] at 0; every budget is an upper bound on an assembled group, and a \
         bound of zero forbids the single-item invocation",
    );
}

#[test]
fn every_input_ownership_level_is_budgeted() {
    // A level's units are exactly what a scheduler chooses, so an unbudgeted
    // level is unbounded in the one quantity the scheduler controls.
    let document = PACKED_VISION_ENCODER.replace(
        "          budgets:\n            - { dimensions: [items], max_total: 32 }\n",
        "",
    );
    assert_reports(
        &document,
        "declares no budget for 'items', the units input 'pixels' packs at ownership level 0",
    );
}

#[test]
fn a_free_dimension_says_how_it_is_reconciled() {
    let document = PACKED_VISION_ENCODER.replace(
        "uniform_dimensions: [channels, height, width]",
        "uniform_dimensions: [channels, height]",
    );
    assert_reports(
        &document,
        "leaves 'width' free on input 'pixels' axis 3 but declares neither a padding entry on it \
         nor a packed axis that consumes it",
    );
}

#[test]
fn a_packed_extent_is_never_a_literal() {
    // A packed extent is the sum of the group's items, so pinning it would be a
    // shape the group has to change and the artifact has forbidden.
    let document = PACKED_VISION_ENCODER.replace(
        "              shape: [items, channels, height, width]\n              batch_layout:\n                kind: token_packed",
        "              shape: [4, channels, height, width]\n              batch_layout:\n                kind: token_packed",
    );
    assert_reports(
        &document,
        "packs items along axis 0 but fixes that axis at 4; a packed extent is the sum of the \
         group's items and cannot be a literal",
    );
}

#[test]
fn a_packed_item_axis_is_never_read_as_a_row_scope_axis() {
    // One request contributing eight images makes items and rows different
    // numbers, so per-row state selected by an item position would address the
    // wrong entries entirely.
    let document = PACKED_VISION_ENCODER.replace(
        "        batch_capacity:\n          uniform_dimensions: [channels, height, width]",
        "        row_scope: { axis: 0, stateful: true }\n        batch_capacity:\n          uniform_dimensions: [channels, height, width]",
    );
    assert_reports(
        &document,
        "declares row_scope on axis 0, which port 'pixels' packs items along; items are not rows",
    );
}

#[test]
fn a_row_scope_axis_is_an_axis_rows_actually_stack_on() {
    let document = PADDED_VISION_ENCODER.replace(
        "        batch_capacity:\n          uniform_dimensions: [height, width]",
        "        row_scope: { axis: 2, stateful: true }\n        batch_capacity:\n          uniform_dimensions: [height, width]",
    );
    assert_reports(
        &document,
        "declares row_scope on axis 2, which no port of the component declares as its request \
         axis",
    );
}

#[test]
fn a_capacity_denies_unknown_fields_and_defaults_the_optional_ones() {
    let capacity: ComponentBatchCapacity =
        serde_yaml::from_str("budgets: [{ dimensions: [items], max_total: 8 }]\n")
            .expect("a capacity with only a budget parses");
    assert!(capacity.uniform_dimensions.is_empty());

    let rejected = serde_yaml::from_str::<ComponentBatchCapacity>("max_items: 8\n")
        .expect_err("an unknown field must be rejected");
    assert!(
        rejected.to_string().contains("max_items"),
        "the error must name the field it does not know: {rejected}"
    );

    // Round-tripping keeps both lists, and drops them again when they are empty.
    let stated = ComponentBatchCapacity {
        uniform_dimensions: vec!["features".to_string()],
        budgets: capacity.budgets.clone(),
    };
    let text = serde_yaml::to_string(&stated).expect("a capacity serializes");
    assert_eq!(
        serde_yaml::from_str::<ComponentBatchCapacity>(&text).expect("it parses back"),
        stated
    );
    let empty = serde_yaml::to_string(&ComponentBatchCapacity::default()).expect("serializes");
    assert!(
        !empty.contains("uniform_dimensions") && !empty.contains("budgets"),
        "an empty capacity states nothing: {empty}"
    );
}

#[test]
fn a_contract_without_padding_stays_free_of_one() {
    let contract: TensorContract =
        serde_yaml::from_str("dtype: float32\nrank: 2\nshape: [batch, hidden]\n")
            .expect("a contract with no padding parses");
    assert!(contract.padding.is_empty(), "absence means no padding");
    let text = serde_yaml::to_string(&contract).expect("a contract serializes");
    assert!(
        !text.contains("padding"),
        "an unpadded contract says nothing about padding: {text}"
    );
}

#[test]
fn the_batching_fields_are_absent_by_default_and_closed_to_older_readers() {
    // Every struct on this path denies unknown fields, so an older reader
    // rejects a document that uses one of these rather than ignoring it. What
    // keeps existing packages working is that all of them are absent by
    // default, not any tolerance in the reader.
    let schema = inference_metadata_schema_json().expect("schema serializes");
    let schema: serde_json::Value = serde_json::from_str(&schema).expect("schema is JSON");

    let contract = &schema["$defs"]["TensorContract"];
    assert_eq!(contract["additionalProperties"], serde_json::json!(false));
    let required = contract["required"].as_array().expect("required fields");
    assert!(
        !required.iter().any(|field| field == "padding"),
        "padding must stay optional: {required:#?}"
    );

    let component = &schema["$defs"]["WorkflowComponent"];
    let required = component["required"].as_array().expect("required fields");
    assert!(
        !required.iter().any(|field| field == "batch_capacity"),
        "batch_capacity must stay optional: {required:#?}"
    );

    for definition in [
        "ComponentBatchCapacity",
        "CapacityBudget",
        "PaddedDimension",
    ] {
        assert_eq!(
            schema["$defs"][definition]["additionalProperties"],
            serde_json::json!(false),
            "{definition} must reject fields it does not know"
        );
    }
}

/// Two content roles cover every ownership level: which level a value serves is
/// stated by the chain that references it, never by its role.
const PACKING_CONTENT_ROLES: [&str; 3] = ["pack_offsets", "pack_owner", "valid_lengths"];

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
    // has to offer the same roles the parser accepts — and the audio side, whose
    // levels mean windows rather than clips, offers the same two.
    let schema = inference_metadata_schema_json().expect("schema serializes");
    let schema: serde_json::Value = serde_json::from_str(&schema).expect("schema is JSON");
    for vocabulary in ["VisionOutputContent", "AudioOutputContent"] {
        let known = schema["$defs"][vocabulary]["oneOf"][0]["enum"]
            .as_array()
            .unwrap_or_else(|| panic!("{vocabulary} vocabulary"));
        for content in PACKING_CONTENT_ROLES {
            assert!(
                known.iter().any(|value| value == content),
                "{vocabulary} must offer {content}: {known:#?}"
            );
        }
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
/// most `frames` frames, and `video.frame_lengths` says how many of them are
/// real rather than sampled-out or padded. The declaration is the one a
/// still-image encoder makes — a temporal dimension is a dimension — and its
/// capacity pins the spatial extents in the same list that would pin a temporal
/// one.
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
          padding: [{ dimension: frames, valid_lengths: video.frame_lengths }]
      - name: video.frame_lengths
        source: video.frame_counts
        content: valid_lengths
        dtype: int64
        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: shared } }
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
              padding: [{ dimension: frames, valid_lengths: video.frame_lengths }]
            frame_lengths: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: shared } }
      video_encoder:
        implementation: { kind: onnx, artifact: video_encoder.onnx }
        batch_capacity:
          uniform_dimensions: [channels, height, width]
          budgets:
            - { dimensions: [frames], max_total: 64 }
        ports:
          inputs:
            frames:
              dtype: float32
              rank: 5
              shape: [batch, frames, channels, height, width]
              batch_layout: { kind: request_aligned, axis: 0 }
              padding: [{ dimension: frames, valid_lengths: lengths }]
            lengths: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: shared } }
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
        outputs: { pixel_values: video.pixel_values, frame_lengths: video.frame_lengths }
      - kind: invoke
        component: video_encoder
        inputs: { frames: video.pixel_values, lengths: video.frame_lengths }
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

#[test]
fn a_clip_pads_in_time_and_names_the_companion_that_says_which_frames_are_real() {
    let metadata = parse(PADDED_VIDEO_ENCODER);
    validate_metadata(&metadata).expect("padded video encoder is valid");
    let program = metadata
        .preprocessing
        .as_ref()
        .expect("fixture declares preprocessing")
        .video
        .as_ref()
        .expect("fixture declares a video program");
    let pixels = program
        .outputs
        .iter()
        .find(|binding| binding.content == "pixels")
        .expect("the program emits pixels");
    let padding = &pixels
        .contract
        .as_ref()
        .expect("the binding states a contract")
        .padding;
    assert_eq!(padding.len(), 1);
    assert_eq!(
        padding[0].dimension, "frames",
        "a temporal extent is padded through the same contract a spatial one is"
    );
}

#[test]
fn a_video_program_is_a_pixel_program_with_a_temporal_axis() {
    let metadata = parse(PADDED_VIDEO_ENCODER);
    let video = metadata
        .preprocessing
        .as_ref()
        .expect("fixture declares preprocessing")
        .video
        .as_ref()
        .expect("fixture declares a video program");
    let ops = video
        .transforms
        .iter()
        .map(|transform| transform.op.as_str())
        .collect::<Vec<_>>();
    assert!(ops.contains(&"sample_frames"));
    assert!(ops.contains(&"pad_frames"));
}

#[test]
fn a_video_program_needs_its_adapter() {
    let document = PADDED_VIDEO_ENCODER.replace(
        "      adapter_abis: { onnx-genai.video-preprocess: \"1\" }\n",
        "",
    );
    assert!(
        !errors(&document).is_empty(),
        "a video program with no declared adapter must be rejected"
    );
}

#[test]
fn uniformity_is_required_of_temporal_and_spatial_dimensions_alike() {
    // A frame-count-pinned encoder lists its temporal symbol in the same list a
    // resolution-pinned one lists its spatial ones. The rule does not know which
    // is which.
    let document = PADDED_VIDEO_ENCODER
        .replace(
            "          uniform_dimensions: [channels, height, width]\n          budgets:\n            - { dimensions: [frames], max_total: 64 }\n",
            "          uniform_dimensions: [frames, channels, height, width]\n",
        )
        .replace(
            "              padding: [{ dimension: frames, valid_lengths: lengths }]\n            lengths: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: shared } }\n",
            "",
        )
        .replace(
            "        inputs: { frames: video.pixel_values, lengths: video.frame_lengths }",
            "        inputs: { frames: video.pixel_values }",
        );
    let metadata = parse(&document);
    validate_metadata(&metadata).expect("pinning the temporal extent is valid");
    let capacity = metadata.pipeline.expect("pipeline").workflow.components["video_encoder"]
        .batch_capacity
        .clone()
        .expect("fixture declares a capacity");
    assert_eq!(
        capacity.uniform_dimensions,
        ["frames", "channels", "height", "width"]
    );
    assert!(
        capacity.budgets.is_empty(),
        "a pinned encoder with nothing free needs no footprint bound"
    );
}

/// The design's worked case, and the deepest chain the schema states: frames are
/// flattened across every clip of every request onto one packed axis, the frame
/// level maps positions to clips, the clip level maps clips to request rows, and
/// each frame's patches pad up to a common extent one length per frame bounds.
///
/// Three kinds of raggedness, three dimensions of one contract, no two competing.
const NESTED_VIDEO_ENCODER: &str = r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, typed_emit]
    inputs:
      pixel_values:
        contract:
          dtype: float32
          rank: 3
          shape: [frames, patches, features]
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: frame_offsets, owner: frame_owner }
              - { offsets: clip_offsets, owner: clip_owner }
          padding: [{ dimension: patches, valid_lengths: patch_lengths }]
        role: { kind: opaque }
        source: { kind: application, name: pixel_values }
      patch_lengths:
        contract: { dtype: int64, rank: 1, shape: [frames], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: patch_lengths }
      frame_offsets:
        contract: { dtype: int64, rank: 1, shape: [frame_offsets_len], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: frame_offsets }
      frame_owner:
        contract: { dtype: int64, rank: 1, shape: [frames], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: frame_owner }
      clip_offsets:
        contract: { dtype: int64, rank: 1, shape: [clip_offsets_len], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: clip_offsets }
      clip_owner:
        contract: { dtype: int64, rank: 1, shape: [clips], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: clip_owner }
    outputs:
      clip_embeddings:
        contract:
          dtype: float32
          rank: 2
          shape: [clips, hidden]
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: clip_offsets, owner: clip_owner }
            packed_extent: preserved
        role: tensor
        stage: pre_adapter
      clip_offsets:
        contract: { dtype: int64, rank: 1, shape: [clip_offsets_len], batch_layout: { kind: shared } }
        role: tensor
        stage: pre_adapter
      clip_owner:
        contract: { dtype: int64, rank: 1, shape: [clips], batch_layout: { kind: shared } }
        role: tensor
        stage: pre_adapter
    components:
      clip_encoder:
        implementation: { kind: onnx, artifact: encoder.onnx }
        batch_capacity:
          uniform_dimensions: [features]
          budgets:
            - { dimensions: [clips], max_total: 4 }
            - { dimensions: [frames], max_total: 64 }
            - { dimensions: [frames, patches], max_total: 65536 }
        ports:
          inputs:
            pixel_values:
              dtype: float32
              rank: 3
              shape: [frames, patches, features]
              batch_layout:
                kind: token_packed
                axis: 0
                levels:
                  - { offsets: frame_offsets, owner: frame_owner }
                  - { offsets: clip_offsets, owner: clip_owner }
              padding: [{ dimension: patches, valid_lengths: patch_lengths }]
            patch_lengths: { dtype: int64, rank: 1, shape: [frames], batch_layout: { kind: shared } }
            frame_offsets: { dtype: int64, rank: 1, shape: [frame_offsets_len], batch_layout: { kind: shared } }
            frame_owner: { dtype: int64, rank: 1, shape: [frames], batch_layout: { kind: shared } }
            clip_offsets: { dtype: int64, rank: 1, shape: [clip_offsets_len], batch_layout: { kind: shared } }
            clip_owner: { dtype: int64, rank: 1, shape: [clips], batch_layout: { kind: shared } }
          outputs:
            clip_embeddings:
              dtype: float32
              rank: 2
              shape: [clips, hidden]
              batch_layout:
                kind: token_packed
                axis: 0
                levels:
                  - { offsets: clip_offsets, owner: clip_owner }
                packed_extent: preserved
            media_tokens:
              dtype: float32
              rank: 2
              shape: [media_tokens_total, hidden]
              batch_layout:
                kind: token_packed
                axis: 0
                levels:
                  - { offsets: media_token_offsets, owner: media_token_owner }
                  - { offsets: clip_offsets, owner: clip_owner }
                packed_extent: produced
            media_token_offsets: { dtype: int64, rank: 1, shape: [media_token_offsets_len], batch_layout: { kind: shared } }
            media_token_owner: { dtype: int64, rank: 1, shape: [media_tokens_total], batch_layout: { kind: shared } }
    steps:
      - kind: invoke
        component: clip_encoder
        inputs:
          pixel_values: pixel_values
          patch_lengths: patch_lengths
          frame_offsets: frame_offsets
          frame_owner: frame_owner
          clip_offsets: clip_offsets
          clip_owner: clip_owner
        outputs:
          clip_embeddings: clips.embeddings
          media_tokens: clips.tokens
          media_token_offsets: clips.token_offsets
          media_token_owner: clips.token_owner
      - kind: emit
        value: clips.embeddings
        output: clip_embeddings
        mode: replace
      - kind: emit
        value: clip_offsets
        output: clip_offsets
        mode: replace
      - kind: emit
        value: clip_owner
        output: clip_owner
        mode: replace
"#;

#[test]
fn a_packing_resolves_frames_through_clips_to_request_rows() {
    let metadata = parse(NESTED_VIDEO_ENCODER);
    validate_metadata(&metadata).expect("nested video encoder is valid");
    let workflow = &metadata.pipeline.expect("pipeline").workflow;
    let layout = &workflow.inputs["pixel_values"].contract.batch_layout;
    let levels = layout.levels();
    assert_eq!(levels.len(), 2, "frames in clips in rows is two levels");
    assert_eq!(levels[0].offsets, "frame_offsets");
    assert_eq!(levels[0].owner, "frame_owner");
    assert_eq!(levels[1].offsets, "clip_offsets");
    assert_eq!(levels[1].owner, "clip_owner");
    assert_eq!(
        layout.packed_axis(),
        Some(0),
        "nesting adds levels over one packed axis, never a second one"
    );

    // The exact shapes the chain composes over, in the order it walks them:
    // one entry per frame, one per clip, and one per parent plus a total.
    let extent = |name: &str| {
        workflow.inputs[name]
            .contract
            .shape
            .as_ref()
            .and_then(|shape| shape.first())
            .cloned()
            .expect("a companion declares its extent")
    };
    let packed = workflow.inputs["pixel_values"]
        .contract
        .shape
        .as_ref()
        .and_then(|shape| shape.first())
        .cloned()
        .expect("a packed value declares its extent");
    assert_eq!(extent("frame_owner"), packed, "one owner entry per frame");
    assert_eq!(
        extent("clip_owner"),
        workflow.outputs["clip_embeddings"]
            .contract
            .shape
            .as_ref()
            .and_then(|shape| shape.first())
            .cloned()
            .expect("the clip embedding declares its extent"),
        "the clip level counts clips, and a clip-packed output counts the same clips"
    );
    assert_ne!(
        extent("frame_offsets"),
        extent("frame_owner"),
        "offsets is one longer than the run it indexes"
    );
    assert_ne!(
        extent("clip_offsets"),
        extent("clip_owner"),
        "offsets is one longer than the run it indexes"
    );

    // A packed value may still pad a different dimension: item count, item
    // extent, and item nesting never compete for the same axis.
    let padding = &workflow.inputs["pixel_values"].contract.padding;
    assert_eq!(padding.len(), 1);
    assert_eq!(padding[0].dimension, "patches");
    assert_eq!(padding[0].valid_lengths, "patch_lengths");
}

#[test]
fn a_mixed_chain_produces_one_level_and_preserves_the_other() {
    // A token-merging encoder decides its own packed extent while leaving the
    // clip-to-row mapping it never touched alone.
    let metadata = parse(NESTED_VIDEO_ENCODER);
    let workflow = metadata.pipeline.expect("pipeline").workflow;
    let tokens = &workflow.components["clip_encoder"].ports.outputs["media_tokens"];
    assert_eq!(
        tokens.batch_layout.packed_extent(),
        Some(PackedExtent::Produced)
    );
    let levels = tokens.batch_layout.levels();
    assert_eq!(levels[0].offsets, "media_token_offsets");
    assert_eq!(
        levels[1].offsets, "clip_offsets",
        "the outer level reuses the input pair the graph did not change"
    );
}

#[test]
fn ownership_composes_only_so_far() {
    let document = NESTED_VIDEO_ENCODER.replace(
        "              - { offsets: clip_offsets, owner: clip_owner }\n          padding:",
        "              - { offsets: clip_offsets, owner: clip_owner }\n              - { offsets: patch_lengths, owner: frame_owner }\n          padding:",
    );
    assert_reports(
        &document,
        "declares 3 ownership levels, more than the 2 a packed value may carry",
    );
}

#[test]
fn a_group_level_must_name_declared_values() {
    let document = NESTED_VIDEO_ENCODER.replace(
        "              - { offsets: clip_offsets, owner: clip_owner }\n          padding:",
        "              - { offsets: nowhere, owner: clip_owner }\n          padding:",
    );
    assert_reports(
        &document,
        "names 'nowhere' as its level 1 offsets, which is not a declared value or port in that \
         scope",
    );
}

#[test]
fn a_group_level_companion_is_a_shared_rank_one_integer() {
    let document = NESTED_VIDEO_ENCODER.replace(
        "      clip_owner:\n        contract: { dtype: int64, rank: 1, shape: [clips], batch_layout: { kind: shared } }",
        "      clip_owner:\n        contract: { dtype: float32, rank: 1, shape: [clips], batch_layout: { kind: shared } }",
    );
    assert_reports(
        &document,
        "level 1 owner map 'clip_owner' is float32 but must be int64",
    );
}

#[test]
fn no_two_ownership_levels_share_a_companion_value() {
    let document = NESTED_VIDEO_ENCODER.replace(
        "              - { offsets: clip_offsets, owner: clip_owner }\n          padding:",
        "              - { offsets: frame_offsets, owner: clip_owner }\n          padding:",
    );
    assert_reports(
        &document,
        "names 'frame_offsets' as both its level 0 offsets and its level 1 offsets; the two are \
         different vectors of different lengths",
    );
}

#[test]
fn a_nested_owner_map_is_as_long_as_the_run_it_indexes() {
    let document = NESTED_VIDEO_ENCODER.replace(
        "            clip_owner: { dtype: int64, rank: 1, shape: [clips], batch_layout: { kind: shared } }",
        "            clip_owner: { dtype: int64, rank: 1, shape: [clip_offsets_len], batch_layout: { kind: shared } }",
    );
    assert_reports(
        &document,
        "declares level 1 offsets 'clip_offsets' and owner map 'clip_owner' with the same extent \
         'clip_offsets_len'",
    );
}

#[test]
fn every_nested_level_is_budgeted() {
    let document =
        NESTED_VIDEO_ENCODER.replace("            - { dimensions: [clips], max_total: 4 }\n", "");
    assert_reports(
        &document,
        "declares no budget for 'clips', the units input 'pixel_values' packs at ownership level 1",
    );
}

/// A serving workflow that hands a packed result back to its caller, together
/// with the companions that make it splittable.
const PACKED_EMIT_WORKFLOW: &str = r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, typed_emit, serving_service_contract]
    serving:
      active: active
      done: done
      accepted_len: accepted_len
      state_service:
        groups:
          cache:
            kind: full_attention
            sequence_axis: 1
            layout: batch_sequence
            logical_lengths: cache_lengths
            aliasing: permitted
    state:
      cache_lengths:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: { kind: request_aligned, axis: 0 }
        scope: invocation
        initializer: cache_lengths.initial
        recurrence: { kind: invariant }
      cache:
        contract:
          dtype: float32
          rank: 2
          shape: [batch, capacity]
          batch_layout: { kind: request_aligned, axis: 0 }
        scope: invocation
        initializer: cache.initial
        recurrence: { kind: invariant }
        service_group: cache
        management: runtime
        release_boundary: invocation
    inputs:
      active:
        contract: { dtype: bool, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: active }
      done:
        contract: { dtype: bool, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: done }
      accepted_len:
        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: accepted_len }
      cache_lengths.initial:
        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: cache_lengths.initial }
      cache.initial:
        contract: { dtype: float32, rank: 2, shape: [batch, capacity], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: cache.initial }
      image_pixels:
        contract:
          dtype: float32
          rank: 4
          shape: [items, channels, height, width]
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: image_offsets, owner: image_owner }
        role: { kind: opaque }
        source: { kind: application, name: image_pixels }
      image_offsets:
        contract: { dtype: int64, rank: 1, shape: [rows_plus_one], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: image_offsets }
      image_owner:
        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: image_owner }
    outputs:
      features:
        contract:
          dtype: float32
          rank: 2
          shape: [items, hidden]
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: image_offsets, owner: image_owner }
            packed_extent: preserved
        role: tensor
        stage: pre_adapter
      image_offsets:
        contract: { dtype: int64, rank: 1, shape: [rows_plus_one], batch_layout: { kind: shared } }
        role: tensor
        stage: pre_adapter
      image_owner:
        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }
        role: tensor
        stage: pre_adapter
    components:
      vision:
        implementation: { kind: onnx, artifact: vision.onnx }
        ports:
          inputs:
            pixels:
              dtype: float32
              rank: 4
              shape: [items, channels, height, width]
              batch_layout:
                kind: token_packed
                axis: 0
                levels:
                  - { offsets: image_offsets, owner: image_owner }
          outputs:
            features:
              dtype: float32
              rank: 2
              shape: [items, hidden]
              batch_layout:
                kind: token_packed
                axis: 0
                levels:
                  - { offsets: image_offsets, owner: image_owner }
                packed_extent: preserved
    steps:
      - kind: invoke
        component: vision
        inputs: { pixels: image_pixels }
        outputs: { features: image_features }
      - kind: emit
        value: image_features
        output: features
        mode: replace
      - kind: emit
        value: image_offsets
        output: image_offsets
        mode: replace
      - kind: emit
        value: image_owner
        output: image_owner
        mode: replace
"#;

#[test]
fn a_packed_emit_hands_back_the_ownership_that_splits_it() {
    validate_metadata(&parse(PACKED_EMIT_WORKFLOW)).expect("a packed emit is describable");
}

#[test]
fn an_ownership_companion_may_be_emitted_by_a_serving_workflow() {
    // A companion is `shared` by construction — it describes the whole packing
    // rather than one row of it — so the rule that a serving workflow emits
    // per-request values with a declared row correspondence has to make room
    // for the very values that state that correspondence, and for nothing else.
    let document = PACKED_EMIT_WORKFLOW
        .replace(
            "      image_owner:\n        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }\n        role: tensor\n        stage: pre_adapter\n",
            "      unrelated:\n        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }\n        role: tensor\n        stage: pre_adapter\n",
        )
        .replace(
            "        value: image_owner\n        output: image_owner\n",
            "        value: image_owner\n        output: unrelated\n",
        );
    assert_reports(
        &document,
        "emits per-request value 'image_owner' without a declared batch_layout",
    );
}

#[test]
fn a_packed_emit_cannot_hide_the_companions_that_describe_it() {
    let document = PACKED_EMIT_WORKFLOW
        .replace(
            "      image_owner:\n        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }\n        role: tensor\n        stage: pre_adapter\n",
            "",
        )
        .replace(
            "      - kind: emit\n        value: image_owner\n        output: image_owner\n        mode: replace\n",
            "",
        );
    assert_reports(
        &document,
        "token_packed output 'features' whose level 0 owner 'image_owner' is not itself a \
         declared workflow output; a caller can only split a packed result with the companions \
         that describe it",
    );
}

#[test]
fn a_media_shaped_append_names_the_axis_it_grows() {
    // The default growth axis is the final one, which is right for a token
    // sequence and wrong for a clip: the last axis of a rank-4 media tensor is a
    // spatial extent that must never be concatenated.
    let document = PACKED_EMIT_WORKFLOW
        .replace(
            "      image_owner:\n        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }\n        role: tensor\n        stage: pre_adapter\n",
            "      image_owner:\n        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }\n        role: tensor\n        stage: pre_adapter\n      clip:\n        contract: { dtype: float32, rank: 4, shape: [items, channels, height, width], batch_layout: { kind: token_packed, axis: 0, levels: [{ offsets: image_offsets, owner: image_owner }] } }\n        role: video\n        stage: pre_adapter\n",
        )
        .replace(
            "      - kind: emit\n        value: image_owner\n        output: image_owner\n        mode: replace\n",
            "      - kind: emit\n        value: image_owner\n        output: image_owner\n        mode: replace\n      - kind: emit\n        value: image_pixels\n        output: clip\n        mode: append\n",
        );
    assert_reports(
        &document,
        "grows a rank-4 output but names no axis; the default final axis is a spatial extent for \
         a value of this rank, so the axis it grows along must be stated",
    );
}

#[test]
fn an_emit_axis_must_be_an_axis_the_value_has() {
    let document = PACKED_EMIT_WORKFLOW.replace(
        "        value: image_owner\n        output: image_owner\n        mode: replace",
        "        value: image_owner\n        output: image_owner\n        mode: append\n        axis: 3",
    );
    assert_reports(&document, "axis is 3, outside the rank 1 of value it emits");
}
