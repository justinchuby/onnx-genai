use std::io::Cursor;

use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use onnx_genai_metadata::{
    PreprocessingSpec, TensorContract, TensorDimension, VisionOutputBinding,
    VisionPreprocessingProgram,
};
use onnx_genai_preprocess::{
    batching::{PackedOwnershipLevel, RequestSpan},
    image::{GroupedVisionPreprocessor, ImageTensorData, MediaItem, MediaRequest},
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Document {
    preprocessing: PreprocessingSpec,
}

fn program(document: &str, kind: &str) -> VisionPreprocessingProgram {
    let preprocessing = serde_yaml::from_str::<Document>(document)
        .expect("grouped preprocessing fixture parses")
        .preprocessing;
    match kind {
        "image" => preprocessing.image.expect("fixture declares image program"),
        "video" => preprocessing.video.expect("fixture declares video program"),
        _ => panic!("unknown fixture kind"),
    }
}

fn png(width: u32, height: u32, color: [u8; 3]) -> Vec<u8> {
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, Rgb(color)));
    let mut encoded = Cursor::new(Vec::new());
    image
        .write_to(&mut encoded, ImageFormat::Png)
        .expect("test PNG encodes");
    encoded.into_inner()
}

const IMAGE_PROGRAM: &str = r#"
preprocessing:
  image:
    transforms:
      - { op: decode, outputs: [decoded] }
      - { op: resize, inputs: [decoded], outputs: [resized], size: 2, mode: stretch,
          interpolation: bilinear }
      - { op: tile, inputs: [resized], outputs: [tiles], tile_size: 2, max_tiles: 2 }
      - { op: rescale, inputs: [tiles], outputs: [scaled], scale: 0.00392156862745098 }
      - { op: patchify, inputs: [scaled], outputs: [patches], patch_size: 1, flatten: true }
      - { op: pad, inputs: [patches], outputs: [padded], target_length: 8, pad_value: 0 }
    outputs:
      - source: padded
        name: media.pixels
        content: pixels
        dtype: float32
        contract:
          dtype: float32
          shape: [items, max_patches, 3]
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: media.offsets, owner: media.owner, extent: produced }
          padding: [{ dimension: max_patches, valid_lengths: media.lengths }]
      - { source: runtime.offsets, name: media.offsets, content: pack_offsets, dtype: int64,
          contract: { dtype: int64, shape: [rows_plus_one], batch_layout: { kind: shared } } }
      - { source: runtime.owner, name: media.owner, content: pack_owner, dtype: int64,
          contract: { dtype: int64, shape: [items], batch_layout: { kind: shared } } }
      - { source: runtime.lengths, name: media.lengths, content: valid_lengths, dtype: int64,
          contract: { dtype: int64, shape: [items], batch_layout: { kind: shared } } }
"#;

const VIDEO_PROGRAM: &str = r#"
preprocessing:
  video:
    transforms:
      - { op: decode, outputs: [decoded] }
      - { op: resize, inputs: [decoded], outputs: [resized], size: 2, mode: stretch,
          interpolation: bilinear }
      - { op: tile, inputs: [resized], outputs: [tiles], tile_size: 2, max_tiles: 2 }
      - { op: rescale, inputs: [tiles], outputs: [scaled], scale: 0.00392156862745098 }
      - { op: patchify, inputs: [scaled], outputs: [patches], patch_size: 1, flatten: true }
      - { op: pad, inputs: [patches], outputs: [padded], target_length: 8, pad_value: 0 }
    outputs:
      - source: padded
        name: video.pixels
        content: pixels
        dtype: float32
        contract:
          dtype: float32
          shape: [frames, max_patches, 3]
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: video.frame_offsets, owner: video.frame_owner, extent: produced }
              - { offsets: video.clip_offsets, owner: video.clip_owner, extent: produced }
          padding: [{ dimension: max_patches, valid_lengths: video.lengths }]
      - { source: runtime.frame_offsets, name: video.frame_offsets, content: pack_offsets,
          dtype: int64, contract: { dtype: int64, shape: [clips_plus_one],
          batch_layout: { kind: shared } } }
      - { source: runtime.frame_owner, name: video.frame_owner, content: pack_owner,
          dtype: int64, contract: { dtype: int64, shape: [frames],
          batch_layout: { kind: shared } } }
      - { source: runtime.clip_offsets, name: video.clip_offsets, content: pack_offsets,
          dtype: int64, contract: { dtype: int64, shape: [rows_plus_one],
          batch_layout: { kind: shared } } }
      - { source: runtime.clip_owner, name: video.clip_owner, content: pack_owner,
          dtype: int64, contract: { dtype: int64, shape: [clips],
          batch_layout: { kind: shared } } }
      - { source: runtime.lengths, name: video.lengths, content: valid_lengths, dtype: int64,
          contract: { dtype: int64, shape: [frames], batch_layout: { kind: shared } } }
"#;

const MIXED_LAYOUT_PROGRAM: &str = r#"
preprocessing:
  image:
    transforms:
      - { op: decode, outputs: [decoded] }
      - { op: resize, inputs: [decoded], outputs: [resized], size: 2, mode: stretch,
          interpolation: bilinear }
      - { op: tile, inputs: [resized], outputs: [tiles], tile_size: 2, max_tiles: 2 }
      - { op: rescale, inputs: [tiles], outputs: [scaled], scale: 0.00392156862745098 }
      - { op: patchify, inputs: [scaled], outputs: [patches], patch_size: 1, flatten: true }
      - { op: emit_grid_coordinates, inputs: [patches], outputs: [grid] }
      - { op: pad, inputs: [patches], outputs: [padded], target_length: 8, pad_value: 0 }
    outputs:
      - source: padded
        name: media.pixels
        content: pixels
        dtype: float32
        contract:
          dtype: float32
          shape: [items, max_patches, 3]
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: media.offsets, owner: media.owner, extent: produced }
          padding: [{ dimension: max_patches, valid_lengths: media.lengths }]
      - source: grid
        name: media.grid
        content: grid_dimensions
        dtype: int64
        contract:
          dtype: int64
          shape: [request_rows, 3]
          batch_layout: GRID_LAYOUT
      - { source: runtime.offsets, name: media.offsets, content: pack_offsets, dtype: int64,
          contract: { dtype: int64, shape: [rows_plus_one], batch_layout: { kind: shared } } }
      - { source: runtime.owner, name: media.owner, content: pack_owner, dtype: int64,
          contract: { dtype: int64, shape: [items], batch_layout: { kind: shared } } }
      - { source: runtime.lengths, name: media.lengths, content: valid_lengths, dtype: int64,
          contract: { dtype: int64, shape: [items], batch_layout: { kind: shared } } }
"#;

fn mixed_layout_program(layout: &str) -> VisionPreprocessingProgram {
    program(
        &MIXED_LAYOUT_PROGRAM.replace("GRID_LAYOUT", layout),
        "image",
    )
}

fn i64_tensor(bundle: &onnx_genai_preprocess::image::GroupedVisionBundle, name: &str) -> Vec<i64> {
    match &bundle.tensors.tensor(name).expect("tensor exists").data {
        ImageTensorData::Int64(values) => values.clone(),
        data => panic!("{name} has unexpected data {data:?}"),
    }
}

#[test]
fn padded_images_emit_lengths_and_preserve_empty_request_boundaries() {
    let wide = png(4, 2, [255, 0, 0]);
    let square = png(2, 2, [0, 0, 255]);
    let processor = GroupedVisionPreprocessor::from_input_and_program(
        "image_preprocess",
        &[-1, -1, 3],
        &program(IMAGE_PROGRAM, "image"),
    )
    .unwrap();
    let requests = [
        MediaRequest::new([MediaItem::single(&wide), MediaItem::single(&square)]),
        MediaRequest::default(),
        MediaRequest::new([MediaItem::single(&square)]),
    ];

    let bundle = processor.preprocess_encoded(&requests).unwrap();
    assert_eq!(
        bundle.tensors.tensor("media.pixels").unwrap().shape,
        [3, 8, 3]
    );
    assert_eq!(i64_tensor(&bundle, "media.offsets"), [0, 2, 2, 3]);
    assert_eq!(i64_tensor(&bundle, "media.owner"), [0, 0, 2]);
    assert_eq!(i64_tensor(&bundle, "media.lengths"), [8, 4, 4]);
    assert_eq!(
        bundle.request_spans(),
        [
            RequestSpan {
                request_index: 0,
                item_offset: 0,
                item_length: 2,
                physical_offset: 0,
                physical_length: 2,
            },
            RequestSpan {
                request_index: 1,
                item_offset: 2,
                item_length: 0,
                physical_offset: 2,
                physical_length: 0,
            },
            RequestSpan {
                request_index: 2,
                item_offset: 2,
                item_length: 1,
                physical_offset: 2,
                physical_length: 1,
            },
        ]
    );
}

#[test]
fn request_aligned_output_rejects_two_rows_from_request_zero_and_none_from_request_one() {
    let wide = png(4, 2, [255, 0, 0]);
    let square = png(2, 2, [0, 0, 255]);
    let processor = GroupedVisionPreprocessor::from_input_and_program(
        "image_preprocess",
        &[-1, -1, 3],
        &mixed_layout_program("{ kind: request_aligned, axis: 0 }"),
    )
    .unwrap();
    let error = processor
        .preprocess_encoded(&[
            MediaRequest::new([MediaItem::single(&wide), MediaItem::single(&square)]),
            MediaRequest::default(),
        ])
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("media.grid"));
    assert!(message.contains("request_aligned"));
    assert!(message.contains("[2, 0]"));
}

#[test]
fn request_expanded_honors_factor_and_rejects_empty_contributions() {
    let wide = png(4, 2, [255, 0, 0]);
    let square = png(2, 2, [0, 0, 255]);
    let processor = GroupedVisionPreprocessor::from_input_and_program(
        "image_preprocess",
        &[-1, -1, 3],
        &mixed_layout_program("{ kind: request_expanded, axis: 0, factor: 2 }"),
    )
    .unwrap();
    let bundle = processor
        .preprocess_encoded(&[
            MediaRequest::new([MediaItem::single(&wide), MediaItem::single(&square)]),
            MediaRequest::new([MediaItem::single(&square), MediaItem::single(&wide)]),
        ])
        .unwrap();
    assert_eq!(bundle.tensors.tensor("media.grid").unwrap().shape, [4, 3]);
    assert_eq!(i64_tensor(&bundle, "media.offsets"), [0, 2, 4]);
    assert_eq!(i64_tensor(&bundle, "media.owner"), [0, 0, 1, 1]);

    let error = processor
        .preprocess_encoded(&[MediaRequest::default(), MediaRequest::default()])
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("request_expanded"));
    assert!(message.contains("2 physical row(s) per request"));
    assert!(message.contains("[0, 0]"));
}

#[test]
fn padding_reference_not_content_role_selects_the_synthesized_length() {
    let square = png(2, 2, [0, 0, 255]);
    let mut declared = program(IMAGE_PROGRAM, "image");
    declared
        .outputs
        .iter_mut()
        .find(|output| output.name == "media.lengths")
        .unwrap()
        .content = "custom_patch_count".to_owned();
    let processor = GroupedVisionPreprocessor::from_input_and_program(
        "image_preprocess",
        &[-1, -1, 3],
        &declared,
    )
    .unwrap();
    let bundle = processor
        .preprocess_encoded(&[MediaRequest::new([MediaItem::single(&square)])])
        .unwrap();
    let lengths = bundle.tensors.tensor("media.lengths").unwrap();
    assert_eq!(lengths.content, "custom_patch_count");
    assert_eq!(lengths.data, ImageTensorData::Int64(vec![4]));
}

#[test]
fn referenced_ownership_bindings_with_correct_roles_are_synthesized() {
    let square = png(2, 2, [0, 0, 255]);
    let processor = GroupedVisionPreprocessor::from_input_and_program(
        "image_preprocess",
        &[-1, -1, 3],
        &program(IMAGE_PROGRAM, "image"),
    )
    .unwrap();
    let bundle = processor
        .preprocess_encoded(&[MediaRequest::new([MediaItem::single(&square)])])
        .unwrap();

    assert_eq!(i64_tensor(&bundle, "media.offsets"), [0, 1]);
    assert_eq!(i64_tensor(&bundle, "media.owner"), [0]);
}

#[test]
fn referenced_ownership_bindings_reject_swapped_roles() {
    let mut declared = program(IMAGE_PROGRAM, "image");
    declared
        .outputs
        .iter_mut()
        .find(|output| output.name == "media.offsets")
        .unwrap()
        .content = "pack_owner".to_owned();
    declared
        .outputs
        .iter_mut()
        .find(|output| output.name == "media.owner")
        .unwrap()
        .content = "pack_offsets".to_owned();

    let error = GroupedVisionPreprocessor::from_input_and_program(
        "image_preprocess",
        &[-1, -1, 3],
        &declared,
    )
    .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("media.offsets"));
    assert!(message.contains("ownership offsets"));
    assert!(message.contains("content role 'pack_owner'"));
    assert!(message.contains("content role 'pack_offsets'"));
}

#[test]
fn referenced_ownership_binding_rejects_wrong_role() {
    let mut declared = program(IMAGE_PROGRAM, "image");
    declared
        .outputs
        .iter_mut()
        .find(|output| output.name == "media.owner")
        .unwrap()
        .content = "pixels".to_owned();

    let error = GroupedVisionPreprocessor::from_input_and_program(
        "image_preprocess",
        &[-1, -1, 3],
        &declared,
    )
    .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("media.owner"));
    assert!(message.contains("ownership owner map"));
    assert!(message.contains("content role 'pixels'"));
    assert!(message.contains("content role 'pack_owner'"));
}

#[test]
fn unreferenced_length_role_lookalike_is_not_intercepted() {
    let mut declared = program(IMAGE_PROGRAM, "image");
    declared.outputs.push(VisionOutputBinding {
        source: "padded".to_owned(),
        name: "media.lookalike".to_owned(),
        content: "valid_lengths".to_owned(),
        dtype: "int64".to_owned(),
        contract: Some(
            serde_yaml::from_str::<TensorContract>(
                "dtype: int64\nshape: [items]\nbatch_layout: { kind: shared }\n",
            )
            .unwrap(),
        ),
        pad_value: None,
        optional: None,
    });

    let error = GroupedVisionPreprocessor::from_input_and_program(
        "image_preprocess",
        &[-1, -1, 3],
        &declared,
    )
    .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("media.lookalike"));
    assert!(message.contains("content 'valid_lengths'"));
    assert!(
        !message.contains("not referenced by any padding"),
        "a role lookalike must remain an ordinary declared processor output: {message}"
    );
}

#[test]
fn nested_video_uses_one_frame_axis_and_two_ownership_levels() {
    let wide = png(4, 2, [255, 0, 0]);
    let blue = png(2, 2, [0, 0, 255]);
    let green = png(2, 2, [0, 255, 0]);
    let processor = GroupedVisionPreprocessor::from_input_and_program(
        "video_preprocess",
        &[-1, -1, 3],
        &program(VIDEO_PROGRAM, "video"),
    )
    .unwrap();
    let requests = [
        MediaRequest::new([
            MediaItem::nested([wide.as_slice(), blue.as_slice()]),
            MediaItem::nested([green.as_slice()]),
        ]),
        MediaRequest::new([MediaItem::nested([blue.as_slice(), green.as_slice()])]),
    ];

    let bundle = processor.preprocess_encoded(&requests).unwrap();
    assert_eq!(
        bundle.tensors.tensor("video.pixels").unwrap().shape,
        [5, 8, 3]
    );
    assert_eq!(i64_tensor(&bundle, "video.frame_offsets"), [0, 2, 3, 5]);
    assert_eq!(i64_tensor(&bundle, "video.frame_owner"), [0, 0, 1, 2, 2]);
    assert_eq!(i64_tensor(&bundle, "video.clip_offsets"), [0, 2, 3]);
    assert_eq!(i64_tensor(&bundle, "video.clip_owner"), [0, 0, 1]);
    assert_eq!(i64_tensor(&bundle, "video.lengths"), [8, 4, 4, 4, 4]);

    let second = bundle.request_local(1).unwrap();
    assert_eq!(second.span.physical_offset, 3);
    assert_eq!(second.span.physical_length, 2);
    assert_eq!(
        second.levels,
        [
            PackedOwnershipLevel {
                offsets: vec![0, 2],
                owner: vec![0, 0],
            },
            PackedOwnershipLevel {
                offsets: vec![0, 1],
                owner: vec![0],
            },
        ]
    );
}

#[test]
fn decode_turn_with_zero_new_media_emits_empty_ranked_tensors_and_rows() {
    let processor = GroupedVisionPreprocessor::from_input_and_program(
        "video_preprocess",
        &[-1, -1, 3],
        &program(VIDEO_PROGRAM, "video"),
    )
    .unwrap();
    let bundle = processor
        .preprocess_encoded(&[MediaRequest::default(), MediaRequest::default()])
        .unwrap();

    assert_eq!(
        bundle.tensors.tensor("video.pixels").unwrap().shape,
        [0, 8, 3]
    );
    assert_eq!(i64_tensor(&bundle, "video.frame_offsets"), [0]);
    assert!(i64_tensor(&bundle, "video.frame_owner").is_empty());
    assert_eq!(i64_tensor(&bundle, "video.clip_offsets"), [0, 0, 0]);
    assert!(i64_tensor(&bundle, "video.clip_owner").is_empty());
    assert!(i64_tensor(&bundle, "video.lengths").is_empty());
    assert_eq!(bundle.request_spans()[0].physical_length, 0);
    assert_eq!(bundle.request_spans()[1].physical_length, 0);
}

#[test]
fn empty_media_rejects_unresolved_symbolic_feature_width() {
    let mut declared = program(VIDEO_PROGRAM, "video");
    let pixels = declared
        .outputs
        .iter_mut()
        .find(|output| output.name == "video.pixels")
        .unwrap();
    pixels.contract.as_mut().unwrap().shape = vec![
        TensorDimension::Symbol("frames".to_owned()),
        TensorDimension::Symbol("max_patches".to_owned()),
        TensorDimension::Symbol("features".to_owned()),
    ];
    let processor = GroupedVisionPreprocessor::from_input_and_program(
        "video_preprocess",
        &[-1, -1, -1],
        &declared,
    )
    .unwrap();
    let error = processor
        .preprocess_encoded(&[MediaRequest::default()])
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("symbolic inner dimension 'features'"));
    assert!(message.contains("refusing to invent extent zero"));
}

#[test]
fn topology_mismatch_and_temporal_container_execution_fail_closed() {
    let frame = png(2, 2, [0, 0, 0]);
    let image_processor = GroupedVisionPreprocessor::from_input_and_program(
        "image_preprocess",
        &[-1, -1, 3],
        &program(IMAGE_PROGRAM, "image"),
    )
    .unwrap();
    let error = image_processor
        .preprocess_encoded(&[MediaRequest::new([MediaItem::nested([
            frame.as_slice(),
            frame.as_slice(),
        ])])])
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("multi-part item requires two levels")
    );

    let temporal = VIDEO_PROGRAM.replacen(
        "      - { op: resize, inputs: [decoded],",
        concat!(
            "      - { op: sample_frames, inputs: [decoded], outputs: [sampled], ",
            "num_frames: 2 }\n",
            "      - { op: resize, inputs: [sampled],"
        ),
        1,
    );
    let error = GroupedVisionPreprocessor::from_input_and_program(
        "video_preprocess",
        &[-1, -1, 3],
        &program(&temporal, "video"),
    )
    .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("sample_frames"));
    assert!(message.contains("never skips a declared temporal transform"));
}
