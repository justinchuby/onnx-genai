//! Hermetic generalized image/video encoder batching through real ORT sessions.

use std::hint::black_box;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
#[cfg(feature = "native-backend")]
use onnx_genai_engine::NativeDecodeDevice;
use onnx_genai_engine::pipeline::{
    BatchContractError, ComposedOwnership, OwnershipLevelValues, PackedOwnership, PipelineTensors,
    batch_contract_error,
};
use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest,
    PipelineGenerateRequest,
};
use onnx_genai_metadata::{PreprocessingSpec, VisionPreprocessingProgram};
use onnx_genai_ort::{Session, SessionOptions, Value};
use onnx_genai_preprocess::image::{
    GroupedVisionBundle, GroupedVisionPreprocessor, ImageTensorData, MediaItem, MediaRequest,
    NamedImageTensor,
};
use onnx_genai_scheduler::BatchAdmissionError;
use onnx_runtime_ir::{
    Attribute, DataType as IrDataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef,
    static_shape,
};
use onnx_std::Model;
use serde::Deserialize;

const IMAGE_PREPROCESSING: &str = r#"
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
          rank: 3
          shape: [items, max_patches, 3]
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: media.offsets, owner: media.owner, extent: produced }
          padding: [{ dimension: max_patches, valid_lengths: media.lengths }]
      - { source: runtime.offsets, name: media.offsets, content: pack_offsets, dtype: int64,
          contract: { dtype: int64, rank: 1, shape: [rows_plus_one],
          batch_layout: { kind: shared } } }
      - { source: runtime.owner, name: media.owner, content: pack_owner, dtype: int64,
          contract: { dtype: int64, rank: 1, shape: [items],
          batch_layout: { kind: shared } } }
      - { source: runtime.lengths, name: media.lengths, content: valid_lengths, dtype: int64,
          contract: { dtype: int64, rank: 1, shape: [items],
          batch_layout: { kind: shared } } }
"#;

const VIDEO_PREPROCESSING: &str = r#"
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
          rank: 3
          shape: [frames, max_patches, 3]
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: video.frame_offsets, owner: video.frame_owner, extent: produced }
              - { offsets: video.clip_offsets, owner: video.clip_owner, extent: produced }
          padding: [{ dimension: max_patches, valid_lengths: video.lengths }]
      - { source: runtime.frame_offsets, name: video.frame_offsets, content: pack_offsets,
          dtype: int64, contract: { dtype: int64, rank: 1, shape: [clips_plus_one],
          batch_layout: { kind: shared } } }
      - { source: runtime.frame_owner, name: video.frame_owner, content: pack_owner,
          dtype: int64, contract: { dtype: int64, rank: 1, shape: [frames],
          batch_layout: { kind: shared } } }
      - { source: runtime.clip_offsets, name: video.clip_offsets, content: pack_offsets,
          dtype: int64, contract: { dtype: int64, rank: 1, shape: [rows_plus_one],
          batch_layout: { kind: shared } } }
      - { source: runtime.clip_owner, name: video.clip_owner, content: pack_owner,
          dtype: int64, contract: { dtype: int64, rank: 1, shape: [clips],
          batch_layout: { kind: shared } } }
      - { source: runtime.lengths, name: video.lengths, content: valid_lengths, dtype: int64,
          contract: { dtype: int64, rank: 1, shape: [frames],
          batch_layout: { kind: shared } } }
"#;

const IMAGE_METADATA: &str = r#"
schema_version: v1.1
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, linear_effects, typed_emit]
    inputs:
      pixels:
        contract:
          dtype: float32
          rank: 3
          shape: [items, max_patches, channels]
          batch_layout:
            kind: token_packed
            axis: 0
            levels: [{ offsets: offsets, owner: owner }]
          padding: [{ dimension: max_patches, valid_lengths: lengths }]
        role: { kind: opaque }
        source: { kind: application, name: pixels }
      offsets:
        contract: { dtype: int64, rank: 1, shape: [rows_plus_one],
          batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: offsets }
      owner:
        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: owner }
      lengths:
        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: lengths }
      request.marker:
        contract: { dtype: float32, rank: 1, shape: [batch],
          batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: request.marker }
      guard:
        contract: { dtype: int64, rank: 1, shape: [1], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: guard }
    outputs:
      features:
        contract:
          dtype: float32
          rank: 3
          shape: [items, max_patches, channels]
          batch_layout:
            kind: token_packed
            axis: 0
            levels: [{ offsets: offsets, owner: owner, extent: preserved }]
          padding: [{ dimension: max_patches, valid_lengths: lengths }]
        role: tensor
        stage: pre_adapter
      offsets:
        contract: { dtype: int64, rank: 1, shape: [rows_plus_one],
          batch_layout: { kind: shared } }
        role: tensor
        stage: pre_adapter
      owner:
        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }
        role: tensor
        stage: pre_adapter
      lengths:
        contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }
        role: tensor
        stage: pre_adapter
    components:
      encoder:
        implementation: { kind: onnx, artifact: encoder.onnx }
BATCH_CAPACITY
        ports:
          inputs:
            pixels:
              dtype: float32
              rank: 3
              shape: [items, max_patches, channels]
              batch_layout:
                kind: token_packed
                axis: 0
                levels: [{ offsets: offsets, owner: owner }]
              padding: [{ dimension: max_patches, valid_lengths: lengths }]
            offsets: { dtype: int64, rank: 1, shape: [rows_plus_one],
              batch_layout: { kind: shared } }
            owner: { dtype: int64, rank: 1, shape: [items],
              batch_layout: { kind: shared } }
            lengths: { dtype: int64, rank: 1, shape: [items],
              batch_layout: { kind: shared } }
            marker: { dtype: float32, rank: 1, shape: [batch],
              batch_layout: { kind: request_aligned, axis: 0 } }
            guard: { dtype: int64, rank: 1, shape: [1],
              batch_layout: { kind: shared } }
          outputs:
            features:
              dtype: float32
              rank: 3
              shape: [items, max_patches, channels]
              batch_layout:
                kind: token_packed
                axis: 0
                levels: [{ offsets: offsets, owner: owner, extent: preserved }]
              padding: [{ dimension: max_patches, valid_lengths: lengths }]
    steps:
      - kind: invoke
        component: encoder
        inputs:
          pixels: pixels
          offsets: offsets
          owner: owner
          lengths: lengths
          marker: request.marker
          guard: guard
        outputs: { features: encoded.features }
      - { kind: emit, value: encoded.features, output: features, mode: replace }
      - { kind: emit, value: offsets, output: offsets, mode: replace }
      - { kind: emit, value: owner, output: owner, mode: replace }
      - { kind: emit, value: lengths, output: lengths, mode: replace }
"#;

const VIDEO_METADATA: &str = r#"
schema_version: v1.1
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, linear_effects, typed_emit]
    inputs:
      pixels:
        contract:
          dtype: float32
          rank: 3
          shape: [frames, max_patches, channels]
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: frame_offsets, owner: frame_owner }
              - { offsets: clip_offsets, owner: clip_owner }
          padding: [{ dimension: max_patches, valid_lengths: lengths }]
        role: { kind: opaque }
        source: { kind: application, name: pixels }
      frame_offsets:
        contract: { dtype: int64, rank: 1, shape: [clips_plus_one],
          batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: frame_offsets }
      frame_owner:
        contract: { dtype: int64, rank: 1, shape: [frames], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: frame_owner }
      clip_offsets:
        contract: { dtype: int64, rank: 1, shape: [rows_plus_one],
          batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: clip_offsets }
      clip_owner:
        contract: { dtype: int64, rank: 1, shape: [clips], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: clip_owner }
      lengths:
        contract: { dtype: int64, rank: 1, shape: [frames], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: lengths }
      request.marker:
        contract: { dtype: float32, rank: 1, shape: [batch],
          batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: request.marker }
      guard:
        contract: { dtype: int64, rank: 1, shape: [1], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: guard }
    outputs:
      features:
        contract:
          dtype: float32
          rank: 3
          shape: [frames, max_patches, channels]
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: frame_offsets, owner: frame_owner, extent: preserved }
              - { offsets: clip_offsets, owner: clip_owner, extent: preserved }
          padding: [{ dimension: max_patches, valid_lengths: lengths }]
        role: tensor
        stage: pre_adapter
      frame_offsets:
        contract: { dtype: int64, rank: 1, shape: [clips_plus_one],
          batch_layout: { kind: shared } }
        role: tensor
        stage: pre_adapter
      frame_owner:
        contract: { dtype: int64, rank: 1, shape: [frames],
          batch_layout: { kind: shared } }
        role: tensor
        stage: pre_adapter
      clip_offsets:
        contract: { dtype: int64, rank: 1, shape: [rows_plus_one],
          batch_layout: { kind: shared } }
        role: tensor
        stage: pre_adapter
      clip_owner:
        contract: { dtype: int64, rank: 1, shape: [clips],
          batch_layout: { kind: shared } }
        role: tensor
        stage: pre_adapter
      lengths:
        contract: { dtype: int64, rank: 1, shape: [frames],
          batch_layout: { kind: shared } }
        role: tensor
        stage: pre_adapter
    components:
      encoder:
        implementation: { kind: onnx, artifact: encoder.onnx }
        batch_capacity:
          uniform_dimensions: [channels]
          budgets:
            - { dimensions: [batch], max_total: 2 }
            - { dimensions: [clips], max_total: 4 }
            - { dimensions: [frames, max_patches], max_total: 48 }
        ports:
          inputs:
            pixels:
              dtype: float32
              rank: 3
              shape: [frames, max_patches, channels]
              batch_layout:
                kind: token_packed
                axis: 0
                levels:
                  - { offsets: frame_offsets, owner: frame_owner }
                  - { offsets: clip_offsets, owner: clip_owner }
              padding: [{ dimension: max_patches, valid_lengths: lengths }]
            frame_offsets: { dtype: int64, rank: 1, shape: [clips_plus_one],
              batch_layout: { kind: shared } }
            frame_owner: { dtype: int64, rank: 1, shape: [frames],
              batch_layout: { kind: shared } }
            clip_offsets: { dtype: int64, rank: 1, shape: [rows_plus_one],
              batch_layout: { kind: shared } }
            clip_owner: { dtype: int64, rank: 1, shape: [clips],
              batch_layout: { kind: shared } }
            lengths: { dtype: int64, rank: 1, shape: [frames],
              batch_layout: { kind: shared } }
            marker: { dtype: float32, rank: 1, shape: [batch],
              batch_layout: { kind: request_aligned, axis: 0 } }
            guard: { dtype: int64, rank: 1, shape: [1],
              batch_layout: { kind: shared } }
          outputs:
            features:
              dtype: float32
              rank: 3
              shape: [frames, max_patches, channels]
              batch_layout:
                kind: token_packed
                axis: 0
                levels:
                  - { offsets: frame_offsets, owner: frame_owner, extent: preserved }
                  - { offsets: clip_offsets, owner: clip_owner, extent: preserved }
              padding: [{ dimension: max_patches, valid_lengths: lengths }]
    steps:
      - kind: invoke
        component: encoder
        inputs:
          pixels: pixels
          frame_offsets: frame_offsets
          frame_owner: frame_owner
          clip_offsets: clip_offsets
          clip_owner: clip_owner
          lengths: lengths
          marker: request.marker
          guard: guard
        outputs: { features: encoded.features }
      - { kind: emit, value: encoded.features, output: features, mode: replace }
      - { kind: emit, value: frame_offsets, output: frame_offsets, mode: replace }
      - { kind: emit, value: frame_owner, output: frame_owner, mode: replace }
      - { kind: emit, value: clip_offsets, output: clip_offsets, mode: replace }
      - { kind: emit, value: clip_owner, output: clip_owner, mode: replace }
      - { kind: emit, value: lengths, output: lengths, mode: replace }
"#;

#[derive(Deserialize)]
struct PreprocessingDocument {
    preprocessing: PreprocessingSpec,
}

#[derive(Clone, Copy)]
enum FixtureKind {
    Image,
    Video,
}

fn preprocessing_program(document: &str, kind: FixtureKind) -> VisionPreprocessingProgram {
    let spec = serde_yaml::from_str::<PreprocessingDocument>(document)
        .expect("preprocessing metadata parses")
        .preprocessing;
    match kind {
        FixtureKind::Image => spec.image.expect("image program"),
        FixtureKind::Video => spec.video.expect("video program"),
    }
}

fn processor(kind: FixtureKind) -> GroupedVisionPreprocessor {
    let (component, document) = match kind {
        FixtureKind::Image => ("image_preprocess", IMAGE_PREPROCESSING),
        FixtureKind::Video => ("video_preprocess", VIDEO_PREPROCESSING),
    };
    GroupedVisionPreprocessor::from_input_and_program(
        component,
        &[-1, -1, 3],
        &preprocessing_program(document, kind),
    )
    .expect("grouped processor builds from authored metadata")
}

fn png(width: u32, height: u32, color: [u8; 3]) -> Vec<u8> {
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, Rgb(color)));
    let mut encoded = Cursor::new(Vec::new());
    image
        .write_to(&mut encoded, ImageFormat::Png)
        .expect("test PNG encodes");
    encoded.into_inner()
}

fn test_root(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-fixtures/media-batching-e2e")
        .join(name)
}

fn image_metadata(max_materialized: Option<usize>) -> String {
    let capacity = max_materialized.map_or_else(String::new, |max_total| {
        format!(
            "        batch_capacity:\n          uniform_dimensions: [channels]\n          budgets:\n            - {{ dimensions: [batch], max_total: 2 }}\n            - {{ dimensions: [items, max_patches], max_total: {max_total} }}"
        )
    });
    IMAGE_METADATA.replace("BATCH_CAPACITY", &capacity)
}

fn build_package(name: &str, kind: FixtureKind, metadata: &str) -> anyhow::Result<PathBuf> {
    let root = test_root(name);
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root)?;
    std::fs::write(root.join("inference_metadata.yaml"), metadata)?;
    onnx_std::save_model(&build_encoder_model(kind), root.join("encoder.onnx"))?;
    Ok(root)
}

fn ort_engine(root: &Path) -> anyhow::Result<Engine> {
    Engine::from_dir_with_session_options(
        root,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Ort,
            ..EngineConfig::default()
        },
        SessionOptions::default().with_intra_op_threads(1),
    )
}

fn tensor_f32(shape: Vec<usize>, values: &[f32]) -> TensorData {
    TensorData::from_raw(
        IrDataType::Float32,
        shape,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
}

fn tensor_i64(shape: Vec<usize>, values: &[i64]) -> TensorData {
    TensorData::from_raw(
        IrDataType::Int64,
        shape,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
}

fn initializer(graph: &mut Graph, name: &str, dtype: IrDataType, data: TensorData) -> ValueId {
    let shape = static_shape(data.dims.iter().copied());
    let value = graph.create_named_value(name, dtype, shape);
    graph.set_initializer(value, WeightRef::Inline(data));
    value
}

fn insert_node(
    graph: &mut Graph,
    op_type: &str,
    inputs: &[ValueId],
    outputs: &[ValueId],
    attributes: &[(&str, Attribute)],
) {
    let mut node = Node::new(
        NodeId(0),
        op_type,
        inputs.iter().copied().map(Some).collect(),
        outputs.to_vec(),
    );
    for (name, value) in attributes {
        node.attributes.insert((*name).to_owned(), value.clone());
    }
    graph.insert_node(node);
}

fn cast_f32(graph: &mut Graph, input: ValueId, output: ValueId) {
    insert_node(
        graph,
        "Cast",
        &[input],
        &[output],
        &[("to", Attribute::Int(1))],
    );
}

fn build_encoder_model(kind: FixtureKind) -> Model {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 18);
    let physical = graph.intern_symbol(match kind {
        FixtureKind::Image => "items",
        FixtureKind::Video => "frames",
    });
    let patches = graph.intern_symbol("max_patches");
    let channels = graph.intern_symbol("channels");
    let batch = graph.intern_symbol("batch");
    let pixels = graph.create_named_value(
        "pixels",
        IrDataType::Float32,
        vec![physical.into(), patches.into(), channels.into()],
    );
    let lengths = graph.create_named_value("lengths", IrDataType::Int64, vec![physical.into()]);
    let marker = graph.create_named_value("marker", IrDataType::Float32, vec![batch.into()]);
    let guard = graph.create_named_value("guard", IrDataType::Int64, static_shape([1]));
    graph.add_input(pixels);

    let (owner, request_owner) = match kind {
        FixtureKind::Image => {
            let offsets_symbol = graph.intern_symbol("rows_plus_one");
            let offsets =
                graph.create_named_value("offsets", IrDataType::Int64, vec![offsets_symbol.into()]);
            let owner = graph.create_named_value("owner", IrDataType::Int64, vec![physical.into()]);
            graph.add_input(offsets);
            graph.add_input(owner);
            (owner, owner)
        }
        FixtureKind::Video => {
            let clips = graph.intern_symbol("clips");
            let clips_plus_one = graph.intern_symbol("clips_plus_one");
            let rows_plus_one = graph.intern_symbol("rows_plus_one");
            let frame_offsets = graph.create_named_value(
                "frame_offsets",
                IrDataType::Int64,
                vec![clips_plus_one.into()],
            );
            let frame_owner =
                graph.create_named_value("frame_owner", IrDataType::Int64, vec![physical.into()]);
            let clip_offsets = graph.create_named_value(
                "clip_offsets",
                IrDataType::Int64,
                vec![rows_plus_one.into()],
            );
            let clip_owner =
                graph.create_named_value("clip_owner", IrDataType::Int64, vec![clips.into()]);
            let request_owner =
                graph.create_named_value("request_owner", IrDataType::Int64, vec![physical.into()]);
            for input in [frame_offsets, frame_owner, clip_offsets, clip_owner] {
                graph.add_input(input);
            }
            insert_node(
                &mut graph,
                "Gather",
                &[clip_owner, frame_owner],
                &[request_owner],
                &[("axis", Attribute::Int(0))],
            );
            (frame_owner, request_owner)
        }
    };
    for input in [lengths, marker, guard] {
        graph.add_input(input);
    }

    let two = initializer(
        &mut graph,
        "two",
        IrDataType::Float32,
        tensor_f32(vec![1], &[2.0]),
    );
    let ten = initializer(
        &mut graph,
        "ten",
        IrDataType::Float32,
        tensor_f32(vec![1], &[10.0]),
    );
    let hundred = initializer(
        &mut graph,
        "hundred",
        IrDataType::Float32,
        tensor_f32(vec![1], &[100.0]),
    );
    let thousand = initializer(
        &mut graph,
        "thousand",
        IrDataType::Float32,
        tensor_f32(vec![1], &[1_000.0]),
    );
    let marker_scale = initializer(
        &mut graph,
        "marker_scale",
        IrDataType::Float32,
        tensor_f32(
            vec![1],
            &[match kind {
                FixtureKind::Image => 1_000.0,
                FixtureKind::Video => 10_000.0,
            }],
        ),
    );
    let axes = initializer(
        &mut graph,
        "axes",
        IrDataType::Int64,
        tensor_i64(vec![2], &[1, 2]),
    );
    let guard_one = initializer(
        &mut graph,
        "guard_one",
        IrDataType::Int64,
        tensor_i64(vec![1], &[1]),
    );

    let shape_1d = vec![physical.into()];
    let shape_3d = vec![physical.into(), 1.into(), 1.into()];
    let full_shape = vec![physical.into(), patches.into(), channels.into()];
    let scaled = graph.create_named_value("scaled", IrDataType::Float32, full_shape.clone());
    insert_node(&mut graph, "Mul", &[pixels, two], &[scaled], &[]);

    let length_f = graph.create_named_value("length_f", IrDataType::Float32, shape_1d.clone());
    cast_f32(&mut graph, lengths, length_f);
    let length_scaled =
        graph.create_named_value("length_scaled", IrDataType::Float32, shape_1d.clone());
    insert_node(&mut graph, "Mul", &[length_f, ten], &[length_scaled], &[]);
    let length_term =
        graph.create_named_value("length_term", IrDataType::Float32, shape_3d.clone());
    insert_node(
        &mut graph,
        "Unsqueeze",
        &[length_scaled, axes],
        &[length_term],
        &[],
    );

    let owner_f = graph.create_named_value("owner_f", IrDataType::Float32, shape_1d.clone());
    cast_f32(&mut graph, owner, owner_f);
    let owner_scaled =
        graph.create_named_value("owner_scaled", IrDataType::Float32, shape_1d.clone());
    insert_node(&mut graph, "Mul", &[owner_f, hundred], &[owner_scaled], &[]);
    let owner_term = graph.create_named_value("owner_term", IrDataType::Float32, shape_3d.clone());
    insert_node(
        &mut graph,
        "Unsqueeze",
        &[owner_scaled, axes],
        &[owner_term],
        &[],
    );

    let request_owner_f =
        graph.create_named_value("request_owner_f", IrDataType::Float32, shape_1d.clone());
    cast_f32(&mut graph, request_owner, request_owner_f);
    let request_scaled =
        graph.create_named_value("request_scaled", IrDataType::Float32, shape_1d.clone());
    insert_node(
        &mut graph,
        "Mul",
        &[request_owner_f, thousand],
        &[request_scaled],
        &[],
    );
    let request_term =
        graph.create_named_value("request_term", IrDataType::Float32, shape_3d.clone());
    insert_node(
        &mut graph,
        "Unsqueeze",
        &[request_scaled, axes],
        &[request_term],
        &[],
    );

    let marker_by_item =
        graph.create_named_value("marker_by_item", IrDataType::Float32, shape_1d.clone());
    insert_node(
        &mut graph,
        "Gather",
        &[marker, request_owner],
        &[marker_by_item],
        &[("axis", Attribute::Int(0))],
    );
    let marker_scaled =
        graph.create_named_value("marker_scaled", IrDataType::Float32, shape_1d.clone());
    insert_node(
        &mut graph,
        "Mul",
        &[marker_by_item, marker_scale],
        &[marker_scaled],
        &[],
    );
    let marker_term =
        graph.create_named_value("marker_term", IrDataType::Float32, shape_3d.clone());
    insert_node(
        &mut graph,
        "Unsqueeze",
        &[marker_scaled, axes],
        &[marker_term],
        &[],
    );

    let guard_div = graph.create_named_value("guard_div", IrDataType::Int64, static_shape([1]));
    insert_node(&mut graph, "Div", &[guard_one, guard], &[guard_div], &[]);
    let guard_term = graph.create_named_value("guard_term", IrDataType::Float32, static_shape([1]));
    cast_f32(&mut graph, guard_div, guard_term);

    let sum0 = graph.create_named_value("sum0", IrDataType::Float32, full_shape.clone());
    let sum1 = graph.create_named_value("sum1", IrDataType::Float32, full_shape.clone());
    let sum2 = graph.create_named_value("sum2", IrDataType::Float32, full_shape.clone());
    let sum3 = graph.create_named_value("sum3", IrDataType::Float32, full_shape.clone());
    let features = graph.create_named_value("features", IrDataType::Float32, full_shape);
    insert_node(&mut graph, "Add", &[scaled, length_term], &[sum0], &[]);
    insert_node(&mut graph, "Add", &[sum0, owner_term], &[sum1], &[]);
    insert_node(&mut graph, "Add", &[sum1, request_term], &[sum2], &[]);
    insert_node(&mut graph, "Add", &[sum2, marker_term], &[sum3], &[]);
    insert_node(&mut graph, "Add", &[sum3, guard_term], &[features], &[]);
    graph.add_output(features);
    Model::new(graph)
}

fn value_from_tensor(tensor: NamedImageTensor) -> anyhow::Result<Value> {
    match tensor.data {
        ImageTensorData::Fp32(values) => {
            Value::from_vec_f32(values, &tensor.shape).map_err(Into::into)
        }
        ImageTensorData::Int64(values) => {
            Value::from_vec_i64(values, &tensor.shape).map_err(Into::into)
        }
        other => anyhow::bail!(
            "media batching fixture only authors float32/int64 outputs, got {other:?}"
        ),
    }
}

fn i64_tensor(bundle: &GroupedVisionBundle, name: &str) -> Vec<i64> {
    match &bundle.tensors.tensor(name).expect("tensor exists").data {
        ImageTensorData::Int64(values) => values.clone(),
        other => panic!("{name} has unexpected data {other:?}"),
    }
}

fn f32_tensor(bundle: &GroupedVisionBundle, name: &str) -> Vec<f32> {
    match &bundle.tensors.tensor(name).expect("tensor exists").data {
        ImageTensorData::Fp32(values) => values.clone(),
        other => panic!("{name} has unexpected data {other:?}"),
    }
}

fn workflow_request(
    bundle: &GroupedVisionBundle,
    marker: &[f32],
    guard: i64,
) -> anyhow::Result<PipelineGenerateRequest> {
    let mut request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(Vec::new())));
    for tensor in &bundle.tensors.tensors {
        let input_name = tensor
            .name
            .strip_prefix("media.")
            .or_else(|| tensor.name.strip_prefix("video."))
            .unwrap_or(&tensor.name);
        request
            .inputs
            .insert(input_name.to_owned(), value_from_tensor(tensor.clone())?);
    }
    request.inputs.insert(
        "request.marker".into(),
        Value::from_slice_f32(marker, &[i64::try_from(marker.len())?])?,
    );
    request
        .inputs
        .insert("guard".into(), Value::from_slice_i64(&[guard], &[1])?);
    Ok(request)
}

fn component_inputs(
    bundle: &GroupedVisionBundle,
    marker: &[f32],
    guard: i64,
    kind: FixtureKind,
) -> anyhow::Result<PipelineTensors> {
    let mut values = PipelineTensors::new();
    let mappings: &[(&str, &str)] = match kind {
        FixtureKind::Image => &[
            ("media.pixels", "pixels"),
            ("media.offsets", "offsets"),
            ("media.owner", "owner"),
            ("media.lengths", "lengths"),
        ],
        FixtureKind::Video => &[
            ("video.pixels", "pixels"),
            ("video.frame_offsets", "frame_offsets"),
            ("video.frame_owner", "frame_owner"),
            ("video.clip_offsets", "clip_offsets"),
            ("video.clip_owner", "clip_owner"),
            ("video.lengths", "lengths"),
        ],
    };
    for (source, port) in mappings {
        values.insert(
            (*port).to_owned(),
            value_from_tensor(
                bundle
                    .tensors
                    .tensor(source)
                    .expect("tensor exists")
                    .clone(),
            )?,
        );
    }
    values.insert(
        "marker".into(),
        Value::from_slice_f32(marker, &[i64::try_from(marker.len())?])?,
    );
    values.insert("guard".into(), Value::from_slice_i64(&[guard], &[1])?);
    Ok(values)
}

fn ownership(bundle: &GroupedVisionBundle, kind: FixtureKind) -> anyhow::Result<PackedOwnership> {
    let (pixels, levels): (&str, Vec<(&str, &str)>) = match kind {
        FixtureKind::Image => ("media.pixels", vec![("media.offsets", "media.owner")]),
        FixtureKind::Video => (
            "video.pixels",
            vec![
                ("video.frame_offsets", "video.frame_owner"),
                ("video.clip_offsets", "video.clip_owner"),
            ],
        ),
    };
    let packed_extent = usize::try_from(bundle.tensors.tensor(pixels).unwrap().shape[0])?;
    let request_count = bundle
        .ownership
        .as_ref()
        .expect("packed fixture")
        .request_count();
    let levels = levels
        .into_iter()
        .map(|(offsets, owners)| {
            Ok(OwnershipLevelValues::new(
                value_from_tensor(bundle.tensors.tensor(offsets).unwrap().clone())?,
                value_from_tensor(bundle.tensors.tensor(owners).unwrap().clone())?,
            ))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(PackedOwnership::new(packed_extent, levels, request_count)?)
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "value {index}: actual {actual}, expected {expected}"
        );
    }
}

fn pipeline_error(
    result: anyhow::Result<PipelineTensors>,
    expected_failure: &str,
) -> anyhow::Error {
    match result {
        Ok(_) => panic!("{expected_failure}"),
        Err(error) => error,
    }
}

fn expected_image(bundle: &GroupedVisionBundle, marker: &[f32]) -> Vec<f32> {
    let pixels = f32_tensor(bundle, "media.pixels");
    let lengths = i64_tensor(bundle, "media.lengths");
    let owner = i64_tensor(bundle, "media.owner");
    pixels
        .chunks_exact(8 * 3)
        .enumerate()
        .flat_map(|(item, row)| {
            let request = usize::try_from(owner[item]).unwrap();
            let bias = lengths[item] as f32 * 10.0
                + owner[item] as f32 * 100.0
                + owner[item] as f32 * 1_000.0
                + marker[request] * 1_000.0
                + 1.0;
            row.iter().map(move |pixel| pixel * 2.0 + bias)
        })
        .collect()
}

fn expected_video(bundle: &GroupedVisionBundle, marker: &[f32]) -> Vec<f32> {
    let pixels = f32_tensor(bundle, "video.pixels");
    let lengths = i64_tensor(bundle, "video.lengths");
    let frame_owner = i64_tensor(bundle, "video.frame_owner");
    let clip_owner = i64_tensor(bundle, "video.clip_owner");
    pixels
        .chunks_exact(8 * 3)
        .enumerate()
        .flat_map(|(frame, row)| {
            let clip = usize::try_from(frame_owner[frame]).unwrap();
            let request = usize::try_from(clip_owner[clip]).unwrap();
            let bias = lengths[frame] as f32 * 10.0
                + frame_owner[frame] as f32 * 100.0
                + clip_owner[clip] as f32 * 1_000.0
                + marker[request] * 10_000.0
                + 1.0;
            row.iter().map(move |pixel| pixel * 2.0 + bias)
        })
        .collect()
}

fn image_requests<'a>(red: &'a [u8], blue: &'a [u8], green: &'a [u8]) -> [MediaRequest<'a>; 2] {
    [
        MediaRequest::new([MediaItem::single(red), MediaItem::single(blue)]),
        MediaRequest::new([MediaItem::single(green)]),
    ]
}

fn video_requests<'a>(
    red: &'a [u8],
    blue: &'a [u8],
    green: &'a [u8],
    yellow: &'a [u8],
    cyan: &'a [u8],
) -> [MediaRequest<'a>; 2] {
    [
        MediaRequest::new([MediaItem::nested([red, blue]), MediaItem::nested([green])]),
        MediaRequest::new([
            MediaItem::nested(std::iter::empty::<&'a [u8]>()),
            MediaItem::nested([yellow, cyan]),
        ]),
    ]
}

#[test]
fn image_requests_preprocess_group_execute_validate_and_split_in_order() -> anyhow::Result<()> {
    let root = build_package("image-e2e", FixtureKind::Image, &image_metadata(Some(32)))?;
    let mut engine = ort_engine(&root)?;
    let red = png(4, 2, [255, 0, 0]);
    let blue = png(2, 2, [0, 0, 255]);
    let green = png(2, 2, [0, 255, 0]);
    let requests = image_requests(&red, &blue, &green);
    let processor = processor(FixtureKind::Image);
    let local0 = processor.preprocess_encoded(&requests[..1])?;
    let local1 = processor.preprocess_encoded(&requests[1..])?;
    let grouped = processor.preprocess_encoded(&requests)?;

    assert_eq!(
        grouped.tensors.tensor("media.pixels").unwrap().shape,
        [3, 8, 3]
    );
    assert_eq!(i64_tensor(&grouped, "media.offsets"), [0, 2, 3]);
    assert_eq!(i64_tensor(&grouped, "media.owner"), [0, 0, 1]);
    assert_eq!(i64_tensor(&grouped, "media.lengths"), [8, 4, 4]);

    let inputs0 = component_inputs(&local0, &[5.0], 1, FixtureKind::Image)?;
    let inputs1 = component_inputs(&local1, &[7.0], 1, FixtureKind::Image)?;
    let groups =
        engine.group_workflow_component_inputs("encoder", &[(101, &inputs0), (202, &inputs1)])?;
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].sequence_ids(), [101, 202]);
    assert_eq!(groups[0].materialized_dimensions()["items"], 3);
    assert_eq!(groups[0].materialized_dimensions()["max_patches"], 8);

    let local_ownership = [
        ownership(&local0, FixtureKind::Image)?,
        ownership(&local1, FixtureKind::Image)?,
    ];
    let composed = ComposedOwnership::compose(&[&local_ownership[0], &local_ownership[1]])?;
    assert_eq!(
        composed.offsets(0).unwrap(),
        i64_tensor(&grouped, "media.offsets")
    );
    assert_eq!(
        composed.owners(0).unwrap(),
        i64_tensor(&grouped, "media.owner")
    );

    let marker = [5.0, 7.0];
    let mut outputs = engine.run_pipeline(workflow_request(&grouped, &marker, 1)?)?;
    let features = outputs.remove("features").expect("features output");
    let expected = expected_image(&grouped, &marker);
    assert_close(&features.to_vec_f32()?, &expected);

    let base = features.data_ptr_addr()?;
    let packed = composed.attach(features)?;
    let first = packed.request_view(0)?;
    let second = packed.request_view(1)?;
    assert_close(
        first.value().to_vec_f32()?.as_slice(),
        &expected[..2 * 8 * 3],
    );
    assert_close(
        second.value().to_vec_f32()?.as_slice(),
        &expected[2 * 8 * 3..],
    );
    assert_eq!(first.value().data_ptr_addr()?, base);
    assert_eq!(
        second.value().data_ptr_addr()?,
        base + 2 * 8 * 3 * std::mem::size_of::<f32>()
    );
    assert_eq!(
        second.ownership().levels()[0]
            .offsets()
            .iter()
            .collect::<Vec<_>>(),
        [0, 1]
    );
    Ok(())
}

#[test]
fn nested_video_preserves_empty_clips_and_decode_has_no_stale_media() -> anyhow::Result<()> {
    let root = build_package("video-e2e", FixtureKind::Video, VIDEO_METADATA)?;
    let mut engine = ort_engine(&root)?;
    let red = png(4, 2, [255, 0, 0]);
    let blue = png(2, 2, [0, 0, 255]);
    let green = png(2, 2, [0, 255, 0]);
    let yellow = png(4, 2, [255, 255, 0]);
    let cyan = png(2, 2, [0, 255, 255]);
    let requests = video_requests(&red, &blue, &green, &yellow, &cyan);
    let processor = processor(FixtureKind::Video);
    let local0 = processor.preprocess_encoded(&requests[..1])?;
    let local1 = processor.preprocess_encoded(&requests[1..])?;
    let grouped = processor.preprocess_encoded(&requests)?;

    assert_eq!(
        grouped.tensors.tensor("video.pixels").unwrap().shape,
        [5, 8, 3]
    );
    assert_eq!(i64_tensor(&grouped, "video.frame_offsets"), [0, 2, 3, 3, 5]);
    assert_eq!(i64_tensor(&grouped, "video.frame_owner"), [0, 0, 1, 3, 3]);
    assert_eq!(i64_tensor(&grouped, "video.clip_offsets"), [0, 2, 4]);
    assert_eq!(i64_tensor(&grouped, "video.clip_owner"), [0, 0, 1, 1]);
    assert_eq!(i64_tensor(&grouped, "video.lengths"), [8, 4, 4, 8, 4]);

    let inputs0 = component_inputs(&local0, &[2.0], 1, FixtureKind::Video)?;
    let inputs1 = component_inputs(&local1, &[3.0], 1, FixtureKind::Video)?;
    let groups =
        engine.group_workflow_component_inputs("encoder", &[(301, &inputs0), (302, &inputs1)])?;
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].sequence_ids(), [301, 302]);
    assert_eq!(groups[0].materialized_dimensions()["frames"], 5);
    assert_eq!(groups[0].materialized_dimensions()["clips"], 4);
    assert_eq!(groups[0].materialized_dimensions()["max_patches"], 8);

    let local_ownership = [
        ownership(&local0, FixtureKind::Video)?,
        ownership(&local1, FixtureKind::Video)?,
    ];
    let composed = ComposedOwnership::compose(&[&local_ownership[0], &local_ownership[1]])?;
    assert_eq!(
        composed.offsets(0).unwrap(),
        i64_tensor(&grouped, "video.frame_offsets")
    );
    assert_eq!(
        composed.owners(0).unwrap(),
        i64_tensor(&grouped, "video.frame_owner")
    );
    assert_eq!(
        composed.offsets(1).unwrap(),
        i64_tensor(&grouped, "video.clip_offsets")
    );
    assert_eq!(
        composed.owners(1).unwrap(),
        i64_tensor(&grouped, "video.clip_owner")
    );

    let marker = [2.0, 3.0];
    let mut outputs = engine.run_pipeline(workflow_request(&grouped, &marker, 1)?)?;
    let features = outputs.remove("features").expect("features output");
    let expected = expected_video(&grouped, &marker);
    assert_close(&features.to_vec_f32()?, &expected);
    let packed = composed.attach(features)?;
    let second = packed.request_view(1)?;
    assert_close(
        second.value().to_vec_f32()?.as_slice(),
        &expected[3 * 8 * 3..],
    );
    assert_eq!(
        second.ownership().levels()[0]
            .offsets()
            .iter()
            .collect::<Vec<_>>(),
        [0, 0, 2]
    );
    assert_eq!(
        second.ownership().levels()[0]
            .owners()
            .iter()
            .collect::<Vec<_>>(),
        [1, 1]
    );
    assert_eq!(
        second.ownership().levels()[1]
            .offsets()
            .iter()
            .collect::<Vec<_>>(),
        [0, 2]
    );

    let empty =
        processor.preprocess_encoded(&[MediaRequest::default(), MediaRequest::default()])?;
    let mut decode_outputs = engine.run_pipeline(workflow_request(&empty, &marker, 1)?)?;
    let decode_features = decode_outputs.remove("features").expect("decode features");
    assert_eq!(decode_features.shape(), [0, 8, 3]);
    assert!(decode_features.to_vec_f32()?.is_empty());
    assert_eq!(i64_tensor(&empty, "video.frame_offsets"), [0]);
    assert_eq!(i64_tensor(&empty, "video.clip_offsets"), [0, 0, 0]);
    assert!(i64_tensor(&empty, "video.frame_owner").is_empty());
    assert!(i64_tensor(&empty, "video.clip_owner").is_empty());
    assert!(i64_tensor(&empty, "video.lengths").is_empty());
    Ok(())
}

#[test]
fn invalid_companions_cardinality_and_capacity_fail_before_ort() -> anyhow::Result<()> {
    let normal_root = build_package(
        "image-fail-closed",
        FixtureKind::Image,
        &image_metadata(Some(32)),
    )?;
    let mut normal = ort_engine(&normal_root)?;
    let red = png(4, 2, [255, 0, 0]);
    let blue = png(2, 2, [0, 0, 255]);
    let green = png(2, 2, [0, 255, 0]);
    let requests = image_requests(&red, &blue, &green);
    let grouped = processor(FixtureKind::Image).preprocess_encoded(&requests)?;

    let mut malformed_offsets = workflow_request(&grouped, &[5.0, 7.0], 0)?;
    malformed_offsets
        .inputs
        .insert("offsets".into(), Value::from_slice_i64(&[0, 3, 2], &[3])?);
    let error = pipeline_error(
        normal.run_pipeline(malformed_offsets),
        "malformed offsets must fail before integer division by zero reaches ORT",
    );
    assert!(matches!(
        batch_contract_error(&error),
        Some(BatchContractError::InvalidOffset { .. })
    ));

    let mut malformed_owner = workflow_request(&grouped, &[5.0, 7.0], 0)?;
    malformed_owner
        .inputs
        .insert("owner".into(), Value::from_slice_i64(&[0, 1, 0], &[3])?);
    let error = pipeline_error(
        normal.run_pipeline(malformed_owner),
        "malformed owner must fail before ORT",
    );
    assert!(matches!(
        batch_contract_error(&error),
        Some(BatchContractError::OwnerOrder { .. })
    ));

    let mut cardinality = workflow_request(&grouped, &[5.0], 0)?;
    cardinality.inputs.insert(
        "request.marker".into(),
        Value::from_slice_f32(&[5.0], &[1])?,
    );
    let error = pipeline_error(
        normal.run_pipeline(cardinality),
        "request-count mismatch must fail before ORT",
    );
    assert!(matches!(
        batch_contract_error(&error),
        Some(BatchContractError::RequestCountMismatch { .. })
    ));

    let budget_root = build_package(
        "image-budget-overflow",
        FixtureKind::Image,
        &image_metadata(Some(23)),
    )?;
    let mut budget = ort_engine(&budget_root)?;
    let error = pipeline_error(
        budget.run_pipeline(workflow_request(&grouped, &[5.0, 7.0], 0)?),
        "3 x 8 materialized padding must exceed a budget of 23 before ORT",
    );
    assert!(matches!(
        batch_contract_error(&error),
        Some(BatchContractError::Admission {
            source: BatchAdmissionError::MaterializedBudgetExceeded {
                materialized: 24,
                max_total: 23,
                ..
            },
            ..
        })
    ));

    let undeclared_root = build_package(
        "image-undeclared-capacity",
        FixtureKind::Image,
        &image_metadata(None),
    )?;
    let mut undeclared = ort_engine(&undeclared_root)?;
    let error = pipeline_error(
        undeclared.run_pipeline(workflow_request(&grouped, &[5.0, 7.0], 0)?),
        "two requests without batch_capacity must fail before ORT",
    );
    assert!(matches!(
        batch_contract_error(&error),
        Some(BatchContractError::UndeclaredCapacity {
            request_count: 2,
            ..
        })
    ));
    Ok(())
}

#[cfg(feature = "native-backend")]
#[test]
fn native_grouped_encoder_fails_closed_without_ort_fallback() -> anyhow::Result<()> {
    let root = build_package(
        "image-native-fail-closed",
        FixtureKind::Image,
        &image_metadata(Some(32)),
    )?;
    let mut engine = Engine::from_dir(
        &root,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cpu),
            ..EngineConfig::default()
        },
    )?;
    let red = png(4, 2, [255, 0, 0]);
    let blue = png(2, 2, [0, 0, 255]);
    let green = png(2, 2, [0, 255, 0]);
    let requests = image_requests(&red, &blue, &green);
    let grouped = processor(FixtureKind::Image).preprocess_encoded(&requests)?;

    let error = pipeline_error(
        engine.run_pipeline(workflow_request(&grouped, &[5.0, 7.0], 1)?),
        "native grouped media execution is not independently proven",
    );
    assert!(matches!(
        batch_contract_error(&error),
        Some(BatchContractError::UnsupportedNativeEncoderBatch {
            request_count: 2,
            ..
        })
    ));
    assert_eq!(
        engine.native_component_run_count(),
        Some(0),
        "the native backend must reject before execution and must not fall back to ORT"
    );
    Ok(())
}

fn named_inputs(values: &PipelineTensors) -> Vec<(&str, &Value)> {
    values
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect()
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn measure(
    warmups: usize,
    iterations: usize,
    mut operation: impl FnMut() -> anyhow::Result<()>,
) -> anyhow::Result<Duration> {
    for _ in 0..warmups {
        operation()?;
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        operation()?;
        samples.push(start.elapsed());
    }
    Ok(median(samples))
}

fn single_row_inputs(
    bundle: &GroupedVisionBundle,
    row: usize,
    kind: FixtureKind,
) -> anyhow::Result<PipelineTensors> {
    let pixels_name = match kind {
        FixtureKind::Image => "media.pixels",
        FixtureKind::Video => "video.pixels",
    };
    let lengths_name = match kind {
        FixtureKind::Image => "media.lengths",
        FixtureKind::Video => "video.lengths",
    };
    let pixels = f32_tensor(bundle, pixels_name);
    let lengths = i64_tensor(bundle, lengths_name);
    let mut values = PipelineTensors::new();
    values.insert(
        "pixels".into(),
        Value::from_slice_f32(&pixels[row * 8 * 3..(row + 1) * 8 * 3], &[1, 8, 3])?,
    );
    values.insert(
        "lengths".into(),
        Value::from_slice_i64(&[lengths[row]], &[1])?,
    );
    values.insert("guard".into(), Value::from_slice_i64(&[1], &[1])?);
    match kind {
        FixtureKind::Image => {
            values.insert("offsets".into(), Value::from_slice_i64(&[0, 1], &[2])?);
            values.insert("owner".into(), Value::from_slice_i64(&[0], &[1])?);
            values.insert("marker".into(), Value::from_slice_f32(&[5.0], &[1])?);
        }
        FixtureKind::Video => {
            values.insert(
                "frame_offsets".into(),
                Value::from_slice_i64(&[0, 1], &[2])?,
            );
            values.insert("frame_owner".into(), Value::from_slice_i64(&[0], &[1])?);
            values.insert("clip_offsets".into(), Value::from_slice_i64(&[0, 1], &[2])?);
            values.insert("clip_owner".into(), Value::from_slice_i64(&[0], &[1])?);
            values.insert("marker".into(), Value::from_slice_f32(&[2.0], &[1])?);
        }
    }
    Ok(values)
}

fn report_fixture_performance(
    label: &str,
    session: &Session,
    engine: &Engine,
    grouped: &PipelineTensors,
    per_item: &[PipelineTensors],
) -> anyhow::Result<()> {
    let grouped_inputs = named_inputs(grouped);
    let item_inputs = per_item.iter().map(named_inputs).collect::<Vec<_>>();
    let dispatch_p50 = measure(8, 31, || {
        let requests = per_item
            .iter()
            .enumerate()
            .map(|(index, inputs)| (u64::try_from(index + 1).unwrap(), inputs))
            .collect::<Vec<_>>();
        black_box(engine.group_workflow_component_inputs("encoder", &requests)?);
        Ok(())
    })?;
    let per_item_p50 = measure(8, 31, || {
        for inputs in &item_inputs {
            black_box(session.run(inputs)?);
        }
        Ok(())
    })?;
    let grouped_p50 = measure(8, 31, || {
        black_box(session.run(&grouped_inputs)?);
        Ok(())
    })?;
    let count = per_item.len() as f64;
    let per_item_throughput = count / per_item_p50.as_secs_f64();
    let grouped_throughput = count / grouped_p50.as_secs_f64();
    assert!(per_item_p50 > Duration::ZERO);
    assert!(grouped_p50 > Duration::ZERO);
    assert!(per_item_throughput.is_finite());
    assert!(grouped_throughput.is_finite());
    eprintln!(
        "media_batch_fixture={label} items={} grouping_dispatch_p50_us={:.3} \
         per_item_ort_p50_us={:.3} grouped_ort_p50_us={:.3} \
         per_item_throughput_items_s={:.1} grouped_throughput_items_s={:.1}",
        per_item.len(),
        dispatch_p50.as_secs_f64() * 1e6,
        per_item_p50.as_secs_f64() * 1e6,
        grouped_p50.as_secs_f64() * 1e6,
        per_item_throughput,
        grouped_throughput,
    );
    Ok(())
}

#[test]
fn fixture_benchmark_reports_grouping_dispatch_and_ort_compute_separately() -> anyhow::Result<()> {
    let image_root = build_package(
        "image-benchmark",
        FixtureKind::Image,
        &image_metadata(Some(32)),
    )?;
    let image_engine = ort_engine(&image_root)?;
    let red = png(4, 2, [255, 0, 0]);
    let blue = png(2, 2, [0, 0, 255]);
    let green = png(2, 2, [0, 255, 0]);
    let image_bundle =
        processor(FixtureKind::Image).preprocess_encoded(&image_requests(&red, &blue, &green))?;
    let image_grouped = component_inputs(&image_bundle, &[5.0, 7.0], 1, FixtureKind::Image)?;
    let image_items = (0..3)
        .map(|row| single_row_inputs(&image_bundle, row, FixtureKind::Image))
        .collect::<anyhow::Result<Vec<_>>>()?;
    report_fixture_performance(
        "image-3x8x3",
        image_engine
            .models()?
            .session("encoder")
            .expect("encoder session"),
        &image_engine,
        &image_grouped,
        &image_items,
    )?;

    let video_root = build_package("video-benchmark", FixtureKind::Video, VIDEO_METADATA)?;
    let video_engine = ort_engine(&video_root)?;
    let yellow = png(4, 2, [255, 255, 0]);
    let cyan = png(2, 2, [0, 255, 255]);
    let video_bundle = processor(FixtureKind::Video)
        .preprocess_encoded(&video_requests(&red, &blue, &green, &yellow, &cyan))?;
    let video_grouped = component_inputs(&video_bundle, &[2.0, 3.0], 1, FixtureKind::Video)?;
    let video_items = (0..5)
        .map(|row| single_row_inputs(&video_bundle, row, FixtureKind::Video))
        .collect::<anyhow::Result<Vec<_>>>()?;
    report_fixture_performance(
        "video-5x8x3-4clips",
        video_engine
            .models()?
            .session("encoder")
            .expect("encoder session"),
        &video_engine,
        &video_grouped,
        &video_items,
    )?;
    Ok(())
}
