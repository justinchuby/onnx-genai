use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_genai_ort::PipelineModelDirectory;
use onnx_std::Model;
use onnx_std::ir::{DataType, Graph, Node, NodeId};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);
type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

struct FixtureDir(PathBuf);

impl FixtureDir {
    fn new(name: &str) -> TestResult<Self> {
        let id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/pipeline-admission-tests")
            .join(format!("{name}-{}-{id}", std::process::id()));
        if path.exists() {
            std::fs::remove_dir_all(&path)?;
        }
        std::fs::create_dir_all(&path)?;
        Ok(Self(path))
    }
}

impl Drop for FixtureDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone)]
struct Port<'a> {
    name: &'a str,
    dtype: DataType,
    shape: Vec<ShapeDim<'a>>,
}

#[derive(Clone)]
enum ShapeDim<'a> {
    Dynamic(&'a str),
    Static(usize),
}

fn symbolic_shape<'a>(names: &[&'a str], tail: usize) -> Vec<ShapeDim<'a>> {
    let mut shape = names
        .iter()
        .map(|name| ShapeDim::Dynamic(name))
        .collect::<Vec<_>>();
    shape.push(ShapeDim::Static(tail));
    shape
}

fn write_identity_model(
    path: &Path,
    inputs: Vec<Port<'_>>,
    outputs: Vec<(&str, &str)>,
) -> TestResult {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 13);
    let mut input_ids = std::collections::BTreeMap::new();
    for input in inputs {
        let shape = input
            .shape
            .into_iter()
            .map(|dim| match dim {
                ShapeDim::Dynamic(name) => graph.intern_symbol(name).into(),
                ShapeDim::Static(value) => value.into(),
            })
            .collect();
        let id = graph.create_named_value(input.name, input.dtype, shape);
        graph.add_input(id);
        input_ids.insert(input.name.to_string(), id);
    }
    for (output_name, input_name) in outputs {
        let input = input_ids[input_name];
        let input_value = graph.value(input);
        let output =
            graph.create_named_value(output_name, input_value.dtype, input_value.shape.clone());
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(input)],
            vec![output],
        ));
        graph.add_output(output);
    }
    let model = Model::new(graph);
    model.to_proto()?;
    onnx_std::save_model(&model, path)?;
    Ok(())
}

fn write_pipeline_models(root: &Path, decoder_per_layer: Port<'_>) -> TestResult {
    write_identity_model(
        &root.join("source.onnx"),
        vec![Port {
            name: "pixels",
            dtype: DataType::Float16,
            shape: symbolic_shape(&["batch", "sequence"], 8),
        }],
        vec![("features", "pixels")],
    )?;

    let token_shape = vec![ShapeDim::Dynamic("batch"), ShapeDim::Dynamic("sequence")];
    write_identity_model(
        &root.join("embedding.onnx"),
        vec![
            Port {
                name: "input_ids",
                dtype: DataType::Int64,
                shape: token_shape,
            },
            Port {
                name: "features",
                dtype: DataType::Float16,
                shape: symbolic_shape(&["batch", "sequence"], 8),
            },
        ],
        vec![("inputs_embeds", "features"), ("per_layer", "features")],
    )?;

    let attention_shape = vec![
        ShapeDim::Dynamic("batch"),
        ShapeDim::Dynamic("total_sequence"),
    ];
    write_identity_model(
        &root.join("decoder.onnx"),
        vec![
            Port {
                name: "inputs_embeds",
                dtype: DataType::Float16,
                shape: symbolic_shape(&["batch", "sequence"], 8),
            },
            decoder_per_layer,
            Port {
                name: "attention_mask",
                dtype: DataType::Int64,
                shape: attention_shape,
            },
        ],
        vec![("logits", "inputs_embeds")],
    )?;
    Ok(())
}

fn base_metadata(include_per_layer: bool, embedding_phase: &str) -> String {
    let per_layer_edge = include_per_layer.then_some(
        r#"
    - from: embedding.per_layer
      to: decoder.per_layer
      dtype: fp16
"#,
    );
    format!(
        r#"
pipeline:
  models:
    source:
      filename: source.onnx
      type: encoder
    embedding:
      filename: embedding.onnx
      type: encoder
      io:
        token_input: input_ids
    decoder:
      filename: decoder.onnx
      type: decoder
      io:
        inputs_embeds_input: inputs_embeds
        attention_mask_input: attention_mask
        logits_output: logits
  dataflow:
    - from: source.features
      to: embedding.features
      dtype: fp16
    - from: embedding.inputs_embeds
      to: decoder.inputs_embeds
      dtype: fp16
{}
  strategy:
    kind: composite
    stages:
      - name: source
        strategy:
          kind: single_pass
          model: source
        run_on: prompt_only
      - name: embedding
        strategy:
          kind: single_pass
          model: embedding
        run_on: {embedding_phase}
      - name: decode
        strategy:
          kind: autoregressive
          decoder: decoder
        run_on: every_step
  phases:
    source:
      run_on: prompt_only
    embedding:
      run_on: {embedding_phase}
    decoder:
      run_on: every_step
"#,
        per_layer_edge.unwrap_or("")
    )
}

fn write_metadata(root: &Path, metadata: &str) -> TestResult {
    std::fs::write(root.join("inference_metadata.yaml"), metadata)?;
    Ok(())
}

fn default_per_layer_port() -> Port<'static> {
    Port {
        name: "per_layer",
        dtype: DataType::Float16,
        shape: vec![
            ShapeDim::Dynamic("batch"),
            ShapeDim::Dynamic("sequence"),
            ShapeDim::Static(8),
        ],
    }
}

fn rejection(root: &Path) -> String {
    PipelineModelDirectory::load(root)
        .expect_err("pipeline admission must reject fixture")
        .to_string()
}

#[test]
fn admission_accepts_executable_multimodel_pipeline() -> TestResult {
    let fixture = FixtureDir::new("valid")?;
    write_pipeline_models(&fixture.0, default_per_layer_port())?;
    write_metadata(&fixture.0, &base_metadata(true, "every_step"))?;

    let directory = PipelineModelDirectory::load(&fixture.0)?;
    assert_eq!(directory.spec.models.len(), 3);
    Ok(())
}

#[test]
fn admission_rejects_unbound_decoder_input() -> TestResult {
    let fixture = FixtureDir::new("unbound")?;
    write_pipeline_models(&fixture.0, default_per_layer_port())?;
    write_metadata(&fixture.0, &base_metadata(false, "every_step"))?;

    let error = rejection(&fixture.0);
    assert!(error.contains("decoder.per_layer"), "{error}");
    assert!(error.contains("unbound"), "{error}");
    assert!(error.contains("regenerate the native sidecar"), "{error}");
    Ok(())
}

#[test]
fn admission_accepts_cached_prompt_only_conditioning() -> TestResult {
    let fixture = FixtureDir::new("cached-conditioning")?;
    write_identity_model(
        &fixture.0.join("conditioning.onnx"),
        vec![Port {
            name: "image_features",
            dtype: DataType::Float16,
            shape: symbolic_shape(&["batch", "image_sequence"], 8),
        }],
        vec![("conditioning", "image_features")],
    )?;
    write_identity_model(
        &fixture.0.join("embedding.onnx"),
        vec![Port {
            name: "seed_embeddings",
            dtype: DataType::Float16,
            shape: symbolic_shape(&["batch", "sequence"], 8),
        }],
        vec![("inputs_embeds", "seed_embeddings")],
    )?;
    write_identity_model(
        &fixture.0.join("decoder.onnx"),
        vec![
            Port {
                name: "inputs_embeds",
                dtype: DataType::Float16,
                shape: symbolic_shape(&["batch", "sequence"], 8),
            },
            Port {
                name: "conditioning",
                dtype: DataType::Float16,
                shape: symbolic_shape(&["batch", "image_sequence"], 8),
            },
            Port {
                name: "position",
                dtype: DataType::Int64,
                shape: vec![ShapeDim::Dynamic("sequence")],
            },
        ],
        vec![("logits", "inputs_embeds")],
    )?;
    write_metadata(
        &fixture.0,
        r#"
pipeline:
  models:
    conditioning:
      filename: conditioning.onnx
      type: encoder
    embedding:
      filename: embedding.onnx
      type: encoder
    decoder:
      filename: decoder.onnx
      type: decoder
      io:
        inputs_embeds_input: inputs_embeds
        position_ids_input: position
        logits_output: logits
  dataflow:
    - from: conditioning.conditioning
      to: decoder.conditioning
      dtype: fp16
    - from: embedding.inputs_embeds
      to: decoder.inputs_embeds
      dtype: fp16
  strategy:
    kind: composite
    stages:
      - name: conditioning
        strategy:
          kind: single_pass
          model: conditioning
        run_on: prompt_only
      - name: embedding
        strategy:
          kind: single_pass
          model: embedding
        run_on: every_step
      - name: decode
        strategy:
          kind: autoregressive
          decoder: decoder
        run_on: every_step
  phases:
    conditioning:
      run_on: prompt_only
    embedding:
      run_on: every_step
    decoder:
      run_on: every_step
  positions:
    input: position
    rank: 1
    axes: [sequence]
    dtype: int64
"#,
    )?;

    PipelineModelDirectory::load(&fixture.0)?;
    Ok(())
}

#[test]
fn admission_accepts_mixed_dataflow_and_external_inputs() -> TestResult {
    let fixture = FixtureDir::new("mixed-provenance")?;
    write_identity_model(
        &fixture.0.join("producer.onnx"),
        vec![Port {
            name: "seed",
            dtype: DataType::Float32,
            shape: vec![ShapeDim::Dynamic("batch"), ShapeDim::Static(4)],
        }],
        vec![("routed", "seed")],
    )?;
    write_identity_model(
        &fixture.0.join("consumer.onnx"),
        vec![
            Port {
                name: "routed",
                dtype: DataType::Float32,
                shape: vec![ShapeDim::Dynamic("batch"), ShapeDim::Static(4)],
            },
            Port {
                name: "request_context",
                dtype: DataType::Float32,
                shape: vec![ShapeDim::Dynamic("batch"), ShapeDim::Static(4)],
            },
        ],
        vec![("output", "routed")],
    )?;
    write_metadata(
        &fixture.0,
        r#"
pipeline:
  models:
    producer:
      filename: producer.onnx
      type: encoder
    consumer:
      filename: consumer.onnx
      type: encoder
  dataflow:
    - from: producer.routed
      to: consumer.routed
      dtype: fp32
  strategy:
    kind: composite
    stages:
      - name: producer
        strategy:
          kind: single_pass
          model: producer
        run_on: prompt_only
      - name: consumer
        strategy:
          kind: single_pass
          model: consumer
        run_on: prompt_only
  phases:
    producer:
      run_on: prompt_only
    consumer:
      run_on: prompt_only
"#,
    )?;

    PipelineModelDirectory::load(&fixture.0)?;
    Ok(())
}

#[test]
fn admission_rejects_convention_looking_undeclared_input() -> TestResult {
    let fixture = FixtureDir::new("undeclared-past-noise")?;
    write_identity_model(
        &fixture.0.join("decoder.onnx"),
        vec![
            Port {
                name: "input_ids",
                dtype: DataType::Int64,
                shape: vec![ShapeDim::Dynamic("batch"), ShapeDim::Dynamic("sequence")],
            },
            Port {
                name: "past_noise",
                dtype: DataType::Float16,
                shape: symbolic_shape(&["batch", "sequence"], 8),
            },
        ],
        vec![("logits", "past_noise")],
    )?;
    write_metadata(
        &fixture.0,
        r#"
pipeline:
  models:
    decoder:
      filename: decoder.onnx
      type: decoder
      io:
        token_input: input_ids
        logits_output: logits
  dataflow: []
  strategy:
    kind: autoregressive
    decoder: decoder
  phases:
    decoder:
      run_on: every_step
"#,
    )?;

    let error = rejection(&fixture.0);
    assert!(error.contains("decoder.past_noise"), "{error}");
    assert!(error.contains("unbound"), "{error}");
    assert!(error.contains("exactly one declared source"), "{error}");
    Ok(())
}

#[test]
fn admission_preserves_component_model_load_context() -> TestResult {
    let fixture = FixtureDir::new("invalid-model")?;
    std::fs::write(fixture.0.join("decoder.onnx"), b"not an ONNX model")?;
    write_metadata(
        &fixture.0,
        r#"
pipeline:
  models:
    decoder:
      filename: decoder.onnx
      type: decoder
      io:
        token_input: input_ids
        logits_output: logits
  dataflow: []
  strategy:
    kind: autoregressive
    decoder: decoder
"#,
    )?;

    let error = rejection(&fixture.0);
    assert!(error.contains("component 'decoder'"), "{error}");
    assert!(error.contains("decoder.onnx"), "{error}");
    assert!(error.contains("could not be loaded"), "{error}");
    assert!(error.contains("How to fix"), "{error}");
    Ok(())
}

#[test]
fn admission_rejects_dataflow_dtype_mismatch() -> TestResult {
    let fixture = FixtureDir::new("dtype")?;
    let mut port = default_per_layer_port();
    port.dtype = DataType::Float32;
    write_pipeline_models(&fixture.0, port)?;
    write_metadata(&fixture.0, &base_metadata(true, "every_step"))?;

    let error = rejection(&fixture.0);
    assert!(error.contains("decoder.per_layer"), "{error}");
    assert!(error.contains("incompatible dtypes"), "{error}");
    assert!(
        error.contains("producer is float16, consumer is float32"),
        "{error}"
    );
    Ok(())
}

#[test]
fn admission_rejects_dataflow_rank_mismatch() -> TestResult {
    let fixture = FixtureDir::new("rank")?;
    let mut port = default_per_layer_port();
    port.shape = vec![ShapeDim::Dynamic("batch"), ShapeDim::Dynamic("sequence")];
    write_pipeline_models(&fixture.0, port)?;
    write_metadata(&fixture.0, &base_metadata(true, "every_step"))?;

    let error = rejection(&fixture.0);
    assert!(error.contains("decoder.per_layer"), "{error}");
    assert!(error.contains("incompatible ranks"), "{error}");
    assert!(
        error.contains("producer rank 3, consumer rank 2"),
        "{error}"
    );
    Ok(())
}

#[test]
fn admission_rejects_unconstructable_declared_image_endpoint() -> TestResult {
    let fixture = FixtureDir::new("modality")?;
    write_identity_model(
        &fixture.0.join("vision.onnx"),
        vec![Port {
            name: "pixel_values",
            dtype: DataType::Float16,
            shape: symbolic_shape(&["batch", "patches"], 8),
        }],
        vec![("image_features", "pixel_values")],
    )?;
    write_metadata(
        &fixture.0,
        r#"
pipeline:
  models:
    vision_encoder:
      filename: vision.onnx
      type: vision_encoder
  dataflow: []
  strategy:
    kind: single_pass
    model: vision_encoder
  phases:
    vision_encoder:
      run_on: prompt_only
  vision:
    image_placeholder_token_id: 7
    tokens_per_tile: 4
"#,
    )?;

    let error = rejection(&fixture.0);
    assert!(error.contains("vision_encoder.pixel_values"), "{error}");
    assert!(error.contains("cannot be constructed"), "{error}");
    assert!(error.contains("preprocessing.image"), "{error}");
    Ok(())
}
