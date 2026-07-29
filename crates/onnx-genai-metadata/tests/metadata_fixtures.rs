use std::path::{Path, PathBuf};

use onnx_genai_metadata::{
    InferenceMetadata, MetadataError, PipelineStrategyKind, RuntimeCapabilities,
    SequenceLengthScalarBroadcast, load_metadata, load_pipeline_spec, validate,
    validate_pipeline_spec,
};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures")
        .join(name)
}

fn crate_fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn example_model(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("examples")
        .join(name)
        .join("inference_metadata.yaml")
}

fn example(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("examples")
        .join(name)
}

#[test]
fn parses_valid_yaml_fixture() {
    let metadata =
        load_metadata(&fixture("sample_metadata.yaml")).expect("valid YAML fixture parses");

    assert_eq!(
        metadata.required_capabilities,
        ["kv_cache", "grouped_query_attention"]
    );

    let model = metadata.model.expect("model section");
    let attention = model.attention.expect("attention section");
    assert_eq!(attention.attention_type, "grouped_query");
    assert_eq!(attention.num_kv_heads, Some(8));
    assert_eq!(attention.num_attention_heads, Some(32));
    assert_eq!(attention.head_dim, Some(128));
    assert_eq!(model.max_sequence_length, Some(131_072));

    let kv_cache = metadata.kv_cache.expect("kv_cache section");
    assert_eq!(kv_cache.native_dtype.as_deref(), Some("bf16"));
    assert_eq!(
        kv_cache.sensitive_layers.as_deref(),
        Some([0, 1, -1].as_slice())
    );

    let structured_output = metadata
        .structured_output
        .expect("structured_output section");
    assert_eq!(
        structured_output.supported_formats.as_deref(),
        Some(
            [
                "json_schema".to_string(),
                "regex".to_string(),
                "context_free_grammar".to_string()
            ]
            .as_slice()
        )
    );
}

#[test]
fn parses_valid_json_fixture() {
    let metadata =
        load_metadata(&fixture("sample_metadata.json")).expect("valid JSON fixture parses");

    assert_eq!(
        metadata.required_capabilities,
        ["kv_cache", "grouped_query_attention"]
    );
    let runtime_configurable = metadata
        .model
        .expect("model section")
        .runtime_configurable
        .expect("runtime_configurable section");
    assert_eq!(runtime_configurable.prefix_cache, Some(true));
    assert_eq!(runtime_configurable.continuous_batching, Some(true));
    assert_eq!(
        runtime_configurable
            .kv_cache
            .expect("kv_cache config")
            .dtype,
        ["fp16", "fp8_e5m2"]
    );
}

#[test]
fn parses_top_level_generation_defaults_from_v1_yaml() {
    let yaml = r#"
schema_version: v1
generation:
  do_sample: true
  temperature: 0.6
  top_k: 20
  top_p: 0.95
  repetition_penalty: 1.0
  num_beams: 1
"#;
    let metadata: InferenceMetadata = serde_yaml::from_str(yaml).expect("metadata parses");
    let generation = metadata.generation.expect("generation section");

    assert_eq!(generation.do_sample, Some(true));
    assert_eq!(generation.temperature, Some(0.6));
    assert_eq!(generation.top_k, Some(20));
    assert_eq!(generation.top_p, Some(0.95));
    assert_eq!(generation.repetition_penalty, Some(1.0));
    assert_eq!(generation.num_beams, Some(1));
}

#[test]
fn attention_key_sequence_lengths_unit_batch_parses_and_round_trips() {
    let yaml = r#"
model:
  attention:
    type: grouped_query
    key_sequence_lengths:
      scalar_broadcast: unit_batch
"#;
    let metadata: InferenceMetadata = serde_yaml::from_str(yaml).expect("metadata parses");
    let scalar_broadcast = metadata
        .model
        .as_ref()
        .and_then(|model| model.attention.as_ref())
        .and_then(|attention| attention.key_sequence_lengths.as_ref())
        .and_then(|spec| spec.scalar_broadcast);
    assert_eq!(
        scalar_broadcast,
        Some(SequenceLengthScalarBroadcast::UnitBatch)
    );

    let value: serde_json::Value =
        serde_yaml::from_str(yaml).expect("YAML converts to a generic value");
    let round_tripped: InferenceMetadata =
        serde_json::from_value(value).expect("JSON value round-trips");
    assert_eq!(
        round_tripped
            .model
            .and_then(|model| model.attention)
            .and_then(|attention| attention.key_sequence_lengths)
            .and_then(|spec| spec.scalar_broadcast),
        Some(SequenceLengthScalarBroadcast::UnitBatch)
    );
}

#[test]
fn attention_key_sequence_lengths_absence_is_strict_default() {
    let metadata: InferenceMetadata = serde_yaml::from_str(
        r#"
model:
  attention:
    type: grouped_query
"#,
    )
    .expect("metadata parses");
    assert!(
        metadata
            .model
            .and_then(|model| model.attention)
            .and_then(|attention| attention.key_sequence_lengths)
            .is_none()
    );
}

#[test]
fn mixture_of_experts_contract_parses_structurally() {
    let yaml = r#"
model:
  mixture_of_experts:
    representation: dense_fallback
    routed_expert_count: 8
    shared_expert_count: 1
    experts_per_token: 2
    expert_intermediate_size: 256
    shared_expert_intermediate_size: 256
    activation: silu
    router:
      score_function: sigmoid
      selection_method: grouped_top_k
      normalize_weights: true
      scaling_factor: 2.5
      group_count: 4
      groups_per_token: 2
      group_score: top_2_sum
"#;
    let metadata: InferenceMetadata = serde_yaml::from_str(yaml).expect("MoE metadata parses");
    let moe = metadata
        .model
        .expect("model section")
        .mixture_of_experts
        .expect("mixture_of_experts section");

    assert_eq!(moe.representation, "dense_fallback");
    assert_eq!(moe.routed_expert_count, 8);
    assert_eq!(moe.shared_expert_count, 1);
    assert_eq!(moe.experts_per_token, 2);
    assert_eq!(moe.expert_intermediate_size, 256);
    assert_eq!(moe.shared_expert_intermediate_size, 256);
    assert_eq!(moe.router.score_function, "sigmoid");
    assert_eq!(moe.router.selection_method, "grouped_top_k");
    assert_eq!(moe.router.group_count, Some(4));
    assert_eq!(moe.router.groups_per_token, Some(2));
    assert_eq!(moe.router.group_score.as_deref(), Some("top_2_sum"));

    let value: serde_json::Value = serde_yaml::from_str(yaml).expect("MoE YAML converts to JSON");
    schema_validator()
        .validate(&value)
        .expect("complete grouped router validates against the schema");
}

#[test]
fn mixture_of_experts_grouped_router_requires_group_contract() {
    let value = serde_json::json!({
        "model": {
            "mixture_of_experts": {
                "representation": "dense_fallback",
                "routed_expert_count": 8,
                "shared_expert_count": 0,
                "experts_per_token": 2,
                "expert_intermediate_size": 256,
                "shared_expert_intermediate_size": 0,
                "activation": "silu",
                "router": {
                    "score_function": "sigmoid",
                    "selection_method": "grouped_top_k",
                    "normalize_weights": true,
                    "scaling_factor": 1.0
                }
            }
        }
    });

    assert!(
        schema_validator().validate(&value).is_err(),
        "grouped_top_k without group dimensions must fail schema validation"
    );
}

#[test]
fn attention_key_sequence_lengths_rejects_unknown_scalar_broadcast() {
    let error = serde_yaml::from_str::<InferenceMetadata>(
        r#"
model:
  attention:
    type: grouped_query
    key_sequence_lengths:
      scalar_broadcast: every_batch
"#,
    )
    .expect_err("unknown compatibility mode must fail");
    assert!(error.to_string().contains("unit_batch"));
}

#[test]
fn malformed_yaml_returns_parse_error() {
    let err = load_metadata(&fixture("malformed_metadata.yaml")).expect_err("malformed YAML fails");

    assert!(matches!(err, MetadataError::Parse(_)));
    assert!(!err.to_string().is_empty());
}

#[test]
fn schema_type_mismatch_returns_parse_error() {
    let err = load_metadata(&fixture("invalid_metadata.yaml")).expect_err("invalid schema fails");

    assert!(matches!(err, MetadataError::Parse(_)));
    assert!(err.to_string().contains("Parse error"));
}

#[test]
fn capability_validation_accepts_default_runtime_capabilities() {
    let metadata =
        load_metadata(&fixture("sample_metadata.yaml")).expect("valid YAML fixture parses");

    validate(&metadata, &RuntimeCapabilities::default()).expect("default runtime supports fixture");
}

#[test]
fn capability_validation_reports_all_unsupported_capabilities() {
    let mut metadata =
        load_metadata(&fixture("sample_metadata.yaml")).expect("valid YAML fixture parses");
    metadata
        .required_capabilities
        .push("vision_encoder".to_string());
    metadata
        .required_capabilities
        .push("speculative_decoding".to_string());

    let unsupported = validate(&metadata, &RuntimeCapabilities::default())
        .expect_err("unsupported capabilities are reported");

    assert_eq!(unsupported, ["vision_encoder", "speculative_decoding"]);
}

#[test]
fn capability_validation_uses_runtime_supported_set() {
    let metadata =
        load_metadata(&fixture("sample_metadata.yaml")).expect("valid YAML fixture parses");
    let runtime = RuntimeCapabilities {
        supported: vec!["kv_cache".to_string()],
    };

    let unsupported = validate(&metadata, &runtime).expect_err("missing GQA support is reported");

    assert_eq!(unsupported, ["grouped_query_attention"]);
}

#[test]
fn parses_and_validates_pipeline_fixture() {
    let spec = load_pipeline_spec(&crate_fixture("pipeline_valid.yaml"))
        .expect("valid pipeline fixture parses and validates");

    assert_eq!(spec.models.len(), 2);
    assert_eq!(
        spec.models["vision_encoder"].filename,
        "vision_encoder.onnx"
    );
    assert_eq!(spec.models["decoder"].role, "decoder");
    assert_eq!(
        spec.models["decoder"].tokenizer.as_deref(),
        Some("tokenizer.json")
    );
    assert_eq!(spec.dataflow[0].from, "vision_encoder.image_features");
    assert!(matches!(
        spec.strategy.kind,
        PipelineStrategyKind::Composite
    ));
    assert_eq!(spec.strategy.stages.len(), 2);
}

#[test]
fn nemotron_audio_pipeline_example_parses_and_validates() {
    let path = example("nemotron-audio/inference_metadata.yaml");
    let metadata = load_metadata(&path).expect("Nemotron audio example parses");
    let pipeline = metadata.pipeline.expect("pipeline section");

    let instance: serde_json::Value =
        serde_yaml::from_str(&std::fs::read_to_string(path).expect("example is readable"))
            .expect("example converts to JSON");
    schema_validator()
        .validate(&instance)
        .expect("Nemotron audio example validates against the JSON schema");
    validate_pipeline_spec(&pipeline).expect("Nemotron audio pipeline is structurally valid");
    assert_eq!(pipeline.models["audio_encoder"].role, "audio_encoder");
    assert_eq!(
        pipeline.models["audio_encoder"]
            .io
            .as_ref()
            .and_then(|io| io.state_pairs.as_ref())
            .map(Vec::len),
        Some(3)
    );
    assert_eq!(
        pipeline.models["prediction_network"]
            .io
            .as_ref()
            .and_then(|io| io.state_pairs.as_ref())
            .map(Vec::len),
        Some(2)
    );
    assert_eq!(pipeline.dataflow.len(), 3);
    assert!(
        !pipeline.dataflow.iter().any(|edge| {
            edge.from == "prediction_network.decoder_output" && edge.to == "joiner.decoder_output"
        }),
        "the schema cannot express the required decoder-output transpose"
    );
    assert!(matches!(
        pipeline.strategy.kind,
        PipelineStrategyKind::Composite
    ));
}

#[test]
fn pipeline_validation_rejects_dangling_edges() {
    let metadata = load_metadata(&crate_fixture("pipeline_dangling.yaml"))
        .expect("fixture parses structurally");
    let spec = metadata.pipeline.expect("pipeline section");
    let err = validate_pipeline_spec(&spec).expect_err("dangling component is rejected");

    assert!(
        err.errors
            .iter()
            .any(|error| error.contains("unknown component"))
    );
}

#[test]
fn pipeline_validation_rejects_cycles() {
    let metadata =
        load_metadata(&crate_fixture("pipeline_cycle.yaml")).expect("fixture parses structurally");
    let spec = metadata.pipeline.expect("pipeline section");
    let err = validate_pipeline_spec(&spec).expect_err("cycle is rejected");

    assert!(
        err.errors
            .iter()
            .any(|error| error.contains("contains a cycle"))
    );
}

#[test]
fn pipeline_validation_accepts_iterative_denoiser_self_edge() {
    // A denoiser fed its own previous-step output (`denoiser.x -> denoiser.y`)
    // is a loop-carried temporal dependency, not a same-step DAG cycle, so it
    // must validate cleanly.
    let spec = load_pipeline_spec(&crate_fixture("pipeline_iterative_self_edge.yaml"))
        .expect("iterative pipeline with a denoiser self-edge validates");

    assert!(matches!(
        spec.strategy.kind,
        PipelineStrategyKind::Iterative
    ));
    assert_eq!(spec.strategy.denoiser.as_deref(), Some("denoiser"));
    assert_eq!(spec.strategy.num_steps, Some(20));
}

#[test]
fn pipeline_validation_accepts_full_diffusion_txt2img_contract() {
    // The hardest generality case: a NON-autoregressive text-to-image latent
    // diffusion pipeline (condition encoder -> scheduler-driven iterative
    // denoiser -> VAE decoder) must parse AND validate through the crate. This
    // is the committed example under `examples/diffusion-metadata/`.
    let spec = load_pipeline_spec(&crate_fixture("pipeline_diffusion_txt2img.yaml"))
        .expect("full diffusion txt2img pipeline parses and validates");

    // Three components: condition encoder, iterative denoiser, VAE decoder.
    assert_eq!(spec.models.len(), 3);
    assert_eq!(spec.models["text_encoder"].role, "encoder");
    assert_eq!(spec.models["denoiser"].role, "denoiser");
    assert_eq!(spec.models["vae_decoder"].role, "vae");

    // Scheduler-driven iterative (diffusion) strategy over N steps.
    let strategy = &spec.strategy;
    assert!(matches!(strategy.kind, PipelineStrategyKind::Iterative));
    assert_eq!(strategy.denoiser.as_deref(), Some("denoiser"));
    assert_eq!(strategy.num_steps, Some(25));
    assert_eq!(strategy.timestep_input.as_deref(), Some("timestep"));

    // Classifier-free guidance is expressible: scale + conditioning port.
    assert_eq!(strategy.guidance_scale, Some(7.5));
    assert_eq!(
        strategy.cfg_conditioning_input.as_deref(),
        Some("encoder_hidden_states")
    );

    // The diffusion scheduler loop contract: sampler kind, noise schedule,
    // prediction parameterization, and Karras sigma spacing.
    let scheduler = strategy
        .scheduler_config
        .as_ref()
        .expect("scheduler_config present");
    assert_eq!(scheduler.kind, "euler");
    assert_eq!(scheduler.num_train_timesteps, Some(1000));
    assert_eq!(scheduler.beta_schedule.as_deref(), Some("scaled_linear"));
    assert_eq!(scheduler.prediction_type.as_deref(), Some("epsilon"));
    assert_eq!(scheduler.use_karras_sigmas, Some(true));

    // The loop-carried latent self-edge and the encoder->denoiser->VAE dataflow
    // are all present and validated (no dangling ports, no same-step cycle).
    assert!(
        spec.dataflow
            .iter()
            .any(|edge| edge.from == "denoiser.out_sample" && edge.to == "denoiser.sample"),
        "loop-carried latent self-edge must be declared"
    );
    assert!(
        spec.dataflow
            .iter()
            .any(|edge| edge.from == "denoiser.sample" && edge.to == "vae_decoder.latent_sample"),
        "final latent must route to the VAE decoder"
    );
}

#[test]
fn pipeline_validation_rejects_self_edge_outside_iterative_denoiser() {
    // A self-edge on a component that is NOT an iterative denoiser has no loop
    // semantics and must be rejected as a cycle.
    let yaml = "
pipeline:
  models:
    encoder:
      filename: encoder.onnx
      type: encoder
    decoder:
      filename: decoder.onnx
      type: decoder
  dataflow:
    - from: encoder.state
      to: encoder.state
  strategy:
    kind: autoregressive
    decoder: decoder
";
    let metadata: onnx_genai_metadata::InferenceMetadata =
        serde_yaml::from_str(yaml).expect("parses");
    let spec = metadata.pipeline.expect("pipeline section");
    let err = validate_pipeline_spec(&spec).expect_err("non-denoiser self-edge is rejected");
    assert!(
        err.errors.iter().any(|e| e.contains("contains a cycle")),
        "unexpected errors: {:?}",
        err.errors
    );
}

#[test]
fn pipeline_validation_rejects_duplicate_destination_edges() {
    // Two producers feeding one destination port is ambiguous and rejected.
    let yaml = "
pipeline:
  models:
    encoder_a:
      filename: a.onnx
      type: encoder
    encoder_b:
      filename: b.onnx
      type: encoder
    decoder:
      filename: decoder.onnx
      type: decoder
  dataflow:
    - from: encoder_a.hidden
      to: decoder.encoder_hidden_states
    - from: encoder_b.hidden
      to: decoder.encoder_hidden_states
  strategy:
    kind: autoregressive
    decoder: decoder
";
    let metadata: onnx_genai_metadata::InferenceMetadata =
        serde_yaml::from_str(yaml).expect("parses");
    let spec = metadata.pipeline.expect("pipeline section");
    let err = validate_pipeline_spec(&spec).expect_err("duplicate destination is rejected");
    assert!(
        err.errors
            .iter()
            .any(|e| e.contains("multiple edges into the same destination")),
        "unexpected errors: {:?}",
        err.errors
    );
}

#[test]
fn pipeline_vision_config_round_trips_via_json() {
    use onnx_genai_metadata::PipelineVisionConfig;

    let json = r#"{"image_placeholder_token_id": 32000, "tokens_per_tile": 256}"#;
    let decoded: PipelineVisionConfig = serde_json::from_str(json).expect("deserializes");
    assert_eq!(decoded.image_placeholder_token_id, Some(32000_i64));
    assert_eq!(decoded.tokens_per_tile, Some(256_usize));

    let round_tripped: PipelineVisionConfig = serde_json::from_str(
        &serde_json::to_string(&serde_json::from_str::<serde_json::Value>(json).unwrap()).unwrap(),
    )
    .expect("value round-trip");
    assert_eq!(round_tripped, decoded);
}

#[test]
fn pipeline_vision_config_round_trips_via_yaml() {
    use onnx_genai_metadata::PipelineVisionConfig;

    let yaml = "image_placeholder_token_id: -1\ntokens_per_tile: 64\n";
    let decoded: PipelineVisionConfig = serde_yaml::from_str(yaml).expect("deserializes");
    assert_eq!(decoded.image_placeholder_token_id, Some(-1_i64));
    assert_eq!(decoded.tokens_per_tile, Some(64_usize));
}

#[test]
fn pipeline_vision_config_optional_fields_round_trip_to_none() {
    use onnx_genai_metadata::PipelineVisionConfig;

    let json = r#"{}"#;
    let decoded: PipelineVisionConfig = serde_json::from_str(json).expect("deserializes");
    assert_eq!(decoded.image_placeholder_token_id, None);
    assert_eq!(decoded.tokens_per_tile, None);
}

#[test]
fn pipeline_spec_vision_field_round_trips() {
    use onnx_genai_metadata::PipelineVisionConfig;

    let yaml = "
pipeline:
  models:
    decoder:
      filename: decoder.onnx
      type: decoder
      tokenizer: tokenizer.json
  strategy:
    kind: autoregressive
    decoder: decoder
  vision:
    image_placeholder_token_id: 32000
    tokens_per_tile: 256
";
    let metadata: onnx_genai_metadata::InferenceMetadata =
        serde_yaml::from_str(yaml).expect("parses");
    let spec = metadata.pipeline.expect("pipeline section");
    let vision = spec.vision.expect("vision section");
    assert_eq!(
        vision,
        PipelineVisionConfig {
            image_placeholder_token_id: Some(32000),
            tokens_per_tile: Some(256),
            ..Default::default()
        }
    );
}

#[test]
fn pipeline_validation_rejects_timesteps_length_mismatch() {
    let yaml = "
pipeline:
  models:
    denoiser:
      filename: denoiser.onnx
      type: denoiser
  dataflow:
    - from: denoiser.out
      to: denoiser.sample
  strategy:
    kind: iterative
    denoiser: denoiser
    num_steps: 3
    timestep_input: t
    timesteps: [0.1, 0.2]
";
    let metadata: onnx_genai_metadata::InferenceMetadata =
        serde_yaml::from_str(yaml).expect("parses");
    let spec = metadata.pipeline.expect("pipeline section");
    let err = validate_pipeline_spec(&spec).expect_err("timesteps length mismatch is rejected");
    assert!(
        err.errors
            .iter()
            .any(|e| e.contains("timesteps has 2 entries")),
        "unexpected errors: {:?}",
        err.errors
    );
}

/// Compiles the committed JSON schema so fixtures can be validated against it.
fn schema_validator() -> jsonschema::Validator {
    let schema_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("schema/inference_metadata.schema.json");
    let schema_text = std::fs::read_to_string(&schema_path).expect("read committed JSON schema");
    let schema: serde_json::Value =
        serde_json::from_str(&schema_text).expect("committed JSON schema is valid JSON");
    jsonschema::validator_for(&schema).expect("committed JSON schema compiles")
}

/// Loads a crate-local YAML fixture as a JSON value for schema validation.
fn fixture_json(name: &str) -> serde_json::Value {
    let text = std::fs::read_to_string(crate_fixture(name)).expect("read fixture");
    serde_yaml::from_str(&text).expect("fixture parses as a structured document")
}

#[test]
fn vlm_packed_fixture_deserializes_and_validates_against_schema() {
    // Rust deserializer round-trip.
    let metadata = load_metadata(&crate_fixture("vlm_packed_valid.yaml"))
        .expect("packed VLM fixture deserializes into the typed contract");

    let preprocessing = metadata.preprocessing.expect("preprocessing section");
    let image = preprocessing.image.expect("image program");
    assert_eq!(
        image.transforms.first().expect("first transform").op,
        "decode"
    );
    assert!(
        image.transforms.iter().any(|t| t.op == "patchify"),
        "program declares a generic patchify op"
    );
    let declared_ops = image
        .transforms
        .iter()
        .map(|transform| transform.op.as_str())
        .collect::<Vec<_>>();
    for required_op in [
        "decode",
        "convert_rgb",
        "resize",
        "rescale",
        "normalize",
        "tile",
        "flatten",
        "patchify",
        "pad",
        "emit_original_size",
        "emit_validity_mask",
        "emit_patch_coordinates",
        "emit_grid_coordinates",
    ] {
        assert!(
            declared_ops.contains(&required_op),
            "packed fixture must declare generic operation {required_op}"
        );
    }

    // Exactly two packed image outputs, bound to arbitrary endpoint names.
    assert_eq!(image.outputs.len(), 2);
    assert_eq!(image.outputs[0].source.as_deref(), Some("padded_patches"));
    assert_eq!(image.outputs[0].name, "vision_encoder.pixel_values");
    assert_eq!(image.outputs[0].content, "pixels");
    assert_eq!(image.outputs[0].dtype, "float32");
    assert_eq!(
        image.outputs[1].source.as_deref(),
        Some("padded_patch_coordinates")
    );
    assert_eq!(image.outputs[1].name, "vision_encoder.pixel_position_ids");
    assert_eq!(image.outputs[1].content, "patch_coordinates");
    assert_eq!(image.outputs[1].dtype, "int64");
    assert_eq!(image.outputs[1].pad_value, Some(-1.0));

    let spec = metadata.pipeline.expect("pipeline section");
    validate_pipeline_spec(&spec).expect("packed VLM pipeline is structurally valid");
    let vision = spec.vision.expect("rich vision expansion config");
    assert_eq!(vision.image_placeholder_token_id, Some(262144));
    assert_eq!(vision.image_token_id, Some(262145));
    assert_eq!(vision.token_count_source.as_deref(), Some("per_patch"));
    assert_eq!(vision.token_count_summary.as_deref(), Some("grid_summary"));
    assert_eq!(vision.image_correspondence.as_deref(), Some("prompt_order"));
    assert_eq!(vision.row_separator_token_id, Some(262146));
    assert_eq!(vision.column_separator_token_id, Some(262147));
    assert_eq!(vision.thumbnail_order.as_deref(), Some("prepend"));

    // JSON-schema validation.
    let validator = schema_validator();
    let instance = fixture_json("vlm_packed_valid.yaml");
    if let Err(error) = validator.validate(&instance) {
        panic!("packed VLM fixture failed JSON-schema validation: {error}");
    }
}

#[test]
fn real_smolvlm_256m_example_parses_and_validates() {
    let path = example_model("smolvlm-256m");
    let metadata = load_metadata(&path).expect("real SmolVLM metadata parses");
    validate(&metadata, &RuntimeCapabilities::default())
        .expect("default runtime advertises the required capabilities");

    let vision = metadata
        .pipeline
        .as_ref()
        .and_then(|pipeline| pipeline.vision.as_ref())
        .expect("vision expansion contract");
    assert_eq!(vision.image_placeholder_token_id, Some(49_190));
    assert_eq!(vision.tokens_per_tile, Some(64));
    assert_eq!(
        metadata.tokens.as_ref().unwrap().vision_start_token_id,
        None
    );
    let image_outputs = &metadata
        .preprocessing
        .as_ref()
        .and_then(|preprocessing| preprocessing.image.as_ref())
        .expect("image preprocessing")
        .outputs;
    assert_eq!(image_outputs[0].name, "vision_encoder.pixel_values");
    assert_eq!(image_outputs[1].name, "vision_encoder.pixel_attention_mask");

    let pipeline = metadata.pipeline.as_ref().expect("pipeline");
    validate_pipeline_spec(pipeline).expect("real SmolVLM pipeline is structurally valid");
    let embedding_io = pipeline.models["token_embedding"]
        .io
        .as_ref()
        .expect("embedding I/O");
    assert_eq!(embedding_io.token_input.as_deref(), Some("input_ids"));
    assert_eq!(embedding_io.hidden_output.as_deref(), Some("inputs_embeds"));
    let decoder_io = pipeline.models["decoder"].io.as_ref().expect("decoder I/O");
    assert_eq!(
        decoder_io.sequence_source,
        Some(onnx_genai_metadata::SequenceInputKind::InputsEmbeds)
    );
    assert_eq!(
        decoder_io.kv_ownership,
        Some(onnx_genai_metadata::KvOwnership::Owned)
    );
    assert_eq!(decoder_io.kv_inputs.as_ref().unwrap().len(), 60);
    assert_eq!(decoder_io.kv_outputs.as_ref().unwrap().len(), 60);

    let schema_validator = schema_validator();
    let instance: serde_json::Value =
        serde_yaml::from_str(&std::fs::read_to_string(path).expect("read example"))
            .expect("example converts to JSON");
    if let Err(error) = schema_validator.validate(&instance) {
        panic!("real SmolVLM example failed JSON-schema validation: {error}");
    }
}

#[test]
fn vlm_multistate_fixture_deserializes_and_validates_against_schema() {
    let metadata = load_metadata(&crate_fixture("vlm_multistate_valid.yaml"))
        .expect("multistate VLM fixture deserializes into the typed contract");

    let spec = metadata.pipeline.expect("pipeline section");
    validate_pipeline_spec(&spec).expect("multistate VLM pipeline is structurally valid");

    // Declared 3-axis position program.
    let positions = spec.positions.expect("position program");
    assert_eq!(positions.input, "position_ids");
    assert_eq!(positions.rank, 3);
    assert_eq!(positions.tensor_rank, Some(3));
    assert_eq!(
        positions.generation.as_deref(),
        Some("processor_coordinates")
    );
    assert_eq!(
        positions.axes.as_deref(),
        Some(
            [
                "temporal".to_string(),
                "height".to_string(),
                "width".to_string()
            ]
            .as_slice()
        )
    );
    assert_eq!(positions.continuation.as_deref(), Some("carry_max"));

    let io = spec.models["decoder"].io.as_ref().expect("decoder io");
    // A decoder may declare BOTH a raw token input and a routed sequence input.
    assert_eq!(io.token_input.as_deref(), Some("input_ids"));
    assert_eq!(io.inputs_embeds_input.as_deref(), Some("inputs_embeds"));
    // Sparse KV ports come from the declared list (two layers => four ports).
    assert_eq!(io.kv_inputs.as_deref().map(<[_]>::len), Some(4));
    assert_eq!(io.kv_outputs.as_deref().map(<[_]>::len), Some(4));
    assert_eq!(io.kv_update.as_deref(), Some("append"));
    // Two fixed loop-carried state pairs with replace semantics.
    let state_pairs = io.state_pairs.as_ref().expect("state pairs");
    assert_eq!(state_pairs.len(), 2);
    assert_eq!(state_pairs[0].init.as_deref(), Some("zeros"));
    assert_eq!(state_pairs[0].update.as_deref(), Some("replace"));

    let validator = schema_validator();
    let instance = fixture_json("vlm_multistate_valid.yaml");
    if let Err(error) = validator.validate(&instance) {
        panic!("multistate VLM fixture failed JSON-schema validation: {error}");
    }
}

#[test]
fn linear_position_program_declares_rank_two_generation() {
    let yaml = r#"
pipeline:
  models:
    decoder:
      filename: decoder.onnx
      type: decoder
  strategy:
    kind: autoregressive
    decoder: decoder
  positions:
    input: position_ids
    rank: 1
    tensor_rank: 2
    generation: linear
    axes: [sequence]
    dtype: int64
    continuation: linear_increment
"#;
    let metadata: InferenceMetadata =
        serde_yaml::from_str(yaml).expect("rank-2 linear position program parses");
    let positions = metadata
        .pipeline
        .as_ref()
        .and_then(|pipeline| pipeline.positions.as_ref())
        .expect("position program");
    assert_eq!(positions.rank, 1);
    assert_eq!(positions.tensor_rank, Some(2));
    assert_eq!(positions.generation.as_deref(), Some("linear"));

    let instance: serde_json::Value =
        serde_yaml::from_str(yaml).expect("linear fixture converts to JSON");
    schema_validator()
        .validate(&instance)
        .expect("rank-2 linear position program validates against JSON schema");
}

#[test]
fn multimodal_fixtures_report_precise_missing_capabilities() {
    let packed =
        load_metadata(&crate_fixture("vlm_packed_valid.yaml")).expect("packed fixture loads");
    let multistate = load_metadata(&crate_fixture("vlm_multistate_valid.yaml"))
        .expect("multistate fixture loads");
    let unsupported_runtime = RuntimeCapabilities {
        supported: Vec::new(),
    };

    assert_eq!(
        validate(&packed, &unsupported_runtime).expect_err("packed capabilities are required"),
        ["image_preprocessing_program", "packed_image_outputs"]
    );
    assert_eq!(
        validate(&multistate, &unsupported_runtime)
            .expect_err("position and state capabilities are required"),
        [
            "position_program",
            "multi_axis_positions",
            "loop_carried_state",
            "dual_sequence_inputs"
        ]
    );
}

#[test]
fn fixed_state_schema_requires_init_and_update_programs() {
    let mut instance = fixture_json("vlm_multistate_valid.yaml");
    let state_pairs = instance
        .pointer_mut("/pipeline/models/decoder/io/state_pairs")
        .and_then(serde_json::Value::as_array_mut)
        .expect("state-pair array");
    state_pairs[0]
        .as_object_mut()
        .expect("state-pair object")
        .remove("update");

    assert!(
        !schema_validator().is_valid(&instance),
        "fixed replacement state pairs without update semantics must fail schema validation"
    );
}

#[test]
fn existing_tiny_gemma4_vlm_metadata_still_deserializes() {
    // Backward compatibility: the committed VLM composite fixture that predates
    // the typed multimodal contract must keep deserializing unchanged.
    let metadata = load_metadata(&fixture("tiny-gemma4-vlm/inference_metadata.yaml"))
        .expect("legacy VLM fixture still deserializes");
    let spec = metadata.pipeline.expect("pipeline section");
    validate_pipeline_spec(&spec).expect("legacy VLM pipeline still validates");
    assert!(matches!(
        spec.strategy.kind,
        PipelineStrategyKind::Composite
    ));
    assert_eq!(spec.models.len(), 3);
}
