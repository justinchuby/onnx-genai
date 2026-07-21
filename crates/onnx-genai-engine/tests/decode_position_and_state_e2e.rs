use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, ResourceLimit,
    ResourceLimits,
};
use onnx_genai_ort::PipelineModels;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-multiaxis-state-decoder")
}

#[test]
fn multiaxis_position_and_state_fixture_matches_contract() -> anyhow::Result<()> {
    let fixture = fixture_dir();
    let models = PipelineModels::load(&fixture)?;
    let positions = models
        .directory
        .spec
        .positions
        .as_ref()
        .expect("fixture declares a position program");
    assert_eq!(positions.input, "position_ids");
    assert_eq!(positions.rank, 3);
    assert_eq!(
        positions
            .axes
            .as_ref()
            .map(|axes| { axes.iter().map(String::as_str).collect::<Vec<_>>() }),
        Some(vec!["first", "second", "third"])
    );
    assert_eq!(positions.continuation.as_deref(), Some("carry_max"));

    let decoder_io = models.directory.spec.models["decoder"]
        .io
        .as_ref()
        .expect("decoder declares explicit I/O");
    assert!(
        decoder_io.token_input.is_some() && decoder_io.inputs_embeds_input.is_some(),
        "raw tokens and a routed sequence are valid simultaneous inputs"
    );
    assert_eq!(
        decoder_io.kv_inputs.as_deref(),
        Some(
            [
                "past.3.key".to_string(),
                "past.3.value".to_string(),
                "past.11.key".to_string(),
                "past.11.value".to_string(),
            ]
            .as_slice()
        ),
        "KV ports preserve sparse declared layer indices"
    );
    assert_eq!(
        decoder_io.state_pairs.as_ref().map(Vec::len),
        Some(2),
        "two fixed replace-state tensors are declared separately from KV"
    );

    let decoder = models
        .session("decoder")
        .expect("fixture decoder session is loaded");
    assert_eq!(
        decoder
            .inputs()
            .iter()
            .find(|input| input.name == "position_ids")
            .map(|input| input.shape.as_slice()),
        Some([3, 1, -1].as_slice())
    );
    for port in [
        "state_a.out",
        "state_b.out",
        "present.3.key",
        "present.3.value",
        "present.11.key",
        "present.11.value",
    ] {
        assert!(
            decoder.output_names().iter().any(|output| output == port),
            "decoder must expose declared output {port}"
        );
    }
    Ok(())
}

fn generation_request() -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![1, 2, 3]));
    request.options = GenerateOptions {
        max_new_tokens: 3,
        temperature: 0.0,
        greedy: true,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };
    request
}

#[test]
fn pipeline_generation_carries_multiaxis_positions_and_fixed_state_after_reset()
-> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&fixture_dir(), EngineConfig::default())?;

    assert_eq!(
        engine.generate(generation_request())?.token_ids,
        vec![6, 15, 24]
    );
    assert_eq!(
        engine.generate(generation_request())?.token_ids,
        vec![6, 15, 24],
        "each public generate call must rebuild the decoder with explicit I/O, position metadata, and fixed state"
    );
    Ok(())
}

#[test]
fn pipeline_load_rejects_fixed_state_over_host_admission_budget() {
    let config = EngineConfig {
        limits: ResourceLimits {
            host_ram_limit: ResourceLimit::Bytes(15),
            ..ResourceLimits::default()
        },
        ..EngineConfig::default()
    };

    let error = match Engine::from_pipeline_dir(&fixture_dir(), config) {
        Ok(_) => panic!("16 bytes of fixed state must not fit a 15-byte host admission budget"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(message.contains("requires 16 bytes"), "{message}");
    assert!(message.contains("budget is 15 bytes"), "{message}");
}
