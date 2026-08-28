use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, GenerateResult,
};
use onnx_genai_ort::SessionOptions;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Once;

fn shared_buffer_fixture() -> anyhow::Result<PathBuf> {
    static ENABLE_SHARED_PRESENT: Once = Once::new();
    ENABLE_SHARED_PRESENT.call_once(|| {
        // This isolated test binary sets the CPU fixed-present opt-in before
        // runtime configuration is first read.
        unsafe { std::env::set_var("ONNX_GENAI_SHARED_KV_PRESENT_BINDING", "1") };
    });
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm-sharedbuffer")
        .canonicalize()?;
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/test-fixtures/live-row-shared");
    fs::create_dir_all(&root)?;
    for entry in fs::read_dir(&source)? {
        let entry = entry?;
        if entry.file_name() != "model.onnx" && entry.file_name() != "inference_metadata.yaml" {
            fs::copy(entry.path(), root.join(entry.file_name()))?;
        }
    }
    let metadata = fs::read_to_string(source.join("inference_metadata.yaml"))?;
    fs::write(
        root.join("inference_metadata.yaml"),
        metadata.replacen("aliasing: forbidden", "aliasing: permitted", 1),
    )?;
    let textproto = fs::read_to_string(source.join("model.onnx.textproto"))?;
    fs::write(
        root.join("model.onnx"),
        onnx_std::textproto::to_binary(&textproto)?,
    )?;
    Ok(root)
}

fn cpu_engine(model_dir: &Path) -> anyhow::Result<Engine> {
    Engine::from_dir_with_session_options(
        model_dir,
        EngineConfig::default(),
        SessionOptions::default().with_intra_op_threads(1),
    )
}

fn ragged_requests() -> Vec<GenerateRequest> {
    [
        (vec![1], 1),
        (vec![1, 2, 3], 4),
        (vec![2, 1], 2),
        (vec![3, 2, 1, 2], 3),
    ]
    .into_iter()
    .map(|(tokens, max_new_tokens)| GenerateRequest {
        prompt: GeneratePrompt::TokenIds(tokens),
        options: GenerateOptions {
            greedy: true,
            stop_on_eos: false,
            max_new_tokens,
            ..GenerateOptions::default()
        },
    })
    .collect()
}

fn isolated_results(
    fixture: &Path,
    requests: &[GenerateRequest],
) -> anyhow::Result<Vec<GenerateResult>> {
    requests
        .iter()
        .cloned()
        .map(|request| cpu_engine(fixture)?.generate(request))
        .collect()
}

#[test]
fn static_shared_batch_matches_isolated_ragged_rows() -> anyhow::Result<()> {
    let fixture = shared_buffer_fixture()?;
    let requests = ragged_requests();
    let expected = isolated_results(&fixture, &requests)?;
    let mut engine = cpu_engine(&fixture)?;
    assert!(
        engine.batching_capability().supports_batching(),
        "fixture must exercise the production shared-forward path"
    );

    assert_eq!(engine.generate_batched_static(requests)?, expected);
    Ok(())
}

#[test]
fn scheduled_shared_batch_matches_isolated_through_backfill_and_row_reuse() -> anyhow::Result<()> {
    let fixture = shared_buffer_fixture()?;
    let requests = ragged_requests();
    let expected = isolated_results(&fixture, &requests)?;
    let mut engine = cpu_engine(&fixture)?;
    assert!(
        engine.batching_capability().supports_batching(),
        "fixture must exercise the production shared-forward path"
    );

    assert_eq!(
        engine.run_continuous_batch_scheduled(requests, 2)?,
        expected,
        "two physical rows must retire unequal requests, admit queued work, and reuse both slots \
         without carrying prior output, RNG, completion, or state ownership"
    );
    Ok(())
}
