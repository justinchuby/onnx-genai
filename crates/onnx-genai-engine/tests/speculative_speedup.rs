use onnx_genai_engine::{Engine, EngineConfig, FinishReason, GeneratePrompt, GenerateRequest};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const DEFAULT_PROMPT: &str = "Once upon a time, there was a small robot who";
const DEFAULT_MAX_NEW_TOKENS: usize = 32;
const DEFAULT_SPECULATIVE_K: usize = 4;
const REQUIRED_SPEEDUP: f64 = 1.5;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn model_path(env_name: &str, default: &str) -> PathBuf {
    std::env::var_os(env_name)
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join(default))
}

fn request(prompt: &str, max_new_tokens: usize, speculative_k: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::Text(prompt.to_string()));
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    request.options.num_speculative_tokens = Some(speculative_k);
    request
}

fn timed_generate(
    engine: &mut Engine,
    prompt: &str,
    max_new_tokens: usize,
    speculative_k: usize,
) -> anyhow::Result<(Vec<u32>, FinishReason, Duration)> {
    let started = Instant::now();
    let result = engine.generate(request(prompt, max_new_tokens, speculative_k))?;
    Ok((result.token_ids, result.finish_reason, started.elapsed()))
}

fn tokens_per_second(tokens: usize, elapsed: Duration) -> f64 {
    tokens as f64 / elapsed.as_secs_f64().max(f64::EPSILON)
}

#[test]
#[ignore = "requires real target/draft models under models/; run scripts/bench_speculative.sh"]
fn speculative_decoding_exceeds_required_speedup_when_models_are_present() -> anyhow::Result<()> {
    let target = model_path("ONNX_GENAI_SPEC_TARGET", "models/tinystories-33m");
    let draft = model_path("ONNX_GENAI_SPEC_DRAFT", "models/tinystories-1m");
    if !target.exists() || !draft.exists() {
        eprintln!(
            "skipping speculative benchmark: target={} exists={} draft={} exists={}",
            target.display(),
            target.exists(),
            draft.display(),
            draft.exists()
        );
        return Ok(());
    }

    let prompt =
        std::env::var("ONNX_GENAI_SPEC_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.to_string());
    let max_new_tokens = std::env::var("ONNX_GENAI_SPEC_MAX_NEW_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_MAX_NEW_TOKENS);
    let speculative_k = std::env::var("ONNX_GENAI_SPEC_K")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_SPECULATIVE_K)
        .max(1);

    let mut baseline = Engine::from_dir(&target, EngineConfig::default())?;
    let (baseline_tokens, baseline_finish, baseline_elapsed) =
        timed_generate(&mut baseline, &prompt, max_new_tokens, speculative_k)?;

    let mut speculative = Engine::from_dir(
        &target,
        EngineConfig {
            draft_model: Some(draft.clone()),
            num_speculative_tokens: speculative_k,
            ..EngineConfig::default()
        },
    )?;
    let (spec_tokens, spec_finish, speculative_elapsed) =
        timed_generate(&mut speculative, &prompt, max_new_tokens, speculative_k)?;

    assert_eq!(
        baseline_tokens, spec_tokens,
        "speculative greedy output must exactly match target-only greedy tokens"
    );
    assert_eq!(baseline_finish, spec_finish);

    let baseline_token_count = baseline_tokens.len();
    let speculative_token_count = spec_tokens.len();
    let baseline_tps = tokens_per_second(baseline_token_count, baseline_elapsed);
    let speculative_tps = tokens_per_second(speculative_token_count, speculative_elapsed);
    let speedup = speculative_tps / baseline_tps;

    eprintln!(
        "speculative_speedup target={} draft={} tokens={} baseline={:.3}s ({:.2} tok/s) speculative={:.3}s ({:.2} tok/s) speedup={:.3}x k={}",
        target.display(),
        draft.display(),
        baseline_token_count,
        baseline_elapsed.as_secs_f64(),
        baseline_tps,
        speculative_elapsed.as_secs_f64(),
        speculative_tps,
        speedup,
        speculative_k
    );

    if std::env::var_os("ONNX_GENAI_SPEC_ALLOW_SLOW").is_none() {
        assert!(
            speedup >= REQUIRED_SPEEDUP,
            "speculative decoding speedup {speedup:.3}x did not meet {REQUIRED_SPEEDUP:.1}x"
        );
    }

    Ok(())
}
