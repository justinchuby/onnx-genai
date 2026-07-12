use onnx_genai_engine::{Engine, EngineConfig, FinishReason, GeneratePrompt, GenerateRequest};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn tiny_scatter_fixture() -> anyhow::Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm-scatter")
        .canonicalize()?)
}

fn token_request(tokens: Vec<u32>, max_new_tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(tokens));
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    request
}

#[test]
fn batched_static_decode_matches_individual_static_generates() -> anyhow::Result<()> {
    let fixture = tiny_scatter_fixture()?;
    let requests = vec![
        token_request(vec![1, 5], 1),
        token_request(vec![2, 6, 7], 3),
        token_request(vec![3], 2),
        token_request(vec![4, 8, 9, 10], 4),
    ];

    let expected = requests
        .iter()
        .cloned()
        .map(|request| Engine::from_dir(&fixture, EngineConfig::default())?.generate(request))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
    let batched = engine.generate_batched_static(requests)?;

    assert_eq!(batched, expected);
    assert!(
        batched
            .iter()
            .all(|result| result.prefix_cache_hit_len == 0)
    );
    assert_eq!(batched[0].finish_reason, FinishReason::MaxTokens);
    Ok(())
}

#[test]
#[ignore = "micro-measurement; run with --ignored --nocapture to inspect tokens/sec"]
fn batched_static_decode_reports_tiny_scatter_throughput() -> anyhow::Result<()> {
    let fixture = tiny_scatter_fixture()?;
    let requests = (0..8)
        .map(|idx| token_request(vec![1 + idx as u32, 5 + (idx as u32 % 4)], 8))
        .collect::<Vec<_>>();

    let mut sequential_engine = Engine::from_dir(&fixture, EngineConfig::default())?;
    let sequential_start = Instant::now();
    let sequential = requests
        .iter()
        .cloned()
        .map(|request| sequential_engine.generate(request))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let sequential_elapsed = sequential_start.elapsed();

    let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
    let batched_start = Instant::now();
    let batched = engine.generate_batched_static(requests)?;
    let batched_elapsed = batched_start.elapsed();

    assert_eq!(batched, sequential);
    let generated_tokens = batched
        .iter()
        .map(|result| result.token_ids.len())
        .sum::<usize>();
    let sequential_tps = generated_tokens as f64 / sequential_elapsed.as_secs_f64();
    let batched_tps = generated_tokens as f64 / batched_elapsed.as_secs_f64();
    eprintln!(
        "tiny-llm-scatter throughput: rows={} tokens={} sequential={:.2} tok/s ({:?}) batched={:.2} tok/s ({:?}) speedup={:.2}x",
        batched.len(),
        generated_tokens,
        sequential_tps,
        sequential_elapsed,
        batched_tps,
        batched_elapsed,
        batched_tps / sequential_tps,
    );
    Ok(())
}
