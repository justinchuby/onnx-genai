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
fn continuous_batch_matches_individual_under_admission_eviction() -> anyhow::Result<()> {
    let fixture = tiny_scatter_fixture()?;
    let requests = (0..16)
        .map(|idx| {
            token_request(
                vec![
                    1 + (idx as u32 % 8),
                    5 + (idx as u32 % 5),
                    9 + (idx as u32 % 3),
                ],
                1 + (idx % 6),
            )
        })
        .collect::<Vec<_>>();

    let expected = requests
        .iter()
        .cloned()
        .map(|request| Engine::from_dir(&fixture, EngineConfig::default())?.generate(request))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
    let continuous = engine.run_continuous_batch(requests, 4)?;

    assert_eq!(continuous, expected);
    assert!(
        continuous
            .iter()
            .all(|result| result.prefix_cache_hit_len == 0)
    );
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

#[test]
#[ignore = "micro-measurement; run with --ignored --nocapture to inspect tokens/sec"]
fn continuous_batch_reports_tiny_scatter_throughput() -> anyhow::Result<()> {
    let fixture = tiny_scatter_fixture()?;
    let requests = (0..16)
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

    let mut static_engine = Engine::from_dir(&fixture, EngineConfig::default())?;
    let static_start = Instant::now();
    let static_batched = static_engine.generate_batched_static(requests.clone())?;
    let static_elapsed = static_start.elapsed();

    let mut continuous4_engine = Engine::from_dir(&fixture, EngineConfig::default())?;
    let continuous4_start = Instant::now();
    let continuous4 = continuous4_engine.run_continuous_batch(requests.clone(), 4)?;
    let continuous4_elapsed = continuous4_start.elapsed();

    let mut continuous8_engine = Engine::from_dir(&fixture, EngineConfig::default())?;
    let continuous8_start = Instant::now();
    let continuous8 = continuous8_engine.run_continuous_batch(requests, 8)?;
    let continuous8_elapsed = continuous8_start.elapsed();

    assert_eq!(static_batched, sequential);
    assert_eq!(continuous4, sequential);
    assert_eq!(continuous8, sequential);
    let generated_tokens = continuous4
        .iter()
        .map(|result| result.token_ids.len())
        .sum::<usize>();
    let sequential_tps = generated_tokens as f64 / sequential_elapsed.as_secs_f64();
    let static_tps = generated_tokens as f64 / static_elapsed.as_secs_f64();
    let continuous4_tps = generated_tokens as f64 / continuous4_elapsed.as_secs_f64();
    let continuous8_tps = generated_tokens as f64 / continuous8_elapsed.as_secs_f64();
    eprintln!(
        "tiny-llm-scatter continuous throughput: requests={} tokens={} sequential={:.2} tok/s ({:?}) static_batch16={:.2} tok/s ({:?}) continuous4={:.2} tok/s ({:?}, speedup={:.2}x) continuous8={:.2} tok/s ({:?}, speedup={:.2}x)",
        continuous4.len(),
        generated_tokens,
        sequential_tps,
        sequential_elapsed,
        static_tps,
        static_elapsed,
        continuous4_tps,
        continuous4_elapsed,
        continuous4_tps / sequential_tps,
        continuous8_tps,
        continuous8_elapsed,
        continuous8_tps / sequential_tps,
    );
    Ok(())
}
