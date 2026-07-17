//! End-to-end continuous-batching parity for SHARED-BUFFER (past/present) GQA
//! models, e.g. Qwen2.5 CUDA. Continuous batching historically supported only
//! static-cache models; this exercises the `BatchedSharedBufferDecodeSession`
//! path where rows of different lengths share one past/present KV buffer.
//!
//! The test is gated on a real shared-buffer model and a working CUDA EP; it
//! auto-skips (returns Ok) when either is unavailable so it is safe in CI.
//! Point it at a model with `ONNX_GENAI_SHARED_BUFFER_MODEL=<dir>`, otherwise it
//! falls back to the in-repo Qwen2.5 CUDA dir if present.

use std::path::{Path, PathBuf};
use std::time::Instant;

use onnx_genai_engine::{Engine, EngineConfig, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::{ExecutionProvider, SessionOptions};

fn shared_buffer_model_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("ONNX_GENAI_SHARED_BUFFER_MODEL") {
        let path = PathBuf::from(dir);
        return path.is_dir().then_some(path);
    }
    let default = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../Microsoft/qwen2.5-0.5b-instruct-cuda-gpu-4/v4");
    default.is_dir().then(|| default)
}

fn cuda_engine(model_dir: &Path) -> anyhow::Result<Engine> {
    Engine::from_dir_with_session_options(
        model_dir,
        EngineConfig::default(),
        SessionOptions::with_execution_provider(ExecutionProvider::Cuda { device_id: 0 }),
    )
}

fn token_request(tokens: Vec<u32>, max_new_tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(tokens));
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    request
}

/// Continuous batching over a shared-buffer model must match running each
/// request on its own, including admission/eviction as rows of different
/// prompt/output lengths join and leave a `max_batch=4` batch.
#[test]
fn shared_buffer_continuous_batch_matches_individual() -> anyhow::Result<()> {
    let Some(model_dir) = shared_buffer_model_dir() else {
        eprintln!("skipping shared-buffer continuous-batch parity; no model dir available");
        return Ok(());
    };
    let probe = match cuda_engine(&model_dir) {
        Ok(engine) => engine,
        Err(error) => {
            eprintln!("skipping shared-buffer continuous-batch parity; CUDA unavailable: {error}");
            return Ok(());
        }
    };
    // Confirm the model actually routes through continuous batching (i.e. it is
    // a shared-buffer or static-cache model). If not, there is nothing to test.
    if probe.continuous_batch_manager(4).is_err() {
        eprintln!("skipping shared-buffer continuous-batch parity; model is not batchable");
        return Ok(());
    }
    drop(probe);

    let requests = vec![
        token_request(vec![9707, 11], 12),
        token_request(vec![785, 4271, 315], 8),
        token_request(vec![40], 16),
        token_request(vec![1986, 374, 264, 1273], 10),
        token_request(vec![15191, 525, 498], 14),
        token_request(vec![47, 1382, 264], 6),
    ];

    let expected = requests
        .iter()
        .cloned()
        .map(|request| cuda_engine(&model_dir)?.generate(request))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut engine = cuda_engine(&model_dir)?;
    let continuous = engine.run_continuous_batch(requests, 4)?;

    assert_eq!(continuous.len(), expected.len());
    for (index, (got, want)) in continuous.iter().zip(&expected).enumerate() {
        assert_eq!(
            got.token_ids, want.token_ids,
            "row {index} continuous tokens diverged from sequential:\n  continuous={:?}\n  sequential={:?}",
            got.token_ids, want.token_ids
        );
    }
    Ok(())
}

/// Wall-clock comparison of shared-buffer continuous batching vs running the
/// same requests sequentially. Ignored by default (needs a real CUDA model);
/// run with `--ignored --nocapture` to print the concurrency speed-up.
#[test]
#[ignore = "GPU throughput measurement; run with --ignored --nocapture"]
fn shared_buffer_continuous_batch_throughput() -> anyhow::Result<()> {
    let Some(model_dir) = shared_buffer_model_dir() else {
        eprintln!("skipping shared-buffer throughput; no model dir available");
        return Ok(());
    };
    if cuda_engine(&model_dir).is_err() {
        eprintln!("skipping shared-buffer throughput; CUDA unavailable");
        return Ok(());
    }

    const CONCURRENCY: usize = 8;
    const NEW_TOKENS: usize = 64;
    let requests: Vec<GenerateRequest> = (0..CONCURRENCY)
        .map(|index| token_request(vec![9707, 11, 358 + index as u32], NEW_TOKENS))
        .collect();

    let sequential_start = Instant::now();
    let mut sequential_tokens = 0usize;
    let mut sequential_engine = cuda_engine(&model_dir)?;
    for request in requests.iter().cloned() {
        sequential_tokens += sequential_engine.generate(request)?.token_ids.len();
    }
    let sequential_elapsed = sequential_start.elapsed();

    let mut engine = cuda_engine(&model_dir)?;
    let batched_start = Instant::now();
    let batched = engine.run_continuous_batch(requests, CONCURRENCY)?;
    let batched_elapsed = batched_start.elapsed();
    let batched_tokens: usize = batched.iter().map(|result| result.token_ids.len()).sum();

    let seq_tps = sequential_tokens as f64 / sequential_elapsed.as_secs_f64();
    let batched_tps = batched_tokens as f64 / batched_elapsed.as_secs_f64();
    eprintln!(
        "shared-buffer throughput @concurrency={CONCURRENCY}: sequential {seq_tps:.1} tok/s ({sequential_tokens} tok in {sequential_elapsed:?}), continuous {batched_tps:.1} tok/s ({batched_tokens} tok in {batched_elapsed:?}), speed-up {:.2}x",
        batched_tps / seq_tps
    );
    Ok(())
}
