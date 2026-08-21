//! Native eager-decode regression for the deterministic tiny GLM-5.2
//! DSA-MoE fixture emitted with two `pkg.nxrt::IndexShare` nodes and fused QMoE.
//!
//! This native-CUDA regression requires both `cuda` and `native-backend`:
//!
//! ```bash
//! CUDA_VISIBLE_DEVICES=1 cargo test -p onnx-genai-engine \
//!   --features cuda,native-backend glm_tiny_qmoe
//! ```
//!
//! The committed fixture is reproducible with:
//!
//! ```bash
//! /path/to/mobius/.venv/bin/python \
//!   tests/fixtures/tiny-glm52-qmoe-indexshare/generate.py \
//!   --mobius-root /path/to/mobius
//! ```
//!
//! `GLM_TINY_QMOE_E2E_DIR` may override the committed fixture. Missing fixture
//! files skip cleanly so source packages that omit binary fixtures remain green.
#![cfg(feature = "native-cuda")]

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest, NativeDecodeDevice,
    SpeculativeMode,
};

const ANCHOR_IDS: &[u32] = &[62, 164, 59, 205, 48, 166, 27, 9, 221, 190, 123, 108];

fn fixture_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("GLM_TINY_QMOE_E2E_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-glm52-qmoe-indexshare")
        });
    let required = [
        "model.onnx",
        "model.onnx.data",
        "inference_metadata.yaml",
        "tokenizer.json",
    ];
    let missing: Vec<_> = required
        .iter()
        .filter(|name| !dir.join(name).is_file())
        .collect();
    if missing.is_empty() {
        Some(dir)
    } else {
        eprintln!(
            "skipping GLM-5.2 native eager regression: fixture {} is missing {}",
            dir.display(),
            missing
                .iter()
                .map(|name| name.as_ref())
                .collect::<Vec<&str>>()
                .join(", ")
        );
        None
    }
}

fn native_engine(dir: &Path, device: NativeDecodeDevice) -> anyhow::Result<Engine> {
    Engine::from_dir(
        dir,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(device),
            ..EngineConfig::default()
        },
    )
}

fn generate(engine: &mut Engine) -> anyhow::Result<Vec<u32>> {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![123]));
    request.options.max_new_tokens = ANCHOR_IDS.len();
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    Ok(engine.generate(request)?.token_ids)
}

fn assert_current_emission(dir: &Path) -> anyhow::Result<()> {
    let model = dir.join("model.onnx");
    let graph = onnx_runtime_loader::load_model(&model)?;
    assert_eq!(
        graph
            .nodes
            .values()
            .filter(|node| node.domain == "pkg.nxrt" && node.op_type == "IndexShare")
            .count(),
        2,
        "{} must contain exactly two pkg.nxrt::IndexShare nodes",
        model.display(),
    );
    assert!(
        graph
            .nodes
            .values()
            .any(|node| node.domain == "com.microsoft" && node.op_type == "QMoE"),
        "{} does not contain fused QMoE",
        model.display()
    );
    Ok(())
}

#[test]
fn glm_tiny_qmoe_native_cpu_eager_decode_locks_anchor_ids() -> anyhow::Result<()> {
    let Some(dir) = fixture_dir() else {
        return Ok(());
    };
    assert_current_emission(&dir)?;

    let mut cpu = native_engine(&dir, NativeDecodeDevice::Cpu)?;
    let tokens = generate(&mut cpu)?;
    eprintln!("glm52 native CPU eager tokens: {tokens:?}");
    assert_eq!(tokens, ANCHOR_IDS);
    Ok(())
}

#[test]
fn glm_tiny_qmoe_native_cuda_eager_matches_cpu_and_declines_capture() -> anyhow::Result<()> {
    let Some(dir) = fixture_dir() else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping GLM-5.2 native CUDA regression: CUDA is unavailable: {error}");
        return Ok(());
    }
    assert_current_emission(&dir)?;

    let mut cpu = native_engine(&dir, NativeDecodeDevice::Cpu)?;
    let cpu_tokens = generate(&mut cpu)?;
    assert_eq!(cpu_tokens, ANCHOR_IDS);

    let mut cuda = native_engine(&dir, NativeDecodeDevice::Cuda { index: Some(0) })?;
    let cuda_tokens = generate(&mut cuda)?;
    let stats = cuda
        .native_cuda_debug_stats()
        .expect("native CUDA engine exposes decode diagnostics");
    eprintln!(
        "glm52 native CUDA eager tokens: {cuda_tokens:?}; captures={} replays={} fallbacks={}",
        stats.graph.captures, stats.graph.replays, stats.graph.fallbacks
    );

    assert_eq!(cuda_tokens, cpu_tokens, "native CUDA diverged from CPU");
    assert_eq!(
        stats.graph.captures, 0,
        "concat/logical IndexShare form must remain eager before S3 capacity emission"
    );
    assert_eq!(stats.graph.replays, 0);
    Ok(())
}

#[test]
fn glm_tiny_qmoe_native_cuda_session_admits_after_workspace_preparation() -> anyhow::Result<()> {
    let Some(dir) = fixture_dir() else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping GLM-5.2 native CUDA admission regression: {error}");
        return Ok(());
    }
    let mut engine = native_engine(&dir, NativeDecodeDevice::Cuda { index: Some(0) })?;
    for turn in 0..2 {
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let admission_events = events.clone();
        let mut admitted = move || admission_events.lock().unwrap().push("admission");
        let token_events = events.clone();
        let mut token = move |_| {
            token_events.lock().unwrap().push("token");
            Ok(())
        };
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![123]));
        request.options.max_new_tokens = 2;
        request.options.temperature = 0.0;
        request.options.greedy = true;
        request.options.stop_on_eos = false;
        let result =
            engine.generate_with_callbacks(request, Some(&mut admitted), Some(&mut token))?;
        assert_eq!(result.token_ids, ANCHOR_IDS[..2]);
        let events = events.lock().unwrap();
        assert_eq!(events.first(), Some(&"admission"), "turn {turn}");
        assert!(events[1..].contains(&"token"), "turn {turn}: {events:?}");
    }
    Ok(())
}

#[test]
fn glm_tiny_qmoe_native_cuda_speculative_verify_uses_width_workspace() -> anyhow::Result<()> {
    let Some(dir) = fixture_dir() else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping GLM-5.2 native CUDA speculative regression: {error}");
        return Ok(());
    }
    let device = NativeDecodeDevice::Cuda { index: Some(0) };
    let mut baseline = native_engine(&dir, device.clone())?;
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![123]));
    request.options.max_new_tokens = ANCHOR_IDS.len() * 2;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    let expected = baseline.generate(request.clone())?;

    let mut engine = native_engine(&dir, device)?;
    request.options.speculative_mode = Some(SpeculativeMode::PromptLookup {
        ngram: 1,
        max_tokens: 4,
    });

    let result = engine.generate(request)?;
    let stats = engine.last_speculative_stats();
    assert_eq!(result.token_ids, expected.token_ids);
    assert!(
        stats.verification_steps > 0 && stats.proposed_tokens > 1,
        "prompt-length-one regression must execute a multi-row verify: {stats:?}"
    );
    Ok(())
}
