//! Golden decode lock for a deterministic tiny DeepSeek-V2-style graph.
//!
//! The real DeepSeek-V2-Lite export currently reaches native execution through
//! standard ONNX `RotaryEmbedding` + `Attention` (not a custom MLA op) plus
//! integer `com.microsoft::QMoE`. This fixture locks that exact native path with
//! a sparse top-k int4 QMoE block.

#![cfg(feature = "native-backend")]

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest, NativeDecodeDevice,
};

const PROMPT_IDS: &[u32] = &[3];
const ANCHOR_IDS: &[u32] = &[11, 11, 11, 11, 11, 11, 11, 11];

fn fixture_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("DEEPSEEK_V2_TINY_QMOE_E2E_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-deepseek-v2-qmoe-attention")
        });
    let required = [
        "model.onnx",
        "inference_metadata.yaml",
        "tokenizer.json",
        "manifest.json",
    ];
    let missing: Vec<_> = required
        .iter()
        .filter(|name| !dir.join(name).is_file())
        .collect();
    if missing.is_empty() {
        Some(dir)
    } else {
        eprintln!(
            "skipping DeepSeek-V2 tiny native golden lock: fixture {} is missing {}",
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
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(PROMPT_IDS.to_vec()));
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
            .filter(|node| node.domain.is_empty() && node.op_type == "Attention")
            .count(),
        1,
        "{} must contain exactly one standard ai.onnx::Attention node",
        model.display()
    );
    assert_eq!(
        graph
            .nodes
            .values()
            .filter(|node| node.domain.is_empty() && node.op_type == "RotaryEmbedding")
            .count(),
        2,
        "{} must contain q/k ai.onnx::RotaryEmbedding nodes",
        model.display()
    );
    assert!(
        graph
            .nodes
            .values()
            .any(|node| node.domain == "com.microsoft" && node.op_type == "QMoE"),
        "{} does not contain fused integer QMoE",
        model.display()
    );
    assert!(
        !graph
            .nodes
            .values()
            .any(|node| node.domain == "pkg.nxrt" && node.op_type == "IndexShare"),
        "{} should lock the DeepSeek path, not the GLM IndexShare path",
        model.display()
    );
    Ok(())
}

#[test]
fn deepseek_v2_tiny_native_cpu_eager_decode_locks_anchor_ids() -> anyhow::Result<()> {
    let Some(dir) = fixture_dir() else {
        return Ok(());
    };
    assert_current_emission(&dir)?;

    let mut cpu = native_engine(&dir, NativeDecodeDevice::Cpu)?;
    let tokens = generate(&mut cpu)?;
    eprintln!("DeepSeek-V2 tiny native CPU eager tokens: {tokens:?}");
    assert_eq!(tokens, ANCHOR_IDS);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn deepseek_v2_tiny_native_cuda_matches_cpu_under_current_graph_policy() -> anyhow::Result<()> {
    let Some(dir) = fixture_dir() else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping DeepSeek-V2 tiny native CUDA golden lock: CUDA unavailable: {error}");
        return Ok(());
    }
    assert_current_emission(&dir)?;

    let mut cpu = native_engine(&dir, NativeDecodeDevice::Cpu)?;
    let cpu_tokens = generate(&mut cpu)?;
    assert_eq!(cpu_tokens, ANCHOR_IDS);

    let graph_policy = std::env::var("ONNX_GENAI_CUDA_GRAPH").ok();
    let mut cuda = native_engine(&dir, NativeDecodeDevice::Cuda { index: Some(0) })?;
    let cuda_tokens = generate(&mut cuda)?;
    let stats = cuda
        .native_cuda_debug_stats()
        .expect("native CUDA engine exposes decode diagnostics");
    eprintln!(
        "DeepSeek-V2 tiny native CUDA tokens: {cuda_tokens:?}; ONNX_GENAI_CUDA_GRAPH={graph_policy:?}; captures={} replays={} fallbacks={} decline={:?}",
        stats.graph.captures,
        stats.graph.replays,
        stats.graph.fallbacks,
        stats.graph.decline_reason
    );
    assert_eq!(cuda_tokens, cpu_tokens, "native CUDA diverged from CPU");
    match graph_policy.as_deref() {
        Some("0") => {
            assert_eq!(stats.graph.captures, 0);
            assert_eq!(stats.graph.replays, 0);
        }
        Some("1") => {
            if stats.graph.captures > 0 {
                assert!(
                    stats.graph.replays > 0,
                    "expected captured decode steps to replay"
                );
            } else {
                let reason = stats
                    .graph
                    .decline_reason
                    .as_deref()
                    .expect("capture-disabled run should report why capture declined");
                assert!(
                    reason.contains("attention_mask_consumers_are_capacity_aware"),
                    "unexpected capture decline reason: {reason}"
                );
            }
        }
        _ => {}
    }
    Ok(())
}
