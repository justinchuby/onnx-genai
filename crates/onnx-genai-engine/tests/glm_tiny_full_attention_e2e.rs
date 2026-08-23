//! End-to-end regression for the deterministic tiny GLM-5.2 **dense
//! full-attention fallback** fixture (`config.use_dsa=False`, mirroring the
//! Mobius CLI's `--glm-full-attention` flag): plain dense MLA with **zero**
//! `pkg.nxrt::IndexShare` nodes, still exporting routed MoE experts as a
//! single fused `com.microsoft::QMoE` node.
//!
//! Unlike the sibling DSA/IndexShare fixture
//! (`glm_tiny_qmoe_native_cuda_e2e.rs`), this graph has no native-only custom
//! op, so it must run correctly through **stock ONNX Runtime** as well as the
//! native CPU/CUDA backends — this is the explicit dense/full-attention
//! fallback stock ORT can execute when a caller cannot or does not want the
//! native runtime's `pkg.nxrt::IndexShare` kernel. Coherence is checked at
//! three levels: structural (no IndexShare, has fused QMoE), a locked native
//! CPU/CUDA anchor (regression), and stock-ORT-vs-native-CPU token agreement
//! (proves the fallback graph is executable and numerically consistent on the
//! path that has no native-only op dependency at all).
//!
//! This native-CUDA build requires both `cuda` and `native-backend`:
//!
//! ```bash
//! CUDA_VISIBLE_DEVICES=1 cargo test -p onnx-genai-engine \
//!   --features cuda,native-backend glm_tiny_full_attention
//! ```
//!
//! The committed fixture is reproducible with:
//!
//! ```bash
//! /path/to/mobius/.venv/bin/python \
//!   tests/fixtures/tiny-glm52-qmoe-indexshare/generate.py \
//!   --mobius-root /path/to/mobius --full-attention
//! ```
//!
//! `GLM_TINY_FULL_ATTENTION_E2E_DIR` may override the committed fixture.
//! Missing fixture files skip cleanly so source packages that omit binary
//! fixtures remain green.
#![cfg(feature = "native-cuda")]

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest, NativeDecodeDevice,
};

/// Locked native CPU/CUDA decode anchor for the committed fixture (seed 0,
/// prompt `[123]`), verified once and pinned as a regression guard: see
/// `glm_tiny_qmoe_native_cuda_e2e.rs`'s `ANCHOR_IDS` for the sibling
/// DSA/IndexShare fixture this pattern mirrors.
const ANCHOR_IDS: &[u32] = &[193, 183, 233, 181, 100, 77, 182, 143, 116, 147, 127, 100];

/// See `glm_tiny_qmoe_native_cuda_e2e.rs::resolve_model_path` for why only
/// `model.onnx.textproto` is committed.
fn resolve_model_path(dir: &Path) -> Option<PathBuf> {
    let onnx = dir.join("model.onnx");
    if onnx.is_file() {
        return Some(onnx);
    }
    let textproto = dir.join("model.onnx.textproto");
    if textproto.is_file() {
        return Some(textproto);
    }
    None
}

fn fixture_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("GLM_TINY_FULL_ATTENTION_E2E_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-glm52-full-attention")
        });
    let mut missing: Vec<String> = Vec::new();
    if resolve_model_path(&dir).is_none() {
        missing.push("model.onnx or model.onnx.textproto".to_string());
    }
    for name in ["inference_metadata.yaml", "tokenizer.json"] {
        if !dir.join(name).is_file() {
            missing.push(name.to_string());
        }
    }
    if missing.is_empty() {
        Some(dir)
    } else {
        eprintln!(
            "skipping GLM-5.2 full-attention fallback regression: fixture {} is missing {}",
            dir.display(),
            missing.join(", ")
        );
        None
    }
}

fn engine(
    dir: &Path,
    backend: EngineDecodeBackend,
    device: Option<NativeDecodeDevice>,
) -> anyhow::Result<Engine> {
    Engine::from_dir(
        dir,
        EngineConfig {
            decode_backend: backend,
            native_device: device,
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

/// Structural gate: the full-attention export must contain **zero**
/// `pkg.nxrt::IndexShare` nodes (that is the entire point of the fallback)
/// while still fusing routed experts into `com.microsoft::QMoE`.
fn assert_current_emission(dir: &Path) -> anyhow::Result<()> {
    let model = resolve_model_path(dir)
        .ok_or_else(|| anyhow::anyhow!("{} has no model.onnx(.textproto)", dir.display()))?;
    let graph = onnx_runtime_loader::load_model(&model)?;
    assert_eq!(
        graph
            .nodes
            .values()
            .filter(|node| node.domain == "pkg.nxrt" && node.op_type == "IndexShare")
            .count(),
        0,
        "{} is the dense full-attention fallback and must contain zero \
         pkg.nxrt::IndexShare nodes",
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
fn glm_tiny_full_attention_structural_has_no_indexshare() -> anyhow::Result<()> {
    let Some(dir) = fixture_dir() else {
        return Ok(());
    };
    assert_current_emission(&dir)
}

#[test]
fn glm_tiny_full_attention_native_cpu_eager_decode_locks_anchor_ids() -> anyhow::Result<()> {
    let Some(dir) = fixture_dir() else {
        return Ok(());
    };
    assert_current_emission(&dir)?;

    let mut cpu = engine(
        &dir,
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cpu),
    )?;
    let tokens = generate(&mut cpu)?;
    eprintln!("glm52 full-attention native CPU eager tokens: {tokens:?}");
    assert_eq!(tokens, ANCHOR_IDS);
    Ok(())
}

#[test]
fn glm_tiny_full_attention_native_cuda_matches_cpu() -> anyhow::Result<()> {
    let Some(dir) = fixture_dir() else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!(
            "skipping GLM-5.2 full-attention native CUDA regression: CUDA is unavailable: {error}"
        );
        return Ok(());
    }
    assert_current_emission(&dir)?;

    let mut cpu = engine(
        &dir,
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cpu),
    )?;
    let cpu_tokens = generate(&mut cpu)?;
    assert_eq!(cpu_tokens, ANCHOR_IDS);

    let mut cuda = engine(
        &dir,
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cuda { index: Some(0) }),
    )?;
    let cuda_tokens = generate(&mut cuda)?;
    eprintln!("glm52 full-attention native CUDA tokens: {cuda_tokens:?}");
    assert_eq!(
        cuda_tokens, cpu_tokens,
        "native CUDA diverged from native CPU"
    );
    Ok(())
}

/// No success-shaped skip: this is the one test proving the dense
/// full-attention export is executable on **stock ONNX Runtime** with no
/// native-only op dependency at all (the entire reason the fallback exists).
/// Its tokens must agree with the native CPU path exactly — both run the same
/// graph and the same greedy/deterministic decode, so any divergence is a
/// real numeric or op-semantics bug, not an expected fallback difference.
#[test]
fn glm_tiny_full_attention_stock_ort_matches_native_cpu() -> anyhow::Result<()> {
    let Some(dir) = fixture_dir() else {
        return Ok(());
    };
    assert_current_emission(&dir)?;

    let mut native = engine(
        &dir,
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cpu),
    )?;
    let native_tokens = generate(&mut native)?;
    assert_eq!(native_tokens, ANCHOR_IDS);

    let mut ort = engine(&dir, EngineDecodeBackend::Ort, None)?;
    let ort_tokens = generate(&mut ort)?;
    eprintln!("glm52 full-attention stock ORT tokens: {ort_tokens:?}");
    assert_eq!(
        ort_tokens, native_tokens,
        "stock ORT execution of the full-attention fallback diverged from the native CPU backend"
    );
    Ok(())
}
