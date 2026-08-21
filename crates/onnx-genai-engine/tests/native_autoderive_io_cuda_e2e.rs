//! Live GPU correctness lock for auto-derived decoder I/O (issue #384).
//!
//! Proves that a STOCK qwen3.6-27b int4 export — a hybrid linear-attention
//! decoder (16 dense GQA layers + 48 recurrent `conv_state`/`recurrent_state`
//! layers) whose `inference_metadata.yaml` declares NO `io` block — loads and
//! decodes natively purely from the graph-derived I/O fallback, with a token
//! stream byte-identical to the native CPU fp32 oracle.
//!
//! ORT-CUDA crashes on this model (internal `stl_vector` assertion), so there is
//! no ORT reference; the trusted oracle is our own native CPU backend, which
//! already threads recurrent state correctly. Any divergence between the CUDA
//! and CPU native runs is therefore a real correctness bug, not an export gap.
//!
//! This is the zero-overlay counterpart of the hand-overlay parity that was
//! proven manually during the #384 re-probe: with auto-derive, the stock export
//! "just works" with no sidecar `io:` block.
//!
//! Note: the CPU oracle run of a 27B model is SLOW (multi-minute prefill +
//! decode); this test is `#[ignore]` and meant to be run deliberately.
//!
//! ```bash
//! QWEN3_6_27B_CUDA_E2E_DIR=/home/justinchu/mary-models/qwen3.6-27b-int4-cuda \
//! CUDA_VISIBLE_DEVICES=2 cargo test -p onnx-genai-engine \
//!   --features cuda,native-backend --test native_autoderive_io_cuda_e2e \
//!   -- --ignored --nocapture
//! ```
#![cfg(feature = "native-cuda")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateRequest, GenerateResult, NativeDecodeDevice,
};

const DEFAULT_MODEL_DIR: &str = "/home/justinchu/mary-models/qwen3.6-27b-int4-cuda";
const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 16;

/// The known-good greedy decode for `PROMPT` on the stock 27B export — the
/// byte-exact correctness lock. With the fused-dispatch loader change the 48
/// `com.microsoft::LinearAttention` function calls stay as ops (no inlined
/// `Scan`) and dispatch to the fused kernel; the token stream must be unchanged.
const GOLDEN_IDS: [u32; 16] = [
    11751, 13, 271, 248068, 271, 248069, 271, 4639, 369, 4252, 13, 11751, 369, 279, 6511, 321,
];

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("QWEN3_6_27B_CUDA_E2E_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
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
            "skipping native auto-derive io e2e: model directory {} is missing {:?}",
            dir.display(),
            missing
        );
        None
    }
}

fn generate(dir: &Path, device: NativeDecodeDevice) -> anyhow::Result<GenerateResult> {
    let mut engine = Engine::from_dir(
        dir,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(device),
            ..EngineConfig::default()
        },
    )?;
    let mut request = GenerateRequest::new(PROMPT.to_string());
    request.options.max_new_tokens = MAX_NEW_TOKENS;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    engine.generate(request)
}

#[test]
#[ignore = "requires the real Qwen3.6-27B int4 hybrid export and a CUDA device (slow CPU oracle)"]
fn stock_export_auto_derives_io_and_matches_cpu_oracle() -> anyhow::Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping native auto-derive io e2e: CUDA unavailable: {error}");
        return Ok(());
    }

    // Native CUDA decode. The stock export has no declared `io:` block, so this
    // only loads if the loader's graph-derived fallback classifies the recurrent
    // conv/recurrent-state ports as loop-carried state pairs (not growable KV).
    let cuda_start = Instant::now();
    let cuda = generate(&dir, NativeDecodeDevice::Cuda { index: Some(0) })?;
    let cuda_elapsed = cuda_start.elapsed();

    // Native CPU fp32 oracle over the same stock export.
    let cpu_start = Instant::now();
    let cpu = generate(&dir, NativeDecodeDevice::Cpu)?;
    let cpu_elapsed = cpu_start.elapsed();

    eprintln!(
        "native auto-derive io e2e:\n  cuda tokens={:?} ({:?})\n  cpu  tokens={:?} ({:?})",
        cuda.token_ids, cuda_elapsed, cpu.token_ids, cpu_elapsed
    );

    assert_eq!(
        cuda.token_ids, cpu.token_ids,
        "auto-derived native CUDA decode diverged from the CPU fp32 oracle — \
         graph-derived io must be byte-exact.\ncuda={:?}\ncpu={:?}",
        cuda.token_ids, cpu.token_ids
    );
    assert_eq!(
        cuda.token_ids.len(),
        MAX_NEW_TOKENS,
        "expected {MAX_NEW_TOKENS} generated tokens, got {}",
        cuda.token_ids.len()
    );
    assert_eq!(
        cuda.token_ids, GOLDEN_IDS,
        "fused-dispatch native CUDA decode diverged from the known-good greedy \
         ids — keeping LinearAttention as a fused op (no inlined Scan) must be \
         byte-exact.\ngot={:?}\nwant={:?}",
        cuda.token_ids, GOLDEN_IDS
    );
    Ok(())
}

/// Structural lock for Part 1 of the fused-dispatch lever: loading the stock 27B
/// through the same claim-driven filtered loader the session uses must keep all
/// 48 `com.microsoft::LinearAttention` calls as fused ops and inline ZERO of
/// their `Scan` bodies. This is the "0 LinearAttention Scans" gate.
#[test]
#[ignore = "requires the real Qwen3.6-27B int4 hybrid export and a CUDA device"]
fn stock_export_keeps_linear_attention_as_fused_op_with_zero_scans() -> anyhow::Result<()> {
    use onnx_runtime_ep_api::ExecutionProvider;

    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let ep = match onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        Ok(ep) => ep,
        Err(error) => {
            eprintln!("skipping 0-Scan structural check: CUDA unavailable: {error}");
            return Ok(());
        }
    };

    let bytes = std::fs::read(dir.join("model.onnx"))?;
    // Same predicate the session builds: keep a function call as an op iff this
    // EP's claim gate reports a fused kernel for it.
    let keep = |node: &onnx_runtime_ir::Node, opset: u64, dtypes: &[onnx_runtime_ir::DataType]| {
        ep.supports_op(node, opset, &[], dtypes, &[]).is_supported()
    };
    let (graph, _weights) =
        onnx_runtime_loader::load_model_bytes_with_weights_filtered(&bytes, &dir, &keep)?;

    let mut linear_attention = 0usize;
    let mut scans = 0usize;
    for id in graph.topological_order().expect("loaded graph is a DAG") {
        match graph.node(id).op_type.as_str() {
            "LinearAttention" => linear_attention += 1,
            "Scan" => scans += 1,
            _ => {}
        }
    }
    eprintln!("loaded 27B plan: LinearAttention fused ops={linear_attention}, Scan ops={scans}");
    assert_eq!(
        linear_attention, 48,
        "expected 48 fused LinearAttention ops kept in the runtime plan"
    );
    assert_eq!(
        scans, 0,
        "expected 0 Scans — every LinearAttention must stay a fused op, not inline its Scan body"
    );
    Ok(())
}
