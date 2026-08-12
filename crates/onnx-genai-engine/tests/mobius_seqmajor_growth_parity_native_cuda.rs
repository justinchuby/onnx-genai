//! Real-model seq-major KV parity **with forced growth**, native CUDA decode.
//!
//! This is the end-to-end validation of the fixed-stride BSNH KV state build
//! (the documented next step after PR #797). #794 patched `qwen2.5-0.5b-q4_0-
//! mobius` with `kv_layout=1` and got identical 32 token IDs — but with the
//! default 256-token initial KV bucket a 32-token run **never grows**, so it
//! never exercised seq-major *growth* geometry, which is exactly what the engine
//! shape/stride build corrects. This harness forces growth with a tiny
//! `ONNX_GENAI_KV_MIN_BUCKET` and asserts the head-major and seq-major token
//! streams are byte-identical across several growth events, capture ON and OFF,
//! under both KV growth mechanisms.
//!
//! ## The two decoupled seq-major levers
//!
//! The engine growth geometry (`ONNX_GENAI_CUDA_KV_LAYOUT`) and the CUDA GQA
//! kernel stride arithmetic (the ONNX `kv_layout` node attribute) are two
//! *separate* levers — a correct seq-major run needs both. So this test uses two
//! model directories:
//!
//! * head-major: the stock export (GQA nodes have no `kv_layout` attribute →
//!   BNSH), run with the default engine layout;
//! * seq-major: a copy whose 24 GQA nodes carry `kv_layout=1` (→ BSNH), run with
//!   `ONNX_GENAI_CUDA_KV_LAYOUT=seq_major`.
//!
//! Patch the seq-major copy once with:
//!
//! ```python
//! import onnx
//! from onnx import helper
//! m = onnx.load("model.onnx", load_external_data=False)
//! for n in m.graph.node:
//!     if n.op_type == "GroupQueryAttention":
//!         n.attribute.append(helper.make_attribute("kv_layout", 1))
//! onnx.save(m, "model.onnx", save_as_external_data=False)  # keep model.onnx.data
//! ```
//!
//! ## The two KV growth mechanisms
//!
//! * **In-place VMM** (the default since #798's managed no-spill VMM;
//!   `commits_on_demand`): growth maps fresh granules onto the *same* base VA. A
//!   seq-major buffer's per-token stride `kv_heads*head_dim` is
//!   capacity-independent, so the live prefix keeps its byte offsets and **no KV
//!   data moves** (`d2d_copy_bytes == 0`), while head-major re-strides every head
//!   stripe. This is the fixed-stride property #797 measured at the driver level,
//!   here on the engine growth path.
//! * **Reallocation** (`ONNX_GENAI_LEGACY_ALLOCATOR=1`, the #755 opt-out): a
//!   bucket-sized growth allocates a *fresh* buffer and copies the live prefix
//!   into it. A new buffer must be filled either way, so seq-major copies the
//!   same byte count as head-major — its only win here is one contiguous copy
//!   instead of `kv_heads` stripes.
//!
//! Run:
//!
//! ```bash
//! MOBIUS_SEQMAJOR_HEAD_DIR=/models/qwen2.5-0.5b-q4_0-mobius \
//! MOBIUS_SEQMAJOR_SEQ_DIR=/models/qwen2.5-0.5b-q4_0-mobius-seqmajor \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine \
//!   --features cuda,native-backend \
//!   --test mobius_seqmajor_growth_parity_native_cuda \
//!   -- --ignored --nocapture --test-threads=1
//! ```
#![cfg(all(feature = "cuda", feature = "native-backend"))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use onnx_genai_engine::{
    CudaKvDebugStats, Engine, EngineConfig, EngineDecodeBackend, GenerateRequest, GenerateResult,
    NativeDecodeDevice,
};

const PROMPT: &str = "The capital of France is";
/// Long enough that a `min_bucket = 8` run crosses several power-of-two bucket
/// boundaries (8 → 16 → 32 → 64), so growth actually happens.
const MAX_NEW_TOKENS: usize = 48;
/// Force growth: initial and subsequent KV buckets round up from here.
const FORCE_GROWTH_MIN_BUCKET: &str = "8";

const DEFAULT_HEAD_DIR: &str = r"C:\Users\justinchu\dev\models\qwen2.5-0.5b-q4_0-mobius";
const DEFAULT_SEQ_DIR: &str = r"C:\Users\justinchu\dev\models\qwen2.5-0.5b-q4_0-mobius-seqmajor";

fn resolve_dir(var: &str, default: &str) -> Option<PathBuf> {
    let dir = std::env::var_os(var)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default));
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
            "skipping mobius seq-major growth parity: {} ({}) missing {}",
            var,
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

struct Run {
    result: GenerateResult,
    stats: Option<CudaKvDebugStats>,
    elapsed_ms: f64,
}

#[derive(Clone, Copy)]
struct Mode {
    seq_major: bool,
    capture: bool,
    vmm_arena: bool,
}

/// Generate greedily on `dir` under `mode`. `ONNX_GENAI_KV_MIN_BUCKET` is set on
/// every run so both layouts exercise the same growth schedule.
fn generate(dir: &Path, mode: Mode) -> anyhow::Result<Run> {
    // SAFETY: single-threaded (`--test-threads=1`); we set process env before
    // constructing the engine, which reads these at load time.
    unsafe {
        std::env::set_var("ONNX_GENAI_KV_MIN_BUCKET", FORCE_GROWTH_MIN_BUCKET);
        std::env::set_var(
            "ONNX_GENAI_CUDA_GRAPH",
            if mode.capture { "1" } else { "0" },
        );
        if mode.vmm_arena {
            // In-place VMM arena (the default since #798's managed no-spill VMM).
            std::env::set_var("ONNX_GENAI_CUDA_VMM", "1");
            std::env::remove_var("ONNX_GENAI_LEGACY_ALLOCATOR");
        } else {
            // Force the legacy reallocation allocator. Since #798 the managed VMM
            // arena is auto-enabled (`auto_dynamic_lending`) even without
            // `ONNX_GENAI_CUDA_VMM`, so the realloc path must be opted into via
            // the #755 legacy knob or this phase would silently run VMM-backed.
            std::env::remove_var("ONNX_GENAI_CUDA_VMM");
            std::env::set_var("ONNX_GENAI_LEGACY_ALLOCATOR", "1");
        }
        if mode.seq_major {
            std::env::set_var("ONNX_GENAI_CUDA_KV_LAYOUT", "seq_major");
        } else {
            std::env::remove_var("ONNX_GENAI_CUDA_KV_LAYOUT");
        }
    }

    let mut engine = Engine::from_dir(
        dir,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cuda { index: Some(0) }),
            ..EngineConfig::default()
        },
    )?;
    let mut request = GenerateRequest::new(PROMPT.to_string());
    request.options.max_new_tokens = MAX_NEW_TOKENS;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;

    let start = Instant::now();
    let result = engine.generate(request)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let stats = engine.native_cuda_debug_stats();
    Ok(Run {
        result,
        stats,
        elapsed_ms,
    })
}

fn report(label: &str, run: &Run) {
    let tok = run.result.token_ids.len();
    eprint!("[{label}] tokens={tok} wall={:.1}ms", run.elapsed_ms);
    if let Some(s) = &run.stats {
        eprint!(
            " seq_major={} committed_bytes={} growth_events={} d2d_copy_bytes={} \
             captures={} invalidations={} replays={} max_len={}",
            s.kv_layout_seq_major,
            s.kv_committed_bytes,
            s.kv_growth_events,
            s.kv_growth_d2d_copy_bytes,
            s.graph.captures,
            s.graph.invalidations,
            s.graph.replays,
            s.max_len,
        );
    } else {
        eprint!(" (no native CUDA stats)");
    }
    eprintln!();
}

/// Run head-major then seq-major under one (growth-mechanism, capture) mode and
/// assert byte-identical output. Returns the (head, seq) stats for measurement.
fn parity_phase(
    head_dir: &Path,
    seq_dir: &Path,
    vmm_arena: bool,
    capture: bool,
) -> anyhow::Result<(Option<CudaKvDebugStats>, Option<CudaKvDebugStats>)> {
    let growth = if vmm_arena { "vmm-inplace" } else { "realloc" };
    let cap = if capture { "capture=ON" } else { "capture=OFF" };
    eprintln!("=== mobius seq-major growth parity, {growth}, {cap} ===");

    let head = generate(
        head_dir,
        Mode {
            seq_major: false,
            capture,
            vmm_arena,
        },
    )?;
    report(&format!("head-major {growth} {cap}"), &head);
    let seq = generate(
        seq_dir,
        Mode {
            seq_major: true,
            capture,
            vmm_arena,
        },
    )?;
    report(&format!("seq-major  {growth} {cap}"), &seq);

    assert!(
        !head.result.token_ids.is_empty(),
        "head-major generated no tokens ({growth}, {cap})"
    );
    assert_eq!(
        head.result.token_ids, seq.result.token_ids,
        "seq-major token stream diverged from head-major ({growth}, {cap}):\n head={:?}\n seq ={:?}",
        head.result.token_ids, seq.result.token_ids
    );
    assert_eq!(
        head.result.text, seq.result.text,
        "seq-major text diverged from head-major ({growth}, {cap})"
    );

    if let (Some(hs), Some(ss)) = (&head.stats, &seq.stats) {
        assert!(
            hs.kv_growth_events > 0 && ss.kv_growth_events > 0,
            "growth did not happen ({growth}, {cap}); raise MAX_NEW_TOKENS or lower min_bucket"
        );
        assert!(
            ss.kv_layout_seq_major && !hs.kv_layout_seq_major,
            "layout resolution wrong ({growth}, {cap}): head.seq_major={}, seq.seq_major={}; \
             check the patched model's kv_layout attribute and the env override",
            hs.kv_layout_seq_major,
            ss.kv_layout_seq_major
        );
    }
    Ok((head.stats, seq.stats))
}

#[test]
#[ignore = "requires the stock + kv_layout=1 mobius exports and a CUDA device"]
fn mobius_seq_major_growth_is_bit_identical_to_head_major() -> anyhow::Result<()> {
    let (Some(head_dir), Some(seq_dir)) = (
        resolve_dir("MOBIUS_SEQMAJOR_HEAD_DIR", DEFAULT_HEAD_DIR),
        resolve_dir("MOBIUS_SEQMAJOR_SEQ_DIR", DEFAULT_SEQ_DIR),
    ) else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping mobius seq-major growth parity: CUDA unavailable: {error}");
        return Ok(());
    }

    // Phase 1 — in-place VMM growth (the default since #798's managed no-spill
    // VMM). This is the path the engine shape/stride change primarily targets:
    // the seq-major fixed stride keeps the live prefix at its byte offsets, so
    // growth copies **zero** KV bytes, while head-major re-strides its head
    // stripes. Core correctness too: byte-identical output, capture OFF then ON.
    let (head_vmm_off, seq_vmm_off) = parity_phase(&head_dir, &seq_dir, true, false)?;
    parity_phase(&head_dir, &seq_dir, true, true)?;
    if let (Some(h), Some(s)) = (&head_vmm_off, &seq_vmm_off) {
        assert_eq!(
            s.kv_growth_d2d_copy_bytes, 0,
            "in-place VMM seq-major growth must move zero KV bytes (fixed stride), got {}",
            s.kv_growth_d2d_copy_bytes
        );
        assert!(
            h.kv_growth_d2d_copy_bytes > 0,
            "in-place VMM head-major growth must re-stride its head stripes (copy > 0), got {}",
            h.kv_growth_d2d_copy_bytes
        );
        eprintln!(
            "FIXED-STRIDE RESULT: in-place VMM growth d2d KV bytes head-major={} vs seq-major={} \
             (0 = no data moved)",
            h.kv_growth_d2d_copy_bytes, s.kv_growth_d2d_copy_bytes
        );
    }

    // Phase 2 — legacy reallocation growth (ONNX_GENAI_LEGACY_ALLOCATOR=1, the
    // #755 opt-out). A fresh buffer is filled on every growth, so seq-major
    // copies the same byte count as head-major (differing only in fragmentation).
    // Kept to prove token parity holds under the legacy allocator too.
    let (head_re_off, seq_re_off) = parity_phase(&head_dir, &seq_dir, false, false)?;
    if let (Some(h), Some(s)) = (&head_re_off, &seq_re_off) {
        assert_eq!(
            s.kv_growth_d2d_copy_bytes, h.kv_growth_d2d_copy_bytes,
            "reallocation growth fills a fresh buffer either way; seq-major should copy the same \
             byte count as head-major, differing only in fragmentation"
        );
    }
    Ok(())
}
