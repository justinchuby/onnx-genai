//! End-to-end parity for the opt-in Marlin int4 M>1 tensor-core GEMM.
//!
//! Prefill runs the `com.microsoft::MatMulNBits` op at M = prompt length (M>1),
//! so a prompt of several tokens exercises the Marlin path while decode stays on
//! the M=1 GEMV. This locks that enabling the Marlin M>1 GEMM
//! (`ONNX_GENAI_MARLIN_M_GT_1=1`) does not change the greedy token stream versus
//! the portable tiled GEMM on real int4 models with asymmetric zero points
//! (glm-4-9b, qwen2.5-14b).
//!
//! ```bash
//! CUDA_VISIBLE_DEVICES=7 cargo test -p onnx-genai-engine \
//!   --features cuda,native-backend --test marlin_m_gt_1_e2e \
//!   -- --ignored --nocapture
//! ```
#![cfg(feature = "native-cuda")]

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateRequest, GenerateResult, NativeDecodeDevice,
};

const GLM_DEFAULT_DIR: &str = "/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda";
const QWEN_DEFAULT_DIR: &str = "/home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx";

// A multi-token prompt so prefill runs MatMulNBits at M>1 (the Marlin path).
const PROMPT: &str = "List three European capital cities and the countries they belong to.";
const MAX_NEW_TOKENS: usize = 24;

fn model_dir(env_key: &str, default_dir: &str) -> Option<PathBuf> {
    let dir = std::env::var_os(env_key)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default_dir));
    let required = ["model.onnx", "model.onnx.data", "tokenizer.json"];
    let missing: Vec<_> = required
        .iter()
        .filter(|name| !dir.join(name).is_file())
        .collect();
    if missing.is_empty() {
        Some(dir)
    } else {
        eprintln!(
            "skipping Marlin M>1 e2e: model directory {} is missing {}",
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

fn generate(dir: &Path) -> anyhow::Result<GenerateResult> {
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
    engine.generate(request)
}

/// Number of extra tiled generations used to decide whether the first
/// tiled/Marlin divergence is a near-tie (see [`assert_marlin_matches_tiled`]).
///
/// Sized to mirror the empirical evidence in PR #962: a 4-run tiled A/B on
/// qwen2.5-14b-int4 showed the *tiled* reference flipping the token-19 argmax on
/// 1 of 4 runs while Marlin stayed deterministic across all runs — i.e. the tiled
/// GEMM's fp32 atomic reduction order is the nondeterministic side at a near-tie.
const TIE_PROBE_RUNS: usize = 4;

/// Verdict of comparing a tiled greedy stream against a Marlin greedy stream,
/// using extra tiled "probe" streams to classify the first divergence.
#[derive(Debug, PartialEq, Eq)]
enum ParityVerdict {
    /// The two streams are identical (full-strength match).
    Identical,
    /// The streams share a prefix but differ in length.
    LengthMismatch { tiled: usize, marlin: usize },
    /// First divergence at `position` is a confirmed near-tie: a tiled probe
    /// reproduced the shared prefix yet produced a different token there
    /// (`prefix_unstable` = no probe could even reproduce the prefix, so the
    /// prefix region itself is tie-dominated). Not a Marlin regression.
    NearTie {
        position: usize,
        prefix_unstable: bool,
    },
    /// First divergence at `position` where the tiled reference stayed
    /// deterministic across every probe — a genuine Marlin regression.
    Regression {
        position: usize,
        tiled: u32,
        marlin: u32,
    },
}

/// Classifies a tiled-vs-Marlin greedy-stream comparison. Pure and GPU-free so
/// the near-tie logic is unit-tested deterministically (see the tests below).
///
/// `probes` are independent re-runs of the *tiled* configuration. Greedy decode
/// is autoregressive and at a near-tie the argmax is nondeterministic (fp
/// atomics), so a single early flip cascades. We therefore look only at the first
/// divergence `d`: if a tiled probe reproduces the identical shared prefix
/// `tiled[..d]` but yields a different token at `d`, the tiled reference is itself
/// nondeterministic there (a near-tie, not a Marlin regression). Only a
/// divergence at a position where every prefix-matching probe agrees with the
/// tiled reference is a regression.
fn classify_parity(tiled: &[u32], marlin: &[u32], probes: &[Vec<u32>]) -> ParityVerdict {
    let Some(d) = tiled.iter().zip(marlin).position(|(t, m)| t != m) else {
        return if tiled.len() == marlin.len() {
            ParityVerdict::Identical
        } else {
            ParityVerdict::LengthMismatch {
                tiled: tiled.len(),
                marlin: marlin.len(),
            }
        };
    };

    let prefix = &tiled[..d];
    let tiled_tok = tiled[d];
    let mut prefix_reproduced = false;
    for probe in probes {
        if probe.get(..d) != Some(prefix) {
            continue;
        }
        prefix_reproduced = true;
        if probe.get(d) != Some(&tiled_tok) {
            return ParityVerdict::NearTie {
                position: d,
                prefix_unstable: false,
            };
        }
    }

    if !prefix_reproduced {
        return ParityVerdict::NearTie {
            position: d,
            prefix_unstable: true,
        };
    }

    ParityVerdict::Regression {
        position: d,
        tiled: tiled_tok,
        marlin: marlin[d],
    }
}

/// Asserts the opt-in Marlin M>1 path does not regress the greedy token stream
/// versus the portable tiled GEMM.
///
/// Subtlety (why this is not a plain `assert_eq!` of two streams): greedy decode
/// is autoregressive, and at a near-degenerate argmax (a near-tie) the chosen
/// token is nondeterministic run-to-run for *either* configuration — the flip
/// comes from fp atomic/reduction-order nondeterminism in the pipeline (the tiled
/// GEMM's fp32 atomics, attention reductions), not from a Marlin error. A single
/// early tie flip then cascades and desyncs the entire tail. See
/// [`classify_parity`] for the near-tie classification (unit-tested separately).
fn assert_marlin_matches_tiled(dir: &Path, label: &str) -> anyhow::Result<()> {
    // SAFETY: ignored e2e test runs serially; no concurrent readers of the flag.
    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cuda");
        // Marlin M>1 is default-ON, so the tiled reference arm opts out
        // explicitly; unsetting the variable would select Marlin for both arms.
        std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "0");
    }
    let tiled = generate(dir)?;

    // SAFETY: see above.
    unsafe {
        std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "1");
    }
    let marlin = generate(dir);

    // SAFETY: clear the flag regardless of the result so it cannot leak.
    unsafe {
        std::env::remove_var("ONNX_GENAI_MARLIN_M_GT_1");
    }
    let marlin = marlin?;

    eprintln!("[{label}] tiled : {:?}", tiled.token_ids);
    eprintln!("[{label}] marlin: {:?}", marlin.token_ids);

    // Only pay for tiled probe re-runs if the two streams actually diverge.
    let diverges = tiled
        .token_ids
        .iter()
        .zip(&marlin.token_ids)
        .any(|(t, m)| t != m)
        || tiled.token_ids.len() != marlin.token_ids.len();
    let mut probes: Vec<Vec<u32>> = Vec::new();
    if diverges {
        eprintln!("[{label}] streams diverge — probing whether the first divergence is a near-tie");
        // These probes must re-run the *tiled reference* to expose its own
        // run-to-run nondeterminism, so they opt out of the (default-ON) Marlin
        // path for the duration of the loop.
        // SAFETY: ignored e2e test runs serially; no concurrent readers.
        unsafe {
            std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "0");
        }
        let collected = (0..TIE_PROBE_RUNS)
            .map(|_| generate(dir).map(|run| run.token_ids))
            .collect::<anyhow::Result<Vec<_>>>();
        // SAFETY: clear regardless of the result so it cannot leak.
        unsafe {
            std::env::remove_var("ONNX_GENAI_MARLIN_M_GT_1");
        }
        probes = collected?;
    }

    match classify_parity(&tiled.token_ids, &marlin.token_ids, &probes) {
        ParityVerdict::Identical => {
            eprintln!(
                "[{label}] greedy streams identical ({} tokens) — full-strength match",
                tiled.token_ids.len()
            );
            Ok(())
        }
        ParityVerdict::NearTie {
            position,
            prefix_unstable,
        } => {
            eprintln!(
                "[{label}] first divergence at token {position} is a near-tie \
                 (prefix_unstable={prefix_unstable}) — nondeterministic in the tiled reference \
                 itself, not a Marlin regression"
            );
            Ok(())
        }
        ParityVerdict::LengthMismatch { tiled, marlin } => {
            panic!(
                "[{label}] Marlin and tiled greedy streams differ in length: {tiled} vs {marlin}"
            )
        }
        ParityVerdict::Regression {
            position,
            tiled: tiled_tok,
            marlin: marlin_tok,
        } => {
            panic!(
                "[{label}] Marlin M>1 changed the greedy token at position {position} \
                 (tiled={tiled_tok}, marlin={marlin_tok}) while the tiled reference stayed \
                 deterministic there across {TIE_PROBE_RUNS} probes — a genuine regression"
            )
        }
    }
}

#[test]
fn classify_parity_identical_streams() {
    let s = vec![1, 2, 3, 4];
    assert_eq!(classify_parity(&s, &s, &[]), ParityVerdict::Identical);
}

#[test]
fn classify_parity_flags_length_mismatch() {
    assert_eq!(
        classify_parity(&[1, 2, 3], &[1, 2, 3, 4], &[]),
        ParityVerdict::LengthMismatch {
            tiled: 3,
            marlin: 4
        }
    );
}

#[test]
fn classify_parity_confirms_near_tie_from_probe() {
    // Streams diverge at index 2; a tiled probe reproduces the prefix [1,2] but
    // yields a different token (99) there → confirmed near-tie.
    let tiled = vec![1, 2, 30, 40];
    let marlin = vec![1, 2, 31, 41];
    let probes = vec![vec![1, 2, 99, 7]];
    assert_eq!(
        classify_parity(&tiled, &marlin, &probes),
        ParityVerdict::NearTie {
            position: 2,
            prefix_unstable: false
        }
    );
}

#[test]
fn classify_parity_reports_regression_when_tiled_is_deterministic() {
    // Diverge at index 2; every prefix-matching probe agrees with tiled (30) →
    // tiled is deterministic there, so Marlin's 31 is a real regression.
    let tiled = vec![1, 2, 30, 40];
    let marlin = vec![1, 2, 31, 41];
    let probes = vec![vec![1, 2, 30, 40], vec![1, 2, 30, 99]];
    assert_eq!(
        classify_parity(&tiled, &marlin, &probes),
        ParityVerdict::Regression {
            position: 2,
            tiled: 30,
            marlin: 31
        }
    );
}

#[test]
fn classify_parity_near_tie_when_prefix_never_reproduced() {
    // Diverge at index 2, but no probe reproduces the prefix [1,2] (the prefix
    // region is itself tie-dominated) → treated as a near-tie, not a regression.
    let tiled = vec![1, 2, 30, 40];
    let marlin = vec![1, 2, 31, 41];
    let probes = vec![vec![1, 77, 30, 40], vec![9, 2, 30, 40]];
    assert_eq!(
        classify_parity(&tiled, &marlin, &probes),
        ParityVerdict::NearTie {
            position: 2,
            prefix_unstable: true
        }
    );
}

/// The M>1 tiled-fallback kernel variants. Every one of these fires only inside
/// the `if m > 1` dispatch arms, so seeing any of them under `ONNX_GENAI_MARLIN_M_GT_1=1`
/// means a MatMulNBits node still escaped the Marlin path at M>1 — which would keep
/// that node declaring `KernelCaptureUnsupported` and keep the captured forward segmented.
const TILED_M_GT_1_VARIANTS: &[&str] = &[
    "gemm_f16_tiled",
    "gemm_f16_tiled_rmsnorm",
    "gate_up_swiglu_prefill",
    "gate_up_swiglu_rmsnorm_prefill",
];

/// The Marlin M>1 variants that should serve every hot MatMulNBits node.
const MARLIN_M_GT_1_VARIANTS: &[&str] = &[
    "gemm_marlin_int4",
    "gemm_marlin_int4_rmsnorm",
    "gate_up_swiglu_marlin_prefill",
];

/// Per-node coverage audit: run a prefill-heavy generation with Marlin enabled,
/// collect the per-op kernel-variant annotations from the runtime tracer, and
/// assert that ZERO MatMulNBits nodes fell back to a tiled M>1 kernel. This is
/// the machine-checkable form of "drive M>1 tiled-fallback count to zero", the
/// prerequisite for the capture `segments -> 1` collapse.
fn audit_marlin_coverage(dir: &Path, label: &str) -> anyhow::Result<()> {
    use onnx_runtime_tracer::TraceVerbosity;
    use std::collections::BTreeMap;

    // SAFETY: ignored e2e test runs serially; no concurrent readers of the flag.
    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cuda");
        std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "1");
    }

    onnx_genai_engine::runtime_trace::reset();
    onnx_genai_engine::runtime_trace::set_recording(true, TraceVerbosity::Decisions);

    let result = generate(dir);

    onnx_genai_engine::runtime_trace::set_recording(false, TraceVerbosity::Ops);
    let events = onnx_genai_engine::runtime_trace::collected_events();

    // SAFETY: clear the flag regardless of the result so it cannot leak.
    unsafe {
        std::env::remove_var("ONNX_GENAI_MARLIN_M_GT_1");
    }
    result?;

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for event in &events {
        if let Some(args) = &event.args
            && let Some(variant) = args.get("kernel_variant").and_then(|v| v.as_str())
        {
            *counts.entry(variant.to_string()).or_default() += 1;
        }
    }

    eprintln!("[{label}] per-variant kernel counts (Marlin M>1 enabled):");
    for (variant, count) in &counts {
        eprintln!("  {count:>6}  {variant}");
    }

    let tiled_hits: Vec<(&str, usize)> = TILED_M_GT_1_VARIANTS
        .iter()
        .filter_map(|name| counts.get(*name).map(|c| (*name, *c)))
        .collect();
    let marlin_total: usize = MARLIN_M_GT_1_VARIANTS
        .iter()
        .filter_map(|name| counts.get(*name))
        .sum();

    eprintln!(
        "[{label}] Marlin M>1 node dispatches: {marlin_total}; tiled M>1 fallbacks: {}",
        tiled_hits.iter().map(|(_, c)| c).sum::<usize>()
    );

    assert!(
        marlin_total > 0,
        "[{label}] no Marlin M>1 variant fired — the audit did not exercise the M>1 path"
    );
    assert!(
        tiled_hits.is_empty(),
        "[{label}] MatMulNBits nodes fell back to a tiled M>1 kernel: {tiled_hits:?} — \
         these still declare KernelCaptureUnsupported at M>1"
    );
    Ok(())
}

#[test]
#[ignore = "requires the real glm-4-9b-int4 export and a CUDA device"]
fn marlin_m_gt_1_coverage_audit_on_glm_4_9b_int4() -> anyhow::Result<()> {
    let Some(dir) = model_dir("GLM_4_9B_CUDA_E2E_DIR", GLM_DEFAULT_DIR) else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping Marlin M>1 coverage audit (glm): CUDA unavailable: {error}");
        return Ok(());
    }
    audit_marlin_coverage(&dir, "glm-4-9b-int4")
}

#[test]
#[ignore = "requires the real qwen2.5-14b-int4-zp export and a CUDA device"]
fn marlin_m_gt_1_coverage_audit_on_qwen2_5_14b_int4() -> anyhow::Result<()> {
    let Some(dir) = model_dir("QWEN2_5_14B_CUDA_E2E_DIR", QWEN_DEFAULT_DIR) else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping Marlin M>1 coverage audit (qwen): CUDA unavailable: {error}");
        return Ok(());
    }
    audit_marlin_coverage(&dir, "qwen2.5-14b-int4")
}

#[test]
#[ignore = "requires the real glm-4-9b-int4 export and a CUDA device"]
fn marlin_m_gt_1_matches_tiled_on_glm_4_9b_int4() -> anyhow::Result<()> {
    let Some(dir) = model_dir("GLM_4_9B_CUDA_E2E_DIR", GLM_DEFAULT_DIR) else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping Marlin M>1 e2e (glm): CUDA unavailable: {error}");
        return Ok(());
    }
    assert_marlin_matches_tiled(&dir, "glm-4-9b-int4")
}

#[test]
#[ignore = "requires the real qwen2.5-14b-int4-zp export and a CUDA device"]
fn marlin_m_gt_1_matches_tiled_on_qwen2_5_14b_int4() -> anyhow::Result<()> {
    let Some(dir) = model_dir("QWEN2_5_14B_CUDA_E2E_DIR", QWEN_DEFAULT_DIR) else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping Marlin M>1 e2e (qwen): CUDA unavailable: {error}");
        return Ok(());
    }
    assert_marlin_matches_tiled(&dir, "qwen2.5-14b-int4")
}
