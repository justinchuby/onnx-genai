//! GAP-3 Inc-A — native pipeline *backend selection* parity (issue #384).
//!
//! Proves the pure-native pipeline **construction + decode** path: requesting
//! `EngineDecodeBackend::Native` for a multi-component (`embedding` every_step →
//! `decoder`) autoregressive pipeline now *constructs a working engine and
//! decodes*, instead of erroring at construction with
//! "native pipeline decode is not yet implemented".
//!
//! Two-tier correctness bar (per the Inc-A design note):
//!   1. **Differential:** pure-native selection (`decode_backend = Native`, no env
//!      flags) produces the *same* generated token ids as the hybrid env-flag
//!      injection path (`ONNX_GENAI_PIPELINE_NATIVE_DECODER=decoder` +
//!      `ONNX_GENAI_PIPELINE_NATIVE_STEP_COMPONENTS=embedding`) — proving the two
//!      native selection sources converge on the same builders.
//!   2. **ORT oracle:** both native paths match the default ORT decode of the same
//!      fixture (its ops are ORT-runnable, so ORT is a real token oracle).
//!
//! Non-vacuity: if construction reverts to the old bail the pure-native run
//! `?`-propagates an error and the test fails; if native selection diverges from
//! ORT/hybrid the token vectors differ and the asserts fire. Each fixture head is
//! closed-form (`[3, 7] -> [0, 5, 6, 7]`), pinning the expected tokens.
//!
//! The `vision_encoder` stage of both fixtures is `prompt_only` and stays on ORT
//! under Inc-A (native prologue is Inc-B); only `embedding` (every_step) and
//! `decoder` run natively here.

use std::path::{Path, PathBuf};

use onnx_genai_engine::pipeline::PipelineGenerateRequest;
use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateOptions, GeneratePrompt, GenerateRequest,
};
use onnx_genai_ort::Value;

const NATIVE_DECODER_ENV: &str = "ONNX_GENAI_PIPELINE_NATIVE_DECODER";
const NATIVE_STEP_COMPONENTS_ENV: &str = "ONNX_GENAI_PIPELINE_NATIVE_STEP_COMPONENTS";
const NATIVE_DECODER_DEVICE_ENV: &str = "ONNX_GENAI_PIPELINE_NATIVE_DECODER_DEVICE";

const EXPECTED_TOKENS: [u32; 4] = [0, 5, 6, 7];

fn fixture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn tiny_pixels() -> anyhow::Result<Value> {
    // pixel_values[1,3,2,2] = i/12; the vision encoder means over channels.
    Value::from_vec_f32((0..12).map(|i| i as f32 / 12.0).collect(), &[1, 3, 2, 2])
        .map_err(Into::into)
}

/// Which native backend selection to drive for one generation.
#[derive(Clone, Copy)]
enum Selection {
    /// Default ORT decode (the token oracle) — no native components.
    Ort,
    /// Hybrid env-flag injection: decoder + embedding selected natively while the
    /// backend stays ORT.
    HybridEnv,
    /// Pure-native backend selection (`decode_backend = Native`): every component
    /// runs natively with no env flags.
    PureNative,
}

fn clear_env() {
    unsafe {
        std::env::remove_var(NATIVE_DECODER_ENV);
        std::env::remove_var(NATIVE_STEP_COMPONENTS_ENV);
        std::env::remove_var(NATIVE_DECODER_DEVICE_ENV);
    }
}

/// One composite generation over `fixture` under the given selection, returning
/// the generated token ids. `decoder_device` pins the native decoder device
/// (`None` = CPU default); the fixture's closed-form head makes the ids exact and
/// device-independent regardless.
fn generate_tokens(
    fixture: &str,
    selection: Selection,
    decoder_device: Option<&str>,
) -> anyhow::Result<Vec<u32>> {
    // Process-global env: set/clear around the single engine construction that
    // reads it. This integration binary owns its own process, and the cases run
    // sequentially, so the two env-sensitive constructions never interleave.
    clear_env();
    if let Some(device) = decoder_device {
        unsafe { std::env::set_var(NATIVE_DECODER_DEVICE_ENV, device) };
    }
    let mut config = EngineConfig::default();
    match selection {
        Selection::Ort => {
            config.decode_backend = EngineDecodeBackend::Ort;
        }
        Selection::HybridEnv => {
            config.decode_backend = EngineDecodeBackend::Ort;
            unsafe {
                std::env::set_var(NATIVE_DECODER_ENV, "decoder");
                std::env::set_var(NATIVE_STEP_COMPONENTS_ENV, "embedding");
            }
        }
        Selection::PureNative => {
            config.decode_backend = EngineDecodeBackend::Native;
        }
    }

    let result = (|| {
        let mut engine = Engine::from_pipeline_dir(&fixture_dir(fixture), config)?;
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![3, 7]));
        request.options = GenerateOptions {
            max_new_tokens: 4,
            temperature: 0.0,
            stop_on_eos: false,
            ..GenerateOptions::default()
        };
        let pipeline_request = PipelineGenerateRequest::new(request)
            .with_input("vision_encoder.pixel_values", tiny_pixels()?);
        let result = engine.generate_with_pipeline_request(pipeline_request)?;
        Ok::<_, anyhow::Error>(result.token_ids)
    })();

    clear_env();
    result
}

/// CPU three-way parity on `tiny-gemma4-vlm`, whose decoder attention runs under
/// the ORT CPU kernel (a real token oracle) as well as the native CPU backend;
/// then, when a GPU is visible, a CUDA differential on the task's named
/// `tiny-gqa-embeds-cuda`.
///
/// Both cases live in one `#[test]` so they run sequentially in a single thread:
/// the native decoder device is selected from a process-global env var, so
/// splitting them into two test functions would let cargo's default parallel test
/// threads race that var.
#[test]
fn pure_native_pipeline_selection_matches_ort_and_hybrid() -> anyhow::Result<()> {
    // --- CPU: full three-way ORT / hybrid / pure-native parity. ---
    const CPU_FIXTURE: &str = "tiny-gemma4-vlm";

    // ORT oracle.
    let ort_tokens = generate_tokens(CPU_FIXTURE, Selection::Ort, None)?;
    assert_eq!(
        ort_tokens,
        EXPECTED_TOKENS.to_vec(),
        "ORT baseline drifted from the fixture's closed-form tokens"
    );

    // Hybrid env-flag native injection == ORT.
    let hybrid_tokens = generate_tokens(CPU_FIXTURE, Selection::HybridEnv, None)?;
    assert_eq!(
        hybrid_tokens, ort_tokens,
        "hybrid env-flag native decode diverged from the ORT oracle"
    );

    // Pure-native backend selection: this construction previously errored with
    // "native pipeline decode is not yet implemented" — reaching here at all
    // proves the bail is gone (non-vacuous).
    let native_tokens = generate_tokens(CPU_FIXTURE, Selection::PureNative, None)?;
    assert_eq!(
        native_tokens, hybrid_tokens,
        "pure-native backend selection diverged from the hybrid native path"
    );
    assert_eq!(
        native_tokens, ort_tokens,
        "pure-native backend selection diverged from the ORT oracle"
    );
    eprintln!(
        "gap3 inc-a cpu backend-selection parity: ort={ort_tokens:?} hybrid={hybrid_tokens:?} \
         pure_native={native_tokens:?}"
    );

    // --- CUDA: differential on the task's named GQA fixture (GPU-gated). ---
    // Its decoder routes KV through a real `GroupQueryAttention` op whose ORT CPU
    // kernel rejects the fixture's `head_size` (so there is no ORT-CPU oracle
    // here); the native CUDA decode path engages whole-graph capture. Assert
    // pure-native selection == hybrid env-flag injection (both native, CUDA
    // decoder) and both == the closed-form tokens.
    if !cuda_device_visible() {
        eprintln!("skipping CUDA GQA differential: no CUDA GPU visible (set CUDA_VISIBLE_DEVICES)");
        return Ok(());
    }
    const CUDA_FIXTURE: &str = "tiny-gqa-embeds-cuda";
    let device = Some("cuda:0");

    let cuda_hybrid = generate_tokens(CUDA_FIXTURE, Selection::HybridEnv, device)?;
    assert_eq!(
        cuda_hybrid,
        EXPECTED_TOKENS.to_vec(),
        "hybrid env-flag native CUDA decode drifted from the closed-form tokens"
    );

    let cuda_native = generate_tokens(CUDA_FIXTURE, Selection::PureNative, device)?;
    assert_eq!(
        cuda_native, cuda_hybrid,
        "pure-native backend selection diverged from the hybrid native CUDA path"
    );
    eprintln!(
        "gap3 inc-a cuda gqa backend-selection parity: hybrid={cuda_hybrid:?} \
         pure_native={cuda_native:?}"
    );
    Ok(())
}

fn cuda_device_visible() -> bool {
    std::env::var_os("CUDA_VISIBLE_DEVICES").is_some()
}
