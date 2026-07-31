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

use onnx_genai_engine::pipeline::{PipelineEngine, PipelineGenerateRequest};
use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateOptions, GeneratePrompt, GenerateRequest,
};
use onnx_genai_kv::MaterializedKv;
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

/// One decode turn over `fixture` on an already-built engine, with an explicit
/// prompt and token budget. Used to drive two prefix-sharing requests through
/// the *same* engine so the paged prefix cache persists between them.
fn run_turn(
    engine: &mut PipelineEngine,
    prompt: Vec<u32>,
    max_new_tokens: usize,
) -> anyhow::Result<Vec<u32>> {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(prompt));
    request.options = GenerateOptions {
        max_new_tokens,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };
    let pipeline_request = PipelineGenerateRequest::new(request)
        .with_input("vision_encoder.pixel_values", tiny_pixels()?);
    Ok(engine
        .generate_with_pipeline_request(pipeline_request)?
        .token_ids)
}

/// A single cold decode of `prompt` under `selection` on a fresh engine — no
/// prior turn, so nothing is reused. Serves as the reuse-independent oracle for
/// the warm run below.
fn cold_tokens(
    fixture: &str,
    selection: Selection,
    prompt: Vec<u32>,
    max_new_tokens: usize,
    decoder_device: Option<&str>,
) -> anyhow::Result<Vec<u32>> {
    clear_env();
    if let Some(device) = decoder_device {
        unsafe { std::env::set_var(NATIVE_DECODER_DEVICE_ENV, device) };
    }
    let mut config = EngineConfig {
        page_size: 2,
        ..EngineConfig::default()
    };
    match selection {
        Selection::Ort => config.decode_backend = EngineDecodeBackend::Ort,
        Selection::PureNative => config.decode_backend = EngineDecodeBackend::Native,
        Selection::HybridEnv => {
            config.decode_backend = EngineDecodeBackend::Ort;
            unsafe {
                std::env::set_var(NATIVE_DECODER_ENV, "decoder");
                std::env::set_var(NATIVE_STEP_COMPONENTS_ENV, "embedding");
            }
        }
    }
    let result = (|| {
        let mut engine = Engine::from_pipeline_dir(&fixture_dir(fixture), config)?;
        run_turn(&mut engine, prompt, max_new_tokens)
    })();
    clear_env();
    result
}

/// The exact request shape `run_turn` drives, for `prompt`, so its prefix key
/// (digest of `pixel_values` + presence keys + tokens) matches the one the paged
/// decode path published under. Used only to read published KV back out.
fn prefix_probe_request(prompt: Vec<u32>) -> anyhow::Result<PipelineGenerateRequest> {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(prompt));
    request.options = GenerateOptions {
        max_new_tokens: 2,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };
    Ok(PipelineGenerateRequest::new(request)
        .with_input("vision_encoder.pixel_values", tiny_pixels()?))
}

/// Materialize the paged KV bytes an engine of `selection` publishes for
/// `shared_prompt` after decoding it once — the reuse-independent reference the
/// warm native run's mirrored pages are compared against, byte for byte.
fn published_prefix_kv(
    fixture: &str,
    selection: Selection,
    shared_prompt: Vec<u32>,
    decoder_device: Option<&str>,
) -> anyhow::Result<Option<MaterializedKv>> {
    clear_env();
    if let Some(device) = decoder_device {
        unsafe { std::env::set_var(NATIVE_DECODER_DEVICE_ENV, device) };
    }
    let mut config = EngineConfig {
        page_size: 2,
        ..EngineConfig::default()
    };
    match selection {
        Selection::Ort => config.decode_backend = EngineDecodeBackend::Ort,
        Selection::PureNative => config.decode_backend = EngineDecodeBackend::Native,
        Selection::HybridEnv => {
            config.decode_backend = EngineDecodeBackend::Ort;
            unsafe {
                std::env::set_var(NATIVE_DECODER_ENV, "decoder");
                std::env::set_var(NATIVE_STEP_COMPONENTS_ENV, "embedding");
            }
        }
    }
    let probe = prefix_probe_request(shared_prompt.clone())?;
    let result = (|| {
        let mut engine = Engine::from_pipeline_dir(&fixture_dir(fixture), config)?;
        run_turn(&mut engine, shared_prompt, 2)?;
        engine.materialize_published_prefix_kv(&probe)
    })();
    clear_env();
    result
}

/// GAP-3 Inc-C — paged native pipeline decode with cross-request KV reuse.
///
/// Inc-A drove the pure-native pipeline through the *non-paged* flat-AR path.
/// Inc-C closes the S2 present-KV mirror bail
/// (`pipeline/decoder_component.rs` `NativePipelineDecoder::mirror_last_present_kv`)
/// and adds `load_paged_prefix`, so the native decoder now pages its KV: it
/// mirrors each step's present KV into the shared paged cache **and** seeds a
/// materialized prefix back out of it, through the same `kv_bridge` geometry the
/// ORT decoder uses.
///
/// Two prefix-sharing requests run through ONE pure-native engine over
/// `tiny-gemma4-vlm` (naive/Concat-KV decoder → host-growable f32 KV →
/// `supports_paged_kv`). The test asserts, together:
///   * **reuse engaged** — the warm request reuses `> 0` prefix tokens, proving
///     the mirror-write actually populated the pages and the seed-read consumed
///     them (a silent full-prefill fallback would report zero);
///   * **geometry correct (tokens)** — the warm tokens equal both a cold
///     pure-native run and the ORT oracle;
///   * **geometry correct (bytes)** — the native-mirrored paged KV for the
///     shared prefix is byte-identical to the ORT-mirrored KV. This fixture's
///     argmax is invariant to the reused-prefix KV, so the token asserts above
///     cannot see a key/value swap or a zeroed mirror; the direct
///     materialized-byte comparison closes that gap.
///
/// Non-vacuity: if construction reverts to the S2 bail the paged native run
/// `?`-errors; if the mirror/seed geometry is wrong the byte assert (and, when
/// it happens to also move argmax, the token asserts) fires; if reuse silently
/// no-ops the `reused > 0` assert fires.
#[test]
fn native_paged_prefix_reuse_matches_fresh_and_ort() -> anyhow::Result<()> {
    const FIXTURE: &str = "tiny-gemma4-vlm";
    // Shared prefix, then a continuation that shares it — the second request must
    // reuse the first's mirrored KV rather than re-prefilling the whole prompt.
    let shared_prompt = vec![3u32, 7, 0, 5];
    let warm_prompt = vec![3u32, 7, 0, 5, 6];

    // Reuse-independent oracles for the warm prompt.
    let ort_cold = cold_tokens(FIXTURE, Selection::Ort, warm_prompt.clone(), 3, None)?;
    let native_cold = cold_tokens(FIXTURE, Selection::PureNative, warm_prompt.clone(), 3, None)?;
    assert_eq!(
        native_cold, ort_cold,
        "cold pure-native paged decode diverged from the ORT oracle"
    );

    // One pure-native engine, two turns: the first primes the prefix cache via
    // the native present-KV mirror; the second reuses it via load_paged_prefix.
    clear_env();
    let mut config = EngineConfig {
        page_size: 2,
        ..EngineConfig::default()
    };
    config.decode_backend = EngineDecodeBackend::Native;
    let native_probe = prefix_probe_request(shared_prompt.clone())?;
    let outcome = (|| -> anyhow::Result<(Vec<u32>, usize, Option<MaterializedKv>)> {
        let mut engine = Engine::from_pipeline_dir(&fixture_dir(FIXTURE), config)?;
        let _first = run_turn(&mut engine, shared_prompt.clone(), 2)?;
        engine.reset_cache_stats();
        let warm = run_turn(&mut engine, warm_prompt.clone(), 3)?;
        let reused = engine.cache_stats().prefix_reused_tokens as usize;
        // Read the native-mirrored pages for the shared prefix straight out of
        // the paged cache, before the engine drops.
        let prefix_kv = engine.materialize_published_prefix_kv(&native_probe)?;
        Ok((warm, reused, prefix_kv))
    })();
    clear_env();
    let (warm, reused, native_prefix_kv) = outcome?;

    assert!(
        reused > 0,
        "paged native decode must reuse the shared prefix (reused {reused} tokens); \
         zero reuse means the present-KV mirror never populated the pages"
    );
    assert_eq!(
        warm, native_cold,
        "warm native prefix reuse diverged from a cold native run — the mirrored/seeded \
         present-KV geometry is wrong"
    );
    assert_eq!(
        warm, ort_cold,
        "warm native prefix reuse diverged from the ORT oracle"
    );

    // Geometry, byte-exact. The token asserts above cannot see a key/value swap,
    // a zeroed mirror, or a head/seq/page-offset error, because this fixture's
    // argmax is invariant to the reused-prefix KV (a fully-zeroed mirror still
    // yields identical tokens). Compare the *materialized paged KV bytes* the
    // native mirror wrote for the shared prefix against the ORT mirror's bytes
    // for the same prefix: both mirror through the same `extract_present_token`/
    // `append_token_kv` geometry, so on correct code they are byte-identical,
    // and any mirror corruption diverges them here even when the tokens agree.
    let native_prefix_kv = native_prefix_kv.expect(
        "native paged decode published no shared-prefix KV to read back — the present-KV \
         mirror never populated the pages",
    );
    let ort_prefix_kv = published_prefix_kv(FIXTURE, Selection::Ort, shared_prompt.clone(), None)?
        .expect("ORT paged decode published no shared-prefix KV reference");
    assert_eq!(
        native_prefix_kv, ort_prefix_kv,
        "native-mirrored paged KV for the shared prefix diverged byte-for-byte from the \
         ORT-mirrored KV — the present-KV mirror geometry (key/value order, head/seq/page \
         layout) is wrong even though the argmax tokens matched"
    );
    eprintln!(
        "gap3 inc-c paged native reuse: reused={reused} warm={warm:?} native_cold={native_cold:?} \
         ort_cold={ort_cold:?} prefix_kv_len={} layers={}",
        native_prefix_kv.sequence_len,
        native_prefix_kv.layers.len()
    );
    Ok(())
}

/// GAP-3 Inc-D — paged native pipeline decode with cross-request KV reuse when
/// the native decoder keeps its present KV **device-resident on CUDA**.
///
/// Inc-C paged only the host-growable f32 KV path; a native CUDA decoder (its KV
/// in device bindings) fell back to the Inc-A non-paged flat-AR path. Inc-D lifts
/// that gate for device-resident f32 rank-4 caches: `mirror_last_present_kv`
/// reads the present KV out of the device binding (`DecodeCudaState::read_present_kv`,
/// physical/capacity shape so the row-major strides address the padded buffer)
/// and `load_paged_prefix` seeds a materialized prefix back into the device
/// bindings (`DecodeCudaState::seed_prefix`), landing in the **same** host f32
/// paged store through the same `extract_present_token` / `append_token_kv`
/// geometry the ORT and host-growable paths use.
///
/// Two prefix-sharing requests run through ONE pure-native engine over
/// `tiny-gemma4-vlm-cuda` with the native decoder pinned to `cuda:0`. Together
/// the asserts prove, on the device path:
///   * **reuse engaged** — the warm request reuses `> 0` prefix tokens, proving
///     the device mirror-write populated the pages and the device seed-read
///     consumed them (a gate revert to non-paged reports zero reuse);
///   * **geometry correct (tokens)** — the warm tokens equal a cold pure-native
///     CUDA run, the ORT oracle, and the fixture's closed-form ids;
///   * **geometry correct (bytes)** — the CUDA-mirrored paged KV for the shared
///     prefix is byte-identical to the ORT-mirrored KV. This fixture's argmax is
///     invariant to the reused-prefix KV, so the byte comparison is what catches
///     a device read/seed value error (the physical-vs-logical *stride* error is
///     invisible at this fixture's single KV head and is proven separately by the
///     `H == 2` unit test `device_kv_view_uses_physical_stride`).
///
/// Skips gracefully when no CUDA GPU is visible.
///
/// Non-vacuity: if the device gate reverts to non-paged the `reused > 0` assert
/// fires; if the device mirror/seed geometry is wrong the byte assert fires; if
/// reuse silently no-ops the `reused > 0` assert fires.
#[test]
fn native_paged_prefix_reuse_matches_ort_on_cuda_device() -> anyhow::Result<()> {
    if !cuda_device_visible() {
        eprintln!("gap3 inc-d: no CUDA device visible; skipping device paged reuse test");
        return Ok(());
    }
    // f32 device-resident KV (Inc-D).
    run_device_paged_reuse_parity("tiny-gemma4-vlm-cuda", "inc-d f32")
}

/// GAP-3 Inc-D.1 — the Inc-D device paged-reuse parity, but on a decoder whose KV
/// cache is **FLOAT16** (`tiny-gemma4-vlm-cuda-f16`), the dtype real exports
/// (gemma4-e2b, likely qwen3-30b-a3b) use.
///
/// Inc-D gated `supports_device_kv_mirror` to f32-only, so an f16 device cache
/// fell back to the Inc-A non-paged path. Inc-D.1 lifts the dtype half of that
/// gate: the device read-out widens the f16 present KV to f32 with the same
/// `half` convert ORT uses, and the reuse-seed narrows f32 back to f16 with the
/// same `half` convert ORT injects with, so the mirrored pages are byte-identical
/// to ORT's despite the f16 device buffer. The same three-way token parity +
/// byte-equality asserts prove it; a revert of the Inc-D.1 gate to f32-only sends
/// this fixture to non-paged and fires the `reused > 0` assert, and a wrong f16
/// convert fires the byte-equality assert.
#[test]
fn native_paged_prefix_reuse_matches_ort_on_cuda_device_f16() -> anyhow::Result<()> {
    if !cuda_device_visible() {
        eprintln!("gap3 inc-d.1: no CUDA device visible; skipping f16 device paged reuse test");
        return Ok(());
    }
    // f16 device-resident KV (Inc-D.1).
    run_device_paged_reuse_parity("tiny-gemma4-vlm-cuda-f16", "inc-d.1 f16")
}

/// Shared body for the Inc-D (f32) and Inc-D.1 (f16) device paged-reuse parity:
/// two prefix-sharing requests through ONE pure-native CUDA engine over `fixture`
/// with the native decoder pinned to `cuda:0`. Together the asserts prove, on the
/// device path:
///   * **reuse engaged** — the warm request reuses `> 0` prefix tokens, proving
///     the device mirror-write populated the pages and the device seed-read
///     consumed them (a gate revert to non-paged reports zero reuse);
///   * **geometry correct (tokens)** — the warm tokens equal a cold pure-native
///     CUDA run, the ORT oracle, and the fixture's closed-form ids;
///   * **geometry correct (bytes)** — the CUDA-mirrored paged KV for the shared
///     prefix is byte-identical to the ORT-mirrored KV (both land f32 in the
///     shared host paged store), catching any device read/seed value error — for
///     the f16 fixture, specifically an f16<->f32 convert that diverges from ORT.
///     The physical-vs-logical *stride* error is invisible at this fixture's
///     single KV head and is proven separately by the `H == 2` unit test
///     `device_kv_view_uses_physical_stride`.
fn run_device_paged_reuse_parity(fixture: &str, label: &str) -> anyhow::Result<()> {
    const DEVICE: &str = "cuda:0";
    let shared_prompt = vec![3u32, 7, 0, 5];
    let warm_prompt = vec![3u32, 7, 0, 5, 6];

    // Reuse-independent oracles for the warm prompt: ORT decode of the same
    // fixture, and a cold pure-native CUDA decode (no prior turn to reuse). The
    // cold-prompt closed-form ids ([3, 7] -> [0, 5, 6, 7]) are asserted by the
    // base `pure_native_pipeline_selection_matches_ort_and_hybrid` test; here the
    // warm prompt continues past them, so ORT is the token oracle.
    let ort_cold = cold_tokens(fixture, Selection::Ort, warm_prompt.clone(), 3, None)?;
    let native_cold = cold_tokens(
        fixture,
        Selection::PureNative,
        warm_prompt.clone(),
        3,
        Some(DEVICE),
    )?;
    assert_eq!(
        native_cold, ort_cold,
        "[{label}] cold pure-native CUDA paged decode diverged from the ORT oracle"
    );

    // One pure-native CUDA engine, two turns: the first primes the prefix cache
    // via the device present-KV mirror; the second reuses it via the device
    // seed (load_paged_prefix -> seed_prefix).
    clear_env();
    unsafe { std::env::set_var(NATIVE_DECODER_DEVICE_ENV, DEVICE) };
    let mut config = EngineConfig {
        page_size: 2,
        ..EngineConfig::default()
    };
    config.decode_backend = EngineDecodeBackend::Native;
    let native_probe = prefix_probe_request(shared_prompt.clone())?;
    let outcome = (|| -> anyhow::Result<(Vec<u32>, usize, Option<MaterializedKv>)> {
        let mut engine = Engine::from_pipeline_dir(&fixture_dir(fixture), config)?;
        let _first = run_turn(&mut engine, shared_prompt.clone(), 2)?;
        engine.reset_cache_stats();
        let warm = run_turn(&mut engine, warm_prompt.clone(), 3)?;
        let reused = engine.cache_stats().prefix_reused_tokens as usize;
        let prefix_kv = engine.materialize_published_prefix_kv(&native_probe)?;
        Ok((warm, reused, prefix_kv))
    })();
    clear_env();
    let (warm, reused, native_prefix_kv) = outcome?;

    assert!(
        reused > 0,
        "[{label}] paged native CUDA decode must reuse the shared prefix (reused {reused} tokens); \
         zero reuse means the device present-KV mirror/seed never populated the pages \
         (or the device paged gate reverted — for f16, back to f32-only)"
    );
    assert_eq!(
        warm, native_cold,
        "[{label}] warm native CUDA prefix reuse diverged from a cold CUDA run — the device \
         mirrored/seeded present-KV geometry is wrong"
    );
    assert_eq!(
        warm, ort_cold,
        "[{label}] warm native CUDA prefix reuse diverged from the ORT oracle"
    );

    // Geometry, byte-exact: the device-mirrored paged KV for the shared prefix
    // must equal the ORT-mirrored KV byte-for-byte (both mirror through the same
    // extract_present_token / append_token_kv geometry into the same host f32
    // store), catching any device read/seed value error the argmax-invariant
    // tokens cannot.
    let native_prefix_kv = native_prefix_kv.expect(
        "native CUDA paged decode published no shared-prefix KV to read back — the device \
         present-KV mirror never populated the pages",
    );
    let ort_prefix_kv = published_prefix_kv(fixture, Selection::Ort, shared_prompt.clone(), None)?
        .expect("ORT paged decode published no shared-prefix KV reference");
    assert_eq!(
        native_prefix_kv, ort_prefix_kv,
        "[{label}] device-mirrored paged KV for the shared prefix diverged byte-for-byte from the \
         ORT-mirrored KV — the device present-KV mirror/seed geometry (or, for f16, the \
         f16<->f32 convert) is wrong even though the argmax tokens matched"
    );
    eprintln!(
        "gap3 {label} device paged reuse: reused={reused} warm={warm:?} native_cold={native_cold:?} \
         ort_cold={ort_cold:?} prefix_kv_len={} layers={}",
        native_prefix_kv.sequence_len,
        native_prefix_kv.layers.len()
    );
    Ok(())
}
