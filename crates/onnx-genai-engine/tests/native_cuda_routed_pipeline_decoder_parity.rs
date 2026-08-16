//! Native multi-component pipeline — increment 3b (issue #384).
//!
//! Proves the **native CUDA decoder binds a generic `Routed` port on-device**:
//! the Gemma4-style VLM composite pipeline gains a second cross-component edge —
//! the every_step `embedding` component emits a `router_state` output routed to a
//! `router_state` input on the decoder. That port has no generated role and is
//! not `inputs_embeds`; it is a `NativeStepInputSource::Routed` input, exactly
//! the class the CUDA decoder refused before Inc3b.
//!
//! Inc3a lifted the CUDA refusal only for `inputs_embeds`; Inc3b generalizes the
//! eager owned-input build so **any** routed port is uploaded per step and bound
//! on-device while the attention mask and KV cache stay device-resident. This
//! test drives the native decoder on CPU vs the CUDA EP (device 4 via
//! `CUDA_VISIBLE_DEVICES`) through the pipeline and asserts identical tokens,
//! proving the routed port reaches the GPU correctly with the KV kept resident.
//!
//! Fixture: `scripts/build_tiny_gemma4_vlm_cuda_routed.py` — closed-form tokens
//! `[0, 5, 6, 7]` (the routed `router_state` is consumed through a real `MatMul`
//! by a zero matrix, so it flows through a CUDA op but contributes nothing).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use onnx_genai_engine::pipeline::{PipelineEngine, PipelineGenerateRequest};
use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateOptions, GeneratePrompt, GenerateRequest,
    NativeDecodeDevice,
};
use onnx_genai_ort::Value;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-gemma4-vlm-cuda-routed")
}

fn tiny_pixels() -> anyhow::Result<Value> {
    Value::from_vec_f32((0..12).map(|i| i as f32 / 12.0).collect(), &[1, 3, 2, 2])
        .map_err(Into::into)
}

fn cuda_device_index() -> Option<u32> {
    std::env::var_os("CUDA_VISIBLE_DEVICES")?;
    Some(0)
}

fn generate_tokens(device: &str) -> anyhow::Result<Vec<u32>> {
    let native_device = if device == "cpu" {
        NativeDecodeDevice::Cpu
    } else {
        NativeDecodeDevice::Cuda {
            index: device
                .strip_prefix("cuda:")
                .and_then(|value| value.parse().ok()),
        }
    };
    let mut engine = Engine::from_pipeline_dir(
        &fixture_dir(),
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(native_device),
            ..EngineConfig::default()
        },
    )?;
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![3, 7]));
    request.options = GenerateOptions {
        max_new_tokens: 4,
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

#[test]
fn native_cuda_routed_pipeline_decoder_matches_cpu_token_ids() -> anyhow::Result<()> {
    let Some(index) = cuda_device_index() else {
        eprintln!("skipping: no CUDA GPU visible (set CUDA_VISIBLE_DEVICES)");
        return Ok(());
    };
    let graph = onnx_runtime_loader::load_model(fixture_dir().join("decoder.onnx.textproto"))?;
    assert!(
        graph
            .nodes
            .values()
            .any(|node| { node.domain == "pkg.nxrt" && node.op_type == "BlockQuantizedMoE" }),
        "routed pipeline regression must contain a workspace-bearing QMoE"
    );

    // Native decoder on CPU binding the routed port host-side — the reference.
    let cpu_tokens = generate_tokens("cpu")?;
    assert_eq!(
        cpu_tokens,
        vec![0, 5, 6, 7],
        "native CPU routed-port baseline drifted"
    );

    // Native decoder on the CUDA EP: the routed `router_state` is uploaded per
    // step and bound on-device; the KV cache stays device-resident on the GPU.
    let cuda_tokens = generate_tokens(&format!("cuda:{index}"))?;
    assert_eq!(
        cuda_tokens, cpu_tokens,
        "native CUDA decoder with a generic routed port diverged from the native CPU baseline"
    );
    Ok(())
}

#[test]
fn native_cuda_routed_qmoe_pipeline_admits_after_workspace() -> anyhow::Result<()> {
    let Some(index) = cuda_device_index() else {
        eprintln!("skipping: no CUDA GPU visible (set CUDA_VISIBLE_DEVICES)");
        return Ok(());
    };
    (|| {
        let mut engine = PipelineEngine::from_dir_with_config(
            &fixture_dir(),
            EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                native_device: Some(NativeDecodeDevice::Cuda { index: Some(index) }),
                allow_runtime_override: true,
                ..EngineConfig::default()
            },
        )?;
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![3, 7]));
        request.options = GenerateOptions {
            max_new_tokens: 2,
            temperature: 0.0,
            stop_on_eos: false,
            ..GenerateOptions::default()
        };
        let pipeline_request = PipelineGenerateRequest::new(request)
            .with_input("vision_encoder.pixel_values", tiny_pixels()?);
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let admitted_events = Arc::clone(&events);
        let mut admitted = move || admitted_events.lock().unwrap().push("admission");
        let token_events = Arc::clone(&events);
        let mut token = move |_| {
            token_events.lock().unwrap().push("token");
            Ok(())
        };
        let generated = engine.generate_with_callbacks(
            pipeline_request,
            Some(&mut admitted),
            Some(&mut token),
        )?;
        assert_eq!(generated.token_ids, vec![0, 5]);
        let events = events.lock().unwrap();
        assert_eq!(events.first(), Some(&"admission"));
        assert!(events[1..].contains(&"token"), "{events:?}");
        Ok::<_, anyhow::Error>(())
    })()
}

#[test]
fn native_cuda_routed_qmoe_exhaustion_precedes_admission_and_recovers() -> anyhow::Result<()> {
    let Some(index) = cuda_device_index() else {
        eprintln!("skipping: no CUDA GPU visible (set CUDA_VISIBLE_DEVICES)");
        return Ok(());
    };
    (|| {
        let mut engine = PipelineEngine::from_dir_with_config(
            &fixture_dir(),
            EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                native_device: Some(NativeDecodeDevice::Cuda { index: Some(index) }),
                allow_runtime_override: true,
                ..EngineConfig::default()
            },
        )?;
        let baseline = engine.resource_snapshot().vram.used;
        engine.set_vram_limit(onnx_genai_engine::ResourceLimit::Bytes(baseline))?;
        assert_eq!(engine.resource_snapshot().vram.limit, baseline);

        let request = || {
            let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![3, 7]));
            request.options = GenerateOptions {
                max_new_tokens: 1,
                temperature: 0.0,
                stop_on_eos: false,
                ..GenerateOptions::default()
            };
            PipelineGenerateRequest::new(request)
                .with_input("vision_encoder.pixel_values", tiny_pixels().unwrap())
        };
        let admitted = Arc::new(AtomicBool::new(false));
        let admitted_flag = Arc::clone(&admitted);
        let mut on_admitted = move || admitted_flag.store(true, Ordering::Relaxed);
        let token = Arc::new(AtomicBool::new(false));
        let token_flag = Arc::clone(&token);
        let mut on_token = move |_| {
            token_flag.store(true, Ordering::Relaxed);
            Ok(())
        };
        let error = engine
            .generate_with_callbacks(request(), Some(&mut on_admitted), Some(&mut on_token))
            .expect_err("workspace exhaustion must reject the routed QMoE pipeline");
        assert!(
            error.chain().any(|cause| {
                matches!(
                    cause.downcast_ref::<onnx_runtime_memory_governor::MemoryError>(),
                    Some(onnx_runtime_memory_governor::MemoryError::TierExhausted {
                        role: onnx_runtime_memory_governor::MemoryRole::Workspace {
                            step_scoped: false
                        },
                        ..
                    })
                )
            }),
            "{error:#}"
        );
        assert!(!admitted.load(Ordering::Relaxed));
        assert!(!token.load(Ordering::Relaxed));
        assert_eq!(engine.resource_snapshot().vram.used, baseline);

        engine.set_vram_limit(onnx_genai_engine::ResourceLimit::Bytes(u64::MAX))?;
        let admitted = Arc::new(AtomicBool::new(false));
        let admitted_flag = Arc::clone(&admitted);
        let mut on_admitted = move || admitted_flag.store(true, Ordering::Relaxed);
        let generated = engine.generate_with_callbacks(request(), Some(&mut on_admitted), None)?;
        assert_eq!(generated.token_ids, vec![0]);
        assert!(admitted.load(Ordering::Relaxed));
        assert_eq!(engine.resource_snapshot().vram.used, baseline);
        Ok::<_, anyhow::Error>(())
    })()
}
