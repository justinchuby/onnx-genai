#![cfg(feature = "native-cuda")]

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest, NativeDecodeDevice,
    ResourceLimit,
};
use onnx_runtime_ep_api::RouteResidencyInstallState;

struct EnvGuard {
    name: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvGuard {
    fn set(name: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        let previous = std::env::var_os(name);
        // SAFETY: this integration test contains one test and therefore has no
        // same-process peer reading these benchmark-only environment controls.
        unsafe { std::env::set_var(name, value) };
        Self { name, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: see `set`; restoration happens before this one-test process exits.
        unsafe {
            if let Some(previous) = self.previous.take() {
                std::env::set_var(self.name, previous);
            } else {
                std::env::remove_var(self.name);
            }
        }
    }
}

fn fixture_dir() -> Option<PathBuf> {
    let Some(dir) = std::env::var_os("FREETOKEN_TINY_QMOE_NATIVE_CUDA_DIR").map(PathBuf::from)
    else {
        eprintln!(
            "skipping production FreeToken lifecycle proof: set \
             FREETOKEN_TINY_QMOE_NATIVE_CUDA_DIR to an external-data, VMM-granule-padded QMoE \
             native model directory"
        );
        return None;
    };
    let missing = ["model.onnx", "inference_metadata.yaml", "tokenizer.json"]
        .into_iter()
        .filter(|name| !dir.join(name).is_file())
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Some(dir)
    } else {
        eprintln!(
            "skipping production FreeToken lifecycle proof: {} is missing {}",
            dir.display(),
            missing.join(", ")
        );
        None
    }
}

fn engine(dir: &Path) -> anyhow::Result<Engine> {
    let host_budget = std::env::var("FREETOKEN_TINY_QMOE_HOST_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1024_u64 * 1024 * 1024);
    Engine::from_dir(
        dir,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cuda { index: Some(0) }),
            limits: onnx_genai_engine::ResourceLimits {
                host_ram_limit: ResourceLimit::Bytes(host_budget),
                ..onnx_genai_engine::ResourceLimits::default()
            },
            ..EngineConfig::default()
        },
    )
}

fn generate(engine: &mut Engine, tokens: usize) -> anyhow::Result<Vec<u32>> {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![3]));
    request.options.max_new_tokens = tokens;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    Ok(engine.generate(request)?.token_ids)
}

#[test]
fn tiny_qmoe_production_generation_proves_gate_miss_page_in_then_hit() -> anyhow::Result<()> {
    let Some(dir) = fixture_dir() else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping production FreeToken lifecycle proof: CUDA unavailable: {error}");
        return Ok(());
    }

    let budget = std::env::var("FREETOKEN_TINY_QMOE_DEVICE_BYTES")
        .unwrap_or_else(|_| (1024_u64 * 1024 * 1024).to_string());
    let _offload = EnvGuard::set("ONNX_GENAI_WEIGHT_OFFLOAD", "1");
    let _accounting = EnvGuard::set("ONNX_GENAI_FREETOKEN_EXPERT_ACCOUNTING", "1");
    let _budget = EnvGuard::set("ONNX_GENAI_WEIGHT_OFFLOAD_DEVICE_BYTES", budget);
    let _graph = EnvGuard::set("ONNX_GENAI_CUDA_GRAPH", "1");

    let off_tokens = {
        let _gate = EnvGuard::set("ONNX_GENAI_WEIGHT_OFFLOAD_COARSE_RESIDENCY_ENABLE", "0");
        let mut off = engine(&dir)?;
        let _ = generate(&mut off, 4)?;
        off.reset_expert_residency_measurement(false)?;
        let tokens = generate(&mut off, 8)?;
        let metrics = off
            .expert_residency_metrics()
            .expect("native CUDA exposes expert residency metrics");
        eprintln!(
            "FreeToken OFF production metrics: {metrics:?}; offload={:?}",
            onnx_runtime_ep_cuda::global_offload_stats()
        );
        assert_eq!(
            metrics.install_state,
            RouteResidencyInstallState::GateDisabled
        );
        assert_eq!(metrics.boundaries, 0);
        assert_eq!(metrics.successful_applications, 0);
        assert!(metrics.selected_bytes > 0);
        assert_eq!(metrics.selected_bytes, metrics.gpu_hit_bytes);
        assert_eq!(metrics.h2d_bytes, 0);
        assert_eq!(metrics.cpu_served_bytes, 0);
        assert!(
            metrics.device_committed_bytes + metrics.host_committed_bytes > 0,
            "OFF must still report authoritative expert physical memory"
        );
        tokens
    };

    let on_tokens = {
        let _gate = EnvGuard::set("ONNX_GENAI_WEIGHT_OFFLOAD_COARSE_RESIDENCY_ENABLE", "1");
        let mut on = engine(&dir)?;
        let _ = generate(&mut on, 4)?;
        on.reset_expert_residency_measurement(true)?;
        let tokens = generate(&mut on, 8)?;
        let metrics = on
            .expert_residency_metrics()
            .expect("native CUDA exposes expert residency metrics");
        eprintln!("FreeToken ON production metrics: {metrics:?}");
        assert!(matches!(
            metrics.install_state,
            RouteResidencyInstallState::Installed { .. }
        ));
        assert!(metrics.boundaries > 0);
        assert_eq!(metrics.boundaries, metrics.applied_boundaries);
        assert_eq!(metrics.boundaries, metrics.successful_applications);
        assert!(metrics.cpu_served_bytes > 0, "forced-cold miss was vacuous");
        assert_eq!(metrics.cpu_served_bytes, metrics.h2d_bytes);
        assert!(metrics.page_ins > 0, "forced-cold page-in was vacuous");
        assert!(metrics.gpu_hit_bytes > 0, "post-page-in hit was vacuous");
        assert_eq!(
            metrics.selected_bytes,
            metrics.gpu_hit_bytes + metrics.cpu_served_bytes
        );
        assert_eq!(metrics.ref_underflows, 0);
        assert_eq!(metrics.byte_underflows, 0);
        assert_eq!(metrics.oversubscribed_bytes, 0);
        assert_eq!(metrics.unaccounted_bytes, 0);
        let graph = on
            .native_cuda_debug_stats()
            .expect("native CUDA exposes graph diagnostics")
            .graph;
        assert!(
            graph.captures > 0,
            "production CUDA graph capture was vacuous"
        );
        assert_eq!(graph.fallbacks, 0);
        tokens
    };

    assert_eq!(
        off_tokens, on_tokens,
        "OFF/ON production generation token IDs must be byte-identical"
    );
    Ok(())
}
