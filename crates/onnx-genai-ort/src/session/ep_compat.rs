use std::collections::BTreeSet;
use std::path::PathBuf;

use onnx_genai_runtime_config::{EpSelection, runtime_config};

/// Broad class of hardware an execution provider targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareKind {
    Cpu,
    Gpu,
    Npu,
    Other,
}

/// Capability-flag vocabulary. These `&str` constants are the *only* stable
/// identifiers the runtime core uses to reason about EP behavior, so
/// decode/allocation code never branches on an EP name.
pub mod capability {
    /// EP honors ORT's pre-bound fixed-capacity `present.*` output contract
    /// that the SharedBuffer O(1)/token decode path needs.
    pub const FIXED_CAPACITY_PRESENT_BINDING: &str = "fixed_capacity_present_binding";
    /// EP supports ORT graph capture/replay.
    pub const GRAPH_CAPTURE: &str = "graph_capture";
    /// EP owns device memory usable for device-resident KV.
    pub const DEVICE_KV: &str = "device_kv";
    /// EP exposes device-resident logits + a device allocator for on-device
    /// argmax/sampling.
    pub const DEVICE_SAMPLING: &str = "device_sampling";
}

/// Capabilities the runtime core reasons about, resolved once per EP.
#[derive(Debug, Clone)]
pub struct EpCapabilities {
    pub name: String,
    pub hardware: HardwareKind,
    pub device_id: Option<i32>,
    pub vendor: Option<String>,
    flags: BTreeSet<String>,
}

impl EpCapabilities {
    pub(crate) fn new(
        name: impl Into<String>,
        hardware: HardwareKind,
        device_id: Option<i32>,
        vendor: Option<String>,
        flags: &[&str],
    ) -> Self {
        Self {
            name: name.into(),
            hardware,
            device_id,
            vendor,
            flags: flags.iter().map(|flag| (*flag).to_string()).collect(),
        }
    }

    /// Whether this EP advertises the given capability flag.
    #[must_use]
    pub fn has(&self, flag: &str) -> bool {
        self.flags.contains(flag)
    }

    /// Whether this EP targets a GPU.
    #[must_use]
    pub fn is_gpu(&self) -> bool {
        self.hardware == HardwareKind::Gpu
    }

    /// Whether this EP is the host (CPU) provider.
    #[must_use]
    pub fn is_host(&self) -> bool {
        self.hardware == HardwareKind::Cpu
    }

    /// Device id this EP is bound to, if any.
    #[must_use]
    pub fn device_id(&self) -> Option<i32> {
        self.device_id
    }

    /// Whether this EP reports an NVIDIA vendor (case-insensitive).
    #[must_use]
    pub fn is_nvidia(&self) -> bool {
        self.vendor
            .as_deref()
            .is_some_and(|vendor| vendor.to_ascii_lowercase().contains("nvidia"))
    }

    /// The default host (CPU) capabilities.
    #[must_use]
    pub fn host() -> Self {
        Self::new(
            "cpu",
            HardwareKind::Cpu,
            None,
            None,
            &[capability::FIXED_CAPACITY_PRESENT_BINDING],
        )
    }
}

/// How an EP is appended to ORT session options. Variants carry only opaque
/// data; the append FFI lives in `session.rs`.
#[derive(Debug, Clone)]
pub(crate) enum AppendStrategy {
    /// CPU / no-op (the host provider is implicit in ORT).
    HostDefault,
    /// Permanent built-in CUDA EP appended via the typed CUDA V2 API.
    #[cfg(feature = "cuda")]
    CudaTyped { device_id: i32 },
    /// CUDA requested without the compile-time `cuda` feature. Preserves the
    /// historical hard error raised at append time.
    #[cfg(not(feature = "cuda"))]
    CudaUnavailable,
    /// Self-registering plugin EP: register the library, match an
    /// EP-reported device name, and append via V2 (Metal today).
    PluginLibrary {
        lib: PathBuf,
        registration_name: String,
        options: Vec<(String, String)>,
        device: Option<String>,
    },
    /// ORT built-in appended by name (WebGPU/CoreML transitional, plus any
    /// unrecognized name attempted by-name with conservative capabilities).
    NamedGeneric {
        ort_name: String,
        provider_name: String,
    },
}

/// An [`EpSelection`] resolved into capabilities plus an append strategy.
#[derive(Debug, Clone)]
pub struct ResolvedEp {
    pub selection: EpSelection,
    pub caps: EpCapabilities,
    pub(crate) strategy: AppendStrategy,
    /// Whether this EP's provider-specific graph-capture env flag is enabled.
    pub(crate) graph_capture_env: bool,
    /// TRANSITIONAL: whether WebGPU session-config entries apply to this EP.
    pub(crate) transitional_webgpu: bool,
}

impl ResolvedEp {
    /// A strict provider must NOT silently fall back to CPU on load failure.
    /// Explicit CUDA and self-registering plugin EPs are strict.
    pub(crate) fn is_strict(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            matches!(
                self.strategy,
                AppendStrategy::CudaTyped { .. } | AppendStrategy::PluginLibrary { .. }
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            matches!(
                self.strategy,
                AppendStrategy::CudaUnavailable | AppendStrategy::PluginLibrary { .. }
            )
        }
    }

    /// Native-runtime plugin bridge metadata, when this EP is backed by an
    /// ORT plugin library. This keeps provider-name knowledge inside
    /// `ep_compat` while allowing the native backend to load the same plugin.
    #[must_use]
    pub fn native_plugin_bridge(&self) -> Option<NativePluginBridge> {
        match &self.strategy {
            AppendStrategy::PluginLibrary {
                lib,
                registration_name,
                ..
            } => Some(NativePluginBridge {
                lib: lib.clone(),
                registration_name: registration_name.clone(),
                provider_name: self.caps.name.clone(),
            }),
            _ => None,
        }
    }
}

/// Metadata needed by the native runtime to load a plugin EP through the
/// ORT C ABI without duplicating provider-name logic outside `ep_compat`.
#[derive(Debug, Clone)]
pub struct NativePluginBridge {
    pub lib: PathBuf,
    pub registration_name: String,
    pub provider_name: String,
}

/// Provider names this build can resolve to something real, in the order a
/// user would try them.
///
/// Lives here because this module is the only place that knows EP names — a
/// caller offering a menu of providers would otherwise keep its own copy and
/// let it drift from the table below. Names appear only when they can
/// actually be selected: `cuda` needs the feature compiled in, and `metal`
/// needs its plugin library configured, so a machine with neither is not
/// offered a provider that would fail to load.
///
/// Other names still work — the table falls back to appending them to ONNX
/// Runtime by name — so this is a menu, not a whitelist.
#[must_use]
pub fn selectable_execution_providers() -> Vec<&'static str> {
    let mut names = vec!["cpu"];
    if cfg!(feature = "cuda") {
        names.push("cuda");
    }
    if cfg!(target_os = "macos")
        && runtime_config()
            .metal_ep_lib
            .as_ref()
            .is_some_and(|library| library.is_file())
    {
        names.push("metal");
    }
    names
}

/// Resolve an [`EpSelection`] into capabilities and an append strategy.
///
/// This is the single compatibility table mapping EP *names* to behavior.
#[must_use]
pub fn resolve_execution_provider(selection: &EpSelection) -> ResolvedEp {
    use capability::{DEVICE_KV, DEVICE_SAMPLING, FIXED_CAPACITY_PRESENT_BINDING, GRAPH_CAPTURE};

    if selection.is_host_default() {
        return ResolvedEp {
            selection: selection.clone(),
            caps: EpCapabilities::host(),
            strategy: AppendStrategy::HostDefault,
            graph_capture_env: false,
            transitional_webgpu: false,
        };
    }

    match selection.name.as_str() {
        // Permanent built-in.
        "cuda" => {
            let device_id = super::cuda_device_id_from_env();
            let caps = EpCapabilities::new(
                "cuda",
                HardwareKind::Gpu,
                Some(device_id),
                Some("NVIDIA".to_string()),
                &[
                    FIXED_CAPACITY_PRESENT_BINDING,
                    GRAPH_CAPTURE,
                    DEVICE_KV,
                    DEVICE_SAMPLING,
                ],
            );
            #[cfg(feature = "cuda")]
            let strategy = AppendStrategy::CudaTyped { device_id };
            #[cfg(not(feature = "cuda"))]
            let strategy = AppendStrategy::CudaUnavailable;
            ResolvedEp {
                selection: selection.clone(),
                caps,
                strategy,
                graph_capture_env: runtime_config().cuda_graph,
                transitional_webgpu: false,
            }
        }
        // TRANSITIONAL: WebGPU is an ORT built-in appended by name today; it
        // will become a self-registering plugin EP. Separator aliases are
        // accepted here (the single EP-name table) rather than in the generic
        // config parser.
        "webgpu" | "web-gpu" | "web_gpu" => ResolvedEp {
            selection: selection.clone(),
            caps: EpCapabilities::new(
                "webgpu",
                HardwareKind::Gpu,
                None,
                None,
                &[FIXED_CAPACITY_PRESENT_BINDING, GRAPH_CAPTURE, DEVICE_KV],
            ),
            strategy: AppendStrategy::NamedGeneric {
                ort_name: "WebGPU".to_string(),
                provider_name: "WebGpuExecutionProvider".to_string(),
            },
            graph_capture_env: runtime_config().webgpu_graph_capture,
            transitional_webgpu: true,
        },
        // TRANSITIONAL: CoreML is an ORT built-in appended by name today.
        "coreml" | "core-ml" | "core_ml" => ResolvedEp {
            selection: selection.clone(),
            caps: EpCapabilities::new("coreml", HardwareKind::Npu, None, None, &[]),
            strategy: AppendStrategy::NamedGeneric {
                ort_name: "CoreML".to_string(),
                provider_name: "CoreMLExecutionProvider".to_string(),
            },
            graph_capture_env: false,
            transitional_webgpu: false,
        },
        // TRANSITIONAL: Metal is loaded from the external onnxruntime-mlx
        // plugin library and appended via the V2 plugin path; it is the only
        // strict provider today. The MLX plugin implements the fixed-capacity
        // in-place-write GQA contract, so Metal carries
        // FIXED_CAPACITY_PRESENT_BINDING (preserving today's SharedBuffer
        // decode path) but no other device capabilities by default.
        "metal" => ResolvedEp {
            selection: selection.clone(),
            caps: EpCapabilities::new(
                "metal",
                HardwareKind::Gpu,
                None,
                None,
                &[FIXED_CAPACITY_PRESENT_BINDING],
            ),
            strategy: AppendStrategy::PluginLibrary {
                lib: runtime_config().metal_ep_lib.clone().unwrap_or_default(),
                registration_name: "onnxruntime_mlx_ep".to_string(),
                options: selection
                    .options
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
                device: None,
            },
            graph_capture_env: false,
            transitional_webgpu: false,
        },
        // Any other name: no plugin library env is configured, so attempt an
        // ORT built-in append by name with conservative capabilities.
        other => {
            tracing::warn!(
                "Unrecognized ONNX_GENAI_EP={other}; attempting to append it to ONNX Runtime by name with conservative capabilities (no device-resident KV/sampling, no graph capture, no fixed-capacity present binding)"
            );
            ResolvedEp {
                selection: selection.clone(),
                caps: EpCapabilities::new(
                    selection.name.clone(),
                    HardwareKind::Other,
                    None,
                    None,
                    &[],
                ),
                strategy: AppendStrategy::NamedGeneric {
                    ort_name: selection.name.clone(),
                    provider_name: format!("{other}ExecutionProvider"),
                },
                graph_capture_env: false,
                transitional_webgpu: false,
            }
        }
    }
}
