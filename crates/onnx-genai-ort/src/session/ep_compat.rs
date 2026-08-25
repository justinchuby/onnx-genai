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
    /// EP permits several threads to call `Run`/`RunWithBinding` on one session
    /// at the same time.
    ///
    /// ORT's session-level contract allows concurrent `Run`, but that only
    /// holds when every provider underneath it does too — a provider that keeps
    /// one mutable per-session context (a QNN HTP context, a plugin EP with a
    /// single command queue) turns a second concurrent run into corruption
    /// rather than contention. So this is declared per EP, and
    /// [`crate::Session::supports_concurrent_run`] is the conjunction over the
    /// providers a session actually resolved to. Absence means "not known to be
    /// safe", which is the answer that fails closed.
    pub const CONCURRENT_RUN: &str = "concurrent_run";
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
            &[
                capability::FIXED_CAPACITY_PRESENT_BINDING,
                capability::CONCURRENT_RUN,
            ],
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
    /// ORT built-in appended by name (WebGPU/CoreML transitional).
    NamedGeneric {
        ort_name: String,
        provider_name: String,
    },
    /// A requested provider name that this compatibility table does not know.
    UnsupportedName { name: String },
    /// A provider entry that could not be resolved because its configuration is
    /// incomplete or contradictory.
    InvalidConfiguration { message: String },
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
    #[cfg(test)]
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

    pub(crate) fn is_unsupported_name(&self) -> bool {
        matches!(
            self.strategy,
            AppendStrategy::UnsupportedName { .. } | AppendStrategy::InvalidConfiguration { .. }
        )
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
/// This table is intentionally a whitelist. Unsupported provider names fail
/// clearly; provider-specific extensions should use the explicit plugin path.
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
    if runtime_config()
        .qnn_ep_lib
        .as_ref()
        .is_some_and(|library| library.is_file())
    {
        names.push("qnn");
    }
    names
}

#[must_use]
pub(crate) fn known_execution_provider_values() -> &'static str {
    "cpu, cuda, webgpu (aliases: web-gpu, web_gpu), coreml (aliases: core-ml, core_ml), metal, qnn (aliases: qnn-htp, qnn_htp), plugin:<library>|name=<registration>|device=<CPU|GPU|NPU>|opt.<key>=<value>"
}

/// Resolve an [`EpSelection`] into capabilities and an append strategy.
///
/// This is the single compatibility table mapping EP *names* to behavior.
#[must_use]
pub fn resolve_execution_provider(selection: &EpSelection) -> ResolvedEp {
    use capability::{
        CONCURRENT_RUN, DEVICE_KV, DEVICE_SAMPLING, FIXED_CAPACITY_PRESENT_BINDING, GRAPH_CAPTURE,
    };

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
            let device_id = selection
                .options
                .get("device_id")
                .and_then(|value| value.parse::<i32>().ok())
                .unwrap_or_else(super::cuda_device_id_from_env);
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
                    CONCURRENT_RUN,
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
        // Qualcomm QNN ships as a self-registering ORT plugin EP. Keep the
        // runtime capabilities conservative until HTP fixed-shape KV and
        // device-resident buffers are proven on hardware.
        "qnn" | "qnn-htp" | "qnn_htp" => ResolvedEp {
            selection: selection.clone(),
            caps: EpCapabilities::new(
                "qnn",
                HardwareKind::Npu,
                None,
                Some("Qualcomm".to_string()),
                &[],
            ),
            strategy: AppendStrategy::PluginLibrary {
                lib: qnn_plugin_library_path(),
                registration_name: "onnxruntime_qnn_ep".to_string(),
                options: qnn_provider_options(selection),
                device: Some(
                    runtime_config()
                        .qnn_device
                        .clone()
                        .unwrap_or_else(|| "NPU".to_string()),
                ),
            },
            graph_capture_env: false,
            transitional_webgpu: false,
        },
        other => ResolvedEp {
            selection: selection.clone(),
            caps: EpCapabilities::new(selection.name.clone(), HardwareKind::Other, None, None, &[]),
            strategy: AppendStrategy::UnsupportedName {
                name: other.to_string(),
            },
            graph_capture_env: false,
            transitional_webgpu: false,
        },
    }
}

fn qnn_plugin_library_path() -> PathBuf {
    runtime_config()
        .qnn_ep_lib
        .clone()
        .unwrap_or_else(|| PathBuf::from(qnn_plugin_library_name()))
}

fn qnn_plugin_library_name() -> &'static str {
    if cfg!(windows) {
        "onnxruntime_providers_qnn.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime_providers_qnn.dylib"
    } else {
        "libonnxruntime_providers_qnn.so"
    }
}

fn qnn_htp_backend_library_name() -> &'static str {
    if cfg!(windows) {
        "QnnHtp.dll"
    } else if cfg!(target_os = "macos") {
        "libQnnHtp.dylib"
    } else {
        "libQnnHtp.so"
    }
}

fn qnn_default_backend_path() -> PathBuf {
    if let Some(path) = runtime_config().qnn_backend_path.clone() {
        return path;
    }
    if let Some(parent) = runtime_config()
        .qnn_ep_lib
        .as_ref()
        .and_then(|path| path.parent())
    {
        return parent.join(qnn_htp_backend_library_name());
    }
    PathBuf::from(qnn_htp_backend_library_name())
}

fn qnn_provider_options(selection: &EpSelection) -> Vec<(String, String)> {
    let config = runtime_config();
    let mut options = selection
        .options
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<Vec<_>>();

    let has_backend_path = provider_option_exists(&options, "backend_path");
    let has_backend_type = provider_option_exists(&options, "backend_type");
    if !has_backend_path && !has_backend_type {
        if let Some(backend_type) = &config.qnn_backend_type {
            options.push(("backend_type".to_string(), backend_type.clone()));
        } else {
            options.push((
                "backend_path".to_string(),
                qnn_default_backend_path().display().to_string(),
            ));
        }
    }
    push_qnn_option_if_absent(
        &mut options,
        "htp_performance_mode",
        &config.qnn_performance_mode,
    );
    push_qnn_option_if_absent(&mut options, "vtcm_mb", &config.qnn_vtcm_mb);
    push_qnn_option_if_absent(&mut options, "htp_arch", &config.qnn_htp_arch);
    push_qnn_option_if_absent(&mut options, "soc_model", &config.qnn_soc_model);
    push_qnn_option_if_absent(&mut options, "device_id", &config.qnn_device_id);
    options
}

fn push_qnn_option_if_absent(
    options: &mut Vec<(String, String)>,
    key: &str,
    value: &Option<String>,
) {
    if let Some(value) = value
        && !provider_option_exists(options, key)
    {
        options.push((key.to_string(), value.clone()));
    }
}

fn provider_option_exists(options: &[(String, String)], key: &str) -> bool {
    options
        .iter()
        .any(|(existing, _)| existing.eq_ignore_ascii_case(key))
}
