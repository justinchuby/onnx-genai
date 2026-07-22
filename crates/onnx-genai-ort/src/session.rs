//! ORT Session — represents a loaded model.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::NonNull;

use onnx_genai_runtime_config::{
    CudaDevice, EpSelection, ExecutionProviderEntry, IntraOpThreads, PluginSpec, runtime_config,
};

use crate::{Allocator, DataType, Environment, IoBinding, MemoryInfo, OrtError, Result, Value};

pub use ep_compat::{
    EpCapabilities, HardwareKind, ResolvedEp, capability, resolve_execution_provider,
};

/// Convenience constructor for an [`EpSelection`] from a bare provider name.
///
/// The runtime core stays EP-agnostic: name resolution happens in
/// [`ep_compat`]. This helper only saves callers from importing [`BTreeMap`].
#[must_use]
pub fn ep_selection(name: impl Into<String>) -> EpSelection {
    EpSelection::new(name.into())
}

/// The ONLY place in the runtime that knows execution-provider *names*.
///
/// `cpu` and `cuda` are the permanent built-in providers. `webgpu`, `coreml`,
/// and `metal` are TRANSITIONAL built-ins: each is expected to become a
/// self-registering plugin EP, at which point its arm here is deleted and it is
/// resolved purely from EP-reported metadata. Everything outside this module
/// makes decode/allocation decisions from [`EpCapabilities`], never from names.
pub mod ep_compat {
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
    }

    /// Resolve an [`EpSelection`] into capabilities and an append strategy.
    ///
    /// This is the single compatibility table mapping EP *names* to behavior.
    #[must_use]
    pub fn resolve_execution_provider(selection: &EpSelection) -> ResolvedEp {
        use capability::{
            DEVICE_KV, DEVICE_SAMPLING, FIXED_CAPACITY_PRESENT_BINDING, GRAPH_CAPTURE,
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
}

/// CUDA attention implementation policy.
///
/// ONNX Runtime's default selects among optimized attention implementations.
/// [`Self::Unfused`] pins the CUDA provider's `sdpa_kernel` option to its
/// standard math implementation, disabling Flash, Lean, fused,
/// memory-efficient, TensorRT-flash, and cuDNN-flash attention dispatch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CudaAttentionMode {
    /// Let ONNX Runtime select the attention implementation.
    #[default]
    Default,
    /// Use ONNX Runtime's standard unfused math attention implementation.
    Unfused,
}

/// Session configuration options.
#[derive(Debug, Clone)]
pub struct SessionOptions {
    /// Execution providers in priority order, resolved to capabilities.
    pub execution_providers: Vec<ResolvedEp>,
    /// Graph optimization level (0=none, 1=basic, 2=extended, 99=all).
    pub optimization_level: i32,
    /// Number of intra-op threads.
    pub intra_op_num_threads: i32,
    /// Number of inter-op threads.
    pub inter_op_num_threads: i32,
    /// Enable execution-provider graph capture. Applied as WebGPU
    /// `enableGraphCapture=1` or CUDA `enable_cuda_graph=1`. Graph capture
    /// requires stable input/output buffer addresses and shapes across runs,
    /// which the device-resident persistent KV IoBinding provides for KV tensors.
    pub graph_capture: bool,
    /// Disable WebGPU/Dawn validation (`validationMode=disabled`). Only applied
    /// when a WebGPU execution provider is selected. Validation is a
    /// debug-oriented overhead layer; disabling it is safe for trusted graphs.
    pub webgpu_disable_validation: bool,
    /// CUDA attention implementation policy.
    ///
    /// Defaults from `ONNX_GENAI_CUDA_ATTENTION` (`default` or `unfused`).
    /// `unfused` is a generic correctness workaround for graphs that encounter
    /// an ONNX Runtime optimized-attention kernel defect; it is never selected
    /// from model identity.
    pub cuda_attention_mode: CudaAttentionMode,
    /// Whether the non-CPU execution provider was auto-selected for this platform
    /// (e.g. the macOS MLX/Metal default) rather than explicitly requested. An
    /// auto-selected provider must fall back to CPU on load failure, even if the
    /// provider would otherwise be strict.
    pub auto_selected: bool,
}

impl Default for SessionOptions {
    fn default() -> Self {
        let mut options = Self::cpu();
        if let Some(execution_providers) = execution_providers_from_env() {
            options.execution_providers = execution_providers;
        } else if let Some(execution_providers) = auto_default_execution_providers() {
            options.execution_providers = execution_providers;
            options.auto_selected = true;
        }
        options.apply_provider_defaults();
        options
    }
}

/// Execution providers to use by default when the user did not set
/// `ONNX_GENAI_EP`.
///
/// On macOS, when the MLX/Metal execution-provider plugin library is available
/// (its path is exposed through `ONNX_GENAI_METAL_EP_LIB` /
/// `ONNX_GENAI_MLX_EP_LIBRARY`, which the Python packages set automatically),
/// prefer it over plain CPU for speed on Apple Silicon. The selection is
/// non-strict: if the plugin fails to load, session creation falls back to CPU.
/// On every other platform, or when no MLX library is configured, this returns
/// `None` (keep the CPU default).
fn auto_default_execution_providers() -> Option<Vec<ResolvedEp>> {
    #[cfg(target_os = "macos")]
    {
        let library = runtime_config().metal_ep_lib.clone()?;
        if library.as_os_str().is_empty() || !library.is_file() {
            return None;
        }
        tracing::info!(
            "Auto-selecting the MLX/Metal execution provider (macOS default) from {}",
            library.display()
        );
        return Some(vec![resolve_execution_provider(&ep_selection("metal"))]);
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

impl SessionOptions {
    fn cpu() -> Self {
        Self {
            execution_providers: vec![resolve_execution_provider(&ep_selection("cpu"))],
            optimization_level: 99,
            intra_op_num_threads: 0, // ORT decides
            inter_op_num_threads: 0,
            graph_capture: false,
            webgpu_disable_validation: false,
            cuda_attention_mode: cuda_attention_mode_from_env(),
            auto_selected: false,
        }
    }

    /// Create default session options with a single explicit execution provider.
    pub fn with_execution_provider(selection: EpSelection) -> Self {
        let mut options = Self {
            execution_providers: vec![resolve_execution_provider(&selection)],
            ..Self::cpu()
        };
        options.apply_provider_defaults();
        options
    }

    /// Capabilities of the first non-host EP, else the host provider.
    fn primary_caps(&self) -> EpCapabilities {
        self.execution_providers
            .iter()
            .find(|ep| !ep.caps.is_host())
            .map(|ep| ep.caps.clone())
            .unwrap_or_else(EpCapabilities::host)
    }

    /// Whether the primary EP's provider-specific graph-capture env flag is set.
    fn primary_graph_capture_env(&self) -> bool {
        self.execution_providers
            .iter()
            .find(|ep| !ep.caps.is_host())
            .is_some_and(|ep| ep.graph_capture_env)
    }

    /// TRANSITIONAL: whether a WebGPU EP is selected (drives WebGPU-specific
    /// session-config entries). Kept here as documented transitional glue until
    /// WebGPU ships as a self-registering plugin EP.
    fn selects_webgpu(&self) -> bool {
        self.execution_providers
            .iter()
            .any(|ep| ep.transitional_webgpu)
    }

    /// Whether a CUDA execution provider is selected in these options.
    pub fn selects_cuda(&self) -> bool {
        self.execution_providers
            .iter()
            .any(|ep| ep.caps.is_nvidia() && ep.caps.is_gpu())
    }

    /// Apply provider performance defaults. WebGPU validation is disabled (pure
    /// overhead reduction), while graph capture follows the primary EP's
    /// capability plus its provider-specific environment flag and remains off by
    /// default.
    fn apply_provider_defaults(&mut self) {
        if self.selects_webgpu() {
            self.webgpu_disable_validation = webgpu_disable_validation_from_env();
        }
        self.graph_capture =
            self.primary_caps().has(capability::GRAPH_CAPTURE) && self.primary_graph_capture_env();
    }

    /// Set the number of ORT intra-op threads.
    ///
    /// Values less than or equal to zero leave thread selection to ORT.
    pub fn with_intra_op_threads(mut self, threads: i32) -> Self {
        self.intra_op_num_threads = threads;
        self
    }

    /// Select the CUDA attention implementation policy.
    ///
    /// Use [`CudaAttentionMode::Unfused`] when an optimized ONNX Runtime CUDA
    /// attention implementation rejects an otherwise valid graph. This maps to
    /// the real CUDA provider option `sdpa_kernel=16` rather than mutating the
    /// process-wide `ORT_DISABLE_*ATTENTION` environment variables.
    pub fn with_cuda_attention_mode(mut self, mode: CudaAttentionMode) -> Self {
        self.cuda_attention_mode = mode;
        self
    }
}

/// Return the execution providers reported by the linked ONNX Runtime build.
pub fn available_execution_providers() -> Result<Vec<String>> {
    let api = crate::error::api()?;
    let get_available = api
        .GetAvailableProviders
        .ok_or(OrtError::ApiUnavailable("GetAvailableProviders"))?;
    let release_available = api
        .ReleaseAvailableProviders
        .ok_or(OrtError::ApiUnavailable("ReleaseAvailableProviders"))?;
    let mut providers_ptr = std::ptr::null_mut();
    let mut provider_count = 0;

    // SAFETY: `providers_ptr` and `provider_count` are valid out-parameters.
    crate::error::check_status(unsafe { get_available(&mut providers_ptr, &mut provider_count) })?;
    if providers_ptr.is_null() {
        return Ok(Vec::new());
    }

    let providers = {
        let mut providers = Vec::with_capacity(provider_count as usize);
        for index in 0..provider_count as isize {
            // SAFETY: ORT returned an array with `provider_count` C string entries.
            let ptr = unsafe { *providers_ptr.offset(index) };
            if !ptr.is_null() {
                // SAFETY: ORT provider names are NUL-terminated strings.
                providers.push(
                    unsafe { CStr::from_ptr(ptr) }
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
        Ok(providers)
    };

    // SAFETY: releases the array returned by `GetAvailableProviders` exactly once.
    crate::error::check_status(unsafe { release_available(providers_ptr, provider_count) })?;
    providers
}

/// Tensor metadata for a model input or output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DataType,
    /// ORT uses negative dimensions for dynamic axes.
    pub shape: Vec<i64>,
}

/// A run failure tagged with whether the model was actually invoked.
#[derive(Debug)]
pub enum RunPhaseError {
    Setup(OrtError),
    Invoked(OrtError),
}

impl RunPhaseError {
    pub fn into_inner(self) -> OrtError {
        match self {
            Self::Setup(err) | Self::Invoked(err) => err,
        }
    }
}

/// An ORT inference session (a loaded model).
pub struct Session {
    ptr: NonNull<onnx_genai_ort_sys::OrtSession>,
    _model_path: String,
    input_names: Vec<String>,
    output_names: Vec<String>,
    inputs: Vec<TensorInfo>,
    outputs: Vec<TensorInfo>,
    /// Execution providers requested for this session (priority order). Used to
    /// decide whether device-resident KV buffers can be allocated.
    execution_providers: Vec<ResolvedEp>,
    /// Whether the session was created with EP graph capture enabled
    /// (CUDA `enable_cuda_graph=1`). Decode runners use this to drive the
    /// static-shape captured-graph replay path.
    graph_capture: bool,
}

impl Session {
    /// Load a model from an ONNX file.
    pub fn new(env: &Environment, path: &Path, options: SessionOptions) -> Result<Self> {
        if !path.exists() {
            return Err(OrtError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Model file not found: {}", path.display()),
            )));
        }

        // Textproto fixtures (`*.textproto`) are git-friendly ONNX protobuf
        // TextFormat. ORT cannot read them from disk, so convert to binary bytes
        // (via onnx-std) and create the session from memory. Binary `.onnx` files
        // continue to load directly from the path. Textproto has no
        // model-directory context, so such fixtures must inline all weights.
        let model_bytes: Option<Vec<u8>> = if is_textproto_path(path) {
            let text = std::fs::read_to_string(path)?;
            Some(onnx_std::textproto::to_binary(&text).map_err(|err| {
                OrtError::InvalidArgument(format!(
                    "failed to convert textproto model {}: {err}",
                    path.display()
                ))
            })?)
        } else {
            None
        };
        #[cfg(windows)]
        let path_c: Vec<u16> = {
            use std::os::windows::ffi::OsStrExt;
            path.as_os_str()
                .encode_wide()
                .chain(std::iter::once(0))
                .collect()
        };
        #[cfg(not(windows))]
        let path_c = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| OrtError::InvalidArgument("model path contains NUL".into()))?;
        let api = crate::error::api()?;

        // Build session options (which registers/appends execution providers)
        // and create the session, dispatching to the from-bytes API for
        // converted textproto models and the from-path API for binary `.onnx`
        // files. Both steps can fail for a requested non-CPU provider, so keep
        // them together behind one closure that can be retried with CPU-only
        // options.
        let create_session =
            |opts: &SessionOptions| -> Result<*mut onnx_genai_ort_sys::OrtSession> {
                let session_options = RawSessionOptions::new(env, opts)?;
                let mut ptr = std::ptr::null_mut();
                match &model_bytes {
                    Some(bytes) => {
                        let create = api
                            .CreateSessionFromArray
                            .ok_or(OrtError::ApiUnavailable("CreateSessionFromArray"))?;
                        // SAFETY: `env` and `session_options` are valid ORT handles,
                        // `bytes` outlives the call, and `ptr` is an out-param.
                        crate::error::check_status(unsafe {
                            create(
                                env.as_ptr(),
                                bytes.as_ptr() as *const std::ffi::c_void,
                                bytes.len(),
                                session_options.as_ptr(),
                                &mut ptr,
                            )
                        })?;
                    }
                    None => {
                        let create = api
                            .CreateSession
                            .ok_or(OrtError::ApiUnavailable("CreateSession"))?;
                        // SAFETY: `env` and `session_options` are valid ORT handles,
                        // `path_c` is NUL-terminated for the call, and `ptr` is an
                        // out-param.
                        crate::error::check_status(unsafe {
                            create(
                                env.as_ptr(),
                                path_c.as_ptr(),
                                session_options.as_ptr(),
                                &mut ptr,
                            )
                        })?;
                    }
                }
                Ok(ptr)
            };

        // Auto-selected providers (e.g. the macOS MLX default) always fall back
        // to CPU; explicitly requested strict providers (CUDA and plugin EPs)
        // fail rather than silently changing the requested device.
        let allow_cpu_fallback = options.auto_selected
            || (requested_non_cpu_provider(&options) && !requested_strict_provider(&options));

        let (ptr, effective_providers) = match create_session(&options) {
            Ok(ptr) => (ptr, options.execution_providers.clone()),
            Err(err) if allow_cpu_fallback => {
                tracing::warn!(
                    "ORT session creation failed with requested execution provider(s): {err}; retrying with CPU"
                );
                let cpu_options = SessionOptions::cpu();
                let ptr = create_session(&cpu_options)?;
                (ptr, cpu_options.execution_providers)
            }
            Err(err) => return Err(err),
        };
        let ptr = NonNull::new(ptr).ok_or(OrtError::NullPointer)?;
        let inputs = query_io(ptr.as_ptr(), IoKind::Input)?;
        let outputs = query_io(ptr.as_ptr(), IoKind::Output)?;
        let input_names = inputs.iter().map(|info| info.name.clone()).collect();
        let output_names = outputs.iter().map(|info| info.name.clone()).collect();

        tracing::info!("Loading model: {}", path.display());

        Ok(Self {
            ptr,
            _model_path: path.display().to_string(),
            input_names,
            output_names,
            inputs,
            outputs,
            execution_providers: effective_providers,
            graph_capture: options.graph_capture,
        })
    }

    /// Run inference with named inputs, returns named outputs.
    pub fn run(&self, inputs: &[(&str, &Value)]) -> Result<Vec<Value>> {
        let input_names: Vec<CString> = inputs
            .iter()
            .map(|(name, _)| {
                CString::new(*name).map_err(|_| {
                    OrtError::InvalidArgument(format!("input name contains NUL: {name}"))
                })
            })
            .collect::<Result<_>>()?;
        let input_name_ptrs: Vec<*const std::ffi::c_char> =
            input_names.iter().map(|name| name.as_ptr()).collect();
        let input_value_ptrs: Vec<*const onnx_genai_ort_sys::OrtValue> =
            inputs.iter().map(|(_, value)| value.as_ptr()).collect();

        let output_names: Vec<CString> = self
            .output_names
            .iter()
            .map(|name| {
                CString::new(name.as_str()).map_err(|_| {
                    OrtError::InvalidArgument(format!("output name contains NUL: {name}"))
                })
            })
            .collect::<Result<_>>()?;
        let output_name_ptrs: Vec<*const std::ffi::c_char> =
            output_names.iter().map(|name| name.as_ptr()).collect();
        let mut output_ptrs = vec![std::ptr::null_mut(); output_names.len()];

        let api = crate::error::api()?;
        let run = api.Run.ok_or(OrtError::ApiUnavailable("Run"))?;
        // SAFETY: All name arrays contain NUL-terminated strings alive for the
        // call. Input OrtValues are valid borrowed handles. `output_ptrs` is an
        // array of nulls for ORT to fill with newly allocated OrtValues.
        crate::error::check_status(unsafe {
            run(
                self.ptr.as_ptr(),
                std::ptr::null(),
                input_name_ptrs.as_ptr(),
                input_value_ptrs.as_ptr(),
                input_value_ptrs.len(),
                output_name_ptrs.as_ptr(),
                output_name_ptrs.len(),
                output_ptrs.as_mut_ptr(),
            )
        })?;

        output_ptrs
            .into_iter()
            .map(|ptr| {
                // SAFETY: On successful Run, ORT filled each output pointer with
                // a newly allocated OrtValue that this wrapper now owns.
                unsafe { Value::from_raw(ptr) }
            })
            .collect()
    }

    /// Run inference using pre-bound I/O (zero-copy for device tensors).
    pub fn run_with_binding(&self, binding: &IoBinding) -> Result<()> {
        let api = crate::error::api()?;
        let run = api
            .RunWithBinding
            .ok_or(OrtError::ApiUnavailable("RunWithBinding"))?;
        // SAFETY: session and binding are valid ORT handles. A null RunOptions
        // means "use defaults" per ORT C API.
        crate::error::check_status(unsafe {
            run(self.ptr.as_ptr(), std::ptr::null(), binding.as_ptr())
        })
    }

    /// Whether this session was created with EP graph capture enabled.
    pub fn graph_capture(&self) -> bool {
        self.graph_capture
    }

    /// The CUDA device id this session runs on, if a CUDA EP was requested.
    pub fn cuda_device_id(&self) -> Option<i32> {
        self.execution_providers.iter().find_map(|ep| {
            if ep.caps.is_nvidia() && ep.caps.is_gpu() {
                ep.caps.device_id()
            } else {
                None
            }
        })
    }

    /// Run inference using pre-bound I/O, selecting a CUDA-graph annotation.
    ///
    /// `graph_annotation_id` maps to the `gpu_graph_id` run-config entry: `-1`
    /// runs without capture or replay (used for prompt/prefill steps whose
    /// shapes differ), while a stable non-negative id captures the graph on the
    /// first run of that shape and replays it on subsequent runs. This is how
    /// the static-shape decode loop replays a single captured decode graph while
    /// leaving the variable-shape prefill uncaptured.
    pub fn run_with_binding_graph(
        &self,
        binding: &IoBinding,
        graph_annotation_id: i32,
    ) -> Result<()> {
        self.run_with_binding_graph_phased(binding, graph_annotation_id)
            .map_err(RunPhaseError::into_inner)
    }

    /// Run with graph annotation while distinguishing setup from invocation failures.
    pub fn run_with_binding_graph_phased(
        &self,
        binding: &IoBinding,
        graph_annotation_id: i32,
    ) -> std::result::Result<(), RunPhaseError> {
        let api = crate::error::api().map_err(RunPhaseError::Setup)?;
        let run = api
            .RunWithBinding
            .ok_or(OrtError::ApiUnavailable("RunWithBinding"))
            .map_err(RunPhaseError::Setup)?;
        let create_opts = api
            .CreateRunOptions
            .ok_or(OrtError::ApiUnavailable("CreateRunOptions"))
            .map_err(RunPhaseError::Setup)?;
        let add_entry = api
            .AddRunConfigEntry
            .ok_or(OrtError::ApiUnavailable("AddRunConfigEntry"))
            .map_err(RunPhaseError::Setup)?;
        let release_opts = api
            .ReleaseRunOptions
            .ok_or(OrtError::ApiUnavailable("ReleaseRunOptions"))
            .map_err(RunPhaseError::Setup)?;

        let mut run_options = std::ptr::null_mut();
        // SAFETY: `run_options` is a valid out-parameter, released below.
        crate::error::check_status(unsafe { create_opts(&mut run_options) })
            .map_err(RunPhaseError::Setup)?;
        let run_options = NonNull::new(run_options)
            .ok_or(OrtError::NullPointer)
            .map_err(RunPhaseError::Setup)?;

        let result = (|| {
            let key = CString::new("gpu_graph_id").expect("literal has no NUL");
            let value =
                CString::new(graph_annotation_id.to_string()).expect("integer string has no NUL");
            // SAFETY: run options handle and NUL-terminated strings are valid.
            crate::error::check_status(unsafe {
                add_entry(run_options.as_ptr(), key.as_ptr(), value.as_ptr())
            })
            .map_err(RunPhaseError::Setup)?;
            // SAFETY: session, run options, and binding are valid ORT handles.
            crate::error::check_status(unsafe {
                run(self.ptr.as_ptr(), run_options.as_ptr(), binding.as_ptr())
            })
            .map_err(RunPhaseError::Invoked)
        })();

        // SAFETY: `run_options` was created above and is released exactly once.
        unsafe { release_opts(run_options.as_ptr()) };
        result
    }

    /// Release a previously captured CUDA graph so the next run of the matching
    /// annotation id re-captures instead of replaying.
    ///
    /// A captured graph replays against the exact device buffer addresses seen
    /// at capture time. When the [`Session`] is reused across independent
    /// generations (the server binds a fresh prefill each request), the next
    /// generation must re-capture rather than replay a stale graph, so callers
    /// release the captured decode graph on reset.
    pub fn release_captured_graph(&self, graph_annotation_id: i32) -> Result<()> {
        let api = crate::error::api()?;
        let Some(release) = api.SessionReleaseCapturedGraph else {
            return Ok(());
        };
        // SAFETY: `self.ptr` is a valid session handle for the session lifetime.
        crate::error::check_status(unsafe { release(self.ptr.as_ptr(), graph_annotation_id) })
    }

    /// Get input names.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Get output names.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Get input tensor metadata.
    pub fn inputs(&self) -> &[TensorInfo] {
        &self.inputs
    }

    /// Get output tensor metadata.
    pub fn outputs(&self) -> &[TensorInfo] {
        &self.outputs
    }

    /// Look up a custom ONNX model metadata value by key.
    pub fn custom_metadata_value(&self, key: &str) -> Result<Option<String>> {
        let key = CString::new(key)
            .map_err(|_| OrtError::InvalidArgument("metadata key contains NUL".into()))?;
        let allocator = Allocator::default_cpu()?;
        let api = crate::error::api()?;
        let get_metadata = api
            .SessionGetModelMetadata
            .ok_or(OrtError::ApiUnavailable("SessionGetModelMetadata"))?;
        let lookup = api
            .ModelMetadataLookupCustomMetadataMap
            .ok_or(OrtError::ApiUnavailable(
                "ModelMetadataLookupCustomMetadataMap",
            ))?;
        let release_metadata = api
            .ReleaseModelMetadata
            .ok_or(OrtError::ApiUnavailable("ReleaseModelMetadata"))?;
        let free = api
            .AllocatorFree
            .ok_or(OrtError::ApiUnavailable("AllocatorFree"))?;

        let mut metadata = std::ptr::null_mut();
        // SAFETY: session is valid and metadata is an out-parameter.
        crate::error::check_status(unsafe { get_metadata(self.ptr.as_ptr(), &mut metadata) })?;
        if metadata.is_null() {
            return Ok(None);
        }

        let result = (|| {
            let mut value_ptr = std::ptr::null_mut();
            // SAFETY: metadata, allocator, and key are valid for the call.
            crate::error::check_status(unsafe {
                lookup(metadata, allocator.as_ptr(), key.as_ptr(), &mut value_ptr)
            })?;
            if value_ptr.is_null() {
                return Ok(None);
            }
            // SAFETY: ORT returned a NUL-terminated string allocated by allocator.
            let value = unsafe { CStr::from_ptr(value_ptr) }
                .to_string_lossy()
                .into_owned();
            crate::error::check_status(unsafe { free(allocator.as_ptr(), value_ptr.cast()) })?;
            Ok(Some(value))
        })();

        // SAFETY: metadata was allocated by ORT and is released once.
        unsafe { release_metadata(metadata) };
        result
    }

    /// Detect whether model metadata declares ORT past/present share-buffer KV.
    pub fn past_present_share_buffer_supported(&self) -> bool {
        ["past_present_share_buffer", "past.present.share_buffer"]
            .iter()
            .filter_map(|key| self.custom_metadata_value(key).ok().flatten())
            .any(|value| {
                matches!(
                    value.to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
    }

    pub(crate) fn as_mut_ptr(&self) -> *mut onnx_genai_ort_sys::OrtSession {
        self.ptr.as_ptr()
    }

    /// Whether a CUDA execution provider is (effectively) active for this session.
    pub fn is_cuda(&self) -> bool {
        self.execution_providers
            .iter()
            .any(|ep| ep.caps.is_nvidia() && ep.caps.is_gpu())
    }

    /// Whether this session's execution provider can accept the runtime-owned,
    /// fixed-capacity present (KV) buffer as a *pre-bound* `present.*` output.
    ///
    /// WHAT: Reports whether the active EP honors ORT's pre-bound,
    /// fixed-capacity present-output contract that the O(1)/token
    /// [`ModelDecodePath::PastPresent`](crate) SharedBuffer decode path depends
    /// on. When TRUE, decode may bind the runtime-owned max-length KV buffer as
    /// the EP's `present.*` output; when FALSE, decode must fall back to the
    /// growing `ZeroCopyRebind` path.
    ///
    /// WHY: CPU, CUDA and WebGPU are the only EPs verified to consume a
    /// fixed-capacity present binding correctly and use the shared buffer
    /// successfully today. The external Metal plugin's growing-shape GQA kernel
    /// instead requests
    /// `capacity + sequence_length` elements at bind time, which fails ORT's
    /// pre-bound output-size check and crashed Metal E2E (see the KV notes in
    /// `onnx-genai-engine`'s `detect_model_decode_path`). Metal therefore
    /// declares NO fixed-capacity present support by default, preserving today's
    /// `ZeroCopyRebind` behavior. Any unverified current or future EP also
    /// defaults to NO, preventing a new provider from reintroducing this crash
    /// class. Concentrating this EP-identity knowledge in a single semantic
    /// capability keeps `is_metal()` out of decode business logic (RULES.md §2).
    ///
    /// HOW: The CPU, CUDA, and WebGPU allowlist returns TRUE. Everything else,
    /// including Metal, returns FALSE unless the operator explicitly opts in via
    /// `ONNX_GENAI_SHARED_KV_PRESENT_BINDING=1` (see
    /// [`shared_kv_present_binding_opt_in_from_env`]), which lets the default
    /// flip to enabled once the MLX/Metal EP is verified on real Apple-silicon
    /// hardware — with no further code change.
    pub fn supports_fixed_capacity_present_binding(&self) -> bool {
        fixed_capacity_present_binding_supported(
            &self.execution_providers,
            shared_kv_present_binding_opt_in_from_env(),
        )
    }

    /// Create a device-resident allocator for KV buffers, if this session runs
    /// on an execution provider that owns device memory (CUDA or WebGPU).
    ///
    /// Returns `Ok(None)` for CPU/unsupported EPs, so callers keep using the CPU
    /// allocator. If a device EP is selected but ORT cannot produce a matching
    /// allocator (e.g. the EP silently fell back to CPU), the error is logged
    /// and `Ok(None)` is returned so decode still works via CPU buffers.
    pub(crate) fn device_kv_allocator(&self) -> Result<Option<Allocator>> {
        if !self
            .execution_providers
            .iter()
            .any(|ep| ep.caps.has(capability::DEVICE_KV))
        {
            return Ok(None);
        }

        // CUDA device-resident KV is on by DEFAULT. Keeping the shared GQA KV
        // buffer in CUDA memory (instead of host memory) eliminates the
        // per-step host<->device KV copies ORT would otherwise insert on every
        // decode step. On Qwen2.5-0.5B this cut `bind_inputs` from ~45ms to
        // ~0.1ms per token and lifted CUDA decode from ~11 to ~265 tok/s
        // (beating Foundry Local) with identical, coherent output. It is
        // therefore no longer gated behind `ONNX_GENAI_DEVICE_KV`; that flag now
        // only opts the still-experimental WebGPU device allocator in (see
        // below).
        #[cfg(feature = "cuda")]
        if let Some(device_id) = self.execution_providers.iter().find_map(|ep| {
            if ep.caps.is_nvidia() && ep.caps.is_gpu() {
                ep.caps.device_id()
            } else {
                None
            }
        }) {
            let memory_info = MemoryInfo::cuda(device_id)?;
            return match Allocator::for_session_device(self.ptr.as_ptr(), memory_info) {
                Ok(allocator) => {
                    tracing::info!(device_id, "allocating shared GQA KV on CUDA device memory");
                    Ok(Some(allocator))
                }
                Err(err) => {
                    tracing::warn!(
                        "Could not create CUDA device KV allocator for device {device_id} ({err}); falling back to CPU KV buffers"
                    );
                    Ok(None)
                }
            };
        }

        // WebGPU device-resident KV remains EXPERIMENTAL (ORT 1.27 WebGPU can
        // segfault during multi-step decode), so it stays opt-in via
        // `ONNX_GENAI_DEVICE_KV=1`.
        if !device_kv_enabled_from_env() {
            return Ok(None);
        }

        let memory_info = match MemoryInfo::webgpu() {
            Ok(info) => info,
            Err(err) => {
                tracing::warn!(
                    "WebGPU device memory info unavailable ({err}); using CPU KV buffers"
                );
                return Ok(None);
            }
        };
        match Allocator::for_session_device(self.ptr.as_ptr(), memory_info) {
            Ok(allocator) => {
                tracing::warn!(
                    "ONNX_GENAI_DEVICE_KV=1: allocating shared GQA KV on the WebGPU device allocator (EXPERIMENTAL; ORT 1.27 WebGPU may segfault during multi-step decode)"
                );
                Ok(Some(allocator))
            }
            Err(err) => {
                tracing::warn!(
                    "Could not create WebGPU device KV allocator ({err}); falling back to CPU KV buffers"
                );
                Ok(None)
            }
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Ok(api) = crate::error::api()
            && let Some(release) = api.ReleaseSession
        {
            // SAFETY: `ptr` is owned by this wrapper and released exactly once here.
            unsafe { release(self.ptr.as_ptr()) };
        }
    }
}

// SAFETY: `Session` owns one `OrtSession` handle plus immutable Rust metadata.
// ONNX Runtime documents `OrtSession::Run`/`RunWithBinding` as safe for
// concurrent calls on the same session; per-run inputs, outputs, and `IoBinding`
// values are supplied by the caller and are not stored in `Session`. `Drop` still
// requires unique ownership and releases the handle exactly once. This would stop
// being sound for an execution provider that violates ORT's concurrent-run
// contract, or if future code cached mutable per-run state inside `Session`.
unsafe impl Send for Session {}
// SAFETY: Shared `&Session` access only permits ORT runs against the thread-safe
// session handle and reads immutable metadata. Callers must not share a mutable
// ORT binding/value through unsafe code across concurrent runs.
unsafe impl Sync for Session {}

struct RawSessionOptions {
    ptr: NonNull<onnx_genai_ort_sys::OrtSessionOptions>,
}

impl RawSessionOptions {
    fn new(env: &Environment, options: &SessionOptions) -> Result<Self> {
        let api = crate::error::api()?;
        let create = api
            .CreateSessionOptions
            .ok_or(OrtError::ApiUnavailable("CreateSessionOptions"))?;
        let mut ptr = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid out-parameter and is owned on success.
        crate::error::check_status(unsafe { create(&mut ptr) })?;
        let this = Self {
            ptr: NonNull::new(ptr).ok_or(OrtError::NullPointer)?,
        };

        if let Some(set_opt) = api.SetSessionGraphOptimizationLevel {
            let level = match options.optimization_level {
                0 => onnx_genai_ort_sys::ORT_DISABLE_ALL,
                1 => onnx_genai_ort_sys::ORT_ENABLE_BASIC,
                2 => onnx_genai_ort_sys::ORT_ENABLE_EXTENDED,
                _ => onnx_genai_ort_sys::ORT_ENABLE_ALL,
            };
            // SAFETY: `this.ptr` is a valid session options handle.
            crate::error::check_status(unsafe { set_opt(this.ptr.as_ptr(), level) })?;
        }
        // Resolve the effective intra-op thread count. An explicit
        // `with_intra_op_threads(n)` (n > 0) always wins so exact-equality tests
        // keep forcing single-thread ORT. When the caller left it at the default
        // (0 = "ORT decides"), `ONNX_GENAI_INTRA_OP_THREADS` may override it.
        // This is the profiler-identified lever: ORT's default oversubscribes
        // Apple-silicon efficiency cores (10-thread decode is ~2x slower than a
        // 6-8 performance-core config), so operators can pin it without a code
        // change. See the CPU decode profiling decision note.
        let effective_intra_op = if options.intra_op_num_threads > 0 {
            options.intra_op_num_threads
        } else {
            intra_op_threads_from_env().unwrap_or(0)
        };
        if effective_intra_op > 0
            && let Some(set_threads) = api.SetIntraOpNumThreads
        {
            // SAFETY: `this.ptr` is a valid session options handle.
            crate::error::check_status(unsafe {
                set_threads(this.ptr.as_ptr(), effective_intra_op)
            })?;
        }
        if options.inter_op_num_threads > 0
            && let Some(set_threads) = api.SetInterOpNumThreads
        {
            // SAFETY: `this.ptr` is a valid session options handle.
            crate::error::check_status(unsafe {
                set_threads(this.ptr.as_ptr(), options.inter_op_num_threads)
            })?;
        }

        append_execution_providers(env, this.ptr.as_ptr(), options)?;
        apply_webgpu_provider_options(this.ptr.as_ptr(), options)?;

        Ok(this)
    }

    fn as_ptr(&self) -> *const onnx_genai_ort_sys::OrtSessionOptions {
        self.ptr.as_ptr()
    }
}

fn execution_providers_from_env() -> Option<Vec<ResolvedEp>> {
    let entries = &runtime_config().execution_providers;
    if entries.is_empty() {
        return None;
    }
    let providers = entries
        .iter()
        .filter_map(|entry| match entry {
            ExecutionProviderEntry::Builtin(selection) if selection.name == "plugin" => {
                let config = runtime_config();
                let library = config.ep_library.clone()?;
                Some(resolve_plugin_selection(
                    selection.clone(),
                    library.clone(),
                    config
                        .ep_registration_name
                        .clone()
                        .unwrap_or_else(|| plugin_registration_name_from_path(&library)),
                    config.ep_options.clone(),
                    config.ep_device.clone(),
                ))
            }
            ExecutionProviderEntry::Builtin(selection) => {
                Some(resolve_execution_provider(selection))
            }
            ExecutionProviderEntry::Plugin(spec) => resolve_inline_plugin(spec),
        })
        .collect::<Vec<_>>();
    (!providers.is_empty()).then_some(providers)
}

fn resolve_inline_plugin(spec: &PluginSpec) -> Option<ResolvedEp> {
    if spec.library.as_os_str().is_empty() {
        tracing::warn!("Ignoring inline plugin entry with an empty library path");
        return None;
    }
    Some(resolve_plugin_selection(
        EpSelection::new("plugin"),
        spec.library.clone(),
        spec.registration_name
            .clone()
            .unwrap_or_else(|| plugin_registration_name_from_path(&spec.library)),
        spec.options.clone(),
        spec.device.clone(),
    ))
}

fn resolve_plugin_selection(
    selection: EpSelection,
    library: std::path::PathBuf,
    registration_name: String,
    options: Vec<(String, String)>,
    device: Option<String>,
) -> ResolvedEp {
    let hardware = match device.as_deref().map(str::to_ascii_uppercase).as_deref() {
        Some("CPU") => HardwareKind::Cpu,
        Some("GPU") => HardwareKind::Gpu,
        Some("NPU") => HardwareKind::Npu,
        _ => HardwareKind::Other,
    };
    ResolvedEp {
        caps: EpCapabilities::new(selection.name.clone(), hardware, None, None, &[]),
        selection,
        strategy: ep_compat::AppendStrategy::PluginLibrary {
            lib: library,
            registration_name,
            options,
            device,
        },
        graph_capture_env: false,
        transitional_webgpu: false,
    }
}

fn requested_non_cpu_provider(options: &SessionOptions) -> bool {
    options
        .execution_providers
        .iter()
        .any(|ep| !ep.caps.is_host())
}

/// Whether `path` names an ONNX protobuf TextFormat fixture (`*.textproto`).
///
/// Textproto models are git-friendly text; ORT cannot read them from disk, so
/// [`Session::new`] converts them to binary bytes and loads them from memory.
fn is_textproto_path(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("textproto"))
}

fn requested_strict_provider(options: &SessionOptions) -> bool {
    options
        .execution_providers
        .iter()
        .any(ResolvedEp::is_strict)
}

fn cuda_device_id_from_env() -> i32 {
    match &runtime_config().cuda_device {
        CudaDevice::Id(device_id) => *device_id,
        CudaDevice::Invalid(value) => {
            tracing::warn!(
                "Ignoring invalid ONNX_GENAI_CUDA_DEVICE={value}; expected a non-negative integer, using device 0"
            );
            0
        }
    }
}

/// Optional intra-op thread override from `ONNX_GENAI_INTRA_OP_THREADS`.
///
/// Only consulted when the caller left `intra_op_num_threads` at the default
/// (0 = "ORT decides"); an explicit `with_intra_op_threads` always wins. A
/// positive integer pins the ORT intra-op pool. This is the profiler-identified
/// CPU decode lever: ORT's default oversubscribes Apple-silicon efficiency
/// cores, so a 10-thread decode measured ~2x slower than a 6-8 performance-core
/// config. Invalid or non-positive values are ignored with a warning.
fn intra_op_threads_from_env() -> Option<i32> {
    match &runtime_config().intra_op_threads {
        IntraOpThreads::Unset => None,
        IntraOpThreads::Count(threads) => Some(*threads),
        IntraOpThreads::Invalid(value) => {
            tracing::warn!(
                "Ignoring invalid ONNX_GENAI_INTRA_OP_THREADS={value}; expected a positive integer"
            );
            None
        }
    }
}

/// Whether to disable WebGPU validation. Default true (safe overhead
/// reduction); set `ONNX_GENAI_WEBGPU_VALIDATION=1` to keep validation on.
fn webgpu_disable_validation_from_env() -> bool {
    !runtime_config().webgpu_validation
}

/// CUDA attention mode from `ONNX_GENAI_CUDA_ATTENTION`.
///
/// ORT exposes the desired behavior as the CUDA provider option
/// `sdpa_kernel=16` (the standard math implementation), so this configuration
/// does not need to mutate ORT's process-wide attention environment variables.
fn cuda_attention_mode_from_env() -> CudaAttentionMode {
    let Some(value) = std::env::var_os("ONNX_GENAI_CUDA_ATTENTION") else {
        return CudaAttentionMode::Default;
    };
    match value.to_string_lossy().trim().to_ascii_lowercase().as_str() {
        "" | "default" | "optimized" => CudaAttentionMode::Default,
        "unfused" => CudaAttentionMode::Unfused,
        invalid => {
            tracing::warn!(
                "Ignoring invalid ONNX_GENAI_CUDA_ATTENTION={invalid}; expected 'default' or 'unfused'"
            );
            CudaAttentionMode::Default
        }
    }
}

/// Whether device-resident KV buffers are enabled. Default **false**: on the
/// ORT 1.27 WebGPU EP, binding a user-pre-allocated `WebGPU_Buffer` device
/// tensor as a persistent in-place `past`/`present` share-buffer segfaults
/// (`EXC_BAD_ACCESS`, call through a null function pointer) during multi-step
/// decode. Set `ONNX_GENAI_DEVICE_KV=1` to opt in experimentally once ORT
/// supports external device KV tensors. See
/// `.squad/decisions/inbox/leon-device-resident-kv.md`.
fn device_kv_enabled_from_env() -> bool {
    runtime_config().device_kv
}

/// Explicit operator opt-in that lets an otherwise unverified EP participate in
/// the fixed-capacity, pre-bound present-output (SharedBuffer) decode path.
///
/// WHAT: Reads `ONNX_GENAI_SHARED_KV_PRESENT_BINDING` and returns TRUE for the
/// usual truthy values (`1`/`true`/`yes`/`on`), FALSE otherwise (including
/// unset).
///
/// WHY: The verified-EP allowlist in [`fixed_capacity_present_binding_supported`]
/// gates the SharedBuffer path. The Metal plugin EP now implements the
/// fixed-capacity in-place-write GQA contract and is on that allowlist, so this
/// flag is no longer needed for Metal. It remains a global operator override so
/// an as-yet-unverified EP (e.g. CoreML) can opt into SharedBuffer without a
/// code change.
///
/// HOW: Consumed only by
/// [`Session::supports_fixed_capacity_present_binding`]; it overrides the
/// conservative capability allowlist.
fn shared_kv_present_binding_opt_in_from_env() -> bool {
    runtime_config().shared_kv_present_binding
}

/// Resolve fixed-capacity present binding from EP capabilities, with an explicit
/// operator override for unverified EPs.
fn fixed_capacity_present_binding_supported(providers: &[ResolvedEp], opt_in: bool) -> bool {
    opt_in
        || !providers.is_empty()
            && providers
                .iter()
                .all(|ep| ep.caps.has(capability::FIXED_CAPACITY_PRESENT_BINDING))
}

/// Apply WebGPU EP provider options via session config entries.
///
/// The WebGPU EP reads these from the merged `ConfigOptions` (see ORT
/// `webgpu_provider_factory.cc`), keyed by the full `ep.webgpuexecutionprovider.*`
/// names. `AddSessionConfigEntry` is the EP-agnostic way to set them. No-ops
/// unless a WebGPU EP is selected.
fn apply_webgpu_provider_options(
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    options: &SessionOptions,
) -> Result<()> {
    if !options.selects_webgpu() {
        return Ok(());
    }
    if options.webgpu_disable_validation {
        add_session_config_entry(
            session_options,
            "ep.webgpuexecutionprovider.validationMode",
            "disabled",
        )?;
    }
    if options.graph_capture {
        add_session_config_entry(
            session_options,
            "ep.webgpuexecutionprovider.enableGraphCapture",
            "1",
        )?;
        tracing::info!("Enabled ONNX Runtime WebGPU graph capture");
    }
    Ok(())
}

fn add_session_config_entry(
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    key: &str,
    value: &str,
) -> Result<()> {
    let api = crate::error::api()?;
    let add = api
        .AddSessionConfigEntry
        .ok_or(OrtError::ApiUnavailable("AddSessionConfigEntry"))?;
    let key_c = CString::new(key)
        .map_err(|_| OrtError::InvalidArgument("session config key contains NUL".into()))?;
    let value_c = CString::new(value)
        .map_err(|_| OrtError::InvalidArgument("session config value contains NUL".into()))?;
    // SAFETY: `session_options` is a valid handle; both C strings are
    // NUL-terminated and live for the call.
    crate::error::check_status(unsafe { add(session_options, key_c.as_ptr(), value_c.as_ptr()) })
}

fn append_execution_providers(
    env: &Environment,
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    options: &SessionOptions,
) -> Result<()> {
    let available = available_execution_providers().unwrap_or_else(|err| {
        tracing::warn!("Could not query available ORT execution providers: {err}");
        Vec::new()
    });
    for provider in &options.execution_providers {
        append_execution_provider(
            env,
            session_options,
            provider,
            options.graph_capture,
            options.cuda_attention_mode,
            &available,
        )?;
    }
    Ok(())
}

fn append_execution_provider(
    env: &Environment,
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    provider: &ResolvedEp,
    graph_capture: bool,
    cuda_attention_mode: CudaAttentionMode,
    available: &[String],
) -> Result<()> {
    use ep_compat::AppendStrategy;
    match &provider.strategy {
        AppendStrategy::HostDefault => Ok(()),
        #[cfg(feature = "cuda")]
        AppendStrategy::CudaTyped { device_id } => append_cuda_execution_provider(
            session_options,
            *device_id,
            graph_capture,
            cuda_attention_mode,
            available,
        ),
        #[cfg(not(feature = "cuda"))]
        AppendStrategy::CudaUnavailable => {
            let _ = (
                session_options,
                graph_capture,
                cuda_attention_mode,
                available,
            );
            Err(OrtError::InvalidArgument(
                "CUDA support not compiled in; rebuild with --features cuda".into(),
            ))
        }
        AppendStrategy::PluginLibrary {
            lib,
            registration_name,
            options,
            device,
        } => append_plugin_execution_provider(
            env,
            session_options,
            registration_name,
            lib,
            options,
            device.as_deref(),
        ),
        AppendStrategy::NamedGeneric {
            ort_name,
            provider_name,
        } => {
            let provider_options = named_provider_options(provider);
            append_named_execution_provider(
                session_options,
                ort_name,
                provider_name,
                &provider_options,
                available,
            )
        }
    }
}

fn named_provider_options(provider: &ResolvedEp) -> Vec<(&str, &str)> {
    provider
        .selection
        .options
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect()
}

/// Derive a stable registration handle for a plugin library from its file name.
///
/// This is only an opaque handle passed to ORT's
/// `RegisterExecutionProviderLibrary`; it does not need to match (and must not
/// be confused with) the provider's internal EP name.
fn plugin_registration_name_from_path(path: &std::path::Path) -> String {
    path.file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .filter(|stem| !stem.is_empty())
        .unwrap_or_else(|| "onnx_genai_ep_plugin".to_string())
}

/// Map a portable hardware-device class string to ORT's generic
/// `OrtHardwareDeviceType`. Accepts `CPU`, `GPU`, and `NPU` case-insensitively.
/// This is intentionally provider-agnostic: it never matches a vendor's device
/// name, only ORT's own hardware-class enum.
fn parse_hardware_device_type(value: &str) -> Option<onnx_genai_ort_sys::OrtHardwareDeviceType> {
    match value.trim().to_ascii_uppercase().as_str() {
        "CPU" => Some(onnx_genai_ort_sys::OrtHardwareDeviceType_CPU),
        "GPU" => Some(onnx_genai_ort_sys::OrtHardwareDeviceType_GPU),
        "NPU" => Some(onnx_genai_ort_sys::OrtHardwareDeviceType_NPU),
        _ => None,
    }
}

/// Register an ORT execution-provider plugin shared library and append every
/// device it contributes to `session_options`.
///
/// The plugin's provider is identified WITHOUT hardcoding its name: we snapshot
/// the environment's EP devices before registration and append only the devices
/// that appear afterwards. This mirrors the documented plugin-EP registration
/// flow (`RegisterExecutionProviderLibrary` + `GetEpDevices` +
/// `SessionOptionsAppendExecutionProvider_V2`) used by packages such as
/// `onnxruntime-ep-openvino`, so it works for any ORT >= 1.22 plugin EP
/// (OpenVINO, NV TensorRT RTX, QNN, ...).
fn append_plugin_execution_provider(
    env: &Environment,
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    registration_name: &str,
    plugin_path: &std::path::Path,
    options: &[(String, String)],
    device_class: Option<&str>,
) -> Result<()> {
    if !plugin_path.is_file() {
        return Err(OrtError::InvalidArgument(format!(
            "execution provider plugin library not found at {}",
            plugin_path.display()
        )));
    }

    let api = crate::error::api()?;
    let get_ep_devices = api
        .GetEpDevices
        .ok_or(OrtError::ApiUnavailable("GetEpDevices"))?;
    let ep_name = api
        .EpDevice_EpName
        .ok_or(OrtError::ApiUnavailable("EpDevice_EpName"))?;
    let append = api
        .SessionOptionsAppendExecutionProvider_V2
        .ok_or(OrtError::ApiUnavailable(
            "SessionOptionsAppendExecutionProvider_V2",
        ))?;

    // ORT plugin registration is process-global. Keep the device snapshot,
    // registration, and provider-name cache update atomic with respect to other
    // environments registering plugins concurrently.
    let discovery_guard = env.lock_plugin_discovery()?;

    // Query the environment's current EP devices as a list of raw pointers.
    let query_devices = || -> Result<Vec<*const onnx_genai_ort_sys::OrtEpDevice>> {
        let mut devices_ptr: *const *const onnx_genai_ort_sys::OrtEpDevice = std::ptr::null();
        let mut count = 0usize;
        // SAFETY: the environment is live; both output pointers are valid.
        crate::error::check_status(unsafe {
            get_ep_devices(env.as_ptr(), &mut devices_ptr, &mut count)
        })?;
        let mut out = Vec::new();
        if !devices_ptr.is_null() {
            for index in 0..count {
                // SAFETY: ORT returned an array of `count` entries.
                let device = unsafe { *devices_ptr.add(index) };
                if !device.is_null() {
                    out.push(device);
                }
            }
        }
        Ok(out)
    };
    // Read an EP device's provider name (discovered, never hardcoded).
    let name_of = |device: *const onnx_genai_ort_sys::OrtEpDevice| -> Option<String> {
        // SAFETY: `device` is owned by the live environment.
        let name_ptr = unsafe { ep_name(device) };
        if name_ptr.is_null() {
            return None;
        }
        // SAFETY: ORT EP names are NUL-terminated strings.
        Some(
            unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned(),
        )
    };

    // Snapshot the EP-name multiset before registering so we can identify the
    // devices the plugin contributes without knowing its name in advance.
    let mut before_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for device in query_devices()? {
        if let Some(name) = name_of(device) {
            *before_counts.entry(name).or_insert(0) += 1;
        }
    }
    let before = before_counts;

    let newly_registered =
        env.register_execution_provider_library(registration_name, plugin_path)?;

    // After registration, group the environment's EP devices by provider name.
    let after: Vec<(*const onnx_genai_ort_sys::OrtEpDevice, String)> = query_devices()?
        .into_iter()
        .filter_map(|device| name_of(device).map(|name| (device, name)))
        .collect();

    // Determine the plugin's provider name.
    //
    // On the first registration for this handle, the provider is discovered by
    // the before/after device diff: the name whose device count grew (never
    // hardcoded). When the library was already registered on this shared
    // environment (e.g. a second session such as a speculative-decode draft),
    // the diff is empty because the devices were already present, so we reuse
    // the provider name discovered on the first registration instead.
    let target_name = if newly_registered {
        let mut after_counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for (_, name) in &after {
            *after_counts.entry(name.as_str()).or_insert(0) += 1;
        }
        let mut new_names: Vec<&str> = after_counts
            .iter()
            .filter(|(name, count)| **count > before.get(**name).copied().unwrap_or(0))
            .map(|(name, _)| *name)
            .collect();
        new_names.sort_unstable();

        // A plugin may expose several provider groupings (e.g. OpenVINO registers
        // both `OpenVINOExecutionProvider` and virtual `OpenVINOExecutionProvider.AUTO`
        // devices). ORT requires every device appended in one call to share a single
        // EP, so choose one provider group deterministically: prefer the base
        // provider name (no `.` suffix) over virtual variants, else the first sorted.
        let discovered = new_names
            .iter()
            .find(|name| !name.contains('.'))
            .or_else(|| new_names.first())
            .map(|name| (*name).to_owned());
        match discovered {
            Some(name) => {
                env.cache_plugin_provider(registration_name, &name)?;
                name
            }
            None => {
                return Err(OrtError::InvalidArgument(format!(
                    "execution provider plugin '{registration_name}' registered from {} but contributed no new execution-provider devices",
                    plugin_path.display()
                )));
            }
        }
    } else {
        match env.cached_plugin_provider(registration_name)? {
            Some(name) => name,
            None => {
                return Err(OrtError::InvalidArgument(format!(
                    "execution provider plugin '{registration_name}' (from {}) was already registered but its provider name is unknown",
                    plugin_path.display()
                )));
            }
        }
    };
    drop(discovery_guard);

    let mut selected: Vec<*const onnx_genai_ort_sys::OrtEpDevice> = after
        .iter()
        .filter(|(_, name)| *name == target_name)
        .map(|(device, _)| *device)
        .collect();
    let selected_name = Some(target_name);

    if selected.is_empty() {
        return Err(OrtError::InvalidArgument(format!(
            "execution provider plugin '{registration_name}' registered from {} but contributed no execution-provider devices",
            plugin_path.display()
        )));
    }

    // If the caller asked for a specific hardware-device class (CPU/GPU/NPU),
    // narrow the selection to a single matching device. A plugin may expose one
    // EP name spanning several hardware devices (e.g. OpenVINO advertising both
    // GPU and CPU); ORT's `AppendExecutionProvider_V2` chooses a device from the
    // list it is given, so filtering here is how a portable device request is
    // honoured. The class is matched against ORT's generic `OrtHardwareDeviceType`
    // enum, never a provider-specific device string.
    if let Some(requested) = device_class {
        if let Some(wanted) = parse_hardware_device_type(requested) {
            match (api.EpDevice_Device, api.HardwareDevice_Type) {
                (Some(ep_device_device), Some(hw_type)) => {
                    let matching: Vec<*const onnx_genai_ort_sys::OrtEpDevice> = selected
                        .iter()
                        .copied()
                        .filter(|device| {
                            // SAFETY: `device` is owned by the live environment; the
                            // returned hardware handle is owned by ORT.
                            let hw = unsafe { ep_device_device(*device) };
                            !hw.is_null() && unsafe { hw_type(hw) } == wanted
                        })
                        .collect();
                    if matching.is_empty() {
                        return Err(OrtError::InvalidArgument(format!(
                            "execution provider plugin '{registration_name}' exposes no {requested} device; \
                             unset ONNX_GENAI_EP_DEVICE or choose an available hardware class"
                        )));
                    }
                    // Keep a single device so the plugin cannot silently fall back to
                    // a different one.
                    selected = vec![matching[0]];
                }
                _ => {
                    // The request cannot be honoured without device introspection;
                    // fail loudly rather than silently running on an arbitrary device.
                    return Err(OrtError::ApiUnavailable(
                        "EpDevice_Device/HardwareDevice_Type (required for ONNX_GENAI_EP_DEVICE selection)",
                    ));
                }
            }
        } else {
            tracing::warn!(
                requested,
                "ONNX_GENAI_EP_DEVICE is not a recognized hardware class (expected CPU, GPU, or NPU); ignoring"
            );
        }
    }

    // Provider options are provider-defined; pass keys/values through verbatim.
    let option_keys = options
        .iter()
        .map(|(key, _)| {
            CString::new(key.as_str())
                .map_err(|_| OrtError::InvalidArgument("EP option key contains NUL".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let option_values = options
        .iter()
        .map(|(_, value)| {
            CString::new(value.as_str())
                .map_err(|_| OrtError::InvalidArgument("EP option value contains NUL".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let key_ptrs = option_keys.iter().map(|k| k.as_ptr()).collect::<Vec<_>>();
    let value_ptrs = option_values.iter().map(|v| v.as_ptr()).collect::<Vec<_>>();
    let (key_ptr, value_ptr) = if options.is_empty() {
        (std::ptr::null(), std::ptr::null())
    } else {
        (key_ptrs.as_ptr(), value_ptrs.as_ptr())
    };

    // SAFETY: selected devices belong to the live environment; the session
    // options handle is valid; the key/value arrays each hold `options.len()`
    // NUL-terminated strings that outlive the call.
    crate::error::check_status(unsafe {
        append(
            session_options,
            env.as_ptr().cast_mut(),
            selected.as_ptr(),
            selected.len(),
            key_ptr,
            value_ptr,
            options.len(),
        )
    })?;
    tracing::info!(
        plugin = %plugin_path.display(),
        registration = registration_name,
        provider = selected_name.as_deref().unwrap_or("<unknown>"),
        devices = selected.len(),
        "Enabled ONNX Runtime execution provider plugin"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn append_cuda_execution_provider(
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    device_id: i32,
    graph_capture: bool,
    attention_mode: CudaAttentionMode,
    available: &[String],
) -> Result<()> {
    const PROVIDER_NAME: &str = "CUDAExecutionProvider";
    if !provider_is_available(PROVIDER_NAME, available) {
        return Err(cuda_provider_unavailable_error(available));
    }

    let api = crate::error::api()?;
    let create = api
        .CreateCUDAProviderOptions
        .ok_or(OrtError::ApiUnavailable("CreateCUDAProviderOptions"))?;
    let update = api
        .UpdateCUDAProviderOptions
        .ok_or(OrtError::ApiUnavailable("UpdateCUDAProviderOptions"))?;
    let append =
        api.SessionOptionsAppendExecutionProvider_CUDA_V2
            .ok_or(OrtError::ApiUnavailable(
                "SessionOptionsAppendExecutionProvider_CUDA_V2",
            ))?;
    let release = api
        .ReleaseCUDAProviderOptions
        .ok_or(OrtError::ApiUnavailable("ReleaseCUDAProviderOptions"))?;

    let mut cuda_options = std::ptr::null_mut();
    // SAFETY: `cuda_options` is a valid out-parameter and is released below.
    crate::error::check_status(unsafe { create(&mut cuda_options) })?;
    let result = (|| {
        let device_id = device_id.to_string();
        let provider_options = cuda_provider_options(device_id, graph_capture, attention_mode);
        let option_keys = provider_options
            .iter()
            .map(|(key, _)| {
                CString::new(key.as_str()).map_err(|_| {
                    OrtError::InvalidArgument("CUDA provider option key contains NUL".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let option_values = provider_options
            .iter()
            .map(|(_, value)| {
                CString::new(value.as_str()).map_err(|_| {
                    OrtError::InvalidArgument("CUDA provider option value contains NUL".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let option_key_ptrs = option_keys
            .iter()
            .map(|key| key.as_ptr())
            .collect::<Vec<_>>();
        let option_value_ptrs = option_values
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();
        // SAFETY: the CUDA options handle and all C string arrays are valid for
        // the calls; `session_options` is a live mutable session-options handle.
        crate::error::check_status(unsafe {
            update(
                cuda_options,
                option_key_ptrs.as_ptr(),
                option_value_ptrs.as_ptr(),
                provider_options.len(),
            )
        })?;
        crate::error::check_status(unsafe { append(session_options, cuda_options) })
    })();
    // SAFETY: `cuda_options` was created above and is released exactly once.
    unsafe { release(cuda_options) };

    match result {
        Ok(()) => {
            tracing::info!(
                device_id,
                graph_capture,
                ?attention_mode,
                "Enabled ONNX Runtime CUDA execution provider"
            );
            Ok(())
        }
        Err(err) => Err(OrtError::SessionCreation(format!(
            "failed to initialize requested CUDAExecutionProvider for device {device_id}: {err}. \
             Verify that {} and its CUDA/cuDNN dependencies are loadable from {}; \
             to intentionally run on CPU, request it explicitly with ONNX_GENAI_EP=cpu",
            cuda_provider_library_name(),
            cuda_library_search_path()
        ))),
    }
}

#[cfg(feature = "cuda")]
fn cuda_provider_options(
    device_id: String,
    graph_capture: bool,
    attention_mode: CudaAttentionMode,
) -> Vec<(String, String)> {
    let mut options = vec![("device_id".to_string(), device_id)];
    if graph_capture {
        options.push(("enable_cuda_graph".to_string(), "1".to_string()));
    }
    if attention_mode == CudaAttentionMode::Unfused {
        // ORT AttentionBackend::MATH is bit 16. A positive sdpa_kernel value is
        // an explicit backend mask, so all optimized paths are disabled without
        // process-global ORT_DISABLE_* environment state.
        options.push(("sdpa_kernel".to_string(), "16".to_string()));
    }
    options
}

#[cfg(feature = "cuda")]
fn cuda_provider_unavailable_error(available: &[String]) -> OrtError {
    OrtError::SessionCreation(format!(
        "CUDAExecutionProvider was requested, but the linked ONNX Runtime does not report it \
         (available providers: {available:?}). The CUDA provider library '{}' is missing or could \
         not be loaded. Put the directory containing both the ONNX Runtime core library and '{}' \
         first in {}, and ensure its CUDA/cuDNN dependencies are loadable; to intentionally run \
         on CPU, request it explicitly with ONNX_GENAI_EP=cpu",
        cuda_provider_library_name(),
        cuda_provider_library_name(),
        cuda_library_search_path()
    ))
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn cuda_provider_library_name() -> &'static str {
    "onnxruntime_providers_cuda.dll"
}

#[cfg(all(feature = "cuda", target_os = "macos"))]
fn cuda_provider_library_name() -> &'static str {
    "libonnxruntime_providers_cuda.dylib"
}

#[cfg(all(feature = "cuda", not(any(target_os = "windows", target_os = "macos"))))]
fn cuda_provider_library_name() -> &'static str {
    "libonnxruntime_providers_cuda.so"
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn cuda_library_search_path() -> &'static str {
    "PATH"
}

#[cfg(all(feature = "cuda", target_os = "macos"))]
fn cuda_library_search_path() -> &'static str {
    "DYLD_LIBRARY_PATH"
}

#[cfg(all(feature = "cuda", not(any(target_os = "windows", target_os = "macos"))))]
fn cuda_library_search_path() -> &'static str {
    "LD_LIBRARY_PATH"
}

fn append_named_execution_provider(
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    api_name: &str,
    provider_name: &str,
    provider_options: &[(&str, &str)],
    available: &[String],
) -> Result<()> {
    if !provider_is_available(provider_name, available) {
        tracing::warn!(
            "Requested ONNX Runtime execution provider {api_name} is unavailable in this build; falling back to CPU. Available providers: {:?}",
            available
        );
        return Ok(());
    }

    let api = crate::error::api()?;
    let append = api
        .SessionOptionsAppendExecutionProvider
        .ok_or(OrtError::ApiUnavailable(
            "SessionOptionsAppendExecutionProvider",
        ))?;
    let api_name = CString::new(api_name)
        .map_err(|_| OrtError::InvalidArgument("execution provider name contains NUL".into()))?;
    let option_keys = provider_options
        .iter()
        .map(|(key, _)| {
            CString::new(*key)
                .map_err(|_| OrtError::InvalidArgument("provider option key contains NUL".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let option_values = provider_options
        .iter()
        .map(|(_, value)| {
            CString::new(*value)
                .map_err(|_| OrtError::InvalidArgument("provider option value contains NUL".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let option_key_ptrs = option_keys
        .iter()
        .map(|key| key.as_ptr())
        .collect::<Vec<_>>();
    let option_value_ptrs = option_values
        .iter()
        .map(|value| value.as_ptr())
        .collect::<Vec<_>>();
    // SAFETY: `session_options` is a valid mutable ORT session options handle,
    // all C strings are NUL-terminated and live for the call, and the key/value
    // arrays have `provider_options.len()` entries.
    match crate::error::check_status(unsafe {
        append(
            session_options,
            api_name.as_ptr(),
            option_key_ptrs.as_ptr(),
            option_value_ptrs.as_ptr(),
            provider_options.len(),
        )
    }) {
        Ok(()) => {
            tracing::info!("Enabled ONNX Runtime execution provider {provider_name}");
            Ok(())
        }
        Err(err) => {
            tracing::warn!(
                "Failed to enable ONNX Runtime execution provider {provider_name}: {err}; falling back to CPU"
            );
            Ok(())
        }
    }
}

fn provider_is_available(provider_name: &str, available: &[String]) -> bool {
    available.iter().any(|provider| {
        provider.eq_ignore_ascii_case(provider_name)
            || provider
                .strip_suffix("ExecutionProvider")
                .is_some_and(|short| short.eq_ignore_ascii_case(provider_name))
            || provider_name
                .strip_suffix("ExecutionProvider")
                .is_some_and(|short| short.eq_ignore_ascii_case(provider))
    })
}

impl Drop for RawSessionOptions {
    fn drop(&mut self) {
        if let Ok(api) = crate::error::api()
            && let Some(release) = api.ReleaseSessionOptions
        {
            // SAFETY: `ptr` is owned by this wrapper and released exactly once here.
            unsafe { release(self.ptr.as_ptr()) };
        }
    }
}

enum IoKind {
    Input,
    Output,
}

fn query_io(
    session: *const onnx_genai_ort_sys::OrtSession,
    kind: IoKind,
) -> Result<Vec<TensorInfo>> {
    let api = crate::error::api()?;
    let mut count = 0usize;
    // SAFETY: `session` is a valid ORT session; `count` is an out-parameter.
    match kind {
        IoKind::Input => {
            let f = api
                .SessionGetInputCount
                .ok_or(OrtError::ApiUnavailable("SessionGetInputCount"))?;
            crate::error::check_status(unsafe { f(session, &mut count) })?;
        }
        IoKind::Output => {
            let f = api
                .SessionGetOutputCount
                .ok_or(OrtError::ApiUnavailable("SessionGetOutputCount"))?;
            crate::error::check_status(unsafe { f(session, &mut count) })?;
        }
    }

    (0..count)
        .map(|index| query_one_io(session, &kind, index))
        .collect()
}

fn query_one_io(
    session: *const onnx_genai_ort_sys::OrtSession,
    kind: &IoKind,
    index: usize,
) -> Result<TensorInfo> {
    let api = crate::error::api()?;
    let allocator = Allocator::default_cpu()?;
    let mut name_ptr = std::ptr::null_mut();
    match kind {
        IoKind::Input => {
            let f = api
                .SessionGetInputName
                .ok_or(OrtError::ApiUnavailable("SessionGetInputName"))?;
            // SAFETY: `session` and allocator are valid; `name_ptr` is an out-param.
            crate::error::check_status(unsafe {
                f(session, index, allocator.as_ptr(), &mut name_ptr)
            })?;
        }
        IoKind::Output => {
            let f = api
                .SessionGetOutputName
                .ok_or(OrtError::ApiUnavailable("SessionGetOutputName"))?;
            // SAFETY: `session` and allocator are valid; `name_ptr` is an out-param.
            crate::error::check_status(unsafe {
                f(session, index, allocator.as_ptr(), &mut name_ptr)
            })?;
        }
    }
    if name_ptr.is_null() {
        return Err(OrtError::NullPointer);
    }
    // SAFETY: ORT returned a valid NUL-terminated name allocated by allocator.
    let name = unsafe { CStr::from_ptr(name_ptr) }
        .to_string_lossy()
        .into_owned();
    let free = api
        .AllocatorFree
        .ok_or(OrtError::ApiUnavailable("AllocatorFree"))?;
    // SAFETY: `name_ptr` was allocated by `allocator` and is freed once.
    crate::error::check_status(unsafe { free(allocator.as_ptr(), name_ptr.cast()) })?;

    let mut type_info = std::ptr::null_mut();
    match kind {
        IoKind::Input => {
            let f = api
                .SessionGetInputTypeInfo
                .ok_or(OrtError::ApiUnavailable("SessionGetInputTypeInfo"))?;
            // SAFETY: `type_info` is an out-parameter.
            crate::error::check_status(unsafe { f(session, index, &mut type_info) })?;
        }
        IoKind::Output => {
            let f = api
                .SessionGetOutputTypeInfo
                .ok_or(OrtError::ApiUnavailable("SessionGetOutputTypeInfo"))?;
            // SAFETY: `type_info` is an out-parameter.
            crate::error::check_status(unsafe { f(session, index, &mut type_info) })?;
        }
    }
    let (dtype, shape) = tensor_info_from_type_info(type_info)?;
    if let Some(release) = api.ReleaseTypeInfo {
        // SAFETY: `type_info` was allocated by ORT and is released once.
        unsafe { release(type_info) };
    }

    Ok(TensorInfo { name, dtype, shape })
}

fn tensor_info_from_type_info(
    type_info: *mut onnx_genai_ort_sys::OrtTypeInfo,
) -> Result<(DataType, Vec<i64>)> {
    if type_info.is_null() {
        return Err(OrtError::NullPointer);
    }
    let api = crate::error::api()?;
    let cast = api
        .CastTypeInfoToTensorInfo
        .ok_or(OrtError::ApiUnavailable("CastTypeInfoToTensorInfo"))?;
    let get_type = api
        .GetTensorElementType
        .ok_or(OrtError::ApiUnavailable("GetTensorElementType"))?;
    let get_dim_count = api
        .GetDimensionsCount
        .ok_or(OrtError::ApiUnavailable("GetDimensionsCount"))?;
    let get_dims = api
        .GetDimensions
        .ok_or(OrtError::ApiUnavailable("GetDimensions"))?;

    let mut tensor_info = std::ptr::null();
    // SAFETY: `type_info` is valid and `tensor_info` is an out-parameter.
    crate::error::check_status(unsafe { cast(type_info, &mut tensor_info) })?;
    if tensor_info.is_null() {
        return Err(OrtError::InvalidArgument(
            "model input/output is not a tensor".into(),
        ));
    }

    let mut dtype = onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    // SAFETY: `tensor_info` is borrowed from `type_info` and valid here.
    crate::error::check_status(unsafe { get_type(tensor_info, &mut dtype) })?;
    let dtype = DataType::from_onnx(dtype)?;

    let mut dim_count = 0usize;
    // SAFETY: `tensor_info` is valid and `dim_count` is an out-parameter.
    crate::error::check_status(unsafe { get_dim_count(tensor_info, &mut dim_count) })?;
    let mut shape = vec![0i64; dim_count];
    // SAFETY: `shape` has `dim_count` slots for ORT to fill.
    crate::error::check_status(unsafe { get_dims(tensor_info, shape.as_mut_ptr(), dim_count) })?;

    Ok((dtype, shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_cuda_provider_names() {
        let available = vec!["CUDAExecutionProvider".to_string()];
        assert!(provider_is_available("CUDAExecutionProvider", &available));
        assert!(provider_is_available("CUDA", &available));
    }

    #[test]
    fn fixed_capacity_present_binding_uses_capabilities_or_opt_in() {
        let resolve = |name: &str| resolve_execution_provider(&ep_selection(name));
        assert!(fixed_capacity_present_binding_supported(
            &[resolve("cpu")],
            false
        ));
        assert!(fixed_capacity_present_binding_supported(
            &[resolve("cuda")],
            false
        ));
        assert!(fixed_capacity_present_binding_supported(
            &[resolve("webgpu")],
            false
        ));
        assert!(fixed_capacity_present_binding_supported(
            &[resolve("metal")],
            false
        ));
        assert!(!fixed_capacity_present_binding_supported(
            &[resolve("coreml")],
            false
        ));
        assert!(!fixed_capacity_present_binding_supported(
            &[resolve("some-unknown-ep")],
            false
        ));
        // The operator opt-in overrides the conservative default.
        assert!(fixed_capacity_present_binding_supported(
            &[resolve("some-unknown-ep")],
            true
        ));
    }

    #[test]
    fn resolves_cpu_to_host_defaults() {
        let resolved = resolve_execution_provider(&ep_selection("cpu"));
        assert!(resolved.caps.is_host());
        assert!(
            resolved
                .caps
                .has(capability::FIXED_CAPACITY_PRESENT_BINDING)
        );
        assert!(!resolved.is_strict());
        assert!(matches!(
            resolved.strategy,
            ep_compat::AppendStrategy::HostDefault
        ));
    }

    #[test]
    fn resolves_cuda_to_nvidia_gpu_capabilities() {
        let resolved = resolve_execution_provider(&ep_selection("cuda"));
        assert!(resolved.caps.is_gpu());
        assert!(resolved.caps.is_nvidia());
        assert!(resolved.caps.device_id().is_some());
        for flag in [
            capability::FIXED_CAPACITY_PRESENT_BINDING,
            capability::GRAPH_CAPTURE,
            capability::DEVICE_KV,
            capability::DEVICE_SAMPLING,
        ] {
            assert!(resolved.caps.has(flag), "cuda should advertise {flag}");
        }
        #[cfg(feature = "cuda")]
        assert!(matches!(
            resolved.strategy,
            ep_compat::AppendStrategy::CudaTyped { .. }
        ));
        #[cfg(not(feature = "cuda"))]
        assert!(matches!(
            resolved.strategy,
            ep_compat::AppendStrategy::CudaUnavailable
        ));
    }

    #[test]
    fn convenience_selection_uses_env_name_normalization() {
        let cuda = ep_selection("CUDA");
        assert_eq!(cuda, EpSelection::new("CUDA"));
        assert!(resolve_execution_provider(&cuda).caps.is_nvidia());

        let cpu = ep_selection(" cpu ");
        assert_eq!(cpu, EpSelection::new(" cpu "));
        assert!(resolve_execution_provider(&cpu).caps.is_host());
    }

    #[test]
    fn resolves_unknown_ep_to_named_generic_other_hardware() {
        let resolved = resolve_execution_provider(&ep_selection("openvino"));
        assert_eq!(resolved.caps.hardware, HardwareKind::Other);
        assert!(!resolved.caps.is_gpu());
        assert!(!resolved.caps.is_host());
        assert!(
            !resolved
                .caps
                .has(capability::FIXED_CAPACITY_PRESENT_BINDING)
        );
        assert!(!resolved.is_strict());
        match &resolved.strategy {
            ep_compat::AppendStrategy::NamedGeneric {
                ort_name,
                provider_name,
            } => {
                assert_eq!(ort_name, "openvino");
                assert_eq!(provider_name, "openvinoExecutionProvider");
            }
            other => panic!("expected NamedGeneric, got {other:?}"),
        }
    }

    #[test]
    fn named_generic_forwards_opaque_provider_options() {
        let config = onnx_genai_runtime_config::RuntimeConfig::from_fn(|name| match name {
            "ONNX_GENAI_EP" => Some("openvino".to_owned()),
            "ONNX_GENAI_EP_OPTIONS" => Some("device_type=GPU,precision=FP16".to_owned()),
            _ => None,
        });
        let ExecutionProviderEntry::Builtin(selection) = &config.execution_providers[0] else {
            panic!("expected named provider selection");
        };
        let resolved = resolve_execution_provider(selection);
        assert_eq!(
            named_provider_options(&resolved),
            vec![("device_type", "GPU"), ("precision", "FP16")]
        );
    }

    #[test]
    fn resolves_webgpu_and_coreml_separator_aliases() {
        for name in ["webgpu", "web-gpu", "web_gpu"] {
            let resolved = resolve_execution_provider(&ep_selection(name));
            assert!(
                resolved.caps.is_gpu(),
                "{name} should resolve to WebGPU GPU caps"
            );
            assert!(
                resolved.transitional_webgpu,
                "{name} should be the WebGPU transitional EP"
            );
            assert!(
                resolved.caps.has(capability::DEVICE_KV),
                "{name} should keep WebGPU device-KV"
            );
            assert_eq!(resolved.caps.name, "webgpu");
        }
        for name in ["coreml", "core-ml", "core_ml"] {
            let resolved = resolve_execution_provider(&ep_selection(name));
            assert_eq!(
                resolved.caps.hardware,
                HardwareKind::Npu,
                "{name} should resolve to CoreML"
            );
            assert_eq!(resolved.caps.name, "coreml");
            assert!(matches!(
                resolved.strategy,
                ep_compat::AppendStrategy::NamedGeneric { .. }
            ));
        }
    }

    #[test]
    fn strict_providers_include_cuda_and_plugins() {
        // CUDA and Metal (a plugin library) are strict: load failure must not
        // silently fall back to CPU. Named-generic providers are non-strict.
        let cuda = SessionOptions::with_execution_provider(ep_selection("cuda"));
        assert!(requested_non_cpu_provider(&cuda));
        assert!(requested_strict_provider(&cuda));

        let metal = SessionOptions::with_execution_provider(ep_selection("metal"));
        assert!(requested_non_cpu_provider(&metal));
        assert!(requested_strict_provider(&metal));

        let webgpu = SessionOptions::with_execution_provider(ep_selection("webgpu"));
        assert!(requested_non_cpu_provider(&webgpu));
        assert!(!requested_strict_provider(&webgpu));

        let cpu = SessionOptions::cpu();
        assert!(!requested_non_cpu_provider(&cpu));
        assert!(!requested_strict_provider(&cpu));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn unfused_cuda_attention_uses_math_provider_option() {
        assert_eq!(
            cuda_provider_options("3".to_string(), true, CudaAttentionMode::Unfused),
            vec![
                ("device_id".to_string(), "3".to_string()),
                ("enable_cuda_graph".to_string(), "1".to_string()),
                ("sdpa_kernel".to_string(), "16".to_string()),
            ]
        );
        assert_eq!(
            cuda_provider_options("0".to_string(), false, CudaAttentionMode::Default),
            vec![("device_id".to_string(), "0".to_string())]
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn unavailable_cuda_error_is_actionable() {
        let error = cuda_provider_unavailable_error(&["CPUExecutionProvider".to_string()]);
        let message = error.to_string();
        assert!(message.contains("CUDAExecutionProvider was requested"));
        assert!(message.contains(cuda_provider_library_name()));
        assert!(message.contains(cuda_library_search_path()));
        assert!(message.contains("ONNX_GENAI_EP=cpu"));
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn auto_default_providers_are_macos_only() {
        // MLX/Metal auto-selection is gated to macOS; every other platform keeps
        // the plain CPU default regardless of environment.
        assert!(super::auto_default_execution_providers().is_none());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_request_requires_compile_time_feature() {
        let resolved = resolve_execution_provider(&ep_selection("cuda"));
        let error = append_execution_provider(
            &Environment::new("cuda-feature-test").expect("environment"),
            std::ptr::null_mut(),
            &resolved,
            false,
            CudaAttentionMode::Default,
            &[],
        )
        .expect_err("CUDA must be rejected without the cargo feature");
        assert!(
            error
                .to_string()
                .contains("CUDA support not compiled in; rebuild with --features cuda")
        );
    }
}
