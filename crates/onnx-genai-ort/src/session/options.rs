use std::ffi::CStr;

use onnx_genai_runtime_config::{EpSelection, runtime_config};

use crate::{OrtError, Result};

use super::CudaAttentionMode;
use super::env_config::{
    cuda_attention_mode_from_runtime_config, execution_providers_from_env,
    webgpu_disable_validation_from_env,
};
use super::ep_compat::{EpCapabilities, ResolvedEp, capability, resolve_execution_provider};

/// Convenience constructor for an [`EpSelection`] from a bare provider name.
///
/// The runtime core stays EP-agnostic: name resolution happens in
/// the `ep_compat` module. This helper only saves callers from importing `BTreeMap`.
#[must_use]
pub fn ep_selection(name: impl Into<String>) -> EpSelection {
    EpSelection::new(name.into())
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
    /// Defaults from the typed runtime configuration registry populated by
    /// `ONNX_GENAI_CUDA_ATTENTION` (`auto`, `fused`, or `unfused`).
    /// `unfused` is a generic correctness workaround for graphs that encounter
    /// an ONNX Runtime optimized-attention kernel defect; it is never selected
    /// from model identity.
    pub cuda_attention_mode: CudaAttentionMode,
    /// Generic ORT session configuration entries applied through
    /// `AddSessionConfigEntry` before provider append/session creation.
    pub session_config_entries: Vec<(String, String)>,
    /// Whether the non-CPU execution provider was auto-selected for this platform
    /// (e.g. the macOS MLX/Metal default) rather than explicitly requested. An
    /// auto-selected provider must fall back to CPU on load failure, even if the
    /// provider would otherwise be strict.
    pub auto_selected: bool,
}

/// ORT session config key that makes a session allocate through allocators
/// registered on the environment instead of creating its own.
pub const USE_ENV_ALLOCATORS: &str = "session.use_env_allocators";

impl SessionOptions {
    /// Route this session's allocations through allocators registered on the
    /// environment.
    ///
    /// Registering an allocator is not enough on its own: without this entry
    /// ORT silently builds the session its own default allocator, so a governed
    /// allocator would be installed, report zero live bytes forever, and look
    /// like the model simply did not allocate.
    pub fn use_env_allocators(&mut self) -> &mut Self {
        if !self
            .session_config_entries
            .iter()
            .any(|(key, _)| key == USE_ENV_ALLOCATORS)
        {
            self.session_config_entries
                .push((USE_ENV_ALLOCATORS.to_string(), "1".to_string()));
        }
        self
    }
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
        // Governance is on by default. Without this entry a session silently
        // builds its own allocator, so a registered governor would report zero
        // bytes forever and be indistinguishable from a model that does not
        // allocate — the budget would be decorative for every default session.
        //
        // Costs nothing when no allocator is registered: ORT then falls back to
        // its own, exactly as before.
        options.use_env_allocators();
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
pub(super) fn auto_default_execution_providers() -> Option<Vec<ResolvedEp>> {
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
        Some(vec![resolve_execution_provider(&ep_selection("metal"))])
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

impl SessionOptions {
    pub(super) fn cpu() -> Self {
        Self {
            execution_providers: vec![resolve_execution_provider(&ep_selection("cpu"))],
            optimization_level: 99,
            intra_op_num_threads: 0, // ORT decides
            inter_op_num_threads: 0,
            graph_capture: false,
            webgpu_disable_validation: false,
            cuda_attention_mode: cuda_attention_mode_from_runtime_config(),
            session_config_entries: runtime_config().session_config_entries.clone(),
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
    pub(super) fn selects_webgpu(&self) -> bool {
        self.execution_providers
            .iter()
            .any(|ep| ep.transitional_webgpu)
    }

    /// Whether a Qualcomm QNN execution provider is selected.
    pub(super) fn selects_qnn(&self) -> bool {
        self.execution_providers
            .iter()
            .any(|ep| ep.caps.name.eq_ignore_ascii_case("qnn"))
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
        if self.selects_qnn() {
            self.apply_qnn_session_config_defaults();
        }
        self.graph_capture =
            self.primary_caps().has(capability::GRAPH_CAPTURE) && self.primary_graph_capture_env();
    }

    fn apply_qnn_session_config_defaults(&mut self) {
        let config = runtime_config();
        if config.qnn_disable_cpu_fallback {
            self.push_session_config_if_absent("session.disable_cpu_ep_fallback", "1");
        }
        if config.qnn_context_enable {
            self.push_session_config_if_absent("ep.context_enable", "1");
        }
        if let Some(path) = &config.qnn_context_file {
            self.push_session_config_if_absent("ep.context_file_path", &path.display().to_string());
        }
        if let Some(value) = &config.qnn_context_embed {
            self.push_session_config_if_absent("ep.context_embed_mode", value);
        }
    }

    fn push_session_config_if_absent(&mut self, key: &str, value: &str) {
        if !self
            .session_config_entries
            .iter()
            .any(|(existing, _)| existing.eq_ignore_ascii_case(key))
        {
            self.session_config_entries
                .push((key.to_string(), value.to_string()));
        }
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

#[cfg(test)]
mod use_env_allocator_tests {
    use super::*;

    /// The session option is the half that makes a registered allocator
    /// observable; without it ORT builds its own and governs nothing.
    #[test]
    fn use_env_allocators_sets_the_config_entry_once() {
        let mut options = SessionOptions::cpu();
        options.use_env_allocators().use_env_allocators();
        let entries: Vec<_> = options
            .session_config_entries
            .iter()
            .filter(|(key, _)| key == USE_ENV_ALLOCATORS)
            .collect();
        assert_eq!(entries.len(), 1, "the entry must not be duplicated");
        assert_eq!(entries[0].1, "1");
    }
}
