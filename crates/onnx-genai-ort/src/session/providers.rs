use std::ffi::CString;

use crate::{Environment, OrtError, Result};

use super::CudaAttentionMode;
#[cfg(feature = "cuda")]
use super::cuda::append_cuda_execution_provider;
use super::ep_compat::{self, ResolvedEp};
use super::options::{SessionOptions, available_execution_providers};
use super::plugin::append_plugin_execution_provider;

#[derive(Debug)]
pub(super) struct ExecutionProviderAppendError {
    pub provider_index: usize,
    pub provider_name: String,
    pub source: OrtError,
}

/// Apply WebGPU EP provider options via session config entries.
///
/// The WebGPU EP reads these from the merged `ConfigOptions` (see ORT
/// `webgpu_provider_factory.cc`), keyed by the full `ep.webgpuexecutionprovider.*`
/// names. `AddSessionConfigEntry` is the EP-agnostic way to set them. No-ops
/// unless a WebGPU EP is selected.
pub(super) fn apply_webgpu_provider_options(
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

pub(super) fn add_session_config_entry(
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

pub(super) fn append_execution_providers(
    env: &Environment,
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    options: &SessionOptions,
) -> std::result::Result<(), ExecutionProviderAppendError> {
    let available = available_execution_providers().unwrap_or_else(|err| {
        tracing::warn!("Could not query available ORT execution providers: {err}");
        Vec::new()
    });
    for (provider_index, provider) in options.execution_providers.iter().enumerate() {
        append_execution_provider(
            env,
            session_options,
            provider,
            options.graph_capture,
            options.forces_cuda_unified_stream(),
            &options.cuda_attention_mode,
            &available,
        )
        .map_err(|source| ExecutionProviderAppendError {
            provider_index,
            provider_name: provider.selection.name.clone(),
            source,
        })?;
    }
    Ok(())
}

pub(super) fn append_execution_provider(
    env: &Environment,
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    provider: &ResolvedEp,
    graph_capture: bool,
    force_cuda_unified_stream: bool,
    cuda_attention_mode: &CudaAttentionMode,
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
            force_cuda_unified_stream,
            cuda_attention_mode,
            available,
        ),
        #[cfg(not(feature = "cuda"))]
        AppendStrategy::CudaUnavailable => {
            let _ = (
                session_options,
                graph_capture,
                force_cuda_unified_stream,
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
        AppendStrategy::UnsupportedName { name } => Err(OrtError::InvalidArgument(format!(
            "Unrecognized ONNX_GENAI_EP value '{name}'. Known values are: {}. \
             To use an EP outside this table, configure it as an explicit plugin \
             with ONNX_GENAI_EP=plugin:<library>|name=<registration>|device=<CPU|GPU|NPU>|opt.<key>=<value>.",
            ep_compat::known_execution_provider_values()
        ))),
        AppendStrategy::InvalidConfiguration { message } => {
            Err(OrtError::InvalidArgument(message.clone()))
        }
    }
}

pub(super) fn named_provider_options(provider: &ResolvedEp) -> Vec<(&str, &str)> {
    provider
        .selection
        .options
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect()
}

/// Map a portable hardware-device class string to ORT's generic
/// `OrtHardwareDeviceType`. Accepts `CPU`, `GPU`, and `NPU` case-insensitively.
/// This is intentionally provider-agnostic: it never matches a vendor's device
/// name, only ORT's own hardware-class enum.
pub(super) fn parse_hardware_device_type(
    value: &str,
) -> Option<onnx_genai_ort_sys::OrtHardwareDeviceType> {
    match value.trim().to_ascii_uppercase().as_str() {
        "CPU" => Some(onnx_genai_ort_sys::OrtHardwareDeviceType_CPU),
        "GPU" => Some(onnx_genai_ort_sys::OrtHardwareDeviceType_GPU),
        "NPU" => Some(onnx_genai_ort_sys::OrtHardwareDeviceType_NPU),
        _ => None,
    }
}

fn append_named_execution_provider(
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    api_name: &str,
    provider_name: &str,
    provider_options: &[(&str, &str)],
    available: &[String],
) -> Result<()> {
    if !provider_is_available(provider_name, available) {
        return Err(named_provider_unavailable_error(
            api_name,
            provider_name,
            available,
        ));
    }

    let api = crate::error::api()?;
    let append = api
        .SessionOptionsAppendExecutionProvider
        .ok_or(OrtError::ApiUnavailable(
            "SessionOptionsAppendExecutionProvider",
        ))?;
    let api_name_c = CString::new(api_name)
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
            api_name_c.as_ptr(),
            option_key_ptrs.as_ptr(),
            option_value_ptrs.as_ptr(),
            provider_options.len(),
        )
    }) {
        Ok(()) => {
            tracing::info!("Enabled ONNX Runtime execution provider {provider_name}");
            Ok(())
        }
        Err(err) => Err(OrtError::SessionCreation(format!(
            "failed to initialize requested ONNX Runtime execution provider {provider_name} \
             ({api_name}): {err}. Available providers: {available:?}. \
             To intentionally run on CPU, request it explicitly with ONNX_GENAI_EP=cpu; \
             to opt into a visible CPU retry for this request, set ONNX_GENAI_EP_FALLBACK=1"
        ))),
    }
}

fn named_provider_unavailable_error(
    api_name: &str,
    provider_name: &str,
    available: &[String],
) -> OrtError {
    OrtError::SessionCreation(format!(
        "ONNX Runtime execution provider {provider_name} ({api_name}) was requested, \
         but the linked ONNX Runtime build does not report it. Available providers: {available:?}. \
         Install or load an ONNX Runtime build that includes {provider_name}; \
         to intentionally run on CPU, request it explicitly with ONNX_GENAI_EP=cpu; \
         to opt into a visible CPU retry for this request, set ONNX_GENAI_EP_FALLBACK=1"
    ))
}

pub(super) fn provider_is_available(provider_name: &str, available: &[String]) -> bool {
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
