use std::ffi::CString;

use crate::{Environment, OrtError, Result};

use super::CudaAttentionMode;
#[cfg(feature = "cuda")]
use super::cuda::append_cuda_execution_provider;
use super::ep_compat::{self, ResolvedEp};
use super::options::{SessionOptions, available_execution_providers};
use super::plugin::append_plugin_execution_provider;

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

/// Append every requested provider, returning the shared CUDA stream that a
/// provider actually adopted.
///
/// The return value is the whole point: a caller that reports "this session
/// computes on stream X" must report the stream the execution provider was
/// really given, not the one the options happened to carry. Those differ
/// whenever the stream belongs to another device, or the session ends up on a
/// provider that takes no stream at all.
pub(super) fn append_execution_providers(
    env: &Environment,
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    options: &SessionOptions,
) -> Result<AdoptedStream> {
    let available = available_execution_providers().unwrap_or_else(|err| {
        tracing::warn!("Could not query available ORT execution providers: {err}");
        Vec::new()
    });
    #[cfg(feature = "cuda")]
    let mut adopted: AdoptedStream = None;
    #[cfg(not(feature = "cuda"))]
    let adopted: AdoptedStream = ();
    for provider in &options.execution_providers {
        #[cfg(feature = "cuda")]
        let stream = shared_stream_for(options, provider);
        #[cfg(not(feature = "cuda"))]
        let stream: Option<std::convert::Infallible> = None;
        append_execution_provider(
            env,
            session_options,
            provider,
            options.graph_capture,
            &options.cuda_attention_mode,
            {
                #[cfg(feature = "cuda")]
                {
                    stream.as_ref().map(|stream| stream.handle())
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = &stream;
                    None
                }
            },
            &available,
        )?;
        #[cfg(feature = "cuda")]
        if stream.is_some() {
            adopted = stream;
        }
    }
    Ok(adopted)
}

/// The shared CUDA stream, if any, that a session using these options adopts.
#[cfg(feature = "cuda")]
pub(super) type AdoptedStream = Option<std::sync::Arc<crate::cuda_rt::CudaComputeStream>>;

/// Without CUDA no provider can adopt a stream, so there is nothing to report.
#[cfg(not(feature = "cuda"))]
pub(super) type AdoptedStream = ();

pub(super) fn append_execution_provider(
    env: &Environment,
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    provider: &ResolvedEp,
    graph_capture: bool,
    cuda_attention_mode: &CudaAttentionMode,
    cuda_user_compute_stream: Option<usize>,
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
            cuda_user_compute_stream,
            available,
        ),
        #[cfg(not(feature = "cuda"))]
        AppendStrategy::CudaUnavailable => {
            let _ = (
                session_options,
                graph_capture,
                cuda_attention_mode,
                cuda_user_compute_stream,
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

/// The shared CUDA stream `provider` adopts, if any.
///
/// A stream is only valid on the device it was created on, and a provider that
/// is not a typed CUDA provider takes no stream at all. `execution_providers`
/// is public and metadata placement rewrites it, so the device can change after
/// a stream was attached. Deciding here, per provider, is what keeps the stream
/// the execution provider receives and the stream the session reports the same
/// value by construction.
#[cfg(feature = "cuda")]
fn shared_stream_for(options: &SessionOptions, provider: &ResolvedEp) -> AdoptedStream {
    let stream = options.cuda_user_compute_stream.as_ref()?;
    let ep_compat::AppendStrategy::CudaTyped { device_id } = &provider.strategy else {
        // A host or plugin provider issues no work on this stream.
        return None;
    };
    if *device_id == stream.device_id() {
        return Some(std::sync::Arc::clone(stream));
    }
    tracing::warn!(
        stream_device = stream.device_id(),
        provider_device = *device_id,
        "ignoring a shared CUDA compute stream created for a different device; this session keeps \
         the stream ONNX Runtime gives it"
    );
    None
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

#[cfg(test)]
mod shared_stream_device_guard {
    use super::*;
    use crate::session::ep_selection;

    /// A typed CUDA provider for `device_id`, built the way the resolver builds
    /// one, so the test exercises the real CUDA-vs-CUDA branch rather than a
    /// selection that resolves to no device at all.
    #[cfg(feature = "cuda")]
    fn cuda_provider(device_id: i32) -> ResolvedEp {
        let mut provider = ep_compat::resolve_execution_provider(&ep_selection("cuda"));
        provider.strategy = ep_compat::AppendStrategy::CudaTyped { device_id };
        provider
    }

    /// The stream reaches a provider on its own device.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn the_owning_device_receives_the_stream() {
        let mut options = SessionOptions::with_execution_provider(ep_selection("cuda"));
        options.share_cuda_compute_stream();
        let Some(stream) = options.cuda_user_compute_stream.clone() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let provider = cuda_provider(stream.device_id());
        assert_eq!(
            shared_stream_for(&options, &provider).map(|s| s.handle()),
            Some(stream.handle()),
        );
    }

    /// A *CUDA* provider on a different device must not receive it.
    ///
    /// This is the branch that matters: both sides are typed CUDA providers
    /// with a real device id, which a selection string that fails to resolve
    /// would never reach.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn another_cuda_device_does_not_receive_the_stream() {
        let mut options = SessionOptions::with_execution_provider(ep_selection("cuda"));
        options.share_cuda_compute_stream();
        let Some(stream) = options.cuda_user_compute_stream.clone() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let other = stream.device_id() + 1;
        let provider = cuda_provider(other);
        assert!(
            shared_stream_for(&options, &provider).is_none(),
            "a stream from device {} must not be handed to a CUDA provider on device {other}",
            stream.device_id(),
        );
    }

    /// A provider that is not a typed CUDA provider takes no stream, so a
    /// session that falls back to the host reports none.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_host_provider_does_not_receive_the_stream() {
        let mut options = SessionOptions::with_execution_provider(ep_selection("cuda"));
        options.share_cuda_compute_stream();
        if options.cuda_user_compute_stream.is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let cpu = ep_compat::resolve_execution_provider(&ep_selection("cpu"));
        assert!(shared_stream_for(&options, &cpu).is_none());
    }

    /// Options with no shared stream are unaffected, on every build.
    #[test]
    fn no_shared_stream_means_no_handle() {
        let options = SessionOptions::with_execution_provider(ep_selection("cpu"));
        let cpu = ep_compat::resolve_execution_provider(&ep_selection("cpu"));
        #[cfg(feature = "cuda")]
        assert!(shared_stream_for(&options, &cpu).is_none());
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (&options, &cpu);
        }
    }
}
