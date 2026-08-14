use std::ffi::CString;

use crate::{OrtError, Result};

use super::CudaAttentionMode;
use super::providers::provider_is_available;

#[cfg(feature = "cuda")]
pub(super) fn append_cuda_execution_provider(
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    device_id: i32,
    graph_capture: bool,
    attention_mode: &CudaAttentionMode,
    user_compute_stream: Option<usize>,
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
        if let Some(stream) = user_compute_stream {
            // The stream is a pointer-valued provider option: the typed entry
            // point is what makes ORT record `has_user_compute_stream` and adopt
            // the caller's stream, which is how upstream ORT GenAI configures
            // its CUDA sessions. The string form does not establish that flag,
            // and a session that half-adopts the stream aborts ORT-managed graph
            // capture with `operation not permitted when stream is capturing`.
            let update_value =
                api.UpdateCUDAProviderOptionsWithValue
                    .ok_or(OrtError::ApiUnavailable(
                        "UpdateCUDAProviderOptionsWithValue",
                    ))?;
            let key = CString::new("user_compute_stream").map_err(|_| {
                OrtError::InvalidArgument("CUDA provider option key contains NUL".into())
            })?;
            // SAFETY: the options handle and key are valid for the call, and
            // `stream` is a live `cudaStream_t` that outlives every session.
            crate::error::check_status(unsafe {
                update_value(cuda_options, key.as_ptr(), stream as *mut std::ffi::c_void)
            })?;
        }
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
                shared_compute_stream = user_compute_stream.is_some(),
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
pub(super) fn cuda_provider_options(
    device_id: String,
    graph_capture: bool,
    attention_mode: &CudaAttentionMode,
) -> Vec<(String, String)> {
    let mut options = vec![("device_id".to_string(), device_id)];
    if graph_capture {
        options.push(("enable_cuda_graph".to_string(), "1".to_string()));
    }
    if attention_mode == &CudaAttentionMode::Unfused {
        // ORT AttentionBackend::MATH is bit 16. A positive sdpa_kernel value is
        // an explicit backend mask, so all optimized paths are disabled without
        // process-global ORT_DISABLE_* environment state.
        options.push(("sdpa_kernel".to_string(), "16".to_string()));
    }
    options
}

#[cfg(feature = "cuda")]
pub(super) fn cuda_provider_unavailable_error(available: &[String]) -> OrtError {
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
pub(super) fn cuda_provider_library_name() -> &'static str {
    "onnxruntime_providers_cuda.dll"
}

#[cfg(all(feature = "cuda", target_os = "macos"))]
pub(super) fn cuda_provider_library_name() -> &'static str {
    "libonnxruntime_providers_cuda.dylib"
}

#[cfg(all(feature = "cuda", not(any(target_os = "windows", target_os = "macos"))))]
pub(super) fn cuda_provider_library_name() -> &'static str {
    "libonnxruntime_providers_cuda.so"
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
pub(super) fn cuda_library_search_path() -> &'static str {
    "PATH"
}

#[cfg(all(feature = "cuda", target_os = "macos"))]
pub(super) fn cuda_library_search_path() -> &'static str {
    "DYLD_LIBRARY_PATH"
}

#[cfg(all(feature = "cuda", not(any(target_os = "windows", target_os = "macos"))))]
pub(super) fn cuda_library_search_path() -> &'static str {
    "LD_LIBRARY_PATH"
}
