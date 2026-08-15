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
        let provider_options = cuda_provider_options(
            device_id,
            graph_capture,
            attention_mode,
            user_compute_stream,
        );
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
            // Second of the two mechanisms, kept because it does not depend on
            // ONNX Runtime parsing a pointer out of text: it assigns both
            // fields directly (`provider_bridge_ort.cc`:
            // `has_user_compute_stream = 1; user_compute_stream = value;`).
            // It is written last so it is the final writer, but correctness no
            // longer rests on that ordering - the string map above carries both
            // keys and is self-sufficient on its own. Nothing here is trusted
            // either way; the readback below is what decides.
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
        // The invariant, checked rather than asserted in a comment: ONNX
        // Runtime must believe it has a user compute stream exactly when we
        // gave it one, so `Session::user_compute_stream` can never name a
        // stream the provider is not computing on.
        //
        // This deliberately verifies the *net* recorded state rather than any
        // one call, because no single call is the invariant. Both mechanisms
        // above independently suffice on ORT 1.28 - `UpdateProviderOptions`
        // parses `user_compute_stream` from the map, copies it because the
        // `has_user_compute_stream` key is present, then derives the flag from
        // `user_compute_stream != nullptr` - so ordering between them is
        // discipline, not correctness, and dropping either one would not be a
        // regression on its own. What must never happen is the options and the
        // getter disagreeing. Reading the options back is how that is settled,
        // and it fires on the cases that matter: a stream that was never
        // configured, and a later string update whose map omits the stream keys
        // and therefore clears the flag.
        //
        // The end-to-end counterpart is
        // `session::tests::cuda_session_computes_on_the_stream_it_reports`,
        // which proves the same agreement by execution: it queues poison, a
        // delay, and then the real input on the reported stream and requires the
        // model to have read the input. Handing ONNX Runtime a decoy stream
        // makes that test return NaN.
        if let Some(stream) = user_compute_stream
            && !records_user_compute_stream(cuda_options, stream)?
        {
            return Err(OrtError::SessionCreation(
                "ONNX Runtime did not record the shared CUDA compute stream (it must report \
                 both has_user_compute_stream=1 and the exact stream address). Every string \
                 update reparses the whole option set and recomputes \
                 has_user_compute_stream, so any update that omits the stream keys clears \
                 it; keep both keys in every provider-option map and leave the typed \
                 UpdateCUDAProviderOptionsWithValue as the last writer."
                    .into(),
            ));
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
    user_compute_stream: Option<usize>,
) -> Vec<(String, String)> {
    let mut options = vec![("device_id".to_string(), device_id)];
    if let Some(stream) = user_compute_stream {
        // Both keys, together, deliberately.
        //
        // ONNX Runtime's string update rebuilds the provider options from a
        // freshly parsed info: it *always* overwrites `has_user_compute_stream`
        // from that parse, and overwrites the pointer only when the
        // `has_user_compute_stream` key is present in the map
        // (`cuda_provider_factory.cc`, `UpdateProviderOptions`). So a later
        // string update that omits these keys would clear the flag while
        // leaving the pointer set - ORT would ignore the stream while this
        // process still believed it was shared. Carrying both keys means any
        // update built from these options preserves the whole configuration.
        options.push(("has_user_compute_stream".to_string(), "1".to_string()));
        options.push(("user_compute_stream".to_string(), stream.to_string()));
    }
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

/// Whether ORT records these provider options as carrying a user compute stream.
///
/// Read back through `GetCUDAProviderOptionsAsString`, which serialises the
/// live option struct, so this reflects what ORT will actually act on rather
/// than what this code believes it set.
#[cfg(feature = "cuda")]
fn records_user_compute_stream(
    cuda_options: *mut onnx_genai_ort_sys::OrtCUDAProviderOptionsV2,
    expected: usize,
) -> Result<bool> {
    let api = crate::error::api()?;
    let Some(as_string) = api.GetCUDAProviderOptionsAsString else {
        // An ORT without the accessor cannot be checked; the typed update is
        // still the last write, which is the invariant this would confirm.
        return Ok(true);
    };
    let allocator = crate::allocator::Allocator::default_cpu()?;
    let mut text: *mut std::os::raw::c_char = std::ptr::null_mut();
    // SAFETY: `cuda_options` is live, the allocator outlives the call, and
    // `text` is an out-parameter owned by that allocator on success.
    crate::error::check_status(unsafe { as_string(cuda_options, allocator.as_ptr(), &mut text) })?;
    if text.is_null() {
        return Ok(false);
    }
    // SAFETY: ORT returned a NUL-terminated string allocated by `allocator`.
    let serialized = unsafe { std::ffi::CStr::from_ptr(text) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: the buffer came from this allocator and is freed exactly once
    // through that allocator's own `Free`.
    unsafe {
        if let Some(free) = (*allocator.as_ptr()).Free {
            free(allocator.as_ptr(), text.cast());
        }
    }
    // Both halves matter: the flag alone can be set while the pointer is null,
    // and the pointer alone is ignored. ORT serialises the pointer as a decimal
    // address (`cuda_execution_provider_info.cc`).
    let entries: Vec<&str> = serialized.split(';').map(str::trim).collect();
    let flagged = entries.contains(&"has_user_compute_stream=1");
    let addressed = entries
        .iter()
        .filter_map(|entry| entry.strip_prefix("user_compute_stream="))
        .any(|value| value.parse::<usize>() == Ok(expected));
    Ok(flagged && addressed)
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
