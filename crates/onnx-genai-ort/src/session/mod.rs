//! ORT Session — represents a loaded model.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::NonNull;

pub use onnx_genai_runtime_config::CudaAttentionMode;
#[cfg(test)]
use onnx_genai_runtime_config::{EpSelection, ExecutionProviderEntry};

use crate::{Allocator, DataType, Environment, IoBinding, MemoryInfo, OrtError, Result, Value};

#[cfg(feature = "cuda")]
mod cuda;
mod env_config;
pub mod ep_compat;
mod options;
mod plugin;
mod providers;

pub use ep_compat::{
    EpCapabilities, HardwareKind, ResolvedEp, capability, resolve_execution_provider,
    selectable_execution_providers,
};
pub use options::{
    SessionOptions, USE_ENV_ALLOCATORS, available_execution_providers, ep_selection,
};

use env_config::{
    cuda_device_id_from_env, device_kv_enabled_from_env, effective_intra_op_threads,
    fixed_capacity_present_binding_supported, is_textproto_path, requested_non_cpu_provider,
    requested_strict_provider, shared_kv_present_binding_opt_in_from_env,
};
use providers::{
    add_session_config_entry, append_execution_providers, apply_webgpu_provider_options,
};

#[cfg(all(test, feature = "cuda"))]
use cuda::{
    cuda_library_search_path, cuda_provider_library_name, cuda_provider_options,
    cuda_provider_unavailable_error,
};
#[cfg(test)]
use options::auto_default_execution_providers;
#[cfg(test)]
use providers::{append_execution_provider, named_provider_options, provider_is_available};

/// Tensor metadata for a model input or output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DataType,
    /// ORT uses negative dimensions for dynamic axes.
    pub shape: Vec<i64>,
}

/// Backend-neutral view of a graph's declared input/output tensor metadata.
///
/// An ORT [`Session`] and a native (nxrt) component expose the exact same
/// name/dtype/shape records, so the decode-time contract (I/O port roles, paged
/// KV layout, and fixed-state budget) can be resolved from either backend
/// through this seam. This is what lets a pipeline whose decoder carries
/// native-only operators — a QMoE artifact that ORT's op type-checker rejects at
/// load, for example — resolve its decode contract from the native loader
/// instead of a redundant ORT session that would never execute.
pub trait GraphIo {
    /// Declared input tensor metadata.
    fn inputs(&self) -> &[TensorInfo];
    /// Declared output tensor metadata.
    fn outputs(&self) -> &[TensorInfo];
    /// Declared input names, in graph order.
    fn input_names(&self) -> &[String];
    /// Declared output names, in graph order.
    fn output_names(&self) -> &[String];
}

/// A standalone, session-free [`GraphIo`] carrying only a graph's declared
/// input/output tensor metadata.
///
/// Built from a native component (or straight from an ONNX graph's value-info)
/// so decode resolution can run without instantiating an ORT session for a
/// component the ORT backend will never execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphIoMetadata {
    input_names: Vec<String>,
    output_names: Vec<String>,
    inputs: Vec<TensorInfo>,
    outputs: Vec<TensorInfo>,
}

impl GraphIoMetadata {
    /// Build from declared input and output tensor metadata; names are derived
    /// from each record in graph order.
    pub fn new(inputs: Vec<TensorInfo>, outputs: Vec<TensorInfo>) -> Self {
        let input_names = inputs.iter().map(|info| info.name.clone()).collect();
        let output_names = outputs.iter().map(|info| info.name.clone()).collect();
        Self {
            input_names,
            output_names,
            inputs,
            outputs,
        }
    }
}

impl GraphIo for GraphIoMetadata {
    fn inputs(&self) -> &[TensorInfo] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorInfo] {
        &self.outputs
    }

    fn input_names(&self) -> &[String] {
        &self.input_names
    }

    fn output_names(&self) -> &[String] {
        &self.output_names
    }
}

impl GraphIo for Session {
    fn inputs(&self) -> &[TensorInfo] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorInfo] {
        &self.outputs
    }

    fn input_names(&self) -> &[String] {
        &self.input_names
    }

    fn output_names(&self) -> &[String] {
        &self.output_names
    }
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
    /// The caller-owned CUDA stream this session was told to compute on, if any.
    /// Kept alive for the session's lifetime so the stream it computes on
    /// cannot be destroyed while ORT still holds it.
    #[cfg(feature = "cuda")]
    user_compute_stream: Option<std::sync::Arc<crate::cuda_rt::CudaComputeStream>>,
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
        let create_session = |opts: &SessionOptions| -> Result<(
            *mut onnx_genai_ort_sys::OrtSession,
            providers::AdoptedStream,
        )> {
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
            Ok((
                ptr,
                providers::clone_adopted(&session_options.adopted_stream),
            ))
        };

        // Auto-selected providers (e.g. the macOS MLX default) always fall back
        // to CPU; explicitly requested strict providers (CUDA and plugin EPs)
        // fail rather than silently changing the requested device.
        let allow_cpu_fallback = options.auto_selected
            || (requested_non_cpu_provider(&options) && !requested_strict_provider(&options));

        // Bound on every build so the fallback arms stay symmetric, but only a
        // CUDA build has a field to put it in.
        #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
        let (ptr, effective_providers, adopted_stream) = match create_session(&options) {
            Ok((ptr, stream)) => (ptr, options.execution_providers.clone(), stream),
            Err(err) if allow_cpu_fallback => {
                tracing::warn!(
                    "ORT session creation failed with requested execution provider(s): {err}; retrying with CPU"
                );
                // The CPU options carry no stream, so a session that fell back
                // reports none, which is exactly what it is using.
                let cpu_options = SessionOptions::cpu();
                let (ptr, stream) = create_session(&cpu_options)?;
                (ptr, cpu_options.execution_providers, stream)
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
            #[cfg(feature = "cuda")]
            user_compute_stream: adopted_stream,
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

    /// The caller-owned CUDA stream this session computes on, when one was
    /// shared with it. Work issued to the same stream needs no device barrier
    /// to be ordered against this session's runs.
    ///
    /// This is the stream an execution provider *adopted*, not the one the
    /// options happened to carry: it is the value `append_execution_providers`
    /// returned, so the provider and this getter cannot disagree. A session
    /// whose provider took no stream - a host provider, or a CUDA provider on
    /// a different device than the stream - reports `None`, because a caller
    /// that ordered work against a stream this session does not use would race
    /// instead of synchronising.
    #[must_use]
    pub fn user_compute_stream(&self) -> Option<usize> {
        #[cfg(feature = "cuda")]
        {
            self.user_compute_stream
                .as_ref()
                .map(|stream| stream.handle())
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
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
    /// The shared CUDA stream a provider actually adopted, so the session can
    /// report exactly the stream it computes on.
    adopted_stream: providers::AdoptedStream,
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
            #[cfg(feature = "cuda")]
            adopted_stream: None,
            #[cfg(not(feature = "cuda"))]
            adopted_stream: (),
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
        let effective_intra_op = effective_intra_op_threads(options);
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

        for (key, value) in &options.session_config_entries {
            add_session_config_entry(this.ptr.as_ptr(), key, value)?;
        }

        // Assign into the existing value rather than rebuilding it: `ptr` is a
        // `NonNull`, which is `Copy`, so struct update syntax would copy the
        // handle into a second `RawSessionOptions` while this one still drops -
        // releasing the same `OrtSessionOptions` twice.
        let mut this = this;
        this.adopted_stream = append_execution_providers(env, this.ptr.as_ptr(), options)?;
        apply_webgpu_provider_options(this.ptr.as_ptr(), options)?;

        Ok(this)
    }

    fn as_ptr(&self) -> *const onnx_genai_ort_sys::OrtSessionOptions {
        self.ptr.as_ptr()
    }
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
mod tests;
