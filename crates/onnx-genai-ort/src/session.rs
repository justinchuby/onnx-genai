//! ORT Session — represents a loaded model.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::NonNull;

use crate::{Allocator, DataType, Environment, IoBinding, MemoryInfo, OrtError, Result, Value};

/// Execution provider selection.
#[derive(Debug, Clone)]
pub enum ExecutionProvider {
    Cpu,
    WebGpu,
    Cuda { device_id: i32 },
    DirectML { device_id: i32 },
    CoreML,
    Qnn,
    OpenVINO,
}

/// Session configuration options.
#[derive(Debug, Clone)]
pub struct SessionOptions {
    /// Execution providers in priority order.
    pub execution_providers: Vec<ExecutionProvider>,
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
}

impl Default for SessionOptions {
    fn default() -> Self {
        let mut options = Self::cpu();
        if let Some(execution_providers) = execution_providers_from_env() {
            options.execution_providers = execution_providers;
        }
        options.apply_provider_defaults();
        options
    }
}

impl SessionOptions {
    fn cpu() -> Self {
        Self {
            execution_providers: vec![ExecutionProvider::Cpu],
            optimization_level: 99,
            intra_op_num_threads: 0, // ORT decides
            inter_op_num_threads: 0,
            graph_capture: false,
            webgpu_disable_validation: false,
        }
    }

    /// Create default session options with a single explicit execution provider.
    pub fn with_execution_provider(provider: ExecutionProvider) -> Self {
        let mut options = Self {
            execution_providers: vec![provider],
            ..Self::cpu()
        };
        options.apply_provider_defaults();
        options
    }

    fn selects_webgpu(&self) -> bool {
        self.execution_providers
            .iter()
            .any(|provider| matches!(provider, ExecutionProvider::WebGpu))
    }

    fn selects_cuda(&self) -> bool {
        self.execution_providers
            .iter()
            .any(|provider| matches!(provider, ExecutionProvider::Cuda { .. }))
    }

    /// Apply provider performance defaults. WebGPU validation is disabled (pure
    /// overhead reduction), while graph capture follows the selected EP's
    /// provider-specific environment flag and remains off by default.
    fn apply_provider_defaults(&mut self) {
        if self.selects_webgpu() {
            self.webgpu_disable_validation = webgpu_disable_validation_from_env();
        }
        self.graph_capture = (self.selects_webgpu()
            && env_flag_enabled("ONNX_GENAI_WEBGPU_GRAPH_CAPTURE"))
            || (self.selects_cuda() && env_flag_enabled("ONNX_GENAI_CUDA_GRAPH"));
    }

    /// Set the number of ORT intra-op threads.
    ///
    /// Values less than or equal to zero leave thread selection to ORT.
    pub fn with_intra_op_threads(mut self, threads: i32) -> Self {
        self.intra_op_num_threads = threads;
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
    execution_providers: Vec<ExecutionProvider>,
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

        let session_options = RawSessionOptions::new(&options)?;
        let path_c = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| OrtError::InvalidArgument("model path contains NUL".into()))?;
        let mut ptr = std::ptr::null_mut();
        let api = crate::error::api()?;
        let create = api
            .CreateSession
            .ok_or(OrtError::ApiUnavailable("CreateSession"))?;
        // SAFETY: `env` and `session_options` are valid ORT handles, `path_c` is
        // NUL-terminated for the duration of the call, and `ptr` is an out-param.
        let create_result = crate::error::check_status(unsafe {
            create(
                env.as_ptr(),
                path_c.as_ptr(),
                session_options.as_ptr(),
                &mut ptr,
            )
        });
        let mut effective_providers = options.execution_providers.clone();
        if let Err(err) = create_result {
            if requested_non_cpu_provider(&options) {
                tracing::warn!(
                    "ORT session creation failed with requested execution provider(s): {err}; retrying with CPU"
                );
                let cpu_options = SessionOptions::cpu();
                let cpu_session_options = RawSessionOptions::new(&cpu_options)?;
                ptr = std::ptr::null_mut();
                // SAFETY: same invariants as above, with fresh CPU-only options.
                crate::error::check_status(unsafe {
                    create(
                        env.as_ptr(),
                        path_c.as_ptr(),
                        cpu_session_options.as_ptr(),
                        &mut ptr,
                    )
                })?;
                effective_providers = cpu_options.execution_providers;
            } else {
                return Err(err);
            }
        }
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

    /// Whether a WebGPU execution provider is (effectively) active for this
    /// session.
    pub fn is_webgpu(&self) -> bool {
        self.execution_providers
            .iter()
            .any(|provider| matches!(provider, ExecutionProvider::WebGpu))
    }

    /// Whether a CUDA execution provider is (effectively) active for this session.
    pub fn is_cuda(&self) -> bool {
        self.execution_providers
            .iter()
            .any(|provider| matches!(provider, ExecutionProvider::Cuda { .. }))
    }

    /// Create a device-resident allocator for KV buffers, if this session runs
    /// on an execution provider that owns device memory (CUDA or WebGPU).
    ///
    /// Returns `Ok(None)` for CPU/unsupported EPs, so callers keep using the CPU
    /// allocator. If a device EP is selected but ORT cannot produce a matching
    /// allocator (e.g. the EP silently fell back to CPU), the error is logged
    /// and `Ok(None)` is returned so decode still works via CPU buffers.
    pub(crate) fn device_kv_allocator(&self) -> Result<Option<Allocator>> {
        if !self.is_webgpu() && !self.is_cuda() {
            return Ok(None);
        }
        if !device_kv_enabled_from_env() {
            return Ok(None);
        }

        #[cfg(feature = "cuda")]
        if let Some(device_id) = self.execution_providers.iter().find_map(|provider| {
            if let ExecutionProvider::Cuda { device_id } = provider {
                Some(*device_id)
            } else {
                None
            }
        }) {
            let memory_info = MemoryInfo::cuda(device_id)?;
            return match Allocator::for_session_device(self.ptr.as_ptr(), memory_info) {
                Ok(allocator) => {
                    tracing::info!(
                        device_id,
                        "ONNX_GENAI_DEVICE_KV=1: allocating shared GQA KV on CUDA"
                    );
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
    fn new(options: &SessionOptions) -> Result<Self> {
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

        append_execution_providers(this.ptr.as_ptr(), options)?;
        apply_webgpu_provider_options(this.ptr.as_ptr(), options)?;

        Ok(this)
    }

    fn as_ptr(&self) -> *const onnx_genai_ort_sys::OrtSessionOptions {
        self.ptr.as_ptr()
    }
}

fn execution_providers_from_env() -> Option<Vec<ExecutionProvider>> {
    let value = std::env::var("ONNX_GENAI_EP").ok()?;
    let provider = match value.trim().to_ascii_lowercase().as_str() {
        "" | "cpu" => ExecutionProvider::Cpu,
        "webgpu" | "web-gpu" | "web_gpu" => ExecutionProvider::WebGpu,
        "cuda" => ExecutionProvider::Cuda {
            device_id: cuda_device_id_from_env(),
        },
        "coreml" | "core-ml" | "core_ml" => ExecutionProvider::CoreML,
        other => {
            tracing::warn!(
                "Ignoring unsupported ONNX_GENAI_EP={other}; expected cpu, webgpu, cuda, or coreml"
            );
            ExecutionProvider::Cpu
        }
    };
    Some(vec![provider])
}

fn requested_non_cpu_provider(options: &SessionOptions) -> bool {
    options
        .execution_providers
        .iter()
        .any(|provider| !matches!(provider, ExecutionProvider::Cpu))
}

fn env_flag_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

fn cuda_device_id_from_env() -> i32 {
    let Ok(value) = std::env::var("ONNX_GENAI_CUDA_DEVICE") else {
        return 0;
    };
    match parse_cuda_device_id(&value) {
        Some(device_id) => device_id,
        None => {
            tracing::warn!(
                "Ignoring invalid ONNX_GENAI_CUDA_DEVICE={value}; expected a non-negative integer, using device 0"
            );
            0
        }
    }
}

fn parse_cuda_device_id(value: &str) -> Option<i32> {
    value
        .trim()
        .parse::<i32>()
        .ok()
        .filter(|device_id| *device_id >= 0)
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
    let value = std::env::var("ONNX_GENAI_INTRA_OP_THREADS").ok()?;
    match value.trim().parse::<i32>() {
        Ok(threads) if threads > 0 => Some(threads),
        _ => {
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
    match std::env::var("ONNX_GENAI_WEBGPU_VALIDATION") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => true,
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
    match std::env::var("ONNX_GENAI_DEVICE_KV") {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
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
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    options: &SessionOptions,
) -> Result<()> {
    let available = available_execution_providers().unwrap_or_else(|err| {
        tracing::warn!("Could not query available ORT execution providers: {err}");
        Vec::new()
    });
    for provider in &options.execution_providers {
        append_execution_provider(session_options, provider, options.graph_capture, &available)?;
    }
    Ok(())
}

fn append_execution_provider(
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    provider: &ExecutionProvider,
    graph_capture: bool,
    available: &[String],
) -> Result<()> {
    match provider {
        ExecutionProvider::Cpu => Ok(()),
        ExecutionProvider::WebGpu => append_named_execution_provider(
            session_options,
            "WebGPU",
            "WebGpuExecutionProvider",
            &[],
            available,
        ),
        ExecutionProvider::Cuda { device_id } => {
            #[cfg(feature = "cuda")]
            {
                append_cuda_execution_provider(
                    session_options,
                    *device_id,
                    graph_capture,
                    available,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (session_options, device_id, graph_capture, available);
                Err(OrtError::InvalidArgument(
                    "CUDA support not compiled in; rebuild with --features cuda".into(),
                ))
            }
        }
        ExecutionProvider::CoreML => append_named_execution_provider(
            session_options,
            "CoreML",
            "CoreMLExecutionProvider",
            &[],
            available,
        ),
        other => {
            tracing::warn!(
                "Execution provider {:?} is not wired in onnx-genai-ort; falling back to CPU",
                other
            );
            Ok(())
        }
    }
}

#[cfg(feature = "cuda")]
fn append_cuda_execution_provider(
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    device_id: i32,
    graph_capture: bool,
    available: &[String],
) -> Result<()> {
    const PROVIDER_NAME: &str = "CUDAExecutionProvider";
    if !provider_is_available(PROVIDER_NAME, available) {
        tracing::warn!(
            "Requested ONNX Runtime execution provider CUDA is unavailable in this build; falling back to CPU. Available providers: {:?}",
            available
        );
        return Ok(());
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
        let mut provider_options = vec![("device_id", device_id.as_str())];
        if graph_capture {
            provider_options.push(("enable_cuda_graph", "1"));
        }
        let option_keys = provider_options
            .iter()
            .map(|(key, _)| {
                CString::new(*key).map_err(|_| {
                    OrtError::InvalidArgument("CUDA provider option key contains NUL".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let option_values = provider_options
            .iter()
            .map(|(_, value)| {
                CString::new(*value).map_err(|_| {
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
                "Enabled ONNX Runtime CUDA execution provider"
            );
            Ok(())
        }
        Err(err) => {
            tracing::warn!(
                "Failed to enable ONNX Runtime CUDA execution provider: {err}; falling back to CPU"
            );
            Ok(())
        }
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
    fn parses_cuda_device_ids() {
        assert_eq!(parse_cuda_device_id("0"), Some(0));
        assert_eq!(parse_cuda_device_id(" 7 "), Some(7));
        assert_eq!(parse_cuda_device_id("-1"), None);
        assert_eq!(parse_cuda_device_id("gpu0"), None);
    }

    #[test]
    fn recognizes_cuda_provider_names() {
        let available = vec!["CUDAExecutionProvider".to_string()];
        assert!(provider_is_available("CUDAExecutionProvider", &available));
        assert!(provider_is_available("CUDA", &available));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_request_requires_compile_time_feature() {
        let error = append_execution_provider(
            std::ptr::null_mut(),
            &ExecutionProvider::Cuda { device_id: 0 },
            false,
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
