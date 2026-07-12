//! ORT Session — represents a loaded model.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::NonNull;

use crate::{Allocator, DataType, Environment, IoBinding, OrtError, Result, Value};

/// Execution provider selection.
#[derive(Debug, Clone)]
pub enum ExecutionProvider {
    Cpu,
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
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            execution_providers: vec![ExecutionProvider::Cpu],
            optimization_level: 99,
            intra_op_num_threads: 0, // ORT decides
            inter_op_num_threads: 0,
        }
    }
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
        crate::error::check_status(unsafe {
            create(
                env.as_ptr(),
                path_c.as_ptr(),
                session_options.as_ptr(),
                &mut ptr,
            )
        })?;
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

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

struct RawSessionOptions {
    ptr: NonNull<onnx_genai_ort_sys::OrtSessionOptions>,
}

impl RawSessionOptions {
    fn new(options: &SessionOptions) -> Result<Self> {
        if options
            .execution_providers
            .iter()
            .any(|provider| !matches!(provider, ExecutionProvider::Cpu))
        {
            return Err(OrtError::InvalidArgument(
                "only the CPU execution provider is wired in Phase 1".into(),
            ));
        }

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
        if options.intra_op_num_threads > 0
            && let Some(set_threads) = api.SetIntraOpNumThreads
        {
            // SAFETY: `this.ptr` is a valid session options handle.
            crate::error::check_status(unsafe {
                set_threads(this.ptr.as_ptr(), options.intra_op_num_threads)
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
