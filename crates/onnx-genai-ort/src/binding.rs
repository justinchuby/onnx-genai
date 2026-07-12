//! ORT IoBinding — pre-bind inputs/outputs to avoid per-run copies.

use std::ffi::CString;
use std::ptr::NonNull;

use crate::{Allocator, MemoryInfo, OrtError, Result, Session, Value};

/// IoBinding allows pre-allocating and binding device tensors.
/// This is critical for KV cache: we keep cache pages on-device and
/// bind them directly without host↔device copies each step.
pub struct IoBinding {
    ptr: NonNull<onnx_genai_ort_sys::OrtIoBinding>,
    _session: *const Session, // reference back to session (non-owning)
}

impl IoBinding {
    /// Create a new IoBinding for a session.
    pub fn new(session: &Session) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let api = crate::error::api()?;
        let create = api
            .CreateIoBinding
            .ok_or(OrtError::ApiUnavailable("CreateIoBinding"))?;
        // SAFETY: `session` is a valid ORT session and `ptr` is an out-param.
        crate::error::check_status(unsafe { create(session.as_mut_ptr(), &mut ptr) })?;
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(OrtError::NullPointer)?,
            _session: session as *const Session,
        })
    }

    /// Bind a pre-existing tensor to a named input.
    pub fn bind_input(&mut self, name: &str, value: &Value) -> Result<()> {
        let name = c_name(name)?;
        let api = crate::error::api()?;
        let bind = api.BindInput.ok_or(OrtError::ApiUnavailable("BindInput"))?;
        // SAFETY: binding and value are valid ORT handles; `name` is
        // NUL-terminated and lives for the call.
        crate::error::check_status(unsafe {
            bind(self.ptr.as_ptr(), name.as_ptr(), value.as_ptr())
        })
    }

    /// Bind output to a specific device (ORT allocates on that device).
    pub fn bind_output_to_device(&mut self, name: &str, memory_info: &MemoryInfo) -> Result<()> {
        let name = c_name(name)?;
        let api = crate::error::api()?;
        let bind = api
            .BindOutputToDevice
            .ok_or(OrtError::ApiUnavailable("BindOutputToDevice"))?;
        // SAFETY: binding and memory info are valid ORT handles; `name` is
        // NUL-terminated and lives for the call.
        crate::error::check_status(unsafe {
            bind(self.ptr.as_ptr(), name.as_ptr(), memory_info.as_ptr())
        })
    }

    /// Bind a pre-existing tensor to a named output.
    pub fn bind_output(&mut self, name: &str, value: &Value) -> Result<()> {
        let name = c_name(name)?;
        let api = crate::error::api()?;
        let bind = api
            .BindOutput
            .ok_or(OrtError::ApiUnavailable("BindOutput"))?;
        // SAFETY: binding and value are valid ORT handles; `name` is
        // NUL-terminated and lives for the call.
        crate::error::check_status(unsafe {
            bind(self.ptr.as_ptr(), name.as_ptr(), value.as_ptr())
        })
    }

    /// Clear all bindings (reuse the binding object).
    pub fn clear(&mut self) -> Result<()> {
        let api = crate::error::api()?;
        let clear_inputs = api
            .ClearBoundInputs
            .ok_or(OrtError::ApiUnavailable("ClearBoundInputs"))?;
        let clear_outputs = api
            .ClearBoundOutputs
            .ok_or(OrtError::ApiUnavailable("ClearBoundOutputs"))?;
        // SAFETY: binding is valid; ORT clear functions do not return status.
        unsafe {
            clear_inputs(self.ptr.as_ptr());
            clear_outputs(self.ptr.as_ptr());
        }
        Ok(())
    }

    pub(crate) fn as_ptr(&self) -> *const onnx_genai_ort_sys::OrtIoBinding {
        self.ptr.as_ptr()
    }

    /// Take the OrtValues produced by the most recent `RunWithBinding`.
    ///
    /// Values are returned in the same order outputs were bound.
    pub fn output_values(&self) -> Result<Vec<Value>> {
        let allocator = Allocator::default_cpu()?;
        let api = crate::error::api()?;
        let get_outputs = api
            .GetBoundOutputValues
            .ok_or(OrtError::ApiUnavailable("GetBoundOutputValues"))?;
        let free = api
            .AllocatorFree
            .ok_or(OrtError::ApiUnavailable("AllocatorFree"))?;
        let mut output_ptrs = std::ptr::null_mut();
        let mut output_count = 0usize;
        // SAFETY: binding and allocator are valid; ORT allocates an array of
        // OrtValue pointers and transfers ownership of each OrtValue to us.
        crate::error::check_status(unsafe {
            get_outputs(
                self.ptr.as_ptr(),
                allocator.as_ptr(),
                &mut output_ptrs,
                &mut output_count,
            )
        })?;
        if output_count == 0 {
            return Ok(Vec::new());
        }
        if output_ptrs.is_null() {
            return Err(OrtError::NullPointer);
        }

        // SAFETY: ORT returned `output_count` pointers in an allocator-owned
        // array. We copy the pointers, free the array, and wrap each value.
        let raw_values = unsafe { std::slice::from_raw_parts(output_ptrs, output_count) }.to_vec();
        crate::error::check_status(unsafe { free(allocator.as_ptr(), output_ptrs.cast()) })?;
        raw_values
            .into_iter()
            .map(|ptr| unsafe { Value::from_raw(ptr) })
            .collect()
    }
}

impl Drop for IoBinding {
    fn drop(&mut self) {
        if let Ok(api) = crate::error::api()
            && let Some(release) = api.ReleaseIoBinding
        {
            // SAFETY: `ptr` is owned by this wrapper and released exactly once here.
            unsafe { release(self.ptr.as_ptr()) };
        }
    }
}

fn c_name(name: &str) -> Result<CString> {
    CString::new(name).map_err(|_| OrtError::InvalidArgument(format!("name contains NUL: {name}")))
}
