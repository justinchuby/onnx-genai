//! ORT IoBinding — pre-bind inputs/outputs to avoid per-run copies.

use crate::{Session, Value, MemoryInfo, Result};

/// IoBinding allows pre-allocating and binding device tensors.
/// This is critical for KV cache: we keep cache pages on-device and
/// bind them directly without host↔device copies each step.
pub struct IoBinding {
    // ptr: *mut ort_sys::OrtIoBinding,
    _session: *const Session, // reference back to session (non-owning)
}

impl IoBinding {
    /// Create a new IoBinding for a session.
    pub fn new(_session: &Session) -> Result<Self> {
        // TODO: Call OrtCreateIoBinding
        Ok(Self {
            _session: _session as *const Session,
        })
    }

    /// Bind a pre-existing tensor to a named input.
    pub fn bind_input(&mut self, _name: &str, _value: &Value) -> Result<()> {
        // TODO: Call OrtBindInput
        Ok(())
    }

    /// Bind output to a specific device (ORT allocates on that device).
    pub fn bind_output_to_device(&mut self, _name: &str, _memory_info: &MemoryInfo) -> Result<()> {
        // TODO: Call OrtBindOutputToDevice
        Ok(())
    }

    /// Bind a pre-existing tensor to a named output.
    pub fn bind_output(&mut self, _name: &str, _value: &Value) -> Result<()> {
        // TODO: Call OrtBindOutput
        Ok(())
    }

    /// Clear all bindings (reuse the binding object).
    pub fn clear(&mut self) -> Result<()> {
        // TODO: Call OrtClearBoundInputs + OrtClearBoundOutputs
        Ok(())
    }
}

impl Drop for IoBinding {
    fn drop(&mut self) {
        // TODO: Call OrtReleaseIoBinding
    }
}
