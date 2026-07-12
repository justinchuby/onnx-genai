//! ORT Session — represents a loaded model.

use std::path::Path;
use crate::{Environment, Value, IoBinding, Result, OrtError};

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

/// An ORT inference session (a loaded model).
pub struct Session {
    // ptr: *mut ort_sys::OrtSession,
    _model_path: String,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl Session {
    /// Load a model from an ONNX file.
    pub fn new(_env: &Environment, path: &Path, _options: SessionOptions) -> Result<Self> {
        if !path.exists() {
            return Err(OrtError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Model file not found: {}", path.display()),
            )));
        }

        // TODO: Call OrtCreateSession via C API
        tracing::info!("Loading model: {}", path.display());

        Ok(Self {
            _model_path: path.display().to_string(),
            input_names: Vec::new(),  // TODO: query from session
            output_names: Vec::new(),
        })
    }

    /// Run inference with named inputs, returns named outputs.
    pub fn run(&self, _inputs: &[(&str, &Value)]) -> Result<Vec<Value>> {
        // TODO: Call OrtRun via C API
        Ok(Vec::new())
    }

    /// Run inference using pre-bound I/O (zero-copy for device tensors).
    pub fn run_with_binding(&self, _binding: &IoBinding) -> Result<()> {
        // TODO: Call OrtRunWithBinding via C API
        Ok(())
    }

    /// Get input names.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Get output names.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // TODO: Call OrtReleaseSession
    }
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}
