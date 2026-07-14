//! Initializer weight resolution: inline data and external `mmap` (§19.2, §12).

use std::collections::HashMap;
use std::path::Path;

use onnx_runtime_ir::{ValueId, WeightRef};

use crate::proto::ModelProto;
use crate::LoaderError;

/// A resolved set of initializer weights, keyed by the value they populate.
#[derive(Debug, Default)]
pub struct WeightStore {
    pub weights: HashMap<ValueId, WeightRef>,
}

impl WeightStore {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Resolve all initializers, memory-mapping external data relative to
/// `model_dir`.
pub fn load_weights(model: &ModelProto, model_dir: &Path) -> Result<WeightStore, LoaderError> {
    let _ = (model, model_dir);
    todo!("ort2-loader: resolve inline + external initializer data, mmap external files")
}
