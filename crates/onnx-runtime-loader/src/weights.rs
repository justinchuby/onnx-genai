//! Initializer weight resolution: inline data and external `mmap` (§19.2, §12).
//!
//! Turns each `TensorProto` initializer into an [`onnx_runtime_ir::WeightRef`]
//! descriptor that the IR stores. Inline tensors keep their bytes; external
//! tensors are described by `(path, offset, length)` and the referenced files
//! are memory-mapped so downstream consumers get zero-copy access via
//! [`WeightStore::bytes`].

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use onnx_runtime_ir::{DataType, TensorData, ValueId, WeightRef};

use crate::proto::onnx::{tensor_proto, ModelProto, TensorProto};
use crate::{pathsafe::guarded_join, LoaderError};

/// A resolved set of initializer weights, keyed by the value they populate,
/// plus the live memory maps backing any external data files.
#[derive(Debug, Default)]
pub struct WeightStore {
    /// IR weight descriptors, keyed by the graph value they initialize.
    pub weights: HashMap<ValueId, WeightRef>,
    /// Live memory maps for external-data files, keyed by absolute path.
    mmaps: HashMap<PathBuf, Mmap>,
}

impl WeightStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve a weight descriptor to its raw little-endian bytes.
    ///
    /// For inline weights this borrows the stored bytes; for external weights
    /// it slices into the memory-mapped file. Returns `None` if an external
    /// mapping is missing or the `[offset, offset+length)` window is out of
    /// bounds.
    pub fn bytes<'a>(&'a self, weight: &'a WeightRef) -> Option<&'a [u8]> {
        match weight {
            WeightRef::Inline(t) => Some(&t.data),
            WeightRef::External {
                path,
                offset,
                length,
                ..
            } => {
                let mmap = self.mmaps.get(path)?;
                mmap.get(*offset..offset.checked_add(*length)?)
            }
        }
    }

    fn mmap_file(&mut self, path: &Path) -> Result<(), LoaderError> {
        if self.mmaps.contains_key(path) {
            return Ok(());
        }
        let file = File::open(path).map_err(|_| LoaderError::ExternalDataNotFound {
            path: path.to_path_buf(),
        })?;
        // SAFETY: we hold the `File` open for the duration of the map and never
        // expose a mutable view. The mapped bytes are treated as immutable
        // weight storage. This is the only `unsafe` in the loader; the IR crate
        // stays `#![forbid(unsafe_code)]`.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| LoaderError::Mmap(e.to_string()))?;
        self.mmaps.insert(path.to_path_buf(), mmap);
        Ok(())
    }
}

/// Resolve all initializers, memory-mapping external data relative to
/// `model_dir`. `name_map` maps initializer names to the graph value ids
/// created by the [`graph_builder`](crate::graph_builder).
pub fn load_weights(
    model: &ModelProto,
    model_dir: &Path,
    name_map: &HashMap<String, ValueId>,
) -> Result<WeightStore, LoaderError> {
    let mut store = WeightStore::new();
    let Some(graph) = model.graph.as_ref() else {
        return Ok(store);
    };

    for init in &graph.initializer {
        let Some(&vid) = name_map.get(&init.name) else {
            // An initializer with no corresponding graph value: skip it. The
            // builder registers a value for every initializer, so this only
            // happens for malformed models.
            continue;
        };
        let weight = resolve_initializer(&mut store, init, model_dir)?;
        store.weights.insert(vid, weight);
    }

    Ok(store)
}

fn resolve_initializer(
    store: &mut WeightStore,
    init: &TensorProto,
    model_dir: &Path,
) -> Result<WeightRef, LoaderError> {
    let dtype = DataType::from_onnx(init.data_type).ok_or_else(|| {
        LoaderError::UnsupportedDataType {
            raw: init.data_type,
            context: format!("initializer {:?}", init.name),
        }
    })?;
    let dims: Vec<usize> = init.dims.iter().map(|&d| d.max(0) as usize).collect();

    if init.data_location == tensor_proto::DataLocation::External as i32 {
        let mut location = None;
        let mut offset: usize = 0;
        let mut length: Option<usize> = None;
        for kv in &init.external_data {
            match kv.key.as_str() {
                "location" => location = Some(kv.value.clone()),
                "offset" => offset = kv.value.parse().unwrap_or(0),
                "length" => length = kv.value.parse().ok(),
                _ => {}
            }
        }
        let location = location.ok_or_else(|| {
            LoaderError::GraphBuild(format!(
                "external initializer {:?} missing 'location'",
                init.name
            ))
        })?;
        let path = resolve_external_path(model_dir, &location)?;
        store.mmap_file(&path)?;
        let numel: usize = dims.iter().product();
        let length = length.unwrap_or_else(|| dtype.storage_bytes(numel));
        // Validate the window lies within the mapped file (catches truncated or
        // mis-described external data early).
        if let Some(mmap) = store.mmaps.get(&path) {
            let end = offset.checked_add(length);
            if end.is_none_or(|e| e > mmap.len()) {
                return Err(LoaderError::Mmap(format!(
                    "external initializer {:?}: window [{offset}, {:?}) exceeds file {} ({} bytes)",
                    init.name,
                    end,
                    path.display(),
                    mmap.len()
                )));
            }
        }
        Ok(WeightRef::External {
            path,
            offset,
            length,
            dtype,
            dims,
        })
    } else {
        let data = tensor_data_from_proto(init, dtype, &dims)?;
        Ok(WeightRef::Inline(data))
    }
}

/// Join an external-data location onto `model_dir`, rejecting paths that can
/// escape the model directory.
fn resolve_external_path(model_dir: &Path, location: &str) -> Result<PathBuf, LoaderError> {
    guarded_join(model_dir, location).map_err(|reason| LoaderError::ExternalDataPath {
        path: location.to_string(),
        reason,
    })
}

/// Convert a `TensorProto`'s payload into an IR [`TensorData`] with raw
/// little-endian bytes (or string payloads for `STRING` tensors).
pub(crate) fn tensor_data_from_proto(
    proto: &TensorProto,
    dtype: DataType,
    dims: &[usize],
) -> Result<TensorData, LoaderError> {
    let mut td = TensorData::from_raw(dtype, dims.to_vec(), Vec::new());
    if !proto.name.is_empty() {
        td.name = Some(proto.name.clone());
    }

    if dtype == DataType::String {
        td.strings = proto
            .string_data
            .iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect();
        return Ok(td);
    }

    // Prefer raw_data when present (the common, mmap-friendly encoding).
    if !proto.raw_data.is_empty() {
        td.data = proto.raw_data.clone();
        return Ok(td);
    }

    // Otherwise serialise the type-specific repeated field to LE bytes.
    td.data = match dtype {
        DataType::Float32 => proto
            .float_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DataType::Float64 => proto
            .double_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DataType::Int64 => proto
            .int64_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DataType::Uint64 | DataType::Uint32 => proto
            .uint64_data
            .iter()
            .flat_map(|v| match dtype {
                DataType::Uint32 => (*v as u32).to_le_bytes().to_vec(),
                _ => v.to_le_bytes().to_vec(),
            })
            .collect(),
        DataType::Int32 => proto
            .int32_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        // Types packed into int32_data at a narrower width.
        DataType::Int16 | DataType::Uint16 | DataType::Float16 | DataType::BFloat16 => proto
            .int32_data
            .iter()
            .flat_map(|v| (*v as u16).to_le_bytes())
            .collect(),
        DataType::Int8 | DataType::Uint8 | DataType::Bool => {
            proto.int32_data.iter().map(|v| *v as u8).collect()
        }
        _ => Vec::new(),
    };
    Ok(td)
}
