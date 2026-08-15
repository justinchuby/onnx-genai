use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use memmap2::Mmap;
use onnx_runtime_ir::WeightRef;

use super::host::{HostOrtValue, HostOrtValueStorage, HostTensorTypeAndShapeInfo, dtype_to_ort};

/// External weight files, mapped once and shared by every tensor in them.
///
/// A model keeps all its weights in one file, and a plugin is asked about
/// every initializer of every subgraph it claims. Opening and reading that
/// file per tensor means re-reading gigabytes for each of a few hundred
/// lookups -- on a 0.5B model claimed as 97 subgraphs it turned a load into
/// something that had not finished after eight minutes. Mapping is also what
/// the loader does with the same files, so the pages are shared rather than
/// duplicated per tensor.
static MAPPED_WEIGHTS: OnceLock<Mutex<HashMap<PathBuf, MappedWeightFile>>> = OnceLock::new();

/// A mapping plus enough of the file's identity to notice it was replaced.
struct MappedWeightFile {
    map: Arc<Mmap>,
    /// Modification time and length at the moment it was mapped.
    ///
    /// Keyed by path alone, the cache would hand a later session the previous
    /// model's weights after a file at the same path was rebuilt -- wrong
    /// numbers, no error. Cheap to check and it makes the cache safe to keep
    /// process-wide.
    identity: (Option<std::time::SystemTime>, u64),
}

fn weight_file_identity(path: &Path) -> (Option<std::time::SystemTime>, u64) {
    match std::fs::metadata(path) {
        Ok(meta) => (meta.modified().ok(), meta.len()),
        Err(_) => (None, 0),
    }
}

fn mapped_weight_file(path: &Path) -> Option<Arc<Mmap>> {
    let identity = weight_file_identity(path);
    let cache = MAPPED_WEIGHTS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(entry) = cache.get(path)
        && entry.identity == identity
    {
        return Some(Arc::clone(&entry.map));
    }
    let file = std::fs::File::open(path).ok()?;
    // SAFETY: the file is opened read-only and the mapping is never handed out
    // mutably. A weight file rewritten under a live mapping would be a problem
    // for the loader's own mapping of the same file first; this adds no new
    // exposure, and the identity check above stops a *new* session inheriting
    // a stale one.
    let map = Arc::new(unsafe { Mmap::map(&file) }.ok()?);
    cache.insert(
        path.to_path_buf(),
        MappedWeightFile {
            map: Arc::clone(&map),
            identity,
        },
    );
    Some(map)
}

pub(super) fn host_ort_value_for_weight(weight: &WeightRef) -> Option<Box<HostOrtValue>> {
    let (dtype, dims, data) = match weight {
        WeightRef::Inline(tensor) => (tensor.dtype, tensor.dims.clone(), tensor.data.clone()),
        WeightRef::External {
            path,
            offset,
            length,
            dtype,
            dims,
        } => {
            let map = mapped_weight_file(path)?;
            let end = offset.checked_add(*length)?;
            // Bounds-checked here so the pointer handed to the plugin is known
            // to be inside the mapping.
            map.get(*offset..end)?;
            return Some(Box::new(HostOrtValue {
                tensor: HostTensorTypeAndShapeInfo {
                    dtype: dtype_to_ort(*dtype),
                    dims: dims.iter().map(|d| *d as i64).collect(),
                },
                storage: HostOrtValueStorage::Mapped {
                    map,
                    offset: *offset,
                    len: *length,
                },
            }));
        }
    };
    Some(Box::new(HostOrtValue {
        tensor: HostTensorTypeAndShapeInfo {
            dtype: dtype_to_ort(dtype),
            dims: dims.into_iter().map(|d| d as i64).collect(),
        },
        storage: HostOrtValueStorage::Owned(data),
    }))
}
