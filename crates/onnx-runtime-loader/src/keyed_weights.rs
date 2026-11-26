use crate::{LoaderError, WeightStore};
use onnx_runtime_ir::{DataType, TensorData, WeightRef};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

mod gguf;
mod safetensors;

#[derive(Clone, Copy, Debug)]
enum Format {
    Gguf,
    SafeTensors,
}

/// One keyed weight container. Paths are caller-supplied, relative to the
/// working directory unless absolute. Mappings run from file key to ONNX
/// initializer name; unmapped keys retain their original names.
#[derive(Clone, Debug)]
pub struct KeyedWeightFile {
    path: PathBuf,
    format: Format,
    renames: Vec<(String, String)>,
}

impl KeyedWeightFile {
    pub fn gguf(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            format: Format::Gguf,
            renames: Vec::new(),
        }
    }

    pub fn safetensors(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            format: Format::SafeTensors,
            renames: Vec::new(),
        }
    }

    /// Rename this file's tensor `key` to `initializer`. Repeated source keys
    /// or destinations are rejected at load time.
    #[must_use]
    pub fn rename(mut self, key: impl Into<String>, initializer: impl Into<String>) -> Self {
        self.renames.push((key.into(), initializer.into()));
        self
    }
}

/// One borrowed tensor from a fully validated SafeTensors container.
/// Offsets, shapes, duplicate keys, dtypes and complete payload coverage have
/// already passed the same parser used for external initializer binding.
pub struct SafeTensorView<'a> {
    pub dtype: DataType,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

/// Inspect caller-owned SafeTensors bytes without mapping or copying payloads.
/// The caller remains responsible for its allocation and dtype capabilities.
pub fn safetensors_views(data: &[u8]) -> Result<BTreeMap<String, SafeTensorView<'_>>, LoaderError> {
    let entries = safetensors::parse(data).map_err(|detail| {
        LoaderError::GraphBuild(format!(
            "SafeTensors: {detail}; provide a complete valid tensor container"
        ))
    })?;
    Ok(entries
        .into_iter()
        .map(|(name, entry)| {
            (
                name,
                SafeTensorView {
                    dtype: entry.dtype,
                    shape: entry.dims,
                    data: &data[entry.offset..entry.offset + entry.len],
                },
            )
        })
        .collect())
}

struct Entry {
    dtype: DataType,
    packed_format: Option<&'static str>,
    dims: Vec<usize>,
    offset: usize,
    len: usize,
}

type Entries = BTreeMap<String, Entry>;

struct Candidate {
    path: PathBuf,
    key: String,
    entry: Entry,
}

#[derive(Default)]
pub(crate) struct KeyedWeights {
    explicit: BTreeMap<String, Candidate>,
    implicit: HashMap<String, Vec<Candidate>>,
}

fn file_error(path: &Path, detail: impl std::fmt::Display) -> LoaderError {
    LoaderError::GraphBuild(format!(
        "keyed weights {}: {detail}; check the container format and weight mapping",
        path.display()
    ))
}

impl KeyedWeights {
    pub(crate) fn load(
        files: &[KeyedWeightFile],
        store: &mut WeightStore,
    ) -> Result<Self, LoaderError> {
        let mut index = Self::default();
        for file in files {
            let path = &file.path;
            store.map_external(path)?;
            let data = store
                .mapped_file_bytes(path)
                .ok_or_else(|| file_error(path, "mapped payload is unavailable"))?;
            let mut entries = match file.format {
                Format::Gguf => gguf::parse(data),
                Format::SafeTensors => safetensors::parse(data),
            }
            .map_err(|detail| file_error(path, detail))?;
            for (key, name) in &file.renames {
                if name.is_empty() {
                    return Err(file_error(
                        path,
                        format!("mapping for key {key:?} has an empty ONNX name"),
                    ));
                }
                let entry = entries.remove(key).ok_or_else(|| {
                    file_error(path, format!("mapping key {key:?} is absent or repeated"))
                })?;
                let candidate = Candidate {
                    path: path.clone(),
                    key: key.clone(),
                    entry,
                };
                if let Some(other) = index.explicit.insert(name.clone(), candidate) {
                    return Err(file_error(
                        path,
                        format!(
                            "ONNX initializer {name:?} is mapped more than once (also key {:?} in {})",
                            other.key,
                            other.path.display()
                        ),
                    ));
                }
            }
            for (key, entry) in entries {
                index
                    .implicit
                    .entry(key.clone())
                    .or_default()
                    .push(Candidate {
                        path: path.clone(),
                        key,
                        entry,
                    });
            }
        }
        Ok(index)
    }

    pub(crate) fn resolve(
        &mut self,
        name: &str,
        dtype: DataType,
        dims: Vec<usize>,
    ) -> Result<WeightRef, LoaderError> {
        let candidate = if let Some(explicit) = self.explicit.remove(name) {
            explicit
        } else {
            match self.implicit.remove(name) {
                None => {
                    return Err(LoaderError::GraphBuild(format!(
                        "external initializer {name:?} has no keyed weight; supply a matching file key or map a key to this ONNX name",
                    )));
                }
                Some(mut matches) if matches.len() == 1 => matches.remove(0),
                Some(_) => {
                    return Err(LoaderError::GraphBuild(format!(
                        "external initializer {name:?} matches more than one file key; explicitly map the desired file's key to this ONNX name",
                    )));
                }
            }
        };
        let Entry {
            dtype: file_dtype,
            packed_format,
            dims: file_dims,
            offset,
            len,
        } = candidate.entry;
        if file_dtype != dtype || file_dims != dims {
            let encoding = packed_format.map_or_else(
                || format!("{file_dtype:?}"),
                |format| format!("{format} packed payload as UINT8"),
            );
            return Err(file_error(
                &candidate.path,
                format!(
                    "key {:?} is {encoding} {file_dims:?}, but ONNX initializer {name:?} requires {dtype:?} {dims:?}; use exactly matching dtype and shape (no casting, transposing or dequantization)",
                    candidate.key
                ),
            ));
        }
        Ok(WeightRef::External {
            path: candidate.path,
            offset,
            length: len,
            dtype,
            dims,
        })
    }

    pub(crate) fn finish(self) -> Result<(), LoaderError> {
        if let Some((name, candidate)) = self.explicit.into_iter().next() {
            return Err(file_error(
                &candidate.path,
                format!(
                    "key {:?} maps to {name:?}, which is not an external initializer with an empty or omitted location in the top-level ONNX graph",
                    candidate.key
                ),
            ));
        }
        Ok(())
    }
}

fn byte_len(dtype: DataType, dims: &[usize]) -> Result<usize, String> {
    TensorData::from_raw(dtype, dims.to_vec(), Vec::new())
        .checked_expected_bytes()
        .ok_or_else(|| format!("{dtype:?} shape {dims:?} overflows the tensor byte count"))
}
