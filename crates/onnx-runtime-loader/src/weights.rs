//! Initializer weight resolution: inline data and external `mmap` (§19.2, §12).
//!
//! Turns each `TensorProto` initializer into an [`onnx_runtime_ir::WeightRef`]
//! descriptor that the IR stores. Inline tensors keep their bytes; external
//! tensors are described by `(path, offset, length)` and the referenced files
//! are memory-mapped so downstream consumers get zero-copy access via
//! [`WeightStore::bytes`].

use std::collections::HashMap;
use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use onnx_runtime_ir::{DataType, TensorData, ValueId, WeightRef};

use crate::proto::onnx::{ModelProto, TensorProto, tensor_proto};
use crate::{LoaderError, pathsafe::guarded_join};

/// A resolved set of initializer weights, keyed by the value they populate,
/// plus the live memory maps backing any external data files.
#[derive(Debug, Default)]
pub struct WeightStore {
    /// IR weight descriptors, keyed by the graph value they initialize.
    pub weights: HashMap<ValueId, WeightRef>,
    /// Live memory maps for external-data files, keyed by absolute path.
    mmaps: HashMap<PathBuf, Mmap>,
}

/// Quantization geometry needed to interpret one expert's packed tensor slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertQuantization {
    /// Number of quantized weight bits per logical value.
    pub bits: usize,
    /// Number of logical input values sharing one scale/zero-point block.
    pub block_size: usize,
    /// Number of quantization blocks represented by each tensor row.
    pub blocks_per_row: usize,
}

/// Physical storage order declared by the model/package layout descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertStorageOrder {
    /// All rows for expert 0, then all rows for expert 1, and so on.
    ExpertMajor,
    /// Expert rows are interleaved and therefore cannot be represented by one range.
    Interleaved,
}

/// Compact layout descriptor used to derive expert byte ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertTensorLayout {
    /// Mandatory layout contract version. Phase 1 supports version 1.
    pub version: u32,
    pub experts: usize,
    pub rows_per_expert: usize,
    /// Stored elements per row (packed bytes for a `Uint8` weight tensor).
    pub storage_elements_per_row: usize,
    pub order: ExpertStorageOrder,
    pub quantization: Option<ExpertQuantization>,
}

/// Why an external tensor cannot use the expert paging path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NonPageableReason {
    InlineTensor,
    UnsupportedLayoutVersion(u32),
    NotExpertMajor,
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    InvalidQuantization(String),
    Range(String),
    ExternalLengthMismatch {
        expected: usize,
        actual: usize,
    },
}

/// Pageability classification. Non-pageable tensors remain valid and use the
/// existing resident/materializing execution path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pageability {
    Pageable,
    NonPageable(NonPageableReason),
}

/// One expert's contiguous byte window in an external initializer mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertWeightRegion {
    pub expert: usize,
    /// Absolute byte offset in the external-data file.
    pub offset: usize,
    pub len: usize,
}

/// Validated expert-region catalog over a [`WeightRef`] in a [`WeightStore`].
///
/// A pageable catalog guarantees that each expert is represented by one
/// overflow-checked external byte range. The quantization metadata is retained
/// so a kernel can decode one range without inspecting or materializing peers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightRegionCatalog {
    path: Option<PathBuf>,
    tensor_offset: usize,
    tensor_len: usize,
    dtype: DataType,
    layout: ExpertTensorLayout,
    regions: Vec<ExpertWeightRegion>,
    pageability: Pageability,
}

impl WeightRegionCatalog {
    /// Classify a weight using an explicit model/package layout descriptor.
    ///
    /// Invalid or non-expert-major layouts produce a non-pageable catalog rather
    /// than failing model load; callers must fall back to the resident path.
    pub fn classify(weight: &WeightRef, layout: ExpertTensorLayout) -> Self {
        let dtype = weight.dtype();
        let dims = weight.dims();
        let (path, tensor_offset, tensor_len) = match weight {
            WeightRef::Inline(tensor) => {
                return Self::non_pageable(
                    None,
                    0,
                    tensor.data.len(),
                    dtype,
                    layout,
                    NonPageableReason::InlineTensor,
                );
            }

            WeightRef::External {
                path,
                offset,
                length,
                ..
            } => (Some(path.clone()), *offset, *length),
        };

        if layout.version != 1 {
            return Self::non_pageable(
                path,
                tensor_offset,
                tensor_len,
                dtype,
                layout.clone(),
                NonPageableReason::UnsupportedLayoutVersion(layout.version),
            );
        }
        if layout.order != ExpertStorageOrder::ExpertMajor {
            return Self::non_pageable(
                path,
                tensor_offset,
                tensor_len,
                dtype,
                layout,
                NonPageableReason::NotExpertMajor,
            );
        }
        let expected_shape = vec![
            layout.experts,
            layout.rows_per_expert,
            layout.storage_elements_per_row,
        ];
        if dims != expected_shape {
            return Self::non_pageable(
                path,
                tensor_offset,
                tensor_len,
                dtype,
                layout,
                NonPageableReason::ShapeMismatch {
                    expected: expected_shape,
                    actual: dims.to_vec(),
                },
            );
        }
        if let Some(quantization) = layout.quantization
            && (!matches!(quantization.bits, 1 | 2 | 4 | 8)
                || quantization.block_size == 0
                || quantization.blocks_per_row == 0)
        {
            return Self::non_pageable(
                path,
                tensor_offset,
                tensor_len,
                dtype,
                layout,
                NonPageableReason::InvalidQuantization(format!(
                    "bits={}, block_size={}, blocks_per_row={}",
                    quantization.bits, quantization.block_size, quantization.blocks_per_row
                )),
            );
        }

        let elements_per_expert = match checked_product(
            &[layout.rows_per_expert, layout.storage_elements_per_row],
            "per-expert element count",
        ) {
            Ok(value) => value,
            Err(error) => {
                return Self::non_pageable(
                    path,
                    tensor_offset,
                    tensor_len,
                    dtype,
                    layout,
                    NonPageableReason::Range(error.to_string()),
                );
            }
        };
        let bytes_per_expert =
            match checked_storage_byte_count(dtype, elements_per_expert, "per-expert byte count") {
                Ok(value) => value,
                Err(error) => {
                    return Self::non_pageable(
                        path,
                        tensor_offset,
                        tensor_len,
                        dtype,
                        layout,
                        NonPageableReason::Range(error.to_string()),
                    );
                }
            };
        let expected_len = match checked_byte_count(
            layout.experts,
            bytes_per_expert,
            "expert tensor byte count",
        ) {
            Ok(value) => value,
            Err(error) => {
                return Self::non_pageable(
                    path,
                    tensor_offset,
                    tensor_len,
                    dtype,
                    layout,
                    NonPageableReason::Range(error.to_string()),
                );
            }
        };
        if expected_len != tensor_len {
            return Self::non_pageable(
                path,
                tensor_offset,
                tensor_len,
                dtype,
                layout,
                NonPageableReason::ExternalLengthMismatch {
                    expected: expected_len,
                    actual: tensor_len,
                },
            );
        }

        let mut regions = Vec::new();
        if let Err(error) = regions.try_reserve_exact(layout.experts) {
            return Self::non_pageable(
                path,
                tensor_offset,
                tensor_len,
                dtype,
                layout,
                NonPageableReason::Range(format!("expert region allocation failed: {error}")),
            );
        }
        for expert in 0..layout.experts {
            let relative = match checked_range(expert, bytes_per_expert, "expert byte range") {
                Ok(value) => value,
                Err(error) => {
                    return Self::non_pageable(
                        path,
                        tensor_offset,
                        tensor_len,
                        dtype,
                        layout,
                        NonPageableReason::Range(error.to_string()),
                    );
                }
            };
            let offset = match tensor_offset.checked_add(relative.start) {
                Some(value) => value,
                None => {
                    return Self::non_pageable(
                        path,
                        tensor_offset,
                        tensor_len,
                        dtype,
                        layout,
                        NonPageableReason::Range("expert absolute offset overflow".into()),
                    );
                }
            };
            regions.push(ExpertWeightRegion {
                expert,
                offset,
                len: bytes_per_expert,
            });
        }

        Self {
            path,
            tensor_offset,
            tensor_len,
            dtype,
            layout,
            regions,
            pageability: Pageability::Pageable,
        }
    }

    /// Build relative expert ranges for a tensor view that the caller has
    /// already established aliases an external mmap initializer.
    ///
    /// This is the kernel-boundary counterpart to [`classify`](Self::classify):
    /// it cannot re-borrow through [`WeightStore`], but applies the same shape,
    /// layout, quantization, overflow, and `isize::MAX` validation.
    pub fn for_mapped_tensor_view(
        dtype: DataType,
        dims: &[usize],
        tensor_len: usize,
        layout: ExpertTensorLayout,
    ) -> Self {
        let synthetic = WeightRef::External {
            path: PathBuf::new(),
            offset: 0,
            length: tensor_len,
            dtype,
            dims: dims.to_vec(),
        };
        let mut catalog = Self::classify(&synthetic, layout);
        catalog.path = None;
        catalog
    }

    fn non_pageable(
        path: Option<PathBuf>,
        tensor_offset: usize,
        tensor_len: usize,
        dtype: DataType,
        layout: ExpertTensorLayout,
        reason: NonPageableReason,
    ) -> Self {
        Self {
            path,
            tensor_offset,
            tensor_len,
            dtype,
            layout,
            regions: Vec::new(),
            pageability: Pageability::NonPageable(reason),
        }
    }

    pub fn pageability(&self) -> &Pageability {
        &self.pageability
    }

    pub fn is_pageable(&self) -> bool {
        matches!(self.pageability, Pageability::Pageable)
    }

    pub fn layout(&self) -> &ExpertTensorLayout {
        &self.layout
    }

    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    pub fn mapped_bytes(&self) -> usize {
        if self.is_pageable() {
            self.tensor_len
        } else {
            0
        }
    }

    pub fn region(&self, expert: usize) -> Option<&ExpertWeightRegion> {
        self.regions.get(expert)
    }

    pub fn relative_range(&self, expert: usize) -> Option<Range<usize>> {
        let region = self.region(expert)?;
        let start = region.offset.checked_sub(self.tensor_offset)?;
        let end = start.checked_add(region.len)?;
        Some(start..end)
    }

    /// Borrow one expert directly from the live read-only mmap.
    pub fn expert_bytes<'a>(&self, store: &'a WeightStore, expert: usize) -> Option<&'a [u8]> {
        let path = self.path.as_ref()?;
        let region = self.region(expert)?;
        store.external_bytes(path, region.offset, region.len)
    }

    pub fn tensor_offset(&self) -> usize {
        self.tensor_offset
    }
}

/// Overflow error shared by loader range catalogs and paging-aware kernels.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
#[error("{0}")]
pub struct WeightRangeError(String);

/// Checked product that still detects overflow when another factor is zero.
pub fn checked_product(factors: &[usize], context: &str) -> Result<usize, WeightRangeError> {
    let mut product = 1usize;
    let mut has_zero = false;
    for &factor in factors {
        if factor == 0 {
            has_zero = true;
        } else {
            product = product
                .checked_mul(factor)
                .ok_or_else(|| WeightRangeError(format!("{context} overflow")))?;
        }
    }
    Ok(if has_zero { 0 } else { product })
}

/// Checked element-size multiplication, rejecting slices larger than `isize`.
pub fn checked_byte_count(
    elements: usize,
    element_size: usize,
    context: &str,
) -> Result<usize, WeightRangeError> {
    let bytes = elements
        .checked_mul(element_size)
        .ok_or_else(|| WeightRangeError(format!("{context} overflow")))?;
    if bytes > isize::MAX as usize {
        return Err(WeightRangeError(format!("{context} exceeds isize::MAX")));
    }
    Ok(bytes)
}

/// Checked storage byte count, including sub-byte ONNX dtypes.
pub fn checked_storage_byte_count(
    dtype: DataType,
    elements: usize,
    context: &str,
) -> Result<usize, WeightRangeError> {
    let bytes = dtype
        .checked_storage_bytes(elements)
        .ok_or_else(|| WeightRangeError(format!("{context} overflow")))?;
    if bytes > isize::MAX as usize {
        return Err(WeightRangeError(format!("{context} exceeds isize::MAX")));
    }
    Ok(bytes)
}

/// Checked fixed-width range `index * width .. start + width`.
pub fn checked_range(
    index: usize,
    width: usize,
    context: &str,
) -> Result<Range<usize>, WeightRangeError> {
    let start = index
        .checked_mul(width)
        .ok_or_else(|| WeightRangeError(format!("{context} start offset overflow")))?;
    let end = start
        .checked_add(width)
        .ok_or_else(|| WeightRangeError(format!("{context} end offset overflow")))?;
    if end > isize::MAX as usize {
        return Err(WeightRangeError(format!("{context} exceeds isize::MAX")));
    }
    Ok(start..end)
}

impl WeightStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Memory-map an external-data file and register it under `path`, so any
    /// [`WeightRef::External`] whose `path` matches resolves zero-copy via
    /// [`bytes`](Self::bytes). Idempotent: mapping the same path twice is a
    /// no-op. This is the programmatic counterpart to the loader path (which
    /// maps files while resolving `TensorProto` initializers), useful when
    /// constructing a store by hand.
    ///
    /// The map is read-only and kept alive for the store's lifetime; callers
    /// must not mutate or unlink the file while the store is live.
    pub fn map_external(&mut self, path: impl AsRef<Path>) -> Result<(), LoaderError> {
        self.mmap_file(path.as_ref())
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

    fn external_bytes(&self, path: &Path, offset: usize, length: usize) -> Option<&[u8]> {
        let mmap = self.mmaps.get(path)?;
        mmap.get(offset..offset.checked_add(length)?)
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
    let dtype =
        DataType::from_onnx(init.data_type).ok_or_else(|| LoaderError::UnsupportedDataType {
            raw: init.data_type,
            context: format!("initializer {:?}", init.name),
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
        DataType::Undefined => {
            return Err(LoaderError::UnsupportedDataType {
                raw: 0,
                context: format!("tensor {:?}", proto.name),
            });
        }
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
        DataType::Complex64 => proto
            .float_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DataType::Complex128 => proto
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
        DataType::Float8E4M3FN
        | DataType::Float8E4M3FNUZ
        | DataType::Float8E5M2
        | DataType::Float8E5M2FNUZ
        | DataType::Float8E8M0
        | DataType::Int4
        | DataType::Uint4
        | DataType::Float4E2M1
        | DataType::Int2
        | DataType::Uint2 => proto.int32_data.iter().map(|v| *v as u8).collect(),
        DataType::String => unreachable!("STRING tensors returned above"),
    };
    Ok(td)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn float8_typed_data_preserves_each_byte() {
        let proto = TensorProto {
            data_type: DataType::Float8E4M3FN.to_onnx(),
            dims: vec![3],
            int32_data: vec![0x01, 0x7f, 0xff],
            ..Default::default()
        };

        let data =
            tensor_data_from_proto(&proto, DataType::Float8E4M3FN, &[3]).expect("tensor data");
        assert_eq!(data.data, [0x01, 0x7f, 0xff]);
    }

    #[test]
    fn four_bit_typed_data_preserves_packed_nibbles() {
        let proto = TensorProto {
            data_type: DataType::Int4.to_onnx(),
            dims: vec![3],
            // ONNX stores two values per int32_data entry: first in the low
            // nibble, second in the high nibble.
            int32_data: vec![0x21, 0x03],
            ..Default::default()
        };

        let data = tensor_data_from_proto(&proto, DataType::Int4, &[3]).expect("tensor data");
        assert_eq!(data.data, [0x21, 0x03]);
    }

    #[test]
    fn two_bit_typed_data_preserves_four_packed_elements_per_byte() {
        let proto = TensorProto {
            data_type: DataType::Int2.to_onnx(),
            dims: vec![5],
            // Elements are packed low-to-high in groups of four.
            int32_data: vec![0b11_10_01_00, 0b0000_0001],
            ..Default::default()
        };

        let data = tensor_data_from_proto(&proto, DataType::Int2, &[5]).expect("tensor data");
        assert_eq!(data.data, [0b11_10_01_00, 0b0000_0001]);
        assert_eq!(data.data.len(), DataType::Int2.storage_bytes(5));
    }

    #[test]
    fn float8e8m0_typed_data_preserves_each_byte() {
        let proto = TensorProto {
            data_type: DataType::Float8E8M0.to_onnx(),
            dims: vec![2],
            int32_data: vec![0x7f, 0xff],
            ..Default::default()
        };

        let data = tensor_data_from_proto(&proto, DataType::Float8E8M0, &[2]).expect("tensor data");
        assert_eq!(data.data, [0x7f, 0xff]);
    }

    fn external_weight(path: PathBuf, length: usize, dims: Vec<usize>) -> WeightRef {
        WeightRef::External {
            path,
            offset: 16,
            length,
            dtype: DataType::Uint8,
            dims,
        }
    }

    fn expert_layout(order: ExpertStorageOrder) -> ExpertTensorLayout {
        ExpertTensorLayout {
            version: 1,
            experts: 3,
            rows_per_expert: 2,
            storage_elements_per_row: 4,
            order,
            quantization: Some(ExpertQuantization {
                bits: 4,
                block_size: 16,
                blocks_per_row: 1,
            }),
        }
    }

    #[test]
    fn expert_major_external_tensor_catalogs_contiguous_ranges() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::current_dir()
            .expect("cwd")
            .join("target")
            .join(format!(
                "weight-region-catalog-{}-{stamp}.bin",
                std::process::id()
            ));
        std::fs::create_dir_all(path.parent().expect("parent")).expect("create target");
        let mut bytes = vec![0u8; 40];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = index as u8;
        }
        std::fs::write(&path, &bytes).expect("write external data");

        let weight = external_weight(path.clone(), 24, vec![3, 2, 4]);
        let catalog =
            WeightRegionCatalog::classify(&weight, expert_layout(ExpertStorageOrder::ExpertMajor));
        assert_eq!(catalog.pageability(), &Pageability::Pageable);
        assert_eq!(catalog.mapped_bytes(), 24);
        assert_eq!(
            catalog.region(1),
            Some(&ExpertWeightRegion {
                expert: 1,
                offset: 24,
                len: 8,
            })
        );

        let mut store = WeightStore::new();
        store.map_external(&path).expect("map external data");
        assert_eq!(catalog.expert_bytes(&store, 1), Some(&bytes[24..32]));
        drop(store);
        std::fs::remove_file(path).expect("remove external data");
    }

    #[test]
    fn interleaved_external_tensor_is_non_pageable_without_error() {
        let weight = external_weight(PathBuf::from("weights.bin"), 24, vec![3, 2, 4]);
        let catalog =
            WeightRegionCatalog::classify(&weight, expert_layout(ExpertStorageOrder::Interleaved));
        assert_eq!(
            catalog.pageability(),
            &Pageability::NonPageable(NonPageableReason::NotExpertMajor)
        );
        assert!(catalog.region(0).is_none());
    }

    #[test]
    fn catalog_range_math_rejects_zero_masked_overflow_and_isize_excess() {
        let overflow = ExpertTensorLayout {
            version: 1,
            experts: 0,
            rows_per_expert: usize::MAX,
            storage_elements_per_row: 2,
            order: ExpertStorageOrder::ExpertMajor,
            quantization: None,
        };
        let weight = external_weight(PathBuf::from("weights.bin"), 0, vec![0, usize::MAX, 2]);
        assert!(matches!(
            WeightRegionCatalog::classify(&weight, overflow).pageability(),
            Pageability::NonPageable(NonPageableReason::Range(message))
                if message.contains("overflow")
        ));

        assert!(
            checked_range(0, isize::MAX as usize + 1, "range")
                .unwrap_err()
                .to_string()
                .contains("isize::MAX")
        );
    }
}
