//! Page table: maps sequences to physical pages.

use crate::{
    Device, KvError, SequenceId,
    fp8::{Fp8Format, decode_f32 as decode_fp8, encode_f32 as encode_fp8},
};
use onnx_genai_metadata::{KvCacheSpec, KvComponentTolerance, LayerPrecisionOverride};
use std::collections::HashMap;

/// Unique page identifier.
pub type PageId = u32;

/// Scalar storage type for KV page tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDType {
    /// 32-bit floating point key/value data.
    F32,
    /// Symmetric signed 8-bit quantized K/V data with external scaling.
    ///
    /// Values are reconstructed as `q as f32 * scale`.
    Int8,
    /// OCP E4M3FN FP8 with a software codec and external scaling.
    Fp8E4M3Fn,
    /// OCP E5M2 FP8 with a software codec and external scaling.
    Fp8E5M2,
}

impl KvDType {
    /// Parse a metadata KV dtype name.
    pub fn from_metadata_name(name: &str) -> Result<Self, KvError> {
        let normalized = name.trim().to_ascii_lowercase().replace('-', "_");
        match normalized.as_str() {
            "float32" | "fp32" | "float16" | "fp16" | "half" | "bfloat16" | "bf16" => Ok(Self::F32),
            "int8" => Ok(Self::Int8),
            "float8_e4m3fn" | "fp8_e4m3fn" | "float8_e4m3" | "fp8_e4m3" => Ok(Self::Fp8E4M3Fn),
            "float8_e5m2" | "fp8_e5m2" => Ok(Self::Fp8E5M2),
            _ => Err(KvError::UnsupportedKvDType(name.to_owned())),
        }
    }

    const fn fp8_format(self) -> Option<Fp8Format> {
        match self {
            Self::Fp8E4M3Fn => Some(Fp8Format::E4M3Fn),
            Self::Fp8E5M2 => Some(Fp8Format::E5M2),
            Self::F32 | Self::Int8 => None,
        }
    }

    const fn is_quantized(self) -> bool {
        !matches!(self, Self::F32)
    }

    const fn precision_rank(self) -> u8 {
        match self {
            Self::F32 => 3,
            Self::Fp8E4M3Fn | Self::Fp8E5M2 => 2,
            Self::Int8 => 1,
        }
    }
}

/// Key/value storage precision for one transformer layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerKvDType {
    pub key: KvDType,
    pub value: KvDType,
}

/// Per-layer KV precision policy derived from inference metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvQuantConfig {
    layers: Vec<LayerKvDType>,
}

impl KvQuantConfig {
    /// Use one dtype for every key and value layer.
    pub fn homogeneous(dtype: KvDType, num_layers: usize) -> Self {
        Self {
            layers: vec![
                LayerKvDType {
                    key: dtype,
                    value: dtype,
                };
                num_layers
            ],
        }
    }

    /// Build a precision policy from `kv_cache` metadata.
    ///
    /// Component defaults override `native_dtype`, per-layer minimum precision
    /// overrides component defaults, and `sensitive_layers` remain in f32.
    pub fn from_metadata(spec: &KvCacheSpec, num_layers: usize) -> Result<Self, KvError> {
        let native_dtype = spec
            .native_dtype
            .as_deref()
            .map(KvDType::from_metadata_name)
            .transpose()?
            .unwrap_or(KvDType::F32);
        let key_tolerance = spec
            .quantization_tolerance
            .as_ref()
            .and_then(|tolerance| tolerance.key.as_ref());
        let value_tolerance = spec
            .quantization_tolerance
            .as_ref()
            .and_then(|tolerance| tolerance.value.as_ref());
        let key_dtype = component_default(key_tolerance, native_dtype)?;
        let value_dtype = component_default(value_tolerance, native_dtype)?;
        let mut config = Self {
            layers: vec![
                LayerKvDType {
                    key: key_dtype,
                    value: value_dtype,
                };
                num_layers
            ],
        };

        apply_layer_overrides(&mut config.layers, key_tolerance, num_layers, KvKind::Key)?;
        apply_layer_overrides(
            &mut config.layers,
            value_tolerance,
            num_layers,
            KvKind::Value,
        )?;
        for &layer in spec.sensitive_layers.as_deref().unwrap_or_default() {
            let layer = resolve_layer_index(layer, num_layers)?;
            config.layers[layer] = LayerKvDType {
                key: KvDType::F32,
                value: KvDType::F32,
            };
        }
        Ok(config)
    }

    pub fn layer(&self, layer: usize) -> Option<LayerKvDType> {
        self.layers.get(layer).copied()
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn dtype(&self, layer: usize, kind: KvKind) -> KvDType {
        let layer = self.layers[layer];
        match kind {
            KvKind::Key => layer.key,
            KvKind::Value => layer.value,
        }
    }
}

fn component_default(
    tolerance: Option<&KvComponentTolerance>,
    fallback: KvDType,
) -> Result<KvDType, KvError> {
    tolerance
        .and_then(|component| component.default.as_deref())
        .map(KvDType::from_metadata_name)
        .transpose()
        .map(|dtype| dtype.unwrap_or(fallback))
}

fn apply_layer_overrides(
    layers: &mut [LayerKvDType],
    tolerance: Option<&KvComponentTolerance>,
    num_layers: usize,
    kind: KvKind,
) -> Result<(), KvError> {
    let Some(overrides) = tolerance.and_then(|component| component.per_layer.as_deref()) else {
        return Ok(());
    };
    for precision_override in overrides {
        apply_layer_override(layers, precision_override, num_layers, kind)?;
    }
    Ok(())
}

fn apply_layer_override(
    layers: &mut [LayerKvDType],
    precision_override: &LayerPrecisionOverride,
    num_layers: usize,
    kind: KvKind,
) -> Result<(), KvError> {
    let dtype = KvDType::from_metadata_name(&precision_override.min_precision)?;
    for &layer in &precision_override.layers {
        let layer = resolve_layer_index(layer, num_layers)?;
        let slot = match kind {
            KvKind::Key => &mut layers[layer].key,
            KvKind::Value => &mut layers[layer].value,
        };
        if dtype.precision_rank() >= slot.precision_rank() {
            *slot = dtype;
        }
    }
    Ok(())
}

fn resolve_layer_index(layer: i32, num_layers: usize) -> Result<usize, KvError> {
    let resolved = if layer < 0 {
        i64::try_from(num_layers).unwrap_or(i64::MAX) + i64::from(layer)
    } else {
        i64::from(layer)
    };
    if resolved < 0 || resolved >= i64::try_from(num_layers).unwrap_or(i64::MAX) {
        return Err(KvError::InvalidKvLayer { layer, num_layers });
    }
    Ok(resolved as usize)
}

/// Tensor geometry and scalar type for one physical page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageTensorConfig {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Tokens per page.
    pub page_size: usize,
    pub dtype: KvDType,
}

impl PageTensorConfig {
    pub fn f32_len_per_page(self) -> usize {
        self.num_layers * 2 * self.num_kv_heads * self.page_size * self.head_dim
    }

    pub fn f32_len_per_token_per_layer(self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    pub fn validate(self) -> bool {
        self.num_layers > 0 && self.num_kv_heads > 0 && self.head_dim > 0 && self.page_size > 0
    }
}

/// K or V selector for page tensor indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvKind {
    Key,
    Value,
}

/// A physical page holding KV data for a fixed number of tokens.
///
/// Logical tensor shape is `[L, 2, H, P, D]`, where axis 1 is `0 = key`,
/// `1 = value`. Physical f32, int8, and fp8 buffers contain only components
/// assigned to that precision. Quantized components use one scale per head.
/// The flat logical offset is:
/// `(((((layer * 2 + kv) * H + head) * P + token_offset) * D) + dim)`.
#[derive(Debug, Clone)]
pub struct Page {
    pub id: PageId,
    /// Number of active references (for CoW).
    pub ref_count: u32,
    /// Which device tier this page lives on.
    pub device: Device,
    /// How many token slots in this page are filled (0..=page_size).
    pub filled: usize,
    /// Last access timestamp (for LRU eviction).
    pub last_access: u64,
    /// Compact page-local f32 storage. Empty when every component is quantized.
    pub data: Vec<f32>,
    /// Compact signed int8 storage.
    pub quantized_data: Vec<i8>,
    /// Compact FP8 bit patterns.
    pub fp8_data: Vec<u8>,
    /// Per-layer, per-K/V, per-head dequantization scales.
    pub quant_scales: Vec<f32>,
    /// First quantization scale, retained for compatibility with int8 callers.
    pub quant_scale: f32,
    storage_layout: Vec<ComponentStorage>,
}

#[derive(Debug, Clone, Copy)]
struct ComponentStorage {
    dtype: KvDType,
    data_offset: usize,
    quantized_offset: usize,
    fp8_offset: usize,
    scale_offset: usize,
}

impl Page {
    fn new(
        id: PageId,
        device: Device,
        config: Option<PageTensorConfig>,
        quant_config: Option<&KvQuantConfig>,
    ) -> Self {
        let mut storage_layout = Vec::new();
        let mut data_len = 0;
        let mut quantized_len = 0;
        let mut fp8_len = 0;
        let mut scale_len = 0;
        if let (Some(config), Some(quant_config)) = (config, quant_config) {
            let component_len = config.num_kv_heads * config.page_size * config.head_dim;
            for layer in 0..config.num_layers {
                for kind in [KvKind::Key, KvKind::Value] {
                    let dtype = quant_config.dtype(layer, kind);
                    storage_layout.push(ComponentStorage {
                        dtype,
                        data_offset: data_len,
                        quantized_offset: quantized_len,
                        fp8_offset: fp8_len,
                        scale_offset: scale_len,
                    });
                    match dtype {
                        KvDType::F32 => data_len += component_len,
                        KvDType::Int8 => {
                            quantized_len += component_len;
                            scale_len += config.num_kv_heads;
                        }
                        KvDType::Fp8E4M3Fn | KvDType::Fp8E5M2 => {
                            fp8_len += component_len;
                            scale_len += config.num_kv_heads;
                        }
                    }
                }
            }
        }
        Self {
            id,
            ref_count: 0,
            device,
            filled: 0,
            last_access: 0,
            data: vec![0.0; data_len],
            quantized_data: vec![0; quantized_len],
            fp8_data: vec![0; fp8_len],
            quant_scales: vec![1.0; scale_len],
            storage_layout,
            quant_scale: 1.0,
        }
    }

    pub fn reset_storage(&mut self, _config: Option<PageTensorConfig>) {
        self.filled = 0;
        self.quant_scale = 1.0;
        self.data.fill(0.0);
        self.quantized_data.fill(0);
        self.fp8_data.fill(0);
        self.quant_scales.fill(1.0);
    }

    pub fn value_at(&self, config: PageTensorConfig, offset: usize) -> f32 {
        let component_len = component_len(config);
        let component = offset / component_len;
        let component_offset = offset % component_len;
        let head = component_offset / (config.page_size * config.head_dim);
        let storage = self.storage_layout[component];
        match storage.dtype {
            KvDType::F32 => self.data[storage.data_offset + component_offset],
            KvDType::Int8 => {
                let scale = self.quant_scales[storage.scale_offset + head];
                f32::from(self.quantized_data[storage.quantized_offset + component_offset]) * scale
            }
            KvDType::Fp8E4M3Fn | KvDType::Fp8E5M2 => {
                let scale = self.quant_scales[storage.scale_offset + head];
                decode_fp8(
                    self.fp8_data[storage.fp8_offset + component_offset],
                    storage.dtype.fp8_format().expect("fp8 dtype"),
                ) * scale
            }
        }
    }

    pub fn dequantized(&self, config: PageTensorConfig) -> Vec<f32> {
        let mut values = vec![0.0; config.f32_len_per_page()];
        for component in 0..self.storage_layout.len() {
            let component_len = component_len(config);
            let logical_offset = component * component_len;
            for component_offset in 0..component_len {
                values[logical_offset + component_offset] =
                    self.value_at(config, logical_offset + component_offset);
            }
        }
        values
    }

    pub fn store_from_f32(&mut self, config: PageTensorConfig, values: &[f32]) {
        assert_eq!(values.len(), config.f32_len_per_page());
        let component_len = component_len(config);
        let head_len = config.page_size * config.head_dim;
        for (component, storage) in self.storage_layout.iter().copied().enumerate() {
            let logical_offset = component * component_len;
            if storage.dtype == KvDType::F32 {
                self.data[storage.data_offset..storage.data_offset + component_len]
                    .copy_from_slice(&values[logical_offset..logical_offset + component_len]);
                continue;
            }
            for head in 0..config.num_kv_heads {
                let logical_head_offset = logical_offset + head * head_len;
                let head_values = &values[logical_head_offset..logical_head_offset + head_len];
                let max_abs = head_values
                    .iter()
                    .filter(|value| value.is_finite())
                    .fold(0.0_f32, |acc, value| acc.max(value.abs()));
                let denominator = match storage.dtype {
                    KvDType::Int8 => 127.0,
                    KvDType::Fp8E4M3Fn | KvDType::Fp8E5M2 => {
                        storage.dtype.fp8_format().expect("fp8 dtype").max_finite()
                    }
                    KvDType::F32 => unreachable!(),
                };
                let scale = if max_abs == 0.0 {
                    1.0
                } else {
                    max_abs / denominator
                };
                self.quant_scales[storage.scale_offset + head] = scale;
                let component_head_offset = head * head_len;
                match storage.dtype {
                    KvDType::Int8 => {
                        let output_offset = storage.quantized_offset + component_head_offset;
                        for (output, value) in self.quantized_data
                            [output_offset..output_offset + head_len]
                            .iter_mut()
                            .zip(head_values)
                        {
                            *output = (value / scale).round().clamp(-127.0, 127.0) as i8;
                        }
                    }
                    KvDType::Fp8E4M3Fn | KvDType::Fp8E5M2 => {
                        let output_offset = storage.fp8_offset + component_head_offset;
                        let format = storage.dtype.fp8_format().expect("fp8 dtype");
                        for (output, value) in self.fp8_data
                            [output_offset..output_offset + head_len]
                            .iter_mut()
                            .zip(head_values)
                        {
                            *output = encode_fp8(*value / scale, format);
                        }
                    }
                    KvDType::F32 => unreachable!(),
                }
            }
        }
        self.quant_scale = self.quant_scales.first().copied().unwrap_or(1.0);
    }

    pub fn has_quantized_storage(&self) -> bool {
        self.storage_layout
            .iter()
            .any(|storage| storage.dtype.is_quantized())
    }
}

const fn component_len(config: PageTensorConfig) -> usize {
    config.num_kv_heads * config.page_size * config.head_dim
}

/// The page table manages the mapping from logical sequences to physical pages.
pub struct PageTable {
    /// Logical sequence → ordered list of page IDs.
    pub sequences: HashMap<SequenceId, Vec<PageId>>,
    /// Logical sequence → current token length.
    pub sequence_lengths: HashMap<SequenceId, usize>,
    /// Logical sequence → absolute position of the first retained token.
    pub sequence_starts: HashMap<SequenceId, usize>,
    /// All physical pages.
    pub pages: HashMap<PageId, Page>,
    /// Free pages per device.
    free_pages: HashMap<Device, Vec<PageId>>,
    /// Tokens per page.
    pub page_size: usize,
    /// Optional tensor storage layout.
    pub tensor_config: Option<PageTensorConfig>,
    /// Per-layer key/value precision policy.
    pub quant_config: Option<KvQuantConfig>,
    /// Monotonic clock for LRU.
    clock: u64,
    /// Maximum number of live pages allowed in the hot tier.
    hot_capacity: usize,
    /// Next page id for cold-offload-backed growth beyond the initial hot pool.
    next_page_id: PageId,
}

impl PageTable {
    pub fn new(page_size: usize, num_gpu_pages: usize) -> Self {
        Self::new_with_storage(page_size, num_gpu_pages, None, None)
    }

    pub fn new_with_tensor_config(
        page_size: usize,
        num_gpu_pages: usize,
        tensor_config: Option<PageTensorConfig>,
    ) -> Self {
        let quant_config =
            tensor_config.map(|config| KvQuantConfig::homogeneous(config.dtype, config.num_layers));
        Self::new_with_storage(page_size, num_gpu_pages, tensor_config, quant_config)
    }

    pub fn new_with_quant_config(
        page_size: usize,
        num_gpu_pages: usize,
        tensor_config: PageTensorConfig,
        quant_config: KvQuantConfig,
    ) -> Result<Self, KvError> {
        if quant_config.num_layers() != tensor_config.num_layers {
            return Err(KvError::InvalidQuantizationConfig(
                "quantization layer count must match tensor config".to_owned(),
            ));
        }
        Ok(Self::new_with_storage(
            page_size,
            num_gpu_pages,
            Some(tensor_config),
            Some(quant_config),
        ))
    }

    fn new_with_storage(
        page_size: usize,
        num_gpu_pages: usize,
        tensor_config: Option<PageTensorConfig>,
        quant_config: Option<KvQuantConfig>,
    ) -> Self {
        if let Some(config) = tensor_config {
            assert_eq!(
                page_size, config.page_size,
                "page_size must match tensor config"
            );
            assert!(config.validate(), "invalid page tensor config");
        }

        let mut pages = HashMap::new();
        let mut free_pages = vec![];
        for i in 0..num_gpu_pages {
            let id = i as PageId;
            pages.insert(
                id,
                Page::new(id, Device::Gpu(0), tensor_config, quant_config.as_ref()),
            );
            free_pages.push(id);
        }

        let mut free_map = HashMap::new();
        free_map.insert(Device::Gpu(0), free_pages);

        Self {
            sequences: HashMap::new(),
            sequence_lengths: HashMap::new(),
            sequence_starts: HashMap::new(),
            pages,
            free_pages: free_map,
            page_size,
            tensor_config,
            quant_config,
            clock: 0,
            hot_capacity: num_gpu_pages,
            next_page_id: num_gpu_pages as PageId,
        }
    }

    /// Allocate a new page on the specified device.
    pub fn allocate(&mut self, device: Device) -> Option<PageId> {
        if matches!(device, Device::Gpu(_))
            && self.free_count(device) == 0
            && self.hot_used_count() >= self.hot_capacity
        {
            self.evict_lru_hot(None).ok()?;
        }

        if let Some(free_list) = self.free_pages.get_mut(&device)
            && let Some(page_id) = free_list.pop()
        {
            if let Some(page) = self.pages.get_mut(&page_id) {
                page.ref_count = 1;
                page.device = device;
                page.reset_storage(self.tensor_config);
                self.clock += 1;
                page.last_access = self.clock;
            }
            return Some(page_id);
        }
        if matches!(device, Device::Gpu(_)) && self.hot_used_count() < self.hot_capacity {
            let page_id = self.next_page_id;
            self.next_page_id = self.next_page_id.saturating_add(1);
            let mut page = Page::new(
                page_id,
                device,
                self.tensor_config,
                self.quant_config.as_ref(),
            );
            page.ref_count = 1;
            self.clock += 1;
            page.last_access = self.clock;
            self.pages.insert(page_id, page);
            return Some(page_id);
        }
        None
    }

    /// Free a page (decrement ref_count; actually free when it hits 0).
    pub fn free(&mut self, page_id: PageId) {
        if let Some(page) = self.pages.get_mut(&page_id) {
            page.ref_count = page.ref_count.saturating_sub(1);
            if page.ref_count == 0 {
                page.reset_storage(self.tensor_config);
                let device = page.device;
                self.free_pages.entry(device).or_default().push(page_id);
            }
        }
    }

    /// Increment a page reference for CoW/prefix sharing.
    pub fn retain(&mut self, page_id: PageId) -> bool {
        if let Some(page) = self.pages.get_mut(&page_id) {
            page.ref_count = page.ref_count.saturating_add(1);
            self.clock += 1;
            page.last_access = self.clock;
            true
        } else {
            false
        }
    }

    /// Get the page list for a sequence.
    pub fn get_sequence(&self, seq: SequenceId) -> Option<&[PageId]> {
        self.sequences.get(&seq).map(|v| v.as_slice())
    }

    pub fn sequence_len(&self, seq: SequenceId) -> Option<usize> {
        self.sequence_lengths.get(&seq).copied()
    }

    pub fn sequence_start(&self, seq: SequenceId) -> Option<usize> {
        self.sequence_starts.get(&seq).copied()
    }

    pub fn set_sequence_len(&mut self, seq: SequenceId, len: usize) {
        if let Some(slot) = self.sequence_lengths.get_mut(&seq) {
            *slot = len;
        }
    }

    pub fn set_sequence_start(&mut self, seq: SequenceId, start: usize) {
        if let Some(slot) = self.sequence_starts.get_mut(&seq) {
            *slot = start;
        }
    }

    /// Create a new sequence (empty).
    pub fn create_sequence(&mut self, seq: SequenceId) {
        self.sequences.insert(seq, Vec::new());
        self.sequence_lengths.insert(seq, 0);
        self.sequence_starts.insert(seq, 0);
    }

    /// Append a page to a sequence.
    pub fn push_page(&mut self, seq: SequenceId, page_id: PageId) {
        if let Some(pages) = self.sequences.get_mut(&seq) {
            pages.push(page_id);
        }
    }

    /// Replace a sequence page at `logical_page_index`.
    pub fn replace_page(&mut self, seq: SequenceId, logical_page_index: usize, page_id: PageId) {
        if let Some(pages) = self.sequences.get_mut(&seq)
            && let Some(slot) = pages.get_mut(logical_page_index)
        {
            *slot = page_id;
        }
    }

    pub fn touch(&mut self, page_id: PageId) {
        if let Some(page) = self.pages.get_mut(&page_id) {
            self.clock += 1;
            page.last_access = self.clock;
        }
    }

    /// Promote a page to the hot tier, evicting the hot LRU page when needed.
    pub fn promote_to_hot(&mut self, page_id: PageId) -> Result<(), KvError> {
        let Some(page) = self.pages.get(&page_id) else {
            return Err(KvError::PageNotFound(page_id));
        };
        if matches!(page.device, Device::Gpu(_)) {
            self.touch(page_id);
            return Ok(());
        }
        if self.hot_capacity == 0 {
            return Err(KvError::OutOfMemory {
                needed: 1,
                available: 0,
            });
        }
        if self.hot_used_count() >= self.hot_capacity {
            self.evict_lru_hot(Some(page_id))?;
        }
        let page = self
            .pages
            .get_mut(&page_id)
            .ok_or(KvError::PageNotFound(page_id))?;
        page.device = Device::Gpu(0);
        self.clock += 1;
        page.last_access = self.clock;
        Ok(())
    }

    /// Evict the least-recently-used hot page to the cold CPU tier.
    pub fn evict_lru_hot(&mut self, exclude: Option<PageId>) -> Result<PageId, KvError> {
        let Some((&victim_id, _)) = self
            .pages
            .iter()
            .filter(|(id, page)| {
                Some(**id) != exclude && page.ref_count > 0 && matches!(page.device, Device::Gpu(_))
            })
            .min_by_key(|(_, page)| page.last_access)
        else {
            return Err(KvError::OutOfMemory {
                needed: 1,
                available: 0,
            });
        };
        let victim = self
            .pages
            .get_mut(&victim_id)
            .ok_or(KvError::PageNotFound(victim_id))?;
        victim.device = Device::Cpu;
        Ok(victim_id)
    }

    pub fn tensor_offset(
        &self,
        layer: usize,
        kind: KvKind,
        head: usize,
        token_offset: usize,
        dim: usize,
    ) -> Option<usize> {
        let config = self.tensor_config?;
        if layer >= config.num_layers
            || head >= config.num_kv_heads
            || token_offset >= config.page_size
            || dim >= config.head_dim
        {
            return None;
        }
        let kv = match kind {
            KvKind::Key => 0,
            KvKind::Value => 1,
        };
        Some(
            ((((layer * 2 + kv) * config.num_kv_heads + head) * config.page_size + token_offset)
                * config.head_dim)
                + dim,
        )
    }

    /// Remove a sequence and return its pages.
    pub fn remove_sequence(&mut self, seq: SequenceId) -> Vec<PageId> {
        self.sequence_lengths.remove(&seq);
        self.sequence_starts.remove(&seq);
        self.sequences.remove(&seq).unwrap_or_default()
    }

    /// Number of free pages on a device.
    pub fn free_count(&self, device: Device) -> usize {
        self.free_pages.get(&device).map_or(0, |v| v.len())
    }

    /// Number of referenced pages resident in the hot tier.
    pub fn hot_used_count(&self) -> usize {
        self.pages
            .values()
            .filter(|page| page.ref_count > 0 && matches!(page.device, Device::Gpu(_)))
            .count()
    }

    /// Configured hot-tier live page capacity.
    pub fn hot_capacity(&self) -> usize {
        self.hot_capacity
    }

    /// Total number of pages.
    pub fn total_pages(&self) -> usize {
        self.pages.len()
    }
}
