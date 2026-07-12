//! Page table: maps sequences to physical pages.

use crate::{Device, KvError, SequenceId};
use std::collections::HashMap;

/// Unique page identifier.
pub type PageId = u32;

/// Scalar storage type for KV page tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDType {
    /// 32-bit floating point key/value data.
    F32,
    /// Symmetric signed 8-bit quantized K/V data with one scale per page.
    ///
    /// Values are reconstructed as `q as f32 * scale`. This is intentionally
    /// an integer scheme rather than fp8 e4m3 so the cache can bound error
    /// without depending on target-specific fp8 support.
    Int8,
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
/// For `PageTensorConfig { L, H, P, D, F32 }`, `data` is a contiguous f32
/// buffer with shape `[L, 2, H, P, D]`, where axis 1 is `0 = key`,
/// `1 = value`. For `KvDType::Int8`, `data` is empty and `quantized_data`
/// stores the same logical layout with `quant_scale`. The flat offset is:
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
    /// Contiguous page-local KV tensor storage. Empty for count-only caches.
    pub data: Vec<f32>,
    /// Quantized page-local KV tensor storage for `KvDType::Int8`.
    pub quantized_data: Vec<i8>,
    /// Symmetric dequantization scale for `quantized_data`.
    pub quant_scale: f32,
}

impl Page {
    fn new(id: PageId, device: Device, page_f32_len: usize, dtype: Option<KvDType>) -> Self {
        let (data, quantized_data, quant_scale) = match dtype {
            Some(KvDType::F32) => (vec![0.0; page_f32_len], Vec::new(), 1.0),
            Some(KvDType::Int8) => (Vec::new(), vec![0; page_f32_len], 1.0),
            None => (Vec::new(), Vec::new(), 1.0),
        };
        Self {
            id,
            ref_count: 0,
            device,
            filled: 0,
            last_access: 0,
            data,
            quantized_data,
            quant_scale,
        }
    }

    pub fn reset_storage(&mut self, config: Option<PageTensorConfig>) {
        self.filled = 0;
        self.quant_scale = 1.0;
        match config.map(|c| c.dtype) {
            Some(KvDType::F32) => self.data.fill(0.0),
            Some(KvDType::Int8) => self.quantized_data.fill(0),
            None => {}
        }
    }

    pub fn value_at(&self, config: PageTensorConfig, offset: usize) -> f32 {
        match config.dtype {
            KvDType::F32 => self.data[offset],
            KvDType::Int8 => self.quantized_data[offset] as f32 * self.quant_scale,
        }
    }

    pub fn dequantized(&self, config: PageTensorConfig) -> Vec<f32> {
        match config.dtype {
            KvDType::F32 => self.data.clone(),
            KvDType::Int8 => self
                .quantized_data
                .iter()
                .map(|&q| q as f32 * self.quant_scale)
                .collect(),
        }
    }

    pub fn store_from_f32(&mut self, config: PageTensorConfig, values: &[f32]) {
        match config.dtype {
            KvDType::F32 => {
                self.data.clear();
                self.data.extend_from_slice(values);
            }
            KvDType::Int8 => {
                let max_abs = values
                    .iter()
                    .fold(0.0_f32, |acc, value| acc.max(value.abs()));
                self.quant_scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
                self.quantized_data.clear();
                self.quantized_data.extend(values.iter().map(|value| {
                    (value / self.quant_scale)
                        .round()
                        .clamp(i8::MIN as f32, i8::MAX as f32) as i8
                }));
                self.data.clear();
            }
        }
    }
}

/// The page table manages the mapping from logical sequences to physical pages.
pub struct PageTable {
    /// Logical sequence → ordered list of page IDs.
    pub sequences: HashMap<SequenceId, Vec<PageId>>,
    /// Logical sequence → current token length.
    pub sequence_lengths: HashMap<SequenceId, usize>,
    /// All physical pages.
    pub pages: HashMap<PageId, Page>,
    /// Free pages per device.
    free_pages: HashMap<Device, Vec<PageId>>,
    /// Tokens per page.
    pub page_size: usize,
    /// Optional tensor storage layout.
    pub tensor_config: Option<PageTensorConfig>,
    /// Monotonic clock for LRU.
    clock: u64,
    /// Maximum number of live pages allowed in the hot tier.
    hot_capacity: usize,
    /// Next page id for cold-offload-backed growth beyond the initial hot pool.
    next_page_id: PageId,
}

impl PageTable {
    pub fn new(page_size: usize, num_gpu_pages: usize) -> Self {
        Self::new_with_tensor_config(page_size, num_gpu_pages, None)
    }

    pub fn new_with_tensor_config(
        page_size: usize,
        num_gpu_pages: usize,
        tensor_config: Option<PageTensorConfig>,
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
        let page_f32_len = tensor_config.map_or(0, PageTensorConfig::f32_len_per_page);
        let dtype = tensor_config.map(|config| config.dtype);

        for i in 0..num_gpu_pages {
            let id = i as PageId;
            pages.insert(id, Page::new(id, Device::Gpu(0), page_f32_len, dtype));
            free_pages.push(id);
        }

        let mut free_map = HashMap::new();
        free_map.insert(Device::Gpu(0), free_pages);

        Self {
            sequences: HashMap::new(),
            sequence_lengths: HashMap::new(),
            pages,
            free_pages: free_map,
            page_size,
            tensor_config,
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
                self.tensor_config
                    .map_or(0, PageTensorConfig::f32_len_per_page),
                self.tensor_config.map(|config| config.dtype),
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

    pub fn set_sequence_len(&mut self, seq: SequenceId, len: usize) {
        if let Some(slot) = self.sequence_lengths.get_mut(&seq) {
            *slot = len;
        }
    }

    /// Create a new sequence (empty).
    pub fn create_sequence(&mut self, seq: SequenceId) {
        self.sequences.insert(seq, Vec::new());
        self.sequence_lengths.insert(seq, 0);
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
