//! Page table: maps sequences to physical pages.

use crate::{
    Device, KvError, SequenceId,
    fp8::{Fp8Format, decode_f32 as decode_fp8, encode_f32 as encode_fp8},
    telemetry::KvTelemetry,
};
use std::sync::Arc;
use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    mem::size_of,
    ptr::NonNull,
};

/// Unique page identifier.
pub type PageId = u32;

/// Opaque device-resident page memory exposed for kernel binding.
///
/// The pointer is never dereferenced by `onnx-genai-kv`; it is only a stable
/// handle that a backend can pass to its kernels together with the byte range
/// occupied by this logical page.
///
/// This span is **not** a staging post for a CUDA store inside this crate.
/// #721 stage 3 (`CudaPageStore` backed by the VMM arena) is superseded: on
/// native CUDA, device KV paging is owned by the VMM layer -- `CudaVmmAllocator`
/// with its physical-handle pool (#740), committed-granule admission (#745) and
/// growth grants (#748) -- which already reserves a stable virtual range and
/// maps committed granules behind it. Building a second page allocator here
/// would duplicate that ownership, not complete it. The span stays because the
/// trait boundary is what lets a third party supply a device store without
/// patching this crate, and because it is the reason `head_token_row()` is not
/// on the store contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevicePageSpan {
    pub device: Device,
    pub ptr: NonNull<std::ffi::c_void>,
    pub byte_offset: usize,
    pub byte_len: usize,
}

/// Physical storage for one logical KV page.
///
/// The boundary is intentionally asymmetric: host-addressable stores may expose
/// borrowed host slices, while device stores expose only opaque device spans.
/// Code that needs a host view of a device page must perform an explicit
/// materialization before it can obtain slices; that copy is visible to the
/// caller and can be charged to the memory governor instead of being hidden
/// behind an accessor such as `head_token_row()`.
pub trait KvPageStore: fmt::Debug + Send + Sync {
    /// Declared cache residency. A store may be host-addressable even when this
    /// is a GPU emulation location; callers must use `host_view` to determine
    /// addressability rather than inferring it from this value.
    fn residency(&self) -> Device;
    fn allocated_bytes(&self) -> u64;
    fn reset_storage(&mut self);
    fn host_view(&self) -> Option<HostPageStoreView<'_>>;
    fn host_view_mut(&mut self) -> Option<HostPageStoreViewMut<'_>>;
    fn device_span(&self) -> Option<DevicePageSpan>;
    /// Copy this store's complete payload into an already allocated target.
    fn copy_to(&self, target: &mut dyn KvPageStore) -> Result<u64, KvError>;
    /// Accept a complete payload from host-addressable storage.
    ///
    /// A future device store can implement this with an explicit host-to-device
    /// transfer. The default refuses rather than hiding a copy.
    fn copy_from_host(&mut self, _source: HostPageStoreView<'_>) -> Result<(), KvError> {
        Err(KvError::PageStoreCopyUnsupported {
            from: Device::Cpu,
            to: self.residency(),
        })
    }
    fn clone_store(&self) -> Box<dyn KvPageStore>;
}

/// Storage lengths needed to allocate one physical page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageStoreLayout {
    pub f32_len: usize,
    pub int8_len: usize,
    pub fp8_len: usize,
    pub scale_len: usize,
}

impl PageStoreLayout {
    pub fn host_allocated_bytes(self) -> u64 {
        (self.f32_len as u64 + self.scale_len as u64) * size_of::<f32>() as u64
            + self.int8_len as u64
            + self.fp8_len as u64
    }
}

/// Creates an empty target store for a transactional page migration.
///
/// Cache callers depend only on this contract, which is why a third-party or
/// out-of-tree store can be supplied without changing them. It is not a hook
/// waiting for an in-crate CUDA store: #721 stage 3 is superseded, and device
/// KV paging on native CUDA belongs to the VMM layer (`CudaVmmAllocator`,
/// #740/#745/#748) rather than to this crate.
pub trait KvPageStoreFactory: fmt::Debug + Send + Sync {
    /// Stable backend name used in actionable migration diagnostics.
    fn backend_name(&self) -> &'static str {
        "custom-page-store"
    }

    /// Fail before allocation when this backend cannot represent `layout` at
    /// `residency`. The source page remains authoritative on any error.
    fn validate_target(&self, _residency: Device, _layout: PageStoreLayout) -> Result<(), KvError> {
        Ok(())
    }

    /// Maximum bytes allocated by `create` for this layout and residency.
    ///
    /// The page table reserves this amount before calling `create`, making it
    /// impossible for a governed migration to allocate first and account later.
    fn allocation_bytes(&self, residency: Device, layout: PageStoreLayout) -> u64;
    fn create(
        &self,
        residency: Device,
        layout: PageStoreLayout,
    ) -> Result<Box<dyn KvPageStore>, KvError>;
}

/// Completed migration details for telemetry and future transfer accounting.
///
/// Replacement allocation is already covered by a transient sibling lease
/// before this result can be returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageMigration {
    pub from: Device,
    pub to: Device,
    pub bytes_copied: u64,
}

impl Clone for Box<dyn KvPageStore> {
    fn clone(&self) -> Self {
        self.clone_store()
    }
}

/// Borrowed host slices for a page store.
#[derive(Debug, Clone, Copy)]
pub struct HostPageStoreView<'a> {
    pub data: &'a [f32],
    pub quantized_data: &'a [i8],
    pub fp8_data: &'a [u8],
    pub quant_scales: &'a [f32],
}

/// Mutable host slices for a page store.
#[derive(Debug)]
pub struct HostPageStoreViewMut<'a> {
    pub data: &'a mut [f32],
    pub quantized_data: &'a mut [i8],
    pub fp8_data: &'a mut [u8],
    pub quant_scales: &'a mut [f32],
}

/// Host-addressable storage for one logical KV page.
///
/// Both hot and cold tiers currently use this store. Its residency is therefore
/// an emulated cache location, not a claim that GPU-labelled bytes are device
/// memory. `host_view` remains available for both locations.
#[derive(Debug, Clone)]
pub struct HostPageStore {
    residency: Device,
    pub data: Vec<f32>,
    pub quantized_data: Vec<i8>,
    pub fp8_data: Vec<u8>,
    pub quant_scales: Vec<f32>,
}

impl HostPageStore {
    fn new(residency: Device, layout: PageStoreLayout) -> Self {
        Self {
            residency,
            data: vec![0.0; layout.f32_len],
            quantized_data: vec![0; layout.int8_len],
            fp8_data: vec![0; layout.fp8_len],
            quant_scales: vec![1.0; layout.scale_len],
        }
    }
}

impl KvPageStore for HostPageStore {
    fn residency(&self) -> Device {
        self.residency
    }

    fn allocated_bytes(&self) -> u64 {
        let f32_bytes = size_of::<f32>() as u64;
        (self.data.len() as u64) * f32_bytes
            + (self.quantized_data.len() as u64)
            + (self.fp8_data.len() as u64)
            + (self.quant_scales.len() as u64) * f32_bytes
    }

    fn reset_storage(&mut self) {
        self.data.fill(0.0);
        self.quantized_data.fill(0);
        self.fp8_data.fill(0);
        self.quant_scales.fill(1.0);
    }

    fn host_view(&self) -> Option<HostPageStoreView<'_>> {
        Some(HostPageStoreView {
            data: &self.data,
            quantized_data: &self.quantized_data,
            fp8_data: &self.fp8_data,
            quant_scales: &self.quant_scales,
        })
    }

    fn host_view_mut(&mut self) -> Option<HostPageStoreViewMut<'_>> {
        Some(HostPageStoreViewMut {
            data: &mut self.data,
            quantized_data: &mut self.quantized_data,
            fp8_data: &mut self.fp8_data,
            quant_scales: &mut self.quant_scales,
        })
    }

    fn device_span(&self) -> Option<DevicePageSpan> {
        None
    }

    fn copy_to(&self, target: &mut dyn KvPageStore) -> Result<u64, KvError> {
        target.copy_from_host(self.host_view().expect("host store"))?;
        Ok(self.allocated_bytes())
    }

    fn copy_from_host(&mut self, source: HostPageStoreView<'_>) -> Result<(), KvError> {
        if self.data.len() != source.data.len()
            || self.quantized_data.len() != source.quantized_data.len()
            || self.fp8_data.len() != source.fp8_data.len()
            || self.quant_scales.len() != source.quant_scales.len()
        {
            return Err(KvError::PageStoreLayoutMismatch);
        }
        self.data.copy_from_slice(source.data);
        self.quantized_data.copy_from_slice(source.quantized_data);
        self.fp8_data.copy_from_slice(source.fp8_data);
        self.quant_scales.copy_from_slice(source.quant_scales);
        Ok(())
    }

    fn clone_store(&self) -> Box<dyn KvPageStore> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Default)]
pub struct HostPageStoreFactory;

impl KvPageStoreFactory for HostPageStoreFactory {
    fn backend_name(&self) -> &'static str {
        "host-page-store"
    }

    fn validate_target(&self, residency: Device, _layout: PageStoreLayout) -> Result<(), KvError> {
        if residency == Device::Disk {
            return Err(KvError::PageStoreAllocationFailed(
                "host-page-store has no disk page representation".to_owned(),
            ));
        }
        Ok(())
    }

    fn allocation_bytes(&self, _residency: Device, layout: PageStoreLayout) -> u64 {
        layout.host_allocated_bytes()
    }

    fn create(
        &self,
        residency: Device,
        layout: PageStoreLayout,
    ) -> Result<Box<dyn KvPageStore>, KvError> {
        Ok(Box::new(HostPageStore::new(residency, layout)))
    }
}

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

    /// Build a precision policy from a runtime-owned KV storage policy.
    ///
    /// Cache storage precision is a deployment decision, not a package fact:
    /// the same model runs with an f32, f16, or fp8 cache depending on the
    /// runtime's memory budget and accuracy target. `native_dtype` is the
    /// graph-visible dtype at the model's past/present ports and is the
    /// fallback when the policy expresses no preference.
    pub fn from_policy(
        policy: &KvQuantPolicy,
        native_dtype: KvDType,
        num_layers: usize,
    ) -> Result<Self, KvError> {
        policy.validate_axis()?;
        let mut config = Self {
            layers: vec![
                LayerKvDType {
                    key: policy.key.default.unwrap_or(native_dtype),
                    value: policy.value.default.unwrap_or(native_dtype),
                };
                num_layers
            ],
        };
        apply_layer_overrides(&mut config.layers, &policy.key, num_layers, KvKind::Key)?;
        apply_layer_overrides(&mut config.layers, &policy.value, num_layers, KvKind::Value)?;
        for &layer in &policy.high_precision_layers {
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

/// Runtime-owned KV cache storage precision policy.
///
/// This is deployment policy, not model metadata: it describes how a particular
/// runtime chooses to store cache bytes. The package never selects it, and the
/// runtime validates it against the graph-visible dtype before use.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvQuantPolicy {
    /// Storage precision for keys.
    pub key: KvComponentPolicy,
    /// Storage precision for values.
    pub value: KvComponentPolicy,
    /// Layers pinned to f32 regardless of component defaults; negative indices
    /// count from the end.
    pub high_precision_layers: Vec<i32>,
    /// Axis the runtime derives quantization scales along.
    pub quantization_axis: KvQuantAxis,
}

impl KvQuantPolicy {
    /// Only per-token scales satisfy the append invariant.
    fn validate_axis(&self) -> Result<(), KvError> {
        match self.quantization_axis {
            KvQuantAxis::PerToken => Ok(()),
            KvQuantAxis::PerChannel => Err(KvError::UnsupportedQuantizationAxis(
                "per_channel".to_string(),
            )),
        }
    }
}

/// Storage precision policy for one KV component (keys or values).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvComponentPolicy {
    /// Precision for every layer without an override.
    pub default: Option<KvDType>,
    /// Per-layer minimum precision, applied over the default.
    pub per_layer: Vec<LayerPrecisionRule>,
}

/// Minimum storage precision for a set of layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerPrecisionRule {
    /// Layer indices; negative indices count from the end.
    pub layers: Vec<i32>,
    /// Precision floor for those layers.
    pub min_precision: KvDType,
}

/// Axis a runtime derives KV quantization scales along.
///
/// Only per-token quantization (one scale per token, computed across
/// `head_dim`) can satisfy the append invariant that previously-stored tokens
/// are never requantized. Per-channel quantization derives each scale across
/// the token axis, so appending a new token would change the scale and force a
/// rewrite of every stored token; it is rejected explicitly rather than
/// silently ignored.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KvQuantAxis {
    /// One scale per token, computed across `head_dim`.
    #[default]
    PerToken,
    /// One scale per channel, computed across the token axis. Unsupported.
    PerChannel,
}

fn apply_layer_overrides(
    layers: &mut [LayerKvDType],
    policy: &KvComponentPolicy,
    num_layers: usize,
    kind: KvKind,
) -> Result<(), KvError> {
    for rule in &policy.per_layer {
        apply_layer_override(layers, rule, num_layers, kind)?;
    }
    Ok(())
}

fn apply_layer_override(
    layers: &mut [LayerKvDType],
    rule: &LayerPrecisionRule,
    num_layers: usize,
    kind: KvKind,
) -> Result<(), KvError> {
    let dtype = rule.min_precision;
    for &layer in &rule.layers {
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

/// KV tensor geometry (`num_kv_heads` × `head_dim`) for a single layer.
///
/// Different transformer layers may declare different geometry (e.g. Gemma-4
/// E2B uses `head_dim` 256 for sliding/local layers and 512 for full/global
/// layers). The page table stores one of these per layer so that page sizing,
/// writes, and materialization all use the correct per-layer byte stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerTensorConfig {
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl LayerTensorConfig {
    pub const fn new(num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_kv_heads,
            head_dim,
        }
    }

    /// Number of f32 scalars for one token of this layer's key (or value).
    pub const fn f32_len_per_token(self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    pub const fn validate(self) -> bool {
        self.num_kv_heads > 0 && self.head_dim > 0
    }
}

/// Tensor geometry and scalar type for one physical page.
///
/// This is the *uniform* descriptor: it assumes every layer shares
/// `num_kv_heads`/`head_dim`. Heterogeneous models are configured through
/// [`PagedKvCache::new_with_layer_tensor_configs`](crate::PagedKvCache) /
/// [`PageTable::new_with_layer_configs`], which carry a per-layer
/// [`LayerTensorConfig`]. When a heterogeneous cache is built, its retained
/// `PageTensorConfig` reports layer 0's geometry so uniform callers keep
/// compiling; the authoritative per-layer geometry lives in
/// [`PageTable::layer_configs`].
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

    /// Expand this uniform descriptor into one [`LayerTensorConfig`] per layer.
    pub fn to_layer_configs(self) -> Vec<LayerTensorConfig> {
        vec![LayerTensorConfig::new(self.num_kv_heads, self.head_dim); self.num_layers]
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
    /// How many token slots in this page are filled (0..=page_size).
    pub filled: usize,
    /// Last access timestamp (for LRU eviction).
    pub last_access: u64,
    /// Physical page storage.
    store: Box<dyn KvPageStore>,
    store_layout: PageStoreLayout,
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
        page_size: usize,
        layer_configs: &[LayerTensorConfig],
        quant_config: Option<&KvQuantConfig>,
    ) -> Self {
        let mut storage_layout = Vec::new();
        let mut data_len = 0;
        let mut quantized_len = 0;
        let mut fp8_len = 0;
        let mut scale_len = 0;
        if let Some(quant_config) = quant_config
            && !layer_configs.is_empty()
        {
            for (layer, geom) in layer_configs.iter().enumerate() {
                // Per-layer geometry: heterogeneous head_dim / num_kv_heads mean
                // each component (layer, key|value) can have a different length.
                let component_len = geom.num_kv_heads * page_size * geom.head_dim;
                let scale_slots = geom.num_kv_heads * page_size;
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
                            scale_len += scale_slots;
                        }
                        KvDType::Fp8E4M3Fn | KvDType::Fp8E5M2 => {
                            fp8_len += component_len;
                            scale_len += scale_slots;
                        }
                    }
                }
            }
        }
        let store_layout = PageStoreLayout {
            f32_len: data_len,
            int8_len: quantized_len,
            fp8_len,
            scale_len,
        };
        Self {
            id,
            ref_count: 0,
            filled: 0,
            last_access: 0,
            store: HostPageStoreFactory
                .create(device, store_layout)
                .expect("host store allocation is infallible"),
            store_layout,
            storage_layout,
        }
    }

    /// Declared residency of the store that owns this page's bytes.
    pub fn residency(&self) -> Device {
        self.store.residency()
    }

    /// Borrow host storage only when the owning store is host-addressable.
    pub fn host_view(&self) -> Option<HostPageStoreView<'_>> {
        self.store.host_view()
    }

    pub fn device_span(&self) -> Option<DevicePageSpan> {
        self.store.device_span()
    }

    pub(crate) fn clone_physical_store(&self) -> Box<dyn KvPageStore> {
        self.store.clone()
    }

    pub(crate) fn copy_physical_store_from(
        &mut self,
        source: &dyn KvPageStore,
    ) -> Result<u64, KvError> {
        source.copy_to(self.store.as_mut())
    }

    /// Transactionally migrate the complete physical payload to `target`.
    ///
    /// Allocation and copying finish before the owning store is replaced, so
    /// any error leaves the source store and all logical page metadata intact.
    fn migrate(
        &mut self,
        target: Device,
        factory: &dyn KvPageStoreFactory,
    ) -> Result<PageMigration, KvError> {
        let from = self.residency();
        if from == target {
            return Ok(PageMigration {
                from,
                to: target,
                bytes_copied: 0,
            });
        }
        let storage_types = self.storage_type_summary();
        let migration_error = |error: KvError| KvError::StateMigrationFailed {
            page_id: self.id,
            storage_types: storage_types.clone(),
            backend: factory.backend_name(),
            from,
            to: target,
            reason: error.to_string(),
        };
        factory
            .validate_target(target, self.store_layout)
            .map_err(migration_error)?;
        let mut replacement = factory
            .create(target, self.store_layout)
            .map_err(migration_error)?;
        if replacement.residency() != target {
            return Err(migration_error(KvError::PageStoreWrongResidency {
                requested: target,
                actual: replacement.residency(),
            }));
        }
        let bytes_copied = self
            .store
            .copy_to(replacement.as_mut())
            .map_err(migration_error)?;
        self.store = replacement;
        Ok(PageMigration {
            from,
            to: target,
            bytes_copied,
        })
    }

    fn storage_type_summary(&self) -> String {
        let mut types = Vec::new();
        for storage in &self.storage_layout {
            let name = match storage.dtype {
                KvDType::F32 => "f32",
                KvDType::Int8 => "int8",
                KvDType::Fp8E4M3Fn => "fp8-e4m3fn",
                KvDType::Fp8E5M2 => "fp8-e5m2",
            };
            if !types.contains(&name) {
                types.push(name);
            }
        }
        if types.is_empty() {
            "bookkeeping-only".to_owned()
        } else {
            types.join(",")
        }
    }

    /// Bytes this page's storage actually occupies.
    ///
    /// Measured from the live buffers rather than recomputed from the layout,
    /// so it cannot drift from what was allocated. A page carries storage only
    /// when the table was built with per-layer geometry; a bookkeeping-only
    /// table reports zero here, which is the honest answer.
    pub fn allocated_bytes(&self) -> u64 {
        self.store.allocated_bytes()
    }

    pub fn reset_storage(&mut self, _config: Option<PageTensorConfig>) {
        self.filled = 0;
        self.store.reset_storage();
    }

    /// Read one scalar for `(component, head, token_offset, dim)`.
    ///
    /// `component` is the flat K/V component index `layer * 2 + kv`
    /// (`0 = key`, `1 = value`). `head_dim` is this layer's head dimension, so
    /// heterogeneous-geometry layers each address their own component slab.
    pub fn value_at_slot(
        &self,
        page_size: usize,
        head_dim: usize,
        component: usize,
        head: usize,
        token_offset: usize,
        dim: usize,
    ) -> Result<f32, KvError> {
        let view = self
            .store
            .host_view()
            .ok_or(KvError::PageNotHostAddressable(self.id))?;
        let storage = self.storage_layout[component];
        let head_len = page_size * head_dim;
        let within = head * head_len + token_offset * head_dim + dim;
        Ok(match storage.dtype {
            KvDType::F32 => view.data[storage.data_offset + within],
            KvDType::Int8 => {
                let scale =
                    view.quant_scales[storage.scale_offset + head * page_size + token_offset];
                f32::from(view.quantized_data[storage.quantized_offset + within]) * scale
            }
            KvDType::Fp8E4M3Fn | KvDType::Fp8E5M2 => {
                let scale =
                    view.quant_scales[storage.scale_offset + head * page_size + token_offset];
                decode_fp8(
                    view.fp8_data[storage.fp8_offset + within],
                    storage.dtype.fp8_format().expect("fp8 dtype"),
                ) * scale
            }
        })
    }

    /// Store one token's `head_dim` values for a single `(component, head)` slot.
    ///
    /// `component` is the flat K/V component index `layer * 2 + kv` (`0 = key`,
    /// `1 = value`). `head_dim`/`page_size` are this layer's geometry, so
    /// heterogeneous-geometry layers write into their own component slab. For
    /// quantized components this computes a per-`(head, token)` scale from
    /// *only* this token's values and writes only this token's bytes, so
    /// previously-stored tokens in the page are never dequantized or
    /// requantized. This bounds the quantization error to a single round-trip
    /// per KV write regardless of how full the page is.
    pub fn write_head_token(
        &mut self,
        page_size: usize,
        head_dim: usize,
        component: usize,
        head: usize,
        token_offset: usize,
        values: &[f32],
    ) -> Result<(), KvError> {
        debug_assert_eq!(values.len(), head_dim);
        let view = self
            .store
            .host_view_mut()
            .ok_or(KvError::PageNotHostAddressable(self.id))?;
        let storage = self.storage_layout[component];
        let head_len = page_size * head_dim;
        let within = head * head_len + token_offset * head_dim;
        match storage.dtype {
            KvDType::F32 => {
                let offset = storage.data_offset + within;
                view.data[offset..offset + head_dim].copy_from_slice(values);
            }
            KvDType::Int8 => {
                let scale = quant_scale(values, 127.0);
                view.quant_scales[storage.scale_offset + head * page_size + token_offset] = scale;
                let offset = storage.quantized_offset + within;
                for (output, value) in view.quantized_data[offset..offset + head_dim]
                    .iter_mut()
                    .zip(values)
                {
                    *output = (value / scale).round().clamp(-127.0, 127.0) as i8;
                }
            }
            KvDType::Fp8E4M3Fn | KvDType::Fp8E5M2 => {
                let format = storage.dtype.fp8_format().expect("fp8 dtype");
                let scale = quant_scale(values, format.max_finite());
                view.quant_scales[storage.scale_offset + head * page_size + token_offset] = scale;
                let offset = storage.fp8_offset + within;
                for (output, value) in view.fp8_data[offset..offset + head_dim]
                    .iter_mut()
                    .zip(values)
                {
                    *output = encode_fp8(value / scale, format);
                }
            }
        }
        Ok(())
    }

    /// Borrow one token's contiguous `head_dim` f32 row for `(component, head,
    /// token_offset)`, or `None` when this component is not stored as F32.
    ///
    /// F32 components lay each `(head, token)` row out contiguously in
    /// [`HostPageStore::data`] (see [`Page::write_head_token`]), so a
    /// runtime-managed (paged) attention reader can attend over the page **in
    /// place** — no dequantization and no copy. A quantized component has no
    /// contiguous f32 row to borrow and returns `None`, so the caller must fall
    /// back to the per-element [`Page::value_at_slot`] dequantizing path.
    pub fn head_token_f32(
        &self,
        page_size: usize,
        head_dim: usize,
        component: usize,
        head: usize,
        token_offset: usize,
    ) -> Option<&[f32]> {
        let storage = self.storage_layout[component];
        if storage.dtype != KvDType::F32 {
            return None;
        }
        let view = self.store.host_view()?;
        let head_len = page_size * head_dim;
        let within = head * head_len + token_offset * head_dim;
        let offset = storage.data_offset + within;
        Some(&view.data[offset..offset + head_dim])
    }

    pub fn has_quantized_storage(&self) -> bool {
        self.storage_layout
            .iter()
            .any(|storage| storage.dtype.is_quantized())
    }
}

/// Symmetric quantization scale for one token/head slice: the max absolute
/// finite value divided by the format's positive dynamic range. Returns `1.0`
/// for an all-zero slice so decode is a no-op.
fn quant_scale(values: &[f32], denominator: f32) -> f32 {
    let max_abs = values
        .iter()
        .filter(|value| value.is_finite())
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / denominator
    }
}

/// The page table manages the mapping from logical sequences to physical pages.
/// Cumulative KV page activity since the table was created.
///
/// Counters rather than a timeline: what a user needs to know is whether the
/// pool is under pressure. A run with evictions or, worse, allocation failures
/// is thrashing, and no per-token latency number explains that on its own.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PageStats {
    /// Pages handed out by [`PageTable::allocate`].
    pub allocations: u64,
    /// Allocations that found no page: the hot pool was exhausted.
    pub allocation_failures: u64,
    /// Pages returned to the free list (the last reference was dropped).
    pub frees: u64,
    /// Pages demoted off the hot tier to make room for an allocation.
    pub hot_evictions: u64,
    /// Pages reclaimed by evicting a cached prefix.
    pub prefix_evictions: u64,
}

/// A readable picture of what the page pool currently holds.
///
/// Cumulative counters answer "what has happened"; this answers "what is here
/// now", which is the question when a pool is filling up and you want to know
/// whether it is one runaway conversation or many small ones sharing badly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageUsage {
    pub page_size: usize,
    /// Pages the pool can hand out before it has to evict.
    pub capacity: usize,
    /// Pages currently referenced by a sequence or a cached prefix.
    pub in_use: usize,
    /// Pages on the free list of the hot tier.
    pub free: usize,
    /// Token slots actually filled across in-use pages, against the slots those
    /// pages could hold. The gap is the cost of paging: partially filled pages.
    pub filled_slots: usize,
    pub slot_capacity: usize,
    /// In-use pages with more than one reference, i.e. genuinely shared.
    pub shared: usize,
    /// How many in-use pages carry each reference count, ascending.
    pub references: Vec<(u32, usize)>,
    /// Live sequences and what they hold.
    pub sequences: Vec<SequenceUsage>,
    /// Pages per tier, for pools that spill beyond the hot tier.
    pub tiers: Vec<(Device, usize)>,
}

/// One live sequence's share of the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceUsage {
    pub sequence: SequenceId,
    pub pages: usize,
    pub tokens: usize,
    /// Of this sequence's pages, how many it shares with someone else. A
    /// conversation that attached to a cached prefix is mostly shared.
    pub shared: usize,
}

/// Cloning duplicates every page's storage but **not** the governor's grant:
/// the copy reports `leased_bytes() == None` and occupies memory nobody leased.
///
/// The clone exists for one caller — a speculative rewind validated on a copy
/// before the real pool is touched (`kv_bridge::rewind_materialized_ort_past`).
/// That is an expensive way to get transactional semantics on a real pool, and
/// removing it is tracked separately; until then the behaviour is pinned by a
/// test rather than left to chance, so nobody reads a clone's zero lease as
/// evidence that pools are ungoverned.
pub struct PageTable {
    /// Logical sequence → ordered list of page IDs.
    pub sequences: HashMap<SequenceId, Vec<PageId>>,
    /// Logical sequence → current token length.
    pub sequence_lengths: HashMap<SequenceId, usize>,
    /// Logical sequence → absolute position of the first retained *window*
    /// token. With attention sinks this is the start of the window run, which
    /// sits after the pinned sink prefix (see `sequence_sink_lens`).
    pub sequence_starts: HashMap<SequenceId, usize>,
    /// Logical sequence → number of pinned leading "attention sink" tokens.
    ///
    /// Zero for the common contiguous case. When positive, the retained buffer
    /// is the disjoint union `[0, sink_len) ∪ [window_start, len)` stored
    /// contiguously as `[sink pages | window pages]`; `sink_len` is always a
    /// multiple of `page_size`, and `window_start >= sink_len`.
    pub sequence_sink_lens: HashMap<SequenceId, usize>,
    /// All physical pages.
    pub pages: HashMap<PageId, Page>,
    /// Free pages per device.
    free_pages: HashMap<Device, Vec<PageId>>,
    /// Tokens per page.
    pub page_size: usize,
    /// Optional tensor storage layout.
    ///
    /// For heterogeneous-geometry caches this reports layer 0's geometry (so
    /// uniform callers keep working); the authoritative per-layer geometry is
    /// [`PageTable::layer_configs`].
    pub tensor_config: Option<PageTensorConfig>,
    /// Authoritative per-layer KV geometry (`num_kv_heads`/`head_dim`).
    ///
    /// Empty when no tensor storage is configured. Length equals the layer
    /// count and every layer may declare a different `head_dim`/`num_kv_heads`.
    pub layer_configs: Vec<LayerTensorConfig>,
    /// Per-layer key/value precision policy.
    pub quant_config: Option<KvQuantConfig>,
    /// Monotonic clock for LRU.
    clock: u64,
    /// Maximum number of live pages allowed in the hot tier.
    hot_capacity: usize,
    /// Next page id for cold-offload-backed growth beyond the initial hot pool.
    next_page_id: PageId,
    /// The governor's grant covering this pool's page storage.
    ///
    /// `None` for a bookkeeping-only table, which occupies nothing and so has
    /// nothing to lease. Dropping the table drops the lease, which is the only
    /// way the bytes are returned — there is deliberately no release call to
    /// forget.
    pool_lease: Option<onnx_runtime_memory_governor::MemoryLease>,
    /// Cumulative page activity, for reporting.
    stats: PageStats,
    /// Optional lock-free mirror of this pool's aggregate state.
    ///
    /// `None` on every path that does not observe the pool, so the cost when
    /// absent is one `Option` check per mutation and no stores.
    telemetry: TelemetryHandle,
    migration_factory: Arc<dyn KvPageStoreFactory>,
}

/// Holds the optional telemetry mirror, and **drops it on clone**.
///
/// A cloned page table is a different pool: it has its own pages, its own
/// reference counts, and its own statistics. Carrying the `Arc` across the
/// clone would give two independent pools one shared set of gauges, so every
/// number would be the sum of two unrelated things. Dropping it means a clone
/// simply publishes nothing until something attaches telemetry to it, which is
/// wrong in no direction.
#[derive(Debug, Default)]
struct TelemetryHandle(Option<Arc<KvTelemetry>>);

impl Clone for TelemetryHandle {
    fn clone(&self) -> Self {
        Self(None)
    }
}

impl Clone for PageTable {
    fn clone(&self) -> Self {
        Self {
            sequences: self.sequences.clone(),
            sequence_lengths: self.sequence_lengths.clone(),
            sequence_starts: self.sequence_starts.clone(),
            sequence_sink_lens: self.sequence_sink_lens.clone(),
            pages: self.pages.clone(),
            free_pages: self.free_pages.clone(),
            page_size: self.page_size,
            tensor_config: self.tensor_config,
            layer_configs: self.layer_configs.clone(),
            quant_config: self.quant_config.clone(),
            stats: self.stats,
            telemetry: self.telemetry.clone(),
            migration_factory: Arc::clone(&self.migration_factory),
            clock: self.clock,
            hot_capacity: self.hot_capacity,
            next_page_id: self.next_page_id,
            // A lease is a grant to one holder. Duplicating it would let two
            // pools claim the same bytes, so the copy carries none and is
            // therefore ungoverned -- see the type's documentation.
            pool_lease: None,
        }
    }
}

pub(crate) struct PageAllocationCheckpoint {
    device: Device,
    reused_page: Option<Page>,
    evicted_hot: Option<(PageId, Box<dyn KvPageStore>)>,
    free_pages: HashMap<Device, Vec<PageId>>,
    stats: PageStats,
    clock: u64,
    next_page_id: PageId,
    persistent_growth: Option<onnx_runtime_memory_governor::MemoryLease>,
    _transient_lease: Option<onnx_runtime_memory_governor::MemoryLease>,
}

impl PageTable {
    /// Replace the factory used for subsequent tier migrations.
    ///
    /// Existing page stores are unchanged until explicitly migrated.
    pub fn set_migration_factory(&mut self, factory: Arc<dyn KvPageStoreFactory>) {
        self.migration_factory = factory;
    }

    /// Migrate one page while preserving its logical identity and metadata.
    pub fn migrate_page(
        &mut self,
        page_id: PageId,
        target: Device,
    ) -> Result<PageMigration, KvError> {
        let factory = Arc::clone(&self.migration_factory);
        let page = self
            .pages
            .get(&page_id)
            .ok_or(KvError::PageNotFound(page_id))?;
        if page.residency() == target {
            return Ok(PageMigration {
                from: target,
                to: target,
                bytes_copied: 0,
            });
        }
        let transient_bytes = factory.allocation_bytes(target, page.store_layout);
        let _transient_lease = self.reserve_transient(transient_bytes)?;
        self.pages
            .get_mut(&page_id)
            .ok_or(KvError::PageNotFound(page_id))?
            .migrate(target, factory.as_ref())
    }

    fn reserve_transient(
        &self,
        bytes: u64,
    ) -> Result<Option<onnx_runtime_memory_governor::MemoryLease>, KvError> {
        self.pool_lease
            .as_ref()
            .map(|lease| {
                lease
                    .reserve_sibling(bytes)
                    .map_err(KvError::MigrationPressure)
            })
            .transpose()
    }

    pub(crate) fn allocation_checkpoint(
        &mut self,
        source_page: PageId,
        device: Device,
    ) -> Result<PageAllocationCheckpoint, KvError> {
        let reused_page_ref = self
            .free_pages
            .get(&device)
            .and_then(|pages| pages.last())
            .and_then(|page_id| self.pages.get(page_id));
        let evicted_page = if matches!(device, Device::Gpu(_))
            && self.free_count(device) == 0
            && self.hot_used_count() >= self.hot_capacity
        {
            self.pages
                .iter()
                .filter(|(_, page)| {
                    page.ref_count > 0 && matches!(page.residency(), Device::Gpu(_))
                })
                .min_by_key(|(_, page)| page.last_access)
                .map(|(page_id, page)| (*page_id, page))
        } else {
            None
        };
        let source_snapshot_bytes = self
            .pages
            .get(&source_page)
            .ok_or(KvError::PageNotFound(source_page))?
            .allocated_bytes();
        let reused_snapshot_bytes = reused_page_ref.map_or(0, Page::allocated_bytes);
        let victim_snapshot_bytes = evicted_page.map_or(0, |(_, page)| page.allocated_bytes());
        let replacement_bytes = if reused_page_ref.is_none() {
            source_snapshot_bytes
        } else {
            0
        };
        let persistent_growth = self.reserve_transient(replacement_bytes)?;
        let transient_bytes = source_snapshot_bytes
            .saturating_add(reused_snapshot_bytes)
            .saturating_add(victim_snapshot_bytes);
        let transient_lease = self.reserve_transient(transient_bytes)?;
        let reused_page = reused_page_ref.cloned();
        let evicted_hot =
            evicted_page.map(|(page_id, page)| (page_id, page.clone_physical_store()));
        Ok(PageAllocationCheckpoint {
            device,
            reused_page,
            evicted_hot,
            free_pages: self.free_pages.clone(),
            stats: self.stats,
            clock: self.clock,
            next_page_id: self.next_page_id,
            persistent_growth,
            _transient_lease: transient_lease,
        })
    }

    pub(crate) fn commit_allocation(
        &mut self,
        checkpoint: &mut PageAllocationCheckpoint,
    ) -> Result<(), KvError> {
        let Some(growth) = checkpoint.persistent_growth.take() else {
            return Ok(());
        };
        self.pool_lease
            .as_mut()
            .ok_or(KvError::MigrationLeaseInvariant(
                "governed growth lost its owning pool lease",
            ))?
            .absorb_sibling(growth)
            .map_err(KvError::MigrationPressure)
    }

    pub(crate) fn rollback_allocation(
        &mut self,
        checkpoint: &mut PageAllocationCheckpoint,
        allocated_page: PageId,
    ) -> Result<(), KvError> {
        if let Some(page) = checkpoint.reused_page.take() {
            self.pages.insert(allocated_page, page);
        } else {
            self.pages.remove(&allocated_page);
        }
        self.free_pages = std::mem::take(&mut checkpoint.free_pages);
        self.stats = checkpoint.stats;
        self.clock = checkpoint.clock;
        self.next_page_id = checkpoint.next_page_id;
        if let Some((page_id, store)) = checkpoint.evicted_hot.take() {
            let page = self
                .pages
                .get_mut(&page_id)
                .ok_or(KvError::PageNotFound(page_id))?;
            debug_assert_eq!(store.residency(), checkpoint.device);
            page.store = store;
        }
        self.note_ref_count_change(1, 0);
        self.publish_counters();
        Ok(())
    }

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

    /// Build a page table with **heterogeneous** per-layer KV geometry.
    ///
    /// Each entry in `layer_configs` declares that layer's `num_kv_heads`/
    /// `head_dim`; layers may differ (e.g. sliding layers use a smaller
    /// `head_dim` than full/global layers). Every layer uses `dtype`.
    pub fn new_with_layer_configs(
        page_size: usize,
        num_gpu_pages: usize,
        dtype: KvDType,
        layer_configs: Vec<LayerTensorConfig>,
    ) -> Self {
        let quant_config = KvQuantConfig::homogeneous(dtype, layer_configs.len());
        Self::new_with_layer_storage(page_size, num_gpu_pages, dtype, layer_configs, quant_config)
    }

    /// Heterogeneous per-layer geometry with an explicit KV precision policy.
    pub fn new_with_layer_quant_config(
        page_size: usize,
        num_gpu_pages: usize,
        dtype: KvDType,
        layer_configs: Vec<LayerTensorConfig>,
        quant_config: KvQuantConfig,
    ) -> Result<Self, KvError> {
        if quant_config.num_layers() != layer_configs.len() {
            return Err(KvError::InvalidQuantizationConfig(
                "quantization layer count must match per-layer tensor config".to_owned(),
            ));
        }
        Ok(Self::new_with_layer_storage(
            page_size,
            num_gpu_pages,
            dtype,
            layer_configs,
            quant_config,
        ))
    }

    fn new_with_layer_storage(
        page_size: usize,
        num_gpu_pages: usize,
        dtype: KvDType,
        layer_configs: Vec<LayerTensorConfig>,
        quant_config: KvQuantConfig,
    ) -> Self {
        assert!(!layer_configs.is_empty(), "layer_configs must be non-empty");
        assert!(
            layer_configs.iter().all(|geom| geom.validate()),
            "invalid per-layer tensor config"
        );
        // Retain a representative uniform descriptor (layer 0 geometry) so
        // uniform callers that read `tensor_config` keep working.
        // TODO(W3): remove uniform `tensor_config` accessor after engine
        // migrates to per-layer geometry.
        let representative = PageTensorConfig {
            num_layers: layer_configs.len(),
            num_kv_heads: layer_configs[0].num_kv_heads,
            head_dim: layer_configs[0].head_dim,
            page_size,
            dtype,
        };
        Self::build(
            page_size,
            num_gpu_pages,
            Some(representative),
            layer_configs,
            Some(quant_config),
        )
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
        let layer_configs = tensor_config
            .map(PageTensorConfig::to_layer_configs)
            .unwrap_or_default();
        Self::build(
            page_size,
            num_gpu_pages,
            tensor_config,
            layer_configs,
            quant_config,
        )
    }

    fn build(
        page_size: usize,
        num_gpu_pages: usize,
        tensor_config: Option<PageTensorConfig>,
        layer_configs: Vec<LayerTensorConfig>,
        quant_config: Option<KvQuantConfig>,
    ) -> Self {
        let mut pages = HashMap::new();
        let mut free_pages = vec![];
        for i in 0..num_gpu_pages {
            let id = i as PageId;
            pages.insert(
                id,
                Page::new(
                    id,
                    Device::Gpu(0),
                    page_size,
                    &layer_configs,
                    quant_config.as_ref(),
                ),
            );
            free_pages.push(id);
        }

        let mut free_map = HashMap::new();
        free_map.insert(Device::Gpu(0), free_pages);

        Self {
            sequences: HashMap::new(),
            sequence_lengths: HashMap::new(),
            sequence_starts: HashMap::new(),
            sequence_sink_lens: HashMap::new(),
            pages,
            free_pages: free_map,
            page_size,
            tensor_config,
            layer_configs,
            quant_config,
            stats: PageStats::default(),
            telemetry: TelemetryHandle::default(),
            migration_factory: Arc::new(HostPageStoreFactory),
            clock: 0,
            hot_capacity: num_gpu_pages,
            next_page_id: num_gpu_pages as PageId,
            pool_lease: None,
        }
    }

    /// Build a pool whose storage is granted by a memory governor.
    ///
    /// The size is planned first and leased before a single page is allocated.
    /// Allocating and asking afterwards is how a budget gets exceeded while
    /// every counter still reports that it was respected, so a refusal here
    /// returns an error rather than a smaller pool: silently shrinking would
    /// trade an explicit failure for mysteriously worse generation quality
    /// later, when the pool ran dry mid-sequence.
    pub fn new_leased(
        page_size: usize,
        num_gpu_pages: usize,
        dtype: KvDType,
        layer_configs: Vec<LayerTensorConfig>,
        governor: &dyn onnx_runtime_memory_governor::MemoryGovernor,
        tier: onnx_runtime_memory_governor::Tier,
        holder: onnx_runtime_memory_governor::HolderId,
    ) -> Result<Self, KvError> {
        let quant_config = KvQuantConfig::homogeneous(dtype, layer_configs.len());
        let planned = Self::planned_pool_bytes(
            page_size,
            num_gpu_pages,
            &layer_configs,
            Some(&quant_config),
        );
        let lease = governor.reserve(
            tier,
            planned,
            onnx_runtime_memory_governor::MemoryRole::KvCache,
            holder,
        )?;
        let mut table = Self::new_with_layer_storage(
            page_size,
            num_gpu_pages,
            dtype,
            layer_configs,
            quant_config,
        );
        // Checked in release too, not just debug. An under-plan means the pool
        // occupies more than it leased while the ledger reports success, which
        // is the one failure this contract exists to prevent -- so it must not
        // be a class of bug that only shows up in a debug build.
        let actual = table.pool_bytes();
        if actual != planned {
            return Err(KvError::PoolSizeMismatch { planned, actual });
        }
        table.pool_lease = Some(lease);
        Ok(table)
    }

    /// Bytes this pool currently holds a grant for, or `None` if ungoverned.
    pub fn leased_bytes(&self) -> Option<u64> {
        self.pool_lease
            .as_ref()
            .map(onnx_runtime_memory_governor::MemoryLease::bytes)
    }
    /// Bytes occupied by pages that are currently referenced by something.
    ///
    /// Each page counts once however many sequences share it, so this is what a
    /// memory lease can be compared against. Free pages in a pre-allocated pool
    /// still occupy memory, so this is a lower bound on the pool, not a
    /// replacement for [`Self::pool_bytes`].
    pub fn resident_bytes(&self) -> u64 {
        self.pages
            .values()
            .filter(|page| page.ref_count > 0)
            .map(Page::allocated_bytes)
            .fold(0u64, u64::saturating_add)
    }

    /// Bytes one page of this pool occupies, or zero for a bookkeeping-only pool.
    pub fn bytes_per_page(&self) -> u64 {
        self.pages.values().next().map_or(0, Page::allocated_bytes)
    }

    /// Bytes the whole page pool actually occupies.
    ///
    /// This is what a memory lease has to cover. It is summed from the live
    /// pages rather than recomputed, so it reports what was allocated even if
    /// the planning arithmetic below ever drifts from the allocator.
    pub fn pool_bytes(&self) -> u64 {
        self.pages
            .values()
            .map(Page::allocated_bytes)
            .fold(0u64, u64::saturating_add)
    }

    /// Bytes a pool of `num_pages` *would* occupy, before allocating one.
    ///
    /// Needed because a lease has to be granted before the memory is taken:
    /// allocating first and asking afterwards is how a budget gets exceeded
    /// while reporting that it was respected. Mirrors [`Page::new`]'s layout,
    /// and a test pins the two together so the mirror cannot rot.
    pub fn planned_pool_bytes(
        page_size: usize,
        num_pages: usize,
        layer_configs: &[LayerTensorConfig],
        quant_config: Option<&KvQuantConfig>,
    ) -> u64 {
        let Some(quant_config) = quant_config else {
            return 0;
        };
        if layer_configs.is_empty() {
            return 0;
        }
        let f32_bytes = size_of::<f32>() as u64;
        let mut per_page = 0u64;
        for (layer, geom) in layer_configs.iter().enumerate() {
            let component_len = (geom.num_kv_heads * page_size * geom.head_dim) as u64;
            let scale_slots = (geom.num_kv_heads * page_size) as u64;
            for kind in [KvKind::Key, KvKind::Value] {
                per_page = per_page.saturating_add(match quant_config.dtype(layer, kind) {
                    KvDType::F32 => component_len.saturating_mul(f32_bytes),
                    KvDType::Int8 => component_len.saturating_add(scale_slots * f32_bytes),
                    KvDType::Fp8E4M3Fn | KvDType::Fp8E5M2 => {
                        component_len.saturating_add(scale_slots * f32_bytes)
                    }
                });
            }
        }
        per_page.saturating_mul(num_pages as u64)
    }

    /// Allocate a new page on the specified device.
    pub fn allocate(&mut self, device: Device) -> Option<PageId> {
        let allocated = self.allocate_page(device);
        match allocated {
            Some(_) => self.stats.allocations += 1,
            None => self.stats.allocation_failures += 1,
        }
        self.publish_counters();
        allocated
    }

    /// Cumulative page activity since this table was created.
    /// Attach a lock-free telemetry mirror to this pool.
    ///
    /// Seeds the mirror from the pool's current state rather than leaving it at
    /// zero, because attaching to an already-warm pool would otherwise publish
    /// a zero that was never true and then drift correct only as pages moved.
    pub fn attach_telemetry(&mut self, telemetry: Arc<KvTelemetry>) {
        telemetry.set_geometry(self.hot_capacity, self.page_size);
        telemetry.publish_counters(&self.stats);
        let (in_use, shared) = self.live_page_counts();
        telemetry.set_live_gauges(in_use, shared);
        self.telemetry = TelemetryHandle(Some(telemetry));
    }

    /// Count live and shared pages by walking the pool.
    ///
    /// `O(pages)`, so this is for seeding and for tests that assert the
    /// incrementally-maintained gauges have not drifted from the truth. The
    /// decode path uses the edge-triggered updates instead.
    pub fn live_page_counts(&self) -> (usize, usize) {
        let mut in_use = 0;
        let mut shared = 0;
        for page in self.pages.values() {
            if page.ref_count > 0 {
                in_use += 1;
            }
            if page.ref_count > 1 {
                shared += 1;
            }
        }
        (in_use, shared)
    }

    /// Mirror the cumulative counters, if telemetry is attached.
    ///
    /// Republishes the whole `PageStats` rather than incrementing individually,
    /// so a counter added later cannot be silently omitted from the published
    /// view. Costs a handful of relaxed stores.
    fn publish_counters(&self) {
        if let Some(telemetry) = &self.telemetry.0 {
            telemetry.publish_counters(&self.stats);
        }
    }

    /// Report a page's reference-count transition to the telemetry gauges.
    fn note_ref_count_change(&self, old: u32, new: u32) {
        if let Some(telemetry) = &self.telemetry.0 {
            telemetry.note_ref_count_change(old, new);
        }
    }

    /// Summarize what the pool is holding right now.
    pub fn usage(&self) -> PageUsage {
        let mut references: BTreeMap<u32, usize> = BTreeMap::new();
        let mut tiers: Vec<(Device, usize)> = Vec::new();
        let mut in_use = 0;
        let mut filled_slots = 0;
        for page in self.pages.values() {
            if page.ref_count == 0 {
                continue;
            }
            in_use += 1;
            filled_slots += page.filled;
            *references.entry(page.ref_count).or_default() += 1;
            let residency = page.residency();
            match tiers.iter_mut().find(|(device, _)| *device == residency) {
                Some((_, count)) => *count += 1,
                None => tiers.push((residency, 1)),
            }
        }
        let mut sequences = self
            .sequences
            .iter()
            .map(|(sequence, pages)| SequenceUsage {
                sequence: *sequence,
                pages: pages.len(),
                tokens: self.sequence_lengths.get(sequence).copied().unwrap_or(0),
                shared: pages
                    .iter()
                    .filter(|page| self.pages.get(page).is_some_and(|page| page.ref_count > 1))
                    .count(),
            })
            .collect::<Vec<_>>();
        sequences.sort_by_key(|usage| (std::cmp::Reverse(usage.pages), usage.sequence));
        PageUsage {
            page_size: self.page_size,
            capacity: self.hot_capacity,
            in_use,
            free: self.free_count(Device::Gpu(0)),
            filled_slots,
            slot_capacity: in_use * self.page_size,
            shared: references
                .iter()
                .filter(|(count, _)| **count > 1)
                .map(|(_, pages)| *pages)
                .sum(),
            references: references.into_iter().collect(),
            sequences,
            tiers,
        }
    }

    pub fn stats(&self) -> PageStats {
        self.stats
    }

    /// Record pages reclaimed by evicting a cached prefix.
    ///
    /// The prefix cache frees those pages through [`free`](Self::free), which
    /// cannot tell a reclaim from an ordinary release, so the owner reports it.
    pub fn note_prefix_eviction(&mut self, pages: u64) {
        self.stats.prefix_evictions += pages;
        self.publish_counters();
    }

    fn allocate_page(&mut self, device: Device) -> Option<PageId> {
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
                page.reset_storage(self.tensor_config);
                self.clock += 1;
                page.last_access = self.clock;
            }
            // A page off the free list had a zero count by construction.
            self.note_ref_count_change(0, 1);
            return Some(page_id);
        }
        if matches!(device, Device::Gpu(_)) && self.hot_used_count() < self.hot_capacity {
            let page_id = self.next_page_id;
            self.next_page_id = self.next_page_id.saturating_add(1);
            let mut page = Page::new(
                page_id,
                device,
                self.page_size,
                &self.layer_configs,
                self.quant_config.as_ref(),
            );
            page.ref_count = 1;
            self.clock += 1;
            page.last_access = self.clock;
            self.pages.insert(page_id, page);
            self.note_ref_count_change(0, 1);
            return Some(page_id);
        }
        None
    }

    /// Free a page (decrement ref_count; actually free when it hits 0).
    pub fn free(&mut self, page_id: PageId) {
        let mut transition = None;
        if let Some(page) = self.pages.get_mut(&page_id) {
            let previous = page.ref_count;
            page.ref_count = page.ref_count.saturating_sub(1);
            transition = Some((previous, page.ref_count));
            if page.ref_count == 0 {
                page.reset_storage(self.tensor_config);
                let device = page.residency();
                self.free_pages.entry(device).or_default().push(page_id);
                self.stats.frees += 1;
            }
        }
        if let Some((previous, current)) = transition {
            self.note_ref_count_change(previous, current);
            self.publish_counters();
        }
    }

    /// Increment a page reference for CoW/prefix sharing.
    pub fn retain(&mut self, page_id: PageId) -> bool {
        let mut transition = None;
        if let Some(page) = self.pages.get_mut(&page_id) {
            let previous = page.ref_count;
            page.ref_count = page.ref_count.saturating_add(1);
            self.clock += 1;
            page.last_access = self.clock;
            transition = Some((previous, page.ref_count));
        }
        match transition {
            Some((previous, current)) => {
                self.note_ref_count_change(previous, current);
                true
            }
            None => false,
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

    /// Number of pinned leading attention-sink tokens for `seq` (0 if none).
    pub fn sequence_sink_len(&self, seq: SequenceId) -> Option<usize> {
        self.sequence_sink_lens.get(&seq).copied()
    }

    pub fn set_sequence_sink_len(&mut self, seq: SequenceId, sink_len: usize) {
        if let Some(slot) = self.sequence_sink_lens.get_mut(&seq) {
            *slot = sink_len;
        }
    }

    /// Create a new sequence (empty).
    pub fn create_sequence(&mut self, seq: SequenceId) {
        self.sequences.insert(seq, Vec::new());
        self.sequence_lengths.insert(seq, 0);
        self.sequence_starts.insert(seq, 0);
        self.sequence_sink_lens.insert(seq, 0);
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
        if matches!(page.residency(), Device::Gpu(_)) {
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
        self.migrate_page(page_id, Device::Gpu(0))?;
        let page = self
            .pages
            .get_mut(&page_id)
            .ok_or(KvError::PageNotFound(page_id))?;
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
                Some(**id) != exclude
                    && page.ref_count > 0
                    && matches!(page.residency(), Device::Gpu(_))
            })
            .min_by_key(|(_, page)| page.last_access)
        else {
            return Err(KvError::OutOfMemory {
                needed: 1,
                available: 0,
            });
        };
        self.migrate_page(victim_id, Device::Cpu)?;
        self.stats.hot_evictions += 1;
        self.publish_counters();
        Ok(victim_id)
    }

    /// Demote every hot page owned exclusively by `seq` to the cold CPU tier.
    ///
    /// Unlike [`evict_lru_hot`](Self::evict_lru_hot), which frees the single
    /// globally-least-recently-used page, this is *sequence-scoped*: it targets
    /// exactly one sequence's hot residency, which is what a scheduler
    /// preemption asks for ("evict sequence S"). Pages shared with another live
    /// sequence or a retained prefix (`ref_count > 1`) are left resident so a
    /// preemption never steals KV still needed by a running peer.
    ///
    /// Each page is copied transactionally into a cold-tier store. Returns the
    /// number of pages demoted.
    pub fn evict_sequence_to_cold(&mut self, seq: SequenceId) -> Result<usize, KvError> {
        let Some(page_ids) = self.sequences.get(&seq).cloned() else {
            return Ok(0);
        };
        let mut demoted = 0;
        for page_id in page_ids {
            let should_demote = self.pages.get(&page_id).is_some_and(|page| {
                page.ref_count <= 1 && matches!(page.residency(), Device::Gpu(_))
            });
            if should_demote {
                self.migrate_page(page_id, Device::Cpu)?;
                demoted += 1;
            }
        }
        self.stats.hot_evictions += demoted as u64;
        self.publish_counters();
        Ok(demoted)
    }

    /// Number of pages backing `seq` currently resident on the hot tier.
    pub fn sequence_hot_pages(&self, seq: SequenceId) -> usize {
        self.sequences.get(&seq).map_or(0, |page_ids| {
            page_ids
                .iter()
                .filter(|page_id| {
                    self.pages
                        .get(page_id)
                        .is_some_and(|page| matches!(page.residency(), Device::Gpu(_)))
                })
                .count()
        })
    }

    /// Per-layer KV geometry for `layer`, or `None` when out of range or when
    /// no tensor storage is configured.
    pub fn layer_config(&self, layer: usize) -> Option<LayerTensorConfig> {
        self.layer_configs.get(layer).copied()
    }

    /// Remove a sequence and return its pages.
    pub fn remove_sequence(&mut self, seq: SequenceId) -> Vec<PageId> {
        self.sequence_lengths.remove(&seq);
        self.sequence_starts.remove(&seq);
        self.sequence_sink_lens.remove(&seq);
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
            .filter(|page| page.ref_count > 0 && matches!(page.residency(), Device::Gpu(_)))
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

#[cfg(test)]
mod migration_tests {
    use super::*;

    #[derive(Debug)]
    struct AllocationFailureFactory;

    impl KvPageStoreFactory for AllocationFailureFactory {
        fn allocation_bytes(&self, _residency: Device, layout: PageStoreLayout) -> u64 {
            layout.host_allocated_bytes()
        }

        fn create(
            &self,
            _residency: Device,
            _layout: PageStoreLayout,
        ) -> Result<Box<dyn KvPageStore>, KvError> {
            Err(KvError::PageStoreAllocationFailed("injected".into()))
        }
    }

    #[derive(Debug)]
    struct CopyFailureFactory;

    impl KvPageStoreFactory for CopyFailureFactory {
        fn allocation_bytes(&self, _residency: Device, layout: PageStoreLayout) -> u64 {
            layout.host_allocated_bytes()
        }

        fn create(
            &self,
            residency: Device,
            layout: PageStoreLayout,
        ) -> Result<Box<dyn KvPageStore>, KvError> {
            Ok(Box::new(CopyFailureStore {
                inner: HostPageStore::new(residency, layout),
            }))
        }
    }

    #[derive(Debug, Clone)]
    struct CopyFailureStore {
        inner: HostPageStore,
    }

    impl KvPageStore for CopyFailureStore {
        fn residency(&self) -> Device {
            self.inner.residency()
        }

        fn allocated_bytes(&self) -> u64 {
            self.inner.allocated_bytes()
        }

        fn reset_storage(&mut self) {
            self.inner.reset_storage();
        }

        fn host_view(&self) -> Option<HostPageStoreView<'_>> {
            self.inner.host_view()
        }

        fn host_view_mut(&mut self) -> Option<HostPageStoreViewMut<'_>> {
            self.inner.host_view_mut()
        }

        fn device_span(&self) -> Option<DevicePageSpan> {
            None
        }

        fn copy_to(&self, target: &mut dyn KvPageStore) -> Result<u64, KvError> {
            self.inner.copy_to(target)
        }

        fn copy_from_host(&mut self, _source: HostPageStoreView<'_>) -> Result<(), KvError> {
            Err(KvError::PageStoreCopyUnsupported {
                from: Device::Cpu,
                to: self.residency(),
            })
        }

        fn clone_store(&self) -> Box<dyn KvPageStore> {
            Box::new(self.clone())
        }
    }

    #[derive(Debug)]
    struct DeviceOnlyFactory;

    impl KvPageStoreFactory for DeviceOnlyFactory {
        fn allocation_bytes(&self, _residency: Device, layout: PageStoreLayout) -> u64 {
            layout.host_allocated_bytes()
        }

        fn create(
            &self,
            residency: Device,
            layout: PageStoreLayout,
        ) -> Result<Box<dyn KvPageStore>, KvError> {
            Ok(Box::new(DeviceOnlyStore {
                residency,
                bytes: vec![
                    0;
                    layout.f32_len * size_of::<f32>()
                        + layout.int8_len
                        + layout.fp8_len
                        + layout.scale_len * size_of::<f32>()
                ],
            }))
        }
    }

    #[derive(Debug, Clone)]
    struct DeviceOnlyStore {
        residency: Device,
        bytes: Vec<u8>,
    }

    impl KvPageStore for DeviceOnlyStore {
        fn residency(&self) -> Device {
            self.residency
        }

        fn allocated_bytes(&self) -> u64 {
            self.bytes.len() as u64
        }

        fn reset_storage(&mut self) {
            self.bytes.fill(0);
        }

        fn host_view(&self) -> Option<HostPageStoreView<'_>> {
            None
        }

        fn host_view_mut(&mut self) -> Option<HostPageStoreViewMut<'_>> {
            None
        }

        fn device_span(&self) -> Option<DevicePageSpan> {
            None
        }

        fn copy_to(&self, _target: &mut dyn KvPageStore) -> Result<u64, KvError> {
            Err(KvError::PageStoreCopyUnsupported {
                from: self.residency,
                to: Device::Cpu,
            })
        }

        fn copy_from_host(&mut self, source: HostPageStoreView<'_>) -> Result<(), KvError> {
            let mut offset = 0;
            for value in source.data.iter().chain(source.quant_scales) {
                let bytes = value.to_ne_bytes();
                self.bytes[offset..offset + bytes.len()].copy_from_slice(&bytes);
                offset += bytes.len();
            }
            for value in source.quantized_data {
                self.bytes[offset] = value.to_ne_bytes()[0];
                offset += 1;
            }
            for value in source.fp8_data {
                self.bytes[offset] = *value;
                offset += 1;
            }
            Ok(())
        }

        fn clone_store(&self) -> Box<dyn KvPageStore> {
            Box::new(self.clone())
        }
    }

    fn mixed_table() -> (PageTable, PageId) {
        let quant = KvQuantConfig {
            layers: vec![
                LayerKvDType {
                    key: KvDType::F32,
                    value: KvDType::Int8,
                },
                LayerKvDType {
                    key: KvDType::Fp8E4M3Fn,
                    value: KvDType::Fp8E5M2,
                },
            ],
        };
        let mut table = PageTable::new_with_layer_quant_config(
            2,
            1,
            KvDType::F32,
            vec![
                LayerTensorConfig {
                    num_kv_heads: 1,
                    head_dim: 2,
                },
                LayerTensorConfig {
                    num_kv_heads: 1,
                    head_dim: 2,
                },
            ],
            quant,
        )
        .unwrap();
        let page_id = table.allocate(Device::Gpu(0)).unwrap();
        let page = table.pages.get_mut(&page_id).unwrap();
        page.write_head_token(2, 2, 0, 0, 0, &[1.25, -2.5]).unwrap();
        page.write_head_token(2, 2, 1, 0, 0, &[3.0, -6.0]).unwrap();
        page.write_head_token(2, 2, 2, 0, 0, &[0.75, -1.5]).unwrap();
        page.write_head_token(2, 2, 3, 0, 0, &[9.5, -4.25]).unwrap();
        page.filled = 1;
        page.ref_count = 3;
        page.last_access = 77;
        (table, page_id)
    }

    fn snapshot(page: &Page) -> (Vec<f32>, Vec<i8>, Vec<u8>, Vec<f32>) {
        let view = page.host_view().unwrap();
        (
            view.data.to_vec(),
            view.quantized_data.to_vec(),
            view.fp8_data.to_vec(),
            view.quant_scales.to_vec(),
        )
    }

    #[test]
    fn migration_copies_every_component_and_preserves_logical_metadata() {
        let (mut table, page_id) = mixed_table();
        let before = snapshot(&table.pages[&page_id]);
        assert!(before.0.iter().any(|value| *value != 0.0));
        assert!(before.1.iter().any(|value| *value != 0));
        assert!(before.2.iter().any(|value| *value != 0));
        assert!(before.3.iter().any(|value| *value != 1.0));

        let demoted = table.migrate_page(page_id, Device::Cpu).unwrap();
        assert_eq!(
            demoted.bytes_copied,
            table.pages[&page_id].allocated_bytes()
        );
        assert_eq!(snapshot(&table.pages[&page_id]), before);
        let page = &table.pages[&page_id];
        assert_eq!(page.id, page_id);
        assert_eq!(page.ref_count, 3);
        assert_eq!(page.filled, 1);
        assert_eq!(page.last_access, 77);

        table.migrate_page(page_id, Device::Gpu(0)).unwrap();
        assert_eq!(snapshot(&table.pages[&page_id]), before);
    }

    #[test]
    fn migration_failures_leave_source_unchanged_and_retry_succeeds() {
        let (mut table, page_id) = mixed_table();
        let before = snapshot(&table.pages[&page_id]);

        table.set_migration_factory(Arc::new(AllocationFailureFactory));
        assert!(table.migrate_page(page_id, Device::Cpu).is_err());
        assert_eq!(table.pages[&page_id].residency(), Device::Gpu(0));
        assert_eq!(snapshot(&table.pages[&page_id]), before);

        table.set_migration_factory(Arc::new(CopyFailureFactory));
        assert!(table.migrate_page(page_id, Device::Cpu).is_err());
        assert_eq!(table.pages[&page_id].residency(), Device::Gpu(0));
        assert_eq!(snapshot(&table.pages[&page_id]), before);

        table.set_migration_factory(Arc::new(HostPageStoreFactory));
        table.migrate_page(page_id, Device::Cpu).unwrap();
        assert_eq!(snapshot(&table.pages[&page_id]), before);
    }

    #[test]
    fn unsupported_migration_names_state_types_backend_and_keeps_source() {
        let (mut table, page_id) = mixed_table();
        let before = snapshot(&table.pages[&page_id]);

        let error = table.migrate_page(page_id, Device::Disk).unwrap_err();
        let message = error.to_string();
        assert!(message.contains(&format!("page {page_id}")));
        assert!(message.contains("f32,int8,fp8-e4m3fn,fp8-e5m2"));
        assert!(message.contains("host-page-store"));
        assert!(message.contains("Disk"));
        assert_eq!(table.pages[&page_id].residency(), Device::Gpu(0));
        assert_eq!(snapshot(&table.pages[&page_id]), before);
        assert_eq!(table.pages[&page_id].filled, 1);
    }

    #[test]
    fn device_only_store_has_no_implicit_host_view_or_materialization() {
        let (mut table, page_id) = mixed_table();
        table.set_migration_factory(Arc::new(DeviceOnlyFactory));
        table.migrate_page(page_id, Device::Gpu(1)).unwrap();
        let page = &table.pages[&page_id];
        assert_eq!(page.residency(), Device::Gpu(1));
        assert!(page.host_view().is_none());
        assert!(matches!(
            page.value_at_slot(2, 2, 0, 0, 0, 0),
            Err(KvError::PageNotHostAddressable(id)) if id == page_id
        ));
        assert!(table.migrate_page(page_id, Device::Cpu).is_err());
    }
}

#[cfg(test)]
mod page_stats_tests {
    use super::*;

    #[test]
    fn allocations_and_frees_are_counted() {
        let mut table = PageTable::new(16, 4);

        let first = table
            .allocate(Device::Gpu(0))
            .expect("a free pool allocates");
        let second = table
            .allocate(Device::Gpu(0))
            .expect("a free pool allocates");
        table.free(first);

        let stats = table.stats();
        assert_eq!(stats.allocations, 2);
        assert_eq!(stats.frees, 1);
        assert_eq!(stats.allocation_failures, 0);
        assert_eq!(stats.hot_evictions, 0);
        table.free(second);
        assert_eq!(table.stats().frees, 2);
    }

    #[test]
    fn a_full_hot_pool_evicts_rather_than_failing() {
        let mut table = PageTable::new(16, 2);

        // Fill the hot pool, then ask for one more: the table demotes the
        // least-recently-used page instead of returning nothing.
        let _first = table.allocate(Device::Gpu(0)).expect("first page");
        let _second = table.allocate(Device::Gpu(0)).expect("second page");
        let third = table.allocate(Device::Gpu(0));

        let stats = table.stats();
        assert!(
            stats.hot_evictions > 0,
            "pressure must be recorded, not silently absorbed: {stats:?}"
        );
        assert!(third.is_some(), "the eviction should have made room");
        assert_eq!(stats.allocations, 3);
    }

    #[test]
    fn an_exhausted_pool_records_the_failure() {
        let mut table = PageTable::new(16, 1);
        let only = table.allocate(Device::Gpu(0)).expect("the single page");

        // Every page is pinned by the one live reference and cannot be demoted
        // twice, so eventually an allocation genuinely fails.
        let mut failures = 0;
        for _ in 0..8 {
            if table.allocate(Device::Gpu(0)).is_none() {
                failures += 1;
            }
        }

        if failures > 0 {
            assert_eq!(
                table.stats().allocation_failures,
                failures,
                "each exhausted allocation must be counted"
            );
        }
        table.free(only);
    }
}

#[cfg(test)]
mod pool_accounting_tests {
    use super::*;

    fn geometry(num_layers: usize, kv_heads: usize, head_dim: usize) -> Vec<LayerTensorConfig> {
        (0..num_layers)
            .map(|_| LayerTensorConfig {
                num_kv_heads: kv_heads,
                head_dim,
            })
            .collect()
    }

    /// The planner and the allocator must agree, for every geometry.
    ///
    /// A lease has to be granted before memory is taken, so the size is
    /// predicted from the layout rather than measured. If that prediction ever
    /// drifts below what `Page::new` really allocates, the pool would occupy
    /// more than it leased while every counter reported that the budget was
    /// respected -- the exact failure this whole contract exists to prevent.
    #[test]
    fn the_planned_pool_size_equals_what_the_pool_actually_allocates() {
        for (label, page_size, pages, layers, kv_heads, head_dim, dtype) in [
            ("f32 uniform", 16, 8, 4, 2, 64, KvDType::F32),
            ("f32 wide", 32, 3, 2, 8, 128, KvDType::F32),
            ("int8", 16, 5, 3, 4, 64, KvDType::Int8),
            ("fp8 e4m3", 16, 5, 3, 4, 64, KvDType::Fp8E4M3Fn),
            ("fp8 e5m2", 8, 7, 6, 2, 32, KvDType::Fp8E5M2),
            ("single page", 4, 1, 1, 1, 8, KvDType::F32),
        ] {
            let configs = geometry(layers, kv_heads, head_dim);
            let quant = KvQuantConfig::homogeneous(dtype, layers);
            let planned = PageTable::planned_pool_bytes(page_size, pages, &configs, Some(&quant));
            let table = PageTable::new_with_layer_configs(page_size, pages, dtype, configs.clone());
            assert_eq!(
                planned,
                table.pool_bytes(),
                "{label}: planned {planned} bytes but allocated {}",
                table.pool_bytes()
            );
            assert!(planned > 0, "{label}: a configured pool must occupy memory");
        }
    }

    /// Heterogeneous layers must be summed per layer, not multiplied by a mean.
    #[test]
    fn planning_handles_layers_with_different_geometry() {
        let configs = vec![
            LayerTensorConfig {
                num_kv_heads: 2,
                head_dim: 64,
            },
            LayerTensorConfig {
                num_kv_heads: 8,
                head_dim: 128,
            },
        ];
        let quant = KvQuantConfig::homogeneous(KvDType::F32, configs.len());
        let planned = PageTable::planned_pool_bytes(16, 4, &configs, Some(&quant));
        let table = PageTable::new_with_layer_configs(16, 4, KvDType::F32, configs);
        assert_eq!(planned, table.pool_bytes());
    }

    /// A table with no per-layer geometry occupies nothing.
    ///
    /// Worth pinning because it is easy to assume the paged cache always owns
    /// the KV. It does not: without tensor geometry it is pure bookkeeping, and
    /// the engine's default construction takes exactly this path. Leasing has
    /// to reflect that rather than reserving for memory nobody took.
    #[test]
    fn a_bookkeeping_only_pool_occupies_no_memory() {
        let table = PageTable::new(16, 64);
        assert_eq!(
            table.pool_bytes(),
            0,
            "a table without per-layer geometry allocated page storage"
        );
        assert_eq!(PageTable::planned_pool_bytes(16, 64, &[], None), 0);
    }

    /// Pool size scales with page count, so the lease does too.
    #[test]
    fn pool_size_scales_linearly_with_page_count() {
        let configs = geometry(2, 4, 32);
        let quant = KvQuantConfig::homogeneous(KvDType::F32, 2);
        let one = PageTable::planned_pool_bytes(16, 1, &configs, Some(&quant));
        let ten = PageTable::planned_pool_bytes(16, 10, &configs, Some(&quant));
        assert_eq!(ten, one * 10);
    }
}

#[cfg(test)]
mod pool_lease_tests {
    use super::*;
    use onnx_runtime_memory_governor::{
        HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, Tier,
    };
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    const HOLDER: HolderId = HolderId::new(7);

    fn configs(layers: usize) -> Vec<LayerTensorConfig> {
        (0..layers)
            .map(|_| LayerTensorConfig {
                num_kv_heads: 2,
                head_dim: 32,
            })
            .collect()
    }

    fn planned(pages: usize, layers: usize) -> u64 {
        let quant = KvQuantConfig::homogeneous(KvDType::F32, layers);
        PageTable::planned_pool_bytes(16, pages, &configs(layers), Some(&quant))
    }

    #[derive(Debug)]
    struct ObservingFactory {
        ledger: Arc<LeaseLedger>,
        observed_used: Arc<AtomicU64>,
        fail: Arc<AtomicBool>,
    }

    impl KvPageStoreFactory for ObservingFactory {
        fn allocation_bytes(&self, _residency: Device, layout: PageStoreLayout) -> u64 {
            layout.host_allocated_bytes()
        }

        fn create(
            &self,
            residency: Device,
            layout: PageStoreLayout,
        ) -> Result<Box<dyn KvPageStore>, KvError> {
            self.observed_used
                .store(self.ledger.used(Tier::Host), Ordering::Relaxed);
            if self.fail.load(Ordering::Relaxed) {
                return Err(KvError::PageStoreAllocationFailed("injected".into()));
            }
            HostPageStoreFactory.create(residency, layout)
        }
    }

    #[test]
    fn full_budget_refuses_migration_before_target_allocation() {
        let want = planned(1, 2);
        let ledger = LeaseLedger::new(0, want, 0);
        let governor = LedgerGovernor::new(Arc::clone(&ledger));
        let mut table = PageTable::new_leased(
            16,
            1,
            KvDType::F32,
            configs(2),
            &governor,
            Tier::Host,
            HOLDER,
        )
        .unwrap();
        let page_id = table.allocate(Device::Gpu(0)).unwrap();
        let before = table.pages[&page_id].host_view().unwrap().data.to_vec();
        let free_before = table.free_pages.clone();
        let stats_before = table.stats();
        let observed = Arc::new(AtomicU64::new(0));
        table.set_migration_factory(Arc::new(ObservingFactory {
            ledger: Arc::clone(&ledger),
            observed_used: Arc::clone(&observed),
            fail: Arc::new(AtomicBool::new(false)),
        }));

        assert!(matches!(
            table.migrate_page(page_id, Device::Cpu),
            Err(KvError::MigrationPressure(_))
        ));
        assert_eq!(observed.load(Ordering::Relaxed), 0);
        assert_eq!(ledger.used(Tier::Host), want);
        assert_eq!(table.pages[&page_id].residency(), Device::Gpu(0));
        assert_eq!(table.pages[&page_id].host_view().unwrap().data, before);
        assert_eq!(table.free_pages, free_before);
        assert_eq!(table.stats(), stats_before);
    }

    #[test]
    fn transient_migration_lease_covers_success_and_failure_then_releases() {
        let page_bytes = planned(1, 2);
        let ledger = LeaseLedger::new(0, page_bytes * 2, 0);
        let governor = LedgerGovernor::new(Arc::clone(&ledger));
        let mut table = PageTable::new_leased(
            16,
            1,
            KvDType::F32,
            configs(2),
            &governor,
            Tier::Host,
            HOLDER,
        )
        .unwrap();
        let page_id = table.allocate(Device::Gpu(0)).unwrap();
        let observed = Arc::new(AtomicU64::new(0));
        let fail = Arc::new(AtomicBool::new(false));
        table.set_migration_factory(Arc::new(ObservingFactory {
            ledger: Arc::clone(&ledger),
            observed_used: Arc::clone(&observed),
            fail: Arc::clone(&fail),
        }));

        table.migrate_page(page_id, Device::Cpu).unwrap();
        assert_eq!(observed.load(Ordering::Relaxed), page_bytes * 2);
        assert_eq!(ledger.used(Tier::Host), page_bytes);

        fail.store(true, Ordering::Relaxed);
        assert!(table.migrate_page(page_id, Device::Gpu(0)).is_err());
        assert_eq!(observed.load(Ordering::Relaxed), page_bytes * 2);
        assert_eq!(ledger.used(Tier::Host), page_bytes);
        assert_eq!(table.pages[&page_id].residency(), Device::Cpu);
    }

    /// A governed pool leases exactly what it occupies.
    #[test]
    fn a_governed_pool_leases_exactly_what_it_allocates() {
        let want = planned(4, 2);
        let governor = LedgerGovernor::new(LeaseLedger::new(want, 0, 0));
        let table = PageTable::new_leased(
            16,
            4,
            KvDType::F32,
            configs(2),
            &governor,
            Tier::Device,
            HOLDER,
        )
        .expect("the tier holds exactly this pool");

        assert_eq!(table.leased_bytes(), Some(want));
        assert_eq!(table.pool_bytes(), want, "leased and occupied must agree");
        assert_eq!(governor.available(Tier::Device), 0);
    }

    /// Too small a budget refuses the pool rather than quietly shrinking it.
    ///
    /// A pool that silently came back smaller would trade an explicit startup
    /// failure for a mid-generation one, when the pool ran dry and mirroring
    /// stopped early -- wrong output with nothing in the logs.
    #[test]
    fn an_insufficient_budget_refuses_the_pool_instead_of_shrinking_it() {
        let want = planned(4, 2);
        let governor = LedgerGovernor::new(LeaseLedger::new(want - 1, 0, 0));
        let error = PageTable::new_leased(
            16,
            4,
            KvDType::F32,
            configs(2),
            &governor,
            Tier::Device,
            HOLDER,
        )
        .map(|_| ())
        .expect_err("one byte short must not be granted");
        assert!(
            matches!(error, KvError::PoolNotLeased(_)),
            "expected a lease refusal, got {error}"
        );
        assert_eq!(
            governor.available(Tier::Device),
            want - 1,
            "a refused pool must not have consumed budget"
        );
    }

    /// Dropping the pool returns its bytes to the tier.
    #[test]
    fn dropping_a_governed_pool_returns_its_bytes() {
        let want = planned(4, 2);
        let governor = LedgerGovernor::new(LeaseLedger::new(want, 0, 0));
        {
            let _table = PageTable::new_leased(
                16,
                4,
                KvDType::F32,
                configs(2),
                &governor,
                Tier::Device,
                HOLDER,
            )
            .expect("the tier holds this pool");
            assert_eq!(governor.available(Tier::Device), 0);
        }
        assert_eq!(
            governor.available(Tier::Device),
            want,
            "dropping the pool did not return its lease"
        );

        // And the tier can grant the whole pool again, which a double release
        // would have turned into spare capacity that does not exist.
        let _second = PageTable::new_leased(
            16,
            4,
            KvDType::F32,
            configs(2),
            &governor,
            Tier::Device,
            HOLDER,
        )
        .expect("the tier is free again");
        assert_eq!(governor.available(Tier::Device), 0);
    }

    /// Two pools cannot both be granted a budget that only fits one.
    #[test]
    fn a_tier_that_fits_one_pool_refuses_the_second() {
        let want = planned(4, 2);
        let governor = LedgerGovernor::new(LeaseLedger::new(want, 0, 0));
        let _first = PageTable::new_leased(
            16,
            4,
            KvDType::F32,
            configs(2),
            &governor,
            Tier::Device,
            HOLDER,
        )
        .expect("the first pool fits");
        PageTable::new_leased(
            16,
            4,
            KvDType::F32,
            configs(2),
            &governor,
            Tier::Device,
            HOLDER,
        )
        .map(|_| ())
        .expect_err("the tier cannot hold a second pool of the same size");
    }
    /// A cloned pool carries no grant, and that is pinned rather than incidental.
    ///
    /// Duplicating a lease would let two pools claim the same bytes. The copy
    /// therefore holds none -- which also means it occupies memory nobody
    /// leased, so this test exists as much to document the hazard as to check
    /// the field.
    #[test]
    fn a_cloned_pool_reports_no_lease() {
        let want = planned(4, 2);
        let governor = LedgerGovernor::new(LeaseLedger::new(want, 0, 0));
        let table = PageTable::new_leased(
            16,
            4,
            KvDType::F32,
            configs(2),
            &governor,
            Tier::Device,
            HOLDER,
        )
        .expect("the tier holds this pool");

        let copy = table.clone();
        assert_eq!(copy.leased_bytes(), None, "a clone duplicated the grant");
        assert_eq!(
            copy.pool_bytes(),
            want,
            "the clone still duplicated the page storage"
        );
        assert_eq!(
            governor.available(Tier::Device),
            0,
            "cloning must not consume further budget"
        );
    }
    /// A page is host memory, so the pool must be charged to the host tier.
    ///
    /// `HostPageStore` owns host vectors for both emulated locations. Charging
    /// them to the device tier -- which the `num_gpu_pages` lineage invites --
    /// would let the pool exhaust host RAM
    /// while the device ledger still reported headroom, which is the governor
    /// failing at the one thing it exists to do. Caught in review before it
    /// shipped, so this test is what keeps it caught.
    #[test]
    fn a_host_backed_pool_is_charged_to_the_host_tier() {
        let want = planned(4, 2);
        // Device has room, host does not. A pool that charged the device tier
        // would be granted here; one that charges host must be refused.
        let governor = LedgerGovernor::new(LeaseLedger::new(want * 8, want - 1, 0));
        PageTable::new_leased(
            16,
            4,
            KvDType::F32,
            configs(2),
            &governor,
            Tier::Host,
            HOLDER,
        )
        .map(|_| ())
        .expect_err("a host tier without room must refuse a host-backed pool");
        assert_eq!(
            governor.available(Tier::Device),
            want * 8,
            "the refusal consumed device budget for a host allocation"
        );
    }
}
