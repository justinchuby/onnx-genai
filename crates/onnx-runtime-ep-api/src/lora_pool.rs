//! Paged adapter weight pool for the Phase-2 multi-adapter LoRA subsystem
//! (design `docs/NATIVE_LORA_DESIGN.md` §J.2).
//!
//! This is the **data plane** the `GroupedLoraDelta` kernel reads from: a
//! host-side arena holding many adapters' already-transposed `A_t`/`B_t` factor
//! pages, stored **adapter-major** and **64-byte aligned** for SIMD GEMM tiles.
//! Pages are admitted at adapter-load time and never copied on the decode hot
//! path — the kernel only takes read views (`&[u8]`) at page offsets, so the
//! per-token path performs **no heap allocation for adapter weights**.
//!
//! ## What lives here vs. what lives above
//!
//! Byte accounting and LRU eviction are self-contained in this type. The design
//! §J.2 sketches the pool as owning a `scheduler::ByteBudget`, but that type
//! lives in `onnx-genai-scheduler`, a *higher* layer than this foundational EP
//! contract crate. Importing it here would invert the dependency graph
//! (`ep-api` is a leaf that `ep-cpu`/`session`/`engine` all depend on). So the
//! pool keeps an equivalent internal saturating byte budget with the same
//! semantics, and the **real** `ByteBudget` governs adapter admission one layer
//! up in the engine (where the scheduler is in scope and the KV budget already
//! lives). See `crates/onnx-genai-engine/src/lora/pool.rs`.
//!
//! ## Alignment
//!
//! Each factor page is a contiguous, 64-byte-aligned buffer. The kernel reads
//! it as a dense row-major `[rows, cols]` tensor with no gather/stride, matching
//! the MLAS/AVX GEMM tile requirement (design §J.2 "contiguous and 64-byte
//! aligned").

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use onnx_runtime_ir::DataType;

/// SIMD GEMM tile alignment for every factor page (design §J.2).
pub const LORA_PAGE_ALIGNMENT: usize = 64;

/// A globally unique adapter identity. The scheduler maps a per-request
/// `adapter_id` to one of these; the reserved [`AdapterId::NULL`] routes a batch
/// row to the base-only (empty) delta.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AdapterId(pub u64);

impl AdapterId {
    /// The reserved "no adapter" identity. A batch row whose segment resolves to
    /// `NULL` receives a zero delta (base-only), so mixed base+adapter batches
    /// cost nothing extra (design §J.4 null page).
    pub const NULL: AdapterId = AdapterId(u64::MAX);

    /// Whether this is the reserved base-only identity.
    pub fn is_null(self) -> bool {
        self == Self::NULL
    }
}

/// A target-module index within an adapter (the position of `q_proj`, `k_proj`,
/// … in the adapter's module list). Ranks differ per module (`rank_pattern`), so
/// the page unit is one `(adapter, module)` factor pair (design §J.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LoraModuleId(pub u32);

/// Which factor of a LoRA pair a page holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoraFactorKind {
    /// `A_t`, shape `[K, rank]`.
    A,
    /// `B_t`, shape `[rank, N]`.
    B,
}

/// Errors raised admitting or looking up pages. Every variant is actionable and
/// the pool never panics on a capacity shortfall or a bad shape (RULES: typed
/// errors, no panic on bad input).
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum LoraPoolError {
    #[error(
        "LoRA pool capacity exceeded: adapter {adapter} module {module} needs {requested} B but \
         the pool ceiling is only {capacity} B; raise the pool budget or reduce resident adapters"
    )]
    CapacityExceeded {
        adapter: u64,
        module: u32,
        requested: u64,
        capacity: u64,
    },
    #[error(
        "LoRA factor for adapter {adapter} module {module} has {actual} bytes but its \
         {rows}x{cols} {dtype:?} shape needs {expected} bytes"
    )]
    ByteShapeMismatch {
        adapter: u64,
        module: u32,
        rows: usize,
        cols: usize,
        dtype: DataType,
        expected: usize,
        actual: usize,
    },
    #[error(
        "LoRA factor pair for adapter {adapter} module {module} disagrees on rank: A_t is \
         [{a_k}, {a_r}] but B_t is [{b_r}, {b_n}] ({a_r} != {b_r})"
    )]
    RankMismatch {
        adapter: u64,
        module: u32,
        a_k: usize,
        a_r: usize,
        b_r: usize,
        b_n: usize,
    },
    #[error("no LoRA page resident for adapter {adapter} module {module}")]
    MissingPage { adapter: u64, module: u32 },
}

/// A 64-byte-aligned owned byte buffer backing one factor page.
///
/// Backing store is a `Vec` of 64-byte-aligned chunks, so the allocation start
/// is 64-byte aligned (the allocator honours the element type's alignment). The
/// logical length is tracked separately because the last chunk may be partly
/// unused.
struct AlignedPageBuffer {
    chunks: Vec<Align64>,
    len: usize,
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Align64([u8; 64]);

impl AlignedPageBuffer {
    fn from_bytes(bytes: &[u8]) -> Self {
        let chunk_count = bytes.len().div_ceil(LORA_PAGE_ALIGNMENT).max(1);
        let mut chunks = vec![Align64([0u8; 64]); chunk_count];
        // SAFETY: `chunks` owns `chunk_count * 64 >= bytes.len()` contiguous
        // bytes starting at a 64-byte-aligned address (Align64's alignment), so
        // this destination slice is in-bounds and correctly aligned.
        let dst = unsafe {
            std::slice::from_raw_parts_mut(chunks.as_mut_ptr() as *mut u8, chunk_count * 64)
        };
        dst[..bytes.len()].copy_from_slice(bytes);
        Self {
            chunks,
            len: bytes.len(),
        }
    }

    fn as_bytes(&self) -> &[u8] {
        // SAFETY: `self.len <= chunks.len() * 64` and the chunks are one
        // contiguous 64-byte-aligned allocation, so this view is in-bounds.
        unsafe { std::slice::from_raw_parts(self.chunks.as_ptr() as *const u8, self.len) }
    }

    fn byte_capacity(&self) -> usize {
        self.chunks.len() * LORA_PAGE_ALIGNMENT
    }
}

/// One resident factor page: aligned bytes plus its logical shape/dtype.
struct FactorPage {
    buffer: AlignedPageBuffer,
    rows: usize,
    cols: usize,
    dtype: DataType,
}

impl FactorPage {
    fn view(&self) -> LoraFactorView<'_> {
        LoraFactorView {
            dtype: self.dtype,
            rows: self.rows,
            cols: self.cols,
            bytes: self.buffer.as_bytes(),
        }
    }
}

/// A resident `(adapter, module)` factor pair (design §J.2 "page = one
/// (adapter, target_module) factor pair").
struct AdapterModulePages {
    a: FactorPage,
    b: FactorPage,
    /// Per-module LoRA scale (`alpha / rank`), applied in fp32 by the kernel.
    scale: f32,
    /// Total resident bytes accounted against the budget (aligned capacity of
    /// both buffers), so a release exactly reverses its admission.
    resident_bytes: u64,
    /// Monotonic recency stamp for LRU eviction (higher = more recently touched).
    recency: u64,
}

/// A read-only view of one resident factor, handed to the kernel. Borrows the
/// pool's aligned arena; no copy is made on the decode hot path.
#[derive(Clone, Copy, Debug)]
pub struct LoraFactorView<'a> {
    pub dtype: DataType,
    pub rows: usize,
    pub cols: usize,
    pub bytes: &'a [u8],
}

/// A read-only view of one resident `(adapter, module)` pair.
#[derive(Clone, Copy, Debug)]
pub struct LoraPagePair<'a> {
    pub a: LoraFactorView<'a>,
    pub b: LoraFactorView<'a>,
    pub scale: f32,
}

/// The host-side paged adapter weight pool (design §J.2).
///
/// Admission (`admit`) and eviction happen at adapter-load time under `&mut
/// self`; the kernel holds an `Arc<LoraWeightPool>` and only ever calls the
/// `&self` read accessors, so the decode hot path is lock-free and
/// allocation-free with respect to adapter weights.
pub struct LoraWeightPool {
    capacity_bytes: u64,
    used_bytes: u64,
    next_recency: u64,
    pages: HashMap<(AdapterId, LoraModuleId), AdapterModulePages>,
}

impl LoraWeightPool {
    /// Create an empty pool with an absolute byte ceiling.
    pub fn with_capacity_bytes(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            next_recency: 0,
            pages: HashMap::new(),
        }
    }

    /// The active byte ceiling.
    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    /// Bytes currently resident (aligned page capacities).
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// Number of resident `(adapter, module)` pairs.
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Whether the pool holds no pages.
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Whether a `(adapter, module)` pair is resident.
    pub fn contains(&self, adapter: AdapterId, module: LoraModuleId) -> bool {
        self.pages.contains_key(&(adapter, module))
    }

    /// Admit one `(adapter, module)` factor pair, copying the already-transposed
    /// `A_t = [K, rank]` and `B_t = [rank, N]` bytes into fresh aligned pages.
    ///
    /// Evicts least-recently-used pages (never the pair being admitted) until the
    /// pair fits, then errors with [`LoraPoolError::CapacityExceeded`] only if the
    /// single pair is larger than the whole pool ceiling. Fails loud on any
    /// shape/byte or rank disagreement rather than storing a corrupt page.
    #[allow(clippy::too_many_arguments)]
    pub fn admit(
        &mut self,
        adapter: AdapterId,
        module: LoraModuleId,
        a_t: LoraFactorInput<'_>,
        b_t: LoraFactorInput<'_>,
        scale: f32,
    ) -> Result<(), LoraPoolError> {
        validate_factor(adapter, module, &a_t)?;
        validate_factor(adapter, module, &b_t)?;
        // Rank agreement: A_t is [K, rank], B_t is [rank, N].
        if a_t.cols != b_t.rows {
            return Err(LoraPoolError::RankMismatch {
                adapter: adapter.0,
                module: module.0,
                a_k: a_t.rows,
                a_r: a_t.cols,
                b_r: b_t.rows,
                b_n: b_t.cols,
            });
        }

        let a_page = FactorPage {
            buffer: AlignedPageBuffer::from_bytes(a_t.bytes),
            rows: a_t.rows,
            cols: a_t.cols,
            dtype: a_t.dtype,
        };
        let b_page = FactorPage {
            buffer: AlignedPageBuffer::from_bytes(b_t.bytes),
            rows: b_t.rows,
            cols: b_t.cols,
            dtype: b_t.dtype,
        };
        let resident_bytes =
            (a_page.buffer.byte_capacity() + b_page.buffer.byte_capacity()) as u64;

        if resident_bytes > self.capacity_bytes {
            return Err(LoraPoolError::CapacityExceeded {
                adapter: adapter.0,
                module: module.0,
                requested: resident_bytes,
                capacity: self.capacity_bytes,
            });
        }

        // If this key is already resident, release its bytes first so the
        // re-admission accounts correctly (a hot-swap of the same slot).
        if let Some(existing) = self.pages.remove(&(adapter, module)) {
            self.used_bytes = self.used_bytes.saturating_sub(existing.resident_bytes);
        }

        self.evict_until_fits(resident_bytes);

        let recency = self.next_recency;
        self.next_recency += 1;
        self.used_bytes += resident_bytes;
        self.pages.insert(
            (adapter, module),
            AdapterModulePages {
                a: a_page,
                b: b_page,
                scale,
                resident_bytes,
                recency,
            },
        );
        Ok(())
    }

    /// Evict a specific pair, releasing its bytes. Returns whether it was present.
    pub fn evict(&mut self, adapter: AdapterId, module: LoraModuleId) -> bool {
        if let Some(page) = self.pages.remove(&(adapter, module)) {
            self.used_bytes = self.used_bytes.saturating_sub(page.resident_bytes);
            true
        } else {
            false
        }
    }

    /// Look up a resident factor pair (read-only, hot-path safe: no LRU mutation,
    /// no allocation). Returns `None` for the reserved null adapter or a missing
    /// page; the kernel treats both as a zero delta / fail-loud as appropriate.
    pub fn pair(&self, adapter: AdapterId, module: LoraModuleId) -> Option<LoraPagePair<'_>> {
        if adapter.is_null() {
            return None;
        }
        let pages = self.pages.get(&(adapter, module))?;
        Some(LoraPagePair {
            a: pages.a.view(),
            b: pages.b.view(),
            scale: pages.scale,
        })
    }

    /// Look up one factor view (read-only). Convenience for tests / device
    /// binders that page a single factor at a time.
    pub fn factor(
        &self,
        adapter: AdapterId,
        module: LoraModuleId,
        kind: LoraFactorKind,
    ) -> Option<LoraFactorView<'_>> {
        let pages = self.pages.get(&(adapter, module))?;
        Some(match kind {
            LoraFactorKind::A => pages.a.view(),
            LoraFactorKind::B => pages.b.view(),
        })
    }

    /// Mark a pair most-recently-used (a load-time / admission-time control-plane
    /// operation — never called on the read hot path).
    pub fn touch(&mut self, adapter: AdapterId, module: LoraModuleId) {
        let recency = self.next_recency;
        if let Some(page) = self.pages.get_mut(&(adapter, module)) {
            page.recency = recency;
            self.next_recency += 1;
        }
    }

    fn evict_until_fits(&mut self, incoming_bytes: u64) {
        while self.used_bytes + incoming_bytes > self.capacity_bytes {
            let Some(victim) = self
                .pages
                .iter()
                .min_by_key(|(_, page)| page.recency)
                .map(|(key, _)| *key)
            else {
                break;
            };
            if let Some(page) = self.pages.remove(&victim) {
                self.used_bytes = self.used_bytes.saturating_sub(page.resident_bytes);
            }
        }
    }
}

/// A borrowed, already-transposed factor to admit: `A_t` is `[K, rank]`, `B_t`
/// is `[rank, N]`, both contiguous row-major in `bytes`.
#[derive(Clone, Copy, Debug)]
pub struct LoraFactorInput<'a> {
    pub dtype: DataType,
    pub rows: usize,
    pub cols: usize,
    pub bytes: &'a [u8],
}

fn validate_factor(
    adapter: AdapterId,
    module: LoraModuleId,
    factor: &LoraFactorInput<'_>,
) -> Result<(), LoraPoolError> {
    let expected = factor
        .dtype
        .checked_storage_bytes(factor.rows.saturating_mul(factor.cols))
        .unwrap_or(usize::MAX);
    if factor.bytes.len() != expected {
        return Err(LoraPoolError::ByteShapeMismatch {
            adapter: adapter.0,
            module: module.0,
            rows: factor.rows,
            cols: factor.cols,
            dtype: factor.dtype,
            expected,
            actual: factor.bytes.len(),
        });
    }
    Ok(())
}

// ===========================================================================
// Process-wide pool registry — the CPU host-pool binding seam.
// ===========================================================================
//
// On the CPU EP the `GroupedLoraDelta` kernel resolves its pool through this
// registry by a `pool_id` op attribute, rather than through the lazy-weight
// seam (`LazyWeightBoundary`). The lazy-weight seam is a *device/paging*
// mechanism: the stock CPU EP does not advertise `nxrt` weight paging, so a lazy
// handle would be materialized resident and never reach the kernel as a handle.
// The pool is host-resident memory that needs no negotiation or materialization,
// so a stable id → `Arc<LoraWeightPool>` handle is the correct, allocation-free
// CPU binding. `LazyWeightBoundary::GroupedLora` still exists (generalized in
// `weight.rs`) as the seam a *paging* EP (CUDA, P2g) will use to bind the pool on
// device.

/// A registration token; drop it (or call [`LoraPoolRegistry::unregister`]) to
/// release the pool from the process registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LoraPoolId(pub u64);

/// Process-wide registry mapping a `pool_id` to a shared pool. Cloneable handle;
/// all clones observe the same registry.
#[derive(Clone, Default)]
pub struct LoraPoolRegistry {
    inner: Arc<RwLock<LoraPoolRegistryState>>,
}

#[derive(Default)]
struct LoraPoolRegistryState {
    next_id: u64,
    pools: HashMap<u64, Arc<LoraWeightPool>>,
}

impl LoraPoolRegistry {
    /// The single process-global registry the CPU kernel factory consults.
    pub fn global() -> &'static LoraPoolRegistry {
        static GLOBAL: std::sync::OnceLock<LoraPoolRegistry> = std::sync::OnceLock::new();
        GLOBAL.get_or_init(LoraPoolRegistry::default)
    }

    /// Register a pool and return its fresh id (bake this into the op attribute).
    pub fn register(&self, pool: Arc<LoraWeightPool>) -> LoraPoolId {
        let mut state = self.write();
        let id = state.next_id;
        state.next_id += 1;
        state.pools.insert(id, pool);
        LoraPoolId(id)
    }

    /// Look up a pool by id.
    pub fn get(&self, id: LoraPoolId) -> Option<Arc<LoraWeightPool>> {
        self.read().pools.get(&id.0).cloned()
    }

    /// Remove a pool from the registry.
    pub fn unregister(&self, id: LoraPoolId) -> Option<Arc<LoraWeightPool>> {
        self.write().pools.remove(&id.0)
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, LoraPoolRegistryState> {
        self.inner
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, LoraPoolRegistryState> {
        self.inner
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn factor<'a>(rows: usize, cols: usize, bytes: &'a [u8]) -> LoraFactorInput<'a> {
        LoraFactorInput {
            dtype: DataType::Float32,
            rows,
            cols,
            bytes,
        }
    }

    #[test]
    fn admit_then_lookup_returns_the_pair_and_pages_are_64b_aligned() {
        let mut pool = LoraWeightPool::with_capacity_bytes(1 << 20);
        let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // [3,2]
        let b = f32_bytes(&[7.0, 8.0, 9.0, 10.0]); // [2,2]
        pool.admit(AdapterId(0), LoraModuleId(0), factor(3, 2, &a), factor(2, 2, &b), 0.5)
            .unwrap();

        let pair = pool.pair(AdapterId(0), LoraModuleId(0)).expect("resident");
        assert_eq!(pair.scale, 0.5);
        assert_eq!((pair.a.rows, pair.a.cols), (3, 2));
        assert_eq!((pair.b.rows, pair.b.cols), (2, 2));
        assert_eq!(pair.a.bytes, a.as_slice());
        assert_eq!(pair.b.bytes, b.as_slice());
        assert_eq!(pair.a.bytes.as_ptr() as usize % LORA_PAGE_ALIGNMENT, 0);
        assert_eq!(pair.b.bytes.as_ptr() as usize % LORA_PAGE_ALIGNMENT, 0);
    }

    #[test]
    fn byte_accounting_tracks_aligned_capacity_and_releases_on_evict() {
        let mut pool = LoraWeightPool::with_capacity_bytes(1 << 20);
        assert_eq!(pool.used_bytes(), 0);
        let a = f32_bytes(&[0.0; 6]);
        let b = f32_bytes(&[0.0; 4]);
        pool.admit(AdapterId(1), LoraModuleId(2), factor(3, 2, &a), factor(2, 2, &b), 1.0)
            .unwrap();
        // Each small buffer rounds up to one 64-byte chunk.
        assert_eq!(pool.used_bytes(), 128);
        assert_eq!(pool.len(), 1);
        assert!(pool.evict(AdapterId(1), LoraModuleId(2)));
        assert_eq!(pool.used_bytes(), 0);
        assert_eq!(pool.len(), 0);
        assert!(!pool.evict(AdapterId(1), LoraModuleId(2)));
    }

    #[test]
    fn lru_eviction_drops_least_recently_admitted_under_pressure() {
        // Capacity for exactly two pairs (128 B each).
        let mut pool = LoraWeightPool::with_capacity_bytes(256);
        let a = f32_bytes(&[1.0; 6]);
        let b = f32_bytes(&[1.0; 4]);
        pool.admit(AdapterId(0), LoraModuleId(0), factor(3, 2, &a), factor(2, 2, &b), 1.0)
            .unwrap();
        pool.admit(AdapterId(1), LoraModuleId(0), factor(3, 2, &a), factor(2, 2, &b), 1.0)
            .unwrap();
        assert_eq!(pool.len(), 2);
        // Touch adapter 0 so adapter 1 becomes the LRU victim.
        pool.touch(AdapterId(0), LoraModuleId(0));
        pool.admit(AdapterId(2), LoraModuleId(0), factor(3, 2, &a), factor(2, 2, &b), 1.0)
            .unwrap();
        assert_eq!(pool.len(), 2);
        assert!(pool.contains(AdapterId(0), LoraModuleId(0)));
        assert!(!pool.contains(AdapterId(1), LoraModuleId(0)), "LRU victim evicted");
        assert!(pool.contains(AdapterId(2), LoraModuleId(0)));
        assert_eq!(pool.used_bytes(), 256);
    }

    #[test]
    fn oversized_pair_reports_typed_capacity_error_without_panicking() {
        let mut pool = LoraWeightPool::with_capacity_bytes(64);
        let a = f32_bytes(&[1.0; 6]);
        let b = f32_bytes(&[1.0; 4]);
        let err = pool
            .admit(AdapterId(0), LoraModuleId(0), factor(3, 2, &a), factor(2, 2, &b), 1.0)
            .unwrap_err();
        assert!(matches!(err, LoraPoolError::CapacityExceeded { .. }));
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn byte_shape_mismatch_is_rejected() {
        let mut pool = LoraWeightPool::with_capacity_bytes(1 << 20);
        let a = f32_bytes(&[1.0; 5]); // wrong: [3,2] needs 6 elements
        let b = f32_bytes(&[1.0; 4]);
        let err = pool
            .admit(AdapterId(0), LoraModuleId(0), factor(3, 2, &a), factor(2, 2, &b), 1.0)
            .unwrap_err();
        assert!(matches!(err, LoraPoolError::ByteShapeMismatch { .. }));
    }

    #[test]
    fn rank_mismatch_between_factors_is_rejected() {
        let mut pool = LoraWeightPool::with_capacity_bytes(1 << 20);
        let a = f32_bytes(&[1.0; 6]); // [3,2] -> rank 2
        let b = f32_bytes(&[1.0; 9]); // [3,3] -> rank 3
        let err = pool
            .admit(AdapterId(0), LoraModuleId(0), factor(3, 2, &a), factor(3, 3, &b), 1.0)
            .unwrap_err();
        assert!(matches!(err, LoraPoolError::RankMismatch { .. }));
    }

    #[test]
    fn null_adapter_never_resolves_a_pair() {
        let pool = LoraWeightPool::with_capacity_bytes(1 << 20);
        assert!(pool.pair(AdapterId::NULL, LoraModuleId(0)).is_none());
    }

    #[test]
    fn registry_register_get_unregister_roundtrip() {
        let registry = LoraPoolRegistry::default();
        let pool = Arc::new(LoraWeightPool::with_capacity_bytes(1 << 20));
        let id = registry.register(Arc::clone(&pool));
        assert!(registry.get(id).is_some());
        assert!(Arc::ptr_eq(&registry.get(id).unwrap(), &pool));
        assert!(registry.unregister(id).is_some());
        assert!(registry.get(id).is_none());
    }
}
