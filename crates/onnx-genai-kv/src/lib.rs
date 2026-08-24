//! Paged KV cache manager.
//!
//! Implements PagedAttention-style memory management with:
//! - Fixed-size page allocation (eliminates fragmentation)
//! - Copy-on-Write forking (cheap session branching)
//! - Tiered storage (GPU → CPU → Disk)
//! - Prefix sharing via radix trie
//! - Rewind/checkpoint operations for speculative decoding
//!
//! ## Sliding-window attention (SWA) & attention sinks (DESIGN §40)
//!
//! Window-bounded KV retention is supported on the paged cache via
//! [`paged_cache::PagedKvCache::apply_sliding_window`] (contiguous window) and
//! [`paged_cache::PagedKvCache::apply_sliding_window_with_sinks`] (StreamingLLM:
//! pinned leading "sink" tokens + trailing window). Sink pinning on the paged
//! cache is **page-granular** (the sink prefix is rounded up to a page
//! boundary); the engine's runtime KV buffer applies the same window/sink
//! **token-exactly**. Both keep O(1)/token cost.
//!
//! Not handled here (deferred to Mobius/ORT, see `.squad/decisions`): hybrid
//! per-layer attention patterns (§40.3) needing per-layer KV buffers, and
//! feeding discontinuous `position_ids` into a contiguous ORT graph (§40.8).

pub mod backing_store;
pub mod connector;
pub mod fp8;
pub mod local_tiered;
pub mod page_table;
pub mod paged_cache;
pub mod paged_index;
pub mod prefix_cache;
pub mod telemetry;
pub mod tiered;

pub use backing_store::{DiskKvBackingStore, InMemoryKvBackingStore, KvBackingStore};
pub use connector::{
    CachePriority, CompressionFormat, ConnectorCapabilities, ConnectorError, ConnectorHealth,
    ConnectorResult, DEFAULT_CHUNK_SIZE, FetchedKv, KvCacheConnector, KvCacheKey, KvCacheLocation,
    KvLayerPayload, KvPayload, KvPayloadDtype, KvStoreEntry, NullConnector, TokenChunk,
    chunk_tokens, hash_tokens,
};
pub use fp8::{Fp8Format, decode_f32 as decode_fp8, encode_f32 as encode_fp8};
pub use local_tiered::{DiskTierConfig, LocalTieredConfig, LocalTieredConnector};
pub use page_table::{
    DevicePageSpan, HostPageStore, HostPageStoreFactory, HostPageStoreView, HostPageStoreViewMut,
    KvComponentPolicy, KvDType, KvKind, KvPageStore, KvPageStoreFactory, KvQuantAxis,
    KvQuantConfig, KvQuantPolicy, LayerKvDType, LayerPrecisionRule, LayerTensorConfig, Page,
    PageId, PageMigration, PageStats, PageStoreLayout, PageTable, PageTensorConfig, PageUsage,
    SequenceUsage,
};
pub use paged_cache::{LayerKv, MaterializedKv, MaterializedLayerKv, PagedKvCache};
pub use paged_index::{
    LatentCacheGeometry, MIN_PAGED_BLOCK_SIZE, PAGED_BLOCK_TABLE_PAD, PAGED_SLOT_EMPTY,
    PagedIndexPlan, PagedKvLayout, PagedRequest, is_valid_paged_block_size, latent_element_offset,
    token_major_element_offset,
};
pub use prefix_cache::PrefixCache;
pub use telemetry::{Applicability, KvNotApplicable, KvTelemetry, KvTelemetrySnapshot};

/// Sequence identifier.
pub type SequenceId = u64;

/// Token identifier.
pub type TokenId = u32;

/// Round a required sequence length up to the shared KV capacity bucket.
///
/// Buckets are powers of two, at least the minimum bucket floor, and clamped to
/// the caller's hard maximum. `ONNX_GENAI_KV_MIN_BUCKET` can raise or lower the
/// floor for deployments that want fewer early reallocations or smaller initial
/// buffers. This is the single capacity policy shared by ORT shared-buffer KV
/// and native CUDA KV; scheduler/resource planning can consume the same source
/// instead of reimplementing it.
pub fn kv_capacity_bucket(len: usize, hard_max: usize) -> usize {
    const MIN_BUCKET_DEFAULT: usize = 256;
    if hard_max == 0 {
        return 0;
    }
    let min_bucket = std::env::var("ONNX_GENAI_KV_MIN_BUCKET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(MIN_BUCKET_DEFAULT);
    len.next_power_of_two().max(min_bucket).min(hard_max)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCapacityGrowth {
    Unchanged,
    Grew {
        old_capacity: usize,
        new_capacity: usize,
        valid_len: usize,
    },
}

pub trait KvCapacityGrowthBackend {
    type Error;
    type GrownBuffers;
    type GrownMask;

    fn current_capacity(&self) -> usize;
    fn hard_max_capacity(&self) -> usize;
    fn valid_len(&self) -> usize;
    fn capacity_exceeded(&self, required: usize) -> Self::Error;
    fn build_grown_buffers(
        &mut self,
        new_capacity: usize,
        valid_len: usize,
    ) -> Result<Self::GrownBuffers, Self::Error>;
    fn build_grown_mask(
        &mut self,
        new_capacity: usize,
        valid_len: usize,
    ) -> Result<Option<Self::GrownMask>, Self::Error>;
    fn invalidate_capture(&mut self) -> Result<(), Self::Error>;
    fn commit_grown_capacity(
        &mut self,
        new_capacity: usize,
        grown_buffers: Self::GrownBuffers,
        grown_mask: Option<Self::GrownMask>,
    ) -> Result<(), Self::Error>;
}

/// Shared grow driver for fixed-capacity KV buckets.
///
/// Backends provide allocation/copy/capture primitives; this function owns the
/// policy: reject above the hard maximum, choose the next shared bucket, build
/// all fallible replacement state before mutating live state, invalidate capture,
/// then commit atomically.
pub fn ensure_kv_capacity<B>(backend: &mut B, required: usize) -> Result<KvCapacityGrowth, B::Error>
where
    B: KvCapacityGrowthBackend,
{
    let hard_max = backend.hard_max_capacity();
    if required > hard_max {
        return Err(backend.capacity_exceeded(required));
    }
    let old_capacity = backend.current_capacity();
    if required <= old_capacity {
        return Ok(KvCapacityGrowth::Unchanged);
    }
    let new_capacity = kv_capacity_bucket(required, hard_max);
    if new_capacity <= old_capacity {
        return Ok(KvCapacityGrowth::Unchanged);
    }
    let valid_len = backend.valid_len();
    let grown_buffers = backend.build_grown_buffers(new_capacity, valid_len)?;
    let grown_mask = backend.build_grown_mask(new_capacity, valid_len)?;
    backend.invalidate_capture()?;
    backend.commit_grown_capacity(new_capacity, grown_buffers, grown_mask)?;
    Ok(KvCapacityGrowth::Grew {
        old_capacity,
        new_capacity,
        valid_len,
    })
}

/// Declared page-store residency.
///
/// `Gpu` is currently a host-backed emulation location in `onnx-genai-kv`;
/// addressability is reported separately by `KvPageStore::host_view`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Gpu(usize), // GPU index
    Cpu,
    Disk,
}

/// Eviction policy for freeing pages under memory pressure.
#[derive(Debug, Clone, Copy)]
pub enum EvictionPolicy {
    /// Least recently used page gets evicted.
    Lru,
    /// Lower-priority sequences evict first.
    Priority,
    /// Metadata-specified sensitive layers stay on GPU.
    LayerAware,
}

/// How a backend can be handed this store's KV without a copy.
///
/// Capability is a type rather than a name-keyed lookup, so a mismatch is a
/// compile-or-construction-time refusal instead of a silent fall back to a
/// slower path whose only symptom is that generation got slower.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvViewKind {
    /// Scattered pages. A backend must read through the store's accessors.
    Paged,
    /// One flat address range whose pages are physically scattered. Satisfies a
    /// backend that requires contiguity, with no copy.
    VirtuallyContiguous,
    /// One real allocation.
    PhysicallyContiguous,
}

/// KV cache operations trait (from spec §4c).
pub trait KvCacheOps {
    /// Truncate cache to position. O(pages_removed).
    fn rewind_to(&mut self, seq: SequenceId, position: usize) -> Result<(), KvError>;

    /// Fork a sequence with CoW semantics.
    fn fork(&mut self, source: SequenceId, position: usize) -> Result<SequenceId, KvError>;

    /// Save cache state for later restore.
    fn checkpoint(&self, seq: SequenceId) -> Result<CacheCheckpoint, KvError>;

    /// Restore from a checkpoint.
    fn restore(&mut self, seq: SequenceId, checkpoint: CacheCheckpoint) -> Result<(), KvError>;

    /// Append new KV entries after a forward pass.
    fn append(&mut self, seq: SequenceId, num_tokens: usize) -> Result<(), KvError>;

    /// Get the current length (in tokens) for a sequence.
    fn len(&self, seq: SequenceId) -> Result<usize, KvError>;

    /// Remove a sequence entirely, freeing all its pages.
    fn remove(&mut self, seq: SequenceId) -> Result<(), KvError>;

    /// Bytes of KV storage this sequence references.
    ///
    /// **Attributed, not exclusive.** A page shared with another sequence — the
    /// normal result of prefix reuse or a fork — is counted in full for every
    /// sequence that references it, because that is what the sequence would need
    /// if it were alone. Summing this across sequences therefore over-counts and
    /// is *not* the store's footprint; use [`Self::resident_bytes`] for that.
    ///
    /// Getting this backwards is how a memory budget starts describing memory
    /// that was never allocated.
    fn sequence_bytes(&self, seq: SequenceId) -> Result<u64, KvError>;

    /// Bytes of KV storage actually occupied by pages currently referenced.
    ///
    /// Counts each page once regardless of how many sequences share it, so this
    /// is the number that can be compared against a memory lease.
    fn resident_bytes(&self) -> u64;

    /// Bytes this store holds a memory grant for, or `None` when ungoverned.
    fn leased_bytes(&self) -> Option<u64>;

    /// What a backend can be handed without a copy.
    fn view(&self) -> KvViewKind;
}

/// A saved cache state for checkpoint/restore.
#[derive(Debug, Clone)]
pub struct CacheCheckpoint {
    pub seq: SequenceId,
    pub position: usize,
    pub page_ids: Vec<PageId>,
}

#[derive(Debug, thiserror::Error)]
pub enum KvError {
    /// The memory governor refused to lease the page pool.
    ///
    /// Reported instead of allocating anyway, so a pool can never occupy more
    /// than it was granted: a budget that is exceeded while reporting success
    /// is worse than no budget.
    #[error("cannot lease the KV page pool: {0}")]
    PoolNotLeased(#[from] onnx_runtime_memory_governor::MemoryError),
    /// The pool allocated a different amount than it leased.
    ///
    /// Only reachable if the size planner and the page allocator disagree,
    /// which would mean the pool silently occupies memory outside its grant.
    /// Refused rather than corrected, because the two must be kept in step
    /// rather than reconciled after the fact.
    #[error(
        "the KV page pool leased {planned} bytes but allocated {actual}; the pool size planner \
         and the page allocator have diverged, so the pool would occupy memory outside its \
         grant. Fix by updating PageTable::planned_pool_bytes to match Page::new's layout"
    )]
    PoolSizeMismatch {
        /// What was leased.
        planned: u64,
        /// What was allocated.
        actual: u64,
    },
    #[error("cannot reserve transient KV migration memory: {0}")]
    MigrationPressure(onnx_runtime_memory_governor::MemoryError),
    #[error("KV migration lease invariant failed: {0}")]
    MigrationLeaseInvariant(&'static str),
    #[error("Sequence {0} not found")]
    SequenceNotFound(SequenceId),
    #[error("Out of memory: need {needed} pages, have {available}")]
    OutOfMemory { needed: usize, available: usize },
    #[error("Invalid position {position} for sequence length {length}")]
    InvalidPosition { position: usize, length: usize },
    #[error("Position {position} was evicted; first retained position is {retained_start}")]
    PositionEvicted {
        position: usize,
        retained_start: usize,
    },
    /// A read at a rewind target below the pinned attention-sink prefix.
    ///
    /// The rewind itself is legal, but it resets the sequence's window
    /// bookkeeping, and a read that does not mutate cannot reproduce the result.
    /// Refused rather than answered with a view the rewind would not produce.
    #[error(
        "cannot materialize sequence at position {position} without rewinding: it is inside the \
         {sink_len} pinned attention-sink tokens, and rewinding there resets the window \
         bookkeeping this read cannot reproduce. Fix by rewinding first and then materializing, \
         or by choosing a position at or above {sink_len}"
    )]
    RewindBelowSinkNotMaterializable {
        /// The requested read position.
        position: usize,
        /// The pinned sink prefix length.
        sink_len: usize,
    },
    #[error("Sliding-window size must be greater than zero")]
    InvalidWindowSize,
    #[error("Tensor storage is not configured for this cache")]
    TensorStorageNotConfigured,
    #[error(
        "Page {0} is not host-addressable; explicitly materialize it before requesting host slices"
    )]
    PageNotHostAddressable(PageId),
    #[error("KV page stores have incompatible storage layouts")]
    PageStoreLayoutMismatch,
    #[error("KV page store cannot copy from {from:?} to {to:?}")]
    PageStoreCopyUnsupported { from: Device, to: Device },
    #[error("KV page store factory returned residency {actual:?} when {requested:?} was requested")]
    PageStoreWrongResidency { requested: Device, actual: Device },
    #[error("KV page store allocation failed: {0}")]
    PageStoreAllocationFailed(String),
    #[error("Invalid KV tensor shape: {0}")]
    InvalidTensorShape(&'static str),
    #[error("Unsupported KV dtype: {0}")]
    UnsupportedKvDType(String),
    #[error(
        "Unsupported KV quantization axis '{0}': only per-token quantization preserves the \
         append-without-requantize invariant"
    )]
    UnsupportedQuantizationAxis(String),
    #[error("Invalid KV layer {layer} for model with {num_layers} layers")]
    InvalidKvLayer { layer: i32, num_layers: usize },
    #[error("Invalid KV quantization config: {0}")]
    InvalidQuantizationConfig(String),
    #[error("Page {0} not found")]
    PageNotFound(PageId),
    /// The KV `page_size` cannot serve as a PagedAttention `block_size`.
    ///
    /// `com.microsoft::PagedAttention` requires `block_size` to be a power of
    /// two and at least 16 (`check_kv_cache`). Refused rather than emitting an
    /// index plan a conforming kernel would reject.
    #[error(
        "page_size {block_size} cannot be a PagedAttention block_size (need power of two >= 16)"
    )]
    PagedInvalidBlockSize { block_size: usize },
    /// A windowed / attention-sink sequence was asked to emit a paged plan.
    ///
    /// The token-major slot mapping in this slice assumes contiguous positions
    /// from zero. Sink-pinned or slid sequences store a disjoint
    /// `[0, sink) ∪ [start, len)` span, so their absolute positions do not map
    /// linearly onto blocks. Refused with the offending bookkeeping rather than
    /// emitting slots that silently address the wrong tokens.
    #[error(
        "sequence {seq} is not contiguous (start {start}, sink_len {sink_len}); paged index \
         emission for windowed/attention-sink sequences is not implemented in this slice"
    )]
    PagedNonContiguousSequence {
        seq: SequenceId,
        start: usize,
        sink_len: usize,
    },
    /// More query tokens were requested than the sequence currently holds.
    #[error(
        "sequence {seq} query_len {query_len} exceeds its context length {context_len}; append \
         the tokens before emitting their slots"
    )]
    PagedQueryExceedsContext {
        seq: SequenceId,
        query_len: usize,
        context_len: usize,
    },
    /// The sequence's context needs more pages than are allocated.
    ///
    /// The plan is read-only and never allocates; the caller must reserve the
    /// pages (via append) before the plan can address their slots.
    #[error(
        "sequence {seq} needs {need_pages} pages for its context but only {have_pages} are \
         allocated; emission never allocates, so append the tokens first"
    )]
    PagedMissingPages {
        seq: SequenceId,
        need_pages: usize,
        have_pages: usize,
    },
    /// A physical page id does not fit in the op's int32 `block_table`.
    #[error("page id {page_id} does not fit in an int32 block_table entry")]
    PagedBlockIdOverflow { page_id: PageId },
    /// A physical slot (or a length) does not fit in an int32 index tensor.
    #[error("paged index value {slot} does not fit in an int32 index tensor")]
    PagedSlotOverflow { slot: i64 },
    /// A LATENT cache geometry is internally inconsistent.
    #[error("invalid LATENT cache geometry: {0}")]
    LatentGeometryInvalid(String),
}

#[cfg(test)]
mod tests {
    use super::{
        KvCapacityGrowth, KvCapacityGrowthBackend, ensure_kv_capacity, kv_capacity_bucket,
    };

    #[test]
    fn kv_capacity_bucket_rounds_to_power_of_two_min_256() {
        let hard = 32_768;
        assert_eq!(kv_capacity_bucket(0, hard), 256);
        assert_eq!(kv_capacity_bucket(1, hard), 256);
        assert_eq!(kv_capacity_bucket(128, hard), 256);
        assert_eq!(kv_capacity_bucket(256, hard), 256);
        assert_eq!(kv_capacity_bucket(257, hard), 512);
        assert_eq!(kv_capacity_bucket(5000, hard), 8192);
    }

    #[test]
    fn kv_capacity_bucket_caps_at_hard_max() {
        assert_eq!(kv_capacity_bucket(20_000, 32_768), 32_768);
        assert_eq!(kv_capacity_bucket(32_768, 32_768), 32_768);
        assert_eq!(kv_capacity_bucket(100, 128), 128);
        assert_eq!(kv_capacity_bucket(100, 0), 0);
    }

    #[test]
    fn kv_capacity_bucket_is_monotonic_and_within_bounds() {
        let hard = 4096;
        let mut previous = 0;
        for len in 0..8192 {
            let bucket = kv_capacity_bucket(len, hard);
            assert!(
                bucket >= previous,
                "len={len} bucket={bucket} previous={previous}"
            );
            assert!(bucket <= hard);
            previous = bucket;
        }
    }

    #[derive(Default)]
    struct RecordingBackend {
        current: usize,
        hard: usize,
        valid: usize,
        events: Vec<&'static str>,
        fail_mask: bool,
    }

    impl KvCapacityGrowthBackend for RecordingBackend {
        type Error = &'static str;
        type GrownBuffers = usize;
        type GrownMask = usize;

        fn current_capacity(&self) -> usize {
            self.current
        }

        fn hard_max_capacity(&self) -> usize {
            self.hard
        }

        fn valid_len(&self) -> usize {
            self.valid
        }

        fn capacity_exceeded(&self, _required: usize) -> Self::Error {
            "capacity exceeded"
        }

        fn build_grown_buffers(
            &mut self,
            new_capacity: usize,
            _valid_len: usize,
        ) -> Result<Self::GrownBuffers, Self::Error> {
            self.events.push("buffers");
            Ok(new_capacity)
        }

        fn build_grown_mask(
            &mut self,
            new_capacity: usize,
            _valid_len: usize,
        ) -> Result<Option<Self::GrownMask>, Self::Error> {
            self.events.push("mask");
            if self.fail_mask {
                return Err("mask failed");
            }
            Ok(Some(new_capacity))
        }

        fn invalidate_capture(&mut self) -> Result<(), Self::Error> {
            self.events.push("invalidate");
            Ok(())
        }

        fn commit_grown_capacity(
            &mut self,
            new_capacity: usize,
            _grown_buffers: Self::GrownBuffers,
            _grown_mask: Option<Self::GrownMask>,
        ) -> Result<(), Self::Error> {
            self.events.push("commit");
            self.current = new_capacity;
            Ok(())
        }
    }

    #[test]
    fn ensure_kv_capacity_orders_fallible_work_before_invalidation() {
        let mut backend = RecordingBackend {
            current: 256,
            hard: 4096,
            valid: 200,
            ..RecordingBackend::default()
        };
        let growth = ensure_kv_capacity(&mut backend, 257).unwrap();
        assert_eq!(
            growth,
            KvCapacityGrowth::Grew {
                old_capacity: 256,
                new_capacity: 512,
                valid_len: 200,
            }
        );
        assert_eq!(backend.events, ["buffers", "mask", "invalidate", "commit"]);
        assert_eq!(backend.current, 512);
    }

    #[test]
    fn ensure_kv_capacity_does_not_invalidate_or_commit_after_prepare_failure() {
        let mut backend = RecordingBackend {
            current: 256,
            hard: 4096,
            valid: 200,
            fail_mask: true,
            ..RecordingBackend::default()
        };
        assert_eq!(ensure_kv_capacity(&mut backend, 257), Err("mask failed"));
        assert_eq!(backend.events, ["buffers", "mask"]);
        assert_eq!(backend.current, 256);
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum InjectedFailure {
        Allocate,
        PrefixCopy,
        Mask,
        InvalidateCapture,
    }

    struct TransactionalFakeBackend {
        current: usize,
        hard: usize,
        valid: usize,
        bytes_per_token: usize,
        buffer_id: usize,
        capture_generation: usize,
        capture_valid: bool,
        events: Vec<&'static str>,
        fail_at: Option<InjectedFailure>,
    }

    impl TransactionalFakeBackend {
        fn new(fail_at: Option<InjectedFailure>) -> Self {
            Self {
                current: 256,
                hard: 4096,
                valid: 123,
                bytes_per_token: 32,
                buffer_id: 7,
                capture_generation: 11,
                capture_valid: true,
                events: Vec::new(),
                fail_at,
            }
        }

        fn translated_growth_error(&self, old: usize, new: usize, raw: &str) -> String {
            let new_bytes = new * self.bytes_per_token;
            let transient_peak = (old + new) * self.bytes_per_token;
            format!(
                "injected KV capacity growth failed while growing from {old} to {new} tokens: {raw}. \
                 The attempted new KV allocation is approximately {new_bytes} bytes and the transient peak is approximately {transient_peak} bytes. \
                 KV bytes/token: {}. The session state was left unchanged; reset or retry with a shorter prompt/max_new_tokens, set an explicit KV max length cap, or free VRAM used by other processes.",
                self.bytes_per_token
            )
        }

        fn assert_unchanged(&self) {
            assert_eq!(self.current, 256);
            assert_eq!(self.valid, 123);
            assert_eq!(self.buffer_id, 7);
            assert_eq!(self.capture_generation, 11);
            assert!(self.capture_valid);
        }
    }

    impl KvCapacityGrowthBackend for TransactionalFakeBackend {
        type Error = String;
        type GrownBuffers = usize;
        type GrownMask = usize;

        fn current_capacity(&self) -> usize {
            self.current
        }

        fn hard_max_capacity(&self) -> usize {
            self.hard
        }

        fn valid_len(&self) -> usize {
            self.valid
        }

        fn capacity_exceeded(&self, required: usize) -> Self::Error {
            format!("capacity exceeded at {required}")
        }

        fn build_grown_buffers(
            &mut self,
            new_capacity: usize,
            _valid_len: usize,
        ) -> Result<Self::GrownBuffers, Self::Error> {
            self.events.push("allocate");
            if self.fail_at == Some(InjectedFailure::Allocate) {
                return Err(self.translated_growth_error(self.current, new_capacity, "raw OOM"));
            }
            self.events.push("prefix-copy");
            if self.fail_at == Some(InjectedFailure::PrefixCopy) {
                return Err(self.translated_growth_error(
                    self.current,
                    new_capacity,
                    "raw copy failure",
                ));
            }
            Ok(new_capacity + 1000)
        }

        fn build_grown_mask(
            &mut self,
            new_capacity: usize,
            _valid_len: usize,
        ) -> Result<Option<Self::GrownMask>, Self::Error> {
            self.events.push("mask");
            if self.fail_at == Some(InjectedFailure::Mask) {
                return Err(self.translated_growth_error(
                    self.current,
                    new_capacity,
                    "raw mask allocation failure",
                ));
            }
            Ok(Some(new_capacity + 2000))
        }

        fn invalidate_capture(&mut self) -> Result<(), Self::Error> {
            self.events.push("invalidate");
            if self.fail_at == Some(InjectedFailure::InvalidateCapture) {
                return Err(self.translated_growth_error(
                    self.current,
                    self.current * 2,
                    "raw capture release failure",
                ));
            }
            self.capture_valid = false;
            Ok(())
        }

        fn commit_grown_capacity(
            &mut self,
            new_capacity: usize,
            grown_buffers: Self::GrownBuffers,
            grown_mask: Option<Self::GrownMask>,
        ) -> Result<(), Self::Error> {
            self.events.push("commit");
            assert_eq!(grown_buffers, new_capacity + 1000);
            assert_eq!(grown_mask, Some(new_capacity + 2000));
            self.current = new_capacity;
            self.buffer_id = grown_buffers;
            self.capture_generation += 1;
            self.capture_valid = true;
            Ok(())
        }
    }

    #[test]
    fn injected_allocate_failure_is_actionable_transactional_and_retryable() {
        let mut backend = TransactionalFakeBackend::new(Some(InjectedFailure::Allocate));
        let error = ensure_kv_capacity(&mut backend, 257).unwrap_err();

        assert!(error.contains("growing from 256 to 512 tokens"), "{error}");
        assert!(
            error.contains("transient peak is approximately 24576 bytes"),
            "{error}"
        );
        assert!(error.contains("KV bytes/token: 32"), "{error}");
        assert!(
            error.contains("session state was left unchanged"),
            "{error}"
        );
        assert!(error.contains("shorter prompt/max_new_tokens"), "{error}");
        assert!(error.contains("free VRAM"), "{error}");
        assert_eq!(backend.events, ["allocate"]);
        backend.assert_unchanged();

        backend.fail_at = None;
        let growth = ensure_kv_capacity(&mut backend, 257).unwrap();
        assert_eq!(
            growth,
            KvCapacityGrowth::Grew {
                old_capacity: 256,
                new_capacity: 512,
                valid_len: 123,
            }
        );
        assert_eq!(backend.current, 512);
        assert_eq!(backend.buffer_id, 1512);
        assert_eq!(backend.capture_generation, 12);
        assert!(backend.capture_valid);
    }

    #[test]
    fn injected_mid_sequence_failures_do_not_commit_or_invalidate_capture() {
        for failure in [
            InjectedFailure::PrefixCopy,
            InjectedFailure::Mask,
            InjectedFailure::InvalidateCapture,
        ] {
            let mut backend = TransactionalFakeBackend::new(Some(failure));
            let error = ensure_kv_capacity(&mut backend, 257).unwrap_err();
            assert!(
                error.contains("session state was left unchanged"),
                "{error}"
            );
            backend.assert_unchanged();
            assert!(!backend.events.contains(&"commit"));
            if failure != InjectedFailure::InvalidateCapture {
                assert!(!backend.events.contains(&"invalidate"));
            }
        }
    }
}
