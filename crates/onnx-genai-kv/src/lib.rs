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
pub mod prefix_cache;
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
    KvDType, KvKind, KvQuantConfig, LayerKvDType, LayerTensorConfig, Page, PageId, PageStats,
    PageTable, PageTensorConfig, PageUsage, SequenceUsage,
};
pub use paged_cache::{LayerKv, MaterializedKv, MaterializedLayerKv, PagedKvCache};
pub use prefix_cache::PrefixCache;

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

/// Device tier for page storage.
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
    #[error("Sliding-window size must be greater than zero")]
    InvalidWindowSize,
    #[error("Tensor storage is not configured for this cache")]
    TensorStorageNotConfigured,
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
}
