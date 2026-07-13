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

pub mod connector;
pub mod fp8;
pub mod local_tiered;
pub mod page_table;
pub mod paged_cache;
pub mod prefix_cache;
pub mod tiered;

pub use connector::{
    CachePriority, CompressionFormat, ConnectorCapabilities, ConnectorError, ConnectorHealth,
    ConnectorResult, DEFAULT_CHUNK_SIZE, FetchedKv, KvCacheConnector, KvCacheKey, KvCacheLocation,
    KvStoreEntry, KvTensorRef, NullConnector, TokenChunk, chunk_tokens, hash_tokens,
};
pub use fp8::{Fp8Format, decode_f32 as decode_fp8, encode_f32 as encode_fp8};
pub use local_tiered::{DiskTierConfig, LocalTieredConfig, LocalTieredConnector};
pub use page_table::{
    KvDType, KvKind, KvQuantConfig, LayerKvDType, Page, PageId, PageTable, PageTensorConfig,
};
pub use paged_cache::{LayerKv, MaterializedKv, MaterializedLayerKv, PagedKvCache};
pub use prefix_cache::PrefixCache;

/// Sequence identifier.
pub type SequenceId = u64;

/// Token identifier.
pub type TokenId = u32;

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
