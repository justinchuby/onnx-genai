//! Paged KV cache manager.
//!
//! Implements PagedAttention-style memory management with:
//! - Fixed-size page allocation (eliminates fragmentation)
//! - Copy-on-Write forking (cheap session branching)
//! - Tiered storage (GPU → CPU → Disk)
//! - Prefix sharing via radix trie
//! - Rewind/checkpoint operations for speculative decoding

pub mod page_table;
pub mod paged_cache;
pub mod prefix_cache;
pub mod tiered;

pub use page_table::{Page, PageId, PageTable};
pub use paged_cache::PagedKvCache;
pub use prefix_cache::PrefixCache;

/// Sequence identifier.
pub type SequenceId = u64;

/// Token identifier.
pub type TokenId = u32;

/// Device tier for page storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Gpu(usize),  // GPU index
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
}
