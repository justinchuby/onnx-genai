//! Paged KV cache implementation.

use crate::{
    page_table::{PageId, PageTable},
    CacheCheckpoint, Device, EvictionPolicy, KvCacheOps, KvError, SequenceId,
};

/// Paged KV cache manager.
pub struct PagedKvCache {
    pub page_table: PageTable,
    next_seq_id: SequenceId,
}

impl PagedKvCache {
    pub fn new(page_size: usize, num_gpu_pages: usize) -> Self {
        Self {
            page_table: PageTable::new(page_size, num_gpu_pages),
            next_seq_id: 0,
        }
    }

    /// Create a new sequence, returns its ID.
    pub fn create_sequence(&mut self) -> SequenceId {
        let id = self.next_seq_id;
        self.next_seq_id += 1;
        self.page_table.create_sequence(id);
        id
    }

    /// Evict pages to free memory. Returns number of pages freed.
    pub fn evict(&mut self, _policy: EvictionPolicy, _target: usize) -> usize {
        // TODO: implement eviction strategies
        0
    }
}

impl KvCacheOps for PagedKvCache {
    fn rewind_to(&mut self, seq: SequenceId, position: usize) -> Result<(), KvError> {
        let pages = self.page_table.get_sequence(seq)
            .ok_or(KvError::SequenceNotFound(seq))?;

        let page_size = self.page_table.page_size;
        let pages_needed = (position + page_size - 1) / page_size;

        // Free pages beyond the target position
        let current_pages: Vec<PageId> = pages.to_vec();
        for &page_id in current_pages.iter().skip(pages_needed) {
            self.page_table.free(page_id);
        }

        // Truncate sequence's page list
        if let Some(seq_pages) = self.page_table.sequences.get_mut(&seq) {
            seq_pages.truncate(pages_needed);
        }

        Ok(())
    }

    fn fork(&mut self, source: SequenceId, _position: usize) -> Result<SequenceId, KvError> {
        let source_pages = self.page_table.get_sequence(source)
            .ok_or(KvError::SequenceNotFound(source))?
            .to_vec();

        let new_seq = self.create_sequence();

        // CoW: share pages by incrementing ref_count
        for &page_id in &source_pages {
            if let Some(page) = self.page_table.pages.get_mut(&page_id) {
                page.ref_count += 1;
            }
            self.page_table.push_page(new_seq, page_id);
        }

        Ok(new_seq)
    }

    fn checkpoint(&self, seq: SequenceId) -> Result<CacheCheckpoint, KvError> {
        let pages = self.page_table.get_sequence(seq)
            .ok_or(KvError::SequenceNotFound(seq))?;

        Ok(CacheCheckpoint {
            seq,
            position: pages.len() * self.page_table.page_size,
            page_ids: pages.to_vec(),
        })
    }

    fn restore(&mut self, seq: SequenceId, checkpoint: CacheCheckpoint) -> Result<(), KvError> {
        // Rewind to checkpoint position
        self.rewind_to(seq, checkpoint.position)
    }

    fn append(&mut self, seq: SequenceId, num_tokens: usize) -> Result<(), KvError> {
        let _ = self.page_table.get_sequence(seq)
            .ok_or(KvError::SequenceNotFound(seq))?;

        let page_size = self.page_table.page_size;

        // Allocate new pages as needed
        let mut remaining = num_tokens;
        while remaining > 0 {
            let page_id = self.page_table.allocate(Device::Gpu(0))
                .ok_or(KvError::OutOfMemory { needed: 1, available: 0 })?;
            self.page_table.push_page(seq, page_id);
            remaining = remaining.saturating_sub(page_size);
        }

        Ok(())
    }

    fn len(&self, seq: SequenceId) -> Result<usize, KvError> {
        let pages = self.page_table.get_sequence(seq)
            .ok_or(KvError::SequenceNotFound(seq))?;
        Ok(pages.len() * self.page_table.page_size)
    }

    fn remove(&mut self, seq: SequenceId) -> Result<(), KvError> {
        let pages = self.page_table.remove_sequence(seq);
        for page_id in pages {
            self.page_table.free(page_id);
        }
        Ok(())
    }
}
