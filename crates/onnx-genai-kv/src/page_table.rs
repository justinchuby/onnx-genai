//! Page table: maps sequences to physical pages.

use std::collections::HashMap;
use crate::{Device, SequenceId};

/// Unique page identifier.
pub type PageId = u32;

/// A physical page holding KV data for a fixed number of tokens.
#[derive(Debug)]
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
}

/// The page table manages the mapping from logical sequences to physical pages.
pub struct PageTable {
    /// Logical sequence → ordered list of page IDs.
    pub sequences: HashMap<SequenceId, Vec<PageId>>,
    /// All physical pages.
    pub pages: HashMap<PageId, Page>,
    /// Free pages per device.
    free_pages: HashMap<Device, Vec<PageId>>,
    /// Next page ID to allocate.
    next_id: PageId,
    /// Tokens per page.
    pub page_size: usize,
    /// Monotonic clock for LRU.
    clock: u64,
}

impl PageTable {
    pub fn new(page_size: usize, num_gpu_pages: usize) -> Self {
        let mut pages = HashMap::new();
        let mut free_pages = vec![];

        for i in 0..num_gpu_pages {
            let id = i as PageId;
            pages.insert(id, Page {
                id,
                ref_count: 0,
                device: Device::Gpu(0),
                filled: 0,
                last_access: 0,
            });
            free_pages.push(id);
        }

        let mut free_map = HashMap::new();
        free_map.insert(Device::Gpu(0), free_pages);

        Self {
            sequences: HashMap::new(),
            pages,
            free_pages: free_map,
            next_id: num_gpu_pages as PageId,
            page_size,
            clock: 0,
        }
    }

    /// Allocate a new page on the specified device.
    pub fn allocate(&mut self, device: Device) -> Option<PageId> {
        if let Some(free_list) = self.free_pages.get_mut(&device) {
            if let Some(page_id) = free_list.pop() {
                if let Some(page) = self.pages.get_mut(&page_id) {
                    page.ref_count = 1;
                    page.filled = 0;
                    self.clock += 1;
                    page.last_access = self.clock;
                }
                return Some(page_id);
            }
        }
        None
    }

    /// Free a page (decrement ref_count; actually free when it hits 0).
    pub fn free(&mut self, page_id: PageId) {
        if let Some(page) = self.pages.get_mut(&page_id) {
            page.ref_count = page.ref_count.saturating_sub(1);
            if page.ref_count == 0 {
                let device = page.device;
                self.free_pages.entry(device).or_default().push(page_id);
            }
        }
    }

    /// Get the page list for a sequence.
    pub fn get_sequence(&self, seq: SequenceId) -> Option<&[PageId]> {
        self.sequences.get(&seq).map(|v| v.as_slice())
    }

    /// Create a new sequence (empty).
    pub fn create_sequence(&mut self, seq: SequenceId) {
        self.sequences.insert(seq, Vec::new());
    }

    /// Append a page to a sequence.
    pub fn push_page(&mut self, seq: SequenceId, page_id: PageId) {
        if let Some(pages) = self.sequences.get_mut(&seq) {
            pages.push(page_id);
        }
    }

    /// Remove a sequence and free its pages.
    pub fn remove_sequence(&mut self, seq: SequenceId) -> Vec<PageId> {
        self.sequences.remove(&seq).unwrap_or_default()
    }

    /// Number of free pages on a device.
    pub fn free_count(&self, device: Device) -> usize {
        self.free_pages.get(&device).map_or(0, |v| v.len())
    }

    /// Total number of pages.
    pub fn total_pages(&self) -> usize {
        self.pages.len()
    }
}
