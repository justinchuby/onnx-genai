//! Prefix cache using a radix trie.
//!
//! Prefix nodes reference physical `PageId`s owned by `PageTable`. Inserting a
//! cached prefix retains the pages for the cache itself. Sharing a prefix via
//! `lookup_shared` retains the same pages again for the borrowing sequence;
//! callers must later `release_shared` when that sequence stops using them.
//! Fork/write Copy-on-Write is page-table driven: shared pages have
//! `ref_count > 1`, and writers must copy before mutating.

use crate::{TokenId, page_table::PageId, page_table::PageTable};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

/// Longest-prefix cache lookup result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMatch {
    pub matched_tokens: usize,
    pub page_ids: Vec<PageId>,
}

/// Radix trie for prefix caching.
/// Shares KV pages for common token prefixes across sessions.
pub struct PrefixCache {
    root: TrieNode,
    clock: u64,
}

struct TrieNode {
    children: HashMap<TokenId, Box<TrieNode>>,
    /// KV page IDs for the full prefix ending at this node.
    page_ids: Vec<PageId>,
    /// Optional non-KV state attached to the same semantic prefix.
    snapshot: Option<Arc<dyn Any + Send + Sync>>,
    /// Number of active shared sequence references from `lookup_shared`.
    ref_count: usize,
    last_access: u64,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(0),
            clock: 0,
        }
    }

    /// Find the longest cached prefix for a token sequence without changing
    /// page-table refcounts.
    pub fn lookup(&self, tokens: &[TokenId]) -> (usize, Vec<PageId>) {
        let mut node = &self.root;
        let mut best = PrefixMatch {
            matched_tokens: 0,
            page_ids: Vec::new(),
        };

        for (idx, &token) in tokens.iter().enumerate() {
            match node.children.get(&token) {
                Some(child) => {
                    node = child;
                    if !node.page_ids.is_empty() {
                        best = PrefixMatch {
                            matched_tokens: idx + 1,
                            page_ids: node.page_ids.clone(),
                        };
                    }
                }
                None => break,
            }
        }

        (best.matched_tokens, best.page_ids)
    }

    /// Insert a prefix with its computed KV pages without page-table refcount
    /// changes. Prefer `insert_pages` when physical page ownership matters.
    pub fn insert(&mut self, tokens: &[TokenId], page_ids: &[PageId]) {
        self.insert_inner(tokens, page_ids);
    }

    /// Insert a prefix and retain the physical pages for cache ownership.
    pub fn insert_pages(
        &mut self,
        tokens: &[TokenId],
        page_ids: &[PageId],
        page_table: &mut PageTable,
    ) -> PrefixMatch {
        for &page_id in page_ids {
            page_table.retain(page_id);
        }
        self.insert_inner(tokens, page_ids)
    }

    /// Attach an opaque state snapshot to the trie node for `tokens`.
    ///
    /// The prefix cache owns one strong reference. Callers put any memory lease
    /// required by the snapshot inside the value, so replacing or evicting the
    /// node drops that lease through ordinary ownership.
    pub fn insert_snapshot<T>(&mut self, tokens: &[TokenId], snapshot: Arc<T>) -> PrefixMatch
    where
        T: Any + Send + Sync + 'static,
    {
        self.clock += 1;
        let mut node = &mut self.root;
        for &token in tokens {
            node = node
                .children
                .entry(token)
                .or_insert_with(|| Box::new(TrieNode::new(self.clock)));
            node.last_access = self.clock;
        }
        node.snapshot = Some(snapshot);
        PrefixMatch {
            matched_tokens: tokens.len(),
            page_ids: node.page_ids.clone(),
        }
    }

    /// Find the longest cached prefix carrying a snapshot of type `T`.
    pub fn lookup_snapshot<T>(&mut self, tokens: &[TokenId]) -> Option<(usize, Arc<T>)>
    where
        T: Any + Send + Sync + 'static,
    {
        self.clock += 1;
        let mut node = &mut self.root;
        let mut best_depth = 0;
        let mut best_snapshot = None;

        for (idx, &token) in tokens.iter().enumerate() {
            match node.children.get_mut(&token) {
                Some(child) => {
                    node = child;
                    node.last_access = self.clock;
                    if let Some(snapshot) = &node.snapshot {
                        best_depth = idx + 1;
                        best_snapshot = Some(Arc::clone(snapshot));
                    }
                }
                None => break,
            }
        }

        best_snapshot.and_then(|snapshot| {
            Arc::downcast::<T>(snapshot)
                .ok()
                .map(|snapshot| (best_depth, snapshot))
        })
    }

    /// Find the longest cached prefix and retain its pages for a sharing
    /// sequence. The returned pages can be attached to the sequence page list.
    pub fn lookup_shared(&mut self, tokens: &[TokenId], page_table: &mut PageTable) -> PrefixMatch {
        self.clock += 1;
        let mut node = &mut self.root;
        let mut best_depth = 0;
        let mut best_pages = Vec::new();

        for (idx, &token) in tokens.iter().enumerate() {
            match node.children.get_mut(&token) {
                Some(child) => {
                    node = child;
                    node.last_access = self.clock;
                    if !node.page_ids.is_empty() {
                        best_depth = idx + 1;
                        best_pages = node.page_ids.clone();
                    }
                }
                None => break,
            }
        }

        if best_depth > 0 {
            if let Some(best_node) = self.find_node_mut(&tokens[..best_depth]) {
                best_node.ref_count += 1;
            }
            for &page_id in &best_pages {
                page_table.retain(page_id);
            }
        }

        PrefixMatch {
            matched_tokens: best_depth,
            page_ids: best_pages,
        }
    }

    /// Release a previously shared prefix returned by `lookup_shared`.
    pub fn release_shared(
        &mut self,
        tokens: &[TokenId],
        matched_tokens: usize,
        page_table: &mut PageTable,
    ) -> Vec<PageId> {
        if matched_tokens == 0 || matched_tokens > tokens.len() {
            return Vec::new();
        }
        let Some(node) = self.find_node_mut(&tokens[..matched_tokens]) else {
            return Vec::new();
        };
        if node.ref_count > 0 {
            node.ref_count -= 1;
            for &page_id in &node.page_ids {
                page_table.free(page_id);
            }
        }
        node.page_ids.clone()
    }

    /// Evict least-recently-used inactive cached prefixes until at least
    /// `target_pages` page references have been released from the cache.
    pub fn evict_lru(&mut self, target_pages: usize, page_table: &mut PageTable) -> Vec<PageId> {
        let mut released = Vec::new();
        while released.len() < target_pages {
            let Some(path) = self.find_lru_evictable_path() else {
                break;
            };
            let Some(node) = self.find_node_mut(&path) else {
                break;
            };
            if node.ref_count != 0 || node.page_ids.is_empty() {
                break;
            }
            let pages = std::mem::take(&mut node.page_ids);
            node.snapshot = None;
            for page_id in &pages {
                page_table.free(*page_id);
            }
            page_table.note_prefix_eviction(pages.len() as u64);
            released.extend(pages);
        }
        released
    }

    /// Drop the least-recently-used inactive snapshot, if one exists.
    pub fn evict_lru_snapshot(&mut self) -> bool {
        let Some(path) = self.find_lru_snapshot_path() else {
            return false;
        };
        let Some(node) = self.find_node_mut(&path) else {
            return false;
        };
        node.snapshot.take().is_some()
    }

    /// Detach an exact cached prefix, returning the pages it referenced.
    ///
    /// Unlike [`evict_lru`](Self::evict_lru) this targets a specific prefix and
    /// ignores `ref_count`, so it is the primitive an owner uses for an explicit
    /// remove. It only clears the node's page list and shared ref count; it does
    /// **not** touch page-table ref counts (the caller owns that accounting).
    pub fn remove(&mut self, tokens: &[TokenId]) -> Vec<PageId> {
        let Some(node) = self.find_node_mut(tokens) else {
            return Vec::new();
        };
        node.ref_count = 0;
        node.snapshot = None;
        std::mem::take(&mut node.page_ids)
    }

    /// Number of trie nodes excluding the root.
    pub fn len(&self) -> usize {
        Self::count_nodes(&self.root)
    }

    pub fn is_empty(&self) -> bool {
        self.root.children.is_empty()
    }

    fn insert_inner(&mut self, tokens: &[TokenId], page_ids: &[PageId]) -> PrefixMatch {
        self.clock += 1;
        let mut node = &mut self.root;
        for &token in tokens {
            node = node
                .children
                .entry(token)
                .or_insert_with(|| Box::new(TrieNode::new(self.clock)));
            node.last_access = self.clock;
        }
        node.page_ids = page_ids.to_vec();
        PrefixMatch {
            matched_tokens: tokens.len(),
            page_ids: page_ids.to_vec(),
        }
    }

    fn find_node_mut(&mut self, tokens: &[TokenId]) -> Option<&mut TrieNode> {
        let mut node = &mut self.root;
        for &token in tokens {
            node = node.children.get_mut(&token)?;
        }
        Some(node)
    }

    fn find_lru_evictable_path(&self) -> Option<Vec<TokenId>> {
        let mut best: Option<(u64, Vec<TokenId>)> = None;
        let mut path = Vec::new();
        Self::visit_evictable(&self.root, &mut path, &mut best);
        best.map(|(_, path)| path)
    }

    fn find_lru_snapshot_path(&self) -> Option<Vec<TokenId>> {
        let mut best: Option<(u64, Vec<TokenId>)> = None;
        let mut path = Vec::new();
        Self::visit_snapshot_evictable(&self.root, &mut path, &mut best);
        best.map(|(_, path)| path)
    }

    fn visit_evictable(
        node: &TrieNode,
        path: &mut Vec<TokenId>,
        best: &mut Option<(u64, Vec<TokenId>)>,
    ) {
        if !node.page_ids.is_empty()
            && node.ref_count == 0
            && best
                .as_ref()
                .is_none_or(|(best_access, _)| node.last_access < *best_access)
        {
            *best = Some((node.last_access, path.clone()));
        }
        let mut children = node.children.iter().collect::<Vec<_>>();
        children.sort_by_key(|(token, _)| **token);
        for (&token, child) in children {
            path.push(token);
            Self::visit_evictable(child, path, best);
            path.pop();
        }
    }

    fn visit_snapshot_evictable(
        node: &TrieNode,
        path: &mut Vec<TokenId>,
        best: &mut Option<(u64, Vec<TokenId>)>,
    ) {
        if node.snapshot.is_some()
            && node.ref_count == 0
            && best
                .as_ref()
                .is_none_or(|(best_access, _)| node.last_access < *best_access)
        {
            *best = Some((node.last_access, path.clone()));
        }
        let mut children = node.children.iter().collect::<Vec<_>>();
        children.sort_by_key(|(token, _)| **token);
        for (&token, child) in children {
            path.push(token);
            Self::visit_snapshot_evictable(child, path, best);
            path.pop();
        }
    }

    fn count_nodes(node: &TrieNode) -> usize {
        node.children
            .values()
            .map(|child| 1 + Self::count_nodes(child))
            .sum()
    }
}

impl TrieNode {
    fn new(last_access: u64) -> Self {
        Self {
            children: HashMap::new(),
            page_ids: Vec::new(),
            snapshot: None,
            ref_count: 0,
            last_access,
        }
    }
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, PageTable};

    fn page_table() -> PageTable {
        PageTable::new(2, 8)
    }

    #[test]
    fn prefix_insert_lookup_partial_and_full_match() {
        let mut cache = PrefixCache::new();
        cache.insert(&[1, 2, 3], &[10, 11]);
        cache.insert(&[1, 2, 3, 4, 5], &[10, 11, 12]);

        assert_eq!(cache.lookup(&[1, 2]), (0, Vec::new()));
        assert_eq!(cache.lookup(&[1, 2, 3, 9]), (3, vec![10, 11]));
        assert_eq!(cache.lookup(&[1, 2, 3, 4, 5]), (5, vec![10, 11, 12]));
    }

    #[test]
    fn lookup_shared_increments_and_release_decrements_page_refs() {
        let mut table = page_table();
        let p0 = table.allocate(Device::Gpu(0)).unwrap();
        let p1 = table.allocate(Device::Gpu(0)).unwrap();
        assert_eq!(table.pages[&p0].ref_count, 1);

        let mut cache = PrefixCache::new();
        cache.insert_pages(&[42, 43], &[p0, p1], &mut table);
        assert_eq!(table.pages[&p0].ref_count, 2); // sequence + prefix cache

        let matched = cache.lookup_shared(&[42, 43, 99], &mut table);
        assert_eq!(matched.matched_tokens, 2);
        assert_eq!(matched.page_ids, vec![p0, p1]);
        assert_eq!(table.pages[&p0].ref_count, 3); // plus shared sequence

        let released = cache.release_shared(&[42, 43, 99], matched.matched_tokens, &mut table);
        assert_eq!(released, vec![p0, p1]);
        assert_eq!(table.pages[&p0].ref_count, 2);
    }

    #[test]
    fn eviction_skips_active_refs_and_releases_lru_pages() {
        let mut table = page_table();
        let p0 = table.allocate(Device::Gpu(0)).unwrap();
        let p1 = table.allocate(Device::Gpu(0)).unwrap();
        let p2 = table.allocate(Device::Gpu(0)).unwrap();
        let mut cache = PrefixCache::new();
        cache.insert_pages(&[1], &[p0], &mut table);
        cache.insert_pages(&[2], &[p1], &mut table);
        cache.insert_pages(&[3], &[p2], &mut table);
        let active = cache.lookup_shared(&[1], &mut table);
        assert_eq!(active.page_ids, vec![p0]);

        let evicted = cache.evict_lru(2, &mut table);

        assert_eq!(evicted, vec![p1, p2]);
        assert_eq!(table.pages[&p0].ref_count, 3);
        assert_eq!(table.pages[&p1].ref_count, 1);
        assert_eq!(table.pages[&p2].ref_count, 1);
        assert_eq!(cache.lookup(&[2]), (0, Vec::new()));
        assert_eq!(cache.lookup(&[3]), (0, Vec::new()));
    }

    #[test]
    fn release_and_eviction_are_safe_for_missing_or_active_entries() {
        let mut table = page_table();
        let page = table.allocate(Device::Gpu(0)).unwrap();
        let mut cache = PrefixCache::new();
        cache.insert_pages(&[7, 8], &[page], &mut table);

        assert!(cache.release_shared(&[7, 8], 0, &mut table).is_empty());
        assert!(cache.release_shared(&[7], 2, &mut table).is_empty());
        assert!(cache.release_shared(&[9, 9], 2, &mut table).is_empty());

        let matched = cache.lookup_shared(&[7, 8], &mut table);
        assert_eq!(cache.evict_lru(1, &mut table), Vec::<PageId>::new());
        assert_eq!(
            cache.release_shared(&[7, 8], matched.matched_tokens, &mut table),
            vec![page]
        );
        assert_eq!(cache.evict_lru(1, &mut table), vec![page]);
        assert!(cache.evict_lru(1, &mut table).is_empty());
    }

    #[test]
    fn snapshots_are_looked_up_at_the_longest_semantic_prefix() {
        let mut cache = PrefixCache::new();
        cache.insert_snapshot(&[1, 2], Arc::new(String::from("system")));
        cache.insert_snapshot(&[1, 2, 3, 4], Arc::new(String::from("fork")));

        let (matched, snapshot) = cache
            .lookup_snapshot::<String>(&[1, 2, 3, 4, 99])
            .expect("snapshot hit");

        assert_eq!(matched, 4);
        assert_eq!(snapshot.as_str(), "fork");
    }

    #[test]
    fn interleaved_prefix_borrow_release_and_eviction_stress() {
        const PREFIX_COUNT: usize = 1024;
        const LOOKUP_ROUNDS: usize = 128;

        fn prefix(index: usize) -> [TokenId; 3] {
            [
                7,
                (index / 256) as TokenId + 10,
                (index % 256) as TokenId + 100,
            ]
        }

        let mut table = PageTable::new(1, PREFIX_COUNT);
        let pages = (0..PREFIX_COUNT)
            .map(|_| {
                table
                    .allocate(Device::Gpu(0))
                    .expect("stress pool has room")
            })
            .collect::<Vec<_>>();
        let mut cache = PrefixCache::new();
        for (index, &page) in pages.iter().enumerate() {
            cache.insert_pages(&prefix(index), &[page], &mut table);
            assert_eq!(table.pages[&page].ref_count, 2);
        }

        // EngineDriver serializes logical requests through this same mutation
        // path. Exercise more than 100k interleaved borrower lifetimes.
        for round in 0..LOOKUP_ROUNDS {
            for (index, &page) in pages.iter().enumerate() {
                let mut query = prefix(index).to_vec();
                query.push(10_000 + round as TokenId);
                let matched = cache.lookup_shared(&query, &mut table);
                assert_eq!(matched.matched_tokens, 3);
                assert_eq!(matched.page_ids, vec![page]);
                assert_eq!(table.pages[&page].ref_count, 3);
                assert_eq!(
                    cache.release_shared(&query, matched.matched_tokens, &mut table),
                    vec![page]
                );
                assert_eq!(table.pages[&page].ref_count, 2);
            }
        }

        // Keep one quarter of the prefixes borrowed while LRU reclaims every
        // inactive entry, then release and reclaim the remainder.
        let mut active = Vec::new();
        for index in (0..PREFIX_COUNT).step_by(4) {
            let query = prefix(index).to_vec();
            let matched = cache.lookup_shared(&query, &mut table);
            active.push((query, matched.matched_tokens, pages[index]));
        }
        let evicted = cache.evict_lru(PREFIX_COUNT, &mut table);
        assert_eq!(evicted.len(), PREFIX_COUNT - active.len());
        for &(ref query, matched_tokens, page) in &active {
            assert_eq!(table.pages[&page].ref_count, 3);
            cache.release_shared(query, matched_tokens, &mut table);
            assert_eq!(table.pages[&page].ref_count, 2);
        }
        assert_eq!(
            cache.evict_lru(PREFIX_COUNT, &mut table).len(),
            active.len()
        );
        assert!(pages.iter().all(|page| table.pages[page].ref_count == 1));
    }
}

#[cfg(test)]
mod eviction_stats_tests {
    use super::*;
    use crate::Device;

    #[test]
    fn reclaiming_a_cached_prefix_is_counted_as_an_eviction() {
        let mut table = PageTable::new(16, 8);
        let pages: Vec<PageId> = (0..3)
            .map(|_| table.allocate(Device::Gpu(0)).expect("pool has room"))
            .collect();
        let mut cache = PrefixCache::new();
        cache.insert(&[1, 2, 3], &pages);
        let before = table.stats().prefix_evictions;

        let released = cache.evict_lru(3, &mut table);

        assert_eq!(released.len(), pages.len());
        assert_eq!(
            table.stats().prefix_evictions - before,
            pages.len() as u64,
            "pages reclaimed from the prefix cache must be distinguishable from ordinary frees"
        );
    }
}
