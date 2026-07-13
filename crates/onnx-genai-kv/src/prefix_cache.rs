//! Prefix cache using a radix trie.
//!
//! Prefix nodes reference physical `PageId`s owned by `PageTable`. Inserting a
//! cached prefix retains the pages for the cache itself. Sharing a prefix via
//! `lookup_shared` retains the same pages again for the borrowing sequence;
//! callers must later `release_shared` when that sequence stops using them.
//! Fork/write Copy-on-Write is page-table driven: shared pages have
//! `ref_count > 1`, and writers must copy before mutating.

use crate::{TokenId, page_table::PageId, page_table::PageTable};
use std::collections::HashMap;

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
            for page_id in &pages {
                page_table.free(*page_id);
            }
            released.extend(pages);
        }
        released
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
}
