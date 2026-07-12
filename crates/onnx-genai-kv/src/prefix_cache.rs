//! Prefix cache using a radix trie.

use std::collections::HashMap;
use crate::{TokenId, page_table::PageId};

/// Radix trie for prefix caching.
/// Shares KV pages for common token prefixes across sessions.
pub struct PrefixCache {
    root: TrieNode,
}

struct TrieNode {
    children: HashMap<TokenId, Box<TrieNode>>,
    /// KV page IDs for the tokens leading to this node.
    page_ids: Vec<PageId>,
    /// Number of active references.
    ref_count: usize,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self {
            root: TrieNode {
                children: HashMap::new(),
                page_ids: Vec::new(),
                ref_count: 0,
            },
        }
    }

    /// Find the longest cached prefix for a token sequence.
    /// Returns (number of tokens matched, page IDs for the prefix).
    pub fn lookup(&self, tokens: &[TokenId]) -> (usize, Vec<PageId>) {
        let mut node = &self.root;
        let mut matched = 0;
        let mut pages = Vec::new();

        for &token in tokens {
            match node.children.get(&token) {
                Some(child) => {
                    node = child;
                    matched += 1;
                    pages.extend_from_slice(&node.page_ids);
                }
                None => break,
            }
        }

        (matched, pages)
    }

    /// Insert a prefix with its computed KV pages.
    pub fn insert(&mut self, tokens: &[TokenId], page_ids: &[PageId]) {
        let mut node = &mut self.root;

        for (i, &token) in tokens.iter().enumerate() {
            node = node.children
                .entry(token)
                .or_insert_with(|| Box::new(TrieNode {
                    children: HashMap::new(),
                    page_ids: Vec::new(),
                    ref_count: 0,
                }));

            // Assign page IDs proportionally (simplified)
            if i < page_ids.len() {
                if node.page_ids.is_empty() {
                    node.page_ids.push(page_ids[i]);
                }
            }
        }

        node.ref_count += 1;
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        self.count_nodes(&self.root)
    }

    pub fn is_empty(&self) -> bool {
        self.root.children.is_empty()
    }

    fn count_nodes(&self, node: &TrieNode) -> usize {
        let mut count = node.children.len();
        for child in node.children.values() {
            count += self.count_nodes(child);
        }
        count
    }
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new()
    }
}
