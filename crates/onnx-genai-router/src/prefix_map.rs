//! Prefix-hash → node co-location map (see `docs/DESIGN.md` §34.4/§34.5).
//!
//! Sessions that share a system-prompt prefix are placed on the same node so
//! they can share KV pages. The router keys this on a `u64` prefix hash. The
//! hash is computed by [`hash_system_prompt`] using a **fixed FNV-1a** so it is
//! stable across processes and Rust versions (unlike `DefaultHasher`, which is
//! randomized/unspecified). The router treats the prompt as an opaque byte
//! string — it is fully model-agnostic.

use std::collections::HashMap;

use crate::node::NodeId;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Compute a stable 64-bit hash of a system prompt for prefix co-location.
///
/// Uses FNV-1a over the prompt's UTF-8 bytes. Deterministic across runs and
/// machines, so two router instances agree on the same key for the same
/// prompt. This is a co-location key, not a security primitive.
pub fn hash_system_prompt(system_prompt: &str) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in system_prompt.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Prefix-hash → node co-location table.
#[derive(Debug, Clone, Default)]
pub struct PrefixMap {
    map: HashMap<u64, NodeId>,
}

impl PrefixMap {
    /// Create an empty prefix map.
    pub fn new() -> Self {
        PrefixMap::default()
    }

    /// Record that a prefix hash is co-located on `node`.
    pub fn assign(&mut self, prefix_hash: u64, node: NodeId) {
        self.map.insert(prefix_hash, node);
    }

    /// Look up the node currently co-locating a prefix hash.
    pub fn get(&self, prefix_hash: u64) -> Option<&NodeId> {
        self.map.get(&prefix_hash)
    }

    /// Remove a prefix mapping (e.g. when its node goes away).
    pub fn remove(&mut self, prefix_hash: u64) -> Option<NodeId> {
        self.map.remove(&prefix_hash)
    }

    /// Drop every prefix currently mapped to `node` (e.g. node removed).
    pub fn forget_node(&mut self, node: &NodeId) {
        self.map.retain(|_, n| n != node);
    }

    /// Number of tracked prefixes (exposed for the §34.12 metric).
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether no prefixes are tracked.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_stable_and_deterministic() {
        let a = hash_system_prompt("You are a helpful assistant.");
        let b = hash_system_prompt("You are a helpful assistant.");
        assert_eq!(a, b);
        // Known FNV-1a vector: hash of the empty string is the offset basis.
        assert_eq!(hash_system_prompt(""), FNV_OFFSET_BASIS);
    }

    #[test]
    fn different_prompts_hash_differently() {
        assert_ne!(
            hash_system_prompt("prompt A"),
            hash_system_prompt("prompt B")
        );
    }

    #[test]
    fn assign_and_get_roundtrip() {
        let mut m = PrefixMap::new();
        let h = hash_system_prompt("shared system prompt");
        assert!(m.get(h).is_none());
        m.assign(h, NodeId::new("gpu-1"));
        assert_eq!(m.get(h), Some(&NodeId::new("gpu-1")));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn forget_node_drops_its_prefixes() {
        let mut m = PrefixMap::new();
        m.assign(1, NodeId::new("gpu-0"));
        m.assign(2, NodeId::new("gpu-1"));
        m.assign(3, NodeId::new("gpu-0"));
        m.forget_node(&NodeId::new("gpu-0"));
        assert_eq!(m.len(), 1);
        assert_eq!(m.get(2), Some(&NodeId::new("gpu-1")));
    }
}
