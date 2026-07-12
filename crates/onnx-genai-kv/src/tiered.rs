//! Tiered storage: GPU → CPU → Disk eviction and prefetch.

// TODO: Implement async page migration between device tiers.
// For Phase 1, all pages live on GPU.
// Phase 3 adds CPU offload and async prefetch.
