//! Tiered storage: hot GPU-resident pages with cold CPU offload.
//!
//! The current backend stores both tiers in host RAM, but the page table treats
//! `Device::Gpu(0)` as the hot tier and `Device::Cpu` as the cold tier. Moving
//! a page between tiers updates the page's `device` while preserving its owned
//! K/V payload byte-for-byte. The abstraction is intentionally synchronous for
//! now; a future GPU backend can replace the move hooks with device copies or
//! async prefetch without changing the cache-facing API.
//!
//! Quantized K/V storage supports symmetric int8 and scaled FP8 E4M3FN/E5M2.
//! Each layer, K/V component, and head has an independent scale. On write, f32
//! values are quantized into compact page storage; reads reconstruct f32 values.
