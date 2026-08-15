//! Tiered storage: hot GPU-resident pages with cold CPU offload.
//!
//! The current backend stores both tiers in host RAM, but the page table treats
//! `Device::Gpu(0)` as the hot tier and `Device::Cpu` as the cold tier. These
//! are declared/emulated residency locations: both stores remain
//! host-addressable until Stage 3 supplies a CUDA store. Moving a page now
//! allocates a target store, copies every storage component, and atomically
//! replaces the source. A future GPU backend can implement the same factory and
//! copy contract without changing cache-facing callers.
//!
//! Quantized K/V storage supports symmetric int8 and scaled FP8 E4M3FN/E5M2.
//! Each layer, K/V component, and head has an independent scale. On write, f32
//! values are quantized into compact page storage; reads reconstruct f32 values.
