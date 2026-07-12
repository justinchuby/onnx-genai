//! Tiered storage: hot GPU-resident pages with cold CPU offload.
//!
//! The current backend stores both tiers in host RAM, but the page table treats
//! `Device::Gpu(0)` as the hot tier and `Device::Cpu` as the cold tier. Moving
//! a page between tiers updates the page's `device` while preserving its owned
//! K/V payload byte-for-byte. The abstraction is intentionally synchronous for
//! now; a future GPU backend can replace the move hooks with device copies or
//! async prefetch without changing the cache-facing API.
//!
//! Quantized K/V storage uses `KvDType::Int8`: a symmetric signed int8 payload
//! with one scale per page. On write, the page is quantized from f32; on read,
//! values are reconstructed as `q as f32 * scale`. The expected absolute error
//! for one quantization pass is bounded by roughly `scale / 2` for that page.
