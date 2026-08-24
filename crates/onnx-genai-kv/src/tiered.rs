//! Tiered storage: hot GPU-resident pages with cold CPU offload.
//!
//! The current backend stores both tiers in host RAM, but the page table treats
//! `Device::Gpu(0)` as the hot tier and `Device::Cpu` as the cold tier. These
//! are declared/emulated residency locations: both stores are host-addressable.
//! Moving a page allocates a target store, copies every storage component, and
//! atomically replaces the source.
//!
//! That is the end state for this crate, not a waypoint. The module used to say
//! "until Stage 3 supplies a CUDA store"; #721 stage 3 is superseded, because on
//! native CUDA device KV paging is owned by the VMM layer (`CudaVmmAllocator`,
//! #740/#745/#748) and a second page allocator here would duplicate that
//! ownership rather than complete it. The factory-and-copy contract stays
//! because it is what lets an out-of-tree backend implement a device store
//! without changing cache-facing callers -- an optional view, not a prerequisite
//! anybody in this repository is waiting on.
//!
//! Quantized K/V storage supports symmetric int8 and scaled FP8 E4M3FN/E5M2.
//! Each layer, K/V component, and head has an independent scale. On write, f32
//! values are quantized into compact page storage; reads reconstruct f32 values.
