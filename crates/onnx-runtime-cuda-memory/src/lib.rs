//! CUDA device memory: allocators, virtual memory, and who is charged.
//!
//! # Why this is not in the execution provider
//!
//! An execution provider's job is operators — which kernel runs for which node,
//! how it is dispatched, what cuBLAS and cuDNN are asked to do. *Where the
//! memory came from* is a different question, and one that a component with no
//! interest in operators may legitimately need to answer: a caller wiring up
//! their own device arena, a tool measuring what a model costs, an embedder
//! replacing our allocator with theirs.
//!
//! While these lived in `onnx-runtime-ep-cuda`, taking the allocator meant
//! taking every kernel, cuBLASLt, cuDNN and NVRTC with it. That is a
//! dependency on operator machinery to answer a memory question, and it is
//! backwards.
//!
//! So the split is by concern rather than by convenience:
//!
//! * this crate — device memory: reserving it, committing it, handing it out,
//!   and reporting it to a [`MemoryGovernor`];
//! * `onnx-runtime-ep-cuda` — operators, which *depend on* this crate the way
//!   any consumer of device memory does.
//!
//! [`MemoryGovernor`]: onnx_runtime_memory_governor::MemoryGovernor
//!
//! # What is here
//!
//! [`vmm_allocator::CudaVmmAllocator`] is the **sole built-in CUDA memory
//! mechanism**. It separates virtual from physical memory, reserving one
//! address range and mapping granules under it on demand. That is the mechanism
//! vAttention ([arXiv 2405.04437](https://arxiv.org/abs/2405.04437)) is built
//! on, and it is also what makes charging *every* physical byte to the ledger
//! affordable: granules are 2 MiB, so the allocations that dominate by count
//! are served from memory that is already mapped and never reach the governor.
//!
//! [`virtual_memory::CudaVirtualBacking`] is the CUDA implementation of the
//! platform-independent [`VirtualBacking`] contract, so the same growable
//! buffer works over host or device memory.
//!
//! [`VirtualBacking`]: onnx_runtime_virtual_memory::VirtualBacking
//!
//! # What is deliberately *not* here
//!
//! There is no built-in eager `cuMemAlloc` allocator. It was removed once the
//! VMM arena covered every path that used it, because a second built-in
//! mechanism is a second thing every accounting, capture and teardown invariant
//! has to hold for, and a fallback that is never measured is a fallback nobody
//! can vouch for.
//!
//! That is a statement about the *built-in* mechanism only. The
//! [`DeviceAllocator`] contract is untouched: a caller who wants `cuMemAlloc`,
//! a BFC arena, or anything else writes it and injects it through
//! `CudaExecutionProvider::with_memory`, exactly as the CPU EP and ONNX
//! Runtime's governed allocator already allow.
//!
//! [`DeviceAllocator`]: onnx_runtime_memory_governor::DeviceAllocator
//!
//! # Building without CUDA
//!
//! Everything here is behind the `cuda` feature, matching the execution
//! provider. `cargo build` needs no CUDA toolkit either way: cudarc is used
//! with `dynamic-loading`, so the driver is opened at runtime.

//! # Building without CUDA, continued
//!
//! The modules are not feature-gated. cudarc is used with `dynamic-loading`,
//! so they compile anywhere and only fail at runtime on a machine with no
//! driver -- which is what lets `cargo check --workspace` cover this code on
//! a machine with no GPU. The `cuda` feature exists to match the execution
//! provider's, not to hide the source.

pub mod capture_gate;
pub mod release;
pub mod virtual_memory;
pub mod vmm_allocator;

/// Shared helpers for this crate's real-CUDA integration tests. Compiled only
/// under the `gpu-tests` feature so it never reaches a production build.
/// Integration tests that use it must cfg-gate both the import and the body
/// because `cfg(test)` does not propagate to integration test crates and
/// Cargo silently ignores self dev-dependencies for feature resolution.
#[cfg(feature = "gpu-tests")]
pub mod test_support;
