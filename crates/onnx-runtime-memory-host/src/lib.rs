//! # `onnx-runtime-memory-host`
//!
//! Loads an [nxmem][abi] memory plugin and adapts its C vtables to the
//! internal Rust interfaces in `onnx-runtime-memory-api`.
//!
//! [abi]: onnx_runtime_memory_abi
//!
//! Nothing internal leaks outward. A plugin sees only `#[repr(C)]` plain data,
//! raw pointers, and `extern "C"` function pointers; it never sees a Rust
//! trait object, `Arc`, enum layout, or allocator ownership. In the other
//! direction the loaded mechanism appears to the rest of the runtime as an
//! ordinary [`DeviceAllocator`](onnx_runtime_memory_api::DeviceAllocator) with
//! optional [`VirtualBacking`](onnx_runtime_memory_api::VirtualBacking) and
//! [`SharedMapping`](onnx_runtime_memory_api::SharedMapping) capabilities.
//!
//! # Lock discipline
//!
//! **No ABI call is made while any lock is held.** The one host-side map
//! (address to `allocation_id`) is locked, mutated, and unlocked before the
//! plugin is entered. Governance locks in the governor and provider layers are
//! likewise released before a plugin call, so a plugin is always free to
//! block, call back into the host, or spawn threads.
//!
//! # Pinning and unload
//!
//! Factories, allocators, capability views, shared prefixes, and queued
//! releases all hold an `Arc<PluginModule>`, and [`PluginModule`] declares its
//! `libloading::Library` last. Unload is refused while either the host's own
//! counters or the plugin's [`NxmemUnloadReport`][report] report anything
//! live; the refusal hands the plugin back so the caller can retire the
//! outstanding work and retry.
//!
//! [report]: onnx_runtime_memory_abi::NxmemUnloadReport

pub mod allocator;
pub mod error;
pub mod loader;

pub use allocator::{
    AllocatorCore, HostReclaim, PluginAllocator, PluginSharedMapping, PluginSharedPrefix,
    PluginVirtualBacking, RetiredRelease,
};
pub use error::{PluginError, status_to_memory_error};
pub use loader::{
    HostLiveCounts, MAX_FACTORIES, MemoryPlugin, NegotiatedAbi, PluginFactory, PluginModule,
    UnloadRejection,
};
