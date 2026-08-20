//! CubeCL-backed execution provider for WebGPU (WGSL) and Vulkan (SPIR-V).
//!
//! # Why two providers from one backend
//!
//! CubeCL compiles the same `#[cube]` kernel source to several shader
//! languages. `cubecl-webgpu` selects the WGSL compiler, which runs on every
//! `wgpu` backend (Metal on macOS, Vulkan on Linux, DX12 on Windows, and
//! browser WebGPU). `cubecl-vulkan` selects the SPIR-V compiler pinned to the
//! Vulkan backend, which additionally exposes `bf16`, 8/16-bit integers, and
//! subgroup operations that WGSL does not have.
//!
//! They are deliberately *separate* providers rather than one provider with a
//! hidden heuristic (RULES.md §5): a caller that asks for `cubecl-vulkan` and
//! lands on a host without Vulkan gets a clear failure naming the reason, not a
//! silent WGSL downgrade with different numerics and dtype coverage.
//!
//! # Feature gating
//!
//! With no backend feature the crate still compiles, and every constructor
//! fails closed with a message naming the feature to rebuild with. This mirrors
//! `onnx-runtime-ep-cuda`: the workspace must stay buildable on a host that has
//! no GPU stack at all.
//!
//! # Device memory model
//!
//! CubeCL hands out opaque [`Handle`]s, not addresses, while this workspace's
//! [`DeviceBuffer`] is an address plus a length that callers may offset into
//! (`TensorView::with_byte_offset`). [`memory`] bridges the two with a virtual
//! address table: each allocation is assigned a unique, never-reused range in a
//! synthetic address space, and offset arithmetic on that range resolves back
//! to `(handle, offset)`. The addresses are never dereferenced on the host,
//! which is exactly the contract `DeviceBuffer` documents for non-host-visible
//! devices.
//!
//! [`Handle`]: https://docs.rs/cubecl-runtime
//! [`DeviceBuffer`]: onnx_runtime_ep_api::provider::DeviceBuffer

pub mod backend;

pub use backend::{CubeclBackend, UnavailableReason};

#[cfg(feature = "webgpu")]
pub mod context;
#[cfg(feature = "webgpu")]
pub mod kernels;
#[cfg(feature = "webgpu")]
pub mod memory;
#[cfg(feature = "webgpu")]
pub mod provider;
#[cfg(feature = "webgpu")]
pub mod runtime;

#[cfg(feature = "webgpu")]
pub use kernels::CubeclOpDescriptor;
#[cfg(feature = "webgpu")]
pub use provider::{CubeclExecutionProvider, build_cubecl_registry_descriptors};
