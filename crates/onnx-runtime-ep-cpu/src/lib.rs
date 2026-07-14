//! # `onnx-runtime-ep-cpu`
//!
//! The CPU execution provider for the ORT 2.0 runtime (see `docs/ORT2.md` §4.4
//! and §54 Phase 1). It implements [`onnx_runtime_ep_api::ExecutionProvider`]
//! and hosts oneDNN-backed kernels (MatMul, Add, Relu, Reshape, Transpose,
//! Gather, LayerNorm) via C++ FFI.
//!
//! **Phase 1 skeleton:** [`CpuExecutionProvider`] implements the EP trait with
//! stubbed bodies, and [`kernels`] enumerates the op modules to fill in. The
//! oneDNN FFI (`native-eps/cpu`) is wired via a `build.rs` in a later task.

pub mod kernels;
pub mod provider;

pub use provider::CpuExecutionProvider;
