//! # `onnx-runtime-ep-cpu`
//!
//! The CPU execution provider for the ORT 2.0 runtime (see `docs/ORT2.md` §4.4
//! and §54 Phase 1). It implements [`onnx_runtime_ep_api::ExecutionProvider`]
//! and hosts pure-Rust reference kernels for the Phase-1 op set (`MatMul`,
//! `Add`, `Relu`, `Reshape`, `Transpose`, `Gather`, `LayerNormalization`).
//!
//! ## Correctness first, perf later
//!
//! The Phase-1 exit milestone is a **correctness** goal ("BERT on CPU matches
//! upstream ORT"), so these kernels are straightforward, naive pure-Rust
//! implementations — no C++/FFI, no oneDNN, no `cc` build dependency (oneDNN is
//! not installed on the build host). Each kernel lives behind the
//! [`onnx_runtime_ep_api::Kernel`] trait, leaving a clean seam for a Phase-1.5
//! perf pass to drop in a blocked/SIMD GEMM (oneDNN via FFI, or a Rust BLAS such
//! as `matrixmultiply`/`gemm`) without disturbing the EP contract or the
//! session. See [`kernels::matmul`] for the hot spot.
//!
//! ## `unsafe`
//!
//! The crate is `unsafe`-minimal. The only `unsafe` is the raw device-buffer
//! access the ep-api contract forces (aligned host `alloc`/`dealloc`, `memcpy`,
//! and strided element reads/writes), each isolated and `SAFETY`-documented. All
//! kernel arithmetic is safe Rust operating on dense `Vec<f32>` buffers produced
//! by the two audited accessors in [`kernels`].

pub mod kernels;
pub mod provider;
pub mod strided;

pub use provider::CpuExecutionProvider;
