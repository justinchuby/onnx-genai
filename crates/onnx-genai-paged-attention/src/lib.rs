//! Typed schema validator and CPU reference oracle for
//! `com.microsoft::PagedAttention` version 1, as shipped in ONNX Runtime
//! **1.29.0** (the version this workspace pins — see
//! `crates/onnx-genai-ort/ort-sys/build.rs`).
//!
//! This crate is **test/reference infrastructure**, not a runtime kernel. It
//! exists to:
//!
//! 1. Reproduce, in Rust and from first principles, the exact acceptance rules
//!    of the authoritative helper
//!    (`onnxruntime/contrib_ops/cpu/bert/paged_attention_helper.h`) so that a
//!    native execution provider can validate a `PagedAttention` node and
//!    **reject every unsupported optional mode with a typed reason** rather than
//!    silently miscompute (mirroring the upstream WebGPU EP, which implements
//!    only the `SEPARATE` subset and returns `NOT_IMPLEMENTED` for the rest).
//! 2. Provide a numerically exact CPU **oracle** for the two dense modes the op
//!    supports — `SEPARATE` (dense GQA/MHA) and `LATENT` (absorbed Multi-head
//!    Latent Attention, `v_head_size < head_size`, partial RoPE) — for use as
//!    the correctness gate of a future CUDA `LATENT` kernel.
//!
//! Upstream ORT has **no CPU kernel** for this op; nothing here should be
//! presented as upstream CPU support.
//!
//! ## The two validation layers
//!
//! [`validate::check_inputs`] is the *schema* layer: it accepts exactly what the
//! ORT v1.29.0 helper accepts and returns [`PagedAttentionError::InvalidArgument`]
//! for the same violations. [`backend::check_backend_support`] is the *backend*
//! layer: given a [`backend::NativeSubset`] describing which optional features a
//! particular kernel implements, it returns [`PagedAttentionError::NotImplemented`]
//! for any schema-valid mode outside that subset. Keeping the two separate is the
//! `design-discipline` rule — one question, one mechanism.

pub mod backend;
pub mod oracle;
pub mod params;
pub mod types;
pub mod validate;

pub use backend::{NativeSubset, check_backend_support};
pub use oracle::{PagedAttentionData, paged_attention_reference};
pub use params::{PagedAttentionAttributes, PagedAttentionInputs, PagedAttentionParameters, Shape};
pub use types::{KvCacheDtype, KvCacheLayout, KvQuantType, PagedAttentionError};
