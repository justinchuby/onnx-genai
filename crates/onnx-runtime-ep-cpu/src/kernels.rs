//! CPU kernel modules (Phase 1 targets: MatMul, Add, Relu, Reshape, Transpose,
//! Gather, LayerNorm — ported from ORT's C++ CPU provider over oneDNN, §4.4).
//!
//! Each op becomes a `Kernel` implementation plus a `KernelFactory` registered
//! into the CPU op registry. This module is a placeholder enumerating the
//! Phase 1 op set; the FFI declarations and kernel structs land per-op.

/// The set of ops the CPU EP targets for the Phase 1 BERT-on-CPU milestone.
pub const PHASE1_OPS: &[&str] = &[
    "MatMul",
    "Add",
    "Relu",
    "Reshape",
    "Transpose",
    "Gather",
    "LayerNormalization",
];
