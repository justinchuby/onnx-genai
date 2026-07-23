//! Canonical handling of the default ONNX operator domain.
//!
//! The default ONNX operator set has **two equivalent spellings**: the empty
//! string `""` and the explicit `"ai.onnx"`. They denote the *same* domain.
//!
//! To avoid comparing both spellings in dozens of runtime hot paths, the loader
//! **canonicalizes** the default domain to `""` at the IR-load boundary. This
//! establishes a post-load invariant:
//!
//! > **In loaded IR, the default ONNX domain is always the empty string;
//! > `"ai.onnx"` never appears.**
//!
//! Given that invariant, code operating on already-loaded IR can simply test
//! [`str::is_empty`] (or [`Node::is_default_domain`]). Code that operates on
//! *raw proto* input — before normalization — must still accept both spellings
//! via the free [`is_default_domain`] helper.

/// The explicit spelling of the default ONNX operator domain.
///
/// Semantically identical to the empty string `""`.
pub const AI_ONNX_DOMAIN: &str = "ai.onnx";

/// Canonicalize an operator domain to its post-load form.
///
/// Maps the explicit default-domain spelling `"ai.onnx"` to the canonical empty
/// string `""`; every other domain (including non-default domains such as
/// `com.microsoft`) is returned unchanged.
///
/// The loader applies this at every point it materializes IR from the ONNX
/// proto so that, after load, the IR never contains `"ai.onnx"`.
#[inline]
#[must_use]
pub fn normalize_domain(domain: &str) -> &str {
    if domain == AI_ONNX_DOMAIN { "" } else { domain }
}

/// Whether a **raw** operator domain string denotes the default ONNX domain.
///
/// Accepts *both* spellings (`""` and `"ai.onnx"`). Use this on paths that see
/// raw proto or external input before normalization (loader validation, encoder
/// write-back, display). On already-loaded IR, prefer [`str::is_empty`] or
/// [`Node::is_default_domain`], which rely on the post-load invariant.
#[inline]
#[must_use]
pub fn is_default_domain(domain: &str) -> bool {
    domain.is_empty() || domain == AI_ONNX_DOMAIN
}
