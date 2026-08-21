//! Canonical semantic identity of a metadata document.
//!
//! This is an *identity*, not integrity and not trust. It answers exactly one
//! question: "was this disposable plan, compiled graph, or state checkpoint
//! produced against the same metadata semantics?" It is not a signature, it
//! does not authenticate a producer, and it must never be used to decide
//! whether an artifact is safe to load.
//!
//! The identity is computed over a canonical normalization of the parsed
//! document so that formatting, key order, YAML-versus-JSON encoding, numeric
//! spelling, null and empty optional fields, and skippable profiles do not
//! change it.
//!
//! The normalization is deliberately *syntactic*: it works on the parsed
//! document, not on the typed schema, because the schema is intentionally
//! deserialize-only and has no serializer to round-trip through. One
//! consequence is worth stating plainly: writing an optional field explicitly
//! at its schema default is a different encoding than omitting it, and the two
//! may carry different identities.
//!
//! That asymmetry is safe in the only direction that matters. This identity is
//! consumed to decide whether a *disposable* artifact — a compiled plan, a
//! memory plan, a state checkpoint — may still be reused. A spurious change
//! costs one recompile. A spurious match serves a stale plan against changed
//! semantics. Normalization therefore never merges two documents it cannot
//! prove equivalent from syntax alone.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

/// Prefix identifying the normalization and hash algorithm.
pub const IDENTITY_SCHEME: &str = "onnx-genai-metadata-identity-v1:sha256";

/// Compute the canonical semantic identity of a raw metadata document.
///
/// The input is the document as parsed from YAML or JSON, before it is
/// deserialized into typed structures. Normalization:
///
/// 1. object keys are sorted explicitly, so the identity does not depend on
///    document order or on `serde_json` map ordering,
/// 2. `null` members are dropped, because an explicit null and an absent
///    optional field mean the same thing to a reader,
/// 3. profiles marked `requirement: ignorable` are dropped, because a strict
///    reader that skips them must still compute the same identity, and a
///    container that normalizes to empty is dropped with them,
/// 4. the result is serialized compactly and hashed with SHA-256.
pub fn semantic_identity(document: &serde_json::Value) -> String {
    let encoded = canonicalize(document, &Path::Root).unwrap_or_else(|| "{}".to_string());
    let digest = Sha256::digest(encoded.as_bytes());
    format!("{IDENTITY_SCHEME}:{digest:x}")
}

/// Compute the canonical semantic identity of a serialized metadata document.
///
/// Accepts the YAML or JSON text of the document. YAML is a superset of JSON,
/// so one parse covers both encodings and the identity is independent of which
/// one the package shipped.
pub fn semantic_identity_of_str(content: &str) -> Result<String, crate::MetadataError> {
    let document = serde_yaml::from_str::<serde_json::Value>(content)
        .map_err(|error| crate::MetadataError::Parse(error.to_string()))?;
    Ok(semantic_identity(&document))
}

/// Where the current value sits in the document, so normalization can drop
/// ignorable profiles without a general-purpose query language.
enum Path<'a> {
    Root,
    Key(&'a Path<'a>, &'a str),
    Index,
}

impl Path<'_> {
    /// Whether this path is a member of the top-level `profiles` map.
    fn is_profile(&self) -> bool {
        matches!(self, Path::Key(Path::Root, "profiles"))
    }
}

/// The canonical encoding of `value`, or `None` when it carries no meaning.
///
/// A member carries no meaning when it is null, or when it is a container that
/// normalizes to empty: every collection in this schema defaults to empty, so an
/// absent one and an empty one say the same thing. That equivalence is what lets
/// a reader that skips all of a document's ignorable profiles compute the same
/// identity as a reader that never saw them.
///
/// Object keys are sorted here rather than relying on `serde_json::Map`
/// ordering, which depends on whether the `preserve_order` feature is enabled
/// anywhere in the dependency graph. The identity must not depend on that.
fn canonicalize(value: &serde_json::Value, path: &Path<'_>) -> Option<String> {
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::Object(map) => {
            let members = map
                .iter()
                .filter(|(_, member)| !(path.is_profile() && is_ignorable_profile(member)))
                .filter_map(|(key, member)| {
                    let encoded = canonicalize(member, &Path::Key(path, key))?;
                    Some((key.as_str(), encoded))
                })
                .collect::<BTreeMap<_, _>>();
            if members.is_empty() {
                return None;
            }
            let body = members
                .into_iter()
                .map(|(key, encoded)| {
                    let key = encode_scalar(&serde_json::Value::String(key.to_string()));
                    format!("{key}:{encoded}")
                })
                .collect::<Vec<_>>()
                .join(",");
            Some(format!("{{{body}}}"))
        }
        serde_json::Value::Array(items) => {
            let encoded = items
                .iter()
                .filter_map(|item| canonicalize(item, &Path::Index))
                .collect::<Vec<_>>();
            if encoded.is_empty() {
                return None;
            }
            Some(format!("[{}]", encoded.join(",")))
        }
        scalar => Some(encode_scalar(scalar)),
    }
}

fn encode_scalar(value: &serde_json::Value) -> String {
    // Numbers are normalized so that the identity depends on the value and not
    // on how the producer spelled it. YAML and JSON both accept `128` and
    // `128.0` for the same integer field, and `serde_json::Number` remembers
    // which one it parsed, so encoding the number verbatim would make a
    // cosmetic rewrite invalidate every plan keyed off this identity.
    if let Some(number) = value.as_f64() {
        return encode_number(number);
    }
    serde_json::to_string(value).unwrap_or_default()
}

/// Encode a numeric leaf in a form that depends only on its value.
///
/// Integral values are emitted as integers regardless of whether the source
/// wrote `128`, `128.0`, or `1.28e2`. Non-integral values use the shortest
/// representation that round-trips, which `f64`'s `Display` already provides.
fn encode_number(number: f64) -> String {
    if number.is_finite() && number.fract() == 0.0 && number.abs() < 1e18 {
        return format!("{}", number as i64);
    }
    format!("{number}")
}

fn is_ignorable_profile(profile: &serde_json::Value) -> bool {
    profile
        .get("requirement")
        .and_then(serde_json::Value::as_str)
        == Some("ignorable")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document(text: &str) -> serde_json::Value {
        serde_json::from_str(text).expect("valid json")
    }

    #[test]
    fn identity_is_stable_across_key_order_and_absent_optionals() {
        let a = document(r#"{"schema_version": 1, "model": {"name": "m"}}"#);
        let b = document(r#"{"model": {"name": "m", "note": null}, "schema_version": 1}"#);
        assert_eq!(semantic_identity(&a), semantic_identity(&b));
    }

    #[test]
    fn identity_ignores_skippable_profiles() {
        let without = document(r#"{"schema_version": 1, "profiles": {}}"#);
        let with = document(
            r#"{"schema_version": 1, "profiles": {"x": {"kind": "future",
               "version": 1, "requirement": "ignorable"}}}"#,
        );
        assert_eq!(semantic_identity(&without), semantic_identity(&with));
    }

    #[test]
    fn identity_tracks_required_profiles() {
        let without = document(r#"{"schema_version": 1, "profiles": {}}"#);
        let with = document(
            r#"{"schema_version": 1, "profiles": {"x": {"kind": "embedding",
               "version": 1, "requirement": "required"}}}"#,
        );
        assert_ne!(semantic_identity(&without), semantic_identity(&with));
    }

    #[test]
    fn identity_changes_with_semantics() {
        let a = document(r#"{"schema_version": 1, "model": {"name": "a"}}"#);
        let b = document(r#"{"schema_version": 1, "model": {"name": "b"}}"#);
        assert_ne!(semantic_identity(&a), semantic_identity(&b));
    }

    #[test]
    fn identity_is_independent_of_yaml_or_json_encoding() {
        let json = semantic_identity_of_str(r#"{"schema_version": 1, "model": {"name": "m"}}"#)
            .expect("json parses");
        let yaml = semantic_identity_of_str("schema_version: 1\nmodel:\n  name: m\n")
            .expect("yaml parses");
        assert_eq!(json, yaml);
    }

    #[test]
    fn identity_declares_its_scheme() {
        let value = document(r#"{"schema_version": 1}"#);
        assert!(semantic_identity(&value).starts_with(IDENTITY_SCHEME));
    }
}
