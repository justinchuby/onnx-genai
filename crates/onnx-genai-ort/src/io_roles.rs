//! Declarative model input/output role resolution.
//!
//! Metadata is authoritative. Structural tensor signals are used only when a
//! role is not declared, and historical name matching is the final
//! compatibility fallback.

use crate::{DataType, TensorInfo};

/// How a graph port was assigned its semantic role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoRoleSource {
    Metadata,
    Structure,
    LegacyName,
}

/// One resolved graph port and the signal that selected it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPort {
    pub name: String,
    pub source: IoRoleSource,
}

/// Resolve one semantic role from metadata, tensor structure, then legacy names.
///
/// Structural resolution succeeds only when exactly one tensor matches. An
/// ambiguous structural signal is not guessed; the legacy matcher may still
/// preserve compatibility for an established export convention.
pub fn resolve_port(
    tensors: &[TensorInfo],
    declared: Option<&str>,
    role: &str,
    structural: impl Fn(&TensorInfo) -> bool,
    legacy_name: impl Fn(&str) -> bool,
) -> Result<Option<ResolvedPort>, String> {
    if let Some(name) = declared {
        if tensors.iter().any(|tensor| tensor.name == name) {
            return Ok(Some(ResolvedPort {
                name: name.to_owned(),
                source: IoRoleSource::Metadata,
            }));
        }
        return Err(format!(
            "{role} declares port '{name}', but the graph exposes {:?}",
            tensors
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>()
        ));
    }

    let structural_matches = tensors
        .iter()
        .filter(|tensor| structural(tensor))
        .collect::<Vec<_>>();
    if structural_matches.len() == 1 {
        return Ok(Some(ResolvedPort {
            name: structural_matches[0].name.clone(),
            source: IoRoleSource::Structure,
        }));
    }

    Ok(tensors
        .iter()
        .find(|tensor| legacy_name(&tensor.name))
        .map(|tensor| ResolvedPort {
            name: tensor.name.clone(),
            source: IoRoleSource::LegacyName,
        }))
}

pub fn is_integer_sequence(tensor: &TensorInfo) -> bool {
    matches!(tensor.dtype, DataType::Int32 | DataType::Int64) && matches!(tensor.shape.len(), 1 | 2)
}

pub fn is_embedding_sequence(tensor: &TensorInfo) -> bool {
    matches!(
        tensor.dtype,
        DataType::Float16 | DataType::BFloat16 | DataType::Float32
    ) && tensor.shape.len() == 3
}

pub fn is_score_output(tensor: &TensorInfo) -> bool {
    matches!(
        tensor.dtype,
        DataType::Float16 | DataType::BFloat16 | DataType::Float32
    ) && matches!(tensor.shape.len(), 1..=3)
        && tensor.shape.last().is_some_and(|dimension| *dimension != 0)
}

pub fn legacy_terminal_name(name: &str, candidates: &[&str]) -> bool {
    let lower = name.to_ascii_lowercase();
    candidates
        .iter()
        .any(|candidate| lower == *candidate || lower.ends_with(&format!(".{candidate}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str, dtype: DataType, shape: &[i64]) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
        }
    }

    #[test]
    fn metadata_wins_over_structure_and_names() {
        let tensors = vec![
            tensor("input_ids", DataType::Int64, &[-1, -1]),
            tensor("opaque_tokens", DataType::Int64, &[-1, -1]),
        ];
        let resolved = resolve_port(
            &tensors,
            Some("opaque_tokens"),
            "io.token_input",
            is_integer_sequence,
            |name| legacy_terminal_name(name, &["input_ids"]),
        )
        .unwrap()
        .unwrap();
        assert_eq!(resolved.name, "opaque_tokens");
        assert_eq!(resolved.source, IoRoleSource::Metadata);
    }

    #[test]
    fn unique_structure_is_name_agnostic() {
        let tensors = vec![
            tensor("opaque_tokens", DataType::Int64, &[-1, -1]),
            tensor("features", DataType::Float32, &[-1, -1, 64]),
        ];
        let resolved = resolve_port(
            &tensors,
            None,
            "io.token_input",
            is_integer_sequence,
            |name| legacy_terminal_name(name, &["input_ids"]),
        )
        .unwrap()
        .unwrap();
        assert_eq!(resolved.name, "opaque_tokens");
        assert_eq!(resolved.source, IoRoleSource::Structure);
    }

    #[test]
    fn ambiguous_structure_uses_legacy_fallback() {
        let tensors = vec![
            tensor("input_ids", DataType::Int64, &[-1, -1]),
            tensor("positions", DataType::Int64, &[-1, -1]),
        ];
        let resolved = resolve_port(
            &tensors,
            None,
            "io.token_input",
            is_integer_sequence,
            |name| legacy_terminal_name(name, &["input_ids"]),
        )
        .unwrap()
        .unwrap();
        assert_eq!(resolved.name, "input_ids");
        assert_eq!(resolved.source, IoRoleSource::LegacyName);
    }

    #[test]
    fn ambiguous_structure_without_legacy_signal_is_unresolved() {
        let tensors = vec![
            tensor("first", DataType::Int64, &[-1, -1]),
            tensor("second", DataType::Int64, &[-1, -1]),
        ];
        let resolved = resolve_port(
            &tensors,
            None,
            "io.token_input",
            is_integer_sequence,
            |_| false,
        )
        .unwrap();
        assert_eq!(resolved, None);
    }

    #[test]
    fn missing_declared_port_is_an_error() {
        let error = resolve_port(
            &[tensor("tokens", DataType::Int64, &[-1, -1])],
            Some("missing"),
            "io.token_input",
            is_integer_sequence,
            |_| false,
        )
        .unwrap_err();
        assert!(error.contains("missing"));
    }
}
