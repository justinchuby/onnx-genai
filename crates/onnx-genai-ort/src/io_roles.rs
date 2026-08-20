//! Declarative model input/output role resolution.
//!
//! Metadata is authoritative. Unambiguous tensor-shape signals are the
//! only permitted fallback; graph port names are never interpreted.

use crate::TensorInfo;

/// How a graph port was assigned its semantic role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoRoleSource {
    Metadata,
    Structure,
}

/// One resolved graph port and the signal that selected it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPort {
    pub name: String,
    pub source: IoRoleSource,
}

/// Resolve one semantic role from metadata or tensor structure.
///
/// Structural resolution succeeds only when exactly one tensor matches. An
/// ambiguous structural signal is rejected with an actionable metadata error.
pub fn resolve_port(
    tensors: &[TensorInfo],
    declared: Option<&str>,
    metadata_key: &str,
    structural: impl Fn(&TensorInfo) -> bool,
) -> Result<Option<ResolvedPort>, String> {
    if let Some(name) = declared {
        if tensors.iter().any(|tensor| tensor.name == name) {
            return Ok(Some(ResolvedPort {
                name: name.to_owned(),
                source: IoRoleSource::Metadata,
            }));
        }
        return Err(format!(
            "{metadata_key} declares port '{name}', but the graph exposes {:?}",
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
    if structural_matches.len() > 1 {
        return Err(format!(
            "cannot resolve {metadata_key} from tensor shape because {} ports match: {:?}; \
             declare the port's role in pipeline.workflow.components.<component>.ports.roles",
            structural_matches.len(),
            structural_matches
                .iter()
                .map(|tensor| (&tensor.name, tensor.dtype, &tensor.shape))
                .collect::<Vec<_>>()
        ));
    }
    Ok(None)
}

pub fn is_rank_one_or_two_sequence(tensor: &TensorInfo) -> bool {
    matches!(tensor.shape.len(), 1 | 2)
}

pub fn is_rank_three_sequence(tensor: &TensorInfo) -> bool {
    tensor.shape.len() == 3
}

pub fn is_rank_one_to_three_output(tensor: &TensorInfo) -> bool {
    matches!(tensor.shape.len(), 1..=3)
        && tensor.shape.last().is_some_and(|dimension| *dimension != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataType;

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
            is_rank_one_or_two_sequence,
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
            is_rank_one_or_two_sequence,
        )
        .unwrap()
        .unwrap();
        assert_eq!(resolved.name, "opaque_tokens");
        assert_eq!(resolved.source, IoRoleSource::Structure);
    }

    #[test]
    fn ambiguous_structure_requires_metadata() {
        let tensors = vec![
            tensor("input_ids", DataType::Int64, &[-1, -1]),
            tensor("positions", DataType::Int64, &[-1, -1]),
        ];
        let error = resolve_port(
            &tensors,
            None,
            "io.token_input",
            is_rank_one_or_two_sequence,
        )
        .unwrap_err();
        assert!(error.contains("declare the port's role in"));
    }

    #[test]
    fn missing_declared_port_is_an_error() {
        let error = resolve_port(
            &[tensor("tokens", DataType::Int64, &[-1, -1])],
            Some("missing"),
            "io.token_input",
            is_rank_one_or_two_sequence,
        )
        .unwrap_err();
        assert!(error.contains("missing"));
    }
}
