use std::collections::{BTreeMap, BTreeSet};

use crate::{GenAiConfigError, incomplete};

/// One graph tensor declaration supplied by the package loader.
///
/// This inventory comes from the ONNX graph interface itself, so compatibility
/// conversion can preserve actual sparse ports, ranks, and dtypes instead of
/// expanding architecture-sized guesses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphTensorInfo {
    pub name: String,
    pub dtype: String,
    /// One entry per axis; `None` denotes a symbolic dimension.
    pub dimensions: Vec<Option<usize>>,
}

/// Explicit input/output inventory for one ONNX component.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelGraphInfo {
    pub inputs: Vec<GraphTensorInfo>,
    pub outputs: Vec<GraphTensorInfo>,
}

pub(crate) fn require_same_dtype(
    left: &GraphTensorInfo,
    right: &GraphTensorInfo,
    description: &str,
) -> Result<(), GenAiConfigError> {
    if left.dtype == right.dtype {
        Ok(())
    } else {
        Err(incomplete(format!(
            "{description} dtype agreement: '{}' is {}, but '{}' is {}",
            left.name, left.dtype, right.name, right.dtype
        )))
    }
}

pub(crate) fn split_indexed_pattern(pattern: &str) -> Result<(&str, &str), GenAiConfigError> {
    let Some((prefix, suffix)) = pattern.split_once("%d") else {
        return Err(incomplete(format!(
            "indexed tensor pattern '{pattern}' must contain %d"
        )));
    };
    if suffix.contains("%d") {
        return Err(incomplete(format!(
            "indexed tensor pattern '{pattern}' must contain exactly one %d"
        )));
    }
    Ok((prefix, suffix))
}

pub(crate) fn match_indexed_tensors<'a>(
    tensors: &'a [GraphTensorInfo],
    pattern: &str,
) -> Result<BTreeMap<usize, &'a GraphTensorInfo>, GenAiConfigError> {
    let (prefix, suffix) = split_indexed_pattern(pattern)?;
    let mut matched = BTreeMap::new();
    for tensor in tensors {
        let Some(index) = tensor
            .name
            .strip_prefix(prefix)
            .and_then(|name| name.strip_suffix(suffix))
        else {
            continue;
        };
        let index = index.parse::<usize>().map_err(|_| {
            incomplete(format!(
                "ONNX tensor '{}' matches pattern '{pattern}' but has a non-numeric index",
                tensor.name
            ))
        })?;
        if matched.insert(index, tensor).is_some() {
            return Err(incomplete(format!(
                "ONNX graph has duplicate tensors for pattern '{pattern}' index {index}"
            )));
        }
    }
    Ok(matched)
}

pub(crate) fn exact_index_set(
    maps: &[&BTreeMap<usize, &GraphTensorInfo>],
    description: &str,
) -> Result<Vec<usize>, GenAiConfigError> {
    let Some(first) = maps.first() else {
        return Ok(Vec::new());
    };
    let expected = first.keys().copied().collect::<Vec<_>>();
    if maps
        .iter()
        .skip(1)
        .all(|map| map.keys().copied().eq(expected.iter().copied()))
    {
        Ok(expected)
    } else {
        Err(incomplete(format!(
            "{description} do not have identical layer indices"
        )))
    }
}

pub(crate) fn common_pattern_prefix<'a>(
    first: &'a str,
    second: &'a str,
) -> Result<&'a str, GenAiConfigError> {
    let (first_prefix, _) = split_indexed_pattern(first)?;
    let (second_prefix, _) = split_indexed_pattern(second)?;
    if first_prefix == second_prefix {
        Ok(first_prefix)
    } else {
        Err(incomplete(format!(
            "key/value patterns '{first}' and '{second}' must share the same state prefix"
        )))
    }
}

pub(crate) fn suffix_tensor_map<'a>(
    tensors: &'a [GraphTensorInfo],
    prefix: &str,
    excluded: &BTreeSet<&str>,
    description: &str,
) -> Result<BTreeMap<String, &'a GraphTensorInfo>, GenAiConfigError> {
    let mut matched = BTreeMap::new();
    for tensor in tensors {
        if excluded.contains(tensor.name.as_str()) {
            continue;
        }
        let Some(suffix) = tensor.name.strip_prefix(prefix) else {
            continue;
        };
        if suffix.is_empty() {
            return Err(incomplete(format!(
                "{description} contains an empty suffix for '{}'",
                tensor.name
            )));
        }
        if matched.insert(suffix.to_owned(), tensor).is_some() {
            return Err(incomplete(format!(
                "{description} contains duplicate suffix '{suffix}'"
            )));
        }
    }
    Ok(matched)
}
