//! Built-in validation rules for this validation family.

use std::collections::HashMap;

use onnx_runtime_loader::proto::onnx::{
    AttributeProto, GraphProto, NodeProto, SparseTensorProto, TensorProto, tensor_proto,
};

use super::super::{Severity, ValidationContext, ValidationRule, Violation, ViolationLocation};
use crate::model::Model;

/// Validate dense tensor dimensions, storage-field selection, payload size,
/// segment bounds, and external-data constraints.
pub struct TensorPayloadValidityRule;

impl ValidationRule for TensorPayloadValidityRule {
    fn id(&self) -> &str {
        "proto.tensor_payload_valid"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        let Some(proto) = model.retained_proto() else {
            return Vec::new();
        };
        let mut violations = Vec::new();
        if let Some(graph) = &proto.graph {
            visit_graph_tensors(graph, self.id(), &mut violations);
        }
        for function in &proto.functions {
            for node in &function.node {
                visit_node_tensors(node, &function.name, self.id(), &mut violations);
            }
            for attribute in &function.attribute_proto {
                visit_attribute_tensors(
                    attribute,
                    ViolationLocation::Model,
                    self.id(),
                    &mut violations,
                );
            }
        }
        violations
    }
}

/// Validate sparse tensor COO structure, index type/shape, bounds, ordering,
/// and uniqueness.
pub struct SparseTensorValidityRule;

impl ValidationRule for SparseTensorValidityRule {
    fn id(&self) -> &str {
        "proto.sparse_tensor_valid"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        let Some(proto) = model.retained_proto() else {
            return Vec::new();
        };
        let mut violations = Vec::new();
        if let Some(graph) = &proto.graph {
            visit_graph_sparse_tensors(graph, self.id(), &mut violations);
        }
        for function in &proto.functions {
            for node in &function.node {
                visit_node_sparse_tensors(node, &function.name, self.id(), &mut violations);
            }
            for attribute in &function.attribute_proto {
                visit_attribute_sparse_tensors(
                    attribute,
                    ViolationLocation::Model,
                    self.id(),
                    &mut violations,
                );
            }
        }
        violations
    }
}

fn visit_graph_tensors(graph: &GraphProto, rule_id: &str, violations: &mut Vec<Violation>) {
    for tensor in &graph.initializer {
        check_tensor_payload(tensor, rule_id, violations);
    }
    for sparse in &graph.sparse_initializer {
        if let Some(values) = &sparse.values {
            check_tensor_payload(values, rule_id, violations);
        }
        if let Some(indices) = &sparse.indices {
            check_tensor_payload(indices, rule_id, violations);
        }
    }
    for node in &graph.node {
        visit_node_tensors(node, &graph.name, rule_id, violations);
    }
}

fn visit_node_tensors(
    node: &NodeProto,
    graph_name: &str,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let location = ViolationLocation::Node {
        graph_name: graph_name.to_string(),
        node_name: if node.name.is_empty() {
            format!("<{}>", node.op_type)
        } else {
            node.name.clone()
        },
    };
    for attribute in &node.attribute {
        visit_attribute_tensors(attribute, location.clone(), rule_id, violations);
        if let Some(graph) = &attribute.g {
            visit_graph_tensors(graph, rule_id, violations);
        }
        for graph in &attribute.graphs {
            visit_graph_tensors(graph, rule_id, violations);
        }
    }
}

fn visit_attribute_tensors(
    attribute: &AttributeProto,
    _location: ViolationLocation,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    if let Some(tensor) = &attribute.t {
        check_tensor_payload(tensor, rule_id, violations);
    }
    for tensor in &attribute.tensors {
        check_tensor_payload(tensor, rule_id, violations);
    }
    if let Some(sparse) = &attribute.sparse_tensor {
        if let Some(values) = &sparse.values {
            check_tensor_payload(values, rule_id, violations);
        }
        if let Some(indices) = &sparse.indices {
            check_tensor_payload(indices, rule_id, violations);
        }
    }
    for sparse in &attribute.sparse_tensors {
        if let Some(values) = &sparse.values {
            check_tensor_payload(values, rule_id, violations);
        }
        if let Some(indices) = &sparse.indices {
            check_tensor_payload(indices, rule_id, violations);
        }
    }
}

pub(super) fn check_tensor_payload(
    tensor: &TensorProto,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let location = ViolationLocation::Value {
        value_name: tensor.name.clone(),
    };
    let mut report = |message: String| {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message,
            location: location.clone(),
        });
    };
    let Some(dtype) = tensor_proto::DataType::try_from(tensor.data_type)
        .ok()
        .filter(|dtype| *dtype != tensor_proto::DataType::Undefined)
    else {
        report(format!(
            "TensorProto.data_type {} must be a defined ONNX dtype",
            tensor.data_type
        ));
        return;
    };
    let Some(full_count) = checked_numel(&tensor.dims) else {
        report(
            "TensorProto dimensions must be non-negative and their product must fit usize".into(),
        );
        return;
    };
    let count = if let Some(segment) = &tensor.segment {
        if segment.begin < 0 || segment.end < segment.begin {
            report("TensorProto.segment must satisfy 0 <= begin <= end".into());
            return;
        }
        let Ok(begin) = usize::try_from(segment.begin) else {
            report("TensorProto.segment.begin does not fit usize".into());
            return;
        };
        let Ok(end) = usize::try_from(segment.end) else {
            report("TensorProto.segment.end does not fit usize".into());
            return;
        };
        if end > full_count {
            report(format!(
                "TensorProto.segment end {end} exceeds element count {full_count}"
            ));
        }
        end.saturating_sub(begin)
    } else {
        full_count
    };

    let populated = [
        !tensor.float_data.is_empty(),
        !tensor.int32_data.is_empty(),
        !tensor.string_data.is_empty(),
        !tensor.int64_data.is_empty(),
        !tensor.raw_data.is_empty(),
        !tensor.double_data.is_empty(),
        !tensor.uint64_data.is_empty(),
    ]
    .into_iter()
    .filter(|present| *present)
    .count();
    let data_location = tensor_proto::DataLocation::try_from(tensor.data_location).ok();
    if data_location.is_none() {
        report(format!(
            "TensorProto.data_location {} is not valid",
            tensor.data_location
        ));
        return;
    }
    if data_location == Some(tensor_proto::DataLocation::External) {
        if populated != 0 {
            report("external TensorProto must not contain embedded payload fields".into());
        }
        let mut external = HashMap::new();
        for entry in &tensor.external_data {
            if external
                .insert(entry.key.as_str(), entry.value.as_str())
                .is_some()
            {
                report(format!(
                    "TensorProto.external_data contains duplicate key '{}'",
                    entry.key
                ));
            }
        }
        if external
            .get("location")
            .is_none_or(|value| value.is_empty())
        {
            report("external TensorProto requires a non-empty location entry".into());
        }
        let offset = parse_external_usize(external.get("offset").copied(), "offset", &mut report);
        let length = parse_external_usize(external.get("length").copied(), "length", &mut report);
        if let (Some(offset), Some(length)) = (offset, length)
            && offset.checked_add(length).is_none()
        {
            report("external TensorProto offset + length overflows usize".into());
        }
        return;
    }
    if !tensor.external_data.is_empty() {
        report("inline TensorProto must not contain external_data entries".into());
    }
    if populated > 1 {
        report("TensorProto must use exactly one embedded payload field".into());
        return;
    }
    if count > 0 && populated == 0 {
        report("non-empty TensorProto is missing its payload".into());
        return;
    }
    if populated == 0 {
        return;
    }
    let actual_expected = match dtype {
        _ if !tensor.raw_data.is_empty() => storage_bytes(dtype, count)
            .map(|expected| (tensor.raw_data.len(), expected, "raw_data")),
        tensor_proto::DataType::Float => Some((tensor.float_data.len(), count, "float_data")),
        tensor_proto::DataType::Complex64 => count
            .checked_mul(2)
            .map(|expected| (tensor.float_data.len(), expected, "float_data")),
        tensor_proto::DataType::Double => Some((tensor.double_data.len(), count, "double_data")),
        tensor_proto::DataType::Complex128 => count
            .checked_mul(2)
            .map(|expected| (tensor.double_data.len(), expected, "double_data")),
        tensor_proto::DataType::Int64 => Some((tensor.int64_data.len(), count, "int64_data")),
        tensor_proto::DataType::String => Some((tensor.string_data.len(), count, "string_data")),
        tensor_proto::DataType::Uint32 | tensor_proto::DataType::Uint64 => {
            Some((tensor.uint64_data.len(), count, "uint64_data"))
        }
        tensor_proto::DataType::Uint4
        | tensor_proto::DataType::Int4
        | tensor_proto::DataType::Float4e2m1 => count
            .checked_add(1)
            .map(|value| (tensor.int32_data.len(), value / 2, "int32_data")),
        tensor_proto::DataType::Uint2 | tensor_proto::DataType::Int2 => count
            .checked_add(3)
            .map(|value| (tensor.int32_data.len(), value / 4, "int32_data")),
        _ => Some((tensor.int32_data.len(), count, "int32_data")),
    };
    match actual_expected {
        Some((actual, expected, field)) if actual != expected => report(format!(
            "TensorProto.{field} contains {actual} entries/bytes but {expected} are required"
        )),
        None => report("TensorProto payload size arithmetic overflowed usize".into()),
        _ => {}
    }
}

fn parse_external_usize(
    value: Option<&str>,
    key: &str,
    report: &mut impl FnMut(String),
) -> Option<usize> {
    let value = value?;
    match value.parse::<usize>() {
        Ok(value) => Some(value),
        Err(_) => {
            report(format!(
                "TensorProto.external_data '{key}' must be an unsigned integer"
            ));
            None
        }
    }
}

fn checked_numel(dims: &[i64]) -> Option<usize> {
    dims.iter().try_fold(1usize, |count, &dim| {
        let dim = usize::try_from(dim).ok()?;
        count.checked_mul(dim)
    })
}

fn storage_bytes(dtype: tensor_proto::DataType, count: usize) -> Option<usize> {
    let bits = match dtype {
        tensor_proto::DataType::Uint2 | tensor_proto::DataType::Int2 => 2,
        tensor_proto::DataType::Uint4
        | tensor_proto::DataType::Int4
        | tensor_proto::DataType::Float4e2m1 => 4,
        tensor_proto::DataType::Uint8
        | tensor_proto::DataType::Int8
        | tensor_proto::DataType::Bool
        | tensor_proto::DataType::Float8e4m3fn
        | tensor_proto::DataType::Float8e4m3fnuz
        | tensor_proto::DataType::Float8e5m2
        | tensor_proto::DataType::Float8e5m2fnuz
        | tensor_proto::DataType::Float8e8m0 => 8,
        tensor_proto::DataType::Uint16
        | tensor_proto::DataType::Int16
        | tensor_proto::DataType::Float16
        | tensor_proto::DataType::Bfloat16 => 16,
        tensor_proto::DataType::Float
        | tensor_proto::DataType::Int32
        | tensor_proto::DataType::Uint32 => 32,
        tensor_proto::DataType::Double
        | tensor_proto::DataType::Int64
        | tensor_proto::DataType::Uint64
        | tensor_proto::DataType::Complex64 => 64,
        tensor_proto::DataType::Complex128 => 128,
        tensor_proto::DataType::String | tensor_proto::DataType::Undefined => return None,
    };
    count
        .checked_mul(bits)
        .and_then(|bits| bits.checked_add(7))
        .map(|bits| bits / 8)
}

fn visit_graph_sparse_tensors(graph: &GraphProto, rule_id: &str, violations: &mut Vec<Violation>) {
    for sparse in &graph.sparse_initializer {
        check_sparse_tensor(sparse, true, rule_id, violations);
    }
    for node in &graph.node {
        visit_node_sparse_tensors(node, &graph.name, rule_id, violations);
    }
}

fn visit_node_sparse_tensors(
    node: &NodeProto,
    graph_name: &str,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let location = ViolationLocation::Node {
        graph_name: graph_name.to_string(),
        node_name: if node.name.is_empty() {
            format!("<{}>", node.op_type)
        } else {
            node.name.clone()
        },
    };
    for attribute in &node.attribute {
        visit_attribute_sparse_tensors(attribute, location.clone(), rule_id, violations);
        if let Some(graph) = &attribute.g {
            visit_graph_sparse_tensors(graph, rule_id, violations);
        }
        for graph in &attribute.graphs {
            visit_graph_sparse_tensors(graph, rule_id, violations);
        }
    }
}

fn visit_attribute_sparse_tensors(
    attribute: &AttributeProto,
    _location: ViolationLocation,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    if let Some(sparse) = &attribute.sparse_tensor {
        check_sparse_tensor(sparse, false, rule_id, violations);
    }
    for sparse in &attribute.sparse_tensors {
        check_sparse_tensor(sparse, false, rule_id, violations);
    }
}

pub(super) fn check_sparse_tensor(
    sparse: &SparseTensorProto,
    require_name: bool,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let name = sparse
        .values
        .as_ref()
        .map(|values| values.name.clone())
        .unwrap_or_default();
    let location = ViolationLocation::Value { value_name: name };
    let mut report = |message: String| {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message,
            location: location.clone(),
        });
    };
    let Some(values) = &sparse.values else {
        report("SparseTensorProto.values must be present".into());
        return;
    };
    if require_name && values.name.is_empty() {
        report("sparse initializer values must have a non-empty name".into());
    }
    if sparse.dims.is_empty() || sparse.dims.iter().any(|&dim| dim <= 0) {
        report("SparseTensorProto must have positive rank and dimensions".into());
        return;
    }
    let Some(dense_count) = checked_numel(&sparse.dims) else {
        report("SparseTensorProto dimensions must fit usize".into());
        return;
    };
    if values.dims.len() != 1 {
        report("SparseTensorProto.values must have shape [NNZ]".into());
        return;
    }
    let Ok(nnz) = usize::try_from(values.dims[0]) else {
        report("SparseTensorProto NNZ must be non-negative".into());
        return;
    };
    if nnz > dense_count {
        report(format!(
            "SparseTensorProto NNZ {nnz} exceeds dense element count {dense_count}"
        ));
    }
    let Some(indices) = &sparse.indices else {
        if nnz != 0 {
            report("SparseTensorProto.indices must be present when NNZ is nonzero".into());
        }
        return;
    };
    if indices.data_type != tensor_proto::DataType::Int64 as i32 {
        report("SparseTensorProto.indices must have INT64 dtype".into());
        return;
    }
    let rank = sparse.dims.len();
    let coordinate = match indices.dims.as_slice() {
        [count] if usize::try_from(*count).ok() == Some(nnz) => false,
        [count, width]
            if usize::try_from(*count).ok() == Some(nnz)
                && usize::try_from(*width).ok() == Some(rank) =>
        {
            true
        }
        _ => {
            report(format!(
                "SparseTensorProto.indices shape must be [{nnz}] or [{nnz}, {rank}]"
            ));
            return;
        }
    };
    let Some(index_values) = tensor_i64_values(indices) else {
        report("SparseTensorProto.indices must use int64_data or raw_data".into());
        return;
    };
    let expected_indices = if coordinate {
        nnz.checked_mul(rank)
    } else {
        Some(nnz)
    };
    if expected_indices != Some(index_values.len()) {
        report("SparseTensorProto.indices payload count does not match its shape".into());
        return;
    }
    if coordinate {
        let mut previous: Option<&[i64]> = None;
        for tuple in index_values.chunks(rank.max(1)) {
            if tuple
                .iter()
                .zip(&sparse.dims)
                .any(|(&index, &dim)| index < 0 || index >= dim)
            {
                report("SparseTensorProto coordinate index is out of bounds".into());
            }
            if previous.is_some_and(|prior| prior >= tuple) {
                report(
                    "SparseTensorProto coordinate indices must be lexicographically increasing"
                        .into(),
                );
            }
            previous = Some(tuple);
        }
    } else {
        let Ok(dense_count) = i64::try_from(dense_count) else {
            report("SparseTensorProto dense element count does not fit i64".into());
            return;
        };
        let mut previous = None;
        for &index in &index_values {
            if index < 0 || index >= dense_count {
                report("SparseTensorProto linear index is out of bounds".into());
            }
            if previous.is_some_and(|prior| prior >= index) {
                report("SparseTensorProto linear indices must be strictly increasing".into());
            }
            previous = Some(index);
        }
    }
}

fn tensor_i64_values(tensor: &TensorProto) -> Option<Vec<i64>> {
    if !tensor.int64_data.is_empty() {
        return Some(tensor.int64_data.clone());
    }
    if !tensor.raw_data.len().is_multiple_of(8) {
        return None;
    }
    Some(
        tensor
            .raw_data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bytes| i64::from_le_bytes(*bytes))
            .collect(),
    )
}
