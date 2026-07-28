use super::*;
use onnx_genai_ort::io_roles::{
    is_embedding_sequence, is_integer_sequence, is_score_output, legacy_terminal_name, resolve_port,
};

pub(crate) fn role_tensor_info(
    tensors: &[onnx_runtime_session::IoMeta],
) -> Vec<onnx_genai_ort::TensorInfo> {
    tensors
        .iter()
        .map(|tensor| onnx_genai_ort::TensorInfo {
            name: tensor.name.clone(),
            dtype: match tensor.dtype {
                onnx_runtime_ir::DataType::Float32 => onnx_genai_ort::DataType::Float32,
                onnx_runtime_ir::DataType::Float16 => onnx_genai_ort::DataType::Float16,
                onnx_runtime_ir::DataType::BFloat16 => onnx_genai_ort::DataType::BFloat16,
                onnx_runtime_ir::DataType::Int32 => onnx_genai_ort::DataType::Int32,
                onnx_runtime_ir::DataType::Int64 => onnx_genai_ort::DataType::Int64,
                _ => onnx_genai_ort::DataType::Bool,
            },
            shape: tensor
                .shape
                .iter()
                .map(|dimension| match dimension {
                    onnx_runtime_ir::Dim::Static(value) => *value as i64,
                    onnx_runtime_ir::Dim::Symbolic(_) => -1,
                })
                .collect(),
        })
        .collect()
}

#[derive(Clone, Copy)]
pub(crate) enum StructuralRole {
    IntegerSequence,
    EmbeddingSequence,
    ScoreOutput,
    None,
}

fn structurally_matches(role: StructuralRole, info: &onnx_genai_ort::TensorInfo) -> bool {
    match role {
        StructuralRole::IntegerSequence => is_integer_sequence(info),
        StructuralRole::EmbeddingSequence => is_embedding_sequence(info),
        StructuralRole::ScoreOutput => is_score_output(info),
        StructuralRole::None => false,
    }
}

pub(crate) fn declared_or_detected_input(
    inputs: &[onnx_genai_ort::TensorInfo],
    declared: Option<&str>,
    structural_role: StructuralRole,
    candidates: &[&str],
    field: &str,
) -> anyhow::Result<String> {
    resolve_port(
        inputs,
        declared,
        &format!("native graph metadata io.{field}"),
        |info| structurally_matches(structural_role, info),
        |name| legacy_terminal_name(name, candidates),
    )
    .map_err(anyhow::Error::msg)?
    .map(|resolved| resolved.name)
    .with_context(|| {
        format!(
            "native graph cannot resolve io.{field}; declare the exact port in metadata or export an unambiguous typed tensor (legacy names: {candidates:?})"
        )
    })
}

pub(crate) fn optional_declared_or_detected_input(
    inputs: &[onnx_genai_ort::TensorInfo],
    declared: Option<&str>,
    structural_role: StructuralRole,
    candidates: &[&str],
    field: &str,
) -> anyhow::Result<Option<String>> {
    resolve_port(
        inputs,
        declared,
        &format!("native graph metadata io.{field}"),
        |info| structurally_matches(structural_role, info),
        |name| legacy_terminal_name(name, candidates),
    )
    .map_err(anyhow::Error::msg)
    .map(|resolved| resolved.map(|resolved| resolved.name))
}

pub(crate) fn declared_or_detected_output(
    outputs: &[onnx_genai_ort::TensorInfo],
    declared: Option<&str>,
    structural_role: StructuralRole,
    candidates: &[&str],
    field: &str,
) -> anyhow::Result<String> {
    resolve_port(
        outputs,
        declared,
        &format!("native graph metadata io.{field}"),
        |info| structurally_matches(structural_role, info),
        |name| legacy_terminal_name(name, candidates),
    )
    .map_err(anyhow::Error::msg)?
    .map(|resolved| resolved.name)
    .with_context(|| {
        format!(
            "native graph cannot resolve io.{field}; declare the exact port in metadata or export an unambiguous typed tensor (legacy names: {candidates:?})"
        )
    })
}

pub(crate) fn optional_declared_or_detected_output(
    outputs: &[onnx_genai_ort::TensorInfo],
    declared: Option<&str>,
    structural_role: StructuralRole,
    candidates: &[&str],
    field: &str,
) -> anyhow::Result<Option<String>> {
    resolve_port(
        outputs,
        declared,
        &format!("native graph metadata io.{field}"),
        |info| structurally_matches(structural_role, info),
        |name| legacy_terminal_name(name, candidates),
    )
    .map_err(anyhow::Error::msg)
    .map(|resolved| resolved.map(|resolved| resolved.name))
}

pub(crate) fn is_past_name(name: &str) -> bool {
    has_past_prefix(name, KvNamingConvention::Dotted)
}

pub(crate) fn is_present_name(name: &str) -> bool {
    has_present_prefix(name, KvNamingConvention::Dotted)
}

pub(crate) fn matching_past_name(output: &str, inputs: &[String]) -> Option<String> {
    matching_past_input(output, inputs, KvNamingConvention::Dotted).cloned()
}
