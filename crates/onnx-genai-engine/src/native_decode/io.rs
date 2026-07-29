use super::*;
use onnx_genai_ort::io_roles::{
    is_rank_one_or_two_sequence, is_rank_one_to_three_output, is_rank_three_sequence, resolve_port,
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
        StructuralRole::IntegerSequence => is_rank_one_or_two_sequence(info),
        StructuralRole::EmbeddingSequence => is_rank_three_sequence(info),
        StructuralRole::ScoreOutput => is_rank_one_to_three_output(info),
        StructuralRole::None => false,
    }
}

pub(crate) fn declared_or_detected_input(
    inputs: &[onnx_genai_ort::TensorInfo],
    declared: Option<&str>,
    structural_role: StructuralRole,
    metadata_scope: &str,
    field: &str,
) -> anyhow::Result<String> {
    resolve_port(
        inputs,
        declared,
        &format!("{metadata_scope}.{field}"),
        |info| structurally_matches(structural_role, info),
    )
    .map_err(anyhow::Error::msg)?
    .map(|resolved| resolved.name)
    .with_context(|| {
        format!(
            "native graph cannot resolve {metadata_scope}.{field} from tensor shape; declare the exact graph port in {metadata_scope}.{field}"
        )
    })
}

pub(crate) fn optional_declared_or_detected_input(
    inputs: &[onnx_genai_ort::TensorInfo],
    declared: Option<&str>,
    structural_role: StructuralRole,
    metadata_scope: &str,
    field: &str,
) -> anyhow::Result<Option<String>> {
    resolve_port(
        inputs,
        declared,
        &format!("{metadata_scope}.{field}"),
        |info| structurally_matches(structural_role, info),
    )
    .map_err(anyhow::Error::msg)
    .map(|resolved| resolved.map(|resolved| resolved.name))
}

pub(crate) fn declared_or_detected_output(
    outputs: &[onnx_genai_ort::TensorInfo],
    declared: Option<&str>,
    structural_role: StructuralRole,
    metadata_scope: &str,
    field: &str,
) -> anyhow::Result<String> {
    resolve_port(
        outputs,
        declared,
        &format!("{metadata_scope}.{field}"),
        |info| structurally_matches(structural_role, info),
    )
    .map_err(anyhow::Error::msg)?
    .map(|resolved| resolved.name)
    .with_context(|| {
        format!(
            "native graph cannot resolve {metadata_scope}.{field} from tensor shape; declare the exact graph port in {metadata_scope}.{field}"
        )
    })
}

pub(crate) fn optional_declared_or_detected_output(
    outputs: &[onnx_genai_ort::TensorInfo],
    declared: Option<&str>,
    structural_role: StructuralRole,
    metadata_scope: &str,
    field: &str,
) -> anyhow::Result<Option<String>> {
    resolve_port(
        outputs,
        declared,
        &format!("{metadata_scope}.{field}"),
        |info| structurally_matches(structural_role, info),
    )
    .map_err(anyhow::Error::msg)
    .map(|resolved| resolved.map(|resolved| resolved.name))
}
