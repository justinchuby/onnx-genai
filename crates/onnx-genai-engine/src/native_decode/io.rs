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
        &format!("{metadata_scope} {field}"),
        |info| structurally_matches(structural_role, info),
    )
    .map_err(anyhow::Error::msg)?
    .map(|resolved| resolved.name)
    .with_context(|| {
        format!(
            "native graph cannot resolve {metadata_scope} {field} from tensor shape; \
             declare the port's role in pipeline.workflow.components.<component>.ports.roles"
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
        &format!("{metadata_scope} {field}"),
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
        &format!("{metadata_scope} {field}"),
        |info| structurally_matches(structural_role, info),
    )
    .map_err(anyhow::Error::msg)?
    .map(|resolved| resolved.name)
    .with_context(|| {
        format!(
            "native graph cannot resolve {metadata_scope} {field} from tensor shape; \
             declare the port's role in pipeline.workflow.components.<component>.ports.roles"
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
        &format!("{metadata_scope} {field}"),
        |info| structurally_matches(structural_role, info),
    )
    .map_err(anyhow::Error::msg)
    .map(|resolved| resolved.map(|resolved| resolved.name))
}

/// Coordinate rank of the declared `position_ids` input, derived from the
/// graph's physical shape — the authoritative source (it is exactly what raises
/// the native "position_ids: rank mismatch" error). Physical rank 2 → `1`
/// coordinate axis (the conventional `[1, S]` linear layout); physical rank 3 →
/// the declared **static** leading dim (the multi-axis mrope coordinate-stream
/// count, e.g. `3` for `[3, B, S]`). A non-static leading dim or any other
/// physical rank cannot be resolved from the graph alone and is a loud error. A
/// decoder with no position input returns `1` (unused).
///
/// This keeps the native position layout metadata-driven and general: a rank-2
/// decoder still builds `[1, S]`, a rank-3 mrope decoder builds `[rank, 1, S]`,
/// with no model-name gate and no hardcoded rank.
pub(crate) fn declared_position_rank(
    inputs: &[onnx_genai_ort::TensorInfo],
    position_ids: Option<&str>,
) -> anyhow::Result<usize> {
    let Some(name) = position_ids else {
        return Ok(1);
    };
    let Some(info) = inputs.iter().find(|info| info.name == name) else {
        return Ok(1);
    };
    match info.shape.len() {
        2 => Ok(1),
        3 => {
            let leading = info.shape[0];
            if leading >= 1 {
                Ok(leading as usize)
            } else {
                anyhow::bail!(
                    "native decoder position input '{}' declares a rank-3 shape {:?} with a non-static leading (coordinate) dim; the multi-axis position count must be a concrete dimension",
                    name,
                    info.shape
                )
            }
        }
        other => anyhow::bail!(
            "native decoder position input '{}' has unsupported physical rank {} (shape {:?}); expected rank 2 (linear) or rank 3 (multi-axis mrope)",
            name,
            other,
            info.shape
        ),
    }
}
