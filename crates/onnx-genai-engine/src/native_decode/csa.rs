//! Canonical compressed-attention state lowering for native decode.

use super::*;
use onnx_genai_metadata::{
    CompressedRecordFormat, CompressionRatio, CompressionRecurrence, DecoderStateGroup,
    StateGroupProperties, StateKind, StatePortRole, StateUpdate,
};
use onnx_runtime_ir::Dim;
use onnx_runtime_session::{IoMeta, Tensor};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RecordStateSpec {
    pub group: String,
    pub layer: usize,
    pub role: StatePortRole,
    pub input: String,
    pub output: String,
    pub ratio: CompressionRatio,
    pub sequence_axis: usize,
    pub dtype: DataType,
    rank: usize,
    record_extents: Vec<(usize, usize)>,
    pub record_width_bytes: usize,
}

#[derive(Debug, Default)]
pub(super) struct CompressedStatePlan {
    records: Vec<RecordStateSpec>,
    present_index: HashMap<String, usize>,
    past_index: HashMap<String, usize>,
}

impl CompressedStatePlan {
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn records(&self) -> impl Iterator<Item = &RecordStateSpec> {
        self.records.iter()
    }

    pub fn record_for_present(&self, present: &str) -> Option<&RecordStateSpec> {
        self.present_index
            .get(present)
            .map(|&index| &self.records[index])
    }

    pub fn record_for_past(&self, past: &str) -> Option<&RecordStateSpec> {
        self.past_index.get(past).map(|&index| &self.records[index])
    }

    pub fn contains_past(&self, past: &str) -> bool {
        self.past_index.contains_key(past)
    }

    pub fn past_names(&self) -> impl Iterator<Item = &String> {
        self.past_index.keys()
    }

    pub fn verify_pairing(&self, present_to_past: &HashMap<String, String>) -> anyhow::Result<()> {
        for spec in &self.records {
            if present_to_past.get(&spec.output) != Some(&spec.input) {
                return Err(anyhow::Error::new(
                    CompressedStateLoadRefusal::PairingMismatch {
                        group: spec.group.clone(),
                        layer: spec.layer,
                        role: spec.role,
                        input: spec.input.clone(),
                        output: spec.output.clone(),
                    },
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressedStateLoadRefusal {
    MissingProperties(String),
    UnsupportedRecurrence(String),
    InvalidRecordFormat(String),
    InvalidUpdate(String),
    InvalidRole(String),
    MissingPort(String),
    PortCollision(String),
    DtypeMismatch {
        port: String,
        expected: DataType,
        actual: DataType,
    },
    InvalidSequenceAxis(String),
    InvalidRecordLayout(String),
    PairingMismatch {
        group: String,
        layer: usize,
        role: StatePortRole,
        input: String,
        output: String,
    },
    UnsupportedDevice,
}

impl std::fmt::Display for CompressedStateLoadRefusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingProperties(group) => write!(
                formatter,
                "compressed-attention state group '{group}' has no typed compression properties"
            ),
            Self::UnsupportedRecurrence(group) => write!(
                formatter,
                "compressed-attention state group '{group}' uses multi-token-prediction \
                 recurrence, which native decode cannot approximate"
            ),
            Self::InvalidRecordFormat(group) => write!(
                formatter,
                "compressed-attention state group '{group}' declares FP4 as its compressed-KV \
                 record format; FP4 is valid only for index-key records"
            ),
            Self::InvalidUpdate(group) => write!(
                formatter,
                "compressed-attention state group '{group}' must append records or replace carries"
            ),
            Self::InvalidRole(message) => formatter.write_str(message),
            Self::MissingPort(message) => formatter.write_str(message),
            Self::PortCollision(port) => write!(
                formatter,
                "compressed-attention graph port '{port}' is claimed by more than one state edge"
            ),
            Self::DtypeMismatch {
                port,
                expected,
                actual,
            } => write!(
                formatter,
                "compressed-attention state port '{port}' has dtype {actual:?}, expected {expected:?}"
            ),
            Self::InvalidSequenceAxis(group) => write!(
                formatter,
                "compressed-attention record group '{group}' must declare a valid sequence_axis"
            ),
            Self::InvalidRecordLayout(message) => formatter.write_str(message),
            Self::PairingMismatch {
                group,
                layer,
                role,
                input,
                output,
            } => write!(
                formatter,
                "compressed-attention group '{group}' layer {layer} role '{}' declares \
                 transition '{input}' => '{output}', but the lowered decoder ABI pairs those \
                 ports differently",
                role_name(*role)
            ),
            Self::UnsupportedDevice => formatter.write_str(
                "compressed-attention record state is unsupported by native CUDA: the current \
                 persistent-state path cannot represent per-layer rank-3 record axes, packed \
                 uint8 records, or compression-cadence cursor advancement; use native CPU or \
                 the CUDA state implementation owned by stacked PR #2194",
            ),
        }
    }
}

impl std::error::Error for CompressedStateLoadRefusal {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressedStateTensorPhase {
    Past,
    Present,
}

impl std::fmt::Display for CompressedStateTensorPhase {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Past => "past",
            Self::Present => "present",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressedStateTransitionRefusal {
    DtypeMismatch {
        group: String,
        layer: usize,
        role: StatePortRole,
        phase: CompressedStateTensorPhase,
        port: String,
        expected: DataType,
        actual: DataType,
    },
    RankMismatch {
        group: String,
        layer: usize,
        role: StatePortRole,
        phase: CompressedStateTensorPhase,
        port: String,
        expected: usize,
        actual: usize,
    },
    BatchMismatch {
        group: String,
        layer: usize,
        role: StatePortRole,
        phase: CompressedStateTensorPhase,
        port: String,
        expected: usize,
        actual: usize,
    },
    RecordLayoutMismatch {
        group: String,
        layer: usize,
        role: StatePortRole,
        phase: CompressedStateTensorPhase,
        port: String,
        expected_extents: Vec<(usize, usize)>,
        actual_extents: Vec<(usize, usize)>,
        expected_bytes: usize,
        actual_bytes: usize,
    },
    CursorMismatch {
        group: String,
        layer: usize,
        role: StatePortRole,
        phase: CompressedStateTensorPhase,
        port: String,
        logical_tokens: usize,
        ratio: CompressionRatio,
        expected_records: usize,
        actual_records: usize,
    },
    NonMonotonicCursor {
        group: String,
        layer: usize,
        role: StatePortRole,
        input: String,
        output: String,
        past_records: usize,
        present_records: usize,
    },
}

impl std::fmt::Display for CompressedStateTransitionRefusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DtypeMismatch {
                group,
                layer,
                role,
                phase,
                port,
                expected,
                actual,
            } => write!(
                formatter,
                "compressed-attention group '{group}' layer {layer} role '{}' {phase} port \
                 '{port}' has dtype {actual:?}, expected {expected:?}",
                role_name(*role)
            ),
            Self::RankMismatch {
                group,
                layer,
                role,
                phase,
                port,
                expected,
                actual,
            } => write!(
                formatter,
                "compressed-attention group '{group}' layer {layer} role '{}' {phase} port \
                 '{port}' has rank {actual}, expected {expected}",
                role_name(*role)
            ),
            Self::BatchMismatch {
                group,
                layer,
                role,
                phase,
                port,
                expected,
                actual,
            } => write!(
                formatter,
                "compressed-attention group '{group}' layer {layer} role '{}' {phase} port \
                 '{port}' has batch extent {actual}, expected request batch {expected}",
                role_name(*role)
            ),
            Self::RecordLayoutMismatch {
                group,
                layer,
                role,
                phase,
                port,
                expected_extents,
                actual_extents,
                expected_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "compressed-attention group '{group}' layer {layer} role '{}' {phase} port \
                 '{port}' has record extents {actual_extents:?} ({actual_bytes} bytes), expected \
                 {expected_extents:?} ({expected_bytes} bytes)",
                role_name(*role)
            ),
            Self::CursorMismatch {
                group,
                layer,
                role,
                phase,
                port,
                logical_tokens,
                ratio,
                expected_records,
                actual_records,
            } => write!(
                formatter,
                "compressed-attention group '{group}' layer {layer} role '{}' {phase} port \
                 '{port}' has {actual_records} records at logical token {logical_tokens}; \
                 ratio {ratio:?} requires exactly {expected_records}",
                role_name(*role)
            ),
            Self::NonMonotonicCursor {
                group,
                layer,
                role,
                input,
                output,
                past_records,
                present_records,
            } => write!(
                formatter,
                "compressed-attention group '{group}' layer {layer} role '{}' transition \
                 '{input}' => '{output}' regressed from {past_records} to {present_records} \
                 records",
                role_name(*role)
            ),
        }
    }
}

impl std::error::Error for CompressedStateTransitionRefusal {}

fn role_name(role: StatePortRole) -> &'static str {
    match role {
        StatePortRole::Key => "key",
        StatePortRole::Value => "value",
        StatePortRole::Combined => "combined",
        StatePortRole::CompressedKv => "compressed_kv",
        StatePortRole::CompressionCarry => "compression_carry",
        StatePortRole::IndexKey => "index_key",
        StatePortRole::IndexCarry => "index_carry",
    }
}

fn expected_dtype(role: StatePortRole, format: CompressedRecordFormat) -> DataType {
    match role {
        StatePortRole::CompressedKv => match format {
            CompressedRecordFormat::F32 => DataType::Float32,
            CompressedRecordFormat::Fp8E4m3Block64 | CompressedRecordFormat::Fp4E2m1Block32 => {
                DataType::Uint8
            }
        },
        StatePortRole::IndexKey => DataType::Uint8,
        StatePortRole::CompressionCarry | StatePortRole::IndexCarry => DataType::Float32,
        StatePortRole::Key | StatePortRole::Value | StatePortRole::Combined => DataType::Undefined,
    }
}

fn find_meta<'a>(values: &'a [IoMeta], name: &str) -> Option<&'a IoMeta> {
    values.iter().find(|meta| meta.name == name)
}

fn record_layout(
    group: &str,
    input: &IoMeta,
    output: &IoMeta,
    sequence_axis: usize,
) -> anyhow::Result<(Vec<(usize, usize)>, usize)> {
    if input.shape.len() != output.shape.len() || input.shape.len() < 3 {
        return Err(anyhow::Error::new(
            CompressedStateLoadRefusal::InvalidRecordLayout(format!(
                "compressed-attention record group '{group}' requires matching input/output \
                 rank of at least 3, got {:?} and {:?}",
                input.shape, output.shape
            )),
        ));
    }
    let mut record_extents = Vec::with_capacity(input.shape.len().saturating_sub(2));
    let mut record_elements = 1_usize;
    for (axis, (past, present)) in input
        .shape
        .iter()
        .copied()
        .zip(output.shape.iter().copied())
        .enumerate()
    {
        if axis == sequence_axis {
            continue;
        }
        let extent = match (past, present) {
            (Dim::Static(past), Dim::Static(present)) if past == present => Some(past),
            (Dim::Symbolic(_), Dim::Symbolic(_)) if axis == 0 => None,
            _ => {
                return Err(anyhow::Error::new(
                    CompressedStateLoadRefusal::InvalidRecordLayout(format!(
                        "compressed-attention record group '{group}' axis {axis} must preserve \
                         one descriptor extent across '{}' and '{}', got {past:?} and {present:?}",
                        input.name, output.name
                    )),
                ));
            }
        };
        if axis != 0 {
            let extent = extent.expect("non-batch record extents are static");
            record_extents.push((axis, extent));
            record_elements = record_elements.checked_mul(extent).ok_or_else(|| {
                anyhow::Error::new(CompressedStateLoadRefusal::InvalidRecordLayout(format!(
                    "compressed-attention record group '{group}' record width overflows usize"
                )))
            })?;
        }
    }
    let record_width_bytes = input
        .dtype
        .checked_storage_bytes(record_elements)
        .ok_or_else(|| {
            anyhow::Error::new(CompressedStateLoadRefusal::InvalidRecordLayout(format!(
                "compressed-attention record group '{group}' dtype {:?} has no fixed record \
                 storage width",
                input.dtype
            )))
        })?;
    Ok((record_extents, record_width_bytes))
}

pub(super) fn resolve_compressed_state(
    inputs: &[IoMeta],
    outputs: &[IoMeta],
    groups: &[DecoderStateGroup],
) -> anyhow::Result<CompressedStatePlan> {
    let mut plan = CompressedStatePlan::default();
    let mut occupied = HashSet::new();

    for group in groups
        .iter()
        .filter(|group| group.kind == StateKind::CompressedAttention)
    {
        let Some(StateGroupProperties::CompressedAttention {
            ratio,
            record_format,
            recurrence,
        }) = group.properties
        else {
            return Err(anyhow::Error::new(
                CompressedStateLoadRefusal::MissingProperties(group.name.clone()),
            ));
        };
        if recurrence == CompressionRecurrence::MultiTokenPrediction {
            return Err(anyhow::Error::new(
                CompressedStateLoadRefusal::UnsupportedRecurrence(group.name.clone()),
            ));
        }
        if record_format == CompressedRecordFormat::Fp4E2m1Block32 {
            return Err(anyhow::Error::new(
                CompressedStateLoadRefusal::InvalidRecordFormat(group.name.clone()),
            ));
        }
        let update = group.update.as_ref().unwrap_or(&StateUpdate::Append);
        if matches!(update, StateUpdate::IndexedScatter { .. }) {
            return Err(anyhow::Error::new(
                CompressedStateLoadRefusal::InvalidUpdate(group.name.clone()),
            ));
        }

        for port in &group.ports {
            let Some(role) = port.role else {
                return Err(anyhow::Error::new(CompressedStateLoadRefusal::InvalidRole(
                    format!(
                        "compressed-attention state group '{}' has an untyped edge '{}'=>'{}'",
                        group.name, port.input, port.output
                    ),
                )));
            };
            let is_record = matches!(role, StatePortRole::CompressedKv | StatePortRole::IndexKey);
            let is_carry = matches!(
                role,
                StatePortRole::CompressionCarry | StatePortRole::IndexCarry
            );
            if (!is_record && !is_carry)
                || (is_record && !matches!(update, StateUpdate::Append))
                || (is_carry && !matches!(update, StateUpdate::Replace))
                || (!ratio.has_index_state()
                    && matches!(role, StatePortRole::IndexKey | StatePortRole::IndexCarry))
            {
                return Err(anyhow::Error::new(CompressedStateLoadRefusal::InvalidRole(
                    format!(
                        "compressed-attention state group '{}' role '{}' is incompatible with \
                         ratio {ratio:?} and update {update:?}",
                        group.name,
                        role_name(role)
                    ),
                )));
            }
            for name in [&port.input, &port.output] {
                if !occupied.insert(name.clone()) {
                    return Err(anyhow::Error::new(
                        CompressedStateLoadRefusal::PortCollision(name.clone()),
                    ));
                }
            }

            let input = find_meta(inputs, &port.input).ok_or_else(|| {
                anyhow::Error::new(CompressedStateLoadRefusal::MissingPort(format!(
                    "compressed-attention state group '{}' declares input '{}' but the graph \
                     does not expose it",
                    group.name, port.input
                )))
            })?;
            let output = find_meta(outputs, &port.output).ok_or_else(|| {
                anyhow::Error::new(CompressedStateLoadRefusal::MissingPort(format!(
                    "compressed-attention state group '{}' declares output '{}' but the graph \
                     does not expose it",
                    group.name, port.output
                )))
            })?;
            let expected = expected_dtype(role, record_format);
            for meta in [input, output] {
                if meta.dtype != expected {
                    return Err(anyhow::Error::new(
                        CompressedStateLoadRefusal::DtypeMismatch {
                            port: meta.name.clone(),
                            expected,
                            actual: meta.dtype,
                        },
                    ));
                }
            }

            if is_record {
                let Some(axis) = group.sequence_axis.filter(|axis| {
                    *axis < input.shape.len() && *axis < output.shape.len() && *axis != 0
                }) else {
                    return Err(anyhow::Error::new(
                        CompressedStateLoadRefusal::InvalidSequenceAxis(group.name.clone()),
                    ));
                };
                let Some(layer) = port.layer else {
                    return Err(anyhow::Error::new(CompressedStateLoadRefusal::InvalidRole(
                        format!(
                            "compressed-attention state group '{}' role '{}' must declare its \
                             layer index",
                            group.name,
                            role_name(role)
                        ),
                    )));
                };
                let (record_extents, record_width_bytes) =
                    record_layout(&group.name, input, output, axis)?;
                let index = plan.records.len();
                plan.present_index.insert(port.output.clone(), index);
                plan.past_index.insert(port.input.clone(), index);
                plan.records.push(RecordStateSpec {
                    group: group.name.clone(),
                    layer,
                    role,
                    input: port.input.clone(),
                    output: port.output.clone(),
                    ratio,
                    sequence_axis: axis,
                    dtype: expected,
                    rank: input.shape.len(),
                    record_extents,
                    record_width_bytes,
                });
            }
        }
    }
    Ok(plan)
}

#[derive(Clone, Copy)]
pub(super) struct RecordTensorFacts<'a> {
    dtype: DataType,
    shape: &'a [usize],
}

impl<'a> From<&'a Tensor> for RecordTensorFacts<'a> {
    fn from(tensor: &'a Tensor) -> Self {
        Self {
            dtype: tensor.dtype,
            shape: &tensor.shape,
        }
    }
}

fn validate_tensor_contract(
    spec: &RecordStateSpec,
    tensor: RecordTensorFacts<'_>,
    phase: CompressedStateTensorPhase,
    port: &str,
    batch: usize,
) -> anyhow::Result<usize> {
    let refusal = |reason| Err(anyhow::Error::new(reason));
    if tensor.dtype != spec.dtype {
        return refusal(CompressedStateTransitionRefusal::DtypeMismatch {
            group: spec.group.clone(),
            layer: spec.layer,
            role: spec.role,
            phase,
            port: port.to_string(),
            expected: spec.dtype,
            actual: tensor.dtype,
        });
    }
    if tensor.shape.len() != spec.rank {
        return refusal(CompressedStateTransitionRefusal::RankMismatch {
            group: spec.group.clone(),
            layer: spec.layer,
            role: spec.role,
            phase,
            port: port.to_string(),
            expected: spec.rank,
            actual: tensor.shape.len(),
        });
    }
    if tensor.shape[0] != batch {
        return refusal(CompressedStateTransitionRefusal::BatchMismatch {
            group: spec.group.clone(),
            layer: spec.layer,
            role: spec.role,
            phase,
            port: port.to_string(),
            expected: batch,
            actual: tensor.shape[0],
        });
    }
    let actual_extents = tensor
        .shape
        .iter()
        .copied()
        .enumerate()
        .filter(|(axis, _)| *axis != 0 && *axis != spec.sequence_axis)
        .collect::<Vec<_>>();
    let record_elements = actual_extents
        .iter()
        .try_fold(1_usize, |elements, (_, extent)| {
            elements.checked_mul(*extent)
        })
        .unwrap_or(usize::MAX);
    let actual_bytes = tensor
        .dtype
        .checked_storage_bytes(record_elements)
        .unwrap_or(usize::MAX);
    if actual_extents != spec.record_extents || actual_bytes != spec.record_width_bytes {
        return refusal(CompressedStateTransitionRefusal::RecordLayoutMismatch {
            group: spec.group.clone(),
            layer: spec.layer,
            role: spec.role,
            phase,
            port: port.to_string(),
            expected_extents: spec.record_extents.clone(),
            actual_extents,
            expected_bytes: spec.record_width_bytes,
            actual_bytes,
        });
    }
    Ok(tensor.shape[spec.sequence_axis])
}

pub(super) fn validate_record_transition(
    spec: &RecordStateSpec,
    past: RecordTensorFacts<'_>,
    present: RecordTensorFacts<'_>,
    past_len: usize,
    total_len: usize,
    batch: usize,
) -> anyhow::Result<()> {
    let past_records = validate_tensor_contract(
        spec,
        past,
        CompressedStateTensorPhase::Past,
        &spec.input,
        batch,
    )?;
    let present_records = validate_tensor_contract(
        spec,
        present,
        CompressedStateTensorPhase::Present,
        &spec.output,
        batch,
    )?;
    let expected_past = past_len / spec.ratio.tokens_per_record();
    if past_records != expected_past {
        return Err(anyhow::Error::new(
            CompressedStateTransitionRefusal::CursorMismatch {
                group: spec.group.clone(),
                layer: spec.layer,
                role: spec.role,
                phase: CompressedStateTensorPhase::Past,
                port: spec.input.clone(),
                logical_tokens: past_len,
                ratio: spec.ratio,
                expected_records: expected_past,
                actual_records: past_records,
            },
        ));
    }
    if present_records < past_records {
        return Err(anyhow::Error::new(
            CompressedStateTransitionRefusal::NonMonotonicCursor {
                group: spec.group.clone(),
                layer: spec.layer,
                role: spec.role,
                input: spec.input.clone(),
                output: spec.output.clone(),
                past_records,
                present_records,
            },
        ));
    }
    let expected_present = total_len / spec.ratio.tokens_per_record();
    if present_records != expected_present {
        return Err(anyhow::Error::new(
            CompressedStateTransitionRefusal::CursorMismatch {
                group: spec.group.clone(),
                layer: spec.layer,
                role: spec.role,
                phase: CompressedStateTensorPhase::Present,
                port: spec.output.clone(),
                logical_tokens: total_len,
                ratio: spec.ratio,
                expected_records: expected_present,
                actual_records: present_records,
            },
        ));
    }
    Ok(())
}

pub(super) fn refuse_compressed_records_on_cuda(
    groups: &[DecoderStateGroup],
    device_is_cuda: bool,
) -> anyhow::Result<()> {
    if device_is_cuda
        && groups.iter().any(|group| {
            group.kind == StateKind::CompressedAttention
                && !matches!(group.update, Some(StateUpdate::Replace))
        })
    {
        return Err(anyhow::Error::new(
            CompressedStateLoadRefusal::UnsupportedDevice,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_metadata::{DecoderStatePort, StateGroupCapabilities, StateGroupProperties};
    use onnx_runtime_ir::static_shape;

    fn meta(name: &str, dtype: DataType, shape: &[usize]) -> IoMeta {
        IoMeta {
            name: name.to_string(),
            dtype,
            shape: static_shape(shape.iter().copied()),
        }
    }

    fn group(
        name: &str,
        ratio: CompressionRatio,
        format: CompressedRecordFormat,
        update: StateUpdate,
        ports: Vec<(StatePortRole, &str, &str)>,
    ) -> DecoderStateGroup {
        DecoderStateGroup {
            name: name.to_string(),
            kind: StateKind::CompressedAttention,
            properties: Some(StateGroupProperties::CompressedAttention {
                ratio,
                record_format: format,
                recurrence: CompressionRecurrence::Standard,
            }),
            sequence_axis: matches!(update, StateUpdate::Append).then_some(1),
            update: Some(update),
            capabilities: StateGroupCapabilities::default(),
            ports: ports
                .into_iter()
                .map(|(role, input, output)| DecoderStatePort {
                    role: Some(role),
                    layer: Some(0),
                    input: input.to_string(),
                    output: output.to_string(),
                })
                .collect(),
        }
    }

    #[test]
    fn ratio4_uint8_records_and_f32_carries_resolve() {
        let groups = vec![
            group(
                "records",
                CompressionRatio::Ratio4,
                CompressedRecordFormat::Fp8E4m3Block64,
                StateUpdate::Append,
                vec![
                    (StatePortRole::CompressedKv, "past_kv", "present_kv"),
                    (StatePortRole::IndexKey, "past_index", "present_index"),
                ],
            ),
            group(
                "carries",
                CompressionRatio::Ratio4,
                CompressedRecordFormat::Fp8E4m3Block64,
                StateUpdate::Replace,
                vec![
                    (
                        StatePortRole::CompressionCarry,
                        "past_carry",
                        "present_carry",
                    ),
                    (
                        StatePortRole::IndexCarry,
                        "past_index_carry",
                        "present_index_carry",
                    ),
                ],
            ),
        ];
        let inputs = vec![
            meta("past_kv", DataType::Uint8, &[1, 0, 583]),
            meta("past_index", DataType::Uint8, &[1, 0, 68]),
            meta("past_carry", DataType::Float32, &[1, 8, 2, 1024]),
            meta("past_index_carry", DataType::Float32, &[1, 8, 2, 256]),
        ];
        let outputs = vec![
            meta("present_kv", DataType::Uint8, &[1, 1, 583]),
            meta("present_index", DataType::Uint8, &[1, 1, 68]),
            meta("present_carry", DataType::Float32, &[1, 8, 2, 1024]),
            meta("present_index_carry", DataType::Float32, &[1, 8, 2, 256]),
        ];
        let plan = resolve_compressed_state(&inputs, &outputs, &groups).unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(
            plan.record_for_past("past_kv").unwrap().ratio,
            CompressionRatio::Ratio4
        );
    }

    #[test]
    fn ratio128_f32_records_resolve() {
        let groups = vec![group(
            "records",
            CompressionRatio::Ratio128,
            CompressedRecordFormat::F32,
            StateUpdate::Append,
            vec![(StatePortRole::CompressedKv, "past_kv", "present_kv")],
        )];
        let plan = resolve_compressed_state(
            &[meta("past_kv", DataType::Float32, &[1, 0, 512])],
            &[meta("present_kv", DataType::Float32, &[1, 1, 512])],
            &groups,
        )
        .unwrap();
        assert_eq!(
            plan.record_for_past("past_kv").unwrap().ratio,
            CompressionRatio::Ratio128
        );
    }

    fn facts(dtype: DataType, shape: &[usize]) -> RecordTensorFacts<'_> {
        RecordTensorFacts { dtype, shape }
    }

    fn ratio128_spec() -> RecordStateSpec {
        let plan = resolve_compressed_state(
            &[meta("past_kv", DataType::Float32, &[1, 0, 512])],
            &[meta("present_kv", DataType::Float32, &[1, 1, 512])],
            &[group(
                "records",
                CompressionRatio::Ratio128,
                CompressedRecordFormat::F32,
                StateUpdate::Append,
                vec![(StatePortRole::CompressedKv, "past_kv", "present_kv")],
            )],
        )
        .unwrap();
        plan.records().next().unwrap().clone()
    }

    #[test]
    fn ratio128_transition_accepts_256_and_257_token_boundaries() {
        let spec = ratio128_spec();
        validate_record_transition(
            &spec,
            facts(DataType::Float32, &[1, 1, 512]),
            facts(DataType::Float32, &[1, 2, 512]),
            255,
            256,
            1,
        )
        .unwrap();
        validate_record_transition(
            &spec,
            facts(DataType::Float32, &[1, 2, 512]),
            facts(DataType::Float32, &[1, 2, 512]),
            256,
            257,
            1,
        )
        .unwrap();
    }

    #[test]
    fn lowered_pairing_must_match_the_property_typed_transition() {
        let spec = ratio128_spec();
        let plan = resolve_compressed_state(
            &[meta("past_kv", DataType::Float32, &[1, 0, 512])],
            &[meta("present_kv", DataType::Float32, &[1, 1, 512])],
            &[group(
                "records",
                CompressionRatio::Ratio128,
                CompressedRecordFormat::F32,
                StateUpdate::Append,
                vec![(StatePortRole::CompressedKv, "past_kv", "present_kv")],
            )],
        )
        .unwrap();
        let error = plan
            .verify_pairing(&HashMap::from([(
                spec.output.clone(),
                "another_past".to_string(),
            )]))
            .unwrap_err();
        assert!(
            error
                .downcast_ref::<CompressedStateLoadRefusal>()
                .is_some_and(|reason| matches!(
                    reason,
                    CompressedStateLoadRefusal::PairingMismatch {
                        input,
                        output,
                        ..
                    } if input == "past_kv" && output == "present_kv"
                ))
        );
    }

    #[test]
    fn short_and_long_present_record_cursors_are_typed_refusals() {
        let spec = ratio128_spec();
        for (total_len, records, expected) in [(256, 1, 2), (255, 2, 1)] {
            let error = validate_record_transition(
                &spec,
                facts(DataType::Float32, &[1, 1, 512]),
                facts(DataType::Float32, &[1, records, 512]),
                255,
                total_len,
                1,
            )
            .unwrap_err();
            assert!(
                error
                    .downcast_ref::<CompressedStateTransitionRefusal>()
                    .is_some_and(|reason| matches!(
                        reason,
                        CompressedStateTransitionRefusal::CursorMismatch {
                            phase: CompressedStateTensorPhase::Present,
                            expected_records,
                            actual_records,
                            ..
                        } if *expected_records == expected && *actual_records == records
                    ))
            );
        }
    }

    #[test]
    fn stale_past_record_cursor_is_a_typed_refusal() {
        let spec = ratio128_spec();
        let error = validate_record_transition(
            &spec,
            facts(DataType::Float32, &[1, 0, 512]),
            facts(DataType::Float32, &[1, 2, 512]),
            255,
            256,
            1,
        )
        .unwrap_err();
        assert!(
            error
                .downcast_ref::<CompressedStateTransitionRefusal>()
                .is_some_and(|reason| matches!(
                    reason,
                    CompressedStateTransitionRefusal::CursorMismatch {
                        phase: CompressedStateTensorPhase::Past,
                        expected_records: 1,
                        actual_records: 0,
                        ..
                    }
                ))
        );
    }

    #[test]
    fn regressing_present_record_cursor_is_a_typed_refusal() {
        let spec = ratio128_spec();
        let error = validate_record_transition(
            &spec,
            facts(DataType::Float32, &[1, 2, 512]),
            facts(DataType::Float32, &[1, 1, 512]),
            256,
            257,
            1,
        )
        .unwrap_err();
        assert!(
            error
                .downcast_ref::<CompressedStateTransitionRefusal>()
                .is_some_and(|reason| matches!(
                    reason,
                    CompressedStateTransitionRefusal::NonMonotonicCursor {
                        past_records: 2,
                        present_records: 1,
                        ..
                    }
                ))
        );
    }

    #[test]
    fn batch_dtype_and_record_width_are_descriptor_checked() {
        let spec = ratio128_spec();
        let cases = [
            facts(DataType::Float32, &[2, 1, 512]),
            facts(DataType::Uint8, &[1, 1, 512]),
            facts(DataType::Float32, &[1, 1, 511]),
        ];
        for present in cases {
            let error = validate_record_transition(
                &spec,
                facts(DataType::Float32, &[1, 1, 512]),
                present,
                128,
                129,
                1,
            )
            .unwrap_err();
            assert!(
                error
                    .downcast_ref::<CompressedStateTransitionRefusal>()
                    .is_some_and(|reason| matches!(
                        reason,
                        CompressedStateTransitionRefusal::BatchMismatch { .. }
                            | CompressedStateTransitionRefusal::DtypeMismatch { .. }
                            | CompressedStateTransitionRefusal::RecordLayoutMismatch { .. }
                    ))
            );
        }
    }

    #[test]
    fn cuda_decline_is_typed() {
        let groups = vec![group(
            "records",
            CompressionRatio::Ratio128,
            CompressedRecordFormat::F32,
            StateUpdate::Append,
            vec![(StatePortRole::CompressedKv, "past_kv", "present_kv")],
        )];
        let error = refuse_compressed_records_on_cuda(&groups, true).unwrap_err();
        assert!(
            error
                .downcast_ref::<CompressedStateLoadRefusal>()
                .is_some_and(|reason| *reason == CompressedStateLoadRefusal::UnsupportedDevice)
        );
    }
}
