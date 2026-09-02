//! Canonical compressed-attention state lowering for native decode.

use super::*;
use onnx_genai_metadata::{
    CompressedRecordFormat, CompressionRatio, CompressionRecurrence, DecoderStateGroup,
    StateAliasing, StateGroupProperties, StateKind, StatePortRole, StateUpdate,
};
use onnx_runtime_ir::Dim;
use onnx_runtime_session::{IoMeta, Tensor};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

static COMPRESSED_STATE_MAP_LOOKUPS: AtomicU64 = AtomicU64::new(0);

pub fn compressed_state_map_lookups() -> u64 {
    COMPRESSED_STATE_MAP_LOOKUPS.load(Ordering::Relaxed)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RecordStateSpec {
    pub group: String,
    pub layer: usize,
    pub role: StatePortRole,
    pub input: String,
    pub output: String,
    pub ratio: CompressionRatio,
    pub batch_axis: usize,
    pub sequence_axis: usize,
    pub dtype: DataType,
    rank: usize,
    record_extents: Vec<(usize, usize)>,
    pub record_width_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CarryStateSpec {
    pub group: String,
    pub layer: usize,
    pub role: StatePortRole,
    pub input: String,
    pub output: String,
    pub dtype: DataType,
    batch_axis: usize,
    rank: usize,
    fixed_extents: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompressedStateTransitionIndex {
    Record(usize),
    Carry(usize),
}

#[derive(Debug)]
struct CompressedStateIndexes {
    present: HashMap<String, CompressedStateTransitionIndex>,
    record_past: HashMap<String, usize>,
    state_past_names: HashSet<String>,
    map_lookups: AtomicU64,
}

#[derive(Debug, Default)]
pub(super) struct CompressedStatePlan {
    records: Vec<RecordStateSpec>,
    carries: Vec<CarryStateSpec>,
    indexes: Option<CompressedStateIndexes>,
    required_aliasing_groups: Vec<String>,
}

pub(super) enum CompressedStateTransitionSpec<'a> {
    Record(&'a RecordStateSpec),
    Carry(&'a CarryStateSpec),
}

impl CompressedStatePlan {
    pub fn is_empty(&self) -> bool {
        self.indexes.is_none()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn records(&self) -> impl Iterator<Item = &RecordStateSpec> {
        self.records.iter()
    }

    pub fn transition_for_present(
        &self,
        present: &str,
    ) -> Option<CompressedStateTransitionSpec<'_>> {
        let indexes = self.indexes.as_ref()?;
        indexes.map_lookups.fetch_add(1, Ordering::Relaxed);
        COMPRESSED_STATE_MAP_LOOKUPS.fetch_add(1, Ordering::Relaxed);
        match indexes.present.get(present)? {
            CompressedStateTransitionIndex::Record(index) => {
                Some(CompressedStateTransitionSpec::Record(&self.records[*index]))
            }
            CompressedStateTransitionIndex::Carry(index) => {
                Some(CompressedStateTransitionSpec::Carry(&self.carries[*index]))
            }
        }
    }

    pub fn record_for_past(&self, past: &str) -> Option<&RecordStateSpec> {
        let indexes = self.indexes.as_ref()?;
        indexes.map_lookups.fetch_add(1, Ordering::Relaxed);
        COMPRESSED_STATE_MAP_LOOKUPS.fetch_add(1, Ordering::Relaxed);
        indexes
            .record_past
            .get(past)
            .map(|&index| &self.records[index])
    }

    #[cfg(test)]
    pub fn carry_for_present(&self, present: &str) -> Option<&CarryStateSpec> {
        match self.transition_for_present(present) {
            Some(CompressedStateTransitionSpec::Carry(spec)) => Some(spec),
            Some(CompressedStateTransitionSpec::Record(_)) | None => None,
        }
    }

    pub fn contains_past(&self, past: &str) -> bool {
        let Some(indexes) = &self.indexes else {
            return false;
        };
        indexes.map_lookups.fetch_add(1, Ordering::Relaxed);
        COMPRESSED_STATE_MAP_LOOKUPS.fetch_add(1, Ordering::Relaxed);
        indexes.record_past.contains_key(past)
    }

    pub fn state_past_names(&self) -> impl Iterator<Item = &String> {
        self.indexes
            .iter()
            .flat_map(|indexes| indexes.state_past_names.iter())
    }

    pub fn map_lookups(&self) -> u64 {
        self.indexes
            .as_ref()
            .map(|indexes| indexes.map_lookups.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    pub fn required_aliasing_groups(&self) -> &[String] {
        &self.required_aliasing_groups
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
    InvalidBatchAxis(String),
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
            Self::InvalidBatchAxis(message) => formatter.write_str(message),
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
                "compressed-attention record state is unavailable in this native CUDA decoder: \
                 graph-visible records require typed fixed-capacity device bindings with a \
                 compression-cadence cursor, snapshot/restore ownership, and capture-stable \
                 addresses. Use native CPU or a CUDA implementation that advertises this \
                 state-group capability; native decode will not fall back the whole session to CPU",
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
    CarryLayoutMismatch {
        group: String,
        layer: usize,
        role: StatePortRole,
        phase: CompressedStateTensorPhase,
        port: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    NonContiguousLayout {
        group: String,
        layer: usize,
        role: StatePortRole,
        phase: CompressedStateTensorPhase,
        port: String,
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
            Self::CarryLayoutMismatch {
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
                 '{port}' has shape {actual:?}, expected {expected:?}",
                role_name(*role)
            ),
            Self::NonContiguousLayout {
                group,
                layer,
                role,
                phase,
                port,
            } => write!(
                formatter,
                "compressed-attention group '{group}' layer {layer} role '{}' {phase} port \
                 '{port}' is non-contiguous; root native state snapshot/restore copies exact \
                 contiguous tensor bytes and refuses strided state rather than truncating or \
                 reinterpreting strides",
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
    batch_axis: usize,
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
    match (input.shape[batch_axis], output.shape[batch_axis]) {
        (Dim::Symbolic(_), Dim::Symbolic(_)) => {}
        (Dim::Static(past), Dim::Static(present)) if past == present => {}
        (past, present) => {
            return Err(anyhow::Error::new(
                CompressedStateLoadRefusal::InvalidBatchAxis(format!(
                    "compressed-attention record group '{group}' batch axis {batch_axis} must \
                     preserve one batch extent across '{}' and '{}', got {past:?} and {present:?}",
                    input.name, output.name
                )),
            ));
        }
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
        if axis == batch_axis || axis == sequence_axis {
            continue;
        }
        let extent = match (past, present) {
            (Dim::Static(past), Dim::Static(present)) if past == present => Some(past),
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
        let extent = extent.expect("record extents exclude batch and sequence axes");
        record_extents.push((axis, extent));
        record_elements = record_elements.checked_mul(extent).ok_or_else(|| {
            anyhow::Error::new(CompressedStateLoadRefusal::InvalidRecordLayout(format!(
                "compressed-attention record group '{group}' record width overflows usize"
            )))
        })?;
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

fn carry_layout(
    group: &str,
    input: &IoMeta,
    output: &IoMeta,
    batch_axis: usize,
) -> anyhow::Result<Vec<(usize, usize)>> {
    if input.shape.len() != output.shape.len() {
        return Err(anyhow::Error::new(
            CompressedStateLoadRefusal::InvalidRecordLayout(format!(
                "compressed-attention carry group '{group}' requires matching input/output rank, \
                 got {:?} and {:?}",
                input.shape, output.shape
            )),
        ));
    }
    if input.shape.is_empty() || batch_axis >= input.shape.len() {
        return Err(anyhow::Error::new(
            CompressedStateLoadRefusal::InvalidBatchAxis(format!(
                "compressed-attention carry group '{group}' requires a declared request-batch \
                 axis within its non-scalar input/output rank, got batch axis {batch_axis} for \
                 shapes {:?} and {:?}. Scalar carries are not row-scoped and must be rejected by \
                 metadata validation",
                input.shape, output.shape
            )),
        ));
    }
    let mut fixed_extents = Vec::with_capacity(input.shape.len().saturating_sub(1));
    for (axis, (past, present)) in input
        .shape
        .iter()
        .copied()
        .zip(output.shape.iter().copied())
        .enumerate()
    {
        match (axis == batch_axis, past, present) {
            (true, Dim::Symbolic(_), Dim::Symbolic(_)) => {}
            (true, Dim::Static(past), Dim::Static(present)) if past == present => {}
            (false, Dim::Static(past), Dim::Static(present)) if past == present => {
                fixed_extents.push((axis, past));
            }
            _ => {
                return Err(anyhow::Error::new(
                    CompressedStateLoadRefusal::InvalidRecordLayout(format!(
                        "compressed-attention carry group '{group}' axis {axis} must preserve one \
                         fixed extent across '{}' and '{}', got {past:?} and {present:?}; only \
                         declared batch axis {batch_axis} may be dynamic",
                        input.name, output.name
                    )),
                ));
            }
        }
    }
    Ok(fixed_extents)
}

pub(super) fn resolve_compressed_state(
    inputs: &[IoMeta],
    outputs: &[IoMeta],
    groups: &[DecoderStateGroup],
) -> anyhow::Result<CompressedStatePlan> {
    type LayerProperties = (
        CompressionRatio,
        CompressedRecordFormat,
        CompressionRecurrence,
    );
    type LayerEntry = (LayerProperties, BTreeMap<StatePortRole, (String, String)>);

    let mut plan = CompressedStatePlan::default();
    let mut occupied = HashSet::new();
    let mut layers: BTreeMap<usize, LayerEntry> = BTreeMap::new();
    let mut present_index = HashMap::new();
    let mut record_past_index = HashMap::new();
    let mut state_past_names = HashSet::new();

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
        if group.ports.is_empty() {
            return Err(anyhow::Error::new(CompressedStateLoadRefusal::InvalidRole(
                format!(
                    "compressed-attention state group '{}' has no read-write state ports",
                    group.name
                ),
            )));
        }
        if group.aliasing == StateAliasing::Required {
            plan.required_aliasing_groups.push(group.name.clone());
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
                || (is_carry && group.sequence_axis.is_some())
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
            let Some(layer) = port.layer else {
                return Err(anyhow::Error::new(CompressedStateLoadRefusal::InvalidRole(
                    format!(
                        "compressed-attention state group '{}' role '{}' must declare its layer \
                         index",
                        group.name,
                        role_name(role)
                    ),
                )));
            };
            let Some(batch_axis) = port.batch_axis else {
                return Err(anyhow::Error::new(
                    CompressedStateLoadRefusal::InvalidBatchAxis(format!(
                        "compressed-attention state group '{}' role '{}' has no request-batch \
                         axis; declare a request_aligned batch_layout on its canonical workflow \
                         state cell",
                        group.name,
                        role_name(role)
                    )),
                ));
            };
            let properties = (ratio, record_format, recurrence);
            let layer_entry = layers
                .entry(layer)
                .or_insert_with(|| (properties, BTreeMap::new()));
            if layer_entry.0 != properties {
                return Err(anyhow::Error::new(CompressedStateLoadRefusal::InvalidRole(
                    format!(
                        "compressed-attention layer {layer} has contradictory properties across \
                         state groups, including '{}'",
                        group.name
                    ),
                )));
            }
            if let Some((previous_group, previous_port)) = layer_entry
                .1
                .insert(role, (group.name.clone(), port.input.clone()))
            {
                return Err(anyhow::Error::new(CompressedStateLoadRefusal::InvalidRole(
                    format!(
                        "compressed-attention layer {layer} role '{}' is declared more than once: \
                         group '{previous_group}' port '{previous_port}' and group '{}' port '{}'",
                        role_name(role),
                        group.name,
                        port.input
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
            state_past_names.insert(port.input.clone());

            if is_record {
                let Some(axis) = group.sequence_axis.filter(|axis| {
                    *axis < input.shape.len() && *axis < output.shape.len() && *axis != batch_axis
                }) else {
                    return Err(anyhow::Error::new(
                        CompressedStateLoadRefusal::InvalidSequenceAxis(group.name.clone()),
                    ));
                };
                if batch_axis >= input.shape.len() || batch_axis >= output.shape.len() {
                    return Err(anyhow::Error::new(
                        CompressedStateLoadRefusal::InvalidBatchAxis(format!(
                            "compressed-attention record group '{}' declares batch axis \
                             {batch_axis} outside input/output shapes {:?} and {:?}",
                            group.name, input.shape, output.shape
                        )),
                    ));
                }
                let (record_extents, record_width_bytes) =
                    record_layout(&group.name, input, output, batch_axis, axis)?;
                let index = plan.records.len();
                present_index.insert(
                    port.output.clone(),
                    CompressedStateTransitionIndex::Record(index),
                );
                record_past_index.insert(port.input.clone(), index);
                plan.records.push(RecordStateSpec {
                    group: group.name.clone(),
                    layer,
                    role,
                    input: port.input.clone(),
                    output: port.output.clone(),
                    ratio,
                    batch_axis,
                    sequence_axis: axis,
                    dtype: expected,
                    rank: input.shape.len(),
                    record_extents,
                    record_width_bytes,
                });
            } else {
                let fixed_extents = carry_layout(&group.name, input, output, batch_axis)?;
                let index = plan.carries.len();
                present_index.insert(
                    port.output.clone(),
                    CompressedStateTransitionIndex::Carry(index),
                );
                plan.carries.push(CarryStateSpec {
                    group: group.name.clone(),
                    layer,
                    role,
                    input: port.input.clone(),
                    output: port.output.clone(),
                    dtype: expected,
                    batch_axis,
                    rank: input.shape.len(),
                    fixed_extents,
                });
            }
        }
    }
    for (layer, ((ratio, _, _), roles)) in layers {
        let expected: &[StatePortRole] = if ratio == CompressionRatio::Ratio4 {
            &[
                StatePortRole::CompressedKv,
                StatePortRole::CompressionCarry,
                StatePortRole::IndexKey,
                StatePortRole::IndexCarry,
            ]
        } else {
            &[StatePortRole::CompressedKv, StatePortRole::CompressionCarry]
        };
        let missing = expected
            .iter()
            .copied()
            .filter(|role| !roles.contains_key(role))
            .map(role_name)
            .collect::<Vec<_>>();
        let unexpected = roles
            .keys()
            .copied()
            .filter(|role| !expected.contains(role))
            .map(role_name)
            .collect::<Vec<_>>();
        if !missing.is_empty() || !unexpected.is_empty() {
            return Err(anyhow::Error::new(CompressedStateLoadRefusal::InvalidRole(
                format!(
                    "compressed-attention layer {layer} with ratio {ratio:?} has incomplete typed \
                     state roles; missing {missing:?}, unexpected {unexpected:?}"
                ),
            )));
        }
    }
    if !plan.records.is_empty() || !plan.carries.is_empty() {
        plan.indexes = Some(CompressedStateIndexes {
            present: present_index,
            record_past: record_past_index,
            state_past_names,
            map_lookups: AtomicU64::new(0),
        });
    }
    Ok(plan)
}

#[derive(Clone, Copy)]
pub(super) struct RecordTensorFacts<'a> {
    dtype: DataType,
    shape: &'a [usize],
    contiguous: bool,
}

impl<'a> From<&'a Tensor> for RecordTensorFacts<'a> {
    fn from(tensor: &'a Tensor) -> Self {
        Self {
            dtype: tensor.dtype,
            shape: &tensor.shape,
            contiguous: tensor.layout.is_contiguous(&tensor.shape),
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
    if !tensor.contiguous {
        return refusal(CompressedStateTransitionRefusal::NonContiguousLayout {
            group: spec.group.clone(),
            layer: spec.layer,
            role: spec.role,
            phase,
            port: port.to_string(),
        });
    }
    if tensor.shape[spec.batch_axis] != batch {
        return refusal(CompressedStateTransitionRefusal::BatchMismatch {
            group: spec.group.clone(),
            layer: spec.layer,
            role: spec.role,
            phase,
            port: port.to_string(),
            expected: batch,
            actual: tensor.shape[spec.batch_axis],
        });
    }
    let mut actual = tensor
        .shape
        .iter()
        .copied()
        .enumerate()
        .filter(|(axis, _)| *axis != spec.batch_axis && *axis != spec.sequence_axis);
    let mut record_elements = 1_usize;
    let mut layout_matches = true;
    for expected in &spec.record_extents {
        let Some(found) = actual.next() else {
            layout_matches = false;
            break;
        };
        layout_matches &= found == *expected;
        record_elements = record_elements.saturating_mul(found.1);
    }
    layout_matches &= actual.next().is_none();
    let actual_bytes = tensor
        .dtype
        .checked_storage_bytes(record_elements)
        .unwrap_or(usize::MAX);
    if !layout_matches || actual_bytes != spec.record_width_bytes {
        let actual_extents = tensor
            .shape
            .iter()
            .copied()
            .enumerate()
            .filter(|(axis, _)| *axis != spec.batch_axis && *axis != spec.sequence_axis)
            .collect();
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

fn validate_carry_tensor_contract(
    spec: &CarryStateSpec,
    tensor: RecordTensorFacts<'_>,
    phase: CompressedStateTensorPhase,
    port: &str,
    batch: usize,
) -> anyhow::Result<()> {
    if tensor.dtype != spec.dtype {
        return Err(anyhow::Error::new(
            CompressedStateTransitionRefusal::DtypeMismatch {
                group: spec.group.clone(),
                layer: spec.layer,
                role: spec.role,
                phase,
                port: port.to_string(),
                expected: spec.dtype,
                actual: tensor.dtype,
            },
        ));
    }
    if tensor.shape.len() != spec.rank {
        return Err(anyhow::Error::new(
            CompressedStateTransitionRefusal::RankMismatch {
                group: spec.group.clone(),
                layer: spec.layer,
                role: spec.role,
                phase,
                port: port.to_string(),
                expected: spec.rank,
                actual: tensor.shape.len(),
            },
        ));
    }
    if !tensor.contiguous {
        return Err(anyhow::Error::new(
            CompressedStateTransitionRefusal::NonContiguousLayout {
                group: spec.group.clone(),
                layer: spec.layer,
                role: spec.role,
                phase,
                port: port.to_string(),
            },
        ));
    }
    if tensor.shape[spec.batch_axis] != batch {
        return Err(anyhow::Error::new(
            CompressedStateTransitionRefusal::BatchMismatch {
                group: spec.group.clone(),
                layer: spec.layer,
                role: spec.role,
                phase,
                port: port.to_string(),
                expected: batch,
                actual: tensor.shape[spec.batch_axis],
            },
        ));
    }
    let layout_matches = spec
        .fixed_extents
        .iter()
        .all(|&(axis, extent)| tensor.shape[axis] == extent);
    if !layout_matches {
        let mut expected = tensor.shape.to_vec();
        expected[spec.batch_axis] = batch;
        for &(axis, extent) in &spec.fixed_extents {
            expected[axis] = extent;
        }
        return Err(anyhow::Error::new(
            CompressedStateTransitionRefusal::CarryLayoutMismatch {
                group: spec.group.clone(),
                layer: spec.layer,
                role: spec.role,
                phase,
                port: port.to_string(),
                expected,
                actual: tensor.shape.to_vec(),
            },
        ));
    }
    Ok(())
}

pub(super) fn validate_carry_transition(
    spec: &CarryStateSpec,
    past: RecordTensorFacts<'_>,
    present: RecordTensorFacts<'_>,
    batch: usize,
) -> anyhow::Result<()> {
    validate_carry_tensor_contract(
        spec,
        past,
        CompressedStateTensorPhase::Past,
        &spec.input,
        batch,
    )?;
    validate_carry_tensor_contract(
        spec,
        present,
        CompressedStateTensorPhase::Present,
        &spec.output,
        batch,
    )
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

    fn meta_dims(name: &str, dtype: DataType, shape: Vec<Dim>) -> IoMeta {
        IoMeta {
            name: name.to_string(),
            dtype,
            shape,
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
            layout: match &update {
                StateUpdate::Append => "batch_record_feature",
                StateUpdate::Replace => "batch_carry_slot_stream_feature",
                StateUpdate::IndexedScatter { .. } => "batch_record_feature",
            }
            .to_string(),
            aliasing: StateAliasing::Forbidden,
            update: Some(update),
            reuse: Default::default(),
            capabilities: StateGroupCapabilities::default(),
            ports: ports
                .into_iter()
                .map(|(role, input, output)| DecoderStatePort {
                    role: Some(role),
                    layer: Some(0),
                    batch_axis: Some(0),
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
        let groups = vec![
            group(
                "records",
                CompressionRatio::Ratio128,
                CompressedRecordFormat::F32,
                StateUpdate::Append,
                vec![(StatePortRole::CompressedKv, "past_kv", "present_kv")],
            ),
            group(
                "carries",
                CompressionRatio::Ratio128,
                CompressedRecordFormat::F32,
                StateUpdate::Replace,
                vec![(
                    StatePortRole::CompressionCarry,
                    "past_carry",
                    "present_carry",
                )],
            ),
        ];
        let plan = resolve_compressed_state(
            &[
                meta("past_kv", DataType::Float32, &[1, 0, 512]),
                meta("past_carry", DataType::Float32, &[1, 128, 2, 512]),
            ],
            &[
                meta("present_kv", DataType::Float32, &[1, 1, 512]),
                meta("present_carry", DataType::Float32, &[1, 128, 2, 512]),
            ],
            &groups,
        )
        .unwrap();
        assert_eq!(
            plan.record_for_past("past_kv").unwrap().ratio,
            CompressionRatio::Ratio128
        );
    }

    fn facts(dtype: DataType, shape: &[usize]) -> RecordTensorFacts<'_> {
        RecordTensorFacts {
            dtype,
            shape,
            contiguous: true,
        }
    }

    fn ratio128_spec() -> RecordStateSpec {
        let plan = resolve_compressed_state(
            &[
                meta("past_kv", DataType::Float32, &[1, 0, 512]),
                meta("past_carry", DataType::Float32, &[1, 128, 2, 512]),
            ],
            &[
                meta("present_kv", DataType::Float32, &[1, 1, 512]),
                meta("present_carry", DataType::Float32, &[1, 128, 2, 512]),
            ],
            &[
                group(
                    "records",
                    CompressionRatio::Ratio128,
                    CompressedRecordFormat::F32,
                    StateUpdate::Append,
                    vec![(StatePortRole::CompressedKv, "past_kv", "present_kv")],
                ),
                group(
                    "carries",
                    CompressionRatio::Ratio128,
                    CompressedRecordFormat::F32,
                    StateUpdate::Replace,
                    vec![(
                        StatePortRole::CompressionCarry,
                        "past_carry",
                        "present_carry",
                    )],
                ),
            ],
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
            &[
                meta("past_kv", DataType::Float32, &[1, 0, 512]),
                meta("past_carry", DataType::Float32, &[1, 128, 2, 512]),
            ],
            &[
                meta("present_kv", DataType::Float32, &[1, 1, 512]),
                meta("present_carry", DataType::Float32, &[1, 128, 2, 512]),
            ],
            &[
                group(
                    "records",
                    CompressionRatio::Ratio128,
                    CompressedRecordFormat::F32,
                    StateUpdate::Append,
                    vec![(StatePortRole::CompressedKv, "past_kv", "present_kv")],
                ),
                group(
                    "carries",
                    CompressionRatio::Ratio128,
                    CompressedRecordFormat::F32,
                    StateUpdate::Replace,
                    vec![(
                        StatePortRole::CompressionCarry,
                        "past_carry",
                        "present_carry",
                    )],
                ),
            ],
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
    fn carry_shape_and_dtype_are_checked_before_commit() {
        let plan = resolve_compressed_state(
            &[
                meta("past_kv", DataType::Float32, &[1, 0, 512]),
                meta("past_carry", DataType::Float32, &[1, 128, 2, 512]),
            ],
            &[
                meta("present_kv", DataType::Float32, &[1, 1, 512]),
                meta("present_carry", DataType::Float32, &[1, 128, 2, 512]),
            ],
            &[
                group(
                    "records",
                    CompressionRatio::Ratio128,
                    CompressedRecordFormat::F32,
                    StateUpdate::Append,
                    vec![(StatePortRole::CompressedKv, "past_kv", "present_kv")],
                ),
                group(
                    "carries",
                    CompressionRatio::Ratio128,
                    CompressedRecordFormat::F32,
                    StateUpdate::Replace,
                    vec![(
                        StatePortRole::CompressionCarry,
                        "past_carry",
                        "present_carry",
                    )],
                ),
            ],
        )
        .unwrap();
        let spec = plan.carry_for_present("present_carry").unwrap();
        for present in [
            facts(DataType::Float32, &[1, 127, 2, 512]),
            facts(DataType::Uint8, &[1, 128, 2, 512]),
        ] {
            let error = validate_carry_transition(
                spec,
                facts(DataType::Float32, &[1, 128, 2, 512]),
                present,
                1,
            )
            .unwrap_err();
            assert!(
                error
                    .downcast_ref::<CompressedStateTransitionRefusal>()
                    .is_some_and(|reason| matches!(
                        reason,
                        CompressedStateTransitionRefusal::CarryLayoutMismatch { .. }
                            | CompressedStateTransitionRefusal::DtypeMismatch { .. }
                    ))
            );
        }
    }

    #[test]
    fn rank_zero_carry_is_rejected_as_non_row_scoped_state() {
        let mut carries = group(
            "carries",
            CompressionRatio::Ratio128,
            CompressedRecordFormat::F32,
            StateUpdate::Replace,
            vec![(
                StatePortRole::CompressionCarry,
                "past_carry",
                "present_carry",
            )],
        );
        carries.ports[0].batch_axis = None;
        let error = resolve_compressed_state(
            &[
                meta("past_kv", DataType::Float32, &[1, 0, 8]),
                meta("past_carry", DataType::Float32, &[]),
            ],
            &[
                meta("present_kv", DataType::Float32, &[1, 0, 8]),
                meta("present_carry", DataType::Float32, &[]),
            ],
            &[
                group(
                    "records",
                    CompressionRatio::Ratio128,
                    CompressedRecordFormat::F32,
                    StateUpdate::Append,
                    vec![(StatePortRole::CompressedKv, "past_kv", "present_kv")],
                ),
                carries,
            ],
        )
        .unwrap_err();
        assert!(
            error
                .downcast_ref::<CompressedStateLoadRefusal>()
                .is_some_and(|reason| matches!(
                    reason,
                    CompressedStateLoadRefusal::InvalidBatchAxis(message)
                        if message.contains("request-batch axis")
                )),
            "{error:#}"
        );
    }

    #[test]
    fn rank_one_and_zero_extent_rank_two_carries_are_atomic_shapes() {
        for shape in [vec![1], vec![1, 0]] {
            let plan = resolve_compressed_state(
                &[
                    meta("past_kv", DataType::Float32, &[1, 0, 8]),
                    meta("past_carry", DataType::Float32, &shape),
                ],
                &[
                    meta("present_kv", DataType::Float32, &[1, 0, 8]),
                    meta("present_carry", DataType::Float32, &shape),
                ],
                &[
                    group(
                        "records",
                        CompressionRatio::Ratio128,
                        CompressedRecordFormat::F32,
                        StateUpdate::Append,
                        vec![(StatePortRole::CompressedKv, "past_kv", "present_kv")],
                    ),
                    group(
                        "carries",
                        CompressionRatio::Ratio128,
                        CompressedRecordFormat::F32,
                        StateUpdate::Replace,
                        vec![(
                            StatePortRole::CompressionCarry,
                            "past_carry",
                            "present_carry",
                        )],
                    ),
                ],
            )
            .unwrap();
            let spec = plan.carry_for_present("present_carry").unwrap();
            validate_carry_transition(
                spec,
                facts(DataType::Float32, &shape),
                facts(DataType::Float32, &shape),
                1,
            )
            .unwrap();
        }
    }

    #[test]
    fn dynamic_batch_and_record_axes_follow_declared_axis_numbers() {
        let batch = SymbolId(41);
        let records = SymbolId(42);
        let mut record_group = group(
            "records",
            CompressionRatio::Ratio128,
            CompressedRecordFormat::F32,
            StateUpdate::Append,
            vec![(StatePortRole::CompressedKv, "past_kv", "present_kv")],
        );
        record_group.sequence_axis = Some(2);
        record_group.ports[0].batch_axis = Some(1);
        let mut carry_group = group(
            "carries",
            CompressionRatio::Ratio128,
            CompressedRecordFormat::F32,
            StateUpdate::Replace,
            vec![(
                StatePortRole::CompressionCarry,
                "past_carry",
                "present_carry",
            )],
        );
        carry_group.ports[0].batch_axis = Some(1);
        let plan = resolve_compressed_state(
            &[
                meta_dims(
                    "past_kv",
                    DataType::Float32,
                    vec![Dim::Static(8), Dim::Symbolic(batch), Dim::Symbolic(records)],
                ),
                meta_dims(
                    "past_carry",
                    DataType::Float32,
                    vec![Dim::Static(3), Dim::Symbolic(batch)],
                ),
            ],
            &[
                meta_dims(
                    "present_kv",
                    DataType::Float32,
                    vec![Dim::Static(8), Dim::Symbolic(batch), Dim::Symbolic(records)],
                ),
                meta_dims(
                    "present_carry",
                    DataType::Float32,
                    vec![Dim::Static(3), Dim::Symbolic(batch)],
                ),
            ],
            &[record_group, carry_group],
        )
        .unwrap();
        let record = plan.records().next().unwrap();
        validate_record_transition(
            record,
            facts(DataType::Float32, &[8, 2, 0]),
            facts(DataType::Float32, &[8, 2, 1]),
            0,
            128,
            2,
        )
        .unwrap();
        let carry = plan.carry_for_present("present_carry").unwrap();
        validate_carry_transition(
            carry,
            facts(DataType::Float32, &[3, 2]),
            facts(DataType::Float32, &[3, 2]),
            2,
        )
        .unwrap();
    }

    #[test]
    fn non_contiguous_carry_is_a_typed_refusal() {
        let plan = resolve_compressed_state(
            &[
                meta("past_kv", DataType::Float32, &[1, 0, 8]),
                meta("past_carry", DataType::Float32, &[1, 3]),
            ],
            &[
                meta("present_kv", DataType::Float32, &[1, 0, 8]),
                meta("present_carry", DataType::Float32, &[1, 3]),
            ],
            &[
                group(
                    "records",
                    CompressionRatio::Ratio128,
                    CompressedRecordFormat::F32,
                    StateUpdate::Append,
                    vec![(StatePortRole::CompressedKv, "past_kv", "present_kv")],
                ),
                group(
                    "carries",
                    CompressionRatio::Ratio128,
                    CompressedRecordFormat::F32,
                    StateUpdate::Replace,
                    vec![(
                        StatePortRole::CompressionCarry,
                        "past_carry",
                        "present_carry",
                    )],
                ),
            ],
        )
        .unwrap();
        let spec = plan.carry_for_present("present_carry").unwrap();
        let error = validate_carry_transition(
            spec,
            RecordTensorFacts {
                dtype: DataType::Float32,
                shape: &[1, 3],
                contiguous: false,
            },
            facts(DataType::Float32, &[1, 3]),
            1,
        )
        .unwrap_err();
        assert!(
            error
                .downcast_ref::<CompressedStateTransitionRefusal>()
                .is_some_and(|reason| matches!(
                    reason,
                    CompressedStateTransitionRefusal::NonContiguousLayout {
                        phase: CompressedStateTensorPhase::Past,
                        ..
                    }
                )),
            "{error:#}"
        );
    }

    #[test]
    fn incomplete_layer_roles_are_refused_before_state_is_lowered() {
        let error = resolve_compressed_state(
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
        .unwrap_err();
        assert!(
            error
                .downcast_ref::<CompressedStateLoadRefusal>()
                .is_some_and(|reason| matches!(
                    reason,
                    CompressedStateLoadRefusal::InvalidRole(message)
                        if message.contains("missing [\"compression_carry\"]")
                ))
        );
    }

    #[test]
    fn internal_snapshot_restores_compressed_state_transactionally() -> anyhow::Result<()> {
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tiny-deepseek-v4-csa/model.onnx.textproto");
        let mut session =
            NativeDecodeSession::load_with_resolved_io(&model, NativeDecodeDevice::Cpu)?;
        let mut oracle =
            NativeDecodeSession::load_with_resolved_io(&model, NativeDecodeDevice::Cpu)?;
        let prompt = [1, 2, 3, 4, 5, 6, 7, 8];
        session.decode(&prompt, 0)?;
        oracle.decode(&prompt, 0)?;

        let snapshot = session.snapshot_recurrent_state()?;
        let snapshot_bytes = snapshot
            .host
            .as_ref()
            .expect("CPU snapshot stores host state")
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.as_bytes().to_vec()))
            .collect::<BTreeMap<_, _>>();
        session.decode(&[9, 10, 11, 12], prompt.len())?;
        session.restore_state_snapshot_at(&snapshot, prompt.len())?;
        assert_eq!(session.current_len(), prompt.len());
        for (name, expected) in snapshot_bytes {
            assert_eq!(
                session.past.get(&name).expect("restored state").as_bytes(),
                expected,
                "state '{name}' changed across restore"
            );
        }

        assert_eq!(
            session.decode(&[13], prompt.len())?,
            oracle.decode(&[13], prompt.len())?,
            "restored continuation must match a session that never ran the rejected draft"
        );
        Ok(())
    }

    #[test]
    fn fixed_state_restore_failures_leave_dense_and_compressed_state_unchanged()
    -> anyhow::Result<()> {
        #[derive(Debug, PartialEq, Eq)]
        struct StateIdentity {
            shape: Vec<usize>,
            layout: String,
            address: usize,
            bytes: Vec<u8>,
        }

        fn identity(session: &NativeDecodeSession) -> BTreeMap<String, StateIdentity> {
            session
                .past
                .iter()
                .map(|(name, tensor)| {
                    (
                        name.clone(),
                        StateIdentity {
                            shape: tensor.shape.clone(),
                            layout: format!("{:?}", tensor.layout),
                            address: tensor.device_ptr() as usize,
                            bytes: tensor.as_bytes().to_vec(),
                        },
                    )
                })
                .collect()
        }

        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tiny-deepseek-v4-csa/model.onnx.textproto");
        let prompt = [1, 2, 3, 4, 5, 6, 7, 8];
        for failure in [0usize, 3, 5] {
            let mut session =
                NativeDecodeSession::load_with_resolved_io(&model, NativeDecodeDevice::Cpu)?;
            let mut oracle =
                NativeDecodeSession::load_with_resolved_io(&model, NativeDecodeDevice::Cpu)?;
            let mut sibling =
                NativeDecodeSession::load_with_resolved_io(&model, NativeDecodeDevice::Cpu)?;
            session.decode(&prompt, 0)?;
            oracle.decode(&prompt, 0)?;
            sibling.decode(&prompt, 0)?;
            let snapshot = session.snapshot_recurrent_state()?;
            let fixed_count = snapshot.host.as_ref().expect("host snapshot").len();
            assert_eq!(fixed_count, 6, "fixture fixed-state census drifted");

            session.decode(&[9, 10, 11, 12], prompt.len())?;
            let before = identity(&session);
            let before_len = session.current_len();
            let before_stats = session.compressed_state_path_stats();
            let sibling_before = identity(&sibling);
            let sibling_len = sibling.current_len();

            let error = {
                let _failure = fail_host_fixed_restore_at(failure);
                session
                    .restore_state_snapshot_at(&snapshot, prompt.len())
                    .expect_err("injected fixed-state stage must abort the transaction")
            };
            assert!(
                error.to_string().contains(&format!(
                    "injected host fixed-state restore failure at slot {failure}"
                )),
                "the initiating restore error must remain primary: {error:#}"
            );
            assert_eq!(session.current_len(), before_len);
            assert_eq!(session.compressed_state_path_stats(), before_stats);
            assert_eq!(
                identity(&session),
                before,
                "failure at fixed slot {failure} published a partial dense/fixed rollback"
            );
            assert_eq!(sibling.current_len(), sibling_len);
            assert_eq!(
                identity(&sibling),
                sibling_before,
                "failure at fixed slot {failure} crossed into a sibling session"
            );

            session.restore_state_snapshot_at(&snapshot, prompt.len())?;
            assert_eq!(
                session.decode(&[13], prompt.len())?,
                oracle.decode(&[13], prompt.len())?,
                "clean retry after fixed slot {failure} must match an untouched session"
            );
        }

        let mut session =
            NativeDecodeSession::load_with_resolved_io(&model, NativeDecodeDevice::Cpu)?;
        session.decode(&prompt, 0)?;
        let snapshot = session.snapshot_recurrent_state()?;
        session.decode(&[9, 10], prompt.len())?;
        let fixed = snapshot.host.as_ref().expect("host snapshot");
        let dense_name = session
            .past
            .keys()
            .find(|name| !fixed.contains_key(*name))
            .cloned()
            .expect("fixture carries dense KV");
        let rank = session.past[&dense_name].shape.len();
        session.past.get_mut(&dense_name).expect("dense KV").layout =
            onnx_runtime_ir::TensorLayout::strided(vec![0; rank]);
        let before = identity(&session);
        let before_len = session.current_len();
        let error = session
            .restore_state_snapshot_at(&snapshot, prompt.len())
            .expect_err("non-contiguous dense KV must fail before publication");
        let reported = format!("{error:#}");
        assert!(
            reported.contains("requires contiguous row-major storage"),
            "{reported}"
        );
        assert_eq!(session.current_len(), before_len);
        assert_eq!(
            identity(&session),
            before,
            "non-contiguous input refusal must leave every dense/fixed binding untouched"
        );
        Ok(())
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
