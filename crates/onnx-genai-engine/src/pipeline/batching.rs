//! Generic packed ownership and component-batch admission.
//!
//! Packed tensors have one physical axis. Nested structure is a bounded,
//! innermost-first ownership chain over that axis; it is never interpreted as
//! another tensor axis. Scheduling remains symbol-keyed and validates the
//! materialized footprint immediately before a component reaches a backend.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;
use std::sync::Arc;

use onnx_genai_metadata::{BatchLayout, TensorDimension, WorkflowComponent, WorkflowSpec};
use onnx_genai_ort::{DataType, Value};
use onnx_genai_scheduler::{
    AdmittedBatch, BatchAdmissionError, BatchContribution, BatchDimensionAggregation, BatchPolicy,
    MaterializedBudget, group_batch_contributions, validate_materialized_footprint,
};

use super::{PipelineTensors, WorkflowRuntime};
use crate::config::SessionId;

const MAX_OWNERSHIP_DEPTH: usize = 2;

/// A malformed ownership contract or unsafe component batch.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BatchContractError {
    #[error(
        "{component} value '{value}' declares packed axis {axis}; this runtime only accepts axis \
         zero so every request span is one contiguous no-copy window"
    )]
    PackedAxis {
        component: String,
        value: String,
        axis: usize,
    },
    #[error(
        "{component} value '{value}' has ownership depth {depth}; supported depth is 1..={max_depth}"
    )]
    OwnershipDepth {
        component: String,
        value: String,
        depth: usize,
        max_depth: usize,
    },
    #[error(
        "{component} value '{value}' ownership level {level} references missing {kind} companion \
         '{companion}'"
    )]
    MissingCompanion {
        component: String,
        value: String,
        level: usize,
        kind: &'static str,
        companion: String,
    },
    #[error(
        "{component} value '{value}' ownership level {level} {kind} companion '{companion}' must \
         be a host-resident rank-1 Int64 tensor: {reason}"
    )]
    InvalidCompanion {
        component: String,
        value: String,
        level: usize,
        kind: &'static str,
        companion: String,
        reason: String,
    },
    #[error(
        "{component} value '{value}' ownership level {level} offset {index} is {offset}; offsets \
         must be non-negative, start at zero, and be monotonically non-decreasing"
    )]
    InvalidOffset {
        component: String,
        value: String,
        level: usize,
        index: usize,
        offset: i64,
    },
    #[error(
        "{component} value '{value}' ownership level {level} terminal offset is {actual}, expected \
         {expected} from the level's unit extent"
    )]
    TerminalExtent {
        component: String,
        value: String,
        level: usize,
        actual: usize,
        expected: usize,
    },
    #[error(
        "{component} value '{value}' ownership level {level} owner extent is {actual}, expected \
         {expected}"
    )]
    OwnerExtent {
        component: String,
        value: String,
        level: usize,
        actual: usize,
        expected: usize,
    },
    #[error(
        "{component} value '{value}' ownership level {level} unit {unit} names owner {owner}, but \
         this level has {parent_count} parents"
    )]
    OwnerOutsideLevel {
        component: String,
        value: String,
        level: usize,
        unit: usize,
        owner: i64,
        parent_count: usize,
    },
    #[error(
        "{component} value '{value}' ownership level {level} unit {unit} names owner {actual}, but \
         the offset spans require owner {expected}; owners must preserve contiguous item order"
    )]
    OwnerOrder {
        component: String,
        value: String,
        level: usize,
        unit: usize,
        actual: usize,
        expected: usize,
    },
    #[error(
        "{component} value '{value}' ownership chain ends in {actual} request spans, expected \
         {expected}"
    )]
    RequestExtent {
        component: String,
        value: String,
        actual: usize,
        expected: usize,
    },
    #[error(
        "component '{component}' inputs disagree on logical request count: '{first_value}' carries \
         {first_count}, while '{value}' carries {count}; refusing a batch that could leak one \
         request's values into another"
    )]
    RequestCountMismatch {
        component: String,
        first_value: String,
        first_count: usize,
        value: String,
        count: usize,
    },
    #[error(
        "component '{component}' cannot validate batching before dispatch because required \
         batched input port '{port}' is unavailable; expose the port to the generic runtime or \
         execute this component without a batching contract"
    )]
    MissingBatchInput { component: String, port: String },
    #[error(
        "component '{component}' carries {request_count} requests but declares no batch_capacity; \
         absence permits exactly one request per invocation"
    )]
    UndeclaredCapacity {
        component: String,
        request_count: usize,
    },
    #[error(
        "component '{component}' batch contribution for sequence {sequence_id} carries \
         {request_count} logical requests; the grouping API requires one request per contribution"
    )]
    ContributionNotRequestLocal {
        component: String,
        sequence_id: SessionId,
        request_count: usize,
    },
    #[error(
        "component '{component}' capacity references shape symbol '{symbol}', but no typed input \
         dimension resolves it for runtime admission"
    )]
    MissingCapacityDimension { component: String, symbol: String },
    #[error(
        "component '{component}' shape symbol '{symbol}' has incompatible materialization modes \
         across typed inputs; do not use one symbol for both a packed/request count and a padded \
         or invariant extent"
    )]
    AmbiguousDimension { component: String, symbol: String },
    #[error("component '{component}' batch admission failed before backend enqueue: {source}")]
    Admission {
        component: String,
        #[source]
        source: BatchAdmissionError,
    },
    #[error(
        "packed request index {request} is outside the ownership chain's {request_count} requests"
    )]
    RequestIndex {
        request: usize,
        request_count: usize,
    },
    #[error(
        "ownership level {level} length extent is {actual}, expected {expected} units before slicing"
    )]
    LengthExtent {
        level: usize,
        actual: usize,
        expected: usize,
    },
    #[error("ownership arithmetic overflowed while {operation}")]
    ArithmeticOverflow { operation: &'static str },
    #[error("cannot compose zero ownership chains without a declared ownership depth")]
    EmptyComposition,
    #[error(
        "cannot compose ownership chains with different depths: first depth {expected}, found \
         depth {actual}"
    )]
    CompositionDepth { expected: usize, actual: usize },
    #[error(
        "cannot compose ownership chain {index}: it describes {request_count} requests, expected \
         one request-local chain"
    )]
    CompositionNotLocal { index: usize, request_count: usize },
    #[error(
        "packed payload axis-zero extent is {actual}, but its ownership chain describes {expected} \
         physical items"
    )]
    PayloadExtent { actual: usize, expected: usize },
}

/// Find the typed batch refusal through anyhow context wrappers.
pub fn batch_contract_error(error: &anyhow::Error) -> Option<&BatchContractError> {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<BatchContractError>())
}

/// Runtime-owned companion tensors for one ownership level.
///
/// [`PackedOwnership`] snapshots their Int64 contents during validation because
/// `Value` aliases permit shared in-place mutation. Payload views remain
/// zero-copy ORT aliases; only the small ownership companions cross this copy
/// boundary.
pub struct OwnershipLevelValues {
    offsets: Value,
    owners: Value,
}

impl OwnershipLevelValues {
    pub fn new(offsets: Value, owners: Value) -> Self {
        Self { offsets, owners }
    }
}

/// A validated, bounded ownership chain over one packed physical axis.
pub struct PackedOwnership {
    levels: Vec<OwnedOwnershipLevel>,
    packed_extent: usize,
    request_count: usize,
}

impl PackedOwnership {
    pub fn new(
        packed_extent: usize,
        levels: Vec<OwnershipLevelValues>,
        request_count: usize,
    ) -> Result<Self, BatchContractError> {
        let snapshots = levels
            .into_iter()
            .enumerate()
            .map(|(level, values)| {
                Ok(OwnedOwnershipLevel {
                    offsets: companion_values(
                        "<ownership>",
                        "<packed>",
                        level,
                        "offsets",
                        "offsets",
                        &values.offsets,
                    )?,
                    owners: companion_values(
                        "<ownership>",
                        "<packed>",
                        level,
                        "owner",
                        "owner",
                        &values.owners,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, BatchContractError>>()?;
        validate_ownership_slices(
            "<ownership>",
            "<packed>",
            packed_extent,
            &snapshots,
            Some(request_count),
        )?;
        Ok(Self {
            levels: snapshots,
            packed_extent,
            request_count,
        })
    }

    pub fn packed_extent(&self) -> usize {
        self.packed_extent
    }

    pub fn request_count(&self) -> usize {
        self.request_count
    }

    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// Borrow one request's local ownership, rebasing offsets and owner indices.
    pub fn request_view(
        &self,
        request: usize,
    ) -> Result<PackedRequestView<'_>, BatchContractError> {
        let ranges = request_level_ranges(&self.levels, request, self.request_count)?;
        let levels = self
            .levels
            .iter()
            .zip(&ranges)
            .map(|(level, range)| {
                let offsets = &level.offsets[range.parents.start..=range.parents.end];
                let owners = &level.owners[range.units.clone()];
                let owner_base = i64::try_from(range.parents.start).map_err(|_| {
                    BatchContractError::ArithmeticOverflow {
                        operation: "rebasing request-local owner indices",
                    }
                })?;
                Ok(OwnershipLevelView {
                    offsets: RebasedI64Slice::new(offsets, offsets[0]),
                    owners: RebasedI64Slice::new(owners, owner_base),
                })
            })
            .collect::<Result<Vec<_>, BatchContractError>>()?;
        Ok(PackedRequestView {
            item_span: ranges[0].units.clone(),
            levels,
        })
    }

    /// Slice a per-unit length vector for one request without changing values.
    ///
    /// `level == 0` slices item/frame lengths; `level == 1` slices container
    /// lengths. Unlike offsets and owners, lengths are magnitudes and must not
    /// be rebased.
    pub fn slice_lengths<'a, T>(
        &self,
        request: usize,
        level: usize,
        lengths: &'a [T],
    ) -> Result<&'a [T], BatchContractError> {
        let selected = self
            .levels
            .get(level)
            .ok_or(BatchContractError::OwnershipDepth {
                component: "<ownership>".into(),
                value: "<packed>".into(),
                depth: level + 1,
                max_depth: self.levels.len(),
            })?;
        if lengths.len() != selected.owners.len() {
            return Err(BatchContractError::LengthExtent {
                level,
                actual: lengths.len(),
                expected: selected.owners.len(),
            });
        }
        let ranges = request_level_ranges(&self.levels, request, self.request_count)?;
        Ok(&lengths[ranges[level].units.clone()])
    }
}

/// A borrowed integer slice whose iterator subtracts one request-local base.
#[derive(Clone, Copy)]
pub struct RebasedI64Slice<'a> {
    values: &'a [i64],
    base: i64,
}

impl<'a> RebasedI64Slice<'a> {
    fn new(values: &'a [i64], base: i64) -> Self {
        Self { values, base }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<i64> {
        self.values.get(index).map(|value| *value - self.base)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = i64> + '_ {
        self.values.iter().map(|value| *value - self.base)
    }
}

/// One ownership level restricted to one request.
pub struct OwnershipLevelView<'a> {
    offsets: RebasedI64Slice<'a>,
    owners: RebasedI64Slice<'a>,
}

impl OwnershipLevelView<'_> {
    pub fn offsets(&self) -> RebasedI64Slice<'_> {
        self.offsets
    }

    pub fn owners(&self) -> RebasedI64Slice<'_> {
        self.owners
    }
}

/// Request-local ownership and its physical item span.
pub struct PackedRequestView<'a> {
    item_span: Range<usize>,
    levels: Vec<OwnershipLevelView<'a>>,
}

impl PackedRequestView<'_> {
    pub fn item_span(&self) -> Range<usize> {
        self.item_span.clone()
    }

    pub fn levels(&self) -> &[OwnershipLevelView<'_>] {
        &self.levels
    }
}

/// A packed payload plus the companion tensors that make it splittable.
pub struct PackedTensor {
    value: Arc<Value>,
    ownership: PackedOwnership,
}

impl PackedTensor {
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new(value: Value, ownership: PackedOwnership) -> Result<Self, BatchContractError> {
        let actual =
            axis_zero_extent(&value).map_err(|reason| BatchContractError::InvalidCompanion {
                component: "<ownership>".into(),
                value: "<payload>".into(),
                level: 0,
                kind: "payload",
                companion: "<payload>".into(),
                reason,
            })?;
        if actual != ownership.packed_extent {
            return Err(BatchContractError::PayloadExtent {
                actual,
                expected: ownership.packed_extent,
            });
        }
        Ok(Self {
            value: Arc::new(value),
            ownership,
        })
    }

    pub fn value(&self) -> &Value {
        &self.value
    }

    pub fn ownership(&self) -> &PackedOwnership {
        &self.ownership
    }

    /// Build a no-copy payload view with request-local ownership companions.
    pub fn request_view(&self, request: usize) -> Result<PackedValueView<'_>, BatchContractError> {
        let ownership = self.ownership.request_view(request)?;
        let span = ownership.item_span();
        let value = Value::axis0_view(Arc::clone(&self.value), span.start, span.end - span.start)
            .map_err(|error| BatchContractError::InvalidCompanion {
            component: "<ownership>".into(),
            value: "<payload>".into(),
            level: 0,
            kind: "payload",
            companion: "<payload>".into(),
            reason: error.to_string(),
        })?;
        Ok(PackedValueView { value, ownership })
    }
}

/// A no-copy payload slice and the ownership view whose borrow keeps its chain live.
pub struct PackedValueView<'a> {
    value: Value,
    ownership: PackedRequestView<'a>,
}

impl PackedValueView<'_> {
    pub fn value(&self) -> &Value {
        &self.value
    }

    pub fn ownership(&self) -> &PackedRequestView<'_> {
        &self.ownership
    }
}

/// Owned companions produced by composing request-local chains in row order.
pub struct ComposedOwnership {
    levels: Vec<ComposedOwnershipLevel>,
    packed_extent: usize,
}

struct ComposedOwnershipLevel {
    offsets: Vec<i64>,
    owners: Vec<i64>,
}

impl ComposedOwnership {
    pub fn compose(requests: &[&PackedOwnership]) -> Result<Self, BatchContractError> {
        let Some(first) = requests.first() else {
            return Err(BatchContractError::EmptyComposition);
        };
        let depth = first.depth();
        let mut levels = (0..depth)
            .map(|_| ComposedOwnershipLevel {
                offsets: vec![0],
                owners: Vec::new(),
            })
            .collect::<Vec<_>>();
        let mut packed_extent = 0usize;
        let mut unit_bases = vec![0usize; depth];
        let mut parent_bases = vec![0usize; depth];

        for (index, request) in requests.iter().enumerate() {
            if request.depth() != depth {
                return Err(BatchContractError::CompositionDepth {
                    expected: depth,
                    actual: request.depth(),
                });
            }
            if request.request_count() != 1 {
                return Err(BatchContractError::CompositionNotLocal {
                    index,
                    request_count: request.request_count(),
                });
            }
            for (level_index, source) in request.levels.iter().enumerate() {
                let unit_base = unit_bases[level_index];
                let parent_base = parent_bases[level_index];
                for offset in &source.offsets[1..] {
                    let local = usize::try_from(*offset).map_err(|_| {
                        BatchContractError::ArithmeticOverflow {
                            operation: "converting a local ownership offset",
                        }
                    })?;
                    let rebased = unit_base.checked_add(local).ok_or(
                        BatchContractError::ArithmeticOverflow {
                            operation: "rebasing ownership offsets",
                        },
                    )?;
                    levels[level_index]
                        .offsets
                        .push(i64::try_from(rebased).map_err(|_| {
                            BatchContractError::ArithmeticOverflow {
                                operation: "storing a rebased ownership offset",
                            }
                        })?);
                }
                for &owner in &source.owners {
                    let local = usize::try_from(owner).map_err(|_| {
                        BatchContractError::ArithmeticOverflow {
                            operation: "converting a local owner index",
                        }
                    })?;
                    let rebased = parent_base.checked_add(local).ok_or(
                        BatchContractError::ArithmeticOverflow {
                            operation: "rebasing owner indices",
                        },
                    )?;
                    levels[level_index]
                        .owners
                        .push(i64::try_from(rebased).map_err(|_| {
                            BatchContractError::ArithmeticOverflow {
                                operation: "storing a rebased owner index",
                            }
                        })?);
                }
                unit_bases[level_index] = unit_bases[level_index]
                    .checked_add(source.owners.len())
                    .ok_or(BatchContractError::ArithmeticOverflow {
                        operation: "accumulating ownership units",
                    })?;
                parent_bases[level_index] = parent_bases[level_index]
                    .checked_add(source.offsets.len() - 1)
                    .ok_or(BatchContractError::ArithmeticOverflow {
                        operation: "accumulating ownership parents",
                    })?;
            }
            packed_extent = packed_extent.checked_add(request.packed_extent()).ok_or(
                BatchContractError::ArithmeticOverflow {
                    operation: "accumulating packed payload items",
                },
            )?;
        }

        Ok(Self {
            levels,
            packed_extent,
        })
    }

    pub fn packed_extent(&self) -> usize {
        self.packed_extent
    }

    pub fn request_count(&self) -> usize {
        self.levels
            .last()
            .map_or(0, |level| level.offsets.len() - 1)
    }

    pub fn offsets(&self, level: usize) -> Option<&[i64]> {
        self.levels.get(level).map(|level| level.offsets.as_slice())
    }

    pub fn owners(&self, level: usize) -> Option<&[i64]> {
        self.levels.get(level).map(|level| level.owners.as_slice())
    }

    /// Attach an already-concatenated payload without copying its bytes.
    ///
    /// Composition allocates only the small companion vectors. Concatenating
    /// disjoint request payload allocations remains the caller/backend packer's
    /// responsibility; when the payload already has one owner, this consumes it
    /// and every request view aliases that allocation.
    pub fn attach(self, payload: Value) -> Result<PackedTensor, BatchContractError> {
        let request_count = self.request_count();
        let levels = self
            .levels
            .into_iter()
            .map(|level| {
                let offsets_shape = [i64::try_from(level.offsets.len()).map_err(|_| {
                    BatchContractError::ArithmeticOverflow {
                        operation: "storing the composed offset shape",
                    }
                })?];
                let owners_shape = [i64::try_from(level.owners.len()).map_err(|_| {
                    BatchContractError::ArithmeticOverflow {
                        operation: "storing the composed owner shape",
                    }
                })?];
                let offsets =
                    Value::from_vec_i64(level.offsets, &offsets_shape).map_err(|error| {
                        BatchContractError::InvalidCompanion {
                            component: "<composition>".into(),
                            value: "<packed>".into(),
                            level: 0,
                            kind: "offsets",
                            companion: "<composed>".into(),
                            reason: error.to_string(),
                        }
                    })?;
                let owners = Value::from_vec_i64(level.owners, &owners_shape).map_err(|error| {
                    BatchContractError::InvalidCompanion {
                        component: "<composition>".into(),
                        value: "<packed>".into(),
                        level: 0,
                        kind: "owner",
                        companion: "<composed>".into(),
                        reason: error.to_string(),
                    }
                })?;
                Ok(OwnershipLevelValues::new(offsets, owners))
            })
            .collect::<Result<Vec<_>, BatchContractError>>()?;
        PackedTensor::new(
            payload,
            PackedOwnership::new(self.packed_extent, levels, request_count)?,
        )
    }
}

struct OwnedOwnershipLevel {
    offsets: Vec<i64>,
    owners: Vec<i64>,
}

struct LevelRange {
    parents: Range<usize>,
    units: Range<usize>,
}

fn request_level_ranges(
    levels: &[OwnedOwnershipLevel],
    request: usize,
    request_count: usize,
) -> Result<Vec<LevelRange>, BatchContractError> {
    if request >= request_count {
        return Err(BatchContractError::RequestIndex {
            request,
            request_count,
        });
    }
    let mut ranges = (0..levels.len())
        .map(|_| LevelRange {
            parents: 0..0,
            units: 0..0,
        })
        .collect::<Vec<_>>();
    let mut parents = request..request + 1;
    for level_index in (0..levels.len()).rev() {
        let level = &levels[level_index];
        let start = usize::try_from(level.offsets[parents.start]).map_err(|_| {
            BatchContractError::ArithmeticOverflow {
                operation: "reading a request-local ownership start",
            }
        })?;
        let end = usize::try_from(level.offsets[parents.end]).map_err(|_| {
            BatchContractError::ArithmeticOverflow {
                operation: "reading a request-local ownership end",
            }
        })?;
        ranges[level_index] = LevelRange {
            parents: parents.clone(),
            units: start..end,
        };
        parents = start..end;
    }
    Ok(ranges)
}

fn validate_ownership_slices(
    component: &str,
    value: &str,
    packed_extent: usize,
    levels: &[OwnedOwnershipLevel],
    expected_requests: Option<usize>,
) -> Result<usize, BatchContractError> {
    if levels.is_empty() || levels.len() > MAX_OWNERSHIP_DEPTH {
        return Err(BatchContractError::OwnershipDepth {
            component: component.into(),
            value: value.into(),
            depth: levels.len(),
            max_depth: MAX_OWNERSHIP_DEPTH,
        });
    }
    let mut expected_units = packed_extent;
    for (level_index, level) in levels.iter().enumerate() {
        if level.offsets.is_empty() {
            return Err(BatchContractError::InvalidOffset {
                component: component.into(),
                value: value.into(),
                level: level_index,
                index: 0,
                offset: -1,
            });
        }
        let mut previous = 0usize;
        for (index, &offset) in level.offsets.iter().enumerate() {
            let converted =
                usize::try_from(offset).map_err(|_| BatchContractError::InvalidOffset {
                    component: component.into(),
                    value: value.into(),
                    level: level_index,
                    index,
                    offset,
                })?;
            if (index == 0 && converted != 0) || converted < previous {
                return Err(BatchContractError::InvalidOffset {
                    component: component.into(),
                    value: value.into(),
                    level: level_index,
                    index,
                    offset,
                });
            }
            previous = converted;
        }
        let terminal =
            usize::try_from(*level.offsets.last().expect("checked non-empty")).map_err(|_| {
                BatchContractError::InvalidOffset {
                    component: component.into(),
                    value: value.into(),
                    level: level_index,
                    index: level.offsets.len() - 1,
                    offset: *level.offsets.last().expect("checked non-empty"),
                }
            })?;
        if terminal != expected_units {
            return Err(BatchContractError::TerminalExtent {
                component: component.into(),
                value: value.into(),
                level: level_index,
                actual: terminal,
                expected: expected_units,
            });
        }
        if level.owners.len() != expected_units {
            return Err(BatchContractError::OwnerExtent {
                component: component.into(),
                value: value.into(),
                level: level_index,
                actual: level.owners.len(),
                expected: expected_units,
            });
        }
        let parent_count = level.offsets.len() - 1;
        for parent in 0..parent_count {
            let start = usize::try_from(level.offsets[parent]).expect("validated offset");
            let end = usize::try_from(level.offsets[parent + 1]).expect("validated offset");
            for unit in start..end {
                let owner = level.owners[unit];
                let actual =
                    usize::try_from(owner).map_err(|_| BatchContractError::OwnerOutsideLevel {
                        component: component.into(),
                        value: value.into(),
                        level: level_index,
                        unit,
                        owner,
                        parent_count,
                    })?;
                if actual >= parent_count {
                    return Err(BatchContractError::OwnerOutsideLevel {
                        component: component.into(),
                        value: value.into(),
                        level: level_index,
                        unit,
                        owner,
                        parent_count,
                    });
                }
                if actual != parent {
                    return Err(BatchContractError::OwnerOrder {
                        component: component.into(),
                        value: value.into(),
                        level: level_index,
                        unit,
                        actual,
                        expected: parent,
                    });
                }
            }
        }
        expected_units = parent_count;
    }
    if let Some(expected) = expected_requests
        && expected_units != expected
    {
        return Err(BatchContractError::RequestExtent {
            component: component.into(),
            value: value.into(),
            actual: expected_units,
            expected,
        });
    }
    Ok(expected_units)
}

fn companion_values(
    component: &str,
    value: &str,
    level: usize,
    kind: &'static str,
    companion: &str,
    tensor: &Value,
) -> Result<Vec<i64>, BatchContractError> {
    let invalid = |reason: String| BatchContractError::InvalidCompanion {
        component: component.into(),
        value: value.into(),
        level,
        kind,
        companion: companion.into(),
        reason,
    };
    if tensor.dtype() != DataType::Int64 || tensor.shape().len() != 1 {
        return Err(invalid(format!(
            "got {:?} shape {:?}",
            tensor.dtype(),
            tensor.shape()
        )));
    }
    let host_resident = tensor
        .is_host_resident()
        .map_err(|error| invalid(error.to_string()))?;
    snapshot_companion_i64(tensor, host_resident, |tensor| {
        let device_id = tensor.device_id().map_err(|error| error.to_string())?;
        tensor.to_host_from_cuda(device_id).map_err(|error| {
            format!(
                "failed to stage device-resident ownership metadata from CUDA device \
                 {device_id}: {error}"
            )
        })
    })
    .map_err(invalid)
}

fn snapshot_companion_i64(
    tensor: &Value,
    host_resident: bool,
    stage_to_host: impl FnOnce(&Value) -> Result<Value, String>,
) -> Result<Vec<i64>, String> {
    let staged = (!host_resident)
        .then(|| stage_to_host(tensor))
        .transpose()?;
    staged
        .as_ref()
        .unwrap_or(tensor)
        .to_vec_i64()
        .map_err(|error| error.to_string())
}

fn axis_zero_extent(value: &Value) -> Result<usize, String> {
    value
        .shape()
        .first()
        .copied()
        .ok_or_else(|| "packed value must have rank at least one".to_string())
        .and_then(|extent| {
            usize::try_from(extent)
                .map_err(|_| format!("packed axis extent must be non-negative, got {extent}"))
        })
}

pub(crate) fn validate_workflow_batch_inputs(
    workflow: &WorkflowSpec,
    values: &PipelineTensors,
) -> Result<usize, BatchContractError> {
    let mut request_count = None;
    for (name, input) in &workflow.inputs {
        let Some(payload) = values.get(name) else {
            // Hosted executors may intentionally own all uses of an input, in
            // which case the generic plan does not bind it.
            continue;
        };
        let count = match &input.contract.batch_layout {
            BatchLayout::RequestAligned { axis } => {
                request_axis_count("<workflow>", name, payload, *axis, 1)?
            }
            BatchLayout::RequestExpanded { axis, factor } => {
                request_axis_count("<workflow>", name, payload, *axis, *factor)?
            }
            BatchLayout::TokenPacked { axis, levels } => validate_packed_contract(
                "<workflow>",
                name,
                *axis,
                levels,
                payload,
                |companion| values.get(companion),
                None,
            )?,
            BatchLayout::Shared | BatchLayout::RuntimeSequenceState => continue,
        };
        merge_request_count("<workflow>", &mut request_count, name, count)?;
    }
    Ok(request_count.map_or(1, |(_, count)| count))
}

fn validate_packed_contract<'a>(
    component: &str,
    value: &str,
    axis: usize,
    levels: &[onnx_genai_metadata::OwnershipLevel],
    payload: &Value,
    lookup: impl Fn(&str) -> Option<&'a Value>,
    expected_requests: Option<usize>,
) -> Result<usize, BatchContractError> {
    if axis != 0 {
        return Err(BatchContractError::PackedAxis {
            component: component.into(),
            value: value.into(),
            axis,
        });
    }
    if levels.is_empty() || levels.len() > MAX_OWNERSHIP_DEPTH {
        return Err(BatchContractError::OwnershipDepth {
            component: component.into(),
            value: value.into(),
            depth: levels.len(),
            max_depth: MAX_OWNERSHIP_DEPTH,
        });
    }
    let borrowed = levels
        .iter()
        .enumerate()
        .map(|(level, declaration)| {
            let offsets = lookup(&declaration.offsets).ok_or_else(|| {
                BatchContractError::MissingCompanion {
                    component: component.into(),
                    value: value.into(),
                    level,
                    kind: "offsets",
                    companion: declaration.offsets.clone(),
                }
            })?;
            let owners =
                lookup(&declaration.owner).ok_or_else(|| BatchContractError::MissingCompanion {
                    component: component.into(),
                    value: value.into(),
                    level,
                    kind: "owner",
                    companion: declaration.owner.clone(),
                })?;
            Ok(OwnedOwnershipLevel {
                offsets: companion_values(
                    component,
                    value,
                    level,
                    "offsets",
                    &declaration.offsets,
                    offsets,
                )?,
                owners: companion_values(
                    component,
                    value,
                    level,
                    "owner",
                    &declaration.owner,
                    owners,
                )?,
            })
        })
        .collect::<Result<Vec<_>, BatchContractError>>()?;
    let packed_extent =
        axis_zero_extent(payload).map_err(|reason| BatchContractError::InvalidCompanion {
            component: component.into(),
            value: value.into(),
            level: 0,
            kind: "payload",
            companion: value.into(),
            reason,
        })?;
    validate_ownership_slices(
        component,
        value,
        packed_extent,
        &borrowed,
        expected_requests,
    )
}

pub(crate) fn validate_component_batch_before_enqueue(
    component: &str,
    declaration: &WorkflowComponent,
    inputs: &[(&str, &Value)],
    symbols: &HashMap<String, i64>,
) -> Result<usize, BatchContractError> {
    validate_component_batch_before_enqueue_impl(
        component,
        declaration,
        inputs,
        symbols,
        false,
        None,
    )
}

pub(crate) fn validate_workflow_component_batch_before_enqueue(
    component: &str,
    declaration: &WorkflowComponent,
    inputs: &[(&str, &Value)],
    symbols: &HashMap<String, i64>,
    expected_requests: usize,
    allow_runtime_owned_inputs: bool,
) -> Result<usize, BatchContractError> {
    validate_component_batch_before_enqueue_impl(
        component,
        declaration,
        inputs,
        symbols,
        allow_runtime_owned_inputs,
        Some(expected_requests),
    )
}

fn validate_component_batch_before_enqueue_impl(
    component: &str,
    declaration: &WorkflowComponent,
    inputs: &[(&str, &Value)],
    symbols: &HashMap<String, i64>,
    allow_runtime_owned_inputs: bool,
    expected_requests: Option<usize>,
) -> Result<usize, BatchContractError> {
    let by_port = inputs.iter().copied().collect::<HashMap<_, _>>();
    let mut request_count: Option<(String, usize)> = None;
    for (port, contract) in &declaration.ports.inputs {
        let Some(value) = by_port.get(port.as_str()).copied() else {
            let requires_runtime_admission = declaration.batch_capacity.is_some()
                || matches!(
                    contract.batch_layout,
                    BatchLayout::TokenPacked { .. } | BatchLayout::RequestExpanded { .. }
                );
            if !allow_runtime_owned_inputs && !contract.optional && requires_runtime_admission {
                return Err(BatchContractError::MissingBatchInput {
                    component: component.into(),
                    port: port.clone(),
                });
            }
            continue;
        };
        let count = match &contract.batch_layout {
            BatchLayout::RequestAligned { axis } => {
                request_axis_count(component, port, value, *axis, 1)?
            }
            BatchLayout::RequestExpanded { axis, factor } => {
                request_axis_count(component, port, value, *axis, *factor)?
            }
            BatchLayout::TokenPacked { axis, levels } => validate_packed_contract(
                component,
                port,
                *axis,
                levels,
                value,
                |companion| by_port.get(companion).copied(),
                None,
            )?,
            BatchLayout::Shared | BatchLayout::RuntimeSequenceState => continue,
        };
        merge_request_count(component, &mut request_count, port, count)?;
    }
    if let Some(expected_requests) = expected_requests {
        merge_request_count(
            component,
            &mut request_count,
            "<workflow>",
            expected_requests,
        )?;
    }
    let count = request_count.as_ref().map_or(1, |(_, count)| *count);
    if declaration.batch_capacity.is_none() && count > 1 {
        return Err(BatchContractError::UndeclaredCapacity {
            component: component.into(),
            request_count: count,
        });
    }
    let policy = component_batch_policy(component, declaration)?;
    let dimensions =
        component_materialized_dimensions(component, declaration, inputs, symbols, &policy)?;
    validate_materialized_footprint(&policy, &dimensions).map_err(|source| {
        BatchContractError::Admission {
            component: component.into(),
            source,
        }
    })?;
    Ok(count)
}

pub(crate) fn validate_component_outputs_before_publish(
    component: &str,
    declaration: &WorkflowComponent,
    inputs: &[(&str, &Value)],
    outputs: &[(&str, &Value)],
    expected_requests: usize,
) -> Result<(), BatchContractError> {
    let input_by_port = inputs.iter().copied().collect::<HashMap<_, _>>();
    let output_by_port = outputs.iter().copied().collect::<HashMap<_, _>>();
    let mut request_count = Some(("<inputs>".to_string(), expected_requests));
    for (port, contract) in &declaration.ports.outputs {
        let Some(value) = output_by_port.get(port.as_str()).copied() else {
            continue;
        };
        let count = match &contract.batch_layout {
            BatchLayout::RequestAligned { axis } => {
                request_axis_count(component, port, value, *axis, 1)?
            }
            BatchLayout::RequestExpanded { axis, factor } => {
                request_axis_count(component, port, value, *axis, *factor)?
            }
            BatchLayout::TokenPacked { axis, levels } => validate_packed_contract(
                component,
                port,
                *axis,
                levels,
                value,
                |companion| {
                    output_by_port
                        .get(companion)
                        .copied()
                        .or_else(|| input_by_port.get(companion).copied())
                },
                Some(expected_requests),
            )?,
            BatchLayout::Shared | BatchLayout::RuntimeSequenceState => continue,
        };
        merge_request_count(component, &mut request_count, port, count)?;
    }
    Ok(())
}

fn component_materialized_dimensions(
    component: &str,
    declaration: &WorkflowComponent,
    inputs: &[(&str, &Value)],
    symbols: &HashMap<String, i64>,
    policy: &BatchPolicy,
) -> Result<BTreeMap<String, usize>, BatchContractError> {
    let by_port = inputs.iter().copied().collect::<HashMap<_, _>>();
    policy
        .aggregations
        .keys()
        .map(|symbol| {
            let logical = symbols.get(symbol).copied().ok_or_else(|| {
                BatchContractError::MissingCapacityDimension {
                    component: component.into(),
                    symbol: symbol.clone(),
                }
            })?;
            let mut materialized = usize::try_from(logical).map_err(|_| {
                BatchContractError::MissingCapacityDimension {
                    component: component.into(),
                    symbol: symbol.clone(),
                }
            })?;
            for (port, contract) in &declaration.ports.inputs {
                let Some(value) = by_port.get(port.as_str()).copied() else {
                    continue;
                };
                let Some(shape) = &contract.shape else {
                    continue;
                };
                for (axis, dimension) in shape.iter().enumerate() {
                    if dimension != &TensorDimension::Symbol(symbol.clone()) {
                        continue;
                    }
                    let actual = value.shape().get(axis).copied().ok_or_else(|| {
                        BatchContractError::MissingCapacityDimension {
                            component: component.into(),
                            symbol: symbol.clone(),
                        }
                    })?;
                    materialized = materialized.max(usize::try_from(actual).map_err(|_| {
                        BatchContractError::MissingCapacityDimension {
                            component: component.into(),
                            symbol: symbol.clone(),
                        }
                    })?);
                }
            }
            Ok((symbol.clone(), materialized))
        })
        .collect()
}

fn request_axis_count(
    component: &str,
    port: &str,
    value: &Value,
    axis: usize,
    factor: usize,
) -> Result<usize, BatchContractError> {
    let extent = value
        .shape()
        .get(axis)
        .copied()
        .and_then(|extent| usize::try_from(extent).ok())
        .ok_or_else(|| BatchContractError::MissingCapacityDimension {
            component: component.into(),
            symbol: format!("{port}[axis {axis}]"),
        })?;
    if factor == 0 || extent % factor != 0 {
        return Err(BatchContractError::MissingCapacityDimension {
            component: component.into(),
            symbol: format!("{port}[axis {axis}] / expansion {factor}"),
        });
    }
    Ok(extent / factor)
}

fn merge_request_count(
    component: &str,
    current: &mut Option<(String, usize)>,
    value: &str,
    count: usize,
) -> Result<(), BatchContractError> {
    match current {
        Some((first_value, first_count)) if *first_count != count => {
            Err(BatchContractError::RequestCountMismatch {
                component: component.into(),
                first_value: first_value.clone(),
                first_count: *first_count,
                value: value.into(),
                count,
            })
        }
        Some(_) => Ok(()),
        None => {
            *current = Some((value.into(), count));
            Ok(())
        }
    }
}

pub(crate) fn component_ownership_companions(declaration: &WorkflowComponent) -> BTreeSet<String> {
    declaration
        .ports
        .inputs
        .values()
        .chain(declaration.ports.outputs.values())
        .flat_map(|contract| contract.batch_layout.companions())
        .map(|(_, _, companion)| companion.to_string())
        .collect()
}

pub(crate) fn component_output_ownership_companions(
    declaration: &WorkflowComponent,
) -> BTreeSet<String> {
    component_ownership_companions(declaration)
        .into_iter()
        .filter(|companion| declaration.ports.outputs.contains_key(companion))
        .collect()
}

fn component_batch_policy(
    component: &str,
    declaration: &WorkflowComponent,
) -> Result<BatchPolicy, BatchContractError> {
    let Some(capacity) = &declaration.batch_capacity else {
        return Ok(BatchPolicy {
            max_contributions: Some(1),
            ..BatchPolicy::default()
        });
    };
    let required = capacity
        .uniform_dimensions
        .iter()
        .cloned()
        .chain(
            capacity
                .budgets
                .iter()
                .flat_map(|budget| budget.dimensions.iter().cloned()),
        )
        .collect::<BTreeSet<_>>();
    let mut companion_ports = BTreeSet::new();
    let mut summed_symbols = BTreeSet::new();
    for contract in declaration.ports.inputs.values() {
        if let Some(shape) = &contract.shape {
            for (axis, dimension) in shape.iter().enumerate() {
                if (contract.batch_layout.request_axis() == Some(axis)
                    || contract.batch_layout.packed_axis() == Some(axis))
                    && let TensorDimension::Symbol(symbol) = dimension
                {
                    summed_symbols.insert(symbol.clone());
                }
            }
        }
        for (_, kind, companion) in contract.batch_layout.companions() {
            companion_ports.insert(companion.to_string());
            if kind == "owner"
                && let Some(TensorDimension::Symbol(symbol)) = declaration
                    .ports
                    .inputs
                    .get(companion)
                    .and_then(|owner| owner.shape.as_ref())
                    .and_then(|shape| shape.first())
            {
                // Every owner entry is one unit at this ownership level.
                // Higher-level owner tensors are the only typed place a nested
                // count such as `clips` appears, so classify that symbol as a
                // summed group count rather than treating the companion axis
                // as an invariant tensor dimension.
                summed_symbols.insert(symbol.clone());
            }
        }
        companion_ports.extend(
            contract
                .padding
                .iter()
                .map(|padding| padding.valid_lengths.clone()),
        );
    }
    let mut aggregations = BTreeMap::new();
    for symbol in required {
        let mut aggregation = summed_symbols
            .contains(&symbol)
            .then_some(BatchDimensionAggregation::Sum);
        for (port, contract) in &declaration.ports.inputs {
            if companion_ports.contains(port) {
                continue;
            }
            let Some(shape) = &contract.shape else {
                continue;
            };
            for (axis, dimension) in shape.iter().enumerate() {
                if dimension != &TensorDimension::Symbol(symbol.clone()) {
                    continue;
                }
                let candidate = if contract.batch_layout.request_axis() == Some(axis)
                    || contract.batch_layout.packed_axis() == Some(axis)
                {
                    BatchDimensionAggregation::Sum
                } else {
                    BatchDimensionAggregation::Maximum
                };
                if aggregation.is_some_and(|current| current != candidate) {
                    return Err(BatchContractError::AmbiguousDimension {
                        component: component.into(),
                        symbol,
                    });
                }
                aggregation = Some(candidate);
            }
        }
        aggregations.insert(
            symbol.clone(),
            aggregation.ok_or_else(|| BatchContractError::MissingCapacityDimension {
                component: component.into(),
                symbol,
            })?,
        );
    }
    Ok(BatchPolicy {
        max_contributions: None,
        aggregations,
        uniform_dimensions: capacity.uniform_dimensions.iter().cloned().collect(),
        budgets: capacity
            .budgets
            .iter()
            .map(|budget| MaterializedBudget {
                dimensions: budget.dimensions.clone(),
                max_total: budget.max_total,
            })
            .collect(),
    })
}

impl WorkflowRuntime {
    pub(crate) fn group_component_batch_inputs(
        &self,
        component: &str,
        requests: &[(SessionId, &PipelineTensors)],
    ) -> anyhow::Result<Vec<AdmittedBatch>> {
        let declaration = self
            .plan
            .workflow
            .components
            .get(component)
            .ok_or_else(|| anyhow::anyhow!("workflow component '{component}' is undeclared"))?;
        let policy = component_batch_policy(component, declaration)?;
        let mut contributions = Vec::with_capacity(requests.len());
        for (sequence_id, inputs) in requests {
            let mut symbols = HashMap::new();
            let dynamic = std::collections::HashSet::new();
            let mut resolved = Vec::new();
            for (port, contract) in &declaration.ports.inputs {
                let Some(value) = inputs.get(port) else {
                    if contract.optional {
                        continue;
                    }
                    return Err(anyhow::anyhow!(
                        "component '{component}' batch contribution for sequence {sequence_id} \
                         is missing input port '{port}'"
                    ));
                };
                super::workflow::validate_workflow_value(
                    port,
                    value,
                    contract,
                    &mut symbols,
                    &dynamic,
                )?;
                resolved.push((port.as_str(), value));
            }
            let request_count = validate_component_batch_before_enqueue(
                component,
                declaration,
                &resolved,
                &symbols,
            )?;
            if request_count != 1 {
                return Err(BatchContractError::ContributionNotRequestLocal {
                    component: component.into(),
                    sequence_id: *sequence_id,
                    request_count,
                }
                .into());
            }
            let dimensions = component_materialized_dimensions(
                component,
                declaration,
                &resolved,
                &symbols,
                &policy,
            )?;
            contributions.push(BatchContribution {
                sequence_id: *sequence_id,
                dimensions,
            });
        }
        group_batch_contributions(&policy, &contributions).map_err(|source| {
            BatchContractError::Admission {
                component: component.into(),
                source,
            }
            .into()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn level(offsets: &[i64], owners: &[i64]) -> OwnershipLevelValues {
        OwnershipLevelValues::new(
            Value::from_slice_i64(offsets, &[offsets.len() as i64]).unwrap(),
            Value::from_slice_i64(owners, &[owners.len() as i64]).unwrap(),
        )
    }

    fn ownership_error(result: Result<PackedOwnership, BatchContractError>) -> BatchContractError {
        match result {
            Ok(_) => panic!("malformed ownership must be rejected"),
            Err(error) => error,
        }
    }

    fn component() -> WorkflowComponent {
        serde_yaml::from_str(
            r#"
implementation: { kind: onnx, artifact: encoder.onnx }
batch_capacity:
  uniform_dimensions: [height]
  budgets:
    - { dimensions: [items], max_total: 5 }
ports:
  inputs:
    pixels:
      dtype: float32
      rank: 3
      shape: [items, channels, height]
      batch_layout:
        kind: token_packed
        axis: 0
        levels:
          - { offsets: offsets, owner: owner }
    offsets:
      dtype: int64
      rank: 1
      shape: [rows_plus_one]
      batch_layout: { kind: shared }
    owner:
      dtype: int64
      rank: 1
      shape: [items]
      batch_layout: { kind: shared }
    prompt:
      dtype: int64
      rank: 2
      shape: [batch, sequence]
      batch_layout: { kind: request_aligned, axis: 0 }
  outputs: {}
"#,
        )
        .unwrap()
    }

    fn nested_component() -> WorkflowComponent {
        serde_yaml::from_str(
            r#"
implementation: { kind: onnx, artifact: encoder.onnx }
batch_capacity:
  uniform_dimensions: [height]
  budgets:
    - { dimensions: [clips], max_total: 8 }
ports:
  inputs:
    pixels:
      dtype: float32
      rank: 3
      shape: [frames, channels, height]
      batch_layout:
        kind: token_packed
        axis: 0
        levels:
          - { offsets: frame_offsets, owner: frame_owner }
          - { offsets: clip_offsets, owner: clip_owner }
    frame_offsets: { dtype: int64, rank: 1, shape: [clips_plus_one], batch_layout: { kind: shared } }
    frame_owner: { dtype: int64, rank: 1, shape: [frames], batch_layout: { kind: shared } }
    clip_offsets: { dtype: int64, rank: 1, shape: [rows_plus_one], batch_layout: { kind: shared } }
    clip_owner: { dtype: int64, rank: 1, shape: [clips], batch_layout: { kind: shared } }
  outputs: {}
"#,
        )
        .unwrap()
    }

    fn expanded_component() -> WorkflowComponent {
        serde_yaml::from_str(
            r#"
implementation: { kind: onnx, artifact: encoder.onnx }
batch_capacity:
  budgets:
    - { dimensions: [batch], max_total: 1 }
ports:
  inputs:
    rows:
      dtype: float32
      rank: 2
      shape: [batch, hidden]
      batch_layout: { kind: request_expanded, axis: 0, factor: 2 }
  outputs: {}
"#,
        )
        .unwrap()
    }

    #[test]
    fn flat_ownership_covers_empty_and_multiple_items() {
        let ownership = PackedOwnership::new(3, vec![level(&[0, 0, 3], &[1, 1, 1])], 2).unwrap();
        let empty = ownership.request_view(0).unwrap();
        assert_eq!(empty.item_span(), 0..0);
        assert_eq!(
            empty.levels()[0].offsets().iter().collect::<Vec<_>>(),
            vec![0, 0]
        );
        let items = ownership.request_view(1).unwrap();
        assert_eq!(items.item_span(), 0..3);
        assert_eq!(
            items.levels()[0].owners().iter().collect::<Vec<_>>(),
            vec![0, 0, 0]
        );
    }

    #[test]
    fn composition_preserves_an_empty_request_before_nonempty_items() {
        let empty = PackedOwnership::new(0, vec![level(&[0, 0], &[])], 1).unwrap();
        let nonempty = PackedOwnership::new(2, vec![level(&[0, 2], &[0, 0])], 1).unwrap();
        let composed = ComposedOwnership::compose(&[&empty, &nonempty]).unwrap();
        assert_eq!(composed.offsets(0).unwrap(), &[0, 0, 2]);
        assert_eq!(composed.owners(0).unwrap(), &[1, 1]);
    }

    #[test]
    fn nested_video_ownership_composes_in_exact_request_item_order() {
        let first = PackedOwnership::new(
            3,
            vec![level(&[0, 2, 3], &[0, 0, 1]), level(&[0, 2], &[0, 0])],
            1,
        )
        .unwrap();
        let second =
            PackedOwnership::new(3, vec![level(&[0, 3], &[0, 0, 0]), level(&[0, 1], &[0])], 1)
                .unwrap();
        let composed = ComposedOwnership::compose(&[&first, &second]).unwrap();
        assert_eq!(composed.offsets(0).unwrap(), &[0, 2, 3, 6]);
        assert_eq!(composed.owners(0).unwrap(), &[0, 0, 1, 2, 2, 2]);
        assert_eq!(composed.offsets(1).unwrap(), &[0, 2, 3]);
        assert_eq!(composed.owners(1).unwrap(), &[0, 0, 1]);

        let payload = Value::from_slice_i64(&[10, 11, 12, 20, 21, 22], &[6, 1]).unwrap();
        let payload_ptr = payload.data_ptr_addr().unwrap();
        let packed = composed.attach(payload).unwrap();
        let second_view = packed.request_view(1).unwrap();
        assert_eq!(second_view.value().to_vec_i64().unwrap(), vec![20, 21, 22]);
        assert_eq!(
            second_view.value().data_ptr_addr().unwrap(),
            payload_ptr + 3 * std::mem::size_of::<i64>()
        );
        assert_eq!(
            second_view.ownership().levels()[0]
                .offsets()
                .iter()
                .collect::<Vec<_>>(),
            vec![0, 3]
        );
        assert_eq!(
            second_view.ownership().levels()[1]
                .owners()
                .iter()
                .collect::<Vec<_>>(),
            vec![0]
        );
    }

    #[test]
    fn request_lengths_are_sliced_without_rebasing() {
        let ownership = PackedOwnership::new(4, vec![level(&[0, 2, 4], &[0, 0, 1, 1])], 2).unwrap();
        let lengths = [7, 5, 3, 1];
        assert_eq!(ownership.slice_lengths(1, 0, &lengths).unwrap(), &[3, 1]);
    }

    #[test]
    fn device_companion_path_stages_only_the_metadata_snapshot() {
        let source = Value::from_slice_i64(&[0, 2, 5], &[3]).unwrap();
        let staged = std::cell::Cell::new(false);
        let snapshot = snapshot_companion_i64(&source, false, |value| {
            staged.set(true);
            Value::from_vec_i64(value.to_vec_i64().map_err(|error| error.to_string())?, &[3])
                .map_err(|error| error.to_string())
        })
        .unwrap();

        assert!(staged.get());
        assert_eq!(snapshot, [0, 2, 5]);
        assert_eq!(source.to_vec_i64().unwrap(), [0, 2, 5]);
    }

    #[test]
    fn malformed_offsets_and_terminal_extents_fail_closed() {
        let non_monotonic =
            ownership_error(PackedOwnership::new(2, vec![level(&[0, 2, 1], &[0, 0])], 2));
        assert!(matches!(
            non_monotonic,
            BatchContractError::InvalidOffset { index: 2, .. }
        ));

        let terminal =
            ownership_error(PackedOwnership::new(3, vec![level(&[0, 2], &[0, 0, 0])], 1));
        assert!(matches!(
            terminal,
            BatchContractError::TerminalExtent {
                actual: 2,
                expected: 3,
                ..
            }
        ));
    }

    #[test]
    fn owners_must_stay_internal_and_match_contiguous_spans() {
        let outside = ownership_error(PackedOwnership::new(2, vec![level(&[0, 2], &[0, 1])], 1));
        assert!(matches!(
            outside,
            BatchContractError::OwnerOutsideLevel { unit: 1, .. }
        ));
        let reordered =
            ownership_error(PackedOwnership::new(2, vec![level(&[0, 1, 2], &[1, 0])], 2));
        assert!(matches!(
            reordered,
            BatchContractError::OwnerOrder { unit: 0, .. }
        ));
    }

    #[test]
    fn ownership_depth_is_bounded() {
        let error = ownership_error(PackedOwnership::new(
            1,
            vec![
                level(&[0, 1], &[0]),
                level(&[0, 1], &[0]),
                level(&[0, 1], &[0]),
            ],
            1,
        ));
        assert!(matches!(
            error,
            BatchContractError::OwnershipDepth {
                depth: 3,
                max_depth: 2,
                ..
            }
        ));
    }

    #[test]
    fn nested_owner_unit_symbols_are_summed_not_read_as_global_axes() {
        let policy = component_batch_policy("encoder", &nested_component()).unwrap();
        assert_eq!(policy.aggregations["clips"], BatchDimensionAggregation::Sum);
        assert_eq!(
            policy.aggregations["height"],
            BatchDimensionAggregation::Maximum
        );
    }

    #[test]
    fn component_budget_is_checked_at_the_backend_boundary() {
        let declaration = component();
        let offsets = Value::from_slice_i64(&[0, 5], &[2]).unwrap();
        let owners = Value::from_slice_i64(&[0; 5], &[5]).unwrap();
        let pixels = Value::from_slice_f32(&[0.0; 5 * 3 * 4], &[5, 3, 4]).unwrap();
        let prompt = Value::from_slice_i64(&[1, 2], &[1, 2]).unwrap();
        let inputs = [
            ("pixels", &pixels),
            ("offsets", &offsets),
            ("owner", &owners),
            ("prompt", &prompt),
        ];
        let symbols = HashMap::from([
            ("items".into(), 5),
            ("channels".into(), 3),
            ("height".into(), 4),
            ("batch".into(), 1),
            ("sequence".into(), 2),
            ("rows_plus_one".into(), 2),
        ]);
        assert_eq!(
            validate_component_batch_before_enqueue("encoder", &declaration, &inputs, &symbols)
                .unwrap(),
            1
        );

        let over_offsets = Value::from_slice_i64(&[0, 6], &[2]).unwrap();
        let over_owners = Value::from_slice_i64(&[0; 6], &[6]).unwrap();
        let over_pixels = Value::from_slice_f32(&[0.0; 6 * 3 * 4], &[6, 3, 4]).unwrap();
        let over_inputs = [
            ("pixels", &over_pixels),
            ("offsets", &over_offsets),
            ("owner", &over_owners),
            ("prompt", &prompt),
        ];
        let mut over_symbols = symbols;
        over_symbols.insert("items".into(), 6);
        let error = validate_component_batch_before_enqueue(
            "encoder",
            &declaration,
            &over_inputs,
            &over_symbols,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BatchContractError::Admission {
                source: BatchAdmissionError::MaterializedBudgetExceeded {
                    materialized: 6,
                    max_total: 5,
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn component_request_counts_cannot_cross_leak() {
        let declaration = component();
        let offsets = Value::from_slice_i64(&[0, 1, 2], &[3]).unwrap();
        let owners = Value::from_slice_i64(&[0, 1], &[2]).unwrap();
        let pixels = Value::from_slice_f32(&[0.0; 2 * 3 * 4], &[2, 3, 4]).unwrap();
        let prompt = Value::from_slice_i64(&[1, 2], &[1, 2]).unwrap();
        let inputs = [
            ("pixels", &pixels),
            ("offsets", &offsets),
            ("owner", &owners),
            ("prompt", &prompt),
        ];
        let symbols = HashMap::from([
            ("items".into(), 2),
            ("channels".into(), 3),
            ("height".into(), 4),
            ("batch".into(), 1),
            ("sequence".into(), 2),
            ("rows_plus_one".into(), 3),
        ]);
        let error =
            validate_component_batch_before_enqueue("encoder", &declaration, &inputs, &symbols)
                .unwrap_err();
        assert!(matches!(
            error,
            BatchContractError::RequestCountMismatch {
                first_count: 2,
                count: 1,
                ..
            }
        ));
    }

    #[test]
    fn undeclared_capacity_never_admits_multiple_requests() {
        let mut declaration = component();
        declaration.batch_capacity = None;
        let offsets = Value::from_slice_i64(&[0, 1, 2], &[3]).unwrap();
        let owners = Value::from_slice_i64(&[0, 1], &[2]).unwrap();
        let pixels = Value::from_slice_f32(&[0.0; 2 * 3 * 4], &[2, 3, 4]).unwrap();
        let prompt = Value::from_slice_i64(&[1, 2, 3, 4], &[2, 2]).unwrap();
        let inputs = [
            ("pixels", &pixels),
            ("offsets", &offsets),
            ("owner", &owners),
            ("prompt", &prompt),
        ];
        let symbols = HashMap::from([
            ("items".into(), 2),
            ("channels".into(), 3),
            ("height".into(), 4),
            ("batch".into(), 2),
            ("sequence".into(), 2),
            ("rows_plus_one".into(), 3),
        ]);
        let error =
            validate_component_batch_before_enqueue("encoder", &declaration, &inputs, &symbols)
                .unwrap_err();
        assert_eq!(
            error,
            BatchContractError::UndeclaredCapacity {
                component: "encoder".into(),
                request_count: 2,
            }
        );
    }

    #[test]
    fn hosted_admission_cannot_hide_a_known_request_count() {
        let declaration: WorkflowComponent = serde_yaml::from_str(
            r#"
implementation: { kind: binding }
ports:
  inputs:
    rows:
      dtype: float32
      rank: 2
      shape: [batch, hidden]
      batch_layout: { kind: request_aligned, axis: 0 }
  outputs: {}
"#,
        )
        .unwrap();
        let rows = Value::from_slice_f32(&[0.0; 2 * 3], &[2, 3]).unwrap();
        let symbols = HashMap::from([("batch".into(), 2), ("hidden".into(), 3)]);
        let error = validate_workflow_component_batch_before_enqueue(
            "hosted",
            &declaration,
            &[("rows", &rows)],
            &symbols,
            2,
            true,
        )
        .unwrap_err();
        assert_eq!(
            error,
            BatchContractError::UndeclaredCapacity {
                component: "hosted".into(),
                request_count: 2,
            }
        );
    }

    #[test]
    fn expanded_requests_charge_physical_rows_to_budgets() {
        let declaration = expanded_component();
        let rows = Value::from_slice_f32(&[0.0; 2 * 3], &[2, 3]).unwrap();
        let symbols = HashMap::from([("batch".into(), 1), ("hidden".into(), 3)]);
        let error = validate_component_batch_before_enqueue(
            "encoder",
            &declaration,
            &[("rows", &rows)],
            &symbols,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BatchContractError::Admission {
                source: BatchAdmissionError::MaterializedBudgetExceeded {
                    materialized: 2,
                    max_total: 1,
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn workflow_inputs_must_share_one_logical_request_cardinality() {
        let workflow: WorkflowSpec = serde_yaml::from_str(
            r#"
manifest: { capabilities: [] }
inputs:
  packed:
    contract:
      dtype: float32
      rank: 2
      shape: [items, hidden]
      batch_layout:
        kind: token_packed
        axis: 0
        levels:
          - { offsets: offsets, owner: owner }
    role: { kind: opaque }
    source: { kind: application, name: packed }
  offsets:
    contract: { dtype: int64, rank: 1, shape: [rows_plus_one], batch_layout: { kind: shared } }
    role: { kind: opaque }
    source: { kind: application, name: offsets }
  owner:
    contract: { dtype: int64, rank: 1, shape: [items], batch_layout: { kind: shared } }
    role: { kind: opaque }
    source: { kind: application, name: owner }
  prompt:
    contract:
      dtype: int64
      rank: 2
      shape: [batch, sequence]
      batch_layout: { kind: request_aligned, axis: 0 }
    role: { kind: opaque }
    source: { kind: application, name: prompt }
outputs: {}
components: {}
steps: []
"#,
        )
        .unwrap();
        let values = PipelineTensors::from([
            (
                "packed".into(),
                Value::from_slice_f32(&[1.0, 2.0], &[2, 1]).unwrap(),
            ),
            (
                "offsets".into(),
                Value::from_slice_i64(&[0, 1, 2], &[3]).unwrap(),
            ),
            (
                "owner".into(),
                Value::from_slice_i64(&[0, 1], &[2]).unwrap(),
            ),
            (
                "prompt".into(),
                Value::from_slice_i64(&[7, 8], &[1, 2]).unwrap(),
            ),
        ]);
        let error = validate_workflow_batch_inputs(&workflow, &values).unwrap_err();
        assert!(matches!(
            error,
            BatchContractError::RequestCountMismatch {
                first_count: 2,
                count: 1,
                ..
            } | BatchContractError::RequestCountMismatch {
                first_count: 1,
                count: 2,
                ..
            }
        ));
    }
}
