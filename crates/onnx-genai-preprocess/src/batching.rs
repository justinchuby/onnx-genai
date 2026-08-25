//! Schema-driven encoder grouping and packed ownership bookkeeping.

use std::{collections::BTreeMap, error::Error, fmt};

use onnx_genai_metadata::{BatchLayout, ComponentBatchCapacity, TensorContract, TensorDimension};

/// One pending encoder contribution described by its authored component contract.
///
/// `dimensions` contains the concrete symbol extents for this contribution. The
/// planner never infers semantics from symbol spellings: compatibility comes
/// from `batch_capacity`, `padding`, and `batch_layout`.
#[derive(Debug, Clone, Copy)]
pub struct EncoderWorkItem<'a> {
    pub request_index: usize,
    pub item_index: usize,
    pub component: &'a str,
    pub port: &'a str,
    pub contract: &'a TensorContract,
    pub capacity: Option<&'a ComponentBatchCapacity>,
    pub dimensions: &'a BTreeMap<String, usize>,
}

/// Stable indices into the caller's work-item array for one compatible invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderGroup {
    pub item_indices: Vec<usize>,
}

/// One ownership level over the single physically packed axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedOwnershipLevel {
    pub offsets: Vec<i64>,
    pub owner: Vec<i64>,
}

/// Validated ownership levels, innermost first, for a packed encoder value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedOwnership {
    request_count: usize,
    physical_count: usize,
    levels: Vec<PackedOwnershipLevel>,
}

/// Contiguous group-local ranges belonging to one request row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestSpan {
    pub request_index: usize,
    pub item_offset: usize,
    pub item_length: usize,
    pub physical_offset: usize,
    pub physical_length: usize,
}

/// One request's ownership chain rebased to request-local offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestOwnership {
    pub span: RequestSpan,
    pub levels: Vec<PackedOwnershipLevel>,
}

/// Typed, actionable failures at the grouped preprocessing boundary.
#[derive(Debug)]
pub enum EncoderBatchingError {
    MissingDimension {
        component: String,
        port: String,
        request_index: usize,
        item_index: usize,
        dimension: String,
        declared_by: String,
    },
    BudgetOverflow {
        component: String,
        port: String,
        dimensions: Vec<String>,
    },
    SingleItemExceedsBudget {
        component: String,
        port: String,
        request_index: usize,
        item_index: usize,
        dimensions: Vec<String>,
        materialized: usize,
        max_total: usize,
    },
    Ownership {
        detail: String,
    },
    UnsupportedExecution {
        component: String,
        detail: String,
    },
    Preprocessing {
        detail: String,
    },
}

impl fmt::Display for EncoderBatchingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingDimension {
                component,
                port,
                request_index,
                item_index,
                dimension,
                declared_by,
            } => write!(
                formatter,
                "encoder component '{component}' port '{port}' request {request_index} item \
                 {item_index} does not provide concrete extent '{dimension}' required by \
                 {declared_by}"
            ),
            Self::BudgetOverflow {
                component,
                port,
                dimensions,
            } => write!(
                formatter,
                "encoder component '{component}' port '{port}' budget {dimensions:?} overflowed \
                 while computing its materialized footprint"
            ),
            Self::SingleItemExceedsBudget {
                component,
                port,
                request_index,
                item_index,
                dimensions,
                materialized,
                max_total,
            } => write!(
                formatter,
                "encoder component '{component}' port '{port}' request {request_index} item \
                 {item_index} materializes {materialized} across budget {dimensions:?}, exceeding \
                 declared max_total {max_total}; the item cannot be split into a valid invocation"
            ),
            Self::Ownership { detail } => write!(formatter, "invalid packed ownership: {detail}"),
            Self::UnsupportedExecution { component, detail } => write!(
                formatter,
                "grouped preprocessing for component '{component}' is unsupported: {detail}"
            ),
            Self::Preprocessing { detail } => {
                write!(formatter, "grouped media preprocessing failed: {detail}")
            }
        }
    }
}

impl Error for EncoderBatchingError {}

/// Forms deterministic, stable groups under authored contracts and capacity bounds.
///
/// Groups are returned in first-seen order and items retain their input order
/// inside each group. A component without `batch_capacity` always produces
/// singleton groups.
pub fn plan_encoder_groups(
    items: &[EncoderWorkItem<'_>],
) -> Result<Vec<EncoderGroup>, EncoderBatchingError> {
    let mut groups = Vec::<EncoderGroup>::new();
    for (index, item) in items.iter().enumerate() {
        validate_item_dimensions(item)?;
        if item.capacity.is_none() {
            groups.push(EncoderGroup {
                item_indices: vec![index],
            });
            continue;
        }
        ensure_single_item_within_budgets(item)?;

        let mut selected = None;
        for (group_index, group) in groups.iter().enumerate() {
            let first = &items[group.item_indices[0]];
            if !same_contract(first, item)
                || !dimensions_compatible(first, item)?
                || !group_with_item_fits(group, index, items)?
            {
                continue;
            }
            selected = Some(group_index);
            break;
        }
        match selected {
            Some(group_index) => groups[group_index].item_indices.push(index),
            None => groups.push(EncoderGroup {
                item_indices: vec![index],
            }),
        }
    }
    Ok(groups)
}

fn validate_item_dimensions(item: &EncoderWorkItem<'_>) -> Result<(), EncoderBatchingError> {
    if let Some(shape) = &item.contract.shape {
        for dimension in shape {
            if let TensorDimension::Symbol(symbol) = dimension {
                require_dimension(item, symbol, "the tensor contract shape")?;
            }
        }
    }
    let Some(capacity) = item.capacity else {
        return Ok(());
    };
    for symbol in &capacity.uniform_dimensions {
        require_dimension(item, symbol, "batch_capacity.uniform_dimensions")?;
    }
    for budget in &capacity.budgets {
        for symbol in &budget.dimensions {
            require_dimension(item, symbol, "batch_capacity.budgets")?;
        }
    }
    Ok(())
}

fn require_dimension(
    item: &EncoderWorkItem<'_>,
    dimension: &str,
    declared_by: &str,
) -> Result<usize, EncoderBatchingError> {
    item.dimensions
        .get(dimension)
        .copied()
        .ok_or_else(|| EncoderBatchingError::MissingDimension {
            component: item.component.to_owned(),
            port: item.port.to_owned(),
            request_index: item.request_index,
            item_index: item.item_index,
            dimension: dimension.to_owned(),
            declared_by: declared_by.to_owned(),
        })
}

fn same_contract(left: &EncoderWorkItem<'_>, right: &EncoderWorkItem<'_>) -> bool {
    left.component == right.component
        && left.port == right.port
        && left.contract == right.contract
        && left.capacity == right.capacity
}

fn dimensions_compatible(
    left: &EncoderWorkItem<'_>,
    right: &EncoderWorkItem<'_>,
) -> Result<bool, EncoderBatchingError> {
    let Some(shape) = &left.contract.shape else {
        return Ok(true);
    };
    let capacity = left
        .capacity
        .expect("batched compatibility is called only for declared capacity");
    for (axis, dimension) in shape.iter().enumerate() {
        let TensorDimension::Symbol(symbol) = dimension else {
            continue;
        };
        if dimension_may_vary(left.contract, capacity, axis, symbol) {
            continue;
        }
        if require_dimension(left, symbol, "the tensor contract shape")?
            != require_dimension(right, symbol, "the tensor contract shape")?
        {
            return Ok(false);
        }
    }
    for symbol in &capacity.uniform_dimensions {
        if require_dimension(left, symbol, "batch_capacity.uniform_dimensions")?
            != require_dimension(right, symbol, "batch_capacity.uniform_dimensions")?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn dimension_may_vary(
    contract: &TensorContract,
    capacity: &ComponentBatchCapacity,
    axis: usize,
    symbol: &str,
) -> bool {
    if capacity
        .uniform_dimensions
        .iter()
        .any(|uniform| uniform == symbol)
    {
        return false;
    }
    if contract
        .padding
        .iter()
        .any(|padding| padding.dimension == symbol)
    {
        return true;
    }
    match contract.batch_layout {
        BatchLayout::RequestAligned { axis: request_axis }
        | BatchLayout::RequestExpanded {
            axis: request_axis, ..
        } => request_axis == axis,
        BatchLayout::TokenPacked {
            axis: packed_axis, ..
        } => packed_axis == axis,
        BatchLayout::Shared | BatchLayout::RuntimeSequenceState => false,
    }
}

fn ensure_single_item_within_budgets(
    item: &EncoderWorkItem<'_>,
) -> Result<(), EncoderBatchingError> {
    let capacity = item
        .capacity
        .expect("single-item budgets are checked only for declared capacity");
    for budget in &capacity.budgets {
        let materialized = materialized_budget(&[item], &budget.dimensions)?;
        if materialized > budget.max_total {
            return Err(EncoderBatchingError::SingleItemExceedsBudget {
                component: item.component.to_owned(),
                port: item.port.to_owned(),
                request_index: item.request_index,
                item_index: item.item_index,
                dimensions: budget.dimensions.clone(),
                materialized,
                max_total: budget.max_total,
            });
        }
    }
    Ok(())
}

fn group_with_item_fits(
    group: &EncoderGroup,
    candidate: usize,
    items: &[EncoderWorkItem<'_>],
) -> Result<bool, EncoderBatchingError> {
    let first = &items[group.item_indices[0]];
    let capacity = first
        .capacity
        .expect("group budgets are checked only for declared capacity");
    let mut members = group
        .item_indices
        .iter()
        .map(|index| &items[*index])
        .collect::<Vec<_>>();
    members.push(&items[candidate]);
    for budget in &capacity.budgets {
        if materialized_budget(&members, &budget.dimensions)? > budget.max_total {
            return Ok(false);
        }
    }
    Ok(true)
}

fn materialized_budget(
    items: &[&EncoderWorkItem<'_>],
    dimensions: &[String],
) -> Result<usize, EncoderBatchingError> {
    let first = items[0];
    let outer = items.iter().try_fold(0usize, |total, item| {
        let extent = require_dimension(item, &dimensions[0], "batch_capacity.budgets")?;
        total
            .checked_add(extent)
            .ok_or_else(|| EncoderBatchingError::BudgetOverflow {
                component: first.component.to_owned(),
                port: first.port.to_owned(),
                dimensions: dimensions.to_vec(),
            })
    })?;
    dimensions.iter().skip(1).try_fold(outer, |total, symbol| {
        let extent = items
            .iter()
            .map(|item| require_dimension(item, symbol, "batch_capacity.budgets"))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        total
            .checked_mul(extent)
            .ok_or_else(|| EncoderBatchingError::BudgetOverflow {
                component: first.component.to_owned(),
                port: first.port.to_owned(),
                dimensions: dimensions.to_vec(),
            })
    })
}

impl PackedOwnership {
    /// Builds one level mapping packed items directly into request rows.
    pub fn one_level(item_counts: &[usize]) -> Result<Self, EncoderBatchingError> {
        let level = ownership_level(item_counts, "request item")?;
        Self::from_levels(item_counts.len(), vec![level])
    }

    /// Builds frames → items → requests over one physical frame axis.
    pub fn two_levels(part_counts_by_request: &[Vec<usize>]) -> Result<Self, EncoderBatchingError> {
        let item_counts = part_counts_by_request
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>();
        let part_counts = part_counts_by_request
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let inner = ownership_level(&part_counts, "nested physical part")?;
        let outer = ownership_level(&item_counts, "request item")?;
        Self::from_levels(part_counts_by_request.len(), vec![inner, outer])
    }

    /// Validates an externally produced ownership chain.
    pub fn from_levels(
        request_count: usize,
        levels: Vec<PackedOwnershipLevel>,
    ) -> Result<Self, EncoderBatchingError> {
        if !(1..=2).contains(&levels.len()) {
            return Err(EncoderBatchingError::Ownership {
                detail: format!(
                    "expected one or two levels, got {}; levels are ordered innermost first",
                    levels.len()
                ),
            });
        }
        let mut parent_count = request_count;
        for (level_index, level) in levels.iter().enumerate().rev() {
            validate_level(level_index, level, parent_count)?;
            parent_count = level.owner.len();
        }
        Ok(Self {
            request_count,
            physical_count: parent_count,
            levels,
        })
    }

    pub fn request_count(&self) -> usize {
        self.request_count
    }

    pub fn physical_count(&self) -> usize {
        self.physical_count
    }

    pub fn levels(&self) -> &[PackedOwnershipLevel] {
        &self.levels
    }

    pub fn request_spans(&self) -> Vec<RequestSpan> {
        (0..self.request_count)
            .map(|request_index| self.request_span(request_index))
            .collect()
    }

    pub fn request_local(
        &self,
        request_index: usize,
    ) -> Result<RequestOwnership, EncoderBatchingError> {
        if request_index >= self.request_count {
            return Err(EncoderBatchingError::Ownership {
                detail: format!(
                    "request index {request_index} is outside {} request rows",
                    self.request_count
                ),
            });
        }
        let span = self.request_span(request_index);
        let levels = if self.levels.len() == 1 {
            vec![PackedOwnershipLevel {
                offsets: vec![
                    0,
                    to_i64(span.physical_length, "request-local physical length")?,
                ],
                owner: vec![0; span.physical_length],
            }]
        } else {
            let inner = &self.levels[0];
            let start_item = span.item_offset;
            let end_item = start_item + span.item_length;
            let start_physical = span.physical_offset;
            let local_offsets = inner.offsets[start_item..=end_item]
                .iter()
                .map(|offset| {
                    let offset =
                        usize::try_from(*offset).expect("validated ownership is nonnegative");
                    to_i64(
                        offset - start_physical,
                        "request-local nested ownership offset",
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let local_owner = inner.owner[start_physical..start_physical + span.physical_length]
                .iter()
                .map(|owner| {
                    let owner =
                        usize::try_from(*owner).expect("validated ownership is nonnegative");
                    to_i64(owner - start_item, "request-local nested owner")
                })
                .collect::<Result<Vec<_>, _>>()?;
            vec![
                PackedOwnershipLevel {
                    offsets: local_offsets,
                    owner: local_owner,
                },
                PackedOwnershipLevel {
                    offsets: vec![0, to_i64(span.item_length, "request-local item length")?],
                    owner: vec![0; span.item_length],
                },
            ]
        };
        Ok(RequestOwnership { span, levels })
    }

    fn request_span(&self, request_index: usize) -> RequestSpan {
        let outer = self
            .levels
            .last()
            .expect("ownership has at least one level");
        let item_offset = usize::try_from(outer.offsets[request_index])
            .expect("validated ownership is nonnegative");
        let item_end = usize::try_from(outer.offsets[request_index + 1])
            .expect("validated ownership is nonnegative");
        let (physical_offset, physical_end) = if self.levels.len() == 1 {
            (item_offset, item_end)
        } else {
            let inner = &self.levels[0];
            (
                usize::try_from(inner.offsets[item_offset])
                    .expect("validated ownership is nonnegative"),
                usize::try_from(inner.offsets[item_end])
                    .expect("validated ownership is nonnegative"),
            )
        };
        RequestSpan {
            request_index,
            item_offset,
            item_length: item_end - item_offset,
            physical_offset,
            physical_length: physical_end - physical_offset,
        }
    }
}

fn ownership_level(
    child_counts: &[usize],
    description: &str,
) -> Result<PackedOwnershipLevel, EncoderBatchingError> {
    let mut offsets = Vec::with_capacity(child_counts.len() + 1);
    let mut owner = Vec::new();
    offsets.push(0);
    let mut total = 0usize;
    for (parent, count) in child_counts.iter().copied().enumerate() {
        total = total
            .checked_add(count)
            .ok_or_else(|| EncoderBatchingError::Ownership {
                detail: format!("{description} count overflowed at parent {parent}"),
            })?;
        offsets.push(to_i64(total, description)?);
        owner.extend(std::iter::repeat_n(to_i64(parent, description)?, count));
    }
    Ok(PackedOwnershipLevel { offsets, owner })
}

fn validate_level(
    level_index: usize,
    level: &PackedOwnershipLevel,
    parent_count: usize,
) -> Result<(), EncoderBatchingError> {
    let expected_offsets =
        parent_count
            .checked_add(1)
            .ok_or_else(|| EncoderBatchingError::Ownership {
                detail: format!("level {level_index} parent count overflows usize"),
            })?;
    if level.offsets.len() != expected_offsets {
        return Err(EncoderBatchingError::Ownership {
            detail: format!(
                "level {level_index} has {} offsets for {parent_count} parents; expected \
                 {expected_offsets}",
                level.offsets.len()
            ),
        });
    }
    if level.offsets.first().copied() != Some(0) {
        return Err(EncoderBatchingError::Ownership {
            detail: format!(
                "level {level_index} offset 0 is {:?}; exclusive-prefix offsets must start at 0",
                level.offsets.first()
            ),
        });
    }
    for (index, pair) in level.offsets.windows(2).enumerate() {
        if pair[0] < 0 || pair[1] < pair[0] {
            return Err(EncoderBatchingError::Ownership {
                detail: format!(
                    "level {level_index} offsets are not monotonic at index {index}: {} then {}",
                    pair[0], pair[1]
                ),
            });
        }
    }
    let total =
        usize::try_from(*level.offsets.last().expect("offsets are nonempty")).map_err(|_| {
            EncoderBatchingError::Ownership {
                detail: format!("level {level_index} final offset is negative"),
            }
        })?;
    if level.owner.len() != total {
        return Err(EncoderBatchingError::Ownership {
            detail: format!(
                "level {level_index} final offset is {total}, but its owner map has {} entries",
                level.owner.len()
            ),
        });
    }
    for parent in 0..parent_count {
        let start = usize::try_from(level.offsets[parent]).expect("offsets were validated");
        let end = usize::try_from(level.offsets[parent + 1]).expect("offsets were validated");
        for (child, owner) in level.owner[start..end].iter().enumerate() {
            if usize::try_from(*owner).ok() != Some(parent) {
                return Err(EncoderBatchingError::Ownership {
                    detail: format!(
                        "level {level_index} owner at child {} is {owner}, but offsets assign that \
                         child to parent {parent}",
                        start + child
                    ),
                });
            }
        }
    }
    Ok(())
}

fn to_i64(value: usize, description: &str) -> Result<i64, EncoderBatchingError> {
    i64::try_from(value).map_err(|_| EncoderBatchingError::Ownership {
        detail: format!("{description} {value} exceeds int64"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract(document: &str) -> TensorContract {
        serde_yaml::from_str(document).expect("contract parses")
    }

    fn capacity(document: &str) -> ComponentBatchCapacity {
        serde_yaml::from_str(document).expect("capacity parses")
    }

    fn dimensions(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
        entries
            .iter()
            .map(|(name, extent)| ((*name).to_owned(), *extent))
            .collect()
    }

    #[test]
    fn grouping_is_stable_across_component_contract_uniformity_and_budgets() {
        let primary_contract = contract(
            "dtype: float32\nrank: 3\nshape: [items, patches, features]\n\
             batch_layout: { kind: token_packed, axis: 0, levels: [{ offsets: offsets, owner: owner }] }\n\
             padding: [{ dimension: patches, valid_lengths: lengths }]\n",
        );
        let other_contract = contract(
            "dtype: float16\nrank: 3\nshape: [items, patches, features]\n\
             batch_layout: { kind: token_packed, axis: 0, levels: [{ offsets: offsets, owner: owner }] }\n\
             padding: [{ dimension: patches, valid_lengths: lengths }]\n",
        );
        let capacity = capacity(
            "uniform_dimensions: [features]\n\
             budgets:\n  - { dimensions: [items], max_total: 3 }\n  \
             - { dimensions: [items, patches], max_total: 12 }\n",
        );
        let extents = [
            dimensions(&[("items", 1), ("patches", 4), ("features", 8)]),
            dimensions(&[("items", 1), ("patches", 2), ("features", 8)]),
            dimensions(&[("items", 1), ("patches", 4), ("features", 16)]),
            dimensions(&[("items", 1), ("patches", 4), ("features", 8)]),
            dimensions(&[("items", 1), ("patches", 4), ("features", 8)]),
            dimensions(&[("items", 1), ("patches", 4), ("features", 8)]),
            dimensions(&[("items", 1), ("patches", 4), ("features", 8)]),
        ];
        let work = [
            EncoderWorkItem {
                request_index: 0,
                item_index: 0,
                component: "encoder",
                port: "pixels",
                contract: &primary_contract,
                capacity: Some(&capacity),
                dimensions: &extents[0],
            },
            EncoderWorkItem {
                request_index: 1,
                item_index: 0,
                component: "encoder",
                port: "pixels",
                contract: &primary_contract,
                capacity: Some(&capacity),
                dimensions: &extents[1],
            },
            EncoderWorkItem {
                request_index: 2,
                item_index: 0,
                component: "encoder",
                port: "pixels",
                contract: &primary_contract,
                capacity: Some(&capacity),
                dimensions: &extents[2],
            },
            EncoderWorkItem {
                request_index: 3,
                item_index: 0,
                component: "other",
                port: "pixels",
                contract: &primary_contract,
                capacity: Some(&capacity),
                dimensions: &extents[3],
            },
            EncoderWorkItem {
                request_index: 4,
                item_index: 0,
                component: "encoder",
                port: "pixels",
                contract: &primary_contract,
                capacity: Some(&capacity),
                dimensions: &extents[4],
            },
            EncoderWorkItem {
                request_index: 5,
                item_index: 0,
                component: "encoder",
                port: "other_pixels",
                contract: &primary_contract,
                capacity: Some(&capacity),
                dimensions: &extents[5],
            },
            EncoderWorkItem {
                request_index: 6,
                item_index: 0,
                component: "encoder",
                port: "pixels",
                contract: &other_contract,
                capacity: Some(&capacity),
                dimensions: &extents[6],
            },
        ];

        let groups = plan_encoder_groups(&work).unwrap();
        assert_eq!(
            groups,
            [
                EncoderGroup {
                    item_indices: vec![0, 1, 4]
                },
                EncoderGroup {
                    item_indices: vec![2]
                },
                EncoderGroup {
                    item_indices: vec![3]
                },
                EncoderGroup {
                    item_indices: vec![5]
                },
                EncoderGroup {
                    item_indices: vec![6]
                },
            ]
        );
    }

    #[test]
    fn padded_budget_charges_the_materialized_rectangle() {
        let contract = contract(
            "dtype: float32\nrank: 3\nshape: [items, patches, features]\n\
             batch_layout: { kind: token_packed, axis: 0, levels: [{ offsets: offsets, owner: owner }] }\n\
             padding: [{ dimension: patches, valid_lengths: lengths }]\n",
        );
        let capacity = capacity("budgets: [{ dimensions: [items, patches], max_total: 10 }]\n");
        let extents = [
            dimensions(&[("items", 1), ("patches", 5), ("features", 8)]),
            dimensions(&[("items", 1), ("patches", 1), ("features", 8)]),
            dimensions(&[("items", 1), ("patches", 1), ("features", 8)]),
        ];
        let work = extents
            .iter()
            .enumerate()
            .map(|(request_index, dimensions)| EncoderWorkItem {
                request_index,
                item_index: 0,
                component: "encoder",
                port: "pixels",
                contract: &contract,
                capacity: Some(&capacity),
                dimensions,
            })
            .collect::<Vec<_>>();

        let groups = plan_encoder_groups(&work).unwrap();
        assert_eq!(groups[0].item_indices, [0, 1]);
        assert_eq!(
            groups[1].item_indices,
            [2],
            "three items charge 3 * max(5, 1, 1) = 15, not the valid sum 7"
        );
    }

    #[test]
    fn missing_authored_extent_is_an_actionable_error() {
        let contract = contract("dtype: float32\nrank: 2\nshape: [items, features]\n");
        let capacity = capacity("uniform_dimensions: [features]\n");
        let dimensions = dimensions(&[("items", 1)]);
        let error = plan_encoder_groups(&[EncoderWorkItem {
            request_index: 3,
            item_index: 2,
            component: "vision",
            port: "pixels",
            contract: &contract,
            capacity: Some(&capacity),
            dimensions: &dimensions,
        }])
        .unwrap_err();
        assert!(error.to_string().contains("request 3 item 2"));
        assert!(error.to_string().contains("features"));
    }

    #[test]
    fn nested_ownership_round_trips_request_local_ranges() {
        let ownership = PackedOwnership::two_levels(&[vec![2, 1], Vec::new(), vec![3]]).unwrap();
        assert_eq!(
            ownership.levels(),
            [
                PackedOwnershipLevel {
                    offsets: vec![0, 2, 3, 6],
                    owner: vec![0, 0, 1, 2, 2, 2],
                },
                PackedOwnershipLevel {
                    offsets: vec![0, 2, 2, 3],
                    owner: vec![0, 0, 2],
                },
            ]
        );
        assert_eq!(
            ownership.request_spans(),
            [
                RequestSpan {
                    request_index: 0,
                    item_offset: 0,
                    item_length: 2,
                    physical_offset: 0,
                    physical_length: 3,
                },
                RequestSpan {
                    request_index: 1,
                    item_offset: 2,
                    item_length: 0,
                    physical_offset: 3,
                    physical_length: 0,
                },
                RequestSpan {
                    request_index: 2,
                    item_offset: 2,
                    item_length: 1,
                    physical_offset: 3,
                    physical_length: 3,
                },
            ]
        );
        let local = ownership.request_local(2).unwrap();
        assert_eq!(local.span.physical_offset, 3);
        assert_eq!(
            local.levels,
            [
                PackedOwnershipLevel {
                    offsets: vec![0, 3],
                    owner: vec![0, 0, 0],
                },
                PackedOwnershipLevel {
                    offsets: vec![0, 1],
                    owner: vec![0],
                },
            ]
        );
    }

    #[test]
    fn corrupted_nested_owner_is_rejected_without_clamping() {
        let error = PackedOwnership::from_levels(
            2,
            vec![
                PackedOwnershipLevel {
                    offsets: vec![0, 2, 3],
                    owner: vec![0, 1, 1],
                },
                PackedOwnershipLevel {
                    offsets: vec![0, 1, 2],
                    owner: vec![0, 1],
                },
            ],
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("level 0 owner at child 1"));
        assert!(message.contains("parent 0"));
    }
}
