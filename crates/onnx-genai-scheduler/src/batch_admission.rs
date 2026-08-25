//! Symbol-keyed admission for materialized component batches.
//!
//! This layer knows nothing about tensor axis numbers. A runtime resolves each
//! contract-local shape symbol to the contribution it makes when requests are
//! grouped: packed/request counts add, padded extents take the maximum, and
//! declared uniform symbols must agree. The resulting dimensions are the
//! allocation the backend will actually see, so budget products include padding.

use std::collections::{BTreeMap, BTreeSet};

use onnx_genai_kv::SequenceId;

/// How one request's extent contributes to a materialized group dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchDimensionAggregation {
    /// Counts on packed or request axes concatenate.
    Sum,
    /// Padded or invariant dimensions materialize at the largest extent.
    Maximum,
}

/// One upper bound over a product of materialized shape-symbol extents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializedBudget {
    pub dimensions: Vec<String>,
    pub max_total: usize,
}

/// Component-level grouping and footprint policy.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchPolicy {
    /// Optional hard limit on contributions in one group.
    pub max_contributions: Option<usize>,
    /// Aggregation for every symbol used by grouping or a budget.
    pub aggregations: BTreeMap<String, BatchDimensionAggregation>,
    /// Symbols that must agree before contributions may share a group.
    pub uniform_dimensions: BTreeSet<String>,
    /// Bounds on the materialized group footprint.
    pub budgets: Vec<MaterializedBudget>,
}

/// One request's component-local symbolic extents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchContribution {
    pub sequence_id: SequenceId,
    pub dimensions: BTreeMap<String, usize>,
}

/// A deterministic, admissible group in original request order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedBatch {
    sequence_ids: Vec<SequenceId>,
    materialized_dimensions: BTreeMap<String, usize>,
}

impl AdmittedBatch {
    pub fn sequence_ids(&self) -> &[SequenceId] {
        &self.sequence_ids
    }

    pub fn materialized_dimensions(&self) -> &BTreeMap<String, usize> {
        &self.materialized_dimensions
    }
}

/// A malformed profile or a contribution that cannot fit even by itself.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BatchAdmissionError {
    #[error(
        "batch admission policy references shape symbol '{symbol}' but declares no aggregation; \
         classify it as a summed packed/request count or a maximum padded/invariant extent"
    )]
    MissingAggregation { symbol: String },
    #[error(
        "batch contribution for sequence {sequence_id} does not resolve required shape symbol \
         '{symbol}'; bind every capacity symbol from the component's typed input contracts"
    )]
    MissingDimension {
        sequence_id: SequenceId,
        symbol: String,
    },
    #[error(
        "materialized batch does not resolve required shape symbol '{symbol}'; validate the \
         assembled invocation against its typed component inputs before backend enqueue"
    )]
    MissingMaterializedDimension { symbol: String },
    #[error(
        "batch materialization overflowed while adding shape symbol '{symbol}'; reject the group \
         rather than wrapping its packed/request extent"
    )]
    DimensionOverflow { symbol: String },
    #[error(
        "batch materialized footprint for dimensions {dimensions:?} overflows usize; reduce the \
         group before allocation"
    )]
    FootprintOverflow { dimensions: Vec<String> },
    #[error(
        "batch contribution for sequence {sequence_id} materializes {materialized} units for \
         dimensions {dimensions:?}, exceeding the component limit {max_total}; reduce this \
         request's item/padding extent or use an artifact with a larger declared capacity"
    )]
    BudgetExceeded {
        sequence_id: SequenceId,
        dimensions: Vec<String>,
        materialized: usize,
        max_total: usize,
    },
    #[error(
        "assembled batch materializes {materialized} units for dimensions {dimensions:?}, \
         exceeding the component limit {max_total}; split or reject the group before backend \
         enqueue"
    )]
    MaterializedBudgetExceeded {
        dimensions: Vec<String>,
        materialized: usize,
        max_total: usize,
    },
    #[error(
        "batch policy max_contributions must be at least one when present; omit it for no \
         scheduler row limit"
    )]
    ZeroContributionLimit,
}

/// Form maximal consecutive groups without reordering requests.
///
/// A uniformity mismatch or a group-level budget boundary starts a new group.
/// A single contribution that exceeds a budget is rejected, because no grouping
/// decision can make it safe.
pub fn group_batch_contributions(
    policy: &BatchPolicy,
    contributions: &[BatchContribution],
) -> Result<Vec<AdmittedBatch>, BatchAdmissionError> {
    validate_policy(policy)?;
    validate_contributions(policy, contributions)?;
    if contributions.is_empty() {
        return Ok(Vec::new());
    }

    let mut admitted = Vec::new();
    let mut current_ids = Vec::new();
    let mut current_dimensions = BTreeMap::new();

    for contribution in contributions {
        let candidate = if current_ids.is_empty() {
            contribution.dimensions.clone()
        } else {
            merge_dimensions(policy, &current_dimensions, &contribution.dimensions)?
        };
        let uniform = current_ids.is_empty()
            || policy.uniform_dimensions.iter().all(|symbol| {
                current_dimensions.get(symbol) == contribution.dimensions.get(symbol)
            });
        let within_rows = policy
            .max_contributions
            .is_none_or(|limit| current_ids.len() < limit);
        let within_budgets = footprint_within_budgets(policy, &candidate)?;

        if !current_ids.is_empty() && (!uniform || !within_rows || !within_budgets) {
            admitted.push(AdmittedBatch {
                sequence_ids: std::mem::take(&mut current_ids),
                materialized_dimensions: std::mem::take(&mut current_dimensions),
            });
            current_dimensions = contribution.dimensions.clone();
        } else {
            current_dimensions = candidate;
        }

        validate_single_contribution(policy, contribution)?;
        current_ids.push(contribution.sequence_id);
    }

    admitted.push(AdmittedBatch {
        sequence_ids: current_ids,
        materialized_dimensions: current_dimensions,
    });
    Ok(admitted)
}

/// Validate the already-assembled dimensions immediately before backend enqueue.
pub fn validate_materialized_footprint(
    policy: &BatchPolicy,
    dimensions: &BTreeMap<String, usize>,
) -> Result<(), BatchAdmissionError> {
    validate_policy(policy)?;
    for symbol in required_symbols(policy) {
        if !dimensions.contains_key(&symbol) {
            return Err(BatchAdmissionError::MissingMaterializedDimension { symbol });
        }
    }
    for budget in &policy.budgets {
        let materialized = budget_footprint(dimensions, budget)?;
        if materialized > budget.max_total {
            return Err(BatchAdmissionError::MaterializedBudgetExceeded {
                dimensions: budget.dimensions.clone(),
                materialized,
                max_total: budget.max_total,
            });
        }
    }
    Ok(())
}

fn validate_policy(policy: &BatchPolicy) -> Result<(), BatchAdmissionError> {
    if policy.max_contributions == Some(0) {
        return Err(BatchAdmissionError::ZeroContributionLimit);
    }
    for symbol in required_symbols(policy) {
        if !policy.aggregations.contains_key(&symbol) {
            return Err(BatchAdmissionError::MissingAggregation { symbol });
        }
    }
    Ok(())
}

fn required_symbols(policy: &BatchPolicy) -> BTreeSet<String> {
    policy
        .uniform_dimensions
        .iter()
        .cloned()
        .chain(
            policy
                .budgets
                .iter()
                .flat_map(|budget| budget.dimensions.iter().cloned()),
        )
        .collect()
}

fn validate_contributions(
    policy: &BatchPolicy,
    contributions: &[BatchContribution],
) -> Result<(), BatchAdmissionError> {
    let required = required_symbols(policy);
    for contribution in contributions {
        for symbol in &required {
            if !contribution.dimensions.contains_key(symbol) {
                return Err(BatchAdmissionError::MissingDimension {
                    sequence_id: contribution.sequence_id,
                    symbol: symbol.clone(),
                });
            }
        }
    }
    Ok(())
}

fn merge_dimensions(
    policy: &BatchPolicy,
    current: &BTreeMap<String, usize>,
    next: &BTreeMap<String, usize>,
) -> Result<BTreeMap<String, usize>, BatchAdmissionError> {
    let mut merged = current.clone();
    for (symbol, aggregation) in &policy.aggregations {
        let Some(&next_extent) = next.get(symbol) else {
            continue;
        };
        let extent = match (merged.get(symbol).copied(), aggregation) {
            (None, _) => next_extent,
            (Some(current_extent), BatchDimensionAggregation::Maximum) => {
                current_extent.max(next_extent)
            }
            (Some(current_extent), BatchDimensionAggregation::Sum) => current_extent
                .checked_add(next_extent)
                .ok_or_else(|| BatchAdmissionError::DimensionOverflow {
                    symbol: symbol.clone(),
                })?,
        };
        merged.insert(symbol.clone(), extent);
    }
    Ok(merged)
}

fn footprint_within_budgets(
    policy: &BatchPolicy,
    dimensions: &BTreeMap<String, usize>,
) -> Result<bool, BatchAdmissionError> {
    for budget in &policy.budgets {
        if budget_footprint(dimensions, budget)? > budget.max_total {
            return Ok(false);
        }
    }
    Ok(true)
}

fn validate_single_contribution(
    policy: &BatchPolicy,
    contribution: &BatchContribution,
) -> Result<(), BatchAdmissionError> {
    for budget in &policy.budgets {
        let materialized = budget_footprint(&contribution.dimensions, budget)?;
        if materialized > budget.max_total {
            return Err(BatchAdmissionError::BudgetExceeded {
                sequence_id: contribution.sequence_id,
                dimensions: budget.dimensions.clone(),
                materialized,
                max_total: budget.max_total,
            });
        }
    }
    Ok(())
}

fn budget_footprint(
    dimensions: &BTreeMap<String, usize>,
    budget: &MaterializedBudget,
) -> Result<usize, BatchAdmissionError> {
    budget.dimensions.iter().try_fold(1usize, |total, symbol| {
        let extent = dimensions.get(symbol).copied().ok_or_else(|| {
            BatchAdmissionError::MissingMaterializedDimension {
                symbol: symbol.clone(),
            }
        })?;
        total
            .checked_mul(extent)
            .ok_or_else(|| BatchAdmissionError::FootprintOverflow {
                dimensions: budget.dimensions.clone(),
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dimensions(values: &[(&str, usize)]) -> BTreeMap<String, usize> {
        values
            .iter()
            .map(|(symbol, extent)| ((*symbol).to_string(), *extent))
            .collect()
    }

    fn padded_policy(max_total: usize) -> BatchPolicy {
        BatchPolicy {
            aggregations: BTreeMap::from([
                ("batch".into(), BatchDimensionAggregation::Sum),
                ("max_tiles".into(), BatchDimensionAggregation::Maximum),
                ("height".into(), BatchDimensionAggregation::Maximum),
            ]),
            uniform_dimensions: BTreeSet::from(["height".into()]),
            budgets: vec![MaterializedBudget {
                dimensions: vec!["batch".into(), "max_tiles".into()],
                max_total,
            }],
            ..BatchPolicy::default()
        }
    }

    #[test]
    fn padding_cost_uses_group_rectangle_not_valid_sum() {
        let requests = [
            BatchContribution {
                sequence_id: 10,
                dimensions: dimensions(&[("batch", 1), ("max_tiles", 8), ("height", 32)]),
            },
            BatchContribution {
                sequence_id: 11,
                dimensions: dimensions(&[("batch", 1), ("max_tiles", 2), ("height", 32)]),
            },
        ];
        let groups = group_batch_contributions(&padded_policy(16), &requests).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].sequence_ids(), &[10, 11]);
        assert_eq!(groups[0].materialized_dimensions()["batch"], 2);
        assert_eq!(groups[0].materialized_dimensions()["max_tiles"], 8);

        let split = group_batch_contributions(&padded_policy(12), &requests).unwrap();
        assert_eq!(
            split
                .iter()
                .map(|group| group.sequence_ids())
                .collect::<Vec<_>>(),
            vec![&[10][..], &[11][..]]
        );
    }

    #[test]
    fn uniformity_splits_without_reordering() {
        let requests = [
            BatchContribution {
                sequence_id: 1,
                dimensions: dimensions(&[("batch", 1), ("max_tiles", 1), ("height", 32)]),
            },
            BatchContribution {
                sequence_id: 2,
                dimensions: dimensions(&[("batch", 1), ("max_tiles", 1), ("height", 64)]),
            },
            BatchContribution {
                sequence_id: 3,
                dimensions: dimensions(&[("batch", 1), ("max_tiles", 1), ("height", 32)]),
            },
        ];
        let groups = group_batch_contributions(&padded_policy(64), &requests).unwrap();
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].sequence_ids(), &[1]);
        assert_eq!(groups[1].sequence_ids(), &[2]);
        assert_eq!(groups[2].sequence_ids(), &[3]);
    }

    #[test]
    fn empty_input_forms_no_groups_and_single_over_budget_is_rejected() {
        assert!(
            group_batch_contributions(&padded_policy(8), &[])
                .unwrap()
                .is_empty()
        );
        let error = group_batch_contributions(
            &padded_policy(8),
            &[BatchContribution {
                sequence_id: 7,
                dimensions: dimensions(&[("batch", 1), ("max_tiles", 9), ("height", 32)]),
            }],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BatchAdmissionError::BudgetExceeded {
                sequence_id: 7,
                materialized: 9,
                max_total: 8,
                ..
            }
        ));
    }

    #[test]
    fn summed_extent_overflow_fails_closed() {
        let policy = BatchPolicy {
            aggregations: BTreeMap::from([("items".into(), BatchDimensionAggregation::Sum)]),
            budgets: vec![MaterializedBudget {
                dimensions: vec!["items".into()],
                max_total: usize::MAX,
            }],
            ..BatchPolicy::default()
        };
        let error = group_batch_contributions(
            &policy,
            &[
                BatchContribution {
                    sequence_id: 1,
                    dimensions: dimensions(&[("items", usize::MAX)]),
                },
                BatchContribution {
                    sequence_id: 2,
                    dimensions: dimensions(&[("items", 1)]),
                },
            ],
        )
        .unwrap_err();
        assert_eq!(
            error,
            BatchAdmissionError::DimensionOverflow {
                symbol: "items".into()
            }
        );
    }
}
