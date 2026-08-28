//! Mandatory row-scoped state ABI.
//!
//! Continuous batching moves requests between physical batch rows: a finished
//! request frees its row and a scheduler compacts the survivors down. Every
//! component that keeps per-row state — a LoRA composition, a native grammar
//! parser, a vision encoder result — must be able to follow that move, or the
//! next token is generated against another request's state.
//!
//! This is an ABI invariant, not a negotiated capability. A component that
//! declares row scope in metadata implements [`RowScopedState`]; there is no
//! opt-out and nothing for a package to advertise. Metadata declares the row
//! axis and layout; it never serializes row identities. The permutation passed
//! to [`RowScopedState::compact`] is *positional*: entry `i` names the source
//! position that becomes destination position `i`, so no scheduler slot ID or
//! request epoch ever crosses the boundary.

use anyhow::Context;

/// Per-row state that survives continuous-batching compaction.
pub trait RowScopedState {
    /// Number of rows currently held.
    fn rows(&self) -> usize;

    /// Reorder rows so that destination `i` holds the state of `selection[i]`.
    ///
    /// A source position may repeat, which is how beam search and speculative
    /// row expansion clone a row without naming it. A position absent from the
    /// selection is dropped.
    fn compact(&mut self, selection: &[usize]) -> anyhow::Result<()>;

    /// Drop the state of one row, leaving the remaining rows in place.
    fn release(&mut self, row: usize) -> anyhow::Result<()>;
}

/// One positional operation applied to every carrier in a live batch.
///
/// The operation names source *positions*, never request identities or physical
/// scheduler slots.  Validating all carriers before changing any of them makes
/// a mismatched companion fail before a state carrier can be moved on its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowPlan {
    /// Destination `i` receives the former source position `sources[i]`.
    Select {
        source_rows: usize,
        sources: Vec<usize>,
    },
    /// Drop one position and retain every other position in order.
    Release { source_rows: usize, row: usize },
}

impl RowPlan {
    /// Build a positional selection, including repeated sources for row cloning.
    pub fn select(source_rows: usize, sources: Vec<usize>) -> anyhow::Result<Self> {
        check_selection(&sources, source_rows)?;
        Ok(Self::Select {
            source_rows,
            sources,
        })
    }

    /// Build a release operation for one existing row.
    pub fn release(source_rows: usize, row: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            row < source_rows,
            "cannot release row {row}; only {source_rows} rows exist"
        );
        Ok(Self::Release { source_rows, row })
    }

    /// Keep exactly the active source positions, preserving their order.
    pub fn active(active: &[bool]) -> Self {
        Self::Select {
            source_rows: active.len(),
            sources: active
                .iter()
                .enumerate()
                .filter_map(|(row, active)| active.then_some(row))
                .collect(),
        }
    }

    /// Number of source rows this operation was formed against.
    pub fn source_rows(&self) -> usize {
        match self {
            Self::Select { source_rows, .. } | Self::Release { source_rows, .. } => *source_rows,
        }
    }

    /// Source positions retained by this operation.
    pub fn selected_rows(&self) -> Vec<usize> {
        match self {
            Self::Select { sources, .. } => sources.clone(),
            Self::Release { source_rows, row } => {
                (0..*source_rows).filter(|source| source != row).collect()
            }
        }
    }

    /// Apply this one plan to all row-scoped carriers.
    ///
    /// The preflight validates every carrier's source extent before the first
    /// carrier is changed.  A row-scoped implementation may still report its
    /// own storage failure, but differing row counts never leave a subset of
    /// carriers compacted.
    pub fn apply(&self, carriers: &mut [&mut dyn RowScopedState]) -> anyhow::Result<()> {
        let source_rows = self.source_rows();
        for (carrier, state) in carriers.iter().enumerate() {
            anyhow::ensure!(
                state.rows() == source_rows,
                "row plan expects {source_rows} source rows, but carrier {carrier} holds {}; \
                 refuse to move only part of a batch",
                state.rows()
            );
        }
        match self {
            Self::Select { sources, .. } => {
                for state in carriers {
                    state.compact(sources)?;
                }
            }
            Self::Release { row, .. } => {
                for state in carriers {
                    state.release(*row)?;
                }
            }
        }
        Ok(())
    }
}

/// Validate a positional row selection against a row count.
///
/// Returns the selection unchanged on success. An out-of-range source position
/// is a scheduler bug that would otherwise silently read another request's
/// state, so it fails loudly.
pub fn check_selection(selection: &[usize], rows: usize) -> anyhow::Result<()> {
    for (destination, &source) in selection.iter().enumerate() {
        anyhow::ensure!(
            source < rows,
            "row selection maps destination {destination} to source row {source}, but only \
             {rows} rows exist"
        );
    }
    Ok(())
}

/// Apply a positional row selection to a vector of per-row values.
pub fn gather_rows<T: Clone>(rows: &[T], selection: &[usize]) -> anyhow::Result<Vec<T>> {
    check_selection(selection, rows.len())?;
    Ok(selection
        .iter()
        .map(|&source| rows[source].clone())
        .collect())
}

/// A generic row-scoped table for native and binding components.
///
/// Components whose per-row state is a plain owned value get the mandatory ABI
/// for free by storing it here rather than reimplementing compaction.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RowTable<T> {
    rows: Vec<T>,
}

impl<T> RowTable<T> {
    /// Build a table from one value per batch row, in row order.
    pub fn new(rows: Vec<T>) -> Self {
        Self { rows }
    }

    /// Per-row values in current row order.
    pub fn as_slice(&self) -> &[T] {
        &self.rows
    }

    /// Read one row.
    pub fn get(&self, row: usize) -> Option<&T> {
        self.rows.get(row)
    }

    /// Mutate one row.
    pub fn get_mut(&mut self, row: usize) -> Option<&mut T> {
        self.rows.get_mut(row)
    }

    /// Consume the table into its per-row values.
    pub fn into_inner(self) -> Vec<T> {
        self.rows
    }
}

impl<T: Clone> RowScopedState for RowTable<T> {
    fn rows(&self) -> usize {
        self.rows.len()
    }

    fn compact(&mut self, selection: &[usize]) -> anyhow::Result<()> {
        self.rows = gather_rows(&self.rows, selection).context("row table compaction")?;
        Ok(())
    }

    fn release(&mut self, row: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            row < self.rows.len(),
            "cannot release row {row}; only {} rows exist",
            self.rows.len()
        );
        self.rows.remove(row);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compaction_moves_rows_to_their_new_positions() {
        let mut table = RowTable::new(vec!["a", "b", "c", "d"]);
        table.compact(&[3, 1]).expect("valid selection");
        assert_eq!(table.as_slice(), ["d", "b"]);
    }

    #[test]
    fn compaction_may_clone_a_row_for_beam_or_speculative_expansion() {
        let mut table = RowTable::new(vec!["a", "b"]);
        table.compact(&[1, 1, 0]).expect("valid selection");
        assert_eq!(table.as_slice(), ["b", "b", "a"]);
    }

    #[test]
    fn compaction_rejects_an_out_of_range_source_row() {
        let mut table = RowTable::new(vec!["a", "b"]);
        let error = table
            .compact(&[2])
            .expect_err("an out-of-range source row must fail loudly");
        assert!(format!("{error:#}").contains("source row 2"), "{error:#}");
    }

    #[test]
    fn release_drops_one_row_and_keeps_the_rest_in_order() {
        let mut table = RowTable::new(vec!["a", "b", "c"]);
        table.release(1).expect("valid row");
        assert_eq!(table.as_slice(), ["a", "c"]);
        assert!(table.release(5).is_err(), "releasing a missing row fails");
    }

    #[test]
    fn an_empty_selection_drains_every_row() {
        let mut table = RowTable::new(vec![1, 2, 3]);
        table.compact(&[]).expect("draining is valid");
        assert_eq!(table.rows(), 0);
    }

    #[test]
    fn one_plan_keeps_state_effect_and_output_carriers_aligned() {
        let mut state = RowTable::new(vec!["state-a", "state-b", "state-c"]);
        let mut effect = RowTable::new(vec!["effect-a", "effect-b", "effect-c"]);
        let mut output = RowTable::new(vec!["output-a", "output-b", "output-c"]);
        let plan = RowPlan::select(3, vec![2, 2, 0]).expect("repeated selection is legal");

        plan.apply(&mut [&mut state, &mut effect, &mut output])
            .expect("all carriers follow the same positional plan");

        assert_eq!(state.as_slice(), ["state-c", "state-c", "state-a"]);
        assert_eq!(effect.as_slice(), ["effect-c", "effect-c", "effect-a"]);
        assert_eq!(output.as_slice(), ["output-c", "output-c", "output-a"]);
    }

    #[test]
    fn release_plan_removes_the_same_row_from_every_carrier() {
        let mut state = RowTable::new(vec![1, 2, 3]);
        let mut effect = RowTable::new(vec![10, 20, 30]);
        let plan = RowPlan::release(3, 1).expect("row one exists");

        plan.apply(&mut [&mut state, &mut effect])
            .expect("release is applied uniformly");

        assert_eq!(state.as_slice(), [1, 3]);
        assert_eq!(effect.as_slice(), [10, 30]);
    }

    #[test]
    fn mismatched_carrier_refuses_before_any_carrier_moves() {
        let mut state = RowTable::new(vec![1, 2, 3]);
        let mut output = RowTable::new(vec![10, 20]);
        let plan = RowPlan::select(3, vec![2, 0]).expect("selection is valid for state");

        let error = plan
            .apply(&mut [&mut state, &mut output])
            .expect_err("a shorter carrier cannot receive another row's output");

        assert!(format!("{error:#}").contains("carrier 1"));
        assert_eq!(state.as_slice(), [1, 2, 3], "state was not partially moved");
        assert_eq!(output.as_slice(), [10, 20]);
    }

    #[test]
    fn active_plan_drops_inactive_rows_without_reordering_survivors() {
        let plan = RowPlan::active(&[false, true, false, true]);
        assert_eq!(plan.selected_rows(), vec![1, 3]);
        let mut rows = RowTable::new(vec!["released", "b", "released", "d"]);
        plan.apply(&mut [&mut rows]).expect("active rows compact");
        assert_eq!(rows.as_slice(), ["b", "d"]);
    }
}
