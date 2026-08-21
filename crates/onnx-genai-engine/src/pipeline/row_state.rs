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
}
