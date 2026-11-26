//! Backend-neutral logits whose numeric backing outlives queued row consumers.
#[cfg(feature = "native-backend")]
use anyhow::Context as _;

#[cfg(feature = "native-backend")]
use onnx_runtime_session::OwnedHostArray;
use std::ops::{Deref, Index};
#[cfg(feature = "native-backend")]
use std::{ops::Range, sync::Arc};

pub struct LogitsBatch {
    storage: BatchStorage,
}
enum BatchStorage {
    // Existing ORT/host producer ownership, without a Native governance claim.
    Host(Vec<Vec<f32>>),
    Rows(Vec<LogitsRow>),
    #[cfg(feature = "native-backend")]
    Array {
        owner: Arc<OwnedHostArray<f32>>,
        rows: usize,
        width: usize,
    },
}
/// A consumable row. Shared Native rows are immutable until a fallible unique
/// ownership transition succeeds. No mutable Arc alias or uncharged copy exists.
pub struct LogitsRow {
    storage: RowStorage,
}
enum RowStorage {
    Host(Vec<f32>),
    #[cfg(feature = "native-backend")]
    Array {
        owner: Arc<OwnedHostArray<f32>>,
        range: Range<usize>,
    },
}
impl LogitsBatch {
    pub fn empty() -> Self {
        Self::from_host_rows(Vec::new())
    }
    pub fn from_host_rows(rows: Vec<Vec<f32>>) -> Self {
        Self {
            storage: BatchStorage::Host(rows),
        }
    }
    pub fn from_rows(rows: Vec<LogitsRow>) -> Self {
        Self {
            storage: BatchStorage::Rows(rows),
        }
    }
    #[cfg(feature = "native-backend")]
    pub fn from_array(
        owner: OwnedHostArray<f32>,
        rows: usize,
        width: usize,
    ) -> anyhow::Result<Self> {
        let required = rows.checked_mul(width).ok_or_else(|| anyhow::anyhow!("logits rows {rows} times vocabulary width {width} overflows; use addressable output geometry"))?;
        anyhow::ensure!(
            rows == 0 || width > 0,
            "nonempty logits require a positive vocabulary width"
        );
        anyhow::ensure!(
            required <= owner.len(),
            "logits geometry [{rows}, {width}] requires {required} values but owned backing has {}; match the declared output geometry",
            owner.len()
        );
        Ok(Self {
            storage: BatchStorage::Array {
                owner: Arc::new(owner),
                rows,
                width,
            },
        })
    }
    pub fn try_clone(&self) -> anyhow::Result<Self> {
        let mut rows = Vec::new();
        rows.try_reserve_exact(self.len()).map_err(|error| {
            anyhow::anyhow!(
                "cannot retain {} logits row descriptors: {error}",
                self.len()
            )
        })?;
        for index in 0..self.len() {
            rows.push(self.row_owned(index)?);
        }
        Ok(Self::from_rows(rows))
    }
    pub fn pop(&mut self) -> Option<LogitsRow> {
        if self.is_empty() {
            return None;
        }
        if self.len() == 1 {
            return std::mem::replace(self, Self::empty()).into_iter().next();
        }
        match &mut self.storage {
            BatchStorage::Host(rows) => rows.pop().map(LogitsRow::from_host_row),
            BatchStorage::Rows(rows) => rows.pop(),
            #[cfg(feature = "native-backend")]
            BatchStorage::Array { owner, rows, width } => {
                *rows -= 1;
                Some(LogitsRow {
                    storage: RowStorage::Array {
                        owner: Arc::clone(owner),
                        range: *rows * *width..(*rows + 1) * *width,
                    },
                })
            }
        }
    }
    pub fn len(&self) -> usize {
        match &self.storage {
            BatchStorage::Host(rows) => rows.len(),
            BatchStorage::Rows(rows) => rows.len(),
            #[cfg(feature = "native-backend")]
            BatchStorage::Array { rows, .. } => *rows,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn get(&self, index: usize) -> Option<&[f32]> {
        match &self.storage {
            BatchStorage::Host(rows) => rows.get(index).map(Vec::as_slice),
            BatchStorage::Rows(rows) => rows.get(index).map(LogitsRow::as_slice),
            #[cfg(feature = "native-backend")]
            BatchStorage::Array { owner, rows, width } => {
                if index >= *rows {
                    None
                } else {
                    Some(&owner.as_slice()[index * *width..(index + 1) * *width])
                }
            }
        }
    }
    /// Retain an immutable Native row owner independently of this batch.
    /// Existing Host/ORT storage produces its ordinary explicitly owned copy.
    pub fn row_owned(&self, index: usize) -> anyhow::Result<LogitsRow> {
        match &self.storage {
            BatchStorage::Host(rows) => {
                let source = rows.get(index).ok_or_else(|| {
                    anyhow::anyhow!(
                        "logits row index {index} exceeds Host batch length {}",
                        rows.len()
                    )
                })?;
                let mut copy = Vec::new();
                copy.try_reserve_exact(source.len()).map_err(|error| {
                    anyhow::anyhow!("cannot copy Host logits row {index}: {error}")
                })?;
                copy.extend_from_slice(source);
                Ok(LogitsRow::from_host_row(copy))
            }
            BatchStorage::Rows(rows) => rows
                .get(index)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "logits row index {index} exceeds owned batch length {}",
                        rows.len()
                    )
                })?
                .try_clone(),
            #[cfg(feature = "native-backend")]
            BatchStorage::Array { owner, rows, width } => {
                anyhow::ensure!(
                    index < *rows,
                    "logits row index {index} exceeds Native batch rows {rows}"
                );
                Ok(LogitsRow {
                    storage: RowStorage::Array {
                        owner: Arc::clone(owner),
                        range: index * *width..(index + 1) * *width,
                    },
                })
            }
        }
    }
    pub fn last(&self) -> Option<&[f32]> {
        self.len().checked_sub(1).and_then(|index| self.get(index))
    }
    pub fn iter(&self) -> LogitsIter<'_> {
        LogitsIter {
            batch: self,
            start: 0,
            end: self.len(),
        }
    }
    pub fn row_mut(&mut self, index: usize) -> anyhow::Result<&mut [f32]> {
        match &mut self.storage {
            BatchStorage::Host(rows) => rows
                .get_mut(index)
                .map(Vec::as_mut_slice)
                .ok_or_else(|| anyhow::anyhow!("logits row index {index} is out of bounds")),
            BatchStorage::Rows(rows) => rows
                .get_mut(index)
                .ok_or_else(|| anyhow::anyhow!("logits row index {index} is out of bounds"))?
                .try_mut(),
            #[cfg(feature = "native-backend")]
            BatchStorage::Array { owner, rows, width } => {
                anyhow::ensure!(
                    index < *rows,
                    "logits row index {index} exceeds batch rows {rows}"
                );
                if Arc::get_mut(owner).is_none() {
                    // Admission succeeds before replacing the immutable source.
                    let copy = owner.try_clone().context(
                        "admit unique Native logits batch while retaining its original backing",
                    )?;
                    *owner = Arc::new(copy);
                }
                Ok(&mut Arc::get_mut(owner)
                    .expect("unshared logits backing")
                    .as_mut_slice()[index * *width..(index + 1) * *width])
            }
        }
    }
}
pub struct LogitsIter<'a> {
    batch: &'a LogitsBatch,
    start: usize,
    end: usize,
}
impl<'a> Iterator for LogitsIter<'a> {
    type Item = &'a [f32];
    fn next(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            None
        } else {
            let index = self.start;
            self.start += 1;
            self.batch.get(index)
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len();
        (n, Some(n))
    }
}
impl DoubleEndedIterator for LogitsIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            None
        } else {
            self.end -= 1;
            self.batch.get(self.end)
        }
    }
}
impl ExactSizeIterator for LogitsIter<'_> {
    fn len(&self) -> usize {
        self.end - self.start
    }
}
impl<'a> IntoIterator for &'a LogitsBatch {
    type Item = &'a [f32];
    type IntoIter = LogitsIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl std::iter::FromIterator<LogitsRow> for LogitsBatch {
    fn from_iter<I: IntoIterator<Item = LogitsRow>>(iter: I) -> Self {
        Self::from_rows(iter.into_iter().collect())
    }
}
impl PartialEq for LogitsBatch {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .zip(other.iter())
                .all(|(left, right)| left == right)
    }
}
impl PartialEq for LogitsRow {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl serde::Serialize for LogitsBatch {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}
impl serde::Serialize for LogitsRow {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(self.as_slice(), serializer)
    }
}
impl Index<usize> for LogitsBatch {
    type Output = [f32];
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("logits row index out of bounds")
    }
}
impl LogitsRow {
    pub fn from_host_row(row: Vec<f32>) -> Self {
        Self {
            storage: RowStorage::Host(row),
        }
    }
    /// Native copies share immutable backing; mutation must acquire uniqueness.
    /// Host producer copies retain their existing ordinary allocation boundary.
    pub fn try_clone(&self) -> anyhow::Result<Self> {
        match &self.storage {
            RowStorage::Host(row) => {
                let mut copy = Vec::new();
                copy.try_reserve_exact(row.len()).map_err(|error| {
                    anyhow::anyhow!("cannot copy {} Host logit values: {error}", row.len())
                })?;
                copy.extend_from_slice(row);
                Ok(Self::from_host_row(copy))
            }
            #[cfg(feature = "native-backend")]
            RowStorage::Array { owner, range } => Ok(Self {
                storage: RowStorage::Array {
                    owner: Arc::clone(owner),
                    range: range.clone(),
                },
            }),
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        match &self.storage {
            RowStorage::Host(row) => row,
            #[cfg(feature = "native-backend")]
            RowStorage::Array { owner, range } => &owner.as_slice()[range.clone()],
        }
    }
    pub fn try_mut(&mut self) -> anyhow::Result<&mut [f32]> {
        match &mut self.storage {
            RowStorage::Host(row) => Ok(row),
            #[cfg(feature = "native-backend")]
            RowStorage::Array { owner, range } => {
                if Arc::get_mut(owner).is_none() {
                    let copy = owner.try_clone_range(range.clone()).with_context(|| format!("admit mutable Native logits row {range:?} while retaining its immutable source"))?;
                    let length = copy.len();
                    *owner = Arc::new(copy);
                    *range = 0..length;
                }
                Ok(&mut Arc::get_mut(owner)
                    .expect("unshared logits row backing")
                    .as_mut_slice()[range.clone()])
            }
        }
    }
}
impl Deref for LogitsRow {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}
impl AsRef<[f32]> for LogitsRow {
    fn as_ref(&self) -> &[f32] {
        self.as_slice()
    }
}
impl std::fmt::Debug for LogitsBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}
impl std::fmt::Debug for LogitsRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

/// Scalar consumption keeps the original row allocation alive until the
/// iterator is dropped; values are copied individually, never bulk-converted.
pub struct LogitsValuesIntoIter {
    row: LogitsRow,
    start: usize,
    end: usize,
}
impl Iterator for LogitsValuesIntoIter {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        if self.start == self.end {
            None
        } else {
            let index = self.start;
            self.start += 1;
            Some(self.row.as_slice()[index])
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len();
        (n, Some(n))
    }
}
impl DoubleEndedIterator for LogitsValuesIntoIter {
    fn next_back(&mut self) -> Option<f32> {
        if self.start == self.end {
            None
        } else {
            self.end -= 1;
            Some(self.row.as_slice()[self.end])
        }
    }
}
impl ExactSizeIterator for LogitsValuesIntoIter {
    fn len(&self) -> usize {
        self.end - self.start
    }
}
impl IntoIterator for LogitsRow {
    type Item = f32;
    type IntoIter = LogitsValuesIntoIter;
    fn into_iter(self) -> Self::IntoIter {
        let end = self.len();
        LogitsValuesIntoIter {
            row: self,
            start: 0,
            end,
        }
    }
}
impl<'a> IntoIterator for &'a LogitsRow {
    type Item = &'a f32;
    type IntoIter = std::slice::Iter<'a, f32>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

pub struct LogitsIntoIter {
    storage: IntoIterStorage,
}
enum IntoIterStorage {
    Host(std::vec::IntoIter<Vec<f32>>),
    Rows(std::vec::IntoIter<LogitsRow>),
    #[cfg(feature = "native-backend")]
    Array {
        owner: Option<Arc<OwnedHostArray<f32>>>,
        start: usize,
        end: usize,
        width: usize,
    },
}
impl Iterator for LogitsIntoIter {
    type Item = LogitsRow;
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.storage {
            IntoIterStorage::Rows(rows) => rows.next(),
            IntoIterStorage::Host(rows) => rows.next().map(|row| LogitsRow {
                storage: RowStorage::Host(row),
            }),
            #[cfg(feature = "native-backend")]
            IntoIterStorage::Array {
                owner,
                start,
                end,
                width,
            } => {
                if *start == *end {
                    return None;
                }
                let index = *start;
                *start += 1;
                let backing = if *start == *end {
                    owner.take()?
                } else {
                    Arc::clone(owner.as_ref()?)
                };
                Some(LogitsRow {
                    storage: RowStorage::Array {
                        owner: backing,
                        range: index * *width..(index + 1) * *width,
                    },
                })
            }
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len();
        (n, Some(n))
    }
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}
impl DoubleEndedIterator for LogitsIntoIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        match &mut self.storage {
            IntoIterStorage::Rows(rows) => rows.next_back(),
            IntoIterStorage::Host(rows) => rows.next_back().map(|row| LogitsRow {
                storage: RowStorage::Host(row),
            }),
            #[cfg(feature = "native-backend")]
            IntoIterStorage::Array {
                owner,
                start,
                end,
                width,
            } => {
                if *start == *end {
                    return None;
                }
                *end -= 1;
                let index = *end;
                let backing = if *start == *end {
                    owner.take()?
                } else {
                    Arc::clone(owner.as_ref()?)
                };
                Some(LogitsRow {
                    storage: RowStorage::Array {
                        owner: backing,
                        range: index * *width..(index + 1) * *width,
                    },
                })
            }
        }
    }
}
impl ExactSizeIterator for LogitsIntoIter {
    fn len(&self) -> usize {
        match &self.storage {
            IntoIterStorage::Host(rows) => rows.len(),
            IntoIterStorage::Rows(rows) => rows.len(),
            #[cfg(feature = "native-backend")]
            IntoIterStorage::Array { start, end, .. } => *end - *start,
        }
    }
}
impl IntoIterator for LogitsBatch {
    type Item = LogitsRow;
    type IntoIter = LogitsIntoIter;
    fn into_iter(self) -> Self::IntoIter {
        LogitsIntoIter {
            storage: match self.storage {
                BatchStorage::Host(rows) => IntoIterStorage::Host(rows.into_iter()),
                BatchStorage::Rows(rows) => IntoIterStorage::Rows(rows.into_iter()),
                #[cfg(feature = "native-backend")]
                BatchStorage::Array { owner, rows, width } => IntoIterStorage::Array {
                    owner: Some(owner),
                    start: 0,
                    end: rows,
                    width,
                },
            },
        }
    }
}

#[cfg(all(test, feature = "native-backend"))]
mod tests {
    use super::*;
    use onnx_runtime_ep_cpu::{CpuBufferMemory, CpuExecutionProvider, CpuMemoryResources};
    use onnx_runtime_ir::{DataType, Graph, static_shape};
    use onnx_runtime_memory_api::{DeviceKey, Tier};
    use onnx_runtime_memory_governor::{
        LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
    };
    use onnx_runtime_session::{HostTensorAllocator, InferenceSession};
    fn context(
        limit: u64,
    ) -> (
        HostTensorAllocator,
        ProcessMemoryManager,
        Arc<LedgerGovernor>,
    ) {
        let manager = ProcessMemoryManager::new().unwrap();
        let books = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
            DeviceKey::HOST,
            0,
            limit,
            0,
        )));
        let domain = CpuBufferMemory::governed(&manager, books.clone()).unwrap();
        let cpu = Arc::new(
            CpuExecutionProvider::initialized_with_resources(
                CpuMemoryResources::default()
                    .with_buffer_memory(domain)
                    .unwrap(),
            )
            .unwrap(),
        );
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let value = graph.create_named_value("input", DataType::Uint8, static_shape([0]));
        graph.add_input(value);
        graph.add_output(value);
        let session = InferenceSession::from_graph_with_provider(
            graph,
            Arc::new(onnx_runtime_loader::WeightStore::new()),
            std::path::Path::new("."),
            cpu,
            None,
        )
        .unwrap();
        let allocator = session.host_tensor_allocator().unwrap();
        drop(session);
        assert_eq!(books.used(Tier::Host), 0);
        (allocator, manager, books)
    }
    fn batch(context: &HostTensorAllocator) -> LogitsBatch {
        let tensor = context
            .from_f32(&[2, 3], &[1., 2., 3., 4., 5., 6.])
            .unwrap();
        LogitsBatch::from_array(OwnedHostArray::from_tensor(tensor).unwrap(), 2, 3).unwrap()
    }
    #[test]
    fn last_consumed_row_takes_final_owner_and_retains_unused_capacity() {
        let (context, manager, books) = context(24);
        let mut row = batch(&context).into_iter().last().unwrap();
        assert_eq!(row.as_slice(), &[4., 5., 6.]);
        row.try_mut().unwrap()[0] = 44.;
        assert_eq!(
            books.used(Tier::Host),
            24,
            "row range retains entire original capacity, without a spare iterator Arc or copy"
        );
        assert_eq!(row.as_slice(), &[44., 5., 6.]);
        drop(row);
        assert_eq!(books.used(Tier::Host), 0);
        assert!(manager.snapshot().unwrap().allocations.is_empty());
    }
    #[test]
    fn queued_shared_row_copy_denial_is_retryable_and_last_row_needs_no_new_grant() {
        let (context, manager, books) = context(36);
        let mut iterator = batch(&context).into_iter();
        let mut first = iterator.next().unwrap();
        let blocker = context.array::<u8>(8).unwrap();
        assert!(first.try_mut().is_err());
        assert_eq!(first.as_slice(), &[1., 2., 3.]);
        assert_eq!(books.used(Tier::Host), 32);
        drop(blocker);
        first.try_mut().unwrap()[0] = 11.;
        assert_eq!(books.used(Tier::Host), 36);
        let mut second = iterator.next().unwrap();
        assert!(iterator.next().is_none());
        assert_eq!(second.as_slice(), &[4., 5., 6.]);
        second.try_mut().unwrap()[0] = 44.;
        assert_eq!(
            books.used(Tier::Host),
            36,
            "iterator must transfer its final Arc even while first row owns a separate charged copy"
        );
        drop(first);
        assert_eq!(books.used(Tier::Host), 24);
        drop(second);
        assert_eq!(manager.process_used(Tier::Host), 0);
    }
    #[test]
    fn retained_immutable_view_forces_real_batch_copy_before_mutation() {
        let (context, manager, books) = context(48);
        let mut original = batch(&context);
        let escaped = original.row_owned(0).unwrap();
        let blocker = context.array::<u8>(8).unwrap();
        assert!(original.row_mut(0).is_err());
        assert_eq!(escaped.as_slice(), &[1., 2., 3.]);
        assert_eq!(original[0], [1., 2., 3.]);
        drop(blocker);
        original.row_mut(0).unwrap()[0] = 99.;
        assert_eq!(escaped.as_slice(), &[1., 2., 3.]);
        assert_eq!(books.used(Tier::Host), 48);
        drop(original);
        assert_eq!(books.used(Tier::Host), 24);
        drop(escaped);
        assert!(manager.snapshot().unwrap().allocations.is_empty());
    }
    #[test]
    fn rejected_row_geometry_releases_transferred_owner() {
        let (context, manager, books) = context(24);
        let array = context.array::<f32>(6).unwrap();
        assert!(LogitsBatch::from_array(array, 3, 3).is_err());
        assert_eq!(books.used(Tier::Host), 0);
        let array = context.array::<f32>(0).unwrap();
        assert!(LogitsBatch::from_array(array, usize::MAX, 2).is_err());
        assert_eq!(manager.process_used(Tier::Host), 0);
    }
    #[test]
    fn owner_backed_rows_serialize_with_the_same_wire_matrix_shape() {
        let (context, manager, books) = context(24);
        let batch = batch(&context);
        assert_eq!(
            serde_json::to_value(&batch).unwrap(),
            serde_json::json!([[1., 2., 3.], [4., 5., 6.]])
        );
        let row = batch.into_iter().last().unwrap();
        assert_eq!(
            serde_json::to_value(&row).unwrap(),
            serde_json::json!([4., 5., 6.])
        );
        assert_eq!(books.used(Tier::Host), 24);
        drop(row);
        assert!(manager.snapshot().unwrap().allocations.is_empty());
    }
}
