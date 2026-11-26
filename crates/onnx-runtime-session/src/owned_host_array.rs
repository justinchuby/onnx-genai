//! Uniquely owned initialized typed arrays over the host allocation domain.
use crate::{HostTensorAllocator, SessionError, Tensor};
use onnx_runtime_ir::DataType;
use std::marker::PhantomData;

mod sealed {
    pub trait Element {}
    macro_rules! elements { ($($t:ty),*) => { $(impl Element for $t {})* }; }
    elements!(
        u8, i8, u16, i16, u32, i32, u64, i64, usize, isize, f32, f64, bool
    );
}
/// Primitive values for which zero initialization is valid.
///
/// This is NOT a Pod/byte-serialization contract: bool is zero-valid but not
/// valid for arbitrary byte patterns. No tuples, structs, padding or user-defined
/// types can implement this sealed contract. Arrays expose typed slices only.
pub trait HostElement: sealed::Element + Copy + Send + Sync + 'static {}
impl<T: sealed::Element + Copy + Send + Sync + 'static> HostElement for T {}

/// Initialized typed storage with one unique backing owner, never a shared Arc.
///
/// Mutable access requires &mut self. There is deliberately no Clone, Vec
/// escape, raw byte view, or allocation-bypassing growth method. Every deep copy
/// and resize first admits new backing through the original allocation domain.
pub struct OwnedHostArray<T: HostElement> {
    backing: Tensor,
    allocator: HostTensorAllocator,
    length: usize,
    element: PhantomData<T>,
}
impl HostTensorAllocator {
    pub fn array<T: HostElement>(&self, length: usize) -> crate::Result<OwnedHostArray<T>> {
        let bytes = length.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| SessionError::Internal(format!(
            "host array length {length} with element width {} overflows; reduce the payload length", std::mem::size_of::<T>()
        )))?;
        let backing = self.zeros(DataType::Uint8, vec![bytes])?;
        if !(backing.as_bytes().as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()) {
            return Err(SessionError::Internal("host array backing is insufficiently aligned; fix the provider's host allocation alignment contract".into()));
        }
        Ok(OwnedHostArray {
            backing,
            allocator: self.clone(),
            length,
            element: PhantomData,
        })
    }
}
impl OwnedHostArray<f32> {
    /// Transfer initialized contiguous F32 backing without copying or refunding
    /// capacity. Borrowed/foreign or strided buffers require an explicit copy.
    pub fn from_tensor(mut tensor: Tensor) -> crate::Result<Self> {
        if tensor.dtype != DataType::Float32 || !tensor.layout.is_contiguous(&tensor.shape) {
            return Err(SessionError::Internal(format!(
                "owned F32 array requires contiguous Float32 tensor backing, got {:?} {:?}; explicitly convert/gather the tensor first",
                tensor.dtype, tensor.layout
            )));
        }
        let bytes = tensor.unique_host_bytes_mut()?;
        let length = bytes.len() / std::mem::size_of::<f32>();
        if !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()) {
            return Err(SessionError::Internal("owned F32 array backing is misaligned; fix the Host provider's allocation contract".into()));
        }
        let allocator = tensor.host_tensor_allocator()?;
        Ok(Self {
            backing: tensor,
            allocator,
            length,
            element: PhantomData,
        })
    }
}
impl<T: HostElement> OwnedHostArray<T> {
    pub fn len(&self) -> usize {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: sealed primitives have valid zero representations and no
        // padding. Constructor checks size/alignment and initializes all bytes;
        // subsequent writes are exclusively through valid typed T values.
        unsafe {
            std::slice::from_raw_parts(self.backing.as_bytes().as_ptr().cast::<T>(), self.length)
        }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let bytes = self
            .backing
            .unique_host_bytes_mut()
            .expect("array constructor provides uniquely owned initialized host bytes");
        // SAFETY: exclusive self borrow and unshared backing; the same checked
        // geometry and valid-value invariants as as_slice. No other owner can
        // construct a mutable or immutable view of this private backing.
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<T>(), self.length) }
    }
    pub fn try_clone(&self) -> crate::Result<Self> {
        let mut copy = self.allocator.array(self.length)?;
        copy.as_mut_slice().copy_from_slice(self.as_slice());
        Ok(copy)
    }
    pub fn try_clone_range(&self, range: std::ops::Range<usize>) -> crate::Result<Self> {
        let source = self.as_slice().get(range.clone()).ok_or_else(|| {
            SessionError::Internal(format!(
                "host array range {range:?} exceeds {} elements; select a valid owned row range",
                self.length
            ))
        })?;
        let mut copy = self.allocator.array(source.len())?;
        copy.as_mut_slice().copy_from_slice(source);
        Ok(copy)
    }
    /// Allocate/copy before publishing, including the honest old+new peak.
    pub fn try_resize(&mut self, length: usize) -> crate::Result<()> {
        if self.length == length {
            return Ok(());
        }
        let mut replacement = self.allocator.array(length)?;
        let keep = self.length.min(length);
        replacement.as_mut_slice()[..keep].copy_from_slice(&self.as_slice()[..keep]);
        *self = replacement;
        Ok(())
    }
}
impl<T: HostElement + std::fmt::Debug> std::fmt::Debug for OwnedHostArray<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_cpu::{CpuBufferMemory, CpuExecutionProvider, CpuMemoryResources};
    use onnx_runtime_memory_api::{DeviceKey, Tier};
    use onnx_runtime_memory_governor::{
        LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
    };
    use std::sync::Arc;
    fn fixture(
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
        let memory = CpuBufferMemory::governed(&manager, books.clone()).unwrap();
        let cpu = Arc::new(
            CpuExecutionProvider::initialized_with_resources(
                CpuMemoryResources::default()
                    .with_buffer_memory(memory)
                    .unwrap(),
            )
            .unwrap(),
        );
        (HostTensorAllocator::new(cpu).unwrap(), manager, books)
    }
    #[test]
    fn grow_denial_preserves_values_and_retry_uses_honest_old_new_peak() {
        let (context, manager, books) = fixture(24);
        let mut values = context.array::<f32>(2).unwrap();
        values.as_mut_slice().copy_from_slice(&[1.25, -7.5]);
        let blocker = context.array::<u8>(16).unwrap();
        assert_eq!(books.used(Tier::Host), 24);
        assert!(values.try_resize(3).is_err());
        assert_eq!(values.as_slice(), &[1.25, -7.5]);
        drop(blocker);
        values.try_resize(3).unwrap();
        assert_eq!(values.as_slice(), &[1.25, -7.5, 0.0]);
        assert_eq!(books.used(Tier::Host), 12);
        let copy = values.try_clone().unwrap();
        assert_eq!(books.used(Tier::Host), 24);
        values.as_mut_slice()[0] = 99.0;
        assert_eq!(copy.as_slice(), &[1.25, -7.5, 0.0]);
        drop(values);
        drop(copy);
        assert_eq!(manager.process_used(Tier::Host), 0);
        assert!(manager.snapshot().unwrap().allocations.is_empty());
    }
    #[test]
    fn bool_values_and_zero_length_keep_valid_initialized_storage_and_capacity() {
        let (context, manager, books) = fixture(9);
        let mut flags = context.array::<bool>(4).unwrap();
        assert_eq!(flags.as_slice(), &[false; 4]);
        flags.as_mut_slice()[2] = true;
        let copied = flags.try_clone_range(1..4).unwrap();
        assert_eq!(copied.as_slice(), &[false, true, false]);
        assert!(flags.try_clone_range(3..5).is_err());
        let empty = context.array::<u64>(0).unwrap();
        assert!(empty.as_slice().is_empty());
        assert_eq!(
            books.used(Tier::Host),
            8,
            "4 flags + 3 copied flags + 1 real empty backing byte"
        );
        drop(flags);
        drop(copied);
        drop(empty);
        assert_eq!(books.used(Tier::Host), 0);
        assert_eq!(manager.process_used(Tier::Host), 0);
    }
    #[test]
    fn explicit_sequence_sources_and_concat_share_real_payload_limit() {
        let (context, manager, books) = fixture(10);
        let first =
            crate::sequence::SeqTensor::from_raw_in(&context, DataType::Uint8, vec![3], &[1, 2, 3])
                .unwrap();
        let second =
            crate::sequence::SeqTensor::from_raw_in(&context, DataType::Uint8, vec![2], &[9, 8])
                .unwrap();
        let sequence =
            crate::sequence::SequenceValue::construct(vec![first.clone(), second.clone()]).unwrap();
        let blocker = context.array::<u8>(1).unwrap();
        assert!(crate::sequence::concat_in(&context, &sequence, 0, false).is_err());
        assert_eq!(books.used(Tier::Host), 6);
        assert_eq!(first.contiguous_bytes().unwrap(), [1, 2, 3]);
        drop(blocker);
        let result = crate::sequence::concat_in(&context, &sequence, 0, false).unwrap();
        assert_eq!(result.contiguous_bytes().unwrap(), [1, 2, 3, 9, 8]);
        assert_eq!(books.used(Tier::Host), 10);
        drop(sequence);
        drop(first);
        drop(second);
        assert_eq!(books.used(Tier::Host), 5);
        assert_eq!(result.contiguous_bytes().unwrap(), [1, 2, 3, 9, 8]);
        drop(result);
        assert_eq!(manager.process_used(Tier::Host), 0);
    }

    #[test]
    fn f32_tensor_transfer_keeps_pointer_and_full_funded_capacity() {
        let (context, manager, books) = fixture(24);
        let tensor = context
            .from_f32(&[2, 3], &[1., 2., 3., 4., 5., 6.])
            .unwrap();
        let pointer = tensor.try_as_slice_f32().unwrap().as_ptr();
        let array = OwnedHostArray::from_tensor(tensor).unwrap();
        assert_eq!(array.as_slice().as_ptr(), pointer);
        assert_eq!(array.as_slice(), &[1., 2., 3., 4., 5., 6.]);
        assert_eq!(books.used(Tier::Host), 24);
        assert!(context.array::<u8>(1).is_err());
        drop(array);
        assert_eq!(books.used(Tier::Host), 0);
        assert_eq!(manager.process_used(Tier::Host), 0);
    }

    #[test]
    fn invalid_strided_span_rejects_before_allocation_with_original_owner_intact() {
        let (context, manager, books) = fixture(32);
        let mut source = context.from_f32(&[2, 2], &[1., 2., 3., 4.]).unwrap();
        for strides in [vec![i64::MAX, i64::MAX], vec![-2, 1], vec![3, 1]] {
            source.layout = onnx_runtime_ir::TensorLayout::strided(strides);
            assert!(context.copy_tensor(&source).is_err());
            assert_eq!(
                books.used(Tier::Host),
                16,
                "invalid source geometry must not publish target backing"
            );
        }
        source.layout = onnx_runtime_ir::TensorLayout::contiguous();
        assert_eq!(source.try_as_slice_f32().unwrap(), &[1., 2., 3., 4.]);
        drop(source);
        assert_eq!(manager.process_used(Tier::Host), 0);
    }
}
