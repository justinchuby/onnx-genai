//! A zero-initialised tensor is actually zero, whatever the allocator returns.
//!
//! `zeros_in` used to build a zeroed `Vec` and copy it into the buffer, which
//! made the guarantee obvious and the cost double: the same bytes allocated
//! twice with a memcpy between them, on the path dispatch uses to materialise
//! every kernel output and a hybrid decoder uses to seed a `conv_state` per
//! recurrent layer per sequence.
//!
//! It now zeroes the buffer in place. That is cheaper and just as correct, but
//! it moves the guarantee from "obvious" to "asserted" — so it has to be
//! asserted against an allocator that hands back *dirty* memory. Freshly mapped
//! pages are usually zero already, so a test using the default allocator would
//! pass whether or not the zeroing happened.

use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::Arc;

use onnx_runtime_eager::Tensor;
use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::DataType;
use onnx_runtime_memory_governor::{DeviceAllocator, DeviceKey, MemoryError};

/// Hands back memory filled with a non-zero pattern.
///
/// A real allocator is entitled to do this — `std::alloc::alloc` returns
/// uninitialised bytes — so anything that needs zeros must write them.
#[derive(Debug)]
struct DirtyAllocator;

const POISON: u8 = 0xAB;

impl DeviceAllocator for DirtyAllocator {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        let layout = Layout::from_size_align(bytes.max(1), align).map_err(|_| {
            MemoryError::InvalidRequest {
                tier: "host",
                requested: bytes as u64,
                reason: "the test asked for a layout that is not valid",
            }
        })?;
        // SAFETY: `layout` has a non-zero size and a valid alignment.
        let ptr = unsafe { std::alloc::alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or(MemoryError::InvalidRequest {
            tier: "host",
            requested: bytes as u64,
            reason: "the system allocator refused the test allocation",
        })?;
        // SAFETY: `ptr` is a fresh allocation of at least `bytes.max(1)` bytes.
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), POISON, bytes.max(1)) };
        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        let Ok(layout) = Layout::from_size_align(bytes.max(1), align) else {
            return;
        };
        // SAFETY: delegated to this method's contract.
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
    }

    fn device(&self) -> DeviceKey {
        DeviceKey::HOST
    }
}

#[test]
fn a_zeroed_tensor_is_zero_even_when_the_allocator_returns_dirty_memory() {
    let provider: Arc<dyn ExecutionProvider> =
        Arc::new(CpuExecutionProvider::new().with_memory(Arc::new(DirtyAllocator)));

    // Rank 3 with a static middle axis, the shape a hybrid decoder's
    // `conv_state` has.
    let tensor = Tensor::zeros_in(Arc::clone(&provider), DataType::Float32, vec![1, 64, 3])
        .expect("a zeroed tensor");

    let values = tensor.to_vec_f32();
    assert_eq!(values.len(), 64 * 3);
    assert!(
        values.iter().all(|value| *value == 0.0),
        "the allocator's poison survived into a tensor that claims to be zeroed"
    );
}

#[test]
fn a_zeroed_tensor_of_no_elements_is_still_a_valid_tensor() {
    let provider: Arc<dyn ExecutionProvider> =
        Arc::new(CpuExecutionProvider::new().with_memory(Arc::new(DirtyAllocator)));

    // A growable KV input starts with an empty sequence axis, so this shape is
    // reached on every hybrid decoder's first step.
    let tensor = Tensor::zeros_in(Arc::clone(&provider), DataType::Float32, vec![1, 8, 0, 4])
        .expect("an empty tensor is allocatable");
    assert_eq!(tensor.shape(), &[1, 8, 0, 4]);
    assert!(tensor.to_vec_f32().is_empty());
}
