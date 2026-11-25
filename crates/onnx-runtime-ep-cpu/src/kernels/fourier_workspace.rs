//! Checked, borrowed Fourier scratch. No backing owner, pool or admission policy.

use std::alloc::Layout;

use onnx_runtime_ep_api::{
    EpError, Result, TensorView, WorkspaceLayout, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::DataType;

pub(super) struct FourierWorkspace<T = f32> {
    op: &'static str,
    len: usize,
    layout: WorkspaceLayout<4>,
    element: std::marker::PhantomData<T>,
}

impl<T: Copy + Default> FourierWorkspace<T> {
    pub(super) fn new(op: &'static str, length: usize, active: bool) -> Result<Self> {
        let len = if active { length } else { 0 };
        let region = Layout::array::<T>(len).map_err(|_| EpError::KernelFailed(format!(
            "{op}: transform length {length} exceeds addressable scratch for {}; reduce the transform length",
            std::any::type_name::<T>(),
        )))?;
        let layout = WorkspaceLayout::new([region; 4]).map_err(|error| {
            EpError::KernelFailed(format!("{op}: cannot plan four Fourier buffers: {error}"))
        })?;
        Ok(Self {
            op,
            len,
            layout,
            element: std::marker::PhantomData,
        })
    }

    pub(super) fn requirement(&self) -> WorkspaceRequirement {
        WorkspaceRequirement {
            bytes: self.layout.bytes() as u64,
            alignment: self.layout.alignment(),
            ..WorkspaceRequirement::NONE
        }
    }

    /// The caller proves Host accessibility and exclusive live backing through
    /// the closure. Checked partitioning alone is not that proof. References
    /// cannot escape this higher-ranked closure through its fixed result type.
    pub(super) unsafe fn with_buffers<R>(
        &self,
        workspace: Option<WorkspaceView>,
        execute: impl FnOnce([&mut [T]; 4]) -> Result<R>,
    ) -> Result<R> {
        let regions = self
            .layout
            .bind(workspace)
            .map_err(|error| EpError::KernelFailed(format!("{}: {error}", self.op)))?;
        let buffers = regions.map(|region| {
            let pointer = if self.len == 0 {
                std::ptr::NonNull::<T>::dangling().as_ptr()
            } else {
                region.ptr().0.cast::<T>()
            };
            // SAFETY: caller supplies Host ownership and lifetime; the layout
            // proves disjoint aligned regions. Initialize before forming
            // references, including when the underlying allocation is fresh.
            unsafe {
                for i in 0..self.len {
                    pointer.add(i).write(T::default());
                }
                std::slice::from_raw_parts_mut(pointer, self.len)
            }
        });
        execute(buffers)
    }
}

/// Read one rank-zero Host-accessible integer without materializing a Vec.
pub(super) fn scalar_i64(op: &str, name: &str, input: &TensorView<'_>) -> Result<i64> {
    if input.is_absent() {
        return Err(EpError::KernelFailed(format!(
            "{op}: required `{name}` input is absent"
        )));
    }
    if !input.shape.is_empty() {
        return Err(EpError::KernelFailed(format!(
            "{op}: `{name}` must be a scalar, got shape {:?}",
            input.shape
        )));
    }
    input.validate()?;
    if !input.device.is_host_accessible() {
        return Err(EpError::KernelFailed(format!(
            "{op}: `{name}` must be a Host scalar, got device {:?}; copy it to CPU before execution",
            input.device
        )));
    }
    // SAFETY: the owner supplies one readable Host element; validation checks
    // rank/offset and data_ptr applies that offset. Packed inputs may be unaligned.
    unsafe {
        match input.dtype {
            DataType::Int32 => Ok(input.data_ptr::<i32>().read_unaligned() as i64),
            DataType::Int64 => Ok(input.data_ptr::<i64>().read_unaligned()),
            dtype => Err(EpError::KernelFailed(format!(
                "{op}: `{name}` must be int32 or int64, got {dtype:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::DevicePtrMut;
    use std::mem::MaybeUninit;

    #[test]
    fn initializes_fresh_regions_and_touches_only_the_declared_prefix() {
        let plan = FourierWorkspace::<f32>::new("DFT", 3, true).unwrap();
        assert_eq!(plan.requirement().bytes, 48);
        let mut fresh = [MaybeUninit::<f32>::uninit(); 12];
        let workspace = WorkspaceView::new(DevicePtrMut(fresh.as_mut_ptr().cast()), 48);
        // SAFETY: this stack allocation is aligned, exclusive Host memory.
        unsafe {
            plan.with_buffers(Some(workspace), |buffers| {
                assert!(
                    buffers
                        .iter()
                        .all(|buffer| buffer.iter().all(|value| value.to_bits() == 0))
                );
                Ok(())
            })
            .unwrap();
        }
        let poison = 0x7fc0_1234u32;
        let mut backing = [poison; 18];
        let workspace = WorkspaceView::new(
            DevicePtrMut(unsafe { backing.as_mut_ptr().add(1).cast() }),
            64,
        );
        // SAFETY: the complete four-region prefix is within this exclusive array.
        unsafe {
            plan.with_buffers(Some(workspace), |buffers| {
                for (index, buffer) in buffers.into_iter().enumerate() {
                    assert!(buffer.iter().all(|value| value.to_bits() == 0));
                    buffer.fill((index + 1) as f32);
                }
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(backing[0], poison);
        assert!(backing[13..].iter().all(|&bits| bits == poison));
        for index in 0..4 {
            assert!(
                backing[1 + index * 3..1 + (index + 1) * 3]
                    .iter()
                    .all(|&bits| bits == ((index + 1) as f32).to_bits())
            );
        }
    }

    #[test]
    fn zero_work_and_overflow_do_not_require_an_allocation() {
        let plan = FourierWorkspace::<f32>::new("STFT", usize::MAX, false).unwrap();
        assert_eq!(plan.requirement().bytes, 0);
        // SAFETY: no memory is accessed for empty regions.
        unsafe {
            plan.with_buffers(None, |buffers| {
                assert!(buffers.iter().all(|buffer| buffer.is_empty()));
                Ok(())
            })
            .unwrap();
        }
        assert!(FourierWorkspace::<f32>::new("DFT", usize::MAX, true).is_err());
        assert!(FourierWorkspace::<f32>::new("DFT", isize::MAX as usize / 16 + 1, true).is_err());
    }
}
