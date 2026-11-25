//! Borrowed initialized f32 regions within one existing Host execution lane.
//! No backing owner, pool, admission policy or kernel geometry lives here.

use std::alloc::Layout;

use onnx_runtime_ep_api::{EpError, Result, WorkspaceLayout, WorkspaceRequirement, WorkspaceView};

pub(super) struct FloatWorkspace<const N: usize> {
    op: &'static str,
    layout: WorkspaceLayout<N>,
}

impl<const N: usize> FloatWorkspace<N> {
    pub(super) fn new(op: &'static str, buffers: [(&str, usize); N]) -> Result<Self> {
        let mut regions = [Layout::new::<[f32; 0]>(); N];
        for (region, (name, len)) in regions.iter_mut().zip(buffers) {
            *region = Layout::array::<f32>(len).map_err(|_| EpError::KernelFailed(format!(
                "{op}: `{name}` workspace buffer of {len} f32 elements exceeds addressable capacity; reduce the requested buffer length"
            )))?;
        }
        let layout = WorkspaceLayout::new(regions).map_err(|error| {
            EpError::KernelFailed(format!("{op}: cannot plan f32 workspace: {error}"))
        })?;
        Ok(Self { op, layout })
    }

    pub(super) fn requirement(&self) -> WorkspaceRequirement {
        WorkspaceRequirement {
            bytes: self.layout.bytes() as u64,
            alignment: self.layout.alignment(),
            ..WorkspaceRequirement::NONE
        }
    }

    /// # Safety
    /// The caller proves that nonempty backing is live, Host-accessible and
    /// exclusively writable through the closure, disjoint from all other live
    /// tensors. Layout validation alone cannot establish ownership or lifetime.
    /// The fixed result type prevents scratch references escaping this borrow.
    pub(super) unsafe fn with_buffers<R>(
        &self,
        workspace: Option<WorkspaceView>,
        execute: impl for<'scratch> FnOnce([&'scratch mut [f32]; N]) -> Result<R>,
    ) -> Result<R> {
        // Validate the whole layout before touching any bytes or calling user code.
        let regions = self
            .layout
            .bind(workspace)
            .map_err(|error| EpError::KernelFailed(format!("{}: {error}", self.op)))?;
        let buffers = regions.map(|region| {
            // Length is derived from the checked f32 layout, never duplicated state.
            let len = region.bytes() / std::mem::size_of::<f32>();
            let pointer = if len == 0 {
                std::ptr::NonNull::<f32>::dangling().as_ptr()
            } else {
                region.ptr().0.cast::<f32>()
            };
            // SAFETY: caller proves exclusive live Host storage; checked layout
            // proves alignment and disjointness. Initialize fresh raw backing
            // before forming float references. Empty slices use aligned dangling.
            unsafe {
                pointer.write_bytes(0, len);
                std::slice::from_raw_parts_mut(pointer, len)
            }
        });
        execute(buffers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::DevicePtrMut;
    use std::mem::MaybeUninit;

    #[test]
    fn initializes_fresh_regions_and_touches_only_the_declared_prefix() {
        let plan = FloatWorkspace::new("DFT", [("Fourier", 3); 4]).unwrap();
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
        let plan = FloatWorkspace::new("STFT", [("inactive Fourier", 0); 4]).unwrap();
        assert_eq!(plan.requirement().bytes, 0);
        // SAFETY: no memory is accessed for empty regions.
        unsafe {
            plan.with_buffers(None, |buffers| {
                assert!(buffers.iter().all(|buffer| buffer.is_empty()));
                Ok(())
            })
            .unwrap();
        }
        assert!(FloatWorkspace::new("DFT", [("Fourier", usize::MAX); 4]).is_err());
        assert!(
            FloatWorkspace::new("DFT", [("Fourier", isize::MAX as usize / 16 + 1); 4]).is_err()
        );
    }
    #[test]
    fn unequal_regions_and_empty_holes_share_one_exact_prefix() {
        let plan = FloatWorkspace::new(
            "normalization",
            [("Y", 3), ("omitted stats", 0), ("inv", 2)],
        )
        .unwrap();
        assert_eq!(plan.requirement().bytes, 20);
        assert_eq!(plan.requirement().alignment, 4);
        let poison = 0x7fc0_1234;
        let mut backing = [poison; 9];
        let view = WorkspaceView::new(
            DevicePtrMut(unsafe { backing.as_mut_ptr().add(1).cast() }),
            28,
        );
        // SAFETY: exclusive live Host array covers all regions, including guards.
        unsafe {
            plan.with_buffers(Some(view), |[y, omitted, inv]| {
                assert_eq!((y.len(), omitted.len(), inv.len()), (3, 0, 2));
                assert!(y.iter().chain(inv.iter()).all(|x| x.to_bits() == 0));
                assert!(omitted.as_ptr().is_aligned());
                y.copy_from_slice(&[1., 2., 3.]);
                inv.copy_from_slice(&[4., 5.]);
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(backing[0], poison);
        assert_eq!(
            &backing[1..6],
            &[
                1f32.to_bits(),
                2f32.to_bits(),
                3f32.to_bits(),
                4f32.to_bits(),
                5f32.to_bits()
            ]
        );
        assert!(backing[6..].iter().all(|&bits| bits == poison));
    }

    #[test]
    fn refuses_all_bad_backing_before_initialization_or_callback() {
        let plan = FloatWorkspace::new("test op", [("first", 1), ("second", 3)]).unwrap();
        let mut backing = [0x7fc0_1234u32; 8];
        let ptr = backing.as_mut_ptr().cast::<u8>();
        for bad in [
            None,
            Some(WorkspaceView::new(DevicePtrMut(ptr.cast()), 15)),
            Some(WorkspaceView::new(
                DevicePtrMut(unsafe { ptr.add(1).cast() }),
                16,
            )),
            Some(WorkspaceView::new(
                DevicePtrMut((usize::MAX - 3) as *mut _),
                16,
            )),
        ] {
            let mut called = false;
            // SAFETY: invalid geometry must be rejected without dereferencing it.
            let error = unsafe {
                plan.with_buffers(bad, |_| {
                    called = true;
                    Ok(())
                })
            }
            .unwrap_err();
            assert!(
                error.to_string().contains("test op: ")
                    && error.to_string().contains("workspace needs 16 bytes"),
                "{error}"
            );
            assert!(!called);
            assert!(backing.iter().all(|&bits| bits == 0x7fc0_1234));
        }
    }

    #[test]
    fn failed_consumer_does_not_retain_contents_or_borrows() {
        let plan = FloatWorkspace::new("test op", [("Y", 2)]).unwrap();
        let mut backing = [123f32; 2];
        let view = WorkspaceView::new(DevicePtrMut(backing.as_mut_ptr().cast()), 8);
        // SAFETY: the exclusive Host backing outlives both synchronous borrows.
        let error = unsafe {
            plan.with_buffers(Some(view), |[y]| {
                y.fill(456.);
                Err::<(), _>(EpError::KernelFailed(
                    "consumer rejected a parameter".into(),
                ))
            })
        }
        .unwrap_err();
        assert!(error.to_string().contains("consumer rejected a parameter"));
        assert_eq!(backing, [456.; 2]);
        unsafe {
            plan.with_buffers(Some(view), |[y]| {
                assert!(y.iter().all(|x| x.to_bits() == 0));
                y.fill(789.);
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(backing, [789.; 2]);
        let empty = FloatWorkspace::new("empty", []).unwrap();
        assert_eq!(empty.requirement(), WorkspaceRequirement::NONE);
        // SAFETY: a zero-region plan accesses no backing.
        unsafe {
            empty.with_buffers(None, |[]| Ok(())).unwrap();
        }
    }
}
