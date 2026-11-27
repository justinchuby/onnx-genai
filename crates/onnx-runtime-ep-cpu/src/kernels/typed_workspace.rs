//! Borrowed initialized typed regions within one existing Host execution lane.
//! No backing owner, pool, admission policy or kernel geometry lives here.

use std::{alloc::Layout, marker::PhantomData};

use onnx_runtime_ep_api::{EpError, Result, WorkspaceLayout, WorkspaceRequirement, WorkspaceView};

pub(super) struct TypedWorkspace<T: Copy + Default, const N: usize> {
    op: &'static str,
    layout: WorkspaceLayout<N>,
    element: PhantomData<T>,
}

impl<T: Copy + Default, const N: usize> TypedWorkspace<T, N> {
    pub(super) fn new(op: &'static str, buffers: [(&str, usize); N]) -> Result<Self> {
        if std::mem::size_of::<T>() == 0 {
            return Err(EpError::KernelFailed(format!(
                "{op}: zero-sized workspace element {} is unsupported; use a nonzero-sized storage type",
                std::any::type_name::<T>()
            )));
        }
        let mut regions = [Layout::new::<[T; 0]>(); N];
        for (region, (name, len)) in regions.iter_mut().zip(buffers) {
            *region = Layout::array::<T>(len).map_err(|_| EpError::KernelFailed(format!(
                "{op}: `{name}` workspace buffer of {len} {} elements exceeds addressable capacity; reduce the requested buffer length", std::any::type_name::<T>()
            )))?;
        }
        let layout = WorkspaceLayout::new(regions).map_err(|error| {
            EpError::KernelFailed(format!("{op}: cannot plan typed workspace: {error}"))
        })?;
        Ok(Self {
            op,
            layout,
            element: PhantomData,
        })
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
        execute: impl for<'scratch> FnOnce([&'scratch mut [T]; N]) -> Result<R>,
    ) -> Result<R> {
        // Validate the whole layout before touching any bytes or calling user code.
        let regions = self
            .layout
            .bind(workspace)
            .map_err(|error| EpError::KernelFailed(format!("{}: {error}", self.op)))?;
        let buffers = regions.map(|region| {
            // Length is derived from the checked typed layout, never duplicated state.
            let len = region.bytes() / std::mem::size_of::<T>();
            let pointer = if len == 0 {
                std::ptr::NonNull::<T>::dangling().as_ptr()
            } else {
                region.ptr().0.cast::<T>()
            };
            // SAFETY: caller proves exclusive live Host storage; checked layout
            // proves alignment and disjointness. Initialize fresh raw backing
            // before forming typed references. Empty slices use aligned dangling.
            unsafe {
                for index in 0..len {
                    pointer.add(index).write(T::default());
                }
                std::slice::from_raw_parts_mut(pointer, len)
            }
        });
        execute(buffers)
    }
}

/// Typed payloads plus route indices within the same admitted Host lane.
/// This composes layouts only: no allocation, ownership or admission policy.
pub(super) struct ValueIndexWorkspace<T: Copy + Default, const N: usize> {
    op: &'static str,
    values: TypedWorkspace<T, N>,
    indices: TypedWorkspace<usize, 1>,
    layout: WorkspaceLayout<2>,
}
impl<T: Copy + Default, const N: usize> ValueIndexWorkspace<T, N> {
    pub(super) fn new(
        op: &'static str,
        values: [(&str, usize); N],
        indices: (&str, usize),
    ) -> Result<Self> {
        let values = TypedWorkspace::new(op, values)?;
        let indices = TypedWorkspace::new(op, [indices])?;
        let region = |r: WorkspaceRequirement| {
            Layout::from_size_align(r.bytes as usize, r.alignment).map_err(|e| {
                EpError::KernelFailed(format!(
                    "{op}: invalid workspace sublayout: {e}; reduce prepared dimensions"
                ))
            })
        };
        let layout = WorkspaceLayout::new([
            region(values.requirement())?,
            region(indices.requirement())?,
        ])
        .map_err(|e| {
            EpError::KernelFailed(format!("{op}: cannot combine value/index workspace: {e}"))
        })?;
        Ok(Self {
            op,
            values,
            indices,
            layout,
        })
    }
    pub(super) fn requirement(&self) -> WorkspaceRequirement {
        WorkspaceRequirement {
            bytes: self.layout.bytes() as u64,
            alignment: self.layout.alignment(),
            ..WorkspaceRequirement::NONE
        }
    }
    /// # Safety
    /// Caller lends exclusive live Host backing, disjoint from all tensors.
    /// Full binding precedes initialization; scratch cannot escape the closure.
    pub(super) unsafe fn with_buffers<R>(
        &self,
        view: Option<WorkspaceView>,
        body: impl for<'a> FnOnce([&'a mut [T]; N], &'a mut [usize]) -> Result<R>,
    ) -> Result<R> {
        let [values, indices] = self
            .layout
            .bind(view)
            .map_err(|e| EpError::KernelFailed(format!("{}: {e}", self.op)))?;
        // SAFETY: complete outer layout validated before either subregion is
        // initialized; caller proves lifetime, exclusivity and Host access.
        unsafe {
            self.values.with_buffers(Some(values), |f| {
                self.indices.with_buffers(Some(indices), |[i]| body(f, i))
            })
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
        let plan = TypedWorkspace::<f32, _>::new("DFT", [("Fourier", 3); 4]).unwrap();
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
        let plan = TypedWorkspace::<f32, _>::new("STFT", [("inactive Fourier", 0); 4]).unwrap();
        assert_eq!(plan.requirement().bytes, 0);
        // SAFETY: no memory is accessed for empty regions.
        unsafe {
            plan.with_buffers(None, |buffers| {
                assert!(buffers.iter().all(|buffer| buffer.is_empty()));
                Ok(())
            })
            .unwrap();
        }
        assert!(TypedWorkspace::<f32, _>::new("DFT", [("Fourier", usize::MAX); 4]).is_err());
        assert!(
            TypedWorkspace::<f32, _>::new("DFT", [("Fourier", isize::MAX as usize / 16 + 1); 4])
                .is_err()
        );
    }
    #[test]
    fn unequal_regions_and_empty_holes_share_one_exact_prefix() {
        let plan = TypedWorkspace::<f32, _>::new(
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
        let plan = TypedWorkspace::<f32, _>::new("test op", [("first", 1), ("second", 3)]).unwrap();
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
        let plan = TypedWorkspace::<f32, _>::new("test op", [("Y", 2)]).unwrap();
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
        let empty = TypedWorkspace::<f32, _>::new("empty", []).unwrap();
        assert_eq!(empty.requirement(), WorkspaceRequirement::NONE);
        // SAFETY: a zero-region plan accesses no backing.
        unsafe {
            empty.with_buffers(None, |[]| Ok(())).unwrap();
        }
    }
    #[test]
    fn f64_layout_rejects_f32_sized_backing_before_borrow() {
        let plan =
            TypedWorkspace::<f64, _>::new("DFT", [("real", 2), ("empty", 0), ("imaginary", 1)])
                .unwrap();
        let mut backing = [17.0f64; 4];
        let ptr = DevicePtrMut(backing.as_mut_ptr().cast());
        // SAFETY: this test presents only a short prefix of an exclusive Host array.
        let result = unsafe {
            plan.with_buffers(Some(WorkspaceView::new(ptr, 12)), |_| -> Result<()> {
                panic!("short backing must not reach the consumer")
            })
        };
        assert!(result.is_err());
        assert_eq!(backing, [17.0; 4]);
    }

    #[test]
    fn f64_regions_preserve_eight_byte_layout_and_fresh_initialization() {
        let plan =
            TypedWorkspace::<f64, 2>::new("LpNormalization", [("Y", 3), ("empty", 0)]).unwrap();
        assert_eq!(
            (plan.requirement().bytes, plan.requirement().alignment),
            (24, 8)
        );
        let mut fresh = [MaybeUninit::<f64>::uninit(); 3];
        let view = WorkspaceView::new(DevicePtrMut(fresh.as_mut_ptr().cast()), 24);
        unsafe {
            plan.with_buffers(Some(view), |[y, empty]| {
                assert_eq!(y, &[0.; 3]);
                assert!(empty.is_empty());
                y.copy_from_slice(&[1.0000000000000002, 2., 3.]);
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(unsafe { fresh[0].assume_init() }, 1.0000000000000002);
        let mut backing = [0x7ff8123456789012u64; 5];
        let view = WorkspaceView::new(
            DevicePtrMut(unsafe { backing.as_mut_ptr().add(1).cast() }),
            24,
        );
        unsafe {
            plan.with_buffers(Some(view), |[y, _]| {
                y.fill(4.);
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(backing[0], 0x7ff8123456789012);
        assert_eq!(backing[4], backing[0]);
        assert_eq!(&backing[1..4], &[4f64.to_bits(); 3]);
    }
    #[test]
    fn typed_initialization_uses_default_instead_of_assuming_zero_bits() {
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[repr(C, align(16))]
        struct Element(u64, u64);
        impl Default for Element {
            fn default() -> Self {
                Self(7, 11)
            }
        }
        let plan = TypedWorkspace::<Element, 1>::new("typed test", [("data", 2)]).unwrap();
        assert_eq!(
            (plan.requirement().bytes, plan.requirement().alignment),
            (32, 16)
        );
        let mut fresh = [MaybeUninit::<Element>::uninit(); 2];
        unsafe {
            plan.with_buffers(
                Some(WorkspaceView::new(
                    DevicePtrMut(fresh.as_mut_ptr().cast()),
                    32,
                )),
                |[data]| {
                    assert_eq!(data, &[Element(7, 11); 2]);
                    Ok(())
                },
            )
            .unwrap();
        }
        assert!(
            TypedWorkspace::<(), 1>::new("typed test", [("data", 3)])
                .err()
                .unwrap()
                .to_string()
                .contains("zero-sized")
        );
    }
    #[test]
    fn mixed_regions_initialize_fresh_backing_and_refuse_incomplete_outer_layout() {
        let plan = ValueIndexWorkspace::<f32, _>::new(
            "mixed fixture",
            [("values", 3), ("empty", 0)],
            ("indices", 3),
        )
        .unwrap();
        let req = plan.requirement();
        let mut fresh = [MaybeUninit::<u128>::uninit(); 8];
        let lane = WorkspaceView::new(DevicePtrMut(fresh.as_mut_ptr().cast()), req.bytes as usize);
        // SAFETY: exclusive aligned Host backing covers the complete layout.
        unsafe {
            plan.with_buffers(Some(lane), |[values, empty], indices| {
                assert_eq!(values, [0.; 3]);
                assert!(empty.is_empty());
                assert_eq!(indices, [0; 3]);
                Ok(())
            })
        }
        .unwrap();
        let mut poison = [0xaaaaaaaaaaaaaaaau128; 8];
        let lane = WorkspaceView::new(
            DevicePtrMut(poison.as_mut_ptr().cast()),
            req.bytes as usize - 1,
        );
        // SAFETY: lane names live exclusive backing; incomplete capacity must
        // refuse before any typed subregion is initialized or body is called.
        assert!(
            unsafe {
                plan.with_buffers(Some(lane), |_, _| -> Result<()> {
                    panic!("short outer layout reached user code")
                })
            }
            .is_err()
        );
        assert!(poison.iter().all(|&v| v == 0xaaaaaaaaaaaaaaaau128));
    }
}
