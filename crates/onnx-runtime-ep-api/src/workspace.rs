//! Checked partitioning of executor-owned scratch without allocating or charging.
use crate::{DevicePtrMut, EpError, Result, WorkspaceView};
use std::alloc::Layout;

/// A fixed set of disjoint regions within one executor workspace.
///
/// This describes byte ranges only. Binding produces raw device views, never
/// Rust references: the caller still establishes device accessibility, lifetime
/// and exclusive access before dereferencing them.
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceLayout<const N: usize> {
    offsets: [usize; N],
    regions: [Layout; N],
    layout: Layout,
}

impl<const N: usize> WorkspaceLayout<N> {
    pub fn new(regions: [Layout; N]) -> Result<Self> {
        let mut layout = Layout::new::<()>();
        let mut offsets = [0; N];
        for (index, region) in regions.iter().enumerate() {
            let (next, offset) = layout.extend(*region).map_err(|_| {
                EpError::KernelFailed(format!(
                    "workspace region {index} ({} bytes, alignment {}) exceeds addressable layout capacity; reduce the prepared shape",
                    region.size(), region.align()
                ))
            })?;
            layout = next;
            offsets[index] = offset;
        }
        Ok(Self {
            offsets,
            regions,
            layout,
        })
    }

    pub fn bytes(&self) -> usize {
        self.layout.size()
    }
    pub fn alignment(&self) -> usize {
        self.layout.align()
    }

    /// Validate the complete layout once and split it into nonoverlapping raw
    /// views. Zero-sized regions need no readable or writable backing.
    pub fn bind(&self, workspace: Option<WorkspaceView>) -> Result<[WorkspaceView; N]> {
        let view =
            workspace.unwrap_or_else(|| WorkspaceView::new(DevicePtrMut(std::ptr::null_mut()), 0));
        let address = view.ptr().0 as usize;
        if self.bytes() != 0
            && (view.bytes() < self.bytes()
                || address == 0
                || !address.is_multiple_of(self.alignment())
                || address.checked_add(self.bytes()).is_none())
        {
            return Err(EpError::KernelFailed(format!(
                "workspace needs {} bytes aligned to {}, received {} bytes at {address:#x}; prepare and pass the kernel's workspace requirement before execution",
                self.bytes(),
                self.alignment(),
                view.bytes()
            )));
        }
        Ok(std::array::from_fn(|index| {
            WorkspaceView::new(
                DevicePtrMut(
                    view.ptr()
                        .0
                        .cast::<u8>()
                        .wrapping_add(self.offsets[index])
                        .cast(),
                ),
                self.regions[index].size(),
            )
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitions_mixed_alignment_without_overlap() {
        let layout = WorkspaceLayout::new([
            Layout::from_size_align(3, 1).unwrap(),
            Layout::from_size_align(8, 8).unwrap(),
        ])
        .unwrap();
        assert_eq!((layout.bytes(), layout.alignment()), (16, 8));
        let mut backing = [0u64; 2];
        let view = WorkspaceView::new(DevicePtrMut(backing.as_mut_ptr().cast()), 16);
        let [a, b] = layout.bind(Some(view)).unwrap();
        assert_eq!(a.bytes(), 3);
        assert_eq!(b.bytes(), 8);
        assert_eq!(b.ptr().0 as usize - a.ptr().0 as usize, 8);
    }

    #[test]
    fn rejects_missing_short_misaligned_or_wrapping_backing() {
        let layout = WorkspaceLayout::new([Layout::array::<u64>(2).unwrap()]).unwrap();
        let mut backing = [0u64; 3];
        for view in [
            None,
            Some(WorkspaceView::new(
                DevicePtrMut(backing.as_mut_ptr().cast()),
                8,
            )),
            Some(WorkspaceView::new(
                DevicePtrMut(backing.as_mut_ptr().cast::<u8>().wrapping_add(1).cast()),
                16,
            )),
            Some(WorkspaceView::new(
                DevicePtrMut((usize::MAX - 7) as *mut _),
                16,
            )),
        ] {
            assert!(
                layout
                    .bind(view)
                    .unwrap_err()
                    .to_string()
                    .contains("prepare and pass")
            );
        }
    }

    #[test]
    fn empty_regions_need_no_backing_and_overflow_is_checked() {
        let empty = WorkspaceLayout::new([Layout::array::<f32>(0).unwrap(); 2]).unwrap();
        assert!(
            empty
                .bind(None)
                .unwrap()
                .iter()
                .all(|view| view.bytes() == 0)
        );
        let big = Layout::from_size_align(isize::MAX as usize, 1).unwrap();
        assert!(WorkspaceLayout::new([big, big]).is_err());
    }
}
