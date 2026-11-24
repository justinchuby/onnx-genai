//! Borrowed validation status for packing and routed projection kernels.
//! The four-byte region belongs to the caller lane, never to an individual op.

use std::alloc::Layout;

use onnx_runtime_ep_api::{
    EpError, Result, WorkspaceLayout, WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::packed_routed::Op;

pub(super) fn requirement(needed: bool) -> WorkspaceRequirement {
    if needed {
        WorkspaceRequirement {
            bytes: size_of::<u32>() as u64,
            alignment: align_of::<u32>(),
            lifetime: WorkspaceLifetime::SessionPersistent,
        }
    } else {
        WorkspaceRequirement::NONE
    }
}

pub(super) fn status(workspace: Option<WorkspaceView>, op: &str) -> Result<u64> {
    let layout = WorkspaceLayout::new([Layout::new::<u32>()])?;
    let [flag] = layout.bind(workspace).map_err(|cause| {
        EpError::KernelFailed(format!("cuda_ep {op} validation status: {cause}"))
    })?;
    Ok(flag.ptr().0 as usize as u64)
}

pub(super) fn packing_needed(op: Op, output: Option<&[usize]>) -> bool {
    // Unpack validates its device-side count even when the output is empty.
    !matches!(op, Op::Pack(..)) || output.is_none_or(|shape| !shape.contains(&0))
}

pub(super) fn routing_needed(indices: Option<&[usize]>, scales: Option<&[usize]>) -> bool {
    // Invalid expert IDs must be checked even for a zero-width output. Affine
    // projections also validate the complete scale bank, even with zero routes.
    indices.is_none_or(|shape| !shape.contains(&0))
        || scales.is_some_and(|shape| !shape.contains(&0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::DevicePtrMut;

    #[test]
    fn validation_status_uses_one_borrowed_aligned_region_and_preserves_its_contents() {
        let mut words = [0xdead_beefu32, 0x1234_5678];
        let pointer = words.as_mut_ptr().cast();
        let view = WorkspaceView::new(DevicePtrMut(pointer), size_of_val(&words));
        assert_eq!(
            status(Some(view), "GroupedMatMul").unwrap(),
            pointer as usize as u64
        );
        assert_eq!(words, [0xdead_beef, 0x1234_5678]);
        assert_eq!(requirement(true).bytes, 4);
        assert_eq!(requirement(true).alignment, 4);
        assert_eq!(
            requirement(true).lifetime,
            WorkspaceLifetime::SessionPersistent
        );
        assert_eq!(requirement(false), WorkspaceRequirement::NONE);
    }

    #[test]
    fn validation_status_rejects_missing_short_misaligned_null_or_overflowing_backing() {
        let mut words = [0u32; 2];
        let pointer = words.as_mut_ptr().cast();
        for view in [
            None,
            Some(WorkspaceView::new(DevicePtrMut(pointer), 3)),
            Some(WorkspaceView::new(
                DevicePtrMut(pointer.wrapping_byte_add(1)),
                4,
            )),
            Some(WorkspaceView::new(DevicePtrMut(std::ptr::null_mut()), 4)),
            Some(WorkspaceView::new(
                DevicePtrMut((usize::MAX - 3) as *mut _),
                4,
            )),
        ] {
            let error = status(view, "GroupedMatMul").unwrap_err().to_string();
            assert!(error.contains("GroupedMatMul validation status"), "{error}");
            assert!(error.contains("prepare and pass"), "{error}");
        }
    }

    #[test]
    fn packing_empty_decision_keeps_unpack_count_validation() {
        use onnx_runtime_ir::{Attribute, Node, NodeId};
        for name in ["Pack", "Unpack"] {
            let mut node = Node::new(NodeId(0), name, vec![], vec![]);
            node.attributes.insert("bits".into(), Attribute::Int(2));
            let op = Op::from_node(&node).unwrap();
            assert!(packing_needed(op, None));
            assert!(packing_needed(op, Some(&[4])));
            assert_eq!(packing_needed(op, Some(&[0])), name == "Unpack");
        }
    }

    #[test]
    fn routing_empty_decision_preserves_zero_width_route_and_scale_validation() {
        assert!(!routing_needed(Some(&[0, 2]), None));
        assert!(!routing_needed(Some(&[4, 0]), Some(&[0, 8, 4])));
        assert!(routing_needed(Some(&[2, 3]), None));
        assert!(routing_needed(Some(&[0, 2]), Some(&[4, 8, 4])));
        assert!(routing_needed(None, None));
    }
}
