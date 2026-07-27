use std::ffi::CString;
use std::ptr;

use onnx_genai_ort_sys as ort;
use onnx_runtime_ir::NodeId;

use super::host::{
    HostGraph, HostNode, HostOpAttr, HostOrtValue, HostSupportInfo, HostTensorTypeAndShapeInfo,
    HostTypeInfo, HostValueInfo, ort_api,
};

pub(super) fn cstring_lossless(
    value: &str,
    field: &str,
    node_id: NodeId,
) -> std::result::Result<CString, String> {
    CString::new(value)
        .map_err(|_| format!("{field} on node {node_id:?} contains an interior NUL byte"))
}

pub(super) fn value_ptr(value: &HostValueInfo) -> *const ort::OrtValueInfo {
    (value as *const HostValueInfo).cast::<ort::OrtValueInfo>()
}

pub(super) fn node_from_ptr<'a>(node: *const ort::OrtNode) -> &'a HostNode {
    // SAFETY: All OrtNode pointers handed to the plugin are casts of HostNode
    // references owned by HostGraph for the duration of the plugin call.
    unsafe { &*(node.cast::<HostNode>()) }
}

pub(super) fn value_from_ptr<'a>(value: *const ort::OrtValueInfo) -> &'a HostValueInfo {
    // SAFETY: All OrtValueInfo pointers handed to the plugin are casts of
    // HostValueInfo references owned by HostGraph for the duration of the call.
    unsafe { &*(value.cast::<HostValueInfo>()) }
}

pub(super) fn type_info_from_ptr<'a>(type_info: *const ort::OrtTypeInfo) -> &'a HostTypeInfo {
    // SAFETY: Type-info pointers come from HostValueInfo.type_info.
    unsafe { &*(type_info.cast::<HostTypeInfo>()) }
}

pub(super) fn tensor_info_from_ptr<'a>(
    info: *const ort::OrtTensorTypeAndShapeInfo,
) -> &'a HostTensorTypeAndShapeInfo {
    // SAFETY: Tensor-shape pointers are either borrowed from HostTypeInfo or
    // allocated by GetTensorTypeAndShape until ReleaseTensorTypeAndShapeInfo.
    unsafe { &*(info.cast::<HostTensorTypeAndShapeInfo>()) }
}

pub(super) fn ort_value_from_ptr<'a>(value: *const ort::OrtValue) -> &'a HostOrtValue {
    // SAFETY: OrtValue pointers returned by this Stage-1 bridge are casts of
    // HostOrtValue boxes owned by HostValueInfo initializers.
    unsafe { &*(value.cast::<HostOrtValue>()) }
}

pub(super) fn ort_value_from_mut_ptr<'a>(value: *mut ort::OrtValue) -> &'a mut HostOrtValue {
    // SAFETY: Mutable OrtValue pointers returned for outputs are casts of
    // HostOrtValue entries uniquely borrowed through HostKernelContext.
    unsafe { &mut *(value.cast::<HostOrtValue>()) }
}

pub(super) fn attr_from_ptr<'a>(attr: *const ort::OrtOpAttr) -> &'a HostOpAttr {
    // SAFETY: Attribute pointers point into HostNode.attrs, stable for the call.
    unsafe { &*(attr.cast::<HostOpAttr>()) }
}

pub(super) fn graph_from_ptr<'a>(graph: *const ort::OrtGraph) -> &'a HostGraph {
    // SAFETY: OrtGraph pointers are casts of HostGraph references owned by the
    // caller and live for the duration of plugin capability discovery.
    unsafe { &*(graph.cast::<HostGraph>()) }
}

/// Upper bound on devices accepted from a plugin factory.
///
/// A fixed buffer keeps the call allocation-free; the API reports how many it
/// wanted, and taking the first few of an implausibly long list is better than
/// trusting a length to size a heap allocation.
pub(super) const MAX_PLUGIN_EP_DEVICES: usize = 16;

/// Upper bound on threads holding per-thread plugin state for one kernel.
///
/// Generous against a decode loop, which uses a handful of threads, and small
/// enough that a caller spawning a thread per call is reported rather than
/// silently accumulating plugin resources.
pub(super) const MAX_PLUGIN_THREAD_STATES: usize = 64;

/// The hardware device an `OrtEpDevice` describes.
///
/// # Safety
///
/// `device` must be a live `OrtEpDevice` returned by a plugin factory.
pub(super) unsafe fn ep_device_hardware(
    device: *mut ort::OrtEpDevice,
) -> *const ort::OrtHardwareDevice {
    // SAFETY: `ort_api` returns our own process-lifetime vtable, and the
    // accessor is a pure read on the plugin-owned device the caller vouched for.
    unsafe {
        match (*ort_api()).EpDevice_Device {
            Some(accessor) => accessor(device),
            None => ptr::null(),
        }
    }
}

/// The provider metadata an `OrtEpDevice` carries.
///
/// # Safety
///
/// `device` must be a live `OrtEpDevice` returned by a plugin factory.
pub(super) unsafe fn ep_device_metadata(
    device: *mut ort::OrtEpDevice,
) -> *const ort::OrtKeyValuePairs {
    // SAFETY: as above -- our own vtable, and a pure read on a live device.
    unsafe {
        match (*ort_api()).EpDevice_EpMetadata {
            Some(accessor) => accessor(device),
            None => ptr::null(),
        }
    }
}

pub(super) fn support_from_ptr<'a>(
    support: *mut ort::OrtEpGraphSupportInfo,
) -> &'a mut HostSupportInfo {
    // SAFETY: The support pointer passed to the plugin is a mutable HostSupportInfo.
    unsafe { &mut *(support.cast::<HostSupportInfo>()) }
}
