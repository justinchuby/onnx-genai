//! nxrt ABI vtable definitions.
//!
//! Each object type (factory, EP, kernel, allocator) is represented as a
//! `#[repr(C)]` struct with function pointers -- a C vtable. This avoids
//! fat-pointer / trait-object ABI fragility across the dynamic boundary.
//!
//! # Ownership rules
//!
//! - **Factory**: created by `NxrtCreateEpFactories`, owned by the host.
//!   Released by calling `factory.release(factory.ctx)`.
//! - **EP**: created by `factory.create_ep(...)`, owned by the host.
//!   Released by calling `ep.release(ep.ctx)`.
//! - **Kernel**: created by `ep.compile(...)`, owned by the host.
//!   Released by calling `kernel.release(kernel.ctx)`.
//! - **Allocator**: obtained from `ep.get_allocator(...)`, owned by the host.
//!   Released by calling `allocator.release(allocator.ctx)`.
//!
//! # Lifetime rules
//!
//! - Borrowed pointers (e.g. `NxrtTensorDesc.dims`) are valid only for the
//!   duration of the callback frame they appear in, unless explicitly noted.
//! - Graph/node handles passed to `get_capability` do not outlive the call.
//! - The `name` pointer in `NxrtEpVtable` is valid for the EP's lifetime.

use std::ffi::c_void;
use std::ptr;

use crate::NxrtStatus;
use crate::status::NxrtStatusCode;

/// Map a `DeviceType` to a u32 discriminant for the C ABI.
fn device_type_to_u32(dt: onnx_runtime_ir::DeviceType) -> u32 {
    use onnx_runtime_ir::DeviceType;
    match dt {
        DeviceType::Cpu => 0,
        DeviceType::Cuda => 1,
        DeviceType::Rocm => 2,
        DeviceType::CoreMl => 3,
        DeviceType::Mlx => 4,
        DeviceType::WebGpu => 5,
        DeviceType::Qnn => 6,
        DeviceType::OpenVino => 7,
        DeviceType::Vulkan => 8,
        DeviceType::Custom(id) => 0x1000 | id,
    }
}

// --- Tensor descriptor -------------------------------------------------------

/// Describes a tensor's shape and dtype across the C ABI.
///
/// # Ownership
///
/// This is a **borrowed view**: `dims` points into caller-owned memory valid
/// only for the current callback frame. Do not store the pointer.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NxrtTensorDesc {
    /// Number of dimensions.
    pub ndim: u32,
    _pad: u32,
    /// Pointer to `ndim` dimension sizes. Borrowed for the callback frame.
    pub dims: *const i64,
    /// Element dtype (mirrors `onnx_runtime_ir::DataType` discriminant).
    pub dtype: u32,
    _pad2: u32,
}

// --- Node capability claim ---------------------------------------------------

/// A claim that the EP can handle a specific node.
///
/// # Ownership
///
/// Value type. Does not outlive the `get_capability` callback frame unless
/// copied by the host into its own storage.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxrtNodeCapability {
    /// Node index within the graph.
    pub node_index: u32,
    /// Estimated relative cost (lower = preferred). 0 = unknown.
    pub cost: u32,
}

// --- Allocator vtable --------------------------------------------------------

/// Vtable for device memory operations.
///
/// # Ownership
///
/// Created by `ep.get_allocator()`. The **host owns** the returned vtable and
/// must call `release(ctx)` exactly once when done.
#[repr(C)]
pub struct NxrtAllocatorVtable {
    /// Size of this struct (forward compat).
    pub struct_size: u32,
    _pad: u32,
    /// Allocate `size` bytes with `alignment`. Returns null on failure.
    pub alloc: unsafe extern "C" fn(ctx: *mut c_void, size: usize, align: usize) -> *mut c_void,
    /// Free a previously allocated pointer. No-op if ptr is null.
    pub free: unsafe extern "C" fn(ctx: *mut c_void, ptr: *mut c_void),
    /// Copy `size` bytes from host memory to device memory.
    pub copy_from_host: unsafe extern "C" fn(
        ctx: *mut c_void,
        dst: *mut c_void,
        src: *const u8,
        size: usize,
    ) -> NxrtStatus,
    /// Copy `size` bytes from device memory to host memory.
    pub copy_to_host: unsafe extern "C" fn(
        ctx: *mut c_void,
        dst: *mut u8,
        src: *const c_void,
        size: usize,
    ) -> NxrtStatus,
    /// Synchronize -- wait for pending operations to complete.
    pub sync: unsafe extern "C" fn(ctx: *mut c_void) -> NxrtStatus,
    /// Release the allocator (free ctx). Must be called exactly once.
    pub release: unsafe extern "C" fn(ctx: *mut c_void),
    /// Opaque context pointer passed as first arg to all vtable fns.
    pub ctx: *mut c_void,
}

// --- Kernel vtable -----------------------------------------------------------

/// Vtable for a compiled kernel.
///
/// # Ownership
///
/// Created by `ep.compile()`. The **host owns** the kernel and must call
/// `release(ctx)` exactly once. Input/output device pointers passed to
/// `execute` are borrowed for the call frame only.
#[repr(C)]
pub struct NxrtKernelVtable {
    /// Size of this struct (forward compat).
    pub struct_size: u32,
    _pad: u32,
    /// Execute the kernel.
    pub execute: unsafe extern "C" fn(
        ctx: *mut c_void,
        inputs: *const *const c_void,
        num_inputs: u32,
        outputs: *mut *mut c_void,
        num_outputs: u32,
    ) -> NxrtStatus,
    /// Release the kernel (free ctx). Must be called exactly once.
    pub release: unsafe extern "C" fn(ctx: *mut c_void),
    /// Opaque kernel context.
    pub ctx: *mut c_void,
}

// --- EP vtable ---------------------------------------------------------------

/// Vtable for an execution provider instance.
///
/// # Ownership
///
/// Created by `factory.create_ep()`. The **host owns** the EP and must call
/// `release(ctx)` exactly once. The `name` pointer is valid for the EP lifetime.
#[repr(C)]
pub struct NxrtEpVtable {
    /// Size of this struct (forward compat).
    pub struct_size: u32,
    /// Device type discriminant (mirrors `onnx_runtime_ir::DeviceType`).
    pub device_type: u32,
    /// Device ordinal.
    pub device_id: u32,
    _pad: u32,
    /// Human-readable EP name (null-terminated UTF-8, valid for EP lifetime).
    /// Owned by the EP -- the host must not free it.
    pub name: *const u8,

    /// Query capability for nodes. The EP writes claimed node indices into
    /// `out_claims` (up to `max_claims`) and sets `*out_num_claims`.
    ///
    /// # Lifetime
    ///
    /// All pointer arguments are borrowed for this call frame only.
    pub get_capability: unsafe extern "C" fn(
        ctx: *mut c_void,
        op_types: *const *const u8,
        input_descs: *const *const NxrtTensorDesc,
        num_inputs_per_node: *const u32,
        num_nodes: u32,
        out_claims: *mut NxrtNodeCapability,
        max_claims: u32,
        out_num_claims: *mut u32,
    ) -> NxrtStatus,

    /// Compile a subgraph into a kernel. The host owns the returned kernel.
    pub compile: unsafe extern "C" fn(
        ctx: *mut c_void,
        node_indices: *const u32,
        num_nodes: u32,
        out_kernel: *mut *mut NxrtKernelVtable,
    ) -> NxrtStatus,

    /// Get an allocator for this EP's device. Host owns the result.
    pub get_allocator: unsafe extern "C" fn(
        ctx: *mut c_void,
        out_allocator: *mut *mut NxrtAllocatorVtable,
    ) -> NxrtStatus,

    /// Release the EP (free ctx). Must be called exactly once.
    pub release: unsafe extern "C" fn(ctx: *mut c_void),

    /// Opaque EP context.
    pub ctx: *mut c_void,
}

// --- Factory vtable ----------------------------------------------------------

/// Vtable for an EP factory (creates EP instances).
///
/// # Ownership
///
/// Created by `NxrtCreateEpFactories`. The **host owns** the factory and must
/// call `release(ctx)` exactly once. Each EP created by `create_ep` is
/// independently owned by the host.
#[repr(C)]
pub struct NxrtEpFactoryVtable {
    /// Size of this struct (forward compat).
    pub struct_size: u32,
    /// Number of devices available (0 = use default).
    pub num_devices: u32,
    /// Human-readable factory/EP name (null-terminated, valid for factory lifetime).
    pub name: *const u8,

    /// Create an EP instance for the given device ordinal.
    /// Host owns the returned `*mut NxrtEpVtable`.
    pub create_ep: unsafe extern "C" fn(
        ctx: *mut c_void,
        device_ordinal: u32,
        out_ep: *mut *mut NxrtEpVtable,
    ) -> NxrtStatus,

    /// Release the factory (free ctx). Must be called exactly once.
    pub release: unsafe extern "C" fn(ctx: *mut c_void),

    /// Opaque factory context.
    pub ctx: *mut c_void,
}

// --- Factory creation helper -------------------------------------------------

/// Internal state held by a factory.
struct FactoryInner {
    constructor: Box<dyn Fn() -> Box<dyn onnx_runtime_ep_api::provider::ExecutionProvider>>,
    name_buf: Box<[u8]>,
    device_type: u32,
    device_id: u32,
}

// SAFETY: The constructor is Send+Sync (required by ExecutionProvider: Send+Sync).
unsafe impl Send for FactoryInner {}
unsafe impl Sync for FactoryInner {}

/// Create EP factories from a constructor closure.
///
/// # Safety
///
/// All pointers must be valid per the nxrt ABI contract.
pub unsafe fn create_ep_factories<F>(
    out_factories: *mut *mut NxrtEpFactoryVtable,
    max_factories: usize,
    out_num: *mut usize,
    constructor: F,
) -> NxrtStatus
where
    F: Fn() -> Box<dyn onnx_runtime_ep_api::provider::ExecutionProvider> + 'static,
{
    if max_factories == 0 || out_factories.is_null() || out_num.is_null() {
        return NxrtStatus::from_code_with_message(
            NxrtStatusCode::InvalidArgument,
            "NxrtCreateEpFactories: null pointer or zero max_factories",
        );
    }

    // Probe the EP for metadata.
    let ep = constructor();
    let ep_name = ep.name().to_owned();
    let device_type = device_type_to_u32(ep.device_type());
    let device_id_val = ep.device_id().index;
    drop(ep);

    // Stable null-terminated name.
    let mut name_buf = Vec::with_capacity(ep_name.len() + 1);
    name_buf.extend_from_slice(ep_name.as_bytes());
    name_buf.push(0);
    let name_buf = name_buf.into_boxed_slice();

    let inner = Box::new(FactoryInner {
        constructor: Box::new(constructor),
        name_buf,
        device_type,
        device_id: device_id_val,
    });
    let name_ptr = inner.name_buf.as_ptr();
    let factory_ctx = Box::into_raw(inner) as *mut c_void;

    let factory = Box::new(NxrtEpFactoryVtable {
        struct_size: std::mem::size_of::<NxrtEpFactoryVtable>() as u32,
        num_devices: 1,
        name: name_ptr,
        create_ep: factory_create_ep,
        release: factory_release,
        ctx: factory_ctx,
    });

    unsafe {
        *out_factories = Box::into_raw(factory);
        *out_num = 1;
    }

    NxrtStatus::ok()
}

unsafe extern "C" fn factory_create_ep(
    ctx: *mut c_void,
    _device_ordinal: u32,
    out_ep: *mut *mut NxrtEpVtable,
) -> NxrtStatus {
    crate::status::catch_status_panic(|| {
        if ctx.is_null() || out_ep.is_null() {
            return NxrtStatus::from_code_with_message(
                NxrtStatusCode::InvalidArgument,
                "factory_create_ep: null pointer",
            );
        }
        let inner = unsafe { &*(ctx as *const FactoryInner) };
        let ep = (inner.constructor)();
        drop(ep); // We don't wire EP methods yet; just prove lifecycle works.

        let ep_box: Box<dyn onnx_runtime_ep_api::provider::ExecutionProvider> =
            (inner.constructor)();
        let ep_raw = Box::into_raw(Box::new(ep_box)) as *mut c_void;

        let ep_vtable = Box::new(NxrtEpVtable {
            struct_size: std::mem::size_of::<NxrtEpVtable>() as u32,
            device_type: inner.device_type,
            device_id: inner.device_id,
            _pad: 0,
            name: inner.name_buf.as_ptr(),
            get_capability: ep_get_capability,
            compile: ep_compile,
            get_allocator: ep_get_allocator,
            release: ep_release,
            ctx: ep_raw,
        });
        unsafe { *out_ep = Box::into_raw(ep_vtable) };
        NxrtStatus::ok()
    })
}

unsafe extern "C" fn ep_get_capability(
    _ctx: *mut c_void,
    _op_types: *const *const u8,
    _input_descs: *const *const NxrtTensorDesc,
    _num_inputs_per_node: *const u32,
    _num_nodes: u32,
    _out_claims: *mut NxrtNodeCapability,
    _max_claims: u32,
    out_num_claims: *mut u32,
) -> NxrtStatus {
    // Fail closed: claim nothing until wired to the trait.
    if !out_num_claims.is_null() {
        unsafe { *out_num_claims = 0 };
    }
    NxrtStatus::ok()
}

unsafe extern "C" fn ep_compile(
    _ctx: *mut c_void,
    _node_indices: *const u32,
    _num_nodes: u32,
    out_kernel: *mut *mut NxrtKernelVtable,
) -> NxrtStatus {
    if !out_kernel.is_null() {
        unsafe { *out_kernel = ptr::null_mut() };
    }
    NxrtStatus::from_code_with_message(
        NxrtStatusCode::NotImplemented,
        "ep_compile: not yet wired to trait (fail closed)",
    )
}

unsafe extern "C" fn ep_get_allocator(
    _ctx: *mut c_void,
    out_allocator: *mut *mut NxrtAllocatorVtable,
) -> NxrtStatus {
    if !out_allocator.is_null() {
        unsafe { *out_allocator = ptr::null_mut() };
    }
    NxrtStatus::from_code_with_message(
        NxrtStatusCode::NotImplemented,
        "ep_get_allocator: not yet wired (fail closed)",
    )
}

unsafe extern "C" fn ep_release(ctx: *mut c_void) {
    crate::status::catch_void_panic(|| {
        if !ctx.is_null() {
            let _ = unsafe {
                Box::from_raw(ctx as *mut Box<dyn onnx_runtime_ep_api::provider::ExecutionProvider>)
            };
        }
    });
}

unsafe extern "C" fn factory_release(ctx: *mut c_void) {
    crate::status::catch_void_panic(|| {
        if !ctx.is_null() {
            let _ = unsafe { Box::from_raw(ctx as *mut FactoryInner) };
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_desc_is_repr_c() {
        let size = std::mem::size_of::<NxrtTensorDesc>();
        assert!(size > 0);
    }

    #[test]
    fn factory_vtable_has_struct_size() {
        let size = std::mem::size_of::<NxrtEpFactoryVtable>();
        assert!(size > 0);
    }

    #[test]
    fn device_type_codes_are_stable_and_vulkan_is_appended() {
        use onnx_runtime_ir::DeviceType;

        assert_eq!(device_type_to_u32(DeviceType::Cpu), 0);
        assert_eq!(device_type_to_u32(DeviceType::Cuda), 1);
        assert_eq!(device_type_to_u32(DeviceType::Rocm), 2);
        assert_eq!(device_type_to_u32(DeviceType::CoreMl), 3);
        assert_eq!(device_type_to_u32(DeviceType::Mlx), 4);
        assert_eq!(device_type_to_u32(DeviceType::WebGpu), 5);
        assert_eq!(device_type_to_u32(DeviceType::Qnn), 6);
        assert_eq!(device_type_to_u32(DeviceType::OpenVino), 7);
        assert_eq!(device_type_to_u32(DeviceType::Vulkan), 8);
        assert_eq!(device_type_to_u32(DeviceType::Custom(7)), 0x1007);
    }

    #[test]
    fn create_ep_factories_rejects_null() {
        let status =
            unsafe { create_ep_factories(ptr::null_mut(), 1, ptr::null_mut(), || unreachable!()) };
        assert_eq!(status.status_code(), Some(NxrtStatusCode::InvalidArgument));
    }

    #[test]
    fn create_ep_factories_rejects_zero_max() {
        let mut out: *mut NxrtEpFactoryVtable = ptr::null_mut();
        let mut num: usize = 0;
        let status = unsafe { create_ep_factories(&mut out, 0, &mut num, || unreachable!()) };
        assert_eq!(status.status_code(), Some(NxrtStatusCode::InvalidArgument));
    }
}
