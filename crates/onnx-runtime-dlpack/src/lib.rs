//! DLPack C-ABI structs (v1.0, backward-compatible with the v0.8 unversioned
//! `DLManagedTensor`) and a memory-owning exporter.
//!
//! ## What this crate is
//!
//! A dependency-free, PyO3-free definition of the DLPack exchange ABI plus one
//! job: turn "a base pointer + some fields + an owner that must stay alive" into
//! a `*mut DLManagedTensor` whose `deleter` drops that owner. All of the ABI
//! `unsafe` lives here, contained and documented, so higher crates (the Python
//! binding today; `onnx-runtime-ep-api` / `onnx-runtime-eager` for a future
//! zero-copy *import* path) can borrow buffers across the FFI boundary without
//! re-deriving the pointer bookkeeping or the ownership handshake.
//!
//! ## ABI version
//!
//! We emit the **unversioned** [`DLManagedTensor`] and (on the Python side) the
//! `"dltensor"` PyCapsule. This is the form every current consumer accepts:
//! `torch.from_dlpack`, `numpy.from_dlpack` (numpy ≥ 1.23), CuPy, JAX and MLX
//! all read it. The newer `DLManagedTensorVersioned` (`"dltensor_versioned"`
//! capsule) is strictly opt-in on the consumer side and not universally
//! consumed yet; the version constants below are exported so a versioned path
//! can be layered on without touching this struct.
//!
//! ## Correspondence to `onnx-runtime-ep-api`'s `TensorView` (§5.3)
//!
//! The field mapping is deliberately 1:1 — `data` base pointer, separate
//! `byte_offset`, element-count `strides` (`i64`, negative allowed), `device`,
//! `dtype` — so export is a field-wise shim, never a copy.

#[cfg(target_endian = "big")]
compile_error!(
    "onnx-runtime-dlpack export assumes little-endian byte storage; big-endian targets need a native-endian conversion"
);

use std::any::Any;
use std::ffi::c_void;

/// DLPack ABI major version this crate targets.
pub const DLPACK_MAJOR_VERSION: u32 = 1;
/// DLPack ABI minor version this crate targets.
pub const DLPACK_MINOR_VERSION: u32 = 0;

/// `DLManagedTensorVersioned::flags` bit: the tensor's memory is read-only.
pub const DLPACK_FLAG_BITMASK_READ_ONLY: u64 = 1 << 0;

// ── `DLDeviceType` values (subset we can produce/consume) ──────────────────

/// CPU / host memory (`kDLCPU`).
pub const DL_CPU: i32 = 1;
/// CUDA device memory (`kDLCUDA`).
pub const DL_CUDA: i32 = 2;
/// Pinned CUDA host memory (`kDLCUDAHost`).
pub const DL_CUDA_HOST: i32 = 3;

// ── `DLDataTypeCode` values ────────────────────────────────────────────────

/// Signed integer (`kDLInt`).
pub const DL_INT: u8 = 0;
/// Unsigned integer (`kDLUInt`).
pub const DL_UINT: u8 = 1;
/// IEEE floating point (`kDLFloat`).
pub const DL_FLOAT: u8 = 2;
/// Opaque handle (`kDLOpaqueHandle`).
pub const DL_OPAQUE_HANDLE: u8 = 3;
/// bfloat16 (`kDLBfloat`).
pub const DL_BFLOAT: u8 = 4;
/// Complex (`kDLComplex`).
pub const DL_COMPLEX: u8 = 5;
/// Boolean (`kDLBool`) — 8-bit, matching numpy's `bool_`.
pub const DL_BOOL: u8 = 6;

/// A physical device: a `DLDeviceType` code plus an ordinal.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDevice {
    /// One of the `DL_*` device-type constants.
    pub device_type: i32,
    /// Device ordinal (0 for CPU).
    pub device_id: i32,
}

/// The element type: a code, a bit width, and a SIMD lane count (1 = scalar).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDataType {
    /// One of the `DL_*` type-code constants.
    pub code: u8,
    /// Bit width of a single element (e.g. 32 for `float32`, 8 for `bool`).
    pub bits: u8,
    /// Number of packed lanes; always 1 for the tensors we export.
    pub lanes: u16,
}

/// A plain (non-owning) tensor descriptor — the payload a consumer reads.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLTensor {
    /// Base pointer of the allocation (element origin is `data + byte_offset`).
    pub data: *mut c_void,
    /// Where `data` lives.
    pub device: DLDevice,
    /// Number of dimensions.
    pub ndim: i32,
    /// Element type.
    pub dtype: DLDataType,
    /// Pointer to `ndim` shape entries.
    pub shape: *mut i64,
    /// Pointer to `ndim` stride entries (in **elements**), or null for
    /// row-major contiguous.
    pub strides: *mut i64,
    /// Byte offset from `data` to the first element.
    pub byte_offset: u64,
}

/// An owning tensor descriptor: a [`DLTensor`] plus an opaque `manager_ctx` and
/// a `deleter` the consumer calls exactly once when done.
#[repr(C)]
pub struct DLManagedTensor {
    /// The borrowed tensor.
    pub dl_tensor: DLTensor,
    /// Producer-private context; here it points at the boxed [`ManagedOwner`].
    pub manager_ctx: *mut c_void,
    /// Frees `manager_ctx` (and thus the backing allocation). May be null.
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// The DLPack ABI version carried by [`DLManagedTensorVersioned`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLPackVersion {
    /// ABI major version (breaking).
    pub major: u32,
    /// ABI minor version (backward-compatible).
    pub minor: u32,
}

/// The versioned owning tensor (DLPack v1.0+). Adds an explicit ABI `version`
/// and a `flags` field (notably [`DLPACK_FLAG_BITMASK_READ_ONLY`]) that the
/// unversioned [`DLManagedTensor`] cannot express — which is what lets a
/// consumer such as numpy ≥ 2.1 import the buffer **writable**.
#[repr(C)]
pub struct DLManagedTensorVersioned {
    /// Producer's DLPack ABI version.
    pub version: DLPackVersion,
    /// Producer-private context; points at the boxed owner.
    pub manager_ctx: *mut c_void,
    /// Frees `manager_ctx` (and thus the backing allocation). May be null.
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    /// Bitmask of `DLPACK_FLAG_BITMASK_*` (0 = writable, contiguous).
    pub flags: u64,
    /// The borrowed tensor.
    pub dl_tensor: DLTensor,
}

/// The producer-side context kept alive for as long as a consumer holds the
/// exported tensor. It owns:
///
/// * `_keep_alive` — the real backing memory owner (e.g. an `Arc<Tensor>`); its
///   `Drop` releases the allocation the exported `data` pointer refers to.
/// * `shape` / `strides` — the arrays [`DLTensor::shape`] / [`DLTensor::strides`]
///   point into. They must be owned here so the pointers stay valid for the
///   lifetime of the export, independent of the caller's stack frame.
///
/// The whole struct is heap-boxed; `manager_ctx` is the raw box pointer, and the
/// exported `*mut DLManagedTensor` points at the embedded `managed` field.
struct ManagedOwner {
    _keep_alive: Box<dyn Any + Send>,
    shape: Vec<i64>,
    strides: Vec<i64>,
    managed: DLManagedTensor,
}

/// The `deleter` installed on every tensor [`export`] produces.
///
/// # Safety
///
/// `managed` must be a pointer returned by [`export`] (i.e. the address of the
/// `managed` field of a `Box<ManagedOwner>` whose `manager_ctx` is the box
/// pointer), and must not have been deleted already. DLPack's contract
/// guarantees the deleter is called at most once by the single owner of the
/// tensor, so reconstructing and dropping the box here frees the allocation
/// exactly once.
unsafe extern "C" fn owner_deleter(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    // SAFETY: by the contract above, `manager_ctx` is the `Box::into_raw`
    // pointer of the `ManagedOwner` that embeds `*managed`. Reboxing it and
    // dropping runs `ManagedOwner`'s destructor, which drops `_keep_alive`
    // (releasing the backing buffer) and the shape/stride vectors. The tensor
    // pointer itself becomes dangling on return, which is exactly what DLPack
    // expects after the single deleter call.
    unsafe {
        let ctx = (*managed).manager_ctx as *mut ManagedOwner;
        if !ctx.is_null() {
            drop(Box::from_raw(ctx));
        }
    }
}

/// Export a borrowed buffer as an owning `*mut DLManagedTensor`.
///
/// `keep_alive` is the owner that must outlive every consumer of the returned
/// tensor: dropping it must free (or release the last reference to) the memory
/// `data` points into. It is moved into the tensor's `manager_ctx` and dropped
/// by the `deleter`, so the memory stays valid for exactly as long as the
/// consumer holds the tensor — regardless of what happens to the object that
/// produced it.
///
/// * `data` — base pointer of the allocation (not the element origin).
/// * `shape` — one entry per dimension; its length sets `ndim`.
/// * `strides` — element-count strides, or empty for row-major contiguous
///   (exported as a null `strides`, which consumers read as C-contiguous).
/// * `byte_offset` — offset from `data` to the first element.
///
/// The returned pointer is owned by the caller until it is handed to a consumer
/// via the DLPack protocol; if it is never consumed, the caller must invoke the
/// tensor's `deleter` (see [`release`]) to avoid a leak.
///
/// # Safety
///
/// The caller must guarantee that `data` plus `byte_offset` points to a valid
/// allocation of the described `shape` and `dtype`, and that allocation remains
/// alive until the returned manager's `deleter` runs. Moving the allocation's
/// owner into `keep_alive` typically provides this guarantee. `shape.len()` must
/// equal the exported `ndim`; if `strides` is non-empty, it must contain exactly
/// one stride per shape dimension.
///
/// # Panics
///
/// Panics if `shape.len()` does not fit DLPack's `i32` `ndim`, or if non-empty
/// `strides` does not have one entry per shape dimension.
pub unsafe fn export(
    keep_alive: Box<dyn Any + Send>,
    data: *mut c_void,
    device: DLDevice,
    dtype: DLDataType,
    shape: Vec<i64>,
    strides: Vec<i64>,
    byte_offset: u64,
) -> *mut DLManagedTensor {
    assert!(
        i32::try_from(shape.len()).is_ok(),
        "DLPack ndim exceeds i32::MAX: {}",
        shape.len()
    );
    assert!(
        strides.is_empty() || strides.len() == shape.len(),
        "DLPack strides length ({}) must equal shape length ({})",
        strides.len(),
        shape.len()
    );
    let ndim = i32::try_from(shape.len()).expect("shape length was validated above");
    let mut owner = Box::new(ManagedOwner {
        _keep_alive: keep_alive,
        shape,
        strides,
        managed: DLManagedTensor {
            dl_tensor: DLTensor {
                data,
                device,
                ndim,
                dtype,
                shape: std::ptr::null_mut(),
                strides: std::ptr::null_mut(),
                byte_offset,
            },
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(owner_deleter),
        },
    });

    // The shape/stride Vec buffers are heap allocations owned by `owner`; their
    // addresses are stable for `owner`'s lifetime, so taking pointers into them
    // now (before/after the box moves) is sound — `Box::into_raw` transfers the
    // same heap allocation without moving the pointee.
    owner.managed.dl_tensor.shape = owner.shape.as_mut_ptr();
    owner.managed.dl_tensor.strides = if owner.strides.is_empty() {
        std::ptr::null_mut()
    } else {
        owner.strides.as_mut_ptr()
    };

    let raw = Box::into_raw(owner);
    // SAFETY: `raw` is a freshly leaked, uniquely-owned box pointer. We record
    // it as `manager_ctx` so `owner_deleter` can recover and free it, then
    // return the interior `managed` field address (stable: it lives inside the
    // heap box, which we do not move again until the deleter reboxes it).
    unsafe {
        (*raw).managed.manager_ctx = raw as *mut c_void;
        &raw mut (*raw).managed
    }
}

/// Invoke a managed tensor's own `deleter`, if any.
///
/// Used by an exporter that created a tensor via [`export`] but whose consumer
/// never took ownership (e.g. a PyCapsule that was garbage-collected without
/// being consumed): calling this releases the backing memory instead of leaking
/// it. After this returns, `managed` is dangling and must not be used again.
///
/// # Safety
///
/// `managed` must be a live pointer from [`export`] (or another well-formed
/// DLPack producer) whose `deleter` has **not** already run. Calling it twice is
/// undefined behaviour — the DLPack ownership handshake exists precisely to
/// ensure a single call.
pub unsafe fn release(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    // SAFETY: caller guarantees `managed` is a live, not-yet-deleted tensor.
    unsafe {
        if let Some(deleter) = (*managed).deleter {
            deleter(managed);
        }
    }
}

/// Versioned counterpart of [`ManagedOwner`] (owns a
/// [`DLManagedTensorVersioned`] instead of a [`DLManagedTensor`]).
struct ManagedOwnerVersioned {
    _keep_alive: Box<dyn Any + Send>,
    shape: Vec<i64>,
    strides: Vec<i64>,
    managed: DLManagedTensorVersioned,
}

/// The `deleter` installed on every tensor [`export_versioned`] produces.
///
/// # Safety
///
/// `managed` must be a pointer returned by [`export_versioned`], not previously
/// deleted. See [`owner_deleter`] for the ownership reasoning; this is the
/// versioned analogue.
unsafe extern "C" fn owner_deleter_versioned(managed: *mut DLManagedTensorVersioned) {
    if managed.is_null() {
        return;
    }
    // SAFETY: `manager_ctx` is the `Box::into_raw` pointer of the
    // `ManagedOwnerVersioned` embedding `*managed`; reboxing and dropping frees
    // the backing buffer and the shape/stride vectors exactly once.
    unsafe {
        let ctx = (*managed).manager_ctx as *mut ManagedOwnerVersioned;
        if !ctx.is_null() {
            drop(Box::from_raw(ctx));
        }
    }
}

/// Export a borrowed buffer as an owning `*mut DLManagedTensorVersioned`.
///
/// Identical ownership semantics to [`export`], but emits the DLPack v1.0+
/// versioned struct so `flags` can be carried. `read_only` sets
/// [`DLPACK_FLAG_BITMASK_READ_ONLY`]; pass `false` to let consumers import the
/// buffer as writable (the whole point of a mutable zero-copy borrow).
///
/// # Safety
///
/// The caller must guarantee that `data` plus `byte_offset` points to a valid
/// allocation of the described `shape` and `dtype`, and that allocation remains
/// alive until the returned manager's `deleter` runs. Moving the allocation's
/// owner into `keep_alive` typically provides this guarantee. `shape.len()` must
/// equal the exported `ndim`; if `strides` is non-empty, it must contain exactly
/// one stride per shape dimension.
///
/// # Panics
///
/// Panics if `shape.len()` does not fit DLPack's `i32` `ndim`, or if non-empty
/// `strides` does not have one entry per shape dimension.
#[allow(clippy::too_many_arguments)] // mirrors the DLPack field set 1:1; grouping into a struct would only obscure it
pub unsafe fn export_versioned(
    keep_alive: Box<dyn Any + Send>,
    data: *mut c_void,
    device: DLDevice,
    dtype: DLDataType,
    shape: Vec<i64>,
    strides: Vec<i64>,
    byte_offset: u64,
    read_only: bool,
) -> *mut DLManagedTensorVersioned {
    assert!(
        i32::try_from(shape.len()).is_ok(),
        "DLPack ndim exceeds i32::MAX: {}",
        shape.len()
    );
    assert!(
        strides.is_empty() || strides.len() == shape.len(),
        "DLPack strides length ({}) must equal shape length ({})",
        strides.len(),
        shape.len()
    );
    let ndim = i32::try_from(shape.len()).expect("shape length was validated above");
    let flags = if read_only { DLPACK_FLAG_BITMASK_READ_ONLY } else { 0 };
    let mut owner = Box::new(ManagedOwnerVersioned {
        _keep_alive: keep_alive,
        shape,
        strides,
        managed: DLManagedTensorVersioned {
            version: DLPackVersion { major: DLPACK_MAJOR_VERSION, minor: DLPACK_MINOR_VERSION },
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(owner_deleter_versioned),
            flags,
            dl_tensor: DLTensor {
                data,
                device,
                ndim,
                dtype,
                shape: std::ptr::null_mut(),
                strides: std::ptr::null_mut(),
                byte_offset,
            },
        },
    });

    owner.managed.dl_tensor.shape = owner.shape.as_mut_ptr();
    owner.managed.dl_tensor.strides = if owner.strides.is_empty() {
        std::ptr::null_mut()
    } else {
        owner.strides.as_mut_ptr()
    };

    let raw = Box::into_raw(owner);
    // SAFETY: `raw` is a freshly leaked, uniquely-owned box pointer; record it
    // as `manager_ctx` for the deleter and return the stable interior field.
    unsafe {
        (*raw).managed.manager_ctx = raw as *mut c_void;
        &raw mut (*raw).managed
    }
}

/// Versioned analogue of [`release`].
///
/// # Safety
///
/// `managed` must be a live pointer from [`export_versioned`] whose `deleter`
/// has not already run. Calling it twice is undefined behaviour.
pub unsafe fn release_versioned(managed: *mut DLManagedTensorVersioned) {
    if managed.is_null() {
        return;
    }
    // SAFETY: caller guarantees `managed` is a live, not-yet-deleted tensor.
    unsafe {
        if let Some(deleter) = (*managed).deleter {
            deleter(managed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // The ABI layout must match DLPack's C headers on a 64-bit target, or every
    // consumer reads garbage. These pin the sizes/offsets we rely on.
    #[test]
    fn abi_layout_is_dlpack_compatible() {
        assert_eq!(std::mem::size_of::<DLDevice>(), 8);
        assert_eq!(std::mem::size_of::<DLDataType>(), 4);
        assert_eq!(std::mem::align_of::<DLDataType>(), 2);

        assert_eq!(std::mem::size_of::<DLTensor>(), 48);
        assert_eq!(std::mem::offset_of!(DLTensor, data), 0);
        assert_eq!(std::mem::offset_of!(DLTensor, device), 8);
        assert_eq!(std::mem::offset_of!(DLTensor, ndim), 16);
        assert_eq!(std::mem::offset_of!(DLTensor, dtype), 20);
        assert_eq!(std::mem::offset_of!(DLTensor, shape), 24);
        assert_eq!(std::mem::offset_of!(DLTensor, strides), 32);
        assert_eq!(std::mem::offset_of!(DLTensor, byte_offset), 40);

        assert_eq!(std::mem::size_of::<DLManagedTensor>(), 64);
        assert_eq!(std::mem::offset_of!(DLManagedTensor, dl_tensor), 0);
        assert_eq!(std::mem::offset_of!(DLManagedTensor, manager_ctx), 48);
        assert_eq!(std::mem::offset_of!(DLManagedTensor, deleter), 56);

        assert_eq!(std::mem::size_of::<DLPackVersion>(), 8);
        assert_eq!(std::mem::size_of::<DLManagedTensorVersioned>(), 80);
        assert_eq!(std::mem::offset_of!(DLManagedTensorVersioned, version), 0);
        assert_eq!(std::mem::offset_of!(DLManagedTensorVersioned, manager_ctx), 8);
        assert_eq!(std::mem::offset_of!(DLManagedTensorVersioned, deleter), 16);
        assert_eq!(std::mem::offset_of!(DLManagedTensorVersioned, flags), 24);
        assert_eq!(std::mem::offset_of!(DLManagedTensorVersioned, dl_tensor), 32);
    }

    #[test]
    fn export_sets_fields_and_interior_pointers() {
        let mut data = [1u8, 2, 3, 4];
        let ptr = data.as_mut_ptr() as *mut c_void;
        // SAFETY: `data` remains live until `release` below.
        let managed = unsafe {
            export(
                Box::new(()),
                ptr,
                DLDevice { device_type: DL_CPU, device_id: 0 },
                DLDataType { code: DL_UINT, bits: 8, lanes: 1 },
                vec![2, 2],
                vec![],
                0,
            )
        };
        // SAFETY: `managed` is the just-returned live export pointer.
        unsafe {
            let t = &(*managed).dl_tensor;
            assert_eq!(t.data, ptr);
            assert_eq!(t.ndim, 2);
            assert!(!t.shape.is_null());
            assert!(t.strides.is_null(), "empty strides export as null");
            assert_eq!(std::slice::from_raw_parts(t.shape, 2), &[2, 2]);
            assert_eq!(t.device.device_type, DL_CPU);
            assert!((*managed).deleter.is_some());
            release(managed);
        }
    }

    #[test]
    fn deleter_drops_the_keep_alive_owner_once() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct Tracker;
        impl Drop for Tracker {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }

        // Arc so we can observe strong count going to zero on deletion.
        let owner = Arc::new(Tracker);
        let mut byte = [0u8];
        // SAFETY: `byte` remains live until `release` below.
        let managed = unsafe {
            export(
                Box::new(owner.clone()),
                byte.as_mut_ptr() as *mut c_void,
                DLDevice { device_type: DL_CPU, device_id: 0 },
                DLDataType { code: DL_UINT, bits: 8, lanes: 1 },
                vec![1],
                vec![],
                0,
            )
        };
        assert_eq!(Arc::strong_count(&owner), 2, "export holds a reference");
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);

        // SAFETY: single, first deletion of a live export.
        unsafe { release(managed) };

        assert_eq!(Arc::strong_count(&owner), 1, "deleter released its ref");
        assert_eq!(DROPS.load(Ordering::SeqCst), 0, "our clone still alive");
        drop(owner);
        assert_eq!(DROPS.load(Ordering::SeqCst), 1, "final drop runs once");
    }

    #[test]
    fn strides_round_trip_when_non_empty() {
        let mut data = [0u8; 8];
        // SAFETY: `data` remains live until `release` below.
        let managed = unsafe {
            export(
                Box::new(()),
                data.as_mut_ptr() as *mut c_void,
                DLDevice { device_type: DL_CPU, device_id: 0 },
                DLDataType { code: DL_INT, bits: 32, lanes: 1 },
                vec![2, 1],
                vec![1, 2],
                0,
            )
        };
        // SAFETY: live export pointer.
        unsafe {
            let t = &(*managed).dl_tensor;
            assert!(!t.strides.is_null());
            assert_eq!(std::slice::from_raw_parts(t.strides, 2), &[1, 2]);
            release(managed);
        }
    }

    #[test]
    #[should_panic(expected = "DLPack strides length")]
    fn export_rejects_mismatched_non_empty_strides() {
        let mut data = [0u8; 1];
        // SAFETY: this call must panic before returning a managed pointer.
        let _ = unsafe {
            export(
                Box::new(()),
                data.as_mut_ptr() as *mut c_void,
                DLDevice { device_type: DL_CPU, device_id: 0 },
                DLDataType { code: DL_UINT, bits: 8, lanes: 1 },
                vec![1, 1],
                vec![1],
                0,
            )
        };
    }

    #[test]
    fn versioned_export_carries_version_and_writable_flag() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct Tracker;
        impl Drop for Tracker {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let mut byte = [7u8];
        // SAFETY: `byte` remains live until `release_versioned` below.
        let managed = unsafe {
            export_versioned(
                Box::new(Tracker),
                byte.as_mut_ptr() as *mut c_void,
                DLDevice { device_type: DL_CPU, device_id: 0 },
                DLDataType { code: DL_UINT, bits: 8, lanes: 1 },
                vec![1],
                vec![],
                0,
                false,
            )
        };
        // SAFETY: live versioned export pointer.
        unsafe {
            assert_eq!((*managed).version.major, DLPACK_MAJOR_VERSION);
            assert_eq!((*managed).flags, 0, "writable export has no read-only bit");
            assert_eq!((*managed).dl_tensor.ndim, 1);
            assert_eq!(DROPS.load(Ordering::SeqCst), 0);
            release_versioned(managed);
            assert_eq!(DROPS.load(Ordering::SeqCst), 1, "deleter freed the owner");
        }
    }

    #[test]
    fn versioned_read_only_sets_flag() {
        let mut byte = [0u8];
        // SAFETY: `byte` remains live until `release_versioned` below.
        let managed = unsafe {
            export_versioned(
                Box::new(()),
                byte.as_mut_ptr() as *mut c_void,
                DLDevice { device_type: DL_CPU, device_id: 0 },
                DLDataType { code: DL_UINT, bits: 8, lanes: 1 },
                vec![1],
                vec![],
                0,
                true,
            )
        };
        // SAFETY: live versioned export pointer.
        unsafe {
            assert_eq!((*managed).flags & DLPACK_FLAG_BITMASK_READ_ONLY, DLPACK_FLAG_BITMASK_READ_ONLY);
            release_versioned(managed);
        }
    }
}
