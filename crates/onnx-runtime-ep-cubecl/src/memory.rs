//! A virtual address space over opaque CubeCL buffer handles.
//!
//! # The impedance mismatch
//!
//! [`DeviceBuffer`] and [`TensorView`] model device memory as an *address*: a
//! non-null `*mut c_void` that callers are allowed to offset into
//! (`TensorView::with_byte_offset`) even though, for a non-host-accessible
//! device, they must never dereference it. CUDA satisfies this naturally
//! because a `CUdeviceptr` really is an integer address.
//!
//! CubeCL does not expose addresses at all. A `Handle` is an opaque, reference
//! counted token; there is no public API to obtain a device pointer from one,
//! nor to build one from a foreign pointer. Storing `Box::into_raw` of a handle
//! as the "address" would satisfy non-null but breaks the moment anything adds
//! a byte offset, because `token + 64` is not a token.
//!
//! # The bridge
//!
//! This table hands out *synthetic* addresses: each allocation reserves a
//! unique, never-reused, page-aligned range in a 2^48 synthetic space, and the
//! range is recorded against the handle that backs it. Any address inside the
//! range resolves back to `(handle, offset_within_allocation)`, so pointer
//! arithmetic keeps working and stays checkable — an address that lands outside
//! every live range is a bug we can name instead of a silent wild write.
//!
//! Addresses are never reused after free. That costs nothing (the space is
//! astronomically larger than any process will allocate) and turns
//! use-after-free from undefined behaviour into a precise error.
//!
//! [`DeviceBuffer`]: onnx_runtime_ep_api::provider::DeviceBuffer
//! [`TensorView`]: onnx_runtime_ep_api::TensorView

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Mutex;

use cubecl::server::Handle;
use onnx_runtime_ep_api::{EpError, Result};

/// Base of the synthetic address space.
///
/// Deliberately far above anything a real host allocator returns, so a pointer
/// that was never meant to reach this table is obvious in a debugger and in an
/// error message.
const ADDRESS_SPACE_BASE: u64 = 1 << 44;

/// Granularity every allocation's base address is aligned to, and the minimum
/// distance between two allocations. Larger than any alignment a kernel asks
/// for, so an aligned sub-range never crosses into a neighbour.
const ADDRESS_GRANULE: u64 = 4096;

/// One live allocation.
#[derive(Debug)]
struct Region {
    handle: Handle,
    /// Requested size in bytes. The handle may be backed by more.
    size: usize,
    align: usize,
}

/// The set of live allocations, keyed by synthetic base address.
#[derive(Debug)]
pub struct HandleTable {
    inner: Mutex<TableInner>,
}

#[derive(Debug)]
struct TableInner {
    next_base: u64,
    regions: BTreeMap<u64, Region>,
    /// Bases that have been freed, kept so a use-after-free can be reported as
    /// such instead of as an unknown address.
    freed: BTreeMap<u64, usize>,
    live_bytes: usize,
}

/// A resolved device address: which buffer it names, and how far into it.
#[derive(Debug)]
pub struct Resolved {
    /// The backing CubeCL handle, already offset to the requested address.
    pub handle: Handle,
    /// Bytes from the resolved address to the end of the allocation.
    pub remaining: usize,
}

impl Default for HandleTable {
    fn default() -> Self {
        Self::new()
    }
}

impl HandleTable {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(TableInner {
                next_base: ADDRESS_SPACE_BASE,
                regions: BTreeMap::new(),
                freed: BTreeMap::new(),
                live_bytes: 0,
            }),
        }
    }

    /// Record `handle` as a `size`-byte allocation and return its synthetic
    /// base address.
    pub fn insert(&self, handle: Handle, size: usize, align: usize) -> *mut c_void {
        let mut inner = self.lock();
        let base = inner.next_base;
        // Reserve the allocation plus one granule of guard space, so no two
        // allocations are ever adjacent and an overrun by a few bytes lands in
        // unmapped synthetic space rather than in the next tensor.
        let reserved = (size as u64)
            .next_multiple_of(ADDRESS_GRANULE)
            .saturating_add(ADDRESS_GRANULE);
        inner.next_base = base.saturating_add(reserved);
        inner.live_bytes = inner.live_bytes.saturating_add(size);
        inner.regions.insert(
            base,
            Region {
                handle,
                size,
                align,
            },
        );
        base as *mut c_void
    }

    /// Resolve a device address to its handle and the bytes available from it.
    ///
    /// `len` is the number of bytes the caller intends to touch; it is checked
    /// against the allocation so an out-of-bounds view fails here rather than
    /// producing a truncated dispatch.
    pub fn resolve(&self, ptr: *const c_void, len: usize) -> Result<Resolved> {
        let addr = ptr as u64;
        let inner = self.lock();
        let Some((&base, region)) = inner.regions.range(..=addr).next_back() else {
            return Err(inner.describe_unknown_address(addr, len));
        };
        let offset = (addr - base) as usize;
        if offset > region.size {
            return Err(inner.describe_unknown_address(addr, len));
        }
        let remaining = region.size - offset;
        if len > remaining {
            return Err(EpError::InvalidTensorView {
                reason: format!(
                    "cubecl_ep: view of {len} bytes at offset {offset} exceeds its \
                     {size}-byte allocation (only {remaining} bytes remain). The tensor's \
                     shape/strides describe more data than the buffer bound to it holds.",
                    size = region.size,
                ),
            });
        }
        let mut handle = region.handle.clone();
        if offset > 0 {
            handle = handle.offset_start(offset as u64);
        }
        Ok(Resolved { handle, remaining })
    }

    /// Drop the allocation based at `ptr`, releasing the CubeCL handle.
    pub fn remove(&self, ptr: *mut c_void) -> Result<()> {
        let addr = ptr as u64;
        let mut inner = self.lock();
        match inner.regions.remove(&addr) {
            Some(region) => {
                inner.live_bytes = inner.live_bytes.saturating_sub(region.size);
                inner.freed.insert(addr, region.size);
                // Dropping the handle returns the memory to CubeCL's pool.
                drop(region.handle);
                Ok(())
            }
            None => Err(EpError::KernelFailed(inner.describe_bad_free(addr))),
        }
    }

    /// Total bytes currently handed out, for diagnostics and tests.
    pub fn live_bytes(&self) -> usize {
        self.lock().live_bytes
    }

    /// Number of live allocations, for diagnostics and tests.
    pub fn live_allocations(&self) -> usize {
        self.lock().regions.len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, TableInner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl TableInner {
    /// Explain an address that resolves to no live allocation, distinguishing
    /// use-after-free from a pointer that never came from this EP.
    fn describe_unknown_address(&self, addr: u64, len: usize) -> EpError {
        if let Some((&base, &size)) = self.freed.range(..=addr).next_back()
            && addr < base + size as u64
        {
            return EpError::InvalidTensorView {
                reason: format!(
                    "cubecl_ep: device address {addr:#x} refers to an allocation that was \
                     already deallocated (it was {size} bytes based at {base:#x}). A tensor \
                     view outlived the buffer it points into."
                ),
            };
        }
        if addr < ADDRESS_SPACE_BASE {
            return EpError::InvalidTensorView {
                reason: format!(
                    "cubecl_ep: device address {addr:#x} was not produced by this execution \
                     provider (every cubecl address is at or above {ADDRESS_SPACE_BASE:#x}). \
                     A host pointer or a buffer belonging to another EP reached a cubecl \
                     kernel; inputs must be staged onto the cubecl device first."
                ),
            };
        }
        EpError::InvalidTensorView {
            reason: format!(
                "cubecl_ep: device address {addr:#x} (+{len} bytes) does not fall inside any \
                 live cubecl allocation. {} allocations are live.",
                self.regions.len()
            ),
        }
    }

    fn describe_bad_free(&self, addr: u64) -> String {
        if let Some(size) = self.freed.get(&addr) {
            return format!(
                "cubecl_ep: double free of device address {addr:#x} ({size} bytes); this \
                 buffer was already returned to deallocate()."
            );
        }
        format!(
            "cubecl_ep: deallocate() was given device address {addr:#x}, which this provider \
             never allocated. Buffers must be freed by the EP that produced them."
        )
    }
}

/// A device allocation's alignment as recorded at insert time, for callers that
/// need to reconstruct a `DeviceBuffer`.
impl HandleTable {
    pub fn alignment_of(&self, ptr: *const c_void) -> Option<usize> {
        let inner = self.lock();
        inner.regions.get(&(ptr as u64)).map(|region| region.align)
    }
}
