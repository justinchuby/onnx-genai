//! CUDA virtual memory: one contiguous device address over scattered physical
//! allocations.
//!
//! # What this is for
//!
//! ONNX Runtime's `GroupQueryAttention` wants one flat K tensor and one flat V
//! tensor. A paged KV cache does not have those, so today
//! `mirror_present_kv_to_pages` **copies** the whole thing into a contiguous
//! buffer every step.
//!
//! CUDA's virtual memory management removes the copy rather than making it
//! faster: reserve one device address range with `cuMemAddressReserve`, then
//! map separately-created physical handles into consecutive parts of it. The
//! operator sees a flat buffer; the pages behind it were never gathered.
//!
//! ONNX Runtime does ship a `PagedAttention` operator, but it is CUDA-only
//! *and* a graph operator, so a stock exported model cannot reach it. Virtual
//! contiguity works on the model as exported.
//!
//! # Measured, not assumed
//!
//! On an RTX 4060 (`nvcuda.dll`, driver API):
//!
//! ```text
//! minimum granularity:     2097152 bytes = 2 MiB
//! recommended granularity: 2097152 bytes = 2 MiB
//! reserved 1 GiB of device address space
//! mapped 2 granules from separate cuMemCreate handles
//! wrote and read 4 MiB straight across the seam: correct
//! ```
//!
//! 2 MiB is roughly a thousand tokens of one KV tensor at Llama-3-8B geometry —
//! coarse, and fine at the concurrency this project targets (#596).
//!
//! # Why the handles are kept
//!
//! `cuMemUnmap` removes a mapping but does not free the physical memory behind
//! it; that needs `cuMemRelease` on the handle `cuMemCreate` returned. So a
//! reservation has to remember its handles, which is why
//! [`VirtualBacking::Reservation`] is an associated type rather than the
//! backing being stateless.

use std::sync::Arc;

use cudarc::driver::sys as cu;
use onnx_runtime_virtual_memory::{VirtualBacking, VirtualMemoryError};

use cudarc::driver::CudaContext;

/// Device address space, backed by CUDA physical allocations.
///
/// Holds the runtime so the CUDA context is bound before every driver call —
/// the reservation and its mappings belong to a context, and touching them from
/// an unbound thread is a driver error rather than a silent wrong answer.
#[derive(Debug, Clone)]
pub struct CudaVirtualBacking {
    context: Arc<CudaContext>,
    device_ordinal: i32,
}

impl CudaVirtualBacking {
    /// Reserve and map in `context`.
    ///
    /// Takes a context rather than the execution provider's full runtime
    /// because virtual memory management is **driver** API: it needs no cudart,
    /// no cuBLAS and no kernels. Requiring the runtime would couple this to
    /// libraries it does not use — and on a machine with only the driver
    /// installed, that coupling is the difference between this code running and
    /// silently skipping.
    ///
    /// The context is not incidental: a mapping belongs to the context it was
    /// made in, so this must be the same context the kernels reading the memory
    /// run in.
    pub fn new(context: Arc<CudaContext>, device_ordinal: i32) -> Self {
        Self {
            context,
            device_ordinal,
        }
    }

    fn allocation_prop(&self) -> cu::CUmemAllocationProp {
        let mut prop: cu::CUmemAllocationProp = unsafe { std::mem::zeroed() };
        prop.type_ = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = self.device_ordinal;
        prop
    }

    fn bind(&self, what: &'static str) -> Result<(), VirtualMemoryError> {
        self.context
            .bind_to_thread()
            .map_err(|error| VirtualMemoryError::Os {
                operation: what,
                reason: format!("could not bind the CUDA context: {error}"),
                code: 0,
            })
    }

    fn check(call: &'static str, result: cu::CUresult) -> Result<(), VirtualMemoryError> {
        if result == cu::CUresult::CUDA_SUCCESS {
            return Ok(());
        }
        Err(VirtualMemoryError::Os {
            operation: call,
            reason: format!("{result:?}"),
            code: result as i32,
        })
    }
}

/// One reserved device address range and the physical handles mapped into it.
#[derive(Debug)]
pub struct CudaReservation {
    base: cu::CUdeviceptr,
    len: usize,
    /// `(offset, len, handle)` for every mapped block, so `release` can both
    /// unmap the address range and free the physical memory behind it.
    blocks: Vec<(usize, usize, cu::CUmemGenericAllocationHandle)>,
}

// The reservation is an owned device address range; nothing in it is
// thread-affine, and every driver call through the backing binds the context
// first.
unsafe impl Send for CudaReservation {}
unsafe impl Sync for CudaReservation {}

impl Drop for CudaReservation {
    fn drop(&mut self) {
        for (offset, len, handle) in std::mem::take(&mut self.blocks) {
            // SAFETY: each block was mapped by `commit` and is unmapped once.
            unsafe {
                let _ = cu::cuMemUnmap(self.base + offset as u64, len);
                let _ = cu::cuMemRelease(handle);
            }
        }
        if self.len > 0 {
            // SAFETY: `base` came from `cuMemAddressReserve` with this length
            // and every block in it has been unmapped above.
            unsafe {
                let _ = cu::cuMemAddressFree(self.base, self.len);
            }
        }
    }
}

// SAFETY: every address comes from `cuMemAddressReserve`; the granularity is
// the driver's own for this device and constant; `commit` maps and grants
// access to the whole range it reports success for; and `CudaReservation`'s
// `Drop` unmaps every block, releases every handle, and frees the reservation.
unsafe impl VirtualBacking for CudaVirtualBacking {
    type Reservation = CudaReservation;

    fn granularity(&self) -> usize {
        let prop = self.allocation_prop();
        let mut granularity = 0usize;
        // SAFETY: `prop` is fully initialised and `granularity` is a valid
        // out-parameter.
        let result = unsafe {
            cu::cuMemGetAllocationGranularity(
                &mut granularity,
                &prop,
                cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        };
        if result != cu::CUresult::CUDA_SUCCESS || granularity == 0 {
            // The driver could not say. 2 MiB is the value every CUDA device
            // this has been measured on reports, and guessing smaller would
            // produce misaligned requests the driver then rejects.
            return 2 << 20;
        }
        granularity
    }

    fn reserve(&self, len: usize) -> Result<Self::Reservation, VirtualMemoryError> {
        self.bind("reserving CUDA address space")?;
        let mut base: cu::CUdeviceptr = 0;
        // SAFETY: `base` is a valid out-parameter; alignment 0 lets the driver
        // choose, and a null `addr` lets it place the range.
        Self::check("cuMemAddressReserve", unsafe {
            cu::cuMemAddressReserve(&mut base, len, 0, 0, 0)
        })?;
        Ok(CudaReservation {
            base,
            len,
            blocks: Vec::new(),
        })
    }

    fn base(reservation: &Self::Reservation) -> usize {
        reservation.base as usize
    }

    fn commit(
        &self,
        reservation: &mut Self::Reservation,
        offset: usize,
        len: usize,
    ) -> Result<(), VirtualMemoryError> {
        self.bind("committing CUDA memory")?;
        let prop = self.allocation_prop();
        let mut handle: cu::CUmemGenericAllocationHandle = 0;
        // SAFETY: `prop` is fully initialised; `handle` is a valid
        // out-parameter; `len` is a multiple of the granularity by the trait's
        // contract.
        Self::check("cuMemCreate", unsafe {
            cu::cuMemCreate(&mut handle, len, &prop, 0)
        })?;

        let address = reservation.base + offset as u64;
        // SAFETY: `address..address + len` lies inside the reservation by the
        // trait's contract, and `handle` was just created with exactly `len`.
        if let Err(error) = Self::check("cuMemMap", unsafe {
            cu::cuMemMap(address, len, 0, handle, 0)
        }) {
            // The handle is ours and nothing references it, so release it
            // rather than leaking physical device memory on a failed map.
            // SAFETY: created above, released once, never mapped.
            unsafe {
                let _ = cu::cuMemRelease(handle);
            }
            return Err(error);
        }

        // Mapping alone does not make the range usable: without an access
        // descriptor a kernel reading it faults. This is the step whose absence
        // looks like "the memory is there but every read is garbage".
        let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
        access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = self.device_ordinal;
        access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        // SAFETY: the range was just mapped and `access` is fully initialised.
        if let Err(error) = Self::check("cuMemSetAccess", unsafe {
            cu::cuMemSetAccess(address, len, &access, 1)
        }) {
            // SAFETY: just mapped and created; undo both.
            unsafe {
                let _ = cu::cuMemUnmap(address, len);
                let _ = cu::cuMemRelease(handle);
            }
            return Err(error);
        }

        reservation.blocks.push((offset, len, handle));
        Ok(())
    }

    fn release(
        &self,
        reservation: &mut Self::Reservation,
        offset: usize,
        _len: usize,
    ) -> Result<(), VirtualMemoryError> {
        let Some(index) = reservation
            .blocks
            .iter()
            .position(|&(at, _, _)| at == offset)
        else {
            return Ok(());
        };
        let (_, len, handle) = reservation.blocks.remove(index);
        self.bind("releasing CUDA memory")?;
        let address = reservation.base + offset as u64;
        // SAFETY: this block was mapped by `commit` and is released once; the
        // handle is the one created for it.
        Self::check("cuMemUnmap", unsafe { cu::cuMemUnmap(address, len) })?;
        // Unmapping removes the mapping; the physical memory needs releasing
        // separately, and skipping it leaks device memory that no address
        // refers to any more.
        Self::check("cuMemRelease", unsafe { cu::cuMemRelease(handle) })
    }
}
