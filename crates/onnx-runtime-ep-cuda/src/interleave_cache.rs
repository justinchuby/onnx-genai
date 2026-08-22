//! Owner-scoped cache of derived device buffers keyed by a source device
//! address.
//!
//! # Why this is not a module-global map
//!
//! The int4 interleave is *derived data*: a second device buffer holding the
//! nibble-interleaved form of a packed weight. Caching it under the source
//! weight's device address is only sound while that address still names that
//! weight. An address names a buffer for exactly as long as the allocator that
//! minted it keeps it alive, so the cache is a field of [`CudaRuntime`], the
//! type that owns the allocator, and dies with it.
//!
//! Scoping it that way is the whole point. This cache used to be a process
//! global with a 4096-entry LRU and no teardown, which is the #1726 shape one
//! level down in the stack: a session's weights are freed when its provider
//! goes away, CUDA hands the same device address to the next session's weight,
//! the key matches on `(address, bytes)`, and the GEMV reads the *previous*
//! model's interleaved weights. The right kernel runs and no assertion trips --
//! the numbers are simply not a function of the inputs.
//!
//! The sibling caches in this crate already reason this way. `RepackCache` says
//! it plainly: "the source device address is only meaningful while the owning
//! `MatMulNBitsKernel` is alive: CUDA may reuse it for different constant
//! contents after that kernel/session is released. Keeping the cache on the
//! kernel therefore makes `(address, dimensions)` a valid identity". This cache
//! makes the same argument one level out, at the runtime rather than the
//! kernel, because unlike a repack an interleave is legitimately shared by
//! every kernel variant of a node (prefill and decode are separate kernel
//! instances) and the buffers are large enough that duplicating them per
//! variant would cost more memory than the lever is worth.
//!
//! # Two bounds, and why the runtime is only the outer one
//!
//! Runtime scoping closes the cross-provider collision: a cache never spans two
//! allocators, so an address in a key was always minted by the allocator that
//! is about to be asked about it.
//!
//! It does not close the collision *within* one runtime, and the original
//! version of this module claimed it did — on the premise that the packed
//! weights are graph initializers, which the executor excludes from its
//! liveness-based frees. That premise is true only for the duration of one
//! executor. `Executor::drop` frees every buffer it owns, initializers
//! included, and a provider outlives its executors: sibling plans share one
//! `Arc<dyn ExecutionProvider>`, and the control-flow child-executor cache is a
//! four-entry LRU whose evictions drop an `Executor` and return that plan's
//! initializers to the provider's arena for the next plan to allocate into.
//! An address can therefore be recycled to a different weight many times over
//! while this cache sits there, alive, holding the first one's entry.
//!
//! So the runtime is the outer bound and the source weight is the inner one,
//! and the inner one is load-bearing. [`InterleaveCache::invalidate`] is what
//! enforces it: the provider calls it as it frees each device buffer, so an
//! entry dies at the exact instant its key stops naming a weight.
//!
//! Tying it to the free rather than to a point in teardown is deliberate. The
//! CPU EP's `clear_weight_transpose_caches` and `clear_mlas_packed_caches` are
//! blanket clears ordered before `Executor::drop` frees its buffers, and that
//! ordering is a convention a later edit can reverse. It also cannot be used
//! here: a blanket clear would free interleaves belonging to *other* executors
//! sharing this provider, and a CUDA graph one of those captured has the
//! interleaved pointer baked into its kernel params, so freeing it under a live
//! graph is a use-after-free on replay. Per-buffer invalidation has no ordering
//! to get wrong and touches only what actually died.
//!
//! That argument holds only for weights whose frees this cache is actually told
//! about, and under device weight offload some are not: `MatMulNBits` is one of
//! the boundaries a lazy weight may be paged at, a paged weight is dispatched on
//! the address of its resident page, and pages are retired through
//! `weight_paging`'s own calls rather than through the provider's `deallocate`.
//! Offload and the interleave lever are independent switches, so nothing stops
//! them being on together. [`InterleaveCache::ensure`] therefore refuses to
//! cache at all when the device reports that its frees are not all observed,
//! and the kernel falls back to the ordinary non-interleaved entry. Declining
//! is the whole fix for that configuration: no entry is installed, so there is
//! no stale entry to serve and no per-page bookkeeping to keep correct.

use std::collections::HashMap;
use std::sync::Mutex;

use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{EpError, Result};

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep int4 interleave: {}", message.into()))
}

/// The device operations [`InterleaveCache`] performs.
///
/// A trait rather than direct [`CudaRuntime`] calls so the cache's identity and
/// lifetime rules can be falsified on a host with no GPU, driving the exact
/// `ensure`/`release_all` code the production path runs against a deterministic
/// recycling allocator. A hazard of this class is only visible when an address
/// is actually reused, and a real CUDA allocator reuses addresses on its own
/// schedule -- which is how the equivalent defect went unreproduced for fifteen
/// attempts in #1726.
///
/// [`CudaRuntime`]: crate::runtime::CudaRuntime
pub(crate) trait InterleaveDevice {
    /// An identity for this device that is unique among all devices alive in
    /// this process, and never reused after one goes away.
    ///
    /// A cache binds to the first identity it serves and refuses another, which
    /// is what turns "one cache per runtime" from a convention into something
    /// the code enforces. Reused ids would defeat that -- the whole defect being
    /// guarded against is an identity that gets handed to a second owner -- so
    /// this must not be an address or an ordinal.
    fn interleave_device_id(&self) -> u64;

    /// Allocate `bytes` of device memory.
    fn interleave_alloc(&self, bytes: usize) -> Result<CUdeviceptr>;

    /// Free a pointer previously returned by [`Self::interleave_alloc`].
    ///
    /// # Safety
    /// `ptr` must have come from this device's [`Self::interleave_alloc`] and
    /// must not be freed again.
    unsafe fn interleave_free(&self, ptr: CUdeviceptr);

    /// Write the interleaved form of the `bytes`-byte buffer at `src` into
    /// `dst`, which is at least `bytes` long.
    fn interleave_build(&self, src: CUdeviceptr, dst: CUdeviceptr, bytes: usize) -> Result<()>;

    /// Whether a CUDA graph capture is recording on this device's stream.
    fn interleave_is_capturing(&self) -> Result<bool>;

    /// Whether every free of a weight buffer on this device is observed by
    /// [`InterleaveCache::invalidate`].
    ///
    /// This is the precondition that makes a source address a usable key. The
    /// cache only knows an entry has died because it is told about the free, so
    /// a device with a free path that bypasses the hook cannot be cached for:
    /// the address is recycled behind the cache's back and the next weight to
    /// land there is served the previous one's bytes, which is #1726 exactly.
    ///
    /// The concrete case is device weight offload. A paged weight is dispatched
    /// on the address of its resident page, and pages are retired through
    /// `weight_paging`'s own `free_raw`/`deallocate_span` calls rather than
    /// through the provider's `deallocate`, so nothing reaches this cache. The
    /// two are independent opt-in levers with no mutual exclusion between them,
    /// so the cache declines rather than assuming they are not both on.
    fn interleave_frees_are_observed(&self) -> bool;

    /// Fence work that may still be reading an interleave buffer, before it is
    /// handed back to the allocator.
    ///
    /// Freeing here is not the innocuous act the source weight's free is. The
    /// weight goes onto a deferred-release queue and is only handed out again
    /// once its compute and copy completion events are observed, but an
    /// interleave buffer goes back through `free_raw`, which may park it in a
    /// size-class pool for immediate reuse with no fence at all. The GEMV
    /// launches reading the interleaved copy are the same ones reading the
    /// weight, so without this the block can be reallocated and overwritten on
    /// another stream while a kernel is still reading it.
    fn interleave_drain_before_free(&self) -> Result<()>;
}

/// Cache of interleaved copies, keyed by `(source address, byte length)`.
///
/// The device ordinal is not part of the key: one runtime is one device, and
/// the cache belongs to the runtime.
#[derive(Debug, Default)]
pub(crate) struct InterleaveCache {
    entries: Mutex<HashMap<(CUdeviceptr, usize), CUdeviceptr>>,
    /// `entries.len()`, readable without taking the lock.
    ///
    /// [`Self::invalidate`] runs on every device free the provider performs,
    /// the overwhelming majority of which have nothing to do with int4 weights
    /// -- the lever is off by default, so this is normally zero forever. Stored
    /// with `Release` under the lock and read with `Acquire`, so a reader that
    /// sees a nonzero count also sees the entry that made it nonzero.
    live: std::sync::atomic::AtomicUsize,
    /// Buffers whose entries are gone but which could not be safely freed yet.
    ///
    /// An eviction has two halves: forget the entry, and hand the buffer back.
    /// The first is what closes #1726 and is always safe; the second is not,
    /// during a capture or when the fencing drain fails. Rather than leak the
    /// buffer outright, it is parked here and reclaimed by [`Self::release_all`]
    /// at the runtime's teardown, where the device is synchronized and nothing
    /// can still be reading it.
    retired: Mutex<Vec<CUdeviceptr>>,
    /// The device this cache has bound to, set on first use.
    ///
    /// Belt to the scoping's braces. Placing the cache on [`CudaRuntime`] is
    /// what makes a device address a valid key, but that is a structural
    /// argument, and structure is exactly what a later refactor can undo
    /// without noticing -- this cache was a process global until this commit.
    /// If one is ever shared between two devices again, the second one is
    /// refused here rather than silently served the first one's weights. The
    /// #1726 defect was so expensive precisely because it was silent, so the
    /// backstop fails loud.
    ///
    /// [`CudaRuntime`]: crate::runtime::CudaRuntime
    device: Mutex<Option<u64>>,
}

impl InterleaveCache {
    /// Ensure the interleaved copy of the `bytes`-byte buffer at `packed`
    /// exists, building it once. Returns `(pointer, warm)`; `warm` means the
    /// call allocated, built and synchronized nothing and so is safe to perform
    /// inside a CUDA graph capture.
    ///
    /// A cold miss while capturing is refused rather than served, because
    /// allocating inside a capture invalidates it.
    pub(crate) fn ensure<D: InterleaveDevice>(
        &self,
        device: &D,
        packed: CUdeviceptr,
        bytes: usize,
    ) -> Result<(CUdeviceptr, bool)> {
        self.bind(device.interleave_device_id())?;
        if !device.interleave_frees_are_observed() {
            return Err(error(
                "int4 interleave is unavailable while device weight offload is on: a paged \
                 weight's pages are retired without passing through this cache, so its address \
                 can be recycled to a different weight behind the cache's back (#1726)",
            ));
        }
        let key = (packed, bytes);
        if let Some(hit) = self.lookup(&key) {
            return Ok((hit, true));
        }
        if device.interleave_is_capturing()? {
            return Err(error(
                "int4 interleave cannot allocate during CUDA-graph capture; the weight must be \
                 interleaved during warmup before capture",
            ));
        }
        let built = device.interleave_alloc(bytes)?;
        if let Err(error) = device.interleave_build(packed, built, bytes) {
            // SAFETY: just allocated here and unreachable by anyone else.
            unsafe { device.interleave_free(built) };
            return Err(error);
        }
        let mut entries = self.lock();
        // A racer may have installed the same key while this call was building.
        // Keep whichever landed first so one buffer serves every reader, and
        // free the loser rather than leaking it.
        if let Some(&winner) = entries.get(&key) {
            drop(entries);
            // SAFETY: this call's own duplicate; freed exactly once.
            unsafe { device.interleave_free(built) };
            return Ok((winner, true));
        }
        entries.insert(key, built);
        self.live
            .store(entries.len(), std::sync::atomic::Ordering::Release);
        Ok((built, false))
    }

    /// Forget every entry keyed on the buffer at `source`, freeing what was
    /// built from it.
    ///
    /// Called by the provider as it frees a device buffer, which is the instant
    /// the address stops naming the weight the key was minted for. Doing it
    /// here rather than at some agreed point during teardown is deliberate:
    ///
    /// * It needs no ordering contract. The eviction *is* the free, so there is
    ///   no "clear before you free" convention for a later edit to reverse.
    /// * It is precise, and precision is a safety property here, not a
    ///   nicety. A blanket release at executor teardown would also free
    ///   interleaves belonging to *other* executors on the same provider --
    ///   sibling plans share one -- and a CUDA graph captured by one of those
    ///   siblings has the interleaved pointer baked into its kernel params.
    ///   Freeing it under a live captured graph is a use-after-free on replay.
    ///   Only entries derived from the buffer actually being freed are touched,
    ///   and an executor resets its own graph slot before it frees anything.
    ///
    /// `base`/`len` describe the whole allocation being freed, and any entry
    /// keyed inside it dies. The key is the weight's data pointer, which is the
    /// allocation base plus a byte offset, so matching the base alone would
    /// silently miss any weight held as an offset view of a larger buffer and
    /// leave exactly the stale entry this exists to remove.
    pub(crate) fn invalidate<D: InterleaveDevice>(
        &self,
        device: &D,
        base: CUdeviceptr,
        len: usize,
    ) {
        // The lever is off by default, so this is the entire cost of the hook
        // on the path every device free takes.
        if self.live.load(std::sync::atomic::Ordering::Acquire) == 0 {
            return;
        }
        let end = base.saturating_add(len as CUdeviceptr);
        let mut entries = self.lock();
        // One address can carry entries at several byte lengths: a device
        // pointer plus a length names a prefix, and the length is part of the
        // identity. Every one of them dies with the buffer.
        let doomed: Vec<(CUdeviceptr, usize)> = entries
            .keys()
            .filter(|(packed, _)| *packed >= base && *packed < end)
            .copied()
            .collect();
        let freed: Vec<CUdeviceptr> = doomed
            .iter()
            .filter_map(|key| entries.remove(key))
            .collect();
        self.live
            .store(entries.len(), std::sync::atomic::Ordering::Release);
        drop(entries);
        if freed.is_empty() {
            return;
        }
        // Never synchronize under an active capture: it is illegal, it would
        // invalidate the capture, and freeing here would hand back memory a
        // graph being recorded may already reference. The entries are already
        // out of the map, so #1726 is closed either way; the buffers are parked
        // and reclaimed at teardown. `ensure` refuses a cold miss during capture
        // for the same reason, and this is the other side of that symmetry.
        //
        // Reaching this at all would mean an interleaved weight was freed
        // mid-capture, which is already a use-after-free for the *weight*
        // regardless of this cache -- so it should be unreachable. It is
        // handled rather than asserted because the cost of being wrong is
        // silent corruption, and the cost of the guard is a leaked buffer on a
        // path that should never run.
        if device.interleave_is_capturing().unwrap_or(true) {
            self.retire(freed);
            return;
        }
        // Ordered before the frees, not after: see `interleave_drain_before_free`.
        // A failed drain parks the buffers instead -- the entries are already
        // gone, so #1726 is closed either way, and deferring a free is the
        // strictly safer half of the trade against freeing one a live kernel is
        // still reading.
        if device.interleave_drain_before_free().is_err() {
            self.retire(freed);
            return;
        }
        for ptr in freed {
            // SAFETY: each buffer was allocated by this cache and is freed once
            // -- it was removed from the map above, so no other caller can
            // reach it.
            unsafe { device.interleave_free(ptr) };
        }
    }

    /// Hand every cached buffer back to `device`'s allocator and forget it.
    ///
    /// Called from the runtime's teardown, which is what bounds an entry's life
    /// by the life of the allocator whose address keys it. "Back to the
    /// allocator" is the honest claim: `CudaRuntime::free_raw` may park a block
    /// in its size-class pool rather than call `cuMemFree`. What matters for
    /// #1726 is that the entry is gone, so no later weight can be served it.
    ///
    /// No drain here, unlike [`Self::invalidate`]: this runs from the runtime's
    /// `Drop`, where the primary context is being torn down and the driver
    /// synchronizes the device before any of this memory can be handed to
    /// anyone. There is no next allocation on this runtime to race with.
    pub(crate) fn release_all<D: InterleaveDevice>(&self, device: &D) {
        let mut entries = self.lock();
        let mut drained: Vec<CUdeviceptr> = entries.drain().map(|(_, ptr)| ptr).collect();
        self.live.store(0, std::sync::atomic::Ordering::Release);
        drop(entries);
        // Buffers an eviction could not safely hand back at the time. This is
        // where they are reclaimed, which is what keeps a deferred free a
        // deferral rather than a leak.
        drained.append(&mut self.retired.lock().unwrap_or_else(|e| e.into_inner()));
        for ptr in drained {
            // SAFETY: each buffer was allocated by this cache and is freed once.
            unsafe { device.interleave_free(ptr) };
        }
    }

    /// Park buffers whose entries are already gone but which cannot be handed
    /// back yet. [`Self::release_all`] frees them at the runtime's teardown.
    fn retire(&self, buffers: Vec<CUdeviceptr>) {
        self.retired
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend(buffers);
    }

    /// Entries currently held.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().len()
    }

    /// Buffers evicted but not yet handed back.
    #[cfg(test)]
    pub(crate) fn retired_len(&self) -> usize {
        self.retired.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Bind this cache to `device`, or fail if it already belongs to another.
    fn bind(&self, device: u64) -> Result<()> {
        let mut bound = self.device.lock().unwrap_or_else(|e| e.into_inner());
        match *bound {
            None => {
                *bound = Some(device);
                Ok(())
            }
            Some(owner) if owner == device => Ok(()),
            Some(owner) => Err(error(format!(
                "cache built for device {owner} was asked to serve device {device}; an entry is \
                 keyed by a device address, which only names a weight for the device that minted \
                 it (#1726)"
            ))),
        }
    }

    fn lookup(&self, key: &(CUdeviceptr, usize)) -> Option<CUdeviceptr> {
        self.lock().get(key).copied()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<(CUdeviceptr, usize), CUdeviceptr>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A device whose allocator recycles addresses *deterministically*.
    ///
    /// Freed blocks go on a per-size LIFO free list and the next request of that
    /// size gets the most recently freed address back. That is the point: the
    /// hazard this cache exists to avoid is only observable once an address is
    /// actually reused, and #1726 showed how easily a test can miss it -- a
    /// single free/reallocate pair against a real allocator returns a *fresh*
    /// address and proves nothing. Here reuse is guaranteed by construction, so
    /// a failure is a statement about the cache rather than about the host's
    /// malloc.
    ///
    /// Each block also carries contents, so a test can assert the interleave it
    /// was served describes the weight it asked about rather than merely
    /// checking pointers.
    #[derive(Default)]
    struct RecyclingDevice {
        blocks: Mutex<HashMap<CUdeviceptr, Vec<u8>>>,
        free_lists: Mutex<HashMap<usize, Vec<CUdeviceptr>>>,
        next_address: AtomicUsize,
        allocations: AtomicUsize,
        frees: AtomicUsize,
        builds: AtomicUsize,
        drains: AtomicUsize,
        capturing: std::sync::atomic::AtomicBool,
        /// Blocks a launch is still reading. Freeing one is the use-after-free
        /// [`InterleaveDevice::interleave_drain_before_free`] exists to prevent;
        /// a drain retires the launches and empties this.
        readers: Mutex<Vec<CUdeviceptr>>,
        frees_observed: std::sync::atomic::AtomicBool,
        id: u64,
    }

    impl RecyclingDevice {
        fn new() -> Self {
            static NEXT_ID: AtomicUsize = AtomicUsize::new(1);
            let device = Self {
                next_address: AtomicUsize::new(0x1000),
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed) as u64,
                ..Default::default()
            };
            device.frees_observed.store(true, Ordering::Relaxed);
            device
        }

        /// Model a GEMV launch reading `ptr` that has not yet completed.
        fn launch_reading(&self, ptr: CUdeviceptr) {
            self.readers.lock().unwrap().push(ptr);
        }

        /// Allocate a block holding `contents`, standing in for a weight the
        /// caller (ORT) owns rather than one the cache allocates.
        fn put(&self, contents: &[u8]) -> CUdeviceptr {
            let ptr = self.raw_alloc(contents.len());
            self.blocks.lock().unwrap().insert(ptr, contents.to_owned());
            ptr
        }

        fn free(&self, ptr: CUdeviceptr) {
            // SAFETY: test-only; every caller frees a pointer it owns once.
            unsafe { self.interleave_free(ptr) };
        }

        fn contents(&self, ptr: CUdeviceptr) -> Vec<u8> {
            self.blocks.lock().unwrap().get(&ptr).cloned().unwrap()
        }

        /// Read `bytes` from `ptr`, which may point *into* a block rather than
        /// at its base.
        ///
        /// A weight's `data_ptr()` is its allocation's base plus a byte offset,
        /// so a harness that can only address block bases cannot tell an
        /// eviction keyed on the base from one keyed on the weight -- it bakes
        /// in the coincidence it is supposed to be testing.
        fn read(&self, ptr: CUdeviceptr, bytes: usize) -> Vec<u8> {
            let blocks = self.blocks.lock().unwrap();
            let (base, block) = blocks
                .iter()
                .find(|(base, block)| **base <= ptr && ptr < **base + block.len() as CUdeviceptr)
                .unwrap_or_else(|| panic!("read from {ptr:#x}, which is in no live block"));
            let offset = (ptr - base) as usize;
            assert!(
                offset + bytes <= block.len(),
                "read {bytes} bytes at offset {offset} past the end of a {}-byte block",
                block.len()
            );
            block[offset..offset + bytes].to_vec()
        }

        fn raw_alloc(&self, bytes: usize) -> CUdeviceptr {
            self.allocations.fetch_add(1, Ordering::Relaxed);
            if let Some(recycled) = self
                .free_lists
                .lock()
                .unwrap()
                .get_mut(&bytes)
                .and_then(Vec::pop)
            {
                return recycled;
            }
            // Fresh addresses are spaced so they can never collide with a
            // recycled one by accident.
            self.next_address.fetch_add(0x1_0000, Ordering::Relaxed) as CUdeviceptr
        }

        fn live_blocks(&self) -> usize {
            self.blocks.lock().unwrap().len()
        }

        /// The length of the block at `ptr`, standing in for `DeviceBuffer::len`.
        fn block_len(&self, ptr: CUdeviceptr) -> usize {
            self.blocks.lock().unwrap().get(&ptr).map_or(0, Vec::len)
        }
    }

    impl InterleaveDevice for RecyclingDevice {
        fn interleave_device_id(&self) -> u64 {
            self.id
        }

        fn interleave_alloc(&self, bytes: usize) -> Result<CUdeviceptr> {
            let ptr = self.raw_alloc(bytes);
            self.blocks.lock().unwrap().insert(ptr, vec![0; bytes]);
            Ok(ptr)
        }

        unsafe fn interleave_free(&self, ptr: CUdeviceptr) {
            self.frees.fetch_add(1, Ordering::Relaxed);
            assert!(
                !self.readers.lock().unwrap().contains(&ptr),
                "freed the block at {ptr:#x} while a launch was still reading it; it goes onto \
                 the reuse free list immediately, so the next allocation can overwrite memory a \
                 live kernel is reading"
            );
            let bytes = self
                .blocks
                .lock()
                .unwrap()
                .remove(&ptr)
                .expect("freed a live block")
                .len();
            self.free_lists
                .lock()
                .unwrap()
                .entry(bytes)
                .or_default()
                .push(ptr);
        }

        fn interleave_build(&self, src: CUdeviceptr, dst: CUdeviceptr, bytes: usize) -> Result<()> {
            self.builds.fetch_add(1, Ordering::Relaxed);
            // A device pointer plus a byte length may name a prefix of a block,
            // or a window at an offset inside one, which is what makes `bytes`
            // part of the cache identity.
            let source = self.read(src, bytes);
            // Stand-in for the nibble interleave: any invertible per-byte
            // function whose output identifies the input it came from.
            let built: Vec<u8> = source.iter().map(|b| b.rotate_left(4)).collect();
            self.blocks.lock().unwrap().insert(dst, built);
            Ok(())
        }

        fn interleave_is_capturing(&self) -> Result<bool> {
            Ok(self.capturing.load(Ordering::Relaxed))
        }

        fn interleave_frees_are_observed(&self) -> bool {
            self.frees_observed.load(Ordering::Relaxed)
        }

        fn interleave_drain_before_free(&self) -> Result<()> {
            self.drains.fetch_add(1, Ordering::Relaxed);
            // A drain waits for every launch on the device, so nothing is
            // reading anything once it returns.
            self.readers.lock().unwrap().clear();
            Ok(())
        }
    }

    /// A stand-in for [`CudaRuntime`], which owns a device *and* the cache
    /// keyed by that device's addresses.
    ///
    /// The tests go through this rather than constructing a bare
    /// [`InterleaveCache`], so "two providers" means what it means in
    /// production -- two owners, each with its own cache -- and a change that
    /// put the cache back in one shared place would have to change this type to
    /// keep the tests passing, which is the point.
    ///
    /// [`CudaRuntime`]: crate::runtime::CudaRuntime
    struct FakeRuntime<'a> {
        device: &'a RecyclingDevice,
        interleave: InterleaveCache,
    }

    impl<'a> FakeRuntime<'a> {
        fn new(device: &'a RecyclingDevice) -> Self {
            Self {
                device,
                interleave: InterleaveCache::default(),
            }
        }

        /// Mirrors `CudaRuntime::ensure_interleaved_int4`.
        fn ensure_interleaved_int4(
            &self,
            packed: CUdeviceptr,
            bytes: usize,
        ) -> Result<(CUdeviceptr, bool)> {
            self.interleave.ensure(self.device, packed, bytes)
        }

        fn interleaved_weight_count(&self) -> usize {
            self.interleave.len()
        }

        /// Mirrors `CudaExecutionProvider::deallocate_with_unmapped`: free a
        /// buffer the caller owns, invalidating anything keyed inside it first,
        /// exactly as the provider does.
        ///
        /// The provider knows the allocation, not the weight -- it passes
        /// `buffer.as_ptr()` and `buffer.len()`, while the key was minted from
        /// the weight's `data_ptr()`, which may sit at an offset inside it. The
        /// length is read back from the device here for the same reason, so a
        /// test can key an entry at an offset and still free the buffer the way
        /// production does.
        fn deallocate(&self, base: CUdeviceptr) {
            let len = self.device.block_len(base);
            self.interleave.invalidate(self.device, base, len);
            self.device.free(base);
        }
    }

    /// Mirrors `Drop for CudaRuntime`.
    impl Drop for FakeRuntime<'_> {
        fn drop(&mut self) {
            // Primary-context teardown synchronizes the device, which is the
            // reason `release_all` is allowed not to drain. Modelled here so the
            // fake's teardown is as safe as the real one rather than more
            // permissive than it.
            self.device.interleave_drain_before_free().unwrap();
            self.interleave.release_all(self.device);
        }
    }

    fn weight(seed: u8, bytes: usize) -> Vec<u8> {
        (0..bytes).map(|i| seed.wrapping_add(i as u8)).collect()
    }

    fn interleaved(source: &[u8]) -> Vec<u8> {
        source.iter().map(|b| b.rotate_left(4)).collect()
    }

    /// A short digest of a buffer. Assertions compare digests so a failure
    /// prints something a reader can act on rather than two four-kilobyte byte
    /// vectors.
    fn digest(bytes: &[u8]) -> String {
        let sum = bytes
            .iter()
            .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
        format!("{sum:016x}/{}b", bytes.len())
    }

    /// A weight freed at one device address must not lend its interleave to the
    /// next weight that lands there (#1726 class).
    ///
    /// This is the defect the runtime scoping exists for, expressed as the
    /// lifetime that used to be wrong: with one cache spanning both providers --
    /// which is what a module-global static is -- the second weight is served
    /// the first weight's interleaved bytes. The route is right, the shapes are
    /// right, the pointer is a real live buffer, and the numbers are somebody
    /// else's.
    #[test]
    fn a_recycled_weight_address_must_not_serve_the_previous_weights_interleave() {
        const BYTES: usize = 4096;
        let device = RecyclingDevice::new();

        // Provider 1 loads a weight, routes it, then goes away.
        let first = weight(0x11, BYTES);
        let first_ptr = device.put(&first);
        {
            let provider = FakeRuntime::new(&device);
            let (built, warm) = provider.ensure_interleaved_int4(first_ptr, BYTES).unwrap();
            assert!(!warm, "the first sight of a weight must build it");
            assert_eq!(provider.interleaved_weight_count(), 1);
            assert_eq!(
                digest(&device.contents(built)),
                digest(&interleaved(&first))
            );
        }
        device.free(first_ptr);

        // Provider 2 loads a different weight of the same size. The allocator
        // hands back the address the first weight just released.
        let second = weight(0x77, BYTES);
        let second_ptr = device.put(&second);
        assert_eq!(
            second_ptr, first_ptr,
            "the falsifier is vacuous unless the address is actually recycled"
        );
        assert_ne!(first, second, "the two weights must be distinguishable");

        let provider = FakeRuntime::new(&device);
        let (built, warm) = provider.ensure_interleaved_int4(second_ptr, BYTES).unwrap();
        assert!(
            !warm,
            "a weight this provider has never seen must be built, not served warm"
        );
        assert_eq!(provider.interleaved_weight_count(), 1);
        assert_eq!(
            digest(&device.contents(built)),
            digest(&interleaved(&second)),
            "served the previous weight's interleave at recycled address {second_ptr:#x}"
        );
    }

    /// One runtime outlives many executors, so runtime scoping alone is *not*
    /// enough to keep a device address meaningful.
    ///
    /// This is the gap that scoping the cache to [`CudaRuntime`] left open. A
    /// provider is shared by sibling plans and by the control-flow child
    /// executor cache, which is a 4-entry LRU: evicting a child runs
    /// `Executor::drop`, which hands that child's initializer buffers back to
    /// the provider's arena, and the next child compiled on the same provider
    /// allocates into the addresses just released. The runtime -- and its cache
    /// -- are alive across the whole of that. Bounding an entry by the
    /// allocator's life is therefore the wrong bound; it has to be bounded by
    /// the life of the *source weight* its key names, which is what
    /// [`InterleaveCache::invalidate`] does as the provider frees each buffer.
    ///
    /// Note the single `runtime` here, alive from first line to last: unlike the
    /// two-provider falsifier above, no amount of owner scoping can make this
    /// one pass. Only the release does.
    ///
    /// [`CudaRuntime`]: crate::runtime::CudaRuntime
    #[test]
    fn a_weight_freed_by_one_executor_must_not_lend_its_interleave_to_the_next() {
        const BYTES: usize = 4096;
        let device = RecyclingDevice::new();
        let runtime = FakeRuntime::new(&device);

        // Executor 1 materializes an initializer and routes it.
        let first = weight(0x11, BYTES);
        let first_ptr = device.put(&first);
        let (built, warm) = runtime.ensure_interleaved_int4(first_ptr, BYTES).unwrap();
        assert!(!warm, "the first sight of a weight must build it");
        assert_eq!(
            digest(&device.contents(built)),
            digest(&interleaved(&first)),
            "executor 1 was served an interleave that is not its own weight's"
        );

        // `Executor::drop` frees the buffers it owns through the provider.
        runtime.deallocate(first_ptr);

        // Executor 2 materializes a *different* initializer, which the arena
        // places at the address executor 1 just gave back.
        let second = weight(0xa5, BYTES);
        let second_ptr = device.put(&second);
        assert_eq!(
            second_ptr, first_ptr,
            "the arena must recycle the address or this test proves nothing"
        );
        assert_ne!(first, second, "the two weights must be distinguishable");

        let (rebuilt, warm) = runtime.ensure_interleaved_int4(second_ptr, BYTES).unwrap();
        // Content first: this is the defect itself, and it is what a reader
        // needs to see. `warm` below is the mechanism that caused it.
        assert_eq!(
            digest(&device.contents(rebuilt)),
            digest(&interleaved(&second)),
            "executor 2 was served the interleave of the weight executor 1 freed, at recycled \
             address {second_ptr:#x}"
        );
        assert!(
            !warm,
            "a recycled address must be a cold miss; a warm hit here is the freed weight's entry"
        );
        assert_eq!(
            runtime.interleaved_weight_count(),
            1,
            "executor 2's weight, and only executor 2's, should be cached"
        );
    }

    /// The executor-teardown collision driven past a single free/reallocate
    /// pair, with a weight whose *contents* change under a stable address.
    ///
    /// A child-executor cache churns: four plans resident, each eviction
    /// freeing that plan's initializers and each new plan allocating into the
    /// hole. Six rounds on one runtime, every round asserting the interleave it
    /// was handed describes the weight it asked about, plus a standing check
    /// that the address really is being reused rather than the arena quietly
    /// handing out fresh ones and making the test vacuous.
    #[test]
    fn repeated_executor_teardowns_never_serve_a_previous_plans_interleave() {
        const BYTES: usize = 2048;
        let device = RecyclingDevice::new();
        let runtime = FakeRuntime::new(&device);
        let mut addresses = Vec::new();

        for round in 0..6u8 {
            let w = weight(0x20u8.wrapping_add(round.wrapping_mul(0x31)), BYTES);
            let ptr = device.put(&w);
            addresses.push(ptr);

            let (built, warm) = runtime.ensure_interleaved_int4(ptr, BYTES).unwrap();
            assert_eq!(
                digest(&device.contents(built)),
                digest(&interleaved(&w)),
                "round {round}: served an interleave built for a different weight at {ptr:#x}"
            );
            assert!(
                !warm,
                "round {round}: every plan builds its own weight; a warm hit means the entry \
                 outlived the plan that installed it"
            );

            // This plan is evicted from the child-executor cache, which frees
            // its initializers back through the provider.
            runtime.deallocate(ptr);
        }

        assert!(
            addresses.windows(2).any(|w| w[0] == w[1]),
            "the arena never recycled an address across rounds, so nothing was falsified: {:#x?}",
            addresses
        );
        assert_eq!(
            runtime.interleaved_weight_count(),
            0,
            "every round's entry must have been released with its plan"
        );
    }

    /// Freeing one weight must not disturb another weight's interleave.
    ///
    /// Precision is a safety property here, not an efficiency one. Executors
    /// share a provider, and a CUDA graph captured by one of them has the
    /// interleaved pointer baked into its kernel params, so a blanket release
    /// when some *other* executor tears down would free memory a live graph
    /// replays into. This is the assertion that keeps the eviction narrow, and
    /// it is why the hook is on the individual free rather than on teardown.
    #[test]
    fn freeing_one_weight_leaves_every_other_weights_interleave_alone() {
        const BYTES: usize = 512;
        let device = RecyclingDevice::new();
        let runtime = FakeRuntime::new(&device);

        // Two executors on this one provider, each with its own weight.
        let mine = weight(0x77, BYTES);
        let theirs = weight(0x0e, BYTES);
        let mine_ptr = device.put(&mine);
        let theirs_ptr = device.put(&theirs);
        let (mine_built, _) = runtime.ensure_interleaved_int4(mine_ptr, BYTES).unwrap();
        let (theirs_built, _) = runtime.ensure_interleaved_int4(theirs_ptr, BYTES).unwrap();
        assert_ne!(mine_built, theirs_built);
        assert_eq!(runtime.interleaved_weight_count(), 2);

        // One executor tears down.
        runtime.deallocate(mine_ptr);

        assert_eq!(
            runtime.interleaved_weight_count(),
            1,
            "freeing one weight must evict exactly its own entry"
        );
        let (still, warm) = runtime.ensure_interleaved_int4(theirs_ptr, BYTES).unwrap();
        assert!(warm, "the surviving executor's weight must still be cached");
        assert_eq!(
            still, theirs_built,
            "the surviving executor's interleaved buffer moved; a captured graph holding the old \
             pointer would now replay into freed memory"
        );
        assert_eq!(
            digest(&device.contents(still)),
            digest(&interleaved(&theirs)),
            "the surviving buffer no longer holds its own weight's interleave"
        );

        runtime.deallocate(theirs_ptr);
        assert_eq!(runtime.interleaved_weight_count(), 0);
    }

    /// Freeing a buffer nothing was ever derived from is a no-op, and freeing
    /// the same weight's address twice does not double-free.
    ///
    /// The hook runs on *every* device free the provider performs -- interior
    /// scratch, activations, workspaces -- and with the lever off it has no
    /// work to do on any of them.
    #[test]
    fn invalidating_an_address_the_cache_never_saw_is_a_no_op() {
        const BYTES: usize = 512;
        let device = RecyclingDevice::new();
        let runtime = FakeRuntime::new(&device);

        let scratch = device.put(&weight(0x01, BYTES));
        runtime.deallocate(scratch);
        assert_eq!(runtime.interleaved_weight_count(), 0);

        let w = weight(0x77, BYTES);
        let ptr = device.put(&w);
        runtime.ensure_interleaved_int4(ptr, BYTES).unwrap();
        assert_eq!(runtime.interleaved_weight_count(), 1);

        // A second invalidation of the same address must not free the derived
        // buffer twice; `RecyclingDevice` panics on a double free, so reaching
        // the end of this test is the assertion.
        runtime.interleave.invalidate(&device, ptr, BYTES);
        runtime.interleave.invalidate(&device, ptr, BYTES);
        assert_eq!(runtime.interleaved_weight_count(), 0);
        assert_eq!(
            device.live_blocks(),
            1,
            "only the caller-owned weight should remain live"
        );
        device.free(ptr);
    }

    /// A weight held at an offset inside the buffer being freed must still lose
    /// its interleave.
    ///
    /// The key is the weight's `data_ptr()`, which is the allocation base plus
    /// a byte offset; the provider only knows the allocation. Matching the base
    /// alone leaves exactly the entry this exists to remove, and leaves it
    /// silently -- the eviction reports nothing and the next weight at that
    /// offset is served the dead one's bytes.
    #[test]
    fn a_weight_at_an_offset_inside_the_freed_buffer_loses_its_interleave() {
        const BYTES: usize = 512;
        const OFFSET: CUdeviceptr = BYTES as CUdeviceptr;
        let device = RecyclingDevice::new();
        let runtime = FakeRuntime::new(&device);

        // One allocation holding two weights back to back, the second at an
        // offset -- a registered view, not the root of its buffer.
        let first = weight(0x11, BYTES);
        let second = weight(0x99, BYTES);
        let mut whole = first.clone();
        whole.extend_from_slice(&second);
        let base = device.put(&whole);
        let view = base + OFFSET;

        let (built, _) = runtime.ensure_interleaved_int4(view, BYTES).unwrap();
        assert_eq!(
            digest(&device.contents(built)),
            digest(&interleaved(&second)),
            "the offset view's interleave should be built from the bytes at that offset"
        );

        // The provider frees the allocation, knowing only its base and length.
        runtime.deallocate(base);
        assert_eq!(
            runtime.interleaved_weight_count(),
            0,
            "the entry keyed at base+{OFFSET:#x} must die with the buffer that contained it; \
             matching only the base leaves it alive to serve the next weight at that offset"
        );
    }

    /// An interleave buffer must not go back to the allocator while a launch is
    /// still reading it.
    ///
    /// Its source weight is freed onto a deferred-release queue, held until the
    /// compute and copy completion events are observed. The interleave buffer is
    /// not: it goes through `free_raw`, which may park it in a size-class pool
    /// for immediate reuse with no fence. The GEMV reading the interleaved copy
    /// is the same one reading the weight, so the free has to be fenced or the
    /// block can be handed out and overwritten under a live kernel.
    #[test]
    fn an_interleave_buffer_is_not_freed_under_a_launch_still_reading_it() {
        const BYTES: usize = 1024;
        let device = RecyclingDevice::new();
        let runtime = FakeRuntime::new(&device);

        let ptr = device.put(&weight(0x5a, BYTES));
        let (built, _) = runtime.ensure_interleaved_int4(ptr, BYTES).unwrap();

        // A GEMV is in flight over both the weight and its interleaved copy,
        // as it is whenever the interleaved entry is the one that ran.
        device.launch_reading(ptr);
        device.launch_reading(built);

        // `RecyclingDevice::interleave_free` panics if the block still has a
        // reader, so reaching the assertion is the proof that the drain came
        // first.
        runtime.deallocate(ptr);
        assert_eq!(
            device.drains.load(Ordering::Relaxed),
            1,
            "evicting an entry must drain in-flight work before returning the buffer to the \
             allocator's reuse pool"
        );
        assert_eq!(runtime.interleaved_weight_count(), 0);
    }

    /// A device whose frees are not all observed must not be cached for at all.
    ///
    /// Under weight offload a `MatMulNBits` weight may be paged, and its pages
    /// are retired by `weight_paging` rather than by the provider's
    /// `deallocate`, so no invalidation ever runs for them. The address is
    /// recycled behind the cache's back and the next weight to land there is
    /// served the previous one's bytes -- #1726 again, through a door the
    /// per-free eviction does not cover. Declining is the fix: with no entry
    /// installed there is nothing stale to serve.
    #[test]
    fn a_device_with_unobserved_frees_is_refused_rather_than_cached_for() {
        const BYTES: usize = 512;
        let device = RecyclingDevice::new();
        device.frees_observed.store(false, Ordering::Relaxed);
        let runtime = FakeRuntime::new(&device);

        let ptr = device.put(&weight(0x21, BYTES));
        assert!(
            runtime.ensure_interleaved_int4(ptr, BYTES).is_err(),
            "a device that does not report every weight free must be refused, not served"
        );
        assert_eq!(
            runtime.interleaved_weight_count(),
            0,
            "a refused call must install nothing; an entry here would outlive the page it was \
             keyed on"
        );
        assert_eq!(
            device.builds.load(Ordering::Relaxed),
            0,
            "a refused call must not build either"
        );

        // The page is retired the way `weight_paging` retires one -- straight to
        // the allocator, with nothing telling the cache -- and the address comes
        // back for a different weight.
        device.free(ptr);
        let recycled = device.put(&weight(0xc4, BYTES));
        assert_eq!(
            recycled, ptr,
            "the harness must actually recycle the address for this test to mean anything"
        );
        assert!(
            runtime.ensure_interleaved_int4(recycled, BYTES).is_err(),
            "still refused, so the second weight cannot be served the first one's bytes"
        );
        device.free(recycled);
    }

    /// The same collision, driven long enough that a cache which merely delays
    /// reuse would still be caught.
    ///
    /// One free/reallocate pair is not evidence: #1726 stayed unreproduced
    /// through fifteen attempts because the first pair returned a fresh address.
    /// Eight alternating rounds, each asserting the interleave it received
    /// belongs to the weight it asked about, and a final assertion that the
    /// addresses really were reused across *different* weights.
    #[test]
    fn alternating_weights_on_one_recycled_address_each_get_their_own_interleave() {
        const BYTES: usize = 2048;
        let device = RecyclingDevice::new();
        let weights = [weight(0x05, BYTES), weight(0xa0, BYTES)];
        let mut seen: Vec<(CUdeviceptr, usize)> = Vec::new();
        let mut recycled_across_weights = false;

        for round in 0..8 {
            let which = round % 2;
            let source = &weights[which];
            let ptr = device.put(source);
            let recycled = seen
                .iter()
                .any(|&(seen_ptr, seen_which)| seen_ptr == ptr && seen_which != which);
            recycled_across_weights |= recycled;
            seen.push((ptr, which));

            // A provider per round: one session loading a model, routing it, and
            // being torn down before the next starts.
            let provider = FakeRuntime::new(&device);
            let (built, _) = provider.ensure_interleaved_int4(ptr, BYTES).unwrap();
            assert_eq!(
                digest(&device.contents(built)),
                digest(&interleaved(source)),
                "round {round} (address {ptr:#x} recycled={recycled}): served an interleave \
                 built for a different weight at this address"
            );
            drop(provider);
            device.free(ptr);
        }

        assert!(
            recycled_across_weights,
            "no address was reused across two different weights, so this proved nothing"
        );
    }

    /// Releasing the cache frees every buffer it built.
    ///
    /// The map this replaced held 4096 entries and evicted only under LRU
    /// pressure its own documentation says never arrives, so the interleaved
    /// copies -- a full duplicate of every routed int4 weight -- stayed resident
    /// for the life of the process.
    #[test]
    fn releasing_the_cache_frees_every_buffer_it_built() {
        const BYTES: usize = 512;
        let device = RecyclingDevice::new();
        let sources: Vec<CUdeviceptr> = (0..4).map(|i| device.put(&weight(i, BYTES))).collect();
        let cache = InterleaveCache::default();
        for &ptr in &sources {
            cache.ensure(&device, ptr, BYTES).unwrap();
        }
        assert_eq!(cache.len(), 4);
        assert_eq!(device.live_blocks(), 8, "four weights and four interleaves");

        cache.release_all(&device);

        assert_eq!(cache.len(), 0);
        assert_eq!(
            device.live_blocks(),
            4,
            "release_all must free the interleaves and nothing else"
        );
        assert_eq!(device.frees.load(Ordering::Relaxed), 4);
    }

    /// A second ask for the same live weight is served warm: no allocation, no
    /// build, no sync -- which is what makes a captured graph replay legal.
    #[test]
    fn a_second_ask_for_a_live_weight_is_served_without_building() {
        const BYTES: usize = 256;
        let device = RecyclingDevice::new();
        let ptr = device.put(&weight(3, BYTES));
        let cache = InterleaveCache::default();

        let (first, warm) = cache.ensure(&device, ptr, BYTES).unwrap();
        assert!(!warm);
        let builds = device.builds.load(Ordering::Relaxed);
        let allocations = device.allocations.load(Ordering::Relaxed);

        let (second, warm) = cache.ensure(&device, ptr, BYTES).unwrap();
        assert!(warm, "a cached weight must report warm");
        assert_eq!(first, second);
        assert_eq!(device.builds.load(Ordering::Relaxed), builds);
        assert_eq!(device.allocations.load(Ordering::Relaxed), allocations);

        cache.release_all(&device);
        device.free(ptr);
    }

    /// A cold miss during capture is refused, not built. Allocating inside a
    /// CUDA graph capture invalidates it.
    #[test]
    fn a_cold_miss_during_capture_is_refused() {
        const BYTES: usize = 256;
        let device = RecyclingDevice::new();
        let ptr = device.put(&weight(9, BYTES));
        let cache = InterleaveCache::default();
        device.capturing.store(true, Ordering::Relaxed);

        assert!(cache.ensure(&device, ptr, BYTES).is_err());
        assert_eq!(device.builds.load(Ordering::Relaxed), 0);
        assert_eq!(cache.len(), 0);

        // Warm entries stay legal while capturing: they allocate nothing.
        device.capturing.store(false, Ordering::Relaxed);
        cache.ensure(&device, ptr, BYTES).unwrap();
        device.capturing.store(true, Ordering::Relaxed);
        let (_, warm) = cache.ensure(&device, ptr, BYTES).unwrap();
        assert!(warm);

        device.capturing.store(false, Ordering::Relaxed);
        cache.release_all(&device);
        device.free(ptr);
    }

    /// Byte length is part of the identity: two lengths at the SAME address are
    /// two entries.
    ///
    /// The address must be identical in both asks, or the test proves nothing --
    /// two different addresses miss each other on the address component alone
    /// and would still pass with `bytes` dropped from the key entirely.
    #[test]
    fn byte_length_is_part_of_the_identity() {
        const LONG: usize = 1024;
        const SHORT: usize = 512;
        let device = RecyclingDevice::new();
        let long = weight(0x21, LONG);
        let ptr = device.put(&long);
        let cache = InterleaveCache::default();

        let (built_long, warm) = cache.ensure(&device, ptr, LONG).unwrap();
        assert!(!warm);
        assert_eq!(
            digest(&device.contents(built_long)),
            digest(&interleaved(&long))
        );

        // The same address, a shorter length: a prefix view of one weight must
        // not be served the whole weight's interleave.
        let (built_short, warm) = cache.ensure(&device, ptr, SHORT).unwrap();
        assert!(
            !warm,
            "a different byte length at the same address hit the longer entry, \
             so the length is not part of the identity"
        );
        assert_ne!(built_short, built_long);
        assert_eq!(cache.len(), 2, "the two lengths must be two entries");
        assert_eq!(
            digest(&device.contents(built_short)),
            digest(&interleaved(&long[..SHORT]))
        );

        cache.release_all(&device);
    }

    /// A cache that has served one device refuses another.
    ///
    /// The scoping -- the cache being a field of `CudaRuntime` -- is what makes
    /// a device address a valid key, but that is structure, and a later refactor
    /// can undo structure without noticing; this cache was a process global
    /// until it was moved. If one is ever shared again, the second device is
    /// refused rather than silently served the first one's weights.
    #[test]
    fn a_cache_that_has_served_one_device_refuses_another() {
        const BYTES: usize = 256;
        let first = RecyclingDevice::new();
        let second = RecyclingDevice::new();
        assert_ne!(first.interleave_device_id(), second.interleave_device_id());

        let shared = InterleaveCache::default();
        let ptr = first.put(&weight(1, BYTES));
        shared.ensure(&first, ptr, BYTES).unwrap();

        let error = shared
            .ensure(&second, second.put(&weight(2, BYTES)), BYTES)
            .expect_err("a cache bound to one device must refuse another");
        let message = format!("{error:?}");
        assert!(
            message.contains("1726"),
            "the refusal must name the defect it prevents: {message}"
        );
        assert_eq!(
            second.builds.load(Ordering::Relaxed),
            0,
            "the refused device must not have built anything"
        );

        shared.release_all(&first);
    }

    /// Concurrent first sights of one weight settle on a single buffer, and the
    /// losing racer's allocation is freed rather than leaked.
    #[test]
    fn concurrent_first_sights_keep_one_buffer_and_leak_nothing() {
        const BYTES: usize = 1024;
        let device = RecyclingDevice::new();
        let ptr = device.put(&weight(0x5a, BYTES));
        let cache = InterleaveCache::default();

        let served: Vec<CUdeviceptr> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| scope.spawn(|| cache.ensure(&device, ptr, BYTES).unwrap().0))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let winner = served[0];
        assert!(
            served.iter().all(|&p| p == winner),
            "racers were served different buffers: {served:?}"
        );
        assert_eq!(cache.len(), 1);
        assert_eq!(
            digest(&device.contents(winner)),
            digest(&interleaved(&device.contents(ptr)))
        );
        // Every allocation past the winner's must have been handed back.
        assert_eq!(
            device.allocations.load(Ordering::Relaxed) - device.frees.load(Ordering::Relaxed),
            2,
            "exactly the weight and the surviving interleave stay live"
        );

        cache.release_all(&device);
        device.free(ptr);
    }

    /// An eviction during capture must drop the entry without synchronizing.
    ///
    /// A synchronize is illegal mid-capture and would invalidate the capture,
    /// and the buffer may already be referenced by the graph being recorded.
    /// The entry still has to go -- it is the entry, not the buffer, that
    /// serves the next weight the wrong bytes -- so the eviction happens and
    /// the free does not. `ensure` refuses a cold miss during capture for the
    /// same reason; this is the other half of that symmetry.
    #[test]
    fn an_eviction_during_capture_drops_the_entry_without_synchronizing() {
        const BYTES: usize = 512;
        let device = RecyclingDevice::new();
        let runtime = FakeRuntime::new(&device);

        let ptr = device.put(&weight(0x3c, BYTES));
        let (built, _) = runtime.ensure_interleaved_int4(ptr, BYTES).unwrap();
        assert_eq!(runtime.interleaved_weight_count(), 1);

        // A graph is recording, and it has the interleaved pointer in its
        // kernel params.
        device.capturing.store(true, Ordering::Relaxed);
        device.launch_reading(built);
        runtime.interleave.invalidate(&device, ptr, BYTES);

        assert_eq!(
            runtime.interleaved_weight_count(),
            0,
            "the entry must go even during capture; it is the entry that would serve the next \
             weight at this address the wrong bytes"
        );
        assert_eq!(
            device.drains.load(Ordering::Relaxed),
            0,
            "synchronizing during a capture is illegal and would invalidate it"
        );
        assert_eq!(
            device.frees.load(Ordering::Relaxed),
            0,
            "the buffer must not be handed back under a graph that references it"
        );
        assert_eq!(
            runtime.interleave.retired_len(),
            1,
            "a buffer that could not be handed back must be parked, not dropped on the floor"
        );

        // Deferred, not leaked: the runtime's teardown reclaims it, where the
        // device is synchronized and nothing can still be reading it.
        device.capturing.store(false, Ordering::Relaxed);
        drop(runtime);
        assert_eq!(
            device.frees.load(Ordering::Relaxed),
            1,
            "teardown must reclaim the parked buffer"
        );
        device.free(ptr);
    }

    /// Attaching a pager to a runtime must mark that runtime as paging.
    ///
    /// This is the one link in the refusal no host test can drive: constructing
    /// a `CudaRuntime` needs a device, and this machine class has none. So the
    /// mark lives inside the two constructors themselves -- a pager cannot come
    /// into existence on an unmarked runtime -- and this asserts that it stayed
    /// there, because moving it back out to the call sites is the change that
    /// silently reopens #1726 for anyone who adds a third call site.
    ///
    /// Comments are stripped before scanning. A guard that a comment mentioning
    /// the method can satisfy is not a guard, and "a test that passes for a
    /// reason unrelated to its claim" is the failure mode this whole area keeps
    /// producing.
    #[test]
    // Pure string scanning over a 6,000-line file, with no pointer or aliasing
    // content for Miri to check -- it only makes the Miri run take minutes.
    #[cfg_attr(miri, ignore)]
    fn the_pager_constructors_mark_the_runtime_as_paging() {
        let source = include_str!("weight_paging.rs");
        let code: String = source
            .lines()
            .map(|line| match line.find("//") {
                Some(at) => &line[..at],
                None => line,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let marker = "set_weights_may_be_paged()";

        // The two types that can page a weight off this runtime. Anything that
        // pages goes through one of them.
        for owner in ["CudaWeightPager", "CudaWeightResidency"] {
            let impl_at = code
                .find(&format!(
                    "impl<'a, S: MmapRegionSource + ?Sized> {owner}<'a, S> {{"
                ))
                .or_else(|| code.find(&format!("impl {owner} {{")))
                .unwrap_or_else(|| {
                    panic!(
                        "could not find the `impl {owner}` block in weight_paging.rs; if it was \
                         renamed, update this test rather than deleting it -- otherwise it \
                         passes by finding nothing to check"
                    )
                });
            let ctor_at = code[impl_at..]
                .find("pub fn new(")
                .map(|at| impl_at + at)
                .unwrap_or_else(|| panic!("`{owner}` has no `new` constructor any more"));
            // Skip the signature line, which itself ends in `-> Self {`.
            let sig_end = code[ctor_at..]
                .find('\n')
                .map(|at| ctor_at + at)
                .expect("a constructor signature is followed by a body");
            // The constructor body ends at the struct literal it returns.
            let body_end = code[sig_end..]
                .find("Self {")
                .map(|at| sig_end + at)
                .unwrap_or_else(|| panic!("`{owner}::new` no longer returns a `Self` literal"));
            assert!(
                code[sig_end..body_end].contains(marker),
                "`{owner}::new` does not call {marker}. A paged weight's pages are retired by \
                 weight_paging rather than by the provider's deallocate, so the interleave cache \
                 is never told the address died -- it has to refuse to cache on this runtime at \
                 all, and it only knows to when this is called. Marking inside the constructor \
                 is what makes that unmissable; moving it to the call sites means the next call \
                 site added silently reopens #1726."
            );
        }
    }
}
