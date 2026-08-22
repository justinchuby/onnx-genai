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
//! # Residual obligation
//!
//! Runtime scoping closes the cross-provider collision. It does not by itself
//! prove that a source stays alive *within* one runtime; that rests on the
//! packed weights being graph initializers, which the executor excludes from
//! its liveness-based frees precisely so their buffers survive every run. That
//! premise is the same one `RepackCache` and `Bf16ConstCache` rest on, and it
//! is stated here rather than left implicit.

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
}

/// Cache of interleaved copies, keyed by `(source address, byte length)`.
///
/// The device ordinal is not part of the key: one runtime is one device, and
/// the cache belongs to the runtime.
#[derive(Debug, Default)]
pub(crate) struct InterleaveCache {
    entries: Mutex<HashMap<(CUdeviceptr, usize), CUdeviceptr>>,
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
        Ok((built, false))
    }

    /// Hand every cached buffer back to `device`'s allocator and forget it.
    ///
    /// Called from the runtime's teardown, which is what bounds an entry's life
    /// by the life of the allocator whose address keys it. "Back to the
    /// allocator" is the honest claim: `CudaRuntime::free_raw` may park a block
    /// in its size-class pool rather than call `cuMemFree`. What matters for
    /// #1726 is that the entry is gone, so no later weight can be served it.
    pub(crate) fn release_all<D: InterleaveDevice>(&self, device: &D) {
        let drained: Vec<CUdeviceptr> = self.lock().drain().map(|(_, ptr)| ptr).collect();
        for ptr in drained {
            // SAFETY: each buffer was allocated by this cache and is freed once.
            unsafe { device.interleave_free(ptr) };
        }
    }

    /// Entries currently held.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().len()
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
        capturing: std::sync::atomic::AtomicBool,
        id: u64,
    }

    impl RecyclingDevice {
        fn new() -> Self {
            static NEXT_ID: AtomicUsize = AtomicUsize::new(1);
            Self {
                next_address: AtomicUsize::new(0x1000),
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed) as u64,
                ..Default::default()
            }
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
            let block = self.contents(src);
            // A device pointer plus a byte length may name a prefix of a block,
            // which is what makes `bytes` part of the cache identity.
            assert!(
                bytes <= block.len(),
                "build read {bytes} bytes past the end of a {}-byte block",
                block.len()
            );
            let source = &block[..bytes];
            // Stand-in for the nibble interleave: any invertible per-byte
            // function whose output identifies the input it came from.
            let built: Vec<u8> = source.iter().map(|b| b.rotate_left(4)).collect();
            self.blocks.lock().unwrap().insert(dst, built);
            Ok(())
        }

        fn interleave_is_capturing(&self) -> Result<bool> {
            Ok(self.capturing.load(Ordering::Relaxed))
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
    }

    /// Mirrors `Drop for CudaRuntime`.
    impl Drop for FakeRuntime<'_> {
        fn drop(&mut self) {
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
}
