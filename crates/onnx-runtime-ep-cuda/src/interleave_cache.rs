//! Runtime-scoped immutable int4 conversions. Source-address validity and
//! allocation lifetime are separate: invalidation prevents reuse of an old
//! conversion, while a kernel or captured graph may still own that conversion.
//! Native owners retire through the same governed, stream-ordered release
//! queue as other retained CUDA artifacts; this cache has no raw free path.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::runtime::NativeDeviceAllocation;
use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{DeviceGraphResource, EpError, Result};

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep int4 interleave: {}", message.into()))
}

/// Only explicit constant-input metadata can authorize address-keyed reuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ImmutablePackedWeight {
    ptr: CUdeviceptr,
    bytes: usize,
}

impl ImmutablePackedWeight {
    pub(crate) fn from_input(ptr: CUdeviceptr, bytes: usize, constant: bool) -> Option<Self> {
        constant.then_some(Self { ptr, bytes })
    }
}

pub(crate) trait InterleaveAllocation: std::fmt::Debug + Send + Sync + 'static {
    fn ptr(&self) -> CUdeviceptr;
}

impl InterleaveAllocation for NativeDeviceAllocation {
    fn ptr(&self) -> CUdeviceptr {
        self.ptr()
    }
}

/// Device work stays generic so host tests exercise the real cache against a
/// deterministic recycling allocator, including deferred physical retirement.
pub(crate) trait InterleaveDevice {
    type Allocation: InterleaveAllocation;
    fn interleave_device_id(&self) -> u64;
    fn interleave_alloc(&self, bytes: usize) -> Result<Self::Allocation>;
    fn interleave_build(&self, src: CUdeviceptr, dst: CUdeviceptr, bytes: usize) -> Result<()>;
    fn interleave_is_capturing(&self) -> Result<bool>;
    /// Paged weights bypass source invalidation and cannot use this cache.
    fn interleave_frees_are_observed(&self) -> bool;
}

#[derive(Debug)]
pub(crate) struct InterleaveEntry<A: InterleaveAllocation = NativeDeviceAllocation> {
    source: ImmutablePackedWeight,
    device: u64,
    valid: AtomicBool,
    allocation: A,
}

impl<A: InterleaveAllocation> InterleaveEntry<A> {
    pub(crate) fn ptr(&self) -> CUdeviceptr {
        self.allocation.ptr()
    }

    pub(crate) fn matches(&self, source: ImmutablePackedWeight, device: u64) -> bool {
        self.source == source && self.device == device && self.valid.load(Ordering::Acquire)
    }

    pub(crate) fn identity(owner: &Arc<Self>) -> usize {
        Arc::as_ptr(owner) as usize
    }

    pub(crate) fn device_graph_resource(owner: &Arc<Self>) -> DeviceGraphResource {
        DeviceGraphResource::new(Self::identity(owner), Arc::clone(owner))
    }
}

type Entries<A> = HashMap<(CUdeviceptr, usize), Arc<InterleaveEntry<A>>>;

#[derive(Debug)]
pub(crate) struct InterleaveCache<A: InterleaveAllocation = NativeDeviceAllocation> {
    entries: Mutex<Entries<A>>,
    // Source deallocations usually have no interleave to invalidate.
    live: AtomicUsize,
    device: Mutex<Option<u64>>,
}

impl<A: InterleaveAllocation> Default for InterleaveCache<A> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            live: AtomicUsize::new(0),
            device: Mutex::new(None),
        }
    }
}

impl<A: InterleaveAllocation> InterleaveCache<A> {
    /// Returns a real owner and whether conversion was already prepared.
    /// Caller keeps the owner through launch; captured graphs keep their own pin.
    pub(crate) fn prepare<D: InterleaveDevice<Allocation = A>>(
        &self,
        device: &D,
        source: ImmutablePackedWeight,
    ) -> Result<(Arc<InterleaveEntry<A>>, bool)> {
        let device_id = device.interleave_device_id();
        self.bind(device_id)?;
        if !device.interleave_frees_are_observed() {
            return Err(error(
                "int4 interleave is unavailable while device weight offload is on: paged \
                 weights can reuse source addresses without cache invalidation (#1726); \
                 disable interleave or device weight offload",
            ));
        }
        let key = (source.ptr, source.bytes);
        if let Some(hit) = self.lock().get(&key).cloned() {
            return Ok((hit, true));
        }
        if device.interleave_is_capturing()? {
            return Err(error(
                "cannot prepare an int4 interleave during CUDA graph capture; abort recording \
                 and eagerly warm the exact immutable weight before capture",
            ));
        }
        let allocation = device.interleave_alloc(source.bytes)?;
        // A failed build drops its owner into the ordinary retirement queue,
        // including failures after device work may have been submitted.
        device.interleave_build(source.ptr, allocation.ptr(), source.bytes)?;
        let built = Arc::new(InterleaveEntry {
            source,
            device: device_id,
            valid: AtomicBool::new(true),
            allocation,
        });
        let mut entries = self.lock();
        if let Some(winner) = entries.get(&key).cloned() {
            drop(entries);
            drop(built); // May recursively invalidate through native retirement.
            return Ok((winner, true));
        }
        entries.insert(key, Arc::clone(&built));
        self.live.store(entries.len(), Ordering::Release);
        Ok((built, false))
    }

    /// Revoke source-name validity first. Drop owners only after unlocking:
    /// native retirement itself calls this hook for the derived allocation.
    /// Graph pins preserve old derived bytes without making them cache hits.
    pub(crate) fn invalidate(&self, base: CUdeviceptr, len: usize) {
        if self.live.load(Ordering::Acquire) == 0 {
            return;
        }
        let end = base.saturating_add(len as CUdeviceptr);
        let mut entries = self.lock();
        let doomed: Vec<_> = entries
            .keys()
            .filter(|(ptr, _)| *ptr >= base && *ptr < end)
            .copied()
            .collect();
        let removed: Vec<_> = doomed
            .iter()
            .filter_map(|key| entries.remove(key))
            .collect();
        for entry in &removed {
            entry.valid.store(false, Ordering::Release);
        }
        self.live.store(entries.len(), Ordering::Release);
        drop(entries);
        drop(removed);
    }

    pub(crate) fn release_all(&self) {
        let mut entries = self.lock();
        let removed = std::mem::take(&mut *entries);
        for entry in removed.values() {
            entry.valid.store(false, Ordering::Release);
        }
        self.live.store(0, Ordering::Release);
        drop(entries);
        drop(removed);
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().len()
    }

    fn bind(&self, device: u64) -> Result<()> {
        let mut bound = self.device.lock().unwrap_or_else(|e| e.into_inner());
        match *bound {
            None => {
                *bound = Some(device);
                Ok(())
            }
            Some(owner) if owner == device => Ok(()),
            Some(owner) => Err(error(format!(
                "cache built for device {owner} was asked to serve device {device}; \
                 source addresses cannot cross allocator identities (#1726); use this runtime's cache"
            ))),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Entries<A>> {
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
    #[derive(Clone, Debug, Default)]
    struct RecyclingDevice(Arc<RecyclingState>);

    impl std::ops::Deref for RecyclingDevice {
        type Target = RecyclingState;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    #[derive(Debug, Default)]
    struct RecyclingState {
        blocks: Mutex<HashMap<CUdeviceptr, Vec<u8>>>,
        free_lists: Mutex<HashMap<usize, Vec<CUdeviceptr>>>,
        next_address: AtomicUsize,
        allocations: AtomicUsize,
        frees: AtomicUsize,
        builds: AtomicUsize,
        fail_build: AtomicBool,
        build_barrier: Mutex<Option<Arc<std::sync::Barrier>>>,
        invalidate_on_release: Mutex<Option<std::sync::Weak<InterleaveCache<TestAllocation>>>>,
        completions: AtomicUsize,
        pending: Mutex<Vec<CUdeviceptr>>,
        capturing: std::sync::atomic::AtomicBool,
        /// Blocks a launch is still reading. Retirements remain queued until
        /// the explicit completion observation clears these readers.
        readers: Mutex<Vec<CUdeviceptr>>,
        frees_observed: std::sync::atomic::AtomicBool,
        id: u64,
    }

    impl RecyclingDevice {
        fn new() -> Self {
            static NEXT_ID: AtomicUsize = AtomicUsize::new(1);
            let device = Self(Arc::new(RecyclingState {
                next_address: AtomicUsize::new(0x1000),
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed) as u64,
                ..Default::default()
            }));
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
            if self.capturing.load(Ordering::Relaxed) || self.readers.lock().unwrap().contains(&ptr)
            {
                self.pending.lock().unwrap().push(ptr);
            } else {
                self.physical_free(ptr);
            }
        }

        fn complete_readers(&self) {
            self.completions.fetch_add(1, Ordering::Relaxed);
            self.readers.lock().unwrap().clear();
            let pending = std::mem::take(&mut *self.pending.lock().unwrap());
            for ptr in pending {
                self.physical_free(ptr);
            }
        }

        fn physical_free(&self, ptr: CUdeviceptr) {
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

    #[derive(Debug)]
    struct TestAllocation {
        device: RecyclingDevice,
        ptr: CUdeviceptr,
    }

    impl InterleaveAllocation for TestAllocation {
        fn ptr(&self) -> CUdeviceptr {
            self.ptr
        }
    }

    impl Drop for TestAllocation {
        fn drop(&mut self) {
            let cache = self.device.invalidate_on_release.lock().unwrap().clone();
            if let Some(cache) = cache.and_then(|cache| cache.upgrade()) {
                cache.invalidate(self.ptr, self.device.block_len(self.ptr));
            }
            self.device.free(self.ptr);
        }
    }

    impl InterleaveDevice for RecyclingDevice {
        type Allocation = TestAllocation;

        fn interleave_device_id(&self) -> u64 {
            self.id
        }

        fn interleave_alloc(&self, bytes: usize) -> Result<TestAllocation> {
            let ptr = self.raw_alloc(bytes);
            self.blocks.lock().unwrap().insert(ptr, vec![0; bytes]);
            Ok(TestAllocation {
                device: self.clone(),
                ptr,
            })
        }

        fn interleave_build(&self, src: CUdeviceptr, dst: CUdeviceptr, bytes: usize) -> Result<()> {
            self.builds.fetch_add(1, Ordering::Relaxed);
            let barrier = self.build_barrier.lock().unwrap().clone();
            if let Some(barrier) = barrier {
                barrier.wait();
            }
            if self.fail_build.load(Ordering::Relaxed) {
                self.launch_reading(dst);
                return Err(error(
                    "injected conversion failure after submission; abort warmup",
                ));
            }
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
        interleave: InterleaveCache<TestAllocation>,
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
            snapshot(&self.interleave, self.device, packed, bytes)
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
            self.interleave.invalidate(base, len);
            self.device.free(base);
        }
    }

    /// Mirrors `Drop for CudaRuntime`.
    impl Drop for FakeRuntime<'_> {
        fn drop(&mut self) {
            // Model the runtime's final completion boundary. Ordinary source
            // invalidation never completes readers on their behalf.
            self.device.complete_readers();
            self.interleave.release_all();
        }
    }

    fn snapshot(
        cache: &InterleaveCache<TestAllocation>,
        device: &RecyclingDevice,
        ptr: CUdeviceptr,
        bytes: usize,
    ) -> Result<(CUdeviceptr, bool)> {
        cache
            .prepare(
                device,
                ImmutablePackedWeight::from_input(ptr, bytes, true).unwrap(),
            )
            .map(|(owner, warm)| (owner.ptr(), warm))
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
            "the arena never recycled an address across rounds, so nothing was falsified: {addresses:#x?}"
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
        runtime.interleave.invalidate(ptr, BYTES);
        runtime.interleave.invalidate(ptr, BYTES);
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

    /// Invalidation removes names immediately but queues physical retirement
    /// until the submitted reader completes, without synchronizing inference.
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

        runtime.deallocate(ptr);
        assert_eq!(
            device.completions.load(Ordering::Relaxed),
            0,
            "source invalidation must not synchronize inference streams"
        );
        assert_eq!(runtime.interleaved_weight_count(), 0);
        assert_eq!(device.pending.lock().unwrap().len(), 2);
        assert_eq!(device.frees.load(Ordering::Relaxed), 0);
        let other = device.put(&weight(0x77, BYTES));
        assert_ne!(other, ptr);
        assert_ne!(other, built, "in-flight derived backing was recycled");
        device.complete_readers();
        assert_eq!(device.frees.load(Ordering::Relaxed), 2);
        assert!(device.blocks.lock().unwrap().contains_key(&other));
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
            snapshot(&cache, &device, ptr, BYTES).unwrap();
        }
        assert_eq!(cache.len(), 4);
        assert_eq!(device.live_blocks(), 8, "four weights and four interleaves");

        cache.release_all();

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

        let (first, warm) = snapshot(&cache, &device, ptr, BYTES).unwrap();
        assert!(!warm);
        let builds = device.builds.load(Ordering::Relaxed);
        let allocations = device.allocations.load(Ordering::Relaxed);

        let (second, warm) = snapshot(&cache, &device, ptr, BYTES).unwrap();
        assert!(warm, "a cached weight must report warm");
        assert_eq!(first, second);
        assert_eq!(device.builds.load(Ordering::Relaxed), builds);
        assert_eq!(device.allocations.load(Ordering::Relaxed), allocations);

        cache.release_all();
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

        assert!(snapshot(&cache, &device, ptr, BYTES).is_err());
        assert_eq!(device.builds.load(Ordering::Relaxed), 0);
        assert_eq!(cache.len(), 0);

        // Warm entries stay legal while capturing: they allocate nothing.
        device.capturing.store(false, Ordering::Relaxed);
        snapshot(&cache, &device, ptr, BYTES).unwrap();
        device.capturing.store(true, Ordering::Relaxed);
        let (_, warm) = snapshot(&cache, &device, ptr, BYTES).unwrap();
        assert!(warm);

        device.capturing.store(false, Ordering::Relaxed);
        cache.release_all();
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

        let (built_long, warm) = snapshot(&cache, &device, ptr, LONG).unwrap();
        assert!(!warm);
        assert_eq!(
            digest(&device.contents(built_long)),
            digest(&interleaved(&long))
        );

        // The same address, a shorter length: a prefix view of one weight must
        // not be served the whole weight's interleave.
        let (built_short, warm) = snapshot(&cache, &device, ptr, SHORT).unwrap();
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

        cache.release_all();
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
        snapshot(&shared, &first, ptr, BYTES).unwrap();

        let error = snapshot(&shared, &second, second.put(&weight(2, BYTES)), BYTES)
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

        shared.release_all();
    }

    /// Concurrent first sights of one weight settle on a single buffer, and the
    /// losing racer's allocation is freed rather than leaked.
    #[test]
    fn concurrent_first_sights_keep_one_buffer_and_leak_nothing() {
        const BYTES: usize = 1024;
        let device = RecyclingDevice::new();
        let ptr = device.put(&weight(0x5a, BYTES));
        let cache = InterleaveCache::default();

        *device.build_barrier.lock().unwrap() = Some(Arc::new(std::sync::Barrier::new(8)));
        let served: Vec<CUdeviceptr> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| scope.spawn(|| snapshot(&cache, &device, ptr, BYTES).unwrap().0))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        assert_eq!(
            device.builds.load(Ordering::Relaxed),
            8,
            "all racers must actually build"
        );
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

        cache.release_all();
        device.free(ptr);
    }

    /// An eviction during capture must drop the entry without synchronizing.
    ///
    /// A synchronize is illegal mid-capture and would invalidate the capture,
    /// and the buffer may already be referenced by the graph being recorded.
    /// The entry still has to go -- it is the entry, not the buffer, that
    /// serves the next weight the wrong bytes -- so the eviction happens and
    /// the free does not. `prepare` refuses a cold miss during capture for the
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
        runtime.interleave.invalidate(ptr, BYTES);

        assert_eq!(
            runtime.interleaved_weight_count(),
            0,
            "the entry must go even during capture; it is the entry that would serve the next \
             weight at this address the wrong bytes"
        );
        assert_eq!(
            device.completions.load(Ordering::Relaxed),
            0,
            "synchronizing during a capture is illegal and would invalidate it"
        );
        assert_eq!(
            device.frees.load(Ordering::Relaxed),
            0,
            "the buffer must not be handed back under a graph that references it"
        );
        assert_eq!(
            device.pending.lock().unwrap().len(),
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

    #[test]
    fn graph_pin_keeps_old_conversion_while_reused_source_gets_new_contents() {
        const BYTES: usize = 256;
        let device = RecyclingDevice::new();
        let cache = InterleaveCache::default();
        let a = weight(0x15, BYTES);
        let source_ptr = device.put(&a);
        let source = ImmutablePackedWeight::from_input(source_ptr, BYTES, true).unwrap();
        let (old, _) = cache.prepare(&device, source).unwrap();
        let old_ptr = old.ptr();
        let weak = Arc::downgrade(&old);
        let pin = InterleaveEntry::device_graph_resource(&old);
        let binding = arc_swap::ArcSwapOption::from(Some(old));
        assert_eq!(
            pin.identity(),
            InterleaveEntry::identity(binding.load().as_ref().unwrap())
        );

        cache.invalidate(source_ptr, BYTES);
        assert!(!binding.load().as_ref().unwrap().matches(source, device.id));
        device.free(source_ptr);
        let b = weight(0x72, BYTES);
        let reused = device.put(&b);
        assert_eq!(
            source_ptr, reused,
            "address reuse is required to falsify stale hits"
        );
        let (new, warm) = cache.prepare(&device, source).unwrap();
        assert!(!warm);
        assert_ne!(old_ptr, new.ptr());
        assert_eq!(device.contents(new.ptr()), interleaved(&b));
        assert_eq!(device.contents(old_ptr), interleaved(&a));
        binding.store(None);
        assert!(
            weak.upgrade().is_some(),
            "the graph must own the old conversion"
        );
        cache.release_all();
        drop(new);
        assert_eq!(device.contents(old_ptr), interleaved(&a));
        drop(pin);
        assert!(weak.upgrade().is_none());
        assert!(!device.blocks.lock().unwrap().contains_key(&old_ptr));
        device.free(reused);
        assert_eq!(
            device.allocations.load(Ordering::Relaxed),
            device.frees.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn warm_binding_reuses_owner_without_admission_or_retirement() {
        let device = RecyclingDevice::new();
        let cache = InterleaveCache::default();
        let ptr = device.put(&weight(0x31, 256));
        let source = ImmutablePackedWeight::from_input(ptr, 256, true).unwrap();
        let (owner, _) = cache.prepare(&device, source).unwrap();
        let binding = arc_swap::ArcSwapOption::from(Some(owner));
        let counts = (
            device.allocations.load(Ordering::Relaxed),
            device.frees.load(Ordering::Relaxed),
            device.builds.load(Ordering::Relaxed),
        );
        for _ in 0..1024 {
            let guard = binding.load();
            assert!(guard.as_ref().unwrap().matches(source, device.id));
        }
        assert_eq!(
            counts,
            (
                device.allocations.load(Ordering::Relaxed),
                device.frees.load(Ordering::Relaxed),
                device.builds.load(Ordering::Relaxed)
            )
        );
        // An in-place dynamic input cannot mint a cache key, even if an
        // immutable conversion already exists for the very same address.
        assert!(ImmutablePackedWeight::from_input(ptr, 256, false).is_none());
        device.blocks.lock().unwrap().insert(ptr, weight(0x91, 256));
        assert!(ImmutablePackedWeight::from_input(ptr, 256, false).is_none());
        assert!(
            !binding
                .load()
                .as_ref()
                .unwrap()
                .matches(source, device.id + 1)
        );
        cache.invalidate(ptr, 256);
        assert!(!binding.load().as_ref().unwrap().matches(source, device.id));
        binding.store(None);
        device.free(ptr);
    }

    #[test]
    fn failed_build_retires_submitted_candidate_without_publishing_it() {
        let device = RecyclingDevice::new();
        let cache = InterleaveCache::default();
        let ptr = device.put(&weight(0x12, 256));
        let source = ImmutablePackedWeight::from_input(ptr, 256, true).unwrap();
        device.fail_build.store(true, Ordering::Relaxed);
        let failure = cache.prepare(&device, source).unwrap_err().to_string();
        assert!(failure.contains("after submission"), "{failure}");
        assert_eq!(cache.len(), 0);
        assert_eq!(device.pending.lock().unwrap().len(), 1);
        assert_eq!(device.frees.load(Ordering::Relaxed), 0);
        device.complete_readers();
        assert_eq!(device.frees.load(Ordering::Relaxed), 1);
        device.fail_build.store(false, Ordering::Relaxed);
        let (owner, warm) = cache.prepare(&device, source).unwrap();
        assert!(!warm);
        assert_eq!(
            device.contents(owner.ptr()),
            interleaved(&device.contents(ptr))
        );
        cache.release_all();
        drop(owner);
        device.free(ptr);
    }

    #[test]
    fn invalidation_and_release_drop_recursive_native_owners_outside_cache_lock() {
        let device = RecyclingDevice::new();
        let cache = Arc::new(InterleaveCache::default());
        *device.invalidate_on_release.lock().unwrap() = Some(Arc::downgrade(&cache));
        let ptr = device.put(&weight(0x19, 256));
        snapshot(&cache, &device, ptr, 256).unwrap();
        cache.invalidate(ptr, 256);
        assert_eq!(device.frees.load(Ordering::Relaxed), 1);
        snapshot(&cache, &device, ptr, 256).unwrap();
        cache.release_all();
        assert_eq!(device.frees.load(Ordering::Relaxed), 2);
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
