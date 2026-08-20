#![allow(clippy::uninlined_format_args)]
//! The CUDA execution provider's memory mechanism, across the shared
//! `DeviceAllocator` seam.
//!
//! Two claims are pinned here, and they are opposite sides of the same change.
//!
//! The first is that the **built-in** mechanism is the VMM arena and nothing
//! else. A provider constructed the way production constructs it — no
//! environment variable, no builder call — allocates through the arena, and
//! there is no eager `cuMemAlloc` allocator in the tree for it to degrade to.
//!
//! The second is that removing that built-in implementation did not remove the
//! *capability*. `ExternalEagerAllocator` below is deliberately the thing that
//! was deleted, rebuilt outside the runtime from nothing but public API: a
//! caller who wants `cuMemAlloc` writes ~40 lines and injects them, and the
//! provider then uses that and only that. If injection ever stopped being
//! honoured, these tests are where it would show.
//!
//! Needs a real GPU. Skips loudly when there is none: a skip that reads like a
//! pass is worse than a failure, because nobody investigates it.

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::CudaContext;
use onnx_runtime_memory_governor::{DeviceAllocator, DeviceKey, MemoryError, Tier};

/// An eager `cuMemAlloc` device allocator, written entirely outside the
/// runtime.
///
/// This is not a copy of a production type kept alive for tests: the production
/// type is gone. It exists to demonstrate that the ordinary allocator contract
/// is enough to build the mechanism that was removed, using only public API —
/// which is exactly the compatibility boundary Phase 7 promised to preserve.
#[derive(Debug)]
struct ExternalEagerAllocator {
    context: Arc<CudaContext>,
    device: DeviceKey,
    /// Successful `cuMemAlloc` calls. This is the observable that proves the
    /// provider really routed through *this* allocator rather than keeping its
    /// own.
    cumemalloc_calls: AtomicU64,
    frees: AtomicU64,
}

impl ExternalEagerAllocator {
    fn new(context: Arc<CudaContext>) -> Self {
        let ordinal = context.ordinal() as u32;
        Self {
            context,
            device: DeviceKey::device(ordinal),
            cumemalloc_calls: AtomicU64::new(0),
            frees: AtomicU64::new(0),
        }
    }

    fn cumemalloc_calls(&self) -> u64 {
        self.cumemalloc_calls.load(Ordering::Relaxed)
    }

    fn frees(&self) -> u64 {
        self.frees.load(Ordering::Relaxed)
    }
}

impl DeviceAllocator for ExternalEagerAllocator {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        if align == 0 || !align.is_power_of_two() || align > 256 {
            return Err(MemoryError::InvalidRequest {
                tier: Tier::Device.name(),
                requested: bytes as u64,
                reason: "cuMemAlloc guarantees 256-byte alignment and this allocator does not \
                         over-allocate to exceed it",
            });
        }
        self.context
            .bind_to_thread()
            .map_err(|error| MemoryError::AllocationFailed {
                tier: Tier::Device.name(),
                requested: bytes as u64,
                reason: format!("could not bind the CUDA context: {error}"),
            })?;
        // SAFETY: a fresh device allocation on the bound context, owned here and
        // freed exactly once in `deallocate`.
        let dptr =
            unsafe { cudarc::driver::result::malloc_sync(bytes.max(1)) }.map_err(|error| {
                MemoryError::AllocationFailed {
                    tier: Tier::Device.name(),
                    requested: bytes as u64,
                    reason: format!("cuMemAlloc refused: {error}"),
                }
            })?;
        NonNull::new(dptr as *mut u8)
            .ok_or(MemoryError::AllocationFailed {
                tier: Tier::Device.name(),
                requested: bytes as u64,
                reason: String::from("cuMemAlloc returned a null device pointer"),
            })
            .inspect(|_| {
                self.cumemalloc_calls.fetch_add(1, Ordering::Relaxed);
            })
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _bytes: usize, _align: usize) {
        let _ = self.context.bind_to_thread();
        // SAFETY: the pointer came from `allocate` on this allocator and the
        // caller's contract guarantees a single free.
        let _ = unsafe {
            cudarc::driver::result::free_sync(ptr.as_ptr() as cudarc::driver::sys::CUdeviceptr)
        };
        self.frees.fetch_add(1, Ordering::Relaxed);
    }

    fn device(&self) -> DeviceKey {
        self.device
    }
}

/// Records the size it was told per pointer and reports any free that
/// disagrees, the way a size-classed allocator would notice.
#[derive(Debug)]
struct StrictSizes {
    inner: ExternalEagerAllocator,
    live: std::sync::Mutex<std::collections::HashMap<usize, usize>>,
    mismatches: AtomicU64,
    unknown: AtomicU64,
}

impl StrictSizes {
    fn new(inner: ExternalEagerAllocator) -> Self {
        Self {
            inner,
            live: std::sync::Mutex::new(std::collections::HashMap::new()),
            mismatches: AtomicU64::new(0),
            unknown: AtomicU64::new(0),
        }
    }
}

impl DeviceAllocator for StrictSizes {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        let ptr = self.inner.allocate(bytes, align)?;
        self.live
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(ptr.as_ptr() as usize, bytes);
        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        // Poison-tolerant on purpose. This runs on the deferred-release worker
        // thread. If the test thread ever panics while holding this lock, an
        // `unwrap()` here turns that into a *second* panic on an unrelated
        // thread, and the worker's backtrace then reads like an independent
        // release-path defect when it is only a cascade of the first failure.
        // That is precisely how the first hardware run's log was misread.
        match self
            .live
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&(ptr.as_ptr() as usize))
        {
            Some(allocated) if allocated == bytes => {}
            Some(allocated) => {
                self.mismatches.fetch_add(1, Ordering::Relaxed);
                eprintln!("freed {bytes} bytes for a pointer allocated as {allocated}");
            }
            None => {
                self.unknown.fetch_add(1, Ordering::Relaxed);
                eprintln!("freed a pointer this allocator never handed out");
            }
        }
        // SAFETY: forwarded unchanged from this method's own contract.
        unsafe { self.inner.deallocate(ptr, bytes, align) };
    }

    fn device(&self) -> DeviceKey {
        self.inner.device()
    }
}

/// Block until every deferred device release has terminally completed.
///
/// `deallocate` no longer frees anything by the time it returns: it enqueues
/// the release behind completion events on both streams and answers `Ok(0)`,
/// which its own doc comment calls the truthful answer. An assertion that reads
/// an allocator-side counter therefore has to *observe the queue* rather than
/// assume the free already happened -- exactly what the in-crate test at
/// `provider.rs` (search "the deferred release queue must drain") does.
///
/// This waits; it does not sleep. A drain that times out fails loudly, so a
/// release that never runs still turns the calling test red rather than being
/// papered over.
fn drain_releases(provider: &onnx_runtime_ep_cuda::provider::CudaExecutionProvider, what: &str) {
    assert!(
        provider
            .release_queue()
            .wait_until_idle(std::time::Duration::from_secs(30)),
        "the deferred release queue must drain before {what} is asserted: {:?}",
        provider.deferred_release_stats()
    );
}

fn require_provider(what: &str) -> onnx_runtime_ep_cuda::provider::CudaExecutionProvider {
    match onnx_runtime_ep_cuda::provider::CudaExecutionProvider::new(0) {
        Ok(provider) => provider,
        Err(error) => {
            eprintln!("SKIPPED (no CUDA runtime): {what} did NOT run: {error}");
            panic!("CUDA test path did not run; report as a failed GPU test, not a pass");
        }
    }
}

fn require_context(what: &str) -> Arc<CudaContext> {
    match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => {
            eprintln!("SKIPPED (no CUDA driver): {what} did NOT run: {error}");
            panic!("CUDA test path did not run; report as a failed GPU test, not a pass");
        }
    }
}

/// Criterion 1 and 8: the default provider allocates through the built-in VMM
/// arena, with nothing set in the environment.
///
/// `commits_on_demand` is the behavioural discriminator, not a flag the
/// provider sets about itself: it is true exactly when the mechanism maps
/// physical granules as spans are handed out, and an eager allocator — which
/// takes physical memory at the moment it is asked for — reports false. The
/// second half of the test proves that by construction rather than by
/// assertion: the *same* provider, with an eager allocator injected, flips it.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn the_default_provider_allocates_through_the_built_in_vmm_arena() {
    use onnx_runtime_ep_api::ExecutionProvider;

    assert!(
        std::env::var("ONNX_GENAI_CUDA_VMM").is_err(),
        "this test must reach the arena with no opt-in; something set the removed flag"
    );
    let provider = require_provider("the default built-in mechanism check");
    assert!(
        provider.commits_on_demand(),
        "the default CUDA provider must allocate through the on-demand VMM arena"
    );

    // Memory really comes from it and really works: a mechanism that reported
    // the right capability and handed back unusable memory would pass the
    // assertion above and fail this one.
    let bytes = 1 << 20;
    let buffer = provider.allocate(bytes, 256).expect("device memory");
    let pattern: Vec<u8> = (0..bytes).map(|index| (index % 251) as u8).collect();
    let mut read_back = vec![0u8; bytes];
    unsafe {
        use cudarc::driver::sys as cu;
        let address = buffer.as_ptr() as cu::CUdeviceptr;
        assert_eq!(
            cu::cuMemcpyHtoD_v2(address, pattern.as_ptr().cast(), bytes),
            cu::CUresult::CUDA_SUCCESS
        );
        assert_eq!(
            cu::cuMemcpyDtoH_v2(read_back.as_mut_ptr().cast(), address, bytes),
            cu::CUresult::CUDA_SUCCESS
        );
    }
    assert_eq!(read_back, pattern, "arena memory did not round-trip");
    provider.deallocate(buffer).expect("returned to the arena");

    let eager = require_provider("the eager-contrast half of the default-mechanism check")
        .with_memory(Arc::new(ExternalEagerAllocator::new(require_context(
            "the eager-contrast half of the default-mechanism check",
        ))))
        .expect("an eager allocator for this device is a legal injection");
    assert!(
        !eager.commits_on_demand(),
        "premise: `commits_on_demand` must distinguish the arena from an eager allocator, or \
         the assertion above says nothing about which mechanism is live"
    );
}

/// Criterion 4, and the non-goal that says an injected external eager
/// allocator must not be prohibited: a successful `with_memory` is
/// authoritative — the built-in arena stops serving and the injected mechanism
/// serves everything.
///
/// The counter is read from the injected allocator, so "the provider kept its
/// own arena and ignored the call" is not a passing outcome. The round-trip
/// after it means a mechanism that counted and returned nothing usable fails
/// too.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn an_injected_external_eager_allocator_replaces_the_built_in_arena() {
    use onnx_runtime_ep_api::ExecutionProvider;

    let provider = require_provider("the authoritative-injection check");
    let injected = Arc::new(ExternalEagerAllocator::new(require_context(
        "the authoritative-injection check",
    )));
    assert_eq!(injected.cumemalloc_calls(), 0);

    let provider = provider
        .with_memory(Arc::clone(&injected) as Arc<dyn DeviceAllocator>)
        .expect("an allocator for this EP's own device must be accepted");
    assert!(
        !provider.commits_on_demand(),
        "the injected eager mechanism, not the arena, must now be the live one"
    );

    let buffer = provider.allocate(4096, 256).expect("device memory");
    assert_eq!(
        injected.cumemalloc_calls(),
        1,
        "the allocation must have gone through the injected allocator, not the retired arena"
    );
    unsafe {
        use cudarc::driver::sys as cu;
        let value: u32 = 0x736;
        assert_eq!(
            cu::cuMemcpyHtoD_v2(
                buffer.as_ptr() as cu::CUdeviceptr,
                std::ptr::addr_of!(value).cast(),
                4
            ),
            cu::CUresult::CUDA_SUCCESS,
            "memory from the injected allocator must be usable device memory"
        );
    }
    provider.deallocate(buffer).expect("free via the injection");
    // The release is queued behind both stream tails, so the counter below is
    // read *after* the queue has terminally settled. Nothing here weakens the
    // assertion: a release routed anywhere other than the injected allocator
    // leaves `frees()` at zero no matter how long the drain waits.
    drain_releases(&provider, "the injected allocator's free count");
    assert_eq!(
        injected.frees(),
        1,
        "the release must go back to the injected allocator too"
    );
}

/// An allocator that does not serve this EP's device is refused.
///
/// Pointers handed to `with_memory` are given to kernels as this device's
/// addresses. A host allocator's pointer is a perfectly valid host address, so
/// nothing detects the substitution until a kernel dereferences it on the
/// device -- far from the call that caused it.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn an_allocator_for_the_wrong_device_is_refused() {
    let provider = require_provider("the device-mismatch check");
    let error = provider
        .with_memory(Arc::new(onnx_runtime_memory_governor::HostAllocator))
        .expect_err("host memory is not CUDA device memory");
    let message = error.to_string();
    assert!(
        message.contains("CUDA device 0"),
        "the error must name the device that was expected: {message}"
    );
}

/// Injection is refused once the mechanism it would replace has already served
/// memory, and refused *before* the new allocator is used.
///
/// The replacement is authoritative, so the arena it displaces stops existing.
/// A pointer already handed out has to be released through the mechanism that
/// produced it, so swapping underneath one would strand it.
///
/// # Why the buffer is released before the refused injection
///
/// `with_memory` takes `mut self`, so a *refused* injection still consumes the
/// provider and drops it — the arena is torn down on the error path just as it
/// is on the success path. Holding a live `DeviceBuffer` across that call
/// therefore does not demonstrate "the provider is unchanged by the refusal";
/// it tears the arena down underneath an outstanding pointer, and the later
/// drop of that pointer is a teardown assertion or a use-after-free on a real
/// device. With `mut self` there is no way to hold a provider across a failed
/// `with_memory`, so this asserts what is actually true and actually safe: the
/// mechanism that served the pointer is the one that releases it, and the
/// refusal still fires afterwards because the guard's `served` counter is
/// monotonic — `ep_allocations` is never decremented, so "has served memory"
/// stays true after the buffer is returned.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn injection_is_refused_once_the_live_mechanism_has_served_memory() {
    use onnx_runtime_ep_api::ExecutionProvider;

    let provider = require_provider("the late-injection refusal check");
    let buffer = provider.allocate(4096, 256).expect("device memory");

    // The original mechanism owns the pointer and is the one that releases it.
    provider
        .deallocate(buffer)
        .expect("the mechanism that served the pointer can release it");

    let injected = Arc::new(ExternalEagerAllocator::new(require_context(
        "the late-injection refusal check",
    )));
    let error = provider
        .with_memory(Arc::clone(&injected) as Arc<dyn DeviceAllocator>)
        .expect_err("a mechanism that has served memory cannot be replaced");
    assert!(
        error.to_string().contains("cannot do so underneath"),
        "the refusal must explain what is outstanding: {error}"
    );
    assert_eq!(
        injected.cumemalloc_calls(),
        0,
        "a refused allocator must never have been used"
    );
}

/// A zero-byte allocation reaches `deallocate` with the size it was allocated
/// with.
///
/// The contract says `deallocate` is called with the same `bytes` as
/// `allocate`, and that implementations may rely on it. `cuMemAlloc(0)` fails,
/// so someone must round zero up. When the execution provider did that, the
/// allocator saw 1 byte on the way in and 0 on the way out. `cuMemFree` ignores
/// the size, so nothing broke -- but a size-classed arena, or any third-party
/// allocator with a free list, would return the block to the wrong class.
///
/// `StrictSizes` is what such an allocator would be: it remembers the size it
/// was told per pointer and refuses a free that disagrees. The execution
/// provider must therefore pass the size through unchanged and leave the
/// rounding to the allocator.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_zero_byte_allocation_is_freed_with_the_size_it_was_allocated_with() {
    use onnx_runtime_ep_api::ExecutionProvider;

    let provider = require_provider("the zero-byte size-agreement check");
    let strict = Arc::new(StrictSizes::new(ExternalEagerAllocator::new(
        require_context("the zero-byte size-agreement check"),
    )));
    let provider = provider
        .with_memory(Arc::clone(&strict) as Arc<dyn DeviceAllocator>)
        .expect("an allocator for this EP's own device must be accepted");

    let buffer = provider
        .allocate(0, 256)
        .expect("a zero-byte buffer must still be allocatable");
    provider.deallocate(buffer).expect("and freeable");
    // Same reason as the injection test above: the free is stream-ordered, so
    // the bookkeeping below is read only after the queue has settled.
    drain_releases(&provider, "the strict allocator's size bookkeeping");

    assert_eq!(
        strict.mismatches.load(Ordering::Relaxed),
        0,
        "the size passed to allocate and the size passed to deallocate disagree"
    );
    assert_eq!(
        strict.unknown.load(Ordering::Relaxed),
        0,
        "a pointer was freed that this allocator never handed out"
    );
    // Read the length out before asserting on it. Asserting on
    // `lock().unwrap().len()` directly keeps the guard alive inside the
    // `panic!` that `assert_eq!` expands to, which poisons the mutex on
    // failure and makes the release worker panic too.
    let live = strict
        .live
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();
    assert_eq!(live, 0, "the buffer leaked");
}
