//! Per-phase allocation attribution, exercised with the counting allocator
//! actually installed.
//!
//! This lives in its own test binary because `#[global_allocator]` is a
//! per-binary choice: installing it in the library's unit tests would put a
//! counting wrapper under every other test in the crate. Here it is the point.
//!
//! What this pins is the *attribution*, not a particular number of
//! allocations — the numbers belong with the code that allocates. The property
//! is that an allocation is charged to the innermost phase that was open when
//! it happened, and to nothing at all when no phase is open.

use onnx_runtime_ep_plugin::dispatch_probe::{self, CountingAllocator, Phase};

#[global_allocator]
static ALLOC: CountingAllocator<std::alloc::System> = CountingAllocator::new(std::alloc::System);

/// Allocate `bytes` in a way the optimiser cannot elide.
fn allocate(bytes: usize) -> Vec<u8> {
    let mut v: Vec<u8> = Vec::with_capacity(bytes);
    v.push(1);
    std::hint::black_box(&v);
    v
}

#[test]
fn an_allocation_is_charged_to_the_phase_that_is_open() {
    let before = dispatch_probe::snapshot();
    let guard = Phase::Allocate.enter();
    let held = allocate(4096);
    guard.end();
    let d = dispatch_probe::snapshot().since(&before);
    drop(held);

    if !dispatch_probe::compiled_in() {
        assert_eq!(
            d.phase_allocs,
            [0; Phase::COUNT],
            "a production build must attribute nothing"
        );
        return;
    }

    assert!(
        d.phase_allocs[Phase::Allocate as usize] >= 1,
        "the allocation inside the phase was not charged to it: {:?}",
        d.phase_allocs
    );
    assert!(
        d.phase_alloc_bytes[Phase::Allocate as usize] >= 4096,
        "bytes were not recorded: {:?}",
        d.phase_alloc_bytes
    );
    for (i, n) in d.phase_allocs.iter().enumerate() {
        if i != Phase::Allocate as usize {
            assert_eq!(*n, 0, "phase {i} was charged for another phase's work");
        }
    }
}

#[test]
fn an_allocation_outside_every_phase_is_charged_to_none() {
    let before = dispatch_probe::snapshot();
    let held = allocate(8192);
    let d = dispatch_probe::snapshot().since(&before);
    drop(held);

    assert_eq!(
        d.phase_allocs,
        [0; Phase::COUNT],
        "work outside dispatch must not be attributed to a dispatch phase"
    );
    assert_eq!(d.phase_alloc_bytes, [0; Phase::COUNT]);
}

/// Phases nest — a guard closes at scope exit, not at an early `return`, so
/// `StatusCrossing` opens inside whatever phase was live. The inner phase must
/// take its own allocations *and hand the outer one back*, or everything after
/// the first nested phase is charged to the wrong place (or to nothing).
#[test]
fn a_nested_phase_takes_its_own_allocations_and_restores_the_outer_one() {
    if !dispatch_probe::compiled_in() {
        return;
    }
    let before = dispatch_probe::snapshot();

    let outer = Phase::TensorBind.enter();
    let a = allocate(1024);
    {
        let inner = Phase::StatusCrossing.enter();
        let b = allocate(2048);
        inner.end();
        drop(b);
    }
    let c = allocate(512);
    outer.end();
    drop((a, c));

    let d = dispatch_probe::snapshot().since(&before);
    assert_eq!(
        d.phase_allocs[Phase::StatusCrossing as usize],
        1,
        "the nested phase should own exactly the allocation made inside it"
    );
    assert_eq!(
        d.phase_allocs[Phase::TensorBind as usize],
        2,
        "the outer phase must resume after the nested one closes: {:?}",
        d.phase_allocs
    );
    assert!(d.phase_alloc_bytes[Phase::StatusCrossing as usize] >= 2048);
}

/// The attribution must survive a phase that ends by unwinding out of scope
/// rather than by an explicit `end()`, since that is how every early-return
/// path in `compute_execute` closes its phase.
#[test]
fn a_phase_closed_by_scope_exit_still_restores_the_outer_phase() {
    if !dispatch_probe::compiled_in() {
        return;
    }
    let before = dispatch_probe::snapshot();
    let outer = Phase::DispatchLookup.enter();
    {
        let _inner = Phase::Allocate.enter();
        let b = allocate(64);
        drop(b);
    }
    let c = allocate(128);
    outer.end();
    drop(c);

    let d = dispatch_probe::snapshot().since(&before);
    assert_eq!(d.phase_allocs[Phase::Allocate as usize], 1);
    assert_eq!(
        d.phase_allocs[Phase::DispatchLookup as usize],
        1,
        "scope-exit close did not restore the outer phase: {:?}",
        d.phase_allocs
    );
}

/// Attribution is thread-local: two threads dispatching at once must not be
/// able to charge each other's allocations, which is what makes a per-phase
/// figure meaningful under concurrent `Run`.
#[test]
fn concurrent_phases_do_not_contaminate_each_other() {
    if !dispatch_probe::compiled_in() {
        return;
    }
    let other = std::thread::spawn(|| {
        let before = dispatch_probe::snapshot();
        let g = Phase::KernelInvoke.enter();
        let mut held = Vec::new();
        for _ in 0..50 {
            held.push(allocate(256));
        }
        g.end();
        let d = dispatch_probe::snapshot().since(&before);
        drop(held);
        d
    });

    let before = dispatch_probe::snapshot();
    let g = Phase::MetadataQuery.enter();
    let held = allocate(256);
    g.end();
    let mine = dispatch_probe::snapshot().since(&before);
    drop(held);

    let theirs = other.join().expect("worker panicked");

    assert_eq!(
        mine.phase_allocs[Phase::KernelInvoke as usize],
        0,
        "this thread was charged for the other thread's phase"
    );
    assert_eq!(
        theirs.phase_allocs[Phase::MetadataQuery as usize],
        0,
        "the other thread was charged for this one's phase"
    );
    assert_eq!(mine.phase_allocs[Phase::MetadataQuery as usize], 1);
    assert!(theirs.phase_allocs[Phase::KernelInvoke as usize] >= 50);
}
