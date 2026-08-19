//! End-to-end tests for the nxmem memory ABI.
//!
//! Every test here loads the real `onnx-runtime-memory-testplugin` **cdylib**
//! with `dlopen` and drives it through the real [`MemoryPlugin`] loader and
//! the real [`PluginAllocator`] adapter. Nothing re-implements the state
//! machine under test: the assertions are made against the public entry points
//! the runtime itself calls.
//!
//! The test plugin's mechanisms are all host-memory backed, so the whole suite
//! runs on a machine with no accelerator. That is deliberate: the ABI contract
//! is device-agnostic, and pinning it to CUDA would make it untestable on the
//! development host. The CUDA-specific *users* of this ABI are compile-checked
//! only; they are not exercised here and this suite does not claim to.
//!
//! Required scenarios from the phase's acceptance criteria, and where each
//! lives:
//!
//! | Scenario | Test |
//! |---|---|
//! | version mismatch | `a_plugin_outside_the_hosts_major_range_is_refused` |
//! | short struct | `a_short_allocator_vtable_is_refused_before_any_slot_is_read` |
//! | missing optional capability | `a_mechanism_without_optional_capabilities_reports_none` |
//! | allocation and free | `allocate_and_release_round_trips_through_the_boundary` |
//! | lazy backing | `lazy_backing_commits_decommits_and_reports_mapped_bytes` |
//! | release ordering | `deferred_releases_retire_in_order_and_pin_the_module` |
//! | callback failure | `a_refusing_host_callback_fails_the_allocation_cleanly` |
//! | unload with live objects | `unload_is_refused_while_*` (three tests here, plus the queued-free case in `nxmem_abi_unload_gate.rs`) |
//! | older participant compatibility | `an_older_host_range_still_drives_the_current_plugin`, `a_minor_0_mechanism_works_under_a_minor_1_host` |

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_runtime_memory_abi::{
    NXMEM_ABI_VERSION_MAJOR, NXMEM_ABI_VERSION_MINOR, NXMEM_CAP_ALLOCATOR,
    NXMEM_CAP_DEFERRED_RELEASE, NXMEM_CAP_SHARED_MAPPING, NXMEM_CAP_STRUCTURED_RELEASE,
    NXMEM_CAP_VIRTUAL_BACKING, NxmemVersionRange,
};
use onnx_runtime_memory_api::{
    AllocationCommitRange, AllocationReleaseOutcome, DeviceAllocator, DeviceKey, MemoryError,
};
use onnx_runtime_memory_host::{HostReclaim, MemoryPlugin, PluginAllocator, PluginError};

// ─── locating the plugin ────────────────────────────────────────────────────

#[path = "support/testplugin.rs"]
mod testplugin;
use testplugin::testplugin_path;

/// Serialises the whole suite.
///
/// A plugin module's live-object counters are **process-wide** — that is the
/// contract, because unloading a module is a process-wide act. Two tests
/// holding allocators at the same time would therefore see each other's
/// objects through the unload gate. Every test takes this guard first so it
/// drops last, which makes the counters deterministic without weakening what
/// is being tested.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
}

fn load() -> MemoryPlugin {
    MemoryPlugin::load(testplugin_path()).expect("the test plugin loads under the current host")
}

/// A reclaim hook that records what it was asked and answers as configured.
#[derive(Debug)]
struct ScriptedReclaim {
    calls: AtomicU64,
    last_bytes: AtomicU64,
    refuse: bool,
    grant: u64,
}

impl ScriptedReclaim {
    fn granting(grant: u64) -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicU64::new(0),
            last_bytes: AtomicU64::new(0),
            refuse: false,
            grant,
        })
    }

    fn refusing() -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicU64::new(0),
            last_bytes: AtomicU64::new(0),
            refuse: true,
            grant: 0,
        })
    }
}

impl HostReclaim for ScriptedReclaim {
    fn request_reclaim(&self, _device: DeviceKey, bytes: u64) -> Result<u64, String> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.last_bytes.store(bytes, Ordering::Release);
        if self.refuse {
            Err(String::from("the scripted host has nothing to give back"))
        } else {
            Ok(self.grant)
        }
    }
}

fn open(plugin: &MemoryPlugin, mechanism: &str, required: u64) -> PluginAllocator {
    plugin
        .factory(mechanism)
        .unwrap_or_else(|error| panic!("mechanism `{mechanism}` must exist: {error}"))
        .open(required, None)
        .unwrap_or_else(|error| panic!("mechanism `{mechanism}` must open: {error}"))
}

// ─── loading and negotiation ────────────────────────────────────────────────

#[test]
fn the_plugin_loads_and_publishes_every_mechanism() {
    let _serial = serial();
    let plugin = load();
    let names: Vec<&str> = plugin
        .factories()
        .iter()
        .map(|factory| factory.name())
        .collect();
    assert_eq!(
        names,
        onnx_runtime_memory_testplugin::MECHANISM_NAMES,
        "the loader must surface exactly the mechanisms the plugin published"
    );

    let negotiated = plugin.negotiated();
    assert_eq!(negotiated.major, NXMEM_ABI_VERSION_MAJOR);
    assert_eq!(negotiated.minor, NXMEM_ABI_VERSION_MINOR);
    assert_ne!(
        negotiated.capability_flags & NXMEM_CAP_ALLOCATOR,
        0,
        "a plugin that cannot allocate is useless"
    );
}

/// **Version mismatch.**
///
/// Driven through the real `load_with_host_range` entry point rather than by
/// poking `negotiate` directly, so what is pinned is the loader's refusal, not
/// a private helper's return value.
#[test]
fn a_plugin_outside_the_hosts_major_range_is_refused() {
    let _serial = serial();
    let error = MemoryPlugin::load_with_host_range(
        testplugin_path(),
        NxmemVersionRange {
            major_min: 99,
            major_max: 99,
            minor_min: 0,
            minor_max: 0,
        },
    )
    .expect_err("a host from a different major era must not load this plugin");

    match error {
        PluginError::Negotiation { reason, .. } => {
            assert!(
                reason.contains("99") || reason.to_lowercase().contains("major"),
                "the refusal must name the incompatible major version, got: {reason}"
            );
        }
        other => panic!("expected a negotiation refusal, got {other}"),
    }
}

/// **Older supported participant, host side.**
///
/// A host that only speaks the baseline minor still drives the current plugin:
/// negotiation settles on the baseline and no minor-1 slot is used.
#[test]
fn an_older_host_range_still_drives_the_current_plugin() {
    let _serial = serial();
    let plugin = MemoryPlugin::load_with_host_range(
        testplugin_path(),
        NxmemVersionRange {
            major_min: NXMEM_ABI_VERSION_MAJOR,
            major_max: NXMEM_ABI_VERSION_MAJOR,
            minor_min: 0,
            minor_max: 0,
        },
    )
    .expect("a baseline host must still be able to load a current plugin");

    assert_eq!(plugin.negotiated().minor, 0, "the ceiling is the host's");

    let allocator = open(&plugin, "lazy", NXMEM_CAP_ALLOCATOR);
    assert_eq!(
        allocator.core().abi_minor(),
        0,
        "a mechanism must not offer a slot the negotiated host cannot call"
    );

    // The whole allocate/free cycle still works at the baseline level, using
    // the minor-0 `deallocate` slot rather than structured release.
    let ptr = allocator.allocate(4096, 64).expect("baseline allocation");
    // SAFETY: `ptr` is live, and was allocated with exactly these parameters.
    let outcome = unsafe { allocator.release(ptr, 4096, 64) };
    assert!(
        matches!(outcome, AllocationReleaseOutcome::Complete { .. }),
        "the baseline release path must still report a complete release, got {outcome:?}"
    );
}

/// **Older supported participant, plugin side.**
///
/// `legacy-1-0` declares minor 0 even though the module as a whole speaks
/// minor 1. The host must fall back for that mechanism alone, while a sibling
/// mechanism in the very same module keeps the newer slot.
#[test]
fn a_minor_0_mechanism_works_under_a_minor_1_host() {
    let _serial = serial();
    let plugin = load();
    assert_eq!(plugin.negotiated().minor, NXMEM_ABI_VERSION_MINOR);

    let legacy = open(&plugin, "legacy-1-0", NXMEM_CAP_ALLOCATOR);
    assert_eq!(legacy.core().abi_minor(), 0);
    assert_eq!(
        legacy.core().capability_flags() & NXMEM_CAP_STRUCTURED_RELEASE,
        0,
        "structured release did not exist at minor 0"
    );

    let modern = open(&plugin, "lazy", NXMEM_CAP_ALLOCATOR);
    assert_eq!(
        modern.core().abi_minor(),
        NXMEM_ABI_VERSION_MINOR,
        "a current mechanism in the same module must not be dragged down"
    );

    // Both must work, through the same public release entry point.
    for allocator in [&legacy, &modern] {
        let ptr = allocator.allocate(2048, 32).expect("allocation");
        // SAFETY: live allocation with matching parameters.
        let outcome = unsafe { allocator.release(ptr, 2048, 32) };
        assert!(
            matches!(outcome, AllocationReleaseOutcome::Complete { .. }),
            "both ABI levels must release cleanly, got {outcome:?}"
        );
    }
}

/// **Short struct.**
///
/// The `short-struct` mechanism publishes an allocator vtable whose
/// `struct_size` is smaller than the baseline prefix. The host must refuse it
/// *before* reading any function pointer out of it.
#[test]
fn a_short_allocator_vtable_is_refused_before_any_slot_is_read() {
    let _serial = serial();
    let plugin = load();
    let error = plugin
        .factory("short-struct")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_ALLOCATOR, None)
        .expect_err("an undersized vtable must be refused");

    let text = error.to_string();
    assert!(
        text.contains("struct_size") || text.to_lowercase().contains("short") || text.contains("small"),
        "the refusal must explain that the struct is too short, got: {text}"
    );
}

/// **Missing optional capability.**
///
/// `eager` advertises the allocator capability and nothing else. Asking for an
/// optional capability it does not have is refused, and opening it without
/// asking yields an allocator whose capability views are genuinely absent —
/// not present-but-broken.
#[test]
fn a_mechanism_without_optional_capabilities_reports_none() {
    let _serial = serial();
    let plugin = load();

    let refusal = plugin
        .factory("eager")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_VIRTUAL_BACKING, None)
        .expect_err("a capability the mechanism lacks must be refused, not faked");
    let text = refusal.to_string();
    assert!(
        text.to_lowercase().contains("capab"),
        "the refusal must say which capability is missing, got: {text}"
    );

    let allocator = open(&plugin, "eager", NXMEM_CAP_ALLOCATOR);
    assert!(
        allocator.as_virtual_backing().is_none(),
        "an unsupported capability must be represented as absent"
    );
    assert!(
        allocator.as_shared_mapping().is_none(),
        "an unsupported capability must be represented as absent"
    );
    assert!(
        !allocator.commits_on_demand(),
        "a mechanism with no virtual backing cannot commit on demand"
    );
    assert_eq!(
        allocator.pending_release_count().expect("a query is safe"),
        0,
        "a mechanism with no deferred release has nothing pending"
    );

    // Deferred release is likewise refused rather than silently degraded to an
    // immediate free.
    let ptr = allocator.allocate(1024, 16).expect("allocation");
    // SAFETY: live allocation with matching parameters.
    let error = unsafe { allocator.enqueue_release(ptr, 1024, 16) }
        .expect_err("deferred release is not available on this mechanism");
    assert!(
        matches!(error, MemoryError::AllocationFailed { .. }),
        "got {error:?}"
    );
    // SAFETY: the failed enqueue left the allocation live.
    unsafe { allocator.deallocate(ptr, 1024, 16) };
}

// ─── allocation and release ─────────────────────────────────────────────────

/// **Allocation and free.**
#[test]
fn allocate_and_release_round_trips_through_the_boundary() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(&plugin, "eager", NXMEM_CAP_ALLOCATOR);

    assert_eq!(allocator.core().live_allocation_count(), 0);
    let ptr = allocator.allocate(8192, 256).expect("allocation succeeds");
    assert_eq!(
        ptr.as_ptr() as usize % 256,
        0,
        "the plugin must honour the requested alignment"
    );
    assert_eq!(allocator.core().live_allocation_count(), 1);

    // The memory really is usable across the boundary.
    // SAFETY: the plugin returned 8192 writable bytes at `ptr`.
    unsafe { ptr.as_ptr().write_bytes(0xAB, 8192) };
    // SAFETY: as above.
    assert_eq!(unsafe { ptr.as_ptr().add(8191).read() }, 0xAB);

    // SAFETY: live allocation with matching parameters.
    let outcome = unsafe { allocator.release(ptr, 8192, 256) };
    match outcome {
        AllocationReleaseOutcome::Complete { accounting } => {
            assert_eq!(accounting.allocation_bytes, 8192);
            assert_eq!(accounting.unmapped_bytes, 8192);
        }
        other => panic!("expected a complete release, got {other:?}"),
    }
    assert_eq!(allocator.core().live_allocation_count(), 0);
}

/// An address the host does not know is refused rather than guessed at.
///
/// This is the ABA-safety property: identity is a monotonic id the host
/// assigned, never the address, so a recycled address cannot be mistaken for a
/// live allocation.
#[test]
fn releasing_an_unknown_address_fails_rather_than_guessing() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(&plugin, "eager", NXMEM_CAP_ALLOCATOR);

    let mut stack = 0u64;
    let ptr = NonNull::new((&raw mut stack).cast::<u8>()).expect("non-null");
    // SAFETY: deliberately passing an address this allocator never issued;
    // the contract is that this is reported, not acted on.
    let outcome = unsafe { allocator.release(ptr, 8, 8) };
    match outcome {
        AllocationReleaseOutcome::Failed { failure } => {
            let text = format!("{failure:?}");
            assert!(
                text.contains("live allocation") || text.to_lowercase().contains("recognise"),
                "the failure must say the address is unknown, got: {text}"
            );
        }
        other => panic!("an unknown address must fail, not {other:?}"),
    }
    assert_eq!(stack, 0, "the stack slot must be untouched");
}

/// A size or alignment that disagrees with the live record is refused.
#[test]
fn a_release_that_misdescribes_the_allocation_is_refused() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(&plugin, "eager", NXMEM_CAP_ALLOCATOR);
    let ptr = allocator.allocate(4096, 64).expect("allocation");

    // SAFETY: the address is live; the size is deliberately wrong.
    let outcome = unsafe { allocator.release(ptr, 2048, 64) };
    assert!(
        matches!(outcome, AllocationReleaseOutcome::Failed { .. }),
        "a mismatched size must be refused, got {outcome:?}"
    );
    assert_eq!(
        allocator.core().live_allocation_count(),
        1,
        "a refused release must leave the allocation live, not leak it"
    );

    // SAFETY: the allocation is still live with its original parameters.
    let outcome = unsafe { allocator.release(ptr, 4096, 64) };
    assert!(matches!(outcome, AllocationReleaseOutcome::Complete { .. }));
}

// ─── lazy backing ───────────────────────────────────────────────────────────

/// **Lazy backing.**
#[test]
fn lazy_backing_commits_decommits_and_reports_mapped_bytes() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(
        &plugin,
        "lazy",
        NXMEM_CAP_ALLOCATOR | NXMEM_CAP_VIRTUAL_BACKING,
    );
    assert!(
        allocator.commits_on_demand(),
        "a mechanism with virtual backing commits on demand"
    );
    let backing = allocator
        .as_virtual_backing()
        .expect("virtual backing was required at open time");

    // Reserve without committing.
    let ptr = backing
        .allocate_committed(64 * 1024, 4096, &[])
        .expect("a reservation with nothing committed");
    assert_eq!(
        backing.allocation_committed_bytes(ptr, 64 * 1024, 4096),
        0,
        "nothing was committed yet"
    );

    // Commit a window and watch the committed bytes follow.
    backing
        .commit_allocation_range(ptr, 64 * 1024, 4096, 0, 8192)
        .expect("committing a range");
    assert_eq!(
        backing.allocation_committed_bytes(ptr, 64 * 1024, 4096),
        8192
    );

    let mapped = backing
        .mapped_bytes_for_allocation_ranges(&[AllocationCommitRange {
            ptr,
            allocation_bytes: 64 * 1024,
            align: 4096,
            offset: 0,
            bytes: 100,
        }])
        .expect("a mapped-byte estimate");
    assert_eq!(
        mapped, 4096,
        "a partial granule still maps a whole granule; the estimate must be conservative"
    );
    assert_eq!(
        backing
            .mapped_bytes_for_allocation(64 * 1024, 4096)
            .expect("a whole-allocation estimate"),
        64 * 1024,
        "a whole allocation maps its whole granule-rounded size"
    );

    let unmapped = backing
        .decommit_allocation_range(ptr, 64 * 1024, 4096, 0, 8192)
        .expect("decommitting a range");
    assert_eq!(unmapped, 8192);
    assert_eq!(backing.allocation_committed_bytes(ptr, 64 * 1024, 4096), 0);

    // SAFETY: live allocation with matching parameters.
    let outcome = unsafe { allocator.release(ptr, 64 * 1024, 4096) };
    assert!(matches!(outcome, AllocationReleaseOutcome::Complete { .. }));
}

/// Shared prefixes are reference counted and keep their two accounting axes
/// apart: mapping a prefix again owns no new bytes but does map bytes.
#[test]
fn a_shared_prefix_is_reference_counted_and_costed_once() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(
        &plugin,
        "lazy",
        NXMEM_CAP_ALLOCATOR | NXMEM_CAP_VIRTUAL_BACKING | NXMEM_CAP_SHARED_MAPPING,
    );
    let shared = allocator
        .as_shared_mapping()
        .expect("shared mapping was required at open time");

    let prefix = shared
        .create_shared_prefix(16 * 1024)
        .expect("creating a shared prefix");
    assert_eq!(prefix.requested_bytes(), 16 * 1024);
    assert_ne!(prefix.device_ptr(), 0);

    assert_eq!(
        shared
            .incremental_owned_bytes_for_shared_prefix(prefix.as_ref())
            .expect("the prefix belongs to this mechanism"),
        0,
        "the prefix's physical bytes were charged once at creation"
    );

    let ptr = allocator.allocate(64 * 1024, 4096).expect("allocation");
    let info = shared
        .commit_shared_prefix(prefix.as_ref(), ptr, 64 * 1024, 0)
        .expect("committing the prefix into the allocation");
    assert_eq!(
        info.additional_owned_bytes, 0,
        "re-mapping owns nothing new"
    );
    assert_eq!(
        info.newly_mapped_bytes,
        16 * 1024,
        "but it does map bytes; the axes are distinct"
    );

    // SAFETY: live allocation with matching parameters.
    let outcome = unsafe { allocator.release(ptr, 64 * 1024, 4096) };
    assert!(matches!(outcome, AllocationReleaseOutcome::Complete { .. }));
}

// ─── deferred release ordering ──────────────────────────────────────────────

/// **Release ordering**, plus module pinning across a deferred free.
#[test]
fn deferred_releases_retire_in_order_and_pin_the_module() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(
        &plugin,
        "lazy",
        NXMEM_CAP_ALLOCATOR | NXMEM_CAP_DEFERRED_RELEASE,
    );

    let mut tickets = Vec::new();
    for index in 0..4u64 {
        let bytes = 1024 * (index as usize + 1);
        let ptr = allocator.allocate(bytes, 64).expect("allocation");
        // SAFETY: live allocation with matching parameters.
        let ticket = unsafe { allocator.enqueue_release(ptr, bytes, 64) }.expect("queued release");
        tickets.push(ticket);
    }

    assert_eq!(
        allocator.pending_release_count().expect("query"),
        4,
        "all four releases are still queued"
    );
    assert_eq!(
        plugin.module().host_live_counts().queued_releases,
        4,
        "the host counts them too, so unload stays gated"
    );
    assert!(
        allocator.core().retired_releases().is_empty(),
        "no completion may arrive before the drain"
    );

    // Retire two, then the rest, so partial drains are covered as well.
    assert_eq!(allocator.drain_releases(2).expect("drain"), 2);
    assert_eq!(allocator.pending_release_count().expect("query"), 2);
    assert_eq!(allocator.drain_releases(64).expect("drain"), 2);
    assert_eq!(allocator.pending_release_count().expect("query"), 0);

    let retired = allocator.core().retired_releases();
    assert_eq!(retired.len(), 4, "every queued release must report back");
    let retired_tickets: Vec<u64> = retired.iter().map(|entry| entry.ticket).collect();
    assert_eq!(
        retired_tickets, tickets,
        "completions must arrive in the order the releases were queued"
    );
    for (index, entry) in retired.iter().enumerate() {
        assert_eq!(entry.allocation_bytes, 1024 * (index as u64 + 1));
    }
    assert_eq!(
        plugin.module().host_live_counts().queued_releases,
        0,
        "the host's tally must return to zero once the queue drains"
    );
}

// ─── callback failure ───────────────────────────────────────────────────────

/// **Callback failure.**
///
/// The `callback-probe` mechanism calls the host's reclaim hook on every
/// allocation. A refusing host must produce a clean allocation failure — no
/// abort, no leak, no half-registered allocation.
#[test]
fn a_refusing_host_callback_fails_the_allocation_cleanly() {
    let _serial = serial();
    let plugin = load();
    let reclaim = ScriptedReclaim::refusing();
    let allocator = plugin
        .factory("callback-probe")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_ALLOCATOR, Some(reclaim.clone()))
        .expect("the mechanism opens");

    let error = allocator
        .allocate(4096, 64)
        .expect_err("a refused reclaim must fail the allocation");
    assert!(
        matches!(error, MemoryError::AllocationFailed { .. }),
        "got {error:?}"
    );
    let text = format!("{error:?}");
    assert!(
        text.to_lowercase().contains("callback") || text.contains("reclaim"),
        "the error must name the failing callback, got: {text}"
    );

    assert_eq!(
        reclaim.calls.load(Ordering::Acquire),
        1,
        "the plugin really did reach the host"
    );
    assert_eq!(reclaim.last_bytes.load(Ordering::Acquire), 4096);
    assert_eq!(
        allocator.core().reclaim_calls(),
        1,
        "the host bridge counts the call"
    );
    assert_eq!(
        allocator.core().reclaim_failures(),
        1,
        "and records that it refused"
    );
    assert_eq!(
        allocator.core().live_allocation_count(),
        0,
        "a failed allocation must not be registered"
    );

    // The very same mechanism succeeds once the host cooperates, which proves
    // the failure came from the callback and not from something else.
    let allocator = plugin
        .factory("callback-probe")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_ALLOCATOR, Some(ScriptedReclaim::granting(1 << 20)))
        .expect("the mechanism opens");
    let ptr = allocator
        .allocate(4096, 64)
        .expect("a cooperating host lets the allocation through");
    assert_eq!(allocator.core().reclaim_calls(), 1);
    assert_eq!(allocator.core().reclaim_failures(), 0);
    // SAFETY: live allocation with matching parameters.
    unsafe { allocator.deallocate(ptr, 4096, 64) };
}

/// A host that offers no reclaim hook at all says so explicitly.
///
/// The alternative — answering "reclaimed 0 bytes" — would be a lie a plugin
/// cannot distinguish from a real, unhelpful reclaim. An unsupported
/// capability is represented as unsupported, and the plugin decides what to do
/// about it; `callback-probe` chooses to fail the allocation.
#[test]
fn a_host_with_no_reclaim_hook_reports_the_capability_as_absent() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(&plugin, "callback-probe", NXMEM_CAP_ALLOCATOR);
    let error = allocator
        .allocate(4096, 64)
        .expect_err("this mechanism cannot proceed without a reclaim path");
    let text = format!("{error:?}");
    assert!(
        text.contains("no reclaim path"),
        "the host must say plainly that it offers no reclaim path, got: {text}"
    );
    assert_eq!(
        allocator.core().reclaim_calls(),
        1,
        "the plugin really did try"
    );
    assert_eq!(
        allocator.core().reclaim_failures(),
        1,
        "and the bridge recorded that the host could not help"
    );
    assert_eq!(
        allocator.core().live_allocation_count(),
        0,
        "a failed allocation must not be registered"
    );
}

// ─── unload gating ──────────────────────────────────────────────────────────

/// A plugin with nothing live unloads.
#[test]
fn an_idle_plugin_unloads() {
    let _serial = serial();
    let plugin = load();
    assert_eq!(plugin.module().host_live_counts().total(), 0);
    plugin
        .try_unload()
        .expect("a plugin with nothing live must unload");
}

/// **Unload with live objects**, allocator case.
#[test]
fn unload_is_refused_while_an_allocator_is_open() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(&plugin, "eager", NXMEM_CAP_ALLOCATOR);

    let rejection = plugin
        .try_unload()
        .expect_err("an open allocator must block unload");
    assert!(
        rejection.report.live_allocators >= 1,
        "the plugin itself must report the live allocator: {:?}",
        rejection.report
    );
    let plugin = rejection
        .into_plugin()
        .expect("the refusal hands the plugin back so the caller can retire work");

    drop(allocator);
    plugin
        .try_unload()
        .expect("once the allocator is gone the plugin unloads");
}

/// **Unload with live objects**, allocation case.
#[test]
fn unload_is_refused_while_an_allocation_is_live() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(&plugin, "eager", NXMEM_CAP_ALLOCATOR);
    let ptr = allocator.allocate(4096, 64).expect("allocation");

    let rejection = plugin
        .try_unload()
        .expect_err("a live allocation must block unload");
    assert!(
        rejection.report.live_allocations >= 1,
        "the plugin must report the live allocation: {:?}",
        rejection.report
    );
    let plugin = rejection.into_plugin().expect("the plugin comes back");

    // SAFETY: live allocation with matching parameters.
    unsafe { allocator.deallocate(ptr, 4096, 64) };
    drop(allocator);
    plugin.try_unload().expect("now it unloads");
}

/// **Unload with live objects**, capability-view case.
#[test]
fn unload_is_refused_while_a_shared_prefix_is_held() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(
        &plugin,
        "lazy",
        NXMEM_CAP_ALLOCATOR | NXMEM_CAP_SHARED_MAPPING,
    );
    let prefix = allocator
        .as_shared_mapping()
        .expect("shared mapping was required")
        .create_shared_prefix(8192)
        .expect("creating a prefix");

    let rejection = plugin
        .try_unload()
        .expect_err("a held capability object must block unload");
    assert!(
        rejection.report.live_capabilities >= 1,
        "the plugin must report the held prefix: {:?}",
        rejection.report
    );
    let plugin = rejection.into_plugin().expect("the plugin comes back");

    drop(prefix);
    drop(allocator);
    plugin.try_unload().expect("now it unloads");
}

// ─── cross-provider misuse ──────────────────────────────────────────────────

/// Two allocators from the same module have distinct mechanism identities, and
/// an object from one is refused by the other rather than silently accepted.
#[test]
fn an_object_from_another_mechanism_is_refused() {
    let _serial = serial();
    let plugin = load();
    let first = open(
        &plugin,
        "lazy",
        NXMEM_CAP_ALLOCATOR | NXMEM_CAP_SHARED_MAPPING,
    );
    let second = open(
        &plugin,
        "lazy",
        NXMEM_CAP_ALLOCATOR | NXMEM_CAP_SHARED_MAPPING,
    );
    assert_ne!(
        first.core().mechanism_id(),
        second.core().mechanism_id(),
        "each opened mechanism instance needs its own identity"
    );

    let prefix = first
        .as_shared_mapping()
        .expect("shared mapping")
        .create_shared_prefix(4096)
        .expect("creating a prefix on the first mechanism");

    // The second mechanism must not cost a foreign prefix as free.
    let error = second
        .as_shared_mapping()
        .expect("shared mapping")
        .incremental_owned_bytes_for_shared_prefix(prefix.as_ref())
        .expect_err("a foreign prefix must be rejected, never costed as free");
    let text = format!("{error:?}");
    assert!(
        text.to_lowercase().contains("mechanism") || text.to_lowercase().contains("prefix"),
        "the refusal must name the mismatch, got: {text}"
    );

    let ptr = second.allocate(8192, 4096).expect("allocation");
    let error = second
        .as_shared_mapping()
        .expect("shared mapping")
        .commit_shared_prefix(prefix.as_ref(), ptr, 8192, 0)
        .expect_err("committing a foreign prefix must be refused");
    let text = format!("{error:?}");
    assert!(
        text.to_lowercase().contains("mechanism") || text.to_lowercase().contains("prefix"),
        "the refusal must name the mismatch, got: {text}"
    );

    // SAFETY: live allocation with matching parameters.
    let outcome = unsafe { second.release(ptr, 8192, 4096) };
    assert!(matches!(outcome, AllocationReleaseOutcome::Complete { .. }));
}

/// An allocation made by one mechanism is not releasable by another.
#[test]
fn an_allocation_cannot_be_released_by_a_sibling_mechanism() {
    let _serial = serial();
    let plugin = load();
    let first = open(&plugin, "eager", NXMEM_CAP_ALLOCATOR);
    let second = open(&plugin, "eager", NXMEM_CAP_ALLOCATOR);

    let ptr = first.allocate(4096, 64).expect("allocation");
    // SAFETY: deliberately naming a sibling's allocation; the contract is that
    // this is refused, not acted on.
    let outcome = unsafe { second.release(ptr, 4096, 64) };
    assert!(
        matches!(outcome, AllocationReleaseOutcome::Failed { .. }),
        "a sibling must not be able to free another mechanism's memory, got {outcome:?}"
    );
    assert_eq!(first.core().live_allocation_count(), 1);

    // SAFETY: still live on its owning mechanism.
    let outcome = unsafe { first.release(ptr, 4096, 64) };
    assert!(matches!(outcome, AllocationReleaseOutcome::Complete { .. }));
}

// ─── factory lifetime ───────────────────────────────────────────────────────

/// Every factory the host took is released exactly once when the plugin goes.
///
/// The count is read out of the **loaded module** through a test-only exported
/// symbol. Reading the statically linked `rlib` copy of the same static would
/// observe a different variable entirely and would pass while proving nothing.
///
/// Two handles are opened onto the same module: unloading the second must
/// release its factories while the first keeps the module mapped, which is
/// also what makes the count readable afterwards.
#[test]
fn every_factory_is_released_exactly_once() {
    let _serial = serial();
    let observer = load();
    let releases = |plugin: &MemoryPlugin| -> u64 {
        // SAFETY: the symbol is a `extern "C" fn() -> u64` exported by the
        // test plugin, and the module stays mapped for the borrow.
        let symbol: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = unsafe {
            plugin
                .module()
                .library()
                .get(onnx_runtime_memory_testplugin::SYMBOL_FACTORY_RELEASES)
        }
        .expect("the test plugin exports its introspection symbol");
        // SAFETY: as above.
        unsafe { symbol() }
    };

    let before = releases(&observer);
    let plugin = load();
    let count = plugin.factories().len() as u64;
    assert_eq!(count, 6, "the test plugin publishes six mechanisms");
    plugin.try_unload().expect("an idle plugin unloads");

    assert_eq!(
        releases(&observer) - before,
        count,
        "the host must release each factory exactly once"
    );

    // The observing handle still works, which is what "the module stays mapped
    // while anything pins it" means in practice.
    let allocator = open(&observer, "eager", NXMEM_CAP_ALLOCATOR);
    let ptr = allocator.allocate(1024, 16).expect("allocation");
    // SAFETY: live allocation with matching parameters.
    unsafe { allocator.deallocate(ptr, 1024, 16) };
}

/// An unknown mechanism name is a clear, listing error rather than a panic.
#[test]
fn an_unknown_mechanism_name_is_reported_with_the_available_set() {
    let _serial = serial();
    let plugin = load();
    let error = plugin
        .factory("no-such-mechanism")
        .expect_err("the mechanism does not exist");
    match error {
        PluginError::UnknownMechanism { available, .. } => {
            assert!(
                available.contains("eager") && available.contains("lazy"),
                "the error must list what is available, got: {available}"
            );
        }
        other => panic!("expected an unknown-mechanism error, got {other}"),
    }
}
