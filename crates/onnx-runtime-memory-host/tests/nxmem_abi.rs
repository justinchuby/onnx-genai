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
    NXMEM_CAP_VIRTUAL_BACKING, NxmemAllocatorVtable, NxmemStatusCode, NxmemVersionRange,
};
use onnx_runtime_memory_api::{
    AllocationCommitRange, AllocationReleaseOutcome, DeviceAllocator, DeviceKey, MemoryError,
};
use onnx_runtime_memory_host::{
    AllocatorCore as PluginAllocatorCore, HostReclaim, MemoryPlugin, PluginAllocator, PluginError,
    PluginModule,
};

// ─── locating the plugin ────────────────────────────────────────────────────

#[path = "support/testplugin.rs"]
mod testplugin;
use testplugin::{
    drain_calls, parked_state_is_set, published_structured_slot, structured_releases,
    terminal_releases, testplugin_path,
};

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

/// A reclaim hook that reads the host's *own* release accounting at the moment
/// the plugin calls it.
///
/// Everything else in this suite can only look at the host before or after a
/// plugin call. Some invariants are only about the inside of one: the host
/// increments its queued-release counters before it enters the plugin, and
/// nothing outside the call can tell whether it did, because by the time the
/// call returns the increment and the matching decrement have both landed and
/// an unsigned wrap is indistinguishable from never having happened.
///
/// The weak references are set after `open`, and are weak so this hook can
/// never keep alive the objects it is watching. Nothing here takes a host
/// lock: this runs on the plugin's side of a call the host is already inside,
/// and a host lock held across the boundary would deadlock rather than fail.
#[derive(Debug, Default)]
struct MidCallObserver {
    core: std::sync::OnceLock<std::sync::Weak<PluginAllocatorCore>>,
    module: std::sync::OnceLock<std::sync::Weak<PluginModule>>,
    outstanding: AtomicU64,
    module_queued: AtomicU64,
    retired_at_observation: AtomicU64,
    observations: AtomicU64,
}

impl HostReclaim for MidCallObserver {
    fn request_reclaim(&self, _device: DeviceKey, _bytes: u64) -> Result<u64, String> {
        if let Some(core) = self.core.get().and_then(std::sync::Weak::upgrade) {
            self.outstanding
                .store(core.outstanding_releases(), Ordering::Release);
            self.retired_at_observation
                .store(core.retired_releases().len() as u64, Ordering::Release);
        }
        if let Some(module) = self.module.get().and_then(std::sync::Weak::upgrade) {
            self.module_queued
                .store(module.host_live_counts().queued_releases, Ordering::Release);
        }
        self.observations.fetch_add(1, Ordering::AcqRel);
        Ok(0)
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
///
/// The mechanism is `ahead-of-host`, not `lazy`, and that is the whole point.
/// `lazy` clamps *itself*: at negotiated minor 0 it declares minor 0 and leaves
/// `release_allocation` null, so `structured_releases` cannot move however the
/// host behaves, and an assertion that it did not move is vacuous — both of the
/// host's defences against reaching past the negotiated level could be deleted
/// together without it failing. `ahead-of-host` publishes a populated minor-1
/// `release_allocation` and declares minor 1 whatever the host negotiated,
/// which is legal for a newer sender and puts the decision entirely on the
/// host's side of the boundary.
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

    let allocator = open(&plugin, "ahead-of-host", NXMEM_CAP_ALLOCATOR);
    assert_eq!(
        allocator.core().abi_minor(),
        0,
        "a sender ahead of the ceiling must be read down to it, not followed up to its own \
         level"
    );
    // The premise the assertion below rests on, checked rather than assumed:
    // the sender really did put a callable structured slot on the wire at this
    // negotiated level. Read out of the published struct itself, because the
    // host's own view has already been clamped and so cannot tell a slot it
    // declined from a slot that was never offered. If the mechanism ever
    // reverts to clamping itself, this is what notices.
    assert_eq!(
        published_structured_slot(&plugin),
        1,
        "the sender must have published a populated structured slot at a negotiated minor 0, or \
         the host having stayed out of it is a fact about the sender and not about the host"
    );

    // The whole allocate/free cycle still works at the baseline level, using
    // the minor-0 `deallocate` slot rather than structured release.
    let ptr = allocator.allocate(4096, 64).expect("baseline allocation");
    let terminal_before = terminal_releases(&plugin);
    let structured_before = structured_releases(&plugin);
    // SAFETY: `ptr` is live, and was allocated with exactly these parameters.
    let outcome = unsafe { allocator.release(ptr, 4096, 64) };
    assert!(
        matches!(outcome, AllocationReleaseOutcome::Complete { .. }),
        "the baseline release path must still report a complete release, got {outcome:?}"
    );
    // And it really was the baseline slot. The sender published the minor-1
    // structured slot, claimed the capability, and declared minor 1; nothing on
    // its side declines to serve the call. The only thing that can keep the
    // host out of that slot is the host itself.
    assert_eq!(
        structured_releases(&plugin),
        structured_before,
        "a baseline host must not enter a slot it did not negotiate, whatever the sender \
         happens to publish"
    );
    assert_eq!(
        terminal_releases(&plugin) - terminal_before,
        1,
        "it must still have freed the allocation, through the baseline slot"
    );
    drop(allocator);
    plugin
        .try_unload()
        .expect("the baseline host leaves nothing outstanding");

    // The other half of the same fact, which is what stops the assertion above
    // passing because the slot was never callable in the first place: the very
    // same mechanism, publishing the very same vtable, *is* entered through the
    // structured slot once the negotiated level permits it. The mechanism did
    // not change between the two halves; only the host's ceiling did.
    let current = load();
    assert_eq!(current.negotiated().minor, NXMEM_ABI_VERSION_MINOR);
    let allocator = open(&current, "ahead-of-host", NXMEM_CAP_ALLOCATOR);
    assert_eq!(
        allocator.core().abi_minor(),
        NXMEM_ABI_VERSION_MINOR,
        "at the current ceiling the sender's own level is what applies"
    );
    let ptr = allocator
        .allocate(4096, 64)
        .expect("current-level allocation");
    let structured_before = structured_releases(&current);
    // SAFETY: `ptr` is live, and was allocated with exactly these parameters.
    let outcome = unsafe { allocator.release(ptr, 4096, 64) };
    assert!(
        matches!(outcome, AllocationReleaseOutcome::Complete { .. }),
        "the structured release path must report a complete release, got {outcome:?}"
    );
    assert_eq!(
        structured_releases(&current) - structured_before,
        1,
        "the slot the baseline host stayed out of must be a slot that really works, or \
         staying out of it proves nothing"
    );
    drop(allocator);
    current.try_unload().expect("nothing is outstanding");
}

/// **Capability is a separate axis from level, and the host must honour both.**
///
/// `undeclared-slot` publishes a populated, working minor-1
/// `release_allocation` and declares minor 1, so the host's *level* clamp has
/// nothing to object to — the negotiated ceiling is minor 1 here and the
/// sender is at minor 1. What it never does is claim
/// `NXMEM_CAP_STRUCTURED_RELEASE`. A slot present in a struct of the right
/// size is not a promise to implement it; the declaration is. The host must
/// therefore go on the capability flags and stay on the baseline `deallocate`
/// slot.
///
/// Every other mechanism that publishes the structured slot also claims the
/// capability, which makes the host's capability check unobservable: deleting
/// it changes nothing, because level agreement and capability agreement always
/// coincide. This mechanism is the one place they come apart.
#[test]
fn a_published_slot_the_sender_never_claimed_is_not_entered() {
    let _serial = serial();
    let plugin = load();
    assert_eq!(plugin.negotiated().minor, NXMEM_ABI_VERSION_MINOR);

    let allocator = open(&plugin, "undeclared-slot", NXMEM_CAP_ALLOCATOR);
    assert_eq!(
        allocator.core().abi_minor(),
        NXMEM_ABI_VERSION_MINOR,
        "the level is agreed; the level is not what is in question here"
    );
    assert_eq!(
        allocator.core().capability_flags() & NXMEM_CAP_STRUCTURED_RELEASE,
        0,
        "the sender must not have claimed the capability, or this test proves nothing"
    );
    // And it must nonetheless have published the slot, or "the host stayed out
    // of it" is again a fact about the sender rather than about the host.
    assert_eq!(
        published_structured_slot(&plugin),
        1,
        "the sender must have published a populated structured slot despite not claiming the \
         capability, or there is nothing here for the host's capability check to refuse"
    );

    let ptr = allocator.allocate(4096, 64).expect("allocation");
    let terminal_before = terminal_releases(&plugin);
    let structured_before = structured_releases(&plugin);
    // SAFETY: `ptr` is live, and was allocated with exactly these parameters.
    let outcome = unsafe { allocator.release(ptr, 4096, 64) };
    assert!(
        matches!(outcome, AllocationReleaseOutcome::Complete { .. }),
        "the baseline path must still complete the release, got {outcome:?}"
    );
    assert_eq!(
        structured_releases(&plugin),
        structured_before,
        "a slot the sender never claimed must not be entered, however inviting the pointer in \
         the struct looks"
    );
    assert_eq!(
        terminal_releases(&plugin) - terminal_before,
        1,
        "and the release must still have happened, through the slot the sender did claim"
    );

    drop(allocator);
    // The accessor above names a struct owned by the allocator state. Once
    // that state is gone the pointer must be gone with it, or the next caller
    // reads freed memory — and a stale pointer would also let the accessor
    // keep answering `1` for a sender that no longer exists, which is the one
    // answer that would make the assertions above pass for the wrong reason.
    assert_eq!(
        published_structured_slot(&plugin),
        u64::MAX,
        "no published vtable may remain reachable once the state that owns it has been \
         destroyed"
    );
    plugin.try_unload().expect("nothing is outstanding");
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
    // The fallback is the sender's own doing here: it published no structured
    // slot at all. Stated against the published struct, which is also the one
    // place in the suite where a live vtable must be reported as *not*
    // carrying the slot — without it, an accessor that always answered "yes"
    // would satisfy every other use.
    assert_eq!(
        published_structured_slot(&plugin),
        0,
        "a mechanism at minor 0 must publish no structured slot"
    );

    let modern = open(&plugin, "lazy", NXMEM_CAP_ALLOCATOR);
    assert_eq!(
        modern.core().abi_minor(),
        NXMEM_ABI_VERSION_MINOR,
        "a current mechanism in the same module must not be dragged down"
    );
    assert_eq!(
        published_structured_slot(&plugin),
        1,
        "and its current sibling, published from the same module moments later, must carry one"
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
/// `struct_size` is smaller than the baseline prefix, in an allocation that is
/// genuinely that short. The host must refuse it on the size check, *before*
/// reading any function pointer out of it.
///
/// The assertion is on the status code and on the two sizes, not on the
/// wording. Matching the word "short" in the message cannot distinguish this
/// from a refusal that happened for a completely different reason: the status
/// code's own name contains it, so a host that skipped the size check entirely
/// and tripped over a null `allocate` slot would produce a message containing
/// "short" too, and the test would stay green while testing nothing.
#[test]
fn a_short_allocator_vtable_is_refused_before_any_slot_is_read() {
    let _serial = serial();
    let plugin = load();
    let error = plugin
        .factory("short-struct")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_ALLOCATOR, None)
        .expect_err("an undersized vtable must be refused");

    assert_eq!(
        error.status_code(),
        Some(NxmemStatusCode::ShortStruct),
        "the refusal must be the size check, not something downstream of it: {error}"
    );

    // The message still has to name the real numbers, because a human reading
    // a log needs to know *which* struct disagreed and by how much. The
    // declared size is the fixture's own declaration; the required size is the
    // baseline prefix this host was built against.
    let declared = NxmemAllocatorVtable::MIN_STRUCT_SIZE_MINOR_0 / 2;
    let required = NxmemAllocatorVtable::MIN_STRUCT_SIZE_MINOR_0;
    let text = error.to_string();
    assert!(
        text.contains(&declared.to_string()) && text.contains(&required.to_string()),
        "the refusal must report the declared size {declared} and the required size {required}, \
         got: {text}"
    );
}

/// **The declared prefix bounds the read.**
///
/// `poisoned-tail` publishes a minor-0 allocator vtable out of a full-size
/// heap allocation whose bytes past the declaration are `0xAB` — including a
/// populated `release_allocation` slot the declaration excludes.
///
/// A host that honours `struct_size` reads the baseline prefix and zeroes the
/// rest, so the structured-release slot comes back absent. A host that reads
/// the whole struct *and* skips the level clamp comes back holding
/// `0xABAB_ABAB_ABAB_ABAB` in a function-pointer slot — and would call it.
///
/// The assertion is on the raw post-negotiation slot rather than on release
/// behaviour, because the structured-release path short-circuits on
/// `abi_minor < 1` before it ever looks at the slot: asserting through
/// behaviour would pass no matter what the slot contained.
#[test]
fn a_poisoned_tail_never_reaches_the_hosts_view_of_the_vtable() {
    let _serial = serial();
    let plugin = load();
    let allocator = plugin
        .factory("poisoned-tail")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_ALLOCATOR, None)
        .expect("a minor-0 vtable that declares its own size honestly is acceptable");

    assert!(
        !allocator.core().publishes_structured_release_slot(),
        "a slot past the sender's declared size must never survive into the host's copy"
    );

    // And the allocator is otherwise entirely ordinary, so the refusal above
    // is about the tail and nothing else.
    let ptr = allocator
        .allocate(4096, 256)
        .expect("the mechanism allocates");
    // SAFETY: `ptr` came from this allocator with exactly these parameters.
    unsafe { allocator.deallocate(ptr, 4096, 256) };
}

/// **A refused vtable still owes the plugin a release.**
///
/// `bad-tier` creates real allocator state, returns `Ok`, and publishes a
/// device tier code no host knows. The plugin has therefore done everything
/// right up to the point where the host applies its *own* policy and refuses.
///
/// The `Ok` is what creates the debt: from the plugin's side an allocator
/// exists and will exist until somebody calls `release`. A host that just
/// returns `Err` strands it — and because the module's live-allocator tally is
/// what gates unload, a single refusal like this permanently disables the gate
/// for the life of the process.
///
/// Asserting through `try_unload` rather than through a counter is deliberate:
/// the gate is the thing that actually breaks, so that is what the test
/// exercises.
#[test]
fn a_vtable_the_host_refuses_after_ok_is_still_released() {
    let _serial = serial();
    let plugin = load();

    let error = plugin
        .factory("bad-tier")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_ALLOCATOR, None)
        .expect_err("a tier code this host does not know must be refused");
    assert!(
        matches!(error, PluginError::Contract { .. }),
        "the refusal is the host's own policy, not a plugin failure: {error}"
    );

    // The refusal must have left the module exactly as it found it. If the
    // plugin's state were stranded the report would still show one live
    // allocator and this would come back as a rejection.
    plugin
        .try_unload()
        .expect("a refused open must leave no plugin state behind");
}

/// **A vtable the host cannot even parse is still released.**
///
/// The sibling of the test above, and the harder half. There the host refused
/// a vtable it had read successfully, so it could find `release` the ordinary
/// way. Here `read_prefix` itself refuses — a required slot is null — so the
/// host has no validated view to look in, and the only way to honour the
/// post-`Ok` debt is to read the vtable *again* without validating it and call
/// `release` out of that.
///
/// This is the path that decides whether the abandon route works at all. A
/// re-read that validates is deterministic over the same bytes: it fails
/// identically, returns before reaching `release`, and the plugin's state is
/// stranded for the life of the process with no error anywhere.
#[test]
fn a_vtable_the_host_cannot_parse_is_still_released() {
    let _serial = serial();
    let plugin = load();

    let error = plugin
        .factory("missing-slot")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_ALLOCATOR, None)
        .expect_err("a vtable missing a required slot must be refused");
    assert_eq!(
        error.status_code(),
        Some(NxmemStatusCode::ShortStruct),
        "a null required slot is reported as a short struct, the same code the header \
         contract uses for a vtable that cannot supply what the level requires: {error}"
    );

    // The plugin created real state and took its own reference before
    // returning `Ok`. If the host did not reach `release`, that reference is
    // still held and this comes back as a rejection naming a live allocator.
    plugin
        .try_unload()
        .expect("a vtable the host could not parse must still have been released");
}

/// **A retirable queued release does not leak anything.**
///
/// The counterpart to the test above: leaking is the fallback, not the policy.
/// `lazy` retires what it is asked to, so dropping its allocator drains the
/// queue first and the table is freed normally — the module's queued tally
/// comes back to zero and the gate reopens.
#[test]
fn dropping_an_allocator_drains_what_the_plugin_will_retire() {
    let _serial = serial();
    let plugin = load();
    let allocator = plugin
        .factory("lazy")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_ALLOCATOR | NXMEM_CAP_DEFERRED_RELEASE, None)
        .expect("the mechanism opens");

    let ptr = allocator.allocate(4096, 64).expect("allocation");
    // SAFETY: a live allocation from this allocator with matching parameters.
    let _ticket = unsafe { allocator.enqueue_release(ptr, 4096, 64) }.expect("queued release");
    assert_eq!(allocator.core().outstanding_releases(), 1);

    let before = PluginAllocatorCore::leaked_callback_tables();
    drop(allocator);
    assert_eq!(
        PluginAllocatorCore::leaked_callback_tables(),
        before,
        "a release the plugin is willing to retire must not cost a leaked table"
    );

    // And the gate really did reopen, which is the point of draining.
    plugin
        .try_unload()
        .expect("a drained allocator leaves nothing outstanding");
}

/// **Dropping a plugin with live objects must not unmap it.**
///
/// `try_unload` consumes the plugin, so every early return, `?` and unwind
/// reaches `drop` instead — and `drop` has no channel to refuse through. Its
/// only safe option is to keep the module mapped.
///
/// Whether `dlclose` would *actually* unmap is a platform accident: glibc
/// unmaps a refcount-zero DSO without `DF_1_NODELETE`, macOS commonly
/// declines. Relying on the platform declining is not a safety property, so
/// the assertion is on the host's own decision to leak rather than on whether
/// the mapping survived.
#[test]
fn dropping_a_plugin_with_live_objects_keeps_the_module_mapped() {
    let _serial = serial();
    let before = MemoryPlugin::forced_module_leaks();
    let unmapped_before = PluginModule::modules_unmapped();

    {
        let plugin = load();
        let allocator = plugin
            .factory("eager")
            .expect("the mechanism is published")
            .open(NXMEM_CAP_ALLOCATOR, None)
            .expect("the mechanism opens");
        // The allocator is still live when the plugin goes out of scope, and
        // nothing here calls `try_unload`.
        drop(plugin);
        assert_eq!(
            MemoryPlugin::forced_module_leaks() - before,
            1,
            "dropping a plugin whose allocator is still live must keep the module mapped"
        );
        drop(allocator);
    }

    // The allocator has now gone too, so every reference the host ever handed
    // out is released — every one except the reference the drop above kept on
    // purpose. `PluginModule::drop` is the only path that drops the `library`
    // field, and dropping that field *is* the `dlclose`, so this asserts on
    // the unmap itself rather than on the host's record of having decided
    // against it.
    assert_eq!(
        PluginModule::modules_unmapped(),
        unmapped_before,
        "a module dropped with live objects must never reach dlclose, even after \
         those objects are gone"
    );

    // An idle plugin, by contrast, unloads cleanly and costs nothing.
    let after_live_drop = MemoryPlugin::forced_module_leaks();
    load().try_unload().expect("an idle plugin unloads");
    assert_eq!(
        MemoryPlugin::forced_module_leaks(),
        after_live_drop,
        "a clean unload must not be counted as a forced leak"
    );
    // And it really did unmap, which is what makes the assertion above mean
    // something: the counter is not simply stuck at zero.
    assert_eq!(
        PluginModule::modules_unmapped() - unmapped_before,
        1,
        "the clean unload must be the only module this test unmapped"
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

/// A plugin that keeps part of the mapping must be quarantined, not refunded.
///
/// This drives the real [`DeviceAllocator::release`] entry point rather than
/// the private outcome-interpreting helper, because the branch worth pinning
/// is the one the production caller actually reaches.
#[test]
fn a_partial_release_is_quarantined_rather_than_refunded() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(&plugin, "quarantining", NXMEM_CAP_ALLOCATOR);
    let ptr = allocator.allocate(8192, 64).expect("allocation");

    // SAFETY: the address is live with these exact parameters.
    let outcome = unsafe { allocator.release(ptr, 8192, 64) };
    let AllocationReleaseOutcome::Quarantined {
        accounting,
        residual,
    } = outcome
    else {
        panic!("a partial release must quarantine, got {outcome:?}");
    };
    assert_eq!(
        accounting.allocation_bytes, 8192,
        "the full allocation size is still what was charged"
    );
    assert_eq!(
        accounting.unmapped_bytes, 4096,
        "only the part the plugin really gave back may be credited"
    );
    assert_eq!(
        residual.retained_bytes, 4096,
        "the rest stays owned by the plugin"
    );
    assert_eq!(
        residual.address,
        ptr.as_ptr() as usize,
        "the quarantine record must name the address so it is never reissued"
    );
    assert_eq!(
        allocator.core().live_allocation_count(),
        0,
        "the host no longer tracks it: the plugin does"
    );
}

/// A release state from a later contract level fails closed.
///
/// The host has no way to know whether the memory is safe to reuse, so it must
/// quarantine rather than guess in either direction.
#[test]
fn a_release_state_from_the_future_is_quarantined_not_guessed() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(&plugin, "future-state", NXMEM_CAP_ALLOCATOR);
    let ptr = allocator.allocate(4096, 64).expect("allocation");

    // SAFETY: the address is live with these exact parameters.
    let outcome = unsafe { allocator.release(ptr, 4096, 64) };
    let AllocationReleaseOutcome::Quarantined {
        accounting,
        residual,
    } = outcome
    else {
        panic!("an uninterpretable state must fail closed, got {outcome:?}");
    };
    assert_eq!(
        accounting.unmapped_bytes, 0,
        "nothing may be credited back from a state the host cannot read"
    );
    assert_eq!(
        residual.retained_bytes, 4096,
        "the whole allocation is presumed still owned by the plugin"
    );
    assert_eq!(allocator.core().live_allocation_count(), 0);
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

/// **A committed prefix is a live *view*, and the unload gate is told about
/// it.**
///
/// `live_capabilities` counts the prefix handle; `live_views` counts the
/// mappings made *through* it into allocations, which is a different thing and
/// a strictly longer-lived one — a view keeps a window open into an
/// allocation's address range whether or not the handle that created it still
/// exists. Unmapping the module with a view open would strand that window in
/// freed text, so the plugin must report views separately and the gate must
/// see them.
///
/// This asserts on deltas rather than absolutes because the counter lives
/// inside the loaded module and is process-wide.
#[test]
fn a_committed_shared_prefix_is_reported_as_a_live_view() {
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

    let views_before = plugin
        .module()
        .unload_report()
        .expect("the plugin reports readiness")
        .live_views;

    let prefix = shared
        .create_shared_prefix(16 * 1024)
        .expect("creating a shared prefix");
    assert_eq!(
        plugin
            .module()
            .unload_report()
            .expect("readiness")
            .live_views,
        views_before,
        "a prefix that has not been mapped into anything is not yet a view"
    );

    let ptr = allocator.allocate(64 * 1024, 4096).expect("allocation");
    shared
        .commit_shared_prefix(prefix.as_ref(), ptr, 64 * 1024, 0)
        .expect("committing the prefix into the allocation");

    let committed = plugin
        .module()
        .unload_report()
        .expect("the plugin reports readiness");
    assert_eq!(
        committed.live_views - views_before,
        1,
        "committing a prefix opens exactly one view into the allocation"
    );

    // The gate refuses while the view is open, and the refusal carries the
    // count — this is the path the loader actually reads.
    let rejection = plugin
        .try_unload()
        .expect_err("a plugin with a live view must not unload");
    assert_eq!(
        rejection.report.live_views - views_before,
        1,
        "the refusal must surface the open view, got: {:?}",
        rejection.report
    );
    let plugin = rejection
        .into_plugin()
        .expect("this refusal is recoverable; the caller still owns the handle");

    // A view outlives the handle that made it: dropping the prefix retires the
    // capability, not the mapping.
    drop(prefix);
    assert_eq!(
        plugin
            .module()
            .unload_report()
            .expect("readiness")
            .live_views
            - views_before,
        1,
        "the view belongs to the allocation, not to the handle that opened it"
    );

    // It retires with the allocation it looks into.
    // SAFETY: live allocation with matching parameters.
    let outcome = unsafe { allocator.release(ptr, 64 * 1024, 4096) };
    assert!(matches!(outcome, AllocationReleaseOutcome::Complete { .. }));
    assert_eq!(
        plugin
            .module()
            .unload_report()
            .expect("readiness")
            .live_views,
        views_before,
        "removing the block that was mapped into must retire its views"
    );

    drop(allocator);
    plugin
        .try_unload()
        .expect("nothing is live once the view has retired");
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

// ─── the host's drain loop on drop ──────────────────────────────────────────

/// **An allocation the caller never freed is reported, not abandoned.**
///
/// Dropping an allocator with a live allocation is a host-side bug, but the
/// plugin has already been told that allocation exists — it is in the module's
/// tally and in whatever backing map the mechanism keeps. Simply dropping the
/// host's record would strand the bytes on the plugin's side forever and
/// wedge the unload gate shut on an object nobody can name any more. So the
/// host must push each surviving allocation back through the plugin's
/// terminal slot on its way out.
///
/// Both tallies are checked. The host's own is the weaker of the two — it
/// would return to zero if the host merely stopped counting — so the binding
/// assertion is the module's, which only moves if the plugin was really told.
#[test]
fn dropping_an_allocator_reports_allocations_the_caller_never_freed() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(&plugin, "eager", NXMEM_CAP_ALLOCATOR);

    let _leaked_on_purpose = allocator.allocate(4096, 64).expect("allocation");
    let _also_leaked = allocator.allocate(8192, 256).expect("allocation");
    assert_eq!(
        plugin.module().host_live_counts().allocations,
        2,
        "both allocations are live and neither will be freed by this test"
    );
    assert_eq!(
        plugin
            .module()
            .unload_report()
            .expect("readiness")
            .live_allocations,
        2,
        "and the plugin knows about both of them"
    );

    let terminal_before = terminal_releases(&plugin);
    drop(allocator);

    // The binding assertion. The module's `live_allocations` returning to
    // zero would *not* be enough on its own: this plugin also reclaims
    // whatever is left during its own teardown, so that counter comes back to
    // zero whether or not the host said anything. Counting entries into the
    // terminal slot separates "the host handed these back" from "the plugin
    // cleaned up after it", and only the first is the ABI's guarantee — a
    // real mechanism may not be able to free a block safely on its own
    // schedule.
    assert_eq!(
        terminal_releases(&plugin) - terminal_before,
        2,
        "every allocation the caller left behind must be pushed back through the plugin's \
         terminal slot, not silently dropped"
    );
    assert_eq!(
        plugin
            .module()
            .unload_report()
            .expect("readiness")
            .live_allocations,
        0,
        "and the plugin must end up owning nothing"
    );
    assert_eq!(
        plugin.module().host_live_counts().allocations,
        0,
        "as must the host's own tally"
    );
    plugin
        .try_unload()
        .expect("nothing is left outstanding on either side");
}

/// **The drain on drop is a loop, and it has to be.**
///
/// A mechanism is entitled to retire less than it is offered — a stream-ordered
/// allocator can only retire what its device has actually finished with — so
/// one pass is not enough in general. `drip` retires exactly one release per
/// call whatever budget it is given, which makes the pass count the binding
/// constraint: three queued releases need three passes, and a host that made
/// only one would strand two of them and leak the callback table.
///
/// The pass count is observed from the far side of the ABI, so the assertion is
/// on the host's behaviour rather than on the host's opinion of it.
#[test]
fn dropping_an_allocator_drains_in_as_many_passes_as_the_plugin_needs() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(
        &plugin,
        "drip",
        NXMEM_CAP_ALLOCATOR | NXMEM_CAP_DEFERRED_RELEASE,
    );

    for index in 0..3usize {
        let bytes = 1024 * (index + 1);
        let ptr = allocator.allocate(bytes, 64).expect("allocation");
        // SAFETY: live allocation with matching parameters.
        unsafe { allocator.enqueue_release(ptr, bytes, 64) }.expect("queued release");
    }
    assert_eq!(
        allocator.core().outstanding_releases(),
        3,
        "three releases are queued and this mechanism retires one per call"
    );

    let calls_before = drain_calls(&plugin);
    let leaks_before = PluginAllocatorCore::leaked_callback_tables();
    drop(allocator);

    assert_eq!(
        drain_calls(&plugin) - calls_before,
        3,
        "the host must come back once per release the plugin was willing to retire"
    );
    assert_eq!(
        plugin.module().host_live_counts().queued_releases,
        0,
        "every queued release must have retired, not just the first"
    );
    assert_eq!(
        PluginAllocatorCore::leaked_callback_tables(),
        leaks_before,
        "a fully drained allocator must not leak its callback table"
    );
    plugin
        .try_unload()
        .expect("a fully drained allocator leaves nothing outstanding");
}

/// **Each pass offers an unbounded budget, and that is load-bearing.**
///
/// The pass count is bounded, so the per-pass budget cannot also be. `lazy`
/// retires everything it is offered, and this queues more releases than the
/// host will ever make passes — so the drain completes if and only if a single
/// pass is allowed to retire an unbounded number. A host that offered a small
/// budget per pass would run out of passes with releases still queued.
#[test]
fn dropping_an_allocator_offers_an_unbounded_budget_per_pass() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(
        &plugin,
        "lazy",
        NXMEM_CAP_ALLOCATOR | NXMEM_CAP_DEFERRED_RELEASE,
    );

    // Comfortably more than the host's pass bound, so the budget is the only
    // thing that can carry this.
    const QUEUED: usize = 20;
    for _ in 0..QUEUED {
        let ptr = allocator.allocate(2048, 64).expect("allocation");
        // SAFETY: live allocation with matching parameters.
        unsafe { allocator.enqueue_release(ptr, 2048, 64) }.expect("queued release");
    }
    assert_eq!(allocator.core().outstanding_releases(), QUEUED as u64);

    let calls_before = drain_calls(&plugin);
    let leaks_before = PluginAllocatorCore::leaked_callback_tables();
    drop(allocator);

    assert_eq!(
        drain_calls(&plugin) - calls_before,
        1,
        "an unbounded budget must let a willing mechanism retire the whole queue in one pass"
    );
    assert_eq!(
        plugin.module().host_live_counts().queued_releases,
        0,
        "all twenty releases must have retired inside the host's pass bound"
    );
    assert_eq!(
        PluginAllocatorCore::leaked_callback_tables(),
        leaks_before,
        "nothing was left outstanding, so nothing may be leaked"
    );
    plugin.try_unload().expect("the gate reopens");
}

/// **A completion arriving inside `enqueue_release` must never see an
/// underflowed count.**
///
/// The host increments its queued-release counters *before* entering the
/// plugin, precisely so a completion the plugin reports synchronously from
/// inside that call decrements something that was already incremented. Count
/// afterwards and the callback subtracts one from zero, and an unsigned
/// counter does not go negative — it wraps to `u64::MAX`.
///
/// `reentrant-completion` retires the release from inside `enqueue_release`
/// and then calls the host's reclaim hook, which is what lets the host read
/// its own accounting at exactly that instant. Without that second callback
/// the window closes before anything outside can look into it.
#[test]
fn a_completion_arriving_inside_enqueue_never_sees_an_underflowed_count() {
    let _serial = serial();
    let plugin = load();
    let observer = Arc::new(MidCallObserver::default());
    let allocator = plugin
        .factory("reentrant-completion")
        .expect("the mechanism is published")
        .open(
            NXMEM_CAP_ALLOCATOR | NXMEM_CAP_DEFERRED_RELEASE,
            Some(observer.clone()),
        )
        .expect("the mechanism opens");
    observer
        .core
        .set(Arc::downgrade(allocator.core()))
        .expect("bound once");
    observer
        .module
        .set(Arc::downgrade(plugin.module()))
        .expect("bound once");

    let ptr = allocator.allocate(4096, 64).expect("allocation");
    // SAFETY: live allocation with matching parameters.
    let ticket = unsafe { allocator.enqueue_release(ptr, 4096, 64) }.expect("queued release");

    assert_eq!(
        observer.observations.load(Ordering::Acquire),
        1,
        "the mechanism must actually have re-entered the host from inside enqueue_release; \
         without that this test would be looking at nothing"
    );
    assert_eq!(
        observer.retired_at_observation.load(Ordering::Acquire),
        1,
        "and it must have reported the completion before re-entering, or the window \
         under test never opened"
    );
    assert_eq!(
        observer.outstanding.load(Ordering::Acquire),
        0,
        "the reentrant completion decremented a count the host had already incremented; \
         a count taken afterwards would read u64::MAX here"
    );
    assert_eq!(
        observer.module_queued.load(Ordering::Acquire),
        0,
        "and the same for the module-wide tally the unload gate reads"
    );

    // The end state is ordinary, which is what makes the observation above the
    // only thing this test could have caught.
    let retired = allocator.core().retired_releases();
    assert_eq!(retired.len(), 1, "the completion really did arrive");
    assert_eq!(retired[0].ticket, ticket);
    assert_eq!(allocator.core().outstanding_releases(), 0);
    assert_eq!(plugin.module().host_live_counts().queued_releases, 0);

    let leaks_before = PluginAllocatorCore::leaked_callback_tables();
    drop(allocator);
    assert_eq!(
        PluginAllocatorCore::leaked_callback_tables(),
        leaks_before,
        "nothing is outstanding, so the callback table is freed normally"
    );
    plugin.try_unload().expect("the gate is open");
}

/// **A plugin thread may report a completion after the allocator is gone.**
///
/// This is the guarantee the ABI actually makes — the host's callback table
/// outlives the allocator's final `release` *and every queued release naming
/// it* — and it is the one thing every other test in this suite reaches only
/// by accident, because every other path into `release_completed` runs
/// synchronously inside a host call, where the table is trivially alive.
///
/// `callback-after-drop` refuses to retire through the drain slot, so the
/// host's drop is forced to keep the table alive; the release is then reported
/// from a thread the plugin spawns, after `AllocatorCore::drop` has returned.
/// If the host had freed that box, this would be a use-after-free of the
/// bridge the plugin dereferences.
///
/// Note on tooling: this cannot be run under Miri, which cannot execute the
/// `dlopen`ed cdylib at all — the whole host integration suite is outside it.
/// So the assertions below are ordinary observable ones, not sanitiser
/// coverage, and they are chosen to fail deterministically rather than to rely
/// on a freed allocation happening to look wrong.
#[test]
fn a_plugin_thread_reports_a_completion_after_its_allocator_is_gone() {
    let _serial = serial();
    let plugin = load();
    let allocator = plugin
        .factory("callback-after-drop")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_ALLOCATOR | NXMEM_CAP_DEFERRED_RELEASE, None)
        .expect("the mechanism opens");

    let ptr = allocator.allocate(4096, 64).expect("allocation");
    // SAFETY: live allocation with matching parameters.
    let _ticket = unsafe { allocator.enqueue_release(ptr, 4096, 64) }.expect("queued release");
    assert_eq!(allocator.core().outstanding_releases(), 1);

    let leaks_before = PluginAllocatorCore::leaked_callback_tables();
    let pins_before = Arc::strong_count(plugin.module());
    drop(allocator);
    assert_eq!(
        PluginAllocatorCore::leaked_callback_tables() - leaks_before,
        1,
        "the host must keep the table the plugin still holds a pointer to"
    );
    assert_eq!(
        plugin.module().host_live_counts().queued_releases,
        1,
        "and the release is still outstanding as far as the host knows"
    );

    // The allocator is gone. Now the plugin dereferences the host's callback
    // table from a thread of its own — which is exactly what the contract
    // says it may do, and exactly what freeing the table would have made
    // undefined.
    // SAFETY: an `extern "C" fn() -> u64` exported by the test plugin; the
    // module is still mapped, which is itself part of what is under test.
    let reported = {
        let hook: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = unsafe {
            plugin
                .module()
                .library()
                .get(onnx_runtime_memory_testplugin::SYMBOL_REPORT_PARKED_COMPLETION)
        }
        .expect("the test plugin exports its parked-completion hook");
        // SAFETY: as above.
        unsafe { hook() }
    };
    assert_eq!(
        reported, 1,
        "the hook must actually have reported a completion; a zero here would mean \
         this test proved nothing"
    );

    // The host accepted the completion through a table whose allocator no
    // longer exists, and its accounting moved as a result.
    assert_eq!(
        plugin.module().host_live_counts().queued_releases,
        0,
        "the completion arrived and the host retired the queued release"
    );
    assert_eq!(
        plugin.module().host_live_counts().allocations,
        0,
        "and closed the allocation it named"
    );

    // Both gates are open now, but the module still cannot be unmapped: the
    // leaked table owns a module reference of its own and always will. That
    // reference is the observable half of "the box was forgotten, not freed" —
    // had it been freed, the reference would have gone with it and this unload
    // would succeed.
    assert_eq!(
        Arc::strong_count(plugin.module()),
        pins_before - 1,
        "dropping the allocator releases the core's module reference and keeps the \
         leaked table's own"
    );
    let unmapped_before = PluginModule::modules_unmapped();
    let rejection = plugin
        .try_unload()
        .expect_err("a leaked callback table pins the module for the life of the process");
    assert_eq!(
        rejection.host.total(),
        0,
        "the host's half of the gate is open: {:?}",
        rejection.host
    );
    assert_eq!(
        rejection.report.total(),
        0,
        "and so is the plugin's: {:?}",
        rejection.report
    );
    assert!(
        rejection.reason.contains("pin the module"),
        "the refusal must name the outside handle, got: {}",
        rejection.reason
    );
    assert!(
        rejection.into_plugin().is_none(),
        "this refusal is terminal; there is nothing the caller can retire"
    );
    assert_eq!(
        PluginModule::modules_unmapped(),
        unmapped_before,
        "and the module really did stay mapped"
    );
}

/// **An abandoned park must not leave a pointer to a destroyed state behind.**
///
/// `callback-after-drop` parks its allocator state where a plugin-owned thread
/// can find it later, taking no reference of its own: it relies on a queued
/// release holding the state alive. Open one and drop it with nothing queued
/// and that assumption does not hold — the state's refcount reaches zero and
/// it is destroyed while the parked pointer still names it. Anything that
/// subsequently ran the completion hook would dereference freed memory.
///
/// The pointer is therefore cleared as the state is destroyed. This is the
/// only test that reaches that clearing: every other use of the mechanism
/// keeps a release queued, which keeps the state alive, so the clearing never
/// fires and could be deleted unnoticed. The park is process-global inside the
/// loaded module, so a stale pointer would not merely be latent — it would be
/// waiting for the next test in this binary that calls the hook.
///
/// The assertion reads the pointer rather than following it, so it stays sound
/// precisely when the invariant it checks is broken.
#[test]
fn an_abandoned_park_is_cleared_when_its_state_dies() {
    let _serial = serial();
    let plugin = load();
    assert_eq!(
        parked_state_is_set(&plugin),
        0,
        "no park may be outstanding when this test starts"
    );

    let allocator = plugin
        .factory("callback-after-drop")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_ALLOCATOR | NXMEM_CAP_DEFERRED_RELEASE, None)
        .expect("the mechanism opens");
    assert_eq!(
        parked_state_is_set(&plugin),
        1,
        "opening the mechanism must park its state, or there is nothing here to abandon"
    );

    // Dropped with nothing queued, so nothing else is holding the state.
    drop(allocator);
    assert_eq!(
        parked_state_is_set(&plugin),
        0,
        "a park whose state has been destroyed must not still name it; the hook has no way to \
         tell a stale pointer from a live one and would dereference freed memory"
    );

    plugin
        .try_unload()
        .expect("an abandoned park leaves nothing outstanding");
}

/// **A refused `enqueue_release` leaves the allocation exactly as it was.**
///
/// To queue a release the host must first take the allocation out of its live
/// map — it is about to stop being the host's to free — and it must count the
/// release before entering the plugin. If the plugin then refuses, none of
/// that happened: the plugin has taken on nothing and the allocation is still
/// the caller's, with the caller still holding the pointer.
///
/// A host that only unwound its counters would leave the address unknown to
/// itself, so the caller's eventual `release` would be refused as a stray
/// pointer and the bytes would be stranded with no way to name them. The
/// binding assertion here is therefore the one that follows the refusal: the
/// same pointer must still be releasable, normally, afterwards.
#[test]
fn a_refused_enqueue_leaves_the_allocation_releasable() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(
        &plugin,
        "refusing-deferred",
        NXMEM_CAP_ALLOCATOR | NXMEM_CAP_DEFERRED_RELEASE,
    );

    let ptr = allocator.allocate(4096, 64).expect("allocation");
    // SAFETY: live allocation with matching parameters.
    let error = unsafe { allocator.enqueue_release(ptr, 4096, 64) }
        .expect_err("this mechanism refuses every deferral");
    assert!(
        format!("{error}").contains("refuses to queue"),
        "the refusal must come from the plugin, not from the host's own bookkeeping, \
         got: {error}"
    );

    assert_eq!(
        allocator.core().outstanding_releases(),
        0,
        "a refused deferral is not an outstanding release"
    );
    assert_eq!(
        plugin.module().host_live_counts().queued_releases,
        0,
        "nor a reason to hold the unload gate shut"
    );
    assert_eq!(
        plugin.module().host_live_counts().allocations,
        1,
        "the allocation is still live and still the caller's"
    );

    // The assertion that binds: the host must still recognise the pointer.
    // SAFETY: the allocation was never handed over, so it is still live with
    // exactly these parameters.
    let outcome = unsafe { allocator.release(ptr, 4096, 64) };
    assert!(
        matches!(outcome, AllocationReleaseOutcome::Complete { .. }),
        "an allocation the plugin refused to take must still be releasable, got {outcome:?}"
    );
    assert_eq!(plugin.module().host_live_counts().allocations, 0);

    let leaks_before = PluginAllocatorCore::leaked_callback_tables();
    drop(allocator);
    assert_eq!(
        PluginAllocatorCore::leaked_callback_tables(),
        leaks_before,
        "a refused deferral must not cost a leaked callback table either"
    );
    plugin.try_unload().expect("nothing is outstanding");
}

/// **The callback table's fate is decided after the plugin's `release`, not
/// before it.**
///
/// `AllocatorCore::drop` calls the plugin's `release(ctx)` and only then reads
/// `outstanding_releases` to decide whether to free the callback table or leak
/// it. The in-code comment says that order is load-bearing, because `release`
/// is an ordinary host call and the plugin may touch the table from inside it.
/// Nothing observed it: every other mechanism's `release` calls nothing at all,
/// so freeing the table first would have been a use-after-free that no test
/// performed.
///
/// `complete-on-release` performs it. It refuses to drain, so a queued release
/// is still outstanding when teardown starts, and then reports that release's
/// completion from inside `release` — taking the outstanding count from one to
/// zero at the one instant the host is between those two steps.
///
/// The assertion is deliberately **not** on the use-after-free. Reading freed
/// memory is undefined, so a test built on it passes or fails by accident, and
/// Miri cannot reach this suite at all (it cannot execute a `dlopen`ed cdylib).
/// It is on the decision the ordering changes: reading the count too early
/// reads a one that is about to become a zero, and leaks a table that did not
/// need leaking. That is a real, permanent, deterministic defect, and it is
/// memory-safe to observe in both directions.
#[test]
fn the_callback_table_outlives_the_plugins_own_release_call() {
    let _serial = serial();
    let plugin = load();
    let allocator = open(
        &plugin,
        "complete-on-release",
        NXMEM_CAP_ALLOCATOR | NXMEM_CAP_DEFERRED_RELEASE,
    );

    let ptr = allocator.allocate(4096, 64).expect("allocation");
    // SAFETY: a live allocation from this allocator with matching parameters.
    let _ticket = unsafe { allocator.enqueue_release(ptr, 4096, 64) }.expect("queued release");
    assert_eq!(
        allocator.core().outstanding_releases(),
        1,
        "the release must still be outstanding when teardown begins, or the ordering \
         under test never arises"
    );

    let leaks_before = PluginAllocatorCore::leaked_callback_tables();
    let drain_calls_before = drain_calls(&plugin);
    drop(allocator);

    // The mechanism refuses to drain, so the drop's drain loop cannot be what
    // retired this release: it asked once, was told nothing retired, and gave
    // up. Whatever cleared the outstanding count came from inside `release`.
    assert_eq!(
        drain_calls(&plugin) - drain_calls_before,
        1,
        "a pass that retires nothing must end the drain loop"
    );
    assert_eq!(
        plugin.module().host_live_counts().queued_releases,
        0,
        "the completion reported from inside `release` must have been accepted"
    );
    assert_eq!(
        PluginAllocatorCore::leaked_callback_tables(),
        leaks_before,
        "the table's fate must be read after the plugin has finished with it; a count read \
         before `release` is a count the plugin was still about to change, and leaks a \
         table that nothing could ever name again"
    );
    assert_eq!(
        plugin.module().host_live_counts().total(),
        0,
        "and the accounting must balance, so the gate really did reopen"
    );
    plugin
        .try_unload()
        .expect("a release retired from inside `release` leaves nothing outstanding");
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
        .open(
            NXMEM_CAP_ALLOCATOR,
            Some(ScriptedReclaim::granting(1 << 20)),
        )
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

/// A plugin with nothing live unloads — and really is unmapped.
#[test]
fn an_idle_plugin_unloads() {
    let _serial = serial();
    let plugin = load();
    assert_eq!(plugin.module().host_live_counts().total(), 0);
    let unmapped_before = PluginModule::modules_unmapped();
    plugin
        .try_unload()
        .expect("a plugin with nothing live must unload");
    // `try_unload` returning `Ok` only says the gate opened. This says the
    // module actually reached `dlclose`: nothing outside the handle pinned it,
    // so the last strong reference went and `PluginModule::drop` ran.
    assert_eq!(
        PluginModule::modules_unmapped() - unmapped_before,
        1,
        "a clean unload must actually unmap the module"
    );
}

/// **The unmap count follows `PluginModule::drop`, not `try_unload`.**
///
/// Every other assertion on `modules_unmapped` in this suite either watches it
/// move across a `try_unload`, or watches it stay still across a drop that
/// deliberately leaked. Both shapes are equally satisfied by a counter that
/// lives in `try_unload`'s success arm instead of in `PluginModule::drop` — and
/// a counter that lives there is no longer a proxy for `dlclose` at all, it is
/// a restatement of the decision `try_unload` just made. `MemoryPlugin::drop`'s
/// deliberate `mem::forget` could then be deleted outright without a single
/// test noticing, because nothing would be watching the drop path unmap.
///
/// This is the missing quadrant: an unload that happens *through the drop
/// path*, with no `try_unload` anywhere. The gate is open, so `MemoryPlugin`'s
/// drop takes its early return, ordinary field drops run, the last
/// `Arc<PluginModule>` goes, and the library unmaps.
#[test]
fn dropping_an_idle_plugin_unmaps_it_with_no_try_unload_at_all() {
    let _serial = serial();
    let plugin = load();
    assert_eq!(
        plugin.module().host_live_counts().total(),
        0,
        "nothing is live, so the gate this drop evaluates is open"
    );

    let unmapped_before = PluginModule::modules_unmapped();
    let leaks_before = MemoryPlugin::forced_module_leaks();
    drop(plugin);

    assert_eq!(
        PluginModule::modules_unmapped() - unmapped_before,
        1,
        "an idle plugin that is dropped rather than unloaded must still reach dlclose"
    );
    assert_eq!(
        MemoryPlugin::forced_module_leaks(),
        leaks_before,
        "an open gate is not a forced leak, whichever path reached it"
    );
}

/// **The unmap is counted where the mapping goes, not where a handle goes.**
///
/// The test above pins the counter to *a* drop path, but a counter duplicated
/// into both `try_unload`'s success arm and `MemoryPlugin::drop`'s open-gate
/// branch would satisfy it too — and that counter would still be lying, because
/// neither site can know whether the module actually unmapped. `module()` hands
/// out the `Arc`, so any embedder can hold a module reference that outlives the
/// plugin handle; both those sites would then count an unmap that did not
/// happen.
///
/// So this drives the module through a shutdown in which the unmap happens at a
/// moment when there is **no `MemoryPlugin` in existence and no `try_unload` on
/// the stack**. The only code that can observe it is `PluginModule`'s own
/// `Drop`, which is the one place the `library` field is dropped — and dropping
/// that field is the `dlclose`.
#[test]
fn the_unmap_lands_when_the_last_module_reference_goes_not_when_the_plugin_does() {
    let _serial = serial();
    let plugin = load();
    // An ordinary embedder handle: the loader publishes the `Arc`, so keeping
    // one is a supported thing to do, not a trick.
    let pinned = Arc::clone(plugin.module());

    let unmapped_before = PluginModule::modules_unmapped();
    let leaks_before = MemoryPlugin::forced_module_leaks();
    drop(plugin);
    assert_eq!(
        MemoryPlugin::forced_module_leaks(),
        leaks_before,
        "the gate was open, so this drop is not a forced leak"
    );
    assert_eq!(
        PluginModule::modules_unmapped(),
        unmapped_before,
        "an open gate does not unmap a module somebody else still holds; only the last \
         reference going can do that"
    );

    // Nothing above this line is a `MemoryPlugin` any more, and `try_unload`
    // was never called. This is the moment the mapping actually goes.
    drop(pinned);
    assert_eq!(
        PluginModule::modules_unmapped() - unmapped_before,
        1,
        "the unmap must be counted where the library is dropped, which is the only place \
         that knows it happened"
    );
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
    assert_eq!(
        count,
        onnx_runtime_memory_testplugin::MECHANISM_NAMES.len() as u64,
        "the host must take one factory per published mechanism"
    );
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
