//! Intel ITT (VTune / Inspector) collector (§48.8.2), behind the `itt` cargo
//! feature.
//!
//! This bridges the crate's [`TraceEvent`](crate::TraceEvent)s into Intel
//! **Instrumentation and Tracing Technology** (ITT) annotations via the official
//! [`ittapi`] crate. When an ITT data collector — VTune Profiler, Intel
//! Inspector, or any `libittnotify_collector` — is attached, the runtime's spans
//! appear as *tasks* on its timeline, correlated with the hardware counters and
//! microarchitectural analysis those tools capture. When nothing is attached,
//! every ITT call resolves to the static `ittnotify` stub and does nothing, so
//! the collector is **inert and allocation-light**, never a crash or a slowdown
//! (§48.8.10 "provider unavailable → graceful skip").
//!
//! ## Binding path: the safe `ittapi` crate (no `unsafe` here)
//!
//! Unlike [`cupti`](crate::cupti) — which must `dlopen` `libcupti` and parse raw
//! activity records, and is therefore the crate's one `unsafe` module — the ITT
//! path needs **no `unsafe` in this crate at all**. `ittapi` links the static
//! `ittnotify` C library (compiled from vendored source by `ittapi-sys`, no
//! system dependency) and exposes a safe [`Domain`]/[`Task`] surface. The crate
//! keeps `#![forbid(unsafe_code)]` active for `--features itt` builds; all FFI is
//! confined to the upstream `ittapi` crate.
//!
//! ## Event → ITT mapping
//!
//! | [`TracePhase`](crate::TracePhase) | ITT action |
//! |-----------------------------------|------------|
//! | `Begin` | `__itt_task_begin` — push a live task on this thread's ITT stack |
//! | `End`   | `__itt_task_end` — pop the most recent task on this thread |
//! | `Instant` / `Complete` | a zero-width "point" task (`begin` immediately followed by `end`) |
//! | `Metadata` | ignored (thread/process naming has no ITT-task equivalent here) |
//!
//! ITT's task API is a **per-thread nesting stack**, exactly matching Chrome
//! `B`/`E` semantics, so [`IttCollector`] keeps one task-guard stack per thread
//! (a `thread_local!`) and pushes/pops it on `Begin`/`End`.
//!
//! `Complete` (`"X"`) events — the phase a [`SpanGuard`](crate::SpanGuard) emits
//! on drop — carry a real duration, but ITT is a **live** API: a collector
//! timestamps the moment `begin`/`end` are *called*, and a `Complete` event is
//! only delivered *after* its span has already closed. There is no ITT call to
//! backdate a task, so a `Complete` (like an `Instant`) is bridged as a point
//! annotation at delivery time — enough to mark the op on the VTune timeline.
//! Callers that want ITT tasks with true VTune durations should emit `Begin`/
//! `End` pairs around the live work.
//!
//! ## Note on the §48.8.2 sketch
//!
//! The design sketch shows a `DashMap<String, StringHandle>` string-handle cache
//! and an `ittapi::marker` call. Neither maps onto `ittapi` 0.5: its
//! `StringHandle` is neither `Clone` nor `Send`/`Sync` (an external cache would
//! require `unsafe` to share), and it has no free `marker` function. Both are
//! unnecessary in practice — `ittnotify` already de-duplicates string handles in
//! its own internal table, so passing the event name by reference is correct and
//! cheap. Keeping the collector `unsafe`-free is worth the deviation.
//!
//! [`ittapi`]: https://crates.io/crates/ittapi
//! [`Domain`]: ittapi::Domain
//! [`Task`]: ittapi::Task

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use ittapi::{Domain, Task};

use crate::collector::TraceCollector;
use crate::error::Result;
use crate::event::{TraceEvent, TracePhase};

/// The default ITT domain name used by [`IttFactory`] when no explicit domain is
/// given. ITT groups a program's tasks by domain; `"nxrt"` keeps the runtime's
/// tasks on their own VTune lane.
pub const DEFAULT_ITT_DOMAIN: &str = "nxrt";

/// Environment variables an ITT data collector sets to attach itself to the
/// process (the collector shared library path). Their presence is the canonical
/// signal that ITT annotations will actually be recorded — without a collector,
/// the static `ittnotify` stubs discard everything.
const ITT_COLLECTOR_ENV_VARS: [&str; 2] = ["INTEL_LIBITTNOTIFY64", "INTEL_LIBITTNOTIFY32"];

/// Whether an ITT data collector (VTune / Inspector / `libittnotify_collector`)
/// is attached to this process.
///
/// This is what the [`IttFactory`] checks to decide between producing a live
/// collector and gracefully skipping (`Ok(None)`, §48.8.4). The check is the
/// presence of the collector's `INTEL_LIBITTNOTIFY{64,32}` environment variable,
/// which is how an ITT collector injects itself — the same signal `ittnotify`
/// itself uses to switch from stubs to the real implementation.
///
/// Note that an [`IttCollector`] is safe to build and use **regardless** of this
/// result: when no collector is attached, every ITT call is an inert stub. The
/// flag only lets the factory avoid adding a do-nothing collector to a fan-out.
#[must_use]
pub fn collector_attached() -> bool {
    ITT_COLLECTOR_ENV_VARS
        .iter()
        .any(|var| std::env::var_os(var).is_some())
}

/// Process-global registry of ITT [`Domain`]s, keyed by name.
///
/// ITT domains are process-global and, by design, **never destroyed** (the ITT
/// API has no domain-free call), so each domain is created once and leaked to a
/// `'static` reference. Leaking is the *correct* lifetime here, not a shortcut:
/// it lets per-thread [`Task`] guards borrow the domain for `'static` (see
/// [`TASK_STACK`]) without ever dangling, and de-duplicates domains so repeated
/// [`IttCollector::new`] calls with the same name share one VTune lane.
static DOMAINS: OnceLock<Mutex<HashMap<String, &'static Domain>>> = OnceLock::new();

/// Get (creating once) the process-global [`Domain`] for `name`.
fn domain_for(name: &str) -> &'static Domain {
    let registry = DOMAINS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(domain) = map.get(name) {
        return domain;
    }
    // ITT domains live for the whole process; leaking gives the `'static` a
    // thread-local task guard can safely borrow (see the registry docs).
    let leaked: &'static Domain = Box::leak(Box::new(Domain::new(name)));
    map.insert(name.to_string(), leaked);
    leaked
}

thread_local! {
    /// This thread's stack of open ITT tasks (`Begin` pushes, `End` pops).
    ///
    /// ITT's task API is inherently per-thread and LIFO-nested, matching Chrome
    /// `B`/`E` semantics; keeping the guards thread-local means a `Begin`/`End`
    /// pair recorded on one thread nests correctly regardless of what other
    /// threads are doing. Any guards still open when the thread exits are dropped
    /// then (ending their tasks), so an unbalanced trace degrades gracefully
    /// rather than leaking an open task forever.
    static TASK_STACK: RefCell<Vec<Task<'static>>> = const { RefCell::new(Vec::new()) };
}

/// A [`TraceCollector`] that bridges events into Intel ITT annotations (§48.8.2).
///
/// Construct one with [`IttCollector::new`]; it is always safe to build and use,
/// whether or not an ITT collector is attached (see the [module docs](crate::itt)
/// for the event→ITT mapping and the inert-when-unattached contract). Share it
/// behind an [`Arc`](std::sync::Arc) like any other collector, or hand it to a
/// [`CompositeCollector`](crate::CompositeCollector) to feed VTune *and* an
/// on-disk trace from the same annotations.
pub struct IttCollector {
    /// The process-global ITT domain these tasks are tagged with.
    domain: &'static Domain,
    /// Whether an ITT collector was attached when this collector was built.
    attached: bool,
}

impl IttCollector {
    /// Create a collector that emits ITT tasks under the domain `domain_name`.
    ///
    /// Never fails and never blocks on VTune: with no collector attached the ITT
    /// calls are stubs, so the returned collector is simply inert. Domains are
    /// de-duplicated process-wide, so calling this repeatedly with the same name
    /// reuses one VTune lane.
    #[must_use]
    pub fn new(domain_name: &str) -> Self {
        Self {
            domain: domain_for(domain_name),
            attached: collector_attached(),
        }
    }

    /// Whether an ITT data collector (VTune / Inspector) was attached to the
    /// process when this collector was constructed.
    ///
    /// `false` means every [`emit`](TraceCollector::emit) is an inert ITT stub —
    /// useful for tests and for deciding whether the collector is worth keeping
    /// in a fan-out (though [`IttFactory`] already makes that call).
    #[must_use]
    pub fn attached(&self) -> bool {
        self.attached
    }
}

impl std::fmt::Debug for IttCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `ittapi::Domain` is not `Debug`; report only what is observable.
        f.debug_struct("IttCollector")
            .field("attached", &self.attached)
            .finish_non_exhaustive()
    }
}

impl TraceCollector for IttCollector {
    fn emit(&self, event: &TraceEvent) {
        match event.ph {
            // Live nested task: push a guard; `End` will pop and close it.
            TracePhase::Begin => {
                let task = Task::begin(self.domain, event.name.as_str());
                TASK_STACK.with(|stack| stack.borrow_mut().push(task));
            }
            // Close the most recent task on this thread. A stray `End` with no
            // matching `Begin` is a harmless no-op (empty stack).
            TracePhase::End => {
                TASK_STACK.with(|stack| {
                    let _ = stack.borrow_mut().pop();
                });
            }
            // Point annotations: ITT cannot backdate, so a `Complete`/`Instant`
            // becomes a zero-width task at delivery time (see module docs).
            TracePhase::Instant | TracePhase::Complete => {
                Task::begin(self.domain, event.name.as_str()).end();
            }
            // Thread/process naming has no ITT-task equivalent here.
            TracePhase::Metadata => {}
        }
    }

    fn flush(&self) -> Result<()> {
        // ITT annotations are delivered to the collector as they are emitted;
        // there is nothing buffered on our side to persist.
        Ok(())
    }
}

/// A graceful factory for [`IttCollector`] (§48.8.4).
///
/// Mirrors [`CuptiFactory`](crate::cupti::CuptiFactory): the crate has no
/// collector *registry* yet, so this is a standalone factory whose
/// [`try_create`](IttFactory::try_create) returns `Ok(None)` when no ITT
/// collector is attached — the "provider unavailable → graceful skip" contract
/// (§48.8.10) — and `Ok(Some(collector))` otherwise, ready to add to a
/// [`CompositeCollector`](crate::CompositeCollector).
#[derive(Debug, Default, Clone, Copy)]
pub struct IttFactory;

impl IttFactory {
    /// Try to build an ITT collector under [`DEFAULT_ITT_DOMAIN`].
    ///
    /// Returns `Ok(None)` when no ITT collector (VTune / Inspector) is attached,
    /// so a caller that requested `"itt"` on a machine without VTune is silently
    /// skipped rather than fed a do-nothing collector. Use
    /// [`try_create_in`](IttFactory::try_create_in) to choose the domain.
    ///
    /// # Errors
    ///
    /// Currently infallible; returns [`Result`] to match the §48.8.4 factory
    /// shape and to leave room for initialization that can fail later.
    pub fn try_create(&self) -> Result<Option<Box<dyn TraceCollector>>> {
        self.try_create_in(DEFAULT_ITT_DOMAIN)
    }

    /// Like [`try_create`](IttFactory::try_create) but tags tasks with the given
    /// ITT `domain_name`.
    ///
    /// # Errors
    ///
    /// Currently infallible (see [`try_create`](IttFactory::try_create)).
    pub fn try_create_in(&self, domain_name: &str) -> Result<Option<Box<dyn TraceCollector>>> {
        if !collector_attached() {
            return Ok(None);
        }
        Ok(Some(Box::new(IttCollector::new(domain_name))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(name: &str, ph: TracePhase) -> TraceEvent {
        TraceEvent {
            name: name.to_string(),
            cat: "compute".to_string(),
            ph,
            ts: 0,
            dur: (ph == TracePhase::Complete).then_some(10),
            pid: 1,
            tid: 1,
            scope: None,
            args: Some(json!({ "device": "cpu" })),
        }
    }

    #[test]
    fn new_never_panics_and_reports_attachment() {
        // Building a collector must succeed whether or not VTune is attached, and
        // `attached` must agree with the environment probe.
        let collector = IttCollector::new("test-domain");
        assert_eq!(collector.attached(), collector_attached());
    }

    #[test]
    fn domains_are_deduplicated() {
        // Same name → one leaked domain (one VTune lane), by pointer identity.
        let a = domain_for("dup-domain");
        let b = domain_for("dup-domain");
        assert!(std::ptr::eq(a, b));
    }

    #[test]
    fn emit_never_panics_for_any_phase() {
        // Without a collector attached every ITT call is an inert stub; with one
        // attached the calls are still safe. Either way, none of the phases —
        // including an unbalanced End — may panic.
        let collector = IttCollector::new("phase-domain");
        collector.emit(&event("begin", TracePhase::Begin));
        collector.emit(&event("end", TracePhase::End));
        collector.emit(&event("instant", TracePhase::Instant));
        collector.emit(&event("complete", TracePhase::Complete));
        collector.emit(&event("meta", TracePhase::Metadata));
        // Stray End with an empty stack must be a harmless no-op.
        collector.emit(&event("stray-end", TracePhase::End));
        collector.flush().expect("ITT flush is infallible");
    }

    #[test]
    fn nested_begin_end_balances_the_thread_stack() {
        let collector = IttCollector::new("nest-domain");
        collector.emit(&event("outer", TracePhase::Begin));
        collector.emit(&event("inner", TracePhase::Begin));
        collector.emit(&event("inner", TracePhase::End));
        collector.emit(&event("outer", TracePhase::End));
        // After a balanced sequence the thread-local task stack is empty again.
        TASK_STACK.with(|stack| assert!(stack.borrow().is_empty()));
    }

    #[test]
    fn factory_skips_gracefully_when_unattached() {
        // On a box without an ITT collector this must be Ok(None), never an
        // error; when one is attached, a collector is produced.
        let result = IttFactory
            .try_create()
            .expect("try_create never errors on absence");
        if collector_attached() {
            assert!(result.is_some(), "collector expected when ITT is attached");
        } else {
            assert!(
                result.is_none(),
                "graceful skip expected when no ITT collector is attached"
            );
        }
    }

    #[test]
    fn collector_is_inert_without_a_collector() {
        // Emitting through a collector built without VTune attached must record
        // nothing observable and must not panic.
        let collector = IttCollector::new("inert-domain");
        if !collector.attached() {
            collector.emit(&event("MatMul_0", TracePhase::Begin));
            collector.emit(&event("MatMul_0", TracePhase::End));
            collector.flush().expect("flush is graceful");
        }
    }
}
