use onnx_runtime_memory_api::{MemoryError, MemoryRole, Tier};
use onnx_runtime_memory_governor::{
    HolderId, LeaseAccounting, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryLease,
    PressureRegistration, PressureResponder,
};
use std::error::Error;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

struct Payload {
    _bytes: Vec<u8>,
    _lease: MemoryLease,
}
#[derive(Clone, Copy)]
enum Release {
    Actual,
    Underreport,
    Overreport,
    Error,
    Reenter,
}
struct Cache {
    governor: LedgerGovernor,
    payload: Mutex<Option<Payload>>,
    calls: AtomicUsize,
    want: AtomicU64,
    release: Release,
}
impl PressureResponder for Cache {
    fn on_pressure(&self, tier: Tier, want: u64) -> Result<u64, MemoryError> {
        assert_eq!(tier, Tier::Host);
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.want.store(want, Ordering::Relaxed);
        // Both accounting and registration are safely reentrant. A callback
        // cannot be invoked under either mutex.
        let _probe = self
            .governor
            .reserve(tier, 0, MemoryRole::Activation, HolderId::new(99))?;
        let registry = self.governor.pressure_registry(tier).unwrap();
        let dummy: Arc<dyn PressureResponder> = Arc::new(Noop);
        drop(registry.register(&dummy)?);
        match self.release {
            Release::Overreport => return Ok(want),
            Release::Error => {
                return Err(MemoryError::ReclaimFailed {
                    tier: tier.name(),
                    requested: want,
                    source: Box::new(std::io::Error::other(
                        "fixture cache release failed; restore its allocator",
                    )),
                });
            }
            Release::Reenter => {
                assert!(matches!(
                    self.governor
                        .reserve(tier, 1, MemoryRole::Activation, HolderId::new(99)),
                    Err(MemoryError::TierExhausted { .. })
                ));
                return Ok(0);
            }
            _ => {}
        }
        let retired = self.payload.lock().unwrap().take();
        let bytes = retired
            .as_ref()
            .map_or(0, |payload| payload._bytes.len() as u64);
        drop(retired);
        Ok(if matches!(self.release, Release::Underreport) {
            0
        } else {
            bytes
        })
    }
}
struct Noop;
impl PressureResponder for Noop {
    fn on_pressure(&self, _tier: Tier, _want: u64) -> Result<u64, MemoryError> {
        Ok(0)
    }
}
fn cache(
    governor: &LedgerGovernor,
    bytes: u64,
    release: Release,
) -> (Arc<Cache>, PressureRegistration) {
    let lease = governor
        .reserve(Tier::Host, bytes, MemoryRole::Weights, HolderId::new(1))
        .unwrap();
    let cache = Arc::new(Cache {
        governor: governor.clone(),
        payload: Mutex::new(Some(Payload {
            _bytes: vec![1; bytes as usize],
            _lease: lease,
        })),
        calls: AtomicUsize::new(0),
        want: AtomicU64::new(0),
        release,
    });
    let responder: Arc<dyn PressureResponder> = cache.clone();
    let registration = governor
        .pressure_registry(Tier::Host)
        .unwrap()
        .register(&responder)
        .unwrap();
    (cache, registration)
}
#[test]
fn lease_pressure_reserve_grow_and_sibling_share_shortfall_admission() {
    for path in ["reserve", "grow", "sibling"] {
        let governor = LedgerGovernor::new(LeaseLedger::new(0, 32, 0));
        let (cache, _registration) = cache(&governor, 16, Release::Actual);
        let mut kv = governor
            .reserve(Tier::Host, 8, MemoryRole::KvCache, HolderId::new(2))
            .unwrap();
        assert!(matches!(
            governor.reserve(Tier::Host, 33, MemoryRole::KvCache, HolderId::new(2)),
            Err(MemoryError::TierExhausted { .. })
        ));
        assert!(matches!(
            governor.reserve(Tier::Host, u64::MAX, MemoryRole::KvCache, HolderId::new(2)),
            Err(MemoryError::InvalidRequest { .. })
        ));
        assert_eq!(
            cache.calls.load(Ordering::Relaxed),
            0,
            "impossible requests cannot flush an idle cache"
        );
        let held = match path {
            "reserve" => Some(
                governor
                    .reserve(Tier::Host, 16, MemoryRole::KvCache, HolderId::new(2))
                    .unwrap(),
            ),
            "grow" => {
                kv.grow(16).unwrap();
                None
            }
            "sibling" => Some(kv.reserve_sibling(16).unwrap()),
            _ => unreachable!(),
        };
        assert_eq!(cache.calls.load(Ordering::Relaxed), 1);
        assert_eq!(cache.want.load(Ordering::Relaxed), 8);
        assert!(cache.payload.lock().unwrap().is_none());
        assert_eq!(governor.used(Tier::Host), 24);
        drop(held);
        drop(kv);
        assert_eq!(governor.used(Tier::Host), 0);
    }
}
#[test]
fn lease_pressure_uses_actual_books_and_preserves_callback_errors() {
    for release in [Release::Underreport, Release::Overreport, Release::Error] {
        let governor = LedgerGovernor::new(LeaseLedger::new(0, 16, 0));
        let (cache, _registration) = cache(&governor, 16, release);
        let result = governor.reserve(Tier::Host, 16, MemoryRole::KvCache, HolderId::new(2));
        match release {
            Release::Underreport => {
                drop(result.unwrap());
                assert_eq!(governor.used(Tier::Host), 0);
            }
            Release::Overreport => {
                assert!(matches!(result, Err(MemoryError::TierExhausted { .. })));
                assert_eq!(governor.used(Tier::Host), 16);
            }
            Release::Error => {
                let error = result.unwrap_err();
                assert!(matches!(
                    error,
                    MemoryError::ReclaimFailed { requested: 16, .. }
                ));
                assert!(
                    error
                        .source()
                        .unwrap()
                        .downcast_ref::<std::io::Error>()
                        .is_some()
                );
                assert!(error.to_string().contains("restore its allocator"));
                assert_eq!(governor.used(Tier::Host), 16);
            }
            _ => unreachable!(),
        }
        assert_eq!(
            cache.calls.load(Ordering::Relaxed),
            1,
            "pressure is a bounded pass even when the holder releases nothing"
        );
        drop(cache);
        assert_eq!(governor.used(Tier::Host), 0);
    }
}
#[test]
fn lease_pressure_reentry_is_bounded_and_later_rounds_remain_enabled() {
    let governor = LedgerGovernor::new(LeaseLedger::new(0, 16, 0));
    let (cache, _registration) = cache(&governor, 16, Release::Reenter);
    for round in 1..=3 {
        assert!(matches!(
            governor.reserve(Tier::Host, 1, MemoryRole::KvCache, HolderId::new(2)),
            Err(MemoryError::TierExhausted { .. })
        ));
        assert_eq!(cache.calls.load(Ordering::Relaxed), round);
    }
    assert_eq!(governor.used(Tier::Host), 16);
}
#[test]
fn lease_pressure_registration_is_weak_cancellable_and_authority_scoped() {
    let governor = LedgerGovernor::new(LeaseLedger::new(16, 16, 0));
    let (cache, registration) = cache(&governor, 16, Release::Actual);
    let registry = governor.pressure_registry(Tier::Host).unwrap();
    let foreign = LedgerGovernor::new(LeaseLedger::new(0, 16, 0));
    assert!(
        registry
            .validate(foreign.authority_id(Tier::Host), Tier::Host)
            .is_err()
    );
    assert!(
        registry
            .validate(governor.authority_id(Tier::Host), Tier::Device)
            .is_err()
    );
    drop(registration);
    assert!(
        governor
            .reserve(Tier::Host, 1, MemoryRole::KvCache, HolderId::new(2))
            .is_err()
    );
    assert_eq!(cache.calls.load(Ordering::Relaxed), 0);
    let weak = Arc::downgrade(&cache);
    drop(cache);
    assert!(weak.upgrade().is_none());
    assert_eq!(governor.used(Tier::Host), 0);
    assert_eq!(registry.reclaim(16).unwrap(), 0);
}
#[derive(Debug, Default)]
struct PermissiveBooks(AtomicUsize);
impl LeaseAccounting for PermissiveBooks {
    fn try_claim(&self, _tier: Tier, _bytes: u64, _role: MemoryRole) -> Result<(), MemoryError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn release(&self, _tier: Tier, _bytes: u64) {}
}
#[test]
fn lease_pressure_growth_overflow_is_rejected_before_the_accountant() {
    let books = Arc::new(PermissiveBooks::default());
    let mut lease = MemoryLease::new(
        Tier::Host,
        u64::MAX,
        MemoryRole::KvCache,
        HolderId::new(2),
        books.clone(),
    );
    assert!(matches!(
        lease.grow(1),
        Err(MemoryError::InvalidRequest { .. })
    ));
    assert_eq!(lease.bytes(), u64::MAX);
    assert_eq!(books.0.load(Ordering::Relaxed), 0);
}
