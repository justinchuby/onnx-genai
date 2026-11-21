use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use onnx_runtime_memory_api::{MemoryError, MemoryRole, Tier};
use onnx_runtime_memory_governor::{
    HolderId, LeaseAccounting, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryLease,
};

#[derive(Debug)]
struct ObservedBooks {
    used: AtomicU64,
    claims: AtomicUsize,
    returned: AtomicU64,
}

impl LeaseAccounting for ObservedBooks {
    fn release(&self, _tier: Tier, bytes: u64) {
        self.used.fetch_sub(bytes, Ordering::SeqCst);
        self.returned.fetch_add(bytes, Ordering::SeqCst);
    }

    fn try_claim(&self, _tier: Tier, bytes: u64, _role: MemoryRole) -> Result<(), MemoryError> {
        self.claims.fetch_add(1, Ordering::SeqCst);
        self.used.fetch_add(bytes, Ordering::SeqCst);
        Ok(())
    }
}

fn observed_lease() -> (MemoryLease, Arc<ObservedBooks>) {
    let books = Arc::new(ObservedBooks {
        used: AtomicU64::new(96),
        claims: AtomicUsize::new(0),
        returned: AtomicU64::new(0),
    });
    let lease = MemoryLease::new(
        Tier::Host,
        96,
        MemoryRole::Weights,
        HolderId::new(77),
        books.clone(),
    );
    (lease, books)
}

#[test]
fn group_partition_transfers_ownership_without_accounting_callbacks() {
    let (mut group, books) = observed_lease();
    let first = group.split_off(32).unwrap();
    let second = group.split_off(16).unwrap();
    assert_eq!(group.bytes(), 48);
    for child in [&first, &second] {
        assert_eq!(child.tier(), group.tier());
        assert_eq!(child.role(), group.role());
        assert_eq!(child.holder(), group.holder());
        assert_eq!(child.authority_id(), group.authority_id());
    }
    assert_eq!(books.claims.load(Ordering::SeqCst), 0);
    assert_eq!(books.returned.load(Ordering::SeqCst), 0);
    drop(group);
    assert_eq!(books.used.load(Ordering::SeqCst), 48);
    drop(first);
    assert_eq!(books.used.load(Ordering::SeqCst), 16);
    drop(second);
    assert_eq!(books.used.load(Ordering::SeqCst), 0);
    assert_eq!(books.returned.load(Ordering::SeqCst), 96);
}

#[test]
fn invalid_zero_and_whole_splits_never_lose_or_duplicate_a_grant() {
    let (mut group, books) = observed_lease();
    let failure = group.split_off(97).unwrap_err();
    assert!(matches!(
        failure,
        MemoryError::InvalidRequest {
            requested: 97,
            tier: "host",
            ..
        }
    ));
    assert!(failure.to_string().contains("MemoryLease::bytes()"));
    assert_eq!(group.bytes(), 96);
    drop(group.split_off(0).unwrap());
    assert_eq!(books.returned.load(Ordering::SeqCst), 0);
    let whole = group.split_off(96).unwrap();
    drop(group);
    assert_eq!(books.used.load(Ordering::SeqCst), 96);
    drop(whole);
    assert_eq!(books.used.load(Ordering::SeqCst), 0);
    assert_eq!(books.claims.load(Ordering::SeqCst), 0);
}

#[test]
fn split_and_absorb_preserve_real_ledger_headroom() {
    let governor = LedgerGovernor::new(LeaseLedger::new(0, 100, 0));
    let mut group = governor
        .reserve(Tier::Host, 100, MemoryRole::Weights, HolderId::new(77))
        .unwrap();
    let child = group.split_off(80).unwrap();
    assert_eq!(governor.available(Tier::Host), 0);
    group.absorb_sibling(child).unwrap();
    assert_eq!(group.bytes(), 100);
    assert_eq!(governor.used(Tier::Host), 100);
    let child = group.split_off(32).unwrap();
    assert!(group.reserve_sibling(1).is_err());
    drop(child);
    assert_eq!(governor.available(Tier::Host), 32);
    let other = group.reserve_sibling(32).unwrap();
    assert_eq!(governor.used(Tier::Host), 100);
    drop(group);
    assert_eq!(governor.used(Tier::Host), 32);
    drop(other);
    assert_eq!(governor.used(Tier::Host), 0);
}
