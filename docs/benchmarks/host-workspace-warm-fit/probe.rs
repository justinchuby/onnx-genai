//! Warm reuse mechanism only: no payload access or inference throughput claim.
use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ep_cpu::{CpuExecutionProvider, CpuMemoryResources};
use onnx_runtime_memory_api::{DeviceKey, MemoryError, MemoryRole, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseAccounting, LeaseLedger, LedgerGovernor, MemoryAuthorityId,
    MemoryGovernor, MemoryLease, PressureRegistry, ProcessMemoryManager,
};
use std::{
    hint::black_box,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};
const ROLE: MemoryRole = MemoryRole::Workspace { step_scoped: true };
#[derive(Debug)]
struct Books {
    inner: LedgerGovernor,
    claims: AtomicUsize,
    releases: AtomicUsize,
}
#[derive(Clone)]
struct Counting(Arc<Books>);
impl Counting {
    fn new() -> Self {
        Self(Arc::new(Books {
            inner: LedgerGovernor::new(LeaseLedger::new_for_device(DeviceKey::HOST, 0, 1 << 26, 0)),
            claims: AtomicUsize::new(0),
            releases: AtomicUsize::new(0),
        }))
    }
    fn calls(&self) -> (usize, usize) {
        (
            self.0.claims.load(Ordering::Relaxed),
            self.0.releases.load(Ordering::Relaxed),
        )
    }
}
impl LeaseAccounting for Books {
    fn authority_id(&self) -> Option<MemoryAuthorityId> {
        Some(self.inner.authority_id(Tier::Host))
    }
    fn try_claim(&self, tier: Tier, bytes: u64, role: MemoryRole) -> Result<(), MemoryError> {
        self.claims.fetch_add(1, Ordering::Relaxed);
        self.inner.ledger().try_claim(tier, bytes, role)
    }
    fn release(&self, tier: Tier, bytes: u64) {
        self.releases.fetch_add(1, Ordering::Relaxed);
        self.inner.ledger().release(tier, bytes)
    }
}
impl MemoryGovernor for Counting {
    fn authority_id(&self, tier: Tier) -> MemoryAuthorityId {
        self.0.inner.authority_id(tier)
    }
    fn reserve(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<MemoryLease, MemoryError> {
        self.0.try_claim(tier, bytes, role)?;
        Ok(MemoryLease::new(tier, bytes, role, holder, self.0.clone()))
    }
    fn pressure_registry(&self, tier: Tier) -> Option<PressureRegistry> {
        self.0.inner.pressure_registry(tier)
    }
    fn available(&self, tier: Tier) -> u64 {
        self.0.inner.available(tier)
    }
    fn used(&self, tier: Tier) -> u64 {
        self.0.inner.used(tier)
    }
}
fn cpu() -> f64 {
    let mut u = std::mem::MaybeUninit::<libc::rusage>::uninit();
    assert_eq!(
        unsafe { libc::getrusage(libc::RUSAGE_SELF, u.as_mut_ptr()) },
        0
    );
    let u = unsafe { u.assume_init() };
    u.ru_utime.tv_sec as f64
        + u.ru_utime.tv_usec as f64 / 1e6
        + u.ru_stime.tv_sec as f64
        + u.ru_stime.tv_usec as f64 / 1e6
}
#[inline(never)]
fn control(reps: usize) -> u64 {
    let mut value = black_box(0x7381_5a0b_68d4_119fu64);
    for _ in 0..reps {
        value = value
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
    }
    black_box(value)
}
fn measure<F: FnMut()>(mut f: F, reps: usize) -> (Vec<f64>, Vec<f64>) {
    let mut times = Vec::new();
    let mut wall = Vec::new();
    for _ in 0..9 {
        let start = cpu();
        let instant = Instant::now();
        for _ in 0..reps {
            f();
        }
        times.push((cpu() - start) * 1000.);
        wall.push(instant.elapsed().as_secs_f64() * 1000.);
    }
    (times, wall)
}
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    assert_eq!(args.len(), 4);
    let mode = &args[0];
    let bytes = args[1].parse::<usize>().unwrap();
    let reps = args[2].parse::<usize>().unwrap();
    let control_reps = args[3].parse::<usize>().unwrap();
    assert!(reps > 0 && control_reps > 0);
    let books = Counting::new();
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, Arc::new(books.clone())).unwrap();
    // Positive counter control includes zero-byte issuer calls, outside timing.
    let before = books.calls();
    books.0.try_claim(Tier::Host, 0, ROLE).unwrap();
    books.0.release(Tier::Host, 0);
    assert_eq!(books.calls(), (before.0 + 1, before.1 + 1));
    let cpu_ep = CpuExecutionProvider::initialized_with_resources(
        CpuMemoryResources::default()
            .with_workspace_pool(pool.clone())
            .unwrap(),
    )
    .unwrap();
    let ep: &dyn ExecutionProvider = black_box(&cpu_ep);
    let first = pool.checkout(bytes, 64, ROLE).unwrap();
    let pointer = first.as_ptr();
    let identity = first.identity();
    pool.recycle_completed(first).unwrap();
    for _ in 0..1000 {
        if mode == "pool" {
            let b = pool.checkout(bytes, 64, ROLE).unwrap();
            pool.recycle_completed(b).unwrap();
        } else {
            let b = ep.allocate_workspace(bytes, 64, ROLE).unwrap();
            ep.deallocate_workspace(b).unwrap();
        }
    }
    let calls = books.calls();
    let charged = books.used(Tier::Host);
    assert_eq!(charged, bytes as u64);
    let (times, wall) = if mode == "pool" {
        measure(
            || {
                let b = pool.checkout(black_box(bytes), 64, ROLE).unwrap();
                black_box(b.as_ptr());
                pool.recycle_completed(b).unwrap();
            },
            reps,
        )
    } else {
        assert_eq!(mode, "ep");
        measure(
            || {
                let b = ep.allocate_workspace(black_box(bytes), 64, ROLE).unwrap();
                black_box(b.as_ptr());
                ep.deallocate_workspace(b).unwrap();
            },
            reps,
        )
    };
    assert_eq!(
        books.calls(),
        calls,
        "warm fit must not call any issuer, including zero bytes"
    );
    assert_eq!(books.used(Tier::Host), charged);
    let last = pool.checkout(bytes, 64, ROLE).unwrap();
    assert_eq!(last.as_ptr(), pointer);
    assert_eq!(last.identity(), identity);
    pool.recycle_completed(last).unwrap();
    let mut controls = Vec::new();
    let mut result = 0;
    for _ in 0..9 {
        let start = cpu();
        result = black_box(control(control_reps));
        controls.push((cpu() - start) * 1000.);
    }
    pool.trim_idle(u64::MAX).unwrap();
    assert_eq!(books.used(Tier::Host), 0);
    println!(
        "{{\"mode\":{mode:?},\"bytes\":{bytes},\"reps\":{reps},\"control_reps\":{control_reps},\"cpu_batch_ms\":{times:?},\"wall_batch_ms\":{wall:?},\"control_batch_ms\":{controls:?},\"control_result\":{result},\"warm_issuer_calls\":0,\"funded_bytes\":{charged},\"same_owner\":true}}"
    );
}
