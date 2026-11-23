//! Count allocation and free calls in the real CPU workspace borrow/return
//! boundary. Cold domain/backing setup is outside the observed warm loop.
use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ep_cpu::{CpuExecutionProvider, CpuMemoryResources, DerivedWeightMemory};
use onnx_runtime_memory_api::{DeviceKey, MemoryRole, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

thread_local! {
    // Constant, drop-free TLS cannot allocate recursively in the allocator.
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<(usize, usize)> = const { Cell::new((0,0)) };
}
fn record(free: bool) {
    if ARMED.with(Cell::get) {
        CALLS.with(|calls| {
            let (allocations, frees) = calls.get();
            calls.set((allocations + usize::from(!free), frees + usize::from(free)));
        });
    }
}
struct Counter;
// SAFETY: System receives the original pointers/layouts unchanged; observation
// accesses only constant, allocation-free TLS. Other test threads are unarmed.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(false);
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(false);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(false);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record(true);
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;
fn observed(body: impl FnOnce()) -> (usize, usize) {
    struct Disarm;
    impl Drop for Disarm {
        fn drop(&mut self) {
            ARMED.with(|armed| armed.set(false));
        }
    }
    CALLS.with(|calls| calls.set((0, 0)));
    ARMED.with(|armed| armed.set(true));
    let guard = Disarm;
    body();
    drop(guard);
    CALLS.with(Cell::get)
}

#[test]
fn warm_cpu_workspace_borrow_return_preserves_owner_without_heap_calls() {
    let positive = observed(|| drop(std::hint::black_box(Box::new([3u8; 1024]))));
    assert!(
        positive.0 > 0 && positive.1 > 0,
        "both allocation and free probes must observe their positive control: {positive:?}"
    );
    let books = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
        DeviceKey::HOST,
        0,
        64,
        0,
    )));
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
    let resources = CpuMemoryResources::new(
        DerivedWeightMemory::governed(books.as_ref(), HolderId::new(13)).unwrap(),
        Default::default(),
    )
    .unwrap()
    .with_workspace_pool(pool.clone())
    .unwrap();
    let cpu = CpuExecutionProvider::initialized_with_resources(resources).unwrap();
    let role = MemoryRole::Workspace { step_scoped: true };
    let allocation = cpu.allocate_workspace(64, 16, role).unwrap();
    let wrapper = std::ptr::from_ref(allocation.managed_owner().unwrap());
    let identity = allocation.managed_owner().unwrap().identity();
    cpu.deallocate_workspace(allocation).unwrap();
    let mut same_wrapper = true;
    let mut same_identity = true;
    let counts = observed(|| {
        for iteration in 0..4096 {
            let bytes = if iteration % 2 == 0 { 64 } else { 16 };
            let allocation = cpu.allocate_workspace(bytes, 16, role).unwrap();
            same_wrapper &= std::ptr::from_ref(allocation.managed_owner().unwrap()) == wrapper;
            same_identity &= allocation.managed_owner().unwrap().identity() == identity;
            cpu.deallocate_workspace(allocation).unwrap();
        }
    });
    assert_eq!(books.used(Tier::Host), 64);
    assert_eq!(manager.process_used(Tier::Host), 64);
    drop(cpu);
    pool.trim_idle(64).unwrap();
    assert_eq!(books.used(Tier::Host), 0);
    assert_eq!(
        counts,
        (0, 0),
        "warm CPU lane ownership must not allocate/free wrappers or backing; preserve the existing managed Box across the EP bridge"
    );
    assert!(
        same_wrapper,
        "all warm loans must retain the original ownership box"
    );
    assert!(
        same_identity,
        "all warm loans must retain their original physical backing"
    );
}
