#[cfg(test)]
mod group_allocation_probe {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    thread_local! {
        static ARMED: Cell<bool> = const { Cell::new(false) };
        static CALLS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    }
    fn record(free: bool) {
        if ARMED.with(Cell::get) {
            CALLS.with(|calls| {
                let (a, d) = calls.get();
                calls.set((a + usize::from(!free), d + usize::from(free)));
            });
        }
    }
    struct Counter;
    // Forward unchanged pointers/layouts; observe only constant, drop-free TLS.
    unsafe impl GlobalAlloc for Counter {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {record(false); unsafe {System.alloc(l)}}
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {record(false); unsafe {System.alloc_zeroed(l)}}
        unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {record(false); unsafe {System.realloc(p,l,n)}}
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {record(true); unsafe {System.dealloc(p,l)}}
    }
    #[global_allocator]
    static ALLOCATOR: Counter = Counter;
    pub(crate) fn observed(body: impl FnOnce()) -> (usize,usize) {
        struct Disarm;
        impl Drop for Disarm {fn drop(&mut self){ARMED.with(|armed|armed.set(false));}}
        CALLS.with(|calls|calls.set((0,0)));
        ARMED.with(|armed|armed.set(true));
        let guard=Disarm;body();drop(guard);CALLS.with(Cell::get)
    }
}
