//! Thread-local, size-specific allocation witness with no runtime instrumentation.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static SIZES: Cell<[usize; 2]> = const { Cell::new([0; 2]) };
    static WATCH: Cell<bool> = const { Cell::new(false) };
    static COUNTS: Cell<[usize; 2]> = const { Cell::new([0; 2]) };
}
struct Observer;
fn observe(bytes: usize) {
    let _ = WATCH.try_with(|watch| {
        if watch.get() {
            let _ = COUNTS.try_with(|counts| {
                let mut value = counts.get();
                let sizes = SIZES.with(Cell::get);
                value[0] += usize::from(bytes == sizes[0]);
                value[1] += usize::from(bytes == sizes[1]);
                counts.set(value);
            });
        }
    });
}
// SAFETY: every allocation operation forwards its original pointer/layout to
// System. The observer touches only constant-initialized thread-local Cells.
unsafe impl GlobalAlloc for Observer {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        observe(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        observe(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        observe(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Observer = Observer;
pub fn watched<T>(sizes: [usize; 2], run: impl FnOnce() -> T) -> (T, [usize; 2]) {
    struct Stop;
    impl Drop for Stop {
        fn drop(&mut self) {
            WATCH.with(|watch| watch.set(false));
        }
    }
    SIZES.with(|value| value.set(sizes));
    COUNTS.with(|counts| counts.set([0; 2]));
    WATCH.with(|watch| watch.set(true));
    let stop = Stop;
    let value = run();
    drop(stop);
    (value, COUNTS.with(Cell::get))
}
