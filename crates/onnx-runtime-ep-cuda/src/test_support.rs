//! Shared helpers for this crate's GPU-gated unit tests.

use std::panic;
use std::sync::{Arc, OnceLock};

use crate::runtime::CudaRuntime;

/// Whether this host can construct a [`CudaRuntime`], probed exactly once.
///
/// `CudaRuntime::new` may *panic* rather than return `Err` when a CUDA library
/// is absent: cudarc's dynamic loader `expect()`s the shared object on first
/// use. The probe therefore runs under `catch_unwind` with the panic hook
/// silenced, so a GPU-gated test skips quietly on a host without libcuda
/// instead of spraying a panic message through the CPU-only suite.
///
/// The silencing must happen ONCE for the whole process. `set_hook` installs a
/// *process-global* hook, so the previous arrangement — every kernel test module
/// swapping the hook on every `runtime()` call — was doubly broken under the
/// default parallel harness:
///
/// * While one thread held the silencing hook, a genuine `assert!` failure on
///   another thread was swallowed, and libtest reported that test as `FAILED`
///   with an empty message and no location — leaving nothing to debug.
/// * Two overlapping `take_hook`/`set_hook` pairs lost updates: the second
///   thread could capture the *first* thread's silencing hook as its "previous"
///   and restore it, leaving panics muted for the rest of the run.
///
/// Behind a [`OnceLock`] the swap happens once instead of once per call. A
/// window remains — the one-time probe pays a CUDA context initialisation with
/// the hook silenced, and a CPU-only test panicking in exactly that window
/// would still lose its message — but it is bounded to a single startup
/// interval rather than recurring for every one of the several hundred
/// GPU-gated tests.
fn cuda_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let available = panic::catch_unwind(|| CudaRuntime::new(0).is_ok()).unwrap_or(false);
        panic::set_hook(previous);
        available
    })
}

/// A freshly constructed runtime, or `None` when this host has no usable CUDA
/// device.
///
/// Each caller gets its own [`CudaRuntime`] — tests assert on per-runtime state
/// such as pool occupancy and capture status, so they must not share one. Only
/// the availability *probe* is memoized; once it has succeeded, construction
/// needs no `catch_unwind` because the panicking path has been ruled out.
pub(crate) fn maybe_runtime() -> Option<Arc<CudaRuntime>> {
    if !cuda_available() {
        return None;
    }
    CudaRuntime::new(0).ok().map(Arc::new)
}
