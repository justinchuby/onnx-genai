//! Shared helpers for this crate's GPU-gated unit tests.

use std::ffi::OsString;
use std::panic;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use crate::runtime::CudaRuntime;

/// Process-global lock serialising every test that reads or writes an
/// environment variable that gates production behaviour.
///
/// `cargo test` runs tests as parallel threads inside a **single process**, and
/// `std::env` is process-wide mutable state. A test that calls `set_var`/
/// `remove_var` therefore mutates state that *every other thread* observes. The
/// subtle half is the reader: a test that runs an env-gated code path and
/// asserts the **default** behaviour is just as exposed — it can observe another
/// thread's temporary `set_var` and fail — even though it never touches the
/// environment itself. Both writers and default-readers must serialise on the
/// same lock for the guard to close the race structurally rather than merely
/// shrink its window.
fn env_lock() -> &'static Mutex<()> {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK.get_or_init(|| Mutex::new(()))
}

/// RAII guard that serialises environment-variable access across the crate's
/// unit tests and restores every variable it touched to its prior value on
/// drop.
///
/// Hold it for the **entire body** of any test that depends on an env-gated
/// code path — whether it mutates the variable or relies on its default. The
/// guard takes [`env_lock`] on construction and releases it on drop, so no two
/// such tests run concurrently and no mutation ever leaks past the test that
/// made it (the prior value is restored even if the test panics).
///
/// ```ignore
/// // A test that toggles a variable:
/// let mut env = EnvVarGuard::acquire();
/// env.set("MY_FLAG", "1");
/// // ... assertions ...
/// // drop restores MY_FLAG to whatever it was before.
///
/// // A test that depends on the *default* (unset) value must still lock:
/// let _env = EnvVarGuard::without_var("MY_FLAG");
/// ```
#[must_use = "the guard must outlive the env-dependent code; binding it to `_` drops it immediately"]
pub(crate) struct EnvVarGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(String, Option<OsString>)>,
}

impl EnvVarGuard {
    /// Acquire the global env lock without changing anything. Use in tests that
    /// depend on a variable's *default* value so they serialise against tests
    /// that set it.
    pub(crate) fn acquire() -> Self {
        let lock = env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Self {
            _lock: lock,
            saved: Vec::new(),
        }
    }

    /// Acquire the lock and set `key` to `value`, restoring the prior value on
    /// drop.
    pub(crate) fn with_var(key: &str, value: &str) -> Self {
        let mut guard = Self::acquire();
        guard.set(key, value);
        guard
    }

    /// Acquire the lock and ensure `key` is unset for the guard's lifetime,
    /// restoring the prior value on drop. Use when a test asserts the behaviour
    /// of a variable's absence.
    pub(crate) fn without_var(key: &str) -> Self {
        let mut guard = Self::acquire();
        guard.unset(key);
        guard
    }

    /// Set `key` to `value` while the lock is held, remembering the prior value
    /// so drop can restore it.
    pub(crate) fn set(&mut self, key: &str, value: &str) -> &mut Self {
        self.remember(key);
        // SAFETY: the process-global env lock is held for this guard's lifetime,
        // so no other test thread reads or writes the environment concurrently.
        unsafe { std::env::set_var(key, value) };
        self
    }

    /// Remove `key` while the lock is held, remembering the prior value so drop
    /// can restore it.
    pub(crate) fn unset(&mut self, key: &str) -> &mut Self {
        self.remember(key);
        // SAFETY: see `set`.
        unsafe { std::env::remove_var(key) };
        self
    }

    fn remember(&mut self, key: &str) {
        self.saved.push((key.to_string(), std::env::var_os(key)));
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // Restore in reverse so repeated touches of the same key end on the
        // earliest recorded value.
        for (key, previous) in self.saved.drain(..).rev() {
            match previous {
                // SAFETY: the lock is still held until this guard is fully dropped.
                Some(value) => unsafe { std::env::set_var(&key, value) },
                None => unsafe { std::env::remove_var(&key) },
            }
        }
    }
}

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
