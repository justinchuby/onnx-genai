//! Shared test-only helpers for the CPU execution provider.
//!
//! Compiled `#[cfg(test)]` only; nothing here ships in a production artifact.

use std::ffi::OsString;

/// RAII guard that restores every environment variable it touched on drop --
/// including when the test unwinds.
///
/// # Why this exists
///
/// `cargo test` runs tests as parallel threads inside a single process, so
/// `std::env` is shared mutable state. The pattern this type replaces was:
///
/// ```ignore
/// let previous = std::env::var("K").ok();
/// unsafe { std::env::set_var("K", "v") };
/// assert_eq!(actual, expected);          // <-- unwinds here
/// match previous {                       // <-- never runs
///     Some(v) => unsafe { std::env::set_var("K", v) },
///     None => unsafe { std::env::remove_var("K") },
/// }
/// ```
///
/// A single failing assertion leaks `K` into the rest of the process. The
/// damage is not the one failing test -- it is that every later test in the
/// same binary silently runs under a route it never asked for, so one real
/// failure is amplified into a cascade of misleading ones. Worse, the leaked
/// value can make a *passing* test vacuous: a test asserting default behaviour
/// observes the leaked override and still passes, having measured nothing.
///
/// # Locking is still the caller's job
///
/// This guard deliberately does **not** take a lock. The callers in this crate
/// already serialise on `lock_dispatch_probe()` and/or `backend_env_lock()`,
/// and those two are consistently nested in that order; folding a third lock in
/// here would either duplicate that or invert the order. Acquire the existing
/// lock(s) first, then construct the guard, so the guard -- declared last and
/// therefore dropped first -- restores the environment *while the locks are
/// still held*:
///
/// ```ignore
/// let _probe = lock_dispatch_probe();
/// let _backend = backend_env_lock().lock().unwrap();
/// let _env = EnvVarGuard::set("NXRT_CPU_GEMM_BACKEND", "mlas");
/// ```
///
/// # Scope
///
/// Only variables whose readers consult `std::env` on **every** call are
/// restorable this way. A reader that caches into a `OnceLock` (for example
/// `matmul::half_decode_gemv_enabled`, tracked by #1736) keeps the first value
/// it ever saw, so restoring the variable does not restore the behaviour. Do
/// not use this guard to imply such a variable has been made test-safe.
#[must_use = "the guard must outlive the env-dependent code; binding it to `_` drops it immediately"]
pub(crate) struct EnvVarGuard {
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl EnvVarGuard {
    /// A guard that has not touched anything yet.
    pub(crate) fn new() -> Self {
        Self { saved: Vec::new() }
    }

    /// Set `key` to `value` for the guard's lifetime.
    pub(crate) fn set(key: &'static str, value: &str) -> Self {
        let mut guard = Self::new();
        guard.set_var(key, value);
        guard
    }

    /// Ensure `key` is unset for the guard's lifetime. Use when a test asserts
    /// the behaviour of a variable's *absence*.
    pub(crate) fn unset(key: &'static str) -> Self {
        let mut guard = Self::new();
        guard.remove_var(key);
        guard
    }

    /// Set `key` to `value`, remembering the prior value so drop can restore it.
    pub(crate) fn set_var(&mut self, key: &'static str, value: &str) -> &mut Self {
        self.remember(key);
        // SAFETY: the caller holds this crate's env lock(s) for the guard's
        // lifetime, so no other test thread reads or writes the environment
        // concurrently, and the guard restores the value before releasing them.
        unsafe { std::env::set_var(key, value) };
        self
    }

    /// Remove `key`, remembering the prior value so drop can restore it.
    pub(crate) fn remove_var(&mut self, key: &'static str) -> &mut Self {
        self.remember(key);
        // SAFETY: see `set_var`.
        unsafe { std::env::remove_var(key) };
        self
    }

    /// Record the current value of `key` exactly once, so that repeated
    /// mutations of the same key still restore the value observed on entry.
    fn remember(&mut self, key: &'static str) {
        if self.saved.iter().any(|(seen, _)| *seen == key) {
            return;
        }
        self.saved.push((key, std::env::var_os(key)));
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        for (key, previous) in self.saved.drain(..) {
            match previous {
                // SAFETY: see `EnvVarGuard::set_var`; the caller's lock is held
                // until after this guard is fully dropped.
                Some(value) => unsafe { std::env::set_var(key, value) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::EnvVarGuard;
    use std::sync::{Mutex, OnceLock};

    /// Serialises this module's own tests against each other. The variables
    /// used here are private to these tests, so this lock does not need to be
    /// the crate's dispatch/backend lock.
    fn local_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    const ABSENT: &str = "NXRT_TEST_SUPPORT_ABSENT";
    const PRESENT: &str = "NXRT_TEST_SUPPORT_PRESENT";

    #[test]
    fn a_variable_that_was_absent_is_removed_again() {
        let _lock = local_env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: this module's tests are serialised by `local_env_lock`.
        unsafe { std::env::remove_var(ABSENT) };
        {
            let _env = EnvVarGuard::set(ABSENT, "1");
            assert_eq!(std::env::var(ABSENT).as_deref(), Ok("1"));
        }
        assert!(
            std::env::var_os(ABSENT).is_none(),
            "a variable absent on entry must be absent again after the guard drops"
        );
    }

    #[test]
    fn a_variable_that_was_present_keeps_its_original_value() {
        let _lock = local_env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: see above.
        unsafe { std::env::set_var(PRESENT, "original") };
        {
            let _env = EnvVarGuard::set(PRESENT, "override");
            assert_eq!(std::env::var(PRESENT).as_deref(), Ok("override"));
        }
        assert_eq!(std::env::var(PRESENT).as_deref(), Ok("original"));
        // SAFETY: see above.
        unsafe { std::env::remove_var(PRESENT) };
    }

    #[test]
    fn repeated_mutation_restores_the_value_observed_on_entry() {
        let _lock = local_env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: see above.
        unsafe { std::env::set_var(PRESENT, "entry") };
        {
            let mut env = EnvVarGuard::new();
            env.set_var(PRESENT, "first");
            env.set_var(PRESENT, "second");
            env.remove_var(PRESENT);
            assert!(std::env::var_os(PRESENT).is_none());
        }
        assert_eq!(
            std::env::var(PRESENT).as_deref(),
            Ok("entry"),
            "the value observed on entry must win, not the most recent mutation"
        );
        // SAFETY: see above.
        unsafe { std::env::remove_var(PRESENT) };
    }

    /// `unset` is the entry point for tests that assert a variable's *default*
    /// (absent) behaviour. Covered here rather than only from the
    /// `x86_64`-gated callers, so the helper is exercised -- and is not dead
    /// code -- on every architecture the crate is checked for.
    #[test]
    fn unset_hides_a_present_variable_and_puts_it_back() {
        let _lock = local_env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: this module's tests are serialised by `local_env_lock`.
        unsafe { std::env::set_var(PRESENT, "visible") };
        {
            let _env = EnvVarGuard::unset(PRESENT);
            assert!(
                std::env::var_os(PRESENT).is_none(),
                "the variable must be absent for the guard's lifetime"
            );
        }
        assert_eq!(std::env::var(PRESENT).as_deref(), Ok("visible"));
        // SAFETY: see above.
        unsafe { std::env::remove_var(PRESENT) };
    }

    /// The whole point of the type: the restore must survive an unwind. Without
    /// `Drop` this leaks the override into every later test in the binary.
    #[test]
    fn the_variable_is_restored_even_when_the_test_body_panics() {
        let _lock = local_env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: see above.
        unsafe { std::env::remove_var(ABSENT) };

        let panicked = std::panic::catch_unwind(|| {
            let _env = EnvVarGuard::set(ABSENT, "leaked");
            assert_eq!(std::env::var(ABSENT).as_deref(), Ok("leaked"));
            panic!("simulated assertion failure");
        });

        assert!(panicked.is_err(), "the closure must actually have panicked");
        assert!(
            std::env::var_os(ABSENT).is_none(),
            "an unwind must not leak the override into the rest of the process"
        );
    }
}
