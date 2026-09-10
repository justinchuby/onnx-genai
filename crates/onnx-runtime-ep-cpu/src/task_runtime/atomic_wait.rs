use std::sync::atomic::AtomicU32;

#[cfg(any(target_os = "linux", target_os = "android"))]
mod platform {
    use std::ptr;
    use std::sync::atomic::AtomicU32;

    const WAIT_PRIVATE: libc::c_int = libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG;
    const WAKE_PRIVATE: libc::c_int = libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG;

    #[inline]
    pub(super) fn wait(atomic: &AtomicU32, expected: u32) {
        // SAFETY: AtomicU32::as_ptr supplies the exact `uint32_t *` required by
        // futex while preserving the allocation's provenance. The remaining
        // variadic arguments use the widths and signedness from futex(2):
        // `int`, `uint32_t`, and `const struct timespec *`.
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                atomic.as_ptr(),
                WAIT_PRIVATE,
                expected,
                ptr::null::<libc::timespec>(),
            );
        }
    }

    #[inline]
    pub(super) fn wake_all(atomic: &AtomicU32) {
        // SAFETY: See wait. FUTEX_WAKE interprets `val` as a uint32_t waiter
        // count; i32::MAX preserves atomic-wait's "all practical waiters"
        // behavior while crossing the variadic boundary with the ABI type.
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                atomic.as_ptr(),
                WAKE_PRIVATE,
                i32::MAX as u32,
            );
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
mod platform {
    use std::sync::atomic::AtomicU32;

    #[inline]
    pub(super) fn wait(atomic: &AtomicU32, expected: u32) {
        atomic_wait::wait(atomic, expected);
    }

    #[inline]
    pub(super) fn wake_all(atomic: &AtomicU32) {
        atomic_wait::wake_all(atomic);
    }
}

/// Wait while `atomic` still equals `expected`.
///
/// The operation may return spuriously. Callers must re-check their predicate.
#[inline]
pub(crate) fn wait(atomic: &AtomicU32, expected: u32) {
    platform::wait(atomic, expected);
}

/// Wake every thread currently waiting on `atomic`.
#[inline]
pub(crate) fn wake_all(atomic: &AtomicU32) {
    platform::wake_all(atomic);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    const COMPLETION_TIMEOUT: Duration = Duration::from_secs(10);

    #[test]
    fn a_value_changed_before_wait_is_not_a_lost_wakeup() {
        let value = AtomicU32::new(0);
        value.store(1, Ordering::Release);

        wait(&value, 0);

        assert_eq!(value.load(Ordering::Acquire), 1);
    }

    #[test]
    fn a_change_racing_the_wait_is_not_lost() {
        let iterations = if cfg!(miri) { 8 } else { 256 };
        for _ in 0..iterations {
            let value = Arc::new(AtomicU32::new(0));
            let (armed_tx, armed_rx) = mpsc::channel();
            let (done_tx, done_rx) = mpsc::channel();
            let waiter_value = Arc::clone(&value);
            let waiter = thread::spawn(move || {
                armed_tx.send(()).unwrap();
                wait(&waiter_value, 0);
                assert_eq!(waiter_value.load(Ordering::Acquire), 1);
                done_tx.send(()).unwrap();
            });

            armed_rx.recv().unwrap();
            value.store(1, Ordering::Release);
            wake_all(&value);

            if done_rx.recv_timeout(COMPLETION_TIMEOUT).is_err() {
                // Release a genuinely lost waiter so the failure is reported
                // rather than leaving the test process hung.
                wake_all(&value);
                waiter.join().unwrap();
                panic!("a waiter missed both the value change and its wake");
            }
            waiter.join().unwrap();
        }
    }

    #[test]
    fn wake_all_releases_every_waiter() {
        let value = Arc::new(AtomicU32::new(0));
        let waiter_count = if cfg!(miri) { 2 } else { 8 };
        let (armed_tx, armed_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let waiters: Vec<_> = (0..waiter_count)
            .map(|_| {
                let value = Arc::clone(&value);
                let armed_tx = armed_tx.clone();
                let done_tx = done_tx.clone();
                thread::spawn(move || {
                    armed_tx.send(()).unwrap();
                    wait(&value, 0);
                    assert_eq!(value.load(Ordering::Acquire), 1);
                    done_tx.send(()).unwrap();
                })
            })
            .collect();
        drop(armed_tx);
        drop(done_tx);

        for _ in 0..waiter_count {
            armed_rx.recv().unwrap();
        }
        value.store(1, Ordering::Release);
        wake_all(&value);

        for _ in 0..waiter_count {
            if done_rx.recv_timeout(COMPLETION_TIMEOUT).is_err() {
                wake_all(&value);
                for waiter in waiters {
                    waiter.join().unwrap();
                }
                panic!("wake_all left at least one waiter asleep");
            }
        }
        for waiter in waiters {
            waiter.join().unwrap();
        }
    }
}
