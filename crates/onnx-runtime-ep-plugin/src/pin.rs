//! Keep this plugin library mapped for the life of the process.
//!
//! # The hazard
//!
//! ONNX Runtime loads a plugin EP with `dlopen`/`LoadLibrary` from
//! `RegisterExecutionProviderLibrary`, and drops that reference from
//! `UnregisterExecutionProviderLibrary`. When the reference count reaches zero
//! the loader is free to **unmap the library's text**.
//!
//! That is only safe if nothing outside the library still points into it. A
//! plugin EP does not meet that bar, and cannot be made to:
//!
//! * [`crate::status::set_host_api`] caches the host `OrtApi` in a
//!   process-global — that global lives *inside* this library, so unmapping it
//!   invalidates state ORT may still reach through a stale factory pointer;
//! * every `thread_local!` with drop glue that a kernel touches registers a
//!   destructor **whose function pointer is in this library's text**, against
//!   whichever thread ran the kernel — including ONNX Runtime's own intra-op
//!   threads, which this library neither owns nor can join;
//! * coverage- and PGO-instrumented builds register a profile writer with the
//!   process at load time and run it at process exit, from this library's text.
//!
//! None of those registrations are withdrawn when the library is unmapped, so
//! each is a function pointer into a hole. The failure is a hard
//! `STATUS_ACCESS_VIOLATION` / `SIGSEGV` at teardown, after every test has
//! already reported success — see #1672, #983.
//!
//! # Why this is not already a problem on Linux
//!
//! It is, but glibc hides it. Measured on this repo (see
//! `plugin_survives_unregister.rs`): a bare register → unregister with no
//! session unmaps the cdylib outright (4 mapping entries → 0), while the same
//! sequence *after* a real `Run` leaves all 4 entries in place, because running
//! a kernel pins the DSO. Whether the window is open therefore depends on
//! whether a kernel happened to run first — an accident of glibc's loader, not
//! a property this library establishes. Windows `FreeLibrary` grants no such
//! reprieve.
//!
//! # What this does instead
//!
//! Take one extra, never-released reference to our own module the first time
//! ORT calls in. The library then stays mapped even though ORT's own reference
//! is dropped at unregister, so every pointer above stays valid until the
//! process ends.
//!
//! # What it costs
//!
//! The library's mapping (a few MB of mostly-file-backed pages) is not returned
//! on unregister. In exchange, register → unregister → register of a *different
//! build at the same path* keeps serving the first build. Both are the standard
//! trade for a plugin that publishes callbacks into a process, and the EP is a
//! process singleton in practice: `HOST_ORT_API` is already a process-global,
//! so a second, different build in one process was never supported anyway.

use std::sync::OnceLock;

/// Outcome of the one pin attempt this process makes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PinOutcome {
    /// The extra reference was taken; the library will not be unmapped.
    Pinned,
    /// This code is running from the main executable, not from a dynamically
    /// loaded object. There is no loader reference to drop and nothing to
    /// unmap, so the hazard does not exist and there is nothing to pin. This is
    /// the normal answer in unit and integration tests, which link the plugin
    /// crates statically; it is **not** an expected answer under ORT.
    NotDynamicallyLoaded,
    /// The platform call failed. The library is still usable, but an
    /// unregister may unmap it — the hazard above is live.
    Failed,
    /// No implementation for this target; the hazard is unaddressed.
    Unsupported,
}

static PIN: OnceLock<PinOutcome> = OnceLock::new();

/// Pin this library into the process, once.
///
/// Idempotent and cheap after the first call. Best-effort: a failure is
/// reported through [`pin_outcome`] and to stderr rather than failing plugin
/// load, because an unmap hazard at teardown is strictly better than an EP that
/// refuses to load.
///
/// # Why [`PinOutcome::NotDynamicallyLoaded`] is silent
///
/// It is the correct and expected answer for every test binary that links these
/// crates statically and calls `create_ep_factories*` directly, which is most of
/// the plugin test suite. Warning there would train the reader to ignore the
/// warning. The cost is that a *misdetection* under ORT would skip the pin
/// quietly, so the production property is not left to this function's own
/// reporting: `the_plugin_library_survives_unregister` asserts it end to end
/// through real ONNX Runtime, from `/proc/self/maps`.
pub fn pin_plugin_library() -> PinOutcome {
    *PIN.get_or_init(|| {
        let outcome = pin_once();
        if matches!(outcome, PinOutcome::Failed | PinOutcome::Unsupported) {
            eprintln!(
                "nxrt ep plugin: could not pin the plugin library into the process \
                 ({outcome:?}). Unregistering the EP library may unmap it while \
                 thread-local destructors and instrumentation callbacks still point \
                 into it; see crates/onnx-runtime-ep-plugin/src/pin.rs."
            );
        }
        outcome
    })
}

/// The outcome of this process's pin attempt, or `None` if nothing has called
/// [`pin_plugin_library`] yet.
pub fn pin_outcome() -> Option<PinOutcome> {
    PIN.get().copied()
}

/// An address that is guaranteed to lie inside *this* library's text, used to
/// ask the platform loader which module we are. It must be a real, non-generic
/// function defined in this crate: a generic or `#[inline]` function can be
/// instantiated into a different codegen unit or crate.
#[inline(never)]
extern "C" fn address_anchor() {}

#[cfg(unix)]
fn pin_once() -> PinOutcome {
    use std::ffi::CStr;

    // SAFETY: `info` is written only if `dladdr` returns non-zero, and
    // `address_anchor` is a real function address in this object.
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    let found = unsafe { libc::dladdr(address_anchor as *const libc::c_void, &mut info) };
    if found == 0 || info.dli_fname.is_null() {
        return PinOutcome::Failed;
    }

    // `RTLD_NOLOAD` refuses to load anything new: it either finds the object
    // already mapped -- which it must be, we are running from it -- or fails.
    // The `RTLD_NODELETE` on this handle marks the object un-unloadable, and
    // the handle is deliberately never passed to `dlclose`, so the reference
    // count never returns to zero either. Both alone would be enough; taking
    // both means a platform that ignores one is still covered.
    //
    // `RTLD_LAZY` is not optional padding: POSIX requires exactly one of
    // `RTLD_LAZY`/`RTLD_NOW` in every mode, and glibc rejects a mode without
    // one -- `RTLD_NOLOAD | RTLD_NODELETE` alone returns null with "invalid
    // mode for dlopen()". It binds nothing here, since `RTLD_NOLOAD` resolves
    // an object that is already fully relocated.
    let name: &CStr = unsafe { CStr::from_ptr(info.dli_fname) };

    // A statically linked test binary reaches here too, and `dlopen` of the
    // main executable fails on glibc -- which would report `Failed` for a
    // process that has nothing to pin. Distinguish the two before calling, so a
    // real failure in the cdylib is never masked by a benign one in a test.
    if same_file(name, std::env::current_exe().ok().as_deref()) {
        return PinOutcome::NotDynamicallyLoaded;
    }

    let handle = unsafe {
        libc::dlopen(
            name.as_ptr(),
            libc::RTLD_NOLOAD | libc::RTLD_NODELETE | libc::RTLD_LAZY,
        )
    };
    if handle.is_null() {
        let err = unsafe { libc::dlerror() };
        if !err.is_null() {
            eprintln!(
                "nxrt ep plugin: dlopen(RTLD_NOLOAD) on {} failed: {}",
                name.to_string_lossy(),
                unsafe { CStr::from_ptr(err) }.to_string_lossy()
            );
        }
        return PinOutcome::Failed;
    }
    PinOutcome::Pinned
}

/// Whether `name` and `other` name the same file on disk, comparing canonical
/// paths so a relative `dli_fname` or a symlinked target still matches.
#[cfg(unix)]
fn same_file(name: &std::ffi::CStr, other: Option<&std::path::Path>) -> bool {
    use std::os::unix::ffi::OsStrExt;

    let Some(other) = other else {
        return false;
    };
    let this = std::path::Path::new(std::ffi::OsStr::from_bytes(name.to_bytes()));
    match (this.canonicalize(), other.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => this == other,
    }
}

#[cfg(windows)]
fn pin_once() -> PinOutcome {
    use windows_sys::Win32::Foundation::HMODULE;
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_PIN, GetModuleHandleExW,
    };

    // No executable check on this side: `GetModuleHandleExW` with
    // `FROM_ADDRESS` resolves an address in a statically linked test binary to
    // the `.exe` itself and pinning that succeeds, which is accurate -- the
    // main image is never unmapped. So `Pinned` is the honest answer for both
    // shapes here, and unlike `dlopen` there is no benign failure to mask.
    let mut module: HMODULE = std::ptr::null_mut();
    // With `FROM_ADDRESS` the second parameter is an address, not a string;
    // `GET_MODULE_HANDLE_EX_FLAG_PIN` makes the reference permanent, which is
    // exactly "never unmap this", without leaving a handle to leak.
    let ok = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            address_anchor as *const u16,
            &mut module,
        )
    };
    if ok == 0 || module.is_null() {
        return PinOutcome::Failed;
    }
    PinOutcome::Pinned
}

#[cfg(not(any(unix, windows)))]
fn pin_once() -> PinOutcome {
    PinOutcome::Unsupported
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pin never reports a hard failure, and answers the same way twice.
    ///
    /// These tests link the plugin crate into an executable, so the strongest
    /// available assertion here is "not `Failed`" -- the executable arm is a
    /// legitimate `NotDynamicallyLoaded`. The load-bearing assertion, that a
    /// real cdylib survives `UnregisterExecutionProviderLibrary`, is
    /// `the_plugin_library_survives_unregister` in
    /// `onnx-runtime-ep-cpu-plugin/tests/plugin_survives_unregister.rs`, which
    /// goes through ONNX Runtime and reads `/proc/self/maps`.
    #[test]
    fn pinning_never_fails_and_is_idempotent() {
        let first = pin_plugin_library();
        assert!(
            matches!(first, PinOutcome::Pinned | PinOutcome::NotDynamicallyLoaded),
            "pin attempt reported {first:?}"
        );
        assert_eq!(pin_plugin_library(), first, "pin is not idempotent");
        assert_eq!(pin_outcome(), Some(first));
    }

    /// The executable arm must be reached by *detection*, not by a failed
    /// `dlopen` that happens to look benign. If `same_file` stopped matching,
    /// this process would report `Failed` and the distinction would be lost.
    #[cfg(unix)]
    #[test]
    fn a_statically_linked_binary_is_recognised_rather_than_failing() {
        assert_eq!(
            pin_once(),
            PinOutcome::NotDynamicallyLoaded,
            "the unit-test executable should be detected as not dynamically loaded"
        );
    }

    /// `address_anchor` must resolve to this object rather than to a shared
    /// runtime. Without this, the pin could silently pin libc.
    #[cfg(unix)]
    #[test]
    fn the_anchor_address_resolves_to_this_object() {
        let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
        let found = unsafe { libc::dladdr(address_anchor as *const libc::c_void, &mut info) };
        assert_ne!(found, 0, "dladdr could not attribute the anchor address");
        assert!(!info.dli_fname.is_null(), "dladdr returned no object name");
        let name = unsafe { std::ffi::CStr::from_ptr(info.dli_fname) }.to_string_lossy();
        assert!(
            !name.contains("libc.so") && !name.contains("ld-linux"),
            "the anchor resolved to {name}, not to the plugin object"
        );
    }
}
