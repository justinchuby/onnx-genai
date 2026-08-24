//! The plugin cdylib must still be mapped after ORT unregisters it.
//!
//! # What this pins down
//!
//! `RegisterExecutionProviderLibrary` `dlopen`s this crate's cdylib and
//! `UnregisterExecutionProviderLibrary` drops that reference. Without a
//! deliberate extra reference the loader unmaps the library's text while
//! thread-local destructors, the cached host `OrtApi`, and (under
//! instrumentation) the profile writer still point into it —
//! `onnx-runtime-ep-plugin/src/pin.rs` has the full hazard. This test asserts
//! the extra reference is really taken, end to end, through real ONNX Runtime.
//!
//! # Why there is no `Run` in it, and why adding one would make it vacuous
//!
//! Measured on this repo before the pin existed, three runs each, same host:
//!
//! | sequence                                | mapping entries after unregister |
//! |-----------------------------------------|----------------------------------|
//! | register → unregister                   | 0 (unmapped)                     |
//! | register → session → Run → unregister   | 4 (retained)                     |
//!
//! Running a kernel pins the DSO on glibc as a side effect — a `thread_local!`
//! with drop glue registers a destructor against the object, and glibc will not
//! unload an object with pending TLS destructors. So a version of this test
//! that ran a model would pass whether or not [`pin_plugin_library`] exists: it
//! would assert glibc's accident, not our property. The no-`Run` sequence is
//! the *only* arm on Linux where the pin is load-bearing, which is exactly why
//! it is the arm under test.
//!
//! Verified by mutation: with the `pin_plugin_library()` call removed from
//! `factory::init_host_api`, this test fails with 0 mapping entries (3/3 runs);
//! with it restored it passes with 4 (3/3).
//!
//! # Platform
//!
//! Linux-only, because it reads `/proc/self/maps`. The Windows arm of the pin
//! is asserted by `onnx_runtime_ep_plugin::pin`'s unit tests and, indirectly,
//! by the `Rust coverage (Windows x86_64)` lane, whose `0xC0000005` at process
//! exit (#1672, #983) is what motivated this work.
//!
//! [`pin_plugin_library`]: onnx_runtime_ep_plugin::pin::pin_plugin_library

#![cfg(target_os = "linux")]

use std::ffi::CString;
use std::ptr;

use onnx_genai_ort_sys as ort;
use onnx_runtime_ort_testkit as ort_path;

mod cdylib_resolve;

/// Number of `/proc/self/maps` entries backed by the file named `needle`.
fn mapping_entries(needle: &str) -> usize {
    std::fs::read_to_string("/proc/self/maps")
        .expect("read /proc/self/maps")
        .lines()
        .filter(|line| line.contains(needle))
        .count()
}

/// Skip (or, under `NXRT_REQUIRE_ORT_TESTS=1`, fail) when a resource is absent.
macro_rules! skip_if_missing {
    ($resource:expr, $what:literal) => {
        match $resource {
            Some(v) => v,
            None => {
                assert!(
                    std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() != Ok("1"),
                    concat!("NXRT_REQUIRE_ORT_TESTS=1 but ", $what, " is unavailable")
                );
                eprintln!(concat!("*** SKIPPED: ", $what, " not found ***"));
                return;
            }
        }
    };
}

/// # Safety
/// `lib` must be a loaded `libonnxruntime`.
unsafe fn get_ort_api(lib: &libloading::Library) -> *const ort::OrtApi {
    type GetApiBaseFn = unsafe extern "C" fn() -> *const ort::OrtApiBase;
    let get_api_base: libloading::Symbol<'_, GetApiBaseFn> =
        unsafe { lib.get(b"OrtGetApiBase") }.expect("OrtGetApiBase not found in libonnxruntime");
    let api_base = unsafe { get_api_base() };
    assert!(!api_base.is_null(), "OrtGetApiBase returned null");
    let get_api = unsafe { (*api_base).GetApi }.expect("OrtApiBase::GetApi is null");
    let api = unsafe { get_api(ort::ORT_API_VERSION) };
    assert!(
        !api.is_null(),
        "GetApi returned null — ORT version mismatch?"
    );
    api
}

#[test]
fn the_plugin_library_survives_unregister() {
    let ort_lib_dir = skip_if_missing!(ort_path::find_ort_lib_dir(), "ORT");
    let ep_lib_path = skip_if_missing!(
        cdylib_resolve::find_cpu_plugin_cdylib_optional(),
        "the EP cdylib (run `cargo build -p onnx-runtime-ep-cpu-plugin`)"
    );
    let needle = ep_lib_path
        .file_name()
        .expect("cdylib path has a file name")
        .to_string_lossy()
        .into_owned();

    // Instrument check, before any claim rests on it: the reader must answer 0
    // for a file that is certainly not mapped, and non-zero for one that
    // certainly is. Without this, "still mapped" could be a parser that always
    // finds something, and "not yet mapped" a parser that never does.
    assert_eq!(
        mapping_entries("this-file-is-not-mapped-into-any-process"),
        0,
        "the /proc/self/maps reader reports entries for a file that cannot be mapped"
    );
    let ort_lib_name = ort_path::ort_lib_name();
    let ort_lib_path = ort_lib_dir.join(ort_lib_name);
    let lib = unsafe { libloading::Library::new(&ort_lib_path) }
        .unwrap_or_else(|e| panic!("dlopen {}: {e}", ort_lib_path.display()));
    assert!(
        mapping_entries(ort_lib_name) > 0,
        "the /proc/self/maps reader found no entries for {ort_lib_name}, which was just loaded"
    );

    // Control: this test binary must not already have the cdylib mapped, or
    // "still mapped at the end" would say nothing about the pin. This is why
    // the test owns its whole binary.
    assert_eq!(
        mapping_entries(&needle),
        0,
        "{needle} was already mapped before ORT registered it; another test in \
         this binary loaded it and this test can no longer attribute the mapping"
    );

    let api = unsafe { get_ort_api(&lib) };

    unsafe {
        let mut env: *mut ort::OrtEnv = ptr::null_mut();
        let logid = CString::new("nxrt_pin").unwrap();
        let status =
            ((*api).CreateEnv.unwrap())(ort::ORT_LOGGING_LEVEL_WARNING, logid.as_ptr(), &mut env);
        assert!(status.is_null(), "CreateEnv failed");

        let reg_name = CString::new("cpu_ep_pin").unwrap();
        let ep_path_c = ort_path::OrtPathBuf::new(&ep_lib_path);
        let status = ((*api).RegisterExecutionProviderLibrary.unwrap())(
            env,
            reg_name.as_ptr(),
            ep_path_c.as_ptr(),
        );
        assert!(status.is_null(), "RegisterExecutionProviderLibrary failed");

        let while_registered = mapping_entries(&needle);
        assert!(
            while_registered > 0,
            "{needle} is not mapped even while registered — the test is not \
             observing the library ORT loaded"
        );

        ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_name.as_ptr());
        let after_unregister = mapping_entries(&needle);

        ((*api).ReleaseEnv.unwrap())(env);
        let after_release_env = mapping_entries(&needle);

        assert_eq!(
            after_unregister, while_registered,
            "UnregisterExecutionProviderLibrary unmapped {needle} ({while_registered} \
             mapping entries -> {after_unregister}). Thread-local destructors, the \
             cached host OrtApi and any instrumentation callback registered by this \
             library now point into unmapped text. Is pin_plugin_library() still \
             called from factory::init_host_api?"
        );
        assert_eq!(
            after_release_env, while_registered,
            "releasing the OrtEnv unmapped {needle} ({while_registered} mapping \
             entries -> {after_release_env})"
        );
    }
}
