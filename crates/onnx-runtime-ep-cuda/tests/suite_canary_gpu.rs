#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::unusual_byte_groupings,
    clippy::doc_lazy_continuation,
    clippy::uninlined_format_args,
    clippy::cloned_ref_to_slice_refs,
    clippy::type_complexity,
    clippy::drop_non_drop,
    clippy::manual_repeat_n,
    clippy::manual_is_multiple_of,
    clippy::err_expect,
    clippy::clone_on_copy
)]
//! A canary for the whole CUDA suite: did any of it actually run?
//!
//! # Why this exists
//!
//! Every other `*_gpu.rs` file in this crate returns early when there is no
//! usable GPU. That is correct — they cannot run — but it makes a machine with
//! no CUDA and a machine with broken CUDA indistinguishable from a machine
//! where everything passed. `cargo test` reports both as `ok`.
//!
//! Printing a warning does not fix it. The harness captures the output of a
//! passing test, so an `eprintln!("SKIPPED")` is invisible unless the run also
//! passes `--nocapture` — which nobody does when the suite is green.
//!
//! This was not hypothetical. All 44 GPU test files skipped on a developer
//! machine with a working RTX 4060, for as long as it took to notice that the
//! Rust path had no NVIDIA wheel discovery at all. Nothing was red. Among the
//! things that hid behind it: a tensor-core kernel that could not compile, and
//! an allocation counter that had stopped counting.
//!
//! # What to do about it
//!
//! Set `NXRT_REQUIRE_CUDA=1` anywhere a GPU is supposed to exist — a CI lane
//! with one attached, or a developer machine being used to check GPU work. Then
//! "no GPU" stops being silent and becomes this one test failing, with a
//! message saying what to look at. The other 44 files still skip, but nobody is
//! misled about what their `ok` means.
//!
//! Left unset, this is a no-op, so a CPU-only machine is unaffected.

use onnx_runtime_ep_cuda::CudaExecutionProvider;

/// Whether the caller declared that this machine has a GPU.
fn cuda_is_required() -> bool {
    matches!(
        std::env::var("NXRT_REQUIRE_CUDA").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

#[test]
fn cuda_is_usable_where_it_was_declared_to_be() {
    if !cuda_is_required() {
        return;
    }

    let error = match CudaExecutionProvider::new(0) {
        Ok(_) => return,
        Err(error) => error,
    };

    panic!(
        "NXRT_REQUIRE_CUDA is set, so this machine is supposed to have a working CUDA device, \
         and it does not: {error}\n\n\
         Every other *_gpu.rs test in this crate returns early without a GPU and reports `ok`, \
         so their results in this run mean nothing. Common causes:\n\
         - the NVIDIA wheels are not discoverable. Set NXRT_CUDA_WHEEL_ROOTS to the \
           site-packages directory holding `nvidia/`, or put the component `bin`/`lib` \
           directories on the loader path.\n\
         - cuBLAS/cuBLASLt are genuinely absent: `pip install nvidia-cublas-cu12`.\n\
         - there is no CUDA device, in which case unset NXRT_REQUIRE_CUDA rather than \
           leaving the suite claiming to have tested a GPU."
    );
}

/// The NVRTC headers the f16/bf16 and tensor-core kernels compile against.
///
/// Separate from the check above because they fail differently and are fixed
/// differently: the device can be perfectly usable while `cuda_fp16.h` or
/// `crt/mma.h` is missing, and then only the kernels needing them fail — deep
/// inside NVRTC, with a message naming a header rather than the package that
/// carries it. `crt/mma.h` in particular ships in `nvidia-cuda-nvcc`, not
/// `nvidia-cuda-runtime`, which is not guessable from the error.
#[test]
fn the_nvrtc_headers_are_reachable_where_cuda_was_declared() {
    if !cuda_is_required() {
        return;
    }
    let Ok(provider) = CudaExecutionProvider::new(0) else {
        // The check above already reports this, and reporting it twice would
        // point at the wrong fix.
        return;
    };

    provider
        .runtime()
        .require_nvrtc_tensor_core_headers("suite header check")
        .expect(
            "NXRT_REQUIRE_CUDA is set and the CUDA device works, but the NVRTC headers are not \
             reachable, so every f16/bf16 and tensor-core kernel test will skip or fail. \
             `pip install nvidia-cuda-runtime nvidia-cuda-nvcc`, then point \
             NXRT_CUDA_WHEEL_ROOTS at site-packages",
        );
}
