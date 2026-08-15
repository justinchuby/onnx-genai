//! GPU regression test for the write-after-read (WAR) safety of the *public*
//! [`onnx_runtime_session::drive_double_buffer`] driver on a real
//! [`CudaExecutionProvider`] (issue #87, `docs/memory/WEIGHT_OFFLOAD.md` §4).
//!
//! Unlike the hand-rolled fence loop in the ep-cuda runtime tests, this drives
//! the shipped double-buffer strategy end to end and proves the *driver itself*
//! enforces WAR on buffer reuse: with only two staging buffers, slot `s` is
//! reused every second wave, so the copy that refills `s` for wave `n+1` must
//! not overwrite it while wave `n-1`'s (still-running) consumer is reading it.
//! Each wave's consumer spins on the compute stream before reading its buffer,
//! widening the WAR window; if the driver's `copy_wait_fence` were removed, the
//! reuse prefetch would clobber a buffer mid-read and corrupt an earlier wave's
//! output. Every wave's output must equal that wave's distinct payload.
#![cfg(feature = "cuda")]

use std::ffi::c_void;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use onnx_runtime_ep_api::{DeviceBuffer, ExecutionProvider};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::DeviceId;
use onnx_runtime_session::drive_double_buffer;

const MODULE: &str = "session_driver_war_test";
const SOURCE: &str = r#"
extern "C" __global__ void slow_copy(const float* in, float* out, unsigned long long n, long long spin) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long start = clock64();
    while (clock64() - start < spin) { }
    out[i] = in[i];
}
"#;

#[test]
fn drive_double_buffer_war_safe_across_waves() {
    let Ok(ep) = CudaExecutionProvider::initialized(0) else {
        eprintln!("skipping driver WAR test: CUDA EP unavailable");
        return;
    };
    let runtime = ep.runtime().clone();
    let slow_copy = runtime.nvrtc_function(MODULE, SOURCE, "slow_copy").unwrap();

    let waves = 6usize;
    let n = 2048usize;
    let bytes = n * std::mem::size_of::<f32>();
    let n_u64 = n as u64;
    let spin: i64 = 8_000_000;
    let payload = |w: usize| -> Vec<f32> {
        (0..n)
            .map(|i| 1.0 + (w as f32) * 13.0 + (i % 5) as f32)
            .collect()
    };

    // Pinned host payloads, kept alive for the whole drive, wrapped as
    // host-accessible source buffers so `copy_async` takes the real H2D
    // prefetch path onto the transfer stream.
    let pinned: Vec<_> = (0..waves)
        .map(|w| {
            let mut p = runtime.alloc_pinned(bytes).unwrap();
            let src = payload(w);
            p.as_mut_slice().copy_from_slice(unsafe {
                std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), bytes)
            });
            p
        })
        .collect();
    // SAFETY: each pinned staging region outlives its borrowed handle and every
    // use of it, and is only read (never written) through the borrow.
    let sources: Vec<DeviceBuffer> = pinned
        .iter()
        .map(|p| unsafe {
            DeviceBuffer::from_borrowed_parts(
                p.as_slice().as_ptr() as *mut c_void,
                DeviceId::cpu(),
                bytes,
                1,
            )
        })
        .collect();
    let sizes = vec![bytes; waves];

    // Two device staging buffers, poisoned so a mid-read overwrite corrupts the
    // consumer's read visibly.
    let mut buffers = [
        ep.allocate(bytes, 256).unwrap(),
        ep.allocate(bytes, 256).unwrap(),
    ];
    let poison = vec![-777.0f32; n];
    let poison_bytes = unsafe { std::slice::from_raw_parts(poison.as_ptr().cast::<u8>(), bytes) };
    for b in buffers.iter() {
        unsafe { runtime.htod(poison_bytes, cuptr(b.as_ptr())) }.unwrap();
    }
    runtime.synchronize().unwrap();

    // Per-wave device outputs the slow consumer writes into.
    let results: Vec<DeviceBuffer> = (0..waves)
        .map(|_| ep.allocate(bytes, 256).unwrap())
        .collect();

    // Drive the shipped double-buffer strategy. Each wave's `compute` launches a
    // slow consumer on the compute stream that spins (holding the staging buffer
    // under read) then copies it into the wave output. It does NOT synchronize,
    // so overlap is real and a missing driver WAR fence would let a reuse
    // prefetch clobber the buffer mid-spin.
    drive_double_buffer(&ep, &mut buffers, &sources, &sizes, |w, weights| {
        let in_p = cuptr(weights.as_ptr());
        let out_p = cuptr(results[w].as_ptr());
        let mut consume = runtime.stream().launch_builder(&slow_copy);
        consume.arg(&in_p).arg(&out_p).arg(&n_u64).arg(&spin);
        // SAFETY: `in_p`/`out_p` are live device allocations of `bytes` and the
        // grid covers exactly `n` elements.
        unsafe {
            consume
                .launch(LaunchConfig::for_num_elems(n as u32))
                .unwrap()
        };
        Ok(())
    })
    .unwrap();

    runtime.synchronize().unwrap();

    for (w, result) in results.iter().enumerate() {
        let mut host = vec![0.0f32; n];
        let host_bytes =
            unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), bytes) };
        unsafe { runtime.dtoh(host_bytes, cuptr(result.as_ptr())) }.unwrap();
        assert_eq!(
            host,
            payload(w),
            "wave {w} output corrupted — the driver WAR fence was violated: a reuse \
             prefetch clobbered a staging buffer while this wave's consumer was reading it"
        );
    }

    for b in results {
        ep.deallocate(b).unwrap();
    }
    for b in buffers {
        ep.deallocate(b).unwrap();
    }
}
