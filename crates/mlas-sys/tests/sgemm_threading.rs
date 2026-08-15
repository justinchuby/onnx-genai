//! Threading regression guard for the standalone MLAS SGEMM entry points.
//!
//! This lives in its own integration-test binary **on purpose**.
//! [`mlas_sys::mlas_threading_stats`] reports process-global counters, and
//! `cargo test` runs the unit tests of a crate concurrently inside a single
//! process. If this assertion shared a process with the other MLAS tests, a
//! concurrent `sqnbit_gemm` call — which passes its own non-null threadpool
//! sentinel and therefore *does* drive the backend — could bump the counter
//! between the two samples and let a genuinely broken build pass.
//!
//! A separate file is compiled into a separate binary, so the calls below are
//! the only MLAS work in this process and the delta is unambiguous.

use mlas_sys::{mlas_threading_degree, mlas_threading_stats, sgemm_nn};

fn seq(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.013 + seed).sin()) * 2.0)
        .collect()
}

/// `MlasGemmBatch` must receive a non-null `MLAS_THREADPOOL` so the standalone
/// `MlasTrySimpleParallel` hands its partitions to the registered
/// work-stealing backend.
///
/// Deterministic by construction rather than a timing check: passing `nullptr`
/// makes `MlasStandaloneParallelFor` take its serial fallback loop, so the
/// backend is never entered and `parallel_for_calls` stays flat.
#[test]
fn sgemm_nn_drives_the_registered_parallel_backend() {
    let (m, n, k) = (256usize, 512usize, 256usize);
    let a = seq(m * k, 0.25);
    let b = seq(k * n, 0.75);
    let mut c = vec![0.0f32; m * n];

    // Establish the pool and MLAS platform init before sampling.
    sgemm_nn(m, n, k, &a, &b, &mut c);
    if mlas_threading_degree() < 2 {
        eprintln!("skipped: MLAS threading degree is 1 on this host");
        return;
    }

    let before = mlas_threading_stats().parallel_for_calls;
    sgemm_nn(m, n, k, &a, &b, &mut c);
    let after = mlas_threading_stats().parallel_for_calls;

    assert!(
        after > before,
        "sgemm_nn did not enter the parallel-for backend ({before} -> {after} \
         calls); MlasGemmBatch was likely handed a null MLAS_THREADPOOL, which \
         forces MLAS's serial fallback"
    );
}
