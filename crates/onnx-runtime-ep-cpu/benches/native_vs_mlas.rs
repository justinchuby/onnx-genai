//! Native vs MLAS, in one binary, so a native route cannot graduate on a claim.
//!
//! # What this exists to prevent
//!
//! MLAS is the baseline *inside* our CPU EP. The long-term direction is to
//! replace its routes with our own, and the failure mode that direction invites
//! is a native kernel that is cleaner, more ours, and slower — shipped because
//! it was measured on a different day, a different binary, or a different
//! machine, or not measured at all. `docs/performance/ABSORBING_MLAS.md`
//! records that cross-worktree comparison on a shared runner showed a uniform
//! 0.70–0.82x offset on *byte-identical* kernels, which is larger than most
//! kernel wins.
//!
//! So both routes live in one binary (`backend_ab`), run interleaved under one
//! load, and are compared by the median of many repetitions. The graduation
//! rule in `docs/performance/CPU_MLAS_MIGRATION.md` is applied here rather than
//! left to the reader: a native route replaces MLAS only with correctness plus
//! a **>=5% repeatable win**, and this bench prints the verdict per case.
//!
//! # Running
//!
//! ```text
//! cargo bench -p onnx-runtime-ep-cpu --bench native_vs_mlas
//! ```
//!
//! Pin the CPUs for a quiet measurement (`taskset -c 0-15 …` on Linux). Without
//! the `mlas` feature the bench still runs and reports native timings, marking
//! every case `no-mlas` — a build with nothing to compare against says so
//! rather than reporting a vacuous win.
//!
//! Output is one tab-separated line per case, so two runs can be diffed:
//!
//! ```text
//! family    case            native_ns  mlas_ns  ratio  verdict
//! ```
//!
//! `ratio` is `mlas / native`: above 1.0 means native is faster.

use std::hint::black_box;

use onnx_runtime_ep_cpu::DispatchBackend;
use onnx_runtime_ep_cpu::backend_ab;
use onnx_runtime_ep_cpu::dispatch_ledger::KernelFamily;

/// The graduation threshold from `docs/performance/CPU_MLAS_MIGRATION.md`.
///
/// Native must be at least this much faster than MLAS before replacing it.
const GRADUATION_RATIO: f64 = 1.05;

/// Repetitions of the interleaved A/B pair. The median of these is reported;
/// an odd count keeps the median a measured value rather than an average of
/// two.
const REPS: usize = 21;

/// Process CPU time in seconds.
///
/// Wall clock is the wrong instrument here. MLAS routes may use their own
/// threading while the native baselines are serial, and a wall-clock
/// comparison of those two would credit MLAS for parallelism this bench did
/// not ask for, or penalise it for scheduling noise on a shared runner. CPU
/// time counts the work actually done either way.
#[cfg(unix)]
fn cpu_seconds() -> f64 {
    // SAFETY: `getrusage` fills a fully-initialised `rusage` through the
    // pointer and returns non-zero on failure without touching it.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return 0.0;
    }
    let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    seconds(usage.ru_utime) + seconds(usage.ru_stime)
}

/// Non-unix hosts fall back to wall clock, with the caveat above.
#[cfg(not(unix))]
fn cpu_seconds() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("timings are finite"));
    samples[samples.len() / 2]
}

/// Run `a` and `b` interleaved `REPS` times and return their median seconds.
///
/// Interleaving is what makes the pair comparable: a machine that slows down
/// halfway through slows both routes down together, and the median of pairs
/// absorbs it. Running all of A and then all of B does not.
fn interleaved<A: FnMut(), B: FnMut()>(mut a: A, mut b: B, iters: usize) -> (f64, f64) {
    for _ in 0..3 {
        a();
        b();
    }
    let mut a_samples = Vec::with_capacity(REPS);
    let mut b_samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t0 = cpu_seconds();
        for _ in 0..iters {
            a();
        }
        let t1 = cpu_seconds();
        for _ in 0..iters {
            b();
        }
        let t2 = cpu_seconds();
        a_samples.push(t1 - t0);
        b_samples.push(t2 - t1);
    }
    (median(a_samples), median(b_samples))
}

/// One row of output, with the graduation rule applied.
fn report(family: KernelFamily, case: &str, native_s: f64, mlas_s: Option<f64>, elems: f64) {
    let ns = |s: f64| s * 1e9 / elems;
    let Some(mlas_s) = mlas_s else {
        println!(
            "{}\t{case}\t{:.4}\t-\t-\tno-mlas",
            family.name(),
            ns(native_s)
        );
        return;
    };
    let ratio = if native_s > 0.0 {
        mlas_s / native_s
    } else {
        f64::NAN
    };
    // `graduates` is the only claim this bench makes: it says the measurement
    // clears the bar, not that the replacement is correct. Correctness is
    // `tests/native_vs_mlas_differential.rs`; both are required.
    let verdict = if ratio >= GRADUATION_RATIO {
        "native-graduates"
    } else if ratio >= 1.0 {
        "native-faster-but-under-5%"
    } else {
        "keep-mlas"
    };
    println!(
        "{}\t{case}\t{:.4}\t{:.4}\t{ratio:.3}\t{verdict}",
        family.name(),
        ns(native_s),
        ns(mlas_s)
    );
}

fn bench_gemm() {
    // Decode (m=1) first: it is the shape that dominates token generation and
    // the one where a native route is most likely to win, because MLAS's
    // packing overhead is amortised over a single row.
    let cases: [(&str, usize, usize, usize); 5] = [
        ("decode_1x2048x2048", 1, 2048, 2048),
        ("decode_1x4096x4096", 1, 4096, 4096),
        ("small_16x512x512", 16, 512, 512),
        ("prefill_128x2048x2048", 128, 2048, 2048),
        ("odd_37x1023x511", 37, 1023, 511),
    ];

    let backends = backend_ab::gemm_backends();
    let is_mlas = |b: onnx_runtime_ep_cpu::CpuBackend| {
        backend_ab::gemm_ledger_backend(b) == DispatchBackend::Mlas
    };
    // The *best* native route, not the portable one. Graduation is about
    // whether our own kernels can replace MLAS, and measuring a scalar
    // reference against a vectorised MLAS answers a question nobody asked:
    // `gemm_backends()` orders native routes weakest-first, so the last
    // non-MLAS entry is the one a replacement would actually ship.
    let native = backends
        .iter()
        .copied()
        .rev()
        .find(|b| !is_mlas(*b))
        .expect("gemm_backends always includes a native route");
    let mlas = backends.iter().copied().find(|b| is_mlas(*b));

    for (case, m, k, n) in cases {
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.031)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.017)
            .collect();
        let mut c_native = vec![0.0f32; m * n];
        let mut c_mlas = vec![0.0f32; m * n];
        // Larger shapes get fewer iterations: the point is a stable median, not
        // a fixed amount of work.
        let iters = (1usize << 28) / (m * k * n).max(1) + 1;

        let (native_s, mlas_s) = match mlas {
            Some(mlas) => {
                let (n_s, m_s) = interleaved(
                    || {
                        backend_ab::gemm_f32(native, &a, &b, &mut c_native, m, k, n).unwrap();
                        black_box(&c_native);
                    },
                    || {
                        backend_ab::gemm_f32(mlas, &a, &b, &mut c_mlas, m, k, n).unwrap();
                        black_box(&c_mlas);
                    },
                    iters,
                );
                (n_s, Some(m_s))
            }
            None => {
                let (n_s, _) = interleaved(
                    || {
                        backend_ab::gemm_f32(native, &a, &b, &mut c_native, m, k, n).unwrap();
                        black_box(&c_native);
                    },
                    || {},
                    iters,
                );
                (n_s, None)
            }
        };
        report(
            KernelFamily::MatMulF32,
            case,
            native_s,
            mlas_s,
            (iters * m * n * k) as f64,
        );
    }
}

fn bench_softmax() {
    for (case, rows, cols) in [
        ("decode_1x32000", 1usize, 32_000usize),
        ("attn_32x512", 32, 512),
        ("prefill_128x4096", 128, 4096),
    ] {
        let seed: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 101) as f32 - 50.0) * 0.07)
            .collect();
        let mut native = seed.clone();
        let mut mlas = seed.clone();
        let iters = (1usize << 22) / (rows * cols).max(1) + 1;

        let (native_s, mlas_s) = if backend_ab::mlas_available() {
            let (n_s, m_s) = interleaved(
                || {
                    native.copy_from_slice(&seed);
                    backend_ab::softmax_rows_native(&mut native, rows, cols);
                    black_box(&native);
                },
                || {
                    mlas.copy_from_slice(&seed);
                    backend_ab::softmax_rows_mlas(&mut mlas, rows, cols);
                    black_box(&mlas);
                },
                iters,
            );
            (n_s, Some(m_s))
        } else {
            let (n_s, _) = interleaved(
                || {
                    native.copy_from_slice(&seed);
                    backend_ab::softmax_rows_native(&mut native, rows, cols);
                    black_box(&native);
                },
                || {},
                iters,
            );
            (n_s, None)
        };
        report(
            KernelFamily::Softmax,
            case,
            native_s,
            mlas_s,
            (iters * rows * cols) as f64,
        );
    }
}

fn bench_transcendentals() {
    for (case, len) in [("small_4096", 4096usize), ("large_1Mi", 1 << 20)] {
        let input: Vec<f32> = (0..len)
            .map(|i| ((i % 257) as f32 - 128.0) * 0.05)
            .collect();
        let mut native = vec![0.0f32; len];
        let mut mlas = vec![0.0f32; len];
        let iters = (1usize << 24) / len.max(1) + 1;

        for (route, native_fn, mlas_fn) in [
            (
                "erf",
                backend_ab::erf_native as fn(&[f32], &mut [f32]),
                backend_ab::erf_mlas as fn(&[f32], &mut [f32]) -> bool,
            ),
            (
                "gelu_exact",
                backend_ab::gelu_native as fn(&[f32], &mut [f32]),
                backend_ab::gelu_mlas as fn(&[f32], &mut [f32]) -> bool,
            ),
        ] {
            let (native_s, mlas_s) = if backend_ab::mlas_available() {
                let (n_s, m_s) = interleaved(
                    || {
                        native_fn(&input, &mut native);
                        black_box(&native);
                    },
                    || {
                        mlas_fn(&input, &mut mlas);
                        black_box(&mlas);
                    },
                    iters,
                );
                (n_s, Some(m_s))
            } else {
                let (n_s, _) = interleaved(
                    || {
                        native_fn(&input, &mut native);
                        black_box(&native);
                    },
                    || {},
                    iters,
                );
                (n_s, None)
            };
            report(
                KernelFamily::Activations,
                &format!("{route}/{case}"),
                native_s,
                mlas_s,
                (iters * len) as f64,
            );
        }
    }
}

fn main() {
    println!("# native vs MLAS, one binary, interleaved, median of {REPS} reps");
    println!(
        "# mlas_linked={}  graduation_ratio={GRADUATION_RATIO}",
        backend_ab::mlas_available()
    );
    println!("family\tcase\tnative_ns_per_elem\tmlas_ns_per_elem\tratio\tverdict");
    bench_gemm();
    bench_softmax();
    bench_transcendentals();
}
