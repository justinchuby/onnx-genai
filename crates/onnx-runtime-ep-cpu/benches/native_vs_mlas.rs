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
//! family  case  native_ns  mlas_ns  ratio  cpu_ratio  verdict
//! ```
//!
//! `ratio` is `mlas / native` in **wall** time: above 1.0 means native is
//! faster. `cpu_ratio` is the same quotient in **process CPU** time.
//!
//! # Why both clocks, and why the verdict uses wall
//!
//! These routes do not use the same number of threads. Our `x86_sgemm` GEMM
//! parallelises over column strips on the Rayon pool; MLAS declines to
//! parallelise some shapes (notably `M=1` decode, where the GEMV is
//! bandwidth-bound and extra cores only add traffic). Judged on CPU time alone
//! a native route that is *faster for the user* looks 10x worse purely because
//! 32 workers each billed their share, which would block a graduation that
//! should happen.
//!
//! Wall time is therefore the graduation criterion: it is the latency a token
//! actually costs. CPU time is kept beside it because the opposite failure is
//! just as real — a native route that "wins" by burning the whole machine
//! regresses every concurrent session on the box. A large `ratio` with a small
//! `cpu_ratio` means exactly that, and is a reason to look before graduating.

use std::hint::black_box;
use std::sync::OnceLock;
use std::time::Instant;

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

/// Process CPU time in seconds, across all threads.
///
/// Reported beside wall time rather than instead of it: CPU time counts the
/// work actually done, so it catches a native route that only looks fast
/// because it recruited every core, but it cannot by itself say whether a
/// token got cheaper.
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

/// Non-unix hosts have no cheap portable equivalent, so they report wall time
/// only and print `-` for `cpu_ratio` rather than a number that is really the
/// wall ratio again.
#[cfg(not(unix))]
fn cpu_seconds() -> f64 {
    f64::NAN
}

/// Wall-clock seconds from a fixed process-local origin.
fn wall_seconds() -> f64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// A paired measurement: latency and the machine-wide cost of achieving it.
#[derive(Clone, Copy)]
struct Timing {
    wall: f64,
    cpu: f64,
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples[samples.len() / 2]
}

/// Run `a` and `b` interleaved `REPS` times and return their median timings.
///
/// Interleaving is what makes the pair comparable: a machine that slows down
/// halfway through slows both routes down together, and the median of pairs
/// absorbs it. Running all of A and then all of B does not.
///
/// Wall and CPU medians are taken independently. That is deliberate: each is a
/// robust summary of its own distribution, and the alternative — picking the
/// pair at the median wall sample — would report a CPU figure from a single
/// repetition and inherit all of its noise.
fn interleaved<A: FnMut(), B: FnMut()>(mut a: A, mut b: B, iters: usize) -> (Timing, Timing) {
    for _ in 0..3 {
        a();
        b();
    }
    let mut a_wall = Vec::with_capacity(REPS);
    let mut a_cpu = Vec::with_capacity(REPS);
    let mut b_wall = Vec::with_capacity(REPS);
    let mut b_cpu = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let (w0, c0) = (wall_seconds(), cpu_seconds());
        for _ in 0..iters {
            a();
        }
        let (w1, c1) = (wall_seconds(), cpu_seconds());
        for _ in 0..iters {
            b();
        }
        let (w2, c2) = (wall_seconds(), cpu_seconds());
        a_wall.push(w1 - w0);
        a_cpu.push(c1 - c0);
        b_wall.push(w2 - w1);
        b_cpu.push(c2 - c1);
    }
    (
        Timing {
            wall: median(a_wall),
            cpu: median(a_cpu),
        },
        Timing {
            wall: median(b_wall),
            cpu: median(b_cpu),
        },
    )
}

/// One row of output, with the graduation rule applied.
fn report(family: KernelFamily, case: &str, native: Timing, mlas: Option<Timing>, elems: f64) {
    let ns = |s: f64| s * 1e9 / elems;
    let Some(mlas) = mlas else {
        println!(
            "{}\t{case}\t{:.4}\t-\t-\t-\tno-mlas",
            family.name(),
            ns(native.wall)
        );
        return;
    };
    let quotient = |m: f64, n: f64| if n > 0.0 { m / n } else { f64::NAN };
    let ratio = quotient(mlas.wall, native.wall);
    let cpu_ratio = quotient(mlas.cpu, native.cpu);
    // `graduates` is the only claim this bench makes: it says the measurement
    // clears the bar, not that the replacement is correct. Correctness is
    // `tests/native_vs_mlas_differential.rs`; both are required.
    let verdict = if ratio >= GRADUATION_RATIO {
        // A latency win bought with a large multiple of the machine is not
        // free, so it is flagged rather than waved through. The threshold is
        // the graduation ratio again: native may spend up to 5% more CPU per
        // unit of work without comment.
        if cpu_ratio.is_finite() && cpu_ratio < 1.0 / GRADUATION_RATIO {
            "native-graduates-but-costs-more-cpu"
        } else {
            "native-graduates"
        }
    } else if ratio >= 1.0 {
        "native-faster-but-under-5%"
    } else {
        "keep-mlas"
    };
    let cpu_ratio = if cpu_ratio.is_finite() {
        format!("{cpu_ratio:.3}")
    } else {
        "-".to_string()
    };
    println!(
        "{}\t{case}\t{:.4}\t{:.4}\t{ratio:.3}\t{cpu_ratio}\t{verdict}",
        family.name(),
        ns(native.wall),
        ns(mlas.wall)
    );
}

/// The native GEMM route this host will actually measure.
///
/// Named in the output because a GEMM ratio is unreadable without it: the same
/// table means "our AVX2 SGEMM trails MLAS" on one host and "our portable
/// fallback trails MLAS" on a host without AVX2/FMA, and only the first is a
/// finding about the kernel we ship.
fn gemm_native_backend() -> onnx_runtime_ep_cpu::CpuBackend {
    backend_ab::gemm_backends()
        .into_iter()
        .rev()
        .find(|b| backend_ab::gemm_ledger_backend(*b) != DispatchBackend::Mlas)
        .expect("gemm_backends always includes a native route")
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

    // The *best* native route, not the portable one. Graduation is about
    // whether our own kernels can replace MLAS, and measuring a scalar
    // reference against a vectorised MLAS answers a question nobody asked.
    let native = gemm_native_backend();
    let mlas = backend_ab::gemm_backends()
        .into_iter()
        .find(|b| backend_ab::gemm_ledger_backend(*b) == DispatchBackend::Mlas);

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
        "# mlas_linked={}  graduation_ratio={GRADUATION_RATIO}  threads={}",
        backend_ab::mlas_available(),
        rayon::current_num_threads()
    );
    // Times are per unit of work, so shapes of different sizes are comparable:
    // per multiply-accumulate for GEMM, per element for softmax and the
    // transcendentals. `ratio` is wall, `cpu_ratio` is process CPU; both are
    // mlas/native, so above 1.0 favours native.
    println!("# gemm native backend = {:?}", gemm_native_backend());
    println!("family\tcase\tnative_ns_per_unit\tmlas_ns_per_unit\tratio\tcpu_ratio\tverdict");
    bench_gemm();
    bench_softmax();
    bench_transcendentals();
}
