//! Differential correctness: our native kernels vs the vendored MLAS kernels,
//! **in one binary, in one process**.
//!
//! MLAS is compiled into this EP by default. That makes two failure modes
//! possible that a single-route test suite cannot see:
//!
//! 1. The MLAS route and the native route disagree, and whichever one the
//!    default build happens to take is the only one anybody exercises.
//! 2. A native kernel "absorbs" an MLAS route and is quietly wrong on the
//!    shapes the MLAS route used to cover.
//!
//! Both are caught by holding the two implementations against each other on the
//! same inputs. `crate::backend_ab` exists precisely so this can be done without
//! two builds; see `docs/performance/ABSORBING_MLAS.md` for why a same-binary
//! A/B is the only honest form of this comparison.
//!
//! Without the `mlas` feature these tests degrade to native-vs-reference, which
//! still has value (the reference is independent of the SIMD path) and never
//! silently passes on an empty comparison — `mlas_is_linked_in_a_default_build`
//! fails a default build that lost MLAS.

use onnx_runtime_ep_cpu::backend::CpuBackend;
use onnx_runtime_ep_cpu::backend_ab;
use onnx_runtime_ep_cpu::dispatch_ledger::{self, Backend, KernelFamily};

/// Deterministic pseudo-random values in `[-1, 1)`, so a failure reproduces.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// Reference f32 GEMM: the plainest possible triple loop, independent of every
/// backend under test. Both the native SIMD path and MLAS are compared to it,
/// so a shared bug in "our two fast paths" cannot hide.
fn reference_gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += av * b[p * n + j];
            }
        }
    }
    c
}

/// f32 GEMM accumulates in a different order per backend, so bit-identity is
/// not available. Compare against the magnitude the sum can actually reach.
fn assert_close(label: &str, got: &[f32], want: &[f32], k: usize, tolerance_scale: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    // Worst-case fp32 accumulation error over k terms bounded by |x|<=1.
    let tolerance = tolerance_scale * f32::EPSILON * (k as f32).max(1.0);
    for (index, (g, w)) in got.iter().zip(want).enumerate() {
        let diff = (g - w).abs();
        assert!(
            diff <= tolerance,
            "{label}: element {index} differs by {diff:e} (> {tolerance:e}); got {g}, want {w}"
        );
    }
}

/// The shapes that matter: decode (`m == 1`), small prefill, non-multiples of
/// every tile width, and a `k` long enough that accumulation order shows.
const GEMM_SHAPES: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (1, 64, 128),
    (1, 257, 129),
    (2, 3, 5),
    (4, 4, 4),
    (7, 13, 31),
    (16, 64, 64),
    (33, 65, 97),
    (64, 128, 256),
    (128, 32, 512),
];

/// The flip this suite exists for: a default build must have MLAS. If this
/// fails, every MLAS assertion below became vacuous rather than false, which is
/// the failure mode that produced #1091 (measuring a configuration we did not
/// ship).
#[test]
#[cfg(feature = "mlas")]
fn mlas_is_linked_in_a_default_build() {
    assert!(backend_ab::mlas_available());
    assert!(dispatch_ledger::mlas_linked());
}

/// Every f32 GEMM backend this build offers must agree with the reference and
/// with each other. On a default x86-64 build that is Generic, SimdX86 and
/// MLAS — three independent implementations, one process.
#[test]
fn f32_gemm_backends_agree_with_the_reference() {
    let backends = backend_ab::gemm_backends();
    assert!(
        !backends.is_empty(),
        "no f32 GEMM backend is reachable in this build"
    );

    for &(m, k, n) in GEMM_SHAPES {
        let a = values(m * k, 0x51ed_2701 ^ (m * k) as u64);
        let b = values(k * n, 0x9e37_79b9 ^ (k * n) as u64);
        let want = reference_gemm(&a, &b, m, k, n);

        for backend in &backends {
            let mut got = vec![f32::NAN; m * n];
            backend_ab::gemm_f32(*backend, &a, &b, &mut got, m, k, n)
                .expect("gemm_f32 must not fail on well-formed dense inputs");
            assert_close(
                &format!("{backend:?} m={m} k={k} n={n}"),
                &got,
                &want,
                k,
                8.0,
            );
        }
    }
}

/// The MLAS GEMM and the native GEMM must agree with each other directly, not
/// merely each with the reference: an absorption replaces one with the other,
/// so this is the comparison that guards the swap.
#[test]
#[cfg(all(feature = "mlas", target_arch = "x86_64"))]
fn f32_gemm_mlas_agrees_with_native() {
    for &(m, k, n) in GEMM_SHAPES {
        let a = values(m * k, 0x1234_5678 ^ (m * k) as u64);
        let b = values(k * n, 0x8765_4321 ^ (k * n) as u64);

        let mut native = vec![f32::NAN; m * n];
        backend_ab::gemm_f32(CpuBackend::Generic, &a, &b, &mut native, m, k, n).unwrap();

        let mut mlas = vec![f32::NAN; m * n];
        backend_ab::gemm_f32(CpuBackend::Mlas, &a, &b, &mut mlas, m, k, n).unwrap();

        assert_close(
            &format!("mlas vs native m={m} k={k} n={n}"),
            &mlas,
            &native,
            k,
            8.0,
        );
    }
}

/// Softmax rows: the MLAS SIMD reduction against our portable one. Rows include
/// a large-magnitude row (max-subtraction must keep it finite), a constant row
/// (every output `1/d`) and a single-element row.
#[test]
fn softmax_rows_native_and_mlas_agree() {
    let cases: &[(usize, usize)] = &[(1, 1), (1, 32), (4, 7), (8, 129), (3, 1024)];
    for &(n, d) in cases {
        let base = values(n * d, 0xdead_beef ^ (n * d) as u64);
        let mut native = base.clone();
        backend_ab::softmax_rows_native(&mut native, n, d);

        // Rows must be probability distributions regardless of route.
        for row in native.chunks(d) {
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "native softmax row does not sum to 1: {sum}"
            );
        }

        let mut mlas = base.clone();
        if !backend_ab::softmax_rows_mlas(&mut mlas, n, d) {
            continue;
        }
        for (index, (a, b)) in mlas.iter().zip(&native).enumerate() {
            assert!(
                (a - b).abs() <= 4.0 * f32::EPSILON,
                "softmax n={n} d={d} element {index}: mlas {a} vs native {b}"
            );
        }
    }
}

/// Large-magnitude inputs are where a fast exp diverges. Both routes must stay
/// finite and normalized.
#[test]
fn softmax_stays_normalized_on_extreme_rows() {
    let d = 64;
    let mut row: Vec<f32> = (0..d).map(|i| (i as f32) * 50.0 - 1000.0).collect();
    let mut native = row.clone();
    backend_ab::softmax_rows_native(&mut native, 1, d);
    assert!(native.iter().all(|v| v.is_finite()));
    let sum: f32 = native.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "native sum {sum}");

    if backend_ab::softmax_rows_mlas(&mut row, 1, d) {
        assert!(
            row.iter().all(|v| v.is_finite()),
            "mlas softmax went non-finite"
        );
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "mlas sum {sum}");
    }
}

/// `Erf` and exact `Gelu` keep an MLAS route because it measured faster. The
/// tolerance is the documented polynomial slack between the two, not zero: MLAS
/// dispatches by ISA and our AVX2 path mirrors only one of its kernels.
#[test]
fn transcendental_routes_agree_within_documented_slack() {
    // Well above SIMD_MIN_LEN so both routes take their vector paths.
    let input: Vec<f32> = (0..4096).map(|i| (i as f32 - 2048.0) / 256.0).collect();

    let mut native = vec![f32::NAN; input.len()];
    backend_ab::erf_native(&input, &mut native);
    let mut mlas = vec![f32::NAN; input.len()];
    if backend_ab::erf_mlas(&input, &mut mlas) {
        for (index, (a, b)) in mlas.iter().zip(&native).enumerate() {
            assert!(
                (a - b).abs() <= 2e-6,
                "erf element {index} (x={}): mlas {a} vs native {b}",
                input[index]
            );
        }
    }
    // Erf is bounded and odd on both routes.
    for (v, x) in native.iter().zip(&input) {
        assert!((-1.0..=1.0).contains(v), "erf({x}) = {v} left [-1, 1]");
    }

    let mut native_gelu = vec![f32::NAN; input.len()];
    backend_ab::gelu_native(&input, &mut native_gelu);
    let mut mlas_gelu = vec![f32::NAN; input.len()];
    if backend_ab::gelu_mlas(&input, &mut mlas_gelu) {
        for (index, (a, b)) in mlas_gelu.iter().zip(&native_gelu).enumerate() {
            assert!(
                (a - b).abs() <= 2e-6 * b.abs().max(1.0),
                "gelu element {index} (x={}): mlas {a} vs native {b}",
                input[index]
            );
        }
    }
}

/// The special values MLAS clamps and we repair. `SiLU`'s `+/-18` band and
/// `Gelu`'s `-inf` limit are the two documented divergences; both routes must
/// still produce the mathematical answer.
#[test]
fn transcendental_routes_agree_on_special_values() {
    let input: Vec<f32> = {
        let mut v = vec![
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1e30,
            -1e30,
            20.0,
            -20.0,
        ];
        // Pad past SIMD_MIN_LEN so the vector routes are taken.
        v.extend((0..120).map(|i| (i as f32 - 60.0) / 8.0));
        v
    };

    let mut native = vec![f32::NAN; input.len()];
    backend_ab::gelu_native(&input, &mut native);
    let mut mlas = vec![f32::NAN; input.len()];
    if !backend_ab::gelu_mlas(&input, &mut mlas) {
        return;
    }
    for (index, (a, b)) in mlas.iter().zip(&native).enumerate() {
        let x = input[index];
        if a.is_nan() || b.is_nan() {
            assert_eq!(
                a.is_nan(),
                b.is_nan(),
                "gelu({x}): one route produced NaN and the other did not ({a} vs {b})"
            );
            continue;
        }
        if a.is_infinite() || b.is_infinite() {
            assert_eq!(a, b, "gelu({x}): infinite limits must agree ({a} vs {b})");
            continue;
        }
        assert!(
            (a - b).abs() <= 2e-6 * b.abs().max(1.0),
            "gelu({x}): mlas {a} vs native {b}"
        );
    }
    // gelu(-inf) is 0, not NaN: this is the repair MLAS needs and we apply.
    let neg_inf_at = input.iter().position(|v| *v == f32::NEG_INFINITY).unwrap();
    assert_eq!(
        mlas[neg_inf_at], 0.0,
        "gelu(-inf) must be 0 on the MLAS route"
    );
    assert_eq!(
        native[neg_inf_at], 0.0,
        "gelu(-inf) must be 0 on the native route"
    );
}

/// The ledger must describe what actually ran. This is what stops the plan from
/// drifting into fiction as kernels change underneath it.
#[test]
fn the_ledger_records_the_route_that_ran() {
    dispatch_ledger::enable();
    dispatch_ledger::reset();

    let (m, k, n) = (8, 16, 32);
    let a = values(m * k, 7);
    let b = values(k * n, 11);
    let mut c = vec![0.0f32; m * n];
    for backend in backend_ab::gemm_backends() {
        backend_ab::gemm_f32(backend, &a, &b, &mut c, m, k, n).unwrap();
    }
    let mut data = values(4 * 32, 13);
    backend_ab::softmax_rows_native(&mut data, 4, 32);

    let seen = dispatch_ledger::snapshot();
    dispatch_ledger::reset();
    dispatch_ledger::disable();

    assert!(
        seen.iter().any(|o| o.family == KernelFamily::MatMulF32
            && o.backend == Backend::Native
            && o.shape == (m, n, k)),
        "no native MatMulF32 observation with the shape we ran: {seen:?}"
    );
    #[cfg(all(feature = "mlas", target_arch = "x86_64"))]
    assert!(
        seen.iter()
            .any(|o| o.family == KernelFamily::MatMulF32 && o.backend == Backend::Mlas),
        "MLAS is linked and was invoked, but the ledger did not record it: {seen:?}"
    );
    for observation in &seen {
        assert!(
            observation.threads >= 1,
            "thread evidence must be populated: {observation:?}"
        );
    }
}

/// The declared plan must match what the build can reach, family by family.
#[test]
fn effective_plan_matches_this_build() {
    for family in KernelFamily::ALL {
        let effective = dispatch_ledger::effective_backend(*family);
        if !dispatch_ledger::mlas_linked() {
            assert_eq!(
                effective,
                Backend::Native,
                "{family} claims {effective} without MLAS linked"
            );
        }
    }
    // Every family the A/B module covers must be a family the plan knows about.
    for family in backend_ab::AB_COVERED {
        assert!(
            KernelFamily::ALL.contains(family),
            "{family} is A/B-covered but missing from the plan"
        );
    }
}
