//! Differential correctness: our native kernels vs the vendored MLAS
//! *reference* kernels, **in one binary, in one process**.
//!
//! # This suite is opt-in, and deliberately not part of a default build
//!
//! Production is native: a default build links no MLAS, so run this with
//! `--features mlas` when you want the comparison. That is the whole point —
//! MLAS is a research reference we measure against and absorb from, never
//! something a shipped artifact contains. `default_build_can_reach_no_mlas_route`
//! below is what makes the default configuration state that rather than assume
//! it.
//!
//! # What the comparison catches
//!
//! 1. A native kernel "absorbs" a reference route and is quietly wrong on the
//!    shapes that route used to cover.
//! 2. A native route and the reference disagree, and nobody notices because
//!    only one of them is ever exercised in any single build.
//!
//! Both are caught by holding the two implementations against each other on the
//! same inputs. `onnx_runtime_ep_cpu::backend_ab` exists precisely so this can
//! be done without two builds; see `docs/performance/ABSORBING_MLAS.md` for why
//! a same-binary A/B is the only honest form of this comparison.
//!
//! Without the `mlas` feature these tests degrade to native-vs-independent-
//! reference, which still has value (the triple-loop reference is independent
//! of the SIMD path) and never silently passes on an empty comparison.

// Only the x86-64 MLAS GEMM comparison names a backend explicitly; every other
// route is selected through `backend_ab`. Gated so aarch64 (where MLAS has no
// GEMM route yet) does not fail the `-D warnings` cross-arch lane.
#[cfg(all(feature = "mlas", target_arch = "x86_64"))]
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

/// The comparison this suite exists for is only real when the reference is
/// actually present. Under `--features mlas`, if this fails then every MLAS
/// assertion below became vacuous rather than false — the failure mode that
/// produced #1091 (measuring a configuration we did not ship).
#[test]
#[cfg(feature = "mlas")]
fn the_reference_is_linked_when_this_suite_is_run_with_it() {
    assert!(backend_ab::mlas_available());
    assert!(dispatch_ledger::mlas_linked());
}

/// The mirror assertion, and the one that holds in every shipped build: with
/// default features there is no MLAS to reach, and no family whose *effective*
/// route is MLAS.
///
/// This is the in-crate half of the artifact falsifiers in
/// `crates/onnx-runtime-ep-cpu-plugin/tests/default_artifacts_are_mlas_free.rs`:
/// that suite proves the shipped binary carries no MLAS symbols, this one
/// proves the code could not route to MLAS even if it did. The live-recording
/// half is in `the_ledger_records_the_route_that_ran`, which owns the
/// process-global recorder — two tests toggling it would race each other rather
/// than the code.
#[test]
#[cfg(not(feature = "mlas"))]
fn default_build_can_reach_no_mlas_route() {
    assert!(
        !backend_ab::mlas_available(),
        "a default build must not link the MLAS reference"
    );
    assert!(!dispatch_ledger::mlas_linked());

    assert!(
        !KernelFamily::ALL.is_empty(),
        "probe enumerated no families, so the loop below would pass vacuously"
    );
    for family in KernelFamily::ALL {
        assert_eq!(
            dispatch_ledger::effective_backend(*family),
            Backend::Native,
            "{family}: no family may have a non-native effective route in a default build"
        );
    }
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

// ─── Scaled dot-product attention ───────────────────────────────────────────

/// The SDPA shapes this suite holds the two routes to.
///
/// The grid is chosen for *routes*, not for coverage percentage:
///
/// * `issue_1685` is the exact shape whose output grew an eight-row hole in
///   #1685 — 12 tiles of `128x64`, enough `(b, head)` tiles to fan out and
///   enough rows for MLAS to partition the `probs.V` GEMM across its pool.
/// * `decode_*` is `q_seq == 1`, the shape a generation step actually runs and
///   the one the standing MHA benchmark grid never contained.
/// * `gqa_*` has `kv_heads < heads`, so the KV-head folding in both routes is
///   exercised rather than assumed.
/// * `ragged` has no dimension that is a multiple of any tile width.
struct SdpaShape {
    label: &'static str,
    batch: usize,
    heads: usize,
    kv_heads: usize,
    q_seq: usize,
    kv_seq: usize,
    head_size: usize,
    v_head_size: usize,
}

/// Shorthand so the grid below stays a readable table.
#[allow(clippy::too_many_arguments)] // one positional per shape dimension; a builder would obscure the grid
const fn shape(
    label: &'static str,
    batch: usize,
    heads: usize,
    kv_heads: usize,
    q_seq: usize,
    kv_seq: usize,
    head_size: usize,
    v_head_size: usize,
) -> SdpaShape {
    SdpaShape {
        label,
        batch,
        heads,
        kv_heads,
        q_seq,
        kv_seq,
        head_size,
        v_head_size,
    }
}

const SDPA_SHAPES: &[SdpaShape] = &[
    shape("issue_1685", 1, 12, 4, 128, 128, 64, 64),
    shape("decode_llama_kv1024", 1, 32, 8, 1, 1024, 128, 128),
    shape("decode_short", 2, 8, 8, 1, 129, 80, 80),
    shape("prefill_bert", 1, 12, 12, 128, 128, 64, 64),
    shape("gqa_prefill", 1, 16, 4, 64, 192, 128, 128),
    shape("cross_attention", 1, 8, 8, 48, 300, 64, 64),
    shape("ragged", 3, 6, 3, 7, 71, 33, 17),
    shape("wide_v", 1, 4, 2, 5, 37, 64, 96),
];

fn sdpa_case(shape: &SdpaShape, causal: bool) -> backend_ab::SdpaCase {
    backend_ab::SdpaCase {
        batch: shape.batch,
        num_heads: shape.heads,
        num_kv_heads: shape.kv_heads,
        q_seq: shape.q_seq,
        kv_seq: shape.kv_seq,
        head_size: shape.head_size,
        v_head_size: shape.v_head_size,
        causal,
    }
}

/// Softmax attention is a convex combination of the `V` rows, so every output
/// element must lie inside that head's per-column range of `V`. This is an
/// oracle neither route can satisfy by accident, and it holds in a default
/// build where there is no MLAS to compare against — so this suite still has
/// an independent check when the reference is absent.
fn assert_context_is_a_convex_combination(
    label: &str,
    case: &backend_ab::SdpaCase,
    v: &[f32],
    y: &[f32],
) {
    let heads_per_kv = case.num_heads / case.num_kv_heads;
    for b in 0..case.batch {
        for head in 0..case.num_heads {
            let kv_head = head / heads_per_kv;
            let v_off = ((b * case.num_kv_heads + kv_head) * case.kv_seq) * case.v_head_size;
            for d in 0..case.v_head_size {
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                for j in 0..case.kv_seq {
                    let value = v[v_off + j * case.v_head_size + d];
                    lo = lo.min(value);
                    hi = hi.max(value);
                }
                let y_off = ((b * case.num_heads + head) * case.q_seq) * case.v_head_size;
                for i in 0..case.q_seq {
                    let got = y[y_off + i * case.v_head_size + d];
                    // Slack for fp32 rounding of the weighted sum.
                    let slack = 1e-4 + (hi - lo).abs() * 1e-4;
                    assert!(
                        got >= lo - slack && got <= hi + slack,
                        "{label}: context[b{b} h{head} i{i} d{d}] = {got} escapes the V range \
                         [{lo}, {hi}] — attention output must be a convex combination of V"
                    );
                }
            }
        }
    }
}

/// **Fail loud on lost work.** Both SDPA routes write the whole context tensor
/// with `beta = 0`, so an element the kernel never touched is a defect, not a
/// number.
///
/// This is the guard #1685 asked for. That defect surfaced as an eight-row
/// contiguous run of exact `0.0` inside one `(batch, head)` tile, produced by a
/// GEMM partition that never ran. It was only visible because a *different*
/// test happened to compare two runs; against a zero-filled output buffer a
/// dropped write is indistinguishable from a legitimate zero, and against a
/// single run it is invisible entirely. Poisoning the buffer with `NaN` makes
/// lost work unambiguous and independent of the values involved.
#[test]
fn sdpa_routes_write_every_output_element() {
    assert!(
        !SDPA_SHAPES.is_empty(),
        "empty shape grid would pass vacuously"
    );
    let mut compared_mlas = 0usize;
    for shape in SDPA_SHAPES {
        for causal in [false, true] {
            let case = sdpa_case(shape, causal);
            let q = values(case.q_len(), 0x5DAA_0001);
            let k = values(case.k_len(), 0x5DAA_0002);
            let v = values(case.v_len(), 0x5DAA_0003);

            let mut native = vec![f32::NAN; case.y_len()];
            backend_ab::sdpa_native(&case, &q, &k, &v, &mut native);
            assert_no_unwritten(
                &format!("{} causal={causal} native", shape.label),
                &case,
                &native,
            );
            assert_context_is_a_convex_combination(
                &format!("{} causal={causal} native", shape.label),
                &case,
                &v,
                &native,
            );

            let mut mlas = vec![f32::NAN; case.y_len()];
            if backend_ab::sdpa_mlas(&case, &q, &k, &v, &mut mlas) {
                compared_mlas += 1;
                assert_no_unwritten(
                    &format!("{} causal={causal} mlas", shape.label),
                    &case,
                    &mlas,
                );
                assert_context_is_a_convex_combination(
                    &format!("{} causal={causal} mlas", shape.label),
                    &case,
                    &v,
                    &mlas,
                );
            }
        }
    }
    assert_eq!(
        compared_mlas > 0,
        backend_ab::mlas_available(),
        "the MLAS SDPA route must run exactly when the reference is linked"
    );
}

/// Report an unwritten region the way the defect actually presents: the count,
/// the first index, and the `(head, row)` it decodes to — because "8 rows of
/// one head's tile" is the fingerprint of a dropped GEMM partition, and a bare
/// index is not.
fn assert_no_unwritten(label: &str, case: &backend_ab::SdpaCase, y: &[f32]) {
    let unwritten: Vec<usize> = y
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_nan())
        .map(|(index, _)| index)
        .collect();
    if unwritten.is_empty() {
        return;
    }
    let tile = case.q_seq * case.v_head_size;
    let first = unwritten[0];
    panic!(
        "{label}: {} of {} output elements were never written (first index {first} = \
         batch-head tile {}, row {}, column {}). A `beta = 0` GEMM must overwrite all of C; \
         a contiguous run of whole rows is a dropped work partition, not a numerical error.",
        unwritten.len(),
        y.len(),
        first / tile,
        (first % tile) / case.v_head_size,
        first % case.v_head_size,
    );
}

/// The native route and the MLAS reference must agree on the attention core.
///
/// This is [`KernelFamily::AttentionTranspose`]'s missing A/B. Its plan entry
/// claims `Partial("everything except the two inner GEMMs is native")`, and
/// until now nothing in this suite held that claim to a comparison — the one
/// family with a known reference-route defect was the one family with no
/// same-binary A/B.
///
/// The two routes reassociate both matmuls differently (a GEMM against a
/// hand-rolled dot/axpy loop), so the tolerance is accumulation slack over
/// `kv_seq` terms, not zero.
#[test]
#[cfg(feature = "mlas")]
fn sdpa_native_and_mlas_agree() {
    for shape in SDPA_SHAPES {
        for causal in [false, true] {
            let case = sdpa_case(shape, causal);
            let q = values(case.q_len(), 0x5DAA_0011);
            let k = values(case.k_len(), 0x5DAA_0012);
            let v = values(case.v_len(), 0x5DAA_0013);

            let mut native = vec![f32::NAN; case.y_len()];
            backend_ab::sdpa_native(&case, &q, &k, &v, &mut native);
            let mut mlas = vec![f32::NAN; case.y_len()];
            assert!(backend_ab::sdpa_mlas(&case, &q, &k, &v, &mut mlas));

            // Both outputs are convex combinations of V, so magnitudes are
            // O(1); the error budget is the softmax-weighted sum over kv_seq.
            assert_close(
                &format!("sdpa {} causal={causal}", shape.label),
                &mlas,
                &native,
                case.kv_seq,
                64.0,
            );
        }
    }
}

/// #1685's real trigger was **concurrency**, not shape: several threads driving
/// SDPA at once, each fanning its `(batch, head)` tiles across a pool while the
/// inner GEMMs dispatch onto the shared MLAS pool underneath.
///
/// The pool bug that produced it (a worker from a finished dispatch still
/// inside `run_job` when the next dispatch republished the loop counters —
/// see `crates/mlas-sys/tests/concurrent_dispatch.rs`) only appears when
/// dispatches overlap. A single-threaded shape sweep cannot reach it, which is
/// why the original report reproduced at ~3% per process and not at all in a
/// tight in-process loop.
#[test]
fn concurrent_sdpa_sessions_lose_no_work() {
    let case = backend_ab::SdpaCase {
        batch: 1,
        num_heads: 12,
        num_kv_heads: 4,
        q_seq: 128,
        kv_seq: 128,
        head_size: 64,
        v_head_size: 64,
        causal: true,
    };
    let q = values(case.q_len(), 0x5DAA_0021);
    let k = values(case.k_len(), 0x5DAA_0022);
    let v = values(case.v_len(), 0x5DAA_0023);

    let sessions = 4usize;
    let rounds = 24usize;
    std::thread::scope(|scope| {
        for session in 0..sessions {
            let (case, q, k, v) = (case, &q, &k, &v);
            scope.spawn(move || {
                for round in 0..rounds {
                    let label = format!("session {session} round {round}");
                    let mut native = vec![f32::NAN; case.y_len()];
                    backend_ab::sdpa_native(&case, q, k, v, &mut native);
                    assert_no_unwritten(&format!("{label} native"), &case, &native);

                    let mut mlas = vec![f32::NAN; case.y_len()];
                    if backend_ab::sdpa_mlas(&case, q, k, v, &mut mlas) {
                        assert_no_unwritten(&format!("{label} mlas"), &case, &mlas);
                    }
                }
            });
        }
    });
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

    // The native invariants are asserted before the MLAS gate, not after it.
    // With them below, a `--no-default-features` build returned from the gate
    // having asserted nothing and still reported a pass -- exactly the vacuous
    // green this file claims not to produce.
    let neg_inf_at = input
        .iter()
        .position(|v| *v == f32::NEG_INFINITY)
        .expect("the input carries -inf");
    assert_eq!(
        native[neg_inf_at], 0.0,
        "gelu(-inf) must be 0 on the native route"
    );
    let pos_inf_at = input
        .iter()
        .position(|v| *v == f32::INFINITY)
        .expect("the input carries +inf");
    assert_eq!(
        native[pos_inf_at],
        f32::INFINITY,
        "gelu(+inf) must be +inf on the native route"
    );
    for (index, value) in native.iter().enumerate() {
        assert!(
            !value.is_nan(),
            "gelu({}) produced NaN on the native route",
            input[index]
        );
    }

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
    assert_eq!(
        mlas[neg_inf_at], 0.0,
        "gelu(-inf) must be 0 on the MLAS route"
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
    assert_eq!(
        dispatch_ledger::dropped(),
        0,
        "the ledger truncated this run, so `seen` is a prefix and the assertions \
         above are weaker than they read"
    );

    // The live half of the default-artifact claim: a build with no MLAS must
    // not record a route that used it. Vacuity is already excluded -- the
    // assertions above prove `seen` is non-empty.
    #[cfg(not(feature = "mlas"))]
    for observation in &seen {
        assert_ne!(
            observation.backend,
            Backend::Mlas,
            "a default build recorded an MLAS route: {observation:?}"
        );
        assert_ne!(
            observation.backend,
            Backend::NativeOverMlas,
            "a default build recorded a native-over-MLAS route: {observation:?}"
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

    // The other half of the claim, which the `!mlas_linked()` branch above
    // cannot make: in a build that *does* link MLAS, at least one family has to
    // reach it. Without this the whole ledger could report `Native` everywhere
    // and every assertion here would still hold.
    if dispatch_ledger::mlas_linked() {
        let reaching: Vec<&KernelFamily> = KernelFamily::ALL
            .iter()
            .filter(|f| dispatch_ledger::effective_backend(**f) != Backend::Native)
            .collect();
        assert!(
            !reaching.is_empty(),
            "MLAS is linked but the plan says no family reaches it"
        );
    }

    // The GEMM families are the ones whose reachability is target-dependent
    // (`auto_detect` only returns `Mlas` on non-Apple, non-Android x86-64), so
    // check the ledger against the dispatcher rather than against itself.
    let gemm_is_mlas = dispatch_ledger::gemm_backend_is_mlas();
    for family in [KernelFamily::MatMulF32, KernelFamily::GemmF32] {
        assert_eq!(
            dispatch_ledger::effective_backend(family) != Backend::Native,
            gemm_is_mlas,
            "{family} disagrees with what CpuBackend::auto_detect can reach here"
        );
    }
    // Every family the A/B module covers must be a family the plan knows about.
    for family in backend_ab::AB_COVERED {
        assert!(
            KernelFamily::ALL.contains(family),
            "{family} is A/B-covered but missing from the plan"
        );
    }
}
