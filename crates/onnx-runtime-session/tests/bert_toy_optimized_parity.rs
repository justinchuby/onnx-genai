//! Opt-IN optimization parity for the `bert_toy` model.
//!
//! Companion to `bert_toy_conformance.rs`, which runs the SAME model with
//! optimization OFF (the default) and matches the committed onnxruntime CPU
//! reference to `max_abs = 1.192e-7`. Here we exercise the session's newly
//! activated `optimize` pipeline stage (loader → **optimize** → re-infer shapes
//! → compile → allocate), which is opt-in via the generic `"optimization"`
//! session option and defaults to off.
//!
//! Two levels are validated end-to-end:
//!
//! * `"basic"` — constant folding + dead-node elimination. These passes are
//!   structure-preserving (no new op types), so the executor sees a subset of
//!   the loaded graph and the numerics must stay identical. This is the clean
//!   proof that the wiring — including the post-optimization shape re-inference
//!   — does not perturb results.
//!
//! * `"all"` — the full device-independent pipeline, which additionally runs
//!   operator **fusion**. On this model fusion collapses each LayerNorm
//!   decomposition into a single schema-conformant
//!   `com.microsoft::LayerNormalization`, every MatMul+Add into
//!   `com.microsoft::FusedMatMulBias`, each self-attention SDPA core into a
//!   `com.microsoft::FusedAttention`, and each feed-forward exact-GELU `Erf`
//!   decomposition into a `com.microsoft::Gelu`, all backed by CPU kernels, so
//!   the fused graph runs end-to-end and matches the reference. See
//!   `full_optimization_fusion_path_matches_reference_and_default` (numerics)
//!   and `full_optimization_actually_fuses_layernorm_and_matmul_bias` (proof the
//!   fusions actually fire on the loaded graph).
//!
//! Nothing here special-cases "bert" in library code — the model is a generic
//! fixture and `"optimization"` is a generic, model-agnostic option.

use std::path::{Path, PathBuf};

use onnx_runtime_session::{InferenceSession, SessionError, Tensor};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bert_toy")
}

fn read_bin(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()))
}

fn i64_input(name: &str) -> Tensor {
    let bytes = read_bin(name);
    let data: Vec<i64> = bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(data.len(), 8, "{name} expected 8 int64 values");
    Tensor::from_i64(&[1, 8], &data).unwrap()
}

fn f32_reference(name: &str) -> Vec<f32> {
    read_bin(name)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Build a session at the given optimization level (`None` = default/off) and
/// run the three committed inputs, returning the two model outputs.
fn run_bert(optimization: Option<&str>) -> Result<Vec<Tensor>, SessionError> {
    let model_path = fixture_dir().join("model.onnx.textproto");
    let mut builder = InferenceSession::builder().model(&model_path);
    if let Some(level) = optimization {
        builder = builder.option("optimization", level);
    }
    let mut session = builder.build()?;

    let input_ids = i64_input("input_ids.bin");
    let token_type_ids = i64_input("token_type_ids.bin");
    let input_mask = i64_input("input_mask.bin");
    session.run(&[
        ("input_ids", &input_ids),
        ("token_type_ids", &token_type_ids),
        ("input_mask", &input_mask),
    ])
}

fn max_abs(actual: &[f32], reference: &[f32]) -> f32 {
    assert_eq!(actual.len(), reference.len(), "element count mismatch");
    actual
        .iter()
        .zip(reference)
        .fold(0.0f32, |m, (&a, &r)| m.max((a - r).abs()))
}

const CASES: [(&str, &str, &[usize]); 2] = [
    ("prediction_scores", "prediction_scores.bin", &[1, 8, 99]),
    ("seq_relationship_score", "seq_relationship_score.bin", &[1, 2]),
];

/// `"basic"` optimization (constant folding + dead-node elimination) is
/// numerically faithful: it matches BOTH the onnxruntime reference and the
/// optimization-off output to the conformance tolerance. This validates the
/// activated pipeline — including post-optimization shape re-inference — does
/// not perturb results.
#[test]
fn basic_optimization_matches_reference_and_default() {
    // Same tolerance rationale as bert_toy_conformance.rs (numpy.allclose).
    const ATOL: f32 = 2e-3;
    const RTOL: f32 = 2e-3;

    let opt_off = run_bert(None).expect("build+run with optimization off");
    let opt_basic = run_bert(Some("basic")).expect("build+run with optimization=basic");

    assert_eq!(opt_basic.len(), 2, "expected 2 model outputs");

    let mut overall_ref = 0.0f32;
    let mut overall_vs_off = 0.0f32;

    for (i, (label, bin, shape)) in CASES.iter().enumerate() {
        let basic = opt_basic[i].to_vec_f32();
        let off = opt_off[i].to_vec_f32();
        let reference = f32_reference(bin);

        assert_eq!(
            &opt_basic[i].shape, shape,
            "{label}: shape mismatch with optimization=basic (got {:?}, expected {shape:?})",
            opt_basic[i].shape
        );

        let vs_ref = max_abs(&basic, &reference);
        let vs_off = max_abs(&basic, &off);
        overall_ref = overall_ref.max(vs_ref);
        overall_vs_off = overall_vs_off.max(vs_off);

        eprintln!("{label}: opt=basic vs reference max_abs = {vs_ref:.3e}, vs opt-off max_abs = {vs_off:.3e}");

        let n_fail = basic
            .iter()
            .zip(reference.iter())
            .filter(|&(&a, &r)| (a - r).abs() > ATOL + RTOL * r.abs())
            .count();
        assert_eq!(
            n_fail, 0,
            "{label}: {n_fail} elements exceed atol={ATOL:.0e}+rtol={RTOL:.0e} vs reference",
        );
    }

    eprintln!(
        "bert_toy opt=basic PARITY PASS: vs reference max_abs = {overall_ref:.3e}, \
         vs opt-off max_abs = {overall_vs_off:.3e}"
    );

    // `basic` (constant-fold + DCE) is structure-preserving, so it must be
    // byte-identical to the optimization-off graph. Lock it in as an assertion.
    assert_eq!(
        overall_vs_off, 0.0,
        "opt=basic must be byte-identical to opt-off (structure-preserving passes)"
    );
}

/// `"all"` optimization runs the full device-independent pipeline, including
/// operator **fusion**. On this model fusion collapses each LayerNorm
/// decomposition into a single schema-conformant `com.microsoft::LayerNormalization`
/// (inputs `[X, Scale, B]` + `axis`/`epsilon` attributes) and every
/// `MatMul + Add(bias)` into `com.microsoft::FusedMatMulBias`, both of which now
/// have CPU kernels. The fused path therefore executes end-to-end and must match
/// the reference to the same tolerance as the conformance / `"basic"` checks.
///
/// Unlike `MatMul + Add → FusedMatMulBias` (byte-identical to the original ops),
/// the fused `LayerNormalization` kernel accumulates mean/variance in a single
/// pass, so its result differs from the 10-op decomposition by a few ULPs. The
/// fused `"all"` graph is therefore **not** byte-identical to opt-off — it is
/// close to it, and (the load-bearing check) close to the onnxruntime reference,
/// both well within the conformance tolerance. We assert against the reference,
/// not against opt-off byte-identity.
///
/// (Note: `bert_toy`'s feed-forward blocks use GELU/`Erf`, not `Relu`, so the
/// `MatMul + Add + Relu → FusedGemm` pattern never fires here. That kernel now
/// exists and is validated by the synthetic end-to-end parity test in
/// `fused_gemm_parity.rs`, since no model in this suite contains the Relu form.)
#[test]
fn full_optimization_fusion_path_matches_reference_and_default() {
    // Same tolerance rationale as bert_toy_conformance.rs (numpy.allclose).
    const ATOL: f32 = 2e-3;
    const RTOL: f32 = 2e-3;
    // The vs-opt-off DRIFT ceiling is independent of (and far tighter than) the
    // vs-reference conformance tolerance. LayerNorm AND the new SDPA/attention
    // fusion each perturb numerics by only a few ULPs (actual combined drift
    // ~1.4e-7), so bound it at 1e-5 — tight enough to catch a subtle future
    // numeric regression, with comfortable headroom.
    const DRIFT_ATOL: f32 = 1e-5;

    let opt_off = run_bert(None).expect("build+run with optimization off");
    let opt_all = run_bert(Some("all")).expect("build+run with optimization=all");

    assert_eq!(opt_all.len(), 2, "expected 2 model outputs");

    let mut overall_ref = 0.0f32;
    let mut overall_vs_off = 0.0f32;

    for (i, (label, bin, shape)) in CASES.iter().enumerate() {
        let all = opt_all[i].to_vec_f32();
        let off = opt_off[i].to_vec_f32();
        let reference = f32_reference(bin);

        assert_eq!(
            &opt_all[i].shape, shape,
            "{label}: shape mismatch with optimization=all (got {:?}, expected {shape:?})",
            opt_all[i].shape
        );

        let vs_ref = max_abs(&all, &reference);
        let vs_off = max_abs(&all, &off);
        overall_ref = overall_ref.max(vs_ref);
        overall_vs_off = overall_vs_off.max(vs_off);

        eprintln!("{label}: opt=all vs reference max_abs = {vs_ref:.3e}, vs opt-off max_abs = {vs_off:.3e}");

        let n_fail = all
            .iter()
            .zip(reference.iter())
            .filter(|&(&a, &r)| (a - r).abs() > ATOL + RTOL * r.abs())
            .count();
        assert_eq!(
            n_fail, 0,
            "{label}: {n_fail} elements exceed atol={ATOL:.0e}+rtol={RTOL:.0e} vs reference",
        );
    }

    eprintln!(
        "bert_toy opt=all PARITY PASS: vs reference max_abs = {overall_ref:.3e}, \
         vs opt-off max_abs = {overall_vs_off:.3e}"
    );

    // The load-bearing correctness bound: the fused graph matches the committed
    // onnxruntime reference to the conformance tolerance.
    assert!(
        overall_ref < ATOL,
        "opt=all must match the reference within atol={ATOL:.0e} (got {overall_ref:.3e})"
    );
    // LayerNorm and SDPA/attention fusion change numerics by only a few ULPs, so
    // opt=all stays extremely close to opt-off — but is NOT byte-identical (the
    // fused kernels reduce differently). Bound the drift tightly (1e-5, ~2 orders
    // above the observed ~1.4e-7) so a subtle future numeric regression is
    // caught, without loosening the vs-reference conformance tolerance.
    assert!(
        overall_vs_off < DRIFT_ATOL,
        "opt=all must stay within drift atol={DRIFT_ATOL:.0e} of opt-off (got {overall_vs_off:.3e})"
    );
}

/// End-to-end proof the schema-aware fusions actually **fire on the real loaded
/// `bert_toy` graph** — not just on synthetic unit fixtures. Loads the model
/// through the production loader, runs the same `"all"` optimizer pipeline the
/// session uses (constant-fold → DCE → fusion), and asserts the fused op counts.
///
/// This is the coverage gap Deckard's advisory A1 flagged: the LayerNorm
/// schema-aware path was previously exercised only by unit tests because
/// `bert_toy`'s LayerNorm uses two distinct `Sub(x, mean)` nodes, which the old
/// linear-chain matcher rejected. The DAG-aware matcher now fuses it.
#[test]
fn full_optimization_actually_fuses_layernorm_and_matmul_bias() {
    use onnx_runtime_optimizer::{
        ConstantFolding, DeadNodeElimination, OpFusion, PassContext, run_passes,
    };

    let model_path = fixture_dir().join("model.onnx.textproto");
    let mut graph = onnx_runtime_loader::load_model(&model_path).expect("load bert_toy");

    let count = |g: &onnx_runtime_ir::Graph, op: &str| -> usize {
        g.nodes.values().filter(|n| n.op_type == op).count()
    };

    // Before fusion: raw decomposition, no fused ops.
    let ln_chains_before = count(&graph, "ReduceMean");
    assert_eq!(count(&graph, "LayerNormalization"), 0);
    assert_eq!(count(&graph, "FusedMatMulBias"), 0);
    // Each LayerNorm region has two ReduceMean ops (mean + variance).
    assert_eq!(
        ln_chains_before % 2,
        0,
        "expected an even ReduceMean count (2 per LayerNorm region)"
    );
    let expected_layernorms = ln_chains_before / 2;

    let passes: Vec<Box<dyn onnx_runtime_optimizer::OptimizationPass>> = vec![
        Box::new(ConstantFolding),
        Box::new(DeadNodeElimination),
        Box::new(OpFusion::new()),
    ];
    run_passes(&mut graph, &passes, &PassContext::new()).expect("run optimizer passes");

    let ln_after = count(&graph, "LayerNormalization");
    let mm_after = count(&graph, "FusedMatMulBias");
    let attn_after = count(&graph, "FusedAttention");
    eprintln!(
        "bert_toy fusion counts: LayerNormalization={ln_after} (expected {expected_layernorms}), \
         FusedMatMulBias={mm_after}, FusedAttention={attn_after}"
    );

    // Every LayerNorm region fused, leaving no stray ReduceMean behind.
    assert_eq!(
        ln_after, expected_layernorms,
        "every LayerNorm decomposition must fuse to one LayerNormalization"
    );
    assert_eq!(
        count(&graph, "ReduceMean"),
        0,
        "no ReduceMean should survive: all belonged to fused LayerNorms"
    );
    // Regression guard: MatMul+Add fusion must be unaffected (still 32×).
    assert_eq!(
        mm_after, 32,
        "MatMul+Add → FusedMatMulBias must stay at 32 fusions"
    );

    // Each of the 5 self-attention blocks' SDPA core (QKᵀ·scale + mask →
    // Softmax → ·V) fuses into one com.microsoft::FusedAttention, leaving no
    // standalone Softmax behind.
    assert_eq!(
        attn_after, 5,
        "every SDPA core must fuse to one FusedAttention (bert_toy has 5)"
    );
    assert_eq!(
        count(&graph, "Softmax"),
        0,
        "no Softmax should survive: all belonged to fused attention cores"
    );

    // Each feed-forward block's exact-GELU `Erf` decomposition
    // (0.5·x·(1 + Erf(x / √2))) fuses into one com.microsoft::Gelu, leaving no
    // standalone Erf behind. bert_toy has 6 GELUs (one per FFN block).
    let gelu_after = count(&graph, "Gelu");
    eprintln!("bert_toy fusion counts: Gelu={gelu_after}");
    assert_eq!(
        gelu_after, 6,
        "every GELU decomposition must fuse to one Gelu (bert_toy has 6)"
    );
    assert_eq!(
        count(&graph, "Erf"),
        0,
        "no Erf should survive: all belonged to fused GELUs"
    );
}
