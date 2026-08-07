# Sapper — PR #728 round-7 revision: CENTRAL capture veto

**Agent:** Sapper (Rust/CUDA)
**Branch:** squad/elementwise-capture-seqindep (PR #728)
**Date:** 2026-08-07

## The finding (Harry round-7 REJECT)
Classifier-disqualified nodes could STILL enter CUDA-graph capture because many
CUDA kernels return `CaptureSupport::Supported` unconditionally and never
override `set_capture_seq_independent` (they use the trait default no-op). Harry
named three concrete bypass kernels — `UnaryMathKernel`, `NotKernel`,
`BitwiseNotKernel` — each launching `grid_for(numel())`. A classifier-disqualified
GROWING rank-1 value consumed by any of them enters capture with a grid baked
from the warmed extent → silent decode corruption. A `grep` shows dozens more
kernels returning unconditional `Supported` (activations, attention, matmul,
normalization, gather, cast, hardmax, log_softmax, …), so per-kernel patching of
the three named ops is fragile whack-a-mole.

## Design chosen: CENTRAL HARD VETO (approach (b)), not per-kernel
The capture-admission chokepoint is `Executor::node_capture_reason`
(crates/onnx-runtime-session/src/executor/capture.rs), the single per-node
function `plan_capture_segments` maps over EVERY plan node (capture.rs:676),
itself the only capture-admission path (run.rs:525). I made the build-time
classifier authoritative there: BEFORE consulting the kernel's own
`capture_support()`, if the node's verdict is disqualified
(`!node_capture_seq_independent(&self.graph, node, &self.capture_growing_symbols)`),
capture is declined with a new `SeamReason::ClassifierDisqualified`.

**Why central, not per-kernel (approach (a)):** the per-node verdict originates
from executor/classifier state (`self.capture_growing_symbols`, the build-time
disqualifying-symbol set computed by `compute_capture_disqualifying_symbols`),
which is ALREADY computed independently of any kernel. `node_capture_reason` is an
`Executor` method with direct access to `self.graph` + `self.capture_growing_symbols`,
so the verdict reaches the chokepoint with ZERO new plumbing and WITHOUT depending
on any kernel having stored the flag. Approach (a) (a `capture_seq_independent_hint`
trait accessor) still relies on each kernel storing the flag — the three bypass
kernels don't — so it reintroduces whack-a-mole. Approach (b) closes the ENTIRE
class in one place; kernels are now strictly advisory.

The veto is strictly ADDITIVE: a node is capture-eligible iff (classifier says
seq-independent) AND (kernel says Supported for the shape). Over-declining is
correctness-safe (extra eager nodes); under-declining is corruption — so it biases
to veto. It is identical to the gate the signature-gated pointwise/elementwise/
bitwise/prelu/silu family already applies via `capture_shape_eligible(...)`; the
veto simply applies that same verdict to EVERY node centrally.

## Exact files / functions changed
- **crates/onnx-runtime-session/src/executor/capture.rs**
  - `enum SeamReason`: added `ClassifierDisqualified` variant (+ doc).
  - `SeamReason::path_kind`: maps `ClassifierDisqualified` → `EagerDeviceSeam`.
  - `Executor::node_capture_reason` (capture.rs:556): added the central veto as the
    FIRST check (capture.rs:562-583), before quarantine / capture-region / kernel
    checks.
- **crates/onnx-runtime-session/src/executor/tests.rs**
  - New mock `UnconditionalCaptureKernel` (returns `Supported`, no flag override —
    mirrors the bypass kernels).
  - New helper `build_identity_capture_fixture()`.
  - New tests (below).
  - `warm_decode_seeding_admits_previously_unresolved_capture_safe_node`: added
    `exec.capture_growing_symbols.clear();` to isolate the warm-decode SEEDING
    transition from the (separately-tested) classifier veto. Under the default
    fail-safe classifier the Range output's untraceable data-dependent length is
    disqualifying, which would otherwise mask the unresolved-shape seam this test
    asserts. The test's primary post-seeding assertion is unchanged.

## New tests + fail-pre / pass-post evidence
- `classifier_disqualified_node_is_vetoed_despite_supported_kernel` — builds a real
  Executor, marks a node's OUTPUT shape with a GROWING symbol, wires it to
  `UnconditionalCaptureKernel`, and asserts `node_capture_reason` returns
  `Some(SeamReason::ClassifierDisqualified)`.
  - **fail-pre:** with the veto block removed (old code), the node is ADMITTED
    (`None`) — every structural check passes and the kernel says Supported → test
    FAILS: `classifier-disqualified node must be declined for capture`.
  - **pass-post:** with the veto → test PASSES.
- `classifier_qualified_node_with_supported_kernel_is_admitted` — a seq-independent
  node with the same unconditional-Supported kernel is still admitted (`None`).
  Passes both pre and post (veto is strictly additive).

## Unit-test results
- `cargo test -p onnx-runtime-session --lib` → **147 passed; 0 failed** (2 new +
  the isolated warm-decode test).
- `cargo test -p onnx-runtime-ep-cuda --lib --features cuda` → **289 passed;
  0 failed; 2 ignored** (host tests; unchanged crate).
- `cargo build --release -p onnx-runtime-session -p onnx-runtime-ep-cuda
  --features cuda` → OK.
- `cargo fmt --all --check` → clean.

## Self-audit vs "ANY remaining path"
`plan_capture_segments` (capture.rs:657, called from run.rs:525) is the SOLE
capture-admission chokepoint; it routes every plan node through
`node_capture_reason`, whose FIRST check is the veto. `kernel_capture_decline`
(the only `capture_support()` consumer) runs strictly AFTER the veto, so no
kernel's opinion can re-admit a disqualified node. There is no other path that
places a node into a captured segment. The classifier is now authoritative at the
chokepoint; kernels are advisory only.
