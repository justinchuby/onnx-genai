### 2026-08-07: PR #728 round-7 review
**By:** Harry  
**Verdict: REJECT**

## Blocking finding

1. **The “first check” veto corrupts the public seam classification for host-driven nodes.** `Executor::node_capture_reason` returns `ClassifierDisqualified` before invoking the EP structural policy (`crates/onnx-runtime-session/src/executor/capture.rs:556-615`). Therefore an `If`/`Loop`/`Scan` or Sequence node whose input/output references any disqualifying symbol is reported as `ClassifierDisqualified`, whose `path_kind()` is `EagerDeviceSeam` (`capture.rs:100-110`), even though `ExecutionProvider::plan_capture_region` defines that node as `HostControlFlowOrSequence` (`crates/onnx-runtime-ep-api/src/provider.rs:455-477`) and it actually executes through the host path.

   Concrete regression: a LongRoPE-style `If` with a fail-safe-disqualified symbolic output now appears in the public capture-segmentation report as an eager device seam rather than a host seam. This does not numerically corrupt tensors, but it corrupts the structured capture-path contract and any diagnostics/measurement that distinguish device seams from host round trips. Preserve host-control-flow/sequence precedence, then apply the classifier veto before any kernel admission check.

**Fresh revision owner:** Gaff. Roy, Cohaagen, Deckard, Leon, Batty, Sebastian, and Sapper remain locked out for this artifact.

## Confirmed sound

- `plan_capture_segments` is the sole schedule builder and maps every plan node through `node_capture_reason`; the only stream-capture call is reached through that schedule.
- The veto uses the build-time `capture_growing_symbols` produced by `compute_capture_disqualifying_symbols`, identical to kernel-cache/dispatch classification.
- `node_capture_seq_independent` checks every input and every output.
- Warm-seeded capture planning still routes through the veto; clearing the set in the isolated Range seeding test matches that fixture's denylist-mode state.
- The negative mock faithfully models unconditional-`Supported` kernels and would be admitted on `f6b6203c`; the positive companion remains admitted.
- `ClassifierDisqualified` correctly prevents unconditional kernel support from re-admitting a disqualified device kernel.

## Validation

- `cargo test -p onnx-runtime-session --lib` — **133 passed, 0 failed**.
- `cargo test -p onnx-runtime-ep-cuda --lib --features cuda` — **285 passed, 0 failed, 2 ignored**.
- `git diff --check f6b6203c..9555b354` — clean.

**REJECT: the central veto closes the capture-admission hole, but its precedence newly misreports host control-flow/sequence seams as eager device seams.**
