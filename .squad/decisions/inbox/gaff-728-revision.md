### 2026-08-07: PR #728 round-8 revision — host-seam precedence over classifier veto
**By:** Gaff (revision), finalized by coordinator (commit/push)
**Artifact:** PR #728 branch squad/elementwise-capture-seqindep
**HEAD:** f310ed9f (was 9555b354)

## Finding fixed (Harry round-7 REJECT, .squad/decisions/inbox/harry-728-review7.md)
Sapper's round-7 central capture veto ran as the FIRST check in
`Executor::node_capture_reason` (capture.rs), BEFORE the EP structural policy
`self.ep.plan_capture_region(...)`. Consequence: an `If`/`Loop`/`Scan`/Sequence
node that ALSO references a disqualifying growing symbol was reported with
`SeamReason::ClassifierDisqualified` → `path_kind() == EagerDeviceSeam`, instead
of the correct `HostControlFlowOrSequence` → `HostSeam`. Numerically safe (both
paths stay eager / uncaptured) but corrupts the public capture-segmentation
contract and any diagnostics/measurement that distinguish host round-trips from
eager device seams.

## Fix (surgical reorder only)
`crates/onnx-runtime-session/src/executor/capture.rs`, `node_capture_reason`:
moved the classifier-veto block from the top of the function to AFTER the
`ep.plan_capture_region(...)` structural decline (and after the
`inputs_resolved && outputs_resolved` assertion) and BEFORE the kernel lookup +
`kernel_capture_decline(...)`.

New precedence order in `node_capture_reason`:
1. quarantine (`CaptureRecordingFailed`) — unchanged position;
2. structural: `ep.plan_capture_region` → `HostControlFlowOrSequence` /
   `UnresolvedInput/OutputShape`;
3. **classifier veto → `ClassifierDisqualified`** (NEW position);
4. kernel admission: `kernel_capture_decline`.

The veto still runs strictly BEFORE `kernel_capture_decline`, so an
unconditional-`Supported` kernel can NEVER re-admit a disqualified device node
(the round-7 hole stays closed). `SeamReason::ClassifierDisqualified` and its
`EagerDeviceSeam` mapping are unchanged — a genuinely device-kernel disqualified
node correctly remains an eager device seam; only its precedence relative to the
host/structural seams changed.

## Test
`crates/onnx-runtime-session/src/executor/tests.rs`:
`disqualified_control_flow_node_reports_host_seam_not_device_seam` — turns a
fixture node into an `If`, makes it classifier-disqualified via a growing symbol,
warms an unconditional-`Supported` kernel, and asserts the decline seam is
`HostControlFlowOrSequence` / `HostSeam`, NOT `ClassifierDisqualified`.
- fail-pre (HEAD 9555b354, veto-first): returns `ClassifierDisqualified` /
  `EagerDeviceSeam` → assertion fails.
- pass-post (reordered): returns `HostControlFlowOrSequence` / `HostSeam` → passes.
Sapper's round-7 tests unchanged and still green:
`classifier_disqualified_node_is_vetoed_despite_supported_kernel` (plain device
node still vetoed with `ClassifierDisqualified`) and
`classifier_qualified_node_with_supported_kernel_is_admitted`.

## Validation
- `cargo test -p onnx-runtime-session --lib`: 148 passed, 0 failed.
- Targeted 3 veto tests: all pass.
- `cargo fmt --all --check`: clean.

## Lockout
Locked out of PR #728 after this round if rejected: Roy, Cohaagen, Deckard, Leon,
Batty, Sebastian, Sapper, Gaff. Fresh reviser pool remaining: Pris, Rachael,
Tyrell, Bryant.
