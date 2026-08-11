### 2026-08-07: PR #728 round-8 review
**By:** Harry

**Verdict: APPROVE**

## Confirmed sound

- `Executor::node_capture_reason` now has the required precedence: quarantine
  (`capture.rs:566-577`), EP structural policy (`capture.rs:578-593`), resolved-shape
  contract assertion (`capture.rs:595-598`), classifier hard veto
  (`capture.rs:614-623`), kernel warmth (`capture.rs:640-647`), then kernel admission
  (`capture.rs:648`). Thus a disqualified `If`/`Loop`/`Scan`/Sequence node reports
  `HostControlFlowOrSequence` / `HostSeam`, while a resolved plain device node still
  reports `ClassifierDisqualified`.
- The classifier veto remains before `kernel_capture_decline`: the veto returns at
  `capture.rs:614-623`; kernel admission is not consulted until `capture.rs:648`.
  An unconditional-`Supported` kernel therefore cannot re-admit a disqualified node.
- Moving the veto after the assertion does not create a panic path in-tree.
  `ExecutionProvider::plan_capture_region` must decline unresolved shapes and its
  default implementation returns `UnresolvedOutputShape`/`UnresolvedInputShape`
  before admission (`provider.rs:455-477`). The only in-tree override delegates to
  that policy. Any node reaching the assertion and veto has resolved inputs and
  outputs.
- `disqualified_control_flow_node_reports_host_seam_not_device_seam` calls the real
  `node_capture_reason` path. It makes a default-domain node an `If`, proves the
  structural helper recognizes it, adds a growing symbolic output and proves the
  classifier disqualifies it, installs a warmed unconditional-`Supported` kernel,
  and asserts `HostControlFlowOrSequence` plus `HostSeam`. On `9555b354`, the
  veto-first implementation returns `ClassifierDisqualified`, so the test fails;
  on `f310ed9f`, structural precedence makes it pass.
- Sapper's round-7 negative and positive device-node tests are unchanged: the
  disqualified `Identity` remains vetoed despite `Supported`, while the qualified
  `Identity` remains admitted.
- No bypass admission path remains. Every whole-graph or segmented capture schedule
  maps every plan node through `node_capture_reason` (`capture.rs:660-680`);
  warm-seeded planning uses that same schedule; capture recording consumes only its
  captured segments (`dispatch.rs:183-246`). Control-flow child executors call
  `run_scoped`, which explicitly uses `RunMode::Eager` (`run.rs:26-35`), so nested
  bodies do not bypass this audit.

## Validation

- `cargo test -p onnx-runtime-session --lib` — **148 passed, 0 failed**.
- `cargo test -p onnx-runtime-ep-cuda --lib --features cuda` —
  **289 passed, 0 failed, 2 ignored**.
- `git diff --check 9555b354..f310ed9f` — clean.
- GPU/oracle integration tests were not run, as requested.

**APPROVE: the structural seam precedence is restored without reopening the
classifier admission hole or introducing another capture path.**
