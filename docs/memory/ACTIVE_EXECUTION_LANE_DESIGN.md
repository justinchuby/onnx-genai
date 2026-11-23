# Active execution lane: attribution and ownership proposal

Status: proposed; no implementation or performance result is claimed here.

## Problem and evidence

The [QMoE routed-buffer candidate](../benchmarks/qmoe-routed-buffers/README.md)
puts three internal regions in one governed execution lane. Existing warm-fit
issuer calls are zero, but a single-node H64 decode session still adds about
1.717 microseconds (+1.99%) against the original no-workspace baseline, with a
near-zero identical-baseline null delta. Keep the candidate out of default
merging while resolving that cost. More callbacks or private kernel pools are
not the answer.

The common executor already owns the step workspace during one run and returns
it at the run boundary. An active decode scope could keep that same owner over
a bounded sequence of token runs, then return it on request end, scheduling yield
or phase transition. Retention would not add a new physical grant. It would keep
the original owner and charge, and each operation would still receive only a
workspace view. Returning to idle would preserve the charge until physical trim.

## Existing boundaries

- `crates/onnx-runtime-session/src/executor/run.rs::run_scoped_mode` is one run,
  not a request-wide resource scope.
- `executor/bindings.rs::release_step_workspace` owns the existing return path;
  capture pinning is a separate reason to keep the owner.
- `crates/onnx-genai-engine/src/pipeline/workflow.rs::SessionLeaseGuard` protects
  a logical session writer, not backing ownership or device completion.
- `onnx-runtime-session/src/lib.rs::BlockQuantizedMoeTrafficObserver` provides a
  useful restricted-borrowed-runner and explicit `finish` shape. Do not attach
  resource retention to that observer.

## Minimal ownership direction

A narrow borrowed runner should hold `&mut InferenceSession` and expose only the
prepare/run operations needed by its callers. It must not implement Deref or
DerefMut or return a mutable session, which would permit replacing the borrowed
session while retaining stale scope state. Its lifetime should prevent a second
scope or concurrent mutation through the original session.

Scope retention must be independent of capture pin state. Keep the existing
owner, role checks, checked growth/replacement settlement, inherited control-flow
workspace rules, provider completion and quarantine contracts. Do not add an EP
allocator interface, per-operation grants, a private pool, or a public retention
boolean. A large prefill lane must become reclaimable at the prefill→decode
boundary rather than being implicitly held for the whole request.

Normal exits should explicitly call a fallible finish and report both an inner
run failure and any return failure. Drop is a fallback, never the only normal
cleanup path; failure must remain observable and preserve physical ownership/
charging or provider quarantine. Do not panic or refund before physical release
settles. Ending an active scope does not cancel a valid independent capture pin.

## Real engine integration constraint

`crates/onnx-genai-engine/src/native_decode/backend.rs::NativeLoopAdapter` borrows
the entire `NativeDecodeSession`. Its CPU implementations call `self.session`
through `native_decode/cpu.rs` (ordinary run, device bindings and decode-inline
run paths). Opening a borrow of that field and then lending the whole native
session to the old adapter is not valid. Split the adapter's computation access
into the runner and the remaining decode state/metadata; do not escape the borrow
with a replaceable session or public boolean. The generation-loop entry points
in `native_decode/mod.rs` must establish and finish the scope at actual decode,
yield and phase boundaries, including error/EOS/callback exits.

## Fixed attribution before adoption

Keep the QMoE computation, input and preparation unchanged. Compare the original
no-workspace baseline, the per-run-return routed-buffer candidate, and the same
candidate with one common lane held for N active runs. Record actual checkout,
return and issuer counts, complete outputs/argmaxes, CPU/wall medians and ranges,
null/control behavior, loop time and finish/return time separately. Include the
finish cost in total work; physical trim is a distinct event.

This should establish how much of the measured cost belongs to public lane
roundtrips before committing to an API. A passing result must resolve the
measurable single-row regression rather than merely repeat the zero-issuer
assertion. Test ordinary reuse, growth/failure, phase/yield return and independent
capture/asynchronous ownership behavior. Host doubles can validate contracts;
they do not establish CUDA-device performance or driver correctness.
