# Active execution lane: fixed attribution prototype

**Status: experimental branch only. Do not adopt the public scope API or integrate it into the engine on this evidence.** The bookkeeping reduction is deterministic, but a convincing latency benefit has not been established. The QMoE routed-buffer draft remains gated.

## Question and five arms

The original routed-buffer experiment found an H64 single-row cost of about 1.717µs. This experiment tests whether public lane checkout/return accounts for it. The original no-workspace baseline and old routed candidate are retained unchanged. A new runtime prototype is built once; its unscoped and scoped modes use that exact same binary, with mode selected outside timed loops. Null is an identical copy of the original binary. Model-generation prefix bytes match across old/new probes. All binary, source and probe hashes are retained.

Apple M1 Max/macOS arm64 native CPU, three single-row shapes, five processes per shape/arm, nine batches per process with a pilot-calibrated 100ms CPU target. Five cyclic arm orders balance all positions. The complete build and sweep hold the stable flock shared with the llama.cpp session. Process-start snapshots are retained locally, not continuous foreign-CPU telemetry. No real-model or CUDA throughput claim follows.

## Cost definition

Loop CPU/wall medians and ranges are recorded separately. Total CPU/wall per call adds scope setup and finish to the sum of all nine batches, then divides by total calls; finish is not hidden outside total cost. Lifecycle counters include timer overhead, and getrusage has finite resolution. Scoped finish CPU reads are about 1–4µs for the whole bounded phase, not zero. Each run still drops its output owners before the loop timer stops. All funding equals the declared requirement, and physical trim returns the ledger to zero at process end.

| H / I / threads | scoped vs original, total CPU | scoped vs same-binary unscoped | null vs original | duration / control gates |
|---|---:|---:|---:|---|
| 32 / 32 / 1 | +0.008% | -0.021% | +0.648% | True / True |
| 64 / 64 / 1 | +0.723% | -0.448% | +1.323% | True / True |
| 256 / 256 / 1 | +0.078% | +0.867% | +0.274% | True / False |

All 75 formal complete outputs and argmaxes match. All batches pass the 20ms duration floor. H256 fails the ±5% Relu control gate (null/original about 1.05145); retain that failure and do not interpret its timing as a valid comparison. There was one fixed experiment, not a retry-until-green sequence.

H32 scoped versus new unscoped differs by only about -0.021%; even the new unscoped binary is near the original, unlike the retained old routed binary. That cannot be credited to retention. H64 scoped versus new unscoped is about -0.448%, while its original/null drift is +1.323% and the process ranges overlap. The data does not cleanly attribute or erase the earlier +1.717µs. Do not promote these figures into zero-cost or general speedup claims.

## Prototype contract and tests

The narrow runner exclusively borrows the session, exposes no Deref/DerefMut or mutable-session escape, and retains the existing StepScoped owner. Explicit finish synchronizes/validates before returning it; a pending receipt takes the complete boundary, while an already-consumed receipt requires synchronization but not a second receipt. Capture pins are independent. Failed completion retains backing and poisons subsequent main/sibling prepare/submit/run/capture/replay paths. Consuming provider return errors are not retried. Drop is an observable fallback. No EP allocator API is added.

Eight targeted tests cover fixed-shape reuse, growth, independent capture pins, pending receipts, synchronization/validation failure, owner-consuming return failure, unwind, and failed-session sibling entry points. An 11-run scope observes one actual checkout and one final return through the real CPU provider; the lane remains charged after returning to idle. The full session suite passes 304 tests, two compile-fail doctests pin the exclusive-borrow/no-dereference boundary, strict session all-target Clippy passes, and the engine library compiles. These are Host contracts/doubles, not CUDA-device proof.

An initial teardown fixture incorrectly expected dropping an owning return error to refund capacity. The managed owner correctly quarantined instead. The final test recovers its exact synthetic owner for safe teardown; executor code never retries the consumed owner or waives quarantine. The initial failed log remains labelled separately.

The real generation loop does not use this prototype. Its existing adapter borrows the whole native session; adding this API without a demonstrated benefit would create avoidable lifecycle and integration complexity. A CPU borrowed-state view could make integration safe, but has not been implemented or justified here.

## Decision

Preserve the prototype and full evidence on its experimental branch; keep it out of default merging. Deterministic checkout reduction alone is insufficient justification for the added public lifecycle surface. Continue source/mechanism attribution of the existing public warm-fit path before choosing a production change. Keep the QMoE draft and its original performance gate intact.

## Cleanup

The four hash-verified attribution executable copies and this temporary probe package cache were removed under the shared host lock after the fixed study. Raw records/output files and source/binary hashes remain; original QMoE binaries in their separate folder remain available for ongoing work. cleanup.json records the exact scope and logical sizes.
