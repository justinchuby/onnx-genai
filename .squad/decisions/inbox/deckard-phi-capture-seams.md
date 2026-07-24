### 2026-07-23: Eliminate Phi decode CUDA-graph capture seams (Greater + invariant If)
**By:** Deckard
**Branch:** `perf/phi-capture-seams` (off `origin/main` @ `1073404`) — commit `54cc02f`. **Needs review before merge; not merged to main.**

**Scope:** CUDA-graph capture-seam elimination in the executor / CUDA EP. This is NOT a GEMV micro-opt — `matmul_nbits.rs` kernels untouched.

#### Root cause — confirmed
Marsten's Nsight finding reproduced exactly via `ONNX_GENAI_LOG_CAPTURE_SEGMENTS=1`. Phi-4-mini decode splits into **3 captured graphs across 2 per-token seams**, both inside the LongRoPE `rotemb_caches_subgraph`:
- `Greater` node 8 = `Greater(attn_mask_gather_len, 4096)` → **rank-0 scalar bool**. An **eager device seam**: the CUDA binary-predicate kernel (`BinaryPredKernel`) allocated + uploaded + freed broadcast metadata and `synchronize()`d the stream every call, and hard-declined capture.
- `If` node 13 = `If(Greater.out) → (cos_cache, sin_cache)`. A **host seam**: both branches are just two `Constant`s emitting the *full* rotary caches (else/steady = `[4096,48]` fp16 ≈ 393 KB each). Control flow reads `cond` to the host and re-runs the taken branch — re-materializing and re-copying ~786 KB host→device — **every decode step**.

The `If` executor-timer cost Marsten saw is dominated by this per-step branch re-materialization + child-executor overhead, not GPU compute.

#### Fixes (both capture-safe, byte-identical)
1. **`Greater` capturable** (`kernels/pointwise.rs`, `kernels/elementwise.rs`). Persist broadcast metadata in a `BroadcastMetadataCache` (reused across steps — no per-step alloc/upload/free/sync) and advertise `CaptureSupport::Supported` for a stable dtype/shape signature, exactly mirroring the elementwise `BinaryKernel`. Generalized the eligibility gate so a **rank-0 scalar / single-element** predicate output (the LongRoPE `Greater` shape, which `is_fixed_decode_shape` rejected) qualifies. Result: `Greater` folds into the graph → **3 → 2 graphs**.
2. **Loop-invariant `If` specialization** (`executor.rs::exec_if`). General mechanism, not a Phi hardcode: an `If` whose *taken branch has no outer captures* (`required_outer_names(body).is_empty()`) produces outputs that depend only on its own constants, so once taken with a predicate its outputs are already resident in their persistent buffers. The predicate is **still read every step (the correctness guard)**; only the redundant branch re-execution + its host→device cache copies are skipped. Correctness rails:
   - A branch that reads loop-varying captures is **never** memoized (`taken_branch_is_invariant` gate) → no stale/wrong output. Regression test `if_never_memoizes_branch_that_reads_changing_captures`.
   - A predicate flip re-runs the branch; an output-shape change (LongRoPE short↔long at seq 4096) retires the installed graph via the existing `control_flow_seam_invalidated`. Regression test `if_memoizes_invariant_branch_but_reruns_on_predicate_flip`.
   - The memo is cleared before every capture so freshly reallocated buffers are always repopulated during the capture pass.

#### Results (H200, idle GPU, `--steady --warmups 2 --runs 9 --tokens 120`)
- **Graph count: 3 → 2** (`Greater` seam removed; `If` remains a *cheap, memoized* seam).
- **Throughput: ~193 → ~213 tok/s (+~10%, ~0.47 ms/token)**, matched interleaved before/after runs (baseline binary from `origin/main` vs after; e.g. 193.45→211.12, 197.34→213.63, 195.76→215.61). Absolute numbers drift with thermal state, so the *interleaved* delta is the reliable figure. Recovers roughly half the gap to ORT's 229.62 (native was 193.90); remaining ~ -7% is not from these seams.
- **Correctness: byte-identical generated token ids** before vs after over 150 tokens (`diff` clean).
- **Gate:** `CUDA_VISIBLE_DEVICES=N cargo test -p onnx-runtime-ep-cuda --features cuda --lib` → **192 passed / 0 failed**. Full `onnx-runtime-session --features cuda` suite green (incl. `cuda_control_flow_safety`, `control_flow` 21/21 with 2 new tests).
- **Clippy:** lib targets of both touched crates clean under `-D warnings`. Pre-existing repo-wide clippy debt in unrelated GPU **test** files and `executor.rs` (`let mut input_axes`, `manual_is_multiple_of`, `too_many_arguments`) fails `--all-targets` on `origin/main` *before* my changes too — not introduced here.

#### Attribution / honesty
- The **`Greater` fix alone yields ~0 throughput** (a Greater-only build measured at baseline). It is a device seam with no host sync, so removing it doesn't remove a per-token stall — but it is a correct capture-safety improvement and a prerequisite (3→2 graphs). Essentially all of the +10% is the **`If` memoization** removing the per-token ~786 KB cache re-materialization + child-executor dispatch.
- **Partial vs the "collapse 3→1" goal:** I did **not** capture the `If` branch inline into a single graph. Reaching 1 graph would still require reading `cond` each step for the guard (the flip at seq 4096 changes the rotary cache and *must* be caught — skipping the read entirely, which an early ceiling experiment did, is exactly the wrong-branch corruption we must avoid, and was only "correct" because a 120-token window never crosses 4096). Fully removing the per-step `cond` read would need on-device branch selection (a device `Where`/select graph rewrite keeping both caches resident, or a CUDA device-conditional graph node) — a structural, higher-risk change out of scope for this correctness-critical pass. The memoization already captures the dominant recoverable cost with zero correctness risk, so the single-graph rewrite is deferred as a separate, reviewable follow-up rather than rushed.

**Files changed:** `crates/onnx-runtime-ep-cuda/src/kernels/pointwise.rs`, `crates/onnx-runtime-ep-cuda/src/kernels/elementwise.rs` (expose `BroadcastMetadataCache` + helpers `pub(crate)`), `crates/onnx-runtime-session/src/executor.rs`, `crates/onnx-runtime-session/tests/control_flow.rs` (+2 tests).
