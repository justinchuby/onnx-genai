# Decisions

> Current decision ledger. Full prior history through 2026-07-20T13:35Z is preserved in
> `.squad/decisions/archive/2026-07-20T13-35-00Z-decisions-pre-multistream.md`.

## 2026-07-20 — CPU decode: resident pool and guarded GQA row parallelism

### Keep persistent M=1 decode-pool residency
**By:** Sapper; reviewed by Luv 🟢  
**What:** Run the whole native CPU M=1 forward inside one bounded decode-pool `install`, using a worker-local, nested, panic-safe RAII residency guard so each MatMulNBits call executes inline rather than reinstalling the same pool. `ONNX_GENAI_CPU_DECODE_THREADS=0`, prefill, default-feature-off, and CUDA behavior remain unchanged. Landed on main as `cbacb75`.  
**Why:** Qwen2.5-0.5B int4 decode improved about 3–6% with bit-identical tokens. This proves install crossings were avoidable but not the dominant remaining cost. Luv verified TLS isolation, Rayon semantics, deadlock safety, feature gates, and the CPU/build test matrix.

### Parallelize sufficiently large CPU GQA attention rows
**By:** Roy; reviewed by Luv 🟢  
**What:** Parallelize independent `(batch, query_head, query_sequence)` rows with one Rayon fork-join only above a 163,840 `row × key × head-dimension` work guard; retain serial execution below it. Each task owns a disjoint output row and private score buffer while preserving each row's reduction order. Landed on main as `c391327`.  
**Why:** Short decode regressed when parallelism was unconditional. Guarded parallelism improved 512-token decode throughput by 8.6%, reduced profiled GQA time by 13.9%, and cut 225-token prefill GQA time by 88.3%, with bit-identical 1-thread/8-thread greedy output. A future coverage follow-up may force exact serial/parallel comparison for a large ragged batch.

### Retain Tier-A GQA KV copy cleanup, defer shared append-only KV
**By:** Roy; regression coverage by Pris  
**What:** Borrow contiguous f32 past caches, remove a redundant owned clone, and replace scalar cache materialization loops with contiguous slice copies. Keep attention math and the SSA output contract unchanged. Pris added f16-widening and ragged-per-batch cache-materialization regressions.  
**Why:** The cleanup is bit-identical and removes avoidable work, but measured end-to-end decode was neutral within noise. True O(1)-append shared KV requires runtime aliasing/lifecycle changes and remains deferred.

### Do not land the decode fork-join granularity prototype
**By:** Deckard  
**What:** Revert the coarser 8/12-task MatMulNBits prototype and profiling probes; no commit landed.  
**Why:** Long runs regressed 7.1–8.4%. Post-residency profiling showed serial GQA at about 20.58 ms/token exceeded MatMulNBits at about 15.51 ms/token, so reducing projection task count removed steal slack rather than solving the dominant bottleneck. Revisit only as graph-level projection fusion, after GQA.

## 2026-07-20 — CUDA fused flash attention

### Fuse standard Attention only on measured-winning shapes
**By:** Rachael; reviewed by Chew 🟡  
**What:** Add an NVRTC tiled online-softmax backend behind `AttentionKernel`, including f16 WMMA with f32 accumulation and scalar f32/f16/bf16 support for MHA/GQA/MQA, causal/non-causal attention, and additive mask planes. Auto dispatch retains Phase-2a for decode, `D>128`, unsupported layouts/features, and measured-slower long spans. Landed on main as `a67b7a5`.  
**Why:** H200 f16 S512 improved about 1.53–1.60× and removed 48 MiB score scratch; S2048 regressed heavily when forced, so fallback is part of the design. Chew found the online-softmax merge, WMMA masking/synchronization, numerics, and dispatch sound. Non-blocking coverage remains for explicit Auto fallback gates, non-multiple-of-16 f16 head dimensions, and per-batch/per-head masks.

### Fuse GroupQueryAttention prefill with distinct physical and causal origins
**By:** Bryant, corrected by Rachael after Chew rejection; final review Chew 🟢  
**What:** Reuse the shared flash kernel behind `com.microsoft::GroupQueryAttention` for measured-winning prefill. Cache append and implicit RoPE use `total_length - key_sequence_length`; attention causal masking uses the distinct query start `total_length - query_sequence_length`. The final parity matrix covers 40 scenarios across f32/f16/bf16, MHA/GQA/MQA, fresh/cached/ragged, RoPE, local window, softcap, generic non-WMMA routing, large scores, unequal Q/K lengths, and Auto fallback. Landed on main as `94fa2b6`.  
**Why:** Bryant's first revision incorrectly reused the K append origin for queries when `Sq != Sk`; Chew rejected it and locked that artifact. Rachael's revision made the failing `Sq=2,Sk=4` case pass, tightened tolerances, and preserved exact present K/V. H200 fresh Q512 is about 1.31× faster with 48 MiB scratch saved; cached/large slower shapes fall back. The corrected artifact is approved and no active lockout remains.

## 2026-07-20 — Issue #40 Phase 1 distributed-runtime foundation

### Slice 1a: shared protocol trace + ticketed non-blocking host pressure
**By:** Tyrell; reviewed by Gaff 🟡  
**What:** Add the unpublished `onnx-runtime-protocol-trace` crate with public protocol envelopes/identities and a conformance-only independent `ReplayChecker`; add `HostGovernor` ticketed pressure accounting to `onnx-genai-scheduler`. All state transitions and trace linearization points commit under one short ledger lock; waits occur only on ticket-local condition variables after capacity is atomically charged. Landed on main as `0d1d265`.  
**Why:** The implementation conforms to `PressureProtocol.tla` invariants through an independent deterministic replay campaign and snapshot invariant checks. Gaff approved with two non-blocking issues—terminal-entry reaping and cancel-granted wake-after-unlock—which were folded into slice 1b. The TLC model gate is CI-deferred because Java/TLA tooling is unavailable locally.

### Slice 1b: Communicator + in-process backend + BufferOwnership registry
**By:** Tyrell; reviewed by Gaff 🟢  
**What:** Add unpublished `onnx-runtime-comm` with the async `Communicator` trait, synchronous reference `InProcessCommunicator`, and one-lock `OwnershipRegistry` over read/write lease sets. Dropping an operation handle detaches but does not release storage; terminal completion/abort releases leases, and freed allocation IDs remain tombstoned to prevent reuse/ABA. Reuse the slice-1a trace framework and independently replay `BufferOwnership` events. Landed on main as `e4d2883`.  
**Why:** Gaff verified exactly-one-owner, conflict, release, transfer, generation/ABA, non-blocking-lock, linearization, barrier, mailbox, and deterministic-conformance obligations. Slice 1b also reaps terminal pressure entries and moves all pressure wakeups after unlock. Non-blocking follow-ups for 1c include abort waking barrier waiters, barrier-map cleanup, and documenting tombstone growth.

### Slice 1c: one topology-wide collective ordering authority — IN PROGRESS
**By:** Tyrell  
**What:** Implement direct host rendezvous collectives behind a shared `CollectiveSequencer`; keep canonical submit order independent per communicator group, use one slot for count exchange plus all-to-all-v data, freeze reduction member order with checked arithmetic and per-contribution f16/bf16 rounding, and bound free tombstones with an exact window plus allocator-proven epoch floors.  
**Why:** This maps to `CollectiveOrdering.tla`: ranks may progress asynchronously without divergent order, groups do not acquire a false global enqueue order, completion stays rank-local, and abort freezes submissions before backend wakeup. This slice is not yet landed.

### Phase-1 deferred gates and remaining phases
**By:** Scribe  
**What:** Keep the TLC model gate CI-deferred. After 1c, Phase-1 slice 1d weight residency remains pending; issue #40 Phases 2–4 remain pending.  
**Why:** The landed Rust conformance harnesses provide deterministic implementation-side evidence, but do not replace the configured CI model check or the remaining distributed-runtime roadmap.

## 2026-07-20 — Issue #40 collective ordering completion

### Land slice 1c with serialized abort wakes and broad equivalence coverage
**By:** Tyrell; reviewed by Gaff 🟢  
**What:** Land all seven in-process collectives behind one canonical per-group `CollectiveSequencer`, deterministic member-order reduction, additive independent replay checking, bounded allocation tombstones, and rank-local completion. Abort now holds each rendezvous mutex while notifying its paired condition variable, closing the review's notify-before-park race. Distributed-equals-single-device bitwise coverage spans all_reduce, reduce_scatter, all_gather, broadcast, all_to_all, and all_to_all_v. Landed as `2ffb4e4` with follow-up `128440d`.  
**Why:** Gaff found the architecture and TLA refinement sound but blocked the original revision on a rare abort-path lost wakeup. Tyrell's deterministic waiter gate proved the fix, all comm/trace/scheduler suites passed, and the broadened equivalence matrix preserves fixed-rank-order determinism. TLC remains CI-deferred.

## 2026-07-20 — CUDA graph M4 capture-safety

### Own the CUDA graph lifecycle and exercise native decode replay
**By:** Rachael and Deckard; replay coverage by Pris; reviewed by Chew 🟢  
**What:** Serialize one CUDA graph lifecycle inside `CudaRuntime`, capture/replay only on its dedicated stream, invalidate on generation/binding lifecycle changes, and split capture-end from instantiate so failed instantiation cannot leak the intermediate `CUgraph`. Native decode remains flag-gated and strict-audit: unsupported graphs fall back eagerly. A capture-safe synthetic decoder proves token-exact eager/replay parity across reset, stable addresses, O(1) scalar uploads, two captures, sixteen replays, and zero fallbacks. Landed as `637e247`, `5470c01`, `dd2d807`, and `4755575`.  
**Why:** The first Qwen test exercised only fallback and was rejected as replay evidence. The final synthetic integration test executes the real `NativeDecodeSession::decode_cuda` state machine and resolved the M4 decode-loop review blocker without weakening the all-kernel capture audit.

### Gate MatMulNBits M=1 capture safety to the proven decode path
**By:** Bryant  
**What:** Remove trailing GEMV synchronizations and advertise MatMulNBits capture compatibility only after a successful no-`g_idx`, M=1 decode warmup; prefill, grouped-index, unwarmed, and configuration-changing paths remain ineligible. Runtime D2H helpers explicitly order after the EP stream. Landed as `a210703`.  
**Why:** The proven GEMV path is allocation-free, D2H-free, and synchronization-free, while the excluded paths dequantize, allocate, or validate on the host.

### Make fixed-shape GQA decode capture-safe with detect-before-consume metadata guards
**By:** Deckard, Rachael, and Bryant; reviewed by Chew 🟢  
**What:** Persist GQA scratch and remove the trailing stream sync (`dcb4f1b`); move advancing decode metadata reads and derived lengths on-device (`77829b9`); preserve warmup rejection and add on-device replay bounds checks with sentinel no-write behavior (`82c249d`). The final shared sticky error latch poisons subsequent replay steps after any violation and is polled immediately after logits D2H, before token consumption; explicit graph reset clears it. Landed final as `ca50bae`.  
**Why:** Earlier revisions were rejected for silent clamping and then for allowing a later valid replay to resume over a skipped KV row. The final detect-before-consume latch makes invalid metadata a hard, deterministic failure while valid fixed-capacity f32 one-token replay remains byte-identical and allocation-free.

### Make four normalization variants capture-safe
**By:** Roy; reviewed by Chew 🟢  
**What:** Remove trailing synchronizations from LayerNormalization, RMS/SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization, and SkipLayerNormalization. Keep SkipSimplified broadcast metadata in a mutex-protected, shape-keyed persistent cache and permit capture only after successful single-group warmup. Landed as `6184d82`.  
**Why:** The warmed decode paths now have stable metadata and no per-step allocation, free, upload, host read, or stream synchronization; the full CUDA suite and direct capture/replay byte-parity test passed.

### Bind elementwise capture eligibility to exact warmed signatures
**By:** Sapper and Deckard; reviewed by Chew 🟢  
**What:** Make supported unary and binary floating-point decode kernels capture-safe using persistent broadcast metadata and removed trailing synchronizations. Replace the initial boolean eligibility gate with mutex-protected exact dtype/entry and shape signatures; prefill, i64, errors, and signature changes remain ineligible. Landed final as `85b6f4e`.  
**Why:** Chew rejected the boolean gate because a warmed kernel could later execute a different dtype or shape during capture. Exact signatures close that TOCTOU while preserving numerics and the approved persistent-metadata design.

## 2026-07-21 — CUDA graph M4 end-to-end validation

### Real Qwen2.5 int4 decode captures with zero fallbacks
**By:** Rachael; reviewed by Chew; smoke correction by Pris 🟢  
**What:** Seed unresolved persistent external input/output physical shapes only during capture, keeping eager shape resolution and binding-signature invalidation intact. Constant/Shape metadata reuse and capture-safe integer Sub, ReduceSum, and Gather complete the real Qwen graph while device-side GQA/Reduce/Gather guards still latch errors before token consumption. After Chew caught stale fallback assertions, Pris updated the H200 smoke to require one capture, 62 replays, zero fallbacks, and no fallback reason. Landed as `dda3b25`, `13c094a`, and `42b71f7`.  
**Why:** Qwen2.5-0.5B int4 now captures end to end with token-exact graph ON/OFF parity and zero fallbacks: 70.33 versus 19.99 tok/s at 256 tokens (+251.8%), and 24.25 versus 11.73 tok/s at 1024 tokens (+106.7%). This validates the complete M4 capture-safety track on the real model.


## 2026-07-21 — Perf campaign reconciliation

### H200 native CUDA decode target and profiling baseline
**By:** Ana and Rachael  
**What:** Use ORT GenAI H200 Qwen2.5-0.5B int4 steady-state decode as the performance target: **657.34 tok/s** at 256 tokens (667.43 tok/s at 1024). Native progressed from about **73 → 145 → 192 → 201 tok/s**, but f32 Sq=1 GQA remained dominant: 70.5% of GPU time over 256-token decode and 82.7% over 16-token decode.  
**Why:** GEMV/argmax work is valuable but insufficient alone; the next high-leverage path is replacing serial f32 decode attention and then wiring/validating fp16 flash decode.

### Retile MatMulNBits decode GEMV and approve the result
**By:** Royb; reviewed by Wallace 🟢  
**What:** Retile the M=1 accuracy-level-4 symmetric block-32 CUDA MatMulNBits path, quantizing the f32 activation once with matching warp absmax/round/clamp/scale semantics. Wallace approved Roy's `5dbcbbb` retile.  
**Why:** This moved native decode from roughly 145 tok/s to about 192 tok/s while preserving numerics, but still leaves a large gap to Ana's 657 tok/s ORT target.

### Keep device-side greedy argmax after Batty's rebase repair
**By:** Mariette and Batty; reviewed by Joi 🟢  
**What:** Add allocation-free CUDA f32 greedy argmax with lowest-index tie behavior matching the host sampler. Joi rejected Mariette's rebased `c12e74f` because `DecodeCudaState::run_one_token` was called without the new `TraceContext`; Batty fixed the call and Joi approved `cdf62a0`.  
**Why:** The fixed path builds and measured about **200.97 tok/s**, removing the host argmax bottleneck without changing token selection.

### Land fp16 flash-decode as kernel-only first, then dormant dispatch wiring
**By:** Sebastian; reviewed by Bryant and Holden 🟢  
**What:** Add a capture-safe fp16 flash-decode GQA attention kernel as kernel-only commit `9c6f36b`, approved by Bryant. Wire it through a dormant fp16 dispatch branch at `521438e`, approved by Holden, gated by `q.dtype == Float16` and supported `(q_seq, dim)` while leaving the f32 path first and unchanged.  
**Why:** Split landing keeps the kernel independently reviewed and lets dispatch be enabled safely only for supported fp16 decode shapes.

### Direct fp16 activation × int4 GEMV remains a separate optimization track
**By:** Royb  
**What:** Prototype direct fp16-activation × int4 MatMulNBits GEMV on `wt-fp16-matmul` (`6a1daa2`) to avoid the int8 quantization pass.  
**Why:** This is distinct from fp16 flash attention and should be validated as a separate GEMV optimization before promotion.

### Sequence zero-copy design needs a second Deckard revision
**By:** Zhora and Deckard; reviewed by Luv 🔴  
**What:** Zhora's zero-copy Sequence tensors use shared allocation views with dtype/shape/layout/offset metadata. Luv rejected `ddae7d0`; Deckard closed the original public-output/runtime blockers with `SessionOutput::{Tensor, Sequence}` and related fixes, but Luv's re-review still rejected `cf8888b`.  
**Why:** The direction is acceptable, but remaining correctness/review blockers mean the Sequence zero-copy change is not approved yet.

### Runtime string tensors must use a dedicated host storage variant
**By:** Batty  
**What:** Represent runtime strings with `TensorStorage::{Raw, Strings(Vec<String>)}` or equivalent, expose safe `StringTensorView`/`StringTensorMut`, and never cast byte/device storage to `String`.  
**Why:** String tensors are host-owned structured values, not raw numeric buffers; exhaustive storage keeps executor behavior type-safe.

### PressureProtocol scaffold/fix path and current rejection state
**By:** Sapper, Roy, Deckard, and Pris; reviewed by Holden and Freysa 🔴/🟢 mixed  
**What:** Sapper/Roy added HostGovernor pressure envelopes and replay extension points; Holden rejected the first scaffold until actor ordering was scoped by `(HostId, ActorId)`, which Deckard fixed. Freysa rejected Sapper's HostGovernor revision, locking Sapper out and assigning the fix to Batty; Roy repaired release integrity by retaining authoritative allocations in `Claimed` and enforcing deterministic scheduling. Freysa's 2026-07-21 re-review still rejected `3207c25` because the branch/diff was not review-clean. Pris strengthened forged-release and cancellation synchronization regression tests.  
**Why:** Credit integrity and deterministic admission are the right design constraints, but the pressure implementation is not approved until reviewed from a clean branch with the fixed protocol evidence.

### Graph-capture transparency requires structured reasons across three axes
**By:** Coordinator and Gaff; reviewed by Chew  
**What:** All EPs must surface structured trace reasons for kernel non-selection and graph-capture non-capturability; transparency has three axes: op claim, kernel-variant selection, and capture support. Gaff added `CaptureSupport::{Supported, Unsupported { reason }}` and default compatibility adapters; Chew reviewed the structured reason-carrying design.  
**Why:** Silent bool declines make performance debugging impossible; traces must explain both variant choice and capture segmentation/fallback.

### Decouple CUDA EP claim from segmented graph capture
**By:** Coordinator and Tyrell  
**What:** CUDA EP should claim/run supported subgraphs even when only maximal segments are capturable, interleaving captured runs with eager CUDA runs for non-capturable nodes.  
**Why:** Capturability is an execution scheduling property, not an EP ownership property; partial segmented capture preserves CUDA placement without all-or-nothing fallback.

### Cross-platform support must include Windows ARM64
**By:** Coordinator; audit by Deckard  
**What:** Treat `aarch64-pc-windows-msvc` as a required target alongside Windows x64, macOS x86_64/arm64, and Linux x64. Deckard also flagged truthful CUDA selection, OS-aware library discovery, updated CUDA-12 CUDART candidates, pip/Conda NVIDIA discovery, and preventing Python from advertising CUDA while executing CPU.  
**Why:** Packaging and runtime probing must match the documented support matrix and actual execution provider behavior.

### Publishability of onnx-rs remains required
**By:** Leon  
**What:** Keep `onnx-rs` publishable to crates.io with package metadata and publish workflow coverage.  
**Why:** It is the ONNX standard-library crate for Rust in this workspace and must remain releasable.

### Capture-safe Sq=1 GQA decode kernel approved as prior f32 stepping stone
**By:** Sebastian; reviewed by Bryant 🟢  
**What:** Bryant approved `b6ada01`, a capture-safe warp-parallel Sq=1 GQA decode attention kernel for supported `head_dim <= 128` with zero CUDA-graph fallback.  
**Why:** This was a correct f32 decode-attention stepping stone before the later fp16 flash-decode path.

## 2026-07-21 — fp16 decode, transparent fallback, cross-platform loading, and trace cost

### Land coherent end-to-end fp16 native CUDA decode
**By:** Sebastian; component work by Mariette, Leon, and Roy; reviewed by Bryant, Wallace, and Holden 🟢  
**What:** Thread fp16 activations, KV, logits/argmax, normalization, RoPE, attention, and direct fp16×int4 MatMulNBits through native decode while retaining dtype-gated f32 paths. Leon fixed the rejected fp16 LayerNorm shared-memory reuse race before Bryant approved the normalization/RoPE path. Landed as `c8741ba`.  
**Why:** H200 Qwen2.5-0.5B int4 reached about **344 tok/s** with coherent tokens, CUDA graph capture, and zero fallbacks, up from the approximately **200 tok/s** f32 path; f32 remained unregressed near 200 tok/s.

### Make CUDA-to-CPU fallback observable and optionally strict
**By:** Deckard; reviewed by Batty 🟢  
**What:** Retain a structured `ExecutionProviderFallbackReport`, emit an initialization warning when CUDA declines force whole-session CPU execution, and make `ONNX_GENAI_REQUIRE_CUDA=1` reject that fallback. Landed as `3a8eebe`.  
**Why:** Device selection must not silently advertise CUDA while executing on CPU; callers now receive node/op/reason detail and can opt into strict CUDA-only behavior.

### Use OS-aware CUDA and CUPTI dynamic-library discovery
**By:** Leon and Roy; reviewed by Pris 🟢  
**What:** Select CUDA driver/runtime/library and CUPTI candidates by operating system, including Windows DLL names and pip/Conda layouts. Treat Windows ARM64 as gracefully unavailable before probing x64-only NVIDIA libraries. Landed as `2466016` and `8cd36c3`.  
**Why:** Cross-platform probing must fail normally rather than panic or attempt incompatible binaries. CUPTI discovery remains local to the tracer to avoid an inverted dependency on the CUDA EP.

### Emit per-op CPU bytes/FLOPs only for active trace spans
**By:** Rachael, Gaff, and Deckard; reviewed by Zhora 🟢  
**What:** Annotate major CPU kernel spans with logical tensor bytes and documented FLOP estimates, lazily computing metrics only when a span is active. Keep tracing optional and propagate the `tracing` feature through `bench-native` and `native-backend`. Landed as `61f4d2c`.  
**Why:** Profiles gain arithmetic-intensity and bandwidth inputs without imposing tensor scans, formula work, JSON allocation, or tracer dependencies on default non-tracing builds.



## 2026-07-21 — CI hardening and native CUDA decode wave 1–2

### Cover every offline crate and make warnings blocking on all portable targets
**By:** Batty and Gaff; Windows ARM64 revision by Deckard; reviewed by Hudson 🟢  
**What:** Classify all 38 workspace members by default normal+dev dependencies, explicitly test and cover all 27 pure-offline crates, and enforce blocking rustc and Clippy warnings (`RUSTFLAGS="-D warnings"` and `-- -D warnings`) rather than advisory lanes. The portable matrix retains Linux x64, Windows x64, and macOS ARM64 and adds native Windows ARM64 on `windows-11-arm`/`aarch64-pc-windows-msvc`, with the same 26-crate portable test set and an ARM64 Clippy gate; `mlas-sys` remains Linux-only, while native-ORT and CUDA crates stay outside offline execution. Formatting remains advisory pending the repository-wide sweep.  
**Why:** CI now covers the full offline workspace without triggering ORT downloads, and warnings fail builds across supported portable targets. The final 27-crate Linux lane passed 1,921 tests with 0 failures and 8 ignored; Hudson approved after Deckard closed the initially missing Windows ARM64 gate.

### Keep the measured wave-1 decode optimizations capture-safe
**By:** Leon, Tyrell, Deckard, Sebastian, and Roy  
**What:** Use persistent two-pass multi-block greedy argmax; segment CUDA graphs into maximal capturable runs around eager CUDA seams while retaining whole-subgraph EP ownership; abort/drain failed mid-segment capture before reset; use true multi-CTA split-K fp16 flash decode; and retain Roy's coalesced direct fp16×int4 GEMV retile. All paths preserve fixed device addresses, token semantics, and zero-fallback graph replay.  
**Why:** These changes removed launch/occupancy and GEMV bottlenecks without regressing correctness: argmax reached about 368 tok/s, split-K attention about 398 tok/s at 256 tokens (about 390 at 1024), and the GEMV retile about 423 tok/s. Segmented capture now recovers cleanly from invalidated streams instead of wedging later inference.

### Fuse the single-token GQA preparation chain
**By:** Rachael; reviewed by Holden 🟢  
**What:** For eligible `Sq=Sk=1` aliased fixed-capacity decode, fuse QKV split, query relayout, K/V append, and Q/K RoPE into one kernel and write attention output directly in BSH layout. Keep metadata preparation separate to preserve the capture poison/latch protocol; all other shapes retain the unfused path.  
**Why:** Prep launches fell 75% (192→48 per token), bit-exact fused/unfused and capture tests passed, and H200 throughput rose from about 557 to 615 tok/s with zero fallbacks.

### Use warp-shuffle fp16 skip-RMSNorm
**By:** Sapper; reviewed by Wallace 🟢  
**What:** Replace the fp16 shared-memory reduction tree with a single-warp packed-half2/half4 shuffle reduction, specializing hidden size 896 while retaining a tail-safe generic fp16 path; f32 kernels remain unchanged.  
**Why:** The hot kernel fell from about 6.20 to 5.07 µs/call and stacked decode reached about 579–583 tok/s with identical tokens, full CUDA tests passing, and zero graph fallbacks.

### Specialize the fp16 down-projection GEMV and accept the stacked ORT win
**By:** Luv; reviewed by Pris 🟢  
**What:** Route only `K=4864, N=896, block_size=32` with fp16 scales to a 256-thread, eight-column K-parallel GEMV that stages the activation in permuted half2 shared memory; all other shapes retain the general kernel.  
**Why:** The down-projection kernel fell from about 10.24 to 7.28 µs/call with parity within fp16 tolerance and identical greedy tokens. Stacked with GQA fusion and RMSNorm, native H200 decode reached **663–672 tok/s**, beating the **657 tok/s ORT GenAI** reference with zero fallbacks.

### Require SM-portable correctness and performance for every CUDA EP kernel
**By:** Coordinator directive; validated in wave-2 reviews by Holden, Wallace, and Pris  
**What:** Every `onnx-runtime-ep-cuda` kernel must remain correct and performant across supported NVIDIA SM architectures, not merely `sm_90`. Dispatch must derive the live architecture dynamically, avoid unguarded SM90-only features, keep resource use within portable limits, and preserve capable fallbacks or variants where architecture-specific tuning is necessary.  
**Why:** H200 wins are not acceptable if they break or materially strand devices such as RTX 4060 (`sm_89`). Wave-2 kernels use broadly available primitives and do not raise the minimum architecture.
