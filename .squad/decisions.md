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
