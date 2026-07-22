# Decision: persistent SPMD decode pool (per-op barrier fusion) for native CPU int4 M=1

**Author:** Pris (senior performance engineer)
**Branch:** `perf/decode-barrier` (based on `perf/cpu-ep-mlas` @ `160f4fc`)
**Date:** 2026-07-22
**Status:** Positive result — opt-in, default OFF, no default-path change.
**Reviewer:** pending (rule 9 — non-author review required before merge)
**Flag:** `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1` (default OFF)

---

## TL;DR

`ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1` with `ONNX_GENAI_CPU_DECODE_THREADS=32`
replaces the ~141 per-token rayon fork-join regions with **one persistent SPMD
worker set** driven by a lightweight reusable barrier. It raises steady M=1 int4
decode from the prior best `numa-split` **16.42 tok/s median** to **17.71 tok/s
median (best 18.37)** — **+7.9% median / +8.3% best** — winning **every** one of
4 interleaved rounds, with **exact greedy bit-parity** (byte-identical token ids,
flag ON vs OFF, 64 tokens). It does not reach ORT (26.9) / onnxruntime-genai
(20.8) — the residual gap is memory-latency bound — but it is the best native
result, beats the prior best path, and is shipped OFF by default.

---

## 1. Lever chosen (from `pris-decode-profile.md`)

Profile concluded decode is per-op **fork-join barrier** + memory-latency bound,
not kernel-compute bound (~141 `MatMulNBits`/token, each a separate parallel
region with a join barrier; cross-socket barriers are toxic; the naive dual-node
pool regressed to ~11 tok/s for exactly this reason). Chosen lever = candidate
(a): a **persistent SPMD decode pool** that keeps workers hot for the whole
decode loop and signals per-op completion with **per-node** counters, so the
barrier stays node-local and the repeated fork/dispatch cost of re-creating 141
regions per token disappears. The actual GEMV still runs the existing packed
int4 / MLAS SQNBit kernels (Rule 4 — orchestration changed, not the math).

## 2. Implementation

New file `crates/onnx-runtime-ep-cpu/src/decode_spmd.rs` (persistent pool +
barrier primitive + unit tests), wired into
`crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs` behind the flag,
mirroring the existing `decode_numa.rs` two-level structure so it inherits its
node-local placement and its exactly-associative reduction order.

Barrier / dispatch protocol (portable `std` atomics + `thread::park`):

- Spawn `ONNX_GENAI_CPU_DECODE_THREADS` workers once, pinned one-per-CPU across
  the covering NUMA node(s) using the same runtime `/sys` topology probe as
  `ONNX_GENAI_CPU_DECODE_AFFINITY` (no hardcoded socket/core counts).
- Each op: publish a type-erased job pointer → store per-node completion counts
  → bump a `sequence` counter (SeqCst) the spinning workers watch → unpark only
  workers that actually parked → spin-wait until every node's `pending` hits 0.
- Workers: spin on `sequence` change → run their row-shard → decrement their
  node's `pending`. Per-node counters keep the hot-path cache lines node-local
  (no cross-socket coherency round trip — the thing that sank the naive pool).
- Spin-then-park (spin 4096, yield 64, then `park_timeout(1ms)`), so during
  active decode (ops microseconds apart) workers stay spinning and issue **zero
  syscalls**; they only park on longer idle gaps. Park guard uses SeqCst to
  avoid a lost wakeup; the 1 ms timeout is a backstop.
- `Padded<T>` (`repr(align(128))`) on all shared atomics to avoid false sharing.

### Startup-readiness barrier (a real deadlock I hit and fixed)

A worker that entered its wait loop **after** the first dispatch initialized its
`local_seq` to the already-bumped sequence, missed that op, and its node's
`pending` never reached 0 → hang. Fix: each worker increments a `ready` counter
and starts `local_seq = 0`; `build()` blocks until `ready == total_workers`
before returning, so no op can be published before every worker is waiting for
it. Regression-guarded by a unit test that builds a fresh pool and dispatches
immediately, 40×, plus a 200-dispatch reuse test — both previously hung, now
pass in 0.03 s.

## 3. Bit-parity proof (Rule 8)

Row-sharding a GEMV is exactly associative — each output row is an independent
full-K dot product, no cross-row reduction — so any row partition yields
byte-identical f32. Verified end-to-end:

```
# prompt "Write a Rust function that reverses a linked list", 64 tokens
OFF numa-split : generated_token_ids: [13, 576, 729, 1265, 1896, 264, 25804, ...]
ON  persistent : generated_token_ids: [13, 576, 729, 1265, 1896, 264, 25804, ...]
diff off_ids.txt on_ids.txt  ->  identical  (PARITY OK)
```

Unit test `dispatch_preserves_per_row_reduction_bit_for_bit` asserts the sharded
result equals the flat single-thread computation bit-for-bit.

## 4. Measurement (honest, interleaved, noisy shared host)

Sapphire Rapids Xeon 8480C, 2×48c, 2 NUMA nodes, Qwen2.5-Coder-7B int4, 32
decode threads, `--steady --decode-skip 8 --tokens 96 --warmups 1`, A/B
interleaved per round (load avg ~12→37 over the run — later rounds noisier, SPMD
still won):

| Round | `numa-split` (OFF) | `PERSISTENT_POOL=1` (ON) |
| --- | --- | --- |
| 1 | 16.96 | **18.37** |
| 2 | 16.49 | **17.52** |
| 3 | 16.35 | **17.15** |
| 4 | 16.28 | **17.90** |
| **median** | **16.42** | **17.71** |
| best | 16.96 | 18.37 |

**+7.9% median, +8.3% best over the prior best `numa-split`; wins 4/4 rounds.**
Consistent with an earlier 96-tok interleaved A/B this session (17.95 vs 16.20,
+10.8%). Short-run (≤24 tok) numbers are inflated by cache residency and are not
used. Still below ORT 26.9 / genai 20.8 — residual gap is memory latency, not
addressed by this lever.

## 5. Generality & rules compliance

- **Rule 1 (profile first / good errors):** profiled before writing (see
  `pris-decode-profile.md`); pinning failures report a clear fallback message,
  not a panic.
- **Rule 2 (EP/topology-agnostic):** default OFF; single-node host, non-Linux,
  or cgroup-refused pinning all degrade to the existing bounded-pool behavior.
  Barrier primitive is portable `std` atomics + `thread::park`; only optional CPU
  pinning is Linux-specific (best-effort no-op elsewhere). No x86-only or
  2-node-only assumptions — worker count / node split derive from runtime
  topology and `ONNX_GENAI_CPU_DECODE_THREADS`.
- **Rule 4 (reuse MLAS):** the GEMV still runs the existing packed int4 / MLAS
  SQNBit kernels; only the barrier/orchestration changed.
- **Rule 5 (explicit, inspectable, documented flag):** `ONNX_GENAI_CPU_DECODE_
  PERSISTENT_POOL`, default OFF, documented in `docs/numa-decode-plan.md`.
- **Rule 8 (tests track behavior):** bit-parity unit test + startup-race /
  reuse regression tests + full `onnx-runtime-ep-cpu` suite (692 pass) + clippy
  `-D warnings` clean.
- **Rule 9 (non-author review):** pending — did NOT push; coordinator merges
  after review.

## 6. Follow-ups (not done here)

The remaining gap to ORT is memory-latency bound. Next candidate levers (out of
scope for this change): (b) barrier fusion of consecutive independent M=1
projections (gate+up, Q/K/V) under a *single* dispatch to further cut the ~141
regions; and weight-stream prefetch / packing to hide DRAM latency. Both should
be built on this persistent pool rather than the per-op rayon path.
