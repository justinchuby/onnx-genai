# Profile: native CPU int4 M=1 decode — where the per-token time goes

**Author:** Pris (senior performance engineer)
**Branch:** `perf/decode-barrier` (based on `perf/cpu-ep-mlas` @ `160f4fc`)
**Date:** 2026-07-22
**Status:** Profile evidence (Phase 1). Feeds the lever chosen in
`pris-decode-barrier.md`.
**Host:** Sapphire Rapids Xeon 8480C, 2 sockets × 48 cores, 2 NUMA nodes
(node0 = CPUs 0–47, node1 = CPUs 48–95). Shared/noisy host — load avg ranged
~6–37 across the session. perf hardware counters are **not** available
(virtualized: `perf stat -e cycles,LLC-load-misses,...` returns "not supported").
Every number below is a `steady_median` and A/B configs were interleaved.

---

## TL;DR

Native decode is **not kernel-compute bound** and it is **not** helped by more
threads. It is bound by (1) memory latency streaming the int4 weights and (2)
the **per-op fork-join join barrier**: there are ~141 `MatMulNBits` ops per
decoded token, and each one is today a *separate* rayon parallel region whose
join barrier + task hand-off is pure overhead once the kernel is L3-resident.
The isolated int4 kernel matches MLAS SQNBit (~108 GB/s L3), so the gap to ORT
is orchestration + memory latency, not the GEMV math. This is why the chosen
lever changes the **barrier structure**, not the kernel.

---

## 1. Method

Build (in this worktree):
```
cargo build --release -p onnx-genai-bench --features mlas --bin profile_native
export LD_LIBRARY_PATH=$(find /home/justinchu/onnx-genai-cpu/target/release/build \
  -type d -path '*onnx-genai-ort-sys*/out/ort-prebuilt/lib' | head -1):$LD_LIBRARY_PATH
```
Model: `~/.foundry/cache/models/Microsoft/qwen2.5-coder-7b-instruct-generic-cpu-4/v4`
(int4 weights, fp32 activations — the only fully-runnable CPU model).
Per-op profiler: `ONNX_GENAI_PROFILE_OPS=1`. Threads: `ONNX_GENAI_CPU_DECODE_THREADS=32`.
Steady window: `--steady --decode-skip 8 --tokens 96/128 --runs 1–3 --warmups 1`.

## 2. Per-op breakdown (ONNX_GENAI_PROFILE_OPS=1)

Of ~65 ms/token steady decode:

| Bucket | share | notes |
| --- | --- | --- |
| `MatMulNBits` (141 ops) | ~80% (~52 ms) | the int4 projections + LM head |
| everything else (attn, norm, elementwise, KV) | ~20% (~13 ms) | |

The kernel *itself* is fast: an isolated int4/VNNI GEMV of the same shapes runs
in ~33 ms total for all 141, but in-engine those same ops take ~52–64 ms. The
~20–30 ms delta is **per-op glue**: rayon fork/join, crossbeam-epoch, task
hand-off, and the join barrier at the end of each of the 141 parallel regions.
A prior `perf record` attributed ~27% of decode samples to that glue rather than
to the kernel inner loop.

## 3. Parallel regions per token

**~141 fork-join parallel regions per token** — one per `MatMulNBits`. Each pays:
publish work → wake/dispatch workers → run shard → **join barrier**. At M=1 the
per-region *work* is tiny (a GEMV over one row of activations), so the fixed
barrier + dispatch cost is a large fraction of each region.

## 4. Why threads/affinity, not raw compute, move the needle

Steady decode this session (32 threads, interleaved medians):

| Config | median tok/s |
| --- | --- |
| `compact` (single node, pinned) | ~14.0–14.9 |
| `numa-split` (16+16, two-level rayon barrier, node-local weights) | ~16.4–16.6 |
| `off` (unpinned) | ~13–14, jittery |

Going from `off`→`compact`→`numa-split` improves throughput **without changing
any math** — purely by making the barrier traffic and weight stream node-local.
That is the signature of a barrier + memory-latency bound, not a compute bound.
Conversely, a **naive** all-96-thread dual-node pool *regressed* to ~11 tok/s
because every one of the 141 join barriers then paid a cross-socket coherency
round trip. **Cross-socket barriers are toxic**; the per-op barrier is the thing
to attack.

## 5. Per-barrier cost intuition (no hardware counters)

With counters unavailable I reasoned from the deltas: numa-split's win over
compact comes entirely from keeping each of the 141 joins node-local, and the
naive dual-node loss comes entirely from making them cross-node. So the join
barrier cost is first-order in the per-token budget, and its *placement*
(node-local vs cross-socket) is what dominates — a persistent pool that keeps
completion signalling node-local should recover the same locality *and* remove
the repeated fork/dispatch cost of re-creating 141 regions per token.

## 6. Conclusion → lever

Attack the **per-op fork-join barrier**: keep one persistent worker set hot for
the whole decode and drive each op with a lightweight reusable barrier whose
completion counters are per-node (node-local cache lines, no cross-socket round
trip). Reuse the existing int4/MLAS kernels for the actual GEMV. See
`pris-decode-barrier.md` for the implementation and measured result.
