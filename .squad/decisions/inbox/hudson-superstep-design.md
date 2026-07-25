# Hudson — Whole-step SPMD closure: design + gated-spike prototype (barrier cost measured)

**By:** Hudson (executor dispatch/fusion) · **Date:** 2026-07-25
**Branch:** `perf/decode-superstep` (worktree `onnx-genai-cpu-superstep`), based on main `cedf5d0`
**Host:** Xeon 8480C, 96 logical CPU / 2 NUMA. Shared/contended (loadavg 4–11 during the window); every A/B generation load-gated <11 at `/proc/loadavg` 1-min, interleaved round-robin, median-of-5 (Bishop's method).
**Model:** qwen3-0.6b-generic-cpu-4/v4 (int4 MatMulNBits, acc_level=4), greedy, prompt "The capital of France is".
**Parity bar:** f64-reference / token-exactness (2026-07-25 coordinator decision). This change touches only SCHEDULING, not math, so it must be **bit-identical**.

---

## TL;DR — the hypothesis is WRONG; do NOT build the whole-step rewrite yet

The per-op fork/join barrier is **~2 µs**, so the ~196 barriers in a 0.6B decode step cost **≈ 0.4 ms ≈ 3–4% of an 11.6 ms step — not ~40%.** Collapsing the step to one barrier can recover **at most ~3–4%**, nowhere near the ~1.7× the pivot hoped for. Measured three independent ways (isolated microbench, parked-worst-case microbench, and an in-situ token-exact A/B that *adds* barriers to a live decode step) — all agree. The 0.59× ORT gap (42 vs 71 GB/s effective) lives somewhere **other than fork/join barrier count**. This is the valid negative result the spike was gated to find; it saves a large, high-risk executor rewrite.

---

## 1. Current per-op SPMD dispatch — how many barriers, and what each costs

**Structure** (`decode_spmd.rs`, `kernels/matmul_nbits.rs`, `onnx-runtime-session/src/executor.rs`):
- `with_decode_pool_scope` wraps the **whole** single-token forward once per step. Under the persistent SPMD path the forward runs **inline on the engine/dispatcher thread**, and the executor walks each ONNX node one at a time.
- Each M=1 `MatMulNBits` projection calls `parallel_output_rows` → `SpmdDecodePools::dispatch_output_rows` → `dispatch()` = **one `publish` + `wait` barrier**: bump `sequence`, workers observe it (spinning-hot, or `unpark` if parked), each runs its pre-assigned output-row shard, decrements its node's completion counter; the dispatcher spins on those counters. GQA adds a `dispatch_output_row_blocks` barrier per layer. Small elementwise/norm/RoPE/reshape run **serially on the dispatcher** between barriers.
- **Barrier count / 0.6B step:** 28 layers × 7 MatMulNBits (q,k,v,o,gate,up,down) + 1 lm_head = **197**, plus ~28 GQA row-block dispatches ≈ **~196–225 fork/join barriers per decode step.** Matches Bishop's ORT per-op node count (197 MatMulNBits) and the pivot's "~196".

**Per-barrier cost — isolated microbench** (`decode_spmd::tests::bench_empty_barrier_cost`, empty publish/wait × 200k on the hot worker set):

| workers | per-barrier | ×196 = barrier tax/step |
|--------:|------------:|------------------------:|
| 8  | 1134 ns | 222 µs |
| 16 | 2180 ns | 427 µs |
| **32 (0.6B config)** | **2502 ns** | **490 µs** |
| 48 | 4737 ns | 928 µs |

**Parked worst-case** (`bench_parked_barrier_cost`, a gap forces spin→yield→park so each publish issues a real futex `unpark`): 32 workers = **3.8 µs** (200 µs gap) … **7.4 µs** (2 ms gap). So even if workers park between *every* op the tax is ≤ ~1.46 ms/step. Real decode ops are near-back-to-back (Bishop: executor glue negligible, step is 99% MatMulNBits), so workers stay spinning and the hot ~2.5 µs figure applies.

**In-situ A/B — the decisive test** (`.squad/hudson_superstep_probe.sh`, flag `ONNX_GENAI_DECODE_SUPERSTEP=N` injects N extra *empty* barriers per real dispatch, in the exact position real ones occur; persistent pool forced, 32 workers, median-of-5 interleaved, load-gated):

| N | total barriers/step | tok/s (median) | ms/step |
|--:|--------------------:|---------------:|--------:|
| 0 | ~196  | 86.11 | 11.61 |
| 1 | ~392  | 83.99 | 11.91 |
| 4 | ~980  | 72.77 | 13.74 |

- N0→N1: **+0.30 ms for +196 barriers = 1.5 µs/barrier**
- N0→N4: **+2.13 ms for +784 barriers = 2.7 µs/barrier**
- **Token-exact: N0 == N1 == N4 (generated_token_ids byte-identical). ✅**

The in-situ marginal cost (1.5–2.7 µs) brackets the isolated microbench (2.5 µs). **Sanity check on the hypothesis:** if barriers were 40% of the step, doubling them (N=1) would cut throughput ~30%; observed drop is **2.5%**. Barriers are ~2.5–3.4% of the step.

---

## 2. The staged whole-step-closure design (documented; see §4 for the verdict)

Kept for completeness — this is the design the pivot asked for. **Given §1 it is not worth building for the claimed payoff.**

**Idea:** keep the persistent workers hot across the *whole* step (one dispatch, one join). Instead of the dispatcher publishing 196 ops, publish a single "step program": each worker marches the op chain, synchronizing point-to-point only at a true data dependency (a producer→consumer edge) via lightweight per-stage sequence points rather than a full pool barrier after every op.

**Token-exactness:** preserve each op's reduction/accumulation order and row-sharding exactly (the current row-shard is already bit-identical — no cross-row reduction). Only scheduling changes, so bit-identical is achievable and is the bar.

**Good-citizen budget:** unchanged — same worker count (#160 `--cpu-cores`), same reserved dispatcher CPU (#154), same NUMA layout. The closure changes *when* barriers fire, not how many cores spin.

**Staging (each independently correctness-gated):**
- **Stage 1** — coalesce the independent projections that share an input into one barrier: `q/k/v` (all read post-norm hidden) → 1 dispatch instead of 3; `gate/up` → 1 instead of 2. Removes ~3 barriers/layer ≈ 84/step.
- **Stage 2** — span the full layer stack: workers walk QKV→attn→O→gate/up→down with sequence-point sync at the attention and residual boundaries.
- **Stage 3** — whole step incl. sampling under one resident region.

**Cheaper alt if any barrier win is ever wanted:** raise the spin budget so workers never park during a step (bounds the tax to the hot ~2.5 µs), and Stage-1 coalescing — both far lower risk than a worker-walks-the-graph rewrite. But per §1 the ceiling is ~3–4%.

---

## 3. What was prototyped (Stage-1 increment) + results

Rather than build the full closure to test a hypothesis the microbench already contradicted, I prototyped the **decisive measuring instrument** — the in-situ barrier probe — behind an opt-in, reversible flag (`ONNX_GENAI_DECODE_SUPERSTEP`, default off, per-op path unchanged). It is the smallest change that answers the gating question *end-to-end in the real decode loop, token-exact*, and it does so more conclusively than a partial rewrite (removing barriers is symmetric to adding them: ~2 µs each either way).

- **Code:** `decode_spmd.rs::superstep_probe_barriers()` + `SpmdDecodePools::probe_empty_barriers()`; wired into both int4 SPMD dispatch routes in `matmul_nbits.rs` (hand `parallel_output_rows` and MLAS `run_mlas_shards`). Two ignored microbenches (`bench_empty_barrier_cost`, `bench_parked_barrier_cost`). Driver `.squad/hudson_superstep_probe.sh`.
- **Token-exactness:** N=0/1/4 produce byte-identical `generated_token_ids`. ✅
- **Perf signal:** adding ~196 barriers costs +0.30 ms (2.5%); ~784 costs +2.13 ms. → real per-op barrier ≈ 2 µs, ~196/step ≈ 0.4 ms ≈ 3.4% of the step.
- **Gates:** `cargo test -p onnx-runtime-ep-cpu --features mlas` = 925+10 pass, 0 fail. Clippy x86_64 `--features mlas -D warnings` ✅ and aarch64 no-mlas `-D warnings` ✅.

---

## 4. Honest read — will whole-step closure recover the ~1.7×?

**No.** Three independent measurements agree the per-op fork/join barrier is ~2 µs, so the ~196 barriers cost **~0.4 ms ≈ 3–4% of the 0.6B step.** A whole-step closure (one barrier/step) is bounded above by that — **~3–4% throughput, not 1.7×.** Stage 1 alone (coalescing ~84 barriers) would capture ~1%.

**Why the pivot's inference over-reached:** it read the 42-vs-71 GB/s effective-bandwidth gap as "= non-streaming stall time = per-op barrier overhead." The barrier *latency* is measurably ≤3.4% of the step, so that identity does not hold. Whatever causes the 0.59× gap, it is **not** fork/join barrier count.

**Where the gap more likely lives (for the next spike):**
1. **Thread-scaling efficiency of the sharded int4 GEMV vs ORT's full-width MLAS.** Bishop's "hand beats MLAS SQNBit by 44%" is a *per-op M=1* result; it does not mean our 32-way row-sharded decode reaches the same effective bandwidth ORT's kernel does across the whole step. Small projections (e.g. k/v: 1024 rows / 32 workers = 32 rows/worker) may be too fine-grained to hit steady-state bandwidth per worker — a *work-granularity/imbalance* problem, not a barrier-count problem. Worth profiling per-worker GEMV efficiency and trying a coarser shard / fewer-but-fuller workers.
2. **Dispatcher-serial non-GEMM work** running alone between barriers while 32 workers idle-spin (norm/RoPE/reshape/cast/GQA-setup). Eager profiling shows it small, but it is serialized and could matter more on a clean host.
3. **Memory layout / prefetch / page placement** across the per-step weight stream.

**Recommendation:** close this spike as a **negative result** — do not build the whole-step SPMD closure for the claimed 1.7×. Redirect to (1): measure per-worker int4 GEMV efficiency and shard granularity vs ORT full-width MLAS on a clean host. Keep `ONNX_GENAI_DECODE_SUPERSTEP` in-tree as a permanent barrier-cost probe. The prototype is saved on `perf/decode-superstep`; **no PR opened** (gated spike).
