# Native CPU decode performance investigation

**Date:** 2026-07-20
**Host:** Intel Xeon Platinum 8480C (Sapphire Rapids), 96 physical hardware
threads (2 × 48 cores; no SMT)
**Model:** `/home/justinchu/qwen2.5-0.5b-int4-onnx`, Qwen2, 24 layers,
hidden size 896, 14 Q heads / 2 KV heads, head size 64, vocab 151,936.

## Baseline and method

The release `profile_native` benchmark used CPU EP, the prompt `The capital of
France is`, 128 generated tokens, two warmups, and three measured runs:

| Configuration | Throughput | Time / generated token |
|---|---:|---:|
| Current default (no decode-thread setting) | **18.75 tok/s** | **53.340 ms** |
| Explicit 4-thread decode Rayon pool | 22.17 tok/s | 45.107 ms |
| Explicit 8-thread decode Rayon pool | **22.38 tok/s** | **44.679 ms** |
| Explicit 16-thread decode Rayon pool | 20.03 tok/s | 49.916 ms |

The `ONNX_GENAI_CPU_DECODE_THREADS` sweep (64-token runs) corroborated the
shape: 1/2/4/8/16/32/48/96 threads measured 64.108/50.818/46.685/46.906/
55.024/63.853/67.434/64.808 ms per token.  Thus a 4--8 thread pool is already
an immediately measurable **19%** improvement over the 53.340-ms baseline;
the 96-thread setting is worse than serially sensible sizing.

`perf stat -d` could not collect hardware events in this environment (the PMU
topdown event is unsupported and cycles/instructions are unavailable), so this
report uses the runtime's per-op timer plus `/proc`/`ps -L` observations.

## Core utilization

This is not a 96-core decode.  During a 512-token decode sampled every 0.5 s
with `ps -L`, the default process had 97 threads but accumulated only
approximately **270--281% CPU** (about 2.7 cores).  With
`ONNX_GENAI_CPU_DECODE_THREADS=4`, it had 101 OS threads (the global Rayon
threads still exist plus the private pool), but only approximately
**175--192% CPU** (about 1.8--1.9 cores).  The many threads are mostly parked;
decode consists of many short serially ordered operator launches and small
parallel regions, not a sustained 96-way computation.

`DECODE_POOL` is a process-global, lazily-created optional private Rayon pool
(`crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs:24-26,
628-659`).  It is disabled by default, so the GEMV's `par_chunks_mut` uses the
global Rayon pool.  The M=1 kernel fans each projection into chunks
(`:724-750`); chunk sizing depends on `rayon::current_num_threads`
(`:1086-1118`).  This explains both the parked global workers and the
excessive synchronization cost for the default many-thread path.

## Measured operator attribution

With `ONNX_GENAI_PROFILE_OPS=1`, 32 forward calls (one warmup generation plus
one measured generation, 16 tokens each) attributed executor node time as:

| Operator | Executor node time | Share |
|---|---:|---:|
| `MatMulNBits` | 3961.408 ms | **83.2%** |
| `GroupQueryAttention` | 700.374 ms | **14.7%** |
| `Silu` | 41.665 ms | 0.9% |
| `SkipSimplifiedLayerNormalization` | 29.793 ms | 0.6% |
| `Add` and all remaining nodes | 26.9 ms | 0.6% |

This timing is intrusive and includes prompt/prefill calls, so use the shares,
not its absolute wall time, for ranking.  It confirms that the gap is inside
the native executor's M=1 kernel/attention path, not tokenizer or sampling.
There are 121 int4 `MatMulNBits` nodes per forward
(`docs/BENCH_MLAS_INT4_E2E.md:38-43`).

## Ranked bottlenecks and optimization directions

1. **M=1 MatMulNBits scheduling / thread-pool policy — 83% of executor
   node time; immediate 19% end-to-end win.**
   The hand int4 implementation is already the correct compute backend:
   weights are prepacked once in `PackedInt4Weight` and reused
   (`matmul_nbits.rs:227-247`), and the MLAS M=1 comparison was already
   recovered to the hand route.  The remaining defect is scheduling its 121
   tiny projections through the global 96-worker pool.  Empirically, 4--8
   private decode workers improve 18.75 to 22.38 tok/s, while 32--96 workers
   regress badly.  **Next slice:** make an auto-selected 4--8 thread
   decode pool the default for M=1 (with a documented override), and benchmark
   a small architecture/NUMA sweep.  Expected win: **~19--25%** at this model
   and host (about 22--23 tok/s).  This is the clear first change because it
   is already proven, contained, and does not alter numerical computation.

2. **GroupQueryAttention copies/rebuilds all historical KV every token — 15%
   at short contexts, increasing linearly with context.**
   The model uses GQA and the CPU kernel first copies each past K/V into owned
   `Bhsd` buffers (`group_query_attention.rs:157-170, 397-402`), then allocates
   fresh `present_k` and `present_v` sized for the complete context and copies
   past KV plus the one new token (`:527-557`).  It also allocates a `scores`
   vector per query head (`:575`).  At an average 128-token context, the two
   present KV copies alone are about 3 MiB/token across 24 layers
   (2 tensors × 2 KV heads × 64 × 128 f32 × 24), before the input-cache copies.
   **Direction:** a preallocated/static KV cache with append-in-place views,
   and attention that reads the cache directly; reuse score scratch.  Expected
   near-term win: **10--20% at 128 tokens**, with a larger gain at long
   contexts.  This is the principal follow-up after the scheduling fix.

3. **Per-MatMul temporary output allocation/zeroing — included in the 83%
   MatMulNBits bucket; likely low-single-digit standalone cost.**
   Every `MatMulNBits::execute` creates `vec![0.0; m * n]`
   (`matmul_nbits.rs:199-202`) despite the executor already owning a correctly
   sized output buffer.  The kernel subsequently copies it again with
   `write_dense_f32` (`:339`).  At M=1 this happens 121 times per token.
   This has not been isolated with an allocator profiler, so it must not be
   claimed as the primary ceiling; it is a likely **3--8%** cleanup within the
   matrix bucket.  **Direction:** write directly to a contiguous output view,
   or pool/reuse scratch while retaining non-contiguous correctness handling.

4. **Native host decode binding/KV ownership churn — likely low-to-mid
   single-digit at this sequence length, but coupled to #2.**
   Every step allocates input IDs, a growing attention-mask `Vec`, position
   IDs, an owned binding vector, and a new `HashMap` for all present tensors
   (`native_decode.rs:553-580, 588-620`).  This is not the dominant measured
   node time, but it prevents a zero-allocation decode loop and reinforces the
   full-KV-copy design.  **Direction:** persistent scalar/mask buffers and a
   fixed indexed KV container.  Expected standalone win: **<5%**; bundle it
   with the static-KV work.

5. **Sampling, tokenizer, graph dispatch, and simple elementwise operations —
   residual, not the ceiling.**
   The op profile leaves only 2.1% outside MatMulNBits and attention; simple
   non-linear/normalization nodes total 1.5%.  The decode loop does clone
   prompt/generated vectors while constructing `ProcessorContext`
   (`decode_loop.rs:191-217, 273-280`) and detokenizes each token
   (`:266-273`), but these cannot explain a 5× ORT gap.  Defer this work until
   the two large buckets are addressed.

## Resolution

M=1 decode now defaults to a private 8-worker Rayon pool, capped by
`available_parallelism`. `ONNX_GENAI_CPU_DECODE_THREADS` still accepts a
positive explicit worker count; setting it to `0` is the escape hatch that
restores the old global-pool behavior. The same 128-token, two-warmup,
three-run benchmark measured:

| Configuration | Throughput | Time / generated token |
|---|---:|---:|
| Default (bounded pool, env unset) | **23.46 tok/s** | **42.624 ms** |
| `ONNX_GENAI_CPU_DECODE_THREADS=0` (global pool) | 18.60 tok/s | 53.763 ms |
| `ONNX_GENAI_CPU_DECODE_THREADS=8` | 22.59 tok/s | 44.259 ms |

The new default improved measured throughput by **26.1%** over the explicit
global-pool opt-out. All configurations generated identical token IDs and
began with `Paris. It is the largest city in the country...`.

## Conclusion

The immediate ceiling is **bad M=1 scheduling, not selection of hand versus
MLAS int4 math**. The bounded 8-worker default removes that scheduling defect
and delivers the measured improvement above. The next, larger architectural
slice should eliminate full-KV cache materialization in `GroupQueryAttention`.
Even both changes will not alone establish ORT parity:
the remaining 83% matrix bucket still needs kernel-level work, but they remove
the known scheduling and copy overhead before that work is evaluated.
