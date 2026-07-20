# Persistent decode-pool residency

Status: **partial** (mechanism landed, numerics bit-identical; measured
end-to-end gain below the 20% graduation threshold — see
[Benchmark](#benchmark)).

## Problem

Native CPU decode of `qwen2.5-0.5b-int4` runs at ~23.5 tok/s (~42 ms/token).
The int4 GEMV kernel arithmetic is already good: explicit AVX-VNNI
(`_mm256_dpbusd_epi32`), SIMD nibble unpack, fused per-block scale, weights
prepacked once. **In isolation the kernel sustains 58–83 GB/s on 8 threads**
(`docs/BENCH_MLAS_INT4_E2E.md:124-135`), because the benchmark keeps every
iteration inside a single pool installation.

**End-to-end it collapses to ~9 GB/s.** Root cause: each token runs **121
`MatMulNBits` projections**, and each projection independently called
`with_decode_pool` → `DECODE_POOL.install(...)`. From the external decode
thread, every `install` is a full external-thread-to-pool crossing: task
publication + worker wakeup + join. That is **121 sequential pool
entry/launch/join cycles per token** — latency/scheduling fragmentation, not
intrinsic serial arithmetic and not scalar unpack.

Thread sweep on the same tokens (before this change):

| workers | tok/s | scaling |
|--------:|------:|--------:|
| 1       | 14.71 | 1.00×   |
| 8       | 22.54 | 1.53×   |
| 16      | 20.28 | regress |

8 workers give only 1.53× over 1 (should be near-linear), and 16 regress —
signatures of per-op barrier/wakeup latency dominating, not compute.

## Design: whole-forward residency (candidate b)

Run the **entire M=1 CPU decode forward pass inside one
`DECODE_POOL.install(...)` scope**, so the 121 inner `with_decode_pool` calls
execute **inline** (no re-install) while already resident on a decode-pool
worker. Workers stay warm and steal-ready across all 121 projections.

### Mechanism (in `onnx-runtime-ep-cpu`, where `DECODE_POOL` lives)

`crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs`:

* **`IN_DECODE_POOL: Cell<bool>`** — a thread-local residency flag.
* **`DecodeResidencyGuard`** — RAII guard. `enter()` saves the previous flag
  value and sets it `true`; `Drop` restores the previous value. Restoring the
  *previous* value (not hard-clearing to `false`) keeps nested scopes correct,
  and running in `Drop` guarantees the flag is cleared even if the forward
  **panics** (unwinding runs the guard).
* **`with_decode_pool_scope<R: Send>(f)`** — public scope entry:
  * `Ok(Some(pool))` → `pool.install(move || { let _g = enter(); f() })`. The
    guard is entered **inside** the installed closure, i.e. **on the worker
    thread that actually runs `f`**, not on the caller. This is essential:
    rayon runs the installed closure on a pool worker, and the sequential
    executor then runs all 121 ops on that same worker — so the inner
    `with_decode_pool` calls observe the flag and go inline.
  * `Ok(None)` (opt-out, `ONNX_GENAI_CPU_DECODE_THREADS=0`) → run `f()` inline
    with the flag left `false`; inner `with_decode_pool` keeps its existing
    global-pool behaviour.
  * `Err(_)` (pool build failed) → run `f()` inline, flag `false`; the inner
    `with_decode_pool` calls surface the same error and the forward fails
    identically to the un-scoped path.
* **`with_decode_pool` (per-op helper)** — now checks the flag first: if
  resident, run `operation()` **inline** (no second `install`); otherwise behave
  exactly as before (`install` when a bounded pool exists, else global pool,
  else error). This makes every existing caller correct whether or not it is
  wrapped by the outer scope — fully backward compatible.

`with_decode_pool_scope` is re-exported from the crate root
(`crates/onnx-runtime-ep-cpu/src/lib.rs`).

### Chosen seam and gating

Seam: **`onnx-genai-engine` `native_decode.rs`**, wrapping
`self.session.run(&bindings)` in `with_decode_pool_scope(...)` **only when
`token_ids.len() == 1`** (the M=1 single-token decode case).

Why this seam:
* The engine already depends on `onnx-runtime-ep-cpu` through the
  `native-backend` feature, and `native_decode.rs` is compiled only under that
  feature — so calling `onnx_runtime_ep_cpu::with_decode_pool_scope` needs no
  new dependency and stays feature-gated consistently. Default builds are
  unaffected.
* `token_ids.len()` is the exact M (query seq dim) at this layer, so gating on
  `== 1` cleanly separates single-token **decode** (wrapped) from multi-token
  **prefill** (left on the global pool). The CUDA path returns earlier in
  `decode()` (`self.cuda.is_some()`), so it never reaches this seam.
* The alternative seam (`onnx-runtime-session` `executor.rs`) does not have the
  M/prefill-vs-decode distinction as cleanly available and would install for
  every `run`, so it was rejected in favour of the engine seam.

## Correctness invariants (all preserved)

* **Fast-path guards unchanged**: M=1, bits=4, accuracy=4, block=32, no
  zero-point input, no g_idx. The scope wraps `session.run`; it does not touch
  kernel selection.
* **int4 semantics unchanged**: low nibble = even K, high = odd; symmetric zp
  exactly 8; per-block-32 scale; activation scale applied once; zero-pad partial
  final K block. The scope changes *where* work runs (pool residency), never the
  arithmetic.
* **Bit-identical greedy token IDs** — verified (see below).
* **No decode-pool leakage into prefill or CUDA** — the scope is entered only
  for `token_ids.len() == 1` on the CPU path.
* **Panic-safety** — the TLS flag is cleared by `DecodeResidencyGuard::drop`,
  which runs on unwind.
* **Concurrent-session safety** — `IN_DECODE_POOL` is thread-local, so each
  caller/worker thread's residency state is independent. `DECODE_POOL` is a
  shared global rayon pool; rayon supports concurrent/nested installs, and the
  guard sets/restores per worker. Two sessions decoding on two different threads
  each `install` into the shared pool and each mark residency on their own
  worker; work-stealing correctness is width-independent.
* **Opt-out respected** — `ONNX_GENAI_CPU_DECODE_THREADS=0` → `DECODE_POOL` is
  `None` → scope runs `f()` inline (flag `false`), inner `with_decode_pool` uses
  the global pool. Verified coherent.
* **Global-pool-width caveat (flagged by Deckard)** — while inside the scope,
  any op that uses the rayon **global** pool directly (not via
  `with_decode_pool`) now observes the 8-thread decode pool. For M=1 decode this
  is intended; rayon is work-stealing so correctness is width-independent — no
  decode op relies on global-pool width for correctness.

## Verification

* `cargo build --release -p onnx-runtime-ep-cpu` and the engine with
  `native-backend` — **0 warnings** on touched code.
* `cargo test -p onnx-runtime-ep-cpu` — **649 passed / 1 ignored** (baseline
  635 lib unit tests + 4 new residency tests: guard set/restore, guard
  clear-on-panic, resident inline runs on caller thread, scope marks residency
  and inner call runs inline on the scope worker).
* **Numeric parity (bit-identical):** `--tokens 32 --ep cpu --prompt "The
  capital of France is"` → first token **12095 ("Paris")** and the full 32-ID
  sequence **identical** before and after:
  `[12095, 13, 1084, 374, 279, 7772, 3283, 304, 279, 3146, 323, 279, 6722, 315,
  9625, 13, 1084, 374, 1083, 279, 1429, 94451, 3283, 304, 279, 3146, 13, 1084,
  374, 279, 7772, 3283]`.
* **Opt-out** (`ONNX_GENAI_CPU_DECODE_THREADS=0`, `--tokens 64`) — coherent,
  same greedy continuation, global-pool path.
* **Prefill not regressed** — 40-token prompt (M>1 prefill via global pool):
  baseline 14.82 tok/s vs changed 15.93 tok/s (not slowed).

## Benchmark

`profile_native --tokens 128 --warmups 2 --runs 3 --ep cpu --prompt "The
capital of France is"`, same host, interleaved:

| build             | tok/s | ms/step |
|-------------------|------:|--------:|
| baseline (origin/main) | 23.44 | 42.665 |
| residency (this change) | 24.12 | 41.452 |

Across repeated interleaved runs (warmups 1, runs 2) the change is consistently
faster but by a small margin: baseline ~22.4 tok/s, residency ~23.8 tok/s
(**≈ +3–6%**).

### Outcome: partial — below the 20% graduation threshold

The persistent-scope mechanism is correct and numerically bit-identical, but the
measured end-to-end gain (**~3–6%**) is well below the **≥ 20%** graduation
threshold.

**Where the residual time goes.** Eliminating the 121 per-op `install`
*crossings* recovered only ~3–6%, which means the external-thread-to-pool
crossing was **not** the dominant fragmentation cost. What remains per op:

1. **Per-op `par_iter` fork-join.** Each inline `with_decode_pool` still runs a
   rayon `par_iter`/`join` over the N dimension. The forward's sequential
   executor runs the 121 ops one after another on the single install worker;
   between ops the other 7 workers idle, and each `par_iter` must **re-wake and
   re-join** them. Residency removes the `install` handoff but not this
   per-op barrier/wakeup latency — which is where the 58–83 GB/s (single
   installation, one continuous parallel region) vs ~9 GB/s (121 short parallel
   regions) gap actually lives.
2. **Non-matmul glue.** Attention, RoPE, layernorm, KV materialization and the
   executor's own bookkeeping run between projections and are not addressed by
   this change.

To close the remaining gap a follow-up must remove the **per-op** fork-join,
not just the per-op install — e.g. batch/fuse the projections into fewer, larger
parallel regions, or keep the workers spinning across ops with a lighter-weight
barrier so they do not re-sleep between the 121 `par_iter`s. That is a larger
restructuring and is left for coordinator review as the next slice.
