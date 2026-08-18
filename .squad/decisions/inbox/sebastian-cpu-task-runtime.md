# CPU EP task runtime replaces raw Rayon fan-out in native kernels

**By:** Sebastian (Performance Engineer) — 2026-08-18
**What:** Native CPU kernels no longer fan out through the global Rayon pool.
They call `onnx_runtime_ep_cpu::task_runtime`, which dispatches to ORT's
`KernelContext_ParallelFor` when running inside the plugin EP and to a purpose
-built native pool otherwise. RoPE, Softmax, Transpose and the elementwise
activation fallback are converted; the remaining raw-Rayon sites are unchanged
and still work.

**Why:**

- Rayon parks its workers between parallel regions. Measured on this host, a
  fan-out costs 67 µs back-to-back but 226 µs when it follows a 20 µs gap —
  which is exactly the shape of decode. Documented in
  `docs/benchmarks/2026-08-15-cpu-ep-vs-ort-attention-moe.md` §26/§27.
- The new pool holds its workers in an adaptive spin (20 µs → 500 µs, doubling
  on a catch and halving on a park) so back-to-back and decode-gap dispatch cost
  the same. Measured p50 4.8 µs at a 0 µs gap and 4.9 µs at a 100 µs gap —
  14× and 47× better than Rayon's two numbers, and, more importantly, flat.
- Inside the plugin EP we do not run our own threads at all. Using ORT's own
  intra-op pool is the only way to avoid oversubscribing a host that has already
  sized its pool, and it makes our kernels honour the session's
  `intra_op_num_threads` the way every other ORT kernel does.

**Consequences / rules this sets:**

1. **New parallel kernels use `task_runtime`, not `rayon`.** `for_each_range`,
   `chunk_runs_mut` and `chunks_mut` cover the existing shapes.
2. **Width is inferred, and SMT-capped only above 8 hardware threads.** An
   explicit budget (`set_task_thread_budget`, `ONNX_GENAI_CPU_TASK_THREADS`) is
   honoured exactly. The floor is empirical: below 16 logical CPUs the second
   SMT sibling still pays for these memory-bound kernels, above it does not.
3. **No env-var test hooks in production paths.** Determinism comes from
   `task_runtime::testing` (`force_serial`, `isolated_pool`, `counters`,
   `planned_backend`).
4. **Per-vector SIMD helpers that take a closure must be `#[inline(always)]`.**
   Not a hint: `avx2::map_ps` losing its inline made `Tanh` and `Sigmoid` 2×
   slower on inputs that never reach the parallel path, and the trigger was an
   edit in two unrelated files. `codegen-units = 1` does not prevent this.
