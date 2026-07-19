# CPU kernel performance and numeric quality

The pure-Rust CPU EP has a reusable Criterion harness and fixed numeric golden
tests in
[`crates/onnx-runtime-ep-cpu/benches/`](../crates/onnx-runtime-ep-cpu/benches/README.md).
It currently covers Add, ReduceMean, Gather, and MatMul at small, medium, and
large inference-oriented shapes, including f16/bf16 where supported.

The standing acceptance bar for a kernel rewrite is:

- no numeric-regression failure;
- no slowdown versus ONNX Runtime's CPU EP for matching operation, shape, and
  dtype; and
- no loss of the Rust EP's broader dtype coverage.

The next optimization phase should port the highest-leverage MLAS primitive
first: GEMM/MatMul (then quantized MatMul). MatMul dominates transformer
prefill/projection work, already has representative benchmark coverage, and
provides a backend seam that can be changed without altering the EP contract.
Every port should add relevant shapes/dtypes to both the Criterion and ORT
baseline matrices before replacing the implementation.

For the medium f32 MatMul shape (`32×512×512`), the current warm-cache,
allocation-outside-timing comparison is thread-matched and self-reported by
both harnesses. The Rust EP now defaults to the built-in **`SimdX86`** backend
(MLAS-style packed SIMD f32 SGEMM: `6×16` AVX2/FMA register microkernel, panel
packing, K/N cache blocking, column-strip Rayon parallelism), runtime-selected
via `is_x86_feature_detected!("avx2"/"fma")` with a safe fallback to the
`Generic` blocked GEMM on non-AVX2 hosts:

| Workers | Rust (Generic, old) | Rust (SimdX86, new) | ORT 1.27 CPU EP | New Rust / ORT |
|---:|---:|---:|---:|---:|
| 1 | 2.801 ms | 285 µs | 131 µs | 2.2× |
| 8 | 502 µs | ~155 µs | 30.6 µs | ~5× |

That is a **9.8× single-thread** and **3.2× eight-thread** speedup over the
previous pure-Rust GEMM, closing the ORT gap from ~16–21× to ~2–5×. The larger
`32×1024×1024` f32 shape (no ORT baseline recorded) runs 1.27 ms → 327 µs from 1
to 8 workers. f16/bf16 MatMul also gets faster (same SIMD GEMM after widening);
their remaining cost is the pre-existing widen-to-f32 conversion, not the GEMM.

Remaining gap versus ORT: (1) no AVX-512 microkernel yet (AVX2/FMA only), (2) a
single `6×16` kernel rather than MLAS's shape-specialized kernel family with
software prefetch, and (3) eight-thread scaling is bounded on the small medium
shape (M=32, ~8.4 MFLOP) by per-strip B repacking and Rayon task overhead; MLAS
shares a single packed B panel and partitions work more finely. Larger shapes
scale better (~3.9× at 8 threads on the large shape).

Rust runs inside dedicated 1- and 8-worker Rayon pools. ORT pins the matching
intra-op count and uses one inter-op thread for its one-node model. Thus the
honest current gap is approximately 2–5×, not an unqualified default-thread
comparison. Add, ReduceMean, and Gather are single-threaded internally and are
reported as such. The next port target remains quantized MatMul.
