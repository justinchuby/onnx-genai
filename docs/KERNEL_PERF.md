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
both harnesses:

| Workers | Rust MatMul | ORT 1.27 CPU EP | Rust / ORT |
|---:|---:|---:|---:|
| 1 | 2.801 ms | 131 µs | 21.4× |
| 8 | 502 µs | 30.6 µs | 16.4× |

Rust runs inside dedicated 1- and 8-worker Rayon pools. ORT pins the matching
intra-op count and uses one inter-op thread for its one-node model. Thus the
honest current gap is approximately 16–21×, not an unqualified default-thread
comparison. Add, ReduceMean, and Gather are single-threaded internally and are
reported as such. The acceptance bar above and the recommendation to port MLAS
GEMM/MatMul next remain unchanged.
