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
