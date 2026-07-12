# onnx-genai benchmarks

This crate keeps a fixed Criterion scenario suite for comparing runtime performance across
devices. The no-model suite needs no ONNX Runtime model or GPU:

- tokenizer encode/decode throughput
- greedy, top-k, top-p, and min-p sampling latency
- paged KV cache allocation/deallocation throughput
- seven-stage logit processor chain overhead
- llguidance grammar mask computation

Model scenarios are gated by `bench-ort` and use the committed tiny fixtures:

- end-to-end generation tokens/second (`tiny-llm`)
- prefill latency by context length (`tiny-llm-scatter`)
- static batch throughput by batch size (`tiny-llm-scatter`)

Run the comparable suite:

```bash
scripts/run_benchmarks.sh
```

Include model benchmarks:

```bash
scripts/run_benchmarks.sh --model
```

The runner prints CPU, core count, OS, and rustc followed by a concise Markdown table.
Criterion's complete reports remain in `target/criterion`. Save runner output on each
machine and diff the files; use the same commit, Rust toolchain, power profile, and
execution provider for meaningful comparisons.

For Criterion's detailed HTML reports:

```bash
cargo bench -p onnx-genai-bench --bench no_model
cargo bench -p onnx-genai-bench --features bench-ort --bench model
```
