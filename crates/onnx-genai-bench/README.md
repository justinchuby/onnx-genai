# onnx-genai benchmarks

This crate keeps a fixed Criterion scenario suite for comparing runtime performance across
devices. The no-model suite needs no ONNX Runtime model or GPU:

- tokenizer encode/decode throughput
- greedy, top-k, top-p, and min-p sampling latency
- paged KV cache allocation/deallocation throughput
- seven-stage logit processor chain overhead
- llguidance grammar mask computation

Model scenarios are gated by `bench-ort` and use the committed tiny fixtures:

- end-to-end generation tokens/second (`tiny-llm-scatter`)
- prefill latency by context length (`tiny-llm-scatter`)
- static batch throughput by batch size (`tiny-llm-scatter`)
- continuous (dynamic) batch throughput by admission cap `max_batch` (`tiny-llm-scatter`)
- prefix-cache prefill speedup: cold versus warm (prefix-primed) prefill (`tiny-llm-scatter`)

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

## Engine/ORT versus native nxrt profiling

The opt-in profiling binaries measure different layers:

- `bench-ort` / `profile_decode` measures token generation through the
  `onnx-genai` engine and `onnx-genai-ort` (ORT CUDA EP).
- `bench-native` / `profile_native` measures token generation through the
  native nxrt decoder-with-past adapter and the same engine decode loop as ORT,
  with forward passes through `onnx-runtime-session::InferenceSession::run`.

These results are **not yet comparable head-to-head**. The ORT path has a
generation loop, I/O binding, and KV-cache decode flow. The native session
currently exposes only `run`, so it reports runs/s and ms/run rather than tok/s.
For CLI consistency, `--tokens` is the number of repeated inference calls per
measured run; it does not mean generated tokens.

```bash
cargo run --release -p onnx-genai-bench --features bench-native \
  --bin profile_native -- \
  --model crates/onnx-runtime-session/tests/fixtures/bert_toy \
  --tokens 8 --warmups 1 --runs 3 --ep cpu
```

CUDA kernels expose per-op `cuda_graph_compatible()` predicates, aggregated by
the CUDA EP's `subgraph_graph_capturable()` eligibility gate. The
`onnx-runtime-session` is currently CPU-only and has no session-level CUDA graph
capture/replay API. `--ep cuda --cuda-graph` reports that limitation rather than
silently using CPU or faking graph replay. Unsupported operators retain the
native session's actionable missing-kernel error.

## Cross-runtime HTTP comparison

`compare` measures the deployment-facing OpenAI streaming API against other local runtimes.
It records TTFT, decode throughput, total latency, and estimated prefill throughput for fixed
short- and long-context Qwen2.5 prompts. Unavailable servers or unloaded models are reported
and skipped instead of failing the run.

With onnx-genai, Ollama, and LM Studio already serving their models:

```bash
scripts/compare_runtimes.sh
```

The script writes `docs/benchmarks/YYYY-MM-DD-HOSTNAME.md`. Override `RUNS`, `WARMUPS`,
`MAX_TOKENS`, `OUTPUT`, or any full runtime specification:

```bash
RUNS=10 \
ONNX_RUNTIME='onnx-genai|http://127.0.0.1:8080/v1|qwen2.5-0.5b|ONNX f32|CPU EP; default threads' \
scripts/compare_runtimes.sh
```

Each runtime specification is `NAME|BASE_URL|MODEL|FORMAT|SETTINGS`. Keep format and
quantization labels exact; a comparison across unlabeled quantizations is not meaningful.
