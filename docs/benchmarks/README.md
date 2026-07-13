# Cross-runtime benchmarks

These reports compare onnx-genai with local llama.cpp-based runtimes through their common
OpenAI-compatible streaming HTTP API. The goal is to make the question “is onnx-genai
faster?” reproducible on any machine rather than relying on isolated best-case numbers.

## Methodology

- Common model family and size: Qwen2.5-0.5B-Instruct.
- Fixed explicit system prompt plus committed short- and long-context prompts.
- Greedy generation: `temperature=0`, `top_p=1`, `seed=0`, normally 64 output tokens.
- One discarded warmup and five measured runs by default.
- Reports show median and interpolated p90.
- TTFT runs from request start to the first non-empty streamed content event.
- Decode throughput excludes TTFT.
- Total latency runs through stream completion.
- Estimated prefill throughput is rendered prompt tokens divided by TTFT. It includes HTTP,
  scheduling, template processing, and first-token decode, so it is not a kernel-only metric.
- Benchmarks use realistic runtime-default threading. The ORT single-thread setting required
  by exact-equality tests is intentionally not used.

Run on a quiet machine with a stable power profile. Record the same commit, toolchain,
execution provider, model format, quantization, context limit, and runtime settings.

## Prepare the three runtimes

Build and start onnx-genai:

```bash
scripts/build_qwen.sh
ONNX_GENAI_EP=cpu cargo run --release -p onnx-genai-server -- \
  --model models/qwen2.5-0.5b \
  --model-id qwen2.5-0.5b \
  --addr 127.0.0.1:8080
```

Download the Q8_0 GGUF with LM Studio, then import the same bytes into Ollama:

```bash
~/.cache/lm-studio/bin/lms get \
  'https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/blob/main/Qwen2.5-0.5B-Instruct-Q8_0.gguf' \
  -y
Q8_PATH="$HOME/.cache/lm-studio/models/bartowski/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q8_0.gguf"
mkdir -p models/benchmarks
printf 'FROM %s\n' "$Q8_PATH" > models/benchmarks/Modelfile.q8
ollama create qwen2.5:0.5b-q8-bench -f models/benchmarks/Modelfile.q8
```

Ensure the Ollama service is running (`ollama serve` when it is not managed by the OS).

Start LM Studio and load that Q8_0 model:

```bash
~/.cache/lm-studio/bin/lms server start -p 1234
~/.cache/lm-studio/bin/lms load bartowski/qwen2.5-0.5b-instruct \
  --gpu max --context-length 2048 --parallel 1 \
  --identifier qwen2.5-0.5b-q8-bench -y
```

The 2026-07-12 baseline downloaded a 531.1 MB Q8_0 GGUF once and imported the same file
into Ollama, so both llama.cpp runtimes used identical model bytes. Q8_0 was selected as
the closest readily available GGUF quantization to the current f32 ONNX model. If Q8_0 is
impractical, override the runtime specifications and document the exact fallback quant.

## Run and save a report

```bash
scripts/compare_runtimes.sh
```

The harness probes `/v1/models`, skips unavailable runtimes clearly, and writes
`docs/benchmarks/YYYY-MM-DD-HOSTNAME.md`. For a longer periodic run:

```bash
RUNS=10 WARMUPS=2 scripts/compare_runtimes.sh
```

Runtime model IDs, formats, quantizations, and settings are configurable through
`ONNX_RUNTIME`, `OLLAMA_RUNTIME`, and `LM_STUDIO_RUNTIME`; see
`crates/onnx-genai-bench/README.md`.

## Fairness caveats

ONNX and GGUF weights may use different data types and quantization schemes, so numeric
parity is not expected. Never omit those labels, and do not cherry-pick a favorable quant.
Prefer the closest formats available and explain unavoidable differences. API-level TTFT
also includes transport and scheduling. These single-request results do not measure
concurrent serving throughput.

## Add another machine

1. Check out the same commit and use the same prompts and run counts.
2. Start equivalent model variants and record exact runtime versions/settings.
3. Run `scripts/compare_runtimes.sh`.
4. Review the generated metadata and add an honest verdict plus optimization follow-ups.
5. Keep the generated `YYYY-MM-DD-HOSTNAME.md` file; do not overwrite another machine's run.
