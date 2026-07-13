# Cross-runtime benchmarks

These reports compare onnx-genai with local llama.cpp-based runtimes through their common
OpenAI-compatible streaming HTTP API. The goal is to make the question “is onnx-genai
faster?” reproducible on any machine rather than relying on isolated best-case numbers.

## Methodology

- Primary comparisons use one SHA-256-identified GGUF file as the source for every
  runtime. llama.cpp runtimes load that file directly; Mobius converts those same bytes
  to ONNX with `--keep-quantized`, preserving transformer projection weights as
  `MatMulNBits`.
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

## Primary fair comparison: same-source GGUF

The verified periodic benchmark currently uses the official Q4_0 file. Q4_K_M is
preferred for typical llama.cpp deployment, but the current Mobius Qwen conversion fails
with a quantized weight-shape mismatch, so do not silently substitute or fabricate a
Q4_K_M result.

```bash
mkdir -p models/gguf
curl -L --fail -o models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf
shasum -a 256 models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf
```

The expected SHA-256 for the verified file is
`7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed`.

Convert it with Mobius. `--keep-quantized` is required; without it the command
dequantizes the model and the comparison is invalid.

```bash
cd /Users/justinc/Documents/GitHub/mobius
PYTHONPATH=src conda run -n onnx python -m mobius build-gguf \
  /Users/justinc/Documents/GitHub/onnx-genai/models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf \
  --output /Users/justinc/Documents/GitHub/onnx-genai/models/qwen2.5-0.5b-q4-onnx \
  --keep-quantized
```

Require the `Quantized mode: preserving GGUF quantization as MatMulNBits...` message and
verify the graph contains `com.microsoft.MatMulNBits` nodes. Copy the matching Qwen
tokenizer package files into the output directory, then start onnx-genai:

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
cp models/qwen2.5-0.5b/{tokenizer.json,tokenizer_config.json,vocab.json,merges.txt,genai_config.json} \
  models/qwen2.5-0.5b-q4-onnx/
ONNX_GENAI_EP=cpu cargo run --release -p onnx-genai-server -- \
  --model models/qwen2.5-0.5b-q4-onnx \
  --model-id qwen2.5-0.5b-q4-1to1 \
  --addr 127.0.0.1:8080
```

Import the same file into Ollama:

```bash
mkdir -p models/benchmarks
printf 'FROM %s\n' "$PWD/models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf" \
  > models/benchmarks/Modelfile.q4-1to1
ollama create qwen05-q4-1to1 -f models/benchmarks/Modelfile.q4-1to1
```

Ensure the Ollama service is running (`ollama serve` when it is not managed by the OS).

LM Studio CLI versions that do not accept arbitrary paths require the file in their model
directory. A hard link keeps the bytes identical without another copy:

```bash
LM_DIR="$HOME/.cache/lm-studio/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF"
mkdir -p "$LM_DIR"
ln models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf \
  "$LM_DIR/qwen2.5-0.5b-instruct-q4_0.gguf"
lms server start -p 1234
lms load 'qwen2.5-0.5b-instruct@q4_0' \
  --gpu max --context-length 2048 --parallel 1 \
  --identifier qwen05-q4-1to1 -y
```

Current Mobius preserves the 168 transformer projection matrices as Q4_0 but dequantizes
the quantized token embedding and output head to fp32. Record this importer limitation in
every report until those tensors are preserved too. Also record that ORT CPU EP versus
llama.cpp Metal compares deployable runtime/backend stacks, not an identical device EP.

## Run and save a report

```bash
ONNX_RUNTIME='onnx-genai|http://127.0.0.1:8080/v1|qwen2.5-0.5b-q4-1to1|same-source ONNX MatMulNBits Q4_0|CPU EP; ORT default threads' \
OLLAMA_RUNTIME='Ollama (llama.cpp)|http://127.0.0.1:11434/v1|qwen05-q4-1to1:latest|exact source GGUF Q4_0|Metal/default threads' \
LM_STUDIO_RUNTIME='LM Studio|http://127.0.0.1:1234/v1|qwen05-q4-1to1|exact source GGUF Q4_0|Metal; context=2048; parallel=1' \
scripts/compare_runtimes.sh
```

The harness probes `/v1/models`, skips unavailable runtimes clearly, and writes
`docs/benchmarks/YYYY-MM-DD-HOSTNAME.md`. For a longer periodic run:

```bash
RUNS=10 WARMUPS=2 \
ONNX_RUNTIME='onnx-genai|http://127.0.0.1:8080/v1|qwen2.5-0.5b-q4-1to1|same-source ONNX MatMulNBits Q4_0|CPU EP; ORT default threads' \
OLLAMA_RUNTIME='Ollama (llama.cpp)|http://127.0.0.1:11434/v1|qwen05-q4-1to1:latest|exact source GGUF Q4_0|Metal/default threads' \
LM_STUDIO_RUNTIME='LM Studio|http://127.0.0.1:1234/v1|qwen05-q4-1to1|exact source GGUF Q4_0|Metal; context=2048; parallel=1' \
scripts/compare_runtimes.sh
```

Runtime model IDs, formats, quantizations, and settings are configurable through
`ONNX_RUNTIME`, `OLLAMA_RUNTIME`, and `LM_STUDIO_RUNTIME`; see
`crates/onnx-genai-bench/README.md`.

The canonical same-source invocation is shown in
`2026-07-12-JustindeMacBook-Pro-1to1-q4.md`.

## Fairness caveats

Never call a run 1:1 unless all runtimes derive from the same SHA-256-identified source
file and the report states exactly which tensors retain quantization after conversion.
API-level TTFT includes transport and scheduling. These single-request results do not
measure concurrent serving throughput.

The earlier `2026-07-12-JustindeMacBook-Pro.md` report is a deployment baseline only:
fp32 ONNX versus Q8_0 GGUF. Keep it as a labeled historical footnote, not the primary
runtime comparison.

## Add another machine

1. Check out the same commit and use the same prompts and run counts.
2. Start equivalent model variants and record exact runtime versions/settings.
3. Run the same-source invocation above.
4. Review the generated metadata and add an honest verdict plus optimization follow-ups.
5. Keep the generated `YYYY-MM-DD-HOSTNAME.md` file; do not overwrite another machine's run.
