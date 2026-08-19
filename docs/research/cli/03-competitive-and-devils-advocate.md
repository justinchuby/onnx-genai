# CLI Competitive Analysis + Devil's Advocate

Date: 2026-07-27  
Author: Fact Checker  
Scope: `crates\onnx-genai-cli` only; no CLI source changes.

## Verification legend

- ✅ Verified — confirmed from source/docs.
- ⚠️ Unverified — plausible but not confirmed in authoritative docs/source during this pass.
- ❌ Contradicted — evidence says the opposite.

## 1. Current `onnx-genai` CLI surface

Source inspected: `crates\onnx-genai-cli\src\lib.rs`, `crates\onnx-genai-cli\src\commands.rs`, and `crates\onnx-genai-server\src\cli.rs`.

✅ Verified top-level subcommands:

| Subcommand | Current purpose | Flags / args found |
|---|---|---|
| `serve` | OpenAI-compatible HTTP server | One required source among positional `MODEL`, `--model`, `--models-dir`, `--models-config`; plus `--model-id`, `--node-id`, `--addr`, `--max-output-tokens`, `--max-sessions`, `--max-queue-depth`, `--enable-debug-endpoints`, `--enable-admin-endpoints`, `--max-loaded-models`, `--kv-cache-dtype`; shared engine/CPU flags. |
| `generate` | One-shot text generation; also text-to-image/audio output | Positional `model`; `--prompt`/`-p`; sampling `--max-new-tokens`, `--temperature`, `--top-p`, `--top-k`, repeated `--stop`, `--raw`; attachments `--image`, `--audio`; shared engine/CPU flags; text/image/audio output controls `--stream`, `--output-image`, `--negative-prompt`, `--steps`, `--guidance-scale`, `--seed`, `--height`, `--width`, `--batch-size`, `--tokenizer`, `--text-encoder`, `--vae-decoder`, `--vae-scaling-factor`, `--output-audio`, `--sample-rate`. |
| `run` | Interactive generation REPL | Positional `model`; shared sampling flags; `--image`, `--audio`; shared engine/CPU flags. REPL slash commands: `/help`, `/reset`, `/raw`, `/stats`, `/pages`, `/profile [on|off|trace <path>|trace off|verbosity <level>]`, `/model [path]`, `/session`, `/ep [provider]`, `/backend [auto|ort|native]`, `/system [text]`, `/image [path] [prompt]`, `/audio [path] [prompt]`. |
| `show` | Inspect a model's resolved files/metadata | Positional `model`. |
| `list` / alias `ls` | List model directories | `--models-dir` / `ONNX_GENAI_MODELS_DIR`. |
| `transcribe` | Speech-to-text from files or live stdin | Positional `AUDIO...`; `--language`, `--format text|json|srt`, `--segment-seconds`, `--silence-seconds`, `--silence-threshold`, `--min-segment-seconds`, `--sample-rate`, `--channels`, `--max-new-tokens`; shared engine/CPU flags. |
| `version` | Version and execution providers | No command-local flags. |

✅ Verified global profiling flags: `--profile`, `--profile-json PATH`, `--profile-trace PATH`.

✅ Verified shared engine/CPU flags, identical on `serve`, `generate`, `run` and `transcribe`: `--backend auto|ort|native`, `--device auto|cpu|cuda[:N]`, `--vram-limit`, `--host-ram-limit`, `--cpu-cores`.

## 2. Comparable CLI surfaces checked

### Ollama

✅ Verified from official CLI docs and source/API index: `ollama run`, `launch`, `pull`, `rm`, `ls`, `signin`, `signout`, `create`, `ps`, `stop`, `serve`; source also contains `show MODEL`, `cp SOURCE DESTINATION`, and `push MODEL`. Sources: <https://docs.ollama.com/cli>, <https://docs.ollama.com/llms.txt>, <https://github.com/ollama/ollama/blob/main/cmd/cmd.go>.  
Standout UX: model registry pull/run flow, first-class local model lifecycle (`pull`, `create`, `rm`, `ps`, `stop`), and `launch` integrations for coding tools.

### llama.cpp (`llama-cli` / `llama-server`)

✅ Verified from upstream generated help/readme: `llama-cli` offers direct generation and interactive prompting with a very deep flag surface (`--model`, `--prompt`, `--threads`, `--ctx-size`, `--predict`, `--hf-repo`, device/offload/KV/cache/log flags, etc.). `llama-server` provides OpenAI-compatible chat/completions/responses/embeddings routes, Anthropic Messages compatibility, web UI, continuous batching, multimodal support, monitoring, schema-constrained JSON, tool/function calling, and speculative decoding. Sources: <https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/cli/README.md>, <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>.  
Standout UX: maximum control for systems users plus built-in web UI/server features.

### vLLM

✅ Verified from stable CLI guide: `vllm {chat,complete,serve,launch,bench,collect-env,run-batch}`. `serve` starts the OpenAI-compatible API server; `chat`/`complete` can connect to a running server and support `--quick` and `--stats`; `bench` has `{latency,serve,throughput}`; `run-batch` reads local or remote JSONL; `collect-env` gathers environment details. Source: <https://docs.vllm.ai/en/stable/cli/>.  
Standout UX: production serving, built-in benchmark suite, batch file processing, and searchable grouped help (`vllm serve --help=max`).

### `mlx_lm`

✅ Verified from README and package entry points: commands include `mlx_lm.generate`, `chat`, `server`, `convert`, `lora`, `fuse`, `upload`, `manage`, plus quantization/eval/benchmark/cache utilities such as `cache_prompt`, `benchmark`, `evaluate`, `perplexity`, `awq`, `gptq`, `dynamic_quant`, `share`. Sources: <https://github.com/ml-explore/mlx-lm>, <https://raw.githubusercontent.com/ml-explore/mlx-lm/main/setup.py>.  
Standout UX: Hugging Face integration, conversion/quantization/upload loop, LoRA fine-tuning, and prompt-cache CLI.

### LM Studio `lms`

✅ Verified from official docs: `lms chat`, `get`, `load`, `unload`, `ls`, `ps`, `import`, `server`, `log`, `runtime`, `daemon`, `link`, `clone`, `push`, `dev`, `login`; `lms load` supports GPU/context options and identifiers. Source: <https://lmstudio.ai/docs/cli>.  
Standout UX: desktop + headless bridge with explicit load/unload/runtime/server control.

### Microsoft `onnxruntime-genai`

✅ Verified: the primary surface is Python/C#/C/C++/Java APIs, with Python package install variants for CPU/DirectML/CUDA; the repo documents `python -m onnxruntime_genai.models.builder --help` for model export/conversion and examples for HF, disk, finetuned, quantized, and GGUF inputs. Sources: <https://onnxruntime.ai/docs/genai/>, <https://onnxruntime.ai/docs/genai/howto/install.html>, <https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/src/python/py/models/README.md>, <https://github.com/microsoft/onnxruntime-genai>.  
Standout UX: model-builder/export tooling rather than a broad end-user run/server CLI.

## 3. Feature matrix

| Capability | `onnx-genai` CLI | Ollama | llama.cpp | vLLM | `mlx_lm` |
|---|---|---|---|---|---|
| One-shot generation | ✅ `generate --prompt` (source) | ✅ `ollama run MODEL [PROMPT]` ([docs](https://docs.ollama.com/cli)) | ✅ `llama-cli -m ... -p ...` ([cli](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/cli/README.md)) | ⚠️ Via `vllm complete --quick` against server; not offline one-binary generation ([docs](https://docs.vllm.ai/en/stable/cli/)) | ✅ `mlx_lm.generate` ([README](https://github.com/ml-explore/mlx-lm)) |
| Interactive chat / REPL | ✅ `run` + slash commands (source) | ✅ `ollama run` chat ([docs](https://docs.ollama.com/cli)) | ✅ interactive flags in `llama-cli` ([cli](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/cli/README.md)) | ✅ `vllm chat` against server ([docs](https://docs.vllm.ai/en/stable/cli/)) | ✅ `mlx_lm.chat` ([README](https://github.com/ml-explore/mlx-lm)) |
| OpenAI-compatible server | ✅ `serve` (source) | ✅ REST/OpenAI compatibility documented ([api index](https://docs.ollama.com/llms.txt)) | ✅ chat/completions/responses/embeddings ([server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)) | ✅ core `vllm serve` ([docs](https://docs.vllm.ai/en/stable/cli/)) | ✅ `mlx_lm.server` ([README](https://github.com/ml-explore/mlx-lm)) |
| Model download / registry | ❌ No verified pull/get/download command | ✅ `pull`, `push`, registry/library ([docs](https://docs.ollama.com/cli)) | ✅ HF and Docker repo flags (`--hf-repo`, `--docker-repo`) ([cli](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/cli/README.md)) | ✅ Uses HF model names in `serve`; no local registry lifecycle in CLI docs ([docs](https://docs.vllm.ai/en/stable/cli/)) | ✅ HF Hub default, `--model`, upload ([README](https://github.com/ml-explore/mlx-lm)) |
| Model lifecycle management | ⚠️ `list`, `show`; server has admin load/unload endpoints if enabled, but CLI has no `load/unload/rm/ps` | ✅ `ls`, `ps`, `stop`, `rm`, `create`, `show`, `cp` ([docs/source](https://github.com/ollama/ollama/blob/main/cmd/cmd.go)) | ⚠️ cache-list and server cache flags; fewer registry-style lifecycle commands ([cli](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/cli/README.md)) | ⚠️ serve/bench/run-batch; no model rm/pull lifecycle in CLI guide | ✅ `manage`, plus package-level model commands ([setup.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/setup.py)) |
| Conversion / quantization / fine-tuning | ❌ Not in CLI | ✅ `create` custom model; import docs exist ([docs](https://docs.ollama.com/cli)) | ⚠️ llama.cpp has separate tooling in the broader project, but this pass verified only `llama-cli`/`llama-server` docs | ❌ CLI guide is serve/chat/bench/batch, not conversion | ✅ `convert`, quant commands, `lora`, `fuse`, `upload` ([setup.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/setup.py)) |
| Multimodal input | ✅ `--image`, `--audio`, `/image`, `/audio` (source) | ✅ vision example in CLI docs ([docs](https://docs.ollama.com/cli)) | ✅ multimodal server support ([server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)) | ✅ docs include multimodal feature area; CLI guide not deeply enumerated ([docs](https://docs.vllm.ai/en/stable/cli/)) | ⚠️ MLX LM is primarily LLM text; no verified general image/audio CLI in reviewed README |
| Text-to-image / text-to-audio / transcription | ✅ `generate --output-image`, `--output-audio`; `transcribe` | ❌ Not verified as core Ollama CLI surface | ❌ Not verified as core `llama-cli`/`llama-server` surface | ⚠️ Speech-to-text entrypoint appears in API nav, but not top-level CLI guide | ❌ Not verified in reviewed README/setup |
| Embeddings | ❌ No verified CLI command | ✅ `ollama run embedding...`; API embeddings ([docs](https://docs.ollama.com/cli)) | ✅ embeddings route in server ([server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)) | ⚠️ Not highlighted in reviewed CLI guide | ❌ Not verified as CLI command |
| Benchmarks | ❌ No CLI bench subcommand | ⚠️ API usage stats, but no verified bench command | ⚠️ Server monitoring verified; separate benchmark binaries not checked in depth in this pass | ✅ `vllm bench latency|serve|throughput` ([docs](https://docs.vllm.ai/en/stable/cli/)) | ✅ `mlx_lm.benchmark` entry point ([setup.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/setup.py)) |
| Profiling / tracing | ✅ `--profile`, JSON, Perfetto trace, REPL `/profile` (source) — onnx-genai leads here for built-in local run profiling | ⚠️ No equivalent verified CLI profiler | ✅ `--perf` internal timings ([cli](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/cli/README.md)) | ✅ `--stats`, benchmarks, collect-env; grouped help ([docs](https://docs.vllm.ai/en/stable/cli/)) | ⚠️ benchmark/perplexity exist, but no verified trace profile surface |
| Prompt/KV caching | ✅ REPL `/pages`, memory/KV tuning visible; no reusable prompt-cache file command | ⚠️ Runtime model retention; no verified prompt-cache command | ✅ cache/offload flags verified in CLI help; reusable prompt-cache UX not fully checked | ⚠️ Prefix/KV cache is core vLLM technology, but this pass verified no dedicated CLI cache command | ✅ `mlx_lm.cache_prompt` ([README/setup.py](https://github.com/ml-explore/mlx-lm)) |
| Production operations | ✅ Multi-model serve, queue/session limits, admin/debug endpoints | ✅ daemon/server plus model lifecycle | ✅ web UI, monitoring, batching, continuous decode | ✅ strongest: serve, bench, run-batch, collect-env | ⚠️ local server, less production-oriented than vLLM |

Where `onnx-genai` already leads: unified ONNX-backed text/image/audio/transcription surface; built-in Perfetto/JSON profiling; explicit CPU/resource knobs; REPL commands for EP/backend switching and KV page inspection; metadata-centric `show` for packaged ONNX models.

## 4. Devil's advocate: the case against CLI investment now

Steelman: this repo's primary front is CUDA/perf and model enablement, not a consumer UX product. A polished CLI will not compensate for missing model support, slow decode, fragile EP coverage, or packaging friction. The current CLI is already sufficient as a demo harness and smoke-test entrypoint: it can generate, run a REPL, serve OpenAI-compatible HTTP, inspect packages, transcribe, and profile. The users most likely to matter in the next 30 days are contributors validating kernels/models and integrators hitting the HTTP/Rust/Python surfaces, not end users choosing between Ollama and LM Studio.

Who uses this CLI today? Likely: maintainers, benchmarkers, model-package authors, and early adopters validating ONNX packages locally. It is not yet obviously a shipping product surface like Ollama's `pull/run` loop or LM Studio's desktop/headless ecosystem.

## 5. Load-bearing assumptions

1. CLI improvements will increase adoption despite no built-in model registry/download path.
2. End users want an ONNX-native local inference CLI instead of using Ollama/LM Studio for UX and this repo for runtime/server integration.
3. The team can maintain broad CLI semantics without slowing CUDA/perf/model work.
4. The package format and model zoo are stable enough to expose through user-facing commands.
5. Competitive parity matters more than differentiated ONNX metadata, server, and profiling strengths.

## 6. 30-day pre-mortem

A month from now, the team has shipped cleaner CLI commands but CUDA decode and model-enablement milestones slipped. New commands still require users to manually locate compatible ONNX packages, so the experience loses to `ollama run gemma4` in the first minute. Documentation and tests expand around UX glue while the actual runtime remains the blocker. Worse, polishing user-facing commands creates a support burden: users now report model download, conversion, quantization, and platform issues that the CLI cannot solve.

## 7. Alternative direction

Prioritize one of these before broad CLI polish:

- Python/server client path: make the OpenAI-compatible server and Python bindings the primary integration surface, with examples, smoke tests, and packaging reliability.
- Model acquisition path: invest in documented model packages and/or builder/preprocess tooling so `onnx-genai` has something as simple as `ollama run` to point at later.
- Perf-first CLI: keep CLI improvements narrow to benchmark/profile ergonomics (`bench`, reproducible run manifests, environment capture), because that directly supports the primary CUDA/perf front.

## 8. Fact-check verdict

✅ Verified competitive gaps:

1. No verified model pull/get/download/registry workflow comparable to Ollama, LM Studio, MLX Hub, or llama.cpp HF flags.
2. No CLI conversion/quantization/fine-tuning loop comparable to `mlx_lm.convert/lora/fuse/upload` or ONNX Runtime GenAI's model builder.
3. No dedicated benchmark/batch/evaluation command comparable to `vllm bench`, `vllm run-batch`, or `mlx_lm.benchmark`.

Strongest counter-argument: broad CLI parity is premature if the CLI is mainly a maintainer/demo harness; the highest-leverage work is still model enablement, performance, packaging, and the server/Python integration path.
