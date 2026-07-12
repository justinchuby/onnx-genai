# onnx-genai

A Rust inference runtime for generative AI models, built on ONNX Runtime.

**Reference implementation** of the [ONNX Inference Metadata Standard](https://github.com/onnx/onnx/issues/8184).

## Features

- **Generation:** greedy and categorical sampling with temperature, top-k, top-p,
  min-p, repetition, frequency, and presence controls.
- **Speculative decoding:** separate draft models and model-free prompt
  lookup/n-gram proposals, with greedy target verification and KV rewind.
- **Structured generation:** complete JSON plus llguidance-backed JSON Schema,
  regex, and Lark constraints; fill-in-the-middle (FIM) for compatible coder
  tokenizers.
- **Agent serving:** OpenAI-compatible chat completions, SSE streaming, model
  discovery, persistent sessions, Hugging Face/MiniJinja chat templates
  (including ChatML-style models), and tool calling (`tools`, `tool_choice`,
  and `<tool_call>` parsing).
- **Concurrency:** multi-session generation, prefix reuse, priority scheduling,
  swap preemption, and continuous static-cache batching. A tiny CPU fixture
  measured about 6.2x aggregate fixed-batch throughput; this is not a
  real-model GPU performance claim.
- **KV and long context:** paged allocation, copy-on-write fork, rewind,
  prefix cache, tiered storage, and opt-in int8 KV pages. Mobius static-cache
  models use runtime-owned in-place KV buffers for O(1) work per decoded token
  with respect to context length.
- **Pipelines and models:** metadata-declared multi-model pipelines, a tested
  tiny vision-language pipeline fixture, and real Qwen2.5-0.5B-Instruct and
  TinyStories generation built through Mobius.
- **Execution providers:** select CPU, WebGPU, or CoreML with
  `ONNX_GENAI_EP`; unavailable providers warn and fall back to CPU.
- **Extensibility:** public `Sampler`, `SpeculativeProposer`, logit processor
  registry, and KV/pipeline APIs, plus an internal `DecodeBackend` seam shared
  by dynamic and static-cache decoding.

## Architecture

```text
OpenAI HTTP server / CLI / Rust facade
                    │
     chat templates, tools, constraints
                    │
 Generation engine + shared decode loop
     ├── speculative proposers/samplers
     ├── scheduler + continuous batching
     └── pipeline executor
                    │
 KV management: pages, prefix trie, tiering,
 int8 storage, sessions, static-cache buffers
                    │
 ONNX Runtime sessions + Hugging Face tokenizers
```

The paged KV manager currently supplies allocation, sharing, tiering, and
materialization; true paged-attention kernels are not yet implemented.

## Quick Start

### Build a model with Mobius

`scripts/build_qwen.sh` builds `Qwen/Qwen2.5-0.5B-Instruct` using a local
[Mobius](https://github.com/justinchuby/mobius) checkout:

```bash
MOBIUS_DIR=/path/to/mobius scripts/build_qwen.sh
# Output: models/qwen2.5-0.5b
```

For bounded, in-place KV storage and efficient long-context decode, export a
static-cache model. `MAX_SEQ_LEN` fixes the cache capacity at build time:

```bash
MOBIUS_DIR=/path/to/mobius STATIC_CACHE=1 MAX_SEQ_LEN=8192 scripts/build_qwen.sh
# Output: models/qwen2.5-0.5b-scatter
```

### Run the CLI

```bash
cargo build --release -p onnx-genai -p onnx-genai-server

./target/release/onnx-genai generate \
  --model models/qwen2.5-0.5b \
  --max-new-tokens 64 \
  --temperature 0 \
  --stream \
  "Write a short Rust hello-world program."
```

### Run the OpenAI-compatible server

```bash
./target/release/onnx-genai-server \
  --model models/qwen2.5-0.5b \
  --model-id qwen2.5-0.5b \
  --addr 127.0.0.1:8080
```

Available routes are `GET /health`, `GET /v1/models`,
`POST /v1/chat/completions`, `POST /v1/sessions`, and
`DELETE /v1/sessions/{id}`. Pass a session id as `X-Session-Id` on chat
requests to reuse persistent context.

Chat with constrained JSON output (`"stream": true` enables SSE):

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "Reply with a JSON greeting."}],
    "response_format": {"type": "json_object"},
    "temperature": 0,
    "max_tokens": 64,
    "stream": false
  }'
```

Tool use, with a grammar-enforced required function call:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "What is the weather in Seattle?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"],
          "additionalProperties": false
        }
      }
    }],
    "tool_choice": "required",
    "temperature": 0,
    "max_tokens": 128
  }'
```

The server returns parsed calls in OpenAI `message.tool_calls`; the client is
responsible for executing tools and sending the result in a later message with
role `tool`.

Any OpenAI-compatible agent can use `http://127.0.0.1:8080/v1` as its base
URL and `qwen2.5-0.5b` as the model. The repository also includes a constrained
Hermes-style demonstration harness:

```bash
python3 scripts/coding_agent.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model qwen2.5-0.5b \
  --workdir target/coding-agent-workspace \
  --clean \
  --task "Create hello.py, run it, and report the output."
```

### Execution provider

```bash
ONNX_GENAI_EP=cpu ./target/release/onnx-genai-server --model models/qwen2.5-0.5b
ONNX_GENAI_EP=webgpu ./target/release/onnx-genai-server --model models/qwen2.5-0.5b
ONNX_GENAI_EP=coreml ./target/release/onnx-genai-server --model models/qwen2.5-0.5b
```

## Security

The server defaults to `127.0.0.1:8080` and has **no built-in
authentication**. Do not bind it to a non-loopback address unless it is behind
an authenticated reverse proxy. Server caps limit requested output tokens and
resident sessions (`--max-output-tokens`, `--max-sessions`), session ids come
from the OS CSPRNG, context length is checked when declared by model metadata,
and automatically downloaded ONNX Runtime archives require pinned SHA-256
checksums. Tool execution is always the client's responsibility.

## Coverage

Run `scripts/coverage.sh` to install the required LLVM tools when missing and
print workspace coverage. The final `TOTAL` row is the overall percentage; the
preceding rows identify low-coverage source files. For annotated source, run
`scripts/coverage.sh --html --open` instead of the default summary output.

## Project Structure

```text
crates/
├── onnx-genai/            # Main library facade and CLI
├── onnx-genai-metadata/   # Inference metadata parser and validation
├── onnx-genai-kv/         # Paged, prefix, tiered, and quantized KV storage
├── onnx-genai-scheduler/  # Priority scheduling and preemption
├── onnx-genai-ort/        # ONNX Runtime, tokenizers, templates, decode sessions
├── onnx-genai-engine/     # Generation, constraints, speculation, pipelines
└── onnx-genai-server/     # OpenAI-compatible HTTP server
```

## Status

**Phases 1-3 are complete. Phase 4 is substantially complete.**

Completed work includes end-to-end ORT/tokenizer generation, the CLI and HTTP
server, multi-session/prefix reuse, paged/tiered/int8 KV management,
priority/preemption, draft-model and prompt-lookup speculation, structured
decoding, FIM, chat templates and tool use, multi-model/VLM pipeline execution,
static-cache O(1)-per-token long-context decode, and continuous batched serving.
The OpenAI tool loop has been verified end-to-end with a Hermes coding agent.

Remaining advanced work includes:

- MTP, Medusa/tree, EAGLE, and other DESIGN §27 speculative proposers.
- vLLM `speculators` discovery and compatibility from DESIGN §28.
- Stochastic/rejection-sampling speculative acceptance; current verification
  uses greedy target agreement.
- True paged-attention execution kernels; current paging manages KV storage.
- Automatic hardware-profile probing/matching beyond explicit EP selection and
  metadata capability validation.

See [docs/DESIGN.md](docs/DESIGN.md) for the design and roadmap.

## License

MIT
