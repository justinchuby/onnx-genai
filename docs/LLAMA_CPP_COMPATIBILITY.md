# llama.cpp protocol compatibility

The server embeds the official Llama Chat frontend with a small capability adapter from
[llama.cpp commit 8d60873dfb8619f2b5ed31adfbab07cccb40db5c](https://github.com/ggml-org/llama.cpp/tree/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/ui/apps/chat).
The integration patch gates automatically attached browser tools using the
model's declared `nxrt_capabilities.tools`; a package with no tool protocol can
chat without receiving an unsupported tool request. Ordinary llama-server
connections, which do not send this extension, keep upstream behavior. The
patch and its hash are checked in alongside asset provenance.

Start `onnx-genai serve MODEL` and open `http://127.0.0.1:8080/`.
Node/npm and network access are not needed to build the Rust server or use its UI.
The removed `run` REPL has no alias or migration shim. `generate`, `transcribe`,
model inspection, profiling and command-line streaming remain available.

## Implemented surface

| Method | Paths | Contract |
| --- | --- | --- |
| GET | `/`, embedded asset paths | Official UI, MIME types, revalidation, immutable hashed assets, cross-origin isolation headers |
| GET | `/health`, `/v1/health` | Existing nxrt health response |
| GET | `/models`, `/v1/models` | Configured models, committed load status, input modalities and context length when known |
| GET | `/props` | Model generation defaults, chat template, declared input modalities; router role for multiple configured models |
| POST | `/tokenize` | `content`, `add_special`, `with_pieces`, optional `model`; package vocabulary/tokenizer |
| POST | `/detokenize` | `tokens`, optional `model`; preserves special tokens, rejects absent vocabulary IDs |
| POST | `/apply-template` | Chat request fields; same prompt renderer as chat inference |
| POST | `/v1/chat/completions`, `/chat/completions` | Buffered/streaming text, tools, reasoning, declared image/audio inputs |
| POST | `/v1/completions` | Existing OpenAI text/FIM completion contract |
| POST | `/v1/embeddings` | Existing OpenAI embedding contract |
| POST | `/completion`, `/completions`, `/infill` | Native buffered/streaming completion, raw text/token IDs, declared FIM metadata |
| POST | `/embedding`, `/embeddings` | Native pooled embeddings with requested normalization |
| POST | `/v1/responses`, `/responses` | Responses wire adapter over admitted chat generation |
| POST | `/v1/messages` | Anthropic Messages wire adapter over admitted chat generation |
| POST | `/v1/chat/completions/input_tokens`, `/chat/completions/input_tokens`, `/v1/responses/input_tokens`, `/responses/input_tokens`, `/v1/messages/count_tokens` | Input token counts from the shared inference preparation path |
| GET, DELETE | `/v1/stream` | Byte-offset replay or explicit producer cancellation by `conv_id` |
| POST | `/v1/streams/lookup` | Lookup only requested conversation IDs, including model-qualified IDs |
| POST | `/v1/audio/transcriptions`, `/audio/transcriptions` | Existing multipart transcription contract |
| POST | `/v1/audio/speech`, `/audio/speech`, `/tts` | Declared buffered speech workflow; native input/prompt and raw-prompt handling |
| GET | `/models/sse` | Initial state and changes to committed model load status; one-second polling, no invented progress |
| POST | `/models/load`, `/models/unload` | Configured model IDs; requires `--enable-admin-endpoints` |
| GET | `/metrics` | Existing nxrt Prometheus metrics when compiled with `metrics`; names are not llama metric aliases |

`/props?model=ID&autoload=false` never loads a model. The default is
`autoload=true`. The UI uses `autoload=false` for discovery. Unknown/unloaded
models are reported explicitly. Model load/unload requests complete when the
operation completes; clients can inspect the committed status afterward.
Models are configured with `--models-dir` or `--models-config`; this does not
implement llama.cpp's Hugging Face/GGUF download manager. Load `extra_args` are
rejected because these models execute in-process with explicit nxrt settings.

## Request compatibility

- `model` can be omitted for the default chat model.
- `max_tokens: -1` uses the configured server output cap; context limits still apply.
- `seed: -1` selects an unseeded request; unsigned 64-bit seeds remain accepted.
- `repeat_penalty` and `typ_p` map to the existing repetition/typical controls.
- `logit_bias` accepts OpenAI token-ID objects and native `[token ID or text, bias]`
  pairs. `false` bans a token; finite numbers add a bias before penalties and
  constraints, including under greedy decoding. Unknown IDs and duplicate token
  assignments are rejected. The processor disables incompatible device shortcuts.
- `repeat_last_n: -1` uses full history; non-negative values select the repetition window.
- `chat_template_kwargs` supplies template variables such as `enable_thinking`.
  Structural variables (`messages`, `tools`, generation prompt and BOS/EOS) cannot
  be overridden through kwargs. Unreferenced variables have the template's normal behavior.
- `add_generation_prompt` reaches the prompt renderer.
- Assistant `reasoning_content` is preserved when rendering subsequent turns.
- `stream_options.include_usage` adds an empty-choices usage event after the final
  finish chunk, before `[DONE]`, including the tool-calling stream path.
- DRY breaker strings are tokenized with the requested model and retain complete
  token sequences. Matching does not cross complete breakers; partial sequences
  remain ordinary history. `dry_penalty_last_n` bounds prompt/generated history,
  with -1 for full history and 0 to disable DRY.

The sampler pipeline is nxrt's existing fixed pipeline. Arbitrary `samplers`
orders, dynamic temperature, thinking budgets, continuation of a final assistant
message and other unsupported active controls are rejected with an
`invalid_request_error`. Neutral dynamic-temperature settings are accepted.
Optional progress/control subscription fields from Llama Chat are accepted, but
`/props.nxrt_capabilities` advertises resumable streams and reports realtime
reasoning control as unavailable. Per-token timing/progress events are not promised.
An SSE request with `X-Conversation-Id` retains generation independently of its
HTTP reader. Reconnect with `/v1/stream?conv_id=ID&from=BYTE_OFFSET`; DELETE cancels
the producer. Retention defaults to 16 streams and 4 MiB per stream, with completed
history expiring after five minutes. Old offsets fail explicitly once evicted.
Requests without a conversation ID retain disconnect-cancels behavior.

Native generation accepts one text prompt or one token-ID row. Prompt batches,
`n_predict=0`, and infill extra files require further engine support. Native
embedding normalization supports -1, 0 and positive integer p-norms; zero vectors
remain zero. Token-counting routes use the same prompt renderer and declared media expansion
as generation. They validate attachments and tool capability without executing a
generation. Anthropic streaming receives
the actual admitted prompt count (including media/session expansion) from the
shared chat transaction, carried in its first chunk as `nxrt_prompt_tokens`.

Responses and Messages share the chat tool/package capability checks. Stateful
Responses continuation/storage, hosted tools and explicit thinking budgets are
rejected. Streaming adapters preserve reasoning, tool argument deltas, usage and
termination, and report upstream stream failures as errors.

Native `/tts` accepts `input` or `prompt`, optional model, and the package's
bounded WAV workflow. `raw_prompt=true` bypasses text assembly and requires empty
`instructions`; token and output-unit limits still apply. Voice cloning,
undeclared voices/languages, and active synthesis controls are rejected rather
than ignored. The OpenAI speech request also rejects unknown controls.

## Remaining differences

This is a functional Llama Chat integration, not a claim of complete llama-server
parity. These upstream protocols are still unimplemented:

- Reranking, realtime audio/WebSocket inference and `/v1/systemone`.
- Slot save/restore, LoRA hot swapping and realtime reasoning control.
- GGUF/Hugging Face model downloads and deletion.

`/tools`, `/cors-proxy`, model download/delete and mutable `/props` return
`403 feature_disabled`; they never execute shell commands or proxy requests.
Other unknown paths return JSON 404 errors rather than the HTML app.
Existing nxrt sessions/admin/debug/image APIs remain available with their own
contracts. Capability flags are derived from admitted package metadata; ONNX
models are never advertised as GGUF models and unknown dimensions are not invented.

Serve the UI and API on the same origin (the embedded default, or a reverse
proxy). Cross-origin standalone hosting needs a proxy that explicitly configures
CORS. No permissive CORS policy is enabled implicitly by embedding the UI.

## Refreshing the upstream UI

Run `python3 scripts/update_llama_chat.py` with git, Node >=20 and npm installed.
The script fetches the exact commit, applies
`scripts/llama_chat/capabilities.patch`, builds with the upstream lockfile, copies
the static output and license notices, and records asset SHA-256 hashes. To update upstream, change the explicit commit pin and review
protocol changes before regenerating. Cargo embeds exact asset paths with
`build.rs`; it never runs npm or downloads UI code. Keep assets and notices in
the same change. The dependency notices conservatively include build-time packages.

Use the server's compatibility tests plus CLI generation/transcription tests and
an actual-browser chat smoke test after an update. Tiny fixtures verify protocol
behavior; they are not representative language models or performance benchmarks.

For a browser smoke test, build the server, install Playwright in a temporary
Node environment, then run `node scripts/test_llama_chat.mjs /absolute/path/to/playwright/index.mjs`.
The script starts a tiny-fixture server on port 18080, sends a message through
the actual UI, verifies the answer and persisted conversation, then shuts down
Chrome and the server. Append an optional screenshot path after the Playwright module path to save an image. Coordinate
this inference workload with other sessions using the shared host lock.
