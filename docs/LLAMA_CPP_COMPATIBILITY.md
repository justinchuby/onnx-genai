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
| POST | `/tokenize` | `content`, independent `add_special` / `parse_special`, `with_pieces`, optional `model`; package vocabulary/tokenizer |
| POST | `/detokenize` | `tokens`, optional `model`; preserves special tokens, rejects absent vocabulary IDs |
| POST | `/apply-template` | Chat request fields; same prompt renderer as chat inference |
| POST | `/v1/chat/completions`, `/chat/completions` | Buffered/streaming text, tools, reasoning, declared image/audio inputs |
| POST | `/v1/completions` | OpenAI text/token/mixed prompt batches, shared controls, declared text FIM and optional stream usage |
| POST | `/v1/embeddings` | Existing OpenAI embedding contract |
| POST | `/completion`, `/completions`, `/infill` | Native buffered/streaming completion, bounded text/token/mixed batches, declared FIM metadata |
| POST | `/embedding`, `/embeddings` | Native pooled embeddings with requested normalization |
| POST | `/rerank`, `/reranking`, `/v1/rerank`, `/v1/reranking` | Declared token-pair scoring workflow with Jina/TEI response shapes |
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

- `/tokenize` can parse registered special tokens or treat their spelling as
  ordinary text. This is independent of inserting BOS/EOS tokens, and each
  request keeps its policy isolated from other requests.
- `with_pieces` returns Unicode strings when valid, or byte arrays for incomplete
  UTF-8 token fragments. Byte-level/fallback pieces preserve raw bytes, including
  `0xff`, and omit the grammar adapter's internal special-token framing.
- `model` can be omitted for the default chat model.
- `max_tokens: -1` uses the configured server output cap; context limits still apply.
- `seed: -1` selects an unseeded request; unsigned 64-bit seeds remain accepted.
- `repeat_penalty` and `typ_p` map to the existing repetition/typical controls.
- `logit_bias` accepts OpenAI token-ID objects and native `[token ID or text, bias]`
  pairs. `false` bans a token; finite numbers add a bias before penalties and
  constraints, including under greedy decoding. Unknown IDs and duplicate token
  assignments are rejected. The processor disables incompatible device shortcuts.
- `repeat_last_n` defaults to 64 and selects one prompt-plus-generated history
  window for repetition, frequency and presence penalties. `-1` uses full history;
  `0` disables all three. Counts include prompt tokens from the first generated
  token. Repetition rescales a seen logit once, then frequency and presence are
  subtracted together, following the pinned upstream penalties stage.
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

Legacy `/v1/completions` uses the same generation-control parser, validation,
token biases, model sampling defaults and context admission as chat/native
completion. Its `model` is optional and responses report the resolved model ID.
It retains integer `logprobs` (0..5), accepts `echo=false`, and emits true usage
before `[DONE]` when `stream_options.include_usage=true`. Text output omits stop
strings in both buffered and streaming modes. Unknown controls are rejected;
chat-template fields and tools require the chat endpoint. The default output
budget follows the configured cap and remaining context; `max_tokens=-1` leaves
that limit to the server. Both this endpoint and native `/completion` accept
text, explicit token IDs, mixed rows and bounded prompt batches. The same outer
array classification and token insertion policy applies to both protocols.
OpenAI batches return one response with indexed `choices` and summed prompt/output
usage. Streams use one completion ID, one finish frame per choice, optional
aggregate usage and one final `[DONE]`. For multiple prompts, nxrt workflow
publication extras are wrapped as `{prompt_index, publication}` in buffered
`workflow_outputs` and `workflow_output` SSE records; the inner runtime record
keeps its identity and payload. Single-prompt publication records are unchanged.
Prompts execute sequentially through
ordinary admission; errors or disconnects prevent later submissions. All inputs
are prepared before generation. `X-Session-Id` requires a single prompt, and
`suffix` requires text with declared FIM metadata. The 128-prompt default and
`--max-completion-prompts` limit apply to both endpoints. Probability streams
use the common incremental text decoder for their content as well.

Runtime-owned token policies accept an ordered `samplers` array of at most 64
stages: `penalties`, `dry`, `top_n_sigma`, `top_k`, `typ_p`, `top_p`, `min_p`,
`top_a`, `xtc`, `temperature`, `infill` and `adaptive_p`. Case, kebab/compact spellings and upstream
`nucleus`/`temp`/`typ` aliases are accepted. An empty list applies no optional
stages; omitted stages stay omitted and repeated stages run again. Bias, grammar
and stop contracts always remain active. The default ordering follows the pinned
[upstream sampler pipeline](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/common/common.h),
with the optional nxrt `top_a` extension after min-p. Mirostat mode replaces the
ordered warper list with temperature and Mirostat. `top_n_sigma` uses population
standard deviation of finite logits; non-positive values disable it. Top-k caps
its ranked candidate set at k even when the boundary logits tie. Equal logits
use ascending token IDs consistently in the host, fused and device paths; the
upstream implementation does not specify a stable tie order.

A zero-temperature stage creates a point mass where it occurs; preceding stages
still run. Omitting it does not force greedy selection merely because the
request's temperature value is zero. `dynatemp_range` and `dynatemp_exponent`
control entropy-based temperature at that same position. Candidate truncation
updates its entropy denominator; mask-only stages retain their candidate count.
The device sampler runs only when cached typed transforms prove the actual
parameters and stage order equivalent to its fixed pipeline. Temperature stays
at its requested position: finite-precision scaling can merge distinct logits,
so it cannot move across top-k. The matching fixed-order device path remains
available; other orders execute on host logits.

`/props.nxrt_capabilities.custom_sampler_order` reports the actual executor
capability. Authored sampler graphs without runtime-owned sampling reject custom
orders, active top-n-sigma and dynamic temperature before execution, with typed
400 errors; their properties do not advertise an unusable default stage list.
The explicit `adaptive_p` selector runs once after the other requested stages,
regardless of its listed position or repetitions. `adaptive_target` selects a
probability in [0, 1]; negative values disable adaptation, and it is inactive
without the explicit selector. `adaptive_decay` defaults to 0.9 and accepts
[0, 0.99]. The quadratic/linear probability transform and weighted EMA use the
original probability of each selected token, following the pinned
[upstream adaptive-p implementation](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/src/llama-sampler.cpp).
Request-local checkpoints restore the EMA when speculative choices are discarded. Stateful feedback stages run
on host and cannot be skipped by a device argmax shortcut. Mirostat takes
precedence over the ordered selector list, following upstream.

`infill` follows the upstream ending-mass test, rank-ordered byte-prefix merges
and probability thresholds. Logical candidate identities survive preceding
truncations; mask-only stages retain their slots, including masked ending tokens.
The preferred forced ending comes from the package's optional
`tokenizer.special_tokens.eot_token_id` (schema v1.9+), which must also belong to its EOS set;
otherwise the effective primary EOS is used. Request EOS overrides remain
binding. A forced ending may replace an optional filter's exclusion, but an
input model mask, mandatory token ban or grammar exclusion produces an actionable
error. Ending roles do not imply special-token types: only tokenizer-declared
special pieces are suppressed during prefix matching.
The selector requires lossless tokenizer byte pieces and declared ending roles.

Explicit content continuation accepts `continue_final_message=true` or `"content"` with
`add_generation_prompt=false` and a nonempty final assistant text. The template
boundary is proven by a second render rather than guessed from an EOS spelling
or an earlier matching quote; original prefix whitespace is preserved. Template
rendering, token counting and generation share this prompt. Tool-call messages,
reasoning-only continuation and replay into a persistent inference session are
still rejected with actionable errors. Thinking budgets and other unsupported
active controls also remain explicit request errors.
`response_format: {"type":"json_object"}` continuation validates the original
assistant prefix as partial JSON and constrains prefix plus new suffix at each
step. EOS, stop sequences and output-budget completion use that same complete
value; only newly generated suffix bytes and tokens appear in the response.
Invalid JSON prefixes are rejected before inference admission.
A grammar that leaves no finite model candidate fails before token selection.
JSON generation failures propagate as buffered errors or SSE error events; they
are never replaced with an ungenerated `{}` and a successful stop. JSON-object
budget/context exhaustion before completion still needs the upstream partial-
output/length behavior; that remaining compatibility item is not considered done.
Continuation intent follows the pinned upstream wire forms: `false`/`null` disable
explicit continuation, `true` selects content unless the final message contains
only reasoning, and `"content"` / `"reasoning_content"` select their named fields.
Reasoning continuation is recognized but still refused until its parser state can
be resumed. Unknown values fail with the accepted forms. HTTP rendering/counting/
generation and the public Rust request builders share continuation validation.
JSON Schema continuation seeds llguidance from lossless vocabulary bytes and the
existing assistant prefix. Grammar-only prefix chunks never become model
candidates or response tokens. Optional leading JSON whitespace remains in the
prompt and is skipped only when entering the schema's value recognizer; bytes
inside the value, including UTF-8 fragments across chunks, remain exact. Invalid
schema prefixes return a typed 400 error. Tokenizers without an interpretable
lossless byte codec fail explicitly rather than using lossy token decoding.
Optional progress/control subscription fields from Llama Chat are accepted, but
`/props.nxrt_capabilities` advertises resumable streams and reports realtime
reasoning control as unavailable. Per-token timing/progress events are not promised.
An SSE request with `X-Conversation-Id` retains generation independently of its
HTTP reader. Reconnect with `/v1/stream?conv_id=ID&from=BYTE_OFFSET`; DELETE cancels
the producer. Retention defaults to 16 streams and 4 MiB per stream, with completed
history expiring after five minutes. Old offsets fail explicitly once evicted.
Requests without a conversation ID close their driver reader on disconnect.
The driver carries that cancellation authority into generic workflows and fused
ORT/native decode. Execution observes cancellation between workflow nodes and
after decoder forwards, aborts uncommitted state and releases scheduler capacity.
An in-flight backend call completes before its next checkpoint. Cancellation
races semantic commit atomically; after commit it affects delivery rather than
rolling back durable state. Replay DELETE removes the producer and retained bytes.
The real Chrome smoke verifies the frontend Stop request, worker release and
cancelled-history removal, alongside active refresh and reader recovery.
Host-owned dense/recurrent/compressed state and in-place CPU KV restore complete state when a named conversation
replaces its cached prefix. Complete snapshots are budgeted before mutation.
Backends with device-resident state that lack this checkpoint return
409 for destructive conversation replacement. Appends retain their prefix-safe
path, and stateless requests may discard their internal cache. Complete snapshot
support for those remaining backends is still an implementation item.
Complete compressed-state checkpoints restore saved record cursors and partial
carries; they do not imply arbitrary token-prefix rewind or permission to ignore
the package's prefix-reuse policy.

Native generation accepts text, token IDs, mixed text/token rows and prompt
batches, including text-only `prompt_string` objects. Only a first text segment
receives inserted special tokens; explicit IDs are vocabulary-checked. Batches
are fully prepared before inference, execute sequentially through ordinary driver
admission, and retain input indices in buffered results and SSE. A single result
is an object; multiple buffered results form an array. The default limit is 128
prompts (`--max-completion-prompts`). Disconnects and stream failures prevent
submission of later prompts. Native streams decode incrementally and verify the
published prefix against the final model text.
Token event text comes from one engine incremental decoder across ordinary,
scheduled and speculative generation. Text-based stop checks see decoded
sequence text, including tokenizer-inserted separators, and can actually stop
before the output budget. Committed callback batches verify their full decoded
text before publishing; a decoder that revises a prior prefix defers that batch's
text to its last token. Live callbacks cannot retract published bytes and report
an explicit error if a decoder changes them.
`n_predict=0` (or native `max_tokens=0`) evaluates the prompt without producing
an output token. It uses ordinary driver capacity, prompt-only scheduler budget
and atomic cancellation. Buffered and SSE results contain empty content, zero
predicted tokens and the actual evaluated prompt count, including for batches
and prompts filling the complete context. Declared standalone runtime-owned
decoders support this operation; composite/authored workflows without an explicit
prefill contract return an actionable 400 before execution. Portable ORT runner
KV snapshots retain actual bytes under a host-memory lease and can warm later
requests; `tokens_cached` counts materialized reuse, never a token-only match.
`cache_prompt=false` forces complete prompt evaluation. Other ORT runner modes can still evaluate but report zero cross-request
reuse. Native evaluation retains its ordinary implementation cache.
Native multimodal prompt objects and infill extra files still require further
engine support. Native embedding normalization supports -1, 0 and positive integer p-norms; zero vectors
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

Explicit API failures carry numeric HTTP `code`, `message` and protocol `type`.
Invalid JSON/parameters use `invalid_request_error`; absent models/routes use
`not_found_error`, and permission refusals use `permission_error`. Internal faults
retain `server_error`. Messages routes wrap handler and extractor failures in
Anthropic's `type: error` envelope and preserve retry headers. Stream adapters
preserve the source error's status and protocol classification after headers
have been sent.

Native `/rerank`, `/reranking` and their `/v1` aliases execute a declared
token-pair scoring workflow. Jina `documents` and TEI `texts` response shapes,
stable score ordering, `top_n`, token usage and bounded input admission are
supported. Packages must declare the [reranking v1 contract](RERANKING.md);
ordinary embedding or language models are not advertised as rerankers.

## Remaining differences

This is a functional Llama Chat integration, not a claim of complete llama-server
parity. These upstream protocols are still unimplemented:

- Multimodal reranking, realtime audio/WebSocket inference and `/v1/systemone`.
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
The script uses loopback ports 18080–18082. It first verifies chat, persistence
and byte-exact completed replay. It then builds a temporary long-context version
of the same fixture by repeating its declared position table and updating the
matching shape/context metadata, with explicit size bounds. During a real
16,384-token ONNX generation, a fault proxy disconnects the first browser reader;
the frontend must automatically resume from a nonzero byte offset. The page
then reloads while the producer is still active and must attach again without
resubmitting generation. The final UI text must match the complete retained
answer, with one generation POST and no browser runtime errors. No mocked
inference or artificial network delay makes the generation appear active.
A fresh page then starts a long request, verifies a worker is active, clicks
Stop, checks the explicit DELETE response, waits for worker release and confirms
the cancelled stream is no longer replayable.
Temporary fixtures, proxy, Chrome and server processes are cleaned up, with
bounded startup/observer waits. Append an optional screenshot path after the
Playwright module path to save an image of the baseline chat. Coordinate this
inference workload with other sessions using the shared host lock.
