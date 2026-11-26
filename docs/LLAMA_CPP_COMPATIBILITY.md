# llama.cpp protocol compatibility

The [generated endpoint registration matrix](LLAMA_CPP_ENDPOINT_MATRIX.md) inventories
all 55 static upstream method/path contracts, including aliases and conditional routes.
Its route-presence results are separate from the behavioral evidence below.

Native execution is enabled and selected by default in the engine, facade, CLI
and server. ORT is an explicitly selected reference backend for tests and
comparisons: use `--backend ort` or `EngineDecodeBackend::Ort`. Native failures
remain visible instead of automatically switching production requests to ORT.
The opt-in auto selector is also a reference/testing mode.

Native workflow tensor construction, host access and no-copy aliases do not load
ORT. A missing-library subprocess covers default Native CPU binary generation;
ORT sessions/bindings resolve tensor views explicitly. Metadata-declared MTP
heads also run on the selected Native executor without an ORT environment.
The Python launcher discovers optional reference/plugin paths without importing
their Python packages or preloading their libraries. The default wheel has no
ORT dependency; `ort` and `ort-cuda` extras supply explicit test/reference runs.

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

## API keys

Authentication is optional. Start with `nxrt serve MODEL --api-key-file keys.txt`;
the UTF-8 file contains one key per line, with blank lines and lines starting
with `#` ignored. `--api-key` accepts comma-separated CSV and may be repeated.
The upstream environment variables `LLAMA_API_KEY` and `LLAMA_ARG_API_KEY_FILE`
are also accepted. Keys from both sources are combined.

Send `Authorization: Bearer YOUR_KEY`, a raw `Authorization: YOUR_KEY`, or
`X-Api-Key: YOUR_KEY`. A nonempty Authorization header takes precedence over
X-Api-Key. Missing or incorrect credentials return HTTP 401 with
`error.type: authentication_error` and `error.code: 401`.
Health endpoints and exact embedded UI assets remain public. All other API
requests, including replay, management and realtime upgrade requests, pass
through key admission before routing. With no configured keys, requests keep
the existing open behavior. Server configuration Debug output exposes only
the configured key count; it does not contain credentials or their digests.

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
| POST | `/v1/systemone` | Declared non-generative decision workflows: choice, score and noul; see [contract and limits](SYSTEMONE.md) |
| POST | `/v1/chat/completions/control` | End the current reasoning block of an armed live chat/legacy completion |
| POST | `/v1/completions` | OpenAI text/token/mixed prompt batches, shared controls, declared text FIM and optional stream usage |
| POST | `/v1/embeddings` | Existing OpenAI embedding contract |
| POST | `/completion`, `/completions`, `/infill` | Native buffered/streaming completion, bounded text/token/mixed batches, declared FIM metadata |
| POST | `/embedding`, `/embeddings` | Native pooled embeddings with requested normalization |
| POST | `/rerank`, `/reranking`, `/v1/rerank`, `/v1/reranking` | Declared text/image-pair scoring workflow with Jina/TEI response shapes |
| POST | `/v1/responses`, `/responses` | Responses wire adapter over admitted chat generation |
| POST | `/v1/messages` | Anthropic Messages wire adapter over admitted chat generation |
| POST | `/v1/chat/completions/input_tokens`, `/chat/completions/input_tokens`, `/v1/responses/input_tokens`, `/responses/input_tokens`, `/v1/messages/count_tokens` | Input token counts from the shared inference preparation path |
| GET, DELETE | `/v1/stream` | Byte-offset replay or explicit producer cancellation by `conv_id` |
| POST | `/v1/streams/lookup` | Lookup only requested conversation IDs, including model-qualified IDs |
| POST | `/v1/audio/transcriptions`, `/audio/transcriptions` | Native frame-decoding and generative multipart transcription; typed JSON/SSE, budgets and usage; [contract](TRANSCRIPTION.md) |
| POST | `/v1/audio/speech`, `/audio/speech`, `/tts` | Declared buffered speech workflow; native input/prompt and raw-prompt handling |
| WS | `/v1/realtime` | Metadata-declared native streaming audio/text sessions; [contract and current limits](REALTIME.md) |
| GET | `/models/sse` | Model status, actual per-file download progress, pause/failure and removal events |
| POST | `/models/load`, `/models/unload` | Configured and cached model IDs; cache mode also supports pausing downloads |
| POST, DELETE | `/models` | Opt-in bounded Hugging Face nxrt-package download, resume and deletion |
| GET, POST | `/tools`, `/cors-proxy` | Outside scope: execution and MCP transport belong to an external harness |
| GET, POST | `/lora-adapters` | Model-bound declared catalog and atomic default scale replacement; request `lora` overrides execute actual adapter weights |
| GET | `/slots` | Actual serial Native slot status; model/autoload selection and fail-on-busy polling |
| POST | `/slots/{id_slot}?action=save\|restore\|erase` | Owner-thread state actions under opt-in bounded `--slot-save-path`; nxrt state file encoding |
| GET | `/metrics` | Existing nxrt Prometheus metrics when compiled with `metrics`; names are not llama metric aliases |

Serial Native completion publishes the owner slot snapshot and releases admission
capacity before its terminal event/reply. A following `/slots` read observes the
committed `n_past` and idle state without waiting for a worker-loop epilogue.

`/props?model=ID&autoload=false` never loads a model. The default is
`autoload=true`. The UI uses `autoload=false` for discovery. Unknown/unloaded
models are reported explicitly. Model load/unload requests complete when the
operation completes; clients can inspect the committed status afterward.
Models are configured with `--models-dir` or `--models-config`. Opt-in
`--model-cache-dir` also admits managed Hugging Face nxrt packages, with durable
pause/resume and deletion; see [model management](MODEL_MANAGEMENT.md).
Load/unload requires this cache option or `--enable-admin-endpoints`.
Load `extra_args` are rejected because these models execute in-process with
explicit nxrt settings.

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

Reasoning budgets use `reasoning_budget_tokens` or its upstream
`thinking_budget_tokens` spelling; the former takes precedence if both are sent.
`-1` leaves the counter unlimited, and `0` forces a close as soon as a declared
reasoning block is active. Chat derives its opener/closers and final assistant
prefill from `response_template`. Implicit reasoning channels may end at a
content opener. Patterned boundaries require an explicit matching delimiter;
forcing requires a lossless tokenizer round-trip. Native/legacy callers may
supply `reasoning_budget_start_tag`, `reasoning_budget_end_tags` (or singular
`reasoning_budget_end_tag`) and `generation_prompt` directly. The first closer
is forced after optional `reasoning_budget_message` text. Existing assistant
reasoning prefill counts toward the budget; old conversation turns do not.
Each new reasoning block receives a fresh budget. Pending Unicode completes
before forcing, and global output/context limits can still truncate the close.
A closer banned by `logit_bias` is rejected before generation or SSE admission.

`generation_prompt` also seeds JSON, JSON Schema, Regex and Lark grammars without
requiring an enabled reasoning budget. It is not echoed in newly generated text.
Anthropic `thinking.type=enabled` maps `budget_tokens` through the same counter,
with the upstream default of 10,000 when omitted. Optional control hints on a
model with no usable reasoning protocol leave its ordinary chat behavior intact.
`POST /v1/chat/completions/control` accepts `{"id":"<completion-id>","action":"reasoning_end"}`
for live chat and legacy completion IDs. Requests must opt in with
`reasoning_control=true` and resolve a usable reasoning protocol to be armed.
Unknown/finished IDs and unarmed completions return `success:false`; missing IDs
and unsupported actions return HTTP 400. A successful signal finishes the current
reasoning block using the same Unicode-safe forced close. `nxrt_capabilities.reasoning_control`
reports endpoint availability, not a claim that every model has a reasoning channel.
Registrations are bounded by `max_queue_depth` across the server and released
when their generation consumer ends; replay readers share their producer's
registration. Native completions retain budget support but do not publish a
completion ID for manual control.
In router mode, control requests require a configured `model` id and can only
signal completions owned by that model. Unknown models return 404; a known model
with no matching active completion returns `success:false` without loading it.
Single-model mode ignores the optional model name, matching llama-server.

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

Chat-template assets accept a single source, a `default`/`tool_use` name-to-source
object, or a list of `{name, template}` entries. Independent `chat_template.jinja`
and `chat_templates/*.jinja` files take precedence over config entries. Nonempty
offered tools select `tool_use` when available; otherwise rendering uses the base
source. A tool-only declaration also supplies the base, matching upstream.
Duplicate names, invalid source types and nonempty named sets without a recognized
base report actionable asset errors. Empty declarations retain the documented
no-template fallback. `/props` reports an explicit tool source as
`chat_template_tool_use`; that source does not create tool execution capability
without a registered model protocol.

Explicit continuation accepts `true`, `"content"` or `"reasoning_content"` and
suppresses `add_generation_prompt` even when that flag is true/default, matching
upstream. The selected final assistant text must be nonempty. A template probe
proves its boundary rather than guessing an EOS spelling or an earlier quote;
original prefix whitespace is preserved. Rendering, token counting and generation
share this prompt.

Reasoning continuation requires a declared response template and a chat template
preserving `reasoning_content`. The parser must remain inside that declared
channel after the actual prompt is truncated. Existing reasoning is consumed as
prefill state and excluded from buffered/SSE/logprob output; newly generated
reasoning and answer text retain separate fields. Content continuation uses the
same channel-state check when a response template is declared. Plain Rust request
builders reject reasoning continuation because they have no model parser contract.
Tool-call messages and replay into a persistent inference session remain rejected.
JSON response formats and internal Regex/Lark policies use a channel-aware generation constraint when a response
template is declared. The actual rendered prefill seeds the parser, and model
token candidates constrain only projected content while reasoning remains free.
This applies to implicit prefill and explicit reasoning continuation, with the
same projection for buffered, SSE and logprob output. Capabilities are checked
before generation/SSE admission. Supported combinations currently use literal
boundaries or regex boundaries with a resolved byte DFA, plus
ByteLevel/ByteFallback or canonical non-byte decoders. Byte decoder sequences
also support token-local Replace/Strip before one byte stage, Fuse, and finite
initial/trailing Strip after fusion. These capabilities follow decoder metadata and preserve
generated-stream boundaries, including split Unicode and special-token policy. Pending Unicode bytes,
regex priority, greedy/lazy continuations and EOF assertions share the publication
parser's boundary rules. Regexes requiring unsupported DFA look assertions
and other unresolved contextual byte-decoder stages remain open compatibility
work; errors identify the unsupported decoder path. Trailing stripping retains
bounded UTF-8 suffixes and checks actual reachable endings before admitting a
candidate; a withheld tail cannot hide a schema violation inside an incomplete
value. Combined published/retained bytes share the request response budget, and
composed Strip retention is capped at 4 MiB. Ambiguous suffix viability has an
explicit 4096-state, bounded retained-state analysis budget and fails actionably
when exceeded. Reasoning token-ID boundaries must be stable after their final
token; markers delayed by trailing stripping are rejected before execution.
Projected JSON accepts split Unicode escapes and surrogate pairs, including
string constants, known property names, string patterns and scalar length limits.
An unfinished escape must have a schema-valid scalar completion and never permits
EOF. Original prompt and output spelling is retained. Declared property names,
pattern-property exclusions and additional-property exclusions use the same JSON
quoting lexemes, including DEL and other control escapes. Escaped spelling cannot
change which value schema owns a known key. Raw DEL remains valid JSON string
data and is normalized only for recognition, including free property names and
ordinary values; raw RFC 8259 control bytes remain invalid. Ordinary JSON output without a response template uses the same
byte-oriented schema recognition, including split escapes and surrogate pairs.
Forced tool Lark policies use the same declared projection. Reasoning can precede
the tool envelope, and tool-like examples inside reasoning are excluded from
transactional tool parsing. Auto tool requests share this projection even without
a grammar. Buffered and streamed tool turns retain their reasoning_content.
Projected literal grammars preserve authored backslashes and Unicode spellings;
only JSON Schema policy applies JSON escape normalization. Model-token references,
token-count directives, per-rule temperatures and hidden stops cannot be treated
as bytes and are rejected from projected byte grammars with an actionable error.
Unprojected model-token Lark retains its existing token-aware behavior. User
GBNF uses the separate raw-output path described below.

Tool observation preserves strict byte fragments for resolved byte decoders, so
split Unicode channel markers are never replaced by provisional glyphs in tool
syntax. Unsupported contextual byte decoders defer tool parsing until terminal
canonical decoding; the retained decoder reason is visible in observer Debug
state. Rewind/abort restores the raw input and projected tool-parser history.

Ordinary JSON decoding follows the tokenizer's normal special-token policy;
protocol-channel projection preserves markers for its declared parser. An EOS
role does not make an ordinary token special: its actual decoded bytes must also
finish the JSON value. Canonical byte-concatenating vocabularies use a trie mask
against the same compiled schema. Escape candidates, contextual non-byte decoding
and pending UTF-16 spelling use semantic candidate inspection. Model-token masks
and recognizer-byte masks have separate identities and cannot share cached bits.

`GenerateOptions.max_constraint_bytes` bounds both raw response text and its canonical
JSON prefix, with a 4 MiB Rust default. HTTP preparation uses the configured
`--replay-buffer-bytes` limit for all JSON output modes. Exceeding the candidate
budget rejects that candidate; backend errors preserve their cause. Constraints
are validated during chat/native/legacy completion admission before SSE headers.
The old public character-based `JsonConstraint` was removed; pure request syntax
preflight uses `validate_json_prefix`, while generation uses the shared schema
recognizer.

JSON object defaults follow the pinned [chat conversion](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/tools/server/server-common.cpp#L1420)
and [native schema handler](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/tools/server/server-schema.cpp#L270).
`json_schema: {}` and implicit/empty response-format schemas constrain the root
to an object. Explicit nonempty schemas and boolean schemas keep their JSON
Schema meaning. This applies to eager and lazy generation. The engine's generic
`GenerateConstraint::Json` remains an any-value constraint; the transport selects
the appropriate schema explicitly.

Accepted forms include top-level `json_schema`, `json_object.schema`, flat
`json_schema.schema`, and the OpenAI nested `json_schema.json_schema.schema`
wrapper. Nested wrappers take precedence over a flat schema. An implicit
`json_object` keeps a supplied top-level schema; its explicit `schema` overrides
that field. `text` preserves an explicitly supplied top-level schema. In native
requests, presence of `grammar` takes precedence over legacy `json_schema`, even
for an empty grammar; chat rejects conflicting grammar/schema constraints.
Responses `text.format` uses the same schema selection after wire conversion.
For example, an explicit schema can allow a scalar despite the format label:

```json
{"response_format":{"type":"json_object","schema":{"const":7}}}
```

Without explicit continuation, declared content/reasoning text in the template
prefill is included in the response, matching upstream session projection.
The same include/exclude policy governs buffered, SSE and logprob replay; only
the final assistant region after the declared start anchor is considered.
`response_format: {"type":"json_object"}` continuation validates the original
assistant prefix as partial JSON and constrains prefix plus new suffix at each
step. Grammar validity and EOS eligibility use prefix plus suffix; only newly
generated suffix bytes and tokens appear in the response. A user stop or token/
context limit may end an incomplete value: the response retains actual partial
output and reports `stop` or `length` rather than inventing a completed object.
Invalid JSON prefixes are rejected before inference admission.
A grammar that leaves no finite model candidate fails before token selection.
JSON generation failures propagate as buffered errors or SSE error events; they
are never replaced with an ungenerated `{}` and a successful stop. Stopping
strings are removed before declared response-channel projection, consistently
for buffered, live and logprob SSE output, including reasoning channels.
Text stops match the earliest occurrence in generated output, including inside
one token's decoded fragment and across token boundaries. Configured order
resolves equal-position matches. The selected token remains counted and
committed, while the stop string and following text are hidden. Native
`stopping_word`, finish reasons and usage reflect that actual decoder stop.
The matched word remains reported when EOS or prediction time takes priority
over the final stop type, matching the pinned server's diagnostic behavior.
Token-ID stop sequences retain their token-suffix semantics. Prediction-time
checks use the same visible text boundary as delivery, including held partial
stop prefixes, so hidden newlines do not trigger time expiry.
Continuation intent follows the pinned upstream wire forms: `false`/`null` disable
explicit continuation, `true` selects content unless the final message contains
only reasoning, and `"content"` / `"reasoning_content"` select their named fields.
Unknown values fail with the accepted forms. HTTP rendering/counting/generation
and the public Rust request builders share request-shape validation; model-aware
preparation additionally proves the selected parser channel.
JSON Schema continuation seeds llguidance from lossless vocabulary bytes and the
existing assistant prefix. Grammar-only prefix chunks never become model
candidates or response tokens. Optional leading JSON whitespace remains in the
prompt and is skipped only when entering the schema's value recognizer. Completed
Unicode escapes, surrogate pairs and escaped soliduses in existing JSON strings
are canonicalized only for schema recognition; prompt bytes retain the original
spelling. Literal escaped backslashes keep their meaning. Grammar-only byte chunks
may still split UTF-8 scalars. Invalid schema prefixes return a typed 400 error. Tokenizers without an interpretable
lossless byte codec fail explicitly rather than using lossy token decoding.
`t_max_predict_ms` limits prediction wall time starting with the first generated
token, excluding prefill. Like the pinned upstream server, it stops only when a
newly emitted text fragment contains a newline and elapsed time is strictly
greater than the limit. A newline removed as part of a matched stop word does
not trigger this limit. This is a normal generation stop: accepted output and
its session state commit, with OpenAI `finish_reason: "length"` and native
`stop_type: "limit"`; it does not cancel or roll back the request. Scheduler gaps
count toward this wall limit, unlike the active-work timing counters below.

Omission, `null`, `-1` and `0` disable the limit. Numeric fractions truncate
toward zero before checking the signed 64-bit millisecond range; for example,
`1.5` becomes one millisecond. Unsafe overflow representations are rejected.
Native, OpenAI completion/chat, Responses and Anthropic generation share this
request field. Native speculative, candidate-tree and DFlash execution stop at
the accepted token prefix and finalize physical state before publication.
Generic authored workflows observe emitted tokens and stop before entering the
next loop body. The current body finishes atomically, including its declared
state updates. The request token budget bounds all emitted output even when an
authored loop declares a larger iteration count; a body that would exceed that
budget fails and rolls back the turn before publication.

Text and token-ID stop sequences also apply to generic authored workflows,
candidate-tree and DFlash execution. Specialized drivers truncate the accepted
prefix before repairing state. Text stops and prediction-time limits require a
package tokenizer; token-ID stops, EOS and token budgets do not. Empty stop
sequences are inert. Pending decoder bytes are finalized before semantic commit,
so an EOF text stop receives its actual stop reason, and a decoding or byte-bound
failure leaves session state and callbacks uncommitted.

`n_indent` requests a minimum number of space or tab characters after generated
newlines. Each tab counts once. Omission, `null` and zero disable the control;
numeric fractions truncate toward zero and the resulting value must fit
0 through INT32_MAX. It requires a package tokenizer. The request is shared by
native and OpenAI completion/chat, Responses and Anthropic generation.

Indentation checks follow the pinned server's token-observation order: the first
visible newline arms a later observation, and subsequent observations advance
the newline scan. An insufficiently indented line stops generation normally;
its leading whitespace remains in final text while its body is removed. EOS and
prediction-time labels can take precedence without undoing that text clipping.
The triggering token remains committed and counted. Native reports `limit`;
OpenAI reports `length`. Generic workflows finish the current loop body
atomically; specialized execution repairs the accepted prefix before commit.

SSE publishes text whose indentation is already safe, withholding potentially
retractable line content until the final boundary is known. A request that ends
before the delayed indentation check releases any retained valid suffix. This
avoids the pinned implementation's send-before-truncate ordering producing a
streamed body that disagrees with final text. Token IDs and usage still account
for every committed token.

Ordinary runtime-owned decoder generation returns llama.cpp `timings` in native,
OpenAI completion and chat buffered results and final SSE chunks. Setting
`timings_per_token=true` adds cumulative token-boundary measurements to SSE,
including through the chat reasoning/tool-channel projection. Native/ORT clocks
measure actual forward and selection work before commit-only publication;
network delivery, callback delays and scheduler suspension are excluded. The
first generated token comes from prompt logits, so predicted throughput uses
`predicted_n - 1` generation steps. Prompt timing counts fresh tokens and reports
input reuse separately as `cache_n`; it is unrelated to retained `tokens_cached`.
Standalone prompt-only evaluation (`n_predict=0`) reports the same measurements
with zero predicted tokens and generation duration. A Native full-cache hit
executes no forward and reports zero prompt work/time; the ORT test reference
retains its real final-token reevaluation. Executors without this timing contract
(including ORT speculative and composite paths) omit the measurements. Native
speculative prompt-lookup/MTP loops report actual prompt, proposal, verification
and accepted-path work with cumulative accepted-token counts; rejected draft
work remains included in the generation cost. Multi-token target verification
requires one logits row per draft position and a consistent nonempty vocabulary.
A mismatch returns an actionable error and rolls back provisional state; a
last-token-only target can use `num_speculative_tokens=1`.
When Native speculative verification has proposed tokens, these timing snapshots
also contain `draft_n` and `draft_n_accepted`: cumulative proposed and accepted
draft tokens, excluding the target's bonus token. Acceptance describes the
verified block even when an output stop later truncates publication. These
fields are absent before an actual proposal and on ordinary decode; no zero
measurements are invented for executors without the timing contract. Buffered
completion and SSE project the same engine-owned counter snapshots.

For ordinary decoder SSE on native/OpenAI completions, chat and infill,
`return_progress=true` publishes `prompt_progress` with `total`, `cache`,
`processed` and elapsed `time_ms`. As upstream, `processed` includes cached tokens.
Events describe actual execution boundaries: a whole-prompt forward reports its
start and completion; constructing a semantic-prefix snapshot also reports that
real intermediate boundary. Freshly computed prefix bytes count as processed
work, never input reuse; only previously retained prefix state counts as `cache`.
Progress is provisional execution telemetry and may precede a later error or
cancellation; it does not expose tokens or promise a committed turn. Observer
failure follows ordinary atomic rollback. Native prompt-only SSE also reports
start/completion boundaries, including cache-only hits, and delivers progress
while evaluation is running. Closing its receiver cancels provisional work;
its final result follows state/slot and capacity publication. Native speculative
execution reports the actual initial target-prompt forward and checks cancellation
at proposal/verification boundaries; it currently evaluates a cold prompt and
reports zero input reuse. ORT speculative/composite execution and continuous
batching progress remain open contracts.
`/props.nxrt_capabilities` advertises resumable streams and the live reasoning
control endpoint.
An SSE request with `X-Conversation-Id` retains generation independently of its
HTTP reader. Reconnect with `/v1/stream?conv_id=ID&from=BYTE_OFFSET`; DELETE cancels
the producer. Retention defaults to 16 streams and 4 MiB per stream, with completed
history expiring after five minutes. Old offsets fail explicitly once evicted.
Requests without a conversation ID close their driver reader on disconnect.
SSE generation streams use a 30-second idle comment ping interval by default,
matching the pinned server. `--sse-ping-interval SECONDS` sets the server default;
request `sse_ping_interval` overrides it. JSON numbers truncate toward zero to
integer seconds from -1 through 2147483647; for example, `1.9` selects one second
and `0.5` disables pings. Omission or `null` inherits the configured server
interval, including a disabled default. Values -1 and 0 disable pings. Unsafe
overflow representations, booleans and strings are rejected. The CLI accepts
integer seconds. Data resets the idle timer, and EOF
ends the timer with the source. This applies to native completion/infill,
OpenAI completion/chat, Responses and Anthropic streams. Pings are SSE comments,
not generation events. Resumable streams retain their actual comment bytes,
so reconnect offsets continue to refer to the exact transmitted byte sequence.

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

Native completion/infill final results include the actual detokenized prompt
(with special tokens), `has_new_line` for visible generated text, and the matched
text `stopping_word`. Prompt-only evaluation reports an empty stopping word and
no generated newline. A serial Native executor reports its real `id_slot: 0`.
`generation_settings` reports the admitted output cap and resolved sampling
configuration: sampler order, temperature and probability filters, penalties,
stop strings, explicit seed, EOS policy, preserved tokens, probability reporting
and cache policy. Active dynamic-temperature, DRY, XTC and Mirostat controls
include their resolved parameters. Randomized generation omits an unspecified
seed. Budgets reflect the admitted cap, including model context bounds and zero
for prompt-only evaluation. Batch results retain their own prompt and cap.

Native completion/infill `response_fields` selects fields from the final result.
An omitted or empty array returns the full result; otherwise missing paths are
omitted. Paths traverse JSON object keys separated by `/`; arrays are not indexed
and `~0`/`~1` are literal key text, not JSON Pointer escapes. The pinned
`json_get_nested_values` implementation retains the entire path as the returned
key, for example `"timings/prompt_n"`, rather than renaming it to its leaf as the
README suggests. This also applies to each batch result and prompt-only result.
SSE token/progress events and errors remain unchanged; only the terminal success
object is projected, matching the pinned native task serializer. Batch stream
completion is driven independently of whether `stop` or `index` was selected.
Selection never fabricates a requested field that is absent from the result.

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
requests; reuse counts require materialized state, never a token-only match.
`cache_prompt=false` forces complete prompt evaluation. Other ORT runner modes can still evaluate but report zero cross-request
reuse. Native prompt-only evaluation reuses the actual materialized prefix when
its metadata permits reuse and the physical owner matches. A complete hit
executes no forward; appends evaluate only the suffix. Sliceable KV can rewind a
diverged/shorter prefix; non-sliceable state recomputes instead of claiming a
history-only hit. `cache_prompt=false` forces full Native evaluation too.
Cancelled evaluations discard uncommitted stateless cache state and release
admission capacity. For prompt-only and ordinary generation responses, `tokens_cached` is the materialized
cache retained after the task, matching the pinned
[upstream final-response assignment](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/tools/server/server-context.cpp#L2185).
It is distinct from the engine's `prefix_cache_hit_len`, which measures this
request's input reuse. A cold evaluation can therefore retain a nonempty cache.
Ordinary Native generation also retains newly evaluated state after
`cache_prompt=false`; the option prevents input reuse, rather than discarding
that request's output cache. Retention is captured in the engine result at
completion, so a later request cannot change an earlier response's count.
The count comes from the materialized cursor, not prompt-plus-output token
arithmetic: the final sampled token may not yet have been consumed by a forward.
OpenAI cached-prompt usage continues to report input reuse separately.
Native multimodal prompt objects and infill extra files still require further
engine support. Native embedding normalization supports -1, 0 and positive integer p-norms; zero vectors
remain zero. Token-counting routes use the same prompt renderer and declared media expansion
as generation. They validate attachments and tool capability without executing a
generation. Anthropic streaming receives
the actual admitted prompt count (including media/session expansion) from the
shared chat transaction, carried in its first chunk as `nxrt_prompt_tokens`.

Responses and Messages share the chat tool/package capability checks. Stateful
Responses continuation/storage and hosted tools are rejected. Streaming adapters preserve reasoning, tool argument deltas, usage and
termination, and report upstream stream failures as errors.

Native `/tts` accepts `input` or `prompt`, optional model, and the package's
bounded WAV workflow. `raw_prompt=true` bypasses text assembly and requires empty
`instructions`; token and output-unit limits still apply. Voice cloning,
undeclared voices/languages, and active synthesis controls are rejected rather
than ignored. The OpenAI speech request also rejects unknown controls.

Buffered speech supports native `/tts` `response_format=wav` (default), `pcm`
(float32 little-endian), and `pcm16` (signed 16-bit little-endian). OpenAI
`/v1/audio/speech` and `/audio/speech` accept `wav` and map `pcm` to signed
16-bit samples. Raw PCM responses declare the package's output sample rate,
channel count (when greater than one), encoding and bit depth in `Content-Type`.
The declared output rate and channels are preserved; no implicit mono conversion
or 24 kHz override is applied. Float waveform outputs retain their precision
until transport encoding. An already encoded post-adapter PCM16 WAV can only
provide its existing quantized samples, scaled by 32768 for float transport.
PCM16 waveform encoding rounds `clamp(sample, -1, 1) * 32767`, including WAV;
nonfinite waveform samples fail with their sample index. Streaming speech
remains unsupported for these buffered workflow contracts.

For example, `curl http://localhost:8080/tts -H 'Content-Type: application/json'
-d '{"input":"hello","response_format":"pcm"}' --output speech.f32` returns
interleaved float32 samples at the rate and channels named in the response header.


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

Declared host LoRA serving supports global defaults, per-request overrides,
model isolation and replayed conversation history. See [LoRA serving](LORA_SERVING.md)
for package contracts, memory admission and remaining source/device limitations.

Reranking accepts ordered query/document images through an explicit v2 profile,
including actual native image preprocessing and model-owned token usage. See
[reranking](RERANKING.md) for the package ABI, request shapes and resource bounds.

## Remaining differences

This is a functional Llama Chat integration, not a claim of complete llama-server
parity. These upstream capabilities remain incomplete:

- Realtime grammar forcing beyond declared turn-based tagged-json mask workflows,
  partial function-call fragment projection, broader pressure/overlap acceptance and
  device-backed wire acceptance; see [realtime](REALTIME.md) for the implemented
  WebSocket session modes, resource bounds and Native CPU evidence.
- Video decision inputs, broader audio codecs and additional DSP operations/conventions,
  joint/packed decision heads and shared generation/decision entry points.
- Device-state/multi-sequence slot persistence; arbitrary GGUF/PEFT LoRA imports and device-fused LoRA.
- Bare GGUF graph import, filename-based GGUF quantization selection and selective sidecar management.
  Native nxrt packages can explicitly bind GGUF or SafeTensors weights to their
  ONNX graphs through schema `v1.17` `graph_weights`; decoder, workflow and MTP
  loading share the bindings. Exact tensor dtype/shape matching is required.
  See [inference metadata](genai/MODEL_METADATA.md#components-and-graph).
  Hugging Face nxrt-package download, resume and deletion are supported. Explicit
  `repository:variant` selection supports authored model-package variants,
  including selective installation and shared tokenizer assets; this does not
  infer an executable graph from a bare GGUF repository.

`/v1/systemone` executes text/structured-state decision profiles, with explicit
model-authored prompting, calibration outputs, variants and answer policies.
Core schema v1.11 declares input defaults, instruction/payload rules, option
ordering, score ranges and canonical legends. Decision v2 in core schema v1.14 additionally executes ordered state images,
including `files`/`images` and embedded chat image parts, with actual media-aware
usage. Decision v3 in core schema v1.15 adds ordered mixed image/audio state and
per-question spoken instructions using [supported encoded audio](AUDIO_PREPROCESSING.md#encoded-audio-formats). Video, unsupported audio codecs,
joint/packed heads and shared generation/decision
entry points remain open; see [SystemOne](SYSTEMONE.md). Route presence does not
claim every upstream decision-model format is admitted.

Model download/delete require `--model-cache-dir`; operator-configured paths
cannot be deleted. `--props` enables `POST /props`, matching
upstream's current acknowledgement-only contract: it returns `{"success":true}`
and does not read or change generation settings. `GET /props` advertises the
configured `endpoint_props` value in router and model modes. When disabled, POST
returns upstream-compatible `501 not_supported_error`. This is not an API
for changing generation defaults.
Tool execution belongs to an external harness. nxrt does not register `/tools`
or `/cors-proxy`, launch MCP children, execute filesystem/shell tools, or provide
Docker/Podman/SSH tool runtimes. These are deliberate scope exclusions, not
pending compatibility work. `/props` reports `server_tools: false` in
`nxrt_capabilities` and `cors_proxy_enabled: false` in both router and model modes.
The former tool/MCP/proxy CLI options and library configuration types are removed.

Chat tool calling remains supported: clients send `tools`/`tool_choice`, consume
assistant `tool_calls` (including streaming deltas), execute those calls in their
harness, and return `role: "tool"` messages with the matching `tool_call_id`.
Model metadata still determines which tool protocol is admitted. For example, a
harness can submit `{"role":"tool","tool_call_id":"call_1","content":"result"}`
as part of the next `/v1/chat/completions` request. The server never executes the
function named by a generated call.

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

For API-key onboarding, run
`node scripts/test_llama_chat_auth.mjs /absolute/path/to/playwright/index.mjs /absolute/path/to/onnx-genai-server`
under the same lock. This uses real Chrome and Native CPU inference to enter an
incorrect key, retry a correct key through the UI, chat, reload the persisted
login, and check private byte-exact replay. A fresh browser context must still
require a key. The script uses an ephemeral loopback port and removes its
synthetic key file, browser and server on exit.


## Raw GBNF and llguidance constraints

The chat, legacy completion and native completion endpoints accept the llama.cpp
`grammar` field as GBNF over newly generated raw output. A source beginning
with `%llguidance` instead uses llguidance Lark, preserving its header and
options. Existing assistant
prefill is not consumed, and declared reasoning/content projection does not
remove markers before this grammar sees them. This follows the pinned upstream
[sampling contract](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/common/sampling.cpp).

The supported syntax includes recursive rules, alternatives, groups, comments,
Unicode literals and ranges, negated classes, dot, optional/repeated items and
bounded repetition. Model token references use actual vocabulary IDs (`<[42]>`)
or text that tokenizes to exactly one model token (`<think>`); `!` negates a token
reference. Every legal token spelling remains available to the sampler, including
compound tokens and partial UTF-8 byte pieces. Ending roles are eligible only
when the root is complete, including endings covered by a negated token range.

The parser requires `root`, rejects undefined rules and left recursion, and
reports source line/column diagnostics. It bounds source bytes, rules, nesting,
expanded items and repetitions. As in the pinned upstream parser, minimum
repetitions above 2,000 are refused and larger upper bounds become unbounded.
The model must declare a dense, resolved, token-local byte vocabulary. Actual
raw pieces include special-token spellings; only explicitly declared ending IDs
receive terminal semantics. Token names and angle brackets do not guess roles.
Non-byte contextual decoders and stream-initial/trailing stripping currently return an
actionable capability error before SSE.

Both raw grammar forms conflict with JSON `response_format`, forced `tool_choice`, and
`ignore_eos=true`. Raw GBNF accepts `grammar_lazy=true` with nonempty
`grammar_triggers`. The native numeric trigger tags are 0 (token: requires
`token` and a string `value`), 1 (word), 2 (pattern), and 3 (full pattern).
Token triggers take precedence over patterns on each accepted token; patterns
are checked in configuration order. The trigger itself belongs to the grammar.
A pattern starts replay at its first nonempty capture, or the complete match
when all captures are empty. Replay retains the original model token identity
even when that byte offset falls inside a token. Full patterns match the entire
unconstrained buffer. Matching uses explicit C-locale byte semantics and bounded
backtracking; Unicode text matching and backend-only pattern extensions are not
silently substituted.

A word resolving to one model token must also occur in `preserved_tokens`;
multitoken words use literal byte matching. Before activation the grammar does
not constrain vocabulary candidates or EOS. After activation, ending roles
require a completed root. Reasoning budgets suppress the lazy grammar while the
thinking block is active and replay its complete matched closing sequence when
reasoning ends; that sequence can activate the grammar. Speculative rollback
and replacement reconstruct activation from the surviving history.

For example, these request fields allow a free prefix before a constrained
answer (alongside the route's normal prompt/messages fields):

```json
{
  "grammar": "root ::= \"ANSWER:\" [a-z]+",
  "grammar_lazy": true,
  "grammar_triggers": [{"type": 1, "value": "ANSWER:"}],
  "preserved_tokens": ["ANSWER:"]
}
```

As in the pinned upstream sampler, raw `%llguidance` Lark remains eager and
ignores lazy token/pattern effects, while tokenizer-level word preservation is
still validated. JSON `response_format` and native `json_schema` also accept
lazy activation. They use the same trigger ordering, raw newly generated bytes,
reasoning gate and ending roles as GBNF; assistant prefill and declared content
projection do not feed the activated JSON value. The complete JSON Schema
recognizer and equivalent JSON escape handling are shared with eager JSON.
Before a trigger, arbitrary text and EOS remain legal. After activation the
captured suffix must be a viable schema prefix and EOS requires a complete value;
length limits and explicit stops may still truncate it. Selected triggers are
validated before publication even on the final generation step. A JSON trigger
therefore normally captures the opening brace rather than an answer label:

```json
{
  "response_format": {"type": "json_object"},
  "grammar_lazy": true,
  "grammar_triggers": [{"type": 2, "value": "ANSWER: (\\{)"}]
}
```

Lazy JSON requires the same identity raw byte decoder capability as raw GBNF,
a positive source/output byte budget and `ignore_eos=false`; it conflicts with
forced tool choice. Schema/trigger compilation happens before SSE. Collating symbols (`[[.a.]]`,
`[[.space.]]`) and equivalence classes (`[[=a=]]`) have explicit classic-locale
byte semantics. Because C++ libraries disagree, nxrt accepts the request extension
`grammar_trigger_collation`: omission/null or `"libcxx"` preserves the existing
libc++ contract; `"libstdcxx"` selects GNU collating names and ASCII-case primary
equivalence. For example, `[[=a=]]` matches a in the former and a/A in the latter.
There is no ambient-locale or host-compiler inference, or Unicode/accent folding.
Model properties advertise both choices and the fixed default.

Named punctuation, controls and digits follow the selected library; accepted
names and legacy NUL behavior are covered by paired C++ reference fixtures.
Unsupported names and malformed delimiters fail before SSE with profile and byte
location. The default profile permits collating symbols at both range ends and
independent equivalence/POSIX terms followed by a hyphen: `[[.a.]-[.c.]]` matches
a through c, while `[[=a=]-c]` matches a, hyphen or c. GNU requires a literal
right endpoint (`[[.a.]-c]`) and rejects a class as a range start. Its otherwise
unmatched closing bracket is literal; the default profile rejects it.
Trigger source/input bytes, combined translated
program size, captures, nesting, repetitions, character-class items, history
length and backtracking have declared bounds; exhaustion is an actionable error.
Neutral `grammar_lazy=false`, `grammar_triggers=[]` and `preserved_tokens=[]`
remain accepted. Request-local token preservation and native probability
reporting are described below.


## Native token probabilities

`/completion` and `/infill` accept nonnegative `n_probs` (0 disables reporting)
and boolean `post_sampling_probs`. Counts are clamped to the actual vocabulary.
The default reports the original model distribution before grammar masks,
penalties, temperature or truncation. Setting `post_sampling_probs=true` reports
the processed distribution instead. The sampling path and random stream are
unchanged by reporting.

Buffered responses contain `completion_probabilities`; streaming token events
contain their own probability item. Each item includes `id`, `token`, `bytes` and
either `logprob`/`top_logprobs` or `prob`/`top_probs`. Alternative entries carry
their own IDs, decoded byte pieces and score. Invalid UTF-8 suffixes are omitted
from `token` while original `bytes` remain intact. Zero probabilities use a
finite lowest log-probability in JSON; processed alternatives with zero mass
are excluded.


## Preserved output tokens

`preserved_tokens` accepts token spellings on chat, legacy and native completion
requests. Each spelling is encoded with the requested model, without automatic
BOS/EOS and with registered special tokens enabled. A spelling that resolves to
one ID is retained in output; spellings resolving to multiple or no tokens are
ignored, matching the pinned upstream normalization. Duplicate selections are
collapsed. Token ending roles and stopping controls are unchanged.

Preservation belongs to the request. A resolved `response_template` also retains
its declared model framing symbols so reasoning/content/tool projection receives
the same decoded text as generation. Literal boundary tokens are resolved from
the tokenizer; regex boundaries can declare individual registered spellings in
`response_template.preserved_tokens`. Other special IDs remain hidden. The model
tokenizer is not mutated, and buffered/SSE output uses the same policy, including
logprob delivery and transactional publication. UTF-8 decoder lookbehind remains
request-local. Raw grammar recognition continues to inspect model bytes and IDs.

Audio preprocessing admits package-authored mel counts, sample rates, FFT/hop
and Hann-window lengths through the executable log-mel program. Numerical
acceptance includes an independent direct DFT and Native workflows using
nonstandard geometry. Feature-tensor serving also executes the model-authored
audio program, with explicit named output binding and runtime shape checks;
rank-3 shapes never select a hidden default analysis recipe. Pure-waveform normalization, named transform dataflow and validity publication
execute as described in [audio preprocessing](AUDIO_PREPROCESSING.md). Additional
DSP operations, window/mel conventions and codecs remain open; this does not imply
full audio parity.

File transcription now executes declared one-pass CTC/greedy acoustic workflows
as well as the existing generative speech path. Both aliases expose typed
transcript done/delta events, usage, request output budgets and SSE termination.
Native waveform-dependent HTTP acceptance is documented in
[transcription](TRANSCRIPTION.md); generative ASR uses a versioned model-authored prompt/response contract shared
with CLI transcription. Declared language/context conditioning and bounded
buffered/SSE projection execute as described there; richer generative media
accounting remains open.

CLI transcription also executes frame-decoding packages and uses compiled preprocessing
windows at the actual source rate, preserving exact sample cutoffs; encoded-audio
serving accepts waveform outputs without a mel shape. See [transcription](TRANSCRIPTION.md).

Encoded audio support and decoder limitations are documented in
[audio preprocessing](AUDIO_PREPROCESSING.md#encoded-audio-formats).
