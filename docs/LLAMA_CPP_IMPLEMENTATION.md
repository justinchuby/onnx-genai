# llama.cpp compatibility implementation ledger

Goal: maximize protocol and official-frontend compatibility, implement missing
server features with real runtime semantics, remove the terminal REPL, then
self-review and merge PR #2427.

The upstream baseline is pinned in `vendor/llama-chat/UPSTREAM`. A matching route
name or a successful empty response is not an implemented capability.

## Delivery order and acceptance

- [x] Embed Llama Chat, package provenance and licenses, remove REPL and dependencies.
- [x] Discovery, model properties/management, tokenizer and chat-template endpoints.
- [x] Preserve the noninteractive generation/transcription/profiling commands.
- [x] Create working branch and draft PR; incorporate current main before further work.
- [ ] Resumable SSE: byte-exact replay, explicit cancellation, replacement, bounded history, expiry, frontend refresh/reconnect.
- [ ] Native completion/embedding/infill protocols and native request parameters.
- [x] Token-counting protocols using the inference preparation path, including declared media.
- [ ] Responses and Anthropic Messages, including tools and streamed event contracts.
- [ ] Generation controls/sampling and reporting needed by the upstream frontend.
- [ ] Capability-backed ranking, adapter and slot/state operations; audit the engine boundary before introducing transport support.
- [ ] Optional tools/MCP proxy, model catalogue/download semantics and audio transports; preserve explicit configuration and resource bounds.
- [ ] Full compatibility matrix and executable conformance tests.
- [ ] Fresh full checks, adversarial self-review, PR description update, CI and merge.

## Design constraints

Every wire protocol lowers into the existing admitted model/driver transaction.
The compatibility layer does not choose model architecture or execution providers.
Replay is a transport concern and owns retained response bytes, not inference
state. State export and LoRA/ranking require declared engine capabilities, not
synthetic slots or embedding similarities presented as cross-encoder scores.

All substantial local builds/tests/inference share the stable Darwin host lock
with the performance-research session. These are functional tests and carry no
performance claims.

## Protocol milestone

The implementation now includes bounded resumable SSE (byte offsets, replacement,
explicit cancellation, reader admission and completed-history cleanup), native
completion/infill/pooled embedding adapters, text input-token counting, and
buffered/streamed Responses and Anthropic Messages adapters. These all execute
through the admitted engine driver. A real tiny-model HTTP test covers both
alternate protocols and their streamed usage; synthetic stream tests pin reasoning,
tool indices and incomplete-stream rejection.

This milestone does not check off the full rows above: active-generation browser refresh/reconnect,
native batching/prefill-only, timing/progress, additional
sampling and runtime-backed state/adapter work still need completion.

Sampling follow-up: `logit_bias` now lowers OpenAI objects and native token/text
pairs into one typed engine processor, including explicit bans. Token vocabulary
validation is shared by the server and engine. Real tiny-model generation and
engine processor tests pass, including the greedy/device-shortcut guard.

Latest verified checkpoint: server library 375 passed / 2 inherited ignores,
4 protocol-history tests, real logit-bias HTTP and engine tests, 6 DRY engine
tests, and the multi-token DRY request mapping test pass. Strict server all-target
Clippy passes. The real Chrome smoke now verifies chat, persisted conversation
and byte-identical replay after reload, with no browser errors. CLI library tests:
71 passed. Vendored asset and documented environment-variable checks pass.
No merge has been performed.

Speech follow-up: native `/tts` now lowers input/prompt into the existing declared
speech workflow and returns real PCM WAV. Explicit raw prompts bypass text
assembly while retaining resource limits. Unsupported voice and synthesis fields
are rejected, including on the OpenAI speech API. All 13 selected speech tests
pass, including actual native synthesis; strict server all-target Clippy passes.

Media-counting follow-up: buffered/streamed chat and all token counters now use
one input-preparation path, including tool admission and declared media expansion.
The fallback prompt formatter and `/apply-template` preserve image placeholders
for packages without a chat template. An executable hermetic VLM HTTP test proves
that Chat, Responses and Anthropic media counts equal real generation usage;
the OpenAI count routes also return the required response.input_tokens object.
Full server validation: 369 passed, 2 inherited ignores; strict all-target
Clippy passes.

Tokenizer/error follow-up: native parse_special=false now treats registered
special-token spellings as ordinary text, independently of BOS/EOS insertion,
without mutating shared tokenizer state. Core library and real HTTP tests pass.
Explicit API errors now carry numeric HTTP code and native request/not-found/
permission classifications. Messages errors include the Anthropic envelope even
for JSON extraction failures, retain retry headers, and late SSE errors preserve
the original status/category across adapters. Full server: 375 passed, 2 inherited
ignores; strict all-target Clippy passes. The active-generation browser reconnect
scenario still needs work.

Byte-piece follow-up: native with_pieces now preserves incomplete UTF-8 using
byte arrays. The existing byte vocabulary adapter is invoked once per requested
batch and special framing is removed only when it matches marker+spelling;
literal 0xff remains intact. ByteFallback and ByteLevel engine tests, special-token
framing, invalid-id checks and an HTTP Unicode round-trip pass. Strict server
all-target Clippy passes.

Reranking follow-up: `/rerank`, `/reranking` and both `/v1` aliases now use a
metadata-declared token-pair workflow with the package tokenizer's paired IDs,
attention mask and segment IDs. Jina/TEI shapes, stable ordering, top_n, exact
non-padding token usage and positive document/token budgets are covered by real
ONNX HTTP tests. An actual raw-logit workflow is rejected with repair guidance;
no embedding similarity is substituted for cross-encoder scores. Metadata guards
cover ambiguous/unsupported profiles, vector scores and mismatched fixed inputs.
Adapter and slot/state work in the shared delivery row remains outstanding.

Reranking checkpoint: full server library 379 passed / 2 inherited ignores;
metadata library 76 passed; both real engine ranking tests passed, including
invalid score rejection. Strict server all-target Clippy and pinned asset checks
pass. PR remains draft while the remaining compatibility work continues.
