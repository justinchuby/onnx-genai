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

This milestone does not check off the full rows above: ordinary-decoder
mid-compute cancellation, native prefill-only, timing/progress, additional
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
ignores; strict all-target Clippy passes. Active-generation browser reconnect
is now verified by the browser milestone below.

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

Native-input follow-up: `/tokenize` and native completions share mixed text/token
encoding with the pinned first-segment special-token policy and token vocabulary
validation. Native completion batches are bounded, fully validated before any
inference and executed sequentially through normal admission. Buffered arrays and
SSE retain input indices; a disconnect or failed stream never submits later
prompts. Native streaming now uses the tokenizer's incremental decoder and checks
its published prefix against the final result, fixing missing word separators.
HTTP tests compare mixed and explicit-token inference and buffered/streamed batch
text; lifecycle tests exercise explicit completion, failure and disconnect.

Native-input validation: full server library 384 passed / 2 inherited ignores,
then all 24 selected native tests passed after the final stop-text and UTF-8
regressions; strict all-target Clippy passes. The complete metadata test suite,
including integration tests and the exhaustive workflow matrix, passes. The CI
classification omission from the ranking fixture is fixed in the same branch.

Engine-text follow-up: the shared token commit path retains request-local
incremental decoder state through ordinary, scheduled and speculative generation.
Stop processors now see sequence text with real word separators, and a real
HTTP regression verifies a multi-token text stop terminates at token two.
Post-commit callback publication uses one text-batch decoder across the engine,
workflow replay, candidate-tree and DFlash paths. Staged batches preserve the
exact full decode even when malformed final bytes revise a prefix; live output
reports an actionable non-monotonic-decoder error instead of retracting bytes.
Native transport consumes those canonical text events without decoding again.
The full engine library passed 611 tests, with one existing external FIM-model
ignore. OpenAI chat/completion HTTP tests compare streamed and buffered text,
including tokenizer-inserted spaces. Legacy OpenAI completion generation controls, default-model behavior and prompt
batching are covered below.

Engine-text checks: strict engine/server all-target Clippy, all 71 CLI library
tests and the 291-asset provenance check pass. The server regression run covered
the complete library; its new fixture-specific output assertion was corrected
to compare actual buffered/streamed results and then passed independently.

Legacy-completion follow-up: `/v1/completions` now lowers its wire-only prompt,
suffix and integer logprobs into the common chat/native generation parameters.
It no longer drops seed, logit_bias or other sampling fields. Default-model
resolution, output/context admission, model defaults, stop visibility and opt-in
streamed usage share the existing implementations. Unsupported controls fail
explicitly, and wrapper deserialization preserves parameter paths in errors.
Real ONNX HTTP tests force a non-default token with bias, verify exact buffered/
streamed usage and text, terminate on a declared stop with max_tokens=-1, and
exercise malformed and unknown fields. Full server: 388 passed / 2 inherited
ignores; all 27 native/alternate protocol tests and strict all-target Clippy pass
after the error-path and source-order lint fixes.

OpenAI batching follow-up: `/v1/completions` accepts text, token rows, mixed
rows, text prompt objects and outer prompt batches using the same classifier
and insertion policy as native completions. One immutable sampling object is
shared across batch inputs. All prompts and output budgets are prepared before
model work, with the common positive prompt-count bound. Buffered responses
combine indexed choices and true summed usage; SSE retains one ID, indexed
choice finishes, one aggregate usage frame and one DONE terminator. Explicit
conversation headers are refused for multi-prompt requests. Delivery failures,
truncated sources and disconnects never manufacture completed usage; a failed
input aborts subsequent submissions. Canonical text chunks now also back the
legacy/chat probability stream content, fixing tokenizer separator loss there.
Real ONNX batch tests cover all input shapes with/without logprobs, exact totals,
indices, pre-header validation and single-prompt conversation constraints.

Batch ownership checks: native batches now retain one resolved model handle
through all preparation and execution, so concurrent admin reloads cannot mix
tokenizer/model versions inside a request. Multi-prompt legacy workflow-output
extras carry prompt_index around the unchanged runtime publication, preserving
input ownership in both buffered JSON and SSE. Single-input records keep their
existing shape.

OpenAI-batch validation: full server library 392 passed / 2 inherited ignores.
After the final model-snapshot and publication-ownership checks, all 29 selected
native/alternate-protocol tests and strict all-target Clippy pass.

Active-browser milestone: the real Chrome smoke now covers an initial reader
disconnection followed by automatic replay from a nonzero byte offset, then
a page reload while the same server producer remains active. A temporary,
bounded extension of the tiny fixture's position table permits real 16,384-token
ONNX generation without changing runtime behavior or delaying the network.
The frontend makes exactly one generation POST, automatically issues two replay
requests, renders the complete retained answer and reports no runtime errors.
The final script passed, including subprocess/proxy/fixture cleanup. Existing
HTTP/unit checks cover byte boundaries, reader admission, replacement, history
limits and expiry.

Cancellation audit follow-up: the server creates a receiver-closure probe but
currently attaches GenerationControl only to specialized speculative paths.
DELETE already cancels transport retention, but true mid-compute
cancellation/worker release still needs implementation and executable Stop tests.
The complete resumable-generation delivery row therefore remains unchecked.

Generic-workflow cancellation: the engine accepts GenerationControl for declared
workflow execution, checks it before each authored node and resolves the atomic
commit race before semantic publication. A forwarding host preserves the original
node executor, tool observer and provisional output transport; cancellation uses
the existing transaction abort path. Specialized drivers retain their own control
authority rather than competing with a second commit owner. Tests cancel after
two real hosted loop iterations, retract a prior provisional revision, cover both
commit-race outcomes, and preserve a previously committed session value/version.
Fused decoder propagation and the server's ordinary receiver-closure wiring
remain outstanding; this milestone does not claim frontend Stop is complete.

Generic-cancellation validation: full engine library 613 passed / 1 inherited
ignore; full server library 392 passed / 2 inherited ignores. Final focused
checks selected and passed both cancellation tests plus the session-baseline
regression. Strict all-target Clippy passes for both engine and server.
