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
- [ ] Resumable SSE: byte-exact replay, explicit cancellation, replacement, bounded history, expiry, frontend refresh/reconnect; finish destructive native-cache rollback audit.
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

This milestone does not check off the remaining rows above: native prefill-only,
timing/progress, additional
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

Cancellation audit found a receiver-closure probe attached only to specialized
speculative paths. DELETE initially cancelled transport retention without proof
of mid-compute termination. The decoder and browser milestone below resolves
that gap; transport cancellation alone was not counted as completed delivery.

Generic-workflow cancellation: the engine accepts GenerationControl for declared
workflow execution, checks it before each authored node and resolves the atomic
commit race before semantic publication. A forwarding host preserves the original
node executor, tool observer and provisional output transport; cancellation uses
the existing transaction abort path. Specialized drivers retain their own control
authority rather than competing with a second commit owner. Tests cancel after
two real hosted loop iterations, retract a prior provisional revision, cover both
commit-race outcomes, and preserve a previously committed session value/version.
This generic-workflow checkpoint preceded the fused decoder propagation and
server wiring verified below.

Generic-cancellation validation: full engine library 613 passed / 1 inherited
ignore; full server library 392 passed / 2 inherited ignores. Final focused
checks selected and passed both cancellation tests plus the session-baseline
regression. Strict all-target Clippy passes for both engine and server.

Decoder-cancellation milestone: one driver receiver-closure authority now reaches
generic, ORT and native requests, including cold/session generation. Ordinary
node hosts observe it between authored nodes and after decoder forwards; the
outer decoder participant retains the semantic commit race and baseline rollback.
Speculative blocks observe the same authority without installing a transport
callback or changing token publication. Native accepted-token staging also checks
cancellation. The obsolete unsupported-control type and specialized-only server
gate are removed. Real ORT and native CPU tests cancel after the second forward,
publish no provisional tokens, restore a warm session, release the scheduler and
retry with the same result as an independent engine.

The real Chrome smoke now additionally clicks Stop while a real worker owns a
long generation, observes DELETE /v1/stream returning 204, waits for active_turns
to reach zero and verifies the cancelled replay returns 404. Basic chat, completed
byte replay, injected reader disconnect, active reload, exact final token count
and zero browser errors all pass with the same built server. The full resumable
SSE transport/browser checks pass; the destructive native-cache rollback audit
below still gates the complete row. Remaining compatibility rows and PR merge stay open.

Decoder validation: full engine library 614 passed / 1 inherited ignore; native
feature set 787 passed / 1 inherited ignore. The final focused run selects and
passes three tests covering warm ORT/native sessions, cold native generation and
native prompt lookup. Full server library 392 passed / 2 inherited ignores;
strict server and native-enabled engine all-target Clippy pass.

Commit-boundary self-review: native session generation published its logical
token history and LRU stamp before staged-output validation and the cancellation
commit gate. A new regression reproduced a cancelled warm session growing from
2 to 5 tokens. Logical history is now published only after validation and the
commit race succeed; retention eviction follows transaction commit. The ORT and
native regressions now cover both cancellation after two forwards and cancellation
at BeforeSemanticCommit, including exact token-history/access-stamp preservation
and scheduler release.

Before closing cancellation, audit physical KV restoration when a native request
replaces or rewinds an existing prompt, rather than appending to its cached
prefix. The tiny native fixture has constant logits, so matching retry token IDs
alone cannot prove that old KV bytes were restored. Add a content-dependent state
oracle and verify the backend checkpoint/rollback boundary directly. Native
prefill-only support must use that same trustworthy state boundary; n_predict=0
must evaluate the prompt without sampling, not merely return an empty success.

Commit-boundary validation: the new native case failed before the fix (5 retained
tokens versus the committed 2), then all three selected cancellation tests passed
with both boundary cases. Full native-enabled engine library: 788 passed / 1
inherited ignore. Strict native-enabled engine all-target Clippy passes.

Physical-KV audit: a content-dependent ONNX cache fixture reproduced silent
corruption after cancelled prompt replacement: the restored length was correct,
but KV value 0.0 had become 1.0. Rollback now distinguishes an unchanged prefix
from a destructive overwrite/reset. Host-owned state takes a complete tensor and
hidden-conditioning snapshot, with an explicit governor lease reserved before
copying. Abort consumes the snapshot without allocating another copy. Tests
compare exact KV values/shapes and verify append requests avoid full copies,
snapshot bytes are charged, and transaction resolution releases the reservation.

Named native conversations on in-place/device/compressed backends without a
complete checkpoint now refuse destructive replacement before mutation with a
typed 409. Stateless requests hold only a discardable implementation cache and
continue to accept replacement. The old rollback-poison regression is retained
on the prefix-preserving append path, where rewind is still the correct rollback
primitive. Complete state snapshots for the remaining backends and prefill-only
execution, remain required work; this does not close the cancellation row.

Physical-KV checkpoint validation: the content-dependent test failed before the
fix (1.0 instead of the committed 0.0), then passed with complete snapshots.
Full native-enabled engine library: 790 passed / 1 inherited ignore. Final
focused runs selected and passed exact KV/resource restoration, in-place refusal
plus stateless replacement, and rollback-failure poisoning. The typed server
409 mapping test passes; strict native engine and server all-target Clippy pass.

In-place CPU checkpoint milestone: complete native snapshots now retain live
rows from the persistent GQA bindings, in addition to host tensors and hidden
conditioning. Per-binding geometry and dtype come from the actual binding;
snapshot size excludes unused capacity. Restore preflights every port, dtype,
non-sequence axis and row length before writing, and maps each batch/head block
onto the current physical capacity, including after cache growth. Native engine
tests now exercise successful named-conversation replacement and cancellation
with exact CPU KV snapshot comparison. Stateless replacement remains supported.
Separate multi-head/multi-binding tests overwrite and grow the destination before
restoring, and prove a later incompatible binding cannot partially overwrite an
earlier valid one. Full native engine library: 792 passed / 1 inherited ignore;
strict native engine and server all-target Clippy pass. Device and compressed
state checkpoints and the broader compatibility rows remain unfinished.

Compressed host-state milestone: complete checkpoints now retain compressed
records, record cursors and unfinished ratio-4/ratio-128 carries at the saved
boundary. Direct ONNX tests reset/overwrite the model, restore both borrowed and
owned snapshots, compare every state shape/dtype/byte, and compare subsequent
calculation with an untouched instance. A real Engine test cancels an edited,
partially shared prompt before semantic commit, restores all compressed state,
publishes no tokens, retries successfully and exercises stateless replacement.
Non-sliceable state is reset/re-evaluated for edited prefixes rather than treated
as a token-sliceable cache. Prefix snapshots and stateless cache reuse honor the
canonical prefix_reusable policy; complete transaction recovery does not grant
shared-prefix reuse.

The Engine test exposed a loader refusal that treated seven growing state ports
as ordinary key/value pairs. The shared paged bridge now resolves canonical
role/layer identities, retaining all other ports for the decoder while selecting
only its actual key/value pairs. Port order is irrelevant, and a uniquely
identified unindexed pair needs no invented layer number. The explicit flat ABI
keeps its existing pairing validation when no canonical groups are present.
Tests cover shuffled declarations, unindexed pairs and missing-role refusal.
Full native engine: 795 passed / 1 inherited ignore; full server: 392 passed /
2 inherited ignores. The final canonical-role regression and strict native
engine/server all-target Clippy pass. Device-resident complete checkpoints and
the remaining compatibility features, conformance, review and merge remain open.

Dynamic-temperature follow-up: `dynatemp_range` and `dynatemp_exponent` now lower
into a typed entropy-temperature processor instead of accepting only neutral
values. The interval and exponent formula follows the pinned upstream
[`llama_sampler_temp_ext_apply`](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/src/llama-sampler.cpp).
The processor occupies nxrt's existing temperature stage; custom sampler order
remains outstanding. Zero range disables adaptation, zero temperature retains
greedy mode, and non-finite/negative controls fail with their field names.
A distinct processor capability prevents the fixed device sampler from silently
substituting a constant temperature. Authored sampler graphs without a matching
input contract refuse active dynamic temperature before admission through a typed
400 package-capability error. Formula, point-mass, custom/greedy chain, model
sampling-default and actual seeded HTTP endpoint-equivalence tests accompany the
change, including a fixed-base control that must produce different tokens.

Dynamic-temperature validation after integrating main through #2430: full native
engine library 799 passed / 1 inherited ignore; the authored-sampler refusal
integration filter selected and passed one test; full server library 394 passed /
2 inherited ignores. Strict all-target engine/server Clippy and the native
benchmark library/profiling-bin Clippy pass. The seeded HTTP test also proves
that disabling adaptation changes the fixture's output, so an ignored control
cannot satisfy endpoint equivalence. PR remains draft while the other ledger
items are implemented.

Prompt-only milestone: native `n_predict=0` / `max_tokens=0` now lowers to an
independent engine PrefillRequest rather than a zero-iteration generation loop
or a sampled throwaway token. Both native and ORT perform a real decoder forward,
share ordinary driver/worker admission and atomic cancellation, and publish no
output token. A prompt-only scheduler reservation does not demand space for an
extra output token. The structural admission requires an isolated runtime-owned
decoder/policy/emit loop without authored setup or effects; other packages must
export a prefill operation rather than have their workflow silently skipped.

The real cache test exposed reporting-only ORT cross-request reuse: the old
token-prefix vector remembered token matches after its runner was destroyed. It
is removed. Portable runners now export compact, byte-exact KV snapshots into
the shared prefix trie, with explicit host-memory leases for retained and staging
bytes. A new request imports that state and reports only the materialized prefix;
unsupported or memory-constrained restores perform full evaluation and report
zero reuse. Snapshot byte slicing follows tensor strides and element sizes, not
model identities or fixed attention geometry.

Prefill reuse also honors every declared decoder state group's prefix policy;
`prefix_reusable=false` never publishes or loads a cross-request snapshot.
Restoring a cached prefix requires its actual physical runner state to be
installed before the hit count is returned; the regression inspects those
restored bytes against the saved state. Raw KV clone/prefix extraction preserves
element bytes (including non-contiguous heads and multiple element types).

The zero-output operation is currently exposed by the native completion/infill
adapter. OpenAI-compatible zero-budget request behavior still needs a separate
contract audit; positive-output GenerateOptions retain their existing invariant.
Composite workflow prefill, native media/infill extras, timing/progress, custom
sampler order, device checkpoint/state/adapter operations and the other unchecked
rows remain open. This milestone does not complete the overall goal or authorize
marking the draft PR merged before those delivery requirements are satisfied.

Prompt-only validation: full native engine library 805 passed / 1 inherited ignore
and full server library 395 passed / 2 inherited ignores. Final focused checks
selected and passed all 7 prefill engine tests after physical-byte and cold-start
counter refinements, plus the real native HTTP batch/SSE/context-limit regression.
The authored-workflow refusal integration test selected and passed one test.
Scheduler validation passed all 72 unit and 15 pressure-conformance tests. Both
portable KV byte/invalid-shape tests pass. Strict engine/server/ORT/scheduler
all-target Clippy passes; git diff whitespace checks pass. These are functional
checks, with no performance claims.
