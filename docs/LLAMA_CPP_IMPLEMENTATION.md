# llama.cpp compatibility implementation ledger

Goal: maximize protocol and official-frontend compatibility, implement missing
server features with real runtime semantics and remove the terminal REPL.
Self-review and merge tested increments until all acceptance items are complete.
Tool execution and MCP transport are explicitly excluded by the user's revised
scope; external harnesses own them. Chat tool-call protocols remain required.

The upstream baseline is pinned in `crates/onnx-genai-server/vendor/llama-chat/UPSTREAM`. A matching route
name or a successful empty response is not an implemented capability.

## Current goal status

PR #2427 delivered the first compatibility implementation and is merged. This
is not completion of the original goal. The user requested continuing all
remaining work on 2026-11-23. Unchecked acceptance items below remain required;
a successful merge or an endpoint name alone does not close them.
The supported surface and explicit limitations are documented in
[`LLAMA_CPP_COMPATIBILITY.md`](LLAMA_CPP_COMPATIBILITY.md).

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
- [ ] Complete model catalogue/source-format semantics and audio transports; preserve explicit configuration and resource bounds. Native-package download, resume and deletion are implemented; GGUF import and sidecar selection remain open.
- [ ] Full compatibility matrix and executable conformance tests.
- [ ] Fresh full checks, adversarial self-review, PR description update and merge (CI is currently blocked; user authorized merging after local verification).

## Native slot state and HTTP transport

The engine now exposes owned `NativeSessionSnapshot` captures and transactional
restore into existing sessions. These preserve executable state (live in-place
CPU KV rows, host past tensors including declared recurrent/compressed state,
hidden conditioning and physical cursor) together with authoritative logical
tokens. Capture and restore undo allocations reserve host memory first. Captures
are bound to the original engine instance/generation and remain valid after the
source conversation advances, resets or closes. Restore failures undo physical
writes; failed undo poisons the affected conversations until reset.

Inactive histories and device state are explicitly refused as snapshot sources. Bounded, checksummed
[Native state files](NATIVE_SESSION_STATE.md) now preserve that boundary across
engine instances/restarts after load-time model/weight/metadata/tokenizer identity
validation. Strict schema, layout, cursor, token and checksum checks precede
transactional publication. CPU live KV can restore into a smaller initial
capacity. `GET /slots` now exposes the actual serial Native lane (ID 0),
including owner-thread busy status and the last completed task's retained cursor.
`POST /slots/0?action=save|restore|erase` uses the same owner queue and requires
`--slot-save-path`. Plain filenames, a pinned capability directory, private
atomic staging, exclusive process ownership and file/directory/count quotas
bound filesystem access. Model properties advertise slots only for admitted
executors. Device/multi-sequence persistence and Native fork plans remain open.
Real HTTP tests cover empty/generated cache round trips, continued generation,
busy polling and queued save, corruption, disabled storage and slot selection.

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
Declared host LoRA selection and its HTTP catalog/default controls now execute actual weights; alternative artifact loaders, device-fused application and device/multi-sequence slot persistence remain outstanding.

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

Ordered-sampling milestone: `samplers` now has a bounded typed sequence shared
by native, legacy OpenAI completion and chat requests. Defaults use upstream
penalties/DRY/sigma/K/typical/P/min-p/XTC/temperature ordering, with the nxrt top-A
extension between min-p and XTC. Omissions, empty lists, repetition and upstream
name aliases are executable behavior. Mandatory bias/grammar/stops remain outside
the optional list. Mirostat selects its separate temperature/selector mode.
Top-n-sigma now performs a real finite-logit population-deviation mask.

Device dispatch no longer trusts processor names: a cached typed transform proof
checks both parameter values and equivalent operation order. Adjacent active
top-k/top-p still fuse; fixed-order sampled transforms and rank-preserving
zero-temperature argmax paths remain eligible. A renamed or foreign processor
cannot obtain a shortcut through its diagnostic name. Dynamic temperature tracks
logical candidate counts across truncations, mask-only sigma and XTC removal.
Package capability errors for custom order, sigma and dynamic temperature share
one typed runtime-sampling refusal. Properties advertise only actual executors.

The real HTTP regression swaps penalties and temperature and proves different
expected tokens through all three generation protocols. It also demonstrates
that a temperature omitted from an empty stage list does not force greedy output.
Further sampler conformance work remains, including top-k tie cardinality and
infill selectors; the broad generation-control ledger row stays open.

Ordered-sampling validation: the full native engine library passed 814 cases
with 1 inherited ignore; final refinements passed all 16 selected sampler cases.
The full server library passed 396 cases with 2 inherited ignores. Both selected
authored-sampler integration cases and the actual HTTP ordering/sigma/empty-list
regression passed. Strict native engine and server all-target Clippy passed.
The official frontend passed real Chrome chat and persistence, automatic
nonzero-offset replay and active refresh during a 16,384-token ONNX generation
(one generation POST, two replay requests), and explicit Stop with worker release
and cancelled replay returning 404. No browser errors were reported.
Strict non-native engine all-target Clippy also passes after gating the
native/test-only prefix helper import.

Top-k cardinality follow-up: the ordinary and fused processors share an exactly-k
boundary with a remaining tie quota. The small-k heap retains its bounded
allocation; large-k partitioning derives the quota from its partition. CUDA
uses the same candidate cap, with a token-order tie pass only when the threshold
would exceed k. Stable token-ID ordering makes host/fused/device choices agree;
upstream's partial sort does not promise an ordering among equal logits.
Regression coverage includes heap/partition sizes, high-ranked tokens following
tied entries, NaN, signed zero, infinity, fusion equivalence, the device algorithm
CPU oracle and real ONNX HTTP generation with deliberately equal finite logits.
Infill selectors and the other unchecked generation controls remain open.

The strict tie cap also exposes why temperature cannot commute across top-k in
f32: two distinct logits can round to equal scaled values, changing which token
survives. Transform proofs now retain exact stage order. The fixed-order device
path remains available; a reversed order uses the host rather than changing the
requested distribution. A regression proves the distinct winners on underflow.
Remaining sampler conformance includes device top-p boundary ties in addition to
infill selectors. Missing CUDA driver/NVRTC libraries now return a
capability error before cudarc's dynamic-entry panic path.

Device temperature now divides logits directly, matching the host instead of
forming a reciprocal that can overflow for a finite subnormal temperature.
CUDA library presence is checked before dynamic calls; missing driver/compiler
errors identify the device, missing library and remediation. This macOS host
executes the CPU differential tests and CUDA-feature Rust checks; it cannot prove
execution of the changed CUDA kernel on actual GPU hardware.

Top-k validation: full native engine library 817 passed / 1 inherited ignore.
All 8 selected top-k engine tests and the real HTTP equal-logit generation test
passed. Four selected CUDA-feature ORT cases passed: strict top-k CPU differential,
broad device-algorithm CPU sweep, subnormal-temperature division, and actionable
missing-library errors. Actual GPU subcases were unavailable on this host.
Strict native-engine all-target Clippy passes.
Strict CUDA-feature ORT and server all-target Clippy also pass. CUDA top-k
candidate counts use integer reductions, avoiding float rounding in large
vocabularies; actual GPU execution remains a separate hardware validation gate.

Shared-penalty milestone: one `PenaltiesProcessor` replaces three independent
processors and counts the same prompt-plus-generated suffix once. Repetition
rescales each seen logit once, then count-weighted frequency plus presence is
subtracted in one expression, matching upstream. `GenerateOptions.penalty_window`
now names the scope of all three controls; internal callers and benchmarks use
the new processor directly, with no aliases for our pre-release symbols.

The server parses `repeat_last_n` as a typed common request field, defaults it
to upstream's 64, and advertises that default in properties. Minus one selects
full history, zero disables every penalty, and invalid negative/host-oversized
counts fail with an actionable error. Disabled penalties preserve the device
argmax path. Unit regressions cover cross-boundary prompt/generated windows,
positive and negative coefficients, neutral/zero windows and combined rounding;
the HTTP regression exercises first-token prompt counts through all three APIs.

Shared-penalty validation: full native engine 818 passed / 1 inherited ignore;
the final focused check selected and passed 12 penalty-related cases including
the added zero-window device path and joint-subtraction rounding guard. Full
server library 398 passed / 2 inherited ignores, including the real three-protocol
prompt-count/window regression. Strict engine and server all-target Clippy pass.
The updated benchmark library and native profiling binary also pass strict
Clippy with the native backend enabled.

Adaptive-p milestone: `adaptive_p` is an explicit terminal selector, appended
once after optional stages even if repeated or listed first. The upstream
quadratic/linear probability transform uses the adapted target and maintains
a weighted EMA of original selected probabilities. Defaults are target -1
(disabled) and decay 0.9; valid targets are negative or [0,1], valid decay is
[0,0.99]. Parameters alone do not activate a selector omitted from `samplers`.
Mirostat's separate mode still takes precedence. Native, legacy and chat lower
the same typed configuration and properties advertise the neutral defaults.

The shared processor API now returns Results and records selected-token
feedback before a token can be committed. Ordinary, continuous-batch, draft and
speculative target paths propagate those failures. Adaptive-p checkpoints the
EMA before each staged choice, so reprocessing an accepted prefix rolls back
discarded branches. Histories containing unobserved choices fail explicitly;
normalization and poisoned-state errors retain stage/offset/remediation. A
cached feedback capability prevents device bypass even for processors that
otherwise preserve argmax. Stateless chains skip feedback dispatch entirely.
Public processor callers and benchmark targets handle the new Result contract;
benchmark top-k reference selection also uses the exact candidate cap.

The ORT prompt-lookup regression executes real proposal rejection and verifies
exact output against ordinary adaptive-enabled selection. Native's current
speculative driver only supports empty processor chains and raw greedy choices;
stateful requests therefore use the ordinary native sampler. Its regression
requires the adaptive selector to choose a low-original-probability token for
eight steps and verifies identical results when prompt lookup is requested,
with no speculative proposals reported. This preserves the existing executor
capability boundary rather than claiming native stateful speculation is present.

Adaptive-p validation: full native engine library 825 passed / 1 inherited
ignore; all 4 selected adaptive cases passed after final numerical/decay and
same-length-history-integrity refinements. ORT speculative rejection and native
ordinary-path capability integration each selected and passed one real test.
Full server library 399 passed / 2 inherited ignores, including native/legacy/chat
four-token selection, negative/omitted-selector behavior, invalid parameter
rejection and real sampling-stage failure propagation without a fabricated token.
Strict engine/server all-target Clippy and native benchmark library/profiling/
no-model benchmark Clippy pass. Other unchecked delivery requirements remain open.

Infill-selector milestone: the explicitly ordered `infill` stage implements the
upstream ending-mass check, probability-rank prefix merging, two retention
thresholds and preferred forced ending. It reuses lossless tokenizer byte pieces,
including non-UTF-8 fragments, and excludes tokenizer-declared special pieces from prefix merging. A numeric
ending role alone never changes an ordinary token's byte piece. Immutable lexical prefix ranges are shared across repeated
stages; a live rank-minimum index visits only prefix-related active candidates.
Every merge removes one candidate, bounding merging to O(V log V) rather than
copying upstream's O(V²) nested scan into a public endpoint. A literal small-input
reference checks rank-order equivalence, ties and byte fragments.

Identity-sensitive chains now carry the actual logical candidate ids in addition
to cardinality. Top-k/fused-k/p/probability truncations and XTC describe their
retention; mask-only stages preserve slots. Dynamic temperature after infill
therefore sees the correct denominator, including retained masked EOS records.
An extension that changes cardinality without an identity contract fails clearly.
Chains without such a stage retain their prior count-only/stateless execution.

Package token facts optionally declare `eot_token_id`, constrained to the
existing EOS set. One validated end-token policy flows through ordinary and
continuous/static batch paths; explicit EOS overrides remain authoritative.
The fallback can restore a role excluded by an optional sampler, while the
input/model and mandatory bias/grammar masks remain binding. Unavailable endings, byte codecs
and vocabulary inconsistencies report actionable failures. Schema and token
authority documentation describe the new optional role. Integrated main #2431.

Preferred-ending facts introduce schema v1.9. Both untyped parser presence checks
and typed metadata validation use the same minimum-version gate; older versions
remain accepted without the new role. Future-version refusal tests now use v1.10,
and v1.8 compressed-state documents retain their existing minimum version.
The true HTTP package declares its publication mode and materialized output
family when exercising the preferred EOT role, instead of relabeling a legacy
document with a newer version while omitting its required contracts.

Metaspace byte pieces retain their initial whitespace independently of full-text
first-token decoding. Decoder normalization is local to piece extraction and
never mutates the shared tokenizer; registered special spellings remain literal.
The regression distinguishes space-prefixed and ordinary vocabulary pieces.

Infill validation: full metadata tests passed; full native engine library
834 passed / 1 inherited ignore, and full server library 400 passed / 2 inherited
ignores. The HTTP case covers native/legacy/chat endings, real byte-prefix
selection and a package-declared preferred EOT fallback. After the final
Metaspace correction, all 3 selected tokenization and 7 selected infill cases
passed. Strict metadata, engine and server all-target Clippy and native benchmark
library/profiling/no-model benchmark Clippy passed. These are functional checks
on this macOS ARM64 host; remaining delivery rows above remain open.

Content-continuation follow-up: official Llama Chat's explicit boolean continuation
request now retains the final assistant prefix instead of opening a new message.
A differential template render proves its position and rejects dropped,
duplicated or transformed final content; the built-in formatter preserves the
same original prefix bytes. Template rendering, count routes and real buffered/
streamed generation use this shared preparation. Empty/reasoning-only prefixes,
assistant tool calls and persistent-session replay retain explicit refusals;
reasoning-parser resumption, string continuation modes and upstream's implicit
prefill heuristic remain open work rather than an advertised completed row.
Constrained JSON continuation is also refused until grammar admission consumes
the existing assistant prefix; starting a fresh JSON grammar for a partial object
would accept the request while generating the wrong language.
The behavior is checked against the pinned upstream server's
`common/chat.cpp` continuation admission and
`common/chat-auto-parser-generator.cpp` generation-prefix assembly, together with
Llama Chat's `chat.service.ts` continuation request. Unlike an EOS-name heuristic,
our renderer verifies the loaded template's actual content boundary.

Continuation validation: 2 selected continuation cases passed, followed by the
complete server library (402 passed / 2 inherited ignores) and strict server
all-target Clippy on the final tree. Tests cover exact template/fallback framing,
Unicode and original whitespace, earlier matching text, dropped/duplicated/
transformed content, shared count/generation preparation, buffered and SSE HTTP,
and actionable refusal of conflicting flags, tool-call/reasoning-only prefixes,
JSON grammar without prefix state, and persistent-session replay.

JSON-prefix follow-up: the typed `GenerateConstraint::Json` now owns its existing
text prefix. The shared JSON recognizer consumes that prefix plus the generated
suffix, while response bytes and usage remain suffix-only. The same joined text
controls EOS masks, stop-sequence completion and output-budget validation.
Non-continuation callers explicitly construct an empty prefix; all internal and
HTTP integration callers are updated without retaining the old unit variant.
Invalid prefixes fail before HTTP inference admission and at engine constraint
construction. JSON Schema still needs its own llguidance prefix-state contract.

JSON-prefix validation: both selected engine continuation cases passed; all four
selected server continuation cases passed, including real buffered/SSE suffix-only
generation and a forbidden premature EOS with stronger bias than the valid token.
Full native engine library: 837 passed / 1 inherited ignore. Full server library:
403 passed / 2 inherited ignores. Strict engine/server all-target Clippy passed,
including compilation of the updated public HTTP request-mapping tests.

Schema-prefix milestone: `GenerateConstraint::JsonSchema` now owns its serialized
schema and existing text prefix. Prefilled requests build a lossless byte
vocabulary, frame tokenizer-declared special tokens and every declared ending,
and consume the prefix into llguidance before the first model sample. Prefix
bytes are grammar-only cells of at most 256 bytes, bounding additional trie depth
independently of request length; UTF-8 scalars may span cells. These ids are outside
the model vocabulary and are absent from sampling candidates, response bytes and
usage. No codec fallback silently replaces incomplete bytes with decoded text.

The JSON Schema compiler enters at the value rather than its optional RFC 8259
leading whitespace. That whitespace stays in the prompt; only the recognizer's
entry skips those four ASCII whitespace characters. Interior bytes stay exact.
A vocabulary with no declared EOS receives a recognizer-only ending rather than
silently assigning ordinary token 0 that role. Multiple real endings use the
trie's EOS set. Prefix rejection carries `ConstraintPrefixError` through the
worker boundary to HTTP 400, without classifying error text or converting runtime
failures into request errors. Internal/public HTTP callers use the new schema
variant without keeping a pre-release API alias. Reasoning continuation remains
open work.

The recognizer stores the sampled-vocabulary bound separately from its internal
prefix cells, so a padded model logit tensor cannot expose a prefix cell as a real
token. The regression deliberately includes that out-of-vocabulary candidate
when an integer prefix could otherwise accept its bytes as another digit.
Underlying parser errors remain typed sources inside `ConstraintPrefixError`;
the worker's 400 classification retains the detailed cause and actionable repair.

Schema-prefix validation: all four new engine regressions passed in the full
native engine suite (841 passed / 1 inherited ignore): exact prefix-byte/schema
masks, long UTF-8 chunks without an assumed token-0 EOS, multiple endings with
padded model candidates, and an unsupported trie vocabulary returned as an error
rather than a worker unwind. The real HTTP continuation case now covers both
JSON object and JSON Schema suffix generation, streamed schema output, token
usage and typed HTTP 400 refusal of an undeclared-property prefix. Full server:
403 passed / 2 inherited ignores. Strict engine/server all-target Clippy passed
on the final tree, including updated public HTTP integration callers.

Continuation-intent milestone: a typed `ChatContinuation` replaces the old boolean
field and accepts exactly the pinned upstream boolean/null/content/reasoning forms.
Auto resolution gives existing answer content precedence over reasoning and tests
message-part presence rather than flattening an empty text part into an absent
message. Content-string requests execute the existing proven prompt path;
reasoning intent receives an explicit capability refusal rather than a misleading
JSON type error. Invalid values name accepted forms. The request's one validation
method now serves HTTP extension admission, direct prompt rendering and both public
Rust generation builders, so direct callers cannot silently generate under a
rejected reasoning mode. The public field type and all internal callers change
without a pre-release bool alias. Implicit prefill and reasoning parser resumption
remain open work.

Continuation-intent validation: all 7 selected continuation tests passed, including
wire forms, field/part-aware auto resolution, the public Rust builder and real
HTTP string-content count/generation/SSE paths. Full server library: 406 passed /
2 inherited ignores. Strict server all-target Clippy passed on the final tree.

Declared response parsing milestone: replaced the private-channel presence flag
and hardcoded ATEM text filters with a compiled `response_template` descriptor.
Chat and response contracts load from one strict tokenizer-configuration snapshot
before starting workers. Literal and regex boundaries, last-turn anchoring,
prefill state, split delimiters, repeated text fields, stripping and defaults
follow the pinned
[response-parsing specification](https://github.com/huggingface/transformers/blob/be48e7a0b63d2ff22ce1ebccbccbd0374c1592ef/docs/source/en/chat_response_parsing.md).
The parser excludes original prefix bytes, withholds opaque structured fields,
and fails explicitly on unsupported transforms, invalid declarations, buffer
exhaustion or nonmonotonic streamed replacement. Structured-field projection into
tool calls remains open; an XML tool declaration alone does not prove a chat
recipient protocol.

Buffered chat, live SSE and logprob SSE consume the same declared parser and the
actual rendered text prefix. A canonical incremental decoder preserves protocol
specials and split UTF-8. Declared chat templates own their special tokens, so
encoding does not add a second tokenizer postprocessor BOS/EOS. Undeclared output
remains ordinary text. Parser failures produce explicit SSE error events rather
than a successful terminal chunk. The parser byte budget uses the configured
per-response replay buffer limit. Reasoning continuation admission and lazy
reasoning/JSON grammar activation still remain open requirements.

JSON-failure truthfulness milestone: removed all three transport branches that
converted an incomplete JSON generation error into `{}`, successful `stop` and
zero generated tokens. Buffered and both stream paths now use ordinary typed
failure propagation. Constraint processing refuses an all-masked/nonfinite
candidate set before selection, with grammar name, generated offset and a repair
for incompatible vocabulary or bans; greedy selection cannot fabricate token 0.

The HTTP JSON success fixtures now explicitly declare an actual JSON token and
a tokenizer-special EOS. They run real generation and assert nonzero completion
usage in buffered and streamed responses. A separate original text-only fixture
proves real buffered HTTP 500/SSE error envelopes, with no fabricated JSON delta.
Native-enabled engine library: 842 passed / 1 inherited ignore. Full server:
407 passed / 2 inherited ignores. Both selected public HTTP JSON integration cases
passed, as did strict engine/server all-target Clippy. Parser-module work remains
uncommitted until runtime wiring is finished. JSON-object budget-prefix/length
parity remains open; removing a false success is not claimed to finish that item.

Declared-parser validation: full server library 407 passed / 2 inherited ignores;
the final prompt-special-token regression passed separately (1 selected), and
all 12 response-template selections passed, including real generated buffered,
SSE and logprob SSE output. Both new protocol-decoder tests passed. Strict
engine/server all-target Clippy passed. These checks use the local ORT runtime;
no performance claim is made. Reasoning continuation remains explicitly refused
until its admission and prompt-boundary proof are implemented.

JSON termination parity: removed completeness-only finish rejection from ordinary,
continuous-batch, authored workflow, speculative and native speculative paths.
The grammar owns legal token/EOS selection; explicit stops and output/context
limits independently terminate generation, preserving actual partial text and
true usage. The pinned upstream server applies these same independent stopping
conditions in
[`server-context.cpp`](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/tools/server/server-context.cpp#L2107).
No fallback success is synthesized for a genuine grammar failure.

Stop filtering now precedes chat channel projection in buffered, live and logprob
SSE output. A matched stop is a terminal buffer state, so later decoder bytes
cannot reappear after it. A partial stop prefix flushes only when no match was
completed. This also keeps reasoning-region stop strings out of the response and
avoids treating text joined across separate channel regions as a raw stop.

Termination validation: native-enabled engine library 843 passed / 1 inherited
ignore; the final real JSON Object/Schema context-boundary test selected and
passed 1 additional case, covering both full-context zero-token and one-token
partial output. Final server library 410 passed / 2 inherited ignores, including
real JSON Object/Schema buffered/SSE budget and explicit-stop continuations,
reasoning-channel stops with and without logprobs, and every Unicode split of a
terminal stop boundary. Strict native-engine and server all-target Clippy passed.
JSON SSE now uses ordinary incremental delivery rather than collecting the whole
value before sending content. Remaining delivery rows, including reasoning
continuation and structured tool projection, stay open.

Reasoning continuation now uses the declared response parser's actual prefill
state. Request-shape validation selects and validates the final field; model-aware
rendering proves its unique template position and verifies that the truncated
prefix leaves that same parser channel open. This check also covers content
continuations on packages with declared response templates. Direct model-free
Rust builders refuse reasoning continuation with a missing-contract error.

Explicit continuation suppresses `add_generation_prompt` regardless of the
request's default/explicit true flag, matching pinned upstream
[`common/chat.cpp`](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/common/chat.cpp#L1547).
Reasoning prefix bytes are consumed without echo; ordinary, streamed and logprob
output share the existing generated-only projection. A prefix that closes the
selected channel is rejected before inference. Reasoning plus JSON/forced output
grammars still requires a channel-aware grammar implementation; implicit prefill,
structured tool projection and persistent-session continuation remain open work.

Reasoning-continuation validation: all 9 selected continuation cases passed,
including real ONNX explicit/auto reasoning prefixes, discarded pre-existing
answer content, template/count/inference agreement, buffered/SSE/logprob suffix-only
output, default-true generation-prompt suppression, closed-channel rejection,
missing declarations and constrained-grammar refusal. Full server library:
412 passed / 2 inherited ignores. Strict server all-target Clippy passed.
The unchanged engine layer was not remeasured for this server-only milestone.

Named-template milestone: one loader now owns template presence and asset
selection. It accepts config strings, named objects and name/template lists, plus
independent default and `chat_templates/*.jinja` files. Standalone asset sets take
priority over config entries. The model-free directory API retains its documented
built-in fallback; serving consumes explicit absence from the shared-snapshot
loader instead of repeating filesystem/type presence checks.

Base/tool selection follows the pinned
[`common/chat.cpp`](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/common/chat.cpp#L1519)
nonempty-tool policy and tool-only base fallback. Asset forms and independent-file
priority follow the declared Hugging Face tokenizer format
([pinned loader](https://github.com/huggingface/transformers/blob/be48e7a0b63d2ff22ce1ebccbccbd0374c1592ef/src/transformers/tokenization_utils_base.py#L1822)).
Empty declarations remain absent; nonempty sets need a recognized default/tool-use
base. Duplicate declarations and bad types fail with names and repair guidance;
I/O failures retain their actual path and causal error. `/props` exposes the
explicit tool-use source while existing protocol metadata remains authoritative
for tool execution. Named-template presence does not invent a tool parser.

Named-template validation: full ORT library 187 passed, full server 413 passed /
2 inherited ignores and CLI library 71 passed. After the final empty-asset
refinement, all 17 chat-template cases passed, including the new absence case.
The real HTTP named-template case passed for config lists, maps and independent
files, checking `/props`, exact rendered prompts, tokenizer counts, actual plain/
tool generation and tool-call SSE (with true usage and no raw-tag content leak).
Strict ORT/server/CLI all-target Clippy passed on the final source. No performance
claim is made. Remaining delivery rows and PR merge remain open.

Grammar-state prerequisite: constraint masks now return errors with their original
cause and history position instead of converting backend failures into all-false
masks. The processor checks exact mask dimensions before touching logits.
llguidance tracks committed token IDs, replays replacement/rewound histories from
the original prefilled parser snapshot, reuses valid masks (including terminal
states), and resets after failed replay. Unsupported history splicing is rejected
explicitly rather than silently losing forced tokens.

Validation: full native engine 850 passed / 1 inherited ignore and full server
413 passed / 2 inherited ignores. Subsequent terminal-query and UTF-8 refinements
passed all 3 selected llguidance cases and both selected Lark suffix cases; these
prove first-boundary behavior across a single token, split delimiters and raw
split UTF-8 bytes after prefill. Processor error-cause and malformed-mask cases
also passed. Strict native-engine/server all-target and benchmark Clippy passed.
Criterion functional smoke selected and passed both the five-processor chain and
the cached grammar mask using valid grammar history; no performance claim is made.
Reasoning plus JSON is still refused: shared classifier/grammar lowering and
special framing-token semantics remain necessary before enabling that feature.

Ordinary-prefill projection now differs explicitly from continuation projection.
Pinned upstream `common_chat_session` primes its emitted baseline only for an
explicit continuation without echo. The response parser therefore takes a typed
include/exclude policy: ordinary requests include declared assistant content and
reasoning from the final anchored template region; continuations exclude existing
bytes while retaining their parser state. Buffered output, live SSE and logprob
replay share this policy. Region visibility is named independently of whether its
bytes came from generation. This fixes lost static template prefill without
changing continuation suffix behavior.

Validation: full server library 415 passed / 2 inherited ignores and strict
all-target Clippy passed. After extending the real ONNX fixture to cover both
content and reasoning prefill, all 2 ordinary-prefill cases and the original
real response-projection case passed; strict all-target Clippy passed again.
The fixture checks buffered/SSE, logprobs on/off and stops before/within generated
text. Parser tests cover closed/open repeated regions, Unicode chunk boundaries,
prefix-only completion and last-anchor selection. All 9 selected continuation
cases passed. Constrained prefill and reasoning+JSON still require further work;
this milestone does not enable their missing grammar integration.

Lark-prefill API prerequisite: `GenerateConstraint::Lark` now carries the complete
grammar and existing text prefix explicitly. Processor construction seeds the
same byte-oriented recognizer used for schema prefixes; no duplicate old tuple
API remains. Tool-constraint builders and public HTTP integration tests use an
explicit empty prefix. Grammar-prefix rejection guidance now describes arbitrary
grammars rather than assuming every grammar is JSON Schema.

Actual-tokenizer processor tests cover empty/partial/complete Lark prefixes,
invalid-prefix typed errors, suffix-only next-token selection, EOS eligibility,
and explicit special-token transitions alongside ordinary literal alternatives.
This exposes the engine capability needed by response-channel grammar lowering;
it does not itself enable reasoning+JSON in the server.

Lark-prefill validation: 2 selected engine cases passed under both default and
native-backend features; 3 public HTTP tool-choice tests and all 6 tool-protocol
unit cases passed. The real JSON-continuation regression passed with its updated
generic grammar-prefix repair guidance. Strict default/native engine all-target
and server all-target Clippy passed. The first HTTP filter selected zero tests;
it is not counted as coverage and was replaced by the verified 3-case filter.

Shared response-classifier milestone: the declared response template compiler,
channel state and text projection now live in `onnx-genai-engine::response_template`.
Server loading, continuation validation and buffered/live/logprob output consume
that public engine implementation directly; the former server module is removed.
Private boundary internals remain private. Regex dependencies move with their
owner, with no new package versions introduced.

Regions now store ranges into the request's raw buffer instead of duplicating
body strings. Trimming and omitted-prefix offsets operate on those same bytes;
closed/reopened fields and opaque regions retain the existing matching policy.
This supplies one classifier for subsequent constrained sampling without adding
a second channel grammar or claiming that reasoning+JSON is already enabled.

Validation: all 12 moved classifier cases passed under default features. Full
server library: 403 passed / 2 inherited ignores; the previous 415 count included
the 12 cases now owned by the engine. Coverage still includes actual ONNX normal
prefill/continuation, buffered/SSE/logprobs, stops, named templates and alternate
protocol adapters. No performance claim is made for the storage representation.

Final shared-parser validation: the same 12 cases passed with native-backend;
strict native-engine all-target and server all-target Clippy passed after the
unnecessary lazy region-construction closure was simplified. There are no
remaining duplicate server parser definitions or compatibility aliases.

Response candidate transactions: `ResponseParser::preview` returns a cumulative
candidate projection without publishing or consuming it. A borrow-scoped guard
restores the original buffer length, cursor, active/closed regions and seen flags
on success, error and unwind. Closed region metadata uses copy-on-write sharing;
raw body bytes and published text are not copied into checkpoints. Final preview
uses the same boundary resolution, defaults and required-field checks as finish.

Buffered chat now projects the final raw result once through this candidate path,
rather than imposing streaming append-only delta checks on an intermediate parse.
This fixes a pending regex opener that replaces earlier nonrepeated content only
at EOF: buffered output can return its final empty replacement while live output
still rejects retraction of already published text. Raw stop filtering is retained
before classification. This is a shared-parser capability for the forthcoming
channel constraint, not completed reasoning+JSON wire support.

Validation: all 3 new candidate tests and the buffered projection regression
passed; all 15 classifier tests passed with native-backend. Full server library:
404 passed / 2 inherited ignores. Candidate tests check repeated/replaced fields,
unchanged publication baselines, copy-on-write checkpoint restoration, Unicode,
required-field and budget failures, defaults and injected unwind recovery.

Strict native-engine all-target and server all-target Clippy passed on the final
candidate-preview implementation. The HTTP reasoning+JSON admission guard remains
in place until candidate token decoding and JSON recognizer composition are wired
and verified; no compatibility ledger row is closed by this prerequisite alone.

Byte-candidate prerequisite: response classification now accepts strict UTF-8
byte pieces through `feed_bytes` and `preview_bytes`; text entry points share the
same ingestion path. A typed zero/one/two/three-byte tail retains incomplete
scalars between pieces. Pending bytes count against the existing raw buffer
budget. Invalid sequences preserve the original UTF-8 error and name their raw
offset; a rejected piece appends none of its valid prefix. Final projection
requires a complete scalar. Candidate checkpoints also restore pending bytes.

Four new cases cover every byte split of Unicode reasoning and a regex closer,
three-byte pending tails and budget limits, invalid/overlong/surrogate/out-of-range
encodings with unchanged state, final incomplete sequences, and real ByteFallback
vocabulary pieces forming a Chinese channel delimiter plus JSON text. All four
passed under default features; all 19 classifier cases passed with native-backend.
The channel-to-JSON sampling composition and contextual decoder agreement remain
open; accepting byte pieces alone does not enable reasoning+JSON requests.

Final byte-input validation: full server library 404 passed / 2 inherited ignores;
strict native-engine all-target and server all-target Clippy passed. Existing
text projection, normal prefill, explicit continuation and stop behavior remain
covered by that server run. Reasoning+JSON admission remains unchanged.

Selected-field projection now borrows region slices and joins lazily from the
shared classifier. Ordinary snapshots consume this same iterator; internal
candidate observers can read content without copying unrelated reasoning or
opaque fields. Their callback is borrow-scoped and errors restore parser state.
Tests compose projected content with the actual schema recognizer, proving first
boundary handling and rejection of a later valid JSON object after earlier bad
content. Raw opening markers inside serialized JSON strings are classified before
schema recognition; an escaped opener retains its JSON data meaning instead.

That composition exposed a real schema-prefill gap: llguidance's string lexer
rejects non-control Unicode escapes and escaped soliduses in existing input.
Completed string escapes are now canonicalized only in the recognizer prefix,
including paired UTF-16 surrogates and escaped keys. Literal escaped backslashes
remain literal; invalid complete surrogate escapes retain a typed prefix error
and causal decode detail. Regex/Lark inputs bypass this JSON-only conversion.
The original prompt, counts and response prefix are not rewritten. Partially
written escapes and generation-side escape expansion remain open work.

Validation so far: 2 schema-prefix tests and all 21 classifier cases passed under
default/native features. The real ONNX continuation case now checks escaped keys
and a Unicode/surrogate/solidus value in buffered and logprob SSE requests, exact
raw template suffixes and prompt counts; full server 404 passed / 2 inherited
ignores. Reasoning+JSON token sampling itself remains guarded and incomplete.

Strict native-engine and server all-target Clippy passed. The final literal
backslash and leading-whitespace diagnostic refinements reran both schema-prefix
cases and strict native-engine all-target Clippy successfully. Unicode errors
refer to the original prefix byte offset, including its leading JSON whitespace.

The final escape self-review also preserves ASCII control quoting required by
the backend string-value lexer, including DEL which ordinary JSON serialization
can leave literal. NUL/newline/control/DEL values and an escaped newline property
name now have actual schema next-token checks. All 3 selected schema-prefill
cases and strict native-engine all-target Clippy passed after this refinement.

Reusable JSON byte recognition: `GrammarByteCursor` compiles a schema once against
a byte vocabulary independent of model token IDs. Candidate preview rolls back
accepted bytes after both viable and rejected trials; atomic commit rejects an
invalid suffix without retaining its accepted prefix. Status distinguishes viable
partial content, complete grammar EOF and rejection at a logical byte offset.
Its explicit byte budget is checked before mutating recognizer state. User stops
and length limits remain independent of grammar completeness.

Backend failures reconstruct the committed logical prefix from the initial
compiled state and retain their causal errors. If recovery fails, the cursor
holds both failures and refuses reuse until an explicit reset. Tests exercise
normal/rejected/terminal trials, split UTF-8, invalid bytes, resource/schema errors,
backend recovery and sticky failed recovery. The channel-composition test now
uses one compiled cursor for all candidates, after the same field projection and
JSON prefix normalization, instead of compiling a grammar for every candidate.

Validation: all 5 cursor cases passed under default and native-backend features;
all 21 shared classifier cases passed with the compiled-cursor composition.
Strict native-engine all-target and server all-target Clippy passed. Server
runtime behavior was not changed or remeasured in this milestone. The HTTP joint
reasoning+JSON guard remains; model-token decoding, pending-boundary viability
and actual processor integration still require completion.

Canonical candidate-decoder ownership: tokenizer handles can now be cloned for
an owned request-local sampler, and the incremental text decoder can fork its
existing lookbehind and special-token policy. Candidate completion, alternate
continuations and failed terminal materialization leave the live decoder intact.
The same upstream incremental decoder remains the sole text implementation;
there is no isolated-token concatenation substitute.

A shared decode-loop regression pins the terminal contract: selected EOS IDs
are committed before stop decisions, remain in result history, and are decoded
according to the declared special-token registration. Skip and preserve policies
are distinct from the stop_on_eos control. Template projection sees the actual
preserved spelling, so future JSON EOS checks must project that spelling through
the declaration rather than silently dropping a token because of its role.

Validation: all 8 native tokenization cases and the terminal-contract case passed;
new fork cases also passed under default features. Both ORT tokenizer integration
cases and the request-local special-token policy case passed. Strict native-engine,
ORT and server all-target Clippy passed. No HTTP joint-constraint behavior was
enabled by these ownership and contract changes; processor integration remains
open, including partial-byte/framing viability and decoder context agreement.

### Channel-aware JSON sampling through the production processor chain

`GenerateOptions.response_projection` composes the model tokenizer, declared response framing and a
compiled JSON Schema cursor in the mandatory generation processor chain. The
actual rendered assistant prefix primes all three; JSON validates only the
selected content field, so reasoning can continue before the content boundary.
The server lowers JSON response formats through this processor and validates
its capabilities before admitting generation or writing SSE headers. Explicit
reasoning continuation no longer rejects these supported combinations.

Candidates preserve request-local decoder state, strict split UTF-8 bytes,
actual EOS spelling and the configured termination policy. Pending literal
boundaries use the publication parser's classifier, including overlapping
short/long delimiters. Active JSON string whitespace remains data even when
publication trims field ends. Reopened fields and streaming share one typed
append-only projection rule. Exact history replacement replays the original
prefill; failed candidates restore parser and grammar state, and recovery errors
retain both causes. Byte cursor representation replacement reuses the compiled
schema and preserves the old prefix on failure.

Scope remains explicit: root ByteLevel/ByteFallback and canonical non-byte
decoders are supported; byte-decoder sequences and regex field boundaries still
require further implementation. Incomplete Unicode escape spelling remains an
open lexical compatibility item. These gaps do not close the delivery ledger
or authorize merging this draft.

Validation: seven production-processor cases cover model-token masks, history
rewind/replacement, split Unicode data/framing, overlapping delimiters,
provisional string whitespace and candidate budgets. Six byte-cursor cases
include atomic representation replacement. All 21 response classifier cases
pass. Full default engine: 712 passed / 1 inherited ignore. Full server:
404 passed / 2 inherited ignores. The expanded real ONNX continuation matrix
covers explicit/automatic reasoning resume, JSON/Schema/unconstrained output,
buffered/SSE and logprobs; a higher-scoring invalid schema value is masked.
Regex capability rejection is verified as HTTP 400 before SSE. Strict default
engine/server all-target Clippy passes.
Native-feature validation also passes all seven processor and six cursor cases,
plus strict native-engine all-target Clippy.

### Dependency refresh for integration

Upgrade llguidance, toktrie and toktrie_hf_tokenizers to 1.9.1 and workspace
runtime tokenizers to 0.23.2. The published grammar adapter still requires
0.21 internally; use its public tokenizer JSON interface rather than mixing
Rust types from different versions. Byte vocabulary and grammar construction
share this adapter boundary. Incremental decoding passes the new token-list
argument while retaining the existing request-local state and token policy.

After the upgrade, complete library runs pass: engine 894 / 1 inherited ignore
(with native features unified by the server tests), ORT 188, server 404 /
2 inherited ignores. This includes the Unicode, EOS, response projection,
constrained continuation, cancellation and HTTP protocol regressions.
Strict engine/ORT/server all-target Clippy also passes after the dependency
refresh. Final tokenizer unit and integration regressions pass after preserving
the serialization error cause. Windows coverage exposed an unrelated MLAS build
warning: the AVX2 panel-tail helper was compiled when its sole caller was disabled.
Give the helper the same `not(feature = "mlas")` condition as its caller.

Final CI fixture corrections: migrate the maintained reranking textproto to
IR 11 / opset 24 and supply ReduceSum axes as an int64 tensor input. The fixture
inventory/schema guard passes all 10 cases; both real-engine ranking cases and
all three selected HTTP/config ranking cases pass. The compressed-state public
API test now derives a future version from the declared maximum instead of
misclassifying the supported v1.9 as future; all four public-API cases and all
24 metadata schema-version cases pass. These resolve the Linux Fast/coverage and
ORT-backed CI failures without changing runtime ranking or version semantics.
The complete HTTP integration suite also passes all 37 cases after updating
the suffix-without-FIM refusal assertion to the shared, metadata-driven admission
message. The test verifies HTTP 400, invalid_request_error, the rejected suffix
field and the remedy; no runtime behavior or refusal is weakened.
Windows CI additionally exposed newline-sensitive test construction in the
content-dependent KV cancellation fixture. Build that variant by parsing and
editing ONNX nodes, then canonically printing the graph, instead of matching
formatted source strings. CRLF and compact-layout equivalence and the original
real KV rollback oracle both pass; strict native-engine library/tests Clippy
passes. Production inference behavior is unchanged. Both required Linux checks
passed on the preceding head; the Windows test correction is validated locally
before restoring automatic integration on the updated head.


## Follow-up: byte-prefix regex response boundaries

JSON-channel sampling now accepts boundaries with resolved byte DFAs. The
publication parser and candidate sampler share byte-prefix reachability,
including pending UTF-8 bytes. EOF and look assertions see those bytes before
any scalar is published. A delayed accepting state does not by itself keep a
shorter delimiter open; possible later acceptance and EOF-dependent matches do.
Regex alternative priority and greedy/lazy behavior remain those of the compiled
expression. Look-behind roots and reverse acceptance reachability are computed
once with explicit analysis allocation bounds. Unsupported DFA constructions
retain their backend cause in a typed capability refusal; ordinary buffered
parsing remains available for those expressions.

Validation so far: all 23 original/expanded classifier cases and seven processor
cases passed; full native-enabled engine 897 passed / one inherited ignore and
full server 404 passed / two inherited ignores. Real ONNX HTTP generation covers
regex-framed reasoning+Schema in buffered and logprob SSE paths, masking the
higher-scoring invalid value. Final 25 classifier cases, seven processor cases and strict engine/server
all-target Clippy pass. Literal boundaries use a distinct resolved matcher and do
not construct unused DFA analyses. Unicode word assertions, decoder sequences,
escaped JSON generation and all other unchecked delivery items remain open.

## Projected JSON Unicode escape continuation

Projected schema recognition now treats unfinished JSON string escapes as a
bounded domain of possible Unicode scalars. Each domain is checked against the
compiled byte grammar, including `const`, known property names, patterns and
scalar string-length limits. UTF-8 prefix intervals prune impossible scalar
ranges; escaped ASCII uses individual leaves because its byte order differs.
Surrogate halves must pair correctly. Pending escapes never permit EOF, and a
budget rejection in one search branch does not abort other candidates.

A shared normalizer replaces the old completed-prefill-only helper. Response
sampling, prefill and history replay consume its stable canonical spelling;
actual prompt/output bytes keep their original spelling. Malformed candidate
escapes are masked with typed, byte-offset errors rather than aborting unrelated
candidates. Existing plain token-trie prefills share completed normalization.
Plain token-trie generated escapes remain open work, as do escaped DEL property
names declared by a schema. Raw literal property names preserve existing behavior;
no key-versus-value spelling heuristic is used.

Validation before the final raw-key preservation adjustment: full default engine
722 passed / 1 inherited ignore; full server 405 passed / 2 inherited ignores.
The real ONNX HTTP test covers scalar constants, known keys and surrogate pairs
in buffered and logprob SSE requests with a higher-biased invalid suffix. After the final adjustment, all 5 Unicode prefix cases, 3 existing schema-prefill
cases, 8 production processor cases and strict engine/server all-target Clippy
passed. This increment does not close the unchecked delivery items.


## Shared ordinary and projected JSON recognition

Ordinary JSON, JSON Schema and declared response-content constraints now use one
schema cursor and the same Unicode escape semantics. Plain output follows normal
special-token skipping, while declared channel projection retains protocol markers.
EOS roles never erase ordinary token bytes: an EOS quote can finish a string,
and the same quote is rejected after the value has already finished. Explicit
stops and output/context limits retain their existing partial-value behavior.

A transactional plain projection retains split UTF-8 and restores rejected,
failed or panicking inspections. The old character-only generation constraint
was removed, with no alias. Its pure syntax preflight is now the
`validate_json_prefix` function. One positive `GenerateOptions.max_constraint_bytes`
budget replaces the response-only duplicate; HTTP admission supplies the actual
configured response buffer limit. All constrained chat/native/legacy generation requests
validate capability and prefix before execution or SSE headers. Prefix errors
retain their typed request classification and causal details.

For byte-concatenating plain vocabularies, one external model trie walks the
already compiled byte grammar. No second schema is compiled for the fast path.
Backend mask caches are invalidated across vocabulary identities; raw byte IDs
cannot become model IDs. Escape spellings and contextual non-byte codecs still
use semantic inspection. Pending-escape syntax is checked locally before scanning
unrelated committed content. Backend access remains panic-fenced, with failure
state encoded separately from a usable parser; committed-prefix recovery retains
its source chain.

Validation on the pre-integration baseline: 405 server library tests passed with
2 inherited ignores, and 728 engine tests passed with 1 inherited ignore. Strict
engine/server all-target Clippy passed. New tests compare trie and semantic masks
through rewinds, escapes and malformed/split UTF-8; alternate vocabularies include
model IDs beyond the byte vocabulary and preserve byte-mask state after failures.
The real ONNX buffered/logprob-SSE matrix now runs both with and without a declared
response template. An additional standalone CLI test build exhausted disk space
before producing a result; no CLI test pass is claimed. After integrating main 332396bcf and its runtime/memory changes, fresh full server
405 passed / 2 inherited ignores, full engine 728 passed / 1 inherited ignore,
and strict engine/server all-target Clippy all passed. Other unchecked delivery items remain open.


The shared path also resolves composed byte decoders structurally. Token-local
Replace/Strip stages compile into model pieces before the byte stage; ByteLevel
is fused by definition, while ByteFallback requires explicit Fuse before initial
stream stripping. Ordered initial Strip stages retain partial Unicode characters
and clone with candidate state. Vocabulary trie masks become available once
these stages have resolved, so neither grammar validation nor the fast mask
assumes an unstripped candidate spelling. Registered special tokens are filtered
before transforms only under the normal Skip policy. Declared response projection
preserves their actual transformed text. Unsupported contextual stages report
the exact tokenizer.decoder metadata path instead of silently changing semantics.
Vocabulary predecoding and incremental decoder boundaries now fence backend
panics and preserve actionable token/configuration context and source chains.

The composed-decoder review added 39 passing JSON-filter tests and 4 passing
byte-stream contract tests, including plain/declared split-Unicode continuation,
ordered Unicode prefix stripping, rewind, special-marker preservation and actual
tokenizer panic containment. The full suite on main 332396bcf plus these changes
passed 734 engine tests (1 inherited ignore) and 405 server tests (2 inherited
ignores). These are functional checks, with no performance claim.


## Declared reasoning budgets and grammar prefill

Reasoning budgets lower into a mandatory engine processor shared by host sampling
paths. A resolved token protocol validates open/end/forced IDs, supports finite
or unlimited windows, tracks natural closes, re-arms on repeated blocks and
finishes split Unicode before forcing. Generated-history checkpoints preserve
manual-control revisions and forced-token position through rewind and independent
sampler branches. Existing prefill is charged but never consumes an ungenerated
forced close. Forced IDs cannot resurrect a banned or grammar-rejected logit.

The server resolves chat protocols from declared response boundaries and the
final assistant anchor, with authored closer order retained separately from
longest-match scanning. Native/legacy callers can supply an explicit protocol.
Ordinary models retain compatibility with the upstream UI's unconditional
optional arming hint, while explicit finite protocols fail actionably when they
cannot be represented. JSON/Schema/Regex/Lark share generation_prompt prefill;
the Regex Rust API now carries pattern and prefix with all callers updated.
Anthropic enabled thinking budgets lower to the same request controls.

Real ONNX buffered/logprob-SSE requests prove forced close followed by answer
publication for zero/one-token budgets; native and legacy wire routes preserve
raw protocol text. Actual Native-backend inference also passes the forced-close
contract. The full engine suite passed 932 tests (1 inherited ignore), and the
full server suite passed 432 tests (2 inherited ignores). Strict engine/server
all-target Clippy passed. These are functional checks without performance claims.
Live chat and legacy completions own bounded registrations until their consumer
ends, with replay producers retaining control independently of reader reconnects.
128-bit operating-system-random completion IDs prevent timestamp collisions.
POST /v1/chat/completions/control validates the action and signals the same resolved engine budget;
unknown/finished and unarmed IDs return success:false. Native budget requests do
not yet publish IDs for manual control. The final server suite passed 436 tests (2 inherited ignores), including HTTP
control reaching real ONNX inference, capacity/completed-ID coverage and failure/
disconnect cleanup. Strict server all-target Clippy passed. The manual-signal
inference test signals before submission to avoid racing the tiny fixture; it
does not claim an end-to-end midstream timing test.
Other unchecked delivery items remain required.


## Projected literal grammars and tool observation

One resolved ResponseProjection now owns the template, actual assistant prefill
and response buffer bound for both grammar constraints and transactional tool
observation. Json/JsonSchema/Regex/Lark retain their respective languages;
response-only constraint variants are removed from the pre-release Rust API.
Constraint prefixes must be empty when the projection owns prefill. Plain
constraints retain their own prefix. The shared byte cursor is GrammarByteCursor,
and max_constraint_bytes applies to raw/logical constrained text.

Literal text grammars use the same transactional response projection as JSON
without canonicalizing their escape spellings. Compiled grammar inspection
rejects model-token references, token-based limits, rule temperatures and hidden
stops rather than silently interpreting model token IDs as byte IDs. Unprojected
model-token Lark remains token-aware. The subsequent raw GBNF increment is described below.

Tool observation consumes only the declared content channel, including tool
prefill. Actual ONNX regression first demonstrated HTTP 500 from a reasoning
closer being interpreted as text before a tool envelope; it now passes for
forced Specific tool requests and Auto requests without a grammar. Tool examples
inside reasoning are retained as reasoning and never become committed calls.
Buffered tool turns retain reasoning_content, matching live SSE behavior.
Malformed assistant tool prefill is a typed client error before generation and
SSE admission, with continuation/chat-template repair context.

Resolved byte codecs retain split UTF-8 and special channel markers. Contextual
byte decoders without that streaming contract defer observation until terminal
canonical decoding, with the reason retained in Debug state. Checkpoints and
aborts restore raw bytes, input representation and projected parser history.
Backend decoder panics preserve model-token and tokenizer.decoder context.

On main 01632f548 plus the compatibility increments, the complete engine suite
passed 940 tests (1 inherited ignore), server passed 437 (2 inherited ignores),
and HTTP integration passed all 37 tests. Explicitly enabling the server's
native-backend feature passed 446 server tests (2 inherited ignores), including
the actual Native reasoning-budget and projected-tool cases. Strict engine/server
all-target Clippy passed for the default feature combination and server Clippy
also passed with native-backend enabled. The final tool regression explicitly
selects ORT or Native, covering forced/Auto tools, buffered/SSE/logprob delivery
and reasoning exclusion. These are functional checks without performance claims.
The broader unchecked delivery items remain required.


## Raw GBNF generation constraints

Add an explicit `GenerateConstraint::Gbnf` without a prefix field: user grammars
operate on new raw model output, while internal forced-tool Lark remains scoped
to projected content. This distinction follows the pinned upstream
[grammar parser](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/src/llama-grammar.cpp)
and [sampling initialization](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/common/sampling.cpp).
A bounded AST lowers Unicode classes, literals, recursion, groups, repetitions
and model-token references into model-aware Lark. Numeric IDs must exist in the
loaded vocabulary; named references use actual model tokenization. A resolved
ByteStreamContract proves actual decoder bytes and dense IDs. Raw special-token
pieces remain available to character literals, matching the pinned upstream
[vocabulary cache](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/src/llama-vocab.cpp).
Only explicit ending roles get EOS frames; adapter name-based role guesses and
decoded-token fallback are excluded. Resolved stream-initial and stream-trailing Strip stages affect published text
but do not alter the raw vocabulary used by GBNF, raw Lark, or lazy JSON.
Unresolved contextual decoder transforms remain unsupported. Lazy JSON uses
the same raw identity codec throughout trigger replay and subsequent tokens;
ordinary projected-content constraints retain publication decoding semantics.
JSON normalization also ignores trailing JSON whitespace after complete lexemes,
while retaining string-value spaces and rejecting whitespace-terminated unfinished
numbers/keywords. The raw response-byte budget remains enforced before logical
normalization; neither whitespace nor escape normalization bypasses that budget.

Validation for resolved Strip/raw constraints: full engine library 1111 passed
(1 inherited ignore); full server library 615 passed (2 inherited ignores).
The added real wire test passed all 48 Native/ORT × native/legacy/chat ×
buffered/SSE × GBNF/raw-Lark/lazy-GBNF/lazy-JSON combinations. Strict all-target
engine/server Clippy passed with defaults; strict server Clippy also passed with
`--no-default-features --features metrics,cuda-13000`. Formatting and diff checks
passed. Server network tests ran with local loopback access after all 37 failures
in the restricted run were identified as port permission errors. These are CPU
functional checks; no GPU or performance claim is made.

Adversarial mask tests exposed two real semantic differences: negated token
ranges could allow EOS before the root completed, and llguidance canonical
forcing could exclude a legal compound token. The GBNF path now proves root
completion for all ending roles and disables canonical forcing. Literal `{0}`
is lowered to epsilon because the target Lark parser rejects a zero upper bound.
History rewind/recovery, nullable recursion, expanded repetition, Unicode scalar
limits and incomplete UTF-8 pieces have executable coverage.

HTTP preparation rejects conflicting JSON/forced-tool/ignore-EOS controls and
validates syntax, model token references and decoder capability before SSE.
Shared public request builders apply the same grammar validation. Neutral lazy,
trigger and preserved-token fields are accepted; active forms are rejected.
Native `n_probs` was confirmed as a separate missing protocol feature and remains
required by the overall compatibility goal. Raw `%llguidance` syntax and active
lazy grammars are also still required follow-ups.

Final validation: engine libraries 948 passed / 1 inherited ignore; server
libraries 449 passed / 2 inherited ignores; HTTP integration 37 passed. Strict
engine/server all-target Clippy passed. Explicit Native CPU and ORT tests cover
all three completion endpoints, buffered/logprob SSE, reasoning-budget interaction,
neutral grammar controls, conflicting fields and malformed input before SSE.
This increment does not close the goal.


### Native-default CLI portability

The profiling subprocess test now selects ORT explicitly for its CPU reference
run and independently checks that an ordinary command records Native as its
actual backend. CI previously exposed a Windows process-main stack overflow
once Native became the default. The shared CLI entry point now executes on a
named standard Rust worker, including calls from Python. It honors Rust's
standard worker-stack configuration, preserves panic payloads, and reports
thread startup failures with recovery advice.

Native is the default computation backend. The workflow tensor bridge and the
optional hybrid MTP head still use ORT APIs; eliminating those runtime library
dependencies is separate unfinished work, tracked by a local missing-ORT
subprocess regression. The Python package must retain its runtime dependency
until that regression passes. This change does not claim ORT-free deployment.

Final portability validation: 71 CLI unit tests and both real profiling
subprocesses passed. A separate command with a 1 MiB process-main stack completed
Native generation and recorded `decode_backend=native` and
`execution_provider=native`. Strict all-target engine/server/CLI Clippy passed
with `-D warnings`; formatting and diff checks passed. The integrated parent
runtime had already passed engine 948 / 1 inherited ignore, server 449 / 2
inherited ignores, and HTTP 37. The incidental host-string lint repairs passed
11 CPU text-kernel tests and 6 session text integration tests / 1 inherited
benchmark ignore. Windows CI remains the platform-specific confirmation.

### Raw llguidance protocol follow-up

The pinned upstream sampling path dispatches a `grammar` beginning with
`%llguidance` directly to Lark. The server now performs the same prefix dispatch
on chat, legacy and native completion requests. `GenerateConstraint::RawLark`
distinguishes this raw newly generated stream from the existing content/prefill
Lark constraint. It shares the resolved byte vocabulary and completion-only EOS
contract with GBNF, while preserving authored llguidance options. Source budgets
are enforced for public Rust callers as well as HTTP; parser failures occur
during request preparation before SSE.

Validation: engine 948 passed / 1 inherited ignore; server 449 passed / 2
inherited ignores; HTTP 37 passed; strict engine/server all-target Clippy passed.
A subsequent targeted run selected and passed the Unicode/byte-budget test
after its final budget assertion was added. Coverage includes actual Unicode byte fallback and
compound-token choices, model token IDs, all three buffered/SSE endpoints on
Native CPU and explicit ORT, invalid Lark syntax and out-of-range IDs. Active
lazy grammars, preserved tokens and native probability reporting remain open.


### Native probability protocol follow-up

The pinned upstream server-context `populate_token_probs` distinguishes original
model probabilities from processed sampler candidates. Native requests now parse
`n_probs` and `post_sampling_probs` explicitly. `LogprobSource` selects the
distribution, and an opt-in snapshot preserves model logits before mutation in
shared decoding, continuous rows, speculative acceptance and bonus verification.
No snapshot allocation occurs when reporting is disabled or processed logits
are requested. Existing probability callers retain the processed-logit default. Native post-sampling
reporting uses the effective sampler distribution, including greedy point mass.
A foreign final sampler must request model or processed-logit scores because its
internal distribution is not inspectable.

Committed `GenerateToken` events carry their matching optional `TokenLogprob`;
transactional publication also retains it. Native buffered and SSE outputs
serialize IDs, actual byte pieces, finite scores and top-N alternatives. Partial
UTF-8 is retained in bytes and truncated in token text, zero processed mass is
excluded, and buffered stop-word probabilities are omitted using tokenization
without automatic special tokens, matching the pinned upstream contract.

Final implementation validation passed engine 950 / 1 inherited ignore; server
450 / 2 inherited ignores; HTTP 37; CLI 71. This includes continuous-row
metadata, UTF-8/zero-score, stop-word and greedy point-mass coverage. Full strict
engine/server/CLI Clippy passed with -D warnings, including the final zero-mass
and floating-point-underflow encoding assertions. Lazy activation,
preserved tokens, slot/adapter persistence, model management/audio and the full
acceptance matrix remain open; neither this increment nor a parent PR merge
completes the overall goal.


### Preserved-token prerequisite for lazy activation

The pinned `common_sampling_add_grammar_triggers` requires a single-token word
trigger to appear in `preserved_tokens`. That option also affects rendered
special-token output; accepting it only as a grammar flag would lose real
semantics. This increment resolves the spellings against actual model vocabulary
and carries an immutable selection through live decoding and result publication.

`TokenTextDecoder` filters unselected special IDs before its request-local stream
decoder. Retained IDs are decoded together, preserving codec lookbehind without
changing shared tokenizer flags. Final text and per-token replay use the same
selection, including ordinary/native speculative verification, continuous rows,
candidate-tree and DFlash publication. Results retain the resolved policy so
protocol logprob chunks cannot reconstruct text using a different policy.

A projection binds model-declared framing symbols to its tokenizer and unions
those IDs with request selections. Literal boundaries resolve canonically;
regex fragments can declare `response_template.preserved_tokens` explicitly.
The chat channel gate consumes the engine's filtered text instead of decoding
token IDs through an independent special-token policy. Buffered projection uses
the committed result text. This keeps undeclared special tokens out of content
without dropping the model's declared reasoning and tool frames.

Validation passed: engine 952 / 1 inherited ignore, server 450 / 2 inherited
ignores, HTTP 37 and CLI 71. Coverage includes ByteLevel/ByteFallback Unicode,
decoder forks, invalid engine IDs, continuous-result policy and actual Native
CPU/ORT requests across three endpoints with buffered/SSE and probability output.
Self-review also fenced malformed framing decoder panics with actionable errors;
the full engine suite passed again after that regression was added. Strict
all-target engine/server/CLI Clippy passed on the final source. The Native
benchmark API consumer compiled, without running a benchmark. Formatting and
diff checks passed. Active lazy activation and triggers remain required
follow-ups; the broader goal is open.

### Deferred reference-memory descriptions

`MemoryInfo` now holds an immutable placement description. Construction,
inspection, cloning and dropping an unresolved description do not load ORT.
Only explicit reference-adapter resolution creates a handle, and successful
resolution is cached. The ready owner retains both the handle and description
independently of the original description's lifetime. Resolution errors retain
the original typed cause and name the requested device and memory contract.

ORT allocator callbacks hold the resolved type, so their `Info` callback neither
loads a library nor performs fallible initialization. CPU arena/device kinds,
GPU descriptions and host-access checks retain their declared semantics.
Placement fields and an allocator's description cannot be replaced separately
from its actual resource. All Rust callers use the new immutable accessors.

The Native workflow tensor container and optional hybrid MTP executor still use
ORT APIs. This increment removes eager metadata creation as a prerequisite for
porting those owners; it does not claim ORT-free Native model execution or
remove the Python runtime dependency.

Validation passed: ORT library 190; engine 952 / 1 inherited ignore; server
450 / 2 inherited ignores; HTTP 37; CLI 71 and both profiling subprocesses.
The final CUDA-feature descriptor run selected and passed both tests; its nested
missing-library subprocess also proves exactly one test was selected. These
checks exercise descriptions and CPU reference handles, not GPU execution.
Strict ORT/engine/server/CLI all-target Clippy with the ORT CUDA feature passed;
the final test changes also passed CUDA-feature ORT all-target Clippy. Both
packed-gather IR tests passed. Formatting and diff checks passed. The feature
restacked unchanged onto current main and the full functional graph passed
again. A private versioned-inference-row name addresses an inherited strict
lint without changing registration behavior. Per owner instruction this work
does not wait for blocked remote CI.

### Lazy grammar activation in progress

The raw grammar engine now has an explicit resolved-token activation entry
point. Before a trigger is accepted it leaves valid vocabulary candidates,
including endings, unconstrained. The trigger itself starts the constrained
history, retaining its model token identity rather than converting it to text.
Activation is derived from each supplied history, so replacement and speculative
rollback can return to an untriggered state. Invalid committed suffixes remain
errors and the existing parser reset permits recovery with a valid history.

This is an engine prerequisite, not completed server support. Server admission
still rejects active lazy requests until all trigger modes and their tokenizer
resolution are integrated. The pinned upstream
[`common/sampling.cpp`](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/common/sampling.cpp)
resolves a single-token word into a preserved-token trigger; multi-token words
remain literal patterns. Token triggers take precedence over patterns in each
accepted token. Pattern order is configuration order, and replay starts at the
first nonempty capture, or the full match when none is nonempty. As shown in
[`src/llama-grammar.cpp`](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/src/llama-grammar.cpp),
a replay start can fall inside a token: the first piece is truncated while its
original token identity survives. A synthetic text prefix alone cannot establish
that contract for model-token references. Pattern matching also uses byte-based
ECMAScript regex semantics, which must not silently become Unicode Rust regex
semantics. The upstream raw `%llguidance` branch constructs its sampler before
the GBNF lazy branch; its behavior needs separate conformance coverage.

Four selected activation tests passed, including model-token references, root
completion, rollback/replacement, invalid-history recovery and trigger vocabulary
validation. The full engine library regression passed: 956 tests and one
inherited ignore. Strict engine all-target Clippy and formatting passed.
Word/pattern activation, partial-token replay, reasoning
suppression, server mapping and real HTTP conformance remain required.

### Token-aware GBNF replay prerequisite

A raw replay cursor now compiles the existing bounded GBNF syntax tree into
explicit scalar/token states. Token-reference branches inspect the original
model token ID even when a trigger truncates its piece, including a truncation
inside UTF-8. Character branches instead consume the supplied bytes. Ambiguous
text and token branches survive independently, and cursor clones provide
independent rollback checkpoints. Character classes, finite/unbounded repetition
and consuming recursion use the existing parser's validation. Pending Unicode
prefixes are checked against the next scalar ranges before being admitted.

Expansion checks bound queued and visited stack cells before queue insertion;
accepted replay has a separate explicit byte budget. A rejected piece or budget
error leaves the previous cursor unchanged. This provides the missing replay
capability without mutating llguidance's private token history or replacing a
model-token reference with synthetic text. It is still a prerequisite: the
server's active lazy admission remains closed until trigger matching, vocabulary
resolution and request/runtime integration are complete.

Nine selected tests passed. Besides the truncated-token and Unicode cases, they
cover ambiguity, independent clones, rejection recovery, finite repetition, byte
budget failures and bounded ambiguous recursive expansion. A cross-check walks
accepted histories and compares each next-token decision with the current
llguidance backend for five grammars covering character classes, recursion,
repetition, positive token references and negated token references. This is
regression evidence for those complete-token cases, not a claim of full upstream
or HTTP conformance.

The final engine regression passed 965 tests / one inherited ignore. Strict
engine all-target Clippy, formatting and diff checks passed on this replay
implementation. No active server lazy behavior is claimed by those checks.

### Byte-oriented trigger pattern matching in progress

The trigger matcher now explicitly translates the admitted C-locale
`std::regex<char>` ECMAScript syntax into the bounded `fancy-regex` backend
already present in the dependency graph. Each original input byte is represented
by one Latin-1 scalar; captures are converted back to original byte offsets.
ASCII word boundaries and character classes do not inherit Unicode semantics.
Dot excludes both CR and LF. C-locale named character classes, empty/negated
empty classes, control escapes and byte-narrowed Unicode escapes have explicit
translations. Backend-only group extensions and references preceding their
captures are rejected. Unmatched/open capture references retain the upstream
nonmatching behavior.

The matcher selects the first nonempty captured group, falling back to the
whole match. Captures inside lookahead can therefore move the replay start
beyond the whole match. A pattern with leading `^` and trailing `$` uses actual
complete matching with alternative backtracking, rather than post-filtering
one shorter search result. A standalone harness running the pinned upstream
`llama_grammar_trigger_pattern::find` supplied the capture and invalid-syntax
expectations used by the Rust tests. The harness also exposed JavaScript
backreference/extension differences in a separate candidate library; that
library was not added to the repository.

Compilation limits include 128 group nesting levels, 1024 captures, repetition
bounds of 2000 and a one-MiB translated-program budget. Matching has a one-million
backtrack limit and the explicit input byte budget. Exhaustion is an actionable
error, not a fabricated nonmatch. Seven selected matcher tests passed, including
resource rejection and a catastrophically backtracking expression. The full
engine regression passed 972 tests / one inherited ignore.
Final strict engine all-target Clippy, formatting and diff checks passed.
Character classes separately cap their collected ranges at 65536 items.

This matcher is not yet connected to active lazy server admission. Production
trigger resolution, token-aware constraint ownership, reasoning suppression and
HTTP conformance remain open. C++ collating/equivalence bracket forms also still
need explicit conformance and implementation; the current matcher does not
establish complete coverage of every platform-specific std::regex extension.

### Runtime lazy-GBNF integration — self-review in progress

Typed nonempty `GrammarActivation` and four `GrammarTrigger` variants now lower
through the actual processor chain and HTTP request builders. The runtime
preserves trigger priority and authored pattern order, uses model-backed word
resolution, replays truncated token pieces and reconstructs activation after
rollback. Combined source/program/history budgets are enforced. Errors retain
the original `grammar_triggers[index]` even after token triggers are separated
from patterns. The earlier token-only llguidance activation path was removed;
the unified runtime owns activation.

A protocol-only reasoning cursor reuses the existing reasoning machine. It
hides accepted thinking tokens from the lazy grammar, suppresses its mask while
reasoning is active, and replays the complete matched ending sequence. The
matcher retains its original sequences to recover the longest matched closer.
Forced-history validation remains with the existing budget processor; the
grammar cursor does not reinterpret old accepted tokens with the current live
control revision. Unit coverage includes manual control, finite budgets, partial
closers, existing prefill, longest endings and speculative rollback.

Positive validation passed before the additional adversarial matcher case:
engine 978 / one inherited ignore, server 451 / two inherited ignores, HTTP 37,
CLI 71 and both profile subprocesses. Strict engine/server/CLI all-target Clippy
with the ORT CUDA feature and formatting passed. Real Native and ORT HTTP tests
cover all four trigger tags on five route spellings, buffered and SSE output,
pre-SSE invalid-field rejection, reasoning-budget-zero closure and the pinned
raw-Lark behavior. These are functional CPU tests, not performance/GPU claims.

The change is not ready to merge. Independent upstream std::regex comparison
found that `x(?:(a)|b)+` on `xab` must clear its capture in the final repetition
and return byte offset 0; the current fancy-regex backend returns 1. Finite and
unbounded repetition and a backreference variant reproduce the same semantic
gap. The formal capture-oracle regression now selects one test and fails on
this mismatch. Correct repetition capture clearing is required before landing;
passing ordinary HTTP cases does not establish that compatibility.

### Byte-native trigger VM replaces backend capture semantics

The independent capture-reset failure above is addressed by an owned byte VM,
rather than translating captures through a general regex backend. Repetition
clears the capture span of its body before each iteration; an unset
backreference fails. Lookahead owns an atomic choice stack and shares the
execution budget. Full matching backtracks before rejecting a partial match.
Byte classes, captures and anchors operate directly on the original byte buffer.
The engine no longer directly depends on fancy-regex for this feature.

Execution bounds cover shared instruction fuel and live choice/capture/operation
cells, including nested lookahead and backreference comparisons. Compilation
retains source, node, byte, capture, depth and repetition limits. A fixture from
the pinned upstream find() function and libc++ ECMAScript enumerates 24 patterns
against all 121 abx strings of length zero through four (2904 comparisons),
covering nullable repetition, greediness, capture reset, lookahead and backrefs.
These cases complement the original invalid syntax and resource-limit tests.

### Recovery after a truncated UTF-8 token reference

Self-review extended the production lazy-chain truncation case from a following
raw token reference to a following literal. The previous implementation selected
one test and failed: the accepted token reference's invalid byte suffix poisoned
all later text, leaving no finite candidate. The pinned upstream decoder stores
a negative remaining-byte count for that event, then decodes the next piece
fresh; a harness copied from that decoder verifies the subsequent ASCII `B`.

Replay now retains only a valid incomplete scalar between pieces. Invalid bytes
can satisfy an explicit token terminal without permanently invalidating later
scalar terminals. Current invalid scalar candidates still fail transactionally.
The replay test covers continuation bytes, invalid leading bytes and malformed
sequences, independent clones and recovery after rejecting a wrong next piece.
The production processor-chain test now covers both token and literal successors
when a byte pattern activates inside a multibyte model token.

### Lazy raw GBNF validation completed

The byte VM and UTF-8 recovery fixes passed the complete local functional graph
on the current-main restack: engine 979 / one inherited ignore, server 451 / two
inherited ignores, CLI 71, HTTP 37 and both CLI profile subprocesses. Strict
engine/server/CLI all-target Clippy with the ORT CUDA feature, formatting and diff
checks passed. The earlier capture/reset and truncated-byte failures above are
resolved, with their regression coverage retained. Execution fuel also charges
state/capture copying and sequence expansion, rather than just VM dispatch.

PR #2525 lands this raw-GBNF compatibility increment after local self-review;
the owner explicitly requested bypassing blocked remote CI. Lazy JSON, C++
collating/equivalence classes, remaining model/slot/adapter protocols and complete
Native Value/MTP isolation remain broader goal work. These CPU functional and
compile checks do not establish GPU execution or performance results.


### Neutral workflow tensor storage

Workflow `Value` owns typed fixed-size CPU allocations, aligned raw bytes,
external storage with immutable placement and retained owners, or checked alias
windows. Only an explicitly selected ORT session/binding resolves a reference
view, with fallible admission propagated through all pointer/borrowed-output
callers. The cached view releases before its backing owner. ORT-created outputs
retain an owned ready handle, and native-only destruction never opens ORT.
Mutable pointers are retained from allocation construction instead of deriving
write pointers from shared slice borrows. Alias metadata traversal is iterative
and retains every owner rather than dropping intermediate resource guards.

CPU empty tensors retain zero-initialized host storage; aligned zero-byte raw
allocations keep their existing real-allocation contract. An initial regression
caught an overly broad dangling-pointer shortcut and passed after restricting
it to zero-byte aliases/reference outputs. Metadata shape and byte arithmetic
is checked before allocation. CPU arena/device and external provider placement
remain preserved when a view is explicitly materialized.

Ordinary Native startup no longer creates the optional MTP environment.
The enabled metadata-declared legacy MTP proposer still owns its actual ORT
session/environment together; it is not bypassed or reported as Native-only.
The Python launcher/preload and package dependencies are unchanged in this
increment and remain open isolation work.

Initial verification passed 191 ORT/value tests, including a subprocess that
constructs, mutates, aliases, deep-copies and drops every host dtype with a
missing ORT library. External-owner lifetime, exact nested windows and recovery
after an explicit reference-view failure are covered. Real CLI subprocesses
passed both default Native generation with an unavailable ORT path and explicit
ORT reference generation. The broader engine/server/HTTP regression and strict
CUDA-feature checks are being completed before landing.

Neutral storage validation completed: the full storage-change graph passed
191 ORT/value, 979 engine / one inherited ignore, 451 server / two inherited
ignores and 71 CLI tests. The subsequent capacity/portable-shape checks passed
the 191-value suite again, including the isolated missing-library capacity
regression. Both CLI profile subprocesses and all 38 HTTP tests then passed,
including the independent missing-ORT buffered completion and successful SSE
finish/error assertions. Strict ORT/engine/server/CLI all-target Clippy with the
ORT CUDA feature, formatting and diff checks passed. The two implementation
commits restacked unchanged onto current main; both CLI subprocesses, HTTP 38
and strict checks passed again on that restack. PR #2529 proceeds after local
self-review without waiting for blocked CI, as explicitly requested. GPU
execution/performance and the Python/MTP isolation work remain unproven here.

### Native MTP execution isolation (#2531)

Metadata-declared MTP heads now use the selected Native graph executor instead
of creating an ORT environment/session. One backend-neutral runner owns draft
KV and HC recurrent state for both Native and explicit reference adapters.
Native head metadata comes from the admitted native session; each proposer
keeps independent cache state while head execution shares the session workspace.
Hidden/state projection explicitly materializes host values. KV output owners
remain on their execution device. Failed hidden/state validation leaves the
previous cache unchanged, and backend errors retain their typed causes.

CPU subprocesses with a nonexistent ORT library run attention, rank-4 HC and
growing-cache heads, plus actual Native MTP proposal/verification generation.
Three head steps compare hidden and recurrent state against ORT, growing caches
exercise rewind/replay/reset, and greedy generated tokens match ordinary Native
decode. Local storage (192), MTP reference integration (2) and engine (981 plus
one inherited ignore) suites passed on the restack onto main. Remaining local
surface and feature checks are recorded on the PR before merge. CUDA feature
compilation is not CUDA execution evidence. Accepted-prefix correction/cache
alignment and Python launcher/package isolation remain open.

### Native Python wheel startup isolation

The default CLI/server wheel no longer installs ONNX Runtime or imports/preloads
its Python package. Optional `ort` and `ort-cuda` extras supply explicit reference
runs; MLX installation is opt-in. Library discovery uses top-level module specs
without executing package initializers. Explicit paths retain priority, including
Windows reference-DLL dependency directories whose handles remain owned and are
reused. The existing abi3-py310 extension now admits standard Python 3.10+;
publishing builds on CPython 3.12. PyO3 function registration no longer uses the
deprecated `pyfn` helper. Publishing preserves the complete ABI feature list for
CUDA reference wheels and runs the installed Native command smoke on every CLI
wheel platform instead of inheriting reference-runtime platform skips.

On macOS ARM64 / standard CPython 3.15, an actual development-profile abi3 wheel
passed dependency-metadata inspection, default Native generation without ORT,
and generation with optional package initializers that fail if imported. Explicit
missing-ORT selection fails clearly and a real ORT reference generation succeeds.
A fresh virtual environment installs the default wheel offline with no ORT/MLX
packages; the actual console command passes version, Native generation and missing
explicit-reference smoke checks, and `pip check` reports no missing dependencies.
Seven launcher tests cover discovery, absent/incomplete installs, path priority
and Windows handle lifetime/reuse. Python lint and CLI Python/CUDA-reference
feature strict Clippy pass. After restacking onto the workspace-report main,
the two Native MTP tests and 71 CLI tests pass again, as do the real wheel checks,
formatting and diff checks. Windows loader cases are mocked, and no CUDA
execution or performance result is claimed. This closes the Python CPU startup
isolation gap; the overall compatibility acceptance rows remain open.

### Canonical live reasoning-control routing

The pinned upstream README, server route and bundled frontend use
`POST /v1/chat/completions/control`; the server now registers that exact path.
The earlier local route is removed rather than retained as a pre-release alias.
Control registrations carry the admitted model's identity instead of trusting a
request-supplied model name. Router requests require a configured model and can
signal only that model's live completion. Unknown models return actionable 404
errors; idle or wrong-model lookups return `success:false` without loading a
model. Single-model controls ignore the optional model name. Discovery and
controls share one read-only catalogue-role predicate, preserving existing
`/props` behavior without loading models or updating inference-use timestamps.

Seventeen control-related tests and the full 452-server / two-inherited-ignore
suite passed. A real Native sampler verifies that wrong-model controls leave
its selected tokens unchanged and correct-model control emits the declared
reasoning closer. The test also covers missing/blank models, unknown-model
recovery hints, no idle-model autoload, old-route 404 and poisoned-registry
operation/cause/recovery text. Existing chat/legacy buffered/SSE registrations,
capacity and completion cleanup checks use the canonical path. All 38 HTTP
integration tests, Native-CUDA-feature all-target strict Clippy, formatting and
diff checks passed locally. CUDA execution remains unmeasured. This corrects a
frontend-blocking endpoint gap; remaining acceptance rows stay open.


### Model-bound LoRA serving

`GET/POST /lora-adapters` and per-generation `lora` now lower into the
admitted workflow service. Defaults initialize at scale 1 (or disabled with
`--lora-init-without-apply`), snapshot per request, and remain isolated between
models. The wire parser preserves upstream ID ordering, last duplicate wins,
and missing/null/non-numeric scale defaults. Invalid updates leave defaults
untouched; unavailable owners return an actionable 503.

Real base-plus-low-rank execution supports non-square projections, sparse
bindings and multiple fused slices. Model-governed host leases cover parsed
weights, bounded staging and generated tensors, surviving no-copy aliases.
Token-history replay supports named adapter sessions; other retained state is
refused before mutation. Native generated-token tests run with ORT unavailable.
See [LoRA serving](LORA_SERVING.md) for the executed contract and remaining
source-format, device and embedding gaps. The overall acceptance rows stay open.


### Adapter-aware embedding workflows

Embedding v1 profiles now resolve an explicit floating workflow output and optional
sequence pooling. Embedding calls execute that workflow, so its declared LoRA
updates affect returned vectors. Model defaults and per-request selections share
the generation parser and snapshot once per batch. Native and OpenAI embedding
transports forward overrides rather than dropping them. Decoder-only extraction
refuses nonempty adapter selections, and model properties expose embedding-specific
adapter admission. The compatibility goal remains active for the remaining loader,
device, model-management and full-matrix acceptance items.

### Upstream properties acknowledgement

`--props` enables `POST /props` without loading a model, with the current
upstream acknowledgement-only response. Request fields do not mutate settings;
this matches the pinned server-context.cpp handler, which has no mutation schema.
Both router and model properties report the configured endpoint flag. Router
properties also report that server tools are unavailable independently of loading.


### Tool execution belongs to the external harness

The user explicitly excluded server-provided tool execution from the compatibility
objective. Built-in filesystem/shell tools, `/tools`, stdio MCP, `/cors-proxy`, and
container/SSH runtimes are removed from the server and its remaining-work ledger.
Their CLI/configuration types and dedicated dependencies are removed as well;
there are no compatibility aliases for these prerelease options.

`/props` advertises `server_tools: false` and `cors_proxy_enabled: false`. Unknown
tool/proxy routes return JSON 404. OpenAI/Anthropic/Responses tool-call transport,
model tool-protocol admission and tool-result messages remain in scope. External
harnesses execute calls and supply results to the next inference request.

### Non-generative SystemOne workflow and transport

The decision v1 profile declares an actual probability-head workflow, tokenizer
features and model-authored prompt/answer policies. A schema v1.10 gate applies
in parser and typed admission. `/v1/systemone` routes choice/score/noul requests
through bounded engine workers with zero generation tokens, accurate multi-variant
input usage and actionable shape/calibration errors. Canonical leases retain
converted probabilities through engine-result ownership. A native ONNX fixture
changes its decisions with the input; it does not simulate a chat answer.

This increment does not establish all upstream decision-model conformance.
Multimodal inputs, joint/packed heads and shared workflow entry points remain
required; [SystemOne](SYSTEMONE.md) records the
current executable contract and explicit gaps. Server tools remain out of scope.


### Declared SystemOne input contracts

Core schema v1.11 requires explicit `decision.input_policy`. A single engine
question factory admits defaults, required/optional/forbidden instructions,
payload types, enabled question kinds, dynamic/fixed choices, request/lexical
order, declared score ranges, boolean alias/order rules and legend formatting.
The HTTP transport supplies its operator limits; it does not duplicate question
interpretation. Immutable questions/results retain their admitted contract, with
fixed options and default instructions borrowed from that owner. No model-name
branch selects request semantics. Schema v1.10 keeps its published contract.

Native inference tests exercise missing/null/non-string default types, changed
choice ordering, true-first boolean heads, canonical-null alias precedence,
fixed tasks and one-/eleven-level dynamic score heads. HTTP tests cover authored
limits, payload rejection and wire legends. These tests establish text decision
behavior for the authored fixtures; they do not establish media or GGUF support.


## Managed native-package downloads

`--model-cache-dir` enables `/models` download/delete and cache-backed load/unload
without enabling `/v1/admin/*`. A validated Hub plan pins one immutable commit;
partial files, full quota reservations and publication manifests survive
restart. Canonical registry entries own transfer or package state, and native
worker lifetimes retain package ownership until their threads stop. Only
server-owned cache entries can be removed.

The SSE stream reports upstream model envelopes with actual per-file counters,
bounded notification storage and consumer-side coalescing. Pause and retry
retain the original plan and byte prefixes. Model revision changes require
explicit deletion first. See [model management](MODEL_MANAGEMENT.md) for public
requests, limits and unsupported GGUF/sidecar source selections.


## Multimodal reranking

Core schema v1.12 and reranking v2 declare ordered query/document encoded-image
endpoints and authoritative workflow token usage. Prepared pairs carry their
admitted contract and shared media ownership into the ordinary worker command.
The HTTP layer admits all image positions before fetch, bounds the retained
request media, and preserves Jina/TEI response semantics.

Reusable image-preprocess@2 binds program output names to component-local ports;
each invocation publishes its own SSA values. ABI v1 keeps its established
workflow-scoped single-invocation contract. Chat media capabilities derive from
runtime Media roles, independently of task-owned application inputs. See
[reranking](RERANKING.md) for the public contract and native fixture.


## Ordered image decisions

Decision v2 uses core schema v1.14 to declare ordered image endpoints and a
workflow-owned total prompt-token count. The HTTP transport preserves upstream
files/images/chat-media ordering, strips only image parts from textual state,
and bounds unique retained media through the image loader shared with reranking.
Prepared decisions retain shared encoded images across questions and variants;
native workflow steps perform vision preprocessing and calibrated probability
inference. Negative/underreported usage is rejected, and a typed input-limit
error reaches HTTP as a client error. See [SystemOne](SYSTEMONE.md) for the
explicit contract and the remaining audio/video, joint/packed and shared-entry
work. This increment does not close the overall compatibility acceptance rows.

### SystemOne mixed audio contracts

Decision v3 (core v1.15) binds ordered typed state media and a distinct spoken
instruction endpoint. HTTP extraction preserves state structure and media order;
PCM16 WAV bytes enter authored reusable audio-preprocess@2 components. Actual
workflow token usage includes media expansion and is bounded. Native fixture
and HTTP acceptance cover independent spoken questions, byte-budget sharing,
invalid media and capability discovery; see [SystemOne](SYSTEMONE.md).

## Native prompt-only prefix reuse

`n_predict=0` now retains and reuses actual Native prompt state, with exact full
hits, suffix appends and permitted KV prefix rewinds. The physical owner and the
package's `prefix_reusable` declarations decide reuse; a token-history match
alone does not. Non-sliceable state resets for destructive prefix replacement.
Cold requests force full evaluation. HTTP checks cover buffered/SSE zero-output
responses, cache counts and the committed idle slot cursor. Engine tests compare
content-dependent key/value tensors with a cold oracle across repeated, appended,
diverged, shortened and complete-context prompts, pin no-reset behavior on a full
hit, and preserve the existing cancellation/rollback checks.

Prefill responses also distinguish input reuse from post-task retention:
`PrefillResult.prefix_cache_hit_len` measures reused input, while
`retained_cache_tokens` supplies native-wire `tokens_cached`. Cold evaluation
still retains evaluated cache. ORT reference results inspect governed snapshots
and cached pages rather than claiming token-history-only retention. The same
wire correction for general generation remains required work.


### Ordinary generation retained-cache reporting

`GenerateResult.retained_cache_tokens` records physical state retained at the
completion boundary, separately from `prefix_cache_hit_len` (input reuse).
Native final buffered/SSE responses publish the retained count as `tokens_cached`.
Native stateless cold execution resets before evaluation and retains its newly
evaluated state afterward. ORT ephemeral sessions report only shared snapshot
or page payloads surviving session closure; named and prioritized results report
their materialized session state. Generic workflows without an exposed token-cache
contract report zero. Cached-prompt usage in OpenAI-shaped responses remains an
input-reuse statistic.


### Decoder timing propagation

Ordinary decoder generation records prompt-forward and generation work in the
shared token policy. `GenerationClock` closes each active step before external
callbacks and survives priority-drive suspension. `GenerateToken` snapshots and
`GenerateResult` carry the canonical engine measurements. Native commit-only
journals retain the snapshots; ORT collects a private timing journal, checks its
token ids before commit, then pairs snapshots with authoritative reconstructed
text during post-commit delivery. No timing event can expose an aborted turn.

Native completions and OpenAI completion/chat responses render the pinned
[llama.cpp stats fields and rate denominators](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/tools/server/server-common.cpp#L147).
First-token rate is zero (no extra generation forward), and zero-duration values
remain finite. `timings_per_token=true` publishes cumulative SSE measurements;
chat timing-only chunks have empty choices, as accepted by pinned Llama Chat.
Standalone prefill, speculative execution, and generic authored workflow timing
remain distinct unfinished contracts. Prompt progress events are still open.


### Prompt-forward progress and prefix provenance

`GenerationControl` has an independent typed prompt-progress observer alongside
its cancellation/checkpoint authority. Ordinary decoder hosts report real prompt
forward boundaries, before token publication. Native/OpenAI completion, chat and
infill SSE only serialize these updates for `return_progress=true`. Buffered
collectors and other transports ignore this execution metadata. FIM execution
now uses the same cancellation/progress authority and committed token callbacks.

The pinned [upstream progress assignment](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/tools/server/server-context.cpp#L2306)
counts cached tokens in `processed`. A whole-prompt forward therefore starts at
`cache` and completes at `total`; it does not invent intermediate percentages.
Progress is not semantic state publication: an aborted turn may have reported
work, but still publishes no tokens, workflow output, or success completion.

`PreparedPrompt` distinguishes reused prefix state from a semantic prefix freshly
evaluated in the same request. Native incremental decode resumes from the physical
cursor in either case, while input-reuse statistics use only the former. The
clock includes fresh prefix evaluation in prompt work, and progress reports its
actual intermediate boundary. A compressed record-state fixture with a partial
carry proves fresh prefill and cross-session snapshot restoration match an
independent complete cold evaluation. Declared `prefix_reusable=false` and memory
admission guards remain in place. The new snapshot and its memory lease remain
private until semantic commit; observer failure or cancellation cannot publish
that provisional prefix. Failure injection before prefill, after the partial
boundary and after full prompt evaluation proves cache/history publication stays
atomic and retry recomputes the prefix. SSE replay coverage includes progress
and timing frames in its byte-exact retained history. Speculative/composite,
continuous-batch and standalone prefill progress contracts are still unfinished.


### JSON property-key escape identity

The pinned llguidance 1.9.1 compiler used serialized raw known-key literals but
JSON-quoted generic string lexemes. This rejected escaped DEL in a declared key
and could classify an alternatively escaped known key through a different
property branch. The local compiler patch builds one quoted-name regex (retaining
the existing raw spelling) and reuses that exact regex for property recognition
and pattern/additional exclusion, including impossible optional properties.
The JSON prefix normalizer also canonicalizes legal raw DEL inside strings;
original prompt/output text remains untouched. Normalization-sensitive vocabulary
pieces bypass the literal-byte trie; optimized masks are checked against the
canonical recognition path for split and whole-token DEL keys and values.

The workspace uses one canonical patched dependency in `vendor/llguidance`.
Its MIT license, published archive SHA-256, all original file hashes, and exact
patched-file hashes are recorded there. `scripts/check_llguidance_vendor.py`
verifies the patch footprint without network access and runs in the existing CI
gates. The initial declared-key regression failed before the patch. Conformance
covers every escape split, known/free keys, pattern ownership, unsatisfiable
optional keys, anyOf branches, Unicode/surrogate keys and literal backslashes.
HTTP continuation forces valid/invalid competing suffixes through the actual
model in buffered/SSE mode, with and without response projection.

The former external vendored-package identity did not supply these fixes to
registry-only consumers after Cargo removed local dependency paths. The compiler
is now the first-class `onnx-genai-llguidance` package, published before the engine
with the same audited source and public typed distinction. See the
[compiler artifact verification](GRAMMAR_COMPILER.md) for the exact packaged/local
registry proof; configuring that release graph does not claim an upload occurred.

Full server verification also exposed a realtime disconnect release race.
Admission capacity now drops before the worker reservation advertises idle,
including early initialization failures; the existing disconnect regression
checks both resources before admitting a fresh connection.


### Prompt-only evaluation progress and timings

`n_predict=0` now carries actual `GenerationTimings` from Native prompt evaluation
and the explicit ORT reference path. Prompt tokens count fresh work; reused input
is `cache_n`, retained cache remains `tokens_cached`, and no generation work is
invented. Fully cached Native prompts execute no forward and report zero elapsed
prompt work. Observer delivery, preparation and network waits are outside the
measured forward interval.

The existing atomic prefill transaction emits typed start/completion progress
before commit. Callback failure or cancellation at either boundary rolls back
provisional state and prevents prefix publication. The server reserves admission
before opening SSE, then drains a bounded progress channel concurrently with the
committed result. Dropping the receiver requests cancellation through the shared
control probe; terminal delivery follows slot publication and permit release.
Buffered responses and the final SSE frame include timings; `return_progress`
remains opt-in and does not create output tokens. Native/ORT regressions cover
cold, reused and extended prompts, complete Native cache hits, observer failure
and cancellation, plus real buffered and streamed HTTP responses.


### Native speculative progress and timing reports

The Native prompt-lookup/MTP block driver now uses the shared execution clock and
request control. It reports the real target-prompt start/completion, with zero
input reuse because this path resets the target cache. Proposal, target verify,
accepted-state restoration and token selection contribute to generation work;
progress/checkpoint callbacks and time between blocks do not. Every accepted
output token advances cumulative counts even when several share one verification
forward. Final and per-token measurements travel through the existing commit-only
output journal and server event adapters.

Deterministic clock tests cover paused observer gaps and multiple accepted tokens.
A real repetitive-context fixture forces multi-token acceptance and compares all
outputs with plain greedy decoding, including monotonic per-token measurements
and final-sample identity. Failure/cancellation at prompt and proposal/verification
boundaries aborts without externally publishing tokens, and retry matches a fresh
engine. The server driver regression checks prompt events precede the committed
ordered token/timing journal, including a staged-output failure and retry. Ordinary
native generation keeps its existing path. ORT speculative, composite and
continuous-batch reporting remain separate open work; no GPU performance claim is
made by these CPU functional tests.

The multi-token fixture also exposed a preexisting out-of-bounds acceptance path
when a target returned last-token-only logits for a wider verification window.
The driver now validates exact verify-row count and vocabulary width before
indexing accepted/bonus positions. A malformed-target regression proves this
fails actionably without a panic or token publication and that ordinary retry
matches a fresh engine. The positive fixture explicitly expands its target
logits along the runtime input sequence, rather than claiming multi-token
coverage from the original one-row fixture.

The Native MTP subprocess regression explicitly removes the ORT backend selection
and points its ORT library at a nonexistent path. It validates real proposal and
verification, prompt/final/per-token timing reports, and cancellation/retry after
both head proposal and target verification. This proves the same reporting and
abort contracts on the actual Native head path rather than only prompt lookup.


### Contextual fused trailing Strip decoding

The shared byte-stream contract now resolves finite trailing Strip after fusion
(ByteLevel is already fused). Ordered initial/trailing stages retain only their
bounded suffixes, preserve split UTF-8 and never retract published bytes. Terminal
flushing removes only complete matching characters; incomplete scalars remain
visible to strict terminal validation. Token-local stages before the byte stage
keep their existing semantics. Chunk-local post-ByteFallback Strip still needs
an explicit Fuse, and other unresolved contextual transforms remain rejected.

Content constraints inspect both the provisional retained suffix and actual
reachable terminal outputs. When multiple characters are withheld, a bounded
state search releases only prefixes that a real continuation of the composed
decoder can expose. It includes split-scalar completions and skips unchanged
window-filling runs. This accepts a partially released valid ending without
inventing an ending between composed stages. Published plus retained bytes share
the response limit; per-chain retention and search work have explicit guards.
Literal-byte vocabulary masks remain unchanged for context-free decoders.

Reasoning-budget token markers are admitted only when the trailing decoder leaves
them visible at the declared token boundary. Tool observation uses the same
ordered decoder and includes retained bytes in its response budget. Its deferred
contextual-decoder regression now uses an unresolved Replace stage, preserving
that fallback coverage after fused trailing Strip became supported. Existing
panic fences still preserve dependency decoder failures and their causes.

Conformance covers ASCII/multibyte tails and every chunk width, ordered composed
stages, bounded retention, partial-scalar EOF refusal, large finite windows,
semantic candidate masks, and stable reasoning markers. A real Native HTTP
fixture competes a higher-biased schema-invalid token against a valid token whose
encoded suffix is stripped, through buffered/SSE and ordinary/logprob output.


## Buffered speech PCM transport

Native `/tts` now supports default WAV, float32 little-endian `pcm`, and signed
16-bit `pcm16`; OpenAI speech aliases map `pcm` to signed 16-bit samples. Typed
transport selection reaches the declared audio output through the owner driver.
Package rates and channel layouts are retained and described in raw PCM MIME
parameters. Float waveform encoding precedes quantization; validated post-adapter
PCM16 WAV conversion preserves the already quantized source. Nonfinite samples
and incomplete post-adapter frames fail explicitly. Streaming speech remains open.

Validation on Native CPU: 14 audio unit tests, both ONNX-owned audio workflow
tests, and the complete server library (580 passed, 2 inherited ignores) pass.
Byte checks include stereo interleaving, float precision, symmetric PCM16
rounding, full signed-range post-adapter conversion, and both HTTP PCM mappings.
This checkpoint does not establish audio quality or GPU behavior and does not
complete the remaining compatibility acceptance rows above.


## Native speculative response counters

Native prompt-lookup and MTP timing snapshots now carry actual proposal and
accepted-draft totals through commit-only token publication. The shared server
timing renderer exposes `draft_n` and `draft_n_accepted` only after a real
proposal, matching the pinned upstream response fields. Acceptance counts
verification work, excluding target bonus tokens; it is not reconstructed from
published text length. Ordinary decode omits both fields. ORT speculative,
composite and continuous-batch measurement contracts remain open.

Real Native reporting tests cover multi-token acceptance, per-token/final
snapshot consistency and clean retries after progress/phase cancellation.
Native MTP is exercised in a subprocess with the ORT library unavailable. The
actual server driver checks the same counter snapshots after an aborted turn
and successful retry. This increment does not close the broader acceptance rows.


## SSE idle comment pings

The previously accepted but ignored `sse_ping_interval` now configures actual
idle comment pings for generation transports. A typed policy shares CLI,
server-default, request validation and `/props` reporting. Default 30 seconds
matches the pinned upstream; -1 and 0 disable pings. Data resets the idle timer
and source EOF terminates it. Native completion/infill, OpenAI completion/chat,
Responses and Anthropic adapters apply the same policy. Replay retains exact
comment bytes rather than removing them and shifting reconnect offsets.

Wire tests cover idle reset, disablement and termination; HTTP tests cover
request overrides and rejection across five generation protocols; a real replay
middleware test checks offsets across comment and UTF-8 payload bytes. This
increment does not check off unrelated remaining compatibility acceptance rows.


## Native final-response field projection

`response_fields` now projects actual native completion/infill terminal results,
including buffered batches, SSE batches and prompt-only evaluation. An empty
selector preserves the original response. The pinned source traverses object
keys and retains the complete slash path as the output key; this differs from
its README's leaf-renaming description. Missing paths are omitted, array indices
and JSON Pointer escapes are not inferred. Intermediate token/progress events
and errors keep their protocol contracts. Native batch completion remains tied
to its owner outcome channel rather than projected JSON fields.

Unit cases pin nested objects, empty segments, duplicate/missing paths, arrays
and literal tildes. Real HTTP coverage exercises unchanged empty-selection
results, nested timing selection, buffered/SSE batches without stop/index,
prompt-only output, admitted infill, and invalid selector types. This remains an
increment toward the original compatibility scope, not completion of its rows.

## Prediction wall-time limit

`t_max_predict_ms` now reaches the request-owned decoder stop policy rather than
being accepted and ignored. The pinned server's implementation checks a fresh
emitted newline after strict wall-time expiry, starting with the first generated
token. Null and disabled values preserve ordinary generation; numeric fractions
truncate toward zero, while unsafe overflow representations fail admission.
This is a normal output/state commit with a limit finish reason.

An explicit output-observer capability admits time/stop observers to raw-greedy
Native speculative verification while keeping logits-transforming processors
on their existing fallback path. Stops inside an accepted block restore only
the retained prefix, including hidden conditioning, before token publication.
Injected repair failure publishes no token and permits a clean retry. Generic
authored workflows stop at declared emission boundaries; candidate-tree and
DFlash drivers truncate before accepted-state recomputation.

Local macOS CPU validation includes real Native/ORT buffered and callback
generation, Native multi-token acceptance, continuous-row reuse, pure authored
workflow execution, and actual candidate-tree/DFlash prefix commits and retry.
Full server library tests passed (591, two existing ignores); full engine
library tests passed (1083, one existing ignore). Full DFlash (24), candidate-tree
(30) and authored workflow policy (22) suites passed; the additional DFlash
observer-bound failure check proved no publication and a clean retry. Default
engine/server all-target strict lint, no-default engine strict lint, both Python
binding checks, formatting and diff checks passed after restacking onto main.
These checks validate this increment; the unchecked goal acceptance rows remain
open.

## Numeric SSE ping overrides

The pinned schema treats null numeric fields as absent and converts fractional
numbers before checking hard bounds. `sse_ping_interval` now follows that rule:
null inherits the operator's configured interval, fractions truncate toward
zero, and the resulting value must fit -1 through INT32_MAX. The prediction
millisecond limit and ping interval share safe signed conversion without
emulating undefined upstream overflow casts. Their own units, upper bounds and
defaults remain distinct. CLI ping intervals remain integer arguments.

Coverage includes enabled/disabled custom default inheritance, fractional
disablement and positive intervals, fractional signed boundaries, rejected
types/overflow, and admission across five generation protocols. Existing idle
comment timing, source termination and replay-byte tests remain required.

Full server library validation passed (591 tests, two inherited ignores),
including the existing prediction-time tests after conversion was shared.
Server all-target strict Clippy and diff checks passed on macOS ARM64 CPU.
This closes the numeric ping discrepancy, not the remaining goal acceptance rows.

## Native final request and stop metadata

Native terminal responses now carry detokenized admitted prompt tokens,
resolved supported sampling settings, visible-output newline state and the
matched text stop word. Serial Native executors publish their actual slot ID.
The request preparation boundary captures the normalized settings before driver
submission, including admitted output/context budgets. Prompt-only runs retain
a zero generation cap. Active optional samplers report their parameters, and
explicit seeds remain distinguishable from unspecified random seeds.

The metadata travels with its own batch row through buffered and SSE delivery,
infill and prompt-only execution, and is available to terminal response-field
projection. Tests use an actual model with a tokenizer newline/stop suffix to
distinguish visible newlines from newlines removed inside stop strings. They also
exercise exact prompt tokens, sampler settings, logit biases, batch ownership,
infill prompts and projected streams whose terminal JSON omits stop/index.

Validation after main restack: full server library 592 passed, two inherited
ignores; server all-target strict Clippy and diff checks passed on macOS ARM64
CPU. The remaining goal acceptance rows stay open.

## Stops inside decoded token fragments

The ordinary decoder now detects text stop words inside a selected token rather
than only at the end of accumulated text. It selects the earliest visible byte
position, with configured order resolving ties, and keeps token-ID stops as
suffix matches. A request-owned scan cursor plus longest-stop overlap preserves
cross-token Unicode matches without rescanning the entire history on each step.
The stop configuration is immutable after processor construction.

Engine matching, output buffering, final text clipping and prediction-time
visibility now share textual boundary classification. Incomplete stop prefixes
remain withheld; a newline released when a prefix ceases to match becomes
eligible for time expiry, while a newline hidden by a complete stop does not.
Tests cover true accepted-token counts and stop reasons, stop selection within
one token, native/OpenAI/chat buffered and logprob SSE output, and Native
speculative prefix repair with injected failure and retry.
The native stopping-word diagnostic uses that same match independently of the
final stop type; a real preserved EOS token containing a stop string verifies
that EOS priority retains the matched word in buffered and SSE responses.

The audit also identified remaining work for indentation stopping (`n_indent`)
and general stop-sequence observation in authored candidate-tree/DFlash paths.
These remain part of the original goal, alongside the unchecked acceptance rows.

Validation after restacking onto main 007d488d0: full engine library 1086
passed with one inherited ignore, full server library 593 passed with two
inherited ignores. Default engine/server all-target strict Clippy, no-default
engine strict Clippy, formatting and diff checks passed on macOS ARM64 CPU.
The unchecked goal acceptance rows remain required.


## Authored workflow output stops

The shared output observer now applies ordinary text/token stop sequences to
candidate-tree, DFlash and generic authored loops as well as prediction time.
The shared processor builder preserves observer priority. Token-only controls
work without a tokenizer; text controls require one explicitly. Generic loops
also enforce request-wide token budgets and EOS independently of authored loop
iteration bounds. Their current body finishes atomically before the next body
is suppressed; over-budget atomic emission rejects the whole turn.

Specialized drivers retain the accepted prefix before state recomputation.
Terminal decoder flushing occurs before semantic commit, allowing pending UTF-8
to trigger a text stop and preserving rollback if its byte budget fails.
Tests compare real specialized state continuations with budgeted references,
exercise a real ONNX generic recurrence, and check buffered/callback publication,
EOF decoding, EOS priority, missing-tokenizer admission and failed-turn retry.
Validation: full DFlash 26, candidate-tree 32 and workflow policy 28 tests
passed. After rebasing onto main 1ef2439da, full engine library 1086 passed
with one inherited ignore and full server library 593 passed with two inherited
ignores. Default engine/server all-target strict Clippy, no-default engine
all-target strict Clippy, formatting and diff checks passed on macOS ARM64 CPU.
The general goal remains incomplete; indentation stopping and unchecked
acceptance rows remain required.


## Indentation stopping

`n_indent` is parsed with the pinned numeric domain, mapped
into the shared output-observer chain and ordinary, batched, speculative and
authored workflow result construction. Its stateful scan distinguishes full
stop-word removal from partially held stop prefixes and handles repeated/EOF
observations without advancing a selected token twice. Text clipping is replayed
independently of the final stop label, preserving EOS/time priority. Native
stopping-word diagnostics use raw token text before indentation clipping.

SSE gates publish safe line prefixes and withhold potentially retracted bodies,
then resolve against committed final text. This deliberately avoids the pinned
server's partial-response send occurring before its indentation truncation.
Validation on main b07048488: full DFlash 27, candidate-tree 33 and workflow
policy 30 tests passed; engine library 1095 passed with one inherited ignore,
and server library 599 passed with two inherited ignores. Default engine/server
all-target strict Clippy, no-default engine all-target strict Clippy, GenAI Python
all-target compilation, formatting and diff checks passed on macOS ARM64 CPU.
Real Native ordinary/speculative and ORT reference prefixes, EOS override,
tokenizer admission, HTTP numeric bounds and buffered/SSE equality are covered.
There are no performance or GPU claims. Unchecked goal acceptance rows remain
required; this increment does not complete the broader compatibility goal.


## C-locale collating and equivalence trigger classes

Trigger brackets support collating elements and standalone
equivalence/POSIX classes, including named characters and scalar range endpoints.
Real C++ checks exposed distinct libc++ and libstdc++ classic-locale name and
primary-equivalence behavior. A typed `GrammarCollation` travels with immutable
activation configuration; the request extension `grammar_trigger_collation`
selects `libcxx` (omitted/null default) or `libstdcxx`. Properties advertise both.
Production matching does not depend on the host's compiler or locale.

Paired actual-compiler fixtures cover the same 504 patterns, every one-byte
input for each valid expression, named aliases/controls, negation, range roles,
NUL and primary-key edge cases. Runtime observers retain existing source, input,
program, class-item and VM work bounds. Real Native/ORT HTTP tests exercise both
profiles through pattern/full-pattern activation, captures within model tokens,
framing, buffered/SSE projection and pre-SSE error admission. The 33 GBNF and
5 wire tests passed. Final validation on main 6d9898009: engine library 1096
passed with one inherited ignore; server library 600 passed with two inherited
ignores. Default engine/server all-target strict Clippy, no-default engine
all-target strict Clippy, GenAI Python all-target compilation, formatting and
whole-change whitespace checks passed on macOS ARM64 CPU. Both actual compiler
oracles were regenerated with canonical data formatting. These are functional
checks, with no Linux/GPU execution or performance claims. Lazy JSON, contextual
raw grammars and other unchecked goal rows remain required.

### Lazy JSON Schema activation

JSON response formats and the native `json_schema` alias now lower to an explicit
raw lazy JSON Schema constraint. Admission compiles the existing full llguidance
schema recognizer and validates triggers before SSE. It retains the raw identity
byte-decoder requirement rather than guessing contextual token pieces. Activation
shares GBNF token/word/pattern/full-pattern priority, byte capture offsets,
reasoning suppression, history bounds and rewind behavior. JSON replay seeds the
captured raw bytes into the existing normalized content recognizer, preserving
schema semantics, equivalent escaped spellings and the vocabulary trie mask.
Assistant prefill and content projection do not become part of this raw value.

An optional selected-token feedback capability on constraints closes the last-step
activation gap: grammar validation runs before selected output is committed or
published even when no next mask is requested. An activated ending must complete
the root. Arbitrary output/EOS before activation remains legal; explicit stops
and length limits can truncate an activated viable prefix. Lazy JSON rejects
forced tool choice and `ignore_eos=true`, preserving raw-grammar policy.

Validation on main e645c6662: server library 601 passed with two inherited
ignores; engine library 1101 passed with one inherited ignore. The new Native/ORT
wire test covers 96 successful route/format/trigger/stream combinations, non-JSON
assistant prefill, the native schema alias, pre-SSE schema/trigger/EOS-policy
errors, final-token rejection without publication and retry on the same server.
Engine regressions cover selected-token feedback, rewind, root-complete endings,
equivalent JSON escapes, partial UTF-8 activation and seed byte-budget recovery.
Default engine/server all-target strict Clippy, no-default engine all-target
strict Clippy, GenAI Python all-target compilation, formatting and whole-change
whitespace checks passed. After full tests the only source adjustment was
Clippy's equivalent direct-Result return. These are functional macOS ARM64 CPU
checks; no GPU/Linux execution or performance claims. Contextual raw grammars
and the other unchecked compatibility-goal rows remain open.

### JSON object domains and schema wire forms

Pinned `server-common.cpp:1420-1442` and `server-schema.cpp:270-291` normalize
empty schema objects to object roots. The transport now resolves that default
for native, legacy completion, chat and Responses without changing the engine's
any-value JSON constraint. Boolean and nonempty schemas retain full schema
semantics. Typed schema documents reject non-object/non-boolean values with an
argument-specific error.

Response formats accept optional nested metadata, nested/flat schema forms and
`json_object.schema`; top-level `json_schema` is no longer silently ignored by
chat/completion requests. A shared selection method encodes pinned precedence
and feeds both continuation validation and constraint construction. Native
legacy grammar presence takes precedence, including an empty grammar; chat keeps
its explicit conflict guard. Lazy activation, EOS policy and forced-tool guards
consume the selected schema too. The obsolete `JsonSchemaSpec` API was removed
and all exports/callers updated without aliases.

Validation on main 71ecaa373: server library 608 passed with two inherited
ignores; the full public HTTP integration suite passed 38 tests. Native/ORT
coverage includes 416 successful path/format/eager-lazy/stream combinations,
object-vs-scalar masking, boolean/explicit scalar schemas, all wrappers, prefill,
false-schema pre-SSE refusal and invalid lazy scalar publication. Defaults and
precedence also have request-type and native-parser regressions. Public HTTP
expectations now require object-root schema selection and accept optional
wrapper metadata while retaining malformed-document rejection.

Default all-target strict Clippy and Native-disabled reference strict Clippy
(`--no-default-features --features metrics,cuda-13000`) passed, as did CLI/Python
all-target compilation (`--features python`), formatting and whole-change
whitespace checks. The CUDA selector supplies the mandatory cudarc test binding
version; no GPU execution was performed. Metrics-disabled strict Clippy still
reports eight inherited dead-code diagnostics; the identical errors were
reproduced on unmodified main 71ecaa373 before restoring this branch. This is
recorded rather than claimed as a passing configuration. Checks are functional
macOS ARM64 CPU evidence, with no Linux/GPU or performance claims. The wider
compatibility goal remains open.

### Graph-scoped keyed container weights

Schema v1.17 adds explicit `graph_weights` entries for ONNX graphs, with
package-relative GGUF/SafeTensors paths and container-key renames. One canonical
resolver admits regular files beneath the package root and rejects escaping
symlinks, missing artifacts and aliases that ambiguously bind the same graph.
Decoder sessions, workflow components, MTP heads and target-initializer reuse
consume the same bindings. Auto selects Native for this capability; explicit ORT
and builds lacking Native fail with a configuration remedy.

Execution retains the loader's exact dtype/shape and external-length checks.
There is no inferred architecture, cast, transpose or quantization conversion.
Memory evidence loads the actual keyed graph, and keyed placeholders use tensor
geometry for byte estimates. Native portable state hashes declared container
contents with portable package-relative labels, so relocation preserves identity
and a changed container invalidates existing state before restoration publishes.

Real GGUF and SafeTensors fixtures execute the decoder, a stateless workflow and
three recurrent MTP steps. Decoder generation matches the original ONNX/ORT
reference; saved-state continuation survives package relocation, while altered
container contents are rejected without populating the destination session.
Shape coercion, explicit ORT, missing files, graph aliases and escaping symlinks
have refusal coverage. The complete engine library passed 1108 tests with one
inherited ignore; metadata passed 484 tests, including generated-schema and
extension-registry conformance. Evidence is functional macOS ARM64 CPU only.
Bare GGUF executable-graph import, quantization selection and selective sidecar
management remain open; this increment requires an authored ONNX graph/package.

Default engine/server all-target strict Clippy and the Native-disabled server
reference configuration (`--no-default-features --features metrics,cuda-13000`)
passed, together with formatting and whole-change whitespace checks. The CUDA
feature is a compile-time binding selector, not evidence of GPU execution.


### Explicit model-package variant installation

Named `repository:variant` identifiers now carry an explicit model-package
selection through Hub planning, cache publication/restart, registry loading,
Native/ORT engines and workflow execution. Remote planning and local loading
share manifest/component selection and shared-asset rules. The planner excludes
unselected variant directories before applying package quotas; selected index
JSON is bounded and size/hash checked. Variant-local and shared tokenizer prompt
assets use one ordered resolution path. `/props` advertises both flat nxrt
packages and ORT model-package indexes.

Selected loading checks every referenced canonical path without requiring other
variants to be installed. The explicit validate-all package audit retains its
all-variant checks, and symlink/traversal guards remain in the shared resolver.
Cache-manifest version 2 persists repository and variant separately and rejects
alternative selector spellings or old/future versions; no cache migration shim
or automatic operator-file rewrite was introduced.

Real local Hub tests prove a selected small package is admitted while an
unselected one-GB artifact is not requested, including external indexes, shared
tokenizer/templates, persisted resume/publication and cache reopen. Native and
ORT generation execute the selected package. Both backends also execute a
selected composed decision workflow through the real `/v1/systemone` route.
Unknown variants and twelve corrupt/unsafe index cases fail before model payload
requests. Bare GGUF graph import, filename quantization resolution and selective
sidecar operations remain open.

Selected indexes now retain a borrow of their package owner and resolve metadata
before executable artifacts. Workflow admission therefore still precedes graph
or tokenizer inspection. Manifest decoder and workflow decoder paths must agree;
all manifest-backed workflow graph/tokenizer references share canonical package
containment. Implicit metadata and tokenizer files also reject escaping or broken
symlinks. Public HTTP tests cover selector-bearing progress, shared-template props,
load, actual decisions and deletion that leaves another variant executable.

Final validation after rebasing onto main 73f17da8f: all 20 model-package tests,
200 ORT library tests, 614 server library tests (two inherited ignores), 1108
engine library tests (one inherited ignore) and 38 public HTTP integration tests
passed. Strict all-target Clippy passed for all four changed crates and the
Native-disabled reference server (`metrics,cuda-13000`); CLI/Python all-target
compilation, formatting and whitespace checks passed. The offline route audit
remains 51 present / four external-harness contracts, without claiming that
registration establishes semantic parity. Evidence is functional macOS ARM64
CPU only, with no Linux/GPU execution or performance claims.


## Realtime provisional function argument projection

The declared shared tool parser now exposes bounded cumulative call prefixes.
Tagged JSON uses the same JSON-aware envelope scan for full parsing and prefix
projection, so marker strings inside argument data do not terminate a call.
Duplicate top-level call fields and impossible argument-object starts fail
before provisional publication. Atem XML shares its complete/provisional scanner, retains an open parameter until
its JSON-or-text type is known, then streams fields without waiting for invoke
end. Quoted tag boundaries and exact, unique attributes are resolved before a
prefix is published; incompatible XML text cannot become invented empty args.

Realtime parameter-prefix admission returns a borrowed proof consumed by the
wire renderer. Prefixes can only append, repeated snapshots emit no duplicates,
and completed batch admission checks the raw streamed JSON representation and
its value-equivalence to the authoritative protocol parse before argument-done
or pending harness-call registration. Raw spelling survives canonical parse
reordering/escape normalization. The existing raw byte, call-count, identity and
response-text bounds remain enforced.

The response owns stable wire/model call identity bindings. Early explicit IDs
stay unchanged; a late model ID cannot rename an already published item. The
conversation owns pending wire-ID to complete model-call mappings, so an external
result uses its wire ID while the authored tool-result prompt receives the model
ID. Cancellation/limits do not register unfinished calls. Unfinished limit items
are incomplete and omit argument-done events, while response-level token-limit
status remains completed. Tool execution stays exclusively in the harness.

Forced tagged JSON schemas admit every ordering of the envelope's name, optional
ID and arguments fields, preserving required fields, ID limits and rejection of
extra fields. Parameter validation still uses the backend's supported schema
language rather than claiming a general JSON Schema validator. Broader realtime
forcing, pressure/overlap and device wire acceptance remain open.


Final CPU validation on the #2757 provider/ORT-constructor baseline: metadata
library 106 passed; engine library 1112 passed (1 inherited ignore); realtime
server selection 42 passed; full server library 621 passed (2 inherited ignores).
The Native wire cases include mid-argument waiting/cancellation, late model-ID
round trip with a template assertion, XML parameters before invoke close, and
all six envelope-field orders plus the absent-ID form under actual native masks.
Strict all-target metadata/engine/server Clippy passed with defaults, as did
strict server Clippy with `--no-default-features --features metrics,cuda-13000`.
Formatting/diff checks passed. These are CPU functional checks; device wire
acceptance and performance remain unclaimed. Broader acceptance rows above stay
open; this increment does not complete the overall compatibility objective.


## Tool parameter object-domain consistency

A real Native constrained-workflow counterexample showed `parameters: true`
allowing a higher-ranked array-argument token, only for the declared model
protocol to reject it after selection. Forced envelope grammars and realtime
parameter admission now share object-domain composition. Missing/null/true
schemas resolve to objects, while other documents retain their conditions and
root definitions/references under conjunction with the object requirement.
Existing object-only schemas stay unchanged; invalid allOf/documents and
contradictory constraints are not replaced with permissive fallbacks.

The former failing Native case now selects the lower-ranked valid object for
true, empty, array/object-union and null declarations. Full server validation:
625 passed, 2 inherited ignores. Grammar tests preserve local root references,
required/constant/additional-property conditions and impossible-schema failures.
Strict all-target engine/server Clippy and the explicit ORT reference server
configuration passed, as did formatting/diff checks. This is CPU functional
coverage, without device or performance claims; the overall acceptance list
remains open.


## Tool parameter reference scope

Forced tagged-JSON calls compile each parameter document as its own `%json`
rule. Its root URI, dialect, local pointers, anchors and recursive references
therefore have the same meaning as in standalone parameter admission. Tool
schemas may reuse resource IDs without sharing a registry. Absolute references
to the standalone default root also retain their target.

The outer Lark grammar owns only the call envelope: required name/arguments,
optional bounded ID, exact fields and every field ordering. Keys and identity
values use JSON rules; argument documents occur once and are never rewritten
or embedded under another schema root. The backend's supported schema language
and response byte bounds remain enforced, and no external retrieval is added.

Validation: the Native websocket regression covers local and absolute default-root
references, escaped JSON pointers and Draft 4. Compiled grammar cases additionally
cover anchors, relative resources, recursive roots, duplicate IDs across tools,
literal reference-like annotation data, JSON whitespace, invalid extra/duplicate
fields, required arguments and ID limits. The complete server library passed
627 tests (2 inherited ignores); strict all-target engine/server Clippy with
defaults and strict server Clippy with explicit ORT-reference features
`--no-default-features --features metrics,cuda-13000` passed. Formatting/diff
checks passed. Evidence is functional macOS ARM64 Native CPU, with no device or
performance claims; the broader compatibility acceptance list remains open.

Independent review additionally reproduced a required-tool union regression:
a backend-proven impossible parameter schema must not poison another viable
tool. Multi-tool forced selection now uses the same compiler's typed
unsatisfiability classification to prune only impossible branches. Invalid,
unsupported and unresolved schemas still fail with function context; specifically
selected or wholly impossible choices still fail. Single-tool forcing retains
its direct compilation path. Vendor provenance records the minimal typed-error
export used by this classification.

Final validation including the reviewed union-pruning fix: engine library
1115 passed (1 inherited ignore), server library 629 passed (2 inherited
ignores), and complete HTTP integration target 39 passed. The Native HTTP
case generates a valid tool call with an impossible competing schema. Strict
all-target default engine/server and explicit ORT-reference server Clippy,
format/diff and vendored compiler provenance checks passed. The compiler error
export is available in the bundled source; registry-only Rust publication
requires the patched compiler dependency to be published/upstreamed first.


## Standalone power-spectrogram audio features

Authored `spectrogram` operations now execute explicit power/Hann FFT geometry,
preserve declared analysis rates and produce `[batch,bins,frames]` tensors with
branch-local valid frames and masks. Shared streaming FFT/window analysis keeps
log-mel projection and normalization intact; compiled Spectrum/Mel kinds reject
Whisper normalization on linear power. Other DSP/window/centering conventions
remain open. See [audio preprocessing](AUDIO_PREPROCESSING.md#linear-spectrograms).

Independent review preserved checked mel dimension/allocation admission before
FFT planning. Validation: all 120 preprocessing tests passed, including varied
geometry against an independent f64 DFT; 6 Native spectrogram/waveform/reuse
integration tests passed, including analytic spectrum energy at the actual head.
Full server library: 629 passed, 2 inherited ignores. Strict all-target default
preprocess/engine/server and explicit ORT-reference server Clippy, formatting and
diff checks passed. This is functional macOS ARM64 CPU coverage; no device or
performance claim and no completion of the broader acceptance list.


## Executable behavioral acceptance inventory

The [behavioral matrix](LLAMA_CPP_BEHAVIOR_MATRIX.md) accounts for all 55 pinned
method/path contracts and nine original acceptance categories, including explicit
external-harness exclusions and unfinished requirements. Its source-linked cases
are not fresh execution claims. The optional POSIX receipt runner discovers an
exact test identity before requiring one passed/zero ignored, binds clean source
revision/tree/content plus explicit/implicit Cargo configuration and environment
hashes, and invalidates evidence if inputs change. Timeout/interruption cleanup
terminates the owned local process group before returning to the shared lock.

New explicit Native CPU integration cases cover Responses and input-count aliases,
pooled embedding aliases and full typed SSE ordering/identity/terminal consistency.
The maintained browser driver accepts the chosen built binary and uses ephemeral
fixture/proxy ports. Complete device/model/control cross-products, original CLI
executable evidence and the other listed acceptance gaps remain open.

Validation on the post-spectrogram production tree: all 23 Python checker/runner
regressions and all four explicit Native CPU alias integration tests passed;
strict default/reference server Clippy passed. Fresh Chrome chat/persistence,
byte-exact completed replay, induced reader-disconnect recovery, genuinely active
16,384-token refresh recovery and Stop/worker release passed without page errors.
API-key onboarding rejected the wrong key, accepted and persisted the valid key,
and protected replay. Browser evidence records the built binary/driver hashes;
these fixture results do not close the full model/device/control acceptance list.


## Explicit spectrum projections and periodic windows

Standalone audio spectra now admit explicit power/magnitude and periodic
Hann/Hamming/rectangular selectors. Magnitude projects finite complex FFT values
with stable hypot, so squared-magnitude overflow does not reject representable
amplitudes. Log-mel keeps the original power/Hann arithmetic, mel resource
admission before planning and normalization; geometry/rate/branch/mask guards
are unchanged. Uncentered/complex and other DSP conventions remain open.

Validation: all 122 preprocessing tests passed, including independent f64 DFT
for each projection/window, odd/short/length-one windows, and finite-input
magnitude-vs-power overflow checks. Six Native audio tests passed; the spectrum
case executes all six combinations at multiple amplitudes/lengths against
analytic head outputs. Full server library: 629 passed, 2 inherited ignores.
Strict all-target default preprocess/engine/server and explicit ORT-reference
server Clippy, formatting and diff checks passed. Functional macOS ARM64 CPU
coverage only; no device/performance or whole-goal parity claim.


## Package-bound CLI behavioral evidence

The behavioral inventory now binds each Rust case to an explicit workspace
package manifest and enabled lib/integration target. The checker rejects
mismatched package identities, excluded packages, escaped source paths and
custom test harnesses; execution records the chosen package and target and
binds the package manifests into its receipt. The inventory contains 47 Rust
cases plus one browser case across the existing 55 contracts/nine categories.

Five CLI cases establish removed REPL rejection, the Native default, retained
generation CPU/context controls, transcription input/format/conditioning controls
and global profiling flag parsing. These parser assertions do not establish
end-to-end inference, emitted profiling files or packaged launcher behavior.

Validation: all 78 CLI library tests and 29 Python checker/runner regressions
passed. Strict all-target default CLI/server and explicit ORT-reference CLI
(`--no-default-features --features onnx-genai-server/metrics,cuda-13000`)
Clippy, inventory, Python lint, formatting and diff checks passed. No hardware
performance claim or broader acceptance completion is implied.


## Native duplex response-local constraints

Input-driven duplex responses now arm the admitted constraint on the native
owner after Configure and before their first Audio quantum. Arming performs no
workflow quantum and does not consume usage; subsequent Audio/Poll retains the
same grammar cursor. Active responses own immutable tool/schema snapshots. Mutable response
default updates affect the following response rather than current sampling or
result admission; tool definitions remain frozen after first input because they
participate in session-start conditioning. Dynamic tool-definition updates need
a separate conditioning/state design and remain open. Missing masks and unsupported forcing fail before PCM is
queued. Tool execution remains owned by the external harness.

The exact old-code Native counterexample was reproduced: a higher-ranked invalid
city token was selected before schema rejection, failing the intended mask
assertion (one failed, zero passed/ignored). Nine fixed-source regressions cover
first-token selection, updates/cancel/rearm, cursor progress, admission failure,
non-quantum arming, schema ownership, EOS masking/ignore-mask refusal and finite
turn versus unlimited duplex budget behavior. The dedicated driver requires
clean source trees, binds HEAD/tree/source/manifests/lock and Cargo inputs, and
checks those bindings before and after every exact discovered/executed test.
The broader compatibility acceptance rows remain open.


## Native CLI process-boundary evidence

The behavioral inventory now links five additional real Native CLI process
cases: generation and its JSON profile, emitted CPU trace spans, file/WAV-stdin/
raw-PCM transcription, authored MP3 feature windows, and language/prompt
projection. The inventory has 52 Rust cases plus its separate browser driver.
These fixture processes disable ORT discovery and check actual output/backend
facts. Profiling uses OS temporary directories rather than repository target
paths. Packaged console-launcher acceptance and the full model/device/control
cross-product remain open; parser and process evidence are distinct.


## Publishable maintained grammar dependency

The single audited compiler tree now has the owned package identity
`onnx-genai-llguidance`; engine imports use its canonical Rust name. Its exact
versioned release dependency precedes the engine in the publish graph, retaining
the original MIT source, archive hash and all 71 provenance records. There is no
registry/path switch or duplicate compiler implementation. Two token-range
vector constructions retain their values with lint-clean syntax.

The compiler source passed 56 unit and 2 public-surface tests, strict all-target
Clippy and 20 Python provenance/release/registry checks. An exact 77-file Cargo
archive was served through a temporary sparse registry and consumed without
path/git/patch overrides; its registry-only public tests passed locked/offline.
This proves the artifact surface, not a crates.io publication. The standalone
key-ownership regression uses legal raw DEL spelling; the engine normalizes
arbitrary JSON Unicode escapes before backend recognition. See
[compiler checks](GRAMMAR_COMPILER.md) for reproducible artifact evidence.

### Canonical named-parameter XML generation wire strategy

Forced `atem-xml@v1` generation uses exactly one canonical invoke envelope with
double-quoted `name` attributes and parameter bodies containing serialized JSON.
The five named entities `amp`, `quot`, `apos`, `lt`, and `gt` escape XML syntax;
numeric entities are not assumed to decode. Strings use JSON quotes, for example
`&quot;1&quot;` and `&quot;Paris&quot;`. Raw `1` denotes a number and is never
reinterpreted as a string. Existing unforced inbound XML parsing still accepts
its broader spelling, including raw `1x` as the string `"1x"`.

The prefix projector translates canonical XML into the existing logical tool
grammar. It retains the independent standalone parameter-schema roots. Partial
XML delimiters/entities represent a bounded union of possible logical prefixes,
checked using forks of the already-compiled byte cursor. Closing a parameter
requires actual JSON and immediately checks its permitted object successors;
`1</atem:parameter>` cannot continue a schema that requires the number `12`.
Every completed canonical envelope is also checked by the authoritative inbound
XML scanner. Generation does not promise every inbound raw-text spelling,
optional XML attribute, whitespace spelling, or multi-invoke sequence.

The compiler's opt-in root key policy generates only names explicitly declared
in the normalized root object's `properties` or `required`, with nonempty names
up to the inbound parser's UTF-8 byte limit. It follows normalized references at the
same root and leaves nested JSON keys unchanged. Schemas with no instances under this declared-name generation policy receive
a capability error only after the backend proves that restricted language empty;
this does not assert semantic unsatisfiability of the original schema; invalid/unsupported/resource errors remain distinct.
Controls and Unicode use the existing parser's actual string/entity semantics,
rather than imposing a new standard XML parser. Existing explicit output stops
and resource/token limits can still truncate a viable stream.

Custom `x-guidance` item/key separator regexes and custom whitespace regexes are
outside this canonical XML generation spelling and receive an explicit
capability error. The default JSON lexical language and other supported compiler
options remain attached to the original schema root.

Dynamic root-name XML forcing remains open. Inbound unforced XML continues to
accept dynamic parameter names under its existing admission rules. Nested JSON
objects inside declared parameters retain their original dynamic-key semantics.

Named parameter generation follows the compiler's existing schema-property
ordering. Inbound XML parameter ordering is unchanged. This generation spelling
preserves supported declared-field values without promising all inbound orders.

Policy-only absence of encodable declared fields uses a distinct typed backend
capability error. It cannot be mistaken for semantic schema unsatisfiability or
used to silently discard a selected tool. Unsupported original value schemas,
including optional/dynamic fields and referenced definitions, remain errors
before policy filtering.

The original normalized schema and pending definitions receive a complete
original-compiler preflight on the same builder before root generation policy
compilation. Thus an originally empty cardinality language remains semantic
unsatisfiability and can be pruned normally; a language emptied only by the named
parameter policy remains a distinct capability error. The caller's actual
compiler resource limits apply to both stages; schema references are not
retrieved a second time and no additional byte-token parser is built.

Constrained native cancellation is a terminal control: it executes with a full
all-false vocabulary mask rather than requiring another grammar continuation.
This preserves cancellation and rearming of incomplete envelopes without allowing
text/EOS to escape the existing Native output checks.
