# Decisions

> Current decision ledger. Full prior history through 2026-07-20T13:35Z is preserved in
> `.squad/decisions/archive/2026-07-20T13-35-00Z-decisions-pre-multistream.md`.

> Entries older than 2026-06-21T23:55Z are archived in `.squad/decisions/archive/2026-Q2.md` when present.

<!-- scribe-merge-2026-07-22T14-59-36+0000-wp-b-landed -->
## 2026-07-22 — WP-B optional-modality epic landed and clippy cleanup reconciled

<!-- source: .squad/decisions/inbox/rutger-clippy-cleanup.md -->
### Clear runtime-entry Clippy gates
**By:** Rutger
**Decision:** Landed `6f217a4` clears `-D warnings` for `onnx-genai`, `onnx-runtime-capi`, and `onnx-runtime-python`; tests now resolve the Cargo binary path at runtime, C API maps `RuntimeBroadcastIncompatible` exhaustively to `InvalidArgument`, and Python bindings keep the keyword API with narrow `too_many_arguments` allowances.

<!-- source: .squad/decisions/inbox/zhora-rutger-clippy-review.md -->
### Review clippy cleanup
**By:** Zhora
**Verdict:** 🟢 APPROVE
**Rationale:** Required Clippy and targeted test gates passed; the C-API mapping is covered without a catch-all, runtime binary lookup preserves Cargo profile/target selection, `GenerateOptions` construction keeps defaults, and the scoped Python allowances avoid public API churn.

<!-- source: .squad/decisions/inbox/sapper-wp-b3-revision.md -->
### Land WP-B3 v3 optional-modality admission
**By:** Sapper
**Decision:** Landed `3d84b9b` makes retained raw `GraphProto.input` authoritative for optional-port membership, dtype, rank, and dimensions; raw initializer names only classify graph-default closure, loader behavior stays unchanged, and admission tests cover missing optional ports, fallback mismatches, gated producers, required inputs, and raw symbolic shapes.

<!-- source: .squad/decisions/inbox/bryant-wp-b3-v3-review.md -->
### Review WP-B3 v3 optional-modality admission
**By:** Bryant
**Verdict:** 🟢 APPROVE
**Rationale:** Raw protobuf signatures, initializer/default separation, loader unchanged proof, architecture neutrality, mutation proof, fmt, clippy, and full `onnx-genai-ort` tests all passed; unrelated `tiny-qwen35-mtp` fixture naming was ignored as directed.

<!-- source: .squad/decisions/inbox/chew-wp-b3-review.md -->
### Preserve WP-B3 v2 rejection rationale
**By:** Chew
**Verdict:** 🔴 REJECT for Deckard's prior revision
**Rationale:** Membership/default classification had moved to raw graph inputs, but dtype/rank/static shape still came from loader IR values, so initializer-backed graph inputs could be falsely constrained by initializer shape. Sapper's landed v3 fixed this by deriving signatures directly from raw `ValueInfoProto`.

<!-- source: .squad/decisions/inbox/freysa-wp-b3-review.md -->
### Preserve WP-B3 initial rejection rationale
**By:** Freysa
**Verdict:** 🔴 REJECT for Coco's initial admission work
**Rationale:** Optional-port existence and fallback-shape checks used loader-projected `model.graph.inputs`; initializer-backed raw graph inputs were therefore falsely rejected and graph-default closure was lost. Later revisions moved validation to retained raw protobuf.

<!-- source: .squad/decisions/inbox/deckard-wp-b3-revision.md -->
### Record WP-B3 intermediate revision
**By:** Deckard
**Decision:** Deckard's revision fixed raw graph-input membership and graph-default classification while leaving `graph_builder.rs` unchanged, but review found rank/static shape still sourced from loader IR; it remains historical context, not the landed artifact.

<!-- source: .squad/decisions/inbox/coco-wp-b3.md -->
### Record WP-B3 initial implementation context
**By:** Coco
**Decision:** Coco added optional-port admission coverage for presence keys, fallback rank/static dimensions, mutually exclusive fallback/routed binding, gated producers, and required-input closure. The initial approach was superseded after raw-protobuf authority reviews.

<!-- source: .squad/decisions/inbox/cotton-wp-b2-review.md -->
### Review WP-B2 optional-modality engine runtime
**By:** Cotton
**Verdict:** 🟢 APPROVE
**Rationale:** `PipelineGenerateRequest.present`, absent-modality zero fallback, `when_present` plan gating, destination-key fallback caching, initialized zeros, backward compatibility, and 8 CPU E2E tests passed. Engine behavior stays metadata-only with no model or architecture dispatch.

<!-- source: .squad/decisions/inbox/mariette-wp-b2.md -->
### Land WP-B2 optional-modality engine runtime
**By:** Mariette
**Decision:** Engine runtime landed request presence sets, consistency validation, fixed/symbolic zero fallback creation, gated component/route skipping across plan families, and destination-endpoint fallback pooling. `cargo clippy -p onnx-genai-engine --tests -- -D warnings`, `cargo test -p onnx-genai-engine`, and `cargo build -p onnx-genai-ort` passed; crate fmt failure was baseline-only in unrelated files.

<!-- source: .squad/decisions/inbox/wallace-wp-b4-review.md -->
### Review WP-B4 Mobius optional-audio exporter
**By:** Wallace
**Verdict:** 🟡 APPROVE-WITH-NOTES
**Rationale:** Frozen optional-modality contract, generic emitter, rank adapter, absent shape, and Rust-schema compatibility passed. The only note was missing committed BF16 adapter regression coverage, which Joshi subsequently added.

<!-- source: .squad/decisions/inbox/joshi-wp-b4.md -->
### Land WP-B4 Gemma4 optional-audio export
**By:** Joshi
**Decision:** Mobius PR #419 emits `audio` presence, `embedding.io.optional_inputs.audio_features` with zero fallback `[0, config.hidden_size]`, `audio_encoder.when_present: audio`, and rank-2 masked audio features via a generic metadata emitter. Ruff, metadata, Gemma4 graph/adapter, dtype, and width-probe validations passed.

<!-- source: .squad/decisions/inbox/joshi-wp-b4-bf16.md -->
### Add WP-B4 BF16 adapter regression
**By:** Joshi
**Decision:** Added BF16 coverage for `test_gemma4_audio_encoder_strips_padding_in_graph`, including output dtype verification, closing Wallace's non-blocking note.

<!-- source: .squad/decisions/inbox/tyrell-wp-b-progress.md -->
### Update WP-B progress documentation
**By:** Tyrell
**Decision:** `docs/PROGRESS.md` now records WP-B1, WP-B2, and WP-B4 landings and originally marked WP-B3 as still in review; after `3d84b9b`, WP-B is fully landed and future docs should reflect WP-B3 closure.

<!-- source: .squad/decisions/inbox/taffey-fmt-fix.md -->
### Restore workspace rustfmt gate on main
**By:** Taffey
**Decision:** Reformatted the 89 files reported by workspace `cargo fmt --check` across 25 crates, restoring the formatting gate without logic changes and setting up the later Clippy cleanup.

<!-- scribe-merge-2026-07-22T14-59-36+0000-wp-b-landed-end -->

<!-- scribe-merge-2026-07-22T15-05-00Z-wp-b1-landed-inbox -->
## 2026-07-22 — WP-B1 optional-modality schema landing and inbox reconciliation

<!-- source: .squad/decisions/inbox/bryant-wp-b1-review.md -->
### 2026-07-22: WP-B1 optional-modality schema review
**By:** Bryant
**Verdict:** 🟢 APPROVE
**What:** The generic optional-input fallback and phase-presence schema is backward-compatible, architecture-neutral, fully covered, regenerated, and limited to WP-B1 mechanical schema integration.
**Evidence:**
1. **Schema correctness/backward compatibility:** `ModelIoSpec.optional_inputs` uses `#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]` (`schema.rs:626-630`), and `PhaseConfig.when_present` uses `default` plus `skip_serializing_if = "Option::is_none"` (`schema.rs:1363-1370`). The legacy branch of `optional_modality_schema_round_trips` (`schema.rs:2418-2431`) parses a document lacking both fields, observes empty/`None`, and compares the serialized YAML value to the original without emitted defaults. The full-document branch round-trips the new fields exactly (`schema.rs:2433-2465`).
2. **Generic/explicit contract:** Presence values are documented as opaque and validated through non-empty-string deserializers (`schema.rs:8-34, 632-641, 1363-1370`); no model/architecture dispatch or port-name inference was added. `TensorDimension` is explicitly either `Fixed(i64)` or `Symbol(String)`; deserialization rejects fixed values below zero and empty symbols (`schema.rs:662-694`). The only absent kind is explicit `Zeros`, serialized in snake case. Searches found no new model/vendor/architecture special case.
3. **Test non-vacuity:** The test exercises a legacy document, exact full-document round-trip, `Zeros` → `"zeros"`, and a parsed shape containing both `Fixed(0)` and `Symbol("sequence_len")`; it rejects `-1` and empty presence. Mutation proof: I temporarily changed the fixed-dimension guard from `value >= 0` to `value >= -1` and ran the exact test. It failed at `schema.rs:2471` with `negative fixed dimensions must be rejected` (`0 passed; 1 failed`, exit 101). I reverted the mutation and confirmed the review worktree was clean before gates.
4. **Exhaustive construction sites:** `rg 'ModelIoSpec\s*\{' crates` found only the type plus literals in `metadata/src/parser.rs:247` and `engine/src/native_decode.rs:2629`; both add only an empty `BTreeMap`. `rg 'PhaseConfig\s*\{' crates` found only the type plus two literals in `engine/src/pipeline.rs:4703,4710`; both add only `when_present: None`. No runtime behavior was introduced.
5. **Generated schema:** The committed root `schema/inference_metadata.schema.json` contains `AbsentInputKind`, `AbsentInputSpec`, `OptionalInputSpec`, `optional_inputs`, `when_present`, and `TensorDimension` with integer minimum 0/string minimum length 1. `committed_inference_metadata_schema_is_current` passed.
6. **Gate tails:**
   - `cargo fmt -p onnx-genai-metadata --check`: no output; exit 0.
   - `cargo clippy -p onnx-genai-metadata --tests -- -D warnings`: `Checking onnx-genai-metadata ...`; `Checking jsonschema v0.48.2`; `Finished dev profile ... in 5.25s`; exit 0.
   - `cargo test -p onnx-genai-metadata`: `test result: ok. 24 passed; 0 failed`; schema sync `committed_inference_metadata_schema_is_current ... ok`; `test result: ok. 1 passed; 0 failed`; doc tests 0/0; exit 0.
   - `cargo build -p onnx-genai-engine`: tail compiled `onnx-genai-ort` and `onnx-genai-engine`; `Finished dev profile ... in 13.17s`; exit 0.
7. **Scope discipline:** Merge-base diff changes only metadata `schema.rs`/`parser.rs`, mechanical engine construction sites, and the generated JSON schema. It does not modify `onnx-genai-ort` or `onnx-runtime-loader`, and searches show no engine consumption of `optional_inputs`/`when_present`; WP-B2/WP-C behavior remains out of scope. `git diff --check` passed.
**Why:** Every requested contract, compatibility, validation, construction-site, schema-sync, gate, and scope check passed. The mutation test demonstrates the key rejection assertion is effective rather than vacuous.

<!-- source: .squad/decisions/inbox/deckard-wp-c-rereview.md -->
### 2026-07-22: WP-C admission gate re-review (v2)
**By:** Deckard

**Verdict:** 🔴 REJECT

**Per-finding status**
1. **Resolved.** Temporal shape/name inference and stale-input rejection were removed. Unknown refresh semantics now fail open; the schema-blocker deferral is justified.
2. **Resolved.** External provenance is evaluated per port. The mixed routed plus request-supplied component regression passes.
3. **Resolved.** Admission no longer classifies generated inputs by tensor-name conventions. The `decoder.past_noise` regression rejects with the component-qualified port.
4. **Resolved.** `cargo fmt -p onnx-genai-ort --check` passes.
5. **Partially resolved.** Read, textproto parse, and binary model-load failures preserve the model path and underlying cause. However, unnamed graph input/output failures at `crates/onnx-genai-ort/src/pipeline_admission.rs:87-113` still omit the model path, contrary to the RULES §1 requirement that inspection errors include path and cause.

**Verification run by Deckard**
- `cargo test -p onnx-genai-ort --tests` — PASS
- `cargo test -p onnx-genai-ort --test pipeline_admission` — PASS (9/9)
- `cargo clippy -p onnx-genai-ort --tests -- -D warnings` — PASS
- `cargo fmt -p onnx-genai-ort --check` — PASS

**New defects / gate failures**
- The mandated architecture-name grep is not clean: the authoritative diff contains `tiny-qwen35-mtp` in `crates/onnx-genai-ort/tests/mtp_session.rs:12`. This is a formatting-only test-fixture reference, not architecture-specific admission logic, but it still fails the explicit clean-diff gate.
- Add path-preserving diagnostics (and a regression) for unnamed graph inputs and outputs.

The fail-open temporal/schema deferral is otherwise sound for WP-C: no unsupported name/shape inference remains, and unknown bindings are left to loud runtime diagnostics where the current schema cannot prove invalidity.

**Fix owner:** Gaff

<!-- source: .squad/decisions/inbox/deckard-wp-c-review.md -->
### 2026-07-22: WP-C load-time VLM admission review
**By:** Deckard
**Verdict:** 🔴 REJECT
**Revision owner:** Sapper must own the revision. Leon is locked out as the rejected artifact's author; Deckard is the reviewer and must not revise it.

## Findings

### 1. BLOCKER — stale-input classification is unsound and violates the explicit-metadata rule
**What:** `refresh_required_decoder_inputs` infers temporal semantics from symbolic-dimension intersections and fallback port names (`pipeline_admission.rs:420-475,784-790`) instead of declared metadata.

**Why:** This can both reject valid packages and miss the defect it claims to catch:
- If any decoder input omits the batch symbol, `batch` becomes a supposed sequence symbol. A valid prompt-cached conditioning tensor shaped `[batch, image_sequence, hidden]` is then rejected when fed by a `prompt_only` producer, although the engine explicitly supports cached prompt-only conditioning (`onnx-genai-engine/src/pipeline.rs:1561-1568,1869-1878`).
- If all non-scalar inputs share the primary sequence symbol, that symbol is removed as “common”; a secondary per-token input can remain stale without rejection. The test avoids this by giving `attention_mask` a different `total_sequence` symbol.
- Shape/name inference is not the explicit, inspectable metadata required by RULES.md §2.

**How:** Add explicit per-decoder-input temporal/binding semantics (for example, refreshed-every-step versus fixed prompt conditioning) and validate producer phase against those declarations. Add regressions for valid fixed conditioning plus an unbatched position input, and for a stale secondary sequence input when all relevant ports share the same sequence symbol.

### 2. BLOCKER — valid mixed external/routed components are rejected
**What:** Input closure treats an unbound port as externally supplied only when its entire component has no incoming cross-component edge (`pipeline_admission.rs:485-517`).

**Why:** The runtime accepts direct request tensors keyed by any `component.input_name`, and `component_inputs` checks that direct endpoint before routed dataflow (`onnx-genai-engine/src/pipeline.rs:72-99,1475-1495`). A valid component with one routed input and one request-supplied input is therefore rejected at load time. The gate has invented a component-level provenance rule absent from the metadata and runtime contract.

**How:** Declare external/generated/default/state/dataflow provenance per port and validate exactly one declared source. Add a valid test where a component consumes one edge-fed tensor and one external request tensor.

### 3. BLOCKER — name heuristics let required unbound inputs pass
**What:** When `model.io` is absent, `generated_inputs` classifies decoder inputs by names such as `input_ids`, `attention_mask`, `position_ids`, `past*`, and `cache_*` (`pipeline_admission.rs:577-588,784-808`).

**Why:** An unrelated required input such as `past_noise` is accepted as generated/stateful despite having no KV/state declaration or dataflow source. This misses the required-input defect class and violates RULES.md §2's requirement that assumptions be explicit metadata. The requested model-name grep is clean—there are no Gemma/Qwen/Phi/Llama/model-type hits—but semantic port-name dispatch remains.

**How:** Admission must rely only on `ModelIoSpec`, `positions`, KV/state pairs, strategy-generated ports, graph defaults, declared external inputs, and dataflow. Compatibility conversion must emit those facts or fail. Add negative tests for convention-looking but undeclared ports.

### 4. BLOCKER — required formatting validation fails
**What:** `cargo fmt -p onnx-genai-ort --check` exits 1. The changed `src/lib.rs` has a rustfmt ordering delta around `shared_kv_proposer`, `loader`, and `pipeline_admission`.

**Why:** The review contract requires rejection on fmt failure. The branch's older baseline also contains unrelated crate formatting deltas, and current main is not crate-fmt-clean, but the touched `lib.rs` is itself not formatted.

**How:** Format the touched integration and reconcile the required crate-level fmt check before re-review.

### 5. Error-quality finding — graph inspection discards the useful cause
**What:** `inspect_component_signature` maps every read/parse/load failure to the same message and drops the model path and parser/IO cause (`pipeline_admission.rs:66-83`).

**Why:** RULES.md §1 requires preserving resource path and causal context. “Could not be inspected structurally” does not tell whether the file is missing, unreadable, invalid protobuf, invalid textproto, or otherwise malformed.

**How:** Preserve the underlying error and component model path with contextual wrapping while avoiding URL/secret-bearing content. Other admission errors are generally component.port-named and actionable; no secret/URL leak was observed.

## Test assessment

The six new admission tests pass, use `onnx_std` IR builders, and assert meaningful endpoint/reason text for valid, unbound, stale, dtype, rank, and modality cases. They are tailored to the current heuristics and omit the false-positive/false-negative cases above. The compatibility suite no longer proves that any valid compatibility VLM package loads.

## Validation

- `cargo test -p onnx-genai-ort --tests`: PASS
- `cargo test -p onnx-genai-ort --test pipeline_admission`: PASS (6/6)
- `cargo clippy -p onnx-genai-ort --tests -- -D warnings`: PASS
- `cargo fmt -p onnx-genai-ort --check`: FAIL
- Existing valid VLM engine tests: PASS (3/3)
- Existing VLM server bundle tests: PASS (9/9)

<!-- source: .squad/decisions/inbox/deckard-wp-c-v3-review.md -->
### 2026-07-22: WP-C v3 re-review (finding #5)
**By:** Deckard

**Verdict:** 🔴 REJECT

## Per-item findings

1. **The two diagnostic strings now satisfy RULES.md §1.** The unnamed-input and
   unnamed-output errors include the component, the allowed filesystem model path,
   the underlying cause, why binding/dataflow cannot proceed, and explicit graph
   regeneration guidance (`crates/onnx-genai-ort/src/pipeline_admission.rs:87-94`,
   `:109-116`). No secret or URL is added.

2. **The fixtures are built through the requested IR API.** They use
   `Graph`, `Node`, `Model`, and `Model::to_proto`, and explicitly verify that the
   serialized graph port name is empty
   (`crates/onnx-genai-ort/tests/pipeline_admission.rs:101-132`).

3. **Blocking defect: the new tests are not regressions for the changed
   diagnostics.** Both tests deliberately trigger the pre-existing generic
   `"could not be loaded"` wrapper and assert only that unrelated error plus its
   path (`crates/onnx-genai-ort/tests/pipeline_admission.rs:160-167`,
   `:584-592`). Reverting the v3 changes at
   `src/pipeline_admission.rs:87-94` and `:109-116` would leave both tests green.
   Thus the tests are vacuous with respect to finding #5.

4. **The documented loader limitation is real, but it exposes dead admission
   branches rather than making the tests acceptable.** The loader silently skips
   empty-name graph inputs and outputs before constructing the IR
   (`crates/onnx-runtime-loader/src/graph_builder.rs:118-120`, `:143-146`).
   Consequently admission cannot reach either dedicated unnamed-port rejection,
   and a test-engineered `DataType::Undefined` on the named peer is what causes
   the observed load failure. Handle empty names at the retained-protobuf/loader
   boundary (or otherwise make the dedicated validation reachable), then assert
   the actual unnamed-input/output message, model path, and fix guidance.

5. **No new fmt, clippy, test, or architecture-name regression was found.**
   Findings 1-4 from the earlier review were not reopened.

## Verification

- `cargo fmt -p onnx-genai-ort --check` — exit 0.
  Tail: `EXIT_STATUS=0`
- `cargo clippy -p onnx-genai-ort --tests -- -D warnings` — exit 0.
  Tail: `Finished 'dev' profile ... in 0.17s`; `EXIT_STATUS=0`
- `cargo test -p onnx-genai-ort --test pipeline_admission` — exit 0.
  Tail: `test result: ok. 11 passed; 0 failed; ...`; `EXIT_STATUS=0`
- `cargo test -p onnx-genai-ort --tests` — exit 0.
  Tail: final `tokenizer` test passed; `test result: ok. 1 passed; 0 failed; ...`;
  `EXIT_STATUS=0`
- Admission-logic-only architecture grep — no matches (`grep` exit 1, expected
  for a clean result).

**Specific remaining defect:** the regressions at
`crates/onnx-genai-ort/tests/pipeline_admission.rs:160-167` do not execute or
verify the v3 diagnostics and mask the fact that those production branches are
unreachable after loader projection.

Gaff is locked out from revising this artifact after rejection. Since Leon,
Sapper, and Gaff have now each owned a rejected revision, escalate to Justin or
assign a new owner.

<!-- source: .squad/decisions/inbox/deckard-wp-c-v4-review.md -->
# 🟢 APPROVE — WP-C v4 load-time VLM pipeline admission gate

**Reviewer:** Deckard  
**Commit:** `f3fd686f12ac4b147154194a08fa54bc9fd1a05d`  
**Date:** 2026-07-22

## Findings

1. **WHAT — The raw-protobuf unnamed-port checks are reachable.**  
   **WHY —** `onnx_std::load_model` decodes the original `ModelProto` and stores it
   unchanged in `Model::source_proto` (`crates/onnx-std/src/model.rs:180-197`);
   `Model::to_proto()` returns a clone of that retained proto
   (`crates/onnx-std/src/model.rs:121-135`). This occurs before the execution
   projection drops empty graph input/output names in
   `crates/onnx-runtime-loader/src/graph_builder.rs:118-121,143-147`. The passing
   exact-message tests and mutation result empirically confirm that both new
   branches at `pipeline_admission.rs:99-118` execute.  
   **HOW —** No change required.

2. **WHAT — The two regression tests are non-vacuous and isolate only the intended
   malformed port.**  
   **WHY —** The fixtures use `onnx_std::ir::{Graph, Node, NodeId, DataType}` plus
   `Model`/`Model::to_proto`; all named peers are `Float32`. Each fixture adds
   exactly one unnamed `Float32` top-level input or output and asserts that the
   generated proto contains it (`tests/pipeline_admission.rs:101-145`). The tests
   assert the exact cause, full model path, and matching regeneration guidance
   (`:148-184,601-608`). Commenting out only the two raw-protobuf checks admitted
   both malformed fixtures, producing exactly two failures; restoring them
   returned all 11 tests to green.  
   **HOW —** No change required.

3. **WHAT — Both diagnostics comply with RULES.md §1.**  
   **WHY —** Each message states what is wrong, why execution cannot proceed,
   includes `path.display()`, and gives explicit graph/sidecar regeneration
   guidance (`pipeline_admission.rs:99-118`).  
   **HOW —** No change required.

4. **WHAT — The implementation remains model-architecture agnostic under
   RULES.md §2.**  
   **WHY —** The required architecture-name grep returned no matches. Validation
   is based only on ONNX graph structure.  
   **HOW —** No change required.

5. **WHAT — All requested gates pass on the reviewed commit.**  
   **WHY —** Formatting, clippy with warnings denied, and the complete admission
   integration test all succeeded. The worktree was restored to a clean
   `f3fd686f12ac4b147154194a08fa54bc9fd1a05d` after mutation testing.  
   **HOW —** No change required.

6. **WHAT — The `to_proto()` clone is acceptable.**  
   **WHY —** It is one bounded transient clone per component during model
   admission, not a per-token or steady-state execution cost. I found no
   correctness defect or demonstrated load-time regression that warrants
   blocking this fix.  
   **HOW —** No change required.

## Exact command tails

```text
$ cargo fmt -p onnx-genai-ort --check
FMT_EXIT_STATUS=0

$ cargo clippy -p onnx-genai-ort --tests -- -D warnings
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.17s
CLIPPY_EXIT_STATUS=0

$ cargo test -p onnx-genai-ort --test pipeline_admission
test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
TEST_EXIT_STATUS=0

$ rg -n -i 'qwen|gemma|phi|llama|mistral|deepseek|glm' crates/onnx-genai-ort/src/pipeline_admission.rs
RG_EXIT_STATUS=1

$ cargo test -p onnx-genai-ort --test pipeline_admission  # raw checks commented out
failures:
    admission_rejects_unnamed_graph_input_from_retained_proto
    admission_rejects_unnamed_graph_output_from_retained_proto
test result: FAILED. 9 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
MUTATION_EXIT_STATUS=101

$ cargo test -p onnx-genai-ort --test pipeline_admission  # checks restored
test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
RESTORED_TEST_EXIT_STATUS=0

$ git status --short && git rev-parse HEAD
f3fd686f12ac4b147154194a08fa54bc9fd1a05d
```

**Verdict:** v4 genuinely fixes both fatal v3 issues: admission now observes the
retained source protobuf before loader filtering, and the regressions fail when
that validation is removed.

<!-- source: .squad/decisions/inbox/gaff-wp-c-finding5-fix.md -->
### 2026-07-22: WP-C finding #5 fix (unnamed graph-port diagnostics)
**By:** Gaff
**What:** Updated both unnamed ONNX graph-input and graph-output admission diagnostics to include `path.display()` and retain explicit graph-regeneration guidance. Added separate unnamed-input and unnamed-output regression cases in `crates/onnx-genai-ort/tests/pipeline_admission.rs`, constructing the models through the `onnx_std` IR (`Graph`, `Node`, `Model`, and `Model::to_proto`). Commit: `60e75ef1db831910b36b4b1f27aee22a37304cbf`.
**Why:** RULES.md §1 requires inspection/admission failures to preserve the model path and underlying cause. The protobuf loader currently drops empty-name graph `ValueInfo` entries before the admission scanner can reach its dedicated unnamed-port branches, so the tests document that limitation and assert the model path on the closest reachable component-inspection rejection while verifying that the serialized input/output is genuinely unnamed.

Verification:
- `cargo fmt -p onnx-genai-ort --check` — PASS (exit 0, no output)
- `cargo clippy -p onnx-genai-ort --tests -- -D warnings` — PASS
- `cargo test -p onnx-genai-ort --test pipeline_admission` — PASS (11 passed, 0 failed)
- `cargo test -p onnx-genai-ort --tests` — PASS (81 passed, 0 failed)

Architecture-name grep on the added admission-logic diff (`gemma|qwen|phi|llama|mistral|deepseek`) — clean.

<!-- source: .squad/decisions/inbox/holden-wp-c-v4-fix.md -->
### 2026-07-22: WP-C v4 root-cause fix
**By:** Holden
**What:** Chose direction **B**. Pipeline admission now validates top-level graph input/output names in the retained raw `ModelProto` before scanning the loader's execution IR. Replaced the vacuous unnamed-port fixtures with valid IR-built models whose only defect is an extra unnamed graph input or output, and asserted the precise cause, filesystem model path, and regeneration guidance.
**Why:** `onnx_std::load_model` and `onnx_std::textproto::from_textproto` return `onnx_std::Model`, which retains the exact source protobuf and exposes it through `Model::to_proto()`. Admission therefore already has legitimate access to the raw graph without changing the loader contract. This is the smallest honest way to validate names before `onnx-runtime-loader/src/graph_builder.rs:118-121` and `:143-147` project empty-name ports out of the IR.

**Code changes:**
- `crates/onnx-genai-ort/src/pipeline_admission.rs:82-118` — obtain the retained `ModelProto`, require a graph, and reject empty top-level `GraphProto.input`/`output` names with component, model path, cause, execution impact, and fix guidance.
- `crates/onnx-genai-ort/src/pipeline_admission.rs:120-153` — scan the loaded IR only after raw-name validation; removed the unreachable IR-level unnamed-port rejection closures and documented the loader projection seam.
- `crates/onnx-genai-ort/tests/pipeline_admission.rs:101-184` — rebuilt unnamed-port fixtures exclusively with `ir::Graph`, `ir::Node`, `ir::Model`, and `Model::to_proto`; all named peers now use valid `Float32` types, eliminating the unrelated `DataType::Undefined` load failure.
- `crates/onnx-genai-ort/tests/pipeline_admission.rs:601-608` — renamed the two regressions to state that they exercise retained-protobuf admission.

**Non-vacuity proof:**
- Both tests assert the exact unnamed-input/output cause, the exact filesystem model path, and the corresponding explicit-name regeneration guidance.
- A mutation run removed only the new raw-protobuf input/output checks. Both fixtures were then admitted, so both tests failed at `expect_err`: `0 passed; 2 failed`, `MUTATION_EXIT_STATUS=101`. Restoring the production checks returned both tests to green. Thus reverting the claimed production behavior cannot leave either test passing.

**Verification tails:**
```text
$ cargo fmt -p onnx-genai-ort --check
EXIT_STATUS=0

$ cargo clippy -p onnx-genai-ort --tests -- -D warnings
    Checking onnx-genai-ort v0.1.0-dev.3 (...)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.87s
EXIT_STATUS=0

$ cargo test -p onnx-genai-ort --test pipeline_admission
test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
EXIT_STATUS=0

$ cargo test -p onnx-genai-ort --tests
test tiny_tokenizer_round_trip ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
EXIT_STATUS=0

$ rg -n -i 'qwen|gemma|phi[-_0-9a-z]*|llama|mistral|deepseek|glm[-_0-9a-z]*' crates/onnx-genai-ort/src/pipeline_admission.rs
RG_EXIT_STATUS=1 (1 means clean)
```

**Commit:** `f3fd686f12ac4b147154194a08fa54bc9fd1a05d`

**Residual risk:** `Model::to_proto()` clones the retained protobuf once per component during load-time admission, adding bounded transient load-time memory proportional to model protobuf size. No loader, runtime execution, or admission-name inference contract was expanded.

<!-- source: .squad/decisions/inbox/keaton-phase1-seam.md -->
### 2026-07-22: Split capture-region policy from kernel capture mechanism
**By:** Keaton
**What:** Phase 1 uses a per-node EP hook, `ExecutionProvider::plan_capture_region(node, shape_status) -> Option<StructuralCaptureDecline>`. The EP owns the ordered structural predicates: control-flow/sequence classification, then unresolved output shape, then unresolved input shape. The executor converts that structural result to the existing `CaptureDecline`, and only when the hook admits the node does it apply the existing kernel-cache checks in order: `KernelNotWarmed`, then the compiled kernel's `CaptureSupport` decline (`KernelCaptureUnsupported`). The executor continues to form maximal contiguous segments and enforce persistent graph-output bindings.
**Why:** The executor alone owns the shape-keyed compiled-kernel cache, so kernel warmth and concrete-kernel capture support cannot move behind an EP-only graph hook without changing ownership or behavior. A per-node structural annotation is the clean EP↔executor seam: it passes only the node plus resolved-input/output presence, keeps structural policy model-agnostic and EP-owned, and leaves cache/kernel inspection as executor mechanism. The combined precedence is exactly the pre-refactor order—host/sequence, unresolved output, unresolved input, unwarmed kernel, kernel decline—so every node produces the same `Option<CaptureDecline>`, including identical `SeamReason` and reason text.

<!-- source: .squad/decisions/inbox/leon-keaton-phase1-review.md -->
### 2026-07-22: Phase 1 partial-CUDA-graph EP-capture-hook refactor — INDEPENDENT REVIEW 🟢 GREEN

**By:** Leon (senior engine reviewer; independent — not the author)

**What:** Reviewed Keaton's Phase 1 refactor on `squad/keaton-ep-capture-hook` @ 3390ba6
(EP hook `plan_capture_region` + `StructuralCaptureDecline`/`CaptureRegionShapeStatus`;
executor `node_capture_reason` refactor). Verdict: **🟢 GREEN — safe to merge.**

Checklist results (all verified against merge-base e1eeae4, diff vs origin/main):

1. **Byte-identical precedence ✅** — Combined EP-hook + executor evaluation reproduces the
   pre-refactor `node_capture_reason` exactly. Short-circuit order preserved:
   host/sequence → unresolved-output → unresolved-input → kernel-not-warmed →
   kernel-capture-unsupported. The hook computes control-flow → output → input in that order;
   executor eagerly computes both shape-status booleans but that has no ordering side effect
   (hook returns by precedence). SeamReason mapping is 1:1 and reason STRINGS are character-for-
   character identical to the originals (verified against `origin/main` lines 2650–2712).
2. **Shape-status fidelity ✅** — `outputs_resolved = outputs.all(contains_key)` == old
   `!outputs.any(!contains_key)`. `inputs_resolved` match{Some→contains_key, None→true} exactly
   reproduces old `.map(...).unwrap_or(Some(vec![])).collect::<Option<Vec<_>>>()` (None input =
   present/empty). KernelKey input_shapes reconstruction (`map_or_else(Vec::new, expect)`) yields
   the identical shapes vector; the `.expect`/`assert!` are safe under the hook invariant.
3. **Model-agnostic ✅** — No model-name/architecture branching in the hook, its default impl,
   or `is_control_flow_or_sequence`. Classification is purely structural (op_type + ai.onnx domain).
4. **Default impl vs overrides ✅** — Only the trait default impl exists (grep: zero overrides).
   CPU and CUDA EPs both inherit it → stock EPs behave identically. New provider.rs
   `is_control_flow_or_sequence` op set == old `is_control_flow_op ∪ is_sequence_op` (If/Loop/Scan
   + 8 Sequence ops), same domain guard.
5. **Exhaustiveness/types ✅** — `structural_capture_decline` and `reason()` matches are
   exhaustive (no catch-all). New enum/struct are doc-commented and re-exported via lib.rs.
6. **Build/test/clippy ✅** — All pass:
   - `cargo build -p onnx-runtime-ep-api -p onnx-runtime-session` ✅
   - `cargo build -p onnx-runtime-session --features cuda` ✅
   - `cargo test -p onnx-runtime-session` ✅ (61 lib incl. new parity test + all integration)
   - new `ep_structural_plan_plus_executor_kernel_checks_matches_legacy_declines` ✅ — GENUINE:
     builds a 6-node graph and asserts refactored == an inlined copy of the legacy function AND
     the exact SeamReason sequence [HostControlFlowOrSequence, UnresolvedOutputShape,
     UnresolvedInputShape, KernelNotWarmed, KernelCaptureUnsupported, None]. Adversarially covers
     output-before-input precedence (node1 has BOTH unresolved → asserts Output wins) and
     control-flow-before-shape (node0 is `If` with unresolved shapes → asserts HostControlFlow wins).
   - `cargo clippy … -D warnings` ✅ and `--features cuda` ✅ (both clean)
   - `cargo test -p onnx-genai-engine --features native-backend --lib
     capture_fallback_emits_each_structured_decline_to_tracer` ✅ (1 passed)
7. **Segmentation unchanged ✅** — `plan_capture_segments` and the graph-output persistent-binding
   precondition are untouched by the diff.

**Advisory (non-blocking):** The refactor adds `assert!(inputs_resolved && outputs_resolved, …)`
after the hook admits a node, plus an `.expect` in the KernelKey shape reconstruction. For all
current EPs (default impl only) these never fire. They are an intentional seam-contract guard for
future EP overrides that might admit unresolved shapes; behavior is unchanged for stock EPs. Fine
to merge as-is; worth a doc note in the Phase 2 EP-override guidance.

**Why:** The seam matches design intent (docs/design-ep-partial-cuda-graph.md §9 Phase 1 / Open
Question #1 §10): structural policy moved into the EP hook, kernel mechanism (warmth + compiled
CaptureSupport) stays executor-owned, and segmentation is byte-identical. No precedence reorder,
no shape-status mismatch, no altered reason string, no model-name branching, all checks green.

<!-- source: .squad/decisions/inbox/leon-wp-c-admission-gate.md -->
### 2026-07-22: Add graph-structural pipeline admission before ORT session creation
**By:** Leon
**What:** PipelineModelDirectory now inspects every component's real ONNX input/output signature and rejects non-closed input bindings, prompt-only producers feeding sequence-dependent every-step decoder ports, dtype/rank-incompatible dataflow edges, and incomplete declared image preprocessing/vision construction before PipelineModels creates any ORT session. ONNX graph-input initializers count as defaults; declared KV/fixed state and runtime-generated sequence/mask/position/timestep inputs count as generated or stateful bindings.
**Why:** Multi-model sidecars can be structurally valid metadata while still being non-executable. The gate is model-agnostic: it uses only pipeline components, phases, strategies, dataflow, typed preprocessing declarations, explicit model I/O, and graph-derived names/dtypes/ranks/symbolic dimensions, with no model-family names or fixed architecture counts.

<!-- source: .squad/decisions/inbox/pris-wp-b1-schema.md -->
### 2026-07-22: WP-B1 metadata schema (optional-modality contract)
**By:** Pris
**What:** Added the generic optional-input fallback and phase-presence schema, updated all exhaustive construction sites with mechanical defaults, regenerated the committed JSON schema, and added serde round-trip coverage.
**Why:** Optional modalities require explicit metadata for absent tensors and conditional component execution without model-, architecture-, or port-name inference.

## Exact schema additions

- `ModelIoSpec.optional_inputs: BTreeMap<String, OptionalInputSpec>`
  - `#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]`
- `OptionalInputSpec { presence: String, absent: AbsentInputSpec }`
  - `presence` is enforced as a non-empty opaque string.
- `AbsentInputSpec { kind: AbsentInputKind, shape: Vec<TensorDimension> }`
- `AbsentInputKind::Zeros`
  - `#[serde(rename_all = "snake_case")]`; serializes as `"zeros"`.
- `TensorDimension::{Fixed(i64), Symbol(String)}`
  - Untagged bare integer/string serde representation.
  - Negative fixed dimensions and empty symbols are rejected.
- `PhaseConfig.when_present: Option<String>`
  - `#[serde(default, skip_serializing_if = "Option::is_none")]`
  - Enforced as non-empty when present.

Definitions: `crates/onnx-genai-metadata/src/schema.rs:518-695`,
`crates/onnx-genai-metadata/src/schema.rs:1359-1420`.

## Mechanical construction-site updates

- `crates/onnx-genai-metadata/src/parser.rs:273`
  - `optional_inputs: std::collections::BTreeMap::new()`
- `crates/onnx-genai-engine/src/native_decode.rs:2650`
  - `optional_inputs: BTreeMap::new()`
- `crates/onnx-genai-engine/src/pipeline.rs:4705`
  - `when_present: None`
- `crates/onnx-genai-engine/src/pipeline.rs:4712`
  - `when_present: None`

Re-ran the requested exhaustive-literal grep across `crates/`; no other
`ModelIoSpec` or `PhaseConfig` construction sites require updates.

## Round-trip test

`crates/onnx-genai-metadata/src/schema.rs:2417`
(`optional_modality_schema_round_trips`) proves:

1. A legacy document without either new field deserializes and serializes without emitting defaults.
2. A document containing `optional_inputs` and `when_present` round-trips exactly.
3. `AbsentInputKind::Zeros` serializes as `"zeros"`.
4. `TensorDimension` accepts both `0` and `"sequence_len"`.
5. Negative fixed dimensions and empty presence keys are rejected.

The generated `schema/inference_metadata.schema.json` was refreshed and its
schema-sync test passes.

## Verification tails

`cargo fmt -p onnx-genai-metadata --check`
```text
(no output; exit status 0)
```

`cargo clippy -p onnx-genai-metadata --tests -- -D warnings`
```text
    Checking onnx-genai-metadata v0.1.0-dev.3 (/home/justinchu/onnx-genai-pris-wp-b1/crates/onnx-genai-metadata)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.06s
```

`cargo test -p onnx-genai-metadata`
```text
test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s

     Running tests/schema_sync.rs (target/debug/deps/schema_sync-d71939150098efe1)

running 1 test
test committed_inference_metadata_schema_is_current ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

   Doc-tests onnx_genai_metadata

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

`cargo build -p onnx-genai-engine`
```text
   Compiling onnx-genai-metadata v0.1.0-dev.3 (/home/justinchu/onnx-genai-pris-wp-b1/crates/onnx-genai-metadata)
   Compiling onnx-genai-preprocess v0.1.0-dev.3 (/home/justinchu/onnx-genai-pris-wp-b1/crates/onnx-genai-preprocess)
   Compiling onnx-genai-kv v0.1.0-dev.3 (/home/justinchu/onnx-genai-pris-wp-b1/crates/onnx-genai-kv)
   Compiling onnx-genai-scheduler v0.1.0-dev.3 (/home/justinchu/onnx-genai-pris-wp-b1/crates/onnx-genai-scheduler)
   Compiling onnx-genai-genai-config v0.1.0-dev.3 (/home/justinchu/onnx-genai-pris-wp-b1/crates/onnx-genai-genai-config)
   Compiling onnx-genai-ort v0.1.0-dev.3 (/home/justinchu/onnx-genai-pris-wp-b1/crates/onnx-genai-ort)
   Compiling onnx-genai-engine v0.1.0-dev.3 (/home/justinchu/onnx-genai-pris-wp-b1/crates/onnx-genai-engine)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.73s
```

## Git

- Branch: `squad/pris-wp-b1-schema`
- Commit: `c18807440c79172e73ac73a7924193cb71f01c3d`
- Pushed: `origin/squad/pris-wp-b1-schema`

<!-- source: .squad/decisions/inbox/roy-gemma4-e2b-reexport.md -->
### 2026-07-22: Gemma4 E2B corrected native-contract re-export
**By:** Roy
**What:** Re-exported `google/gemma-4-E2B-it` from Mobius `main` commit `640c1cb` using task `gemma4`, CPU-targeted optimization, fp16 weights, and `--runtime onnx-genai`. The emitted metadata closes over all four ONNX component graphs and passes all five requested contract checks.
**Why:** PR #418 changed native VLM metadata emission from an incomplete prompt-only contract into a graph-derived executable contract. This re-export verifies the merged producer against the real cached E2B checkpoint.

## Export

- **Status:** PASS
- **Mobius commit:** `640c1cb Emit executable native VLM contracts (#418)`
- **Task:** `gemma4` (`gemma4_unified` was not used)
- **Target:** CPU (`--ep cpu`)
- **Package:** `/home/justinchu/mobius/.scratch/gemma4-e2b-native`
- **Package size:** 11G
- **Metadata:** `/home/justinchu/mobius/.scratch/gemma4-e2b-native/inference_metadata.yaml` (19,625 bytes, 948 lines)
- **Export log:** `/home/justinchu/mobius/.scratch/gemma4-e2b-native-export.log`
- **Verification log:** `/home/justinchu/mobius/.scratch/gemma4-e2b-native-verification.txt`

The execution environment disallowed the requested `/tmp` scratch location, so the persistent package was written to Mobius's repo-local `.scratch` directory instead.

```bash
cd /home/justinchu/mobius
HF_HUB_OFFLINE=1 python3 -m mobius build \
  --config /home/justinchu/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03 \
  --task gemma4 \
  .scratch/gemma4-e2b-native \
  --dtype f16 \
  --runtime onnx-genai \
  --ep cpu \
  --optimize
```

The build exited 0 and reported:

```text
Saved decoder to .scratch/gemma4-e2b-native/decoder/model.onnx
Saved vision_encoder to .scratch/gemma4-e2b-native/vision_encoder/model.onnx
Saved audio_encoder to .scratch/gemma4-e2b-native/audio_encoder/model.onnx
Saved embedding to .scratch/gemma4-e2b-native/embedding/model.onnx
  inference_metadata: .scratch/gemma4-e2b-native/inference_metadata.yaml
```

No Mobius source files were modified.

## Relevant exact metadata excerpts

### Decoder sequence inputs

```yaml
- name: inputs_embeds
  dtype: fp16
  rank: 3
  shape:
  - batch
  - sequence_len
  - 1536
  source:
    kind: dataflow
    from: embedding.inputs_embeds
```

```yaml
- name: per_layer_inputs
  dtype: fp16
  rank: 3
  shape:
  - batch
  - sequence_len
  - 8960
  source:
    kind: dataflow
    from: embedding.per_layer_inputs
```

### Dataflow and every-step phases

```yaml
dataflow:
- from: embedding.inputs_embeds
  to: decoder.inputs_embeds
  dtype: fp16
  rank: 3
  device_transfer: false
- from: embedding.per_layer_inputs
  to: decoder.per_layer_inputs
  dtype: fp16
  rank: 3
  device_transfer: false
- from: vision_encoder.image_features
  to: embedding.image_features
  dtype: fp16
  rank: 2
  device_transfer: false
strategy:
  kind: composite
  stages:
  - name: run_vision_encoder
    strategy:
      kind: single_pass
      model: vision_encoder
    run_on: prompt_only
  - name: run_audio_encoder
    strategy:
      kind: single_pass
      model: audio_encoder
    run_on: prompt_only
  - name: run_embedding
    strategy:
      kind: single_pass
      model: embedding
    run_on: every_step
  - name: run_decoder
    strategy:
      kind: autoregressive
      decoder: decoder
    run_on: every_step
phases:
  decoder:
    run_on: every_step
  vision_encoder:
    run_on: prompt_only
  audio_encoder:
    run_on: prompt_only
  embedding:
    run_on: every_step
```

### Representative typed KV declarations

The metadata contains the corresponding key and value declarations for layers 0 through 14. These exact excerpts show both trailing dimensions:

```yaml
- name: past_key_values.0.key
  dtype: fp16
  rank: 4
  shape:
  - batch
  - 1
  - past_sequence_len
  - 256
  source:
    kind: stateful
    from: decoder.present.0.key
    update: append
```

```yaml
- name: past_key_values.4.key
  dtype: fp16
  rank: 4
  shape:
  - batch
  - 1
  - past_sequence_len
  - 512
  source:
    kind: stateful
    from: decoder.present.4.key
    update: append
```

The parsed per-layer K/V trailing dimensions were:

```text
layer:    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
head_dim: 256 256 256 256 512 256 256 256 256 512 256 256 256 256 512
```

Every key and value input and output is `dtype: fp16`, `rank: 4`.

### Vision endpoints

```yaml
vision_encoder:
  filename: vision_encoder/model.onnx
  type: vision_encoder
  io:
    inputs:
    - name: pixel_values
      dtype: fp16
      rank: 3
      shape:
      - batch
      - num_patches
      - 768
      source:
        kind: generated
        generator: image_preprocessing
    - name: pixel_position_ids
      dtype: int64
      rank: 3
      shape:
      - batch
      - num_patches
      - 2
      source:
        kind: generated
        generator: image_preprocessing
```

```yaml
outputs:
- name: vision_encoder.pixel_values
  content: pixels
  dtype: fp16
- name: vision_encoder.pixel_position_ids
  content: patch_coordinates
  dtype: int64
  pad_value: -1
```

## Requested verification

### 1. Decoder consumes both every-step embedding outputs — PASS

Evidence:

- `embedding.inputs_embeds -> decoder.inputs_embeds`, `dtype: fp16`, `rank: 3`.
- `embedding.per_layer_inputs -> decoder.per_layer_inputs`, `dtype: fp16`, `rank: 3`.
- Both decoder inputs declare their source as the matching embedding endpoint.
- Decoder phase is `run_on: every_step`.

### 2. Embedding emits/runs every step — PASS

Evidence:

- Embedding declares both `inputs_embeds` and `per_layer_inputs` outputs.
- `run_embedding` stage is `run_on: every_step`.
- `pipeline.phases.embedding.run_on` is `every_step`.

### 3. All 15 typed mixed-dimension K/V pairs — PASS

Programmatic metadata inspection found:

```text
kv_input_tensors=30
kv_output_tensors=30
kv_layers=15
layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
kv_head_dims=[256, 512]
```

All 30 state inputs and all 30 state outputs explicitly declare `dtype: fp16` and `rank: 4`. Layers 4, 9, and 14 retain head dimension 512; the other layers retain 256.

### 4. Typed vision endpoints — PASS

Evidence:

- `pixel_values`: fp16, rank 3, `[batch, num_patches, 768]`.
- `pixel_position_ids`: int64, rank 3, `[batch, num_patches, 2]`.
- The image preprocessor emits the exact qualified endpoints `vision_encoder.pixel_values` and `vision_encoder.pixel_position_ids` with matching dtypes.

### 5. No model-name/model-type hardcoded contract assumptions — PASS

Metadata grep:

```bash
grep -Ein '(^|[[:space:]])(model_name|model_type)[[:space:]]*:|google/gemma|gemma-4|E2B' \
  inference_metadata.yaml
```

Result: no matches.

A broader identity grep has one descriptive architecture value:

```text
13:  architecture: gemma4_text
```

This is the standard top-level architecture descriptor, not a pipeline/preprocessing/IO dispatch condition. The native metadata emitter itself contains no `gemma`, checkpoint ID, or E2B branch. Its sole `model_type` identifier is a generic helper parameter used to write component roles such as `vision_encoder`; all emitted ports, edges, phases, dtypes, ranks, shapes, KV geometry, and image bindings are derived structurally.

## Closure and consumer validation

The emitted declarations exactly matched the saved ONNX graph ports:

```text
closure_decoder=inputs_match:true outputs_match:true graph_inputs:34 declared_inputs:34 graph_outputs:31 declared_outputs:31
closure_vision_encoder=inputs_match:true outputs_match:true graph_inputs:2 declared_inputs:2 graph_outputs:1 declared_outputs:1
closure_audio_encoder=inputs_match:true outputs_match:true graph_inputs:2 declared_inputs:2 graph_outputs:2 declared_outputs:2
closure_embedding=inputs_match:true outputs_match:true graph_inputs:3 declared_inputs:3 graph_outputs:2 declared_outputs:2
```

The current onnx-genai native consumer also parsed and resolved the package:

```text
runtime_parse=PASS models=4 model_paths=4 metadata=/home/justinchu/mobius/.scratch/gemma4-e2b-native/inference_metadata.yaml
```

## Native E2E gap

The corrected emission itself is complete for the requested checks, and the native runtime loader accepts it. A normal image-only generation E2E remains blocked by the known optional-audio contract gap: this four-model checkpoint's embedding graph requires external rank-2 fp16 `embedding.audio_features`, while the audio encoder produces rank-3 features and is therefore correctly not connected by an incompatible guessed edge. A caller must provide compatible external audio features, or WP-B must add typed audio flattening/optional-modality/default semantics. Full ORT token generation was not claimed or run.

<!-- source: .squad/decisions/inbox/roy-gemma4-e2b-topology.md -->
### 2026-07-22: Gemma4 E2B emitted ONNX runtime topology
**By:** Roy
**What:** Exported the cached `google/gemma-4-E2B-it` checkpoint through Mobius task `gemma4` with fp16 CUDA-targeted optimization and captured the exact emitted ONNX and metadata contract. The real package is a **four-model** vision+audio+embedding+decoder topology, not the assumed three-model VLM topology.
**Why:** Runtime work must be driven by the actual graph ports, dtypes, ranks, phases, and dataflow, not by reading `_gemma4.py` or adding model-name branches. This artifact identifies which generic primitives already exist in onnx-genai and which producer/runtime contracts still block real E2B execution.

## Export result

- **Status:** succeeded; no Mobius source changes.
- **Mobius task:** `gemma4` (`Gemma4Task`).
- **Duration:** 86 seconds.
- **Output:** `/home/justinchu/gemma4-e2b-onnx`, 11,272,112,857 bytes (`du -sh`: 11G).
- **External data:** default ONNX external-data files (`model.onnx.data`).
- **Topology correction:** four ONNX models were emitted because the cached source config contains an audio tower: `vision_encoder`, `audio_encoder`, `embedding`, `decoder`.
- **Assistant note:** `google/gemma-4-E2B-it-assistant` remains cached at `/home/justinchu/.cache/huggingface/hub/models--google--gemma-4-E2B-it-assistant` for a later speculative-decoding test; it was not exported here.

Exact working command:

```bash
cd /home/justinchu/mobius
HF_HUB_OFFLINE=1 python3 -m mobius build --config /home/justinchu/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03 --task gemma4 /home/justinchu/gemma4-e2b-onnx --dtype f16 --runtime onnx-genai --ep cuda --optimize
```

The CLI accepts `f16`/`float16`, not `fp16`. The initially preferred `--model google/gemma-4-E2B-it` offline path could not resolve the cache because `refs/main` points to an incomplete snapshot; using the complete local snapshot through `--config` kept the build fully offline.

## Emitted ONNX I/O contract

| Model file | Direction | Tensor | Dtype | Shape |
|---|---|---|---|---|
| `audio_encoder/model.onnx` | input | `input_features` | `FLOAT16` | `[batch, time, 128]` |
| `audio_encoder/model.onnx` | input | `input_features_mask` | `BOOL` | `[batch, time]` |
| `audio_encoder/model.onnx` | output | `audio_features` | `FLOAT16` | `[batch, floor(floor(time/2 - 1/2)/2) + 1, 1536]` |
| `audio_encoder/model.onnx` | output | `audio_features_mask` | `BOOL` | `[batch, _d1]` |
| `decoder/model.onnx` | input | `inputs_embeds` | `FLOAT16` | `[batch, sequence_len, 1536]` |
| `decoder/model.onnx` | input | `attention_mask` | `INT64` | `[batch, past_seq_len + seq_len]` |
| `decoder/model.onnx` | input | `per_layer_inputs` | `FLOAT16` | `[batch, sequence_len, 8960]` |
| `decoder/model.onnx` | input | `past_key_values.0.key` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.0.value` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.1.key` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.1.value` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.2.key` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.2.value` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.3.key` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.3.value` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.4.key` | `FLOAT16` | `[batch, 1, past_sequence_len, 512]` |
| `decoder/model.onnx` | input | `past_key_values.4.value` | `FLOAT16` | `[batch, 1, past_sequence_len, 512]` |
| `decoder/model.onnx` | input | `past_key_values.5.key` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.5.value` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.6.key` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.6.value` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.7.key` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.7.value` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.8.key` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.8.value` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.9.key` | `FLOAT16` | `[batch, 1, past_sequence_len, 512]` |
| `decoder/model.onnx` | input | `past_key_values.9.value` | `FLOAT16` | `[batch, 1, past_sequence_len, 512]` |
| `decoder/model.onnx` | input | `past_key_values.10.key` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.10.value` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.11.key` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.11.value` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.12.key` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.12.value` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.13.key` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.13.value` | `FLOAT16` | `[batch, 1, past_sequence_len, 256]` |
| `decoder/model.onnx` | input | `past_key_values.14.key` | `FLOAT16` | `[batch, 1, past_sequence_len, 512]` |
| `decoder/model.onnx` | input | `past_key_values.14.value` | `FLOAT16` | `[batch, 1, past_sequence_len, 512]` |
| `decoder/model.onnx` | output | `logits` | `FLOAT16` | `[batch, sequence_len, 262144]` |
| `decoder/model.onnx` | output | `present.0.key` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.0.value` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.1.key` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.1.value` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.2.key` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.2.value` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.3.key` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.3.value` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.4.key` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 512]` |
| `decoder/model.onnx` | output | `present.4.value` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 512]` |
| `decoder/model.onnx` | output | `present.5.key` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.5.value` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.6.key` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.6.value` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.7.key` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.7.value` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.8.key` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.8.value` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.9.key` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 512]` |
| `decoder/model.onnx` | output | `present.9.value` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 512]` |
| `decoder/model.onnx` | output | `present.10.key` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.10.value` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.11.key` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.11.value` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.12.key` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.12.value` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.13.key` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.13.value` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 256]` |
| `decoder/model.onnx` | output | `present.14.key` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 512]` |
| `decoder/model.onnx` | output | `present.14.value` | `FLOAT16` | `[batch, 1, past_sequence_len + sequence_len, 512]` |
| `embedding/model.onnx` | input | `input_ids` | `INT64` | `[batch, sequence_len]` |
| `embedding/model.onnx` | input | `image_features` | `FLOAT16` | `[num_image_tokens, 1536]` |
| `embedding/model.onnx` | input | `audio_features` | `FLOAT16` | `[num_audio_tokens, 1536]` |
| `embedding/model.onnx` | output | `inputs_embeds` | `FLOAT16` | `[batch, sequence_len, 1536]` |
| `embedding/model.onnx` | output | `per_layer_inputs` | `FLOAT16` | `[batch, sequence_len, 8960]` |
| `vision_encoder/model.onnx` | input | `pixel_values` | `FLOAT16` | `[batch, num_patches, 768]` |
| `vision_encoder/model.onnx` | input | `pixel_position_ids` | `INT64` | `[batch, num_patches, 2]` |
| `vision_encoder/model.onnx` | output | `image_features` | `FLOAT16` | `[_d0*batch, 1536]` |

## Generated `inference_metadata.yaml` (verbatim)

```yaml
required_capabilities:
- kv_cache
- grouped_query_attention
model:
  attention:
    type: grouped_query
    num_attention_heads: 8
    num_kv_heads: 1
    head_dim: 256
    sliding_window: 512
  architecture: gemma4_text
  max_sequence_length: 131072
pipeline:
  models:
    vision_encoder:
      filename: vision_encoder/model.onnx
      type: vision_encoder
    audio_encoder:
      filename: audio_encoder/model.onnx
      type: audio_encoder
    embedding:
      filename: embedding/model.onnx
      type: encoder
    decoder:
      filename: decoder/model.onnx
      type: decoder
      tokenizer: tokenizer.json
  dataflow:
  - from: vision_encoder.image_features
    to: embedding.image_features
    dtype: fp16
    device_transfer: false
  - from: audio_encoder.audio_features
    to: embedding.audio_features
    dtype: fp16
    device_transfer: false
  - from: embedding.inputs_embeds
    to: decoder.inputs_embeds
    dtype: fp16
    device_transfer: false
  strategy:
    kind: composite
    stages:
    - name: encode_vision
      strategy:
        kind: single_pass
        model: vision_encoder
      run_on: prompt_only
    - name: encode_audio
      strategy:
        kind: single_pass
        model: audio_encoder
      run_on: prompt_only
    - name: fuse_embeddings
      strategy:
        kind: single_pass
        model: embedding
      run_on: prompt_only
    - name: decode
      strategy:
        kind: autoregressive
        decoder: decoder
      run_on: every_step
  phases:
    vision_encoder:
      run_on: prompt_only
    audio_encoder:
      run_on: prompt_only
    embedding:
      run_on: prompt_only
    decoder:
      run_on: every_step
```

## Runtime gap analysis

### Contract facts that replace assumptions

1. **The export is four-model, not three-model.** The embedding graph requires both `image_features` and `audio_features`. The audio encoder emits rank-3 `audio_features` plus `audio_features_mask`, while embedding expects rank-2 `audio_features`; the emitted metadata routes the former directly and ignores the mask. An image-only run therefore needs either a generically selected vision-only export or declared optional-modality/default/reshape semantics.
2. **Vision has two typed endpoints:** fp16 `pixel_values [B,N,768]` and int64 `pixel_position_ids [B,N,2]`. The generated YAML declares neither `preprocessing.image` outputs nor `pipeline.vision` expansion, so the server cannot construct or bind either endpoint from an image request.
3. **Embedding produces two sequence-dependent decoder inputs:** fp16 `inputs_embeds [B,S,1536]` and fp16 `per_layer_inputs [B,S,8960]`. The YAML routes only `inputs_embeds` and marks embedding `prompt_only`; `per_layer_inputs` is therefore absent at decoder binding and neither output is refreshed during decode.
4. **The optimized decoder has no `input_ids` and no `position_ids` input.** Its non-KV inputs are `inputs_embeds`, `attention_mask`, and `per_layer_inputs`, followed by 15 K/V pairs. A Gemma-specific position-ID workaround would be wrong for this artifact.
5. **Metadata is not an executable closure over graph inputs.** It omits explicit component `io`, the `per_layer_inputs` edge, image preprocessing/expansion, optional modality semantics, and exact graph-derived KV declarations. A producer-side contract validator should reject this sidecar before packaging.

### Known Leon VLM gaps, checked against current source and this export

| Area | Current onnx-genai support | What still blocks this package |
|---|---|---|
| Multi-endpoint vision inputs | **Generic primitive now exists.** The typed image program/server bundle resolves arbitrary named outputs, declared dtypes, packed/rank-3 tensors, and auxiliary coordinates (`state.rs` typed binding path; preprocess packed tests include Gemma-shaped pixels + positions). | Mobius emitted no typed image program, no endpoint bindings, and no `pipeline.vision` placeholder/expansion contract. The runtime must not fall back to literal `pixel_values` discovery or rank-4 assumptions. |
| Generic `every_step` upstream execution | **Generic primitive now exists.** The engine topologically runs declared `every_step` components and routes all outputs; `vlm_multibinding_pipeline_e2e` proves two refreshed outputs plus simultaneous raw IDs. | The sidecar incorrectly marks embedding `prompt_only` and emits only one of two embedding→decoder edges. Fix emission; do not reintroduce a one-output or model-name special case. |
| Decoder position-id rank/shape | **Generic declared position programs now exist.** | This optimized E2B decoder exposes **no position input**, so position generation is not a blocker for this model. Keep rank/axes metadata-driven for other VLMs; do not add a Gemma branch or invent `[1,S]`. |
| Optional modality/audio path | Server audio discovery is still literal and Float32-only, while this graph declares fp16 `input_features` plus a bool mask. Prompt component execution requires every graph input. | Either export a generic vision-only package, or add typed optional-modality execution/defaults and audio tensor bundles/transforms. Direct rank-3 audio→rank-2 embedding routing is not executable as emitted. |

All follow-up changes must obey `RULES.md` §2: derive behavior from metadata, graph I/O, shapes, dtypes, registries, and explicit configuration; no `gemma4`/model-name dispatch, fixed 35-layer/280-patch constants, or semantic port-name guessing.

## Ordered minimal generic follow-up work packages

1. **WP-A — Mobius executable-contract emission (small/medium, exporter owner).** Introspect every graph port and emit explicit model `io`, exact KV pairs, all dataflow edges (including `embedding.per_layer_inputs -> decoder.per_layer_inputs`), and phase `embedding: every_step`. Emit typed image transforms/outputs and token expansion from processor config. Add a closure validator: every required ONNX input must be external, generated, stateful, defaulted, or fed by exactly one edge; every declared edge must match dtype/rank.
2. **WP-B — Generic modality selection or optional-component/default semantics (medium, exporter + metadata/runtime).** For a vision request, either build a graph-derived vision-only package whose embedding has no audio input, or declare optional audio components and a typed zero/empty/default path. If audio remains, declare both fp16 features and bool mask plus the generic flatten/strip-padding transform required to satisfy the embedding rank-2 input. No model-family conditionals.
3. **WP-C — Package admission/load gate (small, runtime loader/server).** Fail before model loading when preprocessing/vision expansion is absent, a required component input is unbound, phase/dataflow leaves a decoder input stale, or an edge has incompatible dtype/rank. Errors must name the exact component.port and instruct regeneration with a corrected native sidecar.
4. **WP-D — Real E2B parity ladder (medium, validation owner; depends A-C).** With one fixed image/prompt, compare vision outputs, both embedding outputs, prefill logits, and one decode step against a Mobius/ORT reference; then perform the OpenAI image-chat smoke test. Assert that both sequence outputs refresh at decode and keep the emitted 15-pair mixed-256/512-head-dimension KV contract.

No new decoder position work is required for this emitted E2B graph. The architecture-neutral position-program implementation remains necessary for models whose ONNX graph actually declares higher-rank position inputs.

## WP-A corrected export verification

Re-exported from Mobius branch `vlm-wp-a-executable-contract` to
`/home/justinchu/gemma4-e2b-onnx-wp-a` with the same offline command and `--dtype f16`.
The persisted sidecar was revalidated against all four saved ONNX graphs:
`CLOSURE_VALIDATION=PASS`, 15 K/V layers (30 state inputs and 30 state outputs), mixed
trailing dimensions `[256, 512]`, and typed fp16 pixels + int64 patch coordinates.

```yaml
dataflow:
- from: embedding.inputs_embeds
  to: decoder.inputs_embeds
  dtype: fp16
  rank: 3
  device_transfer: false
- from: embedding.per_layer_inputs
  to: decoder.per_layer_inputs
  dtype: fp16
  rank: 3
  device_transfer: false
- from: vision_encoder.image_features
  to: embedding.image_features
  dtype: fp16
  rank: 2
  device_transfer: false
strategy:
  kind: composite
  stages:
  - name: run_vision_encoder
    strategy: {kind: single_pass, model: vision_encoder}
    run_on: prompt_only
  - name: run_audio_encoder
    strategy: {kind: single_pass, model: audio_encoder}
    run_on: prompt_only
  - name: run_embedding
    strategy: {kind: single_pass, model: embedding}
    run_on: every_step
  - name: run_decoder
    strategy: {kind: autoregressive, decoder: decoder}
    run_on: every_step
phases:
  decoder: {run_on: every_step}
  vision_encoder: {run_on: prompt_only}
  audio_encoder: {run_on: prompt_only}
  embedding: {run_on: every_step}
```

The old rank-3 `audio_encoder.audio_features` → rank-2 `embedding.audio_features` edge
is intentionally absent. The embedding port is explicitly declared as an external request
input until WP-B supplies optional-modality/default or typed audio flattening semantics.

<!-- source: .squad/decisions/inbox/roy-wp-a-contract-emission.md -->
### 2026-07-22: Emit graph-closed native VLM package contracts
**By:** Roy
**What:** Mobius native VLM metadata now emits typed `io.inputs`/`io.outputs` for every component directly from ONNX graph ports (name, dtype, rank, symbolic shape, and input source), routes every dtype/rank-compatible graph edge, marks sequence-producing upstream components `every_step`, declares their token-stream input, and validates the complete sidecar before writing it. Decoder KV input/output lists and geometry come from the real sparse graph ports; the Gemma4 E2B export produced 30 state tensors = 15 K/V layers with mixed 256/512 trailing dimensions. Typed image outputs are exact qualified endpoints derived from the structural processor registry: fp16 `vision_encoder.pixel_values [B,N,768]` and int64 `vision_encoder.pixel_position_ids [B,N,2]`, with patch-budget transforms and coordinate-derived token expansion.

Before, Gemma4 E2B routed only `embedding.inputs_embeds`, ran embedding only during the prompt, omitted typed component ports/KV geometry/image bindings, and emitted an incompatible rank-3 audio-output → rank-2 embedding-input edge. After:

```yaml
dataflow:
- from: embedding.inputs_embeds
  to: decoder.inputs_embeds
  dtype: fp16
  rank: 3
  device_transfer: false
- from: embedding.per_layer_inputs
  to: decoder.per_layer_inputs
  dtype: fp16
  rank: 3
  device_transfer: false
- from: vision_encoder.image_features
  to: embedding.image_features
  dtype: fp16
  rank: 2
  device_transfer: false
strategy:
  kind: composite
  stages:
  - name: run_vision_encoder
    strategy: {kind: single_pass, model: vision_encoder}
    run_on: prompt_only
  - name: run_audio_encoder
    strategy: {kind: single_pass, model: audio_encoder}
    run_on: prompt_only
  - name: run_embedding
    strategy: {kind: single_pass, model: embedding}
    run_on: every_step
  - name: run_decoder
    strategy: {kind: autoregressive, decoder: decoder}
    run_on: every_step
phases:
  decoder: {run_on: every_step}
  vision_encoder: {run_on: prompt_only}
  audio_encoder: {run_on: prompt_only}
  embedding: {run_on: every_step}
```

The incompatible audio edge is no longer guessed: `embedding.audio_features` is explicitly an external request input until optional-modality/typed-audio transforms are declared (WP-B).

**Why:** A sidecar is executable only when every required `component.port` has exactly one declared source: external, generated, stateful, defaulted, or one compatible dataflow edge. The producer-side validator checks the sidecar against every real graph input/output, rejects missing/duplicate sources and dtype/rank-mismatched edges with WHAT/WHY/HOW errors naming the exact endpoint, and is invoked before YAML serialization. All behavior is derived from graph I/O, shapes/dtypes, processor configuration, and structural registries; there is no model-family dispatch, fixed layer count, patch count, or KV dimension.

Mobius delivery: branch `vlm-wp-a-executable-contract`, commit `6ae7017`, PR
https://github.com/onnxruntime/mobius/pull/418.

<!-- source: .squad/decisions/inbox/sapper-wp-c-revision.md -->
### 2026-07-22: WP-C admission gate revision
**By:** Sapper
**What:** Revised `squad/leon-vlm-admission-gate` to remove symbolic-shape and port-name semantic inference, validate bindings per port, preserve ONNX model path and parser/I/O causes, and format the `onnx-genai-ort` crate. Temporal producer-phase rejection now fails open because today's metadata does not declare per-port refresh semantics. Binding closure uses only explicit `ModelIoSpec`, positions, KV/cross-KV/state declarations, strategy-generated ports, graph defaults, preprocessing outputs, and dataflow; components without an explicit decoder I/O contract remain eligible for request-supplied `component.port` tensors. Added regressions admitting cached prompt-only `[batch, image_sequence, hidden]` conditioning and mixed routed/request inputs, rejecting undeclared `decoder.past_noise`, and preserving model-load context. Updated the loader fixture to declare decoder I/O explicitly. Missing temporal/external-port schema facts are recorded separately in `sapper-wp-c-schema-blocker.md`.
**Why:** Deckard rejected the prior gate because shape/name heuristics falsely rejected valid cached conditioning, missed undeclared convention-looking ports, and imposed component-level provenance. The narrowed gate rejects only violations supported by explicit metadata or graph facts and otherwise prefers runtime diagnostics over speculative load-time rejection.

**Pushed branch HEAD:** `0b60958624a54e82ca48bc0fa0cea8f0b9388197`

**Verification:**
- `cargo test -p onnx-genai-ort --tests` — PASS
- `cargo test -p onnx-genai-ort --test pipeline_admission` — PASS (9/9)
- `cargo clippy -p onnx-genai-ort --tests -- -D warnings` — PASS
- `cargo fmt -p onnx-genai-ort --check` — PASS

<!-- source: .squad/decisions/inbox/sapper-wp-c-schema-blocker.md -->
### 2026-07-22: WP-C metadata facts intentionally left fail-open
**By:** Sapper
**What:** The current metadata contract has no per-port temporal semantic (fixed prompt conditioning versus refreshed every step) and no explicit list of request-supplied external pipeline ports. The revision therefore removes temporal stale-input rejection and treats otherwise-unbound ports as request-external unless an autoregressive decoder has an explicit `ModelIoSpec`; only then can an undeclared required decoder port be rejected.
**Why:** Shape symbols, port names, and component-level dataflow topology cannot prove temporal or external-binding semantics. Adding the missing fields requires metadata-schema and emitter work outside WP-C; failing open avoids false rejection while retaining sound closure checks where today's explicit decoder I/O contract proves a port has no source.

<!-- source: .squad/decisions/inbox/sebastian-wp-a-review.md -->
### 2026-07-22: Review of mobius PR #418 "VLM WP-A executable-contract emission"

**Reviewer:** Sebastian (independent; author Roy is locked out)
**Repo/branch:** onnxruntime/mobius `vlm-wp-a-executable-contract` @ `6ae7017` (base `00c8fac` / PR #416)
**Scope:** `src/mobius/integrations/onnx_genai/inference_metadata.py` (+374), `..._test.py` (+176)

## Verdict: 🟢 APPROVE (do NOT merge — review only)

Emission is structural/graph-derived, generalizes per model CATEGORY, and satisfies every WP-A requirement. Tests genuinely cover the new behavior across three distinct VLM categories. No model-name/architecture dispatch. Ruff clean, 40/40 tests pass.

## WP-A requirements — all verified

1. **`embedding.per_layer_inputs -> decoder.per_layer_inputs` edge** — ✓ Built by structural output→input name+dtype+rank matching across all components (`build_native_vlm_package_metadata`, lines 1158-1186), not hardcoded. Asserted present in `test_gemma4_routes_all_embedding_outputs` (test lines 416-421).
2. **Embedding phase `run_on: every_step`** — ✓ Derived: `_sequence_decoder_inputs` finds decoder inputs whose leading dims track `logits` dims (lines 812-832); any component feeding one is marked `every_step` (`downstream_to_decoder`, lines 1216-1244). Not name-forced. Asserted (test line 198, 204).
3. **Explicit typed `io` for ALL components incl. 15 KV pairs derived FROM THE GRAPH (mixed 256/512)** — ✓ `_port_metadata` emits name/dtype/rank/shape for every port; `_state_and_kv_pairs` pairs `past_key_values.<layer>.<role>` ↔ `present.<layer>.<role>` via regex + `config.layer_types`, raising on unclassifiable ports (lines 591-680). Trailing dims come straight from graph shapes. Test uses mixed `kv_head_dims=[8,16,8]` and asserts `past_key_values.1.key` shape[-1]==16 (test line 470) — proves dims are read, not hardcoded.
4. **Typed vision endpoints fp16 pixel_values + int64 pixel_position_ids** — ✓ Registry-driven `_resolve_image_program` matches structural rank/dtype signatures (`_match_packed_coordinates`: fp float rank-3 pixels + int64 rank-3 coords with last-dim 2). Dtypes taken from graph ports; endpoints named from `port.name`. Asserted (test lines 472-484), incl. `pad_value: -1` for coordinates.
5. **Producer-side closure validator** — ✓ `validate_executable_closure` (lines 913-1075) checks: every graph input has exactly one source (external/generated/stateful/defaulted/dataflow); every edge maps real output→input with matching dtype/rank; declared io matches graph ports exactly. Invoked before serialization (line 1334). Emits WHAT/WHY/HOW errors. Negative test removes the per_layer_inputs edge and asserts rejection naming `decoder.per_layer_inputs` (test lines 486-496).

## RULES.md §2/§2.1 compliance

- **No model-name/architecture branching.** `grep` for gemma/qwen/phi/llama/architecture==/model_type== in the source found only one unrelated TTS comment. Dispatch is on structural package roles (`vision_encoder`/`embedding`/`decoder` component keys) = model CATEGORY, which the topology note explicitly sanctions.
- **No fixed constants.** No hardcoded 35-layer/280-patch/256/512 KV dims; all derived from graph shapes and `config.layer_types`/processor config.
- **Assumptions explicit in metadata.** Unsupported vision signatures and unclassifiable state ports fail loudly with regenerate-instructions rather than guessing.
- **Audio edge correctly deferred to WP-B.** The incompatible rank-3 `audio_encoder.audio_features` → rank-2 `embedding.audio_features` edge is intentionally NOT emitted; `embedding.audio_features` is declared `external/request`. Asserted (test lines 191, 197).

## Test quality

Tests are non-trivial and category-diverse, proving generalization not overfit:
- `test_gemma4_routes_all_embedding_outputs` — full 4-model topology (vision+audio+embedding+decoder), mixed KV dims, per_layer edge, every_step, typed image outputs, closure negative case.
- `test_qwen_packed_grid_rank3_positions...` — area-grid processor, mrope, `linear_attention` layer types (sparse/replace state).
- `test_phi_routes_both_modality_gates...` — dynamic-HD crop-mask processor.
- Negative tests: unsupported signature, missing components, rank-3 positions requiring registry, equal-shape KV still declared KV.
- Three cached-processor tests match emitted programs against real processor configs.

Verified locally: `ruff check` + `ruff format --check` clean; `pytest inference_metadata_test.py` = 40 passed. (lintrunner 0.12.7 adapter env was broken — `lintrunner_adapters` not importable — so ran `ruff` directly per fallback; this is an environment issue, not a PR defect.)

## Non-blocking observations (do not require changes before merge)

- `vision_encoder` (`prompt_only`) and `decoder` (`every_step`) `run_on` are role-assigned rather than structurally derived, unlike embedding/audio. Correct for these categories today; a future refactor could derive all phases uniformly for robustness. Not blocking.
- Emission still branches on the literal component key `"audio_encoder"` for the `type` label (line 1238). This is a category label, not model dispatch; acceptable, but a role registry keyed on structure would be cleaner long-term.

## Recommendation

Approve for merge by an authorized non-author (coordinator or Justin). WP-B (optional-modality/typed-audio) and WP-C (runtime admission gate) remain the correct next work; nothing in this PR blocks them.

### Fold processed inbox notes
**By:** Scribe
**What:** Merged and cleared `bryant-wp-b1-review.md`, `deckard-wp-c-rereview.md`, `deckard-wp-c-review.md`, `deckard-wp-c-v3-review.md`, `deckard-wp-c-v4-review.md`, `gaff-wp-c-finding5-fix.md`, `holden-wp-c-v4-fix.md`, `keaton-phase1-seam.md`, `leon-keaton-phase1-review.md`, `leon-wp-c-admission-gate.md`, `pris-wp-b1-schema.md`, `roy-gemma4-e2b-reexport.md`, `roy-gemma4-e2b-topology.md`, `roy-wp-a-contract-emission.md`, `sapper-wp-c-revision.md`, `sapper-wp-c-schema-blocker.md`, `sebastian-wp-a-review.md`. Preserved active reference/in-flight files `keaton-native-specdecode-design.md`, `leon-vlm-scope.md`, `rachael-wp-b-optional-modality-design.md`, `zhora-deepseek-scope.md`.
**Why:** Completed implementation, review, revision, benchmark, and schema notes belong in the current decision ledger; active scope/design files remain in the inbox until their work lands.

<!-- scribe-merge-2026-07-22T12-00-00Z-phase0-7b-cudagraph -->
## 2026-07-22 — Partial CUDA-graph Phase 0 and Qwen2.5-7B CUDA-graph benchmark

<!-- source: .squad/decisions/inbox/deckard-luv-phase0-review.md -->
### 2026-07-22: Review verdict — Luv Phase 0 partial-CUDA-graph capture-path-kind (🟢 GREEN)

**By:** Deckard

**What:** Independent read-only review of `squad/luv-capture-pathkind` (commit 3c94a57) diffed against merge-base with `origin/main`. Changed: `executor.rs` (+`CapturePathKind`/`SeamReason` enums, `CaptureDecline.seam_reason: Option<SeamReason>`, seam-kind label in `log_capture_segmentation`, `CaptureDecline::node` now takes a `SeamReason`), `lib.rs` (re-exports + doc), `native_decode.rs` (+1 field in a test fixture), docs. **Verdict: 🟢 GREEN — safe to merge.**

**Why:**
1. **Byte-identical behavior — PASS.** Only removed string literal is the log-format line (now inserts `[{seam_label}]`); zero decline `reason` strings were removed or altered. Segmentation logic in `plan_capture_segments` is unchanged — `declines[pi].is_none()` still drives partitioning; boundaries pushed identically. Classification is derived *from* existing decline causes, not a replacement.
2. **Correct mapping — PASS.** All 5 per-node causes map correctly: control-flow/sequence→`HostControlFlowOrSequence`→`HostSeam`; unresolved output→`UnresolvedOutputShape`; unresolved input→`UnresolvedInputShape`; kernel-not-warmed→`KernelNotWarmed`; kernel-capture-unsupported→`KernelCaptureUnsupported` — the last four→`EagerDeviceSeam`. Graph-level persistent-device-binding hard-abort (`CaptureDecline::graph`) intentionally carries `seam_reason: None` ("graph-level hard preconditions"), which is correct — it is a whole-graph abort, not a per-node seam.
3. **Model-agnostic — PASS.** No model-name/architecture string branching; classification is purely structural (RULES.md §2/§2.1 respected).
4. **Exhaustiveness — PASS.** `SeamReason::path_kind` and `CapturePathKind::label` use exhaustive matches with no catch-all `_ =>`; `CapturePathKind`/`SeamReason` re-exported from `lib.rs` and doc-commented.
5. **fmt/clippy — PASS.** `cargo fmt -p onnx-runtime-session -- --check` clean; `cargo clippy -p onnx-runtime-session --all-targets -- -D warnings` clean; `--features cuda` clippy clean.
6. **Tests — PASS.** `cargo test -p onnx-runtime-session` = 60 passed, incl. new `seam_reasons_map_to_structural_capture_paths` (genuinely asserts all 5 reason→kind→label mappings + `CaptureRegion` label). `cargo test -p onnx-genai-engine --features native-backend capture_fallback_emits_each_structured_decline_to_tracer` = 1 passed.
7. **Log output — PASS.** Seam-kind label uses `boundary.seam_reason.map(SeamReason::label).unwrap_or("unclassified-seam")`; behind the verbose diagnostic flag; no existing test asserts on the literal log string, so no format-assertion breakage.

Conclusion: purely additive structural diagnostics, correct, model-agnostic, all gates green. Approved for merge.

<!-- source: .squad/decisions/inbox/gaff-qwen7b-cudagraph.md -->
### 2026-07-22: Qwen2.5-7B int4 CUDA-graph auto-enable benchmark
**By:** Gaff
**What:** Benchmarked Qwen2.5-7B int4 on one NVIDIA H200 at `bd3d95a` using `profile_native --ep cuda --prompt Hello --tokens 128 --warmups 2 --runs 3 --steady`, `ONNX_GENAI_DEVICE_KV=1`, and identical greedy decoding. Run A left `ONNX_GENAI_CUDA_GRAPH` unset; Run B set it to `0`. A companion 16-token diagnostic confirmed graph state and fallback counters.
**Why:** Validate that metadata/structure-driven CUDA-graph auto-enable generalizes beyond Qwen2.5-0.5B and Phi-4-mini without architecture or model-name keying.

| Metric | Run A — auto | Run B — forced eager |
|---|---:|---:|
| Median throughput | **231.73 tok/s** | **180.50 tok/s** |
| Median decode latency | **4.315 ms/token** | **5.540 ms/token** |
| Throughput speedup vs eager | **+28.38%** | baseline |
| Token-exact A/B | **Yes** | **Yes** |
| Capture engaged | **Yes** | No (explicitly disabled) |
| Zero fallbacks | **Yes** | Yes |
| Capture diagnostic | `enabled=true`, 1 capture, 14 replays, 0 fallbacks; 1 captured segment, 0 eager seams | `enabled=false`, 0 captures, 0 replays, 0 fallbacks |
| Kernels/token | N/A — `profile_native` does not surface GPU kernel-launch counts | N/A |
| GPU-busy | N/A — `profile_native` does not surface GPU utilization | N/A |
| Fraction of 4.8 TB/s ÷ 3.5 GB/token ceiling | **16.90%** | **13.16%** |

The 128-token outputs were identical token-for-token across A and B. Auto-enable generalized cleanly to Qwen2.5-7B: CUDA plus owned device KV selected whole-step capture automatically, with one captured segment, no eager seams, and zero fallbacks. The **28.38%** gain is smaller than Qwen2.5-0.5B's 87.7% and Phi-4-mini's 41.0%, as expected for a larger decode that spends more time streaming/dequantizing int4 weights and less proportionally on launch overhead, but it remains substantial. The simple peak-bandwidth roofline is about 1,371 tok/s; measured auto throughput is 16.90% of that ceiling, and this ratio should not be interpreted as pure bandwidth efficiency because int4 dequantization and compute also constrain decode.

<!-- source: .squad/decisions/inbox/luv-capture-pathkind.md -->
### 2026-07-22: Formalize partial CUDA-graph capture path kinds
**By:** Luv
**What:** Added `CapturePathKind` and `SeamReason`, attached optional seam classification metadata to `CaptureDecline`, propagated it through `CaptureSchedule` boundaries, and added seam-kind labels to `ONNX_GENAI_LOG_CAPTURE_SEGMENTS` output without changing capture partitioning or existing reason strings.
**Why:** Phase 0 of the partial-CUDA-graph EP-claim design requires structural, model-agnostic diagnostics that distinguish captured regions, eager device seams, and host seams before EP-owned planning is introduced.

| SeamReason | CapturePathKind |
|---|---|
| `HostControlFlowOrSequence` | `HostSeam` |
| `UnresolvedOutputShape` | `EagerDeviceSeam` |
| `UnresolvedInputShape` | `EagerDeviceSeam` |
| `KernelNotWarmed` | `EagerDeviceSeam` |
| `KernelCaptureUnsupported` | `EagerDeviceSeam` |

**Files touched:**
- `crates/onnx-runtime-session/src/executor.rs`
- `crates/onnx-runtime-session/src/lib.rs`
- `crates/onnx-genai-engine/src/native_decode.rs`
- `docs/design-ep-partial-cuda-graph.md`
- `docs/CUDA_GRAPH_CAPTURE.md`

**Verification:**
- `cargo fmt -p onnx-runtime-session` — PASS.
- `cargo test -p onnx-runtime-session seam_reasons_map_to_structural_capture_paths` — PASS (1 focused unit test).
- `cargo build -p onnx-runtime-session` — PASS.
- `cargo build -p onnx-runtime-session --features cuda` — PASS.
- `cargo test -p onnx-runtime-session` — PASS (all session unit, integration, and doc tests; one manual performance audit and one doc test remained ignored).
- `cargo clippy -p onnx-runtime-session --all-targets -- -D warnings` — PASS.
- `cargo test -p onnx-genai-engine --features native-backend capture_fallback_emits_each_structured_decline_to_tracer` — PASS (1 focused compatibility test).

### Fold processed Phase 0 and 7B CUDA-graph inbox notes
**By:** Scribe
**What:** Merged and cleared `deckard-luv-phase0-review.md`, `gaff-qwen7b-cudagraph.md`, `luv-capture-pathkind.md`. Preserved active scope/design files `zhora-deepseek-scope.md`, `leon-vlm-scope.md`, and `keaton-native-specdecode-design.md`.
**Why:** Landed implementation, independent green review, benchmark results, and progress-log updates belong in the current decision ledger; active scope notes remain in the inbox.

<!-- scribe-merge-2026-07-22T00-00-00Z-cudagraph-autoenable -->
## 2026-07-22 — CUDA graph auto-enable, GQA/VLM closure, and inbox reconciliation

### Land metadata-driven native CUDA graph auto-enable
**By:** Batty; reviewed by Leon 🟢
**What:** Merged `batty-45` to main as `610bde0`, auto-enabling whole-step CUDA graph capture in `native_decode.rs` whenever metadata and device bindings prove the native decode topology graph-safe. Environment precedence remains explicit-disable first, then explicit-enable, then metadata auto-enable; capture-safety fallback remains transparent.
**Why:** Gaff's H200 profile showed native decode was launch/CPU-dispatch bound rather than bandwidth-bound. Auto-enable turned proven graph-safe models on by default without model-name gates.
**Validation:** Leon reviewed `squad/batty-cudagraph-autoenable` 🟢 GREEN with 7/7 criteria passing. H200 results were token-exact with zero fallbacks: Qwen2.5-0.5B improved **441.49→828.54 tok/s (+87.7%)** and Phi-4-mini improved **67.32→94.91 tok/s (+41.0%)**.

### Close GQA `seqlens_k` exporter-shape blocker
**By:** Chew and Roy; reviewed by Deckard 🟢
**What:** Accepted canonical dense contiguous int32 `seqlens_k` shapes `[batch_size]` and `[batch_size, 1]`, normalized trailing singleton shape for capture signatures, and revised non-contiguous diagnostics to name both accepted shapes. Coordinator merged the fix to main as `f4484e7`.
**Why:** Real Foundry Qwen2.5-1.5B and Phi-4-mini exports provide `[batch_size, 1]`; scalar-only support did not unblock those models. Deckard's initial review was 🔴 only for diagnostic wording; re-review passed after Roy's correction.

### Record native CUDA benchmark and model-coverage outcomes
**By:** Gaff, Okonkwo, Chew, Deckard, Pris, Holden, and Tyrell
**What:** Folded the decode roofline and re-benchmark sequence: Qwen2.5-0.5B baseline native CUDA decode around 435 tok/s before CUDA graph auto-enable; Qwen2.5-1.5B first blocked by `[batch,1]` GQA lengths, then by M=5 prefill until the SwiGLU M>1 path landed; Phi-4-mini native CUDA validated on H200 after int4 zero-points and partial-RoPE fixes. The native CPU coverage census, DS-1 dynamic shape-chain validation, DS native E2E exact parity, MLA conformance guard, and progress-log updates are now represented here or in existing 2026-07-22 ledger sections.
**Why:** These notes establish which blockers were generic runtime gaps, which were already closed on main, and which measurements motivated CUDA graph auto-enable rather than model-specific dispatch.

### Fold VLM WP1 runtime-contract and CI notes
**By:** Rachael, Roy, Deckard, Leon, and Sebastian
**What:** Preserved the VLM WP1 review sequence: Leon rejected non-executable metadata revisions, Roy/Rachael moved preprocessing metadata toward explicit runtime contracts, Deckard fixed Qwen temporal patch packing order, and Leon re-reviewed the temporal-order fix 🟢. Sebastian made PR #416 schema/processor tests offline-safe by skipping unavailable local assets rather than failing CI.
**Why:** VLM metadata must be executable through declared processor/registry contracts, not shape-only JSON acceptance; cached-processor parity gates must be environment-aware.

### Fold partial CUDA-graph EP-claim design notes
**By:** Keaton; reviewed by Fact Checker 🟡
**What:** Recorded the proposed partial CUDA-graph capture design for EP subgraph claiming, with whole-step capture prioritized first and partial capture constrained by static seam-output and KV-append invariants.
**Why:** The design remains a follow-up proposal; whole-step capture is the immediate path for fixed-topology device-resident decode.

### Fold processed inbox notes
**By:** Scribe
**What:** Merged and cleared `batty-cudagraph-autoenable.md`, `chew-gqa-batch1.md`, `chew-model-coverage-census.md`, `coordinator-gqa-merge.md`, `deckard-ds1-shapechain.md`, `deckard-dsnative.md`, `deckard-gqa-batch1-review.md`, `deckard-gqa-rereview.md`, `deckard-mla-conformance-review.md`, `deckard-wp1-packer-fix.md`, `factchecker-keaton-epclaim-review.md`, `gaff-decode-profile.md`, `gaff-native-rebench.md`, `gaff-native-rebench2.md`, `gaff-native-rebench3.md`, `gaff-phi4-bench.md`, `gaff-phi4-benchmark.md`, `holden-partial-rotary.md`, `keaton-epclaim-design.md`, `keaton-epclaim-v2.md`, `leon-batty-cudagraph-review.md`, `leon-wp1-rereview.md`, `leon-wp1-review.md`, `okonkwo-gqa-decode-bench.md`, `pris-ds1-testreview.md`, `pris-gqa-scalar-seqlens-plan.md`, `pris-holden-rotary-review.md`, `pris-mla-conformance.md`, `rachael-wp1-revision.md`, `roy-gqa-batch1-revision.md`, `roy-wp1-revision.md`, `sebastian-mobius416-ci.md`, `tyrell-progress-0722.md`, `zhora-glm-l4-fix.md`. Preserved active scope/design files `zhora-deepseek-scope.md`, `leon-vlm-scope.md`, and `keaton-native-specdecode-design.md`.
**Why:** Completed implementation, review, benchmark, CI, and duplicate ledger artifacts belong in the current decision ledger; active scope notes remain in the inbox.

<!-- scribe-merge-2026-07-22T00-00-00Z-int4-zp -->
## 2026-07-22 — Phi-4-mini int4 zero-point blocker closure

### Close BLOCKER #3: explicit int4 zero-points in native CUDA fp16 GEMV
**By:** Sapper; reviewed by Holden 🟢
**What:** Merged commit `48de993`, threading packed per-block int4 `zero_points` plus `zp_row_bytes` through the native CUDA fp16 GEMV path so asymmetric int4 MatMulNBits models such as Phi-4-mini decode with explicit zero points. Null zero-point inputs preserve the existing symmetric zp=8 fast paths.
**Why:** Removes BLOCKER #3 with a structural, model-agnostic asymmetric int4 path while keeping M==1 capture safety, SM-portable arithmetic, and symmetric no-regress behavior.
**Validation:** Holden's non-author review passed all five criteria (SM-portability, capture-safety, symmetric no-regress, genericity, correctness). H200 validation passed 6/6 unit tests and 18/18 `matmul_nbits_gpu` integration tests, including explicit-zp CPU-reference and capture-replay coverage.

### Fold processed int4 zero-point inbox notes
**By:** Scribe
**What:** Merged and cleared `sapper-int4-zp.md` and `holden-int4-zp-review.md`.
**Why:** The implementation and independent green review are now represented in the ledger; unrelated active inbox artifacts remain untouched.

<!-- scribe-merge-2026-07-22T06-17-16Z -->
## 2026-07-22 — Native proposer contract and Qwen0.5B H200 benchmark

### Land metadata-driven native proposer execution contract
**By:** Batty; reviewed by Deckard 🟢
**What:** Land commit `96c79d0`, replacing hardcoded native proposer assumptions with metadata-driven `sequence_source` (`input_ids`/`inputs_embeds`), `kv_ownership` (`owned`/`shared`), explicit shared-KV ports, and semantic output roles (`logits_output`/`hidden_output`). Defaults preserve legacy token-id + owned-KV behavior; CPU shared-KV proposer execution is complete.
**Why:** Embedding-driven shared-KV assistants must be activated by declared contracts rather than model or tensor-name assumptions. CUDA device-buffer shared-KV aliasing remains explicitly scoped out until device binding alias/reference support lands.

### Record Qwen2.5-0.5B native CUDA H200 decode benchmark
**By:** Gaff
**What:** Qwen2.5-0.5B native CUDA decode on H200 measured **437.76 tok/s median** (**2.284 ms/token**), with coherent deterministic output. This is **15.2% faster** than the user's RTX 4060 380 tok/s reference and **2.83%** of the H200 weight-bound roofline.
**Why:** Establishes the current native-path performance point for the 0.5B model on shared H200 hardware and shows the path is coherent but still far from the weight-bound ceiling.

### Fold processed proposer and benchmark inbox notes
**By:** Scribe
**What:** Merged and cleared `batty-proposer-contract.md`, `deckard-batty-proposer-review.md`, and `gaff-qwen05-bench.md` when present.
**Why:** Landed implementation, review, and benchmark records belong in the ledger; active unrelated inbox artifacts remain in place.

<!-- scribe-merge-2026-07-22T05-52-21Z -->
## 2026-07-22 — Fused CUDA SwiGLU M>1 prefill merge

### Land generic fused gate/up SwiGLU M>1 prefill
**By:** Bryant; reviewed by Deckard 🟢
**What:** Land commit `97e0cb4` from `wt-swiglu-prefill`, extending `run_f16_gate_up_swiglu` so M>1 prefill runs the existing portable fp16 MatMulNBits tiled GEMM twice (gate into scratch, up into output) and then applies the existing fp16 SiluMul in place. The M=1 paired GEMV path remains unchanged and capture-safe; M>1 explicitly records `last_call_capture_safe=false`.
**Why:** The graph optimizer removes the unfused gate/up nodes, so the fused node must handle prompt rows as well as decode. Review confirmed bit-exact M=1 and M>1 coverage, SM portability, generic dispatch, correct capture flag behavior, and scratch lifetime safety; H200 rebuild plus 4 SwiGLU tests passed before merge.

### Fold processed SwiGLU inbox notes
**By:** Scribe
**What:** Merged and cleared `bryant-swiglu-prefill.md` and `deckard-bryant-swiglu-review.md`. Preserved unrelated active in-flight deliverables in `.squad/decisions/inbox/`.
**Why:** Landed implementation and review decisions belong in the ledger; active scope/review/revision artifacts should remain in the inbox until their work lands.

<!-- scribe-merge-2026-07-22T04:39Z -->
## 2026-07-22 — CPU SLN, stale-shape recompute, nbits prefill GEMM, and stale test merges

### Land fp16/bf16 CPU SimplifiedLayerNormalization
**By:** Deckard; reviewed by Gaff 🟢
**What:** Land commit `74a80ce` extending the CPU `SimplifiedLayerNormalization` kernel to accept Float16, BFloat16, Float32, and Float64 inputs/scales by widening to f32 for RMS-style accumulation and narrowing normalized plus optional inverse-standard-deviation outputs to the declared dtype. Dtype-parameterized tests cover last-axis and multi-axis shapes.
**Why:** Half-precision Foundry exports were rejected at `input_layernorm`; the generic widen/compute/narrow path removes that CPU decode gap without model, hidden-size, or shape gates.

### Land live runtime shape recompute for elementwise broadcasts
**By:** Pris; reviewed by Leon 🟢
**What:** Land commit `79b2bfc` recomputing standard multidirectional elementwise output geometry from concrete runtime input shapes before allocation, with actionable broadcast-incompatibility errors and coverage for a `ReduceSum -> Squeeze -> Cast -> Slice -> Add` data-dependent chain.
**Why:** Loader-resolved shapes can be stale for runtime view/data-dependent chains; using live broadcast shapes unblocks GLM-5.2-tiny indexing `Add` nodes while preserving strict ONNX equal-or-one semantics.

### Land portable fp16 MatMulNBits M>1 prefill GEMM
**By:** Sapper; reviewed by Batty 🟢
**What:** Land commit `54b49eb` adding a structural CUDA fp16-activation MatMulNBits prefill path for int4/int8 block-32 weights using a portable 16x16 tiled CUDA-core GEMM with fp32 accumulation, fp16 output, implicit/explicit zero points, tail handling, and f64-oracle parity.
**Why:** Native fp16 MatMulNBits previously rejected every M>1 prompt; the new path enables native multi-token prefill while preserving the unchanged capture-safe M=1 decode GEMVs.

### Refresh stale MatMulNBits unsupported-width coverage
**By:** Hudson
**What:** Land commit `764a208` updating the CPU MatMulNBits factory rejection test to use unsupported `bits=3`, assert the current `{2, 4, 8}` contract, and add positive factory coverage for `bits=8`.
**Why:** The old test treated now-supported `bits=8` as invalid and broke the CPU suite on main after int8 support landed.

### Fold processed landed inbox notes
**By:** Scribe
**What:** Merged and deduplicated `deckard-sln-fp16.md`, `gaff-sln-fp16-review.md`, `pris-stale-shape.md`, `leon-stale-shape-review.md`, `sapper-nbits-prefill.md`, `batty-nbits-prefill-review.md`, and `hudson-stale-nbits-test.md`. Preserved active or not-yet-main GQA/VLM/specdecode/model-coverage scope and revision artifacts.
**Why:** Landed implementation and review decisions belong in the ledger; active scope, review, and revision files should remain in the inbox until their work lands.

<!-- scribe-merge-2026-07-22T03:37:44Z -->
## 2026-07-22 — GQA scalar seqlens_k and int8 fp16 default-zp test merges

### Land GQA scalar `seqlens_k` support
**By:** Deckard; reviewed by Roy 🟢
**What:** Land commit `4ceaa7b` enabling declared unit-batch scalar `seqlens_k` for structurally detected GroupQueryAttention only. The contract remains strict-by-default (`PerBatchOnly`), rejects batch>1 scalar lengths, regenerates schema metadata, and keeps CUDA graph capture safe because validation is pure CPU shape inspection with no device allocation, D2H copy, sync, or pointer rebinding.
**Why:** ORT-GenAI GQA exports may provide scalar key sequence lengths for unit-batch decode; accepting that explicit metadata contract generically unblocks Phi-4-mini and Qwen2.5-1.5B decode without broad scalar coercion.

### Land int8 fp16 implicit-zero-point GPU parity coverage
**By:** Deckard; reviewed by Tyrell 🟢
**What:** Land commit `0d618de` adding fp16 int8 block-32 MatMulNBits CUDA parity coverage when the optional zero-point graph input is omitted, with the independent reference using default zp=128. The batch also retains explicit-zero-point coverage and verifies CUDA-graph replay is bit-exact with the preceding eager output on H200.
**Why:** The implicit/default zero-point path is distinct from explicit zero-points and needs direct regression coverage for fp16 output parity and capture determinism.

### Record VLM WP1 emission review lockout
**By:** Sapper; reviewed by Leon 🔴
**What:** PR #416 / VLM WP1 emission is blocked. Sapper is locked out of revising this artifact; a different agent must derive processor operations from explicit processor config/registry entries, make position/state roles registry/config-driven, add real cached-model HF processor comparisons, and fail unsupported signatures with actionable regenerate-or-register errors.
**Why:** Although schema/port validation and CLI/metadata tests passed, emitted preprocessing programs were not runtime-correct for Qwen3-VL, Gemma4, or Phi4MM, and some roles were inferred from shape/position rather than declared metadata.

### Fold processed inbox notes
**By:** Scribe
**What:** Merged and deduplicated `deckard-int8-zp-test.md`, `roy-gqa-review.md`, `tyrell-int8-zp-review.md`, and `leon-wp1-review.md` into this ledger. Preserved active research/scope artifacts in the inbox, including `zhora-deepseek-scope.md`, `leon-vlm-scope.md`, `keaton-native-specdecode-design.md`, `pris-gqa-scalar-seqlens-plan.md`, and `chew-model-coverage-census.md` if present.
**Why:** Review verdicts, lockouts, and landed implementation decisions belong in the current ledger; active research artifacts remain available for ongoing work.

<!-- scribe-merge-2026-07-22T09:30Z -->
## 2026-07-22 — DeepSeek shape-chain, MLA conformance, and active inbox fold

### Land DS-1 generic dynamic shape-chain propagation
**By:** Chew; reviewed by Rachael 🟢
**What:** Land commit `d653879` (reviewed work `chew-79`) extending generic runtime shape-chain propagation so a dynamically resolved `Slice` can feed `Unsqueeze` and subsequent broadcast/movement. `Unsqueeze` output rank is computed as input rank plus `len(axes)`, using the ONNX domain/opset registry and no node-name keying. Native Rust DeepSeek-V2 tiny CPU E2E now generates `[42, 237, 198, 2, 186, 81, 210, 149]`.
**Why:** Dynamic output sizing must remain model-agnostic and registry-driven while covering DeepSeek-V2 decode graphs that pass shape values through movement/broadcast chains.

### Land DS-3 MLA cached-decode parity coverage
**By:** Pris; reviewed by Tyrell 🟢
**What:** Land commit `8aba045` strengthening standard Attention/MLA tests for `qk_head_dim != v_head_dim` (192 vs 128), 3-D BSH, explicit head attrs, non-empty past K/V, prefill+decode+full-seq parity, GQA (`kv=2`) and MQA (`kv=1`), with an independent scalar SDPA oracle. CPU 33/33 and CUDA 23/23 pass.
**Why:** Cached decode must preserve asymmetric QK/V head-width semantics and parity across CPU/CUDA without relying on model-specific assumptions.

### Keep generic scalar `seqlens_k` GQA support explicit and unit-batch scoped
**By:** Pris and Deckard
**What:** Preserve the long-lived scalar-seqlens implementation plan, and fold Deckard's landed decision to emit `model.attention.key_sequence_lengths.scalar_broadcast: unit_batch` only for structurally detected ORT-GenAI GroupQueryAttention exports.
**Why:** Scalar key sequence lengths should be accepted only under a declared, validated unit-batch GQA contract, not as a broad shape coercion.

### Fold remaining processed inbox decisions and reviews
**By:** Scribe
**What:** Processed and deduplicated the non-preserved decision inbox notes. Key folded outcomes: block-32 int8 MatMulNBits CUDA support and review; VLM WP1/WP5/WP6 metadata/loader/server-bundle work and reviews; Gemma4 auxiliary output binding plus structural capture guard; H200 multi-model roofline and megakernel feasibility notes; KV logical-shape and fp16 GQA decode coverage; and DeepSeek validation/review records already represented by the DS-1/DS-3 entries above. Processed files:
- `ana-fp16-next-levers.md`
- `ana-h200-baseline-roofline.md`
- `ana-megakernel-feasibility.md`
- `ana-wave2-roofline-558.md`
- `ana-wave3-roofline-691.md`
- `batty-auxbind.md`
- `chew-ds1-shape-chain.md`
- `chew-ds3-mla.md`
- `chew-leon-auxguard-review.md`
- `deckard-gqa-fp16.md`
- `deckard-gqa-scalar-seqlens.md`
- `deckard-int8-matmulnbits.md`
- `gaff-ds3-review.md`
- `gaff-kv-review.md`
- `leon-auxbind-review.md`
- `leon-auxguard.md`
- `leon-kv-logical-shape.md`
- `leon-vlm-wp5-finalize.md`
- `leon-vlm-wp5-rebase.md`
- `leon-vlm-wp5-urlfix.md`
- `luv-vlm-wp5-rereview.md`
- `luv-vlm-wp5-rereview2.md`
- `luv-vlm-wp5-review.md`
- `luv-vlm-wp6-rereview.md`
- `luv-vlm-wp6-review.md`
- `luv-wp4-review.md`
- `pris-deepseek-e2e-val.md`
- `pris-ds3-mla-conformance.md`
- `pris-gqa-fp16-review.md`
- `rachael-ds1-review.md`
- `rachael-vlm-wp5.md`
- `roy-int8-matmulnbits-review.md`
- `sapper-glm-pr404.md`
- `sapper-vlm-wp1-emission.md`
- `sapper-vlm-wp6-fix.md`
- `sebastian-gemma4-perf.md`
- `sebastian-gemma4-reprobe.md`
- `sebastian-h200-multimodel-bench.md`
- `tyrell-ds3-review.md`
- `zhora-vlm-wp5-fix.md`
- `zhora-vlm-wp6.md`
**Why:** The inbox should retain only long-lived active research/scope artifacts while merged decisions live in the current ledger.

### Preserve active research and scope artifacts in the inbox
**By:** Scribe
**What:** Left `zhora-deepseek-scope.md`, `leon-vlm-scope.md`, `pris-gqa-scalar-seqlens-plan.md`, and `keaton-native-specdecode-design.md` in `.squad/decisions/inbox/`.
**Why:** These artifacts remain active references and should not be collapsed into the ledger yet.

<!-- scribe-merge-2026-07-21T23:55Z -->
## 2026-07-21 — VLM WP2/WP3, opset-24 CUDA, ScatterElements, and DS-1

### Land VLM WP0 metadata contract and source-compatible hotfix
**By:** Sapper; hotfix by Rachael; reviewed by Luv 🟢  
**What:** Land architecture-neutral typed multimodal metadata as commit `0f6ffbd`, then make additive WP0 fields `Default`-derived in hotfix `1b66d0f` so downstream literal construction sites keep building.  
**Why:** VLM routing must be metadata-driven rather than model-flavored, and optional multimodal fields must be source-compatible as the contract grows.

### Land native CUDA opset-24 ConstantOfShape, Gelu, and OneHot
**By:** Batty; reviewed by Pris 🟢  
**What:** Land commit `ea4036d` with generic native CUDA handlers for standard-domain ConstantOfShape, Gelu, and OneHot, preserving opset-aware semantics including negative-index behavior.  
**Why:** Opset-24 Gemma/DeepSeek-style graphs should stay native instead of falling back because construction, activation, or indexing handlers are missing.

### Replace VLM every-step model bindings with a generic Kahn executor
**By:** Sapper; reviewed by Luv 🟢  
**What:** Land VLM WP3 as commit `3aec9f3`, replacing model-flavored `EmbedsStepBinding` with a metadata-driven every-step executor that topologically schedules declared inputs, outputs, and dependencies using Kahn sorting.  
**Why:** Autoregressive VLM step execution must follow the declared metadata graph, not hard-coded architecture names.

### Land DS-1 generic runtime shape propagation with bounded materialization
**By:** Deckard; revision by Holden; rereview by Pris 🟢  
**What:** Land commit `1584fb3` for DeepSeek-V2 dynamic `Slice -> Unsqueeze` shape propagation, reusing the opset-aware shape-inference registry and permitting host materialization only after dtype, rank, and element-cap gates pass.  
**Why:** Runtime output sizing should reuse the same generic ONNX shape rules as kernels while preventing unbounded host copies from hostile or accidental shapes.

### Broaden native CUDA ScatterElements dtype coverage portably
**By:** Deckard; reviewed by Chew 🟢  
**What:** Land commit `5b01a01` covering fp16/bf16/fp32/int64 data with int32/int64 indices. Serial single-threaded reduction avoids half atomics, remains SM-portable, and is CUDA-graph capture-safe.  
**Why:** Valid ONNX ScatterElements graphs should not decline native placement solely because a supported data/index dtype pairing was absent.

### Land VLM WP2 native image processor after numerics and allocation fixes
**By:** Leon; revision by Sapper; final review Pris 🟢  
**What:** Land commit `5c48ba5` for generic metadata-declared image preprocessing. The accepted path preserves bit-exact `f32::from(v) / 255.0` Divide semantics (not reciprocal multiply; 126/256 bytes otherwise differ by 1 ULP), uses `try_reserve_exact` bounded allocations, rejects degenerate dimensions, and pins patch-size-2 HF fixtures by SHA.  
**Why:** VLM processors need multi-output metadata-declared preprocessing without legacy numerical drift or unbounded metadata-derived allocation.

### Preserve review lockouts from this segment
**By:** Scribe  
**What:** Record active lockout history: WP2 had Chew 🔴, locking Leon+Chew out until Sapper revised and Pris approved; WP4 had Gaff 🔴, locking Zhora+Gaff out while Batty revises; DS-1 had Gaff 🔴, after which Holden revised and Pris approved.  
**Why:** Rejected artifacts and reviewers stay locked out for their correction cycle, while accepted third-agent revisions become the authoritative artifacts.

### Treat CUDA 13 NVRTC on H200 as current-good
**By:** Scribe  
**What:** The CUDA crate pins `cudarc` `cuda-13000` with dynamic loading, and NVRTC 13 builds and runs GPU tests successfully on H200.  
**Why:** The older belief that this host requires CUDA 12.6 NVRTC is stale and should not guide future debugging or setup.

### Additional inbox decisions folded and deduped
**By:** Scribe  
**What:** Processed non-preserved decision inbox artifacts, deduping items already represented above or in the active ledger. Folded summaries:  
- `batty-clippy-hygiene.md` — 2026-07-21: Clear engine and ORT clippy warnings; By: Batty; What: Cleared all `cargo clippy --all-targets --features cuda -- -D warnings` diagnostics in `onnx-genai-engine` and `onnx-genai-ort` without changing public APIs or runtime logic..
- `brigitte-wp3-argmax-expose.md` — 2026-07-21: Expose and verify ORT multi-row device argmax; By: Brigitte; What: Added `DeviceSampler::argmax_rows(&self, DataType, usize, usize, usize) -> Result<Vec<u32>>`, implemented by `CudaSampler` through its existing `pub(crate) CudaSampler::argmax_rows` entry point. Coverage is f32, f16, an….
- `chew-flash-tc-adjudication.md` — Chew — Adjudication: `flash_attention_f16_tc` numerics dispute (Holden vs Deckard).
- `deckard-ep-transparency.md` — Decision: Production per-op executor spans + kernel-variant & capture-rejection reasons (native EP).
- `deckard-flash-tc-fix.md` — Deckard — flash_attention_f16_tc wmma parity investigation + permanent gate.
- `fenster-fixture-fix.md` — 2026-07-21: Treat binary/textproto twins as one model; By: Fenster; What: Chose Option A. `ModelDirectory` now collapses `<name>.onnx.textproto` when the same-stem `<name>.onnx` exists and prefers the binary; distinct model names remain ambiguous..
- `gaff-clippy-review.md` — 2026-07-21: Clippy hygiene review (Batty 2a0555b); By: Gaff; What: Approved commit `2a0555b` as pure Clippy hygiene. The six-file diff contains iterator idioms, redundant-clone removal in CUDA sampler tests, a let-chain, `then_some`, literal digit regrouping, a rustdoc blank line, and….
- `holden-attn-cliff-investigation.md` — Holden — Attention "cliff at ~pos 30" investigation (native CUDA, Qwen2.5-0.5B-int4).
- `holden-wp1-verify-review.md` — Review: WP1 — Native M=K verify + rewind primitive (option b) + (c)-ready guard.
- `hudson-fixture-fix-review.md` — 2026-07-21: loader same-stem fix review; By: Hudson; What: Binary/textproto twins are correctly treated as one logical model, with the binary preferred..
- `hudson-wp3-argmax-review.md` — Hudson review — WP3-prep multi-row device argmax.
- `joshi-rmsnorm-generic.md` — 2026-07-21: Select fp16 SkipRMSNorm warp half4 by structural capability; By: Joshi; What: Generalized `skip_rmsnorm_f16_warp_896` into `skip_rmsnorm_f16_warp_half4`. The kernel now receives and uses runtime `norm_size`, iterates `norm_size / (32 lanes * 4 halves)` half4 chunks per lane, divides the sum of sq….
- `kowalski-wave4-profile.md` — 2026-07-21: Wave-4 stacked CUDA profile; By: Kowalski; What: Treat wave-4 native CUDA fp16 decode as approximately 759 tok/s at 256 tokens and 789 tok/s at 1024 tokens, with about 227 launches/token, zero CUDA-graph fallbacks, and coherent decode..
- `pris-fusion-genericity-review.md` — Review: Fusion-genericity remediation (wt-fusion-generic @ 19b3b91).
- `pris-opset24-review.md` — Kernel Review — Native CUDA opset-24 op handlers.
- `pris-rmsnorm-review.md` — 2026-07-21: RMSNorm genericity review (Joshi 53d55e1); By: Pris; What: Reviewed branch `wt-rmsnorm-generic` @ 53d55e1, which replaces the.
- `ripley-wp2-native-driver.md` — WP2 — Native speculative driver (host-argmax accept).
- `sapper-fusion-genericity.md` — Decision: CUDA wave-4 fusions gate on structure + capability, not Qwen dims.
- `sebastian-multimodel-bench.md` — 2026-07-21: H200 native CUDA multi-model benchmark; By: Sebastian; What: Current `main` (`035ad9f`) measured Qwen2.5-0.5B int4 at **771.40 tok/s median** (766.49/773.62/771.40), 1 prompt token, 256 output tokens, 5 warmups per independent process, CUDA graph + device KV + strict CUDA, and ze….
- `solveig-wp1-verify-primitive.md` — Decision: WP1 — Native M=K verify + rewind primitive (option b) + (c)-ready guard.
- `wallace-ep-transparency-review.md` — 2026-07-21: EP transparency backbone review; By: Wallace; What: Deckard's per-op executor span backbone (`exec_plan_node`) is a genuine LIVE span, and the re-instrumented kernels attach kernel-variant + capture-status reasons to it in the real native decode path — my original dead-w….
- `wallace-wp2-driver-review.md` — WP2 native speculative driver — review.  
**Why:** The inbox should hold only living research artifacts; segment decisions belong in the active ledger.

## 2026-07-20 — CPU decode: resident pool and guarded GQA row parallelism

### Keep persistent M=1 decode-pool residency
**By:** Sapper; reviewed by Luv 🟢  
**What:** Run the whole native CPU M=1 forward inside one bounded decode-pool `install`, using a worker-local, nested, panic-safe RAII residency guard so each MatMulNBits call executes inline rather than reinstalling the same pool. `ONNX_GENAI_CPU_DECODE_THREADS=0`, prefill, default-feature-off, and CUDA behavior remain unchanged. Landed on main as `cbacb75`.  
**Why:** Qwen2.5-0.5B int4 decode improved about 3–6% with bit-identical tokens. This proves install crossings were avoidable but not the dominant remaining cost. Luv verified TLS isolation, Rayon semantics, deadlock safety, feature gates, and the CPU/build test matrix.

### Parallelize sufficiently large CPU GQA attention rows
**By:** Roy; reviewed by Luv 🟢  
**What:** Parallelize independent `(batch, query_head, query_sequence)` rows with one Rayon fork-join only above a 163,840 `row × key × head-dimension` work guard; retain serial execution below it. Each task owns a disjoint output row and private score buffer while preserving each row's reduction order. Landed on main as `c391327`.  
**Why:** Short decode regressed when parallelism was unconditional. Guarded parallelism improved 512-token decode throughput by 8.6%, reduced profiled GQA time by 13.9%, and cut 225-token prefill GQA time by 88.3%, with bit-identical 1-thread/8-thread greedy output. A future coverage follow-up may force exact serial/parallel comparison for a large ragged batch.

### Retain Tier-A GQA KV copy cleanup, defer shared append-only KV
**By:** Roy; regression coverage by Pris  
**What:** Borrow contiguous f32 past caches, remove a redundant owned clone, and replace scalar cache materialization loops with contiguous slice copies. Keep attention math and the SSA output contract unchanged. Pris added f16-widening and ragged-per-batch cache-materialization regressions.  
**Why:** The cleanup is bit-identical and removes avoidable work, but measured end-to-end decode was neutral within noise. True O(1)-append shared KV requires runtime aliasing/lifecycle changes and remains deferred.

### Do not land the decode fork-join granularity prototype
**By:** Deckard  
**What:** Revert the coarser 8/12-task MatMulNBits prototype and profiling probes; no commit landed.  
**Why:** Long runs regressed 7.1–8.4%. Post-residency profiling showed serial GQA at about 20.58 ms/token exceeded MatMulNBits at about 15.51 ms/token, so reducing projection task count removed steal slack rather than solving the dominant bottleneck. Revisit only as graph-level projection fusion, after GQA.

## 2026-07-20 — CUDA fused flash attention

### Fuse standard Attention only on measured-winning shapes
**By:** Rachael; reviewed by Chew 🟡  
**What:** Add an NVRTC tiled online-softmax backend behind `AttentionKernel`, including f16 WMMA with f32 accumulation and scalar f32/f16/bf16 support for MHA/GQA/MQA, causal/non-causal attention, and additive mask planes. Auto dispatch retains Phase-2a for decode, `D>128`, unsupported layouts/features, and measured-slower long spans. Landed on main as `a67b7a5`.  
**Why:** H200 f16 S512 improved about 1.53–1.60× and removed 48 MiB score scratch; S2048 regressed heavily when forced, so fallback is part of the design. Chew found the online-softmax merge, WMMA masking/synchronization, numerics, and dispatch sound. Non-blocking coverage remains for explicit Auto fallback gates, non-multiple-of-16 f16 head dimensions, and per-batch/per-head masks.

### Fuse GroupQueryAttention prefill with distinct physical and causal origins
**By:** Bryant, corrected by Rachael after Chew rejection; final review Chew 🟢  
**What:** Reuse the shared flash kernel behind `com.microsoft::GroupQueryAttention` for measured-winning prefill. Cache append and implicit RoPE use `total_length - key_sequence_length`; attention causal masking uses the distinct query start `total_length - query_sequence_length`. The final parity matrix covers 40 scenarios across f32/f16/bf16, MHA/GQA/MQA, fresh/cached/ragged, RoPE, local window, softcap, generic non-WMMA routing, large scores, unequal Q/K lengths, and Auto fallback. Landed on main as `94fa2b6`.  
**Why:** Bryant's first revision incorrectly reused the K append origin for queries when `Sq != Sk`; Chew rejected it and locked that artifact. Rachael's revision made the failing `Sq=2,Sk=4` case pass, tightened tolerances, and preserved exact present K/V. H200 fresh Q512 is about 1.31× faster with 48 MiB scratch saved; cached/large slower shapes fall back. The corrected artifact is approved and no active lockout remains.

## 2026-07-20 — Issue #40 Phase 1 distributed-runtime foundation

### Slice 1a: shared protocol trace + ticketed non-blocking host pressure
**By:** Tyrell; reviewed by Gaff 🟡  
**What:** Add the unpublished `onnx-runtime-protocol-trace` crate with public protocol envelopes/identities and a conformance-only independent `ReplayChecker`; add `HostGovernor` ticketed pressure accounting to `onnx-genai-scheduler`. All state transitions and trace linearization points commit under one short ledger lock; waits occur only on ticket-local condition variables after capacity is atomically charged. Landed on main as `0d1d265`.  
**Why:** The implementation conforms to `PressureProtocol.tla` invariants through an independent deterministic replay campaign and snapshot invariant checks. Gaff approved with two non-blocking issues—terminal-entry reaping and cancel-granted wake-after-unlock—which were folded into slice 1b. The TLC model gate is CI-deferred because Java/TLA tooling is unavailable locally.

### Slice 1b: Communicator + in-process backend + BufferOwnership registry
**By:** Tyrell; reviewed by Gaff 🟢  
**What:** Add unpublished `onnx-runtime-comm` with the async `Communicator` trait, synchronous reference `InProcessCommunicator`, and one-lock `OwnershipRegistry` over read/write lease sets. Dropping an operation handle detaches but does not release storage; terminal completion/abort releases leases, and freed allocation IDs remain tombstoned to prevent reuse/ABA. Reuse the slice-1a trace framework and independently replay `BufferOwnership` events. Landed on main as `e4d2883`.  
**Why:** Gaff verified exactly-one-owner, conflict, release, transfer, generation/ABA, non-blocking-lock, linearization, barrier, mailbox, and deterministic-conformance obligations. Slice 1b also reaps terminal pressure entries and moves all pressure wakeups after unlock. Non-blocking follow-ups for 1c include abort waking barrier waiters, barrier-map cleanup, and documenting tombstone growth.

### Slice 1c: one topology-wide collective ordering authority — IN PROGRESS
**By:** Tyrell  
**What:** Implement direct host rendezvous collectives behind a shared `CollectiveSequencer`; keep canonical submit order independent per communicator group, use one slot for count exchange plus all-to-all-v data, freeze reduction member order with checked arithmetic and per-contribution f16/bf16 rounding, and bound free tombstones with an exact window plus allocator-proven epoch floors.  
**Why:** This maps to `CollectiveOrdering.tla`: ranks may progress asynchronously without divergent order, groups do not acquire a false global enqueue order, completion stays rank-local, and abort freezes submissions before backend wakeup. This slice is not yet landed.

### Phase-1 deferred gates and remaining phases
**By:** Scribe  
**What:** Keep the TLC model gate CI-deferred. After 1c, Phase-1 slice 1d weight residency remains pending; issue #40 Phases 2–4 remain pending.  
**Why:** The landed Rust conformance harnesses provide deterministic implementation-side evidence, but do not replace the configured CI model check or the remaining distributed-runtime roadmap.

## 2026-07-20 — Issue #40 collective ordering completion

### Land slice 1c with serialized abort wakes and broad equivalence coverage
**By:** Tyrell; reviewed by Gaff 🟢  
**What:** Land all seven in-process collectives behind one canonical per-group `CollectiveSequencer`, deterministic member-order reduction, additive independent replay checking, bounded allocation tombstones, and rank-local completion. Abort now holds each rendezvous mutex while notifying its paired condition variable, closing the review's notify-before-park race. Distributed-equals-single-device bitwise coverage spans all_reduce, reduce_scatter, all_gather, broadcast, all_to_all, and all_to_all_v. Landed as `2ffb4e4` with follow-up `128440d`.  
**Why:** Gaff found the architecture and TLA refinement sound but blocked the original revision on a rare abort-path lost wakeup. Tyrell's deterministic waiter gate proved the fix, all comm/trace/scheduler suites passed, and the broadened equivalence matrix preserves fixed-rank-order determinism. TLC remains CI-deferred.

## 2026-07-20 — CUDA graph M4 capture-safety

### Own the CUDA graph lifecycle and exercise native decode replay
**By:** Rachael and Deckard; replay coverage by Pris; reviewed by Chew 🟢  
**What:** Serialize one CUDA graph lifecycle inside `CudaRuntime`, capture/replay only on its dedicated stream, invalidate on generation/binding lifecycle changes, and split capture-end from instantiate so failed instantiation cannot leak the intermediate `CUgraph`. Native decode remains flag-gated and strict-audit: unsupported graphs fall back eagerly. A capture-safe synthetic decoder proves token-exact eager/replay parity across reset, stable addresses, O(1) scalar uploads, two captures, sixteen replays, and zero fallbacks. Landed as `637e247`, `5470c01`, `dd2d807`, and `4755575`.  
**Why:** The first Qwen test exercised only fallback and was rejected as replay evidence. The final synthetic integration test executes the real `NativeDecodeSession::decode_cuda` state machine and resolved the M4 decode-loop review blocker without weakening the all-kernel capture audit.

### Gate MatMulNBits M=1 capture safety to the proven decode path
**By:** Bryant  
**What:** Remove trailing GEMV synchronizations and advertise MatMulNBits capture compatibility only after a successful no-`g_idx`, M=1 decode warmup; prefill, grouped-index, unwarmed, and configuration-changing paths remain ineligible. Runtime D2H helpers explicitly order after the EP stream. Landed as `a210703`.  
**Why:** The proven GEMV path is allocation-free, D2H-free, and synchronization-free, while the excluded paths dequantize, allocate, or validate on the host.

### Make fixed-shape GQA decode capture-safe with detect-before-consume metadata guards
**By:** Deckard, Rachael, and Bryant; reviewed by Chew 🟢  
**What:** Persist GQA scratch and remove the trailing stream sync (`dcb4f1b`); move advancing decode metadata reads and derived lengths on-device (`77829b9`); preserve warmup rejection and add on-device replay bounds checks with sentinel no-write behavior (`82c249d`). The final shared sticky error latch poisons subsequent replay steps after any violation and is polled immediately after logits D2H, before token consumption; explicit graph reset clears it. Landed final as `ca50bae`.  
**Why:** Earlier revisions were rejected for silent clamping and then for allowing a later valid replay to resume over a skipped KV row. The final detect-before-consume latch makes invalid metadata a hard, deterministic failure while valid fixed-capacity f32 one-token replay remains byte-identical and allocation-free.

### Make four normalization variants capture-safe
**By:** Roy; reviewed by Chew 🟢  
**What:** Remove trailing synchronizations from LayerNormalization, RMS/SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization, and SkipLayerNormalization. Keep SkipSimplified broadcast metadata in a mutex-protected, shape-keyed persistent cache and permit capture only after successful single-group warmup. Landed as `6184d82`.  
**Why:** The warmed decode paths now have stable metadata and no per-step allocation, free, upload, host read, or stream synchronization; the full CUDA suite and direct capture/replay byte-parity test passed.

### Bind elementwise capture eligibility to exact warmed signatures
**By:** Sapper and Deckard; reviewed by Chew 🟢  
**What:** Make supported unary and binary floating-point decode kernels capture-safe using persistent broadcast metadata and removed trailing synchronizations. Replace the initial boolean eligibility gate with mutex-protected exact dtype/entry and shape signatures; prefill, i64, errors, and signature changes remain ineligible. Landed final as `85b6f4e`.  
**Why:** Chew rejected the boolean gate because a warmed kernel could later execute a different dtype or shape during capture. Exact signatures close that TOCTOU while preserving numerics and the approved persistent-metadata design.

## 2026-07-21 — CUDA graph M4 end-to-end validation

### Real Qwen2.5 int4 decode captures with zero fallbacks
**By:** Rachael; reviewed by Chew; smoke correction by Pris 🟢  
**What:** Seed unresolved persistent external input/output physical shapes only during capture, keeping eager shape resolution and binding-signature invalidation intact. Constant/Shape metadata reuse and capture-safe integer Sub, ReduceSum, and Gather complete the real Qwen graph while device-side GQA/Reduce/Gather guards still latch errors before token consumption. After Chew caught stale fallback assertions, Pris updated the H200 smoke to require one capture, 62 replays, zero fallbacks, and no fallback reason. Landed as `dda3b25`, `13c094a`, and `42b71f7`.  
**Why:** Qwen2.5-0.5B int4 now captures end to end with token-exact graph ON/OFF parity and zero fallbacks: 70.33 versus 19.99 tok/s at 256 tokens (+251.8%), and 24.25 versus 11.73 tok/s at 1024 tokens (+106.7%). This validates the complete M4 capture-safety track on the real model.


## 2026-07-21 — Perf campaign reconciliation

### H200 native CUDA decode target and profiling baseline
**By:** Ana and Rachael  
**What:** Use ORT GenAI H200 Qwen2.5-0.5B int4 steady-state decode as the performance target: **657.34 tok/s** at 256 tokens (667.43 tok/s at 1024). Native progressed from about **73 → 145 → 192 → 201 tok/s**, but f32 Sq=1 GQA remained dominant: 70.5% of GPU time over 256-token decode and 82.7% over 16-token decode.  
**Why:** GEMV/argmax work is valuable but insufficient alone; the next high-leverage path is replacing serial f32 decode attention and then wiring/validating fp16 flash decode.

### Retile MatMulNBits decode GEMV and approve the result
**By:** Royb; reviewed by Wallace 🟢  
**What:** Retile the M=1 accuracy-level-4 symmetric block-32 CUDA MatMulNBits path, quantizing the f32 activation once with matching warp absmax/round/clamp/scale semantics. Wallace approved Roy's `5dbcbbb` retile.  
**Why:** This moved native decode from roughly 145 tok/s to about 192 tok/s while preserving numerics, but still leaves a large gap to Ana's 657 tok/s ORT target.

### Keep device-side greedy argmax after Batty's rebase repair
**By:** Mariette and Batty; reviewed by Joi 🟢  
**What:** Add allocation-free CUDA f32 greedy argmax with lowest-index tie behavior matching the host sampler. Joi rejected Mariette's rebased `c12e74f` because `DecodeCudaState::run_one_token` was called without the new `TraceContext`; Batty fixed the call and Joi approved `cdf62a0`.  
**Why:** The fixed path builds and measured about **200.97 tok/s**, removing the host argmax bottleneck without changing token selection.

### Land fp16 flash-decode as kernel-only first, then dormant dispatch wiring
**By:** Sebastian; reviewed by Bryant and Holden 🟢  
**What:** Add a capture-safe fp16 flash-decode GQA attention kernel as kernel-only commit `9c6f36b`, approved by Bryant. Wire it through a dormant fp16 dispatch branch at `521438e`, approved by Holden, gated by `q.dtype == Float16` and supported `(q_seq, dim)` while leaving the f32 path first and unchanged.  
**Why:** Split landing keeps the kernel independently reviewed and lets dispatch be enabled safely only for supported fp16 decode shapes.

### Direct fp16 activation × int4 GEMV remains a separate optimization track
**By:** Royb  
**What:** Prototype direct fp16-activation × int4 MatMulNBits GEMV on `wt-fp16-matmul` (`6a1daa2`) to avoid the int8 quantization pass.  
**Why:** This is distinct from fp16 flash attention and should be validated as a separate GEMV optimization before promotion.

### Sequence zero-copy design needs a second Deckard revision
**By:** Zhora and Deckard; reviewed by Luv 🔴  
**What:** Zhora's zero-copy Sequence tensors use shared allocation views with dtype/shape/layout/offset metadata. Luv rejected `ddae7d0`; Deckard closed the original public-output/runtime blockers with `SessionOutput::{Tensor, Sequence}` and related fixes, but Luv's re-review still rejected `cf8888b`.  
**Why:** The direction is acceptable, but remaining correctness/review blockers mean the Sequence zero-copy change is not approved yet.

### Runtime string tensors must use a dedicated host storage variant
**By:** Batty  
**What:** Represent runtime strings with `TensorStorage::{Raw, Strings(Vec<String>)}` or equivalent, expose safe `StringTensorView`/`StringTensorMut`, and never cast byte/device storage to `String`.  
**Why:** String tensors are host-owned structured values, not raw numeric buffers; exhaustive storage keeps executor behavior type-safe.

### PressureProtocol scaffold/fix path and current rejection state
**By:** Sapper, Roy, Deckard, and Pris; reviewed by Holden and Freysa 🔴/🟢 mixed  
**What:** Sapper/Roy added HostGovernor pressure envelopes and replay extension points; Holden rejected the first scaffold until actor ordering was scoped by `(HostId, ActorId)`, which Deckard fixed. Freysa rejected Sapper's HostGovernor revision, locking Sapper out and assigning the fix to Batty; Roy repaired release integrity by retaining authoritative allocations in `Claimed` and enforcing deterministic scheduling. Freysa's 2026-07-21 re-review still rejected `3207c25` because the branch/diff was not review-clean. Pris strengthened forged-release and cancellation synchronization regression tests.  
**Why:** Credit integrity and deterministic admission are the right design constraints, but the pressure implementation is not approved until reviewed from a clean branch with the fixed protocol evidence.

### Graph-capture transparency requires structured reasons across three axes
**By:** Coordinator and Gaff; reviewed by Chew  
**What:** All EPs must surface structured trace reasons for kernel non-selection and graph-capture non-capturability; transparency has three axes: op claim, kernel-variant selection, and capture support. Gaff added `CaptureSupport::{Supported, Unsupported { reason }}` and default compatibility adapters; Chew reviewed the structured reason-carrying design.  
**Why:** Silent bool declines make performance debugging impossible; traces must explain both variant choice and capture segmentation/fallback.

### Decouple CUDA EP claim from segmented graph capture
**By:** Coordinator and Tyrell  
**What:** CUDA EP should claim/run supported subgraphs even when only maximal segments are capturable, interleaving captured runs with eager CUDA runs for non-capturable nodes.  
**Why:** Capturability is an execution scheduling property, not an EP ownership property; partial segmented capture preserves CUDA placement without all-or-nothing fallback.

### Cross-platform support must include Windows ARM64
**By:** Coordinator; audit by Deckard  
**What:** Treat `aarch64-pc-windows-msvc` as a required target alongside Windows x64, macOS x86_64/arm64, and Linux x64. Deckard also flagged truthful CUDA selection, OS-aware library discovery, updated CUDA-12 CUDART candidates, pip/Conda NVIDIA discovery, and preventing Python from advertising CUDA while executing CPU.  
**Why:** Packaging and runtime probing must match the documented support matrix and actual execution provider behavior.

### Publishability of onnx-rs remains required
**By:** Leon  
**What:** Keep `onnx-rs` publishable to crates.io with package metadata and publish workflow coverage.  
**Why:** It is the ONNX standard-library crate for Rust in this workspace and must remain releasable.

### Capture-safe Sq=1 GQA decode kernel approved as prior f32 stepping stone
**By:** Sebastian; reviewed by Bryant 🟢  
**What:** Bryant approved `b6ada01`, a capture-safe warp-parallel Sq=1 GQA decode attention kernel for supported `head_dim <= 128` with zero CUDA-graph fallback.  
**Why:** This was a correct f32 decode-attention stepping stone before the later fp16 flash-decode path.

## 2026-07-21 — fp16 decode, transparent fallback, cross-platform loading, and trace cost

### Land coherent end-to-end fp16 native CUDA decode
**By:** Sebastian; component work by Mariette, Leon, and Roy; reviewed by Bryant, Wallace, and Holden 🟢  
**What:** Thread fp16 activations, KV, logits/argmax, normalization, RoPE, attention, and direct fp16×int4 MatMulNBits through native decode while retaining dtype-gated f32 paths. Leon fixed the rejected fp16 LayerNorm shared-memory reuse race before Bryant approved the normalization/RoPE path. Landed as `c8741ba`.  
**Why:** H200 Qwen2.5-0.5B int4 reached about **344 tok/s** with coherent tokens, CUDA graph capture, and zero fallbacks, up from the approximately **200 tok/s** f32 path; f32 remained unregressed near 200 tok/s.

### Make CUDA-to-CPU fallback observable and optionally strict
**By:** Deckard; reviewed by Batty 🟢  
**What:** Retain a structured `ExecutionProviderFallbackReport`, emit an initialization warning when CUDA declines force whole-session CPU execution, and make `ONNX_GENAI_REQUIRE_CUDA=1` reject that fallback. Landed as `3a8eebe`.  
**Why:** Device selection must not silently advertise CUDA while executing on CPU; callers now receive node/op/reason detail and can opt into strict CUDA-only behavior.

### Use OS-aware CUDA and CUPTI dynamic-library discovery
**By:** Leon and Roy; reviewed by Pris 🟢  
**What:** Select CUDA driver/runtime/library and CUPTI candidates by operating system, including Windows DLL names and pip/Conda layouts. Treat Windows ARM64 as gracefully unavailable before probing x64-only NVIDIA libraries. Landed as `2466016` and `8cd36c3`.  
**Why:** Cross-platform probing must fail normally rather than panic or attempt incompatible binaries. CUPTI discovery remains local to the tracer to avoid an inverted dependency on the CUDA EP.

### Emit per-op CPU bytes/FLOPs only for active trace spans
**By:** Rachael, Gaff, and Deckard; reviewed by Zhora 🟢  
**What:** Annotate major CPU kernel spans with logical tensor bytes and documented FLOP estimates, lazily computing metrics only when a span is active. Keep tracing optional and propagate the `tracing` feature through `bench-native` and `native-backend`. Landed as `61f4d2c`.  
**Why:** Profiles gain arithmetic-intensity and bandwidth inputs without imposing tensor scans, formula work, JSON allocation, or tracer dependencies on default non-tracing builds.



## 2026-07-21 — CI hardening and native CUDA decode wave 1–2

### Cover every offline crate and make warnings blocking on all portable targets
**By:** Batty and Gaff; Windows ARM64 revision by Deckard; reviewed by Hudson 🟢  
**What:** Classify all 38 workspace members by default normal+dev dependencies, explicitly test and cover all 27 pure-offline crates, and enforce blocking rustc and Clippy warnings (`RUSTFLAGS="-D warnings"` and `-- -D warnings`) rather than advisory lanes. The portable matrix retains Linux x64, Windows x64, and macOS ARM64 and adds native Windows ARM64 on `windows-11-arm`/`aarch64-pc-windows-msvc`, with the same 26-crate portable test set and an ARM64 Clippy gate; `mlas-sys` remains Linux-only, while native-ORT and CUDA crates stay outside offline execution. Formatting remains advisory pending the repository-wide sweep.  
**Why:** CI now covers the full offline workspace without triggering ORT downloads, and warnings fail builds across supported portable targets. The final 27-crate Linux lane passed 1,921 tests with 0 failures and 8 ignored; Hudson approved after Deckard closed the initially missing Windows ARM64 gate.

### Keep the measured wave-1 decode optimizations capture-safe
**By:** Leon, Tyrell, Deckard, Sebastian, and Roy  
**What:** Use persistent two-pass multi-block greedy argmax; segment CUDA graphs into maximal capturable runs around eager CUDA seams while retaining whole-subgraph EP ownership; abort/drain failed mid-segment capture before reset; use true multi-CTA split-K fp16 flash decode; and retain Roy's coalesced direct fp16×int4 GEMV retile. All paths preserve fixed device addresses, token semantics, and zero-fallback graph replay.  
**Why:** These changes removed launch/occupancy and GEMV bottlenecks without regressing correctness: argmax reached about 368 tok/s, split-K attention about 398 tok/s at 256 tokens (about 390 at 1024), and the GEMV retile about 423 tok/s. Segmented capture now recovers cleanly from invalidated streams instead of wedging later inference.

### Fuse the single-token GQA preparation chain
**By:** Rachael; reviewed by Holden 🟢  
**What:** For eligible `Sq=Sk=1` aliased fixed-capacity decode, fuse QKV split, query relayout, K/V append, and Q/K RoPE into one kernel and write attention output directly in BSH layout. Keep metadata preparation separate to preserve the capture poison/latch protocol; all other shapes retain the unfused path.  
**Why:** Prep launches fell 75% (192→48 per token), bit-exact fused/unfused and capture tests passed, and H200 throughput rose from about 557 to 615 tok/s with zero fallbacks.

### Use warp-shuffle fp16 skip-RMSNorm
**By:** Sapper; reviewed by Wallace 🟢  
**What:** Replace the fp16 shared-memory reduction tree with a single-warp packed-half2/half4 shuffle reduction, specializing hidden size 896 while retaining a tail-safe generic fp16 path; f32 kernels remain unchanged.  
**Why:** The hot kernel fell from about 6.20 to 5.07 µs/call and stacked decode reached about 579–583 tok/s with identical tokens, full CUDA tests passing, and zero graph fallbacks.

### Specialize the fp16 down-projection GEMV and accept the stacked ORT win
**By:** Luv; reviewed by Pris 🟢  
**What:** Route only `K=4864, N=896, block_size=32` with fp16 scales to a 256-thread, eight-column K-parallel GEMV that stages the activation in permuted half2 shared memory; all other shapes retain the general kernel.  
**Why:** The down-projection kernel fell from about 10.24 to 7.28 µs/call with parity within fp16 tolerance and identical greedy tokens. Stacked with GQA fusion and RMSNorm, native H200 decode reached **663–672 tok/s**, beating the **657 tok/s ORT GenAI** reference with zero fallbacks.

### Require SM-portable correctness and performance for every CUDA EP kernel
**By:** Coordinator directive; validated in wave-2 reviews by Holden, Wallace, and Pris  
**What:** Every `onnx-runtime-ep-cuda` kernel must remain correct and performant across supported NVIDIA SM architectures, not merely `sm_90`. Dispatch must derive the live architecture dynamically, avoid unguarded SM90-only features, keep resource use within portable limits, and preserve capable fallbacks or variants where architecture-specific tuning is necessary.  
**Why:** H200 wins are not acceptable if they break or materially strand devices such as RTX 4060 (`sm_89`). Wave-2 kernels use broadly available primitives and do not raise the minimum architecture.

## 2026-07-21 — Native CUDA decode wave 3 and CUDA CI

### Use 16-way split-K for long-context fp16 GQA decode
**By:** Sebastian; reviewed by Holden 🟢
**What:** Raise fp16 flash-decode `MAX_SPLITS` from 8 to 16, retaining device-side capture-safe split selection, deterministic fixed-order merging, and the single-stream shared-scratch invariant. Landed as `3b972bf`.
**Why:** Independent H200 review measured 1024-token decode improving from about 647 to 693 tok/s (+7.1%) while 256-token throughput remained flat, with identical greedy tokens, zero graph fallbacks, bounded 2.03 MiB scratch, and no SM90-only dependency.

### Fuse SwiGLU SiLU and multiply in one CUDA kernel
**By:** Mariette; reviewed by Pris 🟢
**What:** Fuse eligible equal-shape, single-consumer `Mul(Silu(gate), up)` patterns into one capture-safe f32/f16/bf16 pointwise kernel, preserving separate fallback paths and kernel-variant trace reasons. Landed as `12e48b8`.
**Why:** The fusion halves activation launches from 48 to 24 per token and improved authoritative 256-token H200 decode from about 673 to 689 tok/s, with identical tokens, zero graph fallbacks, full CUDA parity, and portable primitives suitable for sm_89.

### Record the stacked wave-3 performance baseline
**By:** Kowalski
**What:** Treat the fresh shared-H200 re-profile as the current wave-3 baseline: median throughput about 691 tok/s at 256 tokens and 712 tok/s at 1024 tokens, with zero CUDA graph fallbacks. Recorded in `docs/PROGRESS.md` by `f42ca3f`.
**Why:** The stacked GQA split and SwiGLU fusion gains reproduce together, remain coherent, and place native CUDA decode above the 657 tok/s ORT GenAI reference at 256 tokens.

### Gate CUDA EP Clippy warnings in CI
**By:** Gaff; reviewed by Wallace 🟢
**What:** Clear all 21 existing `onnx-runtime-ep-cuda` Clippy warnings without adding allows, remove no-op explicit drops of non-owning `TensorMut` views, and add `cargo clippy -p onnx-runtime-ep-cuda --features cuda -- -D warnings` to the `cuda-compile` job. Landed as `22ec87e`.
**Why:** CUDA EP warnings are now blocking in CI. Review verified the lint rewrites and drop removals preserve behavior and ownership, with builds, tests, Clippy, YAML parsing, and a zero-fallback performance sanity run passing.


## 2026-07-21 — Native CUDA decode wave 4

### Fold batch-1 GQA metadata into fused decode preparation
**By:** Luv; reviewed by Holden 🟢  
**What:** For eligible batch-1, `Sq=Sk=1`, fixed-capacity aliased-device-KV decode, derive GQA metadata inside each fused prep CTA and have block 0 write the attention arrays; unsupported shapes retain the separate metadata kernel. Landed as `bd30e6c`.  
**Why:** The change preserves latch-first poison propagation, all bounds/error bits, sentinel/no-write behavior, capture safety, and SM portability while removing 24 launches/token. Independent H200 review measured roughly 691→710 tok/s at 256 tokens with exact tokens and zero fallbacks.

### Fuse MatMulNBits-adjacent QKV bias and paired gate/up SwiGLU
**By:** Rachael; reviewed by Pris 🟢  
**What:** Fold eligible QKV bias Adds into the MatMulNBits epilogue with exact two-op fp16 rounding, and collapse the validated Qwen 0.5B gate/up projections plus SwiGLU into one paired capture-safe kernel. Strict initializer, shape, dtype, consumer, and graph-output gates preserve unfused fallback. Landed as `102fee9`.  
**Why:** GPU bit-exact tests and end-to-end greedy tokens match the two-op baseline, with zero graph fallbacks and portable primitives. Stacked on the GQA metadata fold, H200 reached about **759 tok/s at 256 tokens** and **789 tok/s at 1024 tokens**, saving about 72 launches/token.

### Drop the CUDA replay binding-cache prototype — DEAD END
**By:** Deckard  
**What:** Do not merge or re-attempt commit `14a1d8f`, which cached validated device-I/O metadata and raw external addresses for CUDA-graph replay.  
**Why:** Two paired H200 measurements showed only **+0.23%** (+1.60 tok/s), below the 0.5% noise threshold, while the exact-identity/raw-address predicate adds correctness sensitivity on the replay hot path. Revisit only with materially stronger isolated evidence and a safer design.

### Keep Ana wave-3 roofline as the current roofline of record
**By:** Scribe  
**What:** Preserve `.squad/decisions/inbox/ana-wave3-roofline-691.md` as the current roofline artifact: wave 4 achieved about **759 tok/s**, within its **750–790 tok/s** ceiling.  
**Why:** The artifact remains the authoritative lever ranking and ceiling analysis after wave-4 validation.
