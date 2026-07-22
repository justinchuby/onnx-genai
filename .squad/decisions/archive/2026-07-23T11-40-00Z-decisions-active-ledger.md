# Archived active decisions ledger at 2026-07-23T11:40Z

This archive was created by Scribe because `.squad/decisions.md` was 310698 bytes, above the 51200-byte aggressive archive gate.
The full active ledger before the 2026-07-23T11:40Z inbox merge follows.

# Decisions

> Current decision ledger. Full prior history through 2026-07-20T13:35Z is preserved in
> `.squad/decisions/archive/2026-07-20T13-35-00Z-decisions-pre-multistream.md`.

> Entries older than 2026-06-21T23:55Z are archived in `.squad/decisions/archive/2026-Q2.md` when present.

<!-- scribe-merge-2026-07-23T06-05-30Z-moe-phase1-and-perf-fixes -->
## 2026-07-23 — MoE Phase 1, native GAP 2, Qwen3 perf fix, and model generality

Decision archive gate checked at 2026-07-23T06:05Z: active ledger was 288450 bytes before merge; cutoff for seven-day retention was 2026-07-16T06:05Z. No active top-level dated sections older than 2026-07-16 were present, so no archive entries were eligible in this pass.

<!-- source: .squad/decisions/inbox/hassan-model-generality.md -->
### 2026-07-23: Llama-3.2-1B-Instruct generality + H200 bench
**By:** Hassan
**What:** Chose Llama-3.2-1B-Instruct, exported as Q4_K_M through Mobius's standard block-32 MatMulNBits path. Fast-path default: yes after a generic Mobius fix to emit `kv_cache.native_dtype` from model dtype. Native/ORT decode was 97.26/589.66 tok/s at 128 and 97.41/536.78 tok/s at 1024; median prefills were 26.529/3.350 ms and 27.843/3.583 ms respectively. Short output was coherent: “Paris. The capital of Germany is Berlin. The capital of Italy is Rome.”
**Why:** Before the generic exporter fix, `kv_cache.native_dtype` was absent, so the runtime's GQA + supported-KV-dtype + max-sequence-length shared-buffer gate failed. With unmodified generated metadata and both device-KV/CUDA-graph environment overrides unset, native reported auto graph capture, zero fallbacks, and zero KV H2D/D2H transfers. Generality passes: no runtime architecture-name dispatch was found. Performance does not approach ORT because the Q6_K tied embedding/head is dequantized to fp16 and leaves an unfused initializer `Transpose` plus dense output-head `MatMul`, which dominates native trace time.
<!-- source: .squad/decisions/inbox/hythe-deepseek-moe-phase1.md -->
### 2026-07-23: DeepSeek MoE/QMoE Phase 1 (mobius export + dense fallback)
**By:** Hythe
**What:** Implemented and tested the coherent dense-reference subset of Phase 1. Mobius now documents its standard-ONNX per-expert path as the dense fallback, fixes DeepSeek grouped routing to mask excluded groups with negative infinity, distinguishes `noaux_tc` top-2-sum group scoring from group-limited maximum scoring, and passes a tiny CPU ONNX Runtime differential test including routed and shared experts. Decoder export now emits generic `model.mixture_of_experts` metadata: representation, routed/shared expert counts, experts per token, routed/shared intermediate sizes, activation, router score function, selection method, normalization, scaling, and grouped-routing parameters. onnx-genai adds matching typed schema declarations, generated JSON Schema, conditional grouped-router validation, tests, and status documentation. Mobius commit `ac7d110` is PR #423; onnx-genai commit `fd95bc0` is pushed on `squad/hythe-deepseek-moe-phase1`.
**Why:** A standard-op dense graph is portable across CPU execution providers and provides the correctness oracle required before grouped kernels or streaming. Keeping router policy explicit preserves bias-corrected selection separately from aggregation and avoids every model-name/runtime heuristic prohibited by RULES.md §2/§2.1. Remaining Phase 1 work is fused `com.microsoft::MoE` export where the single-router-input schema is exact, fused `com.microsoft::QMoE` export with positional optional inputs, byte-for-byte or converted QMoE/MatMulNBits packing validation (layout, transpose, scales, zero points, prepacking), and fused-vs-dense differential tests.
<!-- source: .squad/decisions/inbox/isidore-gqa-alias-fix.md -->
### 2026-07-23: decode.rs GQA-alias consistency fix
**By:** Isidore
**What:** Updated `is_group_query_attention()` to accept `grouped_query` and `group_query` alongside the existing GQA spellings, and expanded unit coverage for normalized case, space, and hyphen variants plus non-GQA values. Branch: `squad/isidore-gqa-alias`; commit: `acffa2171532d4e1e4dd4a4137ba933bef5be9be`.
**Why:** The metadata schema lists `grouped_query` as valid GQA vocabulary, but the runtime rejected it and disabled the shared-buffer KV fast path. The runtime now recognizes every GQA-family value listed by `ATTENTION_TYPE`, plus the symmetric `group_query` alias, without model-specific branching. Verification: `cargo test -p onnx-genai-engine --lib is_group_query -- --nocapture` passed (1 passed, 0 failed); `cargo clippy -p onnx-genai-engine --all-targets -- -D warnings` completed cleanly; `cargo fmt -p onnx-genai-engine` completed cleanly. Model-name grep found only the pre-existing Gemma/Mistral comment at decode.rs:1635 and no model names in new logic.
<!-- source: .squad/decisions/inbox/joe-hythe-moe-review.md -->
### 2026-07-23: Review — DeepSeek MoE/QMoE Phase 1 (dense-reference subset)

**Verdict:** 🟢 APPROVED
**Reviewer:** Joe (independent; author was Hythe)
**Scope:** onnx-genai `squad/hythe-deepseek-moe-phase1` @ fd95bc0; mobius `squad/hythe-deepseek-moe-phase1` @ ac7d110 (PR #423)

---

## Summary

Both repos deliver a clean, structural, additive Phase 1 dense-reference subset. No
model-name gating in runtime logic, metadata is generic and append-only, JSON Schema
matches the Rust schema, the router policy is kept separate from aggregation, and the
CPU ONNXRuntime-vs-NumPy differential test genuinely exercises grouped routing with
negative-infinity masking, `noaux_tc` top-2-sum group scoring, and routed + shared
experts. All targeted tests pass on disk in read-only worktrees.

---

## Findings

### 1. No model-name gating (RULES.md §2/§2.1) — PASS
- **WHAT:** `grep -rniE "gemma|qwen|phi[^a-z]|llama|mistral|deepseek" crates/*/src/` returns
  only pre-existing matches in `onnx-genai-bench` binaries (default model paths + doc
  comments). None are introduced by this diff; `schema.rs` has zero matches.
- **WHY:** The changed onnx-genai files (`schema.rs`, tests, docs, JSON schema) contain no
  family branching in logic.
- **HOW verified:** grepped `schema.rs` directly (NONE) and cross-checked `git diff
  --name-only` against grep hits — no overlap.
- **mobius:** `decoder_metadata.py::moe_metadata_from_config` is fully data-driven (reads
  `num_local_experts`, `num_experts_per_tok`, `n_group`, `topk_method`, etc.). Branching is
  on config *values* (e.g. `topk_method == "noaux_tc"`), never on a model name. The MoE
  metadata emitted is structural: expert counts, widths, top-k, routing policy as data.
  `deepseek.py` is a model-definition file (models/) and legitimately model-specific;
  its branching is on `self.topk_method`, a config value.

### 2. Metadata additive & explicit — PASS
- **WHAT:** `MixtureOfExpertsSpec`/`MoERouterSpec` are new append-only fields on
  `ModelCapabilities` (`mixture_of_experts: Option<...>`, `#[serde(default, skip_...)]`).
  Vocabularies use the existing `extensible_string!` forward-compatible pattern.
- **WHY:** No existing field changed types or semantics; optional with skip-if-none, so
  existing metadata still round-trips.
- **HOW verified:** Read full `schema.rs` diff; all additions. No implicit model-specific
  defaults — every required field must be supplied; grouped fields are conditionally
  required via `allOf/if/then` on `selection_method == grouped_top_k`.

### 3. JSON schema matches Rust schema — PASS
- **WHAT:** `schema/inference_metadata.schema.json` adds `MixtureOfExpertsSpec`,
  `MoERouterSpec`, and the four extensible vocab defs, matching field names, `minimum`
  bounds, nullability, and the grouped-router conditional `required`.
- **HOW verified:** `tests/schema_sync.rs::committed_inference_metadata_schema_is_current`
  PASSES — the committed JSON is byte-current with the generated schema.

### 4. Conformance correctness — PASS (strongest evidence)
- **WHAT:** `deepseek_test.py::test_deepseek_dense_moe_fallback_matches_numpy_reference`
  builds a tiny grouped, bias-corrected MoE (`n_group=2, topk_group=1,
  num_experts_per_tok=2, n_shared_experts=1, scoring_func=sigmoid, topk_method=noaux_tc`),
  runs it under `CPUExecutionProvider`, and asserts allclose vs a NumPy reference.
- **WHY it is a real test of the fix:** router logits are zeroed so all original sigmoid
  scores = 0.5; the `e_score_correction_bias = [-0.7,-0.6,-1.5,-1.4]` makes the winning
  group's corrected scores *negative*. If the gate multiplied by zero (the old bug),
  excluded experts (score 0) would beat the selected group's negative scores in TopK. The
  test only passes because excluded groups are masked with `-inf` (the fix in
  `deepseek.py`). It also asserts `all(node.domain != "com.microsoft")` (true dense
  standard-op fallback) and includes shared-expert aggregation. `noaux_tc` top-2-sum group
  scoring vs `ReduceMax` group-limited scoring is correctly split by the `if
  topk_method == "noaux_tc"` branch.
- **HOW verified:** ran the test — PASS.

### 5. Router policy separation — PASS
- **WHAT:** Bias-corrected selection (grouped TopK over corrected scores + `-inf` mask)
  operates on the routing scores; aggregation weights use the *original* scores
  (normalized, `routed_scaling_factor` applied) separately. Metadata mirrors this:
  `MoERouterSpec` carries `score_function`/`selection_method`/`normalize_weights`/
  `scaling_factor`/group params independently of expert-FFN fields.
- **HOW verified:** read `deepseek.py` gate + `moe_metadata_from_config`; the NumPy
  reference in the passing test reconstructs selection and aggregation independently.

### 6. Tests match disk reality (test-discipline) — PASS
- New onnx-genai tests (`mixture_of_experts_contract_parses_structurally`,
  `..._grouped_router_requires_group_contract`) assert exactly the fields present in the
  schema; both pass. mobius `test_moe_metadata_from_config_is_structural` asserts the full
  emitted dict, `MatMul` count `== 16`, and the incomplete-contract `ValueError` path — all
  pass. Assertion counts correspond to the code under test.

---

## Advisory (non-blocking)
- `moe_metadata_from_config` emits `selection_method="grouped_top_k"` only when
  `group_count > 1 AND topk_method != "greedy"`; a config with `n_group>1` but greedy
  routing emits plain `top_k` and drops group dims. Correct for the Phase 1 dense subset,
  but worth an explicit note when fused `MoE`/`QMoE` export lands so grouped structure
  isn't silently dropped for greedy-grouped variants.
- `_moe.py` change is docstring-only (clarifies dense-fallback role); no behavior change —
  fine.

---

## Tests run
- **onnx-genai** (worktree `origin/squad/hythe-deepseek-moe-phase1`):
  `cargo test -p onnx-genai-metadata` → **29 passed, 0 failed** (incl. 2 new MoE tests +
  `schema_sync` current check).
- **mobius** (worktree `origin/squad/hythe-deepseek-moe-phase1`, `.venv` py, PYTHONPATH →
  worktree src): `pytest deepseek_test.py decoder_metadata_test.py` → **12 passed, 0
  failed** (incl. the CPU ORT-vs-NumPy differential oracle).

No blocking defects found. Recommend dispatching the merge agent.

— Joe
<!-- source: .squad/decisions/inbox/joi-qwen3-rebench-postfix.md -->
### 2026-07-23: Qwen3 Mobius metadata fix restores default shared-KV performance
**By:** Joi
**What:** End-to-end H200 verification passed using Mobius export commit `820c1d7494b438f2f51f3ddeb5595914dfac0422`. Important repository-state caveat: after fetching, `origin/main` was `38cb789a51e68b5907d82fa67704a73fdef80902`, did not contain `820c1d7`, and PR #422 still reported OPEN, so the fresh export was made from the confirmed PR commit rather than current `main`.

Fresh package: `/home/justinchu/mobius/.scratch/qwen3-0.6b-int4-cuda-postfix`

```yaml
model:
  attention:
    type: grouped_query_attention
kv_cache:
  native_dtype: float16
```

H200 ORT CUDA eager decode (`ONNX_GENAI_DEVICE_KV=1`, `ONNX_GENAI_CUDA_GRAPH=0`, 2 warmups, 3 runs):

| Model/configuration | 128 tok/s | 1024 tok/s |
|---|---:|---:|
| Qwen3 old export | 197.49 | 64.01 |
| Qwen3 fresh `820c1d7` export | **441.96** | **374.50** |
| Qwen2.5-0.5B control, prior | 570.61 | 501.88 |
| Qwen2.5-0.5B control, rerun | **577.24** | **498.67** |

The Qwen3 fresh package required no metadata override. Its `ort.bind_inputs` stayed approximately constant at 52.35/54.51 us per token for 128/1024, versus the old growing-path 1.093/6.938 ms per token, confirming the fast shared-buffer KV path is now the default for the corrected export. The Qwen2.5 control remained within normal run variance, so no regression was observed.

**Why:** Canonical `grouped_query_attention` plus `kv_cache.native_dtype: float16` satisfies the runtime metadata gate and eliminates the severe context-length throughput collapse. The requested `bench,cuda` Cargo feature spelling is stale; the current benchmark builds with `--features cuda-ort`. CUDA ORT also required `.ort-cuda-1.27/root/lib` before the CPU-only ORT directory in `LD_LIBRARY_PATH`.
<!-- source: .squad/decisions/inbox/kandel-sebastian-gap2-review.md -->
### 2026-07-23: Review — Sebastian Native GAP 2 (generic native decode step inputs)

**By:** Kandel (architecture/interface reviewer — read-only, reviewer independence enforced)

**What:** 🟢 **APPROVE**

Reviewed branch `squad/sebastian-native-gap2` @ `a7a2a5b` (vs `origin/main`):
`native_decode.rs` (+653/-146), `native_component.rs` (+8), `pipeline.rs` (+14).

GAP 2 replaces `NativeDecodeSession`'s fixed `input_ids`/`attention_mask`/`position_ids`
fields with a `Vec<NativeStepInputBinding>` (`name` + `source` role) covering
`TokenIds | InputsEmbeds | AttentionMask | PositionIds | Routed`. All non-KV graph
ports are enumerated and each resolves to a declared generated role or falls back to
`Routed` (resolved by exact port name from pipeline-supplied tensors).

**Checklist results (file:line evidence):**

1. **No hardcoded architecture (RULES §2/§2.1) — PASS.** `grep` for family names hits
   ONLY test identifiers/paths (`native_decode.rs:3882,3888,4017`). Binding is driven by
   `sequence_source` + `ModelIoSpec` declared roles and exact port names — no model-name
   or dimension-value branch gates logic (`native_decode.rs:857-909`).
2. **Token path + CUDA fast path unchanged — PASS.** `decode()` now delegates to
   `decode_with_step_inputs(..., &[])` (`:2168`); `decode_host` reproduces the old
   token/mask/position construction and single-token `with_decode_pool_scope` gating
   verbatim (`:249-338`). CUDA still routes to `decode_cuda` unchanged (`:224-231`);
   generic path is additive.
3. **KV present→past publication preserved — PASS.** `decode_host` KV fill + `present_to_past`
   republication + seq-axis length check are identical to the removed code (`:314-382`).
4. **Correct generality — PASS.** Duplicate declared-role collision (`:127-131`),
   missing sequence-source input (`:148-155`), duplicate routed input (`:264-266`),
   missing embeds/routed tensor (`:286-303`), and unknown/undeclared supplied ports
   (`:307-313`) all error explicitly. Nothing is silently dropped or mis-routed.
5. **GAP 3 boundary — PASS.** CUDA + embeds/routed bails clearly at load (`:158-168`) and
   at step (`:225-229`); `pipeline.rs` still returns the actionable GAP 3 error naming the
   precise next blocker (`pipeline.rs:205-213`). ORT every-step executor generality in
   `pipeline.rs` (~L1952-2027, 2145-2259) is untouched — diff only edits doc/error text.
6. **Build/clippy/tests — PASS.** `cargo build -p onnx-genai-engine --features native-backend`
   OK; `cargo clippy --all-targets -- -D warnings` clean; 3 new tests
   (`native_target_step_preserves_token_driven_binding`,
   `..._binds_declared_inputs_embeds_instead_of_tokens`,
   `..._resolves_routed_component_output_by_declared_port`) pass. Suite: 165 passed / 16
   failed — the 16 are the pre-existing protobuf-fixture decode failures (confirmed
   identical 16 on `origin/main`: 162 passed / 16 failed). **Zero new failures.**

**Why:** Clean, additive generalization that binds arbitrary metadata-declared named step
inputs (embeddings, attention mask, position ids, exact-name routed tensors) while
preserving the token-driven decode and CUDA shared-buffer fast path byte-for-byte. Error
boundaries are strong and explicit at every mismatch, and the GAP 3 stop is a clear error
rather than a silent wrong result. Correctly targets the Gemma4-E2B-class contract
(inputs_embeds + per-layer routed inputs + mixed-width KV) without any architecture
hardcoding.

**Non-blocking notes (no action required):**
- The target decoder's `attention_mask` moved from required (`declared_or_detected_input`)
  to optional (`optional_declared_or_detected_input`, `:772`). This is an intentional
  relaxation — a token graph that genuinely lacks a mask port now loads maskless instead
  of erroring. Common models still auto-detect `attention_mask` and behave identically.
- `decode_with_step_inputs` still requires non-empty `token_ids` even for embeds models
  (used for length/position bookkeeping, `:215`). By design; caller supplies token count.

No reviewer-lockout reassignment needed (APPROVE).
<!-- source: .squad/decisions/inbox/marlowe-hythe-moe-merge.md -->
### 2026-07-23: Hythe DeepSeek MoE Phase 1 landing
**By:** Marlowe
**What:** Fast-forward landed the onnx-genai portion on `main` at `751f06838b29dce6afadd7e61f63c066765f65a9`. The metadata test run passed: 29 unit/fixture tests, 1 schema-sync integration test, and 0 doc tests failed. Mobius PR #423 was not merged, so it has no merge SHA.
**Why:** The onnx-genai commit rebased cleanly onto `origin/main` at `a7a2a5b`; no conflicts or manual file edits were required. Fast-forward ancestry was confirmed immediately before pushing.

Mobius PR #423 remained `BLOCKED` with red CI despite being mergeable and current with `main` (0 commits behind), so no base update was needed. Passing checks included architecture diff, benchmarks, build, CodeQL, L1/L3, and all Linux/Windows Python test matrices. Blocking checks were:

- `Lint`: Ruff formatting failure in `src/mobius/integrations/onnx_genai/decoder_metadata.py`.
- `Integration (fast)`: collection error because `libcudart.so.13` was unavailable.
- `L5 Generation E2E`: collection error because `libcudart.so.13` was unavailable.
- `codecov/patch`: failed.
- L4 Golden Comparison was still pending.

Per the green-only merge rule, PR #423 remains open and unmerged.
<!-- source: .squad/decisions/inbox/marlowe-sebastian-gap2-merge.md -->
### 2026-07-23: Sebastian Native GAP 2 merged
**By:** Marlowe
**What:** Fast-forwarded origin/main to `a7a2a5b3247e032ed581dc4ef598171e83498019` after rebasing `squad/sebastian-native-gap2` onto origin/main.
**Validation:** Build passed; clippy passed; tests passed with 228 passed, 0 failed, and 16 ignored. Sebastian's 3 native-backend GAP-2 tests passed. Zero new failures.
**Conflicts:** None; the rebase completed cleanly.
<!-- source: .squad/decisions/inbox/rutger-isidore-gqa-review.md -->
### 2026-07-23: Review of isidore GQA-alias fix
**By:** Rutger
**Verdict:** 🟢
**What:** PASS — no blocking or minor findings.
- **Correctness — What:** Exact matching now recognizes every GQA spelling in `ATTENTION_TYPE` (`grouped_query`, `group_query_attention`, `grouped_query_attention`, `gqa`) plus the requested `group_query` short alias. **Why:** Case folding and single space/hyphen normalization preserve existing behavior without admitting MHA or unrelated attention types. **How:** Compared `decode.rs` with `schema.rs:1967-1983`; tests reject `multi_head_attention`, `mha`, and empty input.
- **Model/architecture rules — What:** No model-name or hardcoded-architecture logic was added. **Why:** The gate remains driven solely by `model.attention.type`, KV dtype metadata, maximum sequence length, and session capability. **How:** Changed-line grep for Gemma/Qwen/Phi/Llama/Mistral was empty; the diff changes only the generic matcher and its unit test.
- **Tests/build — What:** Alias, normalization, and negative cases are covered and validation is clean. **Why:** The focused test exercises both new aliases and representative case/space/hyphen forms. **How:** `cargo test -p onnx-genai-engine --lib is_group_query -- --nocapture` passed (1/1); `cargo clippy -p onnx-genai-engine --all-targets -- -D warnings` passed.
**Why:** The patch is a narrow metadata-vocabulary compatibility fix that enables the existing shared-buffer/device-KV path for structurally declared GQA while leaving non-GQA models on their prior path.
<!-- source: .squad/decisions/inbox/sebastian-native-gap2.md -->
### 2026-07-23: Native target decode accepts generic metadata-driven step inputs
**By:** Sebastian
**What:** GAP 2 landed on branch `squad/sebastian-native-gap2` at commit `a7a2a5b`. Changed `crates/onnx-genai-engine/src/native_decode.rs`, `native_component.rs`, and `pipeline.rs`. `NativeDecodeSession` now stores a graph-ordered generic binding list whose sources are token IDs, input embeddings, attention mask, position IDs, or an exact-name routed tensor. `decode_with_step_inputs` generates declared runtime roles, resolves every other non-KV graph input from named routed tensors, and preserves present-to-past KV publication. Token-driven decode and its CUDA fast path remain unchanged; generic embedding/routed execution is CPU-native. Added focused tiny-graph tests for token binding preservation, embedding-driven target binding, and component-output-to-target routed binding. Updated the native pipeline error to identify GAP 3 only.
**Why:** Embedding-driven target graphs can require multiple refreshed decoder inputs, so fixed architecture-specific fields cannot express their contract. Exact metadata role names plus pipeline dataflow destination ports provide an architecture-neutral binding seam without model-name or dimension branches. Validation: default and native builds passed; strict native clippy passed with `-D warnings`; default engine/metadata tests passed; focused new native tests passed; full native-feature tests reported the unchanged 165 passed / 16 unrelated fixture failures / 1 ignored. `cargo fmt -p onnx-genai-engine -p onnx-genai-metadata` completed. The changed-Rust model-name grep was empty. GAP 2 is complete; GAP 3 remains converting `DecodeState` and `PipelineDecodeLoopBackend` from ORT `Value`/`Session` ownership to backend-neutral tensors/component sessions.

<!-- scribe-merge-2026-07-23T01-00-00Z-gap1-mla-qwen3-merges -->
## 2026-07-23 — Native GAP 1, MLA conformance, Qwen3 bench, and inbox reconciliation

Decision archive gate checked at 2026-07-23T01:00Z: active ledger was 163948 bytes before merge; because it exceeded 51200 bytes, entries older than 2026-07-16T01:00Z were eligible for archival. No active entries older than that cutoff were present, so no archive file was written.

<!-- source: .squad/decisions/inbox/dave-mobius-decoder-metadata-fix.md -->
### 2026-07-23: mobius decoder metadata GQA+KV emission fix
**By:** Dave
**What:** Mobius now emits `grouped_query_attention` for GQA decoders and infers canonical `kv_cache.native_dtype` (`float16`, `bfloat16`, or `float32`) from the model activation/compute dtype, independent of weight quantization. PR: https://github.com/onnxruntime/mobius/pull/422. Validation: ONNX GenAI integration tests passed (90 passed, 3 skipped); Ruff format/check passed. The existing Qwen3-0.6B int4 CUDA artifact was re-verified with a metadata-only probe using its FLOAT16 KV graph inputs, producing `attention.type: grouped_query_attention` and `kv_cache.native_dtype: float16` without rerunning the GPU export or overwriting the artifact.
**Why:** The runtime enables device/shared-buffer KV only when metadata both identifies GQA with a recognized attention type and declares a supported floating-point KV dtype. Mobius previously emitted the unrecognized `grouped_query` value and omitted `kv_cache` whenever callers left `kv_native_dtype=None`, forcing generic GQA decoder exports—including int4-weight/fp16-activation models—onto the slow growing ZeroCopyRebind path.

<!-- source: .squad/decisions/inbox/deckard-cudaattn-revision.md -->
### 2026-07-22: Route CUDA attention mode through RuntimeConfig
**By:** Deckard
**What:** Moved `ONNX_GENAI_CUDA_ATTENTION` parsing into `onnx-genai-runtime-config` as the typed `RuntimeConfig::cuda_attention_mode` field with `Auto`, `Fused`, `Unfused`, and invalid-value preservation. Added default, valid-value, alias, and invalid-value registry tests; ORT now consumes only `runtime_config()`, while `Unfused` still sets `sdpa_kernel=16`. Preserved Howie's actionable CUDA provider-missing errors. Formatting passed; CUDA-feature clippy passed with warnings denied; the requested two-crate test suite passed 100 tests; and the CUDA provider-option test passed without a GPU.
**Why:** The binding runtime-config decision requires every new runtime flag to be declared, parsed, documented, and tested in the single typed registry rather than read directly at an ORT call site.

<!-- source: .squad/decisions/inbox/deckard-gap1-schema-fix-merge.md -->
### 2026-07-23: GAP 1 schema_sync fix + merge
**By:** Deckard
**What:** The blocker was a pre-existing schema-generation determinism bug, not a GAP 1 schema change. Schema generation now recursively sorts JSON object keys before pretty-printing, and both the generator and sync test use that canonical codepath. The committed schema was reordered without semantic changes. GAP 1 and the fix merged to `origin/main` at `c47534d`.
**Why:** On both the GAP 1 branch and pre-fix `origin/main` (`a9370e2`), the isolated schema test passed while the combined metadata+engine graph failed with semantically identical JSON and ordering-only differences. The isolated `cargo tree` had no `serde_json/preserve_order`; the combined graph enabled it through `llguidance`, changing `schemars` map ordering. Isolated, combined, and native-backend schema tests passed after the fix, as did the full combined metadata+engine tests, default workspace build, targeted Clippy with warnings denied, and metadata fmt check.

<!-- source: .squad/decisions/inbox/deckard-sebastian-gap1-review.md -->
### 2026-07-22: Review of sebastian native gap 1 (ComponentSession interface)
**By:** Deckard
**Verdict:** 🟢 APPROVE

**What:** Reviewed `squad/sebastian-native-interface` @ `b6dcb32` (merge-base origin/main `1ae99b4`). Sebastian introduces the backend-neutral `ComponentSession` seam plus neutral tensor vocabulary (`ComponentTensor` = raw LE bytes, `ComponentIo`, `ComponentDataType`, `ComponentError`) in `onnx-genai-metadata`, an ORT adapter (`OrtComponentSession` + `Value::from_raw_bytes`/`to_raw_bytes` + `TensorBacking::Bytes`), a native adapter (`NativeComponentSession`, feature-gated), and rewires `pipeline.rs::from_dir_with_schedulers` to select ONE backend for the whole pipeline. Every claim in the task was verified true.

**Evidence:**
- **Diff scope:** 1 commit, +1081/-28 across 9 files, all coherent to gap 1 (metadata component.rs, ort component.rs/value.rs/lib.rs, engine native_component.rs/pipeline.rs/lib.rs + 1-line `pub(crate)` visibility bump in engine.rs). No scope creep.
- **No dependency cycle:** `onnx-genai-metadata/Cargo.toml` deps = `thiserror` only (a true leaf crate). Both `onnx-genai-ort` and `onnx-genai-engine` already depend on it. Trait genuinely lives where both backends can implement it without a cycle. ✓
- **Behavior preservation (ORT):** The native branch returns early; the ORT path falls through to the identical `PipelineModels::load_with_options(...)` call as before. Cambodia's `every_step` executor (pipeline.rs ~L1952-2027/2145-2259) is NOT in the diff — untouched. The ORT round-trip parity test (`named_tensor_run_round_trip_matches_session`) runs the raw `Session` directly and asserts `tensor.as_bytes() == reference[0].to_raw_bytes()` byte-for-byte through the seam. ✓
- **Actionable error (RULES §1):** Native path returns a detailed message naming the precise next blocker — generalizing native target decode beyond token-id-only to metadata-declared `inputs_embeds`/routed per-layer inputs, then moving `DecodeState`/pipeline execution off ORT `Value`/`Session` (gaps 2/3) — plus the ORT fallback instruction. No generic panic. When built without the feature, a distinct actionable "compiled without 'native-backend' feature" error. ✓
- **No hardcoded architecture (§2/§2.1):** `grep -rniE "gemma|qwen|phi[^a-z]|llama|mistral"` over the three touched logic files = empty. ✓
- **Tensor neutrality:** dtype coverage comprehensive (fp32/fp16/bf16/fp8e4m3/fp8e5m2/int8/16/32/64/uint8/16/32/64/bool) with correct `size_of`. Raw LE bytes with static-shape + byte-length validation in `from_raw`. Uses `dim as usize`/`as i64` only — no `#[cfg]` pointer-width assumptions; arm64/Windows-arm64 safe (usize is 64-bit there). ✓
- **Build:** `cargo build -p onnx-genai-metadata -p onnx-runtime-session -p onnx-genai-engine` — clean.
- **Clippy:** `-D warnings --all-targets` on all three crates — clean, zero warnings.
- **Test (default):** metadata component tests pass; the one failure `committed_inference_metadata_schema_is_current` (schema_sync) is pre-existing/environmental (committed schema out of date) — branch does not touch `schema/`, `schema.rs`, the generator, or `tests/`.
- **Native feature build:** clean except the pre-existing `native_decode.rs:17` unused `BTreeMap` warning (native_decode.rs untouched by this branch — out of scope, sebastian flagged it correctly).
- **Native feature test:** branch = 162 passed / 16 failed. Baseline `1ae99b4` = 160 passed / **17 failed**. I captured both failing sets: the branch's 16 failures are EXACTLY the base's 16 embedding+engine fixture failures (all environmental — `failed to decode Protobuf message: invalid wire type value: 6`, i.e. unmaterialized LFS fixtures). The 17th base failure (`auto_backend_rejects_pipeline_component_requiring_native`) is now FIXED by the renamed passing test. **Net: -1 failure, +2 passing, ZERO new failures, ZERO new warnings.** New unit tests in native_component/ort component/metadata all pass.

**Non-blocking observations (no change required before merge):**
- `Value::to_raw_bytes` assumes a host-resident tensor (documented); correct for the CPU-only pipeline component path.
- `static_numel` overflow branch synthesizes a `ByteLengthMismatch{dtype: Uint8, expected: usize::MAX}` — slightly misleading label but only reachable on absurd shape overflow. Cosmetic.

Foundational interface is cycle-free, behavior-preserving on the ORT path, and correctly gates native behind an actionable gap-2/3 error. Cleared for a merge agent to land on main.

<!-- source: .squad/decisions/inbox/eldon-zhora-mla-merge.md -->
### 2026-07-23: Zhora MLA conformance merge
**By:** Eldon
**What:** Merged as a9370e2 on origin/main.
**Why:** Rebased cleanly onto origin/main; CPU crate build, clippy with -D warnings, all tests (656 passed, 2 ignored across suites), and crate-scoped rustfmt check passed; fast-forward ancestry confirmed before push.

<!-- source: .squad/decisions/inbox/gaff-deckard-rereview.md -->
### Approve Deckard's RuntimeConfig revision for ORT CUDA attention
**By:** Gaff
**Verdict:** 🟢 APPROVE revised commits `496a200` and `1f24046`; fast-forward merged to `main` at `1f24046`.
**Rationale:** `ONNX_GENAI_CUDA_ATTENTION` is parsed only by the typed `RuntimeConfig` registry as `CudaAttentionMode`; registry coverage verifies Auto/Fused/Unfused, aliases, invalid preservation, and the default. ORT consumes `runtime_config()` without a raw environment read, Unfused maps to `sdpa_kernel=16`, explicit missing-CUDA errors remain actionable, and changed runtime logic contains no model-family branching. Changed-crate fmt, CUDA-feature Clippy with `-D warnings`, runtime-config tests (14 passed), raw-read grep, and `git diff --check` all passed after rebasing onto `origin/main`.

<!-- source: .squad/decisions/inbox/garland-wp5.md -->
### 2026-07-22: Server multimodal bundle and post-expansion admission verified
**By:** Garland
**What:** Verified and extended WP5 server coverage for metadata-declared multimodal endpoint bundles, prompt-order per-image expansion, and context admission after expansion. Added an end-to-end rank-4 compatibility test whose declared graph endpoint is `vision_encoder.image_tensor`, proving the server does not depend on the literal `pixel_values` name. Gap #4 is fully closed for metadata-declared typed outputs, including multi-output bundles and the rank-4 compatibility shape. Gap #6 is fully closed: both streaming and non-streaming chat preprocess and expand before context-cap admission. Gap #5 is partially closed: prompt-order image summaries, per-image tile/patch counts, thumbnail placement, distinct image-token IDs, and supported per-tile separators are materialized into the final token prompt before the engine call.
**Why:** The current engine request API accepts named tensors and token IDs but has no typed per-image expansion-summary field; the server therefore resolves supported expansion contracts before submission and retains summaries through the driver seam for ordering validation. Deferred dependencies: `token_count_source=from_grid`, explicit-index correspondence, and patch-grid row/column separators require a fully specified processor-summary interpretation and the WP4 grid/position path; the frozen schema names summary tensors but does not define enough server-side arithmetic/correspondence semantics to implement these generically without inference. The pre-existing compatibility-package server unit test remains red on `origin/main` because WP1 package admission rejects an external WP6 fixture's rank-2-to-rank-3 dataflow mismatch; fixing that fixture is outside the server-only WP5 scope.

<!-- source: .squad/decisions/inbox/holden-sebastian-gap1-merge.md -->
### 2026-07-22: Sebastian GAP 1 merge blocked by verification
**By:** Holden
**What:** No SHA was merged or pushed; `origin/main` remains `8d9d2fa279a0463b9fc6c02932ebd7b9ec775fb4`. Commit `b6dcb32` rebased cleanly to candidate `65fc88a12cd840023da5a9faf3662b6fb07802de`.
**Why:** Default build and targeted Clippy passed, but the requested combined metadata/engine test command failed `committed_inference_metadata_schema_is_current` because generated schema property ordering differed from the committed schema. The isolated schema-sync test passed, but this non-LFS failure required stopping before push.

<!-- source: .squad/decisions/inbox/joi-gemma4-e2b-gaps.md -->
### 2026-07-22: Gemma4 E2B text pipeline reaches H200 ORT CUDA; native pipeline remains blocked
**By:** Joi
**What:** The four-model Mobius package produced coherent text through onnx-genai's metadata-driven PipelineEngine on H200: `"The capital of France is **Paris**."`, 140.09 tok/s steady median (7.138 ms/token; five runs, two warmups, skip eight). The run required a generic optional image/audio metadata overlay, the CUDA-enabled ORT `root/lib`, and disabling ORT's optimized attention implementations. Pure-Rust multi-model execution remains unavailable.
**Why:** The exported package predates optional-modality PR #419 and has no absent image/audio closure. Current Mobius only declares audio optional; text-only also needs an optional image input and vision presence gate. `PipelineEngine` explicitly rejects the native backend at `crates/onnx-genai-engine/src/pipeline.rs:189-208`. ORT CUDA's optimized Attention path returns `cudaErrorInvalidValue` for the valid fp16 GQA Attention node; disabling flash/lean/fused/memory-efficient/cuDNN-flash paths succeeds. The originally supplied ORT library directory has no CUDA provider library, and `onnx-genai-ort/src/session.rs:1747-1752` otherwise warns and silently falls back to CPU.

Dependency-ordered, file-disjoint work packages:

1. **Mobius text-only closure:** update `src/mobius/tasks/_gemma4.py`, `tests/build_graph_test.py`, and `src/mobius/integrations/onnx_genai/inference_metadata_test.py` to mark `image_features` optional and gate `vision_encoder`; re-export from Mobius `54be48a+`.
2. **Pure-Rust pipeline sessions:** refactor `crates/onnx-genai-engine/src/pipeline.rs` behind a backend-neutral component-session interface and load native sessions for every declared component. No model-name dispatch.
3. **ORT CUDA Attention fix:** upstream reproduction/fix for standard ONNX fp16 GQA Attention (`q_heads=8`, `kv_heads=1`, head size 256); until then use the five explicit `ORT_DISABLE_*ATTENTION=1` variables.
4. **VLM image E2E:** independently validate Gemma4 typed preprocessing and placeholder expansion in `crates/onnx-genai-server/src/image_input.rs`.
5. **Audio E2E:** independently implement typed fp16+bool audio request construction in `crates/onnx-genai-preprocess/src/audio.rs` and `crates/onnx-genai-server/src/driver.rs`.

<!-- source: .squad/decisions/inbox/joi-gemma4-native-gaps.md -->
### 2026-07-22: Gemma4 E2B pure-Rust native pipeline remains blocked at backend construction
**By:** Joi
**What:** Running the unmodified package with `ONNX_GENAI_BACKEND=native profile_native --pipeline --model /home/justinchu/mobius/.scratch/gemma4-e2b-native ...` fails before model loading with: `native backend not supported for pipeline models; set decode_backend = EngineDecodeBackend::Ort (or ONNX_GENAI_BACKEND=ort)`. No native tok/s can be reported. The actionable gaps, in dependency order, are:

1. Add a backend-neutral pipeline component-session interface. `PipelineEngine::from_dir_with_schedulers` explicitly rejects `EngineDecodeBackend::Native`, then unconditionally loads `onnx_genai_ort::PipelineModels` (`crates/onnx-genai-engine/src/pipeline.rs:189-208`). `PipelineModels` owns only ORT `Session` objects and constructs every component through ORT (`crates/onnx-genai-ort/src/loader.rs:197-223`). The interface must expose graph I/O metadata, named-tensor execution, and output names for either ORT `Session` or native `InferenceSession`; backend selection must instantiate every declared component consistently.
2. Generalize native target decode beyond token-ID-only single-model sessions. `NativeDecodeSession` is a decoder-specific adapter with a fixed input/KV field set (`crates/onnx-genai-engine/src/native_decode.rs:190-205`) and explicitly rejects `sequence_source: inputs_embeds` (`native_decode.rs:716-729`). Gemma4 declares `inputs_embeds_input: inputs_embeds` and `sequence_source: inputs_embeds`, while also requiring routed `per_layer_inputs` (`/home/justinchu/mobius/.scratch/gemma4-e2b-native/inference_metadata.yaml:663-667,885-895`). Native pipeline decode therefore needs metadata-declared arbitrary named step inputs, including both refreshed embedding outputs, without architecture-specific port names or dimensions.
3. Convert pipeline decode state/execution from ORT-only values and sessions. `pipeline.rs` imports ORT `Session`/`Value` directly (`crates/onnx-genai-engine/src/pipeline.rs:23-34`); `DecodeState` is constructed around `&Session` and stores ORT `Value` (`crates/onnx-genai-engine/src/decode.rs:693-735`); `PipelineDecodeLoopBackend` stores `&Session` for the decoder and every step component and calls ORT `session.run` (`pipeline.rs:2181-2244`). Introduce backend-neutral tensor/state operations or a parallel native decode-loop backend rather than converting through ORT types.
4. Re-export text-only optional-modality metadata. The local package still declares `embedding.image_features` and `embedding.audio_features` as required inputs (`inference_metadata.yaml:850-867`) and runs vision/audio prompt stages unconditionally (`inference_metadata.yaml:901-932`); it contains no `optional_inputs` or `when_present` declarations. Once native pipeline construction is implemented, an unmodified text-only request will still hit the generic missing-input path (`crates/onnx-genai-engine/src/pipeline.rs:1693-1718`). Mobius should emit explicit presence gates and zero fallbacks from metadata, not a Gemma-specific runtime workaround.

The merged generic `every_step` executor is not the remaining blocker: it binds token inputs from explicit metadata, resolves all routed inputs, publishes every output, and re-reads decoder routes each step (`crates/onnx-genai-engine/src/pipeline.rs:1952-2027,2145-2259`). Thus both `inputs_embeds` and `per_layer_inputs` are refreshable once sessions are backend-neutral. Literal server `pixel_values` discovery is also not on the text-only benchmark path, and decoder position/KV handling was not reached because native pipeline construction fails first.
**Why:** WP2/WP3 removed the previously suspected multi-output preprocessing and one-output step-binding limitations, but they did not add native multi-model session ownership. The correct next work is an architecture-neutral backend abstraction plus embedding-driven native target decode, followed by producer-emitted optional-modality metadata; changing runtime behavior based on Gemma/model names would violate RULES.md §2/§2.1.

<!-- source: .squad/decisions/inbox/joi-qwen3-0.6b-bench.md -->
### 2026-07-22: Qwen3-0.6B CUDA smoke passes but metadata disables shared KV
**By:** Joi
**What:** The exported Qwen3-0.6B int4 package loaded on ORT 1.27 CUDA and coherently generated ` Paris, and the capital of Italy is Rome. The capital of France is also the capital of the` for the 20-token smoke. As exported it measured 197.49 tok/s at 128 and 64.01 tok/s at 1024, equal to 1.57% and 0.61% of the explicit 3.35 TB/s weight+KV rooflines (12,585 and 10,549 tok/s). A metadata-only diagnostic using `attention.type: grouped_query_attention` plus `kv_cache.native_dtype: float16` reached 429.95/381.49 tok/s; matched Qwen2.5 ORT eager controls reached 570.61/501.88 tok/s.
**Why:** Qwen3 QK-norm and the graph are functionally supported, but this export is not a clean performance win. It lacks `genai_config.json`; its native metadata emits the unrecognized shared-KV alias `grouped_query` and omits the fp16 KV dtype, so onnx-genai selects growing `ZeroCopyRebind` despite device KV. Corrected metadata removes the length collapse, after which Qwen3 remains about 24% slower than Qwen2.5 under matched eager ORT conditions; QK-norm is only one contributor alongside 28 layers, 8 KV heads, and larger attention geometry.

<!-- source: .squad/decisions/inbox/keaton-native-specdecode-design.md -->
# Design: Speculative decoding on the FAST native CUDA decode path

- **Author:** Keaton (systems architect)
- **Date:** 2026-07-21
- **Status:** Design-only proposal (P0 blocker). No code changed. Read-only investigation.
- **Requested by:** Justin Chu
- **Scope:** Make speculative decoding *multiplicative* on the 762 tok/s native decode path, which today rejects speculation outright.

---

## 0. Problem statement & verified ground truth

Speculation is implemented and **correct** — token-identical to greedy across draft / prompt-lookup / MTP / EAGLE-3 / Gemma4 shared-KV proposers (`docs/PROGRESS.md` L244, L272). But it is trapped on the **ORT** path and **explicitly rejected on native**:

- `engine.rs:2340 reject_native_request_speculation()` bails for *every* non-`None` `SpeculativeMode` and for any `num_speculative_tokens`.
- Native engine construction hard-wires all proposers off: `draft/mtp/eagle3/shared_kv_proposer = None`, `speculative_mode = SpeculativeMode::None` (`engine.rs:1004-1011`).
- So native gets **zero** of the decode win; and the only measured *real* speculation (Gemma4 E2B shared-KV on H200) runs at **0.53× — slower than greedy** — because the drafter's lm_head compute cost exceeds the tokens it saves, and the drafter rarely gets 2+ ahead (`PROGRESS.md` L329: acceptance ~25%, `multi_token_accepts=0`; Leon's fix lifted this to 70.6% / 12-of-17 but raw speedup is still <1×, L272).

The native fast path today (verified in `native_decode.rs`):

- Plain decode is a **single-token (M=1) CUDA graph**: fixed-shape `input_ids [1,1]` / `position_ids [1,1]` device bindings, a `max_len`-capacity device KV whose *logical* length grows via `set_logical_len` while the *physical* pointer/topology stays invariant, and an attention mask grown in place via `extend_mask`. Topology is step-invariant; only **buffer contents** change → that is *why* M=1 is capture-safe with **0 fallbacks** (`run_one_token` L774-821; `DecodeCudaGraphPhase` NeedsWarmup→Armed→Ready).
- Greedy runs a **device argmax** on the single logits row and returns only a token id (`read_greedy_result` L833-845, `device_argmax`), so no host logit copy on the hot path.
- **A multi-token (M=K) eager path ALREADY EXISTS** and is exercised by prefill: `decode_cuda` L480-537 calls `invalidate_graph`, rebuilds **host** `input_ids/position_ids` of shape `[1,K]`, and runs `run_with_device_bindings(host_inputs, &mut state.bindings[..base_binding_count])` — host CPU inputs + device KV/mask bindings — returning full `[K,vocab]` host logits. This is the raw material for verify.
- **Rewind already exists and is device-KV correct**: `NativeDecodeSession::rewind` (L1006-1038) → for CUDA it calls `state.invalidate_graph` then `state.rewind(target_len)` (L756-763), which zeroes the mask tail `[target_len, logical_len)` and truncates the KV *logical* length. Physical buffer bytes beyond `target_len` are left stale but hidden by logical shape — correct, because the next pass overwrites those positions before they are ever attended.

Other verified facts used below:

- Multi-row `argmax_rows` / `sample_rows` device kernels exist in the **ORT** crate (`device_sampler.rs` L114-119 "handling rows>1 lets speculative decoding argmax every verified position in a single launch", L647-733) but are **UNUSED** — no caller outside `device_sampler.rs`. The native crate has its own *single-row* device argmax on `logits_binding`; it has **no multi-row primitive yet**.
- The ORT speculative acceptance loop copies the full `[K,vocab]` logits to host as `Vec<Vec<f32>>` and selects tokens host-side (`speculative.rs` L1440-1492).
- The native graph lifecycle uses `reset_device_graph()` + `DecodeCudaGraphPhase` (clean re-warm), **not** the ORT `gpu_graph_id` annotation-id protocol (`session.rs:754-810`). The "re-capture under a held annotation id corrupts the ORT heap" hazard is an **ORT-path** hazard; the native path sidesteps it entirely by destroying and rebuilding the graph exec via `reset_device_graph`. This is a structural advantage for native spec.
- `NgramProposer` (prompt-lookup) is **model-free**: `NgramProposer::new(ngram, max_tokens)` + `propose(ctx)` needs only the context tokens in `SpeculativeProposerContext` — no session, no runner, no lm_head (`speculative.rs:542-604`). Ideal first increment.
- The plain path runs through the shared `run_decode_loop`/`DecodeLoopBackend` via `NativeLoopAdapter` (`native_decode.rs:1049+`). The ORT speculative loop is a **separate** method (`generate_speculative`) — native must get its **own** driver, not reuse the ORT one.

---

## 1. Architecture: native speculative driver beside the plain native loop

Two peer drivers behind one dispatch, sharing the *same* `NativeDecodeSession`:

```
generate_native_with_callback (engine.rs)
    ├─ speculation OFF  → native_session.generate_with_callback   (EXISTING, untouched)
    │                        → run_decode_loop(NativeLoopAdapter)  → M=1 captured graph, device argmax
    └─ speculation ON   → native_session.generate_speculative(...) (NEW)
                             → NativeSpeculativeDriver loop:
                                  propose K  (NgramProposer, model-free, host)
                                  verify     NativeDecodeSession::decode_verify(&[t1..tK])  (M=K)
                                  accept     device multi-row argmax over [K+1, vocab]
                                  rewind     NativeDecodeSession::rewind(past + accepted)
```

Key structural decisions:

1. **The plain loop is never touched.** `generate_with_callback` → `run_decode_loop` → `NativeLoopAdapter` stays byte-identical. The M=1 captured-graph greedy fast path is unchanged. All new code lives in a *new* `generate_speculative` entry + a *new* `NativeSpeculativeDriver` struct + a *new* verify primitive. See §6 (non-regression).
2. **The driver owns the outer token loop itself** (it cannot use `run_decode_loop`, whose contract is one-token-per-step). It reuses the existing `ProcessorChain`, EOS handling, `max_new_tokens`/`max_context` checks, and streaming callback so behavior matches the plain loop for accepted tokens.
3. **Proposer stays host-side and pluggable** via the existing `SpeculativeProposerContext`. The n-gram proposer needs nothing device-side; later proposers (draft/shared-KV/MTP/EAGLE) plug into the *same* verify/accept/rewind machinery — only `propose()` changes.
4. **Verify + accept become the fast, device-resident core.** The target forward over K tokens is a single native pass; acceptance is a single device kernel returning K+1 token ids. No `[K,vocab]` host copy.

---

## 2. The M=K verify decision (the crux) + capture-safety analysis

Speculative verify must run K candidate tokens (K=2..8) through the target in **one** pass and read one predicted token per position. Three options:

### (a) Per-K CUDA-graph capture buckets
Capture a distinct graph for each fixed K in a small set (e.g. {2,4,8}), each with `input_ids/position_ids [1,K]` device bindings.
- **Capture-safety:** topology is fixed *per bucket*, so each is individually capture-safe *if* the M=K attention kernels pass the pre-capture audit. Risk: the 0-fallback property was proven only for M=1; K-query attention may trip a non-capturable op → a *new* fallback surface per bucket.
- **0-fallback property:** now must hold for N buckets, not 1. N× the audit risk, N× warmup cost (each bucket needs its own NeedsWarmup→Armed→Ready).
- **Device-KV correctness:** same data-driven mask/KV mechanism as M=1 → fine.
- **Perf:** best steady-state (graph replay on every verify) but forces the proposer to emit exactly a bucketed K (pad to nearest bucket).
- **Verdict:** highest steady-state perf, highest capture risk + warmup/memory cost. Over-engineered for increment 1.

### (b) Uncaptured eager M=K path — **ALREADY EXISTS** (`decode_cuda` L480-537)
Run K tokens eagerly (`invalidate_graph` + host `[1,K]` inputs + device KV bindings), return `[K,vocab]`.
- **Capture-safety:** N/A — no capture, so **zero** new capture-audit risk. Cannot regress the M=1 0-fallback property (different code path).
- **0-fallback property:** trivially preserved for the plain path (untouched); the spec path simply doesn't capture.
- **Device-KV correctness:** already proven — this is the prefill path; `set_logical_len(total_len)` advances KV correctly; `rewind` truncates.
- **Perf:** loses graph replay *for the verify pass only*. But: (i) the spec loop is **pure M=K** — it never interleaves M=1 steps, so there is **no capture thrash** (`invalidate_graph` fires once at entry, not per token); (ii) batching K tokens into one pass amortizes the ~227 launches/token across K tokens (≈K× fewer launches/token), which recovers much of what graph capture bought (the graph win was *CPU launch-overhead* elimination); (iii) target decode is weight-bandwidth-bound, so K query rows cost ≈ same wall-time as 1 row (weights streamed once).
- **Verdict:** simplest, correct today, near-zero risk, and *already implemented*. The perf gap vs. capture is smaller than it looks because of launch amortization + bandwidth-bound verify.

### (c) Single fixed M=maxK captured graph with padding
Capture **one** graph at `M = maxK`; when the proposer offers fewer than maxK, pad unused query rows with dummy input ids (their logits are ignored, their KV writes are rewound).
- **Capture-safety:** a **single** topology → a single pre-capture audit, exactly like M=1. The 0-fallback property is proven **once** for maxK and then holds for every step. This is the *only* option that keeps "one audited, capture-safe graph."
- **0-fallback property:** one graph, one audit — cleanest.
- **Device-KV correctness:** same data-driven mask/KV mechanism; padded rows write KV at positions `[past+realK, past+maxK)` which are **rewound** every step, so they never leak into attention of committed tokens. Mask marks all maxK positions valid during the pass.
- **Perf:** graph replay on **every** verify pass (full 762-class launch reduction) *plus* K-token batching. Cost: always computes maxK rows even when fewer are proposed — but decode is bandwidth-bound, so the marginal cost of extra query rows is small (weights loaded once). This is the classic "verification is nearly free" property that makes speculation multiplicative.
- **Verdict:** best end-state — graph replay + single audit + bandwidth-cheap padding.

### Recommendation

**Target architecture = (c) single fixed M=maxK captured, padded graph.** It is the only option that preserves the hard-won 0-fallback graph-replay win on the verify pass while keeping a *single* capture audit, and its padding cost is nearly free on a bandwidth-bound decode.

**But sequence through (b) first.** Increment 1 lands the verify/rewind/accept machinery on the **existing eager M=K path** (option b) — zero capture risk, correct today, fastest path to an end-to-end measurable number. Only after correctness + a measured speedup are proven does WP-later graduate the verify pass to (c) as a *pure perf lever*, gated behind the same flag. This de-risks the P0: we prove the loop works before we touch capture.

> **One-line rule for the coordinator:** ship (b) to prove it, upgrade to (c) to make it fast. Both behind the speculation flag; the plain M=1 path is never in the blast radius.

---

## 3. Rewind / KV-correctness protocol

The accept loop must, per outer step:

1. `past = session.current_len()` (committed length).
2. Propose `K` tokens (host).
3. `verify_logits_or_ids = session.decode_verify(&draft[0..K], past)` — advances device KV to `past+K`, mask valid over `[0, past+K)`.
4. Determine `accepted = j` (longest prefix where target argmax == draft) and the **bonus token** `t_{j}` = target argmax at position `j` (the free correct token verify always yields).
5. Commit `draft[0..j] ++ [t_j]` (that is `j+1` tokens) to the output/stream/processor chain.
6. `session.rewind(past + j + 1)` — roll device KV back to the committed length (dropping the `K-j-1` unaccepted KV columns and, under option (c), the padded columns).

Correctness invariants (all verified against existing code):

- **`rewind` is device-KV correct.** `state.rewind(target)` zeroes mask `[target, logical)` and truncates KV logical length (L756-763). Because unaccepted/padded positions are always `> target`, their stale physical bytes are hidden and are overwritten by the *next* pass before being attended. No corruption.
- **Graph state.** Under **(b)** verify already runs `invalidate_graph` (L480) so there is no captured state to corrupt; the *next* verify re-warms cleanly via `NeedsWarmup`. Under **(c)** the captured M=maxK graph must **not** be invalidated on the accept/rewind (rewind only rewrites mask/KV *contents* + logical shapes, which is exactly the data-driven mutation the captured graph tolerates — identical in spirit to how M=1 replays across growing KV). `rewind`'s current `invalidate_graph` call is correct for (b) but must be made **conditional** for (c): keep the capture, rewind only buffer contents/shapes. This is the single subtle correctness point for the (c) upgrade and gets its own test.
- **No annotation-id hazard.** Native uses `reset_device_graph` + phase reset, not `gpu_graph_id` held-id re-capture — the ORT heap-corruption hazard does not apply. This is why native is a *safer* home for spec than ORT.
- **CPU inputs rebind each step.** The eager M=K path already rebuilds host `input_ids/position_ids` per pass (L481-496); under (c) the padded input-id/position buffers are device bindings whose *contents* are rewritten each step (like the M=1 `write_decode_inputs`), never re-topologized.

---

## 4. Device-side verified-row acceptance (API change)

**Today (ORT):** verify returns `Vec<Vec<f32>>` `[K+1, vocab]`; host runs argmax/sampling (`speculative.rs` L1440-1492). That is `(K+1) × vocab × 4 bytes` host copy per step (e.g. 9 × 151936 × 4 ≈ 5.5 MB/step at K=8) — a real cost that erodes speedup.

**Native target:** select target ids **on-device** and copy back only `K+1` token ids (≈36 bytes).

- **New native primitive:** `DecodeCudaState::verify_argmax_rows(rows) -> Vec<TokenId>` (or write into a caller `[u32; maxK+1]` to avoid per-step alloc, mirroring `argmax_into`). It runs the multi-row argmax kernel over the `[rows, vocab]` device logits binding — the native analogue of the ORT `argmax_rows` that already exists but is unused. Reuse the same one-block-per-row kernel design (`device_sampler.rs` L114-119) in the native/ep-cuda crate, or lift the ORT kernel into a shared location. It must also poll the shared capture-error word (same detection-before-consumption discipline as `read_greedy_result`, L833-845) so a bad replay rejects the whole step.
- **New verify API on the backend:**
  - `decode_verify_argmax(&mut self, draft: &[TokenId], past: usize) -> anyhow::Result<Vec<TokenId>>` returns the `K+1` **selected target ids** (positions `0..=K`), not host logits. The driver compares element-wise with `draft` to find `accepted`, and reads the bonus token at index `accepted`.
  - Greedy-only for increment 1 (`options.greedy || temperature==0`). This is exactly the regime where the M=1 device argmax fast path already applies, so it is behavior-consistent.
- **Where sampling slots in later:** temperature/top-k/top-p verify uses the existing device `sample_rows` (`device_sampler.rs` L745-756) instead of `argmax_rows`, with the documented shared-rng-per-row simplification. That is a *later* WP; increment 1 is greedy verify only. For non-greedy / processor-chain / logprobs requests, the driver **falls back** to reading host `[K,vocab]` logits and using the existing host `select_next_token_with_rng` path (correctness over speed) — never disabling the plain fast path.

---

## 5. Proposer sequencing on the fast path

**First increment = prompt-lookup / n-gram (`NgramProposer`).** Confirmed correct choice:

- **Model-free:** `NgramProposer::new(ngram, max_tokens)` + `propose(ctx)` uses only context tokens — no draft session, no lm_head, no KV, no device residency (`speculative.rs:542-604`). It isolates the variable under test to *exactly* the native verify/rewind/accept machinery.
- **Measurable win without draft complexity:** on a repetitive/long-context prompt (code, JSON, retrieval, chat with quoted context) n-gram acceptance is high and its proposal cost is ≈0, so any accepted token is pure speedup. This proves the native loop is multiplicative *before* any draft-model cost is in play.
- **Same socket for everything else:** once the native verify/accept/rewind loop is proven with n-gram, `DraftModelProposer` / `SharedKvProposer` / MTP / EAGLE-3 plug into the identical loop — only the `propose()` branch changes (they already all produce `draft_tokens: Vec<TokenId>` via `SpeculativeProposerContext`). No further verify/rewind work needed for them.

Sequencing: **n-gram (WP4) → draft-model / shared-KV (follow-on, out of scope here but unblocked).**

---

## 6. Why real Gemma spec is <1×, and whether native flips it

**Root cause (verified, `PROGRESS.md` L272, L329):** the Gemma4 E2B shared-KV drafter runs its own internal **lm_head** to propose each token. On the ORT path that drafter forward + lm_head + host logit copies costs a large fraction of a target step; with acceptance ~25% and `multi_token_accepts=0` (drafter never gets 2+ ahead), the target still runs nearly every position, so total work = target + drafter > target alone → **0.53×**. Leon's embedding fix (`inputs_embeds = concat(input_embedding(last), last_hidden)`) lifted acceptance to 70.6% and `multi_token_accepts` 0→12/17 — proving the *acceptance* side is fixable — but raw speedup is still <1× because the **drafter cost per proposal** dominates.

**Does the fast path plausibly flip it >1×? Honest answer: for a *cheap/free* proposer, yes; for the current Gemma drafter, only maybe, and not from the native loop alone.**

- The native fast path removes several *fixed* costs that erode speedup: **no `[K,vocab]` host logit copy** (device argmax returns K+1 ids), **graph-replayed verify** (option c), **aliased shared-KV** (no re-materialization), and **launch amortization** across K. These raise the ceiling for *every* proposer.
- **For n-gram (proposal cost ≈ 0):** speedup ≈ `1 + E[accepted]` limited by verify overhead. On a favorable prompt this is comfortably >1× — this is the increment we can *prove* flips the sign.
- **For the Gemma drafter:** the binding constraint is **drafter lm_head compute vs. tokens saved**, not host-copy overhead. Native helps (device-resident drafter, aliased shared-KV, no host round-trips) and *raising acceptance to 70%* helps, but if the drafter forward is an appreciable fraction of a target step, the acceptance rate must clear `draft_cost_fraction` for net >1×. Native improves the constant factors on both sides; whether that clears the bar is an **empirical** question that only the benchmark answers. **We must not promise native alone flips Gemma >1×.** The honest framing: native is a *necessary* enabler (it removes the overheads that made even n-gram impossible), and it *plus* the acceptance fix *plus* a cheaper drafter head is the plausible route. The acceptance-rate ↔ draft-cost tradeoff is fundamental: `speedup ≈ (1 + α·K) / (1 + c)` where α is per-position acceptance and c is drafter-cost-as-fraction-of-target; native shrinks `c` (device residency, aliasing) and the fix raises `α`, but neither guarantees the product > 1 for an expensive head.

---

## 7. Non-regression guarantee (the 762 tok/s path stays byte-identical)

Zero-cost-when-off, enforced structurally:

1. **Separate entry point.** Speculation dispatches to a *new* `generate_speculative`; when off, control never enters new code — `generate_with_callback` → `run_decode_loop` → `NativeLoopAdapter` is untouched, so the M=1 captured-graph greedy path is *literally the same instructions*.
2. **Flag/request-gated.** Wire speculation through the existing `SpeculativeMode` + `num_speculative_tokens` request options (config.rs:381). `reject_native_request_speculation` is *narrowed* (WP2): reject only the *unimplemented* modes; allow the implemented native mode(s). Default remains `SpeculativeMode::None` → plain path.
3. **No shared mutable state added to the hot path.** The new verify primitive and multi-row argmax live in new methods; the M=1 `run_one_token` / `read_greedy_result` are not edited. The `rewind` change for option (c) is guarded so the *plain* path (which never calls verify) sees identical behavior.
4. **Guard test:** a byte-identical + tok/s non-regression check — same prompt, spec off, asserts token stream and a tok/s floor vs. the current 762-class baseline (extend the existing native decode bench/soak). This is WP4's exit gate.

---

## 8. Transparency (Deckard's trace spans + kernel-variant records)

Preserve and extend observability:

- The verify pass reuses the existing `TraceContext` and per-op spans already threaded through `run_one_token`/`decode_cuda` (`self.trace`, `trace_capture_declines`). The M=K verify pass must emit its own span so per-op timings remain attributable.
- **New per-step speculative record** (extend `SpeculativeStats`, already tracked: `proposed_tokens`, `accepted_tokens`, `multi_token_accepts`, `verification_steps`): add/emit **K (proposed)**, **accepted count**, and **per-phase timing** (propose / verify-forward / device-argmax-accept / rewind). Surface via the same stats channel the ORT path uses so dashboards are uniform.
- **Kernel-variant records:** the multi-row argmax and the M=K verify forward must register their kernel-variant identity (dtype path, rows, vocab) the same way the M=1 argmax + GEMV variants do, so Deckard's kernel-variant ledger stays complete for the fast path.
- Capture-error polling (option c) reuses the existing shared capture-error word + detection-before-consumption, so a bad replay is *observable* (latched flag → structured bail) rather than silent.

---

## 9. Phased, FILE-DISJOINT work packages

Ordered by dependency. Files are chosen to minimize overlap so the coordinator can fan out WP1/WP3 primitives in parallel, then WP2 wires, then WP4 proves.

### WP1 — Native M=K verify + rewind primitive (option b) + (c)-ready rewind guard
- **Files:** `crates/onnx-genai-engine/src/native_decode.rs`; `crates/onnx-runtime-ep-cuda/src/graph.rs` (+ its executor glue for the padded-capture upgrade only).
- **Do:** expose `decode_verify(draft, past) -> [K,vocab]`/`decode_verify_argmax(draft, past) -> Vec<TokenId>` built on the existing eager M=K path (L480-537); make `rewind`'s `invalidate_graph` call **conditional** so option (c) can retain the captured graph across rewind (contents-only mutation). Add the padded single-M=maxK capture behind an internal switch (dormant until WP4 flips it).
- **Depends on:** nothing (uses existing eager path + rewind).
- **Risk:** Medium. The (c) rewind-without-invalidate correctness is the one subtle point — ship (b) first with `invalidate_graph` retained, land (c) guarded.
- **Exit criterion:** unit test — decode(K) then rewind(past+j) leaves `current_len == past+j`, mask/KV logical shapes correct, and a subsequent decode produces logits **bit-identical** to a fresh M=1 decode from that committed prefix (proves no KV corruption).

### WP2 — Stop rejecting + native speculative driver
- **Files:** `crates/onnx-genai-engine/src/engine.rs` (narrow `reject_native_request_speculation` L2340; add `generate_speculative` dispatch + native proposer wiring in the native constructor L991-1011); `crates/onnx-genai-engine/src/config.rs` (validate the allowed native mode).
- **Do:** allow prompt-lookup on native; construct the `NgramProposer`; add the outer `NativeSpeculativeDriver` loop (propose → verify → accept → rewind → commit/stream) reusing `ProcessorChain`/EOS/`max_new_tokens`/callback.
- **Depends on:** WP1 (verify+rewind primitive), WP3 (accept API — can stub with host argmax until WP3 lands).
- **Risk:** Medium. Loop correctness (EOS mid-accepted-run, context-limit mid-verify, streaming order).
- **Exit criterion:** native prompt-lookup produces a token stream **identical to native greedy** on a fixed prompt (env-gated integration test, mirrors `tests/milestone_b_real.rs` style).

### WP3 — Device verified-row acceptance
- **Files:** `crates/onnx-genai-ort/src/device_sampler.rs` (expose/relocate the multi-row `argmax_rows` so native can use it — it is currently unused/private); `crates/onnx-runtime-ep-cuda/` native argmax wiring; native branch in `crates/onnx-genai-engine/src/speculative.rs` (or a new `native_speculative.rs`) that calls device accept instead of the host `Vec<Vec<f32>>` compare.
- **Do:** wire `decode_verify_argmax` to a multi-row device argmax over `[K+1, vocab]`, returning only K+1 ids + capture-error poll. Keep host-logit fallback for non-greedy/processor/logprobs.
- **Depends on:** WP1 (verify pass produces the device logits binding).
- **Risk:** Medium — dtype coverage (f16/bf16/f32), capture-error latch integration.
- **Exit criterion:** greedy verify selects the **same** K+1 ids as the host `[K,vocab]` argmax reference, with **zero** `[K,vocab]`→host copies on the greedy path (assert via transfer-stats counters, like `CudaKvDebugStats`).

### WP4 — Prompt-lookup native e2e + benchmark/test + option-(c) enable
- **Files:** `crates/onnx-genai-bench/` (new native-spec bench scenario); `crates/onnx-genai-engine/tests/` (native spec e2e + non-regression guard); scripts under `scripts/`.
- **Do:** end-to-end native prompt-lookup on a repetitive prompt; measure tok/s vs. plain native; flip the option-(c) captured verify on and compare; add the **spec-off byte-identical + tok/s-floor** non-regression guard (§7).
- **Depends on:** WP1–WP3.
- **Risk:** Low-Medium — bench variance on shared H200 (use median-of-3, matching the 762 methodology).
- **Exit criterion:** **>1× decode speedup** on a favorable prompt with token-identical output, AND spec-off path proven byte-identical at the 762-class tok/s floor. This is the go/no-go gate (§11).

---

## 10. Expected speedup (honest range)

- **Prompt-lookup increment (favorable, repetitive prompt):** **1.3×–2.2×** decode. Proposal cost ≈ 0, so speedup ≈ `1 + E[accepted per step]` discounted by verify overhead; with option (c) graph-replayed verify and K≈4–8 and moderate n-gram hit rates this lands comfortably >1×. On **non-repetitive** prompts n-gram acceptance collapses → **~1.0× (neutral)**; the driver must early-exit to the plain path when the proposer returns empty, so worst case is *no regression*, not a slowdown.
- **Real Gemma-4 E2B draft increment:** honest range **0.7×–1.4×**, *conditional*. Native removes host-copy + launch overhead and aliases shared-KV; with acceptance ~70% (post-fix) it *can* clear 1× **iff** the drafter head cost stays a small fraction of a target step. If the lm_head remains expensive, expect **<1× even on native** — that is a *proposer-cost* problem native cannot fully solve. Do not commit to >1× for Gemma without the WP4 measurement.

---

## 11. Kill-criterion

Abandon native spec (or park it behind an off-by-default flag with a documented negative result) if, after WP4:

- **Prompt-lookup cannot beat 1.15× median** (3-sample) on a *favorable* repetitive prompt on H200 — i.e. even the zero-cost proposer with device-side accept and graph-replayed verify fails to be multiplicative. That would mean the verify/rewind overhead itself eats the win, and no real draft model can succeed either.
- **OR** the spec-off path cannot be kept byte-identical at the 762-class tok/s floor — i.e. enabling the feature regresses the plain path. Non-negotiable.

If prompt-lookup clears >1.15× but Gemma draft stays <1×, that is **not** a native-spec kill — it is a *drafter-cost* finding: keep native prompt-lookup (shipped win) and route the Gemma head-cost problem to a separate proposer-optimization track (cheaper/quantized drafter lm_head, MTP/EAGLE with smaller heads).

---

## Plain-text summary

- **Recommended M=K verify approach:** target end-state **(c) — one fixed M=maxK CUDA graph with padding** (single capture audit, preserves the 0-fallback graph-replay win, padding is ~free on bandwidth-bound decode), but **sequence through (b) — the eager M=K path that already exists** for increment 1 (zero capture risk, correct today). Ship (b) to prove it; upgrade to (c) to make it fast. The plain M=1 path is never in the blast radius.
- **First increment / proposer:** **prompt-lookup / n-gram** — model-free (`NgramProposer::new(ngram, max_tokens)`), no draft LM, isolates the native verify/rewind/accept machinery and gives a measurable win on repetitive prompts. Draft-model / shared-KV / MTP / EAGLE plug into the identical loop afterward.
- **Four work packages in order:** WP1 native M=K verify+rewind primitive (`native_decode.rs` + `ep-cuda/graph.rs`) → WP2 stop-rejecting + native speculative driver (`engine.rs` + `config.rs`) → WP3 device verified-row acceptance (`device_sampler.rs` multi-row argmax + native accept branch) → WP4 prompt-lookup e2e + benchmark + non-regression guard (`onnx-genai-bench` + tests). WP1 and WP3 primitives can start in parallel; WP2 wires; WP4 proves.
- **Honest expected speedup:** prompt-lookup **1.3×–2.2×** on favorable prompts (**~1.0×, no regression** on unfavorable, via early-exit); real Gemma-4 draft **0.7×–1.4×, conditional** on the drafter lm_head cost — native removes host-copy/launch/KV-realization overhead and the acceptance fix raises hit rate, but an expensive drafter head can still keep Gemma <1× (a proposer-cost problem, not a native-loop problem).
- **Go/no-go:** GO if native prompt-lookup beats **1.15× median (3-sample)** on H200 with token-identical output **and** the spec-off path stays byte-identical at the 762-class tok/s floor. KILL native spec only if the zero-cost proposer itself can't clear 1.15× or if enabling the feature regresses the plain path. A Gemma-draft <1× result is a drafter-cost finding, not a native-spec kill — keep the shipped prompt-lookup win and split the head-cost problem into its own track.

<!-- source: .squad/decisions/inbox/kowalski-zhora-mla-conformance-review.md -->
### 2026-07-22: Zhora DeepSeek MLA conformance review
**By:** Kowalski
**What:** 🟢 APPROVE `91e747d27d8309da27671cc2263580f624d94487`.
**Why:** Independent review found the focused CPU Attention conformance addition correct, green, model-agnostic, and scoped.

Evidence:
- Diff versus `origin/main` is exactly 51 test-only lines in `crates/onnx-runtime-ep-cpu/src/kernels/attention.rs`; there are no `executor.rs`, `.squad/`, documentation, or unrelated changes. `git diff --check` passed.
- The test genuinely uses asymmetric widths: Q/K head dimension `2`, V head dimension `1`; Q has hidden width `4*2=8`, K `2*2=4`, V `2*1=2`, and Y is allocated/asserted at `4*1=4`. It also exercises GQA (`4` query heads mapped in pairs to `2` KV heads), non-empty cached decode, and present K/V concatenation.
- Independent SDPA recomputation used `scores = QK^T/sqrt(2)`. Since each Q vector is zero, every head has scores `[0,0,0]` and softmax probabilities `[1/3,1/3,1/3]`. KV head 0 values `[1,3,5]` produce `3`; KV head 1 values `[2,4,8]` produce `14/3`. GQA therefore produces `[3,3,14/3,14/3]`, exactly matching the golden. Independently verified cache layouts are present K `[1,0,0,1,10,20,2,0,0,2,30,40]` and present V `[1,3,5,2,4,8]`.
- `cargo build -p onnx-runtime-ep-cpu`: passed.
- `cargo clippy -p onnx-runtime-ep-cpu -- -D warnings`: passed.
- `cargo test -p onnx-runtime-ep-cpu mla_gqa_decode_with_asymmetric_head_dims_matches_hand_computed_result -- --nocapture`: passed (`1 passed`, `646 filtered out`).
- Native CPU DeepSeek-V2-tiny E2E was independently rerun with `--features native-backend`; it passed and generated `[42, 237, 198, 2, 186, 81, 210, 149]`. The first attempt without that required feature failed at configuration validation, then the correctly configured command passed.
- RULES.md model-name grep found only pre-existing `llama.cpp` provenance comments/test names in `block_quantized_matmul.rs`; the changed code adds only an `mla` test name/comment and no model-specific runtime logic.

<!-- source: .squad/decisions/inbox/kowalski-zhora-unsqueeze-review.md -->
### 2026-07-22: APPROVE zhora DeepSeek Unsqueeze test coverage
**By:** Kowalski
**What:** Reviewed the test-only Unsqueeze dynamic-output-shape regression coverage and approved it. The opset-17 input axes `[0, -1]` correctly produce `[1, 2, 3, 1]`, and the opset-11 attribute axes `[1, -1]` correctly produce `[2, 1, 3, 1]`.
**Why:** `cargo fmt -p onnx-runtime-session --check`, `cargo clippy -p onnx-runtime-session --all-targets -- -D warnings`, `cargo test -p onnx-runtime-session dynamic_output_shapes_unsqueeze -- --nocapture`, and `cargo test -p onnx-runtime-session` all passed. Fast-forward merged SHA: `8d9d2fa279a0463b9fc6c02932ebd7b9ec775fb4`.

<!-- source: .squad/decisions/inbox/leon-vlm-scope.md -->
### 2026-07-21: Next VLM runtime scope
**By:** Leon (Engine Dev)
**Requested by:** Justin Chu
**Scope:** Read-only investigation of onnx-genai runtime, Mobius export, and local validation assets. This file is the only artifact created.

# Executive conclusion

The repository has a real but narrow multimodal vertical slice: a metadata-declared `vision_encoder -> embedding/fusion -> inputs_embeds decoder` pipeline works end-to-end for the committed deterministic `tiny-gemma4-vlm` fixture. Native image loading, rank-4 RGB resize/normalize/tiling, one-placeholder expansion, and OpenAI `image_url` ingestion also exist.

No real Mobius VLM package found locally is currently runnable end-to-end by the server without additional work. The first recommended real target is **Gemma4 E2B**, because its gated source checkpoint and processor config are already cached locally and Mobius exports its three-model ONNX topology. The shortest correct path is not a Gemma-specific branch: it is a typed, architecture-neutral multimodal tensor contract; a generic multi-output image processor; generic per-step pipeline component execution; and native Mobius metadata emission.

Qwen2.5/3-VL is the next architecture class after Gemma4. The complete local Qwen3.5 Foundry package is valuable as a stress target, but it additionally needs 3-axis MRoPE and generic fixed recurrent-state carry, so it should not be the first runtime milestone.

# A. Current VLM state

## A1. Runtime: what already works

### Composite vision -> fusion -> autoregressive decode

`PipelineEngine::run_autoregressive` tokenizes, optionally expands image placeholders, seeds prompt component inputs, executes prompt components, then drives the decoder (`crates/onnx-genai-engine/src/pipeline.rs:329-428`). The Gemma4-style seams are:

- `seed_prompt_token_inputs` supplies prompt IDs to an embedding/fusion model (`pipeline.rs:1483-1517`).
- `embeds_step_binding` finds a decoder fed through `inputs_embeds`, resolves its embedding component, reuses prompt embeddings for prefill, and re-runs that component for each generated token (`pipeline.rs:1519-1617`).
- `PipelineDecodeLoopBackend::step_extras` implements the prefill-versus-single-token embedding behavior (`pipeline.rs:1776-1837`).

The committed proof is `crates/onnx-genai-engine/tests/gemma4_vlm_pipeline_e2e.rs:1-24,42-66`, backed by `tests/fixtures/tiny-gemma4-vlm/inference_metadata.yaml` and `scripts/build_tiny_gemma4_vlm.py:2-45,329-381`. It asserts exact generated IDs `[0,5,6,7]`, including a first token that depends on image features. This is a genuine contract test, but the fixture deliberately has only one rank-4 image tensor, one embedding output, ordinary KV, and no position-ID input.

### Image ingestion and preprocessing

`onnx-genai-preprocess` already provides:

- RGB decoding, resize, interpolation, crop/pad, normalization, fixed/dynamic-anyres tiling, and thumbnail ordering (`crates/onnx-genai-preprocess/src/image.rs:395-537,559-724`).
- A rich multi-image placeholder expander with separate placeholder/image token IDs, per-image tile grids, row/column separators, and thumbnail ordering (`image.rs:67-97,169-337`).

The server accepts base64 data URIs and HTTP(S) image URLs with size/time limits (`crates/onnx-genai-server/src/image_input.rs:44-107`). Chat routes validate image use against pipeline handles and invoke preprocessing (`crates/onnx-genai-server/src/routes.rs:1218-1248,1305-1320`).

### Structurally compatible class today

With a hand-authored native sidecar, the current path can support a single-image, rank-4 RGB VLM whose:

1. vision encoder has exactly one `pixel_values` float32 input;
2. embedding graph accepts prompt IDs plus fixed image features and emits only the decoder's `inputs_embeds` sequence input;
3. decoder has no simultaneous raw token input, uses ordinary rank-2 positions or no positions, and has ordinary declared K/V pairs; and
4. prompt expansion is a uniform repetition of one placeholder token.

That can describe a simplified LLaVA/PaliGemma/InternVL-style three-model split, but no real such package is committed or locally validated in this repository.

## A2. Runtime: what is not wired

### Server discovers one literal, rank-4 RGB input

Pipeline startup searches sessions for an input literally named `pixel_values`, rejects more than one match, requires `Float32`, and constructs the rank-4 `ImagePreprocessor` (`crates/onnx-genai-server/src/state.rs:409-435`). `ImagePreprocessor::from_input_and_metadata` rejects any rank other than four and identifies RGB layout from a dimension equal to 3 (`crates/onnx-genai-preprocess/src/image.rs:395-418`). `VisionInputSpec` and server `ImageTensor` carry only one endpoint, one `Vec<f32>`, and aggregate tile count (`crates/onnx-genai-server/src/image_input.rs:9-41,44-61`).

This excludes all three priority real interfaces:

- **Gemma4 E2B/12B:** Mobius emits `pixel_values [B,N,3*P^2]` plus `pixel_position_ids [B,N,2]` (`/home/justinchu/mobius/src/mobius/tasks/_gemma4.py:317-362`).
- **Qwen2.5/3/3.5-VL:** Mobius emits packed `pixel_values [total_patches,pixel_dim]` plus `image_grid_thw [num_images,3]` (`.../_vision_language_3model.py:98-157`).
- **Phi4 multimodal:** Mobius emits rank-4 pixels plus `image_sizes` and `image_attention_mask` (`.../_phi4mm_multimodal.py:86-129`).

### Rich expansion exists but the live path discards it

The engine's live API carries only `num_image_tiles` (`pipeline.rs:71-103`). `expand_image_placeholders_count_based` supports exactly one placeholder, multiplies `tokens_per_tile * aggregate_tiles`, and repeats the placeholder token itself (`pipeline.rs:1629-1736`). The server reduces preprocessing output to aggregate `num_tiles` (`image_input.rs:56-61`; `routes.rs:1315-1320`). It therefore loses per-image tile counts, grids, thumbnail placement, separate image-token IDs, and separators already represented by `ImageTilingSummary`/`TokenExpansionConfig`.

The server also performs its context-cap check before image preprocessing and placeholder expansion (`routes.rs:1289-1305`), so request accounting is based on the unexpanded prompt rather than the actual prefill length.

### Per-step fusion is a one-output special case

`embeds_step_binding` deliberately returns `None` whenever the decoder has any token-ID input (`pipeline.rs:1537-1544`). It recognizes `inputs_embeds` and the fusion token input by historical names (`pipeline.rs:1545-1579`), and `step_extras` extracts only one configured fusion output (`pipeline.rs:1814-1836`). Other dataflow inputs are captured once by `decoder_extra_inputs` and reused unchanged (`pipeline.rs:1457-1480`).

That is insufficient for:

- **Gemma4 E2B:** Mobius embedding can emit both `inputs_embeds` and `per_layer_inputs`; the decoder consumes both (`/home/justinchu/mobius/src/mobius/tasks/_gemma4.py:238-247,271-309,418-464`). Both are sequence-dependent and must be regenerated for the single running token after prefill. Today `per_layer_inputs` would be frozen at prompt shape/content.
- **Gemma4 12B/bidirectional variants:** decoder may consume both `input_ids` and routed embedding outputs (`_gemma4.py:283-309`). Runtime explicit I/O rejects simultaneous `token_input` and `inputs_embeds_input` (`crates/onnx-genai-engine/src/decode.rs:278-313`), and the special fusion path opts out when `input_ids` exists.
- **Phi4 multimodal:** embedding can emit `vision_gate` and `speech_gate` alongside `inputs_embeds`; the decoder consumes those gates (`_phi4mm_multimodal.py:175-232`). The current converter/runtime does not route and refresh all of them.

The general missing primitive is: **execute declared upstream `every_step` components in topological order on prefill and decode, then route all of their outputs into the decoder**. The existing `phases` schema can express this intent, but the autoregressive executor currently collects only `prompt_only` components (`pipeline.rs:3903-3919`), so a component marked `every_step` is skipped unless covered by the one-output embedding special case.

### Decoder positions are fixed to ordinary 1-D positions

`run_decode_step_with_extra` always constructs `position_ids` as `[1, sequence_length]` (`crates/onnx-genai-engine/src/decode.rs:1543-1573`). Mobius explicitly exports Qwen VL decoders with `[3,batch,sequence]` MRoPE positions (`/home/justinchu/mobius/src/mobius/tasks/_base.py:200-220,253-257`). The real local Qwen3.5 decoder confirms `position_ids [3,B,S]`.

### Decoder state is KV-specific

`ResolvedIo` only represents positional K/V input-output pairs (`decode.rs:249-361`), and the decode input builder supports token IDs, mask, positions, KV, or fixed pipeline extras (`decode.rs:1553-1595`). The real Qwen3.5 decoder has only eight ordinary attention K/V layers (3, 7, ..., 31) plus 24 layers each of fixed-shape `conv_state` and `recurrent_state`; those states must be zero-initialized and replaced by their matching outputs every step. They are neither K/V nor fixed conditioning.

### Pipeline package discovery requires native metadata

`PipelineModelDirectory::load` requires `inference_metadata.{yaml,yml,json}` and loads the pipeline spec from it (`crates/onnx-genai-ort/src/loader.rs:82-135`). The server chooses pipeline mode only when native metadata already contains `pipeline` (`crates/onnx-genai-server/src/state.rs:349-367`). Single-model loading has `genai_config.json` compatibility conversion, but pipeline loading does not.

The current `genai_config` multimodal converter is not sufficient by itself: it creates vision, embedding, and decoder models/dataflow, but marks embedding `every_step` (`crates/onnx-genai-genai-config/src/lib.rs:606-694`) while `composite_encode_decode` includes only the first encoder and decoder stages (`lib.rs:904-920`). It also emits only `image_placeholder_token_id`, not a full preprocessing/expansion contract.

## A3. Mobius export state

### Architectures Mobius can build as ONNX components

Current Mobius registrations include Gemma4 and Gemma4 unified, LLaVA variants, BLIP-2, Idefics, InternVL, PaliGemma, Pixtral/Mistral3, SmolVLM, Mllama, Phi4 multimodal, Qwen2/2.5/3-VL, hybrid Qwen3.5-VL, Hunyuan VL, and others (`/home/justinchu/mobius/src/mobius/_registry.py:556-609`). Important concrete builders are:

- generic three-model vision/embedding/decoder and Qwen packed-patch/MRoPE variants (`src/mobius/tasks/_vision_language_3model.py:34-186`);
- Gemma4 E2B and unified 12B vision, embedding, and decoder interfaces (`src/mobius/tasks/_gemma4.py:227-362,418-527`);
- Phi4 multimodal four-model interfaces (`src/mobius/tasks/_phi4mm_multimodal.py:80-232`).

Registry presence means the graph builder exists, not that every architecture has a passing real-model ORT golden test.

### Native onnx-genai metadata emission is currently decoder-only

The checked-out Mobius `src/mobius/integrations/onnx_genai/inference_metadata.py` derives only attention dimensions, context length, and KV dtype (`lines 58-89`) and writes that dictionary as `inference_metadata.yaml` (`lines 129-150`). It does not inspect `ModelPackage.models`, emit `pipeline`, emit component I/O/dataflow/phases, copy tokenizer/processor assets, or emit image preprocessing/token expansion.

Mobius's ORT-GenAI exporter is materially ahead: it detects VLM packages, introspects vision/embedding graph ports, selects processor config assets, and emits multimodal mappings (`src/mobius/integrations/ort_genai/auto_export.py:757-804`). That implementation is the best producer-side reference, but its output contract is ORT-GenAI-specific.

`docs/PROGRESS.md:814-818` claims Mobius commit `f313bd1` added native composite emission. That object is absent from every current Mobius ref (`git branch -a --contains f313bd1` reports a malformed/unknown object), and the current source contradicts the note. Treat the progress entry as historical/unverified, not current capability.

# B. Concrete export-to-runtime gap list

1. **No portable typed VLM package contract.** `InferenceMetadata` has no typed top-level preprocessing section (`crates/onnx-genai-metadata/src/schema.rs:8-86`). `PipelineVisionConfig` has only placeholder ID and tokens-per-tile (`schema.rs:698-743`). The preprocessor privately parses an untyped `preprocessing.image` document (`image.rs:348-355,540-556`).
2. **Mobius does not emit native composite metadata.** It exports the ONNX components but not the sidecar that names ports, phases, preprocessing, expansion, positions, and state.
3. **Runtime image preprocessing returns one rank-4 float tensor.** Real Gemma4, Qwen VL, and Phi4 require packed/rank-3 or auxiliary tensors and model-declared dtype.
4. **Server discovers inputs by the literal name `pixel_values`.** It cannot consume metadata-declared endpoint bundles and rejects non-float32 input.
5. **Per-image prompt structure is dropped.** The live request carries aggregate tile count only; multi-image ordering and separator semantics cannot be correct.
6. **Actual expanded length is not available at admission/context checking.** Preprocessing and expansion must precede final context/KV sizing.
7. **Autoregressive `every_step` components are not generally executed.** The special case refreshes only one `inputs_embeds` output and cannot handle Gemma4 `per_layer_inputs` or Phi gates.
8. **Decoder I/O forbids valid dual sequence inputs.** A graph cannot declare both raw `input_ids` and routed `inputs_embeds` even if both are real graph inputs.
9. **Position construction is fixed to rank-2 linear IDs.** Qwen VL MRoPE cannot run.
10. **Loop-carried state is KV-only.** Qwen3.5 DeltaNet convolution/recurrent states cannot run; sparse attention-layer K/V lists must be emitted from actual graph ports rather than expanded from total layer count.
11. **Pipeline loader has no compatibility fallback.** The complete local Foundry Qwen package has `genai_config.json` but no native sidecar.
12. **No real VLM E2E baseline exists.** The committed test proves orchestration only; there is no image-quality or real-checkpoint token parity test.

# C. Dependency-ordered, file-disjoint work plan

The packages below intentionally do not share files, so separate agents can own them without merge contention. Dependencies are logical/API dependencies.

## WP0 — Typed multimodal metadata contract (P0; blocks WP1/WP2/WP3/WP4)

**Owner shape:** metadata/schema agent.

**Files:**

- `crates/onnx-genai-metadata/src/schema.rs`
- `crates/onnx-genai-metadata/tests/metadata_fixtures.rs`
- new `crates/onnx-genai-metadata/tests/fixtures/vlm_packed_valid.yaml`
- new `crates/onnx-genai-metadata/tests/fixtures/vlm_multistate_valid.yaml`
- `schema/inference_metadata.schema.json`

**Concrete contract:**

- Add typed `preprocessing.image` transforms and named tensor outputs. Required generic operations: decode/convert RGB, resize, rescale/normalize, tile, flatten/patchify, pad, emit original size, emit validity mask, emit patch/grid coordinates. Outputs bind to arbitrary pipeline endpoints; names such as `pixel_position_ids` are data, not runtime branches.
- Replace minimal `PipelineVisionConfig` with the already-proven rich expansion fields: placeholder token, emitted image token, per-tile/per-patch count source, per-image correspondence, optional separators, and thumbnail order.
- Add declared position-input generation/continuation semantics sufficient for rank-2 linear and rank-N multimodal coordinates. It must be parameterized by axes/sections and processor summaries, never by model family.
- Add generic loop-carried state pairs `{input, output, init, update}` for fixed recurrent tensors. Keep append/growing KV semantics separate.
- Permit graph I/O to declare raw token input and routed sequence inputs simultaneously.
- Add capability strings so unsupported processors/position/state programs fail at load with a precise missing-capability error.

**Acceptance:** both fixtures validate against generated JSON schema; one describes two packed image outputs and rich expansion, the other describes 3-axis positions plus sparse KV and fixed replacement-state pairs. No schema field or enum contains `gemma`, `qwen`, `phi`, a fixed layer count, hidden size, patch size, or magic tensor dimension.

## WP1 — Mobius native VLM package emission (P1; depends WP0; parallel with WP2/WP3/WP4)

**Owner shape:** Mobius exporter agent.

**Files in `/home/justinchu/mobius`:**

- `src/mobius/integrations/onnx_genai/inference_metadata.py`
- `src/mobius/integrations/onnx_genai/inference_metadata_test.py`
- `src/mobius/__main__.py`
- `tests/cli_test.py`

**Implementation:** inspect `ModelPackage.models` graph I/O and component roles; emit all models, topological dataflow, prompt/every-step phases, all embedding-to-decoder outputs, explicit component `io`, sparse actual KV pairs, fixed state pairs, typed processor program, expansion, and position contract. Reuse/refactor the concepts in ORT-GenAI emission, but dispatch processor operations through a registry/config description rather than `model_type` branches. Copy tokenizer, chat template, and processor config needed by the runtime.

**Acceptance:** no-weight or tiny builds for (1) cached Gemma4 E2B, (2) a Qwen VL config, and (3) Phi4MM produce native sidecars that pass onnx-genai schema validation and exactly match every ONNX input/output name. Gemma4 metadata routes both `inputs_embeds` and `per_layer_inputs`; Phi routes both gates; Qwen declares rank-3 positions and actual sparse/fixed state pairs when present. `mobius build ... --runtime onnx-genai` leaves a self-contained directory loadable as a pipeline without hand editing.

## WP2 — Generic multi-output image processor (P1; depends WP0; parallel with WP1/WP3/WP4)

**Owner shape:** preprocessing agent.

**Files:**

- `crates/onnx-genai-preprocess/src/image.rs`
- new `crates/onnx-genai-preprocess/src/image/packed.rs`
- `crates/onnx-genai-preprocess/src/lib.rs`

**Implementation:** change image processing from one `ImageTensor<Vec<f32>>` to a typed named tensor bundle plus per-image expansion summary. Execute the WP0 operation descriptors. Preserve the current rank-4 path, and add packed patches, padding/sentinel coordinates, grid THW, original-size tensors, and validity masks. Output dtype conversion must be declared (`fp32`, `fp16`, `bf16`, integer/bool auxiliary tensors), not inferred from model identity.

**Acceptance:** deterministic unit vectors cover:

- rank-4 NCHW existing behavior unchanged;
- Gemma4-shaped `[B,N,3*P^2]` pixels plus `[B,N,2]` coordinates with `(-1,-1)` padding;
- Qwen-shaped `[total_patches,pixel_dim]` plus `[num_images,3]` grid;
- Phi-shaped pixels plus original sizes and patch-validity mask;
- two images preserve per-image order and expansion summaries.

Reference outputs must match a checked-in small numerical fixture generated once from the corresponding HF processors; runtime tests must not download models.

## WP3 — Generic autoregressive step-component execution (P1; depends WP0; parallel with WP1/WP2/WP4)

**Owner shape:** pipeline engine agent.

**Files:**

- `crates/onnx-genai-engine/src/pipeline.rs`
- `crates/onnx-genai-engine/tests/gemma4_vlm_pipeline_e2e.rs`
- new `crates/onnx-genai-engine/tests/vlm_multibinding_pipeline_e2e.rs`
- `scripts/build_tiny_gemma4_vlm.py`
- new `scripts/build_tiny_vlm_multibinding.py`
- `tests/fixtures/tiny-gemma4-vlm/inference_metadata.yaml`
- new `tests/fixtures/tiny-vlm-multibinding/`

**Implementation:** replace the architecture-flavored `EmbedsStepBinding` with a generic topological executor for declared `every_step` components. On prefill, run them over the full expanded prompt; on decode, seed their declared token inputs with the running token and run them over one position. Route every produced output to decoder inputs for that same step. Fixed conditioning remains prompt-cached. Use explicit component `io`, not suffix checks. Preserve prompt-only and final-only semantics.

**Acceptance:** existing exact-token tiny Gemma4 test remains green after migrating metadata to generic phase semantics. New fixture's embedding emits both `inputs_embeds` and a second sequence-dependent tensor; generation fails if either is stale and passes with exact IDs when both are refreshed. A second assertion covers a decoder that also consumes raw `input_ids`, proving no token/embed exclusivity in pipeline execution.

## WP4 — Generic position programs and loop-carried decoder state (P2; depends WP0; parallel with WP1/WP2/WP3)

**Owner shape:** decode-state agent.

**Files:**

- `crates/onnx-genai-engine/src/decode.rs`
- new `crates/onnx-genai-engine/tests/decode_position_and_state_e2e.rs`
- new `scripts/build_tiny_multiaxis_state_decoder.py`
- new `tests/fixtures/tiny-multiaxis-state-decoder/`

**Implementation:** resolve position tensors from WP0 metadata rather than always generating `[1,S]`; support declared rank/axes and prefill-to-decode continuation. Generalize decoder state to initialize and carry arbitrary declared input/output pairs with `replace` semantics, while retaining KV append/share-buffer behavior separately. Remove the explicit-I/O rejection of simultaneous token and routed sequence inputs. Validate every declared graph port and shape-compatible initialization at load.

**Acceptance:** synthetic decoder asserts exact 3-axis prefill and next-token positions, zero initialization of two fixed state tensors, state replacement over at least two steps, and sparse KV indices. All existing text, sliding-window, and shared-buffer decode tests remain green.

## WP5 — Server/driver multimodal tensor bundle and admission ordering (P2; depends WP2 + WP3; WP4 for Qwen)

**Owner shape:** serving agent.

**Files:**

- `crates/onnx-genai-server/src/image_input.rs`
- `crates/onnx-genai-server/src/driver.rs`
- `crates/onnx-genai-server/src/state.rs`
- `crates/onnx-genai-server/src/routes.rs`
- new `crates/onnx-genai-server/tests/vlm_image_bundle.rs`

**Implementation:** discover processor bindings from typed metadata, not literal session input names. Carry a vector/map of typed tensors and full per-image expansion summary through `PipelineInputTensor`. Inject all endpoints into `PipelineGenerateRequest`. Tokenize, preprocess, expand, then enforce context/admission limits using final prefill length. Support multiple image content parts in prompt order. Fail with endpoint, expected dtype/shape, and missing metadata operation when a contract is incomplete.

**Acceptance:** an OpenAI chat request with a data-URI image against a packed two-input tiny fixture reaches both vision inputs, expands exactly one matching placeholder, and generates deterministic output. A two-image test proves placeholder/image ordering. Negative tests cover missing placeholder, wrong image count, and expanded-context overflow.

## WP6 — Pipeline compatibility loading for existing ORT-GenAI packages (P3; depends WP0 + WP3 + WP4)

**Owner shape:** compatibility/loader agent.

**Files:**

- `crates/onnx-genai-genai-config/src/lib.rs`
- `crates/onnx-genai-genai-config/tests/` (new VLM fixtures/tests)
- `crates/onnx-genai-ort/src/loader.rs`
- new `crates/onnx-genai-ort/tests/pipeline_genai_fallback.rs`

**Implementation:** when native metadata is absent, convert `genai_config.json`, `config.json`, and processor config into an in-memory typed pipeline only when those files explicitly provide every required semantic. Include embedding as an `every_step` stage, all declared dataflow, rank/dtype preprocessing, positions, actual KV pairs, and fixed state pairs. Do not guess from `model.type`; if an old package lacks explicit state/position facts, fail and tell the user to regenerate native metadata.

**Acceptance:** the local Foundry Qwen3.5 directory is recognized as a pipeline without a hand-written sidecar; loader reports its two vision inputs, rank-3 positions, eight K/V layer pairs, and 48 fixed state pairs. A deliberately incomplete compatibility fixture fails with a what/why/how-to-fix error rather than architecture inference.

## WP7 — Real-model validation ladder (P3; depends WP1-WP5; Qwen3.5 also depends WP6)

**Owner shape:** validation agent.

**Files:**

- new `scripts/validate_vlm_pipeline.py`
- new `crates/onnx-genai-engine/tests/real_vlm_env.rs`
- new `docs/benchmarks/<date>-real-vlm.md`

**Milestone order and acceptance:**

1. **Gemma4 E2B, first real milestone.** Export cached source through Mobius with native metadata. Compare vision outputs, embedding outputs (including `per_layer_inputs`), prefill logits, and one decode step against HF/Mobius ORT reference with fixed image/prompt. Then server-smoke `Describe this image.` with the correct image token sequence. Environment-gated due model size.
2. **Qwen2.5/3-VL, second architecture milestone.** Validate packed patches + `image_grid_thw` + rank-3 MRoPE on a small checkpoint; exact prefill logits and one generated token.
3. **Foundry Qwen3.5, stress milestone.** Validate the complete local package through compatibility loading, including recurrent-state carry, then one-token image generation. This is not a CI test.
4. **Phi4MM, follow-on.** Validate multi-output image preprocessing and embedding gates; audio/mixed modality stays a separate scope.

# D. Available real models and fixtures

## Immediately usable

- **Committed deterministic contract fixture:** `/home/justinchu/onnx-genai/tests/fixtures/tiny-gemma4-vlm/`. Small, exact, CI-safe.
- **Gemma4 E2B source checkpoint and processor:** `/home/justinchu/.cache/huggingface/hub/models--google--gemma-4-E2B-it/`. A snapshot contains the ~10.2 GB weights, tokenizer, chat template, and `processor_config.json`. Config declares image token `258880`, `hidden_size_per_layer_input=256`, 35 text layers, patch size 16, and processor `max_soft_tokens/image_seq_length=280`.
- **Complete real Qwen3.5 VLM ONNX package:** `/home/justinchu/.foundry/cache/models/Microsoft/qwen3.5-9b-generic-cpu-2/v2/`. Contains `vision.onnx`, `embedding.onnx`, `text.onnx`, external data, tokenizer, `processor_config.json`, `config.json`, and `genai_config.json`. Its interfaces are:
  - vision: `pixel_values [num_patches,1536]`, `image_grid_thw [1,3]` -> `image_features [num_logical_patches,4096]`;
  - embedding: `input_ids`, `image_features` -> `inputs_embeds`;
  - decoder: `inputs_embeds`, mask, `position_ids [3,B,S]`, eight sparse K/V pairs, and 48 fixed DeltaNet state pairs.
- **Image fixture:** `/home/justinchu/Olive-recipes/Qwen-Qwen2.5-VL-3B-Instruct/cat.jpeg` (also present in Qwen3 recipe directories).

## Useful but not a VLM package

- `/home/justinchu/gemma4-e2b-onnx`, `gemma4-e2b-onnx-target`, and `gemma4-e2b-assistant-onnx` are real text/speculative decoder artifacts, not a vision+embedding package. The benchmark `docs/benchmarks/2026-07-15-real-native-gemma4e2b.md:1-39` proves real native text decode only.
- `/home/justinchu/ana-bench/qwen-oga-cuda-graph-a4` contains one text Qwen `model.onnx`/`genai_config.json`, not a VLM.
- The cached `Qwen/Qwen3-VL-2B-Instruct` directory currently contains config only, not model weights; it is not an offline-ready export source.

## Not found locally

No complete Phi-vision/Phi4MM ONNX package was found under the inspected home/model fixture paths. No real multimodal Gemma4 ONNX trio (`vision_encoder` + `embedding` + decoder) was found; it must be exported from the cached source checkpoint.

# E. RULES.md section 2 / 2.1 risks and guardrails

`RULES.md:20-29` requires all architectural assumptions to be explicit metadata, with no model/vendor/EP dispatch. `RULES.md:30-37` requires structural, EP-internal graph fusion.

Review-blocking risks:

1. **Do not add `Gemma4`, `Qwen`, `Phi`, or model-name modes** to metadata, preprocessing, server discovery, pipeline execution, or decode.
2. **Do not bake fixed shapes or counts** such as 280 patches, three MRoPE axes, 35 layers, 1536 patch width, 4096 hidden size, every-fourth attention, or Phi's 448 image size into runtime code. These are fixture/model metadata values.
3. **Do not discover semantic ports from names** such as `pixel_values`, `image_grid_thw`, `per_layer_inputs`, `vision_gate`, `conv_state`, or suffixes. Emit exact graph ports and their roles in metadata; missing roles fail clearly.
4. **Do not implement vision projector/fusion math in Rust.** Vision tower, projector, feature scatter, per-layer embeddings, and modality gates remain ONNX components connected by dataflow. Rust only executes declared transforms and component topology.
5. **Do not make MRoPE a Qwen branch.** Position generation must be a parameterized position program/declared component with rank, axes, sections, and continuation semantics as data.
6. **Do not treat every non-logits output as KV.** State behavior must be declared (`append/share` KV versus fixed `replace` recurrence), and sparse layers must come from actual emitted port lists.
7. **Do not add model-specific fusion in generic graph code.** Any optimization of patchification/projector or embedding fusion must be a structural pattern inside an EP. The first implementation should preserve separate ONNX components.
8. **Do not silently guess compatibility metadata.** For existing `genai_config.json` packages, reject missing processor/position/state semantics with an actionable regeneration instruction.

Top 3 next VLM work packages in priority order:
1. WP0 — Define the typed, architecture-neutral multimodal preprocessing, expansion, position, and loop-state contract.
2. WP3 — Replace the one-output embedding special case with generic autoregressive every-step component execution.
3. WP2 — Produce typed multi-tensor image bundles for packed patches, coordinates/grids, sizes, and masks.

<!-- source: .squad/decisions/inbox/mariette-qwen3-export.md -->
### 2026-07-22: Export Qwen3-0.6B as an int4 CUDA onnx-genai package
**By:** Mariette
**What:** Exported Qwen/Qwen3-0.6B with Mobius for CUDA/onnx-genai, then applied ORT weight-only int4 RTN quantization because this Mobius branch's `build` CLI has no quantization flags.
**Why:** Provide a Qwen3 package with QK-norm and block-32 MatMulNBits projections for a future H200 roofline benchmark against the existing Qwen2.5 packages.

**Commands:**
```bash
cd /home/justinchu/mobius
.venv/bin/python -m mobius build \
  --model Qwen/Qwen3-0.6B \
  .scratch/qwen3-0.6b-int4-cuda \
  --runtime onnx-genai \
  --execution-provider cuda \
  --dtype f16 \
  --optimize

.venv/bin/python -m onnxruntime.quantization.matmul_nbits_quantizer \
  --input_model .scratch/qwen3-0.6b-int4-cuda/model.onnx \
  --output_model .scratch/qwen3-0.6b-int4-cuda/model.int4.onnx \
  --block_size 32 \
  --bits 4 \
  --quant_method default \
  --symmetric True \
  --accuracy_level 4 \
  --quant_format QOperator
```

**Artifacts:**
- Package: `/home/justinchu/mobius/.scratch/qwen3-0.6b-int4-cuda`
- Full log: `/home/justinchu/mobius/.scratch/qwen3-0.6b-int4-cuda-export.log`
- Final graph/data: `model.onnx` (391,327 bytes) + `model.onnx.data` (569,493,504 bytes)
- Runtime metadata: `inference_metadata.yaml`
- Tokenizer: `tokenizer.json`

**Metadata and structure:**
- Architecture: `qwen3`
- Layers: 28 (graph-derived from 28 GroupQueryAttention nodes)
- Attention heads / KV heads / head dim: 16 / 8 / 128
- Maximum sequence length: 40,960
- Quantization: 196 symmetric `com.microsoft::MatMulNBits` nodes, all `bits=4`, `block_size=32`, `accuracy_level=4`, and no explicit zero-point input; this is exactly 7 decoder projections × 28 layers.
- RoPE: fused in GroupQueryAttention with `do_rotary=1`, `rotary_interleaved=0`; HF config has `rope_theta=1,000,000` and `rope_scaling=null`.
- Qwen3 QK-norm is explicit: 56 `RMSNormalization` nodes, one Q norm and one K norm in each of 28 layers.
- One float `MatMul` remains for the tied embedding/LM-head projection because the ORT quantizer reported that the transposed tied weight was not a direct constant.
- `onnx.checker.check_model` passed; no GPU generation smoke was run.

**Coverage/tooling gaps:**
- Mobius `build` on `dave-wp1-vlm-emission` accepts CUDA/runtime/dtype/optimization flags but exposes no int4/group-size flag. Producing the requested package therefore required a second ORT quantization command after Mobius export.
- `inference_metadata.yaml` declares architecture, attention/KV dimensions, and max sequence length, but not layer count, quantization format, or RoPE theta. Those values above are graph/config-derived.

**Next step:** Run the H200 structural-load and short generation smoke first, then roofline-benchmark decode. Confirm that native CUDA assigns all 196 block-32 MatMulNBits nodes and the QK-norm/GQA path without fallback.

<!-- source: .squad/decisions/inbox/rachael-qwen-ladder-review.md -->
### 2026-07-22: Review — Qwen 1.5B/7B H200 bench ladder
**By:** Rachael
**What:** APPROVE+merged as `c9190c6c3f9f559913814db9ed3d49eafd890692`.
**Roofline check:** At 3.35e12 B/s, 886 tok/s implies 3,781,038,375 B/token. That is not the stated weight-plus-KV model: Qwen2.5-0.5B has 282,190,592 streamed weight bytes and 12,288 KV bytes/cached token, yielding 283,081,472 B/token and 11,834 tok/s at the 128-token window (288,586,496 B/token and 11,608 tok/s at 1024). Thus the prior 810.06/778.59 results are 6.85%/6.71% of this explicit HBM bound; their 91.4%/87.9% figures use the separate practical 886 tok/s gap-free ceiling. Applying the same explicit formula gives 1.5B: 879,014,912 B/token → 3,811 tok/s and 487.66/3,811 = 12.8%; 891,859,968 B/token → 3,756 tok/s and 457.88/3,756 = 12.2%. For 7B: 3,990,248,448 B/token → 840 tok/s and 230.47/840 = 27.5%; 4,015,938,560 B/token → 834 tok/s and 223.38/834 = 26.8%. Joi's percentages are internally consistent.
**Why:** The apparent impossibility comes from treating the practical 886 tok/s ceiling as the physical weight-plus-KV HBM roofline. The document explicitly distinguishes those denominators, its throughput/latency reciprocals and roofline arithmetic check out, and commit scope is only the added benchmark document with no Rust `src/` changes or model-name logic.

<!-- source: .squad/decisions/inbox/rachael-qwen3-bench-review.md -->
### 2026-07-23: Qwen3-0.6B bench doc review
**By:** Rachael
**What:** 🟢 Approved and fast-forward merged as `e04fc2df95ad0740bdc71917a55af7d6b8b52356`.
**Why:** The branch adds only `docs/benchmarks/qwen3-0.6b-h200-2026-07-22.md`; no Rust or runtime logic changed. The 12,585/10,549 tok/s rooflines, achieved percentages, benchmark values, and comparison percentages are internally consistent. The metadata gap is described generically as the unrecognized `grouped_query` alias, absent fp16 KV dtype, and missing compatibility config. No secrets or credentials were found; the absolute paths are benchmark provenance and include only the permitted scratch export and local setup paths.

<!-- source: .squad/decisions/inbox/sebastian-native-interface.md -->
### 2026-07-22: GAP 1 done — backend-neutral pipeline component-session interface
**By:** Sebastian
**Branch:** `squad/sebastian-native-interface`  **Base:** `origin/main` @ `1ae99b4`  **Commit:** `b6dcb32`

**What:** Implemented GAP 1 of Joi's Gemma4 native-backend analysis: a
backend-neutral abstraction for pipeline component sessions. `PipelineEngine`
now constructs every declared component through EITHER ORT `Session` OR the
native `InferenceSession`, driven by `EngineDecodeBackend`, with no ORT-only
type on the engine's construction path. The old hard `Native` rejection at
construction is gone; the native path now fails later at a precise,
actionable next blocker (gaps 2/3) instead of a blanket refusal.

#### Trait design + crate placement
- New module `onnx-genai-metadata/src/component.rs` defines:
  - `ComponentSession` — object-safe trait exposing exactly what neutral
    construction/wiring needs: `inputs()`/`outputs()` (graph I/O metadata:
    name, dtype, shape → rank), default `input_names()`/`output_names()`, and
    `run(&mut self, &[(&str, &ComponentTensor)]) -> Result<Vec<(String, ComponentTensor)>, ComponentError>`
    (named-tensor execution, outputs in `output_names()` order).
  - `ComponentDataType` — 14-variant neutral dtype vocabulary
    (`size_of`/`as_str`/`Display`).
  - `ComponentIo { name, dtype, shape: Vec<i64> }` with `rank()`; negative
    axes = dynamic (ORT convention).
  - `ComponentTensor` — owned host tensor carrying **raw little-endian element
    bytes** in row-major order; `from_raw` enforces static shape and
    `len == numel * dtype.size_of()`. Bytes-as-payload means any dtype
    round-trips without a per-dtype host container and without either
    backend's tensor type entering the interface.
  - `ComponentError` — thiserror enum: `ByteLengthMismatch`, `DynamicShape`,
    `UnsupportedDataType`, `Backend { component, backend, detail }` (all
    RULES §1 actionable).
- **Placement rationale:** the trait lives in `onnx-genai-metadata`, the
  lowest crate **both** backend crates already depend on. `onnx-genai-engine`
  depends on `onnx-genai-ort` (not vice-versa), so putting the trait in the
  ort crate would bar the native backend from implementing it; putting it in
  the engine would force ort to depend "up". Metadata is the only cycle-free
  home both sides can implement against. `&mut self` on `run` accommodates
  native `InferenceSession::run(&mut self)`; ORT's `Session::run(&self)` is a
  no-op under `&mut`.

#### ORT implementation (behavior-preserving)
- New `onnx-genai-ort/src/component.rs`: `OrtComponentSession` wraps an
  existing `Session`. `run` builds ORT `Value`s from the neutral byte tensors,
  forwards to `Session::run` unchanged, and reads outputs back to bytes — a
  pure adapter, so a pipeline routed through the seam produces **byte-identical**
  results to one calling `Session::run` directly (proven by a round-trip test).
- `From<DataType> for ComponentDataType` and reverse (total maps over ORT's
  14 dtypes).
- `onnx-genai-ort/src/value.rs`: added a `TensorBacking::Bytes(Vec<u8>)`
  variant plus `Value::from_raw_bytes(bytes, shape, dtype)` /
  `Value::to_raw_bytes()` so host tensors of any dtype cross the seam as
  opaque LE bytes. No decode/session logic changed.

#### Native implementation (feature-gated)
- New `onnx-genai-engine/src/native_component.rs` (`#[cfg(feature = "native-backend")]`):
  `NativeComponentSession` wraps `InferenceSession`; maps `onnx_runtime_ir`
  dtypes ↔ neutral dtypes (unmapped ir types → `UnsupportedDataType`),
  `Dim::Symbolic → -1`. `load(path, NativeDecodeDevice)` builds a session on
  the requested device.

#### Construction is now backend-neutral
- `pipeline.rs::from_dir_with_schedulers` selects **one** backend for the
  whole pipeline (never a mix):
  - Explicit `Ort`/`Native` resolve **without** touching the model directory
    (bad requests still fail fast — preserves the old non-existent-path test).
  - `Auto` inspects declared operators via `model_requires_native_backend`
    and routes to Native only when some component requires it (previously it
    hard-rejected such pipelines).
- When the backend is Native:
  - built **without** `native-backend` → actionable "compiled without the
    'native-backend' feature" error.
  - built **with** `native-backend` → loads **all** components through
    `NativeComponentSession` via the trait, confirming construction is
    genuinely backend-neutral, then returns the gap-2/3 error (below).
- The **ORT path is unchanged**: same `PipelineModels::load_with_options`
  construction and the merged `every_step` executor (`pipeline.rs`
  ~L1952-2027) are untouched.

#### Where the native path now errors (the precise next blocker)
With `native-backend` on, all components load and expose graph I/O through the
seam, then construction returns:
> "native pipeline decode is not yet implemented. All N component(s) loaded on
> the native backend and expose their graph I/O through the backend-neutral
> component-session interface … The remaining work is wiring these native
> sessions into the pipeline decode loop, which still routes per-step decode
> state through ORT tensors/sessions."

That maps exactly to Joi's gaps 2 → 3, in order:
- **GAP 2 (next):** generalize native target decode beyond token-ID-only
  sessions. `NativeDecodeSession` has a fixed field set and rejects
  `sequence_source: inputs_embeds` (`native_decode.rs:190-205,716-729`);
  Gemma4-class models declare `inputs_embeds` + routed `per_layer_inputs`.
  Needs metadata-declared arbitrary named step inputs (no architecture-specific
  ports).
- **GAP 3 (after):** convert pipeline decode state/execution off ORT `Value`/
  `Session` — `DecodeState` around `&Session`/`Value` (`decode.rs:693-735`),
  `PipelineDecodeLoopBackend` (`pipeline.rs:~2181-2244`). Introduce
  neutral tensor/state ops or a parallel native decode-loop backend.

The merged `every_step` executor is **not** a blocker (it already binds token
inputs from metadata, resolves routed inputs, and republishes outputs each
step) — confirmed preserved this cycle.

#### Test coverage (CPU/ORT only, no GPU)
- metadata: 5 unit tests — tensor byte-length/dynamic-shape invariants, rank,
  dtype sizes.
- ort: 2 unit tests on the tiny-whisper textproto fixture — graph I/O
  exposure, and a run round-trip asserting the seam's output bytes equal
  direct `Session::run` (behavior-preservation proof).
- engine (native): 2 unit tests on a tiny `Add` graph — I/O metadata + named
  run round-trip; plus the rewired-selection test
  `auto_backend_routes_native_only_pipeline_to_the_native_backend`.
- engine (default): rewrote `explicit_native_backend_without_feature_reports_actionable_build_error`
  to assert the new actionable message and the **absence** of the removed
  blanket refusal.

#### Validation
- `cargo build -p onnx-genai-metadata -p onnx-genai-ort -p onnx-genai-engine` → OK.
- `cargo clippy -p onnx-genai-metadata -p onnx-genai-ort -p onnx-genai-engine --all-targets -- -D warnings` (default features) → **clean**.
- `cargo test` for all three crates (default features) → **pass**.
- `cargo test -p onnx-genai-engine --features native-backend`: my new tests
  pass. 16 pre-existing failures remain (single-model `engine::`/`embedding::`
  fixtures that decode textproto as binary protobuf and/or need CUDA) — present
  on `origin/main` too (base showed 17; delta is the extra tests I added). Not
  introduced by this change.
- **Known pre-existing warning:** `cargo clippy -p onnx-genai-engine --features native-backend`
  fails **only** on a pre-existing unused-import (`BTreeMap`,
  `native_decode.rs:17`, used solely under `#[cfg(test)]`). It exists on
  `origin/main`; `native_decode.rs` is out of GAP-1 scope (and belongs to
  gaps 2/3) so I left it untouched. Default-feature clippy is clean.

**Why:** GAP 1 was the top blocker — native pipeline construction failed before
model loading. Construction + the component-session seam are now backend-neutral
with the ORT path byte-preserved, turning "native not supported for pipelines"
into a scoped, ordered pair of next blockers (gaps 2/3). No hardcoded model
architecture (RULES §2/§2.1): all dtype/shape handling is generic;
`grep -niE "gemma|qwen|phi|llama|mistral"` over touched logic is empty.

<!-- source: .squad/decisions/inbox/tessa-progress-qwen-ladder.md -->
# Tessa progress note — Qwen2.5 H200 decode ladder

Commit: `2053a11ae882fed79f75e2dee890dc989bb5e461`

Summary: Added the 2026-07-22 PROGRESS changelog entry for the merged Qwen2.5-1.5B/7B H200 decode roofline ladder report.

<!-- source: .squad/decisions/inbox/zhora-deepseek-mla-conformance.md -->
### 2026-07-22: Re-verify native DeepSeek V2 and add CPU MLA decode conformance
**By:** Zhora
**What:** Native DeepSeek-V2 tiny E2E on CPU generated eight tokens `[42, 237, 198, 2, 186, 81, 210, 149]`, exactly matching the established sequence. No new op or dtype gap appeared after the generic Unsqueeze shape-chain fix. Added a deterministic fp32 CPU standard-Attention test for 3-D BSH decode with `qk_head_dim=2`, `v_head_dim=1`, four query heads, two KV heads, and a non-empty two-token past cache. The test checks hand-computed GQA outputs and verifies Y/present-value use V width while present-key uses Q/K width. fp16 was not added because the CPU Attention kernel currently supports f32 only.
**Why:** This keeps the native DeepSeek smoke green on current `origin/main` and adds a compact hand-computed guard for the asymmetric MLA cached-decode contract without changing kernel logic. The next native decode-parity step requires deterministic Mobius DeepSeek fixtures after DS-0/DS-2; CUDA MLA conformance requires an available GPU and was intentionally not run in this CPU-only round.

<!-- source: .squad/decisions/inbox/zhora-deepseek-native-unsqueeze.md -->
### 2026-07-22: Lock DeepSeek Unsqueeze dynamic-shape coverage
**By:** Zhora
**What:** The reported original failure was `no inferred shape for value v_model.Unsqueeze_9 produced by op Unsqueeze`. Current `origin/main` already contains the generic runtime implementation in `crates/onnx-runtime-session/src/executor.rs:1156-1292`: `dynamic_output_shapes` re-runs the opset-aware ONNX shape-inference registry with concrete runtime input shapes and bounded integer shape data, covering opset-13+ axes inputs, older `axes` attributes, and negative-axis normalization against output rank. This branch adds the focused unit regression `dynamic_output_shapes_unsqueeze_supports_input_and_attribute_axes` at `executor.rs:7123`, directly checking both axes forms and negative axes.
**Why:** The exact requested native command now stops earlier with the actionable feature-gate error (`native decoder backend requires building onnx-genai-engine with the 'native-backend' feature`), so the historical Unsqueeze failure was not reproducible from current main. With `--features native-backend`, the pre-change baseline and post-change run both passed end-to-end and generated 8 tokens: `[42, 237, 198, 2, 186, 81, 210, 149]`; no downstream native gap was observed. The default ORT-backend run also passed with the same 8 tokens. Validation passed: `cargo fmt -p onnx-runtime-session`, `cargo clippy -p onnx-runtime-session --all-targets -- -D warnings`, and `cargo test -p onnx-runtime-session` (62 unit tests plus integration/doc tests). Branch: `squad/zhora-deepseek-native-unsqueeze`; commit: `8d9d2fa279a0463b9fc6c02932ebd7b9ec775fb4`.

<!-- source: .squad/decisions/inbox/zhora-deepseek-scope.md -->
# DeepSeek work scope: unblocked path after V4 upstream block

**By:** Zhora (Server Dev)  
**Requested by:** Justin Chu  
**Date:** 2026-07-21T15:12Z  
**Scope:** Read-only investigation; no source files changed.

## Executive answer

DeepSeek support is real but split across three maturity levels:

1. **DeepSeek-V2/V3 export exists.** Mobius registers `deepseek_v2`, `deepseek_v2_moe`, and `deepseek_v3` to `DeepSeekV3CausalLMModel` (`/home/justinchu/mobius/src/mobius/_registry.py:541-545`). `DeepSeekMLA::forward` lowers MLA to ordinary projections, partial RoPE, and standard ONNX `Attention` (`/home/justinchu/mobius/src/mobius/components/_deepseek_mla.py:107-218`). `DeepSeekMoEGate` implements V2 softmax/group-limited routing and V3 sigmoid/noaux routing (`/home/justinchu/mobius/src/mobius/models/deepseek.py:53-122`), and `_DeepSeekMoEFFN` adds routed plus shared experts (`deepseek.py:308-346`).
2. **DeepSeek-V2 runs end-to-end today through the default ORT backend.** The checked-in gated test loads a Mobius package with `Engine::from_dir` and generates eight tokens (`crates/onnx-genai-engine/tests/deepseek_e2e.rs:1-47`). I ran it against `/home/justinchu/ds-e2e-artifacts/deepseek-v2-tiny`; it passed and generated `[42, 237, 198, 2, 186, 81, 210, 149]`. This is a random-weight structural smoke, not production-weight semantic proof.
3. **Native Rust execution is not end-to-end yet.** With `ONNX_GENAI_BACKEND=native`, the same model fails before attention/MoE execution: `no inferred shape for value v_model.Unsqueeze_9 produced by op Unsqueeze`. The failing graph chain is `Shape/Sub -> Slice -> Unsqueeze` in the attention-mask construction. `dynamic_output_shapes` currently handles dynamic `Slice` but not the following `Unsqueeze` (`crates/onnx-runtime-session/src/executor.rs:988-1024,2728-2788`). This is the first concrete, unblocked runtime task.

DeepSeek-V4 production onboarding remains blocked by the missing official Transformers-compatible reference/config and unresolved official iterative-MTP contract. A V4-Flash preview exporter and substantial CSA runtime kernels exist, but they do not form a verified model E2E path today.

## 1. State today — export

### 1.1 DeepSeek-V2/V3 MLA + MoE

Mobius supports:

- `deepseek_v2` and alias `deepseek_v2_moe` using `deepseek-ai/DeepSeek-V2-Lite` as the catalog model;
- `deepseek_v3` using `deepseek-ai/DeepSeek-V3` (`/home/justinchu/mobius/src/mobius/_registry.py:908-912`);
- variant labels `mla`, `mla+moe`, and `mla+moe` respectively (`_registry.py:1221-1226`).

The model implementation is config-driven:

- `DeepSeekV3TextModel` chooses MLA when `qk_nope_head_dim > 0`, otherwise standard attention, and selects dense vs. MoE layers from `first_k_dense_replace` (`/home/justinchu/mobius/src/mobius/models/deepseek.py:373-425`).
- `DeepSeekMLA` builds low-rank Q and KV projection paths, applies RoPE only to the rotary subspace, supports `qk_head_dim != v_head_dim`, then emits standard `Attention` (`/home/justinchu/mobius/src/mobius/components/_deepseek_mla.py:29-102,107-218`).
- V2 routing is softmax plus greedy/group-limited top-k; V3 routing is sigmoid plus correction bias/noaux group selection (`deepseek.py:53-122,164-202`).
- Shared experts are always added to routed-expert output (`deepseek.py:308-346`).

Existing Mobius verification is stronger for prefill than decode:

- reduced DeepSeek-V2-Lite HF-vs-ONNX prefill logits are compared in `tests/integration_test.py:2132-2153`;
- synthetic DeepSeek-V3 parity has an explicit `0.04` tolerance (`tests/synthetic_parity_test.py:158-164`);
- V3 has no small public checkpoint and is skipped as a 671B model in model coverage (`tests/model_coverage_test.py:203-207`).

**Important MLA limitation:** the export decompresses latent KV into per-head `k_nope` and V before `Attention`, and `Attention` returns ordinary full present K/V (`_deepseek_mla.py:136-218`). It executes MLA math, but does **not** preserve a low-rank latent KV cache. Thus correctness is available; MLA's intended decode-memory/bandwidth benefit is not.

### 1.2 Quantized MoE export

The current Mobius worktree `/home/justinchu/mobius` is on `glm5.2-moe-export` at `cd782dd`. Relevant commits are:

- `93cbcf7` — shared GLM/DeepSeek fused `com.microsoft::QMoE` emitter;
- `cd782dd` — restored DeepSeek grouped top-k selection;
- `2b629cc` — tiny DeepSeek-V2 ONNX export helper.

`_DeepSeekMoEFFN` chooses `FusedQuantizedMoE` when quantization is enabled and `fused_quantized_moe=true` (`/home/justinchu/mobius/src/mobius/models/deepseek.py:315-346`). The emitter packs expert-major weights and supplies separate selection and aggregation tensors to generic `com.microsoft::QMoE` (`/home/justinchu/mobius/src/mobius/components/_moe.py:239-381`). The config explicitly says this shared path is wired for GLM/DeepSeek (`src/mobius/_configs/_base.py:410-413`).

However, only GLM currently has a focused exporter assertion that one QMoE node is emitted per MoE layer (`src/mobius/models/glm_moe_dsa_test.py:174-200`) and an engine QMoE smoke (`crates/onnx-genai-engine/tests/glm_tiny_qmoe_e2e.rs:1-103`). DeepSeek has no equivalent fused-QMoE artifact/test yet.

### 1.3 DeepSeek-V4-Flash preview export

The `dsv4-flash-export` worktree exists at `/home/justinchu/mobius-wt-dsv4`, commit `7e26e6e`; Mobius PR #405 is open draft. It exports:

- V4 projections, Hyper-Connections, sqrt-softplus/hash MoE, clipped SwiGLU, grouped output LoRA;
- learned per-head sinks and dense causal attention;
- retained compressor/indexer tensors;
- target `hidden_states` and a separate MTP sidecar (`/home/justinchu/mobius-wt-dsv4/DSV4_FLASH_EXPORT.md:26-46`; `src/mobius/models/deepseek_v4.py:1-14,289-480,594-764`).

It does **not** execute CSA/HCA. Compressor/indexer tensors are kept reachable through zero-valued shape anchors (`deepseek_v4.py:54-65,455-459`). The design explicitly says dense attention is not numerically equivalent to learned sparse compression at long context (`DSV4_FLASH_EXPORT.md:35-41`). The sidecar emits `mtp_hidden` but no recurrent `mtp_state` (`src/mobius/models/deepseek_v4_flash_test.py:163-183`).

## 2. State today — runtime

### 2.1 Default ORT engine path

`EngineConfig::default()` selects `EngineDecodeBackend::Auto` (`crates/onnx-genai-engine/src/config.rs:453-475,509-525`). Auto uses ORT unless a narrowly detected native-only op is present (`crates/onnx-genai-engine/src/engine.rs:2228-2267`). The detector recognizes only `pkg.nxrt::BlockQuantizedMatMul` (`engine.rs:2329-2367`), so primitive-op DeepSeek V2/V3 and `com.microsoft::QMoE` packages stay on ORT.

The existing DeepSeek-V2 artifact therefore proves the public engine flow and ORT execution, not the native Rust EP. `docs/PROGRESS.md:114-119` accurately calls it a tiny fp32 structural E2E and notes real-weight/native QMoE follow-up.

### 2.2 Native MLA-relevant kernels

The native CPU standard `Attention` kernel accepts explicit `q_num_heads`/`kv_num_heads`, keeps Q/K head width separate from V head width, and sizes output from `v_head_size` (`crates/onnx-runtime-ep-cpu/src/kernels/attention.rs:53-112,324-445`). CUDA standard Attention has the same separate `head_size` and `v_head_size` contract (`crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs:654-767`). This is the right primitive contract for the current decomposed MLA export.

What is missing is a full native graph pass. The first observed blocker is dynamic shape propagation after `Slice`; the executor fallback is model-agnostic but incomplete (`crates/onnx-runtime-session/src/executor.rs:988-1024,2728-2788`). After that is fixed, the same E2E must expose any later unsupported op or dtype gap.

### 2.3 Native MoE kernels

Native generic kernels already exist:

- CPU `com.microsoft::QMoE`: integer 1/2/4/8-bit expert weights, generic expert-major layout, optional mmap route-first selected-expert loading (`crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs:1-16,43-99,101-247`).
- CUDA `com.microsoft::QMoE`: device routing, decode GEMV, grouped prefill GEMM; paging and expert-parallel sharding are deferred (`crates/onnx-runtime-ep-cuda/src/kernels/qmoe.rs:1-7,690-788`).
- CPU/CUDA registries both register `com.microsoft::QMoE` (`crates/onnx-runtime-ep-cpu/src/kernels/mod.rs:372-373`; `crates/onnx-runtime-ep-cuda/src/kernels/mod.rs:335-336`).

Gap: no DeepSeek engine E2E forces these native kernels. The existing GLM QMoE engine test explicitly exercises ORT contrib QMoE, not native Rust (`crates/onnx-genai-engine/tests/glm_tiny_qmoe_e2e.rs:6-12`; `docs/PROGRESS.md:108-119`).

### 2.4 CSA and MTP

CSA runtime work is advanced but disconnected from a model package:

- CPU `pkg.nxrt::CompressedSparseAttention` implements stateful ratio-4/128 compression, index stream, top-k, cache/carry, and learned sink semantics (`crates/onnx-runtime-ep-cpu/src/kernels/compressed_sparse_attention.rs:1-8,151-270`).
- CUDA ratio-4 FP8 six-output execution is device-resident and capture-clean; other configurations retain partial/host-staged paths (`crates/onnx-runtime-ep-cuda/src/kernels/compressed_sparse_attention.rs:942-1070,1960-1971,2171-2190`). The B7 decision records Phase B complete for that specific ratio-4 configuration and notes MTP composite atomicity remains gated on an external artifact (`.squad/decisions/archive/2026-07-20T13-35-00Z-decisions-pre-multistream.md:1871-1884`).
- Mobius PR #405 does not emit this custom op, so these kernels are not model-E2E exercised.

Generic MTP Phase-1 plumbing also exists:

- metadata resolves an MTP sidecar with BSHC layout, exact initializer references, `mtp_hidden`, `mtp_state`, and KV mode (`crates/onnx-genai-metadata/src/parser.rs:51-76,100-187`);
- `MtpDecodeSession::step_with_state` binds rank-4 HC state and requires a recurrent state output when `hc_mult > 1` (`crates/onnx-genai-ort/src/mtp.rs:313-457`);
- `MtpProposer` is constructed once per generation and reused inside the speculative loop (`crates/onnx-genai-engine/src/speculative.rs:1164-1187`).

Gap: the V4-Flash sidecar lacks `mtp_state`, while the official reference does not publish an iterative recurrence/acceptance loop. `docs/DEEPSEEK_CSA_MTP_RUNTIME.md:1485-1515,1585-1625` explicitly marks recurrence, verification, cache lifetime, tie behavior, and exact backend arithmetic as unfrozen.

## 3. BLOCKED vs ACTIONABLE

### BLOCKED — do not schedule as implementation work

1. **Production DeepSeek-V4 onboarding/parity.** Mobius PR #213 is an open draft investigation documenting that no Transformers `configuration_deepseek_v4.py` / `modeling_deepseek_v4.py` exists and the standard AutoConfig/parity workflow cannot run. This blocks a normal verified V4 model onboarding.
2. **Official iterative V4 MTP.** The reference exposes one MTP block but no recurrent state/acceptance algorithm; the current sidecar lacks `mtp_state`. Do not invent recurrence from `mtp_hidden`.
3. **V4 portable tie/quant arithmetic claims.** Top-k tie stability, Hadamard implementation version, and several accumulator details remain unpinned (`docs/DEEPSEEK_CSA_MTP_RUNTIME.md:1597-1619`).
4. **Real V4 E2E in this checkout.** PR #405's manifest entry exists (`tests/e2e/mobius_heads.json:17-27`), but no `deepseek-v4-flash` artifact is present locally.

### ACTIONABLE NOW

1. Make the existing DeepSeek-V2 primitive graph run through the native Rust backend by completing generic dynamic shape propagation.
2. Add deterministic DeepSeek-V2 and tiny-config V3 **decode** parity, not only prefill/structural smoke.
3. Export DeepSeek-V2/V3 int4 fused-QMoE packages and run them through native CPU/CUDA QMoE.
4. Add explicit MLA Attention conformance for `Q/K head_dim != V head_dim` with past-cache decode on CPU and CUDA.
5. Once correctness is locked, design a metadata-declared, EP-internal structural MLA fusion/latent-cache path; the current standard Attention export forfeits MLA cache compression.
6. Green/publish the shared GLM PR because the DeepSeek QMoE implementation is currently carried on that branch.

## 4. Dependency-ordered, file-disjoint work packages

The packages below have disjoint owned files. Acceptance commands may consume another package's artifact but must not edit its files.

### DS-0 — Green and publish the shared GLM/DeepSeek QMoE branch

**Dependency:** none.  
**Suggested owner:** Sapper/Chew.  
**Owned files:** Mobius PR #404 branch only; resolve the branch's current CI/rebase failures without touching onnx-genai files.

Current status: coordinator todo `publish-glm-pr` is pending, but PR #404 already exists and points to `glm5.2-moe-export@cd782dd`. As of this audit it is an open draft, `mergeStateStatus=DIRTY`, with lint/test/integration failures. Thus “publish” now means rebase/green/ready-for-review, not create a new PR. This PR contains the shared QMoE emitter and grouped-top-k repair needed by DeepSeek.

**Acceptance:** PR #404 head contains `93cbcf7` and `cd782dd`, all required checks green, draft status removed or an explicit remaining-review note recorded.

### DS-1 — Native dynamic shape-chain unblock

**Dependency:** none; highest-priority runtime task.  
**Suggested owner:** Deckard.  
**Owned files:**

- `crates/onnx-runtime-session/src/executor.rs`
- `crates/onnx-runtime-session/tests/executor.rs`
- `crates/onnx-runtime-shape-inference/src/handlers/movement.rs`
- `crates/onnx-runtime-shape-inference/tests/graph_inference.rs`

Extend model-agnostic runtime output-shape fallback so a dynamically resolved `Slice` can feed `Unsqueeze` and subsequent broadcast/movement nodes. Reuse ONNX op semantics; do not key on node names such as `model/Unsqueeze_node_9`.

**Acceptance:**

```text
ONNX_GENAI_BACKEND=native ONNX_GENAI_EP=cpu \
DEEPSEEK_V2_TINY_E2E_DIR=/home/justinchu/ds-e2e-artifacts/deepseek-v2-tiny \
cargo test --locked -p onnx-genai-engine --features 'native-backend cuda' \
  --test deepseek_e2e -- --ignored --nocapture
```

must complete eight generated tokens. Add a small `Slice -> Unsqueeze -> comparison/broadcast` executor regression and run the shape-inference/session test suites.

### DS-2 — Deterministic V2/V3 export and fused-QMoE fixtures

**Dependencies:** DS-0.  
**Suggested owner:** Sapper.  
**Owned files:**

- `/home/justinchu/mobius/export_deepseek_v2_tiny.py`
- new `/home/justinchu/mobius/export_deepseek_v3_tiny.py`
- new `/home/justinchu/mobius/src/mobius/models/deepseek_test.py`
- `/home/justinchu/mobius/tests/integration_test.py` DeepSeek-only test section

Produce deterministic packages for:

1. fp32 V2 MLA+MoE;
2. int4 per-expert `MatMulNBits` V2;
3. int4 fused `QMoE` V2;
4. tiny-config V3 sigmoid/noaux QMoE.

Add decode-with-past HF parity, grouped-routing tests, QMoE node-count/layout assertions, and a regression for the repaired `_group_topk_selection` path.

**Acceptance:** ONNX checker passes; V2/V3 prefill plus at least two decode steps match HF logits within calibrated tolerance and exact greedy token identity; fused artifacts contain one `QMoE` per routed MoE layer and no routed-expert `MatMulNBits` nodes.

### DS-3 — MLA Attention primitive conformance

**Dependency:** none; can run in parallel with DS-1/DS-2.  
**Suggested owner:** Pris/Chew.  
**Owned files:**

- `crates/onnx-runtime-ep-cpu/src/kernels/attention.rs` test module only
- `crates/onnx-runtime-ep-cuda/tests/standard_attention_gpu.rs`

Add the exact structural property used by DeepSeek MLA: `qk_head_dim != v_head_dim`, 3-D BSH inputs with explicit head attributes, non-empty past K/V, and prefill/decode parity. Include GQA/MQA head sharing even though today's Mobius MLA expands to equal Q/KV head counts.

**Acceptance:** scalar oracle == CPU; CUDA == CPU within the established numeric bound; output and present-value shapes use V width while present-key uses Q/K width.

### DS-4 — Native CPU DeepSeek QMoE correctness/offload

**Dependencies:** DS-1, DS-2, DS-3.  
**Suggested owner:** Deckard.  
**Owned files:**

- `crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs`
- `crates/onnx-runtime-ep-cpu/src/weight_offload.rs`
- new `crates/onnx-runtime-ep-cpu/tests/deepseek_qmoe.rs`

Feed generic QMoE with V2 softmax/group-masked scores and V3 sigmoid+bias/noaux scores generated by the fixture. Validate selected expert IDs, separate aggregation weights, shared-expert addition at graph level, and resident vs mmap route-first equality. Keep routing outside the kernel as tensor inputs; do not add a `deepseek` mode attribute.

**Acceptance:** 1/2/4/8-bit kernel parity remains green; V2 and V3 fixture rows match the decomposed float expert oracle; `ONNX_GENAI_WEIGHT_OFFLOAD=1` loads only selected expert slices and is token-identical.

### DS-5 — Native CUDA DeepSeek QMoE prefill/decode

**Dependencies:** DS-2, DS-3.  
**Suggested owner:** Leon/Sebastian.  
**Owned files:**

- `crates/onnx-runtime-ep-cuda/src/kernels/qmoe.rs`
- `crates/onnx-runtime-ep-cuda/src/kernels/qmoe_gemm.rs`
- `crates/onnx-runtime-ep-cuda/src/kernels/qmoe_grouping.rs`
- `crates/onnx-runtime-ep-cuda/tests/qmoe_gpu.rs`

Exercise V2/V3 route tensors through decode GEMV and grouped prefill GEMM. Measure route distribution, workspace, and expert grouping; no architecture constants or model-name gates.

**Acceptance:** H200 CPU/CUDA output parity for V2 and V3 route cases; deterministic lower-index tie behavior; prefill uses grouped GEMM for multi-token experts; steady decode has no host expert-routing round trip.

### DS-6 — Engine dual-backend DeepSeek E2E gate

**Dependencies:** DS-1 through DS-5.  
**Suggested owner:** Pris.  
**Owned files:**

- `crates/onnx-genai-engine/tests/deepseek_e2e.rs`
- new `crates/onnx-genai-engine/tests/deepseek_qmoe_e2e.rs`

Turn the current one-backend random smoke into a matrix: ORT fp32, native CPU fp32, ORT QMoE, native CPU QMoE, native CUDA QMoE. Pin artifact commit/config and assert exact token identity between equivalent deterministic packages.

**Acceptance:** prefill plus eight decode tokens for each available backend; QMoE model bytes are checked for the fused op; failures do not silently skip when an explicitly configured artifact exists.

### DS-7 — Metadata-declared latent MLA cache, then EP-internal fusion

**Dependencies:** DS-3 and DS-6; performance work after correctness.  
**Suggested owner:** Roy for contract, then Deckard/CUDA owner for implementation.  
**Owned files:**

- `crates/onnx-genai-metadata/src/schema.rs`
- `crates/onnx-genai-metadata/src/validation.rs`
- `docs/MODEL_METADATA.md`
- `crates/onnx-runtime-ep-cpu/src/optimizer.rs` plus a new CPU MLA kernel
- `crates/onnx-runtime-ep-cuda/src/optimizer.rs` plus a new CUDA MLA kernel

First define generic metadata for latent-KV layout, LoRA ranks, rotary/non-rotary widths, and cache state. Then detect the projection/normalization/split/RoPE/Attention topology structurally inside each EP. Do not inspect `model_type`, initializer names, or DeepSeek dimensions.

**Acceptance:** same logits/tokens as decomposed Attention; cache bytes/token track metadata-derived latent rank instead of full per-head K/V; fusion declines with an actionable reason when metadata or shape/dtype compatibility is missing.

## 5. Available fixtures

### Usable now

`/home/justinchu/ds-e2e-artifacts/deepseek-v2-tiny/`

- `model.onnx` — 129,317 bytes
- `model.onnx.data` — 459,776 bytes
- `inference_metadata.yaml`
- `tokenizer.json`

The graph has 195 nodes, including two standard `Attention` nodes and decomposed MoE (`MatMul`, `TopK`, `GatherElements`, `OneHot`, etc.); it has no QMoE/custom CSA op. Metadata currently labels it generic `multi_head_attention` (`inference_metadata.yaml:1-16`). ORT E2E passes; native currently fails at dynamic `Unsqueeze` shape resolution.

### Not available

- `~/ana-bench`: no DeepSeek path/model found outside its Python environment fixtures.
- No local DeepSeek-V4-Flash ONNX package was found.
- The `tests/e2e/mobius_heads.json` V4 entry is only a manifest pointer, not an artifact (`tests/e2e/mobius_heads.json:17-27`).
- `~/Olive-recipes/deepseek-ai-DeepSeek-R1-Distill-*` contains distill recipes/configs; these are Llama/Qwen-family models, not DeepSeek MLA+MoE ONNX fixtures.
- Mobius `testdata/cases/causal-lm/deepseek-v2-lite.yaml` is an export case descriptor, not a generated model.

## 6. Shared MoE opportunity with GLM

GLM and DeepSeek already share the relevant exporter implementation:

- GLM imports `DeepSeekMLA`, `DeepSeekMoEGate`, and `_DeepSeekMoEFFN` (`/home/justinchu/mobius/src/mobius/models/glm_moe_dsa.py:24-30,150,339-341`).
- `FusedQuantizedMoE` is explicitly shared and generic (`/home/justinchu/mobius/src/mobius/components/_moe.py:239-381`).
- Native CPU/CUDA QMoE kernels consume generic router probability/weight tensors; they do not encode GLM or DeepSeek identity.

Therefore DS-0/DS-2/DS-4/DS-5 unblock both families. The best shared milestone is: **one deterministic GLM artifact and one deterministic DeepSeek artifact, both emitting the same QMoE contract, passing the same native CPU/CUDA kernel suite with family-specific routing computed outside QMoE.**

PR status matters: Mobius PR #404 is already open draft at `cd782dd` but dirty/failing, so the pending `publish-glm-pr` work is on the critical path for stabilizing the shared emitter. DeepSeek should not fork a second MoE implementation while that branch is pending.

## 7. RULES.md §2 / §2.1 risks

1. **Current V4 CSA code contains exact architecture constants.** CPU rejects anything except `D=512`, `RD=64`, and ratio-4 `ID=128` (`crates/onnx-runtime-ep-cpu/src/kernels/compressed_sparse_attention.rs:725-737,1148-1155`). CUDA claim geometry likewise contains fixed 512/128 dimensions and a fixed 128-token dense window (`crates/onnx-runtime-ep-cuda/src/kernels/compressed_sparse_attention.rs:76-79,2271-2434`). Under `RULES.md:20-38`, these are review-blocking if treated as a general runtime architecture. Any future CSA work must express such requirements as versioned op attributes/metadata and shape compatibility, or clearly constrain them as a private schema version while planning a generic successor.
2. **Do not detect MLA by model name or tensor names.** The current artifact metadata says only `multi_head_attention`; a latent-cache optimization must add explicit inspectable metadata rather than infer “DeepSeek” from graph/initializer names (`RULES.md:24-28`).
3. **Fusion belongs inside the EP.** A future MLA optimization must structurally match the low-rank projection/norm/split/RoPE/Attention topology in CPU/CUDA EP optimizer code, not rewrite the generic loader or add a `deepseek_v2` branch (`RULES.md:30-38`).
4. **QMoE must remain family-neutral.** V2/V3/GLM routing differences should remain graph-computed `router_probs` and `router_weights`; adding a `deepseek`/`glm` kernel attribute would violate §2.
5. **Backend selection is currently too narrow.** `model_proto_requires_native_backend` hardcodes one op type (`engine.rs:2351-2367`). If CSA/BQMoE packages are added, replace this with capability/registry-driven selection or explicit metadata; do not grow a model/op-name special-case list.
6. **Tiny fixture dimensions are acceptable only in tests.** Never copy test values such as four heads, hidden 64, or V4's 512/128 widths into runtime dispatch.

## Recommended priority

1. DS-1 native dynamic shape-chain unblock.
2. DS-0 + DS-2 stabilize/publish the shared QMoE exporter and produce DeepSeek V2/V3 deterministic fp32/int4 fixtures.
3. DS-3/DS-4/DS-5/DS-6 prove MLA and QMoE on native CPU/CUDA end-to-end.

Only after these pass should latent MLA-cache optimization (DS-7) become the performance focus. V4-specific export/runtime integration should remain outside the active queue until its official configuration and MTP contract are usable.

**Inbox:** Merged and cleared `dave-mobius-decoder-metadata-fix.md`, `deckard-cudaattn-revision.md`, `deckard-gap1-schema-fix-merge.md`, `deckard-sebastian-gap1-review.md`, `eldon-zhora-mla-merge.md`, `gaff-deckard-rereview.md`, `garland-wp5.md`, `holden-sebastian-gap1-merge.md`, `joi-gemma4-e2b-gaps.md`, `joi-gemma4-native-gaps.md`, `joi-qwen3-0.6b-bench.md`, `keaton-native-specdecode-design.md`, `kowalski-zhora-mla-conformance-review.md`, `kowalski-zhora-unsqueeze-review.md`, `leon-vlm-scope.md`, `mariette-qwen3-export.md`, `rachael-qwen-ladder-review.md`, `rachael-qwen3-bench-review.md`, `sebastian-native-interface.md`, `tessa-progress-qwen-ladder.md`, `zhora-deepseek-mla-conformance.md`, `zhora-deepseek-native-unsqueeze.md`, `zhora-deepseek-scope.md`.
<!-- scribe-merge-2026-07-23T01-00-00Z-gap1-mla-qwen3-merges-end -->

<!-- scribe-merge-2026-07-22T21-35-00Z-wp2-ort-reconciliation -->
## 2026-07-22 — VLM WP1/WP2/WP3 reconciliation and ORT CUDA attention review

Decision archive gate checked at 2026-07-22T21-35-00Z: active ledger was 155203 bytes; no entries older than 2026-07-15T21:35Z were present to archive.

<!-- source: .squad/decisions/inbox/badger-gemma4-text-closure.md -->
### Gemma4 text-only image-modality closure
**By:** Badger
**Decision:** Mobius Gemma4 packages declare `embedding.image_features` as optional with `[0, hidden_size]` zero fallback under the opaque `image` presence key and gate `vision_encoder` on the same key.
**Rationale:** Text-only requests skip vision execution while still providing a valid embedding input, matching the generic optional-audio contract without model-specific runtime logic.

<!-- source: .squad/decisions/inbox/cambodia-vlm-wp3-stepexec.md -->
### Land metadata-driven VLM WP3 step execution
**By:** Cambodia
**Decision:** Branch `squad/cambodia-wp3-step-exec` commit `7c82127` executes every `every_step` component topologically from declared metadata, publishes all outputs to the shared pool, re-reads decoder dataflow inputs each step, and keeps `prompt_only`/`final_only` phase behavior distinct.
**Rationale:** Replaces one-output embedding-specific behavior with a generic component contract; engine tests, Clippy, fmt, VLM E2E tests, IR fixture validation, and architecture-name grep passed. Follow-up: remove transitional decode-side name fallbacks once packages provide complete `ModelIoSpec`.

<!-- source: .squad/decisions/inbox/dave-pr421-review.md -->
### Approve and merge Mobius PR #421 Gemma4 image optionality
**By:** Dave
**Decision:** APPROVE+merged PR #421 as `38cb789a51e68b5907d82fa67704a73fdef80902`.
**Rationale:** Emitted metadata remains generic, graph declarations produce the `image` presence gate and optional zero-shaped `image_features`, text-only execution skips vision and preserves decoder routing, Ruff and targeted tests passed, and substantive CI passed except a queued infrastructure integration job.

<!-- source: .squad/decisions/inbox/dave-vlm-wp1-emission.md -->
### Land native VLM package emission
**By:** Dave
**Decision:** Branch `dave-wp1-vlm-emission` commit `a56e26b` emits VLM components with graph-derived typed I/O, topological dataflow, phases, embedding-to-decoder routes, sparse KV/fixed state pairs, preprocessing, expansion, position programs, runtime assets, schema v1, and WP0 capability names.
**Rationale:** Processor selection is structural/config-driven, decoder state and positions use explicit registries, no model-family string controls pipeline structure, and Tiny IR acceptance packages validate for Gemma4 E2B, Qwen VL, and Phi4MM. Server grid-derived expansion remains runtime follow-up.

<!-- source: .squad/decisions/inbox/eldon-hodge-wp2-review.md -->
### Reject initial VLM WP2 named image processor
**By:** Eldon
**Verdict:** 🔴 REJECT commit `4c49b86f44807b3f8f964e093db120c3bdcc4237`; Sapper must revise and Hodge is locked out for this revision cycle.
**Rationale:** The implementation validated `ImageOutputBinding::source` but discarded it by collapsing all transforms into one global `value_ops` sequence, so branch-selected outputs executed unrelated transforms. Scope, model-family grep, fmt, Clippy, and 44 offline tests passed, but a half-vs-quarter mutation regression proved the source binding was ignored.

<!-- source: .squad/decisions/inbox/eldon-wp2-verdict.md -->
### Approve and fast-forward merge VLM WP2 image processor revision
**By:** Eldon
**Verdict:** 🟢 APPROVE Sapper revision `386e083`, fast-forward merged as `2af64f55424860d8507cfea2eaaefaff23b104d8`.
**Rationale:** Each output preserves and resolves its declared `source`, divergent half/quarter branches produce independent values, unsupported divergent structural branches fail explicitly, runtime logic has no model-family matches, fmt/Clippy/tests passed, and merge scope is only `crates/onnx-genai-preprocess/src/image.rs`.

<!-- source: .squad/decisions/inbox/gaff-ort-review.md -->
### Reject ORT CUDA attention branch pending RuntimeConfig registry integration
**By:** Gaff
**Verdict:** 🔴 REJECT Howie commit `7ff33496bda2`; Howie is locked out of this artifact and Deckard is the reviser.
**Rationale:** `session.rs` reads `ONNX_GENAI_CUDA_ATTENTION` directly with `std::env::var_os`, violating the 2026-07-14 runtime-config decision that new runtime flags must be declared, parsed, documented, and tested in `onnx-genai-runtime-config::RuntimeConfig`, with call sites consuming only `runtime_config()`. CUDA-missing behavior, ORT option mapping evidence, model-family grep, fmt, Clippy, and tests otherwise passed.

<!-- source: .squad/decisions/inbox/hodge-vlm-wp2-image.md -->
### Implement initial WP2 named image preprocessing descriptors
**By:** Hodge
**Decision:** Branch `squad/hodge-wp2-image-processor` commit `4c49b86` preserved/validated WP0 transform inputs/outputs and output sources, executed generic image value operations, and retained typed named bundle outputs without model identity dispatch.
**Rationale:** The work consumed the WP0 named operation graph and added runtime tests with pinned references and architecture-neutral fixtures, but Eldon later rejected it because branch source bindings were validated but not actually executed independently.

<!-- source: .squad/decisions/inbox/howie-ort-cuda-attention.md -->
### Record rejected ORT CUDA attention artifact context
**By:** Howie
**Decision:** Branch `squad/howie-ort-cuda-attention` commit `7ff3349` made explicit CUDA EP requests fail actionably when CUDA ORT providers are unavailable and exposed unfused ORT CUDA attention through a session/provider option and `ONNX_GENAI_CUDA_ATTENTION=unfused`.
**Rationale:** The correctness workaround is generic and H200 reproduced coherent Qwen output with a 146.71 tok/s median, but Gaff rejected the artifact because the environment flag bypassed the required typed `RuntimeConfig` registry.

<!-- source: .squad/decisions/inbox/nandez-wp3-review.md -->
### Approve and merge VLM WP3 step-component execution
**By:** Nandez
**Verdict:** 🟢 APPROVED and fast-forward merged to main at `7c821278db17d66aef0672eb0decbb6b9c669da3`.
**Rationale:** Scope is exactly the authorized files, `decode.rs` is untouched, pipeline model-name grep is empty, execution is metadata-driven/topological/generic with no `EmbedsStepBinding` special case, and fmt, Clippy, and engine tests passed.

<!-- source: .squad/decisions/inbox/rachael-joi-verdict.md -->
### Approve Gemma4 E2B native profiling report
**By:** Rachael
**Verdict:** 🟢 APPROVE Joi's `profile_native` pipeline/steady-window additions and Gemma4 E2B benchmark documentation; rebased merge commit `39b2add`.
**Rationale:** Static review found metadata-driven pipeline selection without model-architecture assumptions; the 140.09 tok/s median is internally consistent, and fmt, package check, bench-native profile check, and Clippy passed. No GPU benchmark was run.

<!-- source: .squad/decisions/inbox/resch-dave-wp1-review.md -->
### Approve and merge Dave WP1 native VLM emission
**By:** Resch
**Verdict:** 🟢 APPROVED Dave's `dave-wp1-vlm-emission` work and squash-merged Mobius PR #420, advancing Mobius main to `38616311ed38db79b7ce0e6d5b2071f14f8da5b8`.
**Rationale:** Production VLM dispatch is structural/config-driven with no model-identity branch, targeted tests and Ruff passed, and emitted position/dataflow/capability/enum values match onnx-genai Rust and JSON schemas.

<!-- source: .squad/decisions/inbox/sapper-wp2-image-processor-rev.md -->
### Revise WP2 image values into independent dataflow branches
**By:** Sapper
**Decision:** Revised `image.rs` so each `OutputSpec` retains its declared source, named rescale/normalize values compile from their own declared input lineage, outputs pack the selected branch, and unsupported structural branches are rejected explicitly.
**Rationale:** Fixes Eldon's rejection of the collapsed global value-op chain; the half-vs-quarter regression proves independent branch selection.

**Inbox:** Merged and cleared `badger-gemma4-text-closure.md`, `cambodia-vlm-wp3-stepexec.md`, `dave-pr421-review.md`, `dave-vlm-wp1-emission.md`, `eldon-hodge-wp2-review.md`, `eldon-wp2-verdict.md`, `gaff-ort-review.md`, `hodge-vlm-wp2-image.md`, `howie-ort-cuda-attention.md`, `nandez-wp3-review.md`, `rachael-joi-verdict.md`, `resch-dave-wp1-review.md`, `sapper-wp2-image-processor-rev.md`. Preserved living scope docs `keaton-native-specdecode-design.md`, `leon-vlm-scope.md`, `zhora-deepseek-scope.md`, and `joi-gemma4-e2b-gaps.md`.
<!-- scribe-merge-2026-07-22T21-35-00Z-wp2-ort-reconciliation-end -->
<!-- scribe-merge-2026-07-22T16-01-00Z-vlm-wp0-landed -->
## 2026-07-22 — VLM WP0 revision, DS-1 shape unblock, and inbox reconciliation

<!-- source: .squad/decisions/inbox/deckard-vlm-wp0-revision.md -->
### VLM WP0 revised and merged
**By:** Deckard
**Decision:** Landed `156853c58e74deaf1e29a3f6da4ac552447e3bbf` after generalizing four doc-comments, rebasing onto `ea2c0b9260eaebcb83358463da351ab426e64958`, and fast-forward merging to main.
**Rationale:** RULES.md §2 model-name gate is now empty; metadata tests, clippy, and fmt are green.

<!-- source: .squad/decisions/inbox/eldon-vlm-wp0-review.md -->
### Reject original VLM WP0 metadata contract
**By:** Eldon
**Decision:** 🔴 Rejected `61dfc4ca209afd19ceaf7fcea695b86abb688ebd`; Stelline was locked out for this revision cycle and Deckard was named reviser.
**Rationale:** The required whole-file RULES.md §2 gate still reported model-family references in `schema.rs`, even though the branch otherwise passed metadata test, clippy, and fmt and preserved the frozen WP-B1 optional-modality fields.

<!-- source: .squad/decisions/inbox/kowalski-ds1-shape-unblock.md -->
### Validate DS-1 native dynamic shape-chain propagation
**By:** Kowalski
**Decision:** Native execution now resolves data-dependent standard-domain `Slice` shapes from runtime integer inputs and reuses ONNX shape inference through `Unsqueeze` and broadcast consumers; the focused regression covers `Shape/Sub -> Slice -> Unsqueeze -> Less` and the DeepSeek-V2 tiny CPU E2E generated eight tokens.
**Rationale:** `cargo fmt -p onnx-runtime-session`, combined session/shape-inference clippy, `cargo test -p onnx-runtime-session`, `cargo test -p onnx-runtime-shape-inference`, and native CPU DS-1 E2E passed; no next native blocker was found.

<!-- source: .squad/decisions/inbox/morton-ds1-review.md -->
### Approve DS-1 dynamic Slice shape regression
**By:** Morton
**Decision:** 🟢 Approved rebased commit `ed8b58e` for merge.
**Rationale:** The regression executes the intended dynamic path without constant folding, uses generic shape-driven ONNX operators, has correct broadcast assertions, and passed fmt, clippy, and the targeted locked test.

<!-- source: .squad/decisions/inbox/niander-h200-bench.md -->
### Record H200 native decode roofline check
**By:** Niander
**Decision:** On main `3d84b9b`, Qwen2.5-0.5B native CUDA with device KV and whole-step graph replay measured 820.65 tok/s at length 128 and 781.20 tok/s at 1024, exceeding the RTX 4060 baseline by roughly 2.1x; eager remained much slower and graph smoke tests had zero fallbacks/KV transfers.
**Rationale:** Qwen stayed near the supplied 886 tok/s roofline, Phi-4-mini remained lower utilization, and ORT profiler comparison was excluded because request setup dominated rather than steady decode.

<!-- source: .squad/decisions/inbox/rachael-wp-b-optional-modality-design.md -->
### Preserve WP-B optional-modality typed contract design
**By:** Rachael
**Decision:** WP-B should use opaque request presence keys, `phases.<component>.when_present`, and `io.optional_inputs.<port>.absent` zero fallbacks keyed by real ONNX input ports; tensor presence consistency is explicit and fallbacks are materialized at destination endpoints.
**Rationale:** The design rejects conditional-edge and runtime rank-adapter scope for WP-B, keeps semantic adaptation in exporter-emitted ONNX graphs, defines WP-B1/B2/B3/B4 ordering, and requires Python ONNX fixtures to use the `onnx_ir` IR API.

<!-- source: .squad/decisions/inbox/stelline-vlm-wp0-metadata.md -->
### Record original VLM WP0 metadata contract attempt
**By:** Stelline
**Decision:** Commit `61dfc4c` added a model-agnostic VLM declaration surface for image preprocessing transforms/outputs, vision prompt expansion, multimodal positions, sparse KV/fixed state pairs, and required capabilities while leaving frozen WP-B optional-modality fields unchanged.
**Rationale:** This was superseded by Eldon's rejection and Deckard's revision because the whole-file RULES.md §2 model-name gate was not clean until the comments were generalized.

**Inbox:** Merged and cleared `deckard-vlm-wp0-revision.md`, `eldon-vlm-wp0-review.md`, `kowalski-ds1-shape-unblock.md`, `morton-ds1-review.md`, `niander-h200-bench.md`, `rachael-wp-b-optional-modality-design.md`, `stelline-vlm-wp0-metadata.md`. Preserved canonical inbox files `keaton-native-specdecode-design.md`, `leon-vlm-scope.md`, and `zhora-deepseek-scope.md`.
<!-- scribe-merge-2026-07-22T16-01-00Z-vlm-wp0-landed-end -->

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
