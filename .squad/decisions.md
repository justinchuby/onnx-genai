# Decisions

Canonical, append-only record of accepted team decisions. Only the Coordinator (via Scribe merge) writes here. Agents drop proposals in `decisions/inbox/`.

---

### 2026-07-20T00:00:00Z: Decisions archive rollover
**By:** Scribe
**What:** Archived all 2026-07-12 entries (68 KB) to `decisions/archive/2026-07-20T00-00-00Z-decisions-pre-0713.md`. decisions.md exceeded the 50 KB threshold; entries older than 7 days (relative to 2026-07-20) were moved to archive. Recent 2026-07-13+ entries are retained below.
**Why:** Keep the hot decisions file lean per Scribe charter (>=50KB → archive entries >7 days).

---


### 2026-07-14T02:37:00Z: Decisions archive rollover
**By:** Scribe
**What:** Archived 2026-07-13 entries + early 2026-07-14 entries (W1, W2, Implementation plan) to `decisions/archive/2026-07-14T02-37-00Z-decisions-pre-w3.md` (~90 KB). decisions.md was 127 KB; size-based archival triggered (>50 KB threshold).
**Why:** Keep the hot decisions file lean per Scribe charter. W3 onward (per-layer KV geometry engine consume, review, Pris W5a, K4 multi-layer, Milestones A and B) is retained in the live file.

---

### 2026-07-14: W3 — Engine consumes per-layer KV geometry; staging shim removed

**By:** Leon (Engine Dev — KV & Buffers)
**What:** Migrated `onnx-genai-engine` off Batty's W2 uniform-field staging shim
onto real per-layer KV geometry, and deleted the shim. The paged KV cache is now
built from a per-layer `Vec<LayerTensorConfig>` derived structurally from each
exported present-KV output shape, so heterogeneous head_dim models (Gemma-4 E2B
sliding=256 / full=512, and Gemma-3 12B by extension) page and slice correctly.
Fully model-agnostic — no model names, no hardcoded 256/512; dims come from ONNX
I/O shapes + `shared_kv.target_layers` metadata.
**Why:** Roy's E2B plan §3c / W3. Closes blocker #4's engine half after W2 landed
the kv-crate half.

#### Files changed
- `crates/onnx-genai-kv/src/paged_cache.rs` — **shim removal**: deleted the
  uniform `MaterializedKv::num_kv_heads` / `MaterializedKv::head_dim` fields and
  the `uniform_heads`/`uniform_head_dim` computation in `materialize_sequence`;
  removed the shim assertions from `heterogeneous_per_layer_geometry_round_trips_within_a_page`.
  Per-layer geometry now lives only on `MaterializedLayerKv`.
- `crates/onnx-genai-engine/src/kv_bridge.rs`
  - `KvModelInfo` gained `layer_configs: Vec<LayerTensorConfig>` (one per exported
    KV layer) + a `layer_tensor_config(layer)` accessor. `tensor_config` is kept
    only as a representative (layer-0) view for uniform-only consumers
    (`num_layers`/`dtype`, connector `KvPayload`).
  - `infer_kv_model_info` builds per-layer configs via new pure helper
    `layer_configs_from_key_outputs(&[TensorInfo])` (reads each exported present-key
    output's shape) instead of `key_outputs[0]` alone.
  - `mirror_present_kv_to_pages` now extracts each layer's present token with that
    layer's own `layer_tensor_config(idx)` (was uniform `tensor_config`).
  - New unit test `layer_configs_are_built_per_exported_kv_layer_with_shared_layer_fold`
    (heterogeneous head_dim + `num_kv_shared_layers` fold + target_layers mapping);
    extended the uniform infer test to assert `layer_configs`.
  - Updated the `MaterializedKv`/`MaterializedLayerKv` test literal for the new API.
- `crates/onnx-genai-engine/src/engine.rs` — target and draft paged caches now
  constructed via `PagedKvCache::new_with_layer_tensor_configs(page_size, dtype,
  layer_configs, num_gpu_pages)` (was `new_with_tensor_config`).
- `crates/onnx-genai-engine/src/speculative.rs` — extracted
  `shared_kv_slices_from_materialized(groups, &MaterializedKv)`; each group now
  reads `num_kv_heads`/`head_dim` from `materialized.layers[target_layers.last()]`
  (the specific target layer) instead of the removed global fields. New unit test
  `shared_kv_slices_pick_per_layer_geometry` (2×8 sliding vs 3×16 full + OOB error).

#### Shim removal — confirmed
`MaterializedKv::num_kv_heads` / `MaterializedKv::head_dim` are **fully deleted**.
No engine or kv reader references them; the only per-layer dims are on
`MaterializedLayerKv`. `grep` across the workspace for the uniform fields returns
nothing but the (unchanged) `PageTensorConfig`/`LayerTensorConfig`/`KvPayload`
fields, which are legitimately uniform.

#### `num_kv_shared_layers` → exported-index mapping (assumption)
The ONNX graph exports only `num_hidden_layers - num_kv_shared_layers` present-KV
entries (the last N layers reuse an earlier layer's K/V), named contiguously
`present.{0..M-1}`. `infer_kv_model_info` sorts by parsed layer index and produces
exactly M `LayerTensorConfig`s. Metadata `shared_kv.target_layers` indices are
therefore interpreted as **direct indices into the exported (post-sharing) KV
list** — `target_layers.last()` selects `materialized.layers[idx]`. No extra
offset is applied because the export has already folded the shared layers. This
is the contract documented in Roy §2/§0 and Batty's W3 guidance. **Assumption:**
the exported present outputs are contiguously numbered from 0 with no gaps for the
shared layers; Sapper's real E2B export (W1) will confirm the exact naming. If the
export instead emits sparse indices (e.g. skips the shared positions), the fold
would need index remapping — flag for W5b's real-fixture gate.

#### Validation (all clean)
- `cargo test -p onnx-genai-kv --lib` → **78 passed** (shim removal, no regressions).
- `cargo test -p onnx-genai-engine --lib` → **107 passed, 1 ignored** (incl. 2 new
  per-layer unit tests).
- `cargo test -p onnx-genai-engine --test gemma4_assistant_full --test gemma4_assistant_metadata_smoke`
  → **2 passed** (uniform shared-KV still token-identical — no regression).
- `cargo clippy -p onnx-genai-kv -p onnx-genai-engine --lib --tests -- -D warnings`
  → **clean**.
- `gemma4_assistant_mixed` (`#[ignore]`d, W5b) still **compiles**; not enabled.
- `cargo build --workspace` → green.

**Do NOT commit** — coordinator lands combined W2+W3. Next in the serialized
`onnx-genai-engine` chain: W4 (contiguous-KV greedy fallback) → W5b (enable the
mixed fixture regression + real E2E gate).

---

### 2026-07-14: W2+W3 per-layer KV geometry review

**By:** Chew (numerics/correctness reviewer)
**What:** 🟡 **SHIP-with-advisories** — commit `9db1a3c` correctly implements
heterogeneous per-layer head_dim for the paged cache + shared-KV speculative
path. All five correctness questions pass for the real E2B export (15 contiguous
present.{0..14}, sliding hd256 / full hd512). One **non-gating** advisory: the
external-KV-**connector** payload path in `kv_bridge.rs` still uses uniform
layer-0 geometry and would corrupt a heterogeneous model *if* the connector is
enabled — but it is dead code for E2B (connector defaults to `Null`). Ship W2+W3;
track the connector path before enabling it on any mixed-geometry model.

**Why (file:line evidence):**

**Q1 — Per-layer byte extraction: CORRECT.**
- `mirror_present_kv_to_pages` extracts each layer with *that* layer's config, not
  a global one: `kv_bridge.rs:232-236` uses `kv_model.layer_tensor_config(layer_idx)`
  and passes it to `extract_present_token`, which sizes/loops on the per-layer
  `config.num_kv_heads/head_dim` (`kv_bridge.rs:262-264`, axes via
  `kv_tensor_axes` `:505-513`). `layer_data` iterates `kv_model.layers` in order
  (`:201-225`) so `layer_idx` aligns with `layer_configs[idx]`.
- Materialize path reads each layer's own `head_dim` from `layer_out.head_dim`
  (`paged_cache.rs:287-306`) and calls `page.value_at_slot(page_size, head_dim, …)`
  — no layer-0/global head_dim survives.

**Q2 — `target_layers.last()` geometry + OOB guard: CORRECT & GUARDED.**
- `shared_kv_slices_from_materialized` picks dims from the specific target layer:
  `speculative.rs:1279-1288` reads `layer.num_kv_heads`/`layer.head_dim` from
  `materialized.layers.get(layer_idx)`. For E2B: sliding `.last()=13`→hd256,
  full `.last()=14`→hd512.
- OOB is a hard error, not a silent misread: `.get(layer_idx).with_context(…)`
  (`speculative.rs:1279-1284`) returns `Err` for any index ≥ layer count; covered
  by the new `target_layers: vec![9]` assertion (`speculative.rs:1394-1398`).

**Q3 — Page sizing is per-layer: CORRECT.**
- `Page::new` iterates `layer_configs` and sizes each component from its own geom:
  `page_table.rs:369-395` (`component_len = geom.num_kv_heads * page_size *
  geom.head_dim`, `scale_slots = geom.num_kv_heads * page_size`), accumulating
  distinct `data_offset`/`scale_offset` per component. Full layers therefore get
  2× the bytes of sliding layers. `allocate` rebuilds pages from the same
  `self.layer_configs` (`page_table.rs:770-774`). No uniform-sizing path remains
  (`component_len(config)` helper deleted, old `tensor_offset` removed
  `page_table.rs:925-931`).

**Q4 — Shim fully removed: CONFIRMED.**
- `MaterializedKv` no longer carries `num_kv_heads`/`head_dim` (`paged_cache.rs:36-50`,
  `:314-320`). Workspace grep for uniform reads on a materialized cache returns
  only `.layers[i].{num_kv_heads,head_dim}` (per-layer, in tests). Old `value_at`
  / `tensor_offset` callers: none. `MaterializedLayerKv` is the sole geometry
  carrier.

**Q5 — Write/read byte symmetry for both head_dims: CORRECT.**
- Write (`write_head_token`, `page_table.rs:474-476`) and read (`value_at_slot`,
  `page_table.rs:433-435`) both compute `within = head*head_len +
  token_offset*head_dim + dim` with `head_len = page_size*head_dim` using the
  **passed per-layer** `head_dim`/`page_size`, indexing the same
  `storage_layout[component]` slab whose offsets were laid out per-layer. Quant
  scale index `head*page_size + token_offset` matches on both sides. Round-trip
  proven for f32 (within/across pages) and fp8 in the three new kv tests
  (`paged_cache.rs:961-1058`).

**Contract vs Sapper W1 export: SATISFIED.**
- Runtime reads geometry structurally from each present-key shape
  (`layer_configs_from_key_outputs`, `kv_bridge.rs:139-146`), never from the
  misleading top-level `model.attention.head_dim: 256`. `layer_configs` and
  `layers` are both built from the same index-sorted `key_outputs`
  (`kv_bridge.rs:76`, `:86`, `:103-119`), so `shared_kv.target_layers` index the
  exported (KV-share-folded) list directly with no offset. Both target/draft
  paged caches are built via `new_with_layer_tensor_configs(page_size, dtype,
  layer_configs, …)` (`engine.rs:227-231`, `:249-256`).

**ADVISORY (non-gating) — connector KvPayload path is uniform-only.**
- `chunk_payload_from_exported` (`kv_bridge.rs:596-632`) and
  `past_kv_from_payloads` (`kv_bridge.rs:649-701`) size/extract **all** layers
  with a single `config.num_kv_heads/head_dim` (layer-0 / `kv_model.tensor_config`,
  `engine.rs:1316-1319`, `:1342`, `:1164`). On a heterogeneous model this would
  read full layers (hd512) as hd256 and mis-shape past tensors → corrupt KV.
- **Why non-gating:** both call sites are reachable only under
  `self.connector.is_active()` (`engine.rs:1091`; `store_connector_prefix`
  gated by a non-`Null` backend `:1316-1321`), and `KvConnectorBackend` defaults
  to `Null` (`config.rs:127,159`). For the E2B shared-KV run no connector is
  configured, so this is dead code. It is also pre-existing and explicitly
  documented as a "uniform-only consumer" in Leon's note — not a regression from
  this commit.
- **Follow-up owner:** the K-series / connector path owner (DESIGN §38) — NOT
  Batty or Leon. Recommend Roy assign a per-layer refactor of
  `chunk_payload_from_exported` / `past_kv_from_payloads` (thread
  `layer_tensor_config(idx)` like `mirror_present_kv_to_pages` does) as a gate
  **before** enabling any external KV connector on a heterogeneous model. Suggest
  tagging alongside the W5b mixed-fixture gate.

**Verdict: 🟡 SHIP-with-advisories.** W2+W3 paged-cache + shared-KV geometry is
byte-correct for the real E2B export; land it. Connector path advisory is the
only open item and is currently inert.

---

### 2026-07-14: Pris W5a — mixed-head_dim Gemma4-assistant fixture

**Date:** 2026-07-14
**By:** Pris (Tester)
**Status:** complete — fixture generated, existing suite green

---

## What was done

Extended the tiny Gemma4-assistant fixture builder to emit a **second** fixture with
**mixed per-layer KV head_dim**, mirroring Gemma-4 E2B's 256 (sliding) / 512 (full)
split at tiny scale. The original uniform-head_dim fixture
(`tests/fixtures/tiny-gemma4-assistant`, `HEAD_DIM=8` for every layer) is **not
changed** — all existing tests continue to pass.

---

## Files changed / added

| File | Change |
|------|--------|
| `scripts/build_tiny_gemma4_assistant_mixed.py` | **New** — the mixed-head_dim fixture builder |
| `tests/fixtures/tiny-gemma4-assistant-mixed/model.onnx` | **New** — target ONNX with heterogeneous KV dims |
| `tests/fixtures/tiny-gemma4-assistant-mixed/assistant/model.onnx` | **New** — assistant ONNX with heterogeneous shared-KV inputs |
| `tests/fixtures/tiny-gemma4-assistant-mixed/tokenizer.json` | **New** — copied from `tiny-llm` |
| `tests/fixtures/tiny-gemma4-assistant-mixed/manifest.json` | **New** — full layer→head_dim + shape documentation |
| `crates/onnx-genai-engine/tests/gemma4_assistant_mixed.rs` | **New** — W5b placeholder test (`#[ignore]`) |

---

## Fixture path

```
tests/fixtures/tiny-gemma4-assistant-mixed/
  model.onnx
  tokenizer.json
  manifest.json
  assistant/
    model.onnx
```

---

## Layer → head_dim map (for W5b assertions)

| Layer index | Group name          | head_dim | Mirrors E2B |
|-------------|---------------------|----------|-------------|
| 0           | `sliding_attention` | **8**    | 256         |
| 1           | `full_attention`    | **16**   | 512         |

Other fixture constants: `VOCAB=32`, `HIDDEN=16` (backbone hidden size), `KV_HEADS=2`, `NUM_LAYERS=2`.

---

## KV output shapes (target `model.onnx`)

| Output name       | Shape                                | head_dim |
|-------------------|--------------------------------------|----------|
| `present.0.key`   | `[batch, 2, total_seq_len, 8]`       | 8        |
| `present.0.value` | `[batch, 2, total_seq_len, 8]`       | 8        |
| `present.1.key`   | `[batch, 2, total_seq_len, 16]`      | 16       |
| `present.1.value` | `[batch, 2, total_seq_len, 16]`      | 16       |
| `logits`          | `[batch, sequence_len, 32]`          | —        |
| `hidden_states.0` | `[batch, sequence_len, 16]`  (f32)   | —        |

---

## Assistant shared-KV input shapes (`assistant/model.onnx`)

| Input name                          | Shape                       | head_dim |
|-------------------------------------|-----------------------------|----------|
| `shared_kv.sliding_attention.key`   | `[batch, 2, kv_len, 8]`     | 8        |
| `shared_kv.sliding_attention.value` | `[batch, 2, kv_len, 8]`     | 8        |
| `shared_kv.full_attention.key`      | `[batch, 2, kv_len, 16]`    | 16       |
| `shared_kv.full_attention.value`    | `[batch, 2, kv_len, 16]`    | 16       |

---

## W5b placeholder

`crates/onnx-genai-engine/tests/gemma4_assistant_mixed.rs` contains
`gemma4_assistant_mixed_speculative_matches_plain_greedy` decorated with:

```rust
#[ignore = "enable after W3 per-layer paged cache lands (see roy-gemma4-e2b-realrun-plan.md §4 W3)"]
```

The test asserts speculative == greedy token-identity on the mixed fixture. It
will not build-break or suite-break in any branch.

**To enable after W3 lands:** remove the `#[ignore]` attribute. No fixture
regeneration needed — the data is already committed. Confirm on CPU.

The assertion that must pass:
- `actual.token_ids == expected.token_ids`
- `stats.proposed_tokens > 0` (proposer was active)
- `stats.verification_steps > 0`

The key runtime fix required (W3): `shared_kv_proposer_slices` in
`speculative.rs:1246-1253` must read `head_dim`/`num_kv_heads` from the
**per-layer** paged config for the specific target layer referenced by each
group's `target_layers.last()`, not from a single global `materialized.head_dim`.

---

## Validation

- Builder runs cleanly: `python3 scripts/build_tiny_gemma4_assistant_mixed.py` ✅
- Mixed dims confirmed: layer 0 KV outputs have `head_dim=8`, layer 1 have `head_dim=16` (verified via `onnx_ir` I/O shape inspection) ✅
- Existing suite: `cargo test -p onnx-genai-engine --lib` → 105 passed, 0 failed ✅
- Existing integration tests: `gemma4_assistant_full` + `gemma4_assistant_metadata_smoke` → 2 passed ✅
- New placeholder: `gemma4_assistant_mixed` → 0 passed, 1 ignored (correctly skipped) ✅

---

### 2026-07-14: K4 Multi-Layer KV Coverage — Pris decision note

**Date:** 2026-07-14  
**Author:** Pris (Tester)  
**Advisory:** Chew review A1, §38 K4

## What was missing

The existing gold test `engine::tests::local_tiered_connector_fetch_reuse_is_token_identical`
and all related connector round-trip tests use a **1-layer** fixture (tiny-llm),
so the `layer_idx*2 = K, layer_idx*2+1 = V` packing contract and multi-layer
ordering were entirely untested. A transpose or layer-index bug would not have
been caught.

## What I added

Two synthetic unit tests, no ONNX model needed, no production-behavior changes.

### 1. `local_tiered::tests::multi_layer_store_fetch_preserves_exact_per_layer_kv_ordering`

**File:** `crates/onnx-genai-kv/src/local_tiered.rs`

- Builds a `KvPayload` with **3 layers, 2 kv_heads, 3 tokens, 4 head_dim**
  in head-major `[num_kv_heads, num_tokens, head_dim]` layout.
- Pattern: `key[l][(h·T+t)·D+d] = 1000·l + 100·h + 10·t + d` (positive);
  `val[l][(h·T+t)·D+d] = -(1000·l + 100·h + 10·t + d)` (negative).
- Stores in `LocalTieredConnector`, fetches back, asserts layer-by-layer exact
  equality on both K and V slots.
- Catches: layer swap, K/V slot swap, head/token/dim transposition.

### 2. `kv_bridge::tests::chunk_payload_from_exported_multilayer_preserves_layer_head_token_dim_ordering`

**File:** `crates/onnx-genai-engine/src/kv_bridge.rs`

- Constructs 3 `ExportedLayerKv` values with shape `[1, 2, 8, 4]`
  (batch=1, 2 kv_heads, 8 tokens total, 4 head_dim).
- Same position-encoding pattern as above.
- Calls `chunk_payload_from_exported(&exported, config, chunk_start=3, num_tokens=4)`.
  `chunk_start=3` ensures `token_pos ≥ 3` on all steps, cleanly exercising the
  sequence-axis detection in `kv_tensor_axes` (avoids the benign batch/sequence
  ambiguity that occurs only at `token_pos=0`).
- Asserts every `(layer, K/V, head, chunk-token, dim)` cell matches the expected
  value derived from the encoding formula.
- No ORT runtime needed — pure unit test within the engine crate.

## No production bug found

The implementation is correct. The `token_pos=0` ambiguity in `kv_tensor_axes`
is benign by construction (batch index 0 always gives the correct offset), and
all extraction logic for `token_pos ≥ 1` is correct. This was purely a
**test gap**, not a defect.

## Test results

```
cargo test -p onnx-genai-kv --lib        → 74 passed, 0 failed
cargo test -p onnx-genai-engine --lib    → 105 passed, 0 failed, 1 ignored
cargo clippy -p onnx-genai-kv -p onnx-genai-engine --lib --tests -- -D warnings → clean
```

## Files changed

- `crates/onnx-genai-kv/src/local_tiered.rs` — added 1 `#[tokio::test]`
- `crates/onnx-genai-engine/src/kv_bridge.rs` — added 1 `#[test]`
- `.squad/decisions/inbox/pris-k4-multilayer-test.md` — this note

---

### 2026-07-14: Milestone A ACHIEVED — real Gemma4 E2B target model, greedy decode, on CUDA

**By:** Sapper
**Status:** ✅ SUCCESS. The real 10.3 GB E2B `model.onnx` loads and generates coherent text on the H200 via the CUDA EP, target-only greedy (no speculation).

---

#### Result

- Prompt `"<bos>The capital of France is"` → **`Paris.`**
- Prompt `"<bos>Once upon a time, in a small village nestled deep in the mountains, there lived a young blacksmith named"` → coherent multi-paragraph story ("Elara. Elara was known throughout the village not just for her skill with the hammer and anvil, but also for her gentle spirit... a traveler, cloaked against the wind, carrying a mysterious satchel."). 160 tokens, fully coherent.

#### CUDA EP evidence (user directive 确保用cuda ep)

- Forced via `ONNX_GENAI_EP=cuda` (`session.rs:668 execution_providers_from_env`). Built `--features cuda` (see build-wiring note below).
- **nvidia-smi during generation: peak 19,127 MiB VRAM** (baseline ~6 GB from another process → ~13 GB is our f16 model + ORT/cuDNN workspace) and **peak 83% GPU compute utilization**. Definitively on-GPU, not a silent CPU fallback.
- ORT emitted `VerifyEachNodeIsAssignedToAnEp` (CUDA EP active; only shape ops on CPU as expected). No "falling back to CPU" warning.

#### Performance (target-only greedy, ~160 tok, short context)

- **Plain contiguous past→present KV path (`shared_buffer:false`): ~166 tok/s decode** (159 tokens in 0.957 s; load+prefill ~4.8 s).
- O(1) share-buffer path (`shared_buffer:true`, `max_len=4096`): ~48.6 tok/s at this length — slower only because it computes attention over the full 4096-capacity buffer every step; it wins at long context. Output byte-identical to the growing path.

---

#### Root cause of initial garbage output — MISSING BOS (a W1 packaging gap, NOT head_dim)

First runs produced degenerate `" France is France is..."`. **This was NOT the heterogeneous head_dim / KV cache** (contrary to the pre-run hypothesis). Root cause: the package tokenizer does **not** auto-prepend `<bos>`:
- `tokenizer_config.json` has no `add_bos_token`; `tokenizer.json` `post_processor` (TemplateProcessing) adds nothing.
- Gemma degenerates hard without BOS. Prepending the literal `<bos>` special token to the prompt immediately produced coherent output.
- **Action for W1/export:** fix the shipped package tokenizer to auto-add BOS (set `add_bos_token: true` and/or a TemplateProcessing single-template that inserts `<bos>`). Until then, callers must prepend `<bos>`.

#### Heterogeneous head_dim (256 sliding / 512 full at layers 4,9,14) — WORKS on the greedy path

Confirmed empirically in BOTH the plain growing path AND the share-buffer path: identical coherent output. Roy's §0 prediction holds — plain greedy uses the head_dim-agnostic contiguous ORT past→present threading (`decode.rs:1107-1117`, opaque per-name values) and does **not** materialize from the paged cache, so the uniform-`head_dim` paged-cache assumption (`kv_bridge.rs infer_kv_model_info`, `paged_cache.rs MaterializedKv`) is never exercised by target-only greedy. **No W4 contiguous-KV fallback was needed for Milestone A.** That fallback is still required for Milestone B (shared-KV speculative), which slices the paged cache.

---

#### How target-only greedy was isolated from the speculative auto-path (config-driven, no model hardcoding)

The shipped merged `inference_metadata.yaml` carries a `speculative:` block. With `EngineConfig::default()`, `engine.rs:266-269 shared_kv_mode_from_metadata` auto-adopts `SpeculativeMode::SharedKv` (Milestone B path) whenever that block is present. To run TARGET-ONLY, I pointed the runtime at a sibling "view" directory `~/gemma4-e2b-onnx-target/`:
- Symlinks to the same `model.onnx`(+`.data`), `tokenizer.json`, `tokenizer_config.json` (no 10 GB copy).
- A **stripped** `inference_metadata.yaml` with **no `speculative:` block** (so `detect_speculator` returns None → no speculative path) and **no share-buffer hints** (no `max_sequence_length`, no kv dtype → `shared_kv_buffer_len_from_metadata` returns None → `detect_model_decode_path` yields the plain contiguous `PastPresent{shared_buffer:false}`). Also dropped `sliding_window` (with it present + a share-buffer hint, `decode.rs:819-822` bails; sliding vs full attention is identical for ≤512-token greedy anyway).

No model names or Gemma-specific logic in runtime code — purely metadata/config-driven.

---

#### Reproduce

`scripts/run_target_greedy_cuda.sh` (uncommitted, left for coordinator review). It builds the target-only view dir + stripped metadata, builds `--features cuda`, sets `ONNX_GENAI_EP=cuda`, and generates. Env knobs: `SRC`, `TARGET_DIR`, `PROMPT`, `MAX_NEW_TOKENS`.

#### One required code change (uncommitted, working tree)

`crates/onnx-genai/Cargo.toml`: added a passthrough feature so the CLI can enable CUDA:
```
[features]
default = []
cuda = ["onnx-genai-ort/cuda"]
```
The CLI crate previously had no `cuda` feature, so `cargo build -p onnx-genai --features cuda` failed ("does not contain this feature"). This 3-line addition is the minimal fix; feature unification routes it to the single `onnx-genai-ort` build. Recommend committing this — the CLI is otherwise unable to select CUDA. (Left unstaged per task instructions.)

#### Follow-ups for the team

1. **W1/export:** ship a BOS-adding tokenizer in the E2B package (blocks coherent output for any BOS-sensitive model otherwise).
2. **Milestone B (Roy §3d "W4 contiguous-KV fallback"):** still needed — shared-KV speculative slices the paged cache, whose per-layer geometry is inferred from layer 0 only (`kv_bridge.rs`, `paged_cache.rs MaterializedKv` single head_dim; `speculative.rs:1246-1253` reuses one global head_dim). Heterogeneous 256/512 will trip there.
3. Consider a first-class "target-only" / "disable speculative" `EngineConfig` switch so proving a target doesn't require a stripped-metadata view dir.

---

### 2026-07-14: Milestone B — Milestone B — real Gemma4 E2B shared-KV speculative decode on CUDA (Leon)

## Result: PASS (token-identity) ✅

Ran the FULL merged package `~/gemma4-e2b-onnx/` (10.3GB E2B target + 359MB
assistant drafter) on CUDA (H200), shared-KV speculative path auto-selected from
the `speculative:` metadata block.

- **Token-identity: TRUE** — shared-KV speculative decode is token-for-token
  identical to plain target-only greedy on the REAL heterogeneous per-layer
  head_dim weights (sliding hd256 / full hd512). Verified on two prompts
  (64 and 96 tokens). This is the Milestone B correctness pass bar.
- **CUDA EP verified** — ~34 GB peak VRAM, up to 50% GPU util during the run,
  no ORT CPU-fallback error (only the benign "shape ops on CPU" warning).
- **Coherent text**, e.g. `"<bos>The capital of France is"` → `" Paris.\n"`;
  a Rayleigh-scattering paragraph for the sky-blue prompt.

## Speedup + acceptance (release build, H200)

| prompt | greedy tok/s | spec tok/s | speedup | acceptance | multi-accept |
|--------|-------------:|-----------:|--------:|-----------:|-------------:|
| capital-of-France (64) | 105.5 | 56.2 | 0.53x | 25.4% | 0 |
| sky-blue (96)          | 110.9 | 60.0 | 0.54x | 25.3% | 0 |

Speculative is currently **slower** on this model. Acceptance is ~25% with
`multi_token_accepts == 0`: the drafter's FIRST proposed token is accepted every
step (the shared-KV benefit), but it never gets 2+ tokens ahead, so each step
commits ~2 tokens while paying K draft forwards + host-side KV materialization.
Per the mission, token-identity (not speedup) is the pass bar, which is met.

Likely acceptance ceiling = the `projected_state` hidden space (Sapper chose the
backbone post-norm last_hidden_state, f32). The drafter threads its OWN
`projected_state` output into the next `inputs_embeds`; after the first token the
threaded state drifts from the space the assistant was trained on, so tokens 2..K
miss. This is a speedup limiter only — the target verifies every token, so greedy
correctness is unaffected. Not chased further here (would risk the pass bar).

## Engine/ORT changes made (config-driven, model-agnostic, no hardcoding)

1. `decode.rs::detect_model_decode_path` — sliding-window models now take the
   bounded paged sliding-window path (`shared_buffer: false`) even when they also
   declare a share-buffer-eligible KV dtype. Previously bailed ("append-only
   shared KV buffer" guard), which blocked every fp16 GQA SWA model (Gemma-style,
   incl. the real E2B target: `sliding_window: 512` + `kv_cache fp16`). The
   append-only single buffer can't express windowed eviction, so it is skipped in
   favor of the paged windowed path, not refused.
2. `shared_kv_proposer.rs` — made dtype-agnostic. The real assistant's
   `inputs_embeds` and `shared_kv.*` inputs are Float16 (was hardcoded Float32).
   Activation dtype is now taken from `inputs_embeds`; shared-KV inputs must match
   it; float outputs are read via a lossless f16→f32 widening.
3. `value.rs` — added `Value::from_f32_slice_as(data, shape, dtype)` (f32 binds
   directly, f16 narrows per-element). Shared by the proposer and kv_bridge.
4. `kv_bridge.rs::load_materialized_past` — injects the target's past KV in the
   graph's declared past-input dtype (f16 for E2B) instead of hardcoded f32. For
   an fp16 model this is the exact inverse of the fp16→f32 widening done when
   mirroring present KV, so no precision is lost.
5. `onnx-genai-engine/Cargo.toml` — added a `cuda` feature forwarding to
   `onnx-genai-ort/cuda` (mirrors the CLI passthrough), so the engine's own tests
   can exercise the CUDA EP.
6. `tests/milestone_b_real.rs` — env-gated (`ONNX_GENAI_MB_FULL` /
   `ONNX_GENAI_MB_TARGET`), `#[ignore]`d real-model harness that asserts
   token-identity and reports acceptance/speedup. No-ops without the env vars, so
   CI stays hermetic.

## Validation

- `cargo test -p onnx-genai-engine --lib` (107 passed) + `--test
  gemma4_assistant_full` + `--test gemma4_assistant_metadata_smoke` — all green.
  Updated one lib test (`windowed_past_present_keeps_absolute_positions_with_bounded_past`)
  that asserted the removed bail; it now asserts the paged windowed path.
- `cargo clippy -p onnx-genai-engine --lib --tests` (+`--features cuda`) and
  `-p onnx-genai-ort` — clean under `-D warnings`.

## Follow-ups (not blocking)

- Speedup: investigate the `projected_state` hidden-space / threading to lift
  acceptance beyond the first token; and reduce per-step host-side KV
  materialization overhead on the shared-KV verify path.

---

### 2026-07-14: Milestone B engine fixes review (10f82b3)
**By:** Chew (numerics/correctness reviewer)
**What:** 🟢 **SHIP.** Leon's fp16 shared-KV speculative decode changes are correct, config-driven, and model-agnostic. The decode-path change is narrower than it looks (it only converts a former bail into the path SWA models already used) and does not touch non-SWA models. All fp16↔f32 conversions are lossless in the required directions and layout-preserving. No new hardcoded dtype or model assumptions. Verified READ-ONLY (git show / view / grep); did not build/test/clippy per instruction.

---

**Why (file:line evidence):**

**1. decode.rs path selection — SAFE for models other than E2B.**
- `crates/onnx-genai-engine/src/decode.rs:817-841`. Old code: `if sliding_window.is_some() && shared_kv_max_len.is_some() { bail }` then `if sliding_window.is_some() { return paged windowed }`. New code drops the bail; `if sliding_window.is_some()` now unconditionally returns `PastPresent { shared_buffer: false, sliding_window, sink_tokens }`, logging a debug when `shared_kv_max_len` is also set.
- **Blast radius is bounded to SWA models only.** Any model with `sliding_window == None` never enters this branch and reaches the unchanged share-buffer logic at `decode.rs:850-873` exactly as before — so non-SWA share-buffer models are **not** diverted (the stated regression risk does not occur). The only behavioral delta is: `sliding_window.is_some() && shared_kv_max_len.is_some()` now takes the paged windowed path instead of erroring, and `sliding_window.is_some()` alone was already on that same path pre-commit. The rationale is sound: an append-only single shared buffer genuinely cannot express windowed eviction, so the paged path is the correct destination for every fp16/fp32 GQA SWA model — config-driven off `sliding_window` metadata, no hardcoded model names (`sliding_window_from_metadata`, `decode.rs:882-895`).
- **StaticCache still guards SWA first** (`decode.rs:801-810`): a SWA model that also matches the static-cache signature still bails there, unaffected.

**2. Updated lib test asserts the CORRECT path, not made-to-pass.**
- `kv_bridge.rs:1199-1210`. Was `assert!(detect_model_decode_path(&session, Some(16), Some(16), Some(2), 0).is_err())`. Now asserts `Ok(PastPresent { shared_buffer: false, max_len: None, sliding_window: Some(2), sink_tokens: None })`. I checked each field against the code: `shared_window=Some(2)` is passed through; `sink_tokens=0` → `(0 > 0).then_some(...)` = `None` (`decode.rs:840`); `max_len: None` and `shared_buffer: false` are literals on that return. The assertion is exact and correct. That `Ok(...)` also proves `StaticCacheDecodeSession::detect` returns `None` for the test session (else it would bail at `decode.rs:803`), consistent with the returned variant.

**3. shared_kv_proposer.rs — dtype-agnostic, widening in the right direction.**
- Activation dtype is derived from the graph (`float_dtype = embeds_input.dtype`, `shared_kv_proposer.rs:330`) and propagated to `signature.dtype` (:356) and to `shared_kv_specs(session, float_dtype)` which requires every `shared_kv.*` input to match it (:387-392) — internal-consistency check, no assumed dtype.
- Inputs bind via `Value::from_f32_slice_as(..., self.signature.dtype)` (:174, :212, :215): f32 → direct copy, f16 → per-element narrow. Narrowing f32 host activations to an fp16 graph input is required (not a lossy bug) — it reproduces what a native fp16 model consumes.
- Outputs read via `to_vec_f32_lossy()` (`:466`, and `value.rs:268-296`): f32 direct, f16 widened losslessly through the `half` crate. Engine-facing API stays f32 regardless of graph dtype. Direction is correct (widen on read, narrow on write).

**4. value.rs from_f32_slice_as — narrowing correct, f32 path is a plain copy.**
- `value.rs:164-179`. `Float32` → `from_slice_f32` (plain copy, no reinterpret). `Float16` → `half::f16::from_f32(x).to_bits()` per element (IEEE-754 round-to-nearest-even), collected into a `Vec<u16>` of length `numel`, then `from_vec_f16_bits` which validates `shape` against `data.len()` (`value.rs:203-217`). Element count is preserved (one u16 per input f32), so **no byte-count/stride bug** and no transpose — shape and data ordering are identical to the previous `from_vec_f32` call site.

**5. kv_bridge.rs load_materialized_past — widen/narrow are true inverses.**
- Mirror (capture): `mirror_present_kv_to_pages` reads present KV via `to_vec_f32_lossy()` (`kv_bridge.rs:208,217`) → fp16→f32 is exact. Paged storage holds f32.
- Inject: `load_materialized_past` narrows back with `from_f32_slice_as(&materialized.layers[idx].{key,value}, &shape, {key,value}_dtype)` where the dtype is the graph's declared **past-input** dtype (`kv_bridge.rs:326-341`). For an fp16 model the stored f32 values are exactly fp16-representable (they originated from fp16 present outputs and are only copied/indexed by `extract_present_token`, no arithmetic), so round-to-nearest narrowing returns the identical fp16 bits → **lossless round-trip**. Shape comes from `past_shape(...)` (`:308-322`), unchanged from before, so no layout/stride change — the commit only swapped the final `Value` dtype, keeping shape and f32 ordering. Using the past-input dtype (not present-output) is the correct model contract and remains exact even in the mixed case (fp16-origin value into an f32 past input).

**6. Cargo.toml — trivial passthrough confirmed.**
- `crates/onnx-genai-engine/Cargo.toml`: `cuda = ["onnx-genai-ort/cuda"]` under a new `[features]` block with `default = []`. Exactly the requested passthrough.

**No remaining hardcoded dtype/model assumptions in scope.** Grep for `Float32`/`from_vec_f32` shows the only remaining hardcoded-f32 injection is `past_kv_from_payloads` (`kv_bridge.rs:719-722`) — the *connector* KV-fetch path, which is explicitly and pre-existingly guarded by `kv_model_past_is_f32` (`kv_bridge.rs:732-748`, "Non-f32 KV ... is skipped so a dtype mismatch can never corrupt injected output"). That path is out of scope for the shared-KV speculative feature and correctly gated; not a regression.

**milestone_b_real.rs** is `#[ignore]`d and no-ops without `ONNX_GENAI_MB_*` env vars (`milestone_b_real.rs:39-49`), config-driven (prompt/budget/paths from env), no model-specific logic — CI stays hermetic.

---

**Non-gating advisories (do not block ship):**
- **[nit, arguably an improvement]** `detect_shared_kv_proposer` reordered so the embeds/logits/projected dtype check (`:322`) now runs *before* the `shared_kv.is_empty()` early-return (`:332`). A graph carrying the exact proposer I/O signature (`inputs_embeds` + `logits` + `projected_state`, no `mtp_hidden`) but with a non-float dtype and zero `shared_kv.*` inputs would now surface an `Err` where it previously returned `Ok(None)`. This is a near-impossible collision (that signature is proposer-specific) and the new behavior is more correct (a malformed proposer should error, not be silently ignored), so it is not a concern — just noted for awareness. Owner if ever revisited: Leon.
- Follow-ups from Leon's decision note (speculative currently 0.53x, ~25% acceptance, `multi_token_accepts == 0` due to `projected_state` hidden-space drift) are speedup-only and do not affect greedy/token-identity correctness — the target verifies every token. Out of scope for this correctness review.

**Verdict: 🟢 SHIP.** Token-identity is the Milestone B pass bar and the numerics support it: every fp16↔f32 conversion is exact in its required direction, the paged-path round-trip is a true inverse for fp16 KV, and the path-selection change cannot regress non-SWA models.

---

### 2026-07-14: `onnx-runtime-ep-api` hardening — DeviceBuffer ownership, DLPack alignment, Cost forward-compatibility (Track B)
**By:** Batty (Engine Dev)
**Commit:** 65ec9f6 | **Reviewed:** 🟡 Holden (safety)
**What:** Replaced the raw-field `DeviceBuffer` skeleton with an encapsulated unique-ownership handle. Key contracts: sole owner (never freed by a different EP); no `Drop` (leaks rather than double-frees); `unsafe` construction; `*const`/`*mut` access split by mutability; `Send`/`Sync` are sound (no safe deref, no interior mutability). Added `byte_offset: usize` + `i64` strides to `TensorView`/`TensorMut` for DLPack spec compliance; `validate()` checks invariants; `storage_bytes` used for int4/uint4. `Cost` made `#[non_exhaustive]` with helpers (`Cost::ZERO`, `Cost::new`, `with_launch_us`, `with_bytes_moved`). Deferred: `EpRegistry::load_legacy` and `OrtGraphView::query_capabilities` left as `todo!()` skeletons.
**Why:** ep-cpu (Track C) and any future EP must uphold these contracts when wiring real memory. Alignment, DLPack compliance, and cost-model extensibility are correctness-critical seams for Phase 2.

---

### 2026-07-14: `onnx-runtime-ep-cpu` — CpuExecutionProvider + 7 Phase-1 kernels (Track C)
**By:** Batty (Engine Dev)
**Commit:** ea30279 | **Reviewed:** 🟡 Chew (numerics) + 🟡 Holden (safety)
**What:** First real EP against the merged `onnx-runtime-ep-api`. Implements `CpuExecutionProvider` (alloc/dealloc/copy/supports_op/get_kernel) with pure-Rust reference kernels for all 7 Phase-1 ops: MatMul (N-D batched + broadcast, strided/transposed), Add (numpy broadcasting), Relu, Reshape, Transpose (perm attr), Gather (axis, neg-idx, N-D), LayerNormalization (axis/epsilon, optional Bias/Mean/InvStdDev). 39 unit tests pass; clippy clean. No C++/FFI/build.rs — zero new dependencies. Storage-bounds enforcement via `strided::view_in_bounds` (handles negative strides); `unsafe` isolated to aligned alloc/dealloc, `copy_nonoverlapping`, and two strided element accessors. Track D (session) MUST call `strided::view_in_bounds` before dispatch; kernels trust their caller for storage bounds.
**Why:** Correctness reference EP for BERT-on-CPU; naive triple-loop GEMM is the Phase-1 exit bar. The `Kernel` trait is the perf swap boundary for a Phase-1.5 oneDNN/BLAS pass.

---

### 2026-07-14: `onnx-runtime-loader` WeightStore re-export + norm_axis fix
**By:** Deckard (Systems Dev)
**Commit:** dd5297d | **Reviewed:** 🟡 Gaff
**What:** Two new public functions: `load_model_with_weights(path)` and `load_model_bytes_with_weights(bytes, base_dir)` both returning `(Graph, Arc<WeightStore>)`; `WeightStore` re-exported from crate root. Existing `load_model`/`load_model_bytes` are thin wrappers (backward-compatible). Track D session usage: store `Arc<WeightStore>` alongside `Graph`; call `store.bytes(weight_ref)` for zero-copy access (handles both `Inline(TensorData)` and `External{path, offset, length}` variants). `norm_axis` fix: positive axis was clamped to `rank` (inclusive), allowing `axis == rank` and causing index panics in `gather`/`concat`; clamped to `rank.saturating_sub(1)`. Well-formed BERT models unaffected.
**Why:** `load_model` was not returning the live `WeightStore`, forcing sessions to re-mmap. The norm_axis off-by-one was latent correctness risk for any future models with axis at rank boundary.

---

### 2026-07-14: `onnx-runtime-loader` (Track A) — ONNX proto pipeline
**By:** Deckard (Systems Dev)
**Commit:** 7e0e367 | **Reviewed:** 🟡 Gaff
**What:** Full ONNX → `onnx_runtime_ir::Graph` pipeline. Vendored `onnx.proto3` (ONNX v1.16.0) compiled via `protox` (pure-Rust, no system `protoc`). `graph_builder`: `GraphProto` → IR `Graph` with typed values, symbolic dim interning, subgraph recursion, `opset_imports`. `weights`: inline (`WeightRef::Inline`) and external (`WeightRef::External` + `memmap2`) initializers. `shape_inference`: topo-order driver + rule table + constant-aware path; covers BERT op set (MatMul, Gemm, broadcasts, unary, LayerNorm, Transpose, Gather, Concat, Shape, Reshape, Unsqueeze, Squeeze, Reduce*). Deferred: Slice, Pad, Conv, Split, NonZero, control-flow (skipped, not `todo!()` panic). 15 tests (11 unit + 4 integration). Smoke-loaded real fixtures (tiny-eagle3, tiny-qwen35-mtp, tiny-llm-scatter, tiny-gemma4-assistant, tiny-whisper) all pass `validate()`.
**Why:** Foundation for Track D (session): `load_model` → `Graph` with initializers, shape-inferred SSA values, and opset context. IR gaps flagged for Roy: `DataType::from_onnx` fp8/int4 numbering, no `DataType::Undefined`, no unknown-rank `Shape`.

---

### 2026-07-14: Perfetto trace export #13 — review decision (🟢 SHIP)
**By:** Deckard (reviewer)
**Commit reviewed:** 8d1bf3d
**What:** 🟢 SHIP. `GET /v1/debug/trace/perfetto` serves Chrome Trace Event Format (Perfetto) document, gated behind `enable_debug_endpoints`. All 6 criteria pass: gate parity (same `if` block as sibling debug routes; 404 when off, not 403/500), no data leak (`TraceEvent.name` is `&'static str` — no runtime strings injectable), refactor safety (`write_trace` delegates to `trace_document`; mutex-guarded, no `unwrap`), honest empty case (well-formed empty trace document on no spans), OTLP deferral (explicit "deferred" status), model-agnostic (subsystem-level stage names, no model names). Metrics `ENDPOINTS` extended to 14 with consistent `endpoint_index`.
**Why:** Security gate and data-leak checks are Deckard's sign-off criteria per reviewer protocol. Zhora validated build/test/clippy separately.

---

### 2026-07-14: Gemma4 E2B shared-KV speculative acceptance — root cause + fix
**By:** Leon (Engine Dev — speculative decode)
**Commit:** 8089a1f | **Reviewed:** 🟡 Chew
**What:** Root cause: the shared-KV assistant's `pre_projection` expects `inputs_embeds = concat(target_input_embedding(last_token), last_hidden_state)` (per HF `SinglePositionMultiTokenCandidateGenerator`). The engine was feeding `concat(prev_hidden, cur_hidden)` — no token embedding, both halves backbone hidden — causing all t2/t3 drafts to be garbage and deterministically rejected (`multi_token_accepts == 0`). Fix: new optional `speculative.input_embedding` field → `LinearEmbedder` on the proposer. `SharedKvProposer::propose` rewritten: seed `last_token = last context token`, each step feeds `concat(embed(last_token), last_hidden)`; position is constant (no `position_ids`; RoPE from frozen shared-KV mask). Results: acceptance 25% → 70.6%, `multi_token_accepts` 0 → 12/17 steps, token-identical preserved. Speculative still 0.58x (drafter compute cost); speedup requires lower `num_speculative_tokens` or lighter lm_head — separate follow-up.
**Why:** The engine contract bug masked by the guaranteed-token free slot. Fix is model-agnostic and config-driven.

---

### 2026-07-14: Gemma4 E2B `input_embedding` durable artifact — Mobius export
**By:** Sapper (model-packaging/export)
**Commit:** 2fed4f7 (mobius repo @ feat/gemma4-assistant-onnx-genai)
**What:** Made `input_embedding.f32` a first-class Mobius export artifact. New `_find_scaled_token_embedding(target_model, hidden_size)` locates the token-embedding `Gather` + post-`Mul` scale from the **target graph** — nothing hardcoded (vocab, hidden, scale all read from graph). New `write_input_embedding_artifact` writes `weight * scale` as raw little-endian f32 `[vocab, hidden]` (1.6 GB for Gemma4 E2B). `speculative.input_embedding` emitted in `_speculative_block`, `generate_merged_inference_metadata`, and YAML serializer. `write_merged_inference_metadata` gained optional `target_model` param. Scale is read from the graph's f16 `Mul` constant (`39.1875` for Gemma4 E2B); differs from Leon's manual `sqrt(1536) = 39.1918` by 1.1e-4 (within one f16 ULP) — negligible acceptance impact. 23 integration tests pass. No engine code touched.
**Why:** Leon's engine fix requires `speculative.input_embedding` in the package. Durable export means `Engine::from_dir` works with no manual extraction steps.


---

### 2026-07-14: `onnx-runtime-session` — sequential CPU executor (Track D)
**By:** Roy (Lead)
**Commit:** 24b8129 | **Reviewed:** 🟢 Chew (numerics) + 🟡 Holden (safety)
**What:** `SessionBuilder::build()` drives loader → EP init → buffer allocation (one `DeviceBuffer` per value, sized from static IR shapes) → kernel compile into a shape-keyed `(NodeId, input_shapes)` cache. `run(inputs)` validates dtype+shape, copies input bytes, walks topo order materializing contiguous `TensorView`/`TensorMut` over each buffer, runs `view_bounds` gate before every dispatch, then collects owned output `Tensor`s. Borrow/aliasing strategy: output buffers removed from map before dispatch (SSA guarantees disjoint), reinserted after; Miri-clean. `Tensor` lives in this crate (Phase-1 CPU; flag for move to shared crate before ep-cuda). 8/8 tests pass; clippy clean.
**Why:** Ties loader + ep-api + ep-cpu into a running inference session. Remaining gaps for BERT milestone: dynamic shapes, op coverage (op set now supplied by ep-cpu expansion), C-API, conformance gate.

---

### 2026-07-14: `onnx-runtime-session` — runtime symbolic-shape resolution (Track D, dynshape)
**By:** Roy (Lead)
**Commit:** da8eab3 | **Reviewed:** 🟡 Holden (safety)
**What:** `run()` now resolves concrete shapes at run time: (1) `bind_symbols` — walks declared loader shape dim-by-dim against actual input shapes, binding `Dim::Symbolic(s) → usize`; rank/static mismatch and symbol conflict are errors. (2) `resolve_all` — substitutes bindings into every value's loader shape; unbound symbol → `UnresolvedShape`. (3) `size_buffers`/`ensure_buffer` — reuse buffer if `buffer_shapes[vid] == dims`, else dealloc + realloc. Kernel cache keyed on resolved (concrete) input shapes; same shapes → hit + reuse; new shape → re-validate + recompile. Static graphs are the zero-symbol special case. 14/14 tests pass; Miri-clean including multi-batch realloc/reuse (`symbolic_batch_matmul_chain_runs_for_multiple_shapes`).
**Why:** BERT inputs carry symbolic dims (`batch`, `seq_len`). Static-only executor could not run real models. Loader (Deckard) owns shape inference; session owns symbol→concrete resolution — the design seam is intentional. Loader gaps for `Attention`/`EmbedLayerNormalization` shape rules flagged for Deckard.

---

### 2026-07-14: `onnx-runtime-capi` — Phase-1 Tier-1 C ABI (Track E)
**By:** Batty (Engine Dev)
**Commit:** 8c9c8fc | **Reviewed:** 🟢 Holden (safety)
**What:** `extern "C"` surface wrapping `onnx-runtime-session`. Opaque handles (`OrtSession`, `OrtValue`, `OrtStatus`) via `Box::into_raw`/`Box::from_raw`. Entry points: `ort2_create_session`, `ort2_release_session`, `ort2_create_tensor` (validates dtype + exact byte-len), `ort2_release_value`, `ort2_run` (atomic commit — all outputs built before any slot written; on error: pre-nulled + freed), `ort2_get_tensor_{dtype,rank,shape,data}`, status accessors. `crate-type = ["lib"]` (cdylib deferred to Phase 2 OrtGetApiBase vtable). Every fallible body wrapped in `catch_unwind` via `guard` helper; every pointer null-checked. `SessionError → OrtErrorCode` mapping covers NoSuchFile/InvalidProtobuf/InvalidArgument/NotImplemented/EpFail/InvalidGraph/Fail. 12/12 tests pass; Miri-clean (`-Zmiri-disable-isolation`).
**Why:** Closes Phase 1. Thin, model-agnostic C marshalling layer; no hardcoded shapes/ops. Once `onnx-runtime-session` runs a full BERT graph, the C ABI drives it with no changes needed here.

---

### 2026-07-14: `onnx-runtime-ep-cpu` — +17 BERT kernels (op expansion for bert_toy)
**By:** Batty (Engine Dev)
**Commit:** e485a83 | **Reviewed:** 🟡 Chew (numerics)
**What:** 17 new kernels added to `onnx-runtime-ep-cpu`, all registered in `PHASE1_OPS` via `build_cpu_registry`. Elementwise binary: `Sub`, `Mul`, `Div`, `Pow`, `Min` (via existing `broadcast_apply`). Unary: `Sqrt`, `Erf` (A&S 7.1.26 in f64, max abs err 1.39e-7), `Tanh`. Type: `Cast` (fixed-width numeric dtypes, float→int truncates, NaN→bool = true). Reduction: `ReduceMean` (multi-axis, keepdims). Shape/movement: `Shape`, `Unsqueeze`, `Expand`, `Slice` (opset-10 input-driven, negative/stepped ranges). Constant: `Constant` (value/value_float(s)/value_int(s)). GEMM: `Gemm` (transA/transB, alpha/beta, bias broadcast). Dtype-generic byte movers (`elem_size`, `to_dense_bytes`, `write_dense_bytes`) added to `kernels/mod.rs`. 90 tests pass; clippy clean; no new dependencies. Softmax intentionally uses opset-13 per-axis semantics (identical to opset-12 coerce on last axis — all bert_toy Softmax nodes). Loader gaps flagged: `Slice`/`Expand`/`Constant` shape inference needed (owner: Deckard).
**Why:** Supplies the op coverage gap for the BERT-on-CPU milestone; executor needs no changes.

---

### 2026-07-14: `onnx-runtime-loader` — const-fold-lite shape inference (Slice/Expand/Constant)
**By:** Deckard (Systems Dev)
**Commit:** b6f032e | **Reviewed:** 🟢 Gaff (correctness)
**What:** Bounded partial evaluator (`ConstEnv: HashMap<ValueId, KnownVal>`) filled in topo order alongside existing shape rules. `KnownVal` = rank-0/1 integer tensor with `IntElem::Const(i64) | IntElem::Sym(SymbolId)`. Bound: rank ≤ 1, numel ≤ 1024 (`MAX_FOLD_ELEMS`), integers/bools only. Value-propagation ops: `Constant`, `Shape` (emits Sym for symbolic dims), `Identity`, `Cast` (integral only), `Unsqueeze`, `Squeeze`, `Concat`, `Gather` (axis-0, 1-D), `Slice` (opset-10), `Reshape`, `Add`/`Sub`/`Mul`/`Div`/`Min`/`Max` (any symbolic operand → fresh symbol). Shape rules added: `Reshape` (symbolic-aware), `Slice` (rank-preserving, symbolic bounds), `Expand` (broadcast vs const/sym target). On `bert_toy_optimized.onnx`: unresolved values 135→50; all 50 residuals are genuine rank-0 scalar `Constant`s. No `UnresolvedShape` for any structural op. Position-slice chain stays symbolic (data-dependent, by design). 27/27 tests pass (including `bert_toy_optimized_every_value_resolves` on real model); clippy clean; `#![forbid(unsafe_code)]` retained; public API unchanged.
**Why:** Session executor errors `UnresolvedShape` for any value the loader leaves shape-less. Batty's ep-cpu data-movement kernels require pre-allocated output views with correct shapes.

---

### 2026-07-14: Chew review — session executor Track D (🟢)
**By:** Chew (correctness/numerics reviewer)
**Target:** `squad/ort2-session` @ `edbc3fd` | **Verdict:** 🟢 SHIP-with-minor-advisories
**What:** Verified topological order (Kahn's algorithm, min-heap tie-break, cycle detection), value dependency resolution (one buffer per `ValueId`, SSA-disjoint), view materialization (contiguous strides, zero offset correct for dedicated per-value buffers), initializer/input binding (dtype+shape validated), output collection (correct prefix slice), shape-keyed cache (no collision — fresh `NodeId` per node). Test references hand-verified in Python (MatMul→Add→LayerNorm→Relu chain). Advisories (non-blocking): optional-input compaction may shift positional alignment for gappy-optional ops; cache key omits dtypes. No correctness bug found.

---

### 2026-07-14: Holden review — session executor Track D (🟡)
**By:** Holden (safety/soundness reviewer)
**Target:** `squad/ort2-session` @ `edbc3fd` | **Verdict:** 🟡 SHIP-with-advisories
**What:** All 5 invariants held: (1) view bounds gated on every input + output before dispatch via `view_bounds`+`?`; (2) single-free via `Option::take` in `Tensor::drop`, `drain` in `Executor::drop`; (3) no cross-EP free — allocator carried on returned `Tensor`; (4) copy size validated before `copy_nonoverlapping`; (5) host malloc is global. Aliasing claim verified: in-place ops cause `CycleDetected` at build. Miri clean. Advisories: A1 — mid-run error path leaks output buffers (`DeviceBuffer` has no `Drop`); A2 — unchecked i64 arithmetic in `view_in_bounds` (theoretical overflow); A3 — cache key omits dtypes. None blocking.

---

### 2026-07-14: Holden review — session dynshape (🟡)
**By:** Holden (safety/soundness reviewer)
**Target:** `squad/ort2-session-dynshape` | **Verdict:** 🟡 SHIP-with-advisories
**What:** Invariant #1 (view bounds) holds against new run-scoped buffers — gate keys off real `buf.len()` not assumed shape, so even stale `buffer_shapes` cannot bypass it. Buffer-reuse cannot yield undersized-but-passing buffer (two independent layers: correct sizing + real-length gate). No new aliasing introduced. Single-free/no-leak on re-allocation — `deallocate(old)` before `allocate`, Miri-clean across batch 2→3→2 reuse test. 14/14 tests pass. Advisories: H-D1 — unchecked `dims.iter().product()` overflows mod 2⁶⁴ and gate is congruent (very low reachability); H-D2 — stale `buffer_shapes` if `allocate` fails post-dealloc (clean error, not UB); Holden-A1 (pre-existing) — mid-run error-path buffer leak unchanged.

---

### 2026-07-14: Holden review — C ABI Track E (🟢)
**By:** Holden (safety/soundness reviewer)
**Target:** `squad/ort2-capi` | **Verdict:** 🟢 SHIP
**What:** Verified all 6 FFI soundness axes: (1) null-guards on every handle and pointer before deref, returning `InvalidArgument`; (2) every fallible body in `catch_unwind` via `guard`; (3) `Box::into_raw`/`Box::from_raw` once each, null-tolerant releases, atomic commit in `ort2_run`; (4) `create_tensor` validates `data_len == storage_bytes(numel)` before slice construction; (5) `CStr::from_ptr(..).to_str()` with UTF-8 error → `InvalidArgument`; (6) 12/12 tests pass, Miri clean. Advisories (non-blocking): A1 — release fns not in `guard` (panic-free today but relies on `Drop` invariants); A2 — `storage_bytes` unchecked multiply (only reached inside `guard`, bounded by prior validation).

---

### 2026-07-14: Chew review — ep-cpu BERT kernels +17 (🟡)
**By:** Chew (correctness/numerics reviewer)
**Target:** `squad/ort2-epcpu-ops` @ `4f2465e` | **Verdict:** 🟡 SHIP-with-advisories
**What:** 90/90 tests pass. No blocking numeric bug for bert_toy. Independently verified: Softmax stability vector `[1000,1001,1002]→[0.090,0.245,0.665]` ✓; broadcast `[3,1]·[1,4]→[3,4]` ✓; Erf max abs err 1.39e-7 ✓; Gemm `[[58,64],[139,154]]` ✓. Elementwise binaries, ReduceMean, Gemm, Slice, Cast, data-movement kernels spec-faithful. Advisories: A1 — Softmax uses opset-13 per-axis (bit-identical to opset-12 coerce only on last axis; bert_toy Softmax all last-axis but model not in-repo — conformance harness must confirm); A2 — `Min` uses `f32::min` (returns non-NaN; ONNX propagates NaN — no bert_toy impact); A3 — Cast float→int saturates on overflow vs ORT UB (documented, no bert_toy impact). Non-blocking hardening for A1 (opset<13 guard when axis≠rank-1) assigned to Roy or Deckard.

---

### 2026-07-14: Gaff review — loader const-fold-lite shape inference (🟢)
**By:** Gaff (correctness reviewer)
**Target:** `squad/ort2-loader-shapeinfer` | **Verdict:** 🟢 SHIP
**What:** No wrong constant found. Every fold aborts via `?`/`None` on missing/non-integer/unfolded operands — never invents a constant. Verified: Gather symbolic index → `None` ✓; Slice requires `all_const` starts/ends ✓; Concat requires all inputs in env ✓; binop any-Sym → fresh symbol (unit-tested) ✓; Reshape handles -1/0 correctly ✓; Slice clamp math correct ✓. Symbolic identities propagate via interned `SymbolId`. Bounds enforced at every entry point. `bert_toy_optimized_every_value_resolves` ran on real model (257 KB, not skipped) — no `UnresolvedShape` on structural ops; position-slice chain correctly symbolic. Advisories: A1 — `Div` truncates toward zero vs floor for negative operands (no positive-dim impact; elem_to_dim maps negatives to fresh symbol); A2 — `Shape` of unresolved input folds to rank-0 (pre-existing, no bert_toy impact). 27/27 tests pass.

### 2026-07-14: ORT2 must support ORT's EPContext node (com.microsoft)
**By:** Coordinator (Justin Chu)
**What:** ORT2 must support ORT's on-disk EPContext contrib operator (domain com.microsoft, variadic inputs/outputs) — distinct from the internal `EpContext` cache struct. Scope: (1) Loader parses EPContext attrs (main_context, ep_cache_context, embed_mode, ep_sdk_version, hardware_architecture, partition_name, source, notes, max_size); (2) Session/EP-API dispatches an EPContext node to the EP whose `source` attribute matches, feeding blob to the EP's load_context path; (3) Generation via ep.context_enable / ep.context_file_path / ep.context_embed_mode session options; (4) C-API surfaces those options. Model-agnostic: dispatch by `source` attribute only — no hardcoded EP names. Roy to author design in docs/ORT2.md (branch squad/ort2-epcontext-design).
**Why:** Central to EP ecosystem interoperability. ORT2 must consume/emit pre-compiled EP-binary models that the ORT ecosystem produces (QNN, OpenVINO, TensorRT, etc.).

---

### 2026-07-14: ORT2 shape inference reference: onnx-shape-inference
**By:** Coordinator (Justin Chu)
**What:** Shape inference (optimizer `ShapeInference` pass + evolution of `onnx-runtime-loader/src/shape_inference.rs`) must follow patterns from https://github.com/justinchuby/onnx-shape-inference: (1) extensible per-op registry keyed by (domain, op_type, opset_version); (2) symbolic dim arithmetic (SymPy-style expr trees over `Dim::Symbolic`); (3) shape DATA propagation as first-class subsystem tracking known values of shape tensors through Shape→Slice→Concat→Reshape chains; (4) strict/permissive merge policies for unifying inferred vs declared shapes.
**Why:** User-designated reference. Keeps inference extensible, opset-aware, model-agnostic, and feeds the optimizer richer shape info.

---

### 2026-07-14: ORT2 `onnx-runtime-optimizer` — Phase-2 optimizer crate
**By:** Roy (Lead)
**What:** New `crates/onnx-runtime-optimizer/` crate (`#![forbid(unsafe_code)]`, depends only on `onnx-runtime-ir`). Implements `OptimizationPass` trait + `PassContext` (empty, `#[non_exhaustive]`) + `run_passes` + `OptimizerError`, and three passes: `ConstantFolding` (integer only, ≤1024 elems, checked arithmetic, fixpoint; folds `Constant`/`Shape`/integer binops), `DeadNodeElimination` (backward reachability from outputs), `OpFusion` (escape-safety rule: non-final matched outputs must stay within matched set; reuses final `ValueId`; patterns: MatMul+Add+Relu→FusedGemm, MatMul+Add→FusedMatMulBias, 9-op LayerNorm). Default pipeline: ConstantFolding → DCE → OpFusion. bert_toy: 384→278 nodes, 0 Constants, 32 FusedMatMulBias; LayerNorm fusion correctly declines (DAG-shaped shared `mean`). 26 unit + 1 real-model integration tests; clippy clean.
**Why:** Foundation for all Phase-2+ graph rewriting; pass contract and fusion safety invariant locked before more passes added.

---

### 2026-07-14: Gaff review — optimizer structural integrity (🟢)
**By:** Gaff (graph/IR integrity reviewer)
**Target:** `squad/ort2-optimizer` @ `87a16d9` | **Verdict:** 🟢 SHIP
**What:** All 6 integrity checks HELD. Node removal/GC via `remove_node` correct; fusion removes last-first, reuses final `ValueId`; ConstantFolding `needed` guard prevents stale initializer; arena safety (stale-id checked before deref); DCE+fusion interaction verified adversarially. `Graph::validate()` postcondition verified as genuinely biting (injected dangling edges and bogus consumer links → `Err`). 27 tests pass; clippy clean. Advisories (non-blocking): A1 — external-input ordering structural not schema-aware; A2 — validate() debug-only (intentional per §18.1).

---

### 2026-07-14: Chew review — optimizer correctness (🟡)
**By:** Chew (correctness reviewer)
**Target:** `squad/ort2-optimizer` @ `87a16d9` | **Verdict:** 🟡 SHIP-with-advisories
**What:** All three passes semantics-correct. OpFusion escape-safety invariant correct (necessary-and-correct condition for not deleting an observed value). ConstantFolding never folds non-const inputs, checked arithmetic (overflow aborts, not wraps), fixpoint correct. DCE from outputs (never from inputs). bert_toy verified: 384→278 nodes, 0 Constants, 32 FusedMatMulBias, validate() clean. Advisories: A1 — fused ops emitted in default ONNX domain (must use private domain e.g. com.ort2.fused before any kernel binds; tie refinement to kernel introduction); A2 — greedy spine matcher under-fuses on multi-successor (never miscompiles); A3 — single-output final-node restriction.

---

### 2026-07-14: Roy — BERT-toy conformance milestone ACHIEVED
**By:** Roy (Lead)
**Branch:** `squad/ort2-bert-conformance`
**What:** `bert_toy_optimized.onnx` (opset 12, 384 nodes) runs end-to-end through `onnx-runtime-session` on CPU and matches onnxruntime 1.27.0/CPUEP. Max error: prediction_scores 1.19e-7 (tolerance 2e-3), seq_relationship_score 6.05e-9. Zero Phase-1 cross-crate bugs. One session-local fix: position-embedding Slice takes a data-dependent `Shape→Cast→Min→Cast` extent requiring JIT dynamic-shape resolution in the executor — model-agnostic (dispatch on op type only; ops without JIT resolution surface `UnresolvedShape`). 15 tests pass; Miri clean.
**Why:** Phase-1 exit milestone. Proves the full stack (loader, ep-cpu, ep-api, session) composes correctly on a real transformer with correct numerics.

---

### 2026-07-14: Chew review — BERT conformance JIT output sizing (🟡)
**By:** Chew (correctness reviewer)
**Target:** `squad/ort2-bert-conformance` | **Verdict:** 🟡 SHIP-with-advisories
**What:** Slice sizer is character-for-character mirror of Slice kernel — buffer always equals what kernel writes. `buffer_as_i64` LE decode correct. JIT loop ordering correct. Conformance harness sound (allclose semantics, both outputs, deterministic inputs). Advisories: A1 — Slice count math duplicated verbatim (extract shared `slice_axis_count` helper — structural risk of silent drift); A2 — pre-existing degenerate Slice corner (not BERT-impacting); A3 — multi-output index robustness when extending beyond Slice; A4 — tolerance comment vs allclose code mismatch.

---

### 2026-07-14: Holden review — BERT conformance JIT alloc/dealloc (🟡)
**By:** Holden (safety/soundness reviewer)
**Target:** `squad/ort2-bert-conformance` | **Verdict:** 🟡 SHIP-with-advisories
**What:** All 4 soundness invariants HELD. view_in_bounds gate on every input+output before dispatch including JIT-sized outputs (JIT sizes first, gate validates JIT shapes against JIT-resized buffers). No use-after-free (error path exits before dealloc/alloc loop). dealloc-before-alloc ordering safe (no live `TensorView` aliasing freed buffer). No new `unsafe`. Miri clean (0 UB; -Zmiri-disable-isolation for disk-read conformance test). Advisories: H-D1 carry-over (unchecked dims.product + storage_bytes in JIT path — same as ensure_buffer; non-regression); multi-output index panic when op returns fewer shapes than outputs.

---

### 2026-07-14: Batty — ep-cpu + session Phase-1 hardening (6 advisories + capi fix)
**By:** Batty (Engine Dev)
**What:** (1) Softmax opset≤12 vs ≥13 dual semantics via `coerce_2d` flag + dual registry (SoftmaxLegacy@1, Softmax@13); `effective_opset` plumbed end-to-end. (2) Min/Max NaN-propagation — explicit `is_nan()` guard before `f32::min/max`. (3) Cast saturate — `num_to_int!` macro converting directly to target type (no i64-intermediate-then-wrap). (4) `checked_numel` + `SessionError::ShapeOverflow` at both alloc sites (H-D1 preliminary). (5) Multi-output `dynamic_output_shapes` guard (`OutputShapeCountMismatch` before index). (6) Slice geometry extracted to shared `slice_plan` + `slice_axes_steps` helper (kernel + sizer share one impl). Also fixed capi `map_session_error` non-exhaustive match — added explicit arms for `SymbolConflict/RankMismatch/UnresolvedShape/ShapeOverflow/OutputShapeCountMismatch` (no catch-all `_`); all-crate build restored.
**Why:** Real correctness gaps (wrong Softmax for non-last-axis opset≤12, NaN swallowed, Cast garbage, panic on future multi-output op) closed before more models arrive. Holden's `view_in_bounds` gate preserved untouched.

---

### 2026-07-14: Holden review — ep-cpu hardening (🔴 → Deckard)
**By:** Holden (safety/soundness reviewer)
**Target:** `squad/ort2-hardening` @ merge-base vs main | **Verdict:** 🔴 REJECTED
**What:** Checks 2–6 HELD (view_in_bounds gate intact, multi-output guard, opset plumbing, capi FFI, tests/clippy/Miri). Check 1 FAILED: `checked_numel` closed dims-product overflow but `DataType::storage_bytes(numel)` still computed `count * byte_size` unchecked. Shape `[2^61]` of f64: checked_numel OK (=2^61), storage_bytes wraps to 0, `.max(1)`→1-byte alloc; `view_in_bounds` i64 gate also wraps → passes → heap OOB in release. **Batty locked out of H-D1 storage-sizing artifact. Fix assigned to Deckard** (or another non-Batty implementer). Re-review by Holden required before merge.
**Why:** Unchecked overflow reaching allocation = 🔴 per soundness rubric; exact H-D1 class.

---

### 2026-07-14: Deckard — H-D1 three-layer overflow fix
**By:** Deckard (Systems Dev)
**Commits (cherry-picked to main):** dbf2d70, 9dcdc04, f749012
**What:** Layer A (`dtype.rs`): `DataType::checked_storage_bytes(count) -> Option<usize>` — `div_ceil(2)` for sub-byte, `checked_mul(byte_size())` for fixed-width; `storage_bytes` reimplemented on top with `.expect`. Layer B (`executor.rs`): `checked_storage_bytes` helper → `SessionError::ShapeOverflow`; both `ensure_buffer` and JIT alloc routed through it; `.max(1)` after checked multiply. Layer C (`strided.rs::view_in_bounds`): address range computed in i128 with `checked_mul`/`checked_add`; overflow → `EpError::InvalidTensorView`. 4 new regression tests; all crate tests + bert_toy green; clippy clean; no new `unsafe`.
**Why:** Closes H-D1 end-to-end at all three layers identified by Holden's 🔴.

---

### 2026-07-14: Holden re-review — H-D1 fix (🟡 SHIP)
**By:** Holden (safety/soundness reviewer)
**Target:** `squad/ort2-hardening` @ `852f262` | **Verdict:** 🟡 SHIP-with-advisories (prior 🔴 cleared)
**What:** All three fix layers HELD. Layer A: `checked_storage_bytes` correct, regression tests pass. Layer B: both alloc sites checked; `.max(1)` after multiply. Layer C: i128 address math cannot itself overflow (inputs bounded by i64/usize; max value ~2^127 < i128::MAX). Original exploit vector (`[2^61]`×f64) → `ShapeOverflow` at allocation; regression test `bounds_reject_overflowing_address_math` confirms. Tests/clippy clean; Miri unavailable (component not installed). Residual advisories (non-blocking, memory-safe): A1 — `storage_bytes` panics (not graceful error) at capi:350, weights.rs:133, tensor.rs:from_raw_in (caught by catch_unwind; fast-follow owner: Leon); C1 — `addressed_elem_range` min/max accumulated in i64 before i128 widening (adversarial-strides only, not reachable via static shapes).
**Why:** H-D1 heap-OOB fully closed end-to-end. Residual advisories are fail-closed memory-safe nits.

---

### 2026-07-14: Chew review — ep-cpu hardening (🟢)
**By:** Chew (correctness/numerics reviewer)
**Target:** `squad/ort2-hardening` @ `830086e` | **Verdict:** 🟢 GREEN
**What:** All 6 fixes compute ONNX-correct results. Softmax dual semantics numerically verified for [2,2] axis-0 (legacy=[0.032,0.087,0.237,0.644] sums-to-1; per-axis=[0.119,0.119,0.881,0.881] column-wise). Min/Max NaN-propagating `combine()` correct. Cast `num_to_int!` converts directly to target type (no i64 wrap). slice_plan dedup byte-identical to prior inline. capi exhaustive match reasonable (caller-error vs internal-error arms). All tests pass; clippy clean. No lockout.

---

### 2026-07-14: Fact-check — EPContext node design docs/ORT2.md §55 (🟡 one required fix)
**By:** Fact Checker
**Target:** `squad/ort2-epcontext-design` @ `c48f5c4` | **Verdict:** 🟡 SHIP-with-one-required-fix
**What:** Op schema §55.2 — all 10 attributes exact vs contrib_defs.cc (names, types, defaults). Session-option key strings exact vs `onnxruntime_session_options_config_keys.h`. embed_mode semantics correct. main_context semantics correct. Model-agnostic dispatch verified (EpContextRegistry keyed by source string, no hardcoded EP names in dispatch path). ❌ Required fix: §21.4 `ep.context_embed_mode` default stated as `1`; ORT runtime default is `0` (`ep_context_options.cc:40`; header: "0: external file (default)"). Roy must change §21.4 default 1→0 and align `EpContextGenOptions.embed_mode` to `ExternalFile`. Do NOT change §55.2 op-attribute default (`1`) — that is correct. Advisory: TOC not updated for §55/§56 renumber (pre-existing mismatch). Roy not locked out — single-cell fix.
**Why:** Spec must match ORT runtime default to avoid silent wrong behavior when ep.context_embed_mode is unset.

---

### 2026-07-14: CUDA EP kernel stack decided (Phase 2)
**By:** Coordinator (Justin Chu directive; cuda-kernel-research agent, opus-4.8)
**What:** Phase-2 `onnx-runtime-ep-cuda` kernel stack: **Foundation** — `cudarc` (Rust CUDA 11.4–13.3 bindings; HuggingFace Candle GPU backend). **Standard GEMM** — cuBLASLt via `cudarc::cublaslt` (fused epilogue GEMM+Bias+Activation since CUDA 12.0). **Custom fused kernels** (LayerNorm/RMSNorm, RoPE, softmax, elementwise fusions) — CuTe (CUTLASS 3.x C++ templates) → `extern "C"` launchers compiled by `nvcc` in `build.rs`, linked as static lib; `#if __CUDA_ARCH__>=900` SM90 gate with SM80 fallback. **Attention** — Phase 2a: cuDNN fused SDPA via `cudarc::cudnn`; Phase 2b: FlashAttention-3 via `flash_attn_shim.cu` C shim around Tri Dao's `hopper/` csrc. **Trivial elementwise** — NVRTC PTX via `cudarc::nvrtc`. **cuTile DEFERRED to Phase 3**: no Hopper (SM90) support in CUDA 13.1 (Ampere/Ada/Blackwell only), Python-only, no Rust path. Re-evaluate when Hopper ships + C++ path lands (~CUDA 14.x / 2027). Rejected: Rust-CUDA (nightly-only, no CUTLASS), Triton AOT (Python dep), TileLang/Mojo (Python). All custom kernels MUST be shape-driven, dtype-parameterized, arch-gated — no hardcoded model constants. Next: Roy updates docs/ORT2.md §15 after EPContext-design branch merges.
**Why:** Research-grade evaluation with dated citations. cuTile disqualified for primary H100/H200 (SM90) target. cudarc+cuBLASLt+CuTe stack is production-proven.

---

### 2026-07-14: `onnx-runtime-shape-inference` — new crate landed
**By:** Roy (ORT2 architect) — `squad/ort2-shape-inference`
**What:** New pure-Rust crate implementing general symbolic shape inference over frozen `onnx-runtime-ir` Graph. 4 pillars: (1) extensible per-op registry keyed `(domain, op_type)` → version-sorted handlers; (2) `DimExpr` canonical integer polynomial (`BTreeMap<monomial, coeff>`) with checked_div for exact symbolic division; (3) shape-DATA propagation side-table tracking `Shape→Gather→Concat→Reshape` chains; (4) Strict/Permissive merge policies. 40+ op handlers including com.microsoft FusedMatMul, Conv/pool family, all transformer ops. `bert_toy_fully_resolves` (384-node graph, `num_unresolved()==0`). `#![forbid(unsafe_code)]`, 56 tests, clippy clean. Does NOT modify `onnx-runtime-ir`. IR-helper proposal deferred. Stopgaps (loader const-fold-lite, session JIT) left in place pending wiring.
**Why:** General extensible symbolic shape engine needed as foundation for planner/allocator/cost-model work; stopgaps are bounded hacks.

---

### 2026-07-14: Chew review — shape-inference op correctness (🔴 REJECT)
**By:** Chew (correctness reviewer) — `squad/ort2-shape-inference` (42 tests green at review time)
**Verdict:** 🔴 REJECT — one blocking defect: `com.microsoft::FusedMatMul` reused plain `matmul` handler, ignoring `transA`/`transB`/`transBatch` attrs. `A=[8,64]·B=[32,64]ᵀ` (transB=1) produced `[8,64]` + spurious contraction error (correct: `[8,32]`). Fix assigned to **Deckard** (Roy locked out as author). All other ops HELD correct: MatMul, Gemm, broadcast ops, all movement/norm/pool/data handlers. Non-blocking advisories: (a) Reduce opset-18 unresolved-axes degrades to reduce-all; (b) Concat/Cast shape-data dtype hard-coded Int64; (c) GatherElements doc comment misleading.
**Why:** FusedMatMul with transB=1 is pervasive in ORT-optimized transformer attention graphs; wrong shape silently corrupts every downstream op.

---

### 2026-07-14: Holden review — shape-inference DimExpr soundness (🔴 REJECT)
**By:** Holden (soundness reviewer) — `squad/ort2-shape-inference`
**Verdict:** 🔴 REJECT — one real soundness bug: `DimExpr` `add`/`sub`/`mul` used unchecked i64 arithmetic. Debug build panics on `2^80` product (e.g. Size of large tensor); release build wraps to bogus concrete dim. Secondary: `checked_div` `coeff % div_coeff` / `coeff / div_coeff` unguarded against `i64::MIN / -1` overflow-panic. Fix assigned to **Deckard**. All other items HELD: canonicalization uniqueness, checked_div correctness, fresh-symbol range safety (anon floor > 0x8000_0000), merge-policy soundness, ShapeData 1024-elem cap, `#![forbid(unsafe_code)]`, miri clean. Advisory: `fresh_symbol` counter unchecked `+=1` (adversarial exhaustion).
**Why:** i64 wrap can silently write a wrong concrete dimension without error; debug panic is reachable on normal large tensors.

---

### 2026-07-14: Gaff review — shape-inference registry/driver/API (🟢 APPROVE)
**By:** Gaff (graph-integrity / API-design) — `squad/ort2-shape-inference` (4 integrity probes)
**Verdict:** 🟢 APPROVE. Registry dispatch (domain normalization, opset boundary, duplicate-replace), topo-order driver (correct read-before-write, field-level write-back via `value_mut`, multi-output, transactional on failure — proven by probe), shape-data side-table integrity (per-call HashMap, no cross-call stale leakage), API design (`InferenceRegistry`/`infer_graph`/`infer_node`, `thiserror` errors, no panic in public paths) all HELD. IR contract NOT modified. `onnx-runtime-ir` diff is zero lines. Roy not locked out. Fix agent NOT required.
**Why:** Structural/API correctness is separate concern from per-op formulas (Chew) and algebra soundness (Holden).

---

### 2026-07-14: Deckard fix — shape-inference (FusedMatMul transpose + DimExpr overflow)
**By:** Deckard (Roy locked out per reviewer-protocol) — commit `09988f3` on `squad/ort2-shape-inference`
**What (blocking fix 1 — Chew 🔴):** New dedicated `fused_matmul` handler reads `transA`/`transB`/`transBatchA`/`transBatchB` (ORT's real A/B-suffixed names) and `alpha` (shape-neutral). `apply_fused_trans` reorders operands to plain `[batch…, row, col]` mirroring ORT `FusedMatMulShapeInference` in contrib_defs.cc; then calls shared `matmul_shape`. Rank-≤1 unchanged (matches ORT). 7 new FusedMatMul tests. **What (blocking fix 2 — Holden 🔴):** All `DimExpr` combiners use `checked_*`; overflow degrades to `DimExpr::overflow()` sentinel (poisoning, no alias via fresh-symbol bypass of intern cache). `checked_div` guards `i64::MIN/-1`. `ConstantOfShape` numel uses `checked_mul` fold. **Advisories applied:** `broadcast_dim` Permissive→fresh symbol; `fresh_symbol` saturating_add; Concat/Cast dtype from first operand; GatherElements doc corrected; Reduce opset-18 unresolved axes. **Result:** 69 tests green debug+release, clippy clean.
**Why:** Both 🔴 rejects fully addressed; advisory items applied opportunistically.

---

### 2026-07-14: Chew re-review — shape-inference FusedMatMul fix (🟢 SHIP)
**By:** Chew (correctness reviewer) — commit `09988f3`, fix author Deckard
**Verdict:** 🟢 SHIP — 🔴 blocker fully resolved. Dedicated `fused_matmul` handler verified line-for-line against ORT contrib_defs.cc: batch prefix (`trans_batch?1:0` .. `trans_batch?rank-1:rank-2`), row (`trans?rank-1:(trans_batch?0:rank-2)`), col (`trans?(trans_batch?0:rank-2):rank-1`), rank-≤1 unchanged. Cited case `[8,64]·[32,64]ᵀ → [8,32]` correct. 7 new FusedMatMul tests pass. All 3 advisories applied. **69 tests / 0 failed.** Roy and Deckard both locked out of this artifact for any future revision cycle.
**Why:** Required re-review after 🔴 reject; confirms fix matches ORT upstream source exactly.

---

### 2026-07-14: Holden re-review — shape-inference DimExpr overflow fix (🟢 GREEN)
**By:** Holden (soundness reviewer) — commit `09988f3`, fix author Deckard
**Verdict:** 🟢 GREEN — rejection fully addressed, no new soundness bug. Every combiner overflow-safe (add/sub/mul all `checked_*`). Overflow-degrade contract sound: `overflow()` sentinel has `overflow:true`, `as_const`/`as_symbol` both return `None`, poison propagates, **no symbol aliasing** (`SymbolInterner::lower` checks `is_overflow()` first → fresh symbol, bypasses equality cache). `checked_div` guards `i64::MIN/-1`. `ConstantOfShape` numel fold can't overflow first. `broadcast_dim` degrade-to-fresh correct. `fresh_symbol` saturating_add. **69 tests green debug AND release.** Non-blocking advisory: `movement.rs` slice-index normalization uses raw i64 on attribute-supplied indices (pre-existing, not a regression from `09988f3`; filed as follow-up).
**Why:** Required re-review after 🔴 reject; confirms no wrap, no debug panic, no bogus alias on new overflow path.

---

---

### 2026-07-14: ORT2 shape-inference wiring — loader owns static inference; const-fold-lite retired (Roy, roy-14)
**By:** Roy (ORT2 architect) — merged `98a3310`
**What:** Wired `onnx-runtime-shape-inference` into the loader. `build_from_bytes_with_weights` now calls `registry.infer_graph(...)` with `MergePolicy::Permissive` after graph build. Loader gained `LoaderError::ShapeInference(#[from] ...)`. Deleted `crates/onnx-runtime-loader/src/shape_inference.rs` (~1.1k LOC `const-fold-lite` `KnownVal`/`ConstEnv` pass) — no back-compat shim (pre-release). Session JIT (`dynamic_output_shapes`) retained as fallback for genuinely data-dependent extents. Fixed `broadcast_dim` in `context.rs` to keep the smaller `SymbolId` representative (not mint fresh) when two symbolic dims meet at a broadcast axis — fixes Expand-contamination regression where data-dependent `from_slice_01` symbols contaminated downstream values. Added `add_two_distinct_symbols_keeps_named_representative` test. `bert_toy` conformance: max_abs 1.192e-7 (unchanged).
**Why:** Loader now owns static shape inference as the architectural seam. const-fold-lite was strictly subsumed by the general crate.

---

### 2026-07-14: ORT2 shape-inference wiring — correctness/conformance review 🟢 (Chew, chew-17)
**By:** Chew (ORT2 correctness reviewer) — reviewed `f4141b9`
**Verdict:** 🟢 GREEN — SHIP
**What:** Broadcast-semantics change conformance-safe (smaller-id always prefers pre-existing session-bindable graph symbol due to `ANON_SYMBOL_FLOOR=0x8000_0000` invariant). const-fold-lite deletion safe (no persistent `env`; optimizer has independent pass). Declared-shape merge preserved. `bert_toy` conformance unchanged (1.192e-7). All op rules pass (52 tests). Advisories: A1 — `broadcast_dim` comment framing slightly misleading ("named" should be "lower-id graph"); A2 — `merge_shapes` both-symbolic keeps inferred over declared (pre-existing, harmless).
**Why:** Required review; confirms no conformance regression.

---

### 2026-07-14: ORT2 shape-inference wiring — soundness review 🟢 (Holden, holden-10)
**By:** Holden (ORT2 soundness reviewer) — reviewed `f4141b9`
**Verdict:** 🟢 GREEN
**What:** Symbol-unification sound (overflow sentinel blocks `as_symbol()`, so `broadcast_dim` representative arm unreachable for overflow exprs; deterministic/order-independent; single topo-pass so no convergence issue). Loader seam transactional (graph mutated only after full write-back; on error graph dropped, never half-mutated). JIT fallback byte-for-byte unchanged (only comment diff in executor.rs). No regressions to `view_in_bounds`/`checked_storage_bytes`/unsafe. Full ORT2 suite green debug+release (0 failures). Advisory: fail-fast coupling — a false-positive op-rule error under Permissive now blocks the load; Chew's op-formula pass should confirm none fire on BERT/opset-12.
**Why:** Required review; confirms soundness.

---

### 2026-07-14: ORT2 IR dtype hardening — ONNX numbering fix + float8/float4 + unknown handling (Deckard, deckard-11)
**By:** Deckard — merged `909f0a0`
**What:** Fixed two wrong `DataType` discriminants in `crates/onnx-runtime-ir/src/dtype.rs`: `Float8E5M2: 18→19`; `Uint4: 23→21`. Added `Float8E4M3FNUZ=18`, `Float8E5M2FNUZ=20`, `Float4E2M1=23` (sub-byte, `bit_size=4`, `is_float=true`, `checked_storage_bytes=count.div_ceil(2)`). All classifiers (`is_float`, `is_int`, `is_sub_byte`, `byte_size`, `bit_size`) and `from_onnx`/`to_onnx` updated. Loader attribute decode: hardened `TENSORS|SPARSE_TENSOR|SPARSE_TENSORS|TYPE_PROTOS` from silent `Ok(None)` to `Err(LoaderError::GraphBuild(...))`. Unknown-rank `Shape` gap documented (fixing requires `Value::shape: Option<Shape>` across frozen IR — deferred). Full ORT2 suite 243 tests green debug+release.
**Why:** Silent `Float4E2M1(23)→Uint4` corrupt-decode and `Uint4(21)→None` load failure are critical bugs for the Gemma quantized-model path.

---

### 2026-07-14: ORT2 IR dtype hardening — numbering correctness review 🟢 (Chew, chew-18)
**By:** Chew (ORT2 correctness reviewer) — reviewed `f965f0b`
**Verdict:** 🟢 APPROVE
**What:** Every discriminant independently verified against ONNX `TensorProto.DataType` spec (all 21 variants, rows 1–23 excl. COMPLEX). `to_onnx = self as i32` from `#[repr(u8)]` confirmed correct. `is_float` includes Float4E2M1 and all float8s; `is_int` excludes them. `byte_size=0`/`bit_size=4`/`checked_storage_bytes=div_ceil(2)` for Float4E2M1 correct. Round-trip and unknown-value tests comprehensive. Advisory: vendored `onnx.proto3` stops at INT4=22 — `FLOAT4E2M1=23` not in-repo; verified against upstream onnx/onnx instead (onnx#4728). Recommended follow-up: bump vendored proto. Owner: Roy/Batty/Leon (Deckard locked out).
**Why:** Required review; confirms numbering correct.

---

### 2026-07-14: ORT2 IR dtype hardening — soundness review 🟡 (Holden, holden-11)
**By:** Holden (ORT2 soundness reviewer) — reviewed `f965f0b`
**Verdict:** 🟡 APPROVE-WITH-FOLLOW-UP
**What:** Sub-byte Float4E2M1 routes through `div_ceil(2)` path (never unchecked multiply). Float8-FNUZ normal 1-byte path. `#![forbid(unsafe_code)]` intact. No new unsafe/unwrap/panic. Fail-closed attr hardening safe (GRAPH/GRAPHS/TENSOR/TypeProto still handled). Residual gap: `value-info` and `attribute-tensor` decode sites (`graph_builder.rs:232,241,357,365,374`) still use `.unwrap_or(Float32)` — silent mislabel for COMPLEX64/future dtypes. Deckard's PROGRESS.md claim "loader surfaces as LoaderError" is accurate only for initializer weights (weights.rs), not these sites. Required follow-up before complex/unmodeled-dtype milestone: make these sites return `Result`. Owner: Roy/Batty/Leon. ORT2 suite 300 tests green debug+release.
**Why:** Required review; net soundness improvement with non-blocking residual gap.

---

### 2026-07-14: Loader dtype-decode sites all fail closed (consolidate silent-Float32 fallbacks)
**By:** Leon (leon-10, opus)
**What:** Made every `DataType::from_onnx(raw) -> None` decode site in `onnx-runtime-loader` fail closed, closing the silent-wrong-type gap Deckard's weights-only hardening left behind. Two-part change:

1. **Fail-close consolidation.** Added `LoaderError::UnsupportedDataType { raw: i32, context: String }` (`crates/onnx-runtime-loader/src/lib.rs`), a generalized variant carrying the raw ONNX i32 plus a context string. Migrated the existing weight path to it and converted all remaining silent `.unwrap_or(DataType::Float32)` (and the map-key `.unwrap_or(DataType::Int64)`) decode sites to it via a new `decode_dtype(raw, || context)` helper in `graph_builder.rs`. Call sites changed:
   - `weights.rs` `resolve_initializer` — reworded to the new variant.
   - `graph_builder.rs` initializer value — `context: "initializer '{name}'"`.
   - `graph_builder.rs` `type_proto_to_dtype_shape` TensorType + SparseTensorType element type (value-info) — `context: "value-info '{name}'"`.
   - `graph_builder.rs` `convert_tensor` (Constant/attribute inline tensors) — `context: "attribute tensor '{name}'"`.
   - `graph_builder.rs` `convert_type_proto` TensorType + SparseTensorType element + MapType key.
   - Preserved intentional non-dtype defaults (untyped value-info, non-tensor container placeholders).

2. **Vendored proto bump (doc/consistency only).** `proto/onnx.proto3` `enum DataType` gained `FLOAT4E2M1 = 23`. No runtime behavior change.

**Tests added:** `unknown_value_info_dtype_is_load_error` and `unknown_attribute_tensor_dtype_is_load_error` in `tests/loader.rs`. All 15 loader tests + 40 ir tests green debug+release. bert_toy conformance max_abs 1.192e-7.

**Branch:** `squad/ort2-dtype-failclose` — merged `06a2423` → `a822a21`.

**Why:** Holden's finding: value-info and attribute-tensor sites still silently relabeled unmodeled dtypes as Float32 after Deckard's weights-only hardening. Failing closed consistently at every decode site guarantees clean contextual errors.

---

### 2026-07-14: Loader dtype fail-close — soundness review 🟢 (Holden, holden-12)
**By:** Holden (ORT2 soundness reviewer) — reviewed `a822a21`; verifying closure of own prior finding
**Verdict:** 🟢 GREEN — finding fully closed, no over-reach, no regressions.

**What:** Grepped entire loader crate. New `decode_dtype(raw, ctx) -> Result` helper routes all real-dtype decode sites. Every site confirmed fail-closed (initializer value, value-info TensorType/SparseTensorType elem, convert_tensor Constant/attribute inline tensors, convert_type_proto Tensor/SparseTensor elem + Map key). No surviving `unwrap_or(Float32)` on any real-dtype site. Intentional non-dtype defaults preserved (untyped value-info, non-tensor containers). `type_proto_to_dtype_shape`/`convert_type_proto`/`value_info_type` signature changes `-> Result<…>` with `?` propagation; transactional-on-failure preserved. Proto bump FLOAT4E2M1=23 unique, correct. Full debug+release ORT2 suite green (ir 40, loader 15+2 new tests, ep-cpu 101, optimizer 27, session 7+1+11, shape-inf 14+3+52, capi 4+9, ep-api 13+4). bert_toy conformance PASS max_abs 1.192e-7.

**Minor advisory (non-blocking):** present-but-UNDEFINED (elem_type=0) on value-info now rejected (correct fail-close for typed I/O); canonical "untyped" models omit the type field and still load.

---

### 2026-07-14: Optimizer fused ops emitted in `com.microsoft` contrib domain; ep-cpu dispatch keyed by (domain, op_type)
**By:** Batty (batty-12, opus)
**Branch:** `squad/ort2-fused-domain` (based on main `06a2423`) — merged to main `8cab9d2`

**What:** The optimizer fusion pass previously emitted fused ops in the reserved default ONNX domain (`""`/`ai.onnx`). Moved all optimizer-produced fused ops to `CONTRIB_DOMAIN = "com.microsoft"` and generalized ep-cpu kernel dispatch to key on `(domain, op_type)` via a new `OpRegistry::supports(op_type, domain)` method.

**Domain chosen — `com.microsoft`:** established ONNX-ecosystem contrib domain where FusedMatMul/LayerNormalization/SkipLayerNormalization/SimplifiedLayerNormalization contrib variants already live. Shape-inference crate already registered handlers there. Interoperable with ORT-exported models.

**Op map:** `LayerNormalization` — emitted by optimizer + runnable kernel; default-domain kernel/shape rule KEPT; `com.microsoft` bindings ADDED (additive). `FusedMatMulBias`/`FusedGemm` — no kernel exists in either domain; left without kernel (correct: kernel-less ops are rejected at placement).

**Files touched:** `onnx-runtime-optimizer/src/fusion.rs` (CONTRIB_DOMAIN const, domain set on fused nodes); `onnx-runtime-ep-api/src/registry.rs` (supports() + norm_domain in both lookup+supports); `onnx-runtime-ep-cpu/src/kernels/mod.rs` (com.microsoft LayerNorm registration); `onnx-runtime-ep-cpu/src/provider.rs` (gate via registry.supports); `onnx-runtime-shape-inference/src/handlers/norm.rs` (com.microsoft LayerNorm rule).

**Verify:** debug+release green for optimizer(27)/ep-cpu(102)/ep-api(17)/shape-inference(70)/session(19). bert_toy conformance max_abs 1.192e-7. clippy clean. `#![forbid(unsafe_code)]` intact.

**Why:** Non-standard fused ops in `ai.onnx` cause opset-validation collision and ambiguous dispatch. A private contrib domain provides unambiguous dispatch keying, and centralizing support decisions on the registry is model-agnostic and future-proof.

---

### 2026-07-14: Fused-op contrib domain — dispatch/registry soundness review 🟢 (Gaff, gaff-7)
**By:** Gaff (ORT2 registry/dispatch/API soundness) — reviewed `1e894de`
**Verdict:** 🟢 GREEN — dispatch set correct, normalization symmetric, no phantom kernel registration.

**What:** Provider gate now accepts exactly the set of registered `(op_type, domain)` pairs via `registry.supports`. Enumerated registry: default-domain registrations == PHASE1_OPS (1:1 verified); `len() == PHASE1_OPS.len() + 2` invariant holds (Softmax-v13 + com.microsoft/LayerNorm, no extras). `ai.onnx`→`""` normalization applied at top of both `lookup` and `supports` — symmetric. Contrib opset: no import → `effective_opset` falls back to `u64::MAX`; `lookup` filters `since_version <= MAX`, picks v1 — no panic. Dual-domain LayerNorm: distinct `OpKey`s (domain differs), distinct HashMap entries, no overwrite; additive only. FusedMatMulBias/FusedGemm have no kernel in either domain; `supports()` returns false — rejected at placement, not execution. `is_phase1_op` kept as `pub` API (harmless). Debug+release all green; bert_toy PASS max_abs 1.192e-7; clippy clean.

---

### 2026-07-14: ORT2 session `optimize` stage activated (opt-in) (Roy, roy-15)
**By:** Roy (ORT2 architect — session pipeline / loader=shape-inference / session=execute seam)
**Branch:** `squad/ort2-session-optimize` (based on `6f2e518`) — merged to main `5a2d527`

**What:** Wired `onnx-runtime-optimizer` into `onnx-runtime-session`'s `build()` pipeline as an explicit opt-in stage. Default behavior (`"optimization"="none"`) is byte-identical to before this change.

**Option surface:** Key `"optimization"` via `SessionBuilder.option(key, value)`.
- `"none"`/`"off"`/`"0"` → `OptimizationLevel::None` (**DEFAULT** — no-op, no optimizer call, no re-inference)
- `"basic"` → ConstantFolding + DeadNodeElimination (structure-preserving, no new op types)
- `"all"` → ConstantFolding + DeadNodeElimination + OpFusion (`optimizer::default_passes()`)
- Unknown keys → `SessionError::UnknownOption`; unknown values → `SessionError::InvalidOption`.

**Pipeline ordering:**
```
load (+ loader shape inference)
  → optimize_graph(level)          [skipped entirely when level == None]
  → add com.microsoft to opset_imports
  → re-run infer_graph(Permissive) [only when passes ran]
  → compile → allocate
```

**Conformance:**
- DEFAULT (opt-off): `bert_toy_conformance` unchanged — max_abs **1.192e-7**. Byte-identical.
- `basic` vs opt-off: max_abs **0.000e0** — byte-identical. Const-fold + DCE + re-inference inert.
- `basic` vs onnxruntime reference: max_abs **1.192e-7** (same as opt-off).

**Documented discrepancy — `all` path not yet executable:**
`OpFusion` is schema-unaware: `FusedMatMulBias`/`FusedGemm` have no CPU kernel; fused `com.microsoft::LayerNormalization` carries 5-input signature incompatible with CPU kernel's 2-3 input arity. Fails cleanly with `SessionError::UnsupportedOp { op_type: "FusedMatMulBias" }` before any numerics. Optimization stays opt-in / default-off. Tripwire test `full_optimization_fusion_path_is_not_yet_executable` asserts the failure and fires loudly when fusion becomes executable. **Follow-up to Batty:** schema-aware `OpFusion` + `FusedMatMulBias`/`FusedGemm` CPU kernels (or gate those patterns).

**Files touched:** `crates/onnx-runtime-session/Cargo.toml` (deps); `crates/onnx-runtime-session/src/lib.rs` (`OptimizationLevel` enum+parse, `SessionError` variants, `optimize_graph()`, `build()` rewrite, unit tests); `crates/onnx-runtime-optimizer/src/lib.rs` (re-export `CONTRIB_DOMAIN`); `crates/onnx-runtime-session/tests/bert_toy_optimized_parity.rs` (new); `docs/PROGRESS.md`.

**Validation:** 53 tests green debug+release (optimizer 26+1, session 12+1+2+11). clippy clean. `#![forbid(unsafe_code)]` intact.

---

### 2026-07-14: ORT2 session `optimize` stage — correctness/conformance review 🟢 (Chew, chew-19)
**By:** Chew (ORT2 correctness/conformance) — reviewed `c92a2f2`
**Verdict:** 🟢 APPROVE

**Scope:** `git diff 6f2e518...c92a2f2` — 7 files, +435/-12.

**Findings:**
1. **Default-off invariance** 🟢 — `optimize_graph()` returns `Ok(())` immediately when `level.passes()` is empty. No passes run, no `com.microsoft` opset import inserted, `infer_graph` NOT re-run. No unconditional second infer_graph. `bert_toy_conformance` unchanged: max_abs 1.192e-7 (debug+release).
2. **`basic` parity is real** 🟢 — `basic` vs opt-off: max_abs 0.000e0 (byte-identical). `basic` vs onnxruntime: max_abs 1.192e-7. Output shapes correct ([1,8,99], [1,2]). No rounding drift.
3. **Re-inference ordering sound** 🟢 — passes → opset import → re-infer on rewritten graph → `from_parts` consumes re-inferred graph. Compile/allocate see post-optimize shapes.
4. **`all`-path gating clean** 🟢 — fails with `UnsupportedOp { op_type: "FusedMatMulBias" }` BEFORE numerics. Tripwire non-tautological: `Ok(_) => panic!` fires the moment fusion becomes executable; `Err(UnsupportedOp{op_type})` asserts `op_type ∈ {FusedMatMulBias, FusedGemm}`. No tolerance widened.
5. **Suite/clippy/unsafe** 🟢 — full ORT2 suite green debug+release (all crates). clippy clean. No new `unsafe`.

**Non-blocking note:** `Err(other) => {}` arm in tripwire accepts any non-`UnsupportedOp` error without asserting fusion-relatedness. Does not mask silent wrong numerics (the `Ok` arm guards correctness). Suggest future tightening.

---

### 2026-07-14: `optimization="all"` fusion path made executable + parity-correct on bert_toy (batty-13)
**By:** Batty — `squad/ort2-fusion-executable` (base main `5a2d527`); merged as `e9bf155`

**What:** Turned the previously-deferred `"all"` optimization path from "not executable" into a byte-identical, parity-validated path on `bert_toy`. Three coordinated changes: schema-aware LayerNorm fusion, a `FusedMatMulBias` CPU kernel (+ shape rule), and flipping the tripwire test into a real parity assertion.

**Schema-aware LayerNorm fusion** (`onnx-runtime-optimizer/src/fusion.rs`): Added `RewriteKind {Structural, LayerNorm}` and `FusionPattern::layernorm()`. Emits `com.microsoft::LayerNormalization` with inputs `[X, Scale, B]` and attributes `axis` (from first ReduceMean axes attr) + `epsilon` (read as f32 from inline initializer via `read_scalar_f32`; falls back to 1e-5 if unreadable). `X = Sub operand ≠ rm1 output`; `Scale = Mul operand ≠ Div output`; `B = final-Add operand ≠ Mul output`. Order-independent disambiguation.

**`FusedMatMulBias` CPU kernel** (`ep-cpu/src/kernels/fused_matmul_bias.rs`): `MatMul(A,B) + bias` (broadcast add), reusing new shared `matmul::matmul_dense` + `add::broadcast_apply`. Registered `("FusedMatMulBias","com.microsoft",1)`.

**Shape rule** (`shape-inference/src/handlers/linalg.rs`): `fused_matmul_bias` output = MatMul(A,B) shape; registered under `com.microsoft`.

**Tripwire → real parity**: `full_optimization_fusion_path_is_not_yet_executable` → `full_optimization_fusion_path_matches_reference_and_default`; asserts `"all"` runs and matches opt-off and reference at existing tolerance (2e-3 atol/rtol — not loosened).

**Parity:** `opt=all` vs opt-off **0.0 (byte-identical)**; vs reference **1.192e-7**. Full suite green debug+release (optimizer 26, ep-cpu 105, shape-inference 70, session 26). Clippy clean. No new unsafe.

**Deferred:** `FusedGemm` (MatMul+Add+Relu) — no Relu-terminated fusion in bert_toy; remains graph-only with no kernel.

---

### 2026-07-14: Review — schema-aware LayerNorm fusion correctness (chew-20)
**By:** Chew (ORT2 correctness) — reviewed `squad/ort2-fusion-executable` @ `0f4811e`
**Verdict:** 🟡 approve with follow-ups (non-blocking)

**Verified correct:**
1. Operand disambiguation is order-independent/model-agnostic (X/Scale/B selected by excluding interior tensors, not by position). Not baked to bert_toy. ✅
2. Epsilon extraction robust for realistic f32 cases (`ConstantFolding` materializes `Constant` nodes to initializers before `OpFusion`). ✅
3. `opt=all` parity real: 1.192e-7 vs reference, 0.0 vs opt-off. Tripwire is a real assertion; no tolerance loosened. ✅
4. `fuses_layernorm_chain` unit test asserts `[X,Scale,B]` inputs + `axis=-1` + `epsilon≈1e-12` (values, not arity). ✅

**Follow-ups (non-blocking):**
- **F1** — axis silently defaults to `-1` when `axes` attribute is absent (opset-18 uses `axes` as input, not attribute). Non-last-axis LayerNorm at opset-18 would be silently wrong. Fix: read `axes` input initializer for opset-18 and validate contiguous-to-end; otherwise decline to fuse.
- **F2** — epsilon silently defaults to `1e-5` if eps operand is not a readable f32 constant (e.g. fp16/fp64 model). Fix: decline to fuse instead of guessing.
- **F3** — no positive data-flow guard (op-type sequence match without verifying interior data-flow). Pre-existing. Also `layernorm_node` returning `Err` hard-errors the whole pass via `?` rather than declining that one match.
- **F4** — nit: `vs_off` byte-identity is observed (0.0) but not asserted. Consider `assert_eq!(overall_vs_off, 0.0)`.

**Ownership:** F1/F2 first. Batty locked out (author); Roy/Deckard/Leon eligible.

---

### 2026-07-14: Review — FusedMatMulBias kernel, shape rule, registry, operand-order (gaff-8)
**By:** Gaff (ORT2 kernel/registry/dispatch) — reviewed `squad/ort2-fusion-executable` @ `0f4811e`
**Verdict:** 🟡 approve with required follow-up (non-blocking for bert_toy)

**Verified correct:**
1. FusedMatMulBias kernel numerics: `matmul_dense(A,B)` then `broadcast_apply(bias)` — full numpy batched/broadcast semantics. ✅
2. Standalone MatMul refactor (`matmul_dense` extraction) byte-for-byte identical to old body; no regression (0.0 vs opt-off). ✅
3. Shape rule consistent: delegates to `matmul_shape`, registered `("com.microsoft","FusedMatMulBias",1)`. ✅
4. Registry/dispatch correct: `OpKey::new("FusedMatMulBias","com.microsoft",1)` registered; `supports()` true; `FusedGemm` intentionally not registered. Domain/op/key consistent across fusion↔kernel↔shape rule. ✅
5. MatMul+Add operand-order generality ROBUST: first-seen ordering over `[MatMul, Add]` chain always yields `[A, B, bias]` regardless of whether Add is `Add(mm,bias)` or `Add(bias,mm)`. Not baked to bert_toy. ✅

**Gap (required follow-up):**
- **G1** — MatMul+Add fusion has no shape guard. An `Add` whose non-matmul operand expands the matmul output shape (more leading dims) fuses to a silent-wrong result: shape rule returns the matmul shape; `broadcast_apply` silently truncates the leading axis. Not exercised by bert_toy (standard bias-add / same-shape cases). Fix: narrow/guard `MatMul+Add → FusedMatMulBias` to decline when the Add's non-matmul operand would expand the matmul output shape. Fix owner: **Roy** or **Deckard** (Batty locked out).

**Suite:** ep-cpu 105, optimizer 26, shape-inference 70, session 26 + bert_toy conformance + opt-parity — 0 failed. Clippy clean. No new unsafe.
