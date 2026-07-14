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

