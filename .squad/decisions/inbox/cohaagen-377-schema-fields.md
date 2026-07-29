### 2026-07-29: #377 explicit metadata fields

**By:** Cohaagen

**What:**

New explicit inference-metadata fields that replace the remaining name-guessing
sites tracked on #377 (after #380 / #382). Every field describes GRAPH
STRUCTURE, never a model family. Mobius/export (Benny's workstream) must emit
exactly these names/types so the runtime never falls back to a name guess.

1. `pipeline.strategy.inner_embedding_output` — `Option<String>` (non-empty).
   - Replaces: `nested_autoregressive.rs::resolve_inner_embedding_output`, which
     picked the inner decoder's per-code embedding output by guessing "the sole
     output whose name does not contain `logits` and is not a present-KV port"
     (`to_ascii_lowercase().contains("logits")` + `is_present_output`).
   - Semantics: the inner (code-predictor) decoder output port whose value is
     threaded back into the next inner step's `inputs_embeds` seed. Declared on
     the `nested_autoregressive` strategy (top-level or composite stage),
     alongside `outer`/`inner`/`num_code_groups`. Absent ⇒ actionable ERROR
     naming `pipeline.strategy.inner_embedding_output`.

2. `model.io.static_cache` — `Option<StaticCacheIoSpec>`.
   New struct `StaticCacheIoSpec`:
   - `write_indices_input: String` (non-empty) — scatter write-position input
     (was hardcoded `"write_indices"`).
   - `kv_sequence_length_input: String` (non-empty) — non-pad KV sequence-length
     input (was hardcoded `"nonpad_kv_seqlen"`).
   - `key_cache_inputs: Vec<String>` / `value_cache_inputs: Vec<String>` —
     per-layer static K/V cache buffer inputs, positional per layer (were
     hardcoded `"key_cache.{i}"` / `"value_cache.{i}"`).
   - `key_cache_outputs: Vec<String>` / `value_cache_outputs: Vec<String>` —
     per-layer updated K/V cache outputs, positionally paired with the inputs
     (were hardcoded `"updated_key_cache.{i}"` / `"updated_value_cache.{i}"`).
   - Replaces: `decode/io.rs::detect_static_cache`, which selected the
     TensorScatter static-cache ABI by hardcoded port names (int vectors are
     shape-indistinguishable, so shape alone cannot disambiguate). When
     `model.io.static_cache` is present it is authoritative and name-agnostic;
     the four cache lists must be equal length and pair positionally. Declared
     but inconsistent (unequal lengths, missing ports) ⇒ ERROR naming the key.

3. (compatibility emission, no schema field) encoder prompt-input role.
   - Replaces: `compatibility.rs:1152` `encoder_input_field.ends_with("audio_features")`.
   - The role (`audio_features_input` vs `token_input`) is now taken directly
     from WHICH explicit genai-config field the exporter declared
     (`model.encoder.inputs.audio_features` vs `.input_ids`), captured in the
     existing match, never re-derived by string-matching the port name.

**Still name-based after this change (documented, needs contract before removal
— unchanged from #382's deferral, NOT regressed here):**
- Paged-KV bridge geometry (`engine/kv_bridge.rs`): key/value substring +
  `kv_layer_index` + `matching_past_input`. Made metadata-authoritative when
  `model.io.kv_inputs`/`kv_outputs` are declared; name matching remains only for
  the no-metadata path. CUDA-only, correctness-critical (mis-resolved port
  corrupts generation), no CPU fixture — full removal deferred with the paged
  contract.
- `decode_contract.rs` KV name convention (`kv_suffix`, `KvNamingConvention`):
  still consumed by the off-limits #99 speculative proposers; cannot be deleted
  from this workstream.

**Why:** Justin's #377 directive — ALL inference/pipeline metadata except io
SHAPE must be EXPLICIT and GENERAL; only io-shape may disambiguate. Name
guessing / historical-name fallback must be replaced by explicit metadata plus a
clear ERROR (naming the exact missing key) when the required metadata is absent.
These fields let the exporter state graph structure directly so the runtime
never interprets a graph port name.

---

**SHIPPED (2026-07-29, branch `squad/377-explicit-metadata`):** All three fields
above landed exactly as specified — `PipelineStrategy.inner_embedding_output:
Option<String>`, `ModelIoSpec.static_cache: Option<StaticCacheIoSpec>` (with
`write_indices_input`, `kv_sequence_length_input`, `key_cache_inputs`,
`value_cache_inputs`, `key_cache_outputs`, `value_cache_outputs`), and the
encoder-role emission change (no new schema field). Committed regenerated
`schema/inference_metadata.schema.json`. Benny/mobius: emit these names verbatim.
