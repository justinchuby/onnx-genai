# Auto-derive `model.io` (kv + recurrent state_pairs) from graph ports

Date: 2026-08-01
By: Cohaagen
Lane: #384 recurrent-state / native decode loader
Status: PR opened (S increment)

## What this unlocks

Stock qwen3.x **linear-attention hybrid** exports (16 dense GQA layers + 48
recurrent `conv_state`/`recurrent_state` layers with a `Scan` body), including
Justin's named **qwen3.6-27b-int4**, now load + decode **natively** with **no
hand-written `io:` overlay** — the loader derives the decoder I/O topology from
the ONNX graph ports.

Before this, the stock export (whose `inference_metadata.yaml` carries no `io`
block) failed native load at token-input resolution:

```
cannot resolve model.io.token_input from tensor shape because 3 ports match:
[input_ids, attention_mask, position_ids]
```

The recurrent hybrids also declare NO dense KV for the linear layers, so shape
inference cannot classify their `conv_state`/`recurrent_state` running-state
ports. During the #384 re-probe I proved a hand overlay (states in
`io.state_pairs`) makes it load + decode byte-identically to the CPU fp32
oracle. This PR removes the overlay requirement.

## The change (blast radius: loader fallback + genai-config helper + tests + docs)

1. **genai-config** (`compatibility.rs`): extracted the pure, config-free core of
   `strict_decoder_state` into `decoder_state_from_patterns(graph, past_key,
   past_value, present_key, present_value, shape_match)`. Added a new public
   `GenAiConfig::derive_decoder_io_from_graph(graph) -> Option<DerivedDecoderIo>`
   that calls it with the conventional onnxruntime-genai patterns
   (`past_key_values.%d.key`/`.value` → `present.%d.key`/`.value`). This is TRUE
   reuse of the guarded derivation — recurrent ports are found by
   `suffix_tensor_map` (never `strict_indexed_kv`), so cross-attention/Whisper KV
   is never misclassified as running state.

2. **One relaxation, isolated to the fallback:** a `StateShapeMatch` param.
   `strict_decoder_state` passes `Exact` (byte-for-byte unchanged). The new
   fallback passes `AllowSymbolic`: a `present.*` state port whose exported shape
   is fully symbolic (`[?,?,?]`) is accepted against its concrete `past_*` input
   (`[?,10240,3]`). This was REQUIRED — the real 27b export leaves present
   recurrent-state shapes fully symbolic, and `Exact` rejected them. A symbolic
   axis means "unknown", not "different"; concrete-vs-concrete mismatch still
   fails.

3. **Engine loader** (`native_decode/load.rs`): in
   `from_session_with_cuda_options_and_io`, when `io.is_none()`, build a
   `ModelGraphInfo` from the session ports and call the helper. **Safety gate:**
   engage the derived spec ONLY when it yields ≥1 recurrent state pair (the case
   shape inference can't handle). Pure-dense decoders derive 0 state pairs → the
   fallback declines → their existing `io=None` path is untouched. Declared `io`
   always wins (fallback runs only when `io` is absent).

## Correctness bar

- **GPU e2e** (`native_autoderive_io_cuda_e2e.rs`, `#[ignore]`): STOCK 27b
  metadata (no overlay) → native CUDA load + decode == native CPU fp32 oracle,
  **byte-identical** token IDs. Proven parity token IDs:
  `[11751, 13, 271, 248068, 271, 248069, 271, 4639, 369, 4252, 13, 11751, ...]`.
- **Unit tests** (genai-config): 27b port layout → 32 kv entries + 96 state
  pairs; qwen3-0.6b dense layout → dense kv + **0** state pairs (gate doesn't
  over-derive). Non-vacuous both directions. The 27b fixture uses fully-symbolic
  present-state shapes to lock the `AllowSymbolic` path.
- **No-regression** (engine): a hybrid decoder with `io=None` now auto-derives
  and loads; a dense ambiguous decoder with `io=None` still fails with the same
  `model.io.token_input` error (fallback declines). Existing declared-io tests
  unchanged and green.

## Honest caveats

- **Eager only, ~5.91 tok/s.** The `Scan` control-flow body declines CUDA-graph
  capture, so these hybrids run eager. Capturing the Scan body is a separate,
  parked perf lane; this PR is a **correctness** unblock, not a perf win.
- **CPU is the oracle, not ORT.** ORT-CUDA crashes on this model class (internal
  `stl_vector` assertion), so there is no ORT reference to compare against; the
  native CPU fp32 path (which already threads recurrent state) is the oracle.
- Follow-up (separate): is the ORT-CUDA crash a mobius export bug or an ORT
  limitation? Affects whether we ever get an ORT baseline for this class.
