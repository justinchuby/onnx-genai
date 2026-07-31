# Inc3c — real-model capture-step-inputs validation (issue #384)

**Author:** Mary (native-pipeline). **Date:** 2026-07-30 / measured-through
2026-07-31. **Branch:** `squad/inc3c-realmodel-capture-validation` (standalone
follow-up to the now-merged #533).

## Measured reproduction (2026-07-31, device 4, release, ORT 1.27.0 CUDA)

Concrete on-main measurement, banked fresh (two-length steady-state, `taskset -c
0-7`, `ONNX_GENAI_ORT_LIB` set → real ORT-CUDA, not CPU fallback):

| qwen3-0.6b decode path | tok/s | vs ORT |
|---|---|---|
| native-CUDA **captured** (`ONNX_GENAI_CUDA_GRAPH=1`) | **614.3** | 1.42× (beats) |
| native-CUDA **eager** (`ONNX_GENAI_CUDA_GRAPH=0`) | **206.5** | 0.48× (loses ~2×) |
| **ORT-CUDA** reference | **433.0** | 1.00× |

**Counter proof that qwen3-0.6b DECLINES the capture-step-inputs flag** (test
`qwen3_0_6b_capture_step_inputs_decline`, GREEN on device 4):
1. `Engine::from_pipeline_dir(qwen3-0.6b)` **refused**: *"metadata has no pipeline
   section"* — it is not a multi-component pipeline.
2. `Engine::from_dir` native-CUDA decode with
   `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS=1` →
   `NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES` = **0**. The flag path is
   pipeline-decoder-only; qwen3-0.6b captures via the *token-id* `run_one_token`
   mechanism (the 614/206 lever above), never the step-inputs path.

So the OFF/ON of *this flag* is meaningless for qwen3-0.6b (always 0/declines);
the table above is its single-graph CUDA-graph capture lever, a faithful
launch-overhead proxy. The captured path still **beats ORT-CUDA 1.42×** while
eager **loses ~2×** — the perf motivation stands.

## Question asked

Does the Inc3c captured step-inputs decode path
(`ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS`) actually ENGAGE and deliver
the captured ceiling on a **real** decoder's multi-component pipeline path — not
just the synthetic `tiny-gqa-embeds-cuda` fixture — with token output unchanged?

## Headline finding — the named target (qwen3-0.6b) is the wrong model class

**qwen3-0.6b cannot exercise the capture-step-inputs flag at all** (proven
concretely above). It is a **single-component** model: `inference_metadata.yaml`
declares `token_input: input_ids`, one `model.onnx`, no embedding/vision split.
It loads via `Engine::from_dir` (the single-graph token-id path), never
`from_pipeline_dir`. The capture-step-inputs path only fires for a
**multi-component `inputs_embeds`/routed** pipeline decoder
(`decode_cuda_captured_step_inputs`, gated in `native_decode/cuda.rs` on a
declared `inputs_embeds`/`Routed` port). qwen3-0.6b has no such port.

So the Part A numbers **612 captured / 220 eager / 443 ORT-CUDA** were measured
on qwen3-0.6b's **single-graph token-id path**, toggling the *general* CUDA-graph
capture (`ONNX_GENAI_CUDA_GRAPH`), as a faithful **proxy** for the per-step
kernel-launch overhead. They are NOT the capture-step-inputs flag path, and
qwen3-0.6b cannot produce an OFF/ON number for that flag. (qwen3-0.6b DOES
capture on its token-id path — it is a GQA-capacity-KV decoder — confirming the
launch-overhead lever is real; it just is not the multi-component seam.)

## Which real models *would* exercise the path — and their status

The capture-step-inputs path needs a real multi-component `inputs_embeds`
decoder. Only three such models are on disk; **all are blocked by independent,
non-Inc3c gaps**:

| Model | Multi-component `inputs_embeds` decoder? | KV → captures? | Status |
|---|---|---|---|
| **qwen3-0.6b-int4** | No — single-component `input_ids` | yes (token-id path) | wrong class; can't exercise the flag |
| **qwen3.5-0.8b-hybrid** | Yes (`embedding.onnx`+`text.onnx`, `inputs_embeds`) | hybrid lin-attn state | **loader-blocked**: `from_pipeline_dir` refuses on vision `Resize.smart_resize=1` (compatibility synthesis can't represent it; needs native `inference_metadata.json`). A text-only stripped copy is also refused (`model.type qwen3_5` *requires* the vision block). |
| **gemma-3n-e2b** | Yes (`embedding`→`decoder`, `inputs_embeds` + routed `per_layer_inputs`) | **yes — GroupQueryAttention, sliding_window 512 → `graph_enabled=true`** | Bool audio-mask (canonical fix in #540); then blocked on **valid vision inputs** for text-only decode |

**Gemma-3n is the closest to a real-model proof and structurally WOULD engage
capture** (its decoder is `GroupQueryAttention` capacity-aware KV — the exact
class that captures, same as the synthetic fixture and the real 35B-A3B target).
Two independent obstacles gated it:

1. **Bool audio-mask (fix delivered separately by #540).** `input_features_mask`
   is `Bool`; the pipeline value/cache clone path
   (`decode/values.rs::clone_value`) hard-errored `unsupported cached ORT value
   dtype: Bool`. During this validation I confirmed a general raw-byte fallback
   (`Value::to_raw_bytes` → `from_raw_bytes`, bit-exact) advances the gemma
   pipeline *past* the Bool error — but Harry's #540 is the **canonical, more
   complete** version of exactly that (both `clone_value` and `clone_owned`,
   host-guarded `as_raw_bytes` so device tensors error precisely instead of
   silent-corrupting, all POD dtypes, round-trip tests in both crates). This PR
   therefore does **not** carry a `clone_value` change; it relies on #540.
2. **Vision required-input (NOT fixed — the remaining gap).** gemma-3n declares
   `vision_encoder.pixel_values` as a *required* request input and the pipeline
   eagerly runs the vision encoder even for a text-only prompt with no image
   tokens. The vision pooler's `OneHot` derives its depth from real image-grid
   structure, so zeroed/synthetic patches are rejected (`Depth is negative`).
   Producing a real-model gemma number needs either a **real image** or an
   **optional-modality skip** (run the encoder only when the prompt carries image
   tokens) — an unrelated pipeline feature, out of Inc3c scope.

The forward-looking harness
`tests/gemma3n_native_cuda_capture_realmodel.rs` encodes the full real-model
proof (dummy audio+vision, native decoder `cuda:0`, counter engagement + OFF==ON
token parity) and **skips gracefully** on the vision-input gap (matching the
`qwen35_0_8b_hybrid` skip-with-explanation precedent). It becomes a live proof
the moment a real image is supplied or vision is made optional.

## Where the engagement proof currently stands

The **non-tautological engagement proof remains the synthetic
`tiny-gqa-embeds-cuda` fixture** (`native_cuda_captured_step_inputs_parity`,
GREEN on device 4): counter `NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES` = 0 with
the flag OFF, = 3 with the flag ON, tokens identical `[0,5,6,7]`. That fixture is
built specifically to mirror the real class — a `GroupQueryAttention`
capacity-aware `inputs_embeds` decoder — i.e. exactly what gemma-3n and 35B-A3B
use. So the proof faithfully models the real seam; what is missing is only an
end-to-end *real weights* run, blocked by the loader/modality gaps above, not by
the optimization.

## Default-on recommendation

- **Safety: already safe to default-on.** When `graph_enabled=false` (naive
  Concat-KV decoders) the capture-step-inputs gate is inert and the path is
  byte-identical to eager. When it engages (GQA-capacity `inputs_embeds`
  decoders) token output is proven identical (fixture). There is graceful eager
  fallback for every decliner. So flipping
  `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS` (and
  `ONNX_GENAI_PIPELINE_NATIVE_DECODER`) on by default carries **no correctness
  risk** for engaging models and **no behavior change** for declining ones.
- **Blocker to *recommending* default-on now:** we do not yet have a GREEN
  **real-weights** e2e capture number (fixture-only). That gap is caused by the
  two unrelated loader/modality blockers, not by the optimization. Recommend:
  **keep default-off until one real multi-component `inputs_embeds` model runs
  the flag e2e** (gemma-3n once vision is optional / imaged, or qwen3.5-hybrid
  once it has native metadata), then flip default-on with eager fallback. The
  perf case is already strong (captured beats ORT-CUDA 1.38×; eager loses ~2×).

## gemma-3n limitation: quick-win-or-slice verdict

- **Bool audio-mask clone:** **quick win** — a general raw-byte clone fallback;
  the canonical version lands in **#540** (`clone_value` + `clone_owned`,
  host-guarded, all POD dtypes). Confirmed it advances gemma-3n past the Bool
  error. This validation PR does not carry it (superseded by #540).
- **Vision-as-optional (text-only decode without image inputs):** **a slice** —
  requires the pipeline to conditionally skip a declared-required modality
  encoder when the prompt has no modality tokens (or accept a real image). Not
  Inc3c; recommend a dedicated follow-up if a gemma-3n real-model perf number is
  wanted.

## Artifacts (this follow-up, additive — #533's reviewed code untouched)

- `crates/onnx-genai-engine/tests/qwen3_0_6b_capture_step_inputs_decline.rs` —
  concrete counter proof that qwen3-0.6b (single-component) DECLINES the
  capture-step-inputs path (from_pipeline_dir refused + counter 0 with flag ON).
- `crates/onnx-genai-engine/tests/gemma3n_native_cuda_capture_realmodel.rs` —
  forward-looking real-model capture-engagement harness (graceful skip on the
  vision-input gap; relies on #540 for the Bool audio-mask clone).
- `crates/onnx-genai-engine/Cargo.toml` — registers the new `[[test]]`s.

The Bool `clone_value`/`clone_owned` fix is delivered by **#540** (Harry), not
here, to avoid a conflicting duplicate in `decode/values.rs`.

Verify: `cargo fmt --check` clean; clippy ×4 (default/native-backend/cuda/
cuda,native-backend) clean; full `cargo test -p onnx-genai-engine
--features cuda,native-backend --no-fail-fast` failing set **17, byte-identical
to base, 0 regressions**; synthetic engagement proof + qwen3-0.6b decline proof
GREEN.
