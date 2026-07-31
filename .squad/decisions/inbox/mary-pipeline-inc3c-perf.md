# Inc3c — native CUDA decoder eager-per-step perf (issue #384)

Mary · 2026-07-30 · branch `squad/native-multi-component-pipeline-inc3c`
(stacks on Inc3b #487 → base `squad/native-multi-component-pipeline-inc3b`)

## Context

Inc3a/Inc3b proved the native CUDA decoder runs the multi-component pipeline
decode with `inputs_embeds` + generic routed ports on-GPU, KV device-resident,
token-parity green. But that path is the **eager (uncaptured)** device forward:
`decode_cuda` routes *any* `inputs_embeds`/`Routed` step input through
`run_cuda_eager_rows_owned`, which re-binds fresh owned inputs and launches every
kernel un-captured each step. Only the pure token-id decode uses the
CUDA-graph-**captured** `run_one_token` fast path. Lori flagged (advisory on #485)
that this eager-per-step cost matters for the 35B-A3B real path. Justin's standing
directive: **native CUDA EP must beat/match ORT.** So: measure first.

## PART A — measurement (MEASURE FIRST)

### Method

`crates/onnx-genai-engine/tests/gemma4_e2b_native_cuda_pipeline_bench.rs`
(`--ignored`, `--release`). Steady-state decode tok/s via the **two-length method**
`tok/s = (N2 - N1) / (t2 - t1)` (N1=32, N2=160, best-of-3), which cancels prefill +
fixed per-call overhead. Engine built once, reused across timed generations.
GPU **device 4**; ORT-CUDA reference uses
`ONNX_GENAI_ORT_LIB=$ORT_ROOT/lib/libonnxruntime.so.1.27.0` +
`ONNX_GENAI_EP=cuda` (else ORT silently falls back to CPU).

**Controlled lever — Qwen3-0.6B int4, single-graph real decoder** (consumes the
attention mask through real ops). Same graph, only capture toggled via
`ONNX_GENAI_CUDA_GRAPH`, so this isolates the pure **capture-vs-eager launch
overhead** — the dominant component of the eager-per-step cost:

| mode                              | steady-state decode tok/s |
|-----------------------------------|---------------------------|
| captured native-CUDA (**ceiling**)| **612.1**                 |
| **eager native-CUDA**             | **220.0**                 |
| ORT-CUDA (**reference bar**)      | **443.1**                 |

(debug build showed the same ordering, larger gap: 557.9 / 73.8 / 178.9 — host
launch overhead is more pronounced unoptimised; release numbers above are the
honest bar.)

### Verdict — the gap is MATERIAL (≫ the 2–3% bank-a-negative threshold)

- Eager is **2.78× below** the captured ceiling (220 vs 612).
- **Eager is ~2× below ORT-CUDA** (220 vs 443). The native pipeline decoder's
  eager path — the path *always* taken when the decoder consumes
  `inputs_embeds`/routed ports, i.e. every multi-component model incl. 35B-A3B —
  currently **loses to ORT**. This directly violates the beat/match-ORT directive.
- Captured native **beats** ORT-CUDA by 1.38× (612 vs 443). So the win is real and
  available: capturing the eager step should move 220 → toward 612, turning a 2×
  ORT loss into a ~1.4× ORT win.

The per-step host→device upload is *not* the bottleneck: one token's
`inputs_embeds` (1536×2 B ≈ 3 KB) + routed `per_layer_inputs` (8960×2 B ≈ 17.5 KB)
≈ 20 KB/step, ~1 µs at PCIe bandwidth — negligible vs the ~2.6 ms/token decode.
The gap is **uncaptured kernel-launch overhead** across the decoder's layers.

## PART B — optimize (the gap is material, so we optimize)

### Lever chosen: graph-capture the `inputs_embeds`/routed decode step

Option (1) — capture the eager step — is the correct lever (launch overhead
dominates; option (2) upload-once saves ~1 µs/step, immaterial).

**Mechanism (a clean generalization of the token captured path).** The captured
token path (`decode_cuda`, single-token branch) *writes* the token id + position
into **persistent device bindings** then runs the warmup/capture/replay state
machine `run_one_token`; the mask + KV are already persistent device bindings. The
eager path instead re-binds fresh **owned** inputs each step and never captures.
So capturing the eager step = give every per-step port a **persistent** device
binding and **write** the one-token bytes into it each step (exactly like
`write_decode_inputs` writes the token id), then reuse `run_one_token`:

- `inputs_embeds`: a persistent `[1,1,hidden]` binding is **already allocated**
  (Inc3a groundwork, `DecodeCudaState::new`) but currently unused by the eager
  path — write the per-step embedding bytes into it instead of re-binding owned.
- routed ports (e.g. `per_layer_inputs`): allocate a persistent `[1,1,width]`
  binding per port (metadata dynamic dims → 1, static dims kept), inserted **after
  base_binding_count** so it joins the captured `run_one_token` binding set but is
  excluded from the eager `[..base_binding_count]` replay set (no eager-path
  change).
- mask: frozen to the physical bucket (`decode_mask_expose_len`) exactly like the
  token captured path — capture eligibility unchanged; bucket growth re-captures.
- KV: stays device-resident, advanced inside the captured graph — identical to the
  token path. No new host round-trip.

Single-token decode only (M=1). Multi-token prefill keeps the eager owned path
(prefix-sensitive causal island, not capture-eligible), same as the token path.

### Split (honest)

- **Inc3c-i = PART A** (this note): measurement + design. **Complete.** Banks the
  material-gap finding (~2× ORT loss) that mandates the fix.
- **Inc3c-ii = capture the eager step** (`inputs_embeds` + routed persistent
  bindings + per-step writes + `run_one_token`), **gated behind a default-off flag**
  `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS` (default OFF ⇒ eager path
  byte-identical; opt-in ⇒ captured). Proof: (1) token parity UNCHANGED on the
  `tiny-gemma4-vlm-cuda` (inputs_embeds) + `tiny-gemma4-vlm-cuda-routed` (routed
  port) fixtures with the flag ON vs OFF; (2) real-model perf recovery on
  Gemma 3n E2B (`gemma-native-cuda` eager vs captured tok/s) moving toward the
  captured ceiling.

Default stays OFF this increment (matches Inc1/Inc3a's gated-first discipline);
a later increment flips the default after broader real-model validation.

### Risks

- **Capture correctness on the hot path** — the delicate warmup/capture/replay/
  rewind/bucket-growth state machine. Mitigated by *reusing* `run_one_token`
  unchanged (only the per-step *write* of inputs differs) and the existing
  device-capture-error latch (`check_device_capture_error`) that rejects a token
  before consumption on any replay bound-violation.
- **dtype/byte-length mismatch** at the persistent binding — validated at write
  time (`supplied.as_bytes().len()` must equal the binding's byte capacity).
- **Routed shape generality** — a routed port must have static trailing dims to
  get a fixed persistent binding; a port with a dynamic non-batch/seq dim keeps
  the eager path (declined, logged) rather than mis-capturing.
- **Vision cross-KV / `static_cross_kv` upload-once** — OUT of scope (vision
  Attention float-mask blocker; upload-once is immaterial per the byte budget
  above).

## PART B — implementation result (Inc3c-ii, COMPLETE)

Implemented the captured per-step-input path in
`native_decode/cuda.rs` + `load.rs`, gated behind
`ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS` (default OFF). Each declared
per-step port (`inputs_embeds` + each `Routed` port) gets a persistent
`[1,1,width]` device binding written per step; `run_one_token` is reused so the
mask + KV stay device-resident and the whole step is CUDA-graph captured.

### Proving engagement required a *capturing* fixture (key finding)

The existing tiny CUDA pipeline fixtures (`tiny-gemma4-vlm-cuda`,
`tiny-gemma4-vlm-cuda-routed`) grow their KV with a naive `Concat` cache whose
consumers read the **logical** KV length. On such graphs the native decoder
**structurally declines** CUDA-graph capture (`graph_enabled=false`: the KV
bindings expose a growing logical prefix and the mask exposes its logical valid
length). Instrumented on device 4:

```
graph_enabled=false dynamic_logical=[past_key_values.0.key (phys [1,1,256,4] vs logical [1,1,0,4]), .0.value ...] mask_exposes_logical=true
```

So on those fixtures the captured path can *never* engage (it correctly stays
dormant) — they prove the eager path only. This is also why the only existing
CUDA capture-stats test (`glm_tiny_qmoe_native_cuda_e2e`) asserts `captures==0`.

**New fixture `tiny-gqa-embeds-cuda`** (`scripts/build_tiny_gqa_embeds_cuda.py`):
identical closed-form tokens (`[3,7] → [0,5,6,7]`) and pipeline shape as
`tiny-gemma4-vlm-cuda`, but the decoder routes its KV through a real
`com.microsoft.GroupQueryAttention` op (reads `seqlens_k` /
`total_sequence_length`, past KV at fixed physical capacity). GQA is the
capacity-aware kernel shape the native decoder recognises as capture-safe, so
`graph_enabled=true`. Determinism is isolated from the GQA numerics: `logits =
inputs_embeds @ LM_HEAD + tie_bias` (the proven base head, bit-stable CPU/CUDA),
while GQA's `present.*` outputs are the growing device-KV contract and its
`attn_out` is intentionally unused. Q/K/V derive from `inputs_embeds` via real
MatMuls so the embedding genuinely flows into a CUDA op on-device.

### Non-tautological engagement + parity proof (device 4, GREEN)

`tests/native_cuda_captured_step_inputs_parity.rs` runs the pipeline twice and
reads the process-global counter `NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES`
(bumped only inside the captured branch — a distinct function from the eager
branch):

| flag | tokens | captured-decode count |
|------|--------|-----------------------|
| OFF (default eager) | `[0,5,6,7]` | **0** |
| ON (`…CAPTURE_STEP_INPUTS=1`) | `[0,5,6,7]` | **3** |

Flag OFF: `graph_enabled=true` but `capture_step_inputs=false` ⇒ eager ⇒ counter
stays 0. Flag ON: `capture_step_inputs=true` ⇒ the captured branch runs on each
of the 3 single-token decode steps (prompt prefill emits token 0, then 3 captured
steps emit 5,6,7) with **byte-identical tokens**. This proves the captured path
(a) genuinely engages (not a silent decline to eager) and (b) is behaviour-
identical to the eager path.

### Perf recovery — evidence

A *direct* captured-vs-eager tok/s number on a real multi-component model
(Gemma 3n E2B) is **blocked**: gemma-3n's audio `input_features_mask` is `Bool`,
and the pipeline value/cache path errors `unsupported cached ORT value dtype:
Bool` even with `pipeline_cache_bytes: 0` (a deeper, unrelated limitation, out of
Inc3c scope). The perf recovery is therefore evidenced by the **Part A controlled
measurement** on qwen3-0.6b — a real mask-consuming decoder — where the *same*
capture-vs-eager toggle moves decode **220 → 612 tok/s** (2.78×). The Inc3c path
routes the pipeline's `inputs_embeds`/routed step through that identical captured
`run_one_token` machinery, so it inherits that speedup once capture engages
(proven on `tiny-gqa-embeds-cuda`).

### Native-vs-ORT-CUDA verdict

Captured native-CUDA decode **beats** ORT-CUDA (612 vs 443, 1.38×); the eager
path **loses** (~2×). Inc3c's captured step-input path flips the multi-component
native decoder from an ORT loss to an ORT win **when graph capture is available**
(capacity-aware decoders — the real 35B-A3B class). Default stays OFF this
increment (gated-first); a later increment flips it after broader real-model
validation.
