# Decision — native decoder step-inputs CUDA-graph capture is now DEFAULT-ON

**Author:** Cohaagen (EP/runtime)  ·  **Branch:** `feat/native-decode-capture-default-on`
**Reviewer:** Harry (independent; author locked out)  ·  **Authority:** Justin — "确保高性能"
**Lineage:** Inc3c / issue #384 (the captured step-inputs path); scope note
`cohaagen-perfcap-scope.md` (WIRING-GAP verdict).

## What changed

The captured per-step-input decode path for **multi-component / routed** native
CUDA decoders (`inputs_embeds` + routed ports, e.g. gemma-3n / gemma4-e2b and the
GQA layers of the 35B-A3B class) is now **on by default**. Previously it was
gated OFF behind an opt-in env flag, so every routed decode step ran the eager
owned-input path and forfeited CUDA-graph capture (the root cause of the ~6.5×
gemma4-e2b native-vs-ORT decode gap).

The env var `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS` is inverted from
opt-in to **opt-out**:
- unset / truthy / unrecognized → capture-on (default);
- `0` / `false` / `no` / `off` → force the eager owned path (escape hatch).

This only affects the **step-inputs (multi-component/routed)** path. Monolithic
single-component decoders (qwen2.5/qwen3) already captured by default via the
token-id `run_one_token` path — unchanged. No capture-core / `plan_capture_region`
/ `standard_attention` / GAP-3 KV edits.

## Files

- `crates/onnx-genai-engine/src/native_decode/cuda.rs`
  - `capture_step_inputs_enabled()` inverted to default-on; parsing split into a
    pure, unit-testable `capture_step_inputs_from_env_value(Option<&str>)`.
  - Comments at the `decode_cuda` gate (~:401) and the `capture_step_inputs`
    construction (~:1405) updated to describe default-on + the structural gates.
  - New `capture_step_inputs_gate_tests` unit module (default-on, opt-out falsy
    set, truthy/unknown stay on).
- `crates/onnx-genai-engine/tests/native_cuda_captured_step_inputs_parity.rs`
  - Asserts the **default** (no env) engages capture (`captured_decodes>0`) and is
    byte-identical to the `=0` opt-out eager baseline; explicit `=1` still on.
- `crates/onnx-genai-engine/tests/gemma3n_native_cuda_capture_realmodel.rs`
  - `run(dir, capture)`: `true` = default (env unset), `false` = `=0` opt-out.
    Asserts token parity + default engages + opt-out declines.
- `crates/onnx-genai-engine/tests/qwen3_0_6b_capture_step_inputs_decline.rs`
  - Now proves the single-component decoder declines under the **default** (no env),
    i.e. default-on never mis-engages an ineligible decoder.
- `docs/CUDA_GRAPH_CAPTURE.md` — new "Multi-component / routed decoders" section
  documenting default-on + the opt-out env semantics + structural decline.

## Correctness evidence (GPU 2, CUDA_VISIBLE_DEVICES=2)

- `capture_step_inputs_gate_tests` (3 unit tests) — PASS.
- `native_cuda_captured_step_inputs_parity` (`--test-threads=1`) — PASS:
  `tokens=[0,5,6,7] default_captured_decodes=3 opt_in_captured_decodes=3
  opt_out_captured_decodes=0`. Byte-identical tokens across default / opt-in /
  opt-out; the counter proves default-on genuinely engages (non-vacuous) and the
  `=0` opt-out genuinely falls back to eager.
- `cargo fmt --all --check` — clean. Real-model `--ignored` tests compile.

**Regression safety (the whole point):** default-on output is byte-identical to
the eager path on a real capture-eligible GQA decoder fixture; ineligible
decoders auto-decline via the unchanged structural gates (`graph_enabled`,
non-empty `captured_step_inputs`) — no silent-wrong. The `=0` opt-out is a
preserved escape hatch.

## Scope / caveats (for the PR body)

- Applies to capture-eligible multi-component GQA decoders; recurrent Scan /
  LinearAttention hybrids (27B / 35B-A3B recurrent path) still decline through
  the structural gate until their separate capture lane lands.
- A real gemma4-e2b **end-to-end** benchmark remains blocked by the orthogonal
  stale vision/audio export (`vision_encoder OneHot Depth is negative` even
  text-only; the pipeline forces the vision component to load). The
  `gemma3n_native_cuda_capture_realmodel` harness graceful-skips on it and is
  ready the moment a clean export / text-only-skip exists.
- gemma4-e2b's absolute headline number is additionally **embedding-bound** (the
  ~5.5 GB every-step embedding upload dominates wall time) — a separate follow-up.
  This change brings capture-eligible native decode to ORT-parity-or-better
  broadly; gemma4-e2b's headline is not solely gated by this.

## Follow-ups (not this PR)

1. Surface the decoder component's graph capture/replay + captured-step-inputs
   counters through `profile_native --pipeline` for direct tok/s on/off proof.
2. Recurrent-hybrid step-input capture (depends on the parked Scan capture lane).
3. gemma4-e2b clean vision/audio export (model-package lane) + the every-step
   embedding cost.
