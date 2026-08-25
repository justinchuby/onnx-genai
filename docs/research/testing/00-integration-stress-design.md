# Integration stress test design

**Author:** Roy (Lead)  
**Date:** 2026-07-27  
**Scope:** Real-model, multi-turn integration stress for `onnx-genai` CLI/engine backends.

## Invariant catalogue

Exact text is not the contract. A stochastic model, different EP kernels, and native-vs-ORT decoding can all produce different tokens and still be correct. The stress layer should assert invariants that must hold for every backend, seed, and sampled path.

| Invariant | Assertion | Why it matters |
|---|---|---|
| Termination | Every generation returns `Finished`, `Stopped`, `Length`, or a classified resource error before a per-token/turn timeout. No unbounded decode loop. | Would turn repeating `<think>` / thinking loops into a deterministic failure without expecting exact text. |
| Non-empty committed turns | A completed assistant turn is committed only if it has visible answer text or an explicitly recorded non-answer outcome. Empty answer text is dropped with a diagnostic. | Prevents context poisoning after reasoning-only truncation or context exhaustion. |
| Reasoning progress | If the template opens a reasoning span, generated text must either close it, hit an explicit max-token/resource stop with the exchange dropped, or trip a repetition/progress guard. The same reasoning delimiter/body window cannot repeat beyond a threshold. | Catches the DeepSeek native CUDA repeating-thinking bug as a missing progress invariant. |
| Repetition bound | For each turn, fail on excessive repeated n-grams, repeated special delimiters, or near-identical fixed-size token windows after warmup. Thresholds are model-class-specific and reported with the repeated span. | Detects degenerate loops while allowing ordinary stochastic repetition. |
| History/token consistency | After every turn: recorded message count, prompt tokens, generated tokens, total positions, and rendered chat-template tokenization agree with the stats/profile JSON. | Finds silent drift between CLI history and engine request state. |
| Context/KV consistency | Logical positions, KV valid length, page allocation, and backend-reported cache state agree. At context pressure, either eviction/rewind happens as requested or a clear resource stop occurs; never "success with inconsistent KV." | Targets the bugs that only appear when long sessions reach real KV limits. |
| Admission liveness | A session admitted for turn `N` can either run turn `N+1` or fail with a classified, recoverable resource reason. It must not permanently wedge future requests after one refusal. | Catches scheduler refusal after several turns and verifies recovery/reset behavior. |
| Sampling observability | Greedy mode is stable for a seed. Stochastic mode with different temperatures/top-p/top-k produces a measurably different token distribution over repeated runs; invalid flags are reflected in session/stats output. | Prevents silently ignored sampling flags without asserting any one sampled answer. |
| Feature state coherence | Prefix-cache hits, speculative accept/reject counts, and rewind/fork positions are monotonic/coherent and match externally visible turn outcomes. | Covers agent-first mechanisms whose correctness is stateful, not single-output. |
| Reproducible failure packet | Every failed invariant writes seed, turn index, prompt, backend, EP, model id/path, ORT library/version, feature flags, and per-turn stats. | A 40-turn failure must be debugged from artifacts, not rerun blindly. |

## Matrix to sweep without exploding cost

Use pairwise/default sweeps, not a full Cartesian product. Each scenario chooses the smallest model/fixture tier that exercises the mechanism.

| Axis | Values | Policy |
|---|---|---|
| Backend | `ort`, `native`, `auto` where meaningful | Per-PR runs CPU ORT/native where available. CUDA native/ORT is self-hosted/manual. |
| EP | CPU, CUDA, Metal | CPU in GitHub CI. CUDA and Metal require local/self-hosted hardware. Metal joins once the EP is stable enough to load these fixtures. |
| Model class | Plain LLM, reasoning LLM, MoE/QMoE, VLM | Tiny fixtures for CI; env-var real models for nightly/manual confirmation. |
| Turn count | Smoke 3-5, stress 20-50, soak 100+ | PR smoke only; nightly stress; manual soak. |
| Context pressure | Normal, near-limit, deliberate exhaustion | Always include at least one tiny fixture with low context so exhaustion is cheap. |
| Sampling | Greedy fixed seed, stochastic seeded sweep | PR checks flag wiring with tiny model; nightly repeats distribution checks. |
| Features | Prefix caching, speculative decoding, rewind/fork, stats/profile JSON | Test each as an invariant overlay on one base conversation, not as separate full matrix rows. |

Recommended scenario names:

1. `ci_tiny_reasoning_pressure_cpu_ort`: **new fixture to create**, not currently on disk: a tiny reasoning LLM with `<think>` behavior, ORT CPU, 8-12 turns, low context, greedy + one stochastic seed.
2. `ci_tiny_plain_multiturn_cpu_ort`: `tests/fixtures/tiny-llm` (`model.onnx.textproto`, tokenizer, manifest), 8-12 turns, `/stats` and history consistency. Existing `crates/onnx-genai-cli/tests/repl_e2e.rs` already uses this fixture.
3. `ci_tiny_plain_multiturn_native`: use an exact native fixture, not a wildcard: `tiny-native-engine` / `tiny-native-sub4-engine` / `tiny-native-cuda-engine` are textproto fixtures; `tiny-native-scalar-gqa` is serialized `model.onnx`. Pick the one the engine-level harness can load for the feature under test.
4. `ci_tiny_vlm_prefix_cpu_ort`: `tests/fixtures/tiny-vlm-image-input` (`decoder.onnx.textproto` + `encoder.onnx.textproto`), repeated image follow-up, encoder/prefix cache invariants. Existing `crates/onnx-genai-cli/tests/repl_e2e.rs` already exercises this shape.
5. `nightly_qmoe_scheduler_pressure`: committed `tests/fixtures/tiny-glm52-qmoe-indexshare` (`model.onnx` + external data) or `GLM_TINY_QMOE_E2E_DIR`, multi-turn plus small KV budget.
6. `nightly_real_reasoning_native_cuda`: `ONNX_GENAI_QWEN3_0_6B_FOUNDRY_DIR` and `QWEN3_0_6B_CUDA_E2E_DIR` are confirmed native-CUDA test env vars today; they are not CPU ORT fixtures. Add a CPU/ORT real-reasoning env var only when such a test exists.
7. `self_hosted_cuda_deepseek_reasoning`: DeepSeek R1 Distill Qwen 1.5B on native CUDA and ORT CUDA, reproducing Justin's two observed failures. No committed fixture or env-var test exists for this exact model today.

## Model fixture tiers

This inventory was checked against `tests/fixtures/` and the crate tree on 2026-07-27. There are no crate-local `fixtures/` directories outside `tests/fixtures/`; crate-level tests consume the repository fixtures and env-var model directories.

| Tier | Fixtures / env vars | Confirmed format | Use |
|---|---|---|---|
| Tier 0 Phase-1-ready text | `tests/fixtures/tiny-llm` | `model.onnx.textproto`, `tokenizer.json`, `manifest.json`; **no `model.onnx`** | Per-PR CLI REPL/history/stats invariant plumbing. Existing `crates/onnx-genai-cli/tests/repl_e2e.rs` uses it, so it is usable by that harness, but implementers must not assume serialized ONNX. |
| Tier 0 Phase-1-ready VLM | `tests/fixtures/tiny-vlm-image-input` | `decoder.onnx.textproto`, `encoder.onnx.textproto`, `inference_metadata.yaml`, `tokenizer.json` | Per-PR repeated-image/encoder-cache REPL stress. Existing `crates/onnx-genai-cli/tests/repl_e2e.rs` uses it. |
| Tier 0 serialized ONNX text/state | `tiny-llm-scatter`, `tiny-llm-sharedbuffer` | `tiny-llm-scatter` has both `model.onnx` and `model.onnx.textproto`; `tiny-llm-sharedbuffer` has `model.onnx` + `model.onnx.data` | Use when a proposed stress harness needs serialized ONNX instead of textproto, especially scatter/shared-buffer state. |
| Tier 0 native-backend fixtures | Exact names: `tiny-native-engine`, `tiny-native-cuda-engine`, `tiny-native-sub4-engine`, `tiny-native-scalar-gqa` | First three are `model.onnx.textproto`; `tiny-native-scalar-gqa` is `model.onnx` + metadata | Native engine stress candidates. Do not cite `tiny-native-*` as if all have the same format. |
| Tier 0.5 missing high-value fixture | Tiny reasoning LLM | **Does not exist today** | Work to create first: emits/handles `<think>`, can close or fail to close reasoning, and has deliberately small context. |
| Tier 1 committed specialized | `tiny-glm52-qmoe-indexshare`, `tiny-gemma4-vlm`, `tiny-mtp-full`, `tiny-eagle3`, `tiny-qwen35-mtp`, `tiny-multiaxis-state-decoder` | Mixed: QMoE is serialized `model.onnx` + data; most others are textproto graphs plus tokenizer/manifest/metadata | Slow/nightly fixtures for MoE/QMoE, VLM, speculative/MTP/Eagle, and stateful decoder behavior. |
| Tier 1 other modality fixtures | `tiny-codec`, `tiny-diffusion`, `tiny-dit-diffusion`, `tiny-masked-diffusion`, `tiny-tts*`, `tiny-txt2img`, `tiny-vlm-multibinding`, `tiny-whisper*` | Mostly `.onnx.textproto` component graphs plus `inference_metadata.yaml`; some include tokenizer/audio/run fixtures | Useful for future pipeline/modality stress, but not Phase 1 text/reasoning stress. |
| Tier 2 confirmed env-gated real/native tests | `QWEN3_0_6B_CUDA_E2E_DIR`, `ONNX_GENAI_QWEN3_0_6B_FOUNDRY_DIR`, `GLM_TINY_QMOE_E2E_DIR` | Test code requires `model.onnx`, metadata, tokenizer, and for Qwen/GLM often external data; Qwen env vars are under `#[cfg(all(feature = "cuda", feature = "native-backend"))]` | Nightly/self-hosted when provisioned. Current tests skip when absent; a CI job that promises one must invert that into a loud provisioning failure. |
| Tier 3 missing manual repro fixture | DeepSeek R1 Distill Qwen 1.5B | **Not present as a committed fixture or named env-var test** | Manual/self-hosted CUDA/ORT repro for Justin's exact failures. |

### Confirmed fixture inventory

Convention (2026-08-14): committed inline-weight ONNX fixtures loaded through our
own loader (`onnx_runtime_loader`, which auto-detects TextFormat via
`is_textproto_path`) are stored as git-friendly `model.onnx.textproto`. Fixtures
are kept as binary `model.onnx` only when they carry external weights
(`model.onnx.data`), are executed by real ONNX Runtime (the `ort` C API cannot
parse TextFormat), or are byte placeholders. The ~28
`crates/onnx-runtime-ep-cpu-plugin/tests/fixtures/*` EP-conformance fixtures are
now textproto (the plugin test harness converts them in-memory to binary and
loads via `CreateSessionFromArray`).

| Fixture | Format on disk | Exercises | Usable by proposed stress? |
|---|---|---|---|
| `tiny-codec` | `encoder.onnx.textproto`, `vocoder.onnx.textproto` | Audio codec pipeline | Later modality/pipeline stress; not Phase 1. |
| `tiny-deepseek-v2-qmoe-attention` | `model.onnx.textproto`, tokenizer, metadata, manifest, generator | DeepSeek-V2 tiny QMoE + Attention native decode | Native QMoE/attention golden-lock; already used by `deepseek_v2_tiny_qmoe_native_e2e.rs`. |
| `tiny-diffusion` | textproto denoiser/VAE graphs | Diffusion pipeline | Later modality stress. |
| `tiny-dit-diffusion` | textproto denoiser | DiT diffusion | Later modality stress. |
| `tiny-eagle3` | `model.onnx.textproto`, manifest | Eagle/speculative-style model fixture | Slow speculative stress candidate. |
| `tiny-gemma4-assistant` | `model.onnx.textproto`, tokenizer, manifest | Gemma-style assistant decoder | Slow text/model-family coverage. |
| `tiny-gemma4-assistant-mixed` | `model.onnx.textproto`, tokenizer, manifest | Mixed Gemma assistant variant | Slow text/model-family coverage. |
| `tiny-gemma4-vlm` | textproto vision/embedding/decoder graphs, tokenizer, metadata | VLM pipeline | Slow VLM stress candidate. |
| `tiny-glm52-qmoe-indexshare` | `model.onnx`, `model.onnx.data`, tokenizer, metadata, generator | GLM 5.2 QMoE + IndexShare | Yes for scheduler/QMoE stress; already used by engine tests. |
| `tiny-llm` | `model.onnx.textproto`, tokenizer, manifest | Plain tiny LLM | Yes for Phase 1 REPL; no serialized `model.onnx`. |
| `tiny-llm-explicit-io` | `model.onnx.textproto`, tokenizer, metadata | Explicit I/O LLM metadata | Engine/metadata stress candidate. |
| `tiny-llm-scatter` | `model.onnx`, `model.onnx.textproto`, tokenizer, manifest | Scatter/static-cache LLM variant | Yes when serialized ONNX is required. |
| `tiny-llm-sharedbuffer` | `model.onnx`, `model.onnx.data`, tokenizer, manifest | Shared-buffer LLM state | Yes for shared-buffer/KV-adjacent stress. |
| `tiny-masked-diffusion` | `lm.onnx.textproto`, metadata | Masked diffusion | Later modality stress. |
| `tiny-mtp-full` | `model.onnx.textproto`, tokenizer, manifest, embeddings | MTP | Slow speculative/MTP stress candidate. |
| `tiny-multiaxis-state-decoder` | textproto decoder/embedding, tokenizer, metadata | Multi-axis state decoder | Engine state stress candidate. |
| `tiny-native-cuda-engine` | `model.onnx.textproto`, tokenizer | Native CUDA-oriented engine fixture | Self-hosted/native feature candidate; textproto format. |
| `tiny-native-engine` | `model.onnx.textproto`, tokenizer | Native engine fixture | Native feature candidate; textproto format. |
| `tiny-native-scalar-gqa` | `model.onnx`, tokenizer, metadata | Native scalar GQA | Native GQA/KV stress candidate with serialized ONNX. |
| `tiny-native-sub4-engine` | `model.onnx.textproto`, tokenizer | Native sub-4-bit fixture | Native quant stress candidate; textproto format. |
| `tiny-qwen35-mtp` | `model.onnx.textproto`, manifest | Qwen MTP | Slow speculative/model-family stress candidate. |
| `tiny-tts` | textproto decoder/vocoder, tokenizer, metadata | TTS | Later audio/pipeline stress. |
| `tiny-tts-nested` | textproto nested TTS graphs, tokenizer, metadata | Nested autoregressive TTS | Later nested-pipeline stress. |
| `tiny-tts-nested-preembed` | textproto nested TTS/pre-embed graphs, tokenizer, metadata | Nested TTS pre-embedding | Later nested-pipeline stress. |
| `tiny-tts-nested-prefill` | textproto nested TTS/prefill graphs, tokenizer, metadata | Nested TTS prefill | Later nested-pipeline stress. |
| `tiny-txt2img` | textproto text encoder/denoiser/VAE, tokenizer, metadata | Text-to-image | Later modality stress. |
| `tiny-vlm-image-input` | textproto encoder/decoder, tokenizer, metadata | VLM image input | Yes for Phase 1 VLM/prefix stress. |
| `tiny-vlm-multibinding` | textproto embedding/decoder, tokenizer, metadata | VLM multi-binding | Slow VLM binding stress candidate. |
| `tiny-whisper` | textproto encoder/decoder, tokenizer, `tiny.wav`, metadata | Whisper/audio transcription | Later audio stress. |
| `tiny-whisper-cross-kv` | textproto encoder/decoder, tokenizer, `tiny.wav`, metadata | Whisper cross-KV | Later cross-KV/audio stress. |

I still recommend the tiny reasoning fixture as the first new fixture. The existing `tiny-llm` is usable by `repl_e2e.rs`, but it is not a reasoning fixture and it is not serialized `model.onnx`. The tiny reasoning fixture gives CI a cheap proxy for reasoning delimiters, hidden/visible answer split, max-token truncation, context pressure, and repetition/progress detection.

## Where it runs

- **Per-PR fast CI:** CPU-only, committed fixtures only. Use `crates/onnx-genai-cli/tests/repl_e2e.rs` for CLI session behavior and a small engine harness for state/KV invariants. The `cli-ort` Linux/Windows lane must continue to fail loudly if ORT is missing; do not silently skip the promised ORT tests.
- **Per-PR slow CI / required before merge for risky changes:** longer tiny-fixture runs, stochastic flag distribution smoke, native CPU when the feature set is compiled, and feature overlays for prefix caching/speculative/rewind.
- **Nightly:** env-var real-model tests on machines that actually provision the model directories. If a nightly lane advertises `ONNX_GENAI_QWEN3_0_6B_FOUNDRY_DIR`, absence is a job failure, not a skip.
- **Self-hosted/manual GPU:** CUDA native/ORT and Metal. GitHub runners do not provide CUDA GPUs, so the DeepSeek native CUDA repetition bug and ORT CUDA shared-GQA admission failure cannot be fully covered in hosted CI.
- **Miri:** keep separate. Miri is valuable for unsafe/state invariants but is not a real-model stress substitute.

## Harness shape

Build on existing machinery instead of parallel tools:

- Extend confirmed harness `crates/onnx-genai-cli/tests/repl_e2e.rs` for user-visible CLI invariants: turn commit/drop behavior, `/session`, `/stats`, `/profile`, `/backend`, `/ep`, and error text.
- Add an engine-level stress harness for invariants that need direct state: KV valid length, scheduler admission, prefix/speculative/rewind counters, and token positions.
- Reuse confirmed bench crate binaries (`crates/onnx-genai-bench/src/bin/profile_native.rs`, `compare.rs`) for backend identity, profile JSON shape, and native-vs-ORT diagnostics. `profile_native` is gated by the bench crate `bench-native` feature. Do not make benchmark throughput the pass/fail criterion except for explicit perf jobs.
- Treat profile/stats JSON as the artifact schema. If a needed field is missing, add it once and make all stress tests consume the same schema.

Confirmed concrete references:

| Reference | Confirmed location / status |
|---|---|
| CLI REPL harness | `crates/onnx-genai-cli/tests/repl_e2e.rs` exists and currently uses `tiny-llm` and `tiny-vlm-image-input`. |
| `cli-ort` CI lane | `.github/workflows/ci.yml` defines `cli-ort` for Linux x86_64 and Windows x86_64 and builds/tests `onnx-genai-cli`. |
| Bench binaries | `crates/onnx-genai-bench/src/bin/profile_native.rs` and `compare.rs` exist; `profile_native` requires the `bench-native` feature in `crates/onnx-genai-bench/Cargo.toml`. |
| `QWEN3_0_6B_CUDA_E2E_DIR` | Used by `crates/onnx-genai-engine/tests/qwen3_0_6b_native_cuda_e2e.rs`; CUDA + native-backend gated; expects `model.onnx`, `model.onnx.data`, metadata, tokenizer. |
| `ONNX_GENAI_QWEN3_0_6B_FOUNDRY_DIR` | Used by `crates/onnx-genai-engine/tests/qwen3_0_6b_foundry_native_cuda_lock.rs`; CUDA + native-backend gated Foundry native-CUDA golden lock. |
| `GLM_TINY_QMOE_E2E_DIR` | Used by `glm_tiny_qmoe_e2e.rs` and `glm_tiny_qmoe_native_cuda_e2e.rs`; can override the committed `tiny-glm52-qmoe-indexshare` fixture for GLM/QMoE tests. |


## Failure diagnosis and reproducibility

Every stress run writes one artifact directory under `target/integration-stress/<scenario>/<timestamp-or-ci-run>/` containing:

- `manifest.json`: git SHA, scenario name, OS/arch, command line, seed, turn count, model path/id/hash where available, backend, EP, feature flags, max tokens, context/KV budget.
- `runtime.json`: `onnx-genai version` output, resolved backend, available ORT providers, and the actual ORT library path/version loaded.
- `turns.jsonl`: one record per turn with prompt id/text hash, generated token count, visible answer length, finish reason, elapsed time, token positions, KV/page stats, scheduler admission outcome, prefix/speculative/rewind counters, repetition metrics, and any dropped-turn reason.
- `transcript.txt`: redacted enough for logs if needed, but complete for local runs.
- `failure.json`: invariant name, turn index, threshold, observed values, and the minimal command to reproduce.

Determinism rule: every stochastic scenario has an explicit seed and records the sampler parameters. Reproduction is `cargo test ... -- --exact <module::path::to::scenario>` or an emitted `onnx-genai run/generate/profile_native` command with the same seed, model, backend, EP, and feature flags. The filter has to be the scenario's *full* module path: `--exact` against a bare name matches nothing, runs zero tests, and still exits 0, so a renamed or mistyped scenario reproduces as a green run that measured nothing (see [`measurement-discipline`](../../../.github/skills/measurement-discipline/SKILL.md) §9).

## Cost and cadence

| Cadence | Required slice | Budget |
|---|---|---|
| Every PR | 2-3 CPU tiny scenarios: reasoning pressure, plain multi-turn stats, VLM prefix reuse. Linux + Windows ORT for CLI contracts. | Seconds to a few minutes. |
| Slow PR tier | Tiny MoE/QMoE, speculative, rewind/fork, stochastic flag smoke, 20-50 turns. | Optional/required by label or touched crates. |
| Nightly | Env-gated QMoE, longer context pressure, 50-100 turns, distribution checks across seeds. Real CPU/ORT reasoning joins only after a confirmed CPU/ORT fixture/env-var test exists. | Tens of minutes. |
| Self-hosted GPU nightly | CUDA native/ORT DeepSeek/Qwen, confirmed native-CUDA Qwen env-var tests, real KV-byte budgets, Metal when hardware exists. | Hardware-dependent. |
| Manual soak | 100+ turns, large models, new EPs/features, pre-release validation. | Not a merge gate. |

## Today's defects mapped to proposed tests

| Defect | Proposed catching test | Would it catch? |
|---|---|---|
| Fixed 128-token budget killed reasoning models after two turns | `ci_tiny_reasoning_pressure_cpu_ort`: tiny reasoning fixture, 8-12 turns, low max-token/context, asserts turn `N+1` still runs and reasoning spans close or drop cleanly. | Yes, once the tiny reasoning fixture exists. |
| `--temperature` / `--top-p` / `--top-k` silently ignored | Stochastic flag observability: fixed seed greedy stability plus repeated seeded stochastic runs where parameter changes alter sampled token distribution and `/session` reports the policy. | Yes. |
| Context exhaustion wrote an empty assistant turn, poisoning history | Reasoning/context pressure invariant: non-empty committed turns; exhausted/truncated reasoning turns are diagnosed and not kept; next `/session` message count is unchanged. | Yes. |
| Scheduler refused admission after several turns | `nightly_qmoe_scheduler_pressure` and GPU KV-budget stress: drive multi-turn sessions under real KV byte limits, assert admission liveness and recovery/reset after refusal. | Yes for CPU/QMoE-shaped scheduler logic; CUDA-specific shared-GQA memory path only on self-hosted GPU. |
| Repeating thinking on native CUDA | `self_hosted_cuda_deepseek_reasoning`: DeepSeek reasoning model, native CUDA, repetition/progress and termination invariants. | Not in GitHub-hosted CI. Yes in self-hosted/manual CUDA; a tiny reasoning fixture can catch generic repetition but not the specific native CUDA backend defect. |

## Phased plan

### Phase 1 — highest-value slice

1. Add the tiny reasoning fixture.
2. Add per-PR CPU ORT stress in `crates/onnx-genai-cli/tests/repl_e2e.rs`: 8-12 turns, low context/max tokens, `/stats` enabled, assertions for termination, non-empty committed turns, message/token consistency, and no excessive reasoning repetition.
3. Add a small stochastic flag observability test against a tiny fixture.
4. Make artifacts mandatory on invariant failure.

This phase directly covers three of the five defects and creates the invariant vocabulary for the rest.

### Phase 2 — stateful engine stress

Add an engine-level harness for scheduler admission, KV/page accounting, prefix caching, speculative decoding, and rewind/fork invariants. Run tiny fixtures per slow PR tier and env-gated real CPU models nightly.

### Phase 3 — hardware lanes

Stand up self-hosted CUDA and Metal lanes with explicit provisioning contracts. A lane that promises CUDA DeepSeek must fail when CUDA/ORT/model provisioning is absent; otherwise it should not be advertised as coverage.

### Phase 4 — soak and release gates

Before releases and major backend changes, run 100+ turn manual/self-hosted soaks across real reasoning, MoE, VLM, and backend-specific models. These are diagnostic gates, not every-PR blockers.
