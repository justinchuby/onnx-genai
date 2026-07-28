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

1. `ci_tiny_reasoning_pressure_cpu_ort`: tiny reasoning fixture, ORT CPU, 8-12 turns, low context, greedy + one stochastic seed.
2. `ci_tiny_plain_multiturn_cpu_native`: tiny plain LLM, native CPU if built, 8-12 turns, stats/profile consistency.
3. `ci_tiny_vlm_prefix_cpu_ort`: existing tiny VLM, repeated image follow-up, encoder/prefix cache invariants.
4. `nightly_qmoe_scheduler_pressure`: `tiny-glm52-qmoe-indexshare` or `GLM_TINY_QMOE_E2E_DIR`, multi-turn plus small KV budget.
5. `nightly_real_reasoning_cpu_ort`: `ONNX_GENAI_QWEN3_0_6B_FOUNDRY_DIR` or equivalent, 20-50 turns, repetition/reasoning invariants.
6. `self_hosted_cuda_deepseek_reasoning`: DeepSeek R1 Distill Qwen 1.5B on native CUDA and ORT CUDA, reproducing Justin's two observed failures.

## Model fixture tiers

| Tier | Fixtures / env vars | Use |
|---|---|---|
| Tier 0 committed tiny | `tests/fixtures/tiny-llm`, `tiny-vlm-image-input`, `tiny-llm-sharedbuffer`, `tiny-native-*` | Per-PR invariant plumbing and CLI/history/stats checks. |
| Tier 0.5 new tiny reasoning | Add a committed tiny reasoning fixture that emits `<think>`, can close or fail to close reasoning, and has a deliberately small context window. | Highest-value addition. It would have caught the fixed token-budget/empty-turn bugs in CI and would turn repeating reasoning into a generic invariant failure. |
| Tier 1 committed specialized | `tiny-glm52-qmoe-indexshare`, `tiny-gemma4-vlm`, MTP/Eagle/speculative fixtures | Nightly/slow CI for MoE, VLM, speculative, and index-sharing state. |
| Tier 2 env-gated real | `QWEN3_0_6B_CUDA_E2E_DIR`, `GLM_TINY_QMOE_E2E_DIR`, `ONNX_GENAI_QWEN3_0_6B_FOUNDRY_DIR` | Nightly where models exist; local/self-hosted for GPU. These must fail loudly when a job promises the model. |
| Tier 3 manual repro | DeepSeek R1 Distill Qwen 1.5B and other large/reasoning models | Manual or self-hosted soak; not GitHub-hosted CI. |

I agree the tiny reasoning fixture is the first fixture to add. It gives CI a cheap proxy for the exact failure class: reasoning delimiters, hidden/visible answer split, max-token truncation, context pressure, and repetition/progress detection.

## Where it runs

- **Per-PR fast CI:** CPU-only, committed fixtures only. Use `repl_e2e.rs` for CLI session behavior and a small engine harness for state/KV invariants. The `cli-ort` Linux/Windows lane must continue to fail loudly if ORT is missing; do not silently skip the promised ORT tests.
- **Per-PR slow CI / required before merge for risky changes:** longer tiny-fixture runs, stochastic flag distribution smoke, native CPU when the feature set is compiled, and feature overlays for prefix caching/speculative/rewind.
- **Nightly:** env-var real-model tests on machines that actually provision the model directories. If a nightly lane advertises `ONNX_GENAI_QWEN3_0_6B_FOUNDRY_DIR`, absence is a job failure, not a skip.
- **Self-hosted/manual GPU:** CUDA native/ORT and Metal. GitHub runners do not provide CUDA GPUs, so the DeepSeek native CUDA repetition bug and ORT CUDA shared-GQA admission failure cannot be fully covered in hosted CI.
- **Miri:** keep separate. Miri is valuable for unsafe/state invariants but is not a real-model stress substitute.

## Harness shape

Build on existing machinery instead of parallel tools:

- Extend `repl_e2e.rs` for user-visible CLI invariants: turn commit/drop behavior, `/session`, `/stats`, `/profile`, `/backend`, `/ep`, and error text.
- Add an engine-level stress harness for invariants that need direct state: KV valid length, scheduler admission, prefix/speculative/rewind counters, and token positions.
- Reuse bench crate binaries (`profile_native`, `compare`) for backend identity, profile JSON shape, and native-vs-ORT diagnostics. Do not make benchmark throughput the pass/fail criterion except for explicit perf jobs.
- Treat profile/stats JSON as the artifact schema. If a needed field is missing, add it once and make all stress tests consume the same schema.

## Failure diagnosis and reproducibility

Every stress run writes one artifact directory under `target/integration-stress/<scenario>/<timestamp-or-ci-run>/` containing:

- `manifest.json`: git SHA, scenario name, OS/arch, command line, seed, turn count, model path/id/hash where available, backend, EP, feature flags, max tokens, context/KV budget.
- `runtime.json`: `onnx-genai version` output, resolved backend, available ORT providers, and the actual ORT library path/version loaded.
- `turns.jsonl`: one record per turn with prompt id/text hash, generated token count, visible answer length, finish reason, elapsed time, token positions, KV/page stats, scheduler admission outcome, prefix/speculative/rewind counters, repetition metrics, and any dropped-turn reason.
- `transcript.txt`: redacted enough for logs if needed, but complete for local runs.
- `failure.json`: invariant name, turn index, threshold, observed values, and the minimal command to reproduce.

Determinism rule: every stochastic scenario has an explicit seed and records the sampler parameters. Reproduction is `cargo test ... -- --exact <scenario>` or an emitted `onnx-genai run/generate/profile_native` command with the same seed, model, backend, EP, and feature flags.

## Cost and cadence

| Cadence | Required slice | Budget |
|---|---|---|
| Every PR | 2-3 CPU tiny scenarios: reasoning pressure, plain multi-turn stats, VLM prefix reuse. Linux + Windows ORT for CLI contracts. | Seconds to a few minutes. |
| Slow PR tier | Tiny MoE/QMoE, speculative, rewind/fork, stochastic flag smoke, 20-50 turns. | Optional/required by label or touched crates. |
| Nightly | Real CPU/ORT reasoning model, env-gated QMoE, longer context pressure, 50-100 turns, distribution checks across seeds. | Tens of minutes. |
| Self-hosted GPU nightly | CUDA native/ORT DeepSeek/Qwen, real KV-byte budgets, Metal when hardware exists. | Hardware-dependent. |
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
2. Add per-PR CPU ORT stress in `repl_e2e.rs`: 8-12 turns, low context/max tokens, `/stats` enabled, assertions for termination, non-empty committed turns, message/token consistency, and no excessive reasoning repetition.
3. Add a small stochastic flag observability test against a tiny fixture.
4. Make artifacts mandatory on invariant failure.

This phase directly covers three of the five defects and creates the invariant vocabulary for the rest.

### Phase 2 — stateful engine stress

Add an engine-level harness for scheduler admission, KV/page accounting, prefix caching, speculative decoding, and rewind/fork invariants. Run tiny fixtures per slow PR tier and env-gated real CPU models nightly.

### Phase 3 — hardware lanes

Stand up self-hosted CUDA and Metal lanes with explicit provisioning contracts. A lane that promises CUDA DeepSeek must fail when CUDA/ORT/model provisioning is absent; otherwise it should not be advertised as coverage.

### Phase 4 — soak and release gates

Before releases and major backend changes, run 100+ turn manual/self-hosted soaks across real reasoning, MoE, VLM, and backend-specific models. These are diagnostic gates, not every-PR blockers.
