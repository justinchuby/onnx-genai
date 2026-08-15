# Leon — History Archive

Compacted from `.squad/agents/leon/history.md` on 2026-07-29 (22,659 → ~8,000 bytes).
Hot history contains recent work; detailed narrative preserved below.

---

## Summary through 2026-07-14T20:05:00Z

### Engine and KV
Implemented attention-sink SWA, SharedKv generalization, connector engine wiring, and real KV byte materialization. Prefix lookup initially remained metric-only until K4 added symmetric f32 payload extraction/injection. Prefix-dependent hashing now proves equal keys imply equal prefixes. Follow-ups remain for multi-layer fixtures, graceful recompute fallback, and heterogeneous connector payloads.

### Gemma4 speculative execution
Migrated engine paths to per-layer KV geometry and helped deliver real heterogeneous Gemma4 E2B execution. Corrected proposer inputs to `embed(last_token) + last_hidden`, raising acceptance from 25% to 70.6% with token identity preserved. Performance remains below greedy and is a separate tuning concern.

### Loader dtype and fusion hardening
Closed silent Float32 fallbacks with `UnsupportedDataType` and fail-closed decoding across all real dtype sites; Holden approved. Added strict LayerNorm operand-order guards and adversarial coverage; Gaff approved.

### EPContext and encoder
Rejected encoder v1 for generic-layer EPContext literals violating the model-agnostic rule; Deckard's v2 passed. Revised EPContext writer sidecar naming after Batty's rejection, but introduced an over-broad duplicate-identity rejection; Leon is locked out of that writer artifact and Gaff's v3 is final. Later unified external-path guarding and explicit C API mapping were approved.

### Product/API and packaging
Renamed the full C ABI from `ort2_*` to `nxrt_*` with no compatibility aliases. Broke the shape-inference/loader publication cycle by making the loader dev-dependency path-only in the packaged manifest; Roy approved.

### Recent validation
Loader opset-import validation for file, from-parts, and nested-subgraph paths merged in `00cda89`; the executor's sentinel failure path is now an unreachable invariant. Holden's final review was green.

- 2026-07-15 — Added Windows oneDNN wheel bundling in `ef89a95`; CI verification is pending.

- 2026-07-16T00:00:01Z — Re-profiled native CPU decode after MatMulNBits threading and landed allocation-free, same-shape contiguous-f32 `Mul` (`347060f`). The guarded non-aliased fast path reduced Mul 3.12→0.25 ms and decode 40.5→44.2 tok/s; Holden 🟡 approved (independent +6.35%).

- 2026-07-16T00:00:00Z — Streamlined M=1 GQA decode to write contiguous f32 attention and present K/V outputs directly (`1fdd1ec`), preserving the generic prefill/strided/non-f32 path. GQA fell 0.865→0.690 ms/step and decode rose 54.38→58.44 tok/s (+7.5%) with exact eight-token output; Sebastian cleared the change and 413 CPU EP tests pass.

- 2026-07-16T00:00:00Z — Repaired the rejected CUDA executor control-flow paths in `5c0f05f` under Deckard's lockout. Non-host SequenceAt values now synchronously upload to correctly stamped CUDA buffers; Scan retains host staging and relies on child-executor H2D. Added CUDA SequenceAt/Scan versus CPU parity coverage; Holden cleared the repair, with exact Qwen tokens, session 112/112, and CUDA EP 117/117.

## 2026-07-16T15:39:27Z — Scribe session update

- 🟢 Reviewed BlockQuantizedMatMul: hand-verified MXFP4 0xD7→12.0/-6.0 and IQ4_NL decoding; unsupported IQ formats fail closed and 420 CPU tests pass.

## 2026-07-16T18:11:48+0000 — IQ-family CPU decode reviews

- 🟢 Cleared Bryant's IQ2_XS/IQ2_S/IQ3_XXS and IQ1_S/IQ1_M implementations after upstream llama.cpp grid, layout, fingerprint, and hand-trace audits.
- CPU `BlockQuantizedMatMul` now covers the complete supported IQ family.

## 2026-07-16T19:05:18+0000 — BlockQuantizedMatMul prefill review

- 🟢 Cleared Joi's `5010261`: all ten formats matched scalar decode bits, selected MXFP4/IQ4_NL/IQ4_XS AVX2 paths were independently checked, and generic GEMM retained K accumulation order.
- Default and oneDNN CPU EP suites each passed 430 tests; M=64 generic matmul gains measured 32–35×.

## 2026-07-16T19:27:57+0000 — CUDA IQ super-block GEMV wave

- 🟢 Cleared Roy's shared `onnx-runtime-quantization` extraction: all seven moved grids/sign tables are byte-identical (IQ1S FNV-1a `0x6703ed863501ae2e`); CPU decode and Joi's AVX2 paths are unchanged, and the standalone crate builds/tests cleanly.

## 2026-07-16T19-27-57+0000 — Scribe session update

- 🟢 Cleared Sapper's `67c1e3b` quantized-matmul shape rules: domains, `N`, symbolic dimensions, dtype preservation, error handling, and 2D/3D coverage are correct (93 unit tests + one doc-test).

## 2026-07-16T23:30:00+0000 — GAFF loader foundation review

- 🟢 Cleared Sapper's `2a9e5b1`: formal subgraph I/O is ordered and typed, recursive scopes isolate inline initializers, and UNDEFINED graph attributes retain populated graph fields.
- The Loop load regression and all 101 loader tests passed; existing validation already permits If/Loop/Scan.

## 2026-07-16T23:58:29+0000 — Comparison/logical inference review

- 🟢 Cleared Chew's `d06d1e7`: all comparison/logical output dtypes are Bool, broadcast/unary shapes hold, and bitwise operators were untouched; 115 tests passed.

## 2026-07-17T00:58:13Z — GAFF Loop remediation

- Under Sapper's lockout, repaired the rejected Loop design: removed the untrusted eager scan reservation and validated every loop-carried output against its initial dtype and full shape.
- Holden 🟢 re-approved the huge-`M` early-exit and second-iteration shape-change regressions; 121 session tests passed and final commit `f6e8ba6` merged. `Scan` is now the remaining control-flow work.

## 2026-07-14T00:00:00Z — Scan hardening and normalization inference

- Repaired Scan stack-shape arithmetic against zero-masked overflow (Holden 🟢) and added BatchNormalization/InstanceNormalization shape inference (Bryant 🟢).

## 2026-07-17T07:19:39Z — WEIGHT_OFFLOAD Phase 1 landed

- Delivered `f601cad`: `WeightRegionCatalog`, route-first mmap QMoE expert selection, and opt-in `ONNX_GENAI_WEIGHT_OFFLOAD=1`; default behavior remains unchanged.
- Chew's corrective `a77eed0` and Nabil 🟢 approval closed the landing; large-model exact-logit/throughput validation remains deferred.

- 2026-07-18 Scribe: Initial Reshape/Split validation and coverage work was superseded by the reviewed correction.

## 2026-07-18T01:20:34Z — CUDA SparseKvGather D==0 fix landed
- Fixed validation ordering in `c2180c9`; three D==0 parity tests passed 12/12, Gorman re-approved, and CUDA SparseKvGather landed.

- 2026-07-18: CUDA CSA claim gate was corrected to mirror CPU ratio-specific contracts and parity tests, then superseded by Deckard's shared attention_bias validation; final CSA approval landed.
- 2026-07-18T05:55:00Z — Added CPU CSA `supports_op` claim validation via unified factory dry-run on lockout reassignment (`2a08ef9`); Deckard approved.
- 2026-07-19: Fixed PR #30 device sampling parity/safety; continuing PR #32 rebase, build, and review-comment fixes.
- 2026-07-19T07:55:00Z: PR #32's EP-capabilities refactor merged at `9683a08` after the rebase and three review fixes.

## 2026-07-19T07:42:20Z — Mobius-head E2E harness landed

- Landed `3d47ea9`: pinned GLM-5.2 and DeepSeek-V4-Flash manifest plus ignored, environment-gated real-engine E2E smoke. Gaff approved; absent artifacts skip cleanly and no download path was added.

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.

## 2026-07-21T05:40:00Z — fp16 decode and cross-platform reconciliation

- Landed OS-aware CUDA dynamic-library candidates and graceful Windows ARM64 unavailability (`2466016`); Pris approved after the CUPTI gap was completed separately.

## 2026-07-21 — Wave-2 and CI milestone
CI now covers all 27 offline crates with warnings-as-errors and native Windows ARM64. Capture-safe native fp16 CUDA decode wave 2 stacked GQA prep fusion, warp-shuffle RMSNorm, and specialized down-projection GEMV on wave 1, reaching 663–672 tok/s on H200 versus ORT GenAI at 657, with zero fallbacks. All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.
- 2026-07-21T23:55Z — VLM WP2 native image processor landed via Sapper revision after Chew rejection; living VLM scope artifact preserved in inbox.

- 2026-07-22T00:00:00Z — Reviewed Batty CUDA-graph auto-enable 🟢 GREEN: 7/7 criteria passed, no model-name gating, correct env/metadata precedence, and capture-safety fallback intact.

## 2026-07-24T15:10:00Z — Phi decode-correctness lock

Authored the Phi-4-mini bit-exact native-CUDA-versus-ORT 64-token decode lock. Following Holden's review, Pris environment-gated and generalized shared `common/decode_lock.rs`; Leon co-owns this Qwen/Phi helper with Batty and Pris.

## 2026-07-26T19:45:52Z — Scribe update

- Spawned as leon-10 on `test/attention-default-domain-capture` for PR #193 default-domain Attention capture-path regression and revert-check; outcome pending.

## 2026-07-26T20:00:00Z — Scribe update

- 2026-07-26T20:00:00Z — Delivered PR #201 capture regression coverage for default-domain Attention staged-KV copy-back under CUDA graph capture; merged to main at `88e48eca`.
## 2026-07-26T22:38:02+00:00 — #88 RoPE capture DoD dispatched

- Dispatched on `test/rope-capture-dod` to add the standalone-RoPE graph record/replay or unfused-model zero-fallback token-parity DoD regression for issue #88. Status at Scribe handoff: in progress.
## 2026-07-26T22:38:02+00:00 — #88 RoPE capture DoD merged

- PR #208 (`test(cuda): cover standalone RoPE graph capture`) merged to main as `5eb0d8db`, closing #88. Chew independently approved with guard-break evidence; Resch's `63e0ef26` fmt-gate repair kept main green for the merge.

## 2026-07-27T13:12:20+00:00 — Roadmap wave-5

- PR #267 for #86 merged: pkg.nxrt::VarlenAttention consumes Attention-24 nonpad_kv_seqlen. Bishop required bf16 coverage; Batty supplied the lockout revision.

## 2026-07-27T12:20:00-07:00 — CLI context exhaustion guard

- Repaired PR #277 lockout defect in `fix/cli-sampling-and-context`: CLI now rejects `prompt_tokens >= effective_max_context` before decode, so exhausted turns do not append empty assistant history.
- Added equality/greater-than boundary unit coverage and preserved the one-token-room healthy path; clippy is clean with the known `pages.rs:129` lint allowed.

## 2026-07-27T15:30:00-07:00 — PR #291 fork/rewind review

- 🔴 Rejected Deckard's runtime fork/rewind PR. Public rewind reuses speculative helpers, but failed/rejected rewinds truncate logical tokens before backend KV/runner rewind succeeds, then reinsert the partially mutated session; this can leave tokens, kv_token_count, KV pages, and decode cursor inconsistent.
- Prefix-cache page retention and fork capability gating looked sound; support matrix is not acceptable until unsupported/rejected rewind paths are transactional. Batty should own revision under Deckard lockout.
- Validation: engine build/fmt/clippy passed; server/CLI builds passed; model-free new KV tests passed. Full engine lib suite failed locally from known ORT API/null-pointer environment mismatch (180 passed, 66 failed, 1 ignored).

## 2026-07-27T17:13:01-07:00 — PR #291 Batty revision re-review

- 🔴 Rejected Batty's transaction-boundary revision. Unsupported sliding-window and ORT-owned-KV paths are now clean and model-free regressions pass, but runner-backed rewind still truncates logical tokens and rewinds paged KV before fallible ORT runner rewind can finish.
- Verified paged materialized rewind uses an independent deep `PagedKvCache` clone before materialization; support matrix remains too strong for static-cache/shared-buffer rows until runner rewind is transactional. Gaff should own next revision under Deckard/Batty lockout.
- Validation: engine build/fmt/clippy passed; server/CLI builds passed; targeted model-free tests passed. Full engine lib suite remained at known local ORT mismatch baseline (182 passed, 66 failed, 1 ignored).

## 2026-07-27T20:54:14-07:00 — PR #291 Gaff Route-B third review

- 🔴 Rejected Gaff's Route-B revision. Public runner-backed `rewind_session_to` now rejects before session removal/token/KV mutation, but the reject policy was implemented in shared `rewind_target_state_to_len` / `rewind_draft_state_to_len`, accidentally disabling speculative runner rewinds for static-cache/PastPresent generation.
- Also flagged stale model-backed checkpoint test still expecting `tiny-llm` PastPresent rewind success despite the support matrix marking runner-backed public rewind unsupported. Sapper should revise under Deckard/Batty/Gaff lockout by splitting public API validation from internal speculative rewind policy.
- Validation: engine build/fmt/clippy passed; server/CLI builds passed; targeted model-free tests passed. Full engine lib suite remained at local ORT mismatch baseline (183 passed, 66 failed, 1 ignored). Conda ORT probe still loaded ORT 1.17/API 17, so model-loading verification remained blocked.

## 2026-07-27T02:00:00Z — Roadmap wave update
- Reviewed PR #300 / #76, requested changes for non-convex capability claims, then approved Rachael union-find + Kahn convexity fix.

## 2026-07-28T04-08-08+0000 — Wave 2 regression/roadmap update
- Approved PR #316 CJK renderer fix and mutation-proof wide-character tests.

## 2026-07-28T03:11:03-07:00 — PR #291 Sapper policy-split fourth review

- 🟢 Approved Sapper's revision. `RewindRunnerPolicy` now keeps public `Engine::rewind_session_to` on `RejectRunnerRewind` while speculative draft alignment, draft rewind, target accept/reject rewind, and overmaterialized cleanup use `AllowRunnerRewind`.
- Verified public runner-backed rewind rejects before scheduler/session/token/KV/runner mutation; policy-boundary tests would fail if allow/reject were flipped. Support matrix now matches the public API contract, with prepared/infallible runner rewind recorded as follow-up.
- Validation: engine build/fmt/clippy passed; server/CLI builds passed; full engine lib suite now runs locally with ORT 1.27/API 27 (252 passed, 0 failed, 1 ignored); targeted failed-rewind/speculative/checkpoint tests passed.

## 2026-07-28T17:40:00+0000
#365 metadata-hints integration merged after the structural identity remediation chain.

## 2026-07-28T13:30:00-07:00 — DeepSeek native+CUDA "repeats thinking, won't stop" investigation

- ⛔ COULD NOT REPRODUCE. Native CUDA EP is unrunnable on this host: cudarc is pinned to the CUDA 13 API set (onnx-runtime-ep-cuda/Cargo.toml:37, cuda-13000 + dynamic-loading) and dlopens cublasLt64_13.dll / cudart64_13.dll, but only CUDA 12 redist exists here (site-packages/nvidia/**); cu13 pip wheels are 0.0.1 stubs; no toolkit/nvcc. GPU present (RTX 4060, driver 591.55) but that alone is insufficient.
- Ran the strongest reachable differential instead: DeepSeek-R1-Distill-Qwen-1.5B (HF cache), --greedy, budgets 30/600/800, two reasoning prompts, ORT CPU vs native CPU. Result: BYTE-IDENTICAL generated text (bodies compared line-for-line) and BOTH terminate on EOS. No loop, no divergence.
- This exonerates the shared native decode stack for the symptom: EOS/stop (decode_loop.rs:311), attention mask + position ids (native_decode/cpu.rs:194-203, cuda.rs:854-895), steady-state KV, greedy argmax. Remaining candidates are native-CUDA kernel numerics (untestable here) or the model's default do_sample/temp=0.6 sampling regime (different fix class) — neither confirmed.
- No production change, no regression test (contingent on a fix; would be manufacturing). Findings + next-engineer plan (needs a CUDA-13 host; dump per-step logits and bisect) in .squad/decisions/inbox/leon-deepseek-repeat.md.

## 2026-07-28T22:10:00+0000 — DeepSeek native+CUDA repeat: CORRECTION + differential completed

- ✅ RETRACTION of my earlier "native CUDA unrunnable" claim. It was wrong. The native CUDA decode path DID load and execute on this RTX 4060. The load-bearing error: I assumed `cudarc` needs `cublasLt64_13.dll`/`cudart64_13.dll`, but both the EP loader (ep-cuda/dynamic_library.rs) and cudarc's own `get_lib_name_candidates` fall back `*_13`→`*_12`, and the CUDA driver API is `nvcuda.dll` (display driver, present). CUDA 12 redist in site-packages/nvidia satisfies it.
- Enablement (feature combo): CLI built `--features native-cuda`, run `--backend native` + `ONNX_GENAI_EP=cuda`. Two NVRTC header blockers, both host-provisioning not code bugs: (1) `cuda_fp16.h` → set CUDA_HOME to the wheel include; (2) fused GQA flash kernel needs `crt/mma.h`, absent from the runtime wheel → fetched `nvidia-cuda-nvcc-cu12`, merged its `include/crt/*` with `cuda_runtime/include`, pointed CUDA_HOME at the merge. No site-packages modified.
- DIFFERENTIAL (the deliverable): native CUDA vs ORT, `--greedy --raw`. Easy apples prompt @800 tok = BYTE-IDENTICAL, both terminate on EOS. Hard "missing dollar" hotel puzzle = BOTH backends fall into the SAME verbatim repetition loop and never emit EOS. Native and ORT diverge only at char 325 (one `,`/`.` token — fp16-GPU vs fp32-CPU tie-break); the loop is identical either way. Default sampling (temp 0.6) also loops on native CUDA.
- CONCLUSION: the repeat/non-termination is NOT a native-backend bug — it reproduces identically on ORT. Per task step 2 (both loop ⇒ model/sampling/stop issue), stop. The user's `CUDA KV capacity exceeded (4097 > 4096)` error is a downstream symptom: the model loop never stops, so native KV hits its default 4096 cap (load.rs:54); ORT has no cap so it just loops on. Root cause = DeepSeek-R1-Distill greedy/`--raw` degeneration (HF card recommends temp 0.6, not greedy; cf. PR #277).
- No production change, no regression test (would assert model behaviour / native==ORT, which holds — nothing in our stack to fix; not manufacturing one). Corrected findings in .squad/decisions/inbox/leon-deepseek-repeat.md.

## Historical summary through 2026-07-28

- Generalized shared KV, attention-sink SWA, connectors, and prefix payload
  materialization; equal prefix keys now prove equal content. Remaining work is
  multi-layer fixtures, graceful recomputation, and heterogeneous connector payloads.
- Delivered heterogeneous Gemma4 E2B speculative execution and corrected proposer inputs
  to `embed(last_token) + last_hidden`; correctness improved while performance tuning remains
  separate.
- Hardened loaders and fusion: unsupported dtypes fail closed; LayerNorm operand ordering is
  guarded; opset imports validate recursively; the `nxrt_*` C ABI replaced `ort2_*`.
- Implemented weight-offload foundations, route-first QMoE selection, CUDA
  `SparseKvGather` D==0 validation, and CPU CSA claim validation.
- Contributed to CUDA graph/capture correctness (SequenceAt/Scan parity, Phi decode lock,
  default-domain Attention and standalone RoPE capture regressions) and to the #291 rewind
  policy split: public runner rewind rejects before mutation while internal speculative rewind
  remains permitted.
- Unified native CUDA and ORT KV capacity policy in `onnx-genai-kv`, including transactional
  growth and injected allocation/copy/mask/capture-failure tests. Native CUDA growth is a
  graph boundary: preserve the prefix, invalidate stale capture, and recapture after growth.
  Real DeepSeek CUDA validation verified 4→8→16 growth and recapture; no speculative
  free-memory ceiling is imposed.

## 2026-07-29T03:45:00+0000 — PR #382 CPU shared-buffer regression lock

- Under Benny's reviewer lockout, added
  `cpu_shared_buffer_continuous_batch_uses_declared_kv_pairs`, using
  `tiny-llm-sharedbuffer` and explicit float32 KV metadata.
- The engine-level CPU test runs continuous batching and compares sequential generation; it
  fails at session construction if declared `model.io.kv_inputs` / `kv_outputs` are not
  threaded to `BatchedSharedBufferDecodeSession`.
- Revert verification proved the test catches the latent #380 regression previously hidden
  because the equivalent CUDA E2E auto-skips without CUDA. The repair and test merged in
  `85b9ba15`.

## 2026-07-28T18:00:00-0700 — PR #385 re-scoped onto #392 (server + Python sampling wiring)

- #392 merged the engine + CLI half of the model-sampling-defaults work to `main`
  (`resolve_sampling_defaults`, `Option`-typed `SamplingOverrides`, CLI wiring). Confirmed #392
  preserved the strict precedence (explicit override > model-declared > greedy fallback) and the
  three-state `Option` typing — no design regression to raise.
- Reset the branch onto `origin/main` and re-applied ONLY the delta #392 left missing (the two
  Copilot findings): server + Python wiring, the misnamed-test fix, and a resolver-level
  temperature-0 → greedy guard. Dropped everything already on `main`. Final diff: 7 files,
  +414/-49, single commit `b78d8bec`.
- Resolution stays at each front end's request-construction boundary (CLI already via #392;
  server via `ModelHandle::generation_defaults` in `prepare_generate_request`/`prepare_completion`;
  Python via `engine.metadata().generation`). Not the engine, because `GenerateOptions` erases
  explicit-vs-unspecified (RULES rule 5). Pipelines + audio pass `None` (no-op).
- Finding 2: renamed `explicit_temperature_zero_forces_greedy` →
  `explicit_greedy_override_is_applied_and_keeps_its_temperature`; the resolver now owns the
  `temperature == 0` → greedy mapping for every consumer (new test
  `resolved_temperature_zero_forces_greedy_without_explicit_greedy`). `temperature: Some(0.0)`
  without greedy = deterministic argmax; sampler never zero-divides (`TemperatureProcessor` only
  inserted when `temperature > 0.0 && != 1.0`).
- Behaviour change: server/Python callers that don't override sampling now decode stochastically
  against `do_sample: true` models, matching the CLI. No greedy-assuming test broke.
- Gates green: fmt --all clean; clippy -D warnings on engine/server/python/cli; engine lib 274,
  server sampling tests + 116 pass (1 pre-existing `vision.onnx`-fixture failure, identical on
  clean main), python 5, cli lib 103.
# Leon — History (compacted 2026-07-29)

**Role:** Engine/KV/runtime-buffer implementer. Runtime owns KV; model geometry from `inference_metadata.yaml`. Preserve device-buffer ownership, past/present aliasing, exact real-model comparison, reviewer lockouts.

**Historical summary through 2026-07-28:** Generalized shared KV, attention-sink SWA, connectors, prefix payload materialization (equal prefix keys prove content equality). Delivered heterogeneous Gemma4 E2B speculative execution (proposer inputs corrected). Hardened loaders/fusion (unsupported dtypes fail-closed, LayerNorm operand-order guarded, opset validation recursive, `nxrt_*` C ABI replaces `ort2_*`). Implemented weight-offload foundations, route-first QMoE, CUDA SparseKvGather D==0 validation, CPU CSA claim validation. Contributed to CUDA graph/capture correctness (SequenceAt/Scan parity, Phi decode lock, default-domain Attention and RoPE capture regressions) and PR #291 rewind policy split (public rewind rejects before mutation, internal speculative rewind allowed). Unified native CUDA/ORT KV capacity policy with transactional growth; real DeepSeek validation verified 4→8→16 growth/recapture.

Older detailed work (2026-07-14 through 2026-07-28) archived in `history-archive.md`.

## Recent work (2026-07-29)

### 2026-07-29T03:45:00+0000 — PR #382 CPU shared-buffer regression lock

- Under Benny's reviewer lockout, added `cpu_shared_buffer_continuous_batch_uses_declared_kv_pairs`, using `tiny-llm-sharedbuffer` and explicit float32 KV metadata.
- Engine-level CPU test runs continuous batching and compares sequential generation; fails at session construction if declared `model.io.kv_inputs` / `kv_outputs` stop reaching `BatchedSharedBufferDecodeSession`.
- Revert verification proved the test catches latent #380 regression previously hidden because equivalent CUDA E2E auto-skips without CUDA. Repair and test merged in `85b9ba15`.

### 2026-07-28T18:00:00-0700 — PR #385 re-scoped onto #392 (server + Python sampling wiring)

- #392 merged engine + CLI half of model-sampling-defaults work to `main` (`resolve_sampling_defaults`, `Option`-typed `SamplingOverrides`, CLI wiring). Strict precedence preserved: explicit override > model-declared > greedy fallback.
- Reset branch onto `origin/main`, re-applied only the delta #392 left missing: server + Python wiring, misnamed-test fix, resolver-level temperature-0 → greedy guard. Final diff: 7 files, +414/-49, single commit `b78d8bec`.
- Server/Python callers now decode stochastically against `do_sample: true` models, matching CLI. No greedy-assuming test broke.
- Gates green: engine lib 274, server sampling 116 pass (1 pre-existing fixture failure)

## 2026-07-29T12:30:00Z — tiny-reasoning-fixture rounds 2–3 (PR #411)

### Round 2 (replaced Batty after Gaff REJECT)
Authored statistical token-stream replacement. Luv ran it alone: 15/15 failures with
fix intact; one green in parallel suite was a fluke. Luv issued REJECT.

### Round 3 — resolved-policy surface (approved `f8ed4fb4`)
Surfaced sampling policy generation actually resolved into `--stats`/`--profile`.
`SamplingPolicy` captured from `turn_options` after `resolve_session_sampling`; same
struct moved into `TurnInput.options` (`:1352`) — no separate display-side resolution.
Two resolution sites unified: one helper called by both `/session` and every turn,
reading live backend on demand. No cache; no staleness across `/reload`/`/ep`/`/backend`.
`interactive.rs:1342-1347` + `generate.rs:122-127`.

Luv approved at `f8ed4fb4`. Mutation: both new tests FAIL 3/3; suite 42+2/44.
Mutated stats line `greedy=true temperature=1 top_k=0` — matches #385/#392 class.

### Delta (`88fa86b5`)
Moved capture inside `run_generation_turn` (`output.rs:206-211`). `turn` bound
immutably; moved into `backend.generate(turn, …)` at line 278 — no window between
capture and use. Divergence structurally impossible. Luv delta-approved after
mutation 3/3 red, isolation 10/10 green, full suite 44/44.

Also contributed to:
- Fixture `manifest.json`/generator string consistency fix (Batty's bug).
- Empty-answer invariant correction (manifest now accurately describes "drop
  whitespace-only" rather than asserting strict non-emptiness).

Durable rules:
- "Instrument the boundary you care about."
- "Two independent resolution sites for one policy is the defect, not an inconvenience."
- "Close a gap by construction rather than by comment where you can."
- "A checked-in fixture must be reproducible from its generator."
(`.squad/decisions.md`, reconstructed rules section, 2026-07-29)

Inbox drop `leon-reasoning-fixture-round3.md` was lost when the worktree was deleted

## 2026-08-10 — EP Plugin Compute Hardening (reviewer rejection fix)

**Context:** Holden's security re-audit flagged two findings (N1, N2) in `compute.rs`/`kernel_ctx.rs`. Deckard locked out; reassigned to Leon.

**N1 (CRITICAL — `compute_execute` panic guard):** Already present in current code — `catch_unwind` wraps the entire `compute_execute` body (confirmed at `compute.rs:~551`). Added test verifying the pattern.

**N2 (HIGH — negative dims wrap to usize::MAX):** Fixed in `kernel_ctx.rs`. Added `validate_dims()` helper that rejects any negative dimension with an actionable error message naming the dim index and value. Zero dims are accepted (legal ONNX).

**Additional hardening:**
- Element-count overflow: all `shape.iter().product()` replaced with `checked_mul` fold in `kernel_ctx.rs:validate_dims`, `compute.rs` intermediate buffer allocation, and `read_i64_tensor`.
- Byte-length overflow: `element_count * byte_size` uses `checked_mul`.
- Zero-dim null-ptr: zero-element tensors are allowed to have null data pointers; only non-zero-element tensors fail on null.
- 7 new unit tests: negative dim rejected, large negative rejected, element-count overflow, byte-length overflow, zero-dim accepted, scalar tensor, normal shape.
- 2 new compute tests: panic-guard pattern, contiguous_strides edge cases.

**Build status:** `cargo build -p onnx-runtime-ep-plugin` fails due to `graph_reader.rs` (Isidore's concurrent edits — missing fields/methods). All errors are confined to that file; `compute.rs` and `kernel_ctx.rs` have zero compile errors and pass clippy when graph_reader is stubbed. Noted for coordinator.

**No public API signatures changed.** `validate_dims` is `pub(crate)` only.
before Scribe ran; content reconstructed into `.squad/decisions.md`.
---

## 2026-08-10 — Clippy dead_code cleanup: validate_dims wired into read_inputs

**Branch:** squad/ep-plugin-export
**Triggered by:** Deckard validation gate failure; Reviewer Rejection Protocol prevented him from editing.

**Finding:** `validate_dims` in `kernel_ctx.rs:23` was reported as dead code by
`cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings`.

**Root cause — real gap, not cosmetic:** `validate_dims` was defined but never
called from `read_inputs`. The production path was silently casting ORT dims via
`.map(|&d| d as usize)` — a bare cast that passes negative dims as huge positive
values, bypassing the negative-dim rejection and overflow checks entirely.
This is exactly the "validation path never connected" scenario flagged in the mission.

**Fix:** Replaced the bare cast in `read_inputs` with a call to `validate_dims`,
so every set of ORT-supplied dims crossing the FFI boundary now goes through the
validated path. No `#[allow(dead_code)]` was used.

**Remaining clippy errors (not my files):**
- `lib.rs:184` — `unused-mut` (`let mut out_num`)
- `lib.rs:189` — `clippy::diverging-sub-expression` (panic! in test guard)
- `ep.rs:499` — `clippy::manual-dangling-ptr` (`1usize as *mut OrtEp`)
These are in Isidore's (`lib.rs`) and Deckard's (`ep.rs`) files; not touched.

**Test result:** `cargo test -p onnx-runtime-ep-plugin --lib` — 82 passed, 0 failed.

---

## 2026-08-10 — EP plugin parity wave: NEW-1 fix + f16/bf16 marshaling

**Branch:** squad/ep-plugin-parity-cuda

### TASK 1 — NEW-1: compute_release_state catch_unwind (compute.rs)

`compute_release_state` was the only `extern "C"` callback in the two owned
files lacking a `catch_unwind` guard. Wrapped the body in
`catch_unwind(AssertUnwindSafe(…))` with `let _ = result` to swallow any panic
(void return — no status channel). The other two callbacks (`compute_create_state`,
`compute_execute`) were already guarded; none were missed.

Added test `release_state_swallows_panic_safely` verifying the guard pattern.

### TASK 2 — f16/bf16 marshaling (kernel_ctx.rs)

Verified against `bindings.rs`:
- `ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10` → `DataType::Float16 = 10` ✅
- `ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16` → `DataType::BFloat16 = 16` ✅
- Both have `byte_size() = 2` — existing `checked_mul` overflow guards cover them.

Exposed `CPU_EP_SUPPORTED_DTYPES: &[DataType]` as a public constant so Deckard
can import it (do not copy) for `GetKernelRegistry` type constraints.

Added 7 new tests covering f16/bf16 round-trip, byte-length, overflow guard,
unsupported-dtype fail-closed, and the supported-dtypes constant.

**Test results:**
- `cargo test -p onnx-runtime-ep-plugin --lib` → 90 passed (was 82)
- `cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings` → clean
- `cargo test -p onnx-runtime-ep-cpu-plugin` → 15 passed

---

## 2026-08-10 — M2-1/M2-2: stream EP memory leak + doc comment (device.rs)

**Branch:** squad/ep-plugin-parity-cuda  
**Triggered by:** Holden milestone-2 audit; Nabil locked out (reviewer rejection).

### M2-1 (MEDIUM): EP instance leaked in `stream_release`

**Root cause:** `factory_create_sync_stream` (factory.rs:668) creates a fresh EP via
`Box::into_raw(ep)` and stores the pointer in `DeviceSyncStream.ep`. The `stream_release`
callback only dropped the `DeviceSyncStream` Box but never reclaimed the EP pointer.

**Fix:** Added `Box::from_raw(stream.ep as *mut dyn ExecutionProvider)` in
`stream_release` after dropping the stream Box. The null guard prevents UB if
somehow called with a null EP.

**Double-free ruling:** ORT header (lines 207–216) confirms `Release` is called
exactly once per created stream. No other code reclaims the stream EP. The allocator
path has its own independent EP instance created in `factory_create_allocator`.

**Regression test:** `stream_release_reclaims_owned_ep_no_leak` — uses an EP whose
Drop increments a static `AtomicUsize`; asserts count goes from 0→1 after release.

**Test fix:** Updated 3 existing stream tests to use `Box::into_raw(Box::new(MockGpuEp))`
instead of stack references, matching the real factory path and preventing UB under
the new release logic.

### M2-2 (LOW): misleading doc comment on `DeviceAllocator::memory_info`

Comment claimed "Owned; freed on drop" but there is no `Drop` impl and the pointer
is ORT-borrowed (`EpDevice_AddAllocatorInfo` stores raw pointer; ORT releases via
`ReleaseEpDevice`). Fixed to: "Borrowed from ORT; NOT freed by this allocator."

### Audit — no other unpaired `into_raw` in device.rs

All `Box::into_raw` in device.rs are in tests and are paired with `Box::from_raw`
or `stream_release`. No factory.rs change needed for this fix.

### Validation

- `cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings` → clean
- `cargo test -p onnx-runtime-ep-plugin --lib` → 133 passed (132 + 1 new)
- `cargo test -p onnx-runtime-ep-cpu-plugin --all-targets` → 17 passed
- `cargo check --workspace` → success


## [Archived 2026-08-12 by Scribe — wave: #31974-regression-cleanup]

## 2026-08-11 — Device data-transfer contract (`transfer.rs`)

**Branch:** `squad/ep-plugin-parity-cuda` (PR #762)

**Created:** `crates/onnx-runtime-ep-plugin/src/transfer.rs` — ORT `OrtDataTransferImpl` adapter.

**What:**
- `DeviceDataTransfer` (basic) and `DeviceDataTransferFull` (with OrtApi) adapters
- Copy-direction matrix: H→D, D→H, D→D(same) supported; cross-device + H→H rejected
- Stream-ordered copy via `copy_async` + `Fence` + `wait_fence`
- Ownership: Box::into_raw/from_raw lifecycle, EP borrowed not owned
- Mock device EP with non-host-dereferenceable address space for testing
- 21 new tests covering direction matrix, fail-closed CanCopy, ownership/leak detection, device-pointer guards

**Validation:**
- `cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings` → clean
- `cargo test -p onnx-runtime-ep-plugin` → 154 lib + 9 parity passed
- `cargo test -p onnx-runtime-ep-cpu-plugin` → 23 passed
- `cargo check --workspace` → success

**Not proven:** Nothing here proves CUDA works. Hardware-gated.

## 2026-08-11 — BL2/BL3: Optional slot positional integrity (PR #762)

**Branch:** `squad/ep-plugin-parity-cuda` (PR #762, draft)
**Triggered by:** Third independent Opus review rejection — silent corruption class.

### BL2 — Omitted optional outputs (graph_reader.rs)

**Root cause:** `filter_map` in `from_ort_graph()` dropped empty-named output slots, compacting the Vec. SkipLayerNormalization with signature `(output, "", "", sum)` became `[output, sum]` (len 2), causing the kernel to write mean into position 1 (which was really the sum slot).

**Fix (preferred — slot-map, not fail-closed):** Empty-named outputs get placeholder ValueIds with `DataType::Undefined`. In compute.rs fast path, Undefined-dtype slots receive local scratch buffers; ORT output indices increment only for present slots. Kernel sees full arity (4) and writes to correct positions.

### BL3 — Omitted optional inputs (compute.rs)

Added `NodeInputSource::Absent` variant. Compute loop provides `TensorView::absent(DataType::Undefined)` for Absent slots. Kernels detect absence via `is_absent()`.

**Caveat:** `ep.rs:597` still emits `Ort(0)` for None inputs — Sebastian's BL1 pass must change it to `Absent`. The single-node fast path works because it passes inputs from ORT directly (no routing table).

### Nonblocker — unwrap_or(DataType::Float32)

All three instances replaced with fail-closed error. A short `output_dtypes` vector is now a hard compute failure, not a silent Float32 guess.

### Tests (all real ORT, all numerical)

| Test | Asserts | Pre-fix behavior |
|------|---------|------------------|
| `skip_layer_norm_output_sum_position` | sum[0..8] == X+skip | Got mean (2.625) |
| `clip_omitted_min_with_max` | Y == clip(X, -∞, 5) | Would alias min=X |
| `skip_layer_norm_omitted_beta_bias` | LN(X+skip, γ=1, β=0) | Would alias β=X |
| `simplified_layer_norm_two_outputs_position` | inv_std correct | Position coverage |

**Validation:**
- `cargo test --no-fail-fast -p onnx-runtime-ep-plugin -p onnx-runtime-ep-cpu-plugin -p onnx-runtime-ep-cuda-plugin` → 215 passed / 0 failed
- `cargo clippy --all-targets -- -D warnings` on all 3 crates → clean
- `cargo fmt --check` → clean

## 2026-08-11 — PR #762 third corrective wave: BL2/BL3 optional slot fidelity

**Task:** Fix BL2 (output slot compaction by filter_map) and BL3 (absent inputs aliased to input 0).
**Commits:** `6ce94f033`, `49f39633b`, `e5dbed0dd`

- `graph_reader.rs` now preserves positional output slots with `ValueId` placeholders for empty-named slots.
- `NodeInputSource::Absent` variant added; compute loop passes `TensorView::absent()` for absent inputs.
- 3 `unwrap_or(DataType::Float32)` fallbacks removed; fails closed with explicit error.

**Outcome:** Fix correct at graph/compute level. However, Luv's review found the optional-slot conformance tests were vacuous — EP was declining the nodes at `ep.rs:275` (Undefined-dtype output check). BL2 fix was dead code in the ORT plugin path. Mariette corrected. Pris found BL1 regression test lacked fallback guard; Rachael hardened.

**Lesson reinforced:** A passing test is not evidence the code under test ran. `disable_cpu_ep_fallback=1` + `Session_GetEpGraphAssignmentInfo` assertions are both required.

## 2026-08-11 — PR #31988 TensorRT build fix

- **Task**: Clear last real build blocker (Build Linux TensorRT x64 Release).
- **Root cause**: `matmul_nbits_cols_per_block_test.cc` (host .cc) included `matmul_4bits_common.cuh` which pulls `<cuda_bf16.h>` → CUB device headers. Host compiler can't resolve `blockIdx`/`__threadfence`.
- **Verdict**: OURS — PR #31678 (unrelated) has TensorRT green; ours red.
- **Fix**: Extracted `SelectColsPerBlock` + constants to `matmul_4bits_cols_per_block.h` (host-only). Test uses host header; `.cuh` re-exports via include.
- **nvcc local**: Installed (12.0). Full compile not feasible (gsl/onnxruntime deps missing) but host header verified standalone with g++.
- **New head**: `34fe91e8dd`

## 2026-08-12 — PR #31988 TensorRT build fix

- **Task**: Clear `Build Linux TensorRT x64 Release` blocker on PR #31988.
- **Root cause**: `matmul_nbits_cols_per_block_test.cc` (host `.cc`) included `matmul_4bits_common.cuh`, which pulls `<cuda_bf16.h>` → CUB device headers. ~40 `'blockIdx' was not declared` errors in host compilation context.
- **Verdict: OURS** (not inherited) — cross-PR comparison: #31678 (unrelated) TensorRT green; #31988 red.
- **Fix**: Extracted `SelectColsPerBlock`, `kColsPerThreadBlock`, `kTargetCtasPerSm` into `matmul_4bits_cols_per_block.h` (host-only, no device includes). Test uses only this header; `.cuh` re-exports via `#include`. All four invariants preserved (routing/output/wide-n/split-K unchanged).
- **Head**: `34fe91e8dd`.

## 2026-08-12 — PR #32001 rejection fixes

Fixed all rejection items (B1, B2, N1–N4) for the Apple Accelerate infrastructure PR. Robust arm64 detection, Darwin-only gating, reinstated MLAS_USE_APPLE_ACCELERATE=1 compile definition, loud CLI failure, CPU EP group placement, and PR body rewrite. Pushed `0d924a421b`. PR stays draft pending Opus re-review.

## 2026-08-12 — PR #32001 rejection fixes (arm64 detection B1 + N1–N4)

Fixed all rejected items on Apple Accelerate infrastructure PR #32001. B1: `CMAKE_OSX_ARCHITECTURES` defined-but-empty + `onnxruntime_target_platform` unset → feature silently disabled on own Apple Silicon hardware. Fixed by reading `CMAKE_OSX_ARCHITECTURES` for arm64/arm64e first, falling back to `CMAKE_SYSTEM_PROCESSOR`. N1: Darwin-only gate. N2: `MLAS_USE_APPLE_ACCELERATE=1` reinstated. N3: Loud `BuildError` on CLI opt-in. N4: Flag in CPU EP argument group. B2: PR body rewritten. Head `0d924a421b`.
