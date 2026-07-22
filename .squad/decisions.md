# Decisions

> Current decision ledger. Full prior history through 2026-07-20T13:35Z is preserved in
> `.squad/decisions/archive/2026-07-20T13-35-00Z-decisions-pre-multistream.md`.

> Entries older than 2026-06-21T23:55Z are archived in `.squad/decisions/archive/2026-Q2.md` when present.

<!-- scribe-merge-2026-07-22T12-00-00Z-phase0-7b-cudagraph -->
## 2026-07-22 — Partial CUDA-graph Phase 0 and Qwen2.5-7B CUDA-graph benchmark

<!-- source: .squad/decisions/inbox/deckard-luv-phase0-review.md -->
### 2026-07-22: Review verdict — Luv Phase 0 partial-CUDA-graph capture-path-kind (🟢 GREEN)

**By:** Deckard

**What:** Independent read-only review of `squad/luv-capture-pathkind` (commit 3c94a57) diffed against merge-base with `origin/main`. Changed: `executor.rs` (+`CapturePathKind`/`SeamReason` enums, `CaptureDecline.seam_reason: Option<SeamReason>`, seam-kind label in `log_capture_segmentation`, `CaptureDecline::node` now takes a `SeamReason`), `lib.rs` (re-exports + doc), `native_decode.rs` (+1 field in a test fixture), docs. **Verdict: 🟢 GREEN — safe to merge.**

**Why:**
1. **Byte-identical behavior — PASS.** Only removed string literal is the log-format line (now inserts `[{seam_label}]`); zero decline `reason` strings were removed or altered. Segmentation logic in `plan_capture_segments` is unchanged — `declines[pi].is_none()` still drives partitioning; boundaries pushed identically. Classification is derived *from* existing decline causes, not a replacement.
2. **Correct mapping — PASS.** All 5 per-node causes map correctly: control-flow/sequence→`HostControlFlowOrSequence`→`HostSeam`; unresolved output→`UnresolvedOutputShape`; unresolved input→`UnresolvedInputShape`; kernel-not-warmed→`KernelNotWarmed`; kernel-capture-unsupported→`KernelCaptureUnsupported` — the last four→`EagerDeviceSeam`. Graph-level persistent-device-binding hard-abort (`CaptureDecline::graph`) intentionally carries `seam_reason: None` ("graph-level hard preconditions"), which is correct — it is a whole-graph abort, not a per-node seam.
3. **Model-agnostic — PASS.** No model-name/architecture string branching; classification is purely structural (RULES.md §2/§2.1 respected).
4. **Exhaustiveness — PASS.** `SeamReason::path_kind` and `CapturePathKind::label` use exhaustive matches with no catch-all `_ =>`; `CapturePathKind`/`SeamReason` re-exported from `lib.rs` and doc-commented.
5. **fmt/clippy — PASS.** `cargo fmt -p onnx-runtime-session -- --check` clean; `cargo clippy -p onnx-runtime-session --all-targets -- -D warnings` clean; `--features cuda` clippy clean.
6. **Tests — PASS.** `cargo test -p onnx-runtime-session` = 60 passed, incl. new `seam_reasons_map_to_structural_capture_paths` (genuinely asserts all 5 reason→kind→label mappings + `CaptureRegion` label). `cargo test -p onnx-genai-engine --features native-backend capture_fallback_emits_each_structured_decline_to_tracer` = 1 passed.
7. **Log output — PASS.** Seam-kind label uses `boundary.seam_reason.map(SeamReason::label).unwrap_or("unclassified-seam")`; behind the verbose diagnostic flag; no existing test asserts on the literal log string, so no format-assertion breakage.

Conclusion: purely additive structural diagnostics, correct, model-agnostic, all gates green. Approved for merge.

<!-- source: .squad/decisions/inbox/gaff-qwen7b-cudagraph.md -->
### 2026-07-22: Qwen2.5-7B int4 CUDA-graph auto-enable benchmark
**By:** Gaff
**What:** Benchmarked Qwen2.5-7B int4 on one NVIDIA H200 at `bd3d95a` using `profile_native --ep cuda --prompt Hello --tokens 128 --warmups 2 --runs 3 --steady`, `ONNX_GENAI_DEVICE_KV=1`, and identical greedy decoding. Run A left `ONNX_GENAI_CUDA_GRAPH` unset; Run B set it to `0`. A companion 16-token diagnostic confirmed graph state and fallback counters.
**Why:** Validate that metadata/structure-driven CUDA-graph auto-enable generalizes beyond Qwen2.5-0.5B and Phi-4-mini without architecture or model-name keying.

| Metric | Run A — auto | Run B — forced eager |
|---|---:|---:|
| Median throughput | **231.73 tok/s** | **180.50 tok/s** |
| Median decode latency | **4.315 ms/token** | **5.540 ms/token** |
| Throughput speedup vs eager | **+28.38%** | baseline |
| Token-exact A/B | **Yes** | **Yes** |
| Capture engaged | **Yes** | No (explicitly disabled) |
| Zero fallbacks | **Yes** | Yes |
| Capture diagnostic | `enabled=true`, 1 capture, 14 replays, 0 fallbacks; 1 captured segment, 0 eager seams | `enabled=false`, 0 captures, 0 replays, 0 fallbacks |
| Kernels/token | N/A — `profile_native` does not surface GPU kernel-launch counts | N/A |
| GPU-busy | N/A — `profile_native` does not surface GPU utilization | N/A |
| Fraction of 4.8 TB/s ÷ 3.5 GB/token ceiling | **16.90%** | **13.16%** |

The 128-token outputs were identical token-for-token across A and B. Auto-enable generalized cleanly to Qwen2.5-7B: CUDA plus owned device KV selected whole-step capture automatically, with one captured segment, no eager seams, and zero fallbacks. The **28.38%** gain is smaller than Qwen2.5-0.5B's 87.7% and Phi-4-mini's 41.0%, as expected for a larger decode that spends more time streaming/dequantizing int4 weights and less proportionally on launch overhead, but it remains substantial. The simple peak-bandwidth roofline is about 1,371 tok/s; measured auto throughput is 16.90% of that ceiling, and this ratio should not be interpreted as pure bandwidth efficiency because int4 dequantization and compute also constrain decode.

<!-- source: .squad/decisions/inbox/luv-capture-pathkind.md -->
### 2026-07-22: Formalize partial CUDA-graph capture path kinds
**By:** Luv
**What:** Added `CapturePathKind` and `SeamReason`, attached optional seam classification metadata to `CaptureDecline`, propagated it through `CaptureSchedule` boundaries, and added seam-kind labels to `ONNX_GENAI_LOG_CAPTURE_SEGMENTS` output without changing capture partitioning or existing reason strings.
**Why:** Phase 0 of the partial-CUDA-graph EP-claim design requires structural, model-agnostic diagnostics that distinguish captured regions, eager device seams, and host seams before EP-owned planning is introduced.

| SeamReason | CapturePathKind |
|---|---|
| `HostControlFlowOrSequence` | `HostSeam` |
| `UnresolvedOutputShape` | `EagerDeviceSeam` |
| `UnresolvedInputShape` | `EagerDeviceSeam` |
| `KernelNotWarmed` | `EagerDeviceSeam` |
| `KernelCaptureUnsupported` | `EagerDeviceSeam` |

**Files touched:**
- `crates/onnx-runtime-session/src/executor.rs`
- `crates/onnx-runtime-session/src/lib.rs`
- `crates/onnx-genai-engine/src/native_decode.rs`
- `docs/design-ep-partial-cuda-graph.md`
- `docs/CUDA_GRAPH_CAPTURE.md`

**Verification:**
- `cargo fmt -p onnx-runtime-session` — PASS.
- `cargo test -p onnx-runtime-session seam_reasons_map_to_structural_capture_paths` — PASS (1 focused unit test).
- `cargo build -p onnx-runtime-session` — PASS.
- `cargo build -p onnx-runtime-session --features cuda` — PASS.
- `cargo test -p onnx-runtime-session` — PASS (all session unit, integration, and doc tests; one manual performance audit and one doc test remained ignored).
- `cargo clippy -p onnx-runtime-session --all-targets -- -D warnings` — PASS.
- `cargo test -p onnx-genai-engine --features native-backend capture_fallback_emits_each_structured_decline_to_tracer` — PASS (1 focused compatibility test).

### Fold processed Phase 0 and 7B CUDA-graph inbox notes
**By:** Scribe
**What:** Merged and cleared `deckard-luv-phase0-review.md`, `gaff-qwen7b-cudagraph.md`, `luv-capture-pathkind.md`. Preserved active scope/design files `zhora-deepseek-scope.md`, `leon-vlm-scope.md`, and `keaton-native-specdecode-design.md`.
**Why:** Landed implementation, independent green review, benchmark results, and progress-log updates belong in the current decision ledger; active scope notes remain in the inbox.

<!-- scribe-merge-2026-07-22T00-00-00Z-cudagraph-autoenable -->
## 2026-07-22 — CUDA graph auto-enable, GQA/VLM closure, and inbox reconciliation

### Land metadata-driven native CUDA graph auto-enable
**By:** Batty; reviewed by Leon 🟢
**What:** Merged `batty-45` to main as `610bde0`, auto-enabling whole-step CUDA graph capture in `native_decode.rs` whenever metadata and device bindings prove the native decode topology graph-safe. Environment precedence remains explicit-disable first, then explicit-enable, then metadata auto-enable; capture-safety fallback remains transparent.
**Why:** Gaff's H200 profile showed native decode was launch/CPU-dispatch bound rather than bandwidth-bound. Auto-enable turned proven graph-safe models on by default without model-name gates.
**Validation:** Leon reviewed `squad/batty-cudagraph-autoenable` 🟢 GREEN with 7/7 criteria passing. H200 results were token-exact with zero fallbacks: Qwen2.5-0.5B improved **441.49→828.54 tok/s (+87.7%)** and Phi-4-mini improved **67.32→94.91 tok/s (+41.0%)**.

### Close GQA `seqlens_k` exporter-shape blocker
**By:** Chew and Roy; reviewed by Deckard 🟢
**What:** Accepted canonical dense contiguous int32 `seqlens_k` shapes `[batch_size]` and `[batch_size, 1]`, normalized trailing singleton shape for capture signatures, and revised non-contiguous diagnostics to name both accepted shapes. Coordinator merged the fix to main as `f4484e7`.
**Why:** Real Foundry Qwen2.5-1.5B and Phi-4-mini exports provide `[batch_size, 1]`; scalar-only support did not unblock those models. Deckard's initial review was 🔴 only for diagnostic wording; re-review passed after Roy's correction.

### Record native CUDA benchmark and model-coverage outcomes
**By:** Gaff, Okonkwo, Chew, Deckard, Pris, Holden, and Tyrell
**What:** Folded the decode roofline and re-benchmark sequence: Qwen2.5-0.5B baseline native CUDA decode around 435 tok/s before CUDA graph auto-enable; Qwen2.5-1.5B first blocked by `[batch,1]` GQA lengths, then by M=5 prefill until the SwiGLU M>1 path landed; Phi-4-mini native CUDA validated on H200 after int4 zero-points and partial-RoPE fixes. The native CPU coverage census, DS-1 dynamic shape-chain validation, DS native E2E exact parity, MLA conformance guard, and progress-log updates are now represented here or in existing 2026-07-22 ledger sections.
**Why:** These notes establish which blockers were generic runtime gaps, which were already closed on main, and which measurements motivated CUDA graph auto-enable rather than model-specific dispatch.

### Fold VLM WP1 runtime-contract and CI notes
**By:** Rachael, Roy, Deckard, Leon, and Sebastian
**What:** Preserved the VLM WP1 review sequence: Leon rejected non-executable metadata revisions, Roy/Rachael moved preprocessing metadata toward explicit runtime contracts, Deckard fixed Qwen temporal patch packing order, and Leon re-reviewed the temporal-order fix 🟢. Sebastian made PR #416 schema/processor tests offline-safe by skipping unavailable local assets rather than failing CI.
**Why:** VLM metadata must be executable through declared processor/registry contracts, not shape-only JSON acceptance; cached-processor parity gates must be environment-aware.

### Fold partial CUDA-graph EP-claim design notes
**By:** Keaton; reviewed by Fact Checker 🟡
**What:** Recorded the proposed partial CUDA-graph capture design for EP subgraph claiming, with whole-step capture prioritized first and partial capture constrained by static seam-output and KV-append invariants.
**Why:** The design remains a follow-up proposal; whole-step capture is the immediate path for fixed-topology device-resident decode.

### Fold processed inbox notes
**By:** Scribe
**What:** Merged and cleared `batty-cudagraph-autoenable.md`, `chew-gqa-batch1.md`, `chew-model-coverage-census.md`, `coordinator-gqa-merge.md`, `deckard-ds1-shapechain.md`, `deckard-dsnative.md`, `deckard-gqa-batch1-review.md`, `deckard-gqa-rereview.md`, `deckard-mla-conformance-review.md`, `deckard-wp1-packer-fix.md`, `factchecker-keaton-epclaim-review.md`, `gaff-decode-profile.md`, `gaff-native-rebench.md`, `gaff-native-rebench2.md`, `gaff-native-rebench3.md`, `gaff-phi4-bench.md`, `gaff-phi4-benchmark.md`, `holden-partial-rotary.md`, `keaton-epclaim-design.md`, `keaton-epclaim-v2.md`, `leon-batty-cudagraph-review.md`, `leon-wp1-rereview.md`, `leon-wp1-review.md`, `okonkwo-gqa-decode-bench.md`, `pris-ds1-testreview.md`, `pris-gqa-scalar-seqlens-plan.md`, `pris-holden-rotary-review.md`, `pris-mla-conformance.md`, `rachael-wp1-revision.md`, `roy-gqa-batch1-revision.md`, `roy-wp1-revision.md`, `sebastian-mobius416-ci.md`, `tyrell-progress-0722.md`, `zhora-glm-l4-fix.md`. Preserved active scope/design files `zhora-deepseek-scope.md`, `leon-vlm-scope.md`, and `keaton-native-specdecode-design.md`.
**Why:** Completed implementation, review, benchmark, CI, and duplicate ledger artifacts belong in the current decision ledger; active scope notes remain in the inbox.

<!-- scribe-merge-2026-07-22T00-00-00Z-int4-zp -->
## 2026-07-22 — Phi-4-mini int4 zero-point blocker closure

### Close BLOCKER #3: explicit int4 zero-points in native CUDA fp16 GEMV
**By:** Sapper; reviewed by Holden 🟢
**What:** Merged commit `48de993`, threading packed per-block int4 `zero_points` plus `zp_row_bytes` through the native CUDA fp16 GEMV path so asymmetric int4 MatMulNBits models such as Phi-4-mini decode with explicit zero points. Null zero-point inputs preserve the existing symmetric zp=8 fast paths.
**Why:** Removes BLOCKER #3 with a structural, model-agnostic asymmetric int4 path while keeping M==1 capture safety, SM-portable arithmetic, and symmetric no-regress behavior.
**Validation:** Holden's non-author review passed all five criteria (SM-portability, capture-safety, symmetric no-regress, genericity, correctness). H200 validation passed 6/6 unit tests and 18/18 `matmul_nbits_gpu` integration tests, including explicit-zp CPU-reference and capture-replay coverage.

### Fold processed int4 zero-point inbox notes
**By:** Scribe
**What:** Merged and cleared `sapper-int4-zp.md` and `holden-int4-zp-review.md`.
**Why:** The implementation and independent green review are now represented in the ledger; unrelated active inbox artifacts remain untouched.

<!-- scribe-merge-2026-07-22T06-17-16Z -->
## 2026-07-22 — Native proposer contract and Qwen0.5B H200 benchmark

### Land metadata-driven native proposer execution contract
**By:** Batty; reviewed by Deckard 🟢
**What:** Land commit `96c79d0`, replacing hardcoded native proposer assumptions with metadata-driven `sequence_source` (`input_ids`/`inputs_embeds`), `kv_ownership` (`owned`/`shared`), explicit shared-KV ports, and semantic output roles (`logits_output`/`hidden_output`). Defaults preserve legacy token-id + owned-KV behavior; CPU shared-KV proposer execution is complete.
**Why:** Embedding-driven shared-KV assistants must be activated by declared contracts rather than model or tensor-name assumptions. CUDA device-buffer shared-KV aliasing remains explicitly scoped out until device binding alias/reference support lands.

### Record Qwen2.5-0.5B native CUDA H200 decode benchmark
**By:** Gaff
**What:** Qwen2.5-0.5B native CUDA decode on H200 measured **437.76 tok/s median** (**2.284 ms/token**), with coherent deterministic output. This is **15.2% faster** than the user's RTX 4060 380 tok/s reference and **2.83%** of the H200 weight-bound roofline.
**Why:** Establishes the current native-path performance point for the 0.5B model on shared H200 hardware and shows the path is coherent but still far from the weight-bound ceiling.

### Fold processed proposer and benchmark inbox notes
**By:** Scribe
**What:** Merged and cleared `batty-proposer-contract.md`, `deckard-batty-proposer-review.md`, and `gaff-qwen05-bench.md` when present.
**Why:** Landed implementation, review, and benchmark records belong in the ledger; active unrelated inbox artifacts remain in place.

<!-- scribe-merge-2026-07-22T05-52-21Z -->
## 2026-07-22 — Fused CUDA SwiGLU M>1 prefill merge

### Land generic fused gate/up SwiGLU M>1 prefill
**By:** Bryant; reviewed by Deckard 🟢
**What:** Land commit `97e0cb4` from `wt-swiglu-prefill`, extending `run_f16_gate_up_swiglu` so M>1 prefill runs the existing portable fp16 MatMulNBits tiled GEMM twice (gate into scratch, up into output) and then applies the existing fp16 SiluMul in place. The M=1 paired GEMV path remains unchanged and capture-safe; M>1 explicitly records `last_call_capture_safe=false`.
**Why:** The graph optimizer removes the unfused gate/up nodes, so the fused node must handle prompt rows as well as decode. Review confirmed bit-exact M=1 and M>1 coverage, SM portability, generic dispatch, correct capture flag behavior, and scratch lifetime safety; H200 rebuild plus 4 SwiGLU tests passed before merge.

### Fold processed SwiGLU inbox notes
**By:** Scribe
**What:** Merged and cleared `bryant-swiglu-prefill.md` and `deckard-bryant-swiglu-review.md`. Preserved unrelated active in-flight deliverables in `.squad/decisions/inbox/`.
**Why:** Landed implementation and review decisions belong in the ledger; active scope/review/revision artifacts should remain in the inbox until their work lands.

<!-- scribe-merge-2026-07-22T04:39Z -->
## 2026-07-22 — CPU SLN, stale-shape recompute, nbits prefill GEMM, and stale test merges

### Land fp16/bf16 CPU SimplifiedLayerNormalization
**By:** Deckard; reviewed by Gaff 🟢
**What:** Land commit `74a80ce` extending the CPU `SimplifiedLayerNormalization` kernel to accept Float16, BFloat16, Float32, and Float64 inputs/scales by widening to f32 for RMS-style accumulation and narrowing normalized plus optional inverse-standard-deviation outputs to the declared dtype. Dtype-parameterized tests cover last-axis and multi-axis shapes.
**Why:** Half-precision Foundry exports were rejected at `input_layernorm`; the generic widen/compute/narrow path removes that CPU decode gap without model, hidden-size, or shape gates.

### Land live runtime shape recompute for elementwise broadcasts
**By:** Pris; reviewed by Leon 🟢
**What:** Land commit `79b2bfc` recomputing standard multidirectional elementwise output geometry from concrete runtime input shapes before allocation, with actionable broadcast-incompatibility errors and coverage for a `ReduceSum -> Squeeze -> Cast -> Slice -> Add` data-dependent chain.
**Why:** Loader-resolved shapes can be stale for runtime view/data-dependent chains; using live broadcast shapes unblocks GLM-5.2-tiny indexing `Add` nodes while preserving strict ONNX equal-or-one semantics.

### Land portable fp16 MatMulNBits M>1 prefill GEMM
**By:** Sapper; reviewed by Batty 🟢
**What:** Land commit `54b49eb` adding a structural CUDA fp16-activation MatMulNBits prefill path for int4/int8 block-32 weights using a portable 16x16 tiled CUDA-core GEMM with fp32 accumulation, fp16 output, implicit/explicit zero points, tail handling, and f64-oracle parity.
**Why:** Native fp16 MatMulNBits previously rejected every M>1 prompt; the new path enables native multi-token prefill while preserving the unchanged capture-safe M=1 decode GEMVs.

### Refresh stale MatMulNBits unsupported-width coverage
**By:** Hudson
**What:** Land commit `764a208` updating the CPU MatMulNBits factory rejection test to use unsupported `bits=3`, assert the current `{2, 4, 8}` contract, and add positive factory coverage for `bits=8`.
**Why:** The old test treated now-supported `bits=8` as invalid and broke the CPU suite on main after int8 support landed.

### Fold processed landed inbox notes
**By:** Scribe
**What:** Merged and deduplicated `deckard-sln-fp16.md`, `gaff-sln-fp16-review.md`, `pris-stale-shape.md`, `leon-stale-shape-review.md`, `sapper-nbits-prefill.md`, `batty-nbits-prefill-review.md`, and `hudson-stale-nbits-test.md`. Preserved active or not-yet-main GQA/VLM/specdecode/model-coverage scope and revision artifacts.
**Why:** Landed implementation and review decisions belong in the ledger; active scope, review, and revision files should remain in the inbox until their work lands.

<!-- scribe-merge-2026-07-22T03:37:44Z -->
## 2026-07-22 — GQA scalar seqlens_k and int8 fp16 default-zp test merges

### Land GQA scalar `seqlens_k` support
**By:** Deckard; reviewed by Roy 🟢
**What:** Land commit `4ceaa7b` enabling declared unit-batch scalar `seqlens_k` for structurally detected GroupQueryAttention only. The contract remains strict-by-default (`PerBatchOnly`), rejects batch>1 scalar lengths, regenerates schema metadata, and keeps CUDA graph capture safe because validation is pure CPU shape inspection with no device allocation, D2H copy, sync, or pointer rebinding.
**Why:** ORT-GenAI GQA exports may provide scalar key sequence lengths for unit-batch decode; accepting that explicit metadata contract generically unblocks Phi-4-mini and Qwen2.5-1.5B decode without broad scalar coercion.

### Land int8 fp16 implicit-zero-point GPU parity coverage
**By:** Deckard; reviewed by Tyrell 🟢
**What:** Land commit `0d618de` adding fp16 int8 block-32 MatMulNBits CUDA parity coverage when the optional zero-point graph input is omitted, with the independent reference using default zp=128. The batch also retains explicit-zero-point coverage and verifies CUDA-graph replay is bit-exact with the preceding eager output on H200.
**Why:** The implicit/default zero-point path is distinct from explicit zero-points and needs direct regression coverage for fp16 output parity and capture determinism.

### Record VLM WP1 emission review lockout
**By:** Sapper; reviewed by Leon 🔴
**What:** PR #416 / VLM WP1 emission is blocked. Sapper is locked out of revising this artifact; a different agent must derive processor operations from explicit processor config/registry entries, make position/state roles registry/config-driven, add real cached-model HF processor comparisons, and fail unsupported signatures with actionable regenerate-or-register errors.
**Why:** Although schema/port validation and CLI/metadata tests passed, emitted preprocessing programs were not runtime-correct for Qwen3-VL, Gemma4, or Phi4MM, and some roles were inferred from shape/position rather than declared metadata.

### Fold processed inbox notes
**By:** Scribe
**What:** Merged and deduplicated `deckard-int8-zp-test.md`, `roy-gqa-review.md`, `tyrell-int8-zp-review.md`, and `leon-wp1-review.md` into this ledger. Preserved active research/scope artifacts in the inbox, including `zhora-deepseek-scope.md`, `leon-vlm-scope.md`, `keaton-native-specdecode-design.md`, `pris-gqa-scalar-seqlens-plan.md`, and `chew-model-coverage-census.md` if present.
**Why:** Review verdicts, lockouts, and landed implementation decisions belong in the current ledger; active research artifacts remain available for ongoing work.

<!-- scribe-merge-2026-07-22T09:30Z -->
## 2026-07-22 — DeepSeek shape-chain, MLA conformance, and active inbox fold

### Land DS-1 generic dynamic shape-chain propagation
**By:** Chew; reviewed by Rachael 🟢
**What:** Land commit `d653879` (reviewed work `chew-79`) extending generic runtime shape-chain propagation so a dynamically resolved `Slice` can feed `Unsqueeze` and subsequent broadcast/movement. `Unsqueeze` output rank is computed as input rank plus `len(axes)`, using the ONNX domain/opset registry and no node-name keying. Native Rust DeepSeek-V2 tiny CPU E2E now generates `[42, 237, 198, 2, 186, 81, 210, 149]`.
**Why:** Dynamic output sizing must remain model-agnostic and registry-driven while covering DeepSeek-V2 decode graphs that pass shape values through movement/broadcast chains.

### Land DS-3 MLA cached-decode parity coverage
**By:** Pris; reviewed by Tyrell 🟢
**What:** Land commit `8aba045` strengthening standard Attention/MLA tests for `qk_head_dim != v_head_dim` (192 vs 128), 3-D BSH, explicit head attrs, non-empty past K/V, prefill+decode+full-seq parity, GQA (`kv=2`) and MQA (`kv=1`), with an independent scalar SDPA oracle. CPU 33/33 and CUDA 23/23 pass.
**Why:** Cached decode must preserve asymmetric QK/V head-width semantics and parity across CPU/CUDA without relying on model-specific assumptions.

### Keep generic scalar `seqlens_k` GQA support explicit and unit-batch scoped
**By:** Pris and Deckard
**What:** Preserve the long-lived scalar-seqlens implementation plan, and fold Deckard's landed decision to emit `model.attention.key_sequence_lengths.scalar_broadcast: unit_batch` only for structurally detected ORT-GenAI GroupQueryAttention exports.
**Why:** Scalar key sequence lengths should be accepted only under a declared, validated unit-batch GQA contract, not as a broad shape coercion.

### Fold remaining processed inbox decisions and reviews
**By:** Scribe
**What:** Processed and deduplicated the non-preserved decision inbox notes. Key folded outcomes: block-32 int8 MatMulNBits CUDA support and review; VLM WP1/WP5/WP6 metadata/loader/server-bundle work and reviews; Gemma4 auxiliary output binding plus structural capture guard; H200 multi-model roofline and megakernel feasibility notes; KV logical-shape and fp16 GQA decode coverage; and DeepSeek validation/review records already represented by the DS-1/DS-3 entries above. Processed files:
- `ana-fp16-next-levers.md`
- `ana-h200-baseline-roofline.md`
- `ana-megakernel-feasibility.md`
- `ana-wave2-roofline-558.md`
- `ana-wave3-roofline-691.md`
- `batty-auxbind.md`
- `chew-ds1-shape-chain.md`
- `chew-ds3-mla.md`
- `chew-leon-auxguard-review.md`
- `deckard-gqa-fp16.md`
- `deckard-gqa-scalar-seqlens.md`
- `deckard-int8-matmulnbits.md`
- `gaff-ds3-review.md`
- `gaff-kv-review.md`
- `leon-auxbind-review.md`
- `leon-auxguard.md`
- `leon-kv-logical-shape.md`
- `leon-vlm-wp5-finalize.md`
- `leon-vlm-wp5-rebase.md`
- `leon-vlm-wp5-urlfix.md`
- `luv-vlm-wp5-rereview.md`
- `luv-vlm-wp5-rereview2.md`
- `luv-vlm-wp5-review.md`
- `luv-vlm-wp6-rereview.md`
- `luv-vlm-wp6-review.md`
- `luv-wp4-review.md`
- `pris-deepseek-e2e-val.md`
- `pris-ds3-mla-conformance.md`
- `pris-gqa-fp16-review.md`
- `rachael-ds1-review.md`
- `rachael-vlm-wp5.md`
- `roy-int8-matmulnbits-review.md`
- `sapper-glm-pr404.md`
- `sapper-vlm-wp1-emission.md`
- `sapper-vlm-wp6-fix.md`
- `sebastian-gemma4-perf.md`
- `sebastian-gemma4-reprobe.md`
- `sebastian-h200-multimodel-bench.md`
- `tyrell-ds3-review.md`
- `zhora-vlm-wp5-fix.md`
- `zhora-vlm-wp6.md`
**Why:** The inbox should retain only long-lived active research/scope artifacts while merged decisions live in the current ledger.

### Preserve active research and scope artifacts in the inbox
**By:** Scribe
**What:** Left `zhora-deepseek-scope.md`, `leon-vlm-scope.md`, `pris-gqa-scalar-seqlens-plan.md`, and `keaton-native-specdecode-design.md` in `.squad/decisions/inbox/`.
**Why:** These artifacts remain active references and should not be collapsed into the ledger yet.

<!-- scribe-merge-2026-07-21T23:55Z -->
## 2026-07-21 — VLM WP2/WP3, opset-24 CUDA, ScatterElements, and DS-1

### Land VLM WP0 metadata contract and source-compatible hotfix
**By:** Sapper; hotfix by Rachael; reviewed by Luv 🟢  
**What:** Land architecture-neutral typed multimodal metadata as commit `0f6ffbd`, then make additive WP0 fields `Default`-derived in hotfix `1b66d0f` so downstream literal construction sites keep building.  
**Why:** VLM routing must be metadata-driven rather than model-flavored, and optional multimodal fields must be source-compatible as the contract grows.

### Land native CUDA opset-24 ConstantOfShape, Gelu, and OneHot
**By:** Batty; reviewed by Pris 🟢  
**What:** Land commit `ea4036d` with generic native CUDA handlers for standard-domain ConstantOfShape, Gelu, and OneHot, preserving opset-aware semantics including negative-index behavior.  
**Why:** Opset-24 Gemma/DeepSeek-style graphs should stay native instead of falling back because construction, activation, or indexing handlers are missing.

### Replace VLM every-step model bindings with a generic Kahn executor
**By:** Sapper; reviewed by Luv 🟢  
**What:** Land VLM WP3 as commit `3aec9f3`, replacing model-flavored `EmbedsStepBinding` with a metadata-driven every-step executor that topologically schedules declared inputs, outputs, and dependencies using Kahn sorting.  
**Why:** Autoregressive VLM step execution must follow the declared metadata graph, not hard-coded architecture names.

### Land DS-1 generic runtime shape propagation with bounded materialization
**By:** Deckard; revision by Holden; rereview by Pris 🟢  
**What:** Land commit `1584fb3` for DeepSeek-V2 dynamic `Slice -> Unsqueeze` shape propagation, reusing the opset-aware shape-inference registry and permitting host materialization only after dtype, rank, and element-cap gates pass.  
**Why:** Runtime output sizing should reuse the same generic ONNX shape rules as kernels while preventing unbounded host copies from hostile or accidental shapes.

### Broaden native CUDA ScatterElements dtype coverage portably
**By:** Deckard; reviewed by Chew 🟢  
**What:** Land commit `5b01a01` covering fp16/bf16/fp32/int64 data with int32/int64 indices. Serial single-threaded reduction avoids half atomics, remains SM-portable, and is CUDA-graph capture-safe.  
**Why:** Valid ONNX ScatterElements graphs should not decline native placement solely because a supported data/index dtype pairing was absent.

### Land VLM WP2 native image processor after numerics and allocation fixes
**By:** Leon; revision by Sapper; final review Pris 🟢  
**What:** Land commit `5c48ba5` for generic metadata-declared image preprocessing. The accepted path preserves bit-exact `f32::from(v) / 255.0` Divide semantics (not reciprocal multiply; 126/256 bytes otherwise differ by 1 ULP), uses `try_reserve_exact` bounded allocations, rejects degenerate dimensions, and pins patch-size-2 HF fixtures by SHA.  
**Why:** VLM processors need multi-output metadata-declared preprocessing without legacy numerical drift or unbounded metadata-derived allocation.

### Preserve review lockouts from this segment
**By:** Scribe  
**What:** Record active lockout history: WP2 had Chew 🔴, locking Leon+Chew out until Sapper revised and Pris approved; WP4 had Gaff 🔴, locking Zhora+Gaff out while Batty revises; DS-1 had Gaff 🔴, after which Holden revised and Pris approved.  
**Why:** Rejected artifacts and reviewers stay locked out for their correction cycle, while accepted third-agent revisions become the authoritative artifacts.

### Treat CUDA 13 NVRTC on H200 as current-good
**By:** Scribe  
**What:** The CUDA crate pins `cudarc` `cuda-13000` with dynamic loading, and NVRTC 13 builds and runs GPU tests successfully on H200.  
**Why:** The older belief that this host requires CUDA 12.6 NVRTC is stale and should not guide future debugging or setup.

### Additional inbox decisions folded and deduped
**By:** Scribe  
**What:** Processed non-preserved decision inbox artifacts, deduping items already represented above or in the active ledger. Folded summaries:  
- `batty-clippy-hygiene.md` — 2026-07-21: Clear engine and ORT clippy warnings; By: Batty; What: Cleared all `cargo clippy --all-targets --features cuda -- -D warnings` diagnostics in `onnx-genai-engine` and `onnx-genai-ort` without changing public APIs or runtime logic..
- `brigitte-wp3-argmax-expose.md` — 2026-07-21: Expose and verify ORT multi-row device argmax; By: Brigitte; What: Added `DeviceSampler::argmax_rows(&self, DataType, usize, usize, usize) -> Result<Vec<u32>>`, implemented by `CudaSampler` through its existing `pub(crate) CudaSampler::argmax_rows` entry point. Coverage is f32, f16, an….
- `chew-flash-tc-adjudication.md` — Chew — Adjudication: `flash_attention_f16_tc` numerics dispute (Holden vs Deckard).
- `deckard-ep-transparency.md` — Decision: Production per-op executor spans + kernel-variant & capture-rejection reasons (native EP).
- `deckard-flash-tc-fix.md` — Deckard — flash_attention_f16_tc wmma parity investigation + permanent gate.
- `fenster-fixture-fix.md` — 2026-07-21: Treat binary/textproto twins as one model; By: Fenster; What: Chose Option A. `ModelDirectory` now collapses `<name>.onnx.textproto` when the same-stem `<name>.onnx` exists and prefers the binary; distinct model names remain ambiguous..
- `gaff-clippy-review.md` — 2026-07-21: Clippy hygiene review (Batty 2a0555b); By: Gaff; What: Approved commit `2a0555b` as pure Clippy hygiene. The six-file diff contains iterator idioms, redundant-clone removal in CUDA sampler tests, a let-chain, `then_some`, literal digit regrouping, a rustdoc blank line, and….
- `holden-attn-cliff-investigation.md` — Holden — Attention "cliff at ~pos 30" investigation (native CUDA, Qwen2.5-0.5B-int4).
- `holden-wp1-verify-review.md` — Review: WP1 — Native M=K verify + rewind primitive (option b) + (c)-ready guard.
- `hudson-fixture-fix-review.md` — 2026-07-21: loader same-stem fix review; By: Hudson; What: Binary/textproto twins are correctly treated as one logical model, with the binary preferred..
- `hudson-wp3-argmax-review.md` — Hudson review — WP3-prep multi-row device argmax.
- `joshi-rmsnorm-generic.md` — 2026-07-21: Select fp16 SkipRMSNorm warp half4 by structural capability; By: Joshi; What: Generalized `skip_rmsnorm_f16_warp_896` into `skip_rmsnorm_f16_warp_half4`. The kernel now receives and uses runtime `norm_size`, iterates `norm_size / (32 lanes * 4 halves)` half4 chunks per lane, divides the sum of sq….
- `kowalski-wave4-profile.md` — 2026-07-21: Wave-4 stacked CUDA profile; By: Kowalski; What: Treat wave-4 native CUDA fp16 decode as approximately 759 tok/s at 256 tokens and 789 tok/s at 1024 tokens, with about 227 launches/token, zero CUDA-graph fallbacks, and coherent decode..
- `pris-fusion-genericity-review.md` — Review: Fusion-genericity remediation (wt-fusion-generic @ 19b3b91).
- `pris-opset24-review.md` — Kernel Review — Native CUDA opset-24 op handlers.
- `pris-rmsnorm-review.md` — 2026-07-21: RMSNorm genericity review (Joshi 53d55e1); By: Pris; What: Reviewed branch `wt-rmsnorm-generic` @ 53d55e1, which replaces the.
- `ripley-wp2-native-driver.md` — WP2 — Native speculative driver (host-argmax accept).
- `sapper-fusion-genericity.md` — Decision: CUDA wave-4 fusions gate on structure + capability, not Qwen dims.
- `sebastian-multimodel-bench.md` — 2026-07-21: H200 native CUDA multi-model benchmark; By: Sebastian; What: Current `main` (`035ad9f`) measured Qwen2.5-0.5B int4 at **771.40 tok/s median** (766.49/773.62/771.40), 1 prompt token, 256 output tokens, 5 warmups per independent process, CUDA graph + device KV + strict CUDA, and ze….
- `solveig-wp1-verify-primitive.md` — Decision: WP1 — Native M=K verify + rewind primitive (option b) + (c)-ready guard.
- `wallace-ep-transparency-review.md` — 2026-07-21: EP transparency backbone review; By: Wallace; What: Deckard's per-op executor span backbone (`exec_plan_node`) is a genuine LIVE span, and the re-instrumented kernels attach kernel-variant + capture-status reasons to it in the real native decode path — my original dead-w….
- `wallace-wp2-driver-review.md` — WP2 native speculative driver — review.  
**Why:** The inbox should hold only living research artifacts; segment decisions belong in the active ledger.

## 2026-07-20 — CPU decode: resident pool and guarded GQA row parallelism

### Keep persistent M=1 decode-pool residency
**By:** Sapper; reviewed by Luv 🟢  
**What:** Run the whole native CPU M=1 forward inside one bounded decode-pool `install`, using a worker-local, nested, panic-safe RAII residency guard so each MatMulNBits call executes inline rather than reinstalling the same pool. `ONNX_GENAI_CPU_DECODE_THREADS=0`, prefill, default-feature-off, and CUDA behavior remain unchanged. Landed on main as `cbacb75`.  
**Why:** Qwen2.5-0.5B int4 decode improved about 3–6% with bit-identical tokens. This proves install crossings were avoidable but not the dominant remaining cost. Luv verified TLS isolation, Rayon semantics, deadlock safety, feature gates, and the CPU/build test matrix.

### Parallelize sufficiently large CPU GQA attention rows
**By:** Roy; reviewed by Luv 🟢  
**What:** Parallelize independent `(batch, query_head, query_sequence)` rows with one Rayon fork-join only above a 163,840 `row × key × head-dimension` work guard; retain serial execution below it. Each task owns a disjoint output row and private score buffer while preserving each row's reduction order. Landed on main as `c391327`.  
**Why:** Short decode regressed when parallelism was unconditional. Guarded parallelism improved 512-token decode throughput by 8.6%, reduced profiled GQA time by 13.9%, and cut 225-token prefill GQA time by 88.3%, with bit-identical 1-thread/8-thread greedy output. A future coverage follow-up may force exact serial/parallel comparison for a large ragged batch.

### Retain Tier-A GQA KV copy cleanup, defer shared append-only KV
**By:** Roy; regression coverage by Pris  
**What:** Borrow contiguous f32 past caches, remove a redundant owned clone, and replace scalar cache materialization loops with contiguous slice copies. Keep attention math and the SSA output contract unchanged. Pris added f16-widening and ragged-per-batch cache-materialization regressions.  
**Why:** The cleanup is bit-identical and removes avoidable work, but measured end-to-end decode was neutral within noise. True O(1)-append shared KV requires runtime aliasing/lifecycle changes and remains deferred.

### Do not land the decode fork-join granularity prototype
**By:** Deckard  
**What:** Revert the coarser 8/12-task MatMulNBits prototype and profiling probes; no commit landed.  
**Why:** Long runs regressed 7.1–8.4%. Post-residency profiling showed serial GQA at about 20.58 ms/token exceeded MatMulNBits at about 15.51 ms/token, so reducing projection task count removed steal slack rather than solving the dominant bottleneck. Revisit only as graph-level projection fusion, after GQA.

## 2026-07-20 — CUDA fused flash attention

### Fuse standard Attention only on measured-winning shapes
**By:** Rachael; reviewed by Chew 🟡  
**What:** Add an NVRTC tiled online-softmax backend behind `AttentionKernel`, including f16 WMMA with f32 accumulation and scalar f32/f16/bf16 support for MHA/GQA/MQA, causal/non-causal attention, and additive mask planes. Auto dispatch retains Phase-2a for decode, `D>128`, unsupported layouts/features, and measured-slower long spans. Landed on main as `a67b7a5`.  
**Why:** H200 f16 S512 improved about 1.53–1.60× and removed 48 MiB score scratch; S2048 regressed heavily when forced, so fallback is part of the design. Chew found the online-softmax merge, WMMA masking/synchronization, numerics, and dispatch sound. Non-blocking coverage remains for explicit Auto fallback gates, non-multiple-of-16 f16 head dimensions, and per-batch/per-head masks.

### Fuse GroupQueryAttention prefill with distinct physical and causal origins
**By:** Bryant, corrected by Rachael after Chew rejection; final review Chew 🟢  
**What:** Reuse the shared flash kernel behind `com.microsoft::GroupQueryAttention` for measured-winning prefill. Cache append and implicit RoPE use `total_length - key_sequence_length`; attention causal masking uses the distinct query start `total_length - query_sequence_length`. The final parity matrix covers 40 scenarios across f32/f16/bf16, MHA/GQA/MQA, fresh/cached/ragged, RoPE, local window, softcap, generic non-WMMA routing, large scores, unequal Q/K lengths, and Auto fallback. Landed on main as `94fa2b6`.  
**Why:** Bryant's first revision incorrectly reused the K append origin for queries when `Sq != Sk`; Chew rejected it and locked that artifact. Rachael's revision made the failing `Sq=2,Sk=4` case pass, tightened tolerances, and preserved exact present K/V. H200 fresh Q512 is about 1.31× faster with 48 MiB scratch saved; cached/large slower shapes fall back. The corrected artifact is approved and no active lockout remains.

## 2026-07-20 — Issue #40 Phase 1 distributed-runtime foundation

### Slice 1a: shared protocol trace + ticketed non-blocking host pressure
**By:** Tyrell; reviewed by Gaff 🟡  
**What:** Add the unpublished `onnx-runtime-protocol-trace` crate with public protocol envelopes/identities and a conformance-only independent `ReplayChecker`; add `HostGovernor` ticketed pressure accounting to `onnx-genai-scheduler`. All state transitions and trace linearization points commit under one short ledger lock; waits occur only on ticket-local condition variables after capacity is atomically charged. Landed on main as `0d1d265`.  
**Why:** The implementation conforms to `PressureProtocol.tla` invariants through an independent deterministic replay campaign and snapshot invariant checks. Gaff approved with two non-blocking issues—terminal-entry reaping and cancel-granted wake-after-unlock—which were folded into slice 1b. The TLC model gate is CI-deferred because Java/TLA tooling is unavailable locally.

### Slice 1b: Communicator + in-process backend + BufferOwnership registry
**By:** Tyrell; reviewed by Gaff 🟢  
**What:** Add unpublished `onnx-runtime-comm` with the async `Communicator` trait, synchronous reference `InProcessCommunicator`, and one-lock `OwnershipRegistry` over read/write lease sets. Dropping an operation handle detaches but does not release storage; terminal completion/abort releases leases, and freed allocation IDs remain tombstoned to prevent reuse/ABA. Reuse the slice-1a trace framework and independently replay `BufferOwnership` events. Landed on main as `e4d2883`.  
**Why:** Gaff verified exactly-one-owner, conflict, release, transfer, generation/ABA, non-blocking-lock, linearization, barrier, mailbox, and deterministic-conformance obligations. Slice 1b also reaps terminal pressure entries and moves all pressure wakeups after unlock. Non-blocking follow-ups for 1c include abort waking barrier waiters, barrier-map cleanup, and documenting tombstone growth.

### Slice 1c: one topology-wide collective ordering authority — IN PROGRESS
**By:** Tyrell  
**What:** Implement direct host rendezvous collectives behind a shared `CollectiveSequencer`; keep canonical submit order independent per communicator group, use one slot for count exchange plus all-to-all-v data, freeze reduction member order with checked arithmetic and per-contribution f16/bf16 rounding, and bound free tombstones with an exact window plus allocator-proven epoch floors.  
**Why:** This maps to `CollectiveOrdering.tla`: ranks may progress asynchronously without divergent order, groups do not acquire a false global enqueue order, completion stays rank-local, and abort freezes submissions before backend wakeup. This slice is not yet landed.

### Phase-1 deferred gates and remaining phases
**By:** Scribe  
**What:** Keep the TLC model gate CI-deferred. After 1c, Phase-1 slice 1d weight residency remains pending; issue #40 Phases 2–4 remain pending.  
**Why:** The landed Rust conformance harnesses provide deterministic implementation-side evidence, but do not replace the configured CI model check or the remaining distributed-runtime roadmap.

## 2026-07-20 — Issue #40 collective ordering completion

### Land slice 1c with serialized abort wakes and broad equivalence coverage
**By:** Tyrell; reviewed by Gaff 🟢  
**What:** Land all seven in-process collectives behind one canonical per-group `CollectiveSequencer`, deterministic member-order reduction, additive independent replay checking, bounded allocation tombstones, and rank-local completion. Abort now holds each rendezvous mutex while notifying its paired condition variable, closing the review's notify-before-park race. Distributed-equals-single-device bitwise coverage spans all_reduce, reduce_scatter, all_gather, broadcast, all_to_all, and all_to_all_v. Landed as `2ffb4e4` with follow-up `128440d`.  
**Why:** Gaff found the architecture and TLA refinement sound but blocked the original revision on a rare abort-path lost wakeup. Tyrell's deterministic waiter gate proved the fix, all comm/trace/scheduler suites passed, and the broadened equivalence matrix preserves fixed-rank-order determinism. TLC remains CI-deferred.

## 2026-07-20 — CUDA graph M4 capture-safety

### Own the CUDA graph lifecycle and exercise native decode replay
**By:** Rachael and Deckard; replay coverage by Pris; reviewed by Chew 🟢  
**What:** Serialize one CUDA graph lifecycle inside `CudaRuntime`, capture/replay only on its dedicated stream, invalidate on generation/binding lifecycle changes, and split capture-end from instantiate so failed instantiation cannot leak the intermediate `CUgraph`. Native decode remains flag-gated and strict-audit: unsupported graphs fall back eagerly. A capture-safe synthetic decoder proves token-exact eager/replay parity across reset, stable addresses, O(1) scalar uploads, two captures, sixteen replays, and zero fallbacks. Landed as `637e247`, `5470c01`, `dd2d807`, and `4755575`.  
**Why:** The first Qwen test exercised only fallback and was rejected as replay evidence. The final synthetic integration test executes the real `NativeDecodeSession::decode_cuda` state machine and resolved the M4 decode-loop review blocker without weakening the all-kernel capture audit.

### Gate MatMulNBits M=1 capture safety to the proven decode path
**By:** Bryant  
**What:** Remove trailing GEMV synchronizations and advertise MatMulNBits capture compatibility only after a successful no-`g_idx`, M=1 decode warmup; prefill, grouped-index, unwarmed, and configuration-changing paths remain ineligible. Runtime D2H helpers explicitly order after the EP stream. Landed as `a210703`.  
**Why:** The proven GEMV path is allocation-free, D2H-free, and synchronization-free, while the excluded paths dequantize, allocate, or validate on the host.

### Make fixed-shape GQA decode capture-safe with detect-before-consume metadata guards
**By:** Deckard, Rachael, and Bryant; reviewed by Chew 🟢  
**What:** Persist GQA scratch and remove the trailing stream sync (`dcb4f1b`); move advancing decode metadata reads and derived lengths on-device (`77829b9`); preserve warmup rejection and add on-device replay bounds checks with sentinel no-write behavior (`82c249d`). The final shared sticky error latch poisons subsequent replay steps after any violation and is polled immediately after logits D2H, before token consumption; explicit graph reset clears it. Landed final as `ca50bae`.  
**Why:** Earlier revisions were rejected for silent clamping and then for allowing a later valid replay to resume over a skipped KV row. The final detect-before-consume latch makes invalid metadata a hard, deterministic failure while valid fixed-capacity f32 one-token replay remains byte-identical and allocation-free.

### Make four normalization variants capture-safe
**By:** Roy; reviewed by Chew 🟢  
**What:** Remove trailing synchronizations from LayerNormalization, RMS/SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization, and SkipLayerNormalization. Keep SkipSimplified broadcast metadata in a mutex-protected, shape-keyed persistent cache and permit capture only after successful single-group warmup. Landed as `6184d82`.  
**Why:** The warmed decode paths now have stable metadata and no per-step allocation, free, upload, host read, or stream synchronization; the full CUDA suite and direct capture/replay byte-parity test passed.

### Bind elementwise capture eligibility to exact warmed signatures
**By:** Sapper and Deckard; reviewed by Chew 🟢  
**What:** Make supported unary and binary floating-point decode kernels capture-safe using persistent broadcast metadata and removed trailing synchronizations. Replace the initial boolean eligibility gate with mutex-protected exact dtype/entry and shape signatures; prefill, i64, errors, and signature changes remain ineligible. Landed final as `85b6f4e`.  
**Why:** Chew rejected the boolean gate because a warmed kernel could later execute a different dtype or shape during capture. Exact signatures close that TOCTOU while preserving numerics and the approved persistent-metadata design.

## 2026-07-21 — CUDA graph M4 end-to-end validation

### Real Qwen2.5 int4 decode captures with zero fallbacks
**By:** Rachael; reviewed by Chew; smoke correction by Pris 🟢  
**What:** Seed unresolved persistent external input/output physical shapes only during capture, keeping eager shape resolution and binding-signature invalidation intact. Constant/Shape metadata reuse and capture-safe integer Sub, ReduceSum, and Gather complete the real Qwen graph while device-side GQA/Reduce/Gather guards still latch errors before token consumption. After Chew caught stale fallback assertions, Pris updated the H200 smoke to require one capture, 62 replays, zero fallbacks, and no fallback reason. Landed as `dda3b25`, `13c094a`, and `42b71f7`.  
**Why:** Qwen2.5-0.5B int4 now captures end to end with token-exact graph ON/OFF parity and zero fallbacks: 70.33 versus 19.99 tok/s at 256 tokens (+251.8%), and 24.25 versus 11.73 tok/s at 1024 tokens (+106.7%). This validates the complete M4 capture-safety track on the real model.


## 2026-07-21 — Perf campaign reconciliation

### H200 native CUDA decode target and profiling baseline
**By:** Ana and Rachael  
**What:** Use ORT GenAI H200 Qwen2.5-0.5B int4 steady-state decode as the performance target: **657.34 tok/s** at 256 tokens (667.43 tok/s at 1024). Native progressed from about **73 → 145 → 192 → 201 tok/s**, but f32 Sq=1 GQA remained dominant: 70.5% of GPU time over 256-token decode and 82.7% over 16-token decode.  
**Why:** GEMV/argmax work is valuable but insufficient alone; the next high-leverage path is replacing serial f32 decode attention and then wiring/validating fp16 flash decode.

### Retile MatMulNBits decode GEMV and approve the result
**By:** Royb; reviewed by Wallace 🟢  
**What:** Retile the M=1 accuracy-level-4 symmetric block-32 CUDA MatMulNBits path, quantizing the f32 activation once with matching warp absmax/round/clamp/scale semantics. Wallace approved Roy's `5dbcbbb` retile.  
**Why:** This moved native decode from roughly 145 tok/s to about 192 tok/s while preserving numerics, but still leaves a large gap to Ana's 657 tok/s ORT target.

### Keep device-side greedy argmax after Batty's rebase repair
**By:** Mariette and Batty; reviewed by Joi 🟢  
**What:** Add allocation-free CUDA f32 greedy argmax with lowest-index tie behavior matching the host sampler. Joi rejected Mariette's rebased `c12e74f` because `DecodeCudaState::run_one_token` was called without the new `TraceContext`; Batty fixed the call and Joi approved `cdf62a0`.  
**Why:** The fixed path builds and measured about **200.97 tok/s**, removing the host argmax bottleneck without changing token selection.

### Land fp16 flash-decode as kernel-only first, then dormant dispatch wiring
**By:** Sebastian; reviewed by Bryant and Holden 🟢  
**What:** Add a capture-safe fp16 flash-decode GQA attention kernel as kernel-only commit `9c6f36b`, approved by Bryant. Wire it through a dormant fp16 dispatch branch at `521438e`, approved by Holden, gated by `q.dtype == Float16` and supported `(q_seq, dim)` while leaving the f32 path first and unchanged.  
**Why:** Split landing keeps the kernel independently reviewed and lets dispatch be enabled safely only for supported fp16 decode shapes.

### Direct fp16 activation × int4 GEMV remains a separate optimization track
**By:** Royb  
**What:** Prototype direct fp16-activation × int4 MatMulNBits GEMV on `wt-fp16-matmul` (`6a1daa2`) to avoid the int8 quantization pass.  
**Why:** This is distinct from fp16 flash attention and should be validated as a separate GEMV optimization before promotion.

### Sequence zero-copy design needs a second Deckard revision
**By:** Zhora and Deckard; reviewed by Luv 🔴  
**What:** Zhora's zero-copy Sequence tensors use shared allocation views with dtype/shape/layout/offset metadata. Luv rejected `ddae7d0`; Deckard closed the original public-output/runtime blockers with `SessionOutput::{Tensor, Sequence}` and related fixes, but Luv's re-review still rejected `cf8888b`.  
**Why:** The direction is acceptable, but remaining correctness/review blockers mean the Sequence zero-copy change is not approved yet.

### Runtime string tensors must use a dedicated host storage variant
**By:** Batty  
**What:** Represent runtime strings with `TensorStorage::{Raw, Strings(Vec<String>)}` or equivalent, expose safe `StringTensorView`/`StringTensorMut`, and never cast byte/device storage to `String`.  
**Why:** String tensors are host-owned structured values, not raw numeric buffers; exhaustive storage keeps executor behavior type-safe.

### PressureProtocol scaffold/fix path and current rejection state
**By:** Sapper, Roy, Deckard, and Pris; reviewed by Holden and Freysa 🔴/🟢 mixed  
**What:** Sapper/Roy added HostGovernor pressure envelopes and replay extension points; Holden rejected the first scaffold until actor ordering was scoped by `(HostId, ActorId)`, which Deckard fixed. Freysa rejected Sapper's HostGovernor revision, locking Sapper out and assigning the fix to Batty; Roy repaired release integrity by retaining authoritative allocations in `Claimed` and enforcing deterministic scheduling. Freysa's 2026-07-21 re-review still rejected `3207c25` because the branch/diff was not review-clean. Pris strengthened forged-release and cancellation synchronization regression tests.  
**Why:** Credit integrity and deterministic admission are the right design constraints, but the pressure implementation is not approved until reviewed from a clean branch with the fixed protocol evidence.

### Graph-capture transparency requires structured reasons across three axes
**By:** Coordinator and Gaff; reviewed by Chew  
**What:** All EPs must surface structured trace reasons for kernel non-selection and graph-capture non-capturability; transparency has three axes: op claim, kernel-variant selection, and capture support. Gaff added `CaptureSupport::{Supported, Unsupported { reason }}` and default compatibility adapters; Chew reviewed the structured reason-carrying design.  
**Why:** Silent bool declines make performance debugging impossible; traces must explain both variant choice and capture segmentation/fallback.

### Decouple CUDA EP claim from segmented graph capture
**By:** Coordinator and Tyrell  
**What:** CUDA EP should claim/run supported subgraphs even when only maximal segments are capturable, interleaving captured runs with eager CUDA runs for non-capturable nodes.  
**Why:** Capturability is an execution scheduling property, not an EP ownership property; partial segmented capture preserves CUDA placement without all-or-nothing fallback.

### Cross-platform support must include Windows ARM64
**By:** Coordinator; audit by Deckard  
**What:** Treat `aarch64-pc-windows-msvc` as a required target alongside Windows x64, macOS x86_64/arm64, and Linux x64. Deckard also flagged truthful CUDA selection, OS-aware library discovery, updated CUDA-12 CUDART candidates, pip/Conda NVIDIA discovery, and preventing Python from advertising CUDA while executing CPU.  
**Why:** Packaging and runtime probing must match the documented support matrix and actual execution provider behavior.

### Publishability of onnx-rs remains required
**By:** Leon  
**What:** Keep `onnx-rs` publishable to crates.io with package metadata and publish workflow coverage.  
**Why:** It is the ONNX standard-library crate for Rust in this workspace and must remain releasable.

### Capture-safe Sq=1 GQA decode kernel approved as prior f32 stepping stone
**By:** Sebastian; reviewed by Bryant 🟢  
**What:** Bryant approved `b6ada01`, a capture-safe warp-parallel Sq=1 GQA decode attention kernel for supported `head_dim <= 128` with zero CUDA-graph fallback.  
**Why:** This was a correct f32 decode-attention stepping stone before the later fp16 flash-decode path.

## 2026-07-21 — fp16 decode, transparent fallback, cross-platform loading, and trace cost

### Land coherent end-to-end fp16 native CUDA decode
**By:** Sebastian; component work by Mariette, Leon, and Roy; reviewed by Bryant, Wallace, and Holden 🟢  
**What:** Thread fp16 activations, KV, logits/argmax, normalization, RoPE, attention, and direct fp16×int4 MatMulNBits through native decode while retaining dtype-gated f32 paths. Leon fixed the rejected fp16 LayerNorm shared-memory reuse race before Bryant approved the normalization/RoPE path. Landed as `c8741ba`.  
**Why:** H200 Qwen2.5-0.5B int4 reached about **344 tok/s** with coherent tokens, CUDA graph capture, and zero fallbacks, up from the approximately **200 tok/s** f32 path; f32 remained unregressed near 200 tok/s.

### Make CUDA-to-CPU fallback observable and optionally strict
**By:** Deckard; reviewed by Batty 🟢  
**What:** Retain a structured `ExecutionProviderFallbackReport`, emit an initialization warning when CUDA declines force whole-session CPU execution, and make `ONNX_GENAI_REQUIRE_CUDA=1` reject that fallback. Landed as `3a8eebe`.  
**Why:** Device selection must not silently advertise CUDA while executing on CPU; callers now receive node/op/reason detail and can opt into strict CUDA-only behavior.

### Use OS-aware CUDA and CUPTI dynamic-library discovery
**By:** Leon and Roy; reviewed by Pris 🟢  
**What:** Select CUDA driver/runtime/library and CUPTI candidates by operating system, including Windows DLL names and pip/Conda layouts. Treat Windows ARM64 as gracefully unavailable before probing x64-only NVIDIA libraries. Landed as `2466016` and `8cd36c3`.  
**Why:** Cross-platform probing must fail normally rather than panic or attempt incompatible binaries. CUPTI discovery remains local to the tracer to avoid an inverted dependency on the CUDA EP.

### Emit per-op CPU bytes/FLOPs only for active trace spans
**By:** Rachael, Gaff, and Deckard; reviewed by Zhora 🟢  
**What:** Annotate major CPU kernel spans with logical tensor bytes and documented FLOP estimates, lazily computing metrics only when a span is active. Keep tracing optional and propagate the `tracing` feature through `bench-native` and `native-backend`. Landed as `61f4d2c`.  
**Why:** Profiles gain arithmetic-intensity and bandwidth inputs without imposing tensor scans, formula work, JSON allocation, or tracer dependencies on default non-tracing builds.



## 2026-07-21 — CI hardening and native CUDA decode wave 1–2

### Cover every offline crate and make warnings blocking on all portable targets
**By:** Batty and Gaff; Windows ARM64 revision by Deckard; reviewed by Hudson 🟢  
**What:** Classify all 38 workspace members by default normal+dev dependencies, explicitly test and cover all 27 pure-offline crates, and enforce blocking rustc and Clippy warnings (`RUSTFLAGS="-D warnings"` and `-- -D warnings`) rather than advisory lanes. The portable matrix retains Linux x64, Windows x64, and macOS ARM64 and adds native Windows ARM64 on `windows-11-arm`/`aarch64-pc-windows-msvc`, with the same 26-crate portable test set and an ARM64 Clippy gate; `mlas-sys` remains Linux-only, while native-ORT and CUDA crates stay outside offline execution. Formatting remains advisory pending the repository-wide sweep.  
**Why:** CI now covers the full offline workspace without triggering ORT downloads, and warnings fail builds across supported portable targets. The final 27-crate Linux lane passed 1,921 tests with 0 failures and 8 ignored; Hudson approved after Deckard closed the initially missing Windows ARM64 gate.

### Keep the measured wave-1 decode optimizations capture-safe
**By:** Leon, Tyrell, Deckard, Sebastian, and Roy  
**What:** Use persistent two-pass multi-block greedy argmax; segment CUDA graphs into maximal capturable runs around eager CUDA seams while retaining whole-subgraph EP ownership; abort/drain failed mid-segment capture before reset; use true multi-CTA split-K fp16 flash decode; and retain Roy's coalesced direct fp16×int4 GEMV retile. All paths preserve fixed device addresses, token semantics, and zero-fallback graph replay.  
**Why:** These changes removed launch/occupancy and GEMV bottlenecks without regressing correctness: argmax reached about 368 tok/s, split-K attention about 398 tok/s at 256 tokens (about 390 at 1024), and the GEMV retile about 423 tok/s. Segmented capture now recovers cleanly from invalidated streams instead of wedging later inference.

### Fuse the single-token GQA preparation chain
**By:** Rachael; reviewed by Holden 🟢  
**What:** For eligible `Sq=Sk=1` aliased fixed-capacity decode, fuse QKV split, query relayout, K/V append, and Q/K RoPE into one kernel and write attention output directly in BSH layout. Keep metadata preparation separate to preserve the capture poison/latch protocol; all other shapes retain the unfused path.  
**Why:** Prep launches fell 75% (192→48 per token), bit-exact fused/unfused and capture tests passed, and H200 throughput rose from about 557 to 615 tok/s with zero fallbacks.

### Use warp-shuffle fp16 skip-RMSNorm
**By:** Sapper; reviewed by Wallace 🟢  
**What:** Replace the fp16 shared-memory reduction tree with a single-warp packed-half2/half4 shuffle reduction, specializing hidden size 896 while retaining a tail-safe generic fp16 path; f32 kernels remain unchanged.  
**Why:** The hot kernel fell from about 6.20 to 5.07 µs/call and stacked decode reached about 579–583 tok/s with identical tokens, full CUDA tests passing, and zero graph fallbacks.

### Specialize the fp16 down-projection GEMV and accept the stacked ORT win
**By:** Luv; reviewed by Pris 🟢  
**What:** Route only `K=4864, N=896, block_size=32` with fp16 scales to a 256-thread, eight-column K-parallel GEMV that stages the activation in permuted half2 shared memory; all other shapes retain the general kernel.  
**Why:** The down-projection kernel fell from about 10.24 to 7.28 µs/call with parity within fp16 tolerance and identical greedy tokens. Stacked with GQA fusion and RMSNorm, native H200 decode reached **663–672 tok/s**, beating the **657 tok/s ORT GenAI** reference with zero fallbacks.

### Require SM-portable correctness and performance for every CUDA EP kernel
**By:** Coordinator directive; validated in wave-2 reviews by Holden, Wallace, and Pris  
**What:** Every `onnx-runtime-ep-cuda` kernel must remain correct and performant across supported NVIDIA SM architectures, not merely `sm_90`. Dispatch must derive the live architecture dynamically, avoid unguarded SM90-only features, keep resource use within portable limits, and preserve capable fallbacks or variants where architecture-specific tuning is necessary.  
**Why:** H200 wins are not acceptable if they break or materially strand devices such as RTX 4060 (`sm_89`). Wave-2 kernels use broadly available primitives and do not raise the minimum architecture.

## 2026-07-21 — Native CUDA decode wave 3 and CUDA CI

### Use 16-way split-K for long-context fp16 GQA decode
**By:** Sebastian; reviewed by Holden 🟢
**What:** Raise fp16 flash-decode `MAX_SPLITS` from 8 to 16, retaining device-side capture-safe split selection, deterministic fixed-order merging, and the single-stream shared-scratch invariant. Landed as `3b972bf`.
**Why:** Independent H200 review measured 1024-token decode improving from about 647 to 693 tok/s (+7.1%) while 256-token throughput remained flat, with identical greedy tokens, zero graph fallbacks, bounded 2.03 MiB scratch, and no SM90-only dependency.

### Fuse SwiGLU SiLU and multiply in one CUDA kernel
**By:** Mariette; reviewed by Pris 🟢
**What:** Fuse eligible equal-shape, single-consumer `Mul(Silu(gate), up)` patterns into one capture-safe f32/f16/bf16 pointwise kernel, preserving separate fallback paths and kernel-variant trace reasons. Landed as `12e48b8`.
**Why:** The fusion halves activation launches from 48 to 24 per token and improved authoritative 256-token H200 decode from about 673 to 689 tok/s, with identical tokens, zero graph fallbacks, full CUDA parity, and portable primitives suitable for sm_89.

### Record the stacked wave-3 performance baseline
**By:** Kowalski
**What:** Treat the fresh shared-H200 re-profile as the current wave-3 baseline: median throughput about 691 tok/s at 256 tokens and 712 tok/s at 1024 tokens, with zero CUDA graph fallbacks. Recorded in `docs/PROGRESS.md` by `f42ca3f`.
**Why:** The stacked GQA split and SwiGLU fusion gains reproduce together, remain coherent, and place native CUDA decode above the 657 tok/s ORT GenAI reference at 256 tokens.

### Gate CUDA EP Clippy warnings in CI
**By:** Gaff; reviewed by Wallace 🟢
**What:** Clear all 21 existing `onnx-runtime-ep-cuda` Clippy warnings without adding allows, remove no-op explicit drops of non-owning `TensorMut` views, and add `cargo clippy -p onnx-runtime-ep-cuda --features cuda -- -D warnings` to the `cuda-compile` job. Landed as `22ec87e`.
**Why:** CUDA EP warnings are now blocking in CI. Review verified the lint rewrites and drop removals preserve behavior and ownership, with builds, tests, Clippy, YAML parsing, and a zero-fallback performance sanity run passing.


## 2026-07-21 — Native CUDA decode wave 4

### Fold batch-1 GQA metadata into fused decode preparation
**By:** Luv; reviewed by Holden 🟢  
**What:** For eligible batch-1, `Sq=Sk=1`, fixed-capacity aliased-device-KV decode, derive GQA metadata inside each fused prep CTA and have block 0 write the attention arrays; unsupported shapes retain the separate metadata kernel. Landed as `bd30e6c`.  
**Why:** The change preserves latch-first poison propagation, all bounds/error bits, sentinel/no-write behavior, capture safety, and SM portability while removing 24 launches/token. Independent H200 review measured roughly 691→710 tok/s at 256 tokens with exact tokens and zero fallbacks.

### Fuse MatMulNBits-adjacent QKV bias and paired gate/up SwiGLU
**By:** Rachael; reviewed by Pris 🟢  
**What:** Fold eligible QKV bias Adds into the MatMulNBits epilogue with exact two-op fp16 rounding, and collapse the validated Qwen 0.5B gate/up projections plus SwiGLU into one paired capture-safe kernel. Strict initializer, shape, dtype, consumer, and graph-output gates preserve unfused fallback. Landed as `102fee9`.  
**Why:** GPU bit-exact tests and end-to-end greedy tokens match the two-op baseline, with zero graph fallbacks and portable primitives. Stacked on the GQA metadata fold, H200 reached about **759 tok/s at 256 tokens** and **789 tok/s at 1024 tokens**, saving about 72 launches/token.

### Drop the CUDA replay binding-cache prototype — DEAD END
**By:** Deckard  
**What:** Do not merge or re-attempt commit `14a1d8f`, which cached validated device-I/O metadata and raw external addresses for CUDA-graph replay.  
**Why:** Two paired H200 measurements showed only **+0.23%** (+1.60 tok/s), below the 0.5% noise threshold, while the exact-identity/raw-address predicate adds correctness sensitivity on the replay hot path. Revisit only with materially stronger isolated evidence and a safer design.

### Keep Ana wave-3 roofline as the current roofline of record
**By:** Scribe  
**What:** Preserve `.squad/decisions/inbox/ana-wave3-roofline-691.md` as the current roofline artifact: wave 4 achieved about **759 tok/s**, within its **750–790 tok/s** ceiling.  
**Why:** The artifact remains the authoritative lever ranking and ceiling analysis after wave-4 validation.
