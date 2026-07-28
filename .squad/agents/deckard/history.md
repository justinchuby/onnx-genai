# deckard — History

## Condensed history through 2026-07-18

- Systems developer on onnx-genai Rust runtime and ORT2 tracks. Delivered and reviewed loader, shape-inference, IR dtype, EPContext, encoder, external-data safety, and CPU/CUDA execution work.
- Repeated review practice: preserve model-agnostic dispatch, fail closed at claim time, use checked arithmetic, maintain byte-exact serialization, and require precision-sensitive tests.
- Owned revisions after reviewer lockouts for shape inference, IR dtype, EPContext writer, and the 2026-07-19 CPU reduction and activation dtype waves.
- Shared lesson: parallel commit-producing work requires separate worktrees; reviewer rejection transfers ownership and must be recorded.

## 2026-07-19T07:42:20Z — CSA B2 landing

- Delivered device ratio-128 compression plus device-resident FP8 cache/carry in `2f5f5e9`; Chew’s review was 🟡 APPROVE-WITH-NITS and the change landed to `main`.

## 2026-07-19T07:42:20Z — CSA B5 review and landing

- Authored the B5 ratio-4 fused candidate assembly. Chew rejected the initial slice for the five-output ratio-4 dispatch bug; Roy corrected the routing and landed `1ddf01b`, with 19/19 H200 parity tests approved.

## 2026-07-19T07:42:20Z — CSA B5 review and landing

- Authored the B5 ratio-4 fused candidate assembly. Chew rejected the initial slice for the five-output ratio-4 dispatch bug; Roy corrected the routing and landed `1ddf01b`, with 19/19 H200 parity tests approved.

- 2026-07-19T12:40Z: Root-caused CUDA token-index-10 drift to SkipSimplifiedLayerNorm RMS FMA contraction; fix already landed in de3c556 and verified at ccf994c. Logged cudarc cuda-12060/cuda-13000 feature-unification build conflict as backlog.

## 2026-07-19T13:10Z — cudarc CUDA-version unification
Fixed the cudarc CUDA-version-feature conflict blocking `onnx-genai-engine --features cuda,native-backend`: ORT keeps CUDA 12.6 as a weak default, while engine disables ORT defaults and selects CUDA 13.0 to align with `onnx-runtime-ep-cuda`. Landed to main as `db3f733`; builds passed and native CUDA Qwen decode parity was revalidated for 64 tokens.
## 2026-07-19T14:10Z — Bitwise/Hardmax lockout revision
- Revised Pris's rejected artifact: fp16/bf16 Hardmax plus stronger bitwise broadcast/rejection and invalid-axis tests. Luv 🟢 approved `7fe8961`; landed as `0b38d59`.


- **2026-07-19T16:15:00Z — CPU-EP fixes:** Corrected omitted-vs-present-empty reduction axes semantics (`6e97ee6`) after Chew’s rejection; also widened Selu/ThresholdedRelu dtype paths, with Sapper subsequently correcting f64 precision (`39edb76`).


## 2026-07-19T18:20:00Z — CPU-EP op coverage 936→975

- Corrected SpaceToDepth DCR ordering and pooling ceil-mode sizing (`014cf02`); also authored AffineGrid/Col2Im/CenterCropPad (`8e49948`).


## 2026-07-19T20:10Z — CPU-EP op coverage Batch 4

- Fixed Pris's rejected EyeLike artifact with checked diagonal arithmetic and checked dtype conversion (`114180e`); Luv approved.
- Authored GridSample 2-D/3-D coverage (`1f63750`); Gaff rejected opset-16 rank-5 acceptance, locking Deckard out before Sapper's approved correction.

## 2026-07-19T18:05Z — DeepSeek-V2 tiny E2E

- Validated the shared MLA + MoE path with a tiny fp32 export: prefill plus eight decode tokens completed without runtime changes.
- Added gated engine coverage (`0caaf32`) and the Mobius export helper (`2b629cc`); Gaff approved both.
- DeepSeek-V4 remains blocked upstream by the missing usable reference configuration/export artifact.

- 2026-07-19: ConvTranspose CPU kernel landed as 7219025 with 11 conformance tests; Gaff approved. Restored DeepSeek grouped top-k routing after QMoE regression (cd782dd), enabling Chew approval. Unique String attempt exposed runtime-layer UB and was superseded by safe removal. MLAS-style SIMD GEMM port remains in progress on `deckard/mlas-gemm`.


### 2026-07-20 — Vendored MLAS CPU-GEMM parity

Recorded the MLAS vendoring spike (`556b0d8`) and multi-threaded Rayon hook (`8764b3d`); provenance was corrected in `ee7a6cd`.

## 2026-07-20T05:20:00Z — MLAS int4 and PackedB milestones

- Landed MLAS PackedB reuse and MLAS SQNBitGemm wiring for CPU MatMul/MatMulNBits (`3eed80a`); f32 direct-output and feature reachability were completed in the same milestone batch. Gaff-51 reviewed the SQNBit change 🟢; int4 decode improved ~1.9× and prefill up to 9.5×.


## 2026-07-20T07:15Z — M-based MLAS int4 routing

- Landed `4bb98be`: M-based `NXRT_SQNBIT_PREFILL_MIN` routing keeps hand int4 for M=1 decode and MLAS for prefill; gaff-52 reviewed 🟢.


## 2026-07-20T13:35:00Z — Multistream performance and issue #40

- Investigated coarser CPU decode fork-join granularity, measured 7–8% regressions, reverted the prototype, and established GQA (20.6 ms) as larger than MatMulNBits (15.5 ms) post-residency.

## 2026-07-21T03:15:00Z — CUDA graph M4 validated
- Fixed CUDA graph handle ownership, persisted GQA decode scratch, hardened replay metadata bounds, and replaced elementwise boolean capture gates with exact warmed signatures (`5470c01`, `dcb4f1b`, `82c249d`, `85b6f4e`).

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.

## 2026-07-21T05:40:00Z — fp16 decode and cross-platform reconciliation

- Landed structured CUDA-decline/whole-session CPU fallback reporting and strict `ONNX_GENAI_REQUIRE_CUDA` enforcement (`3a8eebe`); Batty approved. Also propagated optional CPU tracing to native consumers in `61f4d2c`.


## 2026-07-21 — Wave-2 and CI milestone
CI now covers all 27 offline crates with warnings-as-errors and native Windows ARM64. Capture-safe native fp16 CUDA decode wave 2 stacked GQA prep fusion, warp-shuffle RMSNorm, and specialized down-projection GEMV on wave 1, reaching 663–672 tok/s on H200 versus ORT GenAI at 657, with zero fallbacks. All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.

## 2026-07-21T13:15:00Z — Replay binding cache dropped
- Evaluated capture-generation binding metadata caching; tests passed but paired runs gained only +0.23%. The raw-address correctness-sensitive hot-path change was not merged and is recorded as a dead end not to re-attempt without stronger evidence.
- 2026-07-21T23:55Z — DS-1 generic Slice→Unsqueeze shape propagation landed after Holden bounded materialization and Pris approved; ScatterElements dtype expansion also landed.
## 2026-07-22T12:00:00Z — Luv Phase 0 review
- Independently reviewed Luv's partial-CUDA-graph capture path-kind change at `3c94a57`; approved 🟢 GREEN after confirming additive behavior, correct structural seam mapping, model-agnostic dispatch, exhaustive matches, and clean fmt/clippy/tests.

### 2026-07-22T14:59:36+0000 — WP-B landed
WP-B landed: Deckard's intermediate WP-B3 revision fixed raw membership/default classification but was superseded by Sapper's v3 raw signature fix.
- 2026-07-24T16:04:31Z: Freshly reviewed Keaton's GLM-4-9B native decode-coherence lock: two deterministic GPU-4 golden runs, perturbed-output rejection, native-only comparison semantics, and documentation all verified. 🟢 APPROVE; landed as `13af95d7`.

## 2026-07-26T19:45:52Z — Scribe update

- Spawned as deckard-22 on `perf/cuda-next-wave` for profile-first portable CUDA decode-performance work; outcome pending.

## 2026-07-26T20:00:00Z — Scribe update

- 2026-07-26T20:00:02Z — Delivered portable device-gated symmetric int4 GEMV split-K; after Sebastian fixed the ineffective coverage shape, PR #203 merged to main at `b80a8c83`.

## 2026-07-27T03:45:00-07:00 — PR #227 FP16 GEMV review follow-ups

- Took Chew's approved non-blocking C1/C2 follow-ups from Iran's FP16 CPU EP work: documented the `fcvtl`/`fcvtn` inline-asm rationale and future intrinsic replacement path, tightened FP16 GEMV tolerances to 1e-4 relative / 1e-5 absolute, and exercised model-scale parity at 1/3/7/11 workers.
- Guard proof: perturbing the FP16 batched accumulator by +0.001 made `f16_col_parallel_gemv_matches_reference` fail at max error 0.0010000467 against the new 0.00001 limit; restored code passed the repeated targeted tests and full CPU EP suite.

## 2026-07-27T07:35:00-07:00 — PR #227 review fixes (cache test + SiLU doc)

- Fixed tautological cache-reuse assertion in `matmul.rs`: moved pointer capture before the second `execute()` so the comparison actually spans the call. Guard-break proof: fresh-kernel substitution made the assertion fail with distinct pointers (0x1030b68d0 ≠ 0x1030b6860).
- Fixed conflicting SiLU accuracy claim in `activations.rs`: slice-level doc now reads "~28 ULP worst-case" matching the implementation comment. Grep-confirmed no other "1 ULP" exp-accuracy claim survives in the crate.
- Full verification: fmt ✅, clippy aarch64 ✅, clippy x86_64 ✅, 906 tests passed ✅, NEON SDPA dispatch confirmed ✅.
## 2026-07-27T13:12:20+00:00 — Roadmap wave-5

- Under Moss's lockout, repaired PR #266 ReduceLogSumExp numerical stability with a dedicated two-pass reduction; Ferro approved and the PR merged.

## 2026-07-27T16:44:54Z — Wave 8 update
- Took ownership of PR #276 / #87 fix cycle after Ferro REQUEST-CHANGES; must address GPU test build break and WAR-safe driver semantics/docs.
## 2026-07-27T12:30:00-07:00 — PR #279 CUDA fallback semantics

- Adjudicated Copilot's CUDA documentation findings against source. `ONNX_GENAI_EP=cuda` without `onnx-genai-ort/cuda` is rejected at ORT session creation, runtime CUDA-provider unavailability is also a hard session error, and `ONNX_GENAI_REQUIRE_CUDA=1` only gates native CUDA node-level fallback to CPU after CUDA is compiled and selected. Corrected CLI/cargo/CAPI/Python docs and comments; verified `cargo tree`, `cargo fmt --check`, and `cargo build -p onnx-genai-cli`.
### 2026-07-27 — Runtime capability inventory for REPL design

Authored `docs/research/cli/04-runtime-capability-inventory.md` as Deckard. Key finding: the runtime has strong low-level primitives (paged KV CoW fork, rewind/checkpoint, prefix reuse, speculative stats, continuous batching), but the REPL currently drives mostly CLI-side chat history rather than engine persistent sessions. Session fork and real undo/rewind need new engine APIs; many other high-value REPL surfaces are CLI wiring over existing APIs.
## 2026-07-27T16:44:54Z — Wave 8 update
- Took ownership of PR #276 / #87 fix cycle after Ferro REQUEST-CHANGES; must address GPU test build break and WAR-safe driver semantics/docs.

## 2026-07-27T16:44:54Z — Wave 9 update
Fixed PR #276 after Ferro rejection: build break plus driver-enforced WAR fence/neutering proof; re-review approved and merged as 9ab24fa5.

## 2026-07-27T14:15:00-07:00 — Runtime fork/rewind API

- Added public engine APIs for persistent-session checkpoints and KV rewind (`checkpoint_session`, `restore_session`, `rewind_session_by`, `rewind_session_to`) using the existing target/draft speculative rewind helpers.
- Reserved `fork_session` behind an unconstructible public `SessionForkCapability`; current backends return no capability, and the internal path still fail-closes until decoder runner state can be safely cloned/imported without deep-copying or aliasing KV.
- Documented the API/cost model/invariants/backend matrix in `docs/research/cli/06-fork-rewind-api.md` and added model-free engine unit coverage plus `proptest` randomized fork/rewind/append/remove refcount coverage for paged KV.
## 2026-07-27T14:56:47-07:00 — PR #287 backend flag lockout revision

- Fixed Batty's rejected CLI backend flag revision: profile output, REPL /session, bare /backend, and /stats now use the loaded engine's resolved backend; auto is only shown as a requested backend when it differs.
- Kept --backend on transcribe deliberately because speech transcription drives the same autoregressive pipeline decoder; added parser and invalid-backend coverage plus README documentation.
- Verified cargo build -p onnx-genai-cli, cargo test -p onnx-genai-cli --lib, cargo fmt -p onnx-genai-cli -- --check, cargo clippy -p onnx-genai-cli --all-targets -- -D warnings, and cargo build -p onnx-genai-server.
## 2026-07-27T19:35:00Z — Roadmap wave update
- Fixed PR #288 tests after Moss lockout: LogSoftmax overflow-stability is value-falsifiable; BitShift width guard is locked by source-contract test.
