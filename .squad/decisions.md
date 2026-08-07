# Decisions — live standing directives

Last consolidated: 2026-08-07T00:30:00Z (Scribe C1 diagnosis batch; inbox merged; no size compaction needed)

Standing governance rules and active directives. Full narrative is archived in `.squad/decisions-archive/2026-07.md`, `.squad/decisions-archive/2026-08.md`, and older `.squad/decisions/archive/` files.

This compaction preserved the complete pre-compaction live file in `.squad/decisions-archive/2026-08.md` under "Live decisions snapshot before #695/#700 compaction". Processed inbox drops archived there: cohaagen-695-hybrid-cache-fix.md, cohaagen-qmoe-route-parallel.md, copilot-contract-decisions-q2-q12.md, copilot-plugin-c-abi-everywhere.md, deckard-645-cached-dense-identity.md, harry-700-hybrid-cache-review.md, quaid-676-oracle-testfix.md.

## Ledger health rule

Archive by SIZE, not age. Age-only no-ops during high-volume campaigns because most entries are recent, so the live file can exceed spawn-budget while "older than N days" matches nothing. When over the gate, preserve full history in `.squad/decisions-archive/{YYYY-MM}.md`, dedupe rebase-reintroduced sections, and keep live `decisions.md` to standing directives plus pointers. Concurrent Scribe runs are a structural hazard; assemble from inbox drops and check `git log origin/main..HEAD` before committing.

## Current active wave — 2026-08-06

### Hybrid Mamba continuation correctness fixed (#695/#700); ORT follow-up #701

**By:** Cohaagen (fix), Harry (review), Coordinator (merge/follow-up)

**What:** PR #700 fixed issue #695 by making native host/device KV-mirror prefix reuse unsupported whenever the decoder has recurrent state. `supports_host_kv_mirror` and `supports_device_kv_mirror` now return false for `has_recurrent_state()`, forcing full recompute for hybrid Mamba/attention models instead of reusing attention KV without the recurrent state. Single-shot behavior stays byte-identical. Tests added: always-on gate unit coverage plus an env-gated GPU continuation regression where reused argmax matches the fresh oracle token `33803`.

**Review:** Harry approved #700. Minor residual: the ORT paged-reuse path has a similar hybrid guard concern (`kv_bridge.rs:407`); coordinator filed tracking issue #701.

### 35B-A3B QMoE performance and oracle follow-ups

**By:** Cohaagen, Quaid, Harry, Coordinator

- Native sparse QMoE for 35B-A3B shipped via #625/#676. The correct QMoE continuation token is `33803`; dense int4 token `5342` is the low-precision outlier.
- QMoE router parallelization (`qmoe_route`) changes top-k routing from row-serial single-thread to block-cooperative reduction with the same total-order tie rule, preserving byte-exact selected experts while raising 35B-A3B QMoE decode to about 62 tok/s (#684 context).
- Quaid's #676 follow-up fixes an unsound teacher-forced sub-assertion: reproduced serial and parallel kernels agreed, proving the failing teacher-forced argmax was a test oracle issue rather than a QMoE kernel bug.
- Deckard's cached-dense identity rule: bounded dense-weight memoization uses kernel-local immutable constant-slot identity, mmap metadata when present, one-time hashing per packed matrix/expert slice, and mutex single-flight expansion.

### 35B-A3B CUDA-graph capture fragmentation wave (#708 + follow-ups)

**By:** Cohaagen (investigation/fix), Harry (review)

**What:** CUDA-graph capture repair should target the GatedDeltaNet/runtime seams that actually fragment steady decode. PR #708 shipped the low-risk C3 fix: CUDA Split now derives split sizes from already-resolved output shapes instead of host-reading the split tensor, making GatedDeltaNet Split capture-safe with no host sync. Measured 35B-A3B QMoE hybrid decode improved from 13.415 to 12.132 ms/tok (74.5 to 82.4 tok/s), capture segments dropped 184 to 154, and token@119 stayed byte-exact at `33803`.

**Why:** Steady-state profiling corrected the initial matmul hypothesis: `matmul_nbits` was already 99.8% captured, so a workspace-pool PR would be a no-op and is deprioritized. The true residual fragmentation is in GatedDeltaNet eager islands. C2, eliding the LinearAttention trailing `synchronize()`, was rejected because it changed decode token 33803 to 46283; that barrier is load-bearing. Strict C1 (only `Dim::Static` capture eligibility) was also proven a no-op because GDN pointwise ops carry symbolic batch/sequence dims. The next real fix is executor-context symbol classification: treat batch and decode-step singleton sequence as pinned for capture, while leaving truly growing symbols capture-ineligible.

**Review:** Harry approved #708 after verifying Split sizes come only from static shape inference, cold-kernel `Unsupported`-until-warmed behavior is correct, and copy semantics are unchanged. Nits only: remove/dead-code the obsolete `None` fallback documentation and avoid redundant per-execute re-locking in a later cleanup.

### C1 pointwise capture collapse is mechanically valid but blocked by fp16 oracle lock (#722)

**By:** Cohaagen

**What:** The build-time growing-symbol classifier for C1 is preserved on `squad/elementwise-capture-seqindep` at `36cdd3aa`, but is a NO-GO for shipping until issue #722 is resolved. It mechanically collapses 35B-A3B QMoE hybrid CUDA-graph decode from 154 to 34 captured segments and improves steady decode from 12.263 to 11.763 ms/tok (+4.3% tok/s, 81.5 to 85.0). Capture==eager is byte-exact for both the 35B hybrid and pure-attention qwen2.5-0.5b.

**Why:** The 35B hybrid has a latent fp16 near-tie where baseline captured decode and eager decode diverge around token 20. Heavy accumulators are already fp32; the divergence is an inter-layer fp16 reassociation/rounding coin-flip, not a reduction bug. The current oracle lock passes only on the baseline captured path, while C1 moves the captured stream onto the eager stream and fails the lock. Issue #722 tracks the correction; recommended unblock is to re-anchor the oracle on fp32 teacher-forced adjudication, which both paths satisfy, rather than on one autoregressive captured-path landing.

### VMM KV and dense prefetch standing notes

**By:** Copilot

- Dense prefetch stays eviction-neutral: executor-driven lazy-weight prefetch is scoped to dense MatMulNBits weights, and CUDA admits prefetch only when it fits without eviction or lease growth so MoE behavior and cache-victim selection are unchanged.
- Native CUDA KV bindings may reserve the full-context VMM address range while exposing bucketed physical strides; growth commits the next bucket and repacks valid prefixes in place. Full-context strides committed one granule per head and worsened arena pressure, so floor claims must wait for #694 because ledger-refusal passes made the sweep non-monotonic.

## Extension contract standing directive (#524)

**By:** Justin Chu / contract audit

Every extension seam must expose a stable C ABI with dynamic `.dll`/`.so` loading support **and** a first-class Rust trait; the two surfaces must stay in sync. Ship both upstream ORT plugin-EP ABI and native nxrt ABI, evolving the ORT ABI toward nxrt over time. Do not replace dynamic extension seams with compile-time-only workspace linkage.

## Performance claim discipline

- A per-layer or microbenchmark speedup is not a model-level claim; confirm with Amdahl and real model-level measurement. Always state exact model, dtype, metric, prompt/token regime, host load, and runner.
- Separate measured/estimated/projected. Do not compare measurements under different host load without labeling. Same-run PR-vs-merge-base deltas beat absolute PR numbers.
- A SIMD/accelerated path without a reachability test is equivalent to an unwired placeholder.
- Benchmarks for 35B-A3B must build from a fresh `origin/main` worktree; stale local main caused a false blocker report on 2026-08-03.

## Native-vs-ORT fairness rule

Native-vs-ORT claims must compare the same artifact, quantization, accuracy level, and steady-state methodology with oracle-correct output. If one engine crashes, rejects the graph, or falls back to CPU/different kernels, report a capability gap rather than a throughput multiplier. ORT-CUDA still hard-crashes on 27B/35B-A3B artifacts, so 35B QMoE native tok/s is a standalone native number.

## CUDA / QMoE / hybrid model standing directives

- Classic transformer decode is 100% covered on CUDA for the listed qwen/phi dense families; control-flow ops (`If`/`Loop`/`Scan`) are executor-handled recursively and must not be added to the CUDA EP as normal kernels.
- Qwen3.5 hybrid CUDA coverage includes `CausalConvWithState`, fused `LinearAttention`/Gated DeltaNet, RotaryEmbedding, Bool NonZero, GBQ, rank-3 native positions, and text-only decode pipeline synthesis. Numerics accumulate in f32 and claim gates must reject unsupported configs loudly.
- 27B fused LinearAttention is the active lesson: loader keeps a model-local function as an op iff the selected EP claims it; otherwise inline for byte-identical fallback. Do not revive the removed `ONNX_GENAI_DECODE_INLINE_SCAN` flag.
- 35B-A3B next perf levers after QMoE route parallelism: CUDA-graph capture repair and norm fusion; norm work is roughly 50x above roofline and must be validated at model level.

## Native multi-component pipeline decoder seam

The pipeline decode loop is backend-agnostic through a **stateful** `PipelineDecoderComponent`. Do not drive native pipeline decode through stateless host seams that drop device-resident KV. `NativePipelineDecoder` owns device KV continuity; `PipelineDecodeLoopBackend` holds one component. Rank-3 mRoPE positions derive from declared `position_ids` shape, not model-name gates.

## Metadata and shape-inference rules

- All inference/pipeline metadata except io-shape must be explicit and general. Name guessing is forbidden. Missing required metadata should produce a clear error naming the missing key.
- Shape-inference container support is complete for `ValueType{Tensor|Sequence|Optional|Map}` foundation and Sequence/If/Loop/Scan/SequenceMap threading; tensor path must remain byte-identical. Optional/Map handlers and IR persistence remain deferred until demanded.
- Minimal-build transforms gate on both infrastructure and operator groups; shape-inference registrations use actual ONNX domain/version; attribute-dependent output typing follows the active default/value attribute.

## ORT cached-value cloning

Cloning an ORT cached `Value` covers all POD dtypes via the dtype-agnostic raw-bytes fallback. Use `Value::from_raw_bytes(value.as_raw_bytes()?.to_vec(), shape, dtype)` in terminal arms. Use `as_raw_bytes()` (host-guarded precise error on device tensors), never `to_raw_bytes()`.

## CUDA live weight offload (#63/#87)

Live CUDA weight paging is wired into the decode hot path but gated behind `ONNX_GENAI_WEIGHT_OFFLOAD=1`; default-off is byte-identical. Async page-in is on by default after #544; double-buffer look-ahead remains plan-only until Justin green-lights. Do not retry o_proj 2-way split-K (`K_SPLIT=2`) because it repeatably regressed 7B o_proj GEMV by 0.59%.

## Heterogeneous execution / function inlining

Current public session path selects one EP; `hetero.rs` is not the default stateful executor. Bounded legalization in `hetero::plan` must fail closed when an assigned provider declines a kept function op or function identity is ambiguous. Attribute-parameterized functions require first-class FunctionLibrary/overload-safe IR support before open-ended binding; integrated stateful per-op hetero execution remains tracked separately (#603 family).

## CLI and CI standing directives

- The CLI is a maintainer/developer tool, not a consumer product. Prefer features that shorten debugging/iteration or expose engine behavior. Remote-client mode, model registry/pull lifecycle, and conversion/quantization/fine-tune loops are explicitly rejected as CLI features.
- The REPL is the primary CLI investment; preserve native scrollback via ratatui inline viewport rather than full-screen alternate screen.
- Run tests on every platform. Linux fast jobs are early signal only; they do not replace full platform gates. Instrument coverage only where informative.
- A step that warns instead of failing is not verification: check HTTP status explicitly and validate archive magic bytes before extracting.

## Testing discipline

Assert on what the code did, not summaries. Run new tests in isolation before trusting full-suite green. A fixture whose every assertion is “the turn was dropped” cannot distinguish correct behavior from total breakage. Resolve shared policy once via a shared helper instead of duplicating stale resolution at two sites.

## Model artifact hygiene

Fetch large external models only when needed, measure, and delete immediately. Do not leave benchmark models in `models/` or worktrees.

## Active historical pointers

For detailed per-PR narrative, use archives rather than expanding this live file. Primary locations: `.squad/decisions-archive/2026-07.md` for pre-August ledger, CUDA parity waves, Mac CPU EP/perf methodology, and July CLI/runtime records; `.squad/decisions-archive/2026-08.md` for fused LinearAttention, hetero legalization, 35B-A3B QMoE, #695/#700 hybrid cache fix, and August Scribe batches; older material remains under `.squad/decisions/archive/`.
