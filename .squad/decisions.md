# Decisions — live standing directives

Last consolidated: 2026-08-12T04:30:00Z (Scribe #31974 coverage wave; 2 inbox drops merged; narrative waves from 2026-08-10 through 2026-08-12 archived to decisions-archive/2026-08.md; live file compacted from 50,788 bytes to ~19 KB)
Last consolidated: 2026-08-11T17:55:00Z (Scribe issue-triage session + autonomous fixes; 7 inbox drops merged — 6 new: mobius io-metadata #477 + cosmos3-edge readiness assessment, GBQ zero-point #785, ORT recurrent guard/loader dedup #786, VLM fixture #788, DRY decoder-io #784, VMM contiguous-VA investigation; 1 deduped: qwen35-27b-native already recorded under #779. 30-day archive gate evaluated at 28KB: no dated entries older than 2026-07-12, so nothing archived. Prior: 2026-08-11T16:03:10Z Scribe TopK-perf + 27B-native batch; 37 inbox drops merged, full narrative archived to 2026-08.md)

Standing governance rules and active directives. Full narrative is archived in `.squad/decisions-archive/2026-07.md`, `.squad/decisions-archive/2026-08.md`, and older `.squad/decisions/archive/` files.

This compaction preserved the complete pre-compaction live file in `.squad/decisions-archive/2026-08.md` under "Live decisions snapshot before #695/#700 compaction". Processed inbox drops archived there: cohaagen-695-hybrid-cache-fix.md, cohaagen-qmoe-route-parallel.md, copilot-contract-decisions-q2-q12.md, copilot-plugin-c-abi-everywhere.md, deckard-645-cached-dense-identity.md, harry-700-hybrid-cache-review.md, quaid-676-oracle-testfix.md.
Narrative waves through 2026-08-06 (hybrid Mamba #695/#700, QMoE #676, CUDA-graph #708, C1 capture) archived to `.squad/decisions-archive/2026-08.md`.

## Ledger health rule

Archive by SIZE, not age. Age-only no-ops during high-volume campaigns because most entries are recent, so the live file can exceed spawn-budget while "older than N days" matches nothing. When over the gate, preserve full history in `.squad/decisions-archive/{YYYY-MM}.md`, dedupe rebase-reintroduced sections, and keep live `decisions.md` to standing directives plus pointers. Concurrent Scribe runs are a structural hazard; assemble from inbox drops and check `git log origin/main..HEAD` before committing.

## Current active wave — 2026-08-11 (issue-triage session + autonomous fixes)

### mobius #477: emit full `io` metadata for streamed-weight decoders (external repo, NOT merged)

**By:** Cohaagen

**What:** Fixed the root cause in **mobius** that shipped thin onnx-genai
`inference_metadata.yaml` (no `model.io` block) for large streamed-weight decoders —
opened mobius PR **onnxruntime/mobius#477** (branch
`cohaagen/onnx-genai-io-metadata-robustness`, based on mobius `main`, **NOT merged**
per the never-self-merge-mobius directive; left for the mobius team). Root cause:
`src/mobius/integrations/onnx_genai/auto_export.py` `_add_explicit_io_to_file`
returned **silently** whenever a package component lacked a `.graph` attribute;
external-data/streamed-weight decoders don't retain the `ir.Graph` in memory, so the
sidecar shipped with only `model.attention` + `kv_cache` and no `io` (this produced
the thin qwen3.6-27b-int4-cuda sidecar that forced the #779 runtime workaround and
broke ORT-genai's `token_input` resolution). Fix: when `.graph` is absent, reload the
ONNX graph from the sibling `model.onnx` via `onnx_ir.load` (external data resolves
relative to the model file) and derive ports from it; in-memory fast path preserved;
if the graph is truly unavailable (memory *and* disk), emit a loud `logging.warning`
and skip — never silently ship thin metadata. Attribute-driven, no model-name gates.
3 regression tests added; onnx_genai suite 115 passed / 3 skipped; lintrunner clean.

**Validation:** re-emitted the real Qwen3.6-27B sidecar (metadata-only, against the
existing on-disk `model.onnx`, no re-export) with the patch — the `io` block now
appears with **32 KV entries + 96 conv/recurrent `state_pairs`**,
`token_input: input_ids`, `logits_output: logits`. Confirms the fix removes the need
for the #779 runtime workaround and unblocks ORT-genai on the same artifact. #779
remains a valuable safety net; new artifacts built with patched mobius ship complete
`io` and won't depend on it.

### Cosmos / world-model (cosmos3-edge) native-EP readiness (assessment only, no code)

**By:** Cohaagen

**What:** Assessed whether cosmos-edge world models can run on the native CUDA EP
(mobius already supports cosmos: `models/cosmos.py`, `models/cosmos3_omni.py`,
`_configs/per_model/_cosmos3_edge_vision.py`; `mobius list models` exposes
`cosmos3_edge`, `cosmos3_edge_text`, `cosmos3_omni`).
- **`cosmos3_edge_text` (plain decoder) — likely quick native-CUDA win.** Pure GQA
  decoder (`CausalLMModel`) with two quirks: a non-gated squared-ReLU FFN (`FCMLP`,
  `down_proj(relu2(up_proj(x)))`) and 3D mrope that reduces to standard 1D RoPE for
  text-only. **No hybrid recurrent/conv state** (unlike Qwen3.6-27B) — standard
  append KV only. After the mobius #477 fix its sidecar gains a complete `io` block,
  and `relu2 = Relu(x)²` decomposes to ONNX ops the EP already runs. Recommendation:
  build and smoke-test load/decode on native CUDA EP; only risk is activation-op
  coverage, easily verified.
- **Full `cosmos3_edge` VL pipeline — larger lift.** Metadata is not the blocker
  (mobius emits the encoders→embedding→AR composite). Native EP still needs: SigLIP-
  style vision encoder execution + image preprocessing; the pixel-shuffle merger
  projector (2×2 block merge + matmul); and `inputs_embeds` fusion (scatter image
  features at `image_token_id`) — the main missing runtime piece. NVIDIA publishes no
  modeling code, so pixel-shuffle parity is L1 graph-construction confidence only.
- **Recommended sequencing:** (1) land mobius io fix #477, (2) prove
  `cosmos3_edge_text` on native CUDA EP, (3) scope full VL (vision encoder ops +
  projector + inputs_embeds fusion) as a separate track. No cosmos code implemented
  (report only).

### CUDA GatherBlockQuantized default symmetric zero-point (#702, PR #785 MERGED)

**By:** Quaid

**What:** The CUDA `com.microsoft::GatherBlockQuantized` kernel
(`crates/onnx-runtime-ep-cuda/src/kernels/gather_block_quantized.rs`) left dequant
`offset = 0` when the optional `zero_points` input was absent, while the CPU
reference and ORT use the symmetric midpoint `default_zp = 1 << (bits - 1)`.
GGUF-style embedding tables in mobius-converted 14B/27B models carry no explicit
zero-point, so CUDA dequantized every embedding against 0 instead of 8 (int4) —
empty output (immediate EOS) on the 14B, non-finite logits on the 27B: a
native-vs-CPU correctness divergence (#702). Fix initializes
`int offset = 1 << (bits - 1);` in the NVRTC source and only overrides from
`zero_points` when non-null; explicit-zero_points path byte-unchanged. General
fix, no model-name gate.

**Tests:** CUDA parity oracle extended to the symmetric midpoint for `with_zp==false`
(proves CUDA==CPU/ORT across int4/int8, fp16/fp32) plus host-only guard
`gather_block_quantized_source_uses_symmetric_default_zero_point`; both pass on RTX
GPU. Incidentally fixed 4 pre-existing `gqa_decode_fp16.rs` call sites missing the
`kv_layout` arg (#782) by passing `0` (legacy BNSH).

### ORT paged-KV recurrent guard (#701) + loader error dedup (#467) (PR #786 MERGED)

**By:** Roy

- **#701:** Added `ort_session_has_recurrent_state(session, io)` in
  `crates/onnx-genai-engine/src/kv_bridge.rs`, mirroring the native
  `has_recurrent_state()` gate (#700). Purely structural (RULES.md §2): a state is
  recurrent only when the I/O spec declares it a loop-carried `state_pair` input AND
  that input's shape has a static penultimate (feature) axis. Threaded into the ORT
  paged-reuse **decision** in `Engine::prepare_session_prefix` (`engine/runtime.rs`)
  via a new accessor — not deep inside `load_materialized_past`. No-op for every
  attention-only model loadable today; trips only for hybrid recurrent models,
  forcing correct full recompute.
- **#467:** Hoisted the triplicated `"model directory does not exist: {}"` literal
  in `crates/onnx-genai-ort/src/loader.rs` to a single `model_dir_missing_err(root)`
  referenced at all three `pub fn load*` sites; error text byte-identical.
- Incidental: added missing `kv_layout: None` to `engine/load.rs` `ModelIoSpec`
  (native-backend) — a pre-existing `--features native-backend` build break (#782).
- Verified: `cargo build -p onnx-genai-ort`, `cargo build -p onnx-genai-engine
  --features native-backend`, `cargo test -p onnx-genai-ort loader` (9), `cargo test
  -p onnx-genai-engine --lib kv_bridge` (23, incl. 2 new).

### VLM compat fixture graphs + server CI re-enable (#686, PR #788 MERGED)

**By:** Fenster

**What:** `onnx-genai-server`'s sidecar-free VLM compatibility fixture
(`vlm-executable`) had no executable ONNX graphs, so its test was red and the crate
was deny-listed from CI. Root cause: the full VLM synthesizer
(`to_strict_pipeline_metadata`) omitted `sequence_source`, so decode I/O resolution
(`decode/resolved_io.rs`) defaulted to `TokenIds` and failed to resolve the
`inputs_embeds` (rank-3) decoder port. Fix (route b — explicit metadata in the
production synth): declare `decoder_io.insert("sequence_source", "inputs_embeds")`,
mirroring the text-only fallback and matching real split VLMs (Mobius Gemma4,
onnxruntime-genai). Added reproducible tiny identity fixture graphs
(`vision.onnx`/`embedding.onnx`/`text.onnx`) via
`scripts/build_vlm_executable_fixture.py`, removed the `onnx-genai-server` deny-list
entry in `.github/scripts/workspace_test_packages.py`, and added it to `ORT_BACKED`
(loads ORT at runtime). Coverage guard: 40 tested, 5 denied.

**Evidence:** target test
`sidecar_free_compatibility_package_builds_server_pipeline_and_preprocesses_image`
PASSES; `onnx-genai-genai-config` 33/33. Two `onnx-genai-server` failures are
unrelated to #686 (one parallel-load flake that passes in isolation; one
pre-existing GPU-dependent VRAM-ledger test, verified environmental with changes
stashed).

### DRY decoder-io derivation glue into a shared helper (PR #784 MERGED)

**By:** Cohaagen

**What:** `NativeDecodeLoad::derive_fallback_io` (`native_decode/load.rs`, live
`InferenceSession` ports) and `maybe_fill_hybrid_io_from_graph` (`engine/load.rs`,
disk graph, `#[cfg(feature = "native-backend")]`) duplicated ~40 lines building an
identical `ModelIoSpec` from a graph-derived `DerivedDecoderIo`. Extracted one
shared helper `GenAiConfig::derive_model_io_spec_from_graph(graph) ->
Option<ModelIoSpec>` in `onnx-genai-genai-config/src/compatibility.rs` (which
already owns `DerivedDecoderIo`/`ModelGraphInfo` and depends on `onnx-genai-metadata`
— no cycle). It encapsulates canonical derivation → empty-`state_pairs`
recurrent-hybrid gate → name-presence port binding → `ModelIoSpec` assembly.
Behavior-preserving: the authoritative `io.is_some()`-wins gate, the
`state_pairs.is_empty()` safety gate, and the native-backend cfg gating are
unchanged; engine-side spec now also carries `kv_layout: None` explicitly, unifying
the two specs. Validated: fmt, `build -p onnx-genai-engine --features native-backend`
clean, `test -p onnx-genai-genai-config derive` (5, incl. 2 new), `test
-p onnx-genai-engine --features native-backend --lib native_decode` (68).

### VMM contiguous-VA-per-sequence KV: crux answered, bucket-stride floor stands

**By:** Copilot (investigation)

**What:** Isolating GPU test (`vmm_kv_contiguous_tail_gpu.rs`) confirmed the decode
kernel's **read pattern**, not the VA reservation, forces the physical commit: a
read bounded to live length leaves the reservation tail uncommitted; a read one byte
into the tail faults `CUDA_ERROR_INVALID_VALUE` (non-sticky) — #721 stage 4 in
isolation. A fixed full-context stride (the "one flat VA, never re-strided" ideal)
pays an `objects × granule` floor because KV is head-major: qwen2.5-0.5b = 96
head-stripes × 2 MiB = 192 MiB for ~12 KiB of content (1.5 GB at 32K, 32× over
bucket growth). The landed `kv_commits_on_demand` path (#682/#740/#748) is the right
realization at bucket stride (full-context VA reserved, only current bucket
committed, stable `device_ptrs`, verified on real qwen2.5-0.5b) but still
re-strides/re-captures on growth. Low committed bytes AND no re-capture cannot both
hold under head-major layout without sub-bucket commit (open: does the ORT GQA CUDA
kernel touch the bucket tail `[logical_len, bucket)`?) or a seq-major KV layout
(needs re-exported models / custom kernel). Device KV paging is owned by the CUDA VMM
layer (`CudaVmmAllocator` granule mapping under a fixed reservation + governor
grants), not `onnx-genai-kv`; the native CUDA decode path has no
`PageTable`/`PagedKvCache` consumer — that machinery stays host-only.

> **Dedup note:** Qwen3.5/3.6-27B hybrid GDN native-CUDA enablement via io-derivation
> (Cohaagen) is already recorded above as "27B hybrid GDN native CUDA enablement via
> io-derivation (#779, ...)"; the `cohaagen-qwen35-27b-native.md` inbox drop was
> consolidated there and not duplicated here.

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

> **Update (2026-08-11):** This classifier shipped via PR #728 (APPROVED round 8) with an authoritative symbol-provenance record, central hard veto, host-seam precedence, and a restored fatal oracle tripwire. See "PR #728 C1 growing-symbol capture classifier — APPROVED" in the 2026-08-11 wave above. The note below is retained as the original diagnosis.

**By:** Cohaagen

**What:** The build-time growing-symbol classifier for C1 is preserved on `squad/elementwise-capture-seqindep` at `36cdd3aa`, but is a NO-GO for shipping until issue #722 is resolved. It mechanically collapses 35B-A3B QMoE hybrid CUDA-graph decode from 154 to 34 captured segments and improves steady decode from 12.263 to 11.763 ms/tok (+4.3% tok/s, 81.5 to 85.0). Capture==eager is byte-exact for both the 35B hybrid and pure-attention qwen2.5-0.5b.

**Why:** The 35B hybrid has a latent fp16 near-tie where baseline captured decode and eager decode diverge around token 20. Heavy accumulators are already fp32; the divergence is an inter-layer fp16 reassociation/rounding coin-flip, not a reduction bug. The current oracle lock passes only on the baseline captured path, while C1 moves the captured stream onto the eager stream and fails the lock. Issue #722 tracks the correction; recommended unblock is to re-anchor the oracle on fp32 teacher-forced adjudication, which both paths satisfy, rather than on one autoregressive captured-path landing.

### VMM KV and dense prefetch standing notes

**By:** Copilot

- Dense prefetch stays eviction-neutral: executor-driven lazy-weight prefetch is scoped to dense MatMulNBits weights, and CUDA admits prefetch only when it fits without eviction or lease growth so MoE behavior and cache-victim selection are unchanged.
- Native CUDA KV bindings may reserve the full-context VMM address range while exposing bucketed physical strides; growth commits the next bucket and repacks valid prefixes in place. Full-context strides committed one granule per head and worsened arena pressure, so floor claims must wait for #694 because ledger-refusal passes made the sweep non-monotonic.
- Async weight page-in is the default (`ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN=1`): fence-ordered copies stream directly from external mmap regions into reusable pinned staging instead of materializing an owned host tensor per miss (sync demand-copy retained at `=0` for A/B). Reports materialize/H2D/wait/sync counters.
- **Managed no-spill VMM is becoming the default (#755), owner-directed:** when a model exceeds the resolved device budget the runtime auto-enables weight streaming/offload rather than hard-OOMing; the explicit `serve --vram-limit` becomes an override. Hard prerequisite is **#716** (offload and CUDA-graph capture are mutually exclusive today — the pager's alloc/copy/free ops are capture-illegal, so flipping the default before #716 would silently disable capture for exactly the models this helps and give back the 154→34 collapse). Sequencing: (1) #735 strategy inference runs unconditionally and reports the plan; (2) #716 makes offload capture-compatible under stable VA slots; (3) flip default with a one-release opt-out; (4) publish interleaved same-session comparison. Must not regress: no silent spill to WDDM shared memory; capacity refusal stays a pre-header 429 (#743); public budgets stay committed physical bytes; a model that fits in VRAM must not start paging.
- **Multi-request KV batching uses VMM contiguous virtual addresses, not paged attention (#750), owner-directed:** each sequence gets its own contiguous device VA with physical granules mapped on demand, so the attention kernel keeps seeing a flat contiguous KV buffer and never walks a page table; `#721` stage-3 device-resident paged KV (`CudaPageStore`) is superseded unless this route fails. The route reduces to one question — can the decode kernel be bounded to the live sequence length instead of the padded shape? (#721 stage-4 committed 1.5 GB full-context VA vs 48 MB bucket growth, 32× worse, because the kernel read the padded shape and relied on masking.) Granules come from the #740 authority-scoped shared pool via `carve()`; commit floors are per-object, not per-token; `cuMemMap` during capture is not proven replayable; never unmap while a replay may be in flight.
- **Transactional mapped growth / shared device authority:** mapped-capacity lending is coordinated by the shared memory authority through weakly-registered `ReclaimableMappedHolder`s and RAII `MappedGrowthGrant`s (victim allowance transferred before callbacks; only `mapped_bytes - new_limit` physically reclaimed). VMM weight admission uses two authority-coordinated constraints — mapped granules consume the cache's weight allowance while newly-created handles consume global physical headroom — reserved transactionally, releasing new handles on failed transactions. `MemoryGovernor` exposes a stable `MemoryAuthorityId`; `VirtualBuffer` rejects a different governor before reserving/committing. Multi-model servers own one concurrency-safe device authority per backend/device domain and inject it into every construction path. QMoE workspace is resolved and reserved as one session-persistent peak before the admission callback (shared planning/execution layout helper), not inside execute. Managed VMM/pool construction failure is fatal before model allocation; insufficient reclaim is a typed capacity refusal → HTTP 429 with `Retry-After`, never 500/InvalidRequest.
- **LinearAttention trailing stream sync is not load-bearing for capture:** it launches one same-stream kernel that reads each state column before overwriting it, so stream ordering suffices; the sync only surfaced eager launch errors and caused the 33 `CaptureRecordingFailed` seams in #728. A capture/replay regression proved four in-place recurrent decode steps byte-identical to eager with no capture-time allocations (H200 same-model measurement still recommended before relying on the segment collapse).

### 2026-08-11: Whole-step megakernel is the next bounded batch-1 latency lever

**By:** Roper

**What:** Roper's read-only scoping concluded that a whole-step/persistent megakernel is the only remaining lever that attacks the current batch-1 decode regime's GPU-side bubbles after CUDA graph capture. vLLM full CUDA graph and llama.cpp are capture/per-op systems, not true megakernels; Mirage MPK is not directly adoptable for this Rust/ONNX/int4 QMoE stack.

**Why:** Prior FC2 fusion, ILP, and vectorized-int4 attempts show QMoE decode is occupancy/MLP/barrier bound rather than launch or bandwidth bound. Recommendation is Phase 0 only first: a persistent single-op QMoE decode kernel that removes FC1/FC3/activation/FC2/combine scratch round-trips while preserving fp32 accumulation order. Gate continuation on oracle margin `0.09375` and at least a 3% model wall-clock win.

**Detail:** Full feasibility memo archived in `.squad/decisions-archive/2026-08.md` under "Roper megakernel feasibility inbox drop".

## Current active wave — 2026-08-11 (TopK correctness/perf + 27B hybrid native)

### CPU-EP TopK: k-major output layout for non-final axes (#774, MERGED)

**By:** Coordinator (fixed inline after a sub-agent canary false-positive)

**What:** `TopKKernel::execute` (`crates/onnx-runtime-ep-cpu/src/kernels/selection.rs`) built outputs by sequential `push` in `outer→inner→rank` order, giving layout `[outer][inner][k]`. ONNX TopK keeps the input shape with `shape[axis]=k`, i.e. k-major `[outer][k][inner]`; winner rank `r` for slot `(outer,i)` must land at flat index `(outer*k + r)*inner + i`. The two layouts are equal only when `inner==1` (final axis), so the bug was latent for the common final-axis/router/argmax case but corrupted values+indices order for any non-final-axis TopK with `k>1` (dtype-independent). Fix pre-sizes outputs to `numel(shape)` and writes each element to its strided destination; removed the now-unused `sorted` field (kernel always emits sorted winners, satisfying both `sorted=1` and `sorted=0`). CUDA EP / ORT already emit k-major.

**Validation:** New discriminating tests `topk_non_final_axis_uses_k_major_layout` ([2,3,2] axis=1 k=2) and `topk_first_axis_is_k_major` ([3,2] axis=0 k=2) assert k-major values AND indices; final-axis tests unchanged. `cargo test -p onnx-runtime-ep-cpu --lib selection` 22/22; fmt+clippy clean.

### CPU-EP TopK: O(width) partial-select instead of full axis sort (#775, MERGED)

**By:** Coordinator

**What:** Partial-selection TopK now uses `select_nth_unstable_by` introselect (O(width)) instead of fully sorting each axis slice (O(w log w)), matching the sampler `top_k_threshold` technique. Selection semantics/order preserved. 24/24 selection tests pass.

### 27B hybrid GDN native CUDA enablement via io-derivation (#779, auto-merge awaiting CI)

**By:** Cohaagen

**What:** Qwen3.5/3.6-27B hybrid GDN native CUDA was blocked because the 27B artifact ships a thin `inference_metadata.yaml` with no `io` port contract → `resolve_kv_layers` returns `None` → "per-layer KV page geometry unknown". Fix: `maybe_fill_hybrid_io_from_graph` in `engine/load.rs` auto-derives the decoder `io` contract from the ONNX graph's port inventory, gated on non-empty `state_pairs` (recurrent-hybrid only). DRY, no model-name gate — unblocks the whole hybrid GDN family. Byte-exact: native argmax `11751` " Paris" == fp32 oracle, top-1 margin 2.549 nats. Locked by `qwen35_27b_hybrid_native_cuda_e2e.rs`. Cohaagen's decision memo merges via the #779 branch, not this worktree's inbox.

### GLM-4-9B and DeepSeek-V2-Lite native CUDA load unblock (#770/#771)

**By:** Cohaagen (fix), Harry (review — both APPROVE)

- **GLM-4-9B (#770):** blocker was KV admission, not an unsupported op — native reserved metadata `max_sequence_length=131072` (5 GiB) instead of the effective runtime cap. Native KV reservation now charges the actual runtime CUDA KV capacity (`cuda_kv_debug_stats().hard_max_len`, e.g. `ONNX_GENAI_CUDA_KV_MAX_LEN=4096`) before falling back to metadata. Partial-RoPE GQA already handled (rotary width derived from cos/sin cache). Native greedy loads at 93.6 tok/s; golden lock passes.
- **DeepSeek-V2-Lite QMoE (#771):** blocker was static QMoE placement rejecting `Cast(fp16 initializer→fp32)` scale inputs ("QMoE input 3 is not a graph initializer"). Placement now accepts a one-hop default-domain `Cast(initializer)` as initializer-backed for weight-region classification; runtime still receives the fp32 Cast value. Native greedy at 52.5 tok/s; golden lock passes.
- No model-name gates; both are DRY runtime/placement behavior. Harry's one hardening follow-up recommended; reviews are not a merge gate (auto-merge).

### QMoE decode fusion / megakernel experiment wave — one SHIP, four NO-SHIP

**By:** Cohaagen, Quaid

**Shipped:** Conservative small-route (`rows==1`, `routes<=16`) decode fusions on the 35B-A3B QMoE hybrid path:
- Fused FC1 gate/up + SwiGLU into one `qmoe_gate_up_activate_*` kernel (removes the `qmoe_activate` launch and `fc1_output` scratch round-trip). ~3.3% model win (11.511→11.126 ms/tok).
- One-CTA-per-output-task GEMV for small-route decode (skips the trailing `__syncthreads()` reuse barrier; fp32 K-reduction order unchanged). `qmoe_linear_f32` ~31.8→27.6 us. (#764 wave.)
- Oracle held byte-exact: teacher-forced token `33803`, margin `logprob(33803)-logprob(5342)=0.09375`.

**NO-SHIP (kept as memos, reverted code):** all failed the >3% model-level gate and/or the oracle tripwire —
- FC2/down+combine fusion: +0.08% (noise); serializes route loop, adds sync on a numerically sensitive path.
- ILP-2 K-unroll of `qmoe_linear_impl`: regressed (`qmoe_linear_f32` 27.6→32.2 us; active warps 13.2→9.7; occupancy loss > latency hidden).
- Megakernel Phase 0 (persistent single-op QMoE decode, counter-based producer/consumer): oracle byte-exact but decode **regressed +7.4%** — occupancy trap; Quaid recommends NOT greenlighting Phase 1 as specified. Note: current main (#766) already fuses FC1+FC3+activate, so the real decode chain is 4 launches / 2 scratch, not the stale "5-launch/4-scratch" spec.
- Int4 DP4A / 128-bit vectorized weight-stream: +4.90% ms/tok, QMoE op +25%; teacher-forced margin unchanged (0.09375) but autoregressive #722 tripwire drifted to token 13 (outside benign {33803,46283}).

**Standing lesson:** 35B QMoE decode is ~10–20× above the int4 bandwidth roofline (~0.1–0.24 ms/tok floor vs ~2.8 ms/tok measured); it is latency/occupancy/barrier-bound, not bandwidth- or launch-bound, so scalar ILP and simple scratch removal do not clear the noise floor. The next real lever is a designed fused/persistent decode kernel that keeps FC1 activations local while exposing enough FC2 output-feature parallelism, gated on the 35B teacher-forced oracle margin as the primary acceptance gate. Full memos in `.squad/decisions-archive/2026-08.md`.

### PR #728 C1 growing-symbol capture classifier — APPROVED (round 8)

**By:** Gaff (final revision), Harry (review), plus Roy/Sapper/Sebastian/Leon/Batty/Deckard revision rounds

**What:** Supersedes the "#722 blocked" note above for the classifier design. The build-time growing-symbol classifier that collapses 35B-A3B QMoE hybrid CUDA-graph decode 154→34 captured segments (~4% tok/s) is now correct and merged. Final shipped design after 8 review rounds:
- **Authoritative symbol-provenance/unification record.** Shape inference records every symbol unification/derived-symbol lineage at its single chokepoint and persists it on the `Graph`; the executor closes its growing set over that authoritative map (union-find equivalence closure), covering elementwise, MatMul/Einsum/Concat/Expand broadcast, and `Reshape([-1])`/`Flatten` derived expressions — with zero op-enumeration in the executor. Classifier defaults **fail-safe** (an op stays eager unless its shape lineage is provably pinned).
- **Central hard veto.** The classifier verdict is enforced as a hard capture veto in `Executor::node_capture_reason` before any kernel-specific `capture_support()`, closing bypasses from kernels that returned `Supported` unconditionally (`UnaryMathKernel`/`NotKernel`/`BitwiseNotKernel`, etc.). `capture_shape_eligible` reduced to `seq_independent` (removed the permissive `numel==1 || is_fixed_decode_shape` OR).
- **Host-seam precedence.** Veto runs AFTER the EP structural policy (`plan_capture_region`) so `If`/`Loop`/`Scan`/Sequence nodes report `HostControlFlowOrSequence`/`HostSeam`, not `EagerDeviceSeam` — preserving the public capture-segmentation contract.
- **CSA coverage:** `CompressedSparseAttention` growing `selections` on output-5 last axis is collected; generic declared `past…`/`present…` rank-4 KV I/O scan collects growing symbols without per-model gates. Rank-4 + symbolic-penultimate guard keeps fixed-capacity Mamba/GDN recurrent state capturable.
- **Oracle tripwire restored to FATAL:** autoregressive check asserts token ∈ {`33803`, `46283`} (the two benign #722 fp16-tie outcomes); any other token (5342/279/unrelated) fails CI. Final measurement 34 segments / ~83–87 tok/s, GDN pointwise stays capturable. Full round-by-round narrative archived in `.squad/decisions-archive/2026-08.md`.

### CI is asynchronous (governance)

**By:** Copilot, at owner's direction

Unless the user explicitly requests otherwise, agents and the coordinator do **not** wait for CI before continuing, reporting, or merging. Required local targeted tests, Clippy, builds, and hardware probes remain blocking; CI runs asynchronously and later failures are fixed forward. Do not poll or hold a turn open for CI, and do not keep completed worktrees solely for CI. An explicit user instruction may make a particular CI result blocking. Source — User: "现在规定所有人除非明确指令否则不要等ci。本地测试。ci看到有问题再修。可以fix forward".

### Design autonomy and parallel worktrees (governance)

**By:** Copilot, at owner's direction

The coordinator may independently make architecture/design optimizations when evidence supports them; direction-changing decisions must update durable design docs with measurement, falsifier, limitations, and rollback/override path. Prefer parallel agents in separate git worktrees when work has no overlapping files, shared mutable state, or unresolved common contract; keep changes to the same core contract or continuous call chain serial until the shared dependency lands. Source — User: "如果需要设计上的优化 你可以自行裁决 更新设计文档。" / "还有如果可以最好并行多worktree推进".

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

## Testing and CI standing directives (additions 2026-08-11)

- **`cargo test --workspace` silently truncates on failure.** Always use `--no-fail-fast`. A run reporting "1555/2" was really 4580 passed / 20 failed / 436 ignored across 304 binaries. Fail-fast mode exits at the first failing binary and reports wrong totals; this masked real failures across the session.
- **Never commit `.squad/` files to external repos.** Deleting the files in a follow-up commit does not remove the content — git history retains it and the delete commit's message re-exposes the path. If `.squad/` is accidentally committed, purge via `git filter-branch` or `git-filter-repo` and force-push. This was discovered on upstream ORT PRs #31973 and #31974; both branches required history purge.
- **An agent's self-report is not evidence.** Sapper reported all four CUDA defects fixed; independent review found a use-after-free, a panic bomb making success unreachable, and a direction classification gap. Nabil's B2 deferral cited an API that did not exist — the API was present in the generated bindings. Verify implementation claims via command output, code reading, and test results; never accept "implemented" on face value.
- **Reviewer lockout is enforced end-to-end.** Sapper authored CUDA fixes → rejected by Gaff → Nabil fixed B1/B3/S4 → Batty fixed B2. No author revised their own rejected artifact. The chain must close with an independent verifier confirming each fix.

## Active historical pointers

For detailed per-PR narrative, use archives rather than expanding this live file. Primary locations: `.squad/decisions-archive/2026-07.md` for pre-August ledger, CUDA parity waves, Mac CPU EP/perf methodology, and July CLI/runtime records; `.squad/decisions-archive/2026-08.md` for fused LinearAttention, hetero legalization, 35B-A3B QMoE, #695/#700 hybrid cache fix, and August Scribe batches; older material remains under `.squad/decisions/archive/`.


## ORT plugin-EP ABI standing directives

### OrtMemoryInfo lifetime (USE-AFTER-FREE — caused real bugs)

`EpDevice_AddAllocatorInfo(_In_ OrtEpDevice*, _In_ const OrtMemoryInfo*)` stores the raw pointer; ORT does NOT copy it. **Do NOT call `ReleaseMemoryInfo` after a successful `AddAllocatorInfo`.** ORT releases it when `ReleaseEpDevice` is called. Release only on failure. Use `CreateMemoryInfo_V2` with explicit `OrtMemoryInfoDeviceType_CPU` / `OrtDeviceMemoryType_DEFAULT`; the legacy `CreateCpuMemoryInfo` leaves those fields uninitialized, producing garbage DeviceType:64 / MemoryType:28 after repeated register/unregister cycles.

### OrtGraph*/OrtNode* scope (CACHING BUG — caused real bugs)

`OrtGraph*` and `OrtNode*` handles passed to `GetCapability` / `Compile` callbacks must NOT be stored or cached beyond the callback return. ORT may free them immediately after. Copy all needed attributes and initializers into owned Rust data structures during the callback.

### Shape-inference fail-closed policy

`ShapeInference::for_op` / `for_node` return `Declined { op_type, domain }` for any op with no modelled rule. `infer_shapes` turns `Declined` into an error status — ORT receives a proper failure, not silently-wrong output tensors. Do not reintroduce a silent `SameAsInput(0)` fallback.

### Evidence discipline for implementation claims

A previous session reported the adapter crate as "Implemented (v1)" when it did not compile. **Implementation claims require quoted command output as evidence** (`cargo check` / `cargo test` output). "Passes locally" is not evidence; command transcript is.

## CI and workflow standing directives

**CI is asynchronous.** Do not wait for CI before continuing, reporting, or merging. Required local targeted tests, Clippy, builds, and hardware probes remain blocking. Fix CI failures found later in follow-up commits.

**Design autonomy.** The coordinator may make architecture and design decisions when evidence supports them. Direction-changing decisions must update durable design documentation (measurement, falsifier, limitations, rollback path). When work is separable without shared mutable state, prefer parallel agents in separate worktrees.

## Memory governance standing directives (2026-08-06/08/09)

- `MemoryGovernor` exposes a stable `MemoryAuthorityId`; each backing authority is named at construction; `VirtualBuffer` rejects a different governor before reserving or committing.
- CUDA weight residency admission uses two constraints: mapped granules vs. weight allowance, newly created handles vs. global physical headroom. Failed transactions release newly created handles.
- Multi-model server owns one concurrency-safe device authority per backend/device domain; engine host/disk ledgers remain private.
- QMoE workspace: kernels declare typed workspace requirements; native CUDA prefill resolves QMoE shapes and reserves one reusable session-persistent workspace peak before the admission callback.
- Explicit byte `--vram-limit` enforced at engine load: native CUDA derives offload budget; non-offload backends fail at load if weights exceed limit. Derived budget = VRAM limit minus device KV/recurrent state, and must meet the largest lazy-weight node working set.
- CUDA weight offload defaults to async mmap-backed page-in with fence-ordered copy into reusable pinned staging. Synchronous demand-copy path available via `ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN=0`.

Full narrative in `.squad/decisions-archive/2026-08.md` (DROP sections: copilot-memory-authority-contract, copilot-committed-granule-admission, copilot-shared-device-authority, copilot-qmoe-workspace-stage0, copilot-vram-limit-load-enforcement, copilot-705-weight-offload-prefetch, copilot-mapped-growth-grant, copilot-ci-is-asynchronous, copilot-design-autonomy-and-parallel-work).


## 2026-08-12 — Apple scope: macOS arm64 / Apple Silicon ONLY

**By:** @justinchuby (scope correction), Mariette (PR #31993 lockout revision), Coco (PR #32001 lockout revision), Coordinator (verification)

### ⚠️ STANDING CONSTRAINT — applies to ALL future Apple work

**Apple scope is macOS arm64 / Apple Silicon ONLY.** Intel Mac and universal2 are out of scope. Gate with `APPLE + arm64/aarch64`. Do not add x86_64 Apple slices or Intel fallback tests. Do not let universal-binary concerns block enabling ARM kernels. Preserve the portable non-Apple fallback when the compile option is off. **iOS is not implied** — unless separately justified, Apple work stays scoped to macOS arm64.

This **narrows** the earlier Apple framework policy entry (Accelerate/BNNS/vDSP eligible when Apple-only, opt-in, portable fallback): that policy still stands, but its platform scope is now macOS arm64 only. **Read both entries together — neither stands alone.**

**Rescoping is not the same as removing a guard.** The `#if defined(__APPLE__) && defined(MLAS_TARGET_ARM64)` compile-time gate stays — it prevents the kernel reaching targets without FEAT_FP16. What was removed was the x86_64 *test slice*, not the *gate*.

**Use the tree's existing arch idiom.** `onnxruntime_target_platform STREQUAL "arm64"` is the canonical upstream variable, already used at `cmake/CMakeLists.txt:567/575/589` — prefer it over inventing a new condition from `CMAKE_OSX_ARCHITECTURES`.

### PR #31993 (Mariette, lockout revision) — rescoped to macOS arm64 only

- Removed the `#else` branch in `test_cast_fp16.cpp` that asserted null dispatch pointers on non-ARM64 Apple (x86_64 slice test, now out of scope).
- Rescoped commit messages and PR body from universal2/iOS/Intel to macOS arm64 only.
- Compile-time gate `#if defined(__APPLE__) && defined(MLAS_TARGET_ARM64)` unchanged.
- Positive `ASSERT_NE(...Kernel, nullptr)` dispatch assertions survive — test remains non-vacuous.
- Head: `68ee0de`.

### PR #32001 (Coco, lockout revision) — rescoped to macOS arm64 only

- Added `onnxruntime_target_platform STREQUAL "arm64"` condition to `cmake/CMakeLists.txt`.
- Implemented as `elseif` after the `if(NOT APPLE)` check, using warn-and-disable to match `onnxruntime_USE_SVE`/`onnxruntime_USE_KLEIDIAI`.
- Rescoped option description, MLAS comment (removed "iOS 4.0+"/"macOS 10.3+") and `build_args.py` help text.
- Verified no-behaviour-change-when-disabled on Linux x86-64.
- Head: `52db6351b5`. PR remains draft.

### Durable lessons

- **Non-arm64 Apple slices are out of scope.** Do not add `#else` branches for x86_64 Apple test slices; do not reference universal2 or Intel Mac in commit messages or PR bodies for ARM kernel work.
- **`onnxruntime_target_platform` is the canonical arch variable on Apple.** Already used at CMakeLists.txt:567/575/589; do not invent an alternative from `CMAKE_OSX_ARCHITECTURES`.
- **warn-and-disable, not FATAL_ERROR**, for platform-check failures in optional ISA options (per SVE/KleidiAI idiom).

---


## Archived narrative waves pointer (2026-08-12)

All wave narratives from 2026-08-10 (EP plugin export), 2026-08-11 (PR #762 parity, upstream CI correction, Apple MLAS f16 cast), and 2026-08-12 (CUDA MatMulNBits, Apple framework infra, rejection-response wave, PR #31973 threshold fixes) are in `.squad/decisions-archive/2026-08.md` under "ARCHIVED 2026-08-12T04:30:00Z".

**Status snapshot (2026-08-12T04:30:00Z):**

| PR | Status | Head |
|----|--------|------|
| #31985 | **MERGED** `f2dfa4e9eb` | — |
| #31973 | Draft — threshold + comment fixes landed | — |
| #31974 | Draft — coverage wave landed (`a12c7ddde3`), no blockers |  |
| #31988 | Draft — **parked pending GPU** | `dc1e173e4b` |
| #31993 | Draft | `02a9f34` |
| #32001 | Draft | `0d924a421b` |
| #32003 | Draft | `23dcfddaaf` |

## 2026-08-12 — BF16 PrePack and generic-broadcast coverage (PR #31974 Opus v2 wave)

**By:** Chew (coverage), Holden (delta review), Coordinator (verification)

### Durable lessons

- **Test the path real models take.** BF16 scale/bias are always constant initializers in practice — they route through `PrePack()` at session init (bf16→f32 conversion). The graph-input path was well tested; the PrePack path had zero coverage. `is_initializer=true` in OpTester triggers PrePack. Coverage gaps on paths models always take are worse than gaps on exotic paths.
- **Internal blocker labels (`B1`–`B6`, `N1`–`N4`) are a leak class.** They mean nothing upstream and reveal internal process. Sweep for them alongside `squad`, `nxrt`, persona names, and internal issue numbers before every push.
- **When correcting a tolerance comment, verify against the checker, not intuition.** The real behaviour is `absolute + relative × |expected|` (numpy.isclose semantics, `checkers.cc:117-120`). A stated absolute tolerance understates effective tolerance. Confirm the regression-detecting margin survives the relative component — here it did: 0.00011 effective vs 0.004 pre-fix error (36× margin).
- **Apple/macOS CI dependency downloads fail often and before compilation.** Observed causes: cpuinfo, XNNPACK, eigen3, FXdiv (`status_code: 60`) and vcpkg bootstrap (`curl failed to verify the legitimacy of the server`). `gh run rerun` refuses fork-PR jobs; only retrigger is a push. Do not mark ready while CI is red even when the failure is provably infra.

### Test counts after coverage wave (fresh build, coordinator-verified)

- 20/20 BF16 tests pass (was 17)
- 106/106 LayerNorm suite tests pass (was 103)
- 7/7 SkipLayerNorm PrePack validation tests pass

