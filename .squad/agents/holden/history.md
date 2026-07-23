# Holden — History

## Project Context (joined day)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **State when joined:** Phases 1-4 done; tool use/grammar/chat-template; Qwen2.5-0.5B runs; Hermes agent E2E; long-context O(1)/token via static-cache in-place KV. Working on DESIGN §26 batched serving + reviews.
- **Requested by:** Justin Chu
- **Joined:** 2026-07-12

## 2026-07-12T13:14:00-07:00 — Security audit merged
Holden's unsafe/resource/supply-chain audit is now in decisions. Current unsafe invariants are documented and sound under today's constraints; cargo audit found 0 vulns and 2 unmaintained transitive tokenizers warnings.


### 2026-07-12T14:50:00-07:00
Recurring audit convention is canonical: `.github/workflows/audit.yml` runs weekly and on dependency changes; fresh cargo-audit found 0 vulnerabilities. Continue periodic security review passes.

## 2026-07-14T02:37:00Z — Reviewed ep-api + ep-cpu (safety)
- **ep-api (65ec9f6):** 🟡 safety — DeviceBuffer ownership, Send/Sync soundness, unsafe construction contracts.
- **ep-cpu (ea30279):** 🟡 safety — strided::view_in_bounds enforcement, isolated unsafe blocks (aligned alloc/dealloc, copy_nonoverlapping, two strided accessors), no cross-EP free.

## 2026-07-14T05:04:00Z — ORT2 safety reviews: session base + dynshape + capi

- **squad/ort2-session** review (🟡): All 5 invariants held (view bounds, single-free, no cross-EP free, copy size ≤ min(src,dst), host-global). Aliasing: in-place ops cause CycleDetected at build. Miri clean. Advisories: A1 mid-run error-path buffer leak; A2 unchecked i64 in `view_in_bounds`; A3 cache key omits dtypes.
- **squad/ort2-session-dynshape** review (🟡): Invariant #1 holds against run-scoped buffers (gate keys off real `buf.len()`). Single-free on realloc verified. 14/14 Miri-clean. Advisories: H-D1 unchecked shape-multiply overflow; H-D2 stale buffer_shapes if allocate fails post-dealloc; Holden-A1 (pre-existing) mid-run leak unchanged.
- **squad/ort2-capi** review (🟢): All 6 FFI axes pass. 12/12 Miri-clean. Advisories: A1 release fns not in guard; A2 storage_bytes unchecked multiply (bounded by prior validation).

## 2026-07-14T06:06:00Z — H-D1 Re-Review (holden-7) → 🟡 SHIP

- **holden-7:** Re-reviewed Deckard's H-D1 fix on `squad/ort2-hardening` @ 852f262.
- Prior 🔴 cleared: all three fix layers (dtype.rs checked_storage_bytes, executor.rs both alloc sites, strided.rs i128 address math) confirmed correct. Original exploit vector `[2^61]`×f64 dead — `ShapeOverflow` at allocation before any buffer exists.
- **Verdict: 🟡 SHIP-with-advisories.** Residual advisories A1 (peripheral panic paths → graceful error) and C1 (addressed_elem_range i64 before i128 widening) are memory-safe non-blocking; suggested fast-follow owner: Leon (neither Deckard nor Batty).
- Fix cherry-picked to main: dbf2d70, 9dcdc04, f749012.

## 2026-07-14T07:20:00Z — ORT2 shape-inference crate review cycle

- **holden-8:** Reviewed `onnx-runtime-shape-inference` DimExpr symbolic-algebra soundness. 🔴 REJECT — `DimExpr` `add`/`sub`/`mul` used unchecked i64: debug panics, release wraps to bogus dim on large tensors (`2^80` product). Secondary: `checked_div` unguarded against `i64::MIN/-1`. All other items HELD. Fix assigned to Deckard.
- **holden-9:** Re-reviewed Deckard's overflow fix (`09988f3`). 🟢 GREEN — all combiners use `checked_*`; `overflow()` sentinel never aliases (SymbolInterner bypasses cache); poison propagates; no wrap in debug or release. 69 tests green debug+release. Advisory noted: `movement.rs` slice-index raw i64 (pre-existing, filed as follow-up).

## 2026-07-14T08:40:00Z — ORT2 shape-inference wiring + IR dtype hardening reviews

- **holden-10:** Reviewed Roy's shape-inference wiring (`f4141b9`). 🟢 GREEN. Symbol-unification sound (overflow sentinel gates representative arm; deterministic order-independent; single topo-pass). Loader seam transactional (graph mutated only after full write-back). JIT fallback comment-only diff. No regressions to view_in_bounds/checked_storage_bytes/unsafe. Full ORT2 suite green debug+release. Non-blocking advisory: stricter fail-fast coupling means op-rule false-positives now block load; Chew to confirm none fire on BERT/opset-12.
- **holden-11:** Reviewed Deckard's IR dtype hardening (`f965f0b`). 🟡 APPROVE-WITH-FOLLOW-UP. Net soundness improvement; no regression; no new unsafe. Float4E2M1 routes through `div_ceil(2)` (overflow-safe). Fail-closed attr hardening safe. 300 tests green debug+release. **Required follow-up:** `graph_builder.rs` value-info (L232,241) and attribute-tensor (L357,365,374) still `.unwrap_or(Float32)` — silent mislabel for unmodeled dtypes. Deckard PROGRESS.md claim overstated. Owner: Roy/Batty/Leon before complex-dtype milestone.

## 2026-07-14T10:00:00Z — ORT2 dtype fail-close review (holden-12)

- **Task:** Review leon-10's dtype fail-close work on `squad/ort2-dtype-failclose` (`a822a21`). Verify closure of own prior holden-11 finding (value-info + attribute-tensor `.unwrap_or(Float32)` sites).
- **Verdict:** 🟢 GREEN — finding fully closed, no over-reach, no regressions.
- **Key checks:** All 8 real-dtype decode sites confirmed fail-closed via `decode_dtype`. No surviving `unwrap_or(Float32)` on real-dtype sites. Signature changes `-> Result<…>` with `?`; transactional-on-failure preserved. Proto bump correct. Full ORT2 suite (262 tests) green debug+release. bert_toy PASS max_abs 1.192e-7. Non-blocking advisory: present-but-UNDEFINED elem_type=0 on value-info now rejected (correct fail-close for typed I/O).

- 2026-07-14T19:05:00Z — Rejected sentinel-leaking UnsupportedOp diagnostics, then approved the final enriched-error plus loader fail-fast validation solution merged in `00cda89`. Also reviewed pipeline API seams GREEN.


## 2026-07-14T20:05:00Z — Loader validation review
Reviewed Batty's unified load-time `validate_model()` checks for unsupported control flow and dangling tensor references 🟢. Both disk/bytes and session load paths are covered; merged as `2a99eec`.

- 2026-07-15 — Reviewed the complete cross-platform oneDNN wheel branch; approved `ef89a95`.

### 2026-07-16T00:00:00Z — Performance-and-design wave
Reviewed the threading change; post-fix verdict was needs-review/ship.

- 2026-07-16T00:00:01Z — 🟡 Approved Leon's guarded contiguous-f32 native CPU Mul fast path (`347060f`); aliasing, striding, broadcasting, and non-f32 remain generic. Independent decode result: +6.35%.

### 2026-07-16T00:00:03Z — Decode thread-cap safety review cycle
Rejected Deckard's initial thread-cap environment parsing because unusable or huge values could abort inference or trigger unsafe worker creation; Deckard was locked out of the revision. Re-reviewed Sebastian's pure bounded resolver and cleared it: invalid values fall back safely, valid values cap at available parallelism, prefill remains untouched, and 413 tests passed.

### 2026-07-16T00:00:00Z — nxrt Python genai review cycle
Rejected Rachael's `RefCell`/`#[pyclass(unsendable)]` Engine because cross-thread use caused `PanicException`, and because the lockfile was stale. Cleared Sebastian's `Mutex`/sendable revision (`41d8c31`): GIL-released, fail-fast `try_lock` access preserves engine safety; the locked build and 19 Rust binding tests passed.

### 2026-07-16T10:18:11Z — Native CUDA decode M1a review
Cleared Deckard's `f795d45` executor EP-polymorphism refactor: it preserves CPU-only construction and behavior, adds no CUDA/device branching, unsafe, downcasting, or steady-state virtual dispatch. Independent validation confirmed 413/413 CPU EP tests and exact eight-token output; M2 device tensors/on-device coverage is next, pending the design decisions and CUDA GQA/device-KV prerequisites.

## 2026-07-16T00:00:00Z — CUDA M2 op-coverage review
- 🟢 Cleared Luv's `16c1e92`: exact CPU/CUDA f32 domain/opset registration parity for SiLU and standard SimplifiedLayerNormalization, stable math, and independent references.
- Confirmed 114/114 CUDA tests passed.

## 2026-07-16T00:00:00Z — CUDA M2 executor safety review cycle
- 🔴 Rejected Deckard's `1a2deca` CUDA executor wiring after confirming Qwen token parity but finding unsafe CUDA SequenceAt host-pointer dispatch and Scan host writes to device allocations. Deckard was locked out and Leon assigned the repair.
- 🟢 Cleared Leon's `5c0f05f`: SequenceAt synchronously uploads into CUDA storage, Scan uses CPU staging plus child-executor H2D, and substantive CUDA control-flow parity passed. Qwen tokens remained exact; session CPU 112/112 and CUDA EP 117/117 passed.

## 2026-07-16T14:20:00Z — SM-general CUDA NVRTC review
- 🟢 Cleared Wallace's `b56c5cb`: selected-device capability derives NVRTC PTX/CUBIN targets without a hardcoded SM90; unsupported-PTX fallback remains correct. Validated 117 CUDA tests and all 6 GQA tests.


## 2026-07-16T19:27:57+0000Z — Native backend selector review cycle

🔴 Rejected Deckard's `66ec4b8` and locked Deckard out over op-type-only detection plus silently ignored speculation, pipeline, and device selections. 🟢 Cleared Batty's `2ae464b`: exact domain/opset v1 detection and explicit unsupported errors are covered; default Auto→ORT behavior remains intact.

## 2026-07-16T23:58:29+0000 — GAFF ChildExecutor review

- 🟡 Advisory-cleared Sapper's ChildExecutor foundation: lexical captures/scoped initializers and nested behavior are correct; 114 session tests passed.
- Follow-up `gaff-exec-cache-lru`: one-entry cache safely recompiles `A → B → A`; add multi-signature caching plus permanent shadowing/nested-cache tests.

## 2026-07-17T00:19:41+0000 — GAFF If review

- 🟢 Cleared Sapper's `7a369ef`: branch cache separation, fresh captures, initializers, and output checks are correct; 117 session tests passed.


## 2026-07-17T00:58:13Z — GAFF Loop reject-to-clear cycle

- 🔴 Rejected Sapper's `8052891` Loop because eager scan reservation from input-controlled `M` enabled an `i64::MAX` early-exit capacity-overflow DoS, and loop-carried shapes were not invariant.
- Sapper was locked out; 🟢 cleared Leon's final `f6e8ba6` revision after huge-trip early-exit and carried-shape regressions. Session build and 121 tests passed; only `Scan` remains.

## 2026-07-14T00:00:00Z — Scan and QMoE final safety gates

- Cleared Leon’s Scan overflow repair and Holden’s own QMoE follow-up: allocations are bounded at `isize::MAX`, and valid odd affine-int4 block rows remain accepted.

- 2026-07-18 Scribe: RoPE review rejected silent invalid positions, then approved validated commit 74a891b.

- 2026-07-18: PR triage recorded five merged PRs, Attention landing, and PR #25 lifecycle follow-up.

## 2026-07-18T06:30:00Z — CUDA GLM standard claim gates

- Hardened CUDA GLM standard-op claim gates in `030faa1`, fixing claim-then-fail dtype/attribute gaps across RMSNorm, RoPE, TopK, CumSum, Gather, GatherElements, ScatterElements, Where, and Expand.
- Added shared `standard_claims.rs` validation and reported CUDA EP suite success: 238 passed, 0 failed.

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.


## 2026-07-21 — Wave-2 and CI milestone
CI now covers all 27 offline crates with warnings-as-errors and native Windows ARM64. Capture-safe native fp16 CUDA decode wave 2 stacked GQA prep fusion, warp-shuffle RMSNorm, and specialized down-projection GEMV on wave 1, reaching 663–672 tok/s on H200 versus ORT GenAI at 657, with zero fallbacks. All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.

## 2026-07-21T11:15:00Z — Long-context GQA review
- 🟢 Approved Sebastian's 16-way split-K change after verifying exact scratch bounds, deterministic merge order, capture safety, parity, and SM portability. Reproduced about 693 tok/s at 1024 tokens versus 647 baseline with no 256 regression.

## 2026-07-21T13:15:00Z — GQA metadata-fold review
- 🟢 Approved Luv’s batch-1 metadata fold after verifying exact metadata parity, poison/latch safety, capture behavior, portability, zero fallbacks, and the independent throughput win.
- 2026-07-21T23:55Z — Revised DS-1 with dtype/rank/element-cap materialization gates after Gaff rejection; Pris approved the landed path.
## 2026-07-22T00:00:00Z — Reviewed BLOCKER #3 int4 zero-point fix

- Holden reviewed Sapper's native CUDA fp16 int4 GEMV explicit-zero-point fix and returned 🟢 GREEN across all five criteria: SM-portability, capture-safety, symmetric no-regress, genericity, and correctness.
- Validation evidence: 6/6 unit tests and 18/18 `matmul_nbits_gpu` integration tests passed; Sapper's fix merged to main as `48de993`.
## 2026-07-23T22-29-16Z — DeepSeek dtod fix review
- 🟢 Cleared Rachael's `1fe314f` dtod synchronization fix after verifying the non-blocking/default-stream race, regression failure without the fix, CUDA gate `202/0`, clippy clean, capture safety, and no meaningful Qwen/Phi perf regression.
