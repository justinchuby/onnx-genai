# Decisions

Canonical, append-only record of accepted team decisions. Only the Coordinator (via Scribe merge) writes here. Agents drop proposals in `decisions/inbox/`.

---

### 2026-07-20T00:00:00Z: Decisions archive rollover
**By:** Scribe
**What:** Archived all 2026-07-12 entries (68 KB) to `decisions/archive/2026-07-20T00-00-00Z-decisions-pre-0713.md`. decisions.md exceeded the 50 KB threshold; entries older than 7 days (relative to 2026-07-20) were moved to archive. Recent 2026-07-13+ entries are retained below.
**Why:** Keep the hot decisions file lean per Scribe charter (>=50KB → archive entries >7 days).

---


### 2026-07-14T02:37:00Z: Decisions archive rollover
**By:** Scribe
**What:** Archived 2026-07-13 entries + early 2026-07-14 entries (W1, W2, Implementation plan) to `decisions/archive/2026-07-14T02-37-00Z-decisions-pre-w3.md` (~90 KB). decisions.md was 127 KB; size-based archival triggered (>50 KB threshold).
**Why:** Keep the hot decisions file lean per Scribe charter. W3 onward (per-layer KV geometry engine consume, review, Pris W5a, K4 multi-layer, Milestones A and B) is retained in the live file.

---


> Earlier entries archived to `.squad/decisions/archive/2026-07-14-part1.md`

### 2026-07-14: `onnx-runtime-ep-cpu` — +17 BERT kernels (op expansion for bert_toy)
**By:** Batty (Engine Dev)
**Commit:** e485a83 | **Reviewed:** 🟡 Chew (numerics)
**What:** 17 new kernels added to `onnx-runtime-ep-cpu`, all registered in `PHASE1_OPS` via `build_cpu_registry`. Elementwise binary: `Sub`, `Mul`, `Div`, `Pow`, `Min` (via existing `broadcast_apply`). Unary: `Sqrt`, `Erf` (A&S 7.1.26 in f64, max abs err 1.39e-7), `Tanh`. Type: `Cast` (fixed-width numeric dtypes, float→int truncates, NaN→bool = true). Reduction: `ReduceMean` (multi-axis, keepdims). Shape/movement: `Shape`, `Unsqueeze`, `Expand`, `Slice` (opset-10 input-driven, negative/stepped ranges). Constant: `Constant` (value/value_float(s)/value_int(s)). GEMM: `Gemm` (transA/transB, alpha/beta, bias broadcast). Dtype-generic byte movers (`elem_size`, `to_dense_bytes`, `write_dense_bytes`) added to `kernels/mod.rs`. 90 tests pass; clippy clean; no new dependencies. Softmax intentionally uses opset-13 per-axis semantics (identical to opset-12 coerce on last axis — all bert_toy Softmax nodes). Loader gaps flagged: `Slice`/`Expand`/`Constant` shape inference needed (owner: Deckard).
**Why:** Supplies the op coverage gap for the BERT-on-CPU milestone; executor needs no changes.

---

### 2026-07-14: `onnx-runtime-loader` — const-fold-lite shape inference (Slice/Expand/Constant)
**By:** Deckard (Systems Dev)
**Commit:** b6f032e | **Reviewed:** 🟢 Gaff (correctness)
**What:** Bounded partial evaluator (`ConstEnv: HashMap<ValueId, KnownVal>`) filled in topo order alongside existing shape rules. `KnownVal` = rank-0/1 integer tensor with `IntElem::Const(i64) | IntElem::Sym(SymbolId)`. Bound: rank ≤ 1, numel ≤ 1024 (`MAX_FOLD_ELEMS`), integers/bools only. Value-propagation ops: `Constant`, `Shape` (emits Sym for symbolic dims), `Identity`, `Cast` (integral only), `Unsqueeze`, `Squeeze`, `Concat`, `Gather` (axis-0, 1-D), `Slice` (opset-10), `Reshape`, `Add`/`Sub`/`Mul`/`Div`/`Min`/`Max` (any symbolic operand → fresh symbol). Shape rules added: `Reshape` (symbolic-aware), `Slice` (rank-preserving, symbolic bounds), `Expand` (broadcast vs const/sym target). On `bert_toy_optimized.onnx`: unresolved values 135→50; all 50 residuals are genuine rank-0 scalar `Constant`s. No `UnresolvedShape` for any structural op. Position-slice chain stays symbolic (data-dependent, by design). 27/27 tests pass (including `bert_toy_optimized_every_value_resolves` on real model); clippy clean; `#![forbid(unsafe_code)]` retained; public API unchanged.
**Why:** Session executor errors `UnresolvedShape` for any value the loader leaves shape-less. Batty's ep-cpu data-movement kernels require pre-allocated output views with correct shapes.

---

### 2026-07-14: Chew review — session executor Track D (🟢)
**By:** Chew (correctness/numerics reviewer)
**Target:** `squad/ort2-session` @ `edbc3fd` | **Verdict:** 🟢 SHIP-with-minor-advisories
**What:** Verified topological order (Kahn's algorithm, min-heap tie-break, cycle detection), value dependency resolution (one buffer per `ValueId`, SSA-disjoint), view materialization (contiguous strides, zero offset correct for dedicated per-value buffers), initializer/input binding (dtype+shape validated), output collection (correct prefix slice), shape-keyed cache (no collision — fresh `NodeId` per node). Test references hand-verified in Python (MatMul→Add→LayerNorm→Relu chain). Advisories (non-blocking): optional-input compaction may shift positional alignment for gappy-optional ops; cache key omits dtypes. No correctness bug found.

---

### 2026-07-14: Holden review — session executor Track D (🟡)
**By:** Holden (safety/soundness reviewer)
**Target:** `squad/ort2-session` @ `edbc3fd` | **Verdict:** 🟡 SHIP-with-advisories
**What:** All 5 invariants held: (1) view bounds gated on every input + output before dispatch via `view_bounds`+`?`; (2) single-free via `Option::take` in `Tensor::drop`, `drain` in `Executor::drop`; (3) no cross-EP free — allocator carried on returned `Tensor`; (4) copy size validated before `copy_nonoverlapping`; (5) host malloc is global. Aliasing claim verified: in-place ops cause `CycleDetected` at build. Miri clean. Advisories: A1 — mid-run error path leaks output buffers (`DeviceBuffer` has no `Drop`); A2 — unchecked i64 arithmetic in `view_in_bounds` (theoretical overflow); A3 — cache key omits dtypes. None blocking.

---

### 2026-07-14: Holden review — session dynshape (🟡)
**By:** Holden (safety/soundness reviewer)
**Target:** `squad/ort2-session-dynshape` | **Verdict:** 🟡 SHIP-with-advisories
**What:** Invariant #1 (view bounds) holds against new run-scoped buffers — gate keys off real `buf.len()` not assumed shape, so even stale `buffer_shapes` cannot bypass it. Buffer-reuse cannot yield undersized-but-passing buffer (two independent layers: correct sizing + real-length gate). No new aliasing introduced. Single-free/no-leak on re-allocation — `deallocate(old)` before `allocate`, Miri-clean across batch 2→3→2 reuse test. 14/14 tests pass. Advisories: H-D1 — unchecked `dims.iter().product()` overflows mod 2⁶⁴ and gate is congruent (very low reachability); H-D2 — stale `buffer_shapes` if `allocate` fails post-dealloc (clean error, not UB); Holden-A1 (pre-existing) — mid-run error-path buffer leak unchanged.

---

### 2026-07-14: Holden review — C ABI Track E (🟢)
**By:** Holden (safety/soundness reviewer)
**Target:** `squad/ort2-capi` | **Verdict:** 🟢 SHIP
**What:** Verified all 6 FFI soundness axes: (1) null-guards on every handle and pointer before deref, returning `InvalidArgument`; (2) every fallible body in `catch_unwind` via `guard`; (3) `Box::into_raw`/`Box::from_raw` once each, null-tolerant releases, atomic commit in `ort2_run`; (4) `create_tensor` validates `data_len == storage_bytes(numel)` before slice construction; (5) `CStr::from_ptr(..).to_str()` with UTF-8 error → `InvalidArgument`; (6) 12/12 tests pass, Miri clean. Advisories (non-blocking): A1 — release fns not in `guard` (panic-free today but relies on `Drop` invariants); A2 — `storage_bytes` unchecked multiply (only reached inside `guard`, bounded by prior validation).

---

### 2026-07-14: Chew review — ep-cpu BERT kernels +17 (🟡)
**By:** Chew (correctness/numerics reviewer)
**Target:** `squad/ort2-epcpu-ops` @ `4f2465e` | **Verdict:** 🟡 SHIP-with-advisories
**What:** 90/90 tests pass. No blocking numeric bug for bert_toy. Independently verified: Softmax stability vector `[1000,1001,1002]→[0.090,0.245,0.665]` ✓; broadcast `[3,1]·[1,4]→[3,4]` ✓; Erf max abs err 1.39e-7 ✓; Gemm `[[58,64],[139,154]]` ✓. Elementwise binaries, ReduceMean, Gemm, Slice, Cast, data-movement kernels spec-faithful. Advisories: A1 — Softmax uses opset-13 per-axis (bit-identical to opset-12 coerce only on last axis; bert_toy Softmax all last-axis but model not in-repo — conformance harness must confirm); A2 — `Min` uses `f32::min` (returns non-NaN; ONNX propagates NaN — no bert_toy impact); A3 — Cast float→int saturates on overflow vs ORT UB (documented, no bert_toy impact). Non-blocking hardening for A1 (opset<13 guard when axis≠rank-1) assigned to Roy or Deckard.

---

### 2026-07-14: Gaff review — loader const-fold-lite shape inference (🟢)
**By:** Gaff (correctness reviewer)
**Target:** `squad/ort2-loader-shapeinfer` | **Verdict:** 🟢 SHIP
**What:** No wrong constant found. Every fold aborts via `?`/`None` on missing/non-integer/unfolded operands — never invents a constant. Verified: Gather symbolic index → `None` ✓; Slice requires `all_const` starts/ends ✓; Concat requires all inputs in env ✓; binop any-Sym → fresh symbol (unit-tested) ✓; Reshape handles -1/0 correctly ✓; Slice clamp math correct ✓. Symbolic identities propagate via interned `SymbolId`. Bounds enforced at every entry point. `bert_toy_optimized_every_value_resolves` ran on real model (257 KB, not skipped) — no `UnresolvedShape` on structural ops; position-slice chain correctly symbolic. Advisories: A1 — `Div` truncates toward zero vs floor for negative operands (no positive-dim impact; elem_to_dim maps negatives to fresh symbol); A2 — `Shape` of unresolved input folds to rank-0 (pre-existing, no bert_toy impact). 27/27 tests pass.

### 2026-07-14: ORT2 must support ORT's EPContext node (com.microsoft)
**By:** Coordinator (Justin Chu)
**What:** ORT2 must support ORT's on-disk EPContext contrib operator (domain com.microsoft, variadic inputs/outputs) — distinct from the internal `EpContext` cache struct. Scope: (1) Loader parses EPContext attrs (main_context, ep_cache_context, embed_mode, ep_sdk_version, hardware_architecture, partition_name, source, notes, max_size); (2) Session/EP-API dispatches an EPContext node to the EP whose `source` attribute matches, feeding blob to the EP's load_context path; (3) Generation via ep.context_enable / ep.context_file_path / ep.context_embed_mode session options; (4) C-API surfaces those options. Model-agnostic: dispatch by `source` attribute only — no hardcoded EP names. Roy to author design in docs/ORT2.md (branch squad/ort2-epcontext-design).
**Why:** Central to EP ecosystem interoperability. ORT2 must consume/emit pre-compiled EP-binary models that the ORT ecosystem produces (QNN, OpenVINO, TensorRT, etc.).

---

### 2026-07-14: ORT2 shape inference reference: onnx-shape-inference
**By:** Coordinator (Justin Chu)
**What:** Shape inference (optimizer `ShapeInference` pass + evolution of `onnx-runtime-loader/src/shape_inference.rs`) must follow patterns from https://github.com/justinchuby/onnx-shape-inference: (1) extensible per-op registry keyed by (domain, op_type, opset_version); (2) symbolic dim arithmetic (SymPy-style expr trees over `Dim::Symbolic`); (3) shape DATA propagation as first-class subsystem tracking known values of shape tensors through Shape→Slice→Concat→Reshape chains; (4) strict/permissive merge policies for unifying inferred vs declared shapes.
**Why:** User-designated reference. Keeps inference extensible, opset-aware, model-agnostic, and feeds the optimizer richer shape info.

---

### 2026-07-14: ORT2 `onnx-runtime-optimizer` — Phase-2 optimizer crate
**By:** Roy (Lead)
**What:** New `crates/onnx-runtime-optimizer/` crate (`#![forbid(unsafe_code)]`, depends only on `onnx-runtime-ir`). Implements `OptimizationPass` trait + `PassContext` (empty, `#[non_exhaustive]`) + `run_passes` + `OptimizerError`, and three passes: `ConstantFolding` (integer only, ≤1024 elems, checked arithmetic, fixpoint; folds `Constant`/`Shape`/integer binops), `DeadNodeElimination` (backward reachability from outputs), `OpFusion` (escape-safety rule: non-final matched outputs must stay within matched set; reuses final `ValueId`; patterns: MatMul+Add+Relu→FusedGemm, MatMul+Add→FusedMatMulBias, 9-op LayerNorm). Default pipeline: ConstantFolding → DCE → OpFusion. bert_toy: 384→278 nodes, 0 Constants, 32 FusedMatMulBias; LayerNorm fusion correctly declines (DAG-shaped shared `mean`). 26 unit + 1 real-model integration tests; clippy clean.
**Why:** Foundation for all Phase-2+ graph rewriting; pass contract and fusion safety invariant locked before more passes added.

---

### 2026-07-14: Gaff review — optimizer structural integrity (🟢)
**By:** Gaff (graph/IR integrity reviewer)
**Target:** `squad/ort2-optimizer` @ `87a16d9` | **Verdict:** 🟢 SHIP
**What:** All 6 integrity checks HELD. Node removal/GC via `remove_node` correct; fusion removes last-first, reuses final `ValueId`; ConstantFolding `needed` guard prevents stale initializer; arena safety (stale-id checked before deref); DCE+fusion interaction verified adversarially. `Graph::validate()` postcondition verified as genuinely biting (injected dangling edges and bogus consumer links → `Err`). 27 tests pass; clippy clean. Advisories (non-blocking): A1 — external-input ordering structural not schema-aware; A2 — validate() debug-only (intentional per §18.1).

---

### 2026-07-14: Chew review — optimizer correctness (🟡)
**By:** Chew (correctness reviewer)
**Target:** `squad/ort2-optimizer` @ `87a16d9` | **Verdict:** 🟡 SHIP-with-advisories
**What:** All three passes semantics-correct. OpFusion escape-safety invariant correct (necessary-and-correct condition for not deleting an observed value). ConstantFolding never folds non-const inputs, checked arithmetic (overflow aborts, not wraps), fixpoint correct. DCE from outputs (never from inputs). bert_toy verified: 384→278 nodes, 0 Constants, 32 FusedMatMulBias, validate() clean. Advisories: A1 — fused ops emitted in default ONNX domain (must use private domain e.g. com.ort2.fused before any kernel binds; tie refinement to kernel introduction); A2 — greedy spine matcher under-fuses on multi-successor (never miscompiles); A3 — single-output final-node restriction.

---

### 2026-07-14: Roy — BERT-toy conformance milestone ACHIEVED
**By:** Roy (Lead)
**Branch:** `squad/ort2-bert-conformance`
**What:** `bert_toy_optimized.onnx` (opset 12, 384 nodes) runs end-to-end through `onnx-runtime-session` on CPU and matches onnxruntime 1.27.0/CPUEP. Max error: prediction_scores 1.19e-7 (tolerance 2e-3), seq_relationship_score 6.05e-9. Zero Phase-1 cross-crate bugs. One session-local fix: position-embedding Slice takes a data-dependent `Shape→Cast→Min→Cast` extent requiring JIT dynamic-shape resolution in the executor — model-agnostic (dispatch on op type only; ops without JIT resolution surface `UnresolvedShape`). 15 tests pass; Miri clean.
**Why:** Phase-1 exit milestone. Proves the full stack (loader, ep-cpu, ep-api, session) composes correctly on a real transformer with correct numerics.

---

### 2026-07-14: Chew review — BERT conformance JIT output sizing (🟡)
**By:** Chew (correctness reviewer)
**Target:** `squad/ort2-bert-conformance` | **Verdict:** 🟡 SHIP-with-advisories
**What:** Slice sizer is character-for-character mirror of Slice kernel — buffer always equals what kernel writes. `buffer_as_i64` LE decode correct. JIT loop ordering correct. Conformance harness sound (allclose semantics, both outputs, deterministic inputs). Advisories: A1 — Slice count math duplicated verbatim (extract shared `slice_axis_count` helper — structural risk of silent drift); A2 — pre-existing degenerate Slice corner (not BERT-impacting); A3 — multi-output index robustness when extending beyond Slice; A4 — tolerance comment vs allclose code mismatch.

---

### 2026-07-14: Holden review — BERT conformance JIT alloc/dealloc (🟡)
**By:** Holden (safety/soundness reviewer)
**Target:** `squad/ort2-bert-conformance` | **Verdict:** 🟡 SHIP-with-advisories
**What:** All 4 soundness invariants HELD. view_in_bounds gate on every input+output before dispatch including JIT-sized outputs (JIT sizes first, gate validates JIT shapes against JIT-resized buffers). No use-after-free (error path exits before dealloc/alloc loop). dealloc-before-alloc ordering safe (no live `TensorView` aliasing freed buffer). No new `unsafe`. Miri clean (0 UB; -Zmiri-disable-isolation for disk-read conformance test). Advisories: H-D1 carry-over (unchecked dims.product + storage_bytes in JIT path — same as ensure_buffer; non-regression); multi-output index panic when op returns fewer shapes than outputs.

---

### 2026-07-14: Batty — ep-cpu + session Phase-1 hardening (6 advisories + capi fix)
**By:** Batty (Engine Dev)
**What:** (1) Softmax opset≤12 vs ≥13 dual semantics via `coerce_2d` flag + dual registry (SoftmaxLegacy@1, Softmax@13); `effective_opset` plumbed end-to-end. (2) Min/Max NaN-propagation — explicit `is_nan()` guard before `f32::min/max`. (3) Cast saturate — `num_to_int!` macro converting directly to target type (no i64-intermediate-then-wrap). (4) `checked_numel` + `SessionError::ShapeOverflow` at both alloc sites (H-D1 preliminary). (5) Multi-output `dynamic_output_shapes` guard (`OutputShapeCountMismatch` before index). (6) Slice geometry extracted to shared `slice_plan` + `slice_axes_steps` helper (kernel + sizer share one impl). Also fixed capi `map_session_error` non-exhaustive match — added explicit arms for `SymbolConflict/RankMismatch/UnresolvedShape/ShapeOverflow/OutputShapeCountMismatch` (no catch-all `_`); all-crate build restored.
**Why:** Real correctness gaps (wrong Softmax for non-last-axis opset≤12, NaN swallowed, Cast garbage, panic on future multi-output op) closed before more models arrive. Holden's `view_in_bounds` gate preserved untouched.

---

### 2026-07-14: Holden review — ep-cpu hardening (🔴 → Deckard)
**By:** Holden (safety/soundness reviewer)
**Target:** `squad/ort2-hardening` @ merge-base vs main | **Verdict:** 🔴 REJECTED
**What:** Checks 2–6 HELD (view_in_bounds gate intact, multi-output guard, opset plumbing, capi FFI, tests/clippy/Miri). Check 1 FAILED: `checked_numel` closed dims-product overflow but `DataType::storage_bytes(numel)` still computed `count * byte_size` unchecked. Shape `[2^61]` of f64: checked_numel OK (=2^61), storage_bytes wraps to 0, `.max(1)`→1-byte alloc; `view_in_bounds` i64 gate also wraps → passes → heap OOB in release. **Batty locked out of H-D1 storage-sizing artifact. Fix assigned to Deckard** (or another non-Batty implementer). Re-review by Holden required before merge.
**Why:** Unchecked overflow reaching allocation = 🔴 per soundness rubric; exact H-D1 class.

---

### 2026-07-14: Deckard — H-D1 three-layer overflow fix
**By:** Deckard (Systems Dev)
**Commits (cherry-picked to main):** dbf2d70, 9dcdc04, f749012
**What:** Layer A (`dtype.rs`): `DataType::checked_storage_bytes(count) -> Option<usize>` — `div_ceil(2)` for sub-byte, `checked_mul(byte_size())` for fixed-width; `storage_bytes` reimplemented on top with `.expect`. Layer B (`executor.rs`): `checked_storage_bytes` helper → `SessionError::ShapeOverflow`; both `ensure_buffer` and JIT alloc routed through it; `.max(1)` after checked multiply. Layer C (`strided.rs::view_in_bounds`): address range computed in i128 with `checked_mul`/`checked_add`; overflow → `EpError::InvalidTensorView`. 4 new regression tests; all crate tests + bert_toy green; clippy clean; no new `unsafe`.
**Why:** Closes H-D1 end-to-end at all three layers identified by Holden's 🔴.

---

### 2026-07-14: Holden re-review — H-D1 fix (🟡 SHIP)
**By:** Holden (safety/soundness reviewer)
**Target:** `squad/ort2-hardening` @ `852f262` | **Verdict:** 🟡 SHIP-with-advisories (prior 🔴 cleared)
**What:** All three fix layers HELD. Layer A: `checked_storage_bytes` correct, regression tests pass. Layer B: both alloc sites checked; `.max(1)` after multiply. Layer C: i128 address math cannot itself overflow (inputs bounded by i64/usize; max value ~2^127 < i128::MAX). Original exploit vector (`[2^61]`×f64) → `ShapeOverflow` at allocation; regression test `bounds_reject_overflowing_address_math` confirms. Tests/clippy clean; Miri unavailable (component not installed). Residual advisories (non-blocking, memory-safe): A1 — `storage_bytes` panics (not graceful error) at capi:350, weights.rs:133, tensor.rs:from_raw_in (caught by catch_unwind; fast-follow owner: Leon); C1 — `addressed_elem_range` min/max accumulated in i64 before i128 widening (adversarial-strides only, not reachable via static shapes).
**Why:** H-D1 heap-OOB fully closed end-to-end. Residual advisories are fail-closed memory-safe nits.

---

### 2026-07-14: Chew review — ep-cpu hardening (🟢)
**By:** Chew (correctness/numerics reviewer)
**Target:** `squad/ort2-hardening` @ `830086e` | **Verdict:** 🟢 GREEN
**What:** All 6 fixes compute ONNX-correct results. Softmax dual semantics numerically verified for [2,2] axis-0 (legacy=[0.032,0.087,0.237,0.644] sums-to-1; per-axis=[0.119,0.119,0.881,0.881] column-wise). Min/Max NaN-propagating `combine()` correct. Cast `num_to_int!` converts directly to target type (no i64 wrap). slice_plan dedup byte-identical to prior inline. capi exhaustive match reasonable (caller-error vs internal-error arms). All tests pass; clippy clean. No lockout.

---

### 2026-07-14: Fact-check — EPContext node design docs/ORT2.md §55 (🟡 one required fix)
**By:** Fact Checker
**Target:** `squad/ort2-epcontext-design` @ `c48f5c4` | **Verdict:** 🟡 SHIP-with-one-required-fix
**What:** Op schema §55.2 — all 10 attributes exact vs contrib_defs.cc (names, types, defaults). Session-option key strings exact vs `onnxruntime_session_options_config_keys.h`. embed_mode semantics correct. main_context semantics correct. Model-agnostic dispatch verified (EpContextRegistry keyed by source string, no hardcoded EP names in dispatch path). ❌ Required fix: §21.4 `ep.context_embed_mode` default stated as `1`; ORT runtime default is `0` (`ep_context_options.cc:40`; header: "0: external file (default)"). Roy must change §21.4 default 1→0 and align `EpContextGenOptions.embed_mode` to `ExternalFile`. Do NOT change §55.2 op-attribute default (`1`) — that is correct. Advisory: TOC not updated for §55/§56 renumber (pre-existing mismatch). Roy not locked out — single-cell fix.
**Why:** Spec must match ORT runtime default to avoid silent wrong behavior when ep.context_embed_mode is unset.

---

### 2026-07-14: CUDA EP kernel stack decided (Phase 2)
**By:** Coordinator (Justin Chu directive; cuda-kernel-research agent, opus-4.8)
**What:** Phase-2 `onnx-runtime-ep-cuda` kernel stack: **Foundation** — `cudarc` (Rust CUDA 11.4–13.3 bindings; HuggingFace Candle GPU backend). **Standard GEMM** — cuBLASLt via `cudarc::cublaslt` (fused epilogue GEMM+Bias+Activation since CUDA 12.0). **Custom fused kernels** (LayerNorm/RMSNorm, RoPE, softmax, elementwise fusions) — CuTe (CUTLASS 3.x C++ templates) → `extern "C"` launchers compiled by `nvcc` in `build.rs`, linked as static lib; `#if __CUDA_ARCH__>=900` SM90 gate with SM80 fallback. **Attention** — Phase 2a: cuDNN fused SDPA via `cudarc::cudnn`; Phase 2b: FlashAttention-3 via `flash_attn_shim.cu` C shim around Tri Dao's `hopper/` csrc. **Trivial elementwise** — NVRTC PTX via `cudarc::nvrtc`. **cuTile DEFERRED to Phase 3**: no Hopper (SM90) support in CUDA 13.1 (Ampere/Ada/Blackwell only), Python-only, no Rust path. Re-evaluate when Hopper ships + C++ path lands (~CUDA 14.x / 2027). Rejected: Rust-CUDA (nightly-only, no CUTLASS), Triton AOT (Python dep), TileLang/Mojo (Python). All custom kernels MUST be shape-driven, dtype-parameterized, arch-gated — no hardcoded model constants. Next: Roy updates docs/ORT2.md §15 after EPContext-design branch merges.
**Why:** Research-grade evaluation with dated citations. cuTile disqualified for primary H100/H200 (SM90) target. cudarc+cuBLASLt+CuTe stack is production-proven.

---

### 2026-07-14: `onnx-runtime-shape-inference` — new crate landed
**By:** Roy (ORT2 architect) — `squad/ort2-shape-inference`
**What:** New pure-Rust crate implementing general symbolic shape inference over frozen `onnx-runtime-ir` Graph. 4 pillars: (1) extensible per-op registry keyed `(domain, op_type)` → version-sorted handlers; (2) `DimExpr` canonical integer polynomial (`BTreeMap<monomial, coeff>`) with checked_div for exact symbolic division; (3) shape-DATA propagation side-table tracking `Shape→Gather→Concat→Reshape` chains; (4) Strict/Permissive merge policies. 40+ op handlers including com.microsoft FusedMatMul, Conv/pool family, all transformer ops. `bert_toy_fully_resolves` (384-node graph, `num_unresolved()==0`). `#![forbid(unsafe_code)]`, 56 tests, clippy clean. Does NOT modify `onnx-runtime-ir`. IR-helper proposal deferred. Stopgaps (loader const-fold-lite, session JIT) left in place pending wiring.
**Why:** General extensible symbolic shape engine needed as foundation for planner/allocator/cost-model work; stopgaps are bounded hacks.

---

### 2026-07-14: Chew review — shape-inference op correctness (🔴 REJECT)
**By:** Chew (correctness reviewer) — `squad/ort2-shape-inference` (42 tests green at review time)
**Verdict:** 🔴 REJECT — one blocking defect: `com.microsoft::FusedMatMul` reused plain `matmul` handler, ignoring `transA`/`transB`/`transBatch` attrs. `A=[8,64]·B=[32,64]ᵀ` (transB=1) produced `[8,64]` + spurious contraction error (correct: `[8,32]`). Fix assigned to **Deckard** (Roy locked out as author). All other ops HELD correct: MatMul, Gemm, broadcast ops, all movement/norm/pool/data handlers. Non-blocking advisories: (a) Reduce opset-18 unresolved-axes degrades to reduce-all; (b) Concat/Cast shape-data dtype hard-coded Int64; (c) GatherElements doc comment misleading.
**Why:** FusedMatMul with transB=1 is pervasive in ORT-optimized transformer attention graphs; wrong shape silently corrupts every downstream op.

---

### 2026-07-14: Holden review — shape-inference DimExpr soundness (🔴 REJECT)
**By:** Holden (soundness reviewer) — `squad/ort2-shape-inference`
**Verdict:** 🔴 REJECT — one real soundness bug: `DimExpr` `add`/`sub`/`mul` used unchecked i64 arithmetic. Debug build panics on `2^80` product (e.g. Size of large tensor); release build wraps to bogus concrete dim. Secondary: `checked_div` `coeff % div_coeff` / `coeff / div_coeff` unguarded against `i64::MIN / -1` overflow-panic. Fix assigned to **Deckard**. All other items HELD: canonicalization uniqueness, checked_div correctness, fresh-symbol range safety (anon floor > 0x8000_0000), merge-policy soundness, ShapeData 1024-elem cap, `#![forbid(unsafe_code)]`, miri clean. Advisory: `fresh_symbol` counter unchecked `+=1` (adversarial exhaustion).
**Why:** i64 wrap can silently write a wrong concrete dimension without error; debug panic is reachable on normal large tensors.

---

### 2026-07-14: Gaff review — shape-inference registry/driver/API (🟢 APPROVE)
**By:** Gaff (graph-integrity / API-design) — `squad/ort2-shape-inference` (4 integrity probes)
**Verdict:** 🟢 APPROVE. Registry dispatch (domain normalization, opset boundary, duplicate-replace), topo-order driver (correct read-before-write, field-level write-back via `value_mut`, multi-output, transactional on failure — proven by probe), shape-data side-table integrity (per-call HashMap, no cross-call stale leakage), API design (`InferenceRegistry`/`infer_graph`/`infer_node`, `thiserror` errors, no panic in public paths) all HELD. IR contract NOT modified. `onnx-runtime-ir` diff is zero lines. Roy not locked out. Fix agent NOT required.
**Why:** Structural/API correctness is separate concern from per-op formulas (Chew) and algebra soundness (Holden).

---

### 2026-07-14: Deckard fix — shape-inference (FusedMatMul transpose + DimExpr overflow)
**By:** Deckard (Roy locked out per reviewer-protocol) — commit `09988f3` on `squad/ort2-shape-inference`
**What (blocking fix 1 — Chew 🔴):** New dedicated `fused_matmul` handler reads `transA`/`transB`/`transBatchA`/`transBatchB` (ORT's real A/B-suffixed names) and `alpha` (shape-neutral). `apply_fused_trans` reorders operands to plain `[batch…, row, col]` mirroring ORT `FusedMatMulShapeInference` in contrib_defs.cc; then calls shared `matmul_shape`. Rank-≤1 unchanged (matches ORT). 7 new FusedMatMul tests. **What (blocking fix 2 — Holden 🔴):** All `DimExpr` combiners use `checked_*`; overflow degrades to `DimExpr::overflow()` sentinel (poisoning, no alias via fresh-symbol bypass of intern cache). `checked_div` guards `i64::MIN/-1`. `ConstantOfShape` numel uses `checked_mul` fold. **Advisories applied:** `broadcast_dim` Permissive→fresh symbol; `fresh_symbol` saturating_add; Concat/Cast dtype from first operand; GatherElements doc corrected; Reduce opset-18 unresolved axes. **Result:** 69 tests green debug+release, clippy clean.
**Why:** Both 🔴 rejects fully addressed; advisory items applied opportunistically.

---

### 2026-07-14: Chew re-review — shape-inference FusedMatMul fix (🟢 SHIP)
**By:** Chew (correctness reviewer) — commit `09988f3`, fix author Deckard
**Verdict:** 🟢 SHIP — 🔴 blocker fully resolved. Dedicated `fused_matmul` handler verified line-for-line against ORT contrib_defs.cc: batch prefix (`trans_batch?1:0` .. `trans_batch?rank-1:rank-2`), row (`trans?rank-1:(trans_batch?0:rank-2)`), col (`trans?(trans_batch?0:rank-2):rank-1`), rank-≤1 unchanged. Cited case `[8,64]·[32,64]ᵀ → [8,32]` correct. 7 new FusedMatMul tests pass. All 3 advisories applied. **69 tests / 0 failed.** Roy and Deckard both locked out of this artifact for any future revision cycle.
**Why:** Required re-review after 🔴 reject; confirms fix matches ORT upstream source exactly.

---

### 2026-07-14: Holden re-review — shape-inference DimExpr overflow fix (🟢 GREEN)
**By:** Holden (soundness reviewer) — commit `09988f3`, fix author Deckard
**Verdict:** 🟢 GREEN — rejection fully addressed, no new soundness bug. Every combiner overflow-safe (add/sub/mul all `checked_*`). Overflow-degrade contract sound: `overflow()` sentinel has `overflow:true`, `as_const`/`as_symbol` both return `None`, poison propagates, **no symbol aliasing** (`SymbolInterner::lower` checks `is_overflow()` first → fresh symbol, bypasses equality cache). `checked_div` guards `i64::MIN/-1`. `ConstantOfShape` numel fold can't overflow first. `broadcast_dim` degrade-to-fresh correct. `fresh_symbol` saturating_add. **69 tests green debug AND release.** Non-blocking advisory: `movement.rs` slice-index normalization uses raw i64 on attribute-supplied indices (pre-existing, not a regression from `09988f3`; filed as follow-up).
**Why:** Required re-review after 🔴 reject; confirms no wrap, no debug panic, no bogus alias on new overflow path.

---

---

### 2026-07-14: ORT2 shape-inference wiring — loader owns static inference; const-fold-lite retired (Roy, roy-14)
**By:** Roy (ORT2 architect) — merged `98a3310`
**What:** Wired `onnx-runtime-shape-inference` into the loader. `build_from_bytes_with_weights` now calls `registry.infer_graph(...)` with `MergePolicy::Permissive` after graph build. Loader gained `LoaderError::ShapeInference(#[from] ...)`. Deleted `crates/onnx-runtime-loader/src/shape_inference.rs` (~1.1k LOC `const-fold-lite` `KnownVal`/`ConstEnv` pass) — no back-compat shim (pre-release). Session JIT (`dynamic_output_shapes`) retained as fallback for genuinely data-dependent extents. Fixed `broadcast_dim` in `context.rs` to keep the smaller `SymbolId` representative (not mint fresh) when two symbolic dims meet at a broadcast axis — fixes Expand-contamination regression where data-dependent `from_slice_01` symbols contaminated downstream values. Added `add_two_distinct_symbols_keeps_named_representative` test. `bert_toy` conformance: max_abs 1.192e-7 (unchanged).
**Why:** Loader now owns static shape inference as the architectural seam. const-fold-lite was strictly subsumed by the general crate.

---

### 2026-07-14: ORT2 shape-inference wiring — correctness/conformance review 🟢 (Chew, chew-17)
**By:** Chew (ORT2 correctness reviewer) — reviewed `f4141b9`
**Verdict:** 🟢 GREEN — SHIP
**What:** Broadcast-semantics change conformance-safe (smaller-id always prefers pre-existing session-bindable graph symbol due to `ANON_SYMBOL_FLOOR=0x8000_0000` invariant). const-fold-lite deletion safe (no persistent `env`; optimizer has independent pass). Declared-shape merge preserved. `bert_toy` conformance unchanged (1.192e-7). All op rules pass (52 tests). Advisories: A1 — `broadcast_dim` comment framing slightly misleading ("named" should be "lower-id graph"); A2 — `merge_shapes` both-symbolic keeps inferred over declared (pre-existing, harmless).
**Why:** Required review; confirms no conformance regression.

---

### 2026-07-14: ORT2 shape-inference wiring — soundness review 🟢 (Holden, holden-10)
**By:** Holden (ORT2 soundness reviewer) — reviewed `f4141b9`
**Verdict:** 🟢 GREEN
**What:** Symbol-unification sound (overflow sentinel blocks `as_symbol()`, so `broadcast_dim` representative arm unreachable for overflow exprs; deterministic/order-independent; single topo-pass so no convergence issue). Loader seam transactional (graph mutated only after full write-back; on error graph dropped, never half-mutated). JIT fallback byte-for-byte unchanged (only comment diff in executor.rs). No regressions to `view_in_bounds`/`checked_storage_bytes`/unsafe. Full ORT2 suite green debug+release (0 failures). Advisory: fail-fast coupling — a false-positive op-rule error under Permissive now blocks the load; Chew's op-formula pass should confirm none fire on BERT/opset-12.
**Why:** Required review; confirms soundness.

---

### 2026-07-14: ORT2 IR dtype hardening — ONNX numbering fix + float8/float4 + unknown handling (Deckard, deckard-11)
**By:** Deckard — merged `909f0a0`
**What:** Fixed two wrong `DataType` discriminants in `crates/onnx-runtime-ir/src/dtype.rs`: `Float8E5M2: 18→19`; `Uint4: 23→21`. Added `Float8E4M3FNUZ=18`, `Float8E5M2FNUZ=20`, `Float4E2M1=23` (sub-byte, `bit_size=4`, `is_float=true`, `checked_storage_bytes=count.div_ceil(2)`). All classifiers (`is_float`, `is_int`, `is_sub_byte`, `byte_size`, `bit_size`) and `from_onnx`/`to_onnx` updated. Loader attribute decode: hardened `TENSORS|SPARSE_TENSOR|SPARSE_TENSORS|TYPE_PROTOS` from silent `Ok(None)` to `Err(LoaderError::GraphBuild(...))`. Unknown-rank `Shape` gap documented (fixing requires `Value::shape: Option<Shape>` across frozen IR — deferred). Full ORT2 suite 243 tests green debug+release.
**Why:** Silent `Float4E2M1(23)→Uint4` corrupt-decode and `Uint4(21)→None` load failure are critical bugs for the Gemma quantized-model path.

---

### 2026-07-14: ORT2 IR dtype hardening — numbering correctness review 🟢 (Chew, chew-18)
**By:** Chew (ORT2 correctness reviewer) — reviewed `f965f0b`
**Verdict:** 🟢 APPROVE
**What:** Every discriminant independently verified against ONNX `TensorProto.DataType` spec (all 21 variants, rows 1–23 excl. COMPLEX). `to_onnx = self as i32` from `#[repr(u8)]` confirmed correct. `is_float` includes Float4E2M1 and all float8s; `is_int` excludes them. `byte_size=0`/`bit_size=4`/`checked_storage_bytes=div_ceil(2)` for Float4E2M1 correct. Round-trip and unknown-value tests comprehensive. Advisory: vendored `onnx.proto3` stops at INT4=22 — `FLOAT4E2M1=23` not in-repo; verified against upstream onnx/onnx instead (onnx#4728). Recommended follow-up: bump vendored proto. Owner: Roy/Batty/Leon (Deckard locked out).
**Why:** Required review; confirms numbering correct.

---

### 2026-07-14: ORT2 IR dtype hardening — soundness review 🟡 (Holden, holden-11)
**By:** Holden (ORT2 soundness reviewer) — reviewed `f965f0b`
**Verdict:** 🟡 APPROVE-WITH-FOLLOW-UP
**What:** Sub-byte Float4E2M1 routes through `div_ceil(2)` path (never unchecked multiply). Float8-FNUZ normal 1-byte path. `#![forbid(unsafe_code)]` intact. No new unsafe/unwrap/panic. Fail-closed attr hardening safe (GRAPH/GRAPHS/TENSOR/TypeProto still handled). Residual gap: `value-info` and `attribute-tensor` decode sites (`graph_builder.rs:232,241,357,365,374`) still use `.unwrap_or(Float32)` — silent mislabel for COMPLEX64/future dtypes. Deckard's PROGRESS.md claim "loader surfaces as LoaderError" is accurate only for initializer weights (weights.rs), not these sites. Required follow-up before complex/unmodeled-dtype milestone: make these sites return `Result`. Owner: Roy/Batty/Leon. ORT2 suite 300 tests green debug+release.
**Why:** Required review; net soundness improvement with non-blocking residual gap.

---

### 2026-07-14: Loader dtype-decode sites all fail closed (consolidate silent-Float32 fallbacks)
**By:** Leon (leon-10, opus)
**What:** Made every `DataType::from_onnx(raw) -> None` decode site in `onnx-runtime-loader` fail closed, closing the silent-wrong-type gap Deckard's weights-only hardening left behind. Two-part change:

1. **Fail-close consolidation.** Added `LoaderError::UnsupportedDataType { raw: i32, context: String }` (`crates/onnx-runtime-loader/src/lib.rs`), a generalized variant carrying the raw ONNX i32 plus a context string. Migrated the existing weight path to it and converted all remaining silent `.unwrap_or(DataType::Float32)` (and the map-key `.unwrap_or(DataType::Int64)`) decode sites to it via a new `decode_dtype(raw, || context)` helper in `graph_builder.rs`. Call sites changed:
   - `weights.rs` `resolve_initializer` — reworded to the new variant.
   - `graph_builder.rs` initializer value — `context: "initializer '{name}'"`.
   - `graph_builder.rs` `type_proto_to_dtype_shape` TensorType + SparseTensorType element type (value-info) — `context: "value-info '{name}'"`.
   - `graph_builder.rs` `convert_tensor` (Constant/attribute inline tensors) — `context: "attribute tensor '{name}'"`.
   - `graph_builder.rs` `convert_type_proto` TensorType + SparseTensorType element + MapType key.
   - Preserved intentional non-dtype defaults (untyped value-info, non-tensor container placeholders).

2. **Vendored proto bump (doc/consistency only).** `proto/onnx.proto3` `enum DataType` gained `FLOAT4E2M1 = 23`. No runtime behavior change.

**Tests added:** `unknown_value_info_dtype_is_load_error` and `unknown_attribute_tensor_dtype_is_load_error` in `tests/loader.rs`. All 15 loader tests + 40 ir tests green debug+release. bert_toy conformance max_abs 1.192e-7.

**Branch:** `squad/ort2-dtype-failclose` — merged `06a2423` → `a822a21`.

**Why:** Holden's finding: value-info and attribute-tensor sites still silently relabeled unmodeled dtypes as Float32 after Deckard's weights-only hardening. Failing closed consistently at every decode site guarantees clean contextual errors.

---

### 2026-07-14: Loader dtype fail-close — soundness review 🟢 (Holden, holden-12)
**By:** Holden (ORT2 soundness reviewer) — reviewed `a822a21`; verifying closure of own prior finding
**Verdict:** 🟢 GREEN — finding fully closed, no over-reach, no regressions.

**What:** Grepped entire loader crate. New `decode_dtype(raw, ctx) -> Result` helper routes all real-dtype decode sites. Every site confirmed fail-closed (initializer value, value-info TensorType/SparseTensorType elem, convert_tensor Constant/attribute inline tensors, convert_type_proto Tensor/SparseTensor elem + Map key). No surviving `unwrap_or(Float32)` on any real-dtype site. Intentional non-dtype defaults preserved (untyped value-info, non-tensor containers). `type_proto_to_dtype_shape`/`convert_type_proto`/`value_info_type` signature changes `-> Result<…>` with `?` propagation; transactional-on-failure preserved. Proto bump FLOAT4E2M1=23 unique, correct. Full debug+release ORT2 suite green (ir 40, loader 15+2 new tests, ep-cpu 101, optimizer 27, session 7+1+11, shape-inf 14+3+52, capi 4+9, ep-api 13+4). bert_toy conformance PASS max_abs 1.192e-7.

**Minor advisory (non-blocking):** present-but-UNDEFINED (elem_type=0) on value-info now rejected (correct fail-close for typed I/O); canonical "untyped" models omit the type field and still load.

---

### 2026-07-14: Optimizer fused ops emitted in `com.microsoft` contrib domain; ep-cpu dispatch keyed by (domain, op_type)
**By:** Batty (batty-12, opus)
**Branch:** `squad/ort2-fused-domain` (based on main `06a2423`) — merged to main `8cab9d2`

**What:** The optimizer fusion pass previously emitted fused ops in the reserved default ONNX domain (`""`/`ai.onnx`). Moved all optimizer-produced fused ops to `CONTRIB_DOMAIN = "com.microsoft"` and generalized ep-cpu kernel dispatch to key on `(domain, op_type)` via a new `OpRegistry::supports(op_type, domain)` method.

**Domain chosen — `com.microsoft`:** established ONNX-ecosystem contrib domain where FusedMatMul/LayerNormalization/SkipLayerNormalization/SimplifiedLayerNormalization contrib variants already live. Shape-inference crate already registered handlers there. Interoperable with ORT-exported models.

**Op map:** `LayerNormalization` — emitted by optimizer + runnable kernel; default-domain kernel/shape rule KEPT; `com.microsoft` bindings ADDED (additive). `FusedMatMulBias`/`FusedGemm` — no kernel exists in either domain; left without kernel (correct: kernel-less ops are rejected at placement).

**Files touched:** `onnx-runtime-optimizer/src/fusion.rs` (CONTRIB_DOMAIN const, domain set on fused nodes); `onnx-runtime-ep-api/src/registry.rs` (supports() + norm_domain in both lookup+supports); `onnx-runtime-ep-cpu/src/kernels/mod.rs` (com.microsoft LayerNorm registration); `onnx-runtime-ep-cpu/src/provider.rs` (gate via registry.supports); `onnx-runtime-shape-inference/src/handlers/norm.rs` (com.microsoft LayerNorm rule).

**Verify:** debug+release green for optimizer(27)/ep-cpu(102)/ep-api(17)/shape-inference(70)/session(19). bert_toy conformance max_abs 1.192e-7. clippy clean. `#![forbid(unsafe_code)]` intact.

**Why:** Non-standard fused ops in `ai.onnx` cause opset-validation collision and ambiguous dispatch. A private contrib domain provides unambiguous dispatch keying, and centralizing support decisions on the registry is model-agnostic and future-proof.

---

### 2026-07-14: Fused-op contrib domain — dispatch/registry soundness review 🟢 (Gaff, gaff-7)
**By:** Gaff (ORT2 registry/dispatch/API soundness) — reviewed `1e894de`
**Verdict:** 🟢 GREEN — dispatch set correct, normalization symmetric, no phantom kernel registration.

**What:** Provider gate now accepts exactly the set of registered `(op_type, domain)` pairs via `registry.supports`. Enumerated registry: default-domain registrations == PHASE1_OPS (1:1 verified); `len() == PHASE1_OPS.len() + 2` invariant holds (Softmax-v13 + com.microsoft/LayerNorm, no extras). `ai.onnx`→`""` normalization applied at top of both `lookup` and `supports` — symmetric. Contrib opset: no import → `effective_opset` falls back to `u64::MAX`; `lookup` filters `since_version <= MAX`, picks v1 — no panic. Dual-domain LayerNorm: distinct `OpKey`s (domain differs), distinct HashMap entries, no overwrite; additive only. FusedMatMulBias/FusedGemm have no kernel in either domain; `supports()` returns false — rejected at placement, not execution. `is_phase1_op` kept as `pub` API (harmless). Debug+release all green; bert_toy PASS max_abs 1.192e-7; clippy clean.

---

### 2026-07-14: ORT2 session `optimize` stage activated (opt-in) (Roy, roy-15)
**By:** Roy (ORT2 architect — session pipeline / loader=shape-inference / session=execute seam)
**Branch:** `squad/ort2-session-optimize` (based on `6f2e518`) — merged to main `5a2d527`

**What:** Wired `onnx-runtime-optimizer` into `onnx-runtime-session`'s `build()` pipeline as an explicit opt-in stage. Default behavior (`"optimization"="none"`) is byte-identical to before this change.

**Option surface:** Key `"optimization"` via `SessionBuilder.option(key, value)`.
- `"none"`/`"off"`/`"0"` → `OptimizationLevel::None` (**DEFAULT** — no-op, no optimizer call, no re-inference)
- `"basic"` → ConstantFolding + DeadNodeElimination (structure-preserving, no new op types)
- `"all"` → ConstantFolding + DeadNodeElimination + OpFusion (`optimizer::default_passes()`)
- Unknown keys → `SessionError::UnknownOption`; unknown values → `SessionError::InvalidOption`.

**Pipeline ordering:**
```
load (+ loader shape inference)
  → optimize_graph(level)          [skipped entirely when level == None]
  → add com.microsoft to opset_imports
  → re-run infer_graph(Permissive) [only when passes ran]
  → compile → allocate
```

**Conformance:**
- DEFAULT (opt-off): `bert_toy_conformance` unchanged — max_abs **1.192e-7**. Byte-identical.
- `basic` vs opt-off: max_abs **0.000e0** — byte-identical. Const-fold + DCE + re-inference inert.
- `basic` vs onnxruntime reference: max_abs **1.192e-7** (same as opt-off).

**Documented discrepancy — `all` path not yet executable:**
`OpFusion` is schema-unaware: `FusedMatMulBias`/`FusedGemm` have no CPU kernel; fused `com.microsoft::LayerNormalization` carries 5-input signature incompatible with CPU kernel's 2-3 input arity. Fails cleanly with `SessionError::UnsupportedOp { op_type: "FusedMatMulBias" }` before any numerics. Optimization stays opt-in / default-off. Tripwire test `full_optimization_fusion_path_is_not_yet_executable` asserts the failure and fires loudly when fusion becomes executable. **Follow-up to Batty:** schema-aware `OpFusion` + `FusedMatMulBias`/`FusedGemm` CPU kernels (or gate those patterns).

**Files touched:** `crates/onnx-runtime-session/Cargo.toml` (deps); `crates/onnx-runtime-session/src/lib.rs` (`OptimizationLevel` enum+parse, `SessionError` variants, `optimize_graph()`, `build()` rewrite, unit tests); `crates/onnx-runtime-optimizer/src/lib.rs` (re-export `CONTRIB_DOMAIN`); `crates/onnx-runtime-session/tests/bert_toy_optimized_parity.rs` (new); `docs/PROGRESS.md`.

**Validation:** 53 tests green debug+release (optimizer 26+1, session 12+1+2+11). clippy clean. `#![forbid(unsafe_code)]` intact.

---

### 2026-07-14: ORT2 session `optimize` stage — correctness/conformance review 🟢 (Chew, chew-19)
**By:** Chew (ORT2 correctness/conformance) — reviewed `c92a2f2`
**Verdict:** 🟢 APPROVE

**Scope:** `git diff 6f2e518...c92a2f2` — 7 files, +435/-12.

**Findings:**
1. **Default-off invariance** 🟢 — `optimize_graph()` returns `Ok(())` immediately when `level.passes()` is empty. No passes run, no `com.microsoft` opset import inserted, `infer_graph` NOT re-run. No unconditional second infer_graph. `bert_toy_conformance` unchanged: max_abs 1.192e-7 (debug+release).
2. **`basic` parity is real** 🟢 — `basic` vs opt-off: max_abs 0.000e0 (byte-identical). `basic` vs onnxruntime: max_abs 1.192e-7. Output shapes correct ([1,8,99], [1,2]). No rounding drift.
3. **Re-inference ordering sound** 🟢 — passes → opset import → re-infer on rewritten graph → `from_parts` consumes re-inferred graph. Compile/allocate see post-optimize shapes.
4. **`all`-path gating clean** 🟢 — fails with `UnsupportedOp { op_type: "FusedMatMulBias" }` BEFORE numerics. Tripwire non-tautological: `Ok(_) => panic!` fires the moment fusion becomes executable; `Err(UnsupportedOp{op_type})` asserts `op_type ∈ {FusedMatMulBias, FusedGemm}`. No tolerance widened.
5. **Suite/clippy/unsafe** 🟢 — full ORT2 suite green debug+release (all crates). clippy clean. No new `unsafe`.

**Non-blocking note:** `Err(other) => {}` arm in tripwire accepts any non-`UnsupportedOp` error without asserting fusion-relatedness. Does not mask silent wrong numerics (the `Ok` arm guards correctness). Suggest future tightening.

---

### 2026-07-14: `optimization="all"` fusion path made executable + parity-correct on bert_toy (batty-13)
**By:** Batty — `squad/ort2-fusion-executable` (base main `5a2d527`); merged as `e9bf155`

**What:** Turned the previously-deferred `"all"` optimization path from "not executable" into a byte-identical, parity-validated path on `bert_toy`. Three coordinated changes: schema-aware LayerNorm fusion, a `FusedMatMulBias` CPU kernel (+ shape rule), and flipping the tripwire test into a real parity assertion.

**Schema-aware LayerNorm fusion** (`onnx-runtime-optimizer/src/fusion.rs`): Added `RewriteKind {Structural, LayerNorm}` and `FusionPattern::layernorm()`. Emits `com.microsoft::LayerNormalization` with inputs `[X, Scale, B]` and attributes `axis` (from first ReduceMean axes attr) + `epsilon` (read as f32 from inline initializer via `read_scalar_f32`; falls back to 1e-5 if unreadable). `X = Sub operand ≠ rm1 output`; `Scale = Mul operand ≠ Div output`; `B = final-Add operand ≠ Mul output`. Order-independent disambiguation.

**`FusedMatMulBias` CPU kernel** (`ep-cpu/src/kernels/fused_matmul_bias.rs`): `MatMul(A,B) + bias` (broadcast add), reusing new shared `matmul::matmul_dense` + `add::broadcast_apply`. Registered `("FusedMatMulBias","com.microsoft",1)`.

**Shape rule** (`shape-inference/src/handlers/linalg.rs`): `fused_matmul_bias` output = MatMul(A,B) shape; registered under `com.microsoft`.

**Tripwire → real parity**: `full_optimization_fusion_path_is_not_yet_executable` → `full_optimization_fusion_path_matches_reference_and_default`; asserts `"all"` runs and matches opt-off and reference at existing tolerance (2e-3 atol/rtol — not loosened).

**Parity:** `opt=all` vs opt-off **0.0 (byte-identical)**; vs reference **1.192e-7**. Full suite green debug+release (optimizer 26, ep-cpu 105, shape-inference 70, session 26). Clippy clean. No new unsafe.

**Deferred:** `FusedGemm` (MatMul+Add+Relu) — no Relu-terminated fusion in bert_toy; remains graph-only with no kernel.

---

### 2026-07-14: Review — schema-aware LayerNorm fusion correctness (chew-20)
**By:** Chew (ORT2 correctness) — reviewed `squad/ort2-fusion-executable` @ `0f4811e`
**Verdict:** 🟡 approve with follow-ups (non-blocking)

**Verified correct:**
1. Operand disambiguation is order-independent/model-agnostic (X/Scale/B selected by excluding interior tensors, not by position). Not baked to bert_toy. ✅
2. Epsilon extraction robust for realistic f32 cases (`ConstantFolding` materializes `Constant` nodes to initializers before `OpFusion`). ✅
3. `opt=all` parity real: 1.192e-7 vs reference, 0.0 vs opt-off. Tripwire is a real assertion; no tolerance loosened. ✅
4. `fuses_layernorm_chain` unit test asserts `[X,Scale,B]` inputs + `axis=-1` + `epsilon≈1e-12` (values, not arity). ✅

**Follow-ups (non-blocking):**
- **F1** — axis silently defaults to `-1` when `axes` attribute is absent (opset-18 uses `axes` as input, not attribute). Non-last-axis LayerNorm at opset-18 would be silently wrong. Fix: read `axes` input initializer for opset-18 and validate contiguous-to-end; otherwise decline to fuse.
- **F2** — epsilon silently defaults to `1e-5` if eps operand is not a readable f32 constant (e.g. fp16/fp64 model). Fix: decline to fuse instead of guessing.
- **F3** — no positive data-flow guard (op-type sequence match without verifying interior data-flow). Pre-existing. Also `layernorm_node` returning `Err` hard-errors the whole pass via `?` rather than declining that one match.
- **F4** — nit: `vs_off` byte-identity is observed (0.0) but not asserted. Consider `assert_eq!(overall_vs_off, 0.0)`.

**Ownership:** F1/F2 first. Batty locked out (author); Roy/Deckard/Leon eligible.

---

### 2026-07-14: Review — FusedMatMulBias kernel, shape rule, registry, operand-order (gaff-8)
**By:** Gaff (ORT2 kernel/registry/dispatch) — reviewed `squad/ort2-fusion-executable` @ `0f4811e`
**Verdict:** 🟡 approve with required follow-up (non-blocking for bert_toy)

**Verified correct:**
1. FusedMatMulBias kernel numerics: `matmul_dense(A,B)` then `broadcast_apply(bias)` — full numpy batched/broadcast semantics. ✅
2. Standalone MatMul refactor (`matmul_dense` extraction) byte-for-byte identical to old body; no regression (0.0 vs opt-off). ✅
3. Shape rule consistent: delegates to `matmul_shape`, registered `("com.microsoft","FusedMatMulBias",1)`. ✅
4. Registry/dispatch correct: `OpKey::new("FusedMatMulBias","com.microsoft",1)` registered; `supports()` true; `FusedGemm` intentionally not registered. Domain/op/key consistent across fusion↔kernel↔shape rule. ✅
5. MatMul+Add operand-order generality ROBUST: first-seen ordering over `[MatMul, Add]` chain always yields `[A, B, bias]` regardless of whether Add is `Add(mm,bias)` or `Add(bias,mm)`. Not baked to bert_toy. ✅

**Gap (required follow-up):**
- **G1** — MatMul+Add fusion has no shape guard. An `Add` whose non-matmul operand expands the matmul output shape (more leading dims) fuses to a silent-wrong result: shape rule returns the matmul shape; `broadcast_apply` silently truncates the leading axis. Not exercised by bert_toy (standard bias-add / same-shape cases). Fix: narrow/guard `MatMul+Add → FusedMatMulBias` to decline when the Add's non-matmul operand would expand the matmul output shape. Fix owner: **Roy** or **Deckard** (Batty locked out).

**Suite:** ep-cpu 105, optimizer 26, shape-inference 70, session 26 + bert_toy conformance + opt-parity — 0 failed. Clippy clean. No new unsafe.

---

### 2026-07-14: ORT2 fusion decline-to-fuse guards (harden `optimization="all"`)

**By:** Roy (ORT2 optimizer/loader)
**Branch:** `squad/ort2-fusion-guards` → merged main `8f222bd`

**What:** Hardened both fusions in `crates/onnx-runtime-optimizer/src/fusion.rs` to
**decline-to-fuse** (leave the original ops) whenever their structural/shape
assumptions can't be proven. Addresses Chew F1/F2/F3/F4 and Gaff Finding 5.

- **LayerNorm** — axis: single concrete from `axes` attr only (axes-as-input/multi-axis/absent → decline); epsilon: concrete f32 scalar constant only (no silent 1e-5); positive structural guard confirming interior data-flow; declining returns `None` via `layernorm_spec` (Option) → fixpoint loop skips (no `?`-propagated hard error).
- **MatMul+Add → FusedMatMulBias** — new `matmul_bias_broadcast_ok` guard: only fuse when bias is a valid trailing broadcast of the matmul output; expanding/unknown shapes → decline.
- **Parity nit (F4):** `assert_eq!(overall_vs_off, 0.0)` for both `"all"` and `"basic"` vs opt-off.
- `bert_toy` still fuses 32× FusedMatMulBias; `"all"` vs opt-off **0.0**, vs reference **1.192e-7**. New 5 decline/positive unit tests.

**Review:** Deckard (deckard-12) — 🟢 APPROVE. Guards correct in both directions (32× FusedMatMulBias preserved, edge cases decline, tests non-tautological, suite green debug+release).

**Advisory A1 (pre-existing, non-blocking):** `bert_toy` LayerNorm never fused e2e — 0 of 12 LN regions fuse due to pre-existing escape rule blocking the 10-op split-diff DAG variant. Addressed by Arc 2 below.

---

### 2026-07-14: ORT2 LayerNorm fusion now fires end-to-end on bert_toy (DAG-aware matcher)

**By:** Batty (ORT2 optimizer/fusion engineer)
**Branch:** `squad/ort2-layernorm-e2e` → merged main `1817890`
**Closes:** Deckard advisory A1 from deckard-12 review

**Root cause diagnosed:** `bert_toy`'s LayerNorm is a **10-op split-diff variant** — the exporter emits two distinct `Sub(x, mean)` nodes (one for variance branch `#50`, one for numerator branch `#54`) instead of CSE-reusing a single diff. The shared `mean` node has two Sub consumers, causing the escape rule (`fusion.rs:190-206`) to reject the match; the structural guard also fails because `Div` reads a different Sub's output.

**Fix:** New **DAG-aware matcher** `FusionPattern::try_match_layernorm` anchored on the `mean` ReduceMean, collecting all `Sub(x, mean)` consumers and following both variance (`Sub→Pow→ReduceMean→Add(eps)→Sqrt`) and numerator (`Sub→Div→Mul→Add`) branches. Accepts both canonical **9-op** (shared Sub) and **10-op** (split Sub) shapes. `layernorm_spec` generalized to 9-or-10 nodes with a "same X" guard. Linear matcher `try_match_from` retained only for MatMul+Add (unchanged). All prior decline guards preserved verbatim.

**Parity updated honestly:** `"all"` vs-opt-off `assert_eq!(…,0.0)` replaced with tight `< atol` drift bound (fused LN kernel reduces in one pass → few-ULP delta). `"basic"` keeps exact `assert_eq!(overall_vs_off, 0.0)`. New e2e test `full_optimization_actually_fuses_layernorm_and_matmul_bias` loads real bert_toy and asserts 12× LayerNormalization / 0 surviving ReduceMean / 32× FusedMatMulBias.

**Parity numbers:** `"all"` vs reference **1.043e-7**, vs opt-off **1.416e-7**; `"basic"` vs reference **1.192e-7**, vs opt-off **0.0**. Tolerance (atol/rtol 2e-3) not loosened.

**Review — Chew (chew-21):** 🟢 APPROVE. DAG matcher correct; over-match declines verified via adversarial probes (`different_x` → declines; `reversed_sub` → fuses, confirming A-CHEW-1 is PRE-EXISTING); 31 optimizer tests + 3 session tests green.
- **A-CHEW-1 (pre-existing, non-blocking):** Sub operand order not checked — `Sub(mean, x)` over-matches with sign-flip. Reproduced identically on base 9-op matcher. Recommend follow-up (owner Roy/Deckard/Leon; Batty locked out).

**Review — Deckard (deckard-13):** 🟢 APPROVE. A1 genuinely closed (real loaded-model e2e test). Parity honest/load-bearing. All numbers reproduced exactly (debug + release). No regression (FusedMatMulBias 32×, all prior decline tests pass).
- **A2 (non-blocking):** 10-op split-diff shape has no isolated synthetic optimizer unit test.
- **A3 (non-blocking):** vs-opt-off drift ceiling 2e-3 vs actual 1.4e-7 (~4 orders of margin); consider tightening to 1e-5.

---

### 2026-07-14: ORT2 LayerNorm centering operand-ORDER guard + split-shape unit test + tightened drift ceiling

**By:** Leon (ORT2 optimizer engineer)
**Branch:** `squad/ort2-layernorm-order-guard` → merged main `a02d46e`
**Closes:** A-CHEW-1 (Sub operand-order sign-flip over-match), A2 (isolated 10-op unit test), A3 (drift ceiling tighten) from chew-21 + deckard-13 reviews of batty-14.

**Problem.** `layernorm_spec` validated centering `Sub` nodes by operand *membership* but not *order*. A reversed `Sub(mean, x)` satisfies membership and was silently rewritten to a `LayerNormalization` that computes `+(x − mean)/std` — a **sign-flipped** result. Chew confirmed this fired on both the 10-op split path and the base 9-op matcher.

**Fix — operand-order guard (A-CHEW-1):** After the existing "same X" membership guard, added:
```rust
let subtracts_x_minus_mean = |sub: &Node| -> bool {
    matches!(sub.inputs.as_slice(), [Some(a), Some(b)] if *a == x && *b == mean)
};
if !subtracts_x_minus_mean(sub_pow) || !subtracts_x_minus_mean(sub_div) {
    return None;
}
```
Requires `Sub` input[0] == X and input[1] == mean; exactly-binary arity enforced. Reversed or ambiguous → decline (no rewrite). Tightens both 9-op and 10-op paths (shared-Sub 9-op: `sub_div == sub_pow`, checked once).

**Fix — isolated 10-op unit test (A2):** Synthetic `layernorm_split_graph(bool)` helper (two distinct `Sub(x, mean)` nodes) + test `fuses_layernorm_split_chain` asserting exactly one `com.microsoft::LayerNormalization`, `[X, Scale, B]`, `axis = -1`, folded epsilon.

**Fix — adversarial decline test (A-CHEW-1):** `declines_layernorm_when_numerator_sub_reversed` — 10-op graph with numerator `Sub(mean, x)`; asserts no fusion, all 10 ops retained.

**Fix — tighten drift ceiling (A3):** Introduced `const DRIFT_ATOL: f32 = 1e-5` scoped to all-vs-opt-off assertion only; vs-reference conformance tolerance (2e-3 atol/rtol) unchanged.

**Verification:** 33 optimizer tests (+2) green debug + release; clippy clean; bert_toy still fuses 12× LayerNormalization + 32× FusedMatMulBias; parity all/ref 1.043e-7, all/off 1.416e-7 (< 1e-5 ceiling).

**Review — Gaff (gaff-10):** 🟢 APPROVE. Guard structural/model-agnostic (`fusion.rs:625-635`); non-tautological positive and adversarial coverage (`fusion.rs:1055-1121`); drift and reference bounds remain separate; 31→33 optimizer tests; debug + release green; clippy `-D warnings` clean; `#![forbid(unsafe_code)]` intact.

---

### 2026-07-14: EPContext §55 loader LOAD path — `EpContextNode` view, blob resolution, path-safety

**By:** Roy (ORT2 loader)
**Branch:** `squad/ort2-epcontext-loader` → merged main `d18a8a3` (part 1)
**Scope:** `crates/onnx-runtime-loader` — §55.3 load path only. Runtime `EpContext` type + `EpContextRegistry` are Deckard's (ep-api, below).

**New module `epcontext.rs`:**

1. **`EpContextNode<'g>`** — typed view over IR `Node`; recognizes `op_type == "EPContext"` && `domain == "com.microsoft"`. Fields: `node`, `source` (§55.6 dispatch key, `Option<&str>`), `main_context` (default `true` when absent), `embed_mode` (`EmbedMode`), `sdk_version`, `partition_name`. Variadic i/o read directly (`inputs()`/`outputs()`), no arity assumed.

2. **Enums:** `EmbedMode { Embedded, ExternalFile }` (default `Embedded`; `0`→External, fail-closed); `EpContextBlob { Embedded(Vec<u8>), External { path, map: Mmap } }` with uniform `bytes()` accessor.

3. **Recognition helpers:** `ep_context_nodes`, `ep_context_node_ids`, `is_ep_context_op` — free functions, IR crate untouched.

4. **`resolve_ep_context(model_dir, node) -> Result<EpContextBlob>`:**
   - `embed_mode=1`: `Embedded(bytes.to_vec())` from `ep_cache_context`.
   - `embed_mode=0`: UTF-8 relative path + traversal guard + `Mmap::map` read-only → `External { path, map }`. Blob bytes opaque — loader never interprets them.

5. **Lossless opaque blob:** `graph_builder` special-cases `ep_cache_context` on EPContext nodes, storing raw bytes as `UINT8 Attribute::Tensor` instead of `String::from_utf8_lossy` (which would corrupt binary vendor blobs). Verified by round-trip test with `0x00/0x80/0xFE/0xFF/0xC3 0x28` payload.

6. **Path-safety:** `resolve_external_path` rejects `is_absolute()`, `Component::ParentDir` (`..`), `Component::RootDir | Prefix` before `join` — lexically contained. Tests: `../evil.bin` and `/etc/passwd` both rejected. Existing `weights.rs` §19.2 guard gap noted as follow-up.

7. **Shape inference:** EPContext unregistered → `InferenceRegistry` leaves unresolved without error; declared output shapes preserved verbatim through `infer_graph`. Asserted by test.

8. **New `LoaderError` variants:** `EpContext(String)`, `EpContextPath { path, reason }`.

9. **Tests:** 7 tests (embedded non-UTF8 blob, external mmap, attribute defaults, explicit attrs + variadic i/o, `../evil.bin` reject, `/etc/passwd` reject, output shape preserved). All green debug + release. Existing 15 loader tests + bert_toy conformance unaffected.

**Review — Gaff (gaff-10):** 🟢 APPROVE. Opaque blob preservation byte-for-exact-byte verified (scope-gated to `is_ep_context_op && attr == "ep_cache_context"`, no regression to other attrs/nodes). Path-safety rejects before join (not after canonicalize) — strictly more protective than weights.rs. mmap unsafe follows weights.rs idiom, no new unsafe. 7/7 epcontext tests + 15/15 loader suite + doctests green; clippy `-D warnings` clean.

---

### 2026-07-14: EPContext §55 ep-api contract — runtime `EpContext`, source-keyed registry, trait methods

**By:** Deckard (ORT2 ep-api)
**Branch:** `squad/ort2-epcontext-epapi` → merged main `d18a8a3` (part 2)
**Scope:** `crates/onnx-runtime-ep-api` — §55.1 / §55.3 dispatch / §55.6 / §55.7. Loader-side is Roy's (above).

**New module `epcontext.rs`:**

1. **`EpContext`** (in-memory §4/§55.1 form):
   ```rust
   pub struct EpContext {
       pub ep_name: String,
       pub ep_version: String,        // maps to ep_sdk_version attr
       pub data: Vec<u8>,             // opaque blob; maps to ep_cache_context
       pub covered_nodes: Vec<NodeId>,
       pub device_fingerprint: String,
   }
   ```
   Derives `Clone, Debug, Default, PartialEq, Eq`; ctor `EpContext::new(..)`.

2. **`EpContextRegistry`** (§55.6): `register(ep, source_keys)` / `claim(source: Option<&str>) -> Option<EpId>`. **Reject-duplicate policy:** second EP on existing key → `EpError::DuplicateContextSource`; same `(key, ep)` re-declare is idempotent. Rationale: two EPs on one source is a config error; last-writer-wins creates order-dependent non-determinism.

3. **Trait methods on `ExecutionProvider`** — all have safe defaults (existing EPs compile unchanged):
   - `fn context_source_keys(&self) -> Vec<String> { Vec::new() }`
   - `fn save_context(&self) -> Result<EpContext>` → default `Err(UnsupportedContext)`
   - `fn load_context(&self, ctx: &EpContext) -> Result<()>` → default `Err(UnsupportedContext)`

4. **`build_ep_context_registry(eps)`** — pure builder; iterates EPs, reads `context_source_keys()`, skips empty-key EPs, propagates `DuplicateContextSource`.

5. **New `EpError` variants:** `NoEpForContext { source_key: Option<String> }`, `UnsupportedContext { ep: String }`, `DuplicateContextSource { source_key, existing, new }`.

6. **⚠️ Naming note for session integrator:** error field is `source_key` (not `source`) — `thiserror` 2.0 auto-treats a field literally named `source` as the `std::error::Error` cause, which `Option<String>` cannot satisfy. Session code: `EpError::NoEpForContext { source_key: node.source.map(str::to_owned) }`.

7. **Shared-checkout race (post-merge note):** deckard-14's commit was recovered from a dangling object after a force-push; parallel commit-producing agents need separate worktrees to avoid this. Lesson logged.

**Verification:** 22 unit + 4 lib ep-api tests green debug + release; ep-cpu 3+11 and session 11 tests unchanged; clippy `-D warnings` clean; no new unsafe.

**Review — Chew (chew-22):** 🟢 APPROVE. Model-agnostic dispatch confirmed (zero hardcoded vendor names in non-test code; `claim` is pure lookup). Reject-duplicate semantics correct and documented. Trait defaults preserve object-safety and don't break existing EPs (ep-cpu 105 + session 12+1+3+11 tests green). `EpContext` struct matches §55.1 field-for-field; save→load round-trip verified. `source_key` naming correct for thiserror 2.0.18. No 🔴 blockers.

---

### 2026-07-14: EPContext session CONSUME path — bypass-placement dispatch + main_context resolution/dedup

**By:** Batty (ORT2 session)
**Branch:** `squad/ort2-epcontext-session` → merged main `46f2861`
**Scope:** `crates/onnx-runtime-session` only (§55.3 dispatch/execution + `main_context`; the session row of §55.7). Loader (`EpContextNode`/`EpContextBlob`/`resolve_ep_context`) is Roy's; ep-api (`EpContext`/`EpContextRegistry`/trait methods) is Deckard's — both used via public APIs, unmodified. The `*_ctx.onnx` writer/dump path (§55.4) is a separate follow-up and is NOT built here.

**New module `session/src/epcontext.rs`**, re-exported from `lib.rs`:

1. **Public entry point:** `load_ep_context_nodes(graph, model_dir, eps) -> Result<EpContextPlacement>` where `EpContextPlacement { handled: Vec<NodeId> }` lists nodes that bypassed placement.

2. **Dispatch flow (§55.3, model-agnostic — pure `source`-key lookup, §55.6):**
   - Enumerate `ep_context_nodes(&graph)` (Roy). Empty → no-op early return.
   - Build `EpContextRegistry` via `build_ep_context_registry(eps)` (Deckard) — propagates `EpError::DuplicateContextSource`.
   - Phase 1 (main_context=true): `registry.claim(node.source)` → `EpError::NoEpForContext { source_key }` if unclaimed (real key, never guessed). `resolve_ep_context` → `ep.load_context(ctx)`.
   - **Payload dedup:** `HashSet<(Option<source>, Vec<u8>)>` gates `load_context` — identical packed binaries load exactly once.
   - Phase 2 (main_context=false): resolve by `(source, partition_name)` against loaded primaries — no second blob load. Missing primary → `SessionError::DanglingEpContext`.

3. **Executor bypass:** `executor.rs` skips EPContext nodes via `is_ep_context_op(op_type, domain)` — never reaches CPU EP kernel dispatch.

4. **Model-dir threading:** `SessionBuilder::build` retains model directory and threads it into `load_ep_context_nodes` so `embed_mode=0` external blob paths resolve relative to model file (per §19.2).

5. **New `SessionError` variant:** `DanglingEpContext { source_key, partition_name }`. Field named `source_key` (not `source`) per thiserror 2.0 constraint.

6. **Error taxonomy:** `EpError::DuplicateContextSource` (config), `EpError::NoEpForContext { source_key }` (unloaded EP), `SessionError::DanglingEpContext { source_key, partition_name }` (bad reference).

7. **Tests:** 7 new tests in `tests/epcontext.rs` (MockCompiledEp): embed round-trip, external .bin round-trip, unclaimed QNN → NoEpForContext, main_context dedup + reference resolution, dangling reference, duplicate-source rejected, session-level unclaimed-node rejection. All green debug + release. Clippy -D warnings clean.

**Review — Deckard (deckard-15):** 🟡 YELLOW — approve with advisories. Phase-1-before-Phase-2 ordering enforced on materialized Vec (graph order irrelevant). DuplicateContextSource and NoEpForContext propagate with real (never guessed) source key. Dedup keyed on (source, bytes) — different sources/binaries never collapsed, shared packed binary loads exactly once. main_context=0 references resolve by (source, partition_name) with DanglingEpContext; no second blob load. Path-traversal guard on consume path, tested. Four non-blocking advisories: (A1) covered_nodes omits deduped sibling primary NodeId; (A2) duplicate (source,partition_name) primaries silently accepted; (A3) returned EpContextPlacement discarded by session (executor self-contained); (A4) add session-level path-traversal test. No blocking defect.

**Review — Chew (chew-23):** 🟡 YELLOW — approve with test advisories. Model-agnostic dispatch confirmed (zero hardcoded vendor names in non-test code; QNN literal only in unclaimed fixture). No CPU fall-through: all session construction paths converge on `from_parts` which calls `load_ep_context_nodes` before `Executor::build`; EPContext nodes skipped by `is_ep_context_op` predicate. 7/7 session epcontext tests + clippy pass. Non-blocking: (1) add positive executor-bypass regression test with claimed mock EP; (2) assert full EpContext fields (ep_name, ep_version, covered_nodes, fingerprint), not only ctx.data.

---

### 2026-07-14: ONNX encoder (IR → ModelProto → bytes) — ORT2 §55.3/§55.4 foundation

**Author:** Roy (roy-18, opus-4.8)
**Artifact:** `crates/onnx-runtime-loader/src/encoder.rs` (+518 lines), `crates/onnx-runtime-loader/tests/encoder.rs` (+488 lines)
**Branch:** `squad/ort2-onnx-encoder` (base `46f2861`)
**Final merge commit:** `de7ccce` (merged as `55c7608` after Deckard's v2 revision)
**Cycle:** v1 authored by Roy → 🟢 Gaff → 🔴 Leon (BLOCK) → v2 by Deckard → 🟢 Leon re-review

#### What landed (v2 — Deckard's revision)

New module `crates/onnx-runtime-loader/src/encoder.rs` — the model-agnostic inverse of `graph_builder` + `weights`. Pure, safe `prost` encoding (no new `unsafe`).

**Public API:**
- `ModelMetadata` — model-level metadata (ir_version default 10, producer fields, graph name, metadata_props).
- `Model<'a>` — holds `&Graph`, `ModelMetadata`, and optional `&WeightStore`.
- `encode_model(&Model) -> Result<Vec<u8>, LoaderError>` — serialize to bytes.
- `write_model(&Model, path) -> Result<(), LoaderError>` — serialize to file.
- `encode_model_proto(&Model) -> Result<ModelProto, LoaderError>` — returns ModelProto before serialization (integration seam for §55.4 EPContext writer).
- `DEFAULT_IR_VERSION: i64 = 10`.

**IR change (v2):** `Attribute::String(String)` → `Attribute::String(Vec<u8>)` and `Attribute::Strings(Vec<String>)` → `Attribute::Strings(Vec<Vec<u8>>)` for byte-preserving STRING attributes. `as_str()` now returns `Option<&str>` (checked UTF-8); `as_bytes()` added for raw access.

**Byte-exact round-trip:** nodes (op_type, domain, I/O order incl. skipped optionals, doc_string, all attributes), initializers (all dtypes, raw little-endian), model metadata (opsets, ir_version, producer fields, metadata_props), symbolic/static dims. Output deterministic (opsets sorted by domain, initializers by value id, attributes by name, nodes by id).

**Fields not encoded:** per-node name (IR has none), TrainingInfoProto, FunctionProto, sparse initializers, quantization annotations, subgraph formal I/O (guard: `encode_attribute` errors on Graph/Graphs attributes rather than emitting a truncated subgraph).

#### v1 blocking violation (Leon/leon-12)

Roy's v1 encoder violated §55.6 model-agnostic hard rule: `encode_node` called `is_ep_context_op` and `encode_attribute` branched on literal `"ep_cache_context"` with UINT8 tensor. The generic attribute layer must contain no op/attribute-name literals. Revision assigned to Deckard (Roy locked out of the encoder artifact for this cycle).

#### Deckard's v2 fix (Option A — byte-preserving STRING in IR)

Root cause: decode path used `String::from_utf8_lossy` for ONNX STRING attributes, forcing Roy to special-case the binary EPContext blob as a UINT8 tensor in decode and then reverse it in encode. Deckard eliminated both special cases by making `Attribute::String` hold `Vec<u8>` throughout the IR. The generic encode/decode layers copy STRING bytes verbatim with no conditional dispatch on op or attribute name. The EPContext consumer (`epcontext.rs`) reads the blob from `Attribute::String` raw bytes directly.

#### Non-blocking advisories (Gaff/gaff-11)

- **A1** — subgraph formal I/O silently omitted (v2 now errors on Graph/Graphs attributes instead).
- **A2** — model metadata silently defaulted when `.with_metadata()` not called (now documented explicitly).
- **A3** — STRING byte-exact doc nuance (first decode from original file is lossy for non-routed strings; ep_cache_context is exempt via generic raw bytes in v2).
- **A4** — external re-inlining bloats output for large models (values intact; `ExternalDataPolicy` recommended as follow-up).

#### Follow-up flags

1. External-data preservation on write (`ExternalDataPolicy` in `Model`/`EncodeOptions`).
2. `embed_mode=0` external blob file naming/writing (writer policy; encoder receives path string directly).
3. Subgraph I/O round-trip (needed before any Loop/If/Scan model is encoded).
4. `encode_model_proto` is the §55.4 writer integration seam (load → partition/compile → build EPContext NodeProto → splice → serialize).

#### Verification (v2)

```
cargo test -p onnx-runtime-loader      # 9 encoder + 15 loader + 7 epcontext — green
cargo test -p onnx-runtime-session     # 34 tests — green (incl. 7 EPContext consume/dedup)
cargo test -p onnx-runtime-ir          # 40 tests — green
cargo clippy -p onnx-runtime-loader --all-targets -- -D warnings  # clean
```

---

### 2026-07-14: Map SessionError::DanglingEpContext → OrtErrorCode::InvalidGraph in CAPI

**Author:** Chew (chew-24, gpt-5.6-sol)
**Artifact:** `onnx-runtime-capi` error mapping
**Commit:** `d3f0c0a`
**What:** Added explicit mapping of `SessionError::DanglingEpContext` to `OrtErrorCode::InvalidGraph` in the non-exhaustive match in the CAPI layer. Retains an explicitly exhaustive `SessionError` match to preserve compile-time guard against future unhandled variants.
**Why:** `DanglingEpContext` (a structurally invalid model reference — `main_context=0` EPContext node with no primary to resolve to) aligns with the existing IR/graph/optimizer/shape-inference error classification. Found via a full cross-crate build gate after the EPContext consume-path merge introduced the new variant on main.


---

### 2026-07-14T16:30:00Z: EPContext §55.4 WRITER / DUMP path (loader-owned, session-driven) — v1

**By:** Batty (batty-16)
**Commit:** `206742e`
**What:** Implemented the EPContext dump/write path — the inverse of the consume path. After an EP compiles a partition, produces `<orig_stem>_ctx.onnx` with the compiled subgraph replaced by a single `com.microsoft::EPContext` node carrying the vendor blob.
**Why:** §55.4 requirement. Enables EP compilation caching and model portability.

#### API

```rust
pub struct EpContextDumpConfig { pub enable: bool; pub file_path: Option<PathBuf>; pub embed_mode: u8 }
pub struct EpContextPartition<'a> { source, ep_sdk_version, partition_name, main_context, blob, covered_nodes }
pub fn dump_ep_context(model, orig_path, partitions, config) -> Result<PathBuf, LoaderError>
pub fn dump_session_ep_context(model, orig_path, partitions, config) -> Result<PathBuf>
```

#### Key design decisions
- **Model-agnostic**: reuses loader's existing EPContext constants (`EP_CONTEXT_OP`, `MS_DOMAIN`); no new op/vendor/model literals in production code.
- **embed_mode=1 (default)**: blob inlined via `Attribute::String(Vec<u8>)` — byte-exact, no UTF-8 decode.
- **embed_mode=0**: sidecar `<ctx_stem>_<source>_<partition>.bin`; source/partition sanitized `[A-Za-z0-9._-]`, non-matching → `_`. Relative filename stored in node.
- **Boundary computation**: inputs/outputs determined deterministically from ascending NodeId; optional slots preserved.
- **Crate layering**: session driver bridges ep-api → loader; loader has no ep-api dependency.
- **Tests**: 3 loader (embed round-trip, external sidecar, two-partition), 3 session (embed byte-exact, external, explicit file_path override).

#### Follow-ups
- SessionBuilder + capi options → `EpContextDumpConfig`
- Real compile/partition wiring (CUDA/QNN)
- Multi-graph `main_context=0` extension

---

### 2026-07-14T16:45:00Z: EPContext §55.4 writer v1 review (Gaff, gaff-12)

**By:** Gaff (gaff-12, reviewer)
**What:** 🟢 GREEN — byte-exact round-trip both modes; sidecar sanitizer resists path traversal. Two non-blocking advisories.
**Why:** Non-author review of batty-16 writer v1 @ `7eb30ff`.

#### Confirmed
- embed_mode=1: `Attribute::String(Vec<u8>)` verbatim; non-UTF-8 blobs byte-exact (test blobs contain `0x80`, `0xFF`, `0xC3`, `0x28`).
- embed_mode=0: `fs::write` verbatim; consume path mmaps back. Round-trip verified.
- `sidecar_filename` sanitizer: every char not `[A-Za-z0-9._-]` → `_`; hostile inputs (`../../etc/passwd`, `..`, NUL, `\`, `/abs`, `....//`) all yield in-directory filenames; no `ParentDir`/`RootDir`/`Prefix` components.
- Node boundary preserved: `X→EPContext→Y` after splice; `ins==["X"]`, `outs==["Y"]`; only EPContext node remains.
- All tests reproduced: `cargo test -p onnx-runtime-loader` (15+3 ok); `cargo test -p onnx-runtime-session` (10 ok).

#### Advisories (non-blocking)
- **A**: Sidecar collision on duplicate sanitized (source, partition_name) — later write silently overwrites. Suggest `_<index>_` disambiguator.
- **B**: Sanitizer test covers only `/`; add regression for `..`, NUL, `\`.

---

### 2026-07-14T16:45:00Z: EPContext §55.4 writer v1 re-review (Deckard, deckard-17)

**By:** Deckard (deckard-17, reviewer)
**What:** 🔴 **BLOCK** — B1: non-injective sidecar names silently overwrite a partition's blob (data loss). Revision owner: **Leon** (Batty locked out of this artifact).
**Why:** Non-author review of batty-16 writer v1.

#### Blocking finding
- **B1**: `sidecar_filename` (`writer.rs:305-320`) uses only `<ctx_stem>_<sanitized source>_<sanitized partition>.bin`. `sanitize_component` is non-injective: `source="Vendor/EP"` and `source="Vendor_EP"` both → `Vendor_EP`; same `partition_name` → identical filename. Second `fs::write` (writer.rs:181-188) truncates first blob; both nodes store same relative path → `resolve_ep_context` returns wrong blob for both. §55.4 byte-exact round-trip violated for legal model-agnostic EP key.
- **Fix required**: injective/hashed identity or collision detection + disambiguation; add two-partition external-mode round-trip test with colliding sanitized components.
- **Test gap**: external test covers only one partition; multi-partition test uses embedded mode.

#### Non-blocking (A1, A2)
- A1: `enable` flag ignored — both dump functions write files even when `enable=false`.
- A2: `partition_boundary` ascending-NodeId order undocumented as ABI.

#### Passing scope
- Model-agnosticism ✅; API/seam ✅; subgraph replacement ✅; consume-path symmetry ✅ (except B1).

---

### 2026-07-14T17:30:00Z: EPContext §55.4 writer v2 — collision fix + enable-gating (Leon, leon-14)

**By:** Leon (leon-14, revision owner)
**Commit:** `7a01f5f` (= `d9a4b6f` on branch)
**What:** Fixed B1 data-loss sidecar aliasing; folded in A1 enable-gating and A2 seam documentation.
**Why:** Deckard deckard-17 🔴 BLOCK; Batty locked out. Leon owns revision.

#### B1 fix
- Sidecar filename now: `<ctx_stem>_p{index}_<sanitized source>_<sanitized partition>.bin`. Partition index from `enumerate()` is injective and deterministic within one dump call — even colliding sanitized components produce distinct files. Each EPContext node stores the filename of its OWN blob.
- **Exact-identity guard**: two partitions sharing the same `source` AND `partition_name` rejected with `LoaderError::EpContext("duplicate partition identity …")`.

#### A1 fix
- `dump_ep_context`: if `!config.enable`, returns path and writes nothing (no sidecars, no model, no side effects before any I/O).
- `dump_session_ep_context`: short-circuits before calling any EP's `save_context`.

#### A2 doc
- Added doc-comment on `partition_boundary` seam: NodeId order is not a versioned ABI; integration owner must verify or extend.

#### Tests added
- `external_dump_colliding_sanitised_sources_do_not_alias` (B1 regression): `Vendor/EP` vs `Vendor_EP`, same partition_name, distinct non-UTF-8 blobs → `p0`/`p1` sidecars distinct; each node resolves its own blob byte-exact.
- `disabled_config_writes_nothing`; `duplicate_partition_identity_is_rejected`; `hostile_source_strings_sanitise_to_safe_bare_filename`.
- Session: `dump_disabled_config_is_a_no_op`.

---

### 2026-07-14T17:45:00Z: EPContext §55.4 writer v2 re-review (Deckard, deckard-18)

**By:** Deckard (deckard-18, re-reviewer)
**What:** 🔴 **BLOCK** — B1 resolved, but exact-identity rejection OVER-FIRES on legitimate distinct primary partitions. Revision owner: **Gaff** (Batty and Leon locked out).
**Why:** Re-review of leon-14 writer v2.

#### B1: resolved ✅
- `enumerate()` provides unique index per partition → `_p{index}_` sidecar names cannot repeat within one dump call.
- B1 regression test reproduced: `1 passed; 0 failed`.
- Hostile-string sanitizer and traversal guard intact.

#### Blocking regression
- v2 blanket-rejects any repeated `(source, partition_name)` pair (`writer.rs:147-160`).
- `partition_name` is optional; an EP can legitimately emit multiple unnamed primary partitions (`main_context=1`).
- Consume path (`session/src/epcontext.rs:109-142`) loads every `main_context=1` node independently; duplicate primary identities are fully loadable with injective sidecar names keeping blobs distinct.
- Session writer emits EVERY partition as `main_context=true` → two distinct same-EP unnamed partitions are wrongly rejected.
- The rejection test (`loader/tests/writer.rs:319-362`) codifies a false restriction.

#### A1: verified ✅; all other suites pass.

---

### 2026-07-14T17:50:00Z: EPContext §55.4 writer v3 — over-broad rejection removed (Gaff, gaff-13)

**By:** Gaff (gaff-13, revision owner)
**Commit:** `0fa025e` (= `6e65e85` on branch)
**What:** Removed the blanket `(source, partition_name)` duplicate-primary rejection introduced in v2. Added positive round-trip proof for two same-source primaries.
**Why:** Deckard deckard-18 🔴 BLOCK. Batty and Leon locked out.

#### Change
- Deleted duplicate-identity `HashSet<(&str, &str)>` guard and loop from `dump_ep_context`.
- Updated `# Errors` doc-comment: two partitions may legitimately share `source`+`partition_name` (safe because of injective per-partition sidecar index + independent consume-path loading).
- Deleted `duplicate_partition_identity_is_rejected` test.
- Added `duplicate_primary_identity_round_trips_external`: two `main_context=true` partitions, same `source="EpA"`, empty `partition_name`, distinct non-UTF-8 blobs, external mode → `m_ctx_p0_EpA.bin`/`m_ctx_p1_EpA.bin` exist with own blobs; reload via `load_model → ep_context_nodes → resolve_ep_context` confirms `r0==b0`, `r1==b1`, `r0!=r1`.

#### Kept intact
- B1 injective sidecar filenames; A1 enable-gating; A2 NodeId-order seam doc; broadened sanitizer test.

---

### 2026-07-14T18:00:00Z: EPContext §55.4 writer v3 re-review (Deckard, deckard-19)

**By:** Deckard (deckard-19, re-reviewer)
**What:** 🟢 **APPROVE** — regression closed; all green.
**Why:** Re-review of gaff-13 writer v3 @ `6e65e85`.

#### Findings
- No `HashSet<(&str, &str)>`, `(part.source, part.partition_name)` insertion, or `"duplicate partition identity"` rejection remains in `writer.rs`. API now explicitly permits repeated identities at `writer.rs:124-132`.
- `duplicate_primary_identity_round_trips_external` passes: two same-identity primaries round-trip byte-exact through external mode; `m_ctx_p0_EpA.bin`/`m_ctx_p1_EpA.bin` confirmed distinct.
- B1 still fixed: `_p{index}_` sidecar names at `writer.rs:168-169, 205-212`; sanitizer-collision regression (`tests/writer.rs:207-279`) passes.
- A1/A2/sanitizer all intact and verified.

#### Verification
- `cargo test -p onnx-runtime-loader`: PASS (encoder 9, EPContext 7, loader 15, writer 7; 0 failed).
- `cargo test -p onnx-runtime-session`: PASS (unit 12, conformance 1, optimizer parity 3, EPContext 11, executor 11; 0 failed).
- `cargo clippy -p onnx-runtime-loader -p onnx-runtime-session --all-targets -- -D warnings`: PASS.
- Eight-crate `cargo build`: PASS.

**Final merged commit on main: `0fa025e`.**

---

### 2026-07-14T18:55:00Z: EPContext DUMP options wired end-to-end (§21.4 / §55.5)

**By:** Chew (chew-25)
**Commit:** `3e8dbde` → cherry-picked to `main` as `c3d454c`
**Branch:** `squad/ort2-epctx-options` (off `origin/main` `0fa025e`)
**Scope:** Session-layer option plumbing that drives the §55.4 writer; capi FFI surface.

#### Option keys (§21.4) — one validating pass in `SessionBuilder::parse_options`

| Key                     | Type              | Default | Validation                                            |
|-------------------------|-------------------|---------|-------------------------------------------------------|
| `ep.context_enable`     | bool              | `false` | `1`/`0`/`true`/`false` (case-insensitive); else `InvalidOption` |
| `ep.context_file_path`  | `Option<PathBuf>` | `None`  | empty/unset → `None` (falls back to `<orig>_ctx.onnx`) |
| `ep.context_embed_mode` | `u8`              | `1`     | `0` external / `1` embed; any other → `InvalidOption` (fail-closed) |

Parsed config (`EpContextDumpConfig { enable, file_path, embed_mode }`) re-exported from `onnx-runtime-session`; stored on `InferenceSession`.

#### Session export entry point
`InferenceSession::export_ep_context(&self, orig_path, &[CompiledPartition]) -> Result<PathBuf>` — no-op when `enable=false`. Compiler-integration seam marked `TODO(compiler)` (no real compiling EP yet; proven with mock). Model-agnostic: no vendor/op/model literals in production `src/`.

#### capi (§55.5)
New ORT-compatible surface: `OrtSessionOptions` opaque handle; `ort2_create/release_session_options`; `ort2_add_session_config_entry`; `ort2_create_session_with_options` (null opts == plain create). Validation stays in session layer; unknown/bad values → `InvalidArgument`.

#### Tests
Session unit (`option_tests`), session e2e (`tests/epcontext.rs`: byte-exact non-UTF-8 round-trip via mock EP), capi (`tests/capi.rs`: plumbing + rejection). All gates green.

---

### 2026-07-14T18:55:00Z: EPContext §55.5 capi FFI safety + e2e round-trip review (Gaff, gaff-14)

**By:** Gaff (gaff-14) — non-author reviewer
**Target:** `squad/ort2-epctx-options` @ `3e8dbde` | **Verdict:** 🟢 GREEN (2 non-blocking advisories)
**Scope:** capi FFI memory safety + end-to-end correctness

**Key findings (all PASS):**
- NULL handling: every entry returns `InvalidArgument` rather than dereferencing null; `options` null in `create_session_with_options` intentionally tolerated (behaves like plain create).
- Invalid UTF-8: `CStr::from_ptr(...).to_str()` mapped to `InvalidArgument` for key/value/model_path — no unwrap.
- Ownership: `create_session_with_options` borrows (not moves) the options handle; caller still owns + must release. No double-free, no UAF.
- Panics: all four entries wrapped in `guard(catch_unwind)`.
- E2e: `builder_options_drive_export_byte_exact` round-trips non-UTF-8 blob byte-exact; `builder_disabled_export_writes_nothing` confirmed no-op.

**Advisories (non-blocking):**
- **A1:** No negative FFI tests for null `key`/`value` or invalid-UTF-8 at the C boundary (code handles correctly; coverage gap only). Follow-up: add tests exercising null/non-UTF-8 into `ort2_add_session_config_entry`.
- **A2:** Released/opaque-handle reuse after `ort2_release_session_options` is UB, unguarded at runtime — matches existing crate opaque-handle contract ("exactly once"); noted, not required.

`cargo test -p onnx-runtime-capi` → 4+13+0 passed; `cargo test -p onnx-runtime-session` → all suites passed; clippy clean.

---

### 2026-07-14T18:55:00Z: EPContext §55.5 parse_options refactor + model-agnosticism + export seam review (Deckard, deckard-20)

**By:** Deckard (deckard-20) — non-author reviewer
**Target:** `squad/ort2-epctx-options` @ `3e8dbde95effb006b28d117e3f8c5491d464e95f` | **Verdict:** 🟢 APPROVE — no regressions or advisories

**Key findings (all PASS):**
- `OptimizationLevel::parse` unchanged; replacement parser initializes same defaults, visits every entry once, retains same `UnknownOption`/`InvalidOption` fallback; no previously recognized key dropped.
- §21.4 validation fail-closed and tested for all three keys (bool forms, mixed case, rejection of `yes`; empty path → `None`; embed_mode `2` → `InvalidOption`).
- Combined-options test proves optimization + all three EPContext fields survive same parse pass.
- Model-agnosticism: no model/vendor/op literals in production `src/`; dump flows through `EpContextDumpConfig`/`CompiledPartition.ep`/`save_context`/`context_source_keys`.
- C API adds no divergent option logic; forwards verbatim to session layer.
- Export seam: `export_ep_context` constructs loader encoder over retained post-optimization graph + live weights; disabled export side-effect-free; `TODO(compiler)` seam correctly placed.
- Executor accessors immutable, `pub(crate)`, borrow from `&self`; weights remain `Arc`-owned.

`cargo test -p onnx-runtime-session` — PASS (unit 18, conformance 1, opt-parity 3, EPContext 13, executor 11). `cargo test -p onnx-runtime-capi` — PASS (unit 4, integration 13). Clippy + eight-crate build PASS.

---

### 2026-07-14T13:55:00Z: External-data path-traversal guard in weights loader (§19.2)

**By:** Deckard (deckard-21, gpt-5.6-sol)
**Commit:** `ba3f67a` (cherry-picked from `340d7b0`)
**What:** External initializer `location` fields are now rejected before mmap if they contain absolute/rooted paths or any `..` component, via a new `LoaderError::ExternalDataPath { path, reason }` variant. The guard (`resolve_external_path`) lives in `weights.rs` and mirrors the pre-existing guard in `epcontext.rs` without sharing code, preserving distinct error provenance.
**Why:** Untrusted ONNX external-data locations could escape the model directory and mmap arbitrary files; this brings the weights loader into security parity with the EPContext guard (§55 load path). The stale "TODO: add guard to weights.rs" note in the epcontext doc comment is cleaned up.
**Tests:** 4 new tests in `external_data_paths.rs` — rejection of absolute and `..`-traversal paths (unix-gated for absolute), positive round-trip for top-level and nested legit paths.
**Status:** MERGED to origin/main.

---

### 2026-07-14T13:55:00Z: Review — external-data path-traversal guard (Gaff, gaff-15)

**By:** Gaff (gaff-15, opus) — non-author reviewer
**Target:** commit `340d7b0` (author Deckard) — `crates/onnx-runtime-loader`
**Verdict:** 🟡 YELLOW — APPROVE with 3 non-blocking advisories

**Key findings (all PASS):**
- Completeness: exactly two untrusted read sites in the loader (`weights.rs:131` external initializer location, `epcontext.rs:241` sidecar) — both now guarded.
- Lexical correctness: absolute, `../x`, `a/../../x`, and any `..` component rejected; `weights.bin`, `./weights.bin`, `subdir/weights.bin` allowed. Verified by throwaway probe.
- Error handling / TOCTOU: guard fires before `store.mmap_file` — reject-before-open, no partial side effect.
- capi `map_loader_error` wildcard: new variant handled correctly via `_ => Fail`; build passes.
- Test quality: asserts specific `LoaderError::ExternalDataPath` variant AND echoed `path`; positive test uses real bytes in `CARGO_TARGET_TMPDIR`.
- Build/clippy/conformance: all green.

**Advisories (non-blocking — follow-up notes):**
1. **(Lexical-only guard / symlinks not resolved):** a `Normal` component that is a symlink to `/etc` would escape at the OS level. Parity with epcontext guard; accepted. Defense-in-depth: `canonicalize + starts_with(model_dir)`.
2. **(capi explicit variant mapping):** `L::ExternalDataPath { .. } => OrtErrorCode::InvalidGraph` (or `Fail`) would be more self-documenting than the wildcard catch-all in `map_loader_error`.
3. **(DRY — `resolve_external_path` duplicated):** `weights.rs` and `epcontext.rs` have near-verbatim guards differing only in error variant. Consider a shared `path_safety` helper (closure/enum for error kind) so future hardening (e.g. symlink resolution) is done once.

---

### 2026-07-14T13:55:00Z: FusedGemm (MatMul+Add+Relu) CPU kernel + shape rule + executable parity

**By:** Batty (batty-17, opus)
**Commit:** `4916618` (cherry-picked from `9e302a6`)
**What:** Added the `com.microsoft::FusedGemm` CPU kernel (`ep-cpu/src/kernels/fused_gemm.rs`) — `Relu(MatMul(A,B)+bias)` — reusing the shared `matmul_dense` GEMM, `broadcast_apply` bias-add, and a new shared `relu::relu_in_place` helper; registered under `("FusedGemm","com.microsoft",1)`. Added a matching shape-inference rule (`shape-inference/src/handlers/linalg.rs`). Extended the fusion bias-broadcast decline-guard in `optimizer/src/fusion.rs` to cover `FusedGemm` alongside `FusedMatMulBias`. Added a synthetic end-to-end parity test (`session/tests/fused_gemm_parity.rs`) — `optimization="none"` vs `"all"` are byte-identical (max_abs 0.0); the fused graph contains exactly 1 `FusedGemm` / 0 stray MatMul+Add+Relu.
**Why:** Completes the fusion trio — `FusedGemm` was registered/emitted by the optimizer but had no kernel or shape rule, so `optimization="all"` on a MatMul+Add+Relu graph would hit `UnsupportedOp`. `bert_toy` uses GELU/Erf (no Relu) so a synthetic test is the only way to exercise the fused path end-to-end.
**Status:** MERGED to origin/main. **Fusion trio complete: LayerNorm ✅ / FusedMatMulBias ✅ / FusedGemm ✅ (all executable).**

---

### 2026-07-14T13:55:00Z: Review — FusedGemm kernel + shape rule + parity (Roy, roy-19)

**By:** Roy (roy-19, opus) — non-author reviewer
**Target:** commit `9e302a6` (author Batty)
**Verdict:** 🟢 GREEN — approve, no blocking issues

**Key findings (all PASS):**
- Fusion-guard generalization: `matmul_bias_broadcast_ok` reads `m.nodes.first()` (MatMul) and `m.nodes.get(1)` (Add) — correct for both 2-node (FusedMatMulBias) and 3-node (FusedGemm) cases. Trailing Relu (node 2) is shape-neutral and correctly ignored by the guard. Verified with a throwaway expanding-bias probe (guard declines; reverted — not committed).
- bert_toy unchanged: all `bert_toy_*` session tests green; FusedGemm never fires on bert_toy (no Relu in FFN).
- Kernel correctness: `FusedGemm::execute` = `matmul_dense(A,B)` → `broadcast_apply(bias)` in place → `relu_in_place(&mut out)`. Stage order is exactly `Relu(MatMul + bias)`. Byte-identical to FusedMatMulBias plus one trailing `relu_in_place`.
- Shape rule: delegates to `matmul_shape` (output == MatMul(A,B)); bias broadcasts; Relu shape-neutral. Registered under `com.microsoft` v1.
- Synthetic parity test quality: tight 1e-6 atol; pre-Relu has negatives so Relu actually clamps; graph asserts 1 FusedGemm / 0 standalone MatMul+Add+Relu.
- No new `unsafe`; `#![forbid(unsafe_code)]` intact in optimizer + shape-inference.
- Build/clippy/all-crate: green.

**Advisory (non-blocking):**
- Consider adding a permanent 3-node `FusedGemm` expanding-bias decline test (mirroring `declines_matmul_add_when_bias_expands`) so the guard generalization is regression-protected in-repo, not just via throwaway. Purely additive; not required for approval.

---

### 2026-07-14T14:50:00Z: DRY external-path guard + explicit capi mapping — Gaff advisories B/C (Leon, d6854c9)
**By:** Leon
**What:** Added a shared `guarded_join` helper in a new `pathsafe.rs` module, used by both `weights.rs` and `epcontext.rs` while preserving their distinct error variants (`LoaderError::ExternalDataPath` / `LoaderError::EpContextPath`). Explicitly mapped both path errors to the C API `InvalidGraph` status in `map_loader_error`.
**Why:** Closes Gaff advisories B (explicit capi mapping) and C (DRY guard) from the external-data path-traversal review with a behavior-identical refactor. `guarded_join` rejects absolute, rooted, `..`-traversing, and parent-dir paths; allows `CurDir` and nested normal components. No new unsafe; existing mmap unsafe blocks untouched.
**Commit:** `e60dd6b` → cherry-picked to main as `d6854c9`.
**Review:** Deckard (deckard-22) 🟢 — behavior-identical, both sites guarded, variants preserved, all builds + tests green.

---

### 2026-07-14T14:50:00Z: Review — DRY guard refactor + capi mapping (Deckard, deckard-22)
**By:** Deckard
**Verdict:** 🟢 APPROVE

**Findings:**
- `pathsafe::guarded_join` is behavior-identical to both prior guards: rejects absolute paths, every `ParentDir` component (including net-inside paths), roots/prefixes; allows `CurDir` and normal nested; returns `base.join(rel_path)`.
- Both `weights.rs` and `epcontext.rs` exclusively route external locations through `guarded_join`; no alternate unguarded `.join` path remains.
- Distinct errors preserved: `LoaderError::ExternalDataPath` / `LoaderError::EpContextPath` with original path and reason unchanged.
- C API mapping explicitly classifies both path variants as `InvalidGraph`, consistent with sibling graph/IR structural failures.
- No new unsafe; no model-specific behavior; no back-compat shim.

**Verification:** `cargo test -p onnx-runtime-loader -p onnx-runtime-capi` — passed; 8-crate `cargo build` — passed; `cargo clippy -p onnx-runtime-loader -p onnx-runtime-capi --all-targets -- -D warnings` — passed; `cargo test -p onnx-runtime-session` — passed.

**Decision:** Approve commit `e60dd6b`; no revision required.

---

### 2026-07-14T14:50:00Z: AttentionFusion — fuse SDPA core into com.microsoft::FusedAttention (Batty, 64edd75)
**By:** Batty
**What:** Fused the scaled-dot-product-attention CORE (`MatMul(Q,Kᵀ)·scale[+mask]→Softmax(axis=-1)→MatMul(·,V)`) into a new `com.microsoft::FusedAttention` node with a CPU kernel and shape rule. This completes the Phase-2 optimizer OpFusion + AttentionFusion sub-items.

**Fused node contract (`com.microsoft::FusedAttention` v1):**
- Inputs: `[Q, K, V]` + optional `[mask]` (iff chain had a mask Add).
- Attributes: `scale` (f32), `k_transposed` (int 0/1: 1=K already Kᵀ; 0=kernel transposes internally).
- Output: final `probs·V` value (original output value id, downstream wiring preserved).

**Key matcher guards (decline-to-fuse — never guess defaults):**
- Anchor: single-in/single-out Softmax with explicit `axis` resolving to last dim (absent → decline).
- Softmax output must be LEFT operand of the ·V MatMul.
- Scale must be `Mul`/`Div` by a concrete scalar f32 (strict `numel==1` inline initializer); its non-const operand must be a MatMul output.
- If mask Add sits between scaling and Softmax, exactly one operand must parse as scaled-scores (both/neither → decline).
- No interior matched value may escape the region.

**Files:** `ep-cpu/src/kernels/fused_attention.rs` (kernel + 5 unit tests), `session/tests/fused_attention_parity.rs`, `optimizer/src/fusion.rs` (`RewriteKind::Attention`, matcher + Roy's expanding-bias FusedGemm decline test folded in), `shape-inference/src/handlers/linalg.rs` (shape rule), plus wiring in `mod.rs`, `provider.rs`, `op_rules.rs`, `bert_toy_optimized_parity.rs`.

**Results on bert_toy:** 5 FusedAttention, 0 surviving Softmax; LayerNorm=12/FusedMatMulBias=32 unaffected. vs onnxruntime reference max_abs **1.043e-7** (bound 2e-3); vs opt-off drift **1.416e-7** (DRIFT_ATOL 1e-5). `"basic"` byte-identical to opt-off.

**Commit:** `39a23c8` → cherry-picked to main as `64edd75`.
**Reviews:** Roy (roy-20) 🟢 (matcher robustness; 4 adversarial declines) + Chew (chew-26) 🟢 (kernel numerics hand-verified; parity 1e-6).

**Follow-up note (Roy, non-blocking):** The matcher enforces last-axis softmax + structural equivalence but not rank-4. On a rank-2 scaled-bias-softmax head with a QKᵀ-style MatMul it could emit FusedAttention — this is semantics-preserving (kernel computes the identical value), but consider a note/test documenting rank-2 equivalence if desired.

---

### 2026-07-14T14:50:00Z: Review — AttentionFusion kernel/numerics/shape (Chew, chew-26)
**By:** Chew (NON-AUTHOR reviewer — kernel/shape/numerics half)
**Artifact:** commit 39a23c8 (author Batty)
**Verdict:** 🟢 GREEN — approve

**Key findings:**
1. `softmax_slices` extraction: purely a visibility change (`fn` → `pub(crate) fn`), body byte-for-byte unchanged. No regression. ✅
2. Kernel numeric correctness: scale-before-mask-before-softmax order correct; both k_transposed branches compute Q·Kᵀ correctly; batched leading dims via `matmul_dense`+`broadcast_shapes`; mask broadcast `[b,1,1,s]→[b,h,sq,sk]` correct. Hand-derived unmasked pre-transposed test confirmed numerically. ✅
3. Unit tests hand-computable and non-tautological: 5 ep-cpu tests check concrete values against independent triple-loop references. ✅
4. k_transposed matcher↔kernel agreement consistent: both paths yield Q·Kᵀ. ✅
5. Shape rule correct and registered: mirrors k_transposed swap, `matmul_shape(q,k_eff)→matmul_shape(scores,v)`. 3 shape tests correct. ✅
6. Parity test quality HIGH, tight, non-tautological: ATOL=1e-6 (not loosened), both masked+unmasked, compares against distinct standalone-kernel path. ✅
7. Safety: zero new `unsafe` blocks; `#![forbid(unsafe_code)]` intact in ir/optimizer/shape-inference crates. ✅

**Build/test:** ep-cpu 113 + shape op_rules 57 + session (fused_attention_parity 2/2, bert_toy_optimized_parity 3/3) all green. clippy -D warnings clean.

---

### 2026-07-14T14:50:00Z: Review — AttentionFusion optimizer/matcher (Roy, roy-20)
**By:** Roy (NON-AUTHOR reviewer — optimizer/matcher half)
**Artifact:** commit 39a23c8 (author Batty)
**Verdict:** 🟢 APPROVE

**Key findings:**
1. Matcher misfire test — CANNOT MISFIRE (adversarial-tested). Four adversarial graphs all correctly declined:
   - Classifier head with bias (`MatMul(x,W)+b→Softmax→MatMul`): both Add operands fail parse_scale → decline. ✅
   - Full SDPA silhouette but scaled tensor is Relu output (not QKᵀ): parse_scale MatMul check → decline. ✅
   - Ambiguous mask (`Add(Div(MM1,c1),Div(MM2,c2))`): both operands parse as scale → decline. ✅
   - Interior value escapes (probs is graph output): escape guard → decline. ✅
2. Decline-guard soundness: every required guard returns None/declines; no silent defaults anywhere. ✅
3. Both scale forms (Div and Mul) correctly handled. ✅
4. k_transposed matcher↔kernel contracts agree (verified both sides). ✅
5. bert_toy fuse + conformance held: FusedAttention=5, Softmax=0; parity within ATOL. ✅
6. Roy's folded FusedGemm advisory: `declines_fused_gemm_when_bias_expands` test present and asserts a decline. ✅
7. Model-agnostic; no new unsafe in matcher. ✅

**Build/test:** optimizer 40 + session 18 + ep-cpu 5 fused_attention tests all green. clippy -D warnings clean.

**Non-blocking follow-up:** Matcher doesn't require rank-4. On a rank-2 QKᵀ-style SDPA the kernel computes the identical value (semantics-preserving). Consider a note/test documenting rank-2 equivalence if desired. NOT required for approval.
