### 2026-08-07: PR #728 round-4 — general symbol-provenance record closes the derived-symbol capture hole; classifier defaults to FAIL-SAFE

**By:** Sebastian

**What**

Harry's round-4 REJECT found a NEW lineage-loss site beyond `broadcast_dim`:
`SymbolInterner::lower` interns a *derived* `DimExpr` (e.g. `seq_kv*8` from
`Reshape([seq_kv,8],[-1])` / `Flatten`) to a BRAND-NEW fresh `SymbolId` and
recorded NOTHING, so `symbol_unifications` carried no edge `seq_kv → fresh`. A
downstream `Sigmoid` carrying only the fresh symbol was wrongly classified
capture-safe → silent decode corruption. This is the 4th reject round, each a
new lineage-loss site (exact-membership → equivalence-closure → non-elementwise
handlers → derived-symbol interning), so I generalized the record into a
symbol-provenance/dependency graph and made the classifier structurally safe.

**Which default I shipped: FAIL-SAFE (default), with the DENYLIST-over-provenance
retained behind a switch.** Per the task's explicit rule for the "can't GPU-
measure locally" case: I implemented BOTH and default to fail-safe
(`ONNX_GENAI_CAPTURE_CLASSIFIER` env: unset/`failsafe`/`1` ⇒ fail-safe,
`denylist`/`deny`/`0` ⇒ denylist). A false "capture-safe" is silent corruption,
so the safe-by-default choice keeps an op eager whenever its shape lineage is not
provably pinned. The coordinator GPU-measures both; if fail-safe holds at/near 34
segments SHIP it (safe AND fast); if it regresses (~94) flip the one env var to
`denylist` — which the Step-1 provenance + closure made safe against the
derived-symbol hole (audit below).

**Step 1 — record derivation provenance at the `lower` chokepoint (additive, byte-identical):**

- `crates/onnx-runtime-shape-inference/src/context.rs`
  - `SymbolInterner` gained `initial_floor: u32` (context.rs:193 — the id floor
    at/above which every symbol is inference-minted), `derivations:
    Vec<(SymbolId,SymbolId)>` (`(derived, source)` directed edges, :220), `opaque:
    Vec<SymbolId>` (:227), plus `record_derivation`, `record_opaque`,
    `initial_floor()`, `derivations()`, `opaque()` accessors.
  - `lower` (context.rs:315) now records lineage at EVERY minting branch, while
    returning the SAME `Dim` as before:
    - cache-insert branch and cache-hit branch → `record_expr_derivation(id,
      expr)` (context.rs:345,350,356): a `(fresh → source)` edge for each distinct
      symbol in the expression (`DimExpr::symbol_ids()`).
    - `is_overflow()` branch (:321) and negative-extent branch → mint fresh AND
      `record_opaque(id)`: the overflow sentinel has dropped its terms, so no
      source is recoverable — mark it opaque so a conservative consumer treats it
      as disqualifying (eager), never constant. (Handles the task's
      "overflow/negative → err toward growing, never toward capturable".)
- `crates/onnx-runtime-ir/src/graph.rs` — additive `Graph` fields:
  `symbol_derivations` (:61), `symbol_opaque` (:67), `inference_symbol_floor:
  Option<u32>` (:74). `Graph` is `Default`/`Clone`; no external struct literals.
- `crates/onnx-runtime-shape-inference/src/infer.rs:181-189` — persist the records
  after write-back: `symbol_unifications` (kept), `symbol_derivations` (sorted +
  deduped — a hot derived dim re-records on each cache hit), `symbol_opaque`, and
  `inference_symbol_floor = Some(interner.initial_floor())`.

**Executor — one closure, two seeds (`kernel_cache.rs`):**

- `close_disqualifying_set` (kernel_cache.rs:412) replaced the old union-find
  `close_growing_under_symbol_unification`. A plain worklist BFS over the directed
  adjacency `{a↔b for each broadcast unification} ∪ {source→derived for each
  derivation}`. Broadcast edges are UNDIRECTED (the two symbols denote the same
  dim); derivation edges are DIRECTED `source→derived` ONLY — so a growing source
  poisons its derived product, but a pinned `batch` is NEVER poisoned by a
  `batch*seq_kv` product that also depends on growing `seq_kv` (no over-poisoning
  of pinned roots).
- `collect_structural_growing_symbols` (:199) factors out the KV-structural seed
  (recognized attention ops ∪ generic declared `past…`/`present…` rank-4 KV I/O ∪
  CSA output-5 `last_axis_outputs=&[5]`).
- `compute_capture_growing_symbols` (:186, DENYLIST) = close(structural ∪ opaque).
- `compute_not_pinned_symbols` (:337, FAIL-SAFE) = close(structural ∪ opaque ∪
  every inference-minted symbol (`id >= floor`) with NO derivation-LHS provenance
  — the untraceable data-dependent/`fresh_dim` symbols). A minted symbol whose
  provenance traces only to pinned roots is NOT seeded, so a `Reshape`/`Flatten`
  product of pinned dims stays capturable (this is why fail-safe + provenance does
  not collapse to the naive-allowlist's 94).
- `CaptureClassifier{FailSafe,Denylist}` + `from_env()` (:262,285); the single
  production entry point `compute_capture_disqualifying_symbols` (:310) dispatches
  (default fail-safe) and logs under `ONNX_GENAI_LOG_GROWING_SYMBOLS`.
- `build.rs:498` now calls `compute_capture_disqualifying_symbols`.
  `node_capture_seq_independent` (exact both-edge membership) is UNCHANGED — the
  switch only changes which symbol set it tests against.

**Failing-pre / passing-post tests (`executor/tests.rs`):**

- `reshape_derived_growing_symbol_keeps_downstream_consumer_eager` (tests.rs:1110)
  and `flatten_derived_growing_symbol_keeps_downstream_consumer_eager` (:1197):
  real inference where declared `past_key` supplies growing `seq_kv`,
  `Reshape([seq_kv,8],[-1])` / `Flatten([batch,seq_kv,8],axis=1)` derives
  `seq_kv*8`, feeding a `Sigmoid`; assert the derived symbol is in the closed set
  and the `Sigmoid` is EAGER. **Confirmed FAIL on HEAD 571ea0d9** (before the fix):
  `got {SymbolId(1)}` — only `seq_kv`, the derived symbol absent — then PASS after.
- `failsafe_pinned_derived_fresh_symbol_stays_capturable` (:1262): `Reshape([batch,
  8],[-1])` derives `batch*8` tracing only to pinned `batch`; asserts fail-safe
  does NOT disqualify it and the consumer stays CAPTURABLE (the anti-94 proof).
- `failsafe_untraceable_minted_symbol_is_eager_but_denylist_admits_it` (:1336): a
  permissive-broadcast degrade (`[batch,4] ⊕ [batch,5]`) mints an unknown fresh
  symbol with no provenance; asserts the DENYLIST admits it (capturable — the
  latent hole) while FAIL-SAFE disqualifies it (eager). Proves the structural win.
- Kept/green: Batty's `matmul_batch_alias_…` + `broadcast_records_symbol_unification…`,
  Leon's `growing_symbol_alias_…` + CSA output-5 (`csa_output5_selections_…`) +
  `benign_fresh_symbol_is_not_growing_and_stays_capturable` (denylist semantics,
  calls `compute_capture_growing_symbols` directly, unaffected by the default).

**Why**

**Completeness argument (how a `Dim::Symbolic` reaches an output edge, and how
each path is covered).** Enumerate every way a symbol lands on a value's shape:

  1. **Declared** on a graph input/initializer (`id < floor`). Root: pinned iff
     not in the structural-growing set; growing iff on a KV seq axis (source 1/2).
  2. **Copied unchanged** from an input (Identity, elementwise passthrough): same
     `SymbolId`, no new lineage — classified as its source.
  3. **Substituted via `broadcast_dim`** (two symbols → one representative): the
     ONLY place inference collapses distinct symbols; recorded in
     `symbol_unifications` (elementwise, MatMul batch, Einsum ellipsis, Concat
     non-concat, Expand — all funnel here). Covered by the undirected closure.
  4. **Interned via `lower`** from a derived `DimExpr` (Reshape/Flatten/Conv/Pool/
     Slice-size/…): the ONLY place a non-bare expression becomes a fresh symbol;
     recorded in `symbol_derivations` (Step 1). Covered by the directed closure.
  5. **Minted directly via `fresh_dim()`/`fresh_symbol()` by a handler**
     (NonZero/Unique/Range/data-dependent Slice, permissive-broadcast degrade) —
     `id >= floor`, NO derivation edge.
  6. **Minted via `lower` overflow/negative degrade** — recorded in
     `symbol_opaque`; `id >= floor`, no derivation edge.

`broadcast_dim` (3) and `lower` (4/6) are the two — and only two — inference
chokepoints where a minted symbol acquires a *recorded* dependency; (5) is the
residual set that mints WITHOUT recording. For the **FAIL-SAFE** classifier every
path is covered: 1–2 resolve to roots; 3 via unification closure; 4 via derivation
closure; **5 and 6 are minted-without-full-provenance (`id >= floor`, not a
derivation-LHS) ⇒ seeded disqualifying ⇒ eager.** So no op can put a symbol on an
output edge and be admitted unless that symbol provably traces (via the recorded
lineage) to pinned, non-growing roots. Unknown ⇒ eager ⇒ safe: the whole
"unrecorded lineage site" bug class is structurally eliminated (no more whack-a-
mole). This is why I default to fail-safe.

**DENYLIST audit (the fallback, if the coordinator flips the switch).** After Step
1 the denylist closes over broadcast(3) + derivation(4) + opaque(6), so the
round-4 derived-symbol hole and the overflow case are closed. The residual risk is
class (5): a handler that mints a `fresh_dim()` from an input carrying a growing
symbol WITHOUT recording provenance — e.g. `transform.rs:137` (`Reshape` non-exact
symbolic division `checked_div == None → ctx.fresh_dim()`), and the ~95
`fresh_dim()`/`fresh_symbol()` call sites across 15 handler files
(`grep -rn "fresh_dim()\|fresh_symbol()" crates/onnx-runtime-shape-inference/src/`).
For the denylist to silently corrupt, such a site must be reachable in the decode
capture region AND its minted dim must actually grow each step. Most of these
(pooling/resize/space_depth/generator/signal/ml/selection) mint from static or
data-dependent-but-not-KV extents; the reachable-in-decode + growing-dependency
intersection is not provably empty, which is exactly why the denylist is the
fallback and fail-safe is the default. If the coordinator ships the denylist, this
audit is the boundary to hold; fail-safe needs no such audit.

**No over-poisoning / byte-identity.** Derivation edges are directed
`source→derived`, so pinned roots are never poisoned through a derived product;
the denylist growing-set stays minimal (structural KV only, plus genuinely
growing-derived symbols). `lower` returns the identical `Dim`; recording is a pure
`Vec` push — inference output is byte-identical (full shape-inference suite green).

**Validation (local, all pass):**
- `cargo fmt --all --check` — clean.
- `cargo clippy -p onnx-runtime-ir -p onnx-runtime-shape-inference -p
  onnx-runtime-session --all-targets` — clean; `cargo clippy -p
  onnx-runtime-session --features cuda --all-targets` — clean.
- `cargo test -p onnx-runtime-ir` — 67 pass.
- `cargo test -p onnx-runtime-shape-inference` — 16 + 41 + 275 + doctests pass
  (byte-identical inference).
- `cargo test -p onnx-runtime-session` — 145 lib + integration all pass (incl. the
  4 new classifier tests).
- `cargo test -p onnx-runtime-session --features cuda --lib` — 148 pass.

**Growing-set / segment count:** NOT measured locally (no GPU / 35B model here).
The denylist growing-set is unchanged in construction from Leon's shipped 11
symbols (structural KV only; derivation edges only add symbols DERIVED from those
11, and none are present in the real 35B decode graph's captured seams unless a
Reshape/Flatten actually bakes a KV length — which is the case this fix targets).

**Coordinator: EXACT commands to run on the shared H200 (I did NOT run these).**
Measure BOTH classifiers (A/B) and both capture ON/OFF. New default = fail-safe
(env unset); flip to denylist with `ONNX_GENAI_CAPTURE_CLASSIFIER=denylist`.

```bash
# (A) FAIL-SAFE (default) — growing/disqualifying set size + segment count on 35B-A3B QMoE:
CUDA_VISIBLE_DEVICES=0 taskset -c 1 \
  ONNX_GENAI_CUDA_KV_MAX_LEN=4096 ONNX_GENAI_LOG_GROWING_SYMBOLS=1 ONNX_GENAI_CUDA_GRAPH=1 \
  cargo run --release -p onnx-runtime-session --features cuda --bin profile_native -- \
  --pipeline --backend native --ep cuda --steady --warmups 2 --runs 3 --tokens 128 --decode-skip 8 -t0

# (B) DENYLIST-over-provenance — same run with the switch flipped:
CUDA_VISIBLE_DEVICES=0 taskset -c 1 \
  ONNX_GENAI_CAPTURE_CLASSIFIER=denylist \
  ONNX_GENAI_CUDA_KV_MAX_LEN=4096 ONNX_GENAI_LOG_GROWING_SYMBOLS=1 ONNX_GENAI_CUDA_GRAPH=1 \
  cargo run --release -p onnx-runtime-session --features cuda --bin profile_native -- \
  --pipeline --backend native --ep cuda --steady --warmups 2 --runs 3 --tokens 128 --decode-skip 8 -t0

# Decision: if (A) holds at/near 34 segments, keep fail-safe as the ship default.
# If (A) regresses materially (≈94), ship denylist (set ONNX_GENAI_CAPTURE_CLASSIFIER=denylist
# in the launch env / make Denylist the const default) and rely on the audit above.

# Byte-exact vs fp32 oracle — for the SHIPPED classifier, capture ON and OFF:
CUDA_VISIBLE_DEVICES=0 ONNX_GENAI_CUDA_GRAPH=1 \
  cargo test --release -p onnx-runtime-session --features cuda \
  qwen36_35b_a3b_qmoe_native_cuda_matches_fp32_oracle -- --nocapture
CUDA_VISIBLE_DEVICES=0 ONNX_GENAI_CUDA_GRAPH=1 \
  cargo test --release -p onnx-runtime-session --features cuda \
  qwen36_35b_a3b_qmoe_native_cuda_hybrid_continuation_matches_fp32_oracle -- --nocapture
# repeat both with ONNX_GENAI_CUDA_GRAPH=0 to confirm capture-OFF parity.
```

Do NOT merge. New HEAD sha: `0fd87df3` (pushed to
`squad/elementwise-capture-seqindep`; this doc commit sits on top of it).
