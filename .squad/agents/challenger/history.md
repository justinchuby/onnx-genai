# Challenger — History

Append-only. Each entry records a claim challenged, the verdict, and what the
challenge changed.

## 2026-08-06 — Hired

Requested by @justinchuby:

> "hire一个新人 叫挑战者 职责是challenge不符合常识或者直觉的claim，每次观察得到的重要、影响技术方向的结果，都让它想想是不是哪里有疏漏"

Created after a session in which several direction-setting measurements turned
out to be narrower or wronger than first reported, and were only caught because
someone happened to ask a second question:

- A weight-prefetch A/B that measured demand fallback against itself, because
  the prefetch guard had silently declined every opportunity (#673).
- A VMM arena that committed ~800 MB less and still needed a *higher* VRAM
  limit — granularity waste, not noise (#682).
- A lookahead-depth sweep whose best median sat inside the baseline's range, so
  no win was established at any depth (#673).
- A KV on-demand-commit test that would have passed unchanged if the feature had
  silently fallen back to eager commit (#682 review).

The common shape: a result was accepted without asking what *else* could produce
it. Challenger's remit is that question.

## 2026-08-10 — ORT Plugin EP Export ABI: Three Claims Challenged

**Claims under review:**
- Claim A (Nabil): export symbol is `CreateEpFactories` — **SOUND**
- Claim B (Pris): export symbol is `CreateEpApiFactories` — **CONTRADICTED**
- Claim C (Pris): e2e test impossible, `nm -D` shows only 2 symbols — **CONTRADICTED**

**Method:** Downloaded ORT 1.27.0 release (SHA-256 verified against `ort-sys/build.rs`),
read `onnxruntime_c_api.h` and `onnxruntime_ep_c_api.h` directly.

**Findings:**
1. The required export symbols are `CreateEpFactories` and `ReleaseEpFactory` (both
   required). The typedef names are `CreateEpApiFactoriesFn` / `ReleaseEpApiFactoryFn`
   but the `dlsym` lookup name is `CreateEpFactories`. Nabil was right; Pris confused
   the typedef name with the export name.
2. `RegisterExecutionProviderLibrary`, `GetEpDevices`, and
   `SessionOptionsAppendExecutionProvider_V2` are all members of the `OrtApi` struct
   (since v1.22). They are invisible to `nm -D` because the entire ORT C API is
   accessed through the `OrtApi` function-pointer struct returned by
   `OrtGetApiBase()->GetApi(version)`. Pris used `nm -D` — the wrong instrument
   entirely. The conclusion that "e2e test is impossible" was invalid.
3. `ort_version_supported` provides forward-compat (ORT skips calling newer members),
   not fail-closed rejection. Justin's fail-closed requirement needs an explicit check.

**What changed:** Full authoritative vtable dump and call sequence written to
`docs/EP_PLUGIN_EXPORT_ABI_TRUTH.md`. Decision record filed. Implementation
unblocked for e2e testing.

## 2026-08-11 — PR #762 fourth review (fbd565160..4757e25b6)

**Task:** Fourth adversarial review of PR #762.

**Verdict:** 2 BLOCKERS.

- **B1 (BLOCKER):** `__absent_output_*` string sentinel is forgeable from model content. In-band signalling; any model naming a tensor `__absent_output_0_2` bypasses dtype validation and allocates a scratch buffer. Replace with out-of-band `HashSet<ValueId>` (arena indices uninfluenceable from model content).
- **B2 (BLOCKER):** `filter_map(|d| d.as_static())` on shape dims destroys rank. `[batch, seq, 768]` → `[768]`. Same class as original output-slot compaction bug. Fix with `map(|d| d.as_static())` → `Vec<Option<usize>>`.
- S1: `conformance_mixed_partition` doesn't assert EP claimed any subgraph.
- S3: `input_slots` logic verified correct for all-present/absent-interior/trailing cases.

Coco fixed both blockers.
