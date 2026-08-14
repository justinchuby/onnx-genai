# Challenger — History Archive

## ARCHIVED 2026-08-12T06:00:00Z (Scribe #762 memory-safety wave compaction)

### 2026-08-06 — Hired
Requested by @justinchuby to challenge non-intuitive claims and direction-setting measurements. Hired after prefetch A/B, VMM arena, lookahead sweep, and KV on-demand tests were accepted without asking what else could produce the result.

### 2026-08-10 — ORT Plugin EP Export ABI: Three Claims Challenged
- Nabil's `CreateEpFactories` export symbol: SOUND.
- Pris's `CreateEpApiFactories` export symbol: CONTRADICTED (typedef name vs dlsym name).
- Pris's "e2e test impossible, nm -D shows 2 symbols": CONTRADICTED (ORT C API delivered via function-pointer struct, invisible to nm -D).
Written `docs/ep-plugin/EP_PLUGIN_EXPORT_ABI_TRUTH.md`. Implementation unblocked for e2e testing.

### 2026-08-11 — PR #762 fourth review (fbd565160..4757e25b6)
Two BLOCKERS: B1 `__absent_output_*` string sentinel forgeable from model content; B2 `filter_map(|d| d.as_static())` destroys rank (`[batch, seq, 768]` → `[768]`). Coco fixed both.

### 2026-08-11 — Re-review PR #31974 (BFloat16 CPU EP LayerNorm)
All 6 original blockers still fixed. 17 tests non-vacuous. Two cosmetic nits. Verdict: ready to leave draft once CI green.

### 2026-08-11 (upstream CI correction wave) — Re-review PR #31974
Post-rebase fresh review. All 6 blockers still fixed. Stat tests fail genuinely against pre-B5 code (bf16 quantization step ~3.9e-3; tolerance 1e-5 — 390× tighter). Verdict: 0 blocking, 0 substantive, 2 nits.

### 2026-08-12 — Adversarial re-review PR #32001 (Apple Accelerate)
Found 1 new BLOCKING bug: `build.py` references `args.use_apple_accelerate` without `add_argument`. B2 still uncorrected (PR body claims FATAL_ERROR, iOS/universal2, contradicting actual code). Verdict: not ready to leave draft.

### 2026-08-12 — Re-review PR #32001 (false positive B-NEW-1)
B-NEW-1 (missing add_argument) disproved: `python3 tools/ci_build/build.py --help` shows flag listed. False positive. Lesson: reviewer blockers must be verified with same standard as author claims.
