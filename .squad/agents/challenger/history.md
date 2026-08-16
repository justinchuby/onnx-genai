# Challenger — History (compacted 2026-08-12T06:00:00Z)

Append-only. Each entry records a claim challenged, the verdict, and what the challenge changed.

## Role
Challenge non-intuitive claims and direction-setting measurements. Hired after several measurements were accepted without asking what else could produce the result. See `history-archive.md` for full history.

## Durable lessons
- A reviewer's "SAFE" is not proof; verify the load-bearing claim independently (re: #31988 n%8 false positive).
- Reviewer blockers must be verified with the same standard as author claims (re: #32001 false-positive B-NEW-1).
- `nm -D` is the wrong instrument for ORT C API presence checks — the API is delivered via a function-pointer struct.

## Recent work (current wave, 2026-08-12)

### 2026-08-12 — Adversarial review PR #762 v4 (Opus fourth review, final)
Reviewed commits af45043fd + b906ab2bb. **0 blockers. 3 substantive. Ready to leave draft.**
- Heap overflow provably gone on all dtype paths (both single-kernel and routed paths traced).
- `RoutedSlotKind` positionally sound — interior/trailing/multiple absent verified.
- `end_version = i32::MAX` correct; `struct_size` guards sound; assignment assertion non-vacuous.
- No fifth absent-slot defect found.
- S1: canary tests allocate `byte_size`; production uses `max(byte_size, 8)`.
- S2: `mark_absent()` advisory-only; no automatic write-dtype enforcement.
- S3: phantom `Buffer` slots for absent outputs inflate `num_intermediate_buffers`.
86 passed / 0 failed. Clippy clean.

Full pre-compaction history in `history-archive.md`.

## 2026-08-12 — PR #31973 delta review (N1 fix)

Reviewed commit a49b702a36: `HasCenteredTwoPassKernel()` arch guard for six precision suites.
- **No blockers.** Guard is correctly scoped, mirrors production `#if` exactly.
- Suites still run on x86 (verified 41/2, 43/43 fresh build).
- Benchmark baseline (Welford fp32) is fair; 11× plausible.
- One nit: `mlas.h` comment says "x86-64" but `#if` covers 32-bit too.
- No leaks. Ready to leave draft.

## 2026-08-12 — PR #31973 wording nit resolved (Deckard follow-up)

Challenger's delta review of `a49b702a36` found one nit: `mlas.h` said "x86-64" while the
`#if` gate covers both AMD64 and IX86. Deckard fixed this to "x86 (32-bit and 64-bit)" across
three files as `4a16925a88`. PR #31973 and #31974 both marked ready for review.

**Board state:** #762 ready · #31985 merged · #31973 ready · #31974 ready · #32001 ready
· #32003 ready · #31993 draft · #31988 draft (parked)
