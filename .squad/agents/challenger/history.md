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
