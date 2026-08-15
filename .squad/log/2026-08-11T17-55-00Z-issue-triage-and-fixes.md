# Session log — 2026-08-11T17:55:00Z — issue triage + autonomous fixes

## Issue triage
- Reviewed ~90 open issues oldest-first.
- **CLOSED 18 stale** issues:
  - Serving-dashboard / citation-tooling audit cluster: **#457–#475**.
  - Removed-code-path defects: **#481**, **#482**, **#768**.

## Autonomous fixes shipped
- **#702** — CUDA `GatherBlockQuantized` default symmetric zero-point
  (`1 << (bits-1)`) when `zero_points` absent; fixes empty/non-finite output on
  mobius-converted 14B/27B vs CPU. **PR #785 — MERGED.**
- **#701** ORT paged-KV recurrent guard (`ort_session_has_recurrent_state`,
  mirrors native #700) + **#467** loader dedup (`model_dir_missing_err`).
  **PR #786 — MERGED.**
- **#686** — VLM compat fixture executable graphs + `sequence_source:
  "inputs_embeds"` synth fix; re-enabled `onnx-genai-server` in CI (moved to
  `ORT_BACKED`). **PR #788 — MERGED.**
- DRY decoder-io glue → shared `GenAiConfig::derive_model_io_spec_from_graph`
  helper (behavior-preserving refactor). **PR #784 — MERGED.**
- CI honesty whitelist for **#776**'s CPU-only test. **PR #789 — auto-merge.**

## External / follow-ups
- **mobius PR #477** opened (io/metadata robustness — `_add_explicit_io_to_file`
  now reloads the on-disk graph for streamed-weight decoders instead of silently
  shipping thin metadata; re-emitted real 27B sidecar with 32 KV + 96 state_pairs)
  — **NOT merged**; never self-merge mobius (awaiting external review).
- `cosmos3_edge_text` flagged as a likely quick native-CUDA win (pure GQA decoder,
  no hybrid recurrent state; full `cosmos3_edge` VL pipeline is a larger lift needing
  vision-encoder + projector + inputs_embeds-fusion runtime coverage).

## Left open (~60) — owner-decision, not agent-actionable
- Architecture/research: memory-line / VMM / governor threads.
- Contract epics: **#489**, **#503–#539**.
- Perf research: **#579**, **#581**.
- Flagged **#490** and **#497** as real defects that are currently policy-gated.

## Notes
- Qwen3.5/3.6-27B hybrid GDN native-CUDA enablement via io-derivation is tracked
  under **#779** (already recorded in decisions.md); the `cohaagen-qwen35-27b-native`
  inbox drop was consolidated there, not duplicated.
