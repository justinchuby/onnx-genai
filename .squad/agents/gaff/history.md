# Gaff — History

## Project context
- Review specialist for onnx-genai correctness, runtime/loader boundaries, transactional semantics, and validation quality.
- Joined 2026-07-12 after phases 1-4, tool use/grammar/chat-template, Qwen2.5-0.5B, Hermes E2E, and static-cache KV work were established.

## Condensed prior record through 2026-07-28
- Reviewed and approved multiple ORT2 loader, shape-inference, fused-domain, EPContext, C-ABI, external-data, and conformance changes after checking byte fidelity, dispatch invariants, path confinement, FFI behavior, and model-backed tests.
- Used reviewer lockout discipline on real blockers: unauthenticated debug exposure, MatMul+Add fusion shape guard gaps, duplicate EPContext primary identity over-rejection, unsupported-op user-facing opset leakage, and thread/benchmark provenance issues.
- Helped consolidate performance and CUDA/native guidance: benchmark comparisons must be matched and reproducible; CUDA EP kernel work must remain correct across supported SM architectures rather than only sm_90.
- Recorded CLI maintainer-tool backlog context: the CLI is a development/maintainer harness, not a consumer product; `docs/research/cli/00-backlog.md` is the backlog source of truth.
- For PR #289, rejected REPL Phase 1 when non-TTY parsing drifted from the plain path, then approved Zhora's revision after parser parity and newline lifecycle fixes.
- For PR #291, owned the runner-backed rewind boundary after Leon's deeper rejection; runner-backed static-cache/shared-buffer rewind now rejects during validation until a transactional prepared rewind exists.
- Authored PR #321 for issue #63, landing Phase-3b live GPU weight-offload allocation, H2D copy, binding, and byte-identical one-weight device execution.
- Recent consolidated decisions remain authoritative in `.squad/decisions.md`; this history was summarized by Scribe because it exceeded the 15KB threshold.

## 2026-07-28T17:40:00+0000
Reviewed #364: blocked the partial-cache stat divergence, then approved the guarded fix.
