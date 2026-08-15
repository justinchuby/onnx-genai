# Lori — History

## 2026-07-29T03:45:00+0000 — PR #382 review

- Requested changes because CUDA-only E2E auto-skipped without CUDA and did not lock the #380 declared-KV threading regression.
- Assigned the repair to Leon under Benny's lockout, independently revert-verified the resulting CPU test, and approved the merged `85b9ba15`.
## 2026-07-30T01:30:00Z — PR #419 CUDA parity review

- Reviewed Kuato's PR #419 CUDA `LpPool`, `CenterCropPad`, and `Col2Im` wave.
- Approved after 217 GPU parity cases covered the new kernels; PR #419 merged as `9eeca36c`.

## 2026-07-30T04:10:00Z — CUDA parity 161 review gate

- Approved Kuato's PR #423 after QLinearMatMul and Resize GPU parity evidence.
- Requested changes on PR #424, locking Kuato out of the revision; then approved Mary's shape-aware claim-gate fix after independent on-device re-run covering 308 GPU parity cases plus claim-gate probe evidence.

## 2026-07-30T09:16:00Z — 27B blocker-chain reviews

- Approved Mary’s rank-3 Conv1D PR #438, now merged; continuing review of Silu inference PR #440.

## 2026-07-30T13:36:00Z — PR #446 offload/capture correctness review

- Independent review of Cohaagen's PR #446 mutual exclusion constraints between weight offload and graph capture; approved and merged.
- Critical correctness gate for native CUDA live weight-offload path (#63).

## 2026-07-31T00:25:00Z — PR #533 native pipeline Inc3c review (native CUDA decode beats ORT)

- Approved Mary's PR #533 (Inc3c): native CUDA decode flips to 1.38x ORT WIN on real qwen3-0.6b via default-off captured step-input binding. Verified byte-identical tokens to eager, real ORT-CUDA baseline (not CPU-fallback), zero regressions, and non-tautological engagement counter (OFF=0/ON=3). Landmark real-model validation.
