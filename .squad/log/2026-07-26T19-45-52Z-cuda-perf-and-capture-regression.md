# CUDA perf and capture-regression session

- **Timestamp:** 2026-07-26T19:45:52Z
- **Requested by:** justinchuby (Justin)
- **Focus:** CUDA perf next-wave (portable) plus PR #193 default-domain Attention capture-path regression.
- **Coordinator cleanup:** Fast-forwarded main to 2b2a2ba4 (#194 formatting-gate-blocking, #196 windows-sys pins). Removed four stale onnx-genai worktrees (wt-dsmla-land with merged perf/deepseek-mla-capture-copyback deleted, wt-ds-semantic, wt-qwen-regr, /tmp/rick-gap3; WIP branches kept). Confirmed GLM/DeepSeek mobius export is already covered by open mobius PRs #404, #423, and #430 awaiting Justin; no duplicate work launched.
- **Deckard:** launched on `perf/cuda-next-wave` for a profile-first portable CUDA decode-perf win; outcome pending.
- **Leon:** launched on `test/attention-default-domain-capture` for synthetic #193 capture-path regression and revert-check; outcome pending.
