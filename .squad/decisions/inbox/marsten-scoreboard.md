### 2026-07-23: Native CUDA versus ORT real-weight baseline
**By:** Marsten
**What:** On `origin/main` revision `1073404`, native CUDA beat ORT GenAI CUDA
for all runnable dense Qwen exports: Qwen2.5-0.5B (+62.73%), 1.5B (+36.77%),
and 7B (+10.82%). Phi-4-mini remains behind: the standing clean mandate
reference is 193.89 versus 229.62 tok/s (-15.56%); this live nine-run snapshot
was 186.19 versus 236.48 tok/s (-21.27%).
**Why:** This records the real-weight baseline before Deckard's Phi
`executor.rs` capture-seam work. GPU 5 was idle before/after testing, but the
shared host produced a wide Phi range, so reserved-host confirmation is needed
before treating the live shortfall versus the clean reference as a regression.
