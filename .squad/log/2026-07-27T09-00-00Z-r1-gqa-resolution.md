# 2026-07-27T09:00:00Z — R1 GQA resolution

Mary completed the DeepSeek/GLM native-CUDA bring-up wave and resolved the R1-Distill divergence. The native GQA non-interleaved-rotary decode path is correct; ORT-CUDA's fp16 token 315 is a near-tie outlier versus native token 374. Lori-2 independently reviewed and approved PR #430, which landed test-only GQA 6:1 non-interleaved-rotary decode regressions for head_dim 64 and 128. Coordinator merged #430 and posted the correction to #384.
