# GLM-4 lock and decode-coverage milestone — 2026-07-24T16:04:31Z

`13af95d7` adds the GLM-4-9B native CUDA 64-token decode golden lock and records the fused-MLP Split seam correction. Every on-box model now has a decode lock: Qwen2.5-0.5B/1.5B/7B, Qwen3-0.6B, Phi-3.5/4-mini, GLM-4-9B, and DeepSeek-R1-1.5B.

The CPU team also merged PRs #105, #108, and #109 (`d0fdfa47`, `00f12b7c`, `33ee4004`), resolving the two long-standing pre-existing CI failures: mlas-sys AVX-512 portability and the DLPack Windows deleter.
