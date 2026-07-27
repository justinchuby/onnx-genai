# Team Focus — now

**Current focus:** Roadmap CUDA/CPU parity, image pipeline, scheduler coverage, and performance.

**MERGED this wave:** PR #280 closed #48 (native SDXL dual encoders + time_ids); PR #276 advanced #87 (async prefetch overlap with Deckard fix); PR #281 closed #49 (native img2img + inpainting); PR #282 advanced #84 (tree-structured speculative decoding core).

**IN FLIGHT:** PR #283 / #50 ControlNet + LoRA is in Batty's fix cycle after Bishop REQUEST-CHANGES; Dallas is locked out from revising that artifact.

**TRACKED FOLLOW-UPS:** #84 tree live-decode wiring is blocked on model-graph 2D-mask change (`decode/step.rs:138` builds only a 1D key mask). Main also has pre-existing Clippy 1.97.0 lint drift at `normalization.rs` and `fused_epilogue_gpu.rs`; CI remains green on the pinned toolchain.

**REMAINING UNBLOCKED roadmap candidates:** #59 continuous batching; #61/#62 KV preemption and paged GQA; #63 GPU weight offload; #65 heterogeneous partition.

**BLOCKED:** Justin merging Mobius #404/#423/#430 (GLM/DeepSeek E2E + CUDA-vs-ORT Foundry-Local benchmark — the core deliverable).

**ON HOLD:** #106 while Justin researches.

**Updated:** 2026-07-27T16:44:54Z
