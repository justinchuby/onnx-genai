# Team Focus — now

**Current focus:** Roadmap CUDA/CPU parity, image pipeline, scheduler coverage, and performance.

**MERGED this wave:** PR #272 closed #47 (DDPM + shifted flow-matching schedulers); PR #274 closed #53 (typed `generate_image` + latent streaming); PR #273 advanced #79 (CUDA BlockQuantizedMoE kernel).

**IN FLIGHT:** PR #276 / #87 async prefetch overlap is in Deckard's fix cycle after Ferro REQUEST-CHANGES; Keaton is locked out from revising that artifact.

**TRACKED TOOLCHAIN FOLLOW-UP:** main has pre-existing Clippy 1.97.0 lint drift at `normalization.rs:2291`; CI remains green on the pinned toolchain.

**REMAINING UNBLOCKED roadmap candidates:** #84 tree speculative decoding; #59 continuous batching; #61/#62 KV preemption and paged GQA; #48/#49/#50 image pipeline (SDXL, img2img, ControlNet); #63 GPU weight offload; #65 heterogeneous partition.

**BLOCKED:** Justin merging Mobius #404/#423/#430 (GLM/DeepSeek E2E + CUDA-vs-ORT Foundry-Local benchmark).

**ON HOLD:** #106 while Justin researches.

**Updated:** 2026-07-27T16:44:54Z
