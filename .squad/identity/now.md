# Team Focus — now

**Current focus:** Roadmap CUDA/CPU parity, scheduler coverage, and performance. Native image-pipeline trilogy is complete.

**MERGED this wave:** PR #280 closed #48 (native SDXL dual encoders + time_ids); PR #281 closed #49 (native img2img + inpainting); PR #283 closed #50 (native ControlNet + LoRA); PR #276 advanced #87 (async prefetch overlap with Deckard fix); PR #282 advanced #84 (tree-structured speculative decoding core).

**COMPLETE:** Native image-pipeline trilogy #48/#49/#50 is complete: SDXL, img2img/inpainting, and ControlNet/LoRA are all closed. #50 landed via PR #283 as `687612f5` after Batty's fix cycle and Bishop re-review.

**TRACKED FOLLOW-UPS:** #84 tree live-decode wiring is blocked on model-graph 2D-mask change (`decode/step.rs:138` builds only a 1D key mask). Main also has pre-existing Clippy 1.97.0 lint drift at `normalization.rs` and `fused_epilogue_gpu.rs`; CI remains green on the pinned toolchain. `decisions.md` is still about 750KB after the 7-day archive gate and is worth a future archive/prune pass.

**REMAINING UNBLOCKED roadmap candidates:** #59 continuous batching; #61/#62 KV preemption and paged GQA; #63 GPU weight offload; #65 heterogeneous partition.

**BLOCKED:** Justin merging Mobius #404/#423/#430 (GLM/DeepSeek E2E + CUDA-vs-ORT Foundry-Local benchmark — the core deliverable).

**ON HOLD:** #106 while Justin researches.

**Updated:** 2026-07-27T16:44:54Z
