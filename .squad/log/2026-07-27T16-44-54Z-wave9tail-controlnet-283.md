# Wave 9 Tail — ControlNet PR #283

**Timestamp:** 2026-07-27T16:44:54Z

PR #283 closed #50 after Bishop requested changes and Batty fixed the ControlNet contract. Final landed behavior binds the single real mobius `controlnet_cond`, treats ControlNet strength as export-fused, fails loudly for multiple ControlNets, and keeps LoRA gate routing. Bishop approved the re-review, PR #283 merged as 687612f5, and #50 closed. The native image-pipeline trilogy is complete: #48 SDXL, #49 img2img/inpaint, and #50 ControlNet/LoRA.
