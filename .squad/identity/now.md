# Team Focus — now

**Current focus:** CUDA op-parity is at 161 covered standard-domain ops after #423/#424. Native large-model 27B decode remains gated on explicit token metadata (#377 / Möbius #434, awaiting Justin merge). ORT-CUDA 27B basic-opt baseline is established at 17.38 tok/s / 18.1 GiB.

**MERGED THIS WAVE:** PR #423 (`eed2fbf2`) added CUDA `QLinearMatMul` and common nearest/linear `Resize`, raising coverage 157→159. PR #424 (`1574e87a`, Mary revision `93d9e7b8`) added CUDA `ConvTranspose` and `GridSample`, raising coverage 159→161. PR #420 (`6610f86f`) widened extended reductions to f16/bf16 with f32 accumulation, clearing the native reduce fallback for Qwen3.6-27B INT4.

**GAP REGISTER:** #67 remaining heavy CUDA gaps are `NonMaxSuppression` and Resize cubic/advanced modes. Native 27B remains blocked on explicit token metadata (#377 / Möbius #434, awaiting Justin merge). MoE 35B still needs `BlockQuantizedMoE` support (#82). ORT-CUDA extended/all graph optimization still aborts on the 27B artifact in upstream ORT CUDA Level2 behavior; basic-opt is only a workaround reference.

**ORT-CUDA BASELINE:** Doug established ORT-CUDA 1.28 basic-optimization Qwen3.6-27B INT4 at 17.38 tok/s, 57.527 ms/token, and 18,127 MiB peak H200 VRAM. ORT 1.27 and 1.28 both abort with extended/all graph optimization.

**OFF-LIMITS:** #54 model-package and #299 LoRA belong to another team; #106 is under Justin's study. Do not touch resting other-squad open PRs (#314, #315, #317, #318, #291, #99).

**Updated:** 2026-07-30T04:10:00Z
