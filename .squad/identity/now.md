# Team Focus — now

**Current focus:** CUDA op-parity at 157 ops + native large-model reduce-fallback cleared (#420); large-model native decode now blocked only on explicit metadata #377/mobius#434; ORT-CUDA 27B baseline 17.38 tok/s established.

**MERGED THIS WAVE:** PR #419 (`9eeca36c`) added CUDA `LpPool`, `CenterCropPad`, and `Col2Im`, raising `CUDA_COVERED_OPS` from 154 to 157 after GPU parity review. PR #420 (`6610f86f`) widened extended reductions to f16/bf16 with f32 accumulation, clearing the native fallback for 96 FP16 `ReduceSumSquare` nodes. PR #418 was also merged for SiLU-marker hardening plus CUDA InstanceNormalization and GroupNormalization.

**LARGE-MODEL STATUS:** Mary root-caused the 27B/35B native E2E blockers and PR #420 cleared the native reduce-fallback. The remaining native 27B decode blocker is explicit metadata emission/consumption for #377 via `onnxruntime/mobius#434`, pending Justin-side Mobius merge.

**ORT-CUDA BASELINE:** Doug established an ORT-CUDA 1.28 basic-optimization Qwen3.6-27B INT4 reference at 17.38 tok/s, 57.527 ms/token, and 18,127 MiB peak H200 VRAM. ORT extended/all graph optimization still aborts in upstream CUDA Level2 optimizer behavior; the basic-opt number is a workaround reference, not the project-default ORT-all path.

**OPEN / PARTIAL:** #67 is at 157 CUDA-covered ops after #419. #384 now has an ORT-CUDA 27B baseline and native status update. #377 native-side work remains dependent on Mobius explicit metadata (#434) and the off-limits #99 speculative-decoding naming-convention cleanup remains out of this workstream.

**OFF-LIMITS:** #54 model-package and #299 LoRA belong to another team; #106 is under Justin's study. Do not touch resting other-squad open PRs (#314, #315, #317, #318, #291, #99).

**Updated:** 2026-07-30T01:30:00Z
