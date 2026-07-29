# Team Focus — now

**Current focus:** CUDA/CPU parity, streaming tool calls, and the next warmup/attention wave.

**MERGED THIS WAVE:** PR #400 (`261fa8f3`) completed issue #183 streaming tool-call SSE deltas; PR #402 (`48582a75`) added the selected CUDA operator-parity batch and raised `CUDA_COVERED_OPS` to 152. Issues #13 (debug endpoints + Perfetto) and #72 (multi-platform CI) are closed.

**OPEN / PARTIAL:**
- #183 still needs a real-model E2E test, a grammar-masking benchmark, and f32-accumulation verification.
- #67 still needs ConvTranspose, GridSample, Resize, NonMaxSuppression, Col2Im, CenterCropPad, InstanceNormalization, GroupNormalization, DFT plus windows, SpaceToDepth, and com.microsoft FusedAttention.
- #9 warmup and #86 CPU PackedVarlenAttention are the next wave.

**BLOCKED GAPS:**
- #75 and #355's container family are gated on an IR SSA `Value`/`TypeInfo` container-element-type extension and require Justin's sign-off.

**Updated:** 2026-07-29T13:54:51+0000
