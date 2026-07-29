### 2026-07-29: CUDA operator parity batch 10
**By:** Ernie
**What:** Added CUDA kernels and CPU-parity coverage for AffineGrid, BatchNormalization, Compress, DynamicQuantizeLinear, GlobalAveragePool, GlobalLpPool, GlobalMaxPool, and LpNormalization. Deferred CenterCropPad, Col2Im, ConvTranspose, GridSample, GroupNormalization, InstanceNormalization, LpPool, NonMaxSuppression, QLinearMatMul, Resize, Unique, and com.microsoft FusedAttention.
**Why:** The selected operators form a reviewable low-risk batch around fixed-width transforms, channel-wise normalization, and block reductions. Heavy geometry, convolution, detection, and data-dependent operators need dedicated follow-up waves.
