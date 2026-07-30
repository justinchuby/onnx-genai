### 2026-07-30: Add common ConvTranspose and GridSample CUDA geometry paths
**By:** Kuato
**What:** CUDA now covers 1-D/2-D ConvTranspose with explicit/VALID padding, strides, dilation, output padding, groups/depthwise, and optional bias, plus 4-D GridSample bilinear/nearest with zeros/border/reflection padding and both align_corners values. SAME auto-padding, output_shape-driven ConvTranspose geometry, cubic GridSample, and volumetric GridSample remain fail-closed.
**Why:** Output-owned NVRTC kernels match the CPU EP formulas without atomic accumulation nondeterminism, raising advertised CUDA coverage from 159 to 161 while refusing geometry modes not validated in this wave.
