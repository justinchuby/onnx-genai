### 2026-07-30: Add QLinearMatMul and common Resize CUDA parity
**By:** Kuato
**What:** CUDA now claims QLinearMatMul for Int8/Uint8 per-tensor and operand-axis quantization, plus Resize nearest/linear with half_pixel, align_corners, and asymmetric coordinates using scales or sizes. Cubic, pytorch_half_pixel, tf_crop_and_resize, half_pixel_symmetric, antialiasing, and non-stretch aspect policies remain fail-closed.
**Why:** These implementations match the CPU EP's integer accumulation/requantization and interpolation formulas while raising standard-domain CUDA parity from 139/145 to 141/145 without claiming unsupported Resize semantics.
