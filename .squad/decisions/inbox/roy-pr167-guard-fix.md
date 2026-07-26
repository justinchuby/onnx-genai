### 2026-07-25: Gate RMSNorm SIMD scaling on exact-identity scale shape
**By:** Roy
**What:** The contiguous normalize-and-scale path now requires the right-aligned scale shape to exactly equal `x_shape[axis..]`. SkipSimplifiedLayerNormalization applies the same identity-shape check to gamma.
**Why:** Equal element counts do not prove identity indexing: for `X=[2,2]`, `axis=1`, and `scale=[2,1]`, the scale varies by group while broadcasting along the normalized axis. Such broadcasts must use the scalar `scale_index` path.
