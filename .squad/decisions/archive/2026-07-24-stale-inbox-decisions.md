# Archived decision inbox entry — 2026-07-24

> Merged during the 2026-07-24 reconciliation, then archived under the seven-day policy because its recorded decision date predates 2026-07-18.

### 2026-06-09: Lock GLM-5.2 native eager decode
**By:** Tyrell
**What:** Commit the deterministic tiny GLM-5.2 DSA-MoE fixture and test native CPU anchor IDs `[62,164,59,205,48,166,27,9,221,190,123,108]`, CUDA parity, and zero CUDA graph captures for the current concat/logical IndexShare emission.
**Why:** GLM-5.2 now decodes end-to-end through two `pkg.nxrt::IndexShare` nodes and fused QMoE. Capture capacity emission is deferred to S3, so clean eager fallback with `captures=0` is the stable pre-S3 contract.
