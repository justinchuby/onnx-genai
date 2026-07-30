### 2026-07-30: Land the remaining tractable CUDA index and pooling operators
**By:** Kuato
**What:** Added CUDA `LpPool`, `CenterCropPad`, and `Col2Im`, raising `CUDA_COVERED_OPS` from 154 to 157. `LpPool` uses a general N-D NVRTC window reduction, while the two index transforms share one dtype-aware NVRTC module.
**Why:** All three operators have compact, model-agnostic GPU implementations and passed CPU-EP parity on GPU 3, including p=1/p=2 pooling geometry, odd mixed crop/pad, and overlapping/dilated Col2Im accumulation. This leaves the six heavier or data-dependent standard-domain gaps for focused waves.
