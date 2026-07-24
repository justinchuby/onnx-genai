# Capture foundation merge — 2026-07-24

`origin/main` advanced from `53f9df6` to `25dbb60`. DeepSeek MLA Attention has zero Attention-path capture fallbacks and improved captured decode from 25.87 to 27.71 tok/s. The active 2.4× fan-out remains executor shape seeding, dense f32 MatMul, movement, and MoE routing; no in-flight scope is recorded as complete.
