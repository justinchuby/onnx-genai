# kuato — History

## 2026-07-28T21:15:00+0000 — PR #378 review
- Independently reviewed Nandez's flaky route-first QMoE residency-test fixes in PR #378 and approved it; merged as `ac75e146`.
## 2026-07-30T01:30:00Z — CUDA parity wave and reduce review

- Authored PR #419 for CUDA `LpPool`, `CenterCropPad`, and `Col2Im`, raising CUDA coverage from 154 to 157; merged as `9eeca36c`.
- Independently reviewed Mary's PR #420 extended-reduction CUDA work and approved after the FP16 `ReduceSumSquare` native fallback was cleared.

## 2026-07-30T04:10:00Z — CUDA parity 161 authorship

- Authored PR #423 (`QLinearMatMul` + common `Resize`), raising CUDA coverage from 157 to 159; merged as `eed2fbf2` after Lori approval.
- Authored PR #424 (`ConvTranspose` + `GridSample`), raising CUDA coverage from 159 to 161; original author was locked out after Lori requested changes, and Mary owned the shape-aware claim-gate revision before merge as `1574e87a`.
