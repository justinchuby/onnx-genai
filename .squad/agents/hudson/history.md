## 2026-07-17 — Shape-inference axis validation

- Landed `cb30ced`: replaced clamping with checked validation for TopK and ArgReduce, Transpose, Unsqueeze, and Gather; added middle-axis, out-of-range, duplicate, and dynamic coverage.

## 2026-07-18T01:20:34Z — MTP Phase 1 remains in flight
- Sidecar metadata and Hyper-Connection adapter work continues in `wt-mtp`; not yet landed.

- 2026-07-18: MTP Phase 1 metadata/HC implementation landed after Batty restored the public MtpConfig compatibility contract; Hudson's initial revision was locked out.
- 2026-07-18T03:50:00Z: Completed MTP Phase 1 remaining-bullet audit; found Phase 1 complete, no code change needed, with `mtp_state` still Mobius-contract-blocked.


## 2026-07-21 — Wave-2 and CI milestone
CI now covers all 27 offline crates with warnings-as-errors and native Windows ARM64. Capture-safe native fp16 CUDA decode wave 2 stacked GQA prep fusion, warp-shuffle RMSNorm, and specialized down-projection GEMV on wave 1, reaching 663–672 tok/s on H200 versus ORT GenAI at 657, with zero fallbacks. All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.

- 2026-07-27T10:09:19Z: Roadmap wave landed: #239/#246/#249/#248/#256/#263/#259 plus fmt gate #264; reviewer-lockout protocol enforced where required.

## 2026-07-27T16:44:54Z — Wave 8 update
- Approved Gorman PR #274 for #53 typed `generate_image` + latent streaming; merged as 270fade1.

## 2026-07-27T19:35:00Z — Roadmap wave update
- Approved PR #292 after mutation-proving eager multi-output dispatch behavior.
