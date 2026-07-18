## 2026-07-17 — Shape-inference axis validation

- Landed `cb30ced`: replaced clamping with checked validation for TopK and ArgReduce, Transpose, Unsqueeze, and Gather; added middle-axis, out-of-range, duplicate, and dynamic coverage.

## 2026-07-18T01:20:34Z — MTP Phase 1 remains in flight
- Sidecar metadata and Hyper-Connection adapter work continues in `wt-mtp`; not yet landed.

- 2026-07-18: MTP Phase 1 metadata/HC implementation landed after Batty restored the public MtpConfig compatibility contract; Hudson's initial revision was locked out.
- 2026-07-18T03:50:00Z: Completed MTP Phase 1 remaining-bullet audit; found Phase 1 complete, no code change needed, with `mtp_state` still Mobius-contract-blocked.
