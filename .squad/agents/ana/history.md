# ana — History

## 2026-07-15T01:52:00Z — Session update

- Hardened DLPack export (`e38eaee`) with unsafe constructor boundaries, validations, zero-size null data, endian guard, and aliasing docs.

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.


## 2026-07-21 — Wave-2 and CI milestone
CI now covers all 27 offline crates with warnings-as-errors and native Windows ARM64. Capture-safe native fp16 CUDA decode wave 2 stacked GQA prep fusion, warp-shuffle RMSNorm, and specialized down-projection GEMV on wave 1, reaching 663–672 tok/s on H200 versus ORT GenAI at 657, with zero fallbacks. All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.

## 2026-07-21T13:15:00Z — Wave-4 roofline validation
- Re-rooflined the 691 tok/s stack: MatMulNBits was 38.5% of wall time at ~323 launches/token, with a projected 750–790 ceiling. Wave 4 reached ~759 tok/s at 256 and ~789 at 1024, validating the artifact as the current roofline of record.
