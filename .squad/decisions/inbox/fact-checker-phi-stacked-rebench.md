### 2026-07-23: Fact-check Phi stacked prefetch + int8 split-K re-benchmark
**By:** Fact Checker
**What:** ⚠️ Overall verification is conditional. Arithmetic, caveats, no-regression framing, and the split-K plausibility check are verified. The Phi median and range agree in the benchmark document, `PROGRESS.md`, and Marsten's note. However, the Qwen2.5-1.5B (617.90 tok/s) and R1-Distill (622.66 tok/s) guard values appear in the benchmark document and Marsten's decision note but are absent from `PROGRESS.md`, so cross-file consistency for every requested number is unverified.
**Why:**
- ✅ **Arithmetic:** `(193.32 - 229.62) / 229.62 * 100 = -15.808727...%`, which rounds to **-15.81%**. The only other newly added percentage claim, `(193.32 - 188.54) / 188.54 * 100 = +2.535271...%`, rounds to **+2.54%**. No arithmetic discrepancy was found in the added benchmark-document percentage deltas.
- ⚠️ **Internal consistency:** 193.32 (seven-run/median-of-7) and 121.21--194.67 agree across all three artifacts. 617.90 and 622.66 agree wherever stated (benchmark document and Marsten note), but `PROGRESS.md` does not state either value.
- ✅ **Caveats:** The benchmark document identifies the final low Phi sample as overlapping host contention, says the host is shared with CPU benchmark jobs, and expressly labels the wide Phi range contention variance rather than a kernel regression. It also says 229.62 is a retained documented canonical ORT reference, not a comparable fresh measurement.
- ✅ **Regression framing:** Qwen 617.90 versus ~622 (-0.66%) and R1-Distill 622.66 versus ~622 (+0.11%) are explicitly called within noise, not improvements or regressions.
- ✅ **Plausibility:** 193.32 is +2.54% over the prefetch-only 188.54. A +2.1% lever would predict about 192.50; the observed +0.44 percentage-point difference is directionally consistent and plausible given the disclosed host contention and 121.21--194.67 spread.

**Merge recommendation:** ⚠️ Hold merge until `PROGRESS.md` includes the two guard values (or the cross-file-consistency claim is narrowed to the Phi values).
