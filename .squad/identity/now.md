# Team Focus — now

**Current focus:** Roadmap CUDA/CPU parity, scheduler coverage, model packaging, and performance. Native image-pipeline trilogy is complete.

**MERGED this wave:** PR #312 closed #65 (heterogeneous CPU/CUDA partition); PR #308 closed #60 (disk-backed KV offload); PR #311 advanced #67 (CUDA op coverage batch 5); PR #309 advanced #86 (varlen packed attention); PR #313 landed decode-garble triage/prevention guard; PR #316 fixed #289 (CJK/wide-char renderer width).

**CLOSED this wave:** #65, #60.

**ADVANCED / STILL OPEN:** #67 remains open for CUDA op coverage batch 6; #86 remains open for exporter/CUDA/f16 deferred work.

**REGRESSION RESOLUTION:** Decode was fine: real fused decode graphs fire compute-in-place aliasing #301 zero times, native==ORT byte-identical, and repeated sentences are natural greedy output. #289 was a CLI renderer bug: `live_turn.rs` used `chars().count()` rather than Unicode display width, causing CJK wrapping/spacing errors; fixed in PR #316.

**REMAINING UNBLOCKED roadmap candidates:** #63 live GPU weight offload; #82 routed-expert paging; #54 ORT model-package MVP; #55 model metadata hints; #72 Windows/macOS CI wheels; #73 minimal operator builds; #67 batch 6; #307 perf-test continuous batching; #299 LoRA loading; #222 graph rewriter; #231 metadata.

**BLOCKED on Justin:** Mobius #404/#423/#430 (GLM/DeepSeek E2E + Foundry-Local benchmark).

**ON HOLD:** #106 while Justin researches.

**Updated:** 2026-07-28T04-08-08+0000
