# Team Focus — now

**Current focus:** Roadmap CUDA/CPU parity, scheduler coverage, model packaging, and performance. Wave 3 merged three roadmap PRs.

**MERGED this wave:** PR #320 closed #307 (continuous-batching throughput benchmark); PR #321 advanced #63 (live GPU weight offload Phase-3b device-binding slice); PR #322 advanced #54 (ORT model-package MVP with security hardening).

**CLOSED this wave:** #307.

**ADVANCED / STILL OPEN:**
- #63 live GPU weight offload remains open for dispatch wiring, multi-page LRU/eviction, prefetch overlap #87, and routed-expert paging #82.
- #54 ORT model-package remains open for CLI tooling, format registry, advanced EP ranking, hashes/signatures, multi-component packages, archives, and registries.

**REMAINING roadmap candidates:** #82 routed-expert paging; #87 compute-transfer overlap / weight-paging prefetch; #55 model metadata hints; #72 Windows/macOS CI wheels; #73 minimal operator builds; #67 batch 6; #222 graph rewriter; #231 metadata; #69 CUDA conformance profiles + GPU CI; #75 ONNX schema/shape-inference catalog.

**BLOCKED on Justin:** Mobius #404/#423/#430 (GLM/DeepSeek E2E + Foundry-Local CUDA-vs-ORT benchmark — the core deliverable).

**ON HOLD:** #106 while Justin researches.

**Other squad open PRs, not ours to merge:** #314, #315, #317, #318, #291, #99.

**Updated:** 2026-07-28T05-49-08+0000
