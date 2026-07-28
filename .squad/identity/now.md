# Team Focus — now

**Current focus:** Roadmap CUDA/CPU parity, scheduler coverage, model packaging, and performance. Wave 4 merged the metadata-hints and minimal-operator-build slices.

**DONE THIS SESSION:** #307, #65, #60, and #289.

**MERGED / PARTIALLY COMPLETE THIS SESSION:** PR #329 (`7876a7ad`) advanced #55 with the typed `onnx_runtime.*` metadata-hints subsystem; PR #327 (`64d99919`) advanced #73 with representative CPU `ops-cnn` feature gating, shared operator catalog, and deterministic minimal-build manifests.

**KNOWN GAP:** `optimizer.rs:82` gates the NCHWC pass on `mlas` only; it must use `all(mlas, ops-cnn)`. This is non-shipping today because full builds pair the features and `mlas` is off by default. Fix it while gating the remaining #73 operator groups.

**ADVANCED / STILL OPEN:**
- #63 live GPU weight offload remains open for dispatch wiring, multi-page LRU/eviction, prefetch overlap #87, and routed-expert paging #82.
- #54 ORT model-package remains open for CLI tooling, format registry, advanced EP ranking, hashes/signatures, multi-component packages, archives, and registries.
- #67 CUDA parity remains open.
- #86 remains advanced and open.
- #55 remains open for scheduler/consumer wiring and `execution_hints.json`/YAML/builder merging.
- #73 remains open for full module gating of remaining operator groups beyond CNN/pooling/spatial.

**REMAINING roadmap candidates:** #82 routed-expert paging; #87 compute-transfer overlap / weight-paging prefetch; #72 Windows/macOS CI wheels; #222 graph rewriter; #231 metadata; #69 CUDA conformance profiles + GPU CI; #75 ONNX schema/shape-inference catalog.

**BLOCKED on Justin:** Mobius #404/#423/#430 (GLM/DeepSeek E2E + Foundry-Local CUDA-vs-ORT benchmark — the core deliverable).

**ON HOLD:** #106 while Justin researches.

**Other squad open PRs, not ours to merge:** #314, #315, #317, #318, #291, #99.

**Updated:** 2026-07-28T06-44-16+0000
