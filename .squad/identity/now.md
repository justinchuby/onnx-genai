# Team Focus — now

**Current focus:** Roadmap CUDA/CPU parity, scheduler coverage, model packaging, performance, and MoE core delivery. Wave 7 added CUDA `Pad`/`Range` and six shape-inference rules.

**DONE THIS SESSION:** PR #338 (`c59383db`) for #67 and PR #339 (`b1f9d3bb`) for #75 merged.

**GAP REGISTER:**
- #326: macOS x86_64 ORT archive extraction is held for Justin; Linux x86_64 and Windows AMD64 CPU wheel publishing is fixed by PR #337 (`5aed2dcf`).
- #75: Sequence/Optional shape-inference needs an SSA type-model change for container element-type propagation.
- #80/#82/#83: GLM/DeepSeek MoE core, including routed-expert paging, and Mobius PRs #404/#423/#430 are BLOCKED on Justin.

**ADVANCED / STILL OPEN:**
- #63 live GPU weight offload remains open for dispatch wiring, multi-page LRU/eviction, prefetch overlap #87, and routed-expert paging #82.
- #54 ORT model-package remains open for CLI tooling, format registry, advanced EP ranking, hashes/signatures, multi-component packages, archives, and registries.
- #67 CUDA parity remains advanced and open at **107/141 CPU parity (136 CUDA ops)**; further batches are planned.
- #86 remains advanced and open.
- #75 shape inference remains partially advanced and open at **187 registered operators** (226 versioned entries); container propagation is deferred pending the SSA type-model change.
- #55 remains open for scheduler/consumer wiring and `execution_hints.json`/YAML/builder merging.
- #73 remains open for full module gating of remaining operator groups beyond CNN/pooling/spatial.
- #80/#82/#83 GLM/DeepSeek MoE core remains advanced and open; Mobius PRs #404/#423/#430 are blocked on Justin.

**REMAINING roadmap candidates:** #82 routed-expert paging; #87 compute-transfer overlap / weight-paging prefetch; #72 Windows/macOS CI wheels; #222 graph rewriter; #231 metadata; #69 CUDA conformance profiles + GPU CI.

**BLOCKED on Justin:** Mobius #404/#423/#430 (GLM/DeepSeek MoE core E2E + Foundry-Local CUDA-vs-ORT benchmark — the core deliverable).

**ON HOLD:** #106 while Justin researches.

**Other squad open PRs, not ours to merge:** #314, #315, #317, #318, #291, #99.

**Updated:** 2026-07-28T09:10:28+00:00
