# Team Focus — now

**Current focus:** Roadmap CUDA/CPU parity, scheduler coverage, and performance. Native image-pipeline trilogy is complete.

**MERGED this wave:** PR #285 closed #74 (CPU standard Conv without MLAS); PR #286 closed #61 (engine-executed KV preemption/eviction); PR #292 closed #78 (eager multi-output dispatch); PR #294 closed #58 (native f16/bf16 CPU GEMM FMA microkernel). PR #293 advanced #75 (ONNX schema/shape-inference catalog 148→164; containers deferred). PR #288 advanced #67 (CUDA EP coverage batch 4; CUDA_COVERED_OPS 118→125).

**CLOSED this wave:** #74, #61, #78, #58.

**ADVANCED this wave:** #75, #67.

**REMAINING UNBLOCKED roadmap candidates:** #59 continuous batching; #62 paged GQA KV; #63 live GPU weight offload; #65 heterogeneous CPU/CUDA partition; #60 disk-backed KV offload; #85 compute-in-place; #76 GraphView/lens EP projection; #82 routed-expert paging; #67 more CUDA op coverage; #75 remaining containers (sequence/optional/Loop/Scan).

**BLOCKED on Justin:** Mobius #404/#423/#430 (GLM/DeepSeek E2E + CUDA-vs-ORT benchmark).

**ON HOLD:** #106 while Justin researches.

**Updated:** 2026-07-27T19:35:00Z
