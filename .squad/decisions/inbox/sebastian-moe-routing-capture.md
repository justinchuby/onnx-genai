# MoE routing capture safety

- Branch: `perf/capture-moe-routing`.
- TopK now folds its eagerly-read scalar K into an exact warmed signature; replay does not perform D2H or synchronize.
- GatherElements now retains shape metadata and validates capture-time indices on device through the shared capture-error word.
- Softmax skips its trailing synchronization while the EP stream is being captured; the cuDNN handle is already created on that stream.
- `indexing_gpu::warmed_moe_routing_ops_capture_without_allocations` verifies warmed TopK (K=6/64), GatherElements, and Softmax graph replay parity without allocation growth.
- Bench/ORT-vs-native-CUDA: deferred to integration because Stage-0 executor shape seeding is required to engage all decode seams.
