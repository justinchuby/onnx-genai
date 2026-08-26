# PR #2107 graph-lifetime pinning

- CUDA graph segments own type-erased strong resource pins in their existing
  lifecycle state. Provisional pins transfer on successful instantiation and
  unwind on abort/failure.
- Sealed planar matmul/MoE bank owners use weak runtime/release-queue links to
  avoid a runtime → graph → bank → runtime cycle. Final release remains
  generation-checked and stream-ordered.
- Planar captured launches reject when the stream has no runtime-owned graph
  resource sink, and all launch buffers must match the admission's provider
  context as well as its CUDA device.
