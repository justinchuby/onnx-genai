### 2026-08-26T08-53-34: Device plugins conservatively decline host value-reading shape rules
**By:** Batty
**What:** Device plugins conservatively decline host value-reading shape rules
**References:** fix/plugin-shape-device-and-dft-axis, Gaff final audit, onnx-runtime-ep-plugin, CPU/CUDA DFT
**Why:** ### 2026-08-26: Device-resident plugin shape policy
**By:** Batty
**What:** ORT plugin inputs and routed intermediates preserve allocator-derived DeviceId. Device plugins reject nodes whose output allocation requires host reading runtime tensor values (shared ConstantOfShape/Expand/STFT/Tile/DFT-with-scalars plus local Reshape/Slice/reduction/window/compress rules) before convex partitioning. No implicit D2H is performed; ORT partitions around the node and may keep device-capable producers on CUDA. DFT defaults are opset-dependent: 17/19 axis=1, 20+ axis=-2 unless explicitly overridden.
**Why:** A non-null OrtValue data pointer may be a CUDA address. Relabeling it CPU makes shape inference dereference device memory on the host (UB). Conservative claim-time decline avoids synchronization and payload transfers until a bounded metadata-transfer contract exists.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
