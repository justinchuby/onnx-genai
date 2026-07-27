### 2026-07-27: Prioritize kernel-backed tensor schemas before container types
**By:** Burke
**What:** Expanded the standard catalog and shape registry with 16 CPU/CUDA-kernel-backed tensor operators, including their active opset boundaries, while deferring sequence/optional and Loop/Scan inference.
**Why:** These operators were executable but could leave outputs unresolved. The current inference `TypeInfo` represents tensors only, so correct sequence/optional inference requires a container-aware type model rather than pretending containers are tensors.
