### 2026-08-26: Package cuDNN for current claim-then-execute behavior; track the contract bug
**By:** Gaff
**What:** CUDA packaging includes cuDNN because Conv/MaxPool/AveragePool are registered and claimed before availability is probed, then require cuDNN at execution. Issue #2198 tracks moving failure before execution. User-facing mentions of deleted ONNX_GENAI_CUDA_VMM are allowed only in explicit history/deletion statements.
**Why:** Missing cuDNN does not currently decline placement, so documentation must not claim it does. Packaging completeness must not normalize claim-then-fail semantics, and deleted flags must not remain actionable guidance.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
