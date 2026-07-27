# Eager multi-output dispatch (#78)

Eager dispatch now has `dispatch_with_outputs(..., output_count, ...)`, which models ONNX's explicit output-slot list and returns materialized leading slots in ONNX order. A one-output `dispatch` wrapper remains for compatibility. Trailing optional slots are omitted by requesting fewer outputs; invalid zero, unsupported extra, and required-output omissions return errors rather than panicking.

The cache key now includes output count and canonicalized attributes because both affect compiled kernel behavior. CPU eager input shape-data is propagated for host scalar/vector control tensors, allowing TopK's runtime `K` and Split's sizes to produce concrete allocation shapes. PyO3's existing feature-gated `nxrt.eager.dispatch` now accepts `outputs=` and exposes all returned tensors.
