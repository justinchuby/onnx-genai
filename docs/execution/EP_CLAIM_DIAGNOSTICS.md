# EP claim diagnostics

Execution providers return a `KernelMatch` from `supports_op`. A declined claim
is represented as:

```rust
KernelMatch::Unsupported {
    reason: Cow<'static, str>,
}
```

The reason is part of the decision, not a separate lookup table. This
colocation is an invariant: changing a claim predicate must change its
actionable diagnostic in the same branch, so the two cannot drift.

## Authoring declines

EP crates can use the exported macros:

```rust
require!(rank == 2, "MatMul: CUDA requires rank-2 inputs, got rank {rank}");
deny!(
    "no CUDA kernel for {domain}::{op} — add a CUDA kernel + register it, \
     or enable CPU/ORT fallback"
);
```

Formatting occurs only when the EP declines. The supported hot path does not
allocate a reason string. Reasons should say what is accepted and, where
possible, how to fix the model, attributes, shapes, or kernel registration.

## Session reporting

The session consumes the reason directly from `KernelMatch::Unsupported`.
CUDA-only coverage reports attach it to the node identity, and dynamic kernel
cache validation includes it in `SessionError::UnsupportedOp`.

Before:

```text
graph/node#42 Foo: unsupported by cuda_ep
```

After:

```text
graph/node#42 Foo: no CUDA kernel for ai.onnx::Foo — op not in the CUDA registry
(add a CUDA kernel + register it, or enable CPU/ORT fallback)
```

Attribute failures are similarly specific:

```text
BlockQuantizedMatMul: CUDA does not support format 'q4_0' — re-export weights
as mxfp4, iq4_nl, iq4_xs, ...
```

This design is a Rust adaptation of the claim-result diagnostics pattern in
the sibling `justinchuby/onnxruntime-mlx` project.
