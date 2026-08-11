### 2026-08-07: PR #728 round-6 final re-review
**By:** Harry
**Verdict:** REJECT

## Blocking finding

1. **The classifier is still bypassed by three shape-dependent CUDA pointwise kernels.** The revised `capture_shape_eligible` is now a hard veto for every caller, and the signature-gated unary/binary/predicate/bitwise/PRelu/SiluMul paths correctly return `Unsupported` when `capture_seq_independent == false`. However, `UnaryMathKernel` (`Abs`, `Neg`, `Exp`, `Sign`, etc.) and `NotKernel` still return `CaptureSupport::Supported` unconditionally and do not store or override `set_capture_seq_independent` (`crates/onnx-runtime-ep-cuda/src/kernels/pointwise.rs:304-387,418-490`). `BitwiseNotKernel` has the same bypass (`crates/onnx-runtime-ep-cuda/src/kernels/bitwise.rs:470-564`). A classifier-disqualified growing rank-1 value consumed by any of these kernels can therefore still enter capture with a launch grid/count baked from the warmed extent.

   This directly fails the requested “ANY remaining path” check. Wire these kernels through the same authoritative flag/signature gate (or enforce the veto centrally before kernel-specific `capture_support`).

2. **The claimed capture-level regression does not exercise `UnaryKernel::run` or `capture_support()`.** `disqualified_growing_reshape_consumer_yields_no_capture_signature` manually calls `capture_shape_eligible` and constructs a `Vec` (`elementwise.rs:1231-1248`); it is a helper test, not an EP/capture-admission test, and cannot catch the bypasses above. Its fail-pre claim is genuine—the old helper admitted `[320]`—but an actual kernel-path regression remains necessary.

## Confirmed

- `capture_shape_eligible(false, shape)` is now always false; all six existing callers are centrally fixed.
- Their struct defaults are fail-safe, and `KernelCache::get_or_create` sets the classifier flag before insertion/return (`kernel_cache.rs:602-631`).
- `is_fixed_decode_shape` has no remaining code references.
- `cargo test -p onnx-runtime-ep-cuda --lib --quiet`: 280 passed, 2 ignored.
- `git diff --check 0fd87df3..f6b6203c`: clean.

**Revision owner:** Sapper. Roy, Cohaagen, Deckard, Leon, Batty, and Sebastian remain locked out.

**REJECT: classifier-disqualified nodes can still reach capture through unconditional `Supported` paths in `UnaryMathKernel`, `NotKernel`, and `BitwiseNotKernel`.**
