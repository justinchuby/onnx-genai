# Project Rules

These durable rules bind every human contributor and every AI coding agent working on nxrt / onnx-genai.

## 1. Error Messages & Debug Experience

**Every failure must help humans and AI agents understand and fix the problem quickly, accurately, and confidently. Users should feel warmly cared for—暖暖的被捧在手心—not abandoned in “a stack trace from hell.”**

- Every user-facing error answers **what failed, why it failed, and how to fix it**. Prefer “expected X, got Y,” “available options are …,” and “did you mean …?” over terse labels or leaked internals.
- Preserve the most useful context at every layer: op type and domain, node and input name, input shapes and dtype, opset, device/EP, model or data file path, configuration key, and the rejected value when relevant.
- Rust error enums (`EagerError`, loader/session/EP errors, and similar types) must have informative displays, not just variant names. Add `anyhow::Context` / `with_context` at boundaries so the causal chain says what operation and resource were involved.
- The C ABI (`nxrt_*`) must return a machine-parseable error code **and** a retrievable, rich human-readable message. Never discard the Rust cause or its actionable context while mapping it across FFI; never unwind across the boundary.
- Planned PyO3 bindings must translate the same context into appropriate Python exceptions, with clean tracebacks and structured, AI-agent-friendly details where practical. Do not replace a precise runtime error with a generic `RuntimeError`.
- CLI and server diagnostics must fail fast and identify the bad argument, request field, file, model component, or configuration, then give a concrete next step. Do not emit opaque “failed,” “internal error,” or bare status-code responses when more is known.
- Avoid panics on user-facing paths whenever a descriptive `Result` is possible. Panic-fence FFI boundaries, fail closed on invalid input, and distinguish user/configuration errors from internal invariant failures.
- Reviewers **must report weak, opaque, context-losing, or unactionable errors as real findings**, not style nitpicks.

See [`docs/ORT2.md` §35](docs/ORT2.md#35-error-recovery--debug-experience), especially its what/why/how-to-fix contract and examples; also §26 for FFI failure handling.

## 2. Stay model-, vendor-, and EP-agnostic

**Library behavior is driven by model metadata, ONNX semantics, registries, and explicit configuration—not hardcoded model, vendor, or execution-provider names.**

- Kernels are shape-driven, dtype-parameterized, and architecture-gated; model dimensions and attention parameters are runtime data.
- Generic loader, IR, session, optimizer, and dispatch code must not special-case model families, op attribute names, vendor strings, or EP names.
- EP selection uses declared capabilities and registry/config keys. An unavailable match produces a clear error rather than a guess or silent fallback.
- **No hardcoded model architecture, anywhere.** Neither inference metadata nor any runtime implementation may bake in a specific model's architecture (layer counts, hidden/intermediate sizes, head counts, exact tensor shapes, magic dimension constants, etc.). Architecture is runtime data derived from the model and its metadata.
- **All assumptions are declared explicitly as metadata.** If a code path depends on an architectural property (shared KV layout, RoPE variant, block-quant size, attention scheme, sliding-window, and so on), that property must be surfaced as explicit, inspectable metadata—never inferred from a model name or silently assumed. Missing metadata fails clearly.

### 2.1 Graph fusion must be generic and EP-internal

**Fusion detects structural patterns, never model identity, and lives inside the EP.**

- Fusion decisions are driven by **op/topology patterns** (e.g. `Add(MatMulNBits, [N]-bias)`, paired gate/up `MatMulNBits` feeding `Mul(Silu(gate), up)`)—never by model name, vendor, or any model-specific hint.
- A fusion must generalize across every model that exhibits the pattern. Optimize **per pattern category**, not for a single model. Correctness is guarded by dtype/shape **compatibility** checks (divisibility, supported block size, supported dtype), **not** by hardcoded magic dimensions (e.g. a specific `K`/`N`). Hardcoded shape constants that only match one model are a review-blocking finding.
- Fusion happens **inside the EP** (as part of its claim/compile), not in generic graph code. The EP may use the IR crate to perform it.
- The IR crate is the sanctioned home for a reusable **pattern-matcher + rewriter**; building that infrastructure there and calling it from EPs is approved and preferred over ad-hoc per-EP string/shape matching.

See [`docs/ORT2.md` §15.1](docs/ORT2.md#151-decision-summary), §55.6, [`docs/MODEL_METADATA.md`](docs/MODEL_METADATA.md), and [`docs/PROGRESS.md`](docs/PROGRESS.md).

## 3. Make pre-release changes cleanly

**Do not add backward-compatibility aliases, deprecation layers, or migration shims for our own pre-release APIs.**

- Rename, remove, or reshape an API completely; update all callers, docs, fixtures, and tests in the same change.
- Do not retain duplicate old symbols “just in case.”
- This does **not** waive compatibility that is itself a product requirement, such as supported ONNX opsets or the documented ORT/plugin ABI surface.

See [`docs/PROGRESS.md`](docs/PROGRESS.md) and [`docs/CRATE_RESERVATION.md`](docs/CRATE_RESERVATION.md).

## 4. Do not rewrite what already works

**Reuse battle-tested libraries for established primitives; write custom kernels only where they provide a measured, necessary advantage.**

- CPU uses the built-in SIMD backend for optimized production paths.
- CUDA uses cuBLAS/cuBLASLt and cuDNN for vendor-optimized paths; CuTe/CUTLASS is for custom fusions those libraries do not provide.
- Profile before replacing a proven implementation. Keep thin seams so reference and optimized implementations remain testable.

See [`docs/ORT2.md` §1](docs/ORT2.md#1-design-principles) and §15.

## 5. Prefer explicit, inspectable behavior

**Debuggability and predictability beat cleverness, silent convenience, and hidden heuristics.**

- Optimization and placement decisions flow through an explicit, inspectable cost model.
- Eager execution never performs implicit cross-device transfers; users request `.to(device)` explicitly.
- Opset choice is explicit and non-surprising. Unsupported kernels, dtypes, attributes, or configurations fail clearly rather than silently changing semantics.

See [`docs/ORT2.md` §1](docs/ORT2.md#1-design-principles), §6, and [`docs/EAGER.md` §1](docs/EAGER.md#1-design-principles) / §13.

## 6. Use the canonical names

**Public names are consistent across each ecosystem.**

- Product, CLI, and planned Python package: `nxrt`.
- C ABI symbols: `nxrt_*`.
- Runtime Rust crates: `onnx-runtime-*`.
- GenAI-stack Rust crates: `onnx-genai-*`.
- Do not reintroduce legacy `ort2_*` public symbols or rename the retained design file `docs/ORT2.md`.

See [`docs/PROGRESS.md`](docs/PROGRESS.md) and [`docs/CRATE_RESERVATION.md`](docs/CRATE_RESERVATION.md).

## 7. Ship stable-ABI Python wheels

**The planned PyO3 bindings support Python 3.10+ while minimizing per-version wheel builds.**

- Standard CPython wheels use `abi3` with a Python 3.10 compatibility floor (`abi3-py310`); the wheel target is py312.
- Free-threaded wheels use `abi3t`; the target is py315.
- Keep standard and free-threaded wheel configurations separate and test both surfaces.

See the recorded Python ABI decision in `.squad/decisions/inbox/coordinator-python-abi3.md`.

## 8. Tests track behavior and APIs

**Behavioral and public-API changes include their tests in the same commit.**

- Add focused regression coverage for the changed success and failure paths, including error content when actionability is part of the contract.
- Update fixtures, expected counts, snapshots, conformance checks, and documentation examples when their underlying API changes.
- Run the smallest relevant test/lint set before landing; do not knowingly leave CI cleanup to the next contributor.

See [`docs/PROGRESS.md`](docs/PROGRESS.md) for the project’s test, conformance, clippy, Miri, and audit expectations.

## 9. Keep history linear and review independent

**`main` has a linear history, and every landed change receives non-author review.**

- Do not create merge commits on `main`; Squad work lands as reviewed, cherry-picked commits.
- The author does not approve their own change. Treat correctness, safety, numerics, API contracts, and diagnostic quality as review gates.
- Keep commits coherent and independently buildable/reviewable.

The repository’s active `main` ruleset requires linear history; the non-author review and cherry-pick workflow is recorded throughout `.squad/decisions.md`.
