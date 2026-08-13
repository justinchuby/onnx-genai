# Project Rules

These durable rules bind every human contributor and every AI coding agent working on nxrt / onnx-genai.

## 1. Error Messages & Debug Experience

**Every failure must tell humans and AI agents what failed, why, and how to fix it; users should feel warmly cared for—暖暖的被捧在手心—not abandoned in “a stack trace from hell.”**

- User-facing errors name the rejected argument, request field, node/input, shape/dtype, opset, device/EP, path, or configuration value when it helps diagnosis.
- Rust errors display useful context, and boundaries add `anyhow::Context` / `with_context` so the causal chain names the operation and resource.
- C ABI calls return a machine-parseable code plus a retrievable rich message; never discard the Rust cause or unwind across FFI.
- Planned PyO3, CLI, and server surfaces preserve the same actionable context instead of generic `RuntimeError`, “failed,” or “internal error.”
- Prefer descriptive `Result`s over panics on user-facing paths; panic-fence FFI, fail closed on invalid input, and separate user/configuration errors from internal invariant failures.
- Reviewers must report weak, opaque, context-losing, or unactionable errors as real findings.

See [`docs/ORT2.md` §35](docs/ORT2.md#35-error-recovery--debug-experience) and §26.

## 2. Stay model-, vendor-, and EP-agnostic

**Runtime behavior is driven by model metadata, ONNX semantics, registries, capabilities, and explicit configuration—not hardcoded identities or hidden guesses.**

- Kernels are shape-driven, dtype-parameterized, and architecture-gated; model dimensions and attention parameters are runtime data.
- **No hardcoded model architecture, anywhere.** Neither inference metadata nor runtime code may bake in layer counts, hidden/intermediate sizes, head counts, exact tensor shapes, or model-specific dimension constants.
- Generic loader, IR, session, optimizer, and dispatch code must not special-case model families, op attributes, vendors, or EPs.
- Architectural assumptions such as KV layout, RoPE variant, block-quant size, attention scheme, or sliding window are explicit, inspectable metadata; missing metadata fails clearly.
- EP selection and fusion use declared capabilities and structural op/topology patterns, never model identity; unsupported matches fail rather than guess or silently fall back.
- A fusion must generalize across every model that exhibits the pattern. Optimize per pattern category, not for one model; hardcoded shape constants that only match one model are review-blocking.
- Fusion lives inside the EP claim/compile path. Reusable pattern matching and rewriting belongs in the IR crate, not ad-hoc per-EP string or shape checks.

See [`docs/ORT2.md` §15.1](docs/ORT2.md#151-decision-summary), §55.6, [`docs/MODEL_METADATA.md`](docs/MODEL_METADATA.md), and [`docs/PROGRESS.md`](docs/PROGRESS.md).

## 3. Make pre-release changes cleanly

**Do not add backward-compatibility aliases, deprecation layers, or migration shims for our own pre-release APIs.**

- Rename, remove, or reshape an API completely; update all callers, docs, fixtures, and tests in the same change.
- Do not retain duplicate old symbols “just in case.”
- This does not waive product compatibility requirements such as supported ONNX opsets or the documented ORT/plugin ABI surface.

See [`docs/PROGRESS.md`](docs/PROGRESS.md) and [`docs/CRATE_RESERVATION.md`](docs/CRATE_RESERVATION.md).

## 4. Do not rewrite what already works

**Reuse battle-tested primitives; write custom kernels only for a measured, necessary advantage.**

- CPU uses the built-in SIMD backend; CUDA uses cuBLAS/cuBLASLt and cuDNN before CuTe/CUTLASS custom fusions.
- Profile before replacing proven implementations, and keep thin seams so reference and optimized paths remain testable.

See [`docs/ORT2.md` §1](docs/ORT2.md#1-design-principles) and §15.

## 5. Prefer explicit, inspectable behavior

**Debuggability and predictability beat cleverness, silent convenience, and hidden heuristics.**

- Optimization and placement decisions flow through explicit, inspectable cost models and capability checks.
- Eager execution never performs implicit cross-device transfers; users request `.to(device)` explicitly.
- Unsupported kernels, dtypes, attributes, opsets, or configurations fail clearly rather than silently changing semantics.

See [`docs/ORT2.md` §1](docs/ORT2.md#1-design-principles), §6, and [`docs/EAGER.md` §1](docs/EAGER.md#1-design-principles) / §13.

## 6. Use Rust types to enforce invariants

**Make invalid states unrepresentable so the compiler enforces invariants that would otherwise need a test, comment, or runtime check.**

- Use newtypes when primitives can be transposed: session ids, token counts, page indices, offsets, and lengths are not interchangeable `usize`s.
- Encode resolved capability in the value a caller holds; hot paths consume proof instead of rediscovering late failure.
- Let ownership and borrowing rule out aliasing. If two objects can mutate state they must not share, fix the design instead of trusting tests.

## 7. Use the canonical names

**Public names are consistent across each ecosystem.**

- Product, CLI, and planned Python package: `nxrt`; C ABI symbols: `nxrt_*`.
- Runtime Rust crates: `onnx-runtime-*`; GenAI-stack Rust crates: `onnx-genai-*`.
- Do not reintroduce legacy `ort2_*` public symbols or rename the retained design file `docs/ORT2.md`.

See [`docs/PROGRESS.md`](docs/PROGRESS.md) and [`docs/CRATE_RESERVATION.md`](docs/CRATE_RESERVATION.md).

## 8. Ship stable-ABI Python wheels

**The planned PyO3 bindings support Python 3.10+ while minimizing per-version wheel builds.**

- Standard CPython wheels use `abi3` with a Python 3.10 compatibility floor (`abi3-py310`); the wheel target is py312.
- Free-threaded wheels use `abi3t`; the target is py315.
- Keep standard and free-threaded wheel configurations separate and test both surfaces.

## 9. Tests track behavior and APIs

**Behavioral and public-API changes include their tests in the same commit.**

- Cover changed success and failure paths, including error text when actionability is part of the contract.
- Update fixtures, expected counts, snapshots, conformance checks, and documentation examples with the API or behavior they describe.
- Run the smallest relevant test/lint set before landing; do not leave known CI cleanup to the next contributor.

See [`docs/PROGRESS.md`](docs/PROGRESS.md) for the project’s test, conformance, clippy, Miri, and audit expectations.

## 10. Keep history linear and review independent

**`main` has a linear history, and every landed change receives non-author review.**

- Do not create merge commits on `main`; Squad work lands as reviewed, cherry-picked commits.
- The author does not approve their own change. Correctness, safety, numerics, API contracts, and diagnostic quality are review gates.
- Keep commits coherent and independently buildable/reviewable.

The repository’s active `main` ruleset requires linear history; the non-author review and cherry-pick workflow is recorded throughout `.squad/decisions.md`.

## 11. Run portably across hardware tiers

**The runtime must run correctly on whatever CPU, GPU, and memory tier it lands on—detecting capabilities at runtime and degrading gracefully, never demanding a specific fast ISA, arch, or footprint.**

- Detect CPU instruction sets at runtime and take the fast path when present (AVX-512/AVX2/NEON/SVE), with a correct portable scalar/generic fallback; a missing fast ISA slows execution, it never fails to run.
- Compile/JIT GPU kernels to the device actually present; do not assume one SM/arch or bake in datacenter constants—tune from queried device properties. Cross-reference the consumer-GPU audit.
- Memory-bandwidth and VRAM tier shape the bottleneck: a feature needing more VRAM/bandwidth than the tier has must degrade or clearly opt out, not crash. A 30B int4 model (~15 GB weights) fits an H200 but not an 8–12 GB consumer GPU—fail clearly per Rule 5 or offer a smaller-footprint path, never silently OOM.
- Perf claims are tier-scoped: state the device/EP/tier a benchmark or "ceiling" was measured on; never generalize one device (e.g. H200) into a universal conclusion. A lever flat on one tier may win on a bandwidth-starved or VRAM-limited tier.
- No hard runtime dependency on a specific vendor toolkit, driver, or arch beyond the declared minimum—keep the graceful-degradation contract consistent with Rule 2 and Rule 5.

See [`docs/portability/2026-07-25-cuda-consumer-gpu-audit.md`](docs/portability/2026-07-25-cuda-consumer-gpu-audit.md), [`docs/CROSS_PLATFORM.md`](docs/CROSS_PLATFORM.md), [`docs/benchmarks/2026-07-25-gqa-decode-avx512.md`](docs/benchmarks/2026-07-25-gqa-decode-avx512.md), and [`docs/research/lowbit-quant-feasibility.md`](docs/research/lowbit-quant-feasibility.md).
