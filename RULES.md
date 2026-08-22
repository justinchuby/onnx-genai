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

See [`docs/status/PROGRESS.md`](docs/status/PROGRESS.md).

## 2. Stay model-, vendor-, and EP-agnostic

**Runtime behavior is driven by model metadata, ONNX semantics, registries, capabilities, and explicit configuration—not hardcoded identities or hidden guesses.**

- Kernels are shape-driven, dtype-parameterized, and architecture-gated; model dimensions and attention parameters are runtime data.
- **No hardcoded model architecture, anywhere.** Neither inference metadata nor runtime code may bake in layer counts, hidden/intermediate sizes, head counts, head sizes (head dims), exact tensor shapes, or model-specific dimension constants.
- **Head size is a fully runtime, per-attention-op parameter.** Loader, attention kernel, GEMV, and KV-cache allocation resolve each op's head size from graph/metadata — never from a fixed value (e.g. 128/256), a fixed count of distinct sizes, or a `dual`-specific branch. Support arbitrary and mixed per-layer/per-component head sizes generally, so a model with three or more distinct head sizes requires no new special case.
- Generic loader, IR, session, optimizer, and dispatch code must not special-case model families, op attributes, vendors, or EPs.
- Architectural assumptions such as KV layout, RoPE variant, block-quant size, attention scheme, or sliding window are explicit, inspectable metadata; missing metadata fails clearly.
- EP selection and fusion use declared capabilities and structural op/topology patterns, never model identity; unsupported matches fail rather than guess or silently fall back.
- A fusion must generalize across every model that exhibits the pattern. Optimize per pattern category, not for one model; hardcoded shape constants that only match one model are review-blocking.
- Fusion lives inside the EP claim/compile path. Reusable pattern matching and rewriting belongs in the IR crate, not ad-hoc per-EP string or shape checks.

See [`docs/genai/MODEL_METADATA.md`](docs/genai/MODEL_METADATA.md) and [`docs/status/PROGRESS.md`](docs/status/PROGRESS.md).

## 3. Make pre-release changes cleanly

**Do not add backward-compatibility aliases, deprecation layers, or migration shims for our own pre-release APIs.**

- Rename, remove, or reshape an API completely; update all callers, docs, fixtures, and tests in the same change.
- Do not retain duplicate old symbols “just in case.”
- This does not waive product compatibility requirements such as supported ONNX opsets or the documented ORT/plugin ABI surface.

See [`docs/status/PROGRESS.md`](docs/status/PROGRESS.md) and [`docs/architecture/CRATE_RESERVATION.md`](docs/architecture/CRATE_RESERVATION.md).

## 4. Prefer explicit, inspectable behavior

**Debuggability and predictability beat cleverness, silent convenience, and hidden heuristics.**

- Optimization and placement decisions flow through explicit, inspectable cost models and capability checks.
- Eager execution never performs implicit cross-device transfers; users request `.to(device)` explicitly.
- Unsupported kernels, dtypes, attributes, opsets, or configurations fail clearly rather than silently changing semantics.

See [`docs/execution/EAGER.md` §1](docs/execution/EAGER.md#1-design-principles) and §13.

## 5. Use Rust types to enforce invariants

**Make invalid states unrepresentable so the compiler enforces invariants that would otherwise need a test, comment, or runtime check.**

- Use newtypes when primitives can be transposed: session ids, token counts, page indices, offsets, and lengths are not interchangeable `usize`s.
- Encode resolved capability in the value a caller holds; hot paths consume proof instead of rediscovering late failure.
- Let ownership and borrowing rule out aliasing. If two objects can mutate state they must not share, fix the design instead of trusting tests.

## 6. Use the canonical names

**Public names are consistent across each ecosystem.**

- Product, CLI, and planned Python package: `nxrt`; C ABI symbols: `nxrt_*`.
- Runtime Rust crates: `onnx-runtime-*`; GenAI-stack Rust crates: `onnx-genai-*`.
- Do not reintroduce legacy `ort2_*` public symbols.

See [`docs/status/PROGRESS.md`](docs/status/PROGRESS.md) and [`docs/architecture/CRATE_RESERVATION.md`](docs/architecture/CRATE_RESERVATION.md).

## 7. Ship stable-ABI Python wheels

**The planned PyO3 bindings support Python 3.10+ while minimizing per-version wheel builds.**

- Standard CPython wheels use `abi3` with a Python 3.10 compatibility floor (`abi3-py310`); the wheel target is py312.
- Free-threaded wheels use `abi3t`; the target is py315.
- Keep standard and free-threaded wheel configurations separate and test both surfaces.

## 8. Tests track behavior and APIs

**Behavioral and public-API changes include their tests in the same commit.**

- Cover changed success and failure paths, including error text when actionability is part of the contract.
- Update fixtures, expected counts, snapshots, conformance checks, and documentation examples with the API or behavior they describe.
- Run the smallest relevant test/lint set before landing; do not leave known CI cleanup to the next contributor.

See [`docs/status/PROGRESS.md`](docs/status/PROGRESS.md) for the project’s test, conformance, clippy, Miri, and audit expectations.

## 9. Run portably across hardware tiers

**The runtime must run correctly on whatever CPU, GPU, and memory tier it lands on—detecting capabilities at runtime and degrading gracefully, never demanding a specific fast ISA, arch, or footprint.**

- Detect CPU instruction sets at runtime and take the fast path when present (AVX-512/AVX2/NEON/SVE), with a correct portable scalar/generic fallback; a missing fast ISA slows execution, it never fails to run.
- Compile/JIT GPU kernels to the device actually present; do not assume one SM/arch or bake in datacenter constants—tune from queried device properties. Cross-reference the consumer-GPU audit.
- Memory-bandwidth and VRAM tier shape the bottleneck: a feature needing more VRAM/bandwidth than the tier has must degrade or clearly opt out, not crash. A 30B int4 model (~15 GB weights) fits an H200 but not an 8–12 GB consumer GPU—fail clearly per Rule 4 or offer a smaller-footprint path, never silently OOM.
- Perf claims are tier-scoped: state the device/EP/tier a benchmark or "ceiling" was measured on; never generalize one device (e.g. H200) into a universal conclusion. A lever flat on one tier may win on a bandwidth-starved or VRAM-limited tier.
- No hard runtime dependency on a specific vendor toolkit, driver, or arch beyond the declared minimum—keep the graceful-degradation contract consistent with Rule 2 and Rule 4.

## 10. Reduce entropy: derive the rule from first principles

**When a new case does not fit, state the principle that decides it—do not append another special case.**

- Before extending a predicate, dispatch table, or allowlist, write down *what property* makes the accepted cases correct; extend only then.
- Two code paths answering the same question are duplicated state. Collapse them into one classification.
- Gate on the operand, topology, or capability that determines correctness—never on op, model, or vendor names, which encode only what has been seen (Rule 2).
- A rejecting rule returns *why*, not just `false` (Rule 1).
- Simplification never costs a guard or a hot path: keep the tests that pinned the old behavior (Rule 8), and prefer a rule computed once to one re-derived per element.

See [`.github/skills/design-discipline`](.github/skills/design-discipline/SKILL.md) for the worked example.
