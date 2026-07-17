# Decisions archive — 2026-07

## 2026-07-16 — CUDA MatMul assertion, GAFF control-flow loader, and Pad axes fixes

### Match CUDA MatMul rejection test to the implemented unsupported case
**By:** Roy; reviewed by Wallace
**What:** Updated `matmul_rejects_unsupported_rank_and_dtype` to assert the actionable Int64 rejection, `MatMul with dtype Int64 is not yet implemented on the CUDA EP`, replacing obsolete “Phase 2a” terminology. Four-dimensional MatMul is already supported.
**Why:** The test must guard the current CUDA EP contract rather than an obsolete implementation phase. Wallace independently verified the error path and a full CUDA suite pass (129/129).

### Preserve formal subgraph I/O and scoped inline initializers in the GAFF loader
**By:** Sapper; reviewed by Leon
**What:** The loader now records ordered typed formal inputs/outputs for graph attributes and stores local body initializers as scoped `WeightRef::Inline` values. Its `UNDEFINED` attribute fallback recognizes populated `g`/`graphs` fields, including recursively nested subgraphs.
**Why:** Future child executors require complete body signatures and local constants. The baseline validation already permits default-domain If/Loop/Scan, so no loader relaxation is required. Leon verified independent nested scopes and the Loop regression; loader build and all 101 tests passed. Next: child-executor plus If/Loop/Scan execution.

### Infer Pad dimensions from opset-18 `axes`
**By:** Joi; reviewed by Bryant
**What:** Pad shape inference now maps begin/end values to the optional opset-18 axes input, including negative axes, while preserving unlisted dimensions. The expanded-Attention regression asserts `[2,3,4,6]`, 144 f32 elements, and 576 bytes.
**Why:** Ignoring `axes` inferred 640 bytes for `[2,3,4,4]` with pads `[0,2]` on `[-1]`, while the CPU kernel correctly produced 576 bytes. Bryant cleared the focused and crate validation. The case now advances past Pad and exposes the separate follow-up: `Less` output dtype inference must be Bool, not Float32.

**Sources:** `roy-cuda-matmul-stale-test.md`, `wallace-roy-cuda-test-review.md`, `sapper-gaff-loader-io.md`, `leon-sapper-gaff-loader-review.md`, `joi-pad-bytecount-fix.md`, and `bryant-joi-pad-review.md`; merged commits `3d19b72`, `2a9e5b1`, and `0a105a4`.
## 2026-07-16 — CUDA parity and native int4 performance wave

### CUDA GroupQueryAttention parity approved (`3820bad`)
**By:** Pris
**What:** 🟢 Approved the CUDA GroupQueryAttention implementation after all eight reviewed ORT/CPU-contract behaviors, five GQA GPU tests, two existing Attention GPU tests, and the full CUDA EP suite passed with real CUDA execution and independent CPU/manual numerical references.
**Why:** The GPU implementation now has numerically validated GQA parity; the nvrtc-12.6 workaround was used for validation.

### Native GEMM prepack and threaded oneDNN approved (`f132d30`)
**By:** Holden
**What:** 🟢 Approved constant-MatMul prepacking and oneDNN threaded GEMM. Constants are identified only from `Graph::initializers`; the kernel cache is per-node and shape-keyed; runtime activations remain uncached; generic and oneDNN numerics pass.
**Why:** The seam produces a measured 1.14× fp32 Gemma4-e2b improvement without changing other kernel implementations. Follow up with non-contiguous/f16 activation-cache tests and bounded accounting for widened f16/bf16 constant caches.

### CUDA SkipSimplifiedLayerNormalization re-review approved (`0284999`)
**By:** Rachael
**What:** 🔴 Rejected `bc379d8` because the registered coverage count (61) disagreed with documentation and lacked an exact cardinality assertion. 🟢 Approved corrective commit `f792bc2`/merged `0284999`: `CUDA_COVERED_OPS` has exactly 61 unique names, enforced before the duplicate guard, and CUDA coverage documentation consistently reports 61.
**Why:** Published coverage must match the advertised registry and be protected against drift. The CUDA EP test suite (79 unit tests and GPU integration suites, including independent residual-RMS validation) passed.

### Native genai-builder compatibility approved (`04709a4`)
**By:** Pris
**What:** 🟢 Approved native CPU support for standard `SimplifiedLayerNormalization` and packed-QKV `GroupQueryAttention`, allowing genai-builder exports to run unmodified.
**Why:** Compatibility is preserved at the standard exported ONNX contract rather than requiring model rewrites.

### Fast prepacked MatMulNBits decode path: reject then re-approve (`3787c21`)
**By:** Holden
**What:** 🔴 Rejected `37b0ced` because its named test used `M=2`, which only exercised the fallback rather than the `M=1` cached GEMV fast path; it also lacked independent q-nibble dequantization and fresh-activation proofs. 🟢 Re-approved after Deckard added the `M=1` fast-path coverage in merged commit `3787c21`.
**Why:** The cache is valid only for constant packed weights and must read activations live. The corrected coverage validates the actual decode path, delivering int4 native decode from 0.19 to 0.50 tok/s; the fp32 cache remains an 8× packed-weight expansion and must remain documented as a material memory trade-off.

**Sources:** `pris-cuda-gqa-review.md`, `holden-perf-matmul-review.md`, `rachael-cuda-skiprms-review.md`, requested `pris-native-compat-review.md` recovery, and `holden-perf-nbits-review.md`; merged commits `3820bad`, `f132d30`, `eea887e`, `04709a4`, `0284999`, and `3787c21`.
## 2026-07-16 — CUDA, serde, model-package, and native decode follow-ups

#### Source: `batty-onnxrs-serde-unify.md`

### 2026-07-16: Unify ONNX-RS string codecs without merging file I/O
**By:** Batty
**What:** The text DSL now uses `to_text`/`to_text_with`/`from_text`, with private `ser.rs` and `de.rs` modules. `TextCodec` plus `Text`, `Json`, and `TextProto` markers provides one generic `serialize`/`deserialize` shape while existing JSON/TextProto and new text free functions remain the direct public APIs.
**Why:** Free functions preserve the established stateless conversion convention and existing call sites, while the trait enables generic string-codec tooling with format-specific options. Binary protobuf and `ModelFormat`/`FormatRegistry` remain separate because paths, external weights, and file detection are different concerns.

#### Source: `coordinator-native-cuda-end-to-end-decode-is-blocked-on-sessio.md`

### 2026-07-16T02-08-39: Native CUDA end-to-end decode is blocked on session-executor EP wiring, not just op gaps
**By:** coordinator
**What:** Native CUDA end-to-end decode is blocked on session-executor EP wiring, not just op gaps
**Why:** Luv's 2026-07-16 CUDA int4 decode benchmark (docs/benchmarks/2026-07-16-cuda-int4-decode.md, commit 78e1259) found the native runtime cannot yet decode on GPU because crates/onnx-runtime-session/src/executor.rs hardcodes ep: Arc<CpuExecutionProvider> and Executor::build takes a concrete CpuExecutionProvider — GPU/device preferences from onnx-genai are accepted but ignored; only the CPU EP is ever instantiated. The executor also deeply assumes CPU host/mmap-borrowed buffers, so running on CUDA needs real device-memory management (device buffer alloc, H2D/D2H at graph boundaries, KV on device) plus per-node EP dispatch. Decision: dedicated multi-step architecture effort AFTER trivial op gaps. Step 1 (Roy, squad/cudaexec): CUDA Gather/Shape/Constant kernels (coverage 62 to 65). Step 2 (future, one focused agent): make Executor EP-generic/dispatchable preserving CPU path bit-identically, then wire device buffers. Do NOT big-bang. The onnx-genai-ORT CUDA path already works; NATIVE pure-Rust CUDA decode is the longer-horizon target.

#### Source: `deckard-cuda-graph-eligibility.md`

### 2026-07-16: Centralize CUDA graph capture eligibility
**By:** Deckard
**What:** Added the public `subgraph_graph_capturable(kernels: &[&dyn Kernel]) -> bool` CUDA EP gate. It declares a subgraph capturable only when every resolved kernel reports `cuda_graph_compatible()`, and capture-facing tests now use this gate rather than reading kernel metadata directly.
**Why:** A native capture executor does not exist yet, but kernel compatibility metadata must have one honest aggregation point for that future executor. Centralizing the all-kernels rule makes Gather's synchronous D2H/stream-sync incompatibility effective at the eligibility boundary without inventing capture/replay infrastructure.

#### Source: `deckard-gather-fix.md`

### 2026-07-16: Validate native axis-0 Gather before copying
**By:** Deckard
**What:** The contiguous axis-0 Gather path now checks all size arithmetic, requires the destination element count to equal `indices × row_elements`, and validates every wrapped index before any row copy.
**Why:** This preserves selected-row memcpy performance while making partial writes and output-bound overruns impossible on malformed output metadata or indices.

#### Source: `fact-checker-model-package-verify.md`

### 2026-07-16: Verification of `docs/MODEL_PACKAGE.md` Microsoft-source claims
**By:** Fact Checker
**What:** Verified the concrete claims in §§2.1–2.4 and §11 against the current `main` branches of `microsoft/onnxruntime` and `microsoft/onnxruntime-genai`, plus PR `microsoft/onnxruntime-genai#2255`.
**Why:** These claims define the external compatibility basis for the proposed design and must match the real Microsoft implementation.

## Overall verdict

**High confidence, with one material overstatement:** the document faithfully represents the ORT standalone library, ORT session integration, GenAI package documentation, and nearly all of PR #2255. However, the Microsoft model-package sources do **not** establish `EPContext` as the package format's required or canonical compiled-graph interchange, and §2.4 overstates PR #2255's test coverage by claiming dedicated relative-path and EP-context-path tests.

Evidence was checked at ONNX Runtime `main` commit `a91b0b49cb0dc9670a8cf93263b3d79ce0dc79a5`, onnxruntime-genai `main`, and PR #2255 (merge commit `2ef64f99339fc6f21831827c24f4dc86206699d6`, merged `2026-07-13T18:37:00Z`).

## Claim-by-claim ratings

| Claim | Rating | Evidence / correction |
|---|---|---|
| A standalone top-level `model_package/` library exists at the root of `microsoft/onnxruntime`. | ✅ Verified | `model_package/` exists independently of `onnxruntime/core/session/model_package/`. Its root contains `CMakeLists.txt`, `README.md`, `include/`, `src/`, and `tests/`. `model_package/README.md` begins: “A standalone C library for reading, authoring, validating, and committing ONNX Runtime model packages.” |
| `model_package/README.md` exists. | ✅ Verified | Real path: `microsoft/onnxruntime/model_package/README.md`. |
| `model_package/include/model_package.h` and `model_package/include/model_package_api.h` exist. | ✅ Verified | Both files are present under `model_package/include/`. |
| The standalone source contains `manifest_parser.cc`, `path_resolver.cc`, `asset_hasher.cc`, `authoring.cc`, and `commit_prune_validate.cc`. | ✅ Verified | Real `model_package/src/` list: `asset_hasher.cc/.h`, `authoring.cc`, `commit_prune_validate.cc`, `manifest_parser.cc/.h`, `model_package_impl.cc/.h`, `path_resolver.cc/.h`, `sha256.cc/.h`, `status_impl.h`. |
| The model package is a directory, not a single archive. | ✅ Verified | `model_package/README.md`: “A package is a directory containing a top-level `manifest.json`” and “A single file is not a package.” `ModelPackage_Commit` writes directory layouts; no archive packer is described. |
| The current schema is “1.x”. | ✅ Verified, with precision | The on-disk form is `"<major>.<minor>"`; current constants are `kMinSupportedSchemaMajor = 1`, `kMaxSupportedSchemaMajor = 1`, `kMaxKnownSchemaMinor = 0`, and newly authored packages use `"1.0"`. Any `1.x` minor is accepted; unsupported majors are rejected. |
| Variants contain consumer-namespaced `executor_info`, including `executor_info["ort"]`. | ✅ Verified | `model_package/README.md`: `executor_info` is a “Map of consumer namespace → string (external file) or object (inline JSON).” The standalone library deliberately does not interpret payloads. `onnxruntime/core/session/model_package/README.md` says ORT owns the `"ort"` slot. |
| Shared assets use `sha256:<64hex>[/sub/path]` references and default directories `shared_assets/sha256-<hex>/`. | ✅ Verified | Both the standalone README and `model_package/src/path_resolver.*` define this URI scheme. `manifest.shared_assets` is an optional location override. |
| The standalone library owns variant selection. | ❌ Contradicted if read that way | The standalone README explicitly lists variant selection under “What the library deliberately does NOT do.” ORT session integration owns selection. The design document itself correctly attributes selection to §2.2 rather than the standalone library. |
| Variant selection is a real ORT model-package concept. | ✅ Verified | `onnxruntime/core/session/model_package/model_package_variant_selector.cc/.h` exists. The integration README documents EP-name/device filtering, `ValidateCompiledModelCompatibilityInfo`, scores 100/50/0/reject, manifest-order tie-breaking, and current use of only the first EP. |
| `onnxruntime/core/session/model_package/` contains the files named in §2.2. | ✅ Verified | Exact current list: `README.md`, `model_package_context.cc`, `model_package_context.h`, `model_package_options.cc`, `model_package_options.h`, `model_package_variant_selector.cc`, `model_package_variant_selector.h`. |
| `onnxruntime/core/session/model_package_api.cc` exists separately. | ✅ Verified | Real path exists: `onnxruntime/core/session/model_package_api.cc`. |
| ORT's `"ort"` payload has `model_file`, `session_options`, and `provider_options`, with model and path-valued options resolved through `ModelPackage_ResolveStringRef`. | ✅ Verified | The integration README gives this exact payload. `model_package_context.cc` resolves `model_file`, `session.model_external_initializers_file_folder_path`, and `ep.context_file_path`. |
| The experimental model-package API is a SinceV28 API for context/options/component selection and session creation. | ✅ Verified | The integration README lists the `OrtModelPackageApi_*_SinceV28` functions and says they are resolved through `OrtApi::GetExperimentalFunction`. |
| “EPContext remains the executable/compiled-graph interchange” is established by the Microsoft model-package sources. | ⚠️ Unverified / overstated | The standalone library explicitly knows nothing about ONNX. ORT's package integration selects a `model_file` and supports the path-valued `ep.context_file_path`, but its README does not require the selected ONNX model to contain `com.microsoft::EPContext` nodes or call EPContext the package interchange. A package may select an ordinary ONNX model. This is a plausible nxrt design choice, not a confirmed model-package schema fact. |
| `microsoft/onnxruntime-genai/docs/model_package.md` exists. | ✅ Verified | Real path exists on `main`; current blob SHA is `2b5676a4ce7fc1139522d48896e67ef13d60cfa0`. |
| The GenAI convention uses one inline component named `model`, with variant directories directly below the package root. | ✅ Verified | The doc says the package owns a single inline component, conventionally named `model`, and its variant directories sit directly at package root. |
| Every variant contains a complete `genai_config.json`, graphs, weights, and variant-specific assets including adapters. | ✅ Verified | The doc says variant directories may contain ONNX graphs, external weights, custom-op libraries, “LoRA adapters,” and other per-variant files. |
| Tokenizer files can be shared through `shared_assets/sha256-<hex>/` and `model.tokenizer_dir: "sha256:<hex>"`. | ✅ Verified | Explicitly documented with layout and JSON examples. |
| Processor config is intentionally kept per variant. | ✅ Verified | The doc explicitly names `model.vision.config_filename` and `model.speech.config_filename` as per-variant. |
| The GenAI doc describes package loading APIs and flat-directory compatibility. | ✅ Verified | It documents `og.Model`, `og.Config.from_package_ep`, `OgaCreateModel`, `OgaCreateConfigFromPackageEp`, C++ wrappers, flat-directory compatibility, and the `OgaRuntimeSettings` restriction. |
| The GenAI doc describes a pack/unpack or package-authoring workflow. | ❌ Contradicted | It describes layout, authoring notes, and loading, but no pack CLI/API or pack/unpack workflow. `docs/MODEL_PACKAGE.md` §2.3 does not incorrectly attribute one; §2.4 correctly says PR #2255 added none. |
| PR #2255 exists with the quoted title. | ✅ Verified | Title: **“Resolve model package paths and path-valued session options through ONNX Runtime.”** URL: `https://github.com/microsoft/onnxruntime-genai/pull/2255`. |
| PR #2255 was merged on 2026-07-13. | ✅ Verified | State `MERGED`; merged at `2026-07-13T18:37:00Z`. |
| PR #2255 changed the nine files listed in §2.4. | ✅ Verified | Exact files: `docs/model_package.md`, `src/config.cpp`, `src/config.h`, `src/models/model.cpp`, `src/models/model_package.cpp`, `src/models/model_package.h`, `src/models/onnxruntime_api.h`, `src/models/onnxruntime_inline.h`, `test/model_package_test.cpp`. They were existing files, not newly introduced files. |
| PR #2255 introduced the model-package format or initial loading API. | ❌ Contradicted | Those came from merged PR #2227, **“Load models from ONNX Runtime model packages”** (merged 2026-06-22). PR #2255 explicitly “Builds on model package loading (#2227)” and tightens path resolution. The design document correctly says it builds on the earlier loading surface. |
| PR #2255 replaced `package:<relative-path>` with ORT-owned `sha256:` references and inline schema-`"1.0"` examples. | ✅ Verified | The diff removes `package:` examples/logic, adds `sha256:<hex>[/tail]`, changes examples from numeric `1` to string `"1.0"`, and uses inline components at package root. |
| `Config` stores a resolver closure capturing `OrtModelPackageContext` to preserve lifetime. | ✅ Verified | `src/models/model.cpp` assigns `config->package_resolver = [package_context](...)`, and the comment says the capture keeps the context alive for the config lifetime. |
| `Config::ResolvePath` delegates package references to ORT and rejects `sha256:` in flat directories. | ✅ Verified | `src/config.cpp` calls `package_resolver(config_path, value)` for packages; flat directories throw a clear error for the `sha256:` prefix. |
| PR #2255 resolves external-initializer and EP-context session-option paths before applying them. | ✅ Verified | `src/models/model.cpp` recognizes both `session.model_external_initializers_file_folder_path` and `ep.context_file_path`, calls `Config::ResolvePath`, then adds the resolved value. |
| Memory-loaded models default the external-initializers folder only if not already configured. | ✅ Verified | `Model::CreateSession` checks `HasConfigEntry(...)` before applying `config_path` as the default. |
| GenAI resolves experimental model-package functions locally rather than vendoring ORT's experimental C++ header. | ✅ Verified | `src/models/onnxruntime_inline.h::GetModelPackageApi()` uses `api->GetExperimentalFunction(...)`; its comment explicitly explains avoiding `onnxruntime_experimental_cxx_api.h`. |
| PR #2255 added a package archive, pack CLI, or new user-facing pack API. | ✅ Verified as false | It added none. No CLI files or public pack APIs are in the diff. |
| PR #2255 tests cover tokenizer shared assets, shared external initializers, relative path resolution, EP-context paths, and flat-directory `sha256:` rejection. | ❌ Partly contradicted | The merged `test/model_package_test.cpp` has dedicated tests `TokenizerResolvesThroughSharedAsset`, `ExternalInitializersFolderResolvesThroughSharedAsset`, and `FlatDirectoryRejectsSha256Reference`. There is no dedicated relative-path-resolution test and no `ep.context_file_path`/EP-context test in the PR's only test file. §2.4 item 7 should not claim those two coverages for PR #2255. |

## Most important corrections

1. **Do not collapse the two ORT paths.** Both are real: root `model_package/` is the standalone schema/library; `onnxruntime/core/session/model_package/` is ORT consumer integration.
2. **Treat EPContext as a proposed nxrt integration choice, not a proven package-schema rule.** The sources support `ep.context_file_path`, but model packages are generic and can select ordinary ONNX models.
3. **PR #2255 did not create the format, initial loading APIs, CLI, or new files.** Initial GenAI package loading was PR #2227. PR #2255 modified nine existing files to delegate path resolution to ORT.
4. **Correct §2.4 test-coverage wording.** The PR has no dedicated relative-reference or EP-context-path test.

#### Source: `freysa-onnxrs-serde-review.md`

### 2026-07-16: onnx-rs unified string-serde review
**By:** Freysa
**Verdict:** 🟡 Approve with notes
**What:** Commit `fc4fa66` provides a coherent `TextCodec` API across `Text`, `Json`, and `TextProto`; preserves the public JSON/TextProto free-function pairs; consistently exposes `to_text`/`to_text_with`/`from_text`; and keeps binary `load_model`/`save_model` separate. Workspace searches found no stale production call sites. The six integration tests use real `onnx_runtime_ir` builders and independent ONNX DSL fixtures/goldens matching upstream `onnx/test/parser_test.py` and `printer_test.py`, rather than self-derived expected values.
**Validation:** `cargo build --workspace` passed. `cargo test -p onnx-rs` passed 79 total tests (72 unit + 6 integration + 1 doctest). `cargo clippy -p onnx-rs --all-targets` passed with only the three acknowledged pre-existing `field_reassign_with_default` warnings. `git diff --check` passed.
**Note:** The updated §5.4/§6 API sketches are accurate, but pre-existing §5.3 prose still says attributes use `{ key: value }` and gives `f32[...]`; the implemented/upstream DSL uses `<key = value>` and `float[...]`. This is non-blocking documentation cleanup.
**Why:** The requested API unification is complete, builds cleanly, has meaningful upstream-derived coverage, and does not conflate string codecs with file-format/weight handling. Only adjacent stale documentation remains.

#### Source: `holden-gather-profile-review.md`

# Holden review — Gather optimization and per-op profiler

- **Reviewed commit:** `15121c6f06328d61a2ff02f94fe30b3b06b4188d`
- **Reviewed at:** 2026-07-16T04:20:00Z
- **Author:** Leon
- **Verdict:** 🔴 REJECT
- **Revision owner:** Deckard (Leon is locked out for this revision cycle)

## Blocking finding

`crates/onnx-runtime-ep-cpu/src/kernels/gather.rs:67-94` bypasses the generic
path's `write_dense_bytes` output-size check and writes
`indices.len() * row_bytes` directly into the output pointer. The fast-path
gate checks contiguity but never verifies that the output view has the expected
element count/shape. A malformed or inconsistent output shape can therefore
return a partial result when oversized, or cause an out-of-bounds write and
undefined behavior when undersized. `TensorMut::validate()` only checks view
metadata; it cannot check backing allocation size.

Deckard must add checked expected-size validation before pointer arithmetic and
copying, returning `EpError` on mismatch, plus a regression test for a
mismatched contiguous output. Add explicit out-of-range-index coverage and,
preferably, a non-contiguous fallback reference test while touching this area.

## Verified

- Existing IR-builder, hand-computed tests cover axis-0 multi-row/multirank,
  int32 and int64 indices, negative indices, and an axis-1 fallback.
- Fast-path gating is axis 0 plus contiguous input/output; unsupported
  fixed-width handling still goes through `elem_size`, and non-axis-0 or
  strided layouts retain the prior materialization path.
- Profiling is disabled through a once-initialized environment flag and branches
  once per executor run; the disabled per-node loop contains no timing,
  allocation, or profiling branch.
- The prior implementation called `to_dense_bytes(data)` before selecting any
  rows, so the reported removal of full embedding-table copies plausibly
  explains the measured speedup.
- `cargo test -p onnx-runtime-ep-cpu -p onnx-runtime-session` passed (512 listed
  tests; Gather's 7 targeted tests passed).
- `cargo build --locked -p onnx-runtime-ep-cpu -p onnx-runtime-session` passed.

---

### 2026-07-16T05:05:00Z — 🟢 APPROVE re-review

- **Reviewed fix:** `edad52652eb34f3b4519faa3ae1a8dc7b2a91bd2`
- **Revision author:** Deckard
- **Verdict:** 🟢 APPROVE — original rejection resolved; ready to merge.

`gather.rs:93-120` validates both views and checked-computes row, source, and
destination sizes; lines 100-107 require the destination logical element count
to equal `indices.len() * row_elements`. Lines 122-126 normalize and range-check
every index before the first write, using i128 at lines 19-30 so negative
wrapping cannot overflow. Lines 130-150 checked-compute each copy range. Given
the prior total-size checks, no data-dependent failure remains after writing
begins, so neither partial nor out-of-bounds writes are reachable under the
tensor-view backing-storage contract.

Out-of-range indices return `KernelFailed`; they are not clamped, consistent
with the unchanged general path. Rust tests at `gather.rs:254-336` cover exact
single/multi-row results, negative indices, invalid-later-index no-partial-write,
mismatched destination no-write, and noncontiguous/general fallback. The six
Python tests in `crates/onnx-runtime-python/tests/test_gather.py` use the ONNX IR
builder and independent `_take` expectations.

Validation passed: 514 Rust tests, 6 Python Gather tests, and the locked build.
A fresh three-run release profile measured 4.95 tok/s with deterministic IDs
`[12095, 13, 1084]` (`Paris. It`), preserving the reported ~4-5 tok/s gain.

#### Source: `holden-matmulnbits-threading-review.md`

### 2026-07-16: Reject MatMulNBits N-threading as currently tuned
**By:** Holden
**What:** 🔴 REJECT commit `2387c4a`. Zhora is locked out; Sebastian should revise the partition policy and benchmark it across thread counts.
**Why:** The anomaly is primarily (a), with a smaller real 96-thread scheduling win—not DRAM saturation or pure noise.

- Model inspection proves all 121 Qwen2.5-0.5B `MatMulNBits` nodes use `accuracy_level=4`, so decode enters `int8_matmul`, not the fp32 `gemv_nk` path. The baseline already used Rayon `par_iter_mut()` over N for every M=1 int8 call; this change is a repartition/serialization policy, not first-time threading.
- Env-gated temporary instrumentation at `RAYON_NUM_THREADS=96` proved 48/121 decode calls are forced serial by `MIN_PARALLEL_DOT_PRODUCTS`: 24× K=896,N=1152 have 1,032,192 terms—only 1.6% below the 1,048,576 cutoff—and 24× K=896,N=896 have 802,816. The other 73 calls did parallelize: K=896,N=4864 used 52–65 distinct workers/call, K=4864,N=896 used 31–41, and the LM head used 77. Thus 40% of real decode matmuls miss the new partition.
- Repeated 96-thread E2E runs show a modest real but overstated/configuration-specific gain: 7 paired processes × 40 measured tokens gave baseline 86.83±3.20 ms/step vs branch 81.24±1.21; paired median speedup 4.92% (mean 6.32%), not a stable 9.2%.
- The policy regresses useful thread counts. Three-process medians: at 8 threads, 32.792→32.761 ms/step (flat); at 24, 33.093→37.431 ms/step (13.1% slower). Op-profile medians similarly worsened at 8 threads (23.829→27.302 ms `MatMulNBits`) and 24 (25.393→31.462 ms). Temporarily removing the 1 Mi cutoff improved the 24-thread branch median to 31.622 ms, isolating the threshold as a material cause.
- This is not aggregate memory-bandwidth saturation. One decode reads at least 493.96 MB of int8-prepacked values + 61.75 MB scales + 61.75 MB block sums = 617.45 MB. At 68.9–75.5 ms, achieved traffic is only ~8.2–9.0 GB/s; even the 8-thread 23.8 ms result is ~25.9 GB/s. Arithmetic intensity is only ~1.6 integer ops/byte, but the low achieved bandwidth proves task launch/synchronization, small-projection gating, and cross-NUMA scheduling dominate—not the box's DRAM ceiling.
- Correctness/safety gates pass: `cargo build -p onnx-runtime-ep-cpu`; 405/405 tests pass; output token IDs remain `[11576, 42740, 11, 358]`. `par_chunks_mut` gives disjoint output slices; weights/scales/activations are immutable, and Rayon uses the existing global pool, so no race or second-pool oversubscription was found.

Sebastian should replace the fixed global cutoff with a thread-count/shape-aware policy and require non-regression at 8/24/96 workers, including explicit task-entry counters for every model projection. After that, the next lever is direct int4 compute (avoid the ~494 MB int8-expanded weight stream and ~61.75 MB block-sum stream), then NUMA-aware placement or fused QKV/gate-up dispatch.

### 2026-07-16: Re-review Sebastian's thread-aware partition
**By:** Holden
**What:** 🟡 ACCEPT commit `485defae`; the prior performance blocker is resolved. Coordinator may rebase and merge. Follow up on the hardware-specific 48-worker policy cliff.
**Why:**

- Independent `profile_native` medians versus current `origin/main` (three interleaved processes, profiling enabled) were: 24 workers **29.37→37.29 tok/s (+27.0%)**, MatMulNBits **26.082→19.016 ms (-27.1%)**; 48 workers **20.17→22.97 (+13.9%)**, **40.020→34.555 ms (-13.7%)**; 96 workers **11.72→14.36 (+22.5%)**, **73.163→59.777 ms (-18.3%)**. Gains are smaller than Sebastian's 48-worker report but real and directionally consistent. Token IDs remained `[11576, 42740, 11, 358]`.
- At one worker, eleven profiling-enabled processes showed wall throughput **11.62→11.43 tok/s (-1.6%)**, despite profiled node execution improving **84.019→81.539 ms** and MatMulNBits **78.006→75.453 ms**. Five longer production-style runs with profiling disabled removed the logging/timing anomaly: median **11.34→11.58 tok/s (+2.1%)**. I do not find a real one-thread regression.
- The old 1-Mi-term hole is closed: at 24/48 workers, even the 896×896 projection exceeds the pool-scaled gate and gets `chunk=36`, so all 121 model MatMulNBits nodes partition. At 96, medium projections intentionally remain serial and the 151936×896 head partitions. Tiny work and one-worker pools remain serial.
- `cargo test -p onnx-runtime-ep-cpu`: **406 passed**, 0 failed; the added serial/parallel test is bit-exact. Existing int8 checks use `0.05 + 5%`; fp32 checks remain near-exact. `par_chunks_mut` gives disjoint output slices, while activations, weights, scales, and block sums are shared immutably. The implementation uses only Rayon's existing pool.
- 🟡 concern: `MANY_THREAD_CUTOFF=48` creates a sharp, host-topology-specific policy change without NUMA awareness. Extra probes still did not regress this model (49 workers **19.47→20.33**, 64 workers **16.76→17.11 tok/s**), so it is not a merge blocker, but a smoother/topology-aware policy should replace the fixed dual-socket heuristic before treating it as generally tuned.

#### Source: `leon-perf-profile.md`

### 2026-07-16: Optimize contiguous axis-0 Gather before further int4 GEMV tuning
**By:** Leon
**What:** Add env-gated per-op executor profiling and make CPU Gather copy only selected rows directly for contiguous axis-0 inputs/outputs, retaining the generic strided fallback.
**Why:** A steady-state Qwen int4 decode step spent 88.37% (688.004 ms) in two Gather calls because each copied the full embedding table. Direct row copies reduced unprofiled latency from 934.809 to 212.295 ms/step (1.07 to 4.71 tok/s); MatMulNBits is now the next dominant bottleneck.

#### Source: `luv-cuda-decode-bench.md`

### 2026-07-16: Native CUDA int4 decode remains blocked
**By:** Luv
**What:** The Qwen2.5-0.5B int4 end-to-end native CUDA benchmark cannot start because `onnx-runtime-session` still instantiates only the CPU EP. The model also contains three op types absent from `CUDA_COVERED_OPS`: `Gather` (2 nodes), `Shape` (1), and `Constant` (2).
**Why:** CUDA EP selection must be wired into the session/executor before throughput can be measured; after that, these graph ops require CUDA kernels or an explicit fallback/folding strategy. The full evidence is in `docs/benchmarks/2026-07-16-cuda-int4-decode.md`.

#### Source: `nabil-model-package-design.md`

### 2026-07-16: Base model packages on ORT schema with an optional nxrt archive envelope
**By:** Nabil
**What:** Adopt ORT model-package schema 1.x and `sha256:` shared assets as the canonical directory format; add `executor_info["nxrt"]`, explicit fallback variants, and a deferred deterministic `.nxpkg` transport archive that extracts before mmap-based loading.
**Why:** This preserves ORT/onnxruntime-genai interoperability and existing EPContext semantics while adding pure-Rust runtime metadata, reproducible compiled-artifact validation, single-file transport, GenAI asset resolution, and minimal-build integration without inventing a competing executable format.

#### Source: `pris-cuda-gsc-review.md`

# Pris review — CUDA Gather / Shape / Constant (`1f3a64f`)

**Verdict: 🔴 REJECT**

**Revision owner: Deckard** (Roy is locked out as the original author).

## Blocking defect

`GatherKernel::cuda_graph_compatible()` returns `true`, but every non-empty
Gather execution performs a synchronous device-to-host index copy through
`CudaRuntime::dtoh()` and explicitly synchronizes after the launch
(`gather.rs:143-153,218,225-226`; `runtime.rs:216-221`). Host readback and
stream synchronization make this implementation unsuitable for CUDA graph
capture, so the capability declaration is false and can cause capture-time
failure. Deckard should return `false` until Gather is made capture-safe, or
redesign validation/execution to avoid host readback and synchronization.

## Verification

- NVRTC was made available before GPU execution through an equivalent
  worktree-local `libnvrtc.so` search path (the environment forbids `/tmp`
  writes).
- `cargo test -p onnx-runtime-ep-cuda --test movement_gpu -- --nocapture`:
  5 passed, 0 failed, 0 ignored. Gather numerics **actually executed**; no
  `UNSUPPORTED_PTX` skip occurred. Axis-0/int64/negative, axis-1/int32, and
  2-D indices all exact-matched independent expected tensors (max abs error 0).
- Shape exact-matched full and negative `start`/`end` slicing as 1-D int64.
- Constant exact-matched fp32 and int64 tensor attributes.
- Independent count: `CUDA_COVERED_OPS` has exactly 65 entries, 65 unique,
  with Gather/Shape/Constant each present once; exact `.len() == 65` assertion
  and coverage documentation are updated.
- Full `cargo test -p onnx-runtime-ep-cuda`: green.
- `cargo build --locked -p onnx-runtime-ep-cuda`: passed.

---

### 2026-07-16: CUDA Gather graph-compatibility re-review
**By:** Pris
**Verdict:** 🔴 REJECT
**Revision owner:** Deckard (Sebastian and Roy are locked out for this revision cycle).

**What:** Commit `0e92c672` makes Gather's declaration locally truthful:
`GatherKernel::cuda_graph_compatible()` now returns `false`, and the GPU test
asserts it. However, the capability flag is not consumed by any executor or
capture path in the workspace. The only call to `cuda_graph_compatible()` is
the new assertion in `tests/movement_gpu.rs`. Therefore Gather is not actually
excluded from graph capture by the claimed per-kernel mechanism.

**Why:** A declaration that no capture scheduler reads cannot enforce
non-capturability. This is especially unsafe because other kernels still return
`true` while calling `CudaRuntime::synchronize()` (for example Cast,
pointwise, Softmax, and several normalization kernels). Deckard must wire the
flag into the real capture eligibility/partitioning path and add a test proving
non-compatible kernels prevent or split capture; alternatively, remove the
unsupported compatibility claim until such a consumer exists and mark every
synchronizing/D2H kernel non-capturable.

**Verification:** Gather numerics genuinely executed with the worktree-local
NVRTC loader path: all 3 Gather reference cases exact-matched (max abs error
0); the full movement test binary passed 5/5. `CUDA_COVERED_OPS.len() == 65`
and `docs/CUDA_COVERAGE.md` still reports 65. Full
`cargo test -p onnx-runtime-ep-cuda` passed, and
`cargo build --locked -p onnx-runtime-ep-cuda` passed.

---

### 2026-07-16: CUDA Gather/Shape/Constant gating review
**By:** Pris
**Verdict:** 🔴 REJECT

**What:** Commit `e029d81` resolves the prior graph-capture blocker:
`subgraph_graph_capturable()` is a public consumer of each kernel's
`cuda_graph_compatible()` value, uses correct all-kernels/AND semantics, and the
GPU Gather tests prove a real Gather kernel makes the gate return false.
Gather's synchronous D2H validation and stream synchronization are therefore
reported honestly. Gather, Shape, and Constant GPU numerics all pass.

The remaining blocker is the stale source-derived audit in
`docs/CUDA_COVERAGE.md:146-176`. It still claims 63 CUDA registry pairs, 62
advertised names, 51 shared CPU pairs, 43 shared standard-domain ops, and lists
Constant/Gather/Shape among 50 CUDA gaps. Those claims contradict the registry
and the same document's updated 65-op statements.

**Why:** The minimal fix is documentation-only: update the audit to 66 registry
pairs, 65 advertised names, 54 shared CPU pairs, 46 shared standard-domain ops,
and 47 standard-domain gaps; add Constant/Gather/Shape to the shared list and
remove them from the gaps list. Deckard, Roy, and Sebastian remain locked out;
any other eligible agent may make this minimal correction.

**Verification:** `cargo test -p onnx-runtime-ep-cuda` ran on the H200 with
NVRTC available: 113 passed, 0 failed, 0 skipped. The movement GPU binary passed
5/5, including all three Gather references and the public capture gate check.
`CUDA_COVERED_OPS.len() == 65` passed, and `git diff --check` passed.

#### Source: `pris-cuda-nbits-review.md`

# Pris review — CUDA MatMulNBits int4 GEMM

**Timestamp:** 2026-07-16T03:05:00Z
**Commit:** `4997676701b7ec3922669ae54ec84728ac6b6d84`
**Verdict:** 🟢 APPROVE

- Inspected the NVRTC dequantization kernel, cuBLASLt GEMM path, registry, tests, and coverage documentation.
- GPU numerical tests actually executed on the H200 with CUDA 12.6 NVRTC; neither test skipped. Maximum absolute errors against the independent in-test reference were `3.814697266e-6` (block 32, symmetric zero point 8, K=45) and `4.577636719e-5` (block 128, explicit packed zero points, batched rank-3 input, K=173).
- Decode follows standard low-nibble-first INT4 packing, per-output/per-block scales, default zero point 8, packed explicit zero points, and correctly bounds the final non-multiple K block. Optional group indices select scale/zero-point groups after host-side range validation.
- The implementation explicitly accepts only f32 activations/scales/output and returns a dtype error for fp16 rather than producing incorrect output. The f32-only/full-f32 temporary dequantization limitation is documented.
- `CUDA_COVERED_OPS` contains exactly 62 unique names; its exact `.len()` assertion is 62. `docs/CUDA_COVERAGE.md` reports 62 advertised CUDA ops and documents `MatMulNBits`.
- Validation passed: 105 crate tests listed and the full `onnx-runtime-ep-cuda` suite passed; `cargo build --locked -p onnx-runtime-ep-cuda` passed.

#### Source: `rachael-int8-nbits-review.md`

### 2026-07-16: Review int8/VNNI MatMulNBits accuracy level 4
**By:** Rachael (Code Reviewer, Numerics)
**Verdict:** 🟡 APPROVE WITH NOTES
**Commit:** `47fbfd4d1242472b83f4c229efba2b8e28b1fce6`

**What:** Approved Sebastian's native CPU `MatMulNBits` int8-activation fast path. The new path is strictly gated by `accuracy_level == 4` and absence of `g_idx`; accuracy level 0/unset reaches the unchanged fp32 prepack/GEMV or dequant/GEMM code, preserving its operation order and bit-level behavior.

**Numerics:** Activation quantization is symmetric per row (`max_abs / 127`, round, clamp to `[-127, 127]`). The u8 `+128` representation is correctly converted back algebraically with `dot(u8, i8) - 128 * sum(weight)`. Weight unpacking follows the official contrib-op contract: earlier K is the low nibble, dequantization is `(q - zero_point) * block_scale`, absent zero point defaults to 8, and scales are per output/per K block. Padded partial-K lanes use activation 128 and weight 0 and are excluded from the block sum, so they contribute zero after correction. Each block's int32 result is dequantized by `activation_scale * block_scale`.

**Intrinsics:** x86 implementations are cfg-gated, runtime-dispatched by `is_x86_feature_detected!`, use unaligned loads only within complete 32-byte chunks, and handle remainders with the scalar implementation. The host exposes AVX-VNNI plus AVX512-VNNI/AVX512VL, so selection resolves to the AVX512-VNNI variant. The test proves selected VNNI and scalar dots are exactly equal. Non-x86/non-VNNI execution falls back to the same scalar dot.

**Tests:** The model-level accuracy-4 tests use an independent fp32 reference that unpacks/dequantizes int4 weights and then performs fp32 matmul. Coverage includes block 32 and 128, M=1 and M=3, and partial K. The prepack test additionally proves the accuracy-4 cache is populated rather than the fp32 cache. On the committed vectors, maximum errors consume about 15.3% and 3.0% of the stated tolerance, respectively; `0.05 + 5% * |reference|` is conservative but reasonable for intentionally lossy activation quantization.

**Verification:**
- `cargo test -p onnx-runtime-ep-cpu`: 400 passed, 0 failed; doc test ignored as expected.
- `cargo build --locked -p onnx-runtime-ep-cpu`: passed.
- `git diff --check 47fbfd4^ 47fbfd4`: passed.

**Non-blocking notes:** Add a model-level accuracy-4 case with explicit asymmetric packed zero points in a future change. A dedicated regression test comparing default-path output bits would make the source-level unchanged-path guarantee explicit, although inspection confirms the old fp32 branch is unchanged.

#### Source: `roy-cuda-gather-shape-constant.md`

### 2026-07-16: CUDA Gather, Shape, and Constant implementation boundary
**By:** Roy
**What:** Registered Gather, Shape, and Constant in the CUDA EP. Gather is an NVRTC axis-parametric indexed-copy kernel; Shape and Constant compute or decode their small host-resident metadata/value payloads and synchronously upload them to CUDA memory.
**Why:** Gather needs true device-side data movement for native decode. Shape has no elementwise compute and Constant's ONNX attribute payload is host metadata, so host + H2D is simpler and correct while keeping each node CUDA-covered. Gather accepts Int32/Int64 indices, validates bounds before launch, and supports negative wrapping.

#### Source: `sebastian-cuda-gather-fix.md`

### 2026-07-16: Mark CUDA Gather non-capturable
**By:** Sebastian
**What:** CUDA Gather now reports `cuda_graph_compatible() == false`; its GPU tests assert the non-capturable contract.
**Why:** Gather performs synchronous device-to-host index validation and synchronizes its CUDA stream. Preserving deterministic ONNX out-of-range errors requires that host validation, so declaring the kernel non-capturable is the clean, truthful fix until validation/error propagation can be redesigned entirely on-device.

#### Source: `sebastian-int8-nbits.md`

### 2026-07-16: Gate native int8 MatMulNBits on accuracy_level=4
**By:** Sebastian
**What:** The native CPU MatMulNBits kernel uses per-row symmetric int8 activation quantization and cached int8 weights only when `accuracy_level=4`; unset/default nodes retain the existing fp32 path. x86-64 selects AVX512-VNNI/AVX512VL, then AVX-VNNI, with a portable scalar fallback.
**Why:** This preserves default numerics while reducing decode weight bandwidth and mapping int8 dot products to VNNI. The Qwen2.5-0.5B native decode benchmark improved from 0.50 to 1.01 tok/s.

#### Source: `sebastian-matmulnbits-threading-fix.md`

### 2026-07-16: Use thread-count-aware MatMulNBits partitioning
**By:** Sebastian
**What:** Partition native CPU MatMulNBits output columns from total dot work and the active Rayon pool size. Pools up to 48 workers use smaller balanced tasks; larger pools require substantially more work per worker, which limits the 96-worker Qwen decode path to the language-head GEMV.
**Why:** The fixed 1 Mi-dot gate left 48/121 decode matmuls serial and regressed 24-worker throughput. Empirical 24/48/96-worker sweeps showed that threading all projections wins on one socket, while dispatching medium projections across the 96-worker dual-socket pool loses to serial execution.

#### Source: `zhora-matmulnbits-threading.md`

### 2026-07-16: Partition CPU MatMulNBits over contiguous N ranges
**By:** Zhora
**What:** Use the existing Rayon pool to partition int8/VNNI and fp32 M=1 MatMulNBits work into contiguous N chunks, with a 1 Mi dot-product serial threshold and at least 16 outputs per task. Larger-M int8 work partitions M and nests N partitioning only when rows underfill the pool.
**Why:** Each output column is independent, contiguous ranges preserve packed-weight locality, and measured thresholding avoids Rayon wake-up overhead on small projections while respecting the configured global pool.

#### Source: `leon-decode-profile2.md`

### 2026-07-16: Fast-path same-shape contiguous f32 Mul on CPU
**By:** Leon
**What:** The CPU elementwise `Mul` kernel now writes same-shape contiguous f32 inputs directly to a non-aliased output; broadcasting, striding, other dtypes, and aliasing retain the generic materializing path.
**Why:** Fresh 24-thread Qwen2.5-0.5B INT4 profiling found `Mul` at 11.73% of node time after MatMulNBits threading. Removing temporary allocations and copies reduced `Mul` from 3.119 to 0.249 ms and improved median decode throughput from 40.50 to 44.22 tok/s (+9.2%).

#### Source: `rachael-silu-fuse.md`, `sebastian-silu-review.md`

### 2026-07-16: Lower exact Sigmoid self-multiply pairs to fused SiLU
**By:** Rachael; reviewed by Sebastian
**What:** The native CPU executor lowers only single-consumer `x * Sigmoid(x)` patterns to `com.microsoft::Silu`; the CPU kernel uses a non-aliasing contiguous-f32 direct-write path and retains the general strided fallback. The rewrite handles either Mul operand order and rejects graph-output or multi-consumer Sigmoid values. Sebastian approved commit `682c93d`; commit `d116a96` adds the multi-consumer negative test.
**Why:** Qwen2.5-0.5B has this exact pattern in all 24 MLP layers. Fusion removes 24 intermediate tensors and dispatches, reducing the former 6.55% Sigmoid share to zero while preserving greedy output tokens. CPU/session tests passed (409/112), and interleaved benchmarks improved from 44.45 to 47.64 tok/s.

#### Source: `roy-matmulnbits-gemv.md`

### 2026-07-16: Stream packed int4 weights in M=1 VNNI GEMV
**By:** Roy
**What:** Route symmetric block-32, no-g_idx, M=1 `MatMulNBits` at `accuracy_level=4` through a runtime-gated AVX-VNNI/AVX512-VNNI kernel that unpacks int4 weights inside the dot product. Retain the existing int8/fp32 paths for unsupported CPUs and other operator shapes.
**Why:** Steady-state profiling put 93.7% of MatMulNBits time in the threaded GEMV while activation quantization and tensor preparation were about 3% each; prepack was one-time. The prior path streamed 617.45 MB/token of expanded weights, scales, and block sums and reduced SIMD lanes every 32 elements. The fused path follows MLAS's packed-weight approach, halves the minimum stream to 308.73 MB/token, reduces paired MatMulNBits time by 14%, and preserves known-good decode tokens.

#### Source: `wallace-matmulnbits-direct-int4-review.md`

### 2026-07-16: Approve direct-int4 VNNI MatMulNBits GEMV
**By:** Wallace
**What:** 🟢 Approve Roy's `4af8646`. Added focused scalar-fallback and direct-int4 serial/parallel partial-K coverage in `c49f878`.
**Why:** Low/high nibble order, symmetric zero point 8, per-block scales, padded K tails, runtime SIMD gates, and disjoint N partitioning are correct. The full CPU EP suite passes 411/411 tests. Independent 24-thread medians measured 45.79→49.96 tok/s; 60-step profiling measured MatMulNBits 16.824→14.537 ms and 84.98%→83.41%. At 96 threads, throughput measured 15.45→28.00 tok/s. Tokens remained `[11576, 42740, 11, 358]`.

#### Source: `luv-mmnb-tiling.md`

### 2026-07-16: Keep the one-column direct-int4 GEMV
**By:** Luv
**What:** Four- and eight-column SIMD tiling were measured and reverted. The production MatMulNBits kernel remains the simpler one-column direct-int4 path.
**Why:** At 24 workers, seven interleaved runs measured 54.91 tok/s for the current kernel versus 52.67 and 52.65 tok/s for tile widths four and eight. Width eight lowered median MatMulNBits time but worsened its mean through long-tail stalls; both wider tiles regressed at 96 workers. The row-major packed-weight layout makes wider tiles open more non-contiguous weight streams while the small activation vector is already cache-resident.

#### Source: `nabil-projfusion-design.md`, `fact-checker-projfusion.md`

### 2026-07-16: Retain projection fusion as a reviewed load-time design
**By:** Nabil; verified by Fact Checker
**What:** `docs/PROJECTION_FUSION.md` proposes a conservative CPU `Executor::build` rewrite that concatenates compatible `MatMulNBits` B/scale rows along N and supplies zero-copy output views. In the inspected Qwen2.5-0.5B artifact, QKV is already packed, so the only directly available target is each gate/up pair (`N=4864|4864 → 9728`); implementation awaits user approval.
**Why:** The build seam is before planning, allocation, and kernel caching, where immutable initializers and mutable graph structure are both available. The verifier confirmed the model topology, packing math, and seam. The potential gain is approximately 2–7%, but the exact 124.6875 MiB newly constructed B+scale payload is a lower bound rather than a guaranteed RSS increase because alignment fallback copies can raise retained or peak memory.

#### Source: `deckard-numa-decode.md`, `holden-numa-decode-review.md`, `sebastian-numa-parse-fix.md`

### 2026-07-16: Keep a safe opt-in decode-only Rayon thread cap
**By:** Deckard; safety revision by Sebastian; reviewed by Holden
**What:** `ONNX_GENAI_CPU_DECODE_THREADS` selects a dedicated Rayon pool only for CPU `MatMulNBits` with `M=1`; prefill and the default global pool remain unchanged. Missing, empty, invalid, zero, negative, and overflowing values fall back to the existing path; valid positive requests are capped at `available_parallelism()`.
**Why:** On the dual-socket Xeon 8480C, a small pinned worker count substantially improves decode (about 60 tok/s at six workers versus roughly 50 tok/s at the 24-thread default). The initial implementation was rejected because unsafe environment values could abort inference or provoke excessive thread creation; the pure bounded resolver closes those cases and was cleared after 413 tests.
## 2026-07-16 — Python bindings wave

#### Sources: `rachael-nxrt-eager-genai.md`, `holden-nxrt-engine-threading.md`, `sebastian-nxrt-mutex-fix.md`, `holden-nxrt-rereview.md`, `batty-onnxrs-pybind.md`, `freysa-onnx-rs-python-review.md`, `deckard-onnxrs-patharg-fix.md`, `freysa-onnxrs-rereview.md`

### Ship thread-safe `nxrt` eager and genai Python APIs
**By:** Rachael; threading revision by Sebastian; reviewed by Holden
**What:** `onnx-runtime-python` now ships default-on, independently selectable `nxrt.eager` and `nxrt.genai` features (the webserver remains excluded). `nxrt.eager` exposes dispatch, opset, and cache statistics; `nxrt.genai.Engine` exposes directory loading, tokenization, generation, and callback streaming. The original `unsendable` `RefCell` Engine wrapper was rejected because cross-thread use raised PyO3 `PanicException`; the merged revision stores the Rust engine in a `Mutex`, releases the GIL for engine work, and uses `try_lock` to return actionable `RuntimeError`s on contention or callback re-entry. `Cargo.lock` was refreshed.
**Why:** Python users need local inference, single-op execution, and generation without pulling in a server or GPU toolkit. A sendable, fail-fast Mutex wrapper supports free-threaded Python safely while preserving engine invariants. Holden cleared the revision after the locked build and all 19 Rust binding tests passed. The final merged commit is `41d8c31`.

### Ship the `onnx_rs` serialization Python module with lossless path handling
**By:** Batty; path revision by Deckard; reviewed by Freysa
**What:** The new abi3-py310 `onnx-rs-python` crate imports as `onnx_rs` and exposes an opaque `Model`, binary load/save, and `to_*`/`from_*` text, JSON, and TextProto codecs. The initial binding was rejected for an `exists()` preflight, lossy path conversion, and swallowing exceptions from `__fspath__`. The merged revision accepts lossless `PathBuf` values, maps actual I/O error kinds to Python exceptions, propagates `__fspath__` failures, and adds six Python path regressions.
**Why:** `onnx_rs` avoids collision with the established `onnx` package and retains onnx-rs's stateless codec convention. Native path preservation and direct filesystem errors are required for valid non-UTF-8 Unix paths and accurate error reporting. Freysa cleared the revision after targeted Rust coverage and all six Python regression tests passed. The final merged fix is `5b348b5`.

#### Sources: `roy-decode-perf2.md`, `wallace-rmsnorm-review.md`

### 2026-07-16: Make fused residual RMSNorm allocation-free on native decode
**By:** Roy; reviewed by Wallace
**What:** `SkipSimplifiedLayerNormalization` now uses a contiguous f32 fast path that borrows input/skip/gamma and writes the residual sum and normalized result directly to their distinct output buffers. The scalar broadcast, strided, statistics-output, and non-f32 fallbacks remain unchanged.
**Why:** The current steady decode profile is MatMulNBits 81.93%, RMSNorm 6.13%, GQA 5.01%, SiLU 3.47%, Add 2.25%, and Mul 0.65%; RMSNorm was the largest low-risk non-matmul lever. RMSNorm fell 1.113→0.742 ms/step (-33.3%), and five alternating 24-token pairs improved 44.20→46.45 tok/s (+9.1% paired median), with identical greedy tokens `[11576, 42740, 11, 358]`. Wallace independently reproduced a 32.7% RMSNorm reduction and +6–16% decode gains, confirmed output disjointness and fast-path guards, and cleared the change. All 413 CPU EP tests passed.

#### Sources: `leon-gqa-perf.md`, `sebastian-gqa-review.md`

### 2026-07-16: Streamline contiguous f32 GQA decode writes
**By:** Leon; reviewed by Sebastian
**What:** The CPU `GroupQueryAttention` M=1 decode path writes contiguous f32 attention and present K/V outputs directly. The guarded path retains the generic narrowing/strided writer for prefill, non-f32, and strided outputs; BSH attention output, BNSH present K/V layout, RoPE, KV append/capacity, and head grouping remain unchanged.
**Why:** Avoiding a redundant f32 narrowing allocation and per-element strided walk reduced GQA from 0.865 to 0.690 ms/step and raised decode from 54.38 to 58.44 tok/s (+7.5%) with exact tokens `[11576, 42740, 11, 358, 614, 264, 3405, 911]`. Sebastian independently measured 0.883→0.457 ms/step and 51.58→59.42 tok/s with the same eight tokens, cleared the change, and confirmed 413 CPU EP tests pass. Merged to `origin/main` as `1fdd1ec`.

### 2026-07-16: Close the CPU elementwise sweep and retain the native CUDA decode design
**By:** Scribe
**What:** The standalone contiguous-f32 residual `Add` fast path is closed as a negative result: its small local improvement regressed paired decode and was reverted (`3c0788a`). `docs/NATIVE_CUDA_DECODE.md` (`b416b7f`, amended by `33beb8d`) is the fact-checked design for the GPU-decode frontier, recommending `Arc<dyn ExecutionProvider>` polymorphism through five milestones: EP-polymorphic execution, target-compatible coverage, O(1) device-resident KV, CUDA graph replay, and performance tuning. It is awaiting user greenlight and is not implemented.
**Why:** The residual `Add` candidate improved Add only 1.3% while reducing end-to-end decode 1.5%, closing the CPU elementwise sweep. Fact Checker verified 14 central design claims, including the executor's concrete CPU EP, object-safe EP dispatch, coverage gaps, packed-QKV CUDA GQA blocker, and O(capacity) CUDA KV update. The amendment requires a real non-null CUDA stream and serialized ownership for non-Send/Sync CUDA graphs; virtual-dispatch cost remains an unmeasured assumption.

#### Sources: `zhora-onnxrs-test-port.md`, `coco-onnxrs-testport-review.md`

### 2026-07-16: Expand onnx-rs upstream text-format coverage within supported IR boundaries
**By:** Zhora; reviewed by Coco
**What:** Merged commit `23e4995` expands `crates/onnx-rs/tests/text_format_port.rs` from 6 to 16 upstream-derived cases, covering attribute kinds, initializers, supported dtypes, node domains, multi-opset models, text round-trips, JSON/TextProto codecs, and malformed input. `cargo test -p onnx-rs` passes 89 tests (72 unit, 16 integration, 1 doctest); the reviewer found no ignored or vacuous cases.
**Why:** The port exercises real parser and IR behavior while documenting current grammar/IR gaps: model-local functions; sequence, optional, and sparse types; complex, int2, and uint2 dtypes; and typed tensor-payload literals remain unsupported.

### 2026-07-16: Complete native CUDA decode Milestone 1a
**By:** Deckard; reviewed by Holden
**What:** Merged `f795d45` makes the native session executor EP-polymorphic: `Executor`, `Executor::build`, and `KernelCache::get_or_create` now retain `Arc<dyn ExecutionProvider>` / `&dyn ExecutionProvider` rather than concrete CPU types. `auto_detect_cpu_ep` still constructs the same initialized `CpuExecutionProvider`; CPU-only session construction, kernel dispatch, host buffers, and device selection are otherwise unchanged.
**Why:** This behavior-preserving seam is Milestone 1a of `docs/NATIVE_CUDA_DECODE.md`. CPU validation passed all 413 EP tests and reproduced exact tokens `[11576, 42740, 11, 358, 614, 264, 3405, 911]`; 59.99 tok/s versus the 58.44 tok/s reference is within noise. Holden cleared the one-file refactor: EP virtual calls remain cache-miss-only, outside steady-state kernel execution, with no CUDA/device branching, unsafe, downcasting, or dispatch-policy changes. M2—device tensors and on-device op coverage—is next, gated on user decisions in the design doc (GPU floor, KV capacity, hard-fail policy, and graph ownership), plus packed-QKV CUDA GQA and O(1) device-KV prerequisites.

#### Sources: `deckard-cuda-m1a.md`, `holden-cuda-m1a-review.md`

#### Sources: `luv-cuda-silu-layernorm.md`, `holden-cuda-silu-layernorm-review.md`

### 2026-07-16: Close CUDA M2 SiLU and standard SimplifiedLayerNormalization coverage
**By:** Luv; reviewed by Holden
**What:** CUDA now registers f32 `com.microsoft::Silu` and standard-domain f32 `ai.onnx::SimplifiedLayerNormalization`, both at since-version 1, matching CPU EP domain/opset/dtype coverage. SiLU uses stable `x * sigmoid(x)` evaluation; the standard-domain normalization reuses the RMS-style CUDA implementation.
**Why:** The executor fuses `x * Sigmoid(x)` to `com.microsoft::Silu`, and Qwen2.5-0.5B-int4 includes standard-domain `SimplifiedLayerNormalization`; both had blocked target-model CUDA coverage. Independent GPU references confirmed SiLU and RMS-style normalization (including `InvStdDev`) with zero reported maximum error for normalization. Holden cleared `16c1e92`; the CUDA suite passed 114 tests, and registration parity is exact without silently claiming unsupported dtypes.

#### Sources: `roy-cuda-gqa-packed.md`, `sebastian-gqa-packed-review.md`, `wallace-gqa-packed-fix.md`, `sebastian-gqa-packed-rereview.md`

### 2026-07-16: Complete CUDA M2 packed-GQA and alias-aware device-KV append under strict lockout
**By:** Roy; rejected and re-reviewed by Sebastian; repaired by Wallace
**What:** CUDA `com.microsoft::GroupQueryAttention` now splits packed `Q|K|V` input when standalone K/V are absent, applies device RoPE to Q and current K, and appends only current-token K/V at absolute `past_len + s` when fixed-capacity past/present device caches alias. The target 14/2/64 geometry preserves seven Q heads per KV head. The completed regression performs a real three-token packed prefill into a device cache, then two pointer-aliased decode appends, validating outputs and full caches against an independent split-half-RoPE, repeat-KV, causal-softmax oracle at `<1e-3`.
**Why:** Sebastian rejected Roy's original `ad73494` artifact because it host-seeded the cache rather than exercising packed prefill-to-aliased decode, and its required GPU test failed `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (5/6). Strict lockout was enforced: Roy did not repair the rejected artifact; Wallace supplied `2b6d654`. The root cause was global, not GQA-specific: CUDA 13.3 NVRTC emits PTX ISA 9.3 that driver 580.105.08 rejects. The shared compile/load path now retries that specific failure with a native `sm_90` CUBIN and applies the successful fallback to subsequent modules; diagnostics remain explicit. Sebastian cleared the repaired merge (`4a34c66`): all 6 GQA tests and all 114 CUDA tests executed and passed, with non-GQA coverage confirming the fallback is global and non-regressing. This establishes reliable native-CUDA kernel loading on the H200 CUDA 13.3 / driver 580.105.08 environment and completes the M2 GQA prerequisite.

#### Sources: `deckard-cuda-executor-wiring.md`, `holden-cuda-executor-wiring-review.md`, `leon-cuda-seqat-scan-fix.md`, `holden-cuda-executor-wiring-rereview.md`

### 2026-07-16: Complete CUDA Native Decode Milestone 2 end-to-end under strict lockout
**By:** Deckard; rejected and re-reviewed by Holden; safety revision by Leon
**What:** Native CUDA decode is now an opt-in, end-to-end executor path, merged as `5c0f05f`. CUDA selection via `DevicePreference::Gpu` or explicit CUDA creates device-resident initializer and execution buffers, performs H2D at graph inputs and D2H at materialized outputs, and stamps views with the selected device; CPU remains the default. Qwen2.5-0.5B-int4 CPU and CUDA decode produced the identical eight-token sequence `[11576, 42740, 11, 358, 614, 264, 3405, 911]`.
**Why:** Holden rejected Deckard's initial wiring commit `1a2deca` despite its correct Qwen parity because CUDA `SequenceAt` could pass host storage to a CUDA kernel and CUDA `Scan` could host-write a device allocation. Deckard was locked out of the revision; Leon repaired both hazards in `5c0f05f` by synchronously uploading non-host `SequenceAt` elements into correctly stamped CUDA buffers and retaining `Scan` slices as host staging tensors that the child executor uploads through `copy_from_host`. The new substantive CUDA control-flow parity test covers `SequenceAt -> Add` and CUDA `Scan` against CPU. Holden independently re-reviewed and cleared the revision: no replacement lifetime, synchronization, or teardown defect was found; the safety test passed, session CPU tests passed 112/112, CUDA EP tests passed 117/117, and engine retained only the 18 known missing-asset failures. This completes M2 after the earlier packed-QKV/GQA, O(1) append, SiLU, SimplifiedLayerNormalization, and SM90 CUBIN-fallback sub-waves. M3 device-resident persistent KV, decode-efficient CUDA `MatMulNBits`, and M4 CUDA graph capture remain.

---
## 2026-07-16 — CUDA SM-general, device-KV M3, and onnx-rs full-spec review wave

**Wave status:** CUDA SM-general (`b56c5cb`) and device-resident KV-cache M3 (`398c536`) are merged and reviewer-cleared. The onnx-rs full-spec serde claim is **rejected**; Zhora is locked out and Batty is revising against current ONNX IR13 with authoritative native text serialization. Do not treat the bound stale-schema implementation as full-spec complete.

#### Source: `holden-smgeneral-review.md`

### 2026-07-16: Review — SM-general CUDA NVRTC
**By:** Holden
**What:** 🟢 CLEAR — runtime NVRTC PTX and native-CUBIN targets are derived from the selected CUDA device's compute capability; the unsupported-PTX fallback remains active and targets that same device architecture.
**Why:** `CudaContext::new(ordinal)` is queried for major/minor attributes, producing `compute_{major}{minor}` and `sm_{major}{minor}` (including SM60, 75, 86, 90, 100, and 120 tests), with no production hardcoded SM90 target. On the 8× H200 host (`compute_cap=9.0`, CUDA 13.3, driver 580.105.08), a fresh-target full suite executed 117 tests with 0 skip markers and passed; all 6 GQA tests passed, demonstrating that the known unsupported-PTX path successfully recompiles/loads native `sm_90` CUBIN. The exact minimal environment initially exposed an unrelated missing cuDNN loader path; adding the installed cuDNN/cublas library directories yielded the clean full run.

#### Source: `sebastian-m3-devkv-review.md`

### 2026-07-16: Review — M3 device-resident CUDA KV cache
**By:** Sebastian
**What:** 🟢 CLEAR commit `60be8a0`. M3 supplies 48 persistent CUDA KV allocations as both graph past inputs and aliased present outputs, suppresses bound-output materialization, preserves their addresses across decode/rewind, separates physical capacity from the mask-derived valid prefix, and performs no KV host transfers during the 16-token real-model smoke.
**Why:** `DeviceIoBinding` counters increment on explicit binding reads/writes (covered by the session alias test), while the KV bindings are only passed as external device pointers; executor inspection found no KV copy/materialization path, and the real-model test observed all KV H2D/D2H counters at zero before/after decode and rewind. GQA detects identical past/present pointers and appends at `past_lengths[b] + s`; attention loops only to `seqlens_k + 1`, while the total-sequence scalar remains physical capacity. The capacity-128/valid-5 GPU test compares against an exact-capacity reference and would reject the old equality rule or capacity-wide attention; all 7 real GQA GPU tests passed.

The divergence is **pre-existing CUDA numerical drift, not an M3 regression**. An isolated 16-token empirical run of parent `dc8eaf4` (M2 host-round-trip KV) produced CPU `[11576,42740,11,358,614,264,3405,911,279,330,34,1027,11766,11635,1,323]` and CUDA `[11576,42740,11,358,614,264,3405,911,279,330,9707,4337,1,2025,304,356]`. M3 produced those exact same CPU and CUDA vectors, including the first mismatch at token index 10 (`34` vs `9707`). Therefore stale capacity, mask/cursor errors, aliasing, or M3 RoPE position drift did not introduce or move the divergence; it remains in the shared M2 CUDA numerical path.

Fresh-target validation passed after adding the installed cuDNN directory to `LD_LIBRARY_PATH` (the exact two-variable invocation initially failed only because `libcudnn.so.9` was not on the loader path). Targeted results: GQA GPU `7 passed`; session external-binding alias/materialization test `1 passed`; real Qwen CPU/CUDA smoke `1 passed` in 82.16s; standard clippy completed with pre-existing warnings only.

#### Source: `rachael-onnxrs-review.md`

### 2026-07-16: Review — onnx-rs full-spec serde
**By:** Rachael
**What:** 🔴 REJECT. The change is lossless for its bound schema, but that schema is stale and the native readable codec delegates ordinary full-spec content to an authoritative opaque protobuf blob. Zhora is locked out; Batty should revise by updating the vendored proto to current ONNX and implementing genuine readable serialization for common ONNX structures.
**Why:** `build.rs` binds `crates/onnx-runtime-loader/proto/onnx.proto3`. Its pre-bump checksum exactly matches ONNX v1.16.2 (IR10); commit `06a2423` manually added only `FLOAT4E2M1` and relabeled it IR11. Official IR11 already includes multi-device support, while current ONNX v1.22.0/main is IR13 and defines `DeviceConfigurationProto`, `NodeDeviceConfigurationProto`, `IntIntListEntryProto`, `ShardingSpecProto`, `ShardedDimProto`, `SimpleShardedDimProto`, `ModelProto.configuration`, `NodeProto.device_configurations`, `FLOAT8E8M0`, `UINT2`, and `INT2`. Thus both “no multi-device protos” and “no INT2/UINT2” are artifacts of the stale vendor, not current-spec facts. The checklist mentions a proto-update follow-up but still claims completion and does not cover that required scope. The native codec emits the entire retained `ModelProto` as base64 for every proto-backed model (`text/ser.rs:63-65`), and the parser immediately returns it while ignoring the visible graph body (`text/de.rs:91-93`); functions, sparse/type/list attributes, optional/sequence/map details, and other common fields are therefore not genuinely textually serialized. Positive checks: all 83 tests pass from a fresh target, no ignored tests or remaining `Unsupported`/`todo!`/`unimplemented!` markers were found in the reviewed codec/test paths, byte-equality assertions cover the stale bound schema, and `cargo clippy -p onnx-rs` passes. `--all-targets -D warnings` additionally finds one test-only `field_reassign_with_default` warning.

#### Source: `zhora-onnxrs-full-spec.md`

### 2026-07-16: onnx-rs full bound-spec serde coverage
**By:** Zhora
**What:** Replaced divergent JSON/TextProto allowlists with descriptor-driven protobuf codecs and retained the complete ModelProto beside the runtime graph projection. Added lossless full-spec custom-text round trips via a protobuf sidecar, completed bound dtype/attribute IR variants, and added exhaustive descriptor/payload/structural tests.
**Why:** The previous codecs rejected or silently omitted valid ONNX fields. The vendored `crates/onnx-runtime-loader/proto/onnx.proto3` must be the single source of truth so schema additions cannot create another hand-maintained coverage gap.

## Bound schema inventory and serde status
All messages below are **DONE** in JSON, protobuf TextFormat, readable text (lossless retained-proto sidecar), and binary model I/O:

- [x] `AttributeProto` — every scalar/message/list field, including `ref_attr_name`, `doc_string`, `sparse_tensor`, `tensors`, `graphs`, `sparse_tensors`, `type_protos`.
- [x] `ValueInfoProto` — `name`, recursive `type`, `doc_string`, `metadata_props`.
- [x] `NodeProto` — inputs/outputs/name/op/domain/overload/attributes/doc/metadata.
- [x] `TrainingInfoProto` — initialization/algorithm graphs and both binding lists.
- [x] `ModelProto` — header/opsets/graph/metadata/training/functions.
- [x] `StringStringEntryProto`.
- [x] `TensorAnnotation` quantization mappings.
- [x] `GraphProto` — dense and sparse initializers, docs, IO/value-info, quantization annotations, metadata.
- [x] `TensorProto` and nested `TensorProto.Segment` — every payload field, external-data/location, docs, metadata.
- [x] `SparseTensorProto`.
- [x] `TensorShapeProto` and nested `Dimension`, including oneof and denotation.
- [x] `TypeProto` plus nested `Tensor`, `Sequence`, `Map`, `Optional`, `SparseTensor`; recursive combinations are round-tripped.
- [x] `OperatorSetIdProto`.
- [x] `FunctionProto` — signature, required/default attributes, nodes, docs, opsets, domain/overload, value-info, metadata.

All bound enums are **DONE**:
- [x] `Version`.
- [x] `AttributeProto.AttributeType`: all 15 values, including tensor/sparse/type lists.
- [x] `TensorProto.DataType`: `UNDEFINED`, FLOAT, UINT8, INT8, UINT16, INT16, INT32, INT64, STRING, BOOL, FLOAT16, DOUBLE, UINT32, UINT64, COMPLEX64, COMPLEX128, BFLOAT16, all four FLOAT8 variants, UINT4, INT4, FLOAT4E2M1.
- [x] `TensorProto.DataLocation`: DEFAULT and EXTERNAL.
- [x] `OperatorStatus`.

## Dtype evidence
- [x] Added `Undefined`, `Complex64`, and `Complex128` to runtime `DataType`; schema and readable-text maps now cover every bound enum value.
- [x] Typed-field and raw-data fixtures cover every concrete bound dtype; STRING uses its required typed `string_data` representation.
- [x] Packed UINT4/INT4/FLOAT4E2M1 layouts are covered with packed-byte payloads.
- [x] BFLOAT16 and FLOAT8E4M3FN/FNUZ/E5M2/E5M2FNUZ are covered.
- [x] The bound IR-v11 proto has **no INT2 or UINT2** (DataType ends at FLOAT4E2M1=23); tests assert their descriptor absence.

## Former TextProto Unsupported fields
- [x] All 24 `UnsupportedMessage`/`UnsupportedScalar` entries were eliminated by deleting the per-codec allowlist. JSON and TextProto now share the generated `FileDescriptorSet` through `prost-reflect`.
- [x] Sparse initializers/attributes, function/training declarations, nested graphs, metadata, tensor segment/docs/metadata, overload/ref attributes, type denotations, and quantization annotations all round-trip byte-exactly.

## Multi-device status
- [x] **Not present in the bound proto, not silently skipped.** The descriptor contains no `DeviceConfigurationProto`, `NodeDeviceConfigurationProto`, `ShardingSpecProto`, or other device/multi-device message. Tests assert these names are absent. Supporting them requires first updating the vendored ONNX proto; descriptor-driven codecs will then pick them up automatically.

## Representation note / deferred runtime execution
- [x] Serde coverage is complete for the bound proto.
- [ ] Runtime execution semantics for sparse initializers, training graphs, and retained local-function declarations remain outside the tensor-centric execution IR. `Model` therefore retains the exact source `ModelProto`; `make_graph_authoritative()` explicitly opts out and documents that those non-execution fields are then dropped. This is not a serde deferral.
- [ ] The readable DSL represents fields outside its native graph syntax through a lossless base64 protobuf sidecar rather than adding bespoke surface syntax for every protobuf field. Native pretty syntax is cosmetic follow-up; round-trip coverage is complete.

## Validation
- `cargo test -p onnx-rs`: 62 unit + 4 full-spec integration + 16 ONNX-port integration + 1 doctest passed.
- `cargo build -p onnx-rs`: passed.
- `cargo clippy -p onnx-rs`: passed clean.
- Wider workspace check reached unrelated `onnx-runtime-cpuinfo` and stopped because its vendored `cpuinfo/CMakeLists.txt` is absent in this checkout; all changed crates compiled before that failure.

#### Source: `wallace-cuda-sm-general.md`

### 2026-07-16: Make CUDA NVRTC architecture targeting device-general
**By:** Wallace
**What:** `CudaRuntime::new` now queries `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` and `_MINOR` through cudarc and caches the reported capability plus derived NVRTC targets. PTX uses `compute_{major}{minor}` and the native fallback uses plain `sm_{major}{minor}` (for example, 9.0 becomes `compute_90` / `sm_90`; no implicit `sm_90a`). Capability query failures or invalid values return actionable CUDA EP errors, with no SM default and no maximum-SM cap. The existing `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` retry and process-local fallback flag are preserved.
**Why:** Hardcoding the H200's SM90 made the CUDA EP non-portable. Pure unit tests now cover SM 60, 75, 80, 86, 89, 90, 100, and 120 derivation. All eight installed H200s report compute capability 9.0. With the dynamic path, `cargo test -p onnx-runtime-ep-cuda` passed all 117 unit/GPU tests, including activations and all six GroupQueryAttention GPU tests; `cargo build` passed and `cargo clippy` completed with only three pre-existing warnings.

#### Source: `roy-cuda-device-kv.md`

### 2026-07-16: Native CUDA decode uses persistent device-resident KV bindings
**By:** Roy
**What:** Added `DecodeCudaState` beside `NativeDecodeSession` with a logical cursor, configurable fixed capacity, one persistent CUDA allocation per key/value tensor, and a fixed-capacity device attention mask. Session `DeviceIoBinding` supplies externally owned device inputs, aliases graph outputs to the same allocation, tracks physical and logical shapes separately, and suppresses bound output materialization. CUDA GQA now treats `total_sequence_length` as physical capacity while `seqlens_k + 1` is the valid prefix; in-place append remains O(new tokens). Rewind/reset move the cursor and update only the mask suffix, never KV bytes.
**Why:** M3 requires stable KV addresses and no context-sized KV PCIe traffic so later CUDA graph capture has fixed pointers. The default capacity is 4096 tokens; `NativeDecodeSession::load_with_cuda_kv_max_len` and `ONNX_GENAI_CUDA_KV_MAX_LEN` override it, and overflow returns a clean pre-launch error. The 16-token Qwen GPU test asserts all 48 KV pointers remain identical across generation and rewind and aggregate KV binding H2D/D2H counters remain exactly zero. Full CPU/CUDA greedy parity past token 10 is deferred: an origin/main M2 probe and M3 both match the first 10 tokens (required first eight `[11576,42740,11,358,614,264,3405,911]`) and diverge afterward, so this is a pre-existing CUDA numerics gap rather than a device-KV regression.

### 2026-07-16: Sub-4-bit IQ/MXFP4 quant — design + CPU proto
**By:** Bryant
**What:** Added `docs/SUB4BIT_QUANT.md` with exact llama.cpp IQ1/IQ2/IQ3 and OCP MXFP4 layouts, recommended linear `MatMulNBits` plus format-explicit block-quantized MatMul/MoE ops, Mobius capability wiring, and an ORT issue draft. Extended the CPU EP's registered `com.microsoft::MatMulNBits` kernel to execute standard linear `bits=2` weights through the f32 correctness path, with partial-block parity and bit-packing tests.
**Why:** Enables huge-MoE sub-4-bit weights to remain compressed and makes top-k expert offload practical on smaller machines without misinterpreting IQ grid bytes as linear integers.
**Follow-ups:** Grid-codebook IQ kernels, full MXFP4 MatMul, direct 2-bit/IQ CPU optimization, CUDA kernels, Mobius export and EP-capability wiring, expert residency/offload, and a fused block-quantized MoE op.

### 2026-07-16: Review — sub-4-bit 2-bit CPU MatMulNBits
**By:** Leon
**What:** 🟢 CLEAR — commit `a5e62d2` correctly adds the CPU affine `bits=2` dequant-to-f32 baseline without changing the effective int4 or accuracy-level-4 int8 behavior. The design document is technically plausible and clearly separates affine int2 from IQ/MXFP4 native formats and deferred optimized/MoE work.
**Why:** Packing is LSB-first and matches the existing int4 convention: `bit_offset = within_block * bits`, followed by shift and mask, yields four 2-bit values per byte in crumb order bits `[1:0]`, `[3:2]`, `[5:4]`, `[7:6]`; int4 still yields low nibble then high nibble. The absent zero point is `1 << (bits - 1)`, so int2 uses `zp=2`; explicit int2 zero points use the same LSB-first two-bit packing, four blocks per byte, matching the ORT layout. Concrete trace: packed `0b11_10_01_00` (`0xE4`) decodes to `q=[0,1,2,3]`; with scale 1 and `zp=2`, weights are `[-2,-1,0,1]`. Activations `[1,10,100,1000]` produce `-2 - 10 + 0 + 1000 = 988`, exactly asserted by the packing test; remaining `0xAA` bytes decode to `q=2` and contribute zero.

Parity integrity is sound: the bits=2 test uses `M=3, K=45, N=7, block_size=32`, exercising two blocks including a partial final block, computes an independently dequantized f32 reference, and checks every output at absolute tolerance `1e-5`. Fixtures are native Rust IR (`Graph`/`Node`/`Model` plus proto encoding), not `onnx.helper`. Optimized int4 paths are explicitly gated by `self.bits == 4`; the generic int4 shifts/masks remain equivalent to the previous low/high-nibble logic. Unsupported `bits=8` is cleanly rejected while missing `bits` still defaults to 4.

Fresh-target validation:
`cargo test -p onnx-runtime-ep-cpu`: `415 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`; doc-tests `0 passed; 0 failed; 1 ignored`.
`cargo build -p onnx-runtime-session -p onnx-genai-engine`: `Finished dev profile` successfully in 11.53s.

### 2026-07-16: DeepSeek-V4-Flash mobius export
**By:** Chew
**What:** Draft PR https://github.com/onnxruntime/mobius/pull/405 adds `deepseek_v4`/GGUF `deepseek4` registration, V4 projections, Hyper-Connections, hash/sqrt-softplus MoE, dense attention fallback, 4/8-bit MatMulNBits graph support, GGUF mappings, tests, and runtime aliases.
**Why:** V4 differs substantially from V3 MLA; this lands the largest weight-compatible standard backbone while keeping unsupported compressed sparse attention explicit rather than guessing its runtime contract.
**Follow-ups:** Add CSA/HCA compression/indexer/attention-sink and MTP runtime paths; add direct packed dynamic int2/int1 and MXFP4 expert support via a runtime custom op plus mobius EP capabilities and/or an ORT issue; optimize split-GGUF expert repacking.

### 2026-07-16: GLM-5.2 mobius export
**By:** Tyrell
**What:** Draft PR https://github.com/onnxruntime/mobius/pull/404 adds `glm_moe_dsa` / `glm-dsa` config, registry, full-attention MLA+MoE graph export, GGUF tensor/config mapping, split-K/V fusion, streamed expert import, Q4/Q8 MatMulNBits normalization, runtime aliasing, tests, and a design note.
**Why:** IndexShare DSA and the GLM-specific improved MTP layer require new cross-layer sparse-attention and speculative-decoding contracts. The PR therefore lands a coherent full-attention backbone first rather than a partially wired sparse path.
**Follow-ups:** Add IndexShare and improved MTP; implement an efficient selected-expert runtime op with native IQ1_M/IQ2_XXS/IQ3_XXS/IQ4_XS support and expose it through Mobius EP capabilities; otherwise pursue the drafted ORT issue for sub-4-bit MatMulNBits/sparse-MoE support. Q4 requantization is the current fallback.

### 2026-07-16: GLM-5.2 IndexShare DSA + MTP export
**By:** Mariette
**What:** PR https://github.com/onnxruntime/mobius/pull/404 now includes commit `590c7da`, implementing portable IndexShare DSA with the official shared-indexer schedule and packed index-key cache, plus the complete layer-78 improved-MTP graph, HF/GGUF mappings, ORT GenAI artifacts, dense-attention fallback, tests, and documentation.
**Why:** Standard ONNX ops preserve DSA selection numerics without a mandatory custom op. MTP is exported as a separate artifact because ORT GenAI does not yet natively orchestrate GLM's speculative iteration state.
**Follow-ups:** Add a selected-token sparse-attention runtime kernel for the advertised FLOP reduction; add independent indexer-cache/control state and native MTP orchestration for `index_share_for_mtp_iteration`; keep IQ1/IQ2/IQ3 quantization in its separate workstream.

### 2026-07-16: onnx-rs proto bump to IR13 + authoritative native text
**By:** Batty
**What:** Landed ONNX v1.22.0 / IR13 bindings, FLOAT8E8M0/UINT2/INT2 runtime dtype and packed-storage support, and lossless multi-device serde for DeviceConfigurationProto, NodeDeviceConfigurationProto, ShardingSpecProto, ShardedDimProto, SimpleShardedDimProto, and IntIntListEntryProto across JSON, TextProto, and native text. Native text commit `67f60c0` replaces the whole-model base64 override with an explicit protobuf-TextFormat residual block; DSL-represented fields are removed from the residual and reconstructed from the readable body, so body edits are authoritative.
**Why:** Addresses Rachael 🔴: the previous binding was stale IR10/partial IR11 and native text ignored readable edits in favor of an opaque source proto.
**Remaining:** No known gaps in the requested serde scope. Upstream v1.22.0 attaches multi-device data at `ModelProto.configuration` and `NodeProto.device_configurations`; it defines no `GraphProto` configuration field. Training execution semantics remain out of scope, while any present training proto stays losslessly preserved by descriptor-driven codecs.

### 2026-07-16: Re-review — onnx-rs IR13 proto + authoritative native text
**By:** Rachael
**What:** 🔴 REJECT. The stale-proto defect is fixed, but the native TextFormat residual can still silently override a readable DSL edit. Zhora and Batty remain locked out; Leon should revise the residual merge and add edit-authority regressions.
**Why:** The vendored proto's SHA-256 exactly matches upstream ONNX v1.22.0, declares IR13, and is compiled by loader `build.rs` through `protox`/`prost-build` into the bindings and descriptor used by onnx-rs. `DeviceConfigurationProto`, `NodeDeviceConfigurationProto`, `ShardingSpecProto`, `ShardedDimProto`, `SimpleShardedDimProto`, and `IntIntListEntryProto` are present; upstream attaches configuration at `ModelProto.configuration` and `NodeProto.device_configurations` (there is no GraphProto configuration field). Descriptor-driven JSON/TextProto and the native residual all pass byte-exact full-spec coverage. FLOAT8E8M0/UINT2/INT2 map to 24/25/26; all legacy/new dtypes have typed/raw round-trip coverage. The existing 2-bit test preserves packed bytes, and an independent pack/unpack check round-tripped UINT2 `[0,1,2,3,1] -> [e4,01]` and INT2 `[-2,-1,0,1,-2] -> [4e,02]`.

The positive `readable_body_edits_are_authoritative_over_extensions` test proves only a top-level graph-name edit. A counter-probe changed the emitted readable attribute `types = <1 types>` to `types = <2 types>`; parsing still returned one `TypeProto`. The printer exposes this count (`text/ser.rs:338`) and the parser constructs the edited count (`text/de.rs:432-444`), but `merge_attribute` explicitly leaves residual `TypeProtos` and `SparseTensors` untouched (`text/extensions.rs:261-267`). Thus the residual remains authoritative over readable fields. Leon should make readable list cardinality authoritative while retaining only non-readable payload details, audit every emitted placeholder similarly, and add failing-then-passing edit tests.

The scoped skip grep found no `Unsupported`, `todo!`, `unimplemented!`, `unreachable!`, or `#[ignore]`. Fresh-target `cargo test -p onnx-rs` passed 62 unit + 5 full-spec + 16 port + 1 doctest (84 total); strict all-target clippy passed; dependent `onnx-runtime-session` and `onnx-genai-engine` builds passed.

### 2026-07-16: onnx-rs native text made authoritative
**By:** Deckard
**What:** The residual merge now starts from readable attributes, uses readable repeated cardinalities for tensor/sparse/type lists, treats edited tensor/type/graph fields as authoritative, and restores only omitted payload/metadata. Opaque byte placeholders use an internal parse sentinel so replacing them with readable strings wins. The full-spec regression edits graph signatures, nested graphs, tensor/type/sparse attributes, opaque strings, and list cardinalities while retaining an exact unedited round-trip.
**Why:** Addresses Rachael 🔴: the residual overrode readable TypeProtos/SparseTensors and could also override opaque-string edits and projected TypeProto edits.

### 2026-07-16: onnx-rs native-text authoritative re-review (3rd)
**By:** Rachael
**What:** 🟢 CLEAR. Verified readable DSL edits win for model headers, graph names/signatures, nodes and attributes, tensor dtype/shape, nested graph signatures, opaque strings, and tensor/sparse/type list cardinalities. The exact adversarial `<1 types>` → `<2 types>` case passes while preserving the first omitted TypeProto payload. Unedited native-text round-trip remains byte-exact.
**Why:** The residual now strips DSL-expressible fields and merge starts from parsed native attributes/cardinalities, restoring only omitted payload and metadata. `cargo test -p onnx-rs` passed all 84 tests, `cargo test -p onnx-runtime-loader` passed, and `cargo clippy -p onnx-rs -- -D warnings` passed.

### 2026-07-16: BlockQuantizedMatMul — MXFP4 + IQ scaffold
**By:** Joi
**What:** Added the private `com.github.onnxruntime.genai::BlockQuantizedMatMul` v1 CPU op with native GGUF blocks, strict shape/layout validation, optional bias, constant-weight dequant caching, and f32 GEMM. MXFP4 is fully implemented with OCP E2M1/E8M0 semantics and llama.cpp nibble layout; IQ4_NL is the first fully implemented IQ/codebook format. IQ1/IQ2/IQ3 and IQ4_XS are recognized but explicitly rejected until audited tables land. Rust dequant/matmul tests and an ONNX IR Python fixture cover the implementation.
**Why:** Enables correctness-first execution of unsloth/llama.cpp native block weights without misinterpreting IQ or MXFP4 bytes as affine NBits, unblocking sub-4-bit GGUF model integration.
**Follow-ups:** Import audited llama.cpp golden vectors and grid tables for IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, and IQ4_XS; add direct CUDA kernels, Mobius capability/export wiring, GGUF-to-ORT MXFP4 layout parity, and fused `BlockQuantizedMoE` execution.

### 2026-07-16: BlockQuantizedMatMul review
**By:** Leon
**What:** 🟢 CLEAR commit `0307138`. Hand-verified MXFP4's OCP E2M1 table, E8M0 scaling/NaN handling, and llama.cpp low-nibble→`j` / high-nibble→`j+16` layout; traced exponent `128` with byte `0xd7` to `12.0` and `-6.0`. Verified IQ4_NL's exact llama.cpp 16-entry codebook and nibble order; traced fp16 scale `0.5` with byte `0xf0` to `-63.5` and `56.5`. Confirmed IQ1_S, IQ2_XXS, IQ3_S, and IQ4_XS all fail kernel creation as recognized-but-unimplemented. CPU EP tests passed 420/420 and the Python ONNX IR fixture passed 1/1.
**Why:** The implementation matches llama.cpp commit `b15ca938` for block sizes, tables, scale conversion, and packed element order, while incomplete IQ formats fail closed instead of decoding silently. `cargo clippy -p onnx-runtime-ep-cpu --lib` completed with only existing warnings and none in Joi's new code; `--all-targets` reaches an unrelated pre-existing denied approximate-constant lint in `elementwise.rs`.
## 2026-07-16 — CUDA sub-4-bit and Mobius export wave

### CUDA MatMulNBits M=1 packed-int4 GEMV
**By:** Roy
**What:** Merged `1de9584`: M=1 no-`g_idx` CUDA `MatMulNBits` decode reads output-major packed int4 weights directly, applies block scales during accumulation, and retains the constrained symmetric block-32 `accuracy_level=4` route. Unsupported shapes and M>1 continue through the established fallback.
**Why:** Avoiding a full f32 weight materialization and parallelizing output dots improves CUDA decode by approximately 68–96% without changing packed-nibble, zero-point, scale, or fallback contracts. Wallace 🟢 verified H200 parity, adversarial layout cases, 120 CUDA tests, and unchanged Qwen tokens.

### Audited IQ4_XS, IQ3_S, and IQ2_XXS CPU decoding
**By:** Bryant
**What:** Merged `f6c530f`: `BlockQuantizedMatMul` now decodes llama.cpp IQ4_XS, IQ3_S, and IQ2_XXS super-blocks. IQ2_XXS sign/grid and IQ3_S grid tables are imported exactly from llama.cpp `b15ca938`; IQ1_S remains explicitly rejected.
**Why:** This expands correctness-first native sub-4-bit execution while retaining fail-closed behavior for formats without audited tables. Leon 🟢 grid-verified block decoding and table contents against upstream.

### Preserve MXFP4 and IQ4_NL blocks in Mobius exports
**By:** Pris
**What:** Mobius PR [#406](https://github.com/onnxruntime/mobius/pull/406) exports MXFP4 and IQ4_NL as `com.github.onnxruntime.genai::BlockQuantizedMatMul` v1 nodes with complete GGUF blocks in uint8 initializers and `format`, `K`, `N`, and `block_layout_version=1` attributes.
**Why:** Non-affine native blocks must reach the runtime unchanged rather than be misread as affine `MatMulNBits`; unsupported IQ formats remain on the existing dequantize/requantize fallback.

### Weight-specific residency design awaits greenlight
**By:** Nabil
**What:** `docs/WEIGHT_OFFLOAD.md` specifies an immutable mmap → bounded host cache → bounded VRAM cache subsystem with expert/page leases, fused-MoE paging, and Resource Governor sub-budgets.
**Why:** Immutable external weight ranges have alignment, representation, transfer, and in-flight lifetime requirements unlike token-indexed mutable KV. The design reuses tiering concepts without coupling to KV structures or connector APIs. **Awaiting user greenlight; not implemented.**

### DeepSeek-V4-Flash MTP and CSA sidecars
**By:** Chew
**What:** Mobius PR [#405](https://github.com/onnxruntime/mobius/pull/405) was updated at `7e26e6e` with the official 0/4/128 CSA compression schedule, compressor and sparse-index tensors, attention sinks, a dense causal fallback, and a separate MTP sidecar plus orchestration contract.
**Why:** Current ONNX Runtime lacks native compressed-KV construction, sparse-index/cache operations, and iterative shared-state MTP orchestration. The export preserves official tensors and usable dense/MTP artifacts without fabricating sparse runtime semantics.
## 2026-07-16 — Full IQ-family CUDA decode, Mobius export, and parity wave

All ten source notes below are newly merged after deduplication. Decisions archive check: `.squad/decisions.md` exceeds 20 KB, but no entries are older than 30 days relative to 2026-07-16; no archive was created.

#### Source: `roy-cuda-sub4bit-gemv.md`

### 2026-07-16: CUDA sub-4-bit decode GEMV
**By:** Roy
**What:** Added an M=1 CUDA `BlockQuantizedMatMul` kernel for `mxfp4` and `iq4_nl`, registered under `com.github.onnxruntime.genai` v1. CUDA placement accepts only static M=1 shapes, layout version 1, and those two formats; IQ4_XS, IQ3_S, IQ2_XXS, and prefill remain on CPU. GPU tests compare random blocks to the CPU op, verify every decoded weight bit-exactly through one-hot GEMV, and cover optional bias.
**Why:** Decode is the hot path for dynamic-quant MoE models, and direct packed-block GEMV avoids materializing f32 weights while preserving the CPU op's GGUF semantics.

Hand-verified blocks:
- MXFP4 exponent byte `128`, quant byte `0xd7`: low nibble `7` at element 0 decodes to `12.0`; high nibble `13` at element 16 decodes to `-6.0`.
- IQ4_NL fp16 scale `0.5`, quant byte `0xf0`: low nibble `0` at element 0 decodes to `-63.5`; high nibble `15` at element 16 decodes to `56.5`.

Performance was not measured. Runtime gaps: IQ4_XS/IQ3_S/IQ2_XXS and M>1 are intentionally unsupported by CUDA and route to CPU; the full CUDA test suite needs the installed Python cuDNN directory on `LD_LIBRARY_PATH`.

#### Source: `wallace-cuda-sub4bit-review.md`
# 2026-07-16 — CUDA sub-4-bit GEMV review

**By:** Wallace
**Verdict:** 🟢 CLEAR
**Reviewed:** `cd02810` (`cuda-sub4bit-gemv`)

## What I verified

- The CUDA registration uses `com.github.onnxruntime.genai::BlockQuantizedMatMul` v1 and defaults `block_layout_version` to 1, matching the CPU reference.
- MXFP4 and IQ4_NL use the CPU reference's exact block sizes, tables, output-major block addressing, low-nibble elements 0–15 / high-nibble elements 16–31 ordering, and per-block scale application.
- MXFP4 E8M0 handling matches the CPU implementation for ordinary exponents, the two f32-subnormal boundary encodings, and reserved `0xff` NaN.
- IQ4_NL loads its fp16 scale little-endian and uses the exact 16-entry signed codebook.
- Weight decode and accumulation are f32. CUDA changes reduction order versus CPU GEMM, but the one-hot tests prove every decoded weight bit-exact against CPU, while the random GEMV test covers f32 accumulation with bias, partial K, multiple blocks, and multiple columns.
- CUDA claims only static M=1 nodes in formats `mxfp4` and `iq4_nl`. IQ4_XS, IQ3_S, IQ2_XXS, all other formats, dynamic decode shapes, and M>1 return `KernelMatch::Unsupported`; the CPU EP registers the same op/domain and remains a placement candidate.
- The kernel is SM-general NVRTC source. No `sm_90` or other fixed architecture is introduced.

## Hand-traced blocks

- **MXFP4:** scale byte `128` gives `e8m0_half_scale = 2^(128-128) = 1`. Quant byte `0xD7`: low code `7` indexes doubled E2M1 value `12`, so element 0 is `12.0`; high code `13` indexes `-6`, so element 16 is `-6.0`.
- **IQ4_NL:** fp16 scale `0.5`, quant byte `0xF0`: low code `0` indexes `-127`, giving element 0 `-63.5`; high code `15` indexes `113`, giving element 16 `56.5`.

## Test results

- H200 detected: compute capability 9.0, driver 580.105.08.
- `cargo build -p onnx-runtime-ep-cuda`: passed.
- `cargo test -p onnx-runtime-ep-cuda`: passed all 124 listed tests after adding the installed cuDNN directory to `LD_LIBRARY_PATH`; the initially requested environment alone could not locate `libcudnn.so.9`.
- `block_quantized_matmul_gpu`: 4/4 passed, including exact per-weight CPU/GPU bit comparisons and the two hand-traced blocks.
- `cargo test -p onnx-runtime-ep-cpu block_quantized`: 9/9 passed, 415 filtered out.

#### Source: `bryant-iq2-iq3-formats.md`
# 2026-07-16 — CPU IQ2_XS, IQ2_S, and IQ3_XXS decoding

**By:** Bryant

**What:** Added fail-closed `BlockQuantizedMatMul` CPU decoding for GGUF `IQ2_XS`,
`IQ2_S`, and `IQ3_XXS`. The exact llama.cpp
`b15ca938ad00aa6b3ee6c2edda7363fd02826b18` `ggml-common.h` tables are vendored:
`iq2xs_grid` (512 `u64` entries), `iq2s_grid` (1024 `u64` entries), and
`iq3xxs_grid` (256 `u32` entries). The shared `ksigns_iq2xs` table remains the
sign source. FNV-1a fingerprints are pinned in tests.

The upstream 256-element block layouts are used without transcoding:

- `IQ2_XS`: `fp16 d; u16 qs[32]; u8 scales[8]` — 74 bytes.
- `IQ2_S`: `fp16 d; u8 qs[64]; u8 qh[8]; u8 scales[8]` — 82 bytes.
- `IQ3_XXS`: `fp16 d; u8 qs[96]` — 98 bytes.

Known blocks were hand-traced against `ggml-quants.c`:

- `IQ2_XS`: `d=2`, scales `0x21`, grids `{0,511,1,510}`, sign indices
  `{0,1,2,3}` produce subscales `{0.75,1.25}` and the exact 32-value test vector.
- `IQ2_S`: `d=2`, scales `0x21`, low indices `{0,0,0,255}`, `qh=0xe4`, and
  explicit signs `{0x00,0x81,0x82,0x03}` select grids `{0,256,512,1023}` with
  subscales `{0.75,1.25}`.
- `IQ3_XXS`: `d=2`, scale nibble `2`, grid pairs
  `{0,255},{1,254},{2,253},{3,252}`, and sign indices `{0,1,2,3}` produce
  `db=2.5` and the exact 32-value test vector.

`IQ1_S` and `IQ1_M` remain rejected because their shared 2048-entry grid requires
a separate import and audit.

**Why:** These formats are native grid/codebook encodings, not affine integers.
Matching llama.cpp's serialized fields, lookup tables, sign application, and
subscale math preserves GGUF interoperability while keeping unaudited IQ1 data
on the explicit reject path.

#### Source: `leon-iq2-iq3-review.md`

# IQ2_XS / IQ2_S / IQ3_XXS CPU decode review

- **Date:** 2026-07-16T17:29:45Z
- **By:** Leon
- **Verdict:** 🟢 CLEAR
- **Reviewed:** Bryant commit `b56f02d` on `cpu-iq2-iq3-formats`

## Evidence

- Compared every numeric entry against llama.cpp commit
  `b15ca938ad00aa6b3ee6c2edda7363fd02826b18`:
  `iq2xs_grid` 512/512 exact, `iq2s_grid` 1024/1024 exact,
  `iq3xxs_grid` 256/256 exact, and `ksigns_iq2xs` 128/128 exact.
  The Rust `1 << j` sign-bit checks exactly reproduce `kmask_iq2xs =
  [1, 2, 4, 8, 16, 32, 64, 128]`. Spot checks included first, quartile,
  midpoint, three-quarter, and final entries of each grid.
- Confirmed upstream layouts: IQ2_XS `2 + 64 + 8 = 74` bytes, IQ2_S
  `2 + 64 + 8 + 8 = 82` bytes, and IQ3_XXS `2 + 96 = 98` bytes, each
  representing 256 weights.
- Independently hand-traced the asserted first 32 weights for all three tests
  from the upstream grids and sign table. IQ2_XS reproduced the 9-bit indices,
  sign indices 0–3, and `db={0.75,1.25}` values; IQ2_S reproduced
  `qh=0xe4` indices `{0,256,512,1023}`, explicit signs, and the same subscales;
  IQ3_XXS reproduced paired grids, packed sign indices 0–3, and
  `db=2*(0.5+2)*0.5=2.5`. All asserted vectors matched exactly.
- Decode field slicing, index extraction, sign application, and nibble
  subscales match `dequantize_row_iq2_xs`, `dequantize_row_iq2_s`, and
  `dequantize_row_iq3_xxs`.
- IQ1_S and IQ1_M remain recognized but rejected during kernel creation, with
  explicit test coverage.
- Required test gate passed: **13 passed, 0 failed, 415 filtered out**.

#### Source: `bryant-iq1-formats.md`

### 2026-07-16: Audited IQ1_S and IQ1_M CPU decoding
**By:** Bryant
**What:** Implemented both IQ1_S and IQ1_M in CPU `BlockQuantizedMatMul`; neither format remains deferred. Vendored all 2048 `iq1s_grid` u64 entries byte-exact from llama.cpp commit `b15ca938` `ggml-common.h`; little-endian-byte FNV-1a is `0x6703ed863501ae2e`. Confirmed IQ1_S is 50 bytes (`fp16 d`, `qs[32]`, `qh[8]` as u16) and IQ1_M is 56 bytes (`qs[32]`, `qh[16]`, `scales[8]`, with global fp16 bits embedded in scale high nibbles).
**Why:** The CPU runtime now matches upstream `dequantize_row_iq1_s` and `dequantize_row_iq1_m` for 11-bit grid assembly, ±0.125 deltas, odd 3-bit multipliers, and IQ1_M's two scales per 32-weight group. Hand-verified IQ1_S with `d=2`, `qh=0xa1c0`: `dl=10`, indices `[0,0,2047,0]`, negative delta, outputs `[-11.25,-11.25,8.75,-11.25]` per 8-value vector. Hand-verified IQ1_M with bitsliced fp16 `0x4000`, scale payload `0x001a`, qh `[0xf0,0x8f]`: `dl=[10,14]`, indices `[0,2047,2047,0]`, deltas `[+,-,-,-]`, outputs `[-8.75,8.75,12.25,-15.75]` per vector.

#### Source: `leon-iq1-review.md`
# 2026-07-16 — IQ1_S / IQ1_M CPU decode review

**By:** Leon

**Verdict:** 🟢 CLEAR

## Grid spot-check

Compared all 2048 `IQ1S_GRID` entries against `iq1s_grid` in llama.cpp commit
`b15ca938` `ggml-common.h`: counts are 2048/2048 and every `u64` is identical.
Both little-endian byte streams recompute to FNV-1a
`0x6703ed863501ae2e`. Spot checks at indices
`0, 1, 37, 255, 511, 1024, 1536, 2047` also match exactly; endpoints are
`0xffffffffffffffff` and `0x0101010101010101`.

## Layout and decode audit

- `IQ1_S` matches upstream `block_iq1_s`: fp16 `d` (2 bytes), `qs[32]`,
  then `qh[16]`, totaling 50 bytes.
- `IQ1_M` matches upstream `block_iq1_m`: `qs[32]`, `qh[16]`, then
  `scales[8]`, totaling 56 bytes. It has no standalone `d`; the fp16 scale is
  reconstructed from the high nibbles of four little-endian scale words.
- The Rust decoders exactly reproduce upstream 11-bit index assembly, signed
  `{-1,0,+1}` grid lanes, `±0.125` delta selection, IQ1_S odd subscale, and
  IQ1_M's two 3-bit odd subscales per 32-weight group.

## Hand-traced blocks

- IQ1_S: `d=2`, odd multiplier `5` gives `dl=10`; `qh=0xa1c0` produces
  indices `[0,0,2047,0]` and delta `-0.125`. Grid lanes `[-1,-1,+1,-1]`
  decode to `[-11.25,-11.25,8.75,-11.25]`.
- IQ1_M: reconstructed fp16 scale is `2`; scale bits give
  `dl=[10,14]`. `qh=[0xf0,0x8f]` produces indices
  `[0,2047,2047,0]` and deltas `[+0.125,-0.125,-0.125,-0.125]`, decoding
  to `[-8.75,8.75,12.25,-15.75]`.

## Test results

With `CARGO_TARGET_DIR=/home/justinchu/onnx-genai/target-leon`,
`cargo test -p onnx-runtime-ep-cpu block_quantized` passed:
**14 passed, 0 failed, 415 filtered out**. `git diff --check` also passed.

#### Source: `pris-mobius-iq-family-export.md`

### 2026-07-16: Export the full runtime-supported GGUF IQ family
**By:** Pris
**What:** Extended Mobius PR #406 so `BlockQuantizedMatMul` preserves raw blocks for MXFP4=39 (`mxfp4`, 32 elements/17 bytes), IQ4_NL=20 (`iq4_nl`, 32/18), IQ4_XS=23 (`iq4_xs`, 256/136), IQ3_S=21 (`iq3_s`, 256/110), IQ3_XXS=18 (`iq3_xxs`, 256/98), IQ2_XXS=16 (`iq2_xxs`, 256/66), IQ2_XS=17 (`iq2_xs`, 256/74), IQ2_S=22 (`iq2_s`, 256/82), IQ1_S=19 (`iq1_s`, 256/50), and IQ1_M=29 (`iq1_m`, 256/56). All nodes use domain `com.github.onnxruntime.genai`, `block_layout_version=1`, and exact `K`, `N`, and lowercase `format` attributes. Added component, repacker, and end-to-end GGUF builder tests covering every format, enum ID, block size, and byte preservation.
**Why:** The CPU runtime now decodes this complete IQ set, so Mobius can retain the source GGUF representation instead of dequantizing and requantizing it. GGUF types outside this explicit table, such as Q5_0, remain on the existing repack/dequantize-requantize fallback because `BlockQuantizedMatMul` does not advertise them.

#### Source: `mariette-pris-iq-export-review.md`

### 2026-07-16: Pris full IQ-family Mobius export review
**By:** Mariette

**Verdict:** 🟢 CLEAR — Mobius commit `5705eed` matches the authoritative onnx-genai CPU `BlockQuantizedMatMul` contract exactly.

#### Format-string cross-check

| GGUF type | Mobius emits | Runtime accepts | Result |
|---|---|---|---|
| MXFP4 | `mxfp4` | `mxfp4` | exact |
| IQ4_NL | `iq4_nl` | `iq4_nl` | exact |
| IQ4_XS | `iq4_xs` | `iq4_xs` | exact |
| IQ3_S | `iq3_s` | `iq3_s` | exact |
| IQ3_XXS | `iq3_xxs` | `iq3_xxs` | exact |
| IQ2_XXS | `iq2_xxs` | `iq2_xxs` | exact |
| IQ2_XS | `iq2_xs` | `iq2_xs` | exact |
| IQ2_S | `iq2_s` | `iq2_s` | exact |
| IQ1_S | `iq1_s` | `iq1_s` | exact |
| IQ1_M | `iq1_m` | `iq1_m` | exact |

#### GGML-number and byte-size check

Cross-checked Mobius against the installed authoritative `gguf.GGMLQuantizationType` and `GGML_QUANT_SIZES`, then against the runtime's `qk()` and `block_bytes()` values.

| Format | ggml type | Elements/block | Bytes/block | Result |
|---|---:|---:|---:|---|
| `iq2_xxs` | 16 | 256 | 66 | exact |
| `iq2_xs` | 17 | 256 | 74 | exact |
| `iq3_xxs` | 18 | 256 | 98 | exact |
| `iq1_s` | 19 | 256 | 50 | exact |
| `iq4_nl` | 20 | 32 | 18 | exact |
| `iq3_s` | 21 | 256 | 110 | exact |
| `iq2_s` | 22 | 256 | 82 | exact |
| `iq4_xs` | 23 | 256 | 136 | exact |
| `iq1_m` | 29 | 256 | 56 | exact |
| `mxfp4` | 39 | 32 | 17 | exact |

`preserve_native_blocks()` validates the complete raw byte count as `N * ceil(K / elements) * bytes` and reshapes without modifying bytes. No stride or off-by-one mismatch found.

#### Node contract and fallback

- Exported op/domain: `com.github.onnxruntime.genai::BlockQuantizedMatMul`, matching runtime registration.
- `K`, `N`, lowercase `format`, and `block_layout_version=1` are all emitted.
- Native selection is an exact ten-type allowlist keyed by ggml enum number. Every other `GGMLQuantizationType` returns no native spec and follows the existing repack or dequantize/requantize path; unsupported formats cannot silently reach the custom op.

#### Tests

- Required gate: `40 passed, 1 skipped, 4368 deselected in 7.74s`.
- `ruff check`: `All checks passed!`
- `git diff 5705eed^ 5705eed --check`: passed.

#### Source: `sapper-cuda-numeric-drift.md`

### 2026-07-16: CUDA RMS reduction FMA caused token-10 drift
**By:** Sapper

**What:** A temporary executor trace dumped every f32 node output for CPU and CUDA on the real Qwen2.5-0.5B int4 decode. Both paths use f32 KV, f32 attention/softmax accumulation, f32 RoPE caches, and fp32 MatMul accumulation; no fp16 KV rounding was involved. Replaying the first tolerance-failing `MatMulNBits` (node 128) with identical CPU input produced bit-exact CPU/CUDA output, ruling out its reduction as the source.

The first recurrent mismatch is `SkipSimplifiedLayerNormalization` at layer 0, token index 1 (node 14): its inputs are bit-exact, but CUDA differs by `9.536743e-7` in 885/896 outputs. NVRTC contracts `ss += sv * sv` into FMA, while CPU performs separately rounded f32 multiply/add. The same issue exists in `SimplifiedLayerNormalization`. Initial token-0 SiLU noise is only `1.192093e-7` and does not explain the known argmax split.

Before the fix, at generated token index 10 the first `atol=1e-5, rtol=1e-4` failure is layer 6 `GroupQueryAttention` (node 84): max abs `1.270249e-4`, 214/896 elements. Layer 7 GQA then amplifies accumulated KV differences to max abs `3.993874e-2`; final logits differ by max abs `1.310799`, and the narrow CPU top-2 margin (`5.833e-3`) flips `34` CPU to `9707` CUDA.

**Root cause:** Real bug, not benign fp16/KV accumulation: unintended CUDA FMA contraction in the sequential RMS-square reductions. The later GQA/logit split is amplification of this recurrent state error.

**Fix:** Commit `de3c556` on branch `cuda-numeric-drift` uses `__fmul_rn` + `__fadd_rn` in CUDA RMS/SkipRMS reductions and adds exact FMA-sensitive GPU regressions. Real-model parity extends from token indices 0..9 to 0..11. The remaining first mismatch is index 12 (`11766` CPU vs `16` CUDA), so residual backend drift still needs a separate follow-up rather than an accepted exact-parity claim.

**Evidence:** Full `onnx-runtime-ep-cuda` suite passed; direct CUDA crate build passed; both new exact numeric tests passed; real Qwen CPU/CUDA smoke passed with the strengthened 12-token assertion. Fixed CUDA sequence begins `[11576,42740,11,358,614,264,3405,911,279,330,34,1027]`.

#### Source: `wallace-cuda-fma-drift-review.md`
# 2026-07-16 — CUDA RMS reduction FMA drift review

**By:** Wallace
**Verdict:** 🟢 CLEAR

## What I verified

- Reviewed `de3c556`. Both CUDA RMS reductions retain f32 and the CPU reference's serial, left-to-right accumulation order.
- `__fmul_rn` followed by `__fadd_rn` explicitly rounds the square before the addition, preventing NVRTC FMA contraction without lowering precision, enabling fast math, or hard-coding an SM architecture.
- The regression vector genuinely distinguishes the operations: separate mul/add produces `0x422e4301`; fused `fmaf` produces `0x422e4302`.
- `native_decode.rs` only strengthens the existing real-model parity assertion from 10 to 12 matching tokens; it is directly tied to the kernel fix.

## GPU verification

Verified on NVIDIA H200, compute capability 9.0, driver 580.105.08.

- `cargo build -p onnx-runtime-ep-cuda`: passed.
- `cargo test -p onnx-runtime-ep-cuda`: passed after adding the installed cuDNN library directory to `LD_LIBRARY_PATH`; the requested bare environment initially reached the GPU tests but failed only because `libcudnn.so.9` was not discoverable.
- `simplified_layer_norm_does_not_contract_square_accumulation`: passed on GPU with exact output and inverse-RMS equality.
- `skip_simplified_layer_norm_does_not_contract_square_accumulation`: passed on GPU with exact output, inverse-RMS, and residual-sum equality.
- `native_decode::tests::native_cuda_qwen_decode_matches_cpu_tokens`: passed against the installed Qwen model, confirming CPU/CUDA token parity through token index 11.

## Secondary-source note

The accepted residual mismatch beginning at token index 12 is likely seeded by the accuracy-level-4 `MatMulNBits` decode path: CUDA reduces per-block scaled dot products across warp lanes, while the CPU AVX/VNNI path accumulates eight scaled partial-dot lanes and horizontally sums them afterward. Those distinct f32 multiplication and accumulation orders can still introduce smaller backend drift. Follow up separately; it does not invalidate this RMS reduction fix.
## 2026-07-16T19:05:18+0000 — CPU BlockQuantizedMatMul prefill and CUDA decode-drift closure

**Sources:** `joi-block-quant-perf.md`, `leon-block-quant-perf-review.md`, `sapper-matmulnbits-acc4-drift.md`, `wallace-silu-acc4-drift-review.md`

### CPU BlockQuantizedMatMul prefill optimization accepted (`5010261`)
**By:** Joi; reviewed 🟢 by Leon

**What:** Dequantization is parallelized over independent K block-row panels for every MXFP4/IQ format. MXFP4, IQ4_NL, and IQ4_XS use bit-exact AVX2 unpackers; all remaining formats retain their scalar reference decoder. Adaptive Rayon row tasks improve the shared generic blocked GEMM while preserving each output element's K accumulation order. The existing GEMM seam retains optional oneDNN routing.

**Why:** The former output-major scalar decode and fixed 64-row GEMM task underused CPU parallelism for M=64 prefill. At M=64/K=4096/N=4096 on a 96-core Xeon 8480C, MXFP4/IQ4_NL decode improved 5.4×/7.6× and generic matmul improved 32.3×/34.6×. All ten formats match scalar reference bits; default and oneDNN CPU EP suites each passed 430 tests.

### CUDA↔CPU decode drift bounded and accepted (`5c7dcc9`)
**By:** Sapper; reviewed 🟢 by Wallace

**What:** The token-12 divergence was caused by layer-0 fused SiLU operation order, not MatMulNBits: CUDA now follows CPU's branch-stable expression with explicitly rounded f32 operations. CUDA acc4 scale/accumulation boundaries also use explicit round-to-nearest operations. CPU/CUDA greedy parity now holds through token index 15.

**Why:** A remaining K=4864 accuracy-level-4 reduction-tree difference is bounded at max absolute `1.9073486e-5` and first amplifies to a token divergence at index 16. Serializing the GPU reduction to emulate CPU exactly costs 8.4% decode throughput, so the parallel warp reduction remains the accepted tolerance. H200 validation passed all 128 CUDA EP tests, exact SiLU-order regression, acc4 comparison, and 16-token parity coverage.
## 2026-07-16T19:27:57+0000 — CUDA IQ super-block GEMV and shared quantization tables

#### Source: `roy-cuda-iq-superblock-gemv.md`
# 2026-07-16 — CUDA IQ super-block GEMV

**By:** Roy

**What:** Extended the static M=1 CUDA `BlockQuantizedMatMul` GEMV with bit-exact on-device decoding for `iq4_xs`, `iq2_xxs`, `iq3_xxs`, `iq2_xs`, `iq2_s`, and `iq3_s`. Existing `mxfp4`/`iq4_nl` support is unchanged. `iq1_s`, `iq1_m`, and every M>1 shape remain CPU-routed.

**Why:** These six 256-element super-block formats cover the prioritized mixed-IQ MoE decode path while preserving fail-closed fallback for the delta-bearing IQ1 layouts. Shipping only formats proven bit-exact against the CPU reference avoids silent model corruption.

**Shared grids:** Added `onnx-runtime-quantization`, containing the audited llama.cpp b15ca938 IQ grids/sign table moved directly from the CPU implementation. CPU decoders import those constants, and CUDA source generation serializes the same Rust constants into device `__constant__` arrays, preventing CPU/GPU table drift. Source-body comparison and existing FNV fingerprints verified all seven moved tables unchanged.

**Hand-verified blocks:**
- IQ4_XS: fp16 scale `0.5`, encoded six-bit sub-scale `34` gives factor `2`; codebook index `0` decodes element 0 to `-127.0`.
- IQ2_XXS: fp16 scale `2.0`, grid/sign/metadata index `0` gives sub-scale `0.25`; grid magnitude `8` decodes element 0 to `2.0`.
- Random blocks for every GPU-supported format passed CPU comparison; one-hot GEMV proved every decoded element bit-exact.

**Deferred:** `iq1_s` and `iq1_m` stay on CPU because their 2048-entry signed grid, delta term, and IQ1_M bitsliced fp16 scale deserve a separate final audit. No performance benchmark was measured in this correctness pass.

**Validation:** `cargo build -p onnx-runtime-ep-cuda`; full CUDA suite 128/128 passed with the installed Python cuDNN library path; CPU `block_quantized` tests 15 passed, 1 benchmark ignored.

#### Source: `leon-quant-crate-refactor-review.md`
# 2026-07-16T19:21:34Z — Quantization crate CPU refactor review

**By:** Leon
**Verdict:** 🟢 CLEAR
**Reviewed:** Roy commit `4ec30be` on `cuda-iq-superblock-gemv`
**Scope:** CPU behavior and shared-table extraction only

## Table identity

Compared parsed values, order, declared lengths, and little-endian byte streams from merge-base `5c7dcc9` against `crates/onnx-runtime-quantization/src/lib.rs`. All moved constants are byte-identical:

- `IQ1S_GRID`: 2048 `u64`; FNV-1a `0x6703ed863501ae2e`
- `IQ2XS_GRID`: 512 `u64`; FNV-1a `0xc9b1ee61e79909bd`
- `IQ2S_GRID`: 1024 `u64`; FNV-1a `0x123edd38a3b62b90`
- `IQ3XXS_GRID`: 256 `u32`; FNV-1a `0xdfa5dc83d6a155d5`
- `IQ2XS_SIGNS`: 128 `u8`; FNV-1a `0xf19ba8f8c3292ba5`
- `IQ2XXS_GRID`: 256 `u64`; FNV-1a `0xbb4ee025b5ac6e8e`
- `IQ3S_GRID`: 512 `u32`; FNV-1a `0xfa37020c25b44829`

The CPU-local `IQ4_NL_CODEBOOK` is unchanged byte-for-byte. The implicit `kmask` sequence remains `1 << j`, yielding `[1, 2, 4, 8, 16, 32, 64, 128]`.

## CPU op unchanged

`block_quantized_matmul.rs` replaces the old include/local table definitions with imports from `onnx-runtime-quantization`. After mechanically accounting for that extraction, the only residual differences are rustfmt import ordering and one blank line; decode, validation, matmul, and test logic are unchanged. Joi's runtime AVX2 selection and MXFP4/IQ4_NL/IQ4_XS SIMD decoders remain intact.

## Build and tests

Using `CARGO_TARGET_DIR=/home/justinchu/onnx-genai/target-leon`:

- `cargo build -p onnx-runtime-ep-cpu`: passed
- `cargo test -p onnx-runtime-ep-cpu block_quantized`: 15 passed, 0 failed, 1 ignored
- `cargo test -p onnx-runtime-quantization`: passed, including doc tests
- `cargo build -p onnx-runtime-quantization`: passed standalone
- `cargo tree -p onnx-runtime-quantization --depth 1`: only the crate itself; its manifest has no dependencies
- `git diff --check`: passed

#### Source: `wallace-cuda-iq-superblock-review.md`

# CUDA IQ super-block GEMV review

**Date:** 2026-07-16T19:21:34+00:00
**By:** Wallace
**Review target:** `cuda-iq-superblock-gemv` at `4ec30be`
**Verdict:** 🟢 CLEAR (GPU scope)

## Per-format decode check

All six CUDA decoders match the CPU decoder's 256-element serialized layout and arithmetic:

- **IQ4_XS (136 bytes):** fp16 `d` at 0..2, 16 high scale bits at 2..4, four low-scale bytes at 4..8, and 128 nibble bytes at 8..136. CUDA reconstructs each six-bit factor as `low | high << 4`, subtracts 32, and applies the IQ4_NL codebook with the same nibble order as CPU.
- **IQ2_XXS (66 bytes):** fp16 `d`, then eight 8-byte group records containing four grid indices followed by little-endian scale/sign metadata. CUDA uses the top metadata nibble for `d * (0.5 + scale) * 0.25`, each seven-bit sign-table index, and the matching 8-byte grid lane.
- **IQ3_XXS (98 bytes):** fp16 `d`, 64 grid-index bytes, then eight little-endian metadata words. CUDA selects the two four-lane grids per vector, uses the full eight-bit sign mask across both grids, and computes `d * (0.5 + scale) * 0.5`.
- **IQ2_XS (74 bytes):** fp16 `d`, 64 bytes of little-endian packed 16-bit values, then eight scale bytes. CUDA splits each packed value into a nine-bit grid index and seven-bit sign-table index and applies the correct scale nibble with `* 0.25`.
- **IQ2_S (82 bytes):** fp16 `d`, 32 low-index bytes, 32 explicit sign bytes, eight high-index bytes, and eight scale bytes. CUDA assembles all ten grid-index bits and applies the explicit sign mask and correct scale nibble.
- **IQ3_S (110 bytes):** fp16 `d`, 64 low-index bytes, eight high-index bytes, 32 explicit sign bytes, and four scale bytes. CUDA selects the correct grid half/high bit, sign bit, and odd sub-scale `d * (1 + 2 * nibble)`.

These formats have no separate `dmin` field. All fp16 conversion, sub-scale multiplication order, grid magnitude extraction, and sign application agree with the CPU implementation. The exhaustive one-hot GPU test compared every decoded depth bit-for-bit for two random blocks of every supported format.

## Hand-traced blocks

- **IQ4_XS:** `d=0.5`, `scales_h[0]=2`, low nibble `2` gives factor `(2 | 2<<4)-32 = 2`, so sub-scale is `1.0`. Quant nibble zero selects codebook `-127`; CPU and CUDA both produce weight 0 = `-127.0`.
- **IQ2_S:** for the CPU fixture with `d=2`, low indices `{0,0,0,255}`, `qh=0xe4`, signs `{0x00,0x81,0x82,0x03}`, and scale `0x21`, vector 1 assembles index `0 | 1<<8 = 256`, sub-scale `2*(0.5+1)*0.25 = 0.75`, and sign mask `0x81`. CUDA reads the same grid and produces `{-18.75,18.75,18.75,18.75,18.75,6,18.75,-6}`.

## Shared-grid storage/indexing

Rust emits the shared crate's exact arrays into NVRTC source as `__device__ __constant__` tables. Numeric u32/u64 literals preserve the shared table values, and least-significant-byte-first shifts match the crate's documented little-endian lane order. Total constant storage is about 17.6 KiB, safely below CUDA's 64 KiB constant-memory limit. H200 NVRTC compilation and all index paths succeeded.

## Dispatch and fallback

`supports_node` admits only the six new formats plus existing MXFP4/IQ4_NL when the static flattened M is exactly 1. IQ1_S, IQ1_M, dynamic M, and M>1 return `KernelMatch::Unsupported`, so placement routes them to the CPU EP. Kernel execution independently rejects M != 1. The explicit fallback test passed.

No fixed `sm_90` target was added; the kernel remains architecture-neutral and uses the runtime's live-device NVRTC target selection.

## GPU verification

Verified on NVIDIA H200, compute capability 9.0, driver 580.105.08.

- Requested direct `cargo build -p onnx-runtime-ep-cuda`: passed.
- Requested bare `cargo test`: reached and passed all four block-quantized GPU tests; the later Conv tests initially failed only because installed `libcudnn.so.9` was not on the loader path.
- Full suite rerun with `/usr/lib/python3.12/site-packages/nvidia/cudnn/lib` in `LD_LIBRARY_PATH`: **128/128 tests passed**.
- Explicit `--test block_quantized_matmul_gpu`: **4/4 passed**, including all-format random GPU-vs-CPU parity, per-depth bit-exact decode, known blocks, and fallback dispatch.
- Existing MXFP4/IQ4_NL cases remain in the bit-exact/random/known-block loops and passed.
- Existing accuracy-level-4 drift regressions passed; measured CPU/CUDA max absolute difference remains the accepted `1.9073486e-5`. RMS anti-contraction tests also passed in the full suite.

---
## 2026-07-16T19:27:57+0000 — CUDA IQ1 GEMV completion

#### Sources: `roy-cuda-iq1-gemv.md`, `wallace-cuda-iq1-review.md`

### Complete the CUDA M=1 IQ decode family with IQ1_S and IQ1_M
**By:** Roy; reviewed by Wallace
**What:** Merged `06c4c06`: static M=1 CUDA `BlockQuantizedMatMul` GEMV now decodes IQ1_S and IQ1_M using the canonical shared 2,048-entry `IQ1S_GRID`. This completes GPU decode for all ten formats: MXFP4, IQ4_NL, IQ4_XS, IQ2_XXS, IQ3_XXS, IQ2_XS, IQ2_S, IQ3_S, IQ1_S, and IQ1_M. M>1 and unknown formats remain unsupported by CUDA so they route to the optimized CPU prefill path.
**Why:** The two IQ1 formats were the remaining CPU decode fallbacks. Sharing the audited grid preserves CPU semantics and eliminates table-transcription drift.

### CUDA IQ1_S/IQ1_M review cleared
**By:** Wallace
**What:** 🟢 CLEAR on H200. The shared grid FNV-1a hash is `0x6703ed863501ae2e`; both known traces and all random/per-weight GPU-versus-CPU comparisons are correct. The CUDA suite passed 129 tests across 15 groups, and the CPU gate passed 15 tests (one ignored). No fixed SM90 target was added.
**Why:** Validation confirms IQ1 index reconstruction, scales, deltas, signs, fallbacks, and runtime-device NVRTC targeting all preserve the established contracts.
## 2026-07-16 — Real-model sub-4-bit validation and export/runtime fixes

### 2026-07-16: Decisions archive eligibility scan
**By:** Scribe
**What:** Scanned `decisions.md` (120,002 bytes) before this merge. No entries predate 2026-06-16, so no entries qualify for the >30-day archive threshold.
**Why:** The 20KB archive gate requires retaining entries that are 30 days old or newer.

### 2026-07-16: DeepSeek CSA and iterative MTP runtime design
**By:** Nabil
**What:** Added `docs/DEEPSEEK_CSA_MTP_RUNTIME.md` (`bca068c`), specifying private-v1 `CompressedSparseAttention` and `SparseKvGather` ops; metadata-declared compressed/index/carry state with checkpointed rollback; CPU-before-CUDA delivery; and a persistent MTP proposer with explicit BSHC `mtp_state`. **AWAITING USER GREENLIGHT.**
**Why:** Mobius PR #405 retains official 0/4/128 compressor/indexer tensors but currently executes zero-valued anchors plus dense sink-aware attention. The official CSA equations/cache layouts must be frozen before kernel work; existing MTP machinery cannot consume the DeepSeek sidecar state or persist it across iterations.

### 2026-07-16: First real-model sub-4-bit native generation validated
**By:** Pris
**What:** Exported `bartowski/Qwen2.5-0.5B-Instruct` IQ4_XS through Mobius PR #406 and produced coherent CPU-native output: “Paris. The capital of the United States is Washington, D. C.” The graph ran 144 `BlockQuantizedMatMul` nodes (120 IQ4_NL, 24 IQ4_XS), with one-shot probes confirming both formats executed without fallback. Scripts and `docs/benchmarks/e2e-sub4bit-validation.md` landed in `2f65135`.
**Why:** This is the first real GGUF → Mobius → onnx-genai generation proof for the sub-4-bit operator family. The run exposed mixed-scaffold selection, custom-opset import, and runtime shape-inference defects.

### 2026-07-16: Quantized matmul shape inference fixed and cleared
**By:** Sapper; reviewed by Leon
**What:** Merged `67c1e3b`, adding shared `A.shape[..-1] + [N]` output-shape inference for `com.github.onnxruntime.genai::BlockQuantizedMatMul` and `com.microsoft::MatMulNBits`, preserving A's dtype and symbolic leading dimensions. Leon confirmed both domains and integer `N` contracts, structured invalid-input errors, 2D/3D coverage, and 93 unit tests plus one doc-test.
**Why:** The missing rules were the critical runtime blocker. This removes the diagnostic patch requirement and unblocks unmodified real-model E2E and the HTTP-server path.

### 2026-07-16: Mobius #406 mixed-quant scaffold and custom opset fixes
**By:** Pris; reviewed by Mariette
**What:** PR #406 was force-pushed with Mobius commit `797fff9`: native IQ/MXFP4 presence chooses a 4-bit/block-32 scaffold for non-native tensors, allowing Q5_1 requantization while preserving native IQ4_XS bytes; `BlockQuantizedMatMul` emission registers `com.github.onnxruntime.genai` opset 1. Coverage includes mixed IQ4_XS/Q5_1/Q8_0 and serialized opset imports. Mariette independently cleared the change (238 tests); Pris's full gate passed 304 tests. The PR remains **AWAITING USER MERGE**.
**Why:** The real-model run revealed an incorrect 8-bit scaffold selection and malformed serialized ONNX lacking the custom-domain import. Pure-Q8 behavior remains unchanged (MatMulNBits only, no genai import).

**Sources:** `nabil-deepseek-csa-mtp-design.md`, `pris-e2e-sub4bit-validation.md`, `sapper-shapeinfer-quant-matmul.md`, `leon-shapeinfer-quant-review.md`, `pris-mobius406-mixedquant-opset-fix.md`, `mariette-mobius406-fix-review.md`.
## 2026-07-16T19:27:57+0000 — Native backend serving

#### Source: `deckard-engine-native-backend.md`
# 2026-07-16 — Engine native decode backend selection

**By:** Deckard

## What

- Added `EngineDecodeBackend::{Auto, Ort, Native}` to `EngineConfig`.
- `Auto` keeps ORT for existing models and, when `native-backend` is compiled,
  inspects ONNX node types and selects native execution for
  `BlockQuantizedMatMul`.
- `Engine::from_dir` can now construct a CPU `NativeDecodeSession` and route
  ordinary generation plus streaming callbacks through the shared decode loop.
- Added a server `native-backend` feature forwarding to the engine, so the
  existing server load path and serialized fallback driver can serve native
  models.
- Added selector coverage and a hermetic Engine-level native generation test.

## Why

Real IQ4_XS generation was already proven through `NativeDecodeSession`, but
`Engine::from_dir` unconditionally created an ORT session. Backend selection
closes that integration gap without changing the default path for existing
models.

## Deferred

- Native continuous batching: requests are serialized by the server fallback
  driver; no native multi-row/static-cache manager exists yet.
- Persistent multi-turn sessions and prefix/KV reuse on native execution.
- External KV connectors and paged-KV import/export for native tensors.
- Speculative decoding, draft models, prompt lookup, MTP, EAGLE-3, and shared-KV
  proposers on the native target backend.
- Pipeline and embedding execution on the native backend.
- Native device selection in `EngineConfig`; this milestone uses CPU.

#### Source: `holden-native-backend-review.md`

### 2026-07-16: Engine native-backend selector review
**By:** Holden
**Verdict:** 🔴 REJECT — Deckard is locked out of this revision; Batty should revise.

**Reviewed:** `66ec4b8c433a8a523246e423f79467da8c1cc938`

**Blocking findings:**

1. `Auto` matches only `node.op_type == "BlockQuantizedMatMul"` (`engine.rs:1953-1959`). ONNX operator identity includes domain and opset; the supported native operator is specifically `com.github.onnxruntime.genai::BlockQuantizedMatMul` v1. An otherwise ORT-compatible model using that op-type string in another domain is incorrectly routed away from ORT. Require the exact domain and supported opset, and test wrong-domain/wrong-version nodes.
2. Per-request speculative overrides are silently ignored on native. `generate_native_with_callback` validates `GenerateOptions` then directly invokes ordinary native decoding (`engine.rs:828-856`), bypassing `should_use_speculative`; therefore `GenerateOptions.speculative_mode`/`num_speculative_tokens` can request prompt lookup, MTP, EAGLE-3, draft, or shared-KV and receive non-speculative output instead of an error. Reject every unsupported request-level mode explicitly.
3. Pipeline and device selection are silently ignored rather than refused. `PipelineEngine::from_dir_with_config` never examines `config.decode_backend` (`pipeline.rs:92-100`), so explicit `Native` and Auto-detected native-only pipeline decoders still enter ORT. The native single-model path also discards the supplied `SessionOptions` and hardcodes `NativeDecodeDevice::Cpu` (`engine.rs:307-318,783-786`), silently overriding requested EP/device settings. Add clear unsupported errors (or explicit documented warning/degradation).

**Positive evidence:**

- Existing non-matching models resolve to ORT; `EngineConfig::default()` is `Auto`.
- Native state is reset before each request (`native_decode.rs:223-227`), and the server owns the engine on one driver thread, so batch-1 requests are serialized safely.
- Embeddings, persistent sessions, continuous batching, KV connectors, and engine-configured speculation have explicit native errors/fallbacks.
- The native integration test genuinely loads, generates, and exercises token callbacks, but uses explicit `Native` with a graph containing no `BlockQuantizedMatMul`; it does not protect Auto selection or the HTTP sub-4-bit path.

**Build/test evidence:**

- Engine default build: PASS.
- Engine `native-backend` build: PASS.
- Server default build: PASS.
- Server `native-backend` build: PASS.
- Native engine suite: 109 passed, 18 known missing `tiny-llm/model.onnx` fixture failures, 1 ignored.
- Targeted `native_engine`: 1 passed.
- Targeted Auto detector unit test: 1 passed, but lacks domain/opset cases.

#### Source: `batty-native-backend-fix.md`

### 2026-07-16: Native backend selector revision after Holden rejection
**By:** Batty
**What:** Addressed all three blockers from Holden's 🔴 review of Deckard's native-backend selector. Auto detection now requires the exact `com.github.onnxruntime.genai::BlockQuantizedMatMul` operator identity with domain opset version 1, with tests for the supported identity, a wrong domain, and an unsupported version. Native generation now rejects request-level draft-model, prompt-lookup, MTP, EAGLE-3, and shared-KV modes plus `num_speculative_tokens`, with generation tests covering the explicit errors. Pipeline loading now rejects explicit Native and Auto-detected native-only components, and native single-model loading rejects non-CPU `SessionOptions`; tests cover explicit/Auto pipeline refusal and non-CPU device refusal.
**Why:** The rejected implementation could route same-named foreign or unsupported operators to native execution and silently ignore speculative, pipeline, and device selections. These changes fail closed with specific errors while preserving Auto as the default and the existing ORT path for non-matching models.

#### Source: `holden-native-backend-rereview.md`

# Native backend re-review

**Date:** 2026-07-16
**By:** Holden
**Verdict:** 🟢 CLEAR

Re-reviewed commit `2ae464b5d894276f38a5855599c0c9124ea23558` against rejected base `66ec4b8`. All three blockers are resolved; no new blocker or regression was found.

## Original blockers

1. **Exact native operator identity — resolved.** Auto-detection now requires node domain `com.github.onnxruntime.genai`, op type `BlockQuantizedMatMul`, and an exact v1 opset import (`crates/onnx-genai-engine/src/engine.rs:1965-1982`). Tests cover incidental strings, correct identity, wrong node domain, and unsupported opset v2 (`engine.rs:2164-2196`).
2. **Per-request speculation rejection — resolved.** Native generation rejects every non-`None` request speculation variant (draft, prompt lookup, MTP, EAGLE-3, shared-KV) and `num_speculative_tokens` before decoding (`engine.rs:840-848,1984-2005`). Integration assertions cover prompt lookup and speculative width errors (`crates/onnx-genai-engine/tests/native_engine.rs:39-76`); the exhaustive enum match covers all current variants.
3. **Pipeline/device selection — resolved.** Explicit Native pipelines are refused, and Auto inspects every component and refuses native-only pipeline models (`crates/onnx-genai-engine/src/pipeline.rs:96-112`), with both paths tested (`pipeline.rs:587-657`). Native single-model construction now receives `SessionOptions` and clearly rejects non-CPU execution providers before loading its CPU session (`engine.rs:306-318,731-755`), tested with WebGPU (`tests/native_engine.rs:78-97`).

## Regression check

- `EngineConfig::default()` remains `Auto` (`crates/onnx-genai-engine/src/config.rs:342-360,390-404`); non-matching models still fall through to ORT (`engine.rs:1925-1939`).
- Default engine and server builds pass.
- Native serving remains single-owner/serialized through the driver fallback (`crates/onnx-genai-server/src/driver.rs:104-121,315-321`), and each native request resets decode state (`crates/onnx-genai-engine/src/native_decode.rs:215-241`).
- Deferred-feature failures remain: engine speculation and KV connectors (`engine.rs:731-745`), persistent sessions/scheduling (`engine.rs:883-889,1137-1141,1203-1305`), embeddings (`crates/onnx-genai-engine/src/embedding.rs:52-65`), and continuous batching's ORT/static-cache requirements (`crates/onnx-genai-engine/src/batched.rs:434-445,586-608`).

## Build/test gate

- `cargo build -p onnx-genai-engine`: PASS
- `cargo build -p onnx-genai-engine --features native-backend`: PASS
- `cargo build -p onnx-genai-server`: PASS
- `cargo test -p onnx-genai-engine --features native-backend`: 111 passed, 18 failed, 1 ignored; all 18 failures are the allowed missing `tiny-llm/model.onnx` fixture failures.
- Targeted backend tests: 6 passed (3 unit/pipeline plus 3 native integration), 0 failed.

#### Source: `roy-native-cuda-device.md`, `wallace-native-cuda-review.md`, `deckard-native-cuda-safe.md`, `wallace-native-cuda-rereview.md`

### 2026-07-16: Native CUDA serving shipped with a fail-fast heterogeneous-placement gate
**By:** Roy, Deckard; reviewed by Wallace
**Status:** 🟢 CLEAR (`fa30410`; supersedes the rejected `559c46f` CUDA-only serving behavior)

**Decision:** Native Engine/server CUDA device plumbing is shipped, but CUDA-only native sessions must probe model coverage at load time and reject real sub-4-bit models that need CPU fallback. The load error explicitly identifies heterogeneous CPU+CUDA placement as unavailable and directs users to native CPU or ORT. It must occur before the server listens, so unsupported models cannot reach a request-time HTTP 500 or hang.

**Rationale:** `559c46f` correctly exposed `NativeDecodeDevice::Cuda` through `EngineConfig`, `SessionOptions`, and server CLI/environment selection, but the real 144-`BlockQuantizedMatMul` IQ4 model failed under a CUDA-only session during prefill and later on `Transpose`. The prior Constant/Gather fixture did not exercise this failure mode. Deckard's `fa30410` adds per-node CUDA capability probing, including symbolic/M>1 `BlockQuantizedMatMul`, and a reachable multi-token sub-4-bit regression: CPU generation succeeds; explicit CUDA and SessionOptions-routed CUDA fail deterministically at load with actionable remediation. A fully CUDA-supported smoke graph remains the positive CUDA parity proof.

**Deferred:** True GPU serving for real sub-4-bit models requires heterogeneous CUDA-first/CPU-fallback placement, cross-device buffers/copies, and M>1 CPU prefill versus M=1 CUDA decode. The design is documented in `docs/HETEROGENEOUS_PLACEMENT.md` and is **AWAITING USER GREENLIGHT**.

### 2026-07-16: Comparison and logical ops infer Bool outputs
**By:** Chew; reviewed by Leon (🟢 CLEAR)
**What:** Commit `d06d1e7` registers `Less`, `LessOrEqual`, `Greater`, `GreaterOrEqual`, `Equal`, `And`, `Or`, `Xor`, and `Not` with `tensor(bool)` output inference. Binary operators retain NumPy broadcast shapes and `Not` retains its input shape. Bitwise operators are intentionally unchanged.
**Why:** ONNX comparison/logical outputs are Bool regardless of their inputs. Coverage includes all five comparisons, three binary logical operators, and `Not`; shape-inference build and tests passed (115 tests). Expanded-Attention now advances past Pad/Less and stops at unsupported `Mod` at node 50, tracked as `mod-op-support`.

### 2026-07-16: GAFF ChildExecutor control-flow foundation
**By:** Sapper; reviewed by Holden (🟡 ADVISORY)
**What:** `onnx-runtime-session` now has crate-internal `ChildExecutor::new/run/stats` for recursive control-flow bodies. It lazily compiles a seeded child plan by external dtype/shape signature, supports lexical captures including transitive nested captures, scopes `WeightRef::Inline` initializers to the child, and returns declared outputs in order. The next execution step is `If`, which selects a child keyed by `(node_id, branch)` and runs it with the live captured scope.
**Why:** The reusable executor avoids rebuilding body setup for every invocation while preserving formal/input shadowing and nested isolation. Build and the 114-test session suite passed. Advisory follow-up `gaff-exec-cache-lru`: the current one-entry cache is correctness-safe but recompiles an `A → B → A` signature sequence; add a signature-keyed multi-plan cache and permanent shadowing/nested-cache tests.
## 2026-07-17 — GAFF If execution and CPU Mod support

### Execute ONNX If branches through cached ChildExecutors
**By:** Sapper; reviewed by Holden (🟢 CLEAR)
**What:** Commit `7a369ef` completes ONNX `If` in `onnx-runtime-session`: BOOL scalar or `[1]` conditions select the corresponding branch, captures are materialized from the live enclosing scope, and branch-specific `ChildExecutor`s are cached by `(node_id, branch)`. The runtime validates branch output count and known dtypes before execution, then binds outputs positionally.
**Why:** Independent branch caches preserve alternating true/false execution while lexical captures stay fresh. Tests cover branch reuse, capture changes, inline initializers, condition validation, and output mismatches; the session build and all 117 tests passed. This completes the loader → ChildExecutor → If GAFF control-flow vertical slice. Loop and Scan remain.

### Add CPU Mod with ONNX fmod modes and NumPy integer semantics
**By:** Joi; reviewed by Bryant (🟡 ADVISORY)
**What:** Added `ai.onnx::Mod` (opset 10+) to CPU execution and shape inference. `fmod=0` implements floor-mod integer semantics (divisor-sign result); `fmod=1` implements C/Rust sign-of-dividend remainder for integers and floats. Shared broadcasting supports signed/unsigned 8–64-bit integers and f16/bf16/f32/f64 where valid; integer zero divisors return zero, matching existing CPU integer `Div`.
**Why:** Expanded Attention had stopped at unsupported `Mod`. CPU EP (435 passed, 1 ignored), shape inference (116 plus doctests), and all 13 official ONNX Mod CPU cases passed. The remaining expanded-Attention blocker is missing logical `And` execution at node 39. Direct BF16 Mod coverage and a ChildExecutor multi-signature cache remain follow-ups.

**Sources:** `sapper-gaff-if.md`, `holden-sapper-gaff-if-review.md`, `joi-mod-op.md`, `bryant-joi-mod-review.md`; merged commits `7a369ef` and `aa7127e`.
## 2026-07-17 — Logical kernels, Expand inference, and GAFF Loop completion

### Execute ONNX logical Bool kernels on CPU
**By:** Chew; reviewed by Bryant (🟢 CLEAR)
**What:** Merged `557ca87`: CPU `And`, `Or`, `Xor`, and `Not` accept Bool tensors, use NumPy broadcasting where applicable, treat every nonzero byte as true, and emit canonical `0`/`1` bytes.
**Why:** Logical semantics must operate on Bool truth values rather than raw bytes (notably two distinct nonzero Xor operands are both true). The broadcast truth-table coverage passed; `onnx-runtime-ep-cpu` build and tests passed (436 passed, 1 ignored). Expanded-Attention conformance advances to node 58.

### Infer ONNX Expand shapes bidirectionally
**By:** Chew; reviewed by Bryant (🟢 CLEAR)
**What:** Merged `14b5136`: `Expand` is registered from opset 8 and infers the bidirectional broadcast of input and known target shapes while preserving the input dtype. A known target-vector length with unavailable values retains a rank of `max(input rank, target length)` using fresh symbols.
**Why:** This correctly handles leading target dimensions, either operand being one, strict incompatible dimensions, and unknown target values. `onnx-runtime-shape-inference` build and 120-test suite passed; expanded-Attention advances past node 58.

### Complete GAFF Loop through ChildExecutor after strict review
**By:** Sapper, revised by Leon; reviewed by Holden (🔴 REJECT → 🟢 CLEAR)
**What:** Sapper's initial `8052891` Loop implementation was rejected and Sapper locked out because scan stacking eagerly reserved from untrusted `M` and loop-carried output shapes were not invariant. Leon's final merged revision `f6e8ba6` removes the untrusted eager reservation and validates each carried output's initial dtype and full shape on every iteration.
**Why:** A condition-terminated loop with `M = i64::MAX` must not capacity-overflow before its first scan slice, and ONNX loop-carried values must retain shape as well as dtype. Regression tests cover huge-trip-count early exit and a second-iteration shape change; `onnx-runtime-session` build and 121-test suite passed. Loader → ChildExecutor → If → Loop is complete; Scan is the remaining control-flow op and is in progress.

**Sources:** `chew-and-logical-kernel.md`, `bryant-chew-logical-kernel-review.md`, `chew-expand-shape-infer.md`, `bryant-chew-expand-review.md`, `sapper-gaff-loop.md`, `holden-sapper-gaff-loop-review.md`, `leon-gaff-loop-fix.md`, `holden-leon-gaff-loop-rereview.md`.

### 2026-07-17: Clear Scatter family shape inference
**By:** Bryant
**Verdict:** 🟢 CLEAR
**What:** Independently reviewed commit `db868a56d99c8369938d5b7496c2d6db4706b3bd`. `ScatterND`, `ScatterElements`, and deprecated `Scatter` copy input 0's complete `TypeInfo`, so output shape and dtype exactly match `data`; `axis` and `reduction` are shape-neutral. All names are registered through the movement handler and exercised through the default registry. Unknown input-0 type leaves the output unresolved without error, matching the crate's convention. The four tests cover ScatterND, non-default-axis ScatterElements, the opset-9 alias, dtype passthrough, and unresolved-data fallback. No incorrect rank or shape validation was added.
**Why:** The implementation follows the ONNX Scatter-family output contract and the registry's range-based dispatch model. Validation passed: `cargo build -p onnx-runtime-shape-inference`; `cargo test -p onnx-runtime-shape-inference` — 124 passed, 0 failed (14 unit + 4 graph + 105 op-rule + 1 doc).
**Blockers:** None.
**Advisories:** None.

### 2026-07-17: Clear BatchNormalization and InstanceNormalization shape inference
**By:** Bryant
**Verdict:** 🟢 CLEAR
**What:** Independently reviewed commit `1ae2b676375ff67c7afca951ac0c9cdfb09fe827`. Both handlers infer only output 0 (`Y`) by cloning input 0 (`X`), preserving its complete symbolic/concrete shape and dtype. Parameter inputs cannot affect the inferred output. The default-domain registrations cover BatchNormalization schema revisions 9/14/15 and InstanceNormalization revision 6 through the registry's range-based dispatch. Unknown `X` leaves outputs unresolved without error or panic, and the handlers add no graph validation.
**Why:** This matches the inference-only runtime contract: the CPU BatchNormalization factory rejects `training_mode != 0`, and its kernel requires exactly one output. If a training-mode node declares additional outputs, shape inference resolves only `Y` and safely leaves every additional slot unresolved; `set_output_type` is bounds-checked, so this degrades gracefully rather than panicking.

**Tests reviewed:** The BatchNormalization test verifies exact `X` shape and Float16 dtype passthrough at opsets 9, 14, and 15; the InstanceNormalization test verifies the same at opset 6; the unknown-`X` test covers both operators.

**Validation:** `cargo build -p onnx-runtime-shape-inference` passed. `cargo test -p onnx-runtime-shape-inference` passed all 127 tests (14 unit + 4 graph integration + 108 operator-rule + 1 doc test).

**Advisory:** No correctness issue. A future regression test could declare BatchNormalization's training outputs and assert that only `Y` resolves, documenting the graceful inference-only behavior already present.

### 2026-07-17: Approve Leon's Scan stacking overflow repair
**By:** Holden
**What:** 🟢 APPROVE `gaff-scan` at `8c38afcfaeed47d8c1cbc907773688828868c2c3`. The blocker from `d3adfa3b` is fixed, including the zero-mask edge.
**Why:** `stack_new_axis` now checks the outer and inner element products, the element-to-byte multiplication, `n * outer` before multiplying by `inner`, the final byte count, and all source/destination offset and range arithmetic. Every check returns `SessionError::ShapeOverflow` before allocation or copy loops. For the exact Float32 repro with input shape `[7_000_000_000_000_000_000, 0, 3]`, scan axis 2, Identity slices `[7_000_000_000_000_000_000, 0]`, and `scan_output_axes=[1]`, `inner=0` no longer masks the overflow in `3 * outer`; the regression receives `ShapeOverflow { value: "stacked tensor output row count", .. }` immediately. The targeted test passed in both debug and release in 0.00s, confirming no huge allocation or loop. The delta is confined to `crates/onnx-runtime-session/`; the previously verified Scan semantics and validations remain covered. `cargo build -p onnx-runtime-session` passed, and `cargo test -p onnx-runtime-session` passed all 127 tests (0 failed; 1 doc-test ignored).

### 2026-07-14: Bound QMoE allocations and accept odd affine block counts
**By:** Holden
**What:** QMoE byte-count preflights now reject allocations above `isize::MAX`, and affine int4 validation requires only block-size alignment while retaining ceil-packed zero-point rows for odd block counts.
**Why:** Rust allocations above `isize::MAX` can panic despite fitting in `usize`. ORT's affine int4 layout permits an odd number of blocks, with the final zero-point byte containing only the remaining nibble; rejecting those layouts was stricter than the format.

### 2026-07-17: Reject Scan due to zero-element stacking overflow/hang
**By:** Holden
**What:** 🔴 REJECT `gaff-scan` at `d3adfa3bf2bfbc514e770168ce23b331d2084dd4`. Sapper is locked out of this revision; Leon should own the repair.
**Why:** Scan's ordinary semantics are otherwise implemented correctly, but its newly added non-leading-output-axis path exposes unchecked shape arithmetic in `stack_new_axis` to untrusted Scan output shapes.

## Blocker

**High — crafted zero-element Scan can panic in debug or effectively hang in release.**

- `executor.rs:4125-4142` checks the per-slice element byte count, but then calls `stack_new_axis`.
- `sequence.rs:336-347` uses unchecked `product`, `* esize`, `n * outer * inner`, and offset arithmetic.
- Concrete zero-byte repro shape: a Float32 scan input shaped `[7_000_000_000_000_000_000, 0, 3]`, scanned on axis 2, with an Identity body producing slices shaped `[7_000_000_000_000_000_000, 0]`, and `scan_output_axes=[1]`. This is a three-trip Scan over an empty tensor.
- At final stacking, `n=3`, `outer=7_000_000_000_000_000_000`, and `inner=0`. Debug evaluation of `n * outer * inner` overflows at `n * outer` and panics. In release it wraps, allocates zero bytes, then executes approximately `3 * outer` zero-length copy iterations, effectively hanging.

The fix must keep stacking arithmetic checked and return a `ShapeOverflow`/control-flow error before allocation or loops. Add a regression covering a huge dimension masked by a zero dimension and a non-leading scan output axis.

## Checks that held

- No eager reservation from `T`: `TensorStackAccumulator::new()` starts empty and accumulation grows incrementally.
- State dtype and shape are captured from initial state and checked every iteration with the state index in the error. The shape-rejection test reaches iteration 2 and asserts the exact mismatch.
- Input axes are defaulted, negative-normalized, sliced correctly, direction-aware, and checked for equal trip counts.
- Output axes are defaulted, negative-normalized, and output directions are applied; the non-zero/reverse test asserts shape and values.
- `num_scan_inputs`, state count, body input arity, and body output arity are validated consistently.
- The six Scan tests assert substantive final states, output shapes/values, zero-trip behavior, and the rejection error.

## Zero-trip punt

Acceptable limitation: when neither body nor parent metadata can determine an empty scan-output element shape, `finish_scan` returns a clear `SessionError::ControlFlow` (`executor.rs:4097-4103`). It does not panic. The supplied zero-trip test covers the metadata-available success path; a graceful-error regression would be useful but is non-blocking.

## Validation

- `cargo build -p onnx-runtime-session`: PASS.
- `cargo test -p onnx-runtime-session`: PASS — 126 passed, 0 failed, 1 ignored doc-test.

### 2026-07-17: Add ONNX Scatter-family shape inference
**By:** Joi
**What:** Added shape inference registrations for ScatterND (opsets 11/13/16/18), ScatterElements (opsets 11/13/16), and deprecated Scatter (opset 9, using the ScatterElements rule). Each output clones input 0's shape and dtype; axis and reduction do not affect shape. When data TypeInfo is unavailable, the output remains unresolved, matching sibling passthrough handlers; known symbolic dimensions are preserved unchanged. Added 4 tests: scatter_nd_preserves_data_shape_and_dtype, scatter_elements_non_default_axis_preserves_data_shape, scatter_deprecated_alias_preserves_data_shape_and_dtype, and scatter_unknown_data_shape_leaves_output_unresolved. Validation passed: cargo build and cargo test for onnx-runtime-shape-inference (124 total tests: 14 unit, 4 graph, 105 op-rule, 1 doctest; 0 failures). Commit: db868a56d99c8369938d5b7496c2d6db4706b3bd.
**Why:** The Scatter family has shape- and dtype-preserving ONNX semantics and complements the existing GatherND/GatherElements inference handlers.

### 2026-07-17: Reject overflowing Scan stack shapes before allocation
**By:** Leon
**What:** Changed `stack_new_axis` to check the outer/inner shape products, element-to-byte multiplication, output row/byte counts, and source/destination offset arithmetic. The guard returns `SessionError::ShapeOverflow`; critically, `n * outer` is checked before multiplication by a zero `inner`, so zero-sized tensors cannot mask the overflow. Added `scan_rejects_zero_element_nonleading_output_axis_stack_overflow`.
**Why:** Holden's crafted three-trip Scan with element shape `[7_000_000_000_000_000_000, 0]` and `scan_output_axes=[1]` previously panicked in debug or wrapped into a massive zero-length copy loop in release. `cargo build -p onnx-runtime-session` and all 127 crate tests passed. Fixed in commit `8c38afcfaeed47d8c1cbc907773688828868c2c3`.

### 2026-07-17: Add BatchNormalization and InstanceNormalization shape inference
**By:** Leon
**What:** Registered ai.onnx `BatchNormalization` for opsets 9, 14, and 15 and `InstanceNormalization` for opset 6. Both inference handlers pass input `X`'s shape and dtype through to output `Y`; BatchNormalization intentionally does not emit training-only outputs. Added 3 tests: `batch_norm_inference_passthrough_opsets_9_14_15`, `instance_norm_passthrough_opset_6`, and `normalization_unknown_x_leaves_output_unresolved`. Commit: `1ae2b676375ff67c7afca951ac0c9cdfb09fe827`.
**Why:** In inference mode, scale, bias, mean, and variance do not affect output shape, and unresolved `X` must leave `Y` unresolved like sibling normalization handlers. `cargo build -p onnx-runtime-shape-inference` passed; `cargo test -p onnx-runtime-shape-inference` passed 127 tests total (14 unit, 4 graph, 108 op-rule, 1 doc-test), with 0 failures.

### 2026-07-14: QMoE checked-arithmetic hardening re-review — REJECT
**By:** Nabil
**What:** 🔴 REJECT commit `436cedc`. Reassign the revision to Holden; Roy and Deckard remain locked out for this revision cycle.
**Why:** `qmoe.rs:498-501` only checks `elements * element_size` for `usize` overflow. It does not reject byte counts above `isize::MAX`, so attacker-controlled shapes whose byte product fits `usize` can still reach `Vec::with_capacity`/`vec!` and panic with capacity overflow before returning an `EpError` (`qmoe.rs:268,434-436,456`; compare the repository's addressability guard in `sequence.rs:234-250`). The three new tests correctly assert `EpError::KernelFailed` for their selected `usize`-overflow paths, including zero-masked multiplication, but none covers this addressability boundary. Additionally, `qmoe.rs:392-399` newly rejects valid affine int4 layouts with an odd number of blocks, contradicting the documented `ceil(blocks / pack_size)` zero-point layout; this is a semantic change and must be removed or separately justified/tested because the hardening fix is required to be purely defensive. The full `onnx-runtime-ep-cpu` suite completed with 444 passed and 1 ignored, but does not cover either gap.

### 2026-07-14: QMoE round-2 fix approved
**By:** Nabil
**What:** 🟢 APPROVE commit `b1c9a55`; the `com.microsoft::QMoE` CPU kernel ships.
**Why:** `checked_byte_count` now rejects counts above `isize::MAX` before input copies, output allocation, and per-expert dequantization allocation. Affine int4 validation accepts block-size-divisible odd-block rows, uses ceil-packed zero-point storage, and the final low nibble is indexed correctly; the new regressions exercise both fixes and all 446 non-ignored `onnx-runtime-ep-cpu` tests pass.

### 2026-07-17: QMoE CPU kernel review — 🔴 REJECT
**By:** Nabil
**What:** Reject commit `3cf25359ed216a0f3c7e1c851c5083ad8ede115b` until the QMoE execution path validates all shape-derived counts and offsets with checked arithmetic and returns `EpError` on overflow. The dequantization and float-MoE equivalence work is otherwise numerically sound.
**Why:**

#### Blocker — unchecked untrusted shape arithmetic

The kernel validates individual dimensions but then performs unchecked products and offset arithmetic on them:

- flattened rows: `qmoe.rs:140`;
- output allocation: `qmoe.rs:254`;
- per-expert dequant allocation: `qmoe.rs:369`;
- expert/row packed, scale, and zero-point offsets: `qmoe.rs:371-379`;
- fused SwiGLU FC1 width: `moe.rs:287-290`.

Overflow can panic in debug builds or wrap in release builds, producing undersized allocations followed by slicing/indexing panics or out-of-bounds behavior instead of a clear `EpError`. This fails the explicit safety gate for untrusted quantized tensor shapes. The replacement must preflight rows, tensor element counts, byte counts, dequantized sizes, and every expert-row stride/offset with `checked_mul`/`checked_add` before materialization or slicing, with adversarial overflow tests.

Per reviewer protocol, Roy is locked out from revising the rejected artifact; a different agent should own the safety revision (Deckard is the recommended owner).

#### Dequantization correctness — clear

- QMoE reuses `matmul_nbits::dequantize_nbits_row`, so packing is consistent with the crate's MatMulNBits path.
- Expert-major row addressing, K-axis LSB-first int4 nibble order, int8 byte order, per-K-block scale selection, packed affine zero points, and symmetric midpoint defaults (`8`/`128`) are correct.
- The affine multi-block fixture uses two blocks with both scales and zero points changing by block, so block-to-scale mapping is genuinely exercised.
- Registration is present beside `MoE` in `kernels/mod.rs`.

#### Float-MoE equivalence — genuine but narrow

`qmoe_int4_single_block_matches_float_moe` is a real differential test: it constructs dequantized float expert tensors, executes the registered trusted `MoE` kernel, executes QMoE from packed weights, and compares every output at `1e-5`. This is not a self-comparison.

All four numeric equivalence cases use `activation_type="identity"` with no biases, FC3, or separate `router_weights`. Shared `routing_weights` and `run_expert` code makes activation/routing reuse structurally strong, but follow-up tests should cover at least one nonlinear activation, fused/unfused SwiGLU/FC3, biases, and separate aggregation weights.

#### Advisory — schema strictness

For affine block quantization, ORT requires K to be divisible by `block_size * pack_size` when zero points are supplied. The kernel checks only divisibility by `block_size` and accepts a ceiling-packed zero-point dimension. Valid inputs decode correctly, but invalid schema shapes are accepted.

#### Deferred scope and GLM-5.2 impact

- Batch-union grouping, caching, and compressed GEMM are performance punts only.
- Native affine int2, IQ1/IQ2/IQ3/IQ4, MXFP4/FP8, and row-wise QMoE are load/run blockers for artifacts encoded in those formats. GLM-5.2 dynamic IQ1/IQ2 packages therefore cannot run through this kernel; a blockwise affine Q4/Q8 requantized export can.
- Sparse mixer is a load/run blocker only for graphs that set `use_sparse_mixer=1`; the current GLM route can remain explicit and does not inherently require it.

#### Validation

- `cargo build -p onnx-runtime-ep-cpu`: passed.
- `cargo test -p onnx-runtime-ep-cpu`: 441 passed, 0 failed, 1 ignored.
- Targeted QMoE tests: 5 passed (int4 single-block, int8, affine multi-block, normalized top-k=2, unsupported block size).

### 2026-07-17: Complete GAFF Scan control-flow execution
**By:** Sapper
**What:** Implemented ONNX Scan for opsets 9/11/16 through the existing ChildExecutor path. Scan validates its body/input/output arity, slices each scan input on its configured (including negative) axis in forward or reverse direction, threads state, and stacks each scan output on its configured axis/direction. Zero-trip execution preserves initial states and constructs typed empty scan outputs from body output specifications.
**Why:** Scan was the remaining GAFF control-flow operator after If and Loop. State outputs are checked against the initial dtype and shape before every threading step, and scan accumulators grow incrementally without reserving from trip-count hints. Added 6 Scan tests: cumulative sum across opsets 9/11/16, multiple inputs with a negative/non-leading axis, reverse input direction, non-zero/negative output axis with reverse output direction, zero-trip typed output, and state shape-change rejection. `cargo build -p onnx-runtime-session` and all 126 session tests passed (1 doctest ignored). Commit: `d3adfa3`.

### 2026-07-17: Rename the custom ONNX operator domain to `pkg.nxrt`
**By:** Wallace
**What:** Replaced 31 serialized/text references to `com.github.onnxruntime.genai` with `pkg.nxrt` across 20 files:
- `crates/onnx-genai-engine/src/engine.rs`
- `crates/onnx-genai-engine/src/pipeline.rs`
- `crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs`
- `crates/onnx-runtime-ep-cpu/src/kernels/mod.rs`
- `crates/onnx-runtime-ep-cuda/src/kernels/block_quantized_matmul.rs`
- `crates/onnx-runtime-ep-cuda/src/kernels/mod.rs`
- `crates/onnx-runtime-ep-cuda/src/provider.rs`
- `crates/onnx-runtime-ep-cuda/tests/block_quantized_matmul_gpu.rs`
- `crates/onnx-runtime-python/tests/test_block_quantized_matmul.py`
- `crates/onnx-runtime-session/src/executor.rs`
- `crates/onnx-runtime-shape-inference/src/handlers/linalg.rs`
- `crates/onnx-runtime-shape-inference/tests/op_rules.rs`
- `docs/DEEPSEEK_CSA_MTP_RUNTIME.md`
- `docs/HETEROGENEOUS_PLACEMENT.md`
- `docs/PROGRESS.md`
- `docs/SUB4BIT_QUANT.md`
- `docs/benchmarks/e2e-sub4bit-validation.md`
- `scripts/e2e-sub4bit-validation.sh`
- `scripts/e2e_sub4bit_export.py`
- `tests/fixtures/tiny-native-sub4-engine/model.onnx`

The shell validation script and serialized ONNX fixture were additional tracked consumers found by the repository-wide scan. The fixture changed only its node-domain and opset-import fields.

**Why:** The runtime emits, detects, validates, registers, and executes the same custom operators, so every producer and consumer must use one self-consistent domain.

**Deliberately unchanged:** The tracer ITT domain constant `"nxrt"`, all `set_process_name("nxrt")` calls, and Python module paths such as `module = "nxrt.genai"` / `"nxrt.eager"` remain unchanged because they are not ONNX operator domains.

**Validation:**
- Repository-wide tracked-byte scan, excluding `.squad/` and `target*/`: 0 old-domain references; 31 `pkg.nxrt` references across 20 files.
- `cargo build -p onnx-runtime-ep-cpu -p onnx-runtime-session -p onnx-genai-engine -p onnx-runtime-shape-inference`: passed.
- `cargo test -p onnx-runtime-ep-cpu -p onnx-runtime-session -p onnx-genai-engine -p onnx-runtime-shape-inference`: engine ran first with 106 passed, 18 failed, 1 ignored. All 18 failures were the known missing `tests/fixtures/tiny-llm/model.onnx` fixture errors; there were no domain-mismatch failures.
- Follow-up `cargo test -p onnx-runtime-ep-cpu -p onnx-runtime-session -p onnx-runtime-shape-inference`: passed.
- `cargo build -p onnx-runtime-ep-cuda` with CUDA environment: passed.
- `cargo test -p onnx-genai-engine --features native-backend --test native_engine`: 5 passed.
- `cargo test -p onnx-genai-engine --features native-backend auto_backend`: 2 passed.

**Commit:** `a776ebee4a3eddffdc6ce018d9c26cfab1a0bba7`
## 2026-07-17 — QMoE sub-byte support

### 2026-07-17: Restrict QMoE sub-byte experts to byte-dividing widths
**By:** Roy
**What:** CPU `com.microsoft::QMoE` now accepts `expert_weight_bits` in `{1, 2, 4, 8}`; unsupported widths, including 3-bit, remain rejected. Checked allocation, layout, range, and `isize::MAX` guards are unchanged.
**Why:** Byte-dividing widths keep packed weights and zero points from crossing byte boundaries. With power-of-two blocks of at least 16, the generalized packing and partial zero-point tails remain byte-aligned.

### 2026-07-17: Approve QMoE 1-bit and 2-bit expert support
**By:** Nabil
**What:** 🟢 Approved `cdb4ee5`, extending CPU `com.microsoft::QMoE` to 1-bit and 2-bit expert weights.
**Why:** Factory acceptance is exactly `{1, 2, 4, 8}`; packing, offsets, masks, affine zero-point tails, and row sizing are correct for byte-dividing widths, while the established 4/8-bit path remains unchanged. New equivalence/rejection coverage and the full crate suite passed (450 passed, 1 ignored).
## 2026-07-17 — Standard shape-inference coverage

### 2026-07-17: Add ONNX shape inference for OneHot, Trilu, spatial rearrangements, and Compress
**By:** Joi
**What:** Added opset-aware inference for OneHot (9/11), Trilu (14), DepthToSpace (1/11/13), SpaceToDepth (1/13), and Compress (9/11). Dynamic extents become fresh symbols; specified input symbols and dtypes are preserved; spatial arithmetic/divisibility checks are checked.
**Why:** These standard operators no longer block graph-shape resolution. The handlers follow ONNX axis, rank, blocksize, mode, dtype, and known-divisibility contracts without panicking on symbolic dimensions; 140 crate tests passed.

### 2026-07-17: Approve OneHot/Trilu/spatial-rearrangement/Compress inference
**By:** Bryant
**What:** 🟢 Approved `98ee7a6`. The five handlers use correct schema-version registrations, shape/dtype contracts, symbolic degradation, known-divisibility validation, and checked `blocksize² * C` arithmetic.
**Why:** Coverage includes OneHot axes/dynamic depth, Compress axis/no-axis, Trilu variants, spatial modes and schema versions, symbolic dimensions, divisibility failures, and blocksize-square overflow. `cargo test -p onnx-runtime-shape-inference` passed all 140 tests.
## 2026-07-17 — Owner-reviewed scale-model design decisions

#### Source: `coordinator-projection-fusion-md-approved-all-10-open-question.md`

### 2026-07-17T05-28-52: PROJECTION_FUSION.md approved — all 10 open questions resolved
**By:** coordinator
**What:** PROJECTION_FUSION.md approved — all 10 open questions resolved
**Why:** ### PROJECTION_FUSION.md — 10 open-question resolutions (approved by @justinchuby)

**Q1 (memory budget):** Physical concat of sibling projection weights (accept ~125 MiB), gated by an owned-bytes release check; if gain <3% or over budget, switch to grouped separate-buffer variant.
**Q2 (Split encoding):** Always materialize an explicit int64 split initializer (no implicit/equal-split assumption).
**Q3 (rollout):** Guard behind ONNX_GENAI_PROJECTION_FUSION=1; default-on only after Qwen validation.
**Q4 (prefill/TTFT):** ≤2% TTFT-regression release gate + prefer packed-GQA bypass to minimize prefill materialization. Multi-turn exposure is bounded because prefix caching (PrefixCache in onnx-genai-kv, wired in engine.rs, tracks prefix_cache_hit_len) already limits per-turn prefill to newly-added tokens.
**Q5 (packed GQA scope):** First version does gate/up only (benchmark Qwen QKV is already packed). Separate-QKV → packed-GQA rewiring is Phase 2.
**Q6 (bias):** Defer post-MatMul bias Add fusion; Phase A tests the no-bias gate/up path first (benchmark Qwen has no bias). Bias fusion moves to Phase D generalization.
**Q7 (explicit zero-point):** Correct handling of explicit-zp (asymmetric) MatMulNBits is a HARD rollout gate. Until supported, fusion MUST safely skip explicit-zp nodes — never miscompute.
**Q8 (code placement):** Build a thin shared `optimizers` module (pass trait + registry + env-flag toggle + ordering) and have EACH EP register its OWN fusion pass set (EP-scoped, per @justinchuby's point that fusion differs per EP, matching ORT's EP-scoped transformer model). Projection fusion lands first as a CPU EP optimizer; CUDA/Metal add their own later.
**Q9 (profiler naming):** Single `fused_projection` profiling category for attribution; retain per-segment sub-labels in debug builds for interpretability.
**Q10 (fallback threshold):** If physical-concat yields <3% decode gain OR exceeds the memory budget, fall back to the grouped separate-buffer variant.

Status: doc to be marked Approved; resolutions appended to §17.

#### Source: `coordinator-deepseek-csa-mtp-runtime-md-9-decisions-confirmed-.md`

### 2026-07-17T05-16-20: DEEPSEEK_CSA_MTP_RUNTIME.md — §9 decisions confirmed + 14 open questions resolved
**By:** coordinator
**What:** DEEPSEEK_CSA_MTP_RUNTIME.md — §9 decisions confirmed + 14 open questions resolved
**Why:** Owner @justinchuby reviewed docs/DEEPSEEK_CSA_MTP_RUNTIME.md. Confirmed the §9 proposed decisions (CSA-1..7, MTP-1..6, GPU-1) EXCEPT the CSA-7 fallback default is flipped (see Q11). Resolved all 14 §10 open questions.

SOURCE OF TRUTH (Q1, and method for Q2-Q7): numerical truth = official HF deepseek-ai/DeepSeek-V4-Flash reference (inference/model.py + inference/kernel.py); run it to dump golden intermediate tensors, freeze as layout-v1 goldens in Phase 0. NeMo (Megatron-Bridge/AutoModel) + arXiv 2606.19348 for cross-check. llama.cpp is NOT faithful (Flash-Attn disabled, DSV4 graph WIP) — do not use for goldens. mobius is our exporter, not ground truth.

Q2 (ratio-128 HCA semantics): pin from llama... no — pin from official reference behavior; do not presuppose "attend all" vs "extra selection"; golden-driven.
Q3-Q7 (cache layout CW/ICW/carry; ratio-4 top-k tie order + duplicate policy + causal boundary masking; compressed RoPE rotated values + retained state; MTP official recurrent state = post-layer pre-_hc_head HC state?; sidecar KV lifetime): ALL pinned from official reference goldens; never inferred from mobius tensor names or by broadcasting mtp_hidden.

Q8 (MTP weight sharing): support BOTH — metadata prefers named model-package components, falls back to target initializer names; tied/quantized embeddings referenced zero-copy (no raw f32 duplication).

Q9 (verification width): contract requires 1+k tokens verified in one forward; static/native runners MAY degrade to a sequence of single-token steps but MUST produce bit-identical acceptance (greedy token identity) — enforce with an equivalence test gate.

Q10 (batching): v1 requires equal-length compression/index cursors within a batch (regular cache layout, simple CUDA kernel). Ragged (per-row cursor length) is an IMMEDIATE fast-follow (owner: "以后 = 明天大概, 验证了的话"), modeled on vLLM PagedAttention per-row block-table indirection + DeepSeek-V4 per-row (offset,length) ragged gather. SparseKvGather + per-forward cursor journals + metadata state-groups already accommodate adding per-row cursor lengths — ragged is an extension, not a rewrite.

Q11 (fallback policy): FLIP of CSA-7 default — package defaults to native_csa_required and REJECTS dense fallback (safest; no silent degradation of the long-context memory advantage). Portability cost accepted.

Q12 (shared GLM primitive boundary): SparseKvGather is the shared correctness/reuse primitive; DeepSeek CompressedSparseAttention and GLM IndexShare DSA each get their OWN fused production op (selection semantics differ and must not be coupled).

Q13 (upstream path): incubate in the private pkg.nxrt domain (like BlockQuantizedMatMul); push an ORT contrib proposal only after BOTH DeepSeek and GLM contracts stabilize and v1 layout + goldens are frozen. Optional non-urgent tracking issue.

Q14 (acceptance tolerance): greedy final tokens MUST be bit-identical (integer argmax exact = hard correctness gate); intermediate compressor/index/attention tensors allow small f16/bf16 atol/rtol deviations, with concrete thresholds CALIBRATED from the official goldens (measure real error distribution, don't guess) and set per-layer for localizability.

Status: design approved to begin Phase 0 (freeze contracts + goldens from the official DeepSeek reference) pending owner's explicit greenlight; no CSA/MTP kernel implementation before goldens are frozen.

#### Source: `coordinator-heterogeneous-placement-md-direction-change-6-ques.md`

### 2026-07-17T04-50-59: HETEROGENEOUS_PLACEMENT.md — direction change + 6 questions resolved
**By:** coordinator
**What:** HETEROGENEOUS_PLACEMENT.md — direction change + 6 questions resolved
**Why:** Owner @justinchuby reviewed docs/HETEROGENEOUS_PLACEMENT.md. Major direction change on Q1 plus resolutions for all 6 open questions.

DIRECTION CHANGE (Q1): The doc's core premise — route M>1 quantized prefill to CPU — is REJECTED as unusable for GLM-5.2/DeepSeek-scale models ("prefill 必须留 CUDA"). Root cause is a kernel gap: CUDA BlockQuantizedMatMul is GEMV-only (M=1). Decision: FIRST build a CUDA BlockQuantizedMatMul M>1 GEMM prefill kernel (done: commit a99f7a8, pending review), so prefill stays batched on CUDA. Heterogeneous placement is put ON HOLD and, when resumed, shrinks to a narrow safety net: CPU fallback ONLY for ops with genuinely no CUDA kernel — never the hot quantized matmul. `DevicePreference::Gpu` = CUDA-preferred; keep a strict CUDA-only mode available for benchmarking/coverage-proof.

Q2 (placement cache granularity): cache by bounded shape CLASS (M=1 vs M>1), not exact shape. Only one placement axis exists today (decode vs prefill); class keying keeps the cache small and high-hit. Two-level exact-key refinement deferred as unneeded.

Q3 (KV at heterogeneous boundary, Phase 1): whole-KV transfer only, and only when a real cross-device boundary exists. Since attention stays on CUDA for both prefill and decode, KV naturally stays on device; range-based KV copies deferred to Phase 2.

Q4 (shared partition-boundary extraction location): extract into a shared, non-serialization utility in the LOADER, reusing the EPContext writer's boundary machinery. Single source of truth; writer stays an encoder, not an execution planner.

Q5 (cost selection): strongly prefer FEWER cross-device transfers (keep runs on one device) by default; only split a run onto CUDA when measured net gain justifies it. Cross-device round-trips usually eat kernel wins on the decode hot path. Full cost-model-driven adaptive placement deferred to Phase 4.

Q6 (release-blocking observability): gate releases on (1) per-node BlockQuantizedMatMul CUDA-decode (M=1) dispatch counts > 0 and == layer count, (2) fallback-reason counts grouped by op/domain/shape, (3) per-token CPU<->CUDA transfer bytes. These are assertable in CI/integration tests to prove decode did not silently fall to CPU. Per-node placement dump kept as an optional debug tool (doc §7).

Status: design ON HOLD pending the CUDA GEMM prefill kernel; will be re-scoped to narrow unsupported-op fallback afterward.

#### Source: `coordinator-weight-offload-md-10-open-questions-resolved-by-ow.md`

### 2026-07-17T04-28-26: WEIGHT_OFFLOAD.md — 10 open questions resolved by owner
**By:** coordinator
**What:** WEIGHT_OFFLOAD.md — 10 open questions resolved by owner
**Why:** Owner @justinchuby resolved all 10 open questions in docs/WEIGHT_OFFLOAD.md during design review.

Q1 lazy-initializer boundary: general executor WeightHandle from the start, compatible with existing ORT plugin EPs via capability detection — paging-capable EPs advertise an nxrt capability flag and get a lazy WeightHandle; stock ORT EPs get a materialized resident tensor fallback. Paging is opt-in, never a correctness dependency.

Q2 MoE op boundary: pkg.nxrt::BlockQuantizedMoE is the offload boundary (only it honors lazy expert leases), capability-negotiated with plain QMoE fallback; mobius emits BlockQuantizedMoE when nxrt capability present else QMoE; file an upstream ORT issue for lazy-external-weight QMoE.

Q3 exporter metadata: hybrid — numeric bindings (FC1/FC2/FC3, scales, zero-points, shared-expert flag, per-expert sizes) explicit op inputs/attrs, NEVER name-inferred; residency metadata in package manifest; format/layout version mandatory+explicit (loader hard-rejects mismatch). Refinement: residency metadata is a compact model-/layer-group-level layout descriptor referenced by a region-group id on the op, NOT per-op/per-expert; byte ranges computed from WeightStore offsets + descriptor.

Q4 host budget: hard cap on OWNED cache bytes is the cross-platform contract; RSS-tightening advisory, off hot path, only on already-evicted pages (no perf regression), behind PageAdvisor trait (madvise POSIX / Offer-DiscardVirtualMemory Windows / no-op fallback). Must be cross-platform + no perf degradation.

Q5 partial-GPU API: byte-budget + explainable placement plan primary; gpu_layers:N compat override reported in bytes.

Q6 mixed CPU/GPU MoE: single-device per layer first (Phase 3); intra-layer expert split deferred as measured optimization.

Q7 tile size: expert-FC-panel default (whole quant blocks; never split a block), byte override snapping to block boundaries, per-format minimums; auto-tune deferred to Phase 4.

Q8 governor arbitration: dynamic with hard KV floor (sized to committed in-flight sequences) + watermark hysteresis + rebalance dwell + admission control at batch formation. MUST be tested thoroughly (oscillation/thrash, KV-floor-breach, admission under continuous batching = hard test gate).

Q9 prefetch: layered opt-in — exact-next-wave default + heat warm-set + router-prediction opt-in (graduates only if measured to help end-to-end without more memory violations/p99). Extension: trait-based ResidencyPolicy public extension point — policy ADVISES, Resource Governor stays AUTHORITATIVE (enforces budgets/KV floor/leases; cancels low-value transfers). Policy proposes, Governor disposes.

Q10 integrity/lifetime: pin file identity cheaply at load (size + mtime/inode or fast header/region-table signature, O(1), no full re-hash), opt-in full-hash for attestation, SIGBUS handler -> clean runtime error, reject truncation/replacement of a live package.

Status: design approved; Phase 1 = mmap disk tier + active-expert CPU MoE access.
## 2026-07-17 — Overnight wave 4–5 landings

#### Source: `deckard-csa-abi-fix.md`

### 2026-07-17: Expose the frozen CSA v1 boundary without claiming stateful support
**By:** Deckard
**What:** Chose Option A. `pkg.nxrt::CompressedSparseAttention` v1 now accepts only the frozen stateful schema (11–20 positional inputs and 3–6 outputs). Execution returns an explicit `Unsupported` error for deferred compression/carry work, with ratio-4 errors also naming deferred index-cache and top-k selection. Roy's assembled-cache Phase-1 implementation remains as an unregistered tested reference.
**Why:** The documented input/output boundary is frozen even though the compression equations and carry contents are not implemented. Registering Roy's narrower 4–6 input seam as v1 would permanently publish an incompatible ABI; retaining it only as a test reference preserves its gather/attention correctness without exposing that ABI.

#### Source: `deckard-onnxrs-checker-strictness.md`

### 2026-07-17: Match onnx-rs checker strictness to ONNX v1.20
**By:** Deckard
**What:** Corrected four retained-proto validation conditions to match `onnx/checker.cc` at tag `v1.20.0`:
- `AttributeProto.type` is required only when `ModelProto.ir_version >= 2`; IR v1 attributes may omit the discriminator.
- `SparseTensorProto.indices` may be absent only when `values.dims == [0]` (NNZ zero); nonzero NNZ still requires indices.
- Sparse tensors require `dims_size() > 0` and every dense dimension strictly greater than zero, with checked dense-size arithmetic.
- Main-graph input/output tensor and sparse-tensor types require `shape`; subgraph, intermediate, attribute, and nested container types do not inherit that shape requirement.
**Why:** These are the field-, version-, and context-sensitive conditions enforced by ONNX v1.20 `check_attribute`, `check_sparse_tensor`, and `check_value_info`. Applying them more broadly rejects valid models; applying them less broadly accepts models rejected by the official checker.

#### Source: `deckard-onnxrs-if-scoping.md`

### 2026-07-17: Bind If captures lexically and merge symbols by namespace
**By:** Deckard
**What:** Infer control-flow subgraphs at their owning node with a lexical scope built from resolved enclosing values. Producer-less values that are neither formal inputs nor local initializers bind by name to that scope; imported symbolic dimensions receive child-local IDs with an explicit child-to-parent mapping. When merging `If` outputs, preserve only equal concrete dimensions or symbols that both branches map to the same parent symbol; otherwise mint a fresh parent symbol, and reject branch rank mismatches.
**Why:** ONNX subgraphs capture enclosing values by name, while each `Graph` owns an independent symbol namespace. Treating captures as placeholder scalars or comparing raw branch `SymbolId` numbers silently produces incorrect and dangling parent shapes.

#### Source: `leon-sequence-module.md`

### 2026-07-17: ONNX sequence value module API
**By:** Leon
**What:** `onnx-runtime-session::sequence` publicly exposes `SeqTensor`, `SequenceValue`, `SequenceError`, `SplitSpec`, `split`, and `concat`. `SeqTensor` wraps the crate's existing `Tensor` in `Arc`; sequence construct/insert/erase/at clone only the Arc handle, and the unit test proves identity with `Arc::ptr_eq`. Sequence homogeneity enforces dtype at construction/insertion; concat additionally enforces ONNX shape compatibility. Split supports per-index slices, scalar chunk sizes, explicit sizes, negative axes, and `keepdims`; concat supports existing-axis concatenation and `new_axis` stacking. Shape, offset, and byte arithmetic is checked, byte counts above `isize::MAX` are rejected, and data allocations use `try_reserve_exact`.
**Why:** ONNX Sequence operators need persistent value semantics without copying tensor payloads, while untrusted model shapes must not wrap arithmetic or panic allocation paths. The next op-graph phase must register and route SequenceEmpty/Construct/Insert/Erase/At/Length/SplitToSequence/ConcatFromSequence, translate tensor inputs and attributes into this API, retain returned `SeqTensor` handles through kernel consumption, and define sequence graph input/output bindings.

#### Source: `mariette-attn-fp16.md`

### 2026-07-17: CUDA Attention f16/bf16 dtype dispatch
**By:** Mariette
**What:** CUDA Attention now uses an `AttentionDtype` tag to map ONNX f32/f16/bf16 to cuBLASLt dtypes, element widths, and NVRTC softmax entry points. The half softmax kernels share a CUDA template with typed load/store helpers; max, exp, and sum reduction arithmetic is f32 before probabilities are narrowed to the IO dtype. The established f32 softmax source and entry point remain unchanged.
**Why:** This mirrors the crate's existing dtype-tag/template dispatch convention and keeps future dtype extension centralized. Both half GEMMs use `CUBLAS_COMPUTE_32F`, so accumulation is fp32 and cuBLASLt chooses an algorithm supported by the detected device without an SM90 or tensor-core-only compute-mode hardcode. GPU reference tests use fp16 `atol=3e-3, rtol=3e-3` (a small multiple of fp16 epsilon across two GEMMs and probability storage) and bf16 `atol=2e-2, rtol=2e-2` (wider for its 7-bit mantissa).

#### Source: `mariette-gpu-sm-general.md`

### 2026-07-17: Make CUDA reduction launches device-capability driven
**By:** Mariette
**What:** The CUDA EP now caches compute capability, per-block/default and opt-in shared-memory limits, maximum threads per block, and multiprocessor count. Every NVRTC kernel that requests dynamic shared memory (attention/standalone softmax, reductions, and normalization variants) derives a power-of-two block size and shared-memory request from those limits, includes the compiled function's own limits/static shared memory, and sets `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` when opt-in is required. The misleading SM90-specific cuBLASLt workspace comment now states that the 32 MiB budget is device-global scratch rather than Hopper shared memory.
**Why:** The only remaining explicit SM90 assumption was documentation around the cuBLASLt workspace; it was not a launch limit. No Hopper-only PTX (`wgmma`, TMA/`cp.async.bulk`, `setmaxnreg`, clusters/`mapa`) or FP8/TF32 compute mode exists in the crate. Capability-derived launch planning prevents current or future kernels from requesting more dynamic shared memory or threads than an older/newer device supports while preserving the existing SM90 path.

#### Source: `roy-deepseek-csa-skeleton.md`

### 2026-07-17: Land the DeepSeek CSA CPU reference seam on assembled f32 KV
**By:** Roy
**What:** Added `pkg.nxrt::SparseKvGather` v1 plus a reusable checked gather/index-planning module, and registered a Phase-1 `pkg.nxrt::CompressedSparseAttention` v1 CPU reference kernel. The fused skeleton consumes projected f32 queries, an assembled dense-window/compressed-history KV cache, selected indices, valid lengths, learned logit sinks, and optional additive bias; it calls the shared gather and computes sink-denominator attention. Exact window-128, ratio-4, and ratio-128 prefill/decode index sequences are unit-tested.
**Why:** The frozen source makes sparse gathering and candidate ordering independently testable, while the runtime does not yet own the complete stateful compressor/carry ABI. FP4/FP8 compressed-KV formats and boolean bias return explicit `Unsupported` errors rather than panicking. Phase 2 must add FP4 block-32 and FP8 block-64 dequantization, stateful compressor/carry and ratio-4 index-cache updates at the full production ABI, then eliminate materialized gathers and optimize decode/prefill performance.

#### Source: `sapper-onnxrs-gaps.md`

### 2026-07-17: Close priority onnx-rs inference-spec gaps
**By:** Sapper
**What:** Added ONNX-ML `TypeProto.Opaque`, expanded protobuf structural checking, and grew the built-in schema registry from 8 to 15 operators (22 versioned entries). Opaque uses `domain = 1`, `name = 2`, and `TypeProto.opaque_type = 7`. Checker additions cover metadata keys, attribute unions/references, recursive container types, dense payloads, sparse COO indices, and multi-device device-group maps. Added Softmax, LayerNormalization, Gather, Reshape, Transpose, Concat, and Slice schemas.
**Why:** These were the highest-priority non-training gaps in `docs/ONNX_RS_SPEC_COVERAGE.md`. Field numbers were verified against the official ONNX v1.20.0 ONNX-ML schema: https://raw.githubusercontent.com/onnx/onnx/v1.20.0/onnx/onnx-ml.proto. Remaining inference gaps are the complete standard/ONNX-ML schema catalogs, function validation, quantization annotation semantics, remaining IR gates, string UTF-8 and packed-padding checks, full mutation/Text grammar, shape inference, and version conversion. Training semantics remain explicitly out of scope.

#### Source: `sapper-onnxrs-round2.md`

### 2026-07-17: onnx-rs round-2 closes catalog-local shape inference and adds 12 schemas
**By:** Sapper
**What:** Added recursive ONNX `If` shape inference with branch output-count/type validation and shape union, plus representative onnx-rs graph tests covering the existing schema families. Added 12 standard schemas: Sigmoid, Tanh, Erf, Sqrt, Exp, Log, Pow, Clip, Expand, Where, ReduceSum, and ReduceMean. The catalog is now 27 operators / 34 versioned entries. Training semantics remain out of scope.
**Why:** Every operator currently registered by onnx-rs now has a shape rule, while unsupported operators still degrade to unknown. The schema batch targets common inference graphs without claiming complete standard or ONNX-ML catalogs.
**Sources verified:** ONNX tag `v1.20.0`: `onnx/defs/math/defs.cc`, `onnx/defs/tensor/defs.cc`, `onnx/defs/reduction/{defs.cc,utils.cc}`, `onnx/defs/schema.cc`, `onnx/defs/controlflow/utils.cc`, `onnx/defs/shape_inference.cc`, and `onnx/checker.cc`. The tag declares standard opset 25. Official `checker.cc` contains neither quantization-annotation target validation nor UTF-8 validation for protobuf byte payloads, so those audit items were reclassified rather than over-constrained.
**Remaining:** Complete standard/ONNX-ML schema and shape catalogs; function signature/default/topology/import/recursion checks; remaining IR-version gates; packed 2-bit/4-bit padding validation; full mutation/text grammar/opset conversion. Training validation/execution remains explicitly out of scope.

**Merged commits:** `ef40af8`, `6e82b05`, `4981dbf`, `62da14b`, `833f6da`, and `11a01a5`.
## 2026-07-17 — Wave 6 landings

#### Source: `deckard-onnxrs-r3-overconstraint.md`

### 2026-07-17: Match ONNX v1.20 checker semantics for three over-constraints
**By:** Deckard
**What:** ModelProto.metadata_props is validated without an IR-version presence gate; FunctionProto.domain accepts the empty default ONNX domain instead of requiring a non-empty value; packed sub-byte tensor payloads validate byte length but do not require unused high bits to be zero.
**Why:** ONNX v1.20 checker.cc applies duplicate-key validation to model metadata at every supported IR, uses `enforce_has_field(function, domain)` rather than a non-empty check, and imposes no value constraint on unused packed lanes. The vendored proto3 Rust representation does not retain scalar string wire presence, so it cannot distinguish an omitted FunctionProto.domain from an explicitly present empty domain after decoding; the typed checker therefore accepts the empty/default representation while preserving all other function checks.

#### Source: `leon-seq-ops.md`

### 2026-07-17: Execute ONNX Sequence operators in the session runtime
**By:** Leon
**What:** The executor keeps heterogeneous sequence values in run-scoped `HashMap<ValueId, SequenceValue>` storage, with a build-time `sequence_values` type set so sequence outputs never receive tensor buffers. `SequenceAt` tensor results retain their `SeqTensor` Arc in `seq_elem_values` for zero-copy host consumption. SequenceEmpty, SequenceConstruct, SequenceInsert, SequenceErase, SequenceAt, SequenceLength, SplitToSequence, and ConcatFromSequence are routed directly by the executor; all data operations delegate to the Phase 1 `SequenceValue`, `split`, and `concat` APIs.
**Why:** EP kernels only accept tensor views, so executor-owned heterogeneous storage is the minimal extension that preserves existing tensor buffer/view conventions while allowing sequence values between graph nodes. Graph-level sequence input/output bindings remain a later phase because the public `InferenceSession::run` boundary is tensor-only; internal sequence graphs are fully executable when they terminate in SequenceAt, SequenceLength, or ConcatFromSequence.

#### Source: `roy-deepseek-csa-p2.md`

### 2026-07-17: DeepSeek CSA Phase 2 compressed-cache dequantization
**By:** Roy
**What:** Added checked CPU reference decoding for FP8 E4M3FN block-64 and FP4 E2M1 block-32 caches. Packed records concatenate an E8M0 power-of-two scale byte with 64 FP8 bytes or 16 adjacent-nibble FP4 bytes; CSA dequantizes to f32, reuses `SparseKvGather`, and accumulates sink-aware attention in f32. The assembled-cache path accepts `cache_format=fp8_e4m3_block64` and `cache_format=fp4_e2m1_block32`.
**Why:** This enables correctness testing of compressed KV without full model goldens. The shared OCP E2M1/E8M0 scalar primitives were factored from `BlockQuantizedMatMul`; QMoE currently only implements integer affine dequant and had no FP4/FP8 helper to reuse. Hand-constructed blocks validate exact scale/code semantics, and FP4/FP8 CSA tests compare compressed execution with the same logical f32 cache. Stateful compressed-KV construction/carry updates, ratio-4 index-key construction/carry updates, and top-k selection remain explicit `Unsupported`; the MTP sidecar remains outside this path. Validation: `cargo test -p onnx-runtime-ep-cpu` passed 475 tests (1 ignored), plus 0 doc tests (1 ignored).

#### Source: `sapper-onnxrs-round3.md`

### 2026-07-17: ONNX-RS round-3 checker and schema coverage
**By:** Sapper
**What:** Bind the checker to ONNX v1.20/IR 13, validate inference-field introduction gates, add structural FunctionProto validation, enforce zero packed padding bits, and register Sub/Div/Neg/Abs/Mod schemas with shape tests.
**Why:** These were the highest-priority non-training gaps in the coverage audit. Function optional fields remain optional; call-site/declaration consistency stays listed as a remaining gap because v1.20 checker.cc also marks that consistency check TODO.
## 2026-07-17 — Wave 6 correction

#### Source: `chew-csa-rope-tail.md`

### 2026-07-17: Preserve the CSA RoPE tail outside FP8 blocks
**By:** Chew
**What:** The assembled CSA FP8 cache record is hybrid: E4M3 block-64 bytes encode only `head_dim - qk_rope_head_dim`, followed by little-endian BF16 values for the RoPE tail. FP4 remains full-width block-32.
**Why:** The frozen DeepSeek contract applies attention FP8 simulation only to non-RoPE dimensions after the BF16 RoPE update, so quantizing the tail changes attention numerics.
## 2026-07-17 — Wave 7 landings and corrections

#### Source: `chew-csa-p3-golden.md`

### 2026-07-17: Ratio-128 carry resets after every emitted block
**By:** Chew
**What:** The ratio-128 CSA path now clears all `[B,128,2,512]` carry slots after every completed block, including incremental decode. Its boundary golden uses varied nonzero tensors at block start 128, exact carry/cache comparisons, and independent pooling, BF16/RMSNorm/RoPE/FP8 oracles. FP8 error is bounded by `16 * block_scale`; FP4 by `1 * block_scale`.
**Why:** The previous decode path retained stale completed-block slots while full prefill recompute cleared them, violating the frozen carry-state contract. E4M3's maximum code spacing is 32 (half-ULP 16), while E2M1's is 2 (half-ULP 1), so these bounds replace arbitrary tolerances.

#### Source: `deckard-if-unknown-fix.md`

### 2026-07-17: Do not seed empty produced shapes as resolved metadata
**By:** Deckard
**What:** Shape inference now treats an empty shape on a produced IR value as omitted placeholder metadata. The producer must infer that output before it can participate in `If` branch merging; non-empty declared intermediate shapes remain available to downstream inference.
**Why:** Seeding empty produced shapes as known rank-0 tensors made unsupported branch outputs appear resolved, causing `If` to invent a scalar output and undercount unresolved values. This preserves concrete branch merges and declared non-empty `value_info` while keeping unresolved branches unresolved.

#### Source: `deckard-onnxrs-r4-fix.md`

### 2026-07-17: Match ONNX v1.20 local-function call-site acceptance
**By:** Deckard
**What:** onnx-rs does not enforce local-function call-site arity, required-attribute, or undeclared-attribute consistency.
**Why:** Official ONNX v1.20 `checker.cc` leaves consistency between local functions and referencing ops as a TODO, so rejecting these mismatches would break checker acceptance continuity.

#### Source: `leon-if-scalar-fix.md`

### 2026-07-17: Preserve absent-versus-scalar shape semantics in If inference
**By:** Leon
**What:** Shape inference seeds every value whose IR shape is marked known, including produced rank-0 tensors with an empty dimension list. Only values explicitly marked with `mark_value_shape_unknown` are treated as shape-absent; unresolved `If` tests must mark both branch and parent outputs unknown.
**Why:** ONNX uses an empty present shape for a known rank-0 scalar, while an absent shape is unknown. Filtering produced empty shapes conflates these states and discards valid scalar metadata.
## 2026-07-17 — Wave 7 source decisions

#### Source: `roy-csa-p3.md`

### 2026-07-17: Land ratio-128 CSA attention-compressor state first
**By:** Roy
**What:** Implement the frozen-v1 ratio-128 attention-compressor stream with persistent compressed records, `[B,128,2,D]` KV/score carry, block-boundary pooling, BF16→RMSNorm→block-start RoPE→hybrid FP8/BF16 finalization, and sink-aware f32 attention. Keep ratio-4 index state/top-k and MTP explicitly Unsupported.
**Why:** Ratio-128 is the foundational non-overlapping state transition and can be validated independently across incremental decode boundaries without inventing the still-deferred ratio-4 selection or any MTP recurrence.

#### Source: `sapper-onnxrs-r4.md`

### 2026-07-17: Round-4 ONNX schema and function-checker slice
**By:** Sapper
**What:** Added a 12-operator normalization/reduction/arg-reduction schema cluster with shape inference coverage, and advanced local-function call-site checking with exact arity plus required/default/undeclared attribute validation.
**Why:** These operators are high-value for transformer and MoE graphs, reuse coherent official reduction semantics, and the call-site slice closes the most actionable structural gap without over-constraining optional default attributes.

#### Source: `wallace-bert-gather.md`

### 2026-07-17: Preserve declared intermediate value_info during shape re-inference
**By:** Wallace
**What:** Whole-graph shape inference now seeds every explicitly known value type, including intermediate ONNX `value_info`, while allowing producer rules to overwrite those seeds when they resolve a fresher type.
**Why:** The optimizer passes did not rewire BERT's Gather. The post-optimization re-inference discarded the declared `[1, 8]` shape of Gather indices `106` because its `Expand` producer could not be re-resolved through a dynamic Slice chain. Gather then treated the missing indices shape as scalar and overwrote output `108` from `[1, 8, 32]` to `[32]`, causing the validated kernel allocation mismatch. Preserving explicit intermediate metadata keeps optimized graphs reference-equivalent without weakening Gather validation.
## 2026-07-17 — Wave 8

### 2026-07-17: CUDA QMoE Phase 1 parity baseline
**By:** Mariette
**What:** CUDA `com.microsoft::QMoE` now implements top-k routing, optional aggregation weights, affine INT4/INT8 block dequantization, biases, FC3/SwiGLU, every CPU-parity activation mode, and f32/f16/bf16 storage with f32 accumulation. Launch sizing derives from compute capability, thread limits, and SM count rather than hardcoding SM90.
**Why:** This establishes a portable GPU QMoE baseline with five CPU-vs-CUDA conformance tests and 148 passing CUDA tests. Paging/prefetch and multi-GPU expert sharding remain deferred.
**Landed:** `0fa3557`; reviewed 🟢 by Holden.

### 2026-07-17: onnx-rs round-5 schemas and conservative dynamic Split/Range inference
**By:** Sapper, repaired by Leon
**What:** Added schemas and shape inference for GatherElements, GatherND, Equal, Greater, Less, And, Or, Not, Cast, Shape, Size, NonZero, Range, and Split, plus a Softmax/LogSoftmax opset 12→13 version-conversion rewrite. Dynamic Split inputs now leave split-axis extents unresolved while preserving non-axis dimensions; Range rejects the rounded `2^63` (`isize::MAX as f64`) boundary.
**Why:** Bryant rejected the original `8d1d41b` because it fabricated equal output shapes for dynamic Split and let floating Range lengths bypass overflow rejection. Sapper was locked out; Leon repaired both failures in `93246b9`, landed as `d39eb54`. `onnx-rs` has 145 and shape-inference 149 passing tests.
**Review:** Bryant 🔴→🟢.

### 2026-07-17: CUDA QMoE Phase 2 batched prefill path
**By:** Mariette
**What:** Prefill now counts tokens per expert on GPU, prefix-scans and buckets them, gathers contiguous f32 inputs, runs tiled INT4/INT8 dequant-GEMM, scatters route slots, and combines weighted top-k routes in order. It selects GEMM for at least two tokens per expert, GEMV otherwise and for M==1; SM-derived 8/4 tiles fall back to 2/1 under shared-memory pressure.
**Why:** The batched path keeps sparse expert prefill on GPU while retaining the optimal GEMV path for small buckets. GEMV-vs-GEMM-vs-CPU-oracle, empty, and all-to-one-expert coverage pass; the CUDA suite has 151 passing tests.
**Landed:** `52918a8`; reviewed 🟢 by Holden.

### 2026-07-17: WEIGHT_OFFLOAD Phase 2 per-engine bounded LFRU cache
**By:** Chew, repaired by Deckard and Leon
**What:** Expert offload uses bounded host-RAM LFRU caches with Arc leases, pinning, hysteresis, strict byte reservation, oversized direct-mmap, zero-byte fallback, governor integration, and metrics. Cache ownership and caps are per engine; uncacheable history is skipped and LFRU-pruned at 4096. Global residency/effective-budget metrics use checked delta aggregation across live caches, and Drop saturating-subtracts each cache contribution exactly once.
**Why:** Nabil rejected `19e90ff` because a last engine could overwrite the global cache budget and history grew without bound. Chew was locked out; Deckard's `da2d1be` added per-engine partitioning and bounded history, but was rejected because global residency remained a clobbered/stale atomic. Deckard was locked out; Leon fixed accounting in `ceda4e7`, landed as `f80ca09`. Leased/pinned entries remain non-evictable, cached output is bit-identical to direct mmap, and zero-byte behavior remains Phase 1-compatible.
**Review:** Nabil 🔴→🔴→🟢; CPU EP 487 and Engine 141 tests pass.
## 2026-07-17 — Overnight perf and feature decisions

### Performance scalability

### 2026-07-17: Graph partition/fusion performance audit (10k+ nodes)
**By:** Wallace
**What:** Verdict: no, the full pipeline is not generally 10k+-scalable yet. Default single-EP Executor build is near-linear (41.6 ms at 20k), but shape inference is O(NV), constant folding/fusion and many-boundary EPContext splicing have quadratic paths, there is no active heterogeneous partitioner, and static buffers have no liveness reuse.
**Why:** Justin: must stay performant on very large models.

### 2026-07-17: Constant folding uses a dependency worklist (O(N+E))
**By:** Wallace
**What:** Replaced global reverse-scan fixpoint with unresolved-input-count worklist.
**Why:** Audit hotspot #2 — O(N^2), 11.6s at 20k nodes. Scheduling-only change; folded set identical.

### 2026-07-17: Review — constant-folding worklist (Wallace e623ee1)
**By:** Deckard
**Verdict:** 🔴
**Scheduling-only (folded set identical)?:** No. The fold/evaluation predicates are otherwise preserved: domain, single-output restriction, supported ops, inline-initializer definition, Shape guards, integer dtype/shape/size checks, and checked arithmetic match the parent implementation. However, FIFO changes the relative order of initially-ready nodes and newly-ready lower-NodeId consumers. A concrete graph with NodeId 0 `Constant -> a`, NodeId 1 `Add(a, initializer) -> b`, and NodeId 2 `Shape(b) -> graph_output` folds all three under the old ascending reverse-scan: Constant makes Add ready, Add folds while Shape still consumes `b`, then Shape folds. The new queue starts `[Constant, Shape]`; Constant appends Add behind Shape, Shape folds first and removes the sole consumer of `b`, then Add fails the unchanged `needed` guard and remains. An external read-only probe against e623ee1 produced `remaining_nodes=1`, `remaining_op=Add`, `b_initializer=false`, with a valid graph. Thus the folded set and serialized graph are not identical.
**Counter correctness (multi-output/repeated/optional inputs):** The edge counters themselves are sound: repeated unresolved inputs are counted and registered once per edge, so they decrement the same number of times; missing optional inputs exclude the node just as evaluation previously refused it; non-constant graph inputs/external initializers never emit readiness; candidates remain single-output, so a multi-output producer cannot spuriously publish all outputs, and dependencies are keyed by the specific `ValueId`. The rejection is the queue/liveness interaction above, not an arithmetic counter underflow.
**Determinism:** Deterministic but not semantics-preserving. Arena iteration and dependent insertion are ascending NodeId, FIFO is stable, and HashMap iteration order is not used for scheduling.
**Termination / no-double-fold:** `unresolved.remove(&nid)` ensures at most one attempt per candidate. The reverse-NodeId chain test genuinely constructs descending dependency NodeIds and proves notification propagation; it would leave nodes behind if decrements/enqueues were broken. It does not cover ready-vs-newly-ready ordering interacting with the `needed` guard.
**Graph-output & liveness:** Graph outputs remain live through `remove_node` and are re-backed by an inline initializer correctly. Consumer bookkeeping is exactly what exposes the semantic regression: folding an independent `Shape` first can make its producer output dead before that producer's scheduled attempt.
**Tests:** `cargo test --locked -p onnx-runtime-optimizer` passed 48 tests total (47 unit + 1 integration; 0 failures, 0 ignored; 0 doc tests). Coverage misses the schedule-sensitive liveness case above.

Wallace is locked out from revising this rejected artifact; the coordinator must assign the fix to a different agent.

### 2026-07-17: Constant folding worklist reproduces original ascending-wave order
**By:** Holden
**What:** Replaced Wallace's FIFO with a wave-structured ascending-NodeId schedule (updates visible to higher NodeIds within a wave, lower NodeIds deferred to next wave), making the folded set byte-identical to the original fixpoint. Kept the O(N+E) edge counters.
**Why:** Deckard 🔴 — FIFO interacted with the liveness `needed` guard, leaving `Add` unfolded in a Constant→Add→Shape graph. Fixes on top of Wallace e623ee1.

### 2026-07-17: Re-review — constant-folding wave order (Holden 80b0aaf)
**By:** Deckard
**Verdict:** 🟢 CLEARS TO LAND
**Counterexample folds all 3?:** Yes. `Constant(0) → Add(1) → Shape(2)` leaves `remaining_nodes == 0`, no `Add`, and serialized output exactly equal to the original ascending-fixpoint reference.
**Wave logic faithful to original?:** Yes. The min-heap is an ascending-pass cursor: a newly-ready consumer above the processed `NodeId` joins the current wave; a lower/equal consumer waits for the next wave. Unique node IDs make equality with a distinct pending node impossible; multiple consumers spanning the cursor are independently placed on the correct side.
**Adversarial probes:** Split fan-out (`Constant(2)` readies `Add(0)` and `Add(4)`) matched pass semantics: 4 runs now, 0 next. The lower-dead-producer graph (`Add(0) → Shape(1)`, then `Constant(2)` readies Add) matched the original and intentionally leaves Add unfolded. Two-producer diamonds with the last producer in both an earlier and later wave matched. A 500,000-case randomized DAG schedule model (2–10 nodes, static/dynamic Shape, duplicate Add inputs, arbitrary NodeId/topology order and liveness outputs) found no divergence. The proposed “higher newly-ready node kills an unrelated deferred lower producer” is not realizable with these foldable operators: Add cannot become ready before every data producer, while Shape has only its triggering input.
**Seeded tests cross-check a faithful reference?:** Yes. `run_reference_ascending_fixpoint` is the pre-Wallace implementation: repeated full ascending `NodeId` snapshots, in-place updates visible later in the pass, and the same domain/output-count/evaluator/`needed` guards. The 32 seeded DAGs serialize the new result and this independent reference and compare bytes.
**Tests:** optimizer 51 passed (50 unit + 1 integration; 0 failed). Deterministic raw-`NodeId` heap ordering; every candidate is dequeued/removed from `unresolved` at most once, so wave churn cannot be infinite.

### 2026-07-17: EPContext splice hoists graph_outputs / avoids per-partition full scans
**By:** Leon
**What:** Precompute graph_outputs once; partition_boundary no longer rebuilds it per partition. Splice pass now ~linear in total covered nodes, not partitions×graph.
**Why:** Audit hotspot #3 — O(P·G) per-partition rescan, ~4.8s at 20k nodes. Serialized model byte-identical.


### WEIGHT_OFFLOAD Phase 3a

### 2026-07-17: Keep Phase 3a device offload planning pure and capability-negotiated
**By:** Chew
**What:** Phase 3a translates `gpu_layers:N` to bytes under a coordinated weight ceiling, places whole layers greedily, protects a committed-sequence KV floor with hysteresis and dwell, and exposes lazy `pkg.nxrt::BlockQuantizedMoE` weights only to EPs advertising `nxrt`. Stock EPs materialize resident weights; live device binding remains explicitly unsupported until Phase 3b.
**Why:** This lands the policy, accounting, and executor seam with CPU-only deterministic tests while preserving Phase 1/2 behavior and preventing compatibility overrides from stealing KV or scratch VRAM.

### 2026-07-17: Review — WEIGHT_OFFLOAD P3a
**By:** Nabil
**Verdict:** 🔴 Red (blocking)
**Findings:**
- **[High]** `crates/onnx-runtime-ep-cpu/src/weight_offload/placement.rs:425-434` ignores `requested_kv_bytes` whenever KV pressure is low and reduces the target directly to the committed floor. After dwell, pending batch-admission demand can therefore be discarded. There is also no admission decision/result at batch formation. This violates §11 Q8 (L745-748), which makes admission control and continuous-batching admission tests hard gates. **Fix: Leon.**
- **[High]** `crates/onnx-runtime-ep-cpu/src/weight_offload/weight_handle.rs:154-173` defines negotiation only inside the CPU EP crate, with no production call sites, while `crates/onnx-runtime-ep-api/src/kernel.rs:130` still exposes only `TensorView` inputs. Consequently, an `nxrt` EP cannot actually receive a lazy `WeightHandle`, and the stock-EP materialized path is not verified through the unchanged executor path. This violates §11 Q1 (L700-703), which requires a general executor handle with capability-detected lazy delivery and resident fallback. **Fix: Deckard.**
- **[Pass]** `plan_placement` is deterministic and explainable, reports `gpu_layers:N` in bytes, and keeps every region of a layer on one device (§11 Q5-Q6).
- **[Pass]** The committed KV floor rejection, watermark hysteresis, dwell behavior, whole-quant-block snapping, P1/P2 preservation, and Phase-3b device-paging deferral are correctly covered. `cargo test --locked -p onnx-runtime-ep-cpu`: 498 passed, 0 failed, 1 ignored.
**Fix owner (if 🔴):** Leon for governor/admission; Deckard for the executor/EP `WeightHandle` seam. Chew remains locked out.

### 2026-07-17: WEIGHT_OFFLOAD P3a fixes — KV admission + WeightHandle EP seam
**By:** Deckard
**What:** KV arbitration now preserves outstanding requested admission capacity under low residency pressure and returns an explicit admitted/rejected batch decision with the granted/current KV budget and limiting factor. The shared ep-api now exposes capability-advertised lazy `WeightHandle` kernel inputs, while the executor selects them only for `pkg.nxrt::BlockQuantizedMoE`; stock EPs retain the resident `TensorView` dispatch path and Phase 3a device binding remains unsupported.
**Why:** §11 Q8 makes deterministic continuous-batch admission a hard gate and defines the committed KV floor as a minimum rather than a target ceiling. §11 Q1 requires a general executor weight handle with capability-detected lazy delivery and resident fallback for existing EPs.

### 2026-07-17: Re-review — WEIGHT_OFFLOAD P3a fixes (Deckard 3425498)
**By:** Nabil
**Verdict:** 🔴
**Finding #1 (KV admission):** Resolved. `requested_kv_bytes` now participates in both pressure branches, every successful arbitration path derives an explicit `KvAdmissionDecision`, dwell/hysteresis returns an explicit rejection rather than dropping demand, and admission uses the returned state's current KV sub-budget. Checked arithmetic bounds the target by `total - scratch` and uses checked/u128 arithmetic.
**Finding #2 (WeightHandle seam):** Not fully resolved. `WeightHandle` correctly moved into ep-api, ep-cpu coherently re-exports it, the executor capability-gates lazy delivery specifically for `pkg.nxrt::BlockQuantizedMoE`, stock EPs retain `TensorView` dispatch, and the lazy materializer keeps `WeightStore` alive via `Arc`. However, the executor still eagerly creates a concrete buffer for every initializer before execution (`executor.rs:925-979`). For a non-host nxrt EP this allocates and copies the entire supposedly lazy weight to device (`executor.rs:970-973`), then replaces its view with `KernelInput::Weight` only at dispatch (`executor.rs:2127-2143`). This double residency can OOM during build and defeats device offload; the CPU-only seam test masks it by borrowing the mmap.
**New issues:** High — nxrt lazy initializers must skip eager resident device allocation when they have no resident consumers, or allocation must be demand-driven/shared so the lazy and resident paths cannot materialize the same weight twice. Add a non-host test EP that fails/counts initializer allocation to prove the lazy path does not allocate/copy the full weight.
**Tests:** ep-api 32 passed, ep-cpu 497 passed (0 failed; ep-cpu 2 ignored across unit/doc targets). Targeted session seam test: 1 passed.

### 2026-07-17: nxrt lazy initializers skip eager device residency
**By:** Leon
**What:** Executor no longer eagerly allocates/copies device buffers for initializers consumed only by lazy nxrt BlockQuantizedMoE kernels; when a resident consumer coexists, all consumers share one eager resident buffer.
**Why:** Nabil re-review 🔴 Finding #2 — double residency → build-time OOM, defeated device offload. Fixes on top of Deckard 3425498.
**Design choice:** skip-when-no-resident-consumer — exclusive lazy nxrt initializers omit non-host eager residency. An initializer with any resident consumer, graph-output use, or producer does not receive a lazy handle; it is allocated once and delivered as the same resident TensorView to every consumer, preventing duplicate materialization and making read order irrelevant. Host-accessible EPs retain the existing zero-copy mmap borrow.

### 2026-07-17: Re-review #2 — WEIGHT_OFFLOAD P3a nxrt alloc fix (Leon aa17f51)
**By:** Nabil
**Verdict:** 🟢
**Skip-eager-alloc correct (lazy-only)?:** Yes. `build_lazy_weight_handles` now requires a producer-less, non-graph-output external initializer with at least one consumer and verifies every consumer is `pkg.nxrt::BlockQuantizedMoE`. Only those handles skip non-host allocation in `Executor::build`; any resident consumer or direct graph-output use prevents the handle and preserves eager residency.
**Both-consumer shared materialization?:** Yes. Mixed lazy/resident consumers fail the all-consumers boundary test, so no lazy handle is created. One eager device buffer/upload backs immutable `TensorView`s for both consumers. The regression test confirms one upload, identical bytes from both outputs, and resident delivery to both kernels.
**Proof test asserts zero eager alloc?:** Yes. The non-host test asserts the lazy case has exactly one total allocation (the output) and zero host uploads, versus two allocations and one upload for the stock EP. The pre-fix unconditional initializer allocation makes the lazy case two allocations, so the regression distinguishes 2 vs 1 rather than merely checking a reduction.
**New issues:** None. The `WeightStore` is retained by `Arc` in the lazy materializer, the skip path adds dtype/shape metadata before continuing and contains no new unwrap/panic, host CPU mmap borrowing and the `producer_less` guard remain intact, stock EPs retain eager residency, and graph-output/mixed-consumer initializers retain backing buffers. **Tests:** ep-api 32, ep-cpu 497, session 134 passed (0 failures; ep-cpu has 1 ignored unit and 1 ignored doc test, session has 1 ignored doc test).


### EP claim diagnostics

### 2026-07-17: EP claim declines carry actionable reasons (mlx-style)
**By:** Roy
**What:** KernelMatch::Unsupported now carries a colocated Cow reason; EPs use deny!/require!; session surfaces per-node reasons.
**Why:** Justin — match onnxruntime-mlx so users can debug why a node fell back off an EP.

### 2026-07-17: Review — EP claim-reason diagnostics (Roy 7ad0567)
**By:** Holden
**Verdict:** 🔴
**M=1 removal safe?:** Yes. The removed gate was only the stale session-owned coverage message claiming `BlockQuantizedMatMul` supported M=1 only; the parent commit already claimed symbolic/M>1 shapes and dispatched `m == 1` to GEMV and `m != 1` to the CUDA GEMM added in `a99f7a8` (`block_quantized_matmul.rs:620-679`). The GPU claim/numerics regression passed 6/6, including M>1 prefill. No architecture hardcode was introduced: device capability is queried at runtime (`runtime.rs:180-203`) and NVRTC targets are derived from it.
**Reason quality:** The new registry, fused-GEMM shape, block-format/layout/attribute, and QMoE quantization reasons name the offending contract and accepted remediation. Blocking gap: the required default “no handler for {op} at opset {v}” path does not exist. `supports_op` has no opset argument and calls opset-blind `OpRegistry::supports` (`ep-api/src/provider.rs:262-275`, `registry.rs:71-80`), so CPU can claim registrations that begin later—for example `Attention` since opset 23 and `Gelu` since opset 20 (`ep-cpu/src/kernels/mod.rs:285-299,347-358`). `get_kernel` then fails with `NoEpForOp`, whose message carries neither domain nor opset (`ep-api/src/lib.rs:51-54`), bypassing the new actionable `KernelMatch` reason (`session/src/executor.rs:202-213`). This leaves an EP-API/session boundary broken and can falsely claim an un-runnable opset. Roy is locked out; Deckard or Leon should own the revision.
**No-alloc-on-accept:** Confirmed. `require!` evaluates `format!` only inside the failed-condition branch, and `deny!` formats only immediately before returning the decline. The accepted path constructs no reason string.
**Boundary coherence:** The new `KernelMatch::Unsupported { reason }` shape is otherwise coherent across ep-api, CPU, CUDA, session, and test EPs; compilation found no bare variant or unhandled match. CUDA coverage collection is a linear per-node DFS with no O(N²) aggregation; it does not currently aggregate counts by op type.
**Tests:** ep-api 29 passed; ep-cpu 488 passed, 2 ignored; session 132 passed, 1 ignored; CUDA 166 passed on NVIDIA H200 (SM 9.0). Targeted CUDA `block_quantized_matmul_gpu`: 6/6 passed.

### 2026-07-17: EP claim path is opset-aware with default fallback reason
**By:** Wallace
**What:** `supports_op` gains an opset argument; too-new ops are declined at claim time with an actionable reason; `NoEpForOp` now carries domain + op + opset; the default `no handler for {domain}::{op} at opset {v}` fallback was added.
**Why:** Holden 🔴 — opset-blind claim falsely claimed later-opset ops (`Attention`@23, `Gelu`@20), then failed with opaque `NoEpForOp`, bypassing actionable reasons. Fixes on top of Roy `7ad0567`.

### 2026-07-17: Re-review — EP opset-aware claim (Wallace dbf7847)
**By:** Holden
**Verdict:** 🟢
**Opset gap closed (decline-at-claim, no NoEpForOp bypass)?:** Yes. `supports_op` now gates the generic registry by `(domain, op, model opset)` and returns `KernelMatch::Unsupported` with domain, model opset, and earliest registered since-opset. Gelu@19 reaches actionable `UnsupportedOp`; Gelu@20 is claimed. The same generic mechanism covers Attention: its registry entries begin at 23, so 22 has no match and 23 matches. The session also translates any drift-induced `NoEpForOp` into the actionable unsupported-op route.
**Since-opset boundary exact (no over-constraint)?:** Yes. Registry matching remains `since_version <= opset`; tests cover Gelu 19/20 and Attention lookup 22/23, including later Attention revisions.
**Per-domain opset source correct?:** Yes. `effective_opset` reads the node's exact domain import, aliases only `""` and `ai.onnx`, and preserves independent imports such as `com.microsoft`; both claim and kernel creation receive that same value.
**All call sites coherent?:** Yes. CPU, CUDA, EPContext/mock EPs, `EpRegistry::candidates_for_op`, executor cache claims, and `collect_cuda_coverage_issues` pass an explicit opset. `NoEpForOp` now carries domain and opset. Decline reasons are formatted only inside the unsupported branch. Wallace did not disturb Roy's previously cleared M=1 removal.
**Tests:** ep-api 29 passed; cpu 489 passed (1 ignored); session 134 passed. CUDA test targets compiled successfully with `cargo test --locked -p onnx-runtime-ep-cuda --no-run`; CUDA tests were not run.

**Decision:** CLEARS TO LAND.


### onnx-rs schema and inference

### 2026-07-17: Use ONNX v1.20 canonical movement schemas and unresolved dynamic shapes
**By:** Sapper
**What:** Round 6 registers Tile-13, Pad-25, ScatterND-18, ScatterElements-18, and ConstantOfShape-25; Slice-13, Concat-13, and Expand-13 were already present. Runtime-computed Slice/Pad/Tile values preserve only justified rank/dimensions, while dynamic Expand and ConstantOfShape shape inputs leave output shape unresolved.
**Why:** These versions and signatures match the ONNX v1.20 opset-25 registry, and unresolved dynamic values must not be mistaken for rank-0 scalars or fabricated concrete shapes.

### 2026-07-17: Reject guaranteed Pad and Concat overflow with symbolic dimensions
**By:** Deckard
**What:** Pad now rejects a positive total padding above `isize::MAX` before inspecting whether the input extent is concrete, and Concat rejects a known concat-axis partial sum above `isize::MAX` even when another extent is symbolic. Normal symbolic cases remain unresolved.
**Why:** ONNX extents are non-negative, so either quantity is a guaranteed lower bound on the final output extent and must be rejected before allocation-size inference can accept it.

### 2026-07-17: Round-7 canonical ONNX schema revisions
**By:** Sapper
**What:** Register MaxPool/AveragePool/GlobalAveragePool/GlobalMaxPool at revision 22, Resize at 19, QuantizeLinear/DequantizeLinear at 21, and DynamicQuantizeLinear at 11.
**Why:** These are the requested canonical revisions from ONNX v1.20.0; quantization revision 21 is the first requested revision covering blocked quantization, uint4/int4, and all four float8 formats.

### 2026-07-17: Review — onnx-rs round-7
**By:** Bryant
**Verdict:** 🔴
**Findings:**
- `crates/onnx-runtime-shape-inference/src/handlers/movement.rs:391-393,406-410` — **High** — Resize acceptance is over-constrained. `roi` rank is checked even when `coordinate_transformation_mode` is not `tf_crop_and_resize`, and inference rejects the officially accepted form with both optional `scales` and `sizes` absent. This violates review clause 1; `tests/op_rules.rs:2349-2366` currently locks in the wrong rejection.
- `crates/onnx-runtime-shape-inference/src/handlers/movement.rs:367-382,452-472` — **Medium** — Resize extent arithmetic lossily converts valid `i64` extents and `isize::MAX` to `f32`. An input extent of `isize::MAX` with scale `1.0` rounds to `2^63` and is rejected, violating checked-overflow clause 4.
- `crates/onnx-runtime-shape-inference/src/handlers/pooling.rs:179-205,287-306` — **Medium** — Symbolic pooling checks only the final lower-bound result, so known partial extents exceeding `isize::MAX` can cancel without rejection. For example, `kernel=isize::MAX`, `dilation=2`, and pads `[isize::MAX,isize::MAX]` produce an effective kernel and pad sum above the limit but a lower-bound output of 2. This violates clause 4.
- `crates/onnx-runtime-shape-inference/src/handlers/data_ops.rs:292-319` — **High** — Rank-1 blocked DequantizeLinear is misclassified as per-axis before `block_size` is considered. Valid `x=[8]`, `scale=[2]`, `zero_point=[2]`, `axis=0`, `block_size=4` is rejected although opset-21 blocked shape is `ceil(8/4)=2`. This violates clause 5.
- `crates/onnx-runtime-shape-inference/tests/op_rules.rs:2191-2479` — **Low** — Coverage exercises ordinary ceil/auto-pad and rank-2 blocked dequantization, but misses ignored non-vector ROI, absent Resize controls, maximum-boundary Resize, cancellation-masked symbolic pool overflow, and rank-1 blocked quantization.

Independent validation passed: onnx-rs 149 tests and shape-inference 173 tests. The failures are uncovered acceptance/overflow cases.

**Fix owner:** Deckard. Sapper is locked out from revising this rejected artifact.

### 2026-07-17: onnx-rs r7 fixes — Resize/pool/dequant acceptance + overflow
**By:** Leon
**What:** Resize now ignores ROI outside `tf_crop_and_resize`, accepts absent scales and sizes with unresolved output dimensions, and performs scale/aspect-ratio extent arithmetic without lossy large-integer conversion. Pooling rejects oversized effective kernels, padding sums, and padded-input partial extents before cancellation. Opset-21 blocked QuantizeLinear and DequantizeLinear are classified by nonzero `block_size` before rank-1 per-axis handling and validate scale/zero-point blocked shapes.
**Why:** Remove over-constraint, enforce checked overflow for symbolic shapes, and classify blocked quantization correctly so inference matches official ONNX v1.20 acceptance.

### 2026-07-17: Re-review — onnx-rs r7 fixes (Leon 7d75c02)
**By:** Bryant
**Verdict:** 🔴
**#1 Resize acceptance:** Resolved. ROI validation is conditional on `tf_crop_and_resize`; ignored/empty ROI, scales-only, sizes-only, and absent scales+sizes are accepted, while simultaneous non-empty scales+sizes remains rejected.
**#2 rank-1 blocked Dequant:** Not fully resolved. Valid rank-1 blocked Quantize/Dequantize shapes now pass and invalid blocked extents fail, but `data_rank == 1` unconditionally forces `axis = 0` (`data_ops.rs:301-303`). Thus a truly invalid rank-1 blocked Dequantize with `axis=1` or `axis=-2` is accepted instead of enforcing the opset-21 range `[-1, 0]`. No regression test covers this reject path.
**#3 Resize integer extents:** Resolved. Scale extents use exact binary-significand integer arithmetic, aspect-ratio extents use checked rational arithmetic, and `isize::MAX` is never converted through float.
**#4 symbolic overflow:** Resolved. Effective-kernel, padding-sum, and known padded-input partial extents are checked against the integer sentinel before cancellation.
**New over-constraint?:** None introduced by the fix. New correctness gap found in rank-1 blocked axis validation, so finding #2 cannot clear.
**Tests:** onnx-rs 149, shape-inference 176 (all passed, including doc-tests).

### 2026-07-17: rank-1 blocked Quantize/Dequantize validates axis range [-1,0]
**By:** Mariette
**What:** Reject out-of-range axis (e.g. 1, -2) for rank-1 blocked Quantize/DequantizeLinear before normalizing; only axis in [-1,0] accepted (−1→0). Added reject + accept regression tests.
**Why:** Bryant re-review 🔴 finding #2 — `data_rank==1` unconditionally forced axis=0, silently accepting illegal axes. Fixes on top of Leon 7d75c02.

### 2026-07-17: Re-review #2 — onnx-rs r7 rank-1 axis fix (Mariette c990f6b)
**By:** Bryant
**Verdict:** 🟢
**Reject path (axis 1, -2):** Resolved. Both operators call the same validator, which now passes the raw axis through `checked_axis`; for rank 1 it accepts exactly `[-1, 0]` and returns a proper `ShapeInferError::Invalid` for every other value. Tests directly cover Dequantize `1`/`-2` and Quantize `1`; Quantize `-2` follows the identical shared path.
**Accept path (axis 0, -1) not over-constrained?:** Yes. Both Quantize and Dequantize explicitly pass with `0` and `-1`; `checked_axis(-1, 1)` normalizes to `0`, preserving the output shape byte-for-byte.
**Raw-before-normalize?:** Yes. `raw_axis` is read first, then validated/normalized by `checked_axis`; the old rank-1 force-to-zero branch is gone.
**New over-constraint?:** None. Higher-rank behavior is unchanged because it already used this exact validation. Rank-1 non-scalar invalid axes are now correctly rejected; scalar-scale paths still return before axis validation.
**Tests:** onnx-rs 149, shape-inference 177, all passing under `cargo test --locked -p onnx-rs -p onnx-runtime-shape-inference`. The new regression fails on the parent implementation because its first illegal rank-1 case returns `Ok`, making `unwrap_err()` panic.

**Decision:** Finding #2 is fully resolved. onnx-rs r7 CLEARS TO LAND.


### Shape inference and CUDA coverage

### 2026-07-17: If shape inference degrades gracefully on output-count mismatch
**By:** Leon
**What:** The If handler now reconciles branch outputs positionally up to the minimum declared/branch output count, ignores extra branch outputs, and leaves extra node outputs unresolved.
**Why:** Shape inference must never fail a model accepted by loader legality checks; output-count mismatches are an over-constraint and must degrade gracefully.

### 2026-07-17: Review — If shape-inference over-constraint hotfix
**By:** Bryant
**Verdict:** 🟡
**Findings:**
- `crates/onnx-runtime-shape-inference/src/infer.rs:198-274` — No blocking issue. Positional inference is bounded by all three output counts; paired dtype/rank/shape reconciliation is unchanged, and `resize_with` only appends unresolved node outputs because `paired_outputs <= node.outputs.len()`. Equal-count behavior is therefore unchanged.
- `crates/onnx-runtime-shape-inference/tests/graph_inference.rs:235-271` — Low: both count-mismatch directions and paired shape inference are exercised, but the new fixtures use the same `Float32` placeholder and branch dtype, so they do not non-degenerately prove dtype propagation under mismatched counts. This is a test-strength advisory, not a landing blocker.
- Scope is limited to the `If` handler and its graph-inference tests.
- Verification passed: `cargo test --locked -p onnx-runtime-shape-inference -p onnx-runtime-loader`, including `identical_value_names_in_separate_subgraph_scopes_load`.

### 2026-07-17: CUDA QMoE coverage follows the ORT affine contract
**By:** Mariette
**What:** CUDA `com.microsoft::QMoE` now accepts affine INT1/INT2 in addition to INT4/INT8 on both decode GEMV and grouped-prefill GEMM paths. All CPU activation contracts are covered by GEMV/GEMM/CPU differential tests. Native IQ/MXFP4 blocks remain explicit `Unsupported`.
**Why:** INT1/INT2 share QMoE's byte-dividing affine packing and are directly CPU-oracle testable. IQ/MXFP4 use self-contained GGUF block layouts that QMoE's separate weights/scales/zero-points inputs cannot represent; those formats need the planned block-quantized MoE operator rather than an ambiguous or incorrect QMoE encoding.


### CSA contract

### 2026-07-17: Follow the source-defined ratio-4 overlap and shared top-k
**By:** Roy
**What:** CSA ratio-4 emits one compressed record per four source tokens. Its overlap factor is two projection channels with an eight-slot carry: previous-block left channels plus current-block right channels. Index-head scores are reduced before one shared top-k; the diagnostic output repeats that shared list across its frozen index-head axis. Equal-score top-k ties remain explicit `Unsupported`.
**Why:** The pinned source and contract lines 1202-1263 define four-token emission and `[B,8,2D]`/`[B,8,2ID]` carry behavior; they do not define a stride-2 compressor. Lines 1281-1303 reduce index heads before `torch.topk`, while lines 1589-1595 state that equal-score tie ordering is not portable.
## 2026-07-17 — Decision inbox merge

<!-- Source: mariette-qmoe-activation-coverage.md -->
### 2026-07-17: QMoE CPU↔CUDA equivalence coverage extended (activations/FC3/bias/router)
**By:** Mariette
**What:** Added equivalence tests for nonlinear activations (gelu/silu/swiglu), FC3 gated MLP, biases, and separate router_weights aggregation. The tests isolate biases and router aggregation with identity activation, exercise FC3 without biases or separate router weights, cover both CUDA GEMV and grouped-GEMM paths, and include a combined GLM-like gated-SiLU case for f32/f16/bf16.
**Why:** Nabil advisory — prior tests only covered identity/no-bias/no-FC3/shared-weights, leaving GLM MoE numerics under-locked. All requested axes are implemented by both kernels; no unsupported-feature gap was found.

<!-- Source: wallace-visible-scope-incremental.md -->
### 2026-07-17: shape inference builds visible_scope only for subgraph-bearing nodes
**By:** Wallace
**What:** visible_scope is now built/extended incrementally and only when a node has child subgraphs; subgraph-free nodes skip scope materialization.
**Why:** Audit hotspot #1 — O(N·V), 1.22s @20k nodes (600× ratio). Inferred shapes byte-identical, closure semantics preserved.

<!-- Source: roy-review-epctx-batching.md -->
### 2026-07-17: Review — EPContext partition-splice batching
**By:** Roy (reviewer)
**Verdict:** 🟢 APPROVE
**What:** Reviewed the full `9362f9b` diff. Boundary inputs/outputs still iterate covered nodes in ascending `NodeId` and use `HashSet`s only for membership/first-seen dedup; no observable order derives from hash iteration. Hoisting `graph_outputs` is equivalent because splicing never mutates `graph.outputs`. Batched removals/insertions preserve partition order, covered-node removal order, recycled `NodeId`s, edge order, producer links, and orphan-value collection. Independently compared the existing multi-partition export against parent `d5f5432`: both serialized files had SHA-256 `1551b4cff1b2b8be5d06e80ad3c1aa183b585d0ce1738e4d9cbae3901f01aeb6`. Ran `cargo test --locked -p onnx-runtime-loader -p onnx-runtime-ir` with the required target directory; all 147 tests passed.
**Why:** The ordering ABI remains ascending-`NodeId` with first-seen dedup, the final graph mutation sequence is observationally equivalent to the prior sequential splice for valid disjoint partitions, and serialized multi-partition output was proven byte-identical to the parent implementation.

<!-- Source: holden-review-qmoe-coverage.md -->
### 2026-07-17: Review — QMoE CPU↔CUDA equivalence coverage
**By:** Holden (reviewer)
**Verdict:** 🟢 APPROVE
**What:** Verified that `run_cpu` obtains the real `CpuExecutionProvider` QMoE kernel and supplies the expected values, while `run_gpu` obtains the CUDA kernel and compares its copied-back output against that CPU result. CPU and CUDA GELU both use the same f64/double tanh approximation with identical constants; SiLU and SwiGLU formulas also match. The tests distinctly exercise ordinary GELU/SiLU, FC3-gated SiLU, unfused SwiGLU with separate FC3, interleaved fused SwiGLU, split fused SwiGLU, FC1/FC2 biases, FC3 bias, and separate aggregation weights that differ materially from the routing logits. The f16/bf16 oracle rounds only the activation input before CPU f32 computation, matching CUDA's widened input and fp32 accumulation, then accounts for output storage rounding. Ran `cargo test --locked -p onnx-runtime-ep-cuda --test qmoe_gpu` with the required CUDA environment: 21 passed, 0 failed; also reran the GELU case with `--nocapture` on available NVIDIA H200 GPUs.
**Why:** The cases use non-trivial quantized weights, non-zero bias vectors, a distinct FC3 seed, and aggregation values unrelated to the logits; the forced routing gives every expert three routes, so thresholds 2 and 1024 genuinely select grouped GEMM and per-route GEMV paths. The tolerances are not laundering an implementation difference: f32 `1e-4` covers CPU-serial versus CUDA-tree reduction order, while f16 `6e-4` and bf16 `4.1e-3` closely track one output-storage unit roundoff (`2^-11` and `2^-8`) plus the same accumulation allowance. At unit scale the new bounds are only about 11%, 4%, and 3% above the prior derived bounds, respectively.

<!-- Source: deckard-cuda-cpu-drift.md -->
### 2026-07-17: CUDA↔CPU numeric drift root cause
**By:** Deckard
**What:** The reported token-index-10 flip was a genuine CUDA RMSNorm bug: NVRTC contracted the recurrent square accumulation to FMA, unlike the CPU oracle. The existing `__fmul_rn`/`__fadd_rn` fix restores that boundary; the next token-12 bug was fused-SiLU operation order and is also already fixed. RoPE/GQA and QMoE stay at `1.49e-8` and `1.86e-8` max absolute error. Residual accuracy-4 MatMulNBits warp/VNNI reduction-order noise is `1.9073486e-5` (6 ULP) per op and first flips the Qwen probe at token 16; no production patch was applied because exact CPU-tree emulation is architecture-specific and previously cost 8.4% decode throughput. Added long-context/per-op error-bound regressions and reporting.
**Why:** GLM long-context accuracy (north-star). Drift first at token index ~10 was accumulating error in RMS normalization across recurrent layers/tokens; after the clear bugs are fixed, the remaining accumulating source is MatMulNBits reduction-order noise, not KV-cache RoPE/GQA or MoE aggregation.

<!-- Source: chew-batch-dead-node-removal.md -->
### 2026-07-17: batch Graph::remove_nodes eliminates O(N²) dead-node deletion
**By:** Chew
**What:** Added `Graph::remove_nodes` (single consumer-retain pass + precomputed I/O sets for orphan GC); DeadNodeElimination now batches. Byte-identical to sequential remove_node.
**Why:** Audit hotspot #6 — per-node consumers.retain + linear inputs/outputs.contains was O(N²) (57ms @20k wide dead nodes).

<!-- Source: bryant-review-visible-scope.md -->
### 2026-07-17: Review — incremental visible_scope
**By:** Bryant (reviewer)
**Verdict:** 🟢 APPROVE
**What:** Closure fidelity is preserved: prior outer outputs and enclosing scopes reach doubly nested subgraphs; prior inner bindings shadow outer names without leaking to sibling or parent scopes; future node outputs are excluded. Loop carried/state inputs and Scan state/scan inputs retain their declared types and shapes. A transient mixed graph covering ordinary nodes, nested If, shadowing, Loop, Scan, and outer captures produced an identical deterministic dtype/shape snapshot on `d245f9e` and `08c8006`; a separate known-`value_info` future-output probe passed only on `08c8006`, confirming corrected ordering. The requested shape-inference, loader, and session test command passed.
**Why:** `child_scope` is created only when a node owns subgraphs. It starts from the enclosing imported scope, adds graph inputs/initializers once, then drains outputs accumulated after each previously inferred node. Current-node outputs are added only after its subgraphs run, and subgraph-local values are never inserted into the parent map. Subgraph-free graphs therefore perform no scope construction or output queueing, while graphs with later subgraphs still accumulate every preceding output. Validation: `CARGO_TARGET_DIR=/home/justinchu/target-vscope cargo test --locked -p onnx-runtime-shape-inference -p onnx-runtime-loader -p onnx-runtime-session` exited successfully.

<!-- Source: roy-registry-index.md -->
### 2026-07-17: OpRegistry indexed by (domain, op_type)
**By:** Roy
**What:** Added a sorted per-(domain,op_type) since_version index; lookup/supports/earliest_since_version now O(log v)/O(1) instead of O(R). Behavior byte-identical incl. ai.onnx/"" aliasing.
**Why:** Audit hotspot #7 — ~2.9M registry key comparisons at 10k-node static compile.

<!-- Source: holden-review-drift-fixes.md -->
### 2026-07-17: Review — CUDA RMSNorm/SiLU numeric-drift fixes
**By:** Holden (reviewer)
**Verdict:** 🔴 REJECT
**What:** The RMSNorm fix is correct: both RMS reductions retain serial f32 accumulation and use explicit `__fmul_rn` then `__fadd_rn`, so NVRTC cannot contract the square accumulation into FMA. The kernel still computes mean-of-squares, adds epsilon, takes reciprocal square root, and applies scale; exact FMA-sensitive GPU regressions pass. The CUDA RMSNorm implementation remains f32-only, so this change neither alters nor validates f16/bf16 behavior. The SiLU f32 fix follows the CPU's branch-stable `x / (1 + exp(-x))` or `(x * exp(x)) / (1 + exp(x))` order with explicit f32 rounding boundaries, and its exact CPU-order regression passes; CUDA SiLU remains f32-only.

`cargo test --locked -p onnx-runtime-ep-cuda` passed all 166 tests on H200. Targeted metrics were GQA/RoPE max absolute error `1.4901161e-8`, QMoE `1.8626451e-8`, and accuracy-level-4 MatMulNBits `1.9073486e-5` (6 ULP). The real-model `native_cuda_qwen_decode_matches_cpu_tokens` test could not reach generation: it fails pre-decode with the reported `no inferred shape for value present.0.key produced by op GroupQueryAttention`. This blocker is pre-existing relative to `07e5545`: that commit changes only three CUDA test files and does not touch engine, session, loader, or shape inference.

**Why:** The kernel fixes themselves are sound, but the token-16 residual-noise acceptance is not. The source test sets `max_new_tokens: 16` and compares only token indices 0–15, exactly stopping before the documented first divergence at index 16. Therefore it does not establish that CUDA and CPU tokens remain equal once the residual MatMulNBits drift is amplified. A reported CPU top-logit margin of `0.6502361` at the divergent step is not a near-tie and cannot be waved off as harmless output noise merely because the originating per-op difference is `1.9073486e-5`. The acceptance criterion is token correctness, and the known token flip remains. Sapper should own the revision: remove the real divergence (without Deckard), or provide a rigorously justified equivalent path, and extend the real decode parity test beyond the first known divergent token (preferably 32–64 tokens).

<!-- Source: wallace-review-batch-remove-nodes.md -->
### 2026-07-17: Review — batch Graph::remove_nodes
**By:** Wallace (reviewer)
**Verdict:** 🟢 APPROVE
**What:** The batch path is semantically equivalent to sequential `remove_node` for valid graphs. Shared values retain surviving consumers in original order; producer links are cleared only when owned by a removed node; orphan collection uses the same producer/consumer/I/O/initializer predicate and preserves graph outputs and initializers. The sequential count simulation handles values shared by multiple removed consumers, slice ordering, duplicate/non-live IDs, value-arena GC order, and unknown type/shape cleanup. `remove_node` itself remains unchanged. The 10,000-trial test uses an actual sequential-removal clone, compares the complete `Graph` debug state, and probes node/value arena free-list order.
**Why:** The implementation filters each touched consumer vector once with `retain`, preserving relative order and every surviving consumer, while precomputed membership sets replace only linear membership checks. Dedicated edge cases cover shared surviving consumers, all-removed consumers, protected outputs/initializers, and duplicate/non-live IDs. `cargo test --locked -p onnx-runtime-ir -p onnx-runtime-optimizer` passed all 99 tests (48 IR, 50 optimizer unit, 1 optimizer integration), plus doc tests.

<!-- Source: sapper-gqa-present-shape.md -->
### 2026-07-17: GQA present K/V outputs always carry a shape (graceful degrade)
**By:** Sapper
**What:** group_query_attention no longer early-returns when past-K/V shape is missing/non-rank4; it emits [batch, kv_heads, opaque_seq, head_dim] so present.N.key/value always have a shape.
**Why:** Session requires op-produced values to carry a shape; missing present shape blocked native engine decode for GQA models (Qwen/GLM). Resolved-past path unchanged (byte-identical).

<!-- Source: leon-fusion-resumable-scan.md -->
### 2026-07-17: fusion uses a resumable ascending-id worklist (was O(F*N) restart)
**By:** Leon
**What:** OpFusion no longer rescans the arena prefix after each rewrite; an ascending-id candidate worklist re-seeds only the affected neighborhood, preserving lowest-id-first fusion order. Output byte-identical to the restart fixpoint.
**Why:** Audit hotspot #5 — restart-from-zero find_match was O(F*N) (364ms @20k MatMul+Add).

<!-- Source: bryant-review-gqa-present.md -->
### 2026-07-17: Review — GQA present K/V graceful shape
**By:** Bryant (reviewer)
**Verdict:** 🟢 APPROVE
**What:** Guard scoping is correct: only the missing/non-rank-4 past-cache guards after output 0 were replaced. Present K/V are structurally `[query batch, kv_num_heads, opaque-or-resolved sequence, output-0 head_dim]`; output-count guards remain intact. The rank-4 constant path retains `max(capacity, total)`, and the missing, rank-4, and non-rank-4 cases are covered by assertions for both present outputs.
**Why:** All prerequisites for attention output 0 remain guarded, and batch, KV heads, and head dimension are derived before present emission. Missing or invalid-rank past shapes degrade only the sequence dimension via `fresh_dim()` rather than inventing a concrete extent. For valid rank-4 past plus scalar total, the resolved branch is unchanged and CASE B confirms the original fixed-capacity result. `cargo test --locked -p onnx-runtime-shape-inference` passed all unit, integration, and doc tests.

<!-- Source: wallace-fusion-review.md -->
### 2026-07-17: Reject fusion resumable-scan optimization
**By:** Wallace
**What:** 🔴 REJECT `a0023bc`. Leon is locked out from revising this artifact; Pris should own the next revision of the differential proof.
**Why:** The hard semantic-preservation proof is insufficient. `crates/onnx-runtime-optimizer/src/fusion.rs:2584-2606` chooses exactly one canned motif per trial, while lines 2608-2626 only append terminal unary noise. It never composes multiple registered fusion candidates, shuffled/non-topological NodeIds, or chained/overlapping default-pattern candidates that can exercise lower-id revisits after mutation; the only explicit overlap is the synthetic `ReluPair` case at lines 2630-2644. Therefore the 5,000 trials do not validate the completeness of `affected_candidate_starts` at lines 251-278 for the adversarial ordering case required by the review mandate. The independent locked test command passed: 48 IR tests, 52 optimizer tests, 1 integration test, and doc tests.

<!-- Source: nabil-glm-readiness-gaps.md -->
### 2026-07-17: GLM-5.2 runtime readiness gap analysis
**By:** Nabil
**What:** A correctness-first f32 GLM target can take the portable CPU path once the Mobius export/domain/cache-shape fixes land, but smooth CPU/CUDA execution still needs `BlockQuantizedMoE` and selected-token DSA. Native CUDA additionally lacks the standard Attention/Rotary/TopK/movement graph core; DeepSeek native CSA and MTP state orchestration remain incomplete.
**Why:** North-star = run GLM smoothly; this enumerates the remaining runtime gaps to dispatch against.

<!-- Source: mariette-drift-revision.md -->
### 2026-07-17: CUDA↔CPU drift revision — token-16 characterization + extended parity
**By:** Mariette (revising Deckard's rejected artifact; Deckard locked out)
**What:** Real bug, not a near-tie: before the revision token 16 had CPU top-2 1181=17.438587 and 330=16.788351 (gap 0.6502361), while CUDA flipped to 330=17.04121 over 1181=16.720116 (gap 0.3210945). The residual MatMulNBits K=4864,N=896 reduction-order delta is bounded at 2.670288e-5 and was not causal. MatMulNBits was left parallel; CPU SiLU/GQA softmax now round f64 exp to f32 like CUDA, and CUDA GQA RoPE prevents FMA contraction. Exact token parity now covers 64 tokens with a 2e-5 recurrent logit bound; token 16 is identical with the same 0.6502361 gap.
**Why:** Holden 🔴 — 16-token test stopped before the known divergence; residual unproven. Kept Deckard's correct RMSNorm/SiLU fixes.

<!-- Source: chew-fusion-revision.md -->
### 2026-07-17: Strengthen resumable fusion-scan differential proof
**By:** Chew
**What:** Replaced the canned differential fixture with 5,000 randomized composite graphs. Every trial exercises all five registered fusion matchers, shuffled insertion and NodeId/free-list assignment, overlapping MatMul candidates sharing Add/Relu nodes, and a test-only chained replacement that must be revisited below the ascending cursor. The restart-scan oracle and resumable implementation are compared through byte-identical canonical graph serialization plus arena-allocation probes.
**Why:** Wallace rejected the original evidence because it did not prove lowest-NodeId ordering or affected-node completeness under overlap, chaining, and varied arena layout. The strengthened test found no completeness or ordering bug, so the resumable implementation was unchanged. The required optimizer/IR gate passed (100 tests total plus integration/doc tests); a release probe on 20,000 MatMul+Add pairs measured 2.61x versus the restart oracle (2.730s → 1.045s), preserving the performance win. Wallace must re-review this revision.

<!-- Source: wallace-fusion-rereview.md -->
### 2026-07-17: Reject fusion resumable-scan revision
**By:** Wallace
**What:** 🔴 REJECT `1c68735`. Chew is now locked out of this revision cycle (Leon remains locked out); Pris should own the next revision of the differential proof.
**Why:** The restart oracle is faithful, all five registered matchers and shared-node structural overlaps are exercised, serialization compares complete live graph state plus future arena allocations, and the required locked suite passes (48 IR + 51 optimizer + 1 integration; doc tests green). However, the chained adversary at `fusion.rs:2823-2834,2903-2912` only exercises the unconditional `revisits.insert(fused_id.0)` path: reverse removal followed by LIFO arena insertion reuses the old `ChainStart` slot, while its external input is a graph input, so `affected_candidate_starts` contributes no predecessor at all. Thus the test still does not prove the completeness of `affected_candidate_starts`, the exact rejected concern. It also consumes every seeded free slot before optimization (`fusion.rs:2846-2867`), so optimizer-entry free-list occupancy/state is not varied. Pris must add an asserted case where a previously scanned lower-ID predecessor is absent from the fused region, is returned by `affected_candidate_starts`, becomes newly matchable only after the higher-ID fusion, and would be missed if affected revisits were disabled; retain randomized nonempty free-list states.

<!-- Source: holden-drift-rereview.md -->
### 2026-07-17: CUDA↔CPU token-16 drift re-review
**By:** Holden
**What:** 🟢 APPROVE commit `87fbc09` and the full `d245f9e..87fbc09` drift change set.
**Why:** The real Qwen native decode test now drives the same recurrent CPU/CUDA path for 64 tokens, checks every step at `2e-5` max-logit error, and explicitly verifies the former step-16 failure point. Two independent runs passed with identical step-16 top-2 logits, token IDs, `0` max error, and `0.6502361` gaps. CPU SiLU/GQA now use the CUDA kernel's existing double-precision `exp` followed by the shared f32 boundary, improving rather than reducing transcendental accuracy. CUDA RoPE uses standard explicit round-to-nearest multiply/add intrinsics, preventing FMA contraction with negligible launch-bound decode cost; its 64-token reference test is bit-exact. The `2.670288e-5` synthetic MatMulNBits difference is deterministic floating reduction-order variance between CPU VNNI lanes and the CUDA warp tree, not a dequantization or token-selection bug. The full locked ep-cuda suite passed (166 tests), and targeted CPU/GQA/session/shape regressions passed.

<!-- Source: deckard-fusion-proof.md -->
### 2026-07-17: Prove affected-start fusion revisits and match-start slot reuse
**By:** Deckard
**What:** Added a 5,000-seed differential test whose lower-NodeId `AdversaryStart` is initially ineligible, then becomes eligible only after a later higher-NodeId fusion replaces a three-node region. The later match consumes the lower start's output as an external input, so `affected_candidate_starts` schedules that already-passed start; instrumentation proves 5,000/5,000 affected behind-cursor revisits (100%), and disabling affected scheduling makes trial 0 fail. Every trial compares nodes, IDs, values/edges, topology, metadata, and observable node/value arena free-list state byte-for-byte with the original restart-scan oracle.
**Why:** No implementation bug was found. The exact proposal that a later fusion consume an older reclaimed low slot is structurally impossible: `apply_fusion_returning_id` removes the current match in reverse, placing its start slot last on the LIFO free-list, then immediately inserts the replacement into that start slot. The test proves this invariant on every fusion and separately probes that the earlier fusion leaves lower interior slots reclaimable in 5,000/5,000 trials. The meaningful adversarial path is nevertheless exercised independently of `fused_id`: the newly eligible lower start differs from the first fused ID and is reached with `ScanCandidateSource::Revisit`. Required optimizer+IR gate passed (48 IR unit, 52 optimizer unit, 1 optimizer integration; doc tests green). Final release audit: 20k no-match fusion 0.782 ms; 20k match-heavy fusion 199.010 ms versus the recorded 364.826 ms baseline (~1.83× faster), with a prior-revision comparison at 206.658 ms showing no observer/proof regression beyond run noise. Wallace must re-review this third-owner revision.

<!-- Source: coordinator-graph-ir-one-mutable-storage-type-borrow-enforced-.md -->
### 2026-07-17T18-05-01: Graph IR: one mutable storage type + borrow-enforced read-only + GraphView lens; no separate immutable Graph representation
**By:** coordinator
**What:** Graph IR: one mutable storage type + borrow-enforced read-only + GraphView lens; no separate immutable Graph representation
**References:** crates/onnx-runtime-ep-api/src/provider.rs:410, crates/onnx-runtime-ep-api/src/abi.rs:12, crates/onnx-runtime-session/src/executor.rs:272
**Why:** ### 2026-07-17: Mutable vs immutable graph representation

**By:** Squad (Coordinator), approved by Justin Chu (justinchuby)

**Decision:** Do NOT introduce a second, immutable Graph storage type.

1. Immutability at the EP boundary is already free via Rust borrows: claim_nodes(&self, graph: &Graph) (ep-api/provider.rs:410) hands EPs an immutable &Graph, compile-time enforced (ORT needs a distinct GraphViewer type only because C++ has no borrow checker + crosses a C ABI).

2. Mutable/immutable is a LIFECYCLE, not two coexisting storage types: load -> optimize -> partition/fuse (mutable Graph) -> freeze -> execute (immutable plan: Vec<NodePlan>, executor.rs:272). The mutable Graph never appears on the per-token hot path.

Chosen shape (three representations by lifecycle phase, ONE storage type):
- Mutable Graph: HashMap-keyed edges (NodeId, input_index) -> O(1) single-edge add/remove; deterministic sorted accessors for reproducibility (user hard requirement).
- GraphView (immutable lens): promote stub OrtGraphView (ep-api/abi.rs:12) into a read-only projection borrowing &Graph + cached topo-order + optional assigned-node filter (mirrors ORT GraphViewer). Ordering materialized ONCE, cached; natural home for arena-tombstone compaction.
- plan: Vec<NodePlan>: existing frozen execution artifact; unchanged.

Consequences: Chew's IR container refactor stays; GraphView lens is an additive follow-up; IR-refactor reviewers must verify NO per-run hot-path graph traversal regression.

<!-- Source: chew-ir-refactor.md -->
### 2026-07-17: Make single-node graph removal fanout-independent
**By:** Chew
**What:** Replaced `Value.consumers: Vec<NodeId>` with a hash-backed `(NodeId, input_index)` use set; added sorted deterministic `uses`/`consumers` boundaries, `Graph::replace_input`, per-value graph-I/O flags, and Vec-indexed topological-order scratch. Updated every workspace edge reader/mutator, added randomized equivalence/topology/serialization determinism proofs, and committed a release benchmark harness plus `docs/IR_CONTAINER_REFACTOR.md`.
**Why:** Removing hub consumers one at a time was quadratic because every removal retained over the full consumer vector and scanned graph I/O. On the same 96-vCPU Xeon 8480C, release median sequential removal improved from 17.137 ms to 1.589 ms at 10k consumers and 58.792 ms to 2.927 ms at 20k; one hub-edge disconnect improved from 3.467 us to 0.683 us and 6.366 us to 0.815 us. Hash iteration is never observable: uses sort by `(NodeId,input_index)`, consumers/traversals sort by `NodeId`, Debug sorts, serialization/topology tests require byte-identical results after shuffled insertion. Wallace (IR owner) must review before merge.

<!-- Source: coordinator-landed-cuda-cpu-token-16-drift-fix-ccf994c-kimi-k3.md -->
### 2026-07-17T18-30-21: Landed CUDA↔CPU token-16 drift fix (ccf994c); Kimi K3 MoE expert-parallelism doc read + kept in mind
**By:** coordinator
**What:** Landed CUDA↔CPU token-16 drift fix (ccf994c); Kimi K3 MoE expert-parallelism doc read + kept in mind
**References:** ccf994c, docs/MOE_EXPERT_PARALLELISM.md, wt-fusion, wt-ir-refactor
**Why:** ## Landed
- **ccf994c** — fix(ep-cuda): resolve/characterize token-16 MatMulNBits drift + extend decode parity to 32-64 tokens (author Mariette; reviewer Holden 🟢 APPROVE, 166 ep-cuda tests pass, drift 2.67e-5 deterministic reduction-order variance, 64-token parity @2e-5). Carried Deckard's per-op drift-bound test (d7cec74). Redundant GQA-present-shape commit auto-dropped (already upstream via c89e61f). Rebased linear, FF-pushed, worktree cleaned.

## Kimi K3 architecture — read & noted (docs/MOE_EXPERT_PARALLELISM.md, commits 222c9a8+4538a96)
Authoritative target specs now in-repo: Kimi K3 = 2.8T / 896 experts / 16 active (~50B active, ~1.4TB FP4). Novel components: KDA (Kimi Delta Attention), AttnRes (cross-layer attention residuals), Gated MLA, Stable LatentMoE + Quantile Balancing. Deployment design: session-per-GPU expert parallelism, control-plane(Rust)/data-plane(GPU) separation, MoeDispatch/MoeGather NCCL custom ops baked into each session graph, distributed PageTable (per-device shards), Mac Studio (MLX+TB5) cross-node path. REINFORCES our direction: frozen-plan hot path + GraphView lens (EPs get immutable graph views), EP-claim of custom NCCL op nodes, per-attention-type KV caching keyed by inference_metadata. No conflict with the in-flight IR refactor.

## In flight
- wallace-30: THIRD review of fusion resumable-scan (Deckard proved affected_candidate_starts completeness 5000/5000 in f997173).
- wallace-31: IR container refactor review (Chew b0869f2: 10k removal 17ms->1.6ms, 20k 59ms->2.9ms).
- nabil-20: Kimi op/kernel-coverage readiness note (companion to the deployment doc).

<!-- Source: wallace-fusion-rereview3.md -->
### 2026-07-17: Third review of resumable fusion scan
**By:** Wallace
**What:** 🟢 APPROVE commits a0023bc, 1c68735, and f997173.
**Why:** The previously unproven behind-cursor case is now adequately resolved.

- **Free-slot trace:** `apply_fusion_returning_id` removes `m.nodes` in reverse order. `Arena::remove` pushes each slot onto a LIFO free list, so the match start (`m.nodes[0]`) is pushed last. The immediately following `insert_node` therefore always gives the fused replacement the match-start NodeId. Every fusion performs this remove-then-single-insert sequence; older freed interior slots remain below the new match-start slot and cannot be consumed by a later fusion replacement. The 5,000-trial probe also confirms the replacement always takes `later_start`, while a subsequent unrelated probe takes the lower interior slot. Thus the claim that older free-slot reuse is structurally impossible for fusion replacements holds.
- **Affected scheduling:** The dedicated adversary creates an existing lower-id start whose eligibility changes after the later fusion. It was revisited through `affected_candidate_starts` in 5,000/5,000 trials. I locally disabled insertion of affected candidates into `revisits`: the targeted test failed on trial 0. After temporarily bypassing its observer-count assertion, the byte-identical restart-reference assertion also failed, confirming a genuinely wrong fixpoint rather than merely missing instrumentation. The broader randomized differential test still passed that mutation, so the dedicated adversary is the load-bearing coverage.
- **Determinism:** Initial candidates come from ascending arena keys, and all revisit resolution is normalized through `BTreeSet`. Although predecessor discovery uses `HashSet`, its iteration order cannot affect graph mutation order because candidates are inserted into the ordered set. The differential tests compare sorted graph state plus future arena allocation sequence. I additionally ran a temporary 100-run reconstruction test using independently created graphs/HashMaps; serialized results were byte-identical.
- **Tests:** `cargo test --locked -p onnx-runtime-optimizer` passed: 52 unit tests, 1 integration test, 0 failures. The worktree was restored clean after mutation probes.

<!-- Source: nabil-kimi-k-readiness.md -->
### 2026-07-17: Treat Kimi MLA/KDA as first-class attention-state gaps
**By:** Nabil
**What:** Kimi K2/K3 readiness must not equate MLA with existing GQA or DeepSeek CSA. Add versioned semantic MLA and KDA boundaries with explicit persistent state, then prioritize `BlockQuantizedMoE` with lazy leases and model-exact MXFP4/MXFP8 negotiation. AttnRes should initially remain graph-visible activation state, not KV state. K3 MTP must remain conditional until a released artifact verifies it.
**Why:** K2's public config proves low-rank latent KV plus a separate RoPE component, while GQA stores conventional full K/V heads and CSA implements ratio-4/128 temporal sparse compression. The current direction already provides the right seams—op/domain/opset dispatch, arbitrary IR nodes, lazy `WeightHandle`, paged checkpoint/restore, and a stateful private CSA precedent—but lacks generalized attention-state lifecycle, native MLA/KDA kernels, live `BlockQuantizedMoE` paging, and multi-device expert placement.

<!-- Source: wallace-ir-refactor-review.md -->
# IR Container Redesign Review — commit b0869f2 (branch `ir-refactor`)

**Reviewer:** Wallace (IR owner) · **Date:** 2026-07-17T18:25Z · **Requested by:** Justin Chu
**Author:** Chew · **Worktree:** /home/justinchu/wt-ir-refactor · **Merge base:** c89e61f

## Verdict: 🟢 APPROVE

The O(n²) single-node removal is genuinely fixed by making consumer edges a
`HashSet<(NodeId, input_index)>` with O(1) keyed removal, and the HARD
determinism requirement is preserved by sorting at every observable boundary.
All four contract items verified; tests green; benchmark re-run confirms scaling.

---

## a. Determinism / reproducibility — PRESERVED ✅

- The new `Consumers` type (value.rs) wraps a `HashSet<Usage>` and **only**
  exposes iteration through sorted snapshots: `uses()` sorts by
  `(NodeId, input_index)`, `nodes()` sorts by `NodeId` + dedups. `Debug` prints
  the sorted `uses()`. Equality is set-based (order-independent).
- Every place that feeds consumer ordering into observable output goes through
  these sorted accessors:
  - `graph.rs` `producers_of`/successor build sort with `sort_unstable_by_key(|n| n.0)`;
  - `topological_order` uses a min-heap keyed on raw `NodeId` (Reverse), so adj
    insertion order is irrelevant;
  - `replace_all_uses` iterates the sorted `uses()` and rewrites graph outputs
    via `set_outputs`;
  - all optimizer/loader/memory consumers now call `graph.consumers()` /
    `graph.uses()` (sorted), never the raw set.
- Audit: no code iterates the raw `HashSet` for output. Only internal
  `insert`/`remove`/`contains` and order-independent `PartialEq` clones touch it.
- **Proof by test:** `consumer_hash_insertion_order_is_not_observable` (IR) and
  `debug_serialization_and_topology_ignore_consumer_hash_insertion_order`
  (loader) shuffle consumer insertion order and assert identical Debug, topo
  order, `uses`, `consumers`, **and byte-identical `encode_model` output**.
  No HashMap-iteration-order leak into any deterministic result.

## b. Byte-identical equivalence vs old algorithm — PRESENT ✅ (with a caveat)

- `single_node_removal_matches_vector_reference_on_random_dags` — 2000 random
  DAGs, new `remove_node` vs an independent reference; asserts Debug, topo,
  per-value `uses` and `consumers` all match.
- `vec_indexed_topology_matches_hashmap_reference_on_random_dags` — 2000 trials,
  new Vec-indexed topo vs HashMap reference.
- `randomized_single_removal_matches_reference_serialization` (loader) — 250
  trials asserting **serialized model bytes** are identical to the reference.
- Caveat (non-blocking): the "reference" is a semantic oracle re-implemented via
  the new `replace_input` API, not the literal pre-refactor `Vec<NodeId>` code
  (which is deleted). It independently reproduces old semantics (per-edge scan +
  orphan GC), so it is a legitimate differential oracle. Acceptable.

## c. Benchmark proves O(n²)→O(n) for single-node removal — CONFIRMED ✅

Re-ran `examples/remove_node_bench` (release) myself on a high-fanout hub graph:

| nodes | sequential_remove (ms) | single_hub_disconnect (µs) |
|------:|-----------------------:|---------------------------:|
| 5000  | 0.790 | 0.561 |
| 10000 | 1.420 | 0.749 |
| 20000 | 3.272 | 0.855 |
| 40000 | 6.670 | 1.046 |

- Single-node disconnect from a value with fanout = node_count stays **flat**
  (~0.56→1.05 µs while node count grows 8×) — the fanout-independence the fix
  targets. Old `consumers.retain(...)` was O(fanout) per removal → O(n²) batch.
- Sequential (remove all) scales **linearly** (~2× time per 2× nodes).
- Directly validates the claim; not just the batch path.

## d. No per-run hot-path regression — CONFIRMED ✅

- `executor.rs` frozen `plan: Vec<NodePlan>` execution path is unchanged; the run
  loop reads `self.plan[..]` / `self.graph.node(plan.node_id)`, never the mutable
  container's consumer sets.
- The only `consumers` reads in session/memory/optimizer/loader are **build-time**:
  `fuse_silu_patterns`, `build_lazy_weight_handles`, memory `liveness`/`validate`,
  loader `partition_boundary`, and fusion pattern matching. None run per token.
- Container mutation cost is therefore build-time only, as designed.

---

## Residual-risk assessment

1. **HashSet memory overhead** — ACCEPTABLE / non-blocking. Per-value
   `HashSet<(NodeId,u32)>` costs more than a `Vec` for the common small-fanout
   case, but it is a build-time structure freed once the plan is frozen. Optional
   future optimization: inline/smallvec storage for fanout ≤ 1–2. Not required.
2. **Non-generational NodeId slot reuse** — ACCEPTABLE / non-blocking. Pre-existing
   arena property, not introduced here. Safe because `remove_node` →
   `disconnect_edges` removes *all* of a node's input edges (via `replace_input(..,
   None)`) before `nodes.remove`, and `connect_edges` re-inserts fresh on insert;
   `validate()` checks consumer/producer link consistency and the new
   `ConsumerLinkMismatch` path. The 2000-trial differential test exercises
   remove-then-reuse. Keep the debug_assert invariants guarding I/O-flag drift.

## Minor observation (non-blocking)

`fusion::find_consumer` now selects the first matching consumer in **NodeId-sorted**
order rather than insertion order. Still fully deterministic (arguably more robust);
just a subtle observable-selection change vs pre-refactor. No action needed.

## Test results

`cargo test --locked -p onnx-runtime-ir -p onnx-runtime-optimizer -p onnx-runtime-shape-inference` — **all green**:
- onnx-runtime-ir: 53 passed (incl. the 4 differential/determinism tests)
- onnx-runtime-optimizer: 50 + 1 passed
- onnx-runtime-shape-inference: 14 + 13 + 154 passed
- (bonus) onnx-runtime-loader `ir_container_determinism`: 2 passed (byte-level)

0 failed across all suites. Workspace-wide build blockage (missing cpuinfo vendor
CMakeLists) is pre-existing and unrelated — confirmed not touched by this change.

**No changes requested. Approved to merge.**

<!-- Source: ripley-cuda-movement.md -->
### 2026-07-17: CUDA movement, Where, and predicate broadcasting
**By:** Ripley
**What:** Added CUDA EP registrations and deterministic kernels for standard-domain `Concat`, `Expand`, `Reshape`, `Slice`, `Split`, `Squeeze`, `Tile` (opset 6), `Transpose`, `Unsqueeze`, and `Where`. `Shape`, `Gather`, and `Constant` already existed; none of the requested movement operators or `Where` were previously registered. Logical `And`/`Or`/`Xor` and f32 comparison `Equal`/`Greater`/`Less`/`GreaterOrEqual`/`LessOrEqual` now use the existing right-aligned broadcast metadata and zero-stride reads from `elementwise.rs`.
**Why:** GLM/DeepSeek portable exports require these standard construction/movement operations and broadcast predicates to remain on the native CUDA EP. Movement and `Where` copy fixed-width element bytes, so they cover f32/f16/bf16, integer, and bool storage without dtype-specific arithmetic; logical ops remain bool and comparisons retain their existing f32 operand coverage. Added 10 construction/Where GPU model tests, two broadcast-family GPU model tests covering all eight logical/comparison ops, and a registry coverage test. `cargo test --locked -p onnx-runtime-ep-cuda` passed all 182 tests.

<!-- Source: coordinator-kimi-readiness-doc-landed-8a73116-native-mla-is-th.md -->
### 2026-07-17T19-17-36: Kimi readiness doc landed (8a73116); native MLA is the headline Kimi/DeepSeek gap — queued for design, GLM CUDA ops remain the active focus
**By:** coordinator
**What:** Kimi readiness doc landed (8a73116); native MLA is the headline Kimi/DeepSeek gap — queued for design, GLM CUDA ops remain the active focus
**References:** 8a73116, docs/KIMI_K_READINESS.md, docs/MOE_EXPERT_PARALLELISM.md, docs/GLM_READINESS_GAPS.md
**Why:** ## Landed
- **8a73116** — docs/KIMI_K_READINESS.md (author Nabil, coordinator-committed + cross-linked to MOE_EXPERT_PARALLELISM.md). Op/kernel-coverage companion to the deployment doc.

## Key finding — native MLA is the headline Kimi/DeepSeek gap
Nabil (well-sourced: K2 config, K3 announcement, Kimi Linear proxy) confirms our GroupQueryAttention and CompressedSparseAttention do NOT implement MLA's latent-KV compression (kv_lora_rank=512, q_lora_rank=1536) or decoupled RoPE (128 non-RoPE + 64 RoPE dims). Verified K2 = 1T/384-expert/top-8 + 1 shared, MLA, FP8 (E4M3, 128x128 blocks), num_nextn_predict_layers=0 (no MTP). Announcement-level K3 = 2.8T/896/16, KDA + AttnRes + Gated MLA + Stable LatentMoE + MXFP4-weight/MXFP8-activation; weights/report due ~2026-07-27.

Top Kimi gaps (in priority): (1) native MLA (latent KV state + projections around attention, no full-KV expansion) — also needed by DeepSeek lineage; (2) KDA recurrent attention (finite-state + short conv, 3:1 KDA:MLA per Kimi Linear) w/ typed persistent state; (3) MXFP4/MXFP8 BlockQuantizedMoE with live offload; (4) 1M-context prefix-cache/state integration; (5) richer EP capability negotiation (attention-state kinds, quant layouts, max state dims).

## Scheduling decision
MLA is NOT on the immediate GLM-5.2 critical path (GLM uses DSA/dense + CSA, not MLA), so it is QUEUED as a design item — do NOT auto-implement a large new attention mechanism without a design pass + user check-in. Current focus stays on GLM: P0 CUDA graph-op completeness (Ripley in flight on movement/construction + Where + broadcast-logical).

<!-- Source: mariette-cuda-movement-review.md -->
# CUDA Movement/Construction + Where + Broadcast-Logical Review

**Reviewer:** Mariette (CUDA kernel reviewer)
**Author (not reviewer):** Ripley
**Worktree:** /home/justinchu/wt-cuda-move — branch `cuda-move` — commit `fba64de`
**Merge base:** `40f6490` (origin/main)
**Date:** 2026-07-17T19:20:00Z
**Requested by:** Justin Chu

## Verdict: 🟢 APPROVE

The diff correctly unblocks native CUDA loading of the GLM/DeepSeek portable exports
(P0 gap in `docs/GLM_READINESS_GAPS.md`). Every added op is semantically faithful to
the CPU reference, deterministic, and dtype-generic where it should be. The gate is
green on an H200.

---

## Test gate (re-run by me, not trusting Ripley's number)

`cargo test --locked -p onnx-runtime-ep-cuda` → **all suites pass, 0 failed**.
Relevant new coverage:
- `construction_gpu.rs`: **10/10** pass.
- `pointwise_gpu.rs` broadcast families pass (`logical_family_numpy_broadcast_matches_cpu_reference`,
  `comparison_family_numpy_broadcast_matches_cpu_reference`, `f16_bf16_numpy_broadcast_matches_cpu_reference`).
- `mod.rs` unit tests (`covered_ops_have_no_duplicates` = 78, `movement_and_where_ops_are_listed_in_coverage`) pass.

## Per-op parity findings (vs `crates/onnx-runtime-ep-cpu/src/kernels/`)

- **Slice** ✅ `slice_plan` is a line-for-line match of the CPU `slice_plan`: negative
  indices, step-sign-dependent clamp bounds `(0..d-1 / -1..d-1)` for negative step vs
  `(0..d / 0..d)` for positive, count formula, omitted-axes default `0..starts.len()`,
  omitted-steps default `1`, negative-axis resolution. **CUDA is strictly more robust**
  on the `dim==0 + negative step` edge (explicit guard avoids the `clamp(0, -1)` panic
  the CPU `slice_plan` would hit) — not a divergence in any reachable-shape case (both
  yield count 0). starts/ends/axes/steps handled as INPUTS at opset≥10. Test exercises
  negative axis + reverse step (`axes=[1,-1]`, `steps=[-1,-2]`) with a hand-verified result.
- **Split** ✅ Precedence identical to CPU: `split` input → `split` attr → even
  (`num_outputs` attr else `outputs.len()`). `even_split` is byte-identical to CPU
  (ceil chunks, smaller final remainder, over-split guard). Negative axis, sum/output
  validation present. Tested via negative axis + `split` input.
- **Transpose** ✅ Default reversed perm when attr absent; arbitrary perm validated as a
  true permutation (dup/range checks); strides from `compute_contiguous_strides`. 3-axis
  perm `[2,0,1]` tested.
- **Concat** ✅ Negative axis; per-input shape compatibility (only axis dim may differ);
  disjoint output writes via `axis_prefix`. Negative-axis multi-input test.
- **Expand / Tile** ✅ Expand uses right-aligned `broadcast_strides` and validates the
  target equals `broadcast_shapes`. Tile uses per-axis `coord % in_dims` with repeats
  validated against input rank (non-negative). Both tested.
- **Squeeze / Reshape / Unsqueeze** ✅ Correctly implemented as dtype-agnostic
  device-to-device copies that trust the pre-computed output shape (numel/dtype checked);
  axes read as INPUT at opset≥13 (Unsqueeze also validates output rank = in-rank + |axes|).
  This is the right design — geometry is owned by shape inference, kernel only moves bytes.
- **Where** ✅ True 3-way right-aligned broadcast (cond/x/y each get independent
  `broadcast_strides` into the output shape), matching CPU `effective_strides`. Bool
  condition (non-zero = true), branch/output dtype equality enforced, dtype-agnostic on
  x/y byte width. All-three-broadcast test present.
- **Broadcast logical/comparison** ✅ `BinaryPredKernel` now derives `out_shape` from
  `broadcast_shapes` and passes right-aligned zero-stride metadata (layout consistent with
  `elementwise::broadcast_metadata`: dims, then a-strides, then b-strides). Bool output
  canonical `1/0`; scalar (rank-0) handled via the `metadata.push(0)` guard. Differential
  broadcast tests vs CPU reference for And/Or/Xor and Equal/Greater/Less/GE/LE.

## Determinism ✅
Every kernel is a bijective one-thread-per-output-element grid-stride map (input→output
gather). No atomics, no accumulation, stable index mapping. Concat/Split launch one
kernel per chunk but each writes a **disjoint** output region, then a single
`synchronize()`. Fully reproducible — meets the hard repo requirement.

## dtype coverage ✅
Movement ops (Concat/Expand/Reshape/Slice/Split/Squeeze/Tile/Transpose/Unsqueeze) and
Where are element-size-generic via `fixed_width`/`byte_size` byte copies — verified with
both f32 and int64 tests, so they are NOT silently f32-only.

## Registration ✅
Registered under domain `""` (normalized to ai.onnx) at `since_version=1` (Tile at 6),
which `OpRegistry::lookup` resolves for any GLM opset (≥13). No duplicate/shadowed
registrations — none of these ops existed in the pre-diff CUDA registry. `CUDA_COVERED_OPS`
count updated 68→78 with a no-duplicates assertion.

## Non-blocking observations (do NOT hold up the merge)
1. **Comparison ops remain f32-only** on CUDA (CPU EP dispatches all numeric dtypes +
   Bool-Equal via `dispatch_arith`). This is **pre-existing** — this diff only added
   broadcasting, not dtypes. If a GLM/DeepSeek graph feeds `Equal`/`Greater` on int64
   (e.g. position-id or mask math), that node will hit the actionable f32-only error on
   CUDA. Recommend a follow-up issue to widen comparison dtype coverage; not required to
   unblock loading.
2. **Split even/`num_outputs`-remainder path** has no dedicated CUDA test (only the
   explicit-`split`-input path is tested). The `even_split` code is byte-identical to the
   CPU version (which is tested), so risk is low — a small follow-up test would close it.

Neither observation is a correctness defect in the submitted code.

## Summary
Rigorously reviewed against the CPU EP; parity, determinism, dtype-genericity,
registration, and differential test coverage all hold, and I independently re-ran the
gate green. **Approved.**

<!-- Source: newt-ep-coverage-diag.md -->
### 2026-07-17: Group CUDA coverage diagnostics by distinct failure class
**By:** Newt
**What:** Session CUDA-only coverage diagnostics now group unsupported nodes by `(domain, op_type, reason)`, report every class with its total node count, and show at most three sorted `scope/node#id` examples per class. Before: `graph #0 (ai.onnx::TopK): ...; ...; and 392 more unsupported node(s)`. After: `ai.onnx::TopK: ... [count=47; examples: graph/node#12, graph/node#98, graph/node#103]; ai.onnx::Trilu: ... [count=6; examples: ...]`.
**Why:** The previous eight-node cap hid distinct missing operators in large exported models. A `BTreeMap` sorts classes by domain, op type, and reason; example identities are also sorted before truncation, preventing graph/subgraph traversal or hash iteration order from leaking into the message. Tests cover repeated nodes, ten distinct failure classes (past the old cap), correct counts, three-example capping, and byte-identical output across two runs. Gate: `cargo test --locked -p onnx-runtime-session` passed.

<!-- Source: bishop-epdiag-review.md -->
### 2026-07-17: Review — grouped CUDA-only coverage diagnostics
**By:** Bishop
**What:** 🟢 APPROVE commit `8d6c349`. The diagnostic reports every distinct `(domain, op_type, reason)` failure class once, with an accurate total count and at most three sorted example node identities.
**Why:** Determinism is explicit: classes are stored in a `BTreeMap<(String, String, String), _>` and each class's node identities are sorted before formatting, so no `HashMap` iteration order leaks into the byte output. No class-level truncation remains. The diff is confined to `crates/onnx-runtime-session/src/executor.rs`; placement/partition logic and `ep-cuda` are untouched. `cargo test --locked -p onnx-runtime-session` passed with zero failures.

<!-- Source: hudson-graphview-design.md -->
# GraphView lens recommendation

**Date:** 2026-07-17  
**Author:** Hudson

## Decision

Adopt an immutable-after-build `GraphView` as the runtime consumption surface
over mutable `Graph`. Build it after all optimization and validation, alongside
the frozen session plan. Cache deterministic topology, dense live node/value
indices, flattened producer/consumer edges, and partition assignment slices.

Use current Kahn topology with ascending finalized `NodeId` tie-breaking.
Filtered EP views are borrowed topological `NodeIndex` slices with membership
metadata, never cloned graphs or repeated full-graph filters.

## Rollout

First replace `Executor::build`'s one-off topology and CUDA coverage traversal,
then add partitioned views and compatibility adapters for
`claim_nodes(&Graph)` / `supports_op`. Promote the ABI `OrtGraphView` only after
the Rust lens is proven.

Require differential tests comparing old/new assignments and subgraph
boundaries plus 10k-node/MoE allocation benchmarks.

## Open questions for Justin

1. Does byte-for-byte determinism cover only the same finalized IR artifact, or
   semantic graphs built through different mutation histories (which need a
   canonical key beyond NodeId)?
2. Prefer an explicit `FrozenGraph` owner versus session-owned graph/cache
   fields with method-scoped borrowing?
3. Should future assignment identity be `EpId` only or a richer
   EP-instance/device/session/expert-shard target?

<!-- Source: vasquez-cuda-indexing.md -->
### 2026-07-17: Add deterministic CUDA router indexing and scan kernels
**By:** Vasquez
**What:** Added native CUDA EP registrations and NVRTC kernels for `ai.onnx::TopK` opset 10, `CumSum` opset 11, `GatherElements` opset 11, and `ScatterElements` opsets 11 and 16. `TopK` supports f32 values with Int64 indices; `CumSum` and `ScatterElements` support f32/Int64; `GatherElements` copies any fixed-width dtype with Int64 indices.
**Why:** GLM/DeepSeek router and mask graphs require these operators to remain CUDA-eligible. TopK uses deterministic per-slice selection and CPU-compatible lower-index tie-breaking (including CPU ordering when `sorted=0`). ScatterElements runs duplicate updates serially in row-major update order, preserving last-write and sequential add/mul/min/max semantics without atomics. CumSum assigns one deterministic sequential scan per outer/inner lane and supports every exclusive/reverse combination. Tests cover TopK largest/smallest/sorted/ties/input-K/non-default axis, GatherElements negative axis/index, both ScatterElements opsets and all reductions with duplicates, and all four CumSum modes with a negative axis. `cargo test --locked -p onnx-runtime-ep-cuda` passed on H200 CUDA.

<!-- Source: mariette-cudaidx-review.md -->
### 2026-07-17: CUDA indexing/scan review — REJECT
**By:** Mariette
**What:** 🔴 Reject commit `10e2bd1`.

**Blocking defects:**
1. `TopK` writes its output with the selected-axis and trailing dimensions transposed when `axis` is not the final dimension and `K > 1`. In `crates/onnx-runtime-ep-cuda/src/kernels/topk.rs:44`, the destination offset is `(outer * inner + i) * k + out`; contiguous output shape replaces the input axis with `K`, so it must be `(outer * k + out) * inner + i`. For `[2, 3]`, `axis=0`, `K=2`, CUDA writes `[4,4,5,2,6,6]`, whereas a contiguous `[2,3]` TopK output is `[4,5,6,4,2,6]`. The new `topk_largest_ties_k_input_and_non_default_axis` test encodes the transposed order, so it does not catch this. Coco must revise this artifact; Vasquez must not revise it.
2. The requested scope is `crates/onnx-runtime-ep-cuda/` only, but the commit also changes `docs/CUDA_COVERAGE.md`. Remove that unrelated out-of-scope change in the revision.

**Verified:** `cargo test --locked -p onnx-runtime-ep-cuda 2>&1 | tail -25` passed (CUDA environment specified by the request; 21 library tests passed and all integration/doc-test groups completed green). The green gate does not cover the layout defect.

**Non-blocking follow-ups:**
- Add CPU-vs-CUDA parity tests rather than CUDA-only fixed expected outputs; repeat ScatterElements float duplicate-index add/mul runs to demonstrate byte-for-byte determinism.
- TopK remains Float32-only; add Int64/Int32 value paths if GLM graph partitioning needs integer TopK data. CumSum and ScatterElements provide Int64 but not Int32.
- Registration is otherwise correct: GatherElements v11; ScatterElements v11 and v16 (the registry selects v16 for opset 18+); CumSum v11; TopK v10. Scatter uses one ordered thread, so it avoids nondeterministic floating atomics and preserves duplicate update order.

<!-- Source: coco-topk-fix.md -->
### 2026-07-17: Correct CUDA TopK non-final-axis output layout
**By:** Coco
**What:** Changed CUDA TopK output and duplicate-suppression indexing to contiguous `[outer, k, inner]` layout. Added GPU parity coverage for axis 0 and a rank-3 middle axis, including duplicate tie-breaking and byte-for-byte repeated-run determinism.
**Why:** The previous `[outer, inner, k]` indexing transposed outputs whenever `inner > 1`; final-axis tests could not expose the defect.

<!-- Source: ferro-graphview-v2.md -->
# GraphView v2 revision

**Date:** 2026-07-17  
**Author:** Ferro  
**Status:** Decision request for Justin

## What changed from v1

The revised design replaces EP-wide flattened assignments with atomic
`PartitionId` / `CompiledPartitionView` objects. These preserve each
`SubgraphClaim`'s nodes, boundaries, connectivity contract, and `meta_def`.

It retracts the prior allocation-free claim for the legacy `supports_op` API:
its contiguous shape/layout slices require cloned per-node arrays. The preferred
path is a view-based `supports_node` migration.

It moves placement and schedule metadata from mutable `Graph` fields into an
immutable frozen plan, allowing the structural graph/cache to freeze before
partitioning. It also corrects cache ownership to
`FrozenGraph { graph, cache }` with borrowed `GraphView`.

The revision corrects consumer cache construction to prefix-fill by scanning
inputs in `(NodeId, input-slot)` order, identifies the topology heap cost, makes
release-mode final validation new work, scopes determinism to one finalized IR
artifact, and freezes nested bodies only after runtime shape preparation.

## Decisions for Justin

1. **Partition model:** Prefer atomic `PartitionId`/`CompiledPartitionView`,
   not `EpId -> Vec<NodeIndex>`; this preserves ORT claim semantics at the cost
   of more plan objects.
2. **Capability API:** Prefer iterator/view `supports_node` migration, not
   cached cloned shape/layout arrays; this avoids hot-path allocation but
   requires an EP API transition.
3. **Freeze boundary:** Prefer a structural freeze before partitioning and
   immutable plan-owned placement/schedule state; this adds plan metadata but
   prevents mutable-IR/view conflicts.
4. **Reproducibility:** Guarantee same-finalized-artifact determinism; size
   reproducible lookup bytes by maximum live ID. Cross-history semantic
   canonicalization remains a later feature.
5. **Target identity:** Prefer `PartitionTarget` richer than `EpId` from the
   outset, covering EP instance, device, session, and expert shard; this adds
   plumbing but avoids redesign for multi-device/MoE.

<!-- Source: frost-graphview-v3.md -->
### 2026-07-17: Make frozen partition storage owned and canonicalize claim sets
**By:** Frost
**What:** `FrozenPlan` stores owned `PartitionDescriptor` values and constructs borrowed `CompiledPartitionView<'_>` values through accessors. `SubgraphClaim.node_ids` are treated as an unordered set and canonicalized into base topological order.
**Why:** Storing borrowed partition views in their owning plan would be self-referential and is not safely implementable as sketched. The EP ABI documents no ordering contract for claim node IDs, so supplied order must not be a rejection condition.

<!-- Source: mariette-cuda-attn-review.md -->
