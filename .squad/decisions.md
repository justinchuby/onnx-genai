# Decisions

Canonical, append-only record of accepted team decisions. Only the Coordinator (via Scribe merge) writes here. Agents drop proposals in `decisions/inbox/`.

---

---

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
