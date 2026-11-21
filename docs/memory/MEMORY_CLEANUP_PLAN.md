# Memory contract cleanup without CUDA hardware

Status: source cleanup completed. GPU-dependent redesign remains deferred.
This plan covers source and host-verifiable cleanup only.

## Scope and order

1. Remove the governor's compatibility allocator module and mechanism re-exports.
   Import mechanism contracts from `onnx-runtime-memory-api` everywhere, including
   tests, examples, manifests and documentation. Keep budget contracts in the governor.
2. Rename the reservation-level `VirtualBacking` trait to `ReservationBacking`.
   Keep the allocator capability named `VirtualBacking`; update implementations,
   callers and documentation without an old-name alias.
3. Move weight-paging diagnostic and experimental configuration into a private
   configuration module. Preserve environment names, defaults, parsing, lazy
   initialization and call sites. Move the relevant host-only tests with it.
4. Correct current-architecture documentation: the built-in CUDA allocator is VMM;
   legacy policy flags do not restore the removed eager allocator. Document the
   mechanism/capability distinction and the verification boundary.
5. Run formatting, relevant host tests, CUDA source compilation and workspace
   compilation where supported. Review the complete diff for accidental behavior
   changes, commit the result, and open a PR with exact validation evidence.

## Invariants

No changes to allocation/adoption order, physical-handle pooling, map/unmap,
budget grants/refunds, eviction selection, stable weight addresses, KV growth,
CUDA Graph lifetime, or deferred-release ordering. Keep existing correctness,
context/lifetime, partial-failure and quarantine guards.

## Deferred until real CUDA validation is available

- Construct the final memory authority before any device allocation and reduce
  startup-ledger adoption states, preserving external EP integration contracts.
- Bind charging behavior to the selected allocation domain rather than callers.
- Consolidate resource ownership and release settlement across memory holders.
- Separate dense streaming and MoE execution policies; evaluate CPU cold experts.

These changes require real-device tests for KV preservation across growth,
CUDA Graph capture/replay with stable addresses, compute/copy stream ordering,
partial map/release failure, physical versus mapped accounting, budget pressure
and token equivalence. CPU tests and compilation do not establish these properties
or performance on a CUDA device.

## Validation and self-review

Completed source work:

- Removed governor mechanism exports and migrated imports/manifests across callers.
- Renamed the reservation trait and its implementations/tests to `ReservationBacking`.
- Extracted paging configuration into `weight_paging/config.rs`, keeping per-knob
  lazy initialization and the dynamically read refill cadence. Four existing
  parser/default tests moved with their code. Public eviction-probe API paths
  remain in `weight_paging`; `config` is a private implementation module.
- Corrected current allocator-policy descriptions, separated the two backing
  contracts, and repaired documentation links discovered by strict rustdoc.

Initial validation on macOS arm64, without CUDA hardware:

| Command | Result |
|---|---|
| `cargo test --offline -p onnx-runtime-memory-api -p onnx-runtime-memory-governor -p onnx-runtime-virtual-memory --no-fail-fast` | Passed: 188 unit/integration tests, plus four doctests (one import example and three removed-path checks); no ignored tests |
| `cargo clippy --offline -p onnx-runtime-memory-api -p onnx-runtime-memory-governor -p onnx-runtime-virtual-memory --all-targets -- -D warnings` | Passed |
| `cargo check --offline -p onnx-runtime-cuda-memory --lib` | Passed; driver is dynamically loaded, not exercised |
| `cargo check --offline -p onnx-runtime-cuda-memory --lib --features gpu-tests` | Passed; compiles the feature's library code, does not execute GPU tests |
| `RUSTDOCFLAGS='-D warnings' cargo doc --offline --no-deps -p onnx-runtime-memory-api -p onnx-runtime-memory-governor -p onnx-runtime-virtual-memory -p onnx-runtime-cuda-memory` | Passed |
| `cargo fmt --all -- --check` and `git diff --check` | Passed |
| Four extracted configuration tests, compiled from the production source as below | Passed; test listing confirmed all four names, execution reported four passed |
| `cargo check --offline --workspace --all-targets` | Blocked before completion by the existing ORT 1.29.0 dependency download; network-enabled retry returned HTTP 404 for the macOS arm64 archive |
| Full CUDA EP test enumeration and all-target/gpu-tests compilation | Also blocked by the transitive `onnx-genai-ort-sys` dependency; not claimed as passed |

The configuration module uses only `std`, so its actual source and existing tests
can be checked independently of the EP's ORT dependency. From the repository root:

```sh
printf '#[path = "%s/crates/onnx-runtime-ep-cuda/src/weight_paging/config.rs"]\nmod config;\n' "$PWD" > /tmp/memory-config-host-tests.rs
rustc --edition=2024 --test -D warnings -A dead_code /tmp/memory-config-host-tests.rs -o /tmp/memory-config-host-tests
/tmp/memory-config-host-tests --list
/tmp/memory-config-host-tests
```

`-A dead_code` applies to this standalone harness because production callers live
in the EP, outside the harness. No parser or GPU implementation is replaced with
a test stub.

Self-review completed before publication:

- Compared executable text for 99 migrated Rust files and the paging
  implementation, normalizing imports, the intended trait/path rename and
  optional trailing commas. No other executable changes were found.
- Compared the extracted configuration function bodies, defaults, local statics
  and all four moved test bodies with the base revision; they are preserved.
- Reviewed governor export removal, dependency declarations and documentation
  separately. Added compile-fail doctests for both removed governor paths and the
  old reservation-trait name; existing allocation/reservation tests remain.
- Allocation/adoption, map/unmap, pooling, grants/refunds, eviction, capture and
  release algorithms are unchanged. Full EP/workspace compile coverage was
  initially incomplete because ORT 1.29.0 was unavailable; see the follow-up
  verification below for the resolved checks.

No throughput or GPU-runtime correctness claim is made by this cleanup.


## Follow-up: PyPI runtime update and full compilation (2026-11-20)

The user authorized upgrading ORT to PyPI's latest stable release. Both
`onnxruntime` and `onnxruntime-gpu` report 1.32.0. Automatic native bootstrap now
uses SHA-256-pinned official PyPI wheels and the matching Microsoft C API 32
headers from NuGet, bundled with license and provenance. GitHub availability no
longer blocks default bootstrap. Python package floors and the producer CI pin
are updated to 1.32 as well. See the [bootstrap documentation](../../crates/onnx-genai-ort/ort-sys/README.md).

Full compilation exposed missing imports/visibility for the extracted
`PIN_PROBE_GRANULE_BYTES`, eviction-order parser and test refill environment name.
Those are corrected without changing their values or behavior. ORT 1.32's new
optional factory capacity callback is left null, as permitted by the upstream
contract: the generic EP cannot report device-global capacity.

| Follow-up command | Result |
|---|---|
| `cargo check --offline --workspace --all-targets` | Passed; existing unrelated unused-import/dead-code warnings remain |
| `cargo check --offline -p onnx-runtime-ep-cuda --all-targets --features gpu-tests` | Passed; compilation only, no CUDA device |
| `cargo test --offline -p onnx-genai-ort-sys` | Passed: seven bootstrap tests and one real-library C API roundtrip |
| `cargo test --offline -p onnx-genai-ort --test runtime_upgrade` | Passed: actual CPU Identity inference, without optional external fixtures |
| `cargo test --offline -p onnx-runtime-ep-plugin --lib` | Passed: 384 tests, none ignored |
| `cargo test --offline -p onnx-runtime-ep-cuda --lib weight_paging::config::tests::` | Passed: all four named tests, first confirmed with `--list` in the real EP crate |
| `cargo clippy --offline -p onnx-genai-ort-sys --all-targets -- -D warnings` | Passed |

Network access was enabled for commands whose build scripts needed to populate
new artifact caches; Cargo dependency resolution remained offline. Source checks
and CPU execution are now verified; CUDA device behavior remains deferred.

Follow-up self-review matched all five wheel URLs and digests against official
PyPI metadata, and checked all three bundled headers byte-for-byte against the
verified Microsoft NuGet package. Bootstrap rejects digest mismatches and unsafe
archive paths before extraction. The EP capacity callback remains explicitly
unsupported; no device-global capacity is inferred from a process budget.

## Pre-merge self-review

Reviewed the complete PR again before merging, including allocation-contract
imports, reservation trait users, configuration initialization, artifact hashes,
ZIP extraction, cache completeness, loader version selection and API 32's optional
factory callback.

- Fixed the runtime smoke test to select CPU explicitly. A default EP could be
  influenced by ambient GPU/MLX settings and would not prove CPU-only execution.
- Scoped non-macOS EP test imports/helpers to their test platform and removed a
  test-only forwarding import. This resolves the strict macOS lint findings
  without changing runtime selection or skipping any runnable test.
- `cargo clippy --offline -p onnx-genai-ort --all-targets -- -D warnings` passed.
- `cargo test --offline -p onnx-genai-ort --test runtime_upgrade --test execution_provider`
  passed: CPU inference and CPU-provider enumeration both executed successfully.
- Publish-order, documented-environment-variable, documented-command and
  feature-gate checks passed. Existing GPU-dependent validation remains deferred.

Merge is conditional on the required checks for the final reviewed commit;
branch protection is not bypassed.


### CI follow-up review

- Fixed a real bootstrap regression: PyPI wheels contain versioned Unix runtime
  files only. Publish both major and unversioned relative links, including repair
  of already-populated caches; regression coverage exercises Linux/macOS names,
  idempotence, incomplete-cache errors and old-cache repair.
- Full CPU ORT integration execution exposed three unary assignment tests missing
  the shared plugin lock. Add the lock at their test boundaries so rejection
  tests retain their zero-compile/zero-execution counter assertions.
- ORT 1.32 now executes the 512-wide quantization block previously rejected by
  older runtimes. Require numerical parity for this case and correct the docs.
- Remove Windows ARM64's stale Visual Studio 2022 generator override; retain the
  native build script's existing Ninja/installed-toolchain discovery.
- Validation: eight bootstrap tests plus one real C API test; strict sys Clippy;
  all 65 runnable CPU plugin ORT integration tests passed with
  `NXRT_REQUIRE_ORT_TESTS=1` (one pre-existing ignored test). Workflow integrity,
  required-tier coverage and whitespace checks passed. Windows verification
  remains delegated to the actual CI runner, not inferred from this Mac.

The repo-wide fixture guard caught opset 21 in the new Identity smoke fixture.
Updated it to the required opset 24, retaining IR 11. Validate the actual CPU
inference and the complete fixture guard together before the final CI rerun.


Expanded local ORT-backed execution exposed a pre-existing host assumption in
`append_chain_error_reports_the_mid_chain_provider_that_failed`: it passed a null
session-options handle while assuming CoreML was unavailable. The official macOS
wheel advertises CoreML, so the test entered the real C API and crashed. Replace
that setup with owned valid CPU session options and a deliberately unsupported
second provider. Preserve the chain index/name/source assertions without relying
on accelerator availability. Complete ORT/server tests now pass: 612 passed,
zero failed, two existing ignored tests across 17 suites; strict ORT all-target
Clippy also passes. No runtime algorithm changed in this follow-up.


## Follow-up: shared weight contracts and external design comparison

The CPU EP's `weight_offload::weight_handle` module was only a compatibility
re-export of `onnx-runtime-ep-api`, with a second re-export at the CPU crate root.
Both surfaces are removed. Shared weight negotiation, leases, binding and provider
capabilities are owned by `onnx-runtime-ep-api`; consumers import them there.
The existing executor and CUDA consumers already use that authority. No allocator,
budget, placement, kernel or release behavior changes in this removal.

Upstream comparison, inspected at fixed revisions:

- [Linger `fb6812d`](https://github.com/simplexai-labs/linger/tree/fb6812dcfdf079c76c3cf9a2a500e0697bab3c45)
  is the application layer; its README identifies PolyStrata as the inference
  engine. Application descriptions and reported performance are not runtime
  validation for nxrt.
- [PolyStrata cache `3e16421`](https://github.com/simplexai-labs/PolyStrata/blob/3e1642161198c50ff1abed4046f4e4f476099dbb/src/core/expert_cache.cpp)
  has an optional CUDA VMM path (`cuMemAddressReserve`, `cuMemCreate`,
  `cuMemMap`, `cuMemSetAccess`) and a `cudaMalloc` fallback. Its design therefore
  does not justify removing our VMM. Mapping policy and allocation mechanism
  remain separate decisions.
- [PolyStrata CPU expert interface](https://github.com/simplexai-labs/PolyStrata/blob/3e1642161198c50ff1abed4046f4e4f476099dbb/src/poly/cpu_experts.hpp)
  uses ggml CPU quantized dot products, groups tokens sharing an expert, and
  exposes overlap/readiness callbacks. These are distinct capabilities; a host
  implementation does not establish GPU overlap correctness.

| Capability | Current nxrt position | Next integration boundary |
|---|---|---|
| Reuse an expert across a token batch | CPU `BlockQuantizedMoE` already groups routed tokens and resolves each projection once per expert group | Preserve grouping and numerical tests; do not add a parallel batching authority |
| Compute cold experts from compressed weights | CPU native-block kernels currently decode routed projections to f32 and execute dense math | Evaluate a bounded compressed-weight compute path against the dense oracle on CPU; qualify formats from projection metadata and measure memory/time before dispatch changes |
| Use CPU and GPU experts within one layer | Static engine placement is advisory; native sessions still own one EP | Requires an executable partition and synchronization contract, plus real-device numerical and failure tests |
| Overlap expert loading and compute | CPU QMoE has a guarded route-first prefetch path; CUDA has separate stream/fence and residency authorities | Reuse their ownership and budget guards; do not copy callbacks into a second unaccounted scheduler |
| Choose a memory tier | Governor owns capacity; placement chooses residency within its grant | Keep byte budgets and explainable eligibility canonical; use measured cost-model rates rather than model or GPU identity guesses |

This comparison adds no upstream code dependency and makes no throughput claim.
The real-CUDA validation requirements above still apply to changes in mapping,
GPU/CPU execution splits, capture lifetime and asynchronous release ordering.
