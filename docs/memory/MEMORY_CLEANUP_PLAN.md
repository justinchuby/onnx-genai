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
| Compute cold experts from compressed weights | CPU `BlockQuantizedMoE` decodes rows when a complete expert f32 projection set exceeds the existing dense-cache budget; fitting sets retain dense GEMM/cache | All 15 formats reuse canonical decoders; this bounds decoded-weight scratch per call. Quantized-domain dots and GPU overlap remain separate work |
| Use CPU and GPU experts within one layer | Static engine placement is advisory; native sessions still own one EP | Requires an executable partition and synchronization contract, plus real-device numerical and failure tests |
| Overlap expert loading and compute | CPU QMoE has a guarded route-first prefetch path; CUDA has separate stream/fence and residency authorities | Reuse their ownership and budget guards; do not copy callbacks into a second unaccounted scheduler |
| Choose a memory tier | Governor owns capacity; placement chooses residency within its grant | Keep byte budgets and explainable eligibility canonical; use measured cost-model rates rather than model or GPU identity guesses |

This comparison adds no upstream code dependency and makes no throughput claim.
The real-CUDA validation requirements above still apply to changes in mapping,
GPU/CPU execution splits, capture lifetime and asynchronous release ordering.


### CPU bounded expert working set

`BlockQuantizedMoE` now admits dense execution only when the complete selected
expert projection set (including an optional third projection) fits
`ONNX_GENAI_CPU_DENSE_WEIGHT_CACHE_BYTES`. Otherwise it decodes one weight row per
explicit Rayon shard and applies that row to every token routed to the expert.
Routing, activation, gates and accumulation share the existing implementation;
format metadata and the canonical decoder remain the only format authority.
Zero budget selects streaming without hashing constant identities or building
dense cache entries. Contiguous packed weights are borrowed; strided inputs copy
only the selected expert's compressed bytes. Mutable inputs are read each call.

For a projection with input width K and N output features, at most
`min(Rayon workers, N) * (K + 256)` f32 values are live as decoded-weight scratch.
This excludes grouped activations, output-sized intermediate storage, compressed
strided copies and other concurrent calls. The cache budget is an eligibility
rule, not a process-wide working-memory grant; evicted Arcs can still be held by
other calls. No second memory governor or environment knob was added.

[CPU measurement and reproduction](../benchmarks/CPU_BLOCK_MOE_ROW_STREAMING.md)
records the measured memory/time effect and unchanged warm-cache control. This
is streamed f32 compute, not ggml integer-domain dot products or LLM throughput.

The measurements also exposed an independent startup cost: the existing host
allocation-cache floor calibration touches approximately 512 MiB when the floor
is not configured. Every measurement arm explicitly sets the same 1 MiB floor
to isolate expert memory. Startup calibration remains unchanged and should be
addressed in a separate allocator-lifecycle change.


### Follow-up: defer host-cache calibration until a retention decision needs it

`LargeAllocCache` construction captures explicit configuration and otherwise
leaves its automatic floor pending. The existing process-wide calibration stays
the sole measured answer. Disabled retention, invalid layouts, requests above
the maximum and requests below every possible automatic floor skip calibration.
The lower bound comes from the same probe ladder and fallback used to resolve
the floor; explicit configured floors, including smaller values, remain honored.

`floor_bytes()` now returns `Option<usize>` without performing a probe.
`resolve_floor_bytes()` explicitly permits calibration. There is no old-signature
shim. Pending/resolved state is encoded with `OnceLock`; automatic resolution
still runs once per process, and initialized instances take the resolved path.
Exact layouts, retained-byte admission, counters, poisoning, drain and concurrent
ownership guards stay in place. Zero-byte layouts are explicitly ineligible even
with a configured floor of zero, matching the documented allocation contract.

[Startup process evidence](../benchmarks/HOST_CACHE_LAZY_FLOOR.md) compares the
unchanged cache implementation with the lazy path and an explicit-floor control.
This avoids needless construction/small-request work; it does not remove the
probe from eligible large workloads or turn a retention cap into a peak-memory
guarantee. The separate global grant and active-allocation work remains open.


### Bounded standalone block projections

`BlockQuantizedMatMul` now applies the same dense-set eligibility rule as the
expert path: use full f32 expansion/GEMM only when the complete matrix fits its
existing cache budget. Otherwise reuse `linear_packed_grouped`, including all
format decoders and the inner serial/MLAS scheduling contract. No new allocator,
cache, format table or scheduler is introduced. Empty batches avoid dense
expansion while keeping argument validation. Fitting constant matrices retain
their existing cached fast path. The per-kernel cap and decoded scratch bound
remain distinct from a model-wide governor grant.

The block projection and MoE tests share one packed-format fixture. IQ1_M now
uses a finite nonzero scale so it exercises arithmetic rather than zero weights.
Half-storage output comparisons decode the actual dtype rather than interpreting
pairs of half values as f32 words. Existing cache/format guards remain intact.


### Owned QMoE host-cache payloads

Cache retention is an immutable policy ceiling, not a standing governor lease.
Each retained or temporary complete expert owns an exact grant from the same
Host source as block expansions. The grant survives cache eviction/provider
teardown while an active `Arc` still owns the expert, and returns only after
payload destruction. An empty cache consumes no Host capacity and can coexist
with host KV that was allocated before any expert is expanded.

`CpuMemoryResources::new(source, limits)` seals the cache and block/QMoE grant
source together. Its private fields prevent foreign-source cache pairing.
`CpuWeightCacheLimits` holds dense/expert retention ceilings, and the consuming
`with_cache_retention` builder captures the legacy planned-family verdict.
Standalone defaults are ungoverned; engine components clone their engine-owned
resource handle. The unused late-adoption API and the separate cache holder have
been removed without aliases. The optional environment ceiling is resolved once
at the cache's first operation; invalid configuration remains a typed error.

Cold publication has a single fill gate. Policy selection and publication hold
the short payload mutex; governor callbacks, dequantization and retired payload
drops do not. Ready hits skip the cold gate and issuer. This removes the old
adoption mutex, budget `RwLock` and operation-lifecycle `RwLock` while preserving
single-flight fills and the bank registry's independent prefetch proof.

Tests cover idle-zero ownership, KV competition and grant refusal, exact partial
refunds, active owners after cache drop, concurrent same-key fills, hard-failure
rollback and observers on reserve/refund. The existing frequency/pinning,
foreign-bank, routing, numerical and prefetch-statistics guards remain.


### Per-tier authority identity

`MemoryGovernor::authority_id(tier)` resolves the same books as a lease or
mapped allowance for that tier. The engine composes shared Device books with
worker-group-local Host/Disk books; a single Device identity incorrectly made
separate Host partitions look interchangeable when engines shared a device.
CPU weight-source anchors validate Host identity, virtual buffers compare their requested
tier, and process registration uses the declared device key's tier. CUDA pools
validate their context against Device identity, while sharing/admission/reclaim
use the physical location's tier: Device for VRAM and Host for Host-NUMA. All governor implementations and callers
use the explicit tier; no old-signature shim is retained. Regressions reproduce
the original identity/lease mismatch and independent Host books behind a shared
Device authority, then check clones and mapped allowances. This fixes
identity routing, not the separate physical-capacity relationship between CPU
hot-tier and Host books.


### Shared-cache prefetch capacity proof

The layer-only bound `experts * expanded_bytes <= budget` was insufficient:
foreign-layer payloads can occupy the rest of the shared partition. A reproduced
case admits the next expert in serial execution by evicting the current expert,
while prefetch's live `Arc` prevents that victim and changes the retained set.

A routed operation now registers its complete expert bank using the canonical
source/layout key and expert cardinality. Active operations and retained entries
hold that bank's metadata; the partition keeps weak registrations and prunes dead
ones on cold registration. Positive-budget prefetch requires the sum of every
live bank's complete expanded set to fit. A shared registry read guard freezes
that upper bound through compute and worker completion: new banks wait, while
operations on known banks remain concurrent. Workers borrow the existing proof
and never reacquire either reader lock behind a pending writer. Zero-budget
streaming needs no registry lock, and retains its single-ahead pipeline.

This is deliberately a conservative eviction proof, not a new memory budget or
a scheduler. Existing source identity, admission frequencies, pinning, exact
output and counter tests remain. Tests cover the foreign-bank counterexample,
exact shared-capacity boundary, retained metadata, expiry/pruning, concurrent
known-bank operations, cold-registration exclusion and worker progress. Full expert payload grants now share the Host source; row scratch and other
allocation classes still need model-wide grant convergence.


### Owned grants for block-quantized CPU derived weights

Native CPU registry construction now consumes one `CpuMemoryResources` bundle:
QMoE host-cache ownership, block-derived Host grants, the per-kernel dense
retention ceiling and the immutable CPU cache-retention verdict. This replaces the
combinatorial provider constructors; all engine and pipeline callers use the
same resource bundle without compatibility aliases. The engine constructs the
Host/Weights source before compiling kernels and shares it across components.
Native decode CUDA also receives the composite governor so Host-NUMA grants use
its Host books rather than the Device-only authority.

BlockQuantizedMatMul and BlockQuantizedMoE distinguish retention policy from
physical payload ownership. A zero-byte anchor supplies sibling grants without
reserving an unused standing allowance. Missing projections are admitted as one
group before any dense allocation, then `MemoryLease::split_off` transfers exact
shares into each buffer without changing the ledger. Partial eviction returns
only the dropped buffers' shares; active owners remain charged after eviction.
Nonconstant dense expansions use the same grants. Capacity refusal uses bounded
row decoding; allocation/configuration errors preserve their typed causes.

A separate cold-fill gate preserves single-flight reuse while grants, decoding
and releases happen outside the cache payload lock. MoE decodes directly into
its final NK layout, avoiding an uncharged complete transpose temporary. Warm
cache hits do not invoke the accountant. Tests exercise shared Host pressure,
complete-group refusal, partial replacement at exact quota, active owners after
LRU eviction, failure rollback, concurrent fills and numerical equivalence
across formats, dtypes, mutable/constant weights and packed storage layouts.

This is a scoped convergence step. Other derived caches, scratch/activation buffers, the physical
relationship between CPU Device and Host books, process-wide capacity and GPU
asynchronous lifetimes remain open. Default standalone CPU resources remain
ungoverned. CPU tests and CUDA compilation do not prove CUDA execution behavior.


### QMoE expert working sets and prefetch

Complete FC1/FC2/(optional FC3) sets acquire their Host grant before allocation.
Cache fills and transients share the sealed resource source. The same owned grant
remains with a retained payload, so there is no standing cache entitlement or
double charge. Capacity refusal chooses NBits row decoding through the shared
per-shard projection driver; allocation/configuration errors retain typed causes.

Streaming-only prefetch obtains a scoped grant for the driver's two simultaneous
temporary experts. Positive-cache prefetch pins initially selected hits, obtains
one admission grant for the complete missing selected set and splits exact
payload shares. A retained share survives on its own; consumed transient shares
and unused shares return independently. The scoped admission bound may exceed
the currently allocated dense payloads and is not an RSS measurement. Failure to
preflight keeps the existing serial path, rather than changing representation
only because an extra overlapping payload is live. Fully cached operations keep
the prefetch path without querying the issuer.

Real mmap fixtures verify exact serial/prefetch outputs and cache statistics,
Host-book peaks at quota boundaries, complete-set refusal, all activation/gates,
partial selected-set refunds and warm issuer bypass. Other derived caches,
scratch/activation accounting, CPU Device/Host physical-capacity relationships,
process-global capacity and asynchronous GPU validation remain open. Host tests
and CUDA compilation do not establish CUDA runtime behavior.


### Provider-scoped CPU cache retention

`CpuMemoryResources::with_cache_retention` carries an immutable `CpuCacheRetention`
verdict captured before registry construction. MatMul, Gemm, FusedMatMulBias,
MatMulNBits, QLinearMatMul and Einsum use it for the cache families already
included in the memory plan. The six process-global admission setters have
been removed. Loading a second model cannot alter the first model's decisions.
Standalone providers default to admitted retention. ISA and compact native
packing eligibility remain governed by their existing format/shape contracts.

Session builders accept an initialized CPU `fallback_execution_provider`.
CUDA whole-session fallback, heterogeneous partitions, plugin bridges,
control-flow child plans and decode/verify siblings carry that configured
provider instead of recreating default CPU resources. Non-CPU fallback
providers fail with an actionable configuration error. Native engine and
workflow callers supply their configured provider; standalone sessions can
still auto-detect a default CPU provider.

This is admission policy, not a new physical quota or grant. Existing shared
transpose/MLAS stores and process scratch ceilings remain shared. Block-derived
and transient QMoE payloads retain their separate owned Host grants. Remaining
work includes actual grants for other legacy packed caches, and the CPU Device/Host physical-capacity relationship. No GPU
execution or throughput claim follows from these changes.

Load-time constant preparation is a provider hook over immutable host operands,
independent of unresolved activation shapes. Its returned state belongs to the
executor. CPU transpose warming uses the same prepack policy and memo mechanism
as concrete kernels, so both static and symbolic graphs retain their load-time
warm path and release its ownership with the session.


### NBits f32 expansions join owned dense weights

`MatMulNBitsFactory::new` captures `CpuMemoryResources`, rather than only a
retention boolean. Every complete f32 expansion, including nonconstant and KN
transient operands, uses the same generic owned-dense cache component as block
formats. Cache keys remain representation-specific; the NBits key is local to
one immutable kernel slot and is never a shared raw address. A retained payload
keeps its exact Host grant through its last Arc, including after kernel drop.
Idle cache ceilings reserve nothing. Cold fills are single-flight; warm hits
claim no new capacity, and disabled/oversize retention does not serialize
transient operations through the cache fill gate.

The common per-kernel f32 retention ceiling is `CpuWeightCacheLimits::dense_bytes`
(default 256 MiB, read from `ONNX_GENAI_CPU_DENSE_WEIGHT_CACHE_BYTES`). The old
block-only environment knob is removed without an alias; benchmark commands and
current documentation use the common name. Retention and execution grants remain
distinct: NBits may use an admitted temporary full expansion even when retention
is off. Existing packed native and fused GEBP routes keep their legality and
priority, with the NT retention decision derived from the actual ceiling.

A resolved Local/NUMA/SPMD placement owns the allocation-peak rule and the actual
placement operation. NUMA/SPMD first-touch copies need two simultaneously live
f32 matrices, so construction preflights 2B and leaves only B charged to the
final payload. The shared cache rejects underspecified peaks, nonexact final
buffers and failed builders without publishing a partial entry or leaking a
grant. This is a weight-buffer admission bound, not an RSS measurement.

Full-grant capacity refusal selects the shared row-decoded projection driver.
NBits retains its serial M=1 dot order; batched projection uses serial per-column
GEMM inside bounded worker shards. The driver accepts an explicit scheduler;
NBits M=1 reuses its deferred flat/NUMA/SPMD dispatch rather than bypassing the
configured budget with ambient Rayon. The isolated SPMD parity fixture now
includes a refused dense grant and asserts that it actually reaches the active
scheduler. Unit output axes avoid a redundant layout copy. One affine decoder
serves NK, KN and refused
grant paths for 2/4/8 bits and group indices. Contiguous packed/scales/zero-point
operands borrow independently, including offsets; only strided or widened
operands materialize. The byte borrowing helper is shared with QMoE.

Host tests cover actual KV competition, independent source/retention policy,
active payloads after kernel drop, construction-peak refunds, refusal before a
full builder, rollback on invalid builders/group indices, all bit widths and
zero-point modes, strided operands, lower precision, bias and multiple rows.
This does not finish compact native/MLAS packing grants, external pressure coordination,
row scratch/activation accounting, CPU Device/Host capacity reconciliation,
process-global capacity or asynchronous GPU validation. Standalone default
resources remain local and ungoverned. No throughput or GPU execution claim.

The NBits planning exports are now `nbits_decode_f32_retention_candidate`,
`planned_nbits_node_retained_bytes` and `planned_nbits_retained_bytes`; engine
callers and fixtures use those names without old aliases. Their comments state
the estimate's limits: compact native packs, transients, scratch and runtime
shape-instance counts are not included. They never authorize physical bytes.
The optional MLAS Apple route test also checks the resolved platform policy,
including the native int8 fallback, rather than assuming linkage guarantees
QNBit/KAI execution. The unchanged main branch reproduces its former failure
in an isolated build directory; the original numerical guard is retained.


### Idle derived-weight reclamation through the physical authority

A governed `DerivedWeightMemory` keeps its zero-byte Host anchor and weak
registrations for the dense and shared QMoE expert caches. Its responder enrolls
with the supplying governor's optional authority/tier `PressureRegistry`; the
registration keeps neither the callback nor its authority alive. Source clones
share ownership of this enrollment. `supports_pressure_reclaim()` distinguishes
a funded source with this capability from a third-party authority that only
funds payloads. Lack of this optional capability preserves normal grants and
bounded refusal, without inventing a second CPU admission policy.

Ordinary `reserve`, existing lease `grow`, and replacement `reserve_sibling`
requests now enter the same physical-accounting admission. On a real refusal,
the ledger drops its claim gate, snapshots weak responders in rotating order,
and makes one bounded pass for the reported shortfall. A holder releases idle
whole objects under its own policy, outside registration/data locks. Admission
retries the actual books even when a callback reports zero; a claimed release
never authorizes allocation by itself. Oversized requests above the reported
total ceiling and hard arithmetic errors cannot flush the caches pointlessly.
Same-authority/tier recursive pressure declines rather than recursing; a later
round remains enabled in release as well as debug builds.

The CPU responder retains LRU/frequency policy and protects active `Arc` readers,
partial-hit projection groups and selected-expert prefetch pins. Ordinary QMoE
frequency pins guide retention, but cannot make an inactive payload immune to
physical Host pressure. Different CPU sources explicitly enrolled with the same
physical Host authority can fund one another and KV growth. Distinct authorities
or tiers cannot reclaim one another's idle weights. Cache membership/eviction
metrics remain separate from the owned grants that return only on the final
payload owner. Warm hits enter neither admission nor pressure.

Process allocation transactions use that same registry when the governor exposes
it; they do not pre-drive or repeat a pressure-capable authority's bounded
physical admission. Ungoverned/third-party allocator contexts can still use their
existing process-level callbacks. Process-quota shortage is a separate constraint.
`PressureResponder` now returns a typed result. CPU callback failures retain their
`EpError` cause through `MemoryError::ReclaimFailed`; failures cannot be relabelled
as ordinary capacity. A failed process transaction refunds its new quota before
returning that cause. Existing KV ownership is unchanged on failed growth.

This is synchronous, bounded admission pressure, not the scheduler's ticketed
host arbitration protocol or a complete process-wide RAM quota. CPU Device/Host
physical-capacity reconciliation, cross-engine host-book unification, compact
native/MLAS owned-pack admission, scratch/activation accounting and process-wide
coverage remain work. CUDA callbacks/runtime lifetime have not been exercised on
this host. No throughput, RSS or NUMA-performance claim is made.


### Owned native NBits preparation

All eight native NBits cache slots now use the same `DerivedWeightMemory`
source as dense expansions and QMoE experts. A `NativePackedWeight` payload type
fixes its physical layout and construction recipe; admission cannot pair an
expanded/tiled payload with a cheaper wire-layout charge. The six layouts cover
wire bytes, direct int4, expanded int8, direct u8, N16 SDOT and KAI qsi4/qsi8.
Runtime geometry determines padding, values and metadata sizes with checked
arithmetic. Invalid dtype, inaccessible Host operands, unsupported layout
geometry and size overflow fail before payload preparation.

Before allocating, the authority grants a bound for the entire transformation:
input gathering, portable half-scale widening, simultaneous input/output buffers,
tile-padded scales/sums/zero-point offsets and first-touch placement overlap.
Once construction destroys its temporary inputs, the prepared value keeps only
its exact payload grant. Data fields drop before returning that grant. Dynamic
preparations release at call end; retained constants release at kernel drop.
Capacity refusal cannot allocate an uncharged copy. All native representations
share one cold-fill admission mechanism, with warm lock-free cache reads and a
single gate preventing duplicate retained preparations. Pressure responders do
not take that cold-fill gate.

Unchanged contiguous Host Uint8/Float32 wire operands still borrow directly,
including with zero Host headroom. Numeric accumulation, ISA selection,
quantization and placement algorithms are unchanged. Tests compare the actual
six payload allocations with their charged bytes over tail/tile geometries,
scales dtypes and symmetric/affine inputs, including failure/refund and warm
reuse under a full Host budget. Real u8 decode and wire 2-bit execution exercise
the governed factory path; raw transformed-layout builders also execute on hosts
without their corresponding instruction-set route. That does not prove x86 ISA
execution or multi-node NUMA placement on this Apple host.

The optional MLAS pack admission is described below. Scratch/activation
allocations still need their own integration. Retained native OnceLock entries
are not yet idle-reclaimable.
Physical Host/Device book reconciliation and process-wide limits remain work.
No speedup, placement performance or CUDA runtime claim is made.


### Owned MLAS SQNBit payloads share Host admission

The optional reference route now obtains actual `DerivedWeightMemory` grants
before allocating standard, N-sharded or imported-prepacked SQNBit weights.
A resolved input plan carries borrowed contiguous Host bytes/f32 scales or
explicit gather/widen work. The grant covers the real packed allocation and
all retained scale/zero-point metadata, plus any overlapping owned input and
portable half-widening buffers during construction. Contiguous wire inputs do
not require a second whole-weight copy. Scale/zero-point storage in `mlas-sys`
is boxed to make the owned payload size exact, including on CompInt8, which
currently retains these copies too. Each packed object keeps its grant until
the final Arc owner drops, after data and metadata have been freed.

Shared-store keys include the physical Host authority, wire versus serialized
source encoding, and scale dtype in addition to the existing operand identities
and quantization geometry. Independently constructed sources/holders on the
same books can pay once. A different authority or explicitly ungoverned source
cannot reuse that paid allocation. The source snapshots its governor identity
even when custom lease accounting has no optional stamp; `host_authority_id()`
exposes this provenance. Weak per-key cold-fill gates prevent duplicate packs
under a one-payload budget, without serializing unrelated keys or locking value
maps during builders. Error/capacity refusals are retryable. Removal, clearing
and losing-candidate destruction return grants outside value-map/registration
locks; clearing membership does not revoke an active Arc's ownership.

A standard-layout MLAS packing capacity refusal lets the existing native
bounded fallback run and leaves the static capability memo unset, so the same
kernel can try packing after capacity returns. Other typed failures still
propagate. A serialized `weight_prepacked=1` operand cannot use the wire decoder:
its capacity failure remains an error, and an incorrect byte length is rejected
before a grant or data allocation. No new compatibility interface, quantization
algorithm, ISA predicate or dispatch threshold was introduced.

This covers the packed payload and retained scale/zero-point copies. Mutable
MLAS GEMM workspace, including the existing eager M=1 workspace, and transient
tile/activation scratch are separate scopes and remain ungoverned. Native/MLAS
OnceLock retention still needs idle reclamation; process-wide quota, physical
Host/Device book reconciliation and CUDA runtime proof remain unfinished. No
RSS, throughput or multi-node placement claim is made.


### Generic f32 GEMM borrows column panels

The thin-M column-parallel fallback no longer creates per-call B element copies
and secondary output strips. B keeps the original matrix stride; workers own
disjoint mutable row slices of the caller output. Dense row blocks and column
strips share the original register tile and K-panel accumulation through a
monomorphized row-access interface. No unsafe aliasing, persistent cache or
additional quota policy is needed for those borrowed elements.

Row-reference descriptors still allocate `O(M * min(N, threads))` metadata;
this is not zero allocation or full scratch governance. A public-kernel
allocator probe marks B as a constant initializer, registers the caller and
every explicit pool worker, observes
an intentional weight-sized positive control, and proves no allocations above
64 KiB on the first and repeated calls for its test geometry. Restoring the
old copy path makes the named guard fail on the first call. Exact panel-order,
partial-tile, output-bound and special-value checks preserve the previous
arithmetic, including empty-K +0 initialization.

See the [scoped diagnostic and allocation proof](../benchmarks/2026-11-21-generic-gemm-borrowed-columns.md).
This removes one family of ungoverned matrix copies; mutable MLAS workspace,
other transient scratch, idle OnceLock reclamation, process quota and physical
Host/Device reconciliation remain unfinished.


### Native NBits node pack ownership

MatMulNBits native layout slots and their single cold-fill gate now form one
shared node owner. The existing executor constant-state hook passes that owner
between activation-shape variants; no process-wide address store, second cache
registry or compatibility interface is added. Both publication and adoption
consume one eligibility rule, and adoption verifies preparation attributes,
constant-input flags and exact `DerivedWeightMemory` source identity. Only a
clone of the same governed source is interchangeable, even behind identical
Host books and holder numbers. Ungoverned standalone peers can still share.

Every layout retains its existing pressure enrollment and owned payload grant.
Dropping the producer kernel does not remove enrollment while a peer lives;
pressure cannot evict active readers, idle slots can be reclaimed, and the next
reader rebuilds under fresh admission. Optional dynamic metadata still forces
call-owned preparation through the existing execution gate. Dense f32 caches
and optional MLAS memo/store owners are unchanged.

[Host evidence](../benchmarks/NBITS_NODE_PACK_SHARING.md) reproduces duplicate
ledger charging in a real executor before the change, then verifies three
shape variants, output bytes, session isolation, all eight slot lifetimes and
concurrent cold builders. This is a node ownership change, not a throughput or
GPU execution result. Other scratch/activation accounting, physical capacity
relationships and CUDA asynchronous validation remain open.


### NBits dense expansions share node ownership

The NBits constant-state payload now carries its f32 NK cache as well as the
native compact layout slots. Its existing cache object owns the cold-fill gate,
pressure enrollment, payloads and grants; sharing an `Arc` to that object
removes shape-local duplication without creating a second cache authority.
The immutable dense-byte retention limit is part of the same resolved sharing
parameters. A different limit, including disabled retention, cannot adopt a
foreign policy even when the grant source and preparation attributes match.

[Host validation](../benchmarks/NBITS_DENSE_NODE_SHARING.md) uses a real
unknown-rank ONNX input to exercise vector, matrix and batched-vector shapes on
every backend without assuming transposed-B GEMM support. Complete outputs and
Host ledger charges distinguish one retained expansion from duplicate copies.
Factory-level tests cover a warm state exported after filling, producer teardown,
active readers, idle reclaim/rebuild and concurrent distinct cold builders under
a single-payload quota. Mutable group indices and zero retention keep execution
transient, with fresh numerical outputs and no retained charge. Dense arithmetic,
placement, decoding, compact formats and optional MLAS memo/store policy remain
unchanged. No memory prediction substitutes for live payload grants.


### Apple dense prefill reuses NK ownership

The native Apple backend now implements transposed-B GEMM, so dense NBits
prefill can consume the same granted NK expansion that decode and node shape
variants already share. Its capability predicate is the existing NT rule,
not a model-specific gate. The cache ceiling and physical grant continue to
control retention and admission; the existing fallbacks remain on refusal.
Actual executor tests cover symbolic prefill/decode row counts and separate
sessions, and a kernel test checks that warm prefill builds only once.

[Validation](../benchmarks/APPLE_NT_GEMM.md) explicitly records NN/NT floating-point
reduction differences and f64-oracle error bounds. No token-equivalence, GPU
behavior or throughput claim follows. The shared BLAS boundary validates buffer
extents and ABI dimensions before unsafe entry; pure native GEMV keeps usize
shape extents. Memory placement and grant/reclaim algorithms are unchanged.


### Reuse-gated Apple dense prefill

The resolved NT capability declares its dense-prefill reuse policy. Apple uses
NT only when immutable weights and the existing dense-byte ceiling admit NK
retention. With disabled/insufficient retention or mutable metadata, it preserves
KN preparation; physical grant refusal still follows the existing fallbacks.
This does not add a budget, model identity gate or geometry tuning table.

[Actual execute evidence](../benchmarks/APPLE_NT_EXECUTE_AB.md) keeps the initial
transient-NT observations separate from a newly recorded reuse-gated corpus.
Build-count, Host-charge and successful-NT-call witnesses accompany every cell.
The source's thread-local diagnostic selector is absent from shipped builds;
its scope/unwind behavior is tested. Process CPU includes framework/worker spin,
and no model throughput, RSS or other-device behavior follows from the numbers.


### Node-scoped optional reference ownership

MLAS SQNBit reference layout maps join the existing per-node constant-state
owner. Shape variants share only after the same source and preparation/retention
policy checks; independent nodes and sessions retain separate maps, even if they
use identical operand addresses and Host books. The global and test-local stores,
installer eviction responsibility and executor-wide clear API are removed.
Dropping an executor releases its kernel and static-state owners before source
buffers, without touching another executor's caches. Active payloads continue to
own Host grants through their final reader. No primitive arithmetic or production
backend choice changes. See the
[reference ownership contract](../performance/CPU_MLAS_MIGRATION.md#node-scoped-ownership-for-optional-sqnbit-reference-packs).

Reference layout maps now participate in idle Host pressure reclamation and
per-kernel strong memos are removed, as described in the follow-up below.
Serialized prepacked sources retain their existing exclusion from node-state
adoption.


### Idle SQNBit layouts join Host pressure reclamation

The optional reference's node maps now enroll weakly with the same derived-weight
source as native caches. Per-call `Arc` readers replace permanent per-kernel
memo ownership. Lookup and idle exclusion use the same map mutex; removed
payloads and their grants drop after all map locks are released. Reclamation
never takes a construction gate, including when a pack's own admission visits
its cache. Real KV reservations/growth and other node admissions can now reclaim
an idle reference pack without clearing any live session globally.

The bounded factory regression includes sharded, full-width, serialized and
noncontiguous-scale preparation, active readers, concurrent execute/pressure,
retry and final-owner refund. Logical constant slots preserve strided preparation
reuse inside the explicit node owner. MLAS scratch/workspace is not covered by
this payload accounting, and serialized bytes still require capacity for a
reference pack rather than taking a wire fallback. No CUDA execution or new
performance claim is involved.


### Ordinary MatMul derived copies honor retention decline

The shared MatMul preparation layer previously retained MLAS f32 panels,
once-widened f16 panels and native contiguous-half copies even with retention
disabled. Both pack entries now require an admitted constant B. Strided f16
copying returns an explicit grant-carrying reader, so a declined caller can
compute through the existing rescue path and release its transient copy without
populating the kernel cache. Existing admitted reuse and borrowed
column-major execution remain available.

The three old-behavior regressions demonstrate the bypass independently. New
bounded real-kernel controls check both verdicts and unchanged expected results,
while avoiding shared test-counter interference. This is retention enforcement,
not physical Host funding by itself. Ordinary f32 constant materialization now
joins Host admission below, followed by native half-copy payloads. SGEMM panels join Host admission below; transposes remain work; ordinary node ownership is consolidated below.


### Ordinary constant materialization joins the Host authority

All four MatMul-prepack consumers (MatMul, Gemm, FusedMatMulBias and FusedGemm)
now receive the provider's immutable resources rather than a bool/default-only
prepack. Both constant operand slots obtain actual Host payload grants before
conversion, retaining them in an executing reader or idle cache. The common
dense-cache authority handles singleflight construction and idle pressure.
Contiguous f32 borrowing remains zero-copy; dynamic activations are a separate
scratch scope. A declined retention policy changes lifetime, not the need to
fund a required constant materialization. Admission refusal stays typed and
retryable rather than allocating an uncharged fallback matrix.

The dtype converter writes directly into an admitted f32 destination and keeps
the existing SIMD conversion helpers. Strided/scalar conversion does not first
allocate a full source-dtype copy. A sealed reader prevents sibling kernels from
forging an uncharged materialized constant through the preparation API.
Address-keyed legacy transpose memos retain their converted source reader until
their key retires, preventing source-address recycling after pressure. This pins
those dependencies; funding/reclaiming transpose payloads themselves remains
separate, as do activation/output scratch and ordinary node ownership is consolidated below. Native contiguous-half copies and SGEMM panels are funded below.

The same provider-level test fails against the parent implementation with Host
usage 0 versus 24 expected bytes. New factory, active-reader, capacity/refusal,
retry, concurrent-fill and transpose-source controls exercise actual ownership.
The planner also includes Gemm and FusedGemm dense preparation, previously
omitted despite using the same prepack, and preserves domain gates.


### Native contiguous-half copies use the same grant lifecycle

Noncontiguous constant f16 B materialization now uses the same dense-weight
cache as f32 expansion, parameterized by its flat payload element (`u16` bits
or `f32`). Both retained and transient copies reserve their actual 2-byte-per-
element payload before allocation. Copying writes directly into that admitted
slice, preserving the existing 2-D parallel and N-D strided routes; no f32
weight matrix is needed. Source validation precedes admission and pointer reads.
Empty payloads return before parallel chunking.

The returned reader owns the grant through computation. Kernel destruction or
cache eviction cannot revoke a live reader; idle retained copies can donate
capacity to KV through the existing weak pressure registration. Retained cold
fills share one payload, while transient calls each require their own capacity.
Refusal propagates a typed, contextual error through both Apple rescue callers
and leaves preparation retryable. Column-major zero-copy execution is unchanged.

The unchanged public-provider regression fails on parent 3e2588329 with Host
usage 0 versus 306 expected bytes. Controls cover both retention verdicts,
negative strides/offsets, concurrent construction, active-reader protection,
KV eviction/refusal/retry, final refund, invalid host sources and empty layouts.
This step funds the native contiguous copy only. Ordinary SGEMM panel funding
follows below; transpose payloads, activation/output scratch remain work; shape-variant ownership is consolidated
below.


### Check SGEMM panel allocation geometry before funding it

The optional ordinary MLAS SGEMM packer now exposes an immutable `PackedBPlan`
whose private fields bind logical dimensions, source extent, native padding and
allocation layout. Planning allocates no payload. The native shim checks column
rounding, products and buffer alignment against MLAS's own constants before its
otherwise-unchecked size query. A disagreement with the standalone generic
layout is an explicit refusal rather than a potentially undersized allocation.
Rust additionally requires addressable source and destination layouts.

`PackedB::new` and `PackedBPlan::pack` return typed errors for geometry, source
length, layout and allocation failure; the old infallible interface is removed.
The plan and resulting pack report actual owned allocation bytes, including
padding and a one-byte empty sentinel. Empty operands do not call the packer.
Ordinary MatMul propagates construction errors and leaves its memo retryable;
a separate cold-fill gate prevents concurrent half conversions/packs while
warm hits avoid that gate. Warm f32 memo hits validate source extent and the
shape-specific packed dimensions before reuse.

This is a prerequisite for panel admission, not panel funding itself. The
next step below adds actual Host grants and idle pressure integration to
ordinary SGEMM panels and the half-to-f32 packing temporary. Existing f32 and
contiguous-half weight funding remains in place. No new GEMM primitive, backend default or performance
claim is introduced by checking the reference packer's allocation geometry.


### Ordinary SGEMM panels share the derived-weight authority

The shared cache now accepts owned weight payloads rather than assuming every
payload is a flat f32 vector. `WeightCache`, `GrantedWeight` and `WeightPayload`
replace the old dense-only names; vector construction remains a specialized
method and every caller uses the new types directly. Flat f32/u16 vectors and
aligned MLAS SGEMM panels share admission-before-build, exact logical/allocation
byte checks, serialized cold fills, active-reader ownership and weak idle
pressure registration. The dense-expansion environment limit remains a separate
policy and is not imposed on ordinary MatMul panels.

Both ordinary f32 panels and once-widened half panels retain their actual native
allocation grant in an executing Arc reader. Removing cache membership or
dropping the kernel cannot revoke that reader. Independent idle panels can
donate capacity to KV, and a subsequent call can rebuild. Keys bind the source,
dtype and K/N geometry, so a different source/shape never receives stale packed
values. Equal-source warm hits skip native layout planning and the fill gate.

Half preparation preflights the packed allocation plus the entire simultaneously
live f32 conversion before either buffer is created. Conversion writes directly
into one fallibly allocated destination. The scratch portion is refunded after
packing; only native panel bytes stay charged. Packing is optional: an admission
refusal invokes no conversion/packing builder, records no failed cache entry and
uses the existing unpacked execution route. Invalid layouts, bad source extents
and actual construction failures remain contextual typed errors.

This closes actual funding for ordinary SGEMM panel payloads and their half
conversion peak. It does not fund unrelated activation/output/workspace scratch,
legacy transpose payloads, and adds no shape sharing by itself; the next step consolidates node ownership.
Forecast changes are not a substitute for these actual grants; future resident
planning can use the checked native layout query.


### One preparation owner per ordinary MatMul node

`MatMulPrepack` keeps each kernel's retention/constant policy and grant source,
while one Arc-owned `MatMulWeights` holds its derived preparations. MatMul, Gemm,
FusedMatMulBias and FusedGemm expose/adopt that owner through the existing
node-scoped Kernel constant-state interface. The executor already exchanges this
state when compiling shape variants, so its generic cache needs no op-specific
logic. Shape-specific geometry, attributes and FLOP metadata stay on the kernel.

Adoption verifies immutable operand flags and the exact DerivedWeightMemory
source. A different grant source is an error, including a distinct holder on the
same Host ledger. Retention-off and runtime-only variants export/adopt no state;
unrecognized state is declined. Each new node/session starts with its own owner,
even when it uses the same provider and original initializer.

Dense f32 copies, contiguous half copies, native SGEMM panels and existing
transpose memos now survive an individual shape kernel's eviction under this
owner. Installed global transpose keys retire at the last node-owner drop; an
executing funded reader still keeps its payload/grant alive independently. The
legacy transpose payloads themselves still need actual grants and reclamation.

Dense-materialization prediction counts each constant non-f32 operand once per
node. Remove the fixed prefill/decode shape multiplier. Required bytes follow
the retained ownership unit and stay constant as activation shapes vary. The
estimate still excludes native panels, transpose payloads and other scratch,
and same-half paths may retain less; actual required copies remain governed.

Public factory controls cover all four consumers and both constant operand
slots under one-payload capacity with pressure reclamation unavailable. The
real executor control varies activation widths, shares within a node, keeps
distinct nodes/sessions separate, and verifies final refunds. It also disables
pressure reclamation so denied optional reference packs cannot distort the
resident-copy measurement on different CPU backends.


### Hand static preparation ownership to runtime kernels

Load-time provider preparation now lives in the existing kernel cache, indexed
by node together with its immutable-input flags. A newly compiled kernel first
tries a sibling owner, then compatible static preparation, and only performs
ordinary preparation if neither is recognized. Unknown or incompatible static
owners remain alive until executor teardown; kernels and these owners are
released before source buffers. This applies to fully static materialization
and symbolic-shape specialization without CPU-specific session dispatch.

The provider API receives immutable-input flags separately from available host
constant descriptors. CPU transpose prewarming returns the same shareable
MatMul owner consumed by runtime kernels, including when another immutable
operand has no load-time host descriptor. No compatibility signature remains.

Direct initializer immutability has one predicate: ONNX constant-initializer
semantics plus no producer. Writable producer-backed default bytes never enter
constant preparation or retained operand conversion. Allocation still follows
storage presence; view dispatch keeps its existing immutable-source path. Static
and runtime flag disagreement declines static adoption and prepares normally.
This does not redesign view provenance or fund legacy transpose payloads.


### Materialized operand identity includes logical layout

Immutable backing bytes can feed runtime views with different logical layouts.
The former dense-f32 and contiguous-half keys used only address/type/element
count, so equal-size views at the same origin could reuse a stale copy. Both
now use one typed copy key containing effective source address (including byte
offset), source dtype, full shape and signed element strides. Output element
width is a type parameter; payload bytes still use checked shape arithmetic.
No compatibility key or alternate classification remains.

The existing node owner and WeightCache admission/pressure/reader rules remain
in force. Distinct views own distinct funded copies; repeated identical views
reuse the same allocation, including after owner adoption by a sibling kernel.
Direct initializer predictions still count one immutable layout per operand;
custom runtime views are not represented by WeightRef and rely on actual grants.
This corrects materialized-copy identity, not transpose funding or native panel
source-lifetime policy.


### Keep derived-weight accounting out of steady cache hits

A warm reader consumes the payload's existing grant. It must not request a
sibling lease, return capacity or enter the accountant/pressure machinery merely
because a kernel executes again. Ordinary MatMul dense and contiguous-half
copies now use the existing single-entry cache lookup before the cold group
builder, avoiding group result/status containers on the warm path. Their full
logical-layout keys store common ranks inline; arbitrary ranks spill correctly
rather than introducing a rank limit. LRU updates and live-reader protection
remain in place, and the cold transactional builder is unchanged.

A real public-provider MatMul control verifies a cold 24-byte dense grant
(and the checked optional research SGEMM panel when that backend is active),
4,096 warm executions with zero further claims, including zero-byte requests,
or releases, and exact payload refunds at teardown. A retention-off control observes a claim and
refund for every transient copy, so a disconnected observer cannot pass. Warm
reuse does not remove admission for real allocations or hide transient costs.

The `weight_admission_ab` release benchmark interleaves real kernel executions
with local/governed resources, a duplicate local A/A arm, and an unaffected
contiguous-f32 control. It separates retained and transient policies, excludes
warm-up, retains per-round samples, and reports both wall and process CPU time.
The host-lock wrapper must cover the whole comparison. Zero warm callbacks are
a structural invariant; timing alone cannot establish universal zero overhead.


### Packed weights retain source identity without pinning source payloads

An owned conversion buffer can leave its dense cache while a packed panel is
still useful. Its allocation address is not a durable identity: the next
conversion can reuse the same storage for different immutable source values.
Keep the typed DenseOperand through the shared MatMul compute helper instead
of erasing its ownership to a slice before optional SGEMM preparation.

A materialized source now identifies its Arc owner through a weak key. The key
reserves the owner's control block, so that identity address cannot be recycled
while the cache entry lives. It keeps neither the vector's heap elements nor
their Host grant alive. The identity type accepts flat-vector owners, whose
payload allocation is separate from the control block. Live panel readers keep
the independent packed payload and its own grant. Original borrowed sources
use the complete logical-view key; this also fixes half-packing collisions
between equal-size strided views of the same immutable backing storage.

A call-owned, ungoverned conversion lacks a proven immutable identity and uses
the existing unpacked computation. The old raw-slice packed-B interface is
removed rather than retained as a compatibility path. No global identity
counter, registry or cache scan is needed. The native production build adds no
identity bookkeeping: weak identities are compiled with the optional reference
packing consumer. The native build retains its original slice compute interface;
the optional packing build carries typed ownership to its consumer.

Controls explicitly move one conversion allocation into a new owner, establish
identical data addresses without allocator assumptions, and verify different
packed numerical results. They also verify that source grants retire while
old panel readers remain valid, identical owners reuse a panel, full half-view
identity separates layouts, and call-owned conversion falls through correctly.
Legacy process-global transpose payload funding/lifetime policy remains work.

### Node-owned, funded half transposes

Native-width f16/bf16 transpose payloads now use the same node-owned
`WeightCache` as other prepared MatMul weights. Their actual `2*K*N` Host grant
is acquired before writing into the admitted output allocation; the transpose
algorithm allocates no second matrix. Identity includes the original contiguous
source, interpretation and complete matrix geometry. Shape variants and fused
operators share the node owner. Idle entries participate in Host pressure
reclamation; executing Arc readers independently keep their payload and grant.
A declined optional transpose runs no builder and can be retried later.

Apple f16 decode can read the original row-major half storage directly after
optional preparation is declined, without requesting a larger widened weight.
MatMul and FusedMatMulBias use the same fallback arithmetic. x86 keeps its
existing untransposed half GEMV. Cold preparation and pressure transitions may
call accounting; 4,096 warm provider executions make no claim/refund calls,
including zero-byte calls. This callback invariant does not assert zero cache
lookup overhead or universal inference latency.

The global half store and its diagnostic/clear APIs are removed. Multiturn
benchmark diagnostics report model Host ledger usage, explicitly including all
model holders. The remaining process-global **f32** transpose store is still
legacy and needs actual funding/reclamation; its installed keys continue to be
retired by the final node owner. Forecasts remain distinct from admission.

On Apple M1 Max, native CPU 1,992 tests / 20 ignored and optional MLAS CPU
2,032 / 35 ignored pass; session passes 277 / 279 tests respectively, with
strict all-target Clippy in both configurations. The public provider regression
fixture, injected with only a test observer into parent `95de9fef1`, compiles,
selects one test, and fails at the first funding assertion: actual Host usage
is 0 instead of 70 bytes. The current fixture also validates inference after
reclamation and exact final refunds. No CUDA or other-ISA execution is claimed.

### Fund the remaining f32 B transpose payloads

The f32 B transpose now uses node-owned `WeightCache` storage too. Identity is
the original logical view (effective address, dtype, shape, signed strides) plus
requested matrix geometry. It never uses a recyclable widening allocation as
identity and does not pin that allocation. Widened sources and transposes carry
independent grants; an executing transpose remains valid after the source copy
is reclaimed. Construction writes directly into the admitted transpose output.

The global f32 store, installation/eviction memos, global clear/byte diagnostics,
and the conversion-owner lookup are removed. `weight_transpose.rs` contains
only the shared transpose algorithm. Node owners share preparation across
shape kernels, while separate nodes/sessions cannot inherit entries at recycled
original addresses. Empty matrices retain nothing.

MatMul's optional f32 transpose can be refused and uses existing raw-layout
arithmetic. Gemm with `transB` requires a row-major B copy for its backend;
disabling retention now funds that copy transiently, and exhaustion returns an
actionable admission error before building. Other call-owned layout/scratch
allocations, including Gemm's A transpose, remain separate work. This step also
does not fuse source widening into transpose construction: a converted B may
still be materialized before looking up an existing transpose.

Forecasts now include the fused MatMul+bias+Relu f32 route on Apple and promoted
vector B operands. The registry forecast check describes declared routes;
actual provider and Gemm tests independently reconcile their Host grants. Source
validation, geometry/overflow, dtype isolation and pure transpose correctness
checks remain. Old global-map storage/clear/installation tests are replaced at
the actual node/cache seam: deterministic address reuse across node lifetimes,
grow/shrink/equal extents, concurrent equal/distinct geometries, shared owners,
pressure refusal/retry, independent readers and exact final refunds.

On Apple M1 Max, native CPU 1,984 / 20 ignored and optional MLAS CPU 2,024 /
35 ignored pass; sessions pass 277 / 279, with strict all-target Clippy in both
configurations. Public MatMul, FusedMatMulBias, Gemm and FusedGemm tests cover
matrix B; the MatMul family also covers vector B. Each admitted warm case makes
zero further claims/releases over 4,096 executions. The old public provider
fixture on parent `43055b185` compiles, selects one test and fails at its first
funding assertion: actual 0 versus required 140 Host bytes. No CUDA or other-ISA
execution or model-throughput result is implied.

A subsequent warm-path optimization uses only a weak last-hit slot for a
single-layout f32 cache. It cannot keep a payload or grant alive after idle
reclamation; multi-layout nodes retain canonical lookup/recency. The initial
full-key lookup added about 100 ns to a tiny case; the final coordinated run
reduces the overall increment to about 13 ns for f64 and 20 ns for f32 on this
Apple host. This remains measurable bookkeeping cost even though warm quota
requests/refunds are zero. CPU_MLAS_MIGRATION.md records the control arms and
ranges; no end-to-end model or other-tier performance is inferred.

### Prepare f32 transposes from original views

B preparation no longer accepts a separately materialized f32 slice. Its only
source is the validated original matrix/vector view. Conversion and transposition
share the tiled transpose traversal and write directly into the admitted final
f32 payload. This avoids a second matrix-sized allocation and removes the API
state in which supplied dense values could disagree with their identity view.
Contiguous aligned f32 keeps the existing tiled copy fast path; strided, reversed
and other float formats use original coordinates with unaligned numeric reads.

The shared prepacked compute path tries the prepared B before asking for a dense
B. Mandatory activation materialization still precedes optional B admission, so
an optional transpose cannot consume memory needed by A. Direct-output execution
keeps its original destination without a result-vector staging allocation. The
low-level slice compute context again carries only the prepack; optional MLAS
packing keeps its typed operand, unchanged.

Decode-only widened B now needs one `4*K*N` derived representation. A session
that also uses ordinary prefill may retain its independent dense B as well, but
KV pressure can reclaim that idle copy without decode rebuilding it or making
new quota calls. Gemm transB also builds its required retained/transient layout
directly from the original view. Other scratch and conservative graph forecasts
remain separate work.

Public factory controls cover MatMul, both fused MatMul forms and Gemm: a one-
representation budget succeeds from cold, 4,096 warm executions issue no further
claims/refunds, and decode continues with no quota calls after KV pressure
reclaims prefill's dense copy. The same test-only fixture compiles and selects
one test on parent `404fd04c4`, then fails at cold requests 3 versus expected 2
(including the setup anchor). All 65,536 bit patterns of f16 and bf16 are compared
bit-for-bit against the old conversion followed by an independent transpose;
f32/f64 special values and before-write extent checks are included too.

## Current design decision: execution workspace domains

Status: design reviewed; the CPU resource-domain foundation and first Gemm
transpose-workspace migration are implemented. Other scratch migration remains
pending. The unpublished per-node mutable-layout
pool prototype was withdrawn. The existing node-owned immutable weight preparation remains the
baseline. The next implementation is a CPU vertical through the existing
executor workspace contract, not another kernel-specific resource pool.

### Resource and ownership boundaries

| Resource | Owner / allocation unit | Authority interaction |
|---|---|---|
| Immutable prepared weights | Node preparation owner; independently reclaimable payloads or build groups | Cold build, growth and actual retirement |
| Temporary scratch | Provider/session resource domain; one exclusive workspace per concurrent execution slot | Slot creation, capacity growth, idle trim and destruction |
| KV and semantic persistent state | Their explicit state owners | Actual growth, shrink and release |
| Kernel-local scratch regions | Aligned borrowed slices of the prepared workspace | No claims or refunds |

Coarse allocation means pooling a real backing allocation with its existing
charge. It does not mean reserving the sum of every forecast at model load, or
introducing a second bank of unused budget credit. Stable-shape sequential
nodes reuse the graph's scratch peak; independent concurrent runs need separate
execution slots. A larger shape grows capacity at a preparation boundary.

Retain object-level weight reclamation: placing all model weights into one
session slab would let a small active reader pin an otherwise reclaimable slab.
A weight build group can preflight its live construction peak once and split
ownership afterward. `MemoryLease::split_off` avoids another claim, but each
child lease still refunds through the issuer on drop; it is not a hot scratch
suballocator.

### Reuse the existing workspace seam

`onnx-runtime-ep-api/src/kernel.rs` already supplies `workspace_requirement`,
`workspace_requirement_for_execution` and `execute_with_workspace`.
`onnx-runtime-session/src/executor/bindings.rs` collects workspace peaks and
`executor/dispatch.rs` passes `WorkspaceView`. Keep this single execution seam.
An EP describes checked size/alignment/layout and consumes borrowed regions;
the common resource layer owns admission, pooling, concurrency and release.
Direct kernel users should use the same public runner for workspace preparation.
Provide one common checked layout/partition helper for overflow, alignment,
capacity and non-overlap, so each EP supplies region requirements rather than
reimplementing raw byte-offset validation and allocator fallback.

Proposed lifecycle over the existing prepared-workspace allocation:

```text
run admission -> borrow exclusive execution slot -> dispatch with WorkspaceView
              -> run completion -> return backing + charge to idle storage
idle pressure / shutdown -> release backing -> settle its charge
```

Separate content lifetime from storage retention. Step-scoped contents expire
at run completion, while their backing may remain idle and funded for reuse.
Do not merely relabel temporary contents as `SessionPersistent`. State whose
values must survive calls remains independently owned and cannot be overwritten
by graph scratch reuse. An asynchronous slot enters pending until its completion
event; capture pinning and quarantine also prevent idle reuse and refunds.

The peak reflects overlap: mutually exclusive branches use a maximum; live
parent regions add to the child's peak; concurrent branches need the sum of
simultaneously active regions. Reuse is legal only at an established completion
or ordering boundary. Do not add per-kernel synchronization to obtain reuse.

Prepared-resource selection and scratch planning must agree. Either acquire and
pin the selected prepared readers before computing scratch, or declare a peak
that covers cache miss/refusal fallback. Workspace admission must not evict a
weight whose presumed presence was used to omit required fallback storage.

### CPU backing and accounting prerequisite

The CPU provider currently inherits the default workspace reservation returning
`None`. Merely advertising kernel workspace therefore does not prove a Host
grant. Wire CPU workspace once at the provider/resource-domain boundary using
`MemoryRole::Workspace` and the existing process memory manager's managed-owner
and physical-release transactions. `DerivedWeightMemory` has a weight-role
anchor; its sibling reservation must not misattribute scratch as weights.

The default CPU allocator includes `LargeAllocCache`. Returning an allocation
to that cache can retain its backing. An upper workspace pool must not refund
capacity while a lower, unaccounted cache still holds the block. Use one reuse
owner: trim either releases backing without reparking it, or transfers the same
managed owner and charge into the lower cache. Logical availability inside a
slot and physically releasable capacity are different facts, not two ledgers.

### Implementation sequence and evidence

1. Introduce borrow/return/idle-trim ownership for execution-slot workspace over
   the existing prepared allocation; centralize its policy in the resource
   domain, preserving active, pending, pinned and quarantined states.
2. Connect CPU Host workspace backing and actual Workspace-role charges, with
   one cache/release owner and an injectable authority for tests.
3. Move dynamic Gemm A layout scratch into this workspace seam. Keep immutable
   B preparation and checked original-view conversion; derive a single Gemm
   layout plan for its selected route and fallback requirements. Gemm already
   participates in workspace preparation; add no new op-name special case.
4. Migrate accumulator/TLS/Einsum scratch through the shared mechanism and
   remove their independent retention/accounting policies as each migrates.
   Replace `is_planned_workspace_node` with compiled workspace-plan facts.

The withdrawn prototype was useful evidence, not an implementation to publish:
per-call transient grants made a tiny dynamic A case roughly 288 -> 403 ns on
M1 Max. Retaining exclusive, refilled buffers eliminated warm issuer callbacks,
but still measured approximately 300--309 ns versus 289 ns; local and governed
arms were nearly equal. Zero issuer calls alone does not prove zero total cost.
The experiment's exact source patch, baseline fixtures, full validation drivers
and timing logs are preserved under `/tmp/nxrt-gemm-a-*`, with provenance in
`nxrt-gemm-a-candidate-provenance.json`. It was not merged into runtime code.

Required validation for the new implementation:

- Sequential heterogeneous kernels share one graph peak rather than a sum of
  node pools; simultaneous runs own independent slots.
- Stable-shape warm multi-node sessions perform no issuer calls, including
  zero-byte claims/refunds; growth admits only at the preparation boundary.
- KV pressure reclaims idle backing while active, pending and pinned backing
  remains funded. Cache refusal after route preparation remains executable
  within the declared live peak or fails before output writes.
- Allocation, execution and release failures preserve backing/charge ownership;
  lower allocator retention never becomes an unaccounted second cache.
- Shape growth, control-flow nesting, alignment and arithmetic overflow have
  explicit success/refusal tests using the public runner/session.
- Measure direct-kernel, common-runner and multi-node-session controls with the
  same host lock, so kernel savings cannot hide dispatcher overhead.

CPU tests and compilation may establish the Host contract. CUDA event ordering,
Graph replay, quarantine and actual physical release still require real-device
validation; no CUDA runtime or throughput claim follows from this design.

### Implemented foundation: managed Host execution workspace

`onnx-runtime-memory-governor::HostWorkspacePool` now borrows and returns whole
managed Host workspace owners. Clones share idle execution lanes. A warm fit
retains the original backing, authority and grant; capacity/alignment upgrades
retain their joint high-water marks and release the replaced idle backing
before admitting its replacement. Independent simultaneous executions receive
independent owners. The pool contains no active owner and rejects returns with
outstanding owned views. Physical release and issuer callbacks run outside pool
locks; failed release preparation preserves its owner in the error, and an
incomplete outcome remains charged/quarantined and is never reused.

The domain retains its pressure responder strongly; the responder weakly
references idle storage. Host authorities with a pressure registry can reclaim
idle slots for KV requests. Other authorities require explicit idle trim.
Private Host backing registrations retire after the scoped binding and verified
idle releases; outstanding/quarantined ownership conservatively keeps its
registration pinned. No empty budget-credit bank or per-kernel pool is added.

CPU workspace allocation now forwards to this domain and returns a managed
`DeviceBuffer` without an outer lease. Workspace backing uses uncached
`HostAllocator`; ordinary tensors keep their existing configured allocator.
Explicitly governed CPU resources require a matching workspace Host authority.
An ungoverned provider lazily creates an ungoverned managed domain reporting
unattributed physical bytes instead of inventing another budget.

Engine construction selects the final shared memory/manager before creating
CPU resources. Workers no longer construct CPU resources against temporary
books and subsequently replace only their governor fields. The Host ledger
identity now actually names the Host device, allowing canonical manager
registration against those same books. The executor records the capacity and
alignment returned by the provider, so a high-water backing is not needlessly
replaced when a later requirement fits it.

Native validation covers 77 governor unit tests (including six workspace
regressions), 36 manager transaction tests, 25 CPU provider tests, 277 session
tests, 15 engine governor tests and six authority tests, with strict all-target
Clippy. The warm checkout/return control observes every issuer callback and
performs 4,096 iterations without claims/refunds, including zero-byte calls.
Other controls cover concurrent slots, KV idle reclamation versus active
owners, alignment/capacity alternation, foreign or borrowed owners, delegated
process capacity, incomplete release and registration retirement. The worker
regression reconciles actual workspace and KV grants against the parent's Host
books. Optional MLAS configuration additionally passes 25 provider tests, 279
session tests and the exact worker regression; strict all-target Clippy passes
across governor, CPU EP, session and engine in both configurations. Raw
validation logs are `/tmp/nxrt-host-workspace-*`.

These foundation checks establish the resource domain; the first Gemm
migration below adds a real kernel consumer. Full control-flow scratch peak
validation, remaining kernel migration and CUDA runtime correctness still
require their own evidence.


### First kernel migration: Gemm transpose layouts

The requested subagent review confirmed the coarse allocation boundary: one
resource domain owns whole execution lanes, each kernel declares checked regions
and fills borrowed storage, and node-owned immutable weights remain separately
reclaimable. EPs keep the existing three workspace methods; no issuer, pressure
responder, mutable node pool or second execution API is added to Gemm.

`WorkspaceLayout<N>` performs fixed-size, stack-only layout arithmetic and binds
one workspace into disjoint raw subviews. Capacity, alignment, layout overflow
and address wrap are checked once. It deliberately does not return safe typed
references: the EP must establish Host/device accessibility, exclusive ownership
and lifetime before dereferencing. Fresh scratch is represented as
`MaybeUninit<f32>`; shared tiled traversal initializes every element once before
returning a readable slice. There is no preliminary zero fill or dense staging
matrix for a converted transpose.

Gemm's A transpose and dynamic/nonretained B transpose now use two simultaneous
regions within the execution slot. The required retained B transpose keeps its
existing node-owned grant and does not depend on an assumed cache hit. A
contiguous f32 unit-axis transpose can borrow the original linear storage;
strided unit-axis inputs still materialize, including negative strides and byte
offsets. Metadata-only planning covers unknown strides; actual views can shrink
the request, including to zero on the contiguous same-half computation route.
Missing or insufficient workspace fails before output writes. Direct kernel
callers must provide the same declared workspace, without a private allocator
fallback.

Immutable A's existing widening memo is preserved before filling its borrowed
transpose region, so a warm constant need not reconvert its source every call.
Dynamic/nonretained transposes fuse conversion directly from the original view.

This first migration intentionally leaves ordinary nontransposed widening,
output/bias buffers and half-kernel internal packing/scratch on their existing
paths. It also does not add an immutable A-transpose cache. Subsequent migration
must account for its own live peak rather than claim that all Gemm allocations
are now governed.

Native validation covers 99 EP API unit tests and 1,997 CPU unit tests (20
existing ignored tests), followed by the final 24 Gemm and three shared-transpose
controls. Final session validation covers 280 native tests; optional MLAS covers
2,038 CPU tests (35 existing ignored tests) and 282 session tests. Strict
all-target Clippy passes across all three crates in both configurations. The real governed Gemm control performs 4,096 workspace
checkout/execute/return cycles with zero issuer callbacks, including zero-byte
calls. A strict prepared two-Gemm graph uses one 32-byte peak allocation rather
than the sum of its 24- and 32-byte regions, and reuses the same managed identity
across calls. Other controls cover fresh uninitialized storage, simultaneous
regions, refill after source changes, half fallback, signed strides, offsets,
unit-axis borrowing, overflow and failure before output writes. These establish
Host ownership and execution behavior, not CUDA correctness or whole-model
throughput. Isolated timing evidence is recorded separately when available.


Nested executors now carry persistent and step workspace separately and select
by the requested lifetime. Forwarding prefers the inherited owner, matching
ordinary kernel dispatch, before considering an older local prepared slot.
Real If/Gemm regressions exercise both branches, smaller/larger persistent
backing, preserved persistent contents, and an older local step slot below or
above the inherited peak. The child creates no additional lane. This is Host
execution evidence; it does not establish asynchronous CUDA completion safety.

Isolated M1 Max native timing used the same stable host flock as the parallel
llama.cpp work, optimized builds, `RAYON_NUM_THREADS=1`, and alternating baseline
and candidate processes (three each). Each process reports the median of 15
batches; the table takes the median across processes. Baseline is `9a691aff5`,
whose tree equals merged foundation `8ff087186`. Kernel rows exclude preparation
and hold their workspace outside the timer; session rows include ordinary
public API dispatch, output collection and identical result checks. The shared
checked-out/returned-owner counter control above measures the issuer separately.

| Workload | Baseline | Workspace implementation |
|---|---:|---:|
| f32 A transpose, M=1 K=16 N=16 (linear-order equivalent) | 0.242 us | 0.218 us |
| f32 A transpose, M=4 K=16 N=16 | 0.335 us | 0.328 us |
| f32 A transpose, M=8 K=128 N=64 | 1.776 us | 1.614 us |
| f32 dynamic A+B transpose, M=2 K=32 N=16 | 1.097 us | 0.917 us |
| f32 no-transpose control, M=4 K=16 N=16 | 0.248 us | 0.253 us |
| Two-Gemm session run | 3.707 us | 3.909 us |
| Two-Gemm session prepare + run | 5.228 us | 5.285 us |

Zero issuer calls does not mean zero total runtime overhead. The tiny two-node
session still costs about 0.2 us more per run; this design amortizes slot handling
across the graph rather than paying it per kernel. Individual transpose wins
are not a whole-model throughput result. These timings cover the CPU/layout and
linear-session paths; nested lifetime selection is covered by the separate
functional regressions. Raw timing rounds, source/binary provenance and drivers
remain under `/tmp/nxrt-gemm-workspace-*`. Further slot/dispatcher improvements
must preserve these no-transpose and end-to-end controls.

### QLinearMatMul accumulator: one funded execution peak

The integer accumulator now belongs to the same execution workspace domain.
Its checked requirement is one logical `M * N * sizeof(i32)` tile, reused
synchronously across broadcast batches; it is not the full batched result size.
Empty batches or zero M/N need no accumulator. A nonempty zero-K output still
initializes its tile before requantizing the zero product. Geometry parsing
checks source spans, broadcast count, tile/output extents and byte capacity
before admission or execution. Common 2-D geometry uses inline shape storage
instead of allocating promotion copies and stride vectors for each query.

The TLS accumulator, independent process counter and public
`qlinear_accumulator_*` / `set_qlinear_accumulator_process_cap_bytes` interfaces
are removed without aliases. Engine resident-weight planning no longer adds a
fixed 128 MiB scratch forecast. Workspace backing and its existing charge are
reused independently of immutable-weight retention policy, and idle pressure
reclamation uses the common domain. The existing preparation inventory now
includes QLinearMatMul and its coverage guard; replacing that inventory with
compiled workspace facts remains separate work.

Native/nonfused accumulation initializes each tile before creating a readable
i32 slice. The MLAS fused reference accepts `MaybeUninit<i32>` scratch directly,
so it adds no memset and promises no initialized scratch afterward. Its zero-K
entry explicitly produces the requantized zero result; ordinary zero-K integer
GEMM overwrites stale accumulators with zero. Safe dense/packed/requantized
reference entry points check matrix element and byte spans before FFI, including
release-mode overflow. The packing constructor and size query also reject
unaddressable inputs.

Validation covers 2,001 native CPU tests and 282 native session tests, 2,042 CPU
tests and 284 session tests with optional MLAS, strict all-target Clippy across
CPU/session/engine in both configurations, ten integer-GEMM reference tests,
and a release-mode overflow refusal regression requiring no large allocation.
A two-node quantized graph uses a single 24-byte workspace peak instead of
summing 16 and 24 bytes. Concurrent calls hold separate funded lanes; a fourth
16-byte call refuses while three fill an actual 48-byte Host quota. Batch refill,
empty dimensions, vector promotion, zero contraction, nonfinite combined
scales and fused-to-native scratch reuse preserve the existing byte results.
The shape-variant pack test reconciles node-owned weight bytes separately from
the shared workspace charge and verifies both disappear through their owners.

The RSS example now uses an actual Host governor and shared workspace pool.
`QLINEAR_HOST_BUDGET_MIB` controls real admission; it reports admitted/refused
lanes and charged workspace bytes, not a fake per-thread retention total.
Scales, zero-point vectors, sign-domain copies, output staging and native/backend
packing scratch remain on their existing paths; accumulator migration is not a
claim that all quantized-kernel allocations or RSS are governed. The subsequent Einsum migration below removes the final consumer of the
legacy accumulator-budget helper.

Preparation now reuses dispatch's existing compiled-kernel bindings, then keeps
one successful metadata-only plan per executor. This memo holds the two
lifetime peaks, not physical storage or quota credit. Its signature uses exact
resolved symbols (including physical-capacity bindings), growing-axis
classification, artifact provider/executor/generation and kernel publication /
eviction counters. The graph, declared input dtypes/presence and immutable
weights are fixed for that executor. Input and external-binding validation
always precedes the memo lookup; unknown shapes that cannot be planned never
publish a memo. A hit still reserves both workspace slots, so a step return or
pressure eviction cannot turn a cached plan into uncharged execution. Metadata
requirements explicitly remain stable cold/warm upper bounds; runtime
refinement still runs at dispatch. Bound diagnostics deliberately bypass the
memo. Peak bytes and alignment are merged independently, including when a
smaller node needs stricter alignment.

Regression coverage includes changed row symbols, capacity refusal without
memo publication, dtype validation on a would-be hit, actual idle backing
reclamation followed by readmission, and 4,096 warm checkout/execute/return
cycles with retained constant B and **zero issuer callbacks**. This is a warm
admission property, not a claim of zero mutex, planning or dispatch cost. The
RSS smoke control admits four 16-byte lanes with a 1 MiB Host budget and
refuses all four with a zero budget; reclaimed charged bytes return to zero.

CPU-only performance controls on the same M1 Max / ARM64 macOS host compare
commit `9679414eb` (old TLS accumulator) with this change. Each mode runs in a
separate process; both arms warm 1,000 calls, collect 15 batches and use the
median of three process medians in alternating order. Both use the same
one-thread environment settings (`RAYON_NUM_THREADS`, `MLAS_NUM_THREADS`,
`OMP_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`); these settings are not a measured
MLAS worker-count claim. The stable host lock covers the entire paired sweep.
Every case verifies output bytes. Direct-kernel timing reuses output and holds
its workspace outside the timed region; the real two-node session timings
include workspace checkout/return and optional constant-weight retention.

| Mode | Native before → after | Optional MLAS before → after |
| --- | --- | --- |
| Tiny kernel, M=1 K=32 N=16 | 0.920 → 0.894 us | 0.788 → 0.747 us |
| Prefill kernel, M=64 K=256 N=256 | 343.237 → 343.343 us | 22.601 → 22.368 us |
| Two-node session run | 5.645 → 5.880 us | 6.192 → 6.571 us |
| Two-node session prepare + run | 5.816 → 6.113 us | 6.294 → 6.662 us |

The first version regressed repeated prepare + run by roughly 3 us: it walked
the graph and rebuilt metadata/cache keys every call. Reusing compiled bindings
and the pure planning memo removes that regression. The residual roughly
0.24–0.38 us/run on this very small graph is real slot/dispatch cost, despite
zero warm issuer callbacks. The larger kernel controls are approximately flat;
none of these CPU microbenchmarks establishes whole-model or CUDA throughput.
Raw initial, prebinding-only and final rounds, drivers, binary/source hashes
and summaries remain under `/tmp/nxrt-qlinear-*`. Consumed baseline worktree,
one-shot mutation scripts and probe binaries are removed after provenance is
recorded; retained drivers can rebuild the documented base for reproduction.


### Einsum accumulator workspace and a plain retention policy

Einsum now borrows its typed GenericNative output accumulator, floating oracle
output and materialized-GEMM canonical f32 output from the same StepScoped
execution domain. Its requirement is a checked output-sized layout using the
actual accumulator width; empty output is NONE. Successful contraction trees,
optimized view routes and scalar-GEMM routes reserve no unused generic output
buffer. Floating GEMM keeps a conservative f32 output bound because the direct
route also depends on output layout and aliasing, which metadata-only planning
does not observe.

The factory compiles tree choices once for each supported accumulator width
(1/2/4/8 bytes). Planning and execution consume the same immutable candidate
index and byte fact, with exact compiled-shape/schema validation. This removes
per-dispatch candidate-map construction instead of adding another mutable EP
cache or resource protocol. The existing 32 MiB tree limit is now explicitly an
algorithm-selection ceiling: larger candidates select lower-memory generic
execution; it is not a physical grant, a process cap or a retention budget.

Generic accumulation writes each final sum into MaybeUninit storage, including
zero reductions, and reads only after all output elements have been written.
Oracle preserves its existing initialization pass. Materialized GEMM explicitly
initializes the destination before calling MatMul, whose current contract
forms initialized f32 references. This adds a real pass on that route; removing
it requires a properly proven overwrite seam in MatMul, not an unsafe cast in
Einsum. Alias, negative-stride, precision and byte-result conformance controls
remain in place.

The TLS weak registry, typed slot owners and process accumulator ledger are
removed, along with the final governed_accumulator_budget module. Removed
scratch live/cap/prediction APIs and the obsolete default-opset compatibility
helper have no aliases. CpuCacheRetention now lives with CpuMemoryResources as
a plain immutable boolean verdict; it owns no scratch, mutex, thread registry
or physical allocation. Engine resident-cache prediction no longer invents a
128 MiB Einsum scratch resident. The benchmark owns an exclusive prepared lane
outside timing, and conformance exercises fresh uninitialized backing instead
of consulting TLS capacity.

Real Host-budget regressions alternate f64/f32 execution and opposite weight
retention policies over the same allocation, observe 4,096 warm
checkout/execute/return cycles with zero issuer callbacks, reject a second
active lane when the true cap is full and return charged bytes to zero on idle
reclamation. A two-reduction graph uses max(24,8)=24 bytes; If branches borrow
the enclosing 8-byte peak rather than reserving a second lane.

This removes the retained accumulator pool, not every Einsum allocation.
Contraction-tree/scalar-GEMM intermediates, operand materializations and layout /
coordinate vectors still use their existing allocations. Those transient allocations remain separate work. The following resource
propagation section addresses the lowered child MatMul source; no claim is
made that all Einsum transient bytes are already governed.

Validation: 1,986 native CPU and 284 native session tests, 2,027 optional-MLAS CPU
and 286 optional-MLAS session tests pass. Each configuration also passes all
three shared-corpus / parallel-tile / 128-operand fallback conformance tests and
the two public numeric/schema contract tests. Strict all-target Clippy passes
for CPU/session/engine in both configurations. The previous private-pool and
budget-helper tests are replaced by real-owner admission/reuse/reclamation
controls; existing arithmetic, precision, route and alias guards remain.

CPU-only paired release controls compare `e4130c5a8` (the previous TLS Einsum)
with this change on the same M1 Max / ARM64 macOS host. Each mode is a separate
process with 1,000 warmups and 15 timed batches; the table uses the median of
three process medians per arm in alternating before/after order. Both arms use
the same thread-count environment settings and hold the stable shared host
flock; those settings do not prove the backend worker count.
Direct-kernel scratch and outputs are prepared outside timing; public-session
rows include checkout/return, and prepare+run also includes preparation. Kernel controls verify nonzero output values and observe the actual execution
route; public-session controls verify their output values through the session
API;
these are CPU controls, not whole-model or CUDA throughput measurements.

| Mode | Native before → after | Optional MLAS before → after |
| --- | --- | --- |
| Tiny generic reduction (2×4 → 2) | 0.358 → 0.326 us | 0.348 → 0.317 us |
| Generic reduction (16×64 → 16) | 3.561 → 3.540 us | 3.456 → 3.430 us |
| Oracle reduction (16×64 → 16) | 4.966 → 4.923 us | 4.830 → 4.807 us |
| Direct GEMM (16×64×16) | 1.683 → 1.643 us | 1.723 → 1.733 us |
| Materialized/permuted GEMM (4,4,8,8 × 8,8,4,4) | 6.829 → 6.906 us | 6.997 → 6.970 us |
| Larger materialized/permuted GEMM (8,8,16,16 × 16,16,8,8) | 65.422 → 65.329 us | 65.380 → 65.266 us |
| Three-input contraction tree (8×16,16×16,16×8) | 36.712 → 19.956 us | 35.637 → 19.363 us |
| Two-reduction session run | 3.499 → 3.633 us | 3.486 → 3.536 us |
| Two-reduction session prepare + run | 3.654 → 3.791 us | 3.671 → 3.637 us |

Removing the private pool helps the tiny kernel. Compiling tree decisions moves
candidate planning out of execution, reducing this tree control by about 46%.
Its sampled native factory median increases from roughly 175 to 259 us; that
is a cold setup tradeoff, based on only three process samples, not a full model
startup result. The mandatory materialized-output initialization does not show
a material larger-case regression here. Tiny session run still costs roughly
0.05–0.13 us more despite zero warm issuer callbacks. Raw rounds, complete
sampling details, drivers, source/binary hashes and summaries remain under
`/tmp/nxrt-einsum-*`. Consumed probe binaries, one-shot generators and baseline
worktree are removed after provenance is recorded.


### Lowered MatMul keeps prepared ownership and the parent resource source

EinsumFactory now accepts CpuMemoryResources instead of only a cache-retention
boolean. Its child MatMul clones that source and policy, using the same
constructor as ordinary MatMul. It no longer creates a default, unattributed
domain inside a governed provider. The old boolean-only constructor is removed
and all callers are updated. Constant-state sharing/adoption delegates to the
existing MatMul node owner, retaining source and immutable-flag checks. Raw
constant bytes are not blindly forwarded to preparation with the wrong lowered
geometry.

Materialized lowering now passes the original ordered view to the existing
MatMul operand preparer, then keeps the returned DenseOperand until GEMM and
output publication finish. Immutable conversions reuse their node-owned grant
and complete original-view key. Optional MLAS packing consumes that same typed
operand: Funded retains its WeightIdentity, Borrowed retains the original view
identity, and call-owned conversion scratch has no retained packing identity.
The lowering no longer erases a granted operand into a raw f32 TensorView.
There is no second cache, scoped retention mask, policy clone or issuer protocol.
Dynamic input eligibility still comes from the immutable node flags and cannot
be promoted by the storage representation.

This also removes repeated immutable conversion work. Funded conversions remain
in the weight ledger instead of being counted again as scratch telemetry. An
explicitly declined retention policy continues to use the existing real
transient-grant semantics for constant conversion; that configuration is not a
zero-callback warm-retention claim. Activation conversion vectors still follow
their existing scratch path.

The typed dense entry checks complete input counts, f32 addressable spans,
checked matrix/batch/result products and output length before computation.
Empty results return before unused batch strides or packing; zero-K fills the
initialized output directly without packing or FFI. Empty Einsum GEMM also
returns before weight preparation. Original direct-view routes keep their
native preparation and backing metadata.

Real-source regressions require an owned lowered B materialization to obtain a
Host grant, preserve outputs on quota refusal and reject foreign-source or
constant-flag adoption. Explicit shape variants share preparation through
4,096 warm calls with zero issuer callbacks, yield idle cache to Host-budget
pressure and re-admit before reuse. The public session test covers several
prefill shapes and M=1 decode, whose GEMV transpose may legitimately add another
representation. Representations remain node-owned and stable across subsequent
shapes; independent sessions own independent preparations.

Prepared-reader tests exercise Generic, the host-selected backend and explicitly
forced MLAS when compiled. Concurrent readers preserve distinct values, active
grants survive cache pressure, and idle owners release all bytes. The recycled
conversion-address regression also invokes the new typed compute entry and
requires reuse of the existing funded pack identity. Malformed counts, byte
spans, broadcast overflow, empty and zero-K cases fail or return before packing.

This fixes actual resource propagation and prepared ownership. Other transient
vectors still need workspace migration, and advisory forecasts do not describe
every lowered weight artifact. Future forecasting should consume compiled
preparation facts rather than append operation-name exceptions to prediction
lists.

The resource-design review keeps the EP contract coarse: metadata declares a
workspace layout, the executor owns admission and checkout/return, and kernels
borrow slices. Immutable derived weights reuse the existing node owner and
source. Lowering passes that owner through instead of adding another cache or
per-operation quota interface. A shape variant shares immutable preparation,
while its mutable execution workspace remains exclusive to its admitted lane.

Validation of this stage: native CPU 1,993 and session 285 tests pass; optional
MLAS CPU 2,034 and session 287 pass. Both shared legal corpus/public-contract
suites pass (3 + 2 each). Strict all-target Clippy passes for CPU/session/engine
in both configurations and the optional reference adapter. Release-mode
prepared-operand tests (2) and geometry overflow/empty guards (1) pass. This is
CPU-only validation; CUDA behavior is untested.

Final paired release controls compare commit 1887ca137 with this stage on Apple
M1 Max / ARM64 / macOS 27. Native selects Accelerate; the optional MLAS sweep
explicitly forces MLAS. Both arms use the same 128 MiB Host cap, actual governed
source and funded workspace outside raw-kernel timing. The previous version's
zero attributed weight bytes do not mean zero physical allocation: its lowered
child used an unrelated default source. All constant controls declare B
immutable; the dynamic control declares neither input immutable.

Each mode runs in an isolated process, with 15 batches and the median of three
process medians per arm, ordered before/after/after/before/before/after. Small
controls warm 1,000 calls; large controls warm 20. Thread environment variables
are identical at 1, without claiming a measured backend worker count. The shared
stable host flock excludes the llama.cpp workload. Route counters and numeric
outputs are checked. These are warm CPU controls, not model/CUDA throughput or
cold-start measurements.

| Mode (all materialized outputs use dcab) | Native before → after (us) | Forced MLAS before → after (us) |
| --- | --- | --- |
| Dynamic direct (16×64×16) | 1.638 → 1.701 | 1.681 → 1.669 |
| Strided constant direct (16×64×16) | 1.785 → 1.812 | 1.676 → 1.690 |
| abxy,xycd (4,4,8,8 × 8,8,4,4) | 6.956 → 3.276 | 6.204 → 2.930 |
| abxy,cdxy (4,4,8,8 each) | 6.968 → 3.278 | 6.223 → 2.928 |
| abxy,cdxy (8,8,16,16 each) | 65.790 → 18.832 | 65.889 → 19.378 |
| abxy,xycd (2,2,32,64 × 32,64,64,64) | 23476.383 → 3350.467 | 22677.867 → 331.650 |
| abxy,cdxy (2,2,32,64 × 64,64,32,64) | 54751.950 → 3284.158 | 54419.042 → 323.383 |
| Tiny Einsum public session run | 3.444 → 3.453 | 3.745 → 3.766 |
| Tiny public session prepare + run | 3.652 → 3.685 | 3.953 → 3.920 |

Retaining original-view conversion and packing removes repeated constant work.
The larger native controls retain 32 MiB of attributed weights; forced MLAS
retains 64 MiB (conversion plus pack), in addition to the 64 KiB output
workspace. The 128 MiB cap admits this real combined peak in both arms. Small
materialized controls retain 4/8 KiB native/MLAS; the 8,8,16,16 case retains
64/128 KiB. This is an explicit memory-for-reuse tradeoff; real-pressure tests
verify reclaim and re-admission. Internal probe labels such as borrowed/owned
are case identifiers, not assertions about the returned storage ownership.

The earlier scoped-policy experiment regressed one forced-MLAS prefill control
by about 13%; it was discarded. The final typed pipeline reduces that control
from 65.889 to 19.378 us. Tiny direct controls still add about 14–62 ns depending
on backend/case. Tiny public run adds 9 ns native and 21 ns forced MLAS; prepare
+ run changes by +33/-33 ns. Zero warm issuer callbacks therefore do not imply
literally zero runtime overhead. Raw rounds, source/driver/binary hashes and
summaries remain under `/tmp/nxrt-einsum-weight-prepared-*`; the superseded
experiment is separately labeled under `/tmp/nxrt-einsum-weight-*`.
