# FreeToken byte estimates and observed production receipts

FreeToken byte evidence has two deliberately separate outputs:

1. `freetoken_byte_ab` is a deterministic **synthetic estimate model**. It
   performs no loader, host allocation/write, CUDA copy, VMM operation, page-in,
   or state publication. Every modeled value is nested under an `estimated_*`
   field and carries `declared_synthetic_model` provenance.
2. `freetoken_byte_accounting_gpu` drives real CUDA runtime and governed weight
   residency operations. Its `onnx-genai.freetoken-observed-bytes.v1` events are
   recorded at the production boundary from actual arguments/results.

Estimated values are never compared or presented as observed bytes.

## Synthetic estimate model

```bash
cargo run -p onnx-genai-bench --no-default-features \
  --bin freetoken_byte_ab -- \
  --fixture deepseek-like \
  --output /datadisks/disk5/justinchu/freetoken-byte-ab/deepseek-like-estimate.json

cargo run -p onnx-genai-bench --no-default-features \
  --bin freetoken_byte_ab -- \
  --fixture glm52-like \
  --output /datadisks/disk5/justinchu/freetoken-byte-ab/glm52-like-estimate.json
```

The output schemas are:

- workload: `onnx-genai.freetoken-byte-ab.workload.v3`;
- estimate run: `onnx-genai.freetoken-byte-ab.estimate-run.v3`;
- estimate comparison: `onnx-genai.freetoken-byte-ab.estimate-comparison.v3`;
- estimate taxonomy: `onnx-genai.freetoken-byte-estimate-taxonomy.v1`.

Run objects contain `estimate_provenance`,
`observed_production_events = not_observed_synthetic_estimate_only`,
`estimated_phases`, `estimated_totals`, `estimated_residency`, and
`estimated_failure`. Comparison deltas are named `estimated_deltas`.

The built-in fixtures are structural analogues, not model gates or checkpoint
claims:

- `deepseek-like`: 256-expert typed banks plus compressed/dense state extents;
- `glm52-like`: grouped expert banks plus recurrent/attention/temporal extents.

They preserve deterministic routes, state progression, and output-token
digests. Declared byte extents answer only “what would this model charge under
these assumptions?” They do not establish what production read, allocated,
copied, mapped, or published.

## Observed production ledger

The CUDA EP exposes a provider-owned session opener:

```text
CudaExecutionProvider::open_observed_byte_session(
    executor,
    generation,
    logical_session,
    event_capacity,
)
```

The provider derives provider/device identity itself. Callers cannot supply a
foreign provider id, attach a ledger through `CudaRuntime`, construct the
private recorder, clone mutation authority, or reuse one recorder on a sibling
provider. The public ledger permits phase selection, snapshot, reset, and close;
production mutation remains inside the exact provider instance.

Every event carries:

- provider;
- CUDA device;
- executor;
- generation;
- logical session;
- ledger epoch;
- submission id;
- event sequence;
- phase;
- category;
- production boundary;
- terminal status;
- exact byte argument/result.

The event ring is preallocated and bounded. Enabled recording performs no
per-event heap allocation, string construction, map lookup, or mutex
acquisition. Default-off providers allocate no ledger and do not read an
environment variable or add a copy/synchronization.

## Atomic submission contract

A submission is assembled in fixed-capacity storage. Commit:

1. validates ledger instance, epoch, open/fault state, event capacity,
   submission/sequence capacity, global category totals, phase totals, and all
   checked additions;
2. writes the complete event batch and terminally removes pending state under
   one short atomic gate.

No event or byte total is visible until all validation succeeds. Overflow at
the first, middle, or last event commits nothing and leaves the submission
pending. It can be explicitly aborted, or dropping it faults the ledger so a
snapshot cannot report success. Repeated finish/abort and close are idempotent.
Snapshot/reset/close use the same linearization authority and refuse in-flight
submissions. A recorder used after reset faults the new epoch rather than
silently dropping telemetry.

## Categories, layers, and provenance

Categories are separate accounting planes. They must not be summed into one
“total bytes” value:

| Category | Observed production meaning |
| --- | --- |
| `source_read` | Bytes returned by an actual source read. The synthetic mmap source has no file read and emits `unsupported`, zero bytes. |
| `mmap_page_in` | Bytes confirmed by an OS page-fault/page-in receipt. The current synthetic source emits `unsupported`, zero bytes. |
| `host_allocation` | Capacity of a successful pinned-host allocation. Pool reuse is `reclaimed`, not a new allocation. |
| `host_write` | Bytes actually copied into pinned staging. |
| `device_allocation` / `device_release` | CUDA runtime allocation/reuse/release results. |
| `h2d`, `d2h`, `d2d` | Transfer payload. Useful bytes require a completed synchronous/event receipt. Enqueue-only paths are `submitted` plus `unsupported`, never useful. |
| `cuda_memset` | Bytes covered by a completed CUDA memset. |
| `vmm_reserve` | Address bytes returned by a real VMM reservation. A provider arena created before ledger attachment is reported `not_observed`, not inferred per page. |
| `vmm_map` / `vmm_unmap` | Physical mapping bytes returned by VMM commit/decommit outcomes. |
| `page_in` | Canonical weight payload successfully published by governed residency. |
| `expert_publication` | Payload made available to the production expert residency owner. |
| `state_publication` | State bytes published by an instrumented state boundary. The current fixture run reports this `not_observed`; it does not infer state traffic. |

Statuses keep attempted, useful, and recovery traffic distinct:

- `submitted`;
- `completed`, `committed`, `published` (useful within their own category);
- `failed`;
- `rolled_back`;
- `quarantined`;
- `reclaimed`;
- `unsupported`.

Logical page/expert payload is not added to physical H2D or VMM bytes.
`vmm_map` may differ from `page_in` because an already-owned granule can be
reused without a new physical map.

## Production instrumentation points

Observed events currently originate from:

- `CudaRuntime::{alloc_raw, free_raw, alloc_pinned, htod, dtoh, dtod,
  htod_async_elapsed_ms, memset_zero}`;
- enqueue-only H2D/D2D runtime paths, explicitly marked unsupported for useful
  completion accounting;
- `PinnedStagingPool::acquire`;
- `fill_staging_from_regions` after exact mmap-region copies;
- `CudaWeightResidency::admit_committed_span` from
  `SpanCommit::newly_mapped_bytes`;
- governed page-in and expert publication;
- deferred VMM unmap/reclaim/quarantine outcomes;
- failed VMM fill rollback/quarantine;
- exact per-bank route-reservation construction, mapping, H2D completion, and
  expert publication;
- `CsaCheckpointJournal::{checkpoint, restore_prefix}` state publication and
  rollback after completed D2D copies.

No harness-layer `stage_expert_load(bytes_per_expert)` exists in observed
accounting.

## A/B workload and denominators

The GPU test executes identical deterministic route lists for baseline and
optimized arms:

- baseline: completed production CUDA H2D streaming for every routed unique
  expert;
- optimized: the existing governed mmap → pinned staging → completed H2D →
  VMM residency → expert publication path.

Both arms verify device bytes against the same synthetic source. Setup,
prefill, direct warmup, replay-labeled workload steps, warmed decode,
verification, failure, and teardown are separate phases. The warmed comparison
uses exactly four decode steps × batch two = eight generated-token
denominators. Validation D2H is recorded under `verification`, never charged to
decode.

The boundary control also captures a fixed-address D2D graph and replays it
three times. Replay output is byte-identical. Because the generic graph replay
path does not yet return a per-copy completion receipt, that captured D2D is
reported as `submitted` + `unsupported`, not useful D2D; the separate
synchronous D2D control supplies the exact completed-byte receipt.

Run on an idle A100:

```bash
CUDA_VISIBLE_DEVICES=<idle-a100> ONNX_GENAI_CUDA_DEVICE=0 \
ONNX_GENAI_FREETOKEN_OBSERVED_OUTPUT_DIR=<private-output-dir> \
cargo test -p onnx-genai-bench --no-default-features \
  --features gpu-tests,cuda-13000 \
  --test freetoken_byte_accounting_gpu -- \
  --test-threads=1 --nocapture
```

The reports contain the complete ordered event ledger and per-category decode
summaries. `category_coverage` distinguishes `observed`, `unsupported`, and
`not_observed`; `mapped_bytes_not_reclaimed` exposes committed VMM bytes that
have not produced a reclaim receipt at snapshot time. No declared extent is
substituted.

## Limitations

The payloads are synthetic and generated in memory. No official DeepSeek or GLM
checkpoint is loaded. These tests establish production boundary accounting,
identity isolation, transactional publication, route equivalence, and exact
byte receipts. They do not establish model quality, checkpoint I/O, tokens per
second, or end-to-end checkpoint throughput.
