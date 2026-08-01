# Decision: weight-offload async page-in → opt-IN (default sync)

- Date: 2026-08-01
- By: Cohaagen (EP/runtime)
- Lineage: #544 (weight-offload) follow-up; supersedes the async-default-ON introduced with the async fence-ordered page-in (issue #87 infra).
- Branch/PR: `fix/weight-offload-async-pagein-opt-in`

## Decision

Invert the default of `ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN`:

- **Before:** async page-in default-ON; only `=0` opted out.
- **After:** async page-in **opt-IN**. Unset / falsey → synchronous page-in (new default). A truthy value (`1`/`true`/`yes`/`on`, case/whitespace-insensitive) opts into async.

Only the DEFAULT changes. The async fence-ordered page-in path is fully preserved behind the flag (it is the A/B "after" arm and the anti-regression fence test still guards it).

## Why — measured A/B (the data)

qwen3-0.6b-int4, native CUDA, weight-offload engaged, eviction/thrash regime (small device budget so every admit evicts):

| Config (device budget 96 MiB) | tok/s |
|---|---|
| async page-in ON  | 12.16 |
| async page-in OFF (sync) | **15.84** |

Sync is **~1.30× faster** in the eviction regime. Async net-regresses and is never a clear win on any on-hand model.

Per-page-in tax breakdown (96 MiB, async), from the inc-2 instrumented probe:

| term | ms |
|---|---|
| materialize (int4 canonical-bytes staging/dequant prep) | 791 |
| pinned host-buffer alloc + copy | 792 |
| raw H2D copy | 46 |
| eviction compute-stream drain | 15 |
| fence wait | 7 |

The transfer (H2D) the async path tries to overlap is ~3% of the cost. Materialize + pinned-staging co-dominate (~48% each) and sit on the critical path; when every admit evicts, the eviction drain re-serializes, so async cannot overlap anything and only adds the non-overlappable pinned-staging alloc. Async becomes a net win only once a warm-host materialize cache keeps pinned canonical bytes warm (removes both dominant terms from the per-page-in critical path).

## Correctness / blast radius

- **Byte-exact:** offloaded == resident token stream is unchanged (weight_paging §9 invariant). Verified on qwen3-0.6b-int4 native CUDA e2e with the NEW sync default: tokens byte-identical to resident baseline; page_ins=12544, evictions=12541 (non-vacuous).
- WAR / eviction-drain safety and fence-ordering primitives are UNTOUCHED. The async fence anti-regression GPU test (`async_pagein_fence_orders_weight_page_in_consumer`) still passes — it drives the async primitives directly, independent of the default.
- Capture interaction: none. Dynamic page-in is outside any captured region (already established); this flag doesn't touch #571 capture.
- Blast radius = flag default + tests + docs only. Pager internals, capture, and GAP-3 untouched.

## Tests / evidence

- Unit (`weight_paging::tests::async_pagein_env_is_opt_in`): `None→false`, `1/true/YES/ On →true`, `0/false/""/maybe→false` (non-vacuous both directions).
- `device_policy_defaults_to_disabled` extended to assert `!policy.async_pagein` (default sync).
- e2e `weight_offload_native_cuda_e2e`: asserts the resolved `from_env()` policy is sync by default AND offloaded tokens == resident (byte-identical) on real fp16/int4 GPU.
- Async fence GPU test unchanged and passing (guards the async path when opted in).

## Escape hatch

Async stays available via `ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN=1`. Revisit the default once a warm-host materialize cache lands — at that point async should overlap and may flip back to a net win.
