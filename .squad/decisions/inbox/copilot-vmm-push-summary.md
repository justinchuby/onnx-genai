### 2026-08-12: What the VMM / offload / streaming / batching push established
**By:** Copilot (coordinator)
**What:** A record of the load-bearing results from this push, so the reasoning
survives in the tree rather than only in issue threads.

Every claim below is backed by a merged, executable test. Where a claim was
*refuted*, that is recorded too — the refutations were often worth more than the
confirmations.

#### The governing rule (measured, #772/#776/#787)

`cuMemMap` maps whole granule-aligned windows onto whole physical granules, so
`committed bytes = granule x (windows containing at least one live byte)`.
**Layout therefore controls residency**, because layout decides where the live
bytes fall in virtual address space. The allocator cannot compact what the layout
scattered.

`CU_MEM_ALLOC_GRANULARITY_MINIMUM == RECOMMENDED == 2 MiB` on this device, so the
floor cannot be fixed by shrinking the granule — only by layout. This is a
**CUDA-specific severity**: minimum mapping granularity spans ~500x across
platforms (Level Zero and Vulkan ~64 KiB, CPU `mmap` 4 KiB), which is why layout
must be a queried per-EP, per-platform capability (#783) rather than a constant.

#### Confirmed

- **Floor is layout-determined:** 768 granules (~1.5 GiB) head-major -> 96 (~192 MiB)
  seq-major -> 1 per sequence (~2 MiB) token-major, the last measured at a **768x**
  reduction (#787).
- **Strided reads are not the obstacle:** seq/head bandwidth ratio 0.80-1.02, and
  a 192 KB token-major stride measured **1.000** at a 6 GiB working set — reads are
  DRAM-bound independently of stride, because device memory is already 2 MiB-page
  backed (#778/#787).
- **Offload and capture are no longer mutually exclusive** (#796). Weights page
  under a stable VA, so page-in remaps physical granules instead of returning a new
  pointer. This unblocked #755.
- **Managed no-spill VMM is the default**, with automatic weight streaming when a
  model exceeds the budget (#798). A fitting model does **not** page: verified
  `FullResident`, offload off, **0 page-ins**.
- **Prefix sharing is sound:** one handle maps into N=8 sequences under captured
  replay; the ledger charges it **once**, keeps it alive until the last sharer, and
  an additional sharer costs **0** incremental bytes (#793/#803).

#### Refuted, and why the refutation mattered

- **"seq-major landed means the 8x floor is realised."** It did not: #794 measured
  head-major and seq-major committing **identical** physical bytes, because the
  bindings did not consume the layout descriptor. Fixed in #797. I had stated the
  8x as achieved and had to correct the floor table before merging.
- **"The decoder structurally declines capture for this model."** It does not
  (#804). `captures=0` came from a cached `ONNX_GENAI_CUDA_GRAPH=0` in a
  long-lived test process. **Two PRs (#794, #801) misattributed a failed
  measurement to this.**
- **"A fixed KV stride removes growth-triggered re-capture."** True in mechanism,
  irrelevant in practice: seq-major moves **0 bytes** on growth versus head-major's
  688,576, and the graph is invalidated anyway because the engine invalidates
  **unconditionally** on growth (#805). The real blocker was never the layout.
- **"Tokens per granule" as a KV cost model.** Wrong for head-major — publicly
  retracted — yet exactly correct for token-major. Same granule, same model, same
  VMM; only the layout differs. It is the clearest demonstration that layout is the
  whole story.

#### The audit's recurring finding (#736, six slices)

Four of five completed slices found **over-reservation** — bytes charged against
the device authority on a path that never uses them — rather than the ungoverned
allocation the audit was framed to look for:

| Slice | Finding |
|---|---|
| #751 IndexShare | staging the common 3-output decode path never touches |
| #795 GQA `WS_SCORES` | needed only on the f32 reference route (~128 MiB hot path) |
| #799 cuBLASLt GEMM | 32 MiB was a heuristic **ceiling**; measured use 0-96 bytes |
| #802 default-domain Attention scores | **genuinely needed** on every route |
| #806 GQA QKV staging | unpacked and fp16/bf16 fused-decode routes charge 0 |

Guidance, now recorded in `MEMORY_ARCHITECTURE.md`: **start from use, not from
allocation.** Locating `alloc_raw` is the easy part; the valuable question is what
the bytes are used for on each dispatch route. And governing a bypass *without*
sizing it to use makes things worse — it converts invisible waste into charged
waste, tightening committed-granule admission (#745) and therefore reducing
admissible concurrency.

#### Method notes that earned their keep

- **Order-dependent test state cost two wrong conclusions this week** — a
  process-frozen `RuntimeConfig` (#804) and a CUDA context warmed by an
  alphabetically-earlier sibling (#797). On a measurement-driven line a wrong
  number does not merely fail, it redirects design. #807 added a debug-only freeze
  guard and a single-stream test helper, plus an inventory of the remaining
  order-dependent tests.
- **Negative results were delivered as first-class outcomes** — the granule lever
  being unavailable, quantization being a split result (tokens/granule improve but
  the byte floor does not), and #802's "genuinely needed". Each closed a question
  rather than leaving it to be re-investigated.
- **Never extrapolate an unmeasured number.** Where a large-model run was
  impossible (`qwen14b-zp` lacks `inference_metadata.yaml` and is not
  native-loadable, #384), that was reported as not measured with the reason.
