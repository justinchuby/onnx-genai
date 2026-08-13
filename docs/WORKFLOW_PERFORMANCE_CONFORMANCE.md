# Workflow performance conformance

Metadata-driven execution is acceptable only when it is competitive with an equivalent native
ONNX composite under identical conditions. Functional equivalence alone is insufficient.

## Comparison contract

Both paths must use the same ONNX operations and weights, ORT build, execution provider, device,
batch and tensor shapes, dtype, sampling algorithm and request parameters, KV mode, stable
bindings, graph-capture setting, warmup, and measured iteration count. The workflow path links
adjacent pure same-device invokes into one execution island. The native baseline is the same
operations authored as one ONNX graph.

Run at least five paired samples with at least 100 iterations each. Alternate which path runs
first, report the median paired throughput ratio, and run on an otherwise idle device. Report
individual samples when variance changes the conclusion.

## Required metrics

`WorkflowPerformanceDiagnostic` reports workflow runtime, TTFT, loop iterations, logical
component invocations, emits, elements, step/s, and element/s.
`ExecutionIslandDiagnostic` reports:

- logical components, linked ONNX nodes, elided component boundaries, and ORT session runs;
- stable-binding, eager, capture, and replay counts;
- H2H, H2D, D2H, and D2D copy counts and bytes;
- explicit device synchronizations;
- stable-binding and external-initializer bytes;
- CUDA total/free-memory samples and the observed post-session high-watermark delta;
- cumulative island runtime and any capture fallback reason.

The CUDA free-memory delta is a process-level observation, not allocator attribution. Measure on
an idle device and also report external process activity. Kernel/provider placement must be
collected from an ORT profile for release measurements; linked node count is not a kernel count.
The profile must report CPU/CUDA node counts, memcpy nodes, total memcpy time, and the dominant
operators.

## Acceptance

- The prepared synthetic policy-chain benchmark must reach a median paired steady-state
  throughput ratio of at least `0.98`.
- Warm-session workflow TTFT overhead must be at most 250 ms for the synthetic policy benchmark.
  Cold-start latency is reported separately because process-global CUDA/cuDNN initialization makes
  sequential in-process cold comparisons order-dependent.
- A capture-eligible island must capture once and replay every measured steady-state invocation.
- Adjacent policy invokes must use one ORT session and have no host round trip for internal
  values. Only declared package outputs may be copied to the host.
- Any material gap must be reported with copy, synchronization, session, capture, memory, and ORT
  profile evidence. A noisy or unavailable measurement does not count as a pass.

Real-model readiness additionally requires decoder+sampler+termination and speculative policy
measurements with the production KV service. The PR remains draft until those producer packages
meet this bar.

## Synthetic benchmark

```bash
ONNX_GENAI_ORT_LIB=/path/to/libonnxruntime.so \
ONNX_GENAI_ORT_LIB_DIR=/path/to/ort/providers \
ONNX_GENAI_WORKFLOW_PERF_EP=cuda \
ONNX_GENAI_WORKFLOW_PERF_ITERS=200 \
ONNX_GENAI_WORKFLOW_PERF_SAMPLES=5 \
cargo test -p onnx-genai-engine --features cuda,cuda-13000 \
  --test workflow_performance_conformance -- --ignored --nocapture
```

The compact test covers a **synthetic policy chain**, not a transformer decoder: its first
component is one Softmax over `[32, 32768]`, followed by sampler and termination components. The
min-p variant adds ReduceMax, threshold, and mask operations, but its greedy ArgMax cannot
demonstrate categorical min-p sampling quality. Both paths prepare persistent bindings once and
keep unchanged inputs resident; rebuilding requests, SSA maps, and buffers is deliberately outside
the per-step timing.

`real_muse_policy_chain_matches_direct_ort` accepts a generated local Muse-Glimmer package through
`ONNX_GENAI_MUSE_WORKFLOW_PACKAGE`. It links the real 30B decoder, generic sampler, and termination
graphs, and compares that island with the same direct ONNX composite using ORT 1.28, stable
bindings, CUDA Graph, one-token decode shapes, five paired 200-iteration samples, and resident
unchanged inputs. This is real decoder compute evidence; a release claim still needs a producer
package exercising its production shared/paged KV service and a separate prefill TTFT measurement.

`workflow_combines_speculation_grammar_and_adaptive_budget` is the speculative capture
conformance: verifier + accepted-length policy and adaptive-budget + guided-sampler each form a
pure island, while the stateful grammar adapter deliberately delimits them. With CUDA graph
capture enabled, three runs must produce one capture and at least one replay per eligible island.

Use `ONNX_GENAI_WORKFLOW_PERF_MIN_RATIO` and
`ONNX_GENAI_WORKFLOW_PERF_MAX_TTFT_OVERHEAD_MS` only to investigate a known failure. Published
acceptance results use the defaults.

## Current diagnostic interpretation

The first workflow invocation discovers artifact-inferred output extents before allocating stable
bindings, so its cold-start latency is higher than a baseline whose output shapes were supplied
directly. Subsequent invocations use stable bindings and CUDA graph replay. Production sessions
should prewarm; eliminating discovery through planner-provided inferred output extents remains a
startup optimization rather than a reason to relax steady-state or warm-TTFT acceptance.

Prepared plans track the backing pointer of each stable input slot. Unchanged values are not copied
again; `set_input` replaces the slot and triggers exactly one refresh. Single-run island outputs
are exposed as no-copy aliases until package-output materialization, while repeated loop-island
outputs retain independent storage. Shared-buffer KV aliases bind declared past/present ports to
the same device allocation and refresh only at the start of a prepared execution.
