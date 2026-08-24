# Documentation index

Topic directories, most-used first. Each entry says what the document is
**authoritative for**, because several of these overlap and the wrong one will
give you a stale answer.

## Start here

| If you want | Read |
|---|---|
| What the project is and how it is put together | [`architecture/DESIGN.md`](architecture/DESIGN.md) |
| What sessions promise about threads, and why | [`architecture/SESSION_CONCURRENCY.md`](architecture/SESSION_CONCURRENCY.md) |
| Which CUDA feature to build with, and why there are two | [`build-features.md`](build-features.md) |
| Why memory is arranged the way it is | [`memory/MEMORY_ARCHITECTURE.md`](memory/MEMORY_ARCHITECTURE.md) |
| The proposed cross-stack memory contracts | [`memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md`](memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md) |

## Directories

### `architecture/`
Project-level structure and the ORT2 direction: [`DESIGN.md`](architecture/DESIGN.md),
[`ORT2.md`](architecture/ORT2.md), [`ORT2-IMPL-PLAN.md`](architecture/ORT2-IMPL-PLAN.md),
[`TRANSITION_PLAN.md`](architecture/TRANSITION_PLAN.md),
[`CPP_VS_RUST.md`](architecture/CPP_VS_RUST.md), the `onnx-rs` IR
([`ONNX_RS.md`](architecture/ONNX_RS.md),
[`ONNX_RS_SPEC_COVERAGE.md`](architecture/ONNX_RS_SPEC_COVERAGE.md),
[`IR_CONTAINER_REFACTOR.md`](architecture/IR_CONTAINER_REFACTOR.md),
[`GRAPHVIEW_LENS_DESIGN.md`](architecture/GRAPHVIEW_LENS_DESIGN.md),
[`INDEXSHARE_DESIGN.md`](architecture/INDEXSHARE_DESIGN.md)), the plugin ABI
([`NXRT_ABI.md`](architecture/NXRT_ABI.md)), and cross-cutting conventions
([`ERROR_AND_LOGGING_CONVENTIONS.md`](architecture/ERROR_AND_LOGGING_CONVENTIONS.md),
[`SESSION_CONCURRENCY.md`](architecture/SESSION_CONCURRENCY.md)
(**authoritative for session thread-safety, session ownership and the exclusive-lease
refusal**),
[`CROSS_PLATFORM.md`](architecture/CROSS_PLATFORM.md),
[`MINIMAL_BUILD.md`](architecture/MINIMAL_BUILD.md),
[`PYTHON.md`](architecture/PYTHON.md),
[`STRING_TENSOR_RUNTIME.md`](architecture/STRING_TENSOR_RUNTIME.md),
[`CRATE_RESERVATION.md`](architecture/CRATE_RESERVATION.md)).

### `memory/`
The memory line — VMM, weight offload, KV residency, prefix sharing.
**[`MEMORY_ARCHITECTURE.md`](memory/MEMORY_ARCHITECTURE.md) is authoritative for
measured behaviour**; [`MEMORY_MANAGEMENT_MODEL_DESIGN.md`](memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md)
(+ [appendix](memory/MEMORY_MANAGEMENT_MODEL_DESIGN_APPENDIX.md)) is the
proposed contract set for the wider ORT stack;
[`WEIGHT_OFFLOAD.md`](memory/WEIGHT_OFFLOAD.md) holds the north-star scorecard.
Supporting investigations: [`KV_INSERTION_DESIGN.md`](memory/KV_INSERTION_DESIGN.md),
[`PREFIX_SHARE_INVESTIGATION.md`](memory/PREFIX_SHARE_INVESTIGATION.md),
[`TOKEN_MAJOR_KV_INVESTIGATION.md`](memory/TOKEN_MAJOR_KV_INVESTIGATION.md),
[`SEQ_MAJOR_KV_INVESTIGATION.md`](memory/SEQ_MAJOR_KV_INVESTIGATION.md),
[`VMM_KV_CONTIGUOUS_VA.md`](memory/VMM_KV_CONTIGUOUS_VA.md),
[`DECODE_POOL_RESIDENCY_DESIGN.md`](memory/DECODE_POOL_RESIDENCY_DESIGN.md),
[`OUTPUT_BUFFER_REUSE_DESIGN.md`](memory/OUTPUT_BUFFER_REUSE_DESIGN.md),
[`GQA_KV_MATERIALIZATION_DESIGN.md`](memory/GQA_KV_MATERIALIZATION_DESIGN.md),
[`PRESSURE_PROTOCOL_IMPL.md`](memory/PRESSURE_PROTOCOL_IMPL.md),
[`native-ort-kv-capacity.md`](memory/native-ort-kv-capacity.md).

### `execution/`
Execution providers, kernels, graph capture and placement:
[`NATIVE_CUDA_DECODE.md`](execution/NATIVE_CUDA_DECODE.md),
[`CUDA_STRATEGY.md`](execution/CUDA_STRATEGY.md),
[`CUDA_COVERAGE.md`](execution/CUDA_COVERAGE.md),
[`CUDA_GRAPH_CAPTURE.md`](execution/CUDA_GRAPH_CAPTURE.md),
[`CUDA_FLASH_ATTENTION.md`](execution/CUDA_FLASH_ATTENTION.md),
[`CUDA_CSA_PHASE_B_PLAN.md`](execution/CUDA_CSA_PHASE_B_PLAN.md),
[`CUDA_EP_STATUS.md`](execution/CUDA_EP_STATUS.md),
[`GQA_DECODE_PARALLEL.md`](execution/GQA_DECODE_PARALLEL.md),
[`design-ep-partial-cuda-graph.md`](execution/design-ep-partial-cuda-graph.md),
[`HETEROGENEOUS_PLACEMENT.md`](execution/HETEROGENEOUS_PLACEMENT.md),
[`GRAPH_PARTITION_PERF.md`](execution/GRAPH_PARTITION_PERF.md),
[`EAGER.md`](execution/EAGER.md),
[`EP_CONFORMANCE.md`](execution/EP_CONFORMANCE.md),
[`EP_CLAIM_DIAGNOSTICS.md`](execution/EP_CLAIM_DIAGNOSTICS.md),
[`OPERATORS.md`](execution/OPERATORS.md).

### `ep-plugin/`
The EP plugin export workstream — ABI truth, gaps, security audit and test plan:
[`EP_PLUGIN_EXPORT.md`](ep-plugin/EP_PLUGIN_EXPORT.md) and its
[`ABI_TRUTH`](ep-plugin/EP_PLUGIN_EXPORT_ABI_TRUTH.md),
[`ABI_GAPS`](ep-plugin/EP_PLUGIN_EXPORT_ABI_GAPS.md),
[`INVENTORY`](ep-plugin/EP_PLUGIN_EXPORT_INVENTORY.md),
[`SECURITY_AUDIT`](ep-plugin/EP_PLUGIN_EXPORT_SECURITY_AUDIT.md),
[`TEST_PLAN`](ep-plugin/EP_PLUGIN_EXPORT_TEST_PLAN.md),
[`PR`](ep-plugin/EP_PLUGIN_EXPORT_PR.md) companions.

### `genai/`
Generation-side design: [`SCHEDULING.md`](genai/SCHEDULING.md),
[`PIPELINE.md`](genai/PIPELINE.md),
[`MODEL_METADATA.md`](genai/MODEL_METADATA.md),
[`MODEL_PACKAGE.md`](genai/MODEL_PACKAGE.md),
[`INFERENCE_METADATA_DECISIONS.md`](genai/INFERENCE_METADATA_DECISIONS.md)
(normative metadata specification and complete built-in capability catalogue),
[`ENCODER_BATCHING.md`](genai/ENCODER_BATCHING.md)
(proposed generic component batching: `batch_capacity`, `pad_mask`, packed
item ownership, phasing and acceptance matrix),
[`WORKFLOW_POLICY_COMPONENTS.md`](WORKFLOW_POLICY_COMPONENTS.md)
(policy components from first principles and the producer contract),
[`MOBIUS_WORKFLOW_PRODUCER.md`](genai/MOBIUS_WORKFLOW_PRODUCER.md),
[`NATIVE_BATCH_DECODE_2B_IMPL_SCOPING.md`](genai/NATIVE_BATCH_DECODE_2B_IMPL_SCOPING.md),
[`DIFFUSION.md`](genai/DIFFUSION.md),
[`COMFYUI_IMPORT.md`](genai/COMFYUI_IMPORT.md),
[`CHAINED_SPECULATIVE_EVIDENCE.md`](genai/CHAINED_SPECULATIVE_EVIDENCE.md)
(what a published chained-speculative package has to prove, and the recorded
H200 run that proves it).

### `quantization/`
Quantized formats and MoE: [`SUB4BIT_QUANT.md`](quantization/SUB4BIT_QUANT.md),
[`EXTENSIBLE_QUANT_TYPES.md`](quantization/EXTENSIBLE_QUANT_TYPES.md),
[`BLOCKQUANTIZEDMOE_DESIGN.md`](quantization/BLOCKQUANTIZEDMOE_DESIGN.md),
[`CPU_QUANT_HALF_GEMM_SCOPING.md`](quantization/CPU_QUANT_HALF_GEMM_SCOPING.md),
[`MOE_SUPPORT.md`](quantization/MOE_SUPPORT.md),
[`MOE_EXPERT_PARALLELISM.md`](quantization/MOE_EXPERT_PARALLELISM.md),
[`PROJECTION_FUSION.md`](quantization/PROJECTION_FUSION.md).

### `models/`
Per-model enablement status: [`DEEPSEEK_CSA_MTP_RUNTIME.md`](models/DEEPSEEK_CSA_MTP_RUNTIME.md),
[`deepseek-native-status-2026-07-25.md`](models/deepseek-native-status-2026-07-25.md),
[`glm-deepseek-enablement.md`](models/glm-deepseek-enablement.md),
[`glm-native-status-2026-07-25.md`](models/glm-native-status-2026-07-25.md),
[`GLM_READINESS_GAPS.md`](models/GLM_READINESS_GAPS.md),
[`KIMI_K_READINESS.md`](models/KIMI_K_READINESS.md).

### `performance/`
Kernel and decode performance investigations:
[`KERNEL_PERF.md`](performance/KERNEL_PERF.md),
[`DECODE_PERF_INVESTIGATION.md`](performance/DECODE_PERF_INVESTIGATION.md),
[`MLX_DECODE_PERF.md`](performance/MLX_DECODE_PERF.md),
[`GEMM_BACKEND_COMPARISON.md`](performance/GEMM_BACKEND_COMPARISON.md),
[`BENCH_MLAS_INT4_E2E.md`](performance/BENCH_MLAS_INT4_E2E.md),
[`MLAS_SYS_SPIKE.md`](performance/MLAS_SYS_SPIKE.md),
[`numa-decode-plan.md`](performance/numa-decode-plan.md).

### `distributed/`
Multi-device and collectives: [`DISTRIBUTED_RUNTIME.md`](distributed/DISTRIBUTED_RUNTIME.md),
[`COLLECTIVE_ORDERING_IMPL.md`](distributed/COLLECTIVE_ORDERING_IMPL.md),
[`COMMUNICATOR_BUFFER_IMPL.md`](distributed/COMMUNICATOR_BUFFER_IMPL.md).

### `status/`
Project state and upstream tracking: [`DECISIONS_FOR_JUSTIN.md`](status/DECISIONS_FOR_JUSTIN.md),
[`UPSTREAM_ORT_ARM_INVENTORY.md`](status/UPSTREAM_ORT_ARM_INVENTORY.md),
[`UPSTREAM_ORT_MATMULNBITS_INVENTORY.md`](status/UPSTREAM_ORT_MATMULNBITS_INVENTORY.md).
Dated test-health snapshots (which suites are green/red/ignored on `main`, and
why): [`2026-08-19-test-health-baseline.md`](status/2026-08-19-test-health-baseline.md)
— **check the date; a baseline decays.**

### `benchmarks/`, `portability/`, `research/`
Dated benchmark runs, including
[`2026-08-21-mobius-workflow-conformance.md`](benchmarks/2026-08-21-mobius-workflow-conformance.md),
plus portability notes and research write-ups.

## Two standing rules

1. **A number without its conditions is not a result.** State the model, the
   hardware, whether the run was solo, and what was held fixed. Wall-clock on a
   WDDM consumer GPU has been observed to range 3.9–28 tok/s across *identical*
   configurations, so a single unqualified figure is not evidence.
2. **When a document and a measurement disagree, the measurement wins and the
   document gets fixed in the same change.** `MEMORY_ARCHITECTURE.md` has been
   wrong at least three times; being the authoritative document does not exempt
   it from verification.

See [`.github/skills/measurement-discipline/SKILL.md`](../.github/skills/measurement-discipline/SKILL.md)
for the failure modes behind these rules, and
[`.github/skills/cuda-perf-measurement/SKILL.md`](../.github/skills/cuda-perf-measurement/SKILL.md)
for which instrument to reach for on the CUDA backend — including the three
device-specific traps (nsys hiding CUDA-graph internals, load cost read as
per-token cost, wall clock that cannot resolve 10%) that have each produced a
confidently backwards answer here.
