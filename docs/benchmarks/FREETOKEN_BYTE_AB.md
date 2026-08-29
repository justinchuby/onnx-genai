# FreeToken residency byte A/B

`freetoken_byte_ab` is the deterministic native-CUDA OFF/ON harness for issue
#1759. It compares process-local byte counters first. Wall clock is optional
corroboration; it is never the evidence for a movement claim.

## Build and marker proof

Build both binaries from the same checkout:

```bash
cargo build --release -p onnx-genai-bench --features native-cuda \
  --bin profile_native --bin freetoken_byte_ab

strings -a target/release/profile_native |
  grep -F ONNX_GENAI_FREETOKEN_BYTE_AB_NATIVE_CUDA_V2_C19E4B7A
```

The runner performs the marker check itself and rejects a binary without it.
Every child is invoked with `--ep cuda --backend native`; an `ort-cuda` report
is rejected by the run contract even when both Cargo features were linked into
one binary.

## Paired run

Use explicit prompt IDs when publishing evidence:

```bash
target/release/freetoken_byte_ab \
  --profile-native target/release/profile_native \
  --model /path/to/native-model-directory \
  --prompt-ids target/freetoken-byte-ab/prompt-ids.json \
  --tokens 128 --decode-skip 8 \
  --device 0 --trials 3 --warmup-seconds 8 \
  --device-budget-bytes <bytes> \
  --host-budget-bytes <bytes> \
  --max-throughput-drift-percent 10 \
  --policy-env ONNX_GENAI_WEIGHT_OFFLOAD_COARSE_RESIDENCY_ENABLE \
  --off-value 0 --on-value 1 \
  --output target/freetoken-byte-ab/report.json
```

The runner always enables the existing weight-offload path in both arms. Only
the named policy control differs. It launches a fresh process for each arm so
process-frozen configuration and cumulative counters cannot leak across the
comparison. Pairs run OFF then ON. Before every child, `nvidia-smi` records the
selected physical device, utilization, clock, power, and compute processes. The
runner waits up to 60 seconds for an exclusively idle probe so utilization
sampling left over from the preceding arm cannot make the next arm
automatically ineligible; persistent activity still fails the idle gate.
After the paired sweep, the runner repeats the first OFF arm/shape. The report
records both throughputs and their absolute drift; the contract fails when
either endpoint was not exclusively idle, the measurement is unavailable, or
drift exceeds the configured threshold.

Wall clock is included in the aggregate only when every pre-run probe identified
an exclusively idle NVIDIA A100, every arm actually warmed for at least 8
seconds, and at least three OFF/ON pairs ran. Otherwise throughput is omitted
while deterministic counters and correctness gates remain usable.

## Required contract

The combined report fails unless:

- model path, prompt token IDs, and requested token count match;
- generated token IDs are byte-identical within every pair and across trials;
- the native-CUDA marker and backend are exact;
- OFF reports the production lifecycle as `GateDisabled`, while ON reports
  `Installed` and at least one successfully reconciled measured boundary;
- the explicit ON setup moves every bindable expert range to host physical
  backing after warm-up, then measured production routes prove a real
  CPU/host-backed miss, completed H2D page-in, and later device hit;
- selected expert bytes equal GPU-hit plus CPU/host-served bytes, and completed
  expert H2D bytes equal CPU/host-served miss bytes;
- prefill/decode per-layer expert totals close exactly against the run totals;
- expert device plus host committed bytes are nonzero and identical across
  OFF/ON, with zero expert underflow, oversubscription, or unaccounted bytes;
- CUDA graph captures are greater than zero and fallbacks are zero;
- peak committed physical bytes do not exceed the managed limit;
- oversubscribed bytes, reference underflows, byte underflows, and
  unaccounted committed bytes are zero.

The run/pair contracts lock token IDs, lifecycle, accounting closure, and
capture safety. The optional tiny-QMoE integration test uses a real
`Engine::generate` session in both OFF and ON processes; it does not call the
residency cache directly. Point it at an external-data, VMM-granule-padded
native QMoE fixture:

```bash
CUDA_VISIBLE_DEVICES=0 \
FREETOKEN_TINY_QMOE_NATIVE_CUDA_DIR=/path/to/tiny-native-qmoe \
cargo test -p onnx-genai-bench \
  --features native-cuda --test freetoken_tiny_qmoe_native_cuda -- --nocapture
```

## Metric schema and accounting boundaries

Run records use `onnx-genai.freetoken-byte-ab.run.v1`; the aggregate uses
`onnx-genai.freetoken-byte-ab.comparison.v1`. Every metric carries `unit`,
`accounting_boundary`, and either `value` or `unavailable_reason`.

| Metric | Boundary |
| --- | --- |
| `route_residency_*` | CUDA-EP production installation and completed request-boundary lifecycle. OFF remains `GateDisabled`; ON must install and every measured boundary must apply and reconcile. |
| `selected_expert_logical_bytes` | Canonical logical bytes in the actual routed-expert union for completed production kernel windows. |
| `gpu_resident_expert_hit_bytes` | Selected bytes backed by device physical memory for the completed kernel window. |
| `cpu_served_expert_bytes` | Selected bytes backed by CPU/host NUMA physical memory for the completed kernel window. |
| `host_to_device_expert_page_in_bytes` | Published only after the content-preserving host-to-device transition completed. Async enqueue is never completion. |
| `prefill_expert_bytes_by_layer`, `decode_expert_bytes_by_layer` | Shape-derived phase and graph-node attribution; totals must close exactly. |
| `expert_*_committed_bytes`, expert safety counters | Tracked expert physical tier at the completion boundary and exact transition-scoped VMM accounting deltas. |
| `model_weight_layout_bytes` | Memory-planner storage sum at load. Analytical layout input, **not** measured HBM traffic. |
| `weight_residency_budget_bytes` | Live CUDA weight-residency budget; every child must equal the explicit `--device-budget-bytes` control. |
| `weight_gpu_resident_hit_bytes` | Existing `GLOBAL_HIT_BYTES`, reset after warm-up; all lazy weights, not expert-only. |
| `weight_h2d_accounted_bytes` | Existing `GLOBAL_HTOD_BYTES`, reset after warm-up. Synchronous/on-demand paths account after completion; async prefetch accounts after enqueue even if later discarded. Canonical payload bytes, not physical bus transactions. |
| `weight_zero_copy_host_read_bytes` | Existing host-mapped in-place read bytes. Not an H2D copy. |
| `weight_page_ins`, `weight_cache_hits` | Existing process-local decisions in the complete measured-generation window. |
| `weight_vram_byte_hit_rate` | `hit / (hit + H2D + zero-copy)` so host-mapped reads remain misses. |
| `weight_*_bytes_per_emitted_token` | Complete-generation aggregate (prefill plus decode) divided by emitted tokens. |
| graph counters | Session lifetime and separately the post-warm-up measurement delta. |
| physical-memory safety counters | The engine VMM arena/governor authority while the engine is alive after generation. |
| wall clock | Host token-callback intervals only; corroborative, never CUDA copy duration. The first OFF arm/shape is remeasured and drift-gated. |

The global `weight_*` counters still mix dense and expert weights. Expert
movement claims therefore use only the production route-boundary fields above;
logical layout bytes and global lazy-weight H2D totals are not relabeled as
physical expert traffic.

## Official-checkpoint command shapes

The prior storage audits are analytical inputs, not benchmark results:

- GLM-5.2 `UD-IQ1_S`: `216,715,360,960` bytes (216.72 GB / 201.83 GiB).
- DeepSeek-V4-Flash source checkpoint: approximately 159.6 GB.

After the exporter/ABI and runtime residency dependencies land, use the same
binary and prompt-ID file in both arms:

```bash
# GLM-5.2 — command shape only
target/release/freetoken_byte_ab \
  --profile-native target/release/profile_native \
  --model /path/to/converted-glm-5.2-native \
  --prompt-ids target/freetoken-byte-ab/glm52-prompt-ids.json \
  --tokens 128 --decode-skip 8 --device <idle-a100> \
  --trials 3 --warmup-seconds 8 --device-budget-bytes <bytes> \
  --output target/freetoken-byte-ab/glm52.json

# DeepSeek-V4 — command shape only
target/release/freetoken_byte_ab \
  --profile-native target/release/profile_native \
  --model /path/to/converted-deepseek-v4-flash-native \
  --prompt-ids target/freetoken-byte-ab/deepseek-v4-prompt-ids.json \
  --tokens 128 --decode-skip 8 --device <idle-a100> \
  --trials 3 --warmup-seconds 8 --device-budget-bytes <bytes> \
  --output target/freetoken-byte-ab/deepseek-v4.json
```

A full-size comparison still requires a converted model whose expert banks are
external-data backed and padded/aligned to the production VMM granule. If that
model, an idle GPU, an installed production binding, graph capture, or any
accounting authority is unavailable, the harness fails closed and records the
exact missing proof rather than estimating it.
