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
  grep -F ONNX_GENAI_FREETOKEN_BYTE_AB_NATIVE_CUDA_V1_7F31A9D2
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
  --policy-env ONNX_GENAI_WEIGHT_OFFLOAD_COARSE_RESIDENCY_ENABLE \
  --off-value 0 --on-value 1 \
  --output target/freetoken-byte-ab/report.json
```

The runner always enables the existing weight-offload path in both arms. Only
the named policy control differs. It launches a fresh process for each arm so
process-frozen configuration and cumulative counters cannot leak across the
comparison. Pairs run OFF then ON. Before every child, `nvidia-smi` records the
selected physical device, utilization, clock, power, and compute processes.

Wall clock is included in the aggregate only when every pre-run probe identified
an exclusively idle NVIDIA A100, every arm actually warmed for at least 8
seconds, and at least three OFF/ON pairs ran. Otherwise throughput is omitted
while deterministic counters and correctness gates remain usable.

## Required contract

The combined report fails unless:

- model path, prompt token IDs, and requested token count match;
- generated token IDs are byte-identical within every pair and across trials;
- the native-CUDA marker and backend are exact;
- CUDA graph captures are greater than zero and fallbacks are zero;
- peak committed physical bytes do not exceed the managed limit;
- oversubscribed bytes, reference underflows, byte underflows, and
  unaccounted committed bytes are zero.

The run/pair contracts lock token IDs and capture safety. Separately, the tiny
QMoE authority test passes two deterministic expert-major
`LazyWeightBoundary::QMoe` banks through the real `CudaWeightResidency`
authority. It verifies copied bytes and constructs a cold page-in, a resident
hit, and an evicting second miss with exact counters. This authority-level slice
does not claim that current main routes production QMoE banks through residency:

```bash
CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-bench \
  --features native-cuda --test freetoken_tiny_qmoe_native_cuda -- --nocapture
```

## Metric schema and accounting boundaries

Run records use `onnx-genai.freetoken-byte-ab.run.v1`; the aggregate uses
`onnx-genai.freetoken-byte-ab.comparison.v1`. Every metric carries `unit`,
`accounting_boundary`, and either `value` or `unavailable_reason`.

| Metric | Boundary |
| --- | --- |
| `model_weight_layout_bytes` | Memory-planner storage sum at load. Analytical layout input, **not** measured HBM traffic. |
| `weight_gpu_resident_hit_bytes` | Existing `GLOBAL_HIT_BYTES`, reset after warm-up; all lazy weights, not expert-only. |
| `weight_h2d_accounted_bytes` | Existing `GLOBAL_HTOD_BYTES`, reset after warm-up. Synchronous/on-demand paths account after completion; async prefetch accounts after enqueue even if later discarded. Canonical payload bytes, not physical bus transactions. |
| `weight_zero_copy_host_read_bytes` | Existing host-mapped in-place read bytes. Not an H2D copy. |
| `weight_page_ins`, `weight_cache_hits` | Existing process-local decisions in the complete measured-generation window. |
| `weight_vram_byte_hit_rate` | `hit / (hit + H2D + zero-copy)` so host-mapped reads remain misses. |
| `weight_*_bytes_per_emitted_token` | Complete-generation aggregate (prefill plus decode) divided by emitted tokens. |
| graph counters | Session lifetime and separately the post-warm-up measurement delta. |
| physical-memory safety counters | The engine VMM arena/governor authority while the engine is alive after generation. |
| wall clock | Host token-callback intervals only; corroborative, never CUDA copy duration. |

Current `origin/main` cannot truthfully populate the expert-specific fields:

- selected expert logical bytes;
- GPU-resident expert hit bytes;
- expert H2D page-in bytes and page-ins;
- CPU-served expert bytes;
- expert byte-hit-rate / bytes per token;
- prefill/decode and per-layer expert attribution.

Those fields are emitted as `null` with the exact reason. The available global
weight counters mix dense and expert weights, so their OFF/ON delta must not be
labeled measured expert traffic. No logical selected-byte count or checkpoint
layout bound is labeled physical HBM movement.

## Official-checkpoint command shapes (do not run yet)

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

A real full-size comparison remains blocked on the production route producer
and readiness lifecycle (#2082), exact per-bank VMM reservations (#2163), and
the model-specific export/state dependencies (including #2063/#2194 for
DeepSeek-V4). On current main the default coarse-residency environment control
has no production caller, so an ON label alone is not evidence that the policy
ran. The harness intentionally reports the missing expert attribution rather
than estimating it.
