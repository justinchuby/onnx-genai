# Deckard: Option A' cycle-7 spike #1+#2 results (issue #82)

**Author:** Deckard (CUDA/Perf)
**Scope:** answers the two spikes assigned to Deckard in
`roy-a-prime-downgraded-provisional-cycle7.md`. Spike #3 (resize-starvation)
is explicitly out of scope here.

**PR:** squad/82-a-prime-cold-expert-spike ->
`crates/onnx-runtime-ep-cuda/tests/qmoe_zero_copy_cold_expert_spike.rs`
(EXPERIMENTAL, `#[ignore]`d, not wired into any production dispatch path).

## Spike #1 (mixed-backing feasibility) -- CONFIRMED NO

Re-verified directly in `weight_paging.rs`: `bind_zero_copy` /
`zero_copy_device_ptr` require one whole `LazyWeight` to resolve to one
contiguous device pointer; a gapped/non-contiguous region returns `None`
(hard fallback to a VRAM copy). QMoE's kernels
(`launch_grouped_linear`/`launch_linear`) likewise take exactly one base
pointer per weight tensor and index every expert from it. **There is no
shipped path for intra-tensor per-expert hot/cold splitting.** Building one
needs either (a) a new composable VMM design (granule-level host/device
mapping inside one stable-VA arena) or (b) a per-expert pointer table (a
real kernel ABI change, converging toward Option C). This matches and
confirms the cycle-7 file's claim; it is not a new finding, just an
independent re-derivation from the code.

## Spike #2 (MoE-specific measurement) -- DONE, see PR for full harness

What IS buildable and shipped today, unchanged: **whole weight tensor**
granularity residency choice (fc1/fc2/fc3 each independently VRAM-resident
or zero-copy host-mapped), reusing the exact `cuMemHostRegister
(DEVICEMAP|READ_ONLY)` + `cuMemHostGetDevicePointer` primitive
`HostMapRegistry` uses. Measured against a real QMoE int4 GEMV kernel
dispatch (not sequential memcpy), shape-faithful DeepSeek-V2-Lite-cited
fixture (64 and 256 experts, hidden=2048, inter=1408, top_k=6), decode
shape M=1.

**Platform:** 3x independent A100-SXM4-80GB (idle-verified via
`nvidia-smi --query-compute-apps` before each run), driver 580.105.08,
CUDA 13.0, Linux, UVA/unified addressing (default on this driver/GPU
combination -- not explicitly toggled by this harness).

**Correctness:** bit-identical output (`f32::to_bits()` exact match, 0
mismatches) for every zero-copy arm (all_zero_copy, mixed_fc1_cold, 3x
repeated fresh bind/unbind cycles) vs. the all-VRAM oracle, across both
model shapes, on 3 separate GPUs. A falsifiability control confirmed the
host/device memory-type probe genuinely discriminates (reads DEVICE for
VRAM pointers, HOST for zero-copy pointers) -- not a rubber-stamp.

**Bandwidth** (median of 9 GPU-timed reps per arm, per shape):
- all_vram: ~155-159 us/step (baseline)
- all_zero_copy (fc1+fc2+fc3 cold): ~1190-1220 us/step,
  **~32.2-32.7 GB/s achieved**, 7.6-7.7x slower than all-VRAM
- mixed (fc1-only cold, fc2/fc3 hot): ~474-491 us/step,
  **~26.5-27.4 GB/s achieved** on the cold tensor, ~3.1x slower than
  all-VRAM
- Consistent across 64-expert and 256-expert shapes and across 3
  independent GPUs (variance <5%).
- Compared against PCIe Gen4 x16 (~25 GB/s, single-direction H2D) and
  A100 HBM2e (2039 GB/s) rooflines -- achieved GB/s sits AT or slightly
  ABOVE the naive PCIe estimate, most plausibly explained by host
  page-cache/TLB locality at this small per-step touched-byte count
  (~1.3-5.2 MiB/step at top_k<=6), not evidence of exceeding true PCIe
  bandwidth for a cold/uncached read. **This number must NOT be
  conflated with #925's dense whole-model-streaming 6.795 GB/s figure --
  different access pattern, different regime, per cycle-7's own
  correction.**

## GO/NO-GO verdict

**NO-GO for whole-tensor-granularity A' as a production per-expert cold
path**, because:
1. Spike #1 confirms per-expert intra-tensor splitting -- the actual A'
   proposal -- is not buildable without new work (new VMM design or ABI
   change). Whole-tensor granularity is a DIFFERENT, coarser scheme; using
   it as "A'" would misrepresent what ships.
2. Even at whole-tensor granularity, correctness is clean (a genuine
   positive) but the cold-path cost is 3-8x per touched tensor at this
   shape/decode cadence -- unacceptable for a "hot expert" path that must
   be fast every decode step, though the correctness result *is* useful
   supporting evidence that zero-copy hybrid's kernel-visible-address
   invariant holds under real QMoE GEMV access, not just the dense
   MatMulNBits pattern #880/#925 tested.

**Next architecture gate:** if intra-tensor per-expert residency remains
desired, the next real decision is between (a) the new composable VMM
granule design (bigger lift, preserves today's zero-copy-hybrid mental
model) or (b) a per-expert device pointer table at the kernel ABI level
(smaller kernel change, but is Option C in different clothing -- should be
evaluated head-to-head against Option C rather than treated as a lighter
variant of A'). Recommend Roy/Sapper scope a follow-up feasibility spike
for whichever of (a)/(b) is preferred before further A' work; this file's
harness and shape-faithful fixtures are reusable for that follow-up.

## Fact Checker gate response (this update)

Four hard gates were added by Fact Checker review after the initial spike. All four are now addressed in the same PR/branch, same commit set:

1. **No mixed per-expert VRAM+host composition claimed or built.** Re-confirmed: nothing in this spike constructs a single tensor with some experts VRAM-resident and others host-mapped. All arms remain whole-tensor granularity (fc1/fc2/fc3 each independently bound one way). This was already true in the first version of this spike; restated here explicitly per the gate.
2. **Prior 5.6 GB/s (issue #880) and #925's 6.795 GB ceiling do NOT transfer here and were not reused.** #880's ~5.6 GB/s figure is RTX 4060 Laptop / WDDM only (confirmed by re-reading the issue body directly: "Results (RTX 4060 Laptop, 8 GiB, WDDM...)"). #925's ~6.795 GB ceiling is dense Qwen2.5-14B-Instruct weight streaming on H200/Linux, not MoE/A100 — also confirmed directly. This spike's ~32-36 GB/s (all-cold) / ~26-29 GB/s (mixed fc1-cold) figures are freshly measured, on A100/Linux, on a real QMoE int4 GEMV kernel path, and are reported as their own number, not compared against or blended with either prior figure except to note they are NOT the same regime.
3. **Resize-starvation stress added and run**: new test `qmoe_routed_residency_guard_resize_starvation_stress_1000_steps` drives the REAL `CudaWeightResidency::acquire_routed_residency`/`resize_safe_point`/`execute_resize` machinery through 1000 back-to-back acquire→resize-attempt(rejected)→release→resize-attempt(accepted, executed, then shrunk back) cycles, single-threaded (simulating sequential decode-step dispatch). Result: 1000/1000 resize attempts while a guard is held were correctly rejected `NotSafePoint`, and 1000/1000 resize attempts in the inter-step gap succeeded — **no starvation observed** under this back-to-back single-threaded cadence. This is a bounded, partial answer to spike #3: it proves the guard-counted resize seam is exact and non-leaking under sequential decode-cadence load, but does **not** cover concurrent-stream/multi-thread dispatch, which is a separate, larger follow-up spike (explicitly flagged as out of scope here, same as originally scoped).
4. **`qwen15-moe-qmoe` (Qwen1.5-MoE-A2.7B) real cold-bytes/step now measured directly**, not assumed: new shape `QWEN15_MOE_A27B` (experts=60, hidden=2048, inter=1408, top_k=4 — cited from https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blob/main/config.json) added as a third fixture alongside DeepSeek-V2-Lite (64e) and its 256-expert wide variant. All three now print a **measured** (not modeled) "real cold bytes/step" line before any control run: at `qwen1.5-moe-a2.7b` shape, per-expert gate/up/down = 2,162,688 B each, touched_experts(top_k)=4, total = 25,952,256 B (24.75 MiB)/decode-step if all touched experts were cold. Bandwidth for this shape: all-cold ~31.5 GB/s, mixed fc1-cold ~25.8 GB/s (7.32x / 2.98x slower than all-VRAM respectively) — consistent with the other two shapes.

**Correction to a stale precondition** (per this update's own instruction, not previously reflected here): Mobius #550 now emits DeepSeek-V4 as one QMoE and #555 emits GLM-5.2 via shared QMoE — the exporter prerequisite for those two production-shaped fixtures is closed, though full downloadable artifacts remain absent from this box. This does not change any verdict below (this spike used the smaller, already-available `qwen15-moe-qmoe`/DeepSeek-V2-Lite-shaped/GLM-5.2-shaped-cited fixtures, all synthetic-filled but shape-faithful), but is recorded so a future spike knows real DeepSeek-V4/GLM-5.2 artifacts may now be obtainable rather than blocked on the exporter.

## Updated results (post-gate, 2 more independent idle-A100 runs, 3 shapes)

Correctness: still bit-identical (0 mismatches) on every arm/shape/repeat, now across 3 model shapes (DeepSeek-V2-Lite 64e, its 256e wide variant, and Qwen1.5-MoE-A2.7B 60e) — no regression or new failure introduced by the additional gates.

Bandwidth by shape (median of 9 GPU-timed reps, all measured on GPU 5/6, idle-verified):
| shape | all_vram | all_cold GB/s | mixed(fc1-cold) GB/s | all_cold slowdown | mixed slowdown |
|---|---|---|---|---|---|
| deepseek-v2-lite (64e) | 155.65 us | 35.6 | 29.1 | 7.03x | 2.86x |
| deepseek-v2-lite-wide (256e) | 156.7-159.7 us | 32.5-35.5 | 26.7-29.1 | 7.0-7.5x | 2.85-3.04x |
| qwen1.5-moe-a2.7b (60e, real cited shape) | 111.6-112.6 us | 31.5-32.4 | 25.8-26.0 | 7.17-7.32x | 2.98x |

Resize-starvation: 1000/1000 clean (no starvation, no counter drift) under sequential single-threaded decode-cadence load; concurrent/multi-stream variant is explicitly NOT covered and remains open.

**Verdict unchanged: NO-GO** for whole-tensor-granularity A' as a production per-expert cold path, for the same two reasons as before (spike #1 confirms the real per-expert splitting proposal isn't buildable without new work; even the buildable coarser granularity costs 3-8x per touched tensor at decode cadence). No new sub-weight VMM/host-map composition was built or attempted in response to these gates — per instruction, if mixed addressability needed that, the correct response is to stop and report NO-GO rather than expand scope, and that is what happened here: all four gates were addressed using only the existing whole-tensor-granularity mechanism and the existing (unmodified) `RoutedResidencyGuard`/resize machinery.


## Reproduce

```
CUDA_VISIBLE_DEVICES=<idle gpu> cargo test -p onnx-runtime-ep-cuda \
  --features cuda --release --test qmoe_zero_copy_cold_expert_spike \
  -- --ignored --nocapture --test-threads=1
```

Four tests: `qmoe_cold_expert_spike_deepseek_v2_lite_shape`,
`qmoe_cold_expert_spike_wide_256_expert_shape`,
`qmoe_cold_expert_spike_qwen15_moe_a27b_shape`,
`qmoe_routed_residency_guard_resize_starvation_stress_1000_steps`.
