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

## Reproduce

```
CUDA_VISIBLE_DEVICES=<idle gpu> cargo test -p onnx-runtime-ep-cuda \
  --features cuda --release --test qmoe_zero_copy_cold_expert_spike \
  -- --ignored --nocapture --test-threads=1
```
