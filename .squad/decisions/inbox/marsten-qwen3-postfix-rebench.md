### 2026-07-24: Qwen3 postfix capture rebenchmark
**By:** Marsten
**What:** Replaced the stale Qwen3-0.6B rows in the native-vs-ORT ladder with a post-`ea452be0` remeasurement using the exact metadata-fixed model directory `/home/justinchu/mobius/.scratch/qwen3-0.6b-int4-cuda-postfix`.
**Why:** The multi-group Q/K RMSNorm capture change reduces decode seams from 29 to 1, invalidating the previous Qwen3 native/ORT ratios.

- Source: `ea452be0740e0a9894fb6df29245e2bbe453d15e`; documentation commit `ec0ef0f9b56706a62346faa2340614e8abc55fca` on branch `docs/qwen3-postfix-capture-rebench`.
- Harness: release `profile_native`, features `mlas,bench-ort,bench-native,cuda`; H200 GPU 3, `CUDA_VISIBLE_DEVICES=3`, `taskset -c 1`, `--ep cuda`, prompt `The capital of France is`, one warmup, `--steady --decode-skip 8`, three separate one-run processes/backend.
- 128 raw tok/s: native 530.51 / 533.44 / 530.68 (median **530.68**); ORT 451.84 / 443.54 / 438.01 (median **443.54**); median ratio **1.197x**.
- 1024 raw tok/s: native 479.42 / 480.29 / 478.96 (median **479.42**); ORT 388.41 / 384.05 / 380.17 (median **384.05**); median ratio **1.248x**.
- Host-load caveat: `uptime` was 16.63/10.41/8.66 immediately before the 128 set and 11.79/9.85/8.53 after; the recorded 1024 set was 6.93/8.61/8.25 before and 16.77/10.98/9.07 after. GPU 3 was allocation-free (4 MiB baseline) and 0–2% utilization in snapshots, but the CPU-loaded shared host prevents an uncontended absolute claim.
- Sanity: native and ORT both emitted the same coherent Paris/Rome continuation in the 128-token run.
