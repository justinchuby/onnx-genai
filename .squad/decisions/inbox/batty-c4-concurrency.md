# Batty c4 concurrency admission — 2026-07-28

## Summary

Sebastian's Layer A c4 gap was serving admission, not engine kernel speed. The server driver was already draining multiple queued commands into its `deferred` FIFO before selecting the first request, but the batch-forming code only looked at the live receiver. Sibling non-session `Generate` commands already in `deferred` therefore missed the initial continuous-batch manager and ran as later solo waves.

## Change

- `crates/onnx-genai-server/src/driver.rs`: batch formation now pulls contiguous non-session generation siblings from `deferred` before polling the receiver.
- The continuous-batch loop also backfills manager capacity from `deferred` before polling new receiver arrivals.
- Admission remains deadline-aware: c1 keeps the solo fast path after the 2 ms shallow-queue window; known in-flight siblings can extend collection to 12 ms, and any multi-request/pending-sibling batch keeps full `max_batch` capacity for late joins.

## Benchmark

Host/model: Snapdragon X Elite, Qwen3-0.6B CPU package at `C:\Users\justinchu\.foundry\cache\models\Microsoft\qwen3-0.6b-generic-cpu-4\v4`, `ONNX_GENAI_INTRA_OP_THREADS=6`, OpenAI streaming chat harness, 1 warmup + measured runs, 127 non-empty chunks/request.

| State | c1 aggregate tok/s | c1 median TTFT ms | c4 aggregate tok/s | c4 median TTFT ms | Notes |
|---|---:|---:|---:|---:|---|
| Before (existing binary, host load variable) | 87.4 | 56.5 | 63.8 | 5452.8 | Reproduced serialization; c4 runs varied 25.6–84.7 tok/s under load. Sebastian's cleaner before was 96.4 / 94.9 tok/s. |
| After (patched validation worktree) | 96.5 | 69.4 | 156.3 | 1188.9 | Clean final window, c4 measured 154.4 / 156.3 / 156.7 tok/s. |

The final c4 median beats Sebastian's Foundry Local c4 reference (146.4 tok/s). c1 aggregate remains at the target band and the solo fast path is still used when no sibling is queued.

## Residual gap / follow-up

`BatchedSharedBufferDecodeSession::step_active` still runs the shared-buffer batch machinery in the ORT crate. This task intentionally did not edit `crates/onnx-genai-ort/src/session/*` because Luba owns concurrent QNN work there. If future c2/c3 or tail latency remains a concern, add shared-buffer active-row compaction or a lean active-prefix path in the ORT batched shared-buffer runner.
