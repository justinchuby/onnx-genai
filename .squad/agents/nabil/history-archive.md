# Nabil — History Archive

## Archived 2026-07-29 (full pre-compaction snapshot)

# Nabil — History

## 2026-07-12: Joined
Hired to lead the ORT plugin-EP integration for a new **Apple Metal/MPS execution provider** for ONNX Runtime (repo `../onnxruntime-mps`). Motivation: onnx-genai is ORT-kernel-bound on Apple Silicon (ORT's generic int4 CPU/WebGPU kernels lag llama.cpp's hand-tuned Metal); a custom MPS EP with hand-tuned kernels can beat everyone on Mac. The EP must support all ops onnx-genai/Mobius use: MatMulNBits (int4), GroupQueryAttention, GatherBlockQuantized, RoPE, RMSNorm, softmax, elementwise. Tested end-to-end by the onnx-genai runtime (`ONNX_GENAI_EP` selects it). Reference kernels: ExecuTorch + PyTorch MPS backends.

### 2026-07-16T00:00:00Z — Performance-and-design wave
Authored the ORT-schema-based model-package design document.

### 2026-07-16T00:00:03Z — Projection-fusion design recorded
Authored `docs/PROJECTION_FUSION.md` for conservative load-time gate/up MatMulNBits fusion. Fact Checker confirmed QKV is already packed, gate/up is the available `4864|4864→9728` target, and qualified the roughly 125 MiB payload as a lower-bound memory cost. The design is awaiting user approval and is not implemented.

### 2026-07-16T00:00:00Z — Native CUDA decode design
Authored `docs/NATIVE_CUDA_DECODE.md` (`b416b7f`) and applied Fact Checker's stream/graph-ownership corrections (`33beb8d`). The fact-checked five-milestone `Arc<dyn ExecutionProvider>` design awaits user greenlight; implementation has not started.

## 2026-07-16T17:00:38+0000 — Weight offload design
- Authored `docs/WEIGHT_OFFLOAD.md` (`f0d0890`): immutable mmap backing feeds bounded host and VRAM caches through weight-specific expert/page leases.
- The design awaits user greenlight; no implementation has started.

## 2026-07-16T19-27-57+0000 — Scribe session update

- Authored `docs/DEEPSEEK_CSA_MTP_RUNTIME.md` (`bca068c`), a native CSA/index-op and persistent iterative-MTP sidecar-state design. It awaits user greenlight.

## 2026-07-14T00:00:00Z — QMoE final approval

- Rejected the initial and first hardening revisions, then approved the final QMoE kernel once overflow checks, allocation addressability, and odd affine-int4 block handling were correct.

## 2026-07-17T02:24:32Z — QMoE int1/int2 review

- 🟢 Cleared `cdb4ee5`: factory gating, packing, zero-point tails, sizing, and existing hardening are correct; full crate suite passed (450 passed, 1 ignored).

## 2026-07-18T04-55-00Z — Scribe session update

- On lockout reassignment, fixed CUDA standard Attention claim validation (`8eb23f1`) so `Undefined` optional mask/past/nonpad slots mean absent while supplied tensors retain strict type and compatibility checks.

## 2026-07-27T18:20:00-07:00 — MLX EP logging framework

- Replaced all 12 `eprintln!`/`eprint!` sites in `onnxruntime-mlx/rust/` with the `log` crate facade + minimal in-crate stderr logger.
- Chose `log` over `tracing` because the plugin is a cdylib with its own statics; the subscriber model has no benefit and adds weight.
- Default: **Warn** only (panics + capture failures). Info via `VERBOSE=1`, Debug via `TRACE=<path>`.
- Verified: build clean (`-D warnings`), 1106 tests pass, stderr is empty by default.
- PR: https://github.com/justinchuby/onnxruntime-mlx/pull/9 (not merged)
- Decision: `.squad/decisions/inbox/nabil-mlx-logging.md`

## 2026-07-28T04-08-08+0000 — Wave 2 regression/roadmap update
- MLX logging decision note was merged into decisions for future backend logging work.
