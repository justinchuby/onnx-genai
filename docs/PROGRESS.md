# onnx-genai — Implementation Progress

Tracks implementation status of `docs/DESIGN.md` (§1–§40). Updated as work lands.

**Published:** `onnx-genai` v0.1.0 + 8 sub-crates on crates.io. CI (fmt/build/test/**blocking clippy**) + scheduled `cargo-audit`. Coverage ~77% line.

_Last updated: 2026-07-13_

## Status by design section

| § | Feature | Status | Notes |
|---|---------|--------|-------|
| 1–8 | Vision, architecture, core components, data flow, concurrency, model dir, crates, deps | ✅ Done | |
| 9 | API surface | 🟡 Partial | chat/completions/models/sessions/status/metrics/audio/embeddings(#7 scaffold)/logprobs(#8) ✅; `/v1/debug/*` (config/sessions/kv/trace) ✅; Perfetto/OTLP export (#13) deferred |
| 11,12,15 | Testing, design decisions | ✅ Done | coverage ~77% |
| 16 | Quantized models | ✅ Done | EP select + int8 KV; fp8 KV = #15 |
| 17 | Diffusion pipeline (image) | ❌ Missing | #16 |
| 18,19 | ORT wrapper, dep graph | ✅ Done | |
| 20 | Generalized pipeline | 🟡 Partial | AR/composite/single-pass/vision/audio ✅; iterative diffusion pending |
| 21 | Tool use / function calling | ✅ Done | Hermes-verified E2E |
| 22 | Grammar constrained decoding | ✅ Done | llguidance JSON-schema/regex/lark |
| 23 | FIM / infilling | ✅ Done | engine + `POST /v1/completions` |
| 24 | Sampling policy | ✅ Done | full sampler suite; **real RNG fixed 2026-07-12** |
| 25 | Extensibility | ✅ Done | DecodeBackend/SpeculativeProposer/Sampler traits |
| 26 | Multi-agent serving | ✅ Done | batched continuous serving (~6× throughput) |
| 27 | Multi-token speculative | ✅ Done | draft + prompt-lookup + MTP + EAGLE-3 |
| 28 | vLLM speculator compat | ✅ Done | config auto-discovery + EAGLE-3 proposer |
| 29 | Language diffusion | ❌ Missing | large |
| 31 | Observability | 🟡 Partial | `/metrics` + `/v1/status` + trace ids + `/v1/debug/*` ✅; Perfetto/OTLP export (feature-gated stub) = #13 |
| 32 | Metrics API | ✅ Done | |
| 34 | Cluster/session router | ❌ Missing | |
| 35 | Native preprocessing | ✅ Done | `onnx-genai-preprocess`: image (bicubic/CLIP + tiling none/fixed_grid/dynamic_anyres) + audio log-mel; audio wired (#12). Multi-tile prompt token-expansion = documented follow-up |
| 36 | Backpressure/lifecycle | ✅ Done | admission cap + 429 ✅; configurable `max_queue_depth` (CLI/env) + 429 Retry-After ✅ |
| 37 | Model lifecycle mgmt | ❌ Missing | single model at startup; #9 |
| 38 | Distributed KV connector | ❌ Missing | local tiered KV only |
| 39 | Paged/radix attention | 🟡 Upstream | Mobius block-table KV graph (Option C, std ops) = draft PR onnxruntime/mobius#395; runtime wiring pending |
| 40 | Sliding window attention | ❌ Missing | new design section; long context on limited HW — not yet implemented |

## Open backlog (GitHub issues)

- **#7** `/v1/embeddings` · **#8** logprobs · **#9** model lifecycle/multi-model · **#13** debug endpoints + Perfetto · **#15** fp8 KV quant · **#16** image diffusion.
- Closed: **#2** server split · **#3** decode ownership · **#4** FIM endpoint · **#5** benchmarks · **#10** EAGLE-3 proposer · **#11** audio log-mel preprocessing · **#12** audio input · **#14** vision preprocessing/tiling.

## Recently completed (this session)

Complete runtime built from scaffold + published: generation (greedy/speculative draft+prompt-lookup+MTP), samplers, FIM, grammar, tool use (Hermes-verified), chat templates, multi-session + prefix cache, paged/tiered/int8 KV, long-context O(1)/token static-cache, batched multi-agent serving, OpenAI HTTP (chat/completions/vision/streaming/sessions), observability, benchmarks (`onnx-genai-bench`), `onnx-genai-preprocess` crate, security hardening, CI + audits. **Fixed: categorical sampling had no RNG (always token 0).**

## Notable design changes / decisions to record

- Preprocessing lives in its own crate `onnx-genai-preprocess` (§35).
- Real-model exact-equality tests use `intra_op_threads=1` (ORT FP determinism).
- Paged/radix attention (§39.4 Option C): Mobius now grows block-table KV via standard ONNX ops (ScatterND + Gather + opset-24 Attention) — draft PR onnxruntime/mobius#395. Same op path supports vLLM PagedAttention AND SGLang RadixAttention (share physical pages via block_table). Runtime-side wiring to consume paged KV is the next step once the PR lands.
- Audio & vision quality gated on real Mobius model packages (Whisper / CLIP+decoder).
- **Benchmarking (§ new):** `onnx-genai-bench` cross-runtime harness (`compare.rs` / `scripts/compare_runtimes.sh`) measures TTFT + decode tok/s vs Ollama (llama.cpp) + LM Studio over the OpenAI API. Goal: beat llama.cpp + LM Studio (Metal). Runs recorded under `docs/benchmarks/`.
  - **Fix (GQA):** Mobius `--ep webgpu` emits `com.microsoft::GroupQueryAttention` — WebGPU Qwen2.5-0.5B now **24 GQA / 0 Attention, 268 WebGPU / 6 CPU, 1 H2D / 0 D2H** (transfers eliminated).
  - **Q4 correctness FIXED** (mobius PR #396): two GGUF→ONNX bugs (missing Qwen2 QKV biases + wrong NEOX reverse-permute) — garbage → coherent.
  - **fp16 GQA KV consumed** by runtime; **runtime now owns the KV cache via our own `InferenceMetadata`** (`inference_metadata.yaml`), NOT ORT-GenAI `genai_config.json` (deleted). GQA op = on-device attention compute only. Mobius emits our config via `--runtime onnx-genai` (PR #398).
  - **In progress:** Q4+GQA WebGPU model (quantized weights + on-device attention) — the fair GPU comparison vs LM Studio Metal.
  - **Q4+GQA WebGPU = 30.5 tok/s** (168 MatMulNBits + 24 GQA); quantized embedding via `GatherBlockQuantized` (272MB→76MB, mobius PR #400); Q4_K_M support (PR #399).
  - **Device-resident KV blocked by ORT 1.27 WebGPU EP** — binding a user-preallocated device tensor as an in-place GQA share-buffer SIGSEGVs on long gens; gated behind `ONNX_GENAI_DEVICE_KV=1`. Safe default (`validationMode=disabled`) ships → **~49.6 tok/s** (no regression). Plumbing ready for when ORT fixes it — and for **CUDA** (mature IoBinding + `enable_cuda_graph`), the likely path to close the gap on H200.
  - **🎯 ROOT CAUSE of the CPU gap found (accuracy_level):** MatMulNBits was emitted with NO `accuracy_level`, so ORT's MLAS ran the slow fp32 dequant+GEMM path instead of the int8 dot-product path (ARM SDOT / x86 VNNI) llama.cpp uses. Fix (mobius PR #402, `default_int4_accuracy_level=4` was dead config, now plumbed). Measured CPU decode: baseline 39.3 → **acc4 91.8 (2.33×)** → **quantized-head + acc4 = 194.7, EXCEEDS LM Studio CPU (157)**. Session was already persistent; graph-opt already ALL; orchestration ~1%. fp16/bf16 levels regress on M1.
  - **Definitive CPU (WARM, corrected 2026-07-13):** at parity-or-ahead of ALL runtimes. Warm decode: short ~212 (ours) ≈ LM Studio 156 / Foundry 212; **long ~175 (ours) LEADS** LM Studio 160 / Foundry 160; beat Ollama both. The earlier "Foundry Local leads 203/166" was a **thermal/under-warmed artifact** (this M1 Max swings 85–216 tok/s unwarmed) + pre-KV-fix. Decisive test: FL's *exact* model through OUR runtime = decode parity (the FL win doesn't reproduce). FL sets zero custom ORT session options — we already match; their fused-QKV (121 vs our 169 MatMulNBits) is decode-neutral (bandwidth-bound). Residual FL edge = ~2-4% prefill/TTFT only.
  - How CPU is achieved: `accuracy_level=4` (int8 MLAS path, 2.3×) + quantized head (#401) + quantized embedding (#400) + in-place O(1)/token GQA KV (fp32-GQA gate fix). Orchestration is ~1% (98.9% ort.session_run); session persistent, KV Arc-rotated.
  - **CUDA (H200):** `--features cuda`, device-resident KV + `enable_cuda_graph`. Runbook committed.
  - **🍎 MLX-native EP (`../onnxruntime-mlx`, renamed from onnxruntime-mps):** **Full pivot to MLX (2026-07-13).** After the hand-kernel EP reached decode 11.3→133 tok/s + prefill GEMM + a validated MLX-prefill/hand-decode hybrid, MLX was found **Pareto-dominant** (decode never slower + prefill 2.5–3× + coherent + memory-stable). Per user directive (全面改mlx / keep one efficient impl), **MLX is now the SOLE compute path** — the whole fused decoder subgraph (prefill AND decode) runs through MLX; **all hand-written `.metal` kernels + machinery were deleted**. Architecture: ONNX fused subgraph → MLX graph → single `mlx_eval` at the boundary → ORT outputs (MatMulNBits→`mlx_quantized_matmul`, GQA→`fast_scaled_dot_product_attention`+`fast_rope`, RMSNorm→`fast_rms_norm`, …). `mlx-c` is a **hard build dep** (mlx 0.32 / mlx-c 0.6). **EP name `MLXExecutionProvider`** (renamed from the interim `MetalEP`); onnx-genai binds by this exact name (`REGISTRATION_NAME` in session.rs) — only point `ONNX_GENAI_METAL_EP_LIB` at the new **`libonnxruntime_mlx_ep.dylib`**. Verified: build green, **ctest 3/3** (mlx_op_tests/mlx_leak_test/mlx_e2e), E2E coherent ("Paris"), memory flat (0.00 MiB/7 cycles), **prefill 2.2–3.5× vs CPU**, decode ~122–148 tok/s. MRR leak fix preserved. Follow-ups: bf16 end-to-end, GatherBlockQuantized asymmetric zero_points, larger-model decode edge.
  - _History (pre-pivot):_ custom Apple Metal ORT plugin EP, empty→working in-session. Skeleton→kernels (366/393 on GPU)→subgraph fusion (366→27)→GQA kernel (→1 subgraph)→uint4 bandwidth loads→**prefill GEMM**→MLX-prefill hybrid. Fixed critical MRR MTLBuffer leak (was crashing machine) + leak-regression test. Modular op-architecture designed; MLX evaluation (Phase-0) → full-swap NO-GO on the 1.25× decode bar, but the user overrode on the one-efficient-impl principle. Team: Nabil/Mariette/Coco/Freysa.
  - Config standard: `schemars` JSON Schema (Draft 2020-12). Mobius PRs #397-402 open (#399 rebased/mergeable), #395 draft.
