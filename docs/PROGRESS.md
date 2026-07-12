# onnx-genai — Implementation Progress

Tracks implementation status of `docs/DESIGN.md` (§1–§38). Updated as work lands.

**Published:** `onnx-genai` v0.1.0 + 8 sub-crates on crates.io. CI (fmt/build/test/**blocking clippy**) + scheduled `cargo-audit`. Coverage ~77% line.

_Last updated: 2026-07-12_

## Status by design section

| § | Feature | Status | Notes |
|---|---------|--------|-------|
| 1–8 | Vision, architecture, core components, data flow, concurrency, model dir, crates, deps | ✅ Done | |
| 9 | API surface | 🟡 Partial | chat/completions/models/sessions/status/metrics ✅; embeddings (#7), audio (#12), `/v1/debug/*` (#13) missing |
| 11,12,15 | Testing, design decisions | ✅ Done | coverage ~77% |
| 16 | Quantized models | ✅ Done | EP select + int8 KV; fp8 KV = #15 |
| 17 | Diffusion pipeline (image) | ❌ Missing | #16 |
| 18,19 | ORT wrapper, dep graph | ✅ Done | |
| 20 | Generalized pipeline | 🟡 Partial | AR/composite/single-pass/vision ✅; iterative/audio pending |
| 21 | Tool use / function calling | ✅ Done | Hermes-verified E2E |
| 22 | Grammar constrained decoding | ✅ Done | llguidance JSON-schema/regex/lark |
| 23 | FIM / infilling | ✅ Done | engine + `POST /v1/completions` |
| 24 | Sampling policy | ✅ Done | full sampler suite; **real RNG fixed 2026-07-12** |
| 25 | Extensibility | ✅ Done | DecodeBackend/SpeculativeProposer/Sampler traits |
| 26 | Multi-agent serving | ✅ Done | batched continuous serving (~6× throughput) |
| 27 | Multi-token speculative | 🟡 Partial | draft + prompt-lookup + MTP ✅; EAGLE-3 = #10 |
| 28 | vLLM speculator compat | 🟡 Partial | config auto-discovery ✅; EAGLE-3 proposer = #10 |
| 29 | Language diffusion | ❌ Missing | large |
| 31 | Observability | 🟡 Partial | `/metrics` + `/v1/status` + trace ids ✅; Perfetto/OTLP/debug = #13 |
| 32 | Metrics API | ✅ Done | |
| 34 | Cluster/session router | ❌ Missing | |
| 35 | Native preprocessing | 🟡 Partial | `onnx-genai-preprocess` crate: image (bicubic/CLIP) + audio log-mel ✅; audio wiring = #12 |
| 36 | Backpressure/lifecycle | 🟡 Partial | admission cap + 429 ✅; queue-depth config pending |
| 37 | Model lifecycle mgmt | ❌ Missing | single model at startup; #9 |
| 38 | Distributed KV connector | ❌ Missing | local tiered KV only |

## Open backlog (GitHub issues)

- **#7** `/v1/embeddings` · **#8** logprobs · **#9** model lifecycle/multi-model · **#10** EAGLE-3 proposer · **#12** audio input (`input_audio` + `/v1/audio/transcriptions`) · **#13** debug endpoints + Perfetto · **#14** vision preprocessing (tiling/anyres remainder) · **#15** fp8 KV quant · **#16** image diffusion.
- Closed: **#2** server split · **#3** decode ownership · **#4** FIM endpoint · **#5** benchmarks · **#11** audio log-mel preprocessing.

## Recently completed (this session)

Complete runtime built from scaffold + published: generation (greedy/speculative draft+prompt-lookup+MTP), samplers, FIM, grammar, tool use (Hermes-verified), chat templates, multi-session + prefix cache, paged/tiered/int8 KV, long-context O(1)/token static-cache, batched multi-agent serving, OpenAI HTTP (chat/completions/vision/streaming/sessions), observability, benchmarks (`onnx-genai-bench`), `onnx-genai-preprocess` crate, security hardening, CI + audits. **Fixed: categorical sampling had no RNG (always token 0).**

## Notable design changes / decisions to record

- Preprocessing lives in its own crate `onnx-genai-preprocess` (§35).
- Real-model exact-equality tests use `intra_op_threads=1` (ORT FP determinism).
- Paged attention deferred: Mobius lacks block-table KV (contiguous static-cache only).
- Audio & vision quality gated on real Mobius model packages (Whisper / CLIP+decoder).
