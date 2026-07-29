# Team Focus — now

**Current focus:** CUDA/CPU parity, scheduler coverage, metadata consumption, and route-first MoE offload.

**MERGED THIS WAVE:** PR #407 (`e8f3a2dc`) closed #9 with registry-backed model warmup and typed admin warmup errors. PR #412 (`cf5a9024`) completed #377 native-side explicit inference metadata for static-cache, nested-AR, paged-KV, and encoder-role paths, including fail-closed `model.io.static_cache`. PR #415 (`8ebf3579`) restored CUDA SiLU decomposition fusion and recovered Qwen2.5-7B native CUDA decode above ORT (252→307.8 tok/s versus ORT 272.4), so the native>ORT directive is PASS again for the measured Foundry models.

**RECENTLY MERGED:** PR #373 (`61d3bdac`) shipped issue #231 declarative name-agnostic I/O detection, operator-agnostic shared-KV contracts, and strict attention sequence-length validation. PR #378 (`ac75e146`) fixed flaky QMoE offload residency tests under coverage. PR #380 (`47c3331d`) landed #377's core name-agnostic model pipeline. PR #382 (`85b9ba15`) landed name-free ORT shared/static-cache adapters and CPU continuous-batch regression coverage. PR #386 (`39c28b44`) completed RNN/GRU/LSTM shape inference under #355. PR #388 (`804ba860`) wired OpenAI Structured Outputs `response_format: json_schema` into the HTTP layer under #183. PR #390 (`0e62150e`) added Llama 3 and Mistral tool-call detection under #183.

**OPEN / PARTIAL:**
- #377 native side is done after #412. Remaining #377-linked work is Mobius explicit metadata emission (`onnxruntime/mobius#434`, awaiting Justin) plus the off-limits `decode_contract.rs` / #99 speculative-decoding naming-convention cleanup.
- #415 follow-up: low-priority `#415-silu-harden` should add hardening/coverage so CUDA SiLU decomposition fusion cannot be accidentally dropped again.
- #355 remains open for the Sequence/Optional/Map container family, requiring an IR `Value`/`TypeInfo` container-element-type extension, plus ONNX-ML.
- #55 remains open for heterogeneous per-node planning, JSON/YAML adapters, and non-CUDA accelerators.
- #82 is done. #87 and #63 need GPU-side prefetch unification and remain blocked on Phase-3b live device binding.
- #183 remains open for end-to-end streaming tool-call-detection coverage for the new formats and per-model stop-token (`<|eom_id|>`) nuances.

**BLOCKED GAPS:**
- #384 tracks the real large-model 27B E2E blocker: the native Unsqueeze rank bug, missing `recurrent_state`, Mobius #432, and external GPU occupation.
- Granite's unfused MatMul MoE export does not engage route-first QMoE offload.

**OFF-LIMITS:** #54 model-package and #299 LoRA belong to another team; #106 is under Justin's study. Do not touch resting other-squad open PRs (#314, #315, #317, #318, #291, #99).

**Session totals:** Eight code PRs merged this run: #380, #382, #386, #388, #390, #407, #412, and #415, advancing #377, #355, #183, #9, and CUDA native-vs-ORT performance; Mobius #434 remains for Justin.

**Updated:** 2026-07-29T11:45:00+0000
