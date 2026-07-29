# Team Focus — now

**Current focus:** CUDA/CPU parity, scheduler coverage, metadata consumption, and route-first MoE offload.

**MERGED THIS WAVE:** PR #373 (`61d3bdac`) shipped issue #231 declarative name-agnostic I/O detection, operator-agnostic shared-KV contracts, and strict attention sequence-length validation. PR #378 (`ac75e146`) fixed flaky QMoE offload residency tests under coverage; its honest scheduler bounds and poison-recovering metrics lock prevent phantom-regression CI noise on unrelated PRs. PR #380 (`47c3331d`) landed #377's CORE name-agnostic model pipeline: decode roles use explicit metadata or a unique shape match, and ambiguous encoder-decoder fixtures declare component I/O. PR #382 (`85b9ba15`) landed name-free ORT shared/static-cache adapters, repaired #380's missing declared-KV threading in batched shared-buffer construction, and added a CPU continuous-batch regression test. PR #386 (`39c28b44`) completed RNN/GRU/LSTM shape inference under #355. PR #388 (`804ba860`) wired OpenAI Structured Outputs `response_format: json_schema` into the HTTP layer under #183. PR #390 (`0e62150e`) added Llama 3 and Mistral tool-call detection under #183.

**OPEN / PARTIAL:**
- #355 remains open for the Sequence/Optional/Map container family, requiring an IR `Value`/`TypeInfo` container-element-type extension, plus ONNX-ML.
- #55 remains open for heterogeneous per-node planning, JSON/YAML adapters, and non-CUDA accelerators.
- #82 is done. #87 and #63 need GPU-side prefetch unification and remain blocked on Phase-3b live device binding.
- #377 remains open for deferred nested-autoregressive routing, KV-bridge geometry, static-cache scatter ABI, and MTP/EAGLE3/Gemma4/shared-KV proposers. The first three require contract/schema plumbing; speculative proposers belong to the other team's #99 and are off-limits.
- #183 remains open for end-to-end streaming tool-call-detection coverage for the new formats and per-model stop-token (`<|eom_id|>`) nuances.

**BLOCKED GAPS:**
- #384 tracks the real large-model 27B E2E blocker: the native Unsqueeze rank bug, missing `recurrent_state`, Mobius #432, and external GPU occupation.
- Granite's unfused MatMul MoE export does not engage route-first QMoE offload.

**OFF-LIMITS:** #54 model-package and #299 LoRA belong to another team; #106 is under Justin's study. Do not touch resting other-squad open PRs (#314, #315, #317, #318, #291, #99).

**Session totals:** Five code PRs merged this run: #380, #382, #386, #388, and #390, advancing #377, #355, and #183; #384 records the large-model E2E gap.

**Updated:** 2026-07-29T06:55:00+0000
