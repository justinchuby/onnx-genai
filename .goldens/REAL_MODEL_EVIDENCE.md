# Real-model corpus run

Machine inventory and results for the corpus the parent asked to be exercised.
Captured at `5a58eafa` with ORT 1.29.0 (`~/.ort129`, `~/.ort129gpu`) on an H200.

## Available packages (8 of 8 candidate dirs present)

| package | shape covered |
|---|---|
| `Phi-3.5-mini-instruct-generic-cpu-2/v2` | int4, shared-buffer past/present |
| `qwen3-0.6b-generic-cpu-4/v4` | int4 past/present |
| `qwen2.5-0.5b-instruct-cuda-gpu-4/v4-bs128` | CUDA |
| `qwen2.5-coder-7b-instruct-generic-cpu-4/v4` | 7B CPU |
| `Phi-4-mini-instruct-cuda-gpu-5/v5` | CUDA |
| `gpt-oss-20b-generic-cpu-1/v1` | MoE |
| `qwen3.5-2b-text-generic-cpu-1/v1` | text decoder |
| `qwen3.5-0.8b-generic-cpu-2/v2` | multi-component (vision+text+embedding) |

DeepSeek weights are present under `/datadisks/disk1/models/ds-r1*` but as
HuggingFace checkpoints, not ONNX packages, so they are not loadable by this
runtime. That is the one **missing artifact**: an ONNX-exported DeepSeek-R1
package (`DEEPSEEK_R1_1_5B_E2E_DIR`). Everything else the corpus asks for is
present and was exercised.

## Lowering conformance — `canonical_lowering_corpus`, 5/5 pass

7 packages lower deterministically, and each lowered decoder's ports mirror the
resolved ABI exactly (no port invented, none dropped). The 8th
(`qwen3.5-0.8b`) is multi-component with no `inference_metadata.yaml`, so it is
neither a bare decoder nor a resolvable workflow; it skips with a printed reason.

## Text-generation parity — 5/5 byte-identical

`capture.sh` greedy streams, prompt `"Hello"`, 24 tokens, before vs after:

    phi35_mini_int4_cpu  phi4_mini_cuda  qwen25_0_5b_cuda  qwen3_0_6b_cpu  gpt_oss_20b_cpu

`diff before.tsv after_lowering.tsv` is empty.

## Divergence / decode-lock locks — pre-existing failures, NOT introduced here

`phi35_mini_divergence` (2) and `qwen3_0_6b_divergence` (2) fail with the locally
available packages. They fail **identically on unmodified `main` (11cb8e5f3)**,
verified by running the same tests in a clean worktree at that commit:

* `cannot resolve the decode ABI's token_input from tensor shape because 2 ports
  match: [input_ids, attention_mask]` — these locks need packages that declare
  `ports.roles`; the foundry-cache builds on this machine do not.
* the `phi35` native decode-loop lock additionally drifts from its recorded
  prefix at index ~86, on `main` as well.

Both are therefore stale locks against absent artifacts, not regressions. They
are recorded here rather than silently skipped so a future run can tell the
difference.
