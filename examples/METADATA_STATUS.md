# Inference metadata execution status

These examples separate **metadata-contract coverage** (what
`inference_metadata` can describe and validate) from **engine execution**
(what the current runtime can run end to end).

| Pipeline | Example | Metadata can DESCRIBE it? | Engine can EXECUTE it E2E today? | Key gaps |
|---|---|---:|---:|---|
| Text LLM (chat) | `onnx-genai run` / `onnx-genai generate` | ✅ | ✅ | Text generation is the supported CLI path; chat templates are applied (PR #134). |
| VLM (SmolVLM) | [`smolvlm-256m/`](smolvlm-256m/) | ✅ | ❌ | Host-side indexed vision/embedding fusion is not expressible. Positional row/column/global/wrapper image-token grammar is unsupported. There is no generic typed port shape/dtype inventory; longest-edge/grid preprocessing remains open-string data. Token 49189 is a bidirectional wrapper, so there is no truthful `vision_start_token_id`. |
| Audio ASR (Nemotron) | [`nemotron-audio/`](nemotron-audio/) | ✅ Mostly | ❌ | No executable audio frontend or chunk scheduler. The decoder→joiner route requires `[B,640,T] → [B,T,640]`, but edges cannot express transpose. Generic typed port inventory is missing, and the conditional RNN-T feedback loop is not runtime-supported. |
| Diffusion txt2img (SD1.5) | [`diffusion-metadata/`](diffusion-metadata/) | ✅ | ⚠️ Partial | The engine implements `run_iterative`, CFG, img2img, and the `ddim`, `euler`, `euler_ancestral`, `dpmpp_2m`, and `masked_diffusion` scheduler registry through the `render_sd` image path—not `onnx-genai run`. VAE latent scaling `0.18215` and geometry `[4,64,64]` / ÷8 are hardcoded in `render_sd`, not metadata. Scheduler construction rejects `v_prediction`/`x0`; only `epsilon` executes. |

`onnx-genai run` is a **text chat REPL**; it accepts neither audio nor image
input. The VLM and audio examples are metadata proofs, not turnkey inference
packages.

## Remaining engineering work (beat-ORT roadmap)

- Add per-CPU ISA paths in `matmul_nbits.rs`: AVX-512 VNNI, AVX2 SQNBit Int8,
  and ARM NEON dot-product.
- Migrate the remaining `group_query_attention`, `compressed_sparse`, `linear`,
  and `fused` attention kernels onto the shared `sdpa.rs` core.
- Reach Whisper token-exact parity with an ORT-compatible MatMulNBits
  `accuracy_level=4` CompInt8 route; add composite cross-cache wiring,
  `UnfoldTensor` timestamps, and non-MLAS `Conv`.
- Complete remaining bf16 operator coverage.
- Run rigorous clean-host native-vs-ORT benchmarks.
- Measure and improve traditional-ML performance on ONNX Model Zoo workloads
  such as ResNet and YOLO.
