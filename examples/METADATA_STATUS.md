# Inference metadata execution status

These examples separate **metadata-contract coverage** (what
`inference_metadata` can describe and validate) from **engine execution**
(what the current runtime can run end to end).

| Pipeline | Example | Metadata can DESCRIBE it? | Engine can EXECUTE it E2E today? | Key gaps |
|---|---|---:|---:|---|
| Text LLM (chat) | `onnx-genai run` / `onnx-genai generate` | ✅ | ✅ | Text generation is the supported CLI path; chat templates are applied (PR #134). |
| VLM (SmolVLM) | [`smolvlm-256m/`](smolvlm-256m/) | ✅ | ❌ | Host-side indexed vision/embedding fusion is not expressible. Positional row/column/global/wrapper image-token grammar is unsupported. There is no generic typed port shape/dtype inventory; longest-edge/grid preprocessing remains open-string data. Token 49189 is a bidirectional wrapper, so there is no truthful `vision_start_token_id`. |
| Audio ASR (Nemotron) | [`nemotron-audio/`](nemotron-audio/) | ⚠️ Hand-authored only | ❌ | RNN-T transducer is a **distinct pipeline family**, not an encoder-decoder. The hand-authored native contract can describe the topology (streaming Conformer encoder with `cache_last_channel`/`cache_last_time` state, LSTM prediction network with `h`/`c` state, joint network, VAD), but the `genai_config.json` **auto-detection fallback cannot** synthesize it and now explicitly declines it (`UnsupportedPipelineFamily: RNN-T transducer`) instead of fabricating Whisper-style cross-attention KV. Executing it E2E would require a joint-network greedy transducer decode loop (`blank_id`, `max_symbols_per_step`), streaming encoder cache-state management, VAD segmentation, and an audio frontend/chunk scheduler. The decoder→joiner route also needs `[B,640,T] → [B,T,640]`, which dataflow edges cannot express (no transpose). |
| Diffusion txt2img (SD1.5) | [`diffusion-metadata/`](diffusion-metadata/) | ✅ | ⚠️ Partial | The engine implements `run_iterative`, CFG, img2img, and the `ddim`, `euler`, `euler_ancestral`, `dpmpp_2m`, and `masked_diffusion` scheduler registry through the `render_sd` image path—not `onnx-genai run`. VAE latent scaling `0.18215` and geometry `[4,64,64]` / ÷8 are hardcoded in `render_sd`, not metadata. All four continuous schedulers now accept `epsilon` (default), `v_prediction`, and `sample`/`x0` prediction types (the model output is converted to epsilon/x0 via the diffusers formulas before the existing step math). The `epsilon` path is byte-identical to before; `v_prediction`/`x0` are unit-tested against the diffusers conversion formulas and verified to step finitely, but end-to-end image-quality parity for `v_prediction` awaits a real v-prediction model (SD 2.x / SDXL refiner) on the host to validate. |

`onnx-genai run` is a **text chat REPL**; it accepts neither audio nor image
input. The VLM and audio examples are metadata proofs, not turnkey inference
packages.

### Encoder-decoder auto-detection vs. RNN-T transducers

The `genai_config.json` compatibility loader recognizes an encoder-decoder
(cross-attention, Whisper-style) package **structurally**, from a declared
`model.encoder` feeding a transformer decoder with self- and cross-attention
KV. A **Conformer-Transducer / RNN-T** package (e.g. Nemotron speech) also
declares a `model.encoder`, but it is a fundamentally different family: a
streaming Conformer encoder with cache state, an **LSTM prediction network**
(`targets` + `lstm_hidden_state`/`lstm_cell_state`, no attention KV), and a
**joint network** (`joint.onnx`) with no cross-attention analog. The loader now
detects this signature (joint network and/or LSTM prediction network) **before**
the encoder-decoder check and declines it honestly
(`UnsupportedPipelineFamily: RNN-T transducer`) rather than silently emitting a
fabricated Whisper-style cross-KV spec that does not match the real graphs. This
is a shape-keyed guard, not a model-name check.

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
