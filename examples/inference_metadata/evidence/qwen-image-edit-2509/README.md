# Qwen-Image-Edit-2509 — end-to-end image edit

A real image-edit workflow executed through the inference-metadata runtime and
scored, pixel for pixel, against the upstream `diffusers` pipeline running the
same model, the same seed and the same schedule.

## Result

| Upstream reference (`diffusers`) | Runtime output (inference metadata) |
| --- | --- |
| ![Upstream diffusers reference output](upstream.png) | ![Runtime output produced through the inference-metadata workflow](runtime.png) |
| [`upstream.png`](upstream.png) — 1216×864 RGB, 1 126 358 bytes | [`runtime.png`](runtime.png) — 1216×864 RGB, 1 124 966 bytes |

Both images are committed unmodified, exactly as the harness wrote them.

## Run configuration

| Field | Value |
| --- | --- |
| Model | [`Qwen/Qwen-Image-Edit-2509`](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) |
| Revision | `d3968ef930e841f4c73640fb8afa3b306a78167e` |
| Precision | `bfloat16` |
| Hardware | NVIDIA H200 (143 771 MiB) |
| Execution provider | CUDA |
| Scheduler | `FlowMatchEulerDiscreteScheduler` (`diffusers` 0.36.0.dev0) |
| Denoising steps | 8 flow-matching steps (16 transformer calls — the schedule is run twice per step for true CFG) |
| `true_cfg_scale` | 4.0 |
| Seed | 42 |
| Output size | 1216×864 (latent grid 76×54, VAE scale factor 8) |
| Image sequence length | 4104 tokens, dynamic-shift `mu` = 0.6939516129032257 |

### Request

| Field | Value |
| --- | --- |
| Edit instruction (prompt) | `Make the cat wear a bright red scarf while preserving its pose.` |
| Negative prompt | a single space (`" "`) — the pipeline default for this model |
| Source image | `pipeline-cat-chonk.jpeg`, a 56 KB photograph of a cat from the Mobius end-to-end test-data set |

The source photograph lives in the producer-side Mobius test-data set and is not
redistributed here; the edit is legible without it, because `upstream.png` shows
the same subject and pose that the instruction asked to preserve. What the two
committed images demonstrate is not *whether* the edit is good — that is the
upstream model's business — but that the runtime reproduces the upstream result.

### Schedule

Sigmas, from the dynamic-shift flow-match schedule (`shift_terminal` 0.02,
`time_shift_type` exponential):

```text
1.0, 0.9160479307174683, 0.820091962814331, 0.7093585729598999,
0.5801501274108887, 0.4274216294288635, 0.24410808086395264,
0.019999980926513672, 0.0
```

## Metrics

[`metrics.json`](metrics.json) is the file the scoring harness emitted, copied
verbatim:

```json
{
  "shape": [
    864,
    1216,
    3
  ],
  "vs_upstream": {
    "psnr_db": 37.09709396916006,
    "max_abs": 0.4923330545425415,
    "mean_abs": 0.007391222752630711,
    "cosine": 0.9997113943099976
  },
  "vs_python_onnx_replay": {
    "psnr_db": 44.23202508642768,
    "max_abs": 0.43944546580314636
  }
}
```

| Comparison | PSNR | Cosine | Mean abs | Max abs |
| --- | --- | --- | --- | --- |
| Runtime vs upstream `diffusers` | **37.0971 dB** | **0.999711** | **0.007391** | 0.492333 |
| Runtime vs a Python ONNX replay of the same graphs | 44.2320 dB | — | — | 0.439445 |

Both comparisons are over the full `864 × 1216 × 3` float image in `[0, 1]`,
before PNG quantization.

### Reading the numbers

37.1 dB against upstream, with cosine similarity 0.999711 and mean absolute
error 0.0074, is agreement at roughly the level of bf16 rounding accumulated
over sixteen transformer calls: the average pixel differs by under 2/255, which
is invisible. `max_abs` of 0.49 is a small number of isolated pixels — expected
where a sharp edge lands on either side of a rounding boundary and then gets
amplified by the VAE decoder, not a structural difference.

The second row is the more diagnostic one. The runtime agrees *more* closely
with a Python replay of the identical ONNX graphs (44.2 dB) than with upstream
PyTorch (37.1 dB). That is the expected ordering: the residual gap to upstream
is dominated by the export itself, not by the runtime's execution of the
workflow. Had the runtime been mis-executing the control flow, the two rows
would not be ordered this way.

### Where the residual comes from

The same run also scored the *exported ONNX graphs*, replayed in Python, against
upstream PyTorch activations captured step by step. This isolates export error
from runtime error. The activations come from a 65 MB archive that is
intentionally not committed:

| Stage (exported ONNX vs upstream PyTorch) | Cosine | PSNR |
| --- | --- | --- |
| VAE encoder (image latents) | 0.9999654 | — |
| Transformer call 0 (noise prediction) | 0.9999742 | — |
| Transformer call 1 (noise prediction) | 0.9999811 | — |
| Latents after step 0 | 0.9999956 | — |
| Latents after step 7 (final) | 0.9982216 | — |
| VAE decoder (image) | 0.9999948 | 53.31 dB |
| Final image | 0.9997947 | 38.17 dB |

Read together with the table above, these numbers close the loop. The export
alone already sits at 38.17 dB against upstream; the runtime lands at 37.10 dB
against the same upstream reference while matching the replay of those exact
graphs at 44.23 dB. The runtime therefore adds roughly a decibel on top of an
error budget that the export had already spent — it is not the dominant term.

The drift across the eight latent steps — 0.9999956 after step 0 falling to
0.9982216 after step 7 — is compounding bf16 error rather than a divergence:
each step consumes the previous step's output, so a fixed per-step rounding
error accumulates. The VAE decoder contributes almost nothing (0.9999948),
which places the residual squarely in the iterated transformer, exactly where
bf16 accumulation is expected to show up.

## Performance

**Pending.** This run measured numerical fidelity only. No end-to-end latency,
per-step time or throughput was recorded for the image-edit path, so no
performance figure is quoted here. The only wall-clock number the run captured
belongs to the text-encoder parity harness and does not describe image
generation, so it is deliberately not reported as workflow performance.

## Reproducing

The workflow itself is exercised by
[`crates/onnx-genai-engine/tests/qwen_image_edit_workflow_e2e.rs`](../../../../crates/onnx-genai-engine/tests/qwen_image_edit_workflow_e2e.rs),
which is `#[ignore]`d because it needs a locally exported bf16 package of about
53 GiB that no CI job carries. With such a package available:

```text
ONNX_GENAI_EP=cuda cargo test -p onnx-genai-engine \
    --test qwen_image_edit_workflow_e2e -- --ignored --nocapture
```

The test writes the emitted image as raw little-endian float bytes plus a shape
sidecar; the producer-side harness converts that to PNG and scores it against
the upstream reference to produce `metrics.json`.

A trimmed, reviewable copy of the workflow metadata that drove this run is not
yet checked in. When it lands it belongs at
`examples/inference_metadata/qwen-image-edit-flow-match.yaml`, and this section
should link to it. It is named here as a path rather than a link on purpose, so
that this document contains no link that does not resolve.

## Files

| File | Bytes | SHA-256 |
| --- | --- | --- |
| [`upstream.png`](upstream.png) | 1 126 358 | `dd83fa209a823b672fc956dee357c6cac1fe7f0d30cc87687032335766d5f9e0` |
| [`runtime.png`](runtime.png) | 1 124 966 | `8dfd52aa52a894ce81aca2e435523200c72ee30f452aff9a3a0253d2bedbe374` |
| [`metrics.json`](metrics.json) | 312 | `1e9b616c693d5a2283932b3f35197bf83ea7b4b8c2ad35f7da87d719c4b31a5a` |
