# Hugging Face inference-metadata examples: operational audit

Live static audit completed at `2026-08-27T19:06:51.564789+00:00` against the
public collection API:

<https://huggingface.co/api/collections/justinchuby/onnx-genai-inference-metadata-examples>

The collection contained 29 repositories: 28 models and one catalogue dataset.
All final revisions below were re-resolved after the hosted repair described
below; none had drifted at final verification. The audit used Python `3.12.9`,
`onnx` `1.22.0`, and a 67,108,864-byte ONNX download cap.

## Meaning of the classifications

- **fully self-contained runnable**: every metadata-referenced ONNX component is
  present, was parsed rather than skipped, declares a component port ABI, and
  has no metadata/graph ABI finding. Every discovered ONNX external-data
  location is present and large enough. Each media request input has a relevant
  bundled media file, raw tensor, or matching NPZ member. This is static package
  eligibility only, not a generation smoke test or universal backend claim.
- **unverified/incomplete ONNX inspection**: at least one ONNX graph was skipped
  or failed to parse, or a metadata-referenced ONNX component has no declared
  port ABI to compare. Hosted execution evidence does not erase that static
  evidence gap.
- **unsupported metadata/ONNX ABI**: parsing completed, but a declared component
  port is not supported by the physical graph ABI.
- **needs external model/media/request input by design**: the package is
  otherwise statically complete, but a media request input has no relevant
  bundled asset. An arbitrary file whose path merely contains `request` or
  `input` is not evidence.
- **metadata-only/example**: no executable model package is intended.
- **broken/missing generated asset**: a required artifact is absent or
  external data is absent or too short.

Do not shorten **fully self-contained runnable** to “verified on every backend.”
The audit did not execute generation. The execution-evidence descriptions below
are separate historical records and are the limit of any runtime claim.

## Reproduce the static audit

Prerequisites are Python 3, PyYAML, `onnx`, network access to the public Hub,
and enough memory for each ONNX file below the selected cap.

```bash
python3 scripts/audit_hf_metadata_collection.py --self-test
python3 scripts/audit_hf_metadata_collection.py \
  --max-onnx-bytes 67108864 \
  --metadata-dir hf-audit-metadata \
  --output hf-audit.json
cargo run -q -p onnx-genai-metadata --bin validate_metadata -- \
  --metadata-only hf-audit-metadata/*/inference_metadata.yaml
```

The JSON records every path, byte size, blob ID, LFS object SHA-256 and pointer
size, metadata input/default source, tokenizer check, ONNX graph I/O, external
data reference, per-component inspection status, request-asset match, ABI
finding, classification reason, dependency version, and audit scope. All 28
metadata files passed the Rust metadata-only semantic validator. That validator
and this audit are static checks, not generation smoke.

The audit parsed 469 of 472 ONNX files without downloading external weight data.
Of 447 metadata-referenced ONNX components, 444 graphs parsed: 426 declared ABIs
matched, 16 parsed components had no declared port ABI, and two parsed EAGLE3
components had unsupported output declarations. The three unparsed referenced
components were the capped Qwen3, Pangu, and ACT graphs: Qwen3 declares an ABI
that could not be inspected, while Pangu and ACT declare no component port ABI.
Thus the script's aggregate `abi_not_declared` count is 18—16 parsed plus two
skipped—not 18 parsed components. Six media request requirements had relevant
assets; Muse's image requirement did not. There were zero ONNX parse errors and
zero missing or short external-data findings. The cap deliberately skipped
three embedded-weight graphs:

- `qwen3-0.6b-onnx-genai/model.onnx`: 524,589,133 bytes;
- `pangu-weather-1h-onnx-catalogue/model.onnx`: 1,181,711,187 bytes;
- `act-aloha-policy-onnx-catalogue/model.onnx`: 137,408,166 bytes.

Their physical ABI is unverified in this run. Pangu and ACT also lack a declared
component port ABI. No classification treats README text or hosted execution as
a substitute for the skipped static inspection.

## Final per-item catalogue

“ONNX inspection” is `parsed/all ONNX files`; ABI counts cover only
metadata-referenced ONNX components.

| # | Repository @ resolved revision | Files / total | LFS | ONNX inspection; referenced ABI | Tokenizer/request evidence | Classification |
|---:|---|---:|---:|---|---|---|
| 0 | [`justinchuby/onnx-genai-inference-metadata-catalogue`](https://huggingface.co/datasets/justinchuby/onnx-genai-inference-metadata-catalogue/tree/9981767dc1a5e5559ab3a58cdb235d8bf53007d2) `9981767dc1a5e5559ab3a58cdb235d8bf53007d2` | 8 / 0.1 MiB | 0 / 0.00 GiB | n/a | catalogue data | **metadata-only/example** |
| 1 | [`justinchuby/qwen2.5-0.5b-instruct-onnx-genai`](https://huggingface.co/justinchuby/qwen2.5-0.5b-instruct-onnx-genai/tree/1eabeec267303a75170ae1b43acf59cb01b47a63) `1eabeec267303a75170ae1b43acf59cb01b47a63` | 21 / 837.0 MiB | 13 / 0.82 GiB | 11/11; ABI 11 pass | tokenizer | **fully self-contained runnable** |
| 2 | [`justinchuby/qwen3-0.6b-onnx-genai`](https://huggingface.co/justinchuby/qwen3-0.6b-onnx-genai/tree/e6fdc5eb2ba34f2163c95255279303b09360b702) `e6fdc5eb2ba34f2163c95255279303b09360b702` | 22 / 511.7 MiB | 12 / 0.50 GiB | 10/11; ABI 10 pass, model skipped | tokenizer | **unverified/incomplete ONNX inspection** |
| 3 | [`justinchuby/deepseek-r1-distill-qwen-1.5b-onnx-genai`](https://huggingface.co/justinchuby/deepseek-r1-distill-qwen-1.5b-onnx-genai/tree/b30d31a08983f5acc013f7f2df95d9f9444b69d2) `b30d31a08983f5acc013f7f2df95d9f9444b69d2` | 21 / 1.29 GiB | 13 / 1.29 GiB | 11/11; ABI 11 pass | tokenizer | **fully self-contained runnable** |
| 4 | [`justinchuby/onnx-genai-example-qwen2-5-0-5b-portable-f32`](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-0-5b-portable-f32/tree/329b806feac8e219970124825d37348f7196152a) `329b806feac8e219970124825d37348f7196152a` | 31 / 1.86 GiB | 12 / 1.85 GiB | 11/11; ABI 11 pass | tokenizer; request | **fully self-contained runnable** |
| 5 | [`justinchuby/onnx-genai-example-qwen2-5-0-5b-cuda-gqa-f16`](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-0-5b-cuda-gqa-f16/tree/96fa048518005bc0eb9df7ad939fb9f0dd172911) `96fa048518005bc0eb9df7ad939fb9f0dd172911` | 32 / 959.5 MiB | 12 / 0.93 GiB | 11/11; ABI 11 pass | tokenizer; request | **fully self-contained runnable** |
| 6 | [`justinchuby/onnx-genai-example-qwen2-5-0-5b-static-cache-f32`](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-0-5b-static-cache-f32/tree/5e94f4ee439af803b990a59094c2902f7bf03a4f) `5e94f4ee439af803b990a59094c2902f7bf03a4f` | 31 / 1.86 GiB | 12 / 1.85 GiB | 11/11; ABI 11 pass | tokenizer; request | **fully self-contained runnable** |
| 7 | [`justinchuby/onnx-genai-example-qwen3-5-0-8b-hybrid-vlm-f32`](https://huggingface.co/justinchuby/onnx-genai-example-qwen3-5-0-8b-hybrid-vlm-f32/tree/a5f905097a0316eb918f71e94fc55084d34f09ca) `a5f905097a0316eb918f71e94fc55084d34f09ca` | 37 / 4.21 GiB | 18 / 4.20 GiB | 14/14; ABI 13 pass, 1 undeclared | relevant request image | **unverified/incomplete ONNX inspection** |
| 8 | [`justinchuby/onnx-genai-example-esm2-t6-8m`](https://huggingface.co/justinchuby/onnx-genai-example-esm2-t6-8m/tree/1954e4b60aa939a8220c884c3408d06ffa34e494) `1954e4b60aa939a8220c884c3408d06ffa34e494` | 18 / 28.5 MiB | 2 / 0.03 GiB | 1/1; ABI 1 pass | tokenizer; request | **fully self-contained runnable** |
| 9 | [`justinchuby/onnx-genai-example-prot-bert`](https://huggingface.co/justinchuby/onnx-genai-example-prot-bert/tree/83d5acc54a7b0eb9f3cd33b668fa8b1fe80e701d) `83d5acc54a7b0eb9f3cd33b668fa8b1fe80e701d` | 18 / 1.56 GiB | 2 / 1.56 GiB | 1/1; ABI 1 pass | tokenizer; request | **fully self-contained runnable** |
| 10 | [`justinchuby/onnx-genai-example-whisper-tiny`](https://huggingface.co/justinchuby/onnx-genai-example-whisper-tiny/tree/4a80120cbe62c10d25ae89d6430896e726565569) `4a80120cbe62c10d25ae89d6430896e726565569` | 38 / 224.9 MiB | 15 / 0.22 GiB | 12/12; ABI 12 pass | tokenizer; relevant `request.flac` | **fully self-contained runnable** |
| 11 | [`justinchuby/onnx-genai-example-wav2vec2-base-960h-ctc`](https://huggingface.co/justinchuby/onnx-genai-example-wav2vec2-base-960h-ctc/tree/820a7c59ad73e088858230d567395d625c3fac04) `820a7c59ad73e088858230d567395d625c3fac04` | 21 / 360.5 MiB | 3 / 0.35 GiB | 1/1; ABI 1 pass | tokenizer; relevant `request.flac` | **fully self-contained runnable** |
| 12 | [`justinchuby/onnx-genai-example-qwen2-5-1-5b-lora-selection`](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-1-5b-lora-selection/tree/5f0878d8f72cb54b284f052b5ce016717898872b) `5f0878d8f72cb54b284f052b5ce016717898872b` | 40 / 3.40 GiB | 15 / 3.39 GiB | 11/11; ABI 11 pass | tokenizer; request and adapter | **fully self-contained runnable** |
| 13 | [`justinchuby/onnx-genai-example-mistral-7b-v0-1-sliding-window`](https://huggingface.co/justinchuby/onnx-genai-example-mistral-7b-v0-1-sliding-window/tree/e6c5b87a25883e2d1560584137bea340e3a8fba2) `e6c5b87a25883e2d1560584137bea340e3a8fba2` | 33 / 13.50 GiB | 13 / 13.50 GiB | 11/11; ABI 11 pass | tokenizer; request | **fully self-contained runnable** |
| 14 | [`justinchuby/onnx-genai-example-qwen3-0-6b-eagle3`](https://huggingface.co/justinchuby/onnx-genai-example-qwen3-0-6b-eagle3/tree/10385b7b8f1a3066d4ff15a72ec2194cce324f19) `10385b7b8f1a3066d4ff15a72ec2194cce324f19` | 52 / 3.06 GiB | 23 / 3.06 GiB | 13/13; ABI 2 unsupported | tokenizer; request | **unsupported metadata/ONNX ABI** |
| 15 | [`justinchuby/onnx-genai-stable-diffusion-bk-sdm-small`](https://huggingface.co/justinchuby/onnx-genai-stable-diffusion-bk-sdm-small/tree/dd7ecd9d50a2210aa796a2efedb5489125f8be37) `dd7ecd9d50a2210aa796a2efedb5489125f8be37` | 48 / 2.35 GiB | 22 / 2.34 GiB | 16/16; ABI 14 pass, 1 undeclared | tokenizer; request | **unverified/incomplete ONNX inspection** |
| 16 | [`justinchuby/onnx-genai-example-qwen-image-edit-2509`](https://huggingface.co/justinchuby/onnx-genai-example-qwen-image-edit-2509/tree/69544173de85fe785b1dfd7e1f5f1795c23bafb1) `69544173de85fe785b1dfd7e1f5f1795c23bafb1` | 86 / 52.67 GiB | 46 / 52.66 GiB | 16/16; ABI 11 pass, 2 undeclared | tokenizer; image/runtime inputs | **unverified/incomplete ONNX inspection** |
| 17 | [`justinchuby/onnx-genai-cogvideox-2b`](https://huggingface.co/justinchuby/onnx-genai-cogvideox-2b/tree/27e85b66e91a2be33be53ec44d0247a9b232220d) `27e85b66e91a2be33be53ec44d0247a9b232220d` | 74 / 24.57 GiB | 45 / 24.56 GiB | 17/17; ABI 16 pass, 1 undeclared | tokenizer; runtime inputs | **unverified/incomplete ONNX inspection** |
| 18 | [`justinchuby/pangu-weather-1h-onnx-catalogue`](https://huggingface.co/justinchuby/pangu-weather-1h-onnx-catalogue/tree/36baa7a9b345c3accf6f9e5a0303d9b6960dea34) `36baa7a9b345c3accf6f9e5a0303d9b6960dea34` | 14 / 1.30 GiB | 3 / 1.30 GiB | 0/1; graph skipped, ABI undeclared | `request.npz`; no tokenizer needed | **unverified/incomplete ONNX inspection** |
| 19 | [`justinchuby/act-aloha-policy-onnx-catalogue`](https://huggingface.co/justinchuby/act-aloha-policy-onnx-catalogue/tree/8428f02d75fb029407e3b699dd81842d7e9bb3ff) `8428f02d75fb029407e3b699dd81842d7e9bb3ff` | 20 / 131.2 MiB | 4 / 0.13 GiB | 0/1; graph skipped, ABI undeclared | relevant `request.npz` image member | **unverified/incomplete ONNX inspection** |
| 20 | [`justinchuby/moshiko-full-duplex-onnx-catalogue`](https://huggingface.co/justinchuby/moshiko-full-duplex-onnx-catalogue/tree/4114e4103f0e9458a2917ecb3048e33b06577891) `4114e4103f0e9458a2917ecb3048e33b06577891` | 43 / 4.02 GiB | 30 / 4.02 GiB | 23/23; ABI 21 pass, 2 undeclared | relevant `request.npz` waveform; bundled SPM | **unverified/incomplete ONNX inspection** |
| 21 | [`justinchuby/onnx-genai-example-gemma4-e2b`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-e2b/tree/a74c3ad0209c4f04251f0c1d48a3796fc63a4a8f) `a74c3ad0209c4f04251f0c1d48a3796fc63a4a8f` | 40 / 9.59 GiB | 13 / 9.59 GiB | 11/11; ABI 1 pass | tokenizer | **fully self-contained runnable** |
| 22 | [`justinchuby/onnx-genai-example-gemma4-e2b-assistant`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-e2b-assistant/tree/27f03dd07510c0a7794c0e04e2d72998b466aff7) `27f03dd07510c0a7794c0e04e2d72998b466aff7` | 33 / 9.92 GiB | 5 / 9.92 GiB | 2/2; ABI 2 pass | tokenizer; repaired target | **fully self-contained runnable** |
| 23 | [`justinchuby/onnx-genai-example-gemma4-26b-a4b`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-26b-a4b/tree/e3336a2baea76d6a759fd32347927ca6ec85fbd1) `e3336a2baea76d6a759fd32347927ca6ec85fbd1` | 29 / 48.78 GiB | 3 / 48.78 GiB | 1/1; ABI 1 pass | tokenizer | **fully self-contained runnable** |
| 24 | [`justinchuby/onnx-genai-example-minimax-music3`](https://huggingface.co/justinchuby/onnx-genai-example-minimax-music3/tree/2c7c9f57c42eb7953a01750d51adb724b3181223) `2c7c9f57c42eb7953a01750d51adb724b3181223` | 75 / 26.63 GiB | 57 / 26.63 GiB | 41/41; ABI 35 pass, 6 undeclared | tokenizer | **unverified/incomplete ONNX inspection** |
| 25 | [`justinchuby/onnx-genai-example-gemma4-e2b-speculative`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-e2b-speculative/tree/6a6d111c877c0b395aff022efa7374de77be2e00) `6a6d111c877c0b395aff022efa7374de77be2e00` | 25 / 9.92 GiB | 5 / 9.92 GiB | 2/2; ABI 2 pass | tokenizer | **fully self-contained runnable** |
| 26 | [`justinchuby/qwen2.5-14b-instruct-int4-zp-onnx`](https://huggingface.co/justinchuby/qwen2.5-14b-instruct-int4-zp-onnx/tree/b25c589c213ec1efe51eabff1bd35c3cc38fbc4f) `b25c589c213ec1efe51eabff1bd35c3cc38fbc4f` | 20 / 7.77 GiB | 12 / 7.76 GiB | 11/11; ABI 11 pass | tokenizer | **fully self-contained runnable** |
| 27 | [`justinchuby/sensenova-u1.5-8b-mot-onnx-canonical`](https://huggingface.co/justinchuby/sensenova-u1.5-8b-mot-onnx-canonical/tree/a57ae0a765ac6ec55ddefaa8af12fdf3c9e670d5) `a57ae0a765ac6ec55ddefaa8af12fdf3c9e670d5` | 250 / 48.03 GiB | 198 / 48.03 GiB | 186/186; ABI 184 pass, 2 undeclared | tokenizer; relevant image/edit requests | **unverified/incomplete ONNX inspection** |
| 28 | [`justinchuby/Muse-Glimmer-30B-ONNX-INT4-CUDA`](https://huggingface.co/justinchuby/Muse-Glimmer-30B-ONNX-INT4-CUDA/tree/76c9896978f6c3f36d75ab7f627521f168cc7010) `76c9896978f6c3f36d75ab7f627521f168cc7010` | 35 / 17.88 GiB | 18 / 17.88 GiB | 14/14; ABI 13 pass, 1 undeclared | tokenizer; no relevant request image | **unverified/incomplete ONNX inspection** |

Final primary classifications: 16 fully self-contained runnable, 11
unverified/incomplete ONNX inspection, one unsupported metadata/ONNX ABI, and
one metadata-only/example. No item is classified broken after the hosted repair.
Muse is in the unverified class because its vision encoder has no declared port
ABI; independently, it is also the sole item with an unmatched media request
requirement.

## ABI, defaults, tokenizer, and runtime evidence

- Of 447 metadata-referenced ONNX components, 426 parsed and passed their
  declared port comparison. Sixteen parsed but had no declared component port
  ABI: Qwen3.5 VLM `vision_encoder`; Stable Diffusion `vae_decoder`; Qwen Image
  Edit `vae_encoder` and `vae_decoder`; CogVideoX `vae_decoder`; Moshiko
  `encoder` and `decoder`; six MiniMax Music3 embedding/head/projection/vocoder
  components; Sensenova `image_gen_embedding` and `vision_encoder`; and Muse
  `vision_encoder`. Pangu `forecast` and ACT `policy` are the other two
  undeclared-ABI components, but their graphs were skipped by the byte cap, so
  they are not part of the 16 parsed components. These are incomplete static ABI
  evidence, not passes. Qwen3's capped `model.onnx` is the remaining referenced
  component and its declared ABI was not inspected.
- The two EAGLE3 referenced components parsed but failed the declared output
  comparison. `proposer/model.onnx` and `verifier/model.onnx` declare
  `present_key_values`, while the physical outputs are `present.N.key` and
  `present.N.value` (one layer for the proposer, 28 for the verifier). Inputs
  use `past_key_values.N.{key,value}` and are accepted by the aggregate-prefix
  check. The hosted chained ORT record does not prove that the unsupported
  metadata output name is accepted by the audited consumer, so the item is not
  runnable-classified.
- Request sources generally have no serialized `source.default`; generation
  constants are literal workflow inputs or package token facts. Thus callers
  must supply the request values named by metadata. The audit only requires
  asset evidence for inputs explicitly carrying a media role, and it matches
  media by type: an image/audio/video file in a request/input evidence path, a
  correspondingly named raw tensor, or a matching NPZ member. The six matches
  were Qwen3.5's request image, Whisper and Wav2Vec2 `request.flac`, ACT's
  `observation.images.top` NPZ member, Moshiko's `waveform` NPZ member, and
  Sensenova's request image. Muse's `request.image` has no relevant image
  asset. JSON or another unrelated request-named file is not accepted by
  itself as media proof.
- All referenced external-data locations found in the 469 parsed graphs exist
  and are large enough for the maximum declared `offset + length`.
- Tokenizer JSON/vocab/config files were parsed when present. Two packages
  intentionally expose different generic tokenizer EOS and workflow package
  EOS values: Qwen3.5 VLM uses metadata `248044` (`<|endoftext|>`) while
  `tokenizer_config.json` resolves `248046` (`<|im_end|>`); Muse uses metadata
  `200008` (`<|eot|>`) while tokenizer config resolves `200001`
  (`<|end_of_text|>`). Runtime termination must use metadata authority rather
  than silently substituting the generic tokenizer default.
- The hosted catalogue dataset has 23 execution-evidence rows. They cover
  direct ORT or generic-workflow runs for the Qwen2.5 variants, Qwen3.5 VLM,
  ESM2, ProtBert, Whisper, Wav2Vec2, LoRA, Mistral, EAGLE3, Stable Diffusion,
  Qwen Image Edit, CogVideoX, Pangu, ACT, Moshiko, Gemma packages, MiniMax
  Music3, and Sensenova. Evidence scope remains the runtime/provider named in
  each row, not universal backend proof. Qwen3, DeepSeek and Muse have
  model-card workflow records; Qwen2.5-14B has only a recipe, not accepted
  execution proof.

### Exact runnable wording and commands

Download an immutable package, including LFS objects:

```bash
hf download REPO_ID --revision EXACT_SHA --local-dir MODEL_DIR
cargo run -q -p onnx-genai-metadata --bin validate_metadata -- MODEL_DIR
```

The second command is package validation, not execution. The live audit in this
report did not run generation. The verified hosted instructions are fixed at
the Qwen model card's
[README correction revision](https://huggingface.co/justinchuby/qwen2.5-0.5b-instruct-onnx-genai/commit/e5cc13d7232bfe2b49a9df4ef13ce60714170106).
The CLI takes a downloaded local package directory after `generate` and the
prompt canonically as its final positional argument. `--prompt` is also
supported; supply exactly one prompt, not both forms. `-p` is not supported.
For this Qwen package's ChatML template, omit `--raw` for unformatted prompts;
use `--raw` only when the caller has already formatted this package's complete
request as ChatML. Raw formatted text bypasses this package's template and its
system/user role separation, so it should not be used where that separation
matters. Revision pinning makes the downloaded contents reproducible, not
trusted or safe by itself; review and trust model packages before running them.

With onnx-genai source
[`3dabd2c0`](https://github.com/justinchuby/onnx-genai/commit/3dabd2c0c2066183407c6bd98372e18e59c9571a)
(workspace version `0.1.0-dev.5`), a fresh download of Qwen revision
`1eabeec267303a75170ae1b43acf59cb01b47a63` and this public command shape
produced a coherent, non-empty response on CPU ORT without a
`missing from input feed` error:

```bash
hf download justinchuby/qwen2.5-0.5b-instruct-onnx-genai \
  --revision 1eabeec267303a75170ae1b43acf59cb01b47a63 \
  --local-dir ./qwen2.5-0.5b-instruct-onnx-genai

ONNX_GENAI_EP=cpu ONNX_GENAI_KV_MAX_LEN=128 \
onnx-genai generate ./qwen2.5-0.5b-instruct-onnx-genai \
  --backend ort \
  --max-new-tokens 16 \
  --temperature 0 \
  --stop '<|im_end|>' \
  'Hello! In one sentence, what is Rust?'
```

Native was not available and was not tested; this is CPU-ORT evidence only.
The current-package CLI smoke above is known-tested only on onnx-genai source
`3dabd2c0` (workspace version `0.1.0-dev.5`). The model card's historical
`nxrt==0.1.0.dev3` output predates the current workflow-metadata revision and
does not demonstrate an nxrt compatibility floor for this snapshot.

The user's exact missing input name remains unreproduced and must not be
invented. Current onnx-genai source builds its CLI bindings against ORT
1.29/API29 and rejects an incompatible loaded runtime/API; ORT 1.28/API28 versus
1.29/API29 is therefore a CLI binary/build/API compatibility boundary, not a
workflow-metadata requirement. That boundary is distinct from a defect in the
hosted metadata or current request binding; the fresh current-path smoke above
did not reproduce such a binding failure.

Static evidence independently shows that
`model.onnx` is 189,243 bytes, references 318 tensors in the present
865,533,952-byte `model.onnx.data`, all ten policy graphs parse, tokenizer EOS
`151645` agrees with metadata, and every inspected port matches. Those facts are
not a model smoke, and the separate smoke does not change the static
classification or any per-item count.

The only hosted packages with self-contained, directly reusable scripts found
by this audit have these exact prerequisites/commands:

```bash
# Pangu: Python, numpy, onnxruntime-gpu with CUDAExecutionProvider.
python3 run.py --request request.npz --output reproduced_output.npz --provider cuda

# ACT: Python NVIDIA CUDA/cuDNN wheels, numpy, onnxruntime-gpu, NVIDIA GPU.
bash run_cuda.sh

# Moshiko: Python NVIDIA CUDA/cuDNN wheels, package's Python dependencies,
# onnxruntime-gpu, NVIDIA GPU.
bash run_cuda.sh
```

Other rows either do not preserve a portable exact command in their revision,
have an incomplete static classification above, or use a script that hard-codes
the publisher's machine path and downloads upstream files. Do not describe
those as turnkey CLI examples; cite their exact provider-specific evidence and
the static limitation independently.

## Hosted correction

At revision `0778733e00713fad71c858553939817a273b7114`,
`onnx-genai-example-gemma4-e2b-assistant` metadata referenced
`target/model.onnx`, but neither it nor its external data existed. The item was
therefore broken at audit time. Authenticated `hf` tooling was available, so the
two exact LFS objects already used by the self-contained speculative package
were copied server-side, without downloading 9.56 GiB:

- `target/model.onnx`: 766,781 bytes, LFS SHA-256
  `c1a83ab620bed9e0867bcc8402f2c42b8d925b8453273d8872cdfc47c5d080f2`,
  pointer size 131;
- `target/model.onnx.data`: 10,263,789,568 bytes, LFS SHA-256
  `e525066ebb0284923ddcf12e8e3030a8cf49beb0757a35c06f373256929577eb`,
  pointer size 136.

Remote result:

- before: `0778733e00713fad71c858553939817a273b7114`;
- after: `27f03dd07510c0a7794c0e04e2d72998b466aff7`;
- commit: <https://huggingface.co/justinchuby/onnx-genai-example-gemma4-e2b-assistant/commit/27f03dd07510c0a7794c0e04e2d72998b466aff7>.

Post-upload API verification found both paths and exact hashes. Both ONNX graphs
parsed, the target graph's external-data extent fit the copied file, and no
metadata-referenced artifact remained missing.

The catalogue dataset was then updated atomically so its inventory and JSONL
row no longer pinned the broken revision:

- before: `8ba416600109201e10256841e20f0c1d2777af6e`;
- after: `9981767dc1a5e5559ab3a58cdb235d8bf53007d2`;
- commit: <https://huggingface.co/datasets/justinchuby/onnx-genai-inference-metadata-catalogue/commit/9981767dc1a5e5559ab3a58cdb235d8bf53007d2>.

Remote verification re-read both files at the new dataset SHA and confirmed the
assistant revision `27f03dd...` and final 10,656,048,308-byte package size.
