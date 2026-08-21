# Inference-metadata end-to-end evidence

Screenshots pasted into pull-request comments are not durable: GitHub's user
attachment URLs are scoped to the uploader's session, they do not render for
every reader, and they disappear entirely once a comment is edited or the
upload expires. Reviewers of the inference-metadata work kept being asked to
trust numbers whose supporting media they could not see.

This directory fixes that by tracking the *actual output artifacts* of real
end-to-end runs in the repository, next to the metrics they justify. Every file
here was produced by a real model on real hardware; nothing is regenerated at
build time and nothing here is a placeholder.

## Index

| Workflow / model | Exact output artifact | Metrics | Hardware / EP | Performance | Proof level |
| --- | --- | --- | --- | --- | --- |
| Image edit (flow matching) — [`Qwen/Qwen-Image-Edit-2509`](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) @ `d3968ef930e841f4c73640fb8afa3b306a78167e` | [`qwen-image-edit-2509/runtime.png`](qwen-image-edit-2509/runtime.png) (1216×864 RGB), scored against [`qwen-image-edit-2509/upstream.png`](qwen-image-edit-2509/upstream.png) | PSNR 37.0971 dB, cosine 0.999711, mean abs 0.007391, max abs 0.492333 — [`qwen-image-edit-2509/metrics.json`](qwen-image-edit-2509/metrics.json) | NVIDIA H200, bfloat16, CUDA execution provider | **Pending** — not measured in this run | Full pixel output compared to the upstream `diffusers` reference image ([details](qwen-image-edit-2509/README.md)) |
| Speech-to-text — Whisper Tiny | [`whisper-tiny/input.wav`](whisper-tiny/input.wav) plus emitted transcript | Exact transcript match; encoder relative L2 `6.58e-06` — [`metrics.json`](whisper-tiny/metrics.json) | Real ONNX and HuggingFace run | **Pending** | Real audio → text, upstream parity ([details](whisper-tiny/README.md)) |
| Full-duplex speech — PersonaPlex | [`personaplex/runtime.wav`](personaplex/runtime.wav) and [`reference.wav`](personaplex/reference.wav) | Waveform corr `0.9999999999997841`; 48/48 output codes — [`metrics.json`](personaplex/metrics.json) | NVIDIA GPU, CUDA, f32 | p50 1,587.60 ms/step | Real audio output and full per-step parity ([details](personaplex/README.md)) |
| Text-to-video — tiny CogVideoX | [`cogvideox-tiny/output.gif`](cogvideox-tiny/output.gif), [`contact-sheet.png`](cogvideox-tiny/contact-sheet.png) | PSNR 84.66 dB, corr `0.9999999983` — [`runtime-parity.json`](cogvideox-tiny/runtime-parity.json) | ONNX runtime workflow | 174.36 ms execution, 8.27 s load | Real 17-frame output and runtime parity ([details](cogvideox-tiny/README.md)) |
| Text-to-image — tiny Stable Diffusion | [`stable-diffusion-tiny/output.png`](stable-diffusion-tiny/output.png) | Exact run parameters — [`run.json`](stable-diffusion-tiny/run.json) | Upstream fixture run | **Pending** | Real generated reference image; no retained runtime render ([details](stable-diffusion-tiny/README.md)) |
| Protein embeddings — ESM-2 and ProtBert | [`protein-encoders/esm2_similarity.png`](protein-encoders/esm2_similarity.png), [`protbert_similarity.png`](protein-encoders/protbert_similarity.png) | Full CPU/CUDA parity, padding isolation, embeddings — [`EVIDENCE.md`](protein-encoders/EVIDENCE.md) | NVIDIA H200, CPU/CUDA | ESM-2 CUDA up to 628k residues/s; ProtBert measurements included | Real protein → embedding outputs, upstream parity, batching, performance |

`Performance` is deliberately empty rather than estimated. The Qwen-Image-Edit
run captured numerical fidelity only; no end-to-end latency or throughput was
recorded, so there is no measured number to quote. It will be filled in when a
timed run produces one.

## What is tracked here, and why

Two rules apply to binaries in this repository, and they point in opposite
directions on purpose.

**Graph fixtures stay text.** Model fixtures under `tests/fixtures/` are stored
as ONNX protobuf TextFormat (`*.onnx.textproto`) so that a graph change shows up
as a reviewable diff. `.gitignore` ignores `*.onnx` precisely to keep that
property, and the handful of fixtures that must stay binary are each listed
there with a written reason. Nothing in this directory changes that: no graph
is added here in binary form.

**Evidence media stays binary, and that is intentional.** A PNG, a WAV or an MP4
is the *result* being asserted, not a source artifact, and it has no meaningful
text form. Re-encoding it would destroy the very bytes under discussion. These
files are therefore committed as-is and are expected to grow the repository by
roughly the size of the artifact — about 1.1 MB per Qwen-Image-Edit image today.
That cost is accepted so a reviewer can open the image directly from the tree,
at any commit, without a working GPU, a 53 GiB model package, or a live
attachment URL.

What does *not* belong here: model weights, exported packages, full tensor
dumps, capture archives, and anything that is large because it is intermediate
rather than because it is the conclusion. The Qwen-Image-Edit run also produced
a 65 MB `upstream_capture.npz` of per-step activations; it stays out of the
tree, and the parity numbers derived from it are quoted in the per-run README
instead.

## Audit at the time of writing

No PNG, audio or video evidence file was tracked anywhere in this repository
before this directory existed, so nothing needed to be moved or deduplicated.
The remaining honest gaps are called out per bundle. In particular, the Stable
Diffusion image is retained reference output rather than a separately saved
runtime render, and the Whisper run did not retain performance timing.

## Adding a new entry

1. Create `<workflow-slug>/` and copy the real output artifact into it
   unmodified. Verify with `sha256sum` that the committed copy is byte-identical
   to the file the run wrote.
2. Copy the metrics file the harness emitted rather than retyping the numbers.
3. Write `<workflow-slug>/README.md` describing the model id, the exact
   revision, dtype, hardware, execution provider, the request that produced the
   output, and how the metrics were computed.
4. Add one row to the index above. Leave `Performance` as **Pending** unless the
   run actually measured it.
