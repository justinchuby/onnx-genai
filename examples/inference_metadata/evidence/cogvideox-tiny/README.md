# CogVideoX — text-to-video evidence

This bundle uses
[`finetrainers/dummy-cogvideox`](https://huggingface.co/finetrainers/dummy-cogvideox)
at revision `bed7eacdd51ae4514f7d687147925c13024a2de3`. It is deliberately
tiny (`16x16`, 17 frames, three DDIM steps), but executes the real
text-encoder/transformer/scheduler/VAE workflow.

Prompt: `a red cube slowly rotating on a wooden table`; seed: `1234`.

![Five frames from the generated clip](contact-sheet.png)

[`output.gif`](output.gif) is the complete 17-frame result, nearest-neighbour
upscaled to make the actual generated pixels inspectable.

| Runtime vs reference | Result |
| --- | ---: |
| Correlation | **0.9999999983** |
| PSNR | **84.66 dB** |
| Mean absolute error | `6.14097e-06` |
| Maximum absolute error | `7.50907e-05` |
| Runtime load | 8,270.09 ms |
| Runtime workflow execution | 174.36 ms |

[`runtime-parity.json`](runtime-parity.json), [`export-parity.json`](export-parity.json),
and [`run.json`](run.json) are the unmodified harness outputs.

