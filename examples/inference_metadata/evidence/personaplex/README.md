# PersonaPlex — full-duplex speech evidence

This is a real 123-step, self-driven CUDA run of the PersonaPlex workflow.
[`reference.wav`](reference.wav) is the upstream waveform and
[`runtime.wav`](runtime.wav) is the ONNX workflow output.

| Metric | Result |
| --- | ---: |
| Output audio | 92,160 samples |
| Waveform correlation | **0.9999999999997841** |
| Maximum absolute waveform error | `5.36442e-07` |
| Minimum per-frame waveform correlation | `0.9999995869` |
| Text argmax agreement | 123 / 123 steps |
| User-code agreement | 48 / 48 |
| Output-code agreement | 48 / 48 |
| End-to-end step latency p50 / p90 | 1,587.60 / 2,493.36 ms |
| Total wall time | 89.00 s |

[`metrics.json`](metrics.json) is the complete per-step record, including
encoder, temporal, depformer, decoder, and LM-only timing. The audio files are
committed so reviewers can listen to both outputs rather than infer audio
quality from a scalar metric.

