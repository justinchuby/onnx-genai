# Whisper Tiny — speech-to-text evidence

[`input.wav`](input.wav) is the real 1.6-second, 16 kHz input used for this
run. [`metrics.json`](metrics.json) records HuggingFace and ONNX results for
both this clip and a 9.115-second clip.

For the committed input, both implementations produced exactly:

```text
 Call of Flower, Man.
```

The token sequences also match before the ONNX end-of-sequence token. Mel
features differ by at most `2.44379e-05`; encoder hidden states differ by at
most `0.00156689` with relative L2 error `6.57501e-06`.

This bundle proves real audio preprocessing, encoder execution, autoregressive
decoding, and text emission. The run did not record defensible end-to-end
latency, so performance is intentionally not claimed.

