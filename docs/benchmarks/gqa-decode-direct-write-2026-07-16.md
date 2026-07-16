# GQA decode direct-write fast path

## Setup

- Model: `qwen2.5-0.5b-int4-onnx`
- CPU threads: `RAYON_NUM_THREADS=24`
- Prompt: `Hello` (`[9707]`)
- Generation: 8 tokens, 1 warmup, 5 measured runs
- Profiling: `ONNX_GENAI_PROFILE_OPS=1`

The GQA timing is the mean of the 40 measured generation steps. Warmup samples
were excluded.

## Result

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| GroupQueryAttention | 0.865 ms/step | 0.690 ms/step | -20.2% |
| GroupQueryAttention share | 4.90% | 4.08% | -0.82 pp |
| Decode throughput | 54.38 tok/s | 58.44 tok/s | +7.5% |

Generated token IDs matched exactly:
`[11576, 42740, 11, 358, 614, 264, 3405, 911]`.

## Change

For the single-token Q/K decode path only, contiguous f32 GQA outputs are copied
directly into their output tensors. This avoids allocating a second narrowing
buffer and walking a generic strided index for the attention and present K/V
outputs. Non-f32, strided, and prefill outputs retain the existing writer.
