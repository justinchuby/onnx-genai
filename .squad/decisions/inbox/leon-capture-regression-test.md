# Default Attention staged-KV capture regression

The CUDA EP now warms persistent scratch for single-token default-domain
Attention decode when dense present K/V outputs alias their growing past K/V
inputs. This makes the staged disjoint KV copy-back recordable in a CUDA graph.

`standard_attention_capture_gpu` compares three captured aliased decode steps
against eager output and K/V cache state. Replacing the two stream-ordered
`dtod_async` copy-backs with synchronous `dtod` fails during capture with
`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.
