# Qwen4-Exp PLE reference evidence

S12 uses Qwen3.8-Flash-Next only as a public conformance fixture for the generic
`onnx-genai.token-context@1` component/state path. Dispatch is based on that
versioned contract and typed ports, never the model or component name.

## Pinned sources

- Official release repository:
  [`QwenLM/Qwen3.8-Flash-Next@69885871`](https://github.com/QwenLM/Qwen3.8-Flash-Next/tree/69885871a64393807d988b27b1b5e380e8f28526)
- Official checkpoint configuration:
  [`Qwen/Qwen3.8-Flash-Next@de4b8e4d`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)
- Pinned Transformers configuration-class defaults:
  [`Qwen4ExpTextConfig@fc5c5bde`](https://github.com/huggingface/transformers/blob/fc5c5bde8e656dad91cbf34e61940d984b1c7b91/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py#L147-L157)
- Public Qwen4-Exp equations used by the checkpoint:
  [`huggingface/transformers@fc5c5bde`](https://github.com/huggingface/transformers/blob/fc5c5bde8e656dad91cbf34e61940d984b1c7b91/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1048-L1260)

The pinned configuration declares vocabulary `248320`, n-gram size `3`, eight
heads per n-gram order, four gated-residual streams, PLE at one-indexed layer
`2`, PLE width `2560`, kernel size `4`, dilation equal to n-gram size (`3`),
base table size `20000000`, and 48 text layers. Together with the pinned
Transformers configuration-class default, the checkpoint configuration
resolves the seed to `1234`.

## Evidence boundary

`generate_qwen4_exp_ple_reference.py` independently expresses the published
token-history/EOS reset, SplitMix-derived hash multipliers, per-head prime
modulo and offsets, learned lookup, key/value projections, grouped RMS
normalization followed by each stream's learned `1 + weight` scale, signed-root
sigmoid gate, depthwise dilated convolution, and residual injection. Distinct
deterministic synthetic scales cover `norm_key`, `norm_query`, and `norm_conv`.
It emits checked-in vectors for full, two-chunk, and single-token decode
boundaries. Serialization sorts object keys, rejects non-finite values, encodes
UTF-8 explicitly, and writes canonical LF bytes without platform text-newline
translation. Regenerate from the repository root with:

```shell
python3 crates/onnx-genai-engine/tests/fixtures/generate_qwen4_exp_ple_reference.py \
  --output crates/onnx-genai-engine/tests/fixtures/qwen4_exp_ple_reference.json
```

The generator's `--check` mode compares exact bytes.

The vectors use deterministic synthetic weights and reduced table/hidden
geometry so they remain small and hermetic. They establish **reference
equations/config conformance with synthetic weights**, not parity with official
checkpoint weights. The Rust test feeds those weights and vector inputs through
the production ONNX workflow; the Python generator does not call the Rust graph
builder or runtime.

The same engine test retains a structurally different alternate geometry,
transaction abort/retry, semantic fork, checkpoint/restore, repeated/reordered/
shrunk row plans, padding validity, and release coverage. The reference package
also renames the component, removes its optional grouping budget, and loads
under an explicit CPU placement policy to prove those policy/name choices do
not select the token-context implementation.
