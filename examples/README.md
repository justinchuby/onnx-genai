# Examples

## Real multimodal model contract

[`smolvlm-256m/`](smolvlm-256m/) validates the inference-metadata contract
against Hugging Face's real SmolVLM-256M ONNX export and records the precise
VLM fit/gap analysis.

## Profile reports

[`profiles/`](profiles/) holds real `--profile` output for the same model on two
execution providers, with the numbers read back in
[`profiles/README.md`](profiles/README.md).

## Decode timeline traces

Perfetto/Chrome decode timelines live in [`traces/`](traces/) — see
[`traces/README.md`](traces/README.md) for how to view and regenerate them.
