# Examples

## Metadata execution status

[`METADATA_STATUS.md`](METADATA_STATUS.md) records which pipeline shapes the
metadata contract can describe, which ones execute end to end today, and the
remaining engine/CPU-EP gaps.

## Real multimodal model contract

[`smolvlm-256m/`](smolvlm-256m/) validates the inference-metadata contract
against Hugging Face's real SmolVLM-256M ONNX export and records the precise
VLM fit/gap analysis.

## Decode timeline traces

Perfetto/Chrome decode timelines live in [`traces/`](traces/) — see
[`traces/README.md`](traces/README.md) for how to view and regenerate them.
