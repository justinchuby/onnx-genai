# Examples

## ComfyUI import, converted and executed

[`comfyui-import/`](comfyui-import/) walks a ComfyUI API-format workflow through
the one-way importer into canonical `pipeline.workflow` metadata and runs it on
the generic workflow engine, with the fixture package and refusal diagnostics.

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
