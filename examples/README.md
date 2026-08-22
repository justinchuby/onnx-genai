# Examples

## Inference metadata visualizer

Open [`inference_metadata/visualizer.html`](inference_metadata/visualizer.html)
directly from disk to inspect YAML or JSON inference metadata. The responsive
single-page viewer includes workflow, state/serving, media, advanced capability,
diagnostics, and recursive full-document views. YAML parsing stays embedded;
workflow graph rendering loads pinned Mermaid 11.12.0 from jsDelivr with
Subresource Integrity, and falls back to safe graph text when it is unavailable.

## ComfyUI import, converted and executed

[`comfyui-import/`](comfyui-import/) walks a ComfyUI API-format workflow through
the one-way importer into canonical `pipeline.workflow` metadata and runs it on
the generic workflow engine, with the fixture package and refusal diagnostics.

## End-to-end evidence

[`inference_metadata/evidence/`](inference_metadata/evidence/) tracks the actual
output artifacts of real end-to-end inference-metadata runs — images, metrics
and run configuration — so the numbers quoted in reviews stay verifiable from
the tree instead of from an expiring pull-request attachment. See
[`inference_metadata/evidence/README.md`](inference_metadata/evidence/README.md)
for the index and for what does and does not belong there.

## Profile reports

[`profiles/`](profiles/) holds real `--profile` output for the same model on two
execution providers, with the numbers read back in
[`profiles/README.md`](profiles/README.md).

## Decode timeline traces

Perfetto/Chrome decode timelines live in [`traces/`](traces/) — see
[`traces/README.md`](traces/README.md) for how to view and regenerate them.
