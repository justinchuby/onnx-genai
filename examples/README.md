# Examples

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
