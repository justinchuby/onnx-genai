### 2026-08-26T08-09-53: Classify ONNX fixture paths before generic docs paths
**By:** Isidore
**What:** Classify ONNX fixture paths before generic docs paths
**References:** PR #2182, .github/scripts/ci_change_scope.sh, .github/workflows/ci.yml
**Why:** CI change-scope now treats *.onnx and *.textproto as code/test-significant before docs/* handling. The shared predicate self-tests Linux/Windows forms and fails closed to full CI if unavailable, preserving ordinary Markdown docs-only behavior while ensuring the Fast fixture census cannot be skipped.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
