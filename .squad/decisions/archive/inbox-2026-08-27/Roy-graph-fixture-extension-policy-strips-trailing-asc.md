### 2026-08-26T11-57-23: Graph fixture extension policy strips trailing ASCII whitespace only
**By:** Roy
**What:** Graph fixture extension policy strips trailing ASCII whitespace only
**References:** PR #2182, branch fix/production-test-ci-honesty
**Why:** WHAT: CI graph-extension classification normalizes slashes, preserves that exact normalized path for equality/diagnostics, strips only trailing ASCII space/tab/CR/LF/VT/FF on a classification copy, rejects empty terminal names, ASCII-lowercases, then inspects the terminal extension. Interior whitespace and trailing dots are not normalized. Bash owns the documented contract; the Rust census mirrors it and a Linux parity check executes both. WHY: Linux/Git permit trailing-whitespace filenames, which let binary ONNX fixtures bypass docs scope and the census. Keeping equality on the untrimmed normalized path avoids changing embedded-source identity while closing only the extension bypass.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
