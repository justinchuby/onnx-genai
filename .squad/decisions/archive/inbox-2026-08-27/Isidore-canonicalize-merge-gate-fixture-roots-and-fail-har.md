### 2026-08-26T16-10-15: Canonicalize merge-gate fixture roots and fail harness reads closed
**By:** Isidore
**What:** Canonicalize merge-gate fixture roots and fail harness reads closed
**References:** PR #2223, scripts/merge_when_green_test.sh, PR #2179 closure rejection
**Why:** The merge_when_green self-test now roots temp state absolutely, converts Windows CARGO_TARGET_TMPDIR paths via cygpath, avoids nested fixture-root side effects, and treats missing fixture files as explicit harness errors before the production gate runs. Production merge behavior remains unchanged.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
