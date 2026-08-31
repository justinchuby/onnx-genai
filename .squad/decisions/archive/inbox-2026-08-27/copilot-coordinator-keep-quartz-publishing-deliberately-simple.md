### 2026-08-18T16-36-25: Keep Quartz publishing deliberately simple
**By:** copilot-coordinator
**What:** Keep Quartz publishing deliberately simple
**References:** #1190, #1210, #1211
**Why:** For PR #1210 and future wiki publishing, prioritize minimal custom code and maintainability over adversarial verification of pinned third-party plugin internals. Exact-pinned Quartz/plugins plus successful deterministic production builds are trusted. Custom validation should remain small: source wikilinks/repository targets, generated internal links/assets/base path, real root landing page, and a tiny literal allowlist for immutable external runtime JS/CSS URLs. Remove Acorn AST/dataflow/resource-graph auditing, large adversarial matrices, and local vendor-manifest/rewrite machinery when exact pinned external URLs are simpler. State the narrower guarantee honestly; this is a documentation site, not a security sandbox.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
