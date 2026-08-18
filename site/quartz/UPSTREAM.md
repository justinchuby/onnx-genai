# Quartz engine

This directory vendors [Quartz](https://github.com/jackyzha0/quartz) v5.0.0 at
commit `ab346fa66a895e12d63a308e70ce330ba795822a`.

The repository keeps the engine source, `package-lock.json`, and
`quartz.lock.json` together so builds never follow a moving Quartz branch or
plugin revision. `quartz.config.yaml` and the `wiki:*` package scripts are the
onnx-genai integration layer. The wiki content remains exclusively in
`../../wiki/`.

The engine includes the upstream `data-basepath` render behavior from Quartz
commit `075afd3f712da0088a07f5284a7b3aba37dd61b6`. This small backport lets newer
pinned plugins resolve runtime navigation beneath GitHub Pages project paths
without otherwise moving the v5.0.0 engine.

The pinned Graph plugin names floating-major jsDelivr builds of D3 and PixiJS.
`scripts/integrate-graph-runtime.mjs` replaces those URLs in the verified plugin
build with project-local URLs and uses the root, lockfile-pinned `d3@7.9.0`,
`pixi.js@8.19.0`, and existing esbuild dependency to produce deterministic ESM
bundles that install the globals required by Graph. The integration also marks
Graph's generated dependency loaders as module scripts. Quartz copies the
generated files from `quartz/static/vendor/` into the published
`static/vendor/` directory. The ignored plugin checkout remains disposable:
the integration is reapplied after every plugin install, including clean CI
rebuilds.

The integration also emits `quartz/static/vendor/manifest.json`: a
deterministic record of each vendor asset's package, exact version, deployed
path/URL, and a sha256 of the bytes actually written to disk. It is the
single source of truth `scripts/audit-runtime-resources.mjs` and
`../scripts/validate_site.py` cross-check against -- so the manifest can
never drift from, or be hand-edited independently of, the bytes actually
shipped.

`scripts/audit-runtime-resources.mjs` performs the non-vacuous runtime/plugin
presence audit `validate_site.py` relies on. It is rooted in the generated
HTML rather than in "every emitted `*.js` file": the `<script src>` tags and
inline `<script>` bodies of the generated pages are the only entry points.
From those roots it parses reachable code with `acorn` into a real AST and
follows reachable static imports, reachable dynamic `import()` calls, and
reachable calls to validated local script loaders. JavaScript under `public/`
that is never reached from HTML is reported as an ignored file and can never
contribute Search, Graph, Explorer, loader, vendor, or `fetchData` evidence.
Within the reachable graph the audit prunes provably unreachable/dead
branches, and only treats a surface as present when something real and
reachable depends on it -- a comment, a discarded `querySelector(...)` result,
an unreferenced decoy bundle, or dead `if (false) {...}` placeholder code
cannot satisfy any of these checks.

Loader and vendor edges are keyed to **lexical binding identity**, not to
function names. Minified bundles reuse short names such as `d` in many
unrelated scopes, so the audit resolves an identifier call target to the AST
node of the binding actually in scope at the call site and credits only that
node's validated loader shape. There is no name-keyed fallback: a same-name
function in another scope, or a parameter that shadows a real loader, cannot
inherit validated loader behavior, in reachable or in dead code.

## Third-party runtime resource policy

`../scripts/validate_site.py` governs two distinct third-party surfaces:

- **Executable CDN references.** jsDelivr serves the same mutable package
  namespace from several hostnames (`cdn.jsdelivr.net`, `fastly.`, `gcore.`,
  `testingcf.`, ...) and as ESM through `esm.run`, so hostnames are
  canonicalized (lowercased, trailing DNS root dot stripped) before policy
  matching and every jsDelivr executable host is in scope. Detection covers
  absolute and scheme-relative forms, `/npm/`, `/gh/` and `/wp/` namespaces,
  query/fragment variants, and raw, decoded or double-encoded dot segments
  (rejected before URL normalization could erase a package escape).
  Allowlisting is deliberately narrower than detection: only the exact
  canonical host spelling `cdn.jsdelivr.net`, the `/npm/` namespace, an exact
  immutable semver, and a reviewed asset path are approved. Today that is a
  single entry, the `latex` plugin's `katex@0.16.11/dist/contrib/copy-tex.min.js`.
  Alternate jsDelivr mirrors, `esm.run`, uppercase or trailing-dot host
  variants never inherit it.
- **Third-party stylesheets.** The `latex` plugin also emits an external
  `<link rel="stylesheet">` to KaTeX's CSS. Stylesheet links are _not_ silently
  outside this control: they are validated against their own exact-URL
  allowlist (`STYLESHEET_ALLOWLIST`), so any other off-origin stylesheet --
  including a different KaTeX version, a jsDelivr mirror, or an unrelated CDN --
  fails the build. Same-origin stylesheets are covered by the existing
  internal link/asset existence checks. Non-loading hints such as
  `<link rel="preconnect">` are outside both policies by design.

To update Quartz, replace the vendored engine from a reviewed upstream tag,
retain the integration files, refresh both lockfiles, and run:

```bash
npm ci
npm run check
npm run wiki:build
```
