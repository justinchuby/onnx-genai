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
presence audit `validate_site.py` relies on: it parses every emitted `*.js`
bundle (and every inline `<script>` body) with `acorn` into a real AST, prunes
provably unreachable/dead branches, and only treats Search, Graph, Explorer,
the local ESM script-loader, Graph's vendor import edges, and the shared
inline `fetchData` surface as present when something real and reachable
depends on them -- a comment, a discarded `querySelector(...)` result, or dead
`if (false) {...}` placeholder code cannot satisfy any of these checks.

To update Quartz, replace the vendored engine from a reviewed upstream tag,
retain the integration files, refresh both lockfiles, and run:

```bash
npm ci
npm run check
npm run wiki:build
```
