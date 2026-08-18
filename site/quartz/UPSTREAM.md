# Quartz engine

This directory vendors [Quartz](https://github.com/jackyzha0/quartz) v5.0.0 at
commit `ab346fa66a895e12d63a308e70ce330ba795822a`.

The engine source, `package-lock.json`, and `quartz.lock.json` are committed
together. Builds therefore use reviewed Quartz and plugin commits rather than a
moving upstream branch. `quartz.config.yaml` and the `wiki:*` package scripts
are the onnx-genai integration layer; `../../wiki/` remains the only authoring
source.

The engine includes the upstream `data-basepath` render behavior from Quartz
commit `075afd3f712da0088a07f5284a7b3aba37dd61b6`. The pinned Search, Graph, and
Explorer revisions use that value for project-site navigation.

The pinned Graph plugin names floating-major D3 and PixiJS URLs. After every
clean plugin build, `../scripts/pin_graph_runtime.py` replaces those two known
strings with exact `d3@7.9.0` and `pixi.js@8.19.0` URLs. The production
validator then accepts only four reviewed full external runtime URLs: those two
scripts plus the exact KaTeX 0.16.11 script and stylesheet emitted by the
pinned Latex plugin.

Validation intentionally stays modest. It checks source wikilinks, repository
targets, generated internal links/assets, `/onnx-genai/` body base paths,
HTML-referenced local resources, the rendered root landing page and separate
README note, origin-root runtime URLs, and the exact external script/stylesheet
allowlist. A successful clean Quartz build and exact dependency/plugin pins are
trusted for plugin behavior; the repository does not maintain a custom
JavaScript semantic analyzer or attempt to prove third-party internals.

To update Quartz or its plugins, review the new immutable commits, refresh the
lockfiles and any exact runtime URL replacements, then run:

```bash
npm ci
npm run check
npm run wiki:build
```
