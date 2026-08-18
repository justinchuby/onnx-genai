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

To update Quartz, replace the vendored engine from a reviewed upstream tag,
retain the integration files, refresh both lockfiles, and run:

```bash
npm ci
npm run check
npm run wiki:build
```
