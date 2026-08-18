# Quartz engine

This directory vendors [Quartz](https://github.com/jackyzha0/quartz) v5.0.0 at
commit `ab346fa66a895e12d63a308e70ce330ba795822a`.

The repository keeps the engine source, `package-lock.json`, and
`quartz.lock.json` together so builds never follow a moving Quartz branch or
plugin revision. `quartz.config.yaml` and the `wiki:*` package scripts are the
onnx-genai integration layer. The wiki content remains exclusively in
`../../wiki/`.

To update Quartz, replace the vendored engine from a reviewed upstream tag,
retain the integration files, refresh both lockfiles, and run:

```bash
npm ci
npm run check
npm run wiki:build
```
