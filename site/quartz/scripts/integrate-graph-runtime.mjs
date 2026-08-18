import { readFile, readdir, rm, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { createHash } from "node:crypto"
import { build } from "esbuild"

const dependencies = {
  d3: {
    version: "7.9.0",
    entry: 'import * as d3 from "d3"; globalThis.d3 = d3; export {};',
    upstream: "https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js",
  },
  "pixi.js": {
    version: "8.19.0",
    entry:
      'import { Application, Container, Graphics, Text } from "pixi.js"; ' +
      "globalThis.PIXI = { Application, Container, Graphics, Text }; export {};",
    upstream: "https://cdn.jsdelivr.net/npm/pixi.js@8/dist/pixi.js",
  },
}

const siteBasePath = "/onnx-genai"
const graphDist = join(".quartz", "plugins", "graph", "dist")
const vendorDirectory = join("quartz", "static", "vendor")

async function installedVersion(packageName) {
  const manifest = JSON.parse(
    await readFile(join("node_modules", packageName, "package.json"), "utf8"),
  )
  return manifest.version
}

const manifestPath = join(vendorDirectory, "manifest.json")

async function sha256(path) {
  const contents = await readFile(path)
  return createHash("sha256").update(contents).digest("hex")
}

async function bundleDependencies() {
  await rm(vendorDirectory, { recursive: true, force: true })
  const assets = []
  for (const [packageName, dependency] of Object.entries(dependencies)) {
    const actualVersion = await installedVersion(packageName)
    if (actualVersion !== dependency.version) {
      throw new Error(
        `${packageName}: expected package-lock version ${dependency.version}, found ${actualVersion}`,
      )
    }

    const fileName = `${packageName.replace(".", "-")}-${dependency.version}.esm.js`
    const outfile = join(vendorDirectory, fileName)
    await build({
      absWorkingDir: process.cwd(),
      bundle: true,
      charset: "utf8",
      format: "esm",
      legalComments: "none",
      minify: true,
      outfile,
      platform: "browser",
      stdin: {
        contents: dependency.entry,
        loader: "js",
        resolveDir: process.cwd(),
      },
      target: ["chrome109", "edge115", "firefox102", "safari15.6"],
      write: true,
    })

    const relativeFile = `static/vendor/${fileName}`
    assets.push({
      package: packageName,
      version: dependency.version,
      file: relativeFile,
      url: `${siteBasePath}/${relativeFile}`,
      sha256: await sha256(outfile),
    })
  }

  // Deterministic manifest: this is the single source of truth the runtime
  // resource audit (site/quartz/scripts/audit-runtime-resources.mjs) and
  // site/scripts/validate_site.py cross-check against emitted bytes on disk,
  // so a hand-edited or commented-out signature cannot counterfeit it.
  assets.sort((a, b) => a.package.localeCompare(b.package))
  const manifest = {
    generator: "site/quartz/scripts/integrate-graph-runtime.mjs",
    basePath: siteBasePath,
    assets,
  }
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`)
}

async function rewriteGraphRuntime() {
  const files = (await readdir(graphDist, { recursive: true }))
    .filter((file) => file.endsWith(".js"))
    .map((file) => join(graphDist, file))
  const classicLoader = 'c.src=o,c.crossOrigin="anonymous"'
  const moduleLoader = 'c.src=o,c.type="module",c.crossOrigin="anonymous"'
  let classicLoaderCount = 0
  let moduleLoaderCount = 0

  for (const file of files) {
    const source = await readFile(file, "utf8")
    classicLoaderCount += source.split(classicLoader).length - 1
    moduleLoaderCount += source.split(moduleLoader).length - 1
  }
  if (classicLoaderCount === 2 && moduleLoaderCount === 0) {
    for (const file of files) {
      const source = await readFile(file, "utf8")
      const rewritten = source.replaceAll(classicLoader, moduleLoader)
      if (rewritten !== source) {
        await writeFile(file, rewritten)
      }
    }
  } else if (classicLoaderCount !== 0 || moduleLoaderCount !== 2) {
    throw new Error(
      `Graph: expected two script loaders, found ${classicLoaderCount} classic and ` +
        `${moduleLoaderCount} module`,
    )
  }

  for (const [packageName, dependency] of Object.entries(dependencies)) {
    const localPath = `${siteBasePath}/static/vendor/${packageName.replace(".", "-")}-${dependency.version}.esm.js`
    let upstreamCount = 0
    let localCount = 0
    const sources = new Map()

    for (const file of files) {
      const source = await readFile(file, "utf8")
      sources.set(file, source)
      upstreamCount += source.split(dependency.upstream).length - 1
      localCount += source.split(localPath).length - 1
    }

    if (upstreamCount === 2 && localCount === 0) {
      for (const [file, source] of sources) {
        const rewritten = source.replaceAll(dependency.upstream, localPath)
        if (rewritten !== source) {
          await writeFile(file, rewritten)
        }
      }
    } else if (upstreamCount !== 0 || localCount !== 2) {
      throw new Error(
        `${packageName}: expected two pinned Graph runtime imports, found ` +
          `${upstreamCount} upstream and ${localCount} local`,
      )
    }
  }
}

await bundleDependencies()
await rewriteGraphRuntime()
console.log("Bundled d3 7.9.0 and pixi.js 8.19.0 locally and integrated the pinned Graph runtime.")
