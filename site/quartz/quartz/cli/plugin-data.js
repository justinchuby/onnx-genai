import fs from "fs"
import path from "path"
import { execSync } from "child_process"
import YAML from "yaml"

const LOCKFILE_PATH = path.join(process.cwd(), "quartz.lock.json")
const PLUGINS_DIR = path.join(process.cwd(), ".quartz", "plugins")
const CONFIG_YAML_PATH = path.join(process.cwd(), "quartz.config.yaml")
const DEFAULT_CONFIG_YAML_PATH = path.join(process.cwd(), "quartz.config.default.yaml")
const TEMPLATES_DIR = path.join(process.cwd(), "quartz", "cli", "templates")

const LEGACY_PLUGINS_JSON_PATH = path.join(process.cwd(), "quartz.plugins.json")
const LEGACY_DEFAULT_PLUGINS_JSON_PATH = path.join(process.cwd(), "quartz.plugins.default.json")

function resolveConfigPath() {
  if (fs.existsSync(CONFIG_YAML_PATH)) return CONFIG_YAML_PATH
  if (fs.existsSync(LEGACY_PLUGINS_JSON_PATH)) return LEGACY_PLUGINS_JSON_PATH
  if (fs.existsSync(DEFAULT_CONFIG_YAML_PATH)) return DEFAULT_CONFIG_YAML_PATH
  if (fs.existsSync(LEGACY_DEFAULT_PLUGINS_JSON_PATH)) return LEGACY_DEFAULT_PLUGINS_JSON_PATH
  return CONFIG_YAML_PATH
}

function resolveDefaultConfigPath() {
  if (fs.existsSync(DEFAULT_CONFIG_YAML_PATH)) return DEFAULT_CONFIG_YAML_PATH
  if (fs.existsSync(LEGACY_DEFAULT_PLUGINS_JSON_PATH)) return LEGACY_DEFAULT_PLUGINS_JSON_PATH
  return DEFAULT_CONFIG_YAML_PATH
}

function readFileAsData(filePath) {
  if (!fs.existsSync(filePath)) return null
  try {
    const raw = fs.readFileSync(filePath, "utf-8")
    if (filePath.endsWith(".yaml") || filePath.endsWith(".yml")) {
      return YAML.parse(raw)
    }
    return JSON.parse(raw)
  } catch {
    return null
  }
}

function writeDataToFile(filePath, data) {
  if (filePath.endsWith(".yaml") || filePath.endsWith(".yml")) {
    const header = "# yaml-language-server: $schema=./quartz/plugins/quartz-plugins.schema.json\n"
    fs.writeFileSync(filePath, header + YAML.stringify(data, { lineWidth: 120 }))
  } else {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2) + "\n")
  }
}

export function readPluginsJson() {
  const configPath = resolveConfigPath()
  return readFileAsData(configPath)
}

export function writePluginsJson(data) {
  const { $schema, ...rest } = data
  writeDataToFile(CONFIG_YAML_PATH, rest)
}

export function readDefaultPluginsJson() {
  const defaultPath = resolveDefaultConfigPath()
  return readFileAsData(defaultPath)
}

export function readLockfile() {
  if (!fs.existsSync(LOCKFILE_PATH)) return null
  try {
    return JSON.parse(fs.readFileSync(LOCKFILE_PATH, "utf-8"))
  } catch {
    return null
  }
}

export function writeLockfile(lockfile) {
  if (lockfile.plugins) {
    const sorted = {}
    for (const key of Object.keys(lockfile.plugins).sort()) {
      sorted[key] = lockfile.plugins[key]
    }
    lockfile = { ...lockfile, plugins: sorted }
  }
  fs.writeFileSync(LOCKFILE_PATH, JSON.stringify(lockfile, null, 2) + "\n")
}

export function isLocalSource(source) {
  if (source.startsWith("./") || source.startsWith("../") || source.startsWith("/")) {
    return true
  }
  // Windows absolute paths (e.g. C:\ or D:/)
  if (/^[A-Za-z]:[\\/]/.test(source)) {
    return true
  }
  return false
}
export function extractPluginName(source) {
  if (isLocalSource(source)) {
    return path.basename(source.replace(/[\/]+$/, ""))
  }
  if (source.startsWith("github:")) {
    const withoutPrefix = source.replace("github:", "")
    const [repoPath] = withoutPrefix.split("#")
    const parts = repoPath.split("/")
    return parts[parts.length - 1]
  }
  if (source.startsWith("git+") || source.startsWith("https://")) {
    const url = source.replace("git+", "")
    const match = url.match(/\/([^/]+?)(?:\.git)?(?:#|$)/)
    return match?.[1] ?? source
  }
  return source
}

export function readManifestFromPackageJson(pluginDir) {
  const pkgPath = path.join(pluginDir, "package.json")
  if (!fs.existsSync(pkgPath)) return null
  try {
    const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf-8"))
    return pkg.quartz ?? null
  } catch {
    return null
  }
}

export function parseGitSource(source) {
  if (isLocalSource(source)) {
    const resolved = path.resolve(source)
    const name = path.basename(resolved)
    return { name, url: resolved, ref: undefined, local: true }
  }
  if (source.startsWith("github:")) {
    const [repoPath, ref] = source.replace("github:", "").split("#")
    const [owner, repo] = repoPath.split("/")
    return { name: repo, url: `https://github.com/${owner}/${repo}.git`, ref }
  }
  if (source.startsWith("git+")) {
    const raw = source.replace("git+", "")
    const [url, ref] = raw.split("#")
    const name = path.basename(url, ".git")
    return { name, url, ref }
  }
  if (source.startsWith("https://")) {
    const [url, ref] = source.split("#")
    const name = path.basename(url, ".git")
    return { name, url, ref }
  }
  throw new Error(`Cannot parse plugin source: ${source}`)
}

export function getGitCommit(pluginDir) {
  try {
    return execSync("git rev-parse HEAD", { cwd: pluginDir, encoding: "utf-8" }).trim()
  } catch {
    return "unknown"
  }
}

export function getPluginDir(name) {
  return path.join(PLUGINS_DIR, name)
}

export function pluginDirExists(name) {
  return fs.existsSync(path.join(PLUGINS_DIR, name))
}

export function ensurePluginsDir() {
  if (!fs.existsSync(PLUGINS_DIR)) {
    fs.mkdirSync(PLUGINS_DIR, { recursive: true })
  }
}

/**
 * Merges quartz.config.yaml, quartz.lock.json, and on-disk manifest data
 * into enriched plugin entries with: name, displayName, source, enabled,
 * options, order, layout, category, installed, locked, manifest,
 * currentCommit, modified.
 */
export function getEnrichedPlugins() {
  const pluginsJson = readPluginsJson()
  const lockfile = readLockfile()

  if (!pluginsJson?.plugins) return []

  return pluginsJson.plugins.map((entry, index) => {
    const name = extractPluginName(entry.source)
    const pluginDir = path.join(PLUGINS_DIR, name)
    const installed = fs.existsSync(pluginDir)
    const locked = lockfile?.plugins?.[name] ?? null
    const manifest = installed ? readManifestFromPackageJson(pluginDir) : null
    const currentCommit = installed ? getGitCommit(pluginDir) : null
    const modified = locked && currentCommit ? currentCommit !== locked.commit : false

    return {
      index,
      name,
      displayName: manifest?.displayName ?? name,
      source: entry.source,
      enabled: entry.enabled ?? true,
      options: entry.options ?? {},
      order: entry.order ?? 50,
      layout: entry.layout ?? null,
      category: manifest?.category ?? "unknown",
      installed,
      locked,
      manifest,
      currentCommit,
      modified,
    }
  })
}

export function getLayoutConfig() {
  const pluginsJson = readPluginsJson()
  return pluginsJson?.layout ?? null
}

export function getGlobalConfig() {
  const pluginsJson = readPluginsJson()
  return pluginsJson?.configuration ?? null
}

export function updatePluginEntry(index, updates) {
  const json = readPluginsJson()
  if (!json?.plugins?.[index]) return false
  Object.assign(json.plugins[index], updates)
  writePluginsJson(json)
  return true
}

export function updateGlobalConfig(updates) {
  const json = readPluginsJson()
  if (!json) return false
  json.configuration = { ...json.configuration, ...updates }
  writePluginsJson(json)
  return true
}

export function updateLayoutConfig(layout) {
  const json = readPluginsJson()
  if (!json) return false
  json.layout = layout
  writePluginsJson(json)
  return true
}

export function reorderPlugin(fromIndex, toIndex) {
  const json = readPluginsJson()
  if (!json?.plugins) return false
  const [moved] = json.plugins.splice(fromIndex, 1)
  json.plugins.splice(toIndex, 0, moved)
  writePluginsJson(json)
  return true
}

export function removePluginEntry(index) {
  const json = readPluginsJson()
  if (!json?.plugins?.[index]) return false
  json.plugins.splice(index, 1)
  writePluginsJson(json)
  return true
}

export function addPluginEntry(entry) {
  const json = readPluginsJson()
  if (!json) return false
  if (!json.plugins) json.plugins = []
  json.plugins.push(entry)
  writePluginsJson(json)
  return true
}

export function configExists() {
  return fs.existsSync(CONFIG_YAML_PATH) || fs.existsSync(LEGACY_PLUGINS_JSON_PATH)
}

export function createConfigFromDefault() {
  const defaultData = readDefaultPluginsJson()
  if (!defaultData) {
    // No default available — create minimal config
    const minimal = {
      configuration: {
        pageTitle: "Quartz",
        enableSPA: true,
        enablePopovers: true,
        analytics: { provider: "plausible" },
        locale: "en-US",
        baseUrl: "quartz.jzhao.xyz",
        ignorePatterns: ["private", "templates", ".obsidian"],
        defaultDateType: "created",
        theme: {
          cdnCaching: true,
          typography: {
            header: "Schibsted Grotesk",
            body: "Source Sans Pro",
            code: "IBM Plex Mono",
          },
          colors: {
            lightMode: {
              light: "#faf8f8",
              lightgray: "#e5e5e5",
              gray: "#b8b8b8",
              darkgray: "#4e4e4e",
              dark: "#2b2b2b",
              secondary: "#284b63",
              tertiary: "#84a59d",
              highlight: "rgba(143, 159, 169, 0.15)",
              textHighlight: "#fff23688",
            },
            darkMode: {
              light: "#161618",
              lightgray: "#393639",
              gray: "#646464",
              darkgray: "#d4d4d4",
              dark: "#ebebec",
              secondary: "#7b97aa",
              tertiary: "#84a59d",
              highlight: "rgba(143, 159, 169, 0.15)",
              textHighlight: "#fff23688",
            },
          },
        },
      },
      plugins: [],
      layout: { groups: {}, byPageType: {} },
    }
    writePluginsJson(minimal)
    return minimal
  }

  const { $schema, ...rest } = defaultData
  writePluginsJson(rest)
  return rest
}

const VALID_TEMPLATES = ["default", "obsidian", "ttrpg", "blog"]

export function createConfigFromTemplate(templateName) {
  if (!VALID_TEMPLATES.includes(templateName)) {
    throw new Error(
      `Unknown template: ${templateName}. Valid templates: ${VALID_TEMPLATES.join(", ")}`,
    )
  }

  const templatePath = path.join(TEMPLATES_DIR, `${templateName}.yaml`)
  const templateData = readFileAsData(templatePath)
  if (!templateData) {
    // Template file missing — fall back to default config creation
    return createConfigFromDefault()
  }

  const { $schema, ...rest } = templateData
  writePluginsJson(rest)
  return rest
}

export const PLUGINS_JSON_PATH = CONFIG_YAML_PATH
export const DEFAULT_PLUGINS_JSON_PATH = DEFAULT_CONFIG_YAML_PATH
export { LOCKFILE_PATH, PLUGINS_DIR }
