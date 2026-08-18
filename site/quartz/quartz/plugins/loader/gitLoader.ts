import fs from "fs"
import path from "path"
import git from "isomorphic-git"
import http from "isomorphic-git/http/node"
import { styleText } from "util"
import { pathToFileURL } from "url"

/**
 * Convert an absolute filesystem path to a file:// URL string for use with dynamic import().
 * On Windows, absolute paths like D:\path\file.js have "D:" interpreted as a URL protocol
 * by Node ESM, so they must be converted to file:// URLs.
 * Non-absolute paths (e.g. npm package names) are returned as-is.
 */
export function toFileUrl(filePath: string): string {
  if (path.isAbsolute(filePath)) {
    return pathToFileURL(filePath).href
  }
  return filePath
}

export interface GitPluginSpec {
  /** Plugin name (used for directory) */
  name: string
  /** Git repository URL or absolute local path */
  repo: string
  /** Git ref (branch, tag, or commit hash). Defaults to 'main' */
  ref?: string
  /** Optional subdirectory within the repo if plugin is not at root */
  subdir?: string
  /** Whether this is a local path source */
  local?: boolean
}

export type PluginInstallSource = string | GitPluginSpec

const PLUGINS_CACHE_DIR = path.join(process.cwd(), ".quartz", "plugins")

/**
 * Check if a source string refers to a local file path.
 * Local sources start with ./, ../, / or a Windows drive letter (e.g. C:\).
 */
export function isLocalSource(source: string): boolean {
  if (source.startsWith("./") || source.startsWith("../") || source.startsWith("/")) {
    return true
  }
  // Windows absolute paths (e.g. C:\ or D:/)
  if (/^[A-Za-z]:[\\/]/.test(source)) {
    return true
  }
  return false
}

/**
 * Parse a plugin source string into a GitPluginSpec
 * Supports:
 * - "./path/to/plugin" or "/absolute/path" -> local path
 * - "github:user/repo" -> https://github.com/user/repo.git
 * - "github:user/repo#ref" -> https://github.com/user/repo.git with specific ref
 * - "git+https://..." -> direct git URL
 * - "https://github.com/..." -> direct https URL
 */
export function parsePluginSource(source: string): GitPluginSpec {
  // Handle local paths
  if (isLocalSource(source)) {
    const resolved = path.resolve(source)
    const name = path.basename(resolved)
    return { name, repo: resolved, local: true }
  }

  // Handle github shorthand: github:user/repo or github:user/repo#ref
  if (source.startsWith("github:")) {
    const withoutPrefix = source.replace("github:", "")
    const [repoPath, ref] = withoutPrefix.split("#")
    const [owner, repo] = repoPath.split("/")

    if (!owner || !repo) {
      throw new Error(`Invalid GitHub source: ${source}. Expected format: github:user/repo`)
    }

    return {
      name: repo,
      repo: `https://github.com/${owner}/${repo}.git`,
      ref: ref || "main",
    }
  }

  // Handle git+https:// protocol
  if (source.startsWith("git+")) {
    const raw = source.replace("git+", "")
    const [url, ref] = raw.split("#")
    const name = extractRepoName(url)
    return { name, repo: url, ref: ref || "main" }
  }

  // Handle direct HTTPS URL (GitHub, GitLab, etc.)
  if (source.startsWith("https://")) {
    const [url, ref] = source.split("#")
    const name = extractRepoName(url)
    return { name, repo: url, ref: ref || "main" }
  }

  // Assume it's a plain repo name and try github
  const parts = source.split("/")
  if (parts.length === 2) {
    return {
      name: parts[1],
      repo: `https://github.com/${source}.git`,
      ref: "main",
    }
  }

  throw new Error(`Cannot parse plugin source: ${source}`)
}

function extractRepoName(url: string): string {
  // Extract repo name from URL like https://github.com/user/repo.git
  const match = url.match(/\/([^\/]+?)(?:\.git)?$/)
  return match ? match[1] : "unknown"
}

/**
 * Install a plugin from a Git repository, or symlink a local plugin.
 */
export async function installPlugin(
  spec: GitPluginSpec,
  options: { verbose?: boolean; force?: boolean } = {},
): Promise<string> {
  const pluginDir = path.join(PLUGINS_CACHE_DIR, spec.name)

  // Local source: symlink instead of clone
  if (spec.local) {
    if (!fs.existsSync(spec.repo)) {
      throw new Error(`Local plugin path does not exist: ${spec.repo}`)
    }

    if (!options.force && fs.existsSync(pluginDir)) {
      // Check if existing entry is already a symlink to the right place
      try {
        const stat = fs.lstatSync(pluginDir)
        if (stat.isSymbolicLink() && fs.realpathSync(pluginDir) === fs.realpathSync(spec.repo)) {
          if (options.verbose) {
            console.log(styleText("cyan", `→`), `Plugin ${spec.name} already linked`)
          }
          return pluginDir
        }
      } catch {
        // stat failed, recreate
      }
    }

    // Clean up if force reinstall or existing non-symlink entry
    if (fs.existsSync(pluginDir)) {
      const stat = fs.lstatSync(pluginDir)
      if (stat.isSymbolicLink()) {
        fs.unlinkSync(pluginDir)
      } else {
        fs.rmSync(pluginDir, { recursive: true })
      }
    }

    // Ensure parent directory exists
    const parentDir = path.dirname(pluginDir)
    if (!fs.existsSync(parentDir)) {
      fs.mkdirSync(parentDir, { recursive: true })
    }

    if (options.verbose) {
      console.log(styleText("cyan", `→`), `Linking ${spec.name} from ${spec.repo}...`)
    }

    fs.symlinkSync(spec.repo, pluginDir, "dir")

    if (options.verbose) {
      console.log(styleText("green", `✓`), `Linked ${spec.name}`)
    }

    return pluginDir
  }

  // Git source: clone
  // Check if already installed
  if (!options.force && fs.existsSync(pluginDir)) {
    // Check if it's a git repo by trying to resolve HEAD
    try {
      await git.resolveRef({ fs, dir: pluginDir, ref: "HEAD" })
      if (options.verbose) {
        console.log(styleText("cyan", `→`), `Plugin ${spec.name} already installed`)
      }
      return pluginDir
    } catch {
      // If git operations fail, re-clone
    }
  }

  // Clean up if force reinstall
  if (options.force && fs.existsSync(pluginDir)) {
    fs.rmSync(pluginDir, { recursive: true })
  }

  if (options.verbose) {
    console.log(styleText("cyan", `→`), `Cloning ${spec.name} from ${spec.repo}#${spec.ref}...`)
  }

  // Clone the repository
  await git.clone({
    fs,
    http,
    dir: pluginDir,
    url: spec.repo,
    ref: spec.ref,
    singleBranch: true,
    depth: 1,
    noCheckout: false,
  })

  if (options.verbose) {
    console.log(styleText("green", `✓`), `Installed ${spec.name}`)
  }

  return pluginDir
}

/**
 * Install multiple plugins from Git repositories
 */
export async function installPlugins(
  sources: PluginInstallSource[],
  options: { verbose?: boolean; force?: boolean } = {},
): Promise<Map<string, string>> {
  const installed = new Map<string, string>()

  for (const source of sources) {
    try {
      const spec = typeof source === "string" ? parsePluginSource(source) : source
      const pluginDir = await installPlugin(spec, options)
      installed.set(spec.name, pluginDir)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      console.error(styleText("red", `✗`), `Failed to install plugin: ${message}`)
    }
  }

  await regeneratePluginIndex(options)

  return installed
}

/**
 * Get the installation directory for a plugin
 */
export function getPluginDir(name: string): string {
  return path.join(PLUGINS_CACHE_DIR, name)
}

/**
 * Check if a plugin is installed
 */
export function isPluginInstalled(name: string): boolean {
  return fs.existsSync(getPluginDir(name))
}

/**
 * Get the entry point for a plugin.
 * Prefers compiled dist/ output over raw src/ to avoid ESM resolution issues.
 */
export function getPluginEntryPoint(name: string, subdir?: string): string {
  const pluginDir = getPluginDir(name)
  const searchDir = subdir ? path.join(pluginDir, subdir) : pluginDir
  // Check package.json exports first (most reliable)
  const pkgJsonPath = path.join(searchDir, "package.json")
  if (fs.existsSync(pkgJsonPath)) {
    try {
      const pkg = JSON.parse(fs.readFileSync(pkgJsonPath, "utf-8"))
      const exportEntry = pkg.exports?.["."]
      const importPath = typeof exportEntry === "string" ? exportEntry : exportEntry?.import
      if (importPath) {
        const resolved = path.join(searchDir, importPath)
        if (fs.existsSync(resolved)) {
          return resolved
        }
      }
      // Fall back to main/module fields
      const mainField = pkg.module ?? pkg.main
      if (mainField) {
        const resolved = path.join(searchDir, mainField)
        if (fs.existsSync(resolved)) {
          return resolved
        }
      }
    } catch {
      // package.json parse error, fall through to candidates
    }
  }

  // Try common entry points — prefer compiled dist/ over raw src/
  const candidates = [
    path.join(searchDir, "dist", "index.js"),
    path.join(searchDir, "dist", "index.mjs"),
    path.join(searchDir, "index.js"),
    path.join(searchDir, "index.ts"),
    path.join(searchDir, "src", "index.js"),
    path.join(searchDir, "src", "index.ts"),
  ]
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate
    }
  }
  // If no entry found, return the search dir and let Node handle it
  return searchDir
}

/**
 * Resolve a subpath export for a plugin (e.g. "./components").
 * Uses package.json exports map, then falls back to dist/ directory structure.
 */
export function getPluginSubpathEntry(
  name: string,
  subpath: string,
  subdir?: string,
): string | null {
  const pluginDir = getPluginDir(name)
  const searchDir = subdir ? path.join(pluginDir, subdir) : pluginDir

  // Check package.json exports map
  const pkgJsonPath = path.join(searchDir, "package.json")
  if (fs.existsSync(pkgJsonPath)) {
    try {
      const pkg = JSON.parse(fs.readFileSync(pkgJsonPath, "utf-8"))
      const exportEntry = pkg.exports?.[subpath]
      const importPath = typeof exportEntry === "string" ? exportEntry : exportEntry?.import
      if (importPath) {
        const resolved = path.join(searchDir, importPath)
        if (fs.existsSync(resolved)) {
          return resolved
        }
      }
    } catch {
      // fall through
    }
  }

  // Fall back: try dist/<subpath>/index.js
  const subpathClean = subpath.replace(/^\.\/?/, "")
  const fallbackCandidates = [
    path.join(searchDir, "dist", subpathClean, "index.js"),
    path.join(searchDir, "dist", `${subpathClean}.js`),
    path.join(searchDir, subpathClean, "index.js"),
  ]

  for (const candidate of fallbackCandidates) {
    if (fs.existsSync(candidate)) {
      return candidate
    }
  }

  return null
}
/**
 * Update all installed plugins
 */
export async function updatePlugins(options: { verbose?: boolean } = {}): Promise<void> {
  if (!fs.existsSync(PLUGINS_CACHE_DIR)) {
    console.log("No plugins installed")
    return
  }

  const plugins = fs.readdirSync(PLUGINS_CACHE_DIR)

  for (const pluginName of plugins) {
    const pluginDir = path.join(PLUGINS_CACHE_DIR, pluginName)

    try {
      // Check if it's a git repo
      await git.resolveRef({ fs, dir: pluginDir, ref: "HEAD" })

      if (options.verbose) {
        console.log(styleText("cyan", `→`), `Updating ${pluginName}...`)
      }

      // Fetch latest
      await git.fetch({
        fs,
        http,
        dir: pluginDir,
        singleBranch: true,
      })

      // Checkout to latest fetched commit
      await git.checkout({
        fs,
        dir: pluginDir,
        ref: "FETCH_HEAD",
        force: true,
      })

      if (options.verbose) {
        console.log(styleText("green", `✓`), `Updated ${pluginName}`)
      }
    } catch (error) {
      if (options.verbose) {
        console.error(styleText("yellow", `⚠`), `Skipping ${pluginName}: Not a git repo`)
      }
    }
  }
}

/**
 * Clean all installed plugins
 */
export function cleanPlugins(): void {
  if (fs.existsSync(PLUGINS_CACHE_DIR)) {
    fs.rmSync(PLUGINS_CACHE_DIR, { recursive: true })
    console.log(styleText("green", `✓`), "Cleaned all plugins")
  }
}

export async function regeneratePluginIndex(options: { verbose?: boolean } = {}): Promise<void> {
  if (!fs.existsSync(PLUGINS_CACHE_DIR)) {
    return
  }

  const plugins = fs.readdirSync(PLUGINS_CACHE_DIR).filter((name) => {
    const pluginPath = path.join(PLUGINS_CACHE_DIR, name)
    return fs.statSync(pluginPath).isDirectory()
  })

  const exports: string[] = []

  for (const pluginName of plugins) {
    const pluginDir = path.join(PLUGINS_CACHE_DIR, pluginName)
    const distIndex = path.join(pluginDir, "dist", "index.d.ts")

    if (!fs.existsSync(distIndex)) {
      if (options.verbose) {
        console.log(styleText("yellow", `⚠`), `Skipping ${pluginName}: no dist/index.d.ts found`)
      }
      continue
    }

    const dtsContent = fs.readFileSync(distIndex, "utf-8")
    const exportedNames = parseExportsFromDts(dtsContent)

    if (exportedNames.length > 0) {
      const namedExports = exportedNames.filter((e) => !e.startsWith("type "))
      const typeExports = exportedNames.filter((e) => e.startsWith("type ")).map((e) => e.slice(5))

      if (namedExports.length > 0) {
        exports.push(`export { ${namedExports.join(", ")} } from "./${pluginName}"`)
      }
      if (typeExports.length > 0) {
        exports.push(`export type { ${typeExports.join(", ")} } from "./${pluginName}"`)
      }
    }
  }

  const indexContent = exports.join("\n") + "\n"
  const indexPath = path.join(PLUGINS_CACHE_DIR, "index.ts")

  fs.writeFileSync(indexPath, indexContent)

  if (options.verbose) {
    console.log(styleText("green", `✓`), `Regenerated plugin index with ${plugins.length} plugins`)
  }
}

const INTERNAL_EXPORTS = new Set(["manifest", "default"])

function parseExportsFromDts(content: string): string[] {
  const exports: string[] = []

  const exportMatches = content.matchAll(/export\s*{\s*([^}]+)\s*}(?:\s*from\s*['"]([^'"]+)['"])?/g)
  for (const match of exportMatches) {
    const fromModule = match[2]
    if (fromModule?.startsWith("@")) {
      continue
    }

    const names = match[1]
      .split(",")
      .map((n) => n.trim())
      .filter(Boolean)
    for (const name of names) {
      const cleanName = name.split(" as ").pop()?.trim() || name.trim()
      if (cleanName && !cleanName.startsWith("_") && !INTERNAL_EXPORTS.has(cleanName)) {
        const finalName = cleanName.replace(/^type\s+/, "")
        if (name.includes("type ")) {
          exports.push(`type ${finalName}`)
        } else {
          exports.push(finalName)
        }
      }
    }
  }

  return exports
}
