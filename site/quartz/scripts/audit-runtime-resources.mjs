#!/usr/bin/env node
/**
 * Non-vacuous static audit of the emitted Quartz browser resource graph.
 *
 * The audit is intentionally rooted in generated HTML instead of every JS file
 * under public/: HTML <script src> tags and inline scripts are the only roots.
 * From those roots it follows reachable static imports, reachable dynamic
 * import() calls, and reachable calls to validated local script loaders. JS
 * files not reached from HTML are reported as ignored and never contribute
 * Search/Graph/Explorer/runtime evidence.
 *
 * Execution model used for emitted browser scripts:
 *   - top-level statements execute in order;
 *   - function bodies execute only when the function is invoked, used as an
 *     IIFE/new Promise executor, or passed to a known callback registration
 *     surface such as addEventListener/addCleanup/timers/promises/forEach;
 *   - intra-procedural constant control flow is pruned, including code after
 *     statically decisive returns and short-circuited logical/conditional arms.
 *
 * A plugin selector is functional only when the selected binding flows into
 * executable behavior (member access/call, callback, argument, control flow,
 * etc.). Merely assigning document.querySelector(...) to an unused binding is
 * reported as vacuous. Graph's script loader is accepted only when the script
 * element that is actually appended has type="module" and its .src derives from
 * the loader parameter (directly or through a narrow local alias chain); vendor
 * manifest URLs must be supplied to that same validated function/parameter.
 *
 * Loader and vendor edges are attributed by lexical binding identity: an
 * identifier call target is resolved to the AST node of the binding in scope at
 * the call site, and only that node's validated loader shape may credit an
 * edge. There is no name-keyed fallback, so minified same-name functions in
 * different scopes cannot receive each other's calls or edges.
 */

import { readFile, readdir } from "node:fs/promises"
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path"
import { parse } from "acorn"

const RUNTIME_SELECTORS = {
  Search: ".search-container",
  Graph: ".graph-container",
  Explorer: ".explorer-ul",
}

const SELECTOR_BY_VALUE = new Map(Object.entries(RUNTIME_SELECTORS).map(([k, v]) => [v, k]))
const APPEND_METHODS = new Set(["appendChild", "insertBefore", "append", "prepend"])
const CALLBACK_METHODS = new Set([
  "addEventListener",
  "addCleanup",
  "then",
  "catch",
  "finally",
  "forEach",
  "map",
  "filter",
  "reduce",
])
const CALLBACK_FUNCTIONS = new Set([
  "addCleanup",
  "setTimeout",
  "setInterval",
  "queueMicrotask",
  "requestAnimationFrame",
])

function fail(message) {
  process.stderr.write(`${message}\n`)
  process.exit(2)
}

function normalizeBasePath(value) {
  const trimmed = value.replace(/^\/+|\/+$/g, "")
  return trimmed ? `/${trimmed}` : ""
}

function slash(path) {
  return path.split(sep).join("/")
}

function stripQueryFragment(value) {
  return value.split(/[?#]/, 1)[0]
}

function isStringLiteral(node) {
  return node?.type === "Literal" && typeof node.value === "string"
}

function literalString(node) {
  return isStringLiteral(node) ? node.value : null
}

function propertyName(member) {
  if (!member || member.type !== "MemberExpression") return null
  if (!member.computed && member.property.type === "Identifier") return member.property.name
  if (member.computed && isStringLiteral(member.property)) return member.property.value
  return null
}

function isMemberCall(node, names) {
  return (
    node?.type === "CallExpression" &&
    node.callee.type === "MemberExpression" &&
    names.has(propertyName(node.callee))
  )
}

function isDocumentCreateScript(node) {
  return (
    node?.type === "CallExpression" &&
    node.callee.type === "MemberExpression" &&
    propertyName(node.callee) === "createElement" &&
    node.callee.object.type === "Identifier" &&
    node.callee.object.name === "document" &&
    literalString(node.arguments[0]) === "script"
  )
}

function constBool(node) {
  if (!node) return undefined
  if (node.type === "Literal") return Boolean(node.value)
  if (node.type === "Identifier" && node.name === "undefined") return false
  if (node.type === "UnaryExpression") {
    if (node.operator === "void") return false
    if (node.operator === "!") {
      const inner = constBool(node.argument)
      return inner === undefined ? undefined : !inner
    }
  }
  if (node.type === "LogicalExpression") {
    const left = constBool(node.left)
    if (node.operator === "&&") {
      if (left === false) return false
      const right = constBool(node.right)
      return left === true && right !== undefined ? right : undefined
    }
    if (node.operator === "||") {
      if (left === true) return true
      const right = constBool(node.right)
      return left === false && right !== undefined ? right : undefined
    }
    if (node.operator === "??") return undefined
  }
  if (node.type === "ConditionalExpression") {
    const test = constBool(node.test)
    if (test === true) return constBool(node.consequent)
    if (test === false) return constBool(node.alternate)
  }
  return undefined
}

async function listJsFiles(root) {
  const entries = await readdir(root, { withFileTypes: true, recursive: true })
  const files = []
  for (const entry of entries) {
    if (entry.isFile() && entry.name.endsWith(".js")) {
      files.push(resolve(join(entry.parentPath ?? entry.path ?? root, entry.name)))
    }
  }
  return files.sort()
}

function makeFacts(publicDir, basePath, manifest) {
  return {
    publicDir,
    basePath,
    manifest,
    bundles: [],
    parseErrors: [],
    surfaces: Object.fromEntries(
      Object.keys(RUNTIME_SELECTORS).map((name) => [
        name,
        { functional: 0, vacuous: 0, dead: 0, files: new Set() },
      ]),
    ),
    loaderNames: new Set(),
    vendorEdges: Object.fromEntries(
      (manifest.assets ?? []).map((asset) => [
        asset.url,
        { edges: 0, dead: 0, file: asset.file, details: [] },
      ]),
    ),
    sharedFetchData: [],
    resourceGraph: {
      htmlRoots: new Set(),
      inlineRoots: new Set(),
      rootScripts: new Set(),
      reachableScripts: new Set(),
      ignoredScripts: [],
      edges: [],
      loaderEdges: [],
      errors: [],
    },
  }
}

class ResourceGraph {
  constructor(publicDir, basePath, facts, allJsFiles) {
    this.publicDir = resolve(publicDir)
    this.basePath = basePath
    this.basePathSlash = `${basePath}/`
    this.facts = facts
    this.allJsFiles = new Set(allJsFiles.map((file) => resolve(file)))
    this.queue = []
    this.enqueued = new Set()
  }

  rel(file) {
    return slash(relative(this.publicDir, resolve(file)))
  }

  underPublic(file) {
    const rel = relative(this.publicDir, resolve(file))
    return rel && !rel.startsWith("..") && !isAbsolute(rel)
  }

  pagePathForHtml(htmlRel) {
    if (htmlRel === "index.html") return `${this.basePath}/`
    if (htmlRel.endsWith("/index.html")) {
      return `${this.basePathSlash}${htmlRel.slice(0, -"index.html".length)}`
    }
    if (htmlRel.endsWith(".html")) return `${this.basePathSlash}${htmlRel.slice(0, -5)}`
    return `${this.basePathSlash}${htmlRel}`
  }

  decodeLocalPath(pathname, raw, owner) {
    let decoded
    try {
      decoded = decodeURIComponent(pathname)
    } catch (error) {
      this.facts.resourceGraph.errors.push(`${owner}: URL cannot be decoded: ${raw}`)
      return null
    }
    if (decoded !== this.basePath && !decoded.startsWith(this.basePathSlash)) {
      this.facts.resourceGraph.errors.push(`${owner}: local URL escapes ${this.basePath}: ${raw}`)
      return null
    }
    const relPath = decoded.slice(this.basePath.length).replace(/^\/+/, "")
    if (!relPath) {
      this.facts.resourceGraph.errors.push(`${owner}: local script URL has no file path: ${raw}`)
      return null
    }
    const file = resolve(this.publicDir, relPath)
    if (!this.underPublic(file)) {
      this.facts.resourceGraph.errors.push(`${owner}: local script path escapes public/: ${raw}`)
      return null
    }
    return file
  }

  resolveHtmlScript(raw, htmlRel) {
    const owner = `${htmlRel}`
    let parsed
    try {
      parsed = new URL(raw, `https://justinchuby.github.io${this.pagePathForHtml(htmlRel)}`)
    } catch (error) {
      this.facts.resourceGraph.errors.push(`${owner}: invalid script URL ${raw}: ${error.message}`)
      return null
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null
    if (parsed.hostname !== "justinchuby.github.io") return null
    return this.decodeLocalPath(parsed.pathname, raw, owner)
  }

  resolveImport(raw, from) {
    const owner = from.kind === "file" ? this.rel(from.file) : from.id
    const specifier = stripQueryFragment(raw)
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(specifier) || specifier.startsWith("//")) {
      let parsed
      try {
        parsed = specifier.startsWith("//") ? new URL(`https:${specifier}`) : new URL(specifier)
      } catch (error) {
        this.facts.resourceGraph.errors.push(
          `${owner}: invalid import URL ${raw}: ${error.message}`,
        )
        return null
      }
      if (parsed.hostname !== "justinchuby.github.io") return null
      return this.decodeLocalPath(parsed.pathname, raw, owner)
    }
    if (specifier.startsWith("/")) {
      return this.decodeLocalPath(
        new URL(`https://justinchuby.github.io${specifier}`).pathname,
        raw,
        owner,
      )
    }
    if (!specifier.startsWith(".")) return null
    const baseDir = from.kind === "file" ? dirname(from.file) : dirname(from.htmlFile)
    const file = resolve(baseDir, specifier)
    if (!this.underPublic(file)) {
      this.facts.resourceGraph.errors.push(`${owner}: imported script escapes public/: ${raw}`)
      return null
    }
    return file
  }

  addRootScript(file, htmlRel, raw) {
    if (!file) return
    const relFile = this.rel(file)
    this.facts.resourceGraph.rootScripts.add(relFile)
    this.addScript(file, { kind: "html-script", from: htmlRel, specifier: raw })
  }

  addScript(file, edge = null) {
    if (!file) return
    file = resolve(file)
    const relFile = this.rel(file)
    if (!this.underPublic(file)) {
      this.facts.resourceGraph.errors.push(
        `${edge?.from ?? relFile}: script escapes public/: ${relFile}`,
      )
      return
    }
    if (!file.endsWith(".js")) {
      this.facts.resourceGraph.errors.push(
        `${edge?.from ?? relFile}: referenced script is not .js: ${relFile}`,
      )
      return
    }
    if (!this.allJsFiles.has(file)) {
      this.facts.resourceGraph.errors.push(
        `${edge?.from ?? relFile}: missing referenced/imported script: ${relFile}`,
      )
      return
    }
    if (edge) {
      this.facts.resourceGraph.edges.push({ ...edge, to: relFile })
    }
    if (!this.enqueued.has(file)) {
      this.enqueued.add(file)
      this.queue.push(file)
      this.facts.resourceGraph.reachableScripts.add(relFile)
    }
  }

  addImport(raw, from, kind) {
    const file = this.resolveImport(raw, from)
    if (!file) return null
    const fromId = from.kind === "file" ? this.rel(from.file) : from.id
    this.addScript(file, { kind, from: fromId, specifier: raw })
    return file
  }

  finishIgnored() {
    const reachable = new Set(this.facts.resourceGraph.reachableScripts)
    this.facts.resourceGraph.ignoredScripts = [...this.allJsFiles]
      .map((file) => this.rel(file))
      .filter((file) => !reachable.has(file))
      .sort()
  }
}

class Env {
  constructor(parent = null) {
    this.parent = parent
    this.bindings = new Map()
    this.functions = new Map()
  }

  get(name) {
    if (this.bindings.has(name)) return this.bindings.get(name)
    return this.parent?.get(name) ?? null
  }

  set(name, value) {
    this.bindings.set(name, value)
  }

  setFunction(name, node) {
    this.functions.set(name, node)
    this.bindings.set(name, { kind: "function", node })
  }

  getFunction(name) {
    if (this.functions.has(name)) return this.functions.get(name)
    const binding = this.bindings.get(name)
    if (binding?.kind === "function") return binding.node
    return this.parent?.getFunction(name) ?? null
  }
}

function isFunctionNode(node) {
  return (
    node?.type === "FunctionDeclaration" ||
    node?.type === "FunctionExpression" ||
    node?.type === "ArrowFunctionExpression"
  )
}

function functionBodyStatements(node) {
  if (!node) return []
  if (node.body?.type === "BlockStatement") return node.body.body
  if (node.body) return [{ type: "ReturnStatement", argument: node.body }]
  return []
}

function identifierName(node) {
  return node?.type === "Identifier" ? node.name : null
}

function declaratorFunctionName(declarator) {
  return declarator.id.type === "Identifier" && isFunctionNode(declarator.init)
    ? declarator.id.name
    : null
}

function assignmentFunctionName(node) {
  return node.left.type === "Identifier" && isFunctionNode(node.right) ? node.left.name : null
}

function collectNamedFunctions(ast) {
  const functions = []
  function visit(node, parent = null) {
    if (!node || typeof node.type !== "string") return
    if (node.type === "FunctionDeclaration" && node.id?.name) {
      functions.push([node.id.name, node])
    }
    if (node.type === "VariableDeclarator") {
      const name = declaratorFunctionName(node)
      if (name) functions.push([name, node.init])
    }
    if (node.type === "AssignmentExpression") {
      const name = assignmentFunctionName(node)
      if (name) functions.push([name, node.right])
    }
    for (const key of Object.keys(node)) {
      if (key === "start" || key === "end" || key === "range" || key === "loc") continue
      const value = node[key]
      if (Array.isArray(value)) {
        for (const item of value) if (item && typeof item.type === "string") visit(item, node)
      } else if (value && typeof value.type === "string") {
        visit(value, node)
      }
    }
  }
  visit(ast)
  return functions
}

/**
 * Lexical binding resolution for identifier call targets.
 *
 * Minified browser bundles reuse short function names (`d`, `I`, `W`, ...) in
 * many unrelated scopes, so resolving a call target by *name* lets a call to
 * one `d` be credited with the behavior validated on a completely different
 * `d`. Vendor/loader edges are therefore keyed to the resolved AST node of the
 * binding that is actually in lexical scope at the call site.
 *
 * The resolution is deliberately conservative: a binding maps to a function
 * node only when it is introduced by a function declaration, a variable
 * declarator with a function initializer, or a named function expression, and
 * only when nothing ever reassigns that binding. Parameters, destructuring
 * targets, imports, classes, and reassigned bindings resolve to `null`, so an
 * unresolved call can never inherit another scope's validated behavior.
 */
class LexicalScope {
  constructor(parent = null) {
    this.parent = parent
    this.bindings = new Map()
  }

  declare(name, node) {
    const value = node ?? null
    if (this.bindings.has(name) && this.bindings.get(name) !== value) {
      this.bindings.set(name, null)
      return
    }
    this.bindings.set(name, value)
  }

  declareIfAbsent(name, node) {
    if (!this.bindings.has(name)) this.bindings.set(name, node ?? null)
  }

  owner(name) {
    if (this.bindings.has(name)) return this
    return this.parent?.owner(name) ?? null
  }

  resolve(name) {
    const owner = this.owner(name)
    return owner ? owner.bindings.get(name) : null
  }
}

function* childNodes(node) {
  for (const key of Object.keys(node)) {
    if (key === "start" || key === "end" || key === "range" || key === "loc") continue
    const value = node[key]
    if (Array.isArray(value)) {
      for (const item of value) if (item && typeof item.type === "string") yield item
    } else if (value && typeof value.type === "string") {
      yield value
    }
  }
}

function patternNames(node, out = []) {
  if (!node) return out
  switch (node.type) {
    case "Identifier":
      out.push(node.name)
      break
    case "ObjectPattern":
      for (const property of node.properties) {
        patternNames(property.type === "RestElement" ? property.argument : property.value, out)
      }
      break
    case "ArrayPattern":
      for (const element of node.elements) patternNames(element, out)
      break
    case "AssignmentPattern":
      patternNames(node.left, out)
      break
    case "RestElement":
      patternNames(node.argument, out)
      break
    default:
      break
  }
  return out
}

function declareLexicalStatement(statement, scope) {
  if (!statement) return
  switch (statement.type) {
    case "FunctionDeclaration":
      if (statement.id?.name) scope.declare(statement.id.name, statement)
      return
    case "ClassDeclaration":
      if (statement.id?.name) scope.declare(statement.id.name, null)
      return
    case "VariableDeclaration":
      if (statement.kind === "var") return
      for (const declarator of statement.declarations) {
        if (declarator.id.type === "Identifier") {
          scope.declare(
            declarator.id.name,
            isFunctionNode(declarator.init) ? declarator.init : null,
          )
        } else {
          for (const name of patternNames(declarator.id)) scope.declare(name, null)
        }
      }
      return
    case "ImportDeclaration":
      for (const specifier of statement.specifiers) scope.declare(specifier.local.name, null)
      return
    case "ExportNamedDeclaration":
    case "ExportDefaultDeclaration":
      declareLexicalStatement(statement.declaration, scope)
      return
    case "LabeledStatement":
      declareLexicalStatement(statement.body, scope)
      return
    default:
      return
  }
}

function declareLexicalStatements(statements, scope) {
  for (const statement of statements) declareLexicalStatement(statement, scope)
}

function declareHoistedVars(statements, scope) {
  const stack = [...statements]
  while (stack.length) {
    const node = stack.pop()
    if (!node) continue
    if (isFunctionNode(node)) continue
    if (node.type === "VariableDeclaration" && node.kind === "var") {
      for (const declarator of node.declarations) {
        if (declarator.id.type === "Identifier") {
          scope.declare(
            declarator.id.name,
            isFunctionNode(declarator.init) ? declarator.init : null,
          )
        } else {
          for (const name of patternNames(declarator.id)) scope.declare(name, null)
        }
      }
    }
    for (const child of childNodes(node)) stack.push(child)
  }
}

function buildLexicalCallTargets(ast) {
  const pending = []
  const programScope = new LexicalScope(null)

  function poison(name, scope) {
    const owner = scope.owner(name) ?? programScope
    owner.declare(name, null)
  }

  function walk(node, scope) {
    if (!node) return
    if (isFunctionNode(node)) {
      const fnScope = new LexicalScope(scope)
      if (node.type === "FunctionExpression" && node.id?.name) {
        fnScope.declareIfAbsent(node.id.name, node)
      }
      for (const param of node.params ?? []) {
        for (const name of patternNames(param)) fnScope.declare(name, null)
      }
      const statements = node.body?.type === "BlockStatement" ? node.body.body : []
      declareHoistedVars(statements, fnScope)
      declareLexicalStatements(statements, fnScope)
      for (const param of node.params ?? []) walk(param, fnScope)
      if (node.body?.type === "BlockStatement") {
        for (const statement of node.body.body) walk(statement, fnScope)
      } else {
        walk(node.body, fnScope)
      }
      return
    }
    switch (node.type) {
      case "BlockStatement": {
        const blockScope = new LexicalScope(scope)
        declareLexicalStatements(node.body, blockScope)
        for (const statement of node.body) walk(statement, blockScope)
        return
      }
      case "ForStatement":
      case "ForInStatement":
      case "ForOfStatement": {
        const loopScope = new LexicalScope(scope)
        const declaration = node.type === "ForStatement" ? node.init : node.left
        if (declaration?.type === "VariableDeclaration") {
          declareLexicalStatement(declaration, loopScope)
        }
        for (const child of childNodes(node)) walk(child, loopScope)
        return
      }
      case "CatchClause": {
        const catchScope = new LexicalScope(scope)
        for (const name of patternNames(node.param)) catchScope.declare(name, null)
        walk(node.body, catchScope)
        return
      }
      case "SwitchStatement": {
        walk(node.discriminant, scope)
        const switchScope = new LexicalScope(scope)
        for (const switchCase of node.cases) {
          declareLexicalStatements(switchCase.consequent, switchScope)
        }
        for (const switchCase of node.cases) {
          if (switchCase.test) walk(switchCase.test, switchScope)
          for (const statement of switchCase.consequent) walk(statement, switchScope)
        }
        return
      }
      case "AssignmentExpression": {
        for (const name of patternNames(node.left)) poison(name, scope)
        walk(node.left, scope)
        walk(node.right, scope)
        return
      }
      case "UpdateExpression": {
        if (node.argument?.type === "Identifier") poison(node.argument.name, scope)
        walk(node.argument, scope)
        return
      }
      case "CallExpression": {
        if (node.callee.type === "Identifier") pending.push({ node, scope, name: node.callee.name })
        for (const child of childNodes(node)) walk(child, scope)
        return
      }
      default: {
        for (const child of childNodes(node)) walk(child, scope)
      }
    }
  }

  declareHoistedVars(ast.body, programScope)
  declareLexicalStatements(ast.body, programScope)
  for (const statement of ast.body) walk(statement, programScope)

  const targets = new Map()
  for (const entry of pending) targets.set(entry.node, entry.scope.resolve(entry.name))
  return targets
}

class LoaderEnv {
  constructor(parent = null) {
    this.parent = parent
    this.aliases = new Map()
    this.elements = new Map()
    this.functions = new Map()
  }

  getAlias(name) {
    if (this.aliases.has(name)) return this.aliases.get(name)
    return this.parent?.getAlias(name) ?? null
  }

  setAlias(name, value) {
    this.aliases.set(name, value)
  }

  getElement(name) {
    if (this.elements.has(name)) return this.elements.get(name)
    return this.parent?.getElement(name) ?? null
  }

  setElement(name, value) {
    this.elements.set(name, value)
  }

  setFunction(name, node) {
    this.functions.set(name, node)
  }

  getFunction(name) {
    if (this.functions.has(name)) return this.functions.get(name)
    return this.parent?.getFunction(name) ?? null
  }
}

function deriveParamIndex(node, env) {
  if (!node) return null
  if (node.type === "Identifier") return env.getAlias(node.name)
  if (node.type === "SequenceExpression" && node.expressions.length) {
    return deriveParamIndex(node.expressions[node.expressions.length - 1], env)
  }
  if (node.type === "ConditionalExpression") {
    const left = deriveParamIndex(node.consequent, env)
    const right = deriveParamIndex(node.alternate, env)
    return left !== null && left === right ? left : null
  }
  if (node.type === "LogicalExpression") {
    const left = deriveParamIndex(node.left, env)
    const right = deriveParamIndex(node.right, env)
    return left !== null && left === right ? left : null
  }
  return null
}

function validateLoaderFunction(fnNode) {
  const params = fnNode.params ?? []
  if (!params.some((param) => param.type === "Identifier")) return null
  const root = new LoaderEnv()
  params.forEach((param, index) => {
    if (param.type === "Identifier") root.setAlias(param.name, index)
  })
  const validAppends = []
  const callCounts = new Map()

  function hoist(statements, env) {
    for (const statement of statements) {
      if (statement.type === "FunctionDeclaration" && statement.id?.name) {
        env.setFunction(statement.id.name, statement)
      }
    }
  }

  function execFunction(node, parentEnv, args = []) {
    const key = `${node.start}:${node.end}`
    const count = callCounts.get(key) ?? 0
    if (count > 4) return
    callCounts.set(key, count + 1)
    const env = new LoaderEnv(parentEnv)
    ;(node.params ?? []).forEach((param, index) => {
      if (param.type === "Identifier") {
        const passed = args[index]
        env.setAlias(param.name, passed?.kind === "param" ? passed.index : null)
      }
    })
    execStatements(functionBodyStatements(node), env)
    callCounts.set(key, count)
  }

  function evalExpression(node, env) {
    if (!node) return { kind: "unknown" }
    if (node.type === "Identifier") {
      const alias = env.getAlias(node.name)
      if (alias !== null) return { kind: "param", index: alias }
      if (env.getElement(node.name)) return { kind: "element", name: node.name }
      const fn = env.getFunction(node.name)
      if (fn) return { kind: "function", node: fn }
      return { kind: "unknown" }
    }
    if (node.type === "Literal") return { kind: "literal", value: node.value }
    if (node.type === "SequenceExpression") {
      let value = { kind: "unknown" }
      for (const expression of node.expressions) value = evalExpression(expression, env)
      return value
    }
    if (node.type === "AssignmentExpression") {
      const value = evalExpression(node.right, env)
      if (node.left.type === "Identifier") {
        env.setAlias(node.left.name, value.kind === "param" ? value.index : null)
      } else if (node.left.type === "MemberExpression") {
        const object = identifierName(node.left.object)
        const element = object ? env.getElement(object) : null
        const prop = propertyName(node.left)
        if (element && prop === "type") element.typeModule = literalString(node.right) === "module"
        if (element && prop === "src") element.srcParamIndex = deriveParamIndex(node.right, env)
      }
      return value
    }
    if (node.type === "CallExpression") {
      if (isDocumentCreateScript(node)) return { kind: "createdScript" }
      if (isMemberCall(node, APPEND_METHODS)) {
        for (const arg of node.arguments) {
          if (arg.type !== "Identifier") continue
          const element = env.getElement(arg.name)
          if (element?.typeModule && element.srcParamIndex !== null) {
            validAppends.push({ paramIndex: element.srcParamIndex })
          }
        }
      }
      if (isFunctionNode(node.callee)) execFunction(node.callee, env)
      if (node.callee.type === "Identifier") {
        const fn = env.getFunction(node.callee.name)
        if (fn)
          execFunction(
            fn,
            env,
            node.arguments.map((arg) => evalExpression(arg, env)),
          )
      }
      for (const arg of node.arguments) {
        if (isFunctionNode(arg) && isMemberCall(node, CALLBACK_METHODS)) execFunction(arg, env)
        else evalExpression(arg, env)
      }
      return { kind: "unknown" }
    }
    if (node.type === "NewExpression") {
      if (
        node.callee.type === "Identifier" &&
        node.callee.name === "Promise" &&
        isFunctionNode(node.arguments[0])
      ) {
        execFunction(node.arguments[0], env)
      } else {
        for (const arg of node.arguments) evalExpression(arg, env)
      }
      return { kind: "unknown" }
    }
    if (node.type === "ConditionalExpression") {
      evalExpression(node.test, env)
      evalExpression(node.consequent, env)
      evalExpression(node.alternate, env)
      return { kind: "unknown" }
    }
    if (node.type === "LogicalExpression" || node.type === "BinaryExpression") {
      evalExpression(node.left, env)
      evalExpression(node.right, env)
      return { kind: "unknown" }
    }
    if (node.type === "UnaryExpression" || node.type === "UpdateExpression") {
      evalExpression(node.argument, env)
      return { kind: "unknown" }
    }
    if (node.type === "MemberExpression") {
      evalExpression(node.object, env)
      if (node.computed) evalExpression(node.property, env)
      return { kind: "unknown" }
    }
    if (isFunctionNode(node)) return { kind: "function", node }
    for (const key of Object.keys(node)) {
      if (key === "start" || key === "end" || key === "range" || key === "loc") continue
      const value = node[key]
      if (Array.isArray(value)) {
        for (const item of value)
          if (item && typeof item.type === "string") evalExpression(item, env)
      } else if (value && typeof value.type === "string") {
        evalExpression(value, env)
      }
    }
    return { kind: "unknown" }
  }

  function execStatement(statement, env) {
    if (!statement) return
    if (statement.type === "FunctionDeclaration") {
      if (statement.id?.name) env.setFunction(statement.id.name, statement)
      return
    }
    if (statement.type === "VariableDeclaration") {
      for (const decl of statement.declarations) {
        if (decl.id.type !== "Identifier") continue
        if (isDocumentCreateScript(decl.init)) {
          env.setElement(decl.id.name, { typeModule: false, srcParamIndex: null })
          continue
        }
        if (isFunctionNode(decl.init)) {
          env.setFunction(decl.id.name, decl.init)
          continue
        }
        const value = evalExpression(decl.init, env)
        env.setAlias(decl.id.name, value.kind === "param" ? value.index : null)
        if (value.kind === "createdScript") {
          env.setElement(decl.id.name, { typeModule: false, srcParamIndex: null })
        }
      }
      return
    }
    if (statement.type === "ExpressionStatement") {
      evalExpression(statement.expression, env)
      return
    }
    if (statement.type === "ReturnStatement" || statement.type === "ThrowStatement") {
      evalExpression(statement.argument, env)
      return
    }
    if (statement.type === "BlockStatement") {
      execStatements(statement.body, new LoaderEnv(env))
      return
    }
    if (statement.type === "IfStatement") {
      evalExpression(statement.test, env)
      execStatement(statement.consequent, new LoaderEnv(env))
      if (statement.alternate) execStatement(statement.alternate, new LoaderEnv(env))
      return
    }
    if (statement.type === "ForStatement") {
      if (statement.init) {
        if (statement.init.type === "VariableDeclaration") execStatement(statement.init, env)
        else evalExpression(statement.init, env)
      }
      if (statement.test) evalExpression(statement.test, env)
      execStatement(statement.body, new LoaderEnv(env))
      if (statement.update) evalExpression(statement.update, env)
      return
    }
    if (statement.type === "WhileStatement" || statement.type === "DoWhileStatement") {
      if (statement.test) evalExpression(statement.test, env)
      execStatement(statement.body, new LoaderEnv(env))
      return
    }
    if (statement.type === "TryStatement") {
      execStatement(statement.block, env)
      if (statement.handler) execStatement(statement.handler.body, env)
      if (statement.finalizer) execStatement(statement.finalizer, env)
      return
    }
    for (const key of Object.keys(statement)) {
      if (key === "start" || key === "end" || key === "range" || key === "loc") continue
      const value = statement[key]
      if (Array.isArray(value)) {
        for (const item of value)
          if (item && typeof item.type === "string") evalExpression(item, env)
      } else if (value && typeof value.type === "string") {
        evalExpression(value, env)
      }
    }
  }

  function execStatements(statements, env) {
    hoist(statements, env)
    for (const statement of statements) execStatement(statement, env)
  }

  execStatements(functionBodyStatements(fnNode), root)
  if (!validAppends.length) return null
  const paramIndexes = [...new Set(validAppends.map((append) => append.paramIndex))].sort(
    (a, b) => a - b,
  )
  return { paramIndexes }
}

class Analyzer {
  constructor({
    ast,
    id,
    file,
    source,
    facts,
    graph,
    from,
    validatedLoaders,
    lexicalCallTargets,
    isInline,
  }) {
    this.ast = ast
    this.id = id
    this.file = file
    this.source = source
    this.facts = facts
    this.graph = graph
    this.from = from
    this.validatedLoaders = validatedLoaders
    this.lexicalCallTargets = lexicalCallTargets
    this.isInline = isInline
    this.selectorEvidence = []
    this.fetchResult = isInline
      ? { id, parseError: null, functional: 0, vacuous: 0, dead: 0 }
      : null
    this.fetchTrackers = []
    this.callCounts = new Map()
  }

  /**
   * Resolve a call to a validated loader by lexical binding identity.
   *
   * The call's identifier callee is resolved to the AST node of the binding
   * that is actually in scope at that call site, and only that node's
   * validated loader shape may credit an import/vendor edge. There is no
   * name-keyed fallback, so a minified same-name function in a different
   * scope cannot inherit another function's loader behavior.
   */
  loaderForCall(node) {
    const target = this.lexicalCallTargets.get(node)
    return target ? (this.validatedLoaders.byNode.get(target) ?? null) : null
  }

  fileKey() {
    return this.file ? this.graph.rel(this.file) : this.id
  }

  recordSurface(surface, classification) {
    const bucket = this.facts.surfaces[surface]
    bucket[classification] += 1
    bucket.files.add(this.fileKey())
  }

  createSelector(surface) {
    const evidence = { surface, consumed: false }
    this.selectorEvidence.push(evidence)
    return { kind: "selector", evidence }
  }

  consume(value) {
    if (!value) return
    if (value.kind === "selector") {
      if (!value.evidence.consumed) {
        value.evidence.consumed = true
        this.recordSurface(value.evidence.surface, "functional")
      }
    }
  }

  finalize() {
    for (const evidence of this.selectorEvidence) {
      if (!evidence.consumed) this.recordSurface(evidence.surface, "vacuous")
    }
    if (this.fetchResult) this.facts.sharedFetchData.push(this.fetchResult)
  }

  markFetchIfContentIndex(node) {
    if (
      node.type === "CallExpression" &&
      node.callee.type === "Identifier" &&
      node.callee.name === "fetch" &&
      literalString(node.arguments[0])?.includes("static/contentIndex.json")
    ) {
      const tracker = this.fetchTrackers[this.fetchTrackers.length - 1]
      if (tracker) tracker.hasFetch = true
    }
  }

  scanDead(node) {
    if (!node || typeof node.type !== "string") return
    if (isMemberCall(node, new Set(["querySelector", "querySelectorAll"]))) {
      const surface = SELECTOR_BY_VALUE.get(literalString(node.arguments[0]))
      if (surface) this.recordSurface(surface, "dead")
    }
    if (
      this.fetchResult &&
      node.type === "VariableDeclarator" &&
      node.id.type === "Identifier" &&
      node.id.name === "fetchData"
    ) {
      this.fetchResult.dead += 1
    }
    if (node.type === "CallExpression" && node.callee.type === "Identifier") {
      const loader = this.loaderForCall(node)
      if (loader) {
        for (const index of loader.paramIndexes) {
          const url = literalString(node.arguments[index])
          if (url && this.facts.vendorEdges[url]) this.facts.vendorEdges[url].dead += 1
        }
      }
    }
    for (const key of Object.keys(node)) {
      if (key === "start" || key === "end" || key === "range" || key === "loc") continue
      const value = node[key]
      if (Array.isArray(value)) {
        for (const item of value) if (item && typeof item.type === "string") this.scanDead(item)
      } else if (value && typeof value.type === "string") {
        this.scanDead(value)
      }
    }
  }

  hoistFunctions(statements, env) {
    for (const statement of statements) {
      if (statement.type === "FunctionDeclaration" && statement.id?.name) {
        env.setFunction(statement.id.name, statement)
      }
    }
  }

  executeFunction(node, args, parentEnv, reason) {
    const key = `${node.start}:${node.end}:${reason}`
    const count = this.callCounts.get(key) ?? 0
    if (count > 4) return { kind: "unknown" }
    this.callCounts.set(key, count + 1)
    const env = new Env(parentEnv)
    ;(node.params ?? []).forEach((param, index) => {
      if (param.type === "Identifier") env.set(param.name, args[index] ?? { kind: "unknown" })
    })
    let result = { kind: "unknown" }
    if (node.body?.type === "BlockStatement") {
      this.execStatements(node.body.body, env)
    } else if (node.body) {
      result = this.execExpression(node.body, env)
    }
    this.callCounts.set(key, count)
    return result
  }

  execStatements(statements, env) {
    this.hoistFunctions(statements, env)
    let terminated = false
    for (const statement of statements) {
      if (terminated) {
        this.scanDead(statement)
        continue
      }
      terminated = this.execStatement(statement, env)
    }
    return terminated
  }

  execStatement(statement, env) {
    if (!statement) return false
    switch (statement.type) {
      case "ImportDeclaration": {
        const specifier = literalString(statement.source)
        if (specifier) this.graph.addImport(specifier, this.from, "static-import")
        return false
      }
      case "FunctionDeclaration": {
        if (statement.id?.name) env.setFunction(statement.id.name, statement)
        return false
      }
      case "VariableDeclaration": {
        for (const decl of statement.declarations) this.execVariableDeclarator(decl, env)
        return false
      }
      case "ExpressionStatement": {
        this.execExpression(statement.expression, env)
        return false
      }
      case "BlockStatement":
        return this.execStatements(statement.body, new Env(env))
      case "ReturnStatement":
      case "ThrowStatement":
        this.execExpression(statement.argument, env, { consume: true })
        return true
      case "BreakStatement":
      case "ContinueStatement":
        return true
      case "IfStatement": {
        const test = this.execExpression(statement.test, env, { consume: true })
        this.consume(test)
        const value = constBool(statement.test)
        if (value === true) {
          const terminated = this.execStatement(statement.consequent, new Env(env))
          if (statement.alternate) this.scanDead(statement.alternate)
          return terminated
        }
        if (value === false) {
          this.scanDead(statement.consequent)
          return statement.alternate ? this.execStatement(statement.alternate, new Env(env)) : false
        }
        const consequentTerminates = this.execStatement(statement.consequent, new Env(env))
        const alternateTerminates = statement.alternate
          ? this.execStatement(statement.alternate, new Env(env))
          : false
        return consequentTerminates && alternateTerminates
      }
      case "ForStatement": {
        if (statement.init) {
          if (statement.init.type === "VariableDeclaration") this.execStatement(statement.init, env)
          else this.execExpression(statement.init, env)
        }
        const testValue = statement.test ? constBool(statement.test) : undefined
        if (statement.test) this.execExpression(statement.test, env, { consume: true })
        if (testValue === false) {
          this.scanDead(statement.body)
          return false
        }
        this.execStatement(statement.body, new Env(env))
        if (statement.update) this.execExpression(statement.update, env)
        return false
      }
      case "ForOfStatement":
      case "ForInStatement": {
        const loopEnv = new Env(env)
        this.consume(this.execExpression(statement.right, env, { consume: true }))
        if (statement.left.type === "VariableDeclaration") {
          for (const decl of statement.left.declarations) {
            if (decl.id.type === "Identifier") loopEnv.set(decl.id.name, { kind: "unknown" })
          }
        } else {
          this.execExpression(statement.left, loopEnv, { consume: true })
        }
        this.execStatement(statement.body, loopEnv)
        return false
      }
      case "WhileStatement": {
        const testValue = statement.test ? constBool(statement.test) : undefined
        this.execExpression(statement.test, env, { consume: true })
        if (testValue === false) {
          this.scanDead(statement.body)
          return false
        }
        this.execStatement(statement.body, new Env(env))
        return false
      }
      case "DoWhileStatement": {
        this.execStatement(statement.body, new Env(env))
        this.execExpression(statement.test, env, { consume: true })
        return false
      }
      case "TryStatement": {
        const blockTerminates = this.execStatement(statement.block, env)
        const handlerTerminates = statement.handler
          ? this.execStatement(statement.handler.body, env)
          : false
        const finalizerTerminates = statement.finalizer
          ? this.execStatement(statement.finalizer, env)
          : false
        return finalizerTerminates || (blockTerminates && handlerTerminates)
      }
      case "SwitchStatement": {
        this.execExpression(statement.discriminant, env, { consume: true })
        let allTerminate = statement.cases.length > 0
        for (const item of statement.cases) {
          if (item.test) this.execExpression(item.test, env, { consume: true })
          allTerminate = this.execStatements(item.consequent, new Env(env)) && allTerminate
        }
        return allTerminate
      }
      default:
        this.execGeneric(statement, env)
        return false
    }
  }

  execVariableDeclarator(decl, env) {
    if (decl.id.type !== "Identifier") {
      this.execExpression(decl.init, env, { consume: true })
      return
    }
    const name = decl.id.name
    if (isFunctionNode(decl.init)) {
      env.setFunction(name, decl.init)
      return
    }
    if (name === "fetchData" && this.fetchResult) {
      const tracker = { hasFetch: false }
      this.fetchTrackers.push(tracker)
      const value = this.execExpression(decl.init, env)
      this.fetchTrackers.pop()
      if (tracker.hasFetch) this.fetchResult.functional += 1
      else this.fetchResult.vacuous += 1
      env.set(name, value)
      return
    }
    const value = this.execExpression(decl.init, env)
    env.set(name, value)
  }

  execGeneric(node, env) {
    for (const key of Object.keys(node)) {
      if (key === "start" || key === "end" || key === "range" || key === "loc") continue
      const value = node[key]
      if (Array.isArray(value)) {
        for (const item of value)
          if (item && typeof item.type === "string") this.execExpression(item, env)
      } else if (value && typeof value.type === "string") {
        this.execExpression(value, env)
      }
    }
  }

  execExpression(node, env, options = {}) {
    if (!node) return { kind: "unknown" }
    switch (node.type) {
      case "Identifier": {
        const value = env.get(node.name) ?? { kind: "unknown" }
        if (options.consume) this.consume(value)
        return value
      }
      case "Literal":
        return { kind: "literal", value: node.value }
      case "ThisExpression":
      case "Super":
        return { kind: "unknown" }
      case "FunctionExpression":
      case "ArrowFunctionExpression":
        return { kind: "function", node }
      case "AssignmentExpression": {
        const right = this.execExpression(node.right, env)
        if (node.left.type === "Identifier") {
          if (isFunctionNode(node.right)) env.setFunction(node.left.name, node.right)
          else env.set(node.left.name, right)
        } else if (node.left.type === "MemberExpression") {
          const object = this.execExpression(node.left.object, env, { consume: true })
          this.consume(object)
          if (node.left.computed) this.execExpression(node.left.property, env, { consume: true })
          this.consume(right)
        } else {
          this.execExpression(node.left, env, { consume: true })
          this.consume(right)
        }
        return right
      }
      case "UpdateExpression":
      case "UnaryExpression": {
        const value = this.execExpression(node.argument, env, { consume: true })
        this.consume(value)
        return { kind: "unknown" }
      }
      case "BinaryExpression": {
        this.consume(this.execExpression(node.left, env, { consume: true }))
        this.consume(this.execExpression(node.right, env, { consume: true }))
        return { kind: "unknown" }
      }
      case "LogicalExpression": {
        const left = this.execExpression(node.left, env, { consume: true })
        this.consume(left)
        const leftConst = constBool(node.left)
        if (node.operator === "&&" && leftConst === false) {
          this.scanDead(node.right)
          return { kind: "literal", value: false }
        }
        if (node.operator === "||" && leftConst === true) {
          this.scanDead(node.right)
          return { kind: "literal", value: true }
        }
        if (node.operator === "??" && node.left.type === "Literal" && node.left.value != null) {
          this.scanDead(node.right)
          return left
        }
        const right = this.execExpression(node.right, env, { consume: true })
        this.consume(right)
        return { kind: "unknown" }
      }
      case "ConditionalExpression": {
        this.consume(this.execExpression(node.test, env, { consume: true }))
        const test = constBool(node.test)
        if (test === true) {
          this.scanDead(node.alternate)
          return this.execExpression(node.consequent, env)
        }
        if (test === false) {
          this.scanDead(node.consequent)
          return this.execExpression(node.alternate, env)
        }
        const left = this.execExpression(node.consequent, env)
        const right = this.execExpression(node.alternate, env)
        return left.kind !== "unknown" ? left : right
      }
      case "SequenceExpression": {
        let value = { kind: "unknown" }
        for (const expression of node.expressions) value = this.execExpression(expression, env)
        return value
      }
      case "MemberExpression": {
        const object = this.execExpression(node.object, env, { consume: true })
        this.consume(object)
        if (node.computed) this.consume(this.execExpression(node.property, env, { consume: true }))
        return { kind: "unknown" }
      }
      case "CallExpression":
        return this.execCall(node, env)
      case "NewExpression": {
        if (
          node.callee.type === "Identifier" &&
          node.callee.name === "Promise" &&
          isFunctionNode(node.arguments[0])
        ) {
          this.executeFunction(node.arguments[0], [], env, "promise-executor")
        } else {
          for (const arg of node.arguments)
            this.consume(this.execExpression(arg, env, { consume: true }))
        }
        return { kind: "unknown" }
      }
      case "ChainExpression":
        return this.execExpression(node.expression, env, options)
      case "ImportExpression": {
        const specifier = literalString(node.source)
        if (specifier) this.graph.addImport(specifier, this.from, "dynamic-import")
        else this.execExpression(node.source, env, { consume: true })
        return { kind: "unknown" }
      }
      case "TemplateLiteral": {
        for (const expression of node.expressions)
          this.consume(this.execExpression(expression, env, { consume: true }))
        return { kind: "unknown" }
      }
      case "ObjectExpression": {
        for (const prop of node.properties) {
          if (prop.type === "Property")
            this.consume(this.execExpression(prop.value, env, { consume: true }))
          else this.execExpression(prop, env, { consume: true })
        }
        return { kind: "unknown" }
      }
      case "ArrayExpression": {
        for (const element of node.elements)
          this.consume(this.execExpression(element, env, { consume: true }))
        return { kind: "unknown" }
      }
      default:
        this.execGeneric(node, env)
        return { kind: "unknown" }
    }
  }

  execCall(node, env) {
    if (isMemberCall(node, new Set(["querySelector", "querySelectorAll"]))) {
      const surface = SELECTOR_BY_VALUE.get(literalString(node.arguments[0]))
      for (const arg of node.arguments) this.execExpression(arg, env, { consume: true })
      return surface ? this.createSelector(surface) : { kind: "unknown" }
    }

    this.markFetchIfContentIndex(node)

    if (node.callee.type === "Import") {
      const specifier = literalString(node.arguments[0])
      if (specifier) this.graph.addImport(specifier, this.from, "dynamic-import")
      return { kind: "unknown" }
    }

    let memberObject = null
    let calleeName = null
    let localFunction = null
    if (node.callee.type === "MemberExpression") {
      memberObject = this.execExpression(node.callee.object, env, { consume: true })
      this.consume(memberObject)
      if (node.callee.computed) this.execExpression(node.callee.property, env, { consume: true })
      calleeName = propertyName(node.callee)
    } else if (node.callee.type === "Identifier") {
      calleeName = node.callee.name
      localFunction = env.getFunction(node.callee.name)
    } else if (isFunctionNode(node.callee)) {
      const args = node.arguments.map((arg) => this.execExpression(arg, env))
      return this.executeFunction(node.callee, args, env, "iife")
    } else {
      this.execExpression(node.callee, env, { consume: true })
    }

    if (node.callee.type === "Identifier") {
      const loader = this.loaderForCall(node)
      if (loader) {
        for (const index of loader.paramIndexes) {
          const url = literalString(node.arguments[index])
          if (!url) continue
          const target = this.graph.addImport(url, this.from, "validated-loader")
          if (!target) continue
          this.facts.loaderNames.add(node.callee.name)
          const edge = {
            file: this.fileKey(),
            loader: node.callee.name,
            parameterIndex: index,
            url,
            target: this.graph.rel(target),
          }
          this.facts.resourceGraph.loaderEdges.push(edge)
          if (this.facts.vendorEdges[url]) {
            this.facts.vendorEdges[url].edges += 1
            this.facts.vendorEdges[url].details.push(edge)
          }
        }
      }
    }

    const isCallbackRegistration =
      (node.callee.type === "MemberExpression" && CALLBACK_METHODS.has(calleeName)) ||
      (node.callee.type === "Identifier" && CALLBACK_FUNCTIONS.has(calleeName))

    let args
    if (localFunction) {
      args = node.arguments.map((arg) => this.execExpression(arg, env))
    } else {
      args = node.arguments.map((arg) => {
        if (isFunctionNode(arg)) return { kind: "function", node: arg }
        const value = this.execExpression(arg, env, { consume: true })
        this.consume(value)
        return value
      })
    }

    if (isCallbackRegistration) {
      for (const arg of args) {
        if (arg.kind === "function") this.executeFunction(arg.node, [], env, "registered-callback")
      }
      for (const rawArg of node.arguments) {
        if (rawArg.type === "Identifier") {
          const fn = env.getFunction(rawArg.name)
          if (fn) this.executeFunction(fn, [], env, "registered-callback")
        }
      }
    }

    if (localFunction) return this.executeFunction(localFunction, args, env, "direct-call")
    return { kind: "unknown" }
  }

  run() {
    const env = new Env()
    this.execStatements(this.ast.body, env)
    this.finalize()
  }
}

async function auditSource({ source, id, file = null, facts, graph, from, isInline = false }) {
  let ast
  try {
    ast = parse(source, { ecmaVersion: "latest", sourceType: "module" })
  } catch (error) {
    facts.parseErrors.push({ file: id, message: error.message })
    if (isInline)
      facts.sharedFetchData.push({
        id,
        parseError: error.message,
        functional: 0,
        vacuous: 0,
        dead: 0,
      })
    return
  }
  if (file) facts.bundles.push(graph.rel(file))
  const namedFunctions = collectNamedFunctions(ast)
  const validatedLoaders = { byNode: new Map() }
  for (const [, fn] of namedFunctions) {
    if (validatedLoaders.byNode.has(fn)) continue
    const loader = validateLoaderFunction(fn)
    if (loader) validatedLoaders.byNode.set(fn, loader)
  }
  const analyzer = new Analyzer({
    ast,
    id,
    file,
    source,
    facts,
    graph,
    from,
    validatedLoaders,
    lexicalCallTargets: buildLexicalCallTargets(ast),
    isInline,
  })
  analyzer.run()
}

async function readStdin() {
  if (process.stdin.isTTY) return ""
  const chunks = []
  for await (const chunk of process.stdin) chunks.push(chunk)
  return Buffer.concat(
    chunks.map((chunk) => (typeof chunk === "string" ? Buffer.from(chunk) : chunk)),
  ).toString("utf8")
}

async function main() {
  const [, , publicDirArg, basePathArg, manifestPathArg] = process.argv
  if (!publicDirArg || !basePathArg || !manifestPathArg) {
    fail("usage: audit-runtime-resources.mjs <publicDir> <basePath> <manifestPath>")
    return
  }
  const publicDir = resolve(publicDirArg)
  const basePath = normalizeBasePath(basePathArg)

  const stdinText = (await readStdin()).trim()
  let documents = []
  if (stdinText) {
    try {
      const parsed = JSON.parse(stdinText)
      if (Array.isArray(parsed.documents)) documents = parsed.documents
      else if (Array.isArray(parsed.inlineScripts)) {
        documents = [{ html: "index.html", scriptSources: [], inlineScripts: parsed.inlineScripts }]
      }
    } catch (error) {
      console.log(
        JSON.stringify({
          manifestError: `invalid inline-scripts payload on stdin: ${error.message}`,
        }),
      )
      return
    }
  }

  let manifest
  try {
    manifest = JSON.parse(await readFile(manifestPathArg, "utf8"))
  } catch (error) {
    console.log(JSON.stringify({ manifestError: `${manifestPathArg}: ${error.message}` }))
    return
  }

  const allJsFiles = await listJsFiles(publicDir)
  const facts = makeFacts(publicDir, basePath, manifest)
  const graph = new ResourceGraph(publicDir, basePath, facts, allJsFiles)

  for (const doc of documents) {
    const htmlRel = doc.html
    facts.resourceGraph.htmlRoots.add(htmlRel)
    const htmlFile = resolve(publicDir, htmlRel)
    for (const rawSource of doc.scriptSources ?? []) {
      const file = graph.resolveHtmlScript(rawSource, htmlRel)
      graph.addRootScript(file, htmlRel, rawSource)
    }
    ;(doc.inlineScripts ?? []).forEach((entry, index) => {
      const source = typeof entry === "string" ? entry : entry.source
      const id =
        typeof entry === "string" ? `${htmlRel}#${index}` : (entry.id ?? `${htmlRel}#${index}`)
      facts.resourceGraph.inlineRoots.add(id)
      auditSource({
        source,
        id,
        facts,
        graph,
        from: { kind: "inline", id, htmlFile },
        isInline: true,
      })
    })
  }

  while (graph.queue.length) {
    const file = graph.queue.shift()
    const source = await readFile(file, "utf8")
    await auditSource({
      source,
      id: graph.rel(file),
      file,
      facts,
      graph,
      from: { kind: "file", file },
    })
  }

  graph.finishIgnored()

  const output = {
    manifest,
    bundleCount: facts.bundles.length,
    parseErrors: facts.parseErrors,
    surfaces: Object.fromEntries(
      Object.entries(facts.surfaces).map(([name, value]) => [
        name,
        {
          functional: value.functional,
          vacuous: value.vacuous,
          dead: value.dead,
          files: [...value.files].sort(),
        },
      ]),
    ),
    loaderNames: [...facts.loaderNames].sort(),
    vendorEdges: facts.vendorEdges,
    sharedFetchData: facts.sharedFetchData,
    resourceGraph: {
      htmlRoots: [...facts.resourceGraph.htmlRoots].sort(),
      inlineRoots: [...facts.resourceGraph.inlineRoots].sort(),
      rootScripts: [...facts.resourceGraph.rootScripts].sort(),
      reachableScripts: [...facts.resourceGraph.reachableScripts].sort(),
      ignoredScripts: facts.resourceGraph.ignoredScripts,
      edges: facts.resourceGraph.edges.sort((a, b) =>
        JSON.stringify(a).localeCompare(JSON.stringify(b)),
      ),
      loaderEdges: facts.resourceGraph.loaderEdges.sort((a, b) =>
        JSON.stringify(a).localeCompare(JSON.stringify(b)),
      ),
      errors: facts.resourceGraph.errors.sort(),
    },
  }
  console.log(JSON.stringify(output))
}

await main()
