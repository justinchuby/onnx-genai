#!/usr/bin/env node
/**
 * Non-vacuous static audit of the emitted Quartz resource graph.
 *
 * This intentionally does NOT use substring/regex signature checks over
 * bundle text: those can be satisfied by a comment, a string that is never
 * passed to a DOM API, or dead code that never runs. Instead every emitted
 * `*.js` bundle under the built `public/` directory is parsed into a real
 * ESTree AST (via acorn) and walked so that:
 *
 *   - comments can never counterfeit a signature (the parser does not even
 *     produce nodes for them),
 *   - statements that are provably unreachable (after `return`/`throw`/
 *     `break`/`continue`, or inside a constant-falsy `if` branch) are
 *     tagged `dead` and never counted as functional evidence,
 *   - a `document.querySelector(...)`/`querySelectorAll(...)` call whose
 *     result is immediately discarded (a bare `ExpressionStatement`) is
 *     tagged `vacuous` rather than functional, because nothing downstream
 *     depends on it,
 *   - Graph's local ESM loader must be a real, reachable "create a
 *     <script>, set `.type='module'` and `.src=...`, then append it to the
 *     document" call graph, and each vendor asset from the build-emitted
 *     manifest must have a real call-site "import edge" invoking that
 *     loader with the asset's exact deployed URL.
 *
 * Output is a single deterministic JSON object on stdout; the caller
 * (site/scripts/validate_site.py) applies pass/fail policy and cross-checks
 * the manifest against on-disk content hashes.
 */

import { readFile, readdir } from "node:fs/promises"
import { join, relative } from "node:path"
import { parse } from "acorn"

const RUNTIME_SELECTORS = {
  Search: ".search-container",
  Graph: ".graph-container",
  Explorer: ".explorer-ul",
}

const APPEND_METHODS = new Set(["appendChild", "insertBefore", "append", "prepend"])

function fail(message) {
  process.stderr.write(`${message}\n`)
  process.exit(2)
}

async function listJsFiles(root) {
  const entries = await readdir(root, { withFileTypes: true, recursive: true })
  const files = []
  for (const entry of entries) {
    if (entry.isFile() && entry.name.endsWith(".js")) {
      files.push(join(entry.parentPath ?? entry.path ?? root, entry.name))
    }
  }
  return files.sort()
}

// ---------------------------------------------------------------------------
// Reachability-aware AST walk.
//
// Threads a `reachable` boolean through the tree so a signature that only
// ever appears after an unconditional terminator, or inside a provably
// constant-false branch, is tagged dead instead of functional. Nested
// function bodies always start reachable: whether the *enclosing* function
// itself is ever invoked is out of scope for a bundle-local static audit,
// consistent with normal dead-code-elimination semantics.
// ---------------------------------------------------------------------------

const TERMINATORS = new Set([
  "ReturnStatement",
  "ThrowStatement",
  "BreakStatement",
  "ContinueStatement",
])

function constBool(node) {
  if (!node) return undefined
  if (node.type === "Literal") return Boolean(node.value)
  if (node.type === "UnaryExpression" && node.operator === "!") {
    const inner = constBool(node.argument)
    return inner === undefined ? undefined : !inner
  }
  if (node.type === "UnaryExpression" && node.operator === "void") return false
  if (node.type === "Identifier" && node.name === "undefined") return false
  return undefined
}

function walk(node, state, visit) {
  if (!node || typeof node.type !== "string") return
  node.__reachable = state.reachable
  visit(node, state)

  if (node.type === "BlockStatement" || node.type === "Program") {
    let reachable = state.reachable
    for (const statement of node.body) {
      walk(statement, { ...state, reachable }, visit)
      if (TERMINATORS.has(statement.type)) reachable = false
    }
    return
  }

  if (node.type === "IfStatement") {
    walk(node.test, state, visit)
    const value = constBool(node.test)
    const consequentReachable = value === false ? false : state.reachable
    const alternateReachable = value === true ? false : state.reachable
    walk(node.consequent, { ...state, reachable: consequentReachable }, visit)
    if (node.alternate) walk(node.alternate, { ...state, reachable: alternateReachable }, visit)
    return
  }

  if (node.type === "SwitchCase") {
    if (node.test) walk(node.test, state, visit)
    for (const statement of node.consequent) walk(statement, state, visit)
    return
  }

  // Generic structural recursion for every other node type/collection.
  for (const key of Object.keys(node)) {
    if (
      key === "__reachable" ||
      key.startsWith("loc") ||
      key === "start" ||
      key === "end" ||
      key === "range"
    ) {
      continue
    }
    const value = node[key]
    if (Array.isArray(value)) {
      for (const item of value) {
        if (item && typeof item.type === "string") walk(item, state, visit)
      }
    } else if (value && typeof value.type === "string") {
      walk(value, state, visit)
    }
  }
}

function isMemberCall(node, propertyNames) {
  return (
    node?.type === "CallExpression" &&
    node.callee.type === "MemberExpression" &&
    !node.callee.computed &&
    node.callee.property.type === "Identifier" &&
    propertyNames.has(node.callee.property.name)
  )
}

function calleeObjectName(node) {
  return node.callee.object.type === "Identifier" ? node.callee.object.name : null
}

/** Classify a querySelector(All) match as functional, vacuous, or dead. */
function classifySelectorUse(call, parent) {
  if (!call.__reachable) return "dead"
  if (!parent) return "vacuous"
  if (parent.type === "ExpressionStatement") return "vacuous"
  return "functional"
}

function boundIdentifierName(call, parent) {
  if (
    parent.type === "VariableDeclarator" &&
    parent.init === call &&
    parent.id.type === "Identifier"
  ) {
    return parent.id.name
  }
  if (
    parent.type === "AssignmentExpression" &&
    parent.right === call &&
    parent.left.type === "Identifier"
  ) {
    return parent.left.name
  }
  return null
}

async function auditFile(file, source, facts) {
  let ast
  try {
    ast = parse(source, { ecmaVersion: "latest", sourceType: "module" })
  } catch (error) {
    facts.parseErrors.push({ file, message: error.message })
    return
  }
  facts.bundles.push(file)

  const parents = new WeakMap()
  const enclosingChainOf = new WeakMap()

  // First pass: record parents and the chain of enclosing function/program
  // ancestors (innermost first) so "is this call's result used?" and
  // "search this function's whole subtree for a matching var-scoped
  // assignment" can both be answered, and so a script-loader can be
  // identified even when the DOM wiring lives in a nested closure (e.g. a
  // `new Promise(function (resolve, reject) {...})` executor) rather than
  // directly in the named function that callers invoke.
  function recordStructure(node, chain, parentNode) {
    if (!node || typeof node.type !== "string") return
    parents.set(node, parentNode ?? null)
    enclosingChainOf.set(node, chain)
    const isFunctionLike =
      node.type === "FunctionDeclaration" ||
      node.type === "FunctionExpression" ||
      node.type === "ArrowFunctionExpression" ||
      node.type === "Program"
    const childChain = isFunctionLike ? [node, ...chain] : chain
    for (const key of Object.keys(node)) {
      if (key === "start" || key === "end" || key === "range" || key === "loc") continue
      const value = node[key]
      if (Array.isArray(value)) {
        for (const item of value) {
          if (item && typeof item.type === "string") recordStructure(item, childChain, node)
        }
      } else if (value && typeof value.type === "string") {
        recordStructure(value, childChain, node)
      }
    }
  }
  recordStructure(ast, [], null)

  function functionName(fnNode) {
    if (fnNode.type === "FunctionDeclaration" && fnNode.id) return fnNode.id.name
    const parent = parents.get(fnNode)
    if (
      parent?.type === "VariableDeclarator" &&
      parent.init === fnNode &&
      parent.id.type === "Identifier"
    ) {
      return parent.id.name
    }
    if (
      parent?.type === "AssignmentExpression" &&
      parent.right === fnNode &&
      parent.left.type === "Identifier"
    ) {
      return parent.left.name
    }
    return null
  }

  // Second pass: reachability-tagged evaluation.
  walk(ast, { reachable: true }, (node) => {
    // --- Search / Graph / Explorer container selectors ---
    if (isMemberCall(node, new Set(["querySelector", "querySelectorAll"]))) {
      const arg = node.arguments[0]
      if (arg && arg.type === "Literal" && typeof arg.value === "string") {
        for (const [surface, selector] of Object.entries(RUNTIME_SELECTORS)) {
          if (arg.value === selector) {
            const parent = parents.get(node)
            const classification = classifySelectorUse(node, parent)
            facts.surfaces[surface][classification] += 1
            facts.surfaces[surface].files.add(relative(facts.publicDir, file))
          }
        }
      }
    }

    // --- document.createElement("script") -> local script-loader function ---
    if (
      isMemberCall(node, new Set(["createElement"])) &&
      calleeObjectName(node) === "document" &&
      node.arguments[0]?.type === "Literal" &&
      node.arguments[0].value === "script"
    ) {
      const parent = parents.get(node)
      const varName = boundIdentifierName(node, parent)
      const chain = enclosingChainOf.get(node)
      const innermostFn = chain?.[0]
      if (varName && innermostFn) {
        facts.__scriptElementCandidates.push({
          varName,
          innermostFn,
          chain,
          reachable: node.__reachable,
        })
      }
    }
  })

  // Third pass: for each createElement("script") candidate, confirm
  // .type = "module", .src = <anything>, and a real DOM-insertion call all
  // reference the same variable, reachably, anywhere in the immediate
  // enclosing function's subtree (covers the common case where the
  // assignments live in a nested Promise executor closing over a `var`),
  // then attribute the loader to the nearest *named* ancestor function --
  // that is the identifier callers actually invoke.
  const loaderNames = new Set()
  for (const candidate of facts.__scriptElementCandidates) {
    if (!candidate.reachable) continue
    let hasType = false
    let hasSrc = false
    let appended = false
    walk(candidate.innermostFn, { reachable: true }, (node) => {
      if (!node.__reachable) return
      if (
        node.type === "AssignmentExpression" &&
        node.left.type === "MemberExpression" &&
        !node.left.computed &&
        node.left.object.type === "Identifier" &&
        node.left.object.name === candidate.varName &&
        node.left.property.type === "Identifier"
      ) {
        if (
          node.left.property.name === "type" &&
          node.right.type === "Literal" &&
          node.right.value === "module"
        ) {
          hasType = true
        }
        if (node.left.property.name === "src") hasSrc = true
      }
      if (isMemberCall(node, APPEND_METHODS)) {
        const referencesVar = node.arguments.some(
          (arg) => arg.type === "Identifier" && arg.name === candidate.varName,
        )
        if (referencesVar) appended = true
      }
    })
    if (hasType && hasSrc && appended) {
      const name = candidate.chain.map(functionName).find((value) => value !== null) ?? null
      if (name) loaderNames.add(name)
    }
  }
  facts.__scriptElementCandidates = []
  for (const name of loaderNames) facts.loaderNames.add(name)

  // Fourth pass: import edges. A call to a known loader name with a literal
  // first argument that is exactly one of the manifest's deployed vendor
  // URLs, in reachable code.
  walk(ast, { reachable: true }, (node) => {
    if (
      node.type === "CallExpression" &&
      node.callee.type === "Identifier" &&
      loaderNames.has(node.callee.name) &&
      node.arguments[0]?.type === "Literal" &&
      typeof node.arguments[0].value === "string"
    ) {
      const url = node.arguments[0].value
      if (url in facts.vendorEdges) {
        facts.vendorEdges[url][node.__reachable ? "edges" : "dead"] += 1
      }
    }
  })
}

// ---------------------------------------------------------------------------
// Shared inline `fetchData` content-index surface.
//
// Every rendered body page ships an inline `<script>` that starts the
// shared content-index fetch promise other runtime surfaces consume. The
// old validator matched this with a regex over raw HTML text, so a
// commented-out or vacuous declaration ("const fetchData = 1;" with the
// real call living only in a `//` comment) would still count. Here we
// parse each inline script body and require a REAL, reachable
// `fetchData` declarator whose initializer contains an actual
// `fetch("...static/contentIndex.json...")` call anywhere in its
// expression tree (covering the `.then(...)` chain Quartz emits).
// ---------------------------------------------------------------------------

function findFetchCall(node) {
  if (!node || typeof node.type !== "string") return null
  if (
    node.type === "CallExpression" &&
    node.callee.type === "Identifier" &&
    node.callee.name === "fetch" &&
    node.arguments[0]?.type === "Literal" &&
    typeof node.arguments[0].value === "string" &&
    node.arguments[0].value.includes("static/contentIndex.json")
  ) {
    return node
  }
  for (const key of Object.keys(node)) {
    if (key === "start" || key === "end" || key === "range" || key === "loc") continue
    const value = node[key]
    if (Array.isArray(value)) {
      for (const item of value) {
        const found = findFetchCall(item)
        if (found) return found
      }
    } else if (value && typeof value.type === "string") {
      const found = findFetchCall(value)
      if (found) return found
    }
  }
  return null
}

function auditInlineScript(id, source) {
  let ast
  try {
    ast = parse(source, { ecmaVersion: "latest", sourceType: "module" })
  } catch (error) {
    return { id, parseError: error.message, functional: 0, vacuous: 0, dead: 0 }
  }
  const result = { id, parseError: null, functional: 0, vacuous: 0, dead: 0 }
  walk(ast, { reachable: true }, (node) => {
    if (
      node.type !== "VariableDeclarator" ||
      node.id.type !== "Identifier" ||
      node.id.name !== "fetchData"
    ) {
      return
    }
    const fetchCall = node.init ? findFetchCall(node.init) : null
    if (!node.__reachable) result.dead += 1
    else if (fetchCall) result.functional += 1
    else result.vacuous += 1
  })
  return result
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
  const publicDir = publicDirArg
  const basePath = `/${basePathArg.replace(/^\/+|\/+$/g, "")}`

  const stdinText = (await readStdin()).trim()
  let inlineScripts = []
  if (stdinText) {
    try {
      const parsed = JSON.parse(stdinText)
      inlineScripts = Array.isArray(parsed.inlineScripts) ? parsed.inlineScripts : []
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

  const facts = {
    publicDir,
    basePath,
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
      (manifest.assets ?? []).map((asset) => [asset.url, { edges: 0, dead: 0, file: asset.file }]),
    ),
    __scriptElementCandidates: [],
  }

  const files = await listJsFiles(publicDir)
  for (const file of files) {
    const source = await readFile(file, "utf8")
    await auditFile(file, source, facts)
  }

  const sharedFetchData = inlineScripts.map(({ id, source }) => auditInlineScript(id, source))

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
    sharedFetchData,
  }
  console.log(JSON.stringify(output))
}

await main()
