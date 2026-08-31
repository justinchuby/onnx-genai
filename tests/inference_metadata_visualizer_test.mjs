import assert from "node:assert/strict";
import {readFileSync, readdirSync} from "node:fs";
import test from "node:test";
import vm from "node:vm";

const htmlPath = new URL("../examples/inference_metadata/visualizer.html", import.meta.url);
const html = readFileSync(htmlPath, "utf8");
const script = id => {
  const match = html.match(new RegExp(`<script\\b[^>]*\\bid="${id}"[^>]*>([\\s\\S]*?)<\\/script>`));
  assert.ok(match, `missing ${id}`);
  return match[1];
};
const context = {console};
context.globalThis = context;
vm.createContext(context);
vm.runInContext(script("vendor-js-yaml"), context);
vm.runInContext(script("visualizer-helpers"), context);
const H = context.VisualizerHelpers;

test("is a single page with one integrity-pinned CDN dependency and license notices", () => {
  const external = html.match(/<(?:script|link|img)\b[^>]*(?:src|href)=["']https?:[^>]*>/gi) ?? [];
  assert.equal(external.length, 1);
  assert.match(external[0], /src="https:\/\/cdn\.jsdelivr\.net\/npm\/mermaid@11\.12\.0\/dist\/mermaid\.min\.js"/);
  assert.match(external[0], /integrity="sha384-o\+g\/BxPwhi0C3RK7oQBxQuNimeafQ3GE\/ST4iT2BxVI4Wzt60SH4pq9iXVYujjaS"/);
  assert.match(external[0], /crossorigin="anonymous"/);
  assert.match(external[0], /referrerpolicy="no-referrer"/);
  assert.match(html, /script-src 'unsafe-inline' https:\/\/cdn\.jsdelivr\.net;/);
  assert.match(html, /connect-src 'none'/);
  assert.match(html, /js-yaml 4\.1\.0, MIT License/);
  assert.match(html, /Mermaid 11\.12\.0, MIT License/);
  assert.match(html, /Content-Security-Policy/);
  assert.doesNotMatch(script("visualizer-helpers"), /\beval\s*\(|\.innerHTML\s*=/);
});

test("parses JSON and YAML and preserves falsy and empty values", () => {
  const json = H.parseMetadata('{"schema_version":"v1","n":null,"f":false,"z":0,"s":"","o":{},"a":[]}');
  const yaml = H.parseMetadata("schema_version: v1\nf: false\nz: 0\ns: ''\no: {}\na: []\n");
  for (const value of [json, yaml]) {
    assert.equal(value.f, false); assert.equal(value.z, 0); assert.equal(value.s, "");
    assert.deepEqual(Object.keys(value.o), []); assert.equal(value.a.length, 0);
  }
  assert.equal(json.n, null);
});

test("recursive leaf accounting is complete and includes unknown future fields", () => {
  const value = H.parseMetadata("schema_version: v1\nknown: [1, null]\nfuture:\n  flag: false\n  empty: {}\n");
  const leaves = H.enumerateLeaves(value);
  assert.deepEqual([...leaves.map(x => x.path)], [
    "$.schema_version", "$.known[0]", "$.known[1]", "$.future.flag", "$.future.empty"
  ]);
  assert.equal(new Set(leaves.map(x => x.path)).size, leaves.length);
});

test("YAML aliases cannot make recursive coverage loop forever", () => {
  const value = H.parseMetadata("schema_version: v1\nnode: &node\n  name: shared\n  self: *node\n");
  const leaves = H.enumerateLeaves(value);
  assert.ok(leaves.some(x => x.kind === "reference"));
  assert.ok(leaves.length < 10);
});

test("loops produce safe structural Mermaid and nested timeline data", () => {
  const value = H.parseMetadata("schema_version: v1\npipeline:\n  workflow:\n    steps:\n      - kind: loop\n        steps:\n          - kind: invoke\n            component: '<img src=x onerror=alert(1)>'\n");
  assert.equal(H.flattenSteps(value.pipeline.workflow.steps).length, 2);
  const graph = H.buildMermaid(value);
  assert.match(graph, /^flowchart TD/);
  assert.doesNotMatch(graph, /<img|onerror|alert\(1\)/);
});

test("K/V aliases pair by numeric layer while preserving heterogeneous geometry", () => {
  const value = H.parseMetadata(`schema_version: v1
pipeline:
  workflow:
    serving:
      state_service:
        groups:
          a: {kind: full_attention, layout: bhsd, sequence_axis: 2, ports: {m: {k: {role: key, layer: 2, input: ki, output: ko}, v: {role: value, layer: 2, input: vi, output: vo}}}}
          b: {kind: recurrent, layout: bsh, sequence_axis: 1, ports: {m: {k: {role: key, layer: 0, input: x, output: y}, v: {role: value, layer: 0, input: q, output: r}}}}
`);
  const kv = H.kvLayers(value);
  assert.equal(kv.length, 2); assert.ok(kv[0].layers[2].key); assert.ok(kv[0].layers[2].value);
  assert.notEqual(kv[0].geometry.layout, kv[1].geometry.layout);
});

test("feature views detect adapters, media, speculative rollback, audio, and image", () => {
  const value = H.parseMetadata(`schema_version: v1
adapters: {artifacts: {}}
speculative: {rollback_state: [cache]}
distributed: {sharding: tensor_parallel}
preprocessing: {image: {value_range: [0, 1]}, audio: {sample_rate: 16000}}
pipeline: {workflow: {outputs: {image: {role: image}, audio: {role: audio}}, components: {}, steps: []}}
`);
  const f = H.featureSummary(value);
  assert.equal(f.features.adapters, true); assert.equal(f.features.media, true);
  assert.equal(f.features.speculative, true); assert.equal(f.features.distributed, true);
});

test("XSS fixtures remain inert data and diagnostics cover requested hints", () => {
  const xss = '<script>globalThis.pwned=true</script><a href="javascript:alert(1)">x</a>';
  const value = H.parseMetadata(JSON.stringify({hash: "old", component: xss, pipeline: {workflow: {components: {}, steps: [{kind: "invoke", component: xss}]}}}));
  assert.equal(value.component, xss); assert.equal(context.pwned, undefined);
  const labels = H.buildMermaid(value); assert.doesNotMatch(labels, /<script|javascript:/);
  const messages = H.diagnostics(value).map(x => x.message).join(" ");
  assert.match(messages, /Missing schema_version/); assert.match(messages, /retired/); assert.match(messages, /missing component/);
});

test("current hashless repository fixtures parse with complete leaf accounting", () => {
  const fixtureRoot = new URL("fixtures/", import.meta.url);
  const files = readdirSync(fixtureRoot, {recursive: true})
    .filter(name => name.endsWith("inference_metadata.yaml"));
  assert.ok(files.length >= 20);
  for (const name of files) {
    const value = H.parseMetadata(readFileSync(new URL(name, fixtureRoot), "utf8"));
    const leaves = H.enumerateLeaves(value);
    assert.ok(leaves.length > 0, name);
    assert.equal(new Set(leaves.map(x => x.path)).size, leaves.length, name);
    assert.equal(Object.hasOwn(value, "hash"), false, name);
  }
});

test("diagnostics identify missing image value range without claiming schema validation", () => {
  const value = H.parseMetadata("schema_version: v1\npipeline: {workflow: {outputs: {image: {role: image, contract: {dtype: float32, rank: 4}}}}}\n");
  assert.match(H.diagnostics(value).map(x => x.message).join(" "), /value_range/);
});

test("HTML exposes all purpose-built views and user actions", () => {
  for (const text of ["Document & core conformance", "Model, package & execution", "Workflow control-flow graph",
    "Nested loop & conditional timeline", "Serving state groups & K/V aliases", "Preprocessing, tokenizer & media",
    "Policy component help", "Adapters, speculative execution & deployment", "Metadata extension registry",
    "Core schema conformance is not a capability", "Runtime optimizations",
    "Workflow quick reference", "Export", "Print", "Reset"]) {
    assert.ok(html.includes(text), text);
  }
  assert.ok(!html.includes("Document & capabilities"));
  assert.ok(!html.includes("Capability catalogue"));
});

// --- Blocker 1: Mermaid labels must never let metadata inject graph/HTML syntax.
const decodeMermaidLabel = quoted => {
  assert.match(quoted, /^"[^"]*"$/, `label is not safely quoted: ${quoted}`);
  return quoted.slice(1, -1);
};

test("mermaidText produces readable, bounded, sanitized quoted labels", () => {
  const adversarial = [
    'A] --> Evil[pwn]',            // bracket + edge-arrow node injection
    'a{b}c|d',                     // rhombus braces + edge-label pipe
    'q"uote and ; semicolon',      // string break-out + statement separator
    "line1\nline2\ttab",           // newline / tab control characters
    "\tleading tab and trailing spaces   ", // leading/trailing whitespace
    "a    b\n\n  c",               // runs of whitespace, blank lines
    '<img src=x onerror=alert(1)>',// HTML event handler
    '<script>globalThis.pwned=1</script>', // script element
    'javascript:alert(1)',         // dangerous URL scheme
    'A & B --- C ==> D',           // ampersand + open/thick links
    'id@{shape: rect}',            // Mermaid @-metadata syntax
    'héllo · 世界 · 😀🧬 · \u202Ereversed', // Unicode incl. astral pairs + bidi control
    // A value far longer than any old truncation limit (200 code points) must survive whole.
    'Z]-->{'.repeat(20) + '😀'.repeat(20) + 'tail; end',
  ];
  for (const payload of adversarial) {
    const label = H.mermaidText(payload);
    const decoded = decodeMermaidLabel(label);
    assert.equal(decoded, H.safeLabel(payload).replace(/\\/g, " "), payload);
    assert.doesNotMatch(decoded, /[\u0000-\u001f\u007f<>"`{}|\\]/, payload);
    assert.doesNotMatch(decoded, /javascript:|onerror|<script|<img|alert\s*\(/i, payload);
    assert.ok([...decoded].length <= 76, payload);
  }
  // Graph labels remain compact; the full metadata value remains available in
  // the timeline and recursive document views.
  const long = "🧬".repeat(130); // 130 astral code points
  assert.equal([...decodeMermaidLabel(H.mermaidText(long))].length, 76);
  assert.equal(decodeMermaidLabel(H.mermaidText("")), "unnamed");
  assert.equal(decodeMermaidLabel(H.mermaidText(null)), "unnamed");
});

test("adversarial workflow labels inject no Mermaid nodes, edges, or markup", () => {
  const payload = 'X"] --> pwned[bad]{q}|e; <img src=x onerror=alert(1)> A-->B';
  const value = H.parseMetadata(JSON.stringify({
    schema_version: "v1",
    pipeline: {workflow: {steps: [{kind: "invoke", component: payload}]}},
  }));
  const graph = H.workflowGraph(value);
  // A single declared step plus the viewer-built terminal yields one edge.
  assert.equal(graph.nodes.length, 2);
  assert.equal(graph.edges.length, 1);
  const step = nodeId(graph, "1. invoke");
  const end = nodeId(graph, "End");
  assert.ok(hasEdge(graph, step, end, null));
  const src = H.buildMermaid(value);
  assert.match(src, /^flowchart TD/);
  // Exactly one viewer-built edge line; arrows inside a quoted label are inert.
  assert.equal(src.split("\n").filter(l => /^\s*s\d+\s+-->/.test(l)).length, 1);
  assert.doesNotMatch(src, /onerror|<img|alert\(1\)|javascript:/i);
  const nodeLine = src.split("\n").find(l => /^\s*s0/.test(l));
  const label = nodeLine.match(/^\s*s0\[("[^"]*")\]$/);
  assert.ok(label, nodeLine);
  assert.equal(decodeMermaidLabel(label[1]), H.safeLabel("1. invoke · " + payload));
});

// --- Blocker 2: structured control-flow graph with correct edges, not a linear chain.
const nodeId = (graph, titleSubstring) => {
  const found = graph.nodes.filter(n => n.title.includes(titleSubstring));
  assert.equal(found.length, 1, `expected exactly one node containing ${titleSubstring}`);
  return found[0].id;
};
const hasEdge = (graph, from, to, label) =>
  graph.edges.some(e => e.from === from && e.to === to && (e.label ?? null) === (label ?? null));

test("conditional steps emit then/else branch edges that rejoin the following step", () => {
  const value = H.parseMetadata(`schema_version: v1
pipeline:
  workflow:
    steps:
      - kind: conditional
        then:
          - {kind: invoke, component: ThenComponent}
        else:
          - {kind: invoke, component: ElseComponent}
      - {kind: invoke, component: AfterComponent}
`);
  const g = H.workflowGraph(value);
  const cond = nodeId(g, "conditional");
  const thenN = nodeId(g, "ThenComponent");
  const elseN = nodeId(g, "ElseComponent");
  const after = nodeId(g, "AfterComponent");
  const end = nodeId(g, "End");
  // Decision fans out to both arms, and both arms converge on the next sibling.
  assert.ok(hasEdge(g, cond, thenN, "then"));
  assert.ok(hasEdge(g, cond, elseN, "else"));
  assert.ok(hasEdge(g, thenN, after, null));
  assert.ok(hasEdge(g, elseN, after, null));
  assert.ok(hasEdge(g, after, end, null));
  // No misleading straight-line chain: the decision never links directly to next.
  assert.ok(!hasEdge(g, cond, after, null));
  assert.equal(g.edges.length, 5);
});

test("on_true/on_false conditionals branch and an absent arm falls through", () => {
  const value = H.parseMetadata(`schema_version: v1
pipeline:
  workflow:
    steps:
      - kind: if
        on_true:
          - {kind: invoke, component: TrueBranch}
      - {kind: invoke, component: JoinComponent}
`);
  const g = H.workflowGraph(value);
  const cond = nodeId(g, "1. if");
  const truth = nodeId(g, "TrueBranch");
  const join = nodeId(g, "JoinComponent");
  const end = nodeId(g, "End");
  assert.ok(hasEdge(g, cond, truth, "on_true"));
  assert.ok(hasEdge(g, truth, join, null));
  // The missing false arm falls through directly to the join step, labelled "else".
  assert.ok(hasEdge(g, cond, join, "else"));
  assert.ok(hasEdge(g, join, end, null));
  assert.equal(g.edges.length, 4);
});

test("loop steps emit body, back-edge, and exit edges around the loop node", () => {
  const value = H.parseMetadata(`schema_version: v1
pipeline:
  workflow:
    steps:
      - kind: loop
        setup:
          - {kind: invoke, component: InitComponent}
        steps:
          - {kind: invoke, component: BodyComponent}
      - {kind: invoke, component: ExitComponent}
`);
  const g = H.workflowGraph(value);
  const loop = nodeId(g, "1. loop");
  const init = nodeId(g, "InitComponent");
  const body = nodeId(g, "BodyComponent");
  const exit = nodeId(g, "ExitComponent");
  const end = nodeId(g, "End");
  // Setup runs once into the loop; body carries a back-edge; exit leaves the loop.
  assert.equal(g.entry, init);
  assert.ok(hasEdge(g, init, loop, null));
  assert.ok(hasEdge(g, loop, body, "body"));
  assert.ok(hasEdge(g, body, loop, "repeat"));
  assert.ok(hasEdge(g, loop, exit, "predicate false / limit"));
  assert.ok(hasEdge(g, exit, end, null));
  // The loop must not be flattened into a straight line through its body.
  assert.ok(!hasEdge(g, body, exit, null));
  assert.equal(g.edges.length, 5);
});
