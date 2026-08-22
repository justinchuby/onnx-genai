import assert from "node:assert/strict";
import {readFileSync, readdirSync} from "node:fs";
import test from "node:test";
import vm from "node:vm";

const htmlPath = new URL("../examples/inference_metadata/visualizer.html", import.meta.url);
const html = readFileSync(htmlPath, "utf8");
const script = id => {
  const match = html.match(new RegExp(`<script id="${id}">([\\s\\S]*?)<\\/script>`));
  assert.ok(match, `missing ${id}`);
  return match[1];
};
const context = {console};
context.globalThis = context;
vm.createContext(context);
vm.runInContext(script("vendor-js-yaml"), context);
vm.runInContext(script("visualizer-helpers"), context);
const H = context.VisualizerHelpers;

test("is a genuinely offline single-file visualizer with pinned license notices", () => {
  for (const tag of html.match(/<(?:script|link|img)\b[^>]*>/gi) ?? []) {
    assert.doesNotMatch(tag, /\bsrc=["']https?:|\bhref=["']https?:|<script[^>]+\bsrc=/i);
  }
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
  for (const text of ["Document & capabilities", "Model, package & execution", "Workflow DAG / control flow",
    "Nested loop & conditional timeline", "Serving state groups & K/V aliases", "Preprocessing, tokenizer & media",
    "Policy component help", "Adapters, speculative execution & deployment", "Export JSON", "Print", "Reset"]) {
    assert.ok(html.includes(text), text);
  }
});
