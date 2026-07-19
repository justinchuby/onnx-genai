// Copyright (c) Microsoft Corporation.
//
// Minimal dependency-free API for the onnx-genai diffusion demo. It drives the
// real onnx-genai binaries: `comfyui_to_metadata` (ComfyUI -> native config)
// and `run_diffusion` (runs a pipeline, dumping each reverse-process step).

import { createServer } from "node:http";
import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync, mkdtempSync, readdirSync, existsSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import YAML from "yaml";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "..", "..", "..");
const PORT = process.env.PORT ? Number(process.env.PORT) : 8787;

function findBinary(name) {
  // Prefer the optimized release build; only fall back to debug (much slower
  // for real models) with a loud warning so the demo stays high-performance.
  const release = join(REPO, "target", "release", name);
  if (existsSync(release)) return release;
  const debug = join(REPO, "target", "debug", name);
  if (existsSync(debug)) {
    console.warn(
      `[diffusion-demo] WARNING: using DEBUG build of '${name}' (slow). ` +
        `For high performance run: cargo build --release -p onnx-genai --bin ${name} ` +
        `(see README).`
    );
    return debug;
  }
  throw new Error(`binary '${name}' not found under target/{release,debug}; build it first (see README)`);
}

function ortLibDir() {
  // Locate <repo>/target/*/build/onnx-genai-ort-sys-*/out/ort-prebuilt/lib,
  // preferring the release profile so it matches the release binary.
  const targetDir = join(REPO, "target");
  const collect = (profile) => {
    const found = [];
    const buildDir = join(targetDir, profile, "build");
    for (const entry of safeReaddir(buildDir)) {
      if (!entry.startsWith("onnx-genai-ort-sys-")) continue;
      const lib = join(buildDir, entry, "out", "ort-prebuilt", "lib");
      if (existsSync(lib)) found.push(lib);
    }
    return found.sort().at(-1);
  };
  return collect("release") ?? collect("debug") ?? "";
}

function safeReaddir(dir) {
  try {
    return readdirSync(dir).filter((e) => {
      try {
        return statSync(join(dir, e)).isDirectory();
      } catch {
        return false;
      }
    });
  } catch {
    return [];
  }
}

const LM_PACKAGE = process.env.ONNX_GENAI_LM_PACKAGE || join(REPO, "tests", "fixtures", "tiny-masked-diffusion");
const SD_PACKAGE = process.env.ONNX_GENAI_SD_PACKAGE || "";

function readBody(req) {
  return new Promise((res) => {
    let data = "";
    req.on("data", (c) => (data += c));
    req.on("end", () => res(data));
  });
}

function json(res, code, obj) {
  res.writeHead(code, {
    "content-type": "application/json",
    "access-control-allow-origin": "*",
    "access-control-allow-headers": "content-type",
  });
  res.end(JSON.stringify(obj));
}

// Translate a ComfyUI API-format workflow JSON into the native config + run params.
function translateComfyui(workflowJson) {
  const bin = findBinary("comfyui_to_metadata");
  const r = spawnSync(bin, [], { input: workflowJson, encoding: "utf8", maxBuffer: 64 << 20 });
  if (r.status !== 0) throw new Error(r.stderr || "comfyui_to_metadata failed");
  return JSON.parse(r.stdout);
}

// Run an iterative pipeline, dumping each step; return the per-step frames and
// the runtime timing parsed from run_diffusion (`load` = model/session load,
// `run` = the pure reverse-process loop, i.e. the ComfyUI-comparable it/s time).
function runPipelineWithDump(packageDir, outputEndpoint, inputs) {
  const bin = findBinary("run_diffusion");
  const dump = mkdtempSync(join(tmpdir(), "ogsteps-"));
  const args = [packageDir, outputEndpoint, join(dump, "out.bin"), ...inputs];
  const r = spawnSync(bin, args, {
    encoding: "utf8",
    maxBuffer: 256 << 20,
    env: { ...process.env, ONNX_GENAI_STEP_DUMP_DIR: dump, DYLD_LIBRARY_PATH: `${ortLibDir()}:${process.env.DYLD_LIBRARY_PATH || ""}` },
  });
  if (r.status !== 0) throw new Error(r.stderr || "run_diffusion failed");
  const frames = readdirSync(dump)
    .filter((f) => f.startsWith("step_") && f.endsWith(".json"))
    .sort()
    .map((f) => JSON.parse(readFileSync(join(dump, f), "utf8")));
  const timingMatch = /\[timing\]\s*load=([\d.]+)ms\s*run=([\d.]+)ms/.exec(r.stderr || "");
  const timing = timingMatch
    ? { loadMs: Number(timingMatch[1]), runMs: Number(timingMatch[2]) }
    : null;
  const stagesPath = join(dump, "stages.json");
  const stages = existsSync(stagesPath)
    ? JSON.parse(readFileSync(stagesPath, "utf8")).stages ?? []
    : [];
  return { frames, timing, stages };
}

// Language diffusion: seed an all-mask sequence and run masked_diffusion.
function runLanguage() {
  const metaPath = join(LM_PACKAGE, "inference_metadata.yaml");
  const metaText = existsSync(metaPath) ? readFileSync(metaPath, "utf8") : "";
  const maskId = Number(/mask_token_id:\s*(-?\d+)/.exec(metaText)?.[1] ?? 1);
  const numSteps = Number(/num_steps:\s*(\d+)/.exec(metaText)?.[1] ?? 4);
  // Sequence length: the fixture is 4; a real LM would size this from the prompt.
  const seqLen = Number(process.env.ONNX_GENAI_LM_SEQ_LEN || 4);
  const seedPath = join(mkdtempSync(join(tmpdir(), "ogseed-")), "seed.i64");
  const buf = Buffer.alloc(seqLen * 8);
  for (let i = 0; i < seqLen; i++) buf.writeBigInt64LE(BigInt(maskId), i * 8);
  writeFileSync(seedPath, buf);
  const { frames, timing, stages } = runPipelineWithDump(LM_PACKAGE, "denoiser.input_ids", [
    `denoiser.input_ids:i64:1,${seqLen}:${seedPath}`,
  ]);
  const metadata = metaText ? YAML.parse(metaText) : null;
  const perf = timing
    ? {
        loadMs: timing.loadMs,
        runMs: timing.runMs,
        numSteps,
        // it/s, exactly as ComfyUI reports it: reverse-process steps per second.
        stepsPerSecond: timing.runMs > 0 ? (numSteps / timing.runMs) * 1000 : null,
        msPerStep: numSteps > 0 ? timing.runMs / numSteps : null,
        // Per-pipeline-stage timings (encode / denoise / decode).
        stages,
        // Per reverse-process step wall-clock (ms), in step order.
        stepMs: frames.map((f) => f.step_ms ?? null),
      }
    : { stages, stepMs: frames.map((f) => f.step_ms ?? null) };
  return { kind: "language", maskId, numSteps, seqLen, metadata, frames, perf };
}

const server = createServer(async (req, res) => {
  if (req.method === "OPTIONS") return json(res, 204, {});
  try {
    if (req.url === "/api/translate-comfyui" && req.method === "POST") {
      return json(res, 200, translateComfyui(await readBody(req)));
    }
    if (req.url === "/api/parse-native" && req.method === "POST") {
      // Accept a native inference_metadata document as YAML or JSON.
      return json(res, 200, { metadata: YAML.parse(await readBody(req)) });
    }
    if (req.url === "/api/run/language" && req.method === "POST") {
      return json(res, 200, runLanguage());
    }
    if (req.url === "/api/run/image" && req.method === "POST") {
      if (!SD_PACKAGE) {
        return json(res, 400, { error: "no Stable Diffusion package configured; set ONNX_GENAI_SD_PACKAGE (see README)" });
      }
      // Image runs are wired the same way (run_diffusion + step dumps); left to
      // the operator to point at a built SD package. Returns the config + note.
      const metaPath = join(SD_PACKAGE, "inference_metadata.yaml");
      return json(res, 200, {
        kind: "image",
        package: SD_PACKAGE,
        metadata: existsSync(metaPath) ? YAML.parse(readFileSync(metaPath, "utf8")) : null,
        note: "image run wiring uses run_diffusion + step dumps; use run_comfyui for the full text-encode + VAE decode path",
      });
    }
    if (req.url === "/api/health") return json(res, 200, { ok: true, lmPackage: LM_PACKAGE, sdPackage: SD_PACKAGE || null });
    return json(res, 404, { error: "not found" });
  } catch (e) {
    return json(res, 500, { error: String(e.message || e) });
  }
});

server.listen(PORT, () => console.log(`[diffusion-demo] API on http://localhost:${PORT}`));
