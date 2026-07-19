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
  if (process.platform === "win32" && !name.endsWith(".exe")) name += ".exe";
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

// ---- GPT-2 byte-level tokenizer decode (dependency-free) ----
// Reverses the standard GPT-2 bytes<->unicode table so we can turn token ids
// back into human-readable text for the language-diffusion animation.
function gpt2ByteDecoder() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const decoder = new Map(); // unicode codepoint -> original byte
  for (let i = 0; i < bs.length; i++) decoder.set(cs[i], bs[i]);
  return decoder;
}

const tokenizerCache = new Map(); // packageDir -> { idToToken, byteDecoder } | null
function loadTokenizer(packageDir) {
  if (tokenizerCache.has(packageDir)) return tokenizerCache.get(packageDir);
  let entry = null;
  const path = join(packageDir, "tokenizer.json");
  if (existsSync(path)) {
    try {
      const tk = JSON.parse(readFileSync(path, "utf8"));
      const vocab = tk?.model?.vocab ?? {};
      const idToToken = [];
      for (const [tokenStr, id] of Object.entries(vocab)) idToToken[id] = tokenStr;
      entry = { idToToken, byteDecoder: gpt2ByteDecoder() };
    } catch {
      entry = null;
    }
  }
  tokenizerCache.set(packageDir, entry);
  return entry;
}

// Decode a single token id to its display text (byte-level -> UTF-8). Returns
// null when the tokenizer is missing, the id is unknown, or the token is a
// special (non-byte-level) token such as <mask>.
function decodeToken(tokenizer, id) {
  if (!tokenizer) return null;
  const s = tokenizer.idToToken[id];
  if (s === undefined) return null;
  const bytes = [];
  for (const ch of s) {
    const b = tokenizer.byteDecoder.get(ch.codePointAt(0));
    if (b === undefined) return s; // special token, show verbatim
    bytes.push(b);
  }
  return Buffer.from(bytes).toString("utf8");
}

// Map an SD-1.5 latent ([1,4,H,W] f32) to a small RGB preview using the
// well-known linear latent->RGB approximation (as used by ComfyUI/A1111 for
// live previews). Returns { w, h, rgb } with rgb a base64 raw RGB24 buffer.
function latentToRgbPreview(data, shape) {
  const h = shape[shape.length - 2];
  const w = shape[shape.length - 1];
  const plane = h * w;
  const factors = [
    [0.298, 0.207, 0.208],
    [0.187, 0.286, 0.173],
    [-0.158, 0.189, 0.264],
    [-0.184, -0.271, -0.473],
  ];
  const out = Buffer.alloc(plane * 3);
  for (let p = 0; p < plane; p++) {
    let r = 0.5;
    let g = 0.5;
    let b = 0.5;
    for (let c = 0; c < 4; c++) {
      const v = data[c * plane + p];
      r += v * factors[c][0];
      g += v * factors[c][1];
      b += v * factors[c][2];
    }
    const o = p * 3;
    out[o] = Math.max(0, Math.min(255, Math.round(r * 255)));
    out[o + 1] = Math.max(0, Math.min(255, Math.round(g * 255)));
    out[o + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
  }
  return { w, h, rgb: out.toString("base64") };
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
  // Decode token ids -> readable text so the UI can animate real words filling
  // in (masked positions stay null). Falls back to numeric ids if no tokenizer.
  const tokenizer = loadTokenizer(LM_PACKAGE);
  const framesWithText = frames.map((f) => {
    const data = f.data.slice(-seqLen);
    const text = data.map((v) => (v === maskId ? null : decodeToken(tokenizer, v)));
    return { ...f, text };
  });
  const finalData = framesWithText.length
    ? framesWithText[framesWithText.length - 1].data.slice(-seqLen)
    : [];
  const decoded = tokenizer
    ? finalData
        .filter((v) => v !== maskId)
        .map((v) => decodeToken(tokenizer, v) ?? "")
        .join("")
        .trim()
    : null;
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
  return { kind: "language", maskId, numSteps, seqLen, metadata, frames: framesWithText, decoded, tokenizer: !!tokenizer, perf };
}

// Render a full image (text-encode + denoise + VAE decode -> PNG) with
// `render_sd`, the driver for the from-scratch Mobius Stable Diffusion 1.x
// package. Accepts a user prompt plus step-count/guidance/seed controls, runs
// the real iterative pipeline, and returns the PNG as a data URL plus per-step
// latent previews and timing. Honors the execution provider from the
// environment (set ONNX_GENAI_EP=metal / cuda in the launching shell; see
// README).
function runImage(options) {
  const prompt = typeof options.prompt === "string" ? options.prompt.trim() : "";
  if (!prompt) throw new Error("prompt is required");
  const negative = typeof options.negative === "string" ? options.negative : "";
  const steps = clampInt(options.steps, 1, 100, 25);
  const guidance = clampFloat(options.guidance, 0, 30, 7.5);
  const seed = clampInt(options.seed, 0, Number.MAX_SAFE_INTEGER, 0);

  const bin = findBinary("render_sd");
  const outPng = join(mkdtempSync(join(tmpdir(), "ogimg-")), "out.png");
  const dump = mkdtempSync(join(tmpdir(), "ogimgsteps-"));
  const started = Date.now();
  const r = spawnSync(
    bin,
    [
      "--pipeline-dir", SD_PACKAGE,
      "--prompt", prompt,
      "--negative", negative,
      "--steps", String(steps),
      "--guidance", String(guidance),
      "--seed", String(seed),
      "--output", outPng,
    ],
    {
      encoding: "utf8",
      maxBuffer: 64 << 20,
      env: {
        ...process.env,
        ONNX_GENAI_STEP_DUMP_DIR: dump,
        DYLD_LIBRARY_PATH: `${ortLibDir()}:${process.env.DYLD_LIBRARY_PATH || ""}`,
      },
    }
  );
  if (r.status !== 0) throw new Error(r.stderr || "render_sd failed");
  const wallMs = Date.now() - started;
  if (!existsSync(outPng)) throw new Error(`render_sd reported success but ${outPng} is missing`);
  const image = `data:image/png;base64,${readFileSync(outPng).toString("base64")}`;

  // render_sd prints a machine-readable timing summary on stdout.
  let summary = null;
  try {
    summary = JSON.parse((r.stdout || "").trim().split("\n").at(-1) ?? "");
  } catch {
    summary = null;
  }

  // Per-step latent previews (noise -> image) for the denoising animation, plus
  // the per-step wall-clock the engine records in each dump.
  const stepFiles = readdirSync(dump)
    .filter((f) => f.startsWith("step_") && f.endsWith(".json"))
    .sort();
  const frames = [];
  const stepMs = [];
  for (const f of stepFiles) {
    const j = JSON.parse(readFileSync(join(dump, f), "utf8"));
    frames.push(latentToRgbPreview(j.data, j.shape));
    stepMs.push(typeof j.step_ms === "number" ? j.step_ms : null);
  }
  const stagesPath = join(dump, "stages.json");
  const stages = existsSync(stagesPath)
    ? JSON.parse(readFileSync(stagesPath, "utf8")).stages ?? []
    : [];

  const renderMatch =
    /\[render\]\s*finite=(\w+)\s*min=([-\d.]+)\s*max=([-\d.]+)\s*mean=([-\d.]+)/.exec(r.stderr || "");
  const render = renderMatch
    ? {
        finite: renderMatch[1] === "true",
        min: Number(renderMatch[2]),
        max: Number(renderMatch[3]),
        mean: Number(renderMatch[4]),
      }
    : null;

  const denoiseMs = summary?.denoise_ms ?? null;
  const perf = {
    numSteps: steps,
    runMs: denoiseMs,
    // it/s = reverse-process steps per second (same metric ComfyUI reports).
    stepsPerSecond:
      summary?.steps_per_second ??
      (denoiseMs && denoiseMs > 0 ? (steps / denoiseMs) * 1000 : null),
    msPerStep: denoiseMs && steps > 0 ? denoiseMs / steps : null,
    stages,
    stepMs,
  };

  const metaPath = join(SD_PACKAGE, "inference_metadata.yaml");
  return {
    kind: "image",
    package: SD_PACKAGE,
    prompt,
    negative,
    steps,
    guidance,
    seed,
    metadata: existsSync(metaPath) ? YAML.parse(readFileSync(metaPath, "utf8")) : null,
    image,
    frames,
    wallMs,
    render,
    perf,
  };
}

// Parse and clamp an integer request field, falling back to `fallback` when the
// value is absent or not a finite number.
function clampInt(value, min, max, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.round(n)));
}

// Parse and clamp a floating-point request field, falling back to `fallback`
// when the value is absent or not a finite number.
function clampFloat(value, min, max, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
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
      const body = await readBody(req);
      let options = {};
      if (body.trim()) {
        try {
          options = JSON.parse(body);
        } catch {
          return json(res, 400, { error: "request body must be JSON" });
        }
      }
      return json(res, 200, runImage(options));
    }
    if (req.url === "/api/health") return json(res, 200, { ok: true, lmPackage: LM_PACKAGE, sdPackage: SD_PACKAGE || null });
    return json(res, 404, { error: "not found" });
  } catch (e) {
    return json(res, 500, { error: String(e.message || e) });
  }
});

server.listen(PORT, () => console.log(`[diffusion-demo] API on http://localhost:${PORT}`));
