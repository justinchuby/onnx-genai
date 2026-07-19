import "./style.css";

// ---- Types matching the native inference_metadata pipeline schema ----
interface ModelSpec {
  filename?: string;
  type?: string;
}
interface Dataflow {
  from: string;
  to: string;
}
interface Strategy {
  kind?: string;
  denoiser?: string;
  num_steps?: number;
  timestep_input?: string;
  guidance_scale?: number;
  cfg_conditioning_input?: string;
  start_step?: number;
  scheduler_config?: Record<string, unknown>;
}
interface Pipeline {
  models?: Record<string, ModelSpec>;
  dataflow?: Dataflow[];
  strategy?: Strategy;
  phases?: Record<string, { run_on?: string }>;
}
interface Metadata {
  pipeline?: Pipeline;
}

interface Frame {
  step: number;
  denoiser: string;
  port: string;
  dtype: string;
  shape: number[];
  data: number[];
  text?: (string | null)[];
}

interface ImageFrame {
  w: number;
  h: number;
  rgb: string; // base64 raw RGB24
}

const el = (tag: string, cls?: string, text?: string): HTMLElement => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

async function postText(url: string, body: string): Promise<any> {
  const r = await fetch(url, { method: "POST", body });
  const text = await r.text();
  let parsed: any;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error(text);
  }
  if (!r.ok) throw new Error(parsed?.error ?? text);
  return parsed;
}

// ---- Pipeline visualization ----
function roleClass(type?: string): string {
  if (type === "denoiser") return "denoiser";
  if (type === "encoder") return "encoder";
  if (type === "vae") return "vae";
  return "";
}

function renderPipeline(meta: Metadata): HTMLElement {
  const wrap = el("div");
  const pipe = meta.pipeline;
  if (!pipe) {
    wrap.appendChild(el("div", "err", "No `pipeline` block found in this config."));
    return wrap;
  }

  const models = pipe.models ?? {};
  const strategy = pipe.strategy ?? {};
  const phases = pipe.phases ?? {};

  // Order components: encoders first, denoiser center, vae last.
  const order = (name: string): number => {
    const t = models[name]?.type;
    if (t === "encoder") return 0;
    if (t === "denoiser") return 1;
    if (t === "vae") return 2;
    return 1.5;
  };
  const names = Object.keys(models).sort((a, b) => order(a) - order(b));

  // --- DAG row ---
  const dagPanel = el("div", "panel");
  dagPanel.appendChild(el("h2", undefined, "Pipeline graph"));
  const dag = el("div", "dag");
  names.forEach((name, i) => {
    const spec = models[name];
    const node = el("div", `node ${roleClass(spec.type)}`);
    node.appendChild(el("div", "role", spec.type ?? "model"));
    node.appendChild(el("div", "name", name));
    if (spec.filename) node.appendChild(el("div", "file", spec.filename));
    if (phases[name]?.run_on) node.appendChild(el("div", "file", `run: ${phases[name].run_on}`));
    if (name === strategy.denoiser) node.appendChild(el("div", "loop-badge", "↻ iterative loop"));
    dag.appendChild(node);
    if (i < names.length - 1) dag.appendChild(el("div", "arrow", "→"));
  });
  dagPanel.appendChild(dag);

  // --- dataflow edges ---
  const edges = pipe.dataflow ?? [];
  if (edges.length) {
    const edgeBox = el("div", "edges");
    edgeBox.appendChild(el("div", undefined, "dataflow:"));
    edges.forEach((e) => {
      const [fromNode] = e.from.split(".");
      const [toNode] = e.to.split(".");
      const isSelf = fromNode === toNode;
      const line = el("div", isSelf ? "self" : undefined, `  ${e.from}  →  ${e.to}${isSelf ? "   (loop-carried)" : ""}`);
      edgeBox.appendChild(line);
    });
    dagPanel.appendChild(edgeBox);
  }

  // --- strategy card ---
  const stratPanel = el("div", "panel");
  stratPanel.appendChild(el("h2", undefined, "Strategy"));
  const grid = el("div", "strategy");
  const sched = strategy.scheduler_config ?? {};
  const chips: [string, unknown][] = [
    ["kind", strategy.kind],
    ["denoiser", strategy.denoiser],
    ["num_steps", strategy.num_steps],
    ["timestep_input", strategy.timestep_input],
    ["guidance_scale", strategy.guidance_scale],
    ["cfg_conditioning", strategy.cfg_conditioning_input],
    ["start_step", strategy.start_step],
    ["scheduler", (sched as any).kind],
  ];
  for (const [k, v] of chips) {
    if (v === undefined || v === null) continue;
    const chip = el("div", "chip");
    chip.appendChild(el("div", "k", k));
    chip.appendChild(el("div", "v", String(v)));
    grid.appendChild(chip);
  }
  // remaining scheduler_config keys
  for (const [k, v] of Object.entries(sched)) {
    if (k === "kind") continue;
    const chip = el("div", "chip");
    chip.appendChild(el("div", "k", `scheduler.${k}`));
    chip.appendChild(el("div", "v", typeof v === "object" ? JSON.stringify(v) : String(v)));
    grid.appendChild(chip);
  }
  stratPanel.appendChild(grid);

  wrap.appendChild(dagPanel);
  wrap.appendChild(stratPanel);
  return wrap;
}

interface StageTiming {
  component: string;
  phase: string;
  ms: number;
  steps?: number;
}
interface Perf {
  loadMs?: number;
  runMs?: number;
  numSteps?: number;
  stepsPerSecond?: number | null;
  msPerStep?: number | null;
  stages?: StageTiming[];
  stepMs?: (number | null)[];
}

// Render a prominent speed card (it/s), directly comparable to ComfyUI,
// plus per-pipeline-stage and per-step timing breakdowns.
function renderPerf(perf: Perf | null | undefined): HTMLElement | null {
  if (!perf) return null;
  const card = el("div", "perf");
  if (perf.stepsPerSecond != null) {
    card.appendChild(el("div", "perf-big", `${perf.stepsPerSecond.toFixed(1)} it/s`));
    const detail = el(
      "div",
      "perf-detail",
      `${perf.numSteps} steps · ${perf.runMs?.toFixed(1)} ms loop` +
        (perf.msPerStep != null ? ` · ${perf.msPerStep.toFixed(2)} ms/step avg` : "") +
        (perf.loadMs != null ? ` · model load ${perf.loadMs.toFixed(0)} ms (excluded)` : "")
    );
    card.appendChild(detail);
    card.appendChild(
      el("div", "perf-note", "it/s = reverse-process steps per second (same metric ComfyUI reports)")
    );
  }

  // Per-pipeline-stage timing (encode / denoise / decode).
  if (perf.stages && perf.stages.length) {
    const stageBox = el("div", "timing-block");
    stageBox.appendChild(el("div", "timing-title", "Pipeline stages"));
    const maxMs = Math.max(...perf.stages.map((s) => s.ms), 0.0001);
    for (const s of perf.stages) {
      const rowEl = el("div", "timing-row");
      const name = s.steps ? `${s.component} (${s.phase}, ${s.steps} steps)` : `${s.component} (${s.phase})`;
      rowEl.appendChild(el("span", "timing-name", name));
      const bar = el("div", "timing-bar");
      const fill = el("div", "timing-fill");
      fill.style.width = `${Math.max(2, (s.ms / maxMs) * 100)}%`;
      bar.appendChild(fill);
      rowEl.appendChild(bar);
      rowEl.appendChild(el("span", "timing-ms", `${s.ms.toFixed(2)} ms`));
      stageBox.appendChild(rowEl);
    }
    card.appendChild(stageBox);
  }

  // Per reverse-process step timing.
  const stepMs = (perf.stepMs ?? []).filter((v): v is number => typeof v === "number");
  if (stepMs.length) {
    const stepBox = el("div", "timing-block");
    stepBox.appendChild(el("div", "timing-title", "Per-step time"));
    const maxMs = Math.max(...stepMs, 0.0001);
    stepMs.forEach((ms, i) => {
      const rowEl = el("div", "timing-row");
      rowEl.appendChild(el("span", "timing-name", `step ${i + 1}`));
      const bar = el("div", "timing-bar");
      const fill = el("div", "timing-fill step");
      fill.style.width = `${Math.max(2, (ms / maxMs) * 100)}%`;
      bar.appendChild(fill);
      rowEl.appendChild(bar);
      rowEl.appendChild(el("span", "timing-ms", `${ms.toFixed(2)} ms`));
      stepBox.appendChild(rowEl);
    });
    card.appendChild(stepBox);
  }

  return card.childNodes.length ? card : null;
}

// ---- Language un-masking animation ----
function renderLanguageRun(container: HTMLElement, frames: Frame[], maskId: number, perf?: Perf | null) {
  container.innerHTML = "";
  if (!frames.length) {
    container.appendChild(el("div", "err", "No frames returned."));
    return;
  }
  const perfCard = renderPerf(perf);
  if (perfCard) container.appendChild(perfCard);
  const last = frames[frames.length - 1];
  const seqLen = last.shape[last.shape.length - 1];

  const sentence = el("div", "sentence");
  container.appendChild(sentence);

  const grid = el("div", "tokens");
  const cells: HTMLElement[] = [];
  for (let i = 0; i < seqLen; i++) {
    const c = el("div", "tok", "▒");
    grid.appendChild(c);
    cells.push(c);
  }
  container.appendChild(grid);

  const controls = el("div", "controls");
  const playBtn = el("button", undefined, "▶ Play") as HTMLButtonElement;
  const slider = el("input") as HTMLInputElement;
  slider.type = "range";
  slider.min = "0";
  slider.max = String(frames.length - 1);
  slider.value = "0";
  const label = el("div", "note");
  controls.appendChild(playBtn);
  controls.appendChild(slider);
  controls.appendChild(label);
  container.appendChild(controls);

  let prev: number[] = [];
  const show = (idx: number) => {
    const f = frames[idx];
    const data = f.data.slice(-seqLen);
    const text = f.text;
    const pieces: string[] = [];
    for (let i = 0; i < seqLen; i++) {
      const v = data[i];
      const masked = v === maskId;
      const justFilled = !masked && prev[i] === maskId;
      const decoded = text && text[i] != null ? (text[i] as string) : null;
      cells[i].textContent = masked ? "▒" : decoded ? decoded.trim() || "␣" : String(v);
      cells[i].className = "tok" + (masked ? "" : " filled") + (justFilled ? " just" : "");
      pieces.push(masked ? "▒" : decoded ?? String(v));
    }
    prev = data;
    sentence.textContent = pieces.join("");
    const remaining = data.filter((v) => v === maskId).length;
    label.textContent = `step ${f.step + 1}/${frames.length} · ${seqLen - remaining}/${seqLen} unmasked`;
    slider.value = String(idx);
  };

  slider.addEventListener("input", () => {
    prev = []; // recompute highlight fresh when scrubbing
    show(Number(slider.value));
  });

  let timer: number | undefined;
  const stop = () => {
    if (timer) clearInterval(timer);
    timer = undefined;
    playBtn.textContent = "▶ Play";
  };
  const play = () => {
    if (timer) return stop();
    playBtn.textContent = "⏸ Pause";
    let idx = 0;
    prev = [];
    show(0);
    timer = window.setInterval(() => {
      idx++;
      if (idx >= frames.length) return stop();
      show(idx);
    }, 350);
  };
  playBtn.addEventListener("click", play);

  show(0);
  play(); // auto-play the un-masking on render
}

// Animate the image reverse process: draw each per-step latent RGB preview
// (noise -> structure) on a canvas, then reveal the final crisp VAE-decoded PNG.
function renderImageRun(
  container: HTMLElement,
  frames: ImageFrame[],
  finalImage: string | null,
  info: { wallMs?: number; render?: { finite?: boolean }; package?: string; prompt?: string; perf?: Perf | null }
) {
  container.innerHTML = "";
  const perfCard = renderPerf(info.perf);
  if (perfCard) container.appendChild(perfCard);
  const figure = el("div", "image-result");
  container.appendChild(figure);

  const DISPLAY = 384;
  const canvas = document.createElement("canvas");
  canvas.width = DISPLAY;
  canvas.height = DISPLAY;
  canvas.className = "denoise-canvas";
  figure.appendChild(canvas);
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingEnabled = true;

  const finalImg = el("img") as HTMLImageElement;
  finalImg.className = "image-final";
  finalImg.alt = "rendered image";
  finalImg.style.display = "none";
  if (finalImage) finalImg.src = finalImage;
  figure.appendChild(finalImg);

  const off = document.createElement("canvas");
  const octx = off.getContext("2d")!;
  const drawFrame = (fr: ImageFrame) => {
    off.width = fr.w;
    off.height = fr.h;
    const bin = atob(fr.rgb);
    const img = octx.createImageData(fr.w, fr.h);
    const n = fr.w * fr.h;
    for (let p = 0, s = 0; p < n; p++) {
      img.data[p * 4] = bin.charCodeAt(s++);
      img.data[p * 4 + 1] = bin.charCodeAt(s++);
      img.data[p * 4 + 2] = bin.charCodeAt(s++);
      img.data[p * 4 + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    ctx.clearRect(0, 0, DISPLAY, DISPLAY);
    ctx.drawImage(off, 0, 0, DISPLAY, DISPLAY);
  };

  const showCanvas = () => {
    canvas.style.display = "";
    finalImg.style.display = "none";
  };
  const showFinal = () => {
    if (!finalImage) return;
    canvas.style.display = "none";
    finalImg.style.display = "";
  };

  const controls = el("div", "controls");
  const playBtn = el("button", undefined, "▶ Replay denoising") as HTMLButtonElement;
  const slider = el("input") as HTMLInputElement;
  slider.type = "range";
  slider.min = "0";
  slider.max = String(frames.length); // last position = final image
  slider.value = "0";
  const label = el("div", "note");
  controls.appendChild(playBtn);
  controls.appendChild(slider);
  controls.appendChild(label);
  container.appendChild(controls);

  const show = (idx: number) => {
    slider.value = String(idx);
    if (idx >= frames.length) {
      showFinal();
      label.textContent = `final image (VAE-decoded) · ${frames.length} denoising steps`;
      return;
    }
    showCanvas();
    drawFrame(frames[idx]);
    label.textContent = `step ${idx + 1}/${frames.length} · latent preview`;
  };

  slider.addEventListener("input", () => show(Number(slider.value)));

  let timer: number | undefined;
  const stop = () => {
    if (timer) clearInterval(timer);
    timer = undefined;
    playBtn.textContent = "▶ Replay denoising";
  };
  const play = () => {
    if (timer) return stop();
    playBtn.textContent = "⏸ Pause";
    let idx = 0;
    show(0);
    timer = window.setInterval(() => {
      idx++;
      if (idx > frames.length) return stop();
      show(idx);
    }, 120);
  };
  playBtn.addEventListener("click", play);

  const meta: string[] = [];
  if (typeof info.wallMs === "number") meta.push(`${(info.wallMs / 1000).toFixed(1)} s wall`);
  if (info.render && info.render.finite === false) meta.push("⚠ non-finite output");
  figure.appendChild(
    el("div", "note", `text-encode → denoise → VAE decode${meta.length ? " · " + meta.join(" · ") : ""}`)
  );
  if (info.prompt) figure.appendChild(el("div", "note", `prompt: “${info.prompt}”`));
  container.appendChild(el("div", "note", `package: ${info.package ?? "(set ONNX_GENAI_SD_PACKAGE)"}`));

  if (frames.length) play(); // auto-play the denoising on render
  else showFinal();
}

// ---- App shell ----
type TabKind = "language" | "image";
let currentTab: TabKind = "language";
let loadedMeta: Metadata | null = null;

const app = document.getElementById("app")!;

function render() {
  app.innerHTML = "";
  app.appendChild(el("h1", undefined, "onnx-genai · diffusion demo"));
  app.appendChild(
    el("div", "sub", "Load a pipeline config (ComfyUI or native inference_metadata), inspect it, and run the real runtime.")
  );

  const tabs = el("div", "tabs");
  (["language", "image"] as TabKind[]).forEach((t) => {
    const b = el("div", "tab" + (t === currentTab ? " active" : ""), t === "language" ? "Language diffusion" : "Image diffusion");
    b.addEventListener("click", () => {
      currentTab = t;
      loadedMeta = null;
      render();
    });
    tabs.appendChild(b);
  });
  app.appendChild(tabs);

  // ---- Config loader ----
  const loader = el("div", "panel");
  loader.appendChild(el("h2", undefined, "1 · Load config"));
  const ta = el("textarea") as HTMLTextAreaElement;
  ta.placeholder =
    currentTab === "language"
      ? "Paste a ComfyUI workflow JSON, or native inference_metadata YAML/JSON…"
      : "Paste a ComfyUI (Stable Diffusion) workflow JSON, or native inference_metadata YAML/JSON…";
  loader.appendChild(ta);

  const row = el("div", "row");
  const comfyBtn = el("button", undefined, "Load as ComfyUI");
  const nativeBtn = el("button", "secondary", "Load as native config");
  row.appendChild(comfyBtn);
  row.appendChild(nativeBtn);
  loader.appendChild(row);
  const loadErr = el("div", "err");
  loadErr.style.display = "none";
  loader.appendChild(loadErr);
  app.appendChild(loader);

  const vizMount = el("div");
  app.appendChild(vizMount);
  const runPanel = el("div", "panel");
  runPanel.style.display = "none";
  runPanel.appendChild(el("h2", undefined, "3 · Run"));

  // Image tab: prompt + generation controls that drive the real SD pipeline.
  let promptInput: HTMLTextAreaElement | null = null;
  let negativeInput: HTMLInputElement | null = null;
  let stepsInput: HTMLInputElement | null = null;
  let guidanceInput: HTMLInputElement | null = null;
  let seedInput: HTMLInputElement | null = null;
  if (currentTab === "image") {
    const promptField = el("div", "field");
    promptField.appendChild(el("label", "field-label", "Prompt"));
    promptInput = el("textarea") as HTMLTextAreaElement;
    promptInput.className = "prompt-input";
    promptInput.placeholder = "a photograph of an astronaut riding a horse";
    promptInput.value = "a photograph of an astronaut riding a horse";
    promptField.appendChild(promptInput);
    runPanel.appendChild(promptField);

    const negField = el("div", "field");
    negField.appendChild(el("label", "field-label", "Negative prompt"));
    negativeInput = el("input") as HTMLInputElement;
    negativeInput.type = "text";
    negativeInput.className = "prompt-input";
    negativeInput.placeholder = "(optional) blurry, low quality";
    negField.appendChild(negativeInput);
    runPanel.appendChild(negField);

    const numRow = el("div", "row controls-row");
    const mkNumber = (label: string, value: string, min: string, max: string, step: string) => {
      const field = el("div", "field field-inline");
      field.appendChild(el("label", "field-label", label));
      const input = el("input") as HTMLInputElement;
      input.type = "number";
      input.min = min;
      input.max = max;
      input.step = step;
      input.value = value;
      field.appendChild(input);
      numRow.appendChild(field);
      return input;
    };
    stepsInput = mkNumber("Steps", "25", "1", "100", "1");
    guidanceInput = mkNumber("Guidance", "7.5", "0", "30", "0.5");
    seedInput = mkNumber("Seed", "0", "0", "999999", "1");
    runPanel.appendChild(numRow);
  }

  const runRow = el("div", "row");
  const runBtn = el("button", undefined, currentTab === "language" ? "Run language diffusion" : "Run image diffusion") as HTMLButtonElement;
  runRow.appendChild(runBtn);
  runPanel.appendChild(runRow);
  const runOut = el("div");
  runPanel.appendChild(runOut);
  const runErr = el("div", "err");
  runErr.style.display = "none";
  runPanel.appendChild(runErr);
  app.appendChild(runPanel);

  // The image tab renders straight from the configured SD package, so expose
  // the run controls immediately without requiring a pasted config first.
  if (currentTab === "image") runPanel.style.display = "";

  const showViz = (meta: Metadata) => {
    loadedMeta = meta;
    vizMount.innerHTML = "";
    const p = el("div", "panel");
    p.appendChild(el("h2", undefined, "2 · Current pipeline config"));
    vizMount.appendChild(p);
    vizMount.appendChild(renderPipeline(meta));
    runPanel.style.display = "";
  };

  const setErr = (node: HTMLElement, msg: string) => {
    node.style.display = "";
    node.textContent = msg;
  };

  comfyBtn.addEventListener("click", async () => {
    loadErr.style.display = "none";
    try {
      const res = await postText("/api/translate-comfyui", ta.value);
      showViz(res.metadata as Metadata);
    } catch (e) {
      setErr(loadErr, String((e as Error).message));
    }
  });
  nativeBtn.addEventListener("click", async () => {
    loadErr.style.display = "none";
    try {
      const res = await postText("/api/parse-native", ta.value);
      showViz(res.metadata as Metadata);
    } catch (e) {
      setErr(loadErr, String((e as Error).message));
    }
  });

  // Language tab: offer the bundled fixture, which runs out of the box.
  if (currentTab === "language") {
    const fixtureRow = el("div", "row");
    const fixtureBtn = el("button", "secondary", "Use bundled fixture (runs immediately)");
    fixtureRow.appendChild(fixtureBtn);
    loader.appendChild(fixtureRow);
    fixtureBtn.addEventListener("click", async () => {
      loadErr.style.display = "none";
      runErr.style.display = "none";
      runBtn.disabled = true;
      runBtn.textContent = "Running…";
      try {
        const res = await postText("/api/run/language", "{}");
        if (res.metadata) showViz(res.metadata as Metadata);
        else {
          loadedMeta = null;
          runPanel.style.display = "";
        }
        renderLanguageRun(runOut, res.frames as Frame[], res.maskId, res.perf as Perf);
      } catch (e) {
        setErr(runErr, String((e as Error).message));
      } finally {
        runBtn.disabled = false;
        runBtn.textContent = "Run language diffusion";
      }
    });
  }

  runBtn.addEventListener("click", async () => {
    runErr.style.display = "none";
    runBtn.disabled = true;
    const original = runBtn.textContent;
    runBtn.textContent = "Running…";
    try {
      if (currentTab === "language") {
        const res = await postText("/api/run/language", "{}");
        if (res.metadata && !loadedMeta) showViz(res.metadata as Metadata);
        renderLanguageRun(runOut, res.frames as Frame[], res.maskId, res.perf as Perf);
      } else {
        const payload = {
          prompt: promptInput?.value ?? "",
          negative: negativeInput?.value ?? "",
          steps: stepsInput ? Number(stepsInput.value) : undefined,
          guidance: guidanceInput ? Number(guidanceInput.value) : undefined,
          seed: seedInput ? Number(seedInput.value) : undefined,
        };
        const res = await postText("/api/run/image", JSON.stringify(payload));
        runOut.innerHTML = "";
        if (res.metadata) showViz(res.metadata as Metadata);
        renderImageRun(runOut, (res.frames as ImageFrame[]) ?? [], (res.image as string) ?? null, {
          wallMs: res.wallMs,
          render: res.render,
          package: res.package,
          prompt: res.prompt,
          perf: res.perf as Perf,
        });
      }
    } catch (e) {
      setErr(runErr, String((e as Error).message));
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = original;
    }
  });
}

render();
