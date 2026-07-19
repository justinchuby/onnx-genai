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

// ---- Language un-masking animation ----
function renderLanguageRun(container: HTMLElement, frames: Frame[], maskId: number) {
  container.innerHTML = "";
  if (!frames.length) {
    container.appendChild(el("div", "err", "No frames returned."));
    return;
  }
  const last = frames[frames.length - 1];
  const seqLen = last.shape[last.shape.length - 1];

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
    for (let i = 0; i < seqLen; i++) {
      const v = data[i];
      const masked = v === maskId;
      const justFilled = !masked && prev[i] === maskId;
      cells[i].textContent = masked ? "▒" : String(v);
      cells[i].className = "tok" + (masked ? "" : " filled") + (justFilled ? " just" : "");
    }
    prev = data;
    const remaining = data.filter((v) => v === maskId).length;
    label.textContent = `step ${f.step + 1}/${frames.length} · ${seqLen - remaining}/${seqLen} unmasked`;
    slider.value = String(idx);
  };

  slider.addEventListener("input", () => {
    prev = []; // recompute highlight fresh when scrubbing
    show(Number(slider.value));
  });

  let timer: number | undefined;
  playBtn.addEventListener("click", () => {
    if (timer) {
      clearInterval(timer);
      timer = undefined;
      playBtn.textContent = "▶ Play";
      return;
    }
    playBtn.textContent = "⏸ Pause";
    let idx = 0;
    prev = [];
    show(0);
    timer = window.setInterval(() => {
      idx++;
      if (idx >= frames.length) {
        clearInterval(timer);
        timer = undefined;
        playBtn.textContent = "▶ Play";
        return;
      }
      show(idx);
    }, 700);
  });

  show(0);
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
        renderLanguageRun(runOut, res.frames as Frame[], res.maskId);
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
        renderLanguageRun(runOut, res.frames as Frame[], res.maskId);
      } else {
        const res = await postText("/api/run/image", "{}");
        runOut.innerHTML = "";
        if (res.metadata) showViz(res.metadata as Metadata);
        runOut.appendChild(el("div", "note", res.note ?? "Image run configured."));
        runOut.appendChild(
          el("div", "note", `package: ${res.package ?? "(set ONNX_GENAI_SD_PACKAGE)"}`)
        );
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
