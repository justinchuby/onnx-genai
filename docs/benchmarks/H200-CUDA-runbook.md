# H200 CUDA benchmark runbook

A complete, copy-pasteable procedure to build, run, and benchmark the onnx-genai
CUDA path on an **NVIDIA H200** (Linux, CUDA 12), against the same-weights
llama.cpp runtimes (LM Studio, Ollama, Foundry Local).

> **Status when written (2026-07-12, from a Mac M1 Max):** every step below is
> assembled from validated component work but **the end-to-end CUDA path has not
> yet run on real Hopper hardware.** The runtime CUDA EP, CUDA-graph, and
> device-resident KV plumbing are implemented and compile behind the `cuda`
> feature; the H200 package (`models/qwen2.5-0.5b-cuda/`) is structurally valid
> (`onnx.checker` + strict shape inference pass) but has only been exercised via
> CPU fallback ("The capital of France is Paris."). Treat this runbook as the
> validation procedure, and run the **coherence gate in §6 first**.
>
> Provenance: Leon's *Feature-gated CUDA EP, CUDA graphs, and device-resident
> GQA KV* decision (env/flags), Sapper's *CUDA-targeted stacked Qwen model*
> decision (build), and the cross-runtime methodology in
> `docs/benchmarks/README.md`.

---

## 0. Machine prerequisites

| Requirement | Value / note |
|---|---|
| GPU | NVIDIA H200 (Hopper, `sm_90`) |
| OS | Linux x86_64 (Ubuntu 22.04 LTS validated toolchain) |
| CUDA | **CUDA 12.x** toolkit + matching driver (`nvidia-smi` reports CUDA ≥ 12.4) |
| cuDNN | **cuDNN 9.x** (ORT 1.27 CUDA EP links cuDNN 9) |
| ONNX Runtime | **onnxruntime-gpu 1.27.0** built/installed **with the CUDA EP** |
| Rust | stable toolchain matching `rustc` in the CI reports (≥ 1.97) |
| Python | conda `onnx` env (Mobius + ONNX tooling) |
| Node | `lms` (LM Studio CLI), `ollama`, `foundry-local-sdk` for comparison runtimes |

Verify the GPU and CUDA/cuDNN stack **before** anything else:

```bash
nvidia-smi                      # H200 visible; driver CUDA >= 12.4
nvcc --version                  # CUDA toolkit 12.x
ls /usr/lib/x86_64-linux-gnu/libcudnn.so.9*   # cuDNN 9 present
```

The onnxruntime-gpu build **must** include the CUDA EP, and its
`libonnxruntime.so` plus the CUDA 12 / cuDNN 9 shared objects must be on the
linker/loader path. Point `ORT_ROOT` at that install and make its `lib/`
discoverable:

```bash
export ORT_ROOT=/opt/onnxruntime-gpu-1.27.0
export LD_LIBRARY_PATH="$ORT_ROOT/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

> If ORT cannot find the CUDA EP at session-create time it silently falls back
> to CPU. The server logs the **effective** execution providers on startup —
> confirm `Cuda` is listed and not just `Cpu`.

---

## 1. Get the source GGUF (same-weights anchor)

The whole comparison is only fair because every runtime consumes the **same**
GGUF bytes. Anchor by SHA-256 and do not re-download if present.

```bash
cd /path/to/onnx-genai
mkdir -p models/gguf
# Preferred: Q4_K_M (matches the CUDA stacked build target). Fall back to Q4_0
# if no Q4_K_M is available locally — record which one you used.
curl -L --fail -o models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
shasum -a 256 models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf

# Q4_0 fallback (verified SHA on the Mac runs):
# 7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed
curl -L --fail -o models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf
shasum -a 256 models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf
```

---

## 2. Build the CUDA ONNX model in Mobius

The H200 target is the **stacked** package: Q4/Q4_K_M `MatMulNBits` projections +
on-device `GroupQueryAttention` + a `GatherBlockQuantized` quantized embedding.

### 2a. Branches / PRs required

The stacked build needs these Mobius pieces merged (or use the integration
branch that already combines them):

| Piece | PR / branch | Purpose |
|---|---|---|
| GGUF→ONNX + GQA base | **#396 (merged)** | `build-gguf` foundation, GQA rewrite |
| Q4_K_M normalization | **#399** | normalize mixed Q4_K_M projections → `MatMulNBits` int4 |
| Quantized embedding | **#400** | `--keep-quantized` → `GatherBlockQuantized` token embedding |
| Everything combined | branch **`int/cuda-stacked`** @ `380acf2` | Q4_K_M + quantized embed + `GatherBlockQuantized` shape/type stamp so exports pass `onnx.checker` |

Fastest path: check out `int/cuda-stacked`, which already integrates #399 and
#400 on top of #396.

```bash
cd /path/to/mobius
git fetch origin
git checkout int/cuda-stacked     # 380acf2 or newer
conda run -n onnx python -m pip install \
  'onnx-shape-inference==0.2.0' 'onnxscript==0.7.1' 'gguf==0.19.0'
```

### 2b. Build the package

```bash
cd /path/to/mobius
PYTHONPATH=src conda run -n onnx python -m mobius build-gguf \
  /path/to/onnx-genai/models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --ep cuda \
  --runtime onnx-genai \
  --keep-quantized \
  --output /path/to/onnx-genai/models/qwen2.5-0.5b-cuda
```

Expected structure of `models/qwen2.5-0.5b-cuda/` (Sapper's validated package):

- 24 `GroupQueryAttention`, 168 `MatMulNBits`, 1 `GatherBlockQuantized`,
  0 `Attention`, fp16 KV I/O, ~73 MiB packed embedding payload.
- `onnx.checker` and strict shape inference pass.
- `inference_metadata.yaml` declares `grouped_query_attention`, fp16 KV,
  `max_sequence_length: 4096`.

> **Caveat (known gap):** `build-gguf` did not yet accept `--runtime onnx-genai`
> in the state Sapper validated. If your Mobius rejects that flag, drop it and
> emit the `inference_metadata.yaml` sidecar separately with the
> `feat/onnx-genai-metadata-export` emitter (`mobius build ... --runtime
> onnx-genai`), then copy it into the package. Confirm the flag support on the
> branch you checked out before relying on it.
> **Note on `--ep cuda`:** for this Qwen graph `--ep cuda` and `--ep webgpu`
> produce byte-identical ONNX + external-data (both allow fp16 GQA + packed
> QKV; Qwen triggers no WebGPU-only rewrites). The `cuda` label is intent, not a
> different graph.

Copy the tokenizer/vocab if not already emitted into the package:

```bash
cd /path/to/onnx-genai
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  [ -f models/qwen2.5-0.5b-cuda/$f ] || cp models/qwen2.5-0.5b/$f models/qwen2.5-0.5b-cuda/
done
```

### 2c. Verify node counts

```bash
cd /path/to/onnx-genai
conda run -n onnx python - <<'PY'
import onnx, collections
m = onnx.load('models/qwen2.5-0.5b-cuda/model.onnx', load_external_data=False)
c = collections.Counter((n.domain, n.op_type) for n in m.graph.node)
for k in [('com.microsoft','GroupQueryAttention'),('com.microsoft','MatMulNBits'),
          ('com.microsoft','GatherBlockQuantized'),('','Attention')]:
    print(k, c.get(k, 0))
PY
# Expect: GQA 24, MatMulNBits 168, GatherBlockQuantized 1, Attention 0
```

---

## 3. Build the onnx-genai server with CUDA

The `cuda` Cargo feature lives on `onnx-genai-ort` and is **default-off** so Mac
CPU/WebGPU builds are unaffected. ORT must be the CUDA-enabled build from §0.

```bash
cd /path/to/onnx-genai
export ORT_ROOT=/opt/onnxruntime-gpu-1.27.0
export LD_LIBRARY_PATH="$ORT_ROOT/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

Runtime env (Leon's flags):

```bash
export ONNX_GENAI_EP=cuda          # select ExecutionProvider::Cuda
export ONNX_GENAI_CUDA_DEVICE=0    # CUDA device id (default 0)
export ONNX_GENAI_CUDA_GRAPH=1     # enable_cuda_graph=1 (CUDA graph capture)
export ONNX_GENAI_DEVICE_KV=1      # device-resident fp16 GQA KV (no host<->device copies)
```

Start the server. Until the coordinator adds a server-level `cuda` forwarding
alias, use the **package-qualified** feature:

```bash
cargo run --release -p onnx-genai-server \
  --features onnx-genai-ort/cuda -- \
  --model models/qwen2.5-0.5b-cuda \
  --model-id qwen2.5-0.5b-cuda \
  --addr 127.0.0.1:8080
```

After the forwarding alias (`cuda = ["onnx-genai-ort/cuda"]`) is added to
`onnx-genai-server`, the shorthand is:

```bash
cargo run --release --features cuda -p onnx-genai-server -- \
  --model models/qwen2.5-0.5b-cuda \
  --model-id qwen2.5-0.5b-cuda \
  --addr 127.0.0.1:8080
```

On startup, confirm the log shows **effective execution providers = [Cuda, Cpu]**
(Cpu is the safety fallback). If it shows Cpu only:

- The ORT build has no CUDA EP, or `LD_LIBRARY_PATH` is missing CUDA/cuDNN, or
- the binary was built without `--features onnx-genai-ort/cuda` (then
  `ONNX_GENAI_EP=cuda` returns *"CUDA support not compiled in; rebuild with
  --features cuda"*).

---

## 4. Start the comparison runtimes (CUDA)

All three consume the **same GGUF** on the GPU via llama.cpp CUDA. Record each
runtime version and offload/context/parallel settings in the report.

### 4a. LM Studio (CUDA)

```bash
LM_DIR="$HOME/.cache/lm-studio/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF"
mkdir -p "$LM_DIR"
ln models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  "$LM_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf"       # hard-link same inode
lms server start -p 1234
lms load 'qwen2.5-0.5b-instruct@q4_k_m' \
  --gpu max --context-length 2048 --parallel 1 \
  --no-speculative-draft-mtp \
  --identifier qwen05-cuda-bench -y
```
Endpoint: `http://127.0.0.1:1234/v1`. On CUDA, `--gpu max` offloads all layers
to the H200.

### 4b. Ollama (CUDA / llama.cpp)

```bash
printf 'FROM %s\n' "$PWD/models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  > models/benchmarks/Modelfile.cuda
ollama create qwen05-cuda -f models/benchmarks/Modelfile.cuda
# Ollama auto-detects CUDA; force all layers on GPU if needed:
#   OLLAMA_NUM_GPU=999 ollama serve
```
Endpoint: `http://127.0.0.1:11434/v1`.

### 4c. Foundry Local (CUDA-GPU variant)

```bash
pip install foundry-local-sdk
foundry service start
# List catalog and pick the qwen2.5-0.5b cuda-gpu variant:
foundry model list | grep -i qwen2.5-0.5b
foundry model run qwen2.5-0.5b-instruct-cuda-gpu   # or the exact catalog id
```
Foundry Local exposes an OpenAI-compatible endpoint (default
`http://127.0.0.1:5273/v1`; confirm with `foundry service status`). Use its
`cuda-gpu` model variant so the comparison runs on the H200, not CPU.

---

## 5. Run the benchmark

`scripts/compare_runtimes.sh` reads `ONNX_RUNTIME` / `LM_STUDIO_RUNTIME` env
vars; the `compare` bin also takes repeated `--runtime NAME|URL|MODEL|FORMAT|SETTINGS`.
Use the tokenizer from the CUDA package so prompt/generated token counts match.

```bash
cd /path/to/onnx-genai
export LM_STUDIO_MODEL_ID=qwen05-cuda-bench
cargo run --release -p onnx-genai-bench --bin compare -- \
  --runs 5 --warmups 1 --max-tokens 128 \
  --tokenizer models/qwen2.5-0.5b-cuda/tokenizer.json \
  --output docs/benchmarks/h200-cuda-vs-lm-studio.md \
  --runtime "onnx-genai CUDA|http://127.0.0.1:8080/v1|qwen2.5-0.5b-cuda|ONNX Q4_K_M; fp16 GQA KV|H200 CUDA EP; enable_cuda_graph=1; device KV=1" \
  --runtime "LM Studio|http://127.0.0.1:1234/v1|${LM_STUDIO_MODEL_ID}|GGUF Q4_K_M|H200 CUDA (llama.cpp); GPU=max; context=2048; parallel=1; speculation=off" \
  --runtime "Ollama|http://127.0.0.1:11434/v1|qwen05-cuda:latest|GGUF Q4_K_M|H200 CUDA (llama.cpp); all layers on GPU" \
  --runtime "Foundry Local|http://127.0.0.1:5273/v1|qwen2.5-0.5b-instruct-cuda-gpu|GGUF Q4_K_M|H200 CUDA-GPU variant"
```

### Metrics produced

| Metric | Meaning | How to read |
|---|---|---|
| **TTFT ms** ↓ | request start → first streamed content | includes HTTP + prefill + first decode |
| **decode tok/s** ↑ | `(generated-1) / (total - TTFT)` | steady-state per-token throughput — the headline number |
| **total ms** ↓ | request → stream close after `[DONE]` | end-to-end latency |
| **estimated prefill tok/s** ↑ | `prompt_tokens / TTFT` | API-level prefill estimate (not kernel-only) |

Cells show **median / p90** over 5 runs after 1 warmup. Greedy decode
(`temperature=0`, `top_p=1`, `seed=0`). Run short (~59 tok) and long (~858 tok)
prompts to separate decode from prefill behavior.

### Fairness notes

- **Same GGUF weights**: every runtime loads the identical SHA-256 file; the
  ONNX package is converted from those same bytes. Record the SHA in the report.
- Remaining format asymmetry: Mobius may still expand some tensors (e.g. an
  fp32 output head) — call it out explicitly, it penalizes ONNX memory traffic.
- Same device (H200), same context (2048), `parallel=1`, speculation off on all
  llama.cpp runtimes so it is decode-vs-decode, not a speculative-decoding win.
- Single-request latency, not concurrent-serving throughput. Quiet machine,
  stable power, warm model residency.

---

## 6. Pre-benchmark coherence gate (DO THIS FIRST)

**Known Hopper/ORT CUDA caveat (from prior memory): some ORT CUDA builds emit
garbled tokens on Hopper.** A fast decode number on incoherent output is
worthless. Before benchmarking, verify the CUDA path produces coherent text:

```bash
# With the CUDA server from §3 running on :8080
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-0.5b-cuda",
       "messages":[{"role":"user","content":"What is the capital of France?"}],
       "temperature":0,"max_tokens":16}' | python3 -m json.tool
```

Expected: a coherent completion containing **"Paris"** (the CPU-fallback
reference answer is *"The capital of France is Paris."*).

If output is garbled on the H200 but coherent on CPU fallback:

1. Re-run with `ONNX_GENAI_CUDA_GRAPH=0` and `ONNX_GENAI_DEVICE_KV=0` to isolate
   whether CUDA-graph capture or device-resident KV is the culprit (device-KV
   in-place share-buffer tensors are the known-fragile path — they SIGSEGV on
   ORT 1.27 WebGPU; validate the CUDA analog carefully).
2. Re-run with only `ONNX_GENAI_EP=cuda` (no graph, no device KV) to confirm the
   base CUDA EP kernels (`GroupQueryAttention`, `MatMulNBits`,
   `GatherBlockQuantized`) are numerically correct on `sm_90`.
3. Only after coherent output is confirmed, re-enable graph + device KV and
   re-verify coherence, **then** benchmark.

Do not publish decode numbers until the coherence gate passes with the exact
flag set used for the measured run.

---

## 7. Checklist

- [ ] `nvidia-smi` shows H200, driver CUDA ≥ 12.4.
- [ ] cuDNN 9 present; `ORT_ROOT` points at onnxruntime-gpu 1.27 **with CUDA EP**.
- [ ] `LD_LIBRARY_PATH` includes ORT lib + CUDA + cuDNN.
- [ ] GGUF downloaded, SHA-256 recorded, hard-linked into LM Studio.
- [ ] Mobius `int/cuda-stacked` (or #396+#399+#400) checked out; deps installed.
- [ ] `models/qwen2.5-0.5b-cuda/` built; node counts verified (24 GQA / 168 MatMulNBits / 1 GatherBlockQuantized / 0 Attention).
- [ ] `inference_metadata.yaml` present (GQA, fp16 KV, max_len 4096).
- [ ] Server built `--features onnx-genai-ort/cuda`; startup log shows effective EP = `[Cuda, Cpu]`.
- [ ] **Coherence gate (§6) passes** with the exact flag set to be benchmarked.
- [ ] LM Studio (CUDA), Ollama (CUDA), Foundry Local (cuda-gpu) all up and reachable.
- [ ] Benchmark run; report written; SHA + versions + settings recorded.
- [ ] LM Studio config restored, temporary hard-links removed (see §9).

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Startup log shows EP = `[Cpu]` only | ORT build lacks CUDA EP, or CUDA/cuDNN not on loader path, or built without `--features` | Fix `ORT_ROOT`/`LD_LIBRARY_PATH`; rebuild with `--features onnx-genai-ort/cuda` |
| `"CUDA support not compiled in; rebuild with --features cuda"` | binary built without the cuda feature | rebuild with `--features onnx-genai-ort/cuda` |
| **Garbled tokens on H200, coherent on CPU** | Hopper/ORT CUDA kernel or fragile device-KV/graph path | §6 isolation: disable `ONNX_GENAI_CUDA_GRAPH` then `ONNX_GENAI_DEVICE_KV`; confirm base EP first |
| SIGSEGV during longer generations with `ONNX_GENAI_DEVICE_KV=1` | in-place share-buffer device tensor bug (seen on ORT 1.27 WebGPU; validate CUDA analog) | set `ONNX_GENAI_DEVICE_KV=0` (CPU-allocated KV) and report; device-KV is experimental/opt-in |
| CUDA graph gives no speedup or errors | growing `attention_mask` each step breaks stable-address replay | benchmark with `ONNX_GENAI_CUDA_GRAPH=0`; graph capture needs fixed-capacity padded mask/device-resident I/O |
| `onnx.checker` fails on the model | missing GatherBlockQuantized shape/type stamp | use `int/cuda-stacked` @ 380acf2+; it stamps the op |
| `build-gguf: unrecognized --runtime onnx-genai` | flag not yet on your branch | emit `inference_metadata.yaml` separately via `feat/onnx-genai-metadata-export`, copy into package |
| Mobius weight-shape validation error on Q4_K_M | older Mobius normalizer | use #399 (Q4_K normalization → MatMulNBits) or fall back to Q4_0 and note it |
| Foundry Local runs on CPU | picked a non-cuda variant | select the explicit `cuda-gpu` model id |
| onnx-genai TTFT very high vs llama.cpp | no fused prefill path / fp32 output head | expected; note prefill is the known deficit, profile separately |

---

## 9. Teardown

```bash
lms unload --all
lms server stop
rm -f "$HOME/.cache/lm-studio/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf"
ollama rm qwen05-cuda 2>/dev/null || true
foundry service stop 2>/dev/null || true
# Restore any backed-up LM Studio config as documented in docs/benchmarks/README.md.
```

---

## 10. Reference: source decisions

- **CUDA EP / flags** — Leon, *Feature-gated CUDA EP, CUDA graphs, and
  device-resident GQA KV* (`.squad/decisions.md`): `ONNX_GENAI_EP=cuda`,
  `ONNX_GENAI_CUDA_DEVICE`, `ONNX_GENAI_CUDA_GRAPH=1`, `ONNX_GENAI_DEVICE_KV=1`;
  V2 CUDA provider API; `cuda` feature on `onnx-genai-ort`.
- **Device-KV SIGSEGV caveat** — Leon, *Device-resident GQA KV + persistent
  IoBinding* — in-place share-buffer device tensors crash on ORT 1.27 WebGPU;
  device KV is opt-in/experimental. Validate the CUDA analog before trusting it.
- **CUDA model build** — Sapper, *CUDA-targeted stacked Qwen model is
  structurally valid* + *Normalize mixed Q4_K_M → MatMulNBits* + *Preserve GGUF
  token embeddings with GatherBlockQuantized*: `int/cuda-stacked` @ 380acf2,
  package structure, `onnx.checker` pass, CPU-fallback coherence.
- **Cross-runtime methodology** — `docs/benchmarks/README.md` and the dated
  same-source reports.
