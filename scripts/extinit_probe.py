#!/usr/bin/env python
"""Feasibility probe for issue #955 — externally-supplied ORT initializers.

This is an EXPERIMENT, not a feature. It answers three questions about whether the
runtime could own initializer loading and hand ORT tensors from its own ledger,
under its own budget (see docs/memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md, section
"Why the ORT path cannot enforce today, stated as a feasibility question"):

  Q1. Does ORT copy again during prepacking? (peak working set, load only)
  Q2. Does a GPU EP always copy initializers to device regardless of arrival?
  Q3. Can our loader produce zero-copy OrtValues over an mmap of the external-data
      blob, with lifetimes that outlive the session?

Metric of record is PEAK WORKING SET (Windows peak_wset via psutil): a
process-local, contention-invariant counter (measurement-discipline SKILL, item 6).
Per-process GPU memory is read from nvidia-smi --query-compute-apps for THIS pid.

Each measurement runs in its own subprocess so peaks never bleed between modes.
Nothing here changes the engine load path; it only loads a model different ways
and reports what ORT then does with host/device memory.

Usage:
  python extinit_probe.py orchestrate <model_dir> [--provider cpu|cuda]
  python extinit_probe.py consume-proof <model_dir>
  python extinit_probe.py run <mode> <model_dir> --provider cpu|cuda   (internal)

For --provider cuda the CUDA-13 runtime DLLs must be resolvable; e.g. prepend the
pip nvidia wheel bin dirs to PATH (nvidia/cu13/bin/x86_64 and nvidia/cudnn/bin).
"""
import argparse
import ctypes
import gc
import json
import mmap
import os
import subprocess
import sys
import time

import numpy as np

ONNX_TO_NP = {
    1: np.float32,
    7: np.int64,
    10: np.float16,
    2: np.uint8,
}


def peak_wset_bytes():
    import psutil

    return psutil.Process().memory_info().peak_wset


def gpu_bytes_for_pid():
    pid = os.getpid()
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001 - probe: any nvidia-smi failure => unknown
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                return int(parts[1]) * 1024 * 1024
            except ValueError:
                return None  # WDDM reports [N/A] for per-process memory
    return 0


_CUDART = None


def _cudart():
    global _CUDART
    if _CUDART is not None:
        return _CUDART
    import glob
    import site

    candidates = []
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    env_dir = os.environ.get("ONNX_CUDA_DLL_DIR")
    if env_dir:
        roots.insert(0, env_dir)
    for root in roots:
        candidates += glob.glob(
            os.path.join(root, "**", "cudart64_1*.dll"), recursive=True
        )
    for path in candidates:
        try:
            os.add_dll_directory(os.path.dirname(path))
        except (OSError, AttributeError):
            pass
        try:
            lib = ctypes.CDLL(path)
            lib.cudaMemGetInfo.argtypes = [
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            _CUDART = lib
            return _CUDART
        except OSError:
            continue
    _CUDART = False
    return _CUDART


def cuda_free_bytes():
    """Device-wide free memory via cudaMemGetInfo. On an otherwise-idle GPU the
    drop in free bytes across session creation equals what the CUDA EP placed on
    the device. Returns None if cudart is unavailable."""
    lib = _cudart()
    if not lib:
        return None
    free = ctypes.c_size_t(0)
    total = ctypes.c_size_t(0)
    rc = lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    if rc != 0:
        return None
    return free.value


def load_external_specs(model_dir):
    """Return (data_path, [(name, np_dtype, dims, offset, length)])."""
    import onnx
    from onnx import TensorProto

    model_path = os.path.join(model_dir, "model.onnx")
    m = onnx.load(model_path, load_external_data=False)
    specs = []
    data_file = None
    for t in m.graph.initializer:
        if t.data_location != TensorProto.EXTERNAL:
            continue
        d = {e.key: e.value for e in t.external_data}
        offset = int(d.get("offset", 0))
        length = int(d.get("length", 0))
        data_file = d.get("location")
        np_dtype = ONNX_TO_NP.get(t.data_type)
        if np_dtype is None:
            raise RuntimeError(f"unmapped dtype {t.data_type} for {t.name}")
        specs.append((t.name, np_dtype, list(t.dims), offset, length))
    return os.path.join(model_dir, data_file), specs


def build_ortvalues_zero_copy(data_path, specs):
    """mmap the external-data blob and produce OrtValues that BORROW it.

    Proves the Q3 mechanism: each numpy array is a view over the mmap (no copy),
    and ortvalue_from_numpy shares that pointer (verified via data_ptr). The mmap
    and the numpy views must be kept alive for the session's lifetime.
    """
    import onnxruntime as ort

    f = open(data_path, "rb")  # noqa: SIM115 - must outlive the session
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    names, ortvalues, keepalive = [], [], [mm, f]
    shared = 0
    for name, np_dtype, dims, offset, length in specs:
        count = length // np.dtype(np_dtype).itemsize
        arr = np.frombuffer(mm, dtype=np_dtype, count=count, offset=offset)
        if dims:
            arr = arr.reshape(dims)
        ov = ort.OrtValue.ortvalue_from_numpy(arr)
        # Q3 evidence: the numpy view is a no-copy window into the read-only mmap
        # (np.frombuffer over mmap does not copy), and the OrtValue shares that
        # exact pointer. So ORT is handed a tensor that borrows our mmap'd blob.
        if ov.data_ptr() == arr.ctypes.data:
            shared += 1
        names.append(name)
        ortvalues.append(ov)
        keepalive.append(arr)
    return names, ortvalues, keepalive, shared, len(specs)


def make_session_options(provider, disable_opt, disable_prepack=False):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if disable_opt:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    else:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if disable_prepack:
        # graph_optimization_level does NOT control kernel prepacking; this config
        # entry does. Toggling it is the only way to isolate the prepack copy (Q1).
        so.add_session_config_entry("session.disable_prepacking", "1")
    return so


def providers_list(provider):
    if provider == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def run_mode(mode, model_dir, provider):
    import onnxruntime as ort

    model_path = os.path.join(model_dir, "model.onnx")
    result = {"mode": mode, "provider": provider}
    disable_opt = "-noopt" in mode
    disable_prepack = "-noprepack" in mode
    base_mode = mode.replace("-noopt", "").replace("-noprepack", "")

    keepalive = None
    shared = total = 0
    free_before = cuda_free_bytes() if provider == "cuda" else None
    t0 = time.time()

    if base_mode == "baseline":
        # Import + external-data enumeration only, no session.
        data_path, specs = load_external_specs(model_dir)
        result["num_external"] = len(specs)
        sess = None
    elif base_mode == "today":
        so = make_session_options(provider, disable_opt, disable_prepack)
        sess = ort.InferenceSession(
            model_path, sess_options=so, providers=providers_list(provider)
        )
    elif base_mode in ("add-external", "add-initializer"):
        data_path, specs = load_external_specs(model_dir)
        names, ortvalues, keepalive, shared, total = build_ortvalues_zero_copy(
            data_path, specs
        )
        so = make_session_options(provider, disable_opt, disable_prepack)
        if base_mode == "add-external":
            so.add_external_initializers(names, ortvalues)
        else:
            for n, ov in zip(names, ortvalues):
                so.add_initializer(n, ov)
        sess = ort.InferenceSession(
            model_path, sess_options=so, providers=providers_list(provider)
        )
    else:
        raise SystemExit(f"unknown mode {mode}")

    load_s = time.time() - t0
    gc.collect()

    result["shared_ortvalues"] = shared
    result["total_ortvalues"] = total
    result["load_seconds"] = round(load_s, 3)
    result["peak_wset_MiB"] = round(peak_wset_bytes() / 1048576, 1)
    gpu = gpu_bytes_for_pid()
    result["gpu_pid_MiB"] = None if gpu is None else round(gpu / 1048576, 1)
    if provider == "cuda":
        free_after = cuda_free_bytes()
        if free_before is not None and free_after is not None:
            result["gpu_alloc_MiB"] = round((free_before - free_after) / 1048576, 1)
        else:
            result["gpu_alloc_MiB"] = None
    if sess is not None:
        result["providers"] = sess.get_providers()

    # Q3 liveness proof: run one inference so ORT actually reads the borrowed
    # host buffers, then confirm our mmap is still ours and readable afterward.
    if sess is not None and base_mode in ("add-external", "add-initializer"):
        try:
            result["ran_inference"] = _try_run(sess)
        except Exception as e:  # noqa: BLE001
            result["ran_inference"] = f"error: {type(e).__name__}: {e}"
        if keepalive is not None:
            mm = keepalive[0]
            # Touch a byte: if ORT had freed our buffer this would be UB/crash.
            result["mmap_still_readable"] = bool(mm[0:1])

    # Keep everything alive until after we've measured/serialized.
    del keepalive, sess
    print(json.dumps(result))


def _build_feeds(sess):
    feeds = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        name = i.name
        t = i.type
        if "int64" in t:
            dtype = np.int64
        elif "int32" in t:
            dtype = np.int32
        elif "float16" in t:
            dtype = np.float16
        else:
            dtype = np.float32
        if "attention_mask" in name:
            feeds[name] = np.ones(shape, dtype=dtype)
        else:
            feeds[name] = np.zeros(shape, dtype=dtype)
    return feeds


def _try_run(sess):
    """Minimal single-token forward pass with zeros for whatever inputs exist."""
    out = sess.run(None, _build_feeds(sess))
    return f"ok, {len(out)} outputs, first shape {list(np.asarray(out[0]).shape)}"


def _argmax_one(model_dir, zero_name):
    """Build ONE session (all external inits via add_initializer over mmap, with
    an optional single weight replaced by a private zeroed buffer) and return the
    argmax of the first-row logits. Runs in its own process for isolation."""
    import onnxruntime as ort

    model_path = os.path.join(model_dir, "model.onnx")
    data_path, specs = load_external_specs(model_dir)
    f = open(data_path, "rb")  # noqa: SIM115 - must outlive the session
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    keep = [mm, f]
    so = ort.SessionOptions()
    for name, np_dtype, dims, offset, length in specs:
        count = length // np.dtype(np_dtype).itemsize
        arr = np.frombuffer(mm, dtype=np_dtype, count=count, offset=offset)
        if dims:
            arr = arr.reshape(dims)
        if name == zero_name:
            arr = np.zeros_like(arr)  # private buffer, not the mmap
        keep.append(arr)
        so.add_initializer(name, ort.OrtValue.ortvalue_from_numpy(arr))
    sess = ort.InferenceSession(
        model_path, sess_options=so, providers=["CPUExecutionProvider"]
    )
    out = np.asarray(sess.run(None, _build_feeds(sess))[0])
    print(json.dumps({"argmax": int(out.reshape(-1, out.shape[-1])[0].argmax())}))


def consume_proof(model_dir, substitute="model.embed_tokens.qweight"):
    """Q3 airtight proof that ORT reads the bytes WE supply (not the file).

    Supply all external initializers via add_initializer over an mmap, then
    re-run with exactly one large weight replaced by a private zeroed buffer.
    If ORT consumed our buffer, the output argmax must change. Each session runs
    in its own process (two sessions with mmap'd initializers in one process can
    trip an access violation on Windows).
    """

    def run(zero_name):
        cmd = [sys.executable, os.path.abspath(__file__), "argmax", model_dir]
        if zero_name:
            cmd += ["--zero", zero_name]
        out = subprocess.check_output(cmd, encoding="utf-8")
        line = [l for l in out.splitlines() if l.startswith("{")][-1]
        return json.loads(line)["argmax"]

    faithful = run(None)
    zeroed = run(substitute)
    print(
        json.dumps(
            {
                "substituted": substitute,
                "argmax_faithful": faithful,
                "argmax_with_our_zeroed_buffer": zeroed,
                "ort_consumed_our_buffer": faithful != zeroed,
            }
        )
    )


def orchestrate(model_dir, provider):
    modes = [
        "baseline",
        "today",
        "today-noprepack",
        "add-external",
        "add-external-noprepack",
        "add-initializer",
        "add-initializer-noprepack",
    ]
    rows = []
    for mode in modes:
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "run",
            mode,
            model_dir,
            "--provider",
            provider,
        ]
        try:
            out = subprocess.check_output(cmd, encoding="utf-8")
            line = [l for l in out.splitlines() if l.startswith("{")][-1]
            rows.append(json.loads(line))
        except subprocess.CalledProcessError as e:
            rows.append({"mode": mode, "error": e.output[-500:] if e.output else str(e)})
    print(json.dumps({"provider": provider, "model_dir": model_dir, "rows": rows}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("orchestrate")
    o.add_argument("model_dir")
    o.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    r = sub.add_parser("run")
    r.add_argument("mode")
    r.add_argument("model_dir")
    r.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    c = sub.add_parser("consume-proof")
    c.add_argument("model_dir")
    a = sub.add_parser("argmax")
    a.add_argument("model_dir")
    a.add_argument("--zero", default=None)
    args = ap.parse_args()
    if args.cmd == "orchestrate":
        orchestrate(args.model_dir, args.provider)
    elif args.cmd == "consume-proof":
        consume_proof(args.model_dir)
    elif args.cmd == "argmax":
        _argmax_one(args.model_dir, args.zero)
    else:
        run_mode(args.mode, args.model_dir, args.provider)


if __name__ == "__main__":
    main()
