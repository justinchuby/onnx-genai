#!/usr/bin/env python3
"""Hardware evidence harness for the plugin-EP governed workspace path (#768).

`scripts/validate_plugin_ep_ort.py` proves the plugin loads and executes an
`Add`->`Mul` graph. Neither of those nodes asks for a workspace, so it proves
nothing about the path PR #830 adds. This script drives a node that *is* served
a `StepScoped` workspace and reports, per run:

1. that the node was **assigned to the plugin EP** (compiled-node counter from
   the loaded cdylib, not just the EP name in `session.get_providers()`);
2. that the **governed workspace path actually ran** (placement-resolution
   counter, which only moves for a served workspace);
3. **numerics** against an independent NumPy reference *and* against ORT's own
   CPU execution of the same graph, with the tolerance printed;
4. **steady-state** behaviour over many runs, so an `nsys` capture of this
   process can be checked for per-step `cuMemAlloc`/`cuMemFree`;
5. the **block address and alignment ORT's scratch actually returned**
   (`NXRT_EP_WORKSPACE_TRACE=1`), which is the only direct evidence of whether
   that scratch is arena-backed and reused;
6. **two-session teardown** on one registered library, checking stderr for the
   `ReleaseEpFactory` blocker diagnostic and device memory for a leak.

Exit codes follow `scripts/cuda_conformance_runner.sh`:

    0  VALIDATED     every requested check passed on this host
    1  FAILED        a check failed - this is a real finding
    2  UNVALIDATED   preconditions absent (no GPU, no ORT, no library); proves
                     nothing and must never be read as success

Two model geometries:

* ``--model attention`` (default): one default-domain `Attention` (opset 23) at
  a **prefill / batched** geometry, which is what makes it `StepScoped`. Decode
  geometry (`batch == 1 and q_seq == 1`) declares `SessionPersistent`, which the
  executor declines, so it is refused here rather than silently measured.
* ``--model addmul``: `Add` -> `Mul`, for the CPU shared-mock plugin whose `Add`
  requests a workspace and whose `Mul` does not. This is how the harness is
  self-tested on a host with no GPU (``--self-test``).

Typical use on the H200::

    cargo build --release -p onnx-runtime-ep-cuda-plugin --features cuda
    python scripts/validate_ep_workspace_h200.py \
        --lib target/release/libonnx_runtime_ep_cuda_plugin.so --nsys

Self-test on any host (no GPU required)::

    cargo build -p onnx-runtime-ep-shared-mock-plugin
    python scripts/validate_ep_workspace_h200.py --self-test

Every check here is meant to be capable of failing, and each one was driven red
before being trusted:

* ``NXRT_HARNESS_FORCE_PERSISTENT=1`` makes the mock declare
  `SessionPersistent`, the executor declines, and the serve checks go red.
* ``--atol 0 --rtol 0`` turns the numerics comparison red.
* ``--model attention --batch 1 --q-seq 1`` is refused as UNVALIDATED rather
  than measured.
* ``NXRT_HARNESS_RETAIN_SESSION=1`` holds a session across unregister; the
  teardown diagnostic must fire. The clean run runs this as a control, because
  "no diagnostic" is only evidence when a diagnostic could have appeared.
* the `nsys` path requires the capture to parse and to contain a kernel-launch
  API that grows with step count; an unreadable or empty report is a failure,
  not a clean allocation count.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

VALIDATED, FAILED, UNVALIDATED = 0, 1, 2

# Counter symbols, in the order they are tried. The CUDA and CPU plugins export
# the `nxrt_ep_*` names; the shared mock predates them and exports its own.
PLACEMENT_SYMBOLS = (
    "nxrt_ep_workspace_placement_queries",
    "nxrt_mock_shared_ep_placement_queries",
)
COMPILED_SYMBOLS = ("nxrt_ep_compiled_node_count",)

# Device-allocation APIs that must not appear per step. `cuMemAlloc_v2` /
# `cudaMalloc` are the allocation entry points; the frees are what make a
# CUDA-graph capture illegal.
ALLOC_APIS = ("cuMemAlloc", "cuMemFree", "cudaMalloc", "cudaFree")
# Stream-ordered variants are allocations too, but they are not what
# requirement 3 names, so they are reported separately instead of being
# silently folded into (or silently excluded from) the blocking counts.
ASYNC_ALLOC_APIS = (
    "cuMemAllocAsync",
    "cuMemFreeAsync",
    "cuMemAllocFromPoolAsync",
    "cudaMallocAsync",
    "cudaFreeAsync",
)
# A run that dispatched work must show launches. If none are parsed, the
# capture or the parse is broken, and "no allocations" from a broken capture is
# not evidence of anything.
LAUNCH_APIS = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx")


# ─── device / environment probing ────────────────────────────────────────────


def cuda_device_count() -> tuple[int, str]:
    """Return `(device_count, detail)` using the driver API, not `nvidia-smi`.

    `nvidia-smi` being absent is not proof of no device, and its presence is not
    proof of a usable one. `cuInit` + `cuDeviceGetCount` is what ORT's CUDA EP
    itself depends on, so that is what gets asked.
    """
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        return 0, f"libcuda.so.1 not loadable: {exc}"
    rc = lib.cuInit(0)
    if rc != 0:
        return 0, f"cuInit(0) returned {rc} (0 is success; 100 is CUDA_ERROR_NO_DEVICE)"
    count = ctypes.c_int(0)
    rc = lib.cuDeviceGetCount(ctypes.byref(count))
    if rc != 0:
        return 0, f"cuDeviceGetCount returned {rc}"
    return count.value, f"cuInit(0) succeeded; {count.value} device(s)"


def device_report() -> dict:
    """Driver/device/toolkit versions, for the evidence record."""
    out: dict = {}
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            q = subprocess.run(
                [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            out["nvidia_smi"] = q.stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - host dependent
            out["nvidia_smi_error"] = str(exc)
    else:
        out["nvidia_smi"] = "not found"
    out["uname"] = " ".join(os.uname())
    return out


def device_memory_used_mib() -> list[int] | None:
    """Per-GPU used memory, for the teardown leak check."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    q = subprocess.run(
        [smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if q.returncode != 0:
        return None
    return [int(line.strip()) for line in q.stdout.splitlines() if line.strip()]


# ─── models and reference numerics ───────────────────────────────────────────


@dataclass
class Geometry:
    batch: int = 2
    q_seq: int = 8
    kv_seq: int = 8
    q_heads: int = 4
    kv_heads: int = 4
    head_size: int = 32

    def step_scoped(self) -> bool:
        """`StandardAttention` declares `SessionPersistent` at, and only at,
        single-token single-batch decode; every other geometry is `StepScoped`
        and is served."""
        return not (self.batch == 1 and self.q_seq == 1)


def build_attention_model(geom: Geometry) -> bytes:
    import onnx
    from onnx import TensorProto, helper

    def vi(name: str, shape: list[int]):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    q = vi("Q", [geom.batch, geom.q_heads, geom.q_seq, geom.head_size])
    k = vi("K", [geom.batch, geom.kv_heads, geom.kv_seq, geom.head_size])
    v = vi("V", [geom.batch, geom.kv_heads, geom.kv_seq, geom.head_size])
    y = vi("Y", [geom.batch, geom.q_heads, geom.q_seq, geom.head_size])
    node = helper.make_node("Attention", ["Q", "K", "V"], ["Y"], name="attn_prefill")
    graph = helper.make_graph([node], "step_scoped_attention", [q, k, v], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model.SerializeToString()


def build_addmul_model(numel: int) -> bytes:
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [numel])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [numel])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [numel])
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "y"], ["s"], name="ws_add"),
            helper.make_node("Mul", ["s", "y"], ["out"], name="no_ws_mul"),
        ],
        "workspace_addmul",
        [x, y],
        [out],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model.SerializeToString()


def make_inputs(model_kind: str, geom: Geometry, numel: int) -> dict:
    import numpy as np

    rng = np.random.default_rng(20260813)
    if model_kind == "attention":
        return {
            "Q": rng.standard_normal((geom.batch, geom.q_heads, geom.q_seq, geom.head_size), dtype=np.float32),
            "K": rng.standard_normal((geom.batch, geom.kv_heads, geom.kv_seq, geom.head_size), dtype=np.float32),
            "V": rng.standard_normal((geom.batch, geom.kv_heads, geom.kv_seq, geom.head_size), dtype=np.float32),
        }
    return {
        "x": rng.standard_normal(numel, dtype=np.float32),
        "y": rng.standard_normal(numel, dtype=np.float32),
    }


def reference_output(model_kind: str, feeds: dict):
    """Independent reference, computed in float64.

    Deliberately not a second call into ORT: the point is to be wrong in
    different ways than the implementation under test. The ORT-CPU run is kept
    as a *second* comparison, not as the reference.
    """
    import numpy as np

    if model_kind != "attention":
        return (feeds["x"].astype(np.float64) + feeds["y"].astype(np.float64)) * feeds["y"].astype(np.float64)

    q = feeds["Q"].astype(np.float64)
    k = feeds["K"].astype(np.float64)
    v = feeds["V"].astype(np.float64)
    q_heads, kv_heads = q.shape[1], k.shape[1]
    if q_heads != kv_heads:  # grouped: repeat each kv head q_heads/kv_heads times
        k = np.repeat(k, q_heads // kv_heads, axis=1)
        v = np.repeat(v, q_heads // kv_heads, axis=1)
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.matmul(weights, v)


# ─── counters read out of the loaded cdylib ──────────────────────────────────


class Counters:
    """Reads the executor's counters from the very library ORT loaded.

    `ctypes.CDLL` on the same path returns the same `dlopen` handle ORT holds,
    so these are the live statics, not a second copy.
    """

    def __init__(self, lib_path: str) -> None:
        self._lib = ctypes.CDLL(lib_path)
        self._placement = self._bind(PLACEMENT_SYMBOLS)
        self._compiled = self._bind(COMPILED_SYMBOLS)

    def _bind(self, names: tuple[str, ...]):
        for name in names:
            try:
                fn = getattr(self._lib, name)
            except AttributeError:
                continue
            fn.restype = ctypes.c_size_t
            fn.argtypes = []
            return fn
        return None

    @property
    def has_placement(self) -> bool:
        return self._placement is not None

    @property
    def has_compiled(self) -> bool:
        return self._compiled is not None

    def placement(self) -> int:
        return int(self._placement()) if self._placement else -1

    def compiled(self) -> int:
        return int(self._compiled()) if self._compiled else -1


# ─── the run itself ──────────────────────────────────────────────────────────


@dataclass
class Result:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def check(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append((name, bool(ok), detail))

    def failed(self) -> list[str]:
        return [name for name, ok, _ in self.checks if not ok]


def select_ep_devices(ort, ep_name: str, device_index: int):
    devices = [d for d in ort.get_ep_devices() if d.ep_name == ep_name]
    if not devices:
        available = sorted({d.ep_name for d in ort.get_ep_devices()})
        raise RuntimeError(f"EP {ep_name!r} not discovered after registration; available: {available}")
    return [devices[min(device_index, len(devices) - 1)]]


def iobinding_device(args) -> tuple[str, int]:
    """Where inputs and outputs are made resident.

    The `attention` model runs on the CUDA plugin, so its I/O is pinned on the
    device; the `addmul` self-test runs on the CPU shared-mock EP, so its I/O
    stays on the host. Device-resident I/O is the whole point of this path: it
    removes the per-`Run` host->device staging (and its transient `cuMemAlloc`s)
    that ORT does when a session is fed host NumPy arrays every step, so the
    only device allocation the `nsys` capture can then see is the executor's own
    served-workspace scratch — which is what requirement 3 is actually about.
    """
    if args.model == "attention":
        return "cuda", args.device_index
    return "cpu", 0


def bind_device_io(ort, sess, feeds, geom, model_kind, device_type, device_id):
    """Bind every input to a device-resident `OrtValue` and bind the output to a
    single reused device `OrtValue`, so repeated `run_with_iobinding` calls do
    no per-`Run` input feeding and no per-`Run` output allocation.

    Returns `(io_binding, output_ortvalue)`; the caller reads the output back to
    the host once (outside any capture) via `output_ortvalue.numpy()`.
    """
    import numpy as np

    io_binding = sess.io_binding()
    # Kept alive by binding into `io_binding`; ORT holds no strong reference to
    # the Python OrtValue, so returning the output value keeps the whole set
    # reachable for the lifetime of the binding's use.
    io_binding._nxrt_inputs = [
        ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(arr), device_type, device_id)
        for arr in feeds.values()
    ]
    for name, value in zip(feeds.keys(), io_binding._nxrt_inputs):
        io_binding.bind_ortvalue_input(name, value)

    out_meta = sess.get_outputs()[0]
    if model_kind == "attention":
        out_shape = [geom.batch, geom.q_heads, geom.q_seq, geom.head_size]
    else:
        out_shape = list(feeds["x"].shape)
    output_value = ort.OrtValue.ortvalue_from_shape_and_type(out_shape, np.float32, device_type, device_id)
    io_binding.bind_ortvalue_output(out_meta.name, output_value)
    return io_binding, output_value


def run_phase(args) -> tuple[int, dict]:
    """Child phase: registers, runs and reports. Trace output goes to stderr,
    the JSON record to stdout, so the parent can read both."""
    import numpy as np
    import onnxruntime as ort

    result = Result()
    lib = str(Path(args.lib).resolve())
    geom = Geometry(
        batch=args.batch,
        q_seq=args.q_seq,
        kv_seq=args.kv_seq,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_size=args.head_size,
    )
    model = build_attention_model(geom) if args.model == "attention" else build_addmul_model(args.numel)
    feeds = make_inputs(args.model, geom, args.numel)
    expected = reference_output(args.model, feeds)

    result.data["ort_version"] = ort.__version__
    result.data["library"] = lib
    result.data["model"] = args.model
    result.data["geometry"] = geom.__dict__ if args.model == "attention" else {"numel": args.numel}

    # ORT's own CPU execution of the same graph: a second opinion that is
    # independent of both our EP and the NumPy reference.
    cpu_sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    cpu_out = cpu_sess.run(None, feeds)[0]
    del cpu_sess

    ort.register_execution_provider_library(args.registration_name, lib)
    counters = Counters(lib)
    # Harness falsifier hook: the shared mock can be told to declare
    # `SessionPersistent`, which the executor declines. Nothing is then served,
    # and `workspace_served` must go red — a check that cannot fail is not a
    # check.
    if os.environ.get("NXRT_HARNESS_FORCE_PERSISTENT") == "1":
        try:
            ctypes.CDLL(lib).nxrt_mock_shared_ep_set_persistent_workspace(ctypes.c_size_t(1))
            result.data["forced_persistent_workspace"] = True
        except AttributeError:
            result.data["forced_persistent_workspace"] = "symbol absent on this library"
    result.data["counter_symbols"] = {
        "placement": counters.has_placement,
        "compiled_nodes": counters.has_compiled,
    }
    if not counters.has_placement:
        result.check(
            "counters_exported",
            False,
            "the library exports no workspace placement counter, so 'the workspace path ran' "
            f"cannot be observed in the ORT process (looked for {list(PLACEMENT_SYMBOLS)})",
        )

    try:
        devices = select_ep_devices(ort, args.ep_name, args.device_index)
        result.data["ep_devices_discovered"] = len(
            [d for d in ort.get_ep_devices() if d.ep_name == args.ep_name]
        )

        so = ort.SessionOptions()
        so.add_provider_for_devices(devices, {})

        compiled_before, placement_before = counters.compiled(), counters.placement()
        sess = ort.InferenceSession(model, sess_options=so)
        providers = sess.get_providers()
        result.data["session_providers"] = providers
        result.check(
            "ep_selected",
            args.ep_name in providers,
            f"session providers = {providers}",
        )

        # Feed via IOBinding with device-resident inputs and a reused device
        # output, so ORT does not stage host arrays to the device every Run. The
        # only per-step device allocation left for `nsys` to see is then the
        # executor's served-workspace scratch, not a harness input-feed artifact.
        io_device_type, io_device_id = iobinding_device(args)
        result.data["io_binding"] = {"device_type": io_device_type, "device_id": io_device_id}
        io_binding, output_value = bind_device_io(
            ort, sess, feeds, geom, args.model, io_device_type, io_device_id
        )

        sess.run_with_iobinding(io_binding)
        actual = output_value.numpy()
        compiled_after, placement_after = counters.compiled(), counters.placement()
        result.data["compiled_nodes_delta"] = compiled_after - compiled_before
        result.data["placement_first_run"] = placement_after - placement_before
        result.check(
            "node_assigned_to_plugin_ep",
            compiled_after - compiled_before > 0,
            f"compiled-node counter moved by {compiled_after - compiled_before} "
            "(0 means ORT kept the node for another EP)",
        )
        result.check(
            "workspace_served",
            placement_after - placement_before > 0,
            f"placement resolutions on the first run = {placement_after - placement_before}; "
            "this counter only moves for a workspace the executor actually served",
        )

        diff_ref = float(np.max(np.abs(actual.astype(np.float64) - expected)))
        denom = float(np.max(np.abs(expected))) or 1.0
        diff_cpu = float(np.max(np.abs(actual.astype(np.float64) - cpu_out.astype(np.float64))))
        result.data["max_abs_err_vs_reference"] = diff_ref
        result.data["max_rel_err_vs_reference"] = diff_ref / denom
        result.data["max_abs_err_vs_ort_cpu"] = diff_cpu
        result.data["tolerance"] = {"atol": args.atol, "rtol": args.rtol}
        result.check(
            "numerics_vs_reference",
            diff_ref <= args.atol + args.rtol * denom,
            f"max|out - float64 reference| = {diff_ref:.3e} "
            f"(atol {args.atol:g} + rtol {args.rtol:g} * {denom:.3g})",
        )
        result.check(
            "numerics_vs_ort_cpu",
            diff_cpu <= args.atol + args.rtol * denom,
            f"max|out - ORT CPU| = {diff_cpu:.3e}",
        )

        # Steady state: the region an nsys capture is expected to cover. The
        # same IOBinding (device-resident inputs, reused device output) is
        # replayed, so no per-step host->device feed happens inside the capture.
        for _ in range(args.warmup):
            sess.run_with_iobinding(io_binding)
        steady_before = counters.placement()
        profiler = start_cuda_profiler() if args.cuda_profiler_range else None
        for _ in range(args.steps):
            sess.run_with_iobinding(io_binding)
        if profiler is not None:
            profiler.stop()
        steady_delta = counters.placement() - steady_before
        result.data["steady_steps"] = args.steps
        result.data["steady_placement_delta"] = steady_delta
        per_step = steady_delta / args.steps if args.steps else 0
        result.data["placements_per_step"] = per_step
        result.check(
            "workspace_served_every_step",
            steady_delta >= args.steps,
            f"{steady_delta} placement resolutions over {args.steps} steps "
            f"({per_step:.2f}/step); < 1/step means the node stopped being served",
        )
        del sess
    finally:
        ort.unregister_execution_provider_library(args.registration_name)

    return (FAILED if result.failed() else VALIDATED), {
        "checks": result.checks,
        "data": result.data,
    }


def start_cuda_profiler():
    """Bracket the steady-state loop with `cudaProfilerStart/Stop` so
    `nsys --capture-range=cudaProfilerApi` records only that region.

    Returns an object with `.stop()`, or `None` when the runtime API is not
    loadable (in which case the capture covers the whole process and session
    setup allocations are inside it - which is why the step-count comparison
    below exists as well)."""

    class _Profiler:
        def __init__(self, lib):
            self._lib = lib

        def stop(self):
            self._lib.cudaProfilerStop()

    for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        if lib.cudaProfilerStart() == 0:
            return _Profiler(lib)
    return None


def teardown_phase(args) -> tuple[int, dict]:
    """Two sessions on one registered library, then release.

    The failure this looks for is the executor reporting that something still
    holds the shared EP when ORT releases the factory, and any device memory
    that does not come back."""
    import onnxruntime as ort

    result = Result()
    lib = str(Path(args.lib).resolve())
    geom = Geometry(args.batch, args.q_seq, args.kv_seq, args.q_heads, args.kv_heads, args.head_size)
    model = build_attention_model(geom) if args.model == "attention" else build_addmul_model(args.numel)
    feeds = make_inputs(args.model, geom, args.numel)

    before = device_memory_used_mib()
    ort.register_execution_provider_library(args.registration_name, lib)
    try:
        devices = select_ep_devices(ort, args.ep_name, args.device_index)
        sessions = []
        options = []
        for _ in range(2):
            so = ort.SessionOptions()
            so.add_provider_for_devices(devices, {})
            sess = ort.InferenceSession(model, sess_options=so)
            sess.run(None, feeds)
            sessions.append(sess)
            options.append(so)
        result.data["sessions"] = len(sessions)
        providers = [s.get_providers() for s in sessions]
        result.data["session_providers"] = providers
        result.check(
            "both_sessions_on_plugin_ep",
            all(args.ep_name in p for p in providers),
            f"providers per session = {providers}",
        )
        # Everything ORT hands out that can hold the shared EP has to be gone
        # before the library is unregistered, or the teardown diagnostic is
        # measuring this harness rather than the executor. The control run
        # (`NXRT_HARNESS_RETAIN_SESSION=1`) deliberately skips this, to prove
        # the diagnostic still fires when something really is holding on.
        if os.environ.get("NXRT_HARNESS_RETAIN_SESSION") == "1":
            result.data["retained_session"] = "control run: a live session is held across unregister"
        else:
            del devices, so, sess
            sessions.clear()
            options.clear()
            gc.collect()
    finally:
        ort.unregister_execution_provider_library(args.registration_name)

    # Re-register and run once more: a factory that tore down cleanly can be
    # brought back up in the same process.
    ort.register_execution_provider_library(args.registration_name + "_again", lib)
    try:
        devices = select_ep_devices(ort, args.ep_name, args.device_index)
        so = ort.SessionOptions()
        so.add_provider_for_devices(devices, {})
        sess = ort.InferenceSession(model, sess_options=so)
        sess.run(None, feeds)
        result.check("reregistration_after_teardown", True, "second registration created a working session")
        del sess, devices, so
        gc.collect()
    finally:
        ort.unregister_execution_provider_library(args.registration_name + "_again")

    after = device_memory_used_mib()
    result.data["device_memory_used_mib"] = {"before": before, "after": after}
    if before and after and len(before) == len(after):
        worst = max(a - b for a, b in zip(before, after))
        result.data["device_memory_growth_mib"] = worst
        result.check(
            "no_device_memory_leak",
            worst <= args.leak_tolerance_mib,
            f"largest per-GPU growth across teardown = {worst} MiB "
            f"(tolerance {args.leak_tolerance_mib} MiB)",
        )
    else:
        # Said out loud, because a teardown section with no leak line reads as
        # "no leak" and it means "not measured".
        result.data["no_device_memory_leak"] = "NOT MEASURED - nvidia-smi unavailable, so the leak dimension is untested"
    return (FAILED if result.failed() else VALIDATED), {"checks": result.checks, "data": result.data}


# ─── trace and nsys parsing ──────────────────────────────────────────────────

TRACE_RE = re.compile(
    r"workspace served node=(?P<node>.+?) bytes=(?P<bytes>\d+) align=(?P<align>\d+) "
    r"requested_block=(?P<block_bytes>\d+) block=0x(?P<block>[0-9a-f]+) "
    r"block_align=(?P<block_align>\d+) ptr=0x(?P<ptr>[0-9a-f]+) skew=(?P<skew>\d+)"
)


def summarize_trace(stderr_text: str) -> dict:
    """Turn the executor's per-serve trace into the arena answer.

    One repeated block address means ORT handed back the same storage - it is
    reused, and no allocation happened after the first. Many addresses inside a
    small span means an arena that bump-allocates; addresses that keep climbing
    without bound are the shape that would imply a real allocation per step."""
    records = [m.groupdict() for m in TRACE_RE.finditer(stderr_text)]
    if not records:
        return {"served": 0}
    blocks = [int(r["block"], 16) for r in records]
    distinct = sorted(set(blocks))
    aligns = sorted({int(r["block_align"]) for r in records})
    skews = sorted({int(r["skew"]) for r in records})
    return {
        "served": len(records),
        "distinct_blocks": len(distinct),
        "block_span_bytes": (distinct[-1] - distinct[0]) if distinct else 0,
        "requested_alignment": sorted({int(r["align"]) for r in records}),
        "block_alignment_observed": aligns,
        "skew_bytes_observed": skews,
        "first_blocks": [f"0x{b:x}" for b in blocks[:4]],
        "reused_single_block": len(distinct) == 1,
        "ort_met_requested_alignment": skews == [0],
    }


def normalize_api_name(name: str) -> str:
    """`nsys` reports the driver API with its ABI version suffix
    (`cuMemAlloc_v2`, `cuMemFree_v2`). Matching the bare name without stripping
    it would miss every real allocation and pass a run that allocates."""
    return re.sub(r"_v\d+$", "", name.strip())


def parse_nsys_api_table(text: str) -> dict:
    """Every API row in an `nsys stats --report cuda_api_sum` report.

    Accepts both the CSV form (`--format csv`) and the whitespace table.
    Names are normalised by stripping the `_vN` ABI suffix, so `cuMemAlloc_v2`
    counts as `cuMemAlloc` while `cudaMallocAsync` stays a separate question.

    Returns every API, not just the allocation ones, because "no allocation
    rows" and "this report was not understood" are otherwise the same answer,
    and only one of them is evidence.
    """
    rows: list[tuple[str, int]] = []
    lines = [line for line in text.splitlines() if line.strip()]

    # Find the header wherever it is, not at line 0: real `nsys stats` prints
    # its own preamble ("Generating SQLite file...", "** CUDA API Summary
    # (cuda_api_sum):") ahead of the table. Deciding the format from the first
    # line alone would drop a perfectly good CSV report into the whitespace
    # parser, which reads no rows out of it, which now fails the run.
    header_at = None
    header_fields: list[str] = []
    for index, line in enumerate(lines):
        fields = [f.strip() for f in line.split(",")]
        if "Num Calls" in fields and "Name" in fields:
            header_at, header_fields = index, fields
            break

    if header_at is not None:
        calls_at = header_fields.index("Num Calls")
        name_at = header_fields.index("Name")
        for row in csv.reader(lines[header_at + 1 :]):
            if len(row) <= max(calls_at, name_at):
                continue
            calls = row[calls_at].strip().replace(",", "")
            name = row[name_at].strip()
            if name and calls.isdigit():
                rows.append((name, int(calls)))
    else:  # whitespace table: Time(%), Total Time, Num Calls, ..., Name
        for line in lines:
            fields = line.split()
            if len(fields) < 4:
                continue
            name = fields[-1]
            numbers = [f.replace(",", "") for f in fields[:-1]]
            if len(numbers) >= 3 and all(_looks_numeric(n) for n in numbers[:3]):
                rows.append((name, int(float(numbers[2]))))

    counts: dict[str, int] = {}
    for raw, calls in rows:
        name = normalize_api_name(raw)
        counts[name] = counts.get(name, 0) + calls
    return counts


def parse_nsys_api_sum(text: str) -> dict:
    """The allocation APIs only, out of [`parse_nsys_api_table`]."""
    table = parse_nsys_api_table(text)
    return {name: calls for name, calls in table.items() if name in ALLOC_APIS or name in ASYNC_ALLOC_APIS}


def _looks_numeric(field: str) -> bool:
    try:
        float(field)
    except ValueError:
        return False
    return True


NSYS_SAMPLE_TABLE = """
 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)  StdDev (ns)        Name
 --------  ---------------  ---------  ----------  ----------  --------  --------  -----------  ---------------
     61.2      1,204,551,1        512    23,526.4    22,101.0    18,004   102,331     4,120.7  cudaLaunchKernel
     12.0        236,004,0         64     3,687.5     3,401.0     2,900    12,004       901.2  cudaMallocAsync
      0.9         17,000,0          3     5,666.6     5,500.0     4,900     6,600       700.1  cuMemAlloc_v2
      0.4          8,000,0          3     2,666.6     2,500.0     2,100     3,300       500.0  cudaFree
"""

NSYS_SAMPLE_CSV = """Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
61.2,12045511,512,23526.4,22101.0,18004,102331,4120.7,cudaLaunchKernel
0.9,170000,3,5666.6,5500.0,4900,6600,700.1,cudaMalloc
0.4,80000,3,2666.6,2500.0,2100,3300,500.0,cuMemFree_v2
"""

# What `nsys stats` really writes: its own progress chatter, then a banner, then
# the table. A parser that only looks at line 0 for the header drops all of it.
NSYS_SAMPLE_CSV_WITH_PREAMBLE = (
    "Generating SQLite file ws_check_n64.sqlite from ws_check_n64.nsys-rep\n"
    "Processing [ws_check_n64.sqlite] with [/opt/nvidia/nsight-systems/host-linux-x64/reports/cuda_api_sum.py]...\n"
    "\n"
    "** CUDA API Summary (cuda_api_sum):\n"
    "\n" + NSYS_SAMPLE_CSV
)


def check_nsys_parser() -> list[tuple[str, bool, str]]:
    """The nsys parser is the one part of this harness that cannot be exercised
    without a GPU, so it is checked against captured report text on every
    self-test rather than trusted."""
    table = parse_nsys_api_sum(NSYS_SAMPLE_TABLE)
    csv_form = parse_nsys_api_sum(NSYS_SAMPLE_CSV)
    return [
        (
            "nsys_parser_reads_num_calls_column",
            table.get("cuMemAlloc") == 3 and table.get("cudaFree") == 3,
            f"table form -> {table} (expected cuMemAlloc=3, cudaFree=3)",
        ),
        (
            "nsys_parser_strips_abi_version_suffix",
            csv_form.get("cuMemFree") == 3,
            f"cuMemFree_v2 must count as cuMemFree; got {csv_form}",
        ),
        (
            "nsys_parser_keeps_async_variants_separate",
            table.get("cudaMallocAsync") == 64 and table.get("cudaMalloc") is None,
            f"cudaMallocAsync must not be counted as cudaMalloc; got {table}",
        ),
        (
            "nsys_parser_reports_nothing_for_a_clean_trace",
            parse_nsys_api_sum("Time (%),Total Time (ns),Num Calls,Name\n100.0,1,512,cudaLaunchKernel\n") == {},
            "a trace with no allocation APIs must produce an empty count, not a default",
        ),
        (
            "nsys_parser_distinguishes_clean_from_unreadable",
            parse_nsys_api_table("Time (%),Total Time (ns),Num Calls,Name\n100.0,1,512,cudaLaunchKernel\n") != {}
            and parse_nsys_api_table("") == {}
            and parse_nsys_api_table("**** cuda_api_sum: no data ****") == {},
            "a clean report still parses rows; an empty or unreadable one parses none, "
            "which is what makes the fail-closed capture check possible",
        ),
        (
            "nsys_parser_finds_the_launch_positive_control",
            any(
                api in parse_nsys_api_table(NSYS_SAMPLE_TABLE) for api in LAUNCH_APIS
            ),
            f"a launch API must be visible in a real report: {sorted(parse_nsys_api_table(NSYS_SAMPLE_TABLE))}",
        ),
        (
            "nsys_parser_skips_the_stats_preamble",
            parse_nsys_api_table(NSYS_SAMPLE_CSV_WITH_PREAMBLE) == parse_nsys_api_table(NSYS_SAMPLE_CSV),
            "nsys stats prints progress lines and a banner before the table; finding the "
            f"header only at line 0 would fail a clean capture. got "
            f"{parse_nsys_api_table(NSYS_SAMPLE_CSV_WITH_PREAMBLE)}",
        ),
    ]


# ─── orchestration ───────────────────────────────────────────────────────────


def child_command(args, phase: str, steps: int | None = None) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        phase,
        "--lib",
        str(Path(args.lib).resolve()),
        "--ep-name",
        args.ep_name,
        "--registration-name",
        args.registration_name,
        "--model",
        args.model,
        "--device-index",
        str(args.device_index),
        "--steps",
        str(steps if steps is not None else args.steps),
        "--warmup",
        str(args.warmup),
        "--numel",
        str(args.numel),
        "--batch",
        str(args.batch),
        "--q-seq",
        str(args.q_seq),
        "--kv-seq",
        str(args.kv_seq),
        "--q-heads",
        str(args.q_heads),
        "--kv-heads",
        str(args.kv_heads),
        "--head-size",
        str(args.head_size),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--leak-tolerance-mib",
        str(args.leak_tolerance_mib),
    ]
    if args.cuda_profiler_range:
        cmd.append("--cuda-profiler-range")
    return cmd


def run_child(args, phase: str, steps: int | None = None, under_nsys: Path | None = None, extra_env: dict | None = None):
    env = dict(os.environ)
    env["NXRT_EP_WORKSPACE_TRACE"] = "1"
    env.update(extra_env or {})
    cmd = child_command(args, phase, steps)
    if under_nsys is not None:
        nsys_cmd = [
            "nsys",
            "profile",
            "--trace=cuda",
            "--force-overwrite=true",
            "-o",
            str(under_nsys),
        ]
        if args.cuda_profiler_range:
            nsys_cmd += ["--capture-range=cudaProfilerApi", "--capture-range-end=stop"]
        cmd = nsys_cmd + cmd
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    record = None
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            record = json.loads(line[len("__RESULT__") :])
    return proc, record


def emit(result_code: int, payload: dict) -> None:
    print("__RESULT__" + json.dumps({"code": result_code, **payload}))


def teardown_blockers(stderr: str) -> list[str]:
    """Lines where the executor reported it could not take exclusive ownership
    of the shared EP and therefore skipped `shutdown()`.

    Matched on two independent phrases from `release_ep_factory_with_teardown`
    so a reworded diagnostic degrades to a noisier match rather than to silence.
    """
    return [
        line
        for line in stderr.splitlines()
        if "ReleaseEpFactory called while" in line or "Skipping explicit shutdown" in line
    ]


def print_checks(title: str, record: dict) -> bool:
    print(f"\n── {title} " + "─" * max(4, 60 - len(title)))
    ok = True
    for name, passed, detail in record.get("checks", []):
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        ok &= passed
    for key, value in record.get("data", {}).items():
        print(f"        {key}: {value}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lib", default="target/release/libonnx_runtime_ep_cuda_plugin.so")
    parser.add_argument("--ep-name", default="cuda_ep")
    parser.add_argument("--registration-name", default="nxrt_ep_under_test")
    parser.add_argument("--model", choices=["attention", "addmul"], default="attention")
    parser.add_argument("--device-index", type=int, default=int(os.environ.get("ONNX_GENAI_CUDA_DEVICE", "0")))
    parser.add_argument("--steps", type=int, default=64, help="steady-state runs")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--numel", type=int, default=1024, help="addmul element count")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--q-seq", type=int, default=8)
    parser.add_argument("--kv-seq", type=int, default=8)
    parser.add_argument("--q-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-size", type=int, default=32)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--leak-tolerance-mib", type=int, default=64)
    parser.add_argument("--nsys", action="store_true", help="capture a CUDA API trace of the steady-state loop")
    parser.add_argument("--nsys-output", default="ws_check")
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="bracket the steady-state loop with cudaProfilerStart/Stop (use with --nsys)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate this harness against the CPU shared-mock plugin; no GPU required",
    )
    parser.add_argument("--allow-cpu", action="store_true", help="skip the GPU precondition")
    parser.add_argument("--phase", choices=["run", "teardown"], help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.self_test:
        args.model = "addmul"
        args.ep_name = "nxrt_shared_mock_ep"
        args.allow_cpu = True
        if args.lib == parser.get_default("lib"):
            args.lib = "target/debug/libonnx_runtime_ep_shared_mock_plugin.so"

    if args.phase:  # child
        try:
            code, payload = (run_phase if args.phase == "run" else teardown_phase)(args)
        except Exception as exc:  # noqa: BLE001 - a child failure is a finding, not a crash report
            emit(FAILED, {"checks": [[f"{args.phase}_phase", False, f"{type(exc).__name__}: {exc}"]], "data": {}})
            return FAILED
        emit(code, payload)
        return code

    if args.model == "attention" and not Geometry(args.batch, args.q_seq).step_scoped():
        print("UNVALIDATED: batch == 1 and q_seq == 1 is the one geometry where Attention")
        print("             declares SessionPersistent, which the executor declines. Nothing")
        print("             would be served, so this run could not be evidence. Use batch > 1")
        print("             or q_seq > 1.")
        return UNVALIDATED

    lib_path = Path(args.lib)
    if not lib_path.exists():
        print(f"UNVALIDATED: plugin library not found: {lib_path}")
        print("  build it first, e.g. cargo build --release -p onnx-runtime-ep-cuda-plugin --features cuda")
        return UNVALIDATED

    devices, detail = cuda_device_count()
    print(f"CUDA driver probe: {detail}")
    for key, value in device_report().items():
        print(f"  {key}: {value}")
    if devices == 0 and not args.allow_cpu:
        print("\nUNVALIDATED: no CUDA device is reachable from this host, so nothing here")
        print("             can speak to the device behaviour of the workspace path.")
        return UNVALIDATED

    overall_ok = True

    if args.self_test:
        overall_ok &= print_checks("nsys report parser", {"checks": check_nsys_parser(), "data": {}})

    proc, record = run_child(args, "run")
    if record is None:
        print("FAILED: run phase produced no result record")
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:], file=sys.stderr)
        return FAILED
    overall_ok &= print_checks("workspace serving", record)
    trace = summarize_trace(proc.stderr)
    print("\n── ORT scratch, as observed " + "─" * 34)
    for key, value in trace.items():
        print(f"        {key}: {value}")
    if trace.get("served", 0) == 0:
        print("  [FAIL] no served-workspace trace lines: the executor never served a workspace")
        overall_ok = False
    else:
        print(
            "  arena conclusion: "
            + (
                "ORT returned one block for every serve - the storage is reused"
                if trace.get("reused_single_block")
                else f"ORT returned {trace['distinct_blocks']} distinct blocks within "
                f"{trace['block_span_bytes']} bytes - reused region, moving offset"
            )
        )

    if args.nsys:
        if shutil.which("nsys") is None:
            print("\n  [SKIP] nsys not on PATH; CUDA API trace not captured")
            overall_ok = False
        else:
            out_small = Path(f"{args.nsys_output}_n{args.steps}")
            out_large = Path(f"{args.nsys_output}_n{args.steps * 4}")
            counts: dict[str, dict] = {}
            tables: dict[str, dict] = {}
            capture_ok = True
            for label, out, steps in (("n", out_small, args.steps), ("4n", out_large, args.steps * 4)):
                proc_n, _ = run_child(args, "run", steps=steps, under_nsys=out)
                rep = out.with_suffix(".nsys-rep")
                stats = subprocess.run(
                    ["nsys", "stats", "--report", "cuda_api_sum", "--format", "csv", str(rep)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                tables[label] = parse_nsys_api_table(stats.stdout)
                counts[label] = {
                    api: n for api, n in tables[label].items() if api in ALLOC_APIS or api in ASYNC_ALLOC_APIS
                }
                print(f"\n── nsys cuda_api_sum ({steps} steps) " + "─" * 26)
                print(f"        report: {rep}")
                print(f"        apis parsed: {len(tables[label])}")
                for api, num in sorted(counts[label].items()):
                    print(f"        {api}: {num} calls")

                # An empty parse and a clean trace are indistinguishable unless
                # the capture is proven to have produced a report this parser
                # understood. Everything below is fail-closed for that reason.
                if proc_n.returncode not in (VALIDATED, FAILED):
                    print(f"  [FAIL] the profiled child exited {proc_n.returncode}; the capture is not a run")
                    capture_ok = False
                if stats.returncode != 0:
                    print(f"  [FAIL] nsys stats exited {stats.returncode}: {stats.stderr.strip()[:300]}")
                    capture_ok = False
                if not tables[label]:
                    print("  [FAIL] no API rows parsed out of the report: either the capture is")
                    print("         empty or this nsys writes a format this parser does not read.")
                    print("         An empty allocation count from an unparsed report is not evidence.")
                    capture_ok = False
                elif not any(api in tables[label] for api in LAUNCH_APIS):
                    print(f"  [FAIL] no kernel-launch API ({'/'.join(LAUNCH_APIS)}) in the report;")
                    print("         a run that dispatched work cannot have launched nothing, so the")
                    print("         capture does not cover the steady-state loop")
                    capture_ok = False
                if not counts[label]:
                    print("        no cuMemAlloc/cuMemFree/cudaMalloc/cudaFree rows in the capture")

            if not capture_ok:
                overall_ok = False
            else:
                launched = {
                    label: sum(tables[label].get(api, 0) for api in LAUNCH_APIS) for label in ("n", "4n")
                }
                print(f"\n  positive control, kernel launches: n={launched['n']} 4n={launched['4n']}")
                if launched["4n"] <= launched["n"]:
                    print("  [FAIL] launches did not grow with step count, so this pair of captures")
                    print("         cannot distinguish a per-step allocation from a one-off one")
                    overall_ok = False

                grew = {
                    api: (counts["n"].get(api, 0), counts["4n"].get(api, 0))
                    for api in set(counts["n"]) | set(counts["4n"])
                    if counts["4n"].get(api, 0) > counts["n"].get(api, 0)
                }
                blocking_grew = {api: pair for api, pair in grew.items() if api in ALLOC_APIS}
                async_grew = {api: pair for api, pair in grew.items() if api in ASYNC_ALLOC_APIS}
                print("  allocation APIs that scale with step count: " + (str(grew) if grew else "none"))
                if async_grew and not blocking_grew:
                    print(f"  note: stream-ordered variants scaled ({async_grew}); requirement 3 names")
                    print("        the blocking APIs, but this says the arena is growing per step")
                if blocking_grew:
                    print("  [FAIL] a blocking allocation API grew with the number of steps: the")
                    print(f"         workspace path is allocating per dispatch {blocking_grew}")
                    overall_ok = False

    proc_t, record_t = run_child(args, "teardown")
    if record_t is None:
        print("\nFAILED: teardown phase produced no result record")
        print(proc_t.stderr[-4000:], file=sys.stderr)
        return FAILED
    overall_ok &= print_checks("two-session teardown", record_t)
    blockers = teardown_blockers(proc_t.stderr)
    print(f"        teardown blocker diagnostics: {len(blockers)}")
    for line in blockers:
        print(f"        > {line}")
    if blockers:
        print("  [FAIL] the explicit shutdown path was skipped: something still held the")
        print("         shared EP when the library was unregistered")
        overall_ok = False
    else:
        # "No diagnostic" is only evidence if a diagnostic could have appeared.
        # The control run holds a session across unregister, which must trip it.
        proc_c, _ = run_child(args, "teardown", extra_env={"NXRT_HARNESS_RETAIN_SESSION": "1"})
        control = teardown_blockers(proc_c.stderr)
        print(f"  control (session deliberately retained): {len(control)} blocker diagnostic(s), child exited {proc_c.returncode}")
        if proc_c.returncode < 0:
            print("        (the control run unregisters the library under a live session, which is")
            print("         an ORT contract violation; the diagnostic is the last thing printed")
            print("         before the unload takes the process down, and that is the point of it)")
        if not control:
            print("  [FAIL] holding a live session across unregister produced no diagnostic, so")
            print("         the clean run above cannot be read as evidence that teardown was clean")
            overall_ok = False

    print("\n" + ("VALIDATED" if overall_ok else "FAILED"))
    return VALIDATED if overall_ok else FAILED


if __name__ == "__main__":
    sys.exit(main())
