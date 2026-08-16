"""Report how many decoder nodes the CUDA plugin-EP CLAIMED (compiled) at
session-creation time, without running inference.

This reads the plugin's process-global `nxrt_ep_compiled_node_count()` counter,
which is incremented once per node the plugin actually compiled into a fused
cuda_ep subgraph during ORT's Compile phase. It therefore reports the cuda_ep
node-assignment count for issue #956 *without* executing the model — which
matters because executing the partitioned decoder currently deadlocks at the
CPU<->GPU boundary (blocker "B3"), so the ORT profile can't be completed.

Usage: python scripts/measure_decoder_claims.py <plugin_lib> <model.onnx> [seq_len]
"""
import ctypes
import os
import sys

import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_decoder_ep import build_inputs  # noqa: E402


def main() -> int:
    lib = os.path.abspath(sys.argv[1])
    model_path = os.path.abspath(sys.argv[2])
    seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    print(f"onnxruntime {ort.__version__}", flush=True)
    print(f"plugin lib: {lib}", flush=True)
    print(f"model: {model_path}  seq_len={seq_len}", flush=True)

    dll = ctypes.CDLL(lib)
    dll.nxrt_ep_compiled_node_count.restype = ctypes.c_size_t
    dll.nxrt_ep_reset_compiled_node_count.restype = None

    ort.register_execution_provider_library("nxrt_ep_cuda", lib)
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "cuda_ep"]
    if not devs:
        print("FAIL: cuda_ep not discovered", flush=True)
        return 1
    print(f"cuda_ep devices discovered: {len(devs)}", flush=True)

    dll.nxrt_ep_reset_compiled_node_count()

    so = ort.SessionOptions()
    so.add_provider_for_devices([devs[0]], {})
    print("creating session (this runs ORT Compile => plugin claims + compiles nodes)...", flush=True)
    sess = ort.InferenceSession(model_path, sess_options=so)
    claimed = dll.nxrt_ep_compiled_node_count()
    print(f"session providers: {sess.get_providers()}", flush=True)
    print(f"\ncuda_ep CLAIMED (compiled) node count: {claimed}", flush=True)
    print(
        "(On main @ HEAD the same model claims 0 — all 585 executed nodes run on "
        "CPUExecutionProvider. The reviewer's before-count.)",
        flush=True,
    )
    # Prove the inputs feed builds too (parity with measure_decoder_ep harness),
    # but DO NOT run: execution deadlocks at the CPU<->GPU boundary (B3).
    _ = build_inputs(model_path, seq_len)
    print("\nNOTE: not executing sess.run — partitioned decode currently deadlocks "
          "in a CUDA memcpy at the CPU<->cuda_ep boundary (blocker B3).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
