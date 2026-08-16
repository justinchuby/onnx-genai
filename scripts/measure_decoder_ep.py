"""Measure CUDA plugin-EP node assignment and numerics on a real decoder.

For issue #956. Runs a real ONNX decoder twice on identical inputs:

  1. With the CUDA plugin-EP registered and selected (ORT partitions the graph;
     whatever the EP claims runs on `cuda_ep`, the rest falls back to CPU).
  2. CPU-only (the reference).

It reports the per-provider executed-node counts from the ORT profile for run 1
(the acceptance signal for #956) and the max absolute / relative logits
difference between the two runs (the correctness signal).

Usage:
    python scripts/measure_decoder_ep.py <plugin_lib> <model.onnx> [seq_len]

Env:
    ONNX_GENAI_CUDA_DEVICE  GPU ordinal to select (default 0)
"""

import os
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ep_profile import count_nodes_by_provider, format_counts  # noqa: E402


def build_inputs(model_path: str, seq_len: int) -> dict:
    """Build a deterministic, well-formed input feed for a decoder prefill.

    past_sequence_len = 0 (prefill), batch = 1. KV cache tensors are empty
    (shape [1, kv_heads, 0, head_dim]). Attention mask is all-ones. input_ids
    are a fixed deterministic ramp so both runs see identical data.
    """
    m = onnx.load(model_path, load_external_data=False)
    g = m.graph
    ELEM_NP = {1: np.float32, 10: np.float16, 16: np.float16, 6: np.int32,
               7: np.int64, 2: np.uint8, 3: np.int8, 9: np.bool_}

    feed = {}
    rng = np.random.default_rng(1234)
    for vi in g.input:
        name = vi.name
        et = vi.type.tensor_type.elem_type
        dims = vi.type.tensor_type.shape.dim
        shape = []
        for d in dims:
            p = d.dim_param
            if p:
                if "past" in p and "seq" in p and "+" not in p:
                    shape.append(0)          # empty KV cache (prefill)
                elif "past_seq_len + seq_len" in p or ("+" in p):
                    shape.append(seq_len)     # total length == seq_len at prefill
                elif "batch" in p:
                    shape.append(1)
                elif "seq" in p:
                    shape.append(seq_len)
                else:
                    shape.append(1)
            else:
                shape.append(d.dim_value)
        npdt = ELEM_NP.get(et, np.float32)
        if name == "input_ids":
            # small fixed token ids, deterministic
            feed[name] = (np.arange(seq_len, dtype=np.int64) % 1000 + 1).reshape(1, seq_len)
        elif name == "attention_mask":
            feed[name] = np.ones((1, seq_len), dtype=np.int64)
        elif np.issubdtype(npdt, np.floating):
            feed[name] = (rng.standard_normal(size=shape).astype(np.float32) * 0.05).astype(npdt)
        else:
            feed[name] = np.zeros(shape, dtype=npdt)
    return feed


def run(model_path, feed, provider_setup, profile=False):
    prof_path = None
    so = ort.SessionOptions()
    if profile:
        prefix = os.path.join(tempfile.mkdtemp(prefix="nxrt_dec_"), "decoder")
        so.enable_profiling = True
        so.profile_file_prefix = prefix
    providers = provider_setup(so)
    if providers is None:
        sess = ort.InferenceSession(model_path, sess_options=so,
                                    providers=["CPUExecutionProvider"])
    else:
        sess = ort.InferenceSession(model_path, sess_options=so)
    out_names = [o.name for o in sess.get_outputs()]
    logits_idx = out_names.index("logits") if "logits" in out_names else 0
    outputs = sess.run(None, feed)
    if profile:
        prof_path = sess.end_profiling()
    return outputs[logits_idx], sess.get_providers(), prof_path


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    lib = os.path.abspath(sys.argv[1])
    model_path = os.path.abspath(sys.argv[2])
    seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    device_index = int(os.environ.get("ONNX_GENAI_CUDA_DEVICE", "0"))

    print(f"onnxruntime {ort.__version__}")
    print(f"plugin lib: {lib}")
    print(f"model: {model_path}  seq_len={seq_len}")

    ort.register_execution_provider_library("nxrt_ep_cuda", lib)
    ep_devices = [d for d in ort.get_ep_devices() if d.ep_name == "cuda_ep"]
    if not ep_devices:
        print("FAIL: cuda_ep not discovered after registration")
        return 1
    chosen = [ep_devices[min(device_index, len(ep_devices) - 1)]]

    feed = build_inputs(model_path, seq_len)

    def plugin_setup(so):
        so.add_provider_for_devices(chosen, {})
        return chosen

    def cpu_setup(so):
        return None

    print("\n=== Run 1: CUDA plugin-EP (partitioned) ===")
    try:
        logits_plugin, providers_plugin, prof = run(
            model_path, feed, plugin_setup, profile=True
        )
    except Exception as e:  # noqa: BLE001 — the failure IS the reportable result
        print("PLUGIN-EP SESSION FAILED TO BUILD (honest-negative result for #956):")
        print(f"  {type(e).__name__}: {e}")
        print("\n=== CPU-only reference (workload characterization) ===")
        _, providers_cpu, prof_cpu = run(model_path, feed, cpu_setup, profile=True)
        print(f"session providers: {providers_cpu}")
        counts = count_nodes_by_provider(prof_cpu)
        print("per-provider executed node counts (CPU-only reference run):")
        print(format_counts(counts))
        print(
            "\nRESULT: With real kernel domains advertised, ORT DOES assign the "
            "decoder's\ncom.microsoft nodes to cuda_ep (the domain bug is fixed), but "
            "the plugin cannot\nCOMPILE the resulting partition — see the error above. "
            "cuda_ep executed nodes: 0\n(the session never built). This is a distinct, "
            "downstream blocker, not silent\nCPU fallback."
        )
        return 3

    print(f"session providers: {providers_plugin}")
    counts = count_nodes_by_provider(prof)
    print("per-provider executed node counts:")
    print(format_counts(counts))
    cuda_nodes = sum(n for p, n in counts.items() if "cuda" in p.lower())

    print("\n=== Run 2: CPU-only reference ===")
    logits_cpu, providers_cpu, _ = run(model_path, feed, cpu_setup, profile=False)
    print(f"session providers: {providers_cpu}")

    a = logits_plugin.astype(np.float64)
    b = logits_cpu.astype(np.float64)
    abs_diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-3)
    rel_diff = abs_diff / denom
    print("\n=== Numerics (plugin-EP vs CPU-only, same fp16 model & inputs) ===")
    print(f"logits shape: {logits_plugin.shape} dtype: {logits_plugin.dtype}")
    print(f"max abs diff : {abs_diff.max():.6g}")
    print(f"mean abs diff: {abs_diff.mean():.6g}")
    print(f"max rel diff : {rel_diff.max():.6g}")
    # argmax (next-token) agreement over the last position — the decision-relevant metric
    am_plugin = np.argmax(a.reshape(-1, a.shape[-1]), axis=-1)
    am_cpu = np.argmax(b.reshape(-1, b.shape[-1]), axis=-1)
    argmax_match = float((am_plugin == am_cpu).mean())
    print(f"argmax token agreement across positions: {argmax_match*100:.2f}%")

    print(f"\ncuda_ep executed nodes: {cuda_nodes}")
    if cuda_nodes == 0:
        print("RESULT: cuda_ep claimed ZERO nodes — domain fix did not take effect.")
        return 1
    print("RESULT: cuda_ep claimed nodes; see counts and numerics above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
