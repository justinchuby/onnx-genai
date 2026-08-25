"""Real ORT CUDA-plugin session validation for pkg.nxrt::DsaIndexSelect v1."""

import ctypes
import gc
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def model_bytes(version: int, storage: int = TensorProto.FLOAT) -> bytes:
    query = helper.make_tensor_value_info("query", storage, [1, 2, 1, 2])
    key = helper.make_tensor_value_info("key", storage, [1, 4, 2])
    weights = helper.make_tensor_value_info("weights", storage, [1, 2, 1])
    bias = helper.make_tensor_value_info("attention_bias", TensorProto.FLOAT, [1, 1, 2, 4])
    output = helper.make_tensor_value_info(
        "selected_indices", TensorProto.INT64, [1, 1, 2, 2]
    )
    node = helper.make_node(
        "DsaIndexSelect",
        ["query", "key", "weights", "attention_bias"],
        ["selected_indices"],
        domain="pkg.nxrt",
        top_k=2,
        scale=1.0,
        weights_scale=1.0,
    )
    graph = helper.make_graph(
        [node],
        f"dsa_index_select_v{version}",
        [query, key, weights, bias],
        [output],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 20),
            helper.make_opsetid("pkg.nxrt", version),
        ],
    )
    model.ir_version = 11
    return model.SerializeToString()


def bind_u64(lib: ctypes.CDLL, name: str):
    function = getattr(lib, name)
    function.argtypes = []
    function.restype = ctypes.c_uint64
    return function


def cpu_reference(
    feeds: dict[str, np.ndarray],
    top_k: int,
    scale: float,
    weights_scale: float,
) -> np.ndarray:
    query = feeds["query"]
    key = feeds["key"]
    weights = feeds["weights"]
    bias = feeds["attention_bias"]
    batch, query_sequence, heads, head_dim = query.shape
    key_sequence = key.shape[1]
    output = np.full((batch, 1, query_sequence, top_k), -1, dtype=np.int64)
    scale = np.float32(scale)
    weights_scale = np.float32(weights_scale)

    for b in range(batch):
        for s in range(query_sequence):
            candidates: list[tuple[np.float32, int]] = []
            for t in range(key_sequence):
                bias_value = np.float32(bias[b, 0, s, t])
                if not bias_value > np.float32(-1e30):
                    continue
                weighted = np.float32(0)
                for h in range(heads):
                    dot = np.float32(0)
                    for d in range(head_dim):
                        product = np.float32(
                            np.float32(query[b, s, h, d])
                            * np.float32(key[b, t, d])
                        )
                        dot = np.float32(dot + product)
                    scored = np.maximum(np.float32(scale * dot), np.float32(0))
                    weight = np.float32(
                        np.float32(weights[b, s, h]) * weights_scale
                    )
                    weighted = np.float32(
                        weighted + np.float32(np.float32(scored) * weight)
                    )
                candidates.append((np.float32(weighted + bias_value), t))
            selected = sorted(candidates, key=lambda item: (-float(item[0]), item[1]))[
                :top_k
            ]
            for slot, position in enumerate(sorted(t for _, t in selected)):
                output[b, 0, s, slot] = position
    return output


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_dsa_index_select_plugin.py <cuda-plugin-cdylib>")
        return 2
    plugin_path = os.path.abspath(sys.argv[1])
    plugin = ctypes.CDLL(plugin_path)

    reset_workspace = plugin.nxrt_cuda_reset_dsa_workspace_stats
    reset_workspace.argtypes = []
    reset_workspace.restype = ctypes.c_bool
    set_capture_replays = plugin.nxrt_cuda_set_dsa_capture_replays_for_test
    set_capture_replays.argtypes = [ctypes.c_uint64]
    set_capture_replays.restype = None
    workspace_allocations = bind_u64(plugin, "nxrt_cuda_dsa_workspace_allocations")
    workspace_releases = bind_u64(plugin, "nxrt_cuda_dsa_workspace_releases")
    workspace_live_bytes = bind_u64(plugin, "nxrt_cuda_dsa_workspace_live_bytes")
    workspace_last_ptr = bind_u64(plugin, "nxrt_cuda_dsa_workspace_last_ptr")
    capture_count = bind_u64(plugin, "nxrt_cuda_dsa_capture_count_for_test")
    captured_replays = bind_u64(plugin, "nxrt_cuda_dsa_captured_replays_for_test")
    capture_error = bind_u64(plugin, "nxrt_cuda_dsa_capture_error_for_test")
    compiled_nodes = bind_u64(plugin, "nxrt_ep_compiled_node_count")
    workspace_placement_queries = bind_u64(
        plugin, "nxrt_ep_workspace_placement_queries"
    )
    reset_placement_queries = plugin.nxrt_ep_reset_workspace_placement_queries
    reset_placement_queries.argtypes = []
    reset_placement_queries.restype = None

    assert reset_workspace(), "workspace accounting was not idle before session creation"
    reset_placement_queries()
    os.environ["ONNX_GENAI_CUDA_RAW_POOL_BYTES"] = "0"
    registration_name = "nxrt_dsa_cuda_validation"
    ort.register_execution_provider_library(registration_name, plugin_path)
    devices = [device for device in ort.get_ep_devices() if device.ep_name == "cuda_ep"]
    assert devices, "CUDA plugin registered but exposed no cuda_ep device"

    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices([devices[0]], {})
    session = ort.InferenceSession(model_bytes(1), sess_options=options)

    feeds = {
        "query": np.array([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=np.float32),
        "key": np.array([[[5.0, 0.0], [0.0, 5.0], [3.0, 3.0], [-1.0, -1.0]]], dtype=np.float32),
        "weights": np.ones((1, 2, 1), dtype=np.float32),
        "attention_bias": np.zeros((1, 1, 2, 4), dtype=np.float32),
    }
    expected = cpu_reference(feeds, top_k=2, scale=1.0, weights_scale=1.0)
    np.testing.assert_array_equal(
        expected, np.array([[[[0, 2], [1, 2]]]], dtype=np.int64)
    )

    set_capture_replays(3)
    (captured_output,) = session.run(None, feeds)
    np.testing.assert_array_equal(captured_output, expected)
    assert compiled_nodes() >= 1, "real ORT session did not compile DsaIndexSelect on cuda_ep"
    assert capture_count() == 1, "session execution did not record one CUDA graph"
    assert captured_replays() == 3, "session execution did not replay the graph three times"
    assert capture_error() == 0, "valid capture/replay poisoned the CUDA capture-error latch"
    assert workspace_allocations() == 1
    assert workspace_releases() == 0
    assert workspace_live_bytes() == 512
    assert workspace_placement_queries() == 0
    first_ptr = workspace_last_ptr()
    assert first_ptr != 0

    for _ in range(3):
        (output,) = session.run(None, feeds)
        np.testing.assert_array_equal(output, expected)
        assert workspace_last_ptr() == first_ptr
        assert workspace_allocations() == 1
        assert workspace_live_bytes() == 512

    del captured_output, output, session, options
    gc.collect()
    assert workspace_live_bytes() == 0, "session teardown leaked DSA workspace bytes"
    assert workspace_allocations() == workspace_releases() == 1

    for storage, dtype in [(TensorProto.FLOAT16, np.float16)]:
        low_options = ort.SessionOptions()
        low_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        low_options.add_provider_for_devices([devices[0]], {})
        low_session = ort.InferenceSession(model_bytes(1, storage), sess_options=low_options)
        low_feeds = {
            name: value.astype(dtype) if name != "attention_bias" else value
            for name, value in feeds.items()
        }
        low_expected = cpu_reference(
            low_feeds, top_k=2, scale=1.0, weights_scale=1.0
        )
        (low_output,) = low_session.run(None, low_feeds)
        np.testing.assert_array_equal(low_output, low_expected)
        del low_output, low_session, low_options
        gc.collect()

    assert workspace_live_bytes() == 0
    assert workspace_allocations() == workspace_releases() == 2

    bf16_options = ort.SessionOptions()
    bf16_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    bf16_options.add_provider_for_devices([devices[0]], {})
    bf16_session = ort.InferenceSession(
        model_bytes(1, TensorProto.BFLOAT16), sess_options=bf16_options
    )
    del bf16_session, bf16_options
    gc.collect()

    bad_options = ort.SessionOptions()
    bad_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    bad_options.add_provider_for_devices([devices[0]], {})
    try:
        ort.InferenceSession(model_bytes(2), sess_options=bad_options)
    except Exception:
        pass
    else:
        raise AssertionError("frozen DsaIndexSelect v2 unexpectedly loaded in a real session")

    print(
        "PASS: real CUDA-plugin session ran f32/f16 v1 with CPU-reference parity, "
        "claimed bf16 v1 without fallback, "
        "captured once, replayed 3x, reused one 512-byte workspace pointer, "
        "released it at teardown, and rejected v2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
