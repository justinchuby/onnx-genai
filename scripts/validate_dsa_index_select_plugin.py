"""Real ORT CUDA-plugin session validation for pkg.nxrt::DsaIndexSelect v1."""

import ctypes
import gc
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


CANONICAL_NAN_BITS = np.uint32(0x7FC00000)


def canonicalize_score(value: np.float32) -> np.float32:
    if np.isnan(value):
        return np.asarray(CANONICAL_NAN_BITS).view(np.float32)[()]
    return np.float32(value)


def ordered_total_key(value: np.float32) -> int:
    bits = int(np.asarray(value, dtype=np.float32).view(np.uint32).item())
    signed_key_bits = bits ^ (0x7FFFFFFF if bits & 0x80000000 else 0)
    return signed_key_bits ^ 0x80000000


def model_bytes(
    version: int,
    storage: int = TensorProto.FLOAT,
    *,
    query_sequence: int = 2,
    key_sequence: int = 4,
    heads: int = 1,
    head_dim: int = 2,
    top_k: int = 2,
    scale: float = 1.0,
    weights_scale: float = 1.0,
) -> bytes:
    query = helper.make_tensor_value_info(
        "query", storage, [1, query_sequence, heads, head_dim]
    )
    key = helper.make_tensor_value_info("key", storage, [1, key_sequence, head_dim])
    weights = helper.make_tensor_value_info(
        "weights", storage, [1, query_sequence, heads]
    )
    bias = helper.make_tensor_value_info(
        "attention_bias", TensorProto.FLOAT, [1, 1, query_sequence, key_sequence]
    )
    output = helper.make_tensor_value_info(
        "selected_indices", TensorProto.INT64, [1, 1, query_sequence, top_k]
    )
    node = helper.make_node(
        "DsaIndexSelect",
        ["query", "key", "weights", "attention_bias"],
        ["selected_indices"],
        domain="pkg.nxrt",
        top_k=top_k,
        scale=scale,
        weights_scale=weights_scale,
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
                candidates.append(
                    (canonicalize_score(np.float32(weighted + bias_value)), t)
                )
            selected = sorted(
                candidates,
                key=lambda item: (-ordered_total_key(item[0]), item[1]),
            )[:top_k]
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
    executions = bind_u64(plugin, "nxrt_cuda_dsa_executions")
    score_launches = bind_u64(plugin, "nxrt_cuda_dsa_score_launches")
    selection_launches = bind_u64(plugin, "nxrt_cuda_dsa_selection_launches")
    last_score_grid_x = bind_u64(plugin, "nxrt_cuda_dsa_last_score_grid_x")
    last_selection_grid_x = bind_u64(
        plugin, "nxrt_cuda_dsa_last_selection_grid_x"
    )
    reset_launches = plugin.nxrt_cuda_reset_dsa_launch_stats
    reset_launches.argtypes = []
    reset_launches.restype = None
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
    reset_launches()
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
    assert session.get_outputs()[0].shape == [1, 1, 2, 2]

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
    assert executions() == score_launches() == selection_launches() == 2
    assert last_score_grid_x() == 1
    assert last_selection_grid_x() == 2
    first_ptr = workspace_last_ptr()
    assert first_ptr != 0

    for _ in range(3):
        (output,) = session.run(None, feeds)
        np.testing.assert_array_equal(output, expected)
        assert workspace_last_ptr() == first_ptr
        assert workspace_allocations() == 1
        assert workspace_live_bytes() == 512
    assert executions() == score_launches() == selection_launches() == 5

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

    for storage, dtype in [
        (TensorProto.FLOAT, np.float32),
        (TensorProto.FLOAT16, np.float16),
    ]:
        large = np.finfo(dtype).max
        with np.errstate(over="ignore", invalid="ignore"):
            dot = np.float32(np.float32(large) * np.float32(large))
            scored = np.maximum(
                np.float32(np.float32(np.finfo(np.float32).max) * dot),
                np.float32(0),
            )
            raw_score = np.float32(scored * np.float32(0))
        assert np.isnan(raw_score), f"{dtype} overflow regression must produce NaN"

        overflow_options = ort.SessionOptions()
        overflow_options.add_session_config_entry(
            "session.disable_cpu_ep_fallback", "1"
        )
        overflow_options.add_provider_for_devices([devices[0]], {})
        overflow_session = ort.InferenceSession(
            model_bytes(
                1,
                storage,
                query_sequence=1,
                key_sequence=3,
                heads=1,
                head_dim=1,
                top_k=2,
                scale=float(np.finfo(np.float32).max),
            ),
            sess_options=overflow_options,
        )
        overflow_feeds = {
            "query": np.array([[[[large]]]], dtype=dtype),
            "key": np.array([[[large], [0.0], [large]]], dtype=dtype),
            "weights": np.array([[[0.0]]], dtype=dtype),
            "attention_bias": np.zeros((1, 1, 1, 3), dtype=np.float32),
        }
        with np.errstate(over="ignore", invalid="ignore"):
            overflow_expected = cpu_reference(
                overflow_feeds,
                top_k=2,
                scale=float(np.finfo(np.float32).max),
                weights_scale=1.0,
            )
        np.testing.assert_array_equal(
            overflow_expected, np.array([[[[0, 2]]]], dtype=np.int64)
        )
        (overflow_output,) = overflow_session.run(None, overflow_feeds)
        np.testing.assert_array_equal(overflow_output, overflow_expected)
        del overflow_output, overflow_session, overflow_options
        gc.collect()

    assert workspace_live_bytes() == 0
    assert workspace_allocations() == workspace_releases() == 4
    assert executions() == score_launches() == selection_launches() == 8

    real_options = ort.SessionOptions()
    real_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    real_options.add_provider_for_devices([devices[0]], {})
    real_session = ort.InferenceSession(
        model_bytes(
            1,
            TensorProto.FLOAT16,
            query_sequence=1,
            key_sequence=2048,
            heads=32,
            head_dim=128,
            top_k=2048,
            scale=float(np.float32(128.0**-0.5)),
            weights_scale=float(np.float32(32.0**-0.5)),
        ),
        sess_options=real_options,
    )
    real_feeds = {
        "query": np.zeros((1, 1, 32, 128), dtype=np.float16),
        "key": np.zeros((1, 2048, 128), dtype=np.float16),
        "weights": np.zeros((1, 1, 32), dtype=np.float16),
        "attention_bias": np.zeros((1, 1, 1, 2048), dtype=np.float32),
    }
    real_expected = np.arange(2048, dtype=np.int64).reshape(1, 1, 1, 2048)
    before_real_executions = executions()
    set_capture_replays(3)
    (real_output,) = real_session.run(None, real_feeds)
    np.testing.assert_array_equal(real_output, real_expected)
    assert capture_count() == 1
    assert captured_replays() == 3
    assert capture_error() == 0
    assert executions() - before_real_executions == 2
    assert executions() == score_launches() == selection_launches()
    assert last_score_grid_x() == 8
    assert last_selection_grid_x() == 1
    assert workspace_live_bytes() == 10240
    real_ptr = workspace_last_ptr()
    assert real_ptr != 0
    del real_output, real_session, real_options
    gc.collect()
    assert workspace_live_bytes() == 0
    assert workspace_allocations() == workspace_releases() == 5

    bf16_options = ort.SessionOptions()
    bf16_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    bf16_options.add_provider_for_devices([devices[0]], {})
    bf16_session = ort.InferenceSession(
        model_bytes(1, TensorProto.BFLOAT16), sess_options=bf16_options
    )
    del bf16_session, bf16_options
    gc.collect()

    newer_import_options = ort.SessionOptions()
    newer_import_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    newer_import_options.add_provider_for_devices([devices[0]], {})
    compiled_before_newer_import = compiled_nodes()
    newer_import_session = ort.InferenceSession(
        model_bytes(2), sess_options=newer_import_options
    )
    assert newer_import_session.get_outputs()[0].shape == [1, 1, 2, 2]
    (newer_import_output,) = newer_import_session.run(None, feeds)
    np.testing.assert_array_equal(newer_import_output, expected)
    assert compiled_nodes() == compiled_before_newer_import + 1
    del newer_import_output, newer_import_session, newer_import_options
    gc.collect()
    assert workspace_live_bytes() == 0
    assert workspace_allocations() == workspace_releases() == 6

    print(
        "PASS: real CUDA-plugin session ran f32/f16 v1 with canonical overflow-NaN "
        "CPU parity, claimed bf16, captured/replayed real H=32 D=128 T=top_k=2048 "
        "with 8 score CTAs + 1 selection CTA and stable workspace, released all "
        "workspace at teardown, and resolved an opset-2 import to v1 semantics"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
