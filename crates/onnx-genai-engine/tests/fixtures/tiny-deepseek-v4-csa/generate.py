"""Export a tiny shape-faithful alternating ratio-4 CSA / ratio-128 HCA
DeepSeek-V4 graph with the merged native-CSA exporter (Mobius #593), synthetic
small weights only. For the onnx-genai native-decode CSA/HCA E2E integration
proof (fixture: tests/fixtures/tiny-deepseek-v4-csa/).

Reproduce from a Mobius checkout that has the merged CSA exporter:

    PYTHONPATH=src python generate.py --out /some/scratch/model.onnx

then copy `<out>/model.onnx`, `<out>/model.onnx.data`, and `<out>_io.json`
(-> io_report.json) next to this script. The geometry is FROZEN to the official
DeepSeek-V4 per-head CSA dims (only batch/seq/heads/index-heads/topk/layers/
hidden/vocab are free); see `tiny_alternating_config` below.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import onnx_ir as ir
import torch

from mobius._builder import build_from_module
from mobius._testing import make_config
from mobius.models.deepseek_v4 import DeepSeekV4CausalLMModel


def tiny_alternating_config(**overrides):
    """Layer 0 = ratio-4 (CSA), layer 1 = ratio-128 (HCA); both native.

    The native ``CompressedSparseAttention`` v1 op is FROZEN to the official
    DeepSeek-V4 per-head CSA geometry: the CPU/CUDA kernel hard-requires
    ``head_dim=512``, ``qk_rope_head_dim=64`` (fp8 record width 583),
    ``index_head_dim=128`` (fp4 index width 68), compressor latent
    ``head_dim``/``2*head_dim`` and index latent ``2*index_head_dim``. Only the
    batch/sequence/``num_heads``/``index_n_heads``/``index_topk`` extents and the
    surrounding model size (layers, hidden, vocab, MoE) are free, so a
    *shape-faithful* tiny fixture keeps the official per-head dims and shrinks
    everything else.
    """
    values = dict(
        model_type="deepseek_v4",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=512,
        q_lora_rank=8,
        qk_rope_head_dim=64,
        o_groups=2,
        o_lora_rank=8,
        num_local_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        num_hash_layers=1,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        swiglu_limit=10.0,
        rope_interleave=True,
        index_n_heads=2,
        index_head_dim=128,
        index_topk=4,
        compress_ratios=[4, 128],
        native_csa=True,
    )
    values.update(overrides)
    return make_config(**values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    config = tiny_alternating_config()
    module = DeepSeekV4CausalLMModel(config)
    package = build_from_module(module, config, task="deepseek-v4")
    model = package["model"]
    graph = model.graph

    # Materialize small synthetic weights directly onto every initializer that
    # the build left without a constant value (build_from_module references the
    # torch params lazily). Random small values are enough to prove attention/
    # state integration; quantized real weights are explicitly not needed here.
    _fill_synthetic_initializers(graph)

    # Report the CSA nodes + full IO so the onnx-genai side can build the
    # DecoderAbi csa_state_groups against the real typed ports.
    csa_nodes = [
        n for n in graph if n.op_type == "CompressedSparseAttention" and n.domain == "pkg.nxrt"
    ]
    report = {
        "csa_node_count": len(csa_nodes),
        "csa_nodes": [
            {
                "name": n.name,
                "domain": n.domain,
                "inputs": [v.name for v in n.inputs],
                "outputs": [v.name for v in n.outputs],
                "attributes": {
                    k: (v.value if not isinstance(v.value, (list, tuple)) else list(v.value))
                    for k, v in n.attributes.items()
                },
            }
            for n in csa_nodes
        ],
        "graph_inputs": [
            {"name": v.name, "dtype": str(v.dtype), "shape": _shape(v)} for v in graph.inputs
        ],
        "graph_outputs": [
            {"name": v.name, "dtype": str(v.dtype), "shape": _shape(v)} for v in graph.outputs
        ],
    }
    print(json.dumps(report, indent=2, default=str))
    with open(args.out.rstrip("/") + "_io.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    package.save(args.out, external_data="onnx", progress_bar=False, check_weights=False)
    print(f"\nSaved tiny alternating CSA/HCA package to {args.out}")


_IR_TO_NP = {
    ir.DataType.FLOAT: np.float32,
    ir.DataType.FLOAT16: np.float16,
    ir.DataType.DOUBLE: np.float64,
    ir.DataType.INT64: np.int64,
    ir.DataType.INT32: np.int32,
    ir.DataType.UINT8: np.uint8,
    ir.DataType.INT8: np.int8,
    ir.DataType.BOOL: np.bool_,
}


def _fill_synthetic_initializers(graph: ir.Graph) -> None:
    for init in graph.initializers.values():
        if init.const_value is not None:
            continue
        if init.shape is None or not init.shape.is_static():
            raise SystemExit(f"initializer {init.name!r} has non-static shape {init.shape}")
        dims = [int(d) for d in init.shape]
        dtype = init.dtype
        np_dtype = _IR_TO_NP.get(dtype, np.float32)
        if np.issubdtype(np_dtype, np.floating):
            arr = (np.random.randn(*dims) * 0.02).astype(np_dtype) if dims else np.array(
                0.02, dtype=np_dtype
            )
        elif np_dtype == np.bool_:
            arr = np.zeros(dims, dtype=np_dtype)
        else:
            # Integer tables (e.g. gate tid2eid) — small in-range indices.
            arr = np.zeros(dims, dtype=np_dtype)
        init.const_value = ir.tensor(arr, name=init.name)


def _shape(v: ir.Value):
    if v.shape is None:
        return None
    out = []
    for d in v.shape:
        out.append(d.value if isinstance(d, ir.SymbolicDim) else d)
    return out


if __name__ == "__main__":
    main()
