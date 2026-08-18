"""Dump the per-step, per-layer expert-selection trace for a trained-router MoE.

Companion to `moe_router_skew.py`, which records aggregate selection *counts*.
A cache simulator needs the *sequence* of selected experts, not just the totals,
so this script writes the full per-step trace to `scripts/moe_expert_trace.json`.

Model: ibm-granite/granite-3.0-1b-a400m-instruct, built f16 dense via Mobius.
32 experts, top-8 routing, 24 layers. Real trained IBM Granite router.

Selection is `TopK(MatMul(hidden, gate))`, an integer top-k that is dtype- and
EP-independent, so the CPU EP picks are identical to CUDA's. Only *which*
experts are chosen is captured here; timing/bandwidth is deliberately not.

Output JSON schema:
    {
      "model": "...", "num_layers": 24, "num_experts": 32, "top_k": 8,
      "prompts": {
        "<name>": {
          "prompt_len": <int>,
          "prefill": [ [ [e0..e7] x 24 layers ] x prompt_len positions ],
          "decode":  [ [ [e0..e7] x 24 layers ] x 64 steps ]
        }, ...
      }
    }
Each innermost list is the set of TOP_K expert ids a layer selected at that
position/step. Prefill entries are per prompt token; decode entries per
generated token.
"""
import json

import numpy as np
import onnx
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = r"C:\Users\justinchu\dev\models\granite-1b-a400m-f16-mobius"
SRC = MODEL_DIR + r"\model.onnx"
PATCHED = MODEL_DIR + r"\model_moe_probe.onnx"
NUM_LAYERS = 24
NUM_EXPERTS = 32
TOP_K = 8
MAX_NEW = 64

PROMPTS = {
    "english_prose": "The history of the Roman Empire is a long and complex subject that spans many centuries of political, military, and cultural change across the Mediterranean world.",
    "code": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]",
    "math": "To solve the quadratic equation, we first compute the discriminant b squared minus four a c, then take the square root and divide.",
}


def patch_outputs():
    m = onnx.load(SRC, load_external_data=False)
    g = m.graph
    existing = {o.name for o in g.output}
    topk_sel = [n.output[1] for n in g.node if n.op_type == "TopK"]
    topk_sel.sort(key=lambda s: int(s.split("TopK_")[1].split("_")[0]))
    for name in topk_sel:
        if name not in existing:
            vi = g.output.add()
            vi.name = name
            vi.type.tensor_type.elem_type = onnx.TensorProto.INT64
    onnx.save(m, PATCHED)
    return topk_sel


def build_feeds(input_ids, past, tok_len_prev):
    seq = input_ids.shape[1]
    feeds = {
        "input_ids": input_ids.astype(np.int64),
        "attention_mask": np.ones((1, tok_len_prev + seq), dtype=np.int64),
        "position_ids": np.arange(tok_len_prev, tok_len_prev + seq, dtype=np.int64)[None, :],
    }
    for i in range(NUM_LAYERS):
        if past is None:
            feeds[f"past_key_values.{i}.key"] = np.zeros((1, 8, 0, 64), dtype=np.float16)
            feeds[f"past_key_values.{i}.value"] = np.zeros((1, 8, 0, 64), dtype=np.float16)
        else:
            feeds[f"past_key_values.{i}.key"] = past[i][0]
            feeds[f"past_key_values.{i}.value"] = past[i][1]
    return feeds


def run(sess, topk_names, prompt_ids):
    out_names = (
        ["logits"]
        + [f"present.{i}.key" for i in range(NUM_LAYERS)]
        + [f"present.{i}.value" for i in range(NUM_LAYERS)]
        + topk_names
    )
    prefill_sel = []  # [pos][layer] = [8]
    decode_sel = []  # [step][layer] = [8]
    ids = np.array(prompt_ids, dtype=np.int64)[None, :]
    past = None
    tok_len_prev = 0
    for step in range(MAX_NEW + 1):
        feeds = build_feeds(ids, past, tok_len_prev)
        outs = sess.run(out_names, feeds)
        logits = outs[0]
        presents_k = outs[1:1 + NUM_LAYERS]
        presents_v = outs[1 + NUM_LAYERS:1 + 2 * NUM_LAYERS]
        sel = outs[1 + 2 * NUM_LAYERS:]  # 24 arrays [1, seq, 8]
        past = [(presents_k[i], presents_v[i]) for i in range(NUM_LAYERS)]
        tok_len_prev += ids.shape[1]
        if step == 0:
            seq_len = sel[0].shape[1]
            for pos in range(seq_len):
                prefill_sel.append(
                    [[int(e) for e in sel[L][0, pos]] for L in range(NUM_LAYERS)]
                )
        else:
            decode_sel.append(
                [[int(e) for e in sel[L][0, -1]] for L in range(NUM_LAYERS)]
            )
        nxt = int(np.argmax(logits[0, -1]))
        ids = np.array([[nxt]], dtype=np.int64)
        if step >= MAX_NEW:
            break
    return prefill_sel, decode_sel


def main():
    topk_names = patch_outputs()
    print(f"patched {len(topk_names)} TopK selected-expert outputs")
    tok = Tokenizer.from_file(MODEL_DIR + r"\tokenizer.json")
    sess = ort.InferenceSession(PATCHED, ort.SessionOptions(), providers=["CPUExecutionProvider"])
    out = {
        "model": "granite-3.0-1b-a400m-instruct (f16 dense Mobius)",
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "prompts": {},
    }
    for name, text in PROMPTS.items():
        ids = tok.encode(text).ids
        prefill_sel, decode_sel = run(sess, topk_names, ids)
        out["prompts"][name] = {
            "prompt_len": len(ids),
            "prefill": prefill_sel,
            "decode": decode_sel,
        }
        print(f"{name}: prefill={len(prefill_sel)} positions, decode={len(decode_sel)} steps")
    with open("scripts/moe_expert_trace.json", "w") as f:
        json.dump(out, f)
    print("wrote scripts/moe_expert_trace.json")


if __name__ == "__main__":
    main()
