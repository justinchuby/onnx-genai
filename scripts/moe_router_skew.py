"""Measure trained-router expert-selection skew for a dense (loop-over-experts)
Mobius MoE export.

Model: ibm-granite/granite-3.0-1b-a400m-instruct, built f16 dense via Mobius.
32 experts, top-8 routing, 24 layers. Real trained IBM Granite router.

We add every per-layer `TopK ..._1` (selected_experts) tensor as a graph output,
then greedily decode real prompts through onnxruntime, recording which experts
each layer selects at every decode step. `reads_per_step` for an expert = number
of steps it was selected / number of steps. Uniform baseline = top_k/num_experts.
"""
import json
import sys
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


def patch_outputs():
    m = onnx.load(SRC, load_external_data=False)
    g = m.graph
    existing = {o.name for o in g.output}
    topk_sel = []
    for n in g.node:
        if n.op_type == "TopK":
            sel = n.output[1]  # indices
            topk_sel.append(sel)
    topk_sel.sort(key=lambda s: int(s.split("TopK_")[1].split("_")[0]))
    for name in topk_sel:
        if name not in existing:
            vi = g.output.add()
            vi.name = name
            vi.type.tensor_type.elem_type = onnx.TensorProto.INT64
    onnx.save(m, PATCHED)
    return topk_sel


def build_feeds(input_ids, past, tok_len_prev):
    batch = 1
    seq = input_ids.shape[1]
    feeds = {
        "input_ids": input_ids.astype(np.int64),
        "attention_mask": np.ones((batch, tok_len_prev + seq), dtype=np.int64),
        "position_ids": np.arange(tok_len_prev, tok_len_prev + seq, dtype=np.int64)[None, :],
    }
    for i in range(NUM_LAYERS):
        if past is None:
            feeds[f"past_key_values.{i}.key"] = np.zeros((batch, 8, 0, 64), dtype=np.float16)
            feeds[f"past_key_values.{i}.value"] = np.zeros((batch, 8, 0, 64), dtype=np.float16)
        else:
            feeds[f"past_key_values.{i}.key"] = past[i][0]
            feeds[f"past_key_values.{i}.value"] = past[i][1]
    return feeds


def run(sess, topk_names, prompt_ids, max_new, provider_note):
    out_names = ["logits"] + [f"present.{i}.key" for i in range(NUM_LAYERS)] \
        + [f"present.{i}.value" for i in range(NUM_LAYERS)] + topk_names
    # per-layer per-expert selection counts, separated into prefill vs decode
    decode_counts = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64)
    prefill_counts = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64)
    # per-step (decode) selected sets, for persistence analysis
    decode_steps_sel = []  # list over steps: array [NUM_LAYERS, TOP_K]

    ids = np.array(prompt_ids, dtype=np.int64)[None, :]
    past = None
    tok_len_prev = 0
    n_decode = 0
    for step in range(max_new + 1):
        feeds = build_feeds(ids, past, tok_len_prev)
        outs = sess.run(out_names, feeds)
        logits = outs[0]
        presents_k = outs[1:1 + NUM_LAYERS]
        presents_v = outs[1 + NUM_LAYERS:1 + 2 * NUM_LAYERS]
        sel = outs[1 + 2 * NUM_LAYERS:]  # 24 arrays [1, seq, 8]
        past = [(presents_k[i], presents_v[i]) for i in range(NUM_LAYERS)]
        seq = ids.shape[1]
        tok_len_prev += seq
        is_prefill = (step == 0)
        for L in range(NUM_LAYERS):
            s = sel[L][0]  # [seq, 8]
            if is_prefill:
                for pos in range(s.shape[0]):
                    for e in s[pos]:
                        prefill_counts[L, int(e)] += 1
            else:
                row = s[-1]  # [8]
                for e in row:
                    decode_counts[L, int(e)] += 1
        if not is_prefill:
            n_decode += 1
            decode_steps_sel.append(np.stack([sel[L][0, -1] for L in range(NUM_LAYERS)]))
        # greedy next token from last position
        nxt = int(np.argmax(logits[0, -1]))
        ids = np.array([[nxt]], dtype=np.int64)
        if step >= max_new:
            break
    return prefill_counts, decode_counts, n_decode, decode_steps_sel


def summarize(name, counts, denom_steps, note):
    # counts: [NUM_LAYERS, NUM_EXPERTS], denom_steps = number of steps
    print(f"\n===== {name} ({note}) — {denom_steps} steps =====")
    # aggregate across layers: each layer picks TOP_K per step
    per_layer_rps = counts / max(denom_steps, 1)  # reads_per_step per (layer,expert)
    flat = per_layer_rps.flatten()
    uniform = TOP_K / NUM_EXPERTS
    print(f"uniform baseline reads_per_step = {uniform:.4f} (top_k {TOP_K}/{NUM_EXPERTS} experts)")
    print(f"per-(layer,expert) reads_per_step: min={flat.min():.4f} "
          f"median={np.median(flat):.4f} mean={flat.mean():.4f} max={flat.max():.4f}")
    # top-k share of read volume, per layer averaged
    shares = []
    gini_list = []
    for L in range(NUM_LAYERS):
        c = counts[L].astype(float)
        tot = c.sum()
        if tot == 0:
            continue
        cs = np.sort(c)[::-1]
        top8_share = cs[:TOP_K].sum() / tot
        shares.append(top8_share)
        # gini
        x = np.sort(c)
        nx = len(x)
        cum = np.cumsum(x)
        gini = (nx + 1 - 2 * (cum.sum() / cum[-1])) / nx if cum[-1] > 0 else 0.0
        gini_list.append(gini)
    print(f"top-{TOP_K} experts' share of layer read volume: "
          f"mean={np.mean(shares):.3f} min={np.min(shares):.3f} max={np.max(shares):.3f} "
          f"(uniform would be {TOP_K/NUM_EXPERTS:.3f})")
    print(f"per-layer Gini of expert read counts: mean={np.mean(gini_list):.3f} "
          f"max={np.max(gini_list):.3f} (0=uniform, 1=all-on-one)")
    # example: layer 12 distribution
    L = 12
    c = counts[L]
    order = np.argsort(c)[::-1]
    print(f"layer {L} top experts (id:count): " +
          ", ".join(f"{int(e)}:{int(c[e])}" for e in order[:10]))
    print(f"layer {L} experts never selected: {int((c==0).sum())}/{NUM_EXPERTS}")


def main():
    topk_names = patch_outputs()
    print(f"patched {len(topk_names)} TopK selected-expert outputs")
    tok = Tokenizer.from_file(MODEL_DIR + r"\tokenizer.json")
    providers = ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    sess = ort.InferenceSession(PATCHED, so, providers=providers)
    prompts = {
        "english_prose": "The history of the Roman Empire is a long and complex subject that spans many centuries of political, military, and cultural change across the Mediterranean world.",
        "code": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]",
        "math": "To solve the quadratic equation, we first compute the discriminant b squared minus four a c, then take the square root and divide.",
    }
    all_decode = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64)
    total_decode_steps = 0
    for pname, ptext in prompts.items():
        enc = tok.encode(ptext)
        pids = enc.ids
        pre, dec, nd, steps_sel = run(sess, topk_names, pids, max_new=64, provider_note="cpu")
        summarize(f"PROMPT[{pname}] PREFILL", pre, len(pids), f"cpu, prompt_len={len(pids)}")
        summarize(f"PROMPT[{pname}] DECODE", dec, nd, "cpu greedy 64 tokens")
        all_decode += dec
        total_decode_steps += nd
    summarize("AGGREGATE DECODE (all prompts)", all_decode, total_decode_steps, "cpu")
    # persistence: does the same expert set recur? measure per-layer, fraction of
    # decode steps whose top expert equals the globally most-frequent expert.
    print("\n===== PERSISTENCE (aggregate decode, all 24 layers) =====")
    most_rps = []
    for L in range(NUM_LAYERS):
        c = all_decode[L]
        most = int(np.argmax(c))
        rps_most = c[most] / max(total_decode_steps, 1)
        most_rps.append(rps_most)
        print(f"layer {L:2d}: hottest expert {most:2d} chosen in "
              f"{rps_most:6.2%} of decode steps (uniform {TOP_K/NUM_EXPERTS:.2%})")
    print(f"hottest-expert reads_per_step across layers: "
          f"min={min(most_rps):.2%} median={np.median(most_rps):.2%} max={max(most_rps):.2%}")
    with open("moe_router_skew_counts.json", "w") as f:
        json.dump({"decode_counts": all_decode.tolist(),
                   "total_decode_steps": int(total_decode_steps),
                   "num_experts": NUM_EXPERTS, "top_k": TOP_K,
                   "num_layers": NUM_LAYERS}, f)
    print("wrote moe_router_skew_counts.json")


if __name__ == "__main__":
    main()
