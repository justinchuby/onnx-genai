"""Granularity trade-curve analysis for sub-granule MoE VMM packing (#1295 follow-up).

Reuses the EXACT probe/run from scripts/moe_router_skew.py (same model, same 3
prompts x 64 greedy decode tokens) but captures the per-step selected-expert sets
so we can compute, for a given static packing of experts into commit-unit chunks:

  - granule waste (committed physical / useful expert bytes), static;
  - selectivity: experts that must be resident per step (hot-chunk union), vs the
    ideal top-8 that are actually used;
  - resident physical MiB per step per layer = the decisive objective; it folds in
    BOTH granule waste AND selectivity loss.

Two packings are compared:
  - index-order (arbitrary): experts 0..31 chunked by E;
  - hotness-sorted (static): experts sorted by aggregate decode selection count
    desc, then chunked by E, so hot experts cluster into coherent-residency chunks.

Model / method identical to moe_router_skew.py: granite-3.0-1b-a400m-instruct f16
Mobius, 32 experts, top-8, 24 layers, real trained IBM router, CPU EP, batch 1,
greedy. Reference baseline = uniform routing (top_k/num_experts = 0.25).
Expert byte size = granite int4 ~= 0.75 MiB; device VMM granule = 2 MiB.
"""
import json
import math
import numpy as np

import moe_router_skew as R

EXPERT_MIB = 0.75          # granite int4 expert (measured, per-expert-paging-churn doc)
GRANULE_MIB = 2.0          # CUDA device VMM granule (#776, hard floor)


def granule_round(mib):
    return math.ceil(mib / GRANULE_MIB) * GRANULE_MIB


def capture_per_step_sets():
    topk_names = R.patch_outputs()
    from tokenizers import Tokenizer
    import onnxruntime as ort
    tok = Tokenizer.from_file(R.MODEL_DIR + r"\tokenizer.json")
    so = ort.SessionOptions()
    sess = ort.InferenceSession(R.PATCHED, so, providers=["CPUExecutionProvider"])
    prompts = {
        "english_prose": "The history of the Roman Empire is a long and complex subject that spans many centuries of political, military, and cultural change across the Mediterranean world.",
        "code": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]",
        "math": "To solve the quadratic equation, we first compute the discriminant b squared minus four a c, then take the square root and divide.",
    }
    all_steps = []           # list over steps of [NUM_LAYERS, TOP_K] arrays
    all_decode = np.zeros((R.NUM_LAYERS, R.NUM_EXPERTS), dtype=np.int64)
    per_prompt_steps = {}
    for pname, ptext in prompts.items():
        pids = tok.encode(ptext).ids
        _pre, dec, _nd, steps_sel = R.run(sess, topk_names, pids, max_new=64, provider_note="cpu")
        all_decode += dec
        per_prompt_steps[pname] = steps_sel
        all_steps.extend(steps_sel)
    return all_steps, all_decode, per_prompt_steps


def eval_packing(steps, agg_counts, experts_per_chunk, order):
    """steps: list of [L,TOP_K]; order: 'index' or 'hotness' (per-layer static).
    Returns dict of averaged metrics over all (layer, step)."""
    L = R.NUM_LAYERS
    N = R.NUM_EXPERTS
    E = experts_per_chunk
    # per-layer expert -> chunk id
    chunk_of = np.zeros((L, N), dtype=np.int64)
    chunk_len = {}       # (layer, chunk) -> num experts
    for layer in range(L):
        if order == "index":
            ranked = list(range(N))
        elif order == "hotness":
            ranked = list(np.argsort(-agg_counts[layer]))   # hottest first
        else:
            raise ValueError(order)
        for pos, e in enumerate(ranked):
            c = pos // E
            chunk_of[layer, e] = c
            chunk_len[(layer, c)] = chunk_len.get((layer, c), 0) + 1
    # static waste: committed (granule-rounded per chunk) / useful
    committed = 0.0
    useful = 0.0
    for (layer, c), ln in chunk_len.items():
        committed += granule_round(ln * EXPERT_MIB)
        useful += ln * EXPERT_MIB
    waste_factor = committed / useful
    chunk_bytes = {k: granule_round(v * EXPERT_MIB) for k, v in chunk_len.items()}

    # per-step selectivity
    resident_experts = []     # experts riding along in hot chunks, per (layer,step)
    resident_mib = []         # committed physical of hot chunks, per (layer,step)
    hot_chunks_per = []
    # track always-on chunks (hot in 100% of steps) per layer
    steps_arr = np.stack(steps)               # [S, L, TOP_K]
    S = steps_arr.shape[0]
    chunk_hot_count = {}                       # (layer,chunk)->#steps hot
    for s in range(S):
        for layer in range(L):
            sel = steps_arr[s, layer]          # 8 expert ids
            hot = set(int(chunk_of[layer, e]) for e in sel)
            hot_chunks_per.append(len(hot))
            re = sum(chunk_len[(layer, c)] for c in hot)
            rm = sum(chunk_bytes[(layer, c)] for c in hot)
            resident_experts.append(re)
            resident_mib.append(rm)
            for c in hot:
                chunk_hot_count[(layer, c)] = chunk_hot_count.get((layer, c), 0) + 1
    always_on = sum(1 for v in chunk_hot_count.values() if v == S)
    total_chunks = len(chunk_len)
    # transferred bytes/step (streaming, no cache): only real expert content in hot
    # chunks is H2D-copied (granule padding is mapped, not copied). ideal = 8*0.75.
    avg_resident_experts = float(np.mean(resident_experts))
    transferred_mib = avg_resident_experts * EXPERT_MIB
    avg_resident_mib = float(np.mean(resident_mib))
    return {
        "E": E,
        "order": order,
        "chunk_useful_mib": E * EXPERT_MIB,
        "chunk_committed_mib": granule_round(E * EXPERT_MIB),
        "waste_factor": waste_factor,
        "avg_hot_chunks": float(np.mean(hot_chunks_per)),
        "avg_resident_experts": avg_resident_experts,   # ideal = 8
        "selectivity_ratio": avg_resident_experts / R.TOP_K,
        "avg_resident_mib_per_layer_step": avg_resident_mib,
        "transferred_mib_per_layer_step": transferred_mib,
        "waste_x_transferred": waste_factor * transferred_mib,
        "always_on_chunks": always_on,
        "total_chunks_per_layer": total_chunks / L,
    }


def main():
    steps, agg, per_prompt = capture_per_step_sets()
    S = len(steps)
    print(f"# captured {S} decode steps x {R.NUM_LAYERS} layers, top-{R.TOP_K} of {R.NUM_EXPERTS}")
    print(f"# expert={EXPERT_MIB} MiB (granite int4), granule={GRANULE_MIB} MiB")
    print(f"# ideal resident experts/step/layer = {R.TOP_K} (the top-8 actually used)")
    print(f"# useful bytes for top-8 = {R.TOP_K*EXPERT_MIB:.2f} MiB/layer/step\n")

    Es = [1, 2, 3, 4, 6, 8, 11, 16, 32]
    hdr = ("E  chunkMiB(use/commit)  waste   residentMiB   transferMiB   waste*transfer   sel%   alwaysOn")
    for order in ("index", "hotness"):
        print(f"## packing = {order}")
        print(hdr)
        rows = []
        for E in Es:
            r = eval_packing(steps, agg, E, order)
            rows.append(r)
            print(f"{r['E']:<2d} {r['chunk_useful_mib']:4.2f}/{r['chunk_committed_mib']:4.1f}"
                  f"            {r['waste_factor']:4.2f}x  {r['avg_resident_mib_per_layer_step']:7.2f}     "
                  f"{r['transferred_mib_per_layer_step']:7.2f}      {r['waste_x_transferred']:7.2f}       "
                  f"{r['selectivity_ratio']*100:5.1f}%   {r['always_on_chunks']}")
        best_res = min(rows, key=lambda x: x["avg_resident_mib_per_layer_step"])
        best_prod = min(rows, key=lambda x: x["waste_x_transferred"])
        print(f"-> min RESIDENT physical at E={best_res['E']} "
              f"({best_res['chunk_committed_mib']:.0f} MiB chunk): "
              f"{best_res['avg_resident_mib_per_layer_step']:.2f} MiB/layer/step")
        print(f"-> min WASTE*TRANSFER at E={best_prod['E']}: {best_prod['waste_x_transferred']:.2f}\n")

    # Cross-layer packing sanity: every layer is active every step, so a chunk that
    # spans layers is hot every step -> selectivity 0. Quantify.
    print("## cross-layer note")
    print("every decode step selects experts in ALL 24 layers, so any chunk spanning")
    print("layers is hot in 100% of steps: cross-layer packing => selectivity ratio =")
    print("total_experts_in_chunk / 8, i.e. strictly worse than within-layer. A 64 MiB")
    print("chunk (~85 experts) MUST span >2 layers on granite (24 MiB/layer bank), so it")
    print("is resident whenever ANY of its ~85 experts is hot = every step.")

    # prompt-stability of the hotness ranking (Q3): does static packing decay?
    print("\n## routing-skew stability across prompts (Q3)")
    rankings = {}
    for pname, psteps in per_prompt.items():
        c = np.zeros((R.NUM_LAYERS, R.NUM_EXPERTS), dtype=np.int64)
        for st in psteps:
            for L in range(R.NUM_LAYERS):
                for e in st[L]:
                    c[L, int(e)] += 1
        rankings[pname] = c
    # For a hotness packing decided on ALL prompts, how well does each prompt's own
    # hot set stay inside the globally-hot chunks? Measure per-prompt resident MiB
    # under the GLOBAL hotness packing vs each prompt's OWN optimal packing.
    for E in (4, 8):
        print(f"  E={E}: resident MiB/layer/step under GLOBAL hotness packing, per prompt:")
        for pname, psteps in per_prompt.items():
            r = eval_packing(psteps, agg, E, "hotness")          # global agg ranking
            r_own = eval_packing(psteps, rankings[pname], E, "hotness")  # own ranking
            print(f"    {pname:14s} global-pack={r['avg_resident_mib_per_layer_step']:6.2f}  "
                  f"own-pack={r_own['avg_resident_mib_per_layer_step']:6.2f}  "
                  f"decay={100*(r['avg_resident_mib_per_layer_step']/r_own['avg_resident_mib_per_layer_step']-1):5.1f}%")


if __name__ == "__main__":
    main()
