"""Composed MoE-offload floor: PCIe bytes/token -> time floor -> VRAM-for-tok/s.

Composes the two prior trace-driven studies into the single number a deployer
needs. It assumes the DRAM tier holds the whole expert bank (so avoidable SSD
traffic is zero -- see the tiering study), leaving DRAM->VRAM PCIe as the only
per-step expert-transfer cost, and applies the best *practical* residency policy
from the single-tier study -- STATIC HOT-PIN (top-B experts by measured global
frequency pinned in VRAM; the streamed remainder crosses PCIe on every use).
Oracle (Belady/MIN) is reported alongside as the unbeatable floor.

Outputs, as a function of VRAM budget (fraction of the expert bank):
  1. PCIe experts/token and bytes/token (@0.75 and @16 MiB experts).
  2. A per-token TIME floor at assumed PCIe bandwidths, and the tok/s ceiling
     from transfer ALONE (before any compute). Bandwidths are INFERRED.
  3. Inversion: minimum VRAM fraction to hit a target tok/s.
  4. The knee: VRAM fraction beyond which extra VRAM buys little.
  5. Batch interaction: the PCIe cost is per *step*; with a batch of W requests
     the batch's non-pinned expert union is loaded once and amortised over W
     tokens, so every figure is reported at a stated batch size W.

Model behind the trace: granite-3.0-1b-a400m-instruct, 32 experts, top-8, 24
layers, real IBM trained router; onnxruntime 1.27.0 CPU EP, batch 1. CPU-only
replay -- no GPU. Provenance hw: i7-13800H (14C/20T), RTX 4060 8 GB, WDDM.

CAVEATS (must be read with the numbers):
  * Bandwidth constants are ASSUMED, not measured on this box (labelled INFERRED).
  * bytes/token is monotone in experts/token only because granite's experts are
    uniform size; heterogeneous experts would need the byte figure directly.
  * Batching cuts *bytes*/token, but is NOT free in wall-clock on this stack: a
    sibling measurement finds M=1->2 costs ~5.4x/step for 2x work (~2.55->14 ms),
    a fixed batch-decode penalty with the GEMV excluded as cause. So "batching is
    free" holds in BYTES only; cross-referenced, do not read as wall-clock-free.
  * A trace-driven sim cannot give achieved wall-clock, the paging mechanism's
    real cost, or whether skew generalises from granite's 8-of-32 router to a
    DeepSeek-class 8-of-256 one (larger banks likely have MORE headroom; inferred).
"""
import bisect
import json
from collections import defaultdict

EXPERT_MIB = {"granite_int4_0p75": 0.75, "target_16": 16.0}
# INFERRED / assumed PCIe bandwidths (GB/s), not measured on this box:
PCIE_BW = {"pcie4_x8_13": 13.0, "pcie4_x16_26": 26.0, "pcie5_x16_55": 55.0}
VRAM_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.75, 0.90]
BATCH_SIZES = [1, 8]
TARGET_TOKS = [5, 10, 20, 40]


def build(trace):
    NL, NE = trace["num_layers"], trace["num_experts"]
    prompts = trace["prompts"]
    names = list(prompts.keys())
    # interleaved requests (concurrent streams) for batching
    per = {n: [] for n in names}
    for n in names:
        for step in prompts[n]["decode"]:
            per[n].append([L * NE + step[L][j] for L in range(NL) for j in range(len(step[L]))])
    nsteps = min(len(per[n]) for n in names)
    reqs = []
    for t in range(nsteps):
        for n in names:
            reqs.append(per[n][t])
    # global frequency for pin ranking + flat group list for oracle
    counts = defaultdict(int)
    for r in reqs:
        for k in r:
            counts[k] += 1
    return reqs, counts, NL * NE


def static_pin_pcie(reqs, pin_set, W):
    """Non-pinned experts loaded per token, batched FIFO in groups of W."""
    total = 0
    for i in range(0, len(reqs), W):
        batch = reqs[i:i + W]
        union = set()
        for r in batch:
            for k in r:
                if k not in pin_set:
                    union.add(k)
        total += len(union)
    return total / len(reqs)


def oracle_pcie(reqs, budget, topk):
    """Belady/MIN page-ins per token at the correct per-layer (8-key) access
    granularity -- the unbeatable floor. Reference string = each request's 192
    keys split into its 24 per-layer top-8 groups, in request order, over one
    global cache of `budget` slots (matches the single-tier sim)."""
    groups = []
    for r in reqs:
        for i in range(0, len(r), topk):
            groups.append(r[i:i + topk])
    nextuse = defaultdict(list)
    for g, grp in enumerate(groups):
        for k in grp:
            nextuse[k].append(g)
    resident = set()
    page_ins = 0
    for g, grp in enumerate(groups):
        for k in grp:
            if k not in resident:
                page_ins += 1
                resident.add(k)
        while len(resident) > budget:
            prot = set(grp)
            cand = [k for k in resident if k not in prot] or list(resident)

            def nxt(k):
                lst = nextuse[k]
                idx = bisect.bisect_right(lst, g)
                return lst[idx] if idx < len(lst) else float("inf")
            victim = max(cand, key=nxt)
            resident.discard(victim)
    return page_ins / len(reqs)  # per token (192 requests)


def main():
    with open("scripts/moe_expert_trace.json") as f:
        trace = json.load(f)
    reqs, counts, n_keys = build(trace)
    ntok = len(reqs)
    topk = trace["top_k"]
    order = sorted(counts, key=lambda k: counts[k], reverse=True)
    esz = EXPERT_MIB["target_16"]

    print(f"model: {trace['model']}  n_keys={n_keys}  tokens(steps)={ntok}  "
          f"expert=16 MiB (also 0.75 MiB int4)")
    print("Assumes DRAM holds the whole bank (SSD out of path); PCIe DRAM->VRAM is")
    print("the only expert-transfer cost. Policy = static hot-pin (practical).")
    print("PCIe bandwidths ASSUMED/INFERRED:", PCIE_BW)

    out = {"model": trace["model"], "n_keys": n_keys, "tokens": ntok,
           "expert_mib": EXPERT_MIB, "pcie_bw_gbps": PCIE_BW,
           "vram_ratios": VRAM_RATIOS, "batch_sizes": BATCH_SIZES,
           "note": "DRAM holds bank; PCIe-only; static hot-pin; bandwidths INFERRED",
           "curve": {}}

    for W in BATCH_SIZES:
        print(f"\n===== batch W={W} (bytes/token amortised over the batch) =====")
        print(" VRAM%  pin  | PCIe exp/tok | GB/tok@16 | ms/tok & tok/s @ 13 / 26 / 55 GB/s"
              " | oracle exp/tok(W=1)")
        rows = {}
        for r in VRAM_RATIOS:
            B = max(1, round(r * n_keys))
            pin = set(order[:B])
            ept = static_pin_pcie(reqs, pin, W)
            gb = ept * esz / 1024.0
            times = {}
            for bw_name, bw in PCIE_BW.items():
                ms = gb / bw * 1000
                times[bw_name] = {"ms_per_tok": ms, "tok_s": 1000.0 / ms if ms else 0}
            orc = oracle_pcie(reqs, B, topk) if W == 1 else None
            rows[f"{r}"] = {"vram_ratio": r, "pin_slots": B,
                            "pcie_exp_per_tok": ept,
                            "pcie_gb_per_tok_16": gb,
                            "pcie_mib_per_tok_16": ept * esz,
                            "pcie_mib_per_tok_0p75": ept * EXPERT_MIB["granite_int4_0p75"],
                            "times": times,
                            "oracle_exp_per_tok": orc}
            t13 = times["pcie4_x8_13"]; t26 = times["pcie4_x16_26"]; t55 = times["pcie5_x16_55"]
            orctxt = f"{orc:6.1f}" if orc is not None else "   -  "
            print(f" {r:4.0%} {B:4d}  | {ept:9.1f}    | {gb:6.2f}    | "
                  f"{t13['ms_per_tok']:5.0f}ms/{t13['tok_s']:4.1f}  "
                  f"{t26['ms_per_tok']:5.0f}ms/{t26['tok_s']:4.1f}  "
                  f"{t55['ms_per_tok']:5.0f}ms/{t55['tok_s']:4.1f} | {orctxt}")
        out["curve"][f"W{W}"] = rows

    # ---- knee: where marginal bytes/token per +5% VRAM flattens ----
    print("\n== KNEE (W=1, @16 MiB, per-step) ==")
    w1 = out["curve"]["W1"]
    prev = None
    for r in VRAM_RATIOS:
        cur = w1[f"{r}"]["pcie_mib_per_tok_16"]
        if prev is not None:
            dr = (r - prev_r)
            marg = (prev - cur) / (dr * 100)  # MiB/token saved per +1% VRAM
            print(f"  {prev_r:4.0%}->{r:4.0%}: {prev:6.0f}->{cur:6.0f} MiB/tok "
                  f"({marg:5.1f} MiB/tok saved per +1% VRAM)")
        prev, prev_r = cur, r

    # ---- inversion: min VRAM fraction to hit target tok/s ----
    print("\n== INVERSION: min VRAM % to hit target tok/s (transfer floor only) ==")
    inv = {}
    for W in BATCH_SIZES:
        rows = out["curve"][f"W{W}"]
        for bw_name, bw in PCIE_BW.items():
            for tgt in TARGET_TOKS:
                need = None
                for r in VRAM_RATIOS:
                    if rows[f"{r}"]["times"][bw_name]["tok_s"] >= tgt:
                        need = r
                        break
                inv[f"W{W}_{bw_name}_{tgt}tok"] = need
                s = f"{need:.0%}" if need is not None else ">90% (not reached in sweep)"
                print(f"  W={W:2d} @ {bw:4.0f} GB/s, target {tgt:3d} tok/s -> VRAM >= {s}")
    out["inversion"] = inv

    with open("scripts/moe_pcie_floor_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote scripts/moe_pcie_floor_results.json")


if __name__ == "__main__":
    main()
