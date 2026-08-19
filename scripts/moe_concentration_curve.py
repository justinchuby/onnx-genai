"""Concentration (Lorenz) curve of MoE expert traffic -- tests the Zipf premise.

An external analysis of MoE offload assumes strongly Zipfian routing, illustrated
as "top 32 of 256 experts (12.5% of the bank) carry ~80% of traffic". Our only
real measurement (granite-3.0-1b-a400m, 32 experts, top-8) earlier found top-8/32
(25% of the bank) carrying 45.4% of reads -- materially milder. Every tiering
decision (how big VRAM/DRAM must be to catch most traffic) depends on this curve,
so we report it from the trace rather than assuming it.

Reports, from `scripts/moe_expert_trace.json`:
  * GLOBAL curve over all (layer, expert) keys: cumulative traffic fraction
    carried by the top p% of keys, for p in {10, 12.5, 25, 50, 75, 90, 100}.
  * PER-LAYER curve: within each layer's 32 experts, the top-25% (top-8) traffic
    share -- reproduces the 45.4% figure and shows how layers 1-2 differ.
  * Gini coefficient (global and per-layer).

Model: granite-3.0-1b-a400m-instruct, 32 experts, top-8, 24 layers, real IBM
trained router; onnxruntime 1.27.0 CPU EP, batch 1.
Hardware for provenance only (no GPU used here): i7-13800H (14C/20T),
RTX 4060 8 GB, WDDM, driver 591.55, CUDA 13.1.
"""
import json

KEY_PCTS = [0.10, 0.125, 0.25, 0.50, 0.75, 0.90, 1.00]
ZIPF_ASSUMED = {0.125: 0.80}  # the external analysis's illustrative point


def cumulative_share(counts):
    """Sorted-desc cumulative traffic fraction at each cumulative-key fraction."""
    total = sum(counts)
    if total == 0:
        return {}
    s = sorted(counts, reverse=True)
    n = len(s)
    out = {}
    for p in KEY_PCTS:
        take = max(1, round(p * n))
        out[p] = sum(s[:take]) / total
    return out


def gini(counts):
    s = sorted(counts)
    n = len(s)
    tot = sum(s)
    if tot == 0 or n == 0:
        return 0.0
    cum = 0
    for i, x in enumerate(s, 1):
        cum += i * x
    return (2 * cum) / (n * tot) - (n + 1) / n


def key_counts(prompt_entries, NL, NE):
    """Global (layer,expert) counts summed across the given prompt entries."""
    c = [0] * (NL * NE)
    for e in prompt_entries:
        for step in e["decode"]:
            for L in range(NL):
                for expert in step[L]:
                    c[L * NE + expert] += 1
    return c


def main():
    with open("scripts/moe_expert_trace.json") as f:
        trace = json.load(f)
    NL, NE, TK = trace["num_layers"], trace["num_experts"], trace["top_k"]
    prompts = trace["prompts"]
    names = list(prompts.keys())

    print(f"model: {trace['model']}  layers={NL} experts={NE} top_k={TK} "
          f"n_keys={NL*NE}")
    print(f"uniform per-token working set = top_k/experts = {TK/NE:.1%} of the bank")

    out = {"model": trace["model"], "num_layers": NL, "num_experts": NE,
           "top_k": TK, "key_pcts": KEY_PCTS, "zipf_assumed": ZIPF_ASSUMED}

    # ---- GLOBAL curve (all prompts combined + per prompt) ----
    combined = key_counts([prompts[n] for n in names], NL, NE)
    accessed = sum(1 for x in combined if x > 0)
    print(f"\n== GLOBAL concentration over {NL*NE} (layer,expert) keys "
          f"({accessed} ever accessed) ==")
    print("  top p% of keys  ->  cumulative % of traffic   (assumed Zipf in [])")
    curve = cumulative_share(combined)
    for p in KEY_PCTS:
        z = ZIPF_ASSUMED.get(p)
        ztxt = f"   [Zipf assumes {z:.0%}]" if z else ""
        print(f"   top {p:5.1%}  ->  {curve[p]:5.1%}{ztxt}")
    g_global = gini(combined)
    print(f"  Gini (global) = {g_global:.3f}")
    out["global_curve"] = {f"{p}": curve[p] for p in KEY_PCTS}
    out["global_gini"] = g_global

    per_prompt = {}
    for n in names:
        c = key_counts([prompts[n]], NL, NE)
        per_prompt[n] = {"curve": cumulative_share(c), "gini": gini(c)}
    out["per_prompt_global_curve"] = {
        n: {f"{p}": per_prompt[n]["curve"][p] for p in KEY_PCTS} for n in names}

    # ---- PER-LAYER curve: within-layer top-25% (top-8) share ----
    print(f"\n== PER-LAYER within-layer concentration (each layer's {NE} experts) ==")
    print("  layer : top-25%(=top-8) traffic share | Gini | hottest-expert share")
    layer_top25, layer_gini = [], []
    per_layer = {}
    for L in range(NL):
        lc = [0] * NE
        for n in names:
            for step in prompts[n]["decode"]:
                for expert in step[L]:
                    lc[expert] += 1
        cur = cumulative_share(lc)
        share25 = cur[0.25]
        g = gini(lc)
        hottest = max(lc) / (sum(lc) / TK) if sum(lc) else 0  # frac of steps
        layer_top25.append(share25)
        layer_gini.append(g)
        per_layer[L] = {"top25_share": share25, "gini": g, "hottest_step_frac": hottest}
        flag = "  <- always-on" if hottest >= 0.999 else ""
        if L < 4 or hottest >= 0.999:
            print(f"   L{L:2d} : {share25:5.1%} | {g:.3f} | {hottest:5.1%}{flag}")
    mean25 = sum(layer_top25) / NL
    print(f"  ... ({NL} layers total)")
    print(f"  per-layer top-8/32 share: mean={mean25:.1%} "
          f"min={min(layer_top25):.1%} max={max(layer_top25):.1%}  "
          f"(uniform baseline={TK/NE:.1%})")
    out["per_layer"] = {str(L): per_layer[L] for L in range(NL)}
    out["per_layer_top25_mean"] = mean25

    # ---- Verdict vs the assumed Zipf ----
    measured_125 = curve[0.125]
    print("\n== VERDICT vs assumed Zipf ==")
    print(f"  Assumed:  top 12.5% of keys carry ~80% of traffic.")
    print(f"  Measured: top 12.5% of keys carry {measured_125:.1%} of traffic "
          f"(granite {NE}-expert top-{TK}).")
    print(f"  Per-layer top-8/32 (=25% of a layer) carries {mean25:.1%} "
          f"(vs the earlier-reported 45.4%).")
    if measured_125 < 0.65:
        print("  => Our routing is MATERIALLY MILDER than the assumed Zipf. Tiering "
              "wins are real but SMALLER than the 80/20 illustration implies.")
    out["measured_top12p5"] = measured_125

    with open("scripts/moe_concentration_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote scripts/moe_concentration_results.json")


if __name__ == "__main__":
    main()
