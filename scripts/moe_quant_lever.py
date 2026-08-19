"""Quantisation lever for MoE offload: expert size vs the PCIe transfer floor.

PCIe bytes/token is the wall (see 2026-08-18-moe-offload-pcie-floor.md). Because
bytes/token scales LINEARLY with expert size, quantisation attacks the wall
directly and with no policy, prediction, or scheduler: halve expert bytes ->
halve the transfer floor -> double the tok/s ceiling. This sweeps expert size
against the same measured static-hot-pin routing curve and prices quantisation
ON THE SAME AXIS as the VRAM (static-pin) lever.

Expert-size sweep is a pure linear rescale of the composed floor; the value here
is (a) the tok/s ceiling and revised min-VRAM table per size, and (b) a direct
same-axis comparison: does a 2x/4x expert-size cut beat what more VRAM buys?

HONESTY CONSTRAINTS (baked into the report, not the arithmetic):
  * Quantisation TRADES ACCURACY -- unlike pinning/batching it is not a free win;
    it is a product decision and is labelled as such.
  * The baseline matters and changes the answer completely:
      - If the target expert is f16/bf16, int8 is 2x and int4 is 4x -- an easy,
        well-understood accuracy trade.
      - If the target expert is ALREADY int4 (16 MiB is plausibly a DeepSeek-class
        int4 expert), going smaller means int3/int2 -- sharp accuracy loss and
        little kernel support.
  * MECHANISM CONSTRAINT (measured by the mechanism agent): only canonical
    MatMulNBits int4 layout can be used file-backed. Marlin and MLAS-prepacked
    tensors need a layout that does not exist on disk and are EXCLUDED from any
    storage-to-device / mmap path. So on the offload path the quantisation lever
    effectively means "canonical int4", not arbitrary sub-int4 formats.

Routing MEASURED on granite-3.0-1b-a400m-instruct (32 experts, top-8, 24 layers,
real IBM trained router; 192 decode steps; CPU-only replay, no GPU). Every
bandwidth constant INFERRED, not measured on this box. Provenance hw:
i7-13800H (14C/20T), RTX 4060 8 GB, WDDM.
"""
import json
from collections import defaultdict

# Expert sizes to sweep (MiB) with the quant level each represents FROM AN F16 BASELINE.
# (If the 16 MiB target is already int4, the smaller rows are int3/int2 -- see report.)
SIZES = [
    (16.0, "f16 baseline (target-class expert)"),
    (8.0, "int8 from f16 (2x)  |  or int2 from a 16 MiB int4 expert"),
    (4.0, "int4 from f16 (4x, canonical MatMulNBits, file-backable)"),
]
PCIE_BW = {"pcie4_x8_13": 13.0, "pcie4_x16_26": 26.0, "pcie5_x16_55": 55.0}
VRAM_RATIOS = [0.05, 0.10, 0.25, 0.50, 0.65, 0.75]
BATCH_SIZES = [1, 8]
TARGET_TOKS = [5, 10, 20, 40]


def build(trace):
    NL, NE = trace["num_layers"], trace["num_experts"]
    prompts = trace["prompts"]
    names = list(prompts.keys())
    per = {n: [] for n in names}
    for n in names:
        for step in prompts[n]["decode"]:
            per[n].append([L * NE + step[L][j] for L in range(NL) for j in range(len(step[L]))])
    nsteps = min(len(per[n]) for n in names)
    reqs = []
    for t in range(nsteps):
        for n in names:
            reqs.append(per[n][t])
    counts = defaultdict(int)
    for r in reqs:
        for k in r:
            counts[k] += 1
    return reqs, counts, NL * NE


def static_pin_pcie(reqs, pin_set, W):
    total = 0
    for i in range(0, len(reqs), W):
        union = set()
        for r in reqs[i:i + W]:
            for k in r:
                if k not in pin_set:
                    union.add(k)
        total += len(union)
    return total / len(reqs)  # experts/token


def main():
    with open("scripts/moe_expert_trace.json") as f:
        trace = json.load(f)
    reqs, counts, n_keys = build(trace)
    order = sorted(counts, key=lambda k: counts[k], reverse=True)

    # precompute experts/token for the static-pin curve (size-independent)
    ept = {}  # (W, ratio) -> experts/token
    for W in BATCH_SIZES:
        for r in VRAM_RATIOS:
            B = max(1, round(r * n_keys))
            ept[(W, r)] = static_pin_pcie(reqs, set(order[:B]), W)

    print(f"model: {trace['model']}  n_keys={n_keys}  tokens={len(reqs)}")
    print("routing MEASURED; expert sizes & bandwidths INFERRED. static hot-pin policy.\n")

    out = {"model": trace["model"], "n_keys": n_keys, "sizes_mib": [s for s, _ in SIZES],
           "pcie_bw_gbps": PCIE_BW, "vram_ratios": VRAM_RATIOS,
           "experts_per_token": {f"W{W}_{r}": ept[(W, r)] for W in BATCH_SIZES for r in VRAM_RATIOS},
           "by_size": {}}

    for esz, label in SIZES:
        print(f"===== expert = {esz:.0f} MiB  [{label}] =====")
        print(" VRAM% | W=1: GB/tok  tok/s@13/26/55 | W=8: GB/tok  tok/s@13/26/55")
        by = {}
        for r in VRAM_RATIOS:
            def cell(W):
                gb = ept[(W, r)] * esz / 1024.0
                ts = {bn: (1000.0 / (gb / bw * 1000) if gb else 0) for bn, bw in PCIE_BW.items()}
                return gb, ts
            gb1, ts1 = cell(1)
            gb8, ts8 = cell(8)
            by[f"{r}"] = {"vram_ratio": r,
                          "W1": {"gb_per_tok": gb1, "toks": ts1},
                          "W8": {"gb_per_tok": gb8, "toks": ts8}}
            print(f" {r:4.0%} | {gb1:6.2f}  {ts1['pcie4_x8_13']:4.1f}/{ts1['pcie4_x16_26']:4.1f}/"
                  f"{ts1['pcie5_x16_55']:5.1f} | {gb8:6.2f}  {ts8['pcie4_x8_13']:4.1f}/"
                  f"{ts8['pcie4_x16_26']:4.1f}/{ts8['pcie5_x16_55']:5.1f}")
        out["by_size"][f"{esz}"] = by
        print()

    # ---- min VRAM % for target tok/s, per expert size (W=1, x8 link) ----
    print("== min VRAM % for target tok/s (W=1, PCIe4 x8 ~13 GB/s, INFERRED) ==")
    print(" expert |   5    10    20    40 tok/s")
    inv = {}
    for esz, _ in SIZES:
        cells = []
        for tgt in TARGET_TOKS:
            need = None
            for r in VRAM_RATIOS:
                gb = ept[(1, r)] * esz / 1024.0
                ts = 1000.0 / (gb / 13.0 * 1000) if gb else 0
                if ts >= tgt:
                    need = r
                    break
            inv[f"{esz}_{tgt}"] = need
            cells.append(f"{need:.0%}" if need is not None else ">75%")
        print(f" {esz:4.0f}MiB | " + "  ".join(f"{c:>4}" for c in cells))
    out["inversion_w1_x8"] = inv

    # ---- SAME-AXIS: quant vs more-VRAM at fixed budget ----
    print("\n== SAME-AXIS: quantisation vs VRAM (bytes/token, W=1) ==")
    e25 = ept[(1, 0.25)]
    e50 = ept[(1, 0.50)]
    e65 = ept[(1, 0.65)]
    print(f"  static-pin lever (16 MiB): VRAM 25% -> {e25*16:6.0f} MiB/tok ; "
          f"50% -> {e50*16:6.0f} ; 65% -> {e65*16:6.0f}")
    print(f"  quant lever (fixed VRAM 25%): 16 MiB -> {e25*16:6.0f} ; "
          f"8 MiB -> {e25*8:6.0f} ; 4 MiB -> {e25*4:6.0f} MiB/tok")
    print(f"  => a 4x expert-size cut at 25% VRAM ({e25*4:.0f} MiB/tok) beats "
          f"static-pin at 65% VRAM 16 MiB ({e65*16:.0f} MiB/tok),")
    print(f"     and they COMPOSE: 4 MiB @ 50% VRAM = {e50*4:.0f} MiB/tok.")
    out["same_axis"] = {"pin16_v25": e25*16, "pin16_v50": e50*16, "pin16_v65": e65*16,
                        "quant_v25_16": e25*16, "quant_v25_8": e25*8, "quant_v25_4": e25*4,
                        "compose_4mib_v50": e50*4}

    with open("scripts/moe_quant_lever_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote scripts/moe_quant_lever_results.json")


if __name__ == "__main__":
    main()
