"""Three-tier MoE expert cache: VRAM (hot) / DRAM (warm) / SSD (cold).

Tests the external analysis's central claim: with a Zipfian-ish router, a DRAM
warm tier can keep the SSD out of the critical path, so DirectStorage/GDS is a
cold-miss backstop, not the main event. The decisive output is *bytes fetched
from SSD per token* as a function of the VRAM and DRAM budgets.

Model (exclusive two-level LRU over experts keyed (layer, expert)):
  * VRAM: `bv` resident slots. A hit here is free (weights already on GPU).
  * DRAM: `bd` warm slots (victim cache below VRAM). Serving a DRAM-resident
    expert costs a DRAM->VRAM copy over PCIe.
  * SSD: everything else (unbounded). Serving here costs an NVMe read.
  On access: VRAM hit -> free; DRAM hit -> promote (PCIe copy), demote VRAM-LRU
  to DRAM; miss -> NVMe read (SSD bytes), insert to VRAM, cascade LRU eviction
  VRAM->DRAM->drop.

We separate COMPULSORY (first-ever touch of a key -- unavoidable, a cold-start
artefact of a short 64-step trace) from CAPACITY SSD traffic (the steady-state
number that decides whether SSD stays out of the hot path).

Bandwidths are ASSUMED constants for order-of-magnitude transfer-time context,
NOT measured on this box; labelled INFERRED. PCIe 4.0 x16 ~26 GB/s (this laptop
4060 may be x8 ~ half that), NVMe Gen4 ~6 GB/s, DDR5 system RAM ~80 GB/s.

Model behind the trace: granite-3.0-1b-a400m-instruct, 32 experts, top-8, 24
layers, real IBM trained router; onnxruntime 1.27.0 CPU EP, batch 1.
Provenance hw (no GPU used): i7-13800H (14C/20T), RTX 4060 8 GB, WDDM.
"""
import json
from collections import OrderedDict

EXPERT_MIB = {"granite_int4_0p75": 0.75, "target_16": 16.0}
# INFERRED / assumed constants (not measured on this box):
PCIE_GBPS = 26.0   # DRAM->VRAM, PCIe 4.0 x16 realistic
NVME_GBPS = 6.0    # SSD->VRAM, NVMe Gen4 realistic
DRAM_GBPS = 80.0   # DDR5 system-RAM bandwidth, for reference only

VRAM_RATIOS = [0.05, 0.10, 0.25]
DRAM_RATIOS = [0.0, 0.25, 0.50, 1.00, 2.00]  # >1.0 means DRAM can hold whole bank


def build_groups(entry, NL, NE):
    groups = []
    for step in entry["decode"]:
        for L in range(NL):
            groups.append([L * NE + e for e in step[L]])
    return groups, len(entry["decode"])


def sim_tiers(groups, bv, bd):
    """Exclusive VRAM/DRAM LRU. Returns per-tier access + fetch counts."""
    vram = OrderedDict()   # key -> None, MRU at end
    dram = OrderedDict()
    seen = set()
    vram_hits = dram_hits = ssd_compulsory = ssd_capacity = 0
    for grp in groups:
        for k in grp:
            if k in vram:
                vram.move_to_end(k)
                vram_hits += 1
            elif k in dram:
                dram_hits += 1                      # PCIe DRAM->VRAM copy
                del dram[k]
                vram[k] = None
                vram.move_to_end(k)
            else:
                if k in seen:
                    ssd_capacity += 1               # NVMe read (avoidable)
                else:
                    ssd_compulsory += 1             # NVMe read (unavoidable first touch)
                    seen.add(k)
                vram[k] = None
                vram.move_to_end(k)
            # cascade eviction VRAM -> DRAM -> drop
            while len(vram) > bv:
                ek, _ = vram.popitem(last=False)
                if bd > 0:
                    dram[ek] = None
                    dram.move_to_end(ek)
            while len(dram) > bd:
                dram.popitem(last=False)
    return {"vram_hits": vram_hits, "dram_hits": dram_hits,
            "ssd_compulsory": ssd_compulsory, "ssd_capacity": ssd_capacity}


def main():
    with open("scripts/moe_expert_trace.json") as f:
        trace = json.load(f)
    NL, NE = trace["num_layers"], trace["num_experts"]
    n_keys = NL * NE
    prompts = trace["prompts"]
    esz = EXPERT_MIB["target_16"]

    print(f"model: {trace['model']}  n_keys={n_keys}  expert={esz:.0f} MiB")
    print(f"ASSUMED bandwidths (INFERRED, not measured): PCIe {PCIE_GBPS} GB/s, "
          f"NVMe {NVME_GBPS} GB/s, DDR5 {DRAM_GBPS} GB/s")

    out = {"model": trace["model"], "n_keys": n_keys, "expert_mib": EXPERT_MIB,
           "assumed_bw_gbps": {"pcie": PCIE_GBPS, "nvme": NVME_GBPS, "dram": DRAM_GBPS},
           "vram_ratios": VRAM_RATIOS, "dram_ratios": DRAM_RATIOS, "workloads": {}}

    for name, entry in prompts.items():
        groups, nsteps = build_groups(entry, NL, NE)
        distinct = len({k for g in groups for k in g})
        print(f"\n===== PROMPT[{name}]  steps={nsteps} distinct-keys={distinct} "
              f"compulsory-floor={distinct/nsteps:.1f} experts/token =====")
        print("  VRAM% DRAM% | SSD-capacity  SSD-total  DRAM(PCIe)  | est. ms/token "
              "(SSD+PCIe transfer, @16MiB)")
        wl = {}
        for vr in VRAM_RATIOS:
            bv = max(1, round(vr * n_keys))
            for dr in DRAM_RATIOS:
                bd = round(dr * n_keys)
                r = sim_tiers(groups, bv, bd)
                cap_pt = r["ssd_capacity"] / nsteps
                comp_pt = r["ssd_compulsory"] / nsteps
                ssd_tot_pt = cap_pt + comp_pt
                dram_pt = r["dram_hits"] / nsteps
                # transfer time (ms) per token at 16 MiB, assumed BW
                ssd_ms = (ssd_tot_pt * esz / 1024) / NVME_GBPS * 1000
                pcie_ms = (dram_pt * esz / 1024) / PCIE_GBPS * 1000
                wl[f"v{vr}_d{dr}"] = {
                    "bv": bv, "bd": bd,
                    "ssd_capacity_per_tok": cap_pt, "ssd_compulsory_per_tok": comp_pt,
                    "dram_pcie_per_tok": dram_pt,
                    "ssd_cap_bytes_mib_16": cap_pt * esz,
                    "ssd_total_bytes_mib_16": ssd_tot_pt * esz,
                    "dram_bytes_mib_16": dram_pt * esz,
                    "est_ms_per_tok_16": ssd_ms + pcie_ms}
                print(f"  {vr:4.0%} {dr:4.0%} | cap={cap_pt:5.2f}  "
                      f"tot={ssd_tot_pt:5.2f}   pcie={dram_pt:5.2f}    | "
                      f"{ssd_ms+pcie_ms:6.2f} ms  (SSD {ssd_ms:5.2f} + PCIe {pcie_ms:5.2f})")
        out["workloads"][name] = wl

    print("\n== READING ==")
    print("  'SSD-capacity' is the avoidable steady-state cold traffic; if a DRAM")
    print("  budget drives it to ~0, the SSD leaves the critical path (DirectStorage")
    print("  becomes a cold-miss backstop). 'SSD-compulsory' is first-touch only, an")
    print("  artefact of the 64-step trace, and would amortise over a long generation.")

    with open("scripts/moe_tier_sim_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote scripts/moe_tier_sim_results.json")


if __name__ == "__main__":
    main()
