"""Offline expert-cache simulator: how much can a residency policy win?

Replays the real trained-router expert-selection trace
(`scripts/moe_expert_trace.json`, produced by `dump_moe_expert_trace.py`) against
a bounded expert cache and reports, per policy per budget ratio:

  * bytes/token    = page-ins/token x expert_size  (THE decision number; reported
                     at 0.75 MiB granite int4 and 16 MiB GLM/DeepSeek-class experts)
  * page-ins/token = page-ins per generated token (size-independent)
  * miss rate      = expert page-ins / expert accesses

Budget is GLOBAL across all 24 layers -- VRAM is one shared pool, not a per-layer
slice. Experts are keyed (layer, expert): expert e in layer L is distinct from
expert e in layer L'. A budget ratio r means round(r * num_experts * num_layers)
resident slots shared across the whole model, so the allocator is free to spend
more slots on layers with hot/always-on experts (layers 1-2 here) and fewer on
layers with flat routing. A per-layer EQUAL-SPLIT arm (each layer gets r*num_experts
slots) is reported only as a baseline to beat.

Access order within a decode step is layer 0..23 sequentially, each accessing its
top-8 experts; that ordering drives LRU/FIFO recency and oracle next-use.

Why this is valid even though granite fits in VRAM: routing skew is a property of
the trained router and the prompt, NOT of VRAM size, so the trace is a legitimate
workload regardless of granite's footprint.

What it does NOT establish (inferred / out of scope):
  * achieved wall-clock or the paging mechanism's real cost;
  * whether skew generalises from granite's 32-expert top-8 router to a
    DeepSeek-class 256-expert top-8 one. A budget ratio means something different
    when the per-token working set is 8/256 = 3% of the bank rather than 8/32 =
    25%: intuitively the larger-bank case should have MORE headroom for a
    residency policy (the hot core is a smaller fraction to pin), but this trace
    cannot demonstrate it. Direction stated; label = INFERRED.
  * bytes/token here is monotone in page-ins/token because granite's experts are
    uniform size; with heterogeneous expert sizes the two could diverge and bytes
    would be the only correct ranking.

Model behind the trace: granite-3.0-1b-a400m-instruct, 32 experts, top-8, 24
layers, real IBM trained router. See:
  wiki/memory/MoE Router Skew and Always-On Experts.md
  docs/benchmarks/2026-08-18-moe-router-skew-granite.md
"""
import bisect
import json
import random
from collections import defaultdict

random.seed(1234)

EXPERT_SIZES_MIB = {"granite_int4_0p75": 0.75, "target_16": 16.0}
BUDGET_RATIOS = [0.10, 0.25, 0.50, 0.75, 0.90]
ONLINE = ["oracle", "random", "fifo", "lru", "lfu"]


def build_groups(prompt_entry, NL, NE):
    """Flat list of access groups in (step, layer) order; keys = L*NE + e."""
    decode = prompt_entry["decode"]  # [step][layer] = [8]
    groups = []
    for step in decode:
        for L in range(NL):
            groups.append([L * NE + e for e in step[L]])
    return groups, len(decode)


def count_keys(groups, n_keys):
    c = [0] * n_keys
    for grp in groups:
        for k in grp:
            c[k] += 1
    return c


def sim_online(groups, budget, policy):
    resident = set()
    load_tick = {}
    last_use = {}
    freq = defaultdict(int)
    tick = 0
    page_ins = 0
    future = defaultdict(list)
    if policy == "oracle":
        for g, grp in enumerate(groups):
            for k in grp:
                future[k].append(g)
    for g, grp in enumerate(groups):
        for k in grp:
            tick += 1
            if k not in resident:
                page_ins += 1
                resident.add(k)
                load_tick[k] = tick
            last_use[k] = tick
            freq[k] += 1
        while len(resident) > budget:
            protected = set(grp)
            cand = [k for k in resident if k not in protected] or list(resident)
            if policy == "random":
                victim = random.choice(cand)
            elif policy == "fifo":
                victim = min(cand, key=lambda k: load_tick[k])
            elif policy == "lru":
                victim = min(cand, key=lambda k: last_use[k])
            elif policy == "lfu":
                victim = min(cand, key=lambda k: (freq[k], last_use[k]))
            elif policy == "oracle":
                def nxt(k):
                    lst = future[k]
                    i = bisect.bisect_right(lst, g)
                    return lst[i] if i < len(lst) else float("inf")
                victim = max(cand, key=nxt)
            else:
                raise ValueError(policy)
            resident.discard(victim)
            load_tick.pop(victim, None)
    return page_ins


def sim_static_pin(groups, budget, counts):
    """Pin the top-`budget` keys by frequency; stream (never retain) the rest."""
    order = sorted(range(len(counts)), key=lambda k: counts[k], reverse=True)
    pinned = set(order[:budget])
    page_ins = 0
    for grp in groups:
        for k in grp:
            if k not in pinned:
                page_ins += 1
    return page_ins


def sim_hybrid(groups, budget, counts, num_steps):
    """Pin always-on core (count >= num_steps), LRU over the rest within budget."""
    core = {k for k in range(len(counts)) if counts[k] >= num_steps}
    if len(core) > budget:
        core = set(sorted(core, key=lambda k: counts[k], reverse=True)[:budget])
    lru_budget = max(budget - len(core), 0)
    resident = set(core)
    last_use = {}
    tick = 0
    page_ins = 0
    for grp in groups:
        for k in grp:
            tick += 1
            if k in core:
                continue
            if k not in resident:
                page_ins += 1
                resident.add(k)
            last_use[k] = tick
        non_core = [k for k in resident if k not in core]
        while len(non_core) > lru_budget:
            protected = set(grp)
            cand = [k for k in non_core if k not in protected] or non_core
            victim = min(cand, key=lambda k: last_use.get(k, 0))
            resident.discard(victim)
            non_core = [k for k in resident if k not in core]
    return page_ins


def sim_equal_split(groups, ratio, NL, NE, policy):
    """Baseline: each layer gets round(ratio*NE) slots; no cross-layer sharing."""
    per_layer = max(1, round(ratio * NE))
    by_layer = defaultdict(list)
    for grp in groups:
        L = grp[0] // NE
        by_layer[L].append([k - L * NE for k in grp])
    total = 0
    for L, seq in by_layer.items():
        if policy == "static_pin":
            lc = [0] * NE
            for step in seq:
                for e in step:
                    lc[e] += 1
            total += sim_static_pin(seq, per_layer, lc)
        else:
            total += sim_online(seq, per_layer, policy)
    return total


def run(name, groups, counts, NL, NE, num_steps, pin_counts=None):
    n_keys = NL * NE
    accesses = num_steps * 8 * NL
    pin = pin_counts if pin_counts is not None else counts
    res = {}
    for ratio in BUDGET_RATIOS:
        budget = max(1, round(ratio * n_keys))
        row = {p: sim_online(groups, budget, p) for p in ONLINE}
        row["static_pin"] = sim_static_pin(groups, budget, pin)
        row["hybrid"] = sim_hybrid(groups, budget, pin, num_steps)
        row["equal_oracle"] = sim_equal_split(groups, ratio, NL, NE, "oracle")
        row["equal_lru"] = sim_equal_split(groups, ratio, NL, NE, "lru")
        res[ratio] = {"budget": budget, "accesses": accesses, "page_ins": row}
    return res


def print_table(name, res, num_steps):
    cols = ["oracle", "random", "fifo", "lru", "lfu", "static_pin", "hybrid",
            "equal_oracle", "equal_lru"]
    print(f"\n===== {name} =====")
    print("cell = bytes/token@16MiB(MiB) | page-ins/token | miss-rate")
    print("ratio(budget)   " + "".join(f"{c:>16}" for c in cols))
    for ratio, r in res.items():
        acc = r["accesses"]
        cells = []
        for c in cols:
            pi = r["page_ins"][c]
            pit = pi / num_steps
            cells.append(f"{pit*16.0:6.0f}|{pit:4.0f}|{pi/acc:.2f}")
        print(f" {ratio:4.0%}({r['budget']:3d})  " + "".join(f"{x:>16}" for x in cells))
    print(" oracle advantage (bytes/token @16MiB):")
    for ratio, r in res.items():
        pi = r["page_ins"]
        o = pi["oracle"] / num_steps * 16.0
        rnd = pi["random"] / num_steps * 16.0
        sp = pi["static_pin"] / num_steps * 16.0
        hy = pi["hybrid"] / num_steps * 16.0
        print(f"  {ratio:4.0%}: oracle={o:6.0f}  random={rnd:6.0f} (oracle saves "
              f"{(1-o/rnd)*100 if rnd else 0:3.0f}%)  static_pin={sp:6.0f} "
              f"({(sp-o)/o*100 if o else 0:+4.0f}% vs oracle)  hybrid={hy:6.0f} "
              f"({(hy-o)/o*100 if o else 0:+4.0f}% vs oracle)")


def main():
    with open("scripts/moe_expert_trace.json") as f:
        trace = json.load(f)
    NL, NE, TK = trace["num_layers"], trace["num_experts"], trace["top_k"]
    prompts = trace["prompts"]
    n_keys = NL * NE

    groups_by, steps_by, counts_by = {}, {}, {}
    for name, e in prompts.items():
        g, ns = build_groups(e, NL, NE)
        groups_by[name], steps_by[name] = g, ns
        counts_by[name] = count_keys(g, n_keys)

    print(f"model: {trace['model']}  layers={NL} experts={NE} top_k={TK} "
          f"n_keys={n_keys}")
    for name in prompts:
        c, ns = counts_by[name], steps_by[name]
        always_on = sum(1 for k in range(n_keys) if c[k] >= ns)
        print(f"  trace[{name}]: steps={ns} always-on keys={always_on} "
              f"hottest key={max(c)/ns:.0%} of steps (uniform={TK/NE:.0%})")
    if all(max(counts_by[n]) / steps_by[n] < 0.4 for n in prompts):
        print("WARNING: selection looks near-uniform -- suspect the capture, not the model.")

    out = {"model": trace["model"], "num_layers": NL, "num_experts": NE, "top_k": TK,
           "n_keys": n_keys, "budget_ratios": BUDGET_RATIOS,
           "expert_sizes_mib": EXPERT_SIZES_MIB, "workloads": {}}
    for name in prompts:
        res = run(name, groups_by[name], counts_by[name], NL, NE, steps_by[name])
        print_table(f"PROMPT[{name}] GLOBAL budget (in-sample static pin)", res, steps_by[name])
        out["workloads"][name] = res

    print("\n\n########## CROSS-PROMPT STATIC-PIN DECAY (global budget) ##########")
    cross = {}
    for name in prompts:
        others = [m for m in prompts if m != name]
        pin = [sum(counts_by[m][k] for m in others) for k in range(n_keys)]
        acc = steps_by[name] * 8 * NL
        rows = {}
        for ratio in BUDGET_RATIOS:
            budget = max(1, round(ratio * n_keys))
            cx = sim_static_pin(groups_by[name], budget, pin) / acc
            ins = sim_static_pin(groups_by[name], budget, counts_by[name]) / acc
            rows[ratio] = {"budget": budget, "cross": cx, "insample": ins}
        cross[name] = rows
        print(f"\n test on {name} (pinned from {others}):")
        for ratio, r in rows.items():
            print(f"  {ratio:4.0%}(b{r['budget']:3d}): cross={r['cross']:.3f} "
                  f"in-sample={r['insample']:.3f} decay={r['cross']-r['insample']:+.3f}")
    out["cross_prompt_static_pin"] = cross

    with open("scripts/moe_cache_sim_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote scripts/moe_cache_sim_results.json")


if __name__ == "__main__":
    main()
