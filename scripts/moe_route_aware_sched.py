"""Route-aware (expert-overlap) request scheduling -- benefit AND cost.

The external analysis claims that dispatching queued requests whose activated
experts overlap -- so a loaded expert serves many requests instead of being
loaded, used once, evicted -- can save more bandwidth than any storage API.
We test it offline on the real trace, and we measure the COST it hides:
reordering makes some requests wait, so we report the latency distribution and
the worst-case delay alongside the bandwidth saving.

Setup. Each decode step is one request needing its 24-layer x top-8 = 192
(layer,expert) keys. The 3 prompts x 64 steps = 192 requests arrive interleaved
(round-robin across prompts -> 3 concurrent streams). A batch of W requests is
served together; with an ideal per-batch cache the GPU loads the UNION of the
batch's expert sets once and serves all W. So:
    load cost of a batch      = |union of its requests' key sets|
    bandwidth per token       = (sum of batch unions) / num_requests * expert
No persistent cache is modelled here, to ISOLATE the scheduling effect from the
residency effect (that is the single-tier sim's job).

Schedulers (both see a lookahead window of the Q oldest undispatched requests):
  * fifo        : take the first W in arrival order.
  * route_aware : greedily pick W whose union is smallest (max overlap).
Sweep W (batch size) and Q (lookahead) -- the win must grow with Q.

Cost: delay_i = dispatch_rank_i - arrival_index_i (positions a request slips
past its arrival order). FIFO delay is 0 by construction; route_aware trades
delay for bandwidth. We report p50/p99/max delay -- a scheduler that halves
bandwidth but wrecks p99 is not a serving win.

Model behind the trace: granite-3.0-1b-a400m-instruct, 32 experts, top-8, 24
layers, real IBM trained router; onnxruntime 1.27.0 CPU EP, batch 1.
Provenance hw (no GPU used): i7-13800H (14C/20T), RTX 4060 8 GB, WDDM.
"""
import json

EXPERT_MIB = 16.0
BATCHES = [2, 4, 8]
LOOKAHEAD_MULT = [1, 2, 4, 8, 1000]  # Q = mult * W  (1000 => whole queue)


def load_requests(trace):
    NL, NE = trace["num_layers"], trace["num_experts"]
    prompts = trace["prompts"]
    names = list(prompts.keys())
    per = {n: [] for n in names}
    for n in names:
        for step in prompts[n]["decode"]:
            s = frozenset(L * NE + step[L][j] for L in range(NL) for j in range(len(step[L])))
            per[n].append(s)
    nsteps = min(len(per[n]) for n in names)
    reqs = []  # interleaved: step0_p0, step0_p1, ... => concurrent streams
    for t in range(nsteps):
        for n in names:
            reqs.append(per[n][t])
    return reqs


def schedule(reqs, W, Q, route_aware):
    n = len(reqs)
    dispatched = [False] * n
    arrival = list(range(n))
    total_union = 0
    delays = []
    rank = 0
    done = 0
    while done < n:
        # window = Q oldest undispatched (in arrival order)
        window = [i for i in arrival if not dispatched[i]][:Q]
        if not window:
            break
        if not route_aware:
            pick = window[:W]
        else:
            pick = [window[0]]
            union = set(reqs[window[0]])
            cand = window[1:]
            while len(pick) < W and cand:
                best, best_inc, best_u = None, None, None
                for c in cand:
                    u = union | reqs[c]
                    inc = len(u) - len(union)
                    if best_inc is None or inc < best_inc:
                        best, best_inc, best_u = c, inc, u
                pick.append(best)
                union = best_u
                cand.remove(best)
        union = set()
        for i in pick:
            union |= reqs[i]
        total_union += len(union)
        for i in pick:
            dispatched[i] = True
            delays.append(rank - i)
            rank += 1
            done += 1
    return total_union, delays


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1))))
    return sorted_vals[k]


def main():
    with open("scripts/moe_expert_trace.json") as f:
        trace = json.load(f)
    reqs = load_requests(trace)
    n = len(reqs)
    keyset = set().union(*reqs)
    print(f"model: {trace['model']}  requests={n}  keys/request=192  "
          f"distinct-keys={len(keyset)}  expert={EXPERT_MIB:.0f} MiB")

    # reference: no batching (each request loads its own 192 keys)
    solo_bt = 192 * EXPERT_MIB
    print(f"no-batching (W=1) bandwidth = {solo_bt:.0f} MiB/token "
          f"(192 experts/token)")

    out = {"model": trace["model"], "num_requests": n, "expert_mib": EXPERT_MIB,
           "solo_bytes_per_token_mib": solo_bt, "results": {}}

    for W in BATCHES:
        print(f"\n===== batch W={W} =====")
        print("   Q   sched        union/tok  bytes/tok@16MiB  vs-FIFO  | delay p50/p99/max")
        fifo_bt = None
        for mult in LOOKAHEAD_MULT:
            Q = min(n, W * mult)
            for route in (False, True):
                tot_union, delays = schedule(reqs, W, Q, route)
                # tot_union sums |union| once per batch; a batch serves W tokens and
                # there are n/W batches, so experts loaded per served token = tot_union/n.
                bt = tot_union / n * EXPERT_MIB
                sd = sorted(delays)
                p50, p99, mx = pct(sd, 0.50), pct(sd, 0.99), max(sd)
                name = "route_aware" if route else "fifo"
                if not route:
                    fifo_bt = bt
                    vs = "  (base)"
                else:
                    vs = f"{(1-bt/fifo_bt)*100:+5.1f}%" if fifo_bt else "   n/a"
                print(f"  {Q:4d}  {name:11s}  {tot_union/n:8.1f}   {bt:8.0f}       "
                      f"{vs}   | {p50:4d}/{p99:4d}/{mx:4d}")
                out["results"][f"W{W}_Q{Q}_{name}"] = {
                    "W": W, "Q": Q, "route_aware": route,
                    "union_per_token": tot_union / n,
                    "bytes_per_token_mib_16": bt,
                    "delay_p50": p50, "delay_p99": p99, "delay_max": mx}
        # theoretical floor: all n in one union / n
        floor_bt = len(keyset) / n * EXPERT_MIB
        print(f"  floor (all {n} co-scheduled, union={len(keyset)}): "
              f"{floor_bt:.0f} MiB/token")

    print("\n== READING ==")
    print("  union/tok is experts loaded per served token; lower is better. FIFO")
    print("  batching already captures the always-on/hot-core overlap (adjacent")
    print("  steps share it); route_aware's extra win is what reordering buys BEYOND")
    print("  that, and delay p99/max is the fairness price. bytes/tok is @16 MiB.")

    with open("scripts/moe_route_aware_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote scripts/moe_route_aware_results.json")


if __name__ == "__main__":
    main()
