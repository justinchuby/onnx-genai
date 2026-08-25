#!/usr/bin/env python3
"""ORT counterpart of `int4_decode_loop_ab`: one Run == one decode token.

The sibling `ort_baseline.py` only covers f32 ops, so there was no way to say
what the int4 decode gap against ONNX Runtime actually is. This builds the same
five llama3-8B `MatMulNBits` projections the native harness drives and times
them through the ORT CPU EP under a matched thread count.

Same five llama3-8B projections, same m=1, same block size, same
accuracy_level, so the number is comparable to the native harness's
`steady` column-2 median. Weights are random but shape/dtype-exact, which is
what a GEMM's cost depends on.
"""

import argparse
import os
import resource
import statistics
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

PROJECTIONS = {
    # llama3-8B.
    "llama": [(4096, 6144, "qkv"), (4096, 4096, "o"), (4096, 14336, "gate"),
              (4096, 14336, "up"), (14336, 4096, "down")],
    # Qwen2.5-7B. Kept identical to the native harness's PROJECTIONS_QWEN --
    # a different n/k ratio (narrow GQA qkv, much wider MLP) is the point, so
    # if these two lists ever drift the comparison silently stops being one.
    "qwen": [(3584, 4608, "qkv"), (3584, 3584, "o"), (3584, 18944, "gate"),
             (3584, 18944, "up"), (18944, 3584, "down")],
}


def build(block_size, accuracy, with_zp, model):
    rng = np.random.default_rng(0)
    nodes, inputs, outputs, inits = [], [], [], []
    for i, (k, n, name) in enumerate(PROJECTIONS[model]):
        blocks = (k + block_size - 1) // block_size
        blob = block_size // 2
        inputs.append(helper.make_tensor_value_info(f"A{i}", TensorProto.FLOAT, [1, k]))
        outputs.append(helper.make_tensor_value_info(f"Y{i}", TensorProto.FLOAT, [1, n]))
        b = rng.integers(0, 256, size=(n, blocks, blob), dtype=np.uint8)
        inits.append(numpy_helper.from_array(b, f"B{i}"))
        sc = (rng.random((n * blocks,)).astype(np.float32) * 0.02 + 0.001)
        inits.append(numpy_helper.from_array(sc, f"S{i}"))
        node_in = [f"A{i}", f"B{i}", f"S{i}"]
        if with_zp:
            # Zero points are packed per *column*, so an odd block count pads
            # each column rather than the flattened whole: N*ceil(blocks/2), not
            # ceil(N*blocks/2). Those agree only for even `blocks`, which every
            # shape here happens to have.
            zp = np.full((n * ((blocks + 1) // 2),), 0x88, dtype=np.uint8)
            inits.append(numpy_helper.from_array(zp, f"Z{i}"))
            node_in.append(f"Z{i}")
        nodes.append(helper.make_node(
            "MatMulNBits", node_in, [f"Y{i}"], domain="com.microsoft",
            name=f"proj_{name}", K=k, N=n, bits=4, block_size=block_size,
            accuracy_level=accuracy))
    graph = helper.make_graph(nodes, "decode_step", inputs, outputs, inits)
    return helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--accuracy", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--zp", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--model", choices=sorted(PROJECTIONS), default="llama")
    # Concurrent sessions, matching the native harness's PROBE_SESSIONS. Without
    # this axis the ORT column is a single-session number being compared against
    # a multi-session native one, which is not a comparison.
    ap.add_argument("--sessions", type=int, default=1)
    args = ap.parse_args()

    model = build(args.block, args.accuracy, args.zp, args.model)
    # ~117 MB of initializers. Unique per process so two invocations cannot
    # race on the same file, and removed on the way out rather than left in the
    # source tree.
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f".ort_matmulnbits_baseline.{os.getpid()}.onnx",
    )
    onnx.save(model, path, save_as_external_data=False)
    try:
        run(args, path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def run(args, path):
    # One path for every session count. The previous split -- `run_concurrent`
    # above 1, a median-of-per-token-times `run_one` at exactly 1 -- meant the
    # baseline changed statistic at the single point where the native arm was
    # being judged. See the ratio-definition table in
    # `int4_decode_loop_ab.rs`.
    return run_concurrent(args, path)


def make_session(args, path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def run_concurrent(args, path):
    """Concurrent sessions, reported as aggregate throughput.

    Ledger §22 established that per-session medians desynchronise under
    contention badly enough to be meaningless -- one repetition read 9.3 ms
    against a 15.9 ms population. Aggregate throughput is the only valid
    multi-session statistic, so that is what this reports, and the native
    harness's `tokens_s_total` column is its counterpart.
    """
    sessions = [make_session(args, path) for _ in range(args.sessions)]
    rng = np.random.default_rng(1)
    feeds = {f"A{i}": rng.standard_normal((1, k)).astype(np.float32)
             for i, (k, _, _) in enumerate(PROJECTIONS[args.model])}
    for sess in sessions:
        for _ in range(3):
            sess.run(None, feeds)

    import threading
    reps = []
    cpu_reps = []
    for _ in range(args.reps):
        start = threading.Barrier(args.sessions + 1)
        done = [0.0] * args.sessions
        def drive(idx):
            sess = sessions[idx]
            start.wait()
            t0 = time.perf_counter()
            for _ in range(args.tokens):
                sess.run(None, feeds)
            done[idx] = time.perf_counter() - t0
        threads = [threading.Thread(target=drive, args=(i,))
                   for i in range(args.sessions)]
        for t in threads:
            t.start()
        start.wait()
        # Bracketed by exactly the two points that bracket `wall`, so
        # `cpu / (wall * threads)` is a busy fraction over the measured window
        # and nothing else. `getrusage(RUSAGE_SELF)` sums every thread of the
        # process, which is what makes it comparable to the native arm's
        # `/proc/self/stat` read -- and both are unaffected by a neighbour
        # stealing wall, unlike `/usr/bin/time`'s wall-derived `Percent of CPU`.
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        reps.append(args.sessions * args.tokens / wall)
        cpu_reps.append((r1.ru_utime - r0.ru_utime,
                         r1.ru_stime - r0.ru_stime, wall))
    # MEDIAN over repetitions, matching the native harness. `max` reported the
    # luckiest run and hid the spread; against a single-shot opponent that is a
    # systematic bias, not a tie-break.
    best = statistics.median(reps)
    spread = (max(reps) - min(reps)) / best * 100.0 if best else 0.0
    print(f"ORT model={args.model} block={args.block} acc={args.accuracy} "
          f"t={args.threads} sessions={args.sessions} zp={int(args.zp)}  "
          f"tokens_s_total={best:.1f}  ms_token_equiv={1e3 * args.sessions / best:.3f}  "
          f"spread_pct={spread:.1f}")
    # Every field comes from ONE repetition -- the one whose throughput is the
    # median under the *native harness's* definition, `sorted[len // 2]`, which
    # is not what `statistics.median` does at an even repetition count (it
    # averages the two middle values, producing a number no repetition
    # produced). Mixing repetitions across fields makes `cpu / (wall * threads)`
    # carry the rep-to-rep spread as bias; the identity
    # `tps = busy * threads / cpu_per_token` then fails by that spread, which is
    # how this was caught. `tokens_s_total` above is left on the original
    # definition so previously published rows stay comparable, and `tps_rep` is
    # emitted next to the CPU fields so a consumer can use a self-consistent
    # triple without having to know which definition produced the headline.
    order = sorted(range(len(reps)), key=lambda i: reps[i])
    rep = order[len(order) // 2]
    user_s, sys_s, wall_s = cpu_reps[rep]
    counted = args.sessions * args.tokens
    cpu_s = user_s + sys_s
    print(f"cpu phase=steady user_s={user_s:.4f} sys_s={sys_s:.4f} "
          f"cpu_s={cpu_s:.4f} wall_s={wall_s:.4f} tokens={counted} "
          f"tps_rep={reps[rep]:.4f} "
          f"cpu_s_per_token={cpu_s / counted:.6f} "
          f"sys_frac={(sys_s / cpu_s if cpu_s else 0.0):.4f}")
    return best


if __name__ == "__main__":
    main()
