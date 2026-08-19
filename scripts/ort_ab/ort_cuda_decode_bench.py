#!/usr/bin/env python3
"""Matched-version ORT-CUDA greedy decode benchmark via a direct
onnxruntime.InferenceSession with GPU io-binding (present->past kept on
device). Works for qwen2 / granite style genai decoder graphs whose inputs are
input_ids, attention_mask, past_key_values.%d.key/value and whose outputs are
logits, present.%d.key/value. No position_ids.

Reports decode tok/s (after --decode-skip warm steps) and the greedy token ids,
so the native-vs-ORT A/B compares the same work. Stamps ort.__version__.
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import onnxruntime as ort


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-ids", required=True,
                    help="comma-separated int64 prompt token ids")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--decode-skip", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompt_ids = [int(x) for x in args.prompt_ids.split(",") if x != ""]

    so = ort.SessionOptions()
    sess = ort.InferenceSession(
        args.model, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    assert "CUDAExecutionProvider" in sess.get_providers(), sess.get_providers()

    inputs = sess.get_inputs()
    has_position_ids = any(i.name == "position_ids" for i in inputs)
    # discover layer count and kv dims from past inputs
    past_keys = [i.name for i in inputs if re.match(r"past_key_values\.\d+\.key", i.name)]
    n_layers = len(past_keys)
    kv_in = next(i for i in inputs if i.name.endswith(".0.key"))
    kv_heads = int(kv_in.shape[1])
    head_size = int(kv_in.shape[3])
    kv_np_dtype = np.float16 if "float16" in kv_in.type else np.float32

    dev = "cuda"
    dev_id = 0

    def empty_kv():
        z = np.zeros((1, kv_heads, 0, head_size), dtype=kv_np_dtype)
        return ort.OrtValue.ortvalue_from_numpy(z, dev, dev_id)

    def run_once(n_tokens: int):
        past = {}
        for l in range(n_layers):
            past[f"past_key_values.{l}.key"] = empty_kv()
            past[f"past_key_values.{l}.value"] = empty_kv()

        cur_ids = np.array([prompt_ids], dtype=np.int64)
        total_len = len(prompt_ids)
        out_tokens = []
        t_decode_start = None
        step = 0
        while step < n_tokens:
            io = sess.io_binding()
            ids_ov = ort.OrtValue.ortvalue_from_numpy(cur_ids, dev, dev_id)
            mask_np = np.ones((1, total_len), dtype=np.int64)
            mask_ov = ort.OrtValue.ortvalue_from_numpy(mask_np, dev, dev_id)
            io.bind_ortvalue_input("input_ids", ids_ov)
            io.bind_ortvalue_input("attention_mask", mask_ov)
            if has_position_ids:
                if step == 0:
                    pos_np = np.arange(total_len, dtype=np.int64).reshape(1, -1)
                else:
                    pos_np = np.array([[total_len - 1]], dtype=np.int64)
                pos_ov = ort.OrtValue.ortvalue_from_numpy(pos_np, dev, dev_id)
                io.bind_ortvalue_input("position_ids", pos_ov)
            for l in range(n_layers):
                io.bind_ortvalue_input(f"past_key_values.{l}.key", past[f"past_key_values.{l}.key"])
                io.bind_ortvalue_input(f"past_key_values.{l}.value", past[f"past_key_values.{l}.value"])
            io.bind_output("logits", "cpu")
            for l in range(n_layers):
                io.bind_output(f"present.{l}.key", dev, dev_id)
                io.bind_output(f"present.{l}.value", dev, dev_id)
            sess.run_with_iobinding(io)

            outs = io.get_outputs()
            names = [o.name for o in sess.get_outputs()]
            omap = dict(zip(names, outs))
            logits = omap["logits"].numpy()  # [1, seq, vocab]
            nxt = int(np.argmax(logits[0, -1, :]))
            out_tokens.append(nxt)

            # present -> past
            for l in range(n_layers):
                past[f"past_key_values.{l}.key"] = omap[f"present.{l}.key"]
                past[f"past_key_values.{l}.value"] = omap[f"present.{l}.value"]

            cur_ids = np.array([[nxt]], dtype=np.int64)
            total_len += 1
            step += 1
            if step == args.decode_skip:
                t_decode_start = time.perf_counter()
        t_end = time.perf_counter()
        timed = max(step - args.decode_skip, 1)
        dt = (t_end - t_decode_start) if t_decode_start else None
        tps = (timed / dt) if dt else None
        return out_tokens, tps

    for _ in range(args.warmups):
        run_once(min(args.tokens, 16))

    tokens, tps = run_once(args.tokens)
    result = {
        "ort_version": ort.__version__,
        "providers": sess.get_providers(),
        "prompt_ids": prompt_ids,
        "tokens": tokens,
        "tok_s": tps,
    }
    print(json.dumps(result))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
