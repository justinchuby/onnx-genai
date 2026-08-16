#!/usr/bin/env python3
"""GQA decode graphs sweeping attended-KV bytes per head across the L3 ratio.

llama3-8b head geometry (32 q heads, 8 kv heads, head_dim 128), so the fused
gate's `window * (k_dim + v_dim) * 4` works out to exactly 1 KiB per attended
token per head. past_seq is chosen so the per-head working set lands on
1/2/4/8/16/32 MiB.

No weights are downloaded (SYNTHETIC DATA); the graphs are single-node and the
tensor contents come from the harness's deterministic pattern.
"""

from pathlib import Path

from gen_gqa import build_gqa

import argparse

DEFAULT_OUT = Path(__file__).resolve().parent / "models" / "l3sweep"

# past_seq -> MiB of attended K+V per KV head at head_dim 128
PASTS = {1023: 1, 2047: 2, 4095: 4, 8191: 8, 16383: 16, 32767: 32}


def main(OUT: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for past, mib in PASTS.items():
        path = OUT / f"l3_llama3_b1_p{past}_{mib}mib.onnx"
        if path.exists():
            print("skip", path.name)
            continue
        build_gqa(
            path,
            batch=1,
            num_heads=32,
            kv_num_heads=8,
            head_dim=128,
            q_seq=1,
            past_seq=past,
        )
        print("wrote", path.name, f"{path.stat().st_size / 2**20:.0f} MiB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    main(ap.parse_args().out)
