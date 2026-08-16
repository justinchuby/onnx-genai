#!/usr/bin/env python3
"""Generate the production-shaped GQA benchmark grid.

Head/kv-head/head_dim triples are taken from public model configs. No weights
are downloaded: the graphs are single-node and the tensor contents are the
benchmark harness's deterministic synthetic pattern (SYNTHETIC DATA).
"""

from pathlib import Path

from gen_gqa import build_gqa

CONFIGS = {
    # name: (num_heads, kv_num_heads, head_dim)
    "qwen3_0p6b": (16, 8, 128),
    "qwen2p5_0p5b": (14, 2, 64),
    "phi3_mini_4k": (32, 32, 96),
    "llama3_8b": (32, 8, 128),
}

import argparse

DEFAULT_OUT = Path(__file__).resolve().parent / "models" / "grid"


def main(OUT: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for name, (h, kv, d) in CONFIGS.items():
        for past in (511, 2047, 8191):
            for batch in (1, 4):
                if batch == 4 and past != 2047:
                    continue
                path = OUT / f"dec_{name}_b{batch}_p{past}.onnx"
                build_gqa(
                    path,
                    batch=batch,
                    num_heads=h,
                    kv_num_heads=kv,
                    head_dim=d,
                    q_seq=1,
                    past_seq=past,
                )
                made.append(path)
        for q in (128, 512):
            path = OUT / f"pre_{name}_b1_q{q}.onnx"
            build_gqa(
                path,
                batch=1,
                num_heads=h,
                kv_num_heads=kv,
                head_dim=d,
                q_seq=q,
                past_seq=0,
            )
            made.append(path)
    for p in made:
        print(p.name, f"{p.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    main(ap.parse_args().out)
