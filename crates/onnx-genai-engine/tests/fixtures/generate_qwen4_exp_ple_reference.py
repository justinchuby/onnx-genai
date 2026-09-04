#!/usr/bin/env python3
"""Generate the deterministic Qwen4-Exp PLE conformance vectors.

Authoritative public sources, pinned 2026-08-28:

* QwenLM/Qwen3.8-Flash-Next release repository:
  https://github.com/QwenLM/Qwen3.8-Flash-Next/tree/69885871a64393807d988b27b1b5e380e8f28526
* Official Qwen/Qwen3.8-Flash-Next config:
  https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json
* Transformers Qwen4-Exp implementation introduced by commit:
  https://github.com/huggingface/transformers/blob/fc5c5bde8e656dad91cbf34e61940d984b1c7b91/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1048-L1260

This is an independently expressed scalar reference for the published
hash/lookup/projection/gating/dilated-convolution equations. It uses small,
deterministic synthetic tables and weights; it is not official-checkpoint
weight parity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
PRIME_1 = 10007

GEOMETRY = {
    "vocab_size": 248_320,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "hc_count": 4,
    "hidden_size": 2,
    "ple_embed_dim": 16,
    "ple_layer_index": 0,
    "conv_kernel": 4,
    "conv_dilation": 3,
    "ngram_vocab_size_base": 31,
    "seed": 1234,
    "eos_token_id": 248_044,
}


def splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def multipliers() -> list[int]:
    maximum = ((1 << 63) - 1) // GEOMETRY["vocab_size"]
    half_bound = max(1, maximum // 2)
    base_seed = GEOMETRY["seed"] + PRIME_1 * GEOMETRY["ple_layer_index"]
    return [
        2 * (splitmix64((base_seed + SPLITMIX_GAMMA * (index + 1)) & MASK64) % half_bound) + 1
        for index in range(GEOMETRY["ngram_size"])
    ]


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def nth_prime_after(start: int, count: int) -> int:
    value = start
    for _ in range(count):
        value += 1
        while not is_prime(value):
            value += 1
    return value


def weights() -> dict[str, list[int] | list[float]]:
    heads = (GEOMETRY["ngram_size"] - 1) * GEOMETRY["heads_per_ngram"]
    sizes = [
        nth_prime_after(GEOMETRY["ngram_vocab_size_base"] - 1, head + 1)
        for head in range(heads)
    ]
    offsets: list[int] = []
    total = 0
    for size in sizes:
        offsets.append(total)
        total += size
    channels = GEOMETRY["hc_count"] * GEOMETRY["hidden_size"]
    return {
        "multipliers": multipliers(),
        "head_vocab_sizes": sizes,
        "head_offsets": offsets,
        "ngram_embedding": [((index % 23) - 11.0) / 16.0 for index in range(total)],
        "key_weights": [((index * 7 % 19) - 9.0) / 32.0 for index in range(heads * channels)],
        "value_weights": [
            ((index * 11 % 17) - 8.0) / 24.0
            for index in range(heads * GEOMETRY["hidden_size"])
        ],
        "norm_key": [((index * 3 % 17) - 8.0) / 32.0 for index in range(channels)],
        "norm_query": [((index * 5 % 19) - 9.0) / 40.0 for index in range(channels)],
        "norm_conv": [((index * 7 % 23) - 11.0) / 48.0 for index in range(channels)],
        "conv_weights": [
            (0.5 ** ((index % GEOMETRY["conv_kernel"]) + 1))
            * (1.0 if (index % GEOMETRY["conv_kernel"]) % 2 == 0 else -1.0)
            for index in range(channels * GEOMETRY["conv_kernel"])
        ],
    }


def rms_norm(values: list[float], learned_weight: list[float]) -> list[float]:
    if len(values) != len(learned_weight):
        raise ValueError(
            f"RMSNorm values and learned weights differ: {len(values)} != {len(learned_weight)}"
        )
    scale = math.sqrt(sum(value * value for value in values) / len(values) + 1.0e-6)
    return [
        value / scale * (1.0 + weight)
        for value, weight in zip(values, learned_weight)
    ]


def matmul(vector: list[float], matrix: list[float], columns: int) -> list[float]:
    return [
        sum(vector[row] * matrix[row * columns + column] for row in range(len(vector)))
        for column in range(columns)
    ]


def shifted_token(history: list[int], position: int, shift: int) -> int:
    if shift == 0:
        return history[position]
    source = position - shift
    if source < 0 or GEOMETRY["eos_token_id"] in history[source:position]:
        return GEOMETRY["eos_token_id"]
    return history[source]


def run_chunk(
    token_history: list[int],
    conv_history: list[list[float]],
    tokens: list[int],
    absolute_start: int,
    table: dict[str, list[int] | list[float]],
) -> tuple[list[float], list[int], list[list[float]]]:
    hidden_size = GEOMETRY["hidden_size"]
    hc_count = GEOMETRY["hc_count"]
    channels = hidden_size * hc_count
    heads_per_ngram = GEOMETRY["heads_per_ngram"]
    full_history = token_history + tokens
    current_normed: list[list[float]] = []
    current_gated: list[list[float]] = []
    input_embeddings: list[list[float]] = []

    for local_position, token in enumerate(tokens):
        absolute_position = absolute_start + local_position
        query = [
            (absolute_position * channels + lane) / 32.0
            for lane in range(channels)
        ]
        input_embeddings.append(query)
        history_position = len(token_history) + local_position
        ids: list[int] = []
        for ngram in range(2, GEOMETRY["ngram_size"] + 1):
            mixed = token * table["multipliers"][0]
            for shift in range(1, ngram):
                mixed ^= shifted_token(full_history, history_position, shift) * table["multipliers"][shift]
            start = (ngram - 2) * heads_per_ngram
            for head in range(start, start + heads_per_ngram):
                ids.append(mixed % table["head_vocab_sizes"][head] + table["head_offsets"][head])

        embedding = [table["ngram_embedding"][index] for index in ids]
        key = matmul(embedding, table["key_weights"], channels)
        value = matmul(embedding, table["value_weights"], hidden_size)
        gated: list[float] = []
        for stream in range(hc_count):
            begin = stream * hidden_size
            end = begin + hidden_size
            key_group = rms_norm(
                key[begin:end],
                table["norm_key"][begin:end],
            )
            query_group = rms_norm(
                query[begin:end],
                table["norm_query"][begin:end],
            )
            gate = sum(left * right for left, right in zip(key_group, query_group)) / math.sqrt(hidden_size)
            signed_root = math.copysign(math.sqrt(max(abs(gate), 1.0e-6)), gate)
            sigmoid = 1.0 / (1.0 + math.exp(-signed_root))
            gated.extend(sigmoid * lane for lane in value)
        current_gated.append(gated)
        normalized: list[float] = []
        for stream in range(hc_count):
            begin = stream * hidden_size
            end = begin + hidden_size
            normalized.extend(
                rms_norm(
                    gated[begin:end],
                    table["norm_conv"][begin:end],
                )
            )
        current_normed.append(normalized)

    all_conv = [
        conv_history[channel] + [row[channel] for row in current_normed]
        for channel in range(channels)
    ]
    hidden: list[float] = []
    for position, (query, gated) in enumerate(zip(input_embeddings, current_gated)):
        for channel in range(channels):
            convolved = sum(
                all_conv[channel][position + tap * GEOMETRY["conv_dilation"]]
                * table["conv_weights"][channel * GEOMETRY["conv_kernel"] + tap]
                for tap in range(GEOMETRY["conv_kernel"])
            )
            conv_silu = convolved / (1.0 + math.exp(-convolved))
            hidden.append(query[channel] + gated[channel] + conv_silu)

    context_len = GEOMETRY["ngram_size"] - 1
    history_len = (GEOMETRY["conv_kernel"] - 1) * GEOMETRY["conv_dilation"]
    next_tokens = full_history[-context_len:]
    next_conv = [channel[-history_len:] for channel in all_conv]
    return hidden, next_tokens, next_conv


def boundary_case(chunks: list[list[int]], table: dict[str, list[int] | list[float]]) -> dict[str, object]:
    channels = GEOMETRY["hc_count"] * GEOMETRY["hidden_size"]
    history_len = (GEOMETRY["conv_kernel"] - 1) * GEOMETRY["conv_dilation"]
    token_history = [GEOMETRY["eos_token_id"]] * (GEOMETRY["ngram_size"] - 1)
    conv_history = [[0.0] * history_len for _ in range(channels)]
    hidden: list[float] = []
    absolute_start = 0
    for chunk in chunks:
        chunk_hidden, token_history, conv_history = run_chunk(
            token_history, conv_history, chunk, absolute_start, table
        )
        hidden.extend(chunk_hidden)
        absolute_start += len(chunk)
    return {
        "chunks": chunks,
        "hidden_states": hidden,
        "token_history": token_history,
        "conv_history": [value for channel in conv_history for value in channel],
    }


def fixture() -> dict[str, object]:
    table = weights()
    tokens = [5, 7, GEOMETRY["eos_token_id"], 13]
    return {
        "provenance": {
            "qwen_release_commit": "69885871a64393807d988b27b1b5e380e8f28526",
            "model_config_revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
            "transformers_implementation_commit": "fc5c5bde8e656dad91cbf34e61940d984b1c7b91",
            "claim": "reference equations/config conformance with deterministic synthetic weights; not official checkpoint weight parity",
        },
        "authoritative_config": {
            "vocab_size": 248_320,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "hc_count": 4,
            "hidden_size": 2560,
            "ple_embed_dim": 2560,
            "ple_layer_ids": [2],
            "ple_conv_kernel_size": 4,
            "conv_dilation_equals_ngram_size": 3,
            "ngram_vocab_size_base": 20_000_000,
            "make_ngram_vocab_size_divisible_by": 128,
            "seed": 1234,
            "num_hidden_layers": 48,
        },
        "synthetic_geometry": GEOMETRY,
        "weights": table,
        "cases": {
            "full": boundary_case([tokens], table),
            "chunked": boundary_case([tokens[:2], tokens[2:]], table),
            "decode": boundary_case([[token] for token in tokens], table),
        },
    }


def rendered() -> bytes:
    document = json.dumps(
        fixture(),
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        separators=(",", ": "),
        sort_keys=True,
    )
    return (document + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--check", type=Path)
    destination.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = rendered()
    if args.check is not None:
        expected = args.check.read_bytes()
        if expected != output:
            raise SystemExit(
                f"{args.check} is stale; regenerate it with "
                f"{Path(__file__)} --output {args.check}"
            )
        return 0
    if args.output is not None:
        args.output.write_bytes(output)
        return 0
    # Bypass TextIOWrapper so Windows cannot translate canonical LF bytes to CRLF.
    sys.stdout.buffer.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
