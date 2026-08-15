"""Decode-throughput benchmark for onnxruntime-genai (oga), for apples-to-apples
comparison against onnx-genai's `profile_decode`.

Usage:
    python oga_bench.py <model_dir> [prompt] [max_new_tokens]

Environment:
    OGA_WARMUPS  number of warmup runs (default 2)
    OGA_RUNS     number of timed runs   (default 3)
    OGA_THREADS  ORT intra-op thread count (inter-op is fixed at 1)
    OGA_DECODE_SKIP  emitted tokens excluded before steady decode timing (default 8)
    OGA_RAW      set to 1 to encode the bare prompt WITHOUT the chat template.
    OGA_EXPECTED_TOKEN_IDS  JSON array of token IDs required for exact parity.
                 By default the prompt is chat-templated so the input matches
                 `profile_decode` (which chat-templates too); feeding oga a raw,
                 untemplated prompt makes it decode a different base-completion
                 sequence and yields an unfair comparison.
"""

import sys, time, os, json
import onnxruntime_genai as og

if len(sys.argv) < 2:
    sys.exit("usage: python oga_bench.py <model_dir> [prompt] [max_new_tokens]")

model_path = sys.argv[1]
prompt = sys.argv[2] if len(sys.argv) > 2 else "Explain the theory of relativity in simple terms."
max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 128
warmups = int(os.environ.get("OGA_WARMUPS", "2"))
runs = int(os.environ.get("OGA_RUNS", "3"))
threads = int(os.environ.get("OGA_THREADS", "0"))
decode_skip = int(os.environ.get("OGA_DECODE_SKIP", "8"))
expected_tokens_json = os.environ.get("OGA_EXPECTED_TOKEN_IDS")
expected_tokens = json.loads(expected_tokens_json) if expected_tokens_json else None
# Apply the model's chat template by default so this matches profile_decode
# (which chat-templates the prompt). Set OGA_RAW=1 to encode the bare prompt.
use_chat_template = os.environ.get("OGA_RAW", "0") != "1"

print(f"oga {og.__version__} model={model_path} threads={threads or 'default'}")
if threads:
    config = og.Config(model_path)
    config.overlay(json.dumps({
        "model": {
            "decoder": {
                "session_options": {
                    "intra_op_num_threads": threads,
                    "inter_op_num_threads": 1,
                }
            }
        }
    }))
    model = og.Model(config)
else:
    model = og.Model(model_path)
tokenizer = og.Tokenizer(model)

# Chat-template the prompt to match profile_decode's instruct-mode input.
if use_chat_template:
    messages = json.dumps([{"role": "user", "content": prompt}])
    templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_tokens = tokenizer.encode(templated)
else:
    input_tokens = tokenizer.encode(prompt)
prompt_len = len(input_tokens)
print(f"prompt_tokens: {prompt_len}")

def one_run():
    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=prompt_len + max_new,
        min_length=prompt_len + max_new,
        do_sample=False,
        repetition_penalty=1.0,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )
    gen = og.Generator(model, params)
    t0 = time.perf_counter()
    gen.append_tokens(input_tokens)
    n = 0
    token_times = []
    while not gen.is_done() and n < max_new:
        gen.generate_next_token()
        n += 1
        token_times.append(time.perf_counter() - t0)
    dt = time.perf_counter() - t0
    seq = gen.get_sequence(0)
    return n, dt, seq, token_times

for _ in range(warmups):
    one_run()

total_tok = 0
total_dt = 0.0
last_seq = None
reference_tokens = None
for run in range(1, runs + 1):
    n, dt, seq, token_times = one_run()
    generated_tokens = [int(token) for token in seq[prompt_len:]]
    if reference_tokens is None:
        reference_tokens = generated_tokens
    elif generated_tokens != reference_tokens:
        sys.exit("greedy decode was not deterministic across measured runs")
    if expected_tokens is not None and generated_tokens != expected_tokens:
        sys.exit(
            "generated token IDs diverged from OGA_EXPECTED_TOKEN_IDS:\n"
            f"expected: {expected_tokens}\n"
            f"actual:   {generated_tokens}"
        )
    if len(token_times) <= decode_skip:
        sys.exit(f"only {len(token_times)} tokens emitted; OGA_DECODE_SKIP={decode_skip}")
    prefill_ms = token_times[0] * 1000.0
    decode_tokens = len(token_times) - decode_skip
    decode_wall = token_times[-1] - token_times[decode_skip - 1]
    decode_tps = decode_tokens / decode_wall
    print(
        f"steady_run {run}: prefill={prefill_ms:.3f} ms "
        f"decode_tokens={decode_tokens} decode_wall={decode_wall*1000:.3f} ms "
        f"decode={decode_wall/decode_tokens*1000:.3f} ms/token "
        f"throughput={decode_tps:.2f} tok/s"
    )
    total_tok += n
    total_dt += dt
    last_seq = seq

if total_tok == 0 or total_dt == 0.0:
    print(f"wall: {total_dt*1000:.3f} ms over {total_tok} tokens ({runs} run(s)) -> no tokens decoded, nothing to measure")
else:
    tps = total_tok / total_dt
    print(f"wall: {total_dt*1000:.3f} ms over {total_tok} tokens ({runs} run(s)) -> {tps:.2f} tok/s, {total_dt/total_tok*1e6:.2f} us/token")
try:
    text = tokenizer.decode(last_seq[prompt_len:])
    print("--- generated text (coherence check) ---")
    print(text)
except Exception as e:
    print("decode failed:", e)
print(f"generated_token_ids: {reference_tokens}")
