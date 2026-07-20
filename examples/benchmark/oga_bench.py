"""Decode-throughput benchmark for onnxruntime-genai (oga), for apples-to-apples
comparison against onnx-genai's `profile_decode`.

Usage:
    python oga_bench.py <model_dir> [prompt] [max_new_tokens]

Environment:
    OGA_WARMUPS  number of warmup runs (default 2)
    OGA_RUNS     number of timed runs   (default 3)
    OGA_RAW      set to 1 to encode the bare prompt WITHOUT the chat template.
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
# Apply the model's chat template by default so this matches profile_decode
# (which chat-templates the prompt). Set OGA_RAW=1 to encode the bare prompt.
use_chat_template = os.environ.get("OGA_RAW", "0") != "1"

print(f"oga {og.__version__} model={model_path}")
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

def one_run():
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=prompt_len + max_new, do_sample=False)
    gen = og.Generator(model, params)
    gen.append_tokens(input_tokens)
    n = 0
    t0 = time.perf_counter()
    while not gen.is_done() and n < max_new:
        gen.generate_next_token()
        n += 1
    dt = time.perf_counter() - t0
    seq = gen.get_sequence(0)
    return n, dt, seq

for _ in range(warmups):
    one_run()

total_tok = 0
total_dt = 0.0
last_seq = None
for _ in range(runs):
    n, dt, seq = one_run()
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
