import os, sys, time
import onnxruntime_genai as og

model_dir = sys.argv[1]
prefill_lens = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["512"])]
decode_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 128
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 3

t0 = time.time()
model = og.Model(model_dir)
try:
    tok = og.Tokenizer(model)
    _enc = lambda s: tok.encode(s)
except Exception as e:
    # og's builtin tokenizer may not support this model's tokenizer class
    # (e.g. ChatGLM4Tokenizer). Throughput does not depend on tokenizer
    # semantics, so fall back to the HF `tokenizers` fast lib on tokenizer.json.
    from tokenizers import Tokenizer as HFTok
    _ht = HFTok.from_file(os.path.join(model_dir, "tokenizer.json"))
    _enc = lambda s: _ht.encode(s).ids
    print(f"og.Tokenizer unavailable ({str(e)[:60]}); using tokenizers.json fallback", flush=True)
print(f"loaded og.Model + tokenizer in {time.time()-t0:.1f}s", flush=True)

base = _enc("The quick brown fox jumps over the lazy dog and then")
def make_ids(n):
    out = []
    while len(out) < n:
        out.extend(base)
    return out[:n]

def run_prefill(n):
    walls = []
    ids = make_ids(n)
    for _ in range(iters + 1):  # +1 warmup
        p = og.GeneratorParams(model)
        try: p.set_search_options(max_length=n + 1, do_sample=False)
        except Exception: pass
        g = og.Generator(model, p)
        s = time.time()
        g.append_tokens(ids)
        walls.append(time.time() - s)
        del g
    walls = walls[1:]
    m = min(walls)
    return m, n / m

def run_decode(prompt_n, ntok):
    ids = make_ids(prompt_n)
    p = og.GeneratorParams(model)
    try: p.set_search_options(max_length=prompt_n + ntok + 4, do_sample=False)
    except Exception: pass
    g = og.Generator(model, p)
    g.append_tokens(ids)
    steps = []
    for _ in range(ntok):
        s = time.time()
        g.generate_next_token()
        steps.append(time.time() - s)
        if g.is_done(): break
    del g
    steps.sort()
    skip = max(1, len(steps)//8)
    win = steps[skip:] if len(steps) > skip else steps
    med = win[len(win)//2]
    return med, 1.0/med

print("== ORT-genai (chatglm builder int4) prefill sweep ==", flush=True)
for n in prefill_lens:
    mn, tps = run_prefill(n)
    print(f"  L={n:<5} min {mn*1000:9.3f} ms ({tps:9.1f} tok/s)", flush=True)

print("== ORT-genai decode steady tok/s ==", flush=True)
med, tps = run_decode(16, decode_tokens)
print(f"  decode: median step {med*1000:.3f} ms ({tps:.2f} tok/s)", flush=True)
