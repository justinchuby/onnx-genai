# Chunked prefill, and why the query axis has to be padded

Prefill is the forward pass that consumes a prompt. Decode is the forward pass
that produces one token. They run the same graph and differ in exactly one
number: how many rows the query axis has. Prefill hands the model the whole
prompt at once (`M` rows for an `M`-token prompt); decode hands it one row.

That single number turns out to drive both the peak memory of a request and the
number of kernels the runtime has to compile. This document explains why prefill
can be split into chunks at all, what chunking buys, and why chunking on its own
left a hole that query-axis padding had to close.

## Why a prompt can be prefilled in pieces

The obvious way to read a 4000-token prompt is to run one forward pass with
`M = 4000`. Chunked prefill instead runs eight passes of `M = 512`, feeding the
prompt in slices. The output is bit-for-bit the same. Three properties make that
true, and all three have to hold.

### 1. The KV cache is append-only

Attention at position `i` reads keys and values for every position `≤ i`. Those
keys and values depend only on the token at each position and the layer's
weights — not on anything that comes later. So the cache is written strictly
left to right, and a position's entry never changes once written.

When chunk `k` runs, positions `[0, k·W)` are already in the cache from earlier
chunks. Chunk `k` computes keys and values for its own `W` positions and appends
them at `[k·W, (k+1)·W)`. Nothing earlier is revisited, nothing earlier is
rewritten. The cache after chunk `k` is byte-identical to the prefix of the cache
a single whole-prompt pass would have produced.

This is the answer to "why doesn't a chunk need all the KV to participate?" — it
*does* attend over all the KV written so far. What it does not need is to
*recompute* it. The earlier chunks' KV is sitting in the cache being read; only
the chunk's own `W` rows of query are new work.

### 2. The causal mask makes the split invisible

A decoder-only model masks position `i` from attending to any position `> i`.
Take the whole-prompt pass and the chunked passes side by side: for a query at
position `p` in chunk `k`, the whole-prompt pass lets it attend to `[0, p]`, and
the chunked pass lets it attend to `[0, k·W) ∪ [k·W, p]` — the same set. The
positions the chunked pass has not computed yet are exactly the positions the
mask would have removed anyway.

So chunking does not approximate anything. It is not a quality/speed trade. The
arithmetic is identical; only the order in which it is issued changes.

### 3. The cache tensor's shape does not move

If the KV tensor were resized to the exact sequence length after every chunk,
every chunk would present a different shape to every attention node, and the
runtime would treat each as a new kernel. `GroupQueryAttention` avoids this by
separating capacity from length: `past_key`/`past_value` are bound at their
*physical capacity* for the whole generation, and the true number of valid
positions arrives in a separate `seqlens_k` input. `onnx-runtime-session` encodes
that fact directly — see `kernel_input_uses_physical_capacity` in
`crates/onnx-runtime-session/src/executor/geometry.rs`, which reports `true` for
GQA's inputs 3 and 4.

The practical consequence is worth stating plainly: **growing the KV cache does
not change any kernel's input shapes.** A generation that runs to 4000 tokens
compiles no more kernels than one that stops at 400.

## What chunking is for

Prefill attention costs `O(M × total)` work and, more importantly, needs
`O(M × total)` of scratch to hold the attention scores. With `M = 4000` that
scratch is enormous, and it is a transient spike — allocated for one pass, freed
after. A single 30B-class model can be sized to fit its weights and its KV cache
comfortably and still fail on the prefill spike alone.

Capping `M` at a chunk width `W` caps the spike at `O(W × total)`. The cost is a
handful of extra kernel launches and a few extra passes over the weights. The
model declares the width it wants in its inference metadata:

```yaml
runtime_configurable:
  chunked_prefill:
    chunk_size: 512
```

In this repository, `NativeDecodeSession::set_prefill_chunk_size` picks that up
and `decode_argmax` (`crates/onnx-genai-engine/src/native_decode/backend.rs`)
slices the prompt with `token_ids.chunks(chunk)` when it is longer than one
chunk.

## The hole chunking left

Chunking fixes `M = W` for every chunk except the last one, which gets the
remainder. A 1137-token prompt with `W = 512` runs `512, 512, 113`. A 1200-token
prompt runs `512, 512, 176`. The remainder is whatever the prompt length happens
to be modulo the chunk width — which is to say, effectively a fresh number for
every request.

That matters because the kernel cache is keyed by node *and input shapes*
(`KernelKey` in `crates/onnx-runtime-session/src/executor/kernel_cache.rs`). A
query width nobody has run before misses on every node in the graph, and for a
30B decoder that is on the order of 890 kernels compiled per request, forever.
Worse, each compiled kernel owns device scratch, so the cache is also VRAM that
the resource governor never sees — the unbounded growth tracked as issue #1362.

The cache bounds itself by keeping only the most recent variants of each node and
evicting the rest. But a bound only helps if the working set fits inside it, and
a working set with a fresh remainder width per request never does. Measured on
`muse-glimmer-30b-int4` serving five assorted prompt lengths twice:

| forward | query rows | kernels compiled | cumulative evictions |
| --- | --- | --- | --- |
| pass 1 | 104 | 1105 | 0 |
| pass 1 | 146 | 888 | 0 |
| pass 1 | 311 | 890 | 52 |
| **pass 2** | **146** | **888** | **5688** |
| **pass 2** | **311** | **888** | **7045** |
| **pass 2** | **37** | **888** | **9759** |

The second pass over the *same prompt lengths* recompiled just as much as the
first, and evictions were still climbing linearly. The cache was thrashing, not
caching.

## Query-axis padding

The fix mirrors what GQA already does on the KV axis. GQA does not resize the
cache to the exact sequence length; it runs at a fixed capacity and carries the
true length beside it. Prefill can do the same on the query axis: run at a
rounded-up width, and carry the true row count beside it.

`prefill_query_width` (in `crates/onnx-genai-engine/src/native_decode/cuda.rs`)
rounds a forward's row count up to one of eight evenly spaced steps below the
chunk width — `{64, 128, 192, 256, 320, 384, 448, 512}` for a 512-wide chunk. The
extra rows are filled by repeating the last real token, and any supplied
per-token port (`inputs_embeds`, and anything else shaped `[1, rows, …]`) is
widened the same way.

The padded rows are wasted arithmetic, but they cannot be wrong:

- **They cannot be read.** The causal mask stops a real row at absolute position
  `p` from attending past `p`, and every padded row sits after every real one.
- **Their logits are dropped.** The result is truncated back to the real row
  count before the caller sees it.
- **Their KV is erased.** After the forward, the session rewinds to
  `past_len + real_rows`, which zeroes the attention mask's tail and rolls the
  cache's logical length back. The next forward writes over the padded entries.

Two cases refuse padding rather than risk it. A decoder with recurrent or
convolutional state is excluded outright: that state is not masked and not
addressed by a logical length, so a duplicated row advances it with no way to
take the step back. And a supplied port that is not shaped `[1, rows, …]` is not
recognizably per-token, so the whole plan is dropped instead of inventing rows
for it. There is also a runtime self-check: if a padded forward returns fewer
logits rows than it was given query rows, that decoder reduces the query axis
internally, its rows cannot be mapped back to input positions, and padding is
disabled for the session before the forward is redone as asked.

Set `ONNX_GENAI_PREFILL_QUERY_PADDING=0` to turn it off.

## Padding alone was not enough

Bounding the width set only helps if the kernel cache's per-node bound is at
least as large as the set. It was 4, chosen when the working set was believed to
be "a chunk shape, its remainder, and the decode shape". With eight ladder steps
plus the single-token decode shape, 4 still thrashed:

| configuration | distinct widths | pass-2 recompiles | evictions after 14 forwards |
| --- | --- | --- | --- |
| no padding | unbounded | ~888 per forward | 9759, still climbing |
| padding, bound 4 | 6 | ~888 per forward | 8402, still climbing |
| padding, bound 10 | 6 | **0** | **156, flat** |

So the two changes are one change. Padding is what makes the width set finite and
knowable; raising the bound to cover that set — the ladder, plus the decode
shape, plus one for a forward too wide to pad — is what makes the finiteness pay.
`DEFAULT_VARIANTS_PER_NODE` is now 10 for that reason.

End-to-end latency was unchanged within noise across prompts of 40 to 900 tokens,
and greedy output was byte-identical with padding on and off. The win is not
speed; it is that a long-running server stops compiling kernels and stops
accumulating kernel scratch it never accounts for.

## Choosing the step count

`PREFILL_QUERY_WIDTH_STEPS = 8` balances two costs that pull opposite ways.
Fewer steps means fewer shapes to cache but more duplicated arithmetic — rounding
to powers of two instead sends 311 rows to 512, and the 65% of wasted work that
buys measurably hurt mid-length prompts (a 200-word prompt went from 4.9 s to
7.0 s). More steps means less waste but a larger working set, which pushes the
per-node bound — and the kernel scratch behind it — up. Eight steps caps the
waste at one step (≤64 rows) while keeping the bound at a number small enough to
be uninteresting.
