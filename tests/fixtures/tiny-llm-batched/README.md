# tiny-llm-batched

`tiny-llm-sharedbuffer` with one declaration changed: its KV group says
`aliasing: permitted` rather than `forbidden`, so the package offers the
runtime the fixed-capacity shared buffer that a row-major batch decodes into.

That single line is what makes the package batchable, and it is the package's
statement rather than the runtime's inference — `Engine::batching_capability()`
reports `supports_batching()` here and not for the `forbidden` sibling, on the
same graph, the same tokenizer and the same execution provider.

It exists so the authored continuous-batch iteration has a hermetic package to
run on. The graph, weights and tokenizer are byte-identical to the sibling's, so
this directory carries no generator of its own: regenerate the sibling with
`tests/fixtures/tiny-llm-sharedbuffer/generate_tiny_llm_sharedbuffer.py`, copy
the result here, and re-apply the one-line `aliasing` change. A generator copied
into this directory would rewrite `manifest.json` and silently drop the
`derived_from` record of where the package came from.
