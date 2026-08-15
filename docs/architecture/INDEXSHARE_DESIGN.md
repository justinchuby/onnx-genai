# IndexShare selected-token attention — frozen v1

**Status:** frozen v1
**Operator:** `pkg.nxrt::IndexShare`, version 1

## Boundary and schema

Version 1 consumes exporter-computed selected indices and owns selected-token
attention plus explicit KV-cache I/O. It does not own top-k selection or the
index-key cache; those are deferred to a later version.

### Inputs

Inputs are positional:

| # | Name | Type and shape | Required |
|---|---|---|---|
| 0 | `query` | f32 `[B, N, S_q, H]` | yes |
| 1 | `key` | f32 `[B, N_kv, S_kv, H]` | yes |
| 2 | `value` | f32 `[B, N_kv, S_kv, H]` | yes |
| 3 | `past_key` | f32 `[B, N_kv, S_past, H]` | no |
| 4 | `past_value` | f32 `[B, N_kv, S_past, H]` | no |
| 5 | `selected_indices` | int32/int64 `[B, I, S_q, K]` | yes |
| 6 | `attention_bias` | f32, broadcastable to `[B, N, S_q, S_past+S_kv]` | no |

`I` is either `N` (per-query-head indices) or `1` (indices shared by all query
heads). Omitted optional input slots use `DataType::Undefined`. `past_key` and
`past_value` are supplied as a pair.

### Outputs

| # | Name | Type and shape | Required |
|---|---|---|---|
| 0 | `output` | f32 `[B, N, S_q, H]` | yes |
| 1 | `present_key` | f32 `[B, N_kv, S_past+S_kv, H]` | no |
| 2 | `present_value` | f32 `[B, N_kv, S_past+S_kv, H]` | no |

The output arity is either one or three. Present K/V is the dense concatenation
`past || current`; the two cache outputs are requested together.

### Attributes

- `num_heads` (required integer, positive)
- `kv_num_heads` (optional integer, defaults to `num_heads`)
- `scale` (optional float, defaults to `1/sqrt(H)`)

No softcap or sparse-mixer knobs are part of v1.

## Frozen semantics

For every `(batch, query head, query position)`, IndexShare attends only to the
dense-cache positions named by `selected_indices`. Query heads map to KV heads
in contiguous GQA groups, requiring `num_heads % kv_num_heads == 0`.

Valid indices are strictly increasing, unique, in range
`[0, S_past+S_kv)`, and therefore already in dense-cache order. `-1` is allowed
only as trailing padding. A row containing only `-1` is invalid. Duplicate,
decreasing, out-of-range, non-trailing-sentinel, and all-empty rows fail
execution with a clear error. Index tensor data is deliberately validated at
execution because claim time has metadata only.

Scores use the deterministic f32 order of standard CPU attention:
`(Q * sqrt(scale)) dot (K * sqrt(scale))`, followed by additive
`attention_bias`. Softmax visits selected positions in ascending dense-cache
order, and each output component accumulates `probability * value` in that same
fixed order.

## Dense additive-mask oracle

The normative oracle constructs a full dense additive selection mask with `0`
at selected cache positions and `-inf` everywhere else, adds the optional
causal/padding `attention_bias`, and runs standard attention over the full dense
present K/V cache. IndexShare gathers the selected subset instead of scanning
the full cache, but the fixed ordering requires exact f32 CPU equality with
that oracle, including a zero output for a fully bias-masked row.

## Claim-time contract

Claim validates attributes, positional arity, presence, dtypes, ranks, and all
relationships provable from static dimensions. It performs no tensor-data
reads or tensor allocation. Dynamic dimensions are not rejected merely because
they are unknown. Runtime validates concrete shapes and all index values.

## Exporter reconciliation

The exact private `pkg.nxrt` operator name and version must be reconciled with
Mobius PR #404 when that exporter change merges. This document freezes the
runtime v1 contract; reconciliation must not silently change its semantics.
