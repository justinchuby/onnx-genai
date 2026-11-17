# Native IR pattern recognition

`onnx_runtime_ir::pattern` is a reusable **match-only** engine. It parses text
once, resolves named definitions to immutable plans, and matches an unchanged
`Graph` through a prepared `MatchSession`. There is no rewriting, partition
selection, constant folding, kernel dispatch, or C++/Python dependency.

## Text and interoperability

ONNX defines an [experimental textual model/function grammar][onnx-text].
That is the syntax foundation, **not** a standard pattern language. This module
implements a documented **subset** of the ORT pattern dialect at commit
`c992a5210bf5bf6a8f8ace36d8c6304ba70f1821`, exposed as
`pattern::ORT_REFERENCE_COMMIT`. It is not complete ORT-source compatibility,
nor an ONNX model parser. `parse_ort` and `parse_native` select the language
explicitly; unsupported forms are not guessed or silently ignored.

The reference contracts are the pinned [ORT documentation][ort-doc] and
[header][ort-header]. The [matcher source][ort-source] establishes injective
operator assignments (`BindOperator`) and deterministic frontier selection
(`MatchPattern`). The [upstream tests][ort-tests] pin optional-slot and global
annotation-alternative behavior. Fixtures under
`crates/onnx-runtime-ir/tests/fixtures/patterns/` preserve literal example text
with MIT attribution. Rust fixtures reproduce those published expectations;
they **do not** execute or build the upstream ORT implementation.

### Supported envelope

| Construct | Behavior |
|---|---|
| Inline `(x) => { a=Identity(x); return Neg(a) }` | Present named inputs, SSA assignments, explicit ordered returns, optional separators, `#` comments |
| Domains | `Identity`, `ai.onnx.Identity`, `ai.onnx::Identity` normalize to the default domain; `vendor.Op` and `vendor::Op` retain their domain |
| Libraries | `pattern Name(inputs) -> (outputs) { ... }`, forward references, positional calls/results, named aliases/groups and expression alternatives |
| Capture identity | Repeated tensor names require identical **scoped SSA values**, including output position; different tensor names may alias; distinct pattern operators must map to distinct actual nodes |
| Wildcard | Each positional `_` is an independent **present** value, never a missing slot |
| Optional input | `None` requires absence, including a truncated tail; `v:tensor[...] \| None` admits present or absent; repeated optional bindings agree including absence |
| Arity | Output count exact; no extra actual inputs beyond the declared fixed signature; missing declared trailing inputs must be optional/`None` |
| Constants | int64/float32 scalar and flat rank-one numeric lists, exact rank/dtype/value, immutable inline initializers and dense `Constant` forms; no arbitrary evaluation |
| Attributes | Typed scalar/string/numeric-list equality, borrowed captures, optional `type \| None` captures, repeated capture presence/content equality; unlisted attributes unconstrained |
| Tensor annotations | ONNX tensor element names, `tensor<T>`, scalar vs any-rank vs fixed/prefix rank, static/named/anonymous dimensions, per-alternative `const` |
| Type alternatives | First globally consistent combination in declaration/call order; failed combinations roll back; a rejecting predicate advances graph assignments/plans, not type combinations |
| `GRAPH` | Named pattern references, ordered formal prefix and exact returns, independent instances, complete body operator coverage (except validated literal Constant producers) |
| Whole graph | `graph_pattern` plus `match_graph`, including unused positional formals, independent output components, duplicate returns and input passthrough |
| Native extensions | Explicit `parse_native`: `*` operator, `...xs` ordered input sequences, `values[min:max]` / `slots[min:*]`, `[Body,Body]` GRAPH-list constraints |

Libraries require an operator-registry callback to disambiguate unresolved bare
calls from misspelled pattern names. This avoids an IR-to-schema dependency:

```rust,ignore
let library = PatternLibrary::parse_ort(source, &|domain, op| {
    schemas.contains_operator(domain, op) // supplied by the consumer
})?;
let pattern = library.pattern("QdqMatMul")?;
```

The callback runs during compilation only, not during graph matching. Qualified
operators do not need this lookup. Inline `Pattern` parsing, like the upstream
standalone parser, treats unresolved call names as operator names.

Named alternative definitions require identical ordered input/output labels and
annotations; use expression alternatives for different signatures. Interfaces,
`implements`, legacy port syntax, legacy instance-only multi-result calls,
replacement syntax, nested operator operands, and model/function text are not
supported. Factor nested operands into SSA assignments. Unsupported syntax
returns a `PatternError` with a category, UTF-8 byte span, and remedy.

Native variadics allow **one** undetermined input segment per operator:

```text
(head, xs:slots[0:*], tail) => {
  a = custom.First(head, ...xs, tail)
  return custom.Second(a, ...xs, tail)
}
```

`slots` preserves missing slots; `values` requires every element present.
The actual signature determines the length after fixed prefix/suffix subtraction.
Repeated sequences require identical ordered values/presence. `Match::sequence`
materializes graph-qualified captures explicitly. Sequence formals in composed
calls and whole-graph interfaces, output variadics, node/graph wildcard captures,
negative-pattern syntax and general producer-path expressions are deferred.
An operator `*` still has the exact output arity declared by its assignment and
must explicitly constrain all owned bodies.

## Using the core

```rust,ignore
let pattern = Pattern::parse_ort(r"
    (x, packed, scale, zero:tensor[...] | None) => {
        weights = DequantizeLinear(packed, scale, zero)
        return MatMul(x, weights)
    }
")?;
let packed = pattern.value_capture("packed")?; // typed, plan-specific key
let session = MatchSession::new(&graph)?;
let accepted = session.match_at_where(&pattern, root, MatchOptions::default(), |m| {
    let weights = m.captured(&packed)?.expect("required capture");
    // EP-owned dtype/layout/numerical eligibility, not part of graph structure.
    Ok(weights.value().dtype == DataType::Uint8)
})?;
```

See the compiled doctest on `pattern` and the executable:

```sh
cargo run -p onnx-runtime-ir --example pattern_qdq
```

`match_at` returns the first match at the specified live root. `visit_at` visits
source-order concrete plans, then deterministic assignments, and accepts an
owning result borrowing the graph; visitor `true` continues, `false` stops.
`find_matches` uses prepared domain/operator buckets, with root IDs ascending,
and deliberately permits overlapping results. It searches only the root scope;
body matching is explicit through graph-attribute references.

The last expanded operator is the anchor regardless of return order. Bound
outputs force precise producers first. For another connected sink, the matcher
uses the highest unmatched pattern operator's lowest eligible input, enumerating
the exact producer-output/consumer-input edge in ascending raw node ID order.
There is no operand permutation, longest-match search or algebraic equivalence.

Value/attribute keys obtained from one `Pattern` cannot be used with another;
clones share plan identity. `value`, `attribute`, `dimension`, `element_type`,
and `instance` diagnose unknown/wrong-kind names. Absent optional captures are
successful `None`, distinct internally from an unbound optional branch variable.
Returned matches own capture arrays and share immutable plans; no references
into reusable scratch can escape.

## Scopes, facts and safety

Preparation checks live identities, producer/use symmetry, graph interfaces,
opset imports and explicit cycles before building `GraphViewCache`. It indexes
names and owned bodies and resolves free source values to visible ancestor
definitions. Child raw IDs never alias parent IDs. A body formal is a fresh
local value, not its parent's actual argument. Local formals/initializers/produced
values stop lexical lookup. Free references need names and must resolve.

For `If(...,then_branch=Branch,else_branch=Branch)`, the two `Branch` instances
have independent values/dimensions. Header bindings beyond actual body formals
are structural leaves, not declarations of capture provenance. Use:

```rust,ignore
let branch = matched.instance("selected.then_branch")?;
let outer = branch.resolve_input("x", matched.graph())?;
```

Resolution checks the actual ancestor chain, including intermediate proxies;
body formals, local initializers and produced values cannot be promoted. All
owned `GRAPH`/`GRAPHS` bodies need explicit pattern constraints. Loop/Scan body
formals and returns match positionally; the engine does not infer recurrence,
iteration count, state semantics, axes or equivalent branches. No model-local
function is automatically expanded: inspect `graph().model_functions` in a
predicate or prepare a separate session for an explicitly selected body.

Like pinned ORT `MatchGraphPattern`, a pattern header can have more values than
the graph has formal inputs. The formal prefix binds positionally; remaining
header values must bind through structure, for example to captures or local
initializers. They are not additional required graph formals. An unused,
unbound extra header value prevents a whole-graph match.

The existing IR can store both an inline `Attribute::Graph` and an indexed
`Graph::subgraphs` copy. Matching uses the indexed body when present, as an
explicit authoritative execution view; it does not traverse both. Native
`GRAPHS` uses attribute-list order, not formatted index-name parsing. Eliminating
duplicate ownership is future IR work, not a second store added by this module.

Capture references contribute dependencies from ancestor producers to containing
owners, projected across nesting. Preparation rejects capture-induced cycles,
including an owner capturing its own output. Matching is not a schedule; root
enumeration uses documented raw-ID order, not an assertion that explicit-edge
topological ordering accounts for captures.

Attributes are read from the provided map. **No schema defaults are inserted.**
To compare with ORT's resolved graph, the adapter must materialize the same
defaults/types in advance. `const` and literals exclude initializer/input overlap.
Literal reads validate recognized inline storage. External bytes are never
loaded implicitly: a literal requiring `WeightRef::External` returns
`MissingMetadata`; materialize through the loader first. Sparse constant
evaluation is unsupported. Numerical float literals treat signed zeros as equal;
repeated float attribute captures compare their stored bits, matching normalized
protobuf-content semantics for the represented fields.

Concrete tensor constraints require known metadata; placeholder dtype/shape is
not proof. Rank-specific constraints reject unknown rank. Repeated dimension
names require known equal static extents or equal nonempty symbolic names.
Unknown singleton dimensions remain unbound. These are constraints, not shape
inference.

### Structural recognition versus deletion boundaries

Default `BoundaryPolicy::Structural` permits intermediate external users.
`OrtBoundary` also requires non-returned operator outputs not to be graph returns,
locally consumed outside the match, or lexically captured by unmatched owners.
Body operators remain separate from parent footprints. For parity comparisons
with upstream `MatchAt`, request this policy explicitly.

This is **not a rewrite authorization or complete ORT optimizer parity**.
It assumes the supplied IR contains all visibility information; an ORT adapter
must represent otherwise external outputs/uses. Provider/layer annotations and
replacement-cycle policy are not represented here. EPs retain alias/lifetime,
side-effect, opset, numerical, shape/layout and kernel-support guards.

### Representation limits

The matcher cannot reconstruct source information the loader discarded:
optional output holes, full non-tensor value types/unknownness, sparse
initializers, byte-string tensor elements and complete function overload/default/
attribute-reference binding. The supported core expects **present dense outputs**
and faithfully represented initializer/input overlap; do not use a lossy adapter
to prove absence, immutability or whole-ONNX conformance. Generic attribute captures
can borrow all represented variants. Repeated GRAPH/GRAPHS, TYPE_PROTO and
SPARSE_TENSORS equality is explicitly unsupported; graph constraints or bounded
consumer predicates are required. No claim of all-ONNX or full ORT compatibility
is made.

## Errors, budgets and costs

Ordinary nonmatches are `Ok(None)` or an empty visitor traversal. Syntax,
unsupported features, malformed patterns/IR, unavailable required payloads,
capture misuse, predicate failures and exhausted budgets are distinct errors.
Callbacks return `Result<bool, PatternError>`: false rejects a candidate with
fresh bindings for the next one; errors propagate. The library cannot bound
arbitrary callback work; callbacks must remain read-only and bounded.

Compilation bounds source to 1 MiB / 65,536 tokens, 256 definitions, 64 concrete
plans/nodes/type combinations/dimensions, and 256 values/annotation occurrences/
expanded calls. Definition and graph nesting are limited to **32**, expression
nesting to 64 and combined assignment/body search depth to 96. These deliberately
stricter depth limits are not the upstream 64-definition ceiling.
`MatchBudget` bounds shared steps, completed candidates, and tensor/attribute/
sequence comparison bytes across an entire public call, including nested bodies.
Tensor comparison accounting includes names, dimensions, string payloads and
per-string metadata, not just numeric data buffers.
There is no recursive user pattern or unbounded variadic splitting.

Plans precompute producer slots. Sessions reuse the existing dense producer/use
cache and add root/name/body/capture indexes. Forced producer paths are bounded
by pattern size; multiple connected sinks can require combinatorial assignment
search and annotation choices can have product cost. Budget exhaustion is an
error, never evidence of a negative match.

This foundation uses per-candidate capture arrays and state cloning on consumer
branches for transactional rollback. Owning matches, nested captures, maps and
`find_matches` allocate. It **does not claim allocation-free matching** or an
advantage over handwritten EP matchers. Reusable undo-log scratch and a shared
multi-rule decision DAG are possible measured optimizations, not parallel
execution backends shipped here.

For a reproducible small measurement:

```sh
cargo build --release -p onnx-runtime-ir --example pattern_qdq
scripts/hostlock.sh run --owner ir-patterns --reason "IR matcher microbenchmark" -- \
  cargo run --quiet --release -p onnx-runtime-ir --example pattern_qdq -- --bench
```

The example asserts hit and same-root-op miss controls, then reports five batches
of 2,000 synchronous calls after 100 warmups: parse/verify, graph preparation,
warm owned-result match, same-op nonmatch and an empty indexed root bucket.
It includes return-value destruction, excludes graph construction, and does not
instrument allocations. Record compiler, hardware, host-lock provenance and
contention alongside results. These synthetic three-node costs do not measure
MLX partition/compile time, many-rule scaling, GPU speed or an EP speedup.

On the initial macOS implementation host, the example and release benchmark
build passed, but the guarded timing command was refused: `hostlock.sh` requires
Linux `/proc` and exits on Darwin without taking a lock. No timing measurements
or speedup claim are attached to this change.

[onnx-text]: https://onnx.ai/onnx/repo-docs/Syntax.html
[ort-doc]: https://github.com/microsoft/onnxruntime/blob/c992a5210bf5bf6a8f8ace36d8c6304ba70f1821/docs/Pattern_Rewriter.md
[ort-header]: https://github.com/microsoft/onnxruntime/blob/c992a5210bf5bf6a8f8ace36d8c6304ba70f1821/onnxruntime/core/optimizer/pattern_rewriter.h
[ort-source]: https://github.com/microsoft/onnxruntime/blob/c992a5210bf5bf6a8f8ace36d8c6304ba70f1821/onnxruntime/core/optimizer/pattern_rewriter.cc#L303-L650
[ort-tests]: https://github.com/microsoft/onnxruntime/blob/c992a5210bf5bf6a8f8ace36d8c6304ba70f1821/onnxruntime/test/optimizer/pattern_rewriter_test.cc#L2718-L3194
