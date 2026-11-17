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
| Interfaces | `interface Unary(x:float[N]) -> (y:float[N])`, `pattern Copy implements Unary { ... }`; non-callable tensor contracts, conjunctive annotations and present outputs |
| Legacy library forms | `(port=local)` returns, scoped `p.port`, one-name multi-result instances, discovered inputs after `implements I: (...) ->`, and `(First \| Second)(x)` |
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

Named alternative definitions require the same interface (or none) and identical
ordered input/output labels and annotations; use expression alternatives for
different signatures. `[]` and `[...]` normalize to the same any-rank constraint.
Result contracts are reconciled without adding an annotation occurrence to
another alternative's actual plan.

Replacement syntax and model/function text are not supported. Nested **pattern
calls through assigned results** are supported; operator/call expressions inside
positional operands are rejected by both this frontend and the pinned ORT parser.
Factor them into SSA assignments. Unsupported syntax
returns a `PatternError` with a category, UTF-8 byte span, and remedy.

For example, these library forms share the same execution core:

```text
interface Pair(x) -> (left, right)
pattern split implements Pair {
  a, b = Split(x)
  return (right=b, left=a)
}
pattern join(x) {
  p = split(x)
  return Add(p.left, p.right)
}
```

Interface return mappings follow the declared port order, not the mapping's
textual order. As in pinned ORT, an ordinary explicit `->` output signature
consumes returned values positionally, even with legacy `(label=value)` syntax;
prefer ordinary positional returns there. A one-name call to a multi-result
pattern creates an **instance only**:
`p` cannot be used as a tensor, annotated or rebound; use its named ports or
`a,b=split(x)`. Calls to single-result patterns expose both the positional tensor
and scoped result aliases. Unnamed returned operator calls export `output`
inside named library bodies; anonymous inline patterns do not invent that alias.

An implementation may declare extra discovered captures:
`pattern scale implements Unary: (alpha:float[N]) -> { y=Mul(x,alpha);return y }`.
Call it with only `scale(x)`; `alpha` is discovered structurally and available
under the call's scope. Extra attribute captures are also supported. Separate
instances keep independent values, attributes, dimensions and type variables.
Interface inputs cannot become attributes, and interface/declared outputs
cannot become absent through an optional alias.

The portable call-argument profile is named values/scoped ports and tensor `_`;
the frontend also permits literal/`None` value actuals as a documented
superset, not a promise that upstream `PatternCall` accepts them. This is native
Rust match-language profile, not the C++ binding-view ABI: definition/member
selection-chain introspection, computed attribute mutation and provider/layer
metadata are not reproduced.

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
`GRAPHS` uses declared attribute-list order and the matching indexed body when
present; indexed copies are not visited again as extra bodies. Eliminating
duplicate ownership is future IR work, not a second store added by this module.
Indexed locations use `GraphAttribute::single(name)` or
`GraphAttribute::element(name, index)`, never a formatted string: a literal
GRAPH named `bodies[0]` cannot collide with element zero of GRAPHS `bodies`.

Capture references contribute dependencies from ancestor producers to containing
owners, projected across nesting. Preparation rejects capture-induced cycles,
including an owner capturing its own output. Matching is not a schedule; root
enumeration uses documented raw-ID order, not an assertion that explicit-edge
topological ordering accounts for captures.

Attributes are read from the provided map. **No schema defaults are inserted.**
Constraining an unresolved function `ref_attr_name` returns `MissingMetadata`;
bind the function's explicit/default attributes first, or inspect the raw
reference metadata without claiming its runtime value.
To compare with ORT's resolved graph, the adapter must materialize the same
defaults/types in advance. `const` and literals exclude initializer/input overlap.
Literal reads validate recognized inline storage. External bytes are never
loaded implicitly: a literal requiring `WeightRef::External` returns
`MissingMetadata`; materialize through the loader first. Sparse constant
evaluation is unsupported. Numerical float literals treat signed zeros as equal;
repeated float attribute captures compare their stored bits, matching normalized
protobuf-content semantics for the represented fields.

Concrete tensor constraints require known dense-tensor metadata; an unannotated
value is not implicitly a tensor. Source values with missing or non-tensor type
descriptors have `DataType::Undefined`, not a fabricated float32 scalar.
Rank-specific constraints reject unknown rank. Repeated dimension
names require known equal static extents or equal nonempty symbolic names.
Unknown singleton dimensions remain unbound. These are constraints, not shape
inference.

### Structural recognition versus deletion boundaries

Default `BoundaryPolicy::Structural` permits intermediate external users.
`OrtBoundary` also requires non-returned operator outputs not to be graph returns,
locally consumed outside the match, or lexically captured by unmatched owners.
Every unproduced pattern handle, including anonymous wildcard call actuals, must
also bind outside the matched operator footprint under that policy.
Body operators remain separate from parent footprints. For parity comparisons
with upstream `MatchAt`, request this policy explicitly.

This is **not a rewrite authorization or complete ORT optimizer parity**.
It assumes the supplied IR contains all visibility information; an ORT adapter
must represent otherwise external outputs/uses. Provider/layer annotations and
replacement-cycle policy are not represented here. EPs retain alias/lifetime,
side-effect, opset, numerical, shape/layout and kernel-support guards.

### Source fidelity versus execution and pattern semantics

The IR shares the existing generated ONNX schema with the loader instead of
maintaining a second, incomplete recursive type language. `Value::declared_type`
retains the declared `TypeProto`: tensor, sparse tensor, sequence, map, optional,
opaque or explicitly unknown. Missing rank differs from scalar rank; dimensions
retain unknown/empty-symbol presence and denotations. `dtype`/`shape` are separate
dense-tensor inference facts, never replacements for a container descriptor.
TYPE_PROTO attributes use the same schema messages.

Use `onnx_runtime_loader::load_model_ir_bytes` (or its path/weight variants) for
structural analysis and round trips. These source-preserving entry points do not
run shape inference or expand local functions. Existing execution loaders
deliberately lower functions and infer shapes; they are not source-preserving
transforms. Model-level metadata is still supplied separately to the encoder.
`onnx-std` model loading uses this source path too; it no longer silently
substitutes fake graph inputs for sparse initializers. Its raw function checker
can validate declarations before strict IR construction.

Local functions retain their exact `(domain, name, overload)` identity, own
imports, formal/default/reference attributes and canonical `FunctionProto`
declaration. That declaration is the serialization source; the derived body is
an execution/analysis view, not a function-rewrite API. Calls retain overload and
raw attribute binding metadata. Execution lowering binds explicit/default
attributes; unresolved references are not fabricated values.

Dense and sparse initializer defaults retain their shared identity with graph
inputs. `Graph::is_initializer` covers either storage form;
`is_constant_initializer` additionally excludes graph inputs. Sparse declarations
retain their original `SparseTensorProto`, including external payload references;
they are not densified or implicitly fetched. String tensor elements are raw
byte vectors, including non-UTF-8 data. The serializer re-emits both storage forms
and the complete ordered graph input interface.

Preserving source information does **not** implement sparse execution, arbitrary
container kernels, function expansion semantics inside the matcher, or all ORT
pattern productions. Unsupported runtime cases must fail explicitly before
tensor allocation/constant specialization. Schema defaults are still a separate
schema-layer responsibility; raw absence remains absence.

Generic captures and predicates can inspect full declared types and sparse
payloads, but recursive type-pattern syntax and sparse literal evaluation are
not implemented. Repeated GRAPH/GRAPHS, TYPE_PROTO and SPARSE_TENSORS equality is
explicitly unsupported; graph constraints or bounded consumer predicates are
required. No claim of all-ONNX execution or complete ORT-source compatibility is
made.

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
