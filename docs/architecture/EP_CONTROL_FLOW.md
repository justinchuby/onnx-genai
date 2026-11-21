# Whole-graph EP claim and control-flow execution

An EP needs the owning graph, its nested bodies and lexical bindings to recognize
control-flow patterns. A leaf `supports_op(Node, ...)` query cannot express this.

`ExecutionProvider::claim_graph(&Graph)` is a pure structural query over the
complete owning graph. Use the IR pattern matcher for recognition. Return live,
unique graph-local anchor IDs; `GraphKernelClaims` validates the result. Claims
are queried again after structural rewrites, so providers must not retain arena
IDs across graph revisions or allocate executor artifacts during discovery.

For each accepted anchor the runtime calls `get_graph_kernel(GraphKernelRequest,
shapes, opset)`. The request includes the same complete owning graph, the anchor,
and its boundary input slots. ONNX positional slots, including omitted optional
inputs, stay intact; distinct lexical captures follow in value-ID order. Native
compilation supplies the executor's private artifact namespace. Foreign ORT
callbacks explicitly use `GraphKernelScope::Unscoped`; providers that require
executor-owned artifacts must decline that scope.

A claimed `If`, `Loop`, or `Scan` executes the compiled EP kernel. Compilation
and execution errors propagate; they do not switch to another implementation.
Unclaimed control flow retains the existing session implementation. No generic
control-flow interpreter was added to the EP layer.

## Lexical dependencies

The IR owns capture analysis through `Graph::required_outer_names` and
`captured_inputs_by_node`. Formal inputs, producers and initializers shadow
ancestor names. Transitive captures propagate through intervening scopes.
Topological sorting, GraphView partition convexity, partition boundaries and
native liveness all include these dependencies. Formal operator input lists
remain separate from captures, preserving operator schemas and optional slots.

Nested session executors close their graph over actual ancestor bindings before
claiming it. ORT also invokes GetCapability independently on nested scopes; the
outbound reader closes only such a callback root using the enclosing node's
implicit-input descriptors. Recursively imported bodies keep their ONNX formal
arity and lexical links.

## Runtime output shapes

Known-rank metadata and unknown rank remain distinct: `Some([])` is a scalar;
`None` is unknown rank. Actual input shapes seed unknown-rank bindings on each
run. Decode memoization remains available for known-rank graphs and is disabled
when unknown rank makes shape substitution insufficient.

Graph kernels can use ordinary preallocated outputs when concrete geometry is
known, or opt into `Kernel::execute_kernel_sized`. The latter reports owned
payloads with actual dtype and shape. The common validator checks all outputs
before publication, including optional slots, dimensions, arithmetic overflow,
dtype and exact payload length. Native external-output capacities are checked
before copying any result. Existing binding ownership, memory budgets, sealed
constant overrides and strided-input materialization still apply.

The native sized-output path currently requires host-owned fixed-width tensor
payloads and host-accessible inputs. Device-workspace sized outputs and container
boundaries are rejected explicitly. This is not a claim of CUDA execution or
CUDA graph-capture support for control flow.

## Plugin bridge

The native ORT graph projection exposes nested graph names, stable parent links,
implicit inputs, inherited operator sets, IR version, symbolic dimension names,
and the distinction between unknown rank and scalar. Outbound import preserves
scalar/list attributes, tensor attributes and binary string contents; attribute
read errors include the owning node and attribute instead of dropping metadata.
Tensor-attribute OrtValues and shape descriptors have explicit owned lifetimes.
For typed proto3 INT/FLOAT attributes whose zero field is omitted on the wire,
ORT 32's explicit field-absence response is decoded as the protobuf scalar
default; other attribute errors still propagate. Real ORT conformance includes
zero-valued integer and float attributes so a reader cannot silently decline
the complete graph at this boundary.
Large initializer payloads remain lazy; discovery does not copy all weights.

For native legacy-plugin claims containing nested bodies, Compute may allocate
owned outputs with their actual runtime geometry. Ordinary plugin tensor-DAG
claims retain preallocated output dispatch. Both paths share the existing
per-thread plugin-state lifetime, bounded state map and Compute error handling.

ORT's name-only subgraph API cannot faithfully distinguish GRAPH-list elements;
the bridge rejects those instead of inventing names. Its STRINGS byte format
uses NUL delimiters; export rejects embedded NUL within a list element. Scalar
STRING and string tensor payloads preserve embedded and trailing NUL bytes.
The recursive ABI transport has an explicit nesting limit of 128.

## Evidence

CPU conformance uses structural EP patterns for If, Loop and Scan, compares
results against the existing CPU control-flow implementation and independent
math, and asserts zero session subgraph executions for claimed programs. Cases
include changing captures and output lengths, zero iterations, omitted Loop
conditions and malformed dynamic payloads. Separate tests traverse the complete
native C graph/claim/Compile/Compute bridge and a real ORT session. Execution
counters ensure host fallback cannot satisfy the real-ORT test.
