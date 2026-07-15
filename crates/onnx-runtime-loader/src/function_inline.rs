//! Model-local function inlining (ONNX function expansion) at load time.
//!
//! An ONNX `ModelProto` may declare reusable subgraphs as `FunctionProto`s in
//! `ModelProto.functions`. A node whose `(domain, op_type, overload)` matches a
//! declared function's `(domain, name, overload)` is a *function call*: it is
//! semantically equivalent to the function body with the call's actual inputs,
//! outputs, and attributes substituted in.
//!
//! Our executor only has kernels for primitive ops, so this module rewrites the
//! `ModelProto` at the proto level — **before** [`crate::graph_builder`] runs —
//! so the rest of the pipeline never sees a function call. Because the rewrite
//! is proto-level, the existing `NodeProto → IR` conversion (attributes,
//! control-flow subgraphs) is reused unchanged.
//!
//! ## Algorithm (standard ONNX function expansion)
//!
//! For each function-call node, we splice in a fresh copy of the matched
//! function body:
//!
//! 1. **Value remapping.** Formal `input[i]`/`output[j]` names are mapped to the
//!    call's actual argument names (positionally). Every *other* value name in
//!    the body (an intermediate result) is renamed to a fresh, per-instantiation
//!    unique name (`__fn{K}_{orig}`) so repeated instantiations never collide.
//!    The empty name `""` (ONNX "absent optional") is never remapped.
//!
//! 2. **Attribute binding.** A body-node attribute with a non-empty
//!    `ref_attr_name = A` is a reference to the function's formal attribute `A`.
//!    It is resolved from the call site (the call node's attribute `A`), else the
//!    function's declared default (`attribute_proto` entry named `A`), else — if
//!    `A` is a required attribute (`FunctionProto.attribute`) — an error; else
//!    the attribute is dropped. Literal (non-`ref`) attributes are kept as-is.
//!
//! 3. **Recursion + fixpoint.** A function body may call other functions; those
//!    calls are expanded recursively to a fixpoint. True recursion (a function
//!    that transitively calls itself) is rejected rather than looped forever.
//!
//! 4. **Control-flow subgraphs.** Function calls may appear inside If/Loop/Scan
//!    subgraph bodies, and function bodies may themselves contain control flow;
//!    both are handled by recursing into every node's `Graph`/`Graphs`
//!    attributes.
//!
//! ## Opset policy
//!
//! `FunctionProto.opset_import` domains/versions are merged into the model's
//! `opset_import`, taking the highest version per domain. Per the ONNX spec the
//! operator schemas for a shared domain must be compatible across the two opset
//! lists, so a version difference is not treated as a conflict; a domain the
//! model does not yet declare is added.
//!
//! ## Overload policy
//!
//! Matching is exact on the full `(domain, name, overload)` triple, so an
//! overload set is disambiguated by the node's `overload` field.

use std::borrow::Cow;
use std::collections::HashMap;

use crate::proto::onnx::{
    AttributeProto, FunctionProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
};
use crate::LoaderError;

/// Unique identity of a model-local function: `(domain, name, overload)`.
type FnKey = (String, String, String);

fn fn_key_of_function(f: &FunctionProto) -> FnKey {
    (f.domain.clone(), f.name.clone(), f.overload.clone())
}

fn fn_key_of_call(n: &NodeProto) -> FnKey {
    (n.domain.clone(), n.op_type.clone(), n.overload.clone())
}

/// Expand every call to a model-local function in `model` into the function's
/// body, so the returned `ModelProto`'s graph (and all nested subgraphs) contain
/// only calls to ops the runtime has kernels for.
///
/// When `model.functions` is empty this is a no-op and the input is borrowed
/// back unchanged (`Cow::Borrowed`). Otherwise a rewritten owned `ModelProto`
/// is returned with `functions` cleared and function opset imports merged in.
pub fn inline_functions(model: &ModelProto) -> Result<Cow<'_, ModelProto>, LoaderError> {
    if model.functions.is_empty() {
        return Ok(Cow::Borrowed(model));
    }

    let mut funcs: HashMap<FnKey, &FunctionProto> = HashMap::new();
    for f in &model.functions {
        funcs.insert(fn_key_of_function(f), f);
    }

    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| LoaderError::GraphBuild("ModelProto has no graph".into()))?;

    let mut counter: usize = 0;
    let mut stack: Vec<FnKey> = Vec::new();
    let new_graph = inline_graph(graph, &funcs, &mut counter, &mut stack)?;

    let mut out = model.clone();
    out.graph = Some(new_graph);
    out.opset_import = merged_opset_imports(model);
    out.functions.clear();
    Ok(Cow::Owned(out))
}

/// Merge every function's `opset_import` into the model's, taking the highest
/// version per domain. Preserves the model's original import ordering, then
/// appends any domains introduced solely by functions (in first-seen order).
fn merged_opset_imports(model: &ModelProto) -> Vec<OperatorSetIdProto> {
    let mut order: Vec<String> = Vec::new();
    let mut best: HashMap<String, i64> = HashMap::new();
    let mut note = |domain: &str, version: i64| {
        let entry = best.entry(domain.to_string());
        match entry {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                if version > *e.get() {
                    *e.get_mut() = version;
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                order.push(domain.to_string());
                e.insert(version);
            }
        }
    };
    for o in &model.opset_import {
        note(&o.domain, o.version);
    }
    for f in &model.functions {
        for o in &f.opset_import {
            note(&o.domain, o.version);
        }
    }
    order
        .into_iter()
        .map(|domain| {
            let version = best[&domain];
            OperatorSetIdProto { domain, version }
        })
        .collect()
}

/// Rewrite `gp` so its node list contains no calls to any declared function.
/// Regular nodes are kept (with their control-flow subgraphs recursively
/// inlined); function-call nodes are replaced by their expanded bodies.
fn inline_graph(
    gp: &GraphProto,
    funcs: &HashMap<FnKey, &FunctionProto>,
    counter: &mut usize,
    stack: &mut Vec<FnKey>,
) -> Result<GraphProto, LoaderError> {
    let mut out = gp.clone();
    out.node = Vec::with_capacity(gp.node.len());
    for node in &gp.node {
        expand_node(node, funcs, counter, stack, &mut out.node)?;
    }
    Ok(out)
}

/// Append the fully-inlined form of `node` to `sink`. If `node` calls a
/// function, its body (recursively inlined) is appended; otherwise the node is
/// appended with its subgraph attributes recursively inlined.
fn expand_node(
    node: &NodeProto,
    funcs: &HashMap<FnKey, &FunctionProto>,
    counter: &mut usize,
    stack: &mut Vec<FnKey>,
    sink: &mut Vec<NodeProto>,
) -> Result<(), LoaderError> {
    if let Some(func) = funcs.get(&fn_key_of_call(node)) {
        instantiate(node, func, funcs, counter, stack, sink)?;
    } else {
        sink.push(inline_subgraph_attrs(node, funcs, counter, stack)?);
    }
    Ok(())
}

/// Return a copy of `node` whose `Graph`/`Graphs` attribute bodies have had any
/// function calls inside them inlined.
fn inline_subgraph_attrs(
    node: &NodeProto,
    funcs: &HashMap<FnKey, &FunctionProto>,
    counter: &mut usize,
    stack: &mut Vec<FnKey>,
) -> Result<NodeProto, LoaderError> {
    let mut out = node.clone();
    for attr in &mut out.attribute {
        if let Some(g) = attr.g.as_mut() {
            *g = inline_graph(g, funcs, counter, stack)?;
        }
        for g in &mut attr.graphs {
            *g = inline_graph(g, funcs, counter, stack)?;
        }
    }
    Ok(out)
}

/// Expand a single function call: substitute actual arguments and attributes
/// into a fresh copy of the function body, then recursively inline any calls the
/// body itself makes. Appends the resulting primitive nodes to `sink`.
fn instantiate(
    call: &NodeProto,
    func: &FunctionProto,
    funcs: &HashMap<FnKey, &FunctionProto>,
    counter: &mut usize,
    stack: &mut Vec<FnKey>,
    sink: &mut Vec<NodeProto>,
) -> Result<(), LoaderError> {
    let key = fn_key_of_function(func);

    if stack.contains(&key) {
        let mut chain: Vec<String> = stack.iter().map(fmt_key).collect();
        chain.push(fmt_key(&key));
        return Err(LoaderError::RecursiveFunction {
            function: fmt_key(&key),
            chain: chain.join(" -> "),
        });
    }

    // Arity: passing *more* actuals than the function declares is illegal;
    // passing fewer is allowed (trailing optionals omitted, mapped to absent).
    if call.input.len() > func.input.len() {
        return Err(LoaderError::FunctionArityMismatch {
            function: fmt_key(&key),
            node: node_label(call),
            kind: "input",
            formal: func.input.len(),
            actual: call.input.len(),
        });
    }
    if call.output.len() > func.output.len() {
        return Err(LoaderError::FunctionArityMismatch {
            function: fmt_key(&key),
            node: node_label(call),
            kind: "output",
            formal: func.output.len(),
            actual: call.output.len(),
        });
    }

    let inst_id = *counter;
    *counter += 1;

    // 1. Value remapping: formals -> actuals, everything else -> fresh.
    let mut rename: HashMap<String, String> = HashMap::new();
    for (i, formal) in func.input.iter().enumerate() {
        if formal.is_empty() {
            continue;
        }
        let actual = call.input.get(i).cloned().unwrap_or_default();
        rename.insert(formal.clone(), actual);
    }
    for (j, formal) in func.output.iter().enumerate() {
        if formal.is_empty() {
            continue;
        }
        let actual = call.output.get(j).cloned().unwrap_or_default();
        rename.insert(formal.clone(), actual);
    }
    // Fresh names for internal (non-formal) value names appearing in the body.
    for bn in &func.node {
        for name in bn.input.iter().chain(bn.output.iter()) {
            if name.is_empty() || rename.contains_key(name) {
                continue;
            }
            rename.insert(name.clone(), format!("__fn{inst_id}_{name}"));
        }
    }

    // 2. Attribute binding + value renaming for each body node.
    stack.push(key.clone());
    let result = (|| {
        let mut instantiated: Vec<NodeProto> = Vec::with_capacity(func.node.len());
        for (idx, bn) in func.node.iter().enumerate() {
            let mut nn = bn.clone();

            // Rename node name to a fresh unique one to avoid duplicate-name
            // collisions between instantiations.
            nn.name = if bn.name.is_empty() {
                format!("__fn{inst_id}_n{idx}")
            } else {
                format!("__fn{inst_id}_{}", bn.name)
            };

            // Bind attributes (resolve ref_attr_name against the call site).
            let mut bound: Vec<AttributeProto> = Vec::with_capacity(nn.attribute.len());
            for attr in &nn.attribute {
                if let Some(resolved) = bind_attribute(attr, call, func, &key)? {
                    bound.push(resolved);
                }
            }
            nn.attribute = bound;

            // Rename value references (inputs/outputs + captured names inside
            // any control-flow subgraph attributes).
            rename_value_refs(&mut nn, &rename);

            instantiated.push(nn);
        }

        // 3. Recursively inline any function calls the body itself makes.
        let mut expanded: Vec<NodeProto> = Vec::new();
        for n in &instantiated {
            expand_node(n, funcs, counter, stack, &mut expanded)?;
        }
        Ok::<Vec<NodeProto>, LoaderError>(expanded)
    })();
    stack.pop();

    sink.extend(result?);
    Ok(())
}

/// Resolve a body-node attribute for a specific instantiation.
///
/// * Literal attribute (`ref_attr_name` empty): kept unchanged.
/// * Reference attribute (`ref_attr_name = A`): replaced by the call-site
///   attribute `A`, else the function's default for `A`, else dropped (if `A` is
///   optional) or an error (if `A` is required). The emitted attribute keeps the
///   body attribute's `name` and has `ref_attr_name` cleared.
///
/// Returns `Ok(None)` when the attribute should be omitted from the node.
fn bind_attribute(
    attr: &AttributeProto,
    call: &NodeProto,
    func: &FunctionProto,
    key: &FnKey,
) -> Result<Option<AttributeProto>, LoaderError> {
    if attr.ref_attr_name.is_empty() {
        return Ok(Some(attr.clone()));
    }
    let a = &attr.ref_attr_name;

    // Call-site value wins.
    if let Some(supplied) = call.attribute.iter().find(|ca| &ca.name == a) {
        let mut bound = supplied.clone();
        bound.name = attr.name.clone();
        bound.ref_attr_name.clear();
        return Ok(Some(bound));
    }
    // Otherwise the function's declared default, if any.
    if let Some(default) = func.attribute_proto.iter().find(|d| &d.name == a) {
        let mut bound = default.clone();
        bound.name = attr.name.clone();
        bound.ref_attr_name.clear();
        return Ok(Some(bound));
    }
    // No value and no default: an error if the attribute is required, else drop.
    if func.attribute.iter().any(|req| req == a) {
        return Err(LoaderError::MissingRequiredFunctionAttribute {
            function: fmt_key(key),
            node: node_label(call),
            attribute: a.clone(),
        });
    }
    Ok(None)
}

/// Apply `rename` to a node's value references: its inputs, its outputs, and any
/// value names captured inside its control-flow subgraph attributes. A name of
/// `""` (absent optional) is left untouched; a name absent from `rename` is left
/// as-is (subgraph-local names live in their own scope).
fn rename_value_refs(node: &mut NodeProto, rename: &HashMap<String, String>) {
    for name in node.input.iter_mut().chain(node.output.iter_mut()) {
        if let Some(new) = rename.get(name.as_str()) {
            *name = new.clone();
        }
    }
    for attr in &mut node.attribute {
        if let Some(g) = attr.g.as_mut() {
            rename_subgraph_refs(g, rename);
        }
        for g in &mut attr.graphs {
            rename_subgraph_refs(g, rename);
        }
    }
}

/// Rename outer-scope value captures inside a subgraph. Only names present in
/// `rename` are rewritten, so a subgraph referencing a function-body value is
/// wired correctly while the subgraph's own local values are left alone.
fn rename_subgraph_refs(gp: &mut GraphProto, rename: &HashMap<String, String>) {
    for n in &mut gp.node {
        rename_value_refs(n, rename);
    }
}

fn fmt_key(key: &FnKey) -> String {
    let (domain, name, overload) = key;
    let d = if domain.is_empty() { "ai.onnx" } else { domain };
    if overload.is_empty() {
        format!("{d}::{name}")
    } else {
        format!("{d}::{name}:{overload}")
    }
}

fn node_label(n: &NodeProto) -> String {
    if n.name.is_empty() {
        format!("<{}::{} (unnamed)>", n.domain, n.op_type)
    } else {
        n.name.clone()
    }
}
