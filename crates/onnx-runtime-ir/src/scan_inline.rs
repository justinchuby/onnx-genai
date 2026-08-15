//! Decode-specialized lowering of single-trip `Scan` bodies (Inc-1b PR-1).
//!
//! [`inline_single_trip_scan_bodies`] rewrites every structurally
//! *single-trip-eligible* `Scan` node into one straight-line iteration of its
//! body spliced directly into the parent graph. It is the graph transform
//! described in the Inc-1b design (`cohaagen-27b-inc1b-design.md` §1): at decode
//! a LinearAttention `Scan` runs its body exactly once (scan-axis extent 1), so
//! the control-flow node degenerates to a straight-line body that — once inlined
//! into the parent's own [`ValueId`] namespace — can later fold into the parent
//! CUDA-graph capture (PR-3) instead of running through a separate child
//! [`Executor`] in eager mode.
//!
//! ## Scope / status (READ THIS)
//!
//! This is a **pure `Graph -> Graph` transform and nothing else**. It is
//! deliberately **not wired into any executor, decoder, loader, or run path** —
//! it is dead code exercised only by this crate's unit tests. Wiring the
//! decode-inline plan in behind a default-off flag (PR-2) and letting the
//! inlined body capture (PR-3) are separate, later PRs and are explicitly out of
//! scope here. This PR changes **zero** runtime behavior.
//!
//! Because `onnx-runtime-ir` is the pure base contract with no dependency on
//! `onnx-runtime-shape-inference`, this transform performs only the structural
//! merge (mirroring the graph-building half of `ChildExecutor::compile`). It
//! produces a graph whose interior values already carry the body's declared
//! shapes, so a caller (PR-2) — or this crate's tests via a dev-dependency — can
//! run `InferenceRegistry::infer_graph` (Permissive) over the result to
//! re-resolve interior shapes exactly as `ChildExecutor::compile` does today.
//!
//! ## What "single-trip-eligible" means here
//!
//! A `Scan` is lowered only when it is *structurally* safe to specialize to one
//! iteration:
//! * default-domain `Scan` with a registered `body` subgraph,
//! * it carries **recurrent state** (`num_state >= 1`) — i.e. a LinearAttention
//!   body threading `state_pairs`, not a pure element-wise map,
//! * every scan input's scan axis is **not a static extent > 1** (`Static(1)` or
//!   dynamic/symbolic — the decode case), so lowering to a single iteration is
//!   valid by construction,
//! * the body contains **no nested control flow** (no `If`/`Loop`/`Scan`
//!   subgraphs), keeping the splice bounded and its captures name-resolvable,
//! * every body lexical capture resolves by name to a value already in parent
//!   scope.
//!
//! Any `Scan` failing these checks — and any graph with no such `Scan` — is left
//! structurally untouched, so the transform is a safe no-op on the prefill /
//! multi-trip / dense paths.

use std::collections::{HashMap, HashSet};

use crate::graph::Graph;
use crate::node::{Attribute, Node, NodeId};
use crate::shape::Dim;
use crate::value::ValueId;

/// Lower every single-trip-eligible `Scan` body into the parent graph.
///
/// Returns a **new** graph; the input is never mutated (matching the crate's
/// `Graph -> Graph` pass convention). Non-eligible `Scan` nodes and graphs with
/// no eligible `Scan` are returned structurally unchanged.
///
/// See the [module docs](self) for the eligibility rules and the honest
/// (unwired, zero-behavior-change) scope of this pass.
#[must_use]
pub fn inline_single_trip_scan_bodies(graph: &Graph) -> Graph {
    let mut out = graph.clone();

    // Snapshot the candidate node ids up front: the set of Scan nodes cannot
    // grow during inlining, and each body lives in its own ValueId namespace, so
    // the bodies are independent and can be lowered one at a time.
    let scan_nodes: Vec<NodeId> = out
        .nodes
        .iter()
        .filter(|(_, node)| is_default_domain_scan(node))
        .map(|(nid, _)| nid)
        .collect();

    for scan_id in scan_nodes {
        if let Some(plan) = ScanInlinePlan::extract(&out, scan_id) {
            plan.apply(&mut out);
        }
    }

    out
}

fn is_default_domain_scan(node: &Node) -> bool {
    node.is_default_domain() && node.op_type == "Scan"
}

/// Everything the splice needs, extracted from one eligible `Scan` before any
/// mutation. Building this returns `None` for a non-eligible `Scan`, which is
/// how the transform stays a no-op on ports it must not touch.
struct ScanInlinePlan {
    scan_id: NodeId,
    body: Graph,
    /// Parent state-input value per body state formal (`num_state` entries).
    state_inputs: Vec<ValueId>,
    /// Parent scan-input value per body scan formal (`num_scan_inputs` entries).
    scan_inputs: Vec<ValueId>,
    /// Parent state-output value per body state output (`num_state` entries).
    state_outputs: Vec<ValueId>,
    /// Parent scan-output value per body scan output (`num_scan_outputs`).
    scan_outputs: Vec<ValueId>,
    /// Normalized (non-negative) scan-input axis per scan input.
    input_axes: Vec<usize>,
    /// Normalized (non-negative) scan-output axis per scan output.
    output_axes: Vec<usize>,
}

impl ScanInlinePlan {
    fn extract(graph: &Graph, scan_id: NodeId) -> Option<Self> {
        let node = graph.try_node(scan_id)?;
        let body = graph.subgraphs.get(&(scan_id, "body".to_string()))?.clone();

        let num_scan_inputs = usize::try_from(node.attr("num_scan_inputs")?.as_int()?).ok()?;
        if num_scan_inputs == 0 || node.inputs.len() < num_scan_inputs {
            return None;
        }
        let num_state = node.inputs.len() - num_scan_inputs;
        // Recurrent state is what makes a Scan single-trip-eligible per the
        // design; a pure element-wise map (num_state == 0) is left alone.
        if num_state == 0 || node.outputs.len() < num_state {
            return None;
        }
        let num_scan_outputs = node.outputs.len() - num_state;

        // The body's formal signature must line up with the node arity.
        if body.inputs.len() != num_state + num_scan_inputs
            || body.outputs.len() != num_state + num_scan_outputs
        {
            return None;
        }

        // A nested control-flow body would require recursively remapping its own
        // captures; keep PR-1 bounded to straight-line (LinearAttention) bodies.
        if body_has_nested_control_flow(&body) {
            return None;
        }

        let state_inputs = collect_present(&node.inputs[..num_state])?;
        let scan_inputs = collect_present(&node.inputs[num_state..])?;
        let state_outputs = node.outputs[..num_state].to_vec();
        let scan_outputs = node.outputs[num_state..].to_vec();

        let input_axes_raw = int_list_attr(node, "scan_input_axes", num_scan_inputs)?;
        let output_axes_raw = int_list_attr(node, "scan_output_axes", num_scan_outputs)?;

        let mut input_axes = Vec::with_capacity(num_scan_inputs);
        for (scan_input, &raw_axis) in scan_inputs.iter().zip(&input_axes_raw) {
            let rank = graph.value(*scan_input).shape.len();
            let axis = normalize_axis(raw_axis, rank)?;
            // Single-trip gate: refuse to statically collapse an axis that is a
            // concrete extent > 1 (a genuine multi-trip / prefill Scan).
            if let Some(Dim::Static(extent)) = graph.value(*scan_input).shape.get(axis)
                && *extent > 1
            {
                return None;
            }
            input_axes.push(axis);
        }

        let mut output_axes = Vec::with_capacity(num_scan_outputs);
        for (scan_output, &raw_axis) in scan_outputs.iter().zip(&output_axes_raw) {
            // scan_output_axes index the *output* tensor (the body slice rank + 1).
            let out_rank = graph.value(*scan_output).shape.len().max(1);
            output_axes.push(normalize_axis(raw_axis, out_rank)?);
        }

        // Every lexical capture must resolve by name in the parent, exactly as
        // ChildExecutor::new does; otherwise the splice would dangle.
        let name_index = parent_name_index(graph);
        for name in body_capture_names(&body) {
            if !name_index.contains_key(&name) {
                return None;
            }
        }

        Some(Self {
            scan_id,
            body,
            state_inputs,
            scan_inputs,
            state_outputs,
            scan_outputs,
            input_axes,
            output_axes,
        })
    }

    fn apply(self, out: &mut Graph) {
        let num_state = self.state_inputs.len();
        let name_index = parent_name_index(out);
        let formal_set: HashSet<ValueId> = self.body.inputs.iter().copied().collect();

        // body ValueId -> fresh parent ValueId (or an existing parent value for
        // formals / captures / promoted initializers).
        let mut remap: HashMap<ValueId, ValueId> = HashMap::new();

        // 1. State formals bind to the Scan's parent state inputs.
        for (i, &parent_state_in) in self.state_inputs.iter().enumerate() {
            remap.insert(self.body.inputs[i], parent_state_in);
        }

        // 2. Scan-input formals bind to a Squeeze that drops the size-1 scan axis.
        for (j, (&parent_scan_in, &axis)) in
            self.scan_inputs.iter().zip(&self.input_axes).enumerate()
        {
            let squeezed = self.emit_squeeze(out, parent_scan_in, axis);
            remap.insert(self.body.inputs[num_state + j], squeezed);
        }

        // 3. Lexical captures resolve by name to values already in parent scope.
        for (vid, value) in self.body.values.iter() {
            let is_capture = value.producer.is_none()
                && !formal_set.contains(&vid)
                && !self.body.initializers.contains_key(&vid);
            if is_capture
                && let Some(name) = &value.name
                && let Some(&parent) = name_index.get(name)
            {
                remap.insert(vid, parent);
            }
        }

        // 4. Body-local initializers become parent initializers.
        for (&vid, weight) in &self.body.initializers {
            let value = self.body.value(vid);
            let promoted = out.create_value(value.dtype, value.shape.clone());
            if let Some(name) = &value.name {
                out.value_mut(promoted).name = Some(name.clone());
            }
            out.set_initializer(promoted, weight.clone());
            remap.insert(vid, promoted);
        }

        // 5. Decide which state outputs can be written directly by their
        //    producer (the common recurrent case) versus needing an Identity
        //    (a pass-through of a formal/capture/initializer).
        let mut direct_state_out: HashSet<ValueId> = HashSet::new();
        let mut identity_state_out: Vec<(ValueId, ValueId)> = Vec::new();
        for (k, &parent_state_out) in self.state_outputs.iter().enumerate() {
            let body_out = self.body.outputs[k];
            let producible = self
                .body
                .try_value(body_out)
                .is_some_and(|v| v.producer.is_some());
            if producible && !remap.contains_key(&body_out) && direct_state_out.insert(body_out) {
                // The producing body node will write straight into the parent
                // present-state value; no copy node needed.
                remap.insert(body_out, parent_state_out);
            } else {
                identity_state_out.push((body_out, parent_state_out));
            }
        }

        // 6. Every remaining body-produced value gets a fresh, anonymous parent
        //    value. This is where interior SSA and scan outputs get their ids.
        for (vid, value) in self.body.values.iter() {
            if remap.contains_key(&vid) || value.producer.is_none() {
                continue;
            }
            let fresh = out.create_value(value.dtype, value.shape.clone());
            remap.insert(vid, fresh);
        }

        // 7. Clone body nodes into the parent in topological order with remapped
        //    edges. Topo order is well-defined because the body is acyclic.
        let order = self
            .body
            .topological_order()
            .expect("scan body must be acyclic");
        for nid in order {
            let bn = self.body.node(nid);
            let inputs = bn
                .inputs
                .iter()
                .map(|slot| slot.map(|v| remap[&v]))
                .collect();
            let outputs = bn.outputs.iter().map(|v| remap[v]).collect();
            let mut nn = Node::new(NodeId(0), bn.op_type.clone(), inputs, outputs);
            nn.name = bn.name.clone();
            nn.domain = bn.domain.clone();
            nn.version = bn.version;
            nn.attributes = bn.attributes.clone();
            nn.doc_string = bn.doc_string.clone();
            out.insert_node(nn);
        }

        // 8. Pass-through state outputs get an Identity from their bound value.
        for (body_out, parent_state_out) in identity_state_out {
            let src = remap[&body_out];
            out.insert_node(Node::new(
                NodeId(0),
                "Identity",
                vec![Some(src)],
                vec![parent_state_out],
            ));
        }

        // 9. Scan outputs get an Unsqueeze re-adding the size-1 scan axis to
        //    match the Scan's declared scan-output rank.
        for (m, (&parent_scan_out, &axis)) in
            self.scan_outputs.iter().zip(&self.output_axes).enumerate()
        {
            let src = remap[&self.body.outputs[num_state + m]];
            self.emit_unsqueeze(out, src, parent_scan_out, axis);
        }

        // 10. Delete the Scan node and drop its subgraph entries so a recycled
        //     NodeId can never re-inherit the stale body.
        out.remove_node(self.scan_id);
        out.subgraphs.retain(|(owner, _), _| *owner != self.scan_id);
    }

    /// Emit `Squeeze(parent_scan_in, axes=[axis])` and return its output value.
    fn emit_squeeze(&self, out: &mut Graph, parent_scan_in: ValueId, axis: usize) -> ValueId {
        let (dtype, mut shape) = {
            let v = out.value(parent_scan_in);
            (v.dtype, v.shape.clone())
        };
        if axis < shape.len() {
            shape.remove(axis);
        }
        let squeezed = out.create_value(dtype, shape);
        out.insert_node(squeeze_like("Squeeze", parent_scan_in, squeezed, axis));
        squeezed
    }

    /// Emit `Unsqueeze(src, axes=[axis]) -> parent_scan_out`.
    fn emit_unsqueeze(&self, out: &mut Graph, src: ValueId, parent_scan_out: ValueId, axis: usize) {
        out.insert_node(squeeze_like("Unsqueeze", src, parent_scan_out, axis));
    }
}

/// Build a `Squeeze`/`Unsqueeze` node using the opset-1 attribute form of
/// `axes`, pinned via `version` so the merged graph resolves it regardless of
/// the parent's model-level opset.
fn squeeze_like(op: &str, input: ValueId, output: ValueId, axis: usize) -> Node {
    let mut node = Node::new(NodeId(0), op, vec![Some(input)], vec![output]);
    node.version = Some(1);
    node.attributes
        .insert("axes".to_string(), Attribute::Ints(vec![axis as i64]));
    node
}

/// Whether a subgraph body itself contains control-flow subgraphs (which PR-1
/// intentionally does not descend into).
fn body_has_nested_control_flow(body: &Graph) -> bool {
    if !body.subgraphs.is_empty() {
        return true;
    }
    body.nodes.iter().any(|(_, node)| {
        node.attributes
            .values()
            .any(|attr| matches!(attr, Attribute::Graph(_) | Attribute::Graphs(_)))
    })
}

/// The names of a body's lexical captures: producer-less, named values that are
/// neither formals nor local initializers (mirrors `ChildExecutor::new`).
fn body_capture_names(body: &Graph) -> Vec<String> {
    let formal_set: HashSet<ValueId> = body.inputs.iter().copied().collect();
    body.values
        .iter()
        .filter_map(|(vid, value)| {
            (value.producer.is_none()
                && !formal_set.contains(&vid)
                && !body.initializers.contains_key(&vid))
            .then(|| value.name.clone())
            .flatten()
        })
        .collect()
}

/// Map each parent value name to its [`ValueId`] for capture resolution.
fn parent_name_index(graph: &Graph) -> HashMap<String, ValueId> {
    let mut index = HashMap::new();
    for (vid, value) in graph.values.iter() {
        if let Some(name) = &value.name {
            index.entry(name.clone()).or_insert(vid);
        }
    }
    index
}

/// Collect the present (non-omitted) value ids of a node input slice, or `None`
/// if any slot is empty (ONNX forbids omitted Scan state/scan inputs).
fn collect_present(slots: &[Option<ValueId>]) -> Option<Vec<ValueId>> {
    slots.iter().copied().collect()
}

/// Read an `Ints` list attribute of the expected length, defaulting a missing
/// attribute to all-zeros (the ONNX default axis).
fn int_list_attr(node: &Node, name: &str, expected: usize) -> Option<Vec<i64>> {
    match node.attr(name) {
        None => Some(vec![0; expected]),
        Some(attr) => {
            let values = attr.as_ints()?;
            (values.len() == expected).then(|| values.to_vec())
        }
    }
}

/// Normalize a possibly-negative ONNX axis into `0..rank`.
fn normalize_axis(axis: i64, rank: usize) -> Option<usize> {
    let rank_i = i64::try_from(rank).ok()?;
    let resolved = if axis < 0 { axis + rank_i } else { axis };
    (0..rank_i).contains(&resolved).then_some(resolved as usize)
}

#[cfg(test)]
mod tests;
