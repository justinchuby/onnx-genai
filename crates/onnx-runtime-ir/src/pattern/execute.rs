use super::*;
use crate::{
    Attribute, Dim, Graph, GraphAttribute, GraphView, GraphViewCache, NodeId, ValueId, WeightRef,
};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Whether escaping intermediate values are allowed. Neither mode mutates IR.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BoundaryPolicy {
    /// Recognize structure even if an intermediate has unrelated users.
    #[default]
    Structural,
    /// Require non-returned outputs to have no users outside the match,
    /// including graph outputs and descendant lexical captures.
    OrtBoundary,
}

/// Per-call bounds, shared by alternatives, nested bodies and predicates.
#[derive(Clone, Copy, Debug)]
pub struct MatchBudget {
    pub steps: usize,
    pub candidates: usize,
    pub tensor_bytes: usize,
}

impl Default for MatchBudget {
    fn default() -> Self {
        Self {
            steps: 100_000,
            candidates: 10_000,
            tensor_bytes: 1 << 20,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MatchOptions {
    pub boundary: BoundaryPolicy,
    pub budget: MatchBudget,
}

/// A borrowed value whose identity includes its owning graph.
#[derive(Clone, Copy, Debug)]
pub struct ScopedValue<'g> {
    graph: &'g Graph,
    id: ValueId,
}

impl PartialEq for ScopedValue<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.graph, other.graph) && self.id == other.id
    }
}
impl Eq for ScopedValue<'_> {}

impl<'g> ScopedValue<'g> {
    pub fn id(self) -> ValueId {
        self.id
    }
    pub fn graph(self) -> &'g Graph {
        self.graph
    }
    pub fn value(self) -> &'g crate::Value {
        self.graph.value(self.id)
    }
}

/// Pre-resolved capture key. Keys cannot be used with a different pattern.
#[derive(Clone, Debug)]
pub struct ValueCapture {
    plans: Arc<[Plan]>,
    slots: Vec<usize>,
}

/// Pre-validated attribute key; lookups borrow the original attribute payload.
#[derive(Clone, Debug)]
pub struct AttributeCapture {
    plans: Arc<[Plan]>,
    names: Vec<String>,
}

impl Pattern {
    pub fn value_capture(&self, name: &str) -> Result<ValueCapture> {
        let slots = self.plans.iter().map(|plan| {
            let slot = plan.names.get(name).copied().ok_or_else(||
                capture_error(format!("tensor capture {name:?} is unavailable in an alternative; use a declared common optional parameter")))?;
            if matches!(plan.values[slot], ValuePlan::Sequence { .. }) {
                return Err(capture_error(format!("{name:?} is a sequence; use Match::sequence instead of a tensor key")));
            }
            Ok(slot)
        }).collect::<Result<_>>()?;
        Ok(ValueCapture {
            plans: self.plans.clone(),
            slots,
        })
    }

    pub fn attribute_capture(&self, name: &str) -> Result<AttributeCapture> {
        let names = self
            .plans
            .iter()
            .map(|plan| {
                plan.attribute_names.get(name).cloned().ok_or_else(|| {
                    capture_error(format!(
                        "attribute capture {name:?} is unavailable; declare it in every alternative"
                    ))
                })
            })
            .collect::<Result<_>>()?;
        Ok(AttributeCapture {
            plans: self.plans.clone(),
            names,
        })
    }
}

/// A known dimension binding. An unknown singleton dimension remains unbound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DimensionValue {
    Static(usize),
    Symbolic(String),
}

/// Match result owns its selected plan and captures, borrowing the unchanged IR.
#[derive(Clone, Debug)]
pub struct Match<'g> {
    plans: Arc<[Plan]>,
    selected: usize,
    graph: &'g Graph,
    values: Vec<Option<Option<ValueId>>>,
    nodes: Vec<NodeId>,
    attributes: BTreeMap<String, Option<&'g Attribute>>,
    dimensions: BTreeMap<String, DimensionValue>,
    elements: BTreeMap<String, DataType>,
    nested: BTreeMap<String, Match<'g>>,
    origins: BTreeMap<ValueId, Vec<ScopedValue<'g>>>,
    sequences: Vec<Option<Vec<Option<ValueId>>>>,
}

impl<'g> Match<'g> {
    pub fn graph(&self) -> &'g Graph {
        self.graph
    }
    /// Matched operators in expanded source order, not a deletion authorization.
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }
    pub fn selected_alternative(&self) -> usize {
        self.selected
    }

    /// Ordered native sequence with graph-qualified values; absent slots remain
    /// `None`. Materialization here is explicit, not performed during parsing.
    pub fn sequence(&self, name: &str) -> Result<Vec<Option<ScopedValue<'g>>>> {
        let slot = self.plan().names.get(name).copied().ok_or_else(|| {
            capture_error(format!(
                "unknown sequence {name:?}; declare a values/slots header capture"
            ))
        })?;
        if !matches!(self.plan().values[slot], ValuePlan::Sequence { .. }) {
            return Err(capture_error(format!(
                "{name:?} is not a sequence; use Match::value"
            )));
        }
        let values = self.sequences[slot].as_ref().ok_or_else(|| {
            capture_error(format!("sequence {name:?} is unbound in this alternative"))
        })?;
        Ok(values
            .iter()
            .map(|value| {
                value.map(|id| ScopedValue {
                    graph: self.graph,
                    id,
                })
            })
            .collect())
    }

    pub fn value(&self, name: &str) -> Result<Option<ScopedValue<'g>>> {
        let slot = self.plan().names.get(name).copied().ok_or_else(|| {
            capture_error(format!(
                "unknown tensor capture {name:?}; inspect the pattern's declared tensor names"
            ))
        })?;
        self.slot(slot)
    }

    pub fn captured(&self, capture: &ValueCapture) -> Result<Option<ScopedValue<'g>>> {
        self.check_key(&capture.plans)?;
        self.slot(capture.slots[self.selected])
    }

    pub fn attribute(&self, name: &str) -> Result<Option<&'g Attribute>> {
        let canonical = self.plan().attribute_names.get(name).ok_or_else(|| {
            capture_error(format!(
                "unknown attribute capture {name:?}; use a keyword capture name"
            ))
        })?;
        self.attributes.get(canonical).copied().ok_or_else(|| {
            capture_error(format!(
                "attribute capture {name:?} is not bound in the selected plan"
            ))
        })
    }

    pub fn captured_attribute(&self, capture: &AttributeCapture) -> Result<Option<&'g Attribute>> {
        self.check_key(&capture.plans)?;
        self.attributes
            .get(&capture.names[self.selected])
            .copied()
            .ok_or_else(|| capture_error("attribute key is unavailable in the selected match"))
    }

    pub fn dimension(&self, name: &str) -> Result<Option<&DimensionValue>> {
        if !self
            .plan()
            .annotations
            .iter()
            .flat_map(|(_, cs)| cs)
            .any(|c| {
                c.dimensions
                    .iter()
                    .flatten()
                    .any(|d| matches!(d, Dimension::Named(n) if n == name))
            })
        {
            return Err(capture_error(format!(
                "dimension {name:?} is not declared in this scope"
            )));
        }
        Ok(self.dimensions.get(name))
    }

    pub fn element_type(&self, name: &str) -> Result<Option<DataType>> {
        if !self
            .plan()
            .annotations
            .iter()
            .flat_map(|(_, cs)| cs)
            .any(|c| c.element_variable.as_deref() == Some(name))
        {
            return Err(capture_error(format!(
                "element variable {name:?} is not declared in this scope"
            )));
        }
        Ok(self.elements.get(name).copied())
    }

    /// Scoped view of a composed call, or a child instance such as `y.body`.
    pub fn instance(&self, path: &str) -> Result<BindingView<'_, 'g>> {
        if let Some(nested) = self.nested.get(path) {
            return Ok(BindingView {
                matched: nested,
                prefix: String::new(),
            });
        }
        let prefix = format!("{path}.");
        if self
            .plan()
            .names
            .keys()
            .any(|name| name.starts_with(&prefix))
            || self
                .plan()
                .attribute_names
                .keys()
                .any(|name| name.starts_with(&prefix))
        {
            Ok(BindingView {
                matched: self,
                prefix,
            })
        } else {
            Err(capture_error(format!(
                "instance {path:?} is unavailable; use a composed result or graph-attribute path"
            )))
        }
    }

    /// Resolve a captured operand to a visible ancestor definition. Formal inputs,
    /// initializers and produced body values cannot escape their owning scope.
    pub fn resolve_input(&self, name: &str, ancestor: &'g Graph) -> Result<ScopedValue<'g>> {
        let value = self.value(name)?.ok_or_else(|| {
            capture_error(format!(
                "capture {name:?} is absent; only a present operand can be resolved"
            ))
        })?;
        if std::ptr::eq(self.graph, ancestor) {
            return Ok(value);
        }
        let origins = self.origins.get(&value.id).ok_or_else(|| capture_error(format!(
            "capture {name:?} is local to this body; select an actual lexical capture, not a formal/initializer/output")))?;
        origins
            .iter()
            .copied()
            .find(|origin| std::ptr::eq(origin.graph, ancestor))
            .ok_or_else(|| {
                capture_error(format!(
                    "capture {name:?} has no visible value in the requested ancestor graph"
                ))
            })
    }

    fn plan(&self) -> &Plan {
        &self.plans[self.selected]
    }
    fn slot(&self, slot: usize) -> Result<Option<ScopedValue<'g>>> {
        if matches!(self.plan().values[slot], ValuePlan::Sequence { .. }) {
            return Err(capture_error(
                "sequence is not a scalar value; use Match::sequence",
            ));
        }
        match self.values[slot] {
            Some(value) => Ok(value.map(|id| ScopedValue {
                graph: self.graph,
                id,
            })),
            None if matches!(self.plan().values[slot], ValuePlan::Optional) => Ok(None),
            None => Err(capture_error(
                "required capture is unbound; the compiled plan is invalid",
            )),
        }
    }
    fn check_key(&self, plans: &Arc<[Plan]>) -> Result<()> {
        if Arc::ptr_eq(&self.plans, plans) {
            Ok(())
        } else {
            Err(capture_error(
                "capture key belongs to a different pattern; obtain it from the matched Pattern",
            ))
        }
    }
}

/// Read-only relative-name access into a composition/body instance.
pub struct BindingView<'m, 'g> {
    matched: &'m Match<'g>,
    prefix: String,
}

impl<'g> BindingView<'_, 'g> {
    pub fn graph(&self) -> &'g Graph {
        self.matched.graph
    }
    pub fn value(&self, name: &str) -> Result<Option<ScopedValue<'g>>> {
        self.matched.value(&format!("{}{name}", self.prefix))
    }
    pub fn attribute(&self, name: &str) -> Result<Option<&'g Attribute>> {
        self.matched.attribute(&format!("{}{name}", self.prefix))
    }
    pub fn dimension(&self, name: &str) -> Result<Option<&DimensionValue>> {
        self.matched.dimension(&format!("{}{name}", self.prefix))
    }
    pub fn resolve_input(&self, name: &str, ancestor: &'g Graph) -> Result<ScopedValue<'g>> {
        self.matched
            .resolve_input(&format!("{}{name}", self.prefix), ancestor)
    }
    pub fn instance(&self, path: &str) -> Result<BindingView<'_, 'g>> {
        self.matched.instance(&format!("{}{path}", self.prefix))
    }
}

struct Scope<'g> {
    graph: &'g Graph,
    cache: GraphViewCache,
    roots: BTreeMap<String, BTreeMap<String, Vec<NodeId>>>,
    children: BTreeMap<(NodeId, String, usize), usize>,
    origins: BTreeMap<ValueId, Vec<ScopedValue<'g>>>,
    capture_owners: BTreeMap<ValueId, BTreeSet<NodeId>>,
    names: HashMap<&'g str, ValueId>,
}

/// Prepared immutable graph tree. Preparation checks edges before constructing
/// the existing cache and resolves lexical captures without merging local IDs.
pub struct MatchSession<'g> {
    scopes: Vec<Scope<'g>>,
}

impl<'g> MatchSession<'g> {
    /// Prepare a complete IR graph. External uses not represented in this graph
    /// must be supplied by an adapter before relying on `OrtBoundary`.
    pub fn new(graph: &'g Graph) -> Result<Self> {
        let mut session = Self { scopes: Vec::new() };
        session.prepare(graph, &[], 0)?;
        Ok(session)
    }

    pub fn match_at(
        &self,
        pattern: &Pattern,
        root: NodeId,
        options: MatchOptions,
    ) -> Result<Option<Match<'g>>> {
        self.match_at_where(pattern, root, options, |_| Ok(true))
    }

    /// Rejecting a complete graph candidate tries the next assignment/plan.
    /// Predicate errors propagate without retry. Callbacks must be read-only.
    pub fn match_at_where(
        &self,
        pattern: &Pattern,
        root: NodeId,
        options: MatchOptions,
        mut predicate: impl FnMut(&Match<'g>) -> Result<bool>,
    ) -> Result<Option<Match<'g>>> {
        let mut found = None;
        self.visit_at(pattern, root, options, |matched| {
            if predicate(&matched)? {
                found = Some(matched);
                Ok(false)
            } else {
                Ok(true)
            }
        })?;
        Ok(found)
    }

    /// Source-order plans, then deterministic slot-connected assignments.
    /// Return `true` from the visitor to continue; `false` stops enumeration.
    pub fn visit_at(
        &self,
        pattern: &Pattern,
        root: NodeId,
        options: MatchOptions,
        mut visitor: impl FnMut(Match<'g>) -> Result<bool>,
    ) -> Result<()> {
        if self.scopes[0].graph.try_node(root).is_none() {
            return Err(graph_error(format!(
                "root {root:?} is not live; use a node from this session's graph"
            )));
        }
        require_operator_entry(pattern)?;
        let mut budget = options.budget;
        self.run(
            pattern,
            0,
            Some(root),
            false,
            options.boundary,
            &mut budget,
            0,
            &mut visitor,
        )?;
        Ok(())
    }

    /// Find entries using prepared domain/op root buckets, never a global
    /// per-pattern producer map. Search is confined to the root graph.
    pub fn find_matches(&self, pattern: &Pattern, options: MatchOptions) -> Result<Vec<Match<'g>>> {
        let mut roots = BTreeSet::new();
        require_operator_entry(pattern)?;
        for plan in pattern.plans.iter() {
            let node = plan.nodes.last().expect("verified anchor");
            if node.op == "*" {
                roots.extend(self.scopes[0].graph.nodes.keys());
            } else if let Some(bucket) = self.scopes[0]
                .roots
                .get(&node.domain)
                .and_then(|ops| ops.get(&node.op))
            {
                roots.extend(bucket);
            }
        }
        let mut budget = options.budget;
        let mut matches = Vec::new();
        for root in roots {
            self.run(
                pattern,
                0,
                Some(root),
                false,
                options.boundary,
                &mut budget,
                0,
                &mut |matched| {
                    matches.push(matched);
                    Ok(true)
                },
            )?;
        }
        Ok(matches)
    }

    /// Match ordered formal inputs/returns and cover the entire root graph.
    pub fn match_graph(
        &self,
        pattern: &Pattern,
        options: MatchOptions,
    ) -> Result<Option<Match<'g>>> {
        let mut budget = options.budget;
        let mut found = None;
        self.run(
            pattern,
            0,
            None,
            true,
            options.boundary,
            &mut budget,
            0,
            &mut |matched| {
                found = Some(matched);
                Ok(false)
            },
        )?;
        Ok(found)
    }

    fn prepare(
        &mut self,
        graph: &'g Graph,
        ancestors: &[(usize, NodeId)],
        depth: usize,
    ) -> Result<usize> {
        if depth >= 32 {
            return Err(budget_error(
                "graph nesting reaches 32; prepare a smaller graph tree",
            ));
        }
        validate_edges(graph)?;
        let mut origins = BTreeMap::new();
        let mut names = HashMap::new();
        for (id, value) in graph.values.iter() {
            if let Some(name) = value.name.as_deref()
                && !name.is_empty()
                && names.insert(name, id).is_some()
            {
                return Err(graph_error(format!(
                    "duplicate value name {name:?}; make SSA names unique in each scope"
                )));
            }
            if value.producer.is_some() || graph.inputs.contains(&id) || graph.is_initializer(id) {
                continue;
            }
            if value.consumers.is_empty() && !value.is_graph_output {
                continue;
            }
            let name = value.name.as_deref().ok_or_else(|| graph_error(format!(
                "source {id:?} has no formal/initializer or lexical name; name it or declare it as a graph input")))?;
            let mut resolved = None;
            for &(ancestor, _) in ancestors.iter().rev() {
                let parent = &self.scopes[ancestor];
                if let Some(&parent_id) = parent.names.get(name) {
                    if parent.graph.value(parent_id).producer.is_none()
                        && !parent.graph.inputs.contains(&parent_id)
                        && !parent.graph.is_initializer(parent_id)
                        && !parent.origins.contains_key(&parent_id)
                    {
                        continue;
                    }
                    let mut chain = vec![ScopedValue {
                        graph: parent.graph,
                        id: parent_id,
                    }];
                    if let Some(origins) = parent.origins.get(&parent_id) {
                        chain.extend(origins);
                    }
                    resolved = Some(chain);
                    break;
                }
            }
            let chain = resolved.ok_or_else(|| graph_error(format!(
                "free value {name:?} has no visible ancestor definition; declare a formal or repair the lexical capture")))?;
            for origin in &chain {
                for &(ancestor, owner) in ancestors {
                    let parent = &mut self.scopes[ancestor];
                    if std::ptr::eq(origin.graph, parent.graph) {
                        if origin.graph.value(origin.id).producer == Some(owner) {
                            return Err(graph_error(format!(
                                "capture {name:?} reads its owning node's output; remove the capture cycle"
                            )));
                        }
                        parent
                            .capture_owners
                            .entry(origin.id)
                            .or_default()
                            .insert(owner);
                    }
                }
            }
            origins.insert(id, chain);
        }
        let cache = GraphViewCache::build(graph).map_err(|error| {
            graph_error(format!(
                "cannot prepare graph indexes: {error}; repair cyclic/dangling IR before matching"
            ))
        })?;
        let mut roots: BTreeMap<String, BTreeMap<String, Vec<NodeId>>> = BTreeMap::new();
        for (id, node) in graph.nodes.iter() {
            roots
                .entry(crate::normalize_domain(&node.domain).into())
                .or_default()
                .entry(node.op_type.clone())
                .or_default()
                .push(id);
        }
        let scope = self.scopes.len();
        self.scopes.push(Scope {
            graph,
            cache,
            roots,
            children: BTreeMap::new(),
            origins,
            capture_owners: BTreeMap::new(),
            names,
        });
        for (owner, node) in graph.nodes.iter() {
            let mut bodies: BTreeMap<(String, usize), &Graph> = BTreeMap::new();
            let mut indexed = std::collections::BTreeSet::new();
            for (name, attribute) in &node.attributes {
                match attribute {
                    Attribute::Graph(body) => {
                        let key = GraphAttribute::single(name.clone());
                        indexed.insert(key.clone());
                        // The executor's indexed body is authoritative when present.
                        bodies.insert(
                            (name.clone(), 0),
                            graph.subgraphs.get(&(owner, key)).unwrap_or(body),
                        );
                    }
                    Attribute::Graphs(list) => {
                        for (index, body) in list.iter().enumerate() {
                            let key = GraphAttribute::element(name.clone(), index);
                            indexed.insert(key.clone());
                            bodies.insert(
                                (name.clone(), index),
                                graph.subgraphs.get(&(owner, key)).unwrap_or(body),
                            );
                        }
                    }
                    _ => {}
                }
            }
            for ((node_id, location), body) in &graph.subgraphs {
                if *node_id == owner && !indexed.contains(location) {
                    if location.index.is_some() || node.attributes.contains_key(&location.name) {
                        return Err(graph_error(format!(
                            "node {owner:?} indexed body {location:?} has no corresponding GRAPH/GRAPHS location; repair its attribute name, kind and list index"
                        )));
                    }
                    bodies.entry((location.name.clone(), 0)).or_insert(body);
                }
            }
            let mut next_ancestors = ancestors.to_vec();
            next_ancestors.push((scope, owner));
            for ((name, index), body) in bodies {
                let child = self.prepare(body, &next_ancestors, depth + 1)?;
                self.scopes[scope]
                    .children
                    .insert((owner, name, index), child);
            }
        }
        validate_capture_cycles(&self.scopes[scope])?;
        Ok(scope)
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        pattern: &Pattern,
        scope: usize,
        root: Option<NodeId>,
        whole: bool,
        boundary: BoundaryPolicy,
        budget: &mut MatchBudget,
        depth: usize,
        visitor: &mut dyn FnMut(Match<'g>) -> Result<bool>,
    ) -> Result<bool> {
        let graph = self.scopes[scope].graph;
        search_depth(depth)?;
        for (selected, plan) in pattern.plans.iter().enumerate() {
            tick(&mut budget.steps, "instruction budget")?;
            let mut state = State::new(plan);
            if whole {
                if plan
                    .inputs
                    .iter()
                    .any(|slot| matches!(plan.values[*slot], ValuePlan::Sequence { .. }))
                {
                    return Err(unsupported(
                        "whole-graph sequence formals are not yet supported; use fixed ordered formals and a variadic operator signature",
                    ));
                }
                let formals: Vec<_> = graph
                    .inputs
                    .iter()
                    .filter(|v| !graph.is_initializer(**v))
                    .copied()
                    .collect();
                if plan.inputs.len() < formals.len() || plan.outputs.len() != graph.outputs.len() {
                    continue;
                }
                let mut valid = true;
                for (&slot, actual) in plan
                    .inputs
                    .iter()
                    .zip(formals)
                    .chain(plan.outputs.iter().zip(graph.outputs.iter().copied()))
                {
                    valid &= bind_value(plan, &mut state, slot, Some(actual));
                }
                if !valid {
                    continue;
                }
            } else if !self.bind_node(
                plan,
                scope,
                plan.nodes.len() - 1,
                root.expect("operator root"),
                &mut state,
                boundary,
                budget,
                depth + 1,
            )? {
                continue;
            }
            if !self.enumerate(
                pattern,
                selected,
                scope,
                state,
                whole,
                boundary,
                budget,
                depth + 1,
                visitor,
            )? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn enumerate(
        &self,
        pattern: &Pattern,
        selected: usize,
        scope: usize,
        state: State<'g>,
        whole: bool,
        boundary: BoundaryPolicy,
        budget: &mut MatchBudget,
        depth: usize,
        visitor: &mut dyn FnMut(Match<'g>) -> Result<bool>,
    ) -> Result<bool> {
        tick(&mut budget.steps, "instruction budget")?;
        search_depth(depth)?;
        let plan = &pattern.plans[selected];
        let prepared = &self.scopes[scope];
        let graph = prepared.graph;
        if state.nodes.iter().all(Option::is_some) {
            tick(&mut budget.candidates, "candidate budget")?;
            let Some((dimensions, elements)) = annotations(plan, prepared, &state, budget)? else {
                return Ok(true);
            };
            if !self.complete(plan, scope, &state, whole, boundary, budget)? {
                return Ok(true);
            }
            let matched = Match {
                plans: pattern.plans.clone(),
                selected,
                graph,
                values: state.values,
                nodes: state.nodes.into_iter().flatten().collect(),
                attributes: state.attributes,
                nested: state.nested,
                dimensions,
                elements,
                origins: prepared.origins.clone(),
                sequences: state.sequences,
            };
            return visitor(matched);
        }
        // A bound output determines exactly one producer, including output identity.
        for ordinal in (0..plan.nodes.len()).rev() {
            if state.nodes[ordinal].is_some() {
                continue;
            }
            let mut producer = None;
            for slot in &plan.nodes[ordinal].outputs {
                if let Some(actual) = state.values[*slot] {
                    let Some(actual) = actual else {
                        return Ok(true);
                    };
                    let Some(node) = graph.value(actual).producer else {
                        return Ok(true);
                    };
                    if producer.is_some_and(|prior| prior != node) {
                        return Ok(true);
                    }
                    producer = Some(node);
                }
            }
            if let Some(node) = producer {
                let mut branch = state;
                if self.bind_node(
                    plan,
                    scope,
                    ordinal,
                    node,
                    &mut branch,
                    boundary,
                    budget,
                    depth + 1,
                )? {
                    return self.enumerate(
                        pattern,
                        selected,
                        scope,
                        branch,
                        whole,
                        boundary,
                        budget,
                        depth + 1,
                        visitor,
                    );
                }
                return Ok(true);
            }
        }
        // Match a remaining connected sink from its lowest eligible input slot.
        let view = GraphView::new(graph, &prepared.cache);
        for ordinal in (0..plan.nodes.len()).rev() {
            if state.nodes[ordinal].is_some() {
                continue;
            }
            for (input_slot, value_slot) in plan.nodes[ordinal].inputs.iter().enumerate() {
                let source = plan.producers[*value_slot];
                if !source.is_some_and(|source| state.nodes[source].is_some()) {
                    continue;
                }
                let Some(Some(actual)) = state.values[*value_slot] else {
                    continue;
                };
                let index = view.value_index(actual).expect("validated live value");
                for usage in view.consumers(index) {
                    let expected_input = match plan.nodes[ordinal].rest {
                        None => Some(input_slot),
                        Some(rest) if input_slot < rest => Some(input_slot),
                        Some(rest) if input_slot > rest => state.sequences
                            [plan.nodes[ordinal].inputs[rest]]
                            .as_ref()
                            .map(|sequence| input_slot + sequence.len() - 1),
                        _ => None,
                    };
                    if expected_input.is_some_and(|expected| usage.input_slot as usize != expected)
                    {
                        continue;
                    }
                    let mut branch = state.clone();
                    if self.bind_node(
                        plan,
                        scope,
                        ordinal,
                        view.node_id(usage.node),
                        &mut branch,
                        boundary,
                        budget,
                        depth + 1,
                    )? && !self.enumerate(
                        pattern,
                        selected,
                        scope,
                        branch,
                        whole,
                        boundary,
                        budget,
                        depth + 1,
                        visitor,
                    )? {
                        return Ok(false);
                    }
                }
                return Ok(true);
            }
        }
        // Whole-graph plans can contain independent components. Each must still
        // be anchored by an output (checked by compilation); no global search.
        Err(PatternError::new(
            PatternErrorKind::InvalidPattern,
            SourceSpan::default(),
            "plan has no connected producer/consumer frontier; connect its operators or anchor all whole-graph outputs",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_node(
        &self,
        plan: &Plan,
        scope: usize,
        ordinal: usize,
        actual: NodeId,
        state: &mut State<'g>,
        boundary: BoundaryPolicy,
        budget: &mut MatchBudget,
        depth: usize,
    ) -> Result<bool> {
        tick(&mut budget.steps, "instruction budget")?;
        search_depth(depth)?;
        let prepared = &self.scopes[scope];
        let Some(node) = prepared.graph.try_node(actual) else {
            return Err(graph_error("producer is not live; repair the graph"));
        };
        let expected = &plan.nodes[ordinal];
        if state.nodes.contains(&Some(actual))
            || (expected.op != "*"
                && (node.op_type != expected.op
                    || crate::normalize_domain(&node.domain) != expected.domain))
            || node.outputs.len() != expected.outputs.len()
            || (expected.rest.is_none() && node.inputs.len() > expected.inputs.len())
        {
            return Ok(false);
        }
        let graph_constraints: usize = expected
            .attributes
            .iter()
            .map(|(_, attr)| match attr {
                AttributePlan::Graph(_) => 1,
                AttributePlan::Graphs(patterns) => patterns.len(),
                _ => 0,
            })
            .sum();
        let graph_count = prepared
            .children
            .keys()
            .filter(|(owner, _, _)| *owner == actual)
            .count();
        if graph_count != graph_constraints {
            return Ok(false);
        }
        state.nodes[ordinal] = Some(actual);
        for (slot, &value) in expected.outputs.iter().zip(&node.outputs) {
            if !bind_value(plan, state, *slot, value) {
                return Ok(false);
            }
        }
        let rest_len = if expected.rest.is_some() {
            let Some(length) = node.inputs.len().checked_sub(expected.inputs.len() - 1) else {
                return Ok(false);
            };
            length
        } else {
            0
        };
        for (index, &slot) in expected.inputs.iter().enumerate() {
            if expected.rest == Some(index) {
                let ValuePlan::Sequence { min, max, holes } = plan.values[slot] else {
                    unreachable!("verified sequence")
                };
                let actual = &node.inputs[index..index + rest_len];
                if rest_len < min
                    || max.is_some_and(|max| rest_len > max)
                    || (!holes && actual.iter().any(Option::is_none))
                {
                    return Ok(false);
                }
                if let Some(prior) = &state.sequences[slot] {
                    if prior != actual {
                        return Ok(false);
                    }
                } else {
                    bytes(
                        budget,
                        rest_len.saturating_mul(size_of::<Option<ValueId>>()),
                    )?;
                    state.sequences[slot] = Some(actual.to_vec());
                }
            } else {
                let actual_index = if expected.rest.is_some_and(|rest| index > rest) {
                    index + rest_len - 1
                } else {
                    index
                };
                if !bind_value(
                    plan,
                    state,
                    slot,
                    node.inputs.get(actual_index).copied().flatten(),
                ) {
                    return Ok(false);
                }
            }
        }
        for (name, constraint) in &expected.attributes {
            let attribute = node.attributes.get(name);
            if let Some(Attribute::Ref { name: formal, .. }) = attribute {
                return Err(PatternError::new(
                    PatternErrorKind::MissingMetadata,
                    SourceSpan::default(),
                    format!(
                        "node {:?} attribute {name:?} refers to unbound function attribute {formal:?}; bind the function call's explicit/default attributes before constraining their values",
                        node.id
                    ),
                ));
            }
            match constraint {
                AttributePlan::Literal(literal) => {
                    let Some(attribute) = attribute else {
                        return Ok(false);
                    };
                    if !literal_equal(attribute, literal, budget)? {
                        return Ok(false);
                    }
                }
                AttributePlan::Capture(capture) => {
                    let (kind, optional) = &plan.attribute_types[capture];
                    if attribute.is_none() && !optional {
                        return Ok(false);
                    }
                    if let Some(attribute) = attribute {
                        if kind
                            .as_deref()
                            .is_some_and(|kind| !attribute_kind(attribute, kind))
                        {
                            return Ok(false);
                        }
                        if let Some(Some(prior)) = state.attributes.get(capture)
                            && !attribute_equal(prior, attribute, budget)?
                        {
                            return Ok(false);
                        }
                    }
                    for (name, required_kind, allows_absence) in &plan.attribute_conjunctions {
                        if name != capture {
                            continue;
                        }
                        if attribute.is_none() && !allows_absence {
                            return Ok(false);
                        }
                        if attribute.is_some_and(|attribute| {
                            required_kind
                                .as_deref()
                                .is_some_and(|kind| !attribute_kind(attribute, kind))
                        }) {
                            return Ok(false);
                        }
                    }
                    if state
                        .attributes
                        .get(capture)
                        .is_some_and(|prior| prior.is_some() != attribute.is_some())
                    {
                        return Ok(false);
                    }
                    state.attributes.insert(capture.clone(), attribute);
                }
                AttributePlan::Graph(pattern) => {
                    if matches!(attribute, Some(Attribute::Graphs(_))) {
                        return Err(unsupported(
                            "GRAPHS constraints require the native dialect; a GRAPH reference cannot stand for a list",
                        ));
                    }
                    let Some(&child) = prepared.children.get(&(actual, name.clone(), 0)) else {
                        return Ok(false);
                    };
                    let mut found = None;
                    self.run(
                        pattern,
                        child,
                        None,
                        true,
                        boundary,
                        budget,
                        depth + 1,
                        &mut |matched| {
                            found = Some(matched);
                            Ok(false)
                        },
                    )?;
                    let Some(found) = found else {
                        return Ok(false);
                    };
                    state
                        .nested
                        .insert(format!("{}.{name}", expected.instance), found);
                }
                AttributePlan::Graphs(patterns) => {
                    let Some(Attribute::Graphs(actual)) = attribute else {
                        return Ok(false);
                    };
                    if actual.len() != patterns.len() {
                        return Ok(false);
                    }
                    for (index, pattern) in patterns.iter().enumerate() {
                        let child = prepared.children[&(node.id, name.clone(), index)];
                        let mut found = None;
                        self.run(
                            pattern,
                            child,
                            None,
                            true,
                            boundary,
                            budget,
                            depth + 1,
                            &mut |matched| {
                                found = Some(matched);
                                Ok(false)
                            },
                        )?;
                        let Some(found) = found else {
                            return Ok(false);
                        };
                        state
                            .nested
                            .insert(format!("{}.{name}[{index}]", expected.instance), found);
                    }
                }
            }
        }
        Ok(true)
    }

    fn complete(
        &self,
        plan: &Plan,
        scope: usize,
        state: &State<'g>,
        whole: bool,
        boundary: BoundaryPolicy,
        budget: &mut MatchBudget,
    ) -> Result<bool> {
        let prepared = &self.scopes[scope];
        let graph = prepared.graph;
        let nodes: HashSet<_> = state.nodes.iter().flatten().copied().collect();
        let mut literals = HashSet::new();
        for (slot, expected) in plan.values.iter().enumerate() {
            if let ValuePlan::Literal(expected) = expected {
                let Some(Some(actual)) = state.values[slot] else {
                    return Ok(false);
                };
                let origin = prepared
                    .origins
                    .get(&actual)
                    .and_then(|chain| chain.last())
                    .copied()
                    .unwrap_or(ScopedValue { graph, id: actual });
                if !constant_equal(origin, expected, budget)? {
                    return Ok(false);
                }
                if std::ptr::eq(origin.graph, graph)
                    && let Some(producer) = graph.value(actual).producer
                {
                    literals.insert(producer);
                }
            }
        }
        if whole
            && graph
                .nodes
                .keys()
                .any(|node| !nodes.contains(&node) && !literals.contains(&node))
        {
            return Ok(false);
        }
        // Header leaves cannot be produced by constrained operators.
        for &slot in &plan.inputs {
            if whole && state.values[slot].is_none() {
                return Ok(false);
            }
            if let Some(Some(value)) = state.values[slot]
                && graph
                    .value(value)
                    .producer
                    .is_some_and(|producer| nodes.contains(&producer))
            {
                return Ok(false);
            }
        }
        if boundary == BoundaryPolicy::OrtBoundary {
            let exposed: HashSet<_> = plan
                .outputs
                .iter()
                .filter_map(|slot| state.values[*slot].flatten())
                .collect();
            for &node in &nodes {
                for output in graph.node(node).output_values() {
                    if exposed.contains(&output) {
                        continue;
                    }
                    if graph.outputs.contains(&output)
                        || graph
                            .value(output)
                            .consumers
                            .uses()
                            .iter()
                            .any(|(user, _)| !nodes.contains(user))
                        || prepared
                            .capture_owners
                            .get(&output)
                            .is_some_and(|owners| owners.iter().any(|owner| !nodes.contains(owner)))
                    {
                        return Ok(false);
                    }
                }
            }
        }
        Ok(true)
    }
}

#[derive(Clone)]
struct State<'g> {
    values: Vec<Option<Option<ValueId>>>,
    nodes: Vec<Option<NodeId>>,
    attributes: BTreeMap<String, Option<&'g Attribute>>,
    nested: BTreeMap<String, Match<'g>>,
    sequences: Vec<Option<Vec<Option<ValueId>>>>,
}
impl State<'_> {
    fn new(plan: &Plan) -> Self {
        Self {
            values: vec![None; plan.values.len()],
            nodes: vec![None; plan.nodes.len()],
            attributes: BTreeMap::new(),
            nested: BTreeMap::new(),
            sequences: vec![None; plan.values.len()],
        }
    }
}

fn bind_value(plan: &Plan, state: &mut State<'_>, slot: usize, actual: Option<ValueId>) -> bool {
    match plan.values[slot] {
        ValuePlan::Sequence { .. } => return false,
        ValuePlan::Absent if actual.is_some() => return false,
        ValuePlan::Present | ValuePlan::Literal(_) if actual.is_none() => return false,
        _ => {}
    }
    if let Some(prior) = state.values[slot] {
        return prior == actual;
    }
    state.values[slot] = Some(actual);
    true
}

type TypeBindings = (BTreeMap<String, DimensionValue>, BTreeMap<String, DataType>);

fn annotations(
    plan: &Plan,
    scope: &Scope<'_>,
    state: &State<'_>,
    budget: &mut MatchBudget,
) -> Result<Option<TypeBindings>> {
    let graph = scope.graph;
    let mut choices = vec![0; plan.annotations.len()];
    loop {
        tick(&mut budget.steps, "annotation budget")?;
        let selected: Vec<_> = plan
            .annotations
            .iter()
            .zip(&choices)
            .map(|((slot, cs), choice)| (*slot, &cs[*choice]))
            .collect();
        let mut counts = HashMap::<&str, usize>::new();
        for (_, c) in &selected {
            for d in c.dimensions.iter().flatten() {
                if let Dimension::Named(name) = d {
                    *counts.entry(name).or_default() += 1;
                }
            }
        }
        let mut dimensions = BTreeMap::new();
        let mut elements = BTreeMap::new();
        let mut valid = true;
        for (slot, c) in selected {
            let Some(Some(actual)) = state.values[slot] else {
                if !c.absent {
                    valid = false;
                    break;
                }
                continue;
            };
            if c.absent {
                valid = false;
                break;
            }
            let value = graph.value(actual);
            if c.tensor && !graph.value_type_is_known(actual) {
                valid = false;
                break;
            }
            let origin = scope
                .origins
                .get(&actual)
                .and_then(|chain| chain.last())
                .copied()
                .unwrap_or(ScopedValue { graph, id: actual });
            if c.immutable && !is_constant(origin) {
                valid = false;
                break;
            }
            if let Some(expected) = c.element
                && (!graph.value_type_is_known(actual) || value.dtype != expected)
            {
                valid = false;
                break;
            }
            if let Some(name) = &c.element_variable
                && graph.value_type_is_known(actual)
                && elements
                    .insert(name.clone(), value.dtype)
                    .is_some_and(|prior| prior != value.dtype)
            {
                valid = false;
                break;
            }
            if let Some(expected) = &c.dimensions {
                if !graph.value_shape_is_known(actual)
                    || (if c.prefix {
                        value.shape.len() < expected.len()
                    } else {
                        value.shape.len() != expected.len()
                    })
                {
                    valid = false;
                    break;
                }
                for (constraint, dimension) in expected.iter().zip(&value.shape) {
                    let known = match dimension {
                        Dim::Static(value) => Some(DimensionValue::Static(*value)),
                        Dim::Symbolic(symbol) => graph
                            .symbol_constraints
                            .get(symbol)
                            .and_then(|c| c.name.as_ref())
                            .filter(|name| !name.is_empty())
                            .map(|name| DimensionValue::Symbolic(name.clone())),
                    };
                    match constraint {
                        Dimension::Any => {}
                        Dimension::Fixed(expected)
                            if known != Some(DimensionValue::Static(*expected)) =>
                        {
                            valid = false;
                            break;
                        }
                        Dimension::Fixed(_) => {}
                        Dimension::Named(name) => {
                            if let Some(known) = known {
                                if dimensions
                                    .insert(name.clone(), known.clone())
                                    .is_some_and(|prior| prior != known)
                                {
                                    valid = false;
                                    break;
                                }
                            } else if counts[name.as_str()] > 1 {
                                valid = false;
                                break;
                            }
                        }
                    }
                }
                if !valid {
                    break;
                }
            }
        }
        if valid {
            return Ok(Some((dimensions, elements)));
        }
        let mut carry = true;
        for (index, choice) in choices.iter_mut().enumerate().rev() {
            *choice += 1;
            if *choice < plan.annotations[index].1.len() {
                carry = false;
                break;
            }
            *choice = 0;
        }
        if carry {
            return Ok(None);
        }
    }
}

fn literal_equal(
    attribute: &Attribute,
    expected: &Literal,
    budget: &mut MatchBudget,
) -> Result<bool> {
    Ok(match (attribute, expected) {
        (Attribute::Int(a), Literal::Int(b)) => {
            bytes(budget, 2 * size_of::<i64>())?;
            a == b
        }
        (Attribute::Float(a), Literal::Float(b)) => {
            bytes(budget, 2 * size_of::<f32>())?;
            a == b
        }
        (Attribute::String(a), Literal::String(b)) => {
            bytes(budget, a.len().saturating_add(b.len()))?;
            a == b
        }
        (Attribute::Ints(a), Literal::Ints(b)) => {
            bytes(budget, a.len().saturating_add(b.len()).saturating_mul(8))?;
            a == b
        }
        (Attribute::Floats(a), Literal::Floats(b)) => {
            bytes(budget, a.len().saturating_add(b.len()).saturating_mul(4))?;
            a == b
        }
        _ => false,
    })
}

fn attribute_kind(attribute: &Attribute, kind: &str) -> bool {
    matches!(
        (attribute, kind),
        (Attribute::Int(_), "int")
            | (Attribute::Float(_), "float")
            | (Attribute::String(_), "string")
            | (Attribute::Ints(_), "ints")
            | (Attribute::Floats(_), "floats")
            | (Attribute::Strings(_), "strings")
            | (Attribute::Tensor(_), "tensor")
    )
}

fn attribute_equal(a: &Attribute, b: &Attribute, budget: &mut MatchBudget) -> Result<bool> {
    Ok(match (a, b) {
        (Attribute::Int(a), Attribute::Int(b)) => {
            bytes(budget, 2 * size_of::<i64>())?;
            a == b
        }
        (Attribute::Float(a), Attribute::Float(b)) => {
            bytes(budget, 2 * size_of::<f32>())?;
            a.to_bits() == b.to_bits()
        }
        (Attribute::String(a), Attribute::String(b)) => {
            bytes(budget, a.len().saturating_add(b.len()))?;
            a == b
        }
        (Attribute::Ints(a), Attribute::Ints(b)) => {
            bytes(budget, a.len().saturating_add(b.len()).saturating_mul(8))?;
            a == b
        }
        (Attribute::Floats(a), Attribute::Floats(b)) => {
            bytes(budget, a.len().saturating_add(b.len()).saturating_mul(4))?;
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
        }
        (Attribute::Strings(a), Attribute::Strings(b)) => {
            for value in a.iter().chain(b) {
                bytes(budget, size_of::<Vec<u8>>().saturating_add(value.len()))?;
            }
            a == b
        }
        (Attribute::Tensor(a), Attribute::Tensor(b)) => tensor_attribute_equal(a, b, budget)?,
        (Attribute::Tensors(a), Attribute::Tensors(b)) => {
            if a.len() != b.len() {
                false
            } else {
                let mut equal = true;
                for (a, b) in a.iter().zip(b) {
                    equal &= tensor_attribute_equal(a, b, budget)?;
                }
                equal
            }
        }
        (Attribute::TypeProto(_), Attribute::TypeProto(_))
        | (Attribute::TypeProtos(_), Attribute::TypeProtos(_)) => {
            return Err(unsupported(
                "repeated TYPE_PROTO equality needs a lossless bounded type representation; inspect the captured types in a predicate",
            ));
        }
        (Attribute::SparseTensor(a), Attribute::SparseTensor(b)) => {
            bytes(
                budget,
                a.dims
                    .len()
                    .saturating_add(b.dims.len())
                    .saturating_mul(size_of::<usize>()),
            )?;
            a.dims == b.dims
                && tensor_attribute_equal(&a.values, &b.values, budget)?
                && tensor_attribute_equal(&a.indices, &b.indices, budget)?
        }
        (Attribute::Graph(_), Attribute::Graph(_))
        | (Attribute::Graphs(_), Attribute::Graphs(_))
        | (Attribute::SparseTensors(_), Attribute::SparseTensors(_)) => {
            return Err(unsupported(
                "repeated GRAPH/GRAPHS/SPARSE_TENSORS attribute equality is unavailable; constrain bodies or use a semantic predicate",
            ));
        }
        _ => false,
    })
}

fn charge_tensor(tensor: &TensorData, budget: &mut MatchBudget) -> Result<()> {
    bytes(
        budget,
        size_of::<DataType>()
            .saturating_add(tensor.data.len())
            .saturating_add(tensor.dims.len().saturating_mul(size_of::<usize>()))
            .saturating_add(tensor.name.as_ref().map_or(0, String::len)),
    )?;
    for value in &tensor.strings {
        bytes(budget, size_of::<Vec<u8>>().saturating_add(value.len()))?;
    }
    Ok(())
}

fn tensor_attribute_equal(
    a: &TensorData,
    b: &TensorData,
    budget: &mut MatchBudget,
) -> Result<bool> {
    charge_tensor(a, budget)?;
    charge_tensor(b, budget)?;
    Ok(a == b)
}

fn tensor_equal(a: &TensorData, b: &TensorData, budget: &mut MatchBudget) -> Result<bool> {
    charge_tensor(a, budget)?;
    charge_tensor(b, budget)?;
    if a.dtype != b.dtype || a.dims != b.dims {
        return Ok(false);
    }
    if a.dtype == DataType::Float32 {
        return Ok(a.data.len() == b.data.len()
            && a.data
                .chunks_exact(4)
                .zip(b.data.chunks_exact(4))
                .all(|(a, b)| {
                    f32::from_le_bytes(a.try_into().expect("chunk"))
                        == f32::from_le_bytes(b.try_into().expect("chunk"))
                }));
    }
    Ok(a.data == b.data && a.strings == b.strings)
}

fn is_constant(value: ScopedValue<'_>) -> bool {
    !value.graph.inputs.contains(&value.id)
        && (value.graph.is_initializer(value.id)
            || value.value().producer.is_some_and(|id| {
                let node = value.graph.node(id);
                crate::is_default_domain(&node.domain)
                    && node.op_type == "Constant"
                    && [
                        "value",
                        "value_int",
                        "value_float",
                        "value_ints",
                        "value_floats",
                    ]
                    .iter()
                    .any(|name| node.attributes.contains_key(*name))
            }))
}

fn constant_equal(
    value: ScopedValue<'_>,
    expected: &TensorData,
    budget: &mut MatchBudget,
) -> Result<bool> {
    if value.graph.inputs.contains(&value.id) {
        return Ok(false);
    }
    if value.graph.sparse_initializers.contains_key(&value.id) {
        return Err(unsupported(
            "dense literal comparison cannot evaluate a sparse initializer; inspect its preserved sparse payload in a predicate",
        ));
    }
    let tensor = if let Some(weight) = value.graph.initializers.get(&value.id) {
        match weight {
            WeightRef::Inline(tensor) => tensor,
            WeightRef::External { .. } => {
                return Err(PatternError::new(
                    PatternErrorKind::MissingMetadata,
                    SourceSpan::default(),
                    format!(
                        "literal at {:?} needs external initializer bytes; materialize the initializer before matching",
                        value.id
                    ),
                ));
            }
        }
    } else {
        let Some(id) = value.value().producer else {
            return Ok(false);
        };
        let node = value.graph.node(id);
        if !crate::is_default_domain(&node.domain) || node.op_type != "Constant" {
            return Ok(false);
        }
        if let Some(Attribute::Tensor(tensor)) = node.attr("value") {
            tensor
        } else {
            let literal = match (expected.dtype, expected.dims.as_slice()) {
                (DataType::Int64, []) => node
                    .attr("value_int")
                    .and_then(Attribute::as_int)
                    .map(Literal::Int),
                (DataType::Float32, []) => node
                    .attr("value_float")
                    .and_then(Attribute::as_float)
                    .map(Literal::Float),
                (DataType::Int64, [_]) => match node.attr("value_ints") {
                    Some(Attribute::Ints(v)) => {
                        bytes(budget, v.len().saturating_mul(8))?;
                        Some(Literal::Ints(v.clone()))
                    }
                    _ => None,
                },
                (DataType::Float32, [_]) => match node.attr("value_floats") {
                    Some(Attribute::Floats(v)) => {
                        bytes(budget, v.len().saturating_mul(4))?;
                        Some(Literal::Floats(v.clone()))
                    }
                    _ => None,
                },
                _ => None,
            };
            let Some(literal) = literal else {
                return Ok(false);
            };
            let actual = match literal {
                Literal::Int(v) => {
                    TensorData::from_raw(DataType::Int64, vec![], v.to_le_bytes().to_vec())
                }
                Literal::Float(v) => {
                    TensorData::from_raw(DataType::Float32, vec![], v.to_le_bytes().to_vec())
                }
                Literal::Ints(v) => TensorData::from_raw(
                    DataType::Int64,
                    vec![v.len()],
                    v.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
                Literal::Floats(v) => TensorData::from_raw(
                    DataType::Float32,
                    vec![v.len()],
                    v.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
                _ => unreachable!(),
            };
            return tensor_equal(&actual, expected, budget);
        }
    };
    if tensor.checked_expected_bytes() != Some(tensor.data.len()) {
        return Err(graph_error(format!(
            "constant {:?} has malformed {:?} payload for {:?}; repair initializer bytes",
            value.id, tensor.dtype, tensor.dims
        )));
    }
    tensor_equal(tensor, expected, budget)
}

fn validate_edges(graph: &Graph) -> Result<()> {
    let mut producers = HashMap::new();
    for (id, node) in graph.nodes.iter() {
        if node.id != id {
            return Err(graph_error(
                "node arena key disagrees with node.id; repair IR identity",
            ));
        }
        for (slot, value) in node.inputs.iter().enumerate() {
            if let Some(value) = value {
                let actual = graph.try_value(*value).ok_or_else(|| {
                    graph_error(format!(
                        "node {id:?} input {slot} references missing {value:?}; repair the edge"
                    ))
                })?;
                if !actual.consumers.contains(id, slot as u32) {
                    return Err(graph_error(format!(
                        "node {id:?} input {slot} has no reverse use; rebuild consistent graph edges"
                    )));
                }
            }
        }
        for (slot, value) in node.outputs.iter().enumerate() {
            let Some(value) = *value else { continue };
            let actual = graph.try_value(value).ok_or_else(|| {
                graph_error(format!(
                    "node {id:?} output slot {slot} references missing {value:?}; create the output first"
                ))
            })?;
            if producers.insert(value, id).is_some() || actual.producer != Some(id) {
                return Err(graph_error(format!(
                    "output {value:?} has inconsistent/multiple producers; use SSA graph mutation APIs"
                )));
            }
        }
    }
    for (id, value) in graph.values.iter() {
        if value.id != id || value.producer != producers.get(&id).copied() {
            return Err(graph_error(format!(
                "value {id:?} has inconsistent producer/identity metadata; repair graph edges"
            )));
        }
        if value.is_graph_input != graph.inputs.contains(&id)
            || value.is_graph_output != graph.outputs.contains(&id)
        {
            return Err(graph_error(format!(
                "value {id:?} graph-interface flags disagree; use set_inputs/set_outputs"
            )));
        }
        for (user, slot) in value.consumers.uses() {
            if graph
                .try_node(user)
                .and_then(|node| node.inputs.get(slot as usize))
                != Some(&Some(id))
            {
                return Err(graph_error(format!(
                    "value {id:?} has stale consumer {user:?}:{slot}; repair graph edges"
                )));
            }
        }
    }
    for value in graph
        .inputs
        .iter()
        .chain(&graph.outputs)
        .chain(graph.initializers.keys())
        .chain(graph.sparse_initializers.keys())
    {
        if graph.try_value(*value).is_none() {
            return Err(graph_error(format!(
                "interface/initializer value {value:?} is not live; repair the graph"
            )));
        }
    }
    for value in &graph.inputs {
        if graph.value(*value).producer.is_some() {
            return Err(graph_error(format!(
                "formal {value:?} has a producer; graph inputs must be sources"
            )));
        }
    }
    if graph.opset_imports.values().any(|version| *version == 0) {
        return Err(graph_error(
            "opset import zero is invalid; supply a positive version",
        ));
    }
    for (owner, _) in graph.subgraphs.keys() {
        if graph.try_node(*owner).is_none() {
            return Err(graph_error(format!(
                "subgraph owner {owner:?} is missing; repair the body index"
            )));
        }
    }
    Ok(())
}

fn validate_capture_cycles(scope: &Scope<'_>) -> Result<()> {
    let graph = scope.graph;
    let mut edges = HashMap::<NodeId, BTreeSet<NodeId>>::new();
    let mut indegree: HashMap<_, usize> = graph.nodes.keys().map(|node| (node, 0)).collect();
    for (id, node) in graph.nodes.iter() {
        for value in node.input_values() {
            if let Some(producer) = graph.value(value).producer {
                edges.entry(producer).or_default().insert(id);
            }
        }
    }
    for (value, owners) in &scope.capture_owners {
        if let Some(producer) = graph.value(*value).producer {
            edges.entry(producer).or_default().extend(owners);
        }
    }
    for users in edges.values() {
        for user in users {
            *indegree.get_mut(user).expect("validated owner") += 1;
        }
    }
    let mut ready: Vec<_> = indegree
        .iter()
        .filter_map(|(id, count)| (*count == 0).then_some(*id))
        .collect();
    let mut visited = 0;
    while let Some(node) = ready.pop() {
        visited += 1;
        if let Some(users) = edges.get(&node) {
            for user in users {
                let count = indegree.get_mut(user).expect("validated edge");
                *count -= 1;
                if *count == 0 {
                    ready.push(*user);
                }
            }
        }
    }
    if visited != graph.num_nodes() {
        return Err(graph_error(
            "explicit/lexical-capture dependencies contain a cycle; remove the cyclic capture or producer edge",
        ));
    }
    Ok(())
}

fn tick(remaining: &mut usize, resource: &str) -> Result<()> {
    *remaining = remaining.checked_sub(1).ok_or_else(|| {
        budget_error(format!(
            "{resource} exhausted; raise MatchOptions.budget or simplify the pattern"
        ))
    })?;
    Ok(())
}
fn search_depth(depth: usize) -> Result<()> {
    if depth > 96 {
        return Err(budget_error(
            "combined graph/assignment search depth exceeds 96; split this entry into shallower patterns",
        ));
    }
    Ok(())
}
fn bytes(budget: &mut MatchBudget, count: usize) -> Result<()> {
    budget.tensor_bytes = budget.tensor_bytes.checked_sub(count).ok_or_else(||
        budget_error("attribute/tensor byte budget exhausted; increase tensor_bytes or avoid large literal comparisons"))?;
    Ok(())
}
fn require_operator_entry(pattern: &Pattern) -> Result<()> {
    if pattern.operator_entry {
        return Ok(());
    }
    Err(PatternError::new(
        PatternErrorKind::InvalidPattern,
        SourceSpan::default(),
        "whole-graph/callable pattern is not verified as an operator entry; select PatternLibrary::pattern for anchored search",
    ))
}

fn capture_error(message: impl Into<String>) -> PatternError {
    PatternError::new(
        PatternErrorKind::InvalidCapture,
        SourceSpan::default(),
        message,
    )
}
fn graph_error(message: impl Into<String>) -> PatternError {
    PatternError::new(
        PatternErrorKind::InvalidGraph,
        SourceSpan::default(),
        message,
    )
}
fn budget_error(message: impl Into<String>) -> PatternError {
    PatternError::new(
        PatternErrorKind::BudgetExceeded,
        SourceSpan::default(),
        message,
    )
}
fn unsupported(message: impl Into<String>) -> PatternError {
    PatternError::new(
        PatternErrorKind::Unsupported,
        SourceSpan::default(),
        message,
    )
}
