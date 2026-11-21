use super::syntax::{Definition, DefinitionKind, Expr, ExprKind, Parameter, RawConstraint};
use super::*;
use std::collections::{BTreeSet, HashSet};

fn invalid(span: SourceSpan, message: impl Into<String>) -> PatternError {
    PatternError::new(PatternErrorKind::InvalidPattern, span, message)
}
fn unsupported(span: SourceSpan, message: impl Into<String>) -> PatternError {
    PatternError::new(PatternErrorKind::Unsupported, span, message)
}
fn limit(span: SourceSpan) -> PatternError {
    PatternError::new(
        PatternErrorKind::BudgetExceeded,
        span,
        "expanded pattern exceeds 64 plans/nodes, 256 values/annotations/calls, or 64 type combinations; factor or simplify it",
    )
}

pub(super) fn inline(definition: Definition) -> Result<Pattern> {
    compile_body(definition, true)
}

pub(super) fn compile_body(definition: Definition, operator_entry: bool) -> Result<Pattern> {
    let mut compiler = Compiler {
        definitions: BTreeMap::new(),
        compiled: BTreeMap::new(),
        active: HashSet::new(),
        heights: BTreeMap::new(),
        registry: None,
    };
    let plans = compiler.body(&definition, 0)?;
    if operator_entry {
        for plan in &plans {
            verify_entry(plan)?;
        }
    }
    Ok(Pattern {
        plans: plans.into(),
        operator_entry,
    })
}

pub(super) fn library(
    definitions: BTreeMap<String, DefinitionKind>,
    registry: &dyn Fn(&str, &str) -> bool,
) -> Result<PatternLibrary> {
    let mut compiler = Compiler {
        definitions,
        compiled: BTreeMap::new(),
        active: HashSet::new(),
        heights: BTreeMap::new(),
        registry: Some(registry),
    };
    let names: Vec<_> = compiler.definitions.keys().cloned().collect();
    for name in names {
        if let DefinitionKind::Interface(inputs, outputs, span) = &compiler.definitions[&name] {
            validate_interface(inputs, outputs, *span)?;
        } else {
            compiler.definition(&name, 0)?;
        }
    }
    Ok(PatternLibrary {
        patterns: compiler.compiled,
    })
}

type OperatorRegistry<'r> = &'r dyn Fn(&str, &str) -> bool;

struct ExpressionPlan {
    plan: Plan,
    outputs: Vec<usize>,
    ports: Vec<Parameter>,
}

struct Compiler<'r> {
    definitions: BTreeMap<String, DefinitionKind>,
    compiled: BTreeMap<String, Pattern>,
    active: HashSet<String>,
    heights: BTreeMap<String, usize>,
    registry: Option<OperatorRegistry<'r>>,
}

impl Compiler<'_> {
    fn definition(&mut self, name: &str, depth: usize) -> Result<Pattern> {
        // Check before consulting the cache so declaration order cannot bypass the bound.
        if depth >= 32 {
            return Err(PatternError::new(
                PatternErrorKind::BudgetExceeded,
                SourceSpan::default(),
                format!("definition dependency depth reaches 32 at {name:?}; flatten the library"),
            ));
        }
        if self.active.contains(name) {
            return Err(invalid(
                SourceSpan::default(),
                format!(
                    "recursive pattern reference {name:?}; pattern dependencies must be acyclic"
                ),
            ));
        }
        if let Some(pattern) = self.compiled.get(name) {
            if depth + self.heights[name] >= 32 {
                return Err(limit(SourceSpan::default()));
            }
            return Ok(pattern.clone());
        }
        let definition = self.definitions.get(name).cloned().ok_or_else(|| {
            invalid(
                SourceSpan::default(),
                format!("unknown pattern {name:?}; define it before selecting it"),
            )
        })?;
        self.active.insert(name.into());
        let dependencies = dependencies(&definition, &self.definitions);
        let mut height = 0;
        for dependency in dependencies {
            self.definition(&dependency, depth + 1)?;
            height = height.max(self.heights[&dependency] + 1);
        }
        self.heights.insert(name.into(), height);
        let plans = match definition {
            DefinitionKind::Interface(..) => {
                return Err(invalid(
                    SourceSpan::default(),
                    format!(
                        "interface {name:?} is a tensor contract, not a callable pattern; select an implementation"
                    ),
                ));
            }
            DefinitionKind::Body(mut body) => {
                if let Some(interface) = &body.interface {
                    let Some(DefinitionKind::Interface(inputs, outputs, span)) =
                        self.definitions.get(interface)
                    else {
                        return Err(invalid(
                            body.span,
                            format!("unknown interface {interface:?}; declare it with interface"),
                        ));
                    };
                    validate_interface(inputs, outputs, *span)?;
                    body.call_parameters = Some(inputs.len());
                    let mut parameters = inputs.clone();
                    parameters.append(&mut body.parameters);
                    body.parameters = parameters;
                    body.output_signature = Some(outputs.clone());
                }
                if let Some(labels) = &body.return_labels {
                    if let Some(signature) = &body.output_signature
                        && body.interface.is_some()
                    {
                        if signature.len() != labels.len() {
                            return Err(invalid(
                                body.span,
                                "return ports differ from the output contract; map every declared label exactly once",
                            ));
                        }
                        let mut returns = Vec::new();
                        for output in signature {
                            let index = labels.iter().position(|label| label == &output.name).ok_or_else(||
                                invalid(body.span, format!("missing return port {:?}; use the declared output labels", output.name)))?;
                            returns.push(body.returns[index].clone());
                        }
                        body.returns = returns;
                    } else if body.output_signature.is_none() {
                        body.output_signature = Some(
                            labels
                                .iter()
                                .map(|name| Parameter {
                                    name: name.clone(),
                                    annotation: None,
                                })
                                .collect(),
                        );
                    }
                }
                self.body(&body, depth + 1)?
            }
            DefinitionKind::Alternatives(members, span) => {
                let mut plans: Vec<Plan> = Vec::new();
                for member in members {
                    let pattern = self.definition(&member, depth + 1)?;
                    for plan in pattern.plans.iter() {
                        if let Some(first) = plans.first()
                            && !same_signature(first, plan)
                        {
                            return Err(unsupported(
                                span,
                                "named alternatives need identical interfaces, ordered ports and annotations; use call-expression alternatives for different signatures",
                            ));
                        }
                        plans.push(plan.clone());
                        if plans.len() > 64 {
                            return Err(limit(span));
                        }
                    }
                }
                plans
            }
        };
        self.active.remove(name);
        let pattern = Pattern {
            plans: plans.into(),
            operator_entry: false,
        };
        self.compiled.insert(name.into(), pattern.clone());
        Ok(pattern)
    }

    fn body(&mut self, definition: &Definition, depth: usize) -> Result<Vec<Plan>> {
        if definition.require_outputs {
            let mut labels: BTreeSet<_> = definition
                .parameters
                .iter()
                .take(
                    definition
                        .call_parameters
                        .unwrap_or(definition.parameters.len()),
                )
                .map(|p| p.name.as_str())
                .collect();
            if let Some(outputs) = &definition.output_signature {
                for output in outputs {
                    if !labels.insert(&output.name) {
                        return Err(invalid(
                            definition.span,
                            "duplicate input/output contract label; give each port a distinct name",
                        ));
                    }
                }
            }
        }
        let mut start = Plan::empty();
        start.interface = definition.interface.clone();
        start.signature = definition.parameters.clone();
        let mut declared = BTreeSet::new();
        for parameter in &definition.parameters {
            if !declared.insert(parameter.name.clone()) {
                return Err(invalid(
                    definition.span,
                    format!(
                        "duplicate parameter {:?}; use a unique name",
                        parameter.name
                    ),
                ));
            }
            let value = match parameter.annotation.as_deref() {
                Some([raw]) if raw.sequence.is_some() => {
                    let (min, max, holes) = raw.sequence.expect("checked sequence");
                    ValuePlan::Sequence { min, max, holes }
                }
                _ => ValuePlan::Present,
            };
            let slot = start.value(value);
            start.names.insert(parameter.name.clone(), slot);
            start
                .binding_annotations
                .insert(parameter.name.clone(), parameter.annotation.clone());
        }
        let mut plans = vec![start];
        for (statement, (targets, expression)) in definition.statements.iter().enumerate() {
            let instance = if matches!(targets[0].name.as_str(), "_" | "None") {
                format!("#statement:{statement}")
            } else {
                targets[0].name.clone()
            };
            let mut next = Vec::new();
            for plan in plans {
                for target in targets {
                    if target.name == "None" || target.name == "_" {
                        continue;
                    }
                    if plan.names.contains_key(&target.name)
                        || plan.attribute_names.contains_key(&target.name)
                        || plan.instance_only.contains(&target.name)
                    {
                        return Err(invalid(
                            expression.span,
                            format!(
                                "binding {:?} is already defined; assignments must be SSA",
                                target.name
                            ),
                        ));
                    }
                }
                for ExpressionPlan {
                    mut plan,
                    outputs,
                    ports,
                } in self.expression(plan, expression, targets.len(), &instance, depth)?
                {
                    if outputs.len() != targets.len() {
                        if targets.len() != 1
                            || targets[0].annotation.is_some()
                            || targets[0].name == "_"
                            || targets[0].name == "None"
                        {
                            return Err(invalid(
                                expression.span,
                                "multi-result instance needs one unannotated name, or exact positional output targets",
                            ));
                        }
                        plan.instance_only.insert(targets[0].name.clone());
                        next.push(plan);
                        if next.len() > 64 {
                            return Err(limit(expression.span));
                        }
                        continue;
                    }
                    for ((target, output), port) in targets.iter().zip(outputs).zip(ports) {
                        if target.name == "None" {
                            plan.values[output] = ValuePlan::Absent;
                            continue;
                        }
                        if target.name == "_" {
                            continue;
                        }
                        if plan.names.insert(target.name.clone(), output).is_some() {
                            return Err(invalid(
                                expression.span,
                                "duplicate assignment targets; use distinct names",
                            ));
                        }
                        annotate(&mut plan, target, output, expression.span, false)?;
                        plan.binding_annotations.insert(
                            target.name.clone(),
                            target.annotation.clone().or(port.annotation),
                        );
                    }
                    next.push(plan);
                    if next.len() > 64 {
                        return Err(limit(expression.span));
                    }
                }
            }
            plans = next;
        }
        for (returned, expression) in definition.returns.iter().enumerate() {
            let instance = format!("#return:{returned}");
            let mut next = Vec::new();
            for plan in plans {
                for ExpressionPlan {
                    mut plan,
                    outputs,
                    ports,
                } in self.expression(plan, expression, 1, &instance, depth)?
                {
                    plan.outputs.extend(outputs);
                    plan.output_signature.extend(ports);
                    next.push(plan);
                    if next.len() > 64 {
                        return Err(limit(expression.span));
                    }
                }
            }
            plans = next;
        }
        for plan in &mut plans {
            let body_annotations = std::mem::take(&mut plan.annotations);
            if let Some(signature) = &definition.output_signature {
                let mut signature = signature.clone();
                if !definition.require_outputs {
                    for (port, inferred) in signature.iter_mut().zip(&plan.output_signature) {
                        port.annotation = inferred.annotation.clone();
                    }
                }
                plan.output_signature = signature;
            }
            for (index, parameter) in definition.parameters.iter().enumerate() {
                let slot = plan.names[&parameter.name];
                let discovered = definition
                    .call_parameters
                    .is_some_and(|count| index >= count);
                if definition.interface.is_some() && !discovered {
                    plan.tensor_formals.insert(slot);
                }
                if plan.attribute_types.contains_key(&parameter.name) {
                    if value_used(plan, slot) || plan.tensor_formals.contains(&slot) {
                        return Err(invalid(
                            definition.span,
                            format!(
                                "{:?} is used as both tensor and attribute; give these bindings distinct names",
                                parameter.name
                            ),
                        ));
                    }
                    annotate(plan, parameter, slot, definition.span, true)?;
                    if !discovered {
                        plan.parameters
                            .push(Formal::Attribute(parameter.name.clone()));
                    }
                    plan.names.remove(&parameter.name);
                } else {
                    annotate(plan, parameter, slot, definition.span, false)?;
                    plan.inputs.push(slot);
                    if !discovered {
                        plan.parameters.push(Formal::Value(slot));
                    } else if !value_used(plan, slot)
                        && !matches!(plan.values[slot], ValuePlan::Optional)
                    {
                        return Err(invalid(
                            definition.span,
                            format!(
                                "discovered input {:?} is unused; remove it or allow None",
                                parameter.name
                            ),
                        ));
                    }
                }
            }
            plan.annotations.extend(body_annotations);
            let signature = plan.output_signature.clone();
            if definition.named {
                if signature.len() != plan.outputs.len() {
                    return Err(invalid(
                        definition.span,
                        "return count differs from -> signature; return every port positionally",
                    ));
                }
                for (parameter, slot) in signature.iter().zip(plan.outputs.clone()) {
                    if plan
                        .names
                        .get(&parameter.name)
                        .is_some_and(|existing| *existing != slot)
                    {
                        return Err(invalid(
                            definition.span,
                            "output signature shadows another value; use its actual label",
                        ));
                    }
                    plan.names.insert(parameter.name.clone(), slot);
                    plan.binding_annotations
                        .insert(parameter.name.clone(), parameter.annotation.clone());
                    if definition.require_outputs {
                        plan.tensor_formals.insert(slot);
                        plan.required_values.insert(slot);
                        annotate(plan, parameter, slot, definition.span, false)?;
                        if parameter
                            .annotation
                            .as_ref()
                            .is_some_and(|constraints| constraints.iter().any(|c| c.name == "None"))
                        {
                            return Err(invalid(
                                definition.span,
                                "declared outputs cannot be optional; remove None from the output contract",
                            ));
                        }
                    }
                }
            }
            verify_plan(plan, definition.span)?;
        }
        let mut common = plans[0].output_signature.clone();
        for plan in &plans[1..] {
            merge_result_ports(&mut common, &plan.output_signature, true, definition.span)?;
        }
        for plan in &mut plans {
            plan.output_signature = common.clone();
            for port in &common {
                plan.binding_annotations
                    .insert(port.name.clone(), port.annotation.clone());
            }
        }
        Ok(plans)
    }

    fn expression(
        &mut self,
        plan: Plan,
        expression: &Expr,
        arity: usize,
        instance: &str,
        depth: usize,
    ) -> Result<Vec<ExpressionPlan>> {
        match &expression.kind {
            ExprKind::AnonymousCall(members, arguments) => {
                let mut reference: Option<Plan> = None;
                let mut results = Vec::new();
                for member in members {
                    let callee = self.definition(member, depth)?;
                    for child in callee.plans.iter() {
                        if reference
                            .as_ref()
                            .is_some_and(|first| !same_signature(first, child))
                        {
                            return Err(invalid(
                                expression.span,
                                "anonymous alternatives need identical interface and ordered ports; use call-expression alternatives for different signatures",
                            ));
                        }
                        reference.get_or_insert_with(|| child.clone());
                    }
                    let call = Expr {
                        kind: ExprKind::Call(member.clone(), arguments.clone(), Vec::new()),
                        span: expression.span,
                    };
                    results.extend(self.expression(plan.clone(), &call, arity, instance, depth)?);
                    if results.len() > 64 {
                        return Err(limit(expression.span));
                    }
                }
                Ok(results)
            }
            ExprKind::Alternatives(branches) => {
                let mut results: Vec<ExpressionPlan> = Vec::new();
                for branch in branches {
                    results.extend(self.expression(
                        plan.clone(),
                        branch,
                        arity,
                        instance,
                        depth,
                    )?);
                    if results.len() > 64 {
                        return Err(limit(expression.span));
                    }
                }
                let mut contracts = results[0].ports.clone();
                for result in &results[1..] {
                    merge_result_ports(&mut contracts, &result.ports, false, expression.span)?;
                }
                for result in &mut results {
                    for (port, contract) in result.ports.iter_mut().zip(&contracts) {
                        port.annotation = contract.annotation.clone();
                    }
                }
                Ok(results)
            }
            ExprKind::Call(name, arguments, attributes) if self.definitions.contains_key(name) => {
                if !attributes.is_empty() {
                    return Err(unsupported(
                        expression.span,
                        "pattern calls take positional arguments, not keyword attributes",
                    ));
                }
                let callee = self.definition(name, depth)?;
                let mut results = Vec::new();
                for child in callee.plans.iter() {
                    if child.parameters.len() != arguments.len()
                        || (child.outputs.len() != arity && arity != 1)
                    {
                        return Err(invalid(
                            expression.span,
                            format!(
                                "call {name:?} expects {} arguments and {} results, got {} and {arity}; use exact positional arity",
                                child.parameters.len(),
                                child.outputs.len(),
                                arguments.len()
                            ),
                        ));
                    }
                    let mut parent = plan.clone();
                    let mut remap = BTreeMap::new();
                    let mut attr_remap = BTreeMap::new();
                    for (formal, actual) in child.parameters.iter().zip(arguments) {
                        match formal {
                            Formal::Value(slot) => {
                                let actual = operand(&mut parent, actual)?;
                                parent.tensor_formals.insert(actual);
                                if !matches!(child.values[*slot], ValuePlan::Optional) {
                                    parent.required_values.insert(actual);
                                }
                                remap.insert(*slot, actual);
                            }
                            Formal::Attribute(name) => {
                                let ExprKind::Name(actual) = &actual.kind else {
                                    return Err(unsupported(
                                        actual.span,
                                        "attribute call arguments must be named captures",
                                    ));
                                };
                                if actual == "_" || actual == "None" {
                                    return Err(unsupported(
                                        expression.span,
                                        "attribute wildcards/None actuals are unsupported; use a named optional attribute capture",
                                    ));
                                }
                                if child
                                    .values
                                    .iter()
                                    .any(|value| matches!(value, ValuePlan::Sequence { .. }))
                                {
                                    return Err(unsupported(
                                        expression.span,
                                        "native sequence formals cannot yet be composed; put the spread in the entry pattern",
                                    ));
                                }
                                parent
                                    .attribute_types
                                    .entry(actual.clone())
                                    .or_insert((None, false));
                                parent
                                    .attribute_names
                                    .insert(actual.clone(), actual.clone());
                                attr_remap.insert(name.clone(), actual.clone());
                            }
                        }
                    }
                    for (slot, value) in child.values.iter().enumerate() {
                        remap
                            .entry(slot)
                            .or_insert_with(|| parent.value(value.clone()));
                    }
                    parent
                        .tensor_formals
                        .extend(child.tensor_formals.iter().map(|slot| remap[slot]));
                    let prefix = format!("{instance}.");
                    parent.instance_only.extend(
                        child
                            .instance_only
                            .iter()
                            .map(|name| format!("{prefix}{name}")),
                    );
                    parent
                        .required_values
                        .extend(child.required_values.iter().map(|slot| remap[slot]));
                    parent
                        .discovered_inputs
                        .extend(child.discovered_inputs.iter().map(|slot| remap[slot]));
                    let formals: BTreeSet<_> = child
                        .parameters
                        .iter()
                        .filter_map(|formal| match formal {
                            Formal::Value(slot) => Some(*slot),
                            Formal::Attribute(_) => None,
                        })
                        .collect();
                    parent.discovered_inputs.extend(
                        child
                            .inputs
                            .iter()
                            .filter(|slot| !formals.contains(slot))
                            .map(|slot| remap[slot]),
                    );
                    for (name, slot) in &child.names {
                        parent.names.insert(format!("{prefix}{name}"), remap[slot]);
                    }
                    for (name, annotation) in &child.binding_annotations {
                        parent.binding_annotations.insert(
                            format!("{prefix}{name}"),
                            scoped_annotation(annotation, &prefix),
                        );
                    }
                    for (name, types) in &child.attribute_types {
                        let mapped = attr_remap
                            .entry(name.clone())
                            .or_insert_with(|| format!("{prefix}{name}"))
                            .clone();
                        // Callee constraints cannot be silently weakened by a caller.
                        if parent.attribute_types.get(&mapped).is_some_and(|prior| {
                            prior.0.is_some() && types.0.is_some() && prior != types
                        }) {
                            return Err(unsupported(
                                expression.span,
                                "conflicting composed attribute annotations; use a single concrete attribute contract",
                            ));
                        }
                        parent.attribute_types.insert(mapped, types.clone());
                    }
                    for (name, (kind, optional)) in &child.attribute_types {
                        parent.attribute_conjunctions.push((
                            attr_remap[name].clone(),
                            kind.clone(),
                            *optional,
                        ));
                    }
                    for (name, kind, optional) in &child.attribute_conjunctions {
                        parent.attribute_conjunctions.push((
                            attr_remap[name].clone(),
                            kind.clone(),
                            *optional,
                        ));
                    }
                    for (name, canonical) in &child.attribute_names {
                        parent
                            .attribute_names
                            .insert(format!("{prefix}{name}"), attr_remap[canonical].clone());
                    }
                    for node in &child.nodes {
                        let mut node = node.clone();
                        node.instance = format!("{prefix}{}", node.instance);
                        node.inputs = node.inputs.iter().map(|slot| remap[slot]).collect();
                        node.outputs = node.outputs.iter().map(|slot| remap[slot]).collect();
                        for (_, attribute) in &mut node.attributes {
                            if let AttributePlan::Capture(name) = attribute {
                                *name = attr_remap[name].clone();
                            }
                        }
                        parent.nodes.push(node);
                    }
                    for (slot, annotations) in &child.annotations {
                        let mut annotations = annotations.clone();
                        for annotation in &mut annotations {
                            if let Some(variable) = &mut annotation.element_variable {
                                *variable = format!("{prefix}{variable}");
                            }
                            if let Some(dimensions) = &mut annotation.dimensions {
                                for dimension in dimensions {
                                    if let Dimension::Named(name) = dimension {
                                        *name = format!("{prefix}{name}");
                                    }
                                }
                            }
                        }
                        parent.annotations.push((remap[slot], annotations));
                    }
                    parent.calls += child.calls + 1;
                    verify_limits(&parent, expression.span)?;
                    let outputs = child.outputs.iter().map(|slot| remap[slot]).collect();
                    let ports = child
                        .output_signature
                        .iter()
                        .map(|port| Parameter {
                            name: port.name.clone(),
                            annotation: scoped_annotation(&port.annotation, &prefix),
                        })
                        .collect();
                    results.push(ExpressionPlan {
                        plan: parent,
                        outputs,
                        ports,
                    });
                }
                Ok(results)
            }
            ExprKind::Call(name, arguments, attributes) => {
                let mut plan = plan;
                let (domain, op) = name
                    .rsplit_once("::")
                    .or_else(|| name.rsplit_once('.'))
                    .map_or(("", name.as_str()), |(domain, op)| (domain, op));
                if name != "*"
                    && domain.is_empty()
                    && self.registry.is_some_and(|registry| !registry("", op))
                {
                    return Err(invalid(
                        expression.span,
                        format!(
                            "unresolved call {name:?}: no named pattern or registered ONNX operator; fix the name or explicitly qualify a custom operator"
                        ),
                    ));
                }
                let mut rest = None;
                let mut inputs = Vec::new();
                for (index, input) in arguments.iter().enumerate() {
                    if let ExprKind::Spread(name) = &input.kind {
                        if rest.replace(index).is_some() {
                            return Err(invalid(
                                input.span,
                                "only one variadic segment is allowed per signature; determine its length from fixed prefix/suffix",
                            ));
                        }
                        let slot = plan.names.get(name).copied().ok_or_else(|| invalid(input.span, "spread name is undeclared; add a values[min:max] or slots[min:max] header"))?;
                        if !matches!(plan.values[slot], ValuePlan::Sequence { .. }) {
                            return Err(invalid(
                                input.span,
                                "spread requires a sequence annotation; declare values[min:max] or slots[min:max]",
                            ));
                        }
                        inputs.push(slot);
                    } else {
                        inputs.push(operand(&mut plan, input)?);
                    }
                }
                let outputs: Vec<_> = (0..arity).map(|_| plan.value(ValuePlan::Present)).collect();
                let mut compiled_attributes = Vec::new();
                for (key, expression) in attributes {
                    let attribute = match &expression.kind {
                        ExprKind::Graph(pattern) => AttributePlan::Graph(pattern.clone()),
                        ExprKind::Literal(literal) => AttributePlan::Literal(literal.clone()),
                        ExprKind::Name(name) if self.definitions.contains_key(name) => {
                            if plan.names.contains_key(name) {
                                return Err(invalid(
                                    expression.span,
                                    "graph pattern reference collides with a binding; rename one of them",
                                ));
                            }
                            AttributePlan::Graph(self.definition(name, depth)?)
                        }
                        ExprKind::GraphList(names) => {
                            let patterns = names
                                .iter()
                                .map(|name| self.definition(name, depth))
                                .collect::<Result<_>>()?;
                            AttributePlan::Graphs(patterns)
                        }
                        ExprKind::Name(name) if name != "_" && name != "None" => {
                            if plan.instance_only.contains(name) {
                                return Err(invalid(
                                    expression.span,
                                    "attribute capture conflicts with a multi-result instance; use a distinct capture name",
                                ));
                            }
                            if plan.names.contains_key(name)
                                && !plan
                                    .signature
                                    .iter()
                                    .any(|parameter| &parameter.name == name)
                            {
                                return Err(invalid(
                                    expression.span,
                                    format!(
                                        "{name:?} is an assigned tensor, not an attribute formal; use a distinct keyword capture"
                                    ),
                                ));
                            }
                            plan.attribute_types
                                .entry(name.clone())
                                .or_insert((None, false));
                            plan.attribute_names.insert(name.clone(), name.clone());
                            AttributePlan::Capture(name.clone())
                        }
                        _ => {
                            return Err(unsupported(
                                expression.span,
                                "attribute constraints must be literals, capture names or concrete graph-pattern references",
                            ));
                        }
                    };
                    compiled_attributes.push((key.clone(), attribute));
                }
                plan.nodes.push(NodePlan {
                    signature: NodeSignature::Positional,
                    domain: crate::normalize_domain(domain).to_string(),
                    op: op.into(),
                    inputs,
                    outputs: outputs.clone(),
                    attributes: compiled_attributes,
                    instance: instance.into(),
                    rest,
                });
                verify_limits(&plan, expression.span)?;
                let ports = (0..arity)
                    .map(|index| Parameter {
                        name: if arity == 1 {
                            "output".into()
                        } else {
                            format!("output{index}")
                        },
                        annotation: None,
                    })
                    .collect();
                Ok(vec![ExpressionPlan {
                    plan,
                    outputs,
                    ports,
                }])
            }
            _ if arity == 1 => {
                let mut plan = plan;
                let value = operand(&mut plan, expression)?;
                let port = if let ExprKind::Name(name) = &expression.kind {
                    Parameter {
                        name: name.rsplit('.').next().unwrap().into(),
                        annotation: plan.binding_annotations.get(name).cloned().flatten(),
                    }
                } else {
                    Parameter {
                        name: "output".into(),
                        annotation: None,
                    }
                };
                Ok(vec![ExpressionPlan {
                    plan,
                    outputs: vec![value],
                    ports: vec![port],
                }])
            }
            _ => Err(unsupported(
                expression.span,
                "multi-result assignment needs an operator or pattern call",
            )),
        }
    }
}

fn scoped_annotation(
    annotation: &Option<Vec<RawConstraint>>,
    prefix: &str,
) -> Option<Vec<RawConstraint>> {
    annotation.as_ref().map(|constraints| {
        constraints
            .iter()
            .cloned()
            .map(|mut constraint| {
                if let Some(name) = &mut constraint.element_variable {
                    *name = format!("{prefix}{name}");
                }
                if let Some(dimensions) = &mut constraint.dimensions {
                    for dimension in dimensions {
                        if let Dimension::Named(name) = dimension {
                            *name = format!("{prefix}{name}");
                        }
                    }
                }
                constraint
            })
            .collect()
    })
}

fn merge_result_ports(
    target: &mut [Parameter],
    source: &[Parameter],
    require_names: bool,
    span: SourceSpan,
) -> Result<()> {
    if target.len() != source.len() {
        return Err(invalid(
            span,
            "expression alternatives have different result arities; return the same number of tensors",
        ));
    }
    for (first, port) in target.iter_mut().zip(source) {
        if require_names && first.name != port.name {
            return Err(invalid(
                span,
                "pattern alternatives change inferred output labels; declare a common output signature",
            ));
        }
        match (&first.annotation, &port.annotation) {
            (Some(a), Some(b)) if a != b => {
                return Err(invalid(
                    span,
                    "expression alternatives have incompatible result annotations; use structurally identical result constraints",
                ));
            }
            (None, _) => first.annotation = port.annotation.clone(),
            _ => {}
        }
    }
    Ok(())
}

fn operand(plan: &mut Plan, expression: &Expr) -> Result<usize> {
    match &expression.kind {
        ExprKind::Name(name) if name == "_" => Ok(plan.value(ValuePlan::Present)),
        ExprKind::Name(name) if name == "None" => Ok(plan.value(ValuePlan::Absent)),
        ExprKind::Name(name) => {
            if plan.instance_only.contains(name) {
                return Err(invalid(
                    expression.span,
                    format!(
                        "instance-only binding {name:?} has multiple results; use {name}.port or positional assignment targets"
                    ),
                ));
            }
            let slot = plan.names.get(name).copied().ok_or_else(|| invalid(expression.span,
                format!("undefined tensor binding {name:?}; declare it in the header or assign it before use")))?;
            if matches!(plan.values[slot], ValuePlan::Sequence { .. }) {
                return Err(invalid(
                    expression.span,
                    "sequence used as one tensor; use ...name in an operator's input signature",
                ));
            }
            Ok(slot)
        }
        ExprKind::Literal(literal) => {
            let tensor = match literal {
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
                Literal::String(_) => {
                    return Err(unsupported(
                        expression.span,
                        "positional string literals are not tensor patterns; capture the value and inspect it",
                    ));
                }
            };
            Ok(plan.value(ValuePlan::Literal(tensor)))
        }
        _ => Err(unsupported(
            expression.span,
            "nested operator operands are not supported; assign each call before using its result",
        )),
    }
}

fn annotate(
    plan: &mut Plan,
    parameter: &Parameter,
    slot: usize,
    span: SourceSpan,
    attribute: bool,
) -> Result<()> {
    let Some(raw) = &parameter.annotation else {
        if !attribute {
            plan.annotations.push((
                slot,
                vec![Constraint {
                    tensor: false,
                    absent: false,
                    immutable: false,
                    element: None,
                    element_variable: None,
                    dimensions: None,
                    prefix: false,
                }],
            ));
        }
        return Ok(());
    };
    if raw.iter().any(|constraint| constraint.sequence.is_some()) {
        if attribute || raw.len() != 1 || !matches!(plan.values[slot], ValuePlan::Sequence { .. }) {
            return Err(unsupported(
                span,
                "sequence annotations are only allowed on native header inputs, without type alternatives",
            ));
        }
        return Ok(());
    }
    if attribute {
        let concrete: Vec<_> = raw.iter().filter(|c| c.name != "None").collect();
        if concrete.len() != 1
            || raw.iter().any(|c| {
                c.immutable
                    || c.element_variable.is_some()
                    || c.dimensions.as_ref().is_none_or(|d| !d.is_empty())
                    || c.prefix
            })
            || ![
                "int", "float", "string", "ints", "floats", "strings", "tensor",
            ]
            .contains(&concrete[0].name.as_str())
        {
            return Err(unsupported(
                span,
                format!(
                    "invalid attribute annotation for {:?}; use one attribute type, optionally | None",
                    parameter.name
                ),
            ));
        }
        let optional = raw.iter().any(|c| c.name == "None");
        plan.attribute_types.insert(
            parameter.name.clone(),
            (Some(concrete[0].name.clone()), optional),
        );
        return Ok(());
    }
    let constraints = raw
        .iter()
        .map(|constraint| constraint_from_raw(constraint, span))
        .collect::<Result<Vec<_>>>()?;
    for (index, constraint) in constraints.iter().enumerate() {
        if constraints[..index].contains(constraint) {
            return Err(invalid(
                span,
                "duplicate annotation alternatives; keep each distinct constraint once",
            ));
        }
    }
    if constraints.iter().all(|c| c.absent) {
        return Err(invalid(
            span,
            "None alone is not an annotation; include a present-value alternative",
        ));
    }
    if constraints.iter().any(|c| c.absent) {
        if !plan.inputs.contains(&slot) && plan.nodes.iter().any(|n| n.outputs.contains(&slot)) {
            return Err(invalid(
                span,
                "assigned outputs cannot be absent; None is only for header slots",
            ));
        }
        plan.values[slot] = ValuePlan::Optional;
    }
    plan.annotations.push((slot, constraints));
    Ok(())
}

fn constraint_from_raw(raw: &RawConstraint, span: SourceSpan) -> Result<Constraint> {
    let element = match raw.name.as_str() {
        "None" | "tensor" | "value" => None,
        name => Some(element_type(name).ok_or_else(|| unsupported(span,
            format!("tensor annotation {name:?} is unsupported; use an ONNX tensor element name or a metadata predicate")))?),
    };
    if raw.name == "None"
        && (raw.immutable
            || raw.element_variable.is_some()
            || raw.dimensions.as_ref().is_none_or(|d| !d.is_empty()))
    {
        return Err(invalid(
            span,
            "None cannot have const, element variables or dimensions",
        ));
    }
    if raw.element_variable.is_some() && raw.name != "tensor" {
        return Err(invalid(
            span,
            "element variables require tensor<T>; do not attach them to a concrete element type",
        ));
    }
    Ok(Constraint {
        tensor: raw.name != "None" && raw.name != "value",
        absent: raw.name == "None",
        immutable: raw.immutable,
        element,
        element_variable: raw.element_variable.clone(),
        dimensions: raw.dimensions.clone(),
        prefix: raw.prefix && raw.dimensions.is_some(),
    })
}

fn element_type(name: &str) -> Option<DataType> {
    Some(match name {
        "float" => DataType::Float32,
        "double" => DataType::Float64,
        "float16" => DataType::Float16,
        "bfloat16" => DataType::BFloat16,
        "int64" => DataType::Int64,
        "int32" => DataType::Int32,
        "int16" => DataType::Int16,
        "int8" => DataType::Int8,
        "uint64" => DataType::Uint64,
        "uint32" => DataType::Uint32,
        "uint16" => DataType::Uint16,
        "uint8" => DataType::Uint8,
        "bool" => DataType::Bool,
        "string" => DataType::String,
        "complex64" => DataType::Complex64,
        "complex128" => DataType::Complex128,
        "float8e4m3fn" => DataType::Float8E4M3FN,
        "float8e4m3fnuz" => DataType::Float8E4M3FNUZ,
        "float8e5m2" => DataType::Float8E5M2,
        "float8e5m2fnuz" => DataType::Float8E5M2FNUZ,
        "float8e8m0" => DataType::Float8E8M0,
        "int4" => DataType::Int4,
        "uint4" => DataType::Uint4,
        "float4e2m1" => DataType::Float4E2M1,
        "int2" => DataType::Int2,
        "uint2" => DataType::Uint2,
        _ => return None,
    })
}

fn value_used(plan: &Plan, slot: usize) -> bool {
    plan.nodes.iter().any(|n| n.inputs.contains(&slot)) || plan.outputs.contains(&slot)
}

fn verify_limits(plan: &Plan, span: SourceSpan) -> Result<()> {
    if plan.nodes.len() > 64
        || plan.values.len() > 256
        || plan.annotations.len() > 256
        || plan.calls > 256
        || plan.attribute_conjunctions.len() > 256
        || plan
            .annotations
            .iter()
            .try_fold(1usize, |n, (_, a)| n.checked_mul(a.len()))
            .is_none_or(|n| n > 64)
    {
        return Err(limit(span));
    }
    Ok(())
}

fn verify_plan(plan: &mut Plan, span: SourceSpan) -> Result<()> {
    verify_limits(plan, span)?;
    plan.producers = vec![None; plan.values.len()];
    for (ordinal, node) in plan.nodes.iter().enumerate() {
        for &output in &node.outputs {
            if plan.producers[output].replace(ordinal).is_some() {
                return Err(invalid(
                    span,
                    "a pattern value has multiple producers; use distinct SSA assignments",
                ));
            }
        }
    }
    let mut live: HashSet<_> = plan.outputs.iter().copied().collect();
    for node in plan.nodes.iter().rev() {
        if !node.outputs.iter().any(|v| live.contains(v)) {
            return Err(invalid(
                span,
                "an operator does not reach any return in this alternative; remove dead pattern work",
            ));
        }
        live.extend(node.inputs.iter().copied());
    }
    for name in plan.attribute_types.keys() {
        if plan
            .names
            .get(name)
            .is_some_and(|slot| value_used(plan, *slot))
        {
            return Err(invalid(
                span,
                format!("capture {name:?} mixes tensor and attribute kinds; use distinct names"),
            ));
        }
    }
    let variables: HashSet<_> = plan
        .annotations
        .iter()
        .flat_map(|(_, constraints)| constraints)
        .flat_map(|c| {
            c.element_variable.iter().map(String::as_str).chain(
                c.dimensions.iter().flatten().filter_map(|d| {
                    if let Dimension::Named(n) = d {
                        Some(n.as_str())
                    } else {
                        None
                    }
                }),
            )
        })
        .collect();
    if variables.iter().any(|name| {
        plan.names.contains_key(*name)
            || plan.attribute_types.contains_key(*name)
            || plan.instance_only.contains(*name)
    }) {
        return Err(invalid(
            span,
            "dimension/element variable collides with a tensor or attribute; use separate names",
        ));
    }
    if variables.len() > 256 {
        return Err(limit(span));
    }
    let elements: HashSet<_> = plan
        .annotations
        .iter()
        .flat_map(|(_, cs)| cs)
        .filter_map(|c| c.element_variable.as_deref())
        .collect();
    if plan
        .annotations
        .iter()
        .flat_map(|(_, cs)| cs)
        .flat_map(|c| c.dimensions.iter().flatten())
        .any(|d| matches!(d, Dimension::Named(name) if elements.contains(name.as_str())))
    {
        return Err(invalid(
            span,
            "a name cannot be both an element-type and dimension variable; use distinct names",
        ));
    }
    Ok(())
}

pub(super) fn verify_entry(plan: &Plan) -> Result<()> {
    for &slot in &plan.inputs {
        if !value_used(plan, slot) && !matches!(plan.values[slot], ValuePlan::Optional) {
            return Err(invalid(
                SourceSpan::default(),
                "required header parameter is unused; remove it, allow None, or select a whole-graph entry for unused formals",
            ));
        }
    }
    let Some(root) = plan.nodes.last() else {
        return Err(invalid(
            SourceSpan::default(),
            "operator entry has no anchor; use graph_pattern and match_graph for passthrough bodies",
        ));
    };
    let produced: HashSet<_> = plan
        .nodes
        .iter()
        .flat_map(|n| &n.outputs)
        .copied()
        .collect();
    let outputs: HashSet<_> = plan.outputs.iter().copied().collect();
    if outputs.len() != plan.outputs.len() || outputs.iter().any(|o| !produced.contains(o)) {
        return Err(invalid(
            SourceSpan::default(),
            "operator entries must return distinct produced values; use whole-graph matching for aliases",
        ));
    }
    let mut connected: HashSet<_> = root
        .inputs
        .iter()
        .chain(&root.outputs)
        .copied()
        .filter(|s| produced.contains(s))
        .collect();
    let mut visited = HashSet::new();
    loop {
        let before = visited.len();
        for (i, node) in plan.nodes.iter().enumerate() {
            if node
                .inputs
                .iter()
                .chain(&node.outputs)
                .any(|s| connected.contains(s))
            {
                visited.insert(i);
                connected.extend(
                    node.inputs
                        .iter()
                        .chain(&node.outputs)
                        .copied()
                        .filter(|s| produced.contains(s)),
                );
            }
        }
        if visited.len() == before {
            break;
        }
    }
    if visited.len() != plan.nodes.len() {
        return Err(invalid(
            SourceSpan::default(),
            "operator entry is disconnected through produced edges; use whole-graph matching or connect its components",
        ));
    }
    Ok(())
}

fn same_signature(a: &Plan, b: &Plan) -> bool {
    a.interface == b.interface
        && a.signature[..a.parameters.len()] == b.signature[..b.parameters.len()]
        && a.output_signature == b.output_signature
}

fn validate_interface(inputs: &[Parameter], outputs: &[Parameter], span: SourceSpan) -> Result<()> {
    if outputs.is_empty() {
        return Err(invalid(span, "interface needs at least one output port"));
    }
    let mut names = BTreeSet::new();
    let mut plan = Plan::empty();
    for (ports, output) in [(inputs, false), (outputs, true)] {
        for port in ports {
            if !names.insert(&port.name) {
                return Err(invalid(
                    span,
                    format!(
                        "duplicate interface port {:?}; input/output labels must be distinct",
                        port.name
                    ),
                ));
            }
            if let Some(alternatives) = &port.annotation {
                for raw in alternatives {
                    let constraint = constraint_from_raw(raw, span)?;
                    if constraint.absent && output {
                        return Err(invalid(
                            span,
                            "interface outputs cannot be optional; remove None",
                        ));
                    }
                }
            }
            let slot = plan.value(ValuePlan::Present);
            plan.names.insert(port.name.clone(), slot);
            annotate(&mut plan, port, slot, span, false)?;
        }
    }
    verify_plan(&mut plan, span)
}

fn dependencies(
    definition: &DefinitionKind,
    definitions: &BTreeMap<String, DefinitionKind>,
) -> BTreeSet<String> {
    fn visit(
        expression: &Expr,
        definitions: &BTreeMap<String, DefinitionKind>,
        found: &mut BTreeSet<String>,
    ) {
        match &expression.kind {
            ExprKind::AnonymousCall(members, arguments) => {
                found.extend(members.iter().cloned());
                for argument in arguments {
                    visit(argument, definitions, found);
                }
            }
            ExprKind::Call(name, inputs, attributes) => {
                if definitions.contains_key(name) {
                    found.insert(name.clone());
                }
                for input in inputs {
                    visit(input, definitions, found);
                }
                for (_, attribute) in attributes {
                    if let ExprKind::Name(name) = &attribute.kind
                        && definitions.contains_key(name)
                    {
                        found.insert(name.clone());
                    }
                    visit(attribute, definitions, found);
                }
            }
            ExprKind::GraphList(names) => found.extend(names.iter().cloned()),
            ExprKind::Alternatives(branches) => {
                for branch in branches {
                    visit(branch, definitions, found);
                }
            }
            _ => {}
        }
    }
    let mut found = BTreeSet::new();
    match definition {
        DefinitionKind::Interface(..) => {}
        DefinitionKind::Alternatives(members, _) => found.extend(members.iter().cloned()),
        DefinitionKind::Body(body) => {
            for (_, expression) in &body.statements {
                visit(expression, definitions, &mut found);
            }
            for expression in &body.returns {
                visit(expression, definitions, &mut found);
            }
        }
    }
    found
}
