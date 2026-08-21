use std::collections::{BTreeMap, BTreeSet};

use crate::schema::{
    EffectTransition, WorkflowBranchEffectMerge, WorkflowCarry, WorkflowLoopCarry,
    WorkflowLoopEffect, WorkflowNode, WorkflowSpec, WorkflowStep,
};

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledWorkflow {
    pub initial_effects: BTreeMap<String, String>,
    pub graph: WorkflowNode,
}

pub fn compile_workflow(workflow: &WorkflowSpec) -> Result<CompiledWorkflow, String> {
    let mut compiler = Compiler {
        workflow,
        next_token: 1,
    };
    let mut domains = BTreeSet::new();
    for component in workflow.components.values() {
        domains.extend(component.effects.iter().cloned());
    }
    domains.extend(workflow.effects.keys().cloned());
    domains.extend(workflow.state.keys().map(|cell| format!("state:{cell}")));
    if workflow.steps.iter().any(contains_emit) {
        domains.insert("stream".to_string());
    }
    let initial_effects = domains
        .into_iter()
        .map(|domain| {
            let token = format!("{domain}.0");
            (domain, token)
        })
        .collect::<BTreeMap<_, _>>();
    let mut effects = initial_effects.clone();
    let graph = compiler.lower_sequence(&workflow.steps, &mut effects)?;
    Ok(CompiledWorkflow {
        initial_effects,
        graph,
    })
}

struct Compiler<'a> {
    workflow: &'a WorkflowSpec,
    next_token: usize,
}

impl Compiler<'_> {
    fn lower_sequence(
        &mut self,
        steps: &[WorkflowStep],
        effects: &mut BTreeMap<String, String>,
    ) -> Result<WorkflowNode, String> {
        let mut nodes = Vec::with_capacity(steps.len());
        for step in steps {
            nodes.push(self.lower_step(step, effects)?);
        }
        Ok(WorkflowNode::Sequence { nodes })
    }

    fn lower_step(
        &mut self,
        step: &WorkflowStep,
        effects: &mut BTreeMap<String, String>,
    ) -> Result<WorkflowNode, String> {
        match step {
            WorkflowStep::Sequence { steps } => self.lower_sequence(steps, effects),
            WorkflowStep::Invoke {
                component,
                inputs,
                outputs,
            } => {
                let declaration = self.workflow.components.get(component).ok_or_else(|| {
                    format!("workflow invoke references unknown component '{component}'")
                })?;
                let mut transitions = BTreeMap::new();
                for domain in &declaration.effects {
                    transitions.insert(domain.clone(), self.transition(domain, effects)?);
                }
                Ok(WorkflowNode::Invoke {
                    component: component.clone(),
                    inputs: inputs.clone(),
                    outputs: outputs.clone(),
                    effects: transitions,
                })
            }
            WorkflowStep::Loop {
                setup,
                steps,
                continue_when,
                max_iterations,
                termination,
                iteration,
                carried,
            } => {
                let setup = self.lower_sequence(setup, effects)?;
                let incoming_effects = effects.clone();
                let body_inputs = incoming_effects
                    .keys()
                    .map(|domain| (domain.clone(), self.token(domain)))
                    .collect::<BTreeMap<_, _>>();
                let mut body_effects = body_inputs.clone();
                let mut lowered_carries = Vec::with_capacity(carried.len());
                for carry in carried {
                    lowered_carries.push(self.lower_carry_read(carry, &mut body_effects)?);
                }
                let body = self.lower_sequence(steps, &mut body_effects)?;
                for carry in &mut lowered_carries {
                    let domain = format!("state:{}", carry.cell);
                    carry.write_effect = self.transition(&domain, &mut body_effects)?;
                }
                let mut loop_effects = BTreeMap::new();
                for (domain, body_input) in body_inputs {
                    let body_output = body_effects
                        .get(&domain)
                        .cloned()
                        .unwrap_or_else(|| body_input.clone());
                    if body_output == body_input {
                        continue;
                    }
                    let incoming = incoming_effects
                        .get(&domain)
                        .cloned()
                        .ok_or_else(|| format!("workflow loop effect '{domain}' has no input"))?;
                    let produces = self.token(&domain);
                    effects.insert(domain.clone(), produces.clone());
                    loop_effects.insert(
                        domain,
                        WorkflowLoopEffect {
                            incoming,
                            body_input,
                            body_output,
                            produces,
                        },
                    );
                }
                Ok(WorkflowNode::Loop {
                    setup: Box::new(setup),
                    body: Box::new(body),
                    continue_when: continue_when.clone(),
                    max_iterations: max_iterations.clone(),
                    termination: termination.clone(),
                    iteration: iteration.clone(),
                    carried: lowered_carries,
                    effects: loop_effects,
                })
            }
            WorkflowStep::Branch {
                predicate,
                cases,
                default,
                outputs,
            } => {
                let incoming = effects.clone();
                let mut lowered_cases = BTreeMap::new();
                let mut case_effects = BTreeMap::new();
                for (case, step) in cases {
                    let mut local = incoming.clone();
                    lowered_cases.insert(case.clone(), self.lower_step(step, &mut local)?);
                    case_effects.insert(case.clone(), local);
                }
                let (lowered_default, default_effects) = if let Some(default) = default {
                    let mut local = incoming.clone();
                    let lowered = self.lower_step(default, &mut local)?;
                    (Some(Box::new(lowered)), Some(local))
                } else {
                    (None, None)
                };
                let mut changed = BTreeSet::new();
                for local in case_effects.values() {
                    changed.extend(
                        local
                            .iter()
                            .filter(|(domain, token)| incoming.get(*domain) != Some(*token))
                            .map(|(domain, _)| domain.clone()),
                    );
                }
                if let Some(local) = &default_effects {
                    changed.extend(
                        local
                            .iter()
                            .filter(|(domain, token)| incoming.get(*domain) != Some(*token))
                            .map(|(domain, _)| domain.clone()),
                    );
                }
                let mut merges = BTreeMap::new();
                for domain in changed {
                    let incoming_token = incoming.get(&domain).cloned().ok_or_else(|| {
                        format!("workflow effect '{domain}' has no initial token")
                    })?;
                    let cases = case_effects
                        .iter()
                        .map(|(case, local)| {
                            (
                                case.clone(),
                                local
                                    .get(&domain)
                                    .cloned()
                                    .unwrap_or_else(|| incoming_token.clone()),
                            )
                        })
                        .collect();
                    let default = default_effects.as_ref().map(|local| {
                        local
                            .get(&domain)
                            .cloned()
                            .unwrap_or_else(|| incoming_token.clone())
                    });
                    let produces = self.token(&domain);
                    effects.insert(domain.clone(), produces.clone());
                    merges.insert(
                        domain,
                        WorkflowBranchEffectMerge {
                            incoming: incoming_token,
                            cases,
                            default,
                            produces,
                        },
                    );
                }
                Ok(WorkflowNode::Branch {
                    predicate: predicate.clone(),
                    cases: lowered_cases,
                    default: lowered_default,
                    outputs: outputs.clone(),
                    effects: merges,
                })
            }
            WorkflowStep::Emit {
                value,
                when,
                valid_length,
                output,
                mode,
                axis,
            } => Ok(WorkflowNode::Emit {
                value: value.clone(),
                when: when.clone(),
                valid_length: valid_length.clone(),
                output: output.clone(),
                mode: mode.clone(),
                axis: *axis,
                effect_name: "stream".to_string(),
                effect: self.transition("stream", effects)?,
            }),
        }
    }

    fn lower_carry_read(
        &mut self,
        carry: &WorkflowCarry,
        effects: &mut BTreeMap<String, String>,
    ) -> Result<WorkflowLoopCarry, String> {
        let state =
            self.workflow.state.get(&carry.cell).ok_or_else(|| {
                format!("workflow loop carries unknown state cell '{}'", carry.cell)
            })?;
        let initial = carry
            .initial
            .clone()
            .unwrap_or_else(|| state.initializer.clone());
        let domain = format!("state:{}", carry.cell);
        Ok(WorkflowLoopCarry {
            cell: carry.cell.clone(),
            current: initial,
            body_input: carry.cell.clone(),
            body_output: carry.next.clone(),
            next: carry.cell.clone(),
            read_effect: self.transition(&domain, effects)?,
            write_effect: EffectTransition {
                consumes: String::new(),
                produces: String::new(),
            },
        })
    }

    fn transition(
        &mut self,
        domain: &str,
        effects: &mut BTreeMap<String, String>,
    ) -> Result<EffectTransition, String> {
        let consumes = effects
            .get(domain)
            .cloned()
            .ok_or_else(|| format!("workflow effect '{domain}' has no initial token"))?;
        let produces = self.token(domain);
        effects.insert(domain.to_string(), produces.clone());
        Ok(EffectTransition { consumes, produces })
    }

    fn token(&mut self, domain: &str) -> String {
        let token = format!("{domain}.{}", self.next_token);
        self.next_token += 1;
        token
    }
}

fn contains_emit(step: &WorkflowStep) -> bool {
    match step {
        WorkflowStep::Sequence { steps } => steps.iter().any(contains_emit),
        WorkflowStep::Loop { setup, steps, .. } => {
            setup.iter().any(contains_emit) || steps.iter().any(contains_emit)
        }
        WorkflowStep::Branch { cases, default, .. } => {
            cases.values().any(contains_emit) || default.as_deref().is_some_and(contains_emit)
        }
        WorkflowStep::Emit { .. } => true,
        WorkflowStep::Invoke { .. } => false,
    }
}
