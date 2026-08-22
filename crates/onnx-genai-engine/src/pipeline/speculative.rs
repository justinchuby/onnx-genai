//! Chained speculative proposal driving, owned by the universal interpreter.
//!
//! A `chained` proposer emits one distribution per invocation, so materializing
//! a proposal block means driving the *same* typed component repeatedly while
//! threading its recurrence forward. That loop used to live in a direct-`Engine`
//! Rust `propose()` bound to ORT `Session`s, which made it a second execution
//! engine that the backend-neutral component seam could not drive. It lives here
//! now: every proposer step runs through
//! [`WorkflowRuntime::invoke_component_values`], so ORT and native execute the
//! identical chain.
//!
//! Everything the loop needs is read from the package's
//! `speculative.proposal_execution` contract — never inferred from a model name,
//! a port spelling, or a tensor shape:
//!
//! * `token_embedding_input` — the proposer port receiving the fused input.
//! * `logits_output` — the port carrying the next-token distribution.
//! * `recurrent[]` — `{state, input, output}` triples the loop threads and
//!   checkpoints, one workflow state cell each.
//! * `folded_carry_output` — the proposer output carrying carry_k (k >= 1).
//! * `folded_carry_seed` — `{component, output}`: the target output read as
//!   carry_0.
//! * `token_embedding` — `{component, table}`: the embedding table the leading
//!   half of the fused input is gathered from.
//!
//! A folded carry re-enters as the *trailing* half of the fused input and owns
//! no state cell, so it is recomputed from committed tokens on rejection rather
//! than restored — which is why it never appears in `rollback_state`.

use std::collections::BTreeMap;

use anyhow::Context as _;
use onnx_genai_metadata::{
    SpeculativeContract, SpeculativeProposalExecution, SpeculativeRecurrenceBinding, WorkflowSpec,
    WorkflowStep,
};
use onnx_genai_ort::{DataType, Value};

use super::{PipelineTensors, WorkflowRuntime};

/// One materialized chained proposal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainedProposal {
    /// The proposal block: the target's guaranteed token followed by the drafts
    /// the proposer chain produced, in order.
    pub tokens: Vec<i64>,
    /// Proposer invocations this proposal cost, including the bootstrap step.
    pub proposer_invocations: usize,
}

impl ChainedProposal {
    /// Draft tokens only — the block minus the target's guaranteed first token.
    pub fn drafts(&self) -> &[i64] {
        &self.tokens[1..]
    }
}

/// How a proposal block compared against the target's own tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposalAcceptance {
    /// Number of proposal-block positions the target confirmed, always >= 1:
    /// position 0 is the target's own guaranteed token.
    pub accepted: usize,
    /// Index of the first rejected draft within the proposal block, if any.
    pub rejected_at: Option<usize>,
    /// Tokens to commit for this verification pass.
    pub committed: Vec<i64>,
}

impl ProposalAcceptance {
    /// Whether a draft was rejected and declared rollback state must be undone.
    pub fn requires_rollback(&self) -> bool {
        self.rejected_at.is_some()
    }
}

/// Inputs a chained proposal needs beyond what the package declares.
#[derive(Debug, Clone, Copy)]
pub struct ChainedProposalOptions {
    /// The last committed context token, whose embedding seeds the first fused
    /// input. carry_0 is the target's hidden state for exactly this position.
    pub seed_token: i64,
    /// The target's own next token, taken for free and used to condition every
    /// draft that follows it.
    pub guaranteed_token: i64,
    /// Proposal block width, capped by the contract's `max_proposal_width`.
    pub width: usize,
}

/// Resolved, contract-derived view of a chained proposal execution.
#[derive(Debug)]
struct ChainedPlan<'a> {
    proposer: &'a str,
    token_embedding_input: &'a str,
    logits_output: &'a str,
    recurrent: &'a [SpeculativeRecurrenceBinding],
    folded_carry_output: Option<&'a str>,
    /// SSA value the target's `folded_carry_seed` output is bound to.
    folded_carry_seed_value: Option<String>,
    /// Embedding table `{component, table}`, when a folded carry is declared.
    token_embedding: Option<(&'a str, &'a str)>,
    /// Proposer port -> SSA value, from the workflow's own invocation of it.
    proposer_bindings: BTreeMap<String, String>,
    /// Proposer port -> SSA value for its outputs, from the same invocation.
    proposer_outputs: BTreeMap<String, String>,
}

impl WorkflowRuntime {
    /// The package's speculative compatibility contract, when it declares one.
    pub fn speculative_contract(&self) -> Option<&SpeculativeContract> {
        self.speculative.as_ref()
    }

    /// Drive a chained speculative proposal through the interpreter's own
    /// component seam.
    ///
    /// `run` holds the SSA values of the completed verification pass — the same
    /// map [`WorkflowRuntime::run_workflow`] produces. Every proposer port is
    /// bound from the workflow's own invocation of that component, so the
    /// borrowed read-only shared KV, position ids, and masks are exactly the
    /// tensors the package declared; only `token_embedding_input` is overridden,
    /// because the fused `concat(embed(last_token), carry)` is what the chain
    /// recomputes each step.
    pub fn propose_chained(
        &self,
        run: &PipelineTensors,
        options: ChainedProposalOptions,
    ) -> anyhow::Result<ChainedProposal> {
        let contract = self
            .speculative
            .as_ref()
            .context("package declares no speculative contract")?;
        let plan = ChainedPlan::resolve(contract, &self.workflow)?;
        anyhow::ensure!(
            options.width >= 1,
            "chained proposal width must be at least 1"
        );
        anyhow::ensure!(
            options.width <= contract.max_proposal_width,
            "requested proposal width {} exceeds the package's max_proposal_width {}",
            options.width,
            contract.max_proposal_width
        );

        let declaration = self
            .workflow
            .components
            .get(plan.proposer)
            .with_context(|| format!("workflow component '{}' is undeclared", plan.proposer))?;
        // A chained step advances exactly one position, so every proposer port
        // that shares the fused input's position symbol narrows to that step's
        // single position. The symbol comes from the declared port contracts, so
        // a package whose position axis sits elsewhere — or whose mask is keyed
        // on a separate `kv_sequence` symbol, as a borrowed-KV drafter's is —
        // narrows exactly the ports it declared and leaves the rest alone.
        let position_symbol = position_symbol(declaration, plan.token_embedding_input);

        // Fixed proposer bindings: everything the workflow binds except the
        // fused input the chain rebuilds each step.
        let mut fixed: Vec<(String, Value)> = Vec::new();
        for (port, value_name) in &plan.proposer_bindings {
            if port == plan.token_embedding_input {
                continue;
            }
            if plan.recurrent.iter().any(|binding| &binding.input == port) {
                continue;
            }
            let value = run.get(value_name).with_context(|| {
                format!(
                    "chained proposer '{}' input '{port}' references unavailable workflow value \
                     '{value_name}'",
                    plan.proposer
                )
            })?;
            let value = self.host_copy_of(value)?;
            let narrowed = match position_symbol
                .as_deref()
                .and_then(|symbol| symbol_axis(declaration, port, symbol))
            {
                Some(axis) => last_position_along(&value, axis).with_context(|| {
                    format!(
                        "failed to narrow chained proposer '{}' input '{port}' to one position",
                        plan.proposer
                    )
                })?,
                None => value,
            };
            fixed.push((port.clone(), narrowed));
        }

        // Loop-carried recurrent state: seeded from the declared workflow state
        // cell, threaded input -> output every step.
        let mut recurrent_state: BTreeMap<&str, Value> = BTreeMap::new();
        for binding in plan.recurrent {
            let seed = self
                .workflow_session_state_value(&binding.state)
                .or_else(|| run.get(&binding.state).and_then(|v| clone_value(v).ok()))
                .with_context(|| {
                    format!(
                        "chained proposer recurrence '{}' has no value for state cell '{}'",
                        binding.input, binding.state
                    )
                })?;
            recurrent_state.insert(binding.input.as_str(), seed);
        }

        // Folded carry: carry_0 is the target output the contract names, taken
        // at its most recent position (see `last_position_rows`).
        let mut carry =
            match &plan.folded_carry_seed_value {
                Some(value_name) => {
                    let seed = run.get(value_name).with_context(|| {
                        format!(
                            "folded_carry_seed names target output value '{value_name}', which the \
                         verification pass did not produce"
                        )
                    })?;
                    Some(last_position_rows(&self.host_copy_of(seed)?).context(
                        "folded_carry_seed must name a float32 per-position hidden output",
                    )?)
                }
                None => None,
            };

        let embedding = match plan.token_embedding {
            Some((component, table)) => Some(self.embedding_table(component, table)?),
            None => None,
        };

        // The fused input's declared shape tells the loop how wide each half is
        // and what batch/query extents to build; take it from the port contract
        // the workflow already bound for the single-pass invocation.
        let fused_template = run
            .get(
                plan.proposer_bindings
                    .get(plan.token_embedding_input)
                    .with_context(|| {
                        format!(
                            "chained proposer '{}' does not bind its declared \
                             token_embedding_input '{}'",
                            plan.proposer, plan.token_embedding_input
                        )
                    })?,
            )
            .context("the workflow's fused proposer input value is unavailable")?;
        anyhow::ensure!(
            fused_template.dtype() == DataType::Float32,
            "chained proposal driving requires a float32 fused input, found {:?}",
            fused_template.dtype()
        );
        // One proposal step advances exactly one position, so the fused input a
        // step binds keeps the workflow's batch extent and its declared width
        // but holds a single position, whatever the single-pass binding's
        // sequence extent was.
        let fused_shape = single_position_shape(fused_template.shape())
            .context("the workflow's fused proposer input has an unusable shape")?;
        let fused_width = *fused_shape
            .last()
            .context("fused proposer input has rank 0")? as usize;
        let batch_rows = fused_shape[..fused_shape.len() - 1]
            .iter()
            .try_fold(1usize, |total, dimension| {
                usize::try_from(*dimension)
                    .ok()
                    .and_then(|dimension| total.checked_mul(dimension))
            })
            .context("fused proposer input has an unusable shape")?;

        let outputs = self.proposer_output_bindings(&plan);
        let mut tokens = Vec::with_capacity(options.width);
        tokens.push(options.guaranteed_token);
        let mut last_token = options.seed_token;
        let mut invocations = 0usize;

        for step in 0..options.width {
            let fused = build_fused_input(
                &fused_shape,
                fused_width,
                batch_rows,
                embedding.as_ref(),
                last_token,
                carry.as_deref(),
            )
            .with_context(|| format!("failed to build fused proposer input at step {step}"))?;

            let mut bound: Vec<(&str, &Value)> = fixed
                .iter()
                .map(|(port, value)| (port.as_str(), value))
                .collect();
            bound.push((plan.token_embedding_input, &fused));
            for (port, value) in &recurrent_state {
                bound.push((port, value));
            }

            let produced = self
                .invoke_component_values(plan.proposer, &bound, &outputs)
                .with_context(|| {
                    format!("chained proposer '{}' step {step} failed", plan.proposer)
                })?;
            invocations += 1;

            let mut logits = None;
            let mut next_carry = None;
            let mut next_recurrent: BTreeMap<&str, Value> = BTreeMap::new();
            for (port, value) in produced {
                if port == plan.logits_output {
                    logits = Some(value);
                } else if Some(port.as_str()) == plan.folded_carry_output {
                    next_carry = Some(value);
                } else if let Some(binding) =
                    plan.recurrent.iter().find(|binding| binding.output == port)
                {
                    next_recurrent.insert(binding.input.as_str(), value);
                }
            }
            let logits = logits.with_context(|| {
                format!(
                    "chained proposer '{}' produced no '{}' output",
                    plan.proposer, plan.logits_output
                )
            })?;
            let drafted = i64::from(self.host_copy_of(&logits)?.argmax_last_row()?);

            if let Some(value) = next_carry {
                carry = Some(
                    last_position_rows(&self.host_copy_of(&value)?).with_context(|| {
                        format!(
                            "chained proposer '{}' folded carry output '{}' is not a float32 \
                             per-position hidden state",
                            plan.proposer,
                            plan.folded_carry_output.unwrap_or_default()
                        )
                    })?,
                );
            } else if plan.folded_carry_output.is_some() {
                anyhow::bail!(
                    "chained proposer '{}' declares folded_carry_output '{}' but produced none",
                    plan.proposer,
                    plan.folded_carry_output.unwrap_or_default()
                );
            }
            for binding in plan.recurrent {
                let value = next_recurrent.remove(binding.input.as_str()).with_context(|| {
                    format!(
                        "chained proposer '{}' declares recurrence output '{}' but produced none",
                        plan.proposer, binding.output
                    )
                })?;
                recurrent_state.insert(binding.input.as_str(), value);
            }

            if step == 0 {
                // carry_0 is the target's hidden state for the last *context*
                // token, so this step reproduces the target's own next-token
                // prediction — which the target already computed and handed over
                // as `guaranteed_token`. Its draft is therefore redundant and
                // discarded; the step is still run because the chain's carry
                // must advance past the guaranteed token before the first real
                // draft conditions on it. This is a property of the folded-carry
                // contract, not of any particular model.
                last_token = options.guaranteed_token;
            } else {
                tokens.push(drafted);
                last_token = drafted;
            }
        }

        Ok(ChainedProposal {
            tokens,
            proposer_invocations: invocations,
        })
    }

    /// Longest proposal prefix the target confirms.
    ///
    /// `target_tokens[i]` is the target's own token for block position `i`, and
    /// one entry beyond the block is required: verification consumed the whole
    /// block, so the target already produced the token that follows it. That
    /// entry is what makes a fully-accepted block commit `width + 1` tokens and
    /// a rejected block commit the target's correction instead of the draft.
    ///
    /// Position 0 matches by construction — it *is* the target's token — so a
    /// proposal always commits at least one token, and the result equals plain
    /// greedy decoding for a `distribution_preserving` package.
    pub fn accept_chained_proposal(
        &self,
        proposal: &ChainedProposal,
        target_tokens: &[i64],
    ) -> anyhow::Result<ProposalAcceptance> {
        anyhow::ensure!(
            target_tokens.len() == proposal.tokens.len() + 1,
            "verifying a {}-token proposal block needs {} target tokens (one beyond the block), \
             found {}",
            proposal.tokens.len(),
            proposal.tokens.len() + 1,
            target_tokens.len()
        );
        let mut accepted = 0;
        let mut rejected_at = None;
        for (position, proposed) in proposal.tokens.iter().enumerate() {
            if target_tokens[position] == *proposed {
                accepted += 1;
            } else {
                rejected_at = Some(position);
                break;
            }
        }
        anyhow::ensure!(
            accepted >= 1,
            "block position 0 must be the target's own guaranteed token, but the target says \
             {} and the block says {}",
            target_tokens[0],
            proposal.tokens[0]
        );
        let mut committed = proposal.tokens[..accepted].to_vec();
        committed.push(target_tokens[accepted]);
        Ok(ProposalAcceptance {
            accepted,
            rejected_at,
            committed,
        })
    }

    /// Collect the declared `rollback_state` cells out of a completed pass.
    ///
    /// Each cell is resolved through the serving state contract to the target
    /// output the pass wrote it from, so the caller never has to know which port
    /// backs which cell. Values are aliased where the backend allows it, keeping
    /// a device-resident cell on the device instead of forcing a host round-trip
    /// just to be checkpointed.
    pub fn speculative_rollback_state(
        &self,
        run: &PipelineTensors,
    ) -> anyhow::Result<PipelineTensors> {
        let contract = self
            .speculative
            .as_ref()
            .context("package declares no speculative contract")?;
        let mut state = PipelineTensors::new();
        for cell in &contract.rollback_state {
            let port =
                target_state_output(&self.workflow, &contract.target, cell).with_context(|| {
                    format!(
                        "rollback_state cell '{cell}' names no output port on the speculative \
                         target '{}'",
                        contract.target
                    )
                })?;
            let value_name = component_invocation(&self.workflow, &contract.target)
                .and_then(|(_, outputs)| outputs.get(&port).cloned())
                .with_context(|| {
                    format!("the workflow binds no value to target output '{port}'")
                })?;
            let value = run.get(&value_name).with_context(|| {
                format!("rollback_state cell '{cell}' has no value '{value_name}' in this pass")
            })?;
            state.insert(cell.clone(), clone_value(value)?);
        }
        Ok(state)
    }

    /// Truncate every declared `rollback_state` cell to `length` positions.
    ///
    /// The sequence axis comes from the serving state-group the cell belongs to,
    /// so a package whose KV is not `[batch, heads, sequence, head_dim]` rolls
    /// back on its own declared axis. Cells outside `rollback_state` — a folded
    /// carry above all — are untouched by design: they are recomputed from the
    /// committed tokens.
    pub fn rollback_speculative_state(
        &self,
        state: &mut PipelineTensors,
        length: usize,
    ) -> anyhow::Result<()> {
        let contract = self
            .speculative
            .as_ref()
            .context("package declares no speculative contract")?;
        for cell in &contract.rollback_state {
            let axis = self.state_sequence_axis(cell).with_context(|| {
                format!("rollback_state cell '{cell}' declares no sequence axis")
            })?;
            let value = state
                .get(cell)
                .with_context(|| format!("rollback_state cell '{cell}' has no value to undo"))?;
            let host = self.host_copy_of(value)?;
            let truncated = truncate_along_axis(&host, axis, length)
                .with_context(|| format!("failed to roll back state cell '{cell}'"))?;
            state.insert(cell.clone(), truncated);
        }
        Ok(())
    }

    /// Sequence axis a state cell rolls back along, from its serving group.
    fn state_sequence_axis(&self, cell: &str) -> Option<usize> {
        let serving = self.workflow.serving.as_ref()?;
        serving
            .state_service
            .groups
            .values()
            .find(|group| {
                group
                    .ports
                    .values()
                    .any(|component| component.contains_key(cell))
            })
            .and_then(|group| group.sequence_axis)
    }

    fn workflow_session_state_value(&self, cell: &str) -> Option<Value> {
        self.workflow_session_state
            .borrow()
            .iter()
            .find(|((_, name), _)| name == cell)
            .and_then(|(_, value)| clone_value(value).ok())
    }

    fn proposer_output_bindings(&self, plan: &ChainedPlan<'_>) -> BTreeMap<String, String> {
        let mut outputs = plan.proposer_outputs.clone();
        // The chain reads its own recurrence and carry even when the single-pass
        // step list ignores them.
        outputs
            .entry(plan.logits_output.to_string())
            .or_insert_with(|| format!("{}.{}", plan.proposer, plan.logits_output));
        if let Some(port) = plan.folded_carry_output {
            outputs
                .entry(port.to_string())
                .or_insert_with(|| format!("{}.{port}", plan.proposer));
        }
        for binding in plan.recurrent {
            outputs
                .entry(binding.output.clone())
                .or_insert_with(|| format!("{}.{}", plan.proposer, binding.output));
        }
        outputs
    }

    /// Read the declared `[vocab, hidden]` embedding table out of a component's
    /// ONNX artifact.
    ///
    /// The contract names both the component and the initializer, so this never
    /// guesses which weight is the embedding: a missing or non-2D initializer is
    /// an error naming the contract field that pointed at it.
    pub fn embedding_table(&self, component: &str, table: &str) -> anyhow::Result<EmbeddingTable> {
        let path = self
            .models
            .directory
            .model_paths
            .get(component)
            .with_context(|| {
                format!("token_embedding names component '{component}', which has no ONNX artifact")
            })?
            .to_path_buf();
        let (graph, weights) = onnx_runtime_loader::load_model_with_weights(&path)
            .with_context(|| format!("failed to read '{component}' to gather '{table}'"))?;
        let (value_id, info) = graph
            .values
            .iter()
            .find(|(_, value)| value.name.as_deref() == Some(table))
            .with_context(|| {
                format!(
                    "token_embedding.table '{table}' is not a value in component '{component}' \
                     ({})",
                    path.display()
                )
            })?;
        let weight = graph.initializers.get(&value_id).with_context(|| {
            format!("token_embedding.table '{table}' in '{component}' is not an initializer")
        })?;
        let shape = onnx_runtime_ir::as_static_shape(&info.shape).with_context(|| {
            format!("token_embedding.table '{table}' has no static [vocab, hidden] shape")
        })?;
        anyhow::ensure!(
            shape.len() == 2,
            "token_embedding.table '{table}' must be a [vocab, hidden] matrix, found rank {}",
            shape.len()
        );
        let (vocab, hidden) = (shape[0], shape[1]);
        anyhow::ensure!(
            info.dtype == onnx_runtime_ir::DataType::Float32,
            "token_embedding.table '{table}' must be float32, found {:?}",
            info.dtype
        );
        let bytes = weights
            .bytes(weight)
            .with_context(|| format!("token_embedding.table '{table}' has no weight bytes"))?;
        anyhow::ensure!(
            bytes.len() == vocab * hidden * std::mem::size_of::<f32>(),
            "token_embedding.table '{table}' holds {} bytes for a [{vocab}, {hidden}] f32 matrix",
            bytes.len()
        );
        let rows = bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        Ok(EmbeddingTable {
            vocab,
            hidden,
            rows,
        })
    }
}

/// A dense `[vocab, hidden]` row-major embedding table read from a component.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingTable {
    vocab: usize,
    hidden: usize,
    rows: Vec<f32>,
}

impl EmbeddingTable {
    pub fn vocab_size(&self) -> usize {
        self.vocab
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// The embedding row for `token`.
    ///
    /// An out-of-range id is an error rather than a clamp: a proposer that
    /// drafted an id the table cannot embed has left the declared vocabulary,
    /// and silently embedding row 0 would hide that.
    pub fn row(&self, token: i64) -> anyhow::Result<&[f32]> {
        let index = usize::try_from(token)
            .ok()
            .filter(|index| *index < self.vocab)
            .with_context(|| {
                format!(
                    "token id {token} has no embedding row in a [{}, {}] table",
                    self.vocab, self.hidden
                )
            })?;
        Ok(&self.rows[index * self.hidden..(index + 1) * self.hidden])
    }
}

impl<'a> ChainedPlan<'a> {
    fn resolve(
        contract: &'a SpeculativeContract,
        workflow: &'a WorkflowSpec,
    ) -> anyhow::Result<Self> {
        let SpeculativeProposalExecution::Chained {
            token_embedding_input,
            logits_output,
            recurrent,
            folded_carry_output,
            folded_carry_seed,
            token_embedding,
        } = &contract.proposal_execution
        else {
            anyhow::bail!(
                "speculative.proposal_execution is not `chained`; block proposers emit their \
                 whole block in one invocation and need no proposal chain"
            );
        };
        anyhow::ensure!(
            !recurrent.is_empty() || folded_carry_output.is_some(),
            "a chained proposer must declare at least one of `recurrent` or \
             `folded_carry_output`"
        );
        let (proposer_bindings, proposer_outputs) =
            component_invocation(workflow, &contract.proposer).with_context(|| {
                format!(
                    "speculative proposer '{}' is never invoked by the workflow, so its ports \
                     cannot be bound",
                    contract.proposer
                )
            })?;
        let folded_carry_seed_value = match folded_carry_seed {
            Some(seed) => {
                anyhow::ensure!(
                    seed.component == contract.target,
                    "folded_carry_seed.component '{}' must be the speculative target '{}'",
                    seed.component,
                    contract.target
                );
                let (_, target_outputs) = component_invocation(workflow, &contract.target)
                    .with_context(|| {
                        format!(
                            "speculative target '{}' is never invoked by the workflow",
                            contract.target
                        )
                    })?;
                Some(
                    target_outputs
                        .get(&seed.output)
                        .with_context(|| {
                            format!(
                                "folded_carry_seed names target output '{}', which the workflow \
                                 does not bind to a value",
                                seed.output
                            )
                        })?
                        .clone(),
                )
            }
            None => None,
        };
        if folded_carry_output.is_some() {
            anyhow::ensure!(
                folded_carry_seed_value.is_some(),
                "a folded-carry proposer must declare `folded_carry_seed`"
            );
            anyhow::ensure!(
                token_embedding.is_some(),
                "a folded-carry proposer must declare `token_embedding`"
            );
            let destination = contract
                .port_bindings
                .get("target_hidden_context")
                .map(String::as_str);
            anyhow::ensure!(
                destination == Some(token_embedding_input.as_str()),
                "a folded carry lands in the fused `token_embedding_input` ('{token_embedding_input}'), \
                 but port_bindings.target_hidden_context names {destination:?}"
            );
        }
        Ok(Self {
            proposer: &contract.proposer,
            token_embedding_input,
            logits_output,
            recurrent,
            folded_carry_output: folded_carry_output.as_deref(),
            folded_carry_seed_value,
            token_embedding: token_embedding
                .as_ref()
                .map(|source| (source.component.as_str(), source.table.as_str())),
            proposer_bindings,
            proposer_outputs,
        })
    }
}

/// Values an interpreter-level speculative construct consumes after a pass.
///
/// Island fusion would otherwise elide them: nothing in the step list reads the
/// target's folded-carry seed or the proposer's own bindings a second time. This
/// is the set that keeps them live, computed from the declared contract so a
/// package without a chained proposer contributes nothing.
pub(super) fn externally_used_values(
    contract: Option<&SpeculativeContract>,
    workflow: &WorkflowSpec,
) -> std::collections::HashSet<String> {
    let mut live = std::collections::HashSet::new();
    let Some(contract) = contract else {
        return live;
    };
    let SpeculativeProposalExecution::Chained {
        folded_carry_seed, ..
    } = &contract.proposal_execution
    else {
        return live;
    };
    if let Some((inputs, outputs)) = component_invocation(workflow, &contract.proposer) {
        live.extend(inputs.into_values());
        live.extend(outputs.into_values());
    }
    if let Some(seed) = folded_carry_seed
        && let Some((_, outputs)) = component_invocation(workflow, &seed.component)
        && let Some(value) = outputs.get(&seed.output)
    {
        live.insert(value.clone());
    }
    // A rejected proposal restores the declared rollback cells, so the values a
    // pass leaves in them must survive fusion too.
    if let Some((_, outputs)) = component_invocation(workflow, &contract.target) {
        for cell in &contract.rollback_state {
            if let Some(port) = target_state_output(workflow, &contract.target, cell)
                && let Some(value) = outputs.get(&port)
            {
                live.insert(value.clone());
            }
        }
    }
    live
}

/// The component output port a serving state group writes `cell` from.
fn target_state_output(workflow: &WorkflowSpec, component: &str, cell: &str) -> Option<String> {
    let service = &workflow.serving.as_ref()?.state_service;
    service.groups.values().find_map(|group| {
        group
            .ports
            .get(component)
            .and_then(|ports| ports.get(cell))
            .and_then(|port| port.output.clone())
    })
}

/// One component's `invoke` bindings: input ports then output ports, each
/// mapping a declared port name to the SSA value the workflow bound it to.
type ComponentBindings = (BTreeMap<String, String>, BTreeMap<String, String>);

/// The workflow's own `invoke` bindings for a component, port -> SSA value.
fn component_invocation(workflow: &WorkflowSpec, component: &str) -> Option<ComponentBindings> {
    fn walk(steps: &[WorkflowStep], component: &str, found: &mut Option<ComponentBindings>) {
        for step in steps {
            match step {
                WorkflowStep::Invoke {
                    component: name,
                    inputs,
                    outputs,
                } if name == component => {
                    if found.is_none() {
                        *found = Some((inputs.clone(), outputs.clone()));
                    }
                }
                WorkflowStep::Sequence { steps } => walk(steps, component, found),
                WorkflowStep::Loop { setup, steps, .. } => {
                    walk(setup, component, found);
                    walk(steps, component, found);
                }
                WorkflowStep::Branch { cases, default, .. } => {
                    for case in cases.values() {
                        walk(std::slice::from_ref(case), component, found);
                    }
                    if let Some(default) = default {
                        walk(std::slice::from_ref(default), component, found);
                    }
                }
                _ => {}
            }
        }
    }
    let mut found = None;
    walk(&workflow.steps, component, &mut found);
    found
}

/// The shape symbol on the fused proposer input's position axis, if it has one.
fn position_symbol(
    declaration: &onnx_genai_metadata::WorkflowComponent,
    port: &str,
) -> Option<String> {
    let shape = declaration.ports.inputs.get(port)?.shape.as_ref()?;
    if shape.len() < 3 {
        return None;
    }
    match &shape[shape.len() - 2] {
        onnx_genai_metadata::TensorDimension::Symbol(symbol) => Some(symbol.clone()),
        onnx_genai_metadata::TensorDimension::Fixed(_) => None,
    }
}

/// The axis of `port` carrying `symbol`, if the port declares it.
fn symbol_axis(
    declaration: &onnx_genai_metadata::WorkflowComponent,
    port: &str,
    symbol: &str,
) -> Option<usize> {
    let shape = declaration.ports.inputs.get(port)?.shape.as_ref()?;
    shape.iter().position(|dimension| {
        matches!(dimension, onnx_genai_metadata::TensorDimension::Symbol(name) if name == symbol)
    })
}

/// Keep only the final index of `axis`, preserving rank.
fn last_position_along(value: &Value, axis: usize) -> anyhow::Result<Value> {
    let shape = value.shape().to_vec();
    anyhow::ensure!(
        axis < shape.len(),
        "position axis {axis} is out of range for a rank-{} tensor",
        shape.len()
    );
    let extent = usize::try_from(shape[axis]).context("negative tensor extent")?;
    anyhow::ensure!(extent > 0, "cannot take the last position of an empty axis");
    if extent == 1 {
        return clone_value(value);
    }
    let outer = shape[..axis]
        .iter()
        .try_fold(1usize, |total, dimension| {
            usize::try_from(*dimension).ok().map(|d| total * d)
        })
        .context("negative tensor extent")?;
    let inner = shape[axis + 1..]
        .iter()
        .try_fold(1usize, |total, dimension| {
            usize::try_from(*dimension).ok().map(|d| total * d)
        })
        .context("negative tensor extent")?;
    let element = value.dtype().size_of();
    let bytes = value.as_raw_bytes()?;
    let block = extent * inner * element;
    let keep = inner * element;
    let mut narrowed = Vec::with_capacity(outer * keep);
    for index in 0..outer {
        let start = index * block + (extent - 1) * keep;
        narrowed.extend_from_slice(&bytes[start..start + keep]);
    }
    let mut new_shape = shape;
    new_shape[axis] = 1;
    Value::from_raw_bytes(narrowed, &new_shape, value.dtype())
        .map_err(|error| anyhow::anyhow!("failed to narrow a proposer input: {error}"))
}

/// The shape one proposal step binds for the fused proposer input.
///
/// A chained step advances exactly one position, so the position axis — the one
/// before the feature axis — collapses to 1. Rank-1 and rank-2 fused inputs have
/// no separate position axis and are passed through unchanged.
fn single_position_shape(shape: &[i64]) -> Option<Vec<i64>> {
    if shape.is_empty() {
        return None;
    }
    let mut resolved = shape.to_vec();
    if resolved.len() >= 3 {
        let position = resolved.len() - 2;
        resolved[position] = 1;
    }
    Some(resolved)
}

/// The most recent position of a `[.., positions, features]` float32 value, one
/// row per batch entry.
///
/// A folded carry is a per-position hidden state: the seed covers every prompt
/// position, and a proposer step covers one. Both reduce the same way — take the
/// final position — so the loop never has to know which of the two it holds.
fn last_position_rows(value: &Value) -> anyhow::Result<Vec<Vec<f32>>> {
    let shape = value.shape();
    let features = *shape.last().context("carry value has rank 0")?;
    let features = usize::try_from(features).context("carry value has a negative width")?;
    anyhow::ensure!(features > 0, "carry value has a zero-width feature axis");
    let data = value.to_vec_f32().map_err(|error| {
        anyhow::anyhow!("last_position_rows: carry value must be float32: {error}")
    })?;
    anyhow::ensure!(
        data.len() % features == 0,
        "carry value holds {} elements, which is not a multiple of its {features}-wide feature axis",
        data.len()
    );
    let positions = if shape.len() >= 3 {
        usize::try_from(shape[shape.len() - 2]).context("carry value has a negative extent")?
    } else {
        1
    };
    anyhow::ensure!(positions > 0, "carry value holds no positions");
    let rows = data.len() / features / positions;
    Ok((0..rows)
        .map(|row| {
            let start = (row * positions + positions - 1) * features;
            data[start..start + features].to_vec()
        })
        .collect())
}

/// Build `concat(embed(last_token), carry)` for one proposer step.
///
/// The fused input's declared width fixes the split: a folded carry occupies the
/// trailing segment and the gathered embedding the leading one. A proposer with
/// a recurrence but no folded carry has no trailing segment, so the whole fused
/// input is the embedding.
fn build_fused_input(
    shape: &[i64],
    width: usize,
    rows: usize,
    embedding: Option<&EmbeddingTable>,
    token: i64,
    carry: Option<&[Vec<f32>]>,
) -> anyhow::Result<Value> {
    let embedded = match embedding {
        Some(table) => Some(table.row(token)?),
        None => None,
    };
    let leading = embedded.map(<[f32]>::len).unwrap_or(0);
    if let Some(carry) = carry {
        anyhow::ensure!(
            carry.len() == rows,
            "folded carry covers {} batch rows, but the fused proposer input has {rows}",
            carry.len()
        );
        let carry_width = carry[0].len();
        anyhow::ensure!(
            leading + carry_width == width,
            "fused proposer input is {width} wide, but embed({leading}) + carry({carry_width}) \
             is {}",
            leading + carry_width
        );
    } else {
        anyhow::ensure!(
            leading == width,
            "fused proposer input is {width} wide, but the gathered embedding is {leading}"
        );
    }
    let mut data = vec![0.0f32; rows * width];
    for row in 0..rows {
        let base = row * width;
        if let Some(embedded) = embedded {
            data[base..base + leading].copy_from_slice(embedded);
        }
        if let Some(carry) = carry {
            data[base + leading..base + width].copy_from_slice(&carry[row]);
        }
    }
    Value::from_vec_f32(data, shape)
        .map_err(|error| anyhow::anyhow!("failed to materialize the fused proposer input: {error}"))
}

/// Copy a value's leading `length` positions along `axis`.
fn truncate_along_axis(value: &Value, axis: usize, length: usize) -> anyhow::Result<Value> {
    let shape = value.shape().to_vec();
    anyhow::ensure!(
        axis < shape.len(),
        "sequence axis {axis} is out of range for a rank-{} tensor",
        shape.len()
    );
    let extent = usize::try_from(shape[axis]).context("negative tensor extent")?;
    anyhow::ensure!(
        length <= extent,
        "cannot roll back to {length} positions: the tensor holds {extent}"
    );
    if length == extent {
        return clone_value(value);
    }
    let outer = shape[..axis]
        .iter()
        .try_fold(1usize, |total, dimension| {
            usize::try_from(*dimension).ok().map(|d| total * d)
        })
        .context("negative tensor extent")?;
    let inner = shape[axis + 1..]
        .iter()
        .try_fold(1usize, |total, dimension| {
            usize::try_from(*dimension).ok().map(|d| total * d)
        })
        .context("negative tensor extent")?;
    let element = value.dtype().size_of();
    let bytes = value.as_raw_bytes()?;
    let mut truncated = Vec::with_capacity(outer * length * inner * element);
    let block = extent * inner * element;
    let keep = length * inner * element;
    for index in 0..outer {
        let start = index * block;
        truncated.extend_from_slice(&bytes[start..start + keep]);
    }
    let mut new_shape = shape;
    new_shape[axis] = i64::try_from(length).context("rollback length exceeds i64")?;
    Value::from_raw_bytes(truncated, &new_shape, value.dtype())
        .map_err(|error| anyhow::anyhow!("failed to materialize the rolled-back state: {error}"))
}

/// Clone a workflow value without changing where it lives.
///
/// A value backed by an allocation it owns — a native session's device buffer,
/// above all — is aliased in O(1) so it stays device-resident. Only an
/// owned-host-backing value is deep-copied.
fn clone_value(value: &Value) -> anyhow::Result<Value> {
    if let Some(aliased) = value.try_alias_clone() {
        return aliased
            .map_err(|error| anyhow::anyhow!("failed to alias a workflow value: {error}"));
    }
    value
        .clone_owned()
        .map_err(|error| anyhow::anyhow!("failed to clone a workflow value: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORKFLOW: &str = r#"
manifest:
  capabilities: [workflow_ssa, typed_emit]
inputs: {}
outputs: {}
components:
  target:
    implementation: {kind: onnx, artifact: target/model.onnx}
    ports:
      inputs: {}
      outputs:
        logits: {dtype: float32, rank: 3, shape: [batch, sequence, vocab]}
        hidden_states.0: {dtype: float32, rank: 3, shape: [batch, sequence, hidden]}
  assistant:
    implementation: {kind: onnx, artifact: assistant/model.onnx}
    ports:
      inputs:
        inputs_embeds: {dtype: float32, rank: 3, shape: [batch, sequence, fused]}
      outputs:
        logits: {dtype: float32, rank: 3, shape: [batch, sequence, vocab]}
steps:
- kind: invoke
  component: assistant
  inputs: {inputs_embeds: request.inputs_embeds}
  outputs: {logits: draft.logits}
- kind: invoke
  component: target
  inputs: {}
  outputs: {logits: target.logits, hidden_states.0: target.hidden_states.0}
"#;

    fn workflow() -> WorkflowSpec {
        serde_yaml::from_str(WORKFLOW).expect("workflow spec")
    }

    fn contract(execution: SpeculativeProposalExecution) -> SpeculativeContract {
        SpeculativeContract {
            proposer: "assistant".to_string(),
            target: "target".to_string(),
            proposal_execution: execution,
            port_bindings: Default::default(),
            shared_state: Default::default(),
            shared_weights: Default::default(),
            vocabulary: onnx_genai_metadata::SpeculativeVocabulary::Identical,
            max_proposal_width: 4,
            distribution_preserving: true,
            rollback_state: Default::default(),
        }
    }

    /// Keeping speculative values live is what stops island fusion from
    /// swallowing a proposal's inputs. It must be scoped to packages that
    /// actually declare a chained proposer: a package without one, or with a
    /// `block` proposer that needs no chain, must contribute nothing, or every
    /// workflow in the repo would silently lose fusion.
    #[test]
    fn only_a_chained_contract_keeps_values_live() {
        assert!(externally_used_values(None, &workflow()).is_empty());
        assert!(
            externally_used_values(
                Some(&contract(SpeculativeProposalExecution::Block)),
                &workflow()
            )
            .is_empty()
        );
    }

    /// A chained proposer's own bindings and its folded-carry seed are the
    /// values the driver reads after the pass, so those exactly are the ones
    /// fusion must not elide.
    #[test]
    fn a_chained_contract_keeps_its_bindings_and_carry_seed_live() {
        let contract = contract(SpeculativeProposalExecution::Chained {
            token_embedding_input: "inputs_embeds".to_string(),
            logits_output: "logits".to_string(),
            recurrent: Vec::new(),
            folded_carry_output: Some("projected_state".to_string()),
            folded_carry_seed: Some(onnx_genai_metadata::SpeculativeValueRef {
                component: "target".to_string(),
                output: "hidden_states.0".to_string(),
            }),
            token_embedding: Some(onnx_genai_metadata::TokenEmbeddingSource {
                component: "target".to_string(),
                table: "hidden_table".to_string(),
            }),
        });
        let live = externally_used_values(Some(&contract), &workflow());
        assert!(live.contains("request.inputs_embeds"), "{live:?}");
        assert!(live.contains("draft.logits"), "{live:?}");
        assert!(live.contains("target.hidden_states.0"), "{live:?}");
        // The target's own logits are emitted by the workflow, not read by the
        // proposal chain, so they are not force-kept here.
        assert!(!live.contains("target.logits"), "{live:?}");
    }

    /// A folded carry re-enters through the fused input and owns no state cell,
    /// so a package that declares one without saying where carry_0 comes from,
    /// or without naming the embedding table, is rejected rather than
    /// convention-inferred.
    #[test]
    fn a_folded_carry_must_declare_its_seed_and_embedding_table() {
        let mut incomplete = contract(SpeculativeProposalExecution::Chained {
            token_embedding_input: "inputs_embeds".to_string(),
            logits_output: "logits".to_string(),
            recurrent: Vec::new(),
            folded_carry_output: Some("projected_state".to_string()),
            folded_carry_seed: None,
            token_embedding: None,
        });
        let error = ChainedPlan::resolve(&incomplete, &workflow())
            .expect_err("a folded carry without a seed must not resolve");
        assert!(
            format!("{error:#}").contains("folded_carry_seed"),
            "{error:#}"
        );

        // With a seed but no embedding table it is still under-declared.
        if let SpeculativeProposalExecution::Chained {
            folded_carry_seed, ..
        } = &mut incomplete.proposal_execution
        {
            *folded_carry_seed = Some(onnx_genai_metadata::SpeculativeValueRef {
                component: "target".to_string(),
                output: "hidden_states.0".to_string(),
            });
        }
        let error = ChainedPlan::resolve(&incomplete, &workflow())
            .expect_err("a folded carry without an embedding table must not resolve");
        assert!(
            format!("{error:#}").contains("token_embedding"),
            "{error:#}"
        );
    }

    /// A chain that declares neither a recurrence nor a folded carry has nothing
    /// to thread, so repeating the proposer would just re-emit one distribution.
    #[test]
    fn a_chained_proposer_must_declare_something_to_thread() {
        let empty = contract(SpeculativeProposalExecution::Chained {
            token_embedding_input: "inputs_embeds".to_string(),
            logits_output: "logits".to_string(),
            recurrent: Vec::new(),
            folded_carry_output: None,
            folded_carry_seed: None,
            token_embedding: None,
        });
        let error = ChainedPlan::resolve(&empty, &workflow())
            .expect_err("a chain with nothing to thread must not resolve");
        assert!(format!("{error:#}").contains("recurrent"), "{error:#}");
    }
}
