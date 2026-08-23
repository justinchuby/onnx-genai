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
            // Narrow first, and where the value already is. Copying the wide
            // tensor down to keep one position of it was the largest transfer
            // in this loop, and every byte of it was discarded on the next
            // line.
            let narrowed = match position_symbol
                .as_deref()
                .and_then(|symbol| symbol_axis(declaration, port, symbol))
            {
                Some(axis) => super::device_ops::tensor_ops_for(value)?
                    .last_along_axis(value, axis)
                    .with_context(|| {
                        format!(
                            "failed to narrow chained proposer '{}' input '{port}' to one \
                             position",
                            plan.proposer
                        )
                    })?,
                // A port with no position symbol binds exactly what the
                // workflow's own invocation bound it, so it is passed through
                // as it stands. Aliasing keeps a device-resident borrowed KV
                // where it is; materializing it here would copy the largest
                // tensor in the loop to learn nothing about it.
                None => clone_value(value)?,
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
        // at its most recent position — and left where that output already is.
        // The chain's residency follows from it: the carry is one half of every
        // fused input the loop builds, so whatever device produced it is the
        // device the fused input is assembled on.
        let carry_seed = match &plan.folded_carry_seed_value {
            Some(value_name) => Some(run.get(value_name).with_context(|| {
                format!(
                    "folded_carry_seed names target output value '{value_name}', which the \
                     verification pass did not produce"
                )
            })?),
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
        // Where the chain does its tensor algebra: where the proposer executes.
        // That is configured, not discovered, so the fused input is built in the
        // proposer's own memory from the first step rather than assembled on the
        // host and uploaded once per draft token.
        let residency = self.component_execution_residency()?;
        let ops = super::device_ops::tensor_ops_for_residency(residency)?;

        let mut carry = match carry_seed {
            Some(seed) => {
                // carry_0 comes from the target's verification pass, which may
                // have left it wherever that pass emitted it. Narrow it first —
                // one position out of the whole prompt — and adopt only that,
                // once per proposal. Every carry after this one is produced by
                // the proposer itself and is already here.
                let narrowed =
                    last_position_of_carry(super::device_ops::tensor_ops_for(seed)?.as_ref(), seed)
                        .context("folded_carry_seed must name a per-position hidden output")?;
                Some(self.adopt_into(ops.as_ref(), &narrowed).with_context(|| {
                    format!("failed to bring the folded carry seed onto {residency}")
                })?)
            }
            None => None,
        };

        let embedding = match plan.token_embedding {
            Some((component, table)) => {
                Some(self.embedding_table_resident(component, table, residency)?)
            }
            None => None,
        };

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

        // The split between the two halves is loop-invariant — the embedding
        // table's width and the carry's are both fixed by the package — so it
        // is checked once here rather than rediscovered every step.
        let leading = embedding
            .as_ref()
            .map(|table| table.hidden_size())
            .unwrap_or(0);
        match &carry {
            Some(carry) => {
                let (carry_rows, carry_width) = trailing_rows_and_width(carry, "folded carry")?;
                anyhow::ensure!(
                    carry_rows == batch_rows,
                    "folded carry covers {carry_rows} batch rows, but the fused proposer input \
                     has {batch_rows}"
                );
                anyhow::ensure!(
                    leading + carry_width == fused_width,
                    "fused proposer input is {fused_width} wide, but embed({leading}) + \
                     carry({carry_width}) is {}",
                    leading + carry_width
                );
            }
            None => anyhow::ensure!(
                leading == fused_width,
                "fused proposer input is {fused_width} wide, but the gathered embedding is \
                 {leading}"
            ),
        }

        // One buffer for the whole chain, in the residency the halves live in.
        // Rebuilding it per step would allocate — and on a device, upload — the
        // concatenation the scatters below write in place.
        let fused = ops
            .zeros(&fused_shape, DataType::Float32)
            .context("failed to allocate the fused proposer input")?;

        let outputs = self.proposer_output_bindings(&plan);
        // Symbols the proposer's *outputs* declare that none of its inputs
        // carry — a vocabulary, above all. The workflow already invoked this
        // component in this very run, so its own bound output values prove the
        // extents; without them the run cannot size a device buffer for those
        // outputs and hands each one back through host memory.
        let output_symbols = proposer_output_symbols(declaration, &plan, run);
        let mut tokens = Vec::with_capacity(options.width);
        tokens.push(options.guaranteed_token);
        let mut last_token = options.seed_token;
        let mut invocations = 0usize;

        for step in 0..options.width {
            if let Some(table) = embedding.as_ref() {
                // The same token embeds into every batch row, so the gather is
                // asked for that row once per row rather than broadcast — the
                // scatter then needs no broadcasting rule to get wrong.
                let embedded = ops
                    .gather_rows(table.value(), &vec![last_token; batch_rows])
                    .with_context(|| {
                        format!("failed to gather the embedding of token {last_token}")
                    })?;
                ops.scatter_into_last_axis(&fused, 0, &embedded)
                    .with_context(|| {
                        format!("failed to place the embedding half of step {step}'s fused input")
                    })?;
            }
            if let Some(carry) = carry.as_ref() {
                ops.scatter_into_last_axis(&fused, leading, carry)
                    .with_context(|| {
                        format!("failed to place the carry half of step {step}'s fused input")
                    })?;
            }
            let mut bound: Vec<(&str, &Value)> = fixed
                .iter()
                .map(|(port, value)| (port.as_str(), value))
                .collect();
            bound.push((plan.token_embedding_input, &fused));
            for (port, value) in &recurrent_state {
                bound.push((port, value));
            }

            let produced = self
                .invoke_component_values(plan.proposer, &bound, &outputs, &output_symbols)
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
            // Four bytes back, not a vocabulary: the winning id is all this
            // step needs, and a device that produced the row can find it.
            let drafted = i64::from(self.argmax_token(&logits)?);

            if let Some(value) = next_carry {
                // Narrow where the proposer left it, then adopt — the same two
                // steps the seed takes. Deriving the operations from the value
                // rather than from the chain's residency is what keeps this
                // correct when a backend publishes an output somewhere other
                // than where the chain assembles; on the common path the
                // residencies already agree and the adoption is an O(1) alias.
                let narrowed = last_position_of_carry(
                    super::device_ops::tensor_ops_for(&value)?.as_ref(),
                    &value,
                )
                .with_context(|| {
                    format!(
                        "chained proposer '{}' folded carry output '{}' is not a per-position \
                         hidden state",
                        plan.proposer,
                        plan.folded_carry_output.unwrap_or_default()
                    )
                })?;
                carry = Some(self.adopt_into(ops.as_ref(), &narrowed).with_context(|| {
                    format!("failed to bring the folded carry onto {residency}")
                })?);
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
            // Narrowing happens where the cell already is. A rejection used to
            // copy the entire KV cache to the host to drop a few positions and
            // upload it again on the next bind — the single largest transfer in
            // the workflow, paid per rejection.
            let truncated = super::device_ops::tensor_ops_for(value)?
                .truncate_axis(value, axis, length)
                .with_context(|| format!("failed to roll back state cell '{cell}'"))?;
            state.insert(cell.clone(), truncated);
        }
        Ok(())
    }

    /// Bring `value` into `ops`' residency, counting anything it brings back.
    ///
    /// Adoption is the seam's one sanctioned crossing, so this is the one place
    /// besides an argmax read-back where a device byte can reach the host — and
    /// therefore the one place besides [`Self::argmax_token`] that has to say
    /// so. A value already in the right residency is aliased and counts
    /// nothing.
    fn adopt_into(
        &self,
        ops: &dyn super::device_ops::ResidentTensorOps,
        value: &Value,
    ) -> anyhow::Result<Value> {
        let source = super::device_ops::residency_of(value)?;
        if source != ops.residency() && ops.residency() == super::device_ops::Residency::Host {
            let bytes = (value.numel() * value.dtype().size_of()) as u64;
            self.device_readback_bytes
                .set(self.device_readback_bytes.get().saturating_add(bytes));
        }
        ops.adopt(value)
    }

    /// The winning token id of a logits row, read where the row already is.
    ///
    /// Four bytes per row, and on a device-resident chain it is the only
    /// per-step transfer there is. Counting it here — together with
    /// [`Self::adopt_into`], the seam's one other crossing, and the
    /// interpreter's own materialization, which counts itself in
    /// `host_staging_count` — accounts for every byte a proposal brings back.
    fn argmax_token(&self, logits: &Value) -> anyhow::Result<u32> {
        let ops = super::device_ops::tensor_ops_for(logits)?;
        let ids = ops.argmax_rows(logits, 1)?;
        let id = *ids.first().context("argmax returned no row")?;
        if ops.residency() != super::device_ops::Residency::Host {
            let bytes = (ids.len() * std::mem::size_of::<u32>()) as u64;
            self.device_readback_bytes
                .set(self.device_readback_bytes.get().saturating_add(bytes));
        }
        Ok(id)
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
    /// The declared embedding table, resident where the chain will gather it.
    ///
    /// Read from the artifact once and — for a device residency — uploaded
    /// once, for the runtime's life. A loaded package's artifact cannot change
    /// under it, so re-reading a `[vocab, hidden]` initializer per proposal
    /// buys nothing; re-uploading one per *draft token* would cost more than
    /// the proposal it feeds. The cache is keyed by residency as well as by
    /// name, because a host table and a device mirror are different tensors
    /// answering the same question.
    pub fn embedding_table_resident(
        &self,
        component: &str,
        table: &str,
        residency: super::device_ops::Residency,
    ) -> anyhow::Result<std::rc::Rc<EmbeddingTable>> {
        let key = (
            component.to_string(),
            table.to_string(),
            residency.cache_key(),
        );
        if let Some(cached) = self.embedding_tables.borrow().get(&key) {
            return Ok(std::rc::Rc::clone(cached));
        }
        let loaded = std::rc::Rc::new(self.embedding_table(component, table)?.into_residency(
            residency,
        ).with_context(|| {
            format!("failed to make embedding table '{table}' of component '{component}' resident on {residency}")
        })?);
        self.embedding_tables
            .borrow_mut()
            .insert(key, std::rc::Rc::clone(&loaded));
        Ok(loaded)
    }

    /// How many times an embedding table was read out of an artifact.
    ///
    /// The cache above is a performance contract, not an implementation detail:
    /// a multi-round speculative decode that re-read the table would be correct
    /// and unusably slow, which is the class of regression a throughput number
    /// does not attribute. This is what a test holds to one read per table.
    pub fn embedding_table_loads(&self) -> u64 {
        self.embedding_table_loads.get()
    }

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
        self.embedding_table_loads
            .set(self.embedding_table_loads.get().saturating_add(1));
        Ok(EmbeddingTable {
            vocab,
            hidden,
            value: Value::from_vec_f32(
                rows,
                &[
                    i64::try_from(vocab).context("embedding vocabulary exceeds i64")?,
                    i64::try_from(hidden).context("embedding width exceeds i64")?,
                ],
            )
            .map_err(|error| {
                anyhow::anyhow!("failed to materialize embedding table '{table}': {error}")
            })?,
        })
    }
}

/// A dense `[vocab, hidden]` row-major embedding table read from a component,
/// held in one residency.
///
/// The table is a *tensor*, not a host array: the gather that reads it happens
/// wherever the proposal chain runs, and a table that only ever exists on the
/// host forces every gathered row across the bus. Holding it as a `Value` is
/// what lets the same type describe the host copy and the device mirror.
pub struct EmbeddingTable {
    vocab: usize,
    hidden: usize,
    value: Value,
}

impl std::fmt::Debug for EmbeddingTable {
    /// A table's contents are a `[vocab, hidden]` matrix that may not even be
    /// host-readable, so the shape is the whole of what is printable.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingTable")
            .field("vocab", &self.vocab)
            .field("hidden", &self.hidden)
            .field("shape", &self.value.shape())
            .finish()
    }
}

impl EmbeddingTable {
    pub fn vocab_size(&self) -> usize {
        self.vocab
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// The table as a `[vocab, hidden]` tensor, for a residency-preserving
    /// gather.
    pub(crate) fn value(&self) -> &Value {
        &self.value
    }

    /// This table mirrored into `residency`, uploading once if it must.
    ///
    /// Takes the table by value because a mirror *replaces* it: the host copy a
    /// device chain was built from has no further reader, and at a real
    /// vocabulary keeping it alive alongside the mirror would double the
    /// footprint of the largest tensor in the package.
    pub(crate) fn into_residency(
        self,
        residency: super::device_ops::Residency,
    ) -> anyhow::Result<Self> {
        if super::device_ops::residency_of(&self.value)? == residency {
            return Ok(self);
        }
        match residency {
            super::device_ops::Residency::Host => anyhow::bail!(
                "an embedding table already resident on a device is not brought back to the host; \
                 gather it where it is, or run this package on the host backend"
            ),
            #[cfg(feature = "ort-cuda")]
            super::device_ops::Residency::Cuda(device) => {
                let mirror = Value::empty_cuda(self.value.shape(), self.value.dtype(), device)
                    .map_err(|error| {
                        anyhow::anyhow!(
                            "failed to allocate a [{}, {}] embedding table on CUDA device \
                                 {device}: {error}",
                            self.vocab,
                            self.hidden
                        )
                    })?;
                mirror
                    .copy_from_cuda(&self.value, device)
                    .map_err(|error| {
                        anyhow::anyhow!(
                            "failed to upload a [{}, {}] embedding table to CUDA device {device}: \
                         {error}",
                            self.vocab,
                            self.hidden
                        )
                    })?;
                Ok(Self {
                    vocab: self.vocab,
                    hidden: self.hidden,
                    value: mirror,
                })
            }
            #[cfg(not(feature = "ort-cuda"))]
            super::device_ops::Residency::Cuda(device) => anyhow::bail!(
                "the proposal chain runs on CUDA device {device}, but this build has no device \
                 tensor operations to hold an embedding table there. Rebuild with the `ort-cuda` \
                 (or `native-cuda`) feature, or run this package on the host backend."
            ),
        }
    }

    /// The embedding row for `token`.
    ///
    /// An out-of-range id is an error rather than a clamp: a proposer that
    /// drafted an id the table cannot embed has left the declared vocabulary,
    /// and silently embedding row 0 would hide that. A device-resident table
    /// has no host rows to borrow and says so.
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
        let rows = self.value.as_slice_f32().map_err(|error| {
            anyhow::anyhow!("this embedding table's rows cannot be read from the host: {error}")
        })?;
        Ok(&rows[index * self.hidden..(index + 1) * self.hidden])
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

/// Shape symbols a proposer's outputs declare that none of its inputs carry.
///
/// A chain step binds one position of each input, so every symbol an input
/// declares is re-bound per step and must never be hinted. What is left is the
/// symbols only the *outputs* mention — a vocabulary, a projected hidden width
/// — whose extents the workflow's own invocation of this same component already
/// proved: the SSA values it bound those outputs to are in this run, with
/// concrete shapes.
///
/// Without those extents an output has no resolvable shape, so no device buffer
/// can be sized for it and the run returns it through host memory — a download
/// of the full logits row, per draft token, which is the transfer the whole
/// chain exists to avoid.
///
/// A symbol two outputs disagree about is dropped rather than guessed: the
/// consequence of omitting a hint is the host fallback that was there before,
/// while the consequence of a wrong one would be a wrongly sized buffer.
fn proposer_output_symbols(
    declaration: &onnx_genai_metadata::WorkflowComponent,
    plan: &ChainedPlan<'_>,
    run: &PipelineTensors,
) -> std::collections::HashMap<String, i64> {
    let input_symbols = declaration
        .ports
        .inputs
        .values()
        .filter_map(|contract| contract.shape.as_ref())
        .flatten()
        .filter_map(|dimension| match dimension {
            onnx_genai_metadata::TensorDimension::Symbol(symbol) => Some(symbol.as_str()),
            onnx_genai_metadata::TensorDimension::Fixed(_) => None,
        })
        .collect::<std::collections::HashSet<_>>();

    let mut hints: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    let mut conflicted: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (port, value_name) in &plan.proposer_outputs {
        let Some(contract) = declaration.ports.outputs.get(port) else {
            continue;
        };
        let Some(shape) = contract.shape.as_ref() else {
            continue;
        };
        let Some(produced) = run.get(value_name) else {
            continue;
        };
        if produced.shape().len() != shape.len() {
            continue;
        }
        for (dimension, extent) in shape.iter().zip(produced.shape()) {
            let onnx_genai_metadata::TensorDimension::Symbol(symbol) = dimension else {
                continue;
            };
            if input_symbols.contains(symbol.as_str()) || *extent < 0 {
                continue;
            }
            match hints.get(symbol.as_str()) {
                Some(known) if known != extent => {
                    conflicted.insert(symbol.clone());
                }
                _ => {
                    hints.insert(symbol.clone(), *extent);
                }
            }
        }
    }
    for symbol in conflicted {
        hints.remove(&symbol);
    }
    hints
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

/// The most recent position of a `[.., positions, features]` per-position
/// value, left where it already is.
///
/// A folded carry is a per-position hidden state: the seed covers every prompt
/// position, and a proposer step covers one. Both reduce the same way — take the
/// final position — so the loop never has to know which of the two it holds.
///
/// The position axis is the one before the feature axis, which is the same
/// layout [`single_position_shape`] collapses when it builds one step's fused
/// input: a carry and the fused input it lands in cannot disagree about where
/// positions are. A rank-2 or smaller carry has no separate position axis and
/// already holds exactly one position.
fn last_position_of_carry(
    ops: &dyn super::device_ops::ResidentTensorOps,
    value: &Value,
) -> anyhow::Result<Value> {
    let shape = value.shape();
    anyhow::ensure!(
        !shape.is_empty(),
        "a folded carry has rank 0, so it has no feature axis"
    );
    match shape.len() >= 3 {
        true => ops.last_along_axis(value, shape.len() - 2),
        false => clone_value(value),
    }
}

/// The `(rows, width)` of a `[.., width]` value, for the fused-input split.
fn trailing_rows_and_width(value: &Value, role: &str) -> anyhow::Result<(usize, usize)> {
    let shape = value.shape();
    let width = usize::try_from(
        *shape
            .last()
            .with_context(|| format!("the {role} has rank 0, so it has no feature axis"))?,
    )
    .with_context(|| format!("the {role} has a negative width"))?;
    anyhow::ensure!(
        width > 0,
        "the {role} has a zero-width feature axis (shape {shape:?})"
    );
    Ok((value.numel() / width, width))
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

#[cfg(test)]
mod rollback_residency_tests {
    use super::*;
    use crate::pipeline::device_ops::{HostTensorOps, ResidentTensorOps};

    /// A batch-1 seq-major cache truncates as a view, not a copy.
    ///
    /// This is the shape the native backend declares, and it is the case that
    /// used to copy the whole KV cache to the host and back on every rejection.
    #[test]
    fn a_contiguous_prefix_truncates_to_the_kept_prefix() {
        // [batch=1, sequence=4, hidden=2]
        let value = Value::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[1, 4, 2])
            .expect("value");
        let truncated = HostTensorOps
            .truncate_axis(&value, 1, 2)
            .expect("truncation succeeds");
        assert_eq!(truncated.shape(), &[1, 2, 2]);
        assert_eq!(
            truncated.to_vec_f32().expect("host read"),
            vec![0.0, 1.0, 2.0, 3.0],
            "the truncation must be the kept prefix, in order"
        );
    }

    /// A strided truncation keeps the declared elements rather than a prefix of
    /// the buffer.
    ///
    /// `[batch, heads, sequence, head_dim]` keeps a prefix of axis 2, which is
    /// interleaved across heads — not a prefix of the buffer. Handing back a
    /// contiguous view here would silently return the wrong elements, so the
    /// strided case copies within its own residency instead.
    #[test]
    fn a_strided_truncation_keeps_the_declared_elements() {
        // [batch=1, heads=2, sequence=2, head_dim=1]
        let value = Value::from_slice_f32(&[0.0, 1.0, 2.0, 3.0], &[1, 2, 2, 1]).expect("value");
        let truncated = HostTensorOps
            .truncate_axis(&value, 2, 1)
            .expect("a head-major cache still truncates");
        assert_eq!(truncated.shape(), &[1, 2, 1, 1]);
        assert_eq!(
            truncated.to_vec_f32().expect("host read"),
            vec![0.0, 2.0],
            "each head keeps its own first position, not the buffer's first two elements"
        );
    }

    /// An unchanged length is a no-op rollback, and still produces the cell.
    #[test]
    fn an_unchanged_length_keeps_every_element() {
        let value = Value::from_slice_f32(&[0.0, 1.0], &[1, 2]).expect("value");
        let kept = HostTensorOps
            .truncate_axis(&value, 1, 2)
            .expect("an unchanged length must succeed");
        assert_eq!(kept.shape(), &[1, 2]);
        assert_eq!(kept.to_vec_f32().expect("host read"), vec![0.0, 1.0]);
    }

    /// A length beyond the cell is refused, naming both extents, rather than
    /// aliasing past the buffer.
    #[test]
    fn a_length_beyond_the_cell_is_refused() {
        let value = Value::from_slice_f32(&[0.0, 1.0], &[1, 2]).expect("value");
        let Err(error) = HostTensorOps.truncate_axis(&value, 1, 5) else {
            panic!("a length past the end must be refused");
        };
        let message = format!("{error:#}");
        assert!(message.contains('5') && message.contains('2'), "{message}");
    }

    /// Truncation is a window of the one narrowing primitive, so the fast
    /// contiguous path cannot disagree with the general one.
    #[test]
    fn truncation_agrees_with_the_slice_primitive() {
        let value =
            Value::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[1, 3, 2]).expect("value");
        for length in 0..=3 {
            let truncated = HostTensorOps
                .truncate_axis(&value, 1, length)
                .expect("truncation")
                .to_vec_f32()
                .expect("host read");
            let sliced = HostTensorOps
                .slice_axis(&value, 1, 0, length)
                .expect("slice")
                .to_vec_f32()
                .expect("host read");
            assert_eq!(truncated, sliced, "length {length}");
        }
    }

    /// A folded carry is narrowed to its final position wherever it lives, and
    /// a carry that already holds one position is left alone.
    #[test]
    fn a_carry_reduces_to_its_final_position() {
        // [batch=1, positions=3, features=2]
        let seeded =
            Value::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[1, 3, 2]).expect("value");
        let reduced = last_position_of_carry(&HostTensorOps, &seeded).expect("carry narrows");
        assert_eq!(reduced.shape(), &[1, 1, 2]);
        assert_eq!(reduced.to_vec_f32().expect("host read"), vec![4.0, 5.0]);

        // Rank 2 has no separate position axis: it is already one position.
        let flat = Value::from_slice_f32(&[7.0, 8.0], &[1, 2]).expect("value");
        let kept = last_position_of_carry(&HostTensorOps, &flat).expect("carry passes through");
        assert_eq!(kept.shape(), &[1, 2]);
        assert_eq!(kept.to_vec_f32().expect("host read"), vec![7.0, 8.0]);
    }
}
