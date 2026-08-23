//! Drive a *published* workflow package from its own declarations.
//!
//! The hermetic fixtures under `tests/fixtures/onnx_genai_workflows` are small
//! enough that a test can spell their geometry out (`KV_HEADS = 2`,
//! `HEAD_DIM = 8`). A real package cannot be treated that way and must not be:
//! writing head counts or hidden widths into a test is exactly the hardcoded
//! architecture [`RULES.md` §2] forbids, and a test that knows them proves
//! nothing about a package that declares them.
//!
//! So everything here is read:
//!
//! * which inputs exist, what element type and rank they have, and which
//!   symbols name their axes — from `pipeline.workflow.inputs`;
//! * what those symbols *are* — from the component graphs the workflow names,
//!   by aligning each bound port's declared shape against the contract's, so a
//!   static graph dimension binds the contract symbol sitting on that axis;
//! * which inputs seed model state, and on which axis that state's positions
//!   live — from `serving.state_service.groups[*]` and the state cells whose
//!   `initializer` names the input.
//!
//! What a caller supplies is what the package says the *application* owns: the
//! prompt, and any application input it names explicitly (an attention mask,
//! above all — a zero-filled one attends to nothing, and no declaration in the
//! package says otherwise). Everything else the application owns is zero-filled
//! at its declared shape.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use onnx_genai_engine::pipeline::speculative::{ChainedProposal, ChainedProposalOptions};
use onnx_genai_engine::{
    Engine, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
    PipelineTensors,
};
use onnx_genai_metadata::{ComponentImplementation, TensorDimension, WorkflowSpec, WorkflowStep};
use onnx_genai_ort::{DataType, Value};

/// How an application input this harness does not otherwise know is filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fill {
    /// Every element zero. The default for an opaque application input.
    Zeros,
    /// Every element one. What an attention mask over a fully visible context
    /// is, and what a zero-filled one would silently not be.
    Ones,
}

/// A published workflow package, bound to the extents its own graphs declare.
pub struct RealWorkflowPackage {
    engine: Engine,
    workflow: WorkflowSpec,
    /// Extent of every symbol the component graphs pin to a constant.
    graph_symbols: BTreeMap<String, i64>,
    /// Inputs that seed a state-service cell, and that cell's position axis.
    state_seed_axis: BTreeMap<String, usize>,
    /// Each state cell's own position axis, for checking a rollback.
    cell_axis: BTreeMap<String, usize>,
    /// Symbols that sit on a state seed's position axis: the *past* length,
    /// which is not the prompt length and must not be bound to it.
    past_symbols: BTreeSet<String>,
    /// The declared input carrying prompt token ids.
    prompt_input: String,
    /// Application inputs the caller fills explicitly, by declared input name.
    fills: HashMap<String, Fill>,
}

impl RealWorkflowPackage {
    /// Bind `engine`'s package to the extents its graphs declare.
    ///
    /// `root` is the package directory the engine was loaded from; the
    /// component artifacts are resolved beneath it exactly as the loader does.
    pub fn new(engine: Engine, root: &Path) -> anyhow::Result<Self> {
        let workflow = engine
            .package_workflow()
            .context(
                "this package declares no pipeline.workflow, so there is nothing to drive \
                 generically; a real-package evidence run needs a workflow package",
            )?
            .clone();
        let graph_symbols = resolve_graph_symbols(&workflow, root)?;
        let (state_seed_axis, cell_axis, past_symbols) = state_seed_axes(&workflow);
        let prompt_input = prompt_token_input(&workflow)?;
        Ok(Self {
            engine,
            workflow,
            graph_symbols,
            state_seed_axis,
            cell_axis,
            past_symbols,
            prompt_input,
            fills: HashMap::new(),
        })
    }

    /// Fill the application input whose *declared source name* is `source`.
    ///
    /// Keyed by the source name rather than the workflow input name because
    /// that is the name the package tells an application to use.
    ///
    /// A source the package does not declare is an error, not a no-op: the
    /// fills a caller asks for are the ones the zero default would silently get
    /// wrong — a zero attention mask attends to nothing, and a run that
    /// degraded that way would still agree with a reference that degraded
    /// identically.
    pub fn fill(mut self, source: &str, fill: Fill) -> anyhow::Result<Self> {
        let name = application_input_named(&self.workflow, source).with_context(|| {
            format!(
                "no declared workflow input takes its value from application source '{source}', \
                 so the fill this run depends on would be silently dropped"
            )
        })?;
        self.fills.insert(name, fill);
        Ok(self)
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    pub fn workflow(&self) -> &WorkflowSpec {
        &self.workflow
    }

    /// Symbols this package's graphs pin, for an evidence record.
    pub fn graph_symbols(&self) -> &BTreeMap<String, i64> {
        &self.graph_symbols
    }

    /// A single forward pass over `tokens` against an empty cache.
    ///
    /// Every pass re-reads the whole context. That is deliberate for evidence:
    /// it removes the cache from the comparison entirely, so a token stream that
    /// matches matches because the *graphs* agree, not because two runs happened
    /// to share a cache state.
    pub fn request(&self, tokens: &[i64]) -> anyhow::Result<PipelineGenerateRequest> {
        let mut symbols = self.graph_symbols.clone();
        let sequence = i64::try_from(tokens.len()).context("context length exceeds i64")?;
        let mut request = PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(Vec::new()),
            options: GenerateOptions {
                max_new_tokens: 1,
                greedy: true,
                temperature: 0.0,
                stop_on_eos: false,
                ..Default::default()
            },
        });
        // The prompt's own axes bind the package's sequence symbols first, so
        // every other input is shaped against the same context.
        bind_contract_symbols(
            &self.workflow,
            &self.prompt_input,
            &[1, sequence],
            &mut symbols,
        )?;
        request = request.with_input(
            self.prompt_input.clone(),
            Value::from_slice_i64(tokens, &[1, sequence])?,
        );

        for (name, input) in &self.workflow.inputs {
            if *name == self.prompt_input || !input.required {
                continue;
            }
            // A state seed's position axis is the *past*, which a pass over the
            // whole context starts empty. Binding it to the prompt's length
            // would ask the graph to attend over a cache that does not exist.
            let past_axis = self.state_seed_axis.get(name).copied();
            let shape = resolve_shape(name, input, &symbols, past_axis)?;
            let dtype = contract_dtype(name, &input.contract)?;
            let fill = self.fills.get(name).copied().unwrap_or(Fill::Zeros);
            request = request.with_input(name.clone(), filled(&shape, dtype, fill)?);
        }
        Ok(request)
    }

    /// Run one pass and keep the whole SSA map, which a proposal chain reads.
    pub fn run(&mut self, tokens: &[i64]) -> anyhow::Result<PipelineTensors> {
        let request = self.request(tokens)?;
        self.engine.run_pipeline_retained(request)
    }

    /// The package's own greedy next token after `tokens`.
    pub fn greedy_next(&mut self, tokens: &[i64]) -> anyhow::Result<i64> {
        let values = self.run(tokens)?;
        Ok(i64::from(logits_of(&values)?.argmax_last_row()?))
    }

    /// `budget` tokens of plain greedy decoding, re-reading the whole context.
    pub fn greedy_decode(&mut self, prompt: &[i64], budget: usize) -> anyhow::Result<Vec<i64>> {
        let mut committed = Vec::with_capacity(budget);
        while committed.len() < budget {
            let mut context = prompt.to_vec();
            context.extend_from_slice(&committed);
            committed.push(self.greedy_next(&context)?);
        }
        Ok(committed)
    }

    /// Materialize a chained proposal of `width` positions after `tokens`.
    pub fn propose(&mut self, tokens: &[i64], width: usize) -> anyhow::Result<ChainedProposal> {
        let values = self.run(tokens)?;
        let guaranteed = i64::from(logits_of(&values)?.argmax_last_row()?);
        self.engine.propose_chained(
            &values,
            ChainedProposalOptions {
                seed_token: *tokens
                    .last()
                    .context("a proposal needs a non-empty context")?,
                guaranteed_token: guaranteed,
                width,
            },
        )
    }

    /// Speculatively decode `budget` tokens at proposal `width`.
    ///
    /// Propose, verify the whole block in one pass, accept the confirmed
    /// prefix, and roll every declared state cell back on a rejection. The
    /// tally is returned so a caller can prove both branches ran rather than
    /// trusting that a rejection ever happened.
    pub fn speculative_decode(
        &mut self,
        prompt: &[i64],
        budget: usize,
        width: usize,
    ) -> anyhow::Result<(Vec<i64>, SpeculativeTally)> {
        let mut committed: Vec<i64> = Vec::new();
        let mut tally = SpeculativeTally::default();
        while committed.len() < budget {
            let mut context = prompt.to_vec();
            context.extend_from_slice(&committed);

            let values = self.run(&context)?;
            let guaranteed = i64::from(logits_of(&values)?.argmax_last_row()?);
            let proposal = self.engine.propose_chained(
                &values,
                ChainedProposalOptions {
                    seed_token: *context.last().context("context is non-empty")?,
                    guaranteed_token: guaranteed,
                    width,
                },
            )?;
            tally.proposed += proposal.drafts().len();
            tally.proposer_invocations += proposal.proposer_invocations;

            // The whole block is verified, guaranteed token included: it is the
            // target's own next token, so the drafts that follow it are only
            // meaningful in a pass that has consumed it.
            let mut block_context = context.clone();
            block_context.extend_from_slice(&proposal.tokens);
            let verified = self.run(&block_context)?;
            let target_tokens =
                block_aligned_predictions(&verified, context.len(), proposal.tokens.len())?;
            let acceptance = self
                .engine
                .accept_chained_proposal(&proposal, &target_tokens)?;
            // Position 0 of a block is the target's own guaranteed token, so
            // the *drafts* accepted are one fewer than the block positions.
            tally.rounds += 1;
            tally.accepted_drafts += acceptance.accepted.saturating_sub(1);
            if acceptance.accepted > 1 {
                tally.rounds_with_an_accepted_draft += 1;
            }
            tally.rejected_drafts += proposal
                .drafts()
                .len()
                .saturating_sub(acceptance.accepted.saturating_sub(1));
            if acceptance.rejected_at.is_some() {
                tally.rejections += 1;
                let mut state = self.engine.speculative_rollback_state(&verified)?;
                let length = context.len() + acceptance.committed.len();
                self.engine.rollback_speculative_state(&mut state, length)?;
                for (cell, value) in &state {
                    // Each cell's position axis comes from *its own* service
                    // group, exactly as the runtime resolves it: a package with
                    // two groups at different axes is not one axis.
                    let axis = self.cell_axis.get(cell).copied().with_context(|| {
                        format!(
                            "rolled-back cell '{cell}' belongs to no state-service group with a \
                             declared sequence_axis, so there is no axis to check it on"
                        )
                    })?;
                    let extent = value.shape().get(axis).copied().with_context(|| {
                        format!(
                            "rolled-back cell '{cell}' has shape {:?}, which has no axis {axis}",
                            value.shape()
                        )
                    })?;
                    anyhow::ensure!(
                        extent as usize == length,
                        "state cell '{cell}' was not rolled back: axis {axis} is {extent}, \
                         expected {length}"
                    );
                }
                tally.rolled_back_cells += state.len();
            } else {
                tally.full_accepts += 1;
            }
            committed.extend_from_slice(&acceptance.committed);
        }
        committed.truncate(budget);
        Ok((committed, tally))
    }
}

/// How a speculative decode run split between acceptance and rejection.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeTally {
    /// Propose/verify rounds run.
    pub rounds: usize,
    /// Rounds in which the target confirmed at least the proposer's first
    /// draft.
    ///
    /// The token stream cannot carry a claim about proposal quality — a
    /// verified block always commits the target's own tokens, whatever the
    /// proposer said — so this is the statistic that can.
    pub rounds_with_an_accepted_draft: usize,
    /// Draft tokens the proposer produced, excluding the target's own token.
    pub proposed: usize,
    /// Draft tokens the target confirmed.
    pub accepted_drafts: usize,
    /// Draft tokens the target contradicted or discarded behind one.
    pub rejected_drafts: usize,
    /// Rounds that ended in a rejection, and therefore a rollback.
    pub rejections: usize,
    /// Rounds in which the target confirmed the whole block.
    pub full_accepts: usize,
    pub proposer_invocations: usize,
    pub rolled_back_cells: usize,
}

/// The emitted logits of a pass.
pub fn logits_of(values: &PipelineTensors) -> anyhow::Result<&Value> {
    values
        .get("logits")
        .context("the package did not emit a 'logits' output for this pass")
}

/// The target's block-aligned tokens, one entry past the block.
///
/// Verification consumes `context + block` in one pass, so the target's token
/// for block position `i` is its prediction at index `context_len - 1 + i`.
pub fn block_aligned_predictions(
    values: &PipelineTensors,
    context_len: usize,
    block_len: usize,
) -> anyhow::Result<Vec<i64>> {
    let logits = logits_of(values)?;
    let vocab = *logits.shape().last().context("logits have a vocab axis")? as usize;
    // Real exports are fp16; widening losslessly is what lets one reader serve
    // both an fp16 and an fp32 package.
    let data = logits
        .to_vec_f32_lossy()
        .map_err(|error| anyhow::anyhow!("reading verification logits: {error}"))?;
    let predictions = data
        .chunks_exact(vocab)
        .map(|row| {
            let mut best = 0usize;
            for (index, value) in row.iter().enumerate() {
                if *value > row[best] {
                    best = index;
                }
            }
            best as i64
        })
        .collect::<Vec<_>>();
    let start = context_len - 1;
    anyhow::ensure!(
        start + block_len < predictions.len(),
        "a verification pass over {context_len} context and {block_len} block positions produced \
         {} prediction rows; block alignment needs {}",
        predictions.len(),
        start + block_len + 1
    );
    Ok(predictions[start..start + block_len + 1].to_vec())
}

/// The declared input carrying prompt token ids.
fn prompt_token_input(workflow: &WorkflowSpec) -> anyhow::Result<String> {
    workflow
        .inputs
        .iter()
        .find(|(_, input)| {
            matches!(
                &input.role,
                onnx_genai_metadata::SemanticInputRole::Runtime {
                    role: onnx_genai_metadata::RuntimeInputRole::PromptTokens,
                    ..
                }
            )
        })
        .map(|(name, _)| name.clone())
        .context(
            "no declared workflow input carries the prompt_tokens runtime role, so nothing names \
             where this package takes its prompt",
        )
}

/// The workflow input whose application source name is `source`.
fn application_input_named(workflow: &WorkflowSpec, source: &str) -> Option<String> {
    workflow
        .inputs
        .iter()
        .find(|(_, input)| {
            matches!(
                &input.source,
                onnx_genai_metadata::WorkflowInputSource::Application { name } if name == source
            )
        })
        .map(|(name, _)| name.clone())
}

/// Inputs that seed a state-service cell, mapped to that state's position axis.
///
/// Returned alongside the symbols sitting on those axes: they measure the past,
/// which a fresh invocation starts empty, and binding them from the prompt
/// would size a cache that has nothing in it.
#[allow(clippy::type_complexity)]
fn state_seed_axes(
    workflow: &WorkflowSpec,
) -> (
    BTreeMap<String, usize>,
    BTreeMap<String, usize>,
    BTreeSet<String>,
) {
    let mut axis_of_cell: BTreeMap<&str, usize> = BTreeMap::new();
    if let Some(serving) = workflow.serving.as_ref() {
        for group in serving.state_service.groups.values() {
            let Some(axis) = group.sequence_axis else {
                continue;
            };
            for aliases in group.ports.values() {
                for cell in aliases.keys() {
                    axis_of_cell.insert(cell.as_str(), axis);
                }
            }
        }
    }
    let mut seeds = BTreeMap::new();
    let mut cells = BTreeMap::new();
    let mut past = BTreeSet::new();
    for (cell_name, cell) in &workflow.state {
        let Some(axis) = axis_of_cell.get(cell_name.as_str()).copied() else {
            continue;
        };
        cells.insert(cell_name.clone(), axis);
        if !workflow.inputs.contains_key(&cell.initializer) {
            continue;
        }
        seeds.insert(cell.initializer.clone(), axis);
        if let Some(TensorDimension::Symbol(symbol)) = cell
            .contract
            .shape
            .as_ref()
            .and_then(|shape| shape.get(axis))
        {
            past.insert(symbol.clone());
        }
    }
    (seeds, cells, past)
}

/// Bind the symbols on `input`'s contract from a concrete shape.
fn bind_contract_symbols(
    workflow: &WorkflowSpec,
    input: &str,
    shape: &[i64],
    symbols: &mut BTreeMap<String, i64>,
) -> anyhow::Result<()> {
    let declared = workflow
        .inputs
        .get(input)
        .with_context(|| format!("'{input}' is not a declared workflow input"))?;
    let Some(contract_shape) = declared.contract.shape.as_ref() else {
        return Ok(());
    };
    for (dimension, extent) in contract_shape.iter().zip(shape) {
        if let TensorDimension::Symbol(symbol) = dimension {
            symbols.insert(symbol.clone(), *extent);
        }
    }
    Ok(())
}

/// Every symbol a component graph pins to a constant.
///
/// A workflow contract says `[batch, full_kv_heads, sequence, full_head_dim]`;
/// the graph the workflow binds that value into says `[batch, 1, past, 512]`.
/// Walking the two shapes together is what turns the package's own names into
/// numbers without a test ever writing one down.
fn resolve_graph_symbols(
    workflow: &WorkflowSpec,
    root: &Path,
) -> anyhow::Result<BTreeMap<String, i64>> {
    let mut symbols: BTreeMap<String, i64> = BTreeMap::new();
    let mut graphs: HashMap<String, onnx_runtime_ir::Graph> = HashMap::new();
    for step in walk_invokes(&workflow.steps) {
        let WorkflowStep::Invoke {
            component, inputs, ..
        } = step
        else {
            continue;
        };
        let Some(declaration) = workflow.components.get(component) else {
            continue;
        };
        let ComponentImplementation::Onnx { artifact } = &declaration.implementation else {
            continue;
        };
        let graph = match graphs.get(component) {
            Some(graph) => graph,
            None => {
                let path: PathBuf = root.join(artifact);
                let graph = onnx_runtime_loader::load_model(&path).with_context(|| {
                    format!(
                        "failed to read component '{component}' at {} to resolve its declared \
                         extents",
                        path.display()
                    )
                })?;
                graphs.entry(component.clone()).or_insert(graph)
            }
        };
        for (port, value) in inputs {
            let Some(input) = workflow.inputs.get(value) else {
                continue;
            };
            let Some(contract_shape) = input.contract.shape.as_ref() else {
                continue;
            };
            let Some(port_shape) = graph_input_shape(graph, port) else {
                continue;
            };
            for (dimension, extent) in contract_shape.iter().zip(port_shape) {
                if let (TensorDimension::Symbol(symbol), Some(extent)) = (dimension, extent) {
                    symbols.insert(symbol.clone(), extent);
                }
            }
        }
    }
    Ok(symbols)
}

/// A graph input's declared dimensions, `None` where the graph is symbolic.
fn graph_input_shape(graph: &onnx_runtime_ir::Graph, port: &str) -> Option<Vec<Option<i64>>> {
    let value = graph
        .inputs
        .iter()
        .filter_map(|id| graph.values.get(*id))
        .find(|value| value.name.as_deref() == Some(port))?;
    Some(
        value
            .shape
            .iter()
            .map(|dimension| dimension.as_static().and_then(|e| i64::try_from(e).ok()))
            .collect(),
    )
}

/// Every `Invoke` in a workflow, including inside control flow.
fn walk_invokes(steps: &[WorkflowStep]) -> Vec<&WorkflowStep> {
    let mut found = Vec::new();
    for step in steps {
        match step {
            WorkflowStep::Invoke { .. } => found.push(step),
            WorkflowStep::Loop { setup, steps, .. } => {
                found.extend(walk_invokes(setup));
                found.extend(walk_invokes(steps));
            }
            WorkflowStep::Branch { cases, default, .. } => {
                for case in cases.values() {
                    found.extend(walk_invokes(std::slice::from_ref(case)));
                }
                if let Some(default) = default {
                    found.extend(walk_invokes(std::slice::from_ref(default.as_ref())));
                }
            }
            _ => {}
        }
    }
    found
}

/// A declared input's concrete shape for this pass.
fn resolve_shape(
    name: &str,
    input: &onnx_genai_metadata::WorkflowInput,
    symbols: &BTreeMap<String, i64>,
    past_axis: Option<usize>,
) -> anyhow::Result<Vec<i64>> {
    let Some(shape) = input.contract.shape.as_ref() else {
        return Ok(vec![1; input.contract.rank]);
    };
    shape
        .iter()
        .enumerate()
        .map(|(axis, dimension)| match dimension {
            TensorDimension::Fixed(extent) => Ok(*extent),
            // A fresh invocation's cache is empty, whatever symbol the package
            // wrote on that axis.
            TensorDimension::Symbol(_) if past_axis == Some(axis) => Ok(0),
            TensorDimension::Symbol(symbol) => symbols.get(symbol).copied().with_context(|| {
                format!(
                    "workflow input '{name}' axis {axis} is symbol '{symbol}', which neither the \
                     prompt nor any component graph binds to an extent; the package must pin it \
                     on a graph port or share it with an input the caller supplies"
                )
            }),
        })
        .collect()
}

/// The ORT element type a contract's dtype names.
fn contract_dtype(
    name: &str,
    contract: &onnx_genai_metadata::TensorContract,
) -> anyhow::Result<DataType> {
    Ok(match contract.dtype.as_str() {
        "float32" | "fp32" => DataType::Float32,
        "float16" | "fp16" => DataType::Float16,
        "bfloat16" | "bf16" => DataType::BFloat16,
        "int64" => DataType::Int64,
        "int32" => DataType::Int32,
        "bool" => DataType::Bool,
        other => anyhow::bail!(
            "workflow input '{name}' declares dtype '{other}', which this evidence harness does \
             not fill; add it here or supply the input explicitly"
        ),
    })
}

/// A zero- or one-filled tensor of `shape` and `dtype`.
fn filled(shape: &[i64], dtype: DataType, fill: Fill) -> anyhow::Result<Value> {
    let numel = shape
        .iter()
        .try_fold(1usize, |total, extent| {
            usize::try_from(*extent)
                .ok()
                .and_then(|e| total.checked_mul(e))
        })
        .with_context(|| format!("unusable tensor shape {shape:?}"))?;
    let unit: Vec<u8> = match (fill, dtype) {
        (Fill::Zeros, _) => vec![0u8; dtype.size_of()],
        (Fill::Ones, DataType::Float32) => 1.0f32.to_le_bytes().to_vec(),
        (Fill::Ones, DataType::Float16) => half::f16::from_f32(1.0).to_le_bytes().to_vec(),
        (Fill::Ones, DataType::BFloat16) => half::bf16::from_f32(1.0).to_le_bytes().to_vec(),
        (Fill::Ones, DataType::Int64) => 1i64.to_le_bytes().to_vec(),
        (Fill::Ones, DataType::Int32) => 1i32.to_le_bytes().to_vec(),
        (Fill::Ones, DataType::Bool) => vec![1u8],
        (Fill::Ones, other) => {
            anyhow::bail!("a one-filled {other:?} tensor has no defined encoding here")
        }
    };
    Value::from_raw_bytes(unit.repeat(numel), shape, dtype)
        .map_err(|error| anyhow::anyhow!("failed to fill a {shape:?} {dtype:?} tensor: {error}"))
}
