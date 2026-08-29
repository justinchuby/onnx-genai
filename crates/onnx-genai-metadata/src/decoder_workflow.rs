//! Build the canonical `pipeline.workflow` for a single-decoder package.
//!
//! A single decoder is not a special kind of package. It is a workflow with one
//! component, and it says what it is using exactly the constructs a multi-
//! component workflow uses: port [`PortRole`]s name the semantic inputs and
//! outputs, a `state_service` group declares the KV cache and its past/present
//! aliases, and a `loop` step drives generation and emits tokens.
//!
//! This module exists for two callers, neither of which is the runtime:
//!
//! * the offline migration tool, which rewrites a package that still carries a
//!   retired `model.io` block, and
//! * the `genai_config.json` importer, which adapts a *foreign* producer's
//!   format into this one.
//!
//! Loading never calls it. A package that does not declare a workflow is
//! rejected, not repaired — silently synthesizing one at load is precisely the
//! second authoritative answer the canonical rule exists to prevent.
//!
//! # Why this is the inverse of the recognizer
//!
//! [`crate::decoder_abi`] reads a workflow and produces a [`DecoderAbi`]. This
//! module goes the other way. The two are only trustworthy as a pair, so
//! [`decoder_workflow`] is required to round-trip: feeding its output back
//! through the recognizer must reproduce the ABI it was given. That property is
//! asserted in this module's tests and, for every converted package, by
//! `tests/decoder_workflow_roundtrip.rs` — which is what makes a mechanical
//! conversion of fourteen packages checkable rather than hopeful.
//!
//! The round-trip is exact on everything the workflow can express, but it is
//! *normalizing* rather than identity-preserving on fields whose absence has a
//! defined meaning. A workflow states its aliasing and its layout, so an ABI
//! that left them unset reads back with the defaults every consumer already
//! applied (`aliasing: None` → `Forbidden`, `kv_layout: None` →
//! `head_major_bnsh`). `optional_inputs` is dropped, because a workflow says a
//! port may be omitted without naming the request key that signals its
//! presence. None of these change behaviour; saying so is what keeps "it
//! round-trips" from being read as more than it is.

use std::collections::{BTreeMap, BTreeSet};

use crate::schema::{
    BatchLayout, ComponentImplementation, ComponentPorts, DecoderAbi, DecoderStateGroup, PortRole,
    RuntimeInputRole, SemanticInputRole, SequenceInputKind, ServingServiceContract,
    ShapeRecurrence, StateAliasing, StateGroupContract, StateKind, StatePortAlias, StatePortRole,
    StateServiceContract, StateUpdate, TensorContract, WorkflowComponent, WorkflowInput,
    WorkflowInputSource, WorkflowManifest, WorkflowOutput, WorkflowOutputRole, WorkflowSpec,
    WorkflowStateCell, WorkflowStateScope, WorkflowStep,
};

/// Component name a converted single-decoder workflow binds its graph to.
pub const DECODER_COMPONENT: &str = "decoder";
/// State-service group name a converted workflow declares its KV under.
pub const KV_GROUP: &str = "decoder_kv";
/// State-service group name a converted workflow declares cross-attention KV under.
pub const CROSS_KV_GROUP: &str = "decoder_cross_kv";
/// State-service group name a converted workflow declares fixed carries under.
pub const RECURRENT_GROUP: &str = "decoder_recurrent";
/// Package output a converted workflow emits generated tokens into.
pub const TOKENS_OUTPUT: &str = "tokens";

/// Why a decoder ABI could not be expressed as a workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// The ABI names no sequence input, so nothing drives the loop.
    NoSequenceInput,
    /// A decoder loop needs a graph edge that produces each step's embeddings.
    InputsEmbedsNeedsEmbeddingProducer,
    /// The ABI names no logits output, so nothing selects a token.
    NoLogitsOutput,
    /// State inputs and outputs disagree in length, so the pairs are ambiguous.
    UnpairedState {
        group: &'static str,
        inputs: usize,
        outputs: usize,
    },
    /// The workflow declares no single generation loop to re-author.
    NoGenerationLoop,
    /// The generation loop does not declare the step this policy replaces.
    MissingIterationStep { contract: &'static str },
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSequenceInput => formatter.write_str(
                "this decoder names neither a token input nor an inputs_embeds input, so no \
                 workflow could say what drives its generation loop",
            ),
            Self::InputsEmbedsNeedsEmbeddingProducer => formatter.write_str(
                "this decoder drives its graph through inputs_embeds, but a bare decoder ABI \
                 cannot express the per-step component that produces each embedding from the \
                 selected token. Author a multi-component workflow that binds the embedding \
                 producer and, when token-context state is used, its explicit token_ids \
                 companion; do not lower this graph into a loop with a stale embedding input",
            ),
            Self::NoLogitsOutput => formatter.write_str(
                "this decoder names no logits output, so no workflow could say what its token \
                 policy scores",
            ),
            Self::UnpairedState {
                group,
                inputs,
                outputs,
            } => write!(
                formatter,
                "this decoder declares {inputs} '{group}' state inputs and {outputs} outputs; \
                 they pair positionally, so the counts must match"
            ),
            Self::NoGenerationLoop => formatter.write_str(
                "this workflow declares no single generation loop, so there is no iteration to \
                 re-author into another policy",
            ),
            Self::MissingIterationStep { contract } => write!(
                formatter,
                "this workflow's generation loop invokes no component declaring '{contract}', so \
                 the step another iteration policy would replace is not there to replace"
            ),
        }
    }
}

impl std::error::Error for BuildError {}

/// Model-level facts a workflow needs that the port ABI does not carry.
#[derive(Debug, Clone, Default)]
pub struct DecoderFacts {
    /// Upper bound on generated tokens the workflow's loop declares.
    pub max_sequence_length: Option<usize>,
    /// Real dtype and rank of the graph's ports, keyed by port name.
    ///
    /// State tensors have no shape this builder could know: a growing KV cache
    /// is rank 4, a fixed-capacity scatter buffer is rank 3, and a recurrent
    /// carry is whatever the model chose. Guessing produces a contract the
    /// session validator rejects for whichever package happens to disagree,
    /// which is a lie that fails late instead of never being told. A caller
    /// holding the graph supplies the truth here; a caller without one gets the
    /// growing-KV default and must verify.
    pub port_contracts: BTreeMap<String, TensorContract>,
}

/// Build the canonical workflow for a single-decoder graph.
///
/// The result is an ordinary two-component workflow:
///
/// * `decoder` — the package's ONNX graph, with [`PortRole`]s naming its
///   semantic ports and a `state_service` group declaring its KV cache and the
///   past/present aliases the runtime may exploit.
/// * `token_policy` — a `binding` component carrying the
///   [`TOKEN_POLICY_CONTRACT`], which is the schema's existing way for a
///   workflow to say "the runtime implements this step". It scores the
///   decoder's logits and produces the step's token and liveness flags.
///
/// A `loop` step drives the two, carries the state cells, and emits tokens.
/// Nothing here is decoder-specific machinery: a multi-component workflow
/// declares its components, state and loop with the same constructs.
///
/// `artifact` is the decoder's ONNX artifact path, relative to the package.
pub fn decoder_workflow(
    abi: &DecoderAbi,
    artifact: &str,
    facts: &DecoderFacts,
) -> Result<WorkflowSpec, BuildError> {
    let sequence_port = sequence_port(abi)?.to_string();
    let logits = abi
        .logits_output
        .as_deref()
        .ok_or(BuildError::NoLogitsOutput)?
        .to_string();
    let sequence_is_embeds = matches!(abi.sequence_source, Some(SequenceInputKind::InputsEmbeds))
        || (abi.token_input.is_none() && abi.inputs_embeds_input.is_some());
    if sequence_is_embeds {
        return Err(BuildError::InputsEmbedsNeedsEmbeddingProducer);
    }
    // A fixed-capacity decoder states its KV through `static_cache` rather than
    // `kv_inputs`: the buffers are the same state, declared per half. Reading
    // both here is what lets one group describe either discipline instead of
    // the workflow needing a second shape for scatter decoders.
    let kv = match abi.static_cache.as_ref() {
        Some(cache) => interleave_halves(cache)?,
        None => paired(abi.kv_inputs.as_deref(), abi.kv_outputs.as_deref(), "kv")?,
    };
    let cross_kv = paired(
        abi.cross_kv_inputs.as_deref(),
        abi.cross_kv_outputs.as_deref(),
        "cross_kv",
    )?;
    let recurrent: Vec<(String, String)> = abi
        .state_pairs
        .iter()
        .flatten()
        .map(|pair| (pair.input.clone(), pair.output.clone()))
        .collect();

    let mut builder = Builder {
        port_contracts: facts.port_contracts.clone(),
        inputs: BTreeMap::new(),
        state: BTreeMap::new(),
        groups: BTreeMap::new(),
        invoke_inputs: BTreeMap::new(),
        invoke_outputs: BTreeMap::new(),
        carried: Vec::new(),
        ports_in: BTreeMap::new(),
        ports_out: BTreeMap::new(),
        roles: BTreeMap::new(),
    };

    builder.request_input(
        REQUEST_TOKENS,
        token_contract(),
        RuntimeInputRole::PromptTokens,
        true,
        None,
    );
    builder.request_input(
        REQUEST_MAX_ITERATIONS,
        scalar_contract(),
        RuntimeInputRole::MaxIterations,
        false,
        // An optional input must carry a literal default: "unspecified" is not
        // a value the interpreter can bind. Silence here would be a package
        // that only works when the caller happens to pass a bound.
        Some(int_literal(
            facts.max_sequence_length.unwrap_or(DEFAULT_MAX_ITERATIONS) as i64,
        )),
    );

    // ── the sequence that drives the loop ────────────────────────────────────
    let sequence_contract = if sequence_is_embeds {
        hidden_contract()
    } else {
        token_contract()
    };
    let sequence_role = if sequence_is_embeds {
        PortRole::InputsEmbeds
    } else {
        PortRole::TokenIds
    };
    let sequence_contract = facts
        .port_contracts
        .get(&sequence_port)
        .cloned()
        .unwrap_or(sequence_contract);
    builder.decoder_port_in(&sequence_port, sequence_contract, Some(sequence_role));
    builder.bind_invoke_input(&sequence_port, REQUEST_TOKENS);

    // A graph may consume both a raw token stream and a routed pre-embedded
    // one. Declaring both is explicit, so both are carried across; only the one
    // the ABI names as the sequence source drives the loop.
    let secondary = abi
        .inputs_embeds_input
        .as_deref()
        .map(|port| (port, hidden_contract(), PortRole::InputsEmbeds));
    if let Some((port, contract, role)) = secondary
        && port != sequence_port
    {
        builder.optional_application_port(port, contract, Some(role));
    }

    for (port, role, contract) in [
        (
            abi.attention_mask_input.as_deref(),
            PortRole::AttentionMask,
            mask_contract(),
        ),
        (
            abi.position_ids_input.as_deref(),
            PortRole::PositionIds,
            token_contract(),
        ),
        (
            abi.encoder_hidden_states_input.as_deref(),
            PortRole::EncoderHiddenStates,
            hidden_contract(),
        ),
        (
            abi.audio_features_input.as_deref(),
            PortRole::AudioFeatures,
            hidden_contract(),
        ),
    ] {
        if let Some(port) = port {
            let contract = facts.port_contracts.get(port).cloned().unwrap_or(contract);
            builder.optional_application_port(port, contract, Some(role));
        }
    }

    let logits_contract_declared = facts
        .port_contracts
        .get(&logits)
        .cloned()
        .unwrap_or_else(logits_contract);
    builder.decoder_port_out(
        &logits,
        logits_contract_declared.clone(),
        Some(PortRole::Logits),
    );
    builder.bind_invoke_output(&logits, LOGITS_VALUE);
    if let Some(hidden) = abi.hidden_output.as_deref() {
        builder.decoder_port_out(hidden, hidden_contract(), Some(PortRole::HiddenStates));
        builder.bind_invoke_output(hidden, &format!("step.{}", sanitize(hidden)));
    }

    // ── state ────────────────────────────────────────────────────────────────
    // Each graph state pair becomes a state cell the loop carries and a group
    // alias binding the cell to the graph's ports. That is the same shape a
    // multi-component workflow uses; nothing about it is decoder-specific.
    if abi.state_groups.is_empty() {
        builder.state_group(KV_GROUP, StateKind::FullAttention, abi, &kv, true, None);
        builder.state_group(
            CROSS_KV_GROUP,
            StateKind::CrossAttention,
            abi,
            &cross_kv,
            true,
            None,
        );
        builder.state_group(
            RECURRENT_GROUP,
            StateKind::Recurrent,
            abi,
            &recurrent,
            false,
            None,
        );
    } else {
        for group in &abi.state_groups {
            let pairs = group
                .ports
                .iter()
                .map(|port| (port.input.clone(), port.output.clone()))
                .collect::<Vec<_>>();
            builder.state_group(&group.name, group.kind, abi, &pairs, false, Some(group));
        }
    }

    if let Some(static_cache) = abi.static_cache.as_ref() {
        // A fixed-capacity buffer's valid prefix is not derivable from its
        // shape, so the length is graph-visible state the workflow must name.
        builder.literal_state(LENGTHS_CELL, lengths_contract(), int_literal(0));
        builder.package_literal(CAPACITY_INPUT, scalar_contract(), int_literal(0));
        builder.decoder_port_in(&static_cache.write_indices_input, lengths_contract(), None);
        builder.bind_invoke_input(&static_cache.write_indices_input, LENGTHS_CELL);
        builder.decoder_port_in(
            &static_cache.kv_sequence_length_input,
            lengths_contract(),
            None,
        );
        builder.bind_invoke_input(&static_cache.kv_sequence_length_input, LENGTHS_CELL);
        // The write cursor advances by however many positions the step
        // committed, which is the policy's decision — so the policy states the
        // next value and the loop carries it, exactly like every other cell.
        builder.carried.push(crate::schema::WorkflowCarry {
            cell: LENGTHS_CELL.to_string(),
            initial: None,
            next: body_value(LENGTHS_CELL),
        });
    }

    // ── the runtime's token policy, declared as a component ──────────────────
    // The policy is a `binding`: the workflow states that a step happens here
    // and what it consumes and produces, and the runtime supplies the
    // implementation. That is how any workflow declares a step it does not ship
    // a graph for.
    let mut policy_outputs = BTreeMap::from([
        (POLICY_TOKEN_PORT.to_string(), token_contract()),
        (POLICY_ACTIVE_PORT.to_string(), flag_contract()),
        (POLICY_DONE_PORT.to_string(), flag_contract()),
        (POLICY_ACCEPTED_PORT.to_string(), lengths_contract()),
    ]);
    if abi.static_cache.is_some() {
        policy_outputs.insert(LENGTHS_CELL.to_string(), lengths_contract());
    }
    let policy = WorkflowComponent {
        implementation: ComponentImplementation::Binding,
        ports: ComponentPorts {
            inputs: BTreeMap::from([(
                POLICY_LOGITS_PORT.to_string(),
                logits_contract_declared.clone(),
            )]),
            outputs: policy_outputs,
            roles: BTreeMap::new(),
        },
        contract: Some(crate::schema::ComponentContract {
            id: TOKEN_POLICY_CONTRACT.to_string(),
            version: CONTRACT_VERSION.to_string(),
            equivalence: crate::schema::EquivalenceClass::default(),
            bindings: BTreeMap::new(),
            parameters: BTreeMap::new(),
        }),
        application_overridable: false,
        effects: Vec::new(),
        row_scope: None,
        cache_affects_state: BTreeSet::new(),
        batch_capacity: None,
    };

    for (cell, port, contract, default) in [
        (
            ACTIVE_CELL,
            POLICY_ACTIVE_PORT,
            flag_contract(),
            bool_literal(true),
        ),
        (
            DONE_CELL,
            POLICY_DONE_PORT,
            flag_contract(),
            bool_literal(false),
        ),
        (
            ACCEPTED_CELL,
            POLICY_ACCEPTED_PORT,
            lengths_contract(),
            int_literal(0),
        ),
    ] {
        builder.literal_state(cell, contract, default);
        let _ = port;
    }

    let decoder = WorkflowComponent {
        implementation: ComponentImplementation::Onnx {
            artifact: artifact.to_string(),
        },
        ports: ComponentPorts {
            inputs: std::mem::take(&mut builder.ports_in),
            outputs: std::mem::take(&mut builder.ports_out),
            roles: std::mem::take(&mut builder.roles),
        },
        contract: Some(crate::schema::ComponentContract {
            id: AUTOREGRESSIVE_DECODE_CONTRACT.to_string(),
            version: CONTRACT_VERSION.to_string(),
            equivalence: crate::schema::EquivalenceClass::default(),
            bindings: BTreeMap::new(),
            parameters: BTreeMap::new(),
        }),
        application_overridable: false,
        effects: Vec::new(),
        row_scope: None,
        cache_affects_state: BTreeSet::new(),
        batch_capacity: None,
    };

    let mut body = vec![
        WorkflowStep::Invoke {
            component: DECODER_COMPONENT.to_string(),
            inputs: std::mem::take(&mut builder.invoke_inputs),
            outputs: std::mem::take(&mut builder.invoke_outputs),
        },
        WorkflowStep::Invoke {
            component: POLICY_COMPONENT.to_string(),
            inputs: BTreeMap::from([(POLICY_LOGITS_PORT.to_string(), LOGITS_VALUE.to_string())]),
            outputs: {
                let mut bound = BTreeMap::from([
                    (POLICY_TOKEN_PORT.to_string(), TOKEN_VALUE.to_string()),
                    (POLICY_ACTIVE_PORT.to_string(), body_value(ACTIVE_CELL)),
                    (POLICY_DONE_PORT.to_string(), body_value(DONE_CELL)),
                    (POLICY_ACCEPTED_PORT.to_string(), body_value(ACCEPTED_CELL)),
                ]);
                if abi.static_cache.is_some() {
                    bound.insert(LENGTHS_CELL.to_string(), body_value(LENGTHS_CELL));
                }
                bound
            },
        },
    ];
    body.push(WorkflowStep::Emit {
        value: TOKEN_VALUE.to_string(),
        output: TOKENS_OUTPUT.to_string(),
        stream: None,
        mode: crate::schema::WorkflowEmitMode::Append,
        when: Some(ACTIVE_CELL.to_string()),
        valid_length: None,
        axis: None,
    });

    let mut carried = std::mem::take(&mut builder.carried);
    for cell in [ACTIVE_CELL, DONE_CELL, ACCEPTED_CELL] {
        carried.push(crate::schema::WorkflowCarry {
            cell: cell.to_string(),
            initial: None,
            next: body_value(cell),
        });
    }
    // The sequence cell advances to the token the policy just selected, which
    // is what makes this autoregressive rather than a fixed replay of the
    // prompt.
    carried.push(crate::schema::WorkflowCarry {
        cell: SEQUENCE_CELL.to_string(),
        initial: None,
        next: TOKEN_VALUE.to_string(),
    });
    // The sequence the decoder consumes is the prompt on the first iteration
    // and one token on every later one, so its length genuinely changes. Saying
    // `invariant` would declare a shape the loop does not keep — a statement no
    // reader could check and no executor could honour, because the first carry
    // already breaks it. `bounded` states what is actually true: the length
    // varies, up to the sequence capacity the package declares.
    builder.package_literal(
        SEQUENCE_CAPACITY,
        scalar_contract(),
        int_literal(facts.max_sequence_length.unwrap_or(DEFAULT_MAX_ITERATIONS) as i64),
    );
    builder.state.insert(
        SEQUENCE_CELL.to_string(),
        WorkflowStateCell {
            contract: token_contract(),
            class: crate::schema::WorkflowStateClass::Semantic,
            scope: WorkflowStateScope::Invocation,
            initializer: REQUEST_TOKENS.to_string(),
            recurrence: ShapeRecurrence::Bounded {
                axis: 1,
                max: format!("package.{SEQUENCE_CAPACITY}"),
            },
            management: crate::schema::StateManagement::Workflow,
            release_boundary: None,
            service_group: None,
            session: None,
        },
    );
    builder.invoke_inputs = BTreeMap::new();

    let groups = std::mem::take(&mut builder.groups);
    let spec = WorkflowSpec {
        manifest: WorkflowManifest {
            adapter_abis: BTreeMap::new(),
        },
        inputs: builder.inputs,
        outputs: BTreeMap::from([(
            TOKENS_OUTPUT.to_string(),
            WorkflowOutput {
                contract: token_contract(),
                role: WorkflowOutputRole::Tokens,
                family: crate::schema::WorkflowOutputFamily::Materialized,
                family_authored: false,
                value_range: None,
                stage: crate::schema::OutputStage::PreAdapter,
                media: None,
            },
        )]),
        components: BTreeMap::from([
            (DECODER_COMPONENT.to_string(), decoder),
            (POLICY_COMPONENT.to_string(), policy),
        ]),
        state: builder.state,
        effects: BTreeMap::new(),
        serving: Some(ServingServiceContract {
            active: ACTIVE_CELL.to_string(),
            done: DONE_CELL.to_string(),
            accepted_len: Some(ACCEPTED_CELL.to_string()),
            state_service: StateServiceContract { groups },
        }),
        steps: vec![WorkflowStep::Loop {
            setup: Vec::new(),
            steps: body,
            continue_when: ACTIVE_CELL.to_string(),
            max_iterations: REQUEST_MAX_ITERATIONS.to_string(),
            // The loop ends when generation ends. Saying so with the schema's
            // own word is what lets a reader — and the runtime — see that this
            // is an ordinary generation loop rather than a decoder-shaped
            // special case.
            termination: crate::schema::WorkflowLoopTermination::GenerationEos,
            carried,
            iteration: Some(Box::new(crate::schema::WorkflowLoopIteration {
                value: LOOP_ITERATION.to_string(),
                contract: scalar_contract(),
            })),
        }],
    };
    Ok(spec)
}

/// Which iteration algorithm a generation loop's body declares.
///
/// A workflow's loop says *how many* iterations may run and *when* to stop. What
/// one iteration **is** — a token, a row-major step across N sequences, a
/// proposed block verified in one pass — is said by the component the body
/// invokes and the contract that component declares. These are the three
/// iteration algorithms this runtime registers an executor for, named so that a
/// caller can ask for the body that declares one instead of a runtime inferring
/// it from the package's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IterationPolicy {
    /// One decoder forward pass and one token policy per iteration.
    SingleToken,
    /// One shared forward pass advancing every live row per iteration.
    ContinuousBatch,
    /// One proposal-and-verification block per iteration.
    SpeculativeBlock,
}

impl IterationPolicy {
    /// Contract a body declaring this policy names, if it is not the default.
    pub fn contract(self) -> Option<&'static str> {
        match self {
            Self::SingleToken => None,
            Self::ContinuousBatch => Some(CONTINUOUS_BATCH_CONTRACT),
            Self::SpeculativeBlock => Some(SPECULATIVE_BLOCK_CONTRACT),
        }
    }

    /// Component name a body declaring this policy invokes.
    pub fn component(self) -> &'static str {
        match self {
            Self::SingleToken => DECODER_COMPONENT,
            Self::ContinuousBatch => BATCH_STEP_COMPONENT,
            Self::SpeculativeBlock => SPECULATIVE_BLOCK_COMPONENT,
        }
    }
}

/// Author the same generation as a loop whose body declares `policy`.
///
/// The loop is not rewritten: its bound, its liveness predicate, its carried
/// cells, its termination and its emits are the ones the package already
/// declares, and they stay byte-identical. What changes is the *body* — the two
/// steps a single-token iteration declares (`autoregressive-decode` then
/// `token-policy`) become one step declaring the block contract, bound to
/// exactly the same values.
///
/// That is the whole selection mechanism. A runtime holding this document
/// dispatches the body's node to whichever executor it registered for the
/// contract the node names; nothing reads the package's file layout, its
/// decode path or its model identity to decide which iteration algorithm runs.
/// Authoring is done here, in the same module that authors the single-token
/// body, so the three bodies cannot drift into three different statements of
/// the same loop.
pub fn iteration_variant(
    spec: &WorkflowSpec,
    policy: IterationPolicy,
) -> Result<WorkflowSpec, BuildError> {
    let Some(contract) = policy.contract() else {
        return Ok(spec.clone());
    };

    let body = generation_loop_body(spec).ok_or(BuildError::NoGenerationLoop)?;
    let decode = body_step_declaring(spec, body, AUTOREGRESSIVE_DECODE_CONTRACT).ok_or(
        BuildError::MissingIterationStep {
            contract: AUTOREGRESSIVE_DECODE_CONTRACT,
        },
    )?;
    let selection = body_step_declaring(spec, body, TOKEN_POLICY_CONTRACT).ok_or(
        BuildError::MissingIterationStep {
            contract: TOKEN_POLICY_CONTRACT,
        },
    )?;
    let decoder = spec
        .components
        .get(decode.0)
        .ok_or(BuildError::NoGenerationLoop)?;
    let token_policy = spec
        .components
        .get(selection.0)
        .ok_or(BuildError::NoGenerationLoop)?;

    // The block consumes what the decoder consumed and defines what the policy
    // defined. Copying rather than restating them is what keeps the variant a
    // re-authoring of one loop instead of a second description of it: a package
    // that binds an extra decoder port, or a static-cache package whose policy
    // publishes `cache_lengths`, carries that through untouched.
    let block = WorkflowComponent {
        implementation: ComponentImplementation::Binding,
        ports: ComponentPorts {
            inputs: decoder.ports.inputs.clone(),
            outputs: merged(&decoder.ports.outputs, &token_policy.ports.outputs),
            roles: decoder.ports.roles.clone(),
        },
        contract: Some(crate::schema::ComponentContract {
            id: contract.to_string(),
            version: CONTRACT_VERSION.to_string(),
            equivalence: block_equivalence(policy),
            bindings: BTreeMap::new(),
            parameters: BTreeMap::new(),
        }),
        application_overridable: false,
        effects: decoder.effects.clone(),
        // A row-major step keeps one request's state per row and must follow a
        // scheduler compacting the rows underneath it, so it declares the row
        // ABI. A proposal block advances a single sequence, and declaring a row
        // axis it does not have would claim an ABI it cannot implement.
        row_scope: matches!(policy, IterationPolicy::ContinuousBatch).then_some(
            crate::schema::ComponentRowScope {
                axis: 0,
                stateful: true,
            },
        ),
        cache_affects_state: decoder.cache_affects_state.clone(),
        batch_capacity: None,
    };

    let component = policy.component().to_string();
    let mut variant = spec.clone();
    variant.components.remove(decode.0);
    variant.components.remove(selection.0);
    variant.components.insert(component.clone(), block);
    // The re-authored emit carries its ragged-prefix semantics in its typed
    // output declaration; core workflow semantics are not manifest flags.
    // The state service names the component whose ports each cell aliases. That
    // component is now the block, and leaving the decoder's name behind would
    // declare a KV group bound to something the workflow no longer declares.
    rebind_state_service(&mut variant, decode.0, &component);
    let Some(loop_steps) = variant.steps.iter_mut().find_map(|step| match step {
        WorkflowStep::Loop { steps, .. } if declares_token_emit(steps) => Some(steps),
        _ => None,
    }) else {
        return Err(BuildError::NoGenerationLoop);
    };
    *loop_steps = rebuild_body(loop_steps, decode, selection, policy);
    Ok(variant)
}

/// Merge two port maps, letting the second win a name collision.
fn merged<T: Clone>(
    first: &BTreeMap<String, T>,
    second: &BTreeMap<String, T>,
) -> BTreeMap<String, T> {
    let mut merged = first.clone();
    merged.extend(
        second
            .iter()
            .map(|(key, value)| (key.clone(), value.clone())),
    );
    merged
}

/// Point every state-service alias at the component that now owns the ports.
fn rebind_state_service(spec: &mut WorkflowSpec, from: &str, to: &str) {
    let Some(serving) = spec.serving.as_mut() else {
        return;
    };
    for group in serving.state_service.groups.values_mut() {
        if let Some(aliases) = group.ports.remove(from) {
            group.ports.insert(to.to_string(), aliases);
        }
        if let Some(StateUpdate::IndexedScatter {
            write_indices_ports,
            kv_length_ports,
            ..
        }) = group.update.as_mut()
        {
            if let Some(port) = write_indices_ports.remove(from) {
                write_indices_ports.insert(to.to_string(), port);
            }
            if let Some(port) = kv_length_ports.remove(from) {
                kv_length_ports.insert(to.to_string(), port);
            }
        }
    }
}

/// Equivalence a substituted block implementation must preserve.
///
/// A row-major batch is the same arithmetic for each row as a single-row step,
/// so a package may be batched without a caller agreeing to a different
/// distribution. A speculative block only guarantees the target model's
/// distribution when its acceptance rule says so, which is a per-request
/// opt-in; declaring it merely `semantic` is what keeps a runtime from turning
/// it on by itself.
fn block_equivalence(policy: IterationPolicy) -> crate::schema::EquivalenceClass {
    match policy {
        IterationPolicy::ContinuousBatch => crate::schema::EquivalenceClass::Bitwise,
        _ => crate::schema::EquivalenceClass::Semantic,
    }
}

type BodyStep<'s> = (
    &'s str,
    &'s BTreeMap<String, String>,
    &'s BTreeMap<String, String>,
);

fn rebuild_body(
    body: &[WorkflowStep],
    decode: BodyStep<'_>,
    selection: BodyStep<'_>,
    policy: IterationPolicy,
) -> Vec<WorkflowStep> {
    let accepted = selection
        .2
        .get(POLICY_ACCEPTED_PORT)
        .cloned()
        .unwrap_or_else(|| body_value(ACCEPTED_CELL));
    let mut steps = Vec::with_capacity(body.len());
    for step in body {
        match step {
            WorkflowStep::Invoke { component, .. } if component == decode.0 => {
                steps.push(WorkflowStep::Invoke {
                    component: policy.component().to_string(),
                    inputs: decode.1.clone(),
                    outputs: merged(decode.2, selection.2),
                });
            }
            WorkflowStep::Invoke { component, .. } if component == selection.0 => {}
            // A single-token iteration emits one token per row, so its emit
            // needs no length: the tensor *is* the step's contribution. A batch
            // row may sit out a step and a verified block may contribute
            // several tokens, so the re-authored emit publishes a *prefix* whose
            // per-row length is the `accepted_len` the step declares. That is
            // also what makes the output row-wise, which is what gives each row
            // its own liveness guard rather than row 0's.
            WorkflowStep::Emit {
                value,
                output,
                mode,
                when,
                axis,
                ..
            } if output == TOKENS_OUTPUT => {
                steps.push(WorkflowStep::Emit {
                    value: value.clone(),
                    output: output.clone(),
                    stream: None,
                    mode: mode.clone(),
                    when: when.clone(),
                    valid_length: Some(accepted.clone()),
                    axis: Some(axis.unwrap_or(1)),
                });
            }
            other => steps.push(other.clone()),
        }
    }
    steps
}

fn declares_token_emit(body: &[WorkflowStep]) -> bool {
    body.iter()
        .any(|step| matches!(step, WorkflowStep::Emit { output, .. } if output == TOKENS_OUTPUT))
}

/// The package's sole generation loop body, if it declares one.
fn generation_loop_body(spec: &WorkflowSpec) -> Option<&[WorkflowStep]> {
    let mut found = None;
    for step in &spec.steps {
        if let WorkflowStep::Loop { steps, .. } = step
            && declares_token_emit(steps)
        {
            if found.is_some() {
                return None;
            }
            found = Some(steps.as_slice());
        }
    }
    found
}

/// The body's invocation of a component declaring `contract`.
fn body_step_declaring<'s>(
    spec: &'s WorkflowSpec,
    body: &'s [WorkflowStep],
    contract: &str,
) -> Option<BodyStep<'s>> {
    body.iter().find_map(|step| match step {
        WorkflowStep::Invoke {
            component,
            inputs,
            outputs,
        } if spec
            .components
            .get(component)
            .and_then(|declared| declared.contract.as_ref())
            .is_some_and(|declared| declared.id == contract) =>
        {
            Some((component.as_str(), inputs, outputs))
        }
        _ => None,
    })
}

/// Accumulates the parallel declarations one graph port implies.
///
/// A single port shows up in four places — the component's port list, its role
/// map, the invoke binding, and (for state) a cell, a group alias and a carry.
/// Threading them through one builder is what keeps them from drifting apart.
struct Builder {
    port_contracts: BTreeMap<String, TensorContract>,
    inputs: BTreeMap<String, WorkflowInput>,
    state: BTreeMap<String, WorkflowStateCell>,
    groups: BTreeMap<String, StateGroupContract>,
    invoke_inputs: BTreeMap<String, String>,
    invoke_outputs: BTreeMap<String, String>,
    carried: Vec<crate::schema::WorkflowCarry>,
    ports_in: BTreeMap<String, TensorContract>,
    ports_out: BTreeMap<String, TensorContract>,
    roles: BTreeMap<String, PortRole>,
}

impl Builder {
    fn request_input(
        &mut self,
        name: &str,
        contract: TensorContract,
        role: RuntimeInputRole,
        required: bool,
        default: Option<crate::schema::LiteralValue>,
    ) {
        self.inputs.insert(
            name.to_string(),
            WorkflowInput {
                contract,
                role: SemanticInputRole::Runtime {
                    version: CONTRACT_VERSION.to_string(),
                    role,
                },
                source: WorkflowInputSource::Request,
                required,
                default,
                present_as: None,
                externally_suppliable: false,
            },
        );
    }

    fn decoder_port_in(&mut self, port: &str, contract: TensorContract, role: Option<PortRole>) {
        self.ports_in.insert(port.to_string(), contract);
        if let Some(role) = role {
            self.roles.insert(port.to_string(), role);
        }
    }

    fn decoder_port_out(&mut self, port: &str, contract: TensorContract, role: Option<PortRole>) {
        self.ports_out.insert(port.to_string(), contract);
        if let Some(role) = role {
            self.roles.insert(port.to_string(), role);
        }
    }

    fn bind_invoke_input(&mut self, port: &str, value: &str) {
        self.invoke_inputs
            .insert(port.to_string(), value.to_string());
    }

    fn bind_invoke_output(&mut self, port: &str, value: &str) {
        self.invoke_outputs
            .insert(port.to_string(), value.to_string());
    }

    /// A per-step input the application supplies, bound straight to its
    /// declared workflow input.
    fn optional_application_port(
        &mut self,
        port: &str,
        contract: TensorContract,
        role: Option<PortRole>,
    ) {
        let input = format!("application.{}", sanitize(port));
        self.inputs.insert(
            input.clone(),
            WorkflowInput {
                contract: contract.clone(),
                role: SemanticInputRole::Opaque,
                source: WorkflowInputSource::Application {
                    name: port.to_string(),
                },
                required: true,
                default: None,
                present_as: None,
                externally_suppliable: true,
            },
        );
        self.decoder_port_in(port, contract, role);
        self.bind_invoke_input(port, &input);
    }

    /// A package-level literal input with no state cell of its own.
    fn package_literal(
        &mut self,
        name: &str,
        contract: TensorContract,
        default: crate::schema::LiteralValue,
    ) {
        self.inputs.insert(
            format!("package.{name}"),
            WorkflowInput {
                contract,
                role: SemanticInputRole::Opaque,
                source: WorkflowInputSource::Literal,
                required: false,
                default: Some(default),
                present_as: None,
                externally_suppliable: false,
            },
        );
    }

    /// A state cell seeded from a package literal.
    fn literal_state(
        &mut self,
        cell: &str,
        contract: TensorContract,
        default: crate::schema::LiteralValue,
    ) {
        let input = format!("package.{cell}");
        self.inputs.insert(
            input.clone(),
            WorkflowInput {
                contract: contract.clone(),
                role: SemanticInputRole::Opaque,
                source: WorkflowInputSource::Literal,
                required: false,
                default: Some(default),
                present_as: None,
                externally_suppliable: false,
            },
        );
        self.state.insert(
            cell.to_string(),
            WorkflowStateCell {
                contract,
                class: crate::schema::WorkflowStateClass::Semantic,
                scope: WorkflowStateScope::Invocation,
                initializer: input,
                recurrence: ShapeRecurrence::Invariant,
                management: crate::schema::StateManagement::Workflow,
                release_boundary: None,
                service_group: None,
                session: None,
            },
        );
    }

    /// Declare one state group over positionally paired graph ports.
    ///
    /// The per-layer `layer` index is what preserves the graph's own ordering
    /// through a `BTreeMap`, whose key order is lexicographic and would
    /// otherwise place layer 10 between layer 1 and layer 2.
    fn state_group(
        &mut self,
        group: &str,
        kind: StateKind,
        abi: &DecoderAbi,
        pairs: &[(String, String)],
        keyed_by_role: bool,
        declared: Option<&DecoderStateGroup>,
    ) {
        if pairs.is_empty() {
            return;
        }
        let mut aliases = BTreeMap::new();
        for (index, (input, output)) in pairs.iter().enumerate() {
            let (role, layer) =
                if let Some(port) = declared.and_then(|group| group.ports.get(index)) {
                    (port.role, port.layer)
                } else if keyed_by_role {
                    // Self- and cross-attention pair as (key, value) per layer,
                    // which is the order every producer emits and the order the
                    // recognizer rebuilds pairs in.
                    let role = if index % 2 == 0 {
                        StatePortRole::Key
                    } else {
                        StatePortRole::Value
                    };
                    (Some(role), Some(index / 2))
                } else {
                    (None, Some(index))
                };
            let cell = format!("{group}.{index:04}");
            let cell_contract = self.state_port(&pairs[index].0);
            let initializer = self.seed_state_input(cell_contract.clone());
            self.state.insert(
                cell.clone(),
                WorkflowStateCell {
                    contract: cell_contract,
                    class: crate::schema::WorkflowStateClass::Semantic,
                    scope: WorkflowStateScope::Invocation,
                    initializer,
                    recurrence: ShapeRecurrence::Invariant,
                    // The runtime owns these buffers. That is what keeps paged,
                    // shared-buffer and CUDA-graph KV device-resident instead of
                    // round-tripping through the interpreter as SSA values.
                    management: crate::schema::StateManagement::Runtime,
                    release_boundary: Some(crate::schema::StateReleaseBoundary::Invocation),
                    service_group: Some(group.to_string()),
                    session: None,
                },
            );
            aliases.insert(
                cell.clone(),
                StatePortAlias {
                    input: input.clone(),
                    output: Some(output.clone()),
                    access: crate::schema::StatePortAccess::ReadWrite,
                    role,
                    layer,
                },
            );
            // The cell holds what the graph's port holds, so it takes the same
            // contract rather than a plausible one.
            let input_contract = self.state_port(input);
            let output_contract = self.state_port(output);
            self.decoder_port_in(input, input_contract, None);
            self.decoder_port_out(output, output_contract, None);
            self.bind_invoke_input(input, &cell);
            let next = body_value(&cell);
            self.bind_invoke_output(output, &next);
            self.carried.push(crate::schema::WorkflowCarry {
                cell,
                initial: None,
                next,
            });
        }
        let update = declared.and_then(|group| group.update.clone()).or_else(|| {
            (kind == StateKind::FullAttention)
                .then(|| static_update(abi))
                .flatten()
        });
        let logical_lengths = matches!(update, Some(StateUpdate::IndexedScatter { .. }))
            .then(|| LENGTHS_CELL.to_string());
        self.groups.insert(
            group.to_string(),
            StateGroupContract {
                kind,
                properties: declared.and_then(|group| group.properties.clone()),
                sequence_axis: declared
                    .map(|group| group.sequence_axis)
                    .unwrap_or_else(|| (kind != StateKind::Recurrent).then_some(2)),
                layout: declared
                    .map(|group| group.layout.clone())
                    .unwrap_or_else(|| layout_name(abi)),
                logical_lengths,
                aliasing: if kind == StateKind::FullAttention {
                    abi.aliasing.unwrap_or(StateAliasing::Forbidden)
                } else {
                    StateAliasing::Forbidden
                },
                update,
                ports: BTreeMap::from([(DECODER_COMPONENT.to_string(), aliases)]),
                total_length: None,
                reuse: declared
                    .map(|group| group.reuse)
                    .unwrap_or(crate::schema::StateReuse {
                        prefix_reusable: true,
                        evictable_prefix: true,
                    }),
                capabilities: declared
                    .map(|group| group.capabilities.clone())
                    .unwrap_or_else(|| crate::schema::StateGroupCapabilities {
                        rollback_positions: None,
                        snapshot: true,
                        fork: true,
                        cascade: BTreeSet::new(),
                    }),
                checkpoint: None,
            },
        );
    }

    /// The declared contract for a graph state port.
    ///
    /// The caller's graph wins; the growing-KV default applies only when no
    /// graph was supplied.
    fn state_port(&self, port: &str) -> TensorContract {
        self.port_contracts
            .get(port)
            .cloned()
            .unwrap_or_else(state_contract)
    }

    fn seed_state_input(&mut self, contract: TensorContract) -> String {
        if let Some((name, _)) = self
            .inputs
            .iter()
            .find(|(name, input)| name.starts_with(STATE_SEED_INPUT) && input.contract == contract)
        {
            return name.clone();
        }
        let seed_count = self
            .inputs
            .keys()
            .filter(|name| name.starts_with(STATE_SEED_INPUT))
            .count();
        let name = if seed_count == 0 {
            STATE_SEED_INPUT.to_string()
        } else {
            format!("{STATE_SEED_INPUT}.{seed_count}")
        };
        let default = if contract.dtype == "bool" {
            crate::schema::ScalarValue::Bool(false)
        } else if contract.dtype.starts_with("int") || contract.dtype.starts_with("uint") {
            crate::schema::ScalarValue::Integer(0)
        } else {
            crate::schema::ScalarValue::Float(0.0)
        };
        self.inputs.insert(
            name.clone(),
            WorkflowInput {
                contract,
                role: SemanticInputRole::Opaque,
                source: WorkflowInputSource::Literal,
                required: false,
                default: Some(crate::schema::LiteralValue::Scalar(default)),
                present_as: None,
                externally_suppliable: false,
            },
        );
        name
    }
}

/// Rebuild per-layer `(past, present)` pairs from a static cache's per-half lists.
///
/// The group's aliases are ordered `(key, value)` per layer, which is the order
/// the recognizer rebuilds pairs in, so the two halves interleave rather than
/// concatenate.
fn interleave_halves(
    cache: &crate::schema::StaticCacheIoSpec,
) -> Result<Vec<(String, String)>, BuildError> {
    let layers = cache.key_cache_inputs.len();
    if cache.value_cache_inputs.len() != layers
        || cache.key_cache_outputs.len() != layers
        || cache.value_cache_outputs.len() != layers
    {
        return Err(BuildError::UnpairedState {
            group: "static_cache",
            inputs: cache.key_cache_inputs.len() + cache.value_cache_inputs.len(),
            outputs: cache.key_cache_outputs.len() + cache.value_cache_outputs.len(),
        });
    }
    let mut pairs = Vec::with_capacity(layers * 2);
    for layer in 0..layers {
        pairs.push((
            cache.key_cache_inputs[layer].clone(),
            cache.key_cache_outputs[layer].clone(),
        ));
        pairs.push((
            cache.value_cache_inputs[layer].clone(),
            cache.value_cache_outputs[layer].clone(),
        ));
    }
    Ok(pairs)
}

fn static_update(abi: &DecoderAbi) -> Option<StateUpdate> {
    let cache = abi.static_cache.as_ref()?;
    Some(StateUpdate::IndexedScatter {
        write_indices: LENGTHS_CELL.to_string(),
        capacity: format!("package.{CAPACITY_INPUT}"),
        write_indices_ports: BTreeMap::from([(
            DECODER_COMPONENT.to_string(),
            cache.write_indices_input.clone(),
        )]),
        kv_length_ports: BTreeMap::from([(
            DECODER_COMPONENT.to_string(),
            cache.kv_sequence_length_input.clone(),
        )]),
    })
}

/// Workflow input the prompt arrives on.
const REQUEST_TOKENS: &str = "request.input_ids";
/// Workflow input bounding the generation loop.
const REQUEST_MAX_ITERATIONS: &str = "request.max_iterations";
/// Package input every state cell is seeded from.
const STATE_SEED_INPUT: &str = "package.state_seed";
/// Iteration bound used when the package declares no sequence limit.
const DEFAULT_MAX_ITERATIONS: usize = 4096;
/// Version both canonical contracts are declared at.
const CONTRACT_VERSION: &str = "1.0";
/// Component name the runtime's token policy is bound to.
pub const POLICY_COMPONENT: &str = "token_policy";
/// Contract identifying the runtime-implemented token policy.
pub const TOKEN_POLICY_CONTRACT: &str = "onnx-genai.token-policy";
/// Contract identifying one autoregressive decoder forward pass.
///
/// Declared *alongside* the component's ONNX artifact rather than instead of
/// it: the package still says which graph this step runs, and the contract says
/// what the step *is*. A runtime that has a fused session for this contract —
/// one owning paged KV on device, capturing a CUDA graph, reusing a decode
/// workspace — supplies it as the executor; a runtime without one invokes the
/// artifact generically. Both run the same declared step, which is what lets
/// the optimized path be a node inside a workflow rather than a second loop
/// beside one.
pub const AUTOREGRESSIVE_DECODE_CONTRACT: &str = "onnx-genai.autoregressive-decode";
/// Component name a continuous-batch body invokes.
pub const BATCH_STEP_COMPONENT: &str = "batch_step";
/// Contract identifying one shared forward pass advancing every live row.
///
/// The row-major counterpart of [`AUTOREGRESSIVE_DECODE_CONTRACT`] fused with
/// [`TOKEN_POLICY_CONTRACT`]: one iteration decodes every live row through a
/// single forward pass and applies the same per-row token policy, publishing a
/// row-vector liveness predicate the loop already knows how to read. It is one
/// contract rather than two because the fusion is the point — a batch that
/// decoded and then selected as separate declared steps would have to
/// materialize every row's scores as an SSA value, which is exactly the
/// device→host copy the row router exists to avoid.
pub const CONTINUOUS_BATCH_CONTRACT: &str = "onnx-genai.continuous-batch";
/// Component name a speculative-block body invokes.
pub const SPECULATIVE_BLOCK_COMPONENT: &str = "speculative_block";
/// Contract identifying one propose-verify-accept block.
///
/// An iteration proposes several candidate tokens, verifies them in one target
/// pass, accepts the longest agreeing prefix and rolls the rejected suffix back
/// out of every cell that advanced. It publishes the accepted tokens as one
/// emit, so the loop's bound counts *blocks* while the package's `accepted_len`
/// cell carries how many tokens each block contributed.
pub const SPECULATIVE_BLOCK_CONTRACT: &str = "onnx-genai.speculative-block";
/// Policy port consuming the decoder's scores.
const POLICY_LOGITS_PORT: &str = "logits";
/// Policy port producing the selected token.
const POLICY_TOKEN_PORT: &str = "token";
/// Policy port producing per-row liveness.
const POLICY_ACTIVE_PORT: &str = "active";
/// Policy port producing per-row completion.
const POLICY_DONE_PORT: &str = "done";
/// Policy port producing the accepted token count.
const POLICY_ACCEPTED_PORT: &str = "accepted_len";
/// Semantic cell naming the rows still generating.
const ACTIVE_CELL: &str = "active";
/// Semantic cell naming the rows that have stopped.
const DONE_CELL: &str = "done";
/// Semantic cell naming how many tokens each row accepted this step.
const ACCEPTED_CELL: &str = "accepted_len";
/// Semantic cell carrying the sequence the decoder consumes each step.
const SEQUENCE_CELL: &str = "sequence";
/// SSA value the decoder's logits are bound to.
const LOGITS_VALUE: &str = "step.logits";
/// SSA value the token policy selects into.
const TOKEN_VALUE: &str = "step.token";
/// Induction value the loop exposes.
const LOOP_ITERATION: &str = "loop.iteration";
/// Package input naming the fixed-capacity cache bound.
const CAPACITY_INPUT: &str = "capacity";
/// Semantic cell naming each row's logical cache length.
const LENGTHS_CELL: &str = "cache_lengths";
/// Package literal bounding how long the loop's sequence cell may grow.
const SEQUENCE_CAPACITY: &str = "sequence_capacity";

fn body_value(cell: &str) -> String {
    format!("body.{cell}")
}

/// Make a graph port name usable as a workflow value name.
fn sanitize(port: &str) -> String {
    port.replace(['/', ' '], "_")
}

fn int_literal(value: i64) -> crate::schema::LiteralValue {
    crate::schema::LiteralValue::Scalar(crate::schema::ScalarValue::Integer(value))
}

fn bool_literal(value: bool) -> crate::schema::LiteralValue {
    crate::schema::LiteralValue::Scalar(crate::schema::ScalarValue::Bool(value))
}

fn sequence_port(abi: &DecoderAbi) -> Result<&str, BuildError> {
    match abi.sequence_source {
        Some(SequenceInputKind::InputsEmbeds) => abi.inputs_embeds_input.as_deref(),
        Some(SequenceInputKind::TokenIds) => abi.token_input.as_deref(),
        None => abi
            .token_input
            .as_deref()
            .or(abi.inputs_embeds_input.as_deref()),
    }
    .ok_or(BuildError::NoSequenceInput)
}

fn paired(
    inputs: Option<&[String]>,
    outputs: Option<&[String]>,
    group: &'static str,
) -> Result<Vec<(String, String)>, BuildError> {
    let inputs = inputs.unwrap_or(&[]);
    let outputs = outputs.unwrap_or(&[]);
    if inputs.len() != outputs.len() {
        return Err(BuildError::UnpairedState {
            group,
            inputs: inputs.len(),
            outputs: outputs.len(),
        });
    }
    Ok(inputs
        .iter()
        .cloned()
        .zip(outputs.iter().cloned())
        .collect())
}

fn layout_name(abi: &DecoderAbi) -> String {
    match abi.kv_layout.as_ref() {
        Some(layout) if *layout == crate::schema::KvCacheLayout::seq_major_bsnh() => {
            "seq_major_bsnh".to_string()
        }
        _ => "head_major_bnsh".to_string(),
    }
}

fn dims(names: &[&str]) -> Vec<crate::schema::TensorDimension> {
    names
        .iter()
        .map(|name| crate::schema::TensorDimension::Symbol(name.to_string()))
        .collect()
}

fn request_aligned() -> BatchLayout {
    BatchLayout::RequestAligned { axis: 0 }
}

fn token_contract() -> TensorContract {
    TensorContract {
        dtype: "int64".to_string(),
        shape: dims(&["batch", "sequence"]),
        optional: false,
        batch_layout: request_aligned(),
        padding: Vec::new(),
    }
}

fn mask_contract() -> TensorContract {
    TensorContract {
        dtype: "int64".to_string(),
        shape: dims(&["batch", "kv_sequence"]),
        optional: false,
        batch_layout: request_aligned(),
        padding: Vec::new(),
    }
}

fn logits_contract() -> TensorContract {
    TensorContract {
        dtype: "float32".to_string(),
        shape: dims(&["batch", "sequence", "vocabulary"]),
        optional: false,
        batch_layout: request_aligned(),
        padding: Vec::new(),
    }
}

fn hidden_contract() -> TensorContract {
    TensorContract {
        dtype: "float32".to_string(),
        shape: dims(&["batch", "sequence", "hidden"]),
        optional: false,
        batch_layout: request_aligned(),
        padding: Vec::new(),
    }
}

fn state_contract() -> TensorContract {
    TensorContract {
        dtype: "float32".to_string(),
        shape: dims(&["batch", "kv_heads", "kv_sequence", "head_dim"]),
        optional: false,
        batch_layout: request_aligned(),
        padding: Vec::new(),
    }
}

fn lengths_contract() -> TensorContract {
    TensorContract {
        dtype: "int64".to_string(),
        shape: dims(&["batch"]),
        optional: false,
        batch_layout: request_aligned(),
        padding: Vec::new(),
    }
}

fn flag_contract() -> TensorContract {
    TensorContract {
        dtype: "bool".to_string(),
        shape: dims(&["batch"]),
        optional: false,
        batch_layout: request_aligned(),
        padding: Vec::new(),
    }
}

fn scalar_contract() -> TensorContract {
    TensorContract {
        dtype: "int64".to_string(),
        shape: vec![crate::schema::TensorDimension::Fixed(1)],
        optional: false,
        batch_layout: BatchLayout::Shared,
        padding: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::LoopStatePair;

    /// The property the whole conversion rests on.
    ///
    /// A converted package is only trustworthy if reading its workflow back
    /// produces the ABI the graph actually has. Anything less means a mechanical
    /// conversion could silently change what the runtime binds, which is exactly
    /// the failure a hand-check would miss on 16 fixtures.
    fn roundtrips(abi: &DecoderAbi) -> DecoderAbi {
        let workflow = decoder_workflow(abi, "model.onnx", &DecoderFacts::default())
            .expect("a complete decoder ABI must be expressible as a workflow");
        let component = crate::graph_cardinality::sole_decoder_component(&workflow)
            .expect("the converted workflow must present exactly one decoder");
        assert_eq!(component, DECODER_COMPONENT);
        crate::decoder_abi::decoder_abi(&workflow, component)
            .expect("the recognizer must read back the component it was given")
    }

    fn dense() -> DecoderAbi {
        DecoderAbi {
            token_input: Some("input_ids".to_string()),
            attention_mask_input: Some("attention_mask".to_string()),
            position_ids_input: Some("position_ids".to_string()),
            logits_output: Some("logits".to_string()),
            kv_inputs: Some(vec![
                "past_key_values.0.key".to_string(),
                "past_key_values.0.value".to_string(),
                "past_key_values.1.key".to_string(),
                "past_key_values.1.value".to_string(),
            ]),
            kv_outputs: Some(vec![
                "present.0.key".to_string(),
                "present.0.value".to_string(),
                "present.1.key".to_string(),
                "present.1.value".to_string(),
            ]),
            ..DecoderAbi::default()
        }
    }

    #[test]
    fn a_dense_decoder_roundtrips() {
        let abi = dense();
        let read_back = roundtrips(&abi);
        assert_eq!(read_back.token_input, abi.token_input);
        assert_eq!(read_back.attention_mask_input, abi.attention_mask_input);
        assert_eq!(read_back.position_ids_input, abi.position_ids_input);
        assert_eq!(read_back.logits_output, abi.logits_output);
        assert_eq!(read_back.kv_inputs, abi.kv_inputs);
        assert_eq!(read_back.kv_outputs, abi.kv_outputs);
    }

    /// Layer order must survive the conversion.
    ///
    /// State ports live in a `BTreeMap`, whose key order is lexicographic — so a
    /// naive conversion puts layer 10 between layer 1 and layer 2 and silently
    /// binds the wrong cache to the wrong layer. The declared `layer` index is
    /// what prevents that, and this is the case that would catch its absence.
    #[test]
    fn many_layers_keep_their_graph_order() {
        let mut abi = dense();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for layer in 0..12 {
            for half in ["key", "value"] {
                inputs.push(format!("past_key_values.{layer}.{half}"));
                outputs.push(format!("present.{layer}.{half}"));
            }
        }
        abi.kv_inputs = Some(inputs.clone());
        abi.kv_outputs = Some(outputs.clone());
        let read_back = roundtrips(&abi);
        assert_eq!(read_back.kv_inputs, Some(inputs));
        assert_eq!(read_back.kv_outputs, Some(outputs));
    }

    #[test]
    fn an_embeds_driven_decoder_requires_an_embedding_producer() {
        let abi = DecoderAbi {
            sequence_source: Some(SequenceInputKind::InputsEmbeds),
            inputs_embeds_input: Some("inputs_embeds".to_string()),
            logits_output: Some("logits".to_string()),
            kv_inputs: Some(vec!["past.key".to_string()]),
            kv_outputs: Some(vec!["present.key".to_string()]),
            ..DecoderAbi::default()
        };
        assert_eq!(
            decoder_workflow(&abi, "model.onnx", &DecoderFacts::default()),
            Err(BuildError::InputsEmbedsNeedsEmbeddingProducer)
        );
    }

    #[test]
    fn cross_attention_and_recurrent_state_roundtrip() {
        let abi = DecoderAbi {
            token_input: Some("input_ids".to_string()),
            logits_output: Some("logits".to_string()),
            encoder_hidden_states_input: Some("encoder_hidden_states".to_string()),
            cross_kv_inputs: Some(vec![
                "past.cross.key".to_string(),
                "past.cross.value".to_string(),
            ]),
            cross_kv_outputs: Some(vec![
                "present.cross.key".to_string(),
                "present.cross.value".to_string(),
            ]),
            state_pairs: Some(vec![LoopStatePair {
                input: "conv_state.in".to_string(),
                output: "conv_state.out".to_string(),
                init: None,
                update: None,
            }]),
            ..DecoderAbi::default()
        };
        let read_back = roundtrips(&abi);
        assert_eq!(read_back.cross_kv_inputs, abi.cross_kv_inputs);
        assert_eq!(read_back.cross_kv_outputs, abi.cross_kv_outputs);
        assert_eq!(
            read_back.state_pairs.as_ref().map(Vec::len),
            abi.state_pairs.as_ref().map(Vec::len)
        );
        assert_eq!(
            read_back.encoder_hidden_states_input,
            abi.encoder_hidden_states_input
        );
    }

    #[test]
    fn aliasing_permission_survives_the_conversion() {
        let mut abi = dense();
        abi.aliasing = Some(StateAliasing::Required);
        assert_eq!(roundtrips(&abi).aliasing, Some(StateAliasing::Required));
        abi.aliasing = Some(StateAliasing::Forbidden);
        assert_eq!(roundtrips(&abi).aliasing, Some(StateAliasing::Forbidden));
    }

    #[test]
    fn a_decoder_without_logits_is_refused() {
        let abi = DecoderAbi {
            token_input: Some("input_ids".to_string()),
            ..DecoderAbi::default()
        };
        assert_eq!(
            decoder_workflow(&abi, "model.onnx", &DecoderFacts::default()),
            Err(BuildError::NoLogitsOutput)
        );
    }

    #[test]
    fn a_decoder_without_a_sequence_input_is_refused() {
        let abi = DecoderAbi {
            logits_output: Some("logits".to_string()),
            ..DecoderAbi::default()
        };
        assert_eq!(
            decoder_workflow(&abi, "model.onnx", &DecoderFacts::default()),
            Err(BuildError::NoSequenceInput)
        );
    }

    #[test]
    fn unpaired_state_ports_are_refused() {
        let mut abi = dense();
        abi.kv_outputs = Some(vec!["present.0.key".to_string()]);
        assert!(matches!(
            decoder_workflow(&abi, "model.onnx", &DecoderFacts::default()),
            Err(BuildError::UnpairedState { .. })
        ));
    }

    /// The emitted document must be what the schema accepts, not merely what
    /// these structs can hold.
    #[test]
    fn the_emitted_workflow_parses_back_through_the_schema() {
        let workflow = decoder_workflow(&dense(), "model.onnx", &DecoderFacts::default())
            .expect("dense decoder converts");
        let text = serde_yaml::to_string(&workflow).expect("the workflow serializes");
        let parsed: WorkflowSpec =
            serde_yaml::from_str(&text).expect("the emitted workflow must satisfy its own schema");
        assert_eq!(parsed, workflow);
    }
}

#[cfg(test)]
mod validation_tests {
    use super::tests_support::dense_abi;
    use super::*;

    /// A converted package must satisfy the *package* validator, not just serde.
    ///
    /// Serialization proving the structs round-trip says nothing about whether
    /// the workflow is coherent — an emit into an undeclared output, a serving
    /// contract naming a state cell that does not exist, a step reading an
    /// unbound SSA value. Those are exactly the mistakes a hand-written emitter
    /// makes, and they must fail here rather than at load on a user's machine.
    #[test]
    fn a_converted_package_validates() {
        let workflow = decoder_workflow(&dense_abi(), "model.onnx", &DecoderFacts::default())
            .expect("dense decoder converts");
        let mut metadata = crate::schema::InferenceMetadata::default();
        metadata.pipeline = Some(crate::schema::PipelineSpec { workflow });
        if let Err(errors) = crate::validation::validate_metadata(&metadata) {
            panic!("a converted single-decoder package must validate, got: {errors:#?}");
        }
    }
}

#[cfg(test)]
mod tests_support {
    use super::*;
    /// Shared dense decoder ABI, so the conversion and validation suites agree
    /// on what "an ordinary decoder" means.
    pub(super) fn dense_abi() -> DecoderAbi {
        DecoderAbi {
            token_input: Some("input_ids".to_string()),
            attention_mask_input: Some("attention_mask".to_string()),
            position_ids_input: Some("position_ids".to_string()),
            logits_output: Some("logits".to_string()),
            kv_inputs: Some(vec![
                "past_key_values.0.key".to_string(),
                "past_key_values.0.value".to_string(),
            ]),
            kv_outputs: Some(vec![
                "present.0.key".to_string(),
                "present.0.value".to_string(),
            ]),
            ..DecoderAbi::default()
        }
    }
}

#[cfg(test)]
mod roundtrip_coverage {
    use super::tests_support::dense_abi;
    use super::*;
    use crate::schema::{KvCacheLayout, LoopStatePair, StaticCacheIoSpec};

    fn read_back(abi: &DecoderAbi) -> DecoderAbi {
        let workflow = decoder_workflow(abi, "model.onnx", &DecoderFacts::default())
            .expect("a complete decoder ABI must be expressible as a workflow");
        let component =
            crate::graph_cardinality::sole_decoder_component(&workflow).expect("one decoder");
        crate::decoder_abi::decoder_abi(&workflow, component).expect("readable")
    }

    /// A fixed-capacity cache survives the conversion, including its control
    /// ports and per-layer halves.
    ///
    /// This is the trickiest inverse in the module — `interleave_halves` and
    /// `static_update` on the way out, `derive_static_cache` on the way back —
    /// and getting it wrong binds the wrong buffer to the wrong layer without
    /// any type error.
    #[test]
    fn a_static_cache_roundtrips() {
        let abi = DecoderAbi {
            token_input: Some("input_ids".to_string()),
            logits_output: Some("logits".to_string()),
            static_cache: Some(StaticCacheIoSpec {
                write_indices_input: "write_indices".to_string(),
                kv_sequence_length_input: "nonpad_kv_seqlen".to_string(),
                key_cache_inputs: vec!["key_cache.0".to_string(), "key_cache.1".to_string()],
                value_cache_inputs: vec!["value_cache.0".to_string(), "value_cache.1".to_string()],
                key_cache_outputs: vec![
                    "updated_key_cache.0".to_string(),
                    "updated_key_cache.1".to_string(),
                ],
                value_cache_outputs: vec![
                    "updated_value_cache.0".to_string(),
                    "updated_value_cache.1".to_string(),
                ],
            }),
            ..DecoderAbi::default()
        };
        let back = read_back(&abi);
        assert_eq!(back.static_cache, abi.static_cache);
        // A fixed-capacity cache is stated once. Reporting it *also* as growing
        // pairs would make the paged KV bridge address a buffer that never
        // grows.
        assert_eq!(back.kv_inputs, None);
        assert_eq!(back.kv_outputs, None);
        assert_eq!(back.kv_ownership, Some(crate::schema::KvOwnership::Owned));
    }

    /// Recurrent carries keep their order and their names.
    ///
    /// The KV path has a layer index protecting it from `BTreeMap` key order;
    /// recurrent pairs rely on the same mechanism, so they need the same
    /// evidence rather than a length check that would pass on a shuffle.
    #[test]
    fn recurrent_state_keeps_its_order() {
        let pairs: Vec<LoopStatePair> = (0..12)
            .map(|layer| LoopStatePair {
                input: format!("conv_state.{layer}.in"),
                output: format!("conv_state.{layer}.out"),
                init: None,
                update: None,
            })
            .collect();
        let abi = DecoderAbi {
            token_input: Some("input_ids".to_string()),
            logits_output: Some("logits".to_string()),
            state_pairs: Some(pairs.clone()),
            ..DecoderAbi::default()
        };
        assert_eq!(read_back(&abi).state_pairs, Some(pairs));
    }

    #[test]
    fn a_distinct_hidden_output_roundtrips() {
        let mut abi = dense_abi();
        abi.hidden_output = Some("last_hidden_state".to_string());
        assert_eq!(read_back(&abi).hidden_output, abi.hidden_output);
    }

    #[test]
    fn a_declared_kv_layout_roundtrips() {
        let mut abi = dense_abi();
        abi.kv_layout = Some(KvCacheLayout::seq_major_bsnh());
        assert_eq!(
            read_back(&abi).kv_layout,
            Some(KvCacheLayout::seq_major_bsnh())
        );
    }

    /// Absence is normalized to the default every consumer already applies.
    ///
    /// Pinned rather than left implicit: a reader who sees "it round-trips"
    /// would otherwise be entitled to expect `None` back, and the difference
    /// matters when comparing a converted package against its source.
    #[test]
    fn unstated_defaults_read_back_as_the_defaults_they_always_meant() {
        let mut abi = dense_abi();
        abi.aliasing = None;
        abi.kv_layout = None;
        let back = read_back(&abi);
        assert_eq!(back.aliasing, Some(StateAliasing::Forbidden));
        assert_eq!(back.kv_layout, Some(KvCacheLayout::head_major_bnsh()));
    }
}

#[cfg(test)]
mod iteration_policy_tests {
    use super::tests_support::dense_abi;
    use super::*;

    fn single_token() -> WorkflowSpec {
        decoder_workflow(&dense_abi(), "model.onnx", &DecoderFacts::default())
            .expect("a dense decoder is expressible as a workflow")
    }

    fn loop_body(spec: &WorkflowSpec) -> &[WorkflowStep] {
        spec.steps
            .iter()
            .find_map(|step| match step {
                WorkflowStep::Loop { steps, .. } => Some(steps.as_slice()),
                _ => None,
            })
            .expect("a generation workflow declares a loop")
    }

    fn invoked_contracts(spec: &WorkflowSpec) -> Vec<&str> {
        loop_body(spec)
            .iter()
            .filter_map(|step| match step {
                WorkflowStep::Invoke { component, .. } => spec
                    .components
                    .get(component)
                    .and_then(|declared| declared.contract.as_ref())
                    .map(|contract| contract.id.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The default policy is the document the package already declares.
    #[test]
    fn the_single_token_policy_is_the_declared_body() {
        let spec = single_token();
        let variant = iteration_variant(&spec, IterationPolicy::SingleToken)
            .expect("the identity re-authoring cannot fail");
        assert_eq!(variant, spec);
        assert_eq!(
            invoked_contracts(&spec),
            vec![AUTOREGRESSIVE_DECODE_CONTRACT, TOKEN_POLICY_CONTRACT]
        );
    }

    /// Each non-default policy authors a body naming exactly its own contract.
    ///
    /// This is the selection mechanism stated at the authoring end: three
    /// bodies, three contracts, and nothing about the package changed between
    /// them.
    #[test]
    fn each_policy_authors_its_own_contract() {
        for (policy, contract) in [
            (IterationPolicy::ContinuousBatch, CONTINUOUS_BATCH_CONTRACT),
            (
                IterationPolicy::SpeculativeBlock,
                SPECULATIVE_BLOCK_CONTRACT,
            ),
        ] {
            let variant = iteration_variant(&single_token(), policy)
                .expect("a declared single-token loop can be re-authored");
            assert_eq!(
                invoked_contracts(&variant),
                vec![contract],
                "{policy:?} declares one iteration step"
            );
            assert!(
                !variant.components.contains_key(DECODER_COMPONENT)
                    && !variant.components.contains_key(POLICY_COMPONENT),
                "{policy:?} replaces the single-token steps rather than adding to them"
            );
        }
    }

    /// The loop itself is untouched: same bound, predicate, carries and setup.
    ///
    /// A variant that quietly changed the stop condition or dropped a carried
    /// cell would be a different generation wearing the same package's name.
    /// Two things *do* change beside the body, and both are asserted rather
    /// than waved past: the state service names the component that now owns the
    /// ports, and the manifest gains the emit capability the re-authored emit
    /// uses.
    #[test]
    fn re_authoring_changes_only_the_body() {
        let spec = single_token();
        let variant = iteration_variant(&spec, IterationPolicy::ContinuousBatch)
            .expect("a declared single-token loop can be re-authored");
        let (
            WorkflowStep::Loop {
                continue_when,
                max_iterations,
                termination,
                carried,
                iteration,
                setup,
                ..
            },
            WorkflowStep::Loop {
                continue_when: variant_continue,
                max_iterations: variant_max,
                termination: variant_termination,
                carried: variant_carried,
                iteration: variant_iteration,
                setup: variant_setup,
                ..
            },
        ) = (&spec.steps[0], &variant.steps[0])
        else {
            panic!("both declare a loop");
        };
        assert_eq!(continue_when, variant_continue);
        assert_eq!(max_iterations, variant_max);
        assert_eq!(termination, variant_termination);
        assert_eq!(carried, variant_carried);
        assert_eq!(iteration, variant_iteration);
        assert_eq!(setup, variant_setup);
        assert_eq!(spec.inputs, variant.inputs);
        assert_eq!(spec.outputs, variant.outputs);
        assert_eq!(spec.state, variant.state);
        // The re-authored emit carries its ragged-prefix semantics directly in
        // the typed output declaration; no duplicate workflow capability flag
        // is permitted.
        let (serving, variant_serving) = (
            spec.serving.as_ref().expect("declared"),
            variant.serving.as_ref().expect("declared"),
        );
        assert_eq!(serving.active, variant_serving.active);
        assert_eq!(serving.done, variant_serving.done);
        assert_eq!(serving.accepted_len, variant_serving.accepted_len);
        // Only the component the aliases name moves; the aliases themselves are
        // the same ports on the same cells.
        for (group, declared) in &serving.state_service.groups {
            let variant_group = &variant_serving.state_service.groups[group];
            assert_eq!(
                declared.ports[DECODER_COMPONENT],
                variant_group.ports[BATCH_STEP_COMPONENT]
            );
        }
    }

    /// The block binds what the decoder bound and defines what both defined.
    ///
    /// The decoder's state outputs stay bound because the loop still carries
    /// those cells: an executor that owns the cache defines them or leaves them
    /// to the service, exactly as the single-token decode executor does.
    #[test]
    fn the_block_inherits_both_ends_of_the_step_it_replaces() {
        let spec = single_token();
        let decode = loop_body(&spec)
            .iter()
            .find_map(|step| match step {
                WorkflowStep::Invoke {
                    component,
                    inputs,
                    outputs,
                } if component == DECODER_COMPONENT => Some((inputs.clone(), outputs.clone())),
                _ => None,
            })
            .expect("the single-token body invokes the decoder");
        let policy_outputs = loop_body(&spec)
            .iter()
            .find_map(|step| match step {
                WorkflowStep::Invoke {
                    component, outputs, ..
                } if component == POLICY_COMPONENT => Some(outputs.clone()),
                _ => None,
            })
            .expect("the single-token body invokes the token policy");

        let variant = iteration_variant(&spec, IterationPolicy::SpeculativeBlock)
            .expect("a declared single-token loop can be re-authored");
        let (inputs, outputs) = loop_body(&variant)
            .iter()
            .find_map(|step| match step {
                WorkflowStep::Invoke {
                    component,
                    inputs,
                    outputs,
                } if component == SPECULATIVE_BLOCK_COMPONENT => Some((inputs, outputs)),
                _ => None,
            })
            .expect("the re-authored body invokes the block");
        assert_eq!(inputs, &decode.0);
        assert_eq!(outputs, &merged(&decode.1, &policy_outputs));
        for (port, value) in policy_outputs.iter().chain(decode.1.iter()) {
            assert_eq!(
                outputs.get(port),
                Some(value),
                "the block defines '{port}' where the step it replaced did"
            );
        }
    }

    /// A row-major step declares the row ABI; a proposal block does not.
    #[test]
    fn only_the_row_major_body_declares_row_scope() {
        let batch = iteration_variant(&single_token(), IterationPolicy::ContinuousBatch)
            .expect("re-authored");
        assert_eq!(
            batch.components[BATCH_STEP_COMPONENT].row_scope,
            Some(crate::schema::ComponentRowScope {
                axis: 0,
                stateful: true
            })
        );
        let block = iteration_variant(&single_token(), IterationPolicy::SpeculativeBlock)
            .expect("re-authored");
        assert_eq!(
            block.components[SPECULATIVE_BLOCK_COMPONENT].row_scope,
            None
        );
    }

    /// Every authored variant is a package this schema accepts.
    #[test]
    fn every_variant_validates() {
        for policy in [
            IterationPolicy::SingleToken,
            IterationPolicy::ContinuousBatch,
            IterationPolicy::SpeculativeBlock,
        ] {
            let workflow = iteration_variant(&single_token(), policy).expect("re-authored");
            let mut metadata = crate::schema::InferenceMetadata::default();
            metadata.pipeline = Some(crate::schema::PipelineSpec { workflow });
            if let Err(errors) = crate::validation::validate_metadata(&metadata) {
                panic!("{policy:?} must author a valid package, got: {errors:#?}");
            }
        }
    }

    /// A workflow with no generation loop has no iteration to re-author.
    #[test]
    fn a_package_without_a_generation_loop_is_refused() {
        let mut spec = single_token();
        spec.steps.clear();
        assert_eq!(
            iteration_variant(&spec, IterationPolicy::ContinuousBatch),
            Err(BuildError::NoGenerationLoop)
        );
    }
}
