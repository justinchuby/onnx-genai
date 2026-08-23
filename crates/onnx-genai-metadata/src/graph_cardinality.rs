//! One structural reading of what a workflow's declared components execute.
//!
//! Two questions used to be answered by two hand-rolled scans of the same
//! document, in two crates:
//!
//! * metadata, the CLI and the server asked "is this package a single
//!   decoder?" from the component's declared [`PortRole`]s, and
//! * the engine loader asked "can the fused decode core stand in for this
//!   package's whole graph?" from the component's declared contract id.
//!
//! Both scans opened with the *same* structural step — filter out the
//! components the runtime implements, and check whether exactly one ONNX graph
//! is left — and then diverged on how they recognized the survivor as a
//! decoder. Nothing named that shared step, so each site rewrote it, and three
//! doc comments had to explain in prose why the two predicates were "not the
//! same predicate" and yet always agreed in practice.
//!
//! This module states the structural step once and layers the two recognizers
//! on top of it:
//!
//! | Layer | Question | Evidence |
//! |---|---|---|
//! | 1 — [`WorkflowClassification::is_single_decoder`] | does this workflow execute exactly one ONNX graph, and is that graph a decoder? | declared [`PortRole`]s |
//! | 2 — [`WorkflowClassification::contracted_single_decoder`] | …*and* does that graph name the step an executor is registered for? | layer 1 **and** the [`AUTOREGRESSIVE_DECODE_CONTRACT`] |
//!
//! Layer 2 is defined as layer 1 plus the contract, so "the loader routed a
//! package the metadata layer does not call a single decoder" is not a state
//! this code can reach — it is not a property two independent scans happen to
//! share, it is how the second layer is built.
//! `decoder_recognizer_agreement.rs` pins that over every workflow-declaring
//! document in the repository and the adversarial shapes none of them cover,
//! and `engine/load.rs` asserts it again from the loader's own predicate.
//!
//! Nothing here reads a component name, a model name or an architecture
//! string. A package may name its components whatever it likes.
//!
//! # This is not the package-shape dispatch `shape_dispatch_gate` bans
//!
//! That gate exists because *which generation loop runs* must be selected by
//! what the workflow authors, never by Rust inspecting the package. Nothing
//! here selects a loop: one interpreter executes every declared workflow. What
//! layer 2 selects is an **executor for one declared step** — the same kind of
//! choice as ORT versus native — and layer 1 answers a question about the
//! serialized document, for `validate_metadata --shape` and for "does this
//! package need multimodal input specs". Both callers predate this module; it
//! removes their duplicate scans rather than adding a branch. The retired
//! enum's name is deliberately not reused.

use crate::decoder_workflow::AUTOREGRESSIVE_DECODE_CONTRACT;
use crate::schema::{ComponentImplementation, PortRole, WorkflowComponent, WorkflowSpec};

/// How many ONNX graphs the workflow's declared components name.
///
/// A component the *runtime* implements (`binding`, such as the token policy)
/// is a declared step with no graph behind it, so it does not count. An
/// `adapter` does: it is an artifact something has to execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphCardinality {
    /// Every declared component is runtime-implemented; there is no graph.
    NoGraph,
    /// Exactly one component names a graph.
    SingleGraph,
    /// Several components name graphs, so no one executor covers the package.
    Composite,
}

/// Why the sole graph component is taken for the package's decoder.
///
/// The two fields are the two layers, recorded separately so a caller can see
/// *which* evidence it is relying on rather than inferring it from a `bool`.
/// They are only ever populated for a [`GraphCardinality::SingleGraph`] workflow:
/// a composite package has a decoder among its components, but the package is
/// not one, and reporting evidence for it invites exactly the misreading that
/// routes a 186-graph package to a single-graph executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecoderEvidence {
    /// The component's declared port roles make it a decoder: it consumes the
    /// autoregressive sequence and either produces logits or owns attention
    /// state.
    pub declared_roles: bool,
    /// The component declares [`AUTOREGRESSIVE_DECODE_CONTRACT`], which is how
    /// a workflow names the step this runtime registered an executor for.
    ///
    /// Recorded independently of `declared_roles` so the contradictory
    /// declaration — a contract with no roles to drive it — is visible rather
    /// than folded away; see [`DecoderEvidence::contradictory`].
    pub declared_decode_contract: bool,
}

impl DecoderEvidence {
    /// Whether the component names the decode step without declaring the roles
    /// that describe how to drive it.
    ///
    /// A producer error, not a package kind: the resolved [`crate::DecoderAbi`]
    /// comes from the roles, so a contract with none behind it promises a step
    /// nothing can execute. Layer 2 declines such a component rather than
    /// handing it to the fused executor to fail on an empty ABI.
    pub fn contradictory(self) -> bool {
        self.declared_decode_contract && !self.declared_roles
    }
}

/// The one structural reading of a workflow, resolved in a single pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkflowClassification<'a> {
    cardinality: GraphCardinality,
    graph_component_count: usize,
    sole_graph_component: Option<&'a str>,
    decoder_component: Option<&'a str>,
    decoder_evidence: DecoderEvidence,
}

impl<'a> WorkflowClassification<'a> {
    /// How many ONNX graphs this workflow names.
    pub fn cardinality(&self) -> GraphCardinality {
        self.cardinality
    }

    /// The number of components that name a graph.
    ///
    /// A diagnostic count. It is what a report prints when it says a package
    /// declares 187 components; nothing routes on it.
    pub fn graph_component_count(&self) -> usize {
        self.graph_component_count
    }

    /// The only component naming a graph, when there is exactly one.
    pub fn sole_graph_component(&self) -> Option<&'a str> {
        self.sole_graph_component
    }

    /// The component this package decodes with, if the workflow has exactly
    /// one such component.
    ///
    /// Answered for composite packages too, which is the whole reason it
    /// exists: a vision-language package has one decoder among its encoder,
    /// projector and decoder components, and the resolver has to say which. A
    /// workflow with several (speculative decoding, an encoder-decoder pair)
    /// has no single answer and yields `None`; those paths address components
    /// explicitly.
    ///
    /// This is deliberately *not* "the package is a decoder". See
    /// [`Self::is_single_decoder`].
    pub fn decoder_component(&self) -> Option<&'a str> {
        self.decoder_component
    }

    /// What makes [`Self::sole_graph_component`] a decoder, layer by layer.
    pub fn decoder_evidence(&self) -> DecoderEvidence {
        self.decoder_evidence
    }

    /// Layer 1 — whether this workflow executes exactly one ONNX graph and
    /// that graph is a decoder.
    ///
    /// This is narrower than "has a recognizable decoder", and the difference
    /// is not academic: a published 187-component any-to-any package has
    /// exactly one component carrying `token_ids`/`logits` roles — its text
    /// head — so the looser reading called it a bare decoder and would have
    /// routed 186 other graphs to an executor that cannot run them. Every
    /// vision-language package classifies the same way.
    pub fn is_single_decoder(&self) -> bool {
        self.decoder_evidence.declared_roles
    }

    /// Layer 2 — the single decoder that also names the decode step, if this
    /// package is one.
    ///
    /// Layer 1 **and** the component declares [`AUTOREGRESSIVE_DECODE_CONTRACT`].
    /// A runtime holding an executor registered for that contract can stand in
    /// for the package's whole graph, because the package's whole graph is that
    /// one step.
    ///
    /// Requiring layer 1 is not an extra gate bolted on to be safe. The fused
    /// executor is driven by the resolved [`crate::DecoderAbi`], which is
    /// derived from the declared roles and from nothing else, so a component
    /// with a contract and no roles has no ABI to drive it. Declining it here
    /// refuses the choice instead of making it and failing on an empty ABI
    /// later, and it is what makes layer 2 a subset of layer 1 by construction
    /// rather than by coincidence.
    pub fn contracted_single_decoder(&self) -> Option<&'a str> {
        let evidence = self.decoder_evidence;
        if evidence.declared_roles && evidence.declared_decode_contract {
            self.sole_graph_component
        } else {
            None
        }
    }
}

/// The components that name something to execute, in the workflow's own order.
///
/// The one place a *cardinality* question about graph components is answered:
/// a `binding` is a step the runtime implements and has no artifact, so it does
/// not count; everything else does, including an `adapter`.
///
/// This is not the only filter over `implementation` in the crate, and
/// deliberately so. `validation.rs` counts `Onnx`-only components when deciding
/// whether a single graph must declare a sequence role, because that check is
/// about which components a decode ABI could be resolved *from* — an adapter is
/// something to execute but never a decoder. Two questions, two filters; what
/// this module owns is "how many graphs, and is the lone one a decoder".
///
/// Callers outside this crate read the resolved [`WorkflowClassification`]
/// rather than re-scanning, which is the duplication this module removes.
pub(crate) fn graph_components(
    workflow: &WorkflowSpec,
) -> impl Iterator<Item = (&str, &WorkflowComponent)> {
    workflow
        .components
        .iter()
        .filter(|(_, component)| {
            !matches!(component.implementation, ComponentImplementation::Binding)
        })
        .map(|(name, component)| (name.as_str(), component))
}

/// Read a workflow's executable shape and its decoder evidence in one pass.
pub fn classify_workflow(workflow: &WorkflowSpec) -> WorkflowClassification<'_> {
    let mut graph_component_count = 0usize;
    let mut first_graph_component = None;
    let mut decoder_component = None;
    let mut decoder_candidates = 0usize;

    for (name, component) in graph_components(workflow) {
        graph_component_count += 1;
        if graph_component_count == 1 {
            first_graph_component = Some(name);
        }
        if is_decoder_by_roles(workflow, name, component) {
            decoder_candidates += 1;
            decoder_component = Some(name);
        }
    }

    let cardinality = match graph_component_count {
        0 => GraphCardinality::NoGraph,
        1 => GraphCardinality::SingleGraph,
        _ => GraphCardinality::Composite,
    };
    let sole_graph_component = matches!(cardinality, GraphCardinality::SingleGraph)
        .then_some(first_graph_component)
        .flatten();
    // Several recognizable decoders is not a decoder: a speculative pair and an
    // encoder-decoder both land here, and both address their components by name.
    let decoder_component = (decoder_candidates == 1)
        .then_some(decoder_component)
        .flatten();

    let decoder_evidence = match sole_graph_component {
        Some(sole) => DecoderEvidence {
            declared_roles: decoder_component == Some(sole),
            declared_decode_contract: workflow.components[sole]
                .contract
                .as_ref()
                .is_some_and(|contract| contract.id == AUTOREGRESSIVE_DECODE_CONTRACT),
        },
        None => DecoderEvidence::default(),
    };

    WorkflowClassification {
        cardinality,
        graph_component_count,
        sole_graph_component,
        decoder_component,
        decoder_evidence,
    }
}

/// Whether `component`'s declared roles make it the autoregressive decoder.
///
/// Structural, never nominal: the decoder is the graph that consumes the
/// autoregressive sequence — a declared `token_ids` or `inputs_embeds` input
/// role — and either produces logits or owns attention state. A graph that
/// consumes a sequence and does neither (a text encoder, a CTC head) is not
/// decoding it.
fn is_decoder_by_roles(workflow: &WorkflowSpec, name: &str, component: &WorkflowComponent) -> bool {
    let roles = &component.ports.roles;
    let consumes_sequence = roles
        .values()
        .any(|role| matches!(role, PortRole::TokenIds | PortRole::InputsEmbeds));
    if !consumes_sequence {
        return false;
    }
    let produces_logits = roles.values().any(|role| *role == PortRole::Logits);
    produces_logits
        || crate::decoder_abi::groups_for(workflow, name)
            .iter()
            .any(|(_, group)| crate::decoder_abi::is_self_attention(group.kind))
}

/// The component a bare decoder package executes, if the workflow has exactly one.
///
/// See [`WorkflowClassification::decoder_component`], which this delegates to.
pub fn sole_decoder_component(workflow: &WorkflowSpec) -> Option<&str> {
    classify_workflow(workflow).decoder_component()
}

/// Whether a workflow is a single-decoder package.
///
/// See [`WorkflowClassification::is_single_decoder`], which this delegates to.
pub fn is_single_decoder_workflow(workflow: &WorkflowSpec) -> bool {
    classify_workflow(workflow).is_single_decoder()
}
