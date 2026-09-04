//! Recognizer that derives a decode-step graph ABI from the canonical workflow.
//!
//! The workflow is the only serialized expression of a package's executable
//! graph ABI. Components declare their ports, invocations bind SSA values to
//! those ports, and the state service declares which port pairs carry model
//! state and how the graph writes into them. Everything a single-decoder fast
//! path needs is already there.
//!
//! This module reads that one representation and produces the resolved
//! [`DecoderAbi`] the optimized decode path consumes. It is a *lowering*: the
//! result is derived, never serialized, and never a second place a producer can
//! state a different answer. There is no other source — a package states its
//! graph ABI in its workflow or it does not load.
//!
//! What is derived, and from where:
//!
//! | Resolved fact | Canonical source |
//! |---|---|
//! | token / embeds / mask / position inputs | `components.<c>.ports.roles` |
//! | logits / hidden outputs | `components.<c>.ports.roles` |
//! | encoder-hidden / audio inputs | `components.<c>.ports.roles` |
//! | KV input/output pairs | self-attention `state_service` groups |
//! | cross-attention KV pairs | `cross_attention` groups |
//! | fixed loop-carried state pairs | `recurrent` groups |
//! | aliasing, layout | the owning group |
//! | fixed-capacity scatter ABI | `StateUpdate::IndexedScatter` |
//! | KV ownership | presence of an owning group |
//! | optional inputs | `TensorContract::optional` |

use std::collections::BTreeMap;

use crate::schema::{
    DecoderAbi, DecoderStateGroup, DecoderStatePort, KvCacheLayout, KvOwnership, LoopStatePair,
    PortRole, SequenceInputKind, StateGroupContract, StateKind, StatePortRole, StateUpdate,
    StaticCacheIoSpec, WorkflowSpec,
};

/// Ordered per-layer view of one state group's port bindings for a component.
struct GroupPorts<'a> {
    /// `(layer, role, state cell, alias)` sorted into graph order.
    aliases: Vec<(
        usize,
        Option<StatePortRole>,
        &'a str,
        &'a crate::schema::StatePortAlias,
    )>,
}

impl<'a> GroupPorts<'a> {
    /// Collect and order this component's aliases in the group.
    ///
    /// Declared `layer` indices order the pairs; when a group binds at most one
    /// alias per role the index is unnecessary and map order is already the
    /// layer order.
    fn collect(group: &'a StateGroupContract, component: &str) -> Option<Self> {
        let bindings = group.ports.get(component)?;
        let mut aliases = bindings
            .iter()
            .enumerate()
            .map(|(position, (cell, alias))| {
                (
                    alias.layer.unwrap_or(position),
                    alias.role,
                    cell.as_str(),
                    alias,
                )
            })
            .collect::<Vec<_>>();
        // Sort by layer first so a producer's label ordering never decides
        // which buffer belongs to which layer, then by role so that a split
        // cache yields a stable key-before-value order within each layer.
        aliases.sort_by_key(|(layer, role, _, _)| (*layer, *role));
        Some(Self { aliases })
    }

    fn pairs(&self) -> impl Iterator<Item = (&'a str, &'a str)> + '_ {
        self.aliases
            .iter()
            .filter(|(_, _, _, alias)| alias.access == crate::schema::StatePortAccess::ReadWrite)
            .filter_map(|(_, _, _, alias)| {
                // A read-write transition always names an output (validation
                // enforces it); the filter_map keeps this total without an
                // unwrap.
                Some((alias.input.as_str(), alias.output.as_deref()?))
            })
    }

    fn with_role(&self, wanted: StatePortRole) -> Vec<&'a crate::schema::StatePortAlias> {
        self.aliases
            .iter()
            .filter(|(_, _, _, alias)| alias.access == crate::schema::StatePortAccess::ReadWrite)
            .filter(|(_, role, _, _)| *role == Some(wanted))
            .map(|(_, _, _, alias)| *alias)
            .collect()
    }
}

pub(crate) fn is_self_attention(kind: StateKind) -> bool {
    matches!(
        kind,
        StateKind::FullAttention
            | StateKind::SlidingAttention
            | StateKind::MultiLatentAttention
            | StateKind::CompressedAttention
    )
}

/// Groups that bind `component`, in a deterministic order.
pub(crate) fn groups_for<'a>(
    workflow: &'a WorkflowSpec,
    component: &str,
) -> Vec<(&'a str, &'a StateGroupContract)> {
    workflow
        .serving
        .iter()
        .flat_map(|serving| serving.state_service.groups.iter())
        .filter(|(_, group)| group.ports.contains_key(component))
        .map(|(name, group)| (name.as_str(), group))
        .collect()
}

/// Find the single port of `component` carrying `role`.
///
/// The role declaration is what names the port. `declared` is consulted only to
/// break ties, never to veto: a producer whose `.onnx` artifact is the
/// authoritative port list may declare the handful of semantic roles — which no
/// graph carries — without also transcribing a contract for every port. Vetoing
/// on absence would silently discard an explicit declaration and send the
/// runtime back to guessing a port by its spelling, which is the one thing this
/// resolver exists to prevent. A name that no graph exposes is caught against
/// the live session, which is strictly stronger than any echo of it here.
fn port_with_role<'a>(
    ports: &'a BTreeMap<String, PortRole>,
    declared: &BTreeMap<String, crate::schema::TensorContract>,
    role: PortRole,
) -> Option<&'a str> {
    let mut matching = ports
        .iter()
        .filter(|(_, declared_role)| **declared_role == role)
        .map(|(port, _)| port.as_str());
    let first = matching.next()?;
    let Some(second) = matching.next() else {
        return Some(first);
    };
    // A duplicated role is a producer error, not a choice to make silently —
    // unless exactly one of the claimants is a port the component declares.
    let mut contracted = std::iter::once(first)
        .chain(std::iter::once(second))
        .chain(matching)
        .filter(|port| declared.contains_key(*port));
    let resolved = contracted.next()?;
    if contracted.next().is_some() {
        return None;
    }
    Some(resolved)
}

/// Derive the decode-step ABI of `component` from the canonical workflow.
///
/// Returns `None` when the workflow does not declare the component. A component
/// that declares no roles yields an empty ABI rather than an error: not every
/// component is a decoder, and the caller decides whether an empty ABI is
/// usable.
pub fn decoder_abi(workflow: &WorkflowSpec, component: &str) -> Option<DecoderAbi> {
    let declaration = workflow.components.get(component)?;
    let roles = &declaration.ports.roles;
    let inputs = &declaration.ports.inputs;
    let outputs = &declaration.ports.outputs;

    let token_input = port_with_role(roles, inputs, PortRole::TokenIds).map(str::to_string);
    let inputs_embeds_input =
        port_with_role(roles, inputs, PortRole::InputsEmbeds).map(str::to_string);

    let groups = groups_for(workflow, component);

    // KV, cross-KV, and fixed carries are three different questions asked of
    // the same declaration: which groups bind this component, and what kind of
    // state each one holds.
    let mut kv_inputs = Vec::new();
    let mut kv_outputs = Vec::new();
    let mut cross_kv_inputs = Vec::new();
    let mut cross_kv_outputs = Vec::new();
    let mut state_pairs = Vec::new();
    let mut aliasing = None;
    let mut kv_layout = None;
    let mut static_cache = None;
    let mut writes_attention_state = false;

    for (_, group) in &groups {
        let Some(ports) = GroupPorts::collect(group, component) else {
            continue;
        };
        if group.kind == StateKind::CompressedAttention {
            let append = !matches!(group.update, Some(StateUpdate::Replace {}));
            for (input, output) in ports.pairs() {
                writes_attention_state = true;
                if append {
                    kv_inputs.push(input.to_string());
                    kv_outputs.push(output.to_string());
                } else {
                    state_pairs.push(LoopStatePair {
                        input: input.to_string(),
                        output: output.to_string(),
                        init: None,
                        update: None,
                    });
                }
            }
            aliasing = Some(match (aliasing, group.aliasing) {
                (None, declared) => declared,
                (Some(current), declared) => strictest(current, declared),
            });
        } else if is_self_attention(group.kind) {
            // A group whose update is `indexed_scatter` is a fixed-capacity
            // cache. Its buffers are fully described by the static-cache ABI, so
            // they must not *also* appear as growing past/present pairs: a
            // runtime that saw both would build a paged bridge over a buffer
            // that never grows and address it with the wrong discipline.
            let scatter = matches!(group.update, Some(StateUpdate::IndexedScatter { .. }));
            for (input, output) in ports.pairs() {
                // A pair exists only where the component writes a `present`
                // back, which is exactly what "owns this state" means. A
                // read-only sharer binds the group but yields no pairs.
                writes_attention_state = true;
                if !scatter {
                    kv_inputs.push(input.to_string());
                    kv_outputs.push(output.to_string());
                }
            }
            // The strictest aliasing any owning group declares governs the
            // component: a runtime may not alias one group's buffers on the
            // strength of another group's permission.
            aliasing = Some(match (aliasing, group.aliasing) {
                (None, declared) => declared,
                (Some(current), declared) => strictest(current, declared),
            });
            if kv_layout.is_none() {
                kv_layout = named_layout(&group.layout);
            }
            if let Some(derived) = derive_static_cache(group, component, &ports) {
                static_cache = Some(derived);
            }
        } else if group.kind == StateKind::CrossAttention {
            for (input, output) in ports.pairs() {
                cross_kv_inputs.push(input.to_string());
                cross_kv_outputs.push(output.to_string());
            }
        } else if group.kind == StateKind::Recurrent {
            for (input, output) in ports.pairs() {
                state_pairs.push(LoopStatePair {
                    input: input.to_string(),
                    output: output.to_string(),
                    init: None,
                    update: None,
                });
            }
        }
    }

    // A component with no owning group either reads state another decoder
    // advances or holds none at all. Both are `shared` from this graph's point
    // of view: it does not advance a past buffer it is responsible for. This is
    // asked of the group bindings rather than of `kv_inputs`, because a
    // fixed-capacity decoder owns its cache while reporting it as a static
    // cache rather than as growing pairs.
    let kv_ownership = Some(if writes_attention_state {
        KvOwnership::Owned
    } else {
        KvOwnership::Shared
    });

    let sequence_source = match (&token_input, &inputs_embeds_input) {
        (Some(_), None) => Some(SequenceInputKind::TokenIds),
        (None, Some(_)) => Some(SequenceInputKind::InputsEmbeds),
        // Declaring both is legal and says nothing about which drives the loop,
        // so the resolved ABI stays silent rather than inventing a preference.
        _ => None,
    };

    // `TensorContract::optional` says a port may be omitted; it does not say
    // which request key signals presence or what value stands in when absent.
    // Those are separate producer facts, so the resolved ABI declares none
    // rather than inventing a presence key that no caller supplies.
    let optional_inputs = BTreeMap::new();

    let state_groups = groups
        .iter()
        .filter_map(|(name, group)| {
            let ports = GroupPorts::collect(group, component)?;
            let ports = ports
                .aliases
                .iter()
                .filter(|(_, _, _, alias)| {
                    alias.access == crate::schema::StatePortAccess::ReadWrite
                })
                .filter_map(|(_, _, cell, alias)| {
                    Some(DecoderStatePort {
                        role: alias.role,
                        layer: alias.layer,
                        batch_axis: workflow
                            .state
                            .get(*cell)
                            .and_then(|state| state.contract.batch_layout.request_axis()),
                        input: alias.input.clone(),
                        output: alias.output.clone()?,
                    })
                })
                .collect();
            Some(DecoderStateGroup {
                name: (*name).to_string(),
                kind: group.kind,
                properties: group.properties.clone(),
                sequence_axis: group.sequence_axis,
                layout: group.layout.clone(),
                aliasing: group.aliasing,
                update: group.update.clone(),
                reuse: group.reuse,
                capabilities: group.capabilities.clone(),
                ports,
            })
        })
        .collect();

    Some(DecoderAbi {
        sequence_source,
        kv_ownership,
        kv_layout,
        token_input,
        inputs_embeds_input,
        attention_mask_input: port_with_role(roles, inputs, PortRole::AttentionMask)
            .map(str::to_string),
        position_ids_input: port_with_role(roles, inputs, PortRole::PositionIds)
            .map(str::to_string),
        logits_output: port_with_role(roles, outputs, PortRole::Logits).map(str::to_string),
        hidden_output: port_with_role(roles, outputs, PortRole::HiddenStates).map(str::to_string),
        kv_inputs: (!kv_inputs.is_empty()).then_some(kv_inputs),
        kv_outputs: (!kv_outputs.is_empty()).then_some(kv_outputs),
        aliasing,
        encoder_hidden_states_input: port_with_role(roles, inputs, PortRole::EncoderHiddenStates)
            .map(str::to_string),
        audio_features_input: port_with_role(roles, inputs, PortRole::AudioFeatures)
            .map(str::to_string),
        cross_kv_inputs: (!cross_kv_inputs.is_empty()).then_some(cross_kv_inputs),
        cross_kv_outputs: (!cross_kv_outputs.is_empty()).then_some(cross_kv_outputs),
        state_pairs: (!state_pairs.is_empty()).then_some(state_pairs),
        optional_inputs,
        static_cache,
        state_groups,
    })
}

/// Map a group's graph-visible element layout onto a named KV layout.
///
/// A group's `layout` is a producer vocabulary describing element order; the
/// named KV layouts are the two strides the KV bridge knows how to address.
/// Only an exact match is a match — a layout this runtime has no stride
/// descriptor for stays `None`, which means "the historical default applies"
/// rather than "this layout is head-major".
fn named_layout(layout: &str) -> Option<KvCacheLayout> {
    match layout {
        "head_major_bnsh" | "bnsh" => Some(KvCacheLayout::head_major_bnsh()),
        "seq_major_bsnh" | "bsnh" => Some(KvCacheLayout::seq_major_bsnh()),
        _ => None,
    }
}

/// Aliasing permission is a promise about graph correctness, so combining two
/// groups keeps the weaker promise.
fn strictest(
    left: crate::schema::StateAliasing,
    right: crate::schema::StateAliasing,
) -> crate::schema::StateAliasing {
    use crate::schema::StateAliasing::{Forbidden, Permitted, Required};
    match (left, right) {
        (Forbidden, _) | (_, Forbidden) => Forbidden,
        (Permitted, _) | (_, Permitted) => Permitted,
        (Required, Required) => Required,
    }
}

/// Rebuild the fixed-capacity scatter ABI from the group that declares it.
///
/// The control ports come from the update discipline, and the per-layer buffers
/// come from the group's own port bindings split by declared half. Nothing here
/// reads a port's spelling.
fn derive_static_cache(
    group: &StateGroupContract,
    component: &str,
    ports: &GroupPorts<'_>,
) -> Option<StaticCacheIoSpec> {
    let StateUpdate::IndexedScatter {
        write_indices_ports,
        kv_length_ports,
        ..
    } = group.update.as_ref()?
    else {
        return None;
    };
    let write_indices_input = write_indices_ports.get(component)?.clone();
    // Without a graph-visible length the valid prefix of a fixed-capacity
    // buffer is not recoverable, and a direct driver cannot bind the ABI.
    let kv_sequence_length_input = kv_length_ports.get(component)?.clone();

    let keys = ports.with_role(StatePortRole::Key);
    let values = ports.with_role(StatePortRole::Value);
    if keys.is_empty() || keys.len() != values.len() {
        return None;
    }

    Some(StaticCacheIoSpec {
        write_indices_input,
        kv_sequence_length_input,
        key_cache_inputs: keys.iter().map(|alias| alias.input.clone()).collect(),
        value_cache_inputs: values.iter().map(|alias| alias.input.clone()).collect(),
        key_cache_outputs: keys
            .iter()
            .filter_map(|alias| alias.output.clone())
            .collect(),
        value_cache_outputs: values
            .iter()
            .filter_map(|alias| alias.output.clone())
            .collect(),
    })
}
