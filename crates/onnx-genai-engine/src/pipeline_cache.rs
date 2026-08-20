//! Reuse across pipeline generations: encoder outputs and decoder KV prefixes.
//!
//! A multimodal turn is expensive twice over. The vision or audio encoder runs a
//! full forward pass over the attachment, and then the decoder prefills a prompt
//! in which that one image has expanded into hundreds or thousands of tokens.
//! A conversation that keeps referring to the same picture pays both costs on
//! every turn, even though neither result can have changed.
//!
//! Two caches remove those repeats:
//!
//! * [`ComponentOutputCache`] memoizes prompt-phase component outputs. A
//!   prompt-phase component is by construction a pure function of its inputs —
//!   that is what distinguishes it from an `every_step` component — so its
//!   outputs are reusable whenever its inputs are bit-identical.
//! * [`RetainedContext`] lets the decoder keep the KV it already computed and
//!   prefill only the tokens the new prompt added.
//!
//! # Why a text prefix cache cannot be reused as-is
//!
//! Ordinary prefix caching keys KV on token ids. That is unsound the moment
//! embeddings enter the prompt from somewhere other than the token embedding
//! table. Placeholder expansion replaces one image with a **run of one repeated
//! placeholder token**, so two entirely different photographs produce *identical*
//! token sequences. Keyed on tokens alone, a cache would serve the first photo's
//! KV for the second one and the model would answer fluently about a picture it
//! was never shown — the worst class of bug, because nothing looks wrong.
//!
//! So the key here is the token sequence **plus a digest of every externally
//! bound input tensor**. Change the picture and the digest changes, the prefix
//! stops matching, and the turn is recomputed.

use onnx_genai_ort::Value;
use std::collections::HashMap;

use crate::TokenId;
use crate::decode::clone_value;

/// 128-bit FNV-1a content digest.
///
/// 128 bits rather than 64 because nothing ever verifies a hit: a collision
/// would silently answer about the wrong attachment. At 64 bits that margin is
/// uncomfortably thin for a value with no fallback; at 128 it is not a risk
/// worth modelling.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Digest(u128);

impl Digest {
    /// The digest as four little-endian 32-bit words.
    ///
    /// [`PrefixCache`](crate::PrefixCache) is a radix trie over token ids, so
    /// the only way to make an attachment part of a prefix key is to spell the
    /// digest in the same alphabet and put it in front of the tokens. Four words
    /// carry all 128 bits, so two requests share a cached prefix only if their
    /// attachments were bit-identical — which is the whole point, since
    /// placeholder expansion makes different images produce identical tokens.
    pub fn words(&self) -> [u32; 4] {
        [
            self.0 as u32,
            (self.0 >> 32) as u32,
            (self.0 >> 64) as u32,
            (self.0 >> 96) as u32,
        ]
    }
}

/// Elements a [`prefix_key`] prepends before the prompt tokens.
pub const PREFIX_KEY_PREAMBLE: usize = 4;

/// The key a multimodal prompt is cached under: its attachments, then its
/// tokens.
///
/// Without the preamble two different photographs would share a cache entry,
/// because expansion replaces each image with the same repeated placeholder
/// token. With it, a changed attachment diverges at the first element and
/// nothing is shared.
pub fn prefix_key(inputs: Digest, tokens: &[TokenId]) -> Vec<TokenId> {
    let mut key = Vec::with_capacity(PREFIX_KEY_PREAMBLE + tokens.len());
    key.extend_from_slice(&inputs.words());
    key.extend_from_slice(tokens);
    key
}

const FNV_OFFSET_BASIS_128: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
const FNV_PRIME_128: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

/// Incremental [`Digest`] builder.
///
/// Absorbs 8 bytes per round rather than one. Standard FNV-1a is byte-at-a-time,
/// which costs tens of milliseconds over a multi-megabyte pixel tensor — real
/// money on a path whose whole purpose is to be cheaper than the encoder. The
/// avalanche behaviour that matters here is unaffected, and the construction is
/// fixed in this file, so digests stay stable for the life of the process pool.
#[derive(Debug)]
pub struct DigestBuilder {
    state: u128,
}

impl DigestBuilder {
    pub fn new() -> Self {
        Self {
            state: FNV_OFFSET_BASIS_128,
        }
    }

    /// Absorb `bytes`, length-prefixed so that concatenating two fields cannot
    /// collide with a single field holding their concatenation.
    pub fn absorb(&mut self, bytes: &[u8]) {
        self.absorb_u64(bytes.len() as u64);
        let (chunks, remainder) = bytes.as_chunks::<8>();
        for chunk in chunks {
            self.mix(u128::from(u64::from_le_bytes(*chunk)));
        }
        let mut tail = [0u8; 8];
        tail[..remainder.len()].copy_from_slice(remainder);
        self.mix(u128::from(u64::from_le_bytes(tail)));
    }

    pub fn absorb_u64(&mut self, value: u64) {
        self.mix(u128::from(value));
    }

    pub fn absorb_str(&mut self, value: &str) {
        self.absorb(value.as_bytes());
    }

    fn mix(&mut self, word: u128) {
        self.state ^= word;
        self.state = self.state.wrapping_mul(FNV_PRIME_128);
    }

    pub fn finish(self) -> Digest {
        Digest(self.state)
    }
}

impl Default for DigestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Absorb a tensor's dtype, shape, and element bytes.
///
/// Returns `false` when the tensor's bytes cannot be read. A key that silently
/// omitted one of its inputs would alias unrelated requests, so the caller must
/// treat this as "not cacheable" rather than as an empty contribution.
#[must_use]
pub fn absorb_value(builder: &mut DigestBuilder, value: &Value) -> bool {
    let Ok(bytes) = value.as_raw_bytes() else {
        return false;
    };
    builder.absorb_u64(value.dtype() as u64);
    builder.absorb_u64(value.shape().len() as u64);
    for &dim in value.shape() {
        builder.absorb_u64(dim as u64);
    }
    builder.absorb(bytes);
    true
}

/// Digest a set of named tensors, order-independently.
///
/// Returns `None` when any tensor is not host-readable, so the caller falls back
/// to recomputing rather than keying on a partial description.
pub fn digest_named_values<'a>(
    label: &str,
    values: impl IntoIterator<Item = (&'a str, &'a Value)>,
) -> Option<Digest> {
    let mut named = values.into_iter().collect::<Vec<_>>();
    named.sort_by_key(|(name, _)| *name);
    let mut builder = DigestBuilder::new();
    builder.absorb_str(label);
    builder.absorb_u64(named.len() as u64);
    for (name, value) in named {
        builder.absorb_str(name);
        if !absorb_value(&mut builder, value) {
            return None;
        }
    }
    Some(builder.finish())
}

/// ONNX operators whose output is not a function of their input alone.
///
/// Memoizing a component containing one of these would freeze the first draw
/// and silently make every later call return it. Taken from the ONNX standard
/// rather than inferred, because "which operators are random" is a property of
/// the operator set, not of any particular model.
const NON_DETERMINISTIC_OPERATORS: &[&str] = &[
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
    "Multinomial",
    "Bernoulli",
    "Dropout",
];

/// Whether every operator in `model` is a pure function of its inputs, and so
/// whether the component's outputs may be memoized.
///
/// A component's declared phase says only *when* it runs, never that it is
/// deterministic, so purity is read off the graph instead of assumed from
/// `run_on: prompt_only`.
///
/// Everything the runtime will execute is walked, not just the main graph: a
/// random operator is no less random for sitting inside a `Loop` body or a
/// model-local function that gets inlined at load.
///
/// The blacklist covers the standard ONNX operator set, where "which operators
/// are random" is fixed by the specification. A custom-domain operator is taken
/// at face value; a package whose encoder contains a non-deterministic custom
/// operator should set `pipeline_cache_bytes` to `0`.
pub fn graph_is_deterministic(model: &onnx_runtime_loader::proto::ModelProto) -> bool {
    let main_graph = model
        .graph
        .as_ref()
        .is_none_or(graph_nodes_are_deterministic);
    // A node may call a model-local function whose body holds the random
    // operator; the caller's own op_type reveals nothing about it.
    main_graph
        && model
            .functions
            .iter()
            .all(|function| nodes_are_deterministic(&function.node))
}

fn graph_nodes_are_deterministic(graph: &onnx_runtime_loader::proto::onnx::GraphProto) -> bool {
    nodes_are_deterministic(&graph.node)
}

fn nodes_are_deterministic(nodes: &[onnx_runtime_loader::proto::onnx::NodeProto]) -> bool {
    nodes.iter().all(|node| {
        !NON_DETERMINISTIC_OPERATORS.contains(&node.op_type.as_str())
            && node.attribute.iter().all(|attribute| {
                attribute
                    .g
                    .as_ref()
                    .is_none_or(graph_nodes_are_deterministic)
                    && attribute.graphs.iter().all(graph_nodes_are_deterministic)
            })
    })
}

/// Memoized outputs of prompt-phase pipeline components.
///
/// Bounded by total cached bytes and evicted least-recently-used, because the
/// entries are encoder outputs whose size scales with the attachment.
pub struct ComponentOutputCache {
    entries: HashMap<Digest, Entry>,
    capacity_bytes: usize,
    bytes: usize,
    clock: u64,
    stats: PipelineCacheStats,
}

struct Entry {
    outputs: Vec<(String, Value)>,
    bytes: usize,
    last_used: u64,
}

/// Counters describing what the pipeline caches did for a generation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PipelineCacheStats {
    /// Prompt-phase component runs served from memoized outputs.
    pub encoder_hits: u64,
    /// Prompt-phase component runs that had to execute.
    pub encoder_misses: u64,
    /// Prompt-phase component runs that could not be keyed, and so ran without
    /// consulting or populating the cache.
    pub encoder_unkeyable: u64,
    /// Entries dropped to stay within the byte budget.
    pub encoder_evictions: u64,
    /// Bytes currently held by memoized outputs.
    pub encoder_bytes: u64,
    /// Prompt tokens whose decoder KV was carried over from the previous turn.
    pub prefix_reused_tokens: u64,
    /// Prompt tokens the decoder had to prefill.
    pub prefill_tokens: u64,
}

impl ComponentOutputCache {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity_bytes,
            bytes: 0,
            clock: 0,
            stats: PipelineCacheStats::default(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.capacity_bytes > 0
    }

    pub fn stats(&self) -> PipelineCacheStats {
        PipelineCacheStats {
            encoder_bytes: self.bytes as u64,
            ..self.stats
        }
    }

    pub fn reset_stats(&mut self) {
        self.stats = PipelineCacheStats::default();
    }

    pub fn note_unkeyable(&mut self) {
        self.stats.encoder_unkeyable += 1;
    }

    pub fn note_prefix_reuse(&mut self, reused: usize, prefilled: usize) {
        self.stats.prefix_reused_tokens += reused as u64;
        self.stats.prefill_tokens += prefilled as u64;
    }

    /// Fetch memoized outputs, cloning them for the caller.
    ///
    /// Clones rather than lends because the caller publishes them into a mutable
    /// tensor pool that later stages may overwrite. The copy is a memcpy of an
    /// encoder output, which is orders of magnitude below the forward pass it
    /// replaces.
    pub fn get(&mut self, key: Digest) -> Option<Vec<(String, Value)>> {
        self.clock += 1;
        let clock = self.clock;
        let entry = self.entries.get_mut(&key)?;
        entry.last_used = clock;
        let cloned = entry
            .outputs
            .iter()
            .map(|(name, value)| clone_value(value).map(|value| (name.clone(), value)))
            .collect::<anyhow::Result<Vec<_>>>();
        match cloned {
            Ok(outputs) => {
                self.stats.encoder_hits += 1;
                Some(outputs)
            }
            // An un-clonable dtype is not an error, just not reusable. Drop the
            // entry so the miss is not paid for on every later turn too.
            Err(_) => {
                self.remove(key);
                None
            }
        }
    }

    pub fn note_miss(&mut self) {
        self.stats.encoder_misses += 1;
    }

    /// Memoize `outputs` under `key`, evicting least-recently-used entries to
    /// stay within the byte budget.
    ///
    /// Silently declines when an output cannot be copied or when the entry alone
    /// exceeds the whole budget: a cache is an optimization, and failing the
    /// generation because it could not be populated would trade a fast path for
    /// a broken one.
    pub fn insert(&mut self, key: Digest, outputs: &[(String, Value)]) {
        if !self.is_enabled() || self.entries.contains_key(&key) {
            return;
        }
        let mut copied = Vec::with_capacity(outputs.len());
        let mut bytes = 0usize;
        for (name, value) in outputs {
            let (Ok(copy), Ok(raw)) = (clone_value(value), value.as_raw_bytes()) else {
                return;
            };
            bytes += raw.len();
            copied.push((name.clone(), copy));
        }
        if bytes > self.capacity_bytes {
            return;
        }
        while self.bytes + bytes > self.capacity_bytes {
            if !self.evict_one() {
                return;
            }
        }
        self.clock += 1;
        self.bytes += bytes;
        self.entries.insert(
            key,
            Entry {
                outputs: copied,
                bytes,
                last_used: self.clock,
            },
        );
    }

    fn remove(&mut self, key: Digest) {
        if let Some(entry) = self.entries.remove(&key) {
            self.bytes -= entry.bytes;
        }
    }

    fn evict_one(&mut self) -> bool {
        let Some(&victim) = self
            .entries
            .iter()
            .min_by_key(|(key, entry)| (entry.last_used, **key))
            .map(|(key, _)| key)
        else {
            return false;
        };
        self.remove(victim);
        self.stats.encoder_evictions += 1;
        true
    }
}

/// The decoder KV a pipeline is still holding from its previous generation.
///
/// `inputs` covers every externally bound tensor — the images and audio for that
/// turn. It is part of the identity of the KV, not metadata about it: the same
/// tokens over a different picture describe different hidden states.
#[derive(Debug, Clone)]
pub struct RetainedContext {
    pub inputs: Digest,
    pub tokens: Vec<TokenId>,
}

impl RetainedContext {
    /// How many leading tokens of `tokens` the retained KV can serve.
    ///
    /// This is the length of the common prefix, so a prompt that *diverges*
    /// from the retained context still reuses the part it shares. That is what
    /// makes forking a conversation, editing an earlier turn, or a reasoning
    /// model reuse anything at all — a reasoning model's replayed history drops
    /// the thinking the KV still contains, so its prompts always diverge.
    ///
    /// When the answer is shorter than the retained context, the caller must
    /// first truncate the KV to match; that can fail, and recomputing is the
    /// fallback.
    ///
    /// Zero when the attachments differ at all: identical tokens over a
    /// different image are a different computation, and tokens alone cannot say
    /// so.
    ///
    /// At least one token is always left to prefill, since a decode step needs
    /// an input to produce logits from.
    pub fn reusable_prefix(&self, inputs: Digest, tokens: &[TokenId]) -> usize {
        if self.inputs != inputs {
            return 0;
        }
        let limit = tokens.len().saturating_sub(1).min(self.tokens.len());
        (0..limit)
            .take_while(|&index| tokens[index] == self.tokens[index])
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_ort::Value;

    fn value(data: &[f32]) -> Value {
        Value::from_slice_f32(data, &[data.len() as i64]).expect("build test tensor")
    }

    fn digest_of(name: &str, data: &[f32]) -> Digest {
        let value = value(data);
        digest_named_values("component", [(name, &value)]).expect("host tensor is digestible")
    }

    mod determinism {
        use super::super::graph_is_deterministic;
        use onnx_runtime_loader::proto::ModelProto;
        use onnx_runtime_loader::proto::onnx::{
            AttributeProto, FunctionProto, GraphProto, NodeProto, attribute_proto,
        };

        fn node(op_type: &str) -> NodeProto {
            NodeProto {
                op_type: op_type.to_string(),
                ..NodeProto::default()
            }
        }

        fn model(nodes: Vec<NodeProto>) -> ModelProto {
            ModelProto {
                graph: Some(GraphProto {
                    node: nodes,
                    ..GraphProto::default()
                }),
                ..ModelProto::default()
            }
        }

        #[test]
        fn a_plain_encoder_graph_is_memoizable() {
            assert!(graph_is_deterministic(&model(vec![
                node("Conv"),
                node("MatMul"),
                node("Softmax"),
            ])));
        }

        #[test]
        fn a_graph_that_draws_random_numbers_is_not() {
            // Memoizing this would freeze the first draw and return it forever.
            assert!(!graph_is_deterministic(&model(vec![
                node("MatMul"),
                node("RandomNormalLike"),
            ])));
        }

        #[test]
        fn a_random_operator_hidden_in_a_subgraph_still_counts() {
            let mut loop_node = node("Loop");
            loop_node.attribute.push(AttributeProto {
                name: "body".to_string(),
                r#type: attribute_proto::AttributeType::Graph as i32,
                g: Some(GraphProto {
                    node: vec![node("RandomUniform")],
                    ..GraphProto::default()
                }),
                ..AttributeProto::default()
            });
            assert!(
                !graph_is_deterministic(&model(vec![loop_node])),
                "a Loop body is no less random for being nested"
            );
        }

        #[test]
        fn a_random_operator_inside_a_model_local_function_still_counts() {
            // The calling node's op_type is the function's name, which says
            // nothing about the body the runtime will inline and run.
            let mut model = model(vec![node("MyFusedEncoderBlock")]);
            model.functions.push(FunctionProto {
                name: "MyFusedEncoderBlock".to_string(),
                node: vec![node("MatMul"), node("RandomNormal")],
                ..FunctionProto::default()
            });
            assert!(!graph_is_deterministic(&model));
        }

        #[test]
        fn a_deterministic_model_local_function_stays_memoizable() {
            let mut model = model(vec![node("MyFusedEncoderBlock")]);
            model.functions.push(FunctionProto {
                name: "MyFusedEncoderBlock".to_string(),
                node: vec![node("MatMul"), node("Add")],
                ..FunctionProto::default()
            });
            assert!(graph_is_deterministic(&model));
        }

        #[test]
        fn a_model_with_no_graph_is_treated_as_memoizable() {
            // Nothing to be non-deterministic; the session would have failed to
            // load long before this mattered.
            assert!(graph_is_deterministic(&ModelProto::default()));
        }
    }

    #[test]
    fn a_prefix_key_carries_the_attachment_ahead_of_the_tokens() {
        let inputs = digest_of("pixels", &[1.0]);
        let key = prefix_key(inputs, &[7, 8, 9]);
        assert_eq!(key.len(), PREFIX_KEY_PREAMBLE + 3);
        assert_eq!(&key[PREFIX_KEY_PREAMBLE..], &[7, 8, 9]);
        assert_eq!(&key[..PREFIX_KEY_PREAMBLE], &inputs.words());
    }

    #[test]
    fn the_same_tokens_over_a_different_attachment_share_no_key_prefix() {
        // The trap this exists for: expansion makes these token sequences
        // identical, so only the preamble can keep the pictures apart.
        let left = prefix_key(digest_of("pixels", &[1.0]), &[7, 7, 7]);
        let right = prefix_key(digest_of("pixels", &[2.0]), &[7, 7, 7]);
        assert_ne!(left[0..PREFIX_KEY_PREAMBLE], right[0..PREFIX_KEY_PREAMBLE]);
        let shared = left.iter().zip(&right).take_while(|(a, b)| a == b).count();
        assert_eq!(shared, 0, "they must diverge at the very first element");
    }

    #[test]
    fn the_same_attachment_shares_everything_up_to_the_prompt_divergence() {
        let inputs = digest_of("pixels", &[1.0]);
        let left = prefix_key(inputs, &[1, 2, 3, 4]);
        let right = prefix_key(inputs, &[1, 2, 9]);
        let shared = left.iter().zip(&right).take_while(|(a, b)| a == b).count();
        assert_eq!(shared, PREFIX_KEY_PREAMBLE + 2);
    }

    #[test]
    fn all_128_bits_reach_the_key() {
        // A digest differing only in its high bits must still change the key,
        // or half the hash would be decoration.
        let low = Digest(1);
        let high = Digest(1u128 << 100);
        assert_ne!(low.words(), high.words());
        assert_ne!(prefix_key(low, &[5]), prefix_key(high, &[5]));
    }

    #[test]
    fn identical_inputs_digest_identically() {
        assert_eq!(
            digest_of("pixels", &[1.0, 2.0]),
            digest_of("pixels", &[1.0, 2.0])
        );
    }

    #[test]
    fn a_different_attachment_digests_differently() {
        assert_ne!(
            digest_of("pixels", &[1.0, 2.0]),
            digest_of("pixels", &[1.0, 3.0])
        );
    }

    #[test]
    fn the_input_name_is_part_of_the_key() {
        assert_ne!(digest_of("pixels", &[1.0]), digest_of("features", &[1.0]));
    }

    #[test]
    fn shape_is_part_of_the_key_even_when_the_bytes_match() {
        let flat = Value::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let square = Value::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_ne!(
            digest_named_values("c", [("x", &flat)]),
            digest_named_values("c", [("x", &square)]),
            "a reshaped tensor is a different input to the encoder"
        );
    }

    #[test]
    fn input_order_does_not_change_the_key() {
        let a = value(&[1.0]);
        let b = value(&[2.0]);
        assert_eq!(
            digest_named_values("c", [("a", &a), ("b", &b)]),
            digest_named_values("c", [("b", &b), ("a", &a)]),
            "inputs are bound by name, so enumeration order must not matter"
        );
    }

    #[test]
    fn a_hit_returns_the_memoized_outputs() {
        let mut cache = ComponentOutputCache::new(1 << 20);
        let key = digest_of("pixels", &[1.0]);
        cache.insert(key, &[("hidden".to_string(), value(&[7.0, 8.0]))]);

        let hit = cache.get(key).expect("the entry was just inserted");
        assert_eq!(hit.len(), 1);
        assert_eq!(hit[0].0, "hidden");
        assert_eq!(hit[0].1.to_vec_f32().unwrap(), vec![7.0, 8.0]);
        assert_eq!(cache.stats().encoder_hits, 1);
    }

    #[test]
    fn a_different_attachment_misses() {
        let mut cache = ComponentOutputCache::new(1 << 20);
        cache.insert(
            digest_of("pixels", &[1.0]),
            &[("hidden".to_string(), value(&[7.0]))],
        );
        assert!(cache.get(digest_of("pixels", &[2.0])).is_none());
    }

    #[test]
    fn a_zero_budget_disables_the_cache() {
        let mut cache = ComponentOutputCache::new(0);
        let key = digest_of("pixels", &[1.0]);
        cache.insert(key, &[("hidden".to_string(), value(&[7.0]))]);
        assert!(cache.get(key).is_none(), "nothing may be retained");
    }

    #[test]
    fn the_least_recently_used_entry_is_evicted_first() {
        // Three f32 entries of 8 bytes each, in a budget that fits two.
        let mut cache = ComponentOutputCache::new(16);
        let (a, b, c) = (
            digest_of("x", &[1.0]),
            digest_of("x", &[2.0]),
            digest_of("x", &[3.0]),
        );
        cache.insert(a, &[("o".to_string(), value(&[1.0, 1.0]))]);
        cache.insert(b, &[("o".to_string(), value(&[2.0, 2.0]))]);
        assert!(cache.get(a).is_some(), "touch a so b becomes the victim");

        cache.insert(c, &[("o".to_string(), value(&[3.0, 3.0]))]);
        assert!(cache.get(a).is_some());
        assert!(cache.get(c).is_some());
        assert!(cache.get(b).is_none(), "b was least recently used");
        assert_eq!(cache.stats().encoder_evictions, 1);
    }

    #[test]
    fn an_entry_larger_than_the_whole_budget_is_declined() {
        let mut cache = ComponentOutputCache::new(8);
        let key = digest_of("x", &[1.0]);
        cache.insert(key, &[("o".to_string(), value(&[1.0, 2.0, 3.0]))]);
        assert!(cache.get(key).is_none());
    }

    #[test]
    fn an_appended_conversation_reuses_the_whole_retained_context() {
        let inputs = digest_of("pixels", &[1.0]);
        let retained = RetainedContext {
            inputs,
            tokens: vec![1, 2, 3],
        };
        assert_eq!(retained.reusable_prefix(inputs, &[1, 2, 3, 4, 5]), 3);
    }

    #[test]
    fn the_same_tokens_over_a_different_image_reuse_nothing() {
        // The whole point: placeholder expansion makes these token sequences
        // identical, so only the input digest can tell the pictures apart.
        let retained = RetainedContext {
            inputs: digest_of("pixels", &[1.0]),
            tokens: vec![1, 2, 3],
        };
        assert_eq!(
            retained.reusable_prefix(digest_of("pixels", &[2.0]), &[1, 2, 3, 4]),
            0,
            "a new image must invalidate the retained KV"
        );
    }

    #[test]
    fn a_diverging_prompt_reuses_what_it_still_shares() {
        // Forking a conversation, editing an earlier turn, and replaying a
        // reasoning model's history all look like this: a common head, then a
        // different tail. The shared head is still worth keeping.
        let inputs = digest_of("pixels", &[1.0]);
        let retained = RetainedContext {
            inputs,
            tokens: vec![1, 2, 3],
        };
        assert_eq!(retained.reusable_prefix(inputs, &[1, 2, 9, 4]), 2);
    }

    #[test]
    fn a_prompt_sharing_nothing_reuses_nothing() {
        let inputs = digest_of("pixels", &[1.0]);
        let retained = RetainedContext {
            inputs,
            tokens: vec![1, 2, 3],
        };
        assert_eq!(retained.reusable_prefix(inputs, &[9, 2, 3, 4]), 0);
    }

    #[test]
    fn a_prompt_that_only_repeats_the_retained_context_still_prefills_a_token() {
        let inputs = digest_of("pixels", &[1.0]);
        let retained = RetainedContext {
            inputs,
            tokens: vec![1, 2, 3],
        };
        assert_eq!(
            retained.reusable_prefix(inputs, &[1, 2, 3]),
            2,
            "a decode step needs at least one input token to produce logits from"
        );
    }
}
