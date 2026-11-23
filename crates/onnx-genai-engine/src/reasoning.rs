//! Token-protocol reasoning budgets shared by all host sampling paths.
use crate::{
    byte_stream::{ByteStreamContract, ByteStreamState},
    logits::{LogitProcessor, ProcessorContext, TokenId},
    tokenization::SpecialTokenHandling,
};
use anyhow::{Context, ensure};
use onnx_genai_ort::Tokenizer;
use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    fmt,
    num::NonZeroUsize,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

/// An explicit token protocol. Prefix tokens are existing assistant prefill,
/// not the complete conversation. The first end sequence is the default closer.
#[derive(Debug, Clone)]
pub struct ReasoningBudgetSpec {
    pub start_sequences: Vec<Vec<TokenId>>,
    /// A declared implicit reasoning channel is active before its prefix tokens.
    pub initially_active: bool,
    pub end_sequences: Vec<Vec<TokenId>>,
    /// Message plus closer. Empty selects the first declared end sequence.
    pub forced_tokens: Vec<TokenId>,
    pub prefix_tokens: Vec<TokenId>,
    /// None leaves the budget unlimited while retaining manual control.
    pub max_tokens: Option<usize>,
}

/// A request-local end-reasoning signal. Independent sampler branches observe
/// the same revision; a draft cannot consume another branch's control event.
#[derive(Debug, Clone, Default)]
pub struct ReasoningControl(Arc<AtomicU64>);
impl ReasoningControl {
    pub fn request_end(&self) -> anyhow::Result<()> {
        self.0
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |revision| {
                revision.checked_add(1)
            })
            .map(|_| ())
            .map_err(|_| {
                anyhow::anyhow!("reasoning control revision overflow; start a new completion")
            })
    }
    fn revision(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }
}

struct Contract {
    start: Matcher,
    end: Matcher,
    forced: Arc<[TokenId]>,
    prefix: Vec<TokenId>,
    limit: Option<usize>,
    initially_active: bool,
    bytes: Option<ByteStreamContract>,
}

/// Validated reasoning protocol and tokenizer capability. Mutable sampling
/// history is created separately for each processor chain.
#[derive(Clone)]
pub struct ReasoningBudget {
    contract: Arc<Contract>,
    control: ReasoningControl,
}
impl fmt::Debug for ReasoningBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReasoningBudget")
            .field("max_tokens", &self.contract.limit)
            .field("forced_tokens", &self.contract.forced)
            .finish_non_exhaustive()
    }
}
impl ReasoningBudget {
    pub fn resolve(tokenizer: &Tokenizer, mut spec: ReasoningBudgetSpec) -> anyhow::Result<Self> {
        ensure!(
            (spec.initially_active || !spec.start_sequences.is_empty())
                && spec
                    .start_sequences
                    .iter()
                    .all(|sequence| !sequence.is_empty()),
            "reasoning_budget_start_tag requires nonempty token sequences; provide a declared reasoning opener"
        );
        ensure!(
            !spec.end_sequences.is_empty()
                && spec
                    .end_sequences
                    .iter()
                    .all(|sequence| !sequence.is_empty()),
            "reasoning_budget_end_tags requires nonempty token sequences; provide a declared reasoning closer"
        );
        if spec.forced_tokens.is_empty() {
            spec.forced_tokens = spec.end_sequences[0].clone();
        }
        ensure!(
            spec.end_sequences
                .iter()
                .any(|sequence| spec.forced_tokens.ends_with(sequence)),
            "reasoning forced_tokens must end with a declared end sequence; append the closer after the budget message"
        );
        let vocabulary = tokenizer
            .inner()
            .get_vocab(true)
            .into_values()
            .collect::<HashSet<_>>();
        for (name, sequences) in [
            ("start", &spec.start_sequences),
            ("end", &spec.end_sequences),
        ] {
            for (index, sequence) in sequences.iter().enumerate() {
                for &id in sequence {
                    ensure!(
                        vocabulary.contains(&id),
                        "reasoning {name} sequence {index} contains absent token ID {id}; repair the tokenizer protocol"
                    );
                }
            }
        }
        for (&id, name) in spec
            .forced_tokens
            .iter()
            .map(|id| (id, "forced_tokens"))
            .chain(spec.prefix_tokens.iter().map(|id| (id, "prefix_tokens")))
        {
            ensure!(
                vocabulary.contains(&id),
                "reasoning {name} contains absent token ID {id}; repair the tokenizer protocol"
            );
        }
        let budget = Self {
            contract: Arc::new(Contract {
                start: Matcher::new(spec.start_sequences),
                end: Matcher::new(spec.end_sequences),
                forced: spec.forced_tokens.into(),
                prefix: spec.prefix_tokens,
                limit: spec.max_tokens,
                initially_active: spec.initially_active,
                bytes: ByteStreamContract::resolve(tokenizer, SpecialTokenHandling::Preserve)
                    .context("resolve reasoning-budget UTF-8 tokenizer contract")?,
            }),
            control: ReasoningControl::default(),
        };
        budget.initial_state()?;
        Ok(budget)
    }
    pub(crate) fn forced_tokens(&self) -> &[TokenId] {
        &self.contract.forced
    }

    pub fn control(&self) -> ReasoningControl {
        self.control.clone()
    }
    pub(crate) fn processor(&self) -> anyhow::Result<BudgetProcessor> {
        Ok(BudgetProcessor {
            budget: self.clone(),
            state: Mutex::new(self.initial_state()?),
        })
    }
    fn initial_state(&self) -> anyhow::Result<State> {
        let mut machine = Machine::default();
        if self.contract.initially_active {
            machine.activate(&self.contract);
        }
        let mut bytes = BytesState::new(&self.contract);
        for &id in &self.contract.prefix {
            let complete = bytes.accept(&self.contract, id)?;
            machine.accept(&self.contract, id, complete, false)?;
        }
        ensure!(
            bytes.complete(),
            "reasoning prefix_tokens end inside a UTF-8 character; provide a complete assistant prefill before enabling the budget"
        );
        Ok(State {
            history: Vec::new(),
            checkpoints: vec![machine],
            bytes: BytesState::new(&self.contract),
        })
    }
}

#[derive(Default)]
struct Node {
    next: BTreeMap<TokenId, usize>,
    failure: usize,
    matched: Option<usize>,
}
struct Matcher {
    nodes: Vec<Node>,
}
impl Matcher {
    fn new(sequences: Vec<Vec<TokenId>>) -> Self {
        let mut nodes = vec![Node::default()];
        let lengths = sequences.iter().map(Vec::len).collect::<Vec<_>>();
        for (index, sequence) in sequences.into_iter().enumerate() {
            let mut at = 0;
            for id in sequence {
                at = if let Some(&child) = nodes[at].next.get(&id) {
                    child
                } else {
                    let child = nodes.len();
                    nodes.push(Node::default());
                    nodes[at].next.insert(id, child);
                    child
                };
            }
            nodes[at].matched = nodes[at].matched.or(Some(index));
        }
        let mut queue = nodes[0].next.values().copied().collect::<VecDeque<_>>();
        while let Some(parent) = queue.pop_front() {
            let children = nodes[parent]
                .next
                .iter()
                .map(|(&id, &child)| (id, child))
                .collect::<Vec<_>>();
            for (id, child) in children {
                let mut failure = nodes[parent].failure;
                while failure != 0 && !nodes[failure].next.contains_key(&id) {
                    failure = nodes[failure].failure;
                }
                let failure = nodes[failure].next.get(&id).copied().unwrap_or(0);
                nodes[child].failure = failure;
                let inherited = nodes[failure].matched;
                if let Some(inherited) = inherited
                    && nodes[child].matched.is_none_or(|own| {
                        lengths[inherited] > lengths[own]
                            || (lengths[inherited] == lengths[own] && inherited < own)
                    })
                {
                    nodes[child].matched = Some(inherited);
                }
                queue.push_back(child);
            }
        }
        Self { nodes }
    }
    fn advance(&self, at: &mut usize, id: TokenId) -> Option<usize> {
        while *at != 0 && !self.nodes[*at].next.contains_key(&id) {
            *at = self.nodes[*at].failure;
        }
        *at = self.nodes[*at].next.get(&id).copied().unwrap_or(0);
        let matched = self.nodes[*at].matched;
        if matched.is_some() {
            *at = 0;
        }
        matched
    }
}

#[derive(Clone, Default)]
enum Phase {
    #[default]
    Idle,
    Counting {
        remaining: Option<NonZeroUsize>,
    },
    WaitingUtf8,
    Forcing {
        remaining: NonZeroUsize,
    },
    Done,
}
#[derive(Clone, Default)]
struct Machine {
    phase: Phase,
    start: usize,
    end: usize,
    control_revision: u64,
}
impl Machine {
    fn activate(&mut self, contract: &Contract) {
        self.phase = match contract.limit {
            Some(0) => Self::forcing(contract),
            limit => Phase::Counting {
                remaining: limit.and_then(NonZeroUsize::new),
            },
        };
        self.end = 0;
    }
    fn forcing(contract: &Contract) -> Phase {
        Phase::Forcing {
            remaining: NonZeroUsize::new(contract.forced.len())
                .expect("resolved nonempty reasoning closer"),
        }
    }
    fn forced(&self, contract: &Contract) -> Option<TokenId> {
        match self.phase {
            Phase::Forcing { remaining } => {
                Some(contract.forced[contract.forced.len() - remaining.get()])
            }
            _ => None,
        }
    }
    fn control(&mut self, contract: &Contract, revision: u64, complete: bool) {
        if self.control_revision != revision {
            self.control_revision = revision;
            if matches!(self.phase, Phase::Counting { .. }) {
                self.end = 0;
                self.phase = if complete {
                    Self::forcing(contract)
                } else {
                    Phase::WaitingUtf8
                };
            }
        }
    }
    fn accept(
        &mut self,
        contract: &Contract,
        id: TokenId,
        complete: bool,
        generated: bool,
    ) -> anyhow::Result<()> {
        match self.phase.clone() {
            Phase::Idle | Phase::Done => {
                if contract.start.advance(&mut self.start, id).is_some() {
                    self.activate(contract);
                }
            }
            phase @ (Phase::Counting { .. } | Phase::WaitingUtf8) => {
                let remaining = match phase {
                    Phase::Counting { remaining } => remaining,
                    _ => None,
                };
                if contract.end.advance(&mut self.end, id).is_some() {
                    self.phase = Phase::Done;
                    return Ok(());
                }
                if matches!(self.phase, Phase::WaitingUtf8) {
                    if complete {
                        self.phase = Self::forcing(contract);
                        self.end = 0;
                    }
                } else if let Some(remaining) = remaining {
                    self.phase = match NonZeroUsize::new(remaining.get() - 1) {
                        Some(remaining) => Phase::Counting {
                            remaining: Some(remaining),
                        },
                        None if complete => {
                            self.end = 0;
                            Self::forcing(contract)
                        }
                        None => {
                            self.end = 0;
                            Phase::WaitingUtf8
                        }
                    };
                }
            }
            Phase::Forcing { remaining } => {
                if generated {
                    ensure!(
                        self.forced(contract) == Some(id),
                        "reasoning history token {id} violates the forced closer at generated boundary; replay tokens admitted by the same reasoning budget"
                    );
                }
                if contract.end.advance(&mut self.end, id).is_some() {
                    self.phase = Phase::Done;
                } else if generated {
                    self.phase = match NonZeroUsize::new(remaining.get() - 1) {
                        Some(remaining) => Phase::Forcing { remaining },
                        None => Phase::Done,
                    };
                }
                // Existing prefill is not sampled: it can naturally close the
                // block, but cannot consume an ungenerated forced sequence.
            }
        }
        Ok(())
    }
}

struct BytesState {
    decoder: Option<ByteStreamState>,
    pending: Vec<u8>,
}
impl BytesState {
    fn new(contract: &Contract) -> Self {
        Self {
            decoder: contract.bytes.as_ref().map(ByteStreamContract::new_state),
            pending: Vec::new(),
        }
    }
    fn complete(&self) -> bool {
        self.pending.is_empty()
            && self
                .decoder
                .as_ref()
                .is_none_or(|decoder| !decoder.has_pending_prefix())
    }
    fn accept(&mut self, contract: &Contract, id: TokenId) -> anyhow::Result<bool> {
        if let (Some(bytes), Some(decoder)) = (&contract.bytes, &mut self.decoder) {
            let piece = bytes.pieces.get(id as usize).with_context(|| {
                format!("reasoning token {id} has no byte-decoder piece; repair tokenizer.json")
            })?;
            let output = decoder.step(piece);
            let mut joined = std::mem::take(&mut self.pending);
            joined.extend_from_slice(&output);
            let mut rest = joined.as_slice();
            loop {
                match std::str::from_utf8(rest) {
                    Ok(_) => break,
                    Err(error) => {
                        rest = &rest[error.valid_up_to()..];
                        if let Some(length) = error.error_len() {
                            rest = &rest[length..];
                        } else {
                            self.pending.extend_from_slice(rest);
                            break;
                        }
                    }
                }
            }
        }
        Ok(self.complete())
    }
}
struct State {
    history: Vec<TokenId>,
    checkpoints: Vec<Machine>,
    bytes: BytesState,
}
pub(crate) struct BudgetProcessor {
    budget: ReasoningBudget,
    state: Mutex<State>,
}
impl BudgetProcessor {
    fn synchronize(&self, state: &mut State, history: &[TokenId]) -> anyhow::Result<()> {
        let shared = state
            .history
            .iter()
            .zip(history)
            .take_while(|(left, right)| left == right)
            .count();
        if shared < state.history.len() {
            state.history.truncate(shared);
            state.checkpoints.truncate(shared + 1);
            state.bytes = BytesState::new(&self.budget.contract);
            for &id in &state.history {
                state.bytes.accept(&self.budget.contract, id)?;
            }
        }
        for &id in &history[state.history.len()..] {
            let complete = state.bytes.accept(&self.budget.contract, id)?;
            let mut machine = state
                .checkpoints
                .last()
                .expect("initial reasoning checkpoint")
                .clone();
            machine.accept(&self.budget.contract, id, complete, true)?;
            state.history.push(id);
            state.checkpoints.push(machine);
        }
        let complete = state.bytes.complete();
        state
            .checkpoints
            .last_mut()
            .expect("initial reasoning checkpoint")
            .control(
                &self.budget.contract,
                self.budget.control.revision(),
                complete,
            );
        Ok(())
    }
}
impl LogitProcessor for BudgetProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) -> anyhow::Result<()> {
        let mut state = self.state.lock().map_err(|_| {
            anyhow::anyhow!("reasoning budget state is poisoned; restart this completion")
        })?;
        let result = (|| {
            self.synchronize(&mut state, &context.generated_tokens)?;
            if let Some(id) = state
                .checkpoints
                .last()
                .expect("initial reasoning checkpoint")
                .forced(&self.budget.contract)
            {
                let value=*logits.get(id as usize).with_context(||format!("reasoning closer token {id} exceeds the {}-token logits vocabulary; repair the model/tokenizer contract",logits.len()))?;
                ensure!(
                    !value.is_nan() && value != f32::NEG_INFINITY,
                    "reasoning closer token {id} is rejected by prior logits/bias/grammar; allow the declared closing protocol or adjust the conflicting constraint"
                );
                logits.fill(f32::NEG_INFINITY);
                logits[id as usize] = value;
            }
            Ok(())
        })();
        if result.is_err() {
            *state = self.budget.initial_state()?;
        }
        result
    }
    fn name(&self) -> &str {
        "reasoning_budget"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::GenerateOptions, processors::build_processor_chain};
    use serde_json::{Value, json};

    fn tokenizer(strip: bool) -> Tokenizer {
        let mut value: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/tiny-llm/tokenizer.json"
        )))
        .unwrap();
        let vocab = value["model"]["vocab"].as_object_mut().unwrap();
        for (id, piece) in [
            (4, "<0xC3>"),
            (5, "<0xA9>"),
            (8, "<0xFF>"),
            (22, "OPEN"),
            (23, "END"),
            (24, "TAIL"),
            (25, "message"),
        ] {
            let old = vocab
                .iter()
                .find(|(_, value)| value.as_u64() == Some(id))
                .unwrap()
                .0
                .clone();
            vocab.remove(&old);
            assert!(vocab.insert(piece.into(), json!(id)).is_none());
        }
        value["model"] = json!({"type":"BPE","vocab":value["model"]["vocab"],"merges":[],"unk_token":"<unk>","byte_fallback":true});
        value["decoder"] = if strip {
            json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":"é","start":1,"stop":0}]})
        } else {
            json!({"type":"ByteFallback"})
        };
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("tokenizer.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        Tokenizer::from_file(path).unwrap()
    }
    fn spec(limit: Option<usize>) -> ReasoningBudgetSpec {
        ReasoningBudgetSpec {
            start_sequences: vec![vec![22]],
            initially_active: false,
            end_sequences: vec![vec![23, 24]],
            forced_tokens: Vec::new(),
            prefix_tokens: vec![22],
            max_tokens: limit,
        }
    }
    fn mask(processor: &BudgetProcessor, history: &[u32]) -> Vec<f32> {
        let mut logits = vec![1.0; 32];
        processor
            .process(
                &mut logits,
                &ProcessorContext {
                    generated_tokens: history.into(),
                    ..Default::default()
                },
            )
            .unwrap();
        logits
    }
    fn forced(mask: Vec<f32>, id: u32) {
        assert_eq!(
            mask.iter()
                .enumerate()
                .filter(|(_, value)| value.is_finite())
                .map(|(index, _)| index as u32)
                .collect::<Vec<_>>(),
            vec![id]
        );
    }

    #[test]
    fn reasoning_budget_zero_forces_a_multitoken_closer_after_prefill() {
        let budget = ReasoningBudget::resolve(&tokenizer(false), spec(Some(0))).unwrap();
        let processor = budget.processor().unwrap();
        forced(mask(&processor, &[]), 23);
        forced(mask(&processor, &[23]), 24);
        assert!(
            mask(&processor, &[23, 24])
                .iter()
                .all(|value| value.is_finite())
        );
        forced(mask(&processor, &[]), 23);
    }
    #[test]
    fn reasoning_budget_counts_tokens_allows_natural_end_and_rearms() {
        let budget = ReasoningBudget::resolve(&tokenizer(false), spec(Some(2))).unwrap();
        let processor = budget.processor().unwrap();
        assert!(mask(&processor, &[]).iter().all(|value| value.is_finite()));
        assert!(mask(&processor, &[6]).iter().all(|value| value.is_finite()));
        forced(mask(&processor, &[6, 7]), 23);
        assert!(
            mask(&processor, &[23, 24])
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            mask(&processor, &[23, 24, 22, 6])
                .iter()
                .all(|value| value.is_finite())
        );
        forced(mask(&processor, &[23, 24, 22, 6, 7]), 23);
    }
    #[test]
    fn reasoning_budget_does_not_consume_ungenerated_forcing_tokens_in_existing_prefill() {
        let mut spec = spec(Some(1));
        spec.prefix_tokens.extend([6, 7, 9]);
        let budget = ReasoningBudget::resolve(&tokenizer(false), spec.clone()).unwrap();
        forced(mask(&budget.processor().unwrap(), &[]), 23);
        spec.prefix_tokens.extend([23, 24]);
        let budget = ReasoningBudget::resolve(&tokenizer(false), spec).unwrap();
        assert!(
            mask(&budget.processor().unwrap(), &[])
                .iter()
                .all(|value| value.is_finite())
        );
    }
    #[test]
    fn reasoning_budget_waits_for_split_utf8_and_contextual_prefix_stripping() {
        for strip in [false, true] {
            let budget = ReasoningBudget::resolve(&tokenizer(strip), spec(Some(1))).unwrap();
            let processor = budget.processor().unwrap();
            assert!(mask(&processor, &[4]).iter().all(|value| value.is_finite()));
            forced(mask(&processor, &[4, 5]), 23);
            forced(mask(&processor, &[4, 5, 23]), 24);
        }
        let budget = ReasoningBudget::resolve(&tokenizer(false), spec(Some(1))).unwrap();
        forced(mask(&budget.processor().unwrap(), &[8]), 23);
    }
    #[test]
    fn reasoning_budget_manual_control_reaches_independent_branches_and_survives_rewinds() {
        let budget = ReasoningBudget::resolve(&tokenizer(false), spec(None)).unwrap();
        let first = budget.processor().unwrap();
        let second = budget.processor().unwrap();
        assert!(mask(&first, &[6]).iter().all(|value| value.is_finite()));
        budget.control().request_end().unwrap();
        forced(mask(&first, &[6]), 23);
        forced(mask(&second, &[6]), 23);
        forced(mask(&first, &[6, 23]), 24);
        forced(mask(&first, &[6]), 23);
        forced(mask(&first, &[]), 23);
        forced(mask(&first, &[23]), 24);
        assert!(
            mask(&first, &[23, 24])
                .iter()
                .all(|value| value.is_finite())
        );
    }
    #[test]
    fn reasoning_budget_manual_control_ignores_idle_but_waits_for_pending_utf8() {
        let mut input = spec(None);
        input.prefix_tokens.clear();
        let budget = ReasoningBudget::resolve(&tokenizer(false), input).unwrap();
        let processor = budget.processor().unwrap();
        budget.control().request_end().unwrap();
        assert!(mask(&processor, &[]).iter().all(|value| value.is_finite()));
        assert!(
            mask(&processor, &[22])
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            mask(&processor, &[22, 4])
                .iter()
                .all(|value| value.is_finite())
        );
        budget.control().request_end().unwrap();
        assert!(
            mask(&processor, &[22, 4])
                .iter()
                .all(|value| value.is_finite())
        );
        forced(mask(&processor, &[22, 4, 5]), 23);
    }
    #[test]
    fn reasoning_budget_forced_message_and_history_errors_preserve_the_contract() {
        let mut input = spec(Some(0));
        input.forced_tokens = vec![25, 23, 24];
        let budget = ReasoningBudget::resolve(&tokenizer(false), input).unwrap();
        let processor = budget.processor().unwrap();
        forced(mask(&processor, &[]), 25);
        forced(mask(&processor, &[25]), 23);
        let error = processor
            .process(
                &mut [0.0; 32],
                &ProcessorContext {
                    generated_tokens: vec![6],
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(format!("{error:#}").contains("history token 6"));
        forced(mask(&processor, &[]), 25);
        let mut logits = vec![0.0; 32];
        logits[25] = f32::NEG_INFINITY;
        let error = processor
            .process(&mut logits, &ProcessorContext::default())
            .unwrap_err();
        assert!(format!("{error:#}").contains("rejected by prior logits/bias/grammar"));
        forced(mask(&processor, &[]), 25);
    }
    #[test]
    fn reasoning_budget_processor_is_mandatory_for_greedy_and_disables_device_shortcuts() {
        let tokenizer = tokenizer(false);
        let options = GenerateOptions {
            greedy: true,
            reasoning_budget: Some(ReasoningBudget::resolve(&tokenizer, spec(Some(0))).unwrap()),
            ..Default::default()
        };
        let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
        assert!(chain.names().contains(&"reasoning_budget"));
        assert!(!chain.preserves_argmax());
        let mut logits = vec![1.0; 32];
        chain
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        forced(logits, 23);
    }
    #[test]
    fn reasoning_budget_rejects_missing_protocol_tokens_and_incomplete_prefill() {
        let tokenizer = tokenizer(false);
        for mutate in [0, 1, 2, 3, 4] {
            let mut input = spec(Some(1));
            match mutate {
                0 => input.start_sequences.clear(),
                1 => input.end_sequences = vec![vec![]],
                2 => input.forced_tokens = vec![6],
                3 => input.end_sequences = vec![vec![100]],
                _ => input.prefix_tokens.push(4),
            }
            assert!(
                ReasoningBudget::resolve(&tokenizer, input).is_err(),
                "case {mutate}"
            );
        }
        let control = ReasoningControl::default();
        control.0.store(u64::MAX, Ordering::Release);
        assert!(
            control
                .request_end()
                .unwrap_err()
                .to_string()
                .contains("overflow")
        );
    }
    #[test]
    fn reasoning_budget_token_matcher_uses_longest_suffix_and_restarts_after_a_match() {
        let matcher = Matcher::new(vec![vec![1, 2], vec![2], vec![1, 2, 3]]);
        let mut cursor = 0;
        assert_eq!(matcher.advance(&mut cursor, 1), None);
        assert_eq!(matcher.advance(&mut cursor, 2), Some(0));
        assert_eq!(cursor, 0);
        assert_eq!(matcher.advance(&mut cursor, 3), None);
        assert_eq!(matcher.advance(&mut cursor, 2), Some(1));
        let matcher = Matcher::new(vec![vec![1, 1, 2]]);
        let mut cursor = 0;
        for id in [1, 1, 1] {
            assert_eq!(matcher.advance(&mut cursor, id), None);
        }
        assert_eq!(matcher.advance(&mut cursor, 2), Some(0));
    }
    #[test]
    fn reasoning_budget_supports_an_explicit_implicit_channel_without_an_opener() {
        let mut input = spec(Some(0));
        input.start_sequences.clear();
        input.prefix_tokens.clear();
        input.initially_active = true;
        let budget = ReasoningBudget::resolve(&tokenizer(false), input.clone()).unwrap();
        forced(mask(&budget.processor().unwrap(), &[]), 23);
        input.max_tokens = Some(1);
        let budget = ReasoningBudget::resolve(&tokenizer(false), input).unwrap();
        let processor = budget.processor().unwrap();
        assert!(mask(&processor, &[]).iter().all(|value| value.is_finite()));
        forced(mask(&processor, &[6]), 23);
    }
    #[test]
    fn reasoning_budget_rejects_banned_closing_tokens_before_generation() {
        let tokenizer = tokenizer(false);
        let mut options = GenerateOptions {
            reasoning_budget: Some(ReasoningBudget::resolve(&tokenizer, spec(None)).unwrap()),
            ..Default::default()
        };
        options.logit_bias.insert(23, crate::logits::LogitBias::Ban);
        let error = options.validate_processors(&tokenizer).unwrap_err();
        assert!(format!("{error:#}").contains("closer token 23 is banned"));
    }
    #[test]
    fn reasoning_budget_generation_prefix_is_consumed_by_regex_and_lark_grammars() {
        let tokenizer = tokenizer(false);
        for constraint in [
            crate::config::GenerateConstraint::Regex {
                pattern: "OPENEND".into(),
                prefix: "OPEN".into(),
            },
            crate::config::GenerateConstraint::Lark {
                grammar: "start: \"OPENEND\"".into(),
                prefix: "OPEN".into(),
            },
        ] {
            let options = GenerateOptions {
                constraint: Some(constraint),
                eos_token_id: Some(3),
                ..Default::default()
            };
            let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
            let mut logits = vec![1.0; 32];
            chain
                .process(&mut logits, &ProcessorContext::default())
                .unwrap();
            assert!(logits[23].is_finite());
            assert!(!logits[24].is_finite());
            let mut logits = vec![1.0; 32];
            chain
                .process(
                    &mut logits,
                    &ProcessorContext {
                        generated_tokens: vec![23],
                        ..Default::default()
                    },
                )
                .unwrap();
            assert!(logits[3].is_finite());
        }
    }
}
