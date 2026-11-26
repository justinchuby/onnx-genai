//! Declared response boundaries. No model identity determines a delimiter.
use anyhow::{Context, ensure};
use regex_automata::{
    Anchored, Input,
    dfa::{Automaton, StartKind, dense},
    meta,
};
use std::ops::Range;

const REGEX_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug)]
struct Boundary {
    regex: meta::Regex,
    partial: PartialBoundary,
    minimum_bytes: Option<usize>,
}

#[derive(Debug)]
struct ConstraintBoundaryRefusal {
    field: String,
    cause: std::sync::Arc<anyhow::Error>,
}
impl std::fmt::Display for ConstraintBoundaryRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "response_template field {:?} cannot resolve its regex byte-prefix boundary for constrained sampling: {:#}; simplify the regex or remove unsupported look assertions",
            self.field, self.cause
        )
    }
}
impl std::error::Error for ConstraintBoundaryRefusal {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.cause.as_ref().as_ref())
    }
}

/// Reachability is computed once from the actual anchored DFA, including every
/// look-behind class. A delayed accepting state alone does not keep a boundary
/// open: only a possible later acceptance or an EOF-dependent match does.
#[derive(Debug)]
struct ByteBoundary {
    dfa: dense::DFA<Vec<u32>>,
    live: Vec<bool>,
}

#[derive(Debug)]
enum PartialBoundary {
    Literal(Vec<String>),
    Exact(Box<ByteBoundary>),
    Buffered(std::sync::Arc<anyhow::Error>),
}

impl ByteBoundary {
    fn new(dfa: dense::DFA<Vec<u32>>) -> anyhow::Result<Self> {
        use regex_automata::util::{primitives::StateID, start};
        // State IDs are documented premultiplied transition-table offsets.
        let slots = dfa.memory_usage() / (dfa.stride() * size_of::<u32>()) + 1;
        let alphabet = dfa
            .byte_classes()
            .representatives(..)
            .filter_map(|unit| unit.as_u8())
            .collect::<Vec<_>>();
        let edge_limit = slots
            .checked_mul(alphabet.len())
            .context("response boundary transition count overflow")?;
        let bytes = slots
            .checked_mul(2 * size_of::<usize>() + 3 + size_of::<StateID>())
            .and_then(|base| {
                edge_limit
                    .checked_mul(2 * size_of::<usize>())
                    .and_then(|edges| base.checked_add(edges))
            })
            .context("response boundary reachability budget overflow")?;
        ensure!(
            bytes <= 8 * REGEX_BYTES,
            "response boundary reachability needs {bytes} bytes, exceeding its {}-byte analysis budget; simplify the boundary regex",
            8 * REGEX_BYTES
        );
        let index = |state: StateID| state.as_usize() >> dfa.stride2();
        let mut seen = vec![false; slots];
        let mut states = Vec::with_capacity(slots);
        for previous in std::iter::once(None).chain((0..=u8::MAX).map(Some)) {
            let state = dfa
                .start_state(
                    &start::Config::new()
                        .anchored(Anchored::Yes)
                        .look_behind(previous),
                )
                .context("resolve anchored response boundary look-behind")?;
            ensure!(
                index(state) < slots,
                "response boundary start state exceeds its transition table"
            );
            if !seen[index(state)] {
                seen[index(state)] = true;
                states.push(state);
            }
        }
        let mut heads = vec![usize::MAX; slots];
        let mut parents = Vec::with_capacity(edge_limit);
        let mut at = 0;
        while at < states.len() {
            let state = states[at];
            at += 1;
            ensure!(
                !dfa.is_quit_state(state),
                "response boundary DFA cannot resolve this byte context; use a regex without unsupported look assertions"
            );
            if dfa.is_dead_state(state) {
                continue;
            }
            for &byte in &alphabet {
                let next = dfa.next_state(state, byte);
                ensure!(
                    !dfa.is_quit_state(next),
                    "response boundary DFA cannot resolve byte {byte:#04x}; use a regex without unsupported look assertions"
                );
                if dfa.is_dead_state(next) {
                    continue;
                }
                let dest = index(next);
                ensure!(
                    dest < slots && parents.len() < edge_limit,
                    "response boundary transitions exceed their resolved analysis capacity"
                );
                parents.push((index(state), heads[dest]));
                heads[dest] = parents.len() - 1;
                if !seen[dest] {
                    seen[dest] = true;
                    states.push(next);
                }
            }
        }
        let mut finish = vec![false; slots];
        let mut queue = Vec::with_capacity(states.len());
        for &state in &states {
            let ending = dfa.next_eoi_state(state);
            ensure!(
                !dfa.is_quit_state(ending),
                "response boundary DFA cannot resolve EOF in this context; simplify the boundary look assertions"
            );
            if dfa.is_match_state(state) || dfa.is_match_state(ending) {
                finish[index(state)] = true;
                queue.push(index(state));
            }
        }
        let mut at = 0;
        while at < queue.len() {
            let mut edge = heads[queue[at]];
            at += 1;
            while edge != usize::MAX {
                let (parent, next) = parents[edge];
                if !finish[parent] {
                    finish[parent] = true;
                    queue.push(parent);
                }
                edge = next;
            }
        }
        let mut live = vec![false; slots];
        for &state in &states {
            live[index(state)] = dfa.is_match_state(dfa.next_eoi_state(state))
                || alphabet
                    .iter()
                    .any(|&byte| finish[index(dfa.next_state(state, byte))]);
        }
        Ok(Self { dfa, live })
    }

    fn holds(&self, input: &str, pending: &[u8], from: usize) -> anyhow::Result<Option<usize>> {
        for start in input[from..]
            .char_indices()
            .map(|(at, _)| from + at)
            .chain(std::iter::once(input.len()))
        {
            let query = Input::new(input)
                .span(start..input.len())
                .anchored(Anchored::Yes);
            let mut state = self
                .dfa
                .start_state_forward(&query)
                .context("start response boundary byte-prefix scan")?;
            for &byte in input.as_bytes()[start..].iter().chain(pending) {
                state = self.dfa.next_state(state, byte);
                ensure!(
                    !self.dfa.is_quit_state(state),
                    "response boundary byte-prefix scan reached an unresolved DFA state"
                );
                if self.dfa.is_dead_state(state) {
                    break;
                }
            }
            if self.live[state.as_usize() >> self.dfa.stride2()] {
                return Ok(Some(start));
            }
        }
        Ok(None)
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Scan {
    pub matched: Option<Range<usize>>,
    pub hold_from: usize,
}

impl Boundary {
    pub fn literal(mut literals: Vec<String>, field: &str) -> anyhow::Result<Self> {
        ensure!(
            !literals.is_empty() && literals.iter().all(|literal| !literal.is_empty()),
            "{field} requires nonempty literal delimiters; remove an absent boundary instead of declaring an empty string or list"
        );
        let mut seen = std::collections::HashSet::new();
        literals.retain(|literal| seen.insert(literal.clone()));
        let mut matching_order = literals.iter().collect::<Vec<_>>();
        matching_order.sort_by_key(|literal| std::cmp::Reverse(literal.chars().count()));
        let pattern = matching_order
            .iter()
            .map(|literal| regex::escape(literal))
            .collect::<Vec<_>>()
            .join("|");
        let minimum_bytes = literals.iter().map(String::len).min();
        Ok(Self {
            regex: Self::regex(&pattern, field)?,
            partial: PartialBoundary::Literal(literals),
            minimum_bytes,
        })
    }

    fn regex(pattern: &str, field: &str) -> anyhow::Result<meta::Regex> {
        meta::Regex::builder()
            .configure(meta::Config::new().nfa_size_limit(Some(REGEX_BYTES))
                .dfa_size_limit(Some(REGEX_BYTES)).hybrid_cache_capacity(REGEX_BYTES))
            .syntax(regex_automata::util::syntax::Config::new()
                .dot_matches_new_line(true).multi_line(false))
            .build(pattern)
            .with_context(|| format!("{field} regex is unsupported; repair the pattern using Unicode-aware regular expressions without lookaround or backreferences"))
    }

    pub fn pattern(pattern: &str, field: &str) -> anyhow::Result<Self> {
        let regex = Self::regex(pattern, field)?;
        let hir = regex_syntax::ParserBuilder::new()
            .dot_matches_new_line(true)
            .multi_line(false)
            .build()
            .parse(pattern)
            .with_context(|| format!("parse {field} regex structure"))?;
        let partial = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .start_kind(StartKind::Anchored)
                    .dfa_size_limit(Some(REGEX_BYTES)),
            )
            .syntax(
                regex_automata::util::syntax::Config::new()
                    .dot_matches_new_line(true)
                    .multi_line(false),
            )
            .build(pattern)
            .map_err(anyhow::Error::new)
            .and_then(ByteBoundary::new);
        let partial = match partial {
            Ok(matcher) => PartialBoundary::Exact(Box::new(matcher)),
            Err(error) => PartialBoundary::Buffered(std::sync::Arc::new(error)),
        };
        Ok(Self {
            regex,
            partial,
            minimum_bytes: hir.properties().minimum_len(),
        })
    }

    /// Authored order is significant when a controller must choose one closer;
    /// scanning still uses longest-first regex priority independently.
    pub fn literal_alternatives(&self) -> Option<&[String]> {
        match &self.partial {
            PartialBoundary::Literal(literals) => Some(literals),
            _ => None,
        }
    }

    pub fn validate_literal(&self, literal: &str) -> anyhow::Result<()> {
        ensure!(
            !literal.is_empty()
                && self
                    .regex
                    .find(literal)
                    .is_some_and(|matched| matched.range() == (0..literal.len())),
            "reasoning delimiter {literal:?} does not fully match its declared response_template boundary; use one declared delimiter or repair the template"
        );
        Ok(())
    }

    pub fn can_match_empty(&self) -> bool {
        self.minimum_bytes == Some(0)
    }

    fn literal_hold(
        &self,
        input: &str,
        pending: &[u8],
        from: usize,
        literals: &[String],
    ) -> Option<usize> {
        input[from..]
            .char_indices()
            .map(|(at, _)| from + at)
            .chain(std::iter::once(input.len()))
            .find(|&start| {
                let part = &input.as_bytes()[start..];
                literals.iter().any(|literal| {
                    let bytes = literal.as_bytes();
                    let length = part.len() + pending.len();
                    length > 0
                        && length < bytes.len()
                        && bytes.starts_with(part)
                        && bytes[part.len()..].starts_with(pending)
                })
            })
    }

    fn hold(&self, input: &str, pending: &[u8], from: usize) -> anyhow::Result<Option<usize>> {
        match &self.partial {
            PartialBoundary::Literal(literals) => {
                Ok(self.literal_hold(input, pending, from, literals))
            }
            PartialBoundary::Exact(matcher) => matcher.holds(input, pending, from),
            PartialBoundary::Buffered(_) => Ok(Some(from)),
        }
    }

    fn scan_pending(
        &self,
        input: &str,
        pending: &[u8],
        from: usize,
        final_chunk: bool,
    ) -> anyhow::Result<Scan> {
        ensure!(
            input.is_char_boundary(from),
            "response parser cursor {from} is outside a UTF-8 boundary; restart this response parser"
        );
        let bytes = if pending.is_empty() {
            std::borrow::Cow::Borrowed(input.as_bytes())
        } else {
            std::borrow::Cow::Owned([input.as_bytes(), pending].concat())
        };
        // EOF/look assertions must see pending bytes too, even though those
        // bytes cannot yet be published as a complete Unicode scalar.
        let found = self
            .regex
            .find(Input::new(bytes.as_ref()).span(from..bytes.len()))
            .map(|matched| matched.range())
            .filter(|range| {
                range.end <= input.len()
                    && input.is_char_boundary(range.start)
                    && input.is_char_boundary(range.end)
            });
        if final_chunk {
            return Ok(Scan {
                matched: found,
                hold_from: input.len(),
            });
        }
        let hold = self.hold(input, pending, from)?.unwrap_or(input.len());
        Ok(Scan {
            matched: found.filter(|range| range.start < hold),
            hold_from: hold,
        })
    }

    #[cfg(test)]
    fn scan_text(&self, input: &str, from: usize, final_chunk: bool) -> anyhow::Result<Scan> {
        self.scan_pending(input, &[], from, final_chunk)
    }
}

#[derive(Debug)]
struct Field {
    name: String,
    open: Option<Boundary>,
    close: Option<Boundary>,
    strip: bool,
    repeats: bool,
    join: String,
    optional: bool,
}

#[derive(Debug, thiserror::Error)]
#[error(
    "response ends with {bytes} incomplete UTF-8 bytes at raw offset {offset}; supply their continuation bytes before finishing the response"
)]
pub(crate) struct IncompleteResponseUtf8 {
    pub(crate) bytes: usize,
    pub(crate) offset: usize,
}
#[derive(Debug, thiserror::Error)]
#[error(
    "response_template required field {name:?} was not generated; verify the prompt prefix and declared boundaries"
)]
pub(crate) struct MissingResponseField {
    name: String,
}
#[derive(Debug, thiserror::Error)]
#[error(
    "response parser buffer would exceed {limit} bytes, including incomplete UTF-8; shorten the assistant prefix or reduce max_tokens"
)]
pub(crate) struct ResponseBufferExceeded {
    pub(crate) limit: usize,
}

/// Validated request-local response framing shared by constraints and tool
/// observation. The rendered assistant prefix is resolved once for both users.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TokenFraming {
    Text,
    Model(std::collections::BTreeSet<u32>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseProjection {
    template: std::sync::Arc<ResponseTemplate>,
    prefix: String,
    max_bytes: usize,
    framing: TokenFraming,
}
impl ResponseProjection {
    pub fn resolve(
        template: std::sync::Arc<ResponseTemplate>,
        prefix: String,
        max_bytes: usize,
        tokenizer: Option<&onnx_genai_ort::Tokenizer>,
    ) -> anyhow::Result<Self> {
        let framing = match tokenizer {
            Some(tokenizer) => TokenFraming::Model(template.framing_tokens(tokenizer)?),
            None => TokenFraming::Text,
        };
        ensure!(
            max_bytes > 0,
            "response projection max_bytes must be positive; configure a nonzero response buffer"
        );
        ResponseParser::new(template.clone(), &prefix, PrefixOutput::Include, max_bytes).context(
            "resolve response projection from the declared template and assistant prefix",
        )?;
        Ok(Self {
            template,
            prefix,
            max_bytes,
            framing,
        })
    }
    /// Token consumers must use the model binding resolved for their tokenizer.
    pub fn validate_tokenizer(&self, tokenizer: &onnx_genai_ort::Tokenizer) -> anyhow::Result<()> {
        let TokenFraming::Model(preserved) = &self.framing else {
            anyhow::bail!(
                "response_projection has a text-only binding; resolve it with Some(model_tokenizer) before token generation"
            );
        };
        ensure!(
            *preserved == self.template.framing_tokens(tokenizer)?,
            "response_projection was resolved for another tokenizer; resolve it using the selected model tokenizer before generation"
        );
        Ok(())
    }
    /// Model-declared framing symbols; text hosts do not interpret token flags.
    pub fn preserved_tokens(&self) -> Option<&std::collections::BTreeSet<u32>> {
        match &self.framing {
            TokenFraming::Text => None,
            TokenFraming::Model(ids) => Some(ids),
        }
    }
    pub fn template(&self) -> &std::sync::Arc<ResponseTemplate> {
        &self.template
    }
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }
    pub(crate) fn project(&self, bytes: &[u8], terminal: bool) -> anyhow::Result<String> {
        let mut parser = ResponseParser::new(
            self.template.clone(),
            &self.prefix,
            PrefixOutput::Include,
            self.max_bytes,
        )?;
        parser.commit_unpublished(bytes)?;
        parser.with_preview(b"", terminal, |parser| Ok(parser.content_text(terminal)))
    }
}

/// Compiled response channels from a declared tokenizer response template.
#[derive(Debug)]
pub struct ResponseTemplate {
    declaration: serde_json::Value,
    anchor: Boundary,
    fields: Vec<Field>,
    implicit: Option<usize>,
    defaults: serde_json::Map<String, serde_json::Value>,
    preserved_tokens: Option<Vec<String>>,
}

/// Declared transitions into reasoning and from reasoning to answer content.
/// An absent opener denotes the template's implicit reasoning channel.
pub struct ReasoningBoundaries<'a> {
    pub start: Option<ReasoningBoundary<'a>>,
    pub end: ReasoningBoundary<'a>,
}

/// A validated channel boundary without exposing the parser's matcher state.
pub struct ReasoningBoundary<'a>(&'a Boundary);
impl ReasoningBoundary<'_> {
    pub fn literal_alternatives(&self) -> Option<&[String]> {
        self.0.literal_alternatives()
    }
    pub fn validate_literal(&self, literal: &str) -> anyhow::Result<()> {
        self.0.validate_literal(literal)
    }
}

impl PartialEq for ResponseTemplate {
    fn eq(&self, other: &Self) -> bool {
        self.declaration == other.declaration
    }
}
impl Eq for ResponseTemplate {}

fn object<'a>(
    value: &'a serde_json::Value,
    scope: &str,
) -> anyhow::Result<&'a serde_json::Map<String, serde_json::Value>> {
    value.as_object().with_context(|| {
        format!("{scope} must be an object; repair the response template declaration")
    })
}
fn boolean(
    map: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: bool,
    scope: &str,
) -> anyhow::Result<bool> {
    map.get(key).map_or(Ok(default), |value| {
        value
            .as_bool()
            .with_context(|| format!("{scope}.{key} must be a boolean"))
    })
}
fn boundary(
    map: &serde_json::Map<String, serde_json::Value>,
    literal: &str,
    pattern: &str,
    scope: &str,
) -> anyhow::Result<Option<Boundary>> {
    ensure!(
        !(map.contains_key(literal) && map.contains_key(pattern)),
        "{scope} cannot declare both {literal} and {pattern}; choose one boundary form"
    );
    if let Some(value) = map.get(literal) {
        let values = if let Some(text) = value.as_str() {
            vec![text.to_owned()]
        } else if let Some(values) = value.as_array() {
            values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(str::to_owned)
                        .with_context(|| format!("{scope}.{literal} must contain only strings"))
                })
                .collect::<anyhow::Result<Vec<_>>>()?
        } else {
            anyhow::bail!(
                "{scope}.{literal} must be a string or string list; repair this boundary"
            );
        };
        return Boundary::literal(values, &format!("{scope}.{literal}")).map(Some);
    }
    map.get(pattern)
        .map(|value| {
            let pattern_text = value
                .as_str()
                .with_context(|| format!("{scope}.{pattern} must be a regex string"))?;
            Boundary::pattern(pattern_text, &format!("{scope}.{pattern}"))
        })
        .transpose()
}

impl ResponseTemplate {
    /// Resolve the model's literal framing symbols once, independently of any
    /// caller's extra preserved tokens. Regex fragments may be declared explicitly.
    pub fn framing_tokens(
        &self,
        tokenizer: &onnx_genai_ort::Tokenizer,
    ) -> anyhow::Result<std::collections::BTreeSet<u32>> {
        let encode = |text: &str| {
            tokenizer.encode_with_options(
                text,
                onnx_genai_ort::TokenizationOptions {
                    add_special: false,
                    parse_special: true,
                },
            )
        };
        let decode = |ids: &[u32]| {
            onnx_genai_llguidance::panic_utils::catch_unwind(std::panic::AssertUnwindSafe(|| {
                tokenizer.decode_with_special_tokens(ids).map_err(anyhow::Error::from)
            })).context("resolve response_template framing through tokenizer.decoder; repair tokenizer.json if decoding failed")
        };
        let mut ids = std::collections::BTreeSet::new();
        if let Some(spellings) = &self.preserved_tokens {
            for (index, spelling) in spellings.iter().enumerate() {
                let encoded = match tokenizer.token_id(spelling) {
                    Some(id) => vec![id],
                    None => {
                        let ids = encode(spelling)?;
                        ensure!(
                            decode(&ids)? == *spelling,
                            "response_template.preserved_tokens[{index}] {spelling:?} is not a declared model token; use an existing vocabulary spelling"
                        );
                        ids
                    }
                };
                ensure!(
                    encoded.len() == 1,
                    "response_template.preserved_tokens[{index}] must identify one model token; use individual registered token spellings"
                );
                ids.insert(encoded[0]);
            }
            return Ok(ids);
        }
        for boundary in self
            .fields
            .iter()
            .flat_map(|field| [field.open.as_ref(), field.close.as_ref()])
            .flatten()
        {
            if let Some(literals) = boundary.literal_alternatives() {
                for literal in literals {
                    let encoded = encode(literal)?;
                    if decode(&encoded)? == *literal {
                        ids.extend(encoded);
                    }
                }
            } else {
                let added = tokenizer.inner().get_added_vocabulary();
                for (&id, token) in added.get_added_tokens_decoder() {
                    if !added.is_special_token(&token.content) {
                        continue;
                    }
                    let piece = decode(&[id])?;
                    if let Some(found) = boundary.regex.find(piece.as_bytes())
                        && found.start() == 0
                        && found.end() == piece.len()
                    {
                        ids.insert(id);
                    }
                }
            }
        }
        Ok(ids)
    }

    fn watched(&self, current: Option<usize>) -> Vec<(usize, bool, &Boundary)> {
        let mut watch = Vec::new();
        if let Some(field) = current.filter(|field| Some(*field) != self.implicit) {
            if let Some(close) = &self.fields[field].close {
                watch.push((field, false, close));
            }
        } else {
            for (field, definition) in self.fields.iter().enumerate() {
                if let Some(open) = &definition.open {
                    watch.push((field, true, open));
                }
            }
            if let Some(field) = self.implicit
                && let Some(close) = &self.fields[field].close
            {
                watch.push((field, false, close));
            }
        }
        watch
    }

    fn generation_start(&self, prompt: &str) -> Option<usize> {
        self.anchor
            .regex
            .find_iter(prompt)
            .last()
            .map(|matched| matched.end())
    }

    /// Isolate only the final rendered assistant prefill, using the declared
    /// anchor rather than matching protocol-looking text in old messages.
    pub fn generation_prefix<'a>(&self, prompt: &'a str) -> anyhow::Result<&'a str> {
        let at=self.generation_start(prompt).context("reasoning budget requires the rendered prompt to contain response_template.start_anchor; repair chat rendering or the declared anchor")?;
        Ok(&prompt[at..])
    }

    fn reasoning_field(&self) -> Option<&Field> {
        self.fields
            .iter()
            .find(|field| field.name == "reasoning_content")
    }

    pub fn has_reasoning_channel(&self) -> bool {
        self.reasoning_field().is_some()
    }

    pub fn reasoning_boundaries(&self) -> anyhow::Result<ReasoningBoundaries<'_>> {
        let reasoning=self.reasoning_field()
            .context("reasoning budget requires response_template.fields.reasoning_content; declare the channel or supply an explicit token protocol")?;
        let end=reasoning.close.as_ref().or_else(||self.fields.iter().find(|field|field.name=="content").and_then(|field|field.open.as_ref()))
            .context("reasoning budget needs a declared reasoning close boundary or content open boundary; declare how the response transitions to its answer")?;
        Ok(ReasoningBoundaries {
            start: reasoning.open.as_ref().map(ReasoningBoundary),
            end: ReasoningBoundary(end),
        })
    }

    pub(crate) fn require_constraint_boundaries(&self) -> anyhow::Result<()> {
        for field in &self.fields {
            for boundary in [&field.open, &field.close].into_iter().flatten() {
                if let PartialBoundary::Buffered(reason) = &boundary.partial {
                    return Err(ConstraintBoundaryRefusal {
                        field: field.name.clone(),
                        cause: reason.clone(),
                    }
                    .into());
                }
            }
        }
        Ok(())
    }

    pub fn compile(value: &serde_json::Value) -> anyhow::Result<std::sync::Arc<Self>> {
        let root = object(value, "response_template")?;
        ensure!(
            root.keys().all(|key| matches!(
                key.as_str(),
                "version"
                    | "fields"
                    | "defaults"
                    | "start_anchor"
                    | "start_anchor_pattern"
                    | "preserved_tokens"
            )),
            "response_template contains unsupported keys; use the version-1 response template fields"
        );
        ensure!(
            root.get("version")
                .is_none_or(|value| value.as_u64() == Some(1)),
            "response_template.version must be 1; re-emit a supported response template"
        );
        let preserved_tokens=root.get("preserved_tokens").map(|value|serde_json::from_value::<Vec<String>>(value.clone())
            .context("response_template.preserved_tokens must be a list of individual model-token spellings")).transpose()?;
        let defaults = root
            .get("defaults")
            .map(|value| object(value, "response_template.defaults").cloned())
            .transpose()?
            .unwrap_or_default();
        for key in ["content", "reasoning_content"] {
            ensure!(
                defaults
                    .get(key)
                    .is_none_or(|value| value.is_string() || value.is_null()),
                "response_template.defaults.{key} must be text or null for a chat text channel"
            );
        }
        let anchor=boundary(root,"start_anchor","start_anchor_pattern","response_template")?.context("response_template requires start_anchor or start_anchor_pattern; declare the final assistant-turn boundary")?;
        ensure!(
            !anchor.can_match_empty(),
            "response_template start anchor must consume text; an empty anchor cannot identify the final assistant turn"
        );
        let raw_fields = object(
            root.get("fields")
                .context("response_template requires fields")?,
            "response_template.fields",
        )?;
        ensure!(
            !raw_fields.is_empty(),
            "response_template.fields must not be empty; declare the output channels"
        );
        let mut names = raw_fields.keys().collect::<Vec<_>>();
        names.sort();
        let mut fields = Vec::with_capacity(names.len());
        let mut implicit = None;
        for name in names {
            let scope = format!("response_template.fields.{name}");
            let map = object(&raw_fields[name], &scope)?;
            ensure!(
                map.keys().all(|key| matches!(
                    key.as_str(),
                    "open"
                        | "open_pattern"
                        | "close"
                        | "close_pattern"
                        | "content"
                        | "content_args"
                        | "repeats"
                        | "join"
                        | "optional"
                        | "transform"
                        | "transform_each"
                )),
                "{scope} contains unsupported field keys; repair this declaration"
            );
            let open = boundary(map, "open", "open_pattern", &scope)?;
            ensure!(
                !open.as_ref().is_some_and(Boundary::can_match_empty),
                "{scope}.open must consume text; a zero-width opener cannot make progress"
            );
            let close = boundary(map, "close", "close_pattern", &scope)?;
            if open.is_none() {
                ensure!(
                    implicit.is_none(),
                    "response_template may declare only one implicit field without an opener; add explicit open boundaries to the others"
                );
                implicit = Some(fields.len());
            }
            let text_channel = matches!(name.as_str(), "content" | "reasoning_content");
            let content = map
                .get("content")
                .map_or(Some("text"), serde_json::Value::as_str)
                .with_context(|| format!("{scope}.content must name a content parser"))?;
            ensure!(
                matches!(
                    content,
                    "text"
                        | "int"
                        | "float"
                        | "bool"
                        | "json"
                        | "xml-inline"
                        | "kv-lines"
                        | "python-call-list"
                ),
                "{scope}.content names unsupported parser {content:?}"
            );
            if text_channel {
                ensure!(
                    content == "text",
                    "{scope} must use the text parser for the chat API text channel"
                );
                ensure!(
                    map.get("transform").is_none_or(serde_json::Value::is_null)
                        && !boolean(map, "transform_each", false, &scope)?,
                    "{scope} has a transform that cannot be projected to a chat text channel; use text regions without transforms"
                );
            }
            let args = map
                .get("content_args")
                .map(|value| object(value, &format!("{scope}.content_args")))
                .transpose()?;
            let strip = if text_channel {
                if let Some(args) = args {
                    ensure!(
                        args.keys().all(|key| key == "strip"),
                        "{scope}.content_args supports only strip for text channels"
                    );
                    boolean(args, "strip", true, &format!("{scope}.content_args"))?
                } else {
                    true
                }
            } else {
                false
            };
            let repeats = boolean(map, "repeats", false, &scope)?;
            let join = map
                .get("join")
                .filter(|value| !value.is_null())
                .map(|value| {
                    value
                        .as_str()
                        .map(str::to_owned)
                        .with_context(|| format!("{scope}.join must be a string"))
                })
                .transpose()?;
            ensure!(
                join.is_none() || repeats,
                "{scope}.join requires repeats=true"
            );
            fields.push(Field {
                name: name.to_owned(),
                open,
                close,
                strip,
                repeats,
                join: join.unwrap_or_default(),
                optional: boolean(map, "optional", true, &scope)?,
            });
        }
        Ok(std::sync::Arc::new(Self {
            declaration: value.clone(),
            anchor,
            fields,
            implicit,
            defaults,
            preserved_tokens,
        }))
    }
}

fn response_whitespace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

#[derive(Debug, Clone)]
struct Region {
    bytes: Range<usize>,
    visible: bool,
}
impl Region {
    fn output<'a>(
        &self,
        field: &Field,
        buffer: &'a str,
        omitted_prefix_end: usize,
        preserve_end: bool,
    ) -> &'a str {
        let raw = &buffer[self.bytes.clone()];
        let text = if field.strip {
            if preserve_end {
                raw.trim_start_matches(response_whitespace)
            } else {
                raw.trim_matches(response_whitespace)
            }
        } else {
            raw
        };
        let start = self.bytes.start + (text.as_ptr() as usize - raw.as_ptr() as usize);
        let end = start + text.len();
        let from = start.max(omitted_prefix_end);
        if from >= end { "" } else { &buffer[from..end] }
    }
}

enum ProjectedRegions<'a> {
    None,
    One(Option<&'a Region>),
    Repeated {
        closed: std::slice::Iter<'a, Region>,
        active: Option<&'a Region>,
    },
}
impl<'a> Iterator for ProjectedRegions<'a> {
    type Item = &'a Region;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::None => None,
            Self::One(region) => region.take(),
            Self::Repeated { closed, active } => closed.next().or_else(|| active.take()),
        }
    }
}

// Borrow selected text directly. Grammar consumers need no copies of unrelated
// reasoning or opaque bodies, and publication uses exactly the same projection.
struct FieldChunks<'a> {
    field: Option<&'a Field>,
    regions: ProjectedRegions<'a>,
    buffer: &'a str,
    omitted_prefix_end: usize,
    first: bool,
    default: Option<&'a str>,
    pending_body: Option<&'a str>,
    preserve_end_for: Option<&'a Region>,
}
impl<'a> Iterator for FieldChunks<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(body) = self.pending_body.take() {
            return Some(body);
        }
        if let Some(default) = self.default.take() {
            return Some(default);
        }
        let field = self.field?;
        for region in self.regions.by_ref() {
            if !region.visible {
                continue;
            }
            let body = region.output(
                field,
                self.buffer,
                self.omitted_prefix_end,
                self.preserve_end_for
                    .is_some_and(|active| std::ptr::eq(active, region)),
            );
            if self.first {
                self.first = false;
                return Some(body);
            }
            self.pending_body = Some(body);
            return Some(&field.join);
        }
        None
    }
}

/// Projected chat text, with protocol framing and opaque fields removed.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ChatText {
    pub content: String,
    pub reasoning: String,
}

/// Whether assistant text already rendered by the template belongs to this output.
#[derive(Debug, Clone, Copy)]
pub enum PrefixOutput {
    /// Emit template-rendered assistant text along with generated text.
    Include,
    /// Consume existing text only as parser state for an explicit continuation.
    Exclude,
}

// A valid UTF-8 prefix can end with at most three incomplete bytes.
#[derive(Debug, Clone, Copy, Default)]
enum Utf8Tail {
    #[default]
    Empty,
    One([u8; 1]),
    Two([u8; 2]),
    Three([u8; 3]),
}
impl Utf8Tail {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Empty => &[],
            Self::One(bytes) => bytes,
            Self::Two(bytes) => bytes,
            Self::Three(bytes) => bytes,
        }
    }
    fn from_incomplete(bytes: &[u8]) -> anyhow::Result<Self> {
        match bytes {
            [] => Ok(Self::Empty),
            [a] => Ok(Self::One([*a])),
            [a, b] => Ok(Self::Two([*a, *b])),
            [a, b, c] => Ok(Self::Three([*a, *b, *c])),
            _ => anyhow::bail!(
                "response UTF-8 decoder retained more than three incomplete bytes; restart this parser and report the invalid decoder state"
            ),
        }
    }
}

/// Request-local response classification and append-only text publication.
/// Matching and field visibility are shared by engine and transport consumers.
pub struct ResponseParser {
    spec: std::sync::Arc<ResponseTemplate>,
    buffer: String,
    utf8_tail: Utf8Tail,
    omitted_prefix_end: usize,
    pos: usize,
    current: Option<usize>,
    region: Option<Region>,
    closed: Vec<std::sync::Arc<Vec<Region>>>,
    seen: Vec<bool>,
    sent: ChatText,
    max_bytes: usize,
}

/// A reopened field may reconstruct a published prefix while it remains open.
/// Once finalized, every projection must extend the established prefix.
#[derive(Debug, thiserror::Error)]
#[error(
    "response_template {field} replaced already streamed text; use buffered output or declare repeats with join for append-only fields"
)]
pub(crate) struct NonMonotonicProjection {
    field: String,
}

pub(crate) fn projection_extension<'a>(
    current: &'a str,
    prior: &str,
    final_chunk: bool,
    field: &str,
) -> anyhow::Result<Option<&'a str>> {
    if let Some(tail) = current.strip_prefix(prior) {
        return Ok(Some(tail));
    }
    if !final_chunk && prior.starts_with(current) {
        return Ok(None);
    }
    Err(NonMonotonicProjection {
        field: field.into(),
    }
    .into())
}

impl ResponseParser {
    pub fn new(
        spec: std::sync::Arc<ResponseTemplate>,
        prefix: &str,
        prefix_output: PrefixOutput,
        max_bytes: usize,
    ) -> anyhow::Result<Self> {
        let at = spec.generation_start(prefix).unwrap_or(0);
        let prefix = &prefix[at..];
        ensure!(
            max_bytes > 0,
            "response parser byte budget must be positive; configure a nonzero buffer budget"
        );
        ensure!(
            prefix.len() <= max_bytes,
            "response parser prefix exceeds the {max_bytes}-byte buffer budget; shorten the assistant prefix"
        );
        let mut parser = Self {
            closed: (0..spec.fields.len())
                .map(|_| std::sync::Arc::new(Vec::new()))
                .collect(),
            seen: vec![false; spec.fields.len()],
            current: spec.implicit,
            spec,
            buffer: prefix.into(),
            utf8_tail: Utf8Tail::Empty,
            omitted_prefix_end: match prefix_output {
                PrefixOutput::Include => 0,
                PrefixOutput::Exclude => prefix.len(),
            },
            pos: 0,
            region: None,
            sent: ChatText::default(),
            max_bytes,
        };
        parser.process(false)?;
        Ok(parser)
    }

    pub fn require_open_text_channel(&self, name: &str) -> anyhow::Result<()> {
        let active = self
            .current
            .map(|field| self.spec.fields[field].name.as_str());
        ensure!(
            active == Some(name),
            "continue_final_message requires the rendered prefix to leave response_template field {name:?} open, but the active field is {active:?}; repair the chat/response template boundaries or remove a closing delimiter from the continuation prefix"
        );
        Ok(())
    }

    fn accumulate(&mut self, range: Range<usize>) {
        if range.is_empty() {
            return;
        }
        if let Some(field) = self.current {
            let region = self.region.get_or_insert(Region {
                bytes: range.start..range.start,
                visible: false,
            });
            region.bytes.end = range.end;
            region.visible |= range.end > self.omitted_prefix_end;
            self.seen[field] = true;
        }
    }
    fn close(&mut self) {
        if let (Some(field), Some(region)) = (self.current, self.region.take()) {
            let closed = std::sync::Arc::make_mut(&mut self.closed[field]);
            if !self.spec.fields[field].repeats {
                closed.clear();
            }
            closed.push(region);
        }
        self.current = self.spec.implicit;
    }
    fn process(&mut self, final_chunk: bool) -> anyhow::Result<()> {
        loop {
            let spec = self.spec.clone();
            let watch = spec.watched(self.current);
            let mut hold = self.buffer.len();
            let mut best: Option<(usize, bool, Range<usize>)> = None;
            for (field, opening, boundary) in watch {
                let scan = boundary.scan_pending(
                    &self.buffer,
                    self.utf8_tail.bytes(),
                    self.pos,
                    final_chunk,
                )?;
                hold = hold.min(scan.hold_from);
                if let Some(range) = scan.matched {
                    let key = (range.start, std::cmp::Reverse(range.len()), !opening, field);
                    let wins = best.as_ref().is_none_or(|(field, opening, range)| {
                        key < (
                            range.start,
                            std::cmp::Reverse(range.len()),
                            !*opening,
                            *field,
                        )
                    });
                    if wins {
                        best = Some((field, opening, range));
                    }
                }
            }
            if let Some((field, opening, range)) =
                best.filter(|(_, _, range)| final_chunk || range.start < hold)
            {
                self.accumulate(self.pos..range.start);
                self.pos = range.end;
                if opening {
                    self.close();
                    self.current = Some(field);
                    self.region = Some(Region {
                        bytes: range.end..range.end,
                        visible: range.end > self.omitted_prefix_end,
                    });
                    self.seen[field] = true;
                } else {
                    let had_region = self.region.is_some();
                    self.close();
                    if !had_region && range.is_empty() {
                        break;
                    }
                }
                continue;
            }
            let end = if final_chunk { self.buffer.len() } else { hold };
            self.accumulate(self.pos..end);
            self.pos = end;
            if final_chunk {
                self.close();
            }
            break;
        }
        Ok(())
    }
    fn field_chunks<'a>(&'a self, name: &str, final_chunk: bool) -> FieldChunks<'a> {
        self.field_chunks_mode(name, final_chunk, false)
    }
    fn field_chunks_mode<'a>(
        &'a self,
        name: &str,
        final_chunk: bool,
        preserve_end: bool,
    ) -> FieldChunks<'a> {
        let index = self.spec.fields.iter().position(|field| field.name == name);
        let default = if final_chunk && index.is_none_or(|index| !self.seen[index]) {
            self.spec
                .defaults
                .get(name)
                .and_then(serde_json::Value::as_str)
        } else {
            None
        };
        let field = index.map(|index| &self.spec.fields[index]);
        let regions = index.map_or(ProjectedRegions::None, |index| {
            let active = if self.current == Some(index) {
                self.region.as_ref()
            } else {
                None
            };
            if self.spec.fields[index].repeats {
                ProjectedRegions::Repeated {
                    closed: self.closed[index].iter(),
                    active,
                }
            } else {
                ProjectedRegions::One(active.or_else(|| self.closed[index].last()))
            }
        });
        FieldChunks {
            field,
            regions,
            buffer: &self.buffer,
            omitted_prefix_end: self.omitted_prefix_end,
            first: true,
            default,
            pending_body: None,
            preserve_end_for: if preserve_end && index == self.current {
                self.region.as_ref()
            } else {
                None
            },
        }
    }

    fn snapshot(&self, final_chunk: bool) -> ChatText {
        ChatText {
            content: self.field_chunks("content", final_chunk).collect(),
            reasoning: self
                .field_chunks("reasoning_content", final_chunk)
                .collect(),
        }
    }
    fn delta(&mut self, final_chunk: bool) -> anyhow::Result<ChatText> {
        let now = self.snapshot(final_chunk);
        let content =
            projection_extension(&now.content, &self.sent.content, final_chunk, "content")?
                .map(str::to_owned);
        let reasoning = projection_extension(
            &now.reasoning,
            &self.sent.reasoning,
            final_chunk,
            "reasoning_content",
        )?
        .map(str::to_owned);
        if content.is_some() {
            self.sent.content = now.content;
        }
        if reasoning.is_some() {
            self.sent.reasoning = now.reasoning;
        }
        Ok(ChatText {
            content: content.unwrap_or_default(),
            reasoning: reasoning.unwrap_or_default(),
        })
    }
    fn append_bytes(&mut self, bytes: &[u8]) -> anyhow::Result<()> {
        let pending = self.utf8_tail.bytes();
        if !self
            .buffer
            .len()
            .checked_add(pending.len())
            .and_then(|size| size.checked_add(bytes.len()))
            .is_some_and(|size| size <= self.max_bytes)
        {
            return Err(ResponseBufferExceeded {
                limit: self.max_bytes,
            }
            .into());
        }
        let input = if pending.is_empty() {
            std::borrow::Cow::Borrowed(bytes)
        } else {
            let mut combined = Vec::with_capacity(pending.len() + bytes.len());
            combined.extend_from_slice(pending);
            combined.extend_from_slice(bytes);
            std::borrow::Cow::Owned(combined)
        };
        let (text, tail) = match std::str::from_utf8(&input) {
            Ok(text) => (text, Utf8Tail::Empty),
            Err(error) if error.error_len().is_none() => {
                let at = error.valid_up_to();
                let text = std::str::from_utf8(&input[..at])
                    .context("decode the validated response UTF-8 prefix")?;
                (text, Utf8Tail::from_incomplete(&input[at..])?)
            }
            Err(error) => return Err(anyhow::Error::new(error).context(format!(
                "response byte input contains invalid UTF-8 at raw offset {}; supply a valid UTF-8 byte sequence or decode malformed pieces before response projection",
                self.buffer.len() + error.valid_up_to()
            ))),
        };
        self.buffer.push_str(text);
        self.utf8_tail = tail;
        Ok(())
    }

    fn require_complete_utf8(&self) -> anyhow::Result<()> {
        if !self.utf8_tail.bytes().is_empty() {
            return Err(IncompleteResponseUtf8 {
                bytes: self.utf8_tail.bytes().len(),
                offset: self.buffer.len(),
            }
            .into());
        }
        Ok(())
    }

    /// Return the cumulative projected channels after a candidate, without consuming it.
    /// Unlike `feed`, this returns a complete snapshot and does not publish a delta.
    /// Final candidates also resolve pending boundaries, defaults and required fields.
    /// The original parser state is restored on both success and failure.
    pub fn preview(&mut self, text: &str, final_chunk: bool) -> anyhow::Result<ChatText> {
        self.preview_bytes(text.as_bytes(), final_chunk)
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.buffer.len() + self.utf8_tail.bytes().len()
    }

    /// Inspect lossless byte pieces; partial UTF-8 is retained only within the trial.
    /// Final candidates must end on a complete UTF-8 scalar.
    pub fn preview_bytes(&mut self, bytes: &[u8], final_chunk: bool) -> anyhow::Result<ChatText> {
        self.with_preview(
            bytes,
            final_chunk,
            |parser| Ok(parser.snapshot(final_chunk)),
        )
    }

    pub(crate) fn with_preview<T>(
        &mut self,
        bytes: &[u8],
        final_chunk: bool,
        inspect: impl FnOnce(&Self) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let trial = ResponseTrial::new(self);
        trial.parser.append_bytes(bytes)?;
        if final_chunk {
            trial.parser.require_complete_utf8()?;
        }
        trial.parser.process(final_chunk)?;
        if final_chunk {
            trial.parser.validate_required()?;
        }
        inspect(trial.parser)
    }

    pub(crate) fn commit_unpublished(&mut self, bytes: &[u8]) -> anyhow::Result<()> {
        self.append_bytes(bytes)?;
        self.process(false)
    }

    pub(crate) fn content_text(&self, final_chunk: bool) -> String {
        self.field_chunks("content", final_chunk).collect()
    }

    pub(crate) fn grammar_content_text(&self) -> String {
        self.field_chunks_mode("content", false, true).collect()
    }

    pub(crate) fn pending_can_be_trimmed(&self) -> bool {
        let pending = self.utf8_tail.bytes();
        if pending.is_empty()
            || self.current.is_none_or(|field| {
                self.spec.fields[field].name != "content" || !self.spec.fields[field].strip
            })
        {
            return false;
        }
        static WHITESPACE: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        WHITESPACE
            .get_or_init(|| {
                (0..=u32::from(char::MAX))
                    .filter_map(char::from_u32)
                    .filter(|c| response_whitespace(*c))
                    .map(|c| c.to_string())
                    .collect()
            })
            .iter()
            .any(|c| c.as_bytes().starts_with(pending))
    }

    pub(crate) fn pending_content_bytes(&self) -> anyhow::Result<Option<&[u8]>> {
        let pending = self.utf8_tail.bytes();
        if pending.is_empty()
            || self
                .current
                .is_none_or(|field| self.spec.fields[field].name != "content")
        {
            return Ok(None);
        }
        for (_, _, boundary) in self.spec.watched(self.current) {
            if boundary.hold(&self.buffer, pending, self.pos)?.is_some() {
                return Ok(None);
            }
        }
        Ok(Some(pending))
    }

    pub fn feed(&mut self, text: &str) -> anyhow::Result<ChatText> {
        self.feed_bytes(text.as_bytes())
    }

    /// Consume strict UTF-8 byte pieces, retaining incomplete scalars between calls.
    /// Invalid sequences fail without appending any bytes from the rejected piece.
    pub fn feed_bytes(&mut self, bytes: &[u8]) -> anyhow::Result<ChatText> {
        self.append_bytes(bytes)?;
        self.process(false)?;
        self.delta(false)
    }

    fn validate_required(&self) -> anyhow::Result<()> {
        for (field, definition) in self.spec.fields.iter().enumerate() {
            if !(definition.optional
                || self.seen[field]
                || self.spec.defaults.contains_key(&definition.name))
            {
                return Err(MissingResponseField {
                    name: definition.name.clone(),
                }
                .into());
            }
        }
        Ok(())
    }

    pub fn finish(mut self) -> anyhow::Result<ChatText> {
        self.require_complete_utf8()?;
        self.process(true)?;
        self.validate_required()?;
        self.delta(true)
    }
}

struct ResponseCheckpoint {
    buffer_len: usize,
    utf8_tail: Utf8Tail,
    pos: usize,
    current: Option<usize>,
    region: Option<Region>,
    closed: Vec<std::sync::Arc<Vec<Region>>>,
    seen: Vec<bool>,
}

struct ResponseTrial<'a> {
    parser: &'a mut ResponseParser,
    checkpoint: ResponseCheckpoint,
}

impl<'a> ResponseTrial<'a> {
    fn new(parser: &'a mut ResponseParser) -> Self {
        let checkpoint = ResponseCheckpoint {
            buffer_len: parser.buffer.len(),
            utf8_tail: parser.utf8_tail,
            pos: parser.pos,
            current: parser.current,
            region: parser.region.clone(),
            closed: parser.closed.clone(),
            seen: parser.seen.clone(),
        };
        Self { parser, checkpoint }
    }
}

impl Drop for ResponseTrial<'_> {
    fn drop(&mut self) {
        let saved = &mut self.checkpoint;
        self.parser.buffer.truncate(saved.buffer_len);
        self.parser.utf8_tail = saved.utf8_tail;
        self.parser.pos = saved.pos;
        self.parser.current = saved.current;
        self.parser.region = saved.region.take();
        self.parser.closed = std::mem::take(&mut saved.closed);
        self.parser.seen = std::mem::take(&mut saved.seen);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn template(repeats: bool) -> std::sync::Arc<ResponseTemplate> {
        ResponseTemplate::compile(&serde_json::json!({
            "start_anchor":"[assistant]",
            "fields":{
                "reasoning_content":{"open":"<r>","close_pattern":"</r>\\s*","repeats":repeats},
                "content":{"close":"[end]"},
                "private":{"open":"<tool>","close":"</tool>","content":"json"}
            }
        }))
        .unwrap()
    }
    fn collected(
        spec: std::sync::Arc<ResponseTemplate>,
        prefix: &str,
        chunks: &[&str],
    ) -> anyhow::Result<ChatText> {
        let mut parser = ResponseParser::new(spec, prefix, PrefixOutput::Exclude, 16384)?;
        let mut output = ChatText::default();
        for chunk in chunks {
            let delta = parser.feed(chunk)?;
            output.content.push_str(&delta.content);
            output.reasoning.push_str(&delta.reasoning);
        }
        let delta = parser.finish()?;
        output.content.push_str(&delta.content);
        output.reasoning.push_str(&delta.reasoning);
        Ok(output)
    }
    #[test]
    fn borrowed_field_chunks_preserve_empty_repeats_joins_and_opaque_projection() {
        let spec = ResponseTemplate::compile(&serde_json::json!({"start_anchor":"A","fields":{
            "content":{"open":"<c>","close":"</c>","repeats":true,"join":"|"},
            "reasoning_content":{"open":"<r>","close":"</r>"},
            "private":{"open":"<tool>","close":"</tool>","content":"json"}
        }}))
        .unwrap();
        let mut parser = ResponseParser::new(spec, "A", PrefixOutput::Include, 1024).unwrap();
        let candidate = b"<c></c><r>secret</r><c>visible</c><tool>{}</tool>";
        let selected = parser
            .with_preview(candidate, true, |view| {
                let chunks = view.field_chunks("content", true).collect::<Vec<_>>();
                assert_eq!(chunks, ["", "|", "visible"]);
                let visible = chunks[2];
                let start = view.buffer.as_ptr() as usize;
                assert!((start..start + view.buffer.len()).contains(&(visible.as_ptr() as usize)));
                Ok(chunks.concat())
            })
            .unwrap();
        assert_eq!(selected, "|visible");
        assert!(parser.buffer.is_empty());
        let output = parser.preview_bytes(candidate, true).unwrap();
        assert_eq!(output.content, selected);
        assert_eq!(output.reasoning, "secret");
    }

    #[test]
    fn projected_content_schema_checks_use_the_first_declared_channel_boundary() {
        use crate::byte_grammar::{GrammarByteCursor, GrammarCandidate};
        use crate::json_prefix::JsonPrefix;
        let spec = ResponseTemplate::compile(&serde_json::json!({"start_anchor":"A","fields":{
            "reasoning_content":{"open":"[r]","close":"[/r]"}, "content":{}
        }}))
        .unwrap();
        let schema = serde_json::json!({"type":"object","properties":{
            "n":{"const":1}, "x":{"type":"string"}
        },"required":["n"]});
        let mut byte_grammar = GrammarByteCursor::json_schema(&schema, 4096).unwrap();
        let mut parser =
            ResponseParser::new(spec, "A[r]existing", PrefixOutput::Exclude, 4096).unwrap();
        for (candidate, expected) in [
            ("new[/r]{\"n\":1}", true),
            ("new[/r]{\"n\":2}", false),
            ("new[/r]bad[/r]{\"n\":1}", false),
            ("new[/r]{\"n\":1,\"x\":\"[r]hidden[/r]tail\"}", false),
            (r#"new[/r]{"n":1,"x":"\u005br]literal"}"#, true),
        ] {
            let accepted = parser
                .with_preview(candidate.as_bytes(), true, |view| {
                    let content = view.field_chunks("content", true).collect::<String>();
                    let canonical = JsonPrefix::parse(&content)?.canonical;
                    Ok(matches!(
                        byte_grammar.preview(canonical.as_bytes())?,
                        GrammarCandidate::Viable { complete: true }
                    ))
                })
                .unwrap();
            assert_eq!(accepted, expected, "candidate={candidate:?}");
            assert_eq!(byte_grammar.committed_bytes(), 0);
            assert_eq!(parser.buffer, "[r]existing");
            parser
                .require_open_text_channel("reasoning_content")
                .unwrap();
        }
        let error = parser
            .with_preview(b"[/r]{\"n\":1}", true, |_| {
                Err::<(), _>(
                    std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "injected schema observer error",
                    )
                    .into(),
                )
            })
            .unwrap_err();
        assert!(error.is::<std::io::Error>());
        assert_eq!(parser.buffer, "[r]existing");
        assert_eq!(
            parser.preview("[/r]{\"n\":1}", true).unwrap().content,
            "{\"n\":1}"
        );
    }

    #[test]
    fn byte_response_every_split_matches_text_projection_across_unicode_boundaries() {
        let spec = ResponseTemplate::compile(&serde_json::json!({
            "start_anchor":"A", "fields":{
                "reasoning_content":{"open":"[r]","close_pattern":"終\\s*"}, "content":{}
            }
        }))
        .unwrap();
        let suffix = "新🙂終 \n{\"n\":1}";
        let expected = collected(spec.clone(), "A[r]既存", &[suffix]).unwrap();
        assert_eq!(expected.content, "{\"n\":1}");
        assert_eq!(expected.reasoning, "新🙂");
        for split in 0..=suffix.len() {
            let mut parser =
                ResponseParser::new(spec.clone(), "A[r]既存", PrefixOutput::Exclude, 1024).unwrap();
            let mut output = parser.feed_bytes(&suffix.as_bytes()[..split]).unwrap();
            for delta in [
                parser.feed_bytes(&suffix.as_bytes()[split..]).unwrap(),
                parser.finish().unwrap(),
            ] {
                output.content.push_str(&delta.content);
                output.reasoning.push_str(&delta.reasoning);
            }
            assert_eq!(output, expected, "byte split={split}");
        }
    }

    #[test]
    fn byte_response_pending_scalars_obey_budget_and_preview_rollback() {
        let spec = ResponseTemplate::compile(
            &serde_json::json!({"start_anchor":"A","fields":{"content":{}}}),
        )
        .unwrap();
        let mut parser = ResponseParser::new(spec.clone(), "A", PrefixOutput::Include, 4).unwrap();
        assert_eq!(parser.feed_bytes(&[0xf0]).unwrap(), ChatText::default());
        assert_eq!(
            parser.preview_bytes(&[0x9f, 0x92], false).unwrap(),
            ChatText::default()
        );
        assert_eq!(parser.utf8_tail.bytes(), [0xf0]);
        let error = parser.preview_bytes(&[0x9f, 0x92], true).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("3 incomplete UTF-8 bytes at raw offset 0"),
            "{error:#}"
        );
        assert_eq!(parser.utf8_tail.bytes(), [0xf0]);
        parser.feed_bytes(&[0x9f, 0x92]).unwrap();
        assert!(
            parser
                .feed_bytes(b" ")
                .unwrap_err()
                .is::<std::str::Utf8Error>()
        );
        assert_eq!(parser.utf8_tail.bytes(), [0xf0, 0x9f, 0x92]);
        assert!(
            parser
                .preview_bytes(&[0xa9, b'x'], false)
                .unwrap_err()
                .to_string()
                .contains("buffer")
        );
        assert_eq!(parser.feed_bytes(&[0xa9]).unwrap().content, "💩");
        assert_eq!(parser.finish().unwrap(), ChatText::default());
        let mut parser = ResponseParser::new(spec, "A", PrefixOutput::Include, 4).unwrap();
        parser.feed_bytes(&[0xe7, 0xb5]).unwrap();
        assert!(
            parser
                .finish()
                .unwrap_err()
                .to_string()
                .contains("2 incomplete UTF-8 bytes")
        );
    }

    #[test]
    fn byte_response_invalid_sequences_preserve_state_and_utf8_error_causes() {
        let spec = ResponseTemplate::compile(
            &serde_json::json!({"start_anchor":"A","fields":{"content":{}}}),
        )
        .unwrap();
        for invalid in [
            &[0xc0, 0xaf][..],
            &[0xe0, 0x80, 0x80],
            &[0xed, 0xa0, 0x80],
            &[0xf4, 0x90, 0x80, 0x80],
            &[0x80],
            &[0xff],
        ] {
            let mut parser =
                ResponseParser::new(spec.clone(), "A", PrefixOutput::Include, 1024).unwrap();
            parser.feed("先").unwrap();
            let bytes = [b"ok".as_slice(), invalid].concat();
            let error = parser.feed_bytes(&bytes).unwrap_err();
            assert!(error.is::<std::str::Utf8Error>(), "{error:#}");
            assert!(error.to_string().contains("raw offset 5"), "{error:#}");
            assert_eq!(parser.buffer, "先");
            assert!(parser.utf8_tail.bytes().is_empty());
            assert_eq!(parser.feed("後").unwrap().content, "後");
        }
    }

    #[test]
    fn byte_response_consumes_real_byte_fallback_pieces_without_replacement_characters() {
        use serde_json::{Value, json};
        let mut value: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/tiny-llm/tokenizer.json"
        )))
        .unwrap();
        let mut vocab = value["model"]["vocab"].as_object().unwrap().clone();
        for (old, replacement, id) in [
            ("hello", "<0xE7>", 4),
            ("world", "<0xB5>", 5),
            ("tok16", "<0x82>", 16),
            ("tok17", "{\"n\":1}", 17),
        ] {
            vocab.remove(old);
            vocab.insert(replacement.into(), json!(id));
        }
        value["model"] = json!({"type":"BPE","unk_token":"<unk>","byte_fallback":true,"vocab":vocab,"merges":[]});
        value["decoder"] = json!({"type":"ByteFallback"});
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("tokenizer.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let tokenizer = onnx_genai_ort::Tokenizer::from_file(path).unwrap();
        let pieces = crate::tokenization::token_pieces(&tokenizer, &[4, 5, 16, 17]).unwrap();
        assert_eq!(pieces.concat(), "終{\"n\":1}".as_bytes());
        let spec = ResponseTemplate::compile(&serde_json::json!({"start_anchor":"A","fields":{
            "reasoning_content":{"open":"[r]","close":"終"}, "content":{}
        }}))
        .unwrap();
        let mut parser =
            ResponseParser::new(spec, "A[r]既存", PrefixOutput::Exclude, 1024).unwrap();
        assert_eq!(parser.feed("新").unwrap().reasoning, "新");
        let mut output = ChatText::default();
        for piece in pieces {
            let delta = parser.feed_bytes(&piece).unwrap();
            output.content.push_str(&delta.content);
            output.reasoning.push_str(&delta.reasoning);
        }
        let tail = parser.finish().unwrap();
        output.content.push_str(&tail.content);
        output.reasoning.push_str(&tail.reasoning);
        assert_eq!(output.content, "{\"n\":1}");
        assert!(output.reasoning.is_empty());
    }

    #[test]
    fn preview_restores_replaced_repeated_regions_and_the_published_baseline() {
        for repeats in [false, true] {
            let spec = template(repeats);
            let mut parser =
                ResponseParser::new(spec.clone(), "[assistant]", PrefixOutput::Exclude, 16384)
                    .unwrap();
            let mut fresh =
                ResponseParser::new(spec, "[assistant]", PrefixOutput::Exclude, 16384).unwrap();
            assert_eq!(
                parser.feed("<r>one</r>x").unwrap(),
                fresh.feed("<r>one</r>x").unwrap()
            );
            let before = parser.closed.clone();
            for _ in 0..2 {
                let trial = parser.preview("<r>two</r>answer[end]", true).unwrap();
                assert_eq!(trial.content, "answer");
                assert_eq!(trial.reasoning, if repeats { "onetwo" } else { "two" });
                assert!(
                    before
                        .iter()
                        .zip(&parser.closed)
                        .all(|(a, b)| std::sync::Arc::ptr_eq(a, b))
                );
                assert_eq!(parser.buffer, "<r>one</r>x");
            }
            assert_eq!(
                parser.feed("tail[end]").unwrap(),
                fresh.feed("tail[end]").unwrap()
            );
            assert_eq!(parser.finish().unwrap(), fresh.finish().unwrap());
        }
    }

    #[test]
    fn preview_failures_and_unwinding_leave_required_fields_and_defaults_unchanged() {
        let spec = ResponseTemplate::compile(&serde_json::json!({
            "start_anchor":"[a]", "fields":{"content":{"open":"<c>","close":"</c>","optional":false}},
            "defaults":{"reasoning_content":"default"}
        })).unwrap();
        let mut parser = ResponseParser::new(spec, "[a]", PrefixOutput::Include, 32).unwrap();
        assert!(
            parser
                .preview("discard", true)
                .unwrap_err()
                .to_string()
                .contains("required field")
        );
        assert!(
            parser
                .preview(&"x".repeat(33), false)
                .unwrap_err()
                .to_string()
                .contains("buffer")
        );
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let trial = ResponseTrial::new(&mut parser);
            trial.parser.append_bytes(b"<c>discard</c>").unwrap();
            trial.parser.process(true).unwrap();
            panic!("inject speculative parser unwind");
        }));
        assert!(panic.is_err());
        assert!(parser.buffer.is_empty());
        assert!(parser.seen.iter().all(|seen| !seen));
        assert!(
            parser
                .preview("", true)
                .unwrap_err()
                .to_string()
                .contains("required field")
        );
        let value = parser.preview("<c>保留</c>", true).unwrap();
        assert_eq!(value.content, "保留");
        assert_eq!(value.reasoning, "default");
        assert!(parser.feed("<c>actual</c>").unwrap().content == "actual");
        assert_eq!(parser.finish().unwrap().reasoning, "default");
    }

    #[test]
    fn preview_final_projection_can_replace_text_that_streaming_already_published() {
        let spec = ResponseTemplate::compile(&serde_json::json!({
            "start_anchor":"[a]", "fields":{"content":{"open_pattern":"<c>\\s*","close":"</c>"}}
        }))
        .unwrap();
        let mut parser =
            ResponseParser::new(spec, "[a]<c>old", PrefixOutput::Include, 1024).unwrap();
        assert_eq!(parser.preview("</c><c>", false).unwrap().content, "old");
        assert_eq!(parser.preview("</c><c>", true).unwrap().content, "");
        assert_eq!(parser.feed("</c><c>").unwrap().content, "old");
        assert!(
            parser
                .finish()
                .unwrap_err()
                .to_string()
                .contains("buffered output")
        );
    }

    #[test]
    fn ordinary_prefill_includes_closed_and_open_regions_even_without_generated_text() {
        let spec = template(true);
        for (prefix, suffix, content, reasoning) in [
            (
                "history[assistant]old[user]again[assistant]<r> first </r><r> second ",
                "尾巴</r>答案",
                "答案",
                "firstsecond 尾巴",
            ),
            ("[assistant] answer ", "", "answer", ""),
            (
                "[assistant]<r> thought </r> answer ",
                "",
                "answer",
                "thought",
            ),
        ] {
            for split in suffix
                .char_indices()
                .map(|(at, _)| at)
                .chain(std::iter::once(suffix.len()))
            {
                let mut parser =
                    ResponseParser::new(spec.clone(), prefix, PrefixOutput::Include, 16384)
                        .unwrap();
                let mut out = parser.feed(&suffix[..split]).unwrap();
                for delta in [
                    parser.feed(&suffix[split..]).unwrap(),
                    parser.finish().unwrap(),
                ] {
                    out.content.push_str(&delta.content);
                    out.reasoning.push_str(&delta.reasoning);
                }
                assert_eq!(out.content, content);
                assert_eq!(out.reasoning, reasoning);
            }
        }
    }

    #[test]
    fn response_template_regions_prime_last_turn_and_exclude_raw_prefix_whitespace() {
        let spec = template(true);
        let prefix = "[assistant]<r>old</r>old answer[user]again[assistant]<r> prefill \n";
        let generated = "new思考</r> \n<tool>{\"secret\":1}</tool>answer[end]";
        let expected = ChatText {
            content: "answer".into(),
            reasoning: "new思考".into(),
        };
        assert_eq!(
            collected(spec.clone(), prefix, &[generated]).unwrap(),
            expected
        );
        for split in generated
            .char_indices()
            .map(|(at, _)| at)
            .chain(std::iter::once(generated.len()))
        {
            assert_eq!(
                collected(
                    spec.clone(),
                    prefix,
                    &[&generated[..split], &generated[split..]]
                )
                .unwrap(),
                expected,
                "split {split}"
            );
        }
        let chunks = generated
            .char_indices()
            .map(|(at, c)| &generated[at..at + c.len_utf8()])
            .collect::<Vec<_>>();
        assert_eq!(collected(spec, prefix, &chunks).unwrap(), expected);
    }
    #[test]
    fn response_template_repeated_channels_join_only_generated_regions() {
        let spec=ResponseTemplate::compile(&serde_json::json!({"start_anchor":"[assistant]","fields":{
            "reasoning_content":{"open":"<r>","close":"</r>","repeats":true,"join":"|"},"content":{}
        }})).unwrap();
        assert_eq!(
            collected(
                spec,
                "[assistant]<r>old</r><r> prefill ",
                &["new</r><r>tail</r>answer"]
            )
            .unwrap(),
            ChatText {
                content: "answer".into(),
                reasoning: "new|tail".into()
            }
        );
    }
    #[test]
    fn response_template_reopened_field_can_rebuild_a_published_prefix_without_echo() {
        let spec = template(false);
        let mut parser =
            ResponseParser::new(spec.clone(), "[assistant]", PrefixOutput::Exclude, 1024).unwrap();
        assert_eq!(parser.feed("<r>one").unwrap().reasoning, "one");
        for chunk in ["</r><r>", "o", "n", "e"] {
            assert!(parser.feed(chunk).unwrap().reasoning.is_empty());
        }
        assert_eq!(parser.feed(" more</r>").unwrap().reasoning, " more");
        assert!(parser.finish().unwrap().reasoning.is_empty());
        let mut parser =
            ResponseParser::new(spec.clone(), "[assistant]", PrefixOutput::Exclude, 1024).unwrap();
        parser.feed("<r>one</r><r>on").unwrap();
        // The whole first feed projects only the most recent value.
        assert_eq!(parser.finish().unwrap().reasoning, "");
        let mut parser =
            ResponseParser::new(spec, "[assistant]", PrefixOutput::Exclude, 1024).unwrap();
        parser.feed("<r>one").unwrap();
        assert!(parser.feed("</r><r>on").unwrap().reasoning.is_empty());
        assert!(
            parser
                .finish()
                .unwrap_err()
                .to_string()
                .contains("buffered output")
        );
    }

    #[test]
    fn response_template_nonmonotonic_fields_fail_streaming_but_buffered_keeps_final_value() {
        let spec = template(false);
        assert_eq!(
            collected(spec.clone(), "[assistant]", &["<r>one</r><r>two</r>answer"])
                .unwrap()
                .reasoning,
            "two"
        );
        let mut parser =
            ResponseParser::new(spec, "[assistant]", PrefixOutput::Exclude, 1024).unwrap();
        assert_eq!(parser.feed("<r>one").unwrap().reasoning, "one");
        let error = parser.feed("</r><r>two").unwrap_err().to_string();
        assert!(error.contains("buffered output"), "{error}");
    }
    #[test]
    fn response_template_validates_configuration_required_regions_and_buffer_budget() {
        for declaration in [
            serde_json::json!({"fields":{"content":{}}}),
            serde_json::json!({"version":2,"start_anchor":"a","fields":{"content":{}}}),
            serde_json::json!({"start_anchor":"a","fields":{"content":{},"reasoning_content":{}}}),
            serde_json::json!({"start_anchor":"a","fields":{"content":{"open_pattern":"^"}}}),
            serde_json::json!({"start_anchor":"a","fields":{"content":{"join":"|"}}}),
            serde_json::json!({"start_anchor":"a","fields":{"content":{"content":"json"}}}),
        ] {
            assert!(
                ResponseTemplate::compile(&declaration).is_err(),
                "{declaration}"
            );
        }
        let spec=ResponseTemplate::compile(&serde_json::json!({"start_anchor":"a","fields":{"content":{"open":"<c>","optional":false}}})).unwrap();
        let mut parser = ResponseParser::new(spec.clone(), "a", PrefixOutput::Exclude, 8).unwrap();
        assert!(
            parser
                .feed("123456789")
                .unwrap_err()
                .to_string()
                .contains("reduce max_tokens")
        );
        assert!(
            parser
                .finish()
                .unwrap_err()
                .to_string()
                .contains("required field")
        );
        assert_eq!(
            collected(spec, "a<c>prefix", &["suffix"]).unwrap().content,
            "suffix"
        );
    }
    #[test]
    fn response_template_declared_text_defaults_apply_only_when_no_region_matched() {
        let spec=ResponseTemplate::compile(&serde_json::json!({"start_anchor":"a","defaults":{"content":"fallback"},"fields":{"reasoning_content":{"open":"<r>","close":"</r>"}}})).unwrap();
        assert_eq!(collected(spec, "a", &[]).unwrap().content, "fallback");
    }

    #[test]
    fn response_template_every_utf8_split_preserves_the_final_boundary() {
        let cases = [
            (
                Boundary::literal(vec!["a".into(), "abcd".into()], "close").unwrap(),
                "bodyabcdtail",
            ),
            (
                Boundary::literal(vec!["</思考>".into()], "close").unwrap(),
                "α body</思考>tail",
            ),
            (
                Boundary::pattern(r"</think>\s*", "close_pattern").unwrap(),
                "α body</think> \n tail",
            ),
            (
                Boundary::pattern(r"<<.+>>", "close_pattern").unwrap(),
                "body<<a>>tail<<b>>final",
            ),
        ];
        for (boundary, input) in cases {
            let expected = boundary.scan_text(input, 0, true).unwrap().matched.unwrap();
            for split in input
                .char_indices()
                .map(|(at, _)| at)
                .chain(std::iter::once(input.len()))
            {
                let scan = boundary.scan_text(&input[..split], 0, false).unwrap();
                match scan.matched {
                    Some(found) => assert_eq!(
                        found, expected,
                        "boundary committed before it was stable at split {split}: {input:?}"
                    ),
                    None => assert!(
                        scan.hold_from <= expected.start,
                        "delimiter prefix leaked at split {split}: {scan:?} {input:?}"
                    ),
                }
            }
        }
    }
    #[test]
    fn response_template_literal_boundaries_hold_partial_unicode_and_prefix_alternatives() {
        let boundary =
            Boundary::literal(vec!["</思考>".into()], "reasoning_content.close").unwrap();
        for suffix in ["<", "</", "</思", "</思考"] {
            let text = format!("thought{suffix}");
            assert_eq!(
                boundary.scan_text(&text, 0, false).unwrap(),
                Scan {
                    matched: None,
                    hold_from: 7
                }
            );
        }
        assert_eq!(
            boundary
                .scan_text("thought</思考>", 0, false)
                .unwrap()
                .matched,
            Some(7..16)
        );
        let overlap = Boundary::literal(vec!["a".into(), "abcd".into()], "close").unwrap();
        assert_eq!(overlap.scan_text("a", 0, false).unwrap().hold_from, 0);
        assert_eq!(overlap.scan_text("ab", 0, false).unwrap().matched, None);
        assert_eq!(
            overlap.scan_text("abcd", 0, false).unwrap().matched,
            Some(0..4)
        );
        assert_eq!(
            overlap.scan_text("ax", 0, false).unwrap().matched,
            Some(0..1)
        );
        assert!(matches!(
            boundary.partial,
            PartialBoundary::Literal(_) | PartialBoundary::Exact(_)
        ));
        assert!(!boundary.can_match_empty());
    }
    #[test]
    fn response_template_identity_matches_canonical_boundary_tie_order() {
        let declaration = serde_json::json!({"start_anchor":"A","fields":{
            "content":{"open":"X","close":"END"},
            "reasoning_content":{"open":"X","close":"END"}
        }});
        let first = ResponseTemplate::compile(&declaration).unwrap();
        let second = ResponseTemplate::compile(&serde_json::json!({"start_anchor":"A","fields":{
            "reasoning_content":{"open":"X","close":"END"},
            "content":{"open":"X","close":"END"}
        }}))
        .unwrap();
        let project = |spec| {
            ResponseParser::new(spec, "A", PrefixOutput::Include, 1024)
                .unwrap()
                .preview("XhelloEND", true)
                .unwrap()
        };
        assert_eq!(project(first.clone()).content, "hello");
        assert_eq!(project(second.clone()).content, "hello");
        assert_eq!(project(first.clone()), project(second.clone()));
        assert_eq!(
            first, second,
            "field names determine canonical tie priority independently of JSON object order"
        );
        assert_eq!(first, ResponseTemplate::compile(&declaration).unwrap());
    }

    #[test]
    fn byte_regex_boundaries_distinguish_delayed_matches_and_unicode_continuations() {
        let boundary = Boundary::pattern("X終|X", "content.close_pattern").unwrap();
        assert_eq!(
            boundary
                .scan_pending("{}X", &[0xe7], 0, false)
                .unwrap()
                .matched,
            None
        );
        assert_eq!(
            boundary
                .scan_pending("{}X", &[0xc3], 0, false)
                .unwrap()
                .matched,
            Some(2..3)
        );
        assert_eq!(
            boundary.scan_text("{}Xé", 0, false).unwrap().matched,
            Some(2..3)
        );
        for pattern in ["X|X終", "X.*?"] {
            let first = Boundary::pattern(pattern, "content.close_pattern").unwrap();
            assert_eq!(
                first
                    .scan_pending("{}X", &[0xe7], 0, false)
                    .unwrap()
                    .matched,
                Some(2..3)
            );
        }
        let greedy_unicode = Boundary::pattern("X.*", "content.close_pattern").unwrap();
        assert_eq!(
            greedy_unicode
                .scan_pending("{}X", &[0xc3], 0, false)
                .unwrap()
                .matched,
            None
        );
        let greedy = Boundary::pattern(r"</think>\s*", "reasoning_content.close_pattern").unwrap();
        assert_eq!(
            greedy
                .scan_pending("x</think> ", &[0xc2], 0, false)
                .unwrap()
                .matched,
            None
        );
        assert_eq!(
            greedy
                .scan_pending("x</think> ", &[0xc3], 0, false)
                .unwrap()
                .matched,
            Some(1..10)
        );
    }

    #[test]
    fn regex_json_capability_refusal_preserves_the_backend_cause() {
        let spec = ResponseTemplate::compile(&serde_json::json!({"start_anchor":"A","fields":{
            "content":{"close_pattern":r"\bEND\b"}
        }}))
        .unwrap();
        let error = spec.require_constraint_boundaries().unwrap_err();
        let refusal = error
            .downcast_ref::<ConstraintBoundaryRefusal>()
            .expect("typed capability refusal");
        assert_eq!(refusal.field, "content");
        assert!(std::error::Error::source(refusal).is_some());
        let mut parser = ResponseParser::new(spec, "A", PrefixOutput::Include, 1024).unwrap();
        assert!(parser.feed("éEND").unwrap().content.is_empty());
        assert_eq!(parser.finish().unwrap().content, "éEND");
    }

    #[test]
    fn byte_regex_boundaries_observe_pending_bytes_for_eof_and_look_behind() {
        let boundary = Boundary::pattern(r"(?-u)\bX\z", "content.close_pattern").unwrap();
        assert_eq!(boundary.scan_text(" X", 1, false).unwrap().matched, None);
        assert_eq!(
            boundary.scan_text(" X", 1, true).unwrap().matched,
            Some(1..2)
        );
        assert_eq!(
            boundary
                .scan_pending(" X", &[0xc3], 1, false)
                .unwrap()
                .matched,
            None
        );
        assert_eq!(boundary.scan_text("AX", 1, true).unwrap().matched, None);
        assert_eq!(
            boundary
                .scan_pending(" X", &[0xc3], 1, false)
                .unwrap()
                .hold_from,
            2
        );
    }

    #[test]
    fn response_template_regex_holds_split_delimiters_and_greedy_trailing_whitespace() {
        let boundary =
            Boundary::pattern(r"</think>\s*", "reasoning_content.close_pattern").unwrap();
        assert_eq!(
            boundary
                .scan_text("thought</thi", 0, false)
                .unwrap()
                .hold_from,
            7
        );
        assert_eq!(
            boundary
                .scan_text("thought</think> \n", 0, false)
                .unwrap()
                .hold_from,
            7
        );
        let complete = "thought</think> \nanswer";
        assert_eq!(
            boundary.scan_text(complete, 0, false).unwrap().matched,
            Some(7..17)
        );
        assert_eq!(
            boundary
                .scan_text("thought</think>", 0, true)
                .unwrap()
                .matched,
            Some(7..15)
        );
        assert!(matches!(
            boundary.partial,
            PartialBoundary::Literal(_) | PartialBoundary::Exact(_)
        ));
    }
    #[test]
    fn response_template_empty_literals_are_invalid_and_zero_width_close_waits_for_eos() {
        assert!(Boundary::literal(vec![String::new()], "close").is_err());
        let boundary = Boundary::pattern("$", "content.close_pattern").unwrap();
        assert!(boundary.can_match_empty());
        assert_eq!(boundary.scan_text("", 0, false).unwrap().matched, None);
        assert_eq!(boundary.scan_text("", 0, true).unwrap().matched, Some(0..0));
        assert_eq!(boundary.scan_text("x", 0, false).unwrap().matched, None);
        assert_eq!(
            boundary.scan_text("x", 0, true).unwrap().matched,
            Some(1..1)
        );
    }

    #[test]
    fn response_template_regex_uses_global_anchors_unicode_and_explicit_finalization() {
        let anchored = Boundary::pattern(r"^α.+Ω$", "content.close_pattern").unwrap();
        assert!(
            anchored
                .scan_text("α\nΩ", 0, false)
                .unwrap()
                .matched
                .is_none()
        );
        assert_eq!(
            anchored.scan_text("α\nΩ", 0, true).unwrap().matched,
            Some(0..5)
        );
        assert!(
            anchored
                .scan_text("xα\nΩ", 1, true)
                .unwrap()
                .matched
                .is_none()
        );
        assert!(
            Boundary::pattern(r"$", "close_pattern")
                .unwrap()
                .can_match_empty()
        );
        assert!(anchored.scan_text("α\nΩ", 1, false).is_err());
    }
    #[test]
    fn reasoning_budget_boundaries_preserve_authored_closer_order_and_final_prefill() {
        let template = ResponseTemplate::compile(&serde_json::json!({"start_anchor":"A","fields":{
            "reasoning_content":{"open":["[r]","[reasoning]"],"close":["X","Xlong"]},"content":{}
        }}))
        .unwrap();
        let boundaries = template.reasoning_boundaries().unwrap();
        assert_eq!(
            boundaries.start.unwrap().literal_alternatives().unwrap(),
            ["[r]", "[reasoning]"]
        );
        assert_eq!(
            boundaries.end.literal_alternatives().unwrap(),
            ["X", "Xlong"]
        );
        boundaries.end.validate_literal("Xlong").unwrap();
        assert!(boundaries.end.validate_literal("wrong").is_err());
        assert_eq!(
            template
                .generation_prefix("A[r]oldXlong user A[r]partial")
                .unwrap(),
            "[r]partial"
        );
        assert!(
            template
                .generation_prefix("missing")
                .unwrap_err()
                .to_string()
                .contains("start_anchor")
        );
        let mut parser =
            ResponseParser::new(template, "A[r]", PrefixOutput::Exclude, 4096).unwrap();
        let mut output = parser.feed("valueX").unwrap();
        output
            .content
            .push_str(&parser.feed("next").unwrap().content);
        assert_eq!(output.content, "next");
    }

    #[test]
    fn reasoning_budget_boundaries_support_implicit_reasoning_and_content_transition() {
        let template = ResponseTemplate::compile(&serde_json::json!({"start_anchor":"A","fields":{
            "reasoning_content":{},"content":{"open":"<answer>"}
        }}))
        .unwrap();
        let boundaries = template.reasoning_boundaries().unwrap();
        assert!(boundaries.start.is_none());
        assert_eq!(boundaries.end.literal_alternatives().unwrap(), ["<answer>"]);
        let absent = ResponseTemplate::compile(
            &serde_json::json!({"start_anchor":"A","fields":{"content":{}}}),
        )
        .unwrap();
        assert!(absent.reasoning_boundaries().is_err());
    }
}
