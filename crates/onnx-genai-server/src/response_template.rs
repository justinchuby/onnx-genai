//! Declared response boundaries. No model identity determines a delimiter.
use anyhow::{Context, ensure};
use regex::{Regex, RegexBuilder};
use regex_automata::{
    Anchored, Input,
    dfa::{Automaton, StartKind, dense},
};
use std::ops::Range;

const REGEX_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug)]
pub(crate) struct Boundary {
    regex: Regex,
    literals: Option<Vec<String>>,
    partial: Option<dense::DFA<Vec<u32>>>,
    minimum_bytes: Option<usize>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Scan {
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
        literals.sort_by_key(|literal| std::cmp::Reverse(literal.chars().count()));
        let pattern = literals
            .iter()
            .map(|literal| regex::escape(literal))
            .collect::<Vec<_>>()
            .join("|");
        let mut boundary = Self::pattern(&pattern, field)?;
        boundary.literals = Some(literals);
        Ok(boundary)
    }

    pub fn pattern(pattern: &str, field: &str) -> anyhow::Result<Self> {
        let regex=RegexBuilder::new(pattern).dot_matches_new_line(true).multi_line(false)
            .size_limit(REGEX_BYTES).dfa_size_limit(REGEX_BYTES).build()
            .with_context(||format!("{field} regex is unsupported; repair the pattern using Unicode-aware regular expressions without lookaround or backreferences"))?;
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
            .ok();
        Ok(Self {
            regex,
            literals: None,
            partial,
            minimum_bytes: hir.properties().minimum_len(),
        })
    }

    pub fn can_match_empty(&self) -> bool {
        self.minimum_bytes == Some(0)
    }

    pub fn scan(&self, input: &str, from: usize, final_chunk: bool) -> anyhow::Result<Scan> {
        ensure!(
            input.is_char_boundary(from),
            "response parser cursor {from} is outside a UTF-8 boundary; restart this response parser"
        );
        let found = self.regex.find_at(input, from).map(|m| m.range());
        if final_chunk {
            return Ok(Scan {
                matched: found,
                hold_from: input.len(),
            });
        }
        if let Some(literals) = &self.literals {
            let mut hold = input.len();
            for (offset, _) in input[from..].char_indices() {
                let start = from + offset;
                if literals.iter().any(|literal| {
                    input[start..].len() < literal.len() && literal.starts_with(&input[start..])
                }) {
                    hold = start;
                    break;
                }
            }
            if let Some(found) = found {
                // A longer literal at the same start can still win after the
                // next chunk. Never commit a shorter complete alternative yet.
                if hold <= found.start {
                    return Ok(Scan {
                        matched: None,
                        hold_from: hold.min(found.start),
                    });
                }
                return Ok(Scan {
                    matched: Some(found),
                    hold_from: input.len(),
                });
            }
            return Ok(Scan {
                matched: None,
                hold_from: hold,
            });
        }
        let Some(dfa) = &self.partial else {
            return Ok(Scan {
                matched: None,
                hold_from: from,
            });
        };
        for start in input[from..]
            .char_indices()
            .map(|(offset, _)| from + offset)
            .chain(std::iter::once(input.len()))
        {
            let query = Input::new(input)
                .span(start..input.len())
                .anchored(Anchored::Yes);
            let Ok(mut state) = dfa.start_state_forward(&query) else {
                return Ok(Scan {
                    matched: None,
                    hold_from: start,
                });
            };
            let mut unknown = false;
            for &byte in &input.as_bytes()[start..] {
                state = dfa.next_state(state, byte);
                if dfa.is_quit_state(state) {
                    unknown = true;
                    break;
                }
                if dfa.is_dead_state(state) {
                    break;
                }
            }
            if unknown || !dfa.is_dead_state(state) {
                return Ok(Scan {
                    matched: None,
                    hold_from: start,
                });
            }
            if found.as_ref().is_some_and(|found| found.start == start) {
                return Ok(Scan {
                    matched: found,
                    hold_from: input.len(),
                });
            }
        }
        Ok(Scan {
            matched: None,
            hold_from: input.len(),
        })
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

#[derive(Debug)]
pub(crate) struct ResponseTemplate {
    anchor: Boundary,
    fields: Vec<Field>,
    implicit: Option<usize>,
    defaults: serde_json::Map<String, serde_json::Value>,
}

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
    pub fn compile(value: &serde_json::Value) -> anyhow::Result<std::sync::Arc<Self>> {
        let root = object(value, "response_template")?;
        ensure!(
            root.keys().all(|key| matches!(
                key.as_str(),
                "version" | "fields" | "defaults" | "start_anchor" | "start_anchor_pattern"
            )),
            "response_template contains unsupported keys; use the version-1 response template fields"
        );
        ensure!(
            root.get("version")
                .is_none_or(|value| value.as_u64() == Some(1)),
            "response_template.version must be 1; re-emit a supported response template"
        );
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
            anchor,
            fields,
            implicit,
            defaults,
        }))
    }
}

#[derive(Debug, Default)]
struct Region {
    text: String,
    omitted_prefix_bytes: usize,
    visible: bool,
}
impl Region {
    fn output<'a>(&'a self, field: &Field) -> &'a str {
        let text = if field.strip {
            self.text
                .trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
        } else {
            &self.text
        };
        let start = text.as_ptr() as usize - self.text.as_ptr() as usize;
        let end = start + text.len();
        let from = start.max(self.omitted_prefix_bytes);
        if from >= end {
            ""
        } else {
            &self.text[from..end]
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct ChatText {
    pub content: String,
    pub reasoning: String,
}

/// Whether assistant text already rendered by the template belongs to this output.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PrefixOutput {
    Include,
    Exclude,
}

pub(crate) struct ResponseParser {
    spec: std::sync::Arc<ResponseTemplate>,
    buffer: String,
    omitted_prefix_end: usize,
    pos: usize,
    current: Option<usize>,
    region: Option<Region>,
    closed: Vec<Vec<Region>>,
    seen: Vec<bool>,
    sent: ChatText,
    max_bytes: usize,
}

impl ResponseParser {
    pub fn new(
        spec: std::sync::Arc<ResponseTemplate>,
        prefix: &str,
        prefix_output: PrefixOutput,
        max_bytes: usize,
    ) -> anyhow::Result<Self> {
        let at = spec
            .anchor
            .regex
            .find_iter(prefix)
            .last()
            .map_or(0, |matched| matched.end());
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
            closed: (0..spec.fields.len()).map(|_| Vec::new()).collect(),
            seen: vec![false; spec.fields.len()],
            current: spec.implicit,
            spec,
            buffer: prefix.into(),
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

    pub(crate) fn require_open_text_channel(&self, name: &str) -> anyhow::Result<()> {
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
            let region = self.region.get_or_insert_with(Region::default);
            region.text.push_str(&self.buffer[range.clone()]);
            region.omitted_prefix_bytes += range
                .end
                .min(self.omitted_prefix_end)
                .saturating_sub(range.start);
            region.visible |= range.end > self.omitted_prefix_end;
            self.seen[field] = true;
        }
    }
    fn close(&mut self) {
        if let (Some(field), Some(region)) = (self.current, self.region.take()) {
            if !self.spec.fields[field].repeats {
                self.closed[field].clear();
            }
            self.closed[field].push(region);
        }
        self.current = self.spec.implicit;
    }
    fn process(&mut self, final_chunk: bool) -> anyhow::Result<()> {
        loop {
            let spec = self.spec.clone();
            let mut watch = Vec::new();
            if let Some(field) = self.current.filter(|field| Some(*field) != spec.implicit) {
                if let Some(close) = &spec.fields[field].close {
                    watch.push((field, false, close));
                }
            } else {
                for (field, definition) in spec.fields.iter().enumerate() {
                    if let Some(open) = &definition.open {
                        watch.push((field, true, open));
                    }
                }
                if let Some(field) = spec.implicit
                    && let Some(close) = &spec.fields[field].close
                {
                    watch.push((field, false, close));
                }
            }
            let mut hold = self.buffer.len();
            let mut best: Option<(usize, bool, Range<usize>)> = None;
            for (field, opening, boundary) in watch {
                let scan = boundary.scan(&self.buffer, self.pos, final_chunk)?;
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
                        visible: range.end > self.omitted_prefix_end,
                        ..Default::default()
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
    fn snapshot(&self, final_chunk: bool) -> ChatText {
        let mut output = if final_chunk {
            ChatText {
                content: self
                    .spec
                    .defaults
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                reasoning: self
                    .spec
                    .defaults
                    .get("reasoning_content")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            }
        } else {
            ChatText::default()
        };
        for (field, definition) in self.spec.fields.iter().enumerate() {
            let target = match definition.name.as_str() {
                "content" => &mut output.content,
                "reasoning_content" => &mut output.reasoning,
                _ => continue,
            };
            if self.seen[field] {
                target.clear();
            }
            let active = if self.current == Some(field) {
                self.region.as_ref()
            } else {
                None
            };
            let regions: Vec<&Region> = if !definition.repeats {
                active
                    .or_else(|| self.closed[field].last())
                    .into_iter()
                    .collect()
            } else {
                self.closed[field].iter().chain(active).collect()
            };
            let mut first = true;
            for region in regions.into_iter().filter(|region| region.visible) {
                if !first {
                    target.push_str(&definition.join);
                }
                first = false;
                target.push_str(region.output(definition));
            }
        }
        output
    }
    fn delta(&mut self, final_chunk: bool) -> anyhow::Result<ChatText> {
        let now = self.snapshot(final_chunk);
        let extension = |current: &str,
                         sent: &str,
                         field: &str|
         -> anyhow::Result<Option<String>> {
            if let Some(tail) = current.strip_prefix(sent) {
                return Ok(Some(tail.to_owned()));
            }
            // A reopened region can rebuild the already-published prefix. Hold
            // that incomplete reconstruction; never echo it or accept a shorter
            // final value as a successful append-only response.
            if !final_chunk && sent.starts_with(current) {
                return Ok(None);
            }
            anyhow::bail!(
                "response_template {field} replaced already streamed text; use buffered output or declare repeats with join for append-only fields"
            )
        };
        let content = extension(&now.content, &self.sent.content, "content")?;
        let reasoning = extension(&now.reasoning, &self.sent.reasoning, "reasoning_content")?;
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
    pub fn feed(&mut self, text: &str) -> anyhow::Result<ChatText> {
        ensure!(
            self.buffer
                .len()
                .checked_add(text.len())
                .is_some_and(|size| size <= self.max_bytes),
            "response parser buffer would exceed {} bytes; shorten the assistant prefix or reduce max_tokens",
            self.max_bytes
        );
        self.buffer.push_str(text);
        self.process(false)?;
        self.delta(false)
    }
    pub fn finish(mut self) -> anyhow::Result<ChatText> {
        self.process(true)?;
        for (field, definition) in self.spec.fields.iter().enumerate() {
            ensure!(
                definition.optional
                    || self.seen[field]
                    || self.spec.defaults.contains_key(&definition.name),
                "response_template required field {:?} was not generated; verify the prompt prefix and declared boundaries",
                definition.name
            );
        }
        self.delta(true)
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
            let expected = boundary.scan(input, 0, true).unwrap().matched.unwrap();
            for split in input
                .char_indices()
                .map(|(at, _)| at)
                .chain(std::iter::once(input.len()))
            {
                let scan = boundary.scan(&input[..split], 0, false).unwrap();
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
                boundary.scan(&text, 0, false).unwrap(),
                Scan {
                    matched: None,
                    hold_from: 7
                }
            );
        }
        assert_eq!(
            boundary.scan("thought</思考>", 0, false).unwrap().matched,
            Some(7..16)
        );
        let overlap = Boundary::literal(vec!["a".into(), "abcd".into()], "close").unwrap();
        assert_eq!(overlap.scan("a", 0, false).unwrap().hold_from, 0);
        assert_eq!(overlap.scan("ab", 0, false).unwrap().matched, None);
        assert_eq!(overlap.scan("abcd", 0, false).unwrap().matched, Some(0..4));
        assert_eq!(overlap.scan("ax", 0, false).unwrap().matched, Some(0..1));
        assert!(boundary.literals.is_some() || boundary.partial.is_some());
        assert!(!boundary.can_match_empty());
    }
    #[test]
    fn response_template_regex_holds_split_delimiters_and_greedy_trailing_whitespace() {
        let boundary =
            Boundary::pattern(r"</think>\s*", "reasoning_content.close_pattern").unwrap();
        assert_eq!(
            boundary.scan("thought</thi", 0, false).unwrap().hold_from,
            7
        );
        assert_eq!(
            boundary
                .scan("thought</think> \n", 0, false)
                .unwrap()
                .hold_from,
            7
        );
        let complete = "thought</think> \nanswer";
        assert_eq!(
            boundary.scan(complete, 0, false).unwrap().matched,
            Some(7..17)
        );
        assert_eq!(
            boundary.scan("thought</think>", 0, true).unwrap().matched,
            Some(7..15)
        );
        assert!(boundary.literals.is_some() || boundary.partial.is_some());
    }
    #[test]
    fn response_template_empty_literals_are_invalid_and_zero_width_close_waits_for_eos() {
        assert!(Boundary::literal(vec![String::new()], "close").is_err());
        let boundary = Boundary::pattern("$", "content.close_pattern").unwrap();
        assert!(boundary.can_match_empty());
        assert_eq!(boundary.scan("", 0, false).unwrap().matched, None);
        assert_eq!(boundary.scan("", 0, true).unwrap().matched, Some(0..0));
        assert_eq!(boundary.scan("x", 0, false).unwrap().matched, None);
        assert_eq!(boundary.scan("x", 0, true).unwrap().matched, Some(1..1));
    }

    #[test]
    fn response_template_regex_uses_global_anchors_unicode_and_explicit_finalization() {
        let anchored = Boundary::pattern(r"^α.+Ω$", "content.close_pattern").unwrap();
        assert!(anchored.scan("α\nΩ", 0, false).unwrap().matched.is_none());
        assert_eq!(anchored.scan("α\nΩ", 0, true).unwrap().matched, Some(0..5));
        assert!(anchored.scan("xα\nΩ", 1, true).unwrap().matched.is_none());
        assert!(
            Boundary::pattern(r"$", "close_pattern")
                .unwrap()
                .can_match_empty()
        );
        assert!(anchored.scan("α\nΩ", 1, false).is_err());
    }
}
