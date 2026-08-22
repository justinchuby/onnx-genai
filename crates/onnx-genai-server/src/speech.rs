use std::{fs, path::Path, sync::Arc};

use anyhow::{Context, bail};
use regex::Regex;
use serde::Deserialize;

const TEXT_ASSEMBLY_ABI: &str = "onnx-genai.text-assembly";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SpeechPromptProcessor {
    pub(crate) max_input_tokens: usize,
    pub(crate) max_output_units: usize,
    segments: Vec<TextSegment>,
    #[serde(default)]
    guidance_rows: Option<GuidanceRows>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct GuidanceRows {
    unconditional_token_id: u32,
    replace_from: usize,
    preserve_trailing: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TextSegment {
    #[serde(default)]
    literal: Option<String>,
    #[serde(default)]
    field: Option<SpeechField>,
    #[serde(default)]
    transforms: Vec<TextTransform>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SpeechField {
    Input,
    Instructions,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum TextTransform {
    RewriteDelimitedTags { open: String, close: String },
    StripMarkdown,
    KeepLeadingBracketTags,
    Replace { from: String, to: String },
    LowercaseBracketTags,
    CollapseNewlines,
    Prefix { value: String },
}

impl SpeechPromptProcessor {
    pub(crate) fn assemble(&self, input: &str, instructions: &str) -> anyhow::Result<String> {
        let mut prompt = String::new();
        for segment in &self.segments {
            match (&segment.literal, segment.field) {
                (Some(literal), None) => prompt.push_str(literal),
                (None, Some(field)) => {
                    let mut value = match field {
                        SpeechField::Input => input.to_string(),
                        SpeechField::Instructions => instructions.to_string(),
                    };
                    for transform in &segment.transforms {
                        value = transform.apply(value)?;
                    }
                    prompt.push_str(&value);
                }
                _ => bail!("text-assembly segment must declare exactly one of literal or field"),
            }
        }
        Ok(prompt)
    }

    pub(crate) fn token_rows(&self, tokens: Vec<u32>) -> anyhow::Result<Vec<Vec<u32>>> {
        let Some(guidance) = &self.guidance_rows else {
            return Ok(vec![tokens]);
        };
        anyhow::ensure!(
            tokens.len() >= guidance.replace_from + guidance.preserve_trailing,
            "tokenized prompt is too short for the declared guidance-row replacement"
        );
        let mut unconditional = tokens.clone();
        let end = unconditional.len() - guidance.preserve_trailing;
        unconditional[guidance.replace_from..end].fill(guidance.unconditional_token_id);
        Ok(vec![tokens, unconditional])
    }
}

impl TextTransform {
    fn apply(&self, value: String) -> anyhow::Result<String> {
        match self {
            Self::RewriteDelimitedTags { open, close } => {
                let pattern = format!("{}(.*?){}", regex::escape(open), regex::escape(close));
                let regex = Regex::new(&pattern)?;
                Ok(regex
                    .replace_all(&value, |captures: &regex::Captures<'_>| {
                        let inner = captures[1].trim();
                        let mut parts = inner.splitn(2, char::is_whitespace);
                        let head = parts.next().unwrap_or_default();
                        parts.next().map_or_else(
                            || head.to_string(),
                            |tail| format!("{head} is {}", tail.trim()),
                        )
                    })
                    .into_owned())
            }
            Self::StripMarkdown => strip_markdown(&value),
            Self::KeepLeadingBracketTags => {
                let tags = Regex::new(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")?;
                Ok(value
                    .split('\n')
                    .map(|line| {
                        tags.captures(line).map_or_else(
                            || line.to_string(),
                            |capture| capture[1].trim().to_string(),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
            Self::Replace { from, to } => Ok(value.replace(from, to)),
            Self::LowercaseBracketTags => {
                let tags = Regex::new(r"\[([^\]]+)\]")?;
                Ok(tags
                    .replace_all(&value, |captures: &regex::Captures<'_>| {
                        format!("[{}]", captures[1].to_lowercase())
                    })
                    .into_owned())
            }
            Self::CollapseNewlines => Ok(Regex::new(r"\n{2,}")?
                .replace_all(&value, "\n")
                .into_owned()),
            Self::Prefix { value: prefix } => Ok(format!("{prefix}{value}")),
        }
    }
}

fn strip_markdown(value: &str) -> anyhow::Result<String> {
    let heading = Regex::new(r"^\s{0,3}#{1,6}\s+")?;
    let bullet = Regex::new(r"^\s*[*+-]\s+")?;
    let bold = Regex::new(r"\*\*([^*]+)\*\*")?;
    let italic = Regex::new(r"(?m)(?P<prefix>^|[^*])\*(?P<text>[^*\n]+)\*(?P<suffix>[^*]|$)")?;
    let rule = Regex::new(r"^\s*[-*_]{3,}\s*$")?;
    let lines = value
        .split('\n')
        .map(|line| {
            let line = heading.replace(line, "");
            let line = bullet.replace(&line, "");
            let mut line = line.into_owned();
            loop {
                let updated = bold.replace_all(&line, "$1").into_owned();
                if updated == line {
                    break;
                }
                line = updated;
            }
            italic
                .replace_all(&line, "${prefix}${text}${suffix}")
                .trim_end()
                .to_string()
        })
        .filter(|line| !rule.is_match(line))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(lines.replace("• ", "").replace("    ", ""))
}

pub(crate) fn load_speech_prompt_processor(
    model_dir: &Path,
) -> anyhow::Result<Option<Arc<SpeechPromptProcessor>>> {
    let metadata_path = ["inference_metadata.yaml", "inference_metadata.yml"]
        .into_iter()
        .map(|name| model_dir.join(name))
        .find(|path| path.is_file());
    let Some(metadata_path) = metadata_path else {
        return Ok(None);
    };
    let document: serde_yaml::Value = serde_yaml::from_str(
        &fs::read_to_string(&metadata_path)
            .with_context(|| format!("failed to read {}", metadata_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", metadata_path.display()))?;
    let components = document
        .get("pipeline")
        .and_then(|value| value.get("workflow"))
        .and_then(|value| value.get("components"))
        .and_then(serde_yaml::Value::as_mapping);
    let Some(components) = components else {
        return Ok(None);
    };
    for component in components.values() {
        let id = component
            .get("contract")
            .and_then(|value| value.get("id"))
            .and_then(serde_yaml::Value::as_str);
        if id != Some(TEXT_ASSEMBLY_ABI) {
            continue;
        }
        let artifact = component
            .get("implementation")
            .and_then(|value| value.get("artifact"))
            .and_then(serde_yaml::Value::as_str)
            .context("text-assembly adapter must declare an artifact")?;
        let path = model_dir.join(artifact);
        let processor = serde_json::from_str::<SpeechPromptProcessor>(
            &fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", path.display()))?;
        if processor.max_input_tokens == 0 || processor.max_output_units == 0 {
            bail!("text-assembly limits must be greater than zero");
        }
        return Ok(Some(Arc::new(processor)));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_driven_program_reproduces_caption_and_lyrics_contract() {
        let processor: SpeechPromptProcessor = serde_json::from_str(
            r#"{
              "max_input_tokens": 5000,
              "max_output_units": 9000,
              "guidance_rows": {
                "unconditional_token_id": 151654,
                "replace_from": 1,
                "preserve_trailing": 2
              },
              "segments": [
                {"literal":"<c>"},
                {"field":"instructions","transforms":[
                  {"kind":"rewrite_delimited_tags","open":"<|","close":"|>"},
                  {"kind":"strip_markdown"},
                  {"kind":"collapse_newlines"}
                ]},
                {"literal":"</c><l>"},
                {"field":"input","transforms":[
                  {"kind":"keep_leading_bracket_tags"},
                  {"kind":"replace","from":"] ","to":"]\n"},
                  {"kind":"replace","from":" [","to":"\n["},
                  {"kind":"replace","from":" ^ ","to":"\n"},
                  {"kind":"lowercase_bracket_tags"},
                  {"kind":"prefix","value":"[start]\n"}
                ]},
                {"literal":"</l>"}
              ]
            }"#,
        )
        .expect("processor");
        let prompt = processor
            .assemble(
                "[VERSE] text discarded\nHello [CHORUS]\nWorld",
                "## Genre\n<|BPM 96|>\n\n**Warm**",
            )
            .expect("assembly");
        assert_eq!(
            prompt,
            "<c>Genre\nBPM is 96\nWarm</c><l>[start]\n[verse]\nHello\n[chorus]\nWorld</l>"
        );
        let rows = processor
            .token_rows(vec![10, 11, 12, 13, 14])
            .expect("guidance rows");
        assert_eq!(
            rows,
            vec![vec![10, 11, 12, 13, 14], vec![10, 151654, 151654, 13, 14]]
        );
    }
}
