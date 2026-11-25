//! Typed, atomic realtime session/default-response configuration.
use crate::{
    tool_protocol::ToolProtocol,
    types::{ChatCompletionRequest, ToolChoiceMode},
};
use anyhow::{Context, ensure};
use onnx_genai_metadata::{RealtimeMode, realtime::RealtimeContract};
use serde_json::{Value, json};
use std::num::NonZeroU32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Modalities {
    Auto,
    Text,
    Audio,
    Both,
}
impl Modalities {
    fn parse(value: &Value) -> anyhow::Result<Self> {
        if value.is_null() {
            return Ok(Self::Auto);
        }
        let values = value.as_array().context(
            "output_modalities must be null or a non-empty array containing text and/or audio",
        )?;
        ensure!(
            !values.is_empty(),
            "output_modalities must be a non-empty array"
        );
        let (mut text, mut audio) = (false, false);
        for value in values {
            match value.as_str() {
                Some("text") => text = true,
                Some("audio") => audio = true,
                _ => anyhow::bail!(
                    "output_modalities contains unknown modality {value}; use text and/or audio"
                ),
            }
        }
        Ok(match (text, audio) {
            (true, true) => Self::Both,
            (false, true) => Self::Audio,
            _ => Self::Text,
        })
    }
    pub(super) fn json(self) -> Value {
        match self {
            Self::Auto => Value::Null,
            Self::Text => json!(["text"]),
            Self::Audio => json!(["audio"]),
            Self::Both => json!(["text", "audio"]),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OutputBudget {
    Unlimited,
    Tokens(NonZeroU32),
}
impl OutputBudget {
    fn parse(value: &Value) -> anyhow::Result<Self> {
        if value == "inf" {
            return Ok(Self::Unlimited);
        }
        let tokens = value
            .as_u64()
            .filter(|tokens| *tokens > 0)
            .context("max_output_tokens must be a positive integer or \"inf\"")?;
        // llama-server caps the wire integer at its signed 32-bit limit.
        Ok(Self::Tokens(
            NonZeroU32::new(tokens.min(i32::MAX as u64) as u32)
                .context("positive output budget became zero")?,
        ))
    }
    pub(super) fn json(self) -> Value {
        match self {
            Self::Unlimited => json!("inf"),
            Self::Tokens(tokens) => json!(tokens.get()),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ResponseOptions {
    pub(super) modalities: Modalities,
    pub(super) tool_choice: ToolChoiceMode,
    pub(super) budget: OutputBudget,
}
impl ResponseOptions {
    pub(super) fn updated(&self, response: &Value, audio: bool) -> anyhow::Result<Self> {
        let object = response.as_object().context("response must be an object")?;
        let mut next = self.clone();
        if let Some(value) = object.get("output_modalities") {
            next.modalities = Modalities::parse(value)?;
        }
        if let Some(value) = object.get("tool_choice") {
            next.tool_choice = serde_json::from_value(value.clone())
                .context("tool_choice must be auto, none or required")?;
        }
        if let Some(value) = object.get("max_output_tokens") {
            next.budget = OutputBudget::parse(value)?;
        }
        ensure!(
            audio || !matches!(next.modalities, Modalities::Audio | Modalities::Both),
            "output_modalities requests audio but this realtime profile declares no audio output; choose text or a speech-capable package"
        );
        Ok(next)
    }
}
pub(super) struct Capabilities<'a> {
    pub(super) contract: &'a RealtimeContract,
    pub(super) tool_protocol: Option<&'a ToolProtocol>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SessionSettings {
    pub(super) mode: RealtimeMode,
    pub(super) instructions: String,
    pub(super) input_rate: NonZeroU32,
    pub(super) input_language: Option<String>,
    pub(super) input_prompt: String,
    pub(super) voice: Option<String>,
    pub(super) output_language: Option<String>,
    pub(super) tools: Value,
    pub(super) response: ResponseOptions,
}
impl SessionSettings {
    pub(super) fn new(mode: RealtimeMode, input_rate: NonZeroU32, budget: OutputBudget) -> Self {
        Self {
            mode,
            instructions: String::new(),
            input_rate,
            input_language: None,
            input_prompt: String::new(),
            voice: None,
            output_language: None,
            tools: json!([]),
            response: ResponseOptions {
                modalities: Modalities::Auto,
                tool_choice: ToolChoiceMode::Auto,
                budget,
            },
        }
    }
    /// Validate a complete candidate before replacing live settings. Invalid
    /// updates cannot leave a partially reconfigured native connection.
    pub(super) fn update(
        &mut self,
        update: &Value,
        capabilities: &Capabilities<'_>,
        started: bool,
    ) -> anyhow::Result<()> {
        let object = update
            .as_object()
            .context("session.update.session must be an object")?;
        let mut next = self.clone();
        if let Some(value) = object.get("type") {
            let mode = match value.as_str() {
                Some("realtime") => {
                    if self.mode == RealtimeMode::Duplex {
                        RealtimeMode::Duplex
                    } else {
                        RealtimeMode::Realtime
                    }
                }
                Some("transcription") => RealtimeMode::Transcription,
                _ => anyhow::bail!("session.type must be realtime or transcription"),
            };
            ensure!(
                mode == self.mode || (!started && self.mode != RealtimeMode::Duplex),
                "session type cannot change after the first audio/message or for a duplex model"
            );
            ensure!(
                capabilities.contract.supports(mode),
                "realtime profile '{}' does not support session type {mode:?}; select a compatible model",
                capabilities.contract.profile()
            );
            next.mode = mode;
        }
        if let Some(value) = object.get("instructions") {
            ensure!(
                !started,
                "session.instructions cannot change after the first audio or message"
            );
            next.instructions = value
                .as_str()
                .context("session.instructions must be a string")?
                .into();
        }
        if let Some(value) = object.get("tools") {
            ensure!(
                !started,
                "session.tools cannot change after the first audio or message"
            );
            crate::tool_protocol::responses_tools(value.clone())?;
            next.tools = value.clone();
        }
        next.response = next.response.updated(
            update,
            capabilities.contract.spec().output_sample_rate_hz.is_some(),
        )?;
        if let Some(audio) = object.get("audio") {
            let audio = audio
                .as_object()
                .context("session.audio must be an object")?;
            if let Some(input) = audio.get("input") {
                let input = input
                    .as_object()
                    .context("session.audio.input must be an object")?;
                if let Some(format) = input.get("format") {
                    let format = format
                        .as_object()
                        .context("session.audio.input.format must be an object")?;
                    if let Some(kind) = format.get("type") {
                        ensure!(
                            kind == "audio/pcm",
                            "session.audio.input.format.type must be audio/pcm (mono PCM16)"
                        );
                    }
                    if let Some(rate) = format.get("rate") {
                        let rate=rate.as_u64().filter(|rate|(8000..=192000).contains(rate)).context("session.audio.input.format.rate must be an integer between 8000 and 192000 Hz")?;
                        next.input_rate = NonZeroU32::new(rate as u32)
                            .context("positive PCM input rate became zero")?;
                    }
                }
                if let Some(detection) = input.get("turn_detection") {
                    ensure!(
                        detection.is_null(),
                        "session.audio.input.turn_detection must be null; commit input_audio_buffer explicitly for this protocol"
                    );
                }
                if let Some(transcription) = input.get("transcription") {
                    let transcription = transcription
                        .as_object()
                        .context("session.audio.input.transcription must be an object")?;
                    if let Some(value) = transcription.get("language") {
                        next.input_language =
                            selection(value, "session.audio.input.transcription.language")?;
                    }
                    if let Some(value) = transcription.get("prompt") {
                        next.input_prompt = value
                            .as_str()
                            .context("session.audio.input.transcription.prompt must be a string")?
                            .into();
                    }
                }
            }
            if let Some(output) = audio.get("output") {
                ensure!(
                    capabilities.contract.spec().output_sample_rate_hz.is_some(),
                    "session.audio.output requires a realtime profile declaring native audio output"
                );
                let output = output
                    .as_object()
                    .context("session.audio.output must be an object")?;
                if let Some(value) = output.get("voice") {
                    next.voice = selection(
                        if value.is_object() {
                            value
                                .get("id")
                                .context("session.audio.output.voice requires an id string")?
                        } else {
                            value
                        },
                        "session.audio.output.voice",
                    )?;
                }
                if let Some(value) = output.get("language") {
                    next.output_language = selection(value, "session.audio.output.language")?;
                }
                ensure!(
                    !started
                        || (next.voice == self.voice
                            && next.output_language == self.output_language),
                    "session output voice/language cannot change after the first audio or message"
                );
                if let Some(format) = output.get("format") {
                    let format = format
                        .as_object()
                        .context("session.audio.output.format must be an object")?;
                    if let Some(kind) = format.get("type") {
                        ensure!(
                            kind == "audio/pcm",
                            "session.audio.output.format.type must be audio/pcm"
                        );
                    }
                    if let Some(rate) = format.get("rate") {
                        ensure!(
                            rate.as_u64()
                                == capabilities
                                    .contract
                                    .spec()
                                    .output_sample_rate_hz
                                    .map(u64::from),
                            "session.audio.output.format.rate must match the package's native output rate {:?}; output resampling is not declared",
                            capabilities.contract.spec().output_sample_rate_hz
                        );
                    }
                }
            }
        }
        let spec = capabilities.contract.spec();
        spec.voices
            .admit_selection(next.voice.as_deref(), "session.audio.output.voice")
            .map_err(anyhow::Error::msg)?;
        spec.languages
            .admit_selection(
                next.input_language.as_deref(),
                "session.audio.input.transcription.language",
            )
            .map_err(anyhow::Error::msg)?;
        spec.languages
            .admit_selection(
                next.output_language.as_deref(),
                "session.audio.output.language",
            )
            .map_err(anyhow::Error::msg)?;
        let request:ChatCompletionRequest=serde_json::from_value(json!({"model":"realtime","messages":[],"tools":crate::tool_protocol::responses_tools(next.tools.clone())?,"tool_choice":choice_name(&next.response.tool_choice)})).context("realtime tools must contain canonical function definitions")?;
        let has_tools = request
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty());
        ensure!(
            next.response.tool_choice != ToolChoiceMode::Required || has_tools,
            "session.tool_choice required needs at least one function in session.tools"
        );
        if has_tools {
            crate::tool_protocol::validate_request(&request, capabilities.tool_protocol, None)?;
        }
        *self = next;
        Ok(())
    }
    pub(super) fn snapshot(&self, id: &str, model: &str, contract: &RealtimeContract) -> Value {
        let mut input = json!({"format":{"type":"audio/pcm","rate":self.input_rate.get()},"turn_detection":null});
        if self.mode == RealtimeMode::Transcription {
            input["transcription"] =
                json!({"model":model,"language":self.input_language,"prompt":self.input_prompt});
            return json!({"id":id,"object":"realtime.transcription_session","type":"transcription","audio":{"input":input}});
        }
        let mut audio = json!({"input":input});
        if let Some(rate) = contract.spec().output_sample_rate_hz {
            audio["output"] = json!({"format":{"type":"audio/pcm","rate":rate},"voice":self.voice,"language":self.output_language});
        }
        json!({"id":id,"object":"realtime.session","type":"realtime","model":model,"instructions":self.instructions,"output_modalities":self.response.modalities.json(),"max_output_tokens":self.response.budget.json(),"tools":self.tools,"tool_choice":choice_name(&self.response.tool_choice),"audio":audio})
    }
}
fn selection(value: &Value, field: &str) -> anyhow::Result<Option<String>> {
    if value.is_null() {
        return Ok(None);
    }
    let value = value
        .as_str()
        .with_context(|| format!("{field} must be a string or null"))?;
    Ok((!value.is_empty()).then(|| value.to_owned()))
}
fn choice_name(choice: &ToolChoiceMode) -> &'static str {
    match choice {
        ToolChoiceMode::Auto => "auto",
        ToolChoiceMode::None => "none",
        ToolChoiceMode::Required => "required",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn contract() -> RealtimeContract {
        let metadata = serde_yaml::from_str(include_str!(
            "../../../../tests/fixtures/onnx_genai_workflows/realtime/inference_metadata.yaml"
        ))
        .unwrap();
        onnx_genai_metadata::realtime::realtime_contract(&metadata)
            .unwrap()
            .unwrap()
    }
    fn settings() -> SessionSettings {
        SessionSettings::new(
            RealtimeMode::Realtime,
            NonZeroU32::new(16000).unwrap(),
            OutputBudget::Unlimited,
        )
    }
    fn caps(contract: &RealtimeContract) -> Capabilities<'_> {
        Capabilities {
            contract,
            tool_protocol: None,
        }
    }
    #[test]
    fn invalid_session_update_does_not_partially_mutate_prior_settings() {
        let contract = contract();
        let mut settings = settings();
        let initial = settings.clone();
        let error=settings.update(&json!({"instructions":"new instructions","audio":{"input":{"format":{"rate":7999}}}}),&caps(&contract),false).unwrap_err().to_string();
        assert!(
            error.contains("8000") && error.contains("192000"),
            "{error}"
        );
        assert_eq!(settings, initial);
        settings.update(&json!({"instructions":"hello","output_modalities":["text"],"audio":{"input":{"format":{"type":"audio/pcm","rate":44100}}}}),&caps(&contract),false).unwrap();
        assert_eq!(settings.instructions, "hello");
        assert_eq!(settings.input_rate.get(), 44100);
        assert_eq!(settings.response.modalities, Modalities::Text);
        assert!(
            settings
                .update(
                    &json!({"audio":{"input":{"turn_detection":{"type":"server_vad"}}}}),
                    &caps(&contract),
                    false
                )
                .unwrap_err()
                .to_string()
                .contains("turn_detection")
        );
    }
    #[test]
    fn first_audio_or_message_freezes_prompt_and_voice_conditioning() {
        let contract = contract();
        let mut settings = settings();
        let caps = Capabilities {
            contract: &contract,
            tool_protocol: None,
        };
        settings
            .update(
                &json!({"audio":{"output":{"voice":{"id":"alice"},"language":"zh"}}}),
                &caps,
                false,
            )
            .unwrap();
        let configured = settings.clone();
        for update in [
            json!({"instructions":"different"}),
            json!({"tools":[]}),
            json!({"type":"transcription"}),
            json!({"audio":{"output":{"voice":"bob"}}}),
        ] {
            assert!(settings.update(&update, &caps, true).is_err());
            assert_eq!(settings, configured);
        }
        settings.update(&json!({"audio":{"output":{"voice":"alice","language":"zh"},"input":{"format":{"rate":48000}}},"max_output_tokens":42}),&caps,true).unwrap();
        assert_eq!(settings.input_rate.get(), 48000);
        assert_eq!(settings.response.budget.json(), 42);
        assert!(
            settings
                .update(
                    &json!({"audio":{"output":{"voice":"unknown"}}}),
                    &caps,
                    false
                )
                .unwrap_err()
                .to_string()
                .contains("unknown selection")
        );
    }
    #[test]
    fn response_overrides_are_typed_and_do_not_replace_session_defaults() {
        let original = settings().response;
        let override_=original.updated(&json!({"output_modalities":["audio","text","audio"],"tool_choice":"none","max_output_tokens":u64::MAX}),true).unwrap();
        assert_eq!(override_.modalities.json(), json!(["text", "audio"]));
        assert_eq!(override_.budget.json(), i32::MAX);
        assert_eq!(original.budget, OutputBudget::Unlimited);
        for bad in [
            json!({"max_output_tokens":0}),
            json!({"max_output_tokens":-1}),
            json!({"max_output_tokens":1.5}),
            json!({"output_modalities":[]}),
            json!({"output_modalities":["video"]}),
            json!({"tool_choice":{"type":"function","name":"x"}}),
        ] {
            assert!(original.updated(&bad, true).is_err(), "accepted {bad}");
        }
        assert!(
            original
                .updated(&json!({"output_modalities":["audio"]}), false)
                .is_err()
        );
    }
    #[test]
    fn realtime_tools_use_canonical_package_protocol_admission() {
        let contract = contract();
        let mut settings = settings();
        let update = json!({"tools":[{"type":"function","name":"weather","parameters":{"type":"object"},"strict":true}],"tool_choice":"required"});
        let error = settings
            .update(&update, &caps(&contract), false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("package.tool_protocol"), "{error}");
        assert_eq!(settings.tools, json!([]));
        let protocol = crate::tool_protocol::resolve(
            &onnx_genai_metadata::ToolProtocolDeclaration {
                identity: "tagged-json".into(),
                version: "v1".into(),
            },
            std::path::Path::new("fixture/inference_metadata.yaml"),
        )
        .unwrap();
        let capabilities = Capabilities {
            contract: &contract,
            tool_protocol: Some(&protocol),
        };
        settings.update(&update, &capabilities, false).unwrap();
        assert_eq!(settings.response.tool_choice, ToolChoiceMode::Required);
        let lowered = crate::tool_protocol::responses_tools(settings.tools.clone()).unwrap();
        assert_eq!(lowered[0]["function"]["name"], "weather");
        assert!(lowered[0]["function"].get("strict").is_none());
        assert!(
            settings
                .update(
                    &json!({"tools":[],"tool_choice":"required"}),
                    &capabilities,
                    false
                )
                .unwrap_err()
                .to_string()
                .contains("at least one")
        );
    }
    #[test]
    fn session_snapshots_match_realtime_and_transcription_wire_shapes() {
        let contract = contract();
        let mut settings = settings();
        let realtime = settings.snapshot("sess_test", "model_test", &contract);
        assert_eq!(realtime["object"], "realtime.session");
        assert_eq!(realtime["audio"]["input"]["format"]["rate"], 16000);
        assert_eq!(realtime["audio"]["output"]["format"]["rate"], 24000);
        assert_eq!(realtime["max_output_tokens"], "inf");
        settings.update(&json!({"type":"transcription","audio":{"input":{"transcription":{"language":"zh","prompt":"names"}}}}),&caps(&contract),false).unwrap();
        let transcription = settings.snapshot("sess_test", "model_test", &contract);
        assert_eq!(transcription["object"], "realtime.transcription_session");
        assert_eq!(
            transcription["audio"]["input"]["transcription"]["language"],
            "zh"
        );
        assert_eq!(
            transcription["audio"]["input"]["transcription"]["model"],
            "model_test"
        );
        assert!(transcription.get("tools").is_none());
        settings.mode = RealtimeMode::Duplex;
        assert!(
            settings
                .update(&json!({"type":"transcription"}), &caps(&contract), false)
                .is_err()
        );
    }
}
