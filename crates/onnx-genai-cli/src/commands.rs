use std::path::PathBuf;

use onnx_genai::engine::EngineDecodeBackend;
use onnx_genai::ort::{SessionOptions, profile::TraceVerbosity};

use super::interactive::{Backend, SessionSettings};

#[derive(Debug, PartialEq, Eq)]
pub(super) enum ReplCommand {
    Help,
    Reset,
    ToggleRaw,
    ToggleStats,
    /// Show what the KV page pool is holding.
    Pages,
    /// Turn the profile report on or off, or report its state.
    Profile(Option<String>),
    /// Load a different model, or report the current one.
    Model(Option<String>),
    /// Switch execution provider, or report the current one.
    ExecutionProvider(Option<String>),
    /// Switch decode backend, or report the current one.
    DecodeBackend(Option<String>),
    System(Option<String>),
    Image {
        path: Option<String>,
        prompt: Option<String>,
    },
    Audio {
        path: Option<String>,
        prompt: Option<String>,
    },
    Unknown(String),
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum ReplLine {
    Command(ReplCommand),
    Prompt(String),
    Empty,
}

/// Load a new session, leaving the caller's current one untouched on failure.
///
/// Building the replacement before swapping is what lets a rejected execution
/// provider or backend be reported without ending the session: an interactive
/// user should be able to try `cuda`, be told it is unavailable, and carry on.
pub(super) fn reload(settings: &SessionSettings) -> anyhow::Result<Backend> {
    Backend::open(settings)
}

/// Providers a default session resolves to, for commands that do not let the
/// user choose one mid-run.
pub(super) fn resolved_default_providers() -> String {
    let options = SessionOptions::default();
    let names = options
        .execution_providers
        .iter()
        .map(|provider| provider.selection.name.as_str())
        .collect::<Vec<_>>();
    if names.is_empty() {
        "cpu".to_string()
    } else {
        names.join(", ")
    }
}

/// Execution providers this session can be switched to.
///
/// Delegated rather than listed here: the runtime owns the mapping from
/// provider names to behavior, and a menu kept in the CLI would drift from it —
/// as it did, by omitting the Metal plugin that macOS auto-selects and then
/// refusing `/ep metal` on a machine already running on it.
pub(super) fn available_execution_providers() -> Vec<&'static str> {
    let mut providers = vec!["auto"];
    providers.extend(onnx_genai::ort::selectable_execution_providers());
    providers
}

pub(super) fn parse_decode_backend(name: &str) -> Result<EngineDecodeBackend, String> {
    match name {
        "auto" => Ok(EngineDecodeBackend::Auto),
        "ort" => Ok(EngineDecodeBackend::Ort),
        "native" => Ok(EngineDecodeBackend::Native),
        other => Err(format!(
            "What: {other:?} is not a decode backend. \
             Why: the choices are fixed by the engine, not by the model. \
             How: use auto, ort, or native."
        )),
    }
}

/// What `/profile <rest>` asked for.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum ProfileSetting {
    /// Report the current state.
    Show,
    /// Turn the report and the timeline on or off together.
    Toggle(bool),
    /// Write the timeline here from now on.
    Trace(PathBuf),
    /// Stop writing a timeline, keeping the report.
    NoTrace,
    /// Record this much detail.
    Verbosity(TraceVerbosity),
}

/// Parse the argument to `/profile`.
///
/// Turning profiling on turns the timeline on with it, at the most detailed
/// level: someone who asks to profile interactively wants to see everything,
/// and can turn detail down afterwards. The one-shot flags keep their own
/// defaults, where the cost matters more and the intent is stated up front.
pub(super) fn parse_profile_setting(argument: Option<&str>) -> Result<ProfileSetting, String> {
    let Some(argument) = argument.map(str::trim).filter(|text| !text.is_empty()) else {
        return Ok(ProfileSetting::Show);
    };
    let (word, rest) = match argument.split_once(char::is_whitespace) {
        Some((word, rest)) => (word, rest.trim()),
        None => (argument, ""),
    };
    match word {
        "on" | "off" => Ok(ProfileSetting::Toggle(word == "on")),
        "trace" => {
            if rest.is_empty() {
                return Err("What: /profile trace needs a file to write to. \
                     Why: the timeline is a document, so it has to go somewhere. \
                     How: write /profile trace run.perfetto.json, or /profile trace off to stop \
                     writing one."
                    .to_string());
            }
            if rest == "off" {
                return Ok(ProfileSetting::NoTrace);
            }
            Ok(ProfileSetting::Trace(PathBuf::from(rest)))
        }
        "verbosity" | "detail" => {
            let levels = TraceVerbosity::ALL
                .iter()
                .map(|level| level.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            if rest.is_empty() {
                return Err(format!(
                    "What: /profile {word} needs a level. \
                     Why: it chooses how much the timeline records. \
                     How: one of {levels} -- for example /profile {word} full."
                ));
            }
            TraceVerbosity::parse(rest)
                .map(ProfileSetting::Verbosity)
                .ok_or_else(|| {
                    format!(
                        "What: {rest:?} is not a detail level. \
                     Why: the timeline records one of a fixed set of levels. \
                     How: use one of {levels}."
                    )
                })
        }
        other => Err(format!(
            "What: {other:?} is not a /profile setting. \
             Why: /profile takes on, off, trace <path>, or verbosity <level>, or nothing to \
             report the current state. \
             How: try /profile on, or /profile trace run.perfetto.json."
        )),
    }
}

/// Start or stop timeline recording at the given detail.
///
/// A no-op without the native backend, whose runtime the tracer records; the
/// ORT-hosted path records its own spans through the profiler either way.
pub(super) fn set_trace_recording(enabled: bool, verbosity: TraceVerbosity) {
    #[cfg(feature = "native-backend")]
    onnx_genai::engine::runtime_trace::set_recording(enabled, verbosity);
    #[cfg(not(feature = "native-backend"))]
    {
        let _ = (enabled, verbosity);
    }
}

/// A command's single optional argument, with blank treated as absent so
/// `/ep` reports the current provider and `/ep cuda` sets one.
fn argument_of(arguments: &str) -> Option<String> {
    let trimmed = arguments.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

pub(super) fn parse_repl_line(line: &str) -> ReplLine {
    if line.trim().is_empty() {
        return ReplLine::Empty;
    }
    let Some(command_line) = line.strip_prefix('/') else {
        return ReplLine::Prompt(line.to_string());
    };

    let mut parts = command_line.splitn(2, char::is_whitespace);
    let command = parts.next().unwrap_or_default();
    let arguments = parts.next().unwrap_or_default().trim();
    let attachment_command = |is_image| {
        let mut attachment_parts = arguments.splitn(2, char::is_whitespace);
        let path = attachment_parts
            .next()
            .filter(|path| !path.is_empty())
            .map(ToString::to_string);
        let prompt = attachment_parts
            .next()
            .map(str::trim)
            .filter(|prompt| !prompt.is_empty())
            .map(ToString::to_string);
        if is_image {
            ReplCommand::Image { path, prompt }
        } else {
            ReplCommand::Audio { path, prompt }
        }
    };

    let command = match command {
        "help" => ReplCommand::Help,
        "reset" => ReplCommand::Reset,
        "raw" => ReplCommand::ToggleRaw,
        "stats" => ReplCommand::ToggleStats,
        "pages" => ReplCommand::Pages,
        "profile" => ReplCommand::Profile((!arguments.is_empty()).then(|| arguments.to_string())),
        "model" => ReplCommand::Model(argument_of(arguments)),
        "ep" => ReplCommand::ExecutionProvider(argument_of(arguments)),
        "backend" => ReplCommand::DecodeBackend(argument_of(arguments)),
        "system" => ReplCommand::System((!arguments.is_empty()).then(|| arguments.to_string())),
        "image" => attachment_command(true),
        "audio" => attachment_command(false),
        _ => ReplCommand::Unknown(format!("/{command}")),
    };
    ReplLine::Command(command)
}
