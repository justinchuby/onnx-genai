use std::path::{Path, PathBuf};

use onnx_genai::ort::profile::TraceVerbosity;

use super::interactive::{Backend, ReplInputMode, SessionSettings};

#[derive(Debug, PartialEq, Eq)]
pub(super) enum ReplCommand {
    Help(Option<String>),
    Reset,
    ToggleRaw,
    ToggleStats,
    Pages,
    Profile(Option<String>),
    Model(Option<String>),
    Session,
    ExecutionProvider(Option<String>),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CommandCategory {
    Help,
    Session,
    ModelRuntime,
    Diagnostics,
    Input,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplCommandKind {
    Help,
    Reset,
    Raw,
    Stats,
    Pages,
    Profile,
    Model,
    Session,
    ExecutionProvider,
    DecodeBackend,
    System,
    Image,
    Audio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CompletionSource {
    None,
    Files,
    ExecutionProviders,
    DecodeBackends,
    ProfileSettings,
}

#[derive(Debug)]
pub(super) struct ReplCommandSpec {
    pub(super) name: &'static str,
    pub(super) aliases: &'static [&'static str],
    pub(super) usage: &'static str,
    pub(super) summary: &'static str,
    pub(super) category: CommandCategory,
    pub(super) completion: CompletionSource,
    kind: ReplCommandKind,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct ReplCompletion {
    pub(super) replacement: String,
    pub(super) display: String,
    pub(super) description: Option<String>,
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) append_space: bool,
}

const COMMANDS: &[ReplCommandSpec] = &[
    ReplCommandSpec {
        name: "help",
        aliases: &[],
        usage: "/help",
        summary: "Show REPL commands.",
        category: CommandCategory::Help,
        completion: CompletionSource::None,
        kind: ReplCommandKind::Help,
    },
    ReplCommandSpec {
        name: "reset",
        aliases: &[],
        usage: "/reset",
        summary: "Clear conversation history and pending attachments.",
        category: CommandCategory::Session,
        completion: CompletionSource::None,
        kind: ReplCommandKind::Reset,
    },
    ReplCommandSpec {
        name: "raw",
        aliases: &[],
        usage: "/raw",
        summary: "Toggle raw prompting without the model chat template.",
        category: CommandCategory::Input,
        completion: CompletionSource::None,
        kind: ReplCommandKind::Raw,
    },
    ReplCommandSpec {
        name: "stats",
        aliases: &[],
        usage: "/stats",
        summary: "Toggle compact per-turn stats.",
        category: CommandCategory::Diagnostics,
        completion: CompletionSource::None,
        kind: ReplCommandKind::Stats,
    },
    ReplCommandSpec {
        name: "pages",
        aliases: &[],
        usage: "/pages",
        summary: "Show the current KV page-pool contents.",
        category: CommandCategory::Diagnostics,
        completion: CompletionSource::None,
        kind: ReplCommandKind::Pages,
    },
    ReplCommandSpec {
        name: "profile",
        aliases: &[],
        usage: "/profile [on|off|trace <path>|verbosity <decisions|ops|full>]",
        summary: "Control per-turn profiling and Perfetto trace output.",
        category: CommandCategory::Diagnostics,
        completion: CompletionSource::ProfileSettings,
        kind: ReplCommandKind::Profile,
    },
    ReplCommandSpec {
        name: "model",
        aliases: &[],
        usage: "/model [path]",
        summary: "Reload another model, or print the current session.",
        category: CommandCategory::ModelRuntime,
        completion: CompletionSource::Files,
        kind: ReplCommandKind::Model,
    },
    ReplCommandSpec {
        name: "session",
        aliases: &[],
        usage: "/session",
        summary: "Print a structured summary of the current session.",
        category: CommandCategory::Session,
        completion: CompletionSource::None,
        kind: ReplCommandKind::Session,
    },
    ReplCommandSpec {
        name: "ep",
        aliases: &[],
        usage: "/ep [name]",
        summary: "Switch execution provider, or report the current one.",
        category: CommandCategory::ModelRuntime,
        completion: CompletionSource::ExecutionProviders,
        kind: ReplCommandKind::ExecutionProvider,
    },
    ReplCommandSpec {
        name: "backend",
        aliases: &[],
        usage: "/backend [auto|ort|native]",
        summary: "Switch decode backend, or report the current one.",
        category: CommandCategory::ModelRuntime,
        completion: CompletionSource::DecodeBackends,
        kind: ReplCommandKind::DecodeBackend,
    },
    ReplCommandSpec {
        name: "system",
        aliases: &[],
        usage: "/system <text>",
        summary: "Set or clear the system message.",
        category: CommandCategory::Input,
        completion: CompletionSource::None,
        kind: ReplCommandKind::System,
    },
    ReplCommandSpec {
        name: "image",
        aliases: &[],
        usage: "/image <path> [prompt text]",
        summary: "Stage an image attachment for the next turn.",
        category: CommandCategory::Input,
        completion: CompletionSource::Files,
        kind: ReplCommandKind::Image,
    },
    ReplCommandSpec {
        name: "audio",
        aliases: &[],
        usage: "/audio <path> [prompt text]",
        summary: "Stage an audio attachment for the next turn.",
        category: CommandCategory::Input,
        completion: CompletionSource::Files,
        kind: ReplCommandKind::Audio,
    },
];

pub(super) fn command_registry() -> &'static [ReplCommandSpec] {
    COMMANDS
}

pub(super) fn render_repl_help() -> String {
    command_registry()
        .iter()
        .map(|command| command.usage)
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn render_command_help(command: &str) -> Option<String> {
    let command = command.trim().trim_start_matches('/');
    find_command(command).map(|spec| {
        format!(
            "{}\n  category: {}\n  {}",
            spec.usage,
            spec.category.name(),
            spec.summary
        )
    })
}

impl CommandCategory {
    fn name(self) -> &'static str {
        match self {
            Self::Help => "help",
            Self::Session => "session",
            Self::ModelRuntime => "model/runtime",
            Self::Diagnostics => "diagnostics",
            Self::Input => "input",
        }
    }
}

/// Load a new session, leaving the caller's current one untouched on failure.
///
/// Building the replacement before swapping is what lets a rejected execution
/// provider or backend be reported without ending the session: an interactive
/// user should be able to try `cuda`, be told it is unavailable, and carry on.
pub(super) fn reload(settings: &SessionSettings) -> anyhow::Result<Backend> {
    Backend::open(settings)
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

pub(super) use onnx_genai_server::runtime_args::parse_decode_backend;

/// What `/profile <rest>` asked for.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum ProfileSetting {
    Show,
    Toggle(bool),
    Trace(PathBuf),
    NoTrace,
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

fn find_command(command: &str) -> Option<&'static ReplCommandSpec> {
    COMMANDS
        .iter()
        .find(|spec| spec.name == command || spec.aliases.contains(&command))
}

fn parse_attachment(arguments: &str, is_image: bool) -> ReplCommand {
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
}

pub(super) fn parse_repl_line(line: &str, mode: ReplInputMode) -> ReplLine {
    if line.trim().is_empty() {
        return ReplLine::Empty;
    }
    if matches!(mode, ReplInputMode::Tty)
        && let Some(prompt) = line.strip_prefix("//")
    {
        return ReplLine::Prompt(format!("/{prompt}"));
    }
    let Some(command_line) = line.strip_prefix('/') else {
        return ReplLine::Prompt(line.to_string());
    };

    let mut parts = command_line.splitn(2, char::is_whitespace);
    let command = parts.next().unwrap_or_default();
    let arguments = parts.next().unwrap_or_default().trim();
    let Some(spec) = find_command(command) else {
        return ReplLine::Command(ReplCommand::Unknown(format!("/{command}")));
    };
    let command = match spec.kind {
        ReplCommandKind::Help => ReplCommand::Help(match mode {
            ReplInputMode::Tty => argument_of(arguments),
            ReplInputMode::Plain => None,
        }),
        ReplCommandKind::Reset => ReplCommand::Reset,
        ReplCommandKind::Raw => ReplCommand::ToggleRaw,
        ReplCommandKind::Stats => ReplCommand::ToggleStats,
        ReplCommandKind::Pages => ReplCommand::Pages,
        ReplCommandKind::Profile => ReplCommand::Profile(argument_of(arguments)),
        ReplCommandKind::Model => ReplCommand::Model(argument_of(arguments)),
        ReplCommandKind::Session => ReplCommand::Session,
        ReplCommandKind::ExecutionProvider => {
            ReplCommand::ExecutionProvider(argument_of(arguments))
        }
        ReplCommandKind::DecodeBackend => ReplCommand::DecodeBackend(argument_of(arguments)),
        ReplCommandKind::System => ReplCommand::System(argument_of(arguments)),
        ReplCommandKind::Image => parse_attachment(arguments, true),
        ReplCommandKind::Audio => parse_attachment(arguments, false),
    };
    ReplLine::Command(command)
}

pub(super) fn complete_repl_line(line: &str, pos: usize) -> Vec<ReplCompletion> {
    let pos = pos.min(line.len());
    let input = &line[..pos];
    if !input.starts_with('/') || input.starts_with("//") {
        return Vec::new();
    }
    let without_slash = &input[1..];
    let Some((command, arguments)) = without_slash.split_once(char::is_whitespace) else {
        let prefix = without_slash;
        return COMMANDS
            .iter()
            .filter(|spec| spec.name.starts_with(prefix))
            .map(|spec| ReplCompletion {
                replacement: format!("/{}", spec.name),
                display: spec.usage.to_string(),
                description: Some(spec.summary.to_string()),
                start: 0,
                end: pos,
                append_space: true,
            })
            .collect();
    };

    let Some(spec) = find_command(command) else {
        return Vec::new();
    };
    let arg_start = 1 + command.len() + 1;
    let prefix = arguments;
    complete_argument(spec.completion, prefix, arg_start, pos)
}

fn complete_argument(
    source: CompletionSource,
    prefix: &str,
    start: usize,
    end: usize,
) -> Vec<ReplCompletion> {
    match source {
        CompletionSource::None => Vec::new(),
        CompletionSource::Files => complete_paths(prefix, start, end),
        CompletionSource::ExecutionProviders => complete_fixed(
            prefix,
            available_execution_providers()
                .into_iter()
                .map(str::to_string)
                .collect(),
            start,
            end,
        ),
        CompletionSource::DecodeBackends => complete_fixed(
            prefix,
            ["auto", "ort", "native"]
                .into_iter()
                .map(String::from)
                .collect(),
            start,
            end,
        ),
        CompletionSource::ProfileSettings => complete_profile(prefix, start, end),
    }
}

fn complete_profile(prefix: &str, start: usize, end: usize) -> Vec<ReplCompletion> {
    let trimmed_start = prefix.len() - prefix.trim_start().len();
    let trimmed = prefix.trim_start();
    if let Some(rest) = trimmed
        .strip_prefix("verbosity")
        .or_else(|| trimmed.strip_prefix("detail"))
    {
        let level_prefix = rest.trim_start();
        let level_start = end.saturating_sub(level_prefix.len());
        return complete_fixed(
            level_prefix,
            TraceVerbosity::ALL
                .iter()
                .map(|level| level.to_string())
                .collect(),
            level_start,
            end,
        );
    }
    if let Some(rest) = trimmed.strip_prefix("trace") {
        let path_prefix = rest.trim_start();
        if path_prefix.is_empty() {
            return complete_fixed(
                "trace",
                vec!["trace".to_string()],
                start + trimmed_start,
                end,
            );
        }
        let path_start = end.saturating_sub(path_prefix.len());
        return complete_paths(path_prefix, path_start, end);
    }
    complete_fixed(
        trimmed,
        ["on", "off", "trace", "verbosity"]
            .into_iter()
            .map(String::from)
            .collect(),
        start + trimmed_start,
        end,
    )
}

fn complete_fixed(
    prefix: &str,
    values: Vec<String>,
    start: usize,
    end: usize,
) -> Vec<ReplCompletion> {
    values
        .into_iter()
        .filter(|value| value.starts_with(prefix))
        .map(|value| ReplCompletion {
            replacement: value.clone(),
            display: value,
            description: None,
            start,
            end,
            append_space: true,
        })
        .collect()
}

fn complete_paths(prefix: &str, start: usize, end: usize) -> Vec<ReplCompletion> {
    let path = Path::new(prefix);
    let (base_dir, file_prefix) = match path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        Some(parent) => (
            parent.to_path_buf(),
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(""),
        ),
        None => (PathBuf::from("."), prefix),
    };
    let Ok(entries) = std::fs::read_dir(&base_dir) else {
        return Vec::new();
    };
    let mut completions = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            name.starts_with(file_prefix).then(|| {
                let mut replacement = match path.parent() {
                    Some(parent) if !parent.as_os_str().is_empty() => {
                        parent.join(&name).display().to_string()
                    }
                    _ => name.clone(),
                };
                let is_dir = entry.file_type().is_ok_and(|kind| kind.is_dir());
                if is_dir {
                    replacement.push(std::path::MAIN_SEPARATOR);
                }
                ReplCompletion {
                    replacement: replacement.clone(),
                    display: replacement,
                    description: None,
                    start,
                    end,
                    append_space: !is_dir,
                }
            })
        })
        .collect::<Vec<_>>();
    completions.sort_by(|left, right| left.display.cmp(&right.display));
    completions
}
