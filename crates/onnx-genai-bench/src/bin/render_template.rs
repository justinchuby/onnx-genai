//! Render a fixed chat-template scenario without loading a model.

use std::path::PathBuf;

use onnx_genai_ort::{ChatMessage, ChatTemplate};

struct Args {
    model: PathBuf,
    scenario: String,
}

fn parse_args() -> Args {
    let mut model = None;
    let mut scenario = None;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = args.next().map(PathBuf::from),
            "--scenario" => scenario = args.next(),
            "--help" | "-h" => {
                eprintln!(
                    "Usage: render_template --model <dir> --scenario <user|system_user|multiturn>"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument: {other}"),
        }
    }

    Args {
        model: model.expect("--model needs a value"),
        scenario: scenario.expect("--scenario needs a value"),
    }
}

fn messages(scenario: &str) -> Vec<ChatMessage> {
    match scenario {
        "user" => vec![ChatMessage::user(
            "Write a short paragraph about the history of computing.",
        )],
        "system_user" => vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hello, who are you?"),
        ],
        "multiturn" => vec![
            ChatMessage::user("Hi"),
            ChatMessage::assistant("Hello! How can I help?"),
            ChatMessage::user("Explain gravity briefly."),
        ],
        _ => panic!("unknown scenario {scenario:?}; expected user, system_user, or multiturn"),
    }
}

fn main() {
    let args = parse_args();
    let template = ChatTemplate::from_model_dir(&args.model).expect("load chat template");
    let rendered = template
        .render(&messages(&args.scenario), None, true)
        .expect("render chat template");
    print!("{rendered}");
}
