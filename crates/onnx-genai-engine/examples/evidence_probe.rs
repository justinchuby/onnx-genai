//! Cross-revision evidence probe.
//!
//! Prints one `case<TAB>value` line per observation so a BEFORE/AFTER `diff` is
//! exact rather than eyeballed. It is deliberately written against only the
//! public API that exists on *both* the baseline and the head revision: a probe
//! that needed a symbol one side lacks could not be compiled on both, and a
//! comparison that ran different code on each side proves nothing.
//!
//! Usage: `evidence_probe <model_dir> <prompt> <max_new_tokens>`
//!
//! Every case prints something. A case that cannot run prints its error rather
//! than being skipped, because a silently absent line is indistinguishable from
//! a passing one in a diff.

use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};

fn ids(tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn greedy(max_new_tokens: usize) -> GenerateOptions {
    GenerateOptions {
        max_new_tokens,
        temperature: 0.0,
        greedy: true,
        stop_on_eos: false,
        top_p: 1.0,
        top_k: 0,
        ..GenerateOptions::default()
    }
}

fn request(prompt: &str, options: GenerateOptions) -> GenerateRequest {
    GenerateRequest {
        prompt: GeneratePrompt::Text(prompt.to_string()),
        options,
    }
}

fn emit(case: &str, value: impl std::fmt::Display) {
    println!("{case}\t{value}");
}

fn report<T>(case: &str, outcome: anyhow::Result<T>, render: impl FnOnce(T) -> String) {
    match outcome {
        Ok(value) => emit(case, render(value)),
        Err(error) => emit(case, format!("ERROR {error:#}")),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().expect("model_dir");
    let prompt = args.next().expect("prompt");
    let budget: usize = args
        .next()
        .expect("max_new_tokens")
        .parse()
        .expect("max_new_tokens must be a number");
    let path = std::path::PathBuf::from(&model_dir);

    let mut engine = match Engine::from_dir(&path, EngineConfig::default()) {
        Ok(engine) => engine,
        Err(error) => {
            emit("load", format!("ERROR {error:#}"));
            return;
        }
    };
    emit("load", "ok");

    // Prefill plus cached decode through the stateless entry point, twice: the
    // second run must reproduce the first exactly or greedy decode is not
    // deterministic, which every golden downstream assumes.
    report(
        "greedy.tokens",
        engine.generate(request(&prompt, greedy(budget))),
        |result| ids(&result.token_ids),
    );
    report(
        "greedy.repeat",
        engine.generate(request(&prompt, greedy(budget))),
        |result| ids(&result.token_ids),
    );
    report(
        "greedy.finish_reason",
        engine.generate(request(&prompt, greedy(budget))),
        |result| format!("{:?}", result.finish_reason),
    );

    // A seeded sample must be reproducible across revisions: the RNG draw, the
    // processor chain order and the truncation are all observable here and
    // nowhere else.
    let seeded = || GenerateOptions {
        max_new_tokens: budget,
        temperature: 0.8,
        greedy: false,
        stop_on_eos: false,
        top_p: 0.95,
        top_k: 50,
        seed: Some(20260823),
        ..GenerateOptions::default()
    };
    report(
        "seeded.tokens",
        engine.generate(request(&prompt, seeded())),
        |result| ids(&result.token_ids),
    );
    report(
        "seeded.repeat",
        engine.generate(request(&prompt, seeded())),
        |result| ids(&result.token_ids),
    );

    // Stopping on the package's own declared end tokens, and what it reports
    // when it does. A revision that lost a declared end token runs past it, and
    // only the finish reason and the length say so.
    report(
        "eos.finish_reason",
        engine.generate(request(
            &prompt,
            GenerateOptions {
                max_new_tokens: budget * 4,
                temperature: 0.0,
                greedy: true,
                stop_on_eos: true,
                top_p: 1.0,
                top_k: 0,
                ..GenerateOptions::default()
            },
        )),
        |result| format!("{:?}/{}", result.finish_reason, result.token_ids.len()),
    );

    // A multi-turn conversation: the second turn continues the first rather
    // than restarting it, which is visible as the session's token count and as
    // the prompt prefix the runtime did not recompute.
    let session = engine.create_session();
    match session {
        Ok(session) => {
            emit("session.open", "ok");
            report(
                "session.turn1",
                engine.generate_in_session(session, request(&prompt, greedy(budget))),
                |result| {
                    format!(
                        "{}|prefix={}",
                        ids(&result.token_ids),
                        result.prefix_cache_hit_len
                    )
                },
            );
            report(
                "session.turn2",
                engine.generate_in_session(session, request(&prompt, greedy(budget))),
                |result| {
                    format!(
                        "{}|prefix={}",
                        ids(&result.token_ids),
                        result.prefix_cache_hit_len
                    )
                },
            );
            report("session.tokens", engine.session_token_count(session), |n| {
                n.to_string()
            });
            report("session.close", engine.close_session(session), |()| {
                "ok".to_string()
            });
        }
        Err(error) => emit("session.open", format!("ERROR {error:#}")),
    }
}
