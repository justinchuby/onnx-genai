//! One-way ComfyUI workflow -> canonical inference-metadata conversion.
//!
//! ```text
//! comfyui_to_metadata [--out <path>] [--adapters <path>] [--textproto] <workflow.json> [...]
//! ```
//!
//! Conversion is fail-closed and structural: every node that can reach the
//! workflow's saved image must be understood, and anything the canonical
//! contract cannot state is an error naming the node and the remedy.
//!
//! There is no export direction. ComfyUI is an import source in exactly the way
//! `genai_config.json` is: once the metadata exists, it is the sole source of
//! execution truth, and nothing reads the ComfyUI document again. A reverse
//! synthesizer would have to approximate facts the canonical contract states
//! precisely, so this tool does not offer one.

use std::path::PathBuf;

use onnx_genai_comfyui_config::{ComponentLayout, ConvertOptions, convert_file, to_yaml};

const USAGE: &str = "usage: comfyui_to_metadata [--out <path>] [--adapters <path>] [--textproto] \
                     <workflow.json> [...]\n\n\
                     Lowers a ComfyUI API-format workflow into canonical pipeline.workflow \
                     inference metadata.\n\n\
                     --out <path>       write the YAML document here (default: stdout)\n\
                     --adapters <path>  the package's own `adapters` contract, required when \
                     the workflow selects LoRAs\n\
                     --textproto        reference `*.onnx.textproto` component artifacts";

fn main() {
    if let Err(message) = run() {
        eprintln!("{message}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut out: Option<PathBuf> = None;
    let mut adapters: Option<PathBuf> = None;
    let mut layout = ComponentLayout::default();
    let mut inputs = Vec::new();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--out" => out = Some(PathBuf::from(expect(&mut arguments, "--out")?)),
            "--adapters" => adapters = Some(PathBuf::from(expect(&mut arguments, "--adapters")?)),
            "--textproto" => layout = ComponentLayout::textproto(),
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other if other.starts_with('-') => return Err(format!("unknown option: {other}")),
            other => inputs.push(PathBuf::from(other)),
        }
    }
    if inputs.is_empty() {
        return Err(USAGE.to_owned());
    }
    if out.is_some() && inputs.len() > 1 {
        return Err("--out writes one document, so it accepts exactly one workflow".to_owned());
    }

    let adapters = match adapters {
        Some(path) => {
            let text = std::fs::read_to_string(&path)
                .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
            let value: serde_json::Value = serde_json::from_str(&text)
                .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
            // Accept either a bare adapter contract or a whole metadata document.
            Some(value.get("adapters").cloned().unwrap_or(value))
        }
        None => None,
    };
    let options = ConvertOptions { layout, adapters };

    for input in inputs {
        let (_, document, report) = convert_file(&input, &options)
            .map_err(|error| format!("failed to convert {}: {error}", input.display()))?;
        let yaml = to_yaml(&document).map_err(|error| error.to_string())?;
        match &out {
            Some(path) => {
                if let Some(parent) = path
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                {
                    std::fs::create_dir_all(parent).map_err(|error| {
                        format!("failed to create {}: {error}", parent.display())
                    })?;
                }
                std::fs::write(path, &yaml)
                    .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
                eprintln!("wrote {}", path.display());
            }
            None => print!("{yaml}"),
        }
        for ignored in &report.ignored_nodes {
            eprintln!("ignored (cannot reach the saved image): {ignored}");
        }
        eprintln!(
            "converted {}: {} steps ({}..{}), solver={}, spacing={}, prediction={}, guidance={}, \
             controlnets={}, adapters={}",
            input.display(),
            report.plan.steps,
            report.plan.start_step,
            report.plan.end_step,
            report.plan.solver.as_str(),
            report.plan.spacing.as_str(),
            report.plan.prediction.as_str(),
            report
                .plan
                .guidance
                .as_ref()
                .map_or("off".to_owned(), |guidance| guidance.scale.to_string()),
            report.plan.controlnets.len(),
            report.adapters.len(),
        );
    }
    Ok(())
}

fn expect(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))
}
