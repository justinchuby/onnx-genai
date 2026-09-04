use std::{env, error::Error, fs, io, path::PathBuf};

use onnx_genai_metadata::extensions::extension_registry_markdown_bytes;

fn main() -> Result<(), Box<dyn Error>> {
    let check = parse_check_arg()?;
    let path = registry_path();
    let generated = extension_registry_markdown_bytes();
    if check {
        let committed = fs::read(&path).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("failed to read {}: {error}", path.display()),
            )
        })?;
        if committed != generated {
            return Err(io::Error::other(format!(
                "{} is stale or not canonical UTF-8/LF; regenerate with \
                 `cargo run -p onnx-genai-metadata --bin gen_extension_registry`",
                path.display()
            ))
            .into());
        }
        println!("{} is current", path.display());
    } else {
        fs::write(&path, generated).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("failed to write {}: {error}", path.display()),
            )
        })?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

fn parse_check_arg() -> Result<bool, Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let check = match args.next() {
        None => false,
        Some(argument) if argument == "--check" => true,
        Some(argument) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "unexpected argument '{}'; expected no argument or --check",
                    argument.to_string_lossy()
                ),
            )
            .into());
        }
    };
    if let Some(argument) = args.next() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unexpected extra argument '{}'", argument.to_string_lossy()),
        )
        .into());
    }
    Ok(check)
}

fn registry_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("docs/genai/METADATA_EXTENSION_REGISTRY.md")
}
