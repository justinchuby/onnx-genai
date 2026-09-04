use std::path::Path;

#[path = "../tests/support/dflash_admission_fixture.rs"]
mod dflash_admission_fixture;

fn main() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dflash-admission");
    let mut args = std::env::args().skip(1);
    let mode = args.next();
    anyhow::ensure!(
        args.next().is_none(),
        "expected no argument or --check, but received extra arguments"
    );
    match mode.as_deref() {
        None => {
            dflash_admission_fixture::write(&root)?;
            println!("regenerated {}", root.display());
        }
        Some("--check") => {
            dflash_admission_fixture::check(&root)?;
            println!("{} is current", root.display());
        }
        Some(argument) => {
            anyhow::bail!("unexpected argument '{argument}'; expected no argument or --check")
        }
    }
    Ok(())
}
