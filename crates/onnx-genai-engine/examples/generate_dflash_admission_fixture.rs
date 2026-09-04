use std::path::Path;

#[path = "../tests/support/dflash_admission_fixture.rs"]
mod dflash_admission_fixture;

fn main() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dflash-admission");
    dflash_admission_fixture::write(&root)?;
    println!("regenerated {}", root.display());
    Ok(())
}
