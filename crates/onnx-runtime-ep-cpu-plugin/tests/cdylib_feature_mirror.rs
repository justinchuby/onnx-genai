//! Fail loudly when a cdylib-affecting feature is not mirrored into the resolver.
//!
//! `cdylib_resolve::features` tells the resolver which features to rebuild the
//! cdylib with. A feature missing from that list does not fail: the resolver
//! rebuilds the library *without* it, overwriting whatever the developer built,
//! and every test in the process then loads a library that is not the one under
//! test. Nothing reports this — the suite goes green against the wrong binary.
//!
//! That is not hypothetical. `dispatch_probe` was omitted when it was added and
//! the probe reported zero for every phase. `research_ablate`, a local research
//! feature for issue #1077, hit it a second time: a whole stage-ablation ladder
//! read as "every stage costs nothing", because the tiers were being applied to
//! a library the harness had already replaced. Both cost a measurement cycle and
//! neither produced a failing test.
//!
//! This test converts that class of mistake into a build failure at the moment
//! the feature is declared.

mod cdylib_resolve;

/// Feature names declared in this package's `Cargo.toml`, in declaration order.
///
/// Hand-rolled rather than pulling in a TOML parser as a dev-dependency: the
/// `[features]` table is a flat `name = [...]` list, and a parser that only has
/// to survive that is smaller than the dependency.
fn declared_features(manifest: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_features = false;
    for raw in manifest.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            in_features = line == "[features]";
            continue;
        }
        if !in_features || line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((name, _)) = line.split_once('=') {
            let name = name.trim();
            if !name.is_empty() && !name.starts_with('#') {
                out.push(name.to_owned());
            }
        }
    }
    out
}

#[test]
fn every_declared_feature_is_mirrored_into_the_cdylib_resolver() {
    let manifest = include_str!("../Cargo.toml");
    let declared = declared_features(manifest);

    assert!(
        !declared.is_empty(),
        "parsed no features out of Cargo.toml — the parser has drifted from the \
         manifest format, so this test is no longer checking anything"
    );

    let missing: Vec<&String> = declared
        .iter()
        .filter(|f| !cdylib_resolve::MIRRORED_FEATURES.contains(&f.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "features declared in Cargo.toml but not mirrored into \
         cdylib_resolve::features(): {missing:?}\n\
         A feature the resolver does not know about is silently dropped when it \
         rebuilds the cdylib, so the tests load a library built WITHOUT it and \
         still pass. Add it to both `features()` and `MIRRORED_FEATURES`, or, if \
         it provably cannot change the cdylib, add it to `MIRRORED_FEATURES` with \
         a comment saying why."
    );

    let stale: Vec<&&str> = cdylib_resolve::MIRRORED_FEATURES
        .iter()
        .filter(|f| !declared.iter().any(|d| d == *f))
        .collect();
    assert!(
        stale.is_empty(),
        "MIRRORED_FEATURES names features that Cargo.toml no longer declares: \
         {stale:?} — the list has drifted and would hide a real omission"
    );
}

#[test]
fn the_manifest_parser_finds_the_features_table_and_stops_at_the_next_one() {
    // Pins the parser itself: without this, a manifest format change would make
    // `declared_features` return nothing and the check above would vacuously
    // pass on an empty list. (The emptiness assert catches that too; this pins
    // the boundary behaviour that makes the emptiness assert meaningful.)
    let manifest = "\
[features]
# a comment
alpha = [\"x/y\"]
beta = []

[dependencies]
gamma = { workspace = true }
";
    assert_eq!(declared_features(manifest), vec!["alpha", "beta"]);
}
