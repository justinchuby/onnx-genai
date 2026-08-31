//! Schema version normalization and the gate that runs before typed parsing.
//!
//! Four spellings of the first schema version are in the wild — absent, `v1`,
//! `1`, and `1.0` — because nothing ever forced a canonical one. They all mean
//! the same document, so a reader normalizes rather than compares strings.
//!
//! The gate matters more than the normalization. Every structure in this schema
//! denies unknown fields, so a reader that met a newer document would report the
//! first field it did not recognize: `unknown field 'batch_capacity'`, which
//! sends a reader looking for a typo in a document that is perfectly correct and
//! merely newer. Reading the version *before* handing the bytes to `serde` turns
//! that into the true statement — this document is newer than this runtime —
//! which is the difference between an upgrade and a bug hunt.

use std::fmt;

/// A `<major>.<minor>` inference-metadata schema version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SchemaVersion {
    pub major: u32,
    pub minor: u32,
}

impl SchemaVersion {
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }
}

impl fmt::Display for SchemaVersion {
    /// The canonical spelling: `v<major>.<minor>`, which is what a document that
    /// needs a version this build knows about should write.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "v{}.{}", self.major, self.minor)
    }
}

/// The version an absent, `v1`, `1`, or `1.0` document means.
pub const INITIAL_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1, 0);

/// The newest version this build can read.
pub const SUPPORTED_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1, 7);

/// The version that first carried encoder batching, padding, ownership levels,
/// and the video preprocessing program.
pub const BATCHING_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1, 1);

/// The version that introduced top-level numeric token authority.
pub const TOKEN_AUTHORITY_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1, 2);

/// The version that introduced the exact package tool-call protocol declaration.
pub const TOOL_PROTOCOL_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1, 3);

/// The version that introduced the graph-internal token-context contract.
pub const TOKEN_CONTEXT_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1, 4);

/// The version that made workflow-native, versioned speculative contracts the
/// only portable speculative authority.
pub const CANONICAL_SPECULATION_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1, 6);
/// The version that introduced the generalized DFlash flat-block contract.
pub const DFLASH_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1, 6);

/// The version that introduced output publication families and typed revision
/// envelopes.
pub const OUTPUT_PROTOCOL_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1, 5);

/// The version that made transaction-scoped publication visibility explicit.
pub const PUBLICATION_MODE_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1, 7);

/// A serialized feature whose presence is bounded by one schema version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaFeature {
    OutputProtocols,
    PublicationMode,
}

impl SchemaFeature {
    pub const fn minimum_version(self) -> SchemaVersion {
        match self {
            Self::OutputProtocols => OUTPUT_PROTOCOL_SCHEMA_VERSION,
            Self::PublicationMode => PUBLICATION_MODE_SCHEMA_VERSION,
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::OutputProtocols => {
                "workflow output families, logical streams, and typed revision operations"
            }
            Self::PublicationMode => {
                "workflow transaction publication mode and typed commit/abort reconciliation"
            }
        }
    }

    const fn declaration(self) -> &'static str {
        match self {
            Self::OutputProtocols => {
                "Declare exactly one of `{ kind: materialized }`, `{ kind: events }`, or \
                 `{ kind: revisions, version: \"1\" }`"
            }
            Self::PublicationMode => {
                "Declare `publication_mode: commit_only` or \
                 `publication_mode: provisional_revisions`"
            }
        }
    }
}

/// Enforce a feature use against the document's authored schema version.
///
/// Parser tree checks and typed validation/admission call this same gate so a
/// field cannot be accepted merely by entering through a different API.
pub fn gate_feature_use(
    version: SchemaVersion,
    feature: SchemaFeature,
    path: &str,
) -> Result<(), String> {
    let required = feature.minimum_version();
    if version >= required {
        return Ok(());
    }
    match feature {
        SchemaFeature::OutputProtocols => Err(format!(
            "{path} is not legal in authored schema version {version}; {} require minimum schema \
             version {required}. Remove the v1.5-only declaration to retain legacy output \
             semantics, or migrate/re-emit the package with `schema_version: \"{required}\"` and \
             explicit output families",
            feature.description()
        )),
        SchemaFeature::PublicationMode => Err(format!(
            "{path} is not legal in authored schema version {version}; \
             `pipeline.workflow.publication_mode` begins and is required in schema version \
             {required}. Remove `pipeline.workflow.publication_mode` to keep a pre-{required} \
             document, or upgrade `schema_version` to \"{required}\" and author a valid mode \
             (`commit_only` or `provisional_revisions`)"
        )),
    }
}

/// Bidirectional gate for a field that became mandatory when its feature was
/// introduced and was absent from the older contract.
pub fn gate_feature_field(
    version: SchemaVersion,
    feature: SchemaFeature,
    path: &str,
    authored: bool,
) -> Result<(), String> {
    let required = feature.minimum_version();
    match (version >= required, authored) {
        (false, true) => gate_feature_use(version, feature, path),
        (true, false) => Err(format!(
            "{path} is required in authored schema version {version}; {} begin at schema version \
             {required}. {}.",
            feature.description(),
            feature.declaration()
        )),
        _ => Ok(()),
    }
}

/// Normalize a declared `schema_version` spelling.
///
/// `None` is the initial version: a document written before anyone thought to
/// state one is a `1.0` document, and always was.
pub fn normalize(spelling: Option<&str>) -> Result<SchemaVersion, String> {
    let Some(raw) = spelling else {
        return Ok(INITIAL_SCHEMA_VERSION);
    };
    let trimmed = raw.trim();
    let digits = trimmed.strip_prefix('v').unwrap_or(trimmed);
    let unreadable = || {
        format!(
            "schema_version '{raw}' is not a version this reader can compare. Write it as \
             'v<major>.<minor>' — '{SUPPORTED_SCHEMA_VERSION}' is the newest this build reads — \
             or leave it out to mean '{INITIAL_SCHEMA_VERSION}'"
        )
    };
    if digits.is_empty() {
        return Err(unreadable());
    }
    let (major, minor) = match digits.split_once('.') {
        Some((major, minor)) => (major, minor),
        None => (digits, "0"),
    };
    let major: u32 = major.parse().map_err(|_| unreadable())?;
    let minor: u32 = minor.parse().map_err(|_| unreadable())?;
    Ok(SchemaVersion::new(major, minor))
}

/// Refuse a document this build is too old to read, before `serde` sees it.
pub fn gate(spelling: Option<&str>) -> Result<SchemaVersion, String> {
    let version = normalize(spelling)?;
    if version.major != SUPPORTED_SCHEMA_VERSION.major {
        return Err(format!(
            "this package declares inference-metadata schema version {version}, and this build \
             reads major version {}. A major version is a different contract, not a longer one, \
             so there is nothing here to read partially: use a runtime that declares \
             {}.x support",
            SUPPORTED_SCHEMA_VERSION.major, version.major
        ));
    }
    if version.minor > SUPPORTED_SCHEMA_VERSION.minor {
        return Err(format!(
            "this package declares inference-metadata schema version {version}, and this build \
             reads up to {SUPPORTED_SCHEMA_VERSION}. The document is not malformed — it uses \
             fields added after this runtime was built, and every structure in this schema \
             refuses fields it does not know rather than ignoring them. Upgrade the runtime, or \
             re-emit the package at {SUPPORTED_SCHEMA_VERSION} without the newer fields"
        ));
    }
    Ok(version)
}

/// The declared `schema_version` of an untyped document.
///
/// `Ok(None)` means the key is absent, which is the initial version. A key that
/// is present but is not a string — `1.1` written unquoted is a YAML float, and
/// `1` an integer — cannot be compared, and is refused here rather than left to
/// `serde`: the point of reading the version first is that the answer names the
/// version, and "invalid type: floating point `1.1`" does not.
pub fn declared_in(document: &serde_yaml::Value) -> Result<Option<&str>, String> {
    let Some(value) = document.get("schema_version") else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    match value.as_str() {
        Some(spelling) => Ok(Some(spelling)),
        None => Err(format!(
            "schema_version must be a quoted string such as '{SUPPORTED_SCHEMA_VERSION}'. This \
             document writes it unquoted, so it reads as a number rather than a version, and \
             '1.10' and '1.1' would be the same document"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_spelling_of_the_first_version_means_the_first_version() {
        for spelling in [None, Some("v1"), Some("1"), Some("1.0"), Some("v1.0")] {
            assert_eq!(
                normalize(spelling).expect("a known spelling normalizes"),
                INITIAL_SCHEMA_VERSION,
                "{spelling:?}"
            );
        }
    }

    #[test]
    fn a_canonical_version_prints_the_way_a_document_should_write_it() {
        assert_eq!(SUPPORTED_SCHEMA_VERSION.to_string(), "v1.7");
        assert_eq!(INITIAL_SCHEMA_VERSION.to_string(), "v1.0");
    }

    #[test]
    fn surrounding_space_is_not_a_different_version() {
        assert_eq!(
            normalize(Some(" 1.3 ")).expect("space normalizes away"),
            TOOL_PROTOCOL_SCHEMA_VERSION
        );
    }

    #[test]
    fn a_spelling_no_one_can_compare_says_how_to_write_one() {
        let error = normalize(Some("latest")).expect_err("'latest' is not a version");
        assert!(
            error.contains("'v<major>.<minor>'")
                && error.contains(&SUPPORTED_SCHEMA_VERSION.to_string()),
            "{error}"
        );
        assert!(normalize(Some("v1.2.3")).is_err());
        assert!(normalize(Some("1.x")).is_err());
        assert!(normalize(Some("v")).is_err());
        assert!(normalize(Some("")).is_err());
    }

    #[test]
    fn a_version_that_is_not_a_string_cannot_be_a_version() {
        let document: serde_yaml::Value =
            serde_yaml::from_str("schema_version: 1.1\n").expect("a tree");
        let error = declared_in(&document).expect_err("an unquoted version is a number");
        assert!(error.contains("quoted string"), "{error}");

        let absent: serde_yaml::Value = serde_yaml::from_str("model: {}\n").expect("a tree");
        assert_eq!(declared_in(&absent).expect("absence is readable"), None);
    }

    #[test]
    fn a_newer_minor_is_refused_by_number_rather_than_by_field_name() {
        let error = gate(Some("1.8")).expect_err("1.8 is newer than this build");
        assert!(
            error.contains("declares inference-metadata schema version v1.8"),
            "{error}"
        );
        assert!(
            error.contains(&format!("reads up to {SUPPORTED_SCHEMA_VERSION}")),
            "{error}"
        );
        assert!(error.contains("refuses fields it does not know"), "{error}");
    }

    #[test]
    fn a_different_major_is_a_different_contract() {
        let error = gate(Some("2.0")).expect_err("2.0 is a different contract");
        assert!(error.contains("major version"), "{error}");
    }

    #[test]
    fn an_older_or_equal_minor_passes_the_gate() {
        assert_eq!(gate(None).expect("absent"), INITIAL_SCHEMA_VERSION);
        assert_eq!(gate(Some("v1")).expect("v1"), INITIAL_SCHEMA_VERSION);
        assert_eq!(gate(Some("1.1")).expect("1.1"), BATCHING_SCHEMA_VERSION);
        assert_eq!(
            gate(Some("1.3")).expect("1.3"),
            TOOL_PROTOCOL_SCHEMA_VERSION
        );
        assert_eq!(
            gate(Some("1.4")).expect("1.4"),
            TOKEN_CONTEXT_SCHEMA_VERSION
        );
        assert_eq!(
            gate(Some("1.5")).expect("1.5"),
            OUTPUT_PROTOCOL_SCHEMA_VERSION
        );
        assert_eq!(
            gate(Some("1.6")).expect("1.6"),
            CANONICAL_SPECULATION_SCHEMA_VERSION
        );
        assert_eq!(gate(Some("1.7")).expect("1.7"), SUPPORTED_SCHEMA_VERSION);
    }

    #[test]
    fn output_protocol_family_gate_is_bidirectional() {
        let path = "pipeline.workflow.outputs.answer.family";
        gate_feature_field(
            SchemaVersion::new(1, 4),
            SchemaFeature::OutputProtocols,
            path,
            false,
        )
        .expect("legacy output omits the later field");
        let below = gate_feature_field(
            SchemaVersion::new(1, 4),
            SchemaFeature::OutputProtocols,
            path,
            true,
        )
        .expect_err("older schema cannot opt into a later field");
        assert_eq!(
            below,
            "pipeline.workflow.outputs.answer.family is not legal in authored schema version \
             v1.4; workflow output families, logical streams, and typed revision operations \
             require minimum schema version v1.5. Remove the v1.5-only declaration to retain \
             legacy output semantics, or migrate/re-emit the package with `schema_version: \
             \"v1.5\"` and explicit output families"
        );

        let missing = gate_feature_field(
            SchemaVersion::new(1, 5),
            SchemaFeature::OutputProtocols,
            path,
            false,
        )
        .expect_err("the introducing version requires its field");
        assert!(
            missing.contains(path) && missing.contains("required"),
            "{missing}"
        );
        gate_feature_field(
            SchemaVersion::new(1, 5),
            SchemaFeature::OutputProtocols,
            path,
            true,
        )
        .expect("v1.5 accepts its authored field");
    }

    #[test]
    fn publication_mode_gate_is_bidirectional() {
        let path = "pipeline.workflow.publication_mode";
        gate_feature_field(
            SchemaVersion::new(1, 6),
            SchemaFeature::PublicationMode,
            path,
            false,
        )
        .expect("a legacy workflow omits the later field");
        let below = gate_feature_field(
            SchemaVersion::new(1, 6),
            SchemaFeature::PublicationMode,
            path,
            true,
        )
        .expect_err("v1.6 cannot opt into the v1.7 field");
        assert_eq!(
            below,
            "pipeline.workflow.publication_mode is not legal in authored schema version v1.6; \
             `pipeline.workflow.publication_mode` begins and is required in schema version v1.7. \
             Remove `pipeline.workflow.publication_mode` to keep a pre-v1.7 document, or upgrade \
             `schema_version` to \"v1.7\" and author a valid mode (`commit_only` or \
             `provisional_revisions`)"
        );
        let missing = gate_feature_field(
            SchemaVersion::new(1, 7),
            SchemaFeature::PublicationMode,
            path,
            false,
        )
        .expect_err("v1.7 must state its publication mode");
        assert!(
            missing.contains(path) && missing.contains("publication_mode"),
            "{missing}"
        );
    }
}
