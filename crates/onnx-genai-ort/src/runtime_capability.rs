//! Runtime capability probes.
//!
//! Some graph lowerings only exist to work around a limitation of the ONNX
//! Runtime that will execute them. A workaround that outlives its limitation is
//! not free — it emits extra nodes, extra launches and extra intermediate
//! buffers — so the planner needs to ask the runtime what it can do rather than
//! assume the worst forever.
//!
//! Each probe here answers one such question about the *loaded* runtime, so a
//! build of this crate keeps working against older and newer ONNX Runtimes
//! without recompilation.

/// The first ONNX Runtime release whose CUDA `ArgMax`/`ArgMin` parallelise a
/// reduction over a wide last axis.
///
/// `None` means no *released* runtime is known to do so, which is the state
/// while microsoft/onnxruntime#32092 exists only as an unmerged pull request. Nothing in this
/// repository may depend on that pull request, so the default answer stays
/// "assume the limitation" and callers keep their workaround.
///
/// When a release ships the fix, set this to that version and every planner
/// that consults [`reduces_wide_last_axis_on_cuda`] stops emitting the
/// workaround against it, with no other change.
///
/// The limitation: ORT's `arg_min_max_last_axis_kernel` assigns one lane per
/// reduced *row*, so a reduction over one very wide row runs serially. Measured
/// on an H200 with ONNX Runtime 1.28, a `[1, 202048]` float32 `ArgMax` costs
/// 4.44 ms, against roughly 4 us for a parallel reduction of the same data.
const FIRST_VERSION_WITH_WIDE_ARG_REDUCTION: Option<(u32, u32)> = None;

/// The loaded ONNX Runtime's version string, if one is loaded.
#[must_use]
pub fn loaded_version() -> Option<String> {
    onnx_genai_ort_sys::loaded_ort_version()
}

/// Whether the loaded runtime reduces a wide last axis efficiently on CUDA.
///
/// Callers that lower a degenerate `ArgMax`/`ArgMin` into something wider
/// should emit the node unchanged when this is true.
#[must_use]
pub fn reduces_wide_last_axis_on_cuda() -> bool {
    let Some(required) = FIRST_VERSION_WITH_WIDE_ARG_REDUCTION else {
        return false;
    };
    onnx_genai_ort_sys::loaded_ort_version()
        .as_deref()
        .and_then(parse_version)
        .is_some_and(|loaded| loaded >= required)
}

/// Parse the leading `major.minor` of an ONNX Runtime version string.
///
/// Release strings look like `1.28.0`; development builds can carry a suffix
/// such as `1.30.0-dev+abcdef`. Anything this cannot parse is treated as
/// unknown by the caller, which is the conservative answer.
fn parse_version(version: &str) -> Option<(u32, u32)> {
    let mut parts = version
        .trim_start_matches('v')
        .split(|c: char| !c.is_ascii_digit());
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_versions_parse() {
        assert_eq!(parse_version("1.28.0"), Some((1, 28)));
        assert_eq!(parse_version("1.30.0"), Some((1, 30)));
        assert_eq!(parse_version("v1.9.2"), Some((1, 9)));
    }

    #[test]
    fn development_versions_parse_to_their_release() {
        assert_eq!(parse_version("1.30.0-dev+0d85ddf"), Some((1, 30)));
        assert_eq!(parse_version("1.29.0-rc1"), Some((1, 29)));
    }

    #[test]
    fn unparseable_versions_are_unknown() {
        assert_eq!(parse_version(""), None);
        assert_eq!(parse_version("unknown"), None);
        // A bare major with no minor cannot be ordered against a requirement.
        assert_eq!(parse_version("2"), None);
    }

    #[test]
    fn ordering_is_by_major_then_minor() {
        // Guards the comparison the probe performs, independently of which
        // release is currently required.
        let required = (1u32, 31u32);
        assert!(parse_version("1.31.0").unwrap() >= required);
        assert!(parse_version("1.32.0").unwrap() >= required);
        assert!(parse_version("2.0.0").unwrap() >= required);
        assert!(parse_version("1.30.0").unwrap() < required);
        assert!(parse_version("1.9.0").unwrap() < required);
    }

    #[test]
    fn no_release_is_assumed_to_have_the_fix_yet() {
        // This is the guard that keeps the repository independent of an
        // unmerged ONNX Runtime pull request. When a release ships the fix,
        // update `FIRST_VERSION_WITH_WIDE_ARG_REDUCTION` and this test.
        assert_eq!(
            FIRST_VERSION_WITH_WIDE_ARG_REDUCTION, None,
            "a released runtime now has the fix: point the planner at it"
        );
    }
}
